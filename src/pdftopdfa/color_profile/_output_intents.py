# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Local OutputIntent validation and repair helpers."""

import logging
import re
from dataclasses import dataclass

from pikepdf import Array, Dictionary, Name, Pdf, String

from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _validate_icc_profile
from ._types import ColorSpaceType

logger = logging.getLogger(__name__)


_ICC_SIGNATURE_TO_COLORSPACE: dict[bytes, tuple[int, ColorSpaceType]] = {
    b"GRAY": (1, ColorSpaceType.DEVICE_GRAY),
    b"RGB ": (3, ColorSpaceType.DEVICE_RGB),
    b"CMYK": (4, ColorSpaceType.DEVICE_CMYK),
}

# ISO 32000-1, Table 365. Other keys would need an ISO 32000 extension declaration,
# and the pipeline does not keep those for third-party extensions.
_DEFINED_OUTPUT_INTENT_SUBTYPES = (Name.GTS_PDFA1, Name.GTS_PDFX, Name.ISO_PDFE1)

# Well-known families in the ICC characterization data registry.
_KNOWN_CONDITION_RE = re.compile(
    r"(?:FOGRA|IFRA)\s*\d+|CGATS\s*TR\s*\d+|JC[A-Z]*\s*\d+",
    re.ASCII | re.IGNORECASE,
)

# Element types that ICC.1 permits for the tags required below. This checks the type
# of each element but does not parse its content.
_ICC_XYZ_TYPE = (b"XYZ ",)
_ICC_CURVE_TYPES = (b"curv", b"para")
_ICC_TAG_TYPES: dict[bytes, tuple[bytes, ...]] = {
    b"desc": (b"desc", b"mluc"),
    b"cprt": (b"text", b"mluc"),
    b"wtpt": _ICC_XYZ_TYPE,
    b"A2B0": (b"mft1", b"mft2", b"mAB "),
    b"kTRC": _ICC_CURVE_TYPES,
    b"rTRC": _ICC_CURVE_TYPES,
    b"gTRC": _ICC_CURVE_TYPES,
    b"bTRC": _ICC_CURVE_TYPES,
    b"rXYZ": _ICC_XYZ_TYPE,
    b"gXYZ": _ICC_XYZ_TYPE,
    b"bXYZ": _ICC_XYZ_TYPE,
}

# Tags that describe a color model without a lookup table, which CMYK requires.
_ICC_MODEL_TAGS: dict[bytes, frozenset[bytes]] = {
    b"GRAY": frozenset({b"kTRC"}),
    b"RGB ": frozenset({b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC"}),
}


@dataclass(frozen=True)
class OutputIntentPreparation:
    """Preservation decision, with a color space if an output profile exists."""

    keep_existing: bool
    document_colorspace: ColorSpaceType | None = None
    reason: str = ""


@dataclass(frozen=True)
class _ValidOutputIntent:
    """Locally validated embedded OutputIntent profile details."""

    intent: Dictionary
    profile_data: bytes
    document_colorspace: ColorSpaceType


def prepare_existing_output_intents(pdf: Pdf) -> OutputIntentPreparation:
    """Validate and repair existing OutputIntents using fail-closed rules.

    Keep compatible intent dictionaries and their individual printing metadata.
    All embedded profiles must describe the same output condition. Byte-identical
    profile streams are unified to the same indirect object (rule 6.2.3-2), and
    only the first PDF/A intent is kept.
    A PDF/A intent can be added using an existing compatible output profile.
    """
    output_intents = pdf.Root.get("/OutputIntents")
    if not isinstance(output_intents, Array) or len(output_intents) == 0:
        return OutputIntentPreparation(
            keep_existing=False,
            reason="OutputIntents missing, empty, or not an array",
        )

    pdfa_intents: list[_ValidOutputIntent] = []
    profiled_intents: list[_ValidOutputIntent] = []
    invalid_additional: list[int] = []
    for index, intent_ref in enumerate(output_intents):
        intent = _resolve_indirect(intent_ref)
        if not isinstance(intent, Dictionary):
            return OutputIntentPreparation(
                keep_existing=False,
                reason="OutputIntent entry is not a dictionary",
            )

        subtype = intent.get("/S")
        if not isinstance(subtype, Name):
            return OutputIntentPreparation(
                keep_existing=False,
                reason="OutputIntent subtype is missing or is not a name",
            )
        if subtype not in _DEFINED_OUTPUT_INTENT_SUBTYPES:
            invalid_additional.append(index)
            continue
        if (
            intent.get("/Type") not in (None, Name.OutputIntent)
            or not isinstance(intent.get("/OutputConditionIdentifier"), String)
            or any(
                intent.get(key) is not None and not isinstance(intent[key], String)
                for key in ("/OutputCondition", "/RegistryName", "/Info")
            )
        ):
            return OutputIntentPreparation(
                keep_existing=False,
                reason="OutputIntent dictionary has missing or malformed fields",
            )
        if subtype == Name("/GTS_PDFX"):
            if "/DestOutputProfileRef" in intent:
                del intent["/DestOutputProfileRef"]
                logger.debug("Removed DestOutputProfileRef from PDF/X OutputIntent")
        # DestOutputProfile is optional for non-PDF/A intents only if they name a
        # standard production condition (ISO 32000-1, Table 365). Rule 6.2.3-2
        # constrains the entries that actually contain an embedded profile.
        if subtype != Name.GTS_PDFA1 and "/DestOutputProfile" not in intent:
            if not _names_standard_condition(intent):
                invalid_additional.append(index)
            continue

        valid_intent = _validate_pdfa_output_intent(intent)
        if valid_intent is None:
            return OutputIntentPreparation(
                keep_existing=False,
                reason="OutputIntent profile failed local validation",
            )
        profiled_intents.append(valid_intent)
        if subtype == Name.GTS_PDFA1:
            pdfa_intents.append(valid_intent)

    # These entries cannot affect the output conditions that remain.
    for index in reversed(invalid_additional):
        logger.warning(
            "Removed OutputIntent %s with an undefined subtype or output condition",
            _resolve_indirect(output_intents[index]).get("/S"),
        )
        del output_intents[index]

    if not profiled_intents:
        return OutputIntentPreparation(
            keep_existing=True,
            reason="Registered output conditions can accompany a new PDF/A profile",
        )

    profile_datas = {intent.profile_data for intent in profiled_intents}
    color_spaces = {intent.document_colorspace for intent in profiled_intents}
    if len(profile_datas) > 1 or len(color_spaces) > 1:
        logger.warning(
            "Multiple OutputIntents reference different ICC profiles. "
            "Replacing existing OutputIntents."
        )
        return OutputIntentPreparation(
            keep_existing=False,
            reason="Multiple OutputIntents reference different ICC profiles",
        )

    first_intent = (pdfa_intents or profiled_intents)[0]
    shared_profile = first_intent.intent["/DestOutputProfile"]
    for valid_intent in profiled_intents:
        valid_intent.intent["/DestOutputProfile"] = shared_profile

    if len(pdfa_intents) > 1:
        _remove_redundant_pdfa_intents(output_intents)

    if not pdfa_intents:
        # Preserve a PDF/X (or other) output condition when converting to PDF/A;
        # adding its PDF/A counterpart does not require discarding that intent.
        pdfa_intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=first_intent.intent.get(
                "/OutputConditionIdentifier", "Custom"
            ),
            DestOutputProfile=shared_profile,
        )
        for key in ("/OutputCondition", "/RegistryName", "/Info"):
            if key in first_intent.intent:
                pdfa_intent[key] = first_intent.intent[key]
        output_intents.append(pdf.make_indirect(pdfa_intent))
        logger.debug("Added PDF/A OutputIntent using the existing output profile")

    return OutputIntentPreparation(
        keep_existing=True,
        document_colorspace=first_intent.document_colorspace,
        reason="Existing output conditions are compatible with PDF/A",
    )


def _names_standard_condition(intent: Dictionary) -> bool:
    """Return whether a profile-less intent names a standard production condition.

    No registry can be queried here, so this is a heuristic. It accepts a condition
    that claims a registry, or that belongs to a well-known family of characterizations
    if /RegistryName is omitted. Custom never names a standard condition.
    """
    identifier = str(intent.get("/OutputConditionIdentifier")).strip()
    if identifier.casefold() in ("", "custom"):
        return False
    registry = intent.get("/RegistryName")
    if isinstance(registry, String) and str(registry).strip():
        return True
    return _KNOWN_CONDITION_RE.fullmatch(identifier) is not None


def _remove_redundant_pdfa_intents(output_intents: Array) -> None:
    """Keep the first PDF/A intent when several describe the same output profile.

    The profile is byte-identical, so only descriptive text can be lost. A single
    PDF/A intent is unambiguous, and other subtypes such as PDF/X remain in place.
    """
    pdfa_positions = [
        index
        for index, entry in enumerate(output_intents)
        if _resolve_indirect(entry).get("/S") == Name.GTS_PDFA1
    ]
    for index in reversed(pdfa_positions[1:]):
        del output_intents[index]
    logger.info(
        "Removed %d redundant PDF/A OutputIntent(s) with an identical profile",
        len(pdfa_positions) - 1,
    )


def _has_usable_tag_table(profile_data: bytes) -> bool:
    """Require tagged data that can describe colors, beyond a plausible header.

    Besides the common description, copyright and white point tags, an output
    profile needs a lookup table, or the tone curves and colorants of its model.
    """
    if len(profile_data) < 132:
        return False
    count = int.from_bytes(profile_data[128:132], "big")
    data_start = 132 + 12 * count
    if data_start > len(profile_data):
        return False
    tags = set()
    for position in range(132, data_start, 12):
        signature = profile_data[position : position + 4]
        offset = int.from_bytes(profile_data[position + 4 : position + 8], "big")
        size = int.from_bytes(profile_data[position + 8 : position + 12], "big")
        if offset < data_start or size < 8 or offset + size > len(profile_data):
            return False
        # Each element starts with its type, which the tag restricts for these tags.
        allowed_types = _ICC_TAG_TYPES.get(signature)
        if allowed_types and profile_data[offset : offset + 4] not in allowed_types:
            return False
        tags.add(signature)
    if not {b"desc", b"cprt", b"wtpt"} <= tags:
        return False
    model_tags = _ICC_MODEL_TAGS.get(profile_data[16:20])
    return b"A2B0" in tags or (model_tags is not None and model_tags <= tags)


def _validate_pdfa_output_intent(intent: Dictionary) -> _ValidOutputIntent | None:
    """Check an embedded output profile and repair its deterministic component count."""
    dest_profile = intent.get("/DestOutputProfile")
    if dest_profile is None:
        return None

    dest_profile = _resolve_indirect(dest_profile)
    try:
        profile_data = bytes(dest_profile.read_bytes())
    except Exception:
        return None

    if len(profile_data) < 128:
        return None
    if profile_data[36:40] != b"acsp":
        return None
    if not _validate_icc_profile(profile_data):
        return None
    if profile_data[12:16] not in {b"mntr", b"prtr"}:
        return None
    if not _has_usable_tag_table(profile_data):
        return None

    signature = profile_data[16:20]
    profile_info = _ICC_SIGNATURE_TO_COLORSPACE.get(signature)
    if profile_info is None:
        return None

    expected_n, document_colorspace = profile_info
    # ISO 32000-1, Table 66 requires an integer, so 3.0 also needs rewriting.
    components = dest_profile.get("/N")
    if (
        isinstance(components, bool)
        or not isinstance(components, int)
        or components != expected_n
    ):
        dest_profile[Name.N] = expected_n
        logger.debug("Repaired DestOutputProfile /N to %d", expected_n)

    return _ValidOutputIntent(
        intent=intent,
        profile_data=profile_data,
        document_colorspace=document_colorspace,
    )
