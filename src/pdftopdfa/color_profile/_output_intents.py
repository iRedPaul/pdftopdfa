# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Local OutputIntent validation and repair helpers."""

import logging
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
    for intent_ref in output_intents:
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
        # DestOutputProfile is optional for non-PDF/A intents (for example a
        # registered PDF/X printing condition). Rule 6.2.3-2 constrains the
        # entries that actually contain an embedded profile.
        if subtype != Name.GTS_PDFA1 and "/DestOutputProfile" not in intent:
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

    signature = profile_data[16:20]
    profile_info = _ICC_SIGNATURE_TO_COLORSPACE.get(signature)
    if profile_info is None:
        return None

    expected_n, document_colorspace = profile_info
    if _as_int(dest_profile.get("/N")) != expected_n:
        dest_profile[Name.N] = expected_n
        logger.debug("Repaired DestOutputProfile /N to %d", expected_n)

    return _ValidOutputIntent(
        intent=intent,
        profile_data=profile_data,
        document_colorspace=document_colorspace,
    )


def _as_int(value: object) -> int | None:
    """Interpret a PDF numeric value without propagating invalid-field errors."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
