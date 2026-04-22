# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Local OutputIntent validation and repair helpers."""

import logging
from dataclasses import dataclass

from pikepdf import Array, Dictionary, Name, Pdf

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
    """Decision from local OutputIntent preparation."""

    keep_existing: bool
    document_colorspace: ColorSpaceType | None = None
    reason: str = ""


@dataclass(frozen=True)
class _ValidOutputIntent:
    """Locally validated PDF/A OutputIntent details."""

    intent: Dictionary
    profile_data: bytes
    document_colorspace: ColorSpaceType


def prepare_existing_output_intents(pdf: Pdf) -> OutputIntentPreparation:
    """Validate and repair existing OutputIntents using fail-closed rules.

    Existing PDF/A OutputIntents are kept only when every relevant local check
    succeeds.  Repair is limited to deterministic, local fixes: removing
    ``/DestOutputProfileRef`` from PDF/X intents and correcting missing or
    wrong ICC stream ``/N`` values.
    """
    output_intents = pdf.Root.get("/OutputIntents")
    if not isinstance(output_intents, Array) or len(output_intents) == 0:
        return OutputIntentPreparation(
            keep_existing=False,
            reason="OutputIntents missing, empty, or not an array",
        )

    pdfa_intents: list[_ValidOutputIntent] = []
    for intent_ref in output_intents:
        intent = _resolve_indirect(intent_ref)
        if not isinstance(intent, Dictionary):
            return OutputIntentPreparation(
                keep_existing=False,
                reason="OutputIntent entry is not a dictionary",
            )

        subtype = intent.get("/S")
        if subtype == Name("/GTS_PDFX"):
            if "/DestOutputProfileRef" in intent:
                del intent["/DestOutputProfileRef"]
                logger.debug("Removed DestOutputProfileRef from PDF/X OutputIntent")
            continue

        if subtype != Name.GTS_PDFA1:
            return OutputIntentPreparation(
                keep_existing=False,
                reason=f"Unhandled OutputIntent subtype: {subtype}",
            )

        valid_intent = _validate_pdfa_output_intent(intent)
        if valid_intent is None:
            return OutputIntentPreparation(
                keep_existing=False,
                reason="PDF/A OutputIntent failed local validation",
            )
        pdfa_intents.append(valid_intent)

    if not pdfa_intents:
        return OutputIntentPreparation(
            keep_existing=False,
            reason="No valid PDF/A OutputIntent found",
        )

    profile_datas = {intent.profile_data for intent in pdfa_intents}
    color_spaces = {intent.document_colorspace for intent in pdfa_intents}
    if len(profile_datas) > 1 or len(color_spaces) > 1:
        logger.warning(
            "Multiple PDF/A OutputIntents reference different ICC profiles. "
            "Replacing existing OutputIntents."
        )
        return OutputIntentPreparation(
            keep_existing=False,
            reason="Multiple PDF/A OutputIntents reference different ICC profiles",
        )

    first_intent = pdfa_intents[0]
    if len(output_intents) != 1:
        pdf.Root.OutputIntents = Array([first_intent.intent])
        logger.debug("Reduced OutputIntents to one valid PDF/A OutputIntent")

    return OutputIntentPreparation(
        keep_existing=True,
        document_colorspace=first_intent.document_colorspace,
        reason="Existing PDF/A OutputIntent is locally valid",
    )


def _validate_pdfa_output_intent(intent: Dictionary) -> _ValidOutputIntent | None:
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
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
