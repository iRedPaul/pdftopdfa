# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ICC color profile management for PDF/A conversion."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import ConversionError
from ..utils import validate_pdfa_level
from ._calibrated import _convert_calibrated_colorspaces
from ._defaults import _apply_default_colorspaces, _apply_defaults_to_ap_entry
from ._detection import (
    _analyze_colorspace,
    _parse_colorspace_array,
    detect_color_spaces,
)
from ._profiles import (
    _create_icc_colorspace,
    _validate_icc_profile,
    get_cmyk_profile,
    get_gray_profile,
    get_profile_for_colorspace,
    get_srgb_profile,
)
from ._transparency import _fix_transparency_group_colorspaces
from ._types import ColorSpaceAnalysis, ColorSpaceType, SpecialColorSpace

logger = logging.getLogger(__name__)

__all__ = [
    "ColorSpaceAnalysis",
    "ColorSpaceType",
    "SpecialColorSpace",
    "_analyze_colorspace",
    "_apply_default_colorspaces",
    "_apply_defaults_to_ap_entry",
    "_convert_calibrated_colorspaces",
    "_create_icc_colorspace",
    "_fix_transparency_group_colorspaces",
    "_parse_colorspace_array",
    "_validate_icc_profile",
    "create_output_intent_for_colorspace",
    "detect_color_spaces",
    "embed_color_profiles",
    "get_cmyk_profile",
    "get_gray_profile",
    "get_profile_for_colorspace",
    "get_srgb_profile",
    "has_output_intent",
]


def has_output_intent(pdf: Pdf) -> bool:
    """
    Check if PDF already has an OutputIntent.

    Args:
        pdf: pikepdf Pdf object.

    Returns:
        True if OutputIntents exists and is non-empty.
    """
    try:
        output_intents = pdf.Root.get("/OutputIntents")
        if output_intents is None:
            return False
        return len(output_intents) > 0
    except (KeyError, AttributeError):
        return False


def create_output_intent_for_colorspace(
    pdf: Pdf,
    colorspace: ColorSpaceType,
    profile_data: bytes,
    level: str = "3b",
) -> Dictionary:
    """
    Create an OutputIntent dictionary for a specific color space.

    Args:
        pdf: pikepdf Pdf object to create the stream in.
        colorspace: The color space type.
        profile_data: Raw ICC profile bytes.
        level: PDF/A conformance level ('2b', '2u', '3b', or '3u').

    Returns:
        OutputIntent Dictionary ready to be added to PDF.

    Raises:
        ConversionError: If profile data is invalid.
    """
    if not _validate_icc_profile(profile_data):
        raise ConversionError("ICC profile is invalid")

    n_components = {
        ColorSpaceType.DEVICE_GRAY: 1,
        ColorSpaceType.DEVICE_RGB: 3,
        ColorSpaceType.DEVICE_CMYK: 4,
    }

    output_condition_ids = {
        ColorSpaceType.DEVICE_GRAY: "sGray",
        ColorSpaceType.DEVICE_RGB: "sRGB",
        ColorSpaceType.DEVICE_CMYK: "FOGRA39",
    }

    info_strings = {
        ColorSpaceType.DEVICE_GRAY: "sGray",
        ColorSpaceType.DEVICE_RGB: "sRGB IEC61966-2.1",
        ColorSpaceType.DEVICE_CMYK: "ISO Coated v2 300% (basICColor)",
    }

    icc_stream = Stream(pdf, profile_data)
    icc_stream.N = n_components[colorspace]

    output_intent = Dictionary(
        Type=Name.OutputIntent,
        S=Name.GTS_PDFA1,
        OutputConditionIdentifier=output_condition_ids[colorspace],
        RegistryName="http://www.color.org",
        Info=info_strings[colorspace],
        DestOutputProfile=icc_stream,
    )

    logger.debug(
        "OutputIntent created for %s (PDF/A-%s)",
        colorspace.value,
        level,
    )
    return output_intent


def embed_color_profiles(
    pdf: Pdf,
    level: str = "3b",
    *,
    replace_existing: bool = True,
    convert_calibrated: bool = True,
) -> list[ColorSpaceType]:
    """
    Detect color spaces and embed appropriate ICC profiles.

    This function analyzes the PDF for used color spaces and embeds
    the corresponding ICC profiles as OutputIntents.

    Args:
        pdf: pikepdf Pdf object to modify.
        level: PDF/A conformance level ('2b', '2u', '3b', or '3u').
        replace_existing: If True, replace existing OutputIntents.
            If False and OutputIntents exist, do nothing.
        convert_calibrated: If True, convert CalGray/CalRGB color spaces
            to ICCBased equivalents.

    Returns:
        List of color space types that were embedded.

    Raises:
        ConversionError: If level is invalid or profiles cannot be embedded.
    """
    level = validate_pdfa_level(level)

    if has_output_intent(pdf):
        if replace_existing:
            # ISO 19005-2 §6.2.3: multiple output intents must reference
            # the same ICC profile.  Compare full profile bytes and keep
            # only the first when they differ.
            from ..utils import resolve_indirect as _ri

            try:
                existing = pdf.Root.get("/OutputIntents")
                if existing is not None and len(existing) > 1:
                    profiles: list[bytes | None] = []
                    for oi in existing:
                        oi = _ri(oi)
                        dest = oi.get("/DestOutputProfile")
                        if dest is not None:
                            dest = _ri(dest)
                            try:
                                profiles.append(bytes(dest.read_bytes()))
                            except Exception:
                                profiles.append(None)
                        else:
                            profiles.append(None)

                    # Deduplicate: keep only the first non-None profile
                    first = next((p for p in profiles if p is not None), None)
                    unique = {p for p in profiles if p is not None}
                    if len(unique) > 1:
                        logger.warning(
                            "Multiple OutputIntents reference different"
                            " ICC profiles (%d unique). Keeping only the"
                            " first and discarding the rest per ISO"
                            " 19005-2 §6.2.3.",
                            len(unique),
                        )
                        # Replace array with only the first entry
                        first_oi = _ri(existing[0])
                        pdf.Root.OutputIntents = Array([first_oi])
                    elif first is not None and any(p is None for p in profiles):
                        # Some intents lack a profile entirely
                        logger.warning(
                            "Some OutputIntents lack a"
                            " DestOutputProfile. Keeping only the first"
                            " OutputIntent.",
                        )
                        first_oi = _ri(existing[0])
                        pdf.Root.OutputIntents = Array([first_oi])
            except Exception:
                pass

            # Rule 6.2.3-3: DestOutputProfileRef shall not be present
            # in any PDF/X OutputIntent.
            remaining = pdf.Root.get("/OutputIntents")
            if remaining is not None:
                for oi in remaining:
                    oi = _ri(oi)
                    if oi.get("/S") == Name("/GTS_PDFX"):
                        if "/DestOutputProfileRef" in oi:
                            del oi["/DestOutputProfileRef"]

            logger.info("Replacing existing OutputIntents")
        else:
            logger.debug("OutputIntents already present, skipping")
            return []

    # Detect color spaces
    analysis = detect_color_spaces(pdf)
    detected = analysis.detected_spaces

    # Default to sRGB if no color spaces detected
    if not detected:
        detected = {ColorSpaceType.DEVICE_RGB}
        logger.debug("No color spaces detected, using default sRGB")

    # PDF/A allows only a single OutputIntent with S=GTS_PDFA1
    # Select dominant color space by priority: CMYK > RGB > Gray
    if ColorSpaceType.DEVICE_CMYK in detected:
        dominant = ColorSpaceType.DEVICE_CMYK
    elif ColorSpaceType.DEVICE_RGB in detected:
        dominant = ColorSpaceType.DEVICE_RGB
    else:
        dominant = ColorSpaceType.DEVICE_GRAY

    icc_stream_cache: dict[ColorSpaceType, Stream] = {}

    profile_data = get_profile_for_colorspace(dominant)
    output_intent = create_output_intent_for_colorspace(
        pdf, dominant, profile_data, level
    )
    pdf.Root.OutputIntents = Array([pdf.make_indirect(output_intent)])

    # Cover non-dominant Device color spaces with Default entries + image fixes.
    # Note: Separation/DeviceN spaces are not converted - they are PDF/A-2/3
    # conformant when an OutputIntent is present (ISO 19005-2, 6.2.4.4).
    # Their alternate spaces may contribute to detected device spaces, which
    # is correct: the alternate space could also be used directly elsewhere.
    device_spaces = detected & {
        ColorSpaceType.DEVICE_GRAY,
        ColorSpaceType.DEVICE_RGB,
        ColorSpaceType.DEVICE_CMYK,
    }
    non_dominant = device_spaces - {dominant}
    if non_dominant:
        _apply_default_colorspaces(pdf, non_dominant, icc_stream_cache)

    # Fix transparency group /CS entries (ISO 19005-2, 6.4)
    tg_fixed = _fix_transparency_group_colorspaces(pdf, icc_stream_cache)
    if tg_fixed > 0:
        logger.info("Transparency group /CS fixed: %d", tg_fixed)

    # Optionally convert CalGray/CalRGB -> ICCBased
    if convert_calibrated:
        _convert_calibrated_colorspaces(pdf, icc_stream_cache)

    logger.info(
        "ICC color profile embedded: %s (PDF/A-%s), detected: %s",
        dominant.value,
        level,
        ", ".join(cs.value for cs in sorted(detected, key=lambda x: x.value)),
    )
    return sorted(detected, key=lambda x: x.value)
