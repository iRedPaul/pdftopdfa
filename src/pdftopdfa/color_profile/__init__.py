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
from ._output_intents import prepare_existing_output_intents
from ._profiles import (
    _create_icc_colorspace,
    _validate_icc_profile,
    get_cmyk_profile,
    get_gray_profile,
    get_profile_for_colorspace,
    get_srgb_profile,
)
from ._transparency import (
    _add_missing_transparency_groups,
    _fix_transparency_group_colorspaces,
)
from ._types import ColorSpaceAnalysis, ColorSpaceType, SpecialColorSpace

logger = logging.getLogger(__name__)

__all__ = [
    "ColorSpaceAnalysis",
    "ColorSpaceType",
    "SpecialColorSpace",
    "_analyze_colorspace",
    "_apply_default_colorspaces",
    "_apply_defaults_to_ap_entry",
    "_add_missing_transparency_groups",
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

    existing_output_intent_colorspace: ColorSpaceType | None = None
    if has_output_intent(pdf):
        if replace_existing:
            output_intent_preparation = prepare_existing_output_intents(pdf)
            if output_intent_preparation.keep_existing:
                existing_output_intent_colorspace = (
                    output_intent_preparation.document_colorspace
                )
                logger.info("Keeping existing PDF/A OutputIntent")
            else:
                logger.info(
                    "Replacing existing OutputIntents: %s",
                    output_intent_preparation.reason,
                )
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

    document_colorspace = existing_output_intent_colorspace or dominant
    icc_stream_cache: dict[ColorSpaceType, Stream] = {}

    # Rule 6.2.10-2: add /Group to transparent pages missing one
    groups_added = _add_missing_transparency_groups(pdf, icc_stream_cache)
    if groups_added > 0:
        logger.info(
            "Added /Group to %d page(s) with transparency (rule 6.2.10-2)",
            groups_added,
        )

    if existing_output_intent_colorspace is None:
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
    non_dominant = device_spaces - {document_colorspace}
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
        "ICC color profile ready: %s (PDF/A-%s), detected: %s",
        document_colorspace.value,
        level,
        ", ".join(cs.value for cs in sorted(detected, key=lambda x: x.value)),
    )
    return sorted(detected, key=lambda x: x.value)
