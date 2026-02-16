# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ICC profile loading, validation, and creation."""

import functools
import logging
from importlib.resources import files

from pikepdf import Array, Name, Pdf, Stream

from ..exceptions import ConversionError
from ._types import _N_COMPONENTS, ColorSpaceType

logger = logging.getLogger(__name__)


def _validate_icc_profile(profile_data: bytes) -> bool:
    """
    Validate ICC profile structure.

    Args:
        profile_data: Raw ICC profile bytes.

    Returns:
        True if valid, False otherwise.
    """
    # ICC profile must have at least 128-byte header
    if len(profile_data) < 128:
        return False

    # Check for 'acsp' signature at bytes 36-39
    signature = profile_data[36:40]
    if signature != b"acsp":
        return False

    # Check declared size matches actual size (bytes 0-3, big-endian)
    declared_size = int.from_bytes(profile_data[0:4], byteorder="big")
    if declared_size != len(profile_data):
        return False

    # Check ICC profile version (bytes 8-11): only v2.x or v4.x allowed
    major_version = profile_data[8]
    if major_version not in (2, 4):
        return False

    # Check ICC profile device class (bytes 12-15):
    # Only mntr, prtr, scnr, spac allowed; nmcl (Named Color) not allowed
    device_class = profile_data[12:16]
    allowed_classes = {b"mntr", b"prtr", b"scnr", b"spac"}
    if device_class not in allowed_classes:
        return False

    return True


@functools.cache
def get_srgb_profile() -> bytes:
    """
    Load sRGB ICC profile from package resources.

    The result is cached so the file is read only once.

    Returns:
        Raw ICC profile bytes.

    Raises:
        ConversionError: If profile cannot be loaded or is invalid.
    """
    try:
        resource_files = files("pdftopdfa") / "resources" / "icc"
        profile_path = resource_files.joinpath("sRGB2014.icc")
        profile_data = profile_path.read_bytes()
    except Exception as e:
        raise ConversionError(f"Could not load ICC profile: {e}") from e

    if not _validate_icc_profile(profile_data):
        raise ConversionError("ICC profile is invalid or corrupted")

    logger.debug("sRGB ICC profile loaded: %d bytes", len(profile_data))
    return profile_data


@functools.cache
def get_gray_profile() -> bytes:
    """
    Load Gray ICC profile from package resources.

    The result is cached so the file is read only once.

    Returns:
        Raw ICC profile bytes.

    Raises:
        ConversionError: If profile cannot be loaded or is invalid.
    """
    try:
        resource_files = files("pdftopdfa") / "resources" / "icc"
        profile_path = resource_files.joinpath("sGray.icc")
        profile_data = profile_path.read_bytes()
    except Exception as e:
        raise ConversionError(f"Could not load Gray ICC profile: {e}") from e

    if not _validate_icc_profile(profile_data):
        raise ConversionError("Gray ICC profile is invalid or corrupted")

    logger.debug("Gray ICC profile loaded: %d bytes", len(profile_data))
    return profile_data


@functools.cache
def get_cmyk_profile() -> bytes:
    """
    Load CMYK ICC profile from package resources.

    The result is cached so the file is read only once.

    Returns:
        Raw ICC profile bytes.

    Raises:
        ConversionError: If profile cannot be loaded or is invalid.
    """
    try:
        resource_files = files("pdftopdfa") / "resources" / "icc"
        profile_path = resource_files.joinpath("ISOcoated_v2_300_bas.icc")
        profile_data = profile_path.read_bytes()
    except Exception as e:
        raise ConversionError(f"Could not load CMYK ICC profile: {e}") from e

    if not _validate_icc_profile(profile_data):
        raise ConversionError("CMYK ICC profile is invalid or corrupted")

    logger.debug("CMYK ICC profile loaded: %d bytes", len(profile_data))
    return profile_data


def get_profile_for_colorspace(colorspace: ColorSpaceType) -> bytes:
    """
    Get the appropriate ICC profile for a color space type.

    Args:
        colorspace: The color space type.

    Returns:
        Raw ICC profile bytes.

    Raises:
        ConversionError: If profile cannot be loaded.
    """
    if colorspace in (ColorSpaceType.DEVICE_GRAY, ColorSpaceType.CAL_GRAY):
        return get_gray_profile()
    elif colorspace in (ColorSpaceType.DEVICE_RGB, ColorSpaceType.CAL_RGB):
        return get_srgb_profile()
    elif colorspace == ColorSpaceType.DEVICE_CMYK:
        return get_cmyk_profile()
    else:
        raise ConversionError(f"Unknown color space: {colorspace}")


def _create_icc_colorspace(
    pdf: Pdf,
    colorspace: ColorSpaceType,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> Array:
    """Create an ICCBased color space array for a Device color space.

    Returns ``[/ICCBased <indirect_icc_stream>]``.  The ICC stream is
    cached in *icc_stream_cache* so that only one stream object is
    created per color space type per document.

    Args:
        pdf: pikepdf Pdf to own the stream.
        colorspace: One of DEVICE_GRAY / DEVICE_RGB / DEVICE_CMYK.
        icc_stream_cache: Mutable dict shared across the conversion run.

    Returns:
        pikepdf Array suitable for use as a color space value.
    """
    if colorspace not in icc_stream_cache:
        profile_data = get_profile_for_colorspace(colorspace)
        icc_stream = pdf.make_stream(profile_data)
        icc_stream[Name.N] = _N_COMPONENTS[colorspace]
        icc_stream_cache[colorspace] = icc_stream

    return Array([Name.ICCBased, icc_stream_cache[colorspace]])
