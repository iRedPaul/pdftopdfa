# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Types, constants, and enums for color profile management."""

from dataclasses import dataclass, field
from enum import Enum

from pikepdf import Name

# Color operators that directly indicate a color space
_GRAY_OPERATORS = frozenset(["g", "G"])
_RGB_OPERATORS = frozenset(["rg", "RG"])
_CMYK_OPERATORS = frozenset(["k", "K"])
_CS_OPERATORS = frozenset(["cs", "CS"])

# Abbreviated inline-image color-space names (PDF spec Table 92) that pikepdf
# does NOT normalise inside Array values (e.g. Indexed base color space).
_INLINE_CS_TO_DEVICE: dict[str, Name] = {
    "/G": Name.DeviceGray,
    "/RGB": Name.DeviceRGB,
    "/CMYK": Name.DeviceCMYK,
    "/DeviceGray": Name.DeviceGray,
    "/DeviceRGB": Name.DeviceRGB,
    "/DeviceCMYK": Name.DeviceCMYK,
}


class ColorSpaceType(Enum):
    """Supported color space types."""

    DEVICE_GRAY = "DeviceGray"
    DEVICE_RGB = "DeviceRGB"
    DEVICE_CMYK = "DeviceCMYK"
    DEVICEN = "DeviceN"
    SEPARATION = "Separation"
    INDEXED = "Indexed"
    ICCBASED = "ICCBased"
    CAL_GRAY = "CalGray"
    CAL_RGB = "CalRGB"
    LAB = "Lab"


_DEFAULT_CS_NAMES: dict[ColorSpaceType, Name] = {
    ColorSpaceType.DEVICE_GRAY: Name.DefaultGray,
    ColorSpaceType.DEVICE_RGB: Name.DefaultRGB,
    ColorSpaceType.DEVICE_CMYK: Name.DefaultCMYK,
}

_DEVICE_CS_NAMES: dict[ColorSpaceType, Name] = {
    ColorSpaceType.DEVICE_GRAY: Name.DeviceGray,
    ColorSpaceType.DEVICE_RGB: Name.DeviceRGB,
    ColorSpaceType.DEVICE_CMYK: Name.DeviceCMYK,
}

_DEVICE_NAME_TO_TYPE: dict[Name, ColorSpaceType] = {
    Name.DeviceGray: ColorSpaceType.DEVICE_GRAY,
    Name.DeviceRGB: ColorSpaceType.DEVICE_RGB,
    Name.DeviceCMYK: ColorSpaceType.DEVICE_CMYK,
}

_N_COMPONENTS: dict[ColorSpaceType, int] = {
    ColorSpaceType.DEVICE_GRAY: 1,
    ColorSpaceType.DEVICE_RGB: 3,
    ColorSpaceType.DEVICE_CMYK: 4,
    ColorSpaceType.CAL_GRAY: 1,
    ColorSpaceType.CAL_RGB: 3,
}


@dataclass
class SpecialColorSpace:
    """Details of a special color space (Separation, DeviceN, or Indexed)."""

    type: str  # "Separation", "DeviceN", or "Indexed"
    alternate_space: str  # The alternate/base color space
    location: str  # Description of where it was found
    obj_ref: object | None = None  # Reference to the object for conversion


@dataclass
class ColorSpaceAnalysis:
    """Result of color space detection in a PDF."""

    device_gray_used: bool = False
    device_rgb_used: bool = False
    device_cmyk_used: bool = False
    devicen_used: bool = False
    separation_used: bool = False
    indexed_with_special_base: bool = False
    cal_gray_used: bool = False
    cal_rgb_used: bool = False
    lab_used: bool = False
    special_colorspaces: list[SpecialColorSpace] = field(default_factory=list)

    @property
    def detected_spaces(self) -> set[ColorSpaceType]:
        """Return set of detected color space types."""
        result: set[ColorSpaceType] = set()
        if self.device_gray_used:
            result.add(ColorSpaceType.DEVICE_GRAY)
        if self.device_rgb_used:
            result.add(ColorSpaceType.DEVICE_RGB)
        if self.device_cmyk_used:
            result.add(ColorSpaceType.DEVICE_CMYK)
        if self.devicen_used:
            result.add(ColorSpaceType.DEVICEN)
        if self.separation_used:
            result.add(ColorSpaceType.SEPARATION)
        if self.indexed_with_special_base:
            result.add(ColorSpaceType.INDEXED)
        return result

    @property
    def has_special_colorspaces(self) -> bool:
        """Return True if any special color spaces were detected."""
        return (
            self.devicen_used or self.separation_used or self.indexed_with_special_base
        )

    @property
    def calibrated_spaces(self) -> set[ColorSpaceType]:
        """Return set of detected calibrated color space types."""
        result: set[ColorSpaceType] = set()
        if self.cal_gray_used:
            result.add(ColorSpaceType.CAL_GRAY)
        if self.cal_rgb_used:
            result.add(ColorSpaceType.CAL_RGB)
        if self.lab_used:
            result.add(ColorSpaceType.LAB)
        return result
