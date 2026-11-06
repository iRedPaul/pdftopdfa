# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Compatibility entry point for the former calibrated-space conversion."""

import logging
import math
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import ConversionError
from ._types import ColorSpaceType

logger = logging.getLogger(__name__)


def _calibration_numbers(value, count: int) -> bool:
    """Check a calibration entry's dimensions and finite PDF-number values."""
    return (
        isinstance(value, Array)
        and len(value) == count
        and all(
            not isinstance(number, bool)
            and isinstance(number, (int, float, Decimal))
            and math.isfinite(number)
            for number in value
        )
    )


def _validate_calibrated_space(cs) -> None:
    """Reject malformed calibration rather than substituting unrelated colorimetry."""
    if len(cs) != 2 or not isinstance(cs[1], Dictionary):
        raise ConversionError(
            f"Malformed {cs[0]} color space: expected calibration dictionary"
        )
    params = cs[1]
    white = params.get("/WhitePoint")
    if not _calibration_numbers(white, 3) or not (
        white[0] > 0 and white[1] == 1 and white[2] > 0
    ):
        raise ConversionError(f"Malformed {cs[0]} color space: invalid /WhitePoint")
    black = params.get("/BlackPoint")
    if "/BlackPoint" in params and (
        not _calibration_numbers(black, 3) or any(number < 0 for number in black)
    ):
        raise ConversionError(f"Malformed {cs[0]} color space: invalid /BlackPoint")
    if "/Gamma" in params:
        gamma = params.Gamma
        if cs[0] == Name.CalGray:
            valid = (
                not isinstance(gamma, bool)
                and isinstance(gamma, (int, float, Decimal))
                and math.isfinite(gamma)
                and gamma > 0
            )
        else:
            valid = _calibration_numbers(gamma, 3)
        if not valid:
            raise ConversionError(f"Malformed {cs[0]} color space: invalid /Gamma")
    if (
        cs[0] == Name.CalRGB
        and "/Matrix" in params
        and not _calibration_numbers(params.Matrix, 9)
    ):
        raise ConversionError("Malformed /CalRGB color space: invalid /Matrix")


def validate_calibrated_colorspaces(pdf: Pdf) -> None:
    """Validate reachable CalGray/CalRGB arrays in every PDF resource context."""
    pending = [pdf.Root]
    visited = set()
    while pending:
        value = pending.pop()
        if not isinstance(value, (Array, Dictionary, Stream)):
            continue
        if value.objgen != (0, 0):
            if value.objgen in visited:
                continue
            visited.add(value.objgen)
        if isinstance(value, Array):
            if (
                value
                and isinstance(value[0], Name)
                and value[0] in (Name.CalGray, Name.CalRGB)
            ):
                _validate_calibrated_space(value)
            pending.extend(value)
        else:
            pending.extend(value.values())


def _convert_calibrated_colorspaces(
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Preserve calibrated spaces requested through the deprecated option.

    CalGray and CalRGB are already permitted by PDF/A-2 and PDF/A-3. Replacing
    them with a generic grayscale or sRGB ICC profile discards their gamma,
    white point, black point, and matrix without converting the color values.
    Until equivalent ICC profiles can be generated from those parameters,
    retain the original color spaces and report that no conversion occurred.

    The signature and zero replacement count are retained for compatibility.
    Neither the PDF nor the supplied ICC stream cache is modified.
    """
    logger.warning(
        "convert_calibrated=True is deprecated: CalGray/CalRGB color spaces "
        "are preserved because generic ICC replacement changes their calibration"
    )
    validate_calibrated_colorspaces(pdf)
    return 0
