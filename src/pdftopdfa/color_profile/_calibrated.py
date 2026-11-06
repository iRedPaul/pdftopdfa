# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Compatibility entry point for the former calibrated-space conversion."""

import logging

from pikepdf import Pdf, Stream

from ._types import ColorSpaceType

logger = logging.getLogger(__name__)


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
    return 0
