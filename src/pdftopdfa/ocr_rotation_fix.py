# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ocrmypdf plugin to normalize visible OCR page boxes.

This plugin derives the replacement PDF page size from the actual rendered
image orientation instead of re-computing it from internal rotation metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import ocrmypdf
import pikepdf
from PIL import Image

if TYPE_CHECKING:
    from ocrmypdf._jobcontext import PageContext

logger = logging.getLogger(__name__)


def _is_landscape(width: float, height: float) -> bool | None:
    """Return the orientation of a rectangle, ignoring square cases."""
    if width == height:
        return None
    return width > height


def _should_swap_visible_page_axis(
    page_width_points: float,
    page_height_points: float,
    image_width: int,
    image_height: int,
) -> bool:
    """Decide whether the visible PDF page axes need swapping.

    The page dimensions come from the original PDF MediaBox, while the image
    dimensions come from the rasterized page after OCRmyPDF applied both the
    existing PDF ``/Rotate`` value and any autorotation correction.
    """
    page_landscape = _is_landscape(page_width_points, page_height_points)
    image_landscape = _is_landscape(float(image_width), float(image_height))
    if page_landscape is None or image_landscape is None:
        return False
    return page_landscape != image_landscape


def _normalize_box(
    source_box: list[float],
    media_box: list[float],
    swap_axis: bool,
) -> list[float]:
    """Map an original page box into the replacement page coordinate system."""
    offset_box = [
        source_box[0] - media_box[0],
        source_box[1] - media_box[1],
        source_box[2] - media_box[0],
        source_box[3] - media_box[1],
    ]
    if not swap_axis:
        return offset_box
    return [offset_box[1], offset_box[0], offset_box[3], offset_box[2]]


def _rewrite_visible_page_boxes(
    output_pdf: Path,
    page_context: PageContext,
    *,
    swap_axis: bool,
) -> None:
    """Rewrite the replacement page boxes using a unified visible orientation."""
    pageinfo = page_context.pageinfo
    original_media_box = [float(value) for value in pageinfo.mediabox]
    page_width_points = 72.0 * float(pageinfo.width_inches)
    page_height_points = 72.0 * float(pageinfo.height_inches)
    target_width = page_height_points if swap_axis else page_width_points
    target_height = page_width_points if swap_axis else page_height_points

    temp_output_pdf = output_pdf.with_name(f"{output_pdf.stem}_rotfix.pdf")

    with pikepdf.open(output_pdf) as pdf:
        pdf_page = pdf.pages[0]
        pdf_page.MediaBox = pikepdf.Array([0, 0, target_width, target_height])

        for box_name in ("CropBox", "TrimBox", "ArtBox", "BleedBox"):
            source_box = [float(value) for value in getattr(pageinfo, box_name.lower())]
            pdf_page[pikepdf.Name(f"/{box_name}")] = pikepdf.Array(
                _normalize_box(source_box, original_media_box, swap_axis)
            )

        pdf.save(temp_output_pdf)

    temp_output_pdf.replace(output_pdf)


@ocrmypdf.hookimpl
def filter_pdf_page(page: PageContext, image_filename: Path, output_pdf: Path) -> Path:
    """Normalize visible page PDF sizing to the actual rendered image orientation."""
    page_width_points = 72.0 * float(page.pageinfo.width_inches)
    page_height_points = 72.0 * float(page.pageinfo.height_inches)

    with Image.open(image_filename) as image:
        image_width, image_height = image.size

    swap_axis = _should_swap_visible_page_axis(
        page_width_points,
        page_height_points,
        image_width,
        image_height,
    )

    with pikepdf.open(output_pdf) as pdf:
        pdf_page = pdf.pages[0]
        current_width = float(pdf_page.mediabox[2]) - float(pdf_page.mediabox[0])
        current_height = float(pdf_page.mediabox[3]) - float(pdf_page.mediabox[1])

    logger.debug(
        "OCR visible-page rotation normalization: original_rotate=%s, "
        "image_size=%sx%s, current_page_size=%.2fx%.2f, swap_axis=%s",
        page.pageinfo.rotation,
        image_width,
        image_height,
        current_width,
        current_height,
        swap_axis,
    )

    _rewrite_visible_page_boxes(output_pdf, page, swap_axis=swap_axis)
    return output_pdf
