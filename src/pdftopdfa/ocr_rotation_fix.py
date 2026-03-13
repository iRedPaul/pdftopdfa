# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ocrmypdf plugin to normalize OCR page rotation handling.

The visible page image produced by OCRmyPDF already has the PDF page's
``/Rotate`` value and any OCR autorotation baked into the rasterized pixels.
This plugin derives the replacement PDF page size from the actual rendered
image orientation instead of re-computing it from internal rotation metadata.
It also compensates for pypdfium rasterization paths that overwrite an
existing page rotation instead of composing with it.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import ocrmypdf
import pikepdf
from PIL import Image

if TYPE_CHECKING:
    from ocrmypdf._jobcontext import PageContext

logger = logging.getLogger(__name__)


def _read_page_rotate(input_file: Path, pageno: int) -> int:
    """Read the existing PDF /Rotate value for a page."""
    with pikepdf.open(input_file) as pdf:
        return int(pdf.pages[pageno - 1].obj.get("/Rotate", 0)) % 360


def _compose_page_rotation(existing_rotate: int, requested_rotate: int | None) -> int:
    """Compose OCRmyPDF's requested rotation with the page's existing /Rotate."""
    if not requested_rotate:
        return existing_rotate % 360
    return (existing_rotate + requested_rotate) % 360


def _write_rotated_temp_pdf(
    input_file: Path,
    *,
    pageno: int,
    composed_rotation: int,
    temp_pdf: Path,
) -> Path:
    """Write a temporary PDF with a composed /Rotate value for the target page."""
    with pikepdf.open(input_file) as pdf:
        pdf.pages[pageno - 1].Rotate = composed_rotation
        pdf.save(temp_pdf)
    return temp_pdf


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


@ocrmypdf.hookimpl(tryfirst=True)
def rasterize_pdf_page(
    input_file: Path,
    output_file: Path,
    raster_device: str,
    raster_dpi,
    pageno: int,
    page_dpi,
    rotation: int | None,
    filter_vector: bool,
    stop_on_soft_error: bool,
    options,
    use_cropbox: bool,
) -> Path | None:
    """Compose existing /Rotate with OCR autorotation for pypdfium rasterization."""
    if options is not None and options.rasterizer == "ghostscript":
        return None
    if not rotation:
        return None

    existing_rotate = _read_page_rotate(input_file, pageno)
    if existing_rotate == 0:
        return None

    from ocrmypdf.builtin_plugins import pypdfium as pypdfium_plugin

    if pypdfium_plugin.pdfium is None:
        return None

    composed_rotation = _compose_page_rotation(existing_rotate, rotation)
    temp_pdf = output_file.with_name(f"{output_file.stem}_rotfix_input.pdf")
    _write_rotated_temp_pdf(
        input_file,
        pageno=pageno,
        composed_rotation=composed_rotation,
        temp_pdf=temp_pdf,
    )

    adjusted_page_dpi = page_dpi
    if page_dpi is not None and rotation % 180 == 90:
        adjusted_page_dpi = page_dpi.flip_axis()

    logger.debug(
        "OCR raster rotation normalization: original_rotate=%s, "
        "requested_correction=%s, composed_rotation=%s, page=%s",
        existing_rotate,
        rotation,
        composed_rotation,
        pageno,
    )

    try:
        return pypdfium_plugin.rasterize_pdf_page(
            temp_pdf,
            output_file,
            raster_device,
            raster_dpi,
            pageno,
            adjusted_page_dpi,
            0,
            filter_vector,
            stop_on_soft_error,
            options,
            use_cropbox,
        )
    finally:
        if not getattr(options, "keep_temporary_files", False):
            with suppress(FileNotFoundError):
                temp_pdf.unlink()


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
