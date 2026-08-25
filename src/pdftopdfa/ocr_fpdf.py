# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Polygon-aware fpdf2 rendering for OCRmyPDF text layers."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ocrmypdf.fpdf_renderer as fpdf_renderer
import ocrmypdf.fpdf_renderer.renderer as renderer_module
from fpdf import FPDF
from ocrmypdf.fpdf_renderer.renderer import (
    Fpdf2PdfRenderer as _Fpdf2PdfRenderer,
)
from ocrmypdf.fpdf_renderer.renderer import (
    WordRenderData,
    _is_rtl_text,
)
from ocrmypdf.models.ocr_element import OcrClass, OcrElement
from pikepdf import Matrix

logger = logging.getLogger(__name__)

_renderer_lock = threading.RLock()


def _valid_polygon(value: Any) -> list[tuple[float, float]] | None:
    try:
        if value is None or len(value) != 4:
            return None
        if any(len(point) != 2 for point in value):
            return None
        points = [(float(point[0]), float(point[1])) for point in value]
    except (IndexError, TypeError, ValueError):
        return None

    if any(not math.isfinite(coordinate) for point in points for coordinate in point):
        return None

    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points,
                [*points[1:], points[0]],
                strict=True,
            )
        )
    )
    return points if area > 1e-6 else None


def _transform_points(
    matrix: Matrix,
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    transformed = [matrix.transform(point) for point in points]
    if any(
        not math.isfinite(coordinate) for point in transformed for coordinate in point
    ):
        return None
    return transformed


class PolygonFpdf2PdfRenderer(_Fpdf2PdfRenderer):
    """Render valid line and word polygons without reducing them to page AABBs."""

    def _render_line(self, pdf: FPDF, line: OcrElement) -> None:
        previous_mcid = getattr(self, "_pdftopdfa_mcid", None)
        previous_emitted = getattr(self, "_pdftopdfa_emitted", False)
        mcid = getattr(line, "_pdftopdfa_mcid", None)
        self._pdftopdfa_mcid = (
            mcid
            if isinstance(mcid, int) and not isinstance(mcid, bool) and mcid >= 0
            else None
        )
        self._pdftopdfa_emitted = False
        try:
            self._render_polygon_line(pdf, line)
            # Every suppression path returns without emitting text, so the
            # manifest line has no marked content to bind to. Report it here;
            # the document manifest drops the line during reconciliation.
            if self._pdftopdfa_mcid is not None and not self._pdftopdfa_emitted:
                logger.warning(
                    "OCR line with MCID %d was suppressed by the renderer: %r",
                    self._pdftopdfa_mcid,
                    line.text,
                )
        finally:
            self._pdftopdfa_mcid = previous_mcid
            self._pdftopdfa_emitted = previous_emitted

    def _render_polygon_line(self, pdf: FPDF, line: OcrElement) -> None:
        line_polygon = _valid_polygon(line.poly)
        try:
            textangle = float(line.textangle or 0.0)
        except (TypeError, ValueError):
            textangle = math.nan
        if line_polygon is None or not math.isfinite(textangle):
            super()._render_line(pdf, line)
            return

        line_points = [
            (
                self.coord_transform.px_to_pt(x),
                self.coord_transform.px_to_pt(y),
            )
            for x, y in line_polygon
        ]
        line_reference_matrix = Matrix().translated(*line_points[0]).rotated(-textangle)
        local_line_points = _transform_points(
            line_reference_matrix.inverse(),
            line_points,
        )
        if local_line_points is None:
            super()._render_line(pdf, line)
            return

        local_left = min(point[0] for point in local_line_points)
        local_top = min(point[1] for point in local_line_points)
        local_right = max(point[0] for point in local_line_points)
        local_bottom = max(point[1] for point in local_line_points)
        line_width = local_right - local_left
        line_height = local_bottom - local_top
        if line_width <= 1e-6 or line_height <= 1e-6:
            super()._render_line(pdf, line)
            return

        slope = 0.0
        intercept = 0.0
        if line.baseline is not None:
            try:
                slope = float(line.baseline.slope)
                intercept = float(line.baseline.intercept)
            except (TypeError, ValueError):
                slope = math.nan
                intercept = math.nan
            if not math.isfinite(slope) or not math.isfinite(intercept):
                super()._render_line(pdf, line)
                return

        line_matrix = line_reference_matrix.translated(local_left, local_top)
        if line.baseline is None:
            default_font_manager = self.multi_font_manager.fonts["NotoSans-Regular"]
            ascent, descent, units_per_em = default_font_manager.get_font_metrics()
            ascent_norm = ascent / units_per_em
            descent_norm = descent / units_per_em
            intercept_pt = (
                -abs(descent_norm) * line_height / (ascent_norm + abs(descent_norm))
            )
        else:
            intercept_pt = self.coord_transform.px_to_pt(intercept)

        slope_angle = math.degrees(math.atan(slope)) if slope != 0.0 else 0.0
        baseline_matrix = (
            line_matrix.translated(0, line_height)
            .translated(0, intercept_pt)
            .rotated(slope_angle)
        )
        font_size = line_height + intercept_pt
        if font_size < 1.0:
            font_size = line_height * 0.8

        words = [
            word
            for word in line.children
            if word.ocr_class == OcrClass.WORD and word.text
        ]
        inv_baseline_matrix = baseline_matrix.inverse()
        projected_words: list[
            tuple[OcrElement, float, float, list[tuple[float, float]]]
        ] = []
        for word in words:
            word_polygon = _valid_polygon(word.poly)
            if word_polygon is None:
                super()._render_line(pdf, line)
                return
            word_points = [
                (
                    self.coord_transform.px_to_pt(x),
                    self.coord_transform.px_to_pt(y),
                )
                for x, y in word_polygon
            ]
            local_word_points = _transform_points(inv_baseline_matrix, word_points)
            if local_word_points is None:
                super()._render_line(pdf, line)
                return
            word_left = min(point[0] for point in local_word_points)
            word_right = max(point[0] for point in local_word_points)
            if word_right - word_left <= 1e-6:
                super()._render_line(pdf, line)
                return
            projected_words.append(
                (word, word_left, word_right - word_left, word_points)
            )

        if not words:
            return
        line_language = line.language
        if self.debug_options.render_line_bbox:
            self._render_debug_line_bbox(
                pdf,
                min(point[0] for point in line_points),
                min(point[1] for point in line_points),
                max(point[0] for point in line_points),
                max(point[1] for point in line_points),
            )
        if not self._check_aspect_ratio_plausible(
            pdf,
            words,
            font_size,
            slope_angle,
            line_width,
            line_height,
            line_language,
        ):
            return

        word_render_data = []
        for word, word_left, word_width, word_points in projected_words:
            if self.debug_options.render_word_bbox:
                self._render_debug_word_bbox(
                    pdf,
                    min(point[0] for point in word_points),
                    min(point[1] for point in word_points),
                    max(point[0] for point in word_points),
                    max(point[1] for point in word_points),
                )

            font_manager = self.multi_font_manager.select_font_for_word(
                word.text,
                line_language,
            )
            font_family = self._register_font(pdf, font_manager)
            pdf.set_font(font_family, size=font_size)

            word_is_rtl = self.invisible_text and _is_rtl_text(word.text)
            if word_is_rtl:
                saved_shaping = pdf.text_shaping
                pdf.text_shaping = None
                natural_width = pdf.get_string_width(word.text)
                pdf.text_shaping = saved_shaping
            else:
                natural_width = pdf.get_string_width(word.text)
            word_tz = (
                (word_width / natural_width) * 100
                if natural_width > 0 and word_width > 0
                else 100.0
            )
            word_render_data.append(
                WordRenderData(
                    text=word.text,
                    x_baseline=word_left,
                    font_family=font_family,
                    word_tz=word_tz,
                    is_rtl=word_is_rtl,
                )
            )

        self._emit_line_bt_block(
            pdf,
            word_render_data,
            baseline_matrix,
            font_size,
            -textangle + slope_angle,
        )

    def _emit_line_bt_block(
        self,
        pdf: FPDF,
        word_render_data: list[WordRenderData],
        baseline_matrix: Matrix,
        font_size: float,
        total_rotation_deg: float,
    ) -> None:
        self._pdftopdfa_emitted = True
        mcid = getattr(self, "_pdftopdfa_mcid", None)
        if mcid is None:
            super()._emit_line_bt_block(
                pdf,
                word_render_data,
                baseline_matrix,
                font_size,
                total_rotation_deg,
            )
            return

        pdf._out(f"/Span <</MCID {mcid}>> BDC")
        try:
            super()._emit_line_bt_block(
                pdf,
                word_render_data,
                baseline_matrix,
                font_size,
                total_rotation_deg,
            )
        finally:
            pdf._out("EMC")


@contextmanager
def install_fpdf_renderer() -> Iterator[None]:
    """Temporarily install the polygon-aware OCRmyPDF fpdf2 renderer."""
    with _renderer_lock:
        previous_module_renderer = renderer_module.Fpdf2PdfRenderer
        previous_exported_renderer = fpdf_renderer.Fpdf2PdfRenderer
        renderer_module.Fpdf2PdfRenderer = PolygonFpdf2PdfRenderer
        fpdf_renderer.Fpdf2PdfRenderer = PolygonFpdf2PdfRenderer
        try:
            yield
        finally:
            renderer_module.Fpdf2PdfRenderer = previous_module_renderer
            fpdf_renderer.Fpdf2PdfRenderer = previous_exported_renderer
