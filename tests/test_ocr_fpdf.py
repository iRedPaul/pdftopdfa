# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for polygon-aware OCR text rendering."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

import ocrmypdf
import ocrmypdf.fpdf_renderer as fpdf_renderer
import pikepdf
import pytest
from ocrmypdf.font import MultiFontManager
from ocrmypdf.models.ocr_element import (
    Baseline,
    BoundingBox,
    OcrClass,
    OcrElement,
)
from pdfminer.high_level import extract_text

from pdftopdfa.ocr_fpdf import PolygonFpdf2PdfRenderer, install_fpdf_renderer


def _ocr_page(
    polygon: list[tuple[float, float]],
    bbox: BoundingBox,
    *,
    page_bbox: BoundingBox,
    text: str,
    textangle: float,
    baseline: Baseline,
    dpi: float,
) -> OcrElement:
    word = OcrElement(
        ocr_class=OcrClass.WORD,
        bbox=bbox,
        poly=polygon,
        text=text,
        direction="ltr",
        language="en",
    )
    line = OcrElement(
        ocr_class=OcrClass.LINE,
        bbox=bbox,
        poly=polygon,
        text=text,
        children=[word],
        direction="ltr",
        language="en",
        baseline=baseline,
        textangle=textangle,
    )
    return OcrElement(
        ocr_class=OcrClass.PAGE,
        bbox=page_bbox,
        children=[line],
        direction="ltr",
        language="en",
        dpi=dpi,
    )


def _render(page: OcrElement, dpi: float, output: Path) -> None:
    font_dir = Path(ocrmypdf.__file__).parent / "data"
    with install_fpdf_renderer():
        fpdf_renderer.Fpdf2MultiPageRenderer(
            pages_data=[(0, page, dpi)],
            multi_font_manager=MultiFontManager(font_dir),
            invisible_text=True,
        ).render(output)


def _text_operators(page: pikepdf.Page) -> bytes:
    names = {"q", "cm", "BT", "Tr", "Td", "Tf", "Tz", "Tj", "ET", "Q"}
    return pikepdf.unparse_content_stream(
        [
            instruction
            for instruction in pikepdf.parse_content_stream(page)
            if str(instruction.operator) in names
        ]
    )


def test_renderer_installation_is_scoped() -> None:
    original_renderer = fpdf_renderer.Fpdf2PdfRenderer

    with install_fpdf_renderer():
        assert fpdf_renderer.Fpdf2PdfRenderer is not original_renderer

    assert fpdf_renderer.Fpdf2PdfRenderer is original_renderer
    with pytest.raises(RuntimeError):
        with install_fpdf_renderer():
            raise RuntimeError("test")
    assert fpdf_renderer.Fpdf2PdfRenderer is original_renderer


def test_renderer_wraps_only_emitted_level_a_lines(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    polygon = [(20, 20), (120, 20), (120, 60), (20, 60)]
    page = _ocr_page(
        polygon,
        BoundingBox(left=20, top=20, right=120, bottom=60),
        page_bbox=BoundingBox(left=0, top=0, right=200, bottom=100),
        text="Tagged",
        textangle=0,
        baseline=Baseline(slope=0, intercept=0),
        dpi=300,
    )
    page.children[0]._pdftopdfa_mcid = 7
    tagged_output = tmp_path / "tagged.pdf"
    discarded_output = tmp_path / "discarded.pdf"

    _render(page, 300, tagged_output)
    with (
        caplog.at_level(logging.WARNING, logger="pdftopdfa.ocr_fpdf"),
        patch.object(
            PolygonFpdf2PdfRenderer,
            "_check_aspect_ratio_plausible",
            return_value=False,
        ),
    ):
        _render(page, 300, discarded_output)

    assert "OCR line with MCID 7 was suppressed by the renderer" in caplog.text

    with (
        pikepdf.Pdf.open(tagged_output) as tagged_pdf,
        pikepdf.Pdf.open(discarded_output) as discarded_pdf,
    ):
        tagged_stream = tagged_pdf.pages[0].Contents.read_bytes()
        discarded_stream = discarded_pdf.pages[0].Contents.read_bytes()

    marker = b"/Span <</MCID 7>> BDC"
    assert tagged_stream.count(marker) == 1
    assert tagged_stream.index(marker) < tagged_stream.index(b"BT")
    assert tagged_stream.index(b"ET") < tagged_stream.index(b"EMC")
    assert b"MCID" not in discarded_stream
    assert b"BDC" not in discarded_stream
    assert b"EMC" not in discarded_stream


def test_renderer_does_not_mark_normal_ocr_lines(tmp_path: Path) -> None:
    page = _ocr_page(
        [(20, 20), (120, 20), (120, 60), (20, 60)],
        BoundingBox(left=20, top=20, right=120, bottom=60),
        page_bbox=BoundingBox(left=0, top=0, right=200, bottom=100),
        text="Normal",
        textangle=0,
        baseline=Baseline(slope=0, intercept=0),
        dpi=300,
    )
    output = tmp_path / "normal.pdf"

    _render(page, 300, output)

    with pikepdf.Pdf.open(output) as document:
        stream = document.pages[0].Contents.read_bytes()
    assert b"MCID" not in stream
    assert b"BDC" not in stream
    assert b"EMC" not in stream


def test_renderer_uses_polygon_rotation_and_height(tmp_path: Path) -> None:
    bbox = BoundingBox(left=20, top=20, right=120, bottom=60)
    page_bbox = BoundingBox(left=0, top=0, right=200, bottom=100)
    baseline = Baseline(slope=0, intercept=0)
    straight_polygon = [(20, 20), (120, 20), (120, 60), (20, 60)]
    slanted_polygon = [(20, 20), (120, 40), (120, 60), (20, 40)]
    angle = math.degrees(math.atan2(20, 100))
    straight = _ocr_page(
        straight_polygon,
        bbox,
        page_bbox=page_bbox,
        text="Polygon",
        textangle=0,
        baseline=baseline,
        dpi=300,
    )
    slanted = _ocr_page(
        slanted_polygon,
        bbox,
        page_bbox=page_bbox,
        text="Polygon",
        textangle=-angle,
        baseline=baseline,
        dpi=300,
    )
    straight_pdf = tmp_path / "straight.pdf"
    slanted_pdf = tmp_path / "slanted.pdf"

    _render(straight, 300, straight_pdf)
    _render(slanted, 300, slanted_pdf)

    with (
        pikepdf.Pdf.open(straight_pdf) as straight_document,
        pikepdf.Pdf.open(slanted_pdf) as slanted_document,
    ):
        straight_stream = straight_document.pages[0].Contents.read_bytes()
        slanted_stream = slanted_document.pages[0].Contents.read_bytes()
        straight_instructions = list(
            pikepdf.parse_content_stream(straight_document.pages[0])
        )
        slanted_instructions = list(
            pikepdf.parse_content_stream(slanted_document.pages[0])
        )

        assert straight_stream != slanted_stream
        assert not any(
            str(instruction.operator) == "cm" for instruction in straight_instructions
        )
        matrix = next(
            instruction.operands
            for instruction in slanted_instructions
            if str(instruction.operator) == "cm"
        )
        font = next(
            instruction.operands
            for instruction in slanted_instructions
            if str(instruction.operator) == "Tf"
        )

    radians = math.radians(angle)
    assert [float(value) for value in matrix[:4]] == pytest.approx(
        [math.cos(radians), -math.sin(radians), math.sin(radians), math.cos(radians)],
        abs=1e-6,
    )
    expected_height = 20 * math.cos(radians) * 72 / 300
    assert float(font[1]) == pytest.approx(expected_height, abs=0.01)
    assert float(font[1]) != pytest.approx(bbox.height * 72 / 300, abs=0.01)


def test_renderer_extracts_vertical_text_with_rotation(tmp_path: Path) -> None:
    polygon = [(40, 10), (60, 10), (60, 110), (40, 110)]
    page = _ocr_page(
        polygon,
        BoundingBox(left=40, top=10, right=60, bottom=110),
        page_bbox=BoundingBox(left=0, top=0, right=100, bottom=140),
        text="Vertical",
        textangle=-90,
        baseline=Baseline(slope=0, intercept=0),
        dpi=300,
    )
    output = tmp_path / "vertical.pdf"

    _render(page, 300, output)

    assert "".join(extract_text(output).split()) == "Vertical"
    with pikepdf.Pdf.open(output) as document:
        matrix = next(
            instruction.operands
            for instruction in pikepdf.parse_content_stream(document.pages[0])
            if str(instruction.operator) == "cm"
        )
    assert [float(value) for value in matrix[:4]] == pytest.approx(
        [0, -1, 1, 0],
        abs=1e-6,
    )


def test_renderer_preserves_physical_geometry_across_dpi(tmp_path: Path) -> None:
    textangle = -5.710593137499642
    baseline = Baseline(
        slope=0.04926108374384239,
        intercept=-9.950371902099896,
    )
    page_300 = _ocr_page(
        [(20, 20), (220, 30), (210, 90), (10, 60)],
        BoundingBox(left=10, top=20, right=220, bottom=90),
        page_bbox=BoundingBox(left=0, top=0, right=400, bottom=200),
        text="Scale",
        textangle=textangle,
        baseline=baseline,
        dpi=300,
    )
    page_150 = _ocr_page(
        [(10, 10), (110, 15), (105, 45), (5, 30)],
        BoundingBox(left=5, top=10, right=110, bottom=45),
        page_bbox=BoundingBox(left=0, top=0, right=200, bottom=100),
        text="Scale",
        textangle=textangle,
        baseline=Baseline(
            slope=baseline.slope,
            intercept=baseline.intercept / 2,
        ),
        dpi=150,
    )
    output_300 = tmp_path / "scale-300.pdf"
    output_150 = tmp_path / "scale-150.pdf"

    _render(page_300, 300, output_300)
    _render(page_150, 150, output_150)

    with (
        pikepdf.Pdf.open(output_300) as document_300,
        pikepdf.Pdf.open(output_150) as document_150,
    ):
        assert _text_operators(document_300.pages[0]) == _text_operators(
            document_150.pages[0]
        )
        instructions = list(pikepdf.parse_content_stream(document_300.pages[0]))
        matrix = next(
            instruction.operands
            for instruction in instructions
            if str(instruction.operator) == "cm"
        )
        font = next(
            instruction.operands
            for instruction in instructions
            if str(instruction.operator) == "Tf"
        )

    baseline_angle = -textangle + math.degrees(math.atan(baseline.slope))
    radians = math.radians(baseline_angle)
    assert [float(value) for value in matrix[:4]] == pytest.approx(
        [math.cos(radians), -math.sin(radians), math.sin(radians), math.cos(radians)],
        abs=1e-6,
    )
    assert float(font[1]) == pytest.approx(12.1792552082, abs=0.01)
