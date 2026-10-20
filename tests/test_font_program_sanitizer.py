# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/font_program.py (truncated hmtx repair)."""

from io import BytesIO

import pikepdf
import pytest
from conftest import new_pdf, open_pdf
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, TTLibError
from fontTools.ttLib.sfnt import SFNTWriter
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.font_program import (
    _pad_truncated_hmtx,
    sanitize_font_programs,
)
from pdftopdfa.sanitizers.font_widths import _parse_w_array, sanitize_font_widths

# C and D share B's advance, so hmtx stores them as trailing side bearings.
_ADVANCES = {".notdef": 500, "A": 700, "B": 600, "C": 600, "D": 600}
_X_MINS = {"A": 50, "B": 60, "C": 70, "D": 80}


def _rect_glyph(x_min: int) -> Glyph:
    pen = TTGlyphPen(None)
    pen.moveTo((x_min, 0))
    pen.lineTo((x_min, 500))
    pen.lineTo((x_min + 400, 500))
    pen.lineTo((x_min + 400, 0))
    pen.closePath()
    return pen.glyph()


def _make_font_data() -> bytes:
    """Builds a TrueType font with numberOfHMetrics < numGlyphs."""
    glyph_order = list(_ADVANCES)
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({ord(name): name for name in _X_MINS})
    glyphs = {name: _rect_glyph(x_min) for name, x_min in _X_MINS.items()}
    glyphs[".notdef"] = Glyph()
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(
        {name: (_ADVANCES[name], _X_MINS.get(name, 0)) for name in glyph_order}
    )
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


def _truncate_hmtx(font_data: bytes, num_bytes: int) -> bytes:
    """Returns *font_data* with the last *num_bytes* of hmtx removed."""
    reader = TTFont(BytesIO(font_data), lazy=True).reader
    tags = list(reader.keys())
    buf = BytesIO()
    writer = SFNTWriter(buf, len(tags), reader.sfntVersion)
    for tag in tags:
        data = reader[tag]
        writer[tag] = data[:-num_bytes] if tag == "hmtx" else data
    writer.close()
    return buf.getvalue()


def _hmtx_metrics(font_data: bytes) -> dict[str, tuple[int, int]]:
    tt_font = TTFont(BytesIO(font_data))
    try:
        return dict(tt_font["hmtx"].metrics)
    finally:
        tt_font.close()


def _add_cidfont_page(pdf: Pdf, font_data: bytes, w_array: list) -> Dictionary:
    """Adds a page using a Type0/CIDFontType2 font with Identity CIDToGIDMap."""
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)
    descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            Flags=32,
            FontBBox=Array([0, -200, 1000, 800]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )
    )
    cidfont = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry="Adobe", Ordering="Identity", Supplement=0
            ),
            FontDescriptor=descriptor,
            DW=500,
            W=Array(w_array),
            CIDToGIDMap=Name.Identity,
        )
    )
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )
    )
    content = pdf.make_stream(b"BT /F1 12 Tf 10 10 Td <0001000200030004> Tj ET")
    page = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=font)),
        Contents=content,
    )
    pdf.pages.append(pikepdf.Page(page))
    return font


def _roundtrip(pdf: Pdf) -> Pdf:
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return open_pdf(buf)


def _font_stream(pdf: Pdf, page_index: int = 0) -> pikepdf.Stream:
    font = pdf.pages[page_index].Resources.Font.F1
    return font.DescendantFonts[0].FontDescriptor.FontFile2


class TestPadTruncatedHmtx:
    def test_missing_side_bearing_is_padded_with_glyph_xmin(self) -> None:
        font_data = _make_font_data()
        truncated = _truncate_hmtx(font_data, 2)
        with pytest.raises(TTLibError, match="not enough 'hmtx' table data"):
            _hmtx_metrics(truncated)

        repaired = _pad_truncated_hmtx(truncated)

        assert repaired is not None
        assert _hmtx_metrics(repaired) == _hmtx_metrics(font_data)

    def test_odd_trailing_byte_is_replaced(self) -> None:
        font_data = _make_font_data()

        repaired = _pad_truncated_hmtx(_truncate_hmtx(font_data, 1))

        assert repaired is not None
        assert _hmtx_metrics(repaired) == _hmtx_metrics(font_data)

    def test_repaired_program_keeps_other_tables_and_valid_checksums(self) -> None:
        font_data = _make_font_data()

        repaired = _pad_truncated_hmtx(_truncate_hmtx(font_data, 2))

        original = TTFont(BytesIO(font_data), lazy=True).reader
        result = TTFont(BytesIO(repaired), lazy=True, checkChecksums=2)
        assert sorted(result.reader.keys()) == sorted(original.keys())
        for tag in original.keys():
            if tag != "head":
                assert result.reader[tag] == original[tag], tag
        # Loading every table with checkChecksums=2 raises on bad checksums
        for tag in result.keys():
            result[tag]

    def test_complete_hmtx_returns_none(self) -> None:
        assert _pad_truncated_hmtx(_make_font_data()) is None

    def test_missing_advance_widths_returns_none(self) -> None:
        # Drops both trailing side bearings and part of B's long metric
        truncated = _truncate_hmtx(_make_font_data(), 6)

        assert _pad_truncated_hmtx(truncated) is None

    def test_non_sfnt_data_returns_none(self) -> None:
        assert _pad_truncated_hmtx(b"%!PS-AdobeFont-1.0: Test") is None


class TestSanitizeFontPrograms:
    def test_truncated_cidfont_program_is_repaired_in_place(self) -> None:
        pdf = new_pdf()
        _add_cidfont_page(pdf, _truncate_hmtx(_make_font_data(), 2), [1, [700]])
        pdf = _roundtrip(pdf)
        stream_objgen = _font_stream(pdf).objgen

        result = sanitize_font_programs(pdf)

        assert result == {"font_hmtx_repaired": 1}
        stream = _font_stream(pdf)
        assert stream.objgen == stream_objgen
        data = bytes(stream.read_bytes())
        assert int(stream.Length1) == len(data)
        assert _hmtx_metrics(data) == _hmtx_metrics(_make_font_data())

    def test_shared_program_is_repaired_once(self) -> None:
        pdf = new_pdf()
        font = _add_cidfont_page(pdf, _truncate_hmtx(_make_font_data(), 2), [1, [700]])
        pdf.pages.append(
            pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 612, 792]),
                    Resources=Dictionary(Font=Dictionary(F1=font)),
                )
            )
        )
        pdf = _roundtrip(pdf)

        assert sanitize_font_programs(pdf) == {"font_hmtx_repaired": 1}

    def test_complete_program_is_not_rewritten(self) -> None:
        pdf = new_pdf()
        _add_cidfont_page(pdf, _make_font_data(), [1, [700]])
        pdf = _roundtrip(pdf)
        original = bytes(_font_stream(pdf).read_raw_bytes())

        assert sanitize_font_programs(pdf) == {"font_hmtx_repaired": 0}
        assert bytes(_font_stream(pdf).read_raw_bytes()) == original

    def test_repair_lets_width_sanitizer_fix_declared_widths(self) -> None:
        """Regression: a truncated hmtx used to hide /W mismatches (6.2.11.5)."""
        pdf = new_pdf()
        # CID 4 (D) is declared narrower than its 600-unit program advance
        _add_cidfont_page(
            pdf, _truncate_hmtx(_make_font_data(), 2), [1, [700, 600, 600, 563]]
        )
        pdf = _roundtrip(pdf)
        assert sanitize_font_widths(pdf)["cidfont_widths_fixed"] == 0

        sanitize_font_programs(pdf)
        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 1
        cidfont = pdf.pages[0].Resources.Font.F1.DescendantFonts[0]
        assert _parse_w_array(cidfont.W)[4] == 600

    def test_sanitize_for_pdfa_repairs_before_width_validation(self) -> None:
        pdf = new_pdf()
        _add_cidfont_page(
            pdf, _truncate_hmtx(_make_font_data(), 2), [1, [700, 600, 600, 563]]
        )
        pdf = _roundtrip(pdf)

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["font_hmtx_repaired"] == 1
        assert result["cidfont_widths_fixed"] == 1
