# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for glyph usage collection from content streams."""

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts.glyph_usage import (
    _extract_char_codes,
    _is_cidfont,
    collect_font_usage,
)


def _make_page_with_content(pdf, content_bytes, font_dict, resources=None):
    """Helper: creates a page with content stream and font resources."""
    if resources is None:
        resources = Dictionary(Font=font_dict)
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=resources,
        Contents=pdf.make_stream(content_bytes),
    )
    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)
    return page


class TestExtractCharCodes:
    """Tests for _extract_char_codes."""

    def test_simple_font_single_byte(self):
        """Simple fonts extract one code per byte."""
        new_pdf()
        s = pikepdf.String(b"ABC")
        codes = _extract_char_codes(s, is_cid=False)
        assert codes == {65, 66, 67}

    def test_cidfont_two_byte(self):
        """CIDFonts extract two-byte big-endian codes."""
        new_pdf()
        # 0x0041 = 65, 0x00E4 = 228
        s = pikepdf.String(b"\x00\x41\x00\xe4")
        codes = _extract_char_codes(s, is_cid=True)
        assert codes == {0x0041, 0x00E4}

    def test_cidfont_odd_bytes_ignores_trailing(self):
        """CIDFont with odd byte count ignores trailing byte."""
        new_pdf()
        s = pikepdf.String(b"\x00\x41\xff")
        codes = _extract_char_codes(s, is_cid=True)
        assert codes == {0x0041}

    def test_empty_string(self):
        """Empty string returns empty set."""
        new_pdf()
        s = pikepdf.String(b"")
        assert _extract_char_codes(s, is_cid=False) == set()
        assert _extract_char_codes(s, is_cid=True) == set()


class TestIsCIDFont:
    """Tests for _is_cidfont."""

    def test_type0_is_cidfont(self):
        """Type0 font is identified as CIDFont."""
        font = Dictionary(Subtype=Name.Type0)
        assert _is_cidfont(font) is True

    def test_truetype_is_not_cidfont(self):
        """TrueType font is not a CIDFont."""
        font = Dictionary(Subtype=Name.TrueType)
        assert _is_cidfont(font) is False

    def test_type1_is_not_cidfont(self):
        """Type1 font is not a CIDFont."""
        font = Dictionary(Subtype=Name.Type1)
        assert _is_cidfont(font) is False

    def test_no_subtype(self):
        """Font without Subtype is not a CIDFont."""
        font = Dictionary()
        assert _is_cidfont(font) is False


class TestCollectFontUsage:
    """Tests for collect_font_usage."""

    def test_tj_operator(self):
        """Collects codes from Tj operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (Hello) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        # Font should have usage for H, e, l, o
        assert len(usage) == 1
        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("H"), ord("e"), ord("l"), ord("o")}
        assert usage[objgen] == expected

    def test_tj_array_operator(self):
        """Collects codes from TJ (array) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # TJ array: [(AB) -100 (CD)]
        content = b"BT /F1 12 Tf [(AB) -100 (CD)] TJ ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("A"), ord("B"), ord("C"), ord("D")}
        assert usage[objgen] == expected

    def test_single_quote_operator(self):
        """Collects codes from ' (single-quote) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (XY) ' ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {ord("X"), ord("Y")}

    def test_double_quote_operator(self):
        """Collects codes from " (double-quote) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # " operator: aw ac string
        content = b'BT /F1 12 Tf 1 2 (ZW) " ET'
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {ord("Z"), ord("W")}

    def test_cidfont_two_byte_decoding(self):
        """CIDFont codes are decoded as 2-byte big-endian."""
        pdf = new_pdf()

        # Build a Type0 CIDFont structure
        desc_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/TestCJK"),
            )
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([desc_font]),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # CID content: two 2-byte codes: 0x0041 and 0x4E2D
        content = b"BT /F1 12 Tf <00414E2D> Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {0x0041, 0x4E2D}

    def test_multi_page_aggregation(self):
        """Usage from multiple pages is aggregated for same font."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/SharedFont"),
        )
        font_obj = pdf.make_indirect(font)

        # Page 1 uses "AB"
        font_dict1 = Dictionary(F1=font_obj)
        _make_page_with_content(pdf, b"BT /F1 12 Tf (AB) Tj ET", font_dict1)

        # Page 2 uses "CD"
        font_dict2 = Dictionary(F1=font_obj)
        _make_page_with_content(pdf, b"BT /F1 12 Tf (CD) Tj ET", font_dict2)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("A"), ord("B"), ord("C"), ord("D")}
        assert usage[objgen] == expected

    def test_multiple_fonts(self):
        """Tracks usage separately for different fonts."""
        pdf = new_pdf()

        font1 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font1"),
        )
        font1_obj = pdf.make_indirect(font1)

        font2 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font2"),
        )
        font2_obj = pdf.make_indirect(font2)

        font_dict = Dictionary(F1=font1_obj, F2=font2_obj)
        content = b"BT /F1 12 Tf (AB) Tj /F2 10 Tf (XY) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        assert usage[font1_obj.objgen] == {ord("A"), ord("B")}
        assert usage[font2_obj.objgen] == {ord("X"), ord("Y")}

    def test_form_xobject(self):
        """Collects usage from Form XObject content streams."""
        pdf = new_pdf()

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/NestedFont"),
        )
        nested_font_obj = pdf.make_indirect(nested_font)

        form_xobj = pdf.make_stream(b"BT /F2 10 Tf (XO) Tj ET")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 200, 200])
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F2=nested_font_obj),
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/PageFont"),
        )
        page_font_obj = pdf.make_indirect(page_font)

        resources = Dictionary(
            Font=Dictionary(F1=page_font_obj),
            XObject=Dictionary(Form1=pdf.make_indirect(form_xobj)),
        )
        content = b"BT /F1 12 Tf (PG) Tj ET /Form1 Do"
        _make_page_with_content(pdf, content, Dictionary(), resources)

        usage = collect_font_usage(pdf)

        assert page_font_obj.objgen in usage
        assert usage[page_font_obj.objgen] == {ord("P"), ord("G")}
        assert nested_font_obj.objgen in usage
        assert usage[nested_font_obj.objgen] == {ord("X"), ord("O")}

    def test_no_content_stream(self):
        """Handles pages without content streams gracefully."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        usage = collect_font_usage(pdf)
        assert usage == {}

    def test_no_text_operators(self):
        """Pages without text operators produce no usage."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # Content with graphics but no text
        content = b"100 200 m 300 400 l S"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)
        assert usage == {}

    def test_direct_font_object_skipped(self):
        """Direct font objects (objgen 0,0) are not tracked."""
        pdf = new_pdf()

        # Direct font (not make_indirect)
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/DirectFont"),
        )
        font_dict = Dictionary(F1=font)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)
        # Direct objects have objgen (0,0) and are skipped
        assert usage == {}
