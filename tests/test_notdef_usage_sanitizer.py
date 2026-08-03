# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for .notdef usage sanitizer (ISO 19005-2, Rule 6.2.11.8)."""

import io
import struct

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, String

from pdftopdfa.fonts.tounicode import parse_tounicode_cmap_sequences
from pdftopdfa.sanitizers import notdef_usage as notdef_usage_mod
from pdftopdfa.sanitizers.notdef_usage import (
    _NotdefCodes,
    sanitize_notdef_usage,
)


def _make_ttfont_bytes(glyph_names, cmap=None):
    """Creates minimal TrueType font data containing given glyph names.

    Uses fontTools to build a minimal font with .notdef + the given names.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    all_names = [".notdef"] + list(glyph_names)
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(all_names)
    fb.setupCharacterMap(cmap or {})
    # Use empty Glyph objects (zero-contour) instead of dicts
    fb.setupGlyf({name: Glyph() for name in all_names})
    fb.setupHorizontalMetrics({name: (500, 0) for name in all_names})
    fb.setupHorizontalHeader()
    fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)
    buf = io.BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


def _make_symbolic_ttfont_bytes(offset=0xF000):
    """Create a TrueType font whose byte 0x41 selects .notdef."""
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.tables._c_m_a_p import cmap_format_4, table__c_m_a_p

    tt_font = TTFont(io.BytesIO(_make_ttfont_bytes(["B"], {0x42: "B"})))
    cmap = table__c_m_a_p()
    cmap.tableVersion = 0
    symbol = cmap_format_4(4)
    symbol.platformID = 3
    symbol.platEncID = 0
    symbol.language = 0
    symbol.cmap = {
        offset + 0x41: ".notdef",
        offset + 0x42: "B",
    }
    unicode = cmap_format_4(4)
    unicode.platformID = 3
    unicode.platEncID = 1
    unicode.language = 0
    unicode.cmap = {0x42: "B"}
    cmap.tables = [symbol, unicode]
    tt_font["cmap"] = cmap
    output = io.BytesIO()
    tt_font.save(output)
    tt_font.close()
    return output.getvalue()


def _make_simple_font(
    pdf,
    first_char=32,
    last_char=114,
    base_font="TestFont",
    glyphs=("A", "B", "C"),
):
    """Creates a simple TrueType font dict with an embedded font program."""
    font_stream = pdf.make_stream(_make_ttfont_bytes(list(glyphs)))
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name(f"/{base_font}"),
        Flags=32,
        FontFile2=font_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name(f"/{base_font}"),
        FirstChar=first_char,
        LastChar=last_char,
        Encoding=Name.WinAnsiEncoding,
        FontDescriptor=fd,
    )
    return font


def _make_page_with_font_and_content(pdf, font_dict, content_bytes):
    """Creates a page with a font resource and content stream."""
    stream = pdf.make_stream(content_bytes)
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
            Contents=stream,
        )
    )
    pdf.pages.append(page)
    return stream


class TestSimpleFontCodeOutsideRange:
    """Tests for character codes outside [FirstChar, LastChar].

    The [FirstChar, LastChar] range only affects widths — the encoding
    still selects the glyph for out-of-range codes, so such codes must
    only be stripped when the encoded glyph is actually missing.
    """

    def test_out_of_range_code_with_glyph_present_kept(self):
        """Code 0xFC beyond LastChar=126 stays when udieresis exists."""
        pdf = new_pdf()
        font = _make_simple_font(
            pdf, first_char=32, last_char=126, glyphs=("A", "udieresis")
        )
        content = b"BT /F1 12 Tf (A\xfc) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A\xfc"

    def test_unencoded_code_outside_range_removed(self):
        """Code 0 (no WinAnsi mapping, glyph missing) is removed."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        # Content: select font, then show string with \x00 + 'A'
        content = b"BT /F1 12 Tf (\x00A) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        # Verify the content stream was modified
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        # Only 'A' (0x41) should remain
        assert bytes(tj_ops[0].operands[0]) == b"A"

    def test_simple_font_code_inside_range_kept(self):
        """Code 65 ('A') with FirstChar=32 is preserved."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        content = b"BT /F1 12 Tf (ABC) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0


class TestFullyStrippedOperator:
    """Tests for operators that become empty after filtering."""

    def test_tj_string_fully_stripped_removed(self):
        """Tj with only .notdef codes removes the entire operator."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        # Only \x00 in the string — fully .notdef
        content = b"BT /F1 12 Tf (\x00) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        # The Tj operator should be completely removed
        assert len(tj_ops) == 0

    def test_quote_operator_fully_stripped_replaced_with_tstar(self):
        """' with only .notdef codes keeps the implicit line advance."""
        pdf = new_pdf()
        font = _make_simple_font(pdf)
        content = b"BT /F1 12 Tf 14 TL (\x00) ' ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        ops = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "'" not in ops
        assert "T*" in ops

    def test_quote_operator_partial_strip_preserves_line_advance(self):
        """' with removed text preserves its line and glyph advances."""
        pdf = new_pdf()
        font = _make_simple_font(pdf)
        content = b"BT /F1 12 Tf 14 TL (\x00A) ' ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        text_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(text_ops) == 1
        assert bytes(text_ops[0].operands[0]) == b"A"
        assert any(
            isinstance(i, pikepdf.ContentStreamInstruction) and str(i.operator) == "T*"
            for i in instructions
        )
        assert any(
            isinstance(i, pikepdf.ContentStreamInstruction) and str(i.operator) == "TJ"
            for i in instructions
        )

    def test_doublequote_fully_stripped_keeps_spacing_and_tstar(self):
        """Emptied " keeps its word/char spacing and line advance."""
        pdf = new_pdf()
        font = _make_simple_font(pdf)
        content = b'BT /F1 12 Tf 14 TL 2 3 (\x00) " ET'
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = [
            i
            for i in list(pikepdf.parse_content_stream(stream))
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        ops = [str(i.operator) for i in instructions]
        assert '"' not in ops
        # The side effects of " must be preserved: aw Tw, ac Tc, T*
        tw_ops = [i for i in instructions if str(i.operator) == "Tw"]
        tc_ops = [i for i in instructions if str(i.operator) == "Tc"]
        assert len(tw_ops) == 1
        assert int(tw_ops[0].operands[0]) == 2
        assert len(tc_ops) == 1
        assert int(tc_ops[0].operands[0]) == 3
        assert "T*" in ops


class TestTJArray:
    """Tests for TJ array operator filtering."""

    def test_tj_array_partial_strip(self):
        """TJ with mixed codes: .notdef codes removed, rest kept."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        # Build content stream manually with TJ array
        # [(\x00A) -10 (BC)] TJ
        # \x00 should be removed from first string, BC stays
        content = b"BT /F1 12 Tf [(\x00A) -10 (BC)] TJ ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "TJ"
        ]
        assert len(tj_ops) == 1
        arr = tj_ops[0].operands[0]
        # First string should be just "A"
        strings = [bytes(elem) for elem in arr if isinstance(elem, String)]
        assert b"A" in strings
        assert b"BC" in strings

    def test_symbolic_program_notdef_preserves_tj_advance(self):
        """A program byte-cmap .notdef is removed with its exact advance."""
        pdf = new_pdf()
        font_data = _make_symbolic_ttfont_bytes()
        descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestSymbol"),
            Flags=4,
            MissingWidth=500,
            FontFile2=pdf.make_stream(font_data),
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestSymbol"),
            FirstChar=0x41,
            LastChar=0x42,
            Widths=Array([600, 700]),
            FontDescriptor=descriptor,
        )
        stream = _make_page_with_font_and_content(
            pdf,
            font,
            b"BT /F1 10 Tf [<41> 25 <42>] TJ ET",
        )

        result = sanitize_notdef_usage(pdf)

        tj = next(
            instruction
            for instruction in pikepdf.parse_content_stream(stream)
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and str(instruction.operator) == "TJ"
        )
        assert result["notdef_usage_fixed"] == 1
        assert [
            bytes(item) if isinstance(item, String) else float(item)
            for item in tj.operands[0]
        ] == [-600.0, 25.0, b"B"]


class TestNoChanges:
    """Tests for PDFs that don't need .notdef usage fixes."""

    def test_no_changes_returns_zero(self):
        """PDF without .notdef references returns 0."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        # All chars are in range (65='A', 66='B', 67='C')
        content = b"BT /F1 12 Tf (ABC) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0

    def test_empty_pdf(self, sample_pdf_obj: Pdf):
        """PDF without text operators returns 0."""
        result = sanitize_notdef_usage(sample_pdf_obj)
        assert result["notdef_usage_fixed"] == 0

    def test_q_restore_restores_previous_font(self):
        """Restoring graphics state also restores the active font."""
        pdf = new_pdf()

        # Font1 provides the endash glyph (WinAnsi 0x96), Font2 does not
        font1 = _make_simple_font(pdf, base_font="Font1", glyphs=("A", "B", "endash"))
        font2 = _make_simple_font(pdf, base_font="Font2", glyphs=("A", "B"))

        stream = pdf.make_stream(
            b"BT /F1 12 Tf ET q BT /F2 12 Tf (A\x96B) Tj ET Q BT (A\x96B) Tj ET"
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font1, F2=font2)),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 3
        assert b"".join(bytes(op.operands[0]) for op in tj_ops[:2]) == b"AB"
        assert bytes(tj_ops[2].operands[0]) == b"A\x96B"


class TestCIDFont:
    """Tests for CIDFont (Type0) .notdef handling."""

    def test_cidfont_cid_zero_removed(self):
        """CID 0 with Identity CIDToGIDMap is removed."""
        pdf = new_pdf()

        # Build a Type0 font with Identity CIDToGIDMap
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # CID 0 = \x00\x00 (.notdef), CID 65 = \x00\x41 (valid)
        content = b"BT /F1 12 Tf <00000041> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        # Only CID 65 (\x00\x41) should remain
        assert bytes(tj_ops[0].operands[0]) == b"\x00\x41"

    def test_invisible_ocr_cid_zero_is_remapped_without_text_loss(self):
        """Rendering mode 3 text gets a real CID and keeps its Unicode."""
        from fontTools.ttLib import TTFont
        from fontTools.ttLib.tables._c_m_a_p import (
            cmap_format_4,
            table__c_m_a_p,
        )

        source_font = TTFont(io.BytesIO(_make_ttfont_bytes(["A"], {0x41: "A"})))
        cmap_table = table__c_m_a_p()
        cmap_table.tableVersion = 0
        cmap_table.tables = []
        for platform_id, encoding_id, mapping in (
            (3, 10, {0x42: "A"}),
            (3, 1, {0x41: "A"}),
        ):
            subtable = cmap_format_4(4)
            subtable.platformID = platform_id
            subtable.platEncID = encoding_id
            subtable.language = 0
            subtable.cmap = mapping
            cmap_table.tables.append(subtable)
        source_font["cmap"] = cmap_table
        buffer = io.BytesIO()
        source_font.save(buffer)
        source_font.close()

        pdf = new_pdf()
        font_stream = pdf.make_stream(buffer.getvalue())
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
            FontDescriptor=Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestCIDFont"),
                FontFile2=font_stream,
            ),
            DW=500,
        )
        tounicode = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            b"1 beginbfchar\n<0000> <0041>\nendbfchar\n"
            b"endcmap\n"
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
            ToUnicode=tounicode,
        )
        stream = _make_page_with_font_and_content(
            pdf,
            type0_font,
            b"BT /F1 12 Tf 3 Tr <0000> Tj ET",
        )

        result = sanitize_notdef_usage(pdf)

        shown = [
            item
            for instruction in pikepdf.parse_content_stream(stream)
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            for item in instruction.operands
            if isinstance(item, String)
        ]
        assert result["notdef_usage_fixed"] == 1
        assert [bytes(item) for item in shown] == [b"\x00\x01"]
        assert parse_tounicode_cmap_sequences(tounicode.read_bytes())[b"\x00\x01"] == (
            0x41,
        )

    def test_invisible_unmapped_cid_zero_preserves_advance_without_text(self):
        """Unmapped invisible .notdef keeps its advance without invented text."""
        pdf = new_pdf()
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType0,
            BaseFont=Name("/TestCIDFont"),
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Japan1"),
                Supplement=4,
            ),
            DW=1000,
        )
        tounicode = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            b"1 beginbfchar\n<0001> <0020>\nendbfchar\n"
            b"endcmap\n"
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
            ToUnicode=tounicode,
        )
        stream = _make_page_with_font_and_content(
            pdf,
            type0_font,
            b"BT /F1 1 Tf 3 Tr -0.006 Tc <0000> Tj ET",
        )

        result = sanitize_notdef_usage(pdf)

        instructions = list(pikepdf.parse_content_stream(stream))
        text_operators = [
            instruction
            for instruction in instructions
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and str(instruction.operator) in {"Tj", "TJ"}
        ]
        mappings = parse_tounicode_cmap_sequences(tounicode.read_bytes())
        assert result["notdef_usage_fixed"] == 1
        assert len(text_operators) == 1
        assert str(text_operators[0].operator) == "TJ"
        assert [float(item) for item in text_operators[0].operands[0]] == [-994.0]
        assert b"\x00\x00" not in mappings
        assert mappings[b"\x00\x01"] == (0x20,)

    def test_invisible_unmapped_cid_in_tj_array_keeps_existing_adjustment(self):
        """TJ cleanup combines preserved advance with existing positioning."""
        pdf = new_pdf()
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType0,
            BaseFont=Name("/TestCIDFont"),
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
            DW=500,
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )
        stream = _make_page_with_font_and_content(
            pdf,
            type0_font,
            b"BT /F1 10 Tf 3 Tr 0.02 Tc [<0000> 25] TJ ET",
        )

        result = sanitize_notdef_usage(pdf)

        instructions = list(pikepdf.parse_content_stream(stream))
        tj = next(
            instruction
            for instruction in instructions
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and str(instruction.operator) == "TJ"
        )
        assert result["notdef_usage_fixed"] == 1
        assert [float(item) for item in tj.operands[0]] == [-477.0]

    def test_embedded_cmap_character_codes_are_mapped_to_cids(self):
        """Custom CMap codes are translated before .notdef filtering."""
        pdf = new_pdf()
        encoding = pdf.make_stream(
            b"""
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 begincidchar
<3F00> 0
<0041> 65
endcidchar
"""
        )
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=encoding,
            DescendantFonts=Array([cidfont]),
        )
        stream = pdf.make_stream(b"BT /F1 12 Tf <3F000041> Tj ET")
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(Font=Dictionary(F1=type0_font)),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        tj = next(
            instruction
            for instruction in pikepdf.parse_content_stream(stream)
            if (
                isinstance(instruction, pikepdf.ContentStreamInstruction)
                and str(instruction.operator) == "Tj"
            )
        )
        assert bytes(tj.operands[0]) == b"\x00A"

    def test_cidfont_type0_keeps_valid_nonsequential_cids(self, monkeypatch):
        """CIDFontType0 keeps valid charset CIDs even when they are high values."""
        pdf = new_pdf()

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType0,
            BaseFont=Name("/TestCIDFontType0"),
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
            FontDescriptor=Dictionary(),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFontType0"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        monkeypatch.setattr(
            notdef_usage_mod,
            "_get_cidfonttype0_valid_cids",
            lambda _cidfont: frozenset({0, 3, 5, 107, 124, 172, 316}),
        )

        content = b"BT /F1 12 Tf <000000AC006B0003007C013C0005> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=type0_font)),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert (
            bytes(tj_ops[0].operands[0]) == b"\x00\xac\x00k\x00\x03\x00|\x01<\x00\x05"
        )

    def test_cidfont_stream_gid_zero_removed(self):
        """CIDs mapping to GID 0 via stream CIDToGIDMap are removed."""
        pdf = new_pdf()

        # Build CIDToGIDMap stream: CID 0 → GID 0, CID 1 → GID 0, CID 2 → GID 42
        map_data = struct.pack(">HHH", 0, 0, 42)
        cidtogidmap_stream = pdf.make_stream(map_data)

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=cidtogidmap_stream,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # CID 0 (\x00\x00) and CID 1 (\x00\x01) → GID 0 → .notdef
        # CID 2 (\x00\x02) → GID 42 → valid
        content = b"BT /F1 12 Tf <000000010002> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        # Only CID 2 (\x00\x02) should remain
        assert bytes(tj_ops[0].operands[0]) == b"\x00\x02"

    def test_cidfont_high_byte_zero_preserved(self):
        """CID 87 (\\x00\\x57) must NOT be treated as .notdef."""
        pdf = new_pdf()

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # CID 87 = \x00\x57, which is NOT .notdef (only CID 0 is)
        content = b"BT /F1 12 Tf <0057> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0


class TestCIDOddLengthString:
    """Tests for malformed CID strings with an odd byte count."""

    @staticmethod
    def _make_identity_type0_font():
        """Type0 font with Identity CIDToGIDMap — only CID 0 is .notdef."""
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        return Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

    def _make_page(self, pdf, content):
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=self._make_identity_type0_font()),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)
        return stream

    def test_incomplete_character_code_is_removed(self):
        """A trailing byte outside the two-byte codespace is .notdef."""
        pdf = new_pdf()
        stream = self._make_page(pdf, b"BT /F1 12 Tf (#) Tj ET")

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        text_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) in {"Tj", "TJ"}
        ]
        assert len(text_ops) == 1
        assert str(text_ops[0].operator) == "TJ"
        assert list(text_ops[0].operands[0]) == [-1000]

    def test_odd_length_string_removes_notdef_and_trailing_byte(self):
        """Both explicit and incomplete .notdef codes are removed."""
        pdf = new_pdf()
        # CID 0 (.notdef) + CID 87 (valid) + trailing lone byte
        stream = self._make_page(pdf, b"BT /F1 12 Tf <0000005700> Tj ET")

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        text_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) in {"Tj", "TJ"}
        ]
        shown = b"".join(
            bytes(item)
            for instruction in text_ops
            for operand in instruction.operands
            for item in (operand if isinstance(operand, pikepdf.Array) else [operand])
            if isinstance(item, pikepdf.String)
        )
        adjustments = [
            float(item)
            for instruction in text_ops
            for operand in instruction.operands
            for item in (operand if isinstance(operand, pikepdf.Array) else [operand])
            if isinstance(item, (int, float))
        ]
        assert shown == b"\x00\x57"
        assert adjustments == [-1000.0, -1000.0]


class TestMultipleFonts:
    """Tests for pages with multiple fonts."""

    def test_multiple_fonts_on_page(self):
        """Different fonts with different notdef code sets."""
        pdf = new_pdf()
        # F1 lacks the endash glyph (WinAnsi 0x96) — code 0x96 is .notdef
        font1 = _make_simple_font(pdf, base_font="Font1", glyphs=("A", "B"))
        # F2 provides endash — code 0x96 maps to a real glyph
        font2 = _make_simple_font(pdf, base_font="Font2", glyphs=("A", "B", "endash"))

        # F1: \x96 should be removed; F2: \x96 should be kept
        content = b"BT /F1 12 Tf (\x96A) Tj /F2 12 Tf (\x96B) Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=font1, F2=font2),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        # Only the first Tj (F1) should be fixed
        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 2
        # First Tj: only 'A' remains
        assert bytes(tj_ops[0].operands[0]) == b"A"
        # Second Tj: \x96B stays (F2 provides the endash glyph)
        assert bytes(tj_ops[1].operands[0]) == b"\x96B"


class TestFormXObject:
    """Tests for .notdef codes in Form XObject content streams."""

    def test_form_xobject_content_stream_fixed(self):
        """Removes .notdef codes from Form XObject content."""
        pdf = new_pdf()

        font = _make_simple_font(pdf, first_char=32, last_char=114)

        form_stream = pdf.make_stream(b"BT /F1 12 Tf (\x00A) Tj ET")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F1=font),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(form_stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"


class TestAPStream:
    """Tests for .notdef codes in Annotation Appearance Streams."""

    def test_ap_stream_content_fixed(self):
        """Removes .notdef codes from annotation AP stream."""
        pdf = new_pdf()

        font = _make_simple_font(pdf, first_char=32, last_char=114)

        ap_stream = pdf.make_stream(b"BT /F1 12 Tf (\x00A) Tj ET")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 20])
        ap_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F1=font),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 200, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.pages[0].Annots = Array([annot])

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(ap_stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"


class TestIntegration:
    """Integration tests with sanitize_for_pdfa."""

    def test_sanitize_for_pdfa_includes_key(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa returns notdef_usage_fixed key."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "notdef_usage_fixed" in result
        assert result["notdef_usage_fixed"] == 0

    def test_sanitize_for_pdfa_fixes_notdef_usage(self):
        """sanitize_for_pdfa actually fixes .notdef references."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        content = b"BT /F1 12 Tf (\x00A) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_for_pdfa(pdf, "2b")

        assert result["notdef_usage_fixed"] == 1


class TestNotdefCodesClass:
    """Unit tests for the _NotdefCodes helper class."""

    def test_explicit_contains(self):
        """Explicit codes are detected by 'in'."""
        codes = _NotdefCodes(frozenset({0, 5, 10}))
        assert 0 in codes
        assert 5 in codes
        assert 10 in codes
        assert 1 not in codes
        assert 255 not in codes

    def test_max_valid_code_contains(self):
        """Codes above max_valid_code are detected."""
        codes = _NotdefCodes(frozenset({0}), max_valid_code=99)
        assert 0 in codes
        assert 99 not in codes
        assert 100 in codes
        assert 1000 in codes

    def test_combined_explicit_and_threshold(self):
        """Both explicit codes and threshold work together."""
        codes = _NotdefCodes(frozenset({0, 3}), max_valid_code=50)
        assert 0 in codes
        assert 3 in codes
        assert 50 not in codes
        assert 51 in codes
        assert 25 not in codes

    def test_bool_empty(self):
        """Empty _NotdefCodes is falsy."""
        assert not _NotdefCodes()
        assert not _NotdefCodes(frozenset())
        assert not _NotdefCodes(frozenset(), None)

    def test_bool_explicit(self):
        """_NotdefCodes with explicit codes is truthy."""
        assert _NotdefCodes(frozenset({0}))

    def test_bool_threshold(self):
        """_NotdefCodes with only max_valid_code is truthy."""
        assert _NotdefCodes(frozenset(), max_valid_code=10)


class TestSimpleFontGlyphMissing:
    """Tests for codes in [FirstChar, LastChar] whose glyph is missing."""

    def test_code_in_range_but_glyph_missing_stripped(self):
        """Code within range but glyph absent from font is stripped."""
        pdf = new_pdf()

        # Font with only 'A' glyph (code 65 in WinAnsi)
        font_data = _make_ttfont_bytes(["A"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=67,
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=fd,
        )

        # 'A' (65) exists, 'B' (66) and 'C' (67) are missing from font
        content = b"BT /F1 12 Tf (ABC) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        # Only 'A' should remain
        assert bytes(tj_ops[0].operands[0]) == b"A"

    def test_code_in_range_glyph_present_kept(self):
        """Code within range with glyph present is kept."""
        pdf = new_pdf()

        # Font with A, B, C glyphs
        font_data = _make_ttfont_bytes(["A", "B", "C"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=67,
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (ABC) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0

    def test_winansi_superscript_two_is_not_stripped(self):
        """WinAnsi code 178 stays when the font provides twosuperior."""
        pdf = new_pdf()

        font_data = _make_ttfont_bytes(["A", "twosuperior"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=178,
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (A\xb2) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A\xb2"

    def test_space_kept_when_subset_glyph_is_renamed_but_mapped_in_cmap(self):
        """Subset glyph renames must not make valid spaces look like .notdef."""
        pdf = new_pdf()

        font_data = _make_ttfont_bytes(
            ["glyph00001", "A"],
            cmap={0x0020: "glyph00001", 0x0041: "A"},
        )
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=32,
            LastChar=65,
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (A A) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A A"

    def test_nbspace_and_soft_hyphen_kept_with_winansi_glyph_names(self):
        """WinAnsi NBSP and soft hyphen stay when space and hyphen exist."""
        pdf = new_pdf()

        font_data = _make_ttfont_bytes(
            ["A", "space", "hyphen"],
            cmap={0x0041: "A", 0x0020: "space", 0x002D: "hyphen"},
        )
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=173,
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (A\xa0\xadA) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A\xa0\xadA"

    def test_no_font_descriptor_leaves_text_untouched(self):
        """Without an embedded font program nothing is stripped.

        Glyph presence cannot be verified, and codes outside
        [FirstChar, LastChar] still select glyphs via the encoding,
        so no code may be treated as .notdef.
        """
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=32,
            LastChar=114,
            Encoding=Name.WinAnsiEncoding,
        )
        content = b"BT /F1 12 Tf (\x00A) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0


class TestCIDFontBeyondNumGlyphs:
    """Tests for CIDs >= numGlyphs with Identity CIDToGIDMap."""

    def test_cid_beyond_num_glyphs_stripped(self):
        """CID >= numGlyphs with Identity mapping is stripped."""
        pdf = new_pdf()

        # Create a font with exactly 3 glyphs: .notdef + 2 real glyphs
        font_data = _make_ttfont_bytes(["glyph1", "glyph2"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCIDFont"),
            FontFile2=font_stream,
        )

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            FontDescriptor=fd,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # numGlyphs = 3 (.notdef + glyph1 + glyph2)
        # CID 0 → .notdef, CID 1 → glyph1 (valid),
        # CID 2 → glyph2 (valid), CID 3 → beyond (invalid)
        # Content: CID 1 (valid) + CID 3 (beyond numGlyphs)
        content = b"BT /F1 12 Tf <00010003> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        # Only CID 1 should remain
        assert bytes(tj_ops[0].operands[0]) == b"\x00\x01"

    def test_cid_within_num_glyphs_kept(self):
        """CIDs within font glyph count are preserved."""
        pdf = new_pdf()

        font_data = _make_ttfont_bytes(["glyph1", "glyph2"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCIDFont"),
            FontFile2=font_stream,
        )

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=Name.Identity,
            FontDescriptor=fd,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # CID 1 and CID 2 — both within numGlyphs=3
        content = b"BT /F1 12 Tf <00010002> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0

    def test_stream_cidtogidmap_gid_beyond_num_glyphs(self):
        """CID mapping to GID >= numGlyphs via stream is stripped."""
        pdf = new_pdf()

        # Font with 3 glyphs: .notdef, glyph1, glyph2
        font_data = _make_ttfont_bytes(["glyph1", "glyph2"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCIDFont"),
            FontFile2=font_stream,
        )

        # CIDToGIDMap stream: CID 0 → GID 1, CID 1 → GID 99 (beyond)
        map_data = struct.pack(">HH", 1, 99)
        cidtogidmap_stream = pdf.make_stream(map_data)

        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name("/TestCIDFont"),
            CIDToGIDMap=cidtogidmap_stream,
            FontDescriptor=fd,
            CIDSystemInfo=Dictionary(
                Registry=String(b"Adobe"),
                Ordering=String(b"Identity"),
                Supplement=0,
            ),
        )
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )

        # CID 0 is always .notdef; CID 1 → GID 99 is beyond numGlyphs=3.
        content = b"BT /F1 12 Tf <00000001> Tj ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type0_font),
                ),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 0
        numeric_tj = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "TJ"
        ]
        assert len(numeric_tj) == 1
        assert float(numeric_tj[0].operands[0][0]) == -2000


class TestTilingPattern:
    """Tests for .notdef codes in Tiling Pattern content streams."""

    def test_tiling_pattern_notdef_stripped(self):
        """Removes .notdef codes from Tiling Pattern content."""
        pdf = new_pdf()

        font = _make_simple_font(pdf, first_char=32, last_char=114)

        pattern_stream = pdf.make_stream(b"BT /F1 12 Tf (\x00A) Tj ET")
        pattern_stream[Name("/PatternType")] = 1
        pattern_stream[Name("/PaintType")] = 1
        pattern_stream[Name("/TilingType")] = 1
        pattern_stream[Name("/BBox")] = Array([0, 0, 100, 100])
        pattern_stream[Name("/XStep")] = 100
        pattern_stream[Name("/YStep")] = 100
        pattern_stream[Name("/Resources")] = Dictionary(
            Font=Dictionary(F1=font),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P1=pattern_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(pattern_stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"

    def test_tiling_pattern_no_notdef_unchanged(self):
        """Pattern without .notdef codes is not modified."""
        pdf = new_pdf()

        font = _make_simple_font(pdf, first_char=32, last_char=114)

        pattern_stream = pdf.make_stream(b"BT /F1 12 Tf (AB) Tj ET")
        pattern_stream[Name("/PatternType")] = 1
        pattern_stream[Name("/PaintType")] = 1
        pattern_stream[Name("/TilingType")] = 1
        pattern_stream[Name("/BBox")] = Array([0, 0, 100, 100])
        pattern_stream[Name("/XStep")] = 100
        pattern_stream[Name("/YStep")] = 100
        pattern_stream[Name("/Resources")] = Dictionary(
            Font=Dictionary(F1=font),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P1=pattern_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 0

    def test_nested_pattern_in_form_xobject(self):
        """Removes .notdef from pattern nested inside Form XObject."""
        pdf = new_pdf()

        font = _make_simple_font(pdf, first_char=32, last_char=114)

        pattern_stream = pdf.make_stream(b"BT /F1 12 Tf (\x00A) Tj ET")
        pattern_stream[Name("/PatternType")] = 1
        pattern_stream[Name("/PaintType")] = 1
        pattern_stream[Name("/TilingType")] = 1
        pattern_stream[Name("/BBox")] = Array([0, 0, 100, 100])
        pattern_stream[Name("/XStep")] = 100
        pattern_stream[Name("/YStep")] = 100
        pattern_stream[Name("/Resources")] = Dictionary(
            Font=Dictionary(F1=font),
        )

        form_stream = pdf.make_stream(b"q Q")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Pattern=Dictionary(P1=pattern_stream),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(pattern_stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"


class TestUnmappedEncodingEntry:
    """Tests for codes with no encoding entry (maps to .notdef)."""

    def test_code_with_no_encoding_entry_flagged_as_notdef(self):
        """Character code with None encoding entry is treated as .notdef.

        When _resolve_simple_font_encoding returns a dict without an
        entry for a given code, that code maps to .notdef per PDF spec.
        """
        pdf = new_pdf()

        # Font only has glyph 'A' — not 'B' or 'C'
        font_data = _make_ttfont_bytes(["A"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        # Encoding dict with only Differences: 65→A
        # No BaseEncoding → StandardEncoding base (sparse).
        # Code 66 resolves to 'B' via StandardEncoding, code 67 to 'C'.
        # Neither 'B' nor 'C' exist in the font → .notdef.
        enc = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name("/A")]),
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=67,
            Encoding=enc,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (ABC) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        # B and C are not in the font → stripped
        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"

    def test_explicit_notdef_in_encoding_flagged(self):
        """Glyph name '.notdef' in encoding Differences is flagged."""
        pdf = new_pdf()

        font_data = _make_ttfont_bytes(["A"])
        font_stream = pdf.make_stream(font_data)

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=font_stream,
        )

        # Differences: code 65→A, code 66→.notdef
        enc = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name("/A"), Name("/.notdef")]),
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=65,
            LastChar=66,
            Encoding=enc,
            FontDescriptor=fd,
        )

        content = b"BT /F1 12 Tf (AB) Tj ET"
        stream = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(stream))
        tj_ops = [
            i
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
            and str(i.operator) == "Tj"
        ]
        assert len(tj_ops) == 1
        assert bytes(tj_ops[0].operands[0]) == b"A"
