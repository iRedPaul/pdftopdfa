# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for .notdef usage sanitizer (ISO 19005-2, Rule 6.2.11.8)."""

import io
import struct

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, String

from pdftopdfa.sanitizers.notdef_usage import (
    _NotdefCodes,
    sanitize_notdef_usage,
)


def _make_simple_font(pdf, first_char=32, last_char=114, base_font="TestFont"):
    """Creates a minimal simple TrueType font dictionary."""
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name(f"/{base_font}"),
        FirstChar=first_char,
        LastChar=last_char,
        Encoding=Name.WinAnsiEncoding,
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
    """Tests for character codes outside [FirstChar, LastChar]."""

    def test_simple_font_code_outside_range_removed(self):
        """Code 0 with FirstChar=32 is removed from Tj string."""
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


class TestMultipleFonts:
    """Tests for pages with multiple fonts."""

    def test_multiple_fonts_on_page(self):
        """Different fonts with different notdef code sets."""
        pdf = new_pdf()
        # F1: FirstChar=32, LastChar=114 — code 0 is .notdef
        font1 = _make_simple_font(pdf, first_char=32, last_char=114, base_font="Font1")
        # F2: FirstChar=0, LastChar=255 — no codes are out-of-range
        font2 = _make_simple_font(pdf, first_char=0, last_char=255, base_font="Font2")

        # F1: \x00 should be removed; F2: \x00 should be kept
        content = b"BT /F1 12 Tf (\x00A) Tj /F2 12 Tf (\x00B) Tj ET"
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
        # Second Tj: \x00B stays (F2 has FirstChar=0)
        assert bytes(tj_ops[1].operands[0]) == b"\x00B"


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


def _make_ttfont_bytes(glyph_names):
    """Creates minimal TrueType font data containing given glyph names.

    Uses fontTools to build a minimal font with .notdef + the given names.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    all_names = [".notdef"] + list(glyph_names)
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(all_names)
    fb.setupCharacterMap({})
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

    def test_no_font_descriptor_falls_back(self):
        """Font without FontDescriptor falls back to range-only check."""
        pdf = new_pdf()
        font = _make_simple_font(pdf, first_char=32, last_char=114)
        # Code 0 is outside range → still stripped
        content = b"BT /F1 12 Tf (\x00A) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_notdef_usage(pdf)

        assert result["notdef_usage_fixed"] == 1


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

        # CID 0 → GID 1 (valid), CID 1 → GID 99 (beyond numGlyphs=3)
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
        assert len(tj_ops) == 1
        # Only CID 0 (→ GID 1, valid) should remain
        assert bytes(tj_ops[0].operands[0]) == b"\x00\x00"


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
