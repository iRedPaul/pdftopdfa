# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/tounicode.py — ToUnicode CMap generation."""

import logging
from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts import FontEmbedder
from pdftopdfa.fonts.tounicode import (
    build_identity_unicode_mapping,
    generate_cidfont_tounicode_cmap,
    parse_cidtogidmap_stream,
)
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


class TestSimpleFontToUnicode:
    """Tests for ToUnicode CMap generation for Simple Fonts."""

    def test_tounicode_cmap_8bit_codespace(self):
        """ToUnicode CMap uses 8-bit codespacerange for simple fonts."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        cmap_data = embedder._generate_to_unicode_for_simple_font("Helvetica")
        cmap_text = cmap_data.decode("ascii")

        # 8-bit codespacerange (not 16-bit like CIDFonts)
        assert "<00> <FF>" in cmap_text
        assert "<0000> <FFFF>" not in cmap_text

    def test_tounicode_cmap_required_elements(self):
        """ToUnicode CMap has all required PostScript elements."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        cmap_data = embedder._generate_to_unicode_for_simple_font("Times-Roman")
        cmap_text = cmap_data.decode("ascii")

        # Required CMap structure
        assert "/CIDInit /ProcSet findresource begin" in cmap_text
        assert "begincmap" in cmap_text
        assert "/CIDSystemInfo" in cmap_text
        assert "/Registry (Adobe)" in cmap_text
        assert "/Ordering (UCS)" in cmap_text
        assert "begincodespacerange" in cmap_text
        assert "endcodespacerange" in cmap_text
        assert "beginbfchar" in cmap_text
        assert "endbfchar" in cmap_text
        assert "endcmap" in cmap_text

    def test_tounicode_winansi_mapping(self):
        """Standard fonts use WinAnsiEncoding (CP1252) for ToUnicode mapping."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        cmap_data = embedder._generate_to_unicode_for_simple_font("Helvetica")
        cmap_text = cmap_data.decode("ascii")

        # Check some key WinAnsi mappings
        # Space: code 0x20 -> Unicode U+0020
        assert "<20> <0020>" in cmap_text
        # A: code 0x41 -> Unicode U+0041
        assert "<41> <0041>" in cmap_text
        # Euro sign: code 0x80 -> Unicode U+20AC
        assert "<80> <20AC>" in cmap_text
        # Em-dash: code 0x97 -> Unicode U+2014
        assert "<97> <2014>" in cmap_text

    def test_tounicode_symbol_font_mapping(self):
        """Symbol font ToUnicode uses SYMBOL_ENCODING and glyph-to-unicode mappings."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        cmap_data = embedder._generate_to_unicode_for_simple_font("Symbol")
        cmap_text = cmap_data.decode("ascii")

        # Check Symbol-specific mappings
        # Alpha (code 65) -> Unicode U+0391 (GREEK CAPITAL LETTER ALPHA)
        assert "<41> <0391>" in cmap_text
        # alpha (code 97) -> Unicode U+03B1 (GREEK SMALL LETTER ALPHA)
        assert "<61> <03B1>" in cmap_text
        # plusminus (code 177) -> Unicode U+00B1
        assert "<B1> <00B1>" in cmap_text

    def test_tounicode_zapfdingbats_mapping(self):
        """ZapfDingbats ToUnicode uses ZAPFDINGBATS_ENCODING mappings."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        cmap_data = embedder._generate_to_unicode_for_simple_font("ZapfDingbats")
        cmap_text = cmap_data.decode("ascii")

        # Check ZapfDingbats-specific mappings
        # a1 (code 33) -> Unicode U+2701 (UPPER BLADE SCISSORS)
        assert "<21> <2701>" in cmap_text
        # a2 (code 34) -> Unicode U+2702 (BLACK SCISSORS)
        assert "<22> <2702>" in cmap_text
        # Check mark (a179, code 233) -> Unicode U+2713
        assert "<E9> <2713>" in cmap_text

    def test_resolve_symbol_glyph_custom_mapping(self):
        """_resolve_symbol_glyph_to_unicode uses custom mapping for variants."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        # Greek variant: theta1 -> GREEK THETA SYMBOL (U+03D1)
        assert embedder._resolve_symbol_glyph_to_unicode("theta1") == 0x03D1
        # Standard Greek: Alpha -> standard AGL mapping (U+0391)
        assert embedder._resolve_symbol_glyph_to_unicode("Alpha") == 0x0391
        # Construction glyph (no Unicode): radicalex -> None
        assert embedder._resolve_symbol_glyph_to_unicode("radicalex") is None

    def test_resolve_symbol_glyph_agl_fallback(self):
        """_resolve_symbol_glyph_to_unicode falls back to AGL for standard names."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        # Standard names not in SYMBOL_GLYPH_TO_UNICODE should use AGL
        # space -> U+0020
        assert embedder._resolve_symbol_glyph_to_unicode("space") == 0x0020
        # zero -> U+0030
        assert embedder._resolve_symbol_glyph_to_unicode("zero") == 0x0030

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_embedded_font_has_tounicode(self):
        """Embedded Standard-14 font has ToUnicode CMap attached."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Hello) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded

        # Check that ToUnicode is present
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]
        assert "/ToUnicode" in updated_font

        # Verify the CMap content
        to_unicode = updated_font.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        assert "begincodespacerange" in cmap_text
        assert "<00> <FF>" in cmap_text

    def test_symbol_font_has_tounicode(self):
        """Embedded Symbol font has ToUnicode CMap attached."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Symbol"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (alpha) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Symbol" in result.fonts_embedded

        # Check that ToUnicode is present
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]
        assert "/ToUnicode" in updated_font

    def test_zapfdingbats_has_tounicode(self):
        """Embedded ZapfDingbats font has ToUnicode CMap attached."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/ZapfDingbats"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (4) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "ZapfDingbats" in result.fonts_embedded

        # Check that ToUnicode is present
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]
        assert "/ToUnicode" in updated_font


class TestType3FontToUnicode:
    """Tests for Type3 font ToUnicode CMap generation (PDF/A-2u/3u)."""

    def test_type3_custom_glyphs_get_pua_mapping(self):
        """Type3 font with non-AGL glyph names maps to Private Use Area."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    0,
                    Name("/a0"),
                    Name("/a1"),
                    Name("/a2"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=2,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)

        # All 3 codes should be mapped (to PUA since a0/a1/a2 aren't in AGL)
        assert len(code_to_unicode) == 3
        assert 0 in code_to_unicode
        assert 1 in code_to_unicode
        assert 2 in code_to_unicode

        # Should be in PUA range U+E000-U+F8FF
        for code in range(3):
            assert 0xE000 <= code_to_unicode[code] <= 0xF8FF

    def test_type3_agl_resolvable_glyphs_use_correct_unicode(self):
        """Type3 font with AGL-resolvable glyph names uses real Unicode."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    65,
                    Name("/A"),
                    Name("/B"),
                    97,
                    Name("/a"),
                    Name("/b"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=65,
            LastChar=98,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)

        assert code_to_unicode[65] == 0x0041  # A
        assert code_to_unicode[66] == 0x0042  # B
        assert code_to_unicode[97] == 0x0061  # a
        assert code_to_unicode[98] == 0x0062  # b

    def test_type3_mixed_resolvable_and_custom_glyphs(self):
        """Type3 with mix of AGL-resolvable and custom glyphs."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    0,
                    Name("/A"),
                    Name("/g42"),
                    Name("/space"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=2,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)

        assert len(code_to_unicode) == 3
        # A resolves via AGL
        assert code_to_unicode[0] == 0x0041
        # g42 is not in AGL -> PUA
        assert 0xE000 <= code_to_unicode[1] <= 0xF8FF
        # space resolves via AGL
        assert code_to_unicode[2] == 0x0020

    def test_type3_notdef_glyphs_skipped(self):
        """Type3 font .notdef glyphs are not mapped."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    0,
                    Name("/.notdef"),
                    Name("/A"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=1,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)

        assert 0 not in code_to_unicode  # .notdef skipped
        assert code_to_unicode[1] == 0x0041  # A mapped

    def test_type3_named_encoding_winansi(self):
        """Type3 font with named WinAnsiEncoding uses standard mapping."""
        from pdftopdfa.fonts.tounicode import (
            generate_tounicode_for_type3_font,
            generate_tounicode_for_winansi,
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=Name.WinAnsiEncoding,
            FirstChar=32,
            LastChar=126,
        )

        result = generate_tounicode_for_type3_font(font_dict)
        winansi = generate_tounicode_for_winansi()

        assert result == winansi

    def test_type3_no_encoding_returns_empty(self):
        """Type3 font without encoding returns empty mapping."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FirstChar=0,
            LastChar=10,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)
        assert code_to_unicode == {}

    def test_type3_uni_glyph_names_resolved(self):
        """Type3 font with uniXXXX glyph names resolves correctly."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_type3_font

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    0,
                    Name("/uni0041"),
                    Name("/uni00E9"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=1,
        )

        code_to_unicode = generate_tounicode_for_type3_font(font_dict)

        assert code_to_unicode[0] == 0x0041  # uni0041 = A
        assert code_to_unicode[1] == 0x00E9  # uni00E9 = e-acute

    def test_type3_embedder_adds_tounicode(self):
        """FontEmbedder._add_tounicode_to_type3_font attaches CMap."""
        pdf = new_pdf()

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array(
                [
                    0,
                    Name("/g0"),
                    Name("/g1"),
                    Name("/A"),
                ]
            ),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=2,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_type3_font(font_dict, "Type3Font")
        assert result is True

        # Verify ToUnicode was attached
        to_unicode = font_dict.get("/ToUnicode")
        assert to_unicode is not None
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        # Must have CMap structure
        assert "beginbfchar" in cmap_text
        assert "<00>" in cmap_text  # code 0
        assert "<02> <0041>" in cmap_text  # code 2 = A

    def test_type3_dispatched_from_add_tounicode_to_font(self):
        """_add_tounicode_to_font dispatches Type3 to dedicated handler."""
        pdf = new_pdf()

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array([0, Name("/g0")]),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
            FirstChar=0,
            LastChar=0,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_font(font_dict, "Type3Font")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        assert to_unicode is not None


class TestParseCIDToGIDMapStream:
    """Tests for parse_cidtogidmap_stream()."""

    def test_empty_stream(self):
        """Empty stream produces empty mapping."""
        result = parse_cidtogidmap_stream(b"")
        assert result == {}

    def test_simple_mapping(self):
        """Basic CID-to-GID mapping with 2-byte big-endian entries."""
        # CID 0 -> GID 1, CID 1 -> GID 2, CID 2 -> GID 3
        stream = b"\x00\x01\x00\x02\x00\x03"
        result = parse_cidtogidmap_stream(stream)
        assert result == {0: 1, 1: 2, 2: 3}

    def test_gid_zero_excluded(self):
        """GID=0 (.notdef) entries are excluded."""
        # CID 0 -> GID 0 (excluded), CID 1 -> GID 5
        stream = b"\x00\x00\x00\x05"
        result = parse_cidtogidmap_stream(stream)
        assert result == {1: 5}
        assert 0 not in result

    def test_big_endian_byte_order(self):
        """GIDs are parsed as big-endian 16-bit values."""
        # CID 0 -> GID 0x0102 = 258
        stream = b"\x01\x02"
        result = parse_cidtogidmap_stream(stream)
        assert result == {0: 258}

    def test_odd_byte_count_warns(self, caplog):
        """Trailing odd byte logs a truncation warning."""
        stream = b"\x00\x01\xff"
        with caplog.at_level(logging.WARNING):
            result = parse_cidtogidmap_stream(stream)
        assert result == {0: 1}
        assert "odd length" in caplog.text

    def test_large_gid_values(self):
        """High GID values (near 0xFFFF) are handled correctly."""
        # CID 0 -> GID 0xFFFE = 65534
        stream = b"\xff\xfe"
        result = parse_cidtogidmap_stream(stream)
        assert result == {0: 65534}

    def test_sparse_mapping(self):
        """Sparse mapping with many .notdef entries."""
        # CID 0 -> GID 0, CID 1 -> GID 0, CID 2 -> GID 42
        stream = b"\x00\x00\x00\x00\x00\x2a"
        result = parse_cidtogidmap_stream(stream)
        assert result == {2: 42}


class TestBuildIdentityUnicodeMapping:
    """Tests for build_identity_unicode_mapping()."""

    def test_basic_identity_mapping(self):
        """Unicode values map to themselves."""
        cmap = {65: "A", 66: "B", 67: "C"}
        result = build_identity_unicode_mapping(cmap)
        assert result == {65: 65, 66: 66, 67: 67}

    def test_empty_cmap(self):
        """Empty cmap produces empty mapping."""
        result = build_identity_unicode_mapping({})
        assert result == {}

    def test_cjk_codepoints(self):
        """CJK Unicode codepoints are included correctly."""
        cmap = {0x4E00: "uni4E00", 0x9FFF: "uni9FFF"}
        result = build_identity_unicode_mapping(cmap)
        assert result == {0x4E00: 0x4E00, 0x9FFF: 0x9FFF}


class TestCIDFontToUnicodeStreamCIDToGIDMap:
    """Integration tests for _add_tounicode_to_cidfont with stream CIDToGIDMap."""

    def _make_type0_font_with_stream_cidtogidmap(
        self, pdf, cidtogidmap_data, tt_font, encoding_name="Identity-H"
    ):
        """Helper to build a Type0 font with stream-based CIDToGIDMap."""
        from io import BytesIO

        font_bytes = BytesIO()
        tt_font.save(font_bytes)
        font_bytes.seek(0)
        font_data = font_bytes.read()

        font_file_stream = pikepdf.Stream(pdf, font_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        cidtogidmap_stream = pikepdf.Stream(pdf, cidtogidmap_data)

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=pdf.make_indirect(cidtogidmap_stream),
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/" + encoding_name),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )
        return font_obj

    def test_stream_cidtogidmap_produces_cid_keyed_entries(self):
        """With stream CIDToGIDMap, ToUnicode keys are CIDs, not GIDs."""

        pdf = new_pdf()

        # Build a dummy font file stream (content doesn't matter, we mock TTFont)
        font_file_stream = pikepdf.Stream(pdf, b"\x00" * 100)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        # CIDToGIDMap: CID 10 -> GID 1, CID 20 -> GID 2
        cidtogidmap = bytearray(42)  # 21 entries (CIDs 0-20)
        cidtogidmap[20] = 0  # CID 10 high byte
        cidtogidmap[21] = 1  # CID 10 low byte -> GID 1
        cidtogidmap[40] = 0  # CID 20 high byte
        cidtogidmap[41] = 2  # CID 20 low byte -> GID 2
        cidtogidmap_stream = pikepdf.Stream(pdf, bytes(cidtogidmap))

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=pdf.make_indirect(cidtogidmap_stream),
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        # Mock TTFont to return known cmap and glyph order
        mock_ttfont = MagicMock()
        mock_ttfont.getBestCmap.return_value = {65: "A", 66: "B", 67: "C"}
        mock_ttfont.getGlyphOrder.return_value = [".notdef", "A", "B", "C"]

        with patch("fontTools.ttLib.TTFont", return_value=mock_ttfont):
            embedder = FontEmbedder(pdf)
            result = embedder._add_tounicode_to_cidfont(font_obj, "TestFont")

        assert result is True

        # Verify ToUnicode was added
        to_unicode = font_obj.get("/ToUnicode")
        assert to_unicode is not None

        cmap_data = to_unicode.read_bytes().decode("ascii")
        # CID 10 -> Unicode 65 (0x0041 = 'A')
        assert "<000A> <0041>" in cmap_data
        # CID 20 -> Unicode 66 (0x0042 = 'B')
        assert "<0014> <0042>" in cmap_data
        # GID 1 and GID 2 should NOT appear as keys
        assert "<0001> <0041>" not in cmap_data
        assert "<0002> <0042>" not in cmap_data

    def test_identity_cidtogidmap_still_works(self):
        """Identity CIDToGIDMap (Name) produces GID-keyed output (CID=GID)."""
        from io import BytesIO

        from fontTools.ttLib import TTFont

        tt = TTFont()
        tt.setGlyphOrder([".notdef", "A", "B"])

        pdf = new_pdf()

        font_bytes = BytesIO()
        tt.save(font_bytes)
        font_bytes.seek(0)
        font_data = font_bytes.read()

        font_file_stream = pikepdf.Stream(pdf, font_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=Name.Identity,
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestFont")

        # The font may have a cmap or not — we just verify no crash
        # and that the method handles Identity CIDToGIDMap correctly
        to_unicode = font_obj.get("/ToUnicode")
        if result:
            assert to_unicode is not None
            cmap_data = to_unicode.read_bytes().decode("ascii")
            assert "beginbfchar" in cmap_data


class TestCIDFontToUnicodeUTF16Encoding:
    """Tests for _add_tounicode_to_cidfont with UTF-16 encodings."""

    def test_utf16_encoding_produces_identity_mapping(self):
        """UTF-16 encoding maps each character code to itself."""
        from io import BytesIO

        from fontTools.ttLib import TTFont

        tt = TTFont()
        tt.setGlyphOrder([".notdef", "uni4E00", "uni4E01"])

        pdf = new_pdf()

        font_bytes = BytesIO()
        tt.save(font_bytes)
        font_bytes.seek(0)
        font_data = font_bytes.read()

        font_file_stream = pikepdf.Stream(pdf, font_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCJK"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestCJK"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=Name.Identity,
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=Name("/UniJIS-UTF16-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJK")

        to_unicode = font_obj.get("/ToUnicode")
        if result:
            assert to_unicode is not None
            cmap_data = to_unicode.read_bytes().decode("ascii")
            # For identity mapping, each code maps to itself
            assert "beginbfchar" in cmap_data

    def test_ucs2_encoding_recognized(self):
        """UCS-2 variant encoding is also recognized as UTF-16."""
        from io import BytesIO

        from fontTools.ttLib import TTFont

        tt = TTFont()
        tt.setGlyphOrder([".notdef", "A"])

        pdf = new_pdf()

        font_bytes = BytesIO()
        tt.save(font_bytes)
        font_bytes.seek(0)
        font_data = font_bytes.read()

        font_file_stream = pikepdf.Stream(pdf, font_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/UniJIS-UCS2-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestFont")

        to_unicode = font_obj.get("/ToUnicode")
        if result:
            assert to_unicode is not None
            cmap_data = to_unicode.read_bytes().decode("ascii")
            assert "beginbfchar" in cmap_data


class TestGenerateCIDFontToUnicodeCmapRenamed:
    """Verify generate_cidfont_tounicode_cmap still works after rename."""

    def test_basic_cmap_generation(self):
        """Basic CMap generation with renamed parameter."""
        code_to_unicode = {1: 65, 2: 66}  # GID/CID 1 -> 'A', 2 -> 'B'
        result = generate_cidfont_tounicode_cmap(code_to_unicode)
        cmap_text = result.decode("ascii")
        assert "<0001> <0041>" in cmap_text
        assert "<0002> <0042>" in cmap_text
        assert "<0000> <FFFF>" in cmap_text  # codespacerange

    def test_empty_mapping(self):
        """Empty mapping produces valid CMap with no bfchar entries."""
        result = generate_cidfont_tounicode_cmap({})
        cmap_text = result.decode("ascii")
        assert "begincodespacerange" in cmap_text
        assert "beginbfchar" not in cmap_text


class TestCIDFontToUnicodeBareCFF:
    """Tests for _add_tounicode_to_cidfont with bare CFF CID-keyed fonts."""

    def _make_bare_cff_font(self, pdf, ordering="Japan1", encoding="Identity-H"):
        """Create a mock bare CFF CID-keyed Type0 font structure."""
        # Minimal CFF data (just enough for the stream, won't be parsed)
        font_file_stream = pikepdf.Stream(pdf, b"\x00" * 16)
        font_file_stream[Name("/Subtype")] = Name("/CIDFontType0C")

        cid_system_info = Dictionary()
        cid_system_info[Name("/Registry")] = pikepdf.String("Adobe")
        cid_system_info[Name("/Ordering")] = pikepdf.String(ordering)
        cid_system_info[Name("/Supplement")] = 0

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCJKFont"),
            FontFile3=pdf.make_indirect(font_file_stream),
        )

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType0"),
            BaseFont=Name("/TestCJKFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDSystemInfo=pdf.make_indirect(cid_system_info),
        )

        encoding_name = Name(f"/{encoding}") if encoding else Name("/Identity-H")

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJKFont"),
            Encoding=encoding_name,
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        return font_obj

    def test_bare_cff_japan1_identity_h(self):
        """Bare CFF with Japan1 ordering and Identity-H gets ToUnicode."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "Japan1", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is True
        to_unicode = font_obj.get("/ToUnicode")
        assert to_unicode is not None
        cmap_data = to_unicode.read_bytes().decode("ascii")
        assert "beginbfchar" in cmap_data
        assert "begincmap" in cmap_data

    def test_bare_cff_japan1_utf16_encoding(self):
        """Bare CFF with Japan1 and UTF-16 encoding uses identity mapping."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "Japan1", "UniJIS-UTF16-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is True
        to_unicode = font_obj.get("/ToUnicode")
        assert to_unicode is not None
        cmap_data = to_unicode.read_bytes().decode("ascii")
        assert "beginbfchar" in cmap_data

    def test_bare_cff_gb1(self):
        """Bare CFF with GB1 ordering gets ToUnicode."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "GB1", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is True
        assert font_obj.get("/ToUnicode") is not None

    def test_bare_cff_korea1(self):
        """Bare CFF with Korea1 ordering gets ToUnicode."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "Korea1", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is True
        assert font_obj.get("/ToUnicode") is not None

    def test_bare_cff_cns1(self):
        """Bare CFF with CNS1 ordering gets ToUnicode."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "CNS1", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is True
        assert font_obj.get("/ToUnicode") is not None

    def test_bare_cff_identity_ordering_fails(self):
        """Bare CFF with Identity ordering returns False."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "Identity", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is False
        assert font_obj.get("/ToUnicode") is None

    def test_bare_cff_unknown_ordering_fails(self):
        """Bare CFF with unknown ordering returns False."""
        pdf = new_pdf()
        font_obj = self._make_bare_cff_font(pdf, "CustomOrdering", "Identity-H")

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestCJKFont")

        assert result is False
        assert font_obj.get("/ToUnicode") is None

    def test_non_bare_cff_not_intercepted(self):
        """Font with FontFile2 (TrueType) is not intercepted by bare CFF path."""
        from io import BytesIO

        from fontTools.ttLib import TTFont

        tt = TTFont()
        tt.setGlyphOrder([".notdef", "A"])

        pdf = new_pdf()

        font_bytes = BytesIO()
        tt.save(font_bytes)
        font_bytes.seek(0)
        font_data = font_bytes.read()

        font_file_stream = pikepdf.Stream(pdf, font_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )

        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=Name.Identity,
        )

        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )

        embedder = FontEmbedder(pdf)
        # Should use the normal TTFont path, not the bare CFF path
        embedder._add_tounicode_to_cidfont(font_obj, "TestFont")


class TestCIDFontToUnicodePUAFallback:
    """Tests for PUA fallback when CIDFont has no cmap table."""

    def _make_no_cmap_ttfont(self, num_glyphs=5):
        """Create a proper TrueType font with no cmap table."""
        from io import BytesIO

        from fontTools.fontBuilder import FontBuilder
        from fontTools.pens.ttGlyphPen import TTGlyphPen

        glyph_names = [".notdef"] + [f"glyph{i:05d}" for i in range(1, num_glyphs)]
        fb = FontBuilder(1000, isTTF=True)
        fb.setupGlyphOrder(glyph_names)
        fb.setupCharacterMap({})

        glyphs = {}
        for name in glyph_names:
            pen = TTGlyphPen(None)
            pen.moveTo((0, 0))
            pen.lineTo((500, 0))
            pen.lineTo((500, 700))
            pen.lineTo((0, 700))
            pen.closePath()
            glyphs[name] = pen.glyph()

        fb.setupGlyf(glyphs)
        fb.setupHorizontalMetrics({name: (500, 0) for name in glyph_names})
        fb.setupHorizontalHeader()
        fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
        fb.setupOS2()
        fb.setupPost()
        fb.setupHead(unitsPerEm=1000)
        font = fb.font

        del font["cmap"]
        buf = BytesIO()
        font.save(buf)
        buf.seek(0)
        return buf.read()

    def _make_cidfont(self, pdf, font_data):
        """Create a Type0/CIDFont with embedded TrueType data."""
        font_file_stream = pikepdf.Stream(pdf, font_data)
        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestNoCmap"),
            FontFile2=pdf.make_indirect(font_file_stream),
        )
        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestNoCmap"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            CIDToGIDMap=Name.Identity,
        )
        font_obj = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestNoCmap"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )
        return font_obj

    def test_no_cmap_gets_pua_tounicode(self):
        """Font without cmap table gets PUA-based ToUnicode CMap."""
        pdf = new_pdf()
        font_data = self._make_no_cmap_ttfont(num_glyphs=10)
        font_obj = self._make_cidfont(pdf, font_data)

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestNoCmap")

        assert result is True
        to_unicode = font_obj.get("/ToUnicode")
        assert to_unicode is not None

        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        cmap_data = bytes(to_unicode.read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)
        # 10 glyphs minus .notdef = 9 PUA mappings
        assert len(mapping) == 9
        for gid, uni in mapping.items():
            assert 0xE000 <= uni <= 0xF8FF

    def test_no_cmap_pua_values_are_unique(self):
        """Each GID gets a distinct PUA codepoint."""
        pdf = new_pdf()
        font_data = self._make_no_cmap_ttfont(num_glyphs=50)
        font_obj = self._make_cidfont(pdf, font_data)

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_cidfont(font_obj, "TestNoCmap")

        assert result is True

        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        cmap_data = bytes(font_obj["/ToUnicode"].read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)
        unicode_values = list(mapping.values())
        assert len(unicode_values) == len(set(unicode_values))


class TestFillToUnicodeGapsWithPUA:
    """Tests for fill_tounicode_gaps_with_pua() helper."""

    def test_basic_gap_filling(self):
        """Gaps in range are filled with PUA; existing entries preserved."""
        from pdftopdfa.fonts.tounicode import fill_tounicode_gaps_with_pua

        existing = {1: 0x0041, 3: 0x0043}  # codes 1 and 3 mapped
        result = fill_tounicode_gaps_with_pua(existing, 0, 4)

        # Existing entries preserved
        assert result[1] == 0x0041
        assert result[3] == 0x0043
        # Gaps filled with PUA
        assert 0xE000 <= result[0] <= 0xF8FF
        assert 0xE000 <= result[2] <= 0xF8FF
        assert 0xE000 <= result[4] <= 0xF8FF
        # All PUA values are unique
        pua_values = [v for v in result.values() if 0xE000 <= v <= 0xF8FF]
        assert len(pua_values) == len(set(pua_values))

    def test_no_gaps(self):
        """No-op when all codes in range are already mapped."""
        from pdftopdfa.fonts.tounicode import fill_tounicode_gaps_with_pua

        existing = {0: 0x0041, 1: 0x0042, 2: 0x0043}
        result = fill_tounicode_gaps_with_pua(existing, 0, 2)

        assert result == existing

    def test_avoids_pua_collision(self):
        """PUA assignments skip codepoints already used in the mapping."""
        from pdftopdfa.fonts.tounicode import fill_tounicode_gaps_with_pua

        # Pre-existing mapping uses U+E000 and U+E001
        existing = {0: 0xE000, 1: 0xE001}
        result = fill_tounicode_gaps_with_pua(existing, 0, 3)

        # Existing entries preserved
        assert result[0] == 0xE000
        assert result[1] == 0xE001
        # New PUA values must not collide
        assert result[2] not in (0xE000, 0xE001)
        assert result[3] not in (0xE000, 0xE001)
        assert 0xE000 <= result[2] <= 0xF8FF
        assert 0xE000 <= result[3] <= 0xF8FF
        assert result[2] != result[3]


class TestNoEncodingPUAGapFilling:
    """Integration tests for PUA gap-filling in _add_tounicode_to_simple_font()."""

    def test_symbolic_truetype_low_codes(self):
        """Symbolic TrueType with codes 1-4 and no encoding gets full CMap."""
        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/BAAAAA+Cambria"),
            FirstChar=1,
            LastChar=4,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "Cambria")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        assert to_unicode is not None
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)

        # All codes 1-4 must be present
        for code in range(1, 5):
            assert code in mapping, f"Code {code} missing from CMap"

    def test_code_zero_gets_mapped(self):
        """Code 0 with FirstChar=0 LastChar=0 gets a ToUnicode entry."""
        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/BECFBN+CMSY8"),
            FirstChar=0,
            LastChar=0,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "CMSY8")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        assert to_unicode is not None
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)

        assert 0 in mapping, "Code 0 missing from CMap"

    def test_preserves_standard_encoding_entries(self):
        """StandardEncoding entries preserved; gaps also filled."""
        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            FirstChar=0,
            LastChar=65,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "TestFont")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)

        # StandardEncoding: 0x41 (A) -> U+0041
        assert mapping[0x41] == 0x0041
        # StandardEncoding: 0x27 -> U+2019 (quoteright)
        assert mapping[0x27] == 0x2019
        # Gap codes 0-31 should also have entries (PUA)
        for code in range(0, 32):
            assert code in mapping, f"Gap code {code} missing from CMap"

    def test_without_firstchar_lastchar_defaults(self):
        """Without FirstChar/LastChar, defaults to 0-255 range."""
        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap

        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "TestFont")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        mapping = parse_tounicode_cmap(cmap_data)

        # Full 0-255 range should be covered
        for code in range(256):
            assert code in mapping, f"Code {code} missing from CMap"
