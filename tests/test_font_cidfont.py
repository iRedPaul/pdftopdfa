# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/cidfont.py — CIDFontBuilder and CJK font index mapping."""

from io import BytesIO

import pytest
from conftest import new_pdf
from font_helpers import _noto_cjk_font_available
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Name

from pdftopdfa.fonts.cidfont import CIDFontBuilder
from pdftopdfa.fonts.metrics import FontMetricsExtractor
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


def _make_test_ttfont(*, glyphs=None, cmap=None, upm=1000):
    """Build a minimal TrueType font for testing.

    Args:
        glyphs: List of glyph names (default: [".notdef", "A", "B", "C"]).
        cmap: Character-to-glyph mapping dict (default: {65: "A", 66: "B", 67: "C"}).
        upm: Units per em (default: 1000).

    Returns:
        Tuple of (font_data bytes, TTFont object).
    """
    if glyphs is None:
        glyphs = [".notdef", "A", "B", "C"]
    if cmap is None:
        cmap = {65: "A", 66: "B", 67: "C"}

    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap(cmap)

    fb.setupGlyf({})
    glyf_table = fb.font["glyf"]
    for gname in glyphs:
        glyf_table[gname] = Glyph()

    metrics = {g: (600, 0) for g in glyphs}
    metrics[".notdef"] = (500, 0)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()

    tt = fb.font
    buf = BytesIO()
    tt.save(buf)
    font_data = buf.getvalue()
    tt.close()

    tt = TTFont(BytesIO(font_data))
    return font_data, tt


class TestCIDFontBuilder:
    """Direct tests for CIDFontBuilder.build_structure()."""

    def test_build_structure_returns_type0_dict(self):
        """build_structure returns a Type0 font dictionary."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            assert result["/Type"] == Name.Font
            assert result["/Subtype"] == Name.Type0
            assert str(result["/BaseFont"]) == "/TestFont"
        finally:
            tt_font.close()

    def test_build_structure_encoding_default_identity_h(self):
        """Default encoding is Identity-H."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            assert str(result["/Encoding"]) == "/Identity-H"
        finally:
            tt_font.close()

    def test_build_structure_encoding_identity_v(self):
        """Vertical encoding Identity-V is applied when requested."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure(
                "TestFont",
                tt_font,
                font_data,
                encoding="Identity-V",
            )

            assert str(result["/Encoding"]) == "/Identity-V"
        finally:
            tt_font.close()

    def test_build_structure_has_descendant_fonts(self):
        """Result has DescendantFonts array with one CIDFontType2 entry."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            desc = result["/DescendantFonts"]
            assert len(desc) == 1
            cid_font = _resolve_indirect(desc[0])
            assert cid_font["/Subtype"] == Name("/CIDFontType2")
            assert str(cid_font["/BaseFont"]) == "/TestFont"
        finally:
            tt_font.close()

    def test_build_structure_has_cidsysteminfo(self):
        """CIDFont descendant has CIDSystemInfo with Adobe/Identity/0."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            csi = cid_font["/CIDSystemInfo"]
            assert str(csi["/Registry"]) == "Adobe"
            assert str(csi["/Ordering"]) == "Identity"
            assert int(csi["/Supplement"]) == 0
        finally:
            tt_font.close()

    def test_build_structure_has_font_descriptor(self):
        """CIDFont descendant has FontDescriptor with required fields."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            fd = _resolve_indirect(cid_font["/FontDescriptor"])
            assert fd["/Type"] == Name.FontDescriptor
            assert str(fd["/FontName"]) == "/TestFont"
            assert "/Flags" in fd
            assert "/FontBBox" in fd
            assert "/Ascent" in fd
            assert "/Descent" in fd
            assert "/CapHeight" in fd
            assert "/StemV" in fd
        finally:
            tt_font.close()

    def test_build_structure_has_fontfile2(self):
        """FontDescriptor contains FontFile2 stream with correct Length1."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            fd = _resolve_indirect(cid_font["/FontDescriptor"])
            ff2 = _resolve_indirect(fd["/FontFile2"])
            assert int(ff2["/Length1"]) == len(font_data)
            embedded = bytes(ff2.read_bytes())
            assert embedded == font_data
        finally:
            tt_font.close()

    def test_build_structure_has_w_array(self):
        """CIDFont has /W array for glyph widths."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            assert "/W" in cid_font
            w = cid_font["/W"]
            assert len(w) > 0
        finally:
            tt_font.close()

    def test_build_structure_has_dw(self):
        """CIDFont has /DW (default width) derived from .notdef."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            dw = int(cid_font["/DW"])
            # .notdef width is 500 at upm=1000 → DW=500
            assert dw == 500
        finally:
            tt_font.close()

    def test_build_structure_dw_scaled_for_nonstandard_upm(self):
        """DW is scaled correctly for non-standard unitsPerEm."""
        font_data, tt_font = _make_test_ttfont(upm=2048)
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            dw = int(cid_font["/DW"])
            # .notdef width 500 at upm=2048 → 500 * 1000/2048 = 244
            assert dw == int(500 * 1000.0 / 2048)
        finally:
            tt_font.close()

    def test_build_structure_has_tounicode(self):
        """Result has ToUnicode CMap stream."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            assert "/ToUnicode" in result
            tu = _resolve_indirect(result["/ToUnicode"])
            cmap_data = bytes(tu.read_bytes()).decode("ascii")
            assert "begincmap" in cmap_data
            assert "endcmap" in cmap_data
        finally:
            tt_font.close()

    def test_build_structure_tounicode_contains_mappings(self):
        """ToUnicode CMap maps GIDs to correct Unicode values."""
        font_data, tt_font = _make_test_ttfont(
            cmap={65: "A", 66: "B"},
        )
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            tu = _resolve_indirect(result["/ToUnicode"])
            cmap_text = bytes(tu.read_bytes()).decode("ascii")
            # GID for 'A' should map to U+0041
            assert "0041" in cmap_text
            # GID for 'B' should map to U+0042
            assert "0042" in cmap_text
        finally:
            tt_font.close()

    def test_build_structure_has_cidtogidmap_identity(self):
        """CIDFont uses /CIDToGIDMap = /Identity."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            assert cid_font["/CIDToGIDMap"] == Name.Identity
        finally:
            tt_font.close()

    def test_build_structure_missing_tables_raises(self):
        """ValueError raised when font lacks head/OS2 tables."""
        font_data, tt_font = _make_test_ttfont()
        # Force-load then remove the head table
        _ = tt_font["head"]
        del tt_font["head"]

        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            with pytest.raises(ValueError, match="missing head/OS2"):
                builder.build_structure("BadFont", tt_font, font_data)
        finally:
            tt_font.close()

    def test_build_structure_nonsymbolic_flags(self):
        """FontDescriptor Flags has Nonsymbolic bit (32) set."""
        font_data, tt_font = _make_test_ttfont()
        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder.build_structure("TestFont", tt_font, font_data)

            cid_font = _resolve_indirect(result["/DescendantFonts"][0])
            fd = _resolve_indirect(cid_font["/FontDescriptor"])
            flags = int(fd["/Flags"])
            # Nonsymbolic bit (32) should be set because is_symbol=False
            assert flags & 32 != 0
        finally:
            tt_font.close()


class TestCIDFontBuilderToUnicode:
    """Tests for CIDFontBuilder._generate_to_unicode_cmap()."""

    def test_empty_cmap_produces_valid_output(self):
        """Font with no cmap table still produces a valid CMap."""
        font_data, tt_font = _make_test_ttfont()
        # Force-load then remove cmap table
        _ = tt_font["cmap"]
        del tt_font["cmap"]

        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder._generate_to_unicode_cmap(tt_font)
            assert isinstance(result, bytes)
            text = result.decode("ascii")
            assert "begincmap" in text
            assert "endcmap" in text
        finally:
            tt_font.close()

    def test_cmap_with_multiple_unicodes_per_gid(self):
        """Only the first Unicode value per GID is stored."""
        font_data, tt_font = _make_test_ttfont(
            glyphs=[".notdef", "A"],
            cmap={65: "A"},
        )

        pdf = new_pdf()
        try:
            builder = CIDFontBuilder(pdf, FontMetricsExtractor())
            result = builder._generate_to_unicode_cmap(tt_font)
            text = result.decode("ascii")
            # GID for 'A' maps to U+0041
            assert "0041" in text
        finally:
            tt_font.close()


class TestConvertWArray:
    """Tests for CIDFontBuilder._convert_w_array_to_pikepdf()."""

    def test_integers_pass_through(self):
        """Integer items in W array are kept as-is."""
        pdf = new_pdf()
        builder = CIDFontBuilder(pdf, FontMetricsExtractor())
        result = builder._convert_w_array_to_pikepdf([1, 2, 3])
        assert result == [1, 2, 3]

    def test_lists_become_arrays(self):
        """List items become pikepdf.Array objects."""
        pdf = new_pdf()
        builder = CIDFontBuilder(pdf, FontMetricsExtractor())
        result = builder._convert_w_array_to_pikepdf([0, [500, 600]])
        assert result[0] == 0
        assert isinstance(result[1], Array)
        assert list(result[1]) == [500, 600]

    def test_empty_input(self):
        """Empty W array returns empty list."""
        pdf = new_pdf()
        builder = CIDFontBuilder(pdf, FontMetricsExtractor())
        result = builder._convert_w_array_to_pikepdf([])
        assert result == []

    def test_mixed_types(self):
        """Mixed ints and lists are correctly converted."""
        pdf = new_pdf()
        builder = CIDFontBuilder(pdf, FontMetricsExtractor())
        result = builder._convert_w_array_to_pikepdf(
            [10, [400, 500, 600], 20, [700]],
        )
        assert result[0] == 10
        assert isinstance(result[1], Array)
        assert result[2] == 20
        assert isinstance(result[3], Array)


class TestCJKFontIndexMapping:
    """Tests for CJK font index selection by ordering."""

    def test_cjk_font_index_values(self):
        """CJK_FONT_INDEX maps known orderings to distinct indices."""
        from pdftopdfa.fonts.constants import CJK_FONT_INDEX

        assert CJK_FONT_INDEX["Japan1"] == 2
        assert CJK_FONT_INDEX["CNS1"] == 1
        assert CJK_FONT_INDEX["GB1"] == 0
        assert CJK_FONT_INDEX["Korea1"] == 3
        assert CJK_FONT_INDEX["Identity"] == 0

    def test_different_orderings_select_different_indices(self):
        """Different CJK scripts map to different TTC indices."""
        from pdftopdfa.fonts.constants import CJK_FONT_INDEX

        indices = {
            CJK_FONT_INDEX["Japan1"],
            CJK_FONT_INDEX["CNS1"],
            CJK_FONT_INDEX["GB1"],
            CJK_FONT_INDEX["Korea1"],
        }
        # GB1 and Identity share index 0, but Japan1/CNS1/Korea1 are distinct
        assert len(indices) == 4

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_loader_selects_font_by_ordering(self):
        """FontLoader.load_cidfont_replacement_by_ordering returns data."""
        from pdftopdfa.fonts.loader import FontLoader

        cache: dict = {}
        loader = FontLoader(cache)

        font_data, tt_font = loader.load_cidfont_replacement_by_ordering("Japan1")
        assert len(font_data) > 0
        assert tt_font is not None
        tt_font.close()

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_loader_unknown_ordering_defaults_to_index_0(self):
        """Unknown ordering falls back to index 0."""
        from pdftopdfa.fonts.loader import FontLoader

        cache: dict = {}
        loader = FontLoader(cache)

        font_data_unknown, _ = loader.load_cidfont_replacement_by_ordering(
            "UnknownScript"
        )
        font_data_gb1, _ = loader.load_cidfont_replacement_by_ordering("GB1")
        # Both should use index 0
        assert font_data_unknown == font_data_gb1
