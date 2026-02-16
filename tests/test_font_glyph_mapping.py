# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/glyph_mapping.py — glyph name mapping."""

from unittest.mock import MagicMock

from conftest import new_pdf

from pdftopdfa.fonts import FontEmbedder


class TestGlyphMapping:
    """Tests for glyph name mapping module."""

    def test_zapfdingbats_mapping_has_required_glyphs(self):
        """ZapfDingbats mapping contains all a1-a206 glyphs."""
        from pdftopdfa.fonts.glyph_mapping import ZAPFDINGBATS_GLYPH_TO_UNICODE

        # Check space
        assert "space" in ZAPFDINGBATS_GLYPH_TO_UNICODE
        assert ZAPFDINGBATS_GLYPH_TO_UNICODE["space"] == 0x0020

        # Check some key dingbat glyphs
        assert "a1" in ZAPFDINGBATS_GLYPH_TO_UNICODE  # Scissors
        assert "a2" in ZAPFDINGBATS_GLYPH_TO_UNICODE
        assert "a206" in ZAPFDINGBATS_GLYPH_TO_UNICODE

        # Check specific Unicode mappings
        assert ZAPFDINGBATS_GLYPH_TO_UNICODE["a1"] == 0x2701  # UPPER BLADE SCISSORS
        assert ZAPFDINGBATS_GLYPH_TO_UNICODE["a2"] == 0x2702  # BLACK SCISSORS

    def test_symbol_mapping_has_construction_glyphs(self):
        """Symbol mapping handles construction glyphs correctly."""
        from pdftopdfa.fonts.glyph_mapping import SYMBOL_GLYPH_TO_UNICODE

        # Construction glyphs should map to None (no Unicode equivalent)
        assert "radicalex" in SYMBOL_GLYPH_TO_UNICODE
        assert SYMBOL_GLYPH_TO_UNICODE["radicalex"] is None

        assert "arrowvertex" in SYMBOL_GLYPH_TO_UNICODE
        assert SYMBOL_GLYPH_TO_UNICODE["arrowvertex"] is None

    def test_symbol_mapping_has_greek_variants(self):
        """Symbol mapping includes Greek variant forms."""
        from pdftopdfa.fonts.glyph_mapping import SYMBOL_GLYPH_TO_UNICODE

        # Greek variant glyphs
        assert "theta1" in SYMBOL_GLYPH_TO_UNICODE
        assert SYMBOL_GLYPH_TO_UNICODE["theta1"] == 0x03D1  # GREEK THETA SYMBOL

        assert "phi1" in SYMBOL_GLYPH_TO_UNICODE
        assert SYMBOL_GLYPH_TO_UNICODE["phi1"] == 0x03D5  # GREEK PHI SYMBOL

    def test_resolve_glyph_name_direct_lookup(self):
        """resolve_glyph_name finds glyph by direct name lookup."""
        from pdftopdfa.fonts.glyph_mapping import resolve_glyph_name

        # Mock cmap and hmtx with Adobe glyph names
        cmap = {0x0041: "A"}
        hmtx = {"A": (600, 0), "Alpha": (700, 0), ".notdef": (500, 0)}

        # Direct lookup should work
        result = resolve_glyph_name("Alpha", cmap, hmtx)
        assert result == "Alpha"

    def test_resolve_glyph_name_custom_mapping(self):
        """resolve_glyph_name uses custom mapping for ZapfDingbats."""
        from pdftopdfa.fonts.glyph_mapping import (
            ZAPFDINGBATS_GLYPH_TO_UNICODE,
            resolve_glyph_name,
        )

        # Simulate Noto Sans Symbols 2 cmap (uses Unicode names)
        cmap = {0x2701: "uni2701", 0x2702: "uni2702"}
        hmtx = {"uni2701": (600, 0), "uni2702": (600, 0), ".notdef": (500, 0)}

        # a1 should resolve to uni2701 via custom mapping
        result = resolve_glyph_name("a1", cmap, hmtx, ZAPFDINGBATS_GLYPH_TO_UNICODE)
        assert result == "uni2701"

    def test_resolve_glyph_name_agl_fallback(self):
        """resolve_glyph_name falls back to AGL2UV for standard names."""
        from pdftopdfa.fonts.glyph_mapping import resolve_glyph_name

        # Simulate font with Unicode-based glyph names
        cmap = {0x0041: "A", 0x0391: "uni0391"}  # A and Alpha in Unicode
        hmtx = {"A": (600, 0), "uni0391": (700, 0), ".notdef": (500, 0)}

        # "Alpha" is in AGL2UV -> 0x0391 -> "uni0391"
        result = resolve_glyph_name("Alpha", cmap, hmtx)
        assert result == "uni0391"

    def test_resolve_glyph_name_not_found(self):
        """resolve_glyph_name returns None for unmapped glyphs."""
        from pdftopdfa.fonts.glyph_mapping import resolve_glyph_name

        cmap = {0x0041: "A"}
        hmtx = {"A": (600, 0), ".notdef": (500, 0)}

        # Non-existent glyph
        result = resolve_glyph_name("nonexistent_glyph_xyz", cmap, hmtx)
        assert result is None

    def test_resolve_glyph_name_construction_glyph(self):
        """resolve_glyph_name returns None for construction glyphs."""
        from pdftopdfa.fonts.glyph_mapping import (
            SYMBOL_GLYPH_TO_UNICODE,
            resolve_glyph_name,
        )

        cmap = {}
        hmtx = {".notdef": (500, 0)}

        # radicalex maps to None in custom mapping
        result = resolve_glyph_name("radicalex", cmap, hmtx, SYMBOL_GLYPH_TO_UNICODE)
        assert result is None

    def test_extract_widths_for_encoding_with_mapping(self):
        """extract_widths_for_encoding correctly uses glyph mapping."""
        from pdftopdfa.fonts.glyph_mapping import ZAPFDINGBATS_GLYPH_TO_UNICODE

        # Create mock TTFont
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "head": MagicMock(unitsPerEm=1000),
                "hmtx": MagicMock(
                    metrics={
                        ".notdef": (500, 0),
                        "space": (250, 0),
                        "uni2701": (800, 0),  # Noto uses Unicode name
                        "uni2702": (750, 0),
                    }
                ),
            }[key]
        )

        # Mock cmap with Unicode-based names
        mock_cmap = {
            0x0020: "space",
            0x2701: "uni2701",
            0x2702: "uni2702",
        }
        mock_tt_font.getBestCmap = MagicMock(return_value=mock_cmap)

        # ZapfDingbats encoding
        test_encoding = {
            32: "space",
            33: "a1",  # -> 0x2701 -> uni2701
            34: "a2",  # -> 0x2702 -> uni2702
        }

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        widths = embedder._metrics.extract_widths_for_encoding(
            mock_tt_font, test_encoding, ZAPFDINGBATS_GLYPH_TO_UNICODE
        )

        assert len(widths) == 256
        assert widths[32] == 250  # space
        assert widths[33] == 800  # a1 -> uni2701
        assert widths[34] == 750  # a2 -> uni2702
        assert widths[0] == 500  # unmapped -> .notdef
