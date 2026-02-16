# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/embedder.py — font embedding and FontEmbedder class."""

from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available, _noto_cjk_font_available
from pikepdf import Array, Dictionary, Name

from pdftopdfa.exceptions import FontEmbeddingError
from pdftopdfa.fonts import (
    FONT_REPLACEMENTS,
    EmbeddingResult,
    FontEmbedder,
    check_font_compliance,
)
from pdftopdfa.fonts.analysis import is_font_embedded
from pdftopdfa.fonts.embedder import _UTF16_ENCODING_NAMES, _is_utf16_encoding
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


class TestFontReplacements:
    """Tests for font mapping."""

    def test_helvetica_variants_mapped(self):
        """All Helvetica variants have replacements."""
        assert "Helvetica" in FONT_REPLACEMENTS
        assert "Helvetica-Bold" in FONT_REPLACEMENTS
        assert "Helvetica-Oblique" in FONT_REPLACEMENTS
        assert "Helvetica-BoldOblique" in FONT_REPLACEMENTS

    def test_times_variants_mapped(self):
        """All Times variants have replacements."""
        assert "Times-Roman" in FONT_REPLACEMENTS
        assert "Times-Bold" in FONT_REPLACEMENTS
        assert "Times-Italic" in FONT_REPLACEMENTS
        assert "Times-BoldItalic" in FONT_REPLACEMENTS

    def test_courier_variants_mapped(self):
        """All Courier variants have replacements."""
        assert "Courier" in FONT_REPLACEMENTS
        assert "Courier-Bold" in FONT_REPLACEMENTS
        assert "Courier-Oblique" in FONT_REPLACEMENTS
        assert "Courier-BoldOblique" in FONT_REPLACEMENTS

    def test_liberation_font_names(self):
        """Standard replacement fonts are Liberation fonts."""
        liberation_fonts = [
            v
            for k, v in FONT_REPLACEMENTS.items()
            if k not in ("Symbol", "ZapfDingbats")
        ]
        for replacement in liberation_fonts:
            assert replacement.startswith("Liberation")
            assert replacement.endswith(".ttf")

    def test_symbol_fonts_mapped(self):
        """Symbol and ZapfDingbats have replacements."""
        assert "Symbol" in FONT_REPLACEMENTS
        assert "ZapfDingbats" in FONT_REPLACEMENTS
        assert FONT_REPLACEMENTS["Symbol"] == "STIXTwoMath-Regular.ttf"
        assert FONT_REPLACEMENTS["ZapfDingbats"] == "NotoSansSymbols2-Regular.ttf"


class TestEmbeddingResult:
    """Tests for the EmbeddingResult data class."""

    def test_default_values(self):
        """Default values are empty lists."""
        result = EmbeddingResult()
        assert result.fonts_embedded == []
        assert result.fonts_failed == []
        assert result.fonts_preserved == []
        assert result.warnings == []

    def test_with_values(self):
        """Values can be set."""
        result = EmbeddingResult(
            fonts_embedded=["Helvetica", "Times-Roman"],
            fonts_failed=["Symbol"],
            fonts_preserved=["Arial"],
            warnings=["Test warning"],
        )
        assert len(result.fonts_embedded) == 2
        assert len(result.fonts_failed) == 1
        assert len(result.fonts_preserved) == 1
        assert len(result.warnings) == 1


class TestFontEmbedder:
    """Tests for the FontEmbedder class."""

    def test_init(self, pdf_with_text_obj):
        """FontEmbedder can be initialized with PDF."""
        embedder = FontEmbedder(pdf_with_text_obj)
        assert embedder.pdf is pdf_with_text_obj
        assert embedder._font_cache == {}

    def test_embed_missing_fonts_returns_result(self, pdf_with_text_obj):
        """embed_missing_fonts returns EmbeddingResult."""
        embedder = FontEmbedder(pdf_with_text_obj)

        # Mock _loader.load_standard14_font to avoid file system access
        with patch.object(embedder._loader, "load_standard14_font") as mock_load:
            # Simulate error when loading
            mock_load.side_effect = FontEmbeddingError("Font not found")

            result = embedder.embed_missing_fonts()

            assert isinstance(result, EmbeddingResult)

    def test_unknown_font_uses_fallback(self):
        """Unknown fonts are embedded using LiberationSans fallback."""
        pdf = new_pdf()

        # Create page with unknown font
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/UnknownFont"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "UnknownFont" in result.fonts_embedded
        assert any("fallback" in w for w in result.warnings)

        # Font should now be embedded
        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    def test_symbol_font_embedding(self):
        """Symbol font is successfully embedded."""
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

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Symbol" in result.fonts_embedded
        assert "Symbol" not in result.fonts_failed

        # Check that font is now embedded
        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    def test_zapfdingbats_font_embedding(self):
        """ZapfDingbats font is successfully embedded."""
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

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "ZapfDingbats" in result.fonts_embedded
        assert "ZapfDingbats" not in result.fonts_failed

        # Check that font is now embedded
        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    def test_cidfont_embedding(self):
        """CIDFont/Type0 is successfully embedded."""
        pdf = new_pdf()

        # Create CIDFont without embedded data
        # DescendantFont without FontDescriptor -> not embedded
        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/MSGothic"),
                # No FontDescriptor = not embedded
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/MSGothic"),
            DescendantFonts=Array([descendant_font]),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        # CIDFont is now automatically embedded
        assert "MSGothic" in result.fonts_embedded
        assert "MSGothic" not in result.fonts_failed


class TestFontMetrics:
    """Tests for font metrics extraction."""

    def test_extract_font_metrics_structure(self):
        """_metrics.extract_metrics returns correct structure."""
        # Create mock TTFont with all required fields for _compute_font_flags
        mock_tables = {
            "head": MagicMock(
                unitsPerEm=1000,
                xMin=-100,
                yMin=-200,
                xMax=800,
                yMax=900,
            ),
            "OS/2": MagicMock(
                sTypoAscender=800,
                sTypoDescender=-200,
                sCapHeight=700,
                sFamilyClass=0,
                fsSelection=0,
            ),
            "post": MagicMock(italicAngle=0, isFixedPitch=0),
        }
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(side_effect=lambda key: mock_tables[key])
        mock_tt_font.__contains__ = MagicMock(return_value=True)
        mock_tt_font.get = MagicMock(side_effect=lambda key: mock_tables.get(key))

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        metrics = embedder._metrics.extract_metrics(mock_tt_font)

        assert "FontBBox" in metrics
        assert "Ascent" in metrics
        assert "Descent" in metrics
        assert "CapHeight" in metrics
        assert "StemV" in metrics
        assert "ItalicAngle" in metrics
        assert "Flags" in metrics

        assert len(metrics["FontBBox"]) == 4
        assert metrics["Flags"] & 32  # Nonsymbolic bit is set


class TestWidthsCalculation:
    """Tests for character width calculation."""

    def test_extract_widths_returns_256_values(self):
        """_metrics.extract_widths returns 256 width values."""
        # Create mock TTFont with minimal structure
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "head": MagicMock(unitsPerEm=1000),
                "hmtx": MagicMock(
                    metrics={
                        ".notdef": (500, 0),
                        "space": (250, 0),
                        "A": (600, 0),
                    }
                ),
            }[key]
        )

        # Mock getBestCmap
        mock_cmap = {32: "space", 65: "A"}
        mock_tt_font.getBestCmap = MagicMock(return_value=mock_cmap)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        widths = embedder._metrics.extract_widths(mock_tt_font)

        assert len(widths) == 256
        assert all(isinstance(w, int) for w in widths)

        # Check specific values
        assert widths[32] == 250  # space
        assert widths[65] == 600  # A


class TestFontEmbedderIntegration:
    """Integration tests for FontEmbedder with real fonts."""

    @pytest.fixture
    def pdf_with_helvetica(self):
        """Creates PDF with non-embedded Helvetica."""
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

        content_stream = pdf.make_stream(b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        yield pdf

    def test_helvetica_not_embedded_initially(self, pdf_with_helvetica):
        """Helvetica is not embedded initially."""
        is_compliant, missing = check_font_compliance(
            pdf_with_helvetica, raise_on_error=False
        )
        assert not is_compliant
        assert "Helvetica" in missing

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_helvetica_embedding(self, pdf_with_helvetica):
        """Helvetica is successfully replaced by LiberationSans."""
        embedder = FontEmbedder(pdf_with_helvetica)
        result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded
        assert result.fonts_failed == []

        # Check that font is now embedded
        is_compliant, missing = check_font_compliance(
            pdf_with_helvetica, raise_on_error=False
        )
        assert is_compliant
        assert missing == []


class TestCIDFontEmbedding:
    """Tests for CIDFont/Type0 embedding."""

    def test_cidfont_embedding_succeeds(self):
        """CIDFont is successfully embedded."""
        pdf = new_pdf()

        # Create CIDFont without embedded data
        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/MSGothic"),
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/MSGothic"),
            DescendantFonts=Array([descendant_font]),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        # CIDFont should now be embedded (no longer in failed)
        assert "MSGothic" in result.fonts_embedded
        assert "MSGothic" not in result.fonts_failed

        # Check that font is now embedded
        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_cidfont_structure_complete(self):
        """Embedded CIDFont has complete structure."""
        pdf = new_pdf()

        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/TestCJKFont"),
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJKFont"),
            DescendantFonts=Array([descendant_font]),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        embedder.embed_missing_fonts()

        # Get the updated font dictionary
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]

        # Check Type0 structure
        assert updated_font.get("/Subtype") == Name.Type0
        assert updated_font.get("/Encoding") == Name("/Identity-H")
        assert "/DescendantFonts" in updated_font
        assert "/ToUnicode" in updated_font

        # Check DescendantFont
        descendants = updated_font.get("/DescendantFonts")
        assert len(descendants) == 1

        cid_font = descendants[0]
        cid_font = _resolve_indirect(cid_font)

        assert cid_font.get("/Subtype") == Name.CIDFontType2
        assert "/CIDSystemInfo" in cid_font
        assert "/FontDescriptor" in cid_font
        assert "/W" in cid_font
        assert "/DW" in cid_font

        # Check FontDescriptor
        font_descriptor = cid_font.get("/FontDescriptor")
        font_descriptor = _resolve_indirect(font_descriptor)

        assert "/FontFile2" in font_descriptor

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_tounicode_cmap_format(self):
        """ToUnicode CMap has correct PostScript format."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        # Load CIDFont
        _, tt_font = embedder._loader.load_cidfont_replacement_by_ordering("Japan1")

        # Generate CMap
        cmap_data = embedder._cidfont_builder._generate_to_unicode_cmap(tt_font)
        cmap_text = cmap_data.decode("ascii")

        # Check required CMap elements
        assert "/CIDInit /ProcSet findresource begin" in cmap_text
        assert "begincmap" in cmap_text
        assert "/CIDSystemInfo" in cmap_text
        assert "begincodespacerange" in cmap_text
        assert "<0000> <FFFF>" in cmap_text
        assert "endcodespacerange" in cmap_text
        assert "beginbfchar" in cmap_text
        assert "endbfchar" in cmap_text
        assert "endcmap" in cmap_text

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_w_array_sparse_format(self):
        """W array uses correct sparse format."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        # Load CIDFont
        _, tt_font = embedder._loader.load_cidfont_replacement_by_ordering("Japan1")

        # Generate W array
        w_array = embedder._metrics.build_cidfont_w_array(tt_font)

        # W array should not be empty
        assert len(w_array) > 0

        # Check format: entries are either
        #   [cid, [widths], ...] (individual format) or
        #   [cid_first, cid_last, width] (range format)
        i = 0
        while i < len(w_array):
            assert isinstance(w_array[i], int)
            if isinstance(w_array[i + 1], list):
                # Individual format: cid [w1 w2 ...]
                for width in w_array[i + 1]:
                    assert isinstance(width, int)
                    assert width >= 0
                i += 2
            else:
                # Range format: cid_first cid_last width
                assert isinstance(w_array[i + 1], int)
                assert isinstance(w_array[i + 2], int)
                assert w_array[i + 1] >= w_array[i]
                assert w_array[i + 2] >= 0
                i += 3

    def test_get_cidfont_encoding_identity_h(self):
        """_get_cidfont_encoding detects Identity-H (horizontal)."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=Name("/Identity-H"),
        )

        embedder = FontEmbedder(pdf)
        encoding = embedder._get_cidfont_encoding(font_dict)

        assert encoding == "Identity-H"

    def test_get_cidfont_encoding_identity_v(self):
        """_get_cidfont_encoding detects Identity-V (vertical)."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=Name("/Identity-V"),
        )

        embedder = FontEmbedder(pdf)
        encoding = embedder._get_cidfont_encoding(font_dict)

        assert encoding == "Identity-V"

    def test_get_cidfont_encoding_default(self):
        """_get_cidfont_encoding returns Identity-H as default."""
        pdf = new_pdf()

        # Font without Encoding
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
        )

        embedder = FontEmbedder(pdf)
        encoding = embedder._get_cidfont_encoding(font_dict)

        assert encoding == "Identity-H"

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_cidfont_preserves_identity_v_encoding(self):
        """CIDFont with Identity-V preserves vertical encoding after embedding."""
        pdf = new_pdf()

        # Create CIDFont with vertical encoding
        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/VerticalCJK"),
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/VerticalCJK"),
            Encoding=Name("/Identity-V"),  # Vertical encoding
            DescendantFonts=Array([descendant_font]),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "VerticalCJK" in result.fonts_embedded

        # Check that encoding is preserved
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]
        assert updated_font.get("/Encoding") == Name("/Identity-V")

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_cidfont_uses_identity_h_by_default(self):
        """CIDFont without explicit encoding uses Identity-H."""
        pdf = new_pdf()

        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/DefaultCJK"),
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/DefaultCJK"),
            # No encoding specified
            DescendantFonts=Array([descendant_font]),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "DefaultCJK" in result.fonts_embedded

        # Check that default encoding is Identity-H
        resources = pdf.pages[0].get("/Resources")
        updated_font = resources["/Font"]["/F1"]
        assert updated_font.get("/Encoding") == Name("/Identity-H")

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_multiple_cidfont_pages(self):
        """CIDFonts are correctly embedded on multiple pages."""
        pdf = new_pdf()

        # Create two pages with the same CIDFont
        for _ in range(2):
            descendant_font = pdf.make_indirect(
                Dictionary(
                    Type=Name.Font,
                    Subtype=Name.CIDFontType2,
                    BaseFont=Name("/SimSun"),
                )
            )

            font_dict = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/SimSun"),
                DescendantFonts=Array([descendant_font]),
            )

            page_dict = Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=font_dict),
                ),
            )

            content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
            page_dict[Name.Contents] = content_stream

            page = pikepdf.Page(page_dict)
            pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        # Font should be in the list only once (deduplicated)
        assert result.fonts_embedded.count("SimSun") == 1
        assert "SimSun" not in result.fonts_failed

        # Both page font objects must actually be embedded
        for page_idx in range(2):
            resources = pdf.pages[page_idx].get("/Resources")
            page_font = resources["/Font"]["/F1"]
            assert is_font_embedded(page_font), (
                f"Font on page {page_idx} was not embedded"
            )

    @pytest.mark.skipif(
        not _noto_cjk_font_available(),
        reason="Noto Sans CJK Font not installed",
    )
    def test_same_base_name_distinct_indirect_fonts_both_embedded(self):
        """Distinct indirect font objects with same base_name are both embedded."""
        pdf = new_pdf()

        # Create two pages, each with a distinct indirect CIDFont
        # named "SimSun" — simulates a merged PDF
        for _ in range(2):
            descendant_font = pdf.make_indirect(
                Dictionary(
                    Type=Name.Font,
                    Subtype=Name.CIDFontType2,
                    BaseFont=Name("/SimSun"),
                )
            )

            font_dict = pdf.make_indirect(
                Dictionary(
                    Type=Name.Font,
                    Subtype=Name.Type0,
                    BaseFont=Name("/SimSun"),
                    DescendantFonts=Array([descendant_font]),
                )
            )

            page_dict = Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=font_dict),
                ),
            )

            content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
            page_dict[Name.Contents] = content_stream

            page = pikepdf.Page(page_dict)
            pdf.pages.append(page)

        # Verify the two font objects are distinct indirect objects
        font_obj_0 = pdf.pages[0].get("/Resources")["/Font"]["/F1"]
        font_obj_1 = pdf.pages[1].get("/Resources")["/Font"]["/F1"]
        assert font_obj_0.objgen != font_obj_1.objgen

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        # Result list should only contain the name once (reporting dedup)
        assert result.fonts_embedded.count("SimSun") == 1
        assert "SimSun" not in result.fonts_failed

        # Both distinct font objects must be embedded
        for page_idx in range(2):
            resources = pdf.pages[page_idx].get("/Resources")
            page_font = resources["/Font"]["/F1"]
            assert is_font_embedded(page_font), (
                f"Distinct font object on page {page_idx} was not embedded"
            )

        # Font compliance should pass
        compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert compliant
        assert not missing


class TestIsUTF16Encoding:
    """Tests for _is_utf16_encoding() and _UTF16_ENCODING_NAMES."""

    def test_utf16_h_variants(self):
        """All UTF-16 horizontal variants are recognized."""
        for name in [
            "UniJIS-UTF16-H",
            "UniGB-UTF16-H",
            "UniCNS-UTF16-H",
            "UniKS-UTF16-H",
        ]:
            assert _is_utf16_encoding(name) is True

    def test_utf16_v_variants(self):
        """All UTF-16 vertical variants are recognized."""
        for name in [
            "UniJIS-UTF16-V",
            "UniGB-UTF16-V",
            "UniCNS-UTF16-V",
            "UniKS-UTF16-V",
        ]:
            assert _is_utf16_encoding(name) is True

    def test_ucs2_variants(self):
        """All UCS-2 variants are recognized."""
        for name in [
            "UniJIS-UCS2-H",
            "UniJIS-UCS2-V",
            "UniGB-UCS2-H",
            "UniGB-UCS2-V",
            "UniCNS-UCS2-H",
            "UniCNS-UCS2-V",
            "UniKS-UCS2-H",
            "UniKS-UCS2-V",
        ]:
            assert _is_utf16_encoding(name) is True

    def test_identity_h_not_utf16(self):
        """Identity-H is NOT a UTF-16 encoding."""
        assert _is_utf16_encoding("Identity-H") is False

    def test_identity_v_not_utf16(self):
        """Identity-V is NOT a UTF-16 encoding."""
        assert _is_utf16_encoding("Identity-V") is False

    def test_empty_string_not_utf16(self):
        """Empty string is not a UTF-16 encoding."""
        assert _is_utf16_encoding("") is False

    def test_all_names_in_frozenset(self):
        """All 16 expected encoding names are in the frozenset."""
        assert len(_UTF16_ENCODING_NAMES) == 16
