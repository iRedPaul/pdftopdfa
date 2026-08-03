# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/embedder.py — font embedding and FontEmbedder class."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available, _noto_cjk_font_available
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name

from pdftopdfa.exceptions import FontEmbeddingError
from pdftopdfa.fonts import (
    FONT_REPLACEMENTS,
    EmbeddingResult,
    FontEmbedder,
    check_font_compliance,
)
from pdftopdfa.fonts.analysis import is_font_embedded
from pdftopdfa.fonts.embedder import (
    _UTF16_ENCODING_NAMES,
    _is_utf16_encoding,
)
from pdftopdfa.fonts.loader import FontLoader
from pdftopdfa.fonts.tounicode import (
    generate_tounicode_cmap_data,
    parse_tounicode_cmap,
    parse_tounicode_cmap_sequences,
)
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


def _build_test_font_file(
    path: Path,
    *,
    family_name: str,
    style_name: str,
    postscript_name: str,
    fstype: int = 0,
    unicode_value: int = 0x41,
    glyph_name: str = "A",
    additional_characters: dict[int, str] | None = None,
) -> None:
    """Create a tiny TrueType font file for loader tests."""
    fb = FontBuilder(1000, isTTF=True)
    character_map = {unicode_value: glyph_name, **(additional_characters or {})}
    glyphs = [".notdef", *dict.fromkeys(character_map.values())]
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap(character_map)
    fb.setupGlyf({gname: Glyph() for gname in glyphs})
    fb.setupHorizontalMetrics(
        {".notdef": (500, 0), **dict.fromkeys(glyphs[1:], (600, 0))}
    )
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable(
        {
            "familyName": family_name,
            "styleName": style_name,
            "psName": postscript_name,
            "fullName": f"{family_name} {style_name}",
        }
    )
    fb.setupOS2(fsType=fstype)
    fb.setupPost()

    buffer = BytesIO()
    fb.font.save(buffer)
    path.write_bytes(buffer.getvalue())


def _build_unembedded_cidfont_pdf(
    pdf: pikepdf.Pdf,
    *,
    cmap_data: bytes,
    content_codes: bytes,
    cmap_name: str = "Custom-H",
    ordering: str = "Japan1",
    supplement: int = 2,
    registry: str = "Adobe",
    tounicode_data: bytes | None = None,
    nested: bool = False,
) -> tuple[pikepdf.Object, pikepdf.Object]:
    """Build a Type 0 test font with a stream CMap and shown text."""
    descendant = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType0,
            BaseFont=Name("/MissingJapaneseFont"),
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String(registry),
                Ordering=pikepdf.String(ordering),
                Supplement=supplement,
            ),
        )
    )
    encoding = pdf.make_indirect(pdf.make_stream(cmap_data))
    encoding[Name.Type] = Name.CMap
    encoding[Name.CMapName] = Name(f"/{cmap_name}")
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/MissingJapaneseFont"),
            Encoding=encoding,
            DescendantFonts=Array([descendant]),
        )
    )
    if tounicode_data is not None:
        font[Name.ToUnicode] = pdf.make_indirect(pdf.make_stream(tounicode_data))

    text = b"BT /F1 12 Tf <" + content_codes.hex().encode("ascii") + b"> Tj ET"
    if nested:
        form = pdf.make_stream(text)
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
        resources = Dictionary(XObject=Dictionary(Fm1=form))
        contents = pdf.make_stream(b"q /Fm1 Do Q")
    else:
        resources = Dictionary(Font=Dictionary(F1=font))
        contents = pdf.make_stream(text)
    pdf.pages.append(
        pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=resources,
                Contents=contents,
            )
        )
    )
    return font, encoding


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

        # Mock loader to avoid file system access
        with patch.object(embedder._loader, "load_replacement_font") as mock_load:
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

    def test_times_new_roman_alias_uses_standard_replacement(self):
        """TimesNewRoman aliases should not fall back to LiberationSans."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TimesNewRomanPSMT"),
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
        with patch.object(
            embedder,
            "_replace_simple_font",
            return_value=(True, False),
        ) as mock_replace:
            result = embedder.embed_missing_fonts()

        assert "TimesNewRomanPSMT" in result.fonts_embedded
        assert result.warnings == []
        assert mock_replace.call_count == 1
        assert mock_replace.call_args.kwargs["use_fallback"] is False


class TestFontLoader:
    """Tests for Windows allowlisted system-font loading."""

    @staticmethod
    def _make_windows_font_path(tmp_path: Path, filename: str) -> tuple[Path, Path]:
        """Return a fake ``%WINDIR%`` and a font path under ``Fonts``."""
        windir = tmp_path / "Windows"
        font_path = windir / "Fonts" / filename
        font_path.parent.mkdir(parents=True)
        return windir, font_path

    @staticmethod
    def _load_policy_font(
        tmp_path: Path,
        *,
        requested_name: str,
        postscript_name: str,
        family_name: str,
        style_name: str,
        use_fallback: bool,
        fstype: int = 0,
    ) -> tuple[bytes, object]:
        """Create a fake Windows font and load it through the policy path."""
        windir, font_path = TestFontLoader._make_windows_font_path(
            tmp_path, f"{postscript_name}.ttf"
        )
        _build_test_font_file(
            font_path,
            family_name=family_name,
            style_name=style_name,
            postscript_name=postscript_name,
            fstype=fstype,
        )

        loader = FontLoader({})
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=True),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
        ):
            return loader.load_replacement_font(
                requested_name,
                use_fallback=use_fallback,
            )

    @pytest.mark.parametrize(
        (
            "requested_name",
            "postscript_name",
            "family_name",
            "style_name",
            "use_fallback",
        ),
        [
            (
                "Times New Roman*",
                "TimesNewRomanPSMT",
                "Times New Roman",
                "Regular",
                True,
            ),
            (
                "Times New Roman,BoldItalic",
                "TimesNewRomanPS-BoldItalicMT",
                "Times New Roman",
                "Bold Italic",
                True,
            ),
            ("Arial,Bold", "Arial-BoldMT", "Arial", "Bold", True),
            (
                "TimesNewRomanPS-BoldMT",
                "TimesNewRomanPS-BoldMT",
                "Times New Roman",
                "Bold",
                False,
            ),
            ("Arial-BoldMT", "Arial-BoldMT", "Arial", "Bold", False),
            ("Calibri", "Calibri", "Calibri", "Regular", True),
            ("Consolas", "Consolas", "Consolas", "Regular", True),
            ("LucidaConsole", "LucidaConsole", "Lucida Console", "Regular", True),
        ],
    )
    def test_windows_allowlisted_font_is_used(
        self,
        tmp_path,
        requested_name,
        postscript_name,
        family_name,
        style_name,
        use_fallback,
    ):
        """Allowlisted Windows fonts under ``%WINDIR%\\Fonts`` are embedded."""
        font_data, tt_font = self._load_policy_font(
            tmp_path,
            requested_name=requested_name,
            postscript_name=postscript_name,
            family_name=family_name,
            style_name=style_name,
            use_fallback=use_fallback,
        )
        try:
            assert len(font_data) > 0
            assert tt_font["name"].getDebugName(6) == postscript_name
        finally:
            tt_font.close()

    def test_non_allowlisted_windows_font_is_ignored(self, tmp_path):
        """Non-allowlisted Windows fonts fall back to bundled replacements."""
        font_data, tt_font = self._load_policy_font(
            tmp_path,
            requested_name="Helvetica",
            postscript_name="Helvetica",
            family_name="Helvetica",
            style_name="Regular",
            use_fallback=False,
        )
        try:
            assert len(font_data) > 0
            assert tt_font["name"].getDebugName(1).startswith("Liberation Sans")
        finally:
            tt_font.close()

    def test_system_font_alias_is_not_reported_as_fallback(self, tmp_path):
        """A matching Windows font does not produce a fallback warning."""
        windir, font_path = self._make_windows_font_path(tmp_path, "times.ttf")
        _build_test_font_file(
            font_path,
            family_name="Times New Roman",
            style_name="Regular",
            postscript_name="TimesNewRomanPSMT",
        )
        pdf = new_pdf()
        pdf.pages.append(
            pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 612, 792]),
                    Resources=Dictionary(
                        Font=Dictionary(
                            F1=Dictionary(
                                Type=Name.Font,
                                Subtype=Name.TrueType,
                                BaseFont=Name("/Times New Roman"),
                            )
                        )
                    ),
                    Contents=pdf.make_stream(b"BT /F1 12 Tf (A) Tj ET"),
                )
            )
        )

        embedder = FontEmbedder(pdf)
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=True),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
        ):
            result = embedder.embed_missing_fonts()

        assert result.fonts_embedded == ["Times New Roman"]
        assert result.warnings == []

    @pytest.mark.parametrize(
        ("requested_name", "postscript_name"),
        [
            ("Arial", "Arial"),
            ("Times New Roman", "TimesNewRomanPSMT*"),
        ],
    )
    def test_request_aliases_do_not_expand_internal_postscript_allowlist(
        self,
        tmp_path,
        requested_name,
        postscript_name,
    ):
        """Request aliases do not authorize non-allowlisted internal names."""
        windir, font_path = self._make_windows_font_path(tmp_path, "spoofed.ttf")
        _build_test_font_file(
            font_path,
            family_name="Spoofed Font",
            style_name="Regular",
            postscript_name=postscript_name,
        )

        loader = FontLoader({})
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=True),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
        ):
            _font_data, tt_font = loader.load_replacement_font(
                requested_name,
                use_fallback=True,
            )

        try:
            assert tt_font["name"].getDebugName(1).startswith("Liberation Sans")
        finally:
            tt_font.close()

    def test_allowlisted_font_outside_windows_fonts_is_ignored(self, tmp_path):
        """Allowlisted fonts outside ``%WINDIR%\\Fonts`` are never used."""
        outside_font = tmp_path / "external" / "Calibri.ttf"
        outside_font.parent.mkdir(parents=True)
        _build_test_font_file(
            outside_font,
            family_name="Calibri",
            style_name="Regular",
            postscript_name="Calibri",
        )

        windir = tmp_path / "Windows"
        (windir / "Fonts").mkdir(parents=True)

        loader = FontLoader({})
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=True),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
            patch.object(
                loader, "_iter_system_font_files", return_value=[outside_font]
            ),
        ):
            _font_data, tt_font = loader.load_replacement_font(
                "Calibri",
                use_fallback=True,
            )

        try:
            assert tt_font["name"].getDebugName(1).startswith("Liberation Sans")
        finally:
            tt_font.close()

    @pytest.mark.parametrize(
        "postscript_name",
        ["Helvetica", "Arial", "TimesNewRomanPSMT*"],
    )
    def test_resolved_postscript_name_must_still_be_allowlisted(
        self,
        tmp_path,
        postscript_name,
    ):
        """The matched file's actual PostScript name is rechecked after lookup."""
        windir, font_path = self._make_windows_font_path(tmp_path, "spoofed.ttf")
        _build_test_font_file(
            font_path,
            family_name="Spoofed Font",
            style_name="Regular",
            postscript_name=postscript_name,
        )

        loader = FontLoader({})
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=True),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
            patch.object(
                loader,
                "_build_system_font_index",
                return_value={
                    loader._normalize_font_lookup_name("Calibri"): (font_path, None)
                },
            ),
        ):
            _font_data, tt_font = loader.load_replacement_font(
                "Calibri",
                use_fallback=True,
            )

        try:
            assert tt_font["name"].getDebugName(1).startswith("Liberation Sans")
        finally:
            tt_font.close()

    def test_non_windows_platform_never_uses_system_fonts(self, tmp_path):
        """macOS/Linux policy always skips local system fonts."""
        windir, font_path = self._make_windows_font_path(tmp_path, "Calibri.ttf")
        _build_test_font_file(
            font_path,
            family_name="Calibri",
            style_name="Regular",
            postscript_name="Calibri",
        )

        loader = FontLoader({})
        with (
            patch.object(FontLoader, "_is_windows_platform", return_value=False),
            patch.dict("os.environ", {"WINDIR": str(windir)}, clear=False),
        ):
            _font_data, tt_font = loader.load_replacement_font(
                "Calibri",
                use_fallback=True,
            )

        try:
            assert tt_font["name"].getDebugName(1).startswith("Liberation Sans")
        finally:
            tt_font.close()

    def test_bitmap_only_system_font_falls_back_to_bundled_replacement(self, tmp_path):
        """Bitmap-only fonts are blocked even when allowlisted."""
        _font_data, tt_font = self._load_policy_font(
            tmp_path,
            requested_name="TimesNewRomanPS-BoldMT",
            postscript_name="TimesNewRomanPS-BoldMT",
            family_name="Times New Roman",
            style_name="Bold",
            use_fallback=False,
            fstype=0x0200,
        )
        try:
            assert tt_font["name"].getDebugName(1).startswith("Liberation Serif")
        finally:
            tt_font.close()

    def test_preview_and_print_system_font_logs_info(self, tmp_path, caplog):
        """Preview & Print fonts stay usable and are logged as info only."""
        caplog.set_level("INFO")
        _font_data, tt_font = self._load_policy_font(
            tmp_path,
            requested_name="Calibri",
            postscript_name="Calibri",
            family_name="Calibri",
            style_name="Regular",
            use_fallback=True,
            fstype=0x0004,
        )
        try:
            assert tt_font["name"].getDebugName(6) == "Calibri"
            assert any("Preview & Print" in record.message for record in caplog.records)
            assert all(record.levelname != "WARNING" for record in caplog.records)
        finally:
            tt_font.close()

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

        # Replacing a CIDFont breaks the glyph-ID mapping of the content
        # stream, so a prominent per-font warning must be emitted.
        assert any("MSGothic" in w and "render" in w for w in result.warnings), (
            result.warnings
        )


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

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_replace_subsetted_standard14_font(self, pdf_with_helvetica):
        """Subsetted embedded Standard-14 fonts are refreshed to full fonts."""
        embedder = FontEmbedder(pdf_with_helvetica)
        result = embedder.embed_missing_fonts()
        assert "Helvetica" in result.fonts_embedded

        font = _resolve_indirect(pdf_with_helvetica.pages[0].Resources["/Font"]["/F1"])
        font[Name.BaseFont] = Name("/ABCDEF+Helvetica")
        font_descriptor = _resolve_indirect(font["/FontDescriptor"])
        font_descriptor[Name.FontName] = Name("/ABCDEF+Helvetica")

        refreshed = FontEmbedder(
            pdf_with_helvetica
        ).replace_subsetted_standard14_fonts()
        font_descriptor = _resolve_indirect(font["/FontDescriptor"])

        assert "Helvetica" in refreshed.fonts_embedded
        assert font.get("/BaseFont") == Name("/Helvetica")
        assert font_descriptor.get("/FontName") == Name("/Helvetica")
        assert is_font_embedded(font)

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_replace_subsetted_standard14_font_preserves_custom_code_mapping(
        self,
        pdf_with_helvetica,
    ):
        """Refresh keeps existing visible text mapping for subset fonts."""
        embedder = FontEmbedder(pdf_with_helvetica)
        result = embedder.embed_missing_fonts()
        assert "Helvetica" in result.fonts_embedded

        font = _resolve_indirect(pdf_with_helvetica.pages[0].Resources["/Font"]["/F1"])
        font[Name.BaseFont] = Name("/ABCDEF+Helvetica")
        font_descriptor = _resolve_indirect(font["/FontDescriptor"])
        font_descriptor[Name.FontName] = Name("/ABCDEF+Helvetica")
        if font.get("/Encoding") is not None:
            del font[Name.Encoding]

        custom_mapping = {
            32: ord("5"),
            33: ord("8"),
            34: ord("4"),
            42: ord("T"),
            43: ord("e"),
            44: ord("c"),
            45: ord("h"),
        }
        font[Name.ToUnicode] = pdf_with_helvetica.make_indirect(
            pikepdf.Stream(
                pdf_with_helvetica,
                generate_tounicode_cmap_data(custom_mapping),
            )
        )

        refreshed = FontEmbedder(
            pdf_with_helvetica
        ).replace_subsetted_standard14_fonts()
        font_descriptor = _resolve_indirect(font["/FontDescriptor"])

        assert "Helvetica" in refreshed.fonts_embedded
        assert font.get("/BaseFont") == Name("/Helvetica")
        assert font_descriptor.get("/FontName") == Name("/Helvetica")
        encoding = _resolve_indirect(font["/Encoding"])
        assert isinstance(encoding, Dictionary)
        assert encoding.get("/BaseEncoding") == Name.WinAnsiEncoding
        differences = list(encoding["/Differences"])
        assert differences[:8] == [
            32,
            Name("/five"),
            Name("/eight"),
            Name("/four"),
            42,
            Name("/T"),
            Name("/e"),
            Name("/c"),
        ]
        parsed_mapping = parse_tounicode_cmap(bytes(font["/ToUnicode"].read_bytes()))
        assert parsed_mapping == custom_mapping

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_collect_subsetted_standard14_font_ids_skips_symbolic_subset_fonts(
        self,
        pdf_with_helvetica,
    ):
        """Symbolic subsetted Standard-14 fonts are excluded from refresh."""
        embedder = FontEmbedder(pdf_with_helvetica)
        result = embedder.embed_missing_fonts()
        assert "Helvetica" in result.fonts_embedded

        font = _resolve_indirect(pdf_with_helvetica.pages[0].Resources["/Font"]["/F1"])
        font[Name.BaseFont] = Name("/ABCDEF+Helvetica")
        font_descriptor = _resolve_indirect(font["/FontDescriptor"])
        font_descriptor[Name.FontName] = Name("/ABCDEF+Helvetica")
        font_descriptor[Name.Flags] = 4

        collected = FontEmbedder(
            pdf_with_helvetica
        ).collect_subsetted_standard14_font_ids()

        assert font.objgen not in collected


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
        not _liberation_fonts_available(),
        reason="Liberation fonts not available",
    )
    def test_empty_descendantfonts_type0_is_embedded_with_tounicode_preserved(self):
        """Malformed Latin Type0 fonts are rebuilt with a CIDFont descendant."""
        pdf = new_pdf()
        custom_mapping = {
            3: ord(" "),
            36: ord("A"),
        }
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/ABCDEF+Arial"),
            DescendantFonts=Array(),
            Encoding=Name("/Identity-H"),
            ToUnicode=pdf.make_indirect(
                pikepdf.Stream(
                    pdf,
                    generate_tounicode_cmap_data(custom_mapping),
                )
            ),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=pdf.make_indirect(font_dict)),
            ),
        )
        content_stream = pdf.make_stream(b"BT /F1 12 Tf <0024> Tj ET")
        page_dict[Name.Contents] = content_stream
        pdf.pages.append(pikepdf.Page(page_dict))

        result = FontEmbedder(pdf).embed_missing_fonts()

        assert "Arial" in result.fonts_embedded
        rebuilt_font = _resolve_indirect(pdf.pages[0].Resources["/Font"]["/F1"])
        assert is_font_embedded(rebuilt_font)
        assert len(rebuilt_font["/DescendantFonts"]) == 1
        parsed_mapping = parse_tounicode_cmap(
            bytes(rebuilt_font.ToUnicode.read_bytes())
        )
        assert parsed_mapping == custom_mapping

    @pytest.mark.parametrize(
        (
            "encoding_name",
            "source_code",
            "code_space_start",
            "code_space_end",
            "unicode_value",
            "glyph_name",
            "cid",
        ),
        [
            (
                "UniJIS-UCS2-H",
                b"\x00A",
                b"\x00\x00",
                b"\xff\xff",
                0x41,
                "A",
                5,
            ),
            (
                "UniJIS-UTF16-H",
                b"\xd8@\xdc\x00",
                b"\xd8\x00\xdc\x00",
                b"\xdb\xff\xdf\xff",
                0x20000,
                "u20000",
                17,
            ),
        ],
    )
    def test_unicode_cmap_is_remapped_to_replacement_gids(
        self,
        tmp_path,
        encoding_name,
        source_code,
        code_space_start,
        code_space_end,
        unicode_value,
        glyph_name,
        cid,
    ):
        """Unicode CMap codes keep their text semantics after substitution."""
        font_path = tmp_path / "replacement.ttf"
        _build_test_font_file(
            font_path,
            family_name="Replacement",
            style_name="Regular",
            postscript_name="Replacement-Regular",
            unicode_value=unicode_value,
            glyph_name=glyph_name,
        )
        font_data = font_path.read_bytes()
        tt_font = TTFont(BytesIO(font_data))
        pdf = new_pdf()
        cid_system_info = Dictionary(
            Registry=pikepdf.String("Adobe"),
            Ordering=pikepdf.String("Japan1"),
            Supplement=2,
        )
        descendant = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType0,
                BaseFont=Name("/MissingJapaneseFont"),
                CIDSystemInfo=cid_system_info,
            )
        )
        encoding = pdf.make_indirect(
            pdf.make_stream(
                (
                    "begincmap\n"
                    "1 begincodespacerange\n"
                    f"<{code_space_start.hex()}> <{code_space_end.hex()}>\n"
                    "endcodespacerange\n"
                    "1 begincidchar\n"
                    f"<{source_code.hex()}> {cid}\n"
                    "endcidchar\n"
                    "endcmap\n"
                ).encode("ascii")
            )
        )
        encoding[Name.Type] = Name.CMap
        encoding[Name.CMapName] = Name(f"/{encoding_name}")
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/MissingJapaneseFont"),
                Encoding=encoding,
                DescendantFonts=Array([descendant]),
            )
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=pdf.make_stream(
                    b"BT /F1 12 Tf <" + source_code.hex().encode("ascii") + b"> Tj ET"
                ),
            )
        )
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        try:
            with patch.object(
                embedder._loader,
                "load_cidfont_replacement_by_ordering",
                return_value=(font_data, tt_font),
            ):
                result = embedder.embed_missing_fonts()

            assert result.fonts_failed == []
            assert font["/Encoding"].objgen == encoding.objgen
            rebuilt_descendant = _resolve_indirect(font["/DescendantFonts"][0])
            assert str(rebuilt_descendant.CIDSystemInfo.Ordering) == "Japan1"
            cidtogid = bytes(rebuilt_descendant.CIDToGIDMap.read_bytes())
            offset = cid * 2
            assert int.from_bytes(
                cidtogid[offset : offset + 2], "big"
            ) == tt_font.getGlyphID(glyph_name)
            assert int(rebuilt_descendant.W[0]) == cid
            assert parse_tounicode_cmap_sequences(
                bytes(font.ToUnicode.read_bytes())
            ) == {source_code: (unicode_value,)}
        finally:
            tt_font.close()

    def test_custom_vertical_unicode_usecmap_is_remapped_in_nested_form(
        self,
        tmp_path,
    ):
        """A custom Unicode usecmap keeps its CIDs, direction, and semantics."""
        font_path = tmp_path / "replacement.ttf"
        _build_test_font_file(
            font_path,
            family_name="Replacement",
            style_name="Regular",
            postscript_name="Replacement-Regular",
        )
        font_data = font_path.read_bytes()
        tt_font = TTFont(BytesIO(font_data))
        pdf = new_pdf()
        source_code = b"\x00A"
        cid = 65535
        cmap_data = (
            b"begincmap\n"
            b"/UniJIS-UCS2-V usecmap\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            b"1 begincidchar\n<0041> 65535\nendcidchar\n"
            b"endcmap\n"
        )
        font, original_encoding = _build_unembedded_cidfont_pdf(
            pdf,
            cmap_data=cmap_data,
            content_codes=source_code,
            cmap_name="Custom-V",
            nested=True,
        )

        embedder = FontEmbedder(pdf)
        try:
            with patch.object(
                embedder._loader,
                "load_cidfont_replacement_by_ordering",
                return_value=(font_data, tt_font),
            ):
                result = embedder.embed_missing_fonts()

            assert result.fonts_failed == []
            assert font["/Encoding"].objgen == original_encoding.objgen
            descendant = _resolve_indirect(font["/DescendantFonts"][0])
            cidtogid = bytes(descendant.CIDToGIDMap.read_bytes())
            assert int.from_bytes(cidtogid[cid * 2 : cid * 2 + 2], "big") == 1
            assert int(descendant.W[0]) == cid
            assert parse_tounicode_cmap_sequences(
                bytes(font.ToUnicode.read_bytes())
            ) == {source_code: (ord("A"),)}
        finally:
            tt_font.close()

    @pytest.mark.parametrize(
        ("source_code", "unicode_value"),
        [(b"\x00B", ord("B")), (b"\xd8\x00", 0xFFFD)],
    )
    def test_unicode_cmap_missing_or_invalid_glyph_uses_fallback(
        self,
        tmp_path,
        source_code,
        unicode_value,
    ):
        """Missing glyphs and invalid Unicode codes use an explicit fallback."""
        font_path = tmp_path / "replacement.ttf"
        _build_test_font_file(
            font_path,
            family_name="Replacement",
            style_name="Regular",
            postscript_name="Replacement-Regular",
            additional_characters={ord("?"): "question"},
        )
        font_data = font_path.read_bytes()
        tt_font = TTFont(BytesIO(font_data))
        pdf = new_pdf()
        font, original_encoding = _build_unembedded_cidfont_pdf(
            pdf,
            cmap_data=(
                b"begincmap\n"
                b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
                b"1 begincidchar\n<"
                + source_code.hex().encode("ascii")
                + b"> 9\nendcidchar\n"
                b"endcmap\n"
            ),
            content_codes=source_code,
            cmap_name="UniJIS-UCS2-H",
        )

        embedder = FontEmbedder(pdf)
        try:
            with patch.object(
                embedder._loader,
                "load_cidfont_replacement_by_ordering",
                return_value=(font_data, tt_font),
            ):
                result = embedder.embed_missing_fonts()

            assert result.fonts_failed == []
            assert font["/Encoding"].objgen == original_encoding.objgen
            descendant = _resolve_indirect(font["/DescendantFonts"][0])
            cidtogid = bytes(descendant.CIDToGIDMap.read_bytes())
            assert int.from_bytes(cidtogid[18:20], "big") == tt_font.getGlyphID(
                "question"
            )
            assert parse_tounicode_cmap_sequences(
                bytes(font.ToUnicode.read_bytes())
            ) == {source_code: (unicode_value,)}
        finally:
            tt_font.close()

    def test_arbitrary_mixed_cmap_uses_authoritative_tounicode(self, tmp_path):
        """Authoritative Unicode drives arbitrary and colliding CID mappings."""
        font_path = tmp_path / "replacement.ttf"
        _build_test_font_file(
            font_path,
            family_name="Replacement",
            style_name="Regular",
            postscript_name="Replacement-Regular",
            additional_characters={ord("B"): "B", ord("?"): "question"},
        )
        font_data = font_path.read_bytes()
        tt_font = TTFont(BytesIO(font_data))
        pdf = new_pdf()
        cmap_data = (
            b"begincmap\n"
            b"2 begincodespacerange\n<20> <7F>\n<0100> <01FF>\n"
            b"endcodespacerange\n"
            b"3 begincidchar\n<41> 7\n<42> 7\n<0123> 8\nendcidchar\n"
            b"endcmap\n"
        )
        tounicode_data = (
            b"begincmap\n"
            b"2 begincodespacerange\n<20> <7F>\n<0100> <01FF>\n"
            b"endcodespacerange\n"
            b"3 beginbfchar\n"
            b"<41> <0041>\n<42> <0042>\n<0123> <00660069>\n"
            b"endbfchar\nendcmap\n"
        )
        font, original_encoding = _build_unembedded_cidfont_pdf(
            pdf,
            cmap_data=cmap_data,
            content_codes=b"AB\x01#",
            tounicode_data=tounicode_data,
        )

        embedder = FontEmbedder(pdf)
        try:
            with patch.object(
                embedder._loader,
                "load_cidfont_replacement_by_ordering",
                return_value=(font_data, tt_font),
            ):
                result = embedder.embed_missing_fonts()

            assert result.fonts_failed == []
            assert font["/Encoding"].objgen == original_encoding.objgen
            descendant = _resolve_indirect(font["/DescendantFonts"][0])
            cidtogid = bytes(descendant.CIDToGIDMap.read_bytes())
            assert int.from_bytes(cidtogid[14:16], "big") == tt_font.getGlyphID("A")
            assert int.from_bytes(cidtogid[16:18], "big") == tt_font.getGlyphID(
                "question"
            )
            assert parse_tounicode_cmap_sequences(
                bytes(font.ToUnicode.read_bytes())
            ) == {
                b"A": (ord("A"),),
                b"B": (ord("B"),),
                b"\x01#": (ord("f"), ord("i")),
            }
        finally:
            tt_font.close()

    @pytest.mark.parametrize("ordering", ["Japan1", "GB1", "CNS1", "Korea1"])
    def test_arbitrary_cmap_uses_adobe_collection_and_unknown_fallback(
        self,
        tmp_path,
        ordering,
    ):
        """Known collection CIDs are remapped and unknown CIDs are explicit."""
        font_path = tmp_path / f"{ordering}.ttf"
        _build_test_font_file(
            font_path,
            family_name="Replacement",
            style_name="Regular",
            postscript_name="Replacement-Regular",
            additional_characters={ord("?"): "question"},
        )
        font_data = font_path.read_bytes()
        tt_font = TTFont(BytesIO(font_data))
        pdf = new_pdf()
        font, original_encoding = _build_unembedded_cidfont_pdf(
            pdf,
            cmap_data=(
                b"begincmap\n"
                b"1 begincodespacerange\n<A1> <A2>\nendcodespacerange\n"
                b"2 begincidchar\n<A1> 34\n<A2> 65535\nendcidchar\n"
                b"endcmap\n"
            ),
            content_codes=b"\xa1\xa2",
            ordering=ordering,
        )

        embedder = FontEmbedder(pdf)
        try:
            with patch.object(
                embedder._loader,
                "load_cidfont_replacement_by_ordering",
                return_value=(font_data, tt_font),
            ):
                result = embedder.embed_missing_fonts()

            assert result.fonts_failed == []
            assert font["/Encoding"].objgen == original_encoding.objgen
            descendant = _resolve_indirect(font["/DescendantFonts"][0])
            cidtogid = bytes(descendant.CIDToGIDMap.read_bytes())
            assert int.from_bytes(cidtogid[68:70], "big") == tt_font.getGlyphID("A")
            assert int.from_bytes(cidtogid[-2:], "big") == tt_font.getGlyphID(
                "question"
            )
            assert parse_tounicode_cmap_sequences(
                bytes(font.ToUnicode.read_bytes())
            ) == {b"\xa1": (ord("A"),), b"\xa2": (0xFFFD,)}
        finally:
            tt_font.close()

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

    def test_get_cidfont_encoding_unicode_cmap_stream_v(self):
        """Vertical writing mode is detected from a Unicode CMap stream."""
        pdf = new_pdf()
        encoding_stream = pdf.make_stream(b"begincmap endcmap")
        encoding_stream[Name.CMapName] = Name("/UniJIS-UCS2-V")
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=encoding_stream,
        )

        assert FontEmbedder(pdf)._get_cidfont_encoding(font_dict) == "Identity-V"

    def test_get_cidfont_encoding_follows_usecmap_stream_entry(self):
        """Vertical writing mode is inherited through a stream /UseCMap entry."""
        pdf = new_pdf()
        base = pdf.make_stream(b"begincmap endcmap")
        base[Name.CMapName] = Name("/UniJIS-UTF16-V")
        encoding = pdf.make_stream(b"begincmap endcmap")
        encoding[Name.CMapName] = Name("/Custom-V")
        encoding[Name.UseCMap] = base
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=encoding,
        )

        assert FontEmbedder(pdf)._get_cidfont_encoding(font_dict) == "Identity-V"

    def test_get_cidfont_encoding_reads_stream_wmode_operator(self):
        """An embedded CMap's WMode operator selects vertical writing."""
        pdf = new_pdf()
        encoding = pdf.make_stream(b"begincmap\n/WMode 1 def\nendcmap")
        encoding[Name.CMapName] = Name("/Custom")
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=encoding,
        )

        assert FontEmbedder(pdf)._get_cidfont_encoding(font_dict) == "Identity-V"

    def test_get_cidfont_encoding_ignores_commented_usecmap(self):
        """Comment text cannot impersonate a predefined vertical usecmap."""
        pdf = new_pdf()
        encoding = pdf.make_stream(b"begincmap\n% /UniJIS-UCS2-V usecmap\nendcmap")
        encoding[Name.CMapName] = Name("/Custom")
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=encoding,
        )

        assert FontEmbedder(pdf)._get_cidfont_encoding(font_dict) == "Identity-H"

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
