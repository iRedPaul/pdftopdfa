# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/analysis.py — font embedding detection and flags."""

from unittest.mock import MagicMock

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts import FontEmbedder, check_font_compliance
from pdftopdfa.fonts.analysis import is_font_embedded, is_symbolic_font
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


class TestIsFontEmbedded:
    """Unit tests for is_font_embedded edge cases."""

    def test_subset_prefix_without_font_descriptor_not_embedded(self):
        """A font with subset prefix but no FontDescriptor is not embedded."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/ABCDEF+FakeFont"),
        )
        assert not is_font_embedded(font_dict)

    def test_subset_prefix_without_font_descriptor_detected_as_missing(
        self,
    ):
        """Subset font without data is reported as missing for compliance."""
        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/GHIJKL+NoData"),
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        content_stream = pdf.make_stream(b"BT /F1 12 Tf (x) Tj ET")
        page_dict[Name.Contents] = content_stream
        pdf.pages.append(pikepdf.Page(page_dict))

        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert not is_compliant
        assert "GHIJKL+NoData" in missing

    def test_empty_fontfile2_stream_not_embedded(
        self,
    ):
        """Font with zero-length FontFile2 stream is not embedded."""
        pdf = new_pdf()
        font_stream = pdf.make_stream(b"")
        font_stream[Name.Length1] = 0
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
                FontFile2=font_stream,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FontDescriptor=fd,
        )
        assert not is_font_embedded(font_dict)

    def test_valid_fontfile2_stream_is_embedded(
        self,
    ):
        """Font with real FontFile2 data is reported as embedded."""
        pdf = new_pdf()
        font_stream = pdf.make_stream(b"\x00\x01\x00\x00" + b"\x00" * 100)
        font_stream[Name.Length1] = 104
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/RealFont"),
                Flags=32,
                FontFile2=font_stream,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/RealFont"),
            FontDescriptor=fd,
        )
        assert is_font_embedded(font_dict)

    def test_type0_empty_descendant_fontfile_not_embedded(self):
        """Type0 with descendant having empty FontFile3 is not embedded."""
        pdf = new_pdf()
        font_stream = pdf.make_stream(b"")
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/CIDFont"),
                Flags=4,
                FontFile3=font_stream,
            )
        )
        desc_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/CIDFont"),
                FontDescriptor=fd,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/CIDFont"),
            DescendantFonts=Array([desc_font]),
            Encoding=Name("/Identity-H"),
        )
        assert not is_font_embedded(font_dict)

    def test_garbage_fontfile2_data_not_embedded(self):
        """FontFile2 with data but invalid signature is not embedded."""
        pdf = new_pdf()
        # 30KB of null bytes — mimics isartor test files
        font_stream = pdf.make_stream(b"\x00" * 30000)
        font_stream[Name.Length1] = 30000
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/FakeFont"),
                Flags=32,
                FontFile2=font_stream,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/FakeFont"),
            FontDescriptor=fd,
        )
        assert not is_font_embedded(font_dict)

    def test_garbage_fontfile_type1_not_embedded(self):
        """FontFile with spaces (invalid Type1 signature) is not embedded."""
        pdf = new_pdf()
        font_stream = pdf.make_stream(b"    " + b" " * 30000)
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BadType1"),
                Flags=32,
                FontFile=font_stream,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/BadType1"),
            FontDescriptor=fd,
        )
        assert not is_font_embedded(font_dict)

    def test_corrupt_fontfile_reference_not_embedded(self):
        """FontDescriptor with FontFile key but unreadable stream is not embedded."""
        pdf = new_pdf()
        # Create a FontDescriptor that points to a non-stream object
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BadFont"),
                Flags=32,
                FontFile2=pdf.make_indirect(Dictionary()),
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/BadFont"),
            FontDescriptor=fd,
        )
        assert not is_font_embedded(font_dict)

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not available",
    )
    def test_embed_font_with_empty_fontfile(self):
        """Font with empty FontFile2 triggers embedding (not skipped)."""
        pdf = new_pdf()
        font_stream = pdf.make_stream(b"")
        font_stream[Name.Length1] = 0
        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/Helvetica"),
                Flags=32,
                FontFile2=font_stream,
            )
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
            FontDescriptor=fd,
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        content_stream = pdf.make_stream(b"BT /F1 12 Tf (test) Tj ET")
        page_dict[Name.Contents] = content_stream
        pdf.pages.append(pikepdf.Page(page_dict))

        with FontEmbedder(pdf) as embedder:
            result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded
        # After embedding, the font should now be embedded
        page = pdf.pages[0]
        embedded_font = _resolve_indirect(page.Resources.Font["/F1"])
        assert is_font_embedded(embedded_font)

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not available",
    )
    def test_acroform_dr_fonts_embedded(self):
        """Font only in AcroForm DR is detected and embedded."""
        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )
        # Create a page (required for embed_missing_fonts)
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        # Add font only in AcroForm DR
        font_ref = pdf.make_indirect(font_dict)
        acroform = Dictionary(
            DR=Dictionary(Font=Dictionary(F1=font_ref)),
        )
        pdf.Root[Name("/AcroForm")] = pdf.make_indirect(acroform)

        with FontEmbedder(pdf) as embedder:
            result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded
        # Verify font is now embedded
        dr_font = _resolve_indirect(
            _resolve_indirect(_resolve_indirect(pdf.Root.AcroForm).DR).Font["/F1"]
        )
        assert is_font_embedded(dr_font)


class TestSymbolFontMetrics:
    """Tests for Symbol font specific functionality."""

    def test_extract_font_metrics_symbolic_flag(self):
        """_metrics.extract_metrics sets correct Symbolic flag."""
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
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
            }[key]
        )
        mock_tt_font.__contains__ = MagicMock(return_value=True)
        mock_tt_font.get = MagicMock(
            side_effect=lambda key: (
                mock_tt_font[key] if key in ["head", "OS/2", "post"] else None
            )
        )

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        # Nonsymbolic (Standard) - check bit is set using bitwise comparison
        metrics = embedder._metrics.extract_metrics(mock_tt_font)
        assert metrics["Flags"] & 32  # Nonsymbolic bit is set

        # Symbolic - check bit is set using bitwise comparison
        metrics_symbol = embedder._metrics.extract_metrics(mock_tt_font, is_symbol=True)
        assert metrics_symbol["Flags"] & 4  # Symbolic bit is set

    def test_extract_widths_for_encoding(self):
        """_metrics.extract_widths_for_encoding uses encoding glyph names."""
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "head": MagicMock(unitsPerEm=1000),
                "hmtx": MagicMock(
                    metrics={
                        ".notdef": (500, 0),
                        "space": (250, 0),
                        "Alpha": (700, 0),
                        "alpha": (600, 0),
                    }
                ),
            }[key]
        )

        test_encoding = {
            32: "space",
            65: "Alpha",
            97: "alpha",
        }

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        widths = embedder._metrics.extract_widths_for_encoding(
            mock_tt_font, test_encoding
        )

        assert len(widths) == 256
        assert widths[32] == 250  # space
        assert widths[65] == 700  # Alpha
        assert widths[97] == 600  # alpha
        # Undefined codes should have .notdef width
        assert widths[0] == 500

    def test_build_encoding_dictionary(self):
        """_build_encoding_dictionary creates correct Differences array."""
        test_encoding = {
            32: "space",
            65: "Alpha",
            66: "Beta",
            97: "alpha",
        }

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        encoding_dict = embedder._build_encoding_dictionary(test_encoding)

        assert encoding_dict[Name.Type] == Name.Encoding
        differences = list(encoding_dict[Name.Differences])

        # Should contain sequences with start code
        # [32, /space, 65, /Alpha, /Beta, 97, /alpha]
        assert 32 in differences
        assert 65 in differences
        assert 97 in differences


class TestFontFlagsComputation:
    """Tests for PDF font flags computation from TrueType data."""

    def _create_mock_font(
        self,
        *,
        is_fixed_pitch: int = 0,
        family_class: int = 0,
        fs_selection: int = 0,
        italic_angle: float = 0,
    ) -> MagicMock:
        """Creates a mock TTFont with configurable properties."""
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
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
                    sFamilyClass=family_class << 8,  # Upper byte is class
                    fsSelection=fs_selection,
                ),
                "post": MagicMock(
                    italicAngle=italic_angle,
                    isFixedPitch=is_fixed_pitch,
                ),
            }[key]
        )
        mock_tt_font.__contains__ = MagicMock(return_value=True)
        mock_tt_font.get = MagicMock(
            side_effect=lambda key: (
                mock_tt_font[key] if key in ["head", "OS/2", "post"] else None
            )
        )
        return mock_tt_font

    def test_flags_nonsymbolic_default(self):
        """Nonsymbolic flag (32) is set for regular fonts."""
        mock_font = self._create_mock_font()

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 32  # Nonsymbolic bit is set
        assert not (flags & 4)  # Symbolic bit is not set

    def test_flags_symbolic(self):
        """Symbolic flag (4) is set for symbol fonts."""
        mock_font = self._create_mock_font()

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font, is_symbol=True)

        assert flags & 4  # Symbolic bit is set
        assert not (flags & 32)  # Nonsymbolic bit is not set

    def test_flags_fixed_pitch_from_post(self):
        """FixedPitch flag (1) is set from post.isFixedPitch."""
        mock_font = self._create_mock_font(is_fixed_pitch=1)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 1  # FixedPitch bit is set
        assert flags & 32  # Nonsymbolic still set

    def test_flags_serif_from_family_class(self):
        """Serif flag (2) is set for sFamilyClass 1-7."""
        # Test class 1 (Oldstyle Serifs)
        mock_font_1 = self._create_mock_font(family_class=1)
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font_1)
        assert flags & 2  # Serif bit is set

        # Test class 4 (Clarendon Serifs)
        mock_font_4 = self._create_mock_font(family_class=4)
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font_4)
        assert flags & 2  # Serif bit is set

        # Test class 7 (Freeform Serifs)
        mock_font_7 = self._create_mock_font(family_class=7)
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font_7)
        assert flags & 2  # Serif bit is set

    def test_flags_sans_serif_no_serif_flag(self):
        """Serif flag (2) is NOT set for sFamilyClass 8 (Sans Serif)."""
        mock_font = self._create_mock_font(family_class=8)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert not (flags & 2)  # Serif bit is NOT set
        assert flags & 32  # Nonsymbolic still set

    def test_flags_italic_from_fsselection(self):
        """Italic flag (64) is set from OS/2.fsSelection bit 0."""
        mock_font = self._create_mock_font(fs_selection=0x0001)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 64  # Italic bit is set

    def test_flags_italic_from_italic_angle(self):
        """Italic flag (64) is set from post.italicAngle != 0."""
        mock_font = self._create_mock_font(italic_angle=-12.0)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 64  # Italic bit is set

    def test_flags_script_from_family_class(self):
        """Script flag (8) is set for sFamilyClass 10."""
        mock_font = self._create_mock_font(family_class=10)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 8  # Script bit is set

    def test_flags_combined_serif_italic(self):
        """Multiple flags can be combined correctly."""
        # Serif (class 2) + Italic (fsSelection bit 0)
        mock_font = self._create_mock_font(family_class=2, fs_selection=0x0001)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font)

        assert flags & 32  # Nonsymbolic
        assert flags & 2  # Serif
        assert flags & 64  # Italic
        assert not (flags & 1)  # Not FixedPitch
        assert not (flags & 8)  # Not Script

    def test_flags_symbol_skips_serif_italic_detection(self):
        """Symbol fonts don't get Serif/Italic/Script flags even if present in data."""
        # Data indicates serif and italic, but is_symbol=True should skip
        mock_font = self._create_mock_font(family_class=2, fs_selection=0x0001)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font, is_symbol=True)

        assert flags & 4  # Symbolic
        assert not (flags & 32)  # Not Nonsymbolic
        assert not (flags & 2)  # Not Serif (skipped for symbol)
        assert not (flags & 64)  # Not Italic (skipped for symbol)

    def test_flags_fixed_pitch_works_for_symbol_fonts(self):
        """FixedPitch flag can be set even for symbol fonts."""
        mock_font = self._create_mock_font(is_fixed_pitch=1)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_font, is_symbol=True)

        assert flags & 4  # Symbolic
        assert flags & 1  # FixedPitch (applies to all fonts)

    def test_flags_no_os2_table(self):
        """Handles missing OS/2 table gracefully."""
        mock_tt_font = MagicMock()
        mock_tt_font.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "post": MagicMock(italicAngle=0, isFixedPitch=0),
            }.get(key)
        )
        mock_tt_font.__contains__ = MagicMock(side_effect=lambda key: key == "post")
        mock_tt_font.get = MagicMock(return_value=None)

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)
        flags = embedder._metrics._compute_font_flags(mock_tt_font)

        # Should only have Nonsymbolic flag
        assert flags & 32  # Nonsymbolic
        assert not (flags & 2)  # No Serif
        assert not (flags & 64)  # No Italic


class TestIsSymbolicFont:
    """Tests for the is_symbolic_font helper."""

    def test_symbolic_flag_set(self):
        """Font with Symbolic flag (bit 3) is detected as symbolic."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Symbol"),
            FontDescriptor=Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/Symbol"),
                Flags=4,  # Symbolic bit
            ),
        )
        assert is_symbolic_font(font_dict) is True

    def test_nonsymbolic_flag(self):
        """Font without Symbolic flag is not symbolic."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Arial"),
            FontDescriptor=Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/Arial"),
                Flags=32,  # Nonsymbolic bit only
            ),
        )
        assert is_symbolic_font(font_dict) is False

    def test_no_font_descriptor(self):
        """Font without FontDescriptor is not symbolic."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )
        assert is_symbolic_font(font_dict) is False

    def test_no_flags(self):
        """Font with FontDescriptor but no Flags is not symbolic."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Arial"),
            FontDescriptor=Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/Arial"),
            ),
        )
        assert is_symbolic_font(font_dict) is False

    def test_combined_flags_with_symbolic(self):
        """Font with multiple flags including Symbolic is detected."""
        # Flags = 36 = Symbolic (4) + Nonsymbolic (32) — unusual but tests bit check
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/CustomFont"),
            FontDescriptor=Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/CustomFont"),
                Flags=36,
            ),
        )
        assert is_symbolic_font(font_dict) is True
