# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/__init__.py — top-level font API."""

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts import (
    FontEmbedder,
    can_derive_unicode,
    get_fonts_missing_tounicode,
)


class TestFontsPreserved:
    """Tests for preservation of already embedded fonts."""

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_embedded_font_is_preserved(self):
        """Already embedded fonts are listed in fonts_preserved."""
        pdf = new_pdf()

        # Create embedded font (with FontDescriptor and FontFile2)
        # We use FontEmbedder itself to create a real embedded font
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

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # First embedding
        embedder = FontEmbedder(pdf)
        result1 = embedder.embed_missing_fonts()
        assert "Helvetica" in result1.fonts_embedded

        # Second embedding - Font should now be listed as preserved
        embedder2 = FontEmbedder(pdf)
        result2 = embedder2.embed_missing_fonts()
        assert "Helvetica" in result2.fonts_preserved
        assert "Helvetica" not in result2.fonts_embedded

    def test_preserved_fonts_not_modified(self):
        """Already embedded fonts are not modified."""
        pdf = new_pdf()

        # Create a simulated embedded font (valid TrueType signature)
        font_data = b"\x00\x01\x00\x00" + b"\x00" * 96
        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
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

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FontDescriptor=font_descriptor,
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
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

        # Get original FontDescriptor reference
        original_descriptor = (
            pdf.pages[0].get("/Resources")["/Font"]["/F1"].get("/FontDescriptor")
        )

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        # Font should be listed as preserved
        assert "TestFont" in result.fonts_preserved
        assert "TestFont" not in result.fonts_embedded

        # FontDescriptor should be unchanged
        updated_descriptor = (
            pdf.pages[0].get("/Resources")["/Font"]["/F1"].get("/FontDescriptor")
        )
        assert original_descriptor == updated_descriptor


class TestUnicodeDerivability:
    """Tests for can_derive_unicode() and unicode_derivable filtering."""

    def test_winansi_encoding_derivable(self):
        """Simple font with WinAnsiEncoding is derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=Name.WinAnsiEncoding,
        )
        assert can_derive_unicode(font_dict) is True

    def test_macroman_encoding_derivable(self):
        """Simple font with MacRomanEncoding is derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=Name.MacRomanEncoding,
        )
        assert can_derive_unicode(font_dict) is True

    def test_encoding_dict_with_agl_differences_derivable(self):
        """Encoding dict with AGL-resolvable Differences is derivable."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.WinAnsiEncoding,
            Differences=Array([65, Name.Aacute]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is True

    def test_encoding_dict_with_unknown_glyph_not_derivable(self):
        """Encoding dict with unresolvable glyph name is not derivable."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.WinAnsiEncoding,
            Differences=Array([65, Name("/glyphXYZ123")]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is False

    def test_cidfont_identity_cidtogidmap_derivable(self):
        """CIDFont with /CIDToGIDMap /Identity and embedded data is derivable."""
        pdf = new_pdf()

        font_stream = pdf.make_stream(b"\x00\x01\x00\x00" + b"\x00" * 96)
        font_stream[Name.Length1] = 100

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestCID"),
                Flags=4,
                FontBBox=Array([0, -200, 1000, 800]),
                ItalicAngle=0,
                Ascent=800,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/TestCID"),
                CIDToGIDMap=Name.Identity,
                FontDescriptor=font_descriptor,
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCID"),
            DescendantFonts=Array([descendant_font]),
        )

        assert can_derive_unicode(font_dict) is True

    def test_cidfont_without_cidtogidmap_not_derivable(self):
        """CIDFont without CIDToGIDMap is not derivable."""
        pdf = new_pdf()

        font_stream = pdf.make_stream(b"\x00\x01\x00\x00" + b"\x00" * 96)
        font_stream[Name.Length1] = 100

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestCID"),
                Flags=4,
                FontBBox=Array([0, -200, 1000, 800]),
                ItalicAngle=0,
                Ascent=800,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        descendant_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.CIDFontType2,
                BaseFont=Name("/TestCID"),
                FontDescriptor=font_descriptor,
            )
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCID"),
            DescendantFonts=Array([descendant_font]),
        )

        assert can_derive_unicode(font_dict) is False

    def test_no_encoding_not_derivable(self):
        """Font without Encoding is not derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
        )
        assert can_derive_unicode(font_dict) is False

    def test_type3_with_standard_encoding_derivable(self):
        """Type3 font with WinAnsiEncoding is derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=Name.WinAnsiEncoding,
        )
        assert can_derive_unicode(font_dict) is True

    def test_type3_without_encoding_not_derivable(self):
        """Type3 font without Encoding is not derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
        )
        assert can_derive_unicode(font_dict) is False

    def test_type3_with_custom_glyph_not_derivable(self):
        """Type3 font with non-AGL glyph names is not derivable."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name("/customglyph123")]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is False

    def test_notdef_in_differences_skipped(self):
        """The .notdef glyph in Differences is skipped (still derivable)."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.WinAnsiEncoding,
            Differences=Array([65, Name.Aacute, Name("/.notdef")]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is True

    def test_encoding_dict_without_base_encoding_derivable(self):
        """Encoding dict without BaseEncoding is derivable."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name.Aacute]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is True

    def test_uni_glyph_names_derivable(self):
        """uniXXXX glyph names in Differences are derivable."""
        encoding_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.WinAnsiEncoding,
            Differences=Array([65, Name("/uni00C9")]),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )
        assert can_derive_unicode(font_dict) is True

    def test_get_fonts_missing_tounicode_skips_derivable(self):
        """get_fonts_missing_tounicode() excludes derivable fonts."""
        pdf = new_pdf()

        # Embedded font with WinAnsiEncoding but no ToUnicode
        font_stream = pdf.make_stream(b"\x00\x01\x00\x00" + b"\x00" * 96)
        font_stream[Name.Length1] = 100

        font_descriptor = pdf.make_indirect(
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

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=font_descriptor,
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
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

        missing = get_fonts_missing_tounicode(pdf)
        assert missing == []

    def test_standard_encoding_derivable(self):
        """Simple font with explicit StandardEncoding is derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/StandardEncoding"),
        )
        assert can_derive_unicode(font_dict) is True

    def test_mmtype1_winansi_derivable(self):
        """MMType1 font with WinAnsiEncoding is derivable."""
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name("/MMType1"),
            BaseFont=Name("/TestMMFont"),
            Encoding=Name.WinAnsiEncoding,
        )
        assert can_derive_unicode(font_dict) is True
