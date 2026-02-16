# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for font subsetting."""

import re
from io import BytesIO

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Stream

from pdftopdfa.fonts.subsetter import (
    FontSubsetter,
    SubsettingResult,
    _build_glyphnames_from_unicode,
    _check_subsetting_allowed,
    _clean_tounicode,
    _find_font_file,
    _generate_subset_prefix,
    _is_subset_font,
    _populate_from_encoding,
    _resolve_simple_font_encoding,
    _subset_font_data,
)
from pdftopdfa.fonts.tounicode import (
    generate_cidfont_tounicode_cmap,
    generate_tounicode_cmap_data,
    parse_tounicode_cmap,
    validate_tounicode_cmap,
)


def _load_liberation_sans() -> bytes:
    """Loads LiberationSans-Regular.ttf for testing."""
    from importlib import resources

    font_path = (
        resources.files("pdftopdfa")
        / "resources"
        / "fonts"
        / "LiberationSans-Regular.ttf"
    )
    return font_path.read_bytes()


def _make_embedded_truetype_font(pdf, font_name, font_data):
    """Helper: creates an embedded TrueType font with FontFile2.

    Returns the indirect font object.
    """
    font_stream = Stream(pdf, font_data)
    font_stream[Name.Length1] = len(font_data)

    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=32,
            FontBBox=Array([-500, -300, 1300, 1000]),
            ItalicAngle=0,
            Ascent=900,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name(f"/{font_name}"),
        FirstChar=0,
        LastChar=255,
        Widths=Array([600] * 256),
        Encoding=Name.WinAnsiEncoding,
        FontDescriptor=font_descriptor,
    )

    return pdf.make_indirect(font)


def _make_embedded_cidfont(pdf, font_name, font_data):
    """Helper: creates an embedded CIDFont (Type0) with FontFile2.

    Returns the indirect font object.
    """
    font_stream = Stream(pdf, font_data)
    font_stream[Name.Length1] = len(font_data)

    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=4,
            FontBBox=Array([-500, -300, 1300, 1000]),
            ItalicAngle=0,
            Ascent=900,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )
    )

    desc_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name(f"/{font_name}"),
            CIDToGIDMap=Name.Identity,
            FontDescriptor=font_descriptor,
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
        )
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type0,
        BaseFont=Name(f"/{font_name}"),
        Encoding=Name("/Identity-H"),
        DescendantFonts=Array([desc_font]),
    )

    return pdf.make_indirect(font)


class TestIsSubsetFont:
    """Tests for _is_subset_font."""

    def test_subset_prefix(self):
        """Detects 6-letter uppercase prefix + plus sign."""
        assert _is_subset_font("ABCDEF+Arial") is True

    def test_no_prefix(self):
        """Regular font names are not subsets."""
        assert _is_subset_font("Arial") is False

    def test_short_prefix(self):
        """Prefix shorter than 6 chars is not a subset tag."""
        assert _is_subset_font("ABC+Arial") is False

    def test_lowercase_prefix(self):
        """Lowercase prefix is not a subset tag."""
        assert _is_subset_font("abcdef+Arial") is False

    def test_mixed_case_prefix(self):
        """Mixed case prefix is not a subset tag."""
        assert _is_subset_font("AbCdEf+Arial") is False


class TestGenerateSubsetPrefix:
    """Tests for _generate_subset_prefix."""

    def test_format(self):
        """Prefix is 6 uppercase letters followed by +."""
        prefix = _generate_subset_prefix()
        assert re.match(r"^[A-Z]{6}\+$", prefix)

    def test_uniqueness(self):
        """Consecutive calls produce different prefixes (with high probability)."""
        prefixes = {_generate_subset_prefix() for _ in range(100)}
        # With 26^6 possibilities, 100 should all be unique
        assert len(prefixes) == 100


class TestSubsetFontData:
    """Tests for _subset_font_data."""

    def test_subset_reduces_size(self):
        """Subsetting with few glyphs produces smaller data."""
        font_data = _load_liberation_sans()
        # Only use a few character codes
        used_codes = {65, 66, 67}  # A, B, C

        result = _subset_font_data(font_data, used_codes, is_cid=False)

        assert result is not None
        assert len(result) < len(font_data)

    def test_subset_cid_mode(self):
        """CID mode subsetting works with GID-based codes."""
        font_data = _load_liberation_sans()
        # Use a few GIDs
        used_codes = {36, 37, 38}

        result = _subset_font_data(font_data, used_codes, is_cid=True)

        assert result is not None
        assert len(result) < len(font_data)

    def test_subset_empty_usage(self):
        """Subsetting with no usage still produces valid font (.notdef)."""
        font_data = _load_liberation_sans()

        result = _subset_font_data(font_data, set(), is_cid=False)

        assert result is not None
        # Should still be smaller (only .notdef retained)
        assert len(result) < len(font_data)

    def test_subset_preserves_gids(self):
        """retain_gids=True means glyph order is preserved."""
        from fontTools.ttLib import TTFont

        font_data = _load_liberation_sans()
        original_font = TTFont(BytesIO(font_data))
        original_font.close()

        used_codes = {65, 66}  # A, B
        result = _subset_font_data(font_data, used_codes, is_cid=False)

        subsetted_font = TTFont(BytesIO(result))
        subsetted_order = subsetted_font.getGlyphOrder()
        subsetted_font.close()

        # With retain_gids=True, the subsetted font should have at least
        # as many glyph slots as the highest retained GID
        assert len(subsetted_order) > 1  # More than just .notdef


class TestFontSubsetter:
    """Tests for FontSubsetter."""

    def test_subset_simple_font(self):
        """Subsets a simple TrueType font and adds prefix."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "LiberationSans", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert "LiberationSans" in result.fonts_subsetted[0]
        assert result.bytes_saved > 0

        # Check BaseFont has subset prefix
        new_name = str(font_obj.get("/BaseFont"))[1:]  # Remove "/"
        assert "+" in new_name
        prefix = new_name.split("+")[0]
        assert len(prefix) == 6
        assert prefix.isupper()

    def test_subset_cidfont(self):
        """Subsets a CIDFont (Type0) and adds prefix."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_cidfont(pdf, "TestCID", font_data)
        font_dict = Dictionary(F1=font_obj)

        # CID content with 2-byte codes
        content = b"BT /F1 12 Tf <00410042> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

    def test_skip_type3_font(self):
        """Type3 fonts are skipped."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
        )
        char_procs = Dictionary()
        font[Name("/CharProcs")] = char_procs
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("Type3" in s for s in result.fonts_skipped)

    def test_skip_already_subsetted_font(self):
        """Fonts with ABCDEF+ prefix are skipped."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "ABCDEF+LiberationSans", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("already subsetted" in s for s in result.fonts_skipped)

    def test_skip_font_without_fontfile2(self):
        """Fonts without FontFile2 are skipped."""
        pdf = new_pdf()

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("no FontFile2 or FontFile3" in s for s in result.fonts_skipped)

    def test_skip_non_embedded_font(self):
        """Non-embedded fonts (no FontDescriptor) are skipped."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Helvetica"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0

    def test_shared_font_across_pages(self):
        """Font shared across pages is only subsetted once."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "SharedFont", font_data)

        for text in [b"(AB)", b"(CD)"]:
            font_dict = Dictionary(F1=font_obj)
            content = b"BT /F1 12 Tf " + text + b" Tj ET"
            page_dict = Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=font_dict),
                Contents=pdf.make_stream(content),
            )
            pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

    def test_subsetting_result_dataclass(self):
        """SubsettingResult initializes with empty defaults."""
        result = SubsettingResult()
        assert result.fonts_subsetted == []
        assert result.fonts_skipped == []
        assert result.warnings == []
        assert result.bytes_saved == 0

    def test_fonttools_error_handled(self):
        """fontTools errors are caught and reported as warnings."""
        pdf = new_pdf()

        # Create a font with invalid font data
        invalid_data = b"not a real font"
        font_stream = Stream(pdf, invalid_data)
        font_stream[Name.Length1] = len(invalid_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BadFont"),
                Flags=32,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/BadFont"),
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        # Should not crash, font should be skipped or warned
        assert len(result.fonts_subsetted) == 0


def _make_cff_otf_font_data() -> bytes:
    """Creates a minimal CFF-based OpenType font programmatically."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.misc.psCharStrings import T2CharString

    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder([".notdef", "A", "B", "C", "space"])
    fb.setupCharacterMap({65: "A", 66: "B", 67: "C", 32: "space"})
    fb.setupHorizontalMetrics(
        {
            ".notdef": (500, 0),
            "A": (600, 0),
            "B": (600, 0),
            "C": (600, 0),
            "space": (250, 0),
        }
    )
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestCFF", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()

    # Build T2CharString objects with minimal programs
    charstrings = {}
    for gn in [".notdef", "A", "B", "C", "space"]:
        cs = T2CharString()
        cs.program = ["endchar"]
        charstrings[gn] = cs

    fb.setupCFF("TestCFF", {}, charstrings, {})

    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


def _make_embedded_cff_otf_font(pdf, font_name, font_data):
    """Helper: creates a simple font with FontFile3 /OpenType.

    Returns the indirect font object.
    """
    font_stream = Stream(pdf, font_data)
    font_stream[Name.Subtype] = Name.OpenType

    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=32,
            FontBBox=Array([-500, -300, 1300, 1000]),
            ItalicAngle=0,
            Ascent=900,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile3=pdf.make_indirect(font_stream),
        )
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name(f"/{font_name}"),
        FirstChar=0,
        LastChar=255,
        Widths=Array([600] * 256),
        Encoding=Name.WinAnsiEncoding,
        FontDescriptor=font_descriptor,
    )

    return pdf.make_indirect(font)


def _make_embedded_cff_otf_cidfont(pdf, font_name, font_data):
    """Helper: creates a CIDFont (Type0) with FontFile3 /OpenType.

    Returns the indirect font object.
    """
    font_stream = Stream(pdf, font_data)
    font_stream[Name.Subtype] = Name.OpenType

    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=4,
            FontBBox=Array([-500, -300, 1300, 1000]),
            ItalicAngle=0,
            Ascent=900,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile3=pdf.make_indirect(font_stream),
        )
    )

    desc_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType0"),
            BaseFont=Name(f"/{font_name}"),
            CIDToGIDMap=Name.Identity,
            FontDescriptor=font_descriptor,
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
        )
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type0,
        BaseFont=Name(f"/{font_name}"),
        Encoding=Name("/Identity-H"),
        DescendantFonts=Array([desc_font]),
    )

    return pdf.make_indirect(font)


class TestSubsetFontDataCFF:
    """Tests for _subset_font_data with CFF/OpenType fonts."""

    def test_cff_otf_reduces_size(self):
        """Subsetting a CFF OTF with few glyphs produces smaller data."""
        font_data = _make_cff_otf_font_data()
        used_codes = {65}  # A

        result = _subset_font_data(font_data, used_codes, is_cid=False)

        assert result is not None
        assert len(result) < len(font_data)

    def test_cff_otf_preserves_otto_signature(self):
        """Subsetted CFF OTF still has OTTO signature."""
        font_data = _make_cff_otf_font_data()
        assert font_data[:4] == b"OTTO"

        result = _subset_font_data(font_data, {65}, is_cid=False)

        assert result is not None
        assert result[:4] == b"OTTO"

    def test_cff_otf_cid_mode(self):
        """CID mode subsetting works with CFF OTF."""
        font_data = _make_cff_otf_font_data()
        used_codes = {1, 2}  # GIDs

        result = _subset_font_data(font_data, used_codes, is_cid=True)

        assert result is not None
        assert len(result) < len(font_data)


class TestFontSubsetterFontFile3:
    """Tests for FontSubsetter with FontFile3 (CFF/OpenType) fonts."""

    def test_subset_simple_font_fontfile3(self):
        """Subsets a simple font with FontFile3 /OpenType."""
        pdf = new_pdf()
        font_data = _make_cff_otf_font_data()

        font_obj = _make_embedded_cff_otf_font(pdf, "TestCFF", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert "TestCFF" in result.fonts_subsetted[0]
        assert result.bytes_saved > 0

        # Check BaseFont has subset prefix
        new_name = str(font_obj.get("/BaseFont"))[1:]
        assert "+" in new_name

        # FontFile3 key should be preserved (not FontFile2)
        fd = font_obj.get("/FontDescriptor")
        assert fd.get("/FontFile3") is not None
        assert fd.get("/FontFile2") is None

        # /Subtype /OpenType should be preserved on the stream
        ff3_stream = fd.get("/FontFile3")
        assert str(ff3_stream.get("/Subtype")) == "/OpenType"

        # No Length1 on FontFile3
        assert ff3_stream.get("/Length1") is None

    def test_subset_cidfont_fontfile3(self):
        """Subsets a CIDFont (Type0) with FontFile3 /OpenType."""
        pdf = new_pdf()
        font_data = _make_cff_otf_font_data()

        font_obj = _make_embedded_cff_otf_cidfont(pdf, "TestCIDCFF", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf <00410042> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

        # FontFile3 key should be preserved
        descendants = font_obj.get("/DescendantFonts")
        desc_font = descendants[0]
        fd = desc_font.get("/FontDescriptor")
        assert fd.get("/FontFile3") is not None
        assert fd.get("/FontFile2") is None


class TestFindFontFile:
    """Tests for _find_font_file helper."""

    def test_finds_fontfile2(self):
        """Returns FontFile2 info when present."""
        pdf = new_pdf()
        font_stream = Stream(pdf, b"fake ttf data")
        font_stream[Name.Length1] = 13

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontFile2=pdf.make_indirect(font_stream),
        )
        fd = pdf.make_indirect(fd)

        info = _find_font_file(fd)
        assert info is not None
        assert info.descriptor_key == Name.FontFile2
        assert info.is_fontfile3 is False

    def test_finds_fontfile3_opentype(self):
        """Returns FontFile3 info when /Subtype /OpenType."""
        pdf = new_pdf()
        font_stream = Stream(pdf, b"fake otf data")
        font_stream[Name.Subtype] = Name.OpenType

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontFile3=pdf.make_indirect(font_stream),
        )
        fd = pdf.make_indirect(fd)

        info = _find_font_file(fd)
        assert info is not None
        assert info.descriptor_key == Name.FontFile3
        assert info.is_fontfile3 is True

    def test_prefers_fontfile2_over_fontfile3(self):
        """FontFile2 is preferred when both are present."""
        pdf = new_pdf()
        ff2_stream = Stream(pdf, b"ttf data")
        ff2_stream[Name.Length1] = 8

        ff3_stream = Stream(pdf, b"otf data")
        ff3_stream[Name.Subtype] = Name.OpenType

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontFile2=pdf.make_indirect(ff2_stream),
            FontFile3=pdf.make_indirect(ff3_stream),
        )
        fd = pdf.make_indirect(fd)

        info = _find_font_file(fd)
        assert info is not None
        assert info.descriptor_key == Name.FontFile2
        assert info.is_fontfile3 is False

    def test_returns_none_no_font_file(self):
        """Returns None when no font file is present."""
        pdf = new_pdf()

        fd = Dictionary(Type=Name.FontDescriptor)
        fd = pdf.make_indirect(fd)

        info = _find_font_file(fd)
        assert info is None

    def test_skips_fontfile3_non_opentype(self):
        """Returns None for FontFile3 with non-OpenType subtype."""
        pdf = new_pdf()
        font_stream = Stream(pdf, b"bare cff data")
        font_stream[Name.Subtype] = Name("/CIDFontType0C")

        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontFile3=pdf.make_indirect(font_stream),
        )
        fd = pdf.make_indirect(fd)

        info = _find_font_file(fd)
        assert info is None


class TestFontFile3BareCFF:
    """Tests that bare CFF programs (not full OTF) skip gracefully."""

    def test_bare_cidfonttype0c_skipped(self):
        """FontFile3 with /CIDFontType0C is skipped (not crashed)."""
        pdf = new_pdf()

        bare_cff_data = b"\x01\x00\x04\x01" + b"\x00" * 100
        font_stream = Stream(pdf, bare_cff_data)
        font_stream[Name.Subtype] = Name("/CIDFontType0C")

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BareCFF"),
                Flags=4,
                FontFile3=pdf.make_indirect(font_stream),
            )
        )

        desc_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType0"),
                BaseFont=Name("/BareCFF"),
                FontDescriptor=font_descriptor,
                CIDSystemInfo=Dictionary(
                    Registry=pikepdf.String("Adobe"),
                    Ordering=pikepdf.String("Identity"),
                    Supplement=0,
                ),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/BareCFF"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([desc_font]),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf <0041> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        # Should be skipped, not crashed
        assert len(result.fonts_subsetted) == 0
        assert any("no FontFile2 or FontFile3" in s for s in result.fonts_skipped)

    def test_bare_type1c_skipped(self):
        """FontFile3 with /Type1C is skipped (not crashed)."""
        pdf = new_pdf()

        bare_cff_data = b"\x01\x00\x04\x01" + b"\x00" * 100
        font_stream = Stream(pdf, bare_cff_data)
        font_stream[Name.Subtype] = Name("/Type1C")

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BareType1C"),
                Flags=32,
                FontFile3=pdf.make_indirect(font_stream),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/BareType1C"),
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        # Should be skipped, not crashed
        assert len(result.fonts_subsetted) == 0
        assert any("no FontFile2 or FontFile3" in s for s in result.fonts_skipped)


def _make_cff_otf_no_cmap_data() -> bytes:
    """Creates a CFF-based OpenType font WITHOUT a cmap table.

    This simulates a bare CFF wrapped in an OpenType container where
    the cmap table is absent — a rare edge case that cannot be subset
    via unicode-based population.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.misc.psCharStrings import T2CharString

    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder([".notdef", "A", "B", "C", "space"])
    fb.setupCharacterMap({65: "A", 66: "B", 67: "C", 32: "space"})
    fb.setupHorizontalMetrics(
        {
            ".notdef": (500, 0),
            "A": (600, 0),
            "B": (600, 0),
            "C": (600, 0),
            "space": (250, 0),
        }
    )
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestNoCmap", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()

    charstrings = {}
    for gn in [".notdef", "A", "B", "C", "space"]:
        cs = T2CharString()
        cs.program = ["endchar"]
        charstrings[gn] = cs
    fb.setupCFF("TestNoCmap", {}, charstrings, {})

    # Remove the cmap table
    del fb.font["cmap"]

    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


class TestCFFNoCmapSubsetting:
    """CFF-in-OpenType fonts without cmap are skipped gracefully."""

    def test_simple_font_cff_no_cmap_skipped(self):
        """Simple font with CFF OTF (no cmap, no encoding) is skipped."""
        pdf = new_pdf()
        font_data = _make_cff_otf_no_cmap_data()

        # Embed as simple font with FontFile3 /OpenType, no Encoding
        font_stream = Stream(pdf, font_data)
        font_stream[Name.Subtype] = Name.OpenType

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/NoCmapCFF"),
                Flags=32,
                FontBBox=Array([-500, -300, 1300, 1000]),
                ItalicAngle=0,
                Ascent=900,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile3=pdf.make_indirect(font_stream),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/NoCmapCFF"),
            FirstChar=0,
            LastChar=255,
            Widths=Array([600] * 256),
            FontDescriptor=font_descriptor,
            # No Encoding
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        # Should be skipped (subsetting failed), not crash
        assert len(result.fonts_subsetted) == 0
        assert any("subsetting failed" in s for s in result.fonts_skipped)

        # Font name should NOT have a subset prefix
        base_font = str(font_obj.get("/BaseFont"))
        assert "+" not in base_font

    def test_cff_no_cmap_with_encoding_also_skipped(self):
        """CFF OTF without cmap is skipped even with encoding.

        fontTools requires a cmap table internally for subsetting,
        so fonts without one cannot be subset regardless of PDF encoding.
        """
        pdf = new_pdf()
        font_data = _make_cff_otf_no_cmap_data()

        font_stream = Stream(pdf, font_data)
        font_stream[Name.Subtype] = Name.OpenType

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/NoCmapWithEnc"),
                Flags=32,
                FontBBox=Array([-500, -300, 1300, 1000]),
                ItalicAngle=0,
                Ascent=900,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile3=pdf.make_indirect(font_stream),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/NoCmapWithEnc"),
            FirstChar=0,
            LastChar=255,
            Widths=Array([600] * 256),
            Encoding=Name.WinAnsiEncoding,
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("subsetting failed" in s for s in result.fonts_skipped)

    def test_subset_font_data_no_cmap_returns_none(self):
        """_subset_font_data returns None for font without cmap."""
        font_data = _make_cff_otf_no_cmap_data()
        used_codes = {65, 66}

        result = _subset_font_data(font_data, used_codes, is_cid=False)

        assert result is None

    def test_subset_font_data_no_cmap_cid_returns_none(self):
        """_subset_font_data CID mode also returns None without cmap.

        fontTools requires cmap internally for all subsetting paths.
        """
        font_data = _make_cff_otf_no_cmap_data()
        used_codes = {1, 2}  # GIDs

        result = _subset_font_data(font_data, used_codes, is_cid=True)

        assert result is None


class TestFontEmbedderSubset:
    """Tests for FontEmbedder.subset_embedded_fonts integration."""

    def test_subset_via_embedder(self):
        """FontEmbedder.subset_embedded_fonts delegates to FontSubsetter."""
        from pdftopdfa.fonts.embedder import FontEmbedder

        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "LiberationSans", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (Test) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with FontEmbedder(pdf) as embedder:
            result = embedder.subset_embedded_fonts()

        assert isinstance(result, SubsettingResult)
        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0


class TestParseToUnicodeCmap:
    """Tests for parse_tounicode_cmap."""

    def test_parse_8bit_bfchar(self):
        """Parses 8-bit bfchar entries."""
        mapping = {65: 0x0041, 66: 0x0042, 67: 0x0043}
        cmap_data = generate_tounicode_cmap_data(mapping)

        result = parse_tounicode_cmap(cmap_data)

        assert result == mapping

    def test_parse_16bit_bfchar(self):
        """Parses 16-bit bfchar entries."""
        mapping = {0x0041: 0x0041, 0x0042: 0x0042}
        cmap_data = generate_cidfont_tounicode_cmap(mapping)

        result = parse_tounicode_cmap(cmap_data)

        assert result == mapping

    def test_parse_bfrange(self):
        """Parses bfrange entries."""
        cmap_data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"1 begincodespacerange\n"
            b"<00> <FF>\n"
            b"endcodespacerange\n"
            b"1 beginbfrange\n"
            b"<41> <43> <0041>\n"
            b"endbfrange\n"
            b"endcmap\n"
        )

        result = parse_tounicode_cmap(cmap_data)

        assert result == {0x41: 0x0041, 0x42: 0x0042, 0x43: 0x0043}

    def test_parse_empty_returns_empty(self):
        """Empty input returns empty dict."""
        assert parse_tounicode_cmap(b"") == {}

    def test_parse_malformed_returns_empty(self):
        """Malformed CMap without valid entries returns empty dict."""
        assert parse_tounicode_cmap(b"not a cmap at all") == {}

    def test_parse_surrogate_pair(self):
        """Parses surrogate pair for Unicode > U+FFFF."""
        # U+1F600 = D83D DE00 as surrogate pair
        mapping = {0x01: 0x1F600}
        cmap_data = generate_tounicode_cmap_data(mapping)

        result = parse_tounicode_cmap(cmap_data)

        assert result == {0x01: 0x1F600}

    def test_roundtrip_8bit(self):
        """Roundtrip: generate then parse 8-bit CMap preserves mapping."""
        original = {32: 0x0020, 65: 0x0041, 97: 0x0061, 255: 0x00FF}
        cmap_data = generate_tounicode_cmap_data(original)
        parsed = parse_tounicode_cmap(cmap_data)

        assert parsed == original

    def test_roundtrip_16bit(self):
        """Roundtrip: generate then parse 16-bit CMap preserves mapping."""
        original = {1: 0x0041, 2: 0x0042, 100: 0x4E2D}
        cmap_data = generate_cidfont_tounicode_cmap(original)
        parsed = parse_tounicode_cmap(cmap_data)

        assert parsed == original


class TestValidateToUnicodeCmap:
    """Tests for validate_tounicode_cmap."""

    def test_valid_8bit_cmap(self):
        """Valid 8-bit CMap passes validation."""
        mapping = {65: 0x0041, 66: 0x0042}
        data = generate_tounicode_cmap_data(mapping)
        # Should not raise
        validate_tounicode_cmap(data)

    def test_valid_16bit_cmap(self):
        """Valid 16-bit CMap passes validation."""
        mapping = {0x0041: 0x0041, 0x0042: 0x0042}
        data = generate_cidfont_tounicode_cmap(mapping)
        validate_tounicode_cmap(data)

    def test_valid_large_mapping(self):
        """CMap with >100 entries (multiple chunks) passes validation."""
        mapping = {i: 0x0020 + i for i in range(250)}
        data = generate_tounicode_cmap_data(mapping)
        validate_tounicode_cmap(data)

    def test_valid_surrogate_pair(self):
        """CMap with surrogate pair entries passes validation."""
        mapping = {1: 0x1F600}
        data = generate_tounicode_cmap_data(mapping)
        validate_tounicode_cmap(data)

    def test_missing_begincmap(self):
        """Rejects CMap missing begincmap."""
        import pytest

        data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"/CIDSystemInfo <<\n"
            b"  /Registry (Adobe)\n"
            b"  /Ordering (UCS)\n"
            b"  /Supplement 0\n"
            b">> def\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"endcmap\n"
            b"CMapName currentdict /CMap defineresource pop\n"
            b"end\nend\n"
        )
        with pytest.raises(ValueError, match="begincmap"):
            validate_tounicode_cmap(data)

    def test_missing_endcmap(self):
        """Rejects CMap missing endcmap."""
        import pytest

        data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"/CIDSystemInfo <<\n"
            b"  /Registry (Adobe)\n"
            b"  /Ordering (UCS)\n"
            b"  /Supplement 0\n"
            b">> def\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"CMapName currentdict /CMap defineresource pop\n"
            b"end\nend\n"
        )
        with pytest.raises(ValueError, match="endcmap"):
            validate_tounicode_cmap(data)

    def test_missing_cidsysteminfo(self):
        """Rejects CMap missing /CIDSystemInfo."""
        import pytest

        data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"begincmap\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"endcmap\n"
            b"CMapName currentdict /CMap defineresource pop\n"
        )
        with pytest.raises(ValueError, match="CIDSystemInfo"):
            validate_tounicode_cmap(data)

    def test_wrong_bfchar_count(self):
        """Rejects bfchar block with wrong entry count."""
        import pytest

        data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"/CIDSystemInfo <<\n"
            b"  /Registry (Adobe)\n"
            b"  /Ordering (UCS)\n"
            b"  /Supplement 0\n"
            b">> def\n"
            b"/CMapName /Adobe-Identity-UCS def\n"
            b"/CMapType 2 def\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"3 beginbfchar\n"
            b"<41> <0041>\n"
            b"<42> <0042>\n"
            b"endbfchar\n"
            b"endcmap\n"
            b"CMapName currentdict /CMap defineresource pop\n"
            b"end\nend\n"
        )
        with pytest.raises(ValueError, match="declares 3.*contains 2"):
            validate_tounicode_cmap(data)

    def test_non_ascii_rejected(self):
        """Rejects CMap with non-ASCII bytes."""
        import pytest

        data = b"\xff\xfe invalid"
        with pytest.raises(ValueError, match="non-ASCII"):
            validate_tounicode_cmap(data)

    def test_wrong_codespacerange_count(self):
        """Rejects codespacerange with wrong declared count."""
        import pytest

        data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"/CIDSystemInfo <<\n"
            b"  /Registry (Adobe)\n"
            b"  /Ordering (UCS)\n"
            b"  /Supplement 0\n"
            b">> def\n"
            b"/CMapName /Adobe-Identity-UCS def\n"
            b"/CMapType 2 def\n"
            b"2 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"endcmap\n"
            b"CMapName currentdict /CMap defineresource pop\n"
            b"end\nend\n"
        )
        with pytest.raises(ValueError, match="codespacerange declares 2"):
            validate_tounicode_cmap(data)


class TestCleanToUnicode:
    """Tests for _clean_tounicode."""

    def test_simple_font_tounicode_cleaned(self):
        """After subsetting, simple font ToUnicode only has used codes."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "LiberationSans", font_data)

        # Add a ToUnicode with entries for A(65), B(66), C(67)
        mapping = {65: 0x0041, 66: 0x0042, 67: 0x0043}
        cmap_data = generate_tounicode_cmap_data(mapping)
        font_obj[Name.ToUnicode] = pdf.make_indirect(Stream(pdf, cmap_data))

        font_dict = Dictionary(F1=font_obj)
        # Content only uses A and B
        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1

        # Parse the cleaned ToUnicode
        tounicode_stream = font_obj.get("/ToUnicode")
        assert tounicode_stream is not None
        cleaned = parse_tounicode_cmap(bytes(tounicode_stream.read_bytes()))

        # Should only have codes 65 and 66 (A and B), not 67 (C)
        assert 65 in cleaned
        assert 66 in cleaned
        assert 67 not in cleaned

    def test_cidfont_tounicode_cleaned(self):
        """After subsetting, CIDFont ToUnicode only has used codes."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_cidfont(pdf, "TestCID", font_data)

        # Add a ToUnicode with entries for GIDs 0x41, 0x42, 0x43
        mapping = {0x0041: 0x0041, 0x0042: 0x0042, 0x0043: 0x0043}
        cmap_data = generate_cidfont_tounicode_cmap(mapping)
        font_obj[Name.ToUnicode] = pdf.make_indirect(Stream(pdf, cmap_data))

        font_dict = Dictionary(F1=font_obj)
        # Content uses codes 0x0041 and 0x0042 only
        content = b"BT /F1 12 Tf <00410042> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1

        # Parse the cleaned ToUnicode
        tounicode_stream = font_obj.get("/ToUnicode")
        assert tounicode_stream is not None
        cleaned = parse_tounicode_cmap(bytes(tounicode_stream.read_bytes()))

        # Should only have codes 0x41 and 0x42, not 0x43
        assert 0x0041 in cleaned
        assert 0x0042 in cleaned
        assert 0x0043 not in cleaned

    def test_font_without_tounicode_not_affected(self):
        """Font without ToUnicode is subsetted without errors."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_obj = _make_embedded_truetype_font(pdf, "LiberationSans", font_data)

        # No ToUnicode entry
        assert font_obj.get("/ToUnicode") is None

        font_dict = Dictionary(F1=font_obj)
        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        # Still no ToUnicode
        assert font_obj.get("/ToUnicode") is None

    def test_stale_entries_removed(self):
        """Stale ToUnicode entries for unused codes are removed."""
        pdf = new_pdf()

        # Build a large mapping with many entries
        full_mapping = {i: i for i in range(32, 128)}
        cmap_data = generate_tounicode_cmap_data(full_mapping)

        # Only codes 65 and 66 are "used"
        used_codes = {65, 66}

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_obj[Name.ToUnicode] = pdf.make_indirect(Stream(pdf, cmap_data))

        _clean_tounicode(font_obj, used_codes, is_cid=False, pdf=pdf)

        # Parse the cleaned ToUnicode
        tounicode_stream = font_obj.get("/ToUnicode")
        cleaned = parse_tounicode_cmap(bytes(tounicode_stream.read_bytes()))

        assert cleaned == {65: 65, 66: 66}

    def test_no_rewrite_when_all_entries_used(self):
        """ToUnicode is not rewritten when all entries are used."""
        pdf = new_pdf()

        mapping = {65: 0x0041, 66: 0x0042}
        cmap_data = generate_tounicode_cmap_data(mapping)

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        original_stream = pdf.make_indirect(Stream(pdf, cmap_data))
        font_obj[Name.ToUnicode] = original_stream

        used_codes = {65, 66}
        original_objgen = font_obj.get("/ToUnicode").objgen

        _clean_tounicode(font_obj, used_codes, is_cid=False, pdf=pdf)

        # Stream object should be unchanged (same objgen)
        assert font_obj.get("/ToUnicode").objgen == original_objgen


def _make_font_with_fstype(fstype_value: int) -> bytes:
    """Creates a minimal TrueType font with a specific fsType value."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A", "B", "space"])
    fb.setupCharacterMap({65: "A", 66: "B", 32: "space"})
    fb.setupGlyf(
        {
            ".notdef": Glyph(),
            "A": Glyph(),
            "B": Glyph(),
            "space": Glyph(),
        }
    )
    fb.setupHorizontalMetrics(
        {
            ".notdef": (500, 0),
            "A": (600, 0),
            "B": (600, 0),
            "space": (250, 0),
        }
    )
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestFS", "styleName": "Regular"})
    fb.setupOS2(fsType=fstype_value)
    fb.setupPost()

    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


class TestCheckSubsettingAllowed:
    """Tests for _check_subsetting_allowed."""

    def test_installable_font_allowed(self):
        """Font with fsType 0 (installable) is allowed."""
        font_data = _make_font_with_fstype(0x0000)
        result = SubsettingResult()
        assert _check_subsetting_allowed(font_data, "Test", result) is True
        assert result.warnings == []
        assert result.fonts_skipped == []

    def test_no_subsetting_bit_blocks(self):
        """Font with fsType 0x0100 (no subsetting) is blocked."""
        font_data = _make_font_with_fstype(0x0100)
        result = SubsettingResult()
        assert _check_subsetting_allowed(font_data, "Test", result) is False
        assert any("no subsetting" in s for s in result.fonts_skipped)
        assert any("No subsetting" in w for w in result.warnings)

    def test_restricted_license_blocks_subsetting(self):
        """Font with fsType 0x0002 (restricted) blocks subsetting."""
        font_data = _make_font_with_fstype(0x0002)
        result = SubsettingResult()
        assert _check_subsetting_allowed(font_data, "Test", result) is False
        assert any("Restricted License" in w for w in result.warnings)
        assert any("embedding not allowed" in s for s in result.fonts_skipped)

    def test_invalid_font_data_allows(self):
        """Invalid font data (no OS/2) defaults to allowing subsetting."""
        result = SubsettingResult()
        assert _check_subsetting_allowed(b"not a font", "Test", result) is True
        assert result.warnings == []


class TestFontSubsetterFsType:
    """Integration tests for fsType checking in FontSubsetter."""

    def test_no_subsetting_font_skipped(self):
        """Font with no-subsetting fsType is skipped by FontSubsetter."""
        pdf = new_pdf()
        font_data = _make_font_with_fstype(0x0100)

        font_obj = _make_embedded_truetype_font(pdf, "NoSubsetFont", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("no subsetting" in s for s in result.fonts_skipped)
        assert any("No subsetting" in w for w in result.warnings)

        # Font name should NOT have a subset prefix
        base_font = str(font_obj.get("/BaseFont"))
        assert "+" not in base_font

    def test_installable_font_subsetted(self):
        """Font with installable fsType is subsetted normally."""
        pdf = new_pdf()
        font_data = _make_font_with_fstype(0x0000)

        font_obj = _make_embedded_truetype_font(pdf, "InstallableFont", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

    def test_restricted_license_skips_subsetting(self):
        """Font with restricted license fsType is skipped with warning."""
        pdf = new_pdf()
        font_data = _make_font_with_fstype(0x0002)

        font_obj = _make_embedded_truetype_font(pdf, "RestrictedFont", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("Restricted License" in w for w in result.warnings)
        assert any("embedding not allowed" in s for s in result.fonts_skipped)

    def test_no_subsetting_cidfont_skipped(self):
        """CIDFont with no-subsetting fsType is skipped."""
        pdf = new_pdf()
        font_data = _make_font_with_fstype(0x0100)

        font_obj = _make_embedded_cidfont(pdf, "NoSubsetCID", font_data)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf <00410042> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 0
        assert any("no subsetting" in s for s in result.fonts_skipped)


class TestResolveSimpleFontEncoding:
    """Tests for _resolve_simple_font_encoding."""

    def test_winansi_encoding(self):
        """WinAnsiEncoding produces glyph name mapping."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=Name.WinAnsiEncoding,
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is not None
        assert result[65] == "A"
        assert result[32] == "space"

    def test_standard_encoding(self):
        """StandardEncoding produces correct glyph name mapping."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/StandardEncoding"),
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is not None
        assert result[65] == "A"
        assert result[225] == "AE"  # StandardEncoding specific

    def test_macroman_encoding(self):
        """MacRomanEncoding produces glyph name mapping."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/MacRomanEncoding"),
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is not None
        assert result[65] == "A"

    def test_encoding_dict_with_differences(self):
        """Encoding dictionary with Differences overrides base entries."""
        pdf = new_pdf()
        enc_dict = Dictionary()
        enc_dict[Name("/BaseEncoding")] = Name.WinAnsiEncoding
        enc_dict[Name("/Differences")] = Array(
            [
                128,
                Name("/Euro"),
                Name("/ellipsis"),
            ]
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=enc_dict,
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is not None
        # Differences override
        assert result[128] == "Euro"
        assert result[129] == "ellipsis"
        # Base encoding still works for other codes
        assert result[65] == "A"

    def test_no_encoding_returns_none(self):
        """Font without /Encoding returns None."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is None

    def test_encoding_dict_no_base(self):
        """Encoding dict without BaseEncoding defaults to StandardEncoding."""
        pdf = new_pdf()
        enc_dict = Dictionary()
        enc_dict[Name("/Differences")] = Array(
            [
                200,
                Name("/fi"),
            ]
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            Encoding=enc_dict,
        )
        font_obj = pdf.make_indirect(font)

        result = _resolve_simple_font_encoding(font_obj)

        assert result is not None
        # Differences override
        assert result[200] == "fi"
        # Base is StandardEncoding
        assert result[65] == "A"


class TestBuildGlyphnamesFromUnicode:
    """Tests for _build_glyphnames_from_unicode."""

    def test_basic_ascii(self):
        """Maps ASCII Unicode values to standard glyph names."""
        mapping = {65: 0x0041, 66: 0x0042, 32: 0x0020}
        result = _build_glyphnames_from_unicode(mapping)

        assert result[65] == "A"
        assert result[66] == "B"
        assert result[32] == "space"

    def test_unknown_unicode_omitted(self):
        """Unicode values without AGL entry are omitted."""
        # U+FFFF has no standard glyph name
        mapping = {1: 0xFFFF}
        result = _build_glyphnames_from_unicode(mapping)

        assert 1 not in result

    def test_empty_input(self):
        """Empty mapping returns empty result."""
        assert _build_glyphnames_from_unicode({}) == {}


class TestPopulateFromEncoding:
    """Tests for _populate_from_encoding."""

    def test_direct_glyph_name_match(self):
        """Glyph names found in font use populate(glyphs=)."""
        from fontTools.fontBuilder import FontBuilder
        from fontTools.subset import Options, Subsetter
        from fontTools.ttLib.tables._g_l_y_f import Glyph

        fb = FontBuilder(1000, isTTF=True)
        fb.setupGlyphOrder([".notdef", "A", "B", "space"])
        fb.setupCharacterMap({65: "A", 66: "B", 32: "space"})
        fb.setupGlyf(
            {
                ".notdef": Glyph(),
                "A": Glyph(),
                "B": Glyph(),
                "space": Glyph(),
            }
        )
        fb.setupHorizontalMetrics(
            {
                ".notdef": (500, 0),
                "A": (600, 0),
                "B": (600, 0),
                "space": (250, 0),
            }
        )
        fb.setupHorizontalHeader(ascent=800, descent=-200)
        fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
        fb.setupOS2()
        fb.setupPost()

        tt_font = fb.font
        options = Options()
        options.retain_gids = True
        subsetter = Subsetter(options=options)

        # Encoding maps code 65 -> "A", code 32 -> "space"
        code_to_glyphname = {65: "A", 32: "space"}
        used_codes = {65, 32}

        _populate_from_encoding(subsetter, tt_font, used_codes, code_to_glyphname)
        subsetter.subset(tt_font)

        # Glyphs A and space should be retained
        glyph_order = tt_font.getGlyphOrder()
        assert "A" in glyph_order
        assert "space" in glyph_order
        tt_font.close()


class TestSubsetWithEncoding:
    """Integration tests for encoding-aware subsetting."""

    def test_subset_with_code_to_glyphname(self):
        """_subset_font_data with code_to_glyphname produces smaller font."""
        font_data = _load_liberation_sans()
        used_codes = {65, 66}  # A, B
        code_to_glyphname = {65: "A", 66: "B"}

        result = _subset_font_data(
            font_data,
            used_codes,
            is_cid=False,
            code_to_glyphname=code_to_glyphname,
        )

        assert result is not None
        assert len(result) < len(font_data)

    def test_subset_with_custom_differences(self):
        """Full integration: font with custom Differences is subsetted."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        # Create font with custom encoding + Differences
        enc_dict = Dictionary()
        enc_dict[Name("/BaseEncoding")] = Name.WinAnsiEncoding
        enc_dict[Name("/Differences")] = Array(
            [
                65,
                Name("/B"),  # Code 65 remapped to glyph "B"
            ]
        )

        font_stream = Stream(pdf, font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/CustomEnc"),
                Flags=32,
                FontBBox=Array([-500, -300, 1300, 1000]),
                ItalicAngle=0,
                Ascent=900,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/CustomEnc"),
            FirstChar=0,
            LastChar=255,
            Widths=Array([600] * 256),
            Encoding=enc_dict,
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # Content uses code 65 (which via Differences maps to glyph "B")
        content = b"BT /F1 12 Tf (A) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

    def test_subset_no_encoding_fallback(self):
        """Font without encoding falls back to codes-as-unicodes."""
        pdf = new_pdf()
        font_data = _load_liberation_sans()

        font_stream = Stream(pdf, font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/NoEnc"),
                Flags=32,
                FontBBox=Array([-500, -300, 1300, 1000]),
                ItalicAngle=0,
                Ascent=900,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        # No Encoding entry
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/NoEnc"),
            FirstChar=0,
            LastChar=255,
            Widths=Array([600] * 256),
            FontDescriptor=font_descriptor,
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0


def _make_symbolic_truetype_font_data() -> bytes:
    """Creates a symbolic TrueType font with (3,0) and (1,0) cmap subtables.

    The font contains glyphs mapped at 0xF020-0xF043 (Microsoft Symbol
    convention) and direct codes 0x20-0x43 (Mac convention).

    Returns:
        Serialized TrueType font bytes.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._c_m_a_p import (
        cmap_format_4,
        table__c_m_a_p,
    )
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    glyph_names = [".notdef", "space", "symbolA", "symbolB", "symbolC"]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap({0x20: "space"})
    fb.setupGlyf({name: Glyph() for name in glyph_names})
    metrics = {name: (500, 0) for name in glyph_names}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestSymbol", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    tt = fb.font

    # Replace cmap with (3,0) and (1,0) subtables
    cmap_table = table__c_m_a_p()
    cmap_table.tableVersion = 0

    # (3,0) Microsoft Symbol: codes at 0xF0XX
    subtable30 = cmap_format_4(4)
    subtable30.platformID = 3
    subtable30.platEncID = 0
    subtable30.format = 4
    subtable30.reserved = 0
    subtable30.length = 0
    subtable30.language = 0
    subtable30.cmap = {
        0xF020: "space",
        0xF041: "symbolA",
        0xF042: "symbolB",
        0xF043: "symbolC",
    }

    # (1,0) Mac Roman: direct codes
    subtable10 = cmap_format_4(4)
    subtable10.platformID = 1
    subtable10.platEncID = 0
    subtable10.format = 4
    subtable10.reserved = 0
    subtable10.length = 0
    subtable10.language = 0
    subtable10.cmap = {
        0x20: "space",
        0x41: "symbolA",
        0x42: "symbolB",
        0x43: "symbolC",
    }

    cmap_table.tables = [subtable10, subtable30]
    tt["cmap"] = cmap_table

    buf = BytesIO()
    tt.save(buf)
    tt.close()
    return buf.getvalue()


def _make_embedded_symbolic_font(pdf, font_name, font_data):
    """Creates an embedded symbolic TrueType font (Flags=4, no Encoding).

    Returns the indirect font object.
    """
    font_stream = Stream(pdf, font_data)
    font_stream[Name.Length1] = len(font_data)

    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=4,  # Symbolic
            FontBBox=Array([-500, -300, 1300, 1000]),
            ItalicAngle=0,
            Ascent=900,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name(f"/{font_name}"),
        FirstChar=0,
        LastChar=255,
        Widths=Array([500] * 256),
        FontDescriptor=font_descriptor,
        # No Encoding — symbolic font
    )

    return pdf.make_indirect(font)


class TestSymbolicTrueTypeSubsetting:
    """Symbolic TrueType fonts should preserve cmap during subsetting."""

    def test_symbolic_font_subsetted(self) -> None:
        """Symbolic TrueType font is subsetted successfully."""
        from fontTools.ttLib import TTFont

        font_data = _make_symbolic_truetype_font_data()
        pdf = new_pdf()
        font_obj = _make_embedded_symbolic_font(pdf, "TestSymbol", font_data)
        font_dict = Dictionary(F1=font_obj)
        content = b"BT /F1 12 Tf <4142> Tj ET"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=font_dict),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        subsetter = FontSubsetter(pdf)
        result = subsetter.subset_all_fonts()

        assert len(result.fonts_subsetted) == 1
        assert result.bytes_saved > 0

        # Verify the subsetted font has a (3,0) cmap subtable
        from pdftopdfa.utils import resolve_indirect

        font = resolve_indirect(font_obj)
        fd = resolve_indirect(font["/FontDescriptor"])
        ff2 = resolve_indirect(fd["/FontFile2"])
        new_data = bytes(ff2.read_bytes())

        tt = TTFont(BytesIO(new_data))
        cmap = tt["cmap"]
        has_30 = any(t.platformID == 3 and t.platEncID == 0 for t in cmap.tables)
        assert has_30, "Symbolic font must have (3,0) cmap after subsetting"

        for table in cmap.tables:
            if table.platformID == 3 and table.platEncID == 0:
                assert len(table.cmap) > 0
                break

        tt.close()

    def test_build_symbolic_truetype_encoding(self) -> None:
        """_build_symbolic_truetype_encoding returns code-to-glyph map."""
        from pdftopdfa.fonts.subsetter import (
            _build_symbolic_truetype_encoding,
        )

        font_data = _make_symbolic_truetype_font_data()
        pdf = new_pdf()
        font_obj = _make_embedded_symbolic_font(pdf, "TestSymbol", font_data)

        encoding = _build_symbolic_truetype_encoding(font_obj, font_data)

        assert encoding is not None
        assert encoding.get(0x41) == "symbolA"
        assert encoding.get(0x42) == "symbolB"
        assert encoding.get(0x20) == "space"

    def test_nonsymbolic_font_returns_none(self) -> None:
        """Non-symbolic font should return None."""
        from pdftopdfa.fonts.subsetter import (
            _build_symbolic_truetype_encoding,
        )

        font_data = _make_symbolic_truetype_font_data()
        pdf = new_pdf()
        font_obj = _make_embedded_truetype_font(pdf, "TestFont", font_data)

        encoding = _build_symbolic_truetype_encoding(font_obj, font_data)

        assert encoding is None

    def test_rebuild_symbolic_cmap(self) -> None:
        """_rebuild_symbolic_cmap restores (3,0) cmap after subsetting."""
        from fontTools.ttLib import TTFont

        from pdftopdfa.fonts.subsetter import (
            _rebuild_symbolic_cmap,
            _subset_font_data,
        )

        font_data = _make_symbolic_truetype_font_data()

        encoding = {0x41: "symbolA", 0x42: "symbolB", 0x20: "space"}
        used_codes = {0x41, 0x42, 0x20}
        subsetted = _subset_font_data(
            font_data,
            used_codes,
            is_cid=False,
            code_to_glyphname=encoding,
        )

        # Verify cmap is empty
        tt = TTFont(BytesIO(subsetted))
        assert len(tt["cmap"].tables) == 0
        tt.close()

        # Rebuild cmap
        fixed = _rebuild_symbolic_cmap(font_data, subsetted)
        assert fixed != subsetted

        # Verify (3,0) cmap is restored
        tt2 = TTFont(BytesIO(fixed))
        has_30 = any(t.platformID == 3 and t.platEncID == 0 for t in tt2["cmap"].tables)
        assert has_30
        tt2.close()
