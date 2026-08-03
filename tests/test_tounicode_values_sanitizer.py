# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for ToUnicode values sanitizer (veraPDF rule 6.2.11.7.2)."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts.tounicode import (
    _is_invalid_unicode,
    filter_invalid_unicode_values,
    generate_cidfont_tounicode_cmap,
    generate_tounicode_cmap_data,
    parse_tounicode_cmap,
    parse_tounicode_cmap_sequences,
    resolve_glyph_to_unicode,
)
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.tounicode_values import (
    fill_tounicode_gaps,
    sanitize_tounicode_values,
)


def _make_tounicode_cmap_8bit(code_to_unicode):
    """Builds a raw 8-bit ToUnicode CMap (no filtering)."""
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo <<",
        "  /Registry (Adobe)",
        "  /Ordering (UCS)",
        "  /Supplement 0",
        ">> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<00> <FF>",
        "endcodespacerange",
    ]
    sorted_codes = sorted(code_to_unicode.keys())
    chunk_size = 100
    for i in range(0, len(sorted_codes), chunk_size):
        chunk = sorted_codes[i : i + chunk_size]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            unicode_val = code_to_unicode[code]
            lines.append(f"<{code:02X}> <{unicode_val:04X}>")
        lines.append("endbfchar")
    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


def _make_tounicode_cmap_16bit(code_to_unicode):
    """Builds a raw 16-bit ToUnicode CMap (no filtering)."""
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo <<",
        "  /Registry (Adobe)",
        "  /Ordering (UCS)",
        "  /Supplement 0",
        ">> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    sorted_codes = sorted(code_to_unicode.keys())
    chunk_size = 100
    for i in range(0, len(sorted_codes), chunk_size):
        chunk = sorted_codes[i : i + chunk_size]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            unicode_val = code_to_unicode[code]
            lines.append(f"<{code:04X}> <{unicode_val:04X}>")
        lines.append("endbfchar")
    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


def _make_simple_font_with_tounicode(pdf, code_to_unicode):
    """Creates a simple TrueType font dict with a ToUnicode stream."""
    cmap_data = _make_tounicode_cmap_8bit(code_to_unicode)
    tounicode = pdf.make_stream(cmap_data)
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name("/TestFont"),
        FirstChar=0,
        LastChar=255,
        Encoding=Name.WinAnsiEncoding,
        ToUnicode=tounicode,
    )
    return font


def _make_cidfont_with_tounicode(pdf, code_to_unicode):
    """Creates a Type0/CIDFont dict with a 16-bit ToUnicode stream."""
    cmap_data = _make_tounicode_cmap_16bit(code_to_unicode)
    tounicode = pdf.make_stream(cmap_data)

    cidfont = Dictionary(
        Type=Name.Font,
        Subtype=Name("/CIDFontType2"),
        BaseFont=Name("/TestCIDFont"),
    )

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type0,
        BaseFont=Name("/TestCIDFont"),
        Encoding=Name("/Identity-H"),
        DescendantFonts=Array([cidfont]),
        ToUnicode=tounicode,
    )
    return font


def _make_page_with_font(pdf, font_dict):
    """Creates a page with a font resource."""
    content = b"BT /F1 12 Tf (A) Tj ET"
    stream = pdf.make_stream(content)
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
    return page


class TestFilterInvalidUnicodeValues:
    """Tests for filter_invalid_unicode_values()."""

    def test_u0000_replaced_with_pua(self):
        mapping = {65: 0x0041, 66: 0x0000}
        result = filter_invalid_unicode_values(mapping)
        assert result[65] == 0x0041
        assert 0xE000 <= result[66] <= 0xF8FF

    def test_ufeff_replaced_with_pua(self):
        mapping = {65: 0x0041, 66: 0xFEFF}
        result = filter_invalid_unicode_values(mapping)
        assert result[65] == 0x0041
        assert 0xE000 <= result[66] <= 0xF8FF

    def test_ufffe_replaced_with_pua(self):
        mapping = {65: 0x0041, 66: 0xFFFE}
        result = filter_invalid_unicode_values(mapping)
        assert result[65] == 0x0041
        assert 0xE000 <= result[66] <= 0xF8FF

    def test_all_valid_unchanged(self):
        mapping = {65: 0x0041, 66: 0x0042, 67: 0x0043}
        result = filter_invalid_unicode_values(mapping)
        assert result == mapping

    def test_pua_avoids_collisions(self):
        """Existing PUA values are not reused."""
        mapping = {65: 0xE000, 66: 0x0000}  # 0xE000 already used
        result = filter_invalid_unicode_values(mapping)
        assert result[65] == 0xE000
        # Should get 0xE001 or higher, not 0xE000
        assert result[66] >= 0xE001
        assert 0xE000 <= result[66] <= 0xF8FF

    def test_multiple_invalid_get_distinct_pua(self):
        mapping = {65: 0x0000, 66: 0xFEFF, 67: 0xFFFE}
        result = filter_invalid_unicode_values(mapping)
        pua_values = {result[65], result[66], result[67]}
        assert len(pua_values) == 3  # All distinct
        assert all(0xE000 <= v <= 0xF8FF for v in pua_values)

    def test_invalid_value_inside_sequence_is_replaced(self):
        mapping = {65: (0x0041, 0x0000), 66: 0xE000}

        result = filter_invalid_unicode_values(mapping)

        assert result[65] == (0x0041, 0xE001)
        assert result[66] == 0xE000

    def test_empty_mapping(self):
        result = filter_invalid_unicode_values({})
        assert result == {}


class TestSanitizeToUnicodeValues:
    """Tests for the sanitize_tounicode_values() sanitizer."""

    def test_invalid_mapping_in_soft_mask_font_is_fixed(self):
        """Fonts reachable only through an SMask group are sanitized."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0000})
        group = pdf.make_stream(b"BT /F1 12 Tf (A) Tj ET")
        group[Name.Type] = Name.XObject
        group[Name.Subtype] = Name.Form
        group[Name.BBox] = Array([0, 0, 10, 10])
        group[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(
                    ExtGState=Dictionary(
                        GS=Dictionary(
                            SMask=Dictionary(
                                S=Name("/Luminosity"),
                                G=group,
                            )
                        )
                    )
                ),
                Contents=pdf.make_stream(b"/GS gs"),
            )
        )
        pdf.pages.append(page)

        result = sanitize_tounicode_values(pdf)

        assert result["tounicode_values_fixed"] == 1
        mapping = parse_tounicode_cmap(bytes(font.ToUnicode.read_bytes()))
        assert 0xE000 <= mapping[65] <= 0xF8FF

    def test_pua_inside_unicode_sequence_is_not_reused(self):
        """Replacement values avoid PUA scalars in multi-codepoint mappings."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0xE000, 66: 0x0000})
        original = bytes(font["/ToUnicode"].read_bytes())
        font["/ToUnicode"].write(original.replace(b"<41> <E000>", b"<41> <0066E000>"))
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)

        assert result["tounicode_values_fixed"] == 1
        mapping = parse_tounicode_cmap_sequences(bytes(font["/ToUnicode"].read_bytes()))
        assert mapping[b"A"] == (ord("f"), 0xE000)
        assert mapping[b"B"] == (0xE001,)

    def test_tounicode_with_u0000_replaced(self):
        """U+0000 in existing ToUnicode is replaced with PUA."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0041, 66: 0x0000})
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1

        # Verify the new CMap no longer has U+0000
        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[65] == 0x0041
        assert new_mapping[66] != 0x0000
        assert 0xE000 <= new_mapping[66] <= 0xF8FF

    def test_tounicode_with_ufeff_replaced(self):
        """U+FEFF in existing ToUnicode is replaced with PUA."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0041, 66: 0xFEFF})
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[66] != 0xFEFF

    def test_tounicode_with_ufffe_replaced(self):
        """U+FFFE in existing ToUnicode is replaced with PUA."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0041, 66: 0xFFFE})
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[66] != 0xFFFE

    def test_tounicode_all_valid_unchanged(self):
        """Valid CMap is not modified."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0041, 66: 0x0042})
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 0

    def test_font_without_tounicode_skipped(self):
        """Font without ToUnicode is not modified."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=32,
            LastChar=114,
            Encoding=Name.WinAnsiEncoding,
        )
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 0

    def test_cidfont_16bit_cmap_fixed(self):
        """16-bit CMap on CIDFont is sanitized correctly."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041, 2: 0x0000, 3: 0xFEFF})
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[1] == 0x0041
        assert new_mapping[2] != 0x0000
        assert new_mapping[3] != 0xFEFF

    def test_valid_ligature_mapping_is_preserved(self):
        """Sanitizing another entry preserves multi-codepoint mappings."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {0: 0x0000, 72: 0x0041})
        cmap = bytes(font["/ToUnicode"].read_bytes()).replace(
            b"<0048> <0041>",
            b"<0048> <00660066>",
        )
        font["/ToUnicode"].write(cmap)
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)

        assert result["tounicode_values_fixed"] == 1
        new_data = bytes(font["/ToUnicode"].read_bytes())
        assert b"<0000> <0000>" not in new_data
        assert b"<0048> <00660066>" in new_data

    def test_incrementing_bfrange_only_replaces_invalid_values(self):
        """Valid destinations in an affected bfrange remain unchanged."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        cmap = bytes(font["/ToUnicode"].read_bytes()).replace(
            b"1 beginbfchar\n<0001> <0041>\nendbfchar",
            b"1 beginbfrange\n<0001> <0003> <0000>\nendbfrange",
        )
        font["/ToUnicode"].write(cmap)
        _make_page_with_font(pdf, font)

        result = sanitize_tounicode_values(pdf)

        assert result["tounicode_values_fixed"] == 1
        new_mapping = parse_tounicode_cmap(bytes(font["/ToUnicode"].read_bytes()))
        assert 0xE000 <= new_mapping[1] <= 0xF8FF
        assert new_mapping[2] == 0x0001
        assert new_mapping[3] == 0x0002

    def test_same_font_on_multiple_pages_fixed_once(self):
        """Same indirect font on two pages is only fixed once."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {65: 0x0041, 66: 0x0000})
        font_ref = pdf.make_indirect(font)

        for _ in range(2):
            content = b"BT /F1 12 Tf (A) Tj ET"
            stream = pdf.make_stream(content)
            page = pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 612, 792]),
                    Resources=Dictionary(
                        Font=Dictionary(F1=font_ref),
                    ),
                    Contents=stream,
                )
            )
            pdf.pages.append(page)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1


class TestResolveGlyphToUnicode:
    """Tests for resolve_glyph_to_unicode() rejecting invalid values."""

    def test_uni0000_returns_none(self):
        assert resolve_glyph_to_unicode("uni0000") is None

    def test_uni_feff_returns_none(self):
        assert resolve_glyph_to_unicode("uniFEFF") is None

    def test_uni_fffe_returns_none(self):
        assert resolve_glyph_to_unicode("uniFFFE") is None

    def test_u0000_returns_none(self):
        assert resolve_glyph_to_unicode("u0000") is None

    def test_u_feff_returns_none(self):
        assert resolve_glyph_to_unicode("uFEFF") is None

    def test_u_fffe_returns_none(self):
        assert resolve_glyph_to_unicode("uFFFE") is None

    def test_uni0041_returns_value(self):
        assert resolve_glyph_to_unicode("uni0041") == 0x0041

    def test_standard_glyph_name_unchanged(self):
        # 'A' is in AGL2UV → should return 0x0041
        assert resolve_glyph_to_unicode("A") == 0x0041


class TestGenerateCmapFiltersInvalid:
    """Tests that CMap generation filters invalid values."""

    def test_8bit_cmap_filters_u0000(self):
        mapping = {65: 0x0041, 66: 0x0000}
        cmap_data = generate_tounicode_cmap_data(mapping)
        parsed = parse_tounicode_cmap(cmap_data)
        assert parsed[65] == 0x0041
        assert parsed[66] != 0x0000
        assert 0xE000 <= parsed[66] <= 0xF8FF

    def test_8bit_cmap_preserves_unicode_sequences(self):
        cmap_data = generate_tounicode_cmap_data({65: (0x0066, 0x0066), 66: 0x100000})

        assert parse_tounicode_cmap_sequences(cmap_data) == {
            b"A": (0x0066, 0x0066),
            b"B": (0x100000,),
        }

    def test_16bit_cmap_filters_ufeff(self):
        mapping = {1: 0x0041, 2: 0xFEFF}
        cmap_data = generate_cidfont_tounicode_cmap(mapping)
        parsed = parse_tounicode_cmap(cmap_data)
        assert parsed[1] == 0x0041
        assert parsed[2] != 0xFEFF
        assert 0xE000 <= parsed[2] <= 0xF8FF


def _make_page_with_cidfont_content(pdf, font_dict, text_bytes):
    """Creates a page with a CIDFont and specific content stream bytes.

    The content stream uses the given raw bytes in a Tj operator.
    For CIDFonts, text_bytes should be 2-byte big-endian character codes.
    """
    # Build content stream: BT /F1 12 Tf <hex> Tj ET
    hex_str = text_bytes.hex().upper()
    content = f"BT /F1 12 Tf <{hex_str}> Tj ET".encode("ascii")
    stream = pdf.make_stream(content)
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
    return page


def _make_page_with_simple_font_content(pdf, font_dict, text_bytes):
    """Creates a page with a simple font and specific content stream bytes."""
    hex_str = text_bytes.hex().upper()
    content = f"BT /F1 12 Tf <{hex_str}> Tj ET".encode("ascii")
    stream = pdf.make_stream(content)
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
    return page


class TestFillToUnicodeGaps:
    """Tests for fill_tounicode_gaps() sanitizer."""

    def test_cidfont_gap_filled_with_pua(self):
        """CIDFont with a used CID not in ToUnicode gets PUA mapping."""
        pdf = new_pdf()
        # ToUnicode only maps CID 1 -> U+0041
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        # Content stream uses CID 1 and CID 5 (2-byte big-endian)
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[1] == 0x0041  # Preserved
        assert 5 in new_mapping  # Gap filled
        assert 0xE000 <= new_mapping[5] <= 0xF8FF  # PUA value

    def test_cidfont_no_gaps_unchanged(self):
        """CIDFont with all used CIDs mapped is not modified."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041, 5: 0x0042})
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 0

    def test_cidfont_multiple_gaps_filled(self):
        """Multiple unmapped CIDs all get distinct PUA values."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        # Content uses CIDs 1, 3, 5, 7
        _make_page_with_cidfont_content(
            pdf,
            font,
            b"\x00\x01\x00\x03\x00\x05\x00\x07",
        )

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[1] == 0x0041
        pua_values = {new_mapping[3], new_mapping[5], new_mapping[7]}
        assert len(pua_values) == 3  # All distinct
        assert all(0xE000 <= v <= 0xF8FF for v in pua_values)

    def test_simple_font_gap_filled(self):
        """Simple font with a used code not in ToUnicode gets PUA."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(
            pdf,
            {65: 0x0041},  # Only 'A' mapped
        )
        # Content uses codes 65 ('A') and 66 ('B') — 66 is unmapped
        _make_page_with_simple_font_content(pdf, font, bytes([65, 66]))

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[65] == 0x0041
        assert 66 in new_mapping
        assert 0xE000 <= new_mapping[66] <= 0xF8FF

    def test_existing_multicodepoint_mapping_is_preserved(self):
        """Adding a gap does not reserialize an existing Unicode sequence."""
        pdf = new_pdf()
        cmap = _make_tounicode_cmap_8bit({1: 0x0041}).replace(
            b"<01> <0041>",
            b"<01> <00660069>",
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=0,
            LastChar=255,
            Encoding=Name.WinAnsiEncoding,
            ToUnicode=pdf.make_stream(cmap),
        )
        _make_page_with_simple_font_content(pdf, font, b"\x01\x02")

        result = fill_tounicode_gaps(pdf)

        assert result["tounicode_gaps_filled"] == 1
        new_data = bytes(font.ToUnicode.read_bytes())
        assert b"<01> <00660069>" in new_data
        mapping = parse_tounicode_cmap_sequences(new_data)
        assert mapping[b"\x01"] == (ord("f"), ord("i"))
        assert 0xE000 <= mapping[b"\x02"][0] <= 0xF8FF

    def test_type0_one_byte_codespace_gap_is_filled(self):
        """Type0 strings are decoded using their declared CMap codespace."""
        pdf = new_pdf()
        encoding = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"1 begincidchar\n<01> 1\nendcidchar\nendcmap"
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=encoding,
            DescendantFonts=Array([]),
            ToUnicode=pdf.make_stream(_make_tounicode_cmap_8bit({1: 0x0041})),
        )
        _make_page_with_cidfont_content(pdf, font, b"\x01\x02")

        result = fill_tounicode_gaps(pdf)

        assert result["tounicode_gaps_filled"] == 1
        mapping = parse_tounicode_cmap_sequences(bytes(font.ToUnicode.read_bytes()))
        assert mapping[b"\x01"] == (0x0041,)
        assert 0xE000 <= mapping[b"\x02"][0] <= 0xF8FF

    def test_unicode_cmap_gap_preserves_space_semantics(self, caplog):
        """UCS-2 character codes retain their inherent Unicode meaning."""
        caplog.set_level("INFO")
        pdf = new_pdf()
        encoding = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            b"2 begincidchar\n<0020> 1\n<0041> 34\nendcidchar\nendcmap"
        )
        encoding[Name.CMapName] = Name("/UniJIS-UCS2-H")
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=encoding,
            DescendantFonts=Array([]),
            ToUnicode=pdf.make_stream(_make_tounicode_cmap_16bit({0x41: 0x41})),
        )
        _make_page_with_cidfont_content(pdf, font, b"\x00A\x00 ")

        result = fill_tounicode_gaps(pdf)

        assert result["tounicode_gaps_filled"] == 1
        mapping = parse_tounicode_cmap_sequences(bytes(font.ToUnicode.read_bytes()))
        assert mapping[b"\x00A"] == (ord("A"),)
        assert mapping[b"\x00 "] == (ord(" "),)
        assert "PUA codepoints" not in caplog.text

    def test_nested_form_pattern_and_appearance_usage_is_collected(self):
        """Gap collection covers nested rendered content streams."""
        pdf = new_pdf()
        font = pdf.make_indirect(_make_simple_font_with_tounicode(pdf, {1: 0x0041}))

        def resources():
            return Dictionary(Font=Dictionary(F1=font))

        form = pdf.make_stream(b"BT /F1 12 Tf <02> Tj ET")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 10, 10])
        form[Name.Resources] = resources()

        pattern = pdf.make_stream(b"BT /F1 12 Tf <03> Tj ET")
        pattern[Name.Type] = Name.Pattern
        pattern[Name.PatternType] = 1
        pattern[Name.PaintType] = 1
        pattern[Name.TilingType] = 1
        pattern[Name.BBox] = Array([0, 0, 10, 10])
        pattern[Name.XStep] = 10
        pattern[Name.YStep] = 10
        pattern[Name.Resources] = resources()

        appearance = pdf.make_stream(b"BT /F1 12 Tf <04> Tj ET")
        appearance[Name.Type] = Name.XObject
        appearance[Name.Subtype] = Name.Form
        appearance[Name.BBox] = Array([0, 0, 10, 10])
        appearance[Name.Resources] = resources()
        annotation = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 10, 10]),
                AP=Dictionary(N=appearance),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(
                    XObject=Dictionary(Fm=form),
                    Pattern=Dictionary(P1=pattern),
                ),
                Contents=pdf.make_stream(b"/Fm Do"),
                Annots=Array([annotation]),
            )
        )
        pdf.pages.append(page)

        result = fill_tounicode_gaps(pdf)

        assert result["tounicode_gaps_filled"] == 1
        mapping = parse_tounicode_cmap(bytes(font.ToUnicode.read_bytes()))
        assert {2, 3, 4} <= mapping.keys()

    def test_soft_mask_group_usage_is_collected(self):
        """Gap filling reaches fonts inside ExtGState soft-mask groups."""
        pdf = new_pdf()
        font = _make_simple_font_with_tounicode(pdf, {1: 0x0041})
        group = pdf.make_stream(b"BT /F1 12 Tf <0102> Tj ET")
        group[Name.Type] = Name.XObject
        group[Name.Subtype] = Name.Form
        group[Name.BBox] = Array([0, 0, 10, 10])
        group[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(
                    ExtGState=Dictionary(
                        GS=Dictionary(
                            SMask=Dictionary(
                                S=Name("/Luminosity"),
                                G=group,
                            )
                        )
                    )
                ),
                Contents=pdf.make_stream(b"/GS gs"),
            )
        )
        pdf.pages.append(page)

        result = fill_tounicode_gaps(pdf)

        assert result["tounicode_gaps_filled"] == 1
        mapping = parse_tounicode_cmap_sequences(bytes(font.ToUnicode.read_bytes()))
        assert mapping[b"\x01"] == (0x0041,)
        assert 0xE000 <= mapping[b"\x02"][0] <= 0xF8FF

    def test_font_without_tounicode_skipped(self):
        """Font without ToUnicode is not modified."""
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=32,
            LastChar=127,
            Encoding=Name.WinAnsiEncoding,
        )
        _make_page_with_simple_font_content(pdf, font, bytes([65]))

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 0
        assert font.get("/ToUnicode") is None

    def test_pua_avoids_collisions(self):
        """PUA assignments avoid existing PUA values in the mapping."""
        pdf = new_pdf()
        # Existing mapping already uses 0xE000
        font = _make_simple_font_with_tounicode(pdf, {65: 0xE000})
        _make_page_with_simple_font_content(pdf, font, bytes([65, 66]))

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert new_mapping[65] == 0xE000
        assert new_mapping[66] >= 0xE001  # Avoids collision

    def test_same_font_on_multiple_pages(self):
        """Same indirect font used on two pages has gaps filled once."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        font_ref = pdf.make_indirect(font)

        # Page 1: uses CID 1 (mapped) and CID 5 (unmapped)
        content1 = b"BT /F1 12 Tf <00010005> Tj ET"
        stream1 = pdf.make_stream(content1)
        page1 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font_ref)),
                Contents=stream1,
            )
        )
        pdf.pages.append(page1)

        # Page 2: uses CID 1 (mapped) and CID 7 (unmapped)
        content2 = b"BT /F1 12 Tf <00010007> Tj ET"
        stream2 = pdf.make_stream(content2)
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font_ref)),
                Contents=stream2,
            )
        )
        pdf.pages.append(page2)

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert 5 in new_mapping
        assert 7 in new_mapping

    def test_tj_array_operator(self):
        """TJ operator (array) with gaps is handled correctly."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})

        # TJ operator: array of strings and kerning adjustments
        content = b"BT /F1 12 Tf [<0001> 50 <0005>] TJ ET"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 1

        new_data = bytes(font["/ToUnicode"].read_bytes())
        new_mapping = parse_tounicode_cmap(new_data)
        assert 5 in new_mapping

    def test_empty_pdf_no_error(self):
        """Empty PDF with no pages does not crash."""
        pdf = new_pdf()
        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 0

    def test_page_without_fonts_skipped(self):
        """Page without font resources does not crash."""
        pdf = new_pdf()
        content = b"1 0 0 1 100 700 cm"
        stream = pdf.make_stream(content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=stream,
            )
        )
        pdf.pages.append(page)

        result = fill_tounicode_gaps(pdf)
        assert result["tounicode_gaps_filled"] == 0


class TestFillToUnicodeGapsIntegration:
    """Integration tests: fill_tounicode_gaps via sanitize_for_pdfa."""

    @pytest.mark.parametrize("level", ["2a", "2u"])
    def test_gaps_filled_at_pdfa2_unicode_levels(self, level):
        """PDF/A-2 Unicode and accessible levels fill ToUnicode gaps."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = sanitize_for_pdfa(pdf, level=level)
        assert result["tounicode_gaps_filled"] == 1

    @pytest.mark.parametrize("level", ["3a", "3u"])
    def test_gaps_filled_at_pdfa3_unicode_levels(self, level):
        """PDF/A-3 Unicode and accessible levels fill ToUnicode gaps."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = sanitize_for_pdfa(pdf, level=level)
        assert result["tounicode_gaps_filled"] == 1

    def test_gaps_not_filled_at_2b_level(self):
        """sanitize_for_pdfa at level '2b' does not fill ToUnicode gaps."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["tounicode_gaps_filled"] == 0

    def test_gaps_not_filled_at_3b_level(self):
        """sanitize_for_pdfa at level '3b' does not fill ToUnicode gaps."""
        pdf = new_pdf()
        font = _make_cidfont_with_tounicode(pdf, {1: 0x0041})
        _make_page_with_cidfont_content(pdf, font, b"\x00\x01\x00\x05")

        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["tounicode_gaps_filled"] == 0


class TestSurrogateCodePointRejection:
    """Tests for Unicode surrogate code point rejection (U+D800–U+DFFF)."""

    def test_is_invalid_unicode_detects_surrogates(self):
        """_is_invalid_unicode returns True for surrogate code points."""
        assert _is_invalid_unicode(0xD800)
        assert _is_invalid_unicode(0xDBFF)
        assert _is_invalid_unicode(0xDC00)
        assert _is_invalid_unicode(0xDFFF)
        assert _is_invalid_unicode(0xDA00)

    def test_is_invalid_unicode_allows_valid(self):
        """_is_invalid_unicode returns False for valid code points."""
        assert not _is_invalid_unicode(0x0041)  # 'A'
        assert not _is_invalid_unicode(0xD7FF)  # just before surrogates
        assert not _is_invalid_unicode(0xE000)  # PUA start (just after)
        assert not _is_invalid_unicode(0xFFFF)

    def test_is_invalid_unicode_original_values(self):
        """_is_invalid_unicode still detects the original invalid values."""
        assert _is_invalid_unicode(0x0000)
        assert _is_invalid_unicode(0xFEFF)
        assert _is_invalid_unicode(0xFFFE)

    def test_filter_replaces_surrogate_with_pua(self):
        """filter_invalid_unicode_values replaces surrogates with PUA."""
        mapping = {65: 0xD800, 66: 0x0042}
        result = filter_invalid_unicode_values(mapping)
        assert 0xE000 <= result[65] <= 0xF8FF
        assert result[66] == 0x0042

    def test_filter_replaces_multiple_surrogates(self):
        """Multiple surrogates get distinct PUA replacements."""
        mapping = {1: 0xD800, 2: 0xDC00, 3: 0xDBFF}
        result = filter_invalid_unicode_values(mapping)
        pua_values = {result[1], result[2], result[3]}
        assert len(pua_values) == 3  # all distinct
        assert all(0xE000 <= v <= 0xF8FF for v in pua_values)

    def test_sanitize_tounicode_fixes_surrogate_in_cmap(self):
        """sanitize_tounicode_values fixes surrogates in existing CMaps."""
        pdf = new_pdf()
        cmap = _make_tounicode_cmap_8bit({65: 0xD800, 66: 0x0042})
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            ToUnicode=pdf.make_indirect(pikepdf.Stream(pdf, cmap)),
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=pdf.make_indirect(font))),
            )
        )
        pdf.pages.append(page)

        result = sanitize_tounicode_values(pdf)
        assert result["tounicode_values_fixed"] == 1

        new_cmap = parse_tounicode_cmap(
            bytes(pdf.pages[0].Resources.Font.F1.ToUnicode.read_bytes())
        )
        assert 0xE000 <= new_cmap[65] <= 0xF8FF
        assert new_cmap[66] == 0x0042

    def test_resolve_glyph_rejects_surrogate_uni(self):
        """resolve_glyph_to_unicode returns None for uniD800."""
        assert resolve_glyph_to_unicode("uniD800") is None
        assert resolve_glyph_to_unicode("uniDBFF") is None
        assert resolve_glyph_to_unicode("uniDC00") is None
        assert resolve_glyph_to_unicode("uniDFFF") is None

    def test_resolve_glyph_rejects_surrogate_u(self):
        """resolve_glyph_to_unicode returns None for uD800."""
        assert resolve_glyph_to_unicode("uD800") is None
        assert resolve_glyph_to_unicode("uDFFF") is None

    def test_generate_cmap_filters_surrogates(self):
        """generate_tounicode_cmap_data filters out surrogates."""
        mapping = {65: 0xD800, 66: 0x0042}
        cmap = generate_tounicode_cmap_data(mapping)
        parsed = parse_tounicode_cmap(cmap)
        assert parsed[66] == 0x0042
        # Surrogate should have been replaced with PUA
        assert 0xE000 <= parsed[65] <= 0xF8FF

    def test_generate_cidfont_cmap_filters_surrogates(self):
        """generate_cidfont_tounicode_cmap filters out surrogates."""
        mapping = {1: 0xDC00, 2: 0x0041}
        cmap = generate_cidfont_tounicode_cmap(mapping)
        parsed = parse_tounicode_cmap(cmap)
        assert parsed[2] == 0x0041
        assert 0xE000 <= parsed[1] <= 0xF8FF
