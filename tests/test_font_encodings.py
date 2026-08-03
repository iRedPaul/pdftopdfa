# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/encodings.py — encoding handling and fixes."""

from io import BytesIO
from types import SimpleNamespace

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts import FontEmbedder
from pdftopdfa.fonts.encodings import (
    STANDARD_ENCODING,
    SYMBOL_ENCODING,
    ZAPFDINGBATS_ENCODING,
)
from pdftopdfa.sanitizers.truetype_encoding import sanitize_truetype_encoding
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


class TestStandardEncodingDict:
    """Direct tests for the STANDARD_ENCODING dictionary."""

    def test_all_keys_are_ints_in_range(self):
        """All keys must be integers in the 0-255 range."""
        for key in STANDARD_ENCODING:
            assert isinstance(key, int)
            assert 0 <= key <= 255

    def test_all_values_are_nonempty_strings(self):
        """All values must be non-empty strings (Adobe glyph names)."""
        for val in STANDARD_ENCODING.values():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_entry_count(self):
        """StandardEncoding has ~149 defined character codes."""
        assert 140 <= len(STANDARD_ENCODING) <= 160

    def test_ascii_letters(self):
        """A-Z (65-90) and a-z (97-122) map to their letter names."""
        for code in range(65, 91):
            assert STANDARD_ENCODING[code] == chr(code)
        for code in range(97, 123):
            assert STANDARD_ENCODING[code] == chr(code)

    def test_digits(self):
        """0-9 (48-57) map to word names."""
        digit_names = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        for i, name in enumerate(digit_names):
            assert STANDARD_ENCODING[48 + i] == name

    def test_key_differences_from_winansi(self):
        """StandardEncoding differs from WinAnsi at 0x27 and 0x60."""
        assert STANDARD_ENCODING[0x27] == "quoteright"
        assert STANDARD_ENCODING[0x60] == "quoteleft"

    def test_control_codes_absent(self):
        """Codes 0-31 are not mapped in StandardEncoding."""
        for code in range(32):
            assert code not in STANDARD_ENCODING

    def test_specific_high_codes(self):
        """Spot-check high-range Adobe-specific mappings."""
        assert STANDARD_ENCODING[174] == "fi"
        assert STANDARD_ENCODING[175] == "fl"
        assert STANDARD_ENCODING[208] == "emdash"
        assert STANDARD_ENCODING[225] == "AE"
        assert STANDARD_ENCODING[251] == "germandbls"

    def test_no_duplicate_values(self):
        """Each glyph name appears at most once."""
        values = list(STANDARD_ENCODING.values())
        assert len(values) == len(set(values))


class TestSymbolEncodingDict:
    """Direct tests for the SYMBOL_ENCODING dictionary."""

    def test_all_keys_are_ints_in_range(self):
        """All keys must be integers in the 0-255 range."""
        for key in SYMBOL_ENCODING:
            assert isinstance(key, int)
            assert 0 <= key <= 255

    def test_all_values_are_nonempty_strings(self):
        """All values must be non-empty strings."""
        for val in SYMBOL_ENCODING.values():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_entry_count(self):
        """Symbol encoding has ~190 entries."""
        assert 180 <= len(SYMBOL_ENCODING) <= 230

    def test_greek_uppercase(self):
        """Codes 65-90 map to Greek uppercase letters."""
        assert SYMBOL_ENCODING[65] == "Alpha"
        assert SYMBOL_ENCODING[66] == "Beta"
        assert SYMBOL_ENCODING[71] == "Gamma"
        assert SYMBOL_ENCODING[68] == "Delta"
        assert SYMBOL_ENCODING[87] == "Omega"

    def test_greek_lowercase(self):
        """Codes 97-122 map to Greek lowercase letters."""
        assert SYMBOL_ENCODING[97] == "alpha"
        assert SYMBOL_ENCODING[98] == "beta"
        assert SYMBOL_ENCODING[103] == "gamma"
        assert SYMBOL_ENCODING[100] == "delta"
        assert SYMBOL_ENCODING[119] == "omega"

    def test_math_symbols(self):
        """Mathematical symbols are correctly mapped."""
        assert SYMBOL_ENCODING[165] == "infinity"
        assert SYMBOL_ENCODING[214] == "radical"
        assert SYMBOL_ENCODING[242] == "integral"
        assert SYMBOL_ENCODING[182] == "partialdiff"

    def test_control_codes_absent(self):
        """Codes 0-31 are not mapped in Symbol encoding."""
        for code in range(32):
            assert code not in SYMBOL_ENCODING

    def test_no_duplicate_values(self):
        """Each glyph name appears at most once."""
        values = list(SYMBOL_ENCODING.values())
        assert len(values) == len(set(values))


class TestZapfDingbatsEncodingDict:
    """Direct tests for the ZAPFDINGBATS_ENCODING dictionary."""

    def test_all_keys_are_ints_in_range(self):
        """All keys must be integers in the 0-255 range."""
        for key in ZAPFDINGBATS_ENCODING:
            assert isinstance(key, int)
            assert 0 <= key <= 255

    def test_all_values_are_nonempty_strings(self):
        """All values must be non-empty strings."""
        for val in ZAPFDINGBATS_ENCODING.values():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_entry_count(self):
        """ZapfDingbats encoding has ~190-210 entries."""
        assert 190 <= len(ZAPFDINGBATS_ENCODING) <= 210

    def test_space_at_32(self):
        """Code 32 maps to 'space'."""
        assert ZAPFDINGBATS_ENCODING[32] == "space"

    def test_glyph_name_format(self):
        """All non-space glyph names follow the 'aNN' or 'aNNN' pattern."""
        import re

        for code, name in ZAPFDINGBATS_ENCODING.items():
            if name != "space":
                assert re.match(r"^a\d+$", name), (
                    f"Code {code}: '{name}' doesn't match aNN pattern"
                )

    def test_control_codes_absent(self):
        """Codes 0-31 are not mapped in ZapfDingbats encoding."""
        for code in range(32):
            assert code not in ZAPFDINGBATS_ENCODING

    def test_specific_mappings(self):
        """Spot-check specific ZapfDingbats mappings."""
        assert ZAPFDINGBATS_ENCODING[33] == "a1"
        assert ZAPFDINGBATS_ENCODING[65] == "a10"
        assert ZAPFDINGBATS_ENCODING[254] == "a191"

    def test_no_duplicate_values(self):
        """Each glyph name appears at most once."""
        values = list(ZAPFDINGBATS_ENCODING.values())
        assert len(values) == len(set(values))


class TestEncodingsCrossValidation:
    """Cross-validation tests between encoding dictionaries."""

    def test_all_share_space_at_32(self):
        """All three encodings map code 32 to 'space'."""
        assert STANDARD_ENCODING[32] == "space"
        assert SYMBOL_ENCODING[32] == "space"
        assert ZAPFDINGBATS_ENCODING[32] == "space"

    def test_standard_and_symbol_differ_at_65(self):
        """StandardEncoding maps 65 to 'A', Symbol maps to 'Alpha'."""
        assert STANDARD_ENCODING[65] == "A"
        assert SYMBOL_ENCODING[65] == "Alpha"

    def test_no_none_values(self):
        """No encoding should have None values."""
        for enc in (STANDARD_ENCODING, SYMBOL_ENCODING, ZAPFDINGBATS_ENCODING):
            assert None not in enc.values()

    @pytest.mark.parametrize(
        "encoding",
        [
            STANDARD_ENCODING,
            SYMBOL_ENCODING,
            ZAPFDINGBATS_ENCODING,
        ],
    )
    def test_127_absent(self, encoding):
        """Code 127 (DEL) is not mapped in any Adobe encoding."""
        assert 127 not in encoding


class TestStandardEncoding:
    """Tests for StandardEncoding support."""

    def test_standard_encoding_key_differences_from_winansi(self):
        """StandardEncoding differs from WinAnsi at specific code points."""
        from pdftopdfa.fonts.tounicode import (
            generate_tounicode_for_standard_encoding,
            generate_tounicode_for_winansi,
        )

        std = generate_tounicode_for_standard_encoding()
        winansi = generate_tounicode_for_winansi()

        # 0x27: StandardEncoding = quoteright (U+2019)
        #        WinAnsiEncoding = quotesingle (U+0027)
        assert std[0x27] == 0x2019
        assert winansi[0x27] == 0x0027

        # 0x60: StandardEncoding = quoteleft (U+2018)
        #        WinAnsiEncoding = grave (U+0060)
        assert std[0x60] == 0x2018
        assert winansi[0x60] == 0x0060

    def test_standard_encoding_has_ascii_range(self):
        """StandardEncoding maps A-Z, a-z, 0-9 correctly."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_standard_encoding

        std = generate_tounicode_for_standard_encoding()

        # A-Z
        assert std[0x41] == 0x0041  # A
        assert std[0x5A] == 0x005A  # Z
        # a-z
        assert std[0x61] == 0x0061  # a
        assert std[0x7A] == 0x007A  # z
        # 0-9
        assert std[0x30] == 0x0030  # 0
        assert std[0x39] == 0x0039  # 9

    def test_standard_encoding_high_range(self):
        """StandardEncoding maps high-range codes correctly."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_standard_encoding

        std = generate_tounicode_for_standard_encoding()

        # 0xC1 (193): grave -> U+0060
        assert std[193] == 0x0060
        # 0xD0 (208): emdash -> U+2014
        assert std[208] == 0x2014
        # 0xE1 (225): AE -> U+00C6
        assert std[225] == 0x00C6
        # 0xFB (251): germandbls -> U+00DF
        assert std[251] == 0x00DF

    def test_standard_encoding_unmapped_codes_absent(self):
        """StandardEncoding has gaps — unmapped codes are absent."""
        from pdftopdfa.fonts.tounicode import generate_tounicode_for_standard_encoding

        std = generate_tounicode_for_standard_encoding()

        # Codes 0-31, 127-160 are mostly unmapped in StandardEncoding
        assert 0 not in std
        assert 127 not in std
        assert 128 not in std
        assert 160 not in std

    def test_no_encoding_uses_standard_encoding(self):
        """Font without /Encoding uses StandardEncoding for ToUnicode."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "Helvetica")
        assert result is True

        # Check the generated CMap uses StandardEncoding values
        to_unicode = font_dict.get("/ToUnicode")
        assert to_unicode is not None
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        # 0x27 should map to U+2019 (quoteright) not U+0027 (quotesingle)
        assert "<27> <2019>" in cmap_text
        # 0x60 should map to U+2018 (quoteleft) not U+0060 (grave)
        assert "<60> <2018>" in cmap_text

    def test_explicit_standard_encoding_name(self):
        """Font with explicit /StandardEncoding Name uses StandardEncoding."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
            Encoding=Name("/StandardEncoding"),
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "Helvetica")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        assert "<27> <2019>" in cmap_text
        assert "<60> <2018>" in cmap_text

    def test_encoding_dict_with_standard_base_encoding(self):
        """Encoding dict with /StandardEncoding BaseEncoding works."""
        pdf = new_pdf()

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name("/StandardEncoding"),
            Differences=Array([65, Name("/Aacute")]),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "TestFont")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        # A (code 65) overridden to Aacute (U+00C1)
        assert "<41> <00C1>" in cmap_text
        # 0x27 should still be quoteright from StandardEncoding base
        assert "<27> <2019>" in cmap_text

    def test_encoding_dict_without_base_encoding_defaults_to_standard(self):
        """Encoding dict without BaseEncoding defaults to StandardEncoding."""
        pdf = new_pdf()

        encoding_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name("/Aacute")]),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestFont"),
            Encoding=encoding_dict,
        )

        embedder = FontEmbedder(pdf)
        result = embedder._add_tounicode_to_simple_font(font_dict, "TestFont")
        assert result is True

        to_unicode = font_dict.get("/ToUnicode")
        to_unicode = _resolve_indirect(to_unicode)
        cmap_data = bytes(to_unicode.read_bytes())
        cmap_text = cmap_data.decode("ascii")

        # Should use StandardEncoding as base, not WinAnsi
        assert "<27> <2019>" in cmap_text
        assert "<60> <2018>" in cmap_text
        # Override should be applied
        assert "<41> <00C1>" in cmap_text


class TestEnsureEncoding:
    """Tests for sanitize_truetype_encoding (ISO 19005-2, 6.2.11.6)."""

    def _make_embedded_font(
        self, pdf, *, subtype="/TrueType", flags=32, encoding=None, has_tounicode=False
    ):
        """Helper to create an embedded font dictionary."""
        font_stream = pdf.make_stream(b"\x00\x01\x00\x00" + b"\x00" * 100)
        font_stream[Name.Length1] = 104

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            Flags=flags,
            FontBBox=Array([0, -200, 1000, 800]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name(subtype),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
        )

        if encoding is not None:
            font_dict[Name.Encoding] = encoding

        if has_tounicode:
            tounicode = pdf.make_stream(b"fake cmap data")
            font_dict[Name.ToUnicode] = pdf.make_indirect(tounicode)

        return font_dict

    def _build_pdf_with_font(self, pdf, font_dict):
        """Helper to add a font to a single-page PDF."""
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        content_stream = pdf.make_stream(b"BT /F1 12 Tf (x) Tj ET")
        page_dict[Name.Contents] = content_stream
        pdf.pages.append(pikepdf.Page(page_dict))

    def test_truetype_without_encoding_gets_winansi(self):
        """Embedded non-symbolic TrueType font gets /WinAnsiEncoding."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(pdf, subtype="/TrueType", flags=32)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") == Name.WinAnsiEncoding

    def test_type1_without_encoding_gets_winansi(self):
        """Embedded non-symbolic Type1 font gets /WinAnsiEncoding."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(pdf, subtype="/Type1", flags=32)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") == Name.WinAnsiEncoding

    def test_font_with_existing_encoding_not_modified(self):
        """Font with existing /Encoding is not overwritten."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=Name.MacRomanEncoding,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") == Name.MacRomanEncoding

    def test_symbolic_font_skipped(self):
        """Symbolic Type1 font does not get /Encoding added."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(pdf, subtype="/Type1", flags=4)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None

    def test_font_with_tounicode_but_no_encoding_gets_encoding(self):
        """Font that already has ToUnicode but no /Encoding still gets encoding."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            has_tounicode=True,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") == Name.WinAnsiEncoding

    def test_non_embedded_font_skipped(self):
        """Non-embedded font does not get /Encoding added."""
        pdf = new_pdf()
        # Font without FontDescriptor/FontFile → not embedded
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None

    def test_cidfont_type0_skipped(self):
        """CIDFont/Type0 does not get /Encoding added."""
        pdf = new_pdf()
        # Minimal Type0 with a descendant
        desc_font = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestCIDFont"),
        )
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([pdf.make_indirect(desc_font)]),
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        # Type0 already has /Encoding (Identity-H), method should not touch it
        assert font_dict.get("/Encoding") == Name("/Identity-H")

    def test_type3_skipped(self):
        """Type3 font does not get /Encoding added."""
        pdf = new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            BaseFont=Name("/TestType3"),
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(),
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([]),
            ),
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        # Type3 should keep its encoding dict, not get WinAnsiEncoding
        enc = font_dict.get("/Encoding")
        assert enc is not None
        assert not isinstance(enc, pikepdf.Name) or str(enc) != "/WinAnsiEncoding"

    # --- FM1: Non-symbolic TrueType encoding fixes ---

    def test_truetype_wrong_encoding_replaced(self):
        """StandardEncoding is rebased without changing its glyph lookup."""
        from pdftopdfa.fonts.subsetter import _resolve_simple_font_encoding
        from pdftopdfa.fonts.tounicode import parse_tounicode_cmap_sequences

        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=Name.StandardEncoding,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        encoding = _resolve_indirect(font_dict.get("/Encoding"))
        assert encoding.get("/BaseEncoding") == Name.WinAnsiEncoding
        resolved = _resolve_simple_font_encoding(font_dict, pdfa_normalized=True)
        assert resolved is not None
        assert resolved[0x27] == "quoteright"
        assert resolved[0x60] == "quoteleft"
        assert resolved[0xA4] == "fraction"
        assert resolved.get(0x80, ".notdef") == ".notdef"
        assert FontEmbedder(pdf)._add_tounicode_to_simple_font(font_dict, "TestFont")
        tounicode = parse_tounicode_cmap_sequences(font_dict.ToUnicode.read_bytes())
        assert tounicode[b"\x27"] == (0x2019,)
        assert tounicode[b"\x60"] == (0x2018,)
        assert tounicode[b"\xa4"] == (0x2044,)

    def test_truetype_encoding_dict_wrong_base_fixed(self):
        """A StandardEncoding base is rebased without losing its mapping."""
        from pdftopdfa.fonts.subsetter import _resolve_simple_font_encoding

        pdf = new_pdf()
        enc_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                BaseEncoding=Name.StandardEncoding,
            )
        )
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=enc_dict,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        enc = _resolve_indirect(font_dict.get("/Encoding"))
        assert enc.get("/BaseEncoding") == Name.WinAnsiEncoding
        resolved = _resolve_simple_font_encoding(font_dict, pdfa_normalized=True)
        assert resolved is not None
        assert resolved[0x27] == "quoteright"
        assert resolved[0x60] == "quoteleft"
        assert resolved[0xA4] == "fraction"

    def test_standard_encoding_valid_differences_survive_rebase(self):
        """Valid overrides compose with StandardEncoding before rebasing."""
        from pdftopdfa.fonts.subsetter import _resolve_simple_font_encoding

        pdf = new_pdf()
        encoding = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.StandardEncoding,
            Differences=Array([0x41, Name("/B")]),
        )
        font = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=encoding,
        )
        self._build_pdf_with_font(pdf, font)

        sanitize_truetype_encoding(pdf)

        resolved = _resolve_simple_font_encoding(font, pdfa_normalized=True)
        assert resolved is not None
        assert resolved[0x41] == "B"
        assert resolved[0x27] == "quoteright"

    def test_truetype_encoding_dict_non_agl_differences_removed(self):
        """TrueType with non-AGL glyph names in Differences gets them removed."""
        pdf = new_pdf()
        enc_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                BaseEncoding=Name.WinAnsiEncoding,
                Differences=Array(
                    [
                        65,
                        Name("/nonExistentGlyph12345"),
                    ]
                ),
            )
        )
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=enc_dict,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        enc = _resolve_indirect(font_dict.get("/Encoding"))
        assert enc.get("/Differences") is None

    def test_truetype_encoding_dict_uni_names_removed(self):
        """TrueType with synthesized uniXXXX names in Differences gets them removed."""
        pdf = new_pdf()
        enc_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                BaseEncoding=Name.WinAnsiEncoding,
                Differences=Array([0, Name("/uniE000"), Name("/uni0001")]),
            )
        )
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=enc_dict,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        enc = _resolve_indirect(font_dict.get("/Encoding"))
        assert enc.get("/Differences") is None

    def test_truetype_macroman_not_modified(self):
        """TrueType with MacRomanEncoding is not modified."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=Name.MacRomanEncoding,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") == Name.MacRomanEncoding

    def test_build_encoding_dictionary_with_base_encoding_is_minimal(self):
        """Preserved encodings keep only AGL overrides that differ from the base."""
        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        encoding = embedder._build_encoding_dictionary(
            {
                0: "uniE000",
                1: "uni0001",
                65: "A",
                66: "B",
                128: "Adieresis",
            },
            base_encoding=Name.WinAnsiEncoding,
        )

        assert encoding.get("/BaseEncoding") == Name.WinAnsiEncoding
        assert list(encoding.get("/Differences")) == [128, Name("/Adieresis")]

    def test_build_encoding_dictionary_keeps_base_without_differences(self):
        """An exact base mapping still declares the base encoding."""
        from pdftopdfa.fonts.subsetter import _resolve_simple_font_encoding

        pdf = new_pdf()
        embedder = FontEmbedder(pdf)

        encoding = embedder._build_encoding_dictionary(
            {65: "A", 66: "B", 0xE4: "adieresis"},
            base_encoding=Name.WinAnsiEncoding,
        )

        font = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=32,
            encoding=pdf.make_indirect(encoding),
        )
        self._build_pdf_with_font(pdf, font)
        sanitize_truetype_encoding(pdf)

        assert encoding.get("/BaseEncoding") == Name.WinAnsiEncoding
        assert list(encoding.get("/Differences")) == []
        assert _resolve_simple_font_encoding(font, pdfa_normalized=True)[0xE4] == (
            "adieresis"
        )

    # --- FM2: Symbolic TrueType encoding removal ---

    def test_symbolic_truetype_encoding_removed(self):
        """Symbolic TrueType with /Encoding gets it removed."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=4,
            encoding=Name.WinAnsiEncoding,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None

    def test_symbolic_truetype_without_encoding_unchanged(self):
        """Symbolic TrueType without /Encoding is not modified."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=4,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None

    # --- FM3: Symbolic TrueType cmap fixes ---

    @staticmethod
    def _make_ttfont_data(cmap_subtables):
        """Build minimal TrueType font bytes with given cmap subtables."""

        from fontTools.fontBuilder import FontBuilder
        from fontTools.ttLib.tables._c_m_a_p import cmap_format_4

        fb = FontBuilder(1000, isTTF=True)
        fb.setupGlyphOrder([".notdef", "A", "B"])
        fb.setupCharacterMap({65: "A", 66: "B"})

        # Draw empty glyphs using pen API
        fb.setupGlyf({})
        pen = fb.setupGlyf({})  # noqa: F841
        glyf_table = fb.font["glyf"]
        from fontTools.ttLib.tables._g_l_y_f import Glyph

        for gname in [".notdef", "A", "B"]:
            glyf_table[gname] = Glyph()

        fb.setupHorizontalMetrics(
            {
                ".notdef": (500, 0),
                "A": (600, 0),
                "B": (600, 0),
            }
        )
        fb.setupHorizontalHeader(ascent=800, descent=-200)
        fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
        fb.setupOS2()
        fb.setupPost()
        tt = fb.font

        # Replace cmap with custom subtables
        from fontTools.ttLib.tables._c_m_a_p import table__c_m_a_p

        cmap = table__c_m_a_p()
        cmap.tableVersion = 0
        cmap.tables = []
        for platform_id, plat_enc_id, mapping in cmap_subtables:
            st = cmap_format_4(4)
            st.platformID = platform_id
            st.platEncID = plat_enc_id
            st.language = 0
            st.cmap = mapping
            cmap.tables.append(st)
        tt["cmap"] = cmap

        buf = BytesIO()
        tt.save(buf)
        tt.close()
        return buf.getvalue()

    def _make_embedded_font_with_ttdata(self, pdf, tt_data, *, flags=4):
        """Create embedded font dict using actual TrueType font data."""
        font_stream = pdf.make_stream(tt_data)
        font_stream[Name.Length1] = len(tt_data)

        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            Flags=flags,
            FontBBox=Array([0, -200, 1000, 800]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile2=pdf.make_indirect(font_stream),
        )

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name("/TrueType"),
            BaseFont=Name("/TestFont"),
            FontDescriptor=pdf.make_indirect(font_descriptor),
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
        )
        return font_dict

    def test_symbolic_truetype_cmap_single_ok(self):
        """Symbolic TrueType with exactly one cmap subtable is not modified."""
        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        # Font data should not be modified (single subtable is OK)
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        result_data = bytes(ff2.read_bytes())
        assert result_data == tt_data

    def test_symbolic_named_encoding_does_not_shadow_mac_cmap(self):
        """A bare invalid Encoding does not replace a symbolic Mac byte cmap."""
        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        font_dict[Name.Encoding] = Name.WinAnsiEncoding
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        assert bytes(ff2.read_bytes()) == tt_data

    def test_symbolic_zapf_difference_migrates_to_program_glyph(self):
        """Zapf /a20 becomes a Microsoft Symbol heavy-check-mark mapping."""
        from importlib import resources

        from fontTools.ttLib import TTFont

        font_path = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "NotoSansSymbols2-Regular.ttf"
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(
            pdf,
            font_path.read_bytes(),
            flags=4,
        )
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([52, Name("/a20")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            unicode_cmap = tt_font.getBestCmap()
            assert unicode_cmap is not None
            expected_glyph = unicode_cmap[0x2714]
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap[0xF034] == expected_glyph
        finally:
            tt_font.close()

    def test_symbolic_difference_does_not_override_ms_symbol_cmap(self):
        """A Microsoft Symbol cmap is authoritative over PDF Differences."""
        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (3, 0, {0xF041: "A"}),
                (3, 1, {65: "A"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([99, Name("/A")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        result = sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None
        assert result["tt_symbolic_cmap_added"] == 0
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap[0xF041] == "A"
            assert 0xF063 not in symbol_cmap.cmap
        finally:
            tt_font.close()

    def test_symbolic_f1_cmap_is_authoritative(self):
        """The valid F100 Microsoft Symbol range maps to PDF bytes."""
        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (3, 0, {0xF141: "A"}),
                (3, 1, {65: "A"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([99, Name("/A")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap[0xF141] == "A"
            assert 0xF063 not in symbol_cmap.cmap
        finally:
            tt_font.close()

    def test_symbolic_difference_uses_all_unicode_cmaps(self):
        """Lower-priority Unicode cmaps can resolve an explicit Difference."""
        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (3, 10, {66: "B"}),
                (3, 1, {0x2714: "A"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([52, Name("/a20")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap[0xF034] == "A"
        finally:
            tt_font.close()

    def test_symbolic_difference_creates_missing_cmap_table(self):
        """A directly named Difference is retained even without a cmap table."""
        from fontTools.ttLib import TTFont

        tt_font = TTFont(BytesIO(self._make_ttfont_data([])))
        del tt_font["cmap"]
        buffer = BytesIO()
        tt_font.save(buffer)
        tt_font.close()

        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(
            pdf,
            buffer.getvalue(),
            flags=4,
        )
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name("/A")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        result = sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None
        assert result["tt_symbolic_cmap_added"] == 1
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap[0xF041] == "A"
        finally:
            tt_font.close()

    @pytest.mark.parametrize(
        ("unicode_mapping", "expected_mapping"),
        [
            ({65: "A", 66: "B"}, {0xF041: "A"}),
            ({}, {0xF041: "A"}),
        ],
        ids=["fallback-remains", "empty-fallback"],
    )
    def test_symbolic_notdef_difference_does_not_override_program_mapping(
        self,
        unicode_mapping,
        expected_mapping,
    ):
        """A symbolic font ignores an explicit .notdef Difference."""
        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (3, 0, {0xF041: "A"}),
                (3, 1, unicode_mapping),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        font_dict[Name.Encoding] = pdf.make_indirect(
            Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name("/.notdef")]),
            )
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        assert font_dict.get("/Encoding") is None
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 0
            )
            assert symbol_cmap.cmap == expected_mapping
        finally:
            tt_font.close()

    def test_symbolic_truetype_cmap_with_ms_symbol_ok(self):
        """Symbolic TrueType with MS Symbol (3,0) cmap is not modified."""
        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
                (3, 0, {0xF041: "A", 0xF042: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        result_data = bytes(ff2.read_bytes())
        assert result_data == tt_data

    def test_symbolic_truetype_cmap_fixed(self):
        """Symbolic TrueType without (3,0) gets MS Symbol cmap added."""

        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
                (3, 1, {65: "A", 66: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        # Font should now have a (3,0) subtable
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        result_data = bytes(ff2.read_bytes())
        tt = TTFont(BytesIO(result_data))
        cmap = tt.get("cmap")
        found_30 = any(st.platformID == 3 and st.platEncID == 0 for st in cmap.tables)
        assert found_30, "Expected (3,0) MS Symbol cmap subtable"
        # Verify mapping is in 0xF000 range
        for st in cmap.tables:
            if st.platformID == 3 and st.platEncID == 0:
                assert 0xF041 in st.cmap
                assert 0xF042 in st.cmap
        tt.close()

    def test_symbolic_truetype_cmap_empty_30_repaired(self):
        """Symbolic TrueType with empty (3,0) cmap gets it repaired."""

        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
                (3, 0, {}),  # Empty (3,0)
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        result_data = bytes(ff2.read_bytes())
        tt = TTFont(BytesIO(result_data))
        cmap = tt.get("cmap")
        # (3,0) should now have entries derived from (1,0) source
        for st in cmap.tables:
            if st.platformID == 3 and st.platEncID == 0:
                assert len(st.cmap) > 0, "Repaired (3,0) should be non-empty"
                assert 0xF041 in st.cmap
                assert 0xF042 in st.cmap
                break
        else:
            raise AssertionError("Expected (3,0) cmap subtable")
        tt.close()

    def test_nonsymbolic_truetype_ms_unicode_cmap_from_mac_roman(self):
        """(3,1) cmap built from (1,0) source translates Mac Roman codes."""

        from fontTools.ttLib import TTFont

        # Mac Roman 0x8A is "ä" (U+00E4), not U+008A
        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 0x8A: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=32)
        enc_dict = Dictionary(
            Type=Name.Encoding,
            BaseEncoding=Name.WinAnsiEncoding,
            Differences=Array([65, Name("/A"), 66, Name("/B")]),
        )
        font_dict[Name.Encoding] = pdf.make_indirect(enc_dict)
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt = TTFont(BytesIO(bytes(ff2.read_bytes())))
        cmap = tt.get("cmap")
        for st in cmap.tables:
            if st.platformID == 3 and st.platEncID == 1:
                assert st.cmap.get(65) == "A"
                assert st.cmap.get(0xE4) == "B", "Mac Roman 0x8A must map to U+00E4"
                assert 0x8A not in st.cmap
                break
        else:
            raise AssertionError("Expected (3,1) MS Unicode cmap subtable")
        tt.close()

    @pytest.mark.parametrize("remove_cmap_table", [False, True])
    def test_nonsymbolic_truetype_empty_cmap_rebuilt_from_glyph_names(
        self,
        remove_cmap_table,
    ):
        """An empty or absent cmap is rebuilt from final encoding glyph names."""
        from fontTools.ttLib import TTFont

        tt_data = self._make_ttfont_data([])
        if remove_cmap_table:
            tt_font = TTFont(BytesIO(tt_data))
            del tt_font["cmap"]
            buffer = BytesIO()
            tt_font.save(buffer)
            tt_font.close()
            tt_data = buffer.getvalue()

        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=32)
        self._build_pdf_with_font(pdf, font_dict)

        result = sanitize_truetype_encoding(pdf)

        assert result["tt_nonsymbolic_cmap_added"] == 1
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        tt_font = TTFont(BytesIO(bytes(ff2.read_bytes())))
        try:
            unicode_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if table.platformID == 3 and table.platEncID == 1
            )
            assert unicode_cmap.cmap[0x41] == "A"
        finally:
            tt_font.close()

    def test_symbolic_truetype_cmap_no_cmap_table(self):
        """Symbolic TrueType with no cmap table at all is not modified."""
        pdf = new_pdf()
        # Use dummy font data (no valid cmap) — the method returns False
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/TrueType",
            flags=4,
        )
        self._build_pdf_with_font(pdf, font_dict)

        # Should not crash, just return without changes
        sanitize_truetype_encoding(pdf)

    # --- Backward compatibility ---

    def test_type1_symbolic_encoding_not_removed(self):
        """Symbolic Type1 font encoding is not removed (only TrueType)."""
        pdf = new_pdf()
        font_dict = self._make_embedded_font(
            pdf,
            subtype="/Type1",
            flags=4,
            encoding=Name.WinAnsiEncoding,
        )
        self._build_pdf_with_font(pdf, font_dict)

        sanitize_truetype_encoding(pdf)

        # Type1 symbolic: encoding should be preserved
        assert font_dict.get("/Encoding") == Name.WinAnsiEncoding

    # --- Integration: subsetting + encoding fix pipeline order ---

    def test_symbolic_truetype_survives_subsetting(self):
        """Symbolic TrueType keeps (3,0) cmap after subsetting."""

        from fontTools.ttLib import TTFont

        # Build a symbolic TrueType with /Encoding (like ZapfDingbats after embedding)
        tt_data = self._make_ttfont_data(
            [
                (1, 0, {65: "A", 66: "B"}),
                (3, 1, {65: "A", 66: "B"}),
            ]
        )
        pdf = new_pdf()
        font_dict = self._make_embedded_font_with_ttdata(pdf, tt_data, flags=4)
        # Give it an Encoding with Differences (like ZapfDingbats gets)
        enc_dict = Dictionary(
            Type=Name.Encoding,
            Differences=Array([65, Name("/A"), 66, Name("/B")]),
        )
        font_dict[Name.Encoding] = pdf.make_indirect(enc_dict)
        self._build_pdf_with_font(pdf, font_dict)

        # Step 1: Subset (uses /Encoding for glyph selection)
        embedder = FontEmbedder(pdf)
        embedder.subset_embedded_fonts()

        # Step 2: Fix encodings (removes /Encoding, adds (3,0) cmap)
        sanitize_truetype_encoding(pdf)

        # Verify: /Encoding removed
        assert font_dict.get("/Encoding") is None

        # Verify: (3,0) cmap present in subsetted font
        fd = _resolve_indirect(font_dict.get("/FontDescriptor"))
        ff2 = _resolve_indirect(fd.get("/FontFile2"))
        result_data = bytes(ff2.read_bytes())
        tt = TTFont(BytesIO(result_data))
        cmap = tt.get("cmap")
        found_30 = any(st.platformID == 3 and st.platEncID == 0 for st in cmap.tables)
        assert found_30, "Expected (3,0) MS Symbol cmap subtable after subset + fix"
        tt.close()


class TestTrueTypeClassification:
    """Tests for normalization of contradictory TrueType classification."""

    def _make_pdf(self, *, flags, cmaps, encoding=None):
        helper = TestEnsureEncoding()
        pdf = new_pdf()
        font = helper._make_embedded_font_with_ttdata(
            pdf,
            helper._make_ttfont_data(cmaps),
            flags=flags,
        )
        if encoding is not None:
            font[Name.Encoding] = encoding
        helper._build_pdf_with_font(pdf, font)
        return pdf, font

    def test_explicit_encoding_wins_when_flags_are_missing(self):
        """Flags=0 does not turn an explicitly encoded font into Symbolic."""
        pdf, font = self._make_pdf(
            flags=0,
            cmaps=[
                (3, 0, {0xF041: "B"}),
                (3, 1, {0x41: "A"}),
            ],
            encoding=Name.WinAnsiEncoding,
        )

        sanitize_truetype_encoding(pdf)

        assert int(font.FontDescriptor.Flags) & (4 | 32) == 32
        assert font.Encoding == Name.WinAnsiEncoding

    def test_conflicting_flags_choose_nonsymbolic_for_unicode_font(self):
        """Flags=36 with a Unicode cmap and Encoding normalizes to Nonsymbolic."""
        pdf, font = self._make_pdf(
            flags=36,
            cmaps=[(3, 1, {0x41: "A"})],
            encoding=Name.WinAnsiEncoding,
        )

        sanitize_truetype_encoding(pdf)

        assert int(font.FontDescriptor.Flags) & (4 | 32) == 32
        assert font.Encoding == Name.WinAnsiEncoding

    def test_conflicting_flags_without_encoding_keep_symbol_byte_lookup(self):
        """Flags=36 without Encoding preserves a Microsoft Symbol cmap."""
        pdf, font = self._make_pdf(
            flags=36,
            cmaps=[(3, 0, {0xF141: "A"})],
        )

        sanitize_truetype_encoding(pdf)

        assert int(font.FontDescriptor.Flags) & (4 | 32) == 4
        assert font.get("/Encoding") is None

    def test_conflicting_flags_with_encoding_keep_mac_byte_lookup(self):
        """Flags=36 preserves the effective Macintosh program lookup."""
        pdf, font = self._make_pdf(
            flags=36,
            cmaps=[(1, 0, {0x41: "B"})],
            encoding=Name.WinAnsiEncoding,
        )

        sanitize_truetype_encoding(pdf)

        assert int(font.FontDescriptor.Flags) & (4 | 32) == 4
        assert font.get("/Encoding") is None

        from fontTools.ttLib import TTFont

        tt_font = TTFont(BytesIO(font.FontDescriptor.FontFile2.read_bytes()))
        try:
            mac_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if (table.platformID, table.platEncID) == (1, 0)
            )
            assert mac_cmap.cmap[0x41] == "B"
        finally:
            tt_font.close()

    def test_mac_byte_lookup_survives_subsetting_pipeline(self):
        """Early flag repair and subsetting preserve the selected program glyph."""
        from fontTools.ttLib import TTFont

        from pdftopdfa.fonts.utils import get_truetype_byte_encoding

        pdf, font = self._make_pdf(
            flags=36,
            cmaps=[
                (1, 0, {0x41: "B"}),
                (3, 1, {0x41: "A", 0x42: "B"}),
            ],
            encoding=Name.WinAnsiEncoding,
        )
        pdf.pages[0].Contents.write(b"BT /F1 12 Tf (A) Tj ET")

        sanitize_truetype_encoding(pdf)
        FontEmbedder(pdf).subset_embedded_fonts()
        sanitize_truetype_encoding(pdf)

        tt_font = TTFont(BytesIO(font.FontDescriptor.FontFile2.read_bytes()))
        try:
            byte_encoding = get_truetype_byte_encoding(tt_font)
            assert byte_encoding is not None
            assert byte_encoding[0] in {(1, 0), (3, 0)}
            assert byte_encoding[2][0x41] == "B"
        finally:
            tt_font.close()

    def test_missing_encoding_with_mac_cmap_keeps_raw_byte_lookup(self):
        """A raw Macintosh byte cmap remains authoritative without Encoding."""
        pdf, font = self._make_pdf(
            flags=32,
            cmaps=[(1, 0, {0x80: "A"})],
        )

        sanitize_truetype_encoding(pdf)

        assert int(font.FontDescriptor.Flags) & (4 | 32) == 4
        assert font.get("/Encoding") is None

    def test_ambiguous_symbol_cmap_is_not_augmented(self):
        """An invalid mixed-range cmap is not reported as repaired."""
        from fontTools.ttLib import TTFont

        pdf, font = self._make_pdf(
            flags=4,
            cmaps=[
                (3, 0, {0xF141: "A", 0xF241: "A"}),
                (3, 1, {0x41: "A"}),
            ],
            encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([0x63, Name("/A")]),
            ),
        )

        result = sanitize_truetype_encoding(pdf)

        assert result["tt_symbolic_cmap_added"] == 0
        tt_font = TTFont(BytesIO(font.FontDescriptor.FontFile2.read_bytes()))
        try:
            symbol_cmap = next(
                table
                for table in tt_font["cmap"].tables
                if (table.platformID, table.platEncID) == (3, 0)
            )
            assert symbol_cmap.cmap == {0xF141: "A", 0xF241: "A"}
        finally:
            tt_font.close()

    def test_full_adobe_glyph_list_difference_is_preserved(self):
        """A legacy AGL name is not mistaken for a non-AGL Difference."""
        from pdftopdfa.fonts.subsetter import _resolve_simple_font_encoding

        pdf, font = self._make_pdf(
            flags=32,
            cmaps=[(3, 1, {0x0410: "A"})],
            encoding=Dictionary(
                Type=Name.Encoding,
                BaseEncoding=Name.WinAnsiEncoding,
                Differences=Array([0x41, Name("/afii10017")]),
            ),
        )

        sanitize_truetype_encoding(pdf)

        assert font.Encoding.get("/Differences") is not None
        resolved = _resolve_simple_font_encoding(font, pdfa_normalized=True)
        assert resolved is not None
        assert resolved[0x41] == "afii10017"

    @pytest.mark.parametrize("offset", [0, 0xF000, 0xF100, 0xF200])
    def test_symbol_program_cmap_drives_tounicode(self, offset):
        """ToUnicode follows every valid Microsoft Symbol byte range."""
        pdf, font = self._make_pdf(
            flags=4,
            cmaps=[
                (3, 0, {offset + 0x41: "A"}),
                (3, 1, {0x0391: "A"}),
            ],
        )
        font.FirstChar = 0x41
        font.LastChar = 0x41

        assert FontEmbedder(pdf)._add_tounicode_to_simple_font(font, "TestFont")

        cmap = font.ToUnicode.read_bytes().decode("ascii")
        assert "<41> <0391>" in cmap

    def test_malformed_character_bounds_do_not_block_tounicode(self):
        """Malformed FirstChar/LastChar values fall back to the byte range."""
        pdf, font = self._make_pdf(
            flags=4,
            cmaps=[
                (3, 0, {0xF041: "A"}),
                (3, 1, {0x0041: "A"}),
            ],
        )
        font.FirstChar = Name("/Invalid")
        font.LastChar = Name("/Invalid")

        assert FontEmbedder(pdf)._add_tounicode_to_simple_font(font, "TestFont")
        assert "<41> <0041>" in font.ToUnicode.read_bytes().decode("ascii")


class TestTrueTypeByteEncoding:
    """Tests for TrueType program byte-cmap resolution."""

    @pytest.mark.parametrize("offset", [0, 0xF000, 0xF100, 0xF200])
    def test_accepts_each_standard_symbol_range(self, offset):
        """Each ISO Microsoft Symbol range resolves to the same PDF byte."""
        from fontTools.ttLib import TTFont

        from pdftopdfa.fonts.utils import get_truetype_byte_encoding

        font_data = TestEnsureEncoding._make_ttfont_data([(3, 0, {offset + 0x41: "A"})])
        tt_font = TTFont(BytesIO(font_data))
        try:
            assert get_truetype_byte_encoding(tt_font) == (
                (3, 0),
                offset,
                {0x41: "A"},
            )
        finally:
            tt_font.close()

    @pytest.mark.parametrize(
        "mapping",
        [
            {0xF041: "A", 0xF141: "A"},
            {0xF041: "A", 0xF141: "B"},
            {0xF041: "A", 0x1041: "B"},
        ],
    )
    def test_rejects_ambiguous_or_invalid_symbol_ranges(self, mapping):
        """Mixed or out-of-range Microsoft Symbol keys are unresolved."""
        from fontTools.ttLib import TTFont

        from pdftopdfa.fonts.utils import get_truetype_byte_encoding

        font_data = TestEnsureEncoding._make_ttfont_data([(3, 0, mapping)])
        tt_font = TTFont(BytesIO(font_data))
        try:
            assert get_truetype_byte_encoding(tt_font) is None
        finally:
            tt_font.close()

    def test_rejects_conflicting_symbol_subtables(self):
        """Two different Microsoft Symbol subtables are ambiguous."""
        from fontTools.ttLib import TTFont
        from fontTools.ttLib.tables._c_m_a_p import cmap_format_4

        from pdftopdfa.fonts.utils import get_truetype_byte_encoding

        font_data = TestEnsureEncoding._make_ttfont_data([(3, 0, {0xF041: "A"})])
        tt_font = TTFont(BytesIO(font_data))
        try:
            conflicting = cmap_format_4(4)
            conflicting.platformID = 3
            conflicting.platEncID = 0
            conflicting.language = 1
            conflicting.cmap = {0xF041: "B"}
            tt_font["cmap"].tables.append(conflicting)
            assert get_truetype_byte_encoding(tt_font) is None
        finally:
            tt_font.close()

    def test_get_any_cmap_skips_format_14_without_cmap_attribute(self):
        """A variation-selector subtable cannot crash cmap fallback."""
        from pdftopdfa.fonts.utils import get_any_cmap

        class FakeFont(dict):
            def getBestCmap(self):  # noqa: N802 - fontTools API
                raise KeyError("no Unicode cmap")

        variation = SimpleNamespace(platformID=0, platEncID=5)
        symbol = SimpleNamespace(
            platformID=3,
            platEncID=0,
            cmap={0xF041: "A"},
        )
        font = FakeFont(cmap=SimpleNamespace(tables=[variation, symbol]))

        assert get_any_cmap(font) == {0xF041: "A"}
