# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/font_metrics.py."""

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.font_metrics import (
    _get_standard14_metrics,
    _read_widths_from_font_dict,
    _unicode_to_winansi,
    _winansi_to_unicode,
    _wrap_text,
    compute_auto_font_size,
    decode_pdf_string,
    encode_for_content_stream,
    get_ascent_descent,
    get_font_bbox,
    get_text_width,
)

# ===================================================================
# Standard-14 Metrics Tests
# ===================================================================


class TestStandard14Metrics:
    """Tests for Standard-14 font metric tables."""

    def test_all_14_fonts_exist(self):
        """All 14 standard fonts have metric entries."""
        names = [
            "Helvetica",
            "Helvetica-Bold",
            "Helvetica-Oblique",
            "Helvetica-BoldOblique",
            "Times-Roman",
            "Times-Bold",
            "Times-Italic",
            "Times-BoldItalic",
            "Courier",
            "Courier-Bold",
            "Courier-Oblique",
            "Courier-BoldOblique",
            "Symbol",
            "ZapfDingbats",
        ]
        for name in names:
            m = _get_standard14_metrics(name)
            assert m is not None, f"Missing metrics for {name}"
            assert "widths" in m
            assert "ascent" in m
            assert "descent" in m

    def test_helvetica_known_widths(self):
        """Helvetica has correct widths for known characters."""
        m = _get_standard14_metrics("Helvetica")
        widths = m["widths"]
        assert widths[65] == 667  # A
        assert widths[32] == 278  # space
        assert widths[105] == 222  # i
        assert widths[77] == 833  # M

    def test_courier_is_monospaced(self):
        """Courier family: all characters are 600 units wide."""
        for name in (
            "Courier",
            "Courier-Bold",
            "Courier-Oblique",
            "Courier-BoldOblique",
        ):
            m = _get_standard14_metrics(name)
            widths = m["widths"]
            for code, w in widths.items():
                assert w == 600, f"{name} code {code} width {w} != 600"

    def test_helvetica_oblique_matches_regular(self):
        """Helvetica-Oblique shares width table with Helvetica."""
        m_reg = _get_standard14_metrics("Helvetica")
        m_obl = _get_standard14_metrics("Helvetica-Oblique")
        assert m_reg["widths"] is m_obl["widths"]

    def test_ascent_descent_plausible(self):
        """Ascent/descent values are plausible for all standard fonts."""
        for name in ("Helvetica", "Times-Roman", "Courier"):
            m = _get_standard14_metrics(name)
            assert m["ascent"] > 0, f"{name} ascent should be positive"
            assert m["descent"] < 0, f"{name} descent should be negative"
            assert m["ascent"] > abs(m["descent"]), (
                f"{name} ascent should exceed |descent|"
            )

    def test_alias_helv(self):
        """Alias 'Helv' resolves to Helvetica."""
        m = _get_standard14_metrics("Helv")
        assert m is not None
        assert m["widths"][65] == 667

    def test_alias_zadb(self):
        """Alias 'ZaDb' resolves to ZapfDingbats."""
        m = _get_standard14_metrics("ZaDb")
        assert m is not None

    def test_alias_arial(self):
        """'Arial' resolves to Helvetica."""
        m = _get_standard14_metrics("Arial")
        assert m is not None
        assert m["widths"][65] == 667

    def test_unknown_font_returns_none(self):
        """Unknown font name returns None."""
        assert _get_standard14_metrics("NonExistentFont") is None


# ===================================================================
# Text Width Tests
# ===================================================================


class TestGetTextWidth:
    """Tests for get_text_width()."""

    def test_empty_string(self):
        """Empty string has width 0."""
        assert get_text_width("", font_name="Helv") == 0.0

    def test_helvetica_known_string(self):
        """Helvetica 'A' at 1000pt should be 667pt."""
        w = get_text_width("A", font_name="Helvetica", font_size=1000.0)
        assert w == 667.0

    def test_helvetica_space(self):
        """Helvetica space at 12pt."""
        w = get_text_width(" ", font_name="Helvetica", font_size=12.0)
        expected = 278 * 12.0 / 1000.0
        assert abs(w - expected) < 0.01

    def test_courier_is_uniform(self):
        """Courier: all characters have same width."""
        w1 = get_text_width("AAAA", font_name="Courier", font_size=12.0)
        w2 = get_text_width("iiii", font_name="Courier", font_size=12.0)
        assert abs(w1 - w2) < 0.01

    def test_helvetica_hello_world(self):
        """'Hello World' in Helvetica at 12pt has expected width."""
        w = get_text_width("Hello World", font_name="Helv", font_size=12.0)
        # H=722, e=556, l=222, l=222, o=556, space=278,
        # W=944, o=556, r=333, l=222, d=556
        expected_units = 722 + 556 + 222 + 222 + 556 + 278 + 944 + 556 + 333 + 222 + 556
        expected = expected_units * 12.0 / 1000.0
        assert abs(w - expected) < 0.01

    def test_font_size_zero(self):
        """Font size 0 returns width 0."""
        assert get_text_width("Hello", font_name="Helv", font_size=0.0) == 0.0

    def test_no_font_fallback(self):
        """Unknown font uses 600-unit (Courier) fallback."""
        w = get_text_width("AB", font_name="UnknownFont", font_size=10.0)
        expected = 2 * 600 * 10.0 / 1000.0
        assert abs(w - expected) < 0.01

    def test_embedded_widths(self):
        """Font dict with /Widths array takes precedence."""
        new_pdf()
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name("/Type1"),
            BaseFont=Name("/Helvetica"),
            FirstChar=65,
            LastChar=67,
            Widths=Array([100, 200, 300]),
        )
        w = get_text_width("ABC", font_dict=font_dict, font_size=10.0)
        expected = (100 + 200 + 300) * 10.0 / 1000.0
        assert abs(w - expected) < 0.01


# ===================================================================
# Ascent/Descent Tests
# ===================================================================


class TestGetAscentDescent:
    """Tests for get_ascent_descent()."""

    def test_helvetica(self):
        """Helvetica has known ascent/descent."""
        a, d = get_ascent_descent(font_name="Helvetica")
        assert a == 718
        assert d == -207

    def test_times_roman(self):
        """Times-Roman has known ascent/descent."""
        a, d = get_ascent_descent(font_name="Times-Roman")
        assert a == 683
        assert d == -217

    def test_courier(self):
        """Courier has known ascent/descent."""
        a, d = get_ascent_descent(font_name="Courier")
        assert a == 629
        assert d == -157

    def test_fallback(self):
        """Unknown font returns generic fallback."""
        a, d = get_ascent_descent(font_name="UnknownFont")
        assert a > 0
        assert d < 0

    def test_font_descriptor(self):
        """Reads from /FontDescriptor when available."""
        new_pdf()
        fd = Dictionary(Ascent=800, Descent=-200)
        font_dict = Dictionary(
            Type=Name.Font,
            FontDescriptor=fd,
        )
        a, d = get_ascent_descent(font_dict)
        assert a == 800.0
        assert d == -200.0


# ===================================================================
# Font BBox Tests
# ===================================================================


class TestGetFontBBox:
    """Tests for get_font_bbox()."""

    def test_helvetica_bbox(self):
        """Helvetica has known bounding box."""
        bbox = get_font_bbox(
            Dictionary(
                Type=Name.Font,
                BaseFont=Name("/Helvetica"),
            )
        )
        assert bbox == (-166, -225, 1000, 931)

    def test_fallback_bbox(self):
        """No font dict returns generic fallback."""
        bbox = get_font_bbox(None)
        assert len(bbox) == 4
        assert bbox[0] < 0 and bbox[3] > 0


# ===================================================================
# Auto-Size Tests
# ===================================================================


class TestComputeAutoFontSize:
    """Tests for compute_auto_font_size()."""

    def test_minimum_size(self):
        """Result is at least 4pt."""
        sz = compute_auto_font_size(
            "X" * 1000,
            font_name="Helv",
            field_width=10,
            field_height=5,
        )
        assert sz >= 4.0

    def test_empty_text(self):
        """Empty text returns reasonable default."""
        sz = compute_auto_font_size(
            "",
            font_name="Helv",
            field_width=200,
            field_height=30,
        )
        assert 4.0 <= sz <= 30.0

    def test_short_text_large_field(self):
        """Short text in large field gets large font."""
        sz = compute_auto_font_size(
            "Hi",
            font_name="Helv",
            field_width=200,
            field_height=30,
        )
        assert sz > 12.0

    def test_long_text_small_field(self):
        """Long text in small field gets smaller font."""
        sz = compute_auto_font_size(
            "This is a very long text string",
            font_name="Helv",
            field_width=100,
            field_height=20,
        )
        assert sz < 12.0

    def test_text_fits_width(self):
        """Computed size makes text fit the available width."""
        text = "Hello World"
        fw, fh = 100.0, 20.0
        sz = compute_auto_font_size(
            text,
            font_name="Helv",
            field_width=fw,
            field_height=fh,
        )
        tw = get_text_width(text, font_name="Helv", font_size=sz)
        margin = 2.0
        assert tw <= fw - 2 * margin + 0.1  # small tolerance

    def test_multiline_auto_size(self):
        """Multiline auto-size considers wrapping."""
        text = "This is a\nmultiline text"
        sz = compute_auto_font_size(
            text,
            font_name="Helv",
            field_width=200,
            field_height=50,
            multiline=True,
        )
        assert 4.0 <= sz <= 50.0


# ===================================================================
# WinAnsiEncoding Tests
# ===================================================================


class TestWinAnsiEncoding:
    """Tests for WinAnsiEncoding mapping functions."""

    def test_ascii_passthrough(self):
        """ASCII range maps to itself."""
        assert _winansi_to_unicode(65) == 65
        assert _unicode_to_winansi(65) == 65

    def test_euro_sign(self):
        """Position 128 is Euro sign (U+20AC)."""
        assert _winansi_to_unicode(128) == 0x20AC
        assert _unicode_to_winansi(0x20AC) == 128

    def test_smart_quotes(self):
        """Smart quote positions map correctly."""
        assert _winansi_to_unicode(145) == 0x2018  # left single quote
        assert _winansi_to_unicode(146) == 0x2019  # right single quote
        assert _winansi_to_unicode(147) == 0x201C  # left double quote
        assert _winansi_to_unicode(148) == 0x201D  # right double quote

    def test_latin1_supplement(self):
        """160-255 range passes through."""
        assert _winansi_to_unicode(224) == 224  # agrave
        assert _unicode_to_winansi(224) == 224

    def test_unmappable_unicode(self):
        """Unmappable Unicode returns None."""
        assert _unicode_to_winansi(0x4E2D) is None  # Chinese character


# ===================================================================
# Text Wrap Tests
# ===================================================================


class TestWrapText:
    """Tests for _wrap_text()."""

    def test_no_wrapping_needed(self):
        """Short text stays on one line."""
        lines = _wrap_text("Hello", None, 12.0, 200.0, "Helv")
        assert len(lines) == 1
        assert lines[0] == "Hello"

    def test_explicit_newlines(self):
        """Explicit \\n creates line breaks."""
        lines = _wrap_text("Line1\nLine2\nLine3", None, 12.0, 200.0, "Helv")
        assert len(lines) == 3
        assert lines[0] == "Line1"
        assert lines[1] == "Line2"
        assert lines[2] == "Line3"

    def test_word_wrapping(self):
        """Long text wraps at word boundaries."""
        text = "The quick brown fox jumps over the lazy dog"
        lines = _wrap_text(text, None, 12.0, 80.0, "Helv")
        assert len(lines) > 1
        # All text should be preserved
        assert " ".join(lines) == text

    def test_cr_lf_handling(self):
        """\\r\\n, \\r, and \\n all treated as line breaks."""
        lines = _wrap_text("A\r\nB\rC\nD", None, 12.0, 200.0, "Helv")
        assert len(lines) == 4

    def test_empty_text(self):
        """Empty text returns single empty line."""
        lines = _wrap_text("", None, 12.0, 200.0, "Helv")
        assert lines == [""]


# ===================================================================
# PDF String Decode/Encode Tests
# ===================================================================


class TestDecodeEncode:
    """Tests for decode_pdf_string() and encode_for_content_stream()."""

    def test_decode_none(self):
        """None value decodes to empty string."""
        assert decode_pdf_string(None) == ""

    def test_decode_simple_string(self):
        """Simple pikepdf String decodes correctly."""
        s = pikepdf.String("Hello")
        assert decode_pdf_string(s) == "Hello"

    def test_encode_ascii(self):
        """ASCII text encodes to same bytes."""
        result = encode_for_content_stream("Hello")
        assert result == b"Hello"

    def test_encode_escapes_parens(self):
        """Parentheses are escaped in encoded output."""
        result = encode_for_content_stream("a(b)c")
        assert result == b"a\\(b\\)c"

    def test_encode_escapes_backslash(self):
        """Backslash is escaped."""
        result = encode_for_content_stream("a\\b")
        assert result == b"a\\\\b"

    def test_encode_unmappable_becomes_question(self):
        """Unmappable Unicode becomes '?'."""
        result = encode_for_content_stream("\u4e2d")
        assert result == b"?"

    def test_encode_decode_roundtrip_latin1(self):
        """Latin-1 characters roundtrip through encode/decode."""
        text = "Hello"
        encoded = encode_for_content_stream(text)
        assert encoded == b"Hello"


# ===================================================================
# Widths from Font Dict Tests
# ===================================================================


class TestReadWidthsFromFontDict:
    """Tests for _read_widths_from_font_dict()."""

    def test_basic_widths_array(self):
        """Reads /Widths with /FirstChar and /LastChar."""
        font_dict = Dictionary(
            FirstChar=65,
            LastChar=67,
            Widths=Array([700, 800, 900]),
        )
        widths = _read_widths_from_font_dict(font_dict)
        assert widths == {65: 700, 66: 800, 67: 900}

    def test_missing_widths(self):
        """Returns empty dict if /Widths is missing."""
        font_dict = Dictionary(FirstChar=65, LastChar=67)
        widths = _read_widths_from_font_dict(font_dict)
        assert widths == {}

    def test_none_font_dict(self):
        """Returns empty dict for None input."""
        assert _read_widths_from_font_dict(None) == {}
