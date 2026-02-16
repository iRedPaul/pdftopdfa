# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/widget_appearance.py."""

import pikepdf
import pytest
from conftest import resolve, save_and_reopen
from pikepdf import Array, Dictionary, Name, Stream

from pdftopdfa.sanitizers.annotations import ensure_appearance_streams
from pdftopdfa.sanitizers.widget_appearance import (
    _build_border_only_appearance,
    _build_button_appearance,
    _build_checkbox_appearance,
    _build_choice_field_appearance,
    _build_comb_field_appearance,
    _build_listbox_appearance,
    _build_multiline_text_appearance,
    _build_pushbutton_appearance,
    _build_radio_appearance,
    _build_signature_appearance,
    _build_single_line_text_appearance,
    _build_text_field_appearance,
    _build_unicode_to_code_map,
    _color_array_to_ops,
    _encode_cid_hex,
    _escape_pdf_string,
    _get_border_style,
    _get_inheritable,
    _get_on_state_name,
    _get_rotation,
    _is_cid_font,
    _parse_da_string,
    _remove_rich_text,
    _resolve_font_resource,
    _rotation_matrix,
    _text_operator,
    create_widget_appearance,
)


def _make_font_dict(pdf):
    """Create a minimal font dictionary for testing."""
    return pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/Type1"),
            BaseFont=Name("/Helvetica"),
        )
    )


def _make_acroform(pdf, font_name="Helv"):
    """Create a minimal AcroForm with a font resource."""
    font = _make_font_dict(pdf)
    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font
    return Dictionary(DR=Dictionary(Font=font_dict))


def _make_widget(pdf, extra=None, rect=None):
    """Create a widget annotation dictionary."""
    d = Dictionary(
        Type=Name.Annot,
        Subtype=Name.Widget,
        Rect=Array(rect or [0, 0, 200, 30]),
    )
    if extra:
        for k, v in extra.items():
            d[Name(k)] = v
    return d


# ===================================================================
# DA Parsing Tests
# ===================================================================


class TestParseDAString:
    """Tests for _parse_da_string()."""

    @pytest.mark.parametrize(
        ("da", "expected_font", "expected_size", "expected_color"),
        [
            ("0.5 g /Helv 12 Tf", "Helv", 12.0, "0.5 g"),
            ("1 0 0 rg /Helv 10 Tf", "Helv", 10.0, "1 0 0 rg"),
            ("0 0 0 1 k /Helv 14 Tf", "Helv", 14.0, "0 0 0 1 k"),
        ],
        ids=["grayscale", "rgb", "cmyk"],
    )
    def test_color_parsing(self, da, expected_font, expected_size, expected_color):
        """Parses color operators from DA string."""
        font_name, font_size, color_ops = _parse_da_string(da)
        assert font_name == expected_font
        assert font_size == expected_size
        assert expected_color in color_ops

    def test_missing_font(self):
        """Returns None font_name when no Tf operator."""
        font_name, font_size, color_ops = _parse_da_string("0 g")
        assert font_name is None
        assert font_size == 12.0

    def test_empty_string(self):
        """Returns defaults for empty DA string."""
        font_name, font_size, color_ops = _parse_da_string("")
        assert font_name is None
        assert font_size == 12.0
        assert color_ops == ""

    def test_none_input(self):
        """Returns defaults for None DA."""
        font_name, font_size, color_ops = _parse_da_string(None)
        assert font_name is None
        assert font_size == 12.0
        assert color_ops == ""

    def test_zero_font_size(self):
        """Auto-size 0 is preserved as 0.0 for callers to compute."""
        font_name, font_size, color_ops = _parse_da_string("/Helv 0 Tf")
        assert font_name == "Helv"
        assert font_size == 0.0

    def test_operator_order(self):
        """Color before Tf is correctly extracted."""
        font_name, font_size, color_ops = _parse_da_string("0 0 0 rg /Arial 9 Tf")
        assert font_name == "Arial"
        assert font_size == 9.0
        assert "0 0 0 rg" in color_ops


# ===================================================================
# Inheritable Attribute Tests
# ===================================================================


class TestGetInheritable:
    """Tests for _get_inheritable()."""

    def test_direct_attribute(self, make_pdf_with_page):
        """Returns attribute directly on annotation."""
        _ = make_pdf_with_page()
        annot = Dictionary(FT=Name("/Tx"))
        result = _get_inheritable(annot, "/FT")
        assert str(result) == "/Tx"

    def test_inherited_from_parent(self, make_pdf_with_page):
        """Returns attribute from parent when not on annotation."""
        pdf = make_pdf_with_page()
        parent = pdf.make_indirect(Dictionary(FT=Name("/Tx")))
        annot = Dictionary(Parent=parent)
        result = _get_inheritable(annot, "/FT")
        assert str(result) == "/Tx"

    def test_inherited_from_grandparent(self, make_pdf_with_page):
        """Returns attribute from grandparent."""
        pdf = make_pdf_with_page()
        grandparent = pdf.make_indirect(Dictionary(FT=Name("/Btn")))
        parent = pdf.make_indirect(Dictionary(Parent=grandparent))
        annot = Dictionary(Parent=parent)
        result = _get_inheritable(annot, "/FT")
        assert str(result) == "/Btn"

    def test_missing_attribute(self):
        """Returns None when attribute not found anywhere."""
        annot = Dictionary()
        result = _get_inheritable(annot, "/FT")
        assert result is None

    def test_acroform_fallback(self):
        """Falls back to AcroForm when not found in hierarchy."""
        annot = Dictionary()
        acroform = Dictionary(DA=pikepdf.String("/Helv 12 Tf"))
        result = _get_inheritable(annot, "/DA", acroform)
        assert result is not None

    def test_annot_overrides_acroform(self):
        """Annotation value takes precedence over AcroForm."""
        annot = Dictionary(Q=0)
        acroform = Dictionary(Q=2)
        result = _get_inheritable(annot, "/Q", acroform)
        assert int(result) == 0


# ===================================================================
# Color Array Tests
# ===================================================================


class TestColorArrayToOps:
    """Tests for _color_array_to_ops()."""

    @pytest.mark.parametrize(
        ("values", "stroke", "expected"),
        [
            ([0.5], False, "0.5 g"),
            ([0], True, "0 G"),
            ([1, 0, 0], False, "1 0 0 rg"),
            ([0, 0, 1], True, "0 0 1 RG"),
            ([0, 0, 0, 1], False, "0 0 0 1 k"),
            ([1, 0, 0, 0], True, "1 0 0 0 K"),
        ],
        ids=[
            "gray-fill",
            "gray-stroke",
            "rgb-fill",
            "rgb-stroke",
            "cmyk-fill",
            "cmyk-stroke",
        ],
    )
    def test_color_ops(self, values, stroke, expected):
        assert _color_array_to_ops(Array(values), stroke=stroke) == expected

    def test_empty_array(self):
        assert _color_array_to_ops(Array([]), stroke=False) == ""

    def test_none(self):
        assert _color_array_to_ops(None) == ""


# ===================================================================
# String Escaping Tests
# ===================================================================


class TestEscapePdfString:
    """Tests for _escape_pdf_string()."""

    def test_plain_text(self):
        assert _escape_pdf_string("Hello") == "Hello"

    def test_parentheses(self):
        assert _escape_pdf_string("a(b)c") == "a\\(b\\)c"

    def test_backslash(self):
        assert _escape_pdf_string("a\\b") == "a\\\\b"

    def test_none(self):
        assert _escape_pdf_string(None) == ""

    def test_mixed_special(self):
        assert _escape_pdf_string("(\\)") == "\\(\\\\\\)"


# ===================================================================
# Font Resource Resolution Tests
# ===================================================================


class TestResolveFontResource:
    """Tests for _resolve_font_resource()."""

    def test_font_from_acroform_dr(self, make_pdf_with_page):
        """Font resolved from AcroForm /DR."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf, "Helv")
        annot = Dictionary()
        result = _resolve_font_resource("Helv", annot, acroform)
        assert result is not None
        assert str(result.get("/BaseFont")) == "/Helvetica"

    def test_font_not_found(self, make_pdf_with_page):
        """Returns None for unknown font name."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf, "Helv")
        result = _resolve_font_resource("UnknownFont", Dictionary(), acroform)
        assert result is None

    def test_none_font_name(self):
        """Returns None for None font name."""
        result = _resolve_font_resource(None, Dictionary(), Dictionary())
        assert result is None

    def test_font_from_annot_dr(self, make_pdf_with_page):
        """Font resolved from annotation's own /DR."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        font_dict = Dictionary()
        font_dict[Name("/Helv")] = font
        annot = Dictionary(DR=Dictionary(Font=font_dict))
        result = _resolve_font_resource("Helv", annot, None)
        assert result is not None


# ===================================================================
# Text Field Appearance Tests
# ===================================================================


class TestBuildTextFieldAppearance:
    """Tests for _build_text_field_appearance()."""

    def test_text_field_with_value(self, make_pdf_with_page):
        """Text field with /V renders text in content stream."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Hello World"),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Hello World" in content
        assert b"Tf" in content
        assert b"BT" in content
        assert b"ET" in content

    def test_text_field_without_value(self, make_pdf_with_page):
        """Text field without /V renders empty text."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" in content
        assert b"() Tj" in content

    def test_text_field_with_background(self, make_pdf_with_page):
        """Text field with /MK /BG renders background color."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/MK": Dictionary(BG=Array([1, 1, 0.8])),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert b"rg" in content
        assert b"re f" in content

    def test_text_field_with_border(self, make_pdf_with_page):
        """Text field with /MK /BC renders border."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/MK": Dictionary(BC=Array([0, 0, 0])),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert b"RG" in content
        assert b"re S" in content

    def test_text_field_alignment_left(self, make_pdf_with_page):
        """Left alignment (Q=0) positions text at left margin."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test"),
                "/Q": 0,
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)

    def test_text_field_alignment_center(self, make_pdf_with_page):
        """Center alignment (Q=1)."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test"),
                "/Q": 1,
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)

    def test_text_field_alignment_right(self, make_pdf_with_page):
        """Right alignment (Q=2)."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test"),
                "/Q": 2,
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)

    def test_text_field_font_from_acroform(self, make_pdf_with_page):
        """Font resolved from AcroForm /DR when not on widget."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test"),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"/Helv" in content

    def test_text_field_fallback_no_font(self, make_pdf_with_page):
        """Falls back to border-only when font not in resources."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/UnknownFont 12 Tf"),
                "/V": pikepdf.String("Test"),
            },
        )
        result = _build_text_field_appearance(pdf, annot, None)
        assert isinstance(result, Stream)
        # Should not contain text operators since font is missing
        content = bytes(result.read_bytes())
        assert b"BT" not in content

    def test_text_field_fallback_no_da(self, make_pdf_with_page):
        """Falls back to border-only when no /DA."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Tx")})
        result = _build_text_field_appearance(pdf, annot, None)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" not in content


# ===================================================================
# Checkbox Appearance Tests
# ===================================================================


class TestBuildCheckboxAppearance:
    """Tests for _build_checkbox_appearance()."""

    def test_returns_state_dictionary(self, make_pdf_with_page):
        """Checkbox returns a Dictionary with Off and On states."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/AS": Name("/Yes"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_checkbox_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result
        assert "/Yes" in result

    def test_off_state_is_empty_box(self, make_pdf_with_page):
        """Off state renders a border rectangle without checkmark."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Btn")}, rect=[0, 0, 20, 20])
        result = _build_checkbox_appearance(pdf, annot)
        off = result.get("/Off")
        assert isinstance(off, Stream)
        content = bytes(off.read_bytes())
        assert b"re S" in content
        # No checkmark lines
        assert b"m" not in content or content.count(b"m") <= content.count(b"re")

    def test_on_state_has_checkmark(self, make_pdf_with_page):
        """On state renders border with checkmark lines."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/AS": Name("/Yes"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_checkbox_appearance(pdf, annot)
        on_key = "/Yes"
        on = result.get(on_key)
        assert isinstance(on, Stream)
        content = bytes(on.read_bytes())
        assert b"re S" in content
        # Checkmark uses line drawing (m, l, S)
        assert b" l " in content

    def test_checkbox_with_mk_colors(self, make_pdf_with_page):
        """Checkbox uses /MK /BC and /BG colors."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/MK": Dictionary(
                    BC=Array([1, 0, 0]),
                    BG=Array([1, 1, 1]),
                ),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_checkbox_appearance(pdf, annot)
        off = result.get("/Off")
        content = bytes(off.read_bytes())
        assert b"RG" in content  # border color
        assert b"rg" in content  # background color

    def test_checkbox_custom_on_state(self, make_pdf_with_page):
        """Checkbox with custom on-state name (not Yes)."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/AS": Name("/1"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_checkbox_appearance(pdf, annot)
        assert "/Off" in result
        assert "/1" in result


# ===================================================================
# Radio Button Appearance Tests
# ===================================================================


class TestBuildRadioAppearance:
    """Tests for _build_radio_appearance()."""

    def test_returns_state_dictionary(self, make_pdf_with_page):
        """Radio returns a Dictionary with Off and On states."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,  # Radio flag
                "/AS": Name("/Choice1"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_radio_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result
        assert "/Choice1" in result

    def test_off_state_is_circle(self, make_pdf_with_page):
        """Off state renders a circle (Bezier curves)."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_radio_appearance(pdf, annot)
        off = result.get("/Off")
        content = bytes(off.read_bytes())
        # Circle uses Bezier curves (c operator)
        assert b" c\n" in content or b" c " in content or content.endswith(b" c")

    def test_on_state_has_dot(self, make_pdf_with_page):
        """On state renders circle with filled dot."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,
                "/AS": Name("/Choice1"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_radio_appearance(pdf, annot)
        on = result.get("/Choice1")
        content = bytes(on.read_bytes())
        # Should have fill operator for the dot
        assert b" f" in content


# ===================================================================
# Pushbutton Appearance Tests
# ===================================================================


class TestBuildPushbuttonAppearance:
    """Tests for pushbutton via _build_button_appearance()."""

    def test_pushbutton_with_caption(self, make_pdf_with_page):
        """Pushbutton renders /MK /CA caption text."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,  # Pushbutton flag
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/MK": Dictionary(CA=pikepdf.String("Submit")),
            },
        )
        result = _build_button_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Submit" in content
        assert b"BT" in content

    def test_pushbutton_without_caption(self, make_pdf_with_page):
        """Pushbutton without caption renders border only."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        result = _build_button_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" not in content

    def test_pushbutton_no_font(self, make_pdf_with_page):
        """Pushbutton without font falls back to border only."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,
                "/MK": Dictionary(CA=pikepdf.String("Click")),
            },
        )
        result = _build_button_appearance(pdf, annot, None)
        assert isinstance(result, Stream)


# ===================================================================
# Choice Field Appearance Tests
# ===================================================================


class TestBuildChoiceFieldAppearance:
    """Tests for _build_choice_field_appearance()."""

    def test_combo_renders_like_text(self, make_pdf_with_page):
        """Combo choice field renders /V value like a text field."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/Ff": 1 << 17,  # Combo flag
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Option A"),
            },
        )
        result = _build_choice_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Option A" in content


# ===================================================================
# Signature Field Appearance Tests
# ===================================================================


class TestBuildSignatureAppearance:
    """Tests for _build_signature_appearance()."""

    def test_unsigned_signature_field(self, make_pdf_with_page):
        """Unsigned signature field (no /V) gets border-only appearance."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Sig"),
            },
        )
        result = _build_signature_appearance(pdf, annot)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" not in content
        assert str(result.get("/Subtype")) == "/Form"

    def test_signed_signature_field(self, make_pdf_with_page):
        """Signed signature field (has /V) still gets border-only appearance."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Sig"),
                "/V": Dictionary(Type=Name("/Sig")),
            },
        )
        result = _build_signature_appearance(pdf, annot)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" not in content

    def test_signature_field_zero_size(self, make_pdf_with_page):
        """Signature field with zero-size rect gets empty stream."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Sig"),
            },
            rect=[100, 100, 100, 100],
        )
        result = _build_signature_appearance(pdf, annot)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert content == b""

    def test_signature_field_with_border(self, make_pdf_with_page):
        """Signature field with /MK /BC renders border."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Sig"),
                "/MK": Dictionary(BC=Array([0, 0, 0])),
            },
        )
        result = _build_signature_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"re S" in content

    def test_signature_integration(self, make_pdf_with_page):
        """Signature widget gets appearance via ensure_appearance_streams."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Sig"),
                },
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        assert isinstance(n, Stream)
        content = bytes(n.read_bytes())
        assert b"BT" not in content


# ===================================================================
# On-State Name Detection Tests
# ===================================================================


class TestGetOnStateName:
    """Tests for _get_on_state_name()."""

    def test_from_as(self):
        """Gets on-state from /AS."""
        annot = Dictionary(AS=Name("/Yes"))
        assert _get_on_state_name(annot) == "Yes"

    def test_from_as_custom(self):
        """Gets custom on-state name from /AS."""
        annot = Dictionary(AS=Name("/1"))
        assert _get_on_state_name(annot) == "1"

    def test_default_yes(self):
        """Defaults to 'Yes' when no state info."""
        annot = Dictionary()
        assert _get_on_state_name(annot) == "Yes"

    def test_off_as_ignored(self):
        """Off /AS is ignored, defaults to Yes."""
        annot = Dictionary(AS=Name("/Off"))
        assert _get_on_state_name(annot) == "Yes"


# ===================================================================
# create_widget_appearance Dispatch Tests
# ===================================================================


class TestCreateWidgetAppearance:
    """Tests for create_widget_appearance() dispatch."""

    def test_dispatches_text_field(self, make_pdf_with_page):
        """Dispatches /FT /Tx to text builder."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Hello"),
            },
        )
        result = create_widget_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Hello" in content

    def test_dispatches_checkbox(self, make_pdf_with_page):
        """Dispatches /FT /Btn (no pushbutton/radio flags) to checkbox."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
            },
            rect=[0, 0, 20, 20],
        )
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result

    def test_dispatches_radio(self, make_pdf_with_page):
        """Dispatches /FT /Btn with radio flag to radio builder."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,
            },
            rect=[0, 0, 20, 20],
        )
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result

    def test_dispatches_choice(self, make_pdf_with_page):
        """Dispatches /FT /Ch to choice builder."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Option"),
            },
        )
        result = create_widget_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)

    def test_dispatches_signature(self, make_pdf_with_page):
        """Dispatches /FT /Sig to signature builder."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Sig")})
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Stream)
        # Signature fields produce border-only (no text operators)
        content = bytes(result.read_bytes())
        assert b"BT" not in content

    def test_unknown_ft_gets_border(self, make_pdf_with_page):
        """Unknown field type gets border-only appearance."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Unknown")})
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Stream)

    def test_no_ft_gets_border(self, make_pdf_with_page):
        """Missing field type gets border-only appearance."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf)
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Stream)

    def test_inherited_ft(self, make_pdf_with_page):
        """Field type inherited from /Parent."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        parent = pdf.make_indirect(
            Dictionary(
                FT=Name("/Tx"),
                DA=pikepdf.String("/Helv 12 Tf"),
            )
        )
        annot = _make_widget(
            pdf,
            {
                "/Parent": parent,
                "/V": pikepdf.String("Inherited"),
            },
        )
        result = create_widget_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Inherited" in content


# ===================================================================
# Integration Tests (ensure_appearance_streams)
# ===================================================================


class TestIntegration:
    """Integration tests: ensure_appearance_streams with Widget annotations."""

    def test_widget_gets_visible_appearance(self, make_pdf_with_page):
        """Widget annotation gets visible (non-empty) appearance."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        font_dict = Dictionary()
        font_dict[Name("/Helv")] = font
        acroform = Dictionary(DR=Dictionary(Font=font_dict))
        pdf.Root[Name("/AcroForm")] = pdf.make_indirect(acroform)
        annot = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Tx"),
                    "/DA": pikepdf.String("/Helv 12 Tf"),
                    "/V": pikepdf.String("Visible"),
                },
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        content = bytes(n.read_bytes())
        assert b"Visible" in content
        assert b"BT" in content

    def test_non_widget_gets_empty_appearance(self, make_pdf_with_page):
        """Non-Widget annotation gets empty appearance stream."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        content = bytes(n.read_bytes())
        assert content == b""

    def test_mixed_widget_and_non_widget(self, make_pdf_with_page):
        """Mixed annotations: widget gets visible, non-widget gets empty."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        font_dict = Dictionary()
        font_dict[Name("/Helv")] = font
        acroform = Dictionary(DR=Dictionary(Font=font_dict))
        pdf.Root[Name("/AcroForm")] = pdf.make_indirect(acroform)
        widget = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Tx"),
                    "/DA": pikepdf.String("/Helv 12 Tf"),
                    "/V": pikepdf.String("Field"),
                },
            )
        )
        text_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([widget, text_annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 2

    def test_existing_ap_preserved(self, make_pdf_with_page):
        """Existing /AP /N is not overwritten."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"existing content")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 200, 30])
        annot = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Tx"),
                    "/AP": Dictionary(N=stream),
                },
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 0

    def test_checkbox_integration(self, make_pdf_with_page):
        """Checkbox widget gets state dictionary as /AP /N."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Btn"),
                    "/AS": Name("/Yes"),
                },
                rect=[0, 0, 20, 20],
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        # Should be a dictionary with states
        assert isinstance(n, Dictionary)

    def test_widget_no_acroform(self, make_pdf_with_page):
        """Widget works even without AcroForm (border-only fallback)."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            _make_widget(
                pdf,
                {
                    "/FT": Name("/Tx"),
                    "/DA": pikepdf.String("/Helv 12 Tf"),
                },
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert "/AP" in resolved


# ===================================================================
# Edge Case Tests
# ===================================================================


class TestEdgeCases:
    """Edge case tests for widget appearance generation."""

    def test_missing_rect(self, make_pdf_with_page):
        """Widget without /Rect gets empty appearance."""
        pdf = make_pdf_with_page()
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
        )
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Stream)

    def test_zero_size_rect(self, make_pdf_with_page):
        """Widget with zero-size /Rect gets empty appearance."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Tx")}, rect=[100, 100, 100, 100])
        result = create_widget_appearance(pdf, annot)
        assert isinstance(result, Stream)

    def test_special_chars_in_value(self, make_pdf_with_page):
        """Special characters in /V are properly escaped."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test (with) parens & \\backslash"),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        # Parentheses should be escaped
        assert b"\\(" in content
        assert b"\\)" in content

    def test_long_text_value(self, make_pdf_with_page):
        """Long text value does not crash (clipping handles overflow)."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        long_text = "A" * 500
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String(long_text),
            },
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"W n" in content  # clipping operator

    def test_border_only_fallback(self, make_pdf_with_page):
        """Border-only appearance is a valid Form XObject."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Tx")})
        result = _build_border_only_appearance(pdf, annot)
        assert isinstance(result, Stream)
        assert str(result.get("/Type")) == "/XObject"
        assert str(result.get("/Subtype")) == "/Form"

    def test_small_checkbox_minimum_size(self, make_pdf_with_page):
        """Very small checkbox gets minimum dimensions."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
            },
            rect=[0, 0, 0, 0],
        )
        result = _build_checkbox_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result

    def test_radio_tiny_radius_fallback(self, make_pdf_with_page):
        """Very small radio falls back to checkbox style."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,
            },
            rect=[0, 0, 1, 1],
        )
        result = _build_radio_appearance(pdf, annot)
        assert isinstance(result, Dictionary)
        assert "/Off" in result


# ===================================================================
# Multiline Text Field Tests
# ===================================================================


class TestMultilineTextAppearance:
    """Tests for _build_multiline_text_appearance()."""

    def test_multiline_with_newlines(self, make_pdf_with_page):
        """Multiline text with \\n renders multiple Tj operators."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,  # Multiline flag
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/V": pikepdf.String("Line1\nLine2\nLine3"),
            },
            rect=[0, 0, 200, 100],
        )
        result = _build_multiline_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert content.count(b"Tj") >= 3
        assert b"Line1" in content
        assert b"Line2" in content
        assert b"Line3" in content

    def test_multiline_word_wrap(self, make_pdf_with_page):
        """Long text word-wraps to multiple lines."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        long_text = "The quick brown fox jumps over the lazy dog"
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String(long_text),
            },
            rect=[0, 0, 100, 60],
        )
        result = _build_multiline_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert content.count(b"Tj") > 1

    def test_multiline_auto_size(self, make_pdf_with_page):
        """Multiline with font_size=0 auto-sizes."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,
                "/DA": pikepdf.String("/Helv 0 Tf"),
                "/V": pikepdf.String("Auto\nsized\ntext"),
            },
            rect=[0, 0, 200, 60],
        )
        result = _build_multiline_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" in content
        assert b"Tj" in content

    def test_multiline_dispatch(self, make_pdf_with_page):
        """Multiline flag dispatches to multiline builder."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/V": pikepdf.String("Line1\nLine2"),
            },
            rect=[0, 0, 200, 60],
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert content.count(b"Tj") >= 2

    def test_multiline_empty_value(self, make_pdf_with_page):
        """Multiline with no value renders cleanly."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,
                "/DA": pikepdf.String("/Helv 10 Tf"),
            },
            rect=[0, 0, 200, 60],
        )
        result = _build_multiline_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)


# ===================================================================
# Comb Field Tests
# ===================================================================


class TestCombFieldAppearance:
    """Tests for _build_comb_field_appearance()."""

    def test_comb_evenly_spaced(self, make_pdf_with_page):
        """Comb field characters are rendered with Td offsets."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,  # Comb flag
                "/MaxLen": 5,
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("ABC"),
            },
            rect=[0, 0, 200, 30],
        )
        result = _build_comb_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        # Should have 3 characters rendered
        assert content.count(b"Tj") == 3
        assert b"(A)" in content
        assert b"(B)" in content
        assert b"(C)" in content

    def test_comb_divider_lines(self, make_pdf_with_page):
        """Comb field draws vertical divider lines."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,
                "/MaxLen": 4,
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("AB"),
                "/MK": Dictionary(BC=Array([0, 0, 0])),
            },
            rect=[0, 0, 200, 30],
        )
        result = _build_comb_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        # Should have 3 vertical dividers (MaxLen-1)
        assert content.count(b"l S") >= 3

    def test_comb_truncates_to_maxlen(self, make_pdf_with_page):
        """Comb field truncates text to MaxLen."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,
                "/MaxLen": 3,
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("ABCDE"),
            },
            rect=[0, 0, 200, 30],
        )
        result = _build_comb_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert content.count(b"Tj") == 3  # Only 3 chars
        assert b"(D)" not in content

    def test_comb_dispatch(self, make_pdf_with_page):
        """Comb flag + MaxLen dispatches to comb builder."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,
                "/MaxLen": 5,
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("AB"),
            },
            rect=[0, 0, 200, 30],
        )
        result = _build_text_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert content.count(b"Tj") == 2

    def test_comb_auto_size(self, make_pdf_with_page):
        """Comb field with font_size=0 auto-sizes."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,
                "/MaxLen": 5,
                "/DA": pikepdf.String("/Helv 0 Tf"),
                "/V": pikepdf.String("AB"),
            },
            rect=[0, 0, 200, 30],
        )
        result = _build_comb_field_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Tj" in content


# ===================================================================
# Border Style Tests
# ===================================================================


class TestBorderStyles:
    """Tests for border style support."""

    def test_solid_border(self, make_pdf_with_page):
        """Solid border (default) uses re S."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/MK": Dictionary(BC=Array([0, 0, 0])),
                "/BS": Dictionary(W=2, S=Name("/S")),
            },
        )
        result = _build_border_only_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"re S" in content

    def test_dashed_border(self, make_pdf_with_page):
        """Dashed border uses dash pattern."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/MK": Dictionary(BC=Array([0, 0, 0])),
                "/BS": Dictionary(W=1, S=Name("/D")),
            },
        )
        result = _build_border_only_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"[3] 0 d" in content

    def test_beveled_border(self, make_pdf_with_page):
        """Beveled border draws light and dark edges."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/MK": Dictionary(BC=Array([0, 0, 0])),
                "/BS": Dictionary(W=2, S=Name("/B")),
            },
        )
        result = _build_border_only_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"1 g" in content  # light edge
        assert b"0.5 g" in content  # dark edge

    def test_inset_border(self, make_pdf_with_page):
        """Inset border draws inverted light/dark edges."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/MK": Dictionary(BC=Array([0, 0, 0])),
                "/BS": Dictionary(W=2, S=Name("/I")),
            },
        )
        result = _build_border_only_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"0.5 g" in content
        assert b"1 g" in content

    def test_underline_border(self, make_pdf_with_page):
        """Underline border draws only bottom line."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/MK": Dictionary(BC=Array([0, 0, 0])),
                "/BS": Dictionary(W=1, S=Name("/U")),
            },
        )
        result = _build_border_only_appearance(pdf, annot)
        content = bytes(result.read_bytes())
        assert b"l S" in content
        # No rectangle stroke
        assert b"re S" not in content

    def test_get_border_style_default(self):
        """Default border style is 'S'."""
        annot = Dictionary()
        assert _get_border_style(annot) == "S"

    def test_get_border_style_from_bs(self):
        """Reads border style from /BS /S."""
        annot = Dictionary(BS=Dictionary(S=Name("/D")))
        assert _get_border_style(annot) == "D"


# ===================================================================
# Rotation Tests
# ===================================================================


class TestRotation:
    """Tests for widget rotation."""

    def test_no_rotation(self):
        """Default rotation is 0."""
        annot = Dictionary()
        assert _get_rotation(annot) == 0

    def test_rotation_90(self):
        """Reads 90-degree rotation from /MK /R."""
        annot = Dictionary(MK=Dictionary(R=90))
        assert _get_rotation(annot) == 90

    def test_rotation_180(self):
        annot = Dictionary(MK=Dictionary(R=180))
        assert _get_rotation(annot) == 180

    def test_rotation_270(self):
        annot = Dictionary(MK=Dictionary(R=270))
        assert _get_rotation(annot) == 270

    def test_rotation_matrix_0(self):
        """0 degrees returns None (no matrix needed)."""
        assert _rotation_matrix(0, 200, 30) is None

    def test_rotation_matrix_90(self):
        """90 degrees produces correct matrix."""
        m = _rotation_matrix(90, 200, 30)
        assert m == [0, 1, -1, 0, 200, 0]

    def test_rotation_matrix_180(self):
        m = _rotation_matrix(180, 200, 30)
        assert m == [-1, 0, 0, -1, 200, 30]

    def test_rotation_matrix_270(self):
        m = _rotation_matrix(270, 200, 30)
        assert m == [0, -1, 1, 0, 0, 30]

    def test_rotated_text_field(self, make_pdf_with_page):
        """90-degree rotated text field gets /Matrix entry."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Rotated"),
                "/MK": Dictionary(R=90),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        matrix = result.get("/Matrix")
        assert matrix is not None
        content = bytes(result.read_bytes())
        assert b"Rotated" in content

    def test_180_rotated_text_field(self, make_pdf_with_page):
        """180-degree rotated text field gets /Matrix entry."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Flipped"),
                "/MK": Dictionary(R=180),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        matrix = result.get("/Matrix")
        assert matrix is not None

    def test_no_rotation_no_matrix(self, make_pdf_with_page):
        """Non-rotated text field has no /Matrix."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Normal"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        matrix = result.get("/Matrix")
        assert matrix is None


# ===================================================================
# Listbox Tests
# ===================================================================


class TestListboxAppearance:
    """Tests for _build_listbox_appearance()."""

    def test_listbox_renders_options(self, make_pdf_with_page):
        """Listbox renders visible options."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/Opt": Array(
                    [
                        pikepdf.String("Alpha"),
                        pikepdf.String("Beta"),
                        pikepdf.String("Gamma"),
                    ]
                ),
                "/V": pikepdf.String("Beta"),
            },
            rect=[0, 0, 150, 80],
        )
        result = _build_listbox_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Alpha" in content
        assert b"Beta" in content
        assert b"Gamma" in content

    def test_listbox_highlight_selected(self, make_pdf_with_page):
        """Listbox highlights selected option with color."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/Opt": Array(
                    [
                        pikepdf.String("Alpha"),
                        pikepdf.String("Beta"),
                    ]
                ),
                "/V": pikepdf.String("Beta"),
            },
            rect=[0, 0, 150, 50],
        )
        result = _build_listbox_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        # Highlight color (blue background)
        assert b"0 0 0.6 rg" in content
        # White text for selected
        assert b"1 g" in content

    def test_listbox_dispatch(self, make_pdf_with_page):
        """Choice field without Combo flag dispatches to listbox."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/Ff": 0,  # No Combo flag
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/Opt": Array([pikepdf.String("One"), pikepdf.String("Two")]),
            },
            rect=[0, 0, 150, 50],
        )
        result = _build_choice_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        assert b"One" in content
        assert b"Two" in content

    def test_listbox_empty_options(self, make_pdf_with_page):
        """Listbox with no options renders cleanly."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/DA": pikepdf.String("/Helv 10 Tf"),
            },
            rect=[0, 0, 150, 50],
        )
        result = _build_listbox_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)

    def test_listbox_scroll_offset(self, make_pdf_with_page):
        """Listbox with /TI skips initial options."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/DA": pikepdf.String("/Helv 10 Tf"),
                "/Opt": Array(
                    [
                        pikepdf.String("First"),
                        pikepdf.String("Second"),
                        pikepdf.String("Third"),
                    ]
                ),
                "/TI": 1,  # Start from second option
            },
            rect=[0, 0, 150, 30],
        )
        result = _build_listbox_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes())
        # "Second" should be visible, "First" may not be
        assert b"Second" in content


# ===================================================================
# Auto-Size Text Field Tests
# ===================================================================


class TestAutoSizeTextField:
    """Tests for auto-size (font_size=0) in text fields."""

    def test_auto_size_single_line(self, make_pdf_with_page):
        """Auto-size computes a font size that fits."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 0 Tf"),
                "/V": pikepdf.String("Auto sized"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"BT" in content
        assert b"Auto sized" in content
        # Font size in Tf should not be 0
        assert b"/Helv 0 Tf" not in content

    def test_auto_size_pushbutton(self, make_pdf_with_page):
        """Pushbutton with auto-size computes font size."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,
                "/DA": pikepdf.String("/Helv 0 Tf"),
                "/MK": Dictionary(CA=pikepdf.String("Click Me")),
            },
        )
        result = create_widget_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        content = bytes(result.read_bytes())
        assert b"Click Me" in content
        assert b"/Helv 0 Tf" not in content


# ===================================================================
# Real Metrics Tests
# ===================================================================


class TestRealMetrics:
    """Tests verifying real font metrics are used."""

    def test_narrow_text_not_clipped(self, make_pdf_with_page):
        """Narrow text (e.g. 'iiii') should not be placed like wide text."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        # "iiii" in Helvetica is much narrower than "MMMM"
        annot_narrow = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("iiii"),
                "/Q": 1,  # center
            },
        )
        annot_wide = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("MMMM"),
                "/Q": 1,  # center
            },
        )
        r_narrow = _build_single_line_text_appearance(pdf, annot_narrow, acroform)
        r_wide = _build_single_line_text_appearance(pdf, annot_wide, acroform)
        c_narrow = bytes(r_narrow.read_bytes()).decode("latin-1")
        c_wide = bytes(r_wide.read_bytes()).decode("latin-1")
        # Extract Td x-positions (the centering offset should differ)
        import re

        m_narrow = re.search(r"([\d.]+)\s+[\d.]+\s+Td", c_narrow)
        m_wide = re.search(r"([\d.]+)\s+[\d.]+\s+Td", c_wide)
        assert m_narrow and m_wide
        # "iiii" is narrower, so its x-offset for centering should be larger
        assert float(m_narrow.group(1)) > float(m_wide.group(1))


# ===================================================================
# Rich Text (/RV) Removal Tests
# ===================================================================


class TestRemoveRichText:
    """Tests for _remove_rich_text() and /RV removal during appearance gen."""

    def test_rv_removed_from_annotation(self, make_pdf_with_page):
        """/RV is removed directly from the annotation dict."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Hello"),
                "/RV": pikepdf.String("<body>Hello</body>"),
            },
        )
        _remove_rich_text(annot)
        assert "/RV" not in annot

    def test_rv_removed_from_parent(self, make_pdf_with_page):
        """/RV is removed from parent field dict."""
        pdf = make_pdf_with_page()
        parent = pdf.make_indirect(
            Dictionary(
                FT=Name("/Tx"),
                DA=pikepdf.String("/Helv 12 Tf"),
                RV=pikepdf.String("<body>Rich</body>"),
            )
        )
        annot = _make_widget(pdf, {"/Parent": parent})
        _remove_rich_text(annot)
        parent_resolved = resolve(parent)
        assert "/RV" not in parent_resolved

    def test_rv_removed_during_create_widget_appearance(self, make_pdf_with_page):
        """/RV is removed when create_widget_appearance processes /Tx."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
                "/V": pikepdf.String("Test"),
                "/RV": pikepdf.String("<body>Test</body>"),
            },
        )
        create_widget_appearance(pdf, annot, acroform)
        assert "/RV" not in annot

    def test_rv_not_removed_from_non_text_field(self, make_pdf_with_page):
        """/RV on non-Tx fields is not touched."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/RV": pikepdf.String("<body>Btn</body>"),
            },
            rect=[0, 0, 20, 20],
        )
        create_widget_appearance(pdf, annot)
        assert "/RV" in annot

    def test_no_rv_no_error(self, make_pdf_with_page):
        """No error when /RV is absent."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        _remove_rich_text(annot)
        assert "/RV" not in annot


# ===================================================================
# Resources Presence Tests
# ===================================================================


class TestWidgetAppearanceResources:
    """Verify generated AP streams always include /Resources."""

    def test_checkbox_streams_have_resources(self, make_pdf_with_page):
        """Checkbox off and on streams have /Resources dictionary."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/AS": Name("/Yes"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_checkbox_appearance(pdf, annot)
        for key in ("/Off", "/Yes"):
            stream = result[key]
            assert stream.get(Name.Resources) is not None, (
                f"Checkbox {key} stream missing /Resources"
            )

    def test_radio_streams_have_resources(self, make_pdf_with_page):
        """Radio off and on streams have /Resources dictionary."""
        pdf = make_pdf_with_page()
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 15,
                "/AS": Name("/Choice1"),
            },
            rect=[0, 0, 20, 20],
        )
        result = _build_radio_appearance(pdf, annot)
        for key in ("/Off", "/Choice1"):
            stream = result[key]
            assert stream.get(Name.Resources) is not None, (
                f"Radio {key} stream missing /Resources"
            )

    def test_border_only_stream_has_resources(self, make_pdf_with_page):
        """Border-only stream has /Resources dictionary."""
        pdf = make_pdf_with_page()
        annot = _make_widget(pdf, {"/FT": Name("/Tx")})
        result = _build_border_only_appearance(pdf, annot)
        assert result.get(Name.Resources) is not None

    def test_pushbutton_no_caption_has_resources(self, make_pdf_with_page):
        """Pushbutton without caption has /Resources dictionary."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        result = _build_button_appearance(pdf, annot, acroform)
        assert isinstance(result, Stream)
        assert result.get(Name.Resources) is not None


# ===================================================================
# CID Font / Non-Latin-1 Text Encoding Tests
# ===================================================================


def _make_tounicode_cmap(*mappings):
    """Build a minimal ToUnicode CMap stream bytes from (code, unicode) pairs."""
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
        f"{len(mappings)} beginbfchar",
    ]
    for code, uni in mappings:
        lines.append(f"<{code:04X}> <{uni:04X}>")
    lines.extend(
        [
            "endbfchar",
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


def _make_cid_font(pdf, font_name="CIDFont", tounicode_data=None):
    """Create a Type0/CID font dictionary for testing.

    Args:
        pdf: pikepdf Pdf object.
        font_name: Name to use for the font.
        tounicode_data: Optional bytes for /ToUnicode stream.

    Returns:
        Indirect font dictionary reference.
    """
    cid_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/" + font_name),
        )
    )
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/Type0"),
            BaseFont=Name("/" + font_name),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cid_font]),
        )
    )
    if tounicode_data is not None:
        font[Name("/ToUnicode")] = pdf.make_stream(tounicode_data)
    return font


def _make_cid_acroform(pdf, font_name="CIDFont", tounicode_data=None):
    """Create an AcroForm with a CID font resource."""
    font = _make_cid_font(pdf, font_name, tounicode_data)
    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font
    return Dictionary(DR=Dictionary(Font=font_dict))


class TestCIDFontDetection:
    """Tests for _is_cid_font()."""

    def test_type0_is_cid(self, make_pdf_with_page):
        """Type0 font is detected as CID."""
        pdf = make_pdf_with_page()
        font = _make_cid_font(pdf, "TestCID")
        assert _is_cid_font(font) is True

    def test_type1_is_not_cid(self, make_pdf_with_page):
        """Type1 font is not CID."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        assert _is_cid_font(font) is False

    def test_none_font_is_not_cid(self):
        """None font resource returns False."""
        assert _is_cid_font(None) is False


class TestUnicodeToCodeMap:
    """Tests for _build_unicode_to_code_map()."""

    def test_with_tounicode_cmap(self, make_pdf_with_page):
        """Builds reverse mapping from ToUnicode CMap."""
        pdf = make_pdf_with_page()
        cmap_data = _make_tounicode_cmap(
            (0x0001, 0x4E2D),  # code 1 -> U+4E2D (中)
            (0x0002, 0x6587),  # code 2 -> U+6587 (文)
        )
        font = _make_cid_font(pdf, "TestCID", cmap_data)
        result = _build_unicode_to_code_map(font)
        assert result[0x4E2D] == 1
        assert result[0x6587] == 2

    def test_without_tounicode(self, make_pdf_with_page):
        """Returns empty dict when no ToUnicode CMap."""
        pdf = make_pdf_with_page()
        font = _make_cid_font(pdf, "TestCID")
        result = _build_unicode_to_code_map(font)
        assert result == {}


class TestCIDHexEncoding:
    """Tests for _encode_cid_hex()."""

    def test_with_tounicode(self, make_pdf_with_page):
        """Encodes using reverse ToUnicode mapping."""
        pdf = make_pdf_with_page()
        cmap_data = _make_tounicode_cmap(
            (0x0001, 0x4E2D),
            (0x0002, 0x6587),
        )
        font = _make_cid_font(pdf, "TestCID", cmap_data)
        result = _encode_cid_hex("\u4e2d\u6587", font)
        assert result == "00010002"

    def test_identity_fallback(self, make_pdf_with_page):
        """Falls back to ord(ch) without ToUnicode CMap."""
        pdf = make_pdf_with_page()
        font = _make_cid_font(pdf, "TestCID")
        result = _encode_cid_hex("\u4e2d", font)
        assert result == "4E2D"

    def test_ascii_identity(self, make_pdf_with_page):
        """ASCII chars use codepoint as code in identity fallback."""
        pdf = make_pdf_with_page()
        font = _make_cid_font(pdf, "TestCID")
        result = _encode_cid_hex("A", font)
        assert result == "0041"


class TestTextOperator:
    """Tests for _text_operator()."""

    def test_cid_font_hex_output(self, make_pdf_with_page):
        """CID font produces hex string operator."""
        pdf = make_pdf_with_page()
        font = _make_cid_font(pdf, "TestCID")
        result = _text_operator("\u4e2d", font)
        assert result == "<4E2D> Tj"

    def test_simple_font_parenthesized(self, make_pdf_with_page):
        """Simple font produces parenthesized string operator."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        result = _text_operator("hello", font)
        assert result == "(hello) Tj"

    def test_simple_font_non_latin1_replaced(self, make_pdf_with_page):
        """Non-Latin-1 chars in simple font are replaced with '?'."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        result = _text_operator("\u4e2d", font)
        assert result == "(?) Tj"

    def test_simple_font_escapes_special(self, make_pdf_with_page):
        """Parentheses and backslashes are escaped for simple fonts."""
        pdf = make_pdf_with_page()
        font = _make_font_dict(pdf)
        result = _text_operator("a(b)c\\d", font)
        assert result == r"(a\(b\)c\\d) Tj"


class TestNonLatin1TextFields:
    """Integration tests for non-Latin-1 text in widget appearance streams."""

    def test_cjk_single_line_no_crash(self, make_pdf_with_page):
        """CJK text in single-line field does not crash (no UnicodeEncodeError)."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("\u4e2d\u6587"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "<" in content
        assert "> Tj" in content

    def test_cjk_multiline_no_crash(self, make_pdf_with_page):
        """CJK text in multiline field does not crash."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 12,  # Multiline flag
                "/V": pikepdf.String("\u4e2d\u6587\n\u65b0\u884c"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
            rect=[0, 0, 200, 100],
        )
        result = _build_multiline_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "> Tj" in content

    def test_cjk_comb_field_no_crash(self, make_pdf_with_page):
        """CJK text in comb field does not crash."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/Ff": 1 << 24,  # Comb flag
                "/MaxLen": 4,
                "/V": pikepdf.String("\u4e2d\u6587"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = _build_comb_field_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "> Tj" in content

    def test_cjk_pushbutton_no_crash(self, make_pdf_with_page):
        """CJK caption on pushbutton does not crash."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        mk = Dictionary()
        mk[Name("/CA")] = pikepdf.String("\u63d0\u4ea4")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Btn"),
                "/Ff": 1 << 16,  # Pushbutton flag
                "/MK": mk,
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = _build_pushbutton_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "> Tj" in content

    def test_cjk_listbox_no_crash(self, make_pdf_with_page):
        """CJK options in listbox do not crash."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Ch"),
                "/Opt": Array(
                    [
                        pikepdf.String("\u9009\u98791"),
                        pikepdf.String("\u9009\u98792"),
                    ]
                ),
                "/V": pikepdf.String("\u9009\u98791"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
            rect=[0, 0, 200, 100],
        )
        result = _build_listbox_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "> Tj" in content

    def test_non_latin1_simple_font_uses_replacement(self, make_pdf_with_page):
        """Non-Latin-1 text in simple font uses '?' replacement, not crash."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("\u4e2d\u6587"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        # Should have '?' replacements, not crash
        assert "(?" in content or "(??" in content

    def test_latin1_text_unchanged(self, make_pdf_with_page):
        """Latin-1 text still works with parenthesized string."""
        pdf = make_pdf_with_page()
        acroform = _make_acroform(pdf)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("Hello World"),
                "/DA": pikepdf.String("/Helv 12 Tf"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "(Hello World) Tj" in content

    def test_cid_font_without_tounicode_uses_identity(self, make_pdf_with_page):
        """CID font without ToUnicode falls back to identity mapping."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("\u4e2d"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        # U+4E2D identity -> hex 4E2D
        assert "<4E2D> Tj" in content

    def test_cid_font_with_tounicode_uses_mapped_codes(self, make_pdf_with_page):
        """CID font with ToUnicode uses mapped character codes."""
        pdf = make_pdf_with_page()
        cmap_data = _make_tounicode_cmap(
            (0x0005, 0x4E2D),
            (0x0006, 0x6587),
        )
        acroform = _make_cid_acroform(pdf, "CIDFont", cmap_data)
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("\u4e2d\u6587"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = _build_single_line_text_appearance(pdf, annot, acroform)
        content = bytes(result.read_bytes()).decode("latin-1")
        # Should use mapped codes 0005 and 0006
        assert "<00050006> Tj" in content

    def test_create_widget_appearance_cjk_no_fallback(self, make_pdf_with_page):
        """Top-level create_widget_appearance with CJK text doesn't fall back."""
        pdf = make_pdf_with_page()
        acroform = _make_cid_acroform(pdf, "CIDFont")
        annot = _make_widget(
            pdf,
            {
                "/FT": Name("/Tx"),
                "/V": pikepdf.String("\u4e2d\u6587"),
                "/DA": pikepdf.String("/CIDFont 12 Tf"),
            },
        )
        result = create_widget_appearance(pdf, annot, acroform)
        # Should be a full text appearance, not border-only fallback
        content = bytes(result.read_bytes()).decode("latin-1")
        assert "BT" in content
        assert "> Tj" in content
