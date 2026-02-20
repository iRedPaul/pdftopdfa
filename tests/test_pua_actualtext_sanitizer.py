# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for PUA ActualText sanitizer (rule 6.2.11.7.3-1)."""

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.pua_actualtext import (
    _is_pua,
    sanitize_pua_actualtext,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_tounicode_cmap_8bit(code_to_unicode):
    """Builds a raw 8-bit ToUnicode CMap."""
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
    """Builds a raw 16-bit ToUnicode CMap."""
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


def _make_page_with_font_and_content(pdf, font, content_bytes):
    """Creates a page with a font resource and content stream."""
    stream = pdf.make_stream(content_bytes)
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
            Contents=stream,
        )
    )
    pdf.pages.append(page)
    return page


def _parse_instructions(page):
    """Parse page content stream into (operator_str, operands) list."""
    contents = page.obj.get("/Contents")
    if contents is None:
        return []
    try:
        contents = contents.get_object()
    except (AttributeError, TypeError):
        pass
    result = []
    for item in pikepdf.parse_content_stream(contents):
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            result.append(("INLINE_IMAGE", []))
        else:
            result.append((str(item.operator), item.operands))
    return result


def _get_operators(page):
    """Return just the operator names from a page's content stream."""
    return [op for op, _ in _parse_instructions(page)]


def _find_bdc_actualtext(page):
    """Find BDC instructions with /ActualText and return their values."""
    results = []
    for op, operands in _parse_instructions(page):
        if op == "BDC" and len(operands) >= 2:
            props = operands[1]
            if isinstance(props, Dictionary):
                at = props.get("/ActualText")
                if at is not None:
                    results.append(bytes(at))
    return results


# ---------------------------------------------------------------------------
# TestIsPua
# ---------------------------------------------------------------------------


class TestIsPua:
    """Tests for _is_pua() boundary detection."""

    def test_bmp_pua_start(self):
        assert _is_pua(0xE000) is True

    def test_bmp_pua_end(self):
        assert _is_pua(0xF8FF) is True

    def test_bmp_pua_middle(self):
        assert _is_pua(0xE100) is True

    def test_below_bmp_pua(self):
        assert _is_pua(0xDFFF) is False

    def test_above_bmp_pua(self):
        assert _is_pua(0xF900) is False

    def test_supplementary_a_start(self):
        assert _is_pua(0xF0000) is True

    def test_supplementary_a_end(self):
        assert _is_pua(0xFFFFD) is True

    def test_below_supplementary_a(self):
        assert _is_pua(0xEFFFF) is False

    def test_above_supplementary_a(self):
        assert _is_pua(0xFFFFE) is False

    def test_supplementary_b_start(self):
        assert _is_pua(0x100000) is True

    def test_supplementary_b_end(self):
        assert _is_pua(0x10FFFD) is True

    def test_above_supplementary_b(self):
        assert _is_pua(0x10FFFE) is False

    def test_ascii_not_pua(self):
        assert _is_pua(0x0041) is False

    def test_cjk_not_pua(self):
        assert _is_pua(0x4E00) is False

    def test_zero_not_pua(self):
        assert _is_pua(0) is False


# ---------------------------------------------------------------------------
# TestSimpleFontWrapping
# ---------------------------------------------------------------------------


class TestSimpleFontWrapping:
    """Tests for simple font PUA wrapping."""

    def test_tj_with_pua_code_wrapped(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", "Tj", "EMC", "ET"]

    def test_non_pua_unchanged(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0x42: 0x0042}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf (AB) Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

        ops = _get_operators(page)
        assert "BDC" not in ops

    def test_mixed_pua_and_non_pua(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0x42: 0x0042, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        # String: A (0x41), \xe0 (PUA), B (0x42)
        content = b"BT /F1 12 Tf <41E042> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", "Tj", "EMC", "ET"]

    def test_multiple_text_operators(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000, 0xE1: 0xE001}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj <E1> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 2

        ops = _get_operators(page)
        assert ops == [
            "BT",
            "Tf",
            "BDC",
            "Tj",
            "EMC",
            "BDC",
            "Tj",
            "EMC",
            "ET",
        ]

    def test_result_counter(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1
        assert "pua_actualtext_warnings" in result


# ---------------------------------------------------------------------------
# TestCIDFontWrapping
# ---------------------------------------------------------------------------


class TestCIDFontWrapping:
    """Tests for CIDFont (Type0) PUA wrapping with 2-byte codes."""

    def test_cidfont_pua_wrapped(self):
        pdf = new_pdf()
        # CID 0x00E0 → U+E000 (PUA)
        mapping = {0x0041: 0x0041, 0x00E0: 0xE000}
        font = _make_cidfont_with_tounicode(pdf, mapping)
        # 2-byte hex: 00E0
        content = b"BT /F1 12 Tf <00E0> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", "Tj", "EMC", "ET"]

    def test_cidfont_non_pua_unchanged(self):
        pdf = new_pdf()
        mapping = {0x0041: 0x0041, 0x0042: 0x0042}
        font = _make_cidfont_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <00410042> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

        ops = _get_operators(page)
        assert "BDC" not in ops


# ---------------------------------------------------------------------------
# TestTJArrayWrapping
# ---------------------------------------------------------------------------


class TestTJArrayWrapping:
    """Tests for TJ array operator wrapping."""

    def test_tj_array_with_pua_element_wrapped(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        # TJ array: [(A) 100 (\xe0)]
        content = b"BT /F1 12 Tf [(A) 100 <E0>] TJ ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", "TJ", "EMC", "ET"]

    def test_tj_array_all_non_pua_unchanged(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0x42: 0x0042}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf [(A) 100 (B)] TJ ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

        ops = _get_operators(page)
        assert "BDC" not in ops


# ---------------------------------------------------------------------------
# TestQuoteOperators
# ---------------------------------------------------------------------------


class TestQuoteOperators:
    """Tests for ' and \" operators with PUA codes."""

    def test_single_quote_with_pua(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> ' ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", "'", "EMC", "ET"]

    def test_double_quote_with_pua(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b'BT /F1 12 Tf 1 2 <E0> " ET'
        page = _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

        ops = _get_operators(page)
        assert ops == ["BT", "Tf", "BDC", '"', "EMC", "ET"]


# ---------------------------------------------------------------------------
# TestExistingActualText
# ---------------------------------------------------------------------------


class TestExistingActualText:
    """Tests that existing /ActualText BDC scopes prevent double-wrapping."""

    def test_already_inside_actualtext_not_double_wrapped(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # Build content with existing BDC /ActualText wrapper
        content = b"BT /F1 12 Tf /Span <</ActualText <FEFF0041>>> BDC <E0> Tj EMC ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_bdc_without_actualtext_still_wrapped(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # BDC with /MCID but no /ActualText
        content = b"BT /F1 12 Tf /Span <</MCID 0>> BDC <E0> Tj EMC ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

    def test_wrapping_resumes_after_emc(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000, 0xE1: 0xE001}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # First Tj inside /ActualText (covered), second outside (not covered)
        content = (
            b"BT /F1 12 Tf /Span <</ActualText <FEFF0041>>> BDC <E0> Tj EMC <E1> Tj ET"
        )
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        # Only the second Tj should be wrapped
        assert result["pua_actualtext_added"] == 1


# ---------------------------------------------------------------------------
# TestActualTextResolution
# ---------------------------------------------------------------------------


class TestActualTextResolution:
    """Tests for ActualText content and encoding."""

    def test_actualtext_utf16be_with_bom(self):
        pdf = new_pdf()
        # Code 0xE0 → U+E000 (PUA), code 0x41 → U+0041
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        sanitize_pua_actualtext(pdf)

        at_values = _find_bdc_actualtext(page)
        assert len(at_values) == 1
        raw = at_values[0]
        # Must start with UTF-16BE BOM
        assert raw[:2] == b"\xfe\xff"

    def test_non_pua_chars_in_actualtext(self):
        pdf = new_pdf()
        # Mixed: A (non-PUA) and 0xE0 (PUA, unresolvable)
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <41E0> Tj ET"
        page = _make_page_with_font_and_content(pdf, font, content)

        sanitize_pua_actualtext(pdf)

        at_values = _find_bdc_actualtext(page)
        assert len(at_values) == 1
        raw = at_values[0]
        # BOM + "A" in UTF-16BE (PUA char omitted as unresolvable)
        text = raw[2:].decode("utf-16-be")
        assert "A" in text

    def test_unresolvable_pua_warning(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        # Font without /Encoding — PUA can't be resolved
        cmap_data = _make_tounicode_cmap_8bit(mapping)
        tounicode = pdf.make_stream(cmap_data)
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FirstChar=0,
            LastChar=255,
            ToUnicode=tounicode,
        )
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        # Font has no encoding to resolve PUA
        assert result["pua_actualtext_warnings"] >= 1


# ---------------------------------------------------------------------------
# TestTraversal
# ---------------------------------------------------------------------------


class TestTraversal:
    """Tests for traversal into Form XObjects, AP streams, etc."""

    def test_form_xobject_stream(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # Create Form XObject with text
        form_content = b"BT /F1 12 Tf <E0> Tj ET"
        form_stream = pdf.make_stream(form_content)
        form_stream[Name("/Subtype")] = Name("/Form")
        form_stream[Name("/BBox")] = Array([0, 0, 100, 100])
        form_stream[Name("/Resources")] = Dictionary(Font=Dictionary(F1=font))

        # Page uses the Form XObject
        page_content = b"/Form1 Do"
        page_stream = pdf.make_stream(page_content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form1=form_stream),
                ),
                Contents=page_stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

    def test_annotation_ap_stream(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # Create AP stream
        ap_content = b"BT /F1 12 Tf <E0> Tj ET"
        ap_stream = pdf.make_stream(ap_content)
        ap_stream[Name("/Subtype")] = Name("/Form")
        ap_stream[Name("/BBox")] = Array([0, 0, 100, 20])
        ap_stream[Name("/Resources")] = Dictionary(Font=Dictionary(F1=font))

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 20]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page_stream = pdf.make_stream(b"")
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=page_stream,
                Annots=Array([annot]),
            )
        )
        pdf.pages.append(page)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

    def test_tiling_pattern_stream(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # Create Tiling Pattern stream
        pat_content = b"BT /F1 12 Tf <E0> Tj ET"
        pattern = pdf.make_stream(pat_content)
        pattern[Name("/PatternType")] = 1
        pattern[Name("/PaintType")] = 1
        pattern[Name("/TilingType")] = 1
        pattern[Name("/BBox")] = Array([0, 0, 100, 100])
        pattern[Name("/XStep")] = 100
        pattern[Name("/YStep")] = 100
        pattern[Name("/Resources")] = Dictionary(Font=Dictionary(F1=font))

        page_content = b"/Pattern cs /P1 scn"
        page_stream = pdf.make_stream(page_content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P1=pattern),
                ),
                Contents=page_stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

    def test_type3_charproc_stream(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        inner_font = _make_simple_font_with_tounicode(pdf, mapping)

        # Type3 font with a CharProc that uses the inner font
        charproc_content = b"BT /F1 12 Tf <E0> Tj ET"
        charproc_stream = pdf.make_stream(charproc_content)

        type3_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            FirstChar=65,
            LastChar=65,
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name("/A")]),
            ),
            CharProcs=Dictionary(A=charproc_stream),
            Resources=Dictionary(Font=Dictionary(F1=inner_font)),
        )

        page_content = b"BT /T3 12 Tf (A) Tj ET"
        page_stream = pdf.make_stream(page_content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(T3=type3_font),
                ),
                Contents=page_stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1

    def test_cycle_detection(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)

        # Create a Form XObject
        form_content = b"BT /F1 12 Tf <E0> Tj ET"
        form_stream = pdf.make_stream(form_content)
        form_stream[Name("/Subtype")] = Name("/Form")
        form_stream[Name("/BBox")] = Array([0, 0, 100, 100])
        form_stream[Name("/Resources")] = Dictionary(
            Font=Dictionary(F1=font),
        )

        # Page references the form, and form is also in second page
        page_content = b"/Form1 Do"
        page_stream = pdf.make_stream(page_content)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form1=form_stream),
                ),
                Contents=page_stream,
            )
        )
        pdf.pages.append(page)

        # Second page referencing same form
        page2_stream = pdf.make_stream(page_content)
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form1=form_stream),
                ),
                Contents=page2_stream,
            )
        )
        pdf.pages.append(page2)

        # Should process the form only once
        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_pdf_no_pages(self):
        pdf = new_pdf()
        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0
        assert result["pua_actualtext_warnings"] == 0

    def test_page_without_fonts(self):
        pdf = new_pdf()
        page_stream = pdf.make_stream(b"q 1 0 0 1 0 0 cm Q")
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=page_stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_font_without_tounicode(self):
        pdf = new_pdf()
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        content = b"BT /F1 12 Tf (Hello) Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_no_text_operators(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        # Content stream with only graphics operations
        content = b"q 1 0 0 1 100 100 cm Q"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_empty_content_stream(self):
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b""
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_text_without_tf(self):
        """Text operator without preceding Tf should not crash."""
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        # Tj without Tf — no font context
        content = b"BT <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_pua_actualtext(pdf)
        assert result["pua_actualtext_added"] == 0

    def test_page_with_array_contents(self):
        """Page with Contents as Array of streams."""
        pdf = new_pdf()
        mapping = {0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        stream1 = pdf.make_stream(b"BT /F1 12 Tf")
        stream2 = pdf.make_stream(b"<E0> Tj ET")
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=Array([stream1, stream2]),
            )
        )
        pdf.pages.append(page)

        # Each stream is processed independently; stream2 has Tj but
        # no Tf, so the font context carries from the parser perspective
        # only within a single stream. This should not crash.
        result = sanitize_pua_actualtext(pdf)
        # May or may not wrap depending on font tracking across streams
        assert isinstance(result["pua_actualtext_added"], int)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with sanitize_for_pdfa()."""

    def test_runs_at_2u(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_for_pdfa(pdf, level="2u")
        assert result["pua_actualtext_added"] >= 1

    def test_runs_at_3u(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_for_pdfa(pdf, level="3u")
        assert result["pua_actualtext_added"] >= 1

    def test_does_not_run_at_2b(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["pua_actualtext_added"] == 0

    def test_does_not_run_at_3b(self):
        pdf = new_pdf()
        mapping = {0x41: 0x0041, 0xE0: 0xE000}
        font = _make_simple_font_with_tounicode(pdf, mapping)
        content = b"BT /F1 12 Tf <E0> Tj ET"
        _make_page_with_font_and_content(pdf, font, content)

        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["pua_actualtext_added"] == 0

    def test_result_keys_present(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)

        result = sanitize_for_pdfa(pdf, level="2u")
        assert "pua_actualtext_added" in result
        assert "pua_actualtext_warnings" in result
