# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for glyph usage collection from content streams."""

from itertools import islice

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts.glyph_usage import (
    FontUsageCache,
    _extract_char_codes,
    _is_cidfont,
    _iter_content_streams_with_resources,
    collect_font_usage,
)


def _make_page_with_content(pdf, content_bytes, font_dict, resources=None):
    """Helper: creates a page with content stream and font resources."""
    if resources is None:
        resources = Dictionary(Font=font_dict)
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=resources,
        Contents=pdf.make_stream(content_bytes),
    )
    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)
    return page


@pytest.mark.parametrize("cid", [False, True])
def test_extgstate_font_selection_and_graphics_state_restore(cid):
    pdf = new_pdf()
    first = pdf.make_indirect(Dictionary(Subtype=Name.Type1, BaseFont=Name.Helvetica))
    second = pdf.make_indirect(
        Dictionary(
            Subtype=Name.Type0 if cid else Name.Type1,
            BaseFont=Name.Courier,
            Encoding=Name("/Identity-H") if cid else Name.WinAnsiEncoding,
        )
    )
    text = b"<0042>" if cid else b"(B)"
    _make_page_with_content(
        pdf,
        b"BT /F1 12 Tf (A) Tj q /GS gs " + text + b" Tj Q (C) Tj ET",
        Dictionary(),
        resources=Dictionary(
            Font=Dictionary(F1=first),
            ExtGState=Dictionary(GS=Dictionary(Font=Array([second, 12]))),
        ),
    )
    usage = collect_font_usage(pdf)
    assert usage[first.objgen] == {65, 67}
    assert usage[second.objgen] == ({b"\x00B"} if cid else {66})


def test_extgstate_type3_charproc_fonts_are_discovered_and_counted():
    from pdftopdfa.fonts.traversal import iter_all_page_fonts
    from pdftopdfa.utils import iter_type3_fonts

    pdf = new_pdf()
    child = pdf.make_indirect(
        Dictionary(
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
        )
    )
    type3 = pdf.make_indirect(
        Dictionary(
            Subtype=Name.Type3,
            CharProcs=Dictionary(A=pdf.make_stream(b"500 0 d0 BT /F1 12 Tf (B) Tj ET")),
            Resources=Dictionary(Font=Dictionary(F1=child)),
        )
    )
    page = _make_page_with_content(
        pdf,
        b"BT /GS gs (A) Tj ET",
        Dictionary(),
        resources=Dictionary(
            ExtGState=Dictionary(GS=Dictionary(Font=Array([type3, 12])))
        ),
    )
    assert {font.objgen for _name, font in iter_all_page_fonts(page)} == {
        type3.objgen,
        child.objgen,
    }
    assert [font.objgen for _name, font in iter_type3_fonts(page.Resources, set())] == [
        type3.objgen
    ]
    usage = collect_font_usage(pdf)
    assert usage[child.objgen] == {66}
    assert usage[type3.objgen] == {65}


def test_resource_graph_traversal_handles_1200_nested_forms_and_type3() -> None:
    """Deep Form graphs are complete instead of stopping at recursion depth."""
    pdf = new_pdf()
    charproc = pdf.make_stream(b"500 0 d0")
    type3 = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1, 1]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(A=charproc),
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name.A]),
            ),
            FirstChar=65,
            LastChar=65,
            Widths=Array([500]),
            Resources=Dictionary(),
        )
    )
    leaf = pdf.make_stream(b"")
    leaf[Name.Type] = Name.XObject
    leaf[Name.Subtype] = Name.Form
    leaf[Name.BBox] = Array([0, 0, 1, 1])
    leaf[Name.Resources] = pdf.make_indirect(Dictionary(Font=Dictionary(T3=type3)))

    root = leaf
    for _ in range(1200):
        form = pdf.make_stream(b"/Next Do")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 1, 1])
        form[Name.Resources] = pdf.make_indirect(
            Dictionary(XObject=Dictionary(Next=root))
        )
        root = form

    page = pdf.add_blank_page(page_size=(10, 10))
    page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=root))
    page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")

    owners = list(_iter_content_streams_with_resources(page))

    assert len(owners) == 1203
    assert any(
        isinstance(owner, pikepdf.Stream) and owner.objgen == charproc.objgen
        for owner, _resources in owners
    )


def test_direct_resource_wrapper_cycle_has_bounded_traversal() -> None:
    """Equivalent direct-resource wrappers do not multiply traversal tasks."""
    pdf = new_pdf()
    charproc = pdf.make_stream(b"500 0 d0")
    type3 = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1, 1]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(A=charproc),
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name.A]),
            ),
            FirstChar=65,
            LastChar=65,
            Widths=Array([500]),
        )
    )
    resources = Dictionary(Font=Dictionary(T3=type3))
    type3[Name.Resources] = resources

    page = pdf.add_blank_page(page_size=(10, 10))
    page.obj[Name.Resources] = resources
    page.obj[Name.Contents] = pdf.make_stream(b"BT /T3 12 Tf (A) Tj ET")

    owners = list(islice(_iter_content_streams_with_resources(page), 8))

    assert len(owners) == 2
    assert owners[0][0].objgen == page.obj.objgen
    assert owners[1][0].objgen == charproc.objgen


class TestExtractCharCodes:
    """Tests for _extract_char_codes."""

    def test_simple_font_single_byte(self):
        """Simple fonts extract one code per byte."""
        new_pdf()
        s = pikepdf.String(b"ABC")
        codes = _extract_char_codes(s, is_cid=False)
        assert codes == {65, 66, 67}

    def test_cidfont_two_byte(self):
        """CIDFonts extract two-byte big-endian codes."""
        new_pdf()
        # 0x0041 = 65, 0x00E4 = 228
        s = pikepdf.String(b"\x00\x41\x00\xe4")
        codes = _extract_char_codes(s, is_cid=True)
        assert codes == {0x0041, 0x00E4}

    def test_cidfont_odd_bytes_ignores_trailing(self):
        """CIDFont with odd byte count ignores trailing byte."""
        new_pdf()
        s = pikepdf.String(b"\x00\x41\xff")
        codes = _extract_char_codes(s, is_cid=True)
        assert codes == {0x0041}

    def test_cidfont_variable_width_codes(self):
        """CIDFont codes can use every width allowed by a CMap."""
        new_pdf()
        ranges = (
            (b"\x00", b"\x7f"),
            (b"\x81\x00", b"\x81\xff"),
            (b"\x82\x00\x00", b"\x82\xff\xff"),
            (b"\x83\x00\x00\x00", b"\x83\xff\xff\xff"),
        )
        value = pikepdf.String(b"\x41\x81\x02\x82\x00\x03\x83\x00\x00\x04")

        codes = _extract_char_codes(value, is_cid=True, code_space_ranges=ranges)

        assert codes == {0x41, 0x8102, 0x820003, 0x83000004}

    def test_empty_string(self):
        """Empty string returns empty set."""
        new_pdf()
        s = pikepdf.String(b"")
        assert _extract_char_codes(s, is_cid=False) == set()
        assert _extract_char_codes(s, is_cid=True) == set()


class TestIsCIDFont:
    """Tests for _is_cidfont."""

    def test_type0_is_cidfont(self):
        """Type0 font is identified as CIDFont."""
        font = Dictionary(Subtype=Name.Type0)
        assert _is_cidfont(font) is True

    def test_truetype_is_not_cidfont(self):
        """TrueType font is not a CIDFont."""
        font = Dictionary(Subtype=Name.TrueType)
        assert _is_cidfont(font) is False

    def test_type1_is_not_cidfont(self):
        """Type1 font is not a CIDFont."""
        font = Dictionary(Subtype=Name.Type1)
        assert _is_cidfont(font) is False

    def test_no_subtype(self):
        """Font without Subtype is not a CIDFont."""
        font = Dictionary()
        assert _is_cidfont(font) is False


class TestCollectFontUsage:
    """Tests for collect_font_usage."""

    def test_tj_operator(self):
        """Collects codes from Tj operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (Hello) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        # Font should have usage for H, e, l, o
        assert len(usage) == 1
        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("H"), ord("e"), ord("l"), ord("o")}
        assert usage[objgen] == expected

    def test_tj_array_operator(self):
        """Collects codes from TJ (array) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # TJ array: [(AB) -100 (CD)]
        content = b"BT /F1 12 Tf [(AB) -100 (CD)] TJ ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("A"), ord("B"), ord("C"), ord("D")}
        assert usage[objgen] == expected

    def test_single_quote_operator(self):
        """Collects codes from ' (single-quote) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        content = b"BT /F1 12 Tf (XY) ' ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {ord("X"), ord("Y")}

    def test_double_quote_operator(self):
        """Collects codes from " (double-quote) operator."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # " operator: aw ac string
        content = b'BT /F1 12 Tf 1 2 (ZW) " ET'
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {ord("Z"), ord("W")}

    def test_cidfont_two_byte_decoding(self):
        """CIDFont codes are decoded as 2-byte big-endian."""
        pdf = new_pdf()

        # Build a Type0 CIDFont structure
        desc_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/TestCJK"),
            )
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCJK"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([desc_font]),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # CID content: two 2-byte codes: 0x0041 and 0x4E2D
        content = b"BT /F1 12 Tf <00414E2D> Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        assert usage[objgen] == {b"\x00A", b"N-"}

    def test_cidfont_uses_declared_one_byte_codespace(self):
        """Custom Type0 CMaps determine character-code boundaries."""
        pdf = new_pdf()
        encoding = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"2 begincidchar\n<01> 1\n<02> 2\nendcidchar\nendcmap"
        )
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/CustomCID"),
                Encoding=encoding,
                DescendantFonts=Array([]),
            )
        )
        _make_page_with_content(
            pdf,
            b"BT /F1 12 Tf <0102> Tj ET",
            Dictionary(F1=font),
        )

        usage = collect_font_usage(pdf)

        assert usage[font.objgen] == {b"\x01", b"\x02"}

    def test_cidfont_uses_tounicode_codespace_for_named_encoding(self):
        """A ToUnicode CMap supplies widths for an unresolved named CMap."""
        pdf = new_pdf()
        tounicode = pdf.make_stream(
            b"begincmap\n"
            b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            b"2 beginbfchar\n<01> <0041>\n<02> <0042>\nendbfchar\nendcmap"
        )
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/CustomCID"),
                Encoding=Name("/CustomEncoding"),
                DescendantFonts=Array([]),
                ToUnicode=tounicode,
            )
        )
        _make_page_with_content(
            pdf,
            b"BT /F1 12 Tf <0102> Tj ET",
            Dictionary(F1=font),
        )

        usage = collect_font_usage(pdf)

        assert usage[font.objgen] == {b"\x01", b"\x02"}

    def test_multi_page_aggregation(self):
        """Usage from multiple pages is aggregated for same font."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/SharedFont"),
        )
        font_obj = pdf.make_indirect(font)

        # Page 1 uses "AB"
        font_dict1 = Dictionary(F1=font_obj)
        _make_page_with_content(pdf, b"BT /F1 12 Tf (AB) Tj ET", font_dict1)

        # Page 2 uses "CD"
        font_dict2 = Dictionary(F1=font_obj)
        _make_page_with_content(pdf, b"BT /F1 12 Tf (CD) Tj ET", font_dict2)

        usage = collect_font_usage(pdf)

        objgen = font_obj.objgen
        assert objgen in usage
        expected = {ord("A"), ord("B"), ord("C"), ord("D")}
        assert usage[objgen] == expected

    def test_multiple_fonts(self):
        """Tracks usage separately for different fonts."""
        pdf = new_pdf()

        font1 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font1"),
        )
        font1_obj = pdf.make_indirect(font1)

        font2 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font2"),
        )
        font2_obj = pdf.make_indirect(font2)

        font_dict = Dictionary(F1=font1_obj, F2=font2_obj)
        content = b"BT /F1 12 Tf (AB) Tj /F2 10 Tf (XY) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        assert usage[font1_obj.objgen] == {ord("A"), ord("B")}
        assert usage[font2_obj.objgen] == {ord("X"), ord("Y")}

    def test_q_restore_restores_previous_font(self):
        """Restoring graphics state also restores the active font."""
        pdf = new_pdf()

        font1 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font1"),
        )
        font1_obj = pdf.make_indirect(font1)

        font2 = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/Font2"),
        )
        font2_obj = pdf.make_indirect(font2)

        font_dict = Dictionary(F1=font1_obj, F2=font2_obj)
        content = b"BT /F1 12 Tf ET q BT /F2 10 Tf (B) Tj ET Q BT (A) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        assert usage[font1_obj.objgen] == {ord("A")}
        assert usage[font2_obj.objgen] == {ord("B")}

    def test_form_xobject(self):
        """Collects usage from Form XObject content streams."""
        pdf = new_pdf()

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/NestedFont"),
        )
        nested_font_obj = pdf.make_indirect(nested_font)

        form_xobj = pdf.make_stream(b"BT /F2 10 Tf (XO) Tj ET")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 200, 200])
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F2=nested_font_obj),
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/PageFont"),
        )
        page_font_obj = pdf.make_indirect(page_font)

        resources = Dictionary(
            Font=Dictionary(F1=page_font_obj),
            XObject=Dictionary(Form1=pdf.make_indirect(form_xobj)),
        )
        content = b"BT /F1 12 Tf (PG) Tj ET /Form1 Do"
        _make_page_with_content(pdf, content, Dictionary(), resources)

        usage = collect_font_usage(pdf)

        assert page_font_obj.objgen in usage
        assert usage[page_font_obj.objgen] == {ord("P"), ord("G")}
        assert nested_font_obj.objgen in usage
        assert usage[nested_font_obj.objgen] == {ord("X"), ord("O")}

    def test_inherited_page_resources(self):
        """Page-tree resources are used when the page has no local dictionary."""
        pdf = new_pdf()
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/InheritedFont"),
            )
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Contents=pdf.make_stream(b"BT /F1 12 Tf (A) Tj ET"),
            )
        )
        pdf.pages.append(page)
        pdf.Root.Pages[Name.Resources] = Dictionary(Font=Dictionary(F1=font))

        usage = collect_font_usage(pdf)

        assert usage[font.objgen] == {ord("A")}

    def test_resource_less_form_is_processed_in_each_pattern_context(self):
        """A shared Form inherits each calling Pattern's effective resources."""
        pdf = new_pdf()
        font_a = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/FontA"),
            )
        )
        font_b = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/FontB"),
            )
        )
        shared_form = pdf.make_stream(b"BT /F1 12 Tf (A) Tj ET")
        shared_form[Name.Type] = Name.XObject
        shared_form[Name.Subtype] = Name.Form
        shared_form[Name.BBox] = Array([0, 0, 10, 10])
        shared_form = pdf.make_indirect(shared_form)

        def pattern(font):
            stream = pdf.make_stream(b"/Fm Do")
            stream[Name.Type] = Name.Pattern
            stream[Name.PatternType] = 1
            stream[Name.PaintType] = 1
            stream[Name.TilingType] = 1
            stream[Name.BBox] = Array([0, 0, 10, 10])
            stream[Name.XStep] = 10
            stream[Name.YStep] = 10
            stream[Name.Resources] = Dictionary(
                Font=Dictionary(F1=font),
                XObject=Dictionary(Fm=shared_form),
            )
            return stream

        _make_page_with_content(
            pdf,
            b"/Pattern cs /P1 scn /P2 scn",
            Dictionary(),
            Dictionary(
                Pattern=Dictionary(
                    P1=pattern(font_a),
                    P2=pattern(font_b),
                )
            ),
        )

        usage = collect_font_usage(pdf)

        assert usage[font_a.objgen] == {ord("A")}
        assert usage[font_b.objgen] == {ord("A")}

    def test_soft_mask_group_inherits_resources_without_recursing_forever(self):
        """A resource-less SMask group uses its effective parent resources."""
        pdf = new_pdf()
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/SoftMaskFont"),
            )
        )
        group = pdf.make_stream(b"BT /F1 12 Tf (S) Tj ET")
        group[Name.Type] = Name.XObject
        group[Name.Subtype] = Name.Form
        group[Name.BBox] = Array([0, 0, 10, 10])
        resources = Dictionary(
            Font=Dictionary(F1=font),
            ExtGState=Dictionary(
                GS=Dictionary(
                    SMask=Dictionary(
                        S=Name("/Luminosity"),
                        G=group,
                    )
                )
            ),
        )
        _make_page_with_content(
            pdf,
            b"/GS gs",
            Dictionary(),
            resources,
        )

        usage = collect_font_usage(pdf)

        assert usage[font.objgen] == {ord("S")}

    def test_resource_less_soft_mask_group_uses_each_effective_context(self):
        """The same SMask group is processed under both Pattern resources."""
        pdf = new_pdf()
        font_a = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/SoftMaskFontA"),
            )
        )
        font_b = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/SoftMaskFontB"),
            )
        )
        shared_group = pdf.make_stream(b"BT /F1 12 Tf (M) Tj ET")
        shared_group[Name.Type] = Name.XObject
        shared_group[Name.Subtype] = Name.Form
        shared_group[Name.BBox] = Array([0, 0, 10, 10])

        def pattern(font):
            stream = pdf.make_stream(b"/GS gs")
            stream[Name.Type] = Name.Pattern
            stream[Name.PatternType] = 1
            stream[Name.PaintType] = 1
            stream[Name.TilingType] = 1
            stream[Name.BBox] = Array([0, 0, 10, 10])
            stream[Name.XStep] = 10
            stream[Name.YStep] = 10
            stream[Name.Resources] = Dictionary(
                Font=Dictionary(F1=font),
                ExtGState=Dictionary(
                    GS=Dictionary(
                        SMask=Dictionary(
                            S=Name("/Luminosity"),
                            G=shared_group,
                        )
                    )
                ),
            )
            return stream

        _make_page_with_content(
            pdf,
            b"/Pattern cs /P1 scn /P2 scn",
            Dictionary(),
            Dictionary(
                Pattern=Dictionary(
                    P1=pattern(font_a),
                    P2=pattern(font_b),
                )
            ),
        )

        usage = collect_font_usage(pdf)

        assert usage[font_a.objgen] == {ord("M")}
        assert usage[font_b.objgen] == {ord("M")}

    def test_no_content_stream(self):
        """Handles pages without content streams gracefully."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        usage = collect_font_usage(pdf)
        assert usage == {}

    def test_no_text_operators(self):
        """Pages without text operators produce no usage."""
        pdf = new_pdf()

        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
        )
        font_obj = pdf.make_indirect(font)
        font_dict = Dictionary(F1=font_obj)

        # Content with graphics but no text
        content = b"100 200 m 300 400 l S"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)
        assert usage == {}

    def test_type3_charproc_with_own_resources(self):
        """Collects usage from Type3 CharProcs using the font's own resources."""
        pdf = new_pdf()

        inner_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/InnerFont"),
        )
        inner_font_obj = pdf.make_indirect(inner_font)

        charproc = pdf.make_stream(b"BT /F2 10 Tf (AB) Tj ET")
        type3 = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(a=pdf.make_indirect(charproc)),
            Resources=Dictionary(Font=Dictionary(F2=inner_font_obj)),
        )
        type3_obj = pdf.make_indirect(type3)

        font_dict = Dictionary(F1=type3_obj)
        content = b"BT /F1 12 Tf (a) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)

        assert inner_font_obj.objgen in usage
        assert usage[inner_font_obj.objgen] == {ord("A"), ord("B")}

    def test_type3_charproc_inherits_enclosing_resources(self):
        """Type3 fonts without /Resources fall back to the enclosing resources."""
        pdf = new_pdf()

        inner_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/InnerFont"),
        )
        inner_font_obj = pdf.make_indirect(inner_font)

        charproc = pdf.make_stream(b"BT /F2 10 Tf (XY) Tj ET")
        type3 = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(a=pdf.make_indirect(charproc)),
        )
        type3_obj = pdf.make_indirect(type3)

        resources = Dictionary(
            Font=Dictionary(F1=type3_obj, F2=inner_font_obj),
        )
        content = b"BT /F1 12 Tf (a) Tj ET"
        _make_page_with_content(pdf, content, Dictionary(), resources)

        usage = collect_font_usage(pdf)

        assert inner_font_obj.objgen in usage
        assert usage[inner_font_obj.objgen] == {ord("X"), ord("Y")}

    def test_direct_font_object_skipped(self):
        """Direct font objects (objgen 0,0) are not tracked."""
        pdf = new_pdf()

        # Direct font (not make_indirect)
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/DirectFont"),
        )
        font_dict = Dictionary(F1=font)

        content = b"BT /F1 12 Tf (AB) Tj ET"
        _make_page_with_content(pdf, content, font_dict)

        usage = collect_font_usage(pdf)
        # Direct objects have objgen (0,0) and are skipped
        assert usage == {}


class TestFontUsageCache:
    """Tests for FontUsageCache."""

    def _make_pdf_with_usage(self):
        pdf = new_pdf()
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/CachedFont"),
            )
        )
        content = b"BT /F1 12 Tf (AB) Tj ET"
        _make_page_with_content(pdf, content, Dictionary(F1=font))
        return pdf, font

    def test_get_returns_collected_usage(self):
        """get() returns the same result as collect_font_usage()."""
        pdf, font = self._make_pdf_with_usage()
        cache = FontUsageCache(pdf)

        assert cache.get() == collect_font_usage(pdf)
        assert cache.get()[font.objgen] == {65, 66}

    def test_get_is_cached_until_invalidated(self):
        """Repeated get() calls reuse the collection; invalidate() drops it."""
        pdf, font = self._make_pdf_with_usage()
        cache = FontUsageCache(pdf)

        first = cache.get()
        assert cache.get() is first

        # Add a second page using the same font with another code.
        content = b"BT /F1 12 Tf (C) Tj ET"
        _make_page_with_content(pdf, content, Dictionary(F1=font))

        # Still cached: the new code is not visible yet.
        assert cache.get() is first

        cache.invalidate()
        refreshed = cache.get()
        assert refreshed is not first
        assert refreshed[font.objgen] == {65, 66, 67}
