# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/font_structure.py (font structure sanitizer)."""

from io import BytesIO

import pikepdf
from conftest import new_pdf
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.font_structure import sanitize_font_structure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tt_font_bytes(
    glyph_widths: dict[str, int] | None = None,
    units_per_em: int = 1000,
) -> bytes:
    """Build a minimal TrueType font and return its bytes."""
    if glyph_widths is None:
        glyph_widths = {".notdef": 500, "A": 600}
    names = list(glyph_widths.keys())
    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(names)
    cmap: dict[int, str] = {}
    for n in names:
        if n == ".notdef":
            continue
        if len(n) == 1 and n.isalpha():
            cmap[ord(n)] = n
    fb.setupCharacterMap(cmap)
    empty = Glyph()
    fb.setupGlyf({n: empty for n in names})
    fb.setupHorizontalMetrics({n: (w, 0) for n, w in glyph_widths.items()})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=units_per_em)
    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


def _add_font_to_page(pdf: Pdf, font_dict: Dictionary) -> None:
    """Add *font_dict* as an indirect object to the first page's /Resources."""
    font_ref = pdf.make_indirect(font_dict)
    page = pdf.pages[0].obj
    if page.get("/Resources") is None:
        page[Name.Resources] = Dictionary()
    res = page[Name.Resources]
    if res.get("/Font") is None:
        res[Name.Font] = Dictionary()
    res[Name.Font][Name.F1] = font_ref


def _new_page_pdf() -> Pdf:
    """Create a tracked PDF with one empty page."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)
    return pdf


# ---------------------------------------------------------------------------
# Rule 6.2.11.2-1 — /Type
# ---------------------------------------------------------------------------


def test_add_type_when_missing():
    pdf = _new_page_pdf()
    font = Dictionary(Subtype=Name.TrueType, BaseFont=Name.ArialMT)
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_type_added"] == 1
    # Verify the fix on the actual object in the page
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font.get("/Type") == Name.Font


def test_type_not_overwritten():
    pdf = _new_page_pdf()
    font = Dictionary(Type=Name.Font, Subtype=Name.TrueType, BaseFont=Name.ArialMT)
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_type_added"] == 0


# ---------------------------------------------------------------------------
# Rule 6.2.11.2-2 — /Subtype
# ---------------------------------------------------------------------------


def test_infer_subtype_truetype_from_fontfile2():
    pdf = _new_page_pdf()
    font_data = _make_tt_font_bytes()
    ff2_stream = pdf.make_stream(font_data)
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.TestFont,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
        FontFile2=ff2_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        BaseFont=Name.TestFont,
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.TrueType


def test_infer_subtype_type1_from_fontfile():
    pdf = _new_page_pdf()
    # Use a tiny placeholder for a FontFile stream (Type1 PFB/PFA)
    ff_stream = pdf.make_stream(b"%!PS-AdobeFont-1.0 stub")
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.StubFont,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
        FontFile=ff_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        BaseFont=Name.StubFont,
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.Type1


def test_infer_subtype_type1c_fontfile3():
    pdf = _new_page_pdf()
    ff3_stream = pdf.make_stream(b"\x01\x00\x00\x00")
    ff3_stream[Name.Subtype] = Name.Type1C
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.CFFFont,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
        FontFile3=ff3_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        BaseFont=Name.CFFFont,
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.Type1


def test_infer_subtype_type3_from_charprocs():
    pdf = _new_page_pdf()
    char_procs = Dictionary(
        A=pdf.make_stream(b"0 0 d0"),
    )
    font = Dictionary(
        Type=Name.Font,
        CharProcs=char_procs,
        FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
        FontBBox=Array([0, 0, 1000, 1000]),
        Encoding=Name.StandardEncoding,
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.Type3


def test_infer_subtype_type0_from_descendantfonts():
    pdf = _new_page_pdf()
    cid_font = Dictionary(
        Type=Name.Font,
        Subtype=Name.CIDFontType2,
        BaseFont=Name.ArialMT,
        CIDSystemInfo=Dictionary(
            Registry=pikepdf.String("Adobe"),
            Ordering=pikepdf.String("Identity"),
            Supplement=0,
        ),
    )
    font = Dictionary(
        Type=Name.Font,
        BaseFont=Name.ArialMT,
        DescendantFonts=Array([pdf.make_indirect(cid_font)]),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.Type0


def test_infer_subtype_cidfont_from_cidsysteminfo():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        BaseFont=Name.ArialMT,
        CIDSystemInfo=Dictionary(
            Registry=pikepdf.String("Adobe"),
            Ordering=pikepdf.String("Identity"),
            Supplement=0,
        ),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.CIDFontType0


def test_subtype_not_changed_when_valid():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        FirstChar=32,
        LastChar=33,
        Widths=Array([250, 278]),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_subtype_fixed"] == 0
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/Subtype"] == Name.TrueType


# ---------------------------------------------------------------------------
# Rule 6.2.11.2-3 — /BaseFont
# ---------------------------------------------------------------------------


def test_add_basefont_from_descriptor():
    pdf = _new_page_pdf()
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.ArialMT,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
    )
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        FirstChar=32,
        LastChar=33,
        Widths=Array([250, 278]),
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_basefont_added"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font["/BaseFont"] == Name.ArialMT


def test_basefont_not_added_to_type3():
    pdf = _new_page_pdf()
    char_procs = Dictionary(A=pdf.make_stream(b"0 0 d0"))
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type3,
        CharProcs=char_procs,
        FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
        FontBBox=Array([0, 0, 1000, 1000]),
        Encoding=Name.StandardEncoding,
        FirstChar=65,
        LastChar=65,
        Widths=Array([600]),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_basefont_added"] == 0
    page_font = pdf.pages[0].Resources.Font.F1
    assert page_font.get("/BaseFont") is None


# ---------------------------------------------------------------------------
# Rules 6.2.11.2-4, -5 — /FirstChar and /LastChar
# ---------------------------------------------------------------------------


def test_add_firstchar_from_widths_and_lastchar():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        LastChar=35,
        Widths=Array([250, 278, 300, 320]),  # 4 entries → FirstChar = 32
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_firstchar_added"] == 1
    assert result["font_lastchar_added"] == 0
    page_font = pdf.pages[0].Resources.Font.F1
    assert int(page_font["/FirstChar"]) == 32


def test_add_lastchar_from_widths_and_firstchar():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        FirstChar=32,
        Widths=Array([250, 278, 300, 320]),  # 4 entries → LastChar = 35
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_lastchar_added"] == 1
    assert result["font_firstchar_added"] == 0
    page_font = pdf.pages[0].Resources.Font.F1
    assert int(page_font["/LastChar"]) == 35


def test_add_both_when_both_missing():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        Widths=Array([250, 278, 300]),  # 3 entries → FirstChar=0, LastChar=2
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_firstchar_added"] == 1
    assert result["font_lastchar_added"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    assert int(page_font["/FirstChar"]) == 0
    assert int(page_font["/LastChar"]) == 2


# ---------------------------------------------------------------------------
# Rule 6.2.11.2-6 — /Widths array size
# ---------------------------------------------------------------------------


def test_widths_padded_when_too_short():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        FirstChar=32,
        LastChar=36,  # expects 5 entries
        Widths=Array([250, 278, 300]),  # only 3
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_widths_size_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    widths = list(page_font["/Widths"])
    assert len(widths) == 5
    assert int(widths[3]) == 0
    assert int(widths[4]) == 0


def test_widths_truncated_when_too_long():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        FirstChar=32,
        LastChar=34,  # expects 3 entries
        Widths=Array([250, 278, 300, 320, 340]),  # 5 — too many
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_widths_size_fixed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    widths = list(page_font["/Widths"])
    assert len(widths) == 3


def test_standard_font_widths_not_fixed():
    pdf = _new_page_pdf()
    # Deliberately wrong Widths size for a Standard-14 font
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type1,
        BaseFont=Name.Helvetica,
        FirstChar=32,
        LastChar=36,  # expects 5 entries
        Widths=Array([250]),  # wrong size — but should be skipped
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_widths_size_fixed"] == 0


# ---------------------------------------------------------------------------
# Rule 6.2.11.2-7 — font stream /Subtype
# ---------------------------------------------------------------------------


def test_remove_invalid_fontfile3_subtype():
    pdf = _new_page_pdf()
    ff3_stream = pdf.make_stream(b"\x01\x00\x00\x00")
    ff3_stream[Name.Subtype] = Name("/BadSubtype")
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.TestFont,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
        FontFile3=ff3_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type1,
        BaseFont=Name.TestFont,
        FirstChar=32,
        LastChar=33,
        Widths=Array([250, 278]),
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_stream_subtype_removed"] == 1
    # Subtype must be gone from the stream
    page_font = pdf.pages[0].Resources.Font.F1
    fd_obj = page_font["/FontDescriptor"]
    ff3_obj = fd_obj["/FontFile3"]
    assert ff3_obj.get("/Subtype") is None


def test_valid_fontfile3_subtypes_kept():
    for valid_st in ("/Type1C", "/CIDFontType0C", "/OpenType"):
        pdf = _new_page_pdf()
        ff3_stream = pdf.make_stream(b"\x01\x00\x00\x00")
        ff3_stream[Name.Subtype] = Name(valid_st)
        fd = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name.TestFont,
            Flags=32,
            FontBBox=Array([0, 0, 1000, 1000]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile3=ff3_stream,
        )
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.TestFont,
            FirstChar=32,
            LastChar=33,
            Widths=Array([250, 278]),
            FontDescriptor=pdf.make_indirect(fd),
        )
        _add_font_to_page(pdf, font)

        result = sanitize_font_structure(pdf)

        assert result["font_stream_subtype_removed"] == 0, (
            f"Subtype {valid_st!r} should not be removed"
        )
        page_font = pdf.pages[0].Resources.Font.F1
        fd_obj = page_font["/FontDescriptor"]
        ff3_obj = fd_obj["/FontFile3"]
        assert ff3_obj["/Subtype"] == Name(valid_st)


def test_remove_subtype_from_fontfile2():
    pdf = _new_page_pdf()
    font_data = _make_tt_font_bytes()
    ff2_stream = pdf.make_stream(font_data)
    # FontFile2 should never have /Subtype — add a bogus one
    ff2_stream[Name.Subtype] = Name.TrueType
    fd = Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name.TestFont,
        Flags=32,
        FontBBox=Array([0, 0, 1000, 1000]),
        ItalicAngle=0,
        Ascent=800,
        Descent=-200,
        CapHeight=700,
        StemV=80,
        FontFile2=ff2_stream,
    )
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.TestFont,
        FirstChar=32,
        LastChar=33,
        Widths=Array([250, 600]),
        FontDescriptor=pdf.make_indirect(fd),
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert result["font_stream_subtype_removed"] == 1
    page_font = pdf.pages[0].Resources.Font.F1
    fd_obj = page_font["/FontDescriptor"]
    ff2_obj = fd_obj["/FontFile2"]
    assert ff2_obj.get("/Subtype") is None


# ---------------------------------------------------------------------------
# No-op on valid font
# ---------------------------------------------------------------------------


def test_no_changes_on_valid_font():
    pdf = _new_page_pdf()
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name.ArialMT,
        FirstChar=32,
        LastChar=34,
        Widths=Array([250, 278, 300]),  # 3 entries = LastChar - FirstChar + 1
    )
    _add_font_to_page(pdf, font)

    result = sanitize_font_structure(pdf)

    assert all(v == 0 for v in result.values()), (
        f"Expected no changes but got: {result}"
    )
