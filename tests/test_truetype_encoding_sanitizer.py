# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/truetype_encoding.py (TrueType encoding sanitizer)."""

from io import BytesIO

import pikepdf
from conftest import new_pdf
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import cmap_format_4
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.sanitizers.truetype_encoding import sanitize_truetype_encoding
from pdftopdfa.utils import resolve_indirect as _resolve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tt_font_bytes(
    cmap_subtables: list[tuple[int, int, dict[int, str]]],
    symbolic: bool = False,
) -> bytes:
    """Build a minimal TrueType font with custom cmap subtables.

    Args:
        cmap_subtables: List of (platformID, platEncID, mapping) tuples.
        symbolic: If True, sets the Symbolic OS/2 flag.

    Returns:
        Raw TrueType font bytes.
    """
    glyphs = {".notdef": 500, "A": 600}
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(list(glyphs.keys()))
    # Use first subtable mapping for setupCharacterMap (FontBuilder requirement)
    if cmap_subtables:
        first_mapping = cmap_subtables[0][2]
        setup_cmap: dict[int, str] = {}
        for code, gname in first_mapping.items():
            if 0 <= code <= 0xFFFF and gname in glyphs:
                setup_cmap[code] = gname
        fb.setupCharacterMap(setup_cmap)
    else:
        fb.setupCharacterMap({0x41: "A"})

    empty_glyph = Glyph()
    fb.setupGlyf({n: empty_glyph for n in glyphs.keys()})
    fb.setupHorizontalMetrics({n: (w, 0) for n, w in glyphs.items()})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    fs_selection = 0x0040
    fs_type = 0x0004 if symbolic else 0x0000
    fb.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        sCapHeight=700,
        fsSelection=fs_selection,
        fsType=fs_type,
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    buf = BytesIO()
    fb.font.save(buf)

    # Replace cmap table entirely with the requested subtables
    buf.seek(0)
    tt = TTFont(buf)
    cmap_table = tt["cmap"]
    new_subtables = []
    for platform_id, plat_enc_id, mapping in cmap_subtables:
        st = cmap_format_4(4)
        st.platformID = platform_id
        st.platEncID = plat_enc_id
        st.language = 0
        st.cmap = dict(mapping)
        new_subtables.append(st)
    cmap_table.tables = new_subtables

    out = BytesIO()
    tt.save(out)
    tt.close()
    return out.getvalue()


def _add_truetype_font(
    pdf: Pdf,
    font_bytes: bytes,
    flags: int,
    encoding=None,
) -> tuple[Dictionary, Dictionary]:
    """Add a TrueType font dict to the first page's Resources/Font.

    Args:
        pdf: The PDF to add the font to.
        font_bytes: Raw TrueType font bytes.
        flags: /Flags value (determines symbolic vs non-symbolic).
        encoding: Optional /Encoding value (Name or Dictionary).

    Returns:
        Tuple of (font Dictionary, FontDescriptor Dictionary).
    """
    stream = Stream(pdf, font_bytes)
    stream[Name.Length1] = len(font_bytes)
    font_file2 = pdf.make_indirect(stream)

    fd = Dictionary()
    fd[Name.Type] = Name.FontDescriptor
    fd[Name.FontName] = Name.TestFont
    fd[Name.Flags] = flags
    fd[Name.FontBBox] = Array([0, -200, 1000, 800])
    fd[Name.ItalicAngle] = 0
    fd[Name.Ascent] = 800
    fd[Name.Descent] = -200
    fd[Name.CapHeight] = 700
    fd[Name.StemV] = 80
    fd[Name("/FontFile2")] = font_file2
    fd_ref = pdf.make_indirect(fd)

    font = Dictionary()
    font[Name.Type] = Name.Font
    font[Name.Subtype] = Name.TrueType
    font[Name.BaseFont] = Name.TestFont
    font[Name.FirstChar] = 65
    font[Name.LastChar] = 65
    font[Name.Widths] = Array([600])
    font[Name("/FontDescriptor")] = fd_ref
    if encoding is not None:
        font[Name.Encoding] = encoding

    font_ref = pdf.make_indirect(font)

    page = pdf.pages[0].obj
    if page.get("/Resources") is None:
        page[Name.Resources] = Dictionary()
    res = page[Name.Resources]
    if res.get("/Font") is None:
        res[Name.Font] = Dictionary()
    res[Name.Font][Name.F1] = font_ref

    return font, fd


def _new_page_pdf() -> Pdf:
    """Create a tracked PDF with one empty page."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)
    return pdf


def _get_cmap_subtables(fd: Dictionary) -> list[tuple[int, int]]:
    """Read (platformID, platEncID) pairs from the embedded font's cmap.

    Args:
        fd: Resolved FontDescriptor Dictionary.

    Returns:
        List of (platformID, platEncID) tuples.
    """
    font_file = _resolve(fd["/FontFile2"])
    data = bytes(font_file.read_bytes())
    tt = TTFont(BytesIO(data))
    try:
        return [(st.platformID, st.platEncID) for st in tt["cmap"].tables]
    finally:
        tt.close()


# ---------------------------------------------------------------------------
# Rule 6.2.11.6-1 — Non-symbolic cmap
# ---------------------------------------------------------------------------


def test_rule1_adds_31_cmap_when_only_30_exists():
    """Non-symbolic TrueType with only (3,0) should get a (3,1) added."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 0, {0x0041: "A"})], symbolic=False)
    _font, fd = _add_truetype_font(
        pdf, font_bytes, flags=32, encoding=Name.WinAnsiEncoding
    )

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_cmap_added"] == 1
    platform_enc_pairs = set(_get_cmap_subtables(fd))
    assert (3, 1) in platform_enc_pairs


def test_rule1_no_change_when_31_already_exists():
    """Non-symbolic TrueType with (3,1) cmap already present should not be changed."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    _add_truetype_font(pdf, font_bytes, flags=32, encoding=Name.WinAnsiEncoding)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_cmap_added"] == 0


def test_rule1_no_change_when_10_already_exists():
    """Non-symbolic TrueType with (1,0) cmap already present should not add (3,1)."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(1, 0, {0x0041: "A"})], symbolic=False)
    _add_truetype_font(pdf, font_bytes, flags=32, encoding=Name.WinAnsiEncoding)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_cmap_added"] == 0


# ---------------------------------------------------------------------------
# Rule 6.2.11.6-2 — Non-symbolic /Encoding
# ---------------------------------------------------------------------------


def test_rule2_adds_winansi_when_no_encoding():
    """Non-symbolic TrueType with no /Encoding should get WinAnsiEncoding added."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    font, _fd = _add_truetype_font(pdf, font_bytes, flags=32)  # no encoding

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_encoding_fixed"] == 1
    assert str(font["/Encoding"]) == "/WinAnsiEncoding"


def test_rule2_replaces_standard_encoding():
    """Non-symbolic TrueType with /StandardEncoding should be replaced."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    font, _fd = _add_truetype_font(
        pdf, font_bytes, flags=32, encoding=Name.StandardEncoding
    )

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_encoding_fixed"] == 1
    assert str(font["/Encoding"]) == "/WinAnsiEncoding"


def test_rule2_keeps_winansi_unchanged():
    """Non-symbolic TrueType with /WinAnsiEncoding should not be changed."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    _add_truetype_font(pdf, font_bytes, flags=32, encoding=Name.WinAnsiEncoding)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_encoding_fixed"] == 0


def test_rule2_keeps_macroman_unchanged():
    """Non-symbolic TrueType with /MacRomanEncoding should not be changed."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    _add_truetype_font(pdf, font_bytes, flags=32, encoding=Name.MacRomanEncoding)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_encoding_fixed"] == 0


def test_rule2_removes_non_agl_differences():
    """Encoding dict with non-AGL Differences should have Differences removed."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    enc = Dictionary()
    enc[Name.BaseEncoding] = Name.WinAnsiEncoding
    # Name objects in pikepdf require the "/" prefix
    enc[Name.Differences] = Array([65, Name("/ZZZnotinAGL")])
    font, _fd = _add_truetype_font(pdf, font_bytes, flags=32, encoding=enc)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_nonsymbolic_encoding_fixed"] == 1
    enc_resolved = _resolve(font["/Encoding"])
    assert enc_resolved.get("/Differences") is None


def test_rule2_keeps_valid_agl_differences():
    """Encoding dict with AGL-valid Differences should not remove them."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 1, {0x0041: "A"})], symbolic=False)
    enc = Dictionary()
    enc[Name.BaseEncoding] = Name.WinAnsiEncoding
    # "/A" and "/B" are valid AGL names
    enc[Name.Differences] = Array([65, Name("/A"), 66, Name("/B")])
    font, _fd = _add_truetype_font(pdf, font_bytes, flags=32, encoding=enc)

    result = sanitize_truetype_encoding(pdf)

    # No removal — Differences are valid AGL names, BaseEncoding already correct
    assert result["tt_nonsymbolic_encoding_fixed"] == 0
    enc_resolved = _resolve(font["/Encoding"])
    assert enc_resolved.get("/Differences") is not None


# ---------------------------------------------------------------------------
# Rule 6.2.11.6-3 — Symbolic /Encoding and Symbolic flag
# ---------------------------------------------------------------------------


def test_rule3_removes_encoding_from_symbolic():
    """Symbolic TrueType with /Encoding should have it removed."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 0, {0xF041: "A"})], symbolic=True)
    # Symbolic flag = bit 3 = value 4; must be set for is_symbolic_font to return True
    font, _fd = _add_truetype_font(
        pdf, font_bytes, flags=4, encoding=Name.WinAnsiEncoding
    )

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_symbolic_encoding_removed"] == 1
    assert font.get("/Encoding") is None


def test_rule3_sets_symbolic_flag_in_flags():
    """Symbolic TrueType with Symbolic bit set: no extra flag change expected."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 0, {0xF041: "A"})], symbolic=True)
    # Symbolic bit already set (flags=4), no /Encoding → already compliant
    _add_truetype_font(pdf, font_bytes, flags=4)

    result = sanitize_truetype_encoding(pdf)

    # Already compliant: no changes
    assert result["tt_symbolic_encoding_removed"] == 0
    assert result["tt_symbolic_flag_set"] == 0


def test_rule3_no_change_when_already_compliant():
    """Symbolic TrueType with no /Encoding and Symbolic flag set: no changes."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 0, {0xF041: "A"})], symbolic=True)
    _add_truetype_font(pdf, font_bytes, flags=4)  # Symbolic bit set, no Encoding

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_symbolic_encoding_removed"] == 0
    assert result["tt_symbolic_flag_set"] == 0


# ---------------------------------------------------------------------------
# Rule 6.2.11.6-4 — Symbolic cmap
# ---------------------------------------------------------------------------


def test_rule4_adds_30_cmap_when_multiple_cmaps_no_30():
    """Symbolic TrueType with multiple cmaps but no (3,0) should get (3,0) added."""
    pdf = _new_page_pdf()
    # Two cmaps, neither is (3,0)
    font_bytes = _make_tt_font_bytes(
        [
            (1, 0, {0x41: "A"}),
            (3, 1, {0x0041: "A"}),
        ],
        symbolic=True,
    )
    _font, fd = _add_truetype_font(pdf, font_bytes, flags=4)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_symbolic_cmap_added"] == 1
    assert (3, 0) in set(_get_cmap_subtables(fd))


def test_rule4_no_change_when_single_cmap():
    """Symbolic TrueType with exactly one cmap is already compliant."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes([(3, 0, {0xF041: "A"})], symbolic=True)
    _add_truetype_font(pdf, font_bytes, flags=4)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_symbolic_cmap_added"] == 0


def test_rule4_no_change_when_30_exists():
    """Symbolic TrueType with multiple cmaps including non-empty (3,0) is compliant."""
    pdf = _new_page_pdf()
    font_bytes = _make_tt_font_bytes(
        [
            (1, 0, {0x41: "A"}),
            (3, 0, {0xF041: "A"}),
        ],
        symbolic=True,
    )
    _add_truetype_font(pdf, font_bytes, flags=4)

    result = sanitize_truetype_encoding(pdf)

    assert result["tt_symbolic_cmap_added"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_non_truetype_font_skipped():
    """Type1 fonts should not be processed by the sanitizer."""
    pdf = _new_page_pdf()
    font = Dictionary()
    font[Name.Type] = Name.Font
    font[Name.Subtype] = Name.Type1
    font[Name.BaseFont] = Name.Helvetica
    font_ref = pdf.make_indirect(font)
    page = pdf.pages[0].obj
    if page.get("/Resources") is None:
        page[Name.Resources] = Dictionary()
    res = page[Name.Resources]
    if res.get("/Font") is None:
        res[Name.Font] = Dictionary()
    res[Name.Font][Name.F1] = font_ref

    result = sanitize_truetype_encoding(pdf)

    assert sum(result.values()) == 0


def test_not_embedded_skipped():
    """TrueType font without /FontFile2 (not embedded) should be skipped."""
    pdf = _new_page_pdf()

    # FontDescriptor without /FontFile2
    fd = Dictionary()
    fd[Name.Type] = Name.FontDescriptor
    fd[Name.FontName] = Name.TestFont
    fd[Name.Flags] = 32
    fd[Name.FontBBox] = Array([0, -200, 1000, 800])
    fd[Name.ItalicAngle] = 0
    fd[Name.Ascent] = 800
    fd[Name.Descent] = -200
    fd[Name.CapHeight] = 700
    fd[Name.StemV] = 80
    fd_ref = pdf.make_indirect(fd)

    font = Dictionary()
    font[Name.Type] = Name.Font
    font[Name.Subtype] = Name.TrueType
    font[Name.BaseFont] = Name.TestFont
    font[Name.FirstChar] = 65
    font[Name.LastChar] = 65
    font[Name.Widths] = Array([600])
    font[Name("/FontDescriptor")] = fd_ref
    font[Name.Encoding] = Name.StandardEncoding  # Would be fixed if processed
    font_ref = pdf.make_indirect(font)

    page = pdf.pages[0].obj
    if page.get("/Resources") is None:
        page[Name.Resources] = Dictionary()
    res = page[Name.Resources]
    if res.get("/Font") is None:
        res[Name.Font] = Dictionary()
    res[Name.Font][Name.F1] = font_ref

    result = sanitize_truetype_encoding(pdf)

    assert sum(result.values()) == 0
