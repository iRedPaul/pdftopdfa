# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/font_notdef.py (.notdef glyph sanitizer)."""

import struct
from io import BytesIO

import pikepdf
from conftest import new_pdf, open_pdf
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.font_notdef import sanitize_font_notdef
from pdftopdfa.utils import resolve_indirect


def _make_ttfont_data(*, include_notdef: bool = True) -> bytes:
    """Creates a minimal TrueType font, optionally without .notdef.

    Args:
        include_notdef: If True, includes a .notdef glyph (valid font).
            If False, builds a font whose glyph-order slot 0 is *not*
            named ``.notdef`` (uses ``glyph00000`` instead).

    Returns:
        Serialized font bytes.
    """
    if include_notdef:
        glyph_names = [".notdef", "space", "A"]
        widths = {".notdef": 500, "space": 250, "A": 600}
    else:
        # Use a placeholder name for glyph 0 so .notdef is absent.
        glyph_names = ["glyph00000", "space", "A"]
        widths = {"glyph00000": 500, "space": 250, "A": 600}

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap({0x20: "space", ord("A"): "A"})
    fb.setupGlyf({name: Glyph() for name in glyph_names})

    metrics = {name: (widths[name], 0) for name in glyph_names}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    tt = fb.font
    buf = BytesIO()
    tt.save(buf)
    tt.close()
    buf.seek(0)
    return buf.read()


def _build_simple_font_pdf(
    pdf: Pdf,
    font_data: bytes,
    *,
    font_file_key: str = "/FontFile2",
) -> None:
    """Adds a page with a simple TrueType font using the given font data."""
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    fd = pdf.make_indirect(
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
        )
    )
    fd[Name(font_file_key)] = pdf.make_indirect(font_stream)

    font_dict = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FontDescriptor=fd,
            FirstChar=32,
            LastChar=65,
            Widths=Array([250] + [0] * 32 + [600]),
            Encoding=Name.WinAnsiEncoding,
        )
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=font_dict)),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _build_cidfont_pdf(pdf: Pdf, font_data: bytes) -> None:
    """Adds a page with a Type0/CIDFont using the given font data."""
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    fd = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCIDFont"),
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

    cidfont = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestCIDFont"),
            CIDSystemInfo=Dictionary(
                Registry="Adobe", Ordering="Identity", Supplement=0
            ),
            FontDescriptor=fd,
            DW=1000,
            W=Array([0, Array([500, 250, 600])]),
            CIDToGIDMap=Name.Identity,
        )
    )

    type0_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=type0_font)),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _roundtrip(pdf: Pdf) -> Pdf:
    """Save and reopen a PDF to get proper indirect references."""
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return open_pdf(buf)


class TestFontWithNotdef:
    """Fonts that already have .notdef should not be modified."""

    def test_font_with_notdef_not_modified(self) -> None:
        font_data = _make_ttfont_data(include_notdef=True)
        pdf = new_pdf()
        _build_simple_font_pdf(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0


class TestFontWithoutNotdef:
    """Fonts missing .notdef should be fixed."""

    def test_font_without_notdef_is_fixed(self) -> None:
        font_data = _make_ttfont_data(include_notdef=False)

        # Verify .notdef is actually absent from the glyph order
        tt = TTFont(BytesIO(font_data))
        assert ".notdef" not in tt.getGlyphOrder(), tt.getGlyphOrder()
        tt.close()

        pdf = new_pdf()
        _build_simple_font_pdf(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 1

        # Verify the font now has .notdef
        from pdftopdfa.utils import resolve_indirect

        font_obj = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        fd = resolve_indirect(font_obj["/FontDescriptor"])
        stream = resolve_indirect(fd["/FontFile2"])
        fixed_data = bytes(stream.read_bytes())
        tt = TTFont(BytesIO(fixed_data))
        assert ".notdef" in tt.getGlyphOrder()
        tt.close()


class TestSkipConditions:
    """Tests for fonts that should be skipped."""

    def test_type3_font_skipped(self) -> None:
        pdf = new_pdf()

        type3_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                FontBBox=Array([0, 0, 1000, 1000]),
                FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
                CharProcs=Dictionary(),
                Encoding=Dictionary(
                    Type=Name.Encoding,
                    Differences=Array([]),
                ),
                FirstChar=0,
                LastChar=0,
                Widths=Array([500]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=type3_font)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0

    def test_font_without_embedded_data_skipped(self) -> None:
        pdf = new_pdf()

        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
            )
        )

        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=fd,
                FirstChar=0,
                LastChar=255,
                Widths=Array([500] * 256),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0

    def test_corrupt_font_data_skipped(self) -> None:
        pdf = new_pdf()

        font_stream = pdf.make_stream(b"not a real font file")
        font_stream[Name.Length1] = 20

        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/BadFont"),
                Flags=32,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/BadFont"),
                FontDescriptor=fd,
                FirstChar=0,
                LastChar=255,
                Widths=Array([500] * 256),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0


def _build_cidfont_pdf_with_explicit_gidmap(pdf: Pdf, font_data: bytes) -> None:
    """Adds a page with a Type0/CIDFont using an explicit CIDToGIDMap stream.

    The map covers CIDs 0-2 mapping to GIDs 0, 1, 2 respectively.
    """
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    fd = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCIDFont"),
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

    # Build an explicit CIDToGIDMap: CID 0→GID 0, CID 1→GID 1, CID 2→GID 2
    gid_map_data = struct.pack(">3H", 0, 1, 2)
    gid_map_stream = pdf.make_stream(gid_map_data)

    cidfont = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestCIDFont"),
            CIDSystemInfo=Dictionary(
                Registry="Adobe", Ordering="Identity", Supplement=0
            ),
            FontDescriptor=fd,
            DW=1000,
            W=Array([0, Array([500, 250, 600])]),
            CIDToGIDMap=pdf.make_indirect(gid_map_stream),
        )
    )

    type0_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=type0_font)),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


class TestCIDFont:
    """Tests for CIDFont (Type0) .notdef checking."""

    def test_cidfont_notdef_checked(self) -> None:
        font_data = _make_ttfont_data(include_notdef=False)
        pdf = new_pdf()
        _build_cidfont_pdf(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 1

    def test_cidfont_with_notdef_not_modified(self) -> None:
        font_data = _make_ttfont_data(include_notdef=True)
        pdf = new_pdf()
        _build_cidfont_pdf(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0

    def test_explicit_cidtogidmap_updated_on_notdef_insert(self) -> None:
        """When .notdef is inserted at GID 0, an explicit CIDToGIDMap
        stream must have every GID incremented by 1."""
        font_data = _make_ttfont_data(include_notdef=False)
        pdf = new_pdf()
        _build_cidfont_pdf_with_explicit_gidmap(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)
        assert result["notdef_fixed"] == 1

        # Verify the CIDToGIDMap was updated
        type0 = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        cidfont = resolve_indirect(resolve_indirect(type0["/DescendantFonts"])[0])
        gidmap = resolve_indirect(cidfont["/CIDToGIDMap"])
        data = bytes(gidmap.read_bytes())
        n = len(data) // 2
        gids = struct.unpack(f">{n}H", data)
        # Original was (0, 1, 2), should now be (1, 2, 3)
        assert gids == (1, 2, 3)

    def test_identity_cidtogidmap_replaced_with_shifted_stream(self) -> None:
        """CIDToGIDMap=/Identity must be replaced with an explicit stream
        that shifts every GID by +1 after .notdef insertion."""
        font_data = _make_ttfont_data(include_notdef=False)
        pdf = new_pdf()
        _build_cidfont_pdf(pdf, font_data)
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)
        assert result["notdef_fixed"] == 1

        type0 = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        cidfont = resolve_indirect(resolve_indirect(type0["/DescendantFonts"])[0])
        # Should now be an explicit stream, not /Identity
        gidmap = resolve_indirect(cidfont["/CIDToGIDMap"])
        data = bytes(gidmap.read_bytes())
        n = len(data) // 2
        gids = struct.unpack(f">{n}H", data)
        # Original font had 3 glyphs (glyph00000, space, A).
        # Identity meant CID 0→GID 0, CID 1→GID 1, CID 2→GID 2.
        # After .notdef insertion, all shift by +1.
        assert gids[:3] == (1, 2, 3)


class TestFontFile3Subtype:
    """FontFile3 streams must preserve /Subtype."""

    def test_fontfile3_subtype_preserved(self) -> None:
        """When replacing a FontFile3 stream, /Subtype must be copied."""
        font_data = _make_ttfont_data(include_notdef=False)
        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name("/Subtype")] = Name("/OpenType")

        fd = pdf.make_indirect(
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
            )
        )
        fd[Name("/FontFile3")] = pdf.make_indirect(font_stream)

        font_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=fd,
                FirstChar=32,
                LastChar=65,
                Widths=Array([250] + [0] * 32 + [600]),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)
        assert result["notdef_fixed"] == 1

        font_obj = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        fd = resolve_indirect(font_obj["/FontDescriptor"])
        stream = resolve_indirect(fd["/FontFile3"])
        assert stream.get("/Subtype") == Name("/OpenType")

    def test_fontfile3_without_subtype_no_error(self) -> None:
        """FontFile3 without /Subtype should not crash."""
        font_data = _make_ttfont_data(include_notdef=False)
        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        # Intentionally no /Subtype

        fd = pdf.make_indirect(
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
            )
        )
        fd[Name("/FontFile3")] = pdf.make_indirect(font_stream)

        font_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=fd,
                FirstChar=32,
                LastChar=65,
                Widths=Array([250] + [0] * 32 + [600]),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_notdef(pdf)
        assert result["notdef_fixed"] == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_pdf_without_fonts(self) -> None:
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_font_notdef(pdf)

        assert result["notdef_fixed"] == 0
