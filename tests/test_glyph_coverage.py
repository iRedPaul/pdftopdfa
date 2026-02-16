# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/glyph_coverage.py (glyph coverage sanitizer)."""

from io import BytesIO

import pikepdf
from conftest import new_pdf, open_pdf
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.psCharStrings import T2CharString
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.glyph_coverage import sanitize_glyph_coverage


def _make_ttfont_data(glyph_count: int = 3) -> bytes:
    """Creates a minimal TrueType font with the given number of glyphs.

    Glyph order: [".notdef", "glyph00001", "glyph00002", ...]

    Args:
        glyph_count: Total number of glyphs including .notdef.

    Returns:
        Serialized font bytes.
    """
    glyph_names = [".notdef"]
    for i in range(1, glyph_count):
        glyph_names.append(f"glyph{i:05d}")

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_names)

    # Minimal cmap
    cmap = {}
    if glyph_count > 1:
        cmap[0x20] = glyph_names[1]
    fb.setupCharacterMap(cmap)

    fb.setupGlyf({name: Glyph() for name in glyph_names})

    metrics = {name: (500, 0) for name in glyph_names}
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


def _make_simple_font_data(*, include_glyph_a: bool = True) -> bytes:
    """Creates a minimal TrueType font for simple font tests.

    Always includes .notdef and space.  Optionally includes glyph "A".

    Args:
        include_glyph_a: Whether to include glyph "A".

    Returns:
        Serialized font bytes.
    """
    glyph_names = [".notdef", "space"]
    cmap: dict[int, str] = {0x20: "space"}

    if include_glyph_a:
        glyph_names.append("A")
        cmap[0x41] = "A"

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({name: Glyph() for name in glyph_names})

    metrics = {name: (500, 0) for name in glyph_names}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestSimple", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    tt = fb.font
    buf = BytesIO()
    tt.save(buf)
    tt.close()
    buf.seek(0)
    return buf.read()


def _build_cidfont_pdf_with_content(
    pdf: Pdf,
    font_data: bytes,
    gids_to_reference: list[int],
    *,
    cidtogidmap: str | bytes = "identity",
) -> None:
    """Builds a CIDFont PDF with a content stream referencing specific GIDs.

    Args:
        pdf: PDF to add the page to.
        font_data: Serialized font bytes.
        gids_to_reference: List of GID values to reference in the content.
        cidtogidmap: "identity" for /Identity, or raw bytes for a
            CIDToGIDMap stream.
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

    cidfont_dict = Dictionary(
        Type=Name.Font,
        Subtype=Name("/CIDFontType2"),
        BaseFont=Name("/TestCIDFont"),
        CIDSystemInfo=Dictionary(Registry="Adobe", Ordering="Identity", Supplement=0),
        FontDescriptor=fd,
        DW=1000,
    )

    cidfont = pdf.make_indirect(cidfont_dict)

    if cidtogidmap == "identity":
        cidfont[Name.CIDToGIDMap] = Name.Identity
    elif isinstance(cidtogidmap, bytes):
        cidfont[Name.CIDToGIDMap] = pdf.make_indirect(pdf.make_stream(cidtogidmap))

    type0_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCIDFont"),
            Encoding=Name("/Identity-H"),
            DescendantFonts=Array([cidfont]),
        )
    )

    # Build content stream with text operators referencing the GIDs
    # Each GID is encoded as 2-byte big-endian in a hex string
    hex_parts = "".join(f"{gid:04X}" for gid in gids_to_reference)
    content = f"BT /F1 12 Tf <{hex_parts}> Tj ET".encode()

    content_stream = pdf.make_stream(content)

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=type0_font)),
        Contents=content_stream,
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _build_simple_font_pdf_with_content(
    pdf: Pdf,
    font_data: bytes,
    char_codes: list[int],
) -> None:
    """Builds a simple font PDF with a content stream using given char codes.

    Args:
        pdf: PDF to add the page to.
        font_data: Serialized font bytes.
        char_codes: Character codes to reference (single-byte values).
    """
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    fd = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestSimple"),
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

    font_dict = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestSimple"),
            FontDescriptor=fd,
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
            Encoding=Name.WinAnsiEncoding,
        )
    )

    # Build content stream referencing the character codes
    hex_str = "".join(f"{c:02X}" for c in char_codes)
    content = f"BT /F1 12 Tf <{hex_str}> Tj ET".encode()

    content_stream = pdf.make_stream(content)

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        Contents=content_stream,
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _roundtrip(pdf: Pdf) -> Pdf:
    """Save and reopen a PDF to get proper indirect references."""
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return open_pdf(buf)


class TestCIDFontIdentityMap:
    """CIDFont with Identity CIDToGIDMap — missing GIDs should be added."""

    def test_missing_gids_are_added(self) -> None:
        """Font has 3 glyphs; content references GID 5 — should add glyphs."""
        font_data = _make_ttfont_data(glyph_count=3)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [1, 5])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 1  # GID 5 is the only missing ref

        # Verify the font now has enough glyphs (at least 6: GIDs 0-5)
        from pdftopdfa.utils import resolve_indirect

        type0 = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        desc = resolve_indirect(resolve_indirect(type0["/DescendantFonts"])[0])
        fd = resolve_indirect(desc["/FontDescriptor"])
        stream = resolve_indirect(fd["/FontFile2"])
        fixed_data = bytes(stream.read_bytes())
        tt = TTFont(BytesIO(fixed_data))
        assert len(tt.getGlyphOrder()) >= 6
        tt.close()

    def test_no_missing_gids(self) -> None:
        """All referenced GIDs exist — no changes."""
        font_data = _make_ttfont_data(glyph_count=5)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [1, 2, 3])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

    def test_multiple_missing_gids(self) -> None:
        """Multiple GIDs beyond the font — all should be added."""
        font_data = _make_ttfont_data(glyph_count=2)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [3, 5, 7])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 3  # GIDs 3, 5, 7

        # Verify the font now has at least 8 glyphs (GIDs 0-7)
        from pdftopdfa.utils import resolve_indirect

        type0 = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        desc = resolve_indirect(resolve_indirect(type0["/DescendantFonts"])[0])
        fd = resolve_indirect(desc["/FontDescriptor"])
        stream = resolve_indirect(fd["/FontFile2"])
        fixed_data = bytes(stream.read_bytes())
        tt = TTFont(BytesIO(fixed_data))
        assert len(tt.getGlyphOrder()) == 8
        assert tt["maxp"].numGlyphs == 8
        tt.close()

    def test_gid_zero_not_missing(self) -> None:
        """GID 0 (.notdef) exists — should not be flagged as missing."""
        font_data = _make_ttfont_data(glyph_count=3)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [0, 1])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0


class TestCIDFontStreamMap:
    """CIDFont with CIDToGIDMap stream — missing GIDs should be added."""

    def test_stream_map_missing_gids(self) -> None:
        """CIDToGIDMap maps CID 1 to GID 10; font has 3 glyphs — should fix."""
        font_data = _make_ttfont_data(glyph_count=3)

        # Build a CIDToGIDMap stream: CID 0 -> GID 0, CID 1 -> GID 10
        # Stream is 2 bytes per CID, big-endian
        import struct

        map_data = struct.pack(">HH", 0, 10)  # CID 0 -> GID 0, CID 1 -> GID 10

        pdf = new_pdf()
        # Content references CID 1 (which maps to GID 10)
        _build_cidfont_pdf_with_content(pdf, font_data, [1], cidtogidmap=map_data)
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 1  # GID 10 was missing


class TestSimpleFont:
    """Simple font with encoding — missing glyph names should be added."""

    def test_missing_glyph_name_added(self) -> None:
        """Font lacks glyph 'A' but encoding references it — should add."""
        font_data = _make_simple_font_data(include_glyph_a=False)

        # Verify 'A' is not in the font
        tt = TTFont(BytesIO(font_data))
        assert "A" not in tt.getGlyphOrder()
        tt.close()

        pdf = new_pdf()
        _build_simple_font_pdf_with_content(pdf, font_data, [0x41])  # 'A'
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 1

        # Verify 'A' is now in the font
        from pdftopdfa.utils import resolve_indirect

        font_obj = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        fd = resolve_indirect(font_obj["/FontDescriptor"])
        stream = resolve_indirect(fd["/FontFile2"])
        fixed_data = bytes(stream.read_bytes())
        tt = TTFont(BytesIO(fixed_data))
        assert "A" in tt.getGlyphOrder()
        tt.close()

    def test_all_glyphs_present(self) -> None:
        """All referenced glyphs exist — no changes."""
        font_data = _make_simple_font_data(include_glyph_a=True)
        pdf = new_pdf()
        # Reference 'space' (0x20) which exists
        _build_simple_font_pdf_with_content(pdf, font_data, [0x20])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0


class TestSkipConditions:
    """Tests for fonts that should be skipped."""

    def test_pdf_without_fonts(self) -> None:
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

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

        content = b"BT /F1 12 Tf (A) Tj ET"
        content_stream = pdf.make_stream(content)

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=type3_font)),
            Contents=content_stream,
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

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

        content = b"BT /F1 12 Tf (A) Tj ET"
        content_stream = pdf.make_stream(content)

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
            Contents=content_stream,
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

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

        cidfont = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/BadFont"),
                CIDSystemInfo=Dictionary(
                    Registry="Adobe", Ordering="Identity", Supplement=0
                ),
                FontDescriptor=fd,
                DW=1000,
                CIDToGIDMap=Name.Identity,
            )
        )

        type0_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/BadFont"),
                Encoding=Name("/Identity-H"),
                DescendantFonts=Array([cidfont]),
            )
        )

        content = b"BT /F1 12 Tf <0005> Tj ET"
        content_stream = pdf.make_stream(content)

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=type0_font)),
            Contents=content_stream,
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_font_without_content_stream(self) -> None:
        """Font exists but no content stream references it — skip."""
        font_data = _make_ttfont_data(glyph_count=3)
        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        fd = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        cidfont = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/TestFont"),
                CIDSystemInfo=Dictionary(
                    Registry="Adobe", Ordering="Identity", Supplement=0
                ),
                FontDescriptor=fd,
                DW=1000,
                CIDToGIDMap=Name.Identity,
            )
        )

        type0_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/TestFont"),
                Encoding=Name("/Identity-H"),
                DescendantFonts=Array([cidfont]),
            )
        )

        # Page has font in resources but empty content
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=type0_font)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

    def test_same_font_on_multiple_pages(self) -> None:
        """Same font used on two pages — should only process once."""
        font_data = _make_ttfont_data(glyph_count=3)
        pdf = new_pdf()

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

        # Two pages using the same font, each referencing GID 5
        for _ in range(2):
            content = b"BT /F1 12 Tf <0005> Tj ET"
            content_stream = pdf.make_stream(content)
            page_dict = Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=type0_font)),
                Contents=content_stream,
            )
            pdf.pages.append(pikepdf.Page(page_dict))

        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        # Only 1 fix, not 2 — font should be processed once
        assert result["glyphs_added"] == 1


class TestIntegration:
    """Integration tests with sanitize_for_pdfa."""

    def test_glyphs_added_in_result(self) -> None:
        """sanitize_for_pdfa includes glyphs_added in result dict."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        font_data = _make_ttfont_data(glyph_count=3)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [1, 5])
        pdf = _roundtrip(pdf)

        result = sanitize_for_pdfa(pdf, level="2b")

        assert "glyphs_added" in result
        assert result["glyphs_added"] == 1

    def test_no_glyphs_needed(self) -> None:
        """sanitize_for_pdfa reports 0 when no glyphs need adding."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        font_data = _make_ttfont_data(glyph_count=5)
        pdf = new_pdf()
        _build_cidfont_pdf_with_content(pdf, font_data, [1, 2])
        pdf = _roundtrip(pdf)

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["glyphs_added"] == 0


# ---------------------------------------------------------------------------
# Type1C (bare CFF) font helpers and tests
# ---------------------------------------------------------------------------


def _make_bare_cff_data(*, glyph_names: list[str] | None = None) -> bytes:
    """Creates a minimal bare CFF font (Type1C) as standalone CFF data.

    Args:
        glyph_names: Glyph names to include. Defaults to [".notdef", "space"].

    Returns:
        Raw CFF table bytes (not wrapped in OTF).
    """
    if glyph_names is None:
        glyph_names = [".notdef", "space"]

    cmap: dict[int, str] = {}
    if "space" in glyph_names:
        cmap[0x20] = "space"
    if "A" in glyph_names:
        cmap[0x41] = "A"
    if "period" in glyph_names:
        cmap[0x2E] = "period"
    if "numbersign" in glyph_names:
        cmap[0x23] = "numbersign"

    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap(cmap)

    # CFF fonts use setupCFF, not setupGlyf
    char_strings = {name: T2CharString() for name in glyph_names}
    for cs in char_strings.values():
        cs.program = [0, "hmoveto", "endchar"]
    fb.setupCFF(
        psName="TestCFF",
        fontInfo={},
        charStringsDict=char_strings,
        privateDict={},
    )
    metrics = {name: (500, 0) for name in glyph_names}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestCFF", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=1000)

    tt = fb.font
    cff_data = tt.getTableData("CFF ")
    tt.close()
    return cff_data


def _build_simple_cff_font_pdf(
    pdf: Pdf,
    cff_data: bytes,
    char_codes: list[int],
) -> None:
    """Builds a simple font PDF using bare CFF data in FontFile3/Type1C.

    Args:
        pdf: PDF to add the page to.
        cff_data: Raw CFF table bytes.
        char_codes: Character codes to reference (single-byte values).
    """
    font_stream = pdf.make_stream(cff_data)
    font_stream[Name.Subtype] = Name("/Type1C")

    fd = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestCFF"),
            Flags=32,
            FontBBox=Array([0, -200, 1000, 800]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile3=pdf.make_indirect(font_stream),
        )
    )

    font_dict = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/TestCFF"),
            FontDescriptor=fd,
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
            Encoding=Name.WinAnsiEncoding,
        )
    )

    hex_str = "".join(f"{c:02X}" for c in char_codes)
    content = f"BT /F1 12 Tf <{hex_str}> Tj ET".encode()
    content_stream = pdf.make_stream(content)

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        Contents=content_stream,
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _make_type1_pfa_data(*, include_dollar: bool = True) -> tuple[bytes, int, int, int]:
    """Creates a minimal Type1 PFA font for testing.

    Uses the real Isartor test font (LUFLYP+ArialMT) which is a Type1
    PFA subset. For tests needing a missing glyph, the 'dollar' glyph
    is intentionally absent in the original.

    Returns a tuple of (font_data, length1, length2, length3).
    If include_dollar is False, uses the original font (without dollar).
    If include_dollar is True, adds the dollar glyph first.
    """
    import os

    # Use the real Isartor test file as the Type1 PFA source
    isartor_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "veraPDF-corpus-staging",
        "Isartor test files",
        "PDFA-1b",
        "6.3 Fonts",
        "6.3.5 Font subsets",
        "isartor-6-3-5-t01-fail-c.pdf",
    )
    if not os.path.exists(isartor_path):
        return None, 0, 0, 0

    _pdf = open_pdf(isartor_path)
    _font = _pdf.pages[0]["/Resources"]["/Font"]["/F0"]
    _desc = _font["/FontDescriptor"]
    _ff = _desc["/FontFile"]
    data = bytes(_ff.read_bytes())
    l1 = int(_ff["/Length1"])
    l2 = int(_ff["/Length2"])
    l3 = int(_ff["/Length3"])

    if include_dollar:
        from pdftopdfa.sanitizers.glyph_coverage import (
            _add_glyphs_to_type1,
        )

        data = _add_glyphs_to_type1(
            data,
            ["dollar"],
            {"dollar": 0},
            length1=l1,
            length2=l2,
        )
        # Recalculate lengths
        text = data.decode("latin-1")
        eexec_pos = text.find("currentfile eexec")
        header_end = text.index("\n", eexec_pos) + 1
        l1 = header_end
        remaining = data[header_end:]
        encrypted_end = len(remaining)
        for i in range(len(remaining) - 1, 0, -1):
            if remaining[i] not in (ord("0"), ord("\n"), ord("\r"), ord(" ")):
                encrypted_end = i + 1
                break
        l2 = encrypted_end
        l3 = len(remaining) - encrypted_end

    return data, l1, l2, l3


def _build_type1_font_pdf(
    pdf: Pdf,
    font_data: bytes,
    length1: int,
    length2: int,
    length3: int,
    char_codes: list[int],
) -> None:
    """Builds a PDF with an embedded Type1 font in /FontFile.

    Args:
        pdf: PDF to add the page to.
        font_data: Raw Type1 PFA font bytes.
        length1: Cleartext portion length.
        length2: Encrypted portion length.
        length3: Trailing zeros length.
        char_codes: Character codes to reference.
    """
    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = length1
    font_stream[Name.Length2] = length2
    font_stream[Name.Length3] = length3

    fd = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/LUFLYP+ArialMT"),
            Flags=32,
            FontBBox=Array([0, -200, 1000, 800]),
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontFile=pdf.make_indirect(font_stream),
        )
    )

    font_dict = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/LUFLYP+ArialMT"),
            FontDescriptor=fd,
            FirstChar=0,
            LastChar=255,
            Widths=Array([500] * 256),
            Encoding=Name.WinAnsiEncoding,
        )
    )

    hex_str = "".join(f"{c:02X}" for c in char_codes)
    content = f"BT /F1 12 Tf <{hex_str}> Tj ET".encode()
    content_stream = pdf.make_stream(content)

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        Contents=content_stream,
    )
    pdf.pages.append(pikepdf.Page(page_dict))


class TestType1Font:
    """Type1 PFA/PFB font — missing glyphs should be added."""

    def test_missing_glyph_added_to_type1(self) -> None:
        """Type1 font missing 'dollar' gets it added."""
        data, l1, l2, l3 = _make_type1_pfa_data(include_dollar=False)
        if data is None:
            return  # Skip if test file not available

        # Verify 'dollar' is not in the font
        from pdftopdfa.sanitizers.glyph_coverage import (
            _get_type1_glyph_names,
        )

        names = _get_type1_glyph_names(data)
        assert names is not None
        assert "dollar" not in names

        pdf = new_pdf()
        _build_type1_font_pdf(pdf, data, l1, l2, l3, [0x24])  # '$'
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 1

        # Verify 'dollar' is now in the font
        from pdftopdfa.utils import resolve_indirect

        font_obj = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        fd_obj = resolve_indirect(font_obj["/FontDescriptor"])
        stream = resolve_indirect(fd_obj["/FontFile"])
        fixed_data = bytes(stream.read_bytes())

        fixed_names = _get_type1_glyph_names(fixed_data)
        assert fixed_names is not None
        assert "dollar" in fixed_names

    def test_type1_all_glyphs_present(self) -> None:
        """Type1 font with all referenced glyphs — no changes."""
        data, l1, l2, l3 = _make_type1_pfa_data(include_dollar=True)
        if data is None:
            return  # Skip if test file not available

        pdf = new_pdf()
        # Reference 'space' (0x20) which exists
        _build_type1_font_pdf(pdf, data, l1, l2, l3, [0x20])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0

    def test_type1_glyph_name_extraction(self) -> None:
        """Verify _get_type1_glyph_names extracts names from PFA font."""
        data, _, _, _ = _make_type1_pfa_data(include_dollar=False)
        if data is None:
            return

        from pdftopdfa.sanitizers.glyph_coverage import (
            _get_type1_glyph_names,
        )

        names = _get_type1_glyph_names(data)
        assert names is not None
        assert ".notdef" in names
        assert "space" in names
        assert "dollar" not in names

    def test_add_glyphs_to_type1(self) -> None:
        """Verify _add_glyphs_to_type1 creates valid font data."""
        data, l1, l2, l3 = _make_type1_pfa_data(include_dollar=False)
        if data is None:
            return

        from pdftopdfa.sanitizers.glyph_coverage import (
            _add_glyphs_to_type1,
            _get_type1_glyph_names,
        )

        result = _add_glyphs_to_type1(
            data,
            ["dollar"],
            {"dollar": 500},
            length1=l1,
            length2=l2,
        )
        assert result is not None
        assert len(result) > len(data)

        # Verify the new font has the dollar glyph
        names = _get_type1_glyph_names(result)
        assert names is not None
        assert "dollar" in names
        # Original glyphs still present
        assert ".notdef" in names
        assert "space" in names


class TestType1CFont:
    """Type1C (bare CFF) font — missing glyphs should be added."""

    def test_missing_glyph_added_to_type1c(self) -> None:
        """Type1C font missing 'A' gets it added."""
        cff_data = _make_bare_cff_data(glyph_names=[".notdef", "space"])

        pdf = new_pdf()
        _build_simple_cff_font_pdf(pdf, cff_data, [0x41])  # 'A'
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 1

        # Verify 'A' is now in the font
        from pdftopdfa.utils import resolve_indirect

        font_obj = resolve_indirect(pdf.pages[0].Resources.Font["/F1"])
        fd_obj = resolve_indirect(font_obj["/FontDescriptor"])
        stream = resolve_indirect(fd_obj["/FontFile3"])
        fixed_cff = bytes(stream.read_bytes())

        # Wrap back in OTF to parse with TTFont
        from pdftopdfa.sanitizers.glyph_coverage import _wrap_cff_in_otf

        otf_data = _wrap_cff_in_otf(fixed_cff)
        tt = TTFont(BytesIO(otf_data))
        assert "A" in tt.getGlyphOrder()
        tt.close()

    def test_type1c_all_glyphs_present(self) -> None:
        """Type1C font with all referenced glyphs — no changes."""
        cff_data = _make_bare_cff_data(glyph_names=[".notdef", "space"])

        pdf = new_pdf()
        # Reference 'space' (0x20) which exists
        _build_simple_cff_font_pdf(pdf, cff_data, [0x20])
        pdf = _roundtrip(pdf)

        result = sanitize_glyph_coverage(pdf)

        assert result["glyphs_added"] == 0
