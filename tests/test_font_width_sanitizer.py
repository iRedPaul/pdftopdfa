# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/font_widths.py (font width sanitizer)."""

from io import BytesIO

import pikepdf
from conftest import new_pdf, open_pdf, resolve
from fontTools.ttLib import TTFont
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.font_widths import (
    _parse_w_array,
    _validate_font_program,
    sanitize_font_widths,
)


def _make_minimal_ttfont(
    *,
    units_per_em: int = 1000,
    glyph_widths: dict[str, int] | None = None,
) -> tuple[bytes, TTFont]:
    """Creates a minimal TrueType font with specified glyph widths.

    Returns:
        Tuple of (font_data_bytes, TTFont object).
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    if glyph_widths is None:
        glyph_widths = {
            ".notdef": 500,
            "space": 250,
            "A": 600,
            "B": 650,
            "C": 700,
        }

    glyph_names = list(glyph_widths.keys())

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_names)

    # Create character mappings (Unicode -> glyph name)
    cmap = {}
    for name in glyph_names:
        if name == ".notdef":
            continue
        if name == "space":
            cmap[0x20] = "space"
        elif len(name) == 1 and name.isalpha():
            cmap[ord(name)] = name
    fb.setupCharacterMap(cmap)

    # Setup glyph outlines (empty glyphs)
    empty = Glyph()
    fb.setupGlyf({name: empty for name in glyph_names})

    # Setup horizontal metrics
    metrics = {}
    for name, width in glyph_widths.items():
        metrics[name] = (width, 0)  # (advance_width, lsb)
    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sCapHeight=700)
    fb.setupPost()
    fb.setupHead(unitsPerEm=units_per_em)

    tt_font = fb.font
    buf = BytesIO()
    tt_font.save(buf)
    buf.seek(0)
    font_data = buf.read()

    # Reparse to get a clean object
    tt_font = TTFont(BytesIO(font_data))

    return font_data, tt_font


def _make_simple_font_with_widths(
    pdf: Pdf,
    *,
    widths: list[int] | None = None,
    first_char: int = 32,
    last_char: int = 67,
    encoding: str = "/WinAnsiEncoding",
    font_data: bytes | None = None,
) -> Dictionary:
    """Creates a TrueType font dict with embedded data and specified widths.

    By default, creates a font covering space (32), A (65), B (66), C (67)
    with intentionally wrong widths for testing.
    """
    if font_data is None:
        font_data, _ = _make_minimal_ttfont()

    num_chars = last_char - first_char + 1
    if widths is None:
        # Mostly wrong widths for testing. Keep space (index 0) correct
        # so that the mismatch ratio stays below the 80% threshold.
        widths = [250] + [999] * (num_chars - 1)

    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    font_descriptor = pdf.make_indirect(
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
            FontFile2=pdf.make_indirect(font_stream),
        )
    )

    font_dict = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.TrueType,
            BaseFont=Name("/TestFont"),
            FontDescriptor=font_descriptor,
            FirstChar=first_char,
            LastChar=last_char,
            Widths=Array(widths),
            Encoding=Name(encoding),
        )
    )

    return font_dict


def _build_pdf_with_font(pdf: Pdf, font: Dictionary) -> None:
    """Add a page with the given font to the PDF."""
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(
            Font=Dictionary(F1=font),
        ),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _roundtrip(pdf: Pdf) -> Pdf:
    """Save and reopen a PDF to get proper indirect references."""
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return open_pdf(buf)


def _make_cidfont_with_widths(
    pdf: Pdf,
    *,
    w_array: list | None = None,
    default_width: int = 1000,
    font_data: bytes | None = None,
) -> Dictionary:
    """Creates a Type0/CIDFont with embedded data and specified /W array."""
    if font_data is None:
        font_data, _ = _make_minimal_ttfont()

    font_stream = pdf.make_stream(font_data)
    font_stream[Name.Length1] = len(font_data)

    font_descriptor = pdf.make_indirect(
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

    if w_array is None:
        # Mostly wrong widths for testing. Keep .notdef (CID 0) correct
        # so that the mismatch ratio stays at 80% (not above).
        w_array = [0, Array([500, 999, 999, 999, 999])]

    cid_system_info = Dictionary(
        Registry="Adobe",
        Ordering="Identity",
        Supplement=0,
    )

    cidfont = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestCIDFont"),
            CIDSystemInfo=cid_system_info,
            FontDescriptor=font_descriptor,
            DW=default_width,
            W=Array(w_array),
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

    return type0_font


class TestSimpleFontWidthFix:
    """Tests for simple font (TrueType) width correction."""

    def test_wrong_widths_are_corrected(self) -> None:
        """Widths that don't match the font program are corrected."""
        pdf = new_pdf()
        font = _make_simple_font_with_widths(pdf)
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_correct_widths_not_modified(self) -> None:
        """Widths matching the font program are not changed."""
        font_data, tt_font = _make_minimal_ttfont()

        # Compute correct widths from the font
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        cmap_table = tt_font.getBestCmap()
        scale = 1000.0 / head.unitsPerEm

        # Build correct widths for WinAnsi codes 32..67
        correct_widths = []
        for code in range(32, 68):
            try:
                char = bytes([code]).decode("cp1252")
                unicode_val = ord(char)
            except UnicodeDecodeError:
                correct_widths.append(0)
                continue
            glyph_name = cmap_table.get(unicode_val)
            if glyph_name and glyph_name in hmtx.metrics:
                w = int(hmtx.metrics[glyph_name][0] * scale)
            else:
                notdef_w = hmtx.metrics.get(".notdef", (500, 0))[0]
                w = int(notdef_w * scale)
            correct_widths.append(w)
        tt_font.close()

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=correct_widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0

    def test_tolerance_within_bounds(self) -> None:
        """Widths within +-2 tolerance are not corrected."""
        font_data, tt_font = _make_minimal_ttfont()

        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        cmap_table = tt_font.getBestCmap()
        scale = 1000.0 / head.unitsPerEm

        # Build widths that are within tolerance (±2)
        widths = []
        for code in range(32, 68):
            try:
                char = bytes([code]).decode("cp1252")
                unicode_val = ord(char)
            except UnicodeDecodeError:
                widths.append(0)
                continue
            glyph_name = cmap_table.get(unicode_val)
            if glyph_name and glyph_name in hmtx.metrics:
                w = int(hmtx.metrics[glyph_name][0] * scale) + 1  # off by 1
            else:
                notdef_w = hmtx.metrics.get(".notdef", (500, 0))[0]
                w = int(notdef_w * scale) + 1
            widths.append(w)
        tt_font.close()

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0

    def test_macroman_encoding(self) -> None:
        """Widths are validated against MacRomanEncoding."""
        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            encoding="/MacRomanEncoding",
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_corrected_widths_are_valid(self) -> None:
        """After correction, widths match the font program."""
        font_data, tt_font = _make_minimal_ttfont()

        # Compute expected widths for space (32) through C (67)
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        cmap_table = tt_font.getBestCmap()
        scale = 1000.0 / head.unitsPerEm

        expected_widths: dict[int, int] = {}
        for code in range(32, 68):
            try:
                char = bytes([code]).decode("cp1252")
                unicode_val = ord(char)
            except UnicodeDecodeError:
                continue
            glyph_name = cmap_table.get(unicode_val)
            if glyph_name and glyph_name in hmtx.metrics:
                expected_widths[code] = int(hmtx.metrics[glyph_name][0] * scale)
        tt_font.close()

        pdf = new_pdf()
        font = _make_simple_font_with_widths(pdf, font_data=font_data)
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        sanitize_font_widths(pdf)

        # Check corrected widths
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        first_char = int(font_obj.FirstChar)

        for code, expected_w in expected_widths.items():
            idx = code - first_char
            if 0 <= idx < len(corrected):
                assert abs(corrected[idx] - expected_w) <= 2, (
                    f"Code {code}: got {corrected[idx]}, expected {expected_w}"
                )


class TestCIDFontWidthFix:
    """Tests for CIDFont width correction."""

    def test_wrong_cidfont_widths_are_corrected(self) -> None:
        """CIDFont /W array with wrong widths is corrected."""
        pdf = new_pdf()
        font = _make_cidfont_with_widths(pdf)
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 1

    def test_correct_cidfont_widths_not_modified(self) -> None:
        """CIDFont with correct /W array is not modified."""
        from pdftopdfa.fonts.metrics import FontMetricsExtractor

        font_data, tt_font = _make_minimal_ttfont()
        extractor = FontMetricsExtractor()
        correct_w = extractor.build_cidfont_w_array(tt_font)
        tt_font.close()

        # Convert to pikepdf-compatible format
        w_pikepdf = []
        for item in correct_w:
            if isinstance(item, list):
                w_pikepdf.append(Array(item))
            else:
                w_pikepdf.append(item)

        pdf = new_pdf()
        font = _make_cidfont_with_widths(pdf, w_array=w_pikepdf, font_data=font_data)
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 0

    def test_cidfont_without_w_array_not_modified(self) -> None:
        """CIDFont without /W array is not touched."""
        font_data, tt_font = _make_minimal_ttfont()
        tt_font.close()

        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestCIDFont"),
                Flags=32,
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
                FontDescriptor=font_descriptor,
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

        _build_pdf_with_font(pdf, type0_font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 0


class TestSkipConditions:
    """Tests for fonts that should be skipped."""

    def test_type3_empty_charprocs_not_modified(self) -> None:
        """Type3 font with empty CharProcs is not modified."""
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

        _build_pdf_with_font(pdf, type3_font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0
        assert result["cidfont_widths_fixed"] == 0
        assert result["type3_font_widths_fixed"] == 0

    def test_font_without_descriptor_skipped(self) -> None:
        """Font without FontDescriptor is skipped."""
        pdf = new_pdf()

        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/Helvetica"),
            )
        )

        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0

    def test_font_without_embedded_data_skipped(self) -> None:
        """Font with FontDescriptor but no FontFile is skipped."""
        pdf = new_pdf()

        font_descriptor = pdf.make_indirect(
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
                FontDescriptor=font_descriptor,
                FirstChar=0,
                LastChar=255,
                Widths=Array([500] * 256),
            )
        )

        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0

    def test_invalid_font_data_skipped(self) -> None:
        """Font with unparseable font data is skipped gracefully."""
        pdf = new_pdf()

        # Embed invalid font data
        font_stream = pdf.make_stream(b"not a real font file")
        font_stream[Name.Length1] = 20

        font_descriptor = pdf.make_indirect(
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
                FontDescriptor=font_descriptor,
                FirstChar=0,
                LastChar=255,
                Widths=Array([500] * 256),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_pdf_without_fonts(self) -> None:
        """PDF without any fonts doesn't crash."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0
        assert result["cidfont_widths_fixed"] == 0

    def test_empty_pdf(self) -> None:
        """Empty PDF (no pages) doesn't crash."""
        pdf = new_pdf()

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0
        assert result["cidfont_widths_fixed"] == 0

    def test_font_without_widths_entry_skipped(self) -> None:
        """Font missing /Widths entry entirely is skipped."""
        font_data, tt_font = _make_minimal_ttfont()
        tt_font.close()

        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=font_descriptor,
                Encoding=Name.WinAnsiEncoding,
            )
        )

        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 0

    def test_multiple_fonts_fixed(self) -> None:
        """Multiple fonts with wrong widths are all corrected."""
        pdf = new_pdf()

        font1 = _make_simple_font_with_widths(pdf)
        font2 = _make_simple_font_with_widths(pdf)

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font1, F2=font2),
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 2

    def test_shared_font_deduplicated(self) -> None:
        """Same font in page-level and Form XObject is counted once."""
        pdf = new_pdf()
        font = _make_simple_font_with_widths(pdf)

        form_xobj = pdf.make_stream(b"")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = pdf.make_indirect(
            Dictionary(Font=Dictionary(F1=font))
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font),
                XObject=Dictionary(Fm1=pdf.make_indirect(form_xobj)),
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1


class TestNestedStructures:
    """Tests that fonts in nested structures are found and fixed."""

    def test_font_in_form_xobject(self) -> None:
        """Font inside a Form XObject gets widths corrected."""
        pdf = new_pdf()
        font = _make_simple_font_with_widths(pdf)

        form_xobj = pdf.make_stream(b"")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = pdf.make_indirect(
            Dictionary(Font=Dictionary(F1=font))
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                XObject=Dictionary(Fm1=pdf.make_indirect(form_xobj)),
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_font_in_annotation_ap(self) -> None:
        """Font inside Annotation Appearance Stream gets widths corrected."""
        pdf = new_pdf()
        font = _make_simple_font_with_widths(pdf)

        ap_stream = pdf.make_stream(b"")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 50, 50])
        ap_stream[Name.Resources] = pdf.make_indirect(
            Dictionary(Font=Dictionary(F1=font))
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Widget"),
                Rect=Array([0, 0, 50, 50]),
                AP=Dictionary(N=pdf.make_indirect(ap_stream)),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Annots=Array([annot]),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1


class TestParseWArray:
    """Tests for the /W array parser."""

    def test_individual_format(self) -> None:
        """Parses [cid [w1 w2 w3]] format."""
        new_pdf()
        w = Array([0, Array([500, 600, 700])])
        result = _parse_w_array(w)
        assert result == {0: 500, 1: 600, 2: 700}

    def test_range_format(self) -> None:
        """Parses [cid_first cid_last width] format."""
        new_pdf()
        w = Array([10, 15, 500])
        result = _parse_w_array(w)
        assert result == {10: 500, 11: 500, 12: 500, 13: 500, 14: 500, 15: 500}

    def test_mixed_formats(self) -> None:
        """Parses mixed individual and range formats."""
        new_pdf()
        w = Array([0, Array([500, 600]), 10, 12, 700])
        result = _parse_w_array(w)
        assert result == {0: 500, 1: 600, 10: 700, 11: 700, 12: 700}

    def test_empty_w_array(self) -> None:
        """Empty /W array returns empty dict."""
        new_pdf()
        w = Array([])
        result = _parse_w_array(w)
        assert result == {}


class TestValidateFontProgram:
    """Tests for font program integrity validation."""

    def test_valid_font_passes(self) -> None:
        """A properly constructed font passes validation."""
        _, tt_font = _make_minimal_ttfont()
        assert _validate_font_program(tt_font) is True
        tt_font.close()

    def test_all_zero_widths_rejected(self) -> None:
        """Font with all zero-width glyphs is rejected."""
        _, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 0,
                "space": 0,
                "A": 0,
                "B": 0,
            }
        )
        assert _validate_font_program(tt_font) is False
        tt_font.close()

    def test_negative_advance_width_rejected(self) -> None:
        """Font with negative advance width is rejected."""
        # fontTools refuses to serialize negative advances (unsigned 16-bit),
        # so we create a valid font and monkey-patch the hmtx table.
        _, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 500,
                "space": 250,
                "A": 600,
            }
        )
        tt_font["hmtx"].metrics["A"] = (-100, 0)
        assert _validate_font_program(tt_font) is False
        tt_font.close()

    def test_corrupted_font_not_corrected(self) -> None:
        """Font with all-zero font program doesn't overwrite declared widths."""
        corrupt_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 0,
                "space": 0,
                "A": 0,
                "B": 0,
                "C": 0,
            }
        )
        tt_font.close()

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=[250, *([500] * 32), 600, 650, 700],
            font_data=corrupt_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        # Widths should NOT be corrected since font program is invalid
        assert result["simple_font_widths_fixed"] == 0

    def test_corrupted_cidfont_not_corrected(self) -> None:
        """CIDFont with all-zero font program doesn't overwrite widths."""
        corrupt_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 0,
                "space": 0,
                "A": 0,
                "B": 0,
                "C": 0,
            }
        )
        tt_font.close()

        pdf = new_pdf()
        font = _make_cidfont_with_widths(
            pdf,
            w_array=[0, Array([500, 250, 600, 650, 700])],
            font_data=corrupt_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 0


class TestMismatchRatioThreshold:
    """Tests for mismatch ratio threshold protection."""

    def test_high_mismatch_ratio_still_corrects(self) -> None:
        """Even when >80% of widths mismatch, correction proceeds.

        The font program integrity is validated by _validate_font_program();
        if it passes, widths from the font program are authoritative.
        """
        # Create font with specific widths
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 500,
                "space": 250,
                "A": 600,
                "B": 650,
                "C": 700,
            }
        )
        tt_font.close()

        # Declare widths that are ALL very different from the font program.
        # Codes 32-67: space(32), A(65), B(66), C(67) are mapped.
        # All 4 comparable widths will mismatch → 100% ratio → still corrected.
        num_chars = 67 - 32 + 1
        all_wrong_widths = [999] * num_chars

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=all_wrong_widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_low_mismatch_ratio_allows_correction(self) -> None:
        """When <80% of widths mismatch, correction proceeds normally."""
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 500,
                "space": 250,
                "A": 600,
                "B": 650,
                "C": 700,
            }
        )

        # Compute correct widths, then make only one wrong
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        cmap_table = tt_font.getBestCmap()
        scale = 1000.0 / head.unitsPerEm

        widths = []
        for code in range(32, 68):
            try:
                char = bytes([code]).decode("cp1252")
                unicode_val = ord(char)
            except UnicodeDecodeError:
                widths.append(0)
                continue
            glyph_name = cmap_table.get(unicode_val)
            if glyph_name and glyph_name in hmtx.metrics:
                widths.append(int(hmtx.metrics[glyph_name][0] * scale))
            else:
                notdef_w = hmtx.metrics.get(".notdef", (500, 0))[0]
                widths.append(int(notdef_w * scale))
        tt_font.close()

        # Make only the first width (space) wrong — 1/4 = 25% mismatch
        widths[0] = 999

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_high_mismatch_ratio_cidfont_still_corrects(self) -> None:
        """CIDFont with >80% mismatch is still corrected."""
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 500,
                "space": 250,
                "A": 600,
                "B": 650,
                "C": 700,
            }
        )
        tt_font.close()

        # All CID widths wrong → 100% mismatch → still corrected
        w_array = [0, Array([999, 999, 999, 999, 999])]

        pdf = new_pdf()
        font = _make_cidfont_with_widths(
            pdf,
            w_array=w_array,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 1


def _make_monospace_ttfont(
    *, is_fixed_pitch: bool = True, width: int = 600
) -> tuple[bytes, "TTFont"]:
    """Creates a monospace font where all glyphs share the same advance width.

    Args:
        is_fixed_pitch: Whether to set the post.isFixedPitch flag.
        width: The uniform advance width for all glyphs.

    Returns:
        Tuple of (font_data_bytes, TTFont object).
    """
    glyph_widths = {
        ".notdef": width,
        "space": width,
        "A": width,
        "B": width,
        "C": width,
    }
    font_data, tt_font = _make_minimal_ttfont(glyph_widths=glyph_widths)
    if is_fixed_pitch:
        tt_font["post"].isFixedPitch = 1
    buf = BytesIO()
    tt_font.save(buf)
    tt_font.close()
    buf.seek(0)
    font_data = buf.read()
    tt_font = TTFont(BytesIO(font_data))
    return font_data, tt_font


class TestMonospaceFontWidthCorrection:
    """Tests that monospace/uniform-width fonts bypass the mismatch threshold."""

    def test_monospace_font_corrected_despite_high_mismatch(self) -> None:
        """Monospace font (isFixedPitch=1) with 100% mismatch is corrected."""
        font_data, tt_font = _make_monospace_ttfont(is_fixed_pitch=True)
        tt_font.close()

        # Declare all-wrong variable widths (100% mismatch)
        num_chars = 67 - 32 + 1
        all_wrong_widths = [999] * num_chars

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=all_wrong_widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_uniform_width_font_corrected_despite_high_mismatch(self) -> None:
        """Font with uniform widths but no isFixedPitch is still corrected."""
        font_data, tt_font = _make_monospace_ttfont(is_fixed_pitch=False)
        tt_font.close()

        num_chars = 67 - 32 + 1
        all_wrong_widths = [999] * num_chars

        pdf = new_pdf()
        font = _make_simple_font_with_widths(
            pdf,
            widths=all_wrong_widths,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

    def test_monospace_cidfont_corrected_despite_high_mismatch(self) -> None:
        """Monospace CIDFont with 100% mismatch is corrected."""
        font_data, tt_font = _make_monospace_ttfont(is_fixed_pitch=True)
        tt_font.close()

        # Declare all-wrong CID widths (100% mismatch)
        w_array = [0, Array([999, 999, 999, 999, 999])]

        pdf = new_pdf()
        font = _make_cidfont_with_widths(
            pdf,
            w_array=w_array,
            font_data=font_data,
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["cidfont_widths_fixed"] == 1


def _make_type3_font(
    pdf: Pdf,
    *,
    widths: list[int] | None = None,
    d_widths: list[int] | None = None,
    use_d1: bool = True,
    font_matrix: list[float] | None = None,
) -> Dictionary:
    """Creates a Type3 font with CharProcs containing d0/d1 operators.

    Args:
        pdf: Pdf to create objects in.
        widths: Declared /Widths array values.
        d_widths: Actual widths in d0/d1 operators (glyph-space).
        use_d1: Use d1 operator (with bbox) instead of d0.
        font_matrix: FontMatrix (default [0.001, 0, 0, 0.001, 0, 0]).
    """
    if font_matrix is None:
        font_matrix = [0.001, 0, 0, 0.001, 0, 0]
    if d_widths is None:
        d_widths = [1000, 750]
    if widths is None:
        # Default: wrong widths for testing
        widths = [500, 500]

    # Create CharProc streams
    char_procs = Dictionary()
    glyph_names = ["square", "triangle"]
    for i, gname in enumerate(glyph_names):
        w = d_widths[i]
        if use_d1:
            stream_data = f"{w} 0 0 0 750 750 d1\n0 0 750 750 re f".encode()
        else:
            stream_data = f"{w} 0 d0\n0 0 750 750 re f".encode()
        char_procs[Name(f"/{gname}")] = pdf.make_indirect(pdf.make_stream(stream_data))

    type3_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array(font_matrix),
            CharProcs=char_procs,
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([97, Name("/square"), Name("/triangle")]),
            ),
            FirstChar=97,
            LastChar=98,
            Widths=Array(widths),
        )
    )

    return type3_font


class TestType3FontWidthFix:
    """Tests for Type3 font width correction."""

    def test_type3_d1_wrong_widths_corrected(self) -> None:
        """Type3 font with d1 operator: wrong widths are corrected."""
        pdf = new_pdf()
        # d_widths=[1000, 750] in glyph space, fm_scale=1.0
        # Expected text-space widths: [1000, 750]
        # Declared widths: [500, 500] — wrong
        font = _make_type3_font(pdf, widths=[500, 500], d_widths=[1000, 750])
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["type3_font_widths_fixed"] == 1

        # Verify corrected widths
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        assert corrected == [1000, 750]

    def test_type3_d0_wrong_widths_corrected(self) -> None:
        """Type3 font with d0 operator: wrong widths are corrected."""
        pdf = new_pdf()
        font = _make_type3_font(
            pdf, widths=[500, 500], d_widths=[1000, 750], use_d1=False
        )
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["type3_font_widths_fixed"] == 1

        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        assert corrected == [1000, 750]

    def test_type3_correct_widths_not_modified(self) -> None:
        """Type3 font with correct widths is not modified."""
        pdf = new_pdf()
        font = _make_type3_font(pdf, widths=[1000, 750], d_widths=[1000, 750])
        _build_pdf_with_font(pdf, font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["type3_font_widths_fixed"] == 0

    def test_type3_no_charprocs_skipped(self) -> None:
        """Type3 font without CharProcs is skipped."""
        pdf = new_pdf()

        type3_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                FontBBox=Array([0, 0, 1000, 1000]),
                FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
                FirstChar=97,
                LastChar=98,
                Widths=Array([500, 500]),
            )
        )

        _build_pdf_with_font(pdf, type3_font)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["type3_font_widths_fixed"] == 0


class TestCFFWidthExtraction:
    """Tests for CFF-specific width extraction improvements."""

    def test_cff_font_matrix_scaling(self) -> None:
        """CFF widths are scaled by FontMatrix for non-standard upm."""
        # Create a CFF font with non-standard FontMatrix
        from fontTools.cffLib import (
            CFFFontSet,
            CharStrings,
            IndexedStrings,
            PrivateDict,
            TopDict,
        )
        from fontTools.ttLib import TTFont as FTTTFont
        from fontTools.ttLib.tables.C_F_F_ import table_C_F_F_

        from pdftopdfa.sanitizers.font_widths import _extract_cff_glyph_widths

        tt = FTTTFont()
        cff = table_C_F_F_()
        tt["CFF "] = cff

        fontset = CFFFontSet()
        cff.cff = fontset
        fontset.major = 1
        fontset.minor = 0
        fontset.fontNames = ["TestCFF"]
        fontset.strings = IndexedStrings()

        top_dict = TopDict()
        top_dict.charset = [".notdef", "space", "A"]
        # upm=2048 → FontMatrix = [1/2048, 0, 0, 1/2048, 0, 0]
        top_dict.FontMatrix = [1.0 / 2048, 0, 0, 1.0 / 2048, 0, 0]

        private = PrivateDict()
        private.nominalWidthX = 0
        private.defaultWidthX = 500
        top_dict.Private = private

        # Create charstrings: space=600, A=1400
        # Width encoding: first stack value before hmoveto is width delta
        # when stack has 2 values (hmoveto normally takes 1).
        # Actual width = width_delta + nominalWidthX.
        from fontTools.misc.psCharStrings import T2CharString

        cs_notdef = T2CharString()
        cs_notdef.program = ["endchar"]  # uses defaultWidthX=500

        cs_space = T2CharString()
        cs_space.program = [600, 0, "hmoveto", "endchar"]  # width=600+0=600

        cs_a = T2CharString()
        cs_a.program = [1400, 0, "hmoveto", "endchar"]  # width=1400+0=1400

        char_strings = CharStrings(
            file=None,
            charset=top_dict.charset,
            globalSubrs=[],
            private=private,
            fdSelect=None,
            fdArray=None,
        )
        char_strings[".notdef"] = cs_notdef
        char_strings["space"] = cs_space
        char_strings["A"] = cs_a

        top_dict.CharStrings = char_strings
        fontset.topDictIndex = [top_dict]
        tt.setGlyphOrder([".notdef", "space", "A"])

        result = _extract_cff_glyph_widths(tt)

        # Expected: raw widths scaled by FontMatrix[0] * 1000
        fm_scale = 1000.0 / 2048
        assert result["space"] == round(600 * fm_scale)
        assert result["A"] == round(1400 * fm_scale)
        assert result[".notdef"] == round(500 * fm_scale)
        tt.close()

    def test_cff_standard_font_matrix_no_scaling(self) -> None:
        """CFF with standard FontMatrix [0.001,...] has scale factor 1.0."""
        from fontTools.cffLib import (
            CFFFontSet,
            CharStrings,
            IndexedStrings,
            PrivateDict,
            TopDict,
        )
        from fontTools.ttLib import TTFont as FTTTFont
        from fontTools.ttLib.tables.C_F_F_ import table_C_F_F_

        from pdftopdfa.sanitizers.font_widths import _extract_cff_glyph_widths

        tt = FTTTFont()
        cff = table_C_F_F_()
        tt["CFF "] = cff

        fontset = CFFFontSet()
        cff.cff = fontset
        fontset.major = 1
        fontset.minor = 0
        fontset.fontNames = ["TestCFF"]
        fontset.strings = IndexedStrings()

        top_dict = TopDict()
        top_dict.charset = [".notdef", "space"]
        # Standard FontMatrix (1000 upm)
        top_dict.FontMatrix = [0.001, 0, 0, 0.001, 0, 0]

        private = PrivateDict()
        private.nominalWidthX = 0
        private.defaultWidthX = 500
        top_dict.Private = private

        from fontTools.misc.psCharStrings import T2CharString

        cs_notdef = T2CharString()
        cs_notdef.program = ["endchar"]

        cs_space = T2CharString()
        cs_space.program = [250, 0, "hmoveto", "endchar"]  # width=250+0=250

        char_strings = CharStrings(
            file=None,
            charset=top_dict.charset,
            globalSubrs=[],
            private=private,
            fdSelect=None,
            fdArray=None,
        )
        char_strings[".notdef"] = cs_notdef
        char_strings["space"] = cs_space

        top_dict.CharStrings = char_strings
        fontset.topDictIndex = [top_dict]
        tt.setGlyphOrder([".notdef", "space"])

        result = _extract_cff_glyph_widths(tt)

        # Standard FontMatrix: scale = 0.001 * 1000 = 1.0
        assert result["space"] == 250
        assert result[".notdef"] == 500
        tt.close()


class TestMissingWidthPriority:
    """Tests that .notdef width from font program takes priority over
    /MissingWidth from FontDescriptor for the fallback width.

    veraPDF rule 6.3.6 checks widthFromFontProgram vs widthFromDictionary.
    For glyphs not in the cmap, veraPDF uses the .notdef glyph width as
    widthFromFontProgram — NOT the /MissingWidth from the descriptor.
    """

    def test_notdef_width_used_over_missing_width(self) -> None:
        """Fallback uses .notdef width (750) instead of MissingWidth (1000).

        Simulates a subset TrueType font where code 36 ($) is used but
        the dollar glyph is not in the cmap.  The font's .notdef has
        advance width 750 (1000-unit space) while MissingWidth=1000.
        The sanitizer should correct /Widths[36] to 750, not 1000.
        """
        # Create a font with .notdef=750 (in 1000 upm)
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 750,
                "space": 250,
                "A": 600,
            }
        )
        tt_font.close()

        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
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
                FontFile2=pdf.make_indirect(font_stream),
                MissingWidth=1000,
            )
        )

        # Widths for codes 32-67: space(32)=250, codes 33-64=0, A(65)=600, ...
        # Code 36 ($) is NOT in the font cmap → should use .notdef=750
        widths = [250] + [0] * 32 + [600, 0, 0]  # 32..67

        font_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=font_descriptor,
                FirstChar=32,
                LastChar=67,
                Widths=Array(widths),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        _build_pdf_with_font(pdf, font_dict)
        pdf = _roundtrip(pdf)

        sanitize_font_widths(pdf)

        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        # Code 36 at index 4: should be 750 (.notdef), not 1000 (MissingWidth)
        assert corrected[4] == 750, (
            f"Code 36 width should be 750 (.notdef), got {corrected[4]}"
        )


class TestNameBasedWidthLookup:
    """Tests that the width sanitizer resolves glyphs by name when the
    cmap lookup fails.  This matches veraPDF's behavior for TrueType
    fonts where glyphs exist in the font program (e.g. added by the
    glyph coverage sanitizer) but are not reachable via the cmap.
    """

    def test_glyph_added_by_name_width_used(self) -> None:
        """Glyph present in hmtx by name but not in cmap uses correct width.

        When glyph_coverage adds an empty glyph named 'dollar' with
        advance=0, the width sanitizer should see it via name lookup
        and set /Widths[36] to 0 (matching the font program), instead
        of falling back to the .notdef width.
        """
        # Create a font with dollar glyph in hmtx but NOT in cmap
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 750,
                "space": 250,
                "A": 600,
            }
        )

        # Add a 'dollar' glyph to the font by name (like glyph_coverage does)
        from fontTools.ttLib.tables._g_l_y_f import Glyph as TtGlyph

        glyph_order = tt_font.getGlyphOrder()
        glyph_order.append("dollar")
        tt_font.setGlyphOrder(glyph_order)
        tt_font["glyf"]["dollar"] = TtGlyph()
        tt_font["hmtx"]["dollar"] = (0, 0)  # advance=0
        tt_font["maxp"].numGlyphs = len(glyph_order)

        # Save modified font
        buf = BytesIO()
        tt_font.save(buf)
        tt_font.close()
        buf.seek(0)
        modified_font_data = buf.read()

        pdf = new_pdf()

        font_stream = pdf.make_stream(modified_font_data)
        font_stream[Name.Length1] = len(modified_font_data)

        font_descriptor = pdf.make_indirect(
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
                FontFile2=pdf.make_indirect(font_stream),
                MissingWidth=500,
            )
        )

        # Code 36 ($) has declared width=999 (wrong)
        # Font program has 'dollar' with advance=0
        # Name lookup should find it and correct to 0
        widths = [250] + [0] * 3 + [999] + [0] * 28 + [600, 0, 0]  # 32..67

        font_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=font_descriptor,
                FirstChar=32,
                LastChar=67,
                Widths=Array(widths),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        _build_pdf_with_font(pdf, font_dict)
        pdf = _roundtrip(pdf)

        result = sanitize_font_widths(pdf)

        assert result["simple_font_widths_fixed"] == 1

        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        # Code 36 at index 4: should be 0 (from 'dollar' glyph), not 750
        assert corrected[4] == 0, (
            f"Code 36 width should be 0 (from 'dollar' glyph), got {corrected[4]}"
        )

    def test_glyph_not_in_font_uses_notdef_fallback(self) -> None:
        """Code with no glyph at all (neither cmap nor name) uses .notdef."""
        font_data, tt_font = _make_minimal_ttfont(
            glyph_widths={
                ".notdef": 750,
                "space": 250,
                "A": 600,
            }
        )
        tt_font.close()

        pdf = new_pdf()

        font_stream = pdf.make_stream(font_data)
        font_stream[Name.Length1] = len(font_data)

        font_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/TestFont"),
                Flags=32,
                FontBBox=Array([0, -200, 1000, 800]),
                FontFile2=pdf.make_indirect(font_stream),
            )
        )

        # Code 36 ($) declared width=999 (wrong)
        # No 'dollar' glyph in font → fallback to .notdef=750
        widths = [250] + [0] * 3 + [999] + [0] * 28 + [600, 0, 0]

        font_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.TrueType,
                BaseFont=Name("/TestFont"),
                FontDescriptor=font_descriptor,
                FirstChar=32,
                LastChar=67,
                Widths=Array(widths),
                Encoding=Name.WinAnsiEncoding,
            )
        )

        _build_pdf_with_font(pdf, font_dict)
        pdf = _roundtrip(pdf)

        sanitize_font_widths(pdf)

        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        corrected = [int(w) for w in resolve(font_obj.Widths)]
        # Code 36 at index 4: should be 750 (.notdef fallback)
        assert corrected[4] == 750, (
            f"Code 36 width should be 750 (.notdef), got {corrected[4]}"
        )
