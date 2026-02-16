# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for recursive font traversal and nested font embedding."""

import pikepdf
import pytest
from conftest import new_pdf
from font_helpers import _liberation_fonts_available
from pikepdf import Array, Dictionary, Name

from pdftopdfa.fonts.traversal import iter_all_page_fonts


class TestIterAllPageFonts:
    """Tests for the iter_all_page_fonts generator."""

    def test_page_level_fonts(self):
        """Yields fonts from page-level Resources/Font."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font_dict),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert len(fonts) == 1
        assert fonts[0][0] == "/F1"
        assert str(fonts[0][1].get("/BaseFont")) == "/Helvetica"

    def test_no_resources(self):
        """Handles pages without Resources gracefully."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert fonts == []

    def test_no_fonts_in_resources(self):
        """Handles Resources without Font dictionary gracefully."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert fonts == []

    def test_form_xobject_fonts(self):
        """Yields fonts from Form XObject Resources."""
        pdf = new_pdf()

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Courier"),
        )

        form_xobj = pdf.make_stream(b"BT /F2 10 Tf (Nested) Tj ET")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 200, 200])
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F2=nested_font),
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
                XObject=Dictionary(Form1=pdf.make_indirect(form_xobj)),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET /Form1 Do")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Helvetica" in font_names
        assert "/Courier" in font_names
        assert len(fonts) == 2

    def test_image_xobject_ignored(self):
        """Does not recurse into Image XObjects."""
        pdf = new_pdf()

        image_xobj = pdf.make_stream(b"\x80")
        image_xobj[Name.Type] = Name.XObject
        image_xobj[Name.Subtype] = Name.Image
        image_xobj[Name.Width] = 1
        image_xobj[Name.Height] = 1
        image_xobj[Name.ColorSpace] = Name.DeviceGray
        image_xobj[Name.BitsPerComponent] = 8

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
                XObject=Dictionary(Im0=pdf.make_indirect(image_xobj)),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert len(fonts) == 1
        assert str(fonts[0][1].get("/BaseFont")) == "/Helvetica"

    def test_nested_form_xobjects(self):
        """Yields fonts from nested Form XObjects (recursive)."""
        pdf = new_pdf()

        # Inner Form XObject with its own font
        inner_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Symbol"),
        )

        inner_form = pdf.make_stream(b"BT /F3 10 Tf (inner) Tj ET")
        inner_form[Name.Type] = Name.XObject
        inner_form[Name.Subtype] = Name.Form
        inner_form[Name.BBox] = Array([0, 0, 100, 100])
        inner_form[Name.Resources] = Dictionary(
            Font=Dictionary(F3=inner_font),
        )

        # Outer Form XObject referencing inner
        outer_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Courier"),
        )

        outer_form = pdf.make_stream(b"BT /F2 10 Tf (outer) Tj ET /InnerForm Do")
        outer_form[Name.Type] = Name.XObject
        outer_form[Name.Subtype] = Name.Form
        outer_form[Name.BBox] = Array([0, 0, 200, 200])
        outer_form[Name.Resources] = Dictionary(
            Font=Dictionary(F2=outer_font),
            XObject=Dictionary(InnerForm=pdf.make_indirect(inner_form)),
        )

        # Page-level font
        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
                XObject=Dictionary(OuterForm=pdf.make_indirect(outer_form)),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET /OuterForm Do")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Helvetica" in font_names
        assert "/Courier" in font_names
        assert "/Symbol" in font_names
        assert len(fonts) == 3

    def test_tiling_pattern_fonts(self):
        """Yields fonts from Tiling Pattern Resources."""
        pdf = new_pdf()

        pattern_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Times-Roman"),
        )

        tiling_pattern = pdf.make_stream(b"BT /F2 8 Tf (tile) Tj ET")
        tiling_pattern[Name.Type] = Name.Pattern
        tiling_pattern[Name.PatternType] = 1  # Tiling
        tiling_pattern[Name.PaintType] = 1
        tiling_pattern[Name.TilingType] = 1
        tiling_pattern[Name.BBox] = Array([0, 0, 50, 50])
        tiling_pattern[Name.XStep] = 50
        tiling_pattern[Name.YStep] = 50
        tiling_pattern[Name.Resources] = Dictionary(
            Font=Dictionary(F2=pattern_font),
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
                Pattern=Dictionary(P1=pdf.make_indirect(tiling_pattern)),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Helvetica" in font_names
        assert "/Times-Roman" in font_names
        assert len(fonts) == 2

    def test_shading_pattern_ignored(self):
        """Does not recurse into Shading Patterns (PatternType=2)."""
        pdf = new_pdf()

        # Shading pattern (PatternType=2) — no Resources
        shading_pattern = Dictionary(
            Type=Name.Pattern,
            PatternType=2,
            Shading=Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
                Coords=Array([0, 0, 1, 1]),
            ),
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
                Pattern=Dictionary(P1=pdf.make_indirect(shading_pattern)),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert len(fonts) == 1
        assert str(fonts[0][1].get("/BaseFont")) == "/Helvetica"

    def test_annotation_appearance_stream_fonts(self):
        """Yields fonts from Annotation Appearance Streams."""
        pdf = new_pdf()

        ap_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Courier"),
        )

        ap_stream = pdf.make_stream(b"BT /F2 10 Tf (annot) Tj ET")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 20])
        ap_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F2=ap_font),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FreeText,
                Rect=Array([100, 700, 300, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
            ),
            Annots=Array([annot]),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Helvetica" in font_names
        assert "/Courier" in font_names
        assert len(fonts) == 2

    def test_annotation_ap_sub_state_dict(self):
        """Yields fonts from AP entries that are sub-state dictionaries."""
        pdf = new_pdf()

        ap_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Times-Roman"),
        )

        on_stream = pdf.make_stream(b"BT /F2 10 Tf (on) Tj ET")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 20, 20])
        on_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F2=ap_font),
        )

        off_stream = pdf.make_stream(b"BT /F2 10 Tf (off) Tj ET")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 20, 20])
        # off_stream shares the same font object, no new font

        # N is a dictionary of sub-states (e.g., checkbox Yes/Off)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(
                    N=Dictionary(
                        Yes=on_stream,
                        Off=off_stream,
                    ),
                ),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Times-Roman" in font_names
        # At least the on_stream font should appear
        assert len(fonts) >= 1

    def test_cycle_detection(self):
        """Handles circular references without infinite recursion."""
        pdf = new_pdf()

        # Create a Form XObject that references itself via XObject dict
        form_xobj = pdf.make_stream(b"/SelfRef Do")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])

        form_ref = pdf.make_indirect(form_xobj)

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Courier"),
        )

        # Self-reference: the form's XObject dict points back to itself
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F1=nested_font),
            XObject=Dictionary(SelfRef=form_ref),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                XObject=Dictionary(Form1=form_ref),
            ),
        )

        content_stream = pdf.make_stream(b"/Form1 Do")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # Should not hang or raise — cycle detection prevents infinite recursion
        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}
        assert "/Courier" in font_names

    def test_multiple_ap_keys(self):
        """Yields fonts from N, R, and D appearance streams."""
        pdf = new_pdf()

        def make_ap_stream(font_name):
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name(f"/{font_name}"),
            )
            stream = pdf.make_stream(b"BT /Fx 10 Tf (x) Tj ET")
            stream[Name.Type] = Name.XObject
            stream[Name.Subtype] = Name.Form
            stream[Name.BBox] = Array([0, 0, 50, 50])
            stream[Name.Resources] = Dictionary(
                Font=Dictionary(Fx=font),
            )
            return stream

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FreeText,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(
                    N=make_ap_stream("Helvetica"),
                    R=make_ap_stream("Courier"),
                    D=make_ap_stream("Times-Roman"),
                ),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        font_names = {str(f[1].get("/BaseFont")) for f in fonts}

        assert "/Helvetica" in font_names
        assert "/Courier" in font_names
        assert "/Times-Roman" in font_names

    def test_no_annotations(self):
        """Handles pages without annotations gracefully."""
        pdf = new_pdf()

        page_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=page_font),
            ),
        )

        content_stream = pdf.make_stream(b"BT /F1 12 Tf (Test) Tj ET")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = list(iter_all_page_fonts(pdf.pages[0]))
        assert len(fonts) == 1


class TestNestedFontEmbedding:
    """Integration tests: embed_missing_fonts finds fonts in nested structures."""

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_embed_font_in_form_xobject(self):
        """embed_missing_fonts embeds font found in Form XObject."""
        from pdftopdfa.fonts import FontEmbedder, check_font_compliance

        pdf = new_pdf()

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        form_xobj = pdf.make_stream(b"BT /F1 10 Tf (nested) Tj ET")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 200, 200])
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F1=nested_font),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                XObject=Dictionary(Form1=pdf.make_indirect(form_xobj)),
            ),
        )

        content_stream = pdf.make_stream(b"/Form1 Do")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded

        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_embed_font_in_annotation_ap(self):
        """embed_missing_fonts embeds font found in Annotation AP stream."""
        from pdftopdfa.fonts import FontEmbedder, check_font_compliance

        pdf = new_pdf()

        ap_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        ap_stream = pdf.make_stream(b"BT /F1 10 Tf (annot) Tj ET")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 20])
        ap_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F1=ap_font),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FreeText,
                Rect=Array([100, 700, 300, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded

        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []

    @pytest.mark.skipif(
        not _liberation_fonts_available(),
        reason="Liberation fonts not installed",
    )
    def test_embed_font_in_tiling_pattern(self):
        """embed_missing_fonts embeds font found in Tiling Pattern."""
        from pdftopdfa.fonts import FontEmbedder, check_font_compliance

        pdf = new_pdf()

        pattern_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        tiling_pattern = pdf.make_stream(b"BT /F1 8 Tf (tile) Tj ET")
        tiling_pattern[Name.Type] = Name.Pattern
        tiling_pattern[Name.PatternType] = 1
        tiling_pattern[Name.PaintType] = 1
        tiling_pattern[Name.TilingType] = 1
        tiling_pattern[Name.BBox] = Array([0, 0, 50, 50])
        tiling_pattern[Name.XStep] = 50
        tiling_pattern[Name.YStep] = 50
        tiling_pattern[Name.Resources] = Dictionary(
            Font=Dictionary(F1=pattern_font),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Pattern=Dictionary(P1=pdf.make_indirect(tiling_pattern)),
            ),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedder = FontEmbedder(pdf)
        result = embedder.embed_missing_fonts()

        assert "Helvetica" in result.fonts_embedded

        is_compliant, missing = check_font_compliance(pdf, raise_on_error=False)
        assert is_compliant
        assert missing == []


class TestNestedFontAnalysis:
    """Integration tests: analyze_fonts discovers fonts in nested structures."""

    def test_analyze_finds_form_xobject_font(self):
        """analyze_fonts discovers font inside Form XObject."""
        from pdftopdfa.fonts.analysis import analyze_fonts

        pdf = new_pdf()

        nested_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Courier"),
        )

        form_xobj = pdf.make_stream(b"BT /F1 10 Tf (nested) Tj ET")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 200, 200])
        form_xobj[Name.Resources] = Dictionary(
            Font=Dictionary(F1=nested_font),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                XObject=Dictionary(Form1=pdf.make_indirect(form_xobj)),
            ),
        )

        content_stream = pdf.make_stream(b"/Form1 Do")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = analyze_fonts(pdf)
        font_names = [f.name for f in fonts]
        assert "Courier" in font_names

    def test_analyze_finds_annotation_ap_font(self):
        """analyze_fonts discovers font inside Annotation AP stream."""
        from pdftopdfa.fonts.analysis import analyze_fonts

        pdf = new_pdf()

        ap_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Times-Roman"),
        )

        ap_stream = pdf.make_stream(b"BT /F1 10 Tf (annot) Tj ET")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 20])
        ap_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F1=ap_font),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FreeText,
                Rect=Array([100, 700, 300, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = analyze_fonts(pdf)
        font_names = [f.name for f in fonts]
        assert "Times-Roman" in font_names

    def test_analyze_finds_tiling_pattern_font(self):
        """analyze_fonts discovers font inside Tiling Pattern."""
        from pdftopdfa.fonts.analysis import analyze_fonts

        pdf = new_pdf()

        pattern_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Symbol"),
        )

        tiling_pattern = pdf.make_stream(b"BT /F1 8 Tf (tile) Tj ET")
        tiling_pattern[Name.Type] = Name.Pattern
        tiling_pattern[Name.PatternType] = 1
        tiling_pattern[Name.PaintType] = 1
        tiling_pattern[Name.TilingType] = 1
        tiling_pattern[Name.BBox] = Array([0, 0, 50, 50])
        tiling_pattern[Name.XStep] = 50
        tiling_pattern[Name.YStep] = 50
        tiling_pattern[Name.Resources] = Dictionary(
            Font=Dictionary(F1=pattern_font),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Pattern=Dictionary(P1=pdf.make_indirect(tiling_pattern)),
            ),
        )

        content_stream = pdf.make_stream(b"")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        fonts = analyze_fonts(pdf)
        font_names = [f.name for f in fonts]
        assert "Symbol" in font_names
