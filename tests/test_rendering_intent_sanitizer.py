# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for rendering intent validation for PDF/A compliance."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.extgstate import sanitize_extgstate
from pdftopdfa.sanitizers.rendering_intent import (
    sanitize_rendering_intent,
)


def _make_pdf_with_extgstate(pdf: Pdf, gs_dict: Dictionary) -> None:
    """Helper: add a page with an ExtGState resource to a PDF."""
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ExtGState=Dictionary(GS0=gs_dict),
            ),
        )
    )
    pdf.pages.append(page)


def _make_pdf_with_content_stream(pdf: Pdf, content: bytes) -> None:
    """Helper: add a page with the given content stream bytes."""
    stream = pdf.make_stream(content)
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=stream,
        )
    )
    pdf.pages.append(page)


# --- ExtGState /RI validation ---


class TestExtGStateRIValidation:
    """Tests for /RI key validation in ExtGState dictionaries."""

    @pytest.mark.parametrize(
        "intent",
        [
            Name.RelativeColorimetric,
            Name.AbsoluteColorimetric,
            Name.Perceptual,
            Name.Saturation,
        ],
    )
    def test_valid_ri_preserved(self, intent: Name):
        """Valid rendering intent values in ExtGState are preserved."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, RI=intent)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 0
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert "/RI" in gs_out
        assert str(gs_out.RI) == str(intent)

    def test_invalid_ri_replaced(self):
        """Invalid /RI value is replaced with /RelativeColorimetric."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, RI=Name("/FooBar"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert str(gs_out.RI) == "/RelativeColorimetric"

    def test_invalid_ri_combined_with_other_fixes(self):
        """Invalid /RI is counted together with other ExtGState fixes."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(
            Type=Name.ExtGState,
            TR=tr_stream,
            RI=Name("/BadIntent"),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 2
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert "/TR" not in gs_out
        assert str(gs_out.RI) == "/RelativeColorimetric"


# --- Content stream ri operator ---


class TestContentStreamRiOperator:
    """Tests for ri operator in page content streams."""

    def test_valid_ri_unchanged(self):
        """Valid ri operator is not modified."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"/Perceptual ri")

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 0
        # Verify the stream still has /Perceptual
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        assert len(instructions) == 1
        assert str(instructions[0].operands[0]) == "/Perceptual"

    def test_invalid_ri_replaced(self):
        """Invalid ri operand is replaced with /RelativeColorimetric."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"/FooBar ri")

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        ri_ops = [i for i in instructions if str(i.operator) == "ri"]
        assert len(ri_ops) == 1
        assert str(ri_ops[0].operands[0]) == "/RelativeColorimetric"

    def test_multiple_ri_operators(self):
        """Multiple invalid ri operators are all fixed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"/Bad1 ri /Bad2 ri /Perceptual ri")

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 2
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        ri_ops = [i for i in instructions if str(i.operator) == "ri"]
        assert len(ri_ops) == 3
        assert str(ri_ops[0].operands[0]) == "/RelativeColorimetric"
        assert str(ri_ops[1].operands[0]) == "/RelativeColorimetric"
        assert str(ri_ops[2].operands[0]) == "/Perceptual"

    def test_ri_mixed_with_other_operators(self):
        """ri operators are fixed while other operators are preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"q /BadIntent ri 1 0 0 1 0 0 cm Q")

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        operators = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "q" in operators
        assert "ri" in operators
        assert "cm" in operators
        assert "Q" in operators


# --- Form XObject ri operator ---


class TestFormXObjectRiOperator:
    """Tests for ri operator in Form XObject content streams."""

    def test_ri_in_form_xobject(self):
        """Fixes ri operator inside a Form XObject."""
        pdf = new_pdf()

        form_stream = pdf.make_stream(b"/InvalidRI ri")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(form_stream))
        ri_ops = [i for i in instructions if str(i.operator) == "ri"]
        assert str(ri_ops[0].operands[0]) == "/RelativeColorimetric"

    def test_ri_in_nested_form_xobjects(self):
        """Fixes ri operator in nested Form XObjects."""
        pdf = new_pdf()

        # Inner form with invalid ri
        inner_form = pdf.make_stream(b"/BadNested ri")
        inner_form[Name.Type] = Name.XObject
        inner_form[Name.Subtype] = Name.Form
        inner_form[Name.BBox] = Array([0, 0, 50, 50])

        # Outer form referencing inner form
        outer_form = pdf.make_stream(b"/BadOuter ri /InnerForm Do")
        outer_form[Name.Type] = Name.XObject
        outer_form[Name.Subtype] = Name.Form
        outer_form[Name.BBox] = Array([0, 0, 100, 100])
        outer_form[Name.Resources] = Dictionary(
            XObject=Dictionary(InnerForm=inner_form),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(OuterForm=outer_form),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 2


# --- Annotation AP stream ri operator ---


class TestAnnotationAPStreamRi:
    """Tests for ri operator in annotation appearance streams."""

    def test_ri_in_ap_stream(self):
        """Fixes ri operator in annotation AP stream."""
        pdf = new_pdf()

        ap_stream = pdf.make_stream(b"/InvalidAP ri")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 20, 20])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.pages[0].Annots = Array([annot])

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        instructions = list(pikepdf.parse_content_stream(ap_stream))
        ri_ops = [i for i in instructions if str(i.operator) == "ri"]
        assert str(ri_ops[0].operands[0]) == "/RelativeColorimetric"

    def test_ri_in_ap_substate_dict(self):
        """Fixes ri operator in AP sub-state dictionary streams."""
        pdf = new_pdf()

        on_stream = pdf.make_stream(b"/BadOn ri")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 20, 20])

        off_stream = pdf.make_stream(b"/BadOff ri")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 20, 20])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(
                    N=Dictionary(On=on_stream, Off=off_stream),
                ),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.pages[0].Annots = Array([annot])

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 2


# --- Contents as array ---


class TestContentsArray:
    """Tests for page Contents as an array of streams."""

    def test_contents_array_of_streams(self):
        """Fixes ri operators in Contents that is an array of streams."""
        pdf = new_pdf()

        stream1 = pdf.make_stream(b"/BadIntent1 ri")
        stream2 = pdf.make_stream(b"/Perceptual ri")
        stream3 = pdf.make_stream(b"/BadIntent2 ri")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=Array([stream1, stream2, stream3]),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 2


# --- No changes needed ---


class TestNoChangesNeeded:
    """Tests for PDFs that don't need rendering intent changes."""

    def test_empty_pdf(self, sample_pdf_obj: Pdf):
        """PDF without ri operators returns zero count."""
        result = sanitize_rendering_intent(sample_pdf_obj)

        assert result["ri_operators_fixed"] == 0

    def test_page_without_contents(self):
        """PDF page without Contents returns zero count."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 0

    def test_content_stream_without_ri(self):
        """Content stream without ri operators returns zero count."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"q 1 0 0 1 0 0 cm Q")

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 0


# --- Undefined operators + resources ---


class TestUndefinedOperatorsAndResources:
    """Tests for rule 6.2.2 operator/resources sanitization."""

    def test_undefined_operator_removed_in_page_content(self):
        """Unknown operators are removed from page content streams."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(
            pdf, b"q 1 0 0 1 0 0 cm /Foo 12 UnknownOperator Q"
        )

        result = sanitize_rendering_intent(pdf)

        assert result["undefined_operators_removed"] == 1
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        operators = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "UnknownOperator" not in operators
        assert "q" in operators
        assert "cm" in operators
        assert "Q" in operators

    def test_form_resources_added_from_parent(self):
        """Missing Form /Resources are added and seeded from parent resources."""
        pdf = new_pdf()

        form_stream = pdf.make_stream(b"/CS0 cs")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ColorSpace=Dictionary(CS0=Name.DeviceRGB),
                    XObject=Dictionary(X0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["resources_dictionaries_added"] >= 1
        form_resources = form_stream.get("/Resources")
        assert isinstance(form_resources, Dictionary)
        assert "/ColorSpace" in form_resources
        assert "/CS0" in form_resources.ColorSpace

    def test_form_resources_entries_merged_from_parent(self):
        """Existing Form /Resources are merged with missing inherited names."""
        pdf = new_pdf()

        form_stream = pdf.make_stream(b"/CS0 cs")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary()

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ColorSpace=Dictionary(CS0=Name.DeviceRGB),
                    XObject=Dictionary(X0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["resources_entries_merged"] >= 1
        form_resources = form_stream.get("/Resources")
        assert isinstance(form_resources, Dictionary)
        assert "/ColorSpace" in form_resources
        assert "/CS0" in form_resources.ColorSpace

    def test_type3_resources_added_from_parent(self):
        """Missing Type3 font /Resources are added from parent resources."""
        pdf = new_pdf()

        charproc_stream = pdf.make_stream(b"/CS0 cs")
        type3_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(a=charproc_stream),
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([0, Name.a]),
            ),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ColorSpace=Dictionary(CS0=Name.DeviceRGB),
                    Font=Dictionary(F1=type3_font),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["resources_dictionaries_added"] >= 1
        font_resources = type3_font.get("/Resources")
        assert isinstance(font_resources, Dictionary)
        assert "/ColorSpace" in font_resources
        assert "/CS0" in font_resources.ColorSpace


# --- Image XObject /Intent ---


class TestImageXObjectIntent:
    """Tests for /Intent key on Image XObjects."""

    def test_valid_intent_preserved(self):
        """Valid /Intent on Image XObject is not modified."""
        pdf = new_pdf()
        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.ColorSpace] = Name.DeviceRGB
        img[Name.BitsPerComponent] = 8
        img[Name.Intent] = Name.Perceptual

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Im0=img)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 0
        assert str(img.Intent) == "/Perceptual"

    def test_invalid_intent_replaced(self):
        """Invalid /Intent on Image XObject is replaced."""
        pdf = new_pdf()
        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.ColorSpace] = Name.DeviceRGB
        img[Name.BitsPerComponent] = 8
        img[Name.Intent] = Name("/Custom")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Im0=img)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 1
        assert str(img.Intent) == "/RelativeColorimetric"

    def test_multiple_invalid_intents(self):
        """Multiple Image XObjects with invalid /Intent are all fixed."""
        pdf = new_pdf()

        img1 = pdf.make_stream(b"\xff\x00\x00")
        img1[Name.Type] = Name.XObject
        img1[Name.Subtype] = Name.Image
        img1[Name.Width] = 1
        img1[Name.Height] = 1
        img1[Name.ColorSpace] = Name.DeviceRGB
        img1[Name.BitsPerComponent] = 8
        img1[Name.Intent] = Name("/unknown")

        img2 = pdf.make_stream(b"\x00\xff\x00")
        img2[Name.Type] = Name.XObject
        img2[Name.Subtype] = Name.Image
        img2[Name.Width] = 1
        img2[Name.Height] = 1
        img2[Name.ColorSpace] = Name.DeviceRGB
        img2[Name.BitsPerComponent] = 8
        img2[Name.Intent] = Name("/Custom")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Im0=img1, Im1=img2)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 2
        assert str(img1.Intent) == "/RelativeColorimetric"
        assert str(img2.Intent) == "/RelativeColorimetric"

    def test_image_without_intent_unchanged(self):
        """Image XObject without /Intent is not modified."""
        pdf = new_pdf()
        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.ColorSpace] = Name.DeviceRGB
        img[Name.BitsPerComponent] = 8

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Im0=img)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 0
        assert "/Intent" not in img

    def test_image_in_form_xobject(self):
        """Invalid /Intent on Image inside a Form XObject is fixed."""
        pdf = new_pdf()

        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.ColorSpace] = Name.DeviceRGB
        img[Name.BitsPerComponent] = 8
        img[Name.Intent] = Name("/BadIntent")

        form = pdf.make_stream(b"/Im0 Do")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=img),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(F0=form)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 1
        assert str(img.Intent) == "/RelativeColorimetric"


# --- Inline image /Intent ---


class TestInlineImageIntent:
    """Tests for /Intent in inline images (BI...ID...EI)."""

    def test_invalid_inline_intent_replaced(self):
        """Invalid /Intent in inline image is replaced."""
        pdf = new_pdf()
        # Build a content stream with an inline image containing invalid /Intent
        content = (
            b"BI\n/W 1 /H 1 /CS /RGB /BPC 8 /Intent /Custom\nID\n\xff\x00\x00\nEI\n"
        )
        _make_pdf_with_content_stream(pdf, content)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] >= 1
        # Verify the content stream was rewritten
        data = pdf.pages[0].Contents.read_bytes()
        assert b"/Intent /RelativeColorimetric" in data
        assert b"/Intent /Custom" not in data

    def test_valid_inline_intent_preserved(self):
        """Valid /Intent in inline image is not modified."""
        pdf = new_pdf()
        content = (
            b"BI\n/W 1 /H 1 /CS /RGB /BPC 8 /Intent /Perceptual\nID\n\xff\x00\x00\nEI\n"
        )
        _make_pdf_with_content_stream(pdf, content)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 0
        data = pdf.pages[0].Contents.read_bytes()
        assert b"/Intent /Perceptual" in data

    def test_inline_image_without_intent_unchanged(self):
        """Inline image without /Intent is not modified."""
        pdf = new_pdf()
        content = b"BI\n/W 1 /H 1 /CS /RGB /BPC 8\nID\n\xff\x00\x00\nEI\n"
        _make_pdf_with_content_stream(pdf, content)

        result = sanitize_rendering_intent(pdf)

        assert result["image_intents_fixed"] == 0

    def test_inline_image_combined_with_ri_operator(self):
        """Both inline image /Intent and ri operator are fixed."""
        pdf = new_pdf()
        content = (
            b"/BadRI ri\n"
            b"BI\n"
            b"/W 1 /H 1 /CS /RGB /BPC 8 /Intent /Custom\n"
            b"ID\n"
            b"\xff\x00\x00"
            b"\nEI\n"
        )
        _make_pdf_with_content_stream(pdf, content)

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        assert result["image_intents_fixed"] >= 1
        data = pdf.pages[0].Contents.read_bytes()
        assert b"/Intent /RelativeColorimetric" in data
        assert b"/Intent /Custom" not in data


# --- Integration ---


class TestIntegration:
    """Integration tests with sanitize_for_pdfa."""

    def test_sanitize_for_pdfa_includes_ri_key(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa returns ri_operators_fixed key."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "ri_operators_fixed" in result
        assert result["ri_operators_fixed"] == 0

    def test_sanitize_for_pdfa_includes_content_stream_622_keys(self):
        """sanitize_for_pdfa returns additional 6.2.2-related counters."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"/BadOp 1 UnknownOperator")

        result = sanitize_for_pdfa(pdf, "3b")

        assert "undefined_operators_removed" in result
        assert "resources_dictionaries_added" in result
        assert "resources_entries_merged" in result
        assert result["undefined_operators_removed"] == 1

    def test_sanitize_for_pdfa_fixes_ri(self):
        """sanitize_for_pdfa actually fixes invalid ri operators."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"/InvalidIntent ri")

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["ri_operators_fixed"] == 1
        contents = pdf.pages[0].Contents
        instructions = list(pikepdf.parse_content_stream(contents))
        ri_ops = [i for i in instructions if str(i.operator) == "ri"]
        assert str(ri_ops[0].operands[0]) == "/RelativeColorimetric"

    def test_sanitize_for_pdfa_fixes_extgstate_ri(self):
        """sanitize_for_pdfa fixes invalid /RI in ExtGState."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, RI=Name("/BadRI"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["extgstate_fixed"] == 1
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert str(gs_out.RI) == "/RelativeColorimetric"

    def test_sanitize_for_pdfa_includes_image_intents_key(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa returns image_intents_fixed key."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "image_intents_fixed" in result
        assert result["image_intents_fixed"] == 0

    def test_sanitize_for_pdfa_fixes_image_xobject_intent(self):
        """sanitize_for_pdfa fixes invalid /Intent on Image XObjects."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.ColorSpace] = Name.DeviceRGB
        img[Name.BitsPerComponent] = 8
        img[Name.Intent] = Name("/Custom")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Im0=img)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["image_intents_fixed"] >= 1
        assert str(img.Intent) == "/RelativeColorimetric"

    def test_sanitize_for_pdfa_fixes_inline_image_intent(self):
        """sanitize_for_pdfa fixes invalid /Intent in inline images."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        content = (
            b"BI\n/W 1 /H 1 /CS /RGB /BPC 8 /Intent /Custom\nID\n\xff\x00\x00\nEI\n"
        )
        _make_pdf_with_content_stream(pdf, content)

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["image_intents_fixed"] >= 1
        data = pdf.pages[0].Contents.read_bytes()
        assert b"/Intent /Custom" not in data


# --- Operator argument count validation ---


class TestOperatorArgCounts:
    """Tests for content stream operator argument count validation."""

    def test_valid_m_operator_preserved(self):
        """moveto with 2 numeric operands is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"100 200 m")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        ops = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "m" in ops

    def test_m_with_wrong_count_removed(self):
        """moveto with 3 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"100 200 300 m")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        ops = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "m" not in ops

    def test_l_with_wrong_count_removed(self):
        """lineto with 1 operand is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"100 200 m 300 l")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        ops = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "m" in ops
        assert "l" not in ops

    def test_re_valid_preserved(self):
        """rectangle with 4 operands is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"10 20 100 50 re")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_re_wrong_count_removed(self):
        """rectangle with 3 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"10 20 100 re")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_rg_valid_preserved(self):
        """setrgbcolor with 3 operands is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"1 0 0 rg")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_rg_wrong_count_removed(self):
        """setrgbcolor with 2 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"1 0 rg")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_rg_stroking_wrong_count_removed(self):
        """stroking setrgbcolor with 4 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"1 0 0 1 RG")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_k_valid_preserved(self):
        """setcmykcolor with 4 operands is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"0 0 0 1 k")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_k_wrong_count_removed(self):
        """setcmykcolor with 3 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"0 0 1 k")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_k_stroking_wrong_count_removed(self):
        """stroking setcmykcolor with 5 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"0 0 0 1 1 K")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_g_valid_preserved(self):
        """setgraycolor with 1 operand is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"0.5 g")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_g_wrong_count_removed(self):
        """setgraycolor with 2 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"0.5 0.5 g")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_g_stroking_wrong_count_removed(self):
        """stroking setgraycolor with 0 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"G")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_cm_valid_preserved(self):
        """concat matrix with 6 operands is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"1 0 0 1 0 0 cm")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_cm_wrong_count_removed(self):
        """concat matrix with 4 operands is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"1 0 0 1 cm")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_d_valid_preserved(self):
        """setdash with array + number is preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"[ 3 ] 0 d")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_d_wrong_count_removed(self):
        """setdash with only 1 operand is removed."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"[ 3 ] d")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_multiple_bad_operators_counted(self):
        """Multiple operators with wrong arg counts are all counted."""
        pdf = new_pdf()
        # m needs 2 args (has 3), rg needs 3 args (has 2)
        _make_pdf_with_content_stream(pdf, b"100 200 300 m 1 0 rg")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 2

    def test_bad_args_mixed_with_valid(self):
        """Bad arg operators removed while valid operators preserved."""
        pdf = new_pdf()
        _make_pdf_with_content_stream(pdf, b"q 100 200 300 m 1 0 0 1 0 0 cm Q")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        ops = [
            str(i.operator)
            for i in instructions
            if isinstance(i, pikepdf.ContentStreamInstruction)
        ]
        assert "q" in ops
        assert "cm" in ops
        assert "Q" in ops
        assert "m" not in ops

    def test_unchecked_operators_not_affected(self):
        """Operators without arg count rules are not removed."""
        pdf = new_pdf()
        # q/Q don't have arg count validation
        _make_pdf_with_content_stream(pdf, b"q Q")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 0

    def test_bad_args_in_form_xobject(self):
        """Bad arg operators in Form XObjects are also removed."""
        pdf = new_pdf()

        form_stream = pdf.make_stream(b"100 200 300 m")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(Form0=form_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1

    def test_non_numeric_args_detected(self):
        """Non-numeric operands for numeric operator are rejected."""
        pdf = new_pdf()
        # "m" expects 2 numeric args, giving it name args
        _make_pdf_with_content_stream(pdf, b"/Foo /Bar m")

        result = sanitize_rendering_intent(pdf)
        assert result["bad_args_operators_removed"] == 1
