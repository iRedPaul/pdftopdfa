# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for rendering intent validation for PDF/A compliance."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

import pdftopdfa.sanitizers.rendering_intent as rendering_intent
from pdftopdfa.fonts.glyph_usage import find_ambiguous_resource_context_streams
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

    def test_shared_resourceless_form_is_cloned_per_page_context(self):
        """A reused Form is relinked to resources from each calling page."""
        pdf = new_pdf()
        font_a = pdf.make_indirect(
            Dictionary(Type=Name.Font, Subtype=Name.Type3, Name=Name("/A"))
        )
        font_b = pdf.make_indirect(
            Dictionary(Type=Name.Font, Subtype=Name.Type3, Name=Name("/B"))
        )
        form = pdf.make_stream(b"BT /F1 12 Tf (A) Tj ET")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])

        for font in (font_a, font_b):
            page = pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 612, 792]),
                    Resources=Dictionary(
                        Font=Dictionary(F1=font),
                        XObject=Dictionary(Fm=form),
                    ),
                    Contents=pdf.make_stream(b"/Fm Do"),
                )
            )
            pdf.pages.append(page)

        sanitize_rendering_intent(pdf)

        forms = [pdf.pages[index].Resources.XObject.Fm for index in range(2)]
        assert forms[0].objgen != forms[1].objgen
        assert forms[0].Resources.Font.F1.objgen == font_a.objgen
        assert forms[1].Resources.Font.F1.objgen == font_b.objgen

    @pytest.mark.parametrize("cycle_length", [1, 2])
    @pytest.mark.parametrize("indirect_resources", [False, True])
    def test_shared_cyclic_form_resources_do_not_clone_indefinitely(
        self, monkeypatch, cycle_length: int, indirect_resources: bool
    ):
        """A stamp reusing a cyclic Form graph must finish without losing resources."""
        pdf = new_pdf()
        forms = [pdf.make_stream(b"/CS0 cs /Bad ri") for _ in range(cycle_length)]
        for index, form in enumerate(forms):
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = Array([0, 0, 100, 100])
            resources = Dictionary(
                XObject=Dictionary(Next=forms[(index + 1) % cycle_length])
            )
            form[Name.Resources] = (
                pdf.make_indirect(resources) if indirect_resources else resources
            )

        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(
            ColorSpace=Dictionary(CS0=Name.DeviceRGB),
            XObject=Dictionary(Fm=forms[0]),
        )
        page.Contents = pdf.make_stream(b"/Fm Do")
        appearance = pdf.make_stream(b"/Fm Do")
        appearance[Name.Type] = Name.XObject
        appearance[Name.Subtype] = Name.Form
        appearance[Name.BBox] = Array([0, 0, 100, 100])
        appearance[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=Name.DeviceGray),
            XObject=Dictionary(Fm=forms[0]),
        )
        page.Annots = Array(
            [
                pdf.make_indirect(
                    Dictionary(
                        Type=Name.Annot,
                        Subtype=Name.Stamp,
                        Rect=Array([0, 0, 100, 100]),
                        AP=Dictionary(N=appearance),
                    )
                )
            ]
        )

        clone_stream = rendering_intent._clone_stream
        clone_count = 0

        def bounded_clone(pdf, source):
            nonlocal clone_count
            clone_count += 1
            assert clone_count <= 2 * cycle_length, "Resource cycle cloned repeatedly"
            return clone_stream(pdf, source)

        monkeypatch.setattr(rendering_intent, "_clone_stream", bounded_clone)
        result = sanitize_rendering_intent(pdf)

        original = page.Resources.XObject.Fm
        cloned = page.Annots[0].AP.N.Resources.XObject.Fm
        assert original.objgen != cloned.objgen
        assert original.Resources.ColorSpace.CS0 == Name.DeviceRGB
        assert cloned.Resources.ColorSpace.CS0 == Name.DeviceGray
        assert result["ri_operators_fixed"] == cycle_length + clone_count
        assert len(page.Annots) == 1

    @pytest.mark.parametrize("indirect_resources", [False, True])
    def test_1200_nested_forms_are_sanitized_without_recursion(
        self, indirect_resources: bool
    ):
        """Resource materialization and operator repair reach a deep leaf."""
        pdf = new_pdf()
        root = pdf.make_stream(b"/Bad ri")
        root[Name.Type] = Name.XObject
        root[Name.Subtype] = Name.Form
        root[Name.BBox] = Array([0, 0, 1, 1])
        leaf_resources = Dictionary()
        root[Name.Resources] = (
            pdf.make_indirect(leaf_resources) if indirect_resources else leaf_resources
        )

        for _ in range(1200):
            form = pdf.make_stream(b"/Next Do")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = Array([0, 0, 1, 1])
            resources = Dictionary(XObject=Dictionary(Next=root))
            form[Name.Resources] = (
                pdf.make_indirect(resources) if indirect_resources else resources
            )
            root = form

        page = pdf.add_blank_page(page_size=(10, 10))
        page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=root))
        page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1

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

    @pytest.mark.parametrize("also_in_fonts", [False, True])
    def test_extgstate_type3_charprocs_sanitized(self, also_in_fonts):
        pdf = new_pdf()
        charproc = pdf.make_stream(b"0 0 d0 /Bad ri 42 UnknownOperator /CS0 cs")
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                CharProcs=Dictionary(a=charproc),
            )
        )
        _make_pdf_with_extgstate(pdf, Dictionary(Font=Array([font, 12])))
        resources = pdf.pages[0].Resources
        resources.ColorSpace = Dictionary(CS0=Name.DeviceRGB)
        if also_in_fonts:
            resources.Font = Dictionary(F1=font)

        result = sanitize_rendering_intent(pdf)

        assert result["ri_operators_fixed"] == 1
        assert result["undefined_operators_removed"] == 1
        instructions = pikepdf.parse_content_stream(charproc)
        assert [str(i.operator) for i in instructions] == ["d0", "ri", "cs"]
        assert instructions[1].operands[0] == Name.RelativeColorimetric
        assert font.Resources.ColorSpace.CS0 == Name.DeviceRGB

    def test_equal_direct_type3_fonts_are_both_materialized(self):
        """Equal direct dictionaries are distinct mutable resource owners."""
        pdf = new_pdf()
        charproc_stream = pdf.make_stream(b"/CS0 cs")

        def make_font() -> Dictionary:
            return Dictionary(
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

        first_font = make_font()
        second_font = make_font()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ColorSpace=Dictionary(CS0=Name.DeviceRGB),
                    Font=Dictionary(F1=first_font, F2=second_font),
                ),
            )
        )
        pdf.pages.append(page)

        sanitize_rendering_intent(pdf)

        for font in (first_font, second_font):
            resources = font.get("/Resources")
            assert isinstance(resources, Dictionary)
            assert resources.ColorSpace.CS0 == Name.DeviceRGB


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


def _resourceless_form(pdf: Pdf, body: bytes = b"/CS0 cs 0 0 m 1 1 l S"):
    form = pdf.make_stream(body)
    form[Name.Type] = Name.XObject
    form[Name.Subtype] = Name.Form
    form[Name.BBox] = Array([0, 0, 1, 1])
    return form


def _count_clones(monkeypatch, limit: int) -> list[int]:
    clone_stream = rendering_intent._clone_stream
    count = [0]

    def bounded_clone(pdf, source):
        count[0] += 1
        assert count[0] <= limit, "context key drifted: streams re-cloned"
        return clone_stream(pdf, source)

    monkeypatch.setattr(rendering_intent, "_clone_stream", bounded_clone)
    return count


class TestResourceContextKeyStability:
    """Clones written into a direct /Resources must not change its context."""

    @pytest.mark.parametrize("form_count", [2, 8, 25])
    def test_sibling_forms_in_direct_resources_cloned_once(
        self, monkeypatch, form_count: int
    ):
        pdf = new_pdf()
        forms = {f"/F{i}": _resourceless_form(pdf) for i in range(form_count)}
        for resources in (
            pdf.make_indirect(Dictionary(XObject=Dictionary(forms))),
            Dictionary(XObject=Dictionary(forms)),
        ):
            page = pdf.add_blank_page(page_size=(10, 10))
            page.Resources = resources
        count = _count_clones(monkeypatch, form_count)

        assert rendering_intent._clone_resource_context_streams(pdf) == form_count
        assert count[0] == form_count
        first, second = (p.Resources.XObject for p in pdf.pages)
        for name in forms:
            assert first[name].objgen != second[name].objgen

    def test_identical_direct_page_resources_share_a_context(self, monkeypatch):
        """Equal direct dictionaries stay one context, as before (no bloat)."""
        pdf = new_pdf()
        logo = _resourceless_form(pdf)
        for _ in range(20):
            page = pdf.add_blank_page(page_size=(10, 10))
            page.Resources = Dictionary(XObject=Dictionary(Logo=logo))
        _count_clones(monkeypatch, 0)

        assert rendering_intent._clone_resource_context_streams(pdf) == 0

    def test_direct_inherited_resources_are_one_context(self, monkeypatch):
        pdf = new_pdf()
        forms = {f"/F{i}": _resourceless_form(pdf) for i in range(4)}
        pdf.add_blank_page(page_size=(10, 10))
        pdf.add_blank_page(page_size=(10, 10))
        for page in pdf.pages:
            del page.obj["/Resources"]
        third = pdf.add_blank_page(page_size=(10, 10))
        third.Resources = pdf.make_indirect(Dictionary(XObject=Dictionary(forms)))
        pdf.Root.Pages.Resources = Dictionary(XObject=Dictionary(forms))
        _count_clones(monkeypatch, 4)

        sanitize_rendering_intent(pdf)

        assert pdf.Root.Pages.Resources.is_indirect


def _type3_font_drawing_form(pdf: Pdf, form: pikepdf.Stream, resources=None):
    """Type3 font whose single glyph draws ``/Fx`` from its context."""
    charproc = pdf.make_stream(b"0 0 d0 /CS0 cs /Fx Do")
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type3,
        FontBBox=Array([0, 0, 1, 1]),
        FontMatrix=Array([1, 0, 0, 1, 0, 0]),
        CharProcs=Dictionary(a=charproc),
        Encoding=Dictionary(Differences=Array([97, Name.a])),
        FirstChar=97,
        LastChar=97,
        Widths=Array([1]),
    )
    if resources is not None:
        font[Name.Resources] = resources
    return pdf.make_indirect(font)


def _page_with(pdf: Pdf, resources: Dictionary) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=(10, 10))
    page.Resources = resources
    page.Contents = pdf.make_stream(b"BT /T 1 Tf (a) Tj ET")
    return page


class TestType3CharProcsPerContext:
    """A Type3 font without /Resources is copied per calling context."""

    def test_shared_font_is_copied_so_charprocs_get_their_context(self):
        pdf = new_pdf()
        form = _resourceless_form(pdf)
        font = _type3_font_drawing_form(pdf, form)
        for space in (Name.DeviceRGB, Name.DeviceGray):
            _page_with(
                pdf,
                Dictionary(
                    ColorSpace=Dictionary(CS0=space),
                    Font=Dictionary(T=font),
                    XObject=Dictionary(Fx=form),
                ),
            )

        sanitize_rendering_intent(pdf)  # previously raised ConversionError

        fonts = [page.Resources.Font.T for page in pdf.pages]
        assert fonts[0].objgen == font.objgen
        assert fonts[1].objgen != font.objgen
        assert fonts[0].CharProcs.a.objgen != fonts[1].CharProcs.a.objgen
        assert fonts[0].Resources.ColorSpace.CS0 == Name.DeviceRGB
        assert fonts[1].Resources.ColorSpace.CS0 == Name.DeviceGray
        assert find_ambiguous_resource_context_streams(pdf) == set()

    def test_font_under_two_names_is_copied_once_per_context(self):
        pdf = new_pdf()
        form = _resourceless_form(pdf)
        font = _type3_font_drawing_form(pdf, form)
        _page_with(
            pdf, Dictionary(Font=Dictionary(T=font), XObject=Dictionary(Fx=form))
        )
        page = _page_with(
            pdf,
            Dictionary(
                ColorSpace=Dictionary(CS0=Name.DeviceGray),
                Font=Dictionary(T=font, U=font),
                XObject=Dictionary(Fx=form),
            ),
        )

        rendering_intent._clone_resource_context_streams(pdf)

        copy_t, copy_u = page.Resources.Font.T, page.Resources.Font.U
        assert copy_t.objgen == copy_u.objgen != font.objgen

    def test_font_referenced_from_extgstate_is_copied(self):
        pdf = new_pdf()
        form = _resourceless_form(pdf)
        font = _type3_font_drawing_form(pdf, form)
        for space in (Name.DeviceRGB, Name.DeviceGray):
            _page_with(
                pdf,
                Dictionary(
                    ColorSpace=Dictionary(CS0=space),
                    ExtGState=Dictionary(GS0=Dictionary(Font=Array([font, 1]))),
                    XObject=Dictionary(Fx=form),
                ),
            )

        sanitize_rendering_intent(pdf)

        first, second = (p.Resources.ExtGState.GS0.Font[0] for p in pdf.pages)
        assert first.objgen != second.objgen
        assert find_ambiguous_resource_context_streams(pdf) == set()

    @pytest.mark.parametrize("own_resources", [False, True])
    def test_font_not_copied_without_a_context_difference(self, own_resources):
        """Equal contexts, or a font with its own /Resources: no copy."""
        pdf = new_pdf()
        form = _resourceless_form(pdf)
        resources = (
            pdf.make_indirect(Dictionary(XObject=Dictionary(Fx=form)))
            if own_resources
            else None
        )
        font = _type3_font_drawing_form(pdf, form, resources)
        for space in (
            Name.DeviceRGB,
            Name.DeviceRGB if not own_resources else Name.DeviceGray,
        ):
            _page_with(
                pdf,
                Dictionary(
                    ColorSpace=Dictionary(CS0=space),
                    Font=Dictionary(T=font),
                    XObject=Dictionary(Fx=form),
                ),
            )

        rendering_intent._clone_resource_context_streams(pdf)

        assert all(p.Resources.Font.T.objgen == font.objgen for p in pdf.pages)


class TestContextCloneReuse:
    """Equal resource contexts get the same clones, so they stay equal."""

    def _type3(self, pdf: Pdf, charprocs: Dictionary) -> Dictionary:
        return pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                FontBBox=Array([0, 0, 1, 1]),
                FontMatrix=Array([1, 0, 0, 1, 0, 0]),
                CharProcs=charprocs,
                Encoding=Dictionary(Differences=Array([97, Name.a])),
                FirstChar=97,
                LastChar=97,
                Widths=Array([1]),
            )
        )

    def test_identical_pages_share_clones_and_font_copy(self):
        pdf = new_pdf()
        logo = _resourceless_form(pdf)
        charprocs = pdf.make_indirect(
            Dictionary(a=pdf.make_stream(b"0 0 d0 /CS0 cs 0 0 1 1 re f"))
        )
        font = self._type3(pdf, charprocs)
        cover = pdf.add_blank_page(page_size=(10, 10))
        cover.Resources = pdf.make_indirect(Dictionary(XObject=Dictionary(Logo=logo)))
        for space in (Name.DeviceRGB,) * 2 + (Name.DeviceGray,) * 2:
            _page_with(
                pdf,
                Dictionary(
                    ColorSpace=Dictionary(CS0=space),
                    Font=Dictionary(T=font),
                    XObject=Dictionary(Logo=logo),
                ),
            )

        sanitize_rendering_intent(pdf)  # previously raised ConversionError

        body = list(pdf.pages)[1:]
        logos = [p.Resources.XObject.Logo.objgen for p in body]
        assert logos[0] == logos[1] and logos[2] == logos[3]
        assert len({*logos, logo.objgen}) == 3
        assert find_ambiguous_resource_context_streams(pdf) == set()

    def test_distinct_fonts_sharing_charprocs_are_separated(self):
        pdf = new_pdf()
        charprocs = pdf.make_indirect(
            Dictionary(a=pdf.make_stream(b"0 0 d0 /CS0 cs 0 0 1 1 re f"))
        )
        fonts = [self._type3(pdf, charprocs) for _ in range(2)]
        for space, font in zip((Name.DeviceRGB, Name.DeviceGray), fonts):
            _page_with(
                pdf,
                Dictionary(ColorSpace=Dictionary(CS0=space), Font=Dictionary(T=font)),
            )

        sanitize_rendering_intent(pdf)  # previously raised ConversionError

        first, second = (p.Resources.Font.T.CharProcs for p in pdf.pages)
        assert first.objgen != second.objgen
        assert find_ambiguous_resource_context_streams(pdf) == set()


class TestNameFreeStreamsAreNotCloned:
    """Streams that look up no resource names need no per-context copies."""

    @pytest.mark.parametrize(
        ("body", "uses_names"),
        [
            (b"0 0 m 1 1 l S", False),
            (b"0 0 d0 0 0 1 1 re f", False),
            (b"/DeviceRGB cs 1 0 0 sc 0 0 1 1 re f", True),
            (b"BI /W 1 /H 1 /IM true ID \x00 EI", False),
            (b"/CS0 cs 0 0 1 1 re f", True),
            (b"/GS0 gs", True),
            (b"/Fm Do", True),
            (b"BT /F1 1 Tf ET", True),
            (b"/P0 scn", True),
            (b"/OC /MC0 BDC EMC", True),
            (b"BI /W 1 /H 1 /CS /CS0 /BPC 8 ID \x00 EI", True),
        ],
    )
    def test_stream_uses_named_resources(self, body: bytes, uses_names: bool):
        from pdftopdfa.fonts.glyph_usage import stream_uses_named_resources

        pdf = new_pdf()
        assert stream_uses_named_resources(pdf.make_stream(body)) is uses_names

    def test_shared_name_free_glyphs_and_forms_are_not_copied(self):
        pdf = new_pdf()
        logo = _resourceless_form(pdf, b"0 0 m 1 1 l S")
        charprocs = pdf.make_indirect(
            Dictionary(a=pdf.make_stream(b"0 0 d0 0 0 1 1 re f"))
        )
        font = TestContextCloneReuse()._type3(pdf, charprocs)
        for space in (Name.DeviceRGB, Name.DeviceGray):
            _page_with(
                pdf,
                Dictionary(
                    ColorSpace=Dictionary(CS0=space),
                    Font=Dictionary(T=font),
                    XObject=Dictionary(Logo=logo),
                ),
            )

        assert rendering_intent._clone_resource_context_streams(pdf) == 0
        sanitize_rendering_intent(pdf)

        assert {p.Resources.Font.T.objgen for p in pdf.pages} == {font.objgen}
        assert {p.Resources.XObject.Logo.objgen for p in pdf.pages} == {logo.objgen}


class TestExplicitResourcesInheritOnlyUsedNames:
    """Made-explicit /Resources must not copy the whole parent dictionary."""

    def _build(self) -> tuple[Pdf, pikepdf.Stream, Dictionary]:
        pdf = new_pdf()
        form = _resourceless_form(pdf, b"0 0 m 100 0 l S")  # uses no names
        glyph = pdf.make_stream(
            b"10 0 0 0 10 10 d1 BI /W 8 /H 1 /IM true /BPC 1 ID \x00 EI"
        )
        font = TestContextCloneReuse()._type3(pdf, Dictionary(a=glyph))
        for with_font in (True, False, True):
            resources = Dictionary(XObject=Dictionary(X=form))
            if with_font:
                resources[Name.Font] = Dictionary(F0=font)
            page = pdf.add_blank_page(page_size=(10, 10))
            page.Resources = resources
            page.Contents = pdf.make_stream(b"/X Do")
        return pdf, form, font

    def test_type3_font_and_form_do_not_form_a_resource_cycle(self):
        pdf, form, font = self._build()

        sanitize_rendering_intent(pdf)

        font = pdf.pages[0].Resources.Font.F0
        form = pdf.pages[0].Resources.XObject.X
        assert "/Resources" in font and "/Resources" in form  # explicit
        assert "/XObject" not in font.Resources  # glyphs draw no XObject
        assert "/Font" not in form.Resources  # form shows no text

    def test_used_names_are_still_inherited(self):
        pdf = new_pdf()
        form = _resourceless_form(pdf, b"/CS0 cs 0 0 m 1 1 l S")
        page = pdf.add_blank_page(page_size=(10, 10))
        page.Resources = Dictionary(
            ColorSpace=Dictionary(CS0=Name.DeviceRGB, CS1=Name.DeviceGray),
            XObject=Dictionary(X=form),
            Font=Dictionary(F1=Dictionary(Type=Name.Font, Subtype=Name.Type1)),
        )
        page.Contents = pdf.make_stream(b"/X Do")

        sanitize_rendering_intent(pdf)

        resources = pdf.pages[0].Resources.XObject.X.Resources
        assert set(resources.keys()) == {"/ColorSpace"}
        assert set(resources.ColorSpace.keys()) == {"/CS0"}
