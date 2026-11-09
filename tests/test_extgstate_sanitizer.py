# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Extended Graphics State sanitization for PDF/A compliance."""

from io import BytesIO
from unittest.mock import patch

import pikepdf
import pytest
from conftest import new_pdf, pdf_object
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import get_cmyk_profile
from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.sanitizers import extgstate
from pdftopdfa.sanitizers.extgstate import (
    _sanitize_shadings_in_resources,
    sanitize_extgstate,
)
from pdftopdfa.sanitizers.structure_limits import sanitize_structure_limits


def _make_form_stream(pdf: Pdf):
    """Create a minimal Form XObject stream for use as SMask /G."""
    s = pdf.make_stream(b"q Q")
    s[Name.Subtype] = Name.Form
    return s


def _make_pdf_with_extgstate(pdf: Pdf, gs_dict: Dictionary) -> None:
    """Helper: add a page with an ExtGState resource to a PDF.

    Args:
        pdf: An open pikepdf Pdf.
        gs_dict: The graphics state dictionary to add as /GS0.
    """
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


class TestTRRemoval:
    """Tests for /TR (transfer function) removal."""

    @pytest.fixture
    def pdf_with_tr_stream(self) -> Pdf:
        """PDF with /TR set to a function stream."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR=tr_stream)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_tr_identity(self) -> Pdf:
        """PDF with /TR set to /Identity name."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    def test_tr_stream_removed(self, pdf_with_tr_stream: Pdf):
        """Removes /TR function stream from ExtGState."""
        result = sanitize_extgstate(pdf_with_tr_stream)

        assert result["extgstate_fixed"] == 1
        resources = pdf_with_tr_stream.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR" not in gs

    def test_tr_identity_removed(self, pdf_with_tr_identity: Pdf):
        """Removes /TR /Identity name from ExtGState."""
        result = sanitize_extgstate(pdf_with_tr_identity)

        assert result["extgstate_fixed"] == 1
        resources = pdf_with_tr_identity.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR" not in gs


class TestTR2Handling:
    """Tests for /TR2 handling (forbidden unless /Default)."""

    @pytest.fixture
    def pdf_with_tr2_default(self) -> Pdf:
        """PDF with /TR2 set to /Default (compliant)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, TR2=Name.Default)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_tr2_stream(self) -> Pdf:
        """PDF with /TR2 set to a function stream (non-compliant)."""
        pdf = new_pdf()
        tr2_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR2=tr2_stream)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_tr2_name(self) -> Pdf:
        """PDF with /TR2 set to a non-Default name (non-compliant)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, TR2=Name.Identity)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    def test_tr2_default_preserved(self, pdf_with_tr2_default: Pdf):
        """Preserves /TR2 /Default (compliant value)."""
        result = sanitize_extgstate(pdf_with_tr2_default)

        assert result["extgstate_fixed"] == 0
        resources = pdf_with_tr2_default.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR2" in gs
        assert str(gs.TR2) == "/Default"

    def test_tr2_stream_removed(self, pdf_with_tr2_stream: Pdf):
        """Removes /TR2 function stream from ExtGState."""
        result = sanitize_extgstate(pdf_with_tr2_stream)

        assert result["extgstate_fixed"] == 1
        resources = pdf_with_tr2_stream.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR2" not in gs

    def test_tr2_non_default_name_removed(self, pdf_with_tr2_name: Pdf):
        """Removes /TR2 with non-Default name value."""
        result = sanitize_extgstate(pdf_with_tr2_name)

        assert result["extgstate_fixed"] == 1
        resources = pdf_with_tr2_name.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR2" not in gs


class TestHTPRemoval:
    """Tests for /HTP (halftone phase) removal."""

    @pytest.fixture
    def pdf_with_htp(self) -> Pdf:
        """PDF with /HTP entry in ExtGState."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, HTP=Array([10, 20]))
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    def test_htp_removed(self, pdf_with_htp: Pdf):
        """Removes /HTP from ExtGState."""
        result = sanitize_extgstate(pdf_with_htp)

        assert result["extgstate_fixed"] == 1
        resources = pdf_with_htp.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/HTP" not in gs


class TestHTHalftone:
    """Tests for /HT (halftone) validation in PDF/A-2/3."""

    @pytest.fixture
    def pdf_with_ht_type5(self) -> Pdf:
        """PDF with /HT containing HalftoneType 5 (allowed in PDF/A-2/3)."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=5)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_ht_type1(self) -> Pdf:
        """PDF with /HT containing HalftoneType 1 (allowed)."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=1)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_ht_default(self) -> Pdf:
        """PDF with /HT set to /Default name (allowed)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, HT=Name.Default)
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    def test_ht_type5_preserved(self, pdf_with_ht_type5: Pdf):
        """Preserves /HT with HalftoneType 5 (allowed in PDF/A-2/3)."""
        result = sanitize_extgstate(pdf_with_ht_type5)

        assert result["extgstate_fixed"] == 0
        resources = pdf_with_ht_type5.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/HT" in gs

    def test_ht_type1_preserved(self, pdf_with_ht_type1: Pdf):
        """Preserves /HT with HalftoneType 1 (allowed)."""
        result = sanitize_extgstate(pdf_with_ht_type1)

        assert result["extgstate_fixed"] == 0
        resources = pdf_with_ht_type1.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/HT" in gs

    def test_ht_default_preserved(self, pdf_with_ht_default: Pdf):
        """Preserves /HT /Default name (allowed)."""
        result = sanitize_extgstate(pdf_with_ht_default)

        assert result["extgstate_fixed"] == 0
        resources = pdf_with_ht_default.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/HT" in gs

    def test_ht_invalid_halftonetype_removed(self) -> None:
        """Removes /HT when HalftoneType is not 1 or 5."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=6, Width=5, Height=5)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert "/HT" not in gs_out

    @pytest.mark.parametrize(
        "tf_value_desc",
        ["stream", "Default", "Identity"],
    )
    def test_ht_transferfunction_removed(self, tf_value_desc: str) -> None:
        """Removes /TransferFunction regardless of value type."""
        pdf = new_pdf()
        if tf_value_desc == "stream":
            tf_value = pdf.make_stream(b"{ }")
        elif tf_value_desc == "Default":
            tf_value = Name.Default
        else:
            tf_value = Name.Identity
        ht_dict = Dictionary(HalftoneType=1, TransferFunction=tf_value)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        ht = pdf.pages[0].Resources.ExtGState.GS0.HT
        assert "/TransferFunction" not in ht

    def test_ht_halftonename_removed(self) -> None:
        """Removes /HalftoneName from halftone dictionary."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=1, HalftoneName=Name("/MyHT"))
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        ht = pdf.pages[0].Resources.ExtGState.GS0.HT
        assert "/HalftoneName" not in ht

    def test_ht_type5_sub_halftones_sanitized(self) -> None:
        """Sanitizes sub-halftones inside a Type 5 composite halftone."""
        pdf = new_pdf()
        tf_stream = pdf.make_stream(b"{ }")
        sub_ht1 = Dictionary(
            HalftoneType=1,
            TransferFunction=tf_stream,
            HalftoneName=Name("/Sub1"),
        )
        sub_ht_default = Dictionary(
            HalftoneType=1,
            HalftoneName=Name("/DefaultHT"),
        )
        ht_dict = Dictionary(HalftoneType=5)
        ht_dict[Name("/Cyan")] = sub_ht1
        ht_dict[Name("/Default")] = sub_ht_default

        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        # sub_ht1: TransferFunction removed + HalftoneName removed = 2
        # sub_ht_default: HalftoneName removed = 1
        assert result["extgstate_fixed"] == 3
        ht = pdf.pages[0].Resources.ExtGState.GS0.HT
        assert "/HalftoneName" not in ht["/Cyan"]
        assert "/TransferFunction" not in ht["/Cyan"]
        assert "/HalftoneName" not in ht["/Default"]

    def test_ht_tree_beyond_python_recursion_limit(self) -> None:
        """Sanitizes the leaf of an exceptionally deep Type 5 halftone tree."""
        pdf = new_pdf()
        halftone = Dictionary(
            HalftoneType=1,
            HalftoneName=Name("/Forbidden"),
        )
        leaf = halftone
        for _ in range(1200):
            parent = Dictionary(HalftoneType=5)
            parent[Name("/Default")] = halftone
            halftone = parent
        _make_pdf_with_extgstate(
            pdf,
            Dictionary(Type=Name.ExtGState, HT=halftone),
        )

        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert "/HalftoneName" not in leaf

    def test_ht_type5_non_primary_transferfunction_added(self) -> None:
        """Adds /TransferFunction for non-primary Type 5 colorants."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=5)
        ht_dict[Name("/Default")] = Dictionary(HalftoneType=1)
        ht_dict[Name("/Red")] = Dictionary(HalftoneType=1)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        red_ht = pdf.pages[0].Resources.ExtGState.GS0.HT["/Red"]
        assert str(red_ht["/TransferFunction"]) == "/Identity"

    def test_ht_type5_non_primary_uses_default_transferfunction(self) -> None:
        """Copies /Default TransferFunction to non-primary Type 5 colorants."""
        pdf = new_pdf()
        tf_stream = pdf.make_stream(b"{ }")
        default_ht = Dictionary(HalftoneType=1, TransferFunction=tf_stream)
        red_ht = Dictionary(HalftoneType=1)
        ht_dict = Dictionary(HalftoneType=5)
        ht_dict[Name("/Default")] = default_ht
        ht_dict[Name("/Red")] = red_ht
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        red_tf = pdf.pages[0].Resources.ExtGState.GS0.HT["/Red"]["/TransferFunction"]
        assert red_tf.objgen == tf_stream.objgen

    def test_ht_type5_primary_transferfunction_removed(self) -> None:
        """Removes /TransferFunction for primary Type 5 colorants."""
        pdf = new_pdf()
        tf_stream = pdf.make_stream(b"{ }")
        ht_dict = Dictionary(HalftoneType=5)
        ht_dict[Name("/Default")] = Dictionary(HalftoneType=1)
        ht_dict[Name("/Cyan")] = Dictionary(HalftoneType=1, TransferFunction=tf_stream)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        cyan_ht = pdf.pages[0].Resources.ExtGState.GS0.HT["/Cyan"]
        assert "/TransferFunction" not in cyan_ht

    def test_ht_no_forbidden_entries(self) -> None:
        """Clean halftone dictionary requires no fixes."""
        pdf = new_pdf()
        ht_dict = Dictionary(HalftoneType=1)
        gs = Dictionary(Type=Name.ExtGState, HT=ht_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0


class TestMultipleEntries:
    """Tests for multiple forbidden entries in same GS / multiple GS dicts."""

    @pytest.fixture
    def pdf_with_multiple_forbidden(self) -> Pdf:
        """PDF with multiple forbidden entries in a single GS dict."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(
            Type=Name.ExtGState,
            TR=tr_stream,
            TR2=Name.Identity,
            HTP=Array([5, 10]),
        )
        _make_pdf_with_extgstate(pdf, gs)
        return pdf

    @pytest.fixture
    def pdf_with_multiple_gs(self) -> Pdf:
        """PDF with multiple GS dicts on same page, each with forbidden entries."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs0 = Dictionary(Type=Name.ExtGState, TR=tr_stream)
        gs1 = Dictionary(Type=Name.ExtGState, HTP=Array([1, 2]))

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ExtGState=Dictionary(GS0=gs0, GS1=gs1),
                ),
            )
        )
        pdf.pages.append(page)
        return pdf

    def test_multiple_forbidden_in_same_gs(self, pdf_with_multiple_forbidden: Pdf):
        """Removes all forbidden entries from a single GS dict."""
        result = sanitize_extgstate(pdf_with_multiple_forbidden)

        assert result["extgstate_fixed"] == 3
        resources = pdf_with_multiple_forbidden.pages[0].Resources
        gs = resources.ExtGState.GS0
        assert "/TR" not in gs
        assert "/TR2" not in gs
        assert "/HTP" not in gs

    def test_multiple_gs_dicts_on_page(self, pdf_with_multiple_gs: Pdf):
        """Processes all GS dicts in ExtGState resource dictionary."""
        result = sanitize_extgstate(pdf_with_multiple_gs)

        assert result["extgstate_fixed"] == 2
        resources = pdf_with_multiple_gs.pages[0].Resources
        assert "/TR" not in resources.ExtGState.GS0
        assert "/HTP" not in resources.ExtGState.GS1


class TestNestedFormXObjects:
    """Tests for ExtGState in Form XObject resources."""

    @pytest.fixture
    def pdf_with_form_xobject_extgstate(self) -> Pdf:
        """PDF with ExtGState inside a Form XObject's Resources."""
        pdf = new_pdf()

        # Create a Form XObject with its own ExtGState
        form_stream = pdf.make_stream(b"q Q")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        tr_stream = pdf.make_stream(b"{ }")
        form_stream[Name.Resources] = Dictionary(
            ExtGState=Dictionary(
                GS0=Dictionary(Type=Name.ExtGState, TR=tr_stream),
            ),
        )

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
        return pdf

    @pytest.fixture
    def pdf_with_deeply_nested_form(self) -> Pdf:
        """PDF with ExtGState in a deeply nested Form XObject."""
        pdf = new_pdf()

        # Inner form with forbidden ExtGState
        inner_form = pdf.make_stream(b"q Q")
        inner_form[Name.Type] = Name.XObject
        inner_form[Name.Subtype] = Name.Form
        inner_form[Name.BBox] = Array([0, 0, 50, 50])
        tr_stream = pdf.make_stream(b"{ }")
        inner_form[Name.Resources] = Dictionary(
            ExtGState=Dictionary(
                GS0=Dictionary(Type=Name.ExtGState, TR=tr_stream),
            ),
        )

        # Outer form referencing inner form
        outer_form = pdf.make_stream(b"/InnerForm Do")
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
        return pdf

    def test_form_xobject_extgstate_sanitized(
        self, pdf_with_form_xobject_extgstate: Pdf
    ):
        """Sanitizes ExtGState inside Form XObject Resources."""
        result = sanitize_extgstate(pdf_with_form_xobject_extgstate)

        assert result["extgstate_fixed"] == 1

    def test_deeply_nested_form_xobject(self, pdf_with_deeply_nested_form: Pdf):
        """Sanitizes ExtGState in deeply nested Form XObjects."""
        result = sanitize_extgstate(pdf_with_deeply_nested_form)

        assert result["extgstate_fixed"] == 1

    def test_very_deep_form_resources_are_iterative(self):
        """A hostile resource depth must not exhaust the Python stack."""
        pdf = new_pdf()
        leaf_gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        resources = Dictionary(ExtGState=Dictionary(GS0=leaf_gs))
        for index in range(1200):
            form = pdf.make_stream(b"q Q")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = Array([0, 0, 1, 1])
            form[Name.Resources] = resources
            resources = Dictionary(XObject=Dictionary({f"/F{index}": form}))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 1, 1]),
                Resources=resources,
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        assert "/TR" not in leaf_gs


class TestResourceDiscovery:
    """Tests for every context that can own a Resources dictionary."""

    def test_inherited_page_resources(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 10, 10]),
            )
        )
        pdf.pages.append(page)
        gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        page.obj.Parent.Resources = Dictionary(ExtGState=Dictionary(GS0=gs))

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        assert "/TR" not in gs

    def test_tiling_pattern_resources(self):
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        pattern = pdf.make_stream(b"/GS0 gs")
        pattern[Name("/PatternType")] = 1
        pattern[Name.Resources] = Dictionary(ExtGState=Dictionary(GS0=gs))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 10, 10]),
                Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        assert "/TR" not in gs

    def test_soft_mask_group_resources(self):
        pdf = new_pdf()
        nested_gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        group = _make_form_stream(pdf)
        group[Name.Resources] = Dictionary(ExtGState=Dictionary(GS1=nested_gs))
        smask = Dictionary(S=Name.Alpha, G=group)
        outer_gs = Dictionary(Type=Name.ExtGState, SMask=smask)
        _make_pdf_with_extgstate(pdf, outer_gs)

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        assert "/TR" not in nested_gs

    def test_acroform_default_resources(self):
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, TR=Name.Identity)
        pdf.Root.AcroForm = Dictionary(
            DR=Dictionary(ExtGState=Dictionary(GS0=gs)),
        )

        result = sanitize_extgstate(pdf)

        assert result["extgstate_fixed"] == 1
        assert "/TR" not in gs


class TestAnnotationAppearanceStreams:
    """Tests for ExtGState in annotation AP stream resources."""

    @pytest.fixture
    def pdf_with_annot_ap_extgstate(self) -> Pdf:
        """PDF with ExtGState in annotation appearance stream Resources."""
        pdf = new_pdf()

        # Create an AP stream (Form XObject) with forbidden ExtGState
        ap_stream = pdf.make_stream(b"q Q")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 20, 20])
        tr_stream = pdf.make_stream(b"{ }")
        ap_stream[Name.Resources] = Dictionary(
            ExtGState=Dictionary(
                GS0=Dictionary(Type=Name.ExtGState, TR=tr_stream),
            ),
        )

        # Create annotation with AP
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

        return pdf

    @pytest.fixture
    def pdf_with_annot_ap_substates(self) -> Pdf:
        """PDF with ExtGState in annotation AP sub-state dictionary."""
        pdf = new_pdf()

        # Create sub-state AP streams
        on_stream = pdf.make_stream(b"q Q")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 20, 20])
        tr_stream = pdf.make_stream(b"{ }")
        on_stream[Name.Resources] = Dictionary(
            ExtGState=Dictionary(
                GS0=Dictionary(Type=Name.ExtGState, TR=tr_stream),
            ),
        )

        off_stream = pdf.make_stream(b"q Q")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 20, 20])
        htp_gs = Dictionary(Type=Name.ExtGState, HTP=Array([5, 10]))
        off_stream[Name.Resources] = Dictionary(
            ExtGState=Dictionary(GS0=htp_gs),
        )

        # AP /N is a dict of sub-states
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

        return pdf

    def test_annot_ap_extgstate_sanitized(self, pdf_with_annot_ap_extgstate: Pdf):
        """Sanitizes ExtGState in annotation AP stream Resources."""
        result = sanitize_extgstate(pdf_with_annot_ap_extgstate)

        assert result["extgstate_fixed"] == 1

    def test_annot_ap_substates_sanitized(self, pdf_with_annot_ap_substates: Pdf):
        """Sanitizes ExtGState in annotation AP sub-state streams."""
        result = sanitize_extgstate(pdf_with_annot_ap_substates)

        assert result["extgstate_fixed"] == 2


class TestNoChangesNeeded:
    """Tests for PDFs that don't need ExtGState changes."""

    def test_no_extgstate(self, sample_pdf_obj: Pdf):
        """PDF without ExtGState returns zero count."""
        result = sanitize_extgstate(sample_pdf_obj)

        assert result["extgstate_fixed"] == 0

    def test_no_resources(self) -> None:
        """PDF page without Resources returns zero count."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_compliant_only_entries(self) -> None:
        """PDF with only compliant ExtGState entries returns zero count."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            CA=0.5,
            ca=0.8,
            BM=Name.Normal,
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

        # Verify original entries are untouched
        resources = pdf.pages[0].Resources
        assert "/CA" in resources.ExtGState.GS0
        assert "/ca" in resources.ExtGState.GS0
        assert "/BM" in resources.ExtGState.GS0

    def test_empty_extgstate_dict(self) -> None:
        """PDF with empty ExtGState dict returns zero count."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(ExtGState=Dictionary()),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0


class TestIntegration:
    """Integration tests with sanitize_for_pdfa."""

    def test_sanitize_for_pdfa_includes_extgstate_key(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa returns extgstate_fixed key."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "extgstate_fixed" in result
        assert result["extgstate_fixed"] == 0

    def test_sanitize_for_pdfa_removes_entries(self) -> None:
        """sanitize_for_pdfa actually removes forbidden ExtGState entries."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR=tr_stream, HTP=Array([1, 2]))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_for_pdfa(pdf, "3b")

        assert result["extgstate_fixed"] == 2
        resources = pdf.pages[0].Resources
        assert "/TR" not in resources.ExtGState.GS0
        assert "/HTP" not in resources.ExtGState.GS0


class TestBlendModeValidation:
    """Tests for /BM (blend mode) validation (ISO 19005-2, 6.4)."""

    @pytest.mark.parametrize("bm", [Name.Normal, Name.Multiply, Name.Compatible])
    def test_valid_blend_mode_preserved(self, bm) -> None:
        """Preserves valid /BM values."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, BM=bm)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert str(pdf.pages[0].Resources.ExtGState.GS0.BM) == str(bm)

    def test_invalid_blend_mode_replaced(self) -> None:
        """Replaces invalid /BM name with /Normal."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, BM=Name("/Invalid"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.BM) == "/Normal"

    def test_invalid_blend_mode_array_fixed(self) -> None:
        """Replaces only invalid entries in /BM array, preserving valid ones."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, BM=Array([Name.Normal, Name("/Bogus")]))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Array)
        assert str(bm[0]) == "/Normal"
        assert str(bm[1]) == "/Normal"

    def test_valid_blend_mode_array_preserved(self) -> None:
        """Preserves /BM array when all entries are valid."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, BM=Array([Name.Multiply, Name.Screen]))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0


class TestOpacityValidation:
    """Tests for /CA and /ca (opacity) validation (ISO 19005-2, 6.4)."""

    def test_valid_ca_preserved(self) -> None:
        """Preserves valid /CA and /ca values in [0.0, 1.0]."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            CA=0.5,
            ca=0.8,
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert float(gs_out.CA) == pytest.approx(0.5)
        assert float(gs_out.ca) == pytest.approx(0.8)

    def test_ca_zero_preserved(self) -> None:
        """Preserves /ca = 0.0 (fully transparent, but valid)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, ca=0.0)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_ca_one_preserved(self) -> None:
        """Preserves /CA = 1.0 (fully opaque, valid)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, CA=1.0)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_ca_above_one_clamped(self) -> None:
        """Clamps /CA > 1.0 to 1.0."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, CA=1.5)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert float(pdf.pages[0].Resources.ExtGState.GS0.CA) == pytest.approx(1.0)

    def test_ca_negative_clamped(self) -> None:
        """Clamps /ca < 0.0 to 0.0."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, ca=-0.5)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert float(pdf.pages[0].Resources.ExtGState.GS0.ca) == pytest.approx(0.0)

    def test_both_ca_out_of_range(self) -> None:
        """Clamps both /CA and /ca when out of range."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            CA=2.0,
            ca=-1.0,
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 2
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert float(gs_out.CA) == pytest.approx(1.0)
        assert float(gs_out.ca) == pytest.approx(0.0)

    def test_non_numeric_ca_reset(self) -> None:
        """Resets non-numeric /CA to 1.0."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, CA=Name("/Bad"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert float(pdf.pages[0].Resources.ExtGState.GS0.CA) == pytest.approx(1.0)


class TestSoftMaskValidation:
    """Tests for /SMask (soft mask) validation (ISO 19005-2, 6.4)."""

    def test_smask_none_preserved(self) -> None:
        """Preserves /SMask /None (no soft mask)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, SMask=Name("/None"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"

    def test_smask_dict_preserved(self) -> None:
        """Preserves /SMask dictionary (valid soft mask)."""
        pdf = new_pdf()
        smask_dict = Dictionary(
            S=Name.Alpha,
            G=_make_form_stream(pdf),
        )
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert "/SMask" in pdf.pages[0].Resources.ExtGState.GS0

    def test_smask_invalid_name_replaced(self) -> None:
        """Replaces invalid /SMask name with /None."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, SMask=Name("/Bad"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"

    def test_smask_malformed_removed(self) -> None:
        """Removes malformed /SMask (non-Name, non-Dictionary)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, SMask=Array([1, 2, 3]))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/SMask" not in pdf.pages[0].Resources.ExtGState.GS0

    @pytest.mark.parametrize(
        ("smask_desc", "smask_factory"),
        [
            ("missing_S", lambda pdf: Dictionary(G=pdf.make_stream(b"q Q"))),
            (
                "invalid_S",
                lambda pdf: Dictionary(S=Name("/Bad"), G=pdf.make_stream(b"q Q")),
            ),
            ("missing_G", lambda _: Dictionary(S=Name.Alpha)),
            ("invalid_G", lambda _: Dictionary(S=Name.Alpha, G=Dictionary())),
        ],
    )
    def test_smask_dict_invalid_replaced(self, smask_desc, smask_factory) -> None:
        """Invalid SMask dicts are replaced with /None."""
        pdf = new_pdf()
        smask_dict = smask_factory(pdf)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"

    @pytest.mark.parametrize(
        "transfer_type",
        ["identity", "exponential", "calculator", "sampled", "stitching"],
    )
    @pytest.mark.parametrize("with_extgstate_tr", [False, True])
    def test_smask_transfer_function_preserved(
        self, transfer_type: str, with_extgstate_tr: bool
    ) -> None:
        """Mask opacity mappings survive even when an ExtGState /TR is removed."""
        pdf = new_pdf()
        if transfer_type == "identity":
            transfer = Name.Identity
        elif transfer_type == "exponential":
            transfer = pdf.make_indirect(
                Dictionary(
                    FunctionType=2,
                    Domain=Array([0, 1]),
                    C0=Array([1]),
                    C1=Array([0]),
                    N=1,
                )
            )
        elif transfer_type == "calculator":
            transfer = pdf.make_stream(b"{ 1 exch sub }")
            transfer.FunctionType = 4
            transfer.Domain = Array([0, 1])
            transfer.Range = Array([0, 1])
        elif transfer_type == "sampled":
            transfer = pdf.make_stream(b"\xff\x00")
            transfer.FunctionType = 0
            transfer.Domain = Array([0, 1])
            transfer.Range = Array([0, 1])
            transfer.Size = Array([2])
            transfer.BitsPerSample = 8
        else:
            transfer = Dictionary(
                FunctionType=3,
                Domain=Array([0, 1]),
                Functions=Array(
                    [Dictionary(FunctionType=2, Domain=Array([0, 1]), N=1)]
                ),
                Bounds=Array(),
                Encode=Array([1, 0]),
            )
        smask_dict = Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=transfer)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        if with_extgstate_tr:
            gs.TR = transfer
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == int(with_extgstate_tr)
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert "/TR" not in gs_out
        assert gs_out.SMask.TR == transfer

        saved = BytesIO()
        pdf.save(saved)
        with pikepdf.open(saved) as reopened:
            restored = reopened.pages[0].Resources.ExtGState.GS0.SMask.TR
            if transfer_type == "identity":
                assert restored == Name.Identity
            elif transfer_type == "exponential":
                assert restored.FunctionType == 2
                assert list(restored.Domain) == [0, 1]
                assert list(restored.C0) == [1]
                assert list(restored.C1) == [0]
                assert restored.N == 1
            elif transfer_type == "calculator":
                assert restored.FunctionType == 4
                assert list(restored.Domain) == [0, 1]
                assert list(restored.Range) == [0, 1]
                assert restored.read_bytes() == b"{ 1 exch sub }"
            elif transfer_type == "sampled":
                assert restored.FunctionType == 0
                assert restored.read_bytes() == b"\xff\x00"
            else:
                assert restored.FunctionType == 3
                assert list(restored.Encode) == [1, 0]

    @pytest.mark.parametrize(
        "transfer",
        [
            42,
            Name.Default,
            Dictionary(FunctionType=2, N=1),
            Dictionary(FunctionType=2, Domain=Array([0, 1]), N=Name.Bad),
            Dictionary(FunctionType=2, Domain=Array([0, 1]), N=1, C0=Array([0, 1])),
            Dictionary(
                FunctionType=2, Domain=Array([0, 1]), N=1, Range=Array([0, 1, 0, 1])
            ),
            Dictionary(FunctionType=2, Domain=Array([0, 1]), N=-1),
            Dictionary(FunctionType=7, Domain=Array([0, 1])),
            # Results beyond the real limit of ISO 32000-1, Table C.1.
            Dictionary(FunctionType=2, Domain=Array([0, 2]), N=2000),
            Dictionary(FunctionType=2, Domain=Array([0, 10]), N=39),
            Dictionary(FunctionType=2, Domain=Array([0.001, 1]), N=-20),
            Dictionary(
                FunctionType=2,
                Domain=Array([0, 2]),
                N=1,
                C0=Array([0]),
                C1=Array([3e38]),
            ),
        ],
    )
    def test_invalid_soft_mask_transfer_uses_identity_default(self, transfer) -> None:
        """Malformed transfers are removed without discarding the valid mask group."""
        pdf = new_pdf()
        mask = Dictionary(
            S=Name.Alpha, G=_make_form_stream(pdf), TR=pdf_object(transfer)
        )
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert "/TR" not in pdf.pages[0].Resources.ExtGState.GS0.SMask
        assert pdf.pages[0].Resources.ExtGState.GS0.SMask.G == mask.G

    @pytest.mark.parametrize(
        "kind",
        ["short_samples", "surplus_samples", "bad_calculator", "cycle", "first_bound"],
    )
    def test_invalid_transfer_programs_do_not_survive(self, kind) -> None:
        """Check sampled payloads, calculator syntax, and recursive stitching graphs.

        ISO 32000-1, 7.3.8.2 makes both too little and too much sample data an error,
        and 7.10.4 requires Domain0 < Bounds0 (only the last bound may equal Domain1).
        """
        pdf = new_pdf()
        if kind == "cycle":
            transfer = pdf.make_indirect(
                Dictionary(
                    FunctionType=3,
                    Domain=Array([0, 1]),
                    Bounds=Array(),
                    Encode=Array([0, 1]),
                )
            )
            transfer.Functions = Array([transfer])
        elif kind == "first_bound":
            identity = Dictionary(FunctionType=2, Domain=Array([0, 1]), N=1)
            transfer = Dictionary(
                FunctionType=3,
                Domain=Array([0, 1]),
                Functions=Array([identity, identity]),
                Bounds=Array([0]),
                Encode=Array([0, 1, 0, 1]),
            )
        else:
            samples = {"short_samples": b"\x00", "surplus_samples": b"\x00\x80\xff"}
            transfer = pdf.make_stream(samples.get(kind, b"{ bogus }"))
            transfer.FunctionType = 0 if kind in samples else 4
            transfer.Domain = transfer.Range = Array([0, 1])
            if kind in samples:
                transfer.Size = Array([2])
                transfer.BitsPerSample = 8
        _make_pdf_with_extgstate(
            pdf,
            Dictionary(
                SMask=Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=transfer)
            ),
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert "/TR" not in pdf.pages[0].Resources.ExtGState.GS0.SMask

    @pytest.mark.parametrize(
        "domain,exponent,c1",
        [
            ([0, 1], 2000, 1),
            ([0, 10], 38, 1),
            ([0.01, 1], -19, 1),
            ([-2, 2], 3, -1),
            ([0, 1], 1, 3e38),
        ],
    )
    def test_exponential_with_large_finite_results_is_preserved(
        self, domain, exponent, c1
    ) -> None:
        """Results that stay within the real limit keep an exponential transfer."""
        pdf = new_pdf()
        transfer = Dictionary(
            FunctionType=2, Domain=Array(domain), N=exponent, C1=Array([c1])
        )
        mask = Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=transfer)
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert "/TR" in pdf.pages[0].Resources.ExtGState.GS0.SMask

    @pytest.mark.parametrize(
        "domain,program,kept",
        [
            ([1, 2], b"{ ln }", True),
            ([0, 1], b"{ ln }", False),
            ([-1, 1], b"{ sqrt }", False),
            ([0, 1], b"{ sqrt }", True),
            ([1, 2], b"{ 1 exch div }", True),
        ],
    )
    def test_calculator_is_checked_over_its_declared_domain(
        self, domain, program, kept
    ) -> None:
        """Whether an operation has a defined result depends on the actual input."""
        pdf = new_pdf()
        mask = self._calculator_mask(pdf, program)
        mask.TR.Domain = Array(domain)
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        sanitize_extgstate(pdf)
        assert ("/TR" in pdf.pages[0].Resources.ExtGState.GS0.SMask) is kept

    def test_calculator_sampling_is_bounded_by_program_size(self) -> None:
        """Thousands of literals cannot make validation quadratic in the stream."""
        literals = b" ".join(b"0.%05d pop" % step for step in range(1, 5001))
        program = extgstate._parse_calculator(b"{ " + literals + b" }")
        samples = extgstate._calculator_samples(program, 0.0, 1.0)
        size = extgstate._calculator_size(program)
        assert size == 10000
        assert len(samples) * size <= extgstate._CALCULATOR_SAMPLE_BUDGET
        assert samples[:3] == [0.0, 1.0, 0.5]

        pdf = new_pdf()
        mask = self._calculator_mask(pdf, b"{ " + literals + b" }")
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    def test_small_calculator_keeps_every_sample(self) -> None:
        """A short program runs on the whole grid and on all its literals."""
        program = extgstate._parse_calculator(b"{ 0.3 sub 0.7 mul }")
        samples = extgstate._calculator_samples(program, 0.0, 1.0)
        assert len(samples) == len(set(samples))
        assert {0.0, 1.0, 0.5, 0.3, 0.7, 1 / 255} <= set(samples)

    def test_shared_stitching_children_are_validated_once(self) -> None:
        """A deep graph that reuses each child twice is not walked exponentially."""
        pdf = new_pdf()
        child = pdf.make_indirect(Dictionary(FunctionType=2, Domain=Array([0, 1]), N=1))
        for _ in range(40):
            child = pdf.make_indirect(
                Dictionary(
                    FunctionType=3,
                    Domain=Array([0, 1]),
                    Functions=Array([child, child]),
                    Bounds=Array([0.5]),
                    Encode=Array([0, 1, 0, 1]),
                )
            )
        _make_pdf_with_extgstate(
            pdf,
            Dictionary(
                SMask=Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=child)
            ),
        )
        with patch.object(
            extgstate,
            "_check_scalar_transfer_function",
            wraps=extgstate._check_scalar_transfer_function,
        ) as checked:
            assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert checked.call_count == 41
        assert pdf.pages[0].Resources.ExtGState.GS0.SMask.TR.objgen == child.objgen

    # A real of magnitude 1e-40, which pikepdf keeps only when parsed: a Decimal
    # assigned from Python is rounded to 0 before it reaches the file.
    _TINY = b"0." + b"0" * 39 + b"1"
    _IDENTITY = b"<< /FunctionType 2 /Domain [0 1] /N 1 >>"
    _NORMALIZED_DICTIONARIES = {
        # Bounds [0] once written, violating Domain0 < Bounds0 (7.10.4).
        "first_bound": b"<< /FunctionType 3 /Domain [0 1] /Functions [ID ID]"
        b" /Bounds [TINY] /Encode [0 1 0 1] >>",
        # Domain [0 1] and Bounds [0] once written.
        "stitching_domain": b"<< /FunctionType 3 /Domain [-TINY 1]"
        b" /Functions [ID ID] /Bounds [0] /Encode [0 1 0 1] >>",
        # Domain [0 1] once written, which a negative N must exclude.
        "negative_exponent_domain": b"<< /FunctionType 2 /Domain [TINY 1] /N -1 >>",
        # N becomes 0, which the Domain [0 1] permits, unlike a negative N.
        "subnormal_exponent": b"<< /FunctionType 2 /Domain [0 1] /N -TINY >>",
        # Domain [0 1] once written, which a non-integer N permits.
        "fractional_exponent_domain": b"<< /FunctionType 2 /Domain [-TINY 1] /N 0.5 >>",
        "subnormal_coefficients": b"<< /FunctionType 2 /Domain [0 1] /N 1"
        b" /C0 [TINY] /C1 [-TINY] >>",
        "subnormal_encode": b"<< /FunctionType 3 /Domain [0 1] /Functions [ID ID]"
        b" /Bounds [0.5] /Encode [TINY 1 0 -TINY] >>",
    }
    _NORMALIZED_CALCULATORS = {
        # Domain [0 1] once written, where { ln } is undefined at 0.
        "calculator_domain": (b"[TINY 1]", b"{ ln }"),
        # Domain [2147483647 2147483647] once written, where the program is
        # undefined, although it is defined at every original input.
        "clamped_domain": (b"[4294967296 4294967297]", b"{ 2147483648 sub ln }"),
    }

    @classmethod
    def _normalized_transfer(cls, pdf: Pdf, case: str):
        """Create a transfer function whose verdict depends on number normalization.

        sanitize_structure_limits() sets a real below 1.175e-38 in magnitude to 0
        and clamps integers to the 32-bit limits, so the validator must decide on
        the values that pass writes, not on the original ones.
        """
        if case in cls._NORMALIZED_CALCULATORS:
            domain, program = cls._NORMALIZED_CALCULATORS[case]
            transfer = pdf.make_stream(program)
            transfer.FunctionType = 4
            transfer.Domain = pikepdf.Object.parse(domain.replace(b"TINY", cls._TINY))
            transfer.Range = Array([-100, 100])
            return transfer
        text = cls._NORMALIZED_DICTIONARIES[case]
        text = text.replace(b"ID", cls._IDENTITY).replace(b"TINY", cls._TINY)
        return pikepdf.Object.parse(text)

    _NORMALIZATION_CASES = [
        ("calculator_domain", False),
        ("first_bound", False),
        ("stitching_domain", False),
        ("negative_exponent_domain", False),
        ("clamped_domain", False),
        ("subnormal_exponent", True),
        ("fractional_exponent_domain", True),
        ("subnormal_coefficients", True),
        ("subnormal_encode", True),
    ]

    @pytest.mark.parametrize("case,kept", _NORMALIZATION_CASES)
    @pytest.mark.parametrize("limits_first", [False, True])
    def test_transfer_is_validated_as_structure_limits_writes_it(
        self, case, kept, limits_first
    ) -> None:
        """The /TR verdict holds for the written numbers, in either pass order."""
        pdf = new_pdf()
        mask = Dictionary(
            S=Name.Alpha,
            G=_make_form_stream(pdf),
            TR=self._normalized_transfer(pdf, case),
        )
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        if limits_first:
            sanitize_structure_limits(pdf)
        sanitize_extgstate(pdf)
        sanitize_structure_limits(pdf)
        smask = pdf.pages[0].Resources.ExtGState.GS0.SMask
        assert ("/TR" in smask) is kept
        if kept:
            # The written function passes on its own, without normalization.
            assert extgstate._is_scalar_transfer_function(smask.TR)

    @pytest.mark.parametrize("case,kept", _NORMALIZATION_CASES)
    def test_full_conversion_keeps_only_valid_written_transfers(
        self, case, kept, tmp_path
    ) -> None:
        """After the whole pipeline, a soft-mask /TR is present only if valid."""
        pdf = new_pdf()
        mask = Dictionary(
            S=Name.Alpha,
            G=_make_form_stream(pdf),
            TR=self._normalized_transfer(pdf, case),
        )
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        pdf.pages[0].Contents = pdf.make_stream(b"/GS0 gs 0 0 10 10 re f")
        source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
        pdf.save(source)

        result = convert_to_pdfa(source, output, level="2b", validate=False)

        assert result.success
        with Pdf.open(output) as converted:
            smask = converted.pages[0].Resources.ExtGState.GS0.SMask
            assert isinstance(smask, Dictionary)
            assert ("/TR" in smask) is kept
            if kept:
                assert extgstate._is_scalar_transfer_function(smask.TR)

    @staticmethod
    def _calculator_mask(pdf: Pdf, program: bytes) -> Dictionary:
        """Create a soft mask whose transfer function runs the given program."""
        transfer = pdf.make_stream(program)
        transfer.FunctionType = 4
        transfer.Domain = transfer.Range = Array([0, 1])
        return Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=transfer)

    @pytest.mark.parametrize(
        "program",
        [
            b"{ 1 exch sub }",
            b"{ % comment\n 0.5 mul }",
            b"{ dup 0.5 lt { 2 mul } if }",
            b"{ dup 0.5 gt { pop 1 } { pop 0 } ifelse }",
            b"{ 0 eq { 1 } { 0 } ifelse }",
            b"{ dup 0 gt true and { 1 exch sub } if }",
            b"{ 1 1 index add exch pop }",
            b"{ 1 2 1 roll pop }",
            b"{ 2 mul 0 0 roll }",
            b"{ dup 1 copy pop pop }",
            b"{ 0 copy }",
            b"{ cvi 1 and cvr }",
            b"{ dup 0 gt { dup 1 gt { pop 1 } if } { pop 0 } ifelse }",
            b"{ 1 cvi copy pop }",
            b"{ 1.9 cvi copy pop }",
            b"{ -1.9 abs cvi copy pop }",
            b"{ 2 1 sub copy pop }",
            b"{ 0 0 add index exch pop }",
            b"{ 3 2 idiv copy pop }",
            b"{ 5 2 mod copy pop }",
            b"{ 3 1 and copy pop }",
            b"{ 1.5 floor cvi copy pop }",
            b"{ 0.5 round cvi copy pop }",
            b"{ -1.4 round cvi abs copy pop }",
            b"{ -1.5 round add }",
            b"{ 2 2 mul 3 sub copy pop }",
            b"{ 3 2 div floor cvi copy pop }",
            b"{ dup 0 gt { 1 } { 1 } ifelse copy pop }",
            b"{ cvi 2 idiv cvr }",
            b"{ cvi 2 mod cvr }",
            b"{ cvi 1 bitshift cvr }",
            b"{ cvi not cvr }",
            b"{ cvi 1 add 1 and cvr }",
            b"{ cvi dup 0 gt { pop 1 } if 1 and cvr }",
            b"{ pop 3 2 xor }",
            b"{ pop 4 sqrt }",
            b"{ pop 1 ln }",
            b"{ pop 100 log }",
            b"{ pop 2 3 exp }",
            b"{ pop 1 1 atan }",
            b"{ pop 90 sin }",
            b"{ 16 sqrt cvi 3 sub copy pop }",
            b"{ sqrt }",
            b"{ 1 add ln }",
            b"{ dup 0 gt { ln } { pop 0 } ifelse }",
            b"{ dup 0 eq { pop 0 } { 1 exch div } ifelse }",
            b"{ false { pop } if }",
            b"{ true { 1 exch sub } if }",
            b"{ true not { pop } if }",
            b"{ 1 2 lt { 1 exch sub } { pop } ifelse }",
            # A guard at the singular input, which is sampled as a literal.
            b"{ dup 0.25 eq { pop 0 } { 0.25 sub 1 exch div } ifelse }",
            b"{ pop -2147483648.0 cvi }",
            # Literal bit shifts are folded, so they can drive stack operators.
            b"{ 1 1 bitshift 1 sub copy pop }",
            b"{ 0 1 bitshift copy }",
            b"{ 8 -3 bitshift copy pop }",
            b"{ pop 2147483647 1 and cvr }",
            # Form feed, tab, CR and LF are PDF white-space characters.
            b"{\x0cpop\t1\r\n}",
            # Large reals within the limit of ISO 32000-1, Table C.1.
            b"{ pop 1" + b"0" * 38 + b".0 }",
            b"{ 1" + b"0" * 18 + b".0 mul 1" + b"0" * 18 + b".0 mul }",
            b"{ pop 10 38 exp }",
        ],
    )
    def test_calculator_with_one_numeric_result_is_preserved(self, program) -> None:
        """Conditionals and literal stack operands can be proven correct."""
        pdf = new_pdf()
        mask = self._calculator_mask(pdf, program)
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert pdf.pages[0].Resources.ExtGState.GS0.SMask.TR.read_bytes() == program

    @pytest.mark.parametrize(
        "program",
        [
            b"{ pop }",
            b"{ dup }",
            b"{ add }",
            b"{ 1 }",
            b"{ pop true }",
            b"{ 0 gt }",
            b"{ true add }",
            b"{ true 1 eq pop }",
            b"{ { 2 mul } if }",
            b"{ 1 { 2 mul } if }",
            b"{ { 2 mul } }",
            b"{ dup 0 gt { pop } if }",
            b"{ dup 0 gt { pop } { pop 1 } ifelse }",
            b"{ dup 0 gt { 2 mul } { 3 mul } { 4 mul } ifelse }",
            b"{ dup 0 gt { pop 1 } { pop true } ifelse }",
            b"{ dup copy }",
            b"{ 5 copy }",
            b"{ 5 index }",
            b"{ 3 1 roll }",
            b"{ dup roll }",
            b"{ 1.0 copy pop }",
            b"{ 1 cvr copy pop }",
            b"{ -1.5 round cvi abs copy pop }",
            b"{ -0.5 round cvi 1 add copy pop }",
            b"{ pop 1.5 2 and }",
            b"{ pop 1.5 2 or }",
            b"{ pop 1.5 2 xor }",
            b"{ pop 1.5 not }",
            b"{ pop 1.5 2 idiv }",
            b"{ 1 and }",
            b"{ not }",
            b"{ 2 idiv }",
            b"{ 2 mod }",
            b"{ 1 bitshift }",
            b"{ cvi 1.5 or }",
            b"{ cvr 1 and }",
            b"{ pop true 1 and }",
            b"{ pop -1 sqrt }",
            b"{ pop 0 ln }",
            b"{ pop -1 ln }",
            b"{ pop 0 log }",
            b"{ pop 0 -1 exp }",
            b"{ pop -1 0.5 exp }",
            b"{ pop 0 0 atan }",
            b"{ ln }",
            b"{ log }",
            b"{ 1 exch div }",
            b"{ 0.5 sub ln }",
            b"{ 0.5 sub 1 exch div }",
            b"{ -1 exp }",
            b"{ true { pop } if }",
            b"{ false { 1 exch sub } { pop } ifelse }",
            b"{ dup 1 gt { pop } if pop }",
            b"{ true copy }",
            b"{ true true roll }",
            b"{ cvi copy }",
            b"{ 1 0 idiv pop }",
            b"{ 1 0 div pop }",
            b"{ 1 0 mod pop }",
            b"{ 2147483647 1 add index }",
            b"{ dup 0 gt { 1 } { 0 } ifelse copy pop }",
            b"{ dup 0 gt { 1 } { 1.0 } ifelse copy pop }",
            b"{" + b" dup" * 150 + b" }",
            b"{ " * 40 + b"}" * 40,
            b"{ 1 exch sub } }",
            b"{ 1 exch sub } { 1 exch sub }",
            b"1 exch sub",
            # Undefined between the ends and the middle of the domain.
            b"{ 0.25 sub 1 exch div }",
            b"{ -0.25 add 1 exch div }",
            b"{ 3 mul 1 sub 1 exch div }",
            b"{ 0.3 sub ln }",
            # cvi and integer-only operators cannot leave the integer range.
            b"{ pop 2147483648.0 cvi }",
            b"{ pop -2147483649.0 cvi }",
            b"{ pop -2147483648 -1 idiv }",
            # Renderers disagree on these shifts, so their result is not known.
            b"{ -1 1 bitshift copy pop }",
            b"{ 1 31 bitshift copy pop }",
            # An integer literal beyond the 32-bit limit is a real.
            b"{ pop 2147483648 1 and cvr }",
            # A vertical tab is not PDF white space, so this is one bad token.
            b"{ pop\x0b1 }",
            # Non-finite values, and reals beyond the limit of ISO 32000-1, Table C.1.
            b"{ pop " + b"1" * 400 + b".0 }",
            b"{ pop " + b"1" * 400 + b" }",
            b"{ pop 4" + b"0" * 38 + b".0 }",
            b"{ pop 1" + b"0" * 200 + b".0 1" + b"0" * 200 + b".0 mul 0 mul }",
            b"{ pop 1" + b"0" * 30 + b".0 dup mul }",
            b"{ pop 1" + b"0" * 30 + b".0 0." + b"0" * 29 + b"1 div }",
            b"{ pop 10 39 exp }",
            b"{ pop 1" + b"0" * 38 + b".0 10 mul }",
            b"{ pop -3" + b"0" * 38 + b".0 dup add }",
            b"{ 1" + b"0" * 30 + b".0 mul 1" + b"0" * 30 + b".0 mul }",
        ],
    )
    def test_calculator_without_provable_result_is_removed(self, program) -> None:
        """Malformed stack effects cannot produce a defined mask mapping."""
        pdf = new_pdf()
        mask = self._calculator_mask(pdf, program)
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        state = pdf.pages[0].Resources.ExtGState.GS0
        assert "/TR" not in state.SMask
        assert state.SMask.G == mask.G

    def test_smask_dict_invalid_bc_removed(self) -> None:
        """Removes malformed /BC (non-Array) from SMask dict."""
        pdf = new_pdf()
        smask_dict = Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), BC=Name("/Bad"))
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert isinstance(gs_out.SMask, Dictionary)
        assert "/BC" not in gs_out.SMask

    def test_smask_dict_valid_with_bc(self) -> None:
        """Preserves valid SMask dict with /BC array."""
        pdf = new_pdf()
        smask_dict = Dictionary(
            S=Name.Alpha,
            G=_make_form_stream(pdf),
            BC=Array([0, 0, 0]),
        )
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert isinstance(gs_out.SMask, Dictionary)
        assert "/BC" in gs_out.SMask

    def test_smask_dict_valid_luminosity(self) -> None:
        """Preserves valid SMask dict with /S /Luminosity."""
        pdf = new_pdf()
        smask_dict = Dictionary(S=Name.Luminosity, G=_make_form_stream(pdf))
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert isinstance(gs_out.SMask, Dictionary)
        assert str(gs_out.SMask.S) == "/Luminosity"


class TestTransparencyCombined:
    """Tests for combined transparency properties."""

    def test_all_transparency_props_valid(self) -> None:
        """PDF with all valid transparency properties returns zero fixes."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Name.Multiply,
            CA=0.7,
            ca=0.3,
            SMask=Name("/None"),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_multiple_transparency_fixes(self) -> None:
        """Fixes multiple transparency issues in one ExtGState."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Name("/Invalid"),
            CA=5.0,
            ca=-2.0,
            SMask=Name("/Bad"),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 4

        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert str(gs_out.BM) == "/Normal"
        assert float(gs_out.CA) == pytest.approx(1.0)
        assert float(gs_out.ca) == pytest.approx(0.0)
        assert str(gs_out.SMask) == "/None"

    def test_transparency_with_forbidden_entries(self) -> None:
        """Fixes both forbidden entries and invalid transparency properties."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(
            Type=Name.ExtGState,
            TR=tr_stream,
            BM=Name("/Bogus"),
            CA=3.0,
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 3

        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        assert "/TR" not in gs_out
        assert str(gs_out.BM) == "/Normal"
        assert float(gs_out.CA) == pytest.approx(1.0)


class TestOverprintIccBasedCmyk:
    """Tests for PDF/A 6.2.4.2 overprint handling with ICCBased CMYK."""

    def _add_icc_cmyk_colorspace(self, pdf: Pdf) -> None:
        """Attach an ICCBased CMYK color space to page resources."""
        icc_stream = pikepdf.Stream(pdf, get_cmyk_profile())
        icc_stream.N = 4
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        pdf.pages[0].Resources[Name.ColorSpace] = Dictionary(CS0=icc_cs)

    def test_opm_reset_when_stroke_overprint_enabled(self) -> None:
        """Sets /OPM 1 -> 0 when /OP true and ICCBased CMYK is present."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OPM=1, OP=True)
        _make_pdf_with_extgstate(pdf, gs)
        self._add_icc_cmyk_colorspace(pdf)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 0

    def test_opm_reset_when_fill_overprint_enabled(self) -> None:
        """Sets /OPM 1 -> 0 when /op true and ICCBased CMYK is present."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OPM=1, op=True)
        _make_pdf_with_extgstate(pdf, gs)
        self._add_icc_cmyk_colorspace(pdf)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 0

    def test_opm_reset_when_overprint_is_in_another_graphics_state(self) -> None:
        """Combines independently persistent OPM and overprint parameters."""
        pdf = new_pdf()
        _make_pdf_with_extgstate(
            pdf,
            Dictionary(Type=Name.ExtGState, OPM=1),
        )
        pdf.pages[0].Resources.ExtGState.GS1 = Dictionary(
            Type=Name.ExtGState,
            OP=True,
        )
        self._add_icc_cmyk_colorspace(pdf)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 0

    def test_opm_unchanged_without_iccbased_cmyk(self) -> None:
        """Leaves OPM untouched when ICCBased CMYK is not used."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OPM=1, OP=True)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 1

    def test_opm_unchanged_when_overprint_not_enabled(self) -> None:
        """Leaves OPM untouched when /OP and /op are false."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OPM=1, OP=False, op=False)
        _make_pdf_with_extgstate(pdf, gs)
        self._add_icc_cmyk_colorspace(pdf)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 1

    def test_opm_conservative_cross_page(self) -> None:
        """OPM is reset even when ICCBased CMYK is on a different page.

        The document-wide flag is intentionally conservative: shared
        ExtGState objects could be referenced from any page.
        """
        pdf = new_pdf()
        # Page 1: ExtGState with OPM=1 and overprint, no ICC CMYK
        gs = Dictionary(Type=Name.ExtGState, OPM=1, OP=True)
        _make_pdf_with_extgstate(pdf, gs)

        # Page 2: has ICC CMYK but no problematic ExtGState
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(),
            )
        )
        pdf.pages.append(page2)
        icc_stream = pikepdf.Stream(pdf, get_cmyk_profile())
        icc_stream.N = 4
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        pdf.pages[1].Resources[Name.ColorSpace] = Dictionary(CS0=icc_cs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert int(pdf.pages[0].Resources.ExtGState.GS0.OPM) == 0


class TestShadingTransferFunctions:
    """Tests for /TR and /TR2 removal from Shading dictionaries."""

    def test_tr_removed_from_shading_in_resources(self) -> None:
        """Removes /TR from a Shading dictionary in page Resources."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR=pdf.make_stream(b"{ }"),
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Shading=Dictionary(Sh0=shading),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/TR" not in pdf.pages[0].Resources.Shading.Sh0

    def test_tr2_removed_from_shading_in_resources(self) -> None:
        """Removes /TR2 from a Shading dictionary in page Resources."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR2=pdf.make_stream(b"{ }"),
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Shading=Dictionary(Sh0=shading),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/TR2" not in pdf.pages[0].Resources.Shading.Sh0

    def test_both_tr_and_tr2_removed_from_shading(self) -> None:
        """Removes both /TR and /TR2 from same Shading dictionary."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR=pdf.make_stream(b"{ }"),
            TR2=pdf.make_stream(b"{ }"),
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Shading=Dictionary(Sh0=shading),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 2
        sh = pdf.pages[0].Resources.Shading.Sh0
        assert "/TR" not in sh
        assert "/TR2" not in sh

    def test_function_preserved_in_shading(self) -> None:
        """Preserves /Function in Shading (defines shading colour, not TR)."""
        pdf = new_pdf()
        func_stream = pdf.make_stream(b"{ }")
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            Function=func_stream,
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Shading=Dictionary(Sh0=shading),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert "/Function" in pdf.pages[0].Resources.Shading.Sh0

    def test_clean_shading_unchanged(self) -> None:
        """Clean Shading dictionary requires no fixes."""
        pdf = new_pdf()
        shading = Dictionary(ShadingType=2, ColorSpace=Name.DeviceRGB)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Shading=Dictionary(Sh0=shading),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_tr_removed_from_shading_in_form_xobject(self) -> None:
        """Removes /TR from Shading inside a Form XObject's Resources."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR=pdf.make_stream(b"{ }"),
        )
        form_stream = pdf.make_stream(b"q Q")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

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

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1

    def test_tr_removed_from_pattern_type2_shading(self) -> None:
        """Removes /TR from Shading inside a PatternType 2 pattern."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR=pdf.make_stream(b"{ }"),
        )
        pattern = Dictionary(PatternType=2, Shading=shading)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P0=pattern),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        pat_shading = pdf.pages[0].Resources.Pattern.P0.Shading
        assert "/TR" not in pat_shading

    def test_tr2_removed_from_pattern_type2_shading(self) -> None:
        """Removes /TR2 from Shading inside a PatternType 2 pattern."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR2=pdf.make_stream(b"{ }"),
        )
        pattern = Dictionary(PatternType=2, Shading=shading)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P0=pattern),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        pat_shading = pdf.pages[0].Resources.Pattern.P0.Shading
        assert "/TR2" not in pat_shading

    def test_pattern_type1_shading_not_affected(self) -> None:
        """PatternType 1 (tiling) patterns are not checked for shading /TR."""
        pdf = new_pdf()
        pattern_stream = pdf.make_stream(b"q Q")
        pattern_stream[Name.PatternType] = 1
        pattern_stream[Name.PaintType] = 1
        pattern_stream[Name.TilingType] = 1
        pattern_stream[Name.BBox] = Array([0, 0, 10, 10])
        pattern_stream[Name.XStep] = 10
        pattern_stream[Name.YStep] = 10

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Pattern=Dictionary(P0=pattern_stream),
                ),
            )
        )
        pdf.pages.append(page)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_sanitize_shadings_in_resources_directly(self) -> None:
        """Direct call to _sanitize_shadings_in_resources works."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            TR=pdf.make_stream(b"{ }"),
            Function=pdf.make_stream(b"{ }"),
        )
        resources = Dictionary(Shading=Dictionary(Sh0=shading))

        removed = _sanitize_shadings_in_resources(resources)
        assert removed == 1
        assert "/TR" not in shading
        assert "/Function" in shading


class TestSMaskGFormXObject:
    """Tests for SMask /G must be a Form XObject (ISO 19005-2, §6.4)."""

    def test_smask_g_form_xobject_preserved(self) -> None:
        """Preserves SMask when /G is a proper Form XObject."""
        pdf = new_pdf()
        g_stream = _make_form_stream(pdf)
        smask_dict = Dictionary(S=Name.Alpha, G=g_stream)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert isinstance(pdf.pages[0].Resources.ExtGState.GS0.SMask, Dictionary)

    def test_smask_g_stream_without_form_subtype_replaced(self) -> None:
        """Replaces SMask with /None when /G stream lacks /Subtype /Form."""
        pdf = new_pdf()
        g_stream = pdf.make_stream(b"q Q")  # No /Subtype
        smask_dict = Dictionary(S=Name.Alpha, G=g_stream)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"

    def test_smask_g_image_xobject_replaced(self) -> None:
        """Replaces SMask with /None when /G is an Image XObject."""
        pdf = new_pdf()
        g_stream = pdf.make_stream(b"\xff" * 10)
        g_stream[Name.Subtype] = Name.Image
        g_stream[Name.Width] = 1
        g_stream[Name.Height] = 1
        g_stream[Name.BitsPerComponent] = 8
        g_stream[Name.ColorSpace] = Name.DeviceGray
        smask_dict = Dictionary(S=Name.Alpha, G=g_stream)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"

    def test_smask_g_wrong_subtype_replaced(self) -> None:
        """Replaces SMask with /None when /G has unexpected /Subtype."""
        pdf = new_pdf()
        g_stream = pdf.make_stream(b"data")
        g_stream[Name.Subtype] = Name("/PS")
        smask_dict = Dictionary(S=Name.Alpha, G=g_stream)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert str(pdf.pages[0].Resources.ExtGState.GS0.SMask) == "/None"


class TestBlendModeArrayPartialFix:
    """Tests for /BM array: preserve valid entries, fix invalid ones."""

    def test_mixed_valid_invalid_preserves_valid(self) -> None:
        """Replaces only invalid entries, preserving valid blend modes."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Array([Name.Multiply, Name("/InvalidMode"), Name.Screen]),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Array)
        assert len(bm) == 3
        assert str(bm[0]) == "/Multiply"
        assert str(bm[1]) == "/Normal"
        assert str(bm[2]) == "/Screen"

    def test_all_invalid_replaced_with_single_name(self) -> None:
        """All-invalid array is replaced with single /Normal Name."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Array([Name("/Bad1"), Name("/Bad2")]),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Name)
        assert str(bm) == "/Normal"

    def test_single_invalid_in_array_replaced(self) -> None:
        """Single invalid entry in a multi-entry array is replaced."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Array([Name.Overlay, Name("/Bogus"), Name.Darken, Name.Lighten]),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Array)
        assert str(bm[0]) == "/Overlay"
        assert str(bm[1]) == "/Normal"
        assert str(bm[2]) == "/Darken"
        assert str(bm[3]) == "/Lighten"

    def test_all_valid_array_preserved(self) -> None:
        """Array with all valid blend modes is preserved unchanged."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Array([Name.Multiply, Name.Screen]),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Array)
        assert str(bm[0]) == "/Multiply"
        assert str(bm[1]) == "/Screen"

    def test_single_element_invalid_array(self) -> None:
        """Single-element array with invalid mode becomes /Normal Name."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            BM=Array([Name("/Invalid")]),
        )
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        bm = pdf.pages[0].Resources.ExtGState.GS0.BM
        assert isinstance(bm, Name)
        assert str(bm) == "/Normal"
