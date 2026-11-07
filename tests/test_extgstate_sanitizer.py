# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Extended Graphics State sanitization for PDF/A compliance."""

from decimal import Decimal
from io import BytesIO

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import get_cmyk_profile
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers.extgstate import (
    _sanitize_shadings_in_resources,
    sanitize_extgstate,
    sanitize_overprint_usage,
)


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
        ],
    )
    def test_invalid_soft_mask_transfer_uses_identity_default(self, transfer) -> None:
        """Malformed transfers are removed without discarding the valid mask group."""
        pdf = new_pdf()
        mask = Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=transfer)
        _make_pdf_with_extgstate(pdf, Dictionary(SMask=mask))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert "/TR" not in pdf.pages[0].Resources.ExtGState.GS0.SMask
        assert pdf.pages[0].Resources.ExtGState.GS0.SMask.G == mask.G

    @pytest.mark.parametrize("kind", ["short_samples", "bad_calculator", "cycle"])
    def test_invalid_transfer_programs_do_not_survive(self, kind) -> None:
        """Check sampled payloads, calculator syntax, and recursive stitching graphs."""
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
        else:
            transfer = pdf.make_stream(
                b"\x00" if kind == "short_samples" else b"{ bogus }"
            )
            transfer.FunctionType = 0 if kind == "short_samples" else 4
            transfer.Domain = transfer.Range = Array([0, 1])
            if kind == "short_samples":
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
    """Overprint repairs depend on actual paints, never resource presence."""

    def _page(self, pdf: Pdf, content: bytes, **gs_values):
        """Create a page with a CMYK ICC space and configurable graphics state."""
        _make_pdf_with_extgstate(pdf, Dictionary(Type=Name.ExtGState, **gs_values))
        page = pdf.pages[-1]
        profile = pdf.make_stream(get_cmyk_profile())
        profile.N = 4
        page.Resources.ColorSpace = Dictionary(CS0=Array([Name.ICCBased, profile]))
        page.Contents = pdf.make_stream(content)
        return page

    def _assert_guard(self, owner, operator: str, count: int = 1) -> None:
        """Require local overprint guards around the expected painting operations."""
        instructions = list(pikepdf.parse_content_stream(owner))
        guarded = 0
        for i, instruction in enumerate(instructions):
            if (
                str(instruction.operator) != operator
                or i == 0
                or i + 1 == len(instructions)
            ):
                continue
            before, after = instructions[i - 1], instructions[i + 1]
            if str(before.operator) != "gs" or str(after.operator) != "gs":
                continue
            states = owner.Resources.ExtGState
            assert states[str(before.operands[0])].OPM == 0
            assert states[str(after.operands[0])].OPM == 1
            guarded += 1
        assert guarded == count

    @pytest.mark.parametrize(
        "paint,colour,flags",
        [
            (b"S", b"CS", {"OP": True, "op": False}),
            (b"f", b"cs", {"OP": False, "op": True}),
        ],
    )
    def test_only_prohibited_paint_is_repaired(self, paint, colour, flags) -> None:
        """Repair the painted side whose effective overprint state is prohibited."""
        pdf = new_pdf()
        page = self._page(
            pdf, b"/GS0 gs /CS0 " + colour + b" 0 0 20 20 re " + paint, OPM=1, **flags
        )
        original_state = page.Resources.ExtGState.GS0
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert original_state.OPM == 1
        assert page.Resources.ExtGState.GS0 == original_state
        self._assert_guard(page, paint.decode())
        # The restoration must also make a second sanitizer pass a no-op.
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    @pytest.mark.parametrize(
        "value,mode,fixes",
        [(Decimal("1.0"), 1, 2), (Decimal("1"), 1, 2), (Decimal("0.0"), 0, 1)],
    )
    def test_equivalent_real_mode_becomes_an_integer(self, value, mode, fixes) -> None:
        """Keep the selected mode while storing the required integer type."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", op=True)
        page.Resources.ExtGState.GS0.OPM = value
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == fixes
        state = page.Resources.ExtGState.GS0
        assert isinstance(state.OPM, int) and state.OPM == mode
        assert b"/OPM 1.0" not in state.unparse()
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    def test_real_mode_without_iccbased_cmyk_is_normalized(self) -> None:
        """The stored type does not depend on whether overprint analysis runs."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OP=True)
        gs.OPM = Decimal("1.0")
        _make_pdf_with_extgstate(pdf, gs)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert isinstance(pdf.pages[0].Resources.ExtGState.GS0.OPM, int)

    @pytest.mark.parametrize(
        "value", [Decimal("0.5"), Decimal("2.0"), 2, -1, True, Name.One]
    )
    def test_undefined_mode_is_removed(self, value) -> None:
        """An optional entry without a defined mode cannot remain in the output."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", op=True)
        page.Resources.ExtGState.GS0.OPM = value
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert "/OPM" not in page.Resources.ExtGState.GS0
        assert page.Resources.ExtGState.GS0.op is True
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    @pytest.mark.parametrize("value", [1, 0])
    def test_valid_integer_mode_is_unchanged(self, value) -> None:
        """Repairing malformed modes must not touch the two defined values."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, OP=False)
        gs.OPM = value
        _make_pdf_with_extgstate(pdf, gs)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert pdf.pages[0].Resources.ExtGState.GS0.OPM == value

    @pytest.mark.parametrize(
        "value,flag,fixes",
        [
            (1, True, 2),
            (Decimal("1.0"), True, 2),
            (0, False, 1),
            (Decimal("0.0"), False, 1),
        ],
    )
    def test_numeric_overprint_flag_becomes_a_boolean(self, value, flag, fixes) -> None:
        """Unambiguous flags keep their meaning, and enabled ones are still repaired."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", OPM=1)
        page.Resources.ExtGState.GS0.op = value
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == fixes
        assert page.Resources.ExtGState.GS0.op is flag
        if flag:
            self._assert_guard(page, "f")
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    @pytest.mark.parametrize("key", ["/OP", "/op"])
    @pytest.mark.parametrize("value", [2, -1, Decimal("0.5"), Name.Yes])
    def test_undefined_overprint_flag_is_removed(self, key, value) -> None:
        """A flag without a defined meaning cannot force the original-input fallback."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", OPM=1)
        page.Resources.ExtGState.GS0[key] = value
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert key not in page.Resources.ExtGState.GS0
        assert page.Resources.ExtGState.GS0.OPM == 1

    @pytest.mark.parametrize("value", [True, False])
    def test_boolean_overprint_flag_is_unchanged(self, value) -> None:
        """The repair leaves correctly typed flags alone."""
        pdf = new_pdf()
        _make_pdf_with_extgstate(pdf, Dictionary(Type=Name.ExtGState, OP=value))
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert pdf.pages[0].Resources.ExtGState.GS0.OP is value

    @pytest.mark.parametrize("key,value", [("/OP", 1), ("/op", 1), ("/OP", 2)])
    def test_conversion_does_not_fall_back_for_numeric_flags(
        self, key, value, tmp_path
    ) -> None:
        """These documents converted before point-of-use overprint analysis."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", OPM=1)
        page.Resources.ExtGState.GS0[key] = value
        source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
        pdf.save(source)
        result = convert_to_pdfa(source, output, level="3b")
        assert result.target_produced and not result.skipped, result

    def test_overprint_analysis_rejects_unrepaired_flag(self) -> None:
        """A dictionary that sanitization did not reach still fails closed."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", OPM=1)
        page.Resources.ExtGState.GS0.OP = 1
        with pytest.raises(ConversionError, match="non-boolean /OP"):
            sanitize_overprint_usage(pdf)

    def test_overprint_analysis_rejects_unrepaired_real_mode(self) -> None:
        """A dictionary that sanitization did not reach still fails closed."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /CS0 cs 0 0 20 20 re f", op=True)
        page.Resources.ExtGState.GS0.OPM = Decimal("1.0")
        with pytest.raises(ConversionError, match="invalid /OPM"):
            sanitize_overprint_usage(pdf)

    def test_independently_persistent_graphics_states(self) -> None:
        """Combine overprint parameters selected by separate ExtGState operations."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /GS1 gs /CS0 cs 0 0 20 20 re f", OPM=1)
        page.Resources.ExtGState.GS1 = Dictionary(OP=True)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert page.Resources.ExtGState.GS0.OPM == 1
        self._assert_guard(page, "f")

    @pytest.mark.parametrize(
        "colorspace,data,cmyk",
        [
            (b"/CMYK", b"000000ff>", True),
            (b"/RGB", b"ff0000>", False),
            (b"/G", b"80>", False),
            (b"[/I /G 1 <00ff>]", b"00>", False),
            (b"[/I /CMYK 1 <00000000000000ff>]", b"00>", True),
        ],
    )
    def test_inline_abbreviations_use_effective_color_space(
        self, colorspace, data, cmyk
    ) -> None:
        """Recognize inline abbreviations and guard only prohibited paints."""
        pdf = new_pdf()
        content = (
            b"/GS0 gs BI /W 1 /H 1 /BPC 8 /F /AHx /CS "
            + colorspace
            + b" ID "
            + data
            + b" EI"
        )
        page = self._page(pdf, content, OPM=1, op=True)
        page.Resources.ColorSpace.DefaultCMYK = page.Resources.ColorSpace.CS0
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == int(cmyk)
        if cmyk:
            self._assert_guard(page, "INLINE IMAGE")
        else:
            assert page.Contents.read_bytes() == content

    @pytest.mark.parametrize("mode", range(8))
    @pytest.mark.parametrize("show", [b"(A) Tj", b"[(A)] TJ", b"(A) '", b'0 0 (A) "'])
    def test_type3_overprint_ignores_only_invisible_text(self, mode, show) -> None:
        """Only mode 3 suppresses Type3 glyph paints, for every text-show operator."""
        pdf = new_pdf()
        content = b"/GS0 gs BT /F 12 Tf " + str(mode).encode() + b" Tr " + show + b" ET"
        page = self._page(pdf, content, OPM=1, OP=True)
        glyph_data = b"600 0 d0 /CS0 cs 0 0 0 1 scn 0 0 20 20 re f"
        glyph = pdf.make_stream(glyph_data)
        page.Resources.Font = Dictionary(
            F=Dictionary(
                Subtype=Name.Type3,
                CharProcs=Dictionary(A=glyph),
                Resources=Dictionary(ColorSpace=page.Resources.ColorSpace),
            )
        )
        if mode == 3:
            assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        else:
            with pytest.raises(ConversionError, match="Type3 glyph"):
                sanitize_extgstate(pdf)
        assert page.Contents.read_bytes() == content
        assert glyph.read_bytes() == glyph_data
        assert page.Resources.ExtGState.GS0.OPM == 1

    def test_repeated_type3_proofs_do_not_exhaust_context_budget(self) -> None:
        """A large harmless Type3 text page needs one proof per glyph context."""
        from pdftopdfa.sanitizers.overprint import _Analyzer, _State

        pdf = new_pdf()
        content = b"/GS0 gs BT /F 12 Tf " + b"(A) Tj " * 3000 + b"ET"
        page = self._page(pdf, content, OPM=1, OP=True)
        charprocs = Dictionary()
        for index in range(50):
            charprocs[f"/Glyph{index}"] = pdf.make_stream(b"0 g 0 0 1 1 re f")
        page.Resources.Font = Dictionary(
            F=Dictionary(
                Subtype=Name.Type3, CharProcs=charprocs, Resources=Dictionary()
            )
        )
        analyzer = _Analyzer()
        analyzer.walk(page, page.Resources, _State())
        assert analyzer.calls == 51
        assert not any(plan.repairs for plan in analyzer.plans.values())
        assert page.Contents.read_bytes() == content

    def test_repeated_soft_mask_uses_share_equivalent_analysis(self) -> None:
        """Repeated mask use is cached without masking a later conflicting context."""
        from pdftopdfa.sanitizers.overprint import _Analyzer, _State

        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs " + b"0 0 20 20 re f " * 2000, OPM=1, OP=False)
        mask = pdf.make_stream(b"/CS0 cs 0 0 20 20 re f")
        mask.Subtype = Name.Form
        mask.Resources = Dictionary(ColorSpace=page.Resources.ColorSpace)
        page.Resources.ExtGState.GS0.SMask = Dictionary(S=Name.Alpha, G=mask)
        analyzer = _Analyzer()
        analyzer.walk(page, page.Resources, _State())
        assert analyzer.calls == 2
        page.Resources.ExtGState.On = Dictionary(OP=True)
        page.Contents = pdf.make_stream(
            page.Contents.read_bytes() + b"/On gs 0 0 20 20 re f"
        )
        with pytest.raises(ConversionError, match="shared form"):
            sanitize_extgstate(pdf)

    @pytest.mark.parametrize(
        "content,flags",
        [
            (b"/GS0 gs 0 0 0 1 k 0 0 20 20 re f", {"OP": True}),
            (b"/GS0 gs /CS0 cs 0 0 20 20 re f", {"OP": False, "op": False}),
            (b"/GS0 gs /CS0 cs 0 0 20 20 re f", {"OP": True, "op": False}),
            (b"/GS0 gs /CS0 CS 0 0 20 20 re S", {"OP": False, "op": True}),
            (b"", {"OP": True}),
        ],
    )
    def test_allowed_or_unused_combinations_unchanged(self, content, flags) -> None:
        """Preserve allowed and unused color-space and overprint combinations."""
        pdf = new_pdf()
        page = self._page(pdf, content, OPM=1, **flags)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert page.Resources.ExtGState.GS0.OPM == 1
        assert page.Contents.read_bytes() == content

    def test_icc_on_different_page_does_not_reset_overprint(self) -> None:
        """An unrelated page cannot change valid DeviceCMYK overprint."""
        pdf = new_pdf()
        first = self._page(pdf, b"/GS0 gs 0 0 0 1 k 0 0 20 20 re f", OPM=1, OP=True)
        del first.Resources["/ColorSpace"]
        second = self._page(pdf, b"/CS0 cs 0 0 20 20 re f", OPM=0)
        first_data, second_data = (
            first.Contents.read_bytes(),
            second.Contents.read_bytes(),
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert first.Resources.ExtGState.GS0.OPM == 1
        assert first.Contents.read_bytes() == first_data
        assert second.Contents.read_bytes() == second_data

    def test_q_q_restore_and_content_array_share_graphics_state(self) -> None:
        """Honor saved state across the combined page content-stream array."""
        pdf = new_pdf()
        page = self._page(pdf, b"", OPM=1, OP=True)
        page.Resources.ExtGState.Off = Dictionary(OP=False)
        page.Contents = Array(
            [
                pdf.make_stream(b"/GS0 gs /CS0 cs q /Off gs 0 0 20 20 re f"),
                pdf.make_stream(b"Q 30 30 20 20 re f"),
            ]
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "f")

    def test_shared_gs_preserves_device_cmyk_paint_on_same_page(self) -> None:
        """Leave shared settings intact when only one paint needs a repair."""
        pdf = new_pdf()
        page = self._page(
            pdf,
            b"/GS0 gs 0 0 0 1 k 0 0 20 20 re f "
            b"/CS0 cs 30 30 20 20 re f 0 0 0 1 k 60 60 20 20 re f",
            OPM=1,
            OP=True,
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        assert page.Resources.ExtGState.GS0.OPM == 1
        self._assert_guard(page, "f")

    def test_nested_form_uses_callers_state_and_inherited_resources(self) -> None:
        """Resolve inherited resources and graphics state inside a nested Form."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /Fm Do", OPM=1, OP=True)
        form = pdf.make_stream(b"/CS0 cs 0 0 20 20 re f")
        form.Subtype = Name.Form
        form.BBox = Array([0, 0, 20, 20])
        page.Resources.XObject = Dictionary(Fm=form)
        original_page = page.Contents.read_bytes()
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(form, "f")
        assert page.Contents.read_bytes() == original_page
        assert page.Resources.ExtGState.GS0.OPM == 1

    def test_shared_form_with_conflicting_states_fails_without_mutation(self) -> None:
        """Reject incompatible shared-Form repairs before modifying any stream."""
        from pdftopdfa.exceptions import ConversionError

        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /Fm Do /Off gs /Fm Do", OPM=1, OP=True)
        page.Resources.ExtGState.Off = Dictionary(OP=False)
        content = b"/CS0 cs 0 0 20 20 re f"
        form = pdf.make_stream(content)
        form.Subtype = Name.Form
        form.BBox = Array([0, 0, 20, 20])
        page.Resources.XObject = Dictionary(Fm=form)
        with pytest.raises(ConversionError, match="shared form"):
            sanitize_extgstate(pdf)
        assert form.read_bytes() == content
        assert page.Resources.ExtGState.GS0.OPM == 1

    def test_default_cmyk_mapping_and_inherited_page_resources(self) -> None:
        """Resolve effective CMYK through inherited DefaultCMYK resources."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs 0 0 0 1 k 0 0 20 20 re f", OPM=1, OP=True)
        page.Resources.ColorSpace.DefaultCMYK = page.Resources.ColorSpace.CS0
        pdf.Root.Pages.Resources = page.Resources
        del page.obj["/Resources"]
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "f")
        assert pdf.Root.Pages.Resources.ExtGState.GS0.OPM == 1

    def test_mixed_fill_stroke_modes_fail_without_corrupting_device_paint(self) -> None:
        """Reject combined paints that would require two overprint modes."""
        from pdftopdfa.exceptions import ConversionError

        pdf = new_pdf()
        content = b"/GS0 gs /CS0 cs 0 0 0 1 K 0 0 20 20 re B"
        page = self._page(pdf, content, OPM=1, OP=True)
        with pytest.raises(ConversionError, match="different overprint modes"):
            sanitize_extgstate(pdf)
        assert page.Contents.read_bytes() == content
        assert page.Resources.ExtGState.GS0.OPM == 1

    @pytest.mark.parametrize("kind", ["image", "shading"])
    def test_image_and_shading_use_their_own_colourspace(self, kind) -> None:
        """Classify image and shading paints by their own source color spaces."""
        pdf = new_pdf()
        page = self._page(
            pdf,
            b"/GS0 gs /Paint " + (b"Do" if kind == "image" else b"sh"),
            OPM=1,
            op=True,
        )
        if kind == "image":
            obj = pdf.make_stream(bytes([0, 0, 0, 255]))
            obj.Subtype = Name.Image
            obj.Width = obj.Height = 1
            obj.BitsPerComponent = 8
            obj.ColorSpace = page.Resources.ColorSpace.CS0
            page.Resources.XObject = Dictionary(Paint=obj)
        else:
            obj = Dictionary(ShadingType=2, ColorSpace=page.Resources.ColorSpace.CS0)
            page.Resources.Shading = Dictionary(Paint=obj)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "Do" if kind == "image" else "sh")

    def test_op_without_op_key_replaces_both_overprint_flags(self) -> None:
        """Apply OP to both sides when the dictionary omits a separate op entry."""
        pdf = new_pdf()
        page = self._page(
            pdf, b"/GS0 gs /Enable gs /CS0 cs 0 0 20 20 re f", OPM=1, OP=False, op=False
        )
        page.Resources.ExtGState.Enable = Dictionary(OP=True)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "f")

    def test_nonstroking_update_does_not_override_stroking_flag(self) -> None:
        """Updating nonstroking overprint must retain the stroking parameter."""
        pdf = new_pdf()
        page = self._page(
            pdf, b"/GS0 gs /DisableFill gs /CS0 CS 0 0 20 20 re S", OPM=1, OP=True
        )
        page.Resources.ExtGState.DisableFill = Dictionary(op=False)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "S")

    @pytest.mark.parametrize("kind", ["pattern", "type3"])
    @pytest.mark.parametrize("colour", [b"1 0 0 rg", b"0 0 0 1 k"])
    def test_unrelated_pattern_and_type3_programs_are_preserved(
        self, kind, colour
    ) -> None:
        """Preserve harmless pattern and glyph programs with unrelated ICC resources."""
        pdf = new_pdf()
        page = self._page(pdf, b"", OPM=1, OP=True)
        program_data = colour + b" 0 0 20 20 re f"
        program = pdf.make_stream(program_data)
        program.Resources = Dictionary()
        if kind == "pattern":
            program.PatternType = 1
            program.PaintType = 1
            program.TilingType = 1
            program.BBox = Array([0, 0, 20, 20])
            program.XStep = program.YStep = 20
            page.Resources.Pattern = Dictionary(P=program)
            content = b"/GS0 gs /Pattern cs /P scn 0 0 20 20 re f"
        else:
            page.Resources.Font = Dictionary(
                F=Dictionary(
                    Type=Name.Font,
                    Subtype=Name.Type3,
                    CharProcs=Dictionary(A=program),
                    Resources=Dictionary(),
                )
            )
            content = b"/GS0 gs BT /F 12 Tf (A) Tj ET"
        page.Contents = pdf.make_stream(content)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert page.Contents.read_bytes() == content
        assert program.read_bytes() == program_data
        assert page.Resources.ExtGState.GS0.OPM == 1

    @pytest.mark.parametrize("paint", [b"", b"/Off gs 0 0 20 20 re f"])
    def test_soft_mask_is_analyzed_at_actual_paint_not_selection(self, paint) -> None:
        """Do not repair unused masks or masks painted after overprint is disabled."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs " + paint, OPM=1, OP=True)
        mask_data = b"/CS0 cs 0 0 20 20 re f"
        mask = pdf.make_stream(mask_data)
        mask.Subtype = Name.Form
        mask.Resources = Dictionary(ColorSpace=page.Resources.ColorSpace)
        page.Resources.ExtGState.GS0.SMask = Dictionary(S=Name.Alpha, G=mask)
        page.Resources.ExtGState.Off = Dictionary(OP=False)
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert mask.read_bytes() == mask_data

    @pytest.mark.parametrize(
        "location", ["page", "inherited_form", "selected_in_parent", "own_form"]
    )
    def test_pattern_uses_its_resource_definition_state(self, location) -> None:
        """Use the initial state of the scope defining the selected pattern."""
        pdf = new_pdf()
        page = self._page(pdf, b"", OPM=1, OP=True)
        pattern_data = b"/CS0 cs 0 1 0 0 scn 0 0 20 20 re f"
        pattern = pdf.make_stream(pattern_data)
        pattern.PatternType = 1
        pattern.PaintType = 1
        pattern.TilingType = 1
        pattern.BBox = Array([0, 0, 20, 20])
        pattern.XStep = pattern.YStep = 20
        pattern.Resources = Dictionary(ColorSpace=page.Resources.ColorSpace)
        page.Resources.Pattern = Dictionary(P=pattern)
        selection = b"/Pattern cs /P scn "
        paint = b"0 0 20 20 re f"
        if location == "page":
            content = b"/GS0 gs " + selection + paint
        else:
            form_data = (
                selection + paint
                if location in {"inherited_form", "own_form"}
                else paint
            )
            form = pdf.make_stream(form_data)
            form.Subtype = Name.Form
            form.BBox = Array([0, 0, 20, 20])
            if location == "selected_in_parent":
                form.Resources = Dictionary()
            elif location == "own_form":
                form.Resources = Dictionary(Pattern=page.Resources.Pattern)
            page.Resources.XObject = Dictionary(Form=form)
            content = b"/GS0 gs "
            if location == "selected_in_parent":
                content += selection
            content += b"/Form Do"
        page.Contents = pdf.make_stream(content)

        if location == "own_form":
            # Its resource is defined in a Form entered with overprint enabled.
            # Unlike the page-defined pattern, this really needs a repair.
            with pytest.raises(ConversionError, match="tiling pattern"):
                sanitize_extgstate(pdf)
            assert pattern.read_bytes() == pattern_data
            return
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert page.Contents.read_bytes() == content
        assert pattern.read_bytes() == pattern_data

    def test_shading_pattern_uses_nonstroking_overprint_only(self) -> None:
        """Ignore the stroking flag when evaluating a shading pattern."""
        pdf = new_pdf()
        content = b"/Pattern cs /P scn 0 0 20 20 re f"
        page = self._page(pdf, content)
        page.Resources.Pattern = Dictionary(
            P=Dictionary(
                PatternType=2,
                ExtGState=Dictionary(OP=True, op=False, OPM=1),
                Shading=Dictionary(
                    ShadingType=2, ColorSpace=page.Resources.ColorSpace.CS0
                ),
            )
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert page.Contents.read_bytes() == content

    def test_shading_pattern_extgstate_is_sanitized(self) -> None:
        """A pattern's own graphics state follows the same rules as a resource."""
        pdf = new_pdf()
        content = b"/Pattern cs /P scn 0 0 20 20 re f"
        page = self._page(pdf, content)
        state = Dictionary(
            TR=Name.Identity, HTP=Array([0, 0]), OP=1, OPM=Decimal("1.0")
        )
        page.Resources.Pattern = Dictionary(
            P=Dictionary(
                PatternType=2,
                ExtGState=state,
                Shading=Dictionary(ShadingType=2, ColorSpace=Name.DeviceGray),
            )
        )
        sanitize_extgstate(pdf)
        cleaned = page.Resources.Pattern.P.ExtGState
        assert "/TR" not in cleaned and "/HTP" not in cleaned
        assert cleaned.OP is True
        assert isinstance(cleaned.OPM, int) and cleaned.OPM == 1

    @pytest.mark.parametrize(
        "mask_paint,fixes",
        [(b"/CS0 cs 0 0 20 20 re f", 1), (b"0 g 0 0 20 20 re f", 0)],
    )
    def test_shading_pattern_soft_mask_is_analyzed(self, mask_paint, fixes) -> None:
        """A mask selected by a shading pattern's ExtGState belongs to that paint."""
        pdf = new_pdf()
        content = b"/Pattern cs /P scn 0 0 20 20 re f"
        page = self._page(pdf, content)
        mask = pdf.make_stream(mask_paint)
        mask.Subtype = Name.Form
        mask.BBox = Array([0, 0, 20, 20])
        mask.Resources = Dictionary(ColorSpace=page.Resources.ColorSpace)
        page.Resources.Pattern = Dictionary(
            P=Dictionary(
                PatternType=2,
                ExtGState=Dictionary(
                    OP=True, op=True, OPM=1, SMask=Dictionary(S=Name.Alpha, G=mask)
                ),
                Shading=Dictionary(ShadingType=2, ColorSpace=Name.DeviceGray),
            )
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == fixes
        assert page.Contents.read_bytes() == content
        if fixes:
            self._assert_guard(mask, "f")
        else:
            assert mask.read_bytes() == mask_paint
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0

    def test_unused_shading_pattern_soft_mask_is_not_repaired(self) -> None:
        """A selected but unpainted pattern does not count as an actual paint."""
        pdf = new_pdf()
        content = b"/Pattern cs /P scn"
        page = self._page(pdf, content)
        mask_data = b"/CS0 cs 0 0 20 20 re f"
        mask = pdf.make_stream(mask_data)
        mask.Subtype = Name.Form
        mask.BBox = Array([0, 0, 20, 20])
        mask.Resources = Dictionary(ColorSpace=page.Resources.ColorSpace)
        page.Resources.Pattern = Dictionary(
            P=Dictionary(
                PatternType=2,
                ExtGState=Dictionary(
                    OP=True, op=True, OPM=1, SMask=Dictionary(S=Name.Alpha, G=mask)
                ),
                Shading=Dictionary(ShadingType=2, ColorSpace=Name.DeviceGray),
            )
        )
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 0
        assert mask.read_bytes() == mask_data

    def test_shading_direct_icc_without_colorspace_table_is_detected(self) -> None:
        """Find a shading ICC profile even without a named ColorSpace table."""
        pdf = new_pdf()
        page = self._page(pdf, b"/GS0 gs /Sh sh", OPM=1, OP=True)
        page.Resources.Shading = Dictionary(
            Sh=Dictionary(ShadingType=2, ColorSpace=page.Resources.ColorSpace.CS0)
        )
        del page.Resources["/ColorSpace"]
        assert sanitize_extgstate(pdf)["extgstate_fixed"] == 1
        self._assert_guard(page, "sh")


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
