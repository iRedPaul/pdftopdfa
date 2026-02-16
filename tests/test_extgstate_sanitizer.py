# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Extended Graphics State sanitization for PDF/A compliance."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import get_cmyk_profile
from pdftopdfa.sanitizers.extgstate import (
    _sanitize_shadings_in_resources,
    sanitize_extgstate,
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
            CA=pikepdf.objects.Decimal("0.5"),
            ca=pikepdf.objects.Decimal("0.8"),
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
            CA=pikepdf.objects.Decimal("0.5"),
            ca=pikepdf.objects.Decimal("0.8"),
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
        gs = Dictionary(Type=Name.ExtGState, ca=pikepdf.objects.Decimal("0.0"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_ca_one_preserved(self) -> None:
        """Preserves /CA = 1.0 (fully opaque, valid)."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, CA=pikepdf.objects.Decimal("1.0"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0

    def test_ca_above_one_clamped(self) -> None:
        """Clamps /CA > 1.0 to 1.0."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, CA=pikepdf.objects.Decimal("1.5"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert float(pdf.pages[0].Resources.ExtGState.GS0.CA) == pytest.approx(1.0)

    def test_ca_negative_clamped(self) -> None:
        """Clamps /ca < 0.0 to 0.0."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, ca=pikepdf.objects.Decimal("-0.5"))
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert float(pdf.pages[0].Resources.ExtGState.GS0.ca) == pytest.approx(0.0)

    def test_both_ca_out_of_range(self) -> None:
        """Clamps both /CA and /ca when out of range."""
        pdf = new_pdf()
        gs = Dictionary(
            Type=Name.ExtGState,
            CA=pikepdf.objects.Decimal("2.0"),
            ca=pikepdf.objects.Decimal("-1.0"),
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

    def test_smask_dict_tr_removed(self) -> None:
        """Removes /TR entry from SMask dict while keeping the SMask."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        smask_dict = Dictionary(S=Name.Alpha, G=_make_form_stream(pdf), TR=tr_stream)
        gs = Dictionary(Type=Name.ExtGState, SMask=smask_dict)
        _make_pdf_with_extgstate(pdf, gs)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        gs_out = pdf.pages[0].Resources.ExtGState.GS0
        # SMask should still be a dict (not replaced with /None)
        assert isinstance(gs_out.SMask, Dictionary)
        assert "/TR" not in gs_out.SMask

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
            CA=pikepdf.objects.Decimal("0.7"),
            ca=pikepdf.objects.Decimal("0.3"),
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
            CA=pikepdf.objects.Decimal("5.0"),
            ca=pikepdf.objects.Decimal("-2.0"),
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
            CA=pikepdf.objects.Decimal("3.0"),
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
