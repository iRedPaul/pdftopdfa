# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/annotations.py."""

import pikepdf
import pytest
from conftest import resolve, save_and_reopen
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.annotations import (
    ensure_appearance_streams,
    fix_annotation_flags,
    fix_annotation_opacity,
    fix_button_appearance_subdicts,
    remove_annotation_colors,
    remove_forbidden_annotations,
    remove_needs_appearances,
    remove_non_normal_appearance_keys,
)
from pdftopdfa.sanitizers.base import (
    ANNOT_FLAG_HIDDEN,
    ANNOT_FLAG_INVISIBLE,
    ANNOT_FLAG_NOROTATE,
    ANNOT_FLAG_NOVIEW,
    ANNOT_FLAG_NOZOOM,
    ANNOT_FLAG_PRINT,
    ANNOT_FLAG_TOGGLENOVIEW,
)


class TestRemoveForbiddenAnnotations:
    """Tests for remove_forbidden_annotations()."""

    def test_no_annotations(self, make_pdf_with_page):
        """Returns 0 for page without annotations."""
        pdf = make_pdf_with_page()
        result = remove_forbidden_annotations(pdf)
        assert result == 0

    @pytest.mark.parametrize(
        "subtype",
        ["/Sound", "/Movie", "/Screen", "/3D", "/RichMedia", "/TrapNet"],
    )
    def test_removes_forbidden_annotation(self, make_pdf_with_page, subtype):
        """Forbidden annotation subtypes are removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name(subtype),
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 1

    def test_keeps_link_annotation(self, make_pdf_with_page):
        """Link annotations (compliant) are kept."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 0
        assert len(pdf.pages[0].Annots) == 1

    def test_keeps_text_annotation(self, make_pdf_with_page):
        """Text annotations (compliant) are kept."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 0
        assert len(pdf.pages[0].Annots) == 1

    def test_mixed_annotations(self, make_pdf_with_page):
        """Only forbidden annotations removed from mixed set."""
        pdf = make_pdf_with_page()
        link = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        sound = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Sound,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        text = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([link, sound, text])
        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 1
        assert len(pdf.pages[0].Annots) == 2

    def test_multiple_pages(self, make_pdf_with_page):
        """Annotations across multiple pages are processed."""
        pdf = make_pdf_with_page()
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page2)

        for page in pdf.pages:
            annot = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Sound,
                    Rect=Array([0, 0, 100, 100]),
                )
            )
            page["/Annots"] = Array([annot])

        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 2

    def test_removes_empty_annots_array(self, make_pdf_with_page):
        """Empty /Annots array is removed after cleaning."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Sound,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        remove_forbidden_annotations(pdf)
        assert "/Annots" not in pdf.pages[0]

    def test_removes_undefined_annotation_subtype(self, make_pdf_with_page):
        """Annotation subtype not defined in ISO 32000 is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/line"),  # Invalid case (defined type is /Line)
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 1
        assert "/Annots" not in pdf.pages[0]

    def test_removes_forbidden_annotation_in_form_xobject_annots(
        self, make_pdf_with_page
    ):
        """Forbidden annotations in nested Form XObject /Annots are removed."""
        pdf = make_pdf_with_page()

        sound = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Sound,
                Rect=Array([0, 0, 10, 10]),
            )
        )
        form = pdf.make_stream(b"")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 10, 10])
        form[Name("/Annots")] = Array([sound])
        form_ref = pdf.make_indirect(form)

        xobjects = Dictionary()
        xobjects[Name("/Fm0")] = form_ref
        resources = Dictionary()
        resources[Name("/XObject")] = xobjects
        pdf.pages[0][Name("/Resources")] = resources

        pdf = save_and_reopen(pdf)
        result = remove_forbidden_annotations(pdf)
        assert result == 1
        page_resources = resolve(pdf.pages[0].get("/Resources"))
        page_xobjects = resolve(page_resources.get("/XObject"))
        form_obj = resolve(page_xobjects.get("/Fm0"))
        assert form_obj.get("/Annots") is None


class TestFixAnnotationFlags:
    """Tests for fix_annotation_flags()."""

    def test_no_annotations(self, make_pdf_with_page):
        """Returns 0 for page without annotations."""
        pdf = make_pdf_with_page()
        result = fix_annotation_flags(pdf)
        assert result == 0

    def test_sets_print_flag(self, make_pdf_with_page):
        """Print flag is set when missing."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                F=0,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F"))
        assert flags & ANNOT_FLAG_PRINT

    def test_keeps_existing_print_flag(self, make_pdf_with_page):
        """Annotation with all required flags already set is unchanged."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_PRINT | ANNOT_FLAG_NOZOOM | ANNOT_FLAG_NOROTATE,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 0

    def test_sets_nozoom_norotate_on_text_annotation(self, make_pdf_with_page):
        """Text annotations get NoZoom and NoRotate flags set."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_PRINT,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        flags = int(pdf.pages[0]["/Annots"][0].get("/F", 0))
        assert flags & ANNOT_FLAG_NOZOOM
        assert flags & ANNOT_FLAG_NOROTATE
        assert flags & ANNOT_FLAG_PRINT

    def test_nozoom_norotate_not_set_on_non_text_annotation(self, make_pdf_with_page):
        """Non-Text annotations do not get NoZoom/NoRotate flags."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_PRINT,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 0
        flags = int(pdf.pages[0]["/Annots"][0].get("/F", 0))
        assert not (flags & ANNOT_FLAG_NOZOOM)
        assert not (flags & ANNOT_FLAG_NOROTATE)

    @pytest.mark.parametrize(
        "forbidden_flag",
        [
            ANNOT_FLAG_INVISIBLE,
            ANNOT_FLAG_HIDDEN,
            ANNOT_FLAG_NOVIEW,
            ANNOT_FLAG_TOGGLENOVIEW,
        ],
        ids=["Invisible", "Hidden", "NoView", "ToggleNoView"],
    )
    def test_removes_forbidden_flag(self, make_pdf_with_page, forbidden_flag):
        """Forbidden annotation flags are removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                F=forbidden_flag | ANNOT_FLAG_PRINT,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F"))
        assert not (flags & forbidden_flag)
        assert flags & ANNOT_FLAG_PRINT

    def test_fixes_multiple_flags_at_once(self, make_pdf_with_page):
        """Multiple flag issues fixed in one pass."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_INVISIBLE | ANNOT_FLAG_HIDDEN,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F"))
        assert not (flags & ANNOT_FLAG_INVISIBLE)
        assert not (flags & ANNOT_FLAG_HIDDEN)
        assert flags & ANNOT_FLAG_PRINT

    def test_default_flags_when_missing(self, make_pdf_with_page):
        """Annotations without /F key get Print flag."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F"))
        assert flags & ANNOT_FLAG_PRINT

    def test_widget_annotation_gets_print_flag(self, make_pdf_with_page):
        """Widget annotations also get Print flag set."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                F=0,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F", 0))
        assert flags & ANNOT_FLAG_PRINT

    def test_widget_gets_hidden_invisible_removed_and_print_set(
        self, make_pdf_with_page
    ):
        """Widget annotations have flags cleaned and Print set."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_HIDDEN | ANNOT_FLAG_INVISIBLE,
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_flags(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        flags = int(resolved.get("/F"))
        assert not (flags & ANNOT_FLAG_HIDDEN)
        assert not (flags & ANNOT_FLAG_INVISIBLE)
        assert flags & ANNOT_FLAG_PRINT


class TestEnsureAppearanceStreams:
    """Tests for ensure_appearance_streams()."""

    def test_no_annotations(self, make_pdf_with_page):
        """Returns 0 for page without annotations."""
        pdf = make_pdf_with_page()
        result = ensure_appearance_streams(pdf)
        assert result == 0

    def test_adds_ap_to_annotation_without_ap(self, make_pdf_with_page):
        """Appearance stream is added to annotation without /AP."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert "/AP" in resolved
        ap = resolve(resolved.get("/AP"))
        assert "/N" in ap

    def test_adds_n_to_existing_ap(self, make_pdf_with_page):
        """Normal appearance added to existing /AP without /N."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        assert "/N" in ap

    def test_skips_popup_annotations(self, make_pdf_with_page):
        """Popup annotations are skipped (exempt per spec)."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Popup"),
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 0

    def test_skips_link_annotations(self, make_pdf_with_page):
        """Link annotations are skipped (exempt per rule 6.3.3-1)."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Link"),
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 0

    def test_adds_ap_to_zero_width_annotation(self, make_pdf_with_page):
        """Zero-width (but non-zero height) annotation is NOT exempt per spec.

        ISO 19005-2 rule 6.3.3 requires BOTH x1==x2 AND y1==y2 for exemption.
        """
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 100, 720]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1

    def test_adds_ap_to_zero_height_annotation(self, make_pdf_with_page):
        """Zero-height (but non-zero width) annotation is NOT exempt per spec.

        ISO 19005-2 rule 6.3.3 requires BOTH x1==x2 AND y1==y2 for exemption.
        """
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 700]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1

    def test_skips_annotation_with_existing_ap_n(self, make_pdf_with_page):
        """Annotations with existing /AP /N are left alone."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 20, 20])
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(N=stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 0

    def test_appearance_stream_uses_rect_dimensions(self, make_pdf_with_page):
        """Created appearance stream BBox matches Rect."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 200, 750]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        ensure_appearance_streams(pdf)
        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        ap_stream = resolve(ap.get("/N"))
        bbox = ap_stream.get("/BBox")
        assert float(bbox[0]) == 0
        assert float(bbox[1]) == 0
        assert float(bbox[2]) == pytest.approx(100, abs=1)
        assert float(bbox[3]) == pytest.approx(50, abs=1)

    def test_multiple_annotations_mixed(self, make_pdf_with_page):
        """Mix of annotations with and without /AP."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 20, 20])

        with_ap = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
                AP=Dictionary(N=stream),
            )
        )
        without_ap = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        popup = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Popup"),
                Rect=Array([0, 0, 50, 50]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([with_ap, without_ap, popup])
        pdf = save_and_reopen(pdf)
        result = ensure_appearance_streams(pdf)
        assert result == 1

    def test_collapses_state_dict_for_non_widget_stamp(self, make_pdf_with_page):
        """Stamp annotation with state dict /AP/N gets collapsed to Stream."""
        pdf = make_pdf_with_page()

        on_stream = pdf.make_stream(b"q Q")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 60, 35])

        off_stream = pdf.make_stream(b"")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 60, 35])

        state_dict = Dictionary()
        state_dict[Name("/On")] = on_stream
        state_dict[Name("/Off")] = off_stream

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([180, 400, 240, 435]),
                AP=Dictionary(N=state_dict),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 1

        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        # Must be a Stream, not a Dictionary
        assert isinstance(n, pikepdf.Stream)

    def test_collapses_empty_state_dict_for_file_attachment(self, make_pdf_with_page):
        """FileAttachment with empty state dict /AP/N gets minimal stream."""
        pdf = make_pdf_with_page()

        empty_dict = Dictionary()

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/FileAttachment"),
                Rect=Array([307, 682, 314, 699]),
                AP=Dictionary(N=empty_dict),
                AS=Name("/Name"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 1

        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        # Must be a Stream, not a Dictionary
        assert isinstance(n, pikepdf.Stream)
        # /AS should be removed since state dict is gone
        assert resolved.get("/AS") is None

    def test_collapses_state_dict_uses_as_matching_entry(self, make_pdf_with_page):
        """When /AS matches a state dict entry, that stream is used."""
        pdf = make_pdf_with_page()

        name_stream = pdf.make_stream(b"q 1 0 0 rg Q")
        name_stream[Name.Type] = Name.XObject
        name_stream[Name.Subtype] = Name.Form
        name_stream[Name.BBox] = Array([0, 0, 7, 17])

        state_dict = Dictionary()
        state_dict[Name("/Name")] = name_stream

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/FileAttachment"),
                Rect=Array([307, 682, 314, 699]),
                AP=Dictionary(N=state_dict),
                AS=Name("/Name"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 1

        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        assert isinstance(n, pikepdf.Stream)
        # The extracted stream should have the original content
        assert bytes(n.read_bytes()) == b"q 1 0 0 rg Q"

    def test_preserves_btn_widget_state_dict(self, make_pdf_with_page):
        """Btn widget with state dict /AP/N is left unchanged."""
        pdf = make_pdf_with_page()

        yes_stream = pdf.make_stream(b"q Q")
        yes_stream[Name.Type] = Name.XObject
        yes_stream[Name.Subtype] = Name.Form
        yes_stream[Name.BBox] = Array([0, 0, 14, 14])

        state_dict = Dictionary()
        state_dict[Name("/Yes")] = yes_stream

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name.Btn,
                Rect=Array([0, 0, 14, 14]),
                AP=Dictionary(N=state_dict),
                AS=Name("/Yes"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 0

        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        # Should still be a Dictionary (state dict), not collapsed
        assert isinstance(n, Dictionary)
        assert not isinstance(n, pikepdf.Stream)

    def test_collapses_state_dict_for_widget_non_btn(self, make_pdf_with_page):
        """Widget without /FT=Btn with state dict /AP/N gets collapsed."""
        pdf = make_pdf_with_page()

        stream = pdf.make_stream(b"q Q")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 100, 20])

        state_dict = Dictionary()
        state_dict[Name("/V")] = stream

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name.Tx,
                Rect=Array([0, 0, 100, 20]),
                AP=Dictionary(N=state_dict),
                AS=Name("/V"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 1

        resolved = resolve(pdf.pages[0].Annots[0])
        ap = resolve(resolved.get("/AP"))
        n = resolve(ap.get("/N"))
        assert isinstance(n, pikepdf.Stream)

    def test_leaves_valid_stream_ap_n_unchanged(self, make_pdf_with_page):
        """Non-widget annotation with valid Stream /AP/N is not modified."""
        pdf = make_pdf_with_page()

        stream = pdf.make_stream(b"q Q")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 20, 20])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(N=stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = ensure_appearance_streams(pdf)
        assert result == 0


class TestRemoveNonNormalAppearanceKeys:
    """Tests for remove_non_normal_appearance_keys()."""

    def test_removes_r_and_d_from_ap_dictionary(self, make_pdf_with_page):
        """Only /N is kept in /AP; /R and /D are removed."""
        pdf = make_pdf_with_page()

        n_stream = pdf.make_stream(b"")
        n_stream[Name.Type] = Name.XObject
        n_stream[Name.Subtype] = Name.Form
        n_stream[Name.BBox] = Array([0, 0, 20, 20])

        r_stream = pdf.make_stream(b"")
        r_stream[Name.Type] = Name.XObject
        r_stream[Name.Subtype] = Name.Form
        r_stream[Name.BBox] = Array([0, 0, 20, 20])

        d_stream = pdf.make_stream(b"")
        d_stream[Name.Type] = Name.XObject
        d_stream[Name.Subtype] = Name.Form
        d_stream[Name.BBox] = Array([0, 0, 20, 20])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=n_stream, R=r_stream, D=d_stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = remove_non_normal_appearance_keys(pdf)
        assert result == 2

        resolved = resolve(pdf.pages[0]["/Annots"][0])
        ap = resolve(resolved.get("/AP"))
        assert ap.get("/N") is not None
        assert ap.get("/R") is None
        assert ap.get("/D") is None

    def test_integration_sanitize_for_pdfa_removes_r_and_d(self, make_pdf_with_page):
        """Integration: sanitize_for_pdfa removes /AP /R and /D keys."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = make_pdf_with_page()
        n_stream = pdf.make_stream(b"")
        n_stream[Name.Type] = Name.XObject
        n_stream[Name.Subtype] = Name.Form
        n_stream[Name.BBox] = Array([0, 0, 10, 10])

        r_stream = pdf.make_stream(b"")
        r_stream[Name.Type] = Name.XObject
        r_stream[Name.Subtype] = Name.Form
        r_stream[Name.BBox] = Array([0, 0, 10, 10])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=n_stream, R=r_stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)

        result = sanitize_for_pdfa(pdf, "2b")
        assert result["annotation_ap_keys_removed"] == 1
        resolved = resolve(pdf.pages[0]["/Annots"][0])
        ap = resolve(resolved.get("/AP"))
        assert ap.get("/N") is not None
        assert ap.get("/R") is None


class TestRemoveNeedsAppearances:
    """Tests for remove_needs_appearances()."""

    def test_no_acroform(self, make_pdf_with_page):
        """Returns False when no /AcroForm exists."""
        pdf = make_pdf_with_page()
        assert remove_needs_appearances(pdf) is False

    def test_acroform_without_flag(self, make_pdf_with_page):
        """Returns False when /AcroForm exists but has no /NeedAppearances."""
        pdf = make_pdf_with_page()
        pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary())
        pdf = save_and_reopen(pdf)
        assert remove_needs_appearances(pdf) is False

    def test_removes_true_flag(self, make_pdf_with_page):
        """Removes /NeedAppearances when set to true."""
        pdf = make_pdf_with_page()
        acroform = Dictionary()
        acroform[Name("/NeedAppearances")] = True
        pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)
        pdf = save_and_reopen(pdf)
        assert remove_needs_appearances(pdf) is True
        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedAppearances") is None

    def test_removes_false_flag(self, make_pdf_with_page):
        """Removes /NeedAppearances even when set to false."""
        pdf = make_pdf_with_page()
        acroform = Dictionary()
        acroform[Name("/NeedAppearances")] = False
        pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)
        pdf = save_and_reopen(pdf)
        assert remove_needs_appearances(pdf) is True
        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedAppearances") is None

    def test_removes_legacy_plural_key(self, make_pdf_with_page):
        """Legacy typo key /NeedsAppearances is also removed."""
        pdf = make_pdf_with_page()
        acroform = Dictionary()
        acroform[Name("/NeedsAppearances")] = True
        pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)
        pdf = save_and_reopen(pdf)
        assert remove_needs_appearances(pdf) is True
        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedsAppearances") is None

    def test_preserves_other_acroform_keys(self, make_pdf_with_page):
        """Other /AcroForm keys are preserved after removing the flag."""
        pdf = make_pdf_with_page()
        acroform = Dictionary()
        acroform[Name("/NeedAppearances")] = True
        acroform[Name("/DA")] = "/Helv 12 Tf 0 g"
        pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)
        pdf = save_and_reopen(pdf)
        remove_needs_appearances(pdf)
        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedAppearances") is None
        assert str(acroform.get("/DA")) == "/Helv 12 Tf 0 g"

    def test_sanitize_for_pdfa_removes_flag(self, make_pdf_with_page):
        """Integration: sanitize_for_pdfa removes /NeedAppearances."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = make_pdf_with_page()
        acroform = Dictionary()
        acroform[Name("/NeedAppearances")] = True
        pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)
        pdf = save_and_reopen(pdf)
        result = sanitize_for_pdfa(pdf, "2b")
        assert result["needs_appearances_removed"] is True
        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedAppearances") is None

    def test_sanitize_for_pdfa_generates_widget_ap_before_removal(
        self, make_pdf_with_page
    ):
        """Widget /AP is generated and /NeedAppearances is removed."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = make_pdf_with_page()
        widget = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Tx"),
                T="Field1",
                Rect=Array([0, 0, 120, 24]),
                V="Hello",
                DA="/Helv 12 Tf 0 g",
                DR=Dictionary(
                    Font=Dictionary(
                        Helv=Dictionary(
                            Type=Name.Font,
                            Subtype=Name("/Type1"),
                            BaseFont=Name("/Helvetica"),
                        )
                    )
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([widget])
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(
                Fields=Array([widget]),
                NeedAppearances=True,
            )
        )
        pdf = save_and_reopen(pdf)

        result = sanitize_for_pdfa(pdf, "2b")
        assert result["needs_appearances_removed"] is True

        resolved_widget = resolve(pdf.pages[0]["/Annots"][0])
        ap = resolve(resolved_widget.get("/AP"))
        assert ap is not None
        assert ap.get("/N") is not None

        acroform = resolve(pdf.Root["/AcroForm"])
        assert acroform.get("/NeedAppearances") is None


class TestFixAnnotationOpacity:
    """Tests for fix_annotation_opacity()."""

    def test_no_annotations(self, make_pdf_with_page):
        """Returns 0 for page without annotations."""
        pdf = make_pdf_with_page()
        result = fix_annotation_opacity(pdf)
        assert result == 0

    def test_no_ca_key_untouched(self, make_pdf_with_page):
        """Annotations without /CA are left unchanged."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 0
        resolved = resolve(pdf.pages[0].Annots[0])
        assert resolved.get("/CA") is None

    def test_ca_1_0_untouched(self, make_pdf_with_page):
        """Annotations with /CA 1.0 are left unchanged."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/CA")] = 1.0
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 0

    @pytest.mark.parametrize(
        "ca_value",
        [0.0, 0.5, 2.0],
        ids=["zero", "below_1", "above_1"],
    )
    def test_ca_not_1_fixed(self, make_pdf_with_page, ca_value):
        """Annotations with /CA != 1.0 are set to 1.0."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/CA")] = ca_value
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert float(resolved.get("/CA")) == 1.0

    def test_multiple_annotations(self, make_pdf_with_page):
        """Multiple annotations with bad /CA are all fixed."""
        pdf = make_pdf_with_page()
        a1 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        a1[Name("/CA")] = 0.3
        a2 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        a2[Name("/CA")] = 0.7
        a3 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        a3[Name("/CA")] = 1.0
        pdf.pages[0]["/Annots"] = Array([a1, a2, a3])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 2

    def test_multiple_pages(self, make_pdf_with_page):
        """Annotations across multiple pages are processed."""
        pdf = make_pdf_with_page()
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page2)

        for page in pdf.pages:
            annot = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Link,
                    Rect=Array([0, 0, 100, 100]),
                )
            )
            annot[Name("/CA")] = 0.5
            page["/Annots"] = Array([annot])

        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 2

    def test_widget_annotation_ca_fixed(self, make_pdf_with_page):
        """Widget annotations with /CA != 1.0 are also fixed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/CA")] = 0.5
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_annotation_opacity(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert float(resolved.get("/CA")) == 1.0

    def test_integration_sanitize_for_pdfa(self, make_pdf_with_page):
        """Integration: sanitize_for_pdfa fixes annotation /CA."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/CA")] = 0.5
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = sanitize_for_pdfa(pdf, "2b")
        assert result["annotation_opacity_fixed"] == 1


class TestRemoveAnnotationColors:
    """Tests for remove_annotation_colors()."""

    def test_no_annotations(self, make_pdf_with_page):
        """Returns 0 for page without annotations."""
        pdf = make_pdf_with_page()
        result = remove_annotation_colors(pdf)
        assert result == 0

    def test_no_color_keys(self, make_pdf_with_page):
        """Returns 0 when annotations have no /C or /IC."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 0

    def test_removes_c_rgb(self, make_pdf_with_page):
        """/C with RGB array is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([1.0, 0.0, 0.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert resolved.get("/C") is None

    def test_removes_c_gray(self, make_pdf_with_page):
        """/C with grayscale array is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Circle,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([0.5])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert resolved.get("/C") is None

    def test_removes_ic(self, make_pdf_with_page):
        """/IC interior color array is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/IC")] = Array([0.0, 1.0, 0.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 1
        resolved = resolve(pdf.pages[0].Annots[0])
        assert resolved.get("/IC") is None

    def test_removes_both_c_and_ic(self, make_pdf_with_page):
        """Both /C and /IC are removed from the same annotation."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([1.0, 0.0, 0.0])
        annot[Name("/IC")] = Array([0.0, 0.0, 1.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 2
        resolved = resolve(pdf.pages[0].Annots[0])
        assert resolved.get("/C") is None
        assert resolved.get("/IC") is None

    def test_multiple_annotations(self, make_pdf_with_page):
        """Colors removed from multiple annotations."""
        pdf = make_pdf_with_page()
        annot1 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot1[Name("/C")] = Array([1.0, 0.0, 0.0])
        annot2 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Circle,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        annot2[Name("/IC")] = Array([0.5])
        pdf.pages[0]["/Annots"] = Array([annot1, annot2])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 2

    def test_preserves_other_keys(self, make_pdf_with_page):
        """Other annotation keys are preserved when /C is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([1.0, 0.0, 0.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        remove_annotation_colors(pdf)
        resolved = resolve(pdf.pages[0].Annots[0])
        assert str(resolved.get("/Subtype")) == "/Square"
        assert resolved.get("/Rect") is not None

    def test_empty_c_array_removed(self, make_pdf_with_page):
        """An empty /C array is also removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 1

    def test_widget_colors_removed(self, make_pdf_with_page):
        """Widget annotation /C is also removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([0.0, 0.0, 1.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_annotation_colors(pdf)
        assert result == 1

    def test_integration_sanitize_for_pdfa(self, make_pdf_with_page):
        """Integration: sanitize_for_pdfa removes annotation colors."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Square,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        annot[Name("/C")] = Array([1.0, 0.0, 0.0])
        annot[Name("/IC")] = Array([0.0, 1.0, 0.0])
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = sanitize_for_pdfa(pdf, "2b")
        assert result["annotation_colors_removed"] == 2


# ===================================================================
# fix_button_appearance_subdicts Tests
# ===================================================================


class TestFixButtonAppearanceSubdicts:
    """Tests for fix_button_appearance_subdicts()."""

    def test_btn_stream_wrapped(self, make_pdf_with_page):
        """Btn widget with /AP/N as Stream gets wrapped in Dictionary."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 50, 20])
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Btn"),
                Rect=Array([0, 0, 50, 20]),
                AP=Dictionary(N=stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_button_appearance_subdicts(pdf)
        assert result == 1
        annot = pdf.pages[0]["/Annots"][0]
        n = annot["/AP"]["/N"]
        # Should now be a Dictionary, not a Stream
        assert isinstance(n, Dictionary)
        assert not isinstance(n, pikepdf.Stream)
        # State key defaults to "Yes"
        assert n.get(Name("/Yes")) is not None
        # /AS should be set
        assert str(annot.get("/AS")) == "/Yes"

    def test_btn_dict_unchanged(self, make_pdf_with_page):
        """Btn widget with /AP/N already a Dictionary is left alone."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 50, 20])
        state_dict = Dictionary()
        state_dict[Name("/Yes")] = stream
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Btn"),
                Rect=Array([0, 0, 50, 20]),
                AP=Dictionary(N=state_dict),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_button_appearance_subdicts(pdf)
        assert result == 0

    def test_non_btn_stream_unchanged(self, make_pdf_with_page):
        """Tx widget with /AP/N as Stream is not touched."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 50, 20])
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Tx"),
                Rect=Array([0, 0, 50, 20]),
                AP=Dictionary(N=stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_button_appearance_subdicts(pdf)
        assert result == 0

    def test_inherited_ft(self, make_pdf_with_page):
        """Btn via /Parent chain is still detected and fixed."""
        pdf = make_pdf_with_page()
        parent = pdf.make_indirect(
            Dictionary(
                FT=Name("/Btn"),
            )
        )
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 50, 20])
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Parent=parent,
                Rect=Array([0, 0, 50, 20]),
                AP=Dictionary(N=stream),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_button_appearance_subdicts(pdf)
        assert result == 1

    def test_as_used_as_key(self, make_pdf_with_page):
        """Annotation with /AS /On uses 'On' as the Dictionary key."""
        pdf = make_pdf_with_page()
        stream = pdf.make_stream(b"")
        stream[Name.Type] = Name.XObject
        stream[Name.Subtype] = Name.Form
        stream[Name.BBox] = Array([0, 0, 50, 20])
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Btn"),
                Rect=Array([0, 0, 50, 20]),
                AP=Dictionary(N=stream),
                AS=Name("/On"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = fix_button_appearance_subdicts(pdf)
        assert result == 1
        annot = pdf.pages[0]["/Annots"][0]
        n = annot["/AP"]["/N"]
        assert isinstance(n, Dictionary)
        assert n.get(Name("/On")) is not None
        assert str(annot.get("/AS")) == "/On"
