# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for WCAG 2.1 PDF accessibility requirements."""

import pytest
from pikepdf import Array, Dictionary, Name, NumberTree, String

from pdftopdfa.utils import resolve_indirect
from pdftopdfa.wcag import apply_wcag_21, prepare_pdfua_document


def test_apply_wcag_sets_structure_tab_order_on_every_page(sample_pdf_obj) -> None:
    page = sample_pdf_obj.pages[0]

    result = apply_wcag_21(sample_pdf_obj)

    assert page.obj["/Tabs"] == Name.S
    assert result["page_tab_orders_set"] == 1


def test_apply_wcag_preserves_existing_structure_tab_order(sample_pdf_obj) -> None:
    sample_pdf_obj.pages[0].obj["/Tabs"] = Name.S

    result = apply_wcag_21(sample_pdf_obj)

    assert result["page_tab_orders_set"] == 0


def test_apply_wcag_labels_an_inherited_required_form_control(sample_pdf_obj) -> None:
    page = sample_pdf_obj.pages[0]
    field = sample_pdf_obj.make_indirect(
        Dictionary(FT=Name.Tx, Ff=2, T=String("Email address"))
    )
    widget = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Parent=field,
            Rect=Array([0, 0, 100, 20]),
        )
    )
    field["/Kids"] = Array([widget])
    page.obj["/Annots"] = Array([widget])
    sample_pdf_obj.Root["/AcroForm"] = Dictionary(Fields=Array([field]))

    result = apply_wcag_21(sample_pdf_obj)

    assert str(field["/TU"]) == "Email address (required)"
    assert "/TU" not in widget
    assert result["required_controls_labeled"] == 1
    assert apply_wcag_21(sample_pdf_obj)["required_controls_labeled"] == 0
    assert str(field["/TU"]) == "Email address (required)"


def test_apply_wcag_localizes_generated_german_accessibility_text(
    sample_pdf_obj,
) -> None:
    sample_pdf_obj.Root["/Lang"] = String("de-DE")
    page = sample_pdf_obj.pages[0]
    widget = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            Ff=2,
            T=String("E-Mail-Adresse"),
            TU=String("Form field"),
            Rect=Array([0, 0, 100, 20]),
        )
    )
    printer_mark = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.PrinterMark,
            Rect=Array([0, 0, 10, 10]),
        )
    )
    page.obj["/Annots"] = Array([widget, printer_mark])

    result = apply_wcag_21(sample_pdf_obj)

    assert str(widget["/TU"]) == "E-Mail-Adresse (Pflichtfeld)"
    assert str(printer_mark["/Contents"]) == "PrinterMark-Anmerkung"
    assert result["required_controls_labeled"] == 1


def test_apply_wcag_synchronizes_an_existing_widget_tooltip(sample_pdf_obj) -> None:
    page = sample_pdf_obj.pages[0]
    field = sample_pdf_obj.make_indirect(
        Dictionary(FT=Name.Tx, Ff=2, T=String("Email address"))
    )
    widget = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Parent=field,
            TU=String("Email address"),
            Rect=Array([0, 0, 100, 20]),
        )
    )
    field["/Kids"] = Array([widget])
    page.obj["/Annots"] = Array([widget])

    result = apply_wcag_21(sample_pdf_obj)

    assert str(field["/TU"]) == "Email address (required)"
    assert str(widget["/TU"]) == "Email address (required)"
    assert result["required_controls_labeled"] == 1


def test_apply_wcag_bounds_required_form_labels(sample_pdf_obj) -> None:
    page = sample_pdf_obj.pages[0]
    widgets = []
    for label in ("a" * 32_760, "\u6f22" * 16_380):
        widget = sample_pdf_obj.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name.Tx,
                Ff=2,
                T=String(label),
                Rect=Array([0, 0, 100, 20]),
            )
        )
        widgets.append(widget)
    page.obj["/Annots"] = Array(widgets)

    result = apply_wcag_21(sample_pdf_obj)

    assert result["required_controls_labeled"] == 2
    for widget in widgets:
        tooltip = widget["/TU"]
        assert len(bytes(tooltip)) <= 32_767
        assert str(tooltip).endswith(" (required)")


def test_apply_wcag_does_not_change_optional_form_label(sample_pdf_obj) -> None:
    page = sample_pdf_obj.pages[0]
    widget = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            T=String("Email address"),
            Rect=Array([0, 0, 100, 20]),
        )
    )
    page.obj["/Annots"] = Array([widget])

    result = apply_wcag_21(sample_pdf_obj)

    assert "/TU" not in widget
    assert result["required_controls_labeled"] == 0


def test_apply_wcag_does_not_use_a_generic_required_form_label(
    sample_pdf_obj,
) -> None:
    page = sample_pdf_obj.pages[0]
    widget = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            Ff=2,
            TU=String("Form field"),
            Rect=Array([0, 0, 100, 20]),
        )
    )
    page.obj["/Annots"] = Array([widget])

    result = apply_wcag_21(sample_pdf_obj)

    assert str(widget["/TU"]) == "Form field"
    assert result["required_controls_labeled"] == 0


def test_apply_wcag_reports_only_undetermined_document_language(
    sample_pdf_obj,
) -> None:
    sample_pdf_obj.Root["/Lang"] = String("und")
    assert apply_wcag_21(sample_pdf_obj)["language_review_required"] is True

    sample_pdf_obj.Root["/Lang"] = String("en")
    assert apply_wcag_21(sample_pdf_obj)["language_review_required"] is False


def test_apply_wcag_adds_decimal_page_labels_and_human_review_limit(
    sample_pdf_obj,
) -> None:
    result = apply_wcag_21(sample_pdf_obj)

    page_labels = resolve_indirect(sample_pdf_obj.Root["/PageLabels"])
    numbers = resolve_indirect(page_labels["/Nums"])
    label = resolve_indirect(numbers[1])
    assert int(numbers[0]) == 0
    assert label["/S"] == Name.D
    assert int(label["/St"]) == 1
    assert result["page_labels_added"] == 1
    assert result["human_review_required"] is True
    assert apply_wcag_21(sample_pdf_obj)["page_labels_added"] == 0


def test_apply_wcag_repairs_an_empty_page_label_number_tree(sample_pdf_obj) -> None:
    sample_pdf_obj.Root["/PageLabels"] = Dictionary()

    result = apply_wcag_21(sample_pdf_obj)

    labels = NumberTree(resolve_indirect(sample_pdf_obj.Root["/PageLabels"]))
    assert list(labels) == [0]
    assert result["page_labels_added"] == 0
    assert result["page_labels_repaired"] == 1


def test_apply_wcag_repairs_page_labels_without_page_zero(sample_pdf_obj) -> None:
    sample_pdf_obj.add_blank_page()
    labels = NumberTree.new(sample_pdf_obj)
    labels[1] = Dictionary(S=Name.D, St=1)
    sample_pdf_obj.Root["/PageLabels"] = labels.obj

    result = apply_wcag_21(sample_pdf_obj)

    repaired = NumberTree(resolve_indirect(sample_pdf_obj.Root["/PageLabels"]))
    assert list(repaired) == [0]
    assert result["page_labels_repaired"] == 1


def test_apply_wcag_repairs_page_labels_with_noninteger_start(
    sample_pdf_obj,
) -> None:
    labels = NumberTree.new(sample_pdf_obj)
    labels[0] = Dictionary(S=Name.D, St=1.5)
    sample_pdf_obj.Root["/PageLabels"] = labels.obj

    result = apply_wcag_21(sample_pdf_obj)

    repaired = NumberTree(resolve_indirect(sample_pdf_obj.Root["/PageLabels"]))
    assert int(resolve_indirect(repaired[0])["/St"]) == 1
    assert result["page_labels_repaired"] == 1


@pytest.mark.parametrize("keys", [(0, 0), (1, 0)])
def test_apply_wcag_repairs_malformed_raw_page_label_keys(
    sample_pdf_obj,
    keys: tuple[int, int],
) -> None:
    sample_pdf_obj.add_blank_page()
    label = Dictionary(S=Name.D, St=1)
    sample_pdf_obj.Root["/PageLabels"] = Dictionary(
        Nums=Array([keys[0], label, keys[1], label])
    )

    result = apply_wcag_21(sample_pdf_obj)

    repaired = resolve_indirect(sample_pdf_obj.Root["/PageLabels"])
    assert [int(value) for value in resolve_indirect(repaired["/Nums"])[::2]] == [0]
    assert result["page_labels_repaired"] == 1


def test_apply_wcag_builds_hierarchical_bookmarks_from_heading_titles(
    sample_pdf_obj,
) -> None:
    second_page = sample_pdf_obj.add_blank_page()
    root = sample_pdf_obj.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = sample_pdf_obj.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    heading = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H1,
            P=document,
            Pg=sample_pdf_obj.pages[0].obj,
            T=String("Overview"),
        )
    )
    subheading = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H2,
            P=document,
            Pg=second_page.obj,
            T=String("Details"),
        )
    )
    document["/K"] = Array([heading, subheading])
    root["/K"] = document
    sample_pdf_obj.Root["/StructTreeRoot"] = root

    result = apply_wcag_21(sample_pdf_obj)

    outlines = resolve_indirect(sample_pdf_obj.Root["/Outlines"])
    first = resolve_indirect(outlines["/First"])
    child = resolve_indirect(first["/First"])
    assert str(first["/Title"]) == "Overview"
    assert str(child["/Title"]) == "Details"
    assert resolve_indirect(child["/Dest"])[0].objgen == second_page.obj.objgen
    assert result["bookmarks_added"] == 2


def test_apply_wcag_builds_bookmarks_from_role_mapped_headings(
    sample_pdf_obj,
) -> None:
    second_page = sample_pdf_obj.add_blank_page()
    root = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructTreeRoot,
            RoleMap=Dictionary(SectionHeading=Name.H1),
        )
    )
    headings = [
        sample_pdf_obj.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name("/SectionHeading"),
                P=root,
                Pg=page.obj,
                T=String(title),
            )
        )
        for page, title in zip(
            sample_pdf_obj.pages,
            ("Overview", "Details"),
            strict=True,
        )
    ]
    root["/K"] = Array(headings)
    sample_pdf_obj.Root["/StructTreeRoot"] = root

    result = apply_wcag_21(sample_pdf_obj)

    outlines = resolve_indirect(sample_pdf_obj.Root["/Outlines"])
    first = resolve_indirect(outlines["/First"])
    second = resolve_indirect(first["/Next"])
    assert str(first["/Title"]) == "Overview"
    assert str(second["/Title"]) == "Details"
    assert resolve_indirect(second["/Dest"])[0].objgen == second_page.obj.objgen
    assert result["bookmarks_added"] == 2


def test_apply_wcag_builds_bookmarks_from_heading_mcr_pages(sample_pdf_obj) -> None:
    second_page = sample_pdf_obj.add_blank_page()
    root = sample_pdf_obj.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    headings = [
        sample_pdf_obj.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.H1,
                P=root,
                K=Dictionary(Type=Name.MCR, Pg=page.obj, MCID=0),
                T=String(title),
            )
        )
        for page, title in zip(
            sample_pdf_obj.pages,
            ("Overview", "Details"),
            strict=True,
        )
    ]
    root["/K"] = Array(headings)
    sample_pdf_obj.Root["/StructTreeRoot"] = root

    result = apply_wcag_21(sample_pdf_obj)

    outlines = resolve_indirect(sample_pdf_obj.Root["/Outlines"])
    first = resolve_indirect(outlines["/First"])
    second = resolve_indirect(first["/Next"])
    first_page = resolve_indirect(first["/Dest"])[0]
    assert first_page.objgen == sample_pdf_obj.pages[0].obj.objgen
    assert resolve_indirect(second["/Dest"])[0].objgen == second_page.obj.objgen
    assert result["bookmarks_added"] == 2


def test_apply_wcag_builds_bookmarks_from_top_level_structure_array(
    sample_pdf_obj,
) -> None:
    second_page = sample_pdf_obj.add_blank_page()
    root = sample_pdf_obj.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    first_heading = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H1,
            P=root,
            Pg=sample_pdf_obj.pages[0].obj,
            T=String("Overview"),
        )
    )
    second_heading = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H1,
            P=root,
            Pg=second_page.obj,
            T=String("Details"),
        )
    )
    root["/K"] = Array([first_heading, second_heading])
    sample_pdf_obj.Root["/StructTreeRoot"] = root

    result = apply_wcag_21(sample_pdf_obj)

    outlines = resolve_indirect(sample_pdf_obj.Root["/Outlines"])
    first = resolve_indirect(outlines["/First"])
    second = resolve_indirect(first["/Next"])
    assert str(first["/Title"]) == "Overview"
    assert str(second["/Title"]) == "Details"
    assert result["bookmarks_added"] == 2


def test_prepare_pdfua_preserves_incidental_printer_mark_annotations(
    sample_pdf_obj,
) -> None:
    page = sample_pdf_obj.pages[0]
    printer_mark = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.PrinterMark,
            Rect=Array([0, 0, 10, 10]),
            StructParent=3,
            StructParents=4,
        )
    )
    link = sample_pdf_obj.make_indirect(
        Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 10, 10]))
    )
    page.obj["/Annots"] = Array([printer_mark, link])

    result = prepare_pdfua_document(sample_pdf_obj)

    assert result["printer_mark_annotations_preserved"] == 1
    assert len(page.obj["/Annots"]) == 2
    assert resolve_indirect(page.obj["/Annots"][0]).objgen == printer_mark.objgen
    assert resolve_indirect(page.obj["/Annots"][1]).objgen == link.objgen
    assert "/StructParent" not in printer_mark
    assert "/StructParents" not in printer_mark


def test_apply_wcag_adds_annotation_descriptions_and_reports_placeholders(
    sample_pdf_obj,
) -> None:
    page = sample_pdf_obj.pages[0]
    link = sample_pdf_obj.make_indirect(
        Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 10, 10]))
    )
    attachment = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.FileAttachment,
            Rect=Array([0, 0, 10, 10]),
            FS=Dictionary(Type=Name.Filespec, Desc=String("Source data")),
        )
    )
    page.obj["/Annots"] = Array([link, attachment])

    result = apply_wcag_21(sample_pdf_obj)

    assert str(link["/Contents"]) == "Link"
    assert str(attachment["/Contents"]) == "Source data"
    assert result["annotation_descriptions_added"] == 2
    assert result["annotation_descriptions_review_required"] == 1


def test_apply_wcag_tags_fallback_annotation_description_language(
    sample_pdf_obj,
) -> None:
    sample_pdf_obj.Root["/Lang"] = String("fr")
    page = sample_pdf_obj.pages[0]
    annotation = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([0, 0, 10, 10]),
            StructParent=0,
        )
    )
    root = sample_pdf_obj.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    owner = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Link,
            P=root,
            Pg=page.obj,
            Lang=String("fr"),
        )
    )
    owner["/K"] = sample_pdf_obj.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=annotation)
    )
    root["/K"] = owner
    parent_tree = NumberTree.new(sample_pdf_obj)
    parent_tree[0] = owner
    root["/ParentTree"] = parent_tree.obj
    sample_pdf_obj.Root["/StructTreeRoot"] = root
    page.obj["/Annots"] = Array([annotation])

    apply_wcag_21(sample_pdf_obj)

    assert str(annotation["/Contents"]) == "Link"
    assert str(owner["/Lang"]) == "en"


def test_apply_wcag_wraps_fallback_annotation_language_for_shared_owner(
    sample_pdf_obj,
) -> None:
    sample_pdf_obj.Root["/Lang"] = String("fr")
    page = sample_pdf_obj.pages[0]
    annotation = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([0, 0, 10, 10]),
            StructParent=0,
        )
    )
    root = sample_pdf_obj.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    owner = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Link,
            P=root,
            Pg=page.obj,
            Lang=String("fr"),
        )
    )
    object_reference = sample_pdf_obj.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=annotation)
    )
    marked_content = Dictionary(Type=Name.MCR, Pg=page.obj, MCID=0)
    owner["/K"] = Array([marked_content, object_reference])
    root["/K"] = owner
    parent_tree = NumberTree.new(sample_pdf_obj)
    parent_tree[0] = owner
    root["/ParentTree"] = parent_tree.obj
    sample_pdf_obj.Root["/StructTreeRoot"] = root
    page.obj["/Annots"] = Array([annotation])

    apply_wcag_21(sample_pdf_obj)

    children = resolve_indirect(owner["/K"])
    wrapper = resolve_indirect(children[1])
    assert str(owner["/Lang"]) == "fr"
    assert str(wrapper["/Lang"]) == "en"
    assert resolve_indirect(wrapper["/K"]).objgen == object_reference.objgen
    assert resolve_indirect(wrapper["/P"]).objgen == owner.objgen
    assert resolve_indirect(NumberTree(root["/ParentTree"])[0]).objgen == wrapper.objgen


def test_apply_wcag_describes_popup_and_printer_mark_annotations(
    sample_pdf_obj,
) -> None:
    page = sample_pdf_obj.pages[0]
    parent = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            Contents=String("Reviewer note"),
        )
    )
    popup = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Popup,
            Rect=Array([0, 0, 10, 10]),
            Parent=parent,
        )
    )
    printer_mark = sample_pdf_obj.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.PrinterMark,
            Rect=Array([0, 0, 10, 10]),
        )
    )
    page.obj["/Annots"] = Array([parent, popup, printer_mark])

    result = apply_wcag_21(sample_pdf_obj)

    assert str(popup["/Contents"]) == "Reviewer note"
    assert str(printer_mark["/Contents"]) == "PrinterMark annotation"
    assert result["annotation_descriptions_added"] == 2
    assert result["annotation_descriptions_review_required"] == 1
