# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for WCAG 2.1 PDF accessibility requirements."""

from pikepdf import Array, Dictionary, Name, String

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
