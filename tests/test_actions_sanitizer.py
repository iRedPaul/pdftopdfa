# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/actions.py."""

import pikepdf
from conftest import new_pdf, save_and_reopen
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.actions import (
    _remove_actions_from_fields,
    _remove_actions_from_outlines,
    remove_actions,
    validate_destinations,
)


class TestRemoveActions:
    """Tests for remove_actions()."""

    def test_no_actions(self, make_pdf_with_page):
        """Returns 0 for PDF without any actions."""
        pdf = make_pdf_with_page()
        result = remove_actions(pdf)
        assert result == 0

    def test_removes_non_compliant_open_action(self, make_pdf_with_page):
        """Non-compliant OpenAction (Launch) is removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Launch, F="malware.exe")
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_keeps_compliant_open_action(self, make_pdf_with_page):
        """Compliant OpenAction (GoTo) is kept."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo, D=Array([pdf.pages[0].obj, Name.Fit])
        )
        remove_actions(pdf)
        assert "/OpenAction" in pdf.Root

    def test_removes_document_aa(self, make_pdf_with_page):
        """Non-compliant Document Additional Actions are removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/AA"] = Dictionary(
            WC=Dictionary(S=Name.Launch, F="evil.exe"),
        )
        result = remove_actions(pdf)
        assert result >= 1
        assert "/AA" not in pdf.Root

    def test_removes_page_aa(self, make_pdf_with_page):
        """Non-compliant Page Additional Actions are removed."""
        pdf = make_pdf_with_page()
        pdf.pages[0]["/AA"] = Dictionary(
            O=Dictionary(S=Name.Launch, F="evil.exe"),
        )
        result = remove_actions(pdf)
        assert result >= 1
        assert "/AA" not in pdf.pages[0]

    def test_page_aa_partial_removal(self, make_pdf_with_page):
        """Page /AA is removed unconditionally (ISO 19005-2 Section 6.6.2)."""
        pdf = make_pdf_with_page()
        pdf.pages[0]["/AA"] = Dictionary(
            O=Dictionary(S=Name.Launch, F="evil.exe"),
            C=Dictionary(
                S=Name.GoTo,
                D=Array([pdf.pages[0].obj, Name.Fit]),
            ),
        )
        result = remove_actions(pdf)
        assert result >= 1
        assert "/AA" not in pdf.pages[0]

    def test_removes_action_without_s_key(self, make_pdf_with_page):
        """Action without /S key on Link annotation is removed as malformed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(Annotation=Dictionary(Type=Name.Annot)),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_removes_annotation_action(self, make_pdf_with_page):
        """Non-compliant annotation actions are removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.Launch, F="evil.exe"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_removes_annotation_aa(self, make_pdf_with_page):
        """Non-compliant annotation Additional Actions are removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                AA=Dictionary(
                    E=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/AA" not in annot

    def test_removes_acroform_field_action(self, make_pdf_with_page):
        """Non-compliant AcroForm field actions are removed."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                A=Dictionary(S=Name.Launch, F="evil.exe"),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field

    def test_removes_acroform_field_aa(self, make_pdf_with_page):
        """Non-compliant AcroForm field Additional Actions removed."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                AA=Dictionary(
                    K=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/AA" not in field

    def test_removes_nested_field_actions(self, make_pdf_with_page):
        """Actions in nested form field children are removed."""
        pdf = make_pdf_with_page()
        child = pdf.make_indirect(
            Dictionary(
                T="child1",
                A=Dictionary(S=Name.Launch, F="evil.exe"),
            )
        )
        parent = pdf.make_indirect(
            Dictionary(
                T="parent",
                Kids=Array([child]),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([parent]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        parent_field = pdf.Root.AcroForm.Fields[0]
        try:
            parent_field = parent_field.get_object()
        except AttributeError:
            pass
        child_field = parent_field.Kids[0]
        try:
            child_field = child_field.get_object()
        except AttributeError:
            pass
        assert "/A" not in child_field

    def test_removes_javascript_open_action(self, make_pdf_with_page):
        """JavaScript OpenAction is removed as non-compliant."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.JavaScript, JS="app.alert('Hello');"
        )
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_javascript_in_document_aa(self, make_pdf_with_page):
        """JavaScript in Document AA is removed as non-compliant."""
        pdf = make_pdf_with_page()
        pdf.Root["/AA"] = Dictionary(
            WC=Dictionary(S=Name.JavaScript, JS="cleanup();"),
        )
        result = remove_actions(pdf)
        assert result >= 1
        assert "/AA" not in pdf.Root

    def test_removes_javascript_annotation_action(self, make_pdf_with_page):
        """JavaScript annotation actions are removed as non-compliant."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.JavaScript, JS="click();"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_removes_javascript_in_acroform_field(self, make_pdf_with_page):
        """JavaScript actions on AcroForm fields are removed."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                A=Dictionary(S=Name.JavaScript, JS="validate();"),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field

    def test_multiple_pages_with_actions(self, make_pdf_with_page):
        """Actions across multiple pages are all removed."""
        pdf = make_pdf_with_page()
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page2)

        pdf.pages[0]["/AA"] = Dictionary(
            O=Dictionary(S=Name.Launch, F="a.exe"),
        )
        pdf.pages[1]["/AA"] = Dictionary(
            O=Dictionary(S=Name.Sound),
        )
        result = remove_actions(pdf)
        assert result >= 2
        assert "/AA" not in pdf.pages[0]
        assert "/AA" not in pdf.pages[1]


class TestNextChainSanitization:
    """Tests for /Next action chain traversal (ISO 19005-2 6.6.1)."""

    def test_open_action_with_non_compliant_next(self, make_pdf_with_page):
        """Compliant OpenAction with forbidden /Next is sanitized."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Dictionary(S=Name.Launch, F="evil.exe"),
        )
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" in pdf.Root
        open_action = pdf.Root.OpenAction
        assert "/Next" not in open_action

    def test_open_action_with_compliant_next(self, make_pdf_with_page):
        """Compliant OpenAction with compliant /Next is preserved."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Dictionary(
                S=Name.URI,
                URI="https://example.com",
            ),
        )
        result = remove_actions(pdf)
        assert result == 0
        assert "/Next" in pdf.Root.OpenAction

    def test_next_array_mixed(self, make_pdf_with_page):
        """Array /Next with mix of compliant/non-compliant actions."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Array(
                [
                    Dictionary(S=Name.URI, URI="https://example.com"),
                    Dictionary(S=Name.Launch, F="evil.exe"),
                    Dictionary(S=Name.GoTo, D=Array([pdf.pages[0].obj, Name.Fit])),
                ]
            ),
        )
        result = remove_actions(pdf)
        assert result == 1
        open_action = pdf.Root.OpenAction
        assert "/Next" in open_action
        # Two compliant actions should remain
        next_val = open_action.Next
        try:
            next_val = next_val.get_object()
        except (AttributeError, ValueError, TypeError):
            pass
        assert isinstance(next_val, Array)
        assert len(next_val) == 2

    def test_next_array_all_non_compliant(self, make_pdf_with_page):
        """/Next array with all non-compliant actions is removed entirely."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Array(
                [
                    Dictionary(S=Name.Launch, F="a.exe"),
                    Dictionary(S=Name.Sound),
                ]
            ),
        )
        result = remove_actions(pdf)
        assert result == 2
        assert "/Next" not in pdf.Root.OpenAction

    def test_next_array_single_remaining_collapsed(self, make_pdf_with_page):
        """/Next array with one remaining entry is collapsed to a dict."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Array(
                [
                    Dictionary(S=Name.URI, URI="https://example.com"),
                    Dictionary(S=Name.Launch, F="evil.exe"),
                ]
            ),
        )
        result = remove_actions(pdf)
        assert result == 1
        next_val = pdf.Root.OpenAction.Next
        try:
            next_val = next_val.get_object()
        except (AttributeError, ValueError, TypeError):
            pass
        # Should be collapsed to a single dict, not an array
        assert isinstance(next_val, Dictionary)
        assert str(next_val.get("/S")) == "/URI"

    def test_deeply_nested_next_chain(self, make_pdf_with_page):
        """Forbidden action hidden 3 levels deep in /Next chain."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Dictionary(
                S=Name.URI,
                URI="https://example.com",
                Next=Dictionary(
                    S=Name.GoTo,
                    D=Array([pdf.pages[0].obj, Name.Fit]),
                    Next=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            ),
        )
        result = remove_actions(pdf)
        assert result == 1
        # First two levels preserved
        assert "/OpenAction" in pdf.Root
        level1 = pdf.Root.OpenAction.Next
        assert str(level1.get("/S")) == "/URI"
        level2 = level1.Next
        assert str(level2.get("/S")) == "/GoTo"
        assert "/Next" not in level2

    def test_annotation_action_with_non_compliant_next(self, make_pdf_with_page):
        """Annotation compliant /A with forbidden /Next is sanitized."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(
                    S=Name.URI,
                    URI="https://example.com",
                    Next=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result == 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" in annot
        assert "/Next" not in annot.A

    def test_field_action_with_non_compliant_next(self, make_pdf_with_page):
        """AcroForm field /A with forbidden /Next is fully removed (Rule 6.4.1)."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                A=Dictionary(
                    S=Name.URI,
                    URI="https://example.com",
                    Next=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result == 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field

    def test_page_aa_action_with_non_compliant_next(self, make_pdf_with_page):
        """Page AA is removed unconditionally (ISO 19005-2 Section 6.6.2)."""
        pdf = make_pdf_with_page()
        pdf.pages[0]["/AA"] = Dictionary(
            O=Dictionary(
                S=Name.GoTo,
                D=Array([pdf.pages[0].obj, Name.Fit]),
                Next=Dictionary(S=Name.Launch, F="evil.exe"),
            ),
        )
        result = remove_actions(pdf)
        assert result >= 1
        assert "/AA" not in pdf.pages[0]


class TestRemoveActionsFromFields:
    """Tests for _remove_actions_from_fields()."""

    def test_empty_fields_array(self):
        """Returns 0 for empty fields array."""
        fields = Array([])
        result = _remove_actions_from_fields(fields)
        assert result == 0

    def test_field_with_compliant_action_removed(self, make_pdf_with_page):
        """Compliant actions (URI) on fields are removed in PDF/A (Rule 6.4.1)."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="link",
                A=Dictionary(S=Name.URI, URI="https://example.com"),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        fields = pdf.Root.AcroForm.Fields
        result = _remove_actions_from_fields(fields)
        assert result == 1
        field = fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field

    def test_field_with_both_a_and_aa(self, make_pdf_with_page):
        """Both /A and /AA non-compliant actions on a field removed."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                A=Dictionary(S=Name.Launch, F="evil.exe"),
                AA=Dictionary(
                    K=Dictionary(S=Name.Sound),
                ),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        fields = pdf.Root.AcroForm.Fields
        result = _remove_actions_from_fields(fields)
        assert result >= 2
        field = fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field
        assert "/AA" not in field


class TestWidgetFieldActionsRule641:
    """Tests for PDF/A Rule 6.4.1: Widgets/Fields must not have /A or /AA."""

    def test_widget_compliant_action_removed(self, make_pdf_with_page):
        """Widget annotation with compliant /A (GoTo) is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(
                    S=Name.GoTo,
                    D=Array([pdf.pages[0].obj, Name.Fit]),
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_widget_compliant_aa_removed(self, make_pdf_with_page):
        """Widget annotation with compliant /AA (GoTo) is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                AA=Dictionary(
                    E=Dictionary(
                        S=Name.GoTo,
                        D=Array([pdf.pages[0].obj, Name.Fit]),
                    ),
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/AA" not in annot

    def test_field_compliant_action_removed(self, make_pdf_with_page):
        """Field with compliant /A (GoTo) is removed via remove_actions()."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                A=Dictionary(
                    S=Name.GoTo,
                    D=Array([pdf.pages[0].obj, Name.Fit]),
                ),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/A" not in field

    def test_field_compliant_aa_removed(self, make_pdf_with_page):
        """Field with compliant /AA is removed."""
        pdf = make_pdf_with_page()
        field = pdf.make_indirect(
            Dictionary(
                T="field1",
                AA=Dictionary(
                    K=Dictionary(
                        S=Name.GoTo,
                        D=Array([pdf.pages[0].obj, Name.Fit]),
                    ),
                ),
            )
        )
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        field = pdf.Root.AcroForm.Fields[0]
        try:
            field = field.get_object()
        except AttributeError:
            pass
        assert "/AA" not in field

    def test_non_widget_compliant_action_kept(self, make_pdf_with_page):
        """Link annotation with compliant /A (URI) is kept (regression)."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.URI, URI="https://example.com"),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result == 0
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" in annot


def _make_pdf_with_outline_actions(actions):
    """Create a minimal PDF with bookmarks carrying the given actions.

    Each entry in *actions* becomes one top-level outline item.  An entry
    can be a single action Dictionary, or a tuple (action, [child_actions])
    for nested bookmarks.

    Returns:
        A save-and-reopened Pdf object.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    def _build_item(pdf, title, action, child_actions=None):
        item = Dictionary(Title=title, Count=0)
        if action is not None:
            item["/A"] = action
        item = pdf.make_indirect(item)
        if child_actions:
            children = []
            for ci, ca in enumerate(child_actions):
                children.append(_build_item(pdf, f"{title}-child{ci}", ca))
            for ci, child in enumerate(children):
                child["/Parent"] = item
                if ci > 0:
                    child["/Prev"] = children[ci - 1]
                if ci < len(children) - 1:
                    child["/Next"] = children[ci + 1]
            item["/First"] = children[0]
            item["/Last"] = children[-1]
            item["/Count"] = len(children)
        return item

    items = []
    for i, entry in enumerate(actions):
        if isinstance(entry, tuple):
            action, child_actions = entry
        else:
            action = entry
            child_actions = None
        items.append(_build_item(pdf, f"BM{i}", action, child_actions))

    root_outline = pdf.make_indirect(Dictionary(Type=Name.Outlines, Count=len(items)))
    for i, item in enumerate(items):
        item["/Parent"] = root_outline
        if i > 0:
            item["/Prev"] = items[i - 1]
        if i < len(items) - 1:
            item["/Next"] = items[i + 1]
    if items:
        root_outline["/First"] = items[0]
        root_outline["/Last"] = items[-1]
    pdf.Root["/Outlines"] = root_outline
    return save_and_reopen(pdf)


class TestOutlineActions:
    """Tests for outline (bookmark) action sanitization."""

    def test_removes_non_compliant_outline_action(self):
        """Non-compliant /Launch action on a bookmark is removed."""
        pdf = _make_pdf_with_outline_actions(
            [
                Dictionary(S=Name.Launch, F="evil.exe"),
            ]
        )
        result = remove_actions(pdf)
        assert result >= 1
        first = pdf.Root.Outlines.First
        try:
            first = first.get_object()
        except (AttributeError, ValueError, TypeError):
            pass
        assert "/A" not in first

    def test_keeps_compliant_outline_action(self):
        """Compliant /GoTo action on a bookmark is preserved."""
        pdf = _make_pdf_with_outline_actions(
            [
                Dictionary(
                    S=Name.GoTo,
                    D=Array([Name("/Fit")]),
                ),
            ]
        )
        result = remove_actions(pdf)
        assert result == 0
        first = pdf.Root.Outlines.First
        try:
            first = first.get_object()
        except (AttributeError, ValueError, TypeError):
            pass
        assert "/A" in first

    def test_removes_multiple_outline_actions(self):
        """Multiple non-compliant bookmark actions across siblings."""
        pdf = _make_pdf_with_outline_actions(
            [
                Dictionary(S=Name.Launch, F="a.exe"),
                Dictionary(S=Name.Sound),
            ]
        )
        result = remove_actions(pdf)
        assert result >= 2

    def test_removes_nested_outline_actions(self):
        """Non-compliant actions in child outline items are removed."""
        pdf = _make_pdf_with_outline_actions(
            [
                (
                    Dictionary(S=Name.GoTo, D=Array([Name("/Fit")])),
                    [Dictionary(S=Name.Movie)],
                ),
            ]
        )
        result = remove_actions(pdf)
        assert result >= 1

    def test_outline_action_next_chain_sanitized(self):
        """Compliant outline action with forbidden /Next is sanitized."""
        pdf = _make_pdf_with_outline_actions(
            [
                Dictionary(
                    S=Name.GoTo,
                    D=Array([Name("/Fit")]),
                    Next=Dictionary(S=Name.Launch, F="evil.exe"),
                ),
            ]
        )
        result = remove_actions(pdf)
        assert result == 1
        first = pdf.Root.Outlines.First
        try:
            first = first.get_object()
        except (AttributeError, ValueError, TypeError):
            pass
        assert "/A" in first
        assert "/Next" not in first.A

    def test_no_outlines(self, make_pdf_with_page):
        """PDF without /Outlines does not cause errors."""
        pdf = make_pdf_with_page()
        result = remove_actions(pdf)
        assert result == 0


class TestRemoveActionsFromOutlines:
    """Tests for _remove_actions_from_outlines() directly."""

    def test_empty_outline_root(self):
        """Returns 0 for outline root with no children."""
        pdf = new_pdf()
        root = pdf.make_indirect(Dictionary(Type=Name.Outlines, Count=0))
        result = _remove_actions_from_outlines(root)
        assert result == 0

    def test_outline_without_action(self):
        """Bookmark with /Dest (no /A) is left untouched."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        item = pdf.make_indirect(
            Dictionary(
                Title="No Action",
                Dest=Array([pdf.pages[0].obj, Name.Fit]),
            )
        )
        root = pdf.make_indirect(
            Dictionary(
                Type=Name.Outlines,
                Count=1,
                First=item,
                Last=item,
            )
        )
        item["/Parent"] = root
        result = _remove_actions_from_outlines(root)
        assert result == 0


class TestNamedActions:
    """Tests for Named action validation (ISO 19005-2 Clause 6.6.1)."""

    def test_keeps_allowed_named_action_nextpage(self, make_pdf_with_page):
        """Named action with /N /NextPage is allowed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name.NextPage)
        result = remove_actions(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_keeps_allowed_named_action_prevpage(self, make_pdf_with_page):
        """Named action with /N /PrevPage is allowed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name.PrevPage)
        result = remove_actions(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_keeps_allowed_named_action_firstpage(self, make_pdf_with_page):
        """Named action with /N /FirstPage is allowed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name.FirstPage)
        result = remove_actions(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_keeps_allowed_named_action_lastpage(self, make_pdf_with_page):
        """Named action with /N /LastPage is allowed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name.LastPage)
        result = remove_actions(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_removes_forbidden_named_action_print(self, make_pdf_with_page):
        """Named action with /N /Print is forbidden and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name.Print)
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_forbidden_named_action_saveas(self, make_pdf_with_page):
        """Named action with /N /SaveAs is forbidden and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name("/SaveAs"))
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_forbidden_named_action_find(self, make_pdf_with_page):
        """Named action with /N /Find is forbidden and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name("/Find"))
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_forbidden_named_action_goback(self, make_pdf_with_page):
        """Named action with /N /GoBack is forbidden and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name("/GoBack"))
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_forbidden_named_action_goforward(self, make_pdf_with_page):
        """Named action with /N /GoForward is forbidden and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named, N=Name("/GoForward"))
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_removes_named_action_without_n_entry(self, make_pdf_with_page):
        """Named action without /N entry is non-compliant and removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(S=Name.Named)
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_forbidden_named_action_in_annotation(self, make_pdf_with_page):
        """Forbidden Named action on an annotation is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.Named, N=Name.Print),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = remove_actions(pdf)
        assert result >= 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_forbidden_named_action_in_next_chain(self, make_pdf_with_page):
        """Forbidden Named action in /Next chain is removed."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Dictionary(S=Name.Named, N=Name.Print),
        )
        result = remove_actions(pdf)
        assert result == 1
        assert "/OpenAction" in pdf.Root
        assert "/Next" not in pdf.Root.OpenAction

    def test_allowed_named_action_in_next_chain_kept(self, make_pdf_with_page):
        """Allowed Named action in /Next chain is preserved."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.GoTo,
            D=Array([pdf.pages[0].obj, Name.Fit]),
            Next=Dictionary(S=Name.Named, N=Name.NextPage),
        )
        result = remove_actions(pdf)
        assert result == 0
        assert "/Next" in pdf.Root.OpenAction


def _make_fake_page_ref(pdf):
    """Create an indirect object that looks like a page but is NOT in pdf.pages."""
    fake = pdf.make_indirect(
        Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 100, 100]))
    )
    return fake


class TestValidateDestinations:
    """Tests for validate_destinations()."""

    def test_no_destinations(self, make_pdf_with_page):
        """Returns 0 for PDF without any destinations."""
        pdf = make_pdf_with_page()
        result = validate_destinations(pdf)
        assert result == 0

    def test_goto_action_valid_page(self, make_pdf_with_page):
        """GoTo action referencing a valid page is kept."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.GoTo, D=Array([pdf.pages[0].obj, Name.Fit])),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 0
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" in annot

    def test_goto_action_invalid_page(self):
        """GoTo action referencing a deleted page is removed."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.GoTo, D=Array([pdf.pages[1].obj, Name.Fit])),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_outline_valid_dest(self, make_pdf_with_page):
        """Outline item with valid /Dest is kept."""
        pdf = make_pdf_with_page()
        item = pdf.make_indirect(
            Dictionary(
                Title="Bookmark",
                Dest=Array([pdf.pages[0].obj, Name.Fit]),
            )
        )
        root_outline = pdf.make_indirect(
            Dictionary(
                Type=Name.Outlines,
                Count=1,
                First=item,
                Last=item,
            )
        )
        item["/Parent"] = root_outline
        pdf.Root["/Outlines"] = root_outline
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 0
        first = pdf.Root.Outlines.First
        try:
            first = first.get_object()
        except AttributeError:
            pass
        assert "/Dest" in first

    def test_outline_invalid_dest(self):
        """Outline item with invalid /Dest has it removed."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        item = pdf.make_indirect(
            Dictionary(
                Title="Bookmark",
                Dest=Array([pdf.pages[1].obj, Name.Fit]),
            )
        )
        root_outline = pdf.make_indirect(
            Dictionary(
                Type=Name.Outlines,
                Count=1,
                First=item,
                Last=item,
            )
        )
        item["/Parent"] = root_outline
        pdf.Root["/Outlines"] = root_outline
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        first = pdf.Root.Outlines.First
        try:
            first = first.get_object()
        except AttributeError:
            pass
        assert "/Dest" not in first
        assert "/Title" in first

    def test_link_annotation_invalid_dest(self):
        """Link annotation with invalid /Dest has it removed."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                Dest=Array([pdf.pages[1].obj, Name.Fit]),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/Dest" not in annot

    def test_named_dest_string_exists(self, make_pdf_with_page):
        """GoTo action with string dest matching a named dest is kept."""
        pdf = make_pdf_with_page()
        pdf.Root["/Names"] = Dictionary(
            Dests=Dictionary(
                Names=Array(
                    [
                        pikepdf.String("chapter1"),
                        Array([pdf.pages[0].obj, Name.Fit]),
                    ]
                ),
            ),
        )
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.GoTo, D=pikepdf.String("chapter1")),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 0
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" in annot

    def test_named_dest_string_missing(self, make_pdf_with_page):
        """GoTo action with string dest not in named dests is removed."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(S=Name.GoTo, D=pikepdf.String("nonexistent")),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 1
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" not in annot

    def test_open_action_array_invalid(self):
        """OpenAction as array with invalid page ref is removed."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        pdf.Root["/OpenAction"] = Array([pdf.pages[1].obj, Name.Fit])
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        assert "/OpenAction" not in pdf.Root

    def test_open_action_array_valid(self, make_pdf_with_page):
        """OpenAction as array with valid page ref is kept."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Array([pdf.pages[0].obj, Name.Fit])
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_gotor_action_not_validated(self, make_pdf_with_page):
        """GoToR action (external file) is not touched."""
        pdf = make_pdf_with_page()
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                A=Dictionary(
                    S=Name.GoToR,
                    F="other.pdf",
                    D=Array([0, Name.Fit]),
                ),
            )
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        pdf = save_and_reopen(pdf)
        result = validate_destinations(pdf)
        assert result == 0
        annot = pdf.pages[0].Annots[0]
        try:
            annot = annot.get_object()
        except AttributeError:
            pass
        assert "/A" in annot

    def test_named_dest_tree_pruned(self):
        """Named dest entries with invalid page refs are removed from tree."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        pdf.Root["/Names"] = Dictionary(
            Dests=Dictionary(
                Names=Array(
                    [
                        pikepdf.String("valid"),
                        Array([pdf.pages[0].obj, Name.Fit]),
                        pikepdf.String("invalid"),
                        Array([pdf.pages[1].obj, Name.Fit]),
                    ]
                ),
            ),
        )
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        names_arr = pdf.Root.Names.Dests.Names
        assert len(names_arr) == 2
        assert str(names_arr[0]) == "valid"

    def test_legacy_dests_dict_pruned(self):
        """Legacy /Root/Dests dict entries with invalid refs are removed."""
        pdf = new_pdf()
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        pdf.pages.append(page2)
        dests = Dictionary()
        dests[Name("/valid")] = Array([pdf.pages[0].obj, Name.Fit])
        dests[Name("/invalid")] = Array([pdf.pages[1].obj, Name.Fit])
        pdf.Root["/Dests"] = pdf.make_indirect(dests)
        pdf = save_and_reopen(pdf)
        del pdf.pages[1]
        result = validate_destinations(pdf)
        assert result == 1
        dests = pdf.Root.Dests
        assert "/valid" in dests
        assert "/invalid" not in dests
