# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/javascript.py."""

from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.javascript import remove_javascript


class TestRemoveJavascript:
    """Tests for remove_javascript()."""

    def test_no_javascript(self, make_pdf_with_page):
        """Returns 0 for PDF without JavaScript."""
        pdf = make_pdf_with_page()
        result = remove_javascript(pdf)
        assert result == 0

    def test_removes_named_javascript(self, make_pdf_with_page):
        """Named JavaScript in Names dict is removed."""
        pdf = make_pdf_with_page()
        js_tree = Dictionary(
            Names=Array(
                [
                    "script1",
                    Dictionary(S=Name.JavaScript, JS="alert('hi');"),
                ]
            )
        )
        pdf.Root["/Names"] = Dictionary(JavaScript=js_tree)
        result = remove_javascript(pdf)
        assert result == 1
        assert "/JavaScript" not in pdf.Root.Names

    def test_no_names_dict(self, make_pdf_with_page):
        """Returns 0 when no Names dict exists."""
        pdf = make_pdf_with_page()
        result = remove_javascript(pdf)
        assert result == 0

    def test_names_without_javascript(self, make_pdf_with_page):
        """Returns 0 when Names dict has no JavaScript entry."""
        pdf = make_pdf_with_page()
        pdf.Root["/Names"] = Dictionary(EmbeddedFiles=Dictionary(Names=Array([])))
        result = remove_javascript(pdf)
        assert result == 0
        assert "/Names" in pdf.Root

    def test_preserves_other_names(self, make_pdf_with_page):
        """Other entries in Names dict are preserved."""
        pdf = make_pdf_with_page()
        js_tree = Dictionary(
            Names=Array(
                [
                    "script1",
                    Dictionary(S=Name.JavaScript, JS="alert('hi');"),
                ]
            )
        )
        pdf.Root["/Names"] = Dictionary(
            JavaScript=js_tree,
            EmbeddedFiles=Dictionary(Names=Array([])),
        )
        result = remove_javascript(pdf)
        assert result == 1
        assert "/JavaScript" not in pdf.Root.Names
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_does_not_touch_open_action(self, make_pdf_with_page):
        """JavaScript OpenAction is not removed (handled by remove_actions)."""
        pdf = make_pdf_with_page()
        pdf.Root["/OpenAction"] = Dictionary(
            S=Name.JavaScript, JS="app.alert('Hello');"
        )
        result = remove_javascript(pdf)
        assert result == 0
        assert "/OpenAction" in pdf.Root

    def test_does_not_touch_document_aa(self, make_pdf_with_page):
        """Document AA is not removed (handled by remove_actions)."""
        pdf = make_pdf_with_page()
        pdf.Root["/AA"] = Dictionary(
            WC=Dictionary(S=Name.JavaScript, JS="cleanup();"),
        )
        result = remove_javascript(pdf)
        assert result == 0
        assert "/AA" in pdf.Root
