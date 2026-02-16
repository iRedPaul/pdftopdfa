# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/xfa.py."""

import logging

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.xfa import remove_xfa_forms


@pytest.fixture
def pdf_with_xfa():
    """PDF with XFA form data and NeedsRendering flag."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    pdf.Root["/AcroForm"] = pdf.make_indirect(
        Dictionary(
            Fields=Array([]),
            XFA=pdf.make_stream(b"<xfa>test</xfa>"),
            NeedsRendering=True,
        )
    )
    yield pdf


@pytest.fixture
def pdf_with_xfa_only():
    """PDF with only XFA (no NeedsRendering)."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    pdf.Root["/AcroForm"] = pdf.make_indirect(
        Dictionary(
            Fields=Array([]),
            XFA=pdf.make_stream(b"<xfa>data</xfa>"),
        )
    )
    yield pdf


class TestRemoveXfaForms:
    """Tests for remove_xfa_forms()."""

    def test_removes_xfa_and_needs_rendering(self, pdf_with_xfa):
        """Both /XFA and /NeedsRendering are removed."""
        result = remove_xfa_forms(pdf_with_xfa)
        assert result == 2
        acroform = pdf_with_xfa.Root.AcroForm
        assert "/XFA" not in acroform
        assert "/NeedsRendering" not in acroform

    def test_removes_xfa_only(self, pdf_with_xfa_only):
        """Only /XFA is removed when /NeedsRendering absent."""
        result = remove_xfa_forms(pdf_with_xfa_only)
        assert result == 1
        acroform = pdf_with_xfa_only.Root.AcroForm
        assert "/XFA" not in acroform

    def test_no_acroform(self, sample_pdf_obj):
        """Returns 0 when no AcroForm exists."""
        result = remove_xfa_forms(sample_pdf_obj)
        assert result == 0

    def test_acroform_without_xfa(self):
        """Returns 0 when AcroForm has no XFA or NeedsRendering."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([])))
        result = remove_xfa_forms(pdf)
        assert result == 0

    def test_needs_rendering_only(self):
        """Removes /NeedsRendering even without /XFA."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(Fields=Array([]), NeedsRendering=True)
        )
        result = remove_xfa_forms(pdf)
        assert result == 1
        assert "/NeedsRendering" not in pdf.Root.AcroForm

    def test_preserves_other_acroform_keys(self, pdf_with_xfa):
        """Other AcroForm keys like /Fields are preserved."""
        remove_xfa_forms(pdf_with_xfa)
        acroform = pdf_with_xfa.Root.AcroForm
        assert "/Fields" in acroform

    def test_xfa_as_array(self):
        """Handles /XFA as array of name/stream pairs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        xfa_array = Array(
            [
                "preamble",
                pdf.make_stream(b"<xfa>preamble</xfa>"),
                "config",
                pdf.make_stream(b"<xfa>config</xfa>"),
            ]
        )
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(Fields=Array([]), XFA=xfa_array)
        )
        result = remove_xfa_forms(pdf)
        assert result == 1
        assert "/XFA" not in pdf.Root.AcroForm


class TestPureXfaWarning:
    """Tests for pure-XFA PDF detection and warning."""

    def test_pure_xfa_no_fields_logs_warning(self, caplog):
        """Pure-XFA PDF with no /Fields logs a WARNING."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        # AcroForm with /XFA but no /Fields at all
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(XFA=pdf.make_stream(b"<xfa>data</xfa>"))
        )

        with caplog.at_level(logging.WARNING):
            result = remove_xfa_forms(pdf)

        assert result == 1
        assert "/XFA" not in pdf.Root.AcroForm
        assert any(
            "Pure-XFA" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_pure_xfa_empty_fields_logs_warning(self, caplog):
        """Pure-XFA PDF with empty /Fields array logs a WARNING."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(
                Fields=Array([]),
                XFA=pdf.make_stream(b"<xfa>data</xfa>"),
            )
        )

        with caplog.at_level(logging.WARNING):
            result = remove_xfa_forms(pdf)

        assert result == 1
        assert any(
            "Pure-XFA" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_xfa_with_fields_no_pure_xfa_warning(self, caplog):
        """XFA PDF with non-empty /Fields does NOT log pure-XFA warning."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        field = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name("/Tx"),
                T="field1",
                Rect=Array([0, 0, 100, 50]),
            )
        )
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(
                Fields=Array([field]),
                XFA=pdf.make_stream(b"<xfa>data</xfa>"),
            )
        )

        with caplog.at_level(logging.WARNING):
            result = remove_xfa_forms(pdf)

        assert result == 1
        assert not any("Pure-XFA" in r.message for r in caplog.records)

    def test_pure_xfa_still_removes_xfa(self, caplog):
        """Pure-XFA PDF has /XFA removed despite the warning."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(XFA=pdf.make_stream(b"<xfa>form</xfa>"))
        )

        with caplog.at_level(logging.WARNING):
            remove_xfa_forms(pdf)

        assert "/XFA" not in pdf.Root.AcroForm
