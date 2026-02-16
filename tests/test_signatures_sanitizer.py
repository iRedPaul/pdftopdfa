# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/signatures.py."""

import logging

from conftest import resolve
from pikepdf import Array, Dictionary, Name

from pdftopdfa.sanitizers.signatures import (
    _collect_signature_fields,
    sanitize_signatures,
)


def _make_sig_dict(pdf, with_type=True):
    """Create a minimal signature dictionary."""
    sig = Dictionary()
    if with_type:
        sig[Name.Type] = Name.Sig
    sig[Name.Filter] = Name("/Adobe.PPKLite")
    sig[Name.SubFilter] = Name("/adbe.pkcs7.detached")
    sig["/ByteRange"] = Array([0, 100, 200, 300])
    sig["/Contents"] = pdf.make_stream(b"\x00" * 64)
    return pdf.make_indirect(sig)


def _make_sig_field(pdf, sig_dict=None, field_name="Signature1"):
    """Create a signature field dictionary."""
    field = Dictionary()
    field[Name.Type] = Name.Annot
    field["/Subtype"] = Name.Widget
    field["/FT"] = Name.Sig
    field["/T"] = field_name
    field["/Rect"] = Array([0, 0, 200, 50])
    if sig_dict is not None:
        field["/V"] = sig_dict
    return pdf.make_indirect(field)


def _setup_acroform(pdf, fields, sigflags=None):
    """Set up AcroForm with given fields."""
    acroform = Dictionary(Fields=Array(fields))
    if sigflags is not None:
        acroform["/SigFlags"] = sigflags
    pdf.Root["/AcroForm"] = pdf.make_indirect(acroform)


def _assert_sig_dict_neutralized(sig_dict):
    """Assert that signature-specific keys were removed."""
    assert sig_dict.get("/Type") is None
    assert sig_dict.get("/Filter") is None
    assert sig_dict.get("/SubFilter") is None
    assert sig_dict.get("/ByteRange") is None
    assert sig_dict.get("/Contents") is None


class TestSanitizeSignaturesNoOp:
    """No-op scenarios."""

    def test_no_acroform(self, make_pdf_with_page):
        """No AcroForm and no signatures -> no changes."""
        pdf = make_pdf_with_page()
        result = sanitize_signatures(pdf)
        assert result["signatures_found"] == 0
        assert result["signatures_removed"] == 0
        assert result["sigflags_fixed"] == 0
        assert result["signatures_type_fixed"] == 0

    def test_unsigned_sig_field(self, make_pdf_with_page):
        """Unsigned field (/FT /Sig without /V) is ignored."""
        pdf = make_pdf_with_page()
        field = _make_sig_field(pdf, sig_dict=None)
        _setup_acroform(pdf, [field])

        result = sanitize_signatures(pdf)
        assert result["signatures_found"] == 0
        assert result["signatures_removed"] == 0


class TestSanitizeSignaturesRemoval:
    """Signature removal and neutralization."""

    def test_signed_field_v_removed_and_dict_neutralized(self, make_pdf_with_page):
        """Signed field has /V removed and its signature dictionary neutralized."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)
        _setup_acroform(pdf, [field])

        result = sanitize_signatures(pdf, "3b")

        assert result["signatures_found"] == 1
        assert result["signatures_removed"] == 2
        assert "/V" not in field
        _assert_sig_dict_neutralized(sig)

    def test_page_annotation_signature_is_handled(self, make_pdf_with_page):
        """Signature field only in page /Annots is still handled."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)
        _setup_acroform(pdf, [])
        pdf.pages[0].obj["/Annots"] = Array([field])

        result = sanitize_signatures(pdf, "2b")

        assert result["signatures_found"] == 1
        assert result["signatures_removed"] == 2
        assert "/V" not in field
        _assert_sig_dict_neutralized(sig)

    def test_catalog_perms_signature_is_removed(self, make_pdf_with_page):
        """Signatures in /Root/Perms are dereferenced and neutralized."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        pdf.Root["/Perms"] = pdf.make_indirect(Dictionary(DocMDP=sig))

        result = sanitize_signatures(pdf, "2b")

        assert result["signatures_found"] == 1
        assert result["signatures_removed"] == 2
        assert "/Perms" not in pdf.Root
        _assert_sig_dict_neutralized(sig)

    def test_orphan_signature_dict_is_neutralized(self, make_pdf_with_page):
        """Unreferenced signature dictionaries are neutralized by global scan."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        assert sig.get("/ByteRange") is not None

        result = sanitize_signatures(pdf)

        assert result["signatures_found"] == 1
        assert result["signatures_removed"] == 1
        _assert_sig_dict_neutralized(sig)


class TestSanitizeSignaturesSigFlags:
    """SigFlags cleanup after signature removal."""

    def test_sigflags_bit_1_is_cleared(self, make_pdf_with_page):
        """Bit 1 (SignaturesExist) is cleared while preserving other bits."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)
        _setup_acroform(pdf, [field], sigflags=3)  # bits 1 + 2

        result = sanitize_signatures(pdf)

        assert result["sigflags_fixed"] == 1
        acroform = resolve(pdf.Root["/AcroForm"])
        assert int(acroform["/SigFlags"]) == 2

    def test_sigflags_removed_when_zero(self, make_pdf_with_page):
        """SigFlags entry is removed when only bit 1 was present."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)
        _setup_acroform(pdf, [field], sigflags=1)

        result = sanitize_signatures(pdf)

        assert result["sigflags_fixed"] == 1
        acroform = resolve(pdf.Root["/AcroForm"])
        assert "/SigFlags" not in acroform


class TestSanitizeSignaturesLogging:
    """Logging behavior."""

    def test_warning_logged_when_signatures_found(self, make_pdf_with_page, caplog):
        """A warning is emitted when signatures are detected."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)
        _setup_acroform(pdf, [field])

        with caplog.at_level(logging.WARNING):
            sanitize_signatures(pdf)

        assert any(
            "digital signature dictionary" in r.message.lower() for r in caplog.records
        )

    def test_no_warning_without_signatures(self, make_pdf_with_page, caplog):
        """No warning is emitted when no signatures are found."""
        pdf = make_pdf_with_page()
        _setup_acroform(pdf, [])

        with caplog.at_level(logging.WARNING):
            sanitize_signatures(pdf)

        assert not any(
            "digital signature dictionary" in r.message.lower() for r in caplog.records
        )


class TestCollectSignatureFields:
    """Tests for _collect_signature_fields()."""

    def test_recursive_kids(self, make_pdf_with_page):
        """Fields nested in /Kids are collected."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        child = _make_sig_field(pdf, sig)

        parent = pdf.make_indirect(Dictionary(Kids=Array([child])))
        child["/Parent"] = parent

        result = _collect_signature_fields(Array([parent]))
        assert len(result) == 1

    def test_cycle_detection(self, make_pdf_with_page):
        """Cycles in /Kids do not cause infinite loops."""
        pdf = make_pdf_with_page()
        sig = _make_sig_dict(pdf)
        field = _make_sig_field(pdf, sig)

        parent = pdf.make_indirect(Dictionary(Kids=Array([field])))
        field["/Parent"] = parent
        field["/Kids"] = Array([parent])

        result = _collect_signature_fields(Array([parent]))
        assert len(result) <= 2

    def test_empty_fields(self):
        """Empty fields array returns empty list."""
        result = _collect_signature_fields(Array([]))
        assert result == []
