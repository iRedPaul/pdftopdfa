# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for utils.py."""

import logging
from pathlib import Path

import pikepdf
import pytest
from conftest import new_pdf, open_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers import (
    ANNOT_FLAG_HIDDEN,
    ANNOT_FLAG_INVISIBLE,
    ANNOT_FLAG_NOROTATE,
    ANNOT_FLAG_NOVIEW,
    ANNOT_FLAG_NOZOOM,
    ANNOT_FLAG_PRINT,
    SUBMITFORM_FLAG_EXPORTFORMAT,
    SUBMITFORM_FLAG_SUBMITPDF,
    SUBMITFORM_FLAG_XFDF,
    _is_javascript_action,
    _is_non_compliant_action,
    ensure_af_relationships,
    ensure_appearance_streams,
    fix_annotation_flags,
    fix_image_interpolate,
    remove_embedded_files,
    remove_forbidden_annotations,
    remove_forbidden_xobjects,
    remove_javascript,
    remove_xfa_forms,
    sanitize_for_pdfa,
)
from pdftopdfa.sanitizers import (
    remove_actions as _remove_actions,
)
from pdftopdfa.utils import (
    LOG_FORMAT,
    SUPPORTED_LEVELS,
    get_pdf_version,
    get_required_pdf_version,
    is_pdf_encrypted,
    setup_logging,
    validate_pdfa_level,
)
from pdftopdfa.utils import (
    resolve_indirect as _resolve_indirect,
)


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_default_level_is_info(self) -> None:
        """Default log level is INFO."""
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_verbose_sets_debug(self) -> None:
        """verbose=True sets DEBUG level."""
        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_quiet_sets_error(self) -> None:
        """quiet=True sets ERROR level."""
        logger = setup_logging(quiet=True)
        assert logger.level == logging.ERROR

    def test_quiet_takes_precedence(self) -> None:
        """quiet takes precedence over verbose."""
        logger = setup_logging(verbose=True, quiet=True)
        assert logger.level == logging.ERROR

    def test_returns_pdftopdfa_logger(self) -> None:
        """Returns pdftopdfa logger."""
        logger = setup_logging()
        assert logger.name == "pdftopdfa"

    def test_has_handler(self) -> None:
        """Logger has exactly one handler."""
        logger = setup_logging()
        assert len(logger.handlers) == 1

    def test_handler_has_formatter(self) -> None:
        """Handler has correct format."""
        logger = setup_logging()
        handler = logger.handlers[0]
        assert handler.formatter is not None
        assert handler.formatter._fmt == LOG_FORMAT


class TestIsPdfEncrypted:
    """Tests for is_pdf_encrypted."""

    def test_unencrypted_pdf(self) -> None:
        """Unencrypted PDF returns False."""
        pdf = new_pdf()
        assert is_pdf_encrypted(pdf) is False

    def test_encrypted_pdf(self, tmp_path: Path) -> None:
        """Encrypted PDF returns True."""
        # Create encrypted PDF
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        encrypted_path = tmp_path / "encrypted.pdf"
        pdf.save(encrypted_path, encryption=pikepdf.Encryption(owner="test"))

        # Open encrypted PDF
        encrypted_pdf = open_pdf(encrypted_path)
        assert is_pdf_encrypted(encrypted_pdf) is True


class TestGetPdfVersion:
    """Tests for get_pdf_version."""

    def test_returns_version_string(self) -> None:
        """Returns version as string."""
        pdf = new_pdf()
        version = get_pdf_version(pdf)
        assert isinstance(version, str)
        # pikepdf creates PDF 1.x by default
        assert version.startswith("1.") or version.startswith("2.")


class TestIsJavascriptAction:
    """Tests for _is_javascript_action."""

    def test_javascript_action(self) -> None:
        """JavaScript Action is detected."""
        action = Dictionary(S=Name.JavaScript, JS="app.alert('test');")
        assert _is_javascript_action(action) is True

    def test_goto_action(self) -> None:
        """GoTo Action is not JavaScript."""
        action = Dictionary(S=Name.GoTo, D="page1")
        assert _is_javascript_action(action) is False

    def test_empty_object(self) -> None:
        """Empty object is not JavaScript."""
        action = Dictionary()
        assert _is_javascript_action(action) is False


class TestIsNonCompliantAction:
    """Tests for _is_non_compliant_action."""

    # --- Non-compliant action types ---

    @pytest.mark.parametrize(
        "action_name",
        ["JavaScript", "Launch", "Sound", "Movie", "SetState", "ImportData"],
    )
    def test_previously_blocklisted_actions(self, action_name: str) -> None:
        """Actions that were on the old blocklist are still non-compliant."""
        action = Dictionary(S=Name(f"/{action_name}"))
        assert _is_non_compliant_action(action) is True

    @pytest.mark.parametrize(
        "action_name",
        ["GoTo3DView", "RichMediaExecute", "Trans", "ResetForm", "Hide", "SetOCGState"],
    )
    def test_newly_caught_non_compliant_actions(self, action_name: str) -> None:
        """Actions now caught by allowlist that the old blocklist missed."""
        action = Dictionary(S=Name(f"/{action_name}"))
        assert _is_non_compliant_action(action) is True

    def test_unknown_action_is_non_compliant(self) -> None:
        """Unknown/future action type is treated as non-compliant."""
        action = Dictionary(S=Name("/SomeFutureAction"))
        assert _is_non_compliant_action(action) is True

    # --- Compliant action types ---

    @pytest.mark.parametrize(
        "action_name",
        ["GoTo", "GoToR", "GoToE", "Thread", "URI"],
    )
    def test_compliant_actions(self, action_name: str) -> None:
        """All ISO 19005-2 allowed actions are compliant."""
        action = Dictionary(S=Name(f"/{action_name}"))
        assert _is_non_compliant_action(action) is False

    # --- Named: /N validation (ISO 19005-2 Clause 6.6.1) ---

    @pytest.mark.parametrize(
        "named_action",
        ["/NextPage", "/PrevPage", "/FirstPage", "/LastPage"],
    )
    def test_named_allowed_actions_are_compliant(self, named_action: str) -> None:
        """Named actions with permitted /N values are compliant."""
        action = Dictionary(S=Name("/Named"), N=Name(named_action))
        assert _is_non_compliant_action(action) is False

    def test_named_without_n_is_non_compliant(self) -> None:
        """Named action without /N key is non-compliant."""
        action = Dictionary(S=Name("/Named"))
        assert _is_non_compliant_action(action) is True

    def test_named_forbidden_action_is_non_compliant(self) -> None:
        """Named action with disallowed /N value is non-compliant."""
        action = Dictionary(S=Name("/Named"), N=Name("/Print"))
        assert _is_non_compliant_action(action) is True

    # --- SubmitForm: flags validation (ISO 19005-2 Clause 6.6.1) ---

    def test_submitform_pdf_format_is_compliant(self) -> None:
        """SubmitForm with SubmitPDF flag (bit 9) is compliant."""
        action = Dictionary(
            S=Name("/SubmitForm"),
            Flags=SUBMITFORM_FLAG_SUBMITPDF,
        )
        assert _is_non_compliant_action(action) is False

    def test_submitform_xfdf_format_is_compliant(self) -> None:
        """SubmitForm with XFDF flag (bit 6) is compliant."""
        action = Dictionary(
            S=Name("/SubmitForm"),
            Flags=SUBMITFORM_FLAG_XFDF,
        )
        assert _is_non_compliant_action(action) is False

    def test_submitform_no_flags_is_non_compliant(self) -> None:
        """SubmitForm without Flags defaults to FDF — non-compliant."""
        action = Dictionary(S=Name("/SubmitForm"))
        assert _is_non_compliant_action(action) is True

    def test_submitform_fdf_format_is_non_compliant(self) -> None:
        """SubmitForm with flags=0 (FDF format) is non-compliant."""
        action = Dictionary(S=Name("/SubmitForm"), Flags=0)
        assert _is_non_compliant_action(action) is True

    def test_submitform_html_format_is_non_compliant(self) -> None:
        """SubmitForm with ExportFormat flag (HTML) is non-compliant."""
        action = Dictionary(
            S=Name("/SubmitForm"),
            Flags=SUBMITFORM_FLAG_EXPORTFORMAT,
        )
        assert _is_non_compliant_action(action) is True

    # --- Edge case: missing /S key ---

    def test_missing_s_key_is_non_compliant(self) -> None:
        """Action without /S key is malformed and non-compliant."""
        action = Dictionary()
        assert _is_non_compliant_action(action) is True


class TestResolveIndirect:
    """Tests for resolve_indirect."""

    def test_returns_original_on_runtime_error(self) -> None:
        """resolve_indirect returns the original object on RuntimeError."""

        class FakeObj:
            def get_object(self):
                raise RuntimeError("get_object failed")

        obj = FakeObj()
        result = _resolve_indirect(obj)
        assert result is obj


class TestRemoveJavascript:
    """Tests for remove_javascript."""

    def test_removes_named_javascript(self) -> None:
        """Removes Named JavaScript from Names Dictionary."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add Named JavaScript
        js_tree = Dictionary(Names=Array(["test", Dictionary(JS="alert('test');")]))
        names = Dictionary(JavaScript=js_tree)
        pdf.Root.Names = names

        removed = remove_javascript(pdf)
        assert removed >= 1
        assert "/JavaScript" not in pdf.Root.Names

    def test_does_not_remove_openaction(self) -> None:
        """JavaScript OpenAction is not removed (handled by remove_actions)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        pdf.Root.OpenAction = Dictionary(S=Name.JavaScript, JS="alert('open');")

        removed = remove_javascript(pdf)
        assert removed == 0
        assert "/OpenAction" in pdf.Root

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        removed = remove_javascript(pdf)
        assert removed == 0

    def test_does_not_remove_catalog_aa(self) -> None:
        """Catalog /AA is not removed (handled by remove_actions)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        pdf.Root.AA = Dictionary(
            WC=Dictionary(S=Name.JavaScript, JS="alert('test');"),
            WS=Dictionary(S=Name.GoTo, D="page1"),
        )

        removed = remove_javascript(pdf)
        assert removed == 0
        assert "/AA" in pdf.Root


class TestRemoveActions:
    """Tests for _remove_actions."""

    def test_removes_launch_openaction(self) -> None:
        """Removes Launch OpenAction."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.Root.OpenAction = Dictionary(S=Name.Launch, F="malware.exe")

        removed = _remove_actions(pdf)
        assert removed >= 1
        assert "/OpenAction" not in pdf.Root

    def test_preserves_goto_openaction(self) -> None:
        """Preserves GoTo OpenAction."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.Root.OpenAction = Dictionary(S=Name.GoTo, D="page1")

        removed = _remove_actions(pdf)
        assert removed == 0
        assert "/OpenAction" in pdf.Root

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        removed = _remove_actions(pdf)
        assert removed == 0

    def test_preserves_gotor_openaction(self) -> None:
        """Preserves GoToR OpenAction (allowed in PDF/A-2/3)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.Root.OpenAction = Dictionary(S=Name.GoToR, F="other.pdf", D="page1")

        removed = _remove_actions(pdf)
        assert removed == 0
        assert "/OpenAction" in pdf.Root

    def test_removes_catalog_aa_with_compliant_actions(self) -> None:
        """Catalog /AA is removed even when compliant actions remain."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # AA with a non-compliant action (Launch) and a compliant action (GoTo)
        pdf.Root.AA = Dictionary(
            WC=Dictionary(S=Name.Launch, F="malware.exe"),
            WS=Dictionary(S=Name.GoTo, D="page1"),
        )

        removed = _remove_actions(pdf)
        assert removed >= 1
        assert "/AA" not in pdf.Root

    def test_removes_page_aa_with_compliant_actions(self) -> None:
        """Page /AA is removed even with only compliant actions (ISO 19005-2 6.6.2)."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Page AA with only compliant actions (GoTo, URI)
        pdf.pages[0].AA = Dictionary(
            O=Dictionary(S=Name.GoTo, D="page1"),
            C=Dictionary(S=Name.URI, URI="https://example.com"),
        )

        removed = _remove_actions(pdf)
        assert removed == 1
        assert "/AA" not in pdf.pages[0]

    def test_removes_page_aa_when_empty_after_cleanup(self) -> None:
        """Page /AA is deleted when empty after removing non-compliant actions."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Page AA with only non-compliant actions
        pdf.pages[0].AA = Dictionary(
            O=Dictionary(S=Name.Launch, F="malware.exe"),
            C=Dictionary(S=Name.Sound),
        )

        removed = _remove_actions(pdf)
        assert removed == 1
        assert "/AA" not in pdf.pages[0]

    def test_removes_non_compliant_action_from_acroform_field(self) -> None:
        """Non-compliant /A on AcroForm field is removed."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        field = Dictionary(
            FT=Name.Btn,
            T="submit",
            A=Dictionary(S=Name.Launch, F="malware.exe"),
        )
        pdf.Root.AcroForm = Dictionary(Fields=Array([field]))

        removed = _remove_actions(pdf)
        assert removed == 1

        resolved_field = pdf.Root.AcroForm.Fields[0]
        resolved_field = _resolve_indirect(resolved_field)
        assert "/A" not in resolved_field

    def test_removes_compliant_action_from_acroform_field(self) -> None:
        """Compliant /A (GoTo) on AcroForm field is removed (Rule 6.4.1)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        field = Dictionary(
            FT=Name.Btn,
            T="navigate",
            A=Dictionary(S=Name.GoTo, D="page1"),
        )
        pdf.Root.AcroForm = Dictionary(Fields=Array([field]))

        removed = _remove_actions(pdf)
        assert removed == 1

        resolved_field = pdf.Root.AcroForm.Fields[0]
        resolved_field = _resolve_indirect(resolved_field)
        assert "/A" not in resolved_field

    def test_removes_non_compliant_aa_from_acroform_field(self) -> None:
        """Non-compliant /AA actions on AcroForm field are removed."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        field = Dictionary(
            FT=Name.Tx,
            T="input",
            AA=Dictionary(
                F=Dictionary(S=Name.Launch, F="evil.exe"),
            ),
        )
        pdf.Root.AcroForm = Dictionary(Fields=Array([field]))

        removed = _remove_actions(pdf)
        assert removed == 1

        resolved_field = pdf.Root.AcroForm.Fields[0]
        resolved_field = _resolve_indirect(resolved_field)
        assert "/AA" not in resolved_field

    def test_removes_actions_from_nested_kids(self) -> None:
        """Non-compliant actions in nested /Kids are removed recursively."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        child = Dictionary(
            FT=Name.Btn,
            T="child_btn",
            A=Dictionary(S=Name.Launch, F="payload.exe"),
        )
        parent = Dictionary(
            T="parent",
            Kids=Array([child]),
        )
        pdf.Root.AcroForm = Dictionary(Fields=Array([parent]))

        removed = _remove_actions(pdf)
        assert removed == 1

        resolved_child = pdf.Root.AcroForm.Fields[0].Kids[0]
        resolved_child = _resolve_indirect(resolved_child)
        assert "/A" not in resolved_child


class TestRemoveEmbeddedFiles:
    """Tests for remove_embedded_files."""

    def test_removes_embedded_files_from_names(self) -> None:
        """Removes EmbeddedFiles from Names Dictionary."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add EmbeddedFiles
        file_spec = Dictionary(
            Type=Name.Filespec,
            F="test.txt",
        )
        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        removed = remove_embedded_files(pdf)
        assert removed >= 1
        assert "/EmbeddedFiles" not in pdf.Root.Names

    def test_removes_fileattachment_annotations(self, tmp_path: Path) -> None:
        """Removes FileAttachment Annotations."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Add FileAttachment Annotation (as indirect object)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FileAttachment,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen to ensure structure is correct
        test_path = tmp_path / "test_fileattach.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_embedded_files(pdf)
        assert removed >= 1
        # Annotations Array should be empty or removed
        annots = pdf.pages[0].get("/Annots")
        if annots is not None:
            assert len(annots) == 0

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        removed = remove_embedded_files(pdf)
        assert removed == 0


class TestRemoveXfaForms:
    """Tests for remove_xfa_forms."""

    def test_removes_xfa_stream(self) -> None:
        """Removes XFA when stored as a stream."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        xfa_stream = pdf.make_stream(b"<xfa>test</xfa>")
        acroform = Dictionary(Fields=Array([]), XFA=xfa_stream)
        pdf.Root.AcroForm = acroform

        removed = remove_xfa_forms(pdf)
        assert removed >= 1
        assert "/XFA" not in pdf.Root.AcroForm

    def test_removes_needs_rendering(self) -> None:
        """Removes NeedsRendering flag."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        acroform = Dictionary(Fields=Array([]), NeedsRendering=True)
        pdf.Root.AcroForm = acroform

        removed = remove_xfa_forms(pdf)
        assert removed >= 1
        assert "/NeedsRendering" not in pdf.Root.AcroForm

    def test_removes_both_xfa_and_needs_rendering(self) -> None:
        """Removes both XFA and NeedsRendering when present."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        xfa_stream = pdf.make_stream(b"<xfa>test</xfa>")
        acroform = Dictionary(Fields=Array([]), XFA=xfa_stream, NeedsRendering=True)
        pdf.Root.AcroForm = acroform

        removed = remove_xfa_forms(pdf)
        assert removed == 2

    def test_preserves_acroform_fields(self) -> None:
        """Preserves regular AcroForm fields while removing XFA."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        field = Dictionary(FT=Name.Tx, T="TestField")
        xfa_stream = pdf.make_stream(b"<xfa>test</xfa>")
        acroform = Dictionary(Fields=Array([field]), XFA=xfa_stream)
        pdf.Root.AcroForm = acroform

        remove_xfa_forms(pdf)
        assert "/Fields" in pdf.Root.AcroForm
        assert len(pdf.Root.AcroForm.Fields) == 1

    def test_no_acroform_returns_zero(self) -> None:
        """PDF without AcroForm returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        assert remove_xfa_forms(pdf) == 0

    def test_acroform_without_xfa_returns_zero(self) -> None:
        """AcroForm without XFA returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        acroform = Dictionary(Fields=Array([]))
        pdf.Root.AcroForm = acroform
        assert remove_xfa_forms(pdf) == 0


class TestRemoveForbiddenAnnotations:
    """Tests for remove_forbidden_annotations."""

    @pytest.mark.parametrize(
        "subtype_name",
        ["Sound", "Movie", "Screen", "3D", "RichMedia"],
    )
    def test_removes_forbidden_annotation(
        self, subtype_name: str, tmp_path: Path
    ) -> None:
        """Removes forbidden annotation subtype."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name(f"/{subtype_name}"),
                Rect=Array([0, 0, 100, 100]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / f"test_{subtype_name}.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_annotations(pdf)
        assert removed == 1
        annots = pdf.pages[0].get("/Annots")
        assert annots is None or len(annots) == 0

    def test_preserves_allowed_annotations(self, tmp_path: Path) -> None:
        """Preserves allowed annotation subtypes like Link, Text, Highlight."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        link_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        text_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
            )
        )
        pdf.pages[0].Annots = Array([link_annot, text_annot])

        test_path = tmp_path / "test_allowed.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_annotations(pdf)
        assert removed == 0
        assert len(pdf.pages[0].Annots) == 2

    def test_removes_only_forbidden_keeps_allowed(self, tmp_path: Path) -> None:
        """Removes forbidden but keeps allowed annotations."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        link_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
            )
        )
        movie_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Movie"),
                Rect=Array([0, 0, 200, 200]),
            )
        )
        pdf.pages[0].Annots = Array([link_annot, movie_annot])

        test_path = tmp_path / "test_mixed.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_annotations(pdf)
        assert removed == 1
        assert len(pdf.pages[0].Annots) == 1
        # Remaining should be Link
        remaining = pdf.pages[0].Annots[0]
        remaining = _resolve_indirect(remaining)
        assert str(remaining.get("/Subtype")) == "/Link"

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        removed = remove_forbidden_annotations(pdf)
        assert removed == 0


class TestSanitizeForPdfa:
    """Tests for sanitize_for_pdfa."""

    def test_invalid_level_raises_error(self) -> None:
        """Invalid level raises ConversionError."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            sanitize_for_pdfa(pdf, level="1b")

    def test_too_many_indirect_objects_raises_error(self) -> None:
        """Exceeds 8,388,607 indirect objects raises ConversionError (rule 6.1.13-7)."""
        from unittest.mock import MagicMock, PropertyMock, patch

        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        mock_objects = MagicMock()
        mock_objects.__len__ = MagicMock(return_value=8_388_608)

        with patch.object(
            type(pdf), "objects", new_callable=PropertyMock, return_value=mock_objects
        ):
            with pytest.raises(ConversionError, match="8,388,607"):
                sanitize_for_pdfa(pdf)

    def test_level_2b_removes_js_keeps_nothing(self) -> None:
        """Level 2b removes JS and files, allows transparency."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add JavaScript OpenAction (removed by remove_actions as non-compliant)
        pdf.Root.OpenAction = Dictionary(S=Name.JavaScript, JS="alert('test');")

        # Add embedded file
        file_spec = Dictionary(Type=Name.Filespec, F="test.txt")
        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["actions_removed"] >= 1
        assert result["files_removed"] >= 1

    def test_level_3b_keeps_embedded_files(self) -> None:
        """Level 3b keeps embedded files."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add embedded file
        file_spec = Dictionary(Type=Name.Filespec, F="test.txt")
        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = sanitize_for_pdfa(pdf, level="3b")

        assert result["files_removed"] == 0
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_2u_removes_embedded_files(self) -> None:
        """Level 2u removes embedded files (same as 2b)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add embedded file
        file_spec = Dictionary(Type=Name.Filespec, F="test.txt")
        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = sanitize_for_pdfa(pdf, level="2u")

        assert result["files_removed"] >= 1

    def test_level_3u_keeps_embedded_files(self) -> None:
        """Level 3u keeps embedded files (same as 3b)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Add embedded file
        file_spec = Dictionary(Type=Name.Filespec, F="test.txt")
        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = sanitize_for_pdfa(pdf, level="3u")

        assert result["files_removed"] == 0
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_returns_correct_dict_structure(self) -> None:
        """Returns correct dictionary."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        result = sanitize_for_pdfa(pdf, level="2b")

        assert "javascript_removed" in result
        assert "files_removed" in result
        assert "embedded_files_kept" in result
        assert "actions_removed" in result
        assert "xfa_removed" in result
        assert "forbidden_annotations_removed" in result
        assert "forbidden_xobjects_removed" in result
        assert "appearance_streams_added" in result
        assert isinstance(result["javascript_removed"], int)
        assert isinstance(result["files_removed"], int)
        assert isinstance(result["embedded_files_kept"], int)
        assert isinstance(result["actions_removed"], int)
        assert isinstance(result["xfa_removed"], int)
        assert isinstance(result["forbidden_annotations_removed"], int)
        assert isinstance(result["forbidden_xobjects_removed"], int)

    def test_level_2b_removes_xfa(self) -> None:
        """Level 2b removes XFA forms."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        xfa_stream = pdf.make_stream(b"<xfa>test</xfa>")
        acroform = Dictionary(Fields=Array([]), XFA=xfa_stream)
        pdf.Root.AcroForm = acroform

        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["xfa_removed"] >= 1

    def test_level_3b_removes_xfa(self) -> None:
        """Level 3b also removes XFA forms."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        xfa_stream = pdf.make_stream(b"<xfa>test</xfa>")
        acroform = Dictionary(Fields=Array([]), XFA=xfa_stream)
        pdf.Root.AcroForm = acroform

        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["xfa_removed"] >= 1


class TestRemoveForbiddenXObjects:
    """Tests for remove_forbidden_xobjects."""

    def test_removes_ps_xobject(self, tmp_path: Path) -> None:
        """PostScript XObject is removed."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create a PS XObject (PostScript)
        ps_stream = pdf.make_stream(b"% PostScript code")
        ps_stream.Subtype = Name.PS

        # Add to page resources
        resources = Dictionary(XObject=Dictionary(PS1=ps_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_ps.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 1
        xobjects = pdf.pages[0].Resources.get("/XObject")
        assert xobjects is None or "/PS1" not in xobjects

    def test_removes_ref_xobject(self, tmp_path: Path) -> None:
        """Reference XObject is removed."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create a Ref XObject (Reference)
        ref_dict = Dictionary(Subtype=Name.Ref)
        ref_obj = pdf.make_indirect(ref_dict)

        # Add to page resources
        resources = Dictionary(XObject=Dictionary(Ref1=ref_obj))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_ref.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 1
        xobjects = pdf.pages[0].Resources.get("/XObject")
        assert xobjects is None or "/Ref1" not in xobjects

    def test_removes_alternates_from_xobject(self, tmp_path: Path) -> None:
        """/Alternates array is removed from XObject."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create an Image XObject with /Alternates
        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        # Add an Alternates array
        img_stream.Alternates = Array([Dictionary(Image=img_stream)])

        # Add to page resources
        resources = Dictionary(XObject=Dictionary(Im1=img_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_alternates.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 1  # One /Alternates removed
        xobjects = pdf.pages[0].Resources.XObject
        im1 = xobjects.get("/Im1")
        im1 = _resolve_indirect(im1)
        assert "/Alternates" not in im1

    def test_removes_opi_from_xobject(self, tmp_path: Path) -> None:
        """/OPI dictionary is removed from XObject."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create an Image XObject with /OPI
        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        img_stream.OPI = Dictionary(Version=Name("/1.3"))

        # Add to page resources
        resources = Dictionary(XObject=Dictionary(Im1=img_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_opi.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 1  # One /OPI removed
        xobjects = pdf.pages[0].Resources.XObject
        im1 = xobjects.get("/Im1")
        im1 = _resolve_indirect(im1)
        assert "/OPI" not in im1

    def test_removes_opi_from_form_xobject(self, tmp_path: Path) -> None:
        """/OPI dictionary is removed from nested Form XObject."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create an Image XObject with /OPI nested inside a Form XObject
        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        img_stream.OPI = Dictionary(Version=Name("/1.3"))

        # Create a Form XObject with nested resources
        form_stream = pdf.make_stream(b"q Q")
        form_stream.Subtype = Name.Form
        form_stream.BBox = Array([0, 0, 100, 100])
        form_stream.Resources = Dictionary(XObject=Dictionary(Im1=img_stream))

        # Add Form to page resources
        resources = Dictionary(XObject=Dictionary(Fm1=form_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_opi_form.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed >= 1  # At least the /OPI removed
        # Check the nested image
        form = pdf.pages[0].Resources.XObject.get("/Fm1")
        form = _resolve_indirect(form)
        nested_xobjects = form.Resources.XObject
        im1 = nested_xobjects.get("/Im1")
        im1 = _resolve_indirect(im1)
        assert "/OPI" not in im1

    def test_preserves_valid_xobjects(self, tmp_path: Path) -> None:
        """Valid Image and Form XObjects are preserved."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create an Image XObject (valid)
        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8

        # Create a Form XObject (valid)
        form_stream = pdf.make_stream(b"q Q")
        form_stream.Subtype = Name.Form
        form_stream.BBox = Array([0, 0, 100, 100])

        # Add to page resources
        resources = Dictionary(XObject=Dictionary(Im1=img_stream, Fm1=form_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_valid.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 0
        xobjects = pdf.pages[0].Resources.XObject
        assert "/Im1" in xobjects
        assert "/Fm1" in xobjects

    def test_processes_nested_form_xobjects(self, tmp_path: Path) -> None:
        """Nested XObjects in Form XObjects are processed."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create a PS XObject inside a Form XObject
        ps_stream = pdf.make_stream(b"% PostScript")
        ps_stream.Subtype = Name.PS

        # Create a Form XObject with nested resources
        form_stream = pdf.make_stream(b"q Q")
        form_stream.Subtype = Name.Form
        form_stream.BBox = Array([0, 0, 100, 100])
        form_stream.Resources = Dictionary(XObject=Dictionary(NestedPS=ps_stream))

        # Add Form to page resources
        resources = Dictionary(XObject=Dictionary(Fm1=form_stream))
        pdf.pages[0].Resources = resources

        # Save and reopen
        test_path = tmp_path / "test_nested.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 1  # Nested PS XObject removed
        form = pdf.pages[0].Resources.XObject.get("/Fm1")
        form = _resolve_indirect(form)
        nested_xobjects = form.Resources.get("/XObject")
        assert nested_xobjects is None or "/NestedPS" not in nested_xobjects

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 0

    def test_page_without_resources_returns_zero(self) -> None:
        """Page without Resources returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)
        # Explicitly ensure no Resources
        if "/Resources" in pdf.pages[0]:
            del pdf.pages[0]["/Resources"]
        removed = remove_forbidden_xobjects(pdf)
        assert removed == 0


class TestFixImageInterpolate:
    """Tests for fix_image_interpolate."""

    def test_fixes_interpolate_true(self, tmp_path: Path) -> None:
        """Image with Interpolate=true is set to false."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        img_stream.Interpolate = True

        resources = Dictionary(XObject=Dictionary(Im1=img_stream))
        pdf.pages[0].Resources = resources

        test_path = tmp_path / "test_interp_true.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_image_interpolate(pdf)
        assert fixed == 1

        xobjects = pdf.pages[0].Resources.XObject
        im1 = xobjects.get("/Im1")
        im1 = _resolve_indirect(im1)
        assert bool(im1.get("/Interpolate")) is False

    def test_preserves_interpolate_false(self, tmp_path: Path) -> None:
        """Image with Interpolate=false is not changed, returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        img_stream.Interpolate = False

        resources = Dictionary(XObject=Dictionary(Im1=img_stream))
        pdf.pages[0].Resources = resources

        test_path = tmp_path / "test_interp_false.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_image_interpolate(pdf)
        assert fixed == 0

    def test_no_interpolate_key(self, tmp_path: Path) -> None:
        """Image without /Interpolate key is not changed, returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8

        resources = Dictionary(XObject=Dictionary(Im1=img_stream))
        pdf.pages[0].Resources = resources

        test_path = tmp_path / "test_no_interp.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_image_interpolate(pdf)
        assert fixed == 0

    def test_nested_form_xobject(self, tmp_path: Path) -> None:
        """Image inside Form XObject with Interpolate=true is fixed."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        img_stream = pdf.make_stream(b"\x00" * 100)
        img_stream.Subtype = Name.Image
        img_stream.Width = 10
        img_stream.Height = 10
        img_stream.ColorSpace = Name.DeviceGray
        img_stream.BitsPerComponent = 8
        img_stream.Interpolate = True

        form_stream = pdf.make_stream(b"q Q")
        form_stream.Subtype = Name.Form
        form_stream.BBox = Array([0, 0, 100, 100])
        form_stream.Resources = Dictionary(XObject=Dictionary(Im1=img_stream))

        resources = Dictionary(XObject=Dictionary(Fm1=form_stream))
        pdf.pages[0].Resources = resources

        test_path = tmp_path / "test_nested_interp.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_image_interpolate(pdf)
        assert fixed == 1

        form = pdf.pages[0].Resources.XObject.get("/Fm1")
        form = _resolve_indirect(form)
        nested_im = form.Resources.XObject.get("/Im1")
        nested_im = _resolve_indirect(nested_im)
        assert bool(nested_im.get("/Interpolate")) is False

    def test_multiple_images(self, tmp_path: Path) -> None:
        """Multiple images, some with Interpolate=true, returns correct count."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Image 1: Interpolate=true
        img1 = pdf.make_stream(b"\x00" * 100)
        img1.Subtype = Name.Image
        img1.Width = 10
        img1.Height = 10
        img1.ColorSpace = Name.DeviceGray
        img1.BitsPerComponent = 8
        img1.Interpolate = True

        # Image 2: Interpolate=false (no fix needed)
        img2 = pdf.make_stream(b"\x00" * 100)
        img2.Subtype = Name.Image
        img2.Width = 10
        img2.Height = 10
        img2.ColorSpace = Name.DeviceGray
        img2.BitsPerComponent = 8
        img2.Interpolate = False

        # Image 3: Interpolate=true
        img3 = pdf.make_stream(b"\x00" * 100)
        img3.Subtype = Name.Image
        img3.Width = 10
        img3.Height = 10
        img3.ColorSpace = Name.DeviceGray
        img3.BitsPerComponent = 8
        img3.Interpolate = True

        resources = Dictionary(XObject=Dictionary(Im1=img1, Im2=img2, Im3=img3))
        pdf.pages[0].Resources = resources

        test_path = tmp_path / "test_multi_interp.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_image_interpolate(pdf)
        assert fixed == 2


class TestGetRequiredPdfVersion:
    """Tests for get_required_pdf_version."""

    def test_level_2b_returns_1_7(self) -> None:
        """Level 2b returns version 1.7."""
        assert get_required_pdf_version("2b") == "1.7"

    def test_level_3b_returns_1_7(self) -> None:
        """Level 3b returns version 1.7."""
        assert get_required_pdf_version("3b") == "1.7"

    @pytest.mark.parametrize(
        "level",
        ["2b", "2u", "3b", "3u"],
    )
    def test_all_valid_levels(self, level: str) -> None:
        """All valid PDF/A levels return 1.7."""
        assert get_required_pdf_version(level) == "1.7"

    def test_unknown_level_returns_default(self) -> None:
        """Unknown level returns default version 1.7."""
        assert get_required_pdf_version("unknown") == "1.7"
        assert get_required_pdf_version("1b") == "1.7"
        assert get_required_pdf_version("") == "1.7"


class TestValidatePdfaLevel:
    """Tests for validate_pdfa_level."""

    @pytest.mark.parametrize("level", ["2b", "2u", "3b", "3u"])
    def test_valid_levels_accepted(self, level: str) -> None:
        """All supported target levels are accepted."""
        assert validate_pdfa_level(level) == level

    @pytest.mark.parametrize("level", ["2B", "2U", "3B", "3U"])
    def test_uppercase_normalized_to_lowercase(self, level: str) -> None:
        """Uppercase input is normalized to lowercase."""
        assert validate_pdfa_level(level) == level.lower()

    @pytest.mark.parametrize("level", ["2a", "3a", "4b", "1b", "", "invalid"])
    def test_unsupported_levels_rejected(self, level: str) -> None:
        """Unsupported levels raise ConversionError."""
        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            validate_pdfa_level(level)

    def test_supported_levels_constant(self) -> None:
        """SUPPORTED_LEVELS contains the expected values."""
        assert SUPPORTED_LEVELS == frozenset({"2b", "2u", "3b", "3u"})


class TestFixAnnotationFlags:
    """Tests for fix_annotation_flags."""

    def test_sets_print_flag_on_annotation(self, tmp_path: Path) -> None:
        """Sets Print flag on annotation without it."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Link annotation without Print flag (F=0)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 100]),
                F=0,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_print_flag.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify Print flag is now set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags & ANNOT_FLAG_PRINT == ANNOT_FLAG_PRINT

    def test_fixes_widget_annotations(self, tmp_path: Path) -> None:
        """Widget annotations (form fields) get the Print flag set."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Widget annotation without Print flag
        widget = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 100]),
                F=0,
            )
        )
        pdf.pages[0].Annots = Array([widget])

        # Save and reopen
        test_path = tmp_path / "test_widget.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Widget annotations must also have Print set
        widget = pdf.pages[0].Annots[0]
        widget = _resolve_indirect(widget)
        flags = int(widget.get("/F", 0))
        assert flags == ANNOT_FLAG_PRINT

    def test_preserves_existing_print_flag(self, tmp_path: Path) -> None:
        """Annotations with all required flags already set are not counted."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Text annotation with all required flags (Print + NoZoom + NoRotate)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=4 | 8 | 16,  # Print | NoZoom | NoRotate
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_existing_print.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 0

        # Verify flags remain unchanged
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags == 4 | 8 | 16

    def test_preserves_other_flags(self, tmp_path: Path) -> None:
        """Other flags are preserved when setting Print flag."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with NoZoom flag (F=8) but no Print flag
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=8,  # NoZoom flag (bit 4)
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_preserve_flags.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify NoZoom, NoRotate, and Print flags are set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags == 4 | 8 | 16  # Print + NoZoom + NoRotate

    def test_pdf_without_annotations_returns_zero(self) -> None:
        """PDF without annotations returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        fixed = fix_annotation_flags(pdf)
        assert fixed == 0

    def test_returns_count_of_fixed_annotations(self, tmp_path: Path) -> None:
        """Returns correct count of fixed annotations."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Three annotations without Print flag
        annots = []
        for i in range(3):
            annot = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([i * 50, 0, i * 50 + 40, 40]),
                    F=0,
                )
            )
            annots.append(annot)
        pdf.pages[0].Annots = Array(annots)

        # Save and reopen
        test_path = tmp_path / "test_count.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 3

    def test_mixed_annotations(self, pdf_with_annotations: Path) -> None:
        """Tests with mixed annotation types from fixture."""
        pdf = open_pdf(pdf_with_annotations)

        # Fixture has: Link (F=0), Widget (F=0), Text (F=4, Print set)
        # Link and Widget need Print; Text needs NoZoom+NoRotate
        fixed = fix_annotation_flags(pdf)
        assert fixed == 3

        # Verify Link has Print flag now
        annots = pdf.pages[0].Annots
        link_annot = annots[0]
        link_annot = _resolve_indirect(link_annot)
        assert int(link_annot.get("/F", 0)) == ANNOT_FLAG_PRINT

        # Widget also gets Print
        widget_annot = annots[1]
        widget_annot = _resolve_indirect(widget_annot)
        assert int(widget_annot.get("/F", 0)) == ANNOT_FLAG_PRINT

        # Text should have Print + NoZoom + NoRotate
        text_annot = annots[2]
        text_annot = _resolve_indirect(text_annot)
        flags = int(text_annot.get("/F", 0))
        assert flags & ANNOT_FLAG_PRINT
        assert flags & ANNOT_FLAG_NOZOOM
        assert flags & ANNOT_FLAG_NOROTATE

    def test_removes_hidden_flag(self, tmp_path: Path) -> None:
        """Removes Hidden flag from annotation."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with Hidden flag (F=2)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_HIDDEN,  # Hidden flag (bit 2, value 2)
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_hidden_flag.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify Hidden flag is removed and Print flag is set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags & ANNOT_FLAG_HIDDEN == 0  # Hidden flag removed
        assert flags & ANNOT_FLAG_PRINT == ANNOT_FLAG_PRINT  # Print flag set

    def test_removes_hidden_flag_preserves_other_flags(self, tmp_path: Path) -> None:
        """Removes Hidden flag while preserving other flags."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with Hidden (2) + NoView (32) flags = 34
        initial_flags = ANNOT_FLAG_HIDDEN | ANNOT_FLAG_NOVIEW
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=initial_flags,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_hidden_preserve_other.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify Hidden and NoView flags are both removed, Print is set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags & ANNOT_FLAG_HIDDEN == 0  # Hidden flag removed
        assert flags & ANNOT_FLAG_NOVIEW == 0  # NoView flag removed
        assert flags & ANNOT_FLAG_PRINT == ANNOT_FLAG_PRINT  # Print flag set

    def test_removes_noview_flag(self, tmp_path: Path) -> None:
        """Removes NoView flag from annotation."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with NoView (32) flag
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_NOVIEW,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_noview.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)

        assert fixed == 1
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags & ANNOT_FLAG_NOVIEW == 0  # NoView removed
        assert flags & ANNOT_FLAG_PRINT == ANNOT_FLAG_PRINT  # Print added

    def test_removes_invisible_flag(self, tmp_path: Path) -> None:
        """Removes Invisible flag from annotation."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with Invisible flag (F=1)
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=ANNOT_FLAG_INVISIBLE,  # Invisible flag (bit 1, value 1)
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_invisible_flag.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify Invisible flag is removed and Print flag is set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        # Text annotation: Print + NoZoom + NoRotate
        assert flags == ANNOT_FLAG_PRINT | ANNOT_FLAG_NOZOOM | ANNOT_FLAG_NOROTATE

    def test_removes_invisible_flag_preserves_other_flags(self, tmp_path: Path) -> None:
        """Removes Invisible flag while preserving other flags."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with Invisible (1) + NoView (32) flags = 33
        initial_flags = ANNOT_FLAG_INVISIBLE | ANNOT_FLAG_NOVIEW
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 100]),
                F=initial_flags,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen
        test_path = tmp_path / "test_invisible_preserve_other.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = fix_annotation_flags(pdf)
        assert fixed == 1

        # Verify Invisible and NoView both removed, Print set
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        flags = int(annot.get("/F", 0))
        assert flags & ANNOT_FLAG_INVISIBLE == 0  # Invisible removed
        assert flags & ANNOT_FLAG_NOVIEW == 0  # NoView removed
        assert flags & ANNOT_FLAG_PRINT == ANNOT_FLAG_PRINT  # Print set


class TestEnsureAppearanceStreams:
    """Tests for ensure_appearance_streams."""

    def test_adds_ap_to_annotation_without_ap(self, tmp_path: Path) -> None:
        """Annotation without /AP gets a minimal Form XObject appearance stream."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 50]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_no_ap.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 1

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        ap = annot.get("/AP")
        assert ap is not None
        n = ap.get("/N")
        assert n is not None
        n = _resolve_indirect(n)
        assert str(n.get("/Type")) == "/XObject"
        assert str(n.get("/Subtype")) == "/Form"
        assert n.get("/BBox") is not None

    def test_adds_n_to_ap_without_n(self, tmp_path: Path) -> None:
        """/AP exists with /R only; /N gets added, /R preserved."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create a rollover appearance stream
        r_stream = pdf.make_stream(b"")
        r_stream[Name.Type] = Name.XObject
        r_stream[Name.Subtype] = Name.Form
        r_stream[Name.BBox] = Array([0, 0, 10, 10])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 50]),
                AP=Dictionary(R=r_stream),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_ap_no_n.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 1

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        ap = annot["/AP"]
        ap = _resolve_indirect(ap)
        assert ap.get("/N") is not None
        assert ap.get("/R") is not None

    def test_skips_popup_annotations(self, tmp_path: Path) -> None:
        """Popup annotation without /AP stays unchanged, returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Popup,
                Rect=Array([0, 0, 100, 50]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_popup.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 0

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        assert annot.get("/AP") is None

    def test_skips_link_annotations(self, tmp_path: Path) -> None:
        """Link annotation without /AP stays unchanged, returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Link,
                Rect=Array([0, 0, 100, 50]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_link.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 0

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        assert annot.get("/AP") is None

    def test_skips_truly_zero_size_annotation(self, tmp_path: Path) -> None:
        """Annotation where x1==x2 AND y1==y2 is exempt (zero-size per spec)."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 200, 100, 200]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_zero_size.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 0

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        assert annot.get("/AP") is None

    def test_adds_ap_to_zero_width_annotation(self, tmp_path: Path) -> None:
        """Annotation with zero width but non-zero height is NOT exempt and gets /AP.

        Rect=[50, 600, 50, 50]: x1==x2 but y1!=y2, so only one pair matches.
        Per ISO 19005-2 rule 6.3.3, BOTH pairs must be equal to be exempt.
        """
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FileAttachment,
                Rect=Array([50, 600, 50, 50]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_zero_width.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 1

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        assert annot.get("/AP") is not None

    def test_skips_annotation_with_existing_ap_n(self, tmp_path: Path) -> None:
        """Annotation that already has /AP /N returns 0."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        n_stream = pdf.make_stream(b"")
        n_stream[Name.Type] = Name.XObject
        n_stream[Name.Subtype] = Name.Form
        n_stream[Name.BBox] = Array([0, 0, 100, 50])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 100, 50]),
                AP=Dictionary(N=n_stream),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_existing_ap_n.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 0

    def test_handles_multiple_annotations(self, tmp_path: Path) -> None:
        """Mix of with/without /AP, correct count returned."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation with /AP /N already
        n_stream = pdf.make_stream(b"")
        n_stream[Name.Type] = Name.XObject
        n_stream[Name.Subtype] = Name.Form
        n_stream[Name.BBox] = Array([0, 0, 50, 50])

        annot_with_ap = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 50, 50]),
                AP=Dictionary(N=n_stream),
            )
        )

        # Two annotations without /AP
        annot_no_ap1 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 0, 150, 50]),
            )
        )
        annot_no_ap2 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Highlight,
                Rect=Array([200, 0, 250, 50]),
            )
        )

        pdf.pages[0].Annots = Array([annot_with_ap, annot_no_ap1, annot_no_ap2])

        test_path = tmp_path / "test_multiple.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 2

    def test_handles_multiple_pages(self, tmp_path: Path) -> None:
        """Annotations on 2 pages are both processed."""
        pdf = new_pdf()

        for _ in range(2):
            page = pikepdf.Page(Dictionary(Type=Name.Page))
            pdf.pages.append(page)

        annot1 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Highlight,
                Rect=Array([0, 0, 100, 50]),
            )
        )
        pdf.pages[0].Annots = Array([annot1])

        annot2 = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([0, 0, 80, 40]),
            )
        )
        pdf.pages[1].Annots = Array([annot2])

        test_path = tmp_path / "test_multi_page.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 2

    def test_empty_pdf_returns_zero(self) -> None:
        """PDF without annotations returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        added = ensure_appearance_streams(pdf)
        assert added == 0

    def test_bbox_derived_from_rect(self, tmp_path: Path) -> None:
        """Rect=[100,200,300,250] produces BBox=[0,0,200,50]."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 200, 300, 250]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_bbox.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        ensure_appearance_streams(pdf)

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        n = annot["/AP"]["/N"]
        n = _resolve_indirect(n)
        bbox = n["/BBox"]
        assert float(bbox[0]) == 0
        assert float(bbox[1]) == 0
        assert float(bbox[2]) == pytest.approx(200.0)
        assert float(bbox[3]) == pytest.approx(50.0)

    def test_annotation_without_rect_gets_zero_bbox(self, tmp_path: Path) -> None:
        """Missing Rect produces BBox=[0,0,0,0]."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Annotation without Rect
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_no_rect.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 1

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        n = annot["/AP"]["/N"]
        n = _resolve_indirect(n)
        bbox = n["/BBox"]
        assert float(bbox[0]) == 0
        assert float(bbox[1]) == 0
        assert float(bbox[2]) == 0
        assert float(bbox[3]) == 0

    def test_widget_annotation_gets_ap(self, tmp_path: Path) -> None:
        """Widget annotation is NOT exempt (only Popup is)."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 30]),
            )
        )
        pdf.pages[0].Annots = Array([annot])

        test_path = tmp_path / "test_widget_ap.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        added = ensure_appearance_streams(pdf)
        assert added == 1

        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        assert annot.get("/AP") is not None

    def test_integration_sanitize_result_key(self) -> None:
        """sanitize_for_pdfa result dict contains 'appearance_streams_added'."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        result = sanitize_for_pdfa(pdf, level="3b")
        assert "appearance_streams_added" in result
        assert isinstance(result["appearance_streams_added"], int)


class TestEnsureAfRelationships:
    """Tests for ensure_af_relationships."""

    def _make_pdf_with_embedded_file(
        self, *, af_relationship: Name | None = None
    ) -> Pdf:
        """Helper: create a PDF with one embedded file in Names."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        file_spec = Dictionary(Type=Name.Filespec, F="test.txt")
        if af_relationship is not None:
            file_spec["/AFRelationship"] = af_relationship

        embedded = Dictionary(Names=Array(["test.txt", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)
        return pdf

    def test_adds_missing_af_relationship(self) -> None:
        """Missing AFRelationship is set to /Unspecified."""
        pdf = self._make_pdf_with_embedded_file()
        fixed = ensure_af_relationships(pdf)
        assert fixed == 1

        fs = pdf.Root.Names.EmbeddedFiles.Names[1]
        assert str(fs.get("/AFRelationship")) == "/Unspecified"

    def test_preserves_valid_af_relationship(self) -> None:
        """Existing valid AFRelationship (/Source) is not changed."""
        pdf = self._make_pdf_with_embedded_file(af_relationship=Name.Source)
        fixed = ensure_af_relationships(pdf)
        assert fixed == 0

        fs = pdf.Root.Names.EmbeddedFiles.Names[1]
        assert str(fs.get("/AFRelationship")) == "/Source"

    def test_replaces_invalid_af_relationship(self) -> None:
        """Invalid AFRelationship is replaced with /Unspecified."""
        pdf = self._make_pdf_with_embedded_file(af_relationship=Name("/InvalidValue"))
        fixed = ensure_af_relationships(pdf)
        assert fixed == 1

        fs = pdf.Root.Names.EmbeddedFiles.Names[1]
        assert str(fs.get("/AFRelationship")) == "/Unspecified"

    def test_builds_root_af_array(self) -> None:
        """Creates /Root/AF array with correct number of entries."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        fs1 = Dictionary(Type=Name.Filespec, F="a.txt")
        fs2 = Dictionary(Type=Name.Filespec, F="b.txt")
        embedded = Dictionary(Names=Array(["a.txt", fs1, "b.txt", fs2]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        ensure_af_relationships(pdf)

        af = pdf.Root.get("/AF")
        assert af is not None
        assert len(af) == 2

    def test_empty_pdf_returns_zero(self) -> None:
        """Empty PDF returns 0 and no /AF is created."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        fixed = ensure_af_relationships(pdf)
        assert fixed == 0
        assert pdf.Root.get("/AF") is None

    def test_processes_file_attachment_annotation(self, tmp_path: Path) -> None:
        """FileAttachment annotation /FS FileSpec is processed."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        file_spec = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="annot.txt"))
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FileAttachment,
                Rect=Array([0, 0, 100, 100]),
                FS=file_spec,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen to ensure indirect references
        test_path = tmp_path / "test_fa_annot.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = ensure_af_relationships(pdf)
        assert fixed == 1

        af = pdf.Root.get("/AF")
        assert af is not None
        assert len(af) == 1

    def test_sanitize_calls_for_3b_3u(self) -> None:
        """sanitize_for_pdfa calls ensure_af_relationships for 3b/3u."""
        pdf = self._make_pdf_with_embedded_file()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["af_relationships_fixed"] == 1
        assert pdf.Root.get("/AF") is not None

        pdf2 = self._make_pdf_with_embedded_file()
        result2 = sanitize_for_pdfa(pdf2, level="3u")
        assert result2["af_relationships_fixed"] == 1

    def test_sanitize_does_not_call_for_2b_2u(self) -> None:
        """sanitize_for_pdfa does NOT call ensure_af_relationships for 2b/2u."""
        pdf = self._make_pdf_with_embedded_file()
        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["af_relationships_fixed"] == 0

        pdf2 = self._make_pdf_with_embedded_file()
        result2 = sanitize_for_pdfa(pdf2, level="2u")
        assert result2["af_relationships_fixed"] == 0

    def test_no_duplicates_in_af_array(self, tmp_path: Path) -> None:
        """Same FileSpec in EmbeddedFiles + annotation is not duplicated in /AF."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        # Create an indirect FileSpec
        file_spec = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="shared.txt"))

        # Add to EmbeddedFiles
        embedded = Dictionary(Names=Array(["shared.txt", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        # Also reference from a FileAttachment annotation
        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FileAttachment,
                Rect=Array([0, 0, 100, 100]),
                FS=file_spec,
            )
        )
        pdf.pages[0].Annots = Array([annot])

        # Save and reopen to get proper indirect refs
        test_path = tmp_path / "test_no_dup.pdf"
        pdf.save(test_path)

        pdf = open_pdf(test_path)
        fixed = ensure_af_relationships(pdf)
        assert fixed == 1  # Only one FileSpec to fix

        af = pdf.Root.get("/AF")
        assert af is not None
        assert len(af) == 1  # No duplicates
