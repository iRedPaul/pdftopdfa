# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for cli.py."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from colorama import Fore, Style
from pikepdf import Pdf

import pdftopdfa.cli as cli_module
from pdftopdfa import __version__
from pdftopdfa.cli import (
    EXIT_CONVERSION_FAILED,
    EXIT_FILE_NOT_FOUND,
    EXIT_GENERAL_ERROR,
    EXIT_PERMISSION_ERROR,
    EXIT_REVIEW_REQUIRED,
    EXIT_SUCCESS,
    EXIT_VALIDATION_FAILED,
    main,
)
from pdftopdfa.converter import ConversionResult, PDFUAStatus, PublicationPolicy
from pdftopdfa.exceptions import ConversionError, OCRError, VeraPDFError
from pdftopdfa.verapdf import is_verapdf_available

OCR_MODEL_ARGS = [
    "--ocr-detection-model-dir",
    "detection-model",
    "--ocr-recognition-model-dir",
    "recognition-model",
]


@pytest.mark.parametrize("processing_only", [False, True])
def test_single_file_preserves_target_created_during_conversion(
    sample_pdf: Path, tmp_path: Path, processing_only: bool
) -> None:
    output = tmp_path / "concurrent.pdf"
    sentinel = b"concurrent writer"
    convert = cli_module.convert_to_pdfa

    def create_target(**kwargs):
        output.write_bytes(sentinel)
        return convert(**kwargs)

    args = [str(sample_pdf), str(output)]
    if processing_only:
        args.append("--no-pdfa")
    with patch.object(cli_module, "convert_to_pdfa", side_effect=create_target):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == EXIT_CONVERSION_FAILED
    assert "already exists" in result.output
    assert output.read_bytes() == sentinel


@pytest.fixture
def runner() -> CliRunner:
    """CLI Test Runner."""
    return CliRunner()


def _run_cli_with_cp1252(*args: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the real CLI with Windows' legacy console encoding."""
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root / "src"), env.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-m", "pdftopdfa", *(str(arg) for arg in args)],
        cwd=project_root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=False,
    )


class TestCliHelp:
    """Tests for --help option."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """--help returns exit code 0 and shows options."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--level" in result.output
        assert "[2a|2b|2u|3a|3b|3u]" in result.output
        assert "--validate" in result.output
        assert "--pdfua" in result.output
        assert "--recursive" in result.output
        assert "--force" in result.output
        assert "--quiet" in result.output
        assert "--verbose" in result.output

    def test_cli_help_shows_ocr_option(self, runner: CliRunner) -> None:
        """--ocr option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--ocr" in result.output
        assert "--ocr-execution-provider [cpu|directml|directml:INDEX]" in result.output
        assert "--ocr-layout" in result.output
        assert "--ocr-figure-text" in result.output
        assert "--deskew" in result.output
        assert "--rotate-pages" in result.output

    def test_cli_help_shows_ocr_force_option(self, runner: CliRunner) -> None:
        """--ocr-force option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--ocr-force" in result.output

    def test_cli_help_shows_skip_any_pdfa_option(self, runner: CliRunner) -> None:
        """--skip-any-pdfa option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--skip-any-pdfa" in result.output

    def test_cli_help_shows_no_pdfa_option(self, runner: CliRunner) -> None:
        """--no-pdfa option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--no-pdfa" in result.output

    def test_cli_help_shows_allow_signature_invalidation_option(
        self, runner: CliRunner
    ) -> None:
        """--allow-signature-invalidation option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--allow-signature-invalidation" in result.output


class TestCliVersion:
    """Tests for --version option."""

    def test_cli_version(self, runner: CliRunner) -> None:
        """--version returns exit code 0 and shows version."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output


class TestCliConsoleOutput:
    """Tests for console-safe output formatting."""

    def test_print_success_uses_plain_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Success output uses a plain text prefix."""

        class FakeStream:
            encoding = "cp1252"

        monkeypatch.setattr(cli_module.sys, "stdout", FakeStream())

        with patch("pdftopdfa.cli.click.echo") as mock_echo:
            cli_module.print_success("Converted")

        mock_echo.assert_called_once_with(
            f"{Fore.GREEN}Success:{Style.RESET_ALL} Converted"
        )

    def test_print_error_sanitizes_unencodable_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error output replaces unsupported characters instead of crashing."""

        class FakeStream:
            encoding = "ascii"

        monkeypatch.setattr(cli_module.sys, "stderr", FakeStream())

        with patch("pdftopdfa.cli.click.echo") as mock_echo:
            cli_module.print_error("Fehler mit Umlaut: ä")

        mock_echo.assert_called_once_with(
            f"{Fore.RED}Error:{Style.RESET_ALL} Fehler mit Umlaut: ?",
            err=True,
        )

    def test_single_file_cli_handles_unicode_name_on_cp1252_console(
        self,
        sample_pdf_bytes: bytes,
        tmp_dir: Path,
    ) -> None:
        """The real single-file CLI replaces an unencodable input name."""
        input_path = tmp_dir / "卍.pdf"
        output_path = tmp_dir / "single-output.pdf"
        input_path.write_bytes(sample_pdf_bytes)

        result = _run_cli_with_cp1252(input_path, output_path)

        assert result.returncode == EXIT_SUCCESS, result.stderr.decode(
            "cp1252", errors="replace"
        )
        assert b"UnicodeEncodeError" not in result.stderr
        assert output_path.exists()

    def test_directory_cli_handles_unicode_name_on_cp1252_console(
        self,
        encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The real directory summary replaces an unencodable input name."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        input_path = input_dir / "卍.pdf"
        input_path.write_bytes(encrypted_pdf.read_bytes())

        result = _run_cli_with_cp1252(input_dir)

        assert result.returncode == EXIT_SUCCESS, result.stderr.decode(
            "cp1252", errors="replace"
        )
        assert b"UnicodeEncodeError" not in result.stderr
        assert (input_dir / "卍_pdfa.pdf").exists()


class TestCliConvert:
    """Tests for file conversion."""

    @patch("pdftopdfa.cli._convert_single_file")
    def test_cli_convert_passes_no_pdfa(
        self,
        mock_convert_single,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Single-file CLI forwards --no-pdfa."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_single.return_value = EXIT_SUCCESS

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--no-pdfa"],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_single.call_args.kwargs["pdfa"] is False

    def test_cli_no_pdfa_uses_processed_name_and_copies_unchanged(
        self, runner: CliRunner, sample_pdf: Path
    ) -> None:
        """Processing-only mode uses its suffix and supports a no-op copy."""
        output_path = sample_pdf.with_name(f"{sample_pdf.stem}_processed.pdf")

        result = runner.invoke(main, [str(sample_pdf), "--no-pdfa"])

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.read_bytes() == sample_pdf.read_bytes()
        assert "Copied unchanged" in result.output
        assert "PDF/A-" not in result.output

    def test_cli_no_pdfa_rejects_validation(
        self, runner: CliRunner, sample_pdf: Path
    ) -> None:
        """Validation is unavailable when PDF/A conversion is disabled."""
        result = runner.invoke(main, [str(sample_pdf), "--no-pdfa", "--validate"])

        assert result.exit_code != EXIT_SUCCESS
        assert "--validate cannot be combined with --no-pdfa" in result.output

    @pytest.mark.parametrize("level", ["2b", "2u", "3b", "3u"])
    def test_cli_pdfua_requires_level_a(
        self, runner: CliRunner, sample_pdf: Path, level: str
    ) -> None:
        """--pdfua is limited to the compatible PDF/A Level A targets."""
        result = runner.invoke(main, [str(sample_pdf), "--pdfua", "--level", level])

        assert result.exit_code != EXIT_SUCCESS
        assert "--pdfua requires --level 2a or 3a" in result.output

    @patch("pdftopdfa.cli._convert_single_file", return_value=EXIT_SUCCESS)
    def test_cli_pdfua_is_forwarded(
        self,
        mock_convert_single: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The CLI forwards the PDF/UA opt-in flag."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--level", "2a", "--pdfua"],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_single.call_args.kwargs["pdfua"] is True

    @patch("pdftopdfa.cli._convert_single_file", return_value=EXIT_SUCCESS)
    def test_cli_forwards_pdfua_enterprise_options(
        self,
        mock_convert_single: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Metadata authority and unsafe publication are explicit CLI inputs."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--level",
                "2a",
                "--pdfua",
                "--publish-noncompliant",
                "--document-title",
                "Annual report",
                "--document-language",
                "en-GB",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        kwargs = mock_convert_single.call_args.kwargs
        assert kwargs["publication_policy"] is PublicationPolicy.ALWAYS
        assert kwargs["document_title"] == "Annual report"
        assert kwargs["document_language"] == "en-GB"

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_writes_machine_readable_audit_report(
        self,
        mock_convert_to_pdfa: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The CLI persists a stable JSON envelope for enterprise ingestion."""
        output_path = tmp_dir / "output.pdf"
        report_path = tmp_dir / "audit.json"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="2a",
            pdfua_status=PDFUAStatus.MACHINE_VALIDATED,
            candidate_sha256="abc123",
            candidate_size=42,
        )

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--audit-report", str(report_path)],
        )

        assert result.exit_code == EXIT_SUCCESS
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == 1
        assert report["results"][0]["pdfua_status"] == "machine_validated"
        assert report["results"][0]["candidate_sha256"] == "abc123"

    @pytest.mark.parametrize(
        ("error", "expected_exit"),
        [
            (ConversionError("conversion stopped"), EXIT_CONVERSION_FAILED),
            (PermissionError("destination locked"), EXIT_PERMISSION_ERROR),
            (RuntimeError("unexpected failure"), EXIT_GENERAL_ERROR),
        ],
    )
    def test_cli_replaces_stale_audit_report_after_fatal_error(
        self,
        error: Exception,
        expected_exit: int,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Fatal invocations atomically replace evidence from an earlier run."""
        output_path = tmp_dir / "output.pdf"
        report_path = tmp_dir / "audit.json"
        report_path.write_text('{"stale": true}', encoding="utf-8")

        with patch("pdftopdfa.cli.convert_to_pdfa", side_effect=error):
            result = runner.invoke(
                main,
                [
                    str(sample_pdf),
                    str(output_path),
                    "--audit-report",
                    str(report_path),
                ],
            )

        assert result.exit_code == expected_exit
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == 1
        assert report["results"] == []
        assert report["fatal_error"] == {
            "error_type": type(error).__name__,
            "exit_code": expected_exit,
            "input_path": str(sample_pdf),
            "message": str(error),
            "output_path": str(output_path),
        }

    def test_cli_replaces_stale_audit_report_after_usage_error(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A rejected option combination cannot leave stale audit evidence."""
        output_path = tmp_dir / "output.pdf"
        report_path = tmp_dir / "audit.json"
        report_path.write_text('{"stale": true}', encoding="utf-8")

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--audit-report",
                str(report_path),
                "--publish-noncompliant",
            ],
        )

        assert result.exit_code == 2
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report == {
            "fatal_error": {
                "error_type": "UsageError",
                "exit_code": 2,
                "input_path": str(sample_pdf),
                "message": "--publish-noncompliant requires --validate or --pdfua",
                "output_path": str(output_path),
            },
            "results": [],
            "schema_version": 1,
        }

    def test_cli_rejects_non_json_audit_report(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Audit output cannot accidentally overwrite a PDF path."""
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                "--audit-report",
                str(tmp_dir / "report.pdf"),
            ],
        )

        assert result.exit_code != EXIT_SUCCESS
        assert "must use a .json filename" in result.output

    def test_cli_rejects_document_title_for_directory(
        self, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """One title cannot be silently applied to every file in a batch."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        result = runner.invoke(main, [str(input_dir), "--document-title", "Title"])

        assert result.exit_code != EXIT_SUCCESS
        assert "only valid for a single PDF" in result.output

    @patch("pdftopdfa.cli._convert_single_file")
    def test_cli_convert_passes_skip_any_pdfa(
        self,
        mock_convert_single,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Single-file CLI forwards --skip-any-pdfa."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_single.return_value = EXIT_SUCCESS

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--skip-any-pdfa"],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_single.call_args.kwargs["skip_any_pdfa"] is True

    @patch("pdftopdfa.cli._convert_single_file", return_value=EXIT_SUCCESS)
    def test_cli_convert_passes_figure_text_ocr(
        self,
        mock_convert_single: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(tmp_dir / "output.pdf"),
                "--level",
                "3a",
                "--ocr-figure-text",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_single.call_args.kwargs["ocr_figure_text"] is True

    @patch("pdftopdfa.cli._convert_single_file")
    def test_cli_convert_passes_allow_signature_invalidation(
        self,
        mock_convert_single,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Single-file CLI forwards --allow-signature-invalidation."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_single.return_value = EXIT_SUCCESS

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--allow-signature-invalidation"],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert (
            mock_convert_single.call_args.kwargs["allow_signature_invalidation"] is True
        )

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_single_file_compliant_skip_is_still_validated(
        self,
        mock_convert_to_pdfa,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Explicit validation also covers an unchanged skipped output."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="2b",
            skipped=True,
        )
        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "3a",
            do_validate=True,
            force=False,
            quiet=True,
        )

        assert result == EXIT_SUCCESS
        assert mock_convert_to_pdfa.call_args.kwargs["validate"] is True

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_single_file_protected_skip_is_not_validated(
        self,
        mock_convert_to_pdfa: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The CLI does not validate when no PDF/A output was created."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level=None,
            skipped=True,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "3a",
            do_validate=True,
            force=False,
            quiet=True,
        )

        assert result == EXIT_SUCCESS
        assert mock_convert_to_pdfa.call_args.kwargs["validate"] is True

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_pdfua_validate_does_not_repeat_automatic_validation(
        self,
        mock_convert_to_pdfa: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The converter owns the mandatory PDF/A and UA-1 validation."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="2a",
        )
        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "2a",
            do_validate=True,
            force=False,
            quiet=True,
            pdfua=True,
        )

        assert result == EXIT_SUCCESS
        assert mock_convert_to_pdfa.call_args.kwargs["validate"] is True

    @patch("pdftopdfa.cli.convert_to_pdfa")
    @pytest.mark.parametrize("quiet", [False, True])
    def test_cli_single_file_known_validation_failure_returns_exit_code(
        self,
        mock_convert_to_pdfa,
        quiet: bool,
        sample_pdf: Path,
        tmp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """PDF/UA failures report both automatic validation profiles."""
        output_path = tmp_dir / "output.pdf"
        validation_errors = [
            "Validation: PDF/A rule failed",
            "PDF/UA validation: PDF/UA rule failed",
        ]
        review_warning = "Generated alternatives require review"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=False,
            input_path=sample_pdf,
            output_path=output_path,
            level="2b",
            warnings=[*validation_errors, review_warning],
            error="Validation failed; output candidate was published",
            validation_failed=True,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "2b",
            do_validate=True,
            force=False,
            quiet=quiet,
            pdfua=True,
        )

        assert result == EXIT_VALIDATION_FAILED
        captured = capsys.readouterr()
        assert all(error in captured.err for error in validation_errors)
        assert (review_warning in captured.out) is not quiet

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_returns_distinct_author_review_status(
        self,
        mock_convert_to_pdfa: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Machine-valid output requiring author review is not a silent success."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="2a",
            pdfua_status=PDFUAStatus.REVIEW_REQUIRED,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "2a",
            do_validate=False,
            force=False,
            quiet=True,
            pdfua=True,
        )

        assert result == EXIT_REVIEW_REQUIRED

    def test_cli_convert_simple(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Successful conversion with exit code 0."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path)])

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()

    def test_cli_convert_with_level(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--level 3b shows 'PDF/A-3b' in output."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main, [str(sample_pdf), str(output_path), "--level", "3b"]
        )

        assert result.exit_code == EXIT_SUCCESS
        assert "PDF/A-3b" in result.output or "3b" in result.output

    @pytest.mark.parametrize("level", ["2a", "3a"])
    @patch("pdftopdfa.cli._convert_single_file", return_value=EXIT_SUCCESS)
    def test_cli_accepts_level_a(
        self,
        mock_convert_single,
        level: str,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The CLI accepts both accessible conformance levels."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main, [str(sample_pdf), str(output_path), "--level", level]
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_single.call_args.args[2] == level

    def test_cli_convert_default_output(
        self, runner: CliRunner, sample_pdf: Path
    ) -> None:
        """Without OUTPUT, *_pdfa.pdf is created."""
        result = runner.invoke(main, [str(sample_pdf)])

        assert result.exit_code == EXIT_SUCCESS

        expected_output = sample_pdf.parent / f"{sample_pdf.stem}_pdfa.pdf"
        assert expected_output.exists()

        # Cleanup
        expected_output.unlink()

    def test_cli_convert_quiet(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--quiet reduces output."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--quiet"])

        assert result.exit_code == EXIT_SUCCESS
        # With --quiet there should be less output
        # Success messages are suppressed
        assert "Converting" not in result.output

    def test_cli_convert_encrypted_pdf_is_converted(
        self, runner: CliRunner, encrypted_pdf: Path, tmp_dir: Path
    ) -> None:
        """An encrypted PDF with an empty user password is converted."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(encrypted_pdf), str(output_path)])

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        assert output_path.read_bytes() != encrypted_pdf.read_bytes()
        with Pdf.open(output_path) as pdf:
            assert pdf.is_encrypted is False
        assert "Converted to PDF/A" in result.output

    def test_cli_password_encrypted_pdf_is_skipped(
        self,
        runner: CliRunner,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The CLI safely copies PDFs that require a user password."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(password_encrypted_pdf), str(output_path), "--level", "2a"],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()
        assert "encrypted" in result.output


class TestCliMissingInput:
    """Tests for missing input file."""

    def test_cli_missing_input(self, runner: CliRunner, tmp_dir: Path) -> None:
        """Missing input file returns exit code 2."""
        nonexistent = tmp_dir / "nonexistent.pdf"

        result = runner.invoke(main, [str(nonexistent)])

        # Click returns exit code 2 for missing file (exists=True)
        assert result.exit_code == EXIT_FILE_NOT_FOUND


class TestCliPermissionErrors:
    """Tests for output permission failures."""

    @patch("pdftopdfa.cli.convert_to_pdfa", side_effect=PermissionError("read-only"))
    def test_cli_permission_error_returns_exit_code(
        self,
        _mock_convert: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A converter permission error uses the documented exit code."""
        result = runner.invoke(main, [str(sample_pdf), str(tmp_dir / "output.pdf")])

        assert result.exit_code == EXIT_PERMISSION_ERROR
        assert "Access denied: read-only" in result.output


class TestCliForceOverwrite:
    """Tests for --force option."""

    def test_cli_refuses_overwrite_without_force(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Without --force, overwriting is refused."""
        output_path = tmp_dir / "output.pdf"
        output_path.write_text("existing content")

        result = runner.invoke(main, [str(sample_pdf), str(output_path)])

        assert result.exit_code == EXIT_GENERAL_ERROR
        assert "already exists" in result.output

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_refused_overwrite_replaces_stale_audit_report(
        self,
        mock_convert_to_pdfa: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A no-overwrite refusal replaces evidence from an earlier run."""
        output_path = tmp_dir / "output.pdf"
        output_path.write_text("existing content")
        report_path = tmp_dir / "audit.json"
        report_path.write_text('{"stale": true}', encoding="utf-8")

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--audit-report",
                str(report_path),
            ],
        )

        assert result.exit_code == EXIT_GENERAL_ERROR
        mock_convert_to_pdfa.assert_not_called()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report == {
            "schema_version": 1,
            "results": [],
            "fatal_error": {
                "error_type": "FileExistsError",
                "exit_code": EXIT_GENERAL_ERROR,
                "input_path": str(sample_pdf),
                "message": (
                    f"Output file already exists: {output_path}. "
                    "Use --force to overwrite."
                ),
                "output_path": str(output_path),
            },
        }

    def test_cli_allows_overwrite_with_force(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """With --force, overwriting is allowed."""
        output_path = tmp_dir / "output.pdf"
        output_path.write_text("existing content")

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--force"])

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        # File should be larger than the original text
        assert output_path.stat().st_size > len("existing content")


class TestCliDirectory:
    """Tests for directory conversion."""

    @patch("pdftopdfa.cli._convert_directory")
    def test_cli_convert_directory_passes_no_pdfa(
        self, mock_convert_directory, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory CLI forwards --no-pdfa."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        mock_convert_directory.return_value = EXIT_SUCCESS

        result = runner.invoke(main, [str(input_dir), "--no-pdfa"])

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_directory.call_args.kwargs["pdfa"] is False

    @patch("pdftopdfa.cli._convert_directory")
    def test_cli_convert_directory_passes_skip_any_pdfa(
        self, mock_convert_directory, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory CLI forwards --skip-any-pdfa."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        mock_convert_directory.return_value = EXIT_SUCCESS

        result = runner.invoke(main, [str(input_dir), "--skip-any-pdfa"])

        assert result.exit_code == EXIT_SUCCESS
        assert mock_convert_directory.call_args.kwargs["skip_any_pdfa"] is True

    @patch("pdftopdfa.cli._convert_directory")
    def test_cli_convert_directory_passes_allow_signature_invalidation(
        self, mock_convert_directory, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory CLI forwards --allow-signature-invalidation."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        mock_convert_directory.return_value = EXIT_SUCCESS

        result = runner.invoke(main, [str(input_dir), "--allow-signature-invalidation"])

        assert result.exit_code == EXIT_SUCCESS
        assert (
            mock_convert_directory.call_args.kwargs["allow_signature_invalidation"]
            is True
        )

    @patch("pdftopdfa.cli._convert_directory")
    def test_cli_convert_directory_passes_processing_flags(
        self, mock_convert_directory, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory CLI forwards both independent processing flags."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        mock_convert_directory.return_value = EXIT_SUCCESS

        result = runner.invoke(
            main,
            [
                str(input_dir),
                "--deskew",
                "--rotate-pages",
                "--ocr-execution-provider",
                "directml",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        kwargs = mock_convert_directory.call_args.kwargs
        assert kwargs["ocr_deskew"] is True
        assert kwargs["ocr_rotate_pages"] is True
        assert kwargs["ocr_execution_provider"] == "directml"

    def test_cli_convert_directory(
        self, runner: CliRunner, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Directory conversion works."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        # Create test PDFs
        for i in range(2):
            (input_dir / f"test{i}.pdf").write_bytes(sample_pdf_bytes)

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_SUCCESS
        assert "2 file(s) successfully converted" in result.output

    def test_cli_convert_directory_recursive(
        self, runner: CliRunner, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """--recursive processes subdirectories."""
        # Separate directories for each run to avoid overwrite conflicts
        input_dir_1 = tmp_dir / "input1"
        input_dir_1.mkdir()
        subdir_1 = input_dir_1 / "subdir"
        subdir_1.mkdir()

        (input_dir_1 / "main.pdf").write_bytes(sample_pdf_bytes)
        (subdir_1 / "sub.pdf").write_bytes(sample_pdf_bytes)

        # Without --recursive: only 1 file
        result_non_recursive = runner.invoke(main, [str(input_dir_1)])
        assert result_non_recursive.exit_code == EXIT_SUCCESS

        input_dir_2 = tmp_dir / "input2"
        input_dir_2.mkdir()
        subdir_2 = input_dir_2 / "subdir"
        subdir_2.mkdir()

        (input_dir_2 / "main.pdf").write_bytes(sample_pdf_bytes)
        (subdir_2 / "sub.pdf").write_bytes(sample_pdf_bytes)

        # With --recursive: both files
        result_recursive = runner.invoke(main, [str(input_dir_2), "--recursive"])
        assert result_recursive.exit_code == EXIT_SUCCESS

    def test_cli_convert_directory_force(
        self, runner: CliRunner, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """--force overwrites existing PDF/A files in directory mode."""
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)

        # First conversion
        result1 = runner.invoke(main, [str(input_dir), str(output_dir)])
        assert result1.exit_code == EXIT_SUCCESS

        output_file = output_dir / "test_pdfa.pdf"
        assert output_file.exists()

        # Without --force: file is skipped (reported as failure)
        result2 = runner.invoke(main, [str(input_dir), str(output_dir)])
        assert result2.exit_code == EXIT_GENERAL_ERROR

        # With --force: file is overwritten successfully
        result3 = runner.invoke(main, [str(input_dir), str(output_dir), "--force"])
        assert result3.exit_code == EXIT_SUCCESS
        assert output_file.exists()
        # File was re-created (content should be valid PDF)
        assert output_file.read_bytes()[:5] == b"%PDF-"

    def test_cli_convert_directory_converts_openable_encrypted_files(
        self, runner: CliRunner, encrypted_pdf: Path, tmp_dir: Path
    ) -> None:
        """Directory mode converts encrypted PDFs that need no password."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "encrypted.pdf").write_bytes(encrypted_pdf.read_bytes())

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_SUCCESS
        assert "1 file(s) successfully converted" in result.output
        assert "skipped and copied unchanged" not in result.output
        with Pdf.open(input_dir / "encrypted_pdfa.pdf") as pdf:
            assert pdf.is_encrypted is False


class TestCliValidation:
    """Tests for --validate option."""

    @patch(
        "pdftopdfa.converter.validate_with_verapdf",
        side_effect=VeraPDFError("veraPDF not installed"),
    )
    def test_cli_missing_validator_withholds_output_and_returns_failure(
        self,
        _mock_validate: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A missing validator suppresses publication of the staged output."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--validate"],
        )

        assert result.exit_code == EXIT_VALIDATION_FAILED
        assert "Validation: veraPDF could not run" in result.output
        assert not output_path.exists()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_cli_encrypted_validation_failure_withholds_output(
        self,
        mock_validate: MagicMock,
        runner: CliRunner,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Protected input still returns the validation failure exit code."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(password_encrypted_pdf), str(output_path), "--validate"],
        )

        assert result.exit_code == EXIT_VALIDATION_FAILED
        assert "validation could not run" in result.output
        assert not output_path.exists()
        mock_validate.assert_not_called()

    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_validation_runtime_error_returns_failure_status(
        self,
        mock_convert_to_pdfa: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A missing or crashed validator returns the validation failure code."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=False,
            input_path=sample_pdf,
            output_path=output_path,
            level="3b",
            warnings=["Validation: veraPDF could not run: veraPDF crashed"],
            error="Validation failed; output was not published",
            validation_failed=True,
            published=False,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "3b",
            do_validate=True,
            force=False,
            quiet=True,
        )

        assert result == EXIT_VALIDATION_FAILED
        assert (
            "Validation: veraPDF could not run: veraPDF crashed"
            in capsys.readouterr().err
        )

    @pytest.mark.skipif(
        not is_verapdf_available(),
        reason="veraPDF not installed",
    )
    def test_cli_convert_with_validate(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--validate performs validation."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--validate"])

        # Should succeed (conversion + validation)
        assert result.exit_code == EXIT_SUCCESS
        assert "validat" in result.output.lower()


class TestCliOcr:
    """Tests for --ocr option."""

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_ocr_flag_default_language(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr without --ocr-lang uses 'en' as default language."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--ocr", *OCR_MODEL_ARGS],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        assert mock_apply_ocr.call_args[0][2] == ["en"]
        assert mock_apply_ocr.call_args.kwargs["detection_model_dir"] == Path(
            "detection-model"
        )
        assert mock_apply_ocr.call_args.kwargs["recognition_model_dir"] == Path(
            "recognition-model"
        )
        assert mock_apply_ocr.call_args.kwargs["ocr_execution_provider"] == "cpu"

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_cli_forwards_layout_configuration(
        self,
        _mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The layout flag reaches the OCR engine."""
        import shutil

        mock_apply_ocr.side_effect = lambda inp, out, *args, **kwargs: (
            shutil.copy2(inp, out) or out
        )
        output_path = tmp_dir / "layout.pdf"
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-layout",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.kwargs["layout"] is True

    def test_cli_layout_requires_text_models(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Layout processing cannot run without the OCR model pair."""
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(tmp_dir / "layout.pdf"),
                "--ocr-layout",
            ],
        )

        assert result.exit_code == 2
        assert "OCR requires --ocr-detection-model-dir" in result.output

    def test_cli_figure_text_requires_level_a(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(tmp_dir / "figure-text.pdf"),
                "--level",
                "3b",
                "--ocr-figure-text",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 2
        assert "--ocr-figure-text requires --level 2a or 3a" in result.output

    def test_cli_figure_text_requires_text_models(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(tmp_dir / "figure-text.pdf"),
                "--level",
                "3a",
                "--ocr-figure-text",
            ],
        )

        assert result.exit_code == 2
        assert "OCR requires --ocr-detection-model-dir" in result.output

    @patch("pdftopdfa.converter.onnxruntime_engine_config")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_cli_ocr_selects_directml(
        self,
        _mock_is_ocr_available,
        mock_apply_ocr,
        _mock_engine_config,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr-execution-provider directml is forwarded explicitly."""
        import shutil

        mock_apply_ocr.side_effect = lambda inp, out, *args, **kwargs: (
            shutil.copy2(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-execution-provider",
                "directml",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.kwargs["ocr_execution_provider"] == "directml"

    @patch("pdftopdfa.converter.onnxruntime_engine_config")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_cli_ocr_selects_a_directml_device(
        self,
        _mock_is_ocr_available,
        mock_apply_ocr,
        _mock_engine_config,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A device index is accepted and forwarded in canonical form."""
        import shutil

        mock_apply_ocr.side_effect = lambda inp, out, *args, **kwargs: (
            shutil.copy2(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-execution-provider",
                "directml:01",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.kwargs["ocr_execution_provider"] == "directml:1"

    @pytest.mark.parametrize(
        "provider",
        ["cuda", "cpu:0", "directml:", "directml:-1", "directml:x", "directml:64"],
    )
    def test_cli_rejects_malformed_execution_provider(
        self,
        provider: str,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Invalid provider strings fail as a usage error, not a conversion."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-execution-provider",
                provider,
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 2
        assert "--ocr-execution-provider" in result.output
        assert not output_path.exists()

    def test_cli_directml_requires_model_pair(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Selecting DirectML enables OCR and therefore requires both models."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-execution-provider",
                "directml",
            ],
        )

        assert result.exit_code == 2
        assert "OCR requires --ocr-detection-model-dir" in result.output
        assert not output_path.exists()

    @patch(
        "pdftopdfa.converter.onnxruntime_engine_config",
        side_effect=OCRError(
            "DirectML was requested, but DmlExecutionProvider is unavailable. "
            "Install pdftopdfa[directml]."
        ),
    )
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_cli_directml_unavailable_fails_without_cpu_fallback(
        self,
        _mock_is_ocr_available,
        mock_apply_ocr,
        _mock_engine_config,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Unavailable DirectML is a clear conversion error, not CPU fallback."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-execution-provider",
                "directml",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_CONVERSION_FAILED
        assert "DmlExecutionProvider is unavailable" in result.output
        assert "pdftopdfa[directml]" in result.output
        assert not output_path.exists()
        mock_apply_ocr.assert_not_called()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_ocr_with_custom_language(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr --ocr-lang de uses 'de' as language."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr",
                "--ocr-lang",
                "de",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        assert mock_apply_ocr.call_args[0][2] == ["de"]

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_ocr_with_multiple_languages(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr --ocr-lang de+en uses multiple languages."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr",
                "--ocr-lang",
                "de+en",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        assert mock_apply_ocr.call_args[0][2] == ["de", "en"]

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_directory_with_ocr(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """--ocr also works with directory conversion."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        # Create test PDF
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)

        result = runner.invoke(main, [str(input_dir), "--ocr", *OCR_MODEL_ARGS])

        assert result.exit_code == EXIT_SUCCESS

    def test_cli_help_shows_only_paddle_model_options(self, runner: CliRunner) -> None:
        """Help exposes model paths and no removed preset/fallback controls."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--ocr-detection-model-dir" in result.output
        assert "--ocr-recognition-model-dir" in result.output
        assert "--ocr-quality" not in result.output
        assert "--ocr-fallback-quality" not in result.output
        assert "--ocr-fallback-after" not in result.output

    def test_cli_ocr_requires_model_pair(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr fails before conversion when the model pair is absent."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--ocr"])

        assert result.exit_code == 2
        assert "OCR requires --ocr-detection-model-dir" in result.output
        assert not output_path.exists()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_model_pair_implies_ocr(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A complete model pair enables OCR without a separate flag."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), *OCR_MODEL_ARGS],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.args[2] == ["en"]

    def test_cli_rejects_detection_model_without_recognition(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """One model directory is rejected before conversion."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-detection-model-dir",
                "detection-model",
            ],
        )

        assert result.exit_code == 2
        assert "must be provided together" in result.output

    def test_cli_rejects_recognition_model_without_detection(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """The inverse incomplete pair is rejected before conversion."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-recognition-model-dir",
                "recognition-model",
            ],
        )

        assert result.exit_code == 2
        assert "must be provided together" in result.output

    @pytest.mark.parametrize("language", ["eng", "deu", "unknown"])
    def test_cli_rejects_unsupported_language(
        self,
        language: str,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Legacy and unknown language codes are rejected without aliases."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr",
                "--ocr-lang",
                language,
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 2
        assert "Unsupported PaddleOCR language code" in result.output

    @pytest.mark.parametrize(
        ("flag", "expected_kwarg"),
        [("--deskew", "deskew"), ("--rotate-pages", "rotate_pages")],
    )
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_processing_flag_implies_ocr(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        flag: str,
        expected_kwarg: str,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Each standalone processing flag enables OCR with English."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / f"{expected_kwarg}.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), flag, *OCR_MODEL_ARGS],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.args[2] == ["en"]
        assert mock_apply_ocr.call_args.kwargs[expected_kwarg] is True

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_processing_flags_can_be_combined(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Deskew and rotation are forwarded independently when combined."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "combined.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--deskew",
                "--rotate-pages",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args.kwargs["deskew"] is True
        assert mock_apply_ocr.call_args.kwargs["rotate_pages"] is True

    def test_cli_deskew_rejects_forced_ocr(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Deskew and forced OCR fail before conversion starts."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--deskew",
                "--ocr-force",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 2
        assert "--deskew cannot be combined with --ocr-force" in result.output

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_ocr_force_implies_ocr(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr-force implies --ocr (no need to pass --ocr separately)."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--ocr-force", *OCR_MODEL_ARGS],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        mock_apply_ocr.assert_called_once()
        assert mock_apply_ocr.call_args[1]["force"] is True

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_cli_ocr_force_with_lang(
        self,
        mock_is_ocr_available,
        mock_apply_ocr,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """--ocr-force works with --ocr-lang."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-force",
                "--ocr-lang",
                "de",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == EXIT_SUCCESS
        assert mock_apply_ocr.call_args[0][2] == ["de"]

    def test_removed_quality_option_is_rejected(
        self,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Removed quality controls are not accepted by Click."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [
                str(sample_pdf),
                str(output_path),
                "--ocr-force",
                "--ocr-quality",
                "fast",
                *OCR_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 2
        assert "No such option '--ocr-quality'" in result.output


class TestDirectoryValidationFailures:
    """Tests for validation failure surfacing in directory mode."""

    @patch("pdftopdfa.cli.convert_directory")
    def test_validation_failure_returns_exit_code(
        self, mock_convert_dir, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory mode returns EXIT_VALIDATION_FAILED on validation failure."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(b"%PDF-1.4 dummy")

        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_dir / "test.pdf",
                output_path=tmp_dir / "test_pdfa.pdf",
                level="2b",
                warnings=["Validation: Rule 6.1.2 failed"],
                validation_failed=True,
            ),
        ]

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_VALIDATION_FAILED

    @patch("pdftopdfa.verapdf.is_verapdf_available", return_value=True)
    @patch("pdftopdfa.cli.convert_directory")
    def test_skipped_validation_failure_returns_exit_code(
        self,
        mock_convert_dir,
        _mock_available: MagicMock,
        runner: CliRunner,
        tmp_dir: Path,
    ) -> None:
        """A skipped, unvalidated copy is still a validation failure."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        input_path = input_dir / "encrypted.pdf"
        input_path.write_bytes(b"%PDF-1.4 dummy")

        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_path,
                output_path=tmp_dir / "encrypted_pdfa.pdf",
                level=None,
                warnings=["Validation: veraPDF could not run"],
                validation_failed=True,
                skipped=True,
            ),
        ]

        result = runner.invoke(main, [str(input_dir), "--validate"])

        assert result.exit_code == EXIT_VALIDATION_FAILED
        assert "1 file(s) failed validation" in result.output

    @patch("pdftopdfa.cli.convert_directory")
    def test_conversion_failure_takes_priority_over_validation(
        self, mock_convert_dir, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """EXIT_CONVERSION_FAILED takes priority over EXIT_VALIDATION_FAILED."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "a.pdf").write_bytes(b"%PDF-1.4 dummy")
        (input_dir / "b.pdf").write_bytes(b"%PDF-1.4 dummy")

        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_dir / "a.pdf",
                output_path=tmp_dir / "a_pdfa.pdf",
                level="2b",
                warnings=["Validation: Rule 6.1.2 failed"],
                validation_failed=True,
            ),
            ConversionResult(
                success=False,
                input_path=input_dir / "b.pdf",
                output_path=tmp_dir / "b_pdfa.pdf",
                level="2b",
                error="PDF processing error",
            ),
        ]

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_CONVERSION_FAILED

    @patch("pdftopdfa.cli.convert_directory")
    def test_validation_failure_summary_output(
        self, mock_convert_dir, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Summary output includes validation failure count and details."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(b"%PDF-1.4 dummy")

        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_dir / "test.pdf",
                output_path=tmp_dir / "test_pdfa.pdf",
                level="2b",
                warnings=["Validation: Rule 6.1.2 failed"],
                validation_failed=True,
            ),
        ]

        result = runner.invoke(main, [str(input_dir)])

        assert "1 file(s) failed validation" in result.output

    @pytest.mark.parametrize("quiet", [False, True])
    @patch("pdftopdfa.cli.convert_directory")
    def test_pdfua_validation_details_are_errors_in_directory_mode(
        self,
        mock_convert_dir: MagicMock,
        runner: CliRunner,
        tmp_dir: Path,
        quiet: bool,
    ) -> None:
        """Directory mode reports both mandatory validation profiles."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        input_path = input_dir / "test.pdf"
        input_path.write_bytes(b"%PDF-1.4 dummy")
        validation_errors = [
            "Validation: PDF/A rule failed",
            "PDF/UA validation: PDF/UA rule failed",
        ]
        mock_convert_dir.return_value = [
            ConversionResult(
                success=False,
                input_path=input_path,
                output_path=tmp_dir / "test_pdfa.pdf",
                level="2a",
                warnings=validation_errors,
                error="Validation failed; output candidate was published",
                validation_failed=True,
            )
        ]
        arguments = [str(input_dir), "--level", "2a", "--pdfua"]
        if quiet:
            arguments.append("--quiet")

        result = runner.invoke(main, arguments)

        assert result.exit_code == EXIT_VALIDATION_FAILED
        assert "1 file(s) failed validation" in result.stderr
        assert all(error in result.stderr for error in validation_errors)
        if quiet:
            assert "Summary:" not in result.output

    @patch("pdftopdfa.cli.convert_directory")
    def test_no_validation_failure_returns_success(
        self, mock_convert_dir, runner: CliRunner, tmp_dir: Path
    ) -> None:
        """Directory mode returns EXIT_SUCCESS when all files pass."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(b"%PDF-1.4 dummy")

        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_dir / "test.pdf",
                output_path=tmp_dir / "test_pdfa.pdf",
                level="2b",
            ),
        ]

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_SUCCESS

    @pytest.mark.parametrize("quiet", [True, False])
    @patch("pdftopdfa.cli.convert_directory")
    def test_review_required_returns_distinct_exit_code(
        self,
        mock_convert_dir: MagicMock,
        runner: CliRunner,
        tmp_dir: Path,
        quiet: bool,
    ) -> None:
        """Batch automation can gate machine-valid files needing review."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        input_path = input_dir / "test.pdf"
        input_path.write_bytes(b"%PDF-1.4 dummy")
        mock_convert_dir.return_value = [
            ConversionResult(
                success=True,
                input_path=input_path,
                output_path=tmp_dir / "test_pdfa.pdf",
                level="2a",
                pdfua_status=PDFUAStatus.REVIEW_REQUIRED,
            )
        ]
        arguments = [str(input_dir), "--level", "2a", "--pdfua"]
        if quiet:
            arguments.append("--quiet")

        result = runner.invoke(main, arguments)

        assert result.exit_code == EXIT_REVIEW_REQUIRED
        assert "review" in result.output.lower()
