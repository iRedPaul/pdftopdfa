# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for cli.py."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from colorama import Fore, Style

import pdftopdfa.cli as cli_module
from pdftopdfa import __version__
from pdftopdfa.cli import (
    EXIT_CONVERSION_FAILED,
    EXIT_FILE_NOT_FOUND,
    EXIT_GENERAL_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_FAILED,
    main,
)
from pdftopdfa.converter import ConversionResult
from pdftopdfa.verapdf import is_verapdf_available

OCR_MODEL_ARGS = [
    "--ocr-detection-model-dir",
    "detection-model",
    "--ocr-recognition-model-dir",
    "recognition-model",
]


@pytest.fixture
def runner() -> CliRunner:
    """CLI Test Runner."""
    return CliRunner()


class TestCliHelp:
    """Tests for --help option."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """--help returns exit code 0 and shows options."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--level" in result.output
        assert "--validate" in result.output
        assert "--recursive" in result.output
        assert "--force" in result.output
        assert "--quiet" in result.output
        assert "--verbose" in result.output

    def test_cli_help_shows_ocr_option(self, runner: CliRunner) -> None:
        """--ocr option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--ocr" in result.output
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

    @patch("pdftopdfa.cli.validate_with_verapdf")
    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_single_file_skipped_result_does_not_validate(
        self,
        mock_convert_to_pdfa,
        mock_validate,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Single-file skipped results bypass manual veraPDF validation."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="3b",
            skipped=True,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "3b",
            do_validate=True,
            force=False,
            quiet=True,
        )

        assert result == EXIT_SUCCESS
        mock_validate.assert_not_called()

    @patch("pdftopdfa.cli.validate_with_verapdf")
    @patch("pdftopdfa.cli.convert_to_pdfa")
    def test_cli_single_file_known_validation_failure_returns_exit_code(
        self,
        mock_convert_to_pdfa,
        mock_validate,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A known compliance failure returns the validation failure exit code."""
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=sample_pdf,
            output_path=output_path,
            level="2b",
            validation_failed=True,
        )

        result = cli_module._convert_single_file(
            sample_pdf,
            str(output_path),
            "2b",
            do_validate=True,
            force=False,
            quiet=True,
        )

        assert result == EXIT_VALIDATION_FAILED
        mock_validate.assert_not_called()

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

    def test_cli_convert_encrypted_pdf_is_skipped(
        self, runner: CliRunner, encrypted_pdf: Path, tmp_dir: Path
    ) -> None:
        """Encrypted PDFs are copied unchanged and reported as skipped."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(encrypted_pdf), str(output_path)])

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()
        assert output_path.read_bytes() == encrypted_pdf.read_bytes()
        assert "Skipped:" in result.output
        assert "encrypted" in result.output


class TestCliMissingInput:
    """Tests for missing input file."""

    def test_cli_missing_input(self, runner: CliRunner, tmp_dir: Path) -> None:
        """Missing input file returns exit code 2."""
        nonexistent = tmp_dir / "nonexistent.pdf"

        result = runner.invoke(main, [str(nonexistent)])

        # Click returns exit code 2 for missing file (exists=True)
        assert result.exit_code == EXIT_FILE_NOT_FOUND


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
            [str(input_dir), "--deskew", "--rotate-pages", *OCR_MODEL_ARGS],
        )

        assert result.exit_code == EXIT_SUCCESS
        kwargs = mock_convert_directory.call_args.kwargs
        assert kwargs["ocr_deskew"] is True
        assert kwargs["ocr_rotate_pages"] is True

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
        from pdftopdfa.cli import EXIT_CONVERSION_FAILED

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
        assert result2.exit_code == EXIT_CONVERSION_FAILED

        # With --force: file is overwritten successfully
        result3 = runner.invoke(main, [str(input_dir), str(output_dir), "--force"])
        assert result3.exit_code == EXIT_SUCCESS
        assert output_file.exists()
        # File was re-created (content should be valid PDF)
        assert output_file.read_bytes()[:5] == b"%PDF-"

    def test_cli_convert_directory_reports_skipped_files(
        self, runner: CliRunner, encrypted_pdf: Path, tmp_dir: Path
    ) -> None:
        """Directory mode reports skipped encrypted PDFs as warnings."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "encrypted.pdf").write_bytes(encrypted_pdf.read_bytes())

        result = runner.invoke(main, [str(input_dir)])

        assert result.exit_code == EXIT_SUCCESS
        assert "0 file(s) successfully converted" in result.output
        assert "1 file(s) skipped and copied unchanged" in result.output
        assert "encrypted.pdf: Conversion skipped: PDF is encrypted" in result.output


class TestCliValidation:
    """Tests for --validate option."""

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
