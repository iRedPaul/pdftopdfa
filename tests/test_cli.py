# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for cli.py."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

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


class TestCliVersion:
    """Tests for --version option."""

    def test_cli_version(self, runner: CliRunner) -> None:
        """--version returns exit code 0 and shows version."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output


class TestCliConvert:
    """Tests for file conversion."""

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


class TestCliValidation:
    """Tests for --validate option."""

    @pytest.mark.skipif(
        not __import__("shutil").which("verapdf"),
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

    def test_cli_ocr_flag_default_language(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr without --ocr-lang uses 'eng' as default language."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--ocr"])

        # Conversion should succeed
        # (OCR is only applied when needed or warning is issued)
        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()

    def test_cli_ocr_with_custom_language(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr --ocr-lang eng uses 'eng' as language."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main, [str(sample_pdf), str(output_path), "--ocr", "--ocr-lang", "eng"]
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()

    def test_cli_ocr_with_multiple_languages(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr --ocr-lang deu+eng uses multiple languages."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main, [str(sample_pdf), str(output_path), "--ocr", "--ocr-lang", "deu+eng"]
        )

        assert result.exit_code == EXIT_SUCCESS
        assert output_path.exists()

    def test_cli_directory_with_ocr(
        self, runner: CliRunner, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """--ocr also works with directory conversion."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        # Create test PDF
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)

        result = runner.invoke(main, [str(input_dir), "--ocr"])

        assert result.exit_code == EXIT_SUCCESS

    def test_cli_ocr_quality_option_in_help(self, runner: CliRunner) -> None:
        """--ocr-quality option appears in help."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--ocr-quality" in result.output

    def test_cli_ocr_quality_default(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr-quality defaults to 'default'."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(main, [str(sample_pdf), str(output_path), "--ocr"])

        assert result.exit_code == EXIT_SUCCESS

    def test_cli_ocr_quality_fast(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr-quality fast is accepted."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--ocr", "--ocr-quality", "fast"],
        )

        assert result.exit_code == EXIT_SUCCESS

    def test_cli_ocr_quality_best(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr-quality best is accepted."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--ocr", "--ocr-quality", "best"],
        )

        assert result.exit_code == EXIT_SUCCESS

    def test_cli_ocr_quality_invalid(
        self, runner: CliRunner, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """--ocr-quality with invalid value is rejected."""
        output_path = tmp_dir / "output.pdf"

        result = runner.invoke(
            main,
            [str(sample_pdf), str(output_path), "--ocr", "--ocr-quality", "ultra"],
        )

        assert result.exit_code == 2  # Click rejects invalid choice


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
