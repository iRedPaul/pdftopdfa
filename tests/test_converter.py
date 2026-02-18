# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for converter.py."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.converter import (
    ConversionResult,
    _compare_pdfa_levels,
    _ensure_binary_comment,
    _truncate_trailing_data,
    _verify_file_structure,
    convert_directory,
    convert_files,
    convert_to_pdfa,
    generate_output_path,
)
from pdftopdfa.exceptions import ConversionError, UnsupportedPDFError
from pdftopdfa.verapdf import VeraPDFResult


class TestComparePdfaLevels:
    """Tests for _compare_pdfa_levels."""

    def test_same_level_returns_zero(self) -> None:
        """Same level returns 0."""
        assert _compare_pdfa_levels("2b", "2b") == 0
        assert _compare_pdfa_levels("1a", "1a") == 0
        assert _compare_pdfa_levels("3u", "3u") == 0

    def test_lower_part_returns_negative(self) -> None:
        """Lower part number returns -1."""
        assert _compare_pdfa_levels("1b", "2b") == -1
        assert _compare_pdfa_levels("2b", "3b") == -1
        assert _compare_pdfa_levels("1a", "3a") == -1

    def test_different_part_returns_negative(self) -> None:
        """Different part number always returns -1 (parts are not ordered)."""
        assert _compare_pdfa_levels("3b", "2b") == -1
        assert _compare_pdfa_levels("2b", "1b") == -1
        assert _compare_pdfa_levels("3a", "1a") == -1

    def test_higher_conformance_returns_positive(self) -> None:
        """Higher conformance (a > u > b) returns 1."""
        assert _compare_pdfa_levels("2a", "2b") == 1
        assert _compare_pdfa_levels("2u", "2b") == 1
        assert _compare_pdfa_levels("2a", "2u") == 1

    def test_lower_conformance_returns_negative(self) -> None:
        """Lower conformance returns -1."""
        assert _compare_pdfa_levels("2b", "2a") == -1
        assert _compare_pdfa_levels("2b", "2u") == -1
        assert _compare_pdfa_levels("2u", "2a") == -1

    def test_different_part_ignores_conformance(self) -> None:
        """Cross-part comparisons always return -1 regardless of conformance."""
        assert _compare_pdfa_levels("3b", "2a") == -1
        assert _compare_pdfa_levels("1a", "2b") == -1

    def test_pdfa4_vs_other_parts(self) -> None:
        """PDF/A-4 vs other parts always returns -1."""
        assert _compare_pdfa_levels("4", "3b") == -1
        assert _compare_pdfa_levels("4e", "2b") == -1
        assert _compare_pdfa_levels("3b", "4") == -1

    def test_pdfa4_same_level(self) -> None:
        """PDF/A-4 vs PDF/A-4 returns 0."""
        assert _compare_pdfa_levels("4", "4") == 0


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_successful_result(self, tmp_dir: Path) -> None:
        """Checks dataclass with success=True."""
        result = ConversionResult(
            success=True,
            input_path=tmp_dir / "input.pdf",
            output_path=tmp_dir / "output.pdf",
            level="2b",
            warnings=["Warning 1"],
            processing_time=1.5,
        )
        assert result.success is True
        assert result.level == "2b"
        assert result.error is None
        assert len(result.warnings) == 1
        assert result.processing_time == 1.5

    def test_failed_result(self, tmp_dir: Path) -> None:
        """Checks error field with success=False."""
        result = ConversionResult(
            success=False,
            input_path=tmp_dir / "input.pdf",
            output_path=tmp_dir / "output.pdf",
            level="2b",
            error="Conversion failed",
        )
        assert result.success is False
        assert result.error == "Conversion failed"

    def test_validation_failed_defaults_to_false(self, tmp_dir: Path) -> None:
        """validation_failed defaults to False."""
        result = ConversionResult(
            success=True,
            input_path=tmp_dir / "input.pdf",
            output_path=tmp_dir / "output.pdf",
            level="2b",
        )
        assert result.validation_failed is False

    def test_validation_failed_set_to_true(self, tmp_dir: Path) -> None:
        """validation_failed can be explicitly set to True."""
        result = ConversionResult(
            success=True,
            input_path=tmp_dir / "input.pdf",
            output_path=tmp_dir / "output.pdf",
            level="2b",
            validation_failed=True,
        )
        assert result.validation_failed is True


class TestGenerateOutputPath:
    """Tests for generate_output_path."""

    def test_default_output_same_directory(self, tmp_dir: Path) -> None:
        """Generates output path in same directory."""
        input_path = tmp_dir / "document.pdf"
        output_path = generate_output_path(input_path)

        assert output_path.parent == tmp_dir
        assert output_path.name == "document_pdfa.pdf"

    def test_custom_output_directory(self, tmp_dir: Path) -> None:
        """Generates output path in custom directory."""
        input_path = tmp_dir / "document.pdf"
        output_dir = tmp_dir / "output"
        output_path = generate_output_path(input_path, output_dir)

        assert output_path.parent == output_dir
        assert output_path.name == "document_pdfa.pdf"


class TestConvertToPdfa:
    """Tests for convert_to_pdfa."""

    def test_convert_simple_pdf(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """Simple conversion with success check."""
        output_path = tmp_dir / "output_pdfa.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        assert result.input_path == sample_pdf
        assert result.output_path == output_path
        assert result.level == "2b"
        assert output_path.exists()

    def test_convert_nonexistent_file(self, tmp_dir: Path) -> None:
        """Non-existent file raises ConversionError."""
        nonexistent = tmp_dir / "nonexistent.pdf"
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(ConversionError):
            convert_to_pdfa(nonexistent, output_path)

    def test_convert_invalid_level_raises_error(self, tmp_dir: Path) -> None:
        """Invalid level raises ConversionError before any processing."""
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            convert_to_pdfa(input_path, output_path, level="invalid")

    def test_convert_encrypted_pdf(self, encrypted_pdf: Path, tmp_dir: Path) -> None:
        """Encrypted PDF raises UnsupportedPDFError."""
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(UnsupportedPDFError, match="encrypted"):
            convert_to_pdfa(encrypted_pdf, output_path)

    @pytest.mark.parametrize("level", ["2b", "2u", "3b", "3u"])
    def test_convert_all_levels(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Conversion works for all PDF/A levels."""
        output_path = tmp_dir / f"output_{level}.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level=level)

        assert result.success is True
        assert result.level == level
        assert output_path.exists()

    def test_convert_with_validation_flag(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """validate=True runs validation without errors for compliant PDF."""
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, validate=True)

        assert result.success is True
        # Compliant PDF should have no validation errors
        has_validation_error = any("Validation:" in w for w in result.warnings)
        assert not has_validation_error
        assert result.validation_failed is False

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_with_failing_validation_sets_flag(
        self, mock_verapdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """validation_failed is True when veraPDF reports non-compliance."""
        mock_verapdf.return_value = MagicMock(
            compliant=False,
            errors=["Rule 6.1.2 failed"],
        )
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, validate=True)

        assert result.success is True
        assert result.validation_failed is True
        assert any("Validation: Rule 6.1.2 failed" in w for w in result.warnings)

    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_language_adds_warning_when_unavailable(
        self, mock_is_ocr_available: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Warning when OCR requested but not available."""
        mock_is_ocr_available.return_value = False
        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(sample_pdf, output_path, ocr_languages=["deu"])

        assert result.success is True
        has_ocr_warning = any("OCR not available" in w for w in result.warnings)
        assert has_ocr_warning

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.needs_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_languages_parameter(
        self,
        mock_is_ocr_available: MagicMock,
        mock_needs_ocr: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_languages is passed through to apply_ocr."""
        mock_is_ocr_available.return_value = True
        mock_needs_ocr.return_value = True

        # apply_ocr should create the temporary file
        def create_ocr_output(
            input_path: Path, output_path: Path, langs: list[str], **kwargs: object
        ) -> Path:
            # Copy input to output (simulates OCR)
            import shutil

            shutil.copy(input_path, output_path)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output

        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(sample_pdf, output_path, ocr_languages=["eng"])

        assert result.success is True
        # Check if apply_ocr was called with the correct languages
        mock_apply_ocr.assert_called_once()
        call_args = mock_apply_ocr.call_args
        assert call_args[0][2] == ["eng"]  # Languages parameter

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.needs_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_adds_warning_message(
        self,
        mock_is_ocr_available: MagicMock,
        mock_needs_ocr: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR execution adds warning with language info."""
        mock_is_ocr_available.return_value = True
        mock_needs_ocr.return_value = True

        def create_ocr_output(
            input_path: Path, output_path: Path, langs: list[str], **kwargs: object
        ) -> Path:
            import shutil

            shutil.copy(input_path, output_path)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output

        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(sample_pdf, output_path, ocr_languages=["deu", "eng"])

        assert result.success is True
        has_ocr_done_warning = any(
            "OCR performed" in w and "deu+eng" in w for w in result.warnings
        )
        assert has_ocr_done_warning

    @patch("pdftopdfa.ocr.needs_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_skips_ocr_when_not_needed(
        self,
        mock_is_ocr_available: MagicMock,
        mock_needs_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR is skipped when PDF already contains text."""
        mock_is_ocr_available.return_value = True
        mock_needs_ocr.return_value = False  # PDF doesn't need OCR

        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(sample_pdf, output_path, ocr_languages=["deu"])

        assert result.success is True
        # No OCR warning should be present
        has_ocr_done_warning = any("OCR performed" in w for w in result.warnings)
        assert not has_ocr_done_warning

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.needs_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_skips_needs_ocr_check(
        self,
        mock_is_ocr_available: MagicMock,
        mock_needs_ocr: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_force=True skips needs_ocr() and calls apply_ocr(force=True)."""
        mock_is_ocr_available.return_value = True

        def create_ocr_output(
            input_path: Path, output_path: Path, langs: list[str], **kwargs: object
        ) -> Path:
            import shutil

            shutil.copy(input_path, output_path)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output

        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            sample_pdf, output_path, ocr_languages=["eng"], ocr_force=True
        )

        assert result.success is True
        # needs_ocr should NOT have been called
        mock_needs_ocr.assert_not_called()
        # apply_ocr should have been called with force=True
        mock_apply_ocr.assert_called_once()
        call_kwargs = mock_apply_ocr.call_args[1]
        assert call_kwargs["force"] is True

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.needs_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_false_checks_needs_ocr(
        self,
        mock_is_ocr_available: MagicMock,
        mock_needs_ocr: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_force=False (default) still calls needs_ocr()."""
        mock_is_ocr_available.return_value = True
        mock_needs_ocr.return_value = False

        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            sample_pdf, output_path, ocr_languages=["eng"], ocr_force=False
        )

        assert result.success is True
        mock_needs_ocr.assert_called_once()
        mock_apply_ocr.assert_not_called()

    def test_upgrades_pdf_version_and_adds_warning(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """PDF version is upgraded from 1.3 to 1.7 with warning."""
        # Verify input PDF has version < 1.7 (pikepdf creates PDF 1.3 by default)
        with Pdf.open(sample_pdf) as input_pdf:
            assert input_pdf.pdf_version < "1.7"

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True

        # Check warning about version upgrade
        has_version_warning = any("PDF version upgraded" in w for w in result.warnings)
        assert has_version_warning

        # Verify output PDF has version >= 1.7
        with Pdf.open(output_path) as output_pdf:
            assert output_pdf.pdf_version >= "1.7"

    @patch("pdftopdfa.converter.embed_color_profiles")
    def test_repairs_late_invalid_utf8_colorspace_names(
        self,
        mock_embed_color_profiles: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Final structure sanitization repairs names introduced late in pipeline."""

        def inject_invalid_name(*args, **kwargs) -> set:
            pdf = args[0]
            page = pdf.pages[0]

            resources = page.obj.get("/Resources")
            if resources is None:
                resources = Dictionary()
                page.obj[Name.Resources] = resources

            colorspaces = resources.get("/ColorSpace")
            if colorspaces is None:
                colorspaces = Dictionary()
                resources[Name.ColorSpace] = colorspaces

            colorspaces[Name("/CSbad")] = Array(
                [
                    Name.Separation,
                    Name("/Custom#c3"),
                    Name.DeviceCMYK,
                    Dictionary(),
                ]
            )

            page.obj[Name.Contents] = pdf.make_stream(b"/CSbad cs 0 scn")
            return set()

        mock_embed_color_profiles.side_effect = inject_invalid_name

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        with Pdf.open(output_path) as output_pdf:
            cs_bad = output_pdf.pages[0].Resources.ColorSpace.CSbad
            # Without late sanitization this would raise UnicodeDecodeError.
            assert str(cs_bad[1]).startswith("/")

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_already_compliant_pdf_is_skipped(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Already-compliant PDF is copied without conversion."""
        mock_detect.return_value = "2b"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="2b")

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        assert result.level == "2b"
        assert any("already valid" in w for w in result.warnings)
        assert output_path.exists()

    def test_corrupt_pdf_raises_conversion_error(self, tmp_dir: Path) -> None:
        """Corrupt PDF triggers PdfError which is wrapped as ConversionError."""
        corrupt_path = tmp_dir / "corrupt.pdf"
        corrupt_path.write_bytes(b"%PDF-1.4 this is not valid pdf content")
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(ConversionError, match="PDF processing error"):
            convert_to_pdfa(corrupt_path, output_path)

    def test_convert_with_calibrated_false(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """convert_calibrated=False produces a valid output."""
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf, output_path, level="2b", convert_calibrated=False
        )

        assert result.success is True
        assert output_path.exists()


class TestConvertDirectory:
    """Tests for convert_directory."""

    def test_convert_empty_directory(self, tmp_dir: Path) -> None:
        """Empty directory returns empty list."""
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()

        results = convert_directory(empty_dir, show_progress=False)
        assert results == []

    def test_convert_directory_nonexistent(self, tmp_dir: Path) -> None:
        """Non-existent directory raises ConversionError."""
        nonexistent = tmp_dir / "nonexistent"

        with pytest.raises(ConversionError, match="does not exist"):
            convert_directory(nonexistent)

    def test_convert_directory_with_pdfs(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Directory with PDFs is processed correctly."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        # Create 3 test PDFs
        for i in range(3):
            (input_dir / f"test{i}.pdf").write_bytes(sample_pdf_bytes)

        results = convert_directory(input_dir, show_progress=False)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_convert_directory_recursive(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Recursive vs. non-recursive processing."""
        # Create separate directories for each test run
        input_dir_1 = tmp_dir / "input1"
        input_dir_1.mkdir()
        subdir_1 = input_dir_1 / "subdir"
        subdir_1.mkdir()
        output_dir_1 = tmp_dir / "output1"

        # PDF in main directory
        (input_dir_1 / "main.pdf").write_bytes(sample_pdf_bytes)
        # PDF in subdirectory
        (subdir_1 / "sub.pdf").write_bytes(sample_pdf_bytes)

        # Non-recursive: only 1 PDF
        results_non_recursive = convert_directory(
            input_dir_1, output_dir=output_dir_1, recursive=False, show_progress=False
        )
        assert len(results_non_recursive) == 1

        # Second directory for recursive test
        input_dir_2 = tmp_dir / "input2"
        input_dir_2.mkdir()
        subdir_2 = input_dir_2 / "subdir"
        subdir_2.mkdir()
        output_dir_2 = tmp_dir / "output2"

        (input_dir_2 / "main.pdf").write_bytes(sample_pdf_bytes)
        (subdir_2 / "sub.pdf").write_bytes(sample_pdf_bytes)

        # Recursive: both PDFs
        results_recursive = convert_directory(
            input_dir_2, output_dir=output_dir_2, recursive=True, show_progress=False
        )
        assert len(results_recursive) == 2

    def test_convert_directory_skips_pdfa_files(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Previous _pdfa.pdf outputs are skipped when output_dir is None."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        (input_dir / "doc.pdf").write_bytes(sample_pdf_bytes)
        (input_dir / "doc_pdfa.pdf").write_bytes(sample_pdf_bytes)

        results = convert_directory(input_dir, show_progress=False)

        assert len(results) == 1
        assert results[0].input_path == input_dir / "doc.pdf"

    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_directory_with_ocr_languages(
        self,
        mock_is_ocr_available: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """ocr_languages parameter is passed through to convert_to_pdfa."""
        mock_is_ocr_available.return_value = False  # OCR not available

        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)

        results = convert_directory(
            input_dir, show_progress=False, ocr_languages=["deu"]
        )

        assert len(results) == 1
        assert results[0].success is True
        # Since OCR is not available, warning should be present
        has_ocr_warning = any("OCR not available" in w for w in results[0].warnings)
        assert has_ocr_warning


class TestConvertFiles:
    """Tests for convert_files."""

    def test_convert_files_basic(self, tmp_dir: Path, sample_pdf_bytes: bytes) -> None:
        """Successful conversion of a file list."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        file_pairs: list[tuple[Path, Path]] = []
        for i in range(3):
            in_path = input_dir / f"test{i}.pdf"
            in_path.write_bytes(sample_pdf_bytes)
            out_path = output_dir / f"test{i}_pdfa.pdf"
            file_pairs.append((in_path, out_path))

        results = convert_files(file_pairs)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.output_path.exists() for r in results)

    def test_convert_files_skip_existing_without_force(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Output exists without force_overwrite -> skip with error."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        in_path = input_dir / "test.pdf"
        in_path.write_bytes(sample_pdf_bytes)
        out_path = output_dir / "test_pdfa.pdf"
        out_path.write_bytes(b"existing content")

        results = convert_files([(in_path, out_path)], force_overwrite=False)

        assert len(results) == 1
        assert results[0].success is False
        assert "already exists" in results[0].error

    def test_convert_files_overwrite_with_force(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """force_overwrite=True overwrites existing output."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        in_path = input_dir / "test.pdf"
        in_path.write_bytes(sample_pdf_bytes)
        out_path = output_dir / "test_pdfa.pdf"
        out_path.write_bytes(b"existing content")

        results = convert_files([(in_path, out_path)], force_overwrite=True)

        assert len(results) == 1
        assert results[0].success is True
        # Output should be a valid PDF now, not "existing content"
        assert out_path.stat().st_size > len(b"existing content")

    def test_convert_files_cancellation(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """cancel_event stops processing."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        file_pairs: list[tuple[Path, Path]] = []
        for i in range(5):
            in_path = input_dir / f"test{i}.pdf"
            in_path.write_bytes(sample_pdf_bytes)
            out_path = output_dir / f"test{i}_pdfa.pdf"
            file_pairs.append((in_path, out_path))

        # Set cancel event before starting
        cancel = threading.Event()
        cancel.set()

        results = convert_files(file_pairs, cancel_event=cancel)

        # Should have processed 0 files (cancelled before first iteration)
        assert len(results) == 0

    def test_convert_files_progress_callback(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """on_progress is called for each file."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        file_pairs: list[tuple[Path, Path]] = []
        for i in range(3):
            in_path = input_dir / f"test{i}.pdf"
            in_path.write_bytes(sample_pdf_bytes)
            out_path = output_dir / f"test{i}_pdfa.pdf"
            file_pairs.append((in_path, out_path))

        progress_calls: list[tuple[int, int, str]] = []

        def on_progress(idx: int, total: int, filename: str) -> None:
            progress_calls.append((idx, total, filename))

        convert_files(file_pairs, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (0, 3, "test0.pdf")
        assert progress_calls[1] == (1, 3, "test1.pdf")
        assert progress_calls[2] == (2, 3, "test2.pdf")

    def test_convert_files_error_continues(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Error on one file doesn't stop others."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        # First file: valid PDF
        good1 = input_dir / "good1.pdf"
        good1.write_bytes(sample_pdf_bytes)
        out1 = output_dir / "good1_pdfa.pdf"

        # Second file: invalid PDF (will cause error)
        bad = input_dir / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        out_bad = output_dir / "bad_pdfa.pdf"

        # Third file: valid PDF
        good2 = input_dir / "good2.pdf"
        good2.write_bytes(sample_pdf_bytes)
        out2 = output_dir / "good2_pdfa.pdf"

        results = convert_files(
            [
                (good1, out1),
                (bad, out_bad),
                (good2, out2),
            ]
        )

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error is not None
        assert results[2].success is True

    def test_convert_files_empty_list(self) -> None:
        """Empty file list returns empty results."""
        results = convert_files([])
        assert results == []


class TestVerifyFileStructure:
    """Tests for _verify_file_structure."""

    def test_valid_pdf_no_warnings(
        self, sample_pdf: Path, tmp_dir: Path, caplog
    ) -> None:
        """Valid converted PDF produces no warnings."""
        import logging

        output_path = tmp_dir / "output.pdf"
        convert_to_pdfa(sample_pdf, output_path, level="2b")

        with caplog.at_level(logging.WARNING):
            _verify_file_structure(output_path, "1.7")

        assert not any("Post-save verification" in r.message for r in caplog.records)

    def test_bad_header_logs_warning(self, tmp_dir: Path, caplog) -> None:
        """File with wrong header produces a warning."""
        import logging

        bad_file = tmp_dir / "bad.pdf"
        bad_file.write_bytes(b"%PDF-2.0 garbage data\n%\xe2\xe3\xcf\xd3\n")

        with caplog.at_level(logging.WARNING):
            _verify_file_structure(bad_file, "1.7")

        assert any("does not start with" in r.message for r in caplog.records)

    def test_nonexistent_file_logs_warning(self, tmp_dir: Path, caplog) -> None:
        """Non-existent file path produces a warning."""
        import logging

        missing = tmp_dir / "missing.pdf"

        with caplog.at_level(logging.WARNING):
            _verify_file_structure(missing, "1.7")

        assert any("could not read" in r.message for r in caplog.records)

    def test_convert_without_validate_runs_verification(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """convert_to_pdfa without validate=True still runs verification."""
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b", validate=False)
        assert result.success is True
        assert output_path.exists()


class TestTruncateTrailingData:
    """Tests for _truncate_trailing_data."""

    def test_no_trailing_data_after_eof_newline(self, tmp_dir: Path) -> None:
        """File ending with %%EOF\\n is not modified."""
        f = tmp_dir / "test.pdf"
        data = b"%PDF-1.7\nsome content\n%%EOF\n"
        f.write_bytes(data)
        assert _truncate_trailing_data(f) is False
        assert f.read_bytes() == data

    def test_no_trailing_data_after_eof_bare(self, tmp_dir: Path) -> None:
        """File ending with %%EOF (no EOL) is not modified."""
        f = tmp_dir / "test.pdf"
        data = b"%PDF-1.7\nsome content\n%%EOF"
        f.write_bytes(data)
        assert _truncate_trailing_data(f) is False
        assert f.read_bytes() == data

    def test_no_trailing_data_after_eof_crlf(self, tmp_dir: Path) -> None:
        """File ending with %%EOF\\r\\n is not modified."""
        f = tmp_dir / "test.pdf"
        data = b"%PDF-1.7\nsome content\n%%EOF\r\n"
        f.write_bytes(data)
        assert _truncate_trailing_data(f) is False
        assert f.read_bytes() == data

    def test_truncates_trailing_data_after_eof_newline(self, tmp_dir: Path) -> None:
        """Trailing data after %%EOF\\n is removed."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n%%EOF\ntrailing junk")
        assert _truncate_trailing_data(f) is True
        assert f.read_bytes() == b"%PDF-1.7\nsome content\n%%EOF\n"

    def test_truncates_trailing_data_after_eof_crlf(self, tmp_dir: Path) -> None:
        """Trailing data after %%EOF\\r\\n is removed."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n%%EOF\r\nextra bytes")
        assert _truncate_trailing_data(f) is True
        assert f.read_bytes() == b"%PDF-1.7\nsome content\n%%EOF\r\n"

    def test_truncates_trailing_data_after_eof_cr(self, tmp_dir: Path) -> None:
        """Trailing data after %%EOF\\r is removed."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n%%EOF\rtrailing")
        assert _truncate_trailing_data(f) is True
        assert f.read_bytes() == b"%PDF-1.7\nsome content\n%%EOF\r"

    def test_truncates_trailing_data_after_bare_eof(self, tmp_dir: Path) -> None:
        """Trailing data directly after %%EOF (no EOL) is removed."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n%%EOFgarbage")
        assert _truncate_trailing_data(f) is True
        assert f.read_bytes() == b"%PDF-1.7\nsome content\n%%EOF"

    def test_uses_last_eof_marker(self, tmp_dir: Path) -> None:
        """Only data after the last %%EOF is truncated."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\ncontent\n%%EOF\nincremental update\n%%EOF\ntrailing")
        assert _truncate_trailing_data(f) is True
        assert f.read_bytes() == (
            b"%PDF-1.7\ncontent\n%%EOF\nincremental update\n%%EOF\n"
        )

    def test_no_eof_marker_returns_false(self, tmp_dir: Path) -> None:
        """File without %%EOF returns False."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n")
        assert _truncate_trailing_data(f) is False

    def test_nonexistent_file_returns_false(self, tmp_dir: Path) -> None:
        """Non-existent file returns False."""
        f = tmp_dir / "missing.pdf"
        assert _truncate_trailing_data(f) is False

    def test_integration_converted_pdf_has_no_trailing_data(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Converted PDF has no trailing data after %%EOF."""
        output = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output, level="2b")
        assert result.success is True

        data = output.read_bytes()
        last_eof = data.rfind(b"%%EOF")
        assert last_eof != -1
        after = last_eof + len(b"%%EOF")
        # Only optional single EOL allowed
        tail = data[after:]
        assert tail in (b"", b"\n", b"\r", b"\r\n")


class TestEnsureBinaryComment:
    """Tests for _ensure_binary_comment."""

    def _has_binary_comment(self, path: Path) -> bool:
        """Check if file has a valid binary comment on the second line."""
        with open(path, "rb") as f:
            header = f.read(64)
        nl = header.find(b"\n")
        if nl == -1:
            return False
        after = nl + 1
        if after >= len(header) or header[after : after + 1] != b"%":
            return False
        comment_end = header.find(b"\n", after)
        if comment_end == -1:
            line = header[after + 1 :]
        else:
            line = header[after + 1 : comment_end]
        if line.endswith(b"\r"):
            line = line[:-1]
        return sum(1 for b in line if b > 127) >= 4

    def test_already_has_binary_comment(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """File with existing binary comment is not modified."""
        output = tmp_dir / "output.pdf"
        convert_to_pdfa(sample_pdf, output, level="2b")

        original_data = output.read_bytes()
        assert self._has_binary_comment(output)
        assert _ensure_binary_comment(output, "1.7") is False
        assert output.read_bytes() == original_data

    def test_missing_binary_comment_is_fixed(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """File without binary comment gets one after re-save."""
        output = tmp_dir / "output.pdf"
        convert_to_pdfa(sample_pdf, output, level="2b")

        # Strip the binary comment line from the saved file
        data = output.read_bytes()
        first_nl = data.find(b"\n")
        second_nl = data.find(b"\n", first_nl + 1)
        stripped = data[: first_nl + 1] + data[second_nl + 1 :]
        output.write_bytes(stripped)

        assert not self._has_binary_comment(output)
        assert _ensure_binary_comment(output, "1.7") is True
        assert self._has_binary_comment(output)

        # Verify the file is still valid
        with Pdf.open(output) as repaired:
            assert len(repaired.pages) == 1

    def test_insufficient_high_bytes_is_fixed(self, tmp_dir: Path) -> None:
        """Comment with < 4 high bytes is treated as missing."""
        import pikepdf

        pdf = Pdf.new()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        path = tmp_dir / "test.pdf"
        pdf.save(path)
        pdf.close()

        # Replace binary comment line with one that has only 2 high bytes
        data = path.read_bytes()
        first_nl = data.find(b"\n")
        second_nl = data.find(b"\n", first_nl + 1)
        weak_comment = b"%\xe2\xe3ab\n"
        patched = data[: first_nl + 1] + weak_comment + data[second_nl + 1 :]
        path.write_bytes(patched)

        assert not self._has_binary_comment(path)
        assert _ensure_binary_comment(path, "1.3") is True
        assert self._has_binary_comment(path)

    def test_nonexistent_file_returns_false(self, tmp_dir: Path) -> None:
        """Non-existent file returns False."""
        f = tmp_dir / "missing.pdf"
        assert _ensure_binary_comment(f, "1.7") is False

    def test_integration_converted_pdf_has_binary_comment(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Converted PDF always has a valid binary comment."""
        output = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output, level="2b")
        assert result.success is True
        assert self._has_binary_comment(output)
