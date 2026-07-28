# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for converter.py."""

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
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
    get_pdfa_save_settings,
    save_pdfa,
)
from pdftopdfa.exceptions import ConversionError, OCRError, VeraPDFError
from pdftopdfa.verapdf import VeraPDFResult

_DETECTION_MODEL_DIR = Path("paddle-detection")
_RECOGNITION_MODEL_DIR = Path("paddle-recognition")


def _write_signed_pdf(source_path: Path, signed_path: Path) -> None:
    """Write a copy of source_path with a live digital signature field."""
    with Pdf.open(source_path) as pdf:
        sig_dict = pdf.make_indirect(
            Dictionary(
                Type=Name.Sig,
                Filter=Name("/Adobe.PPKLite"),
                SubFilter=Name("/adbe.pkcs7.detached"),
                ByteRange=Array([0, 100, 200, 300]),
                Contents=pdf.make_stream(b"\x00" * 64),
            )
        )
        sig_field = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                FT=Name.Sig,
                T="Signature1",
                Rect=Array([0, 0, 200, 50]),
                V=sig_dict,
            )
        )
        pdf.pages[0].obj["/Annots"] = Array([sig_field])
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(Fields=Array([sig_field]), SigFlags=1)
        )
        pdf.save(signed_path)


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
        assert result.skipped is False

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

    def test_processing_only_output_name(self, tmp_dir: Path) -> None:
        """Processing-only outputs use a neutral suffix."""
        input_path = tmp_dir / "document.pdf"

        output_path = generate_output_path(input_path, pdfa=False)

        assert output_path == tmp_dir / "document_processed.pdf"


class TestPdfaSaveSettings:
    """Tests for centralized PDF/A save settings."""

    @pytest.mark.parametrize(
        ("level", "expected_version"),
        [("2b", "1.7"), ("2u", "1.7"), ("3b", "1.7"), ("3u", "1.7")],
    )
    def test_get_pdfa_save_settings(self, level: str, expected_version: str) -> None:
        """PDF/A save settings keep the existing final output behavior."""
        settings = get_pdfa_save_settings(level)

        assert settings["force_version"] == expected_version
        assert settings["linearize"] is False
        assert settings["deterministic_id"] is True
        assert settings["preserve_pdfa"] is True
        assert settings["object_stream_mode"] is pikepdf.ObjectStreamMode.preserve

    def test_save_pdfa_runs_hardening_without_optional_verify(
        self, tmp_dir: Path
    ) -> None:
        """save_pdfa runs required hardening and can skip lightweight verify."""
        pdf = Pdf.new()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        output_path = tmp_dir / "output.pdf"

        try:
            with (
                patch("pdftopdfa.converter._ensure_binary_comment") as mock_ensure,
                patch("pdftopdfa.converter._truncate_trailing_data") as mock_truncate,
                patch("pdftopdfa.converter._verify_file_structure") as mock_verify,
            ):
                save_pdfa(pdf, output_path, "2b", verify=False)
        finally:
            pdf.close()

        assert output_path.exists()
        mock_ensure.assert_called_once_with(output_path, "1.7")
        mock_truncate.assert_called_once_with(output_path)
        mock_verify.assert_not_called()


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

    def test_no_pdfa_without_processing_copies_input_unchanged(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """pdfa=False without OCR options creates an unchanged copy."""
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(sample_pdf, output_path, pdfa=False)

        assert result.success is True
        assert result.skipped is True
        assert result.level is None
        assert output_path.read_bytes() == sample_pdf.read_bytes()
        assert any("copied unchanged" in warning for warning in result.warnings)

    def test_no_pdfa_rejects_validation(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """The public API rejects PDF/A validation in processing-only mode."""
        with pytest.raises(ConversionError, match="validation cannot be used"):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "processed.pdf",
                pdfa=False,
                validate=True,
            )

    @pytest.mark.parametrize("api", [convert_files, convert_directory])
    def test_no_pdfa_batch_apis_reject_validation(
        self, api, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Batch APIs reject validation before processing any inputs."""
        args = (
            [(sample_pdf, tmp_dir / "processed.pdf")]
            if api is convert_files
            else [tmp_dir]
        )

        with pytest.raises(ConversionError, match="validation cannot be used"):
            api(*args, pdfa=False, validate=True)

    @patch("pdftopdfa.ocr.is_ocr_available", return_value=False)
    def test_no_pdfa_fails_when_requested_ocr_is_unavailable(
        self,
        mock_is_ocr_available: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Processing-only mode does not silently ignore unavailable OCR."""
        with pytest.raises(OCRError, match="OCR not available"):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "processed.pdf",
                pdfa=False,
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
                ocr_deskew=True,
            )

        mock_is_ocr_available.assert_called_once()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_no_pdfa_force_ocr_implies_ocr(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Forced OCR activates processing without configured languages."""
        processed_bytes = b"%PDF-force-ocr"

        def create_ocr_output(
            input_path: Path,
            output_path: Path,
            languages: list[str],
            **kwargs: object,
        ) -> Path:
            output_path.write_bytes(processed_bytes)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            pdfa=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert output_path.read_bytes() == processed_bytes
        assert mock_apply_ocr.call_args.args[2] == ["en"]
        assert (
            mock_apply_ocr.call_args.kwargs["detection_model_dir"]
            == _DETECTION_MODEL_DIR
        )
        assert (
            mock_apply_ocr.call_args.kwargs["recognition_model_dir"]
            == _RECOGNITION_MODEL_DIR
        )
        assert mock_apply_ocr.call_args.kwargs["force"] is True
        mock_is_ocr_available.assert_called_once()

    @patch("pdftopdfa.converter.save_pdfa")
    @patch("pdftopdfa.converter.embed_color_profiles")
    @patch("pdftopdfa.converter.sync_metadata")
    @patch("pdftopdfa.converter.sanitize_for_pdfa")
    @patch("pdftopdfa.converter.check_font_compliance")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_no_pdfa_saves_ocr_result_without_pdfa_processing(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_detect_pdfa: MagicMock,
        mock_check_fonts: MagicMock,
        mock_sanitize: MagicMock,
        mock_sync_metadata: MagicMock,
        mock_embed_profiles: MagicMock,
        mock_save_pdfa: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR output is copied directly and all PDF/A stages are bypassed."""
        processed_bytes = b"%PDF-processed-by-ocr"

        def create_ocr_output(
            input_path: Path,
            output_path: Path,
            languages: list[str],
            **kwargs: object,
        ) -> Path:
            output_path.write_bytes(processed_bytes)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            pdfa=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_rotate_pages=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert result.level is None
        assert output_path.read_bytes() == processed_bytes
        assert mock_apply_ocr.call_args.args[2] == ["en"]
        assert mock_apply_ocr.call_args.kwargs["rotate_pages"] is True
        mock_is_ocr_available.assert_called_once()
        for pdfa_stage in (
            mock_detect_pdfa,
            mock_check_fonts,
            mock_sanitize,
            mock_sync_metadata,
            mock_embed_profiles,
            mock_save_pdfa,
        ):
            pdfa_stage.assert_not_called()

    def test_convert_uses_central_pdfa_save(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Final PDF/A output is written through save_pdfa."""
        output_path = tmp_dir / "output_pdfa.pdf"

        with patch("pdftopdfa.converter.save_pdfa", wraps=save_pdfa) as mock_save:
            result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args.args[1] == output_path
        assert call_args.args[2] == "2b"
        assert call_args.kwargs["verify"] is True

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
        """Encrypted PDF is copied unchanged and reported as skipped."""
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(encrypted_pdf, output_path)

        assert result.success is True
        assert result.skipped is True
        assert any("encrypted" in w for w in result.warnings)
        assert output_path.exists()
        assert output_path.read_bytes() == encrypted_pdf.read_bytes()

    def test_no_pdfa_encrypted_pdf_is_copied_unchanged(
        self, encrypted_pdf: Path, tmp_dir: Path
    ) -> None:
        """Encrypted inputs retain the existing protection without a level."""
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(encrypted_pdf, output_path, pdfa=False)

        assert result.success is True
        assert result.skipped is True
        assert result.level is None
        assert output_path.read_bytes() == encrypted_pdf.read_bytes()
        assert any("encrypted" in warning for warning in result.warnings)

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
    def test_convert_with_ocr_language_fails_when_unavailable(
        self, mock_is_ocr_available: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Explicit OCR requests fail closed when the dependency is unavailable."""
        mock_is_ocr_available.return_value = False
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(OCRError, match="OCR not available"):
            convert_to_pdfa(
                sample_pdf,
                output_path,
                ocr_languages=["de"],
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            )

        assert not output_path.exists()

    def test_convert_with_directml_reports_directml_extra_when_ocr_unavailable(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The DirectML error must not recommend installing the CPU runtime."""
        output_path = tmp_dir / "output.pdf"

        with (
            patch("pdftopdfa.converter.onnxruntime_engine_config"),
            patch("pdftopdfa.ocr.is_ocr_available", return_value=False),
            pytest.raises(OCRError, match=r"pdftopdfa\[directml\]"),
        ):
            convert_to_pdfa(
                sample_pdf,
                output_path,
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
                ocr_execution_provider="directml",
            )

        assert not output_path.exists()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_languages_parameter(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_languages is passed through to apply_ocr."""
        mock_is_ocr_available.return_value = True

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

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        # Check if apply_ocr was called with the correct languages
        mock_apply_ocr.assert_called_once()
        call_args = mock_apply_ocr.call_args
        assert call_args[0][2] == ["en"]  # Languages parameter
        assert call_args.kwargs["detection_model_dir"] == _DETECTION_MODEL_DIR
        assert call_args.kwargs["recognition_model_dir"] == _RECOGNITION_MODEL_DIR
        assert call_args.kwargs["ocr_execution_provider"] == "cpu"

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_forwards_model_layout_configuration(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The public API forwards the selected layout mode and model path."""
        import shutil

        mock_apply_ocr.side_effect = lambda source, destination, *args, **kwargs: (
            shutil.copy2(source, destination) or destination
        )
        layout_model_dir = Path("layout-model")

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "layout.pdf",
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_layout="model",
            ocr_layout_model_dir=layout_model_dir,
        )

        assert result.success is True
        assert mock_apply_ocr.call_args.kwargs["layout_mode"] == "model"
        assert mock_apply_ocr.call_args.kwargs["layout_model_dir"] == layout_model_dir

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ocr_layout": "invalid"},
            {"ocr_layout": "model"},
            {"ocr_layout_model_dir": Path("layout-model")},
        ],
    )
    def test_convert_rejects_invalid_layout_configuration(
        self,
        kwargs: dict[str, object],
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Invalid or incomplete layout settings fail before conversion."""
        with pytest.raises(ValueError, match="OCR layout|layout model directory"):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "layout.pdf",
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
                **kwargs,
            )

    @patch("pdftopdfa.converter.onnxruntime_engine_config")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_passes_directml_execution_provider(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        _mock_engine_config: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The public API forwards an explicit DirectML selection."""
        import shutil

        mock_apply_ocr.side_effect = lambda source, destination, *args, **kwargs: (
            shutil.copy2(source, destination) or destination
        )

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "output.pdf",
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_execution_provider="directml",
        )

        assert result.success is True
        assert mock_apply_ocr.call_args.kwargs["ocr_execution_provider"] == "directml"

    @pytest.mark.parametrize(
        ("api_option", "apply_option"),
        [("ocr_deskew", "deskew"), ("ocr_rotate_pages", "rotate_pages")],
    )
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_processing_option_enables_ocr_with_default_language(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        api_option: str,
        apply_option: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Each public processing option enables OCR with English."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / f"{api_option}.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            **{api_option: True},
        )

        assert result.success is True
        assert mock_apply_ocr.call_args.args[2] == ["en"]
        assert mock_apply_ocr.call_args.kwargs[apply_option] is True

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_annotation_on_digital_page_does_not_disable_scan_deskew(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Annotation pages reach deskew planning after clean OCR preparation."""
        input_path = tmp_dir / "mixed-annotation.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(100, 100))
            digital_page = pdf.add_blank_page(page_size=(100, 100))
            annotation = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([0, 0, 10, 10]),
                    Contents="Note",
                )
            )
            digital_page.obj[Name.Annots] = Array([annotation])
            pdf.save(input_path)

        seen_clean_annotations = []

        def create_ocr_output(
            source: Path,
            destination: Path,
            *_args: object,
            **_kwargs: object,
        ) -> Path:
            with Pdf.open(source) as pdf:
                seen_clean_annotations.append(
                    [page.obj.get("/Annots") for page in pdf.pages]
                )
            import shutil

            shutil.copy2(source, destination)
            return destination

        mock_apply_ocr.side_effect = create_ocr_output
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(
            input_path,
            output_path,
            pdfa=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_deskew=True,
        )

        assert result.success is True
        assert mock_apply_ocr.call_args.kwargs["deskew"] is True
        assert mock_apply_ocr.call_args.kwargs["_annotated_pages"] == frozenset({2})
        assert seen_clean_annotations == [[None, None]]
        with Pdf.open(output_path) as pdf:
            assert pdf.pages[0].obj.get("/Annots") is None
            assert len(pdf.pages[1].obj.Annots) == 1

    def test_convert_rejects_deskew_with_forced_ocr(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """The public API rejects deskew combined with forced OCR."""
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(OCRError, match="Deskew cannot be combined"):
            convert_to_pdfa(
                sample_pdf,
                output_path,
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
                ocr_deskew=True,
                ocr_force=True,
            )

    @pytest.mark.parametrize("api_name", ["single", "batch", "directory"])
    def test_ocr_model_pair_is_required_before_processing(
        self, api_name: str, tmp_dir: Path
    ) -> None:
        """Every public converter rejects a partial model pair before I/O."""
        input_path = tmp_dir / "missing.pdf"
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(ValueError, match="must be provided together"):
            if api_name == "single":
                convert_to_pdfa(
                    input_path,
                    output_path,
                    ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                )
            elif api_name == "batch":
                convert_files(
                    [(input_path, output_path)],
                    ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                )
            else:
                convert_directory(
                    input_path,
                    ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                    show_progress=False,
                )

        assert not output_path.exists()

    @pytest.mark.parametrize("api_name", ["single", "batch", "directory"])
    def test_invalid_ocr_execution_provider_is_rejected(
        self,
        api_name: str,
        tmp_dir: Path,
    ) -> None:
        """Every public converter rejects unknown execution providers."""
        input_path = tmp_dir / "missing.pdf"
        output_path = tmp_dir / "output.pdf"

        with pytest.raises(ValueError, match="OCR execution provider"):
            if api_name == "single":
                convert_to_pdfa(
                    input_path,
                    output_path,
                    ocr_execution_provider="cuda",
                )
            elif api_name == "batch":
                convert_files(
                    [(input_path, output_path)],
                    ocr_execution_provider="cuda",
                )
            else:
                convert_directory(
                    input_path,
                    ocr_execution_provider="cuda",
                    show_progress=False,
                )

        assert not output_path.exists()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_cleans_ocr_temp_files_when_apply_ocr_raises(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """All OCR temp files are removed when apply_ocr raises.

        This includes the annotation-stripped copy created for PDFs with
        annotations, which is only cleaned up inline on the success path.
        """
        import tempfile

        from pdftopdfa.exceptions import OCRError

        # PDF with an annotation so the annotation-stripped copy is created
        annotated_pdf = tmp_dir / "annotated.pdf"
        with Pdf.open(sample_pdf) as pdf:
            annot = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([10, 10, 30, 30]),
                )
            )
            pdf.pages[0].obj["/Annots"] = Array([annot])
            pdf.save(annotated_pdf)

        mock_is_ocr_available.return_value = True
        mock_apply_ocr.side_effect = OCRError("OCR failed")

        created: list[Path] = []
        real_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(Path(path))
            return fd, path

        output_path = tmp_dir / "output.pdf"
        with patch(
            "pdftopdfa.converter.tempfile.mkstemp", side_effect=tracking_mkstemp
        ):
            with pytest.raises(OCRError):
                convert_to_pdfa(
                    annotated_pdf,
                    output_path,
                    ocr_languages=["en"],
                    ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                    ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
                )

        # OCR output temp + annotation-stripped copy were created ...
        assert len(created) >= 2
        # ... and none of them survived the failed conversion
        leftovers = [path for path in created if path.exists()]
        assert leftovers == []

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_adds_warning_message(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR execution adds warning with language info."""
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
            sample_pdf,
            output_path,
            ocr_languages=["de", "en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        has_ocr_done_warning = any(
            "OCR performed" in w and "de+en" in w for w in result.warnings
        )
        assert has_ocr_done_warning

    @patch("pdftopdfa.converter.sync_metadata")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_passes_original_metadata_snapshot_to_sync_after_ocr(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_sync_metadata: MagicMock,
        pdf_with_metadata: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR runs still hand original metadata to sync_metadata."""
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
            pdf_with_metadata,
            output_path,
            level="2b",
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        mock_sync_metadata.assert_called_once()

        call_kwargs = mock_sync_metadata.call_args.kwargs
        assert call_kwargs["source_info"]["creator"] == "Test Creator"
        assert call_kwargs["source_info"]["producer"] == "Test Producer"
        assert call_kwargs["source_xmp_tree"] is not None

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_runs_ocr_even_for_text_pdf(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """OCR is invoked and lets ocrmypdf skip text pages per page."""
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
            sample_pdf,
            output_path,
            ocr_languages=["de"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        mock_apply_ocr.assert_called_once()
        assert any("OCR performed" in w for w in result.warnings)

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_implies_ocr_with_default_language(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_force=True enables OCR with English when no language is configured."""
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
            sample_pdf,
            output_path,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=True,
        )

        assert result.success is True
        mock_apply_ocr.assert_called_once()
        assert mock_apply_ocr.call_args.args[2] == ["en"]
        call_kwargs = mock_apply_ocr.call_args[1]
        assert call_kwargs["force"] is True

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_bypasses_same_level_skip(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Forced OCR runs even when the input is already compliant."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_detect.return_value = "2b"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="2b")
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="2b",
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert output_path.exists()
        mock_apply_ocr.assert_called_once()
        mock_verapdf.assert_not_called()

    @pytest.mark.parametrize("processing_option", ["ocr_deskew", "ocr_rotate_pages"])
    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_processing_option_bypasses_same_level_skip(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        processing_option: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Explicit page processing runs for already compliant inputs."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_detect.return_value = "2b"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="2b")
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )
        output_path = tmp_dir / f"{processing_option}.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="2b",
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            **{processing_option: True},
        )

        assert result.success is True
        assert result.skipped is False
        mock_apply_ocr.assert_called_once()
        mock_verapdf.assert_not_called()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_bypasses_skip_any_pdfa(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Forced OCR also overrides skip_any_pdfa pre-check skipping."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_detect.return_value = "3a"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="3a")
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="2b",
            skip_any_pdfa=True,
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert output_path.exists()
        mock_apply_ocr.assert_called_once()
        mock_verapdf.assert_not_called()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_ocr_force_false_still_calls_apply_ocr(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """ocr_force=False still delegates page skipping to apply_ocr."""
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
            sample_pdf,
            output_path,
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=False,
        )

        assert result.success is True
        mock_apply_ocr.assert_called_once()
        assert mock_apply_ocr.call_args[1]["force"] is False

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_signed_pdf_is_skipped_without_explicit_invalidation(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Signed PDFs are copied unchanged by default, even when OCR is requested."""
        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            signed_input,
            output_path,
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_force=True,
        )

        assert result.success is True
        assert result.skipped is True
        assert any("digital signatures" in warning for warning in result.warnings)
        assert output_path.read_bytes() == signed_input.read_bytes()
        mock_is_ocr_available.assert_not_called()
        mock_apply_ocr.assert_not_called()

    def test_signed_pdf_can_be_converted_with_explicit_invalidation(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Opt-in conversion removes live signature structures."""
        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            signed_input,
            output_path,
            allow_signature_invalidation=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert any("removed/invalidated" in warning for warning in result.warnings)

        with Pdf.open(output_path) as output_pdf:
            if "/AcroForm" in output_pdf.Root:
                assert output_pdf.Root.AcroForm.Fields[0].get("/V") is None
            for obj in output_pdf.objects:
                if isinstance(obj, Dictionary):
                    assert obj.get("/ByteRange") is None

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_signed_pdf_skip_precedes_pdfa_skip_logic(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A signed PDF/A input is skipped because conversion would invalidate it."""
        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(signed_input, output_path, level="3b")

        assert result.success is True
        assert result.skipped is True
        assert any("digital signatures" in warning for warning in result.warnings)
        assert output_path.read_bytes() == signed_input.read_bytes()
        mock_detect.assert_not_called()
        mock_verapdf.assert_not_called()

    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_no_pdfa_processing_keeps_signed_pdf_protection(
        self,
        mock_is_ocr_available: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Signed inputs are copied unchanged before processing-only OCR."""
        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)
        output_path = tmp_dir / "processed.pdf"

        result = convert_to_pdfa(
            signed_input,
            output_path,
            pdfa=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_rotate_pages=True,
        )

        assert result.success is True
        assert result.skipped is True
        assert result.level is None
        assert output_path.read_bytes() == signed_input.read_bytes()
        assert any("digital signatures" in warning for warning in result.warnings)
        mock_is_ocr_available.assert_not_called()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_no_pdfa_allows_explicit_signature_invalidation(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Processing-only OCR can explicitly remove live signatures."""
        import shutil

        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)
        output_path = tmp_dir / "processed.pdf"
        mock_apply_ocr.side_effect = lambda inp, out, *args, **kwargs: (
            shutil.copy(inp, out) or out
        )

        result = convert_to_pdfa(
            signed_input,
            output_path,
            pdfa=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_rotate_pages=True,
            allow_signature_invalidation=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert result.level is None
        assert any("PDF processing" in warning for warning in result.warnings)
        with Pdf.open(output_path) as output_pdf:
            if "/AcroForm" in output_pdf.Root:
                assert output_pdf.Root.AcroForm.Fields[0].get("/V") is None
        mock_is_ocr_available.assert_called_once()
        mock_apply_ocr.assert_called_once()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_convert_with_ocr_sanitizes_signed_pdf_before_ocr(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Signed PDFs are neutralized before OCR and do not leak live signatures."""
        mock_is_ocr_available.return_value = True

        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)

        def create_ocr_output(
            input_path: Path, output_path: Path, langs: list[str], **kwargs: object
        ) -> Path:
            import shutil

            with Pdf.open(input_path) as prepared_pdf:
                assert "/AcroForm" not in prepared_pdf.Root
                for obj in prepared_pdf.objects:
                    if isinstance(obj, Dictionary):
                        assert obj.get("/ByteRange") is None

            shutil.copy(input_path, output_path)
            return output_path

        mock_apply_ocr.side_effect = create_ocr_output

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            signed_input,
            output_path,
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            allow_signature_invalidation=True,
        )

        assert result.success is True
        assert sum("digital signature" in warning for warning in result.warnings) == 1
        assert any("removed/invalidated" in warning for warning in result.warnings)

        with Pdf.open(output_path) as output_pdf:
            if "/AcroForm" in output_pdf.Root:
                assert output_pdf.Root.AcroForm.Fields[0].get("/V") is None
            for obj in output_pdf.objects:
                if isinstance(obj, Dictionary):
                    assert obj.get("/ByteRange") is None

        mock_apply_ocr.assert_called_once()

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
        assert result.skipped is True
        assert any("already valid" in w for w in result.warnings)
        assert output_path.exists()
        mock_verapdf.assert_called_once_with(
            sample_pdf,
            flavour="2b",
            non_compliant_log_level=logging.WARNING,
        )

    @pytest.mark.parametrize(
        ("detected_level", "target_level"),
        [("1b", "3b"), ("2b", "3u"), ("3u", "2b")],
    )
    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_skip_any_pdfa_skips_any_verapdf_compliant_pdfa(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        detected_level: str,
        target_level: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """skip_any_pdfa skips any veraPDF-compliant PDF/A claim."""
        mock_detect.return_value = detected_level
        mock_verapdf.return_value = VeraPDFResult(
            compliant=True, flavour=detected_level
        )

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level=target_level,
            skip_any_pdfa=True,
        )

        assert result.success is True
        assert result.level == detected_level
        assert result.skipped is True
        assert any("veraPDF compliant" in w for w in result.warnings)
        assert output_path.exists()
        mock_verapdf.assert_called_once_with(
            sample_pdf,
            flavour=detected_level,
            non_compliant_log_level=logging.WARNING,
        )

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_skip_any_pdfa_disabled_keeps_existing_skip_rules(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Without skip_any_pdfa, cross-part PDFs are still converted."""
        mock_detect.return_value = "1b"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="1b")

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="3b")

        assert result.success is True
        assert result.skipped is False
        assert result.level == "3b"
        assert output_path.exists()
        mock_verapdf.assert_not_called()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_skip_any_pdfa_does_not_skip_non_compliant_pdfa(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """skip_any_pdfa converts files when veraPDF reports non-compliance."""
        mock_detect.return_value = "1b"
        mock_verapdf.return_value = VeraPDFResult(compliant=False, flavour="1b")

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="3b",
            skip_any_pdfa=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert result.level == "3b"
        assert output_path.exists()
        mock_verapdf.assert_called_once()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_skip_any_pdfa_does_not_skip_when_verapdf_unavailable(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """skip_any_pdfa falls back to conversion when veraPDF is unavailable."""
        mock_detect.return_value = "1b"
        mock_verapdf.side_effect = VeraPDFError("veraPDF missing")

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="3b",
            skip_any_pdfa=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert result.level == "3b"
        assert output_path.exists()
        mock_verapdf.assert_called_once()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_skip_any_pdfa_ignores_files_without_pdfa_claim(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """skip_any_pdfa does not validate files without a PDF/A claim."""
        mock_detect.return_value = None

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="3b",
            skip_any_pdfa=True,
        )

        assert result.success is True
        assert result.skipped is False
        assert result.level == "3b"
        assert output_path.exists()
        mock_verapdf.assert_not_called()

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

    @patch("pdftopdfa.converter.check_font_compliance")
    @patch("pdftopdfa.fonts.FontEmbedder")
    def test_font_progress_logs_are_debug_only(
        self,
        mock_font_embedder: MagicMock,
        mock_check_font_compliance: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Font progress logs are hidden at INFO and shown at DEBUG."""
        mock_check_font_compliance.return_value = (False, ["Unknown"])

        embedder = MagicMock()
        embedder.__enter__.return_value = embedder
        embedder.__exit__.return_value = None
        embedder.embed_missing_fonts.return_value = SimpleNamespace(
            fonts_embedded=["FrutigerNextLTW1G-Medium", "Unknown"],
            fonts_failed=[],
            warnings=[],
        )
        embedder.add_tounicode_to_embedded_fonts.return_value = SimpleNamespace(
            fonts_embedded=["Arial"],
            fonts_failed=[],
            warnings=[],
        )
        embedder.subset_embedded_fonts.return_value = SimpleNamespace(
            fonts_subsetted=["Arial", "Unknown"],
            bytes_saved=12871064,
            warnings=[],
        )
        mock_font_embedder.return_value = embedder

        info_output = tmp_dir / "info_output.pdf"
        with caplog.at_level(logging.INFO, logger="pdftopdfa.converter"):
            result = convert_to_pdfa(sample_pdf, info_output, level="2b")

        assert result.success is True
        assert not any(
            record.message.startswith("Attempting to embed missing fonts:")
            or record.message.startswith("Fonts embedded:")
            or record.message.startswith("ToUnicode added to fonts:")
            or record.message.startswith("Fonts subsetted:")
            for record in caplog.records
        )

        caplog.clear()

        debug_output = tmp_dir / "debug_output.pdf"
        with caplog.at_level(logging.DEBUG, logger="pdftopdfa.converter"):
            result = convert_to_pdfa(sample_pdf, debug_output, level="2b")

        assert result.success is True
        assert any(
            record.message == "Attempting to embed missing fonts: Unknown"
            for record in caplog.records
        )
        assert any(
            record.message == "Fonts embedded: FrutigerNextLTW1G-Medium, Unknown"
            for record in caplog.records
        )
        assert any(
            record.message == "ToUnicode added to fonts: Arial"
            for record in caplog.records
        )
        assert any(
            record.message == "Fonts subsetted: Arial, Unknown (saved 12871064 bytes)"
            for record in caplog.records
        )

    @patch("pdftopdfa.converter.check_font_compliance")
    @patch("pdftopdfa.fonts.FontEmbedder")
    def test_refreshes_only_original_subsetted_standard14_fonts(
        self,
        mock_font_embedder: MagicMock,
        mock_check_font_compliance: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Only pre-existing subsetted Standard-14 fonts are refreshed."""
        mock_check_font_compliance.return_value = (False, ["Unknown"])

        embedder = MagicMock()
        embedder.__enter__.return_value = embedder
        embedder.__exit__.return_value = None
        embedder.embed_missing_fonts.return_value = SimpleNamespace(
            fonts_embedded=["Unknown"],
            fonts_failed=[],
            warnings=[],
        )
        embedder.add_tounicode_to_embedded_fonts.return_value = SimpleNamespace(
            fonts_embedded=[],
            fonts_failed=[],
            warnings=[],
        )
        embedder.collect_subsetted_standard14_font_ids.return_value = {(99, 0)}
        embedder.subset_embedded_fonts.return_value = SimpleNamespace(
            fonts_subsetted=["Unknown"],
            bytes_saved=1024,
            warnings=[],
        )
        embedder.replace_subsetted_standard14_fonts.return_value = SimpleNamespace(
            fonts_embedded=[],
            fonts_failed=[],
            warnings=[],
        )
        mock_font_embedder.return_value = embedder

        output_path = tmp_dir / "refresh_subsetted.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        embedder.replace_subsetted_standard14_fonts.assert_called_once_with({(99, 0)})

    @patch("pdftopdfa.converter.check_font_compliance")
    @patch("pdftopdfa.fonts.FontEmbedder")
    def test_deduplicates_embedded_font_programs_after_refresh(
        self,
        mock_font_embedder: MagicMock,
        mock_check_font_compliance: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Converter triggers the embedded-font dedupe pass after refresh."""
        mock_check_font_compliance.return_value = (False, ["Unknown"])

        embedder = MagicMock()
        embedder.__enter__.return_value = embedder
        embedder.__exit__.return_value = None
        embedder.embed_missing_fonts.return_value = SimpleNamespace(
            fonts_embedded=["Unknown"],
            fonts_failed=[],
            warnings=[],
        )
        embedder.add_tounicode_to_embedded_fonts.return_value = SimpleNamespace(
            fonts_embedded=[],
            fonts_failed=[],
            warnings=[],
        )
        embedder.collect_subsetted_standard14_font_ids.return_value = set()
        embedder.subset_embedded_fonts.return_value = SimpleNamespace(
            fonts_subsetted=[],
            bytes_saved=0,
            warnings=[],
        )
        embedder.replace_subsetted_standard14_fonts.return_value = SimpleNamespace(
            fonts_embedded=[],
            fonts_failed=[],
            warnings=[],
        )
        embedder.deduplicate_embedded_font_programs.return_value = SimpleNamespace(
            programs_deduplicated=3,
            bytes_saved_estimate=123456,
        )
        mock_font_embedder.return_value = embedder

        output_path = tmp_dir / "dedupe_fonts.pdf"
        with caplog.at_level(logging.DEBUG, logger="pdftopdfa.converter"):
            result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        embedder.deduplicate_embedded_font_programs.assert_called_once()
        assert any(
            "Deduplicated 3 embedded font program(s) (saved ~123456 bytes)"
            == record.message
            for record in caplog.records
        )

    @patch("pdftopdfa.converter.detect_iso_standards")
    def test_iso_standard_logs_are_debug_only(
        self,
        mock_detect_iso_standards: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ISO standard detection logs are hidden at INFO and shown at DEBUG."""
        mock_detect_iso_standards.return_value = [
            SimpleNamespace(standard="PDF/X", version="4")
        ]

        info_output = tmp_dir / "iso_info_output.pdf"
        with caplog.at_level(logging.INFO, logger="pdftopdfa.converter"):
            result = convert_to_pdfa(sample_pdf, info_output, level="2b")

        assert result.success is True
        assert "ISO standard detected: PDF/X 4" in result.warnings
        assert not any(
            record.message == "ISO standard detected: PDF/X 4"
            for record in caplog.records
        )

        caplog.clear()

        debug_output = tmp_dir / "iso_debug_output.pdf"
        with caplog.at_level(logging.DEBUG, logger="pdftopdfa.converter"):
            result = convert_to_pdfa(sample_pdf, debug_output, level="2b")

        assert result.success is True
        assert any(
            record.message == "ISO standard detected: PDF/X 4"
            for record in caplog.records
        )


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

    @pytest.mark.parametrize("recursive", [False, True])
    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_matches_pdf_suffix_case_insensitively(
        self,
        mock_convert_files: MagicMock,
        recursive: bool,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """PDF suffix matching is case-insensitive in both search modes."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        nested_dir = input_dir / "nested"
        nested_dir.mkdir()

        top_level_pdfs = {
            input_dir / "lower.pdf",
            input_dir / "upper.PDF",
            input_dir / "mixed.PdF",
        }
        nested_pdfs = {
            nested_dir / "nested_lower.pdf",
            nested_dir / "nested_upper.PDF",
            nested_dir / "nested_mixed.pDf",
        }
        for path in top_level_pdfs | nested_pdfs:
            path.write_bytes(sample_pdf_bytes)

        mock_convert_files.return_value = []

        convert_directory(input_dir, recursive=recursive, show_progress=False)

        file_pairs = mock_convert_files.call_args.kwargs["file_pairs"]
        actual_inputs = {input_path for input_path, _ in file_pairs}
        expected_inputs = top_level_pdfs | nested_pdfs if recursive else top_level_pdfs
        assert actual_inputs == expected_inputs

    @pytest.mark.parametrize("recursive", [False, True])
    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_excludes_non_pdf_candidates(
        self,
        mock_convert_files: MagicMock,
        recursive: bool,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Directories and files with similar suffixes are not processed."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        nested_dir = input_dir / "nested"
        nested_dir.mkdir()

        expected_pdf = input_dir / "document.PDF"
        expected_pdf.write_bytes(sample_pdf_bytes)
        (input_dir / "directory.pdf").mkdir()
        for name in ("document.pdfx", "document.pdf.backup", "document.pd", "pdf"):
            (input_dir / name).write_bytes(sample_pdf_bytes)
        (nested_dir / "nested.txt").write_bytes(sample_pdf_bytes)

        mock_convert_files.return_value = []

        convert_directory(input_dir, recursive=recursive, show_progress=False)

        file_pairs = mock_convert_files.call_args.kwargs["file_pairs"]
        assert [input_path for input_path, _ in file_pairs] == [expected_pdf]

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

    @patch("pdftopdfa.converter.convert_files")
    def test_recursive_nested_empty_output_directory(
        self, mock_convert_files: MagicMock, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """An empty nested output directory works on the first run."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = input_dir / "output"
        output_dir.mkdir()
        source = input_dir / "document.pdf"
        source.write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=True,
            show_progress=False,
        )

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, output_dir / "document_pdfa.pdf")
        ]

    @patch("pdftopdfa.converter.convert_files")
    def test_recursive_excludes_nested_output_tree(
        self, mock_convert_files: MagicMock, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """PDFs anywhere below a nested output directory are not inputs."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = input_dir / "export"
        nested_output_dir = output_dir / "archive"
        nested_output_dir.mkdir(parents=True)
        source = input_dir / "document.pdf"
        source.write_bytes(sample_pdf_bytes)
        (output_dir / "old.pdf").write_bytes(sample_pdf_bytes)
        (nested_output_dir / "old_pdfa.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=True,
            show_progress=False,
        )

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, output_dir / "document_pdfa.pdf")
        ]

    def test_recursive_nested_output_is_stable_across_runs(
        self, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Repeated runs do not create output/output paths."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = input_dir / "output"
        (input_dir / "document.pdf").write_bytes(sample_pdf_bytes)

        first_results = convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=True,
            show_progress=False,
        )
        second_results = convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=True,
            show_progress=False,
        )

        assert len(first_results) == 1
        assert len(second_results) == 1
        assert not (output_dir / "output").exists()

    @pytest.mark.parametrize("output_location", ["same", "outside"])
    @patch("pdftopdfa.converter.convert_files")
    def test_recursive_non_nested_output_behavior_is_unchanged(
        self,
        mock_convert_files: MagicMock,
        output_location: str,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Same-directory and external output locations retain their behavior."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        source = input_dir / "document.pdf"
        previous_output = input_dir / "previous_pdfa.pdf"
        source.write_bytes(sample_pdf_bytes)
        previous_output.write_bytes(sample_pdf_bytes)
        output_dir = input_dir if output_location == "same" else tmp_dir / "output"
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=True,
            show_progress=False,
        )

        file_pairs = mock_convert_files.call_args.kwargs["file_pairs"]
        expected_inputs = (
            [source] if output_location == "same" else [source, previous_output]
        )
        assert [input_path for input_path, _ in file_pairs] == expected_inputs

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

    @patch("pdftopdfa.converter.convert_files")
    def test_processing_directory_uses_suffix_and_skips_previous_outputs(
        self, mock_convert_files: MagicMock, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """Processing-only directory mode uses and excludes _processed outputs."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        source = input_dir / "doc.pdf"
        source.write_bytes(sample_pdf_bytes)
        (input_dir / "old_processed.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(input_dir, pdfa=False, show_progress=False)

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, input_dir / "doc_processed.pdf")
        ]
        assert mock_convert_files.call_args.kwargs["pdfa"] is False

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
            input_dir,
            show_progress=False,
            ocr_languages=["de"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "OCR not available" in (results[0].error or "")

    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_passes_skip_any_pdfa(
        self, mock_convert_files: MagicMock, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """convert_directory forwards skip_any_pdfa to convert_files."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(input_dir, show_progress=False, skip_any_pdfa=True)

        assert mock_convert_files.call_args.kwargs["skip_any_pdfa"] is True

    @patch("pdftopdfa.converter.onnxruntime_engine_config")
    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_passes_processing_options(
        self,
        mock_convert_files: MagicMock,
        _mock_engine_config: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """convert_directory forwards both independent processing options."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            show_progress=False,
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_deskew=True,
            ocr_rotate_pages=True,
            ocr_execution_provider="directml",
        )

        kwargs = mock_convert_files.call_args.kwargs
        assert kwargs["ocr_detection_model_dir"] == _DETECTION_MODEL_DIR
        assert kwargs["ocr_recognition_model_dir"] == _RECOGNITION_MODEL_DIR
        assert kwargs["ocr_deskew"] is True
        assert kwargs["ocr_rotate_pages"] is True
        assert kwargs["ocr_execution_provider"] == "directml"

    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_passes_allow_signature_invalidation(
        self, mock_convert_files: MagicMock, tmp_dir: Path, sample_pdf_bytes: bytes
    ) -> None:
        """convert_directory forwards allow_signature_invalidation."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            show_progress=False,
            allow_signature_invalidation=True,
        )

        assert (
            mock_convert_files.call_args.kwargs["allow_signature_invalidation"] is True
        )


class TestConvertFiles:
    """Tests for convert_files."""

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_passes_no_pdfa(
        self, mock_convert_to_pdfa: MagicMock, tmp_dir: Path
    ) -> None:
        """File-list processing forwards processing-only mode."""
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=input_path,
            output_path=output_path,
            level=None,
        )

        results = convert_files([(input_path, output_path)], pdfa=False)

        assert results[0].level is None
        assert mock_convert_to_pdfa.call_args.kwargs["pdfa"] is False

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

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_passes_skip_any_pdfa(
        self, mock_convert_to_pdfa: MagicMock, tmp_dir: Path
    ) -> None:
        """convert_files forwards skip_any_pdfa to convert_to_pdfa."""
        in_path = tmp_dir / "test.pdf"
        out_path = tmp_dir / "test_pdfa.pdf"
        in_path.write_bytes(b"%PDF-1.4 dummy")
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=in_path,
            output_path=out_path,
            level="3b",
        )

        convert_files([(in_path, out_path)], skip_any_pdfa=True)

        assert mock_convert_to_pdfa.call_args.kwargs["skip_any_pdfa"] is True

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_passes_allow_signature_invalidation(
        self, mock_convert_to_pdfa: MagicMock, tmp_dir: Path
    ) -> None:
        """convert_files forwards allow_signature_invalidation to convert_to_pdfa."""
        in_path = tmp_dir / "test.pdf"
        out_path = tmp_dir / "test_pdfa.pdf"
        in_path.write_bytes(b"%PDF-1.4 dummy")
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=in_path,
            output_path=out_path,
            level="3b",
        )

        convert_files([(in_path, out_path)], allow_signature_invalidation=True)

        assert (
            mock_convert_to_pdfa.call_args.kwargs["allow_signature_invalidation"]
            is True
        )

    @patch("pdftopdfa.converter.onnxruntime_engine_config")
    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_passes_processing_options(
        self,
        mock_convert_to_pdfa: MagicMock,
        _mock_engine_config: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """convert_files forwards both independent processing options."""
        in_path = tmp_dir / "test.pdf"
        out_path = tmp_dir / "test_pdfa.pdf"
        in_path.write_bytes(b"%PDF-1.4 dummy")
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=in_path,
            output_path=out_path,
            level="3b",
        )

        convert_files(
            [(in_path, out_path)],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_deskew=True,
            ocr_rotate_pages=True,
            ocr_execution_provider="directml",
        )

        kwargs = mock_convert_to_pdfa.call_args.kwargs
        assert kwargs["ocr_detection_model_dir"] == _DETECTION_MODEL_DIR
        assert kwargs["ocr_recognition_model_dir"] == _RECOGNITION_MODEL_DIR
        assert kwargs["ocr_deskew"] is True
        assert kwargs["ocr_rotate_pages"] is True
        assert kwargs["ocr_execution_provider"] == "directml"

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


class TestStripAnnotationsForOcr:
    """Tests for _strip_annotations_for_ocr."""

    def test_removes_acroform_from_root(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """/AcroForm is stripped from the document root."""
        from pdftopdfa.converter import _strip_annotations_for_ocr

        input_pdf = tmp_dir / "acroform.pdf"
        with Pdf.open(sample_pdf) as pdf:
            pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([])))
            pdf.save(input_pdf)

        clean_pdf = tmp_dir / "clean.pdf"
        removed = _strip_annotations_for_ocr(input_pdf, clean_pdf)

        assert removed is True
        with Pdf.open(clean_pdf) as pdf:
            assert "/AcroForm" not in pdf.Root

    def test_removes_page_annotations(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """/Annots arrays are stripped from all pages."""
        from pdftopdfa.converter import _strip_annotations_for_ocr

        input_pdf = tmp_dir / "annotated.pdf"
        with Pdf.open(sample_pdf) as pdf:
            annot = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([10, 10, 30, 30]),
                )
            )
            pdf.pages[0].obj["/Annots"] = Array([annot])
            pdf.save(input_pdf)

        clean_pdf = tmp_dir / "clean.pdf"
        removed = _strip_annotations_for_ocr(input_pdf, clean_pdf)

        assert removed is True
        with Pdf.open(clean_pdf) as pdf:
            assert pdf.pages[0].get("/Annots") is None
