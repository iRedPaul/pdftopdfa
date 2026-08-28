# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for converter.py."""

import ctypes
import errno
import logging
import os
import subprocess
import sys
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
from pdftopdfa.exceptions import (
    ConversionError,
    OCRError,
    UnsupportedPDFError,
    VeraPDFError,
)
from pdftopdfa.metadata import NAMESPACES, embed_xmp_metadata
from pdftopdfa.staging import publish_staged_file as publish_staged_file_impl
from pdftopdfa.staging import staged_file_snapshot
from pdftopdfa.tagging import ensure_logical_structure
from pdftopdfa.utils import resolve_indirect
from pdftopdfa.validator import detect_iso_standards
from pdftopdfa.verapdf import (
    VeraPDFResult,
    is_verapdf_available,
    validate_with_verapdf,
)

_DETECTION_MODEL_DIR = Path("paddle-detection")
_RECOGNITION_MODEL_DIR = Path("paddle-recognition")


def _windows_dacl_sddl(path: Path) -> str:
    """Return a Windows file DACL as SDDL for inheritance assertions."""

    from ctypes import wintypes

    security_descriptor = ctypes.c_void_p()
    sddl = wintypes.LPWSTR()
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_security.restype = wintypes.DWORD
    result = get_security(
        str(path),
        1,
        0x00000004,
        None,
        None,
        None,
        None,
        ctypes.byref(security_descriptor),
    )
    if result:
        raise ctypes.WinError(result)
    convert = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.c_void_p,
    ]
    convert.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    try:
        if not convert(
            security_descriptor,
            1,
            0x00000004,
            ctypes.byref(sddl),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return sddl.value
    finally:
        if sddl:
            local_free(sddl)
        local_free(security_descriptor)


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
        [
            ("2a", "1.7"),
            ("2b", "1.7"),
            ("2u", "1.7"),
            ("3a", "1.7"),
            ("3b", "1.7"),
            ("3u", "1.7"),
        ],
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

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_rejects_document_without_pages(self, level: str, tmp_dir: Path) -> None:
        """A page-less input cannot produce a valid PDF/A document."""
        input_path = tmp_dir / "empty.pdf"
        output_path = tmp_dir / f"empty-{level}.pdf"
        with Pdf.new() as pdf:
            pdf.save(input_path)

        with pytest.raises(UnsupportedPDFError, match="contains no pages"):
            convert_to_pdfa(input_path, output_path, level=level)

        assert not output_path.exists()

    def test_convert_simple_pdf(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """Simple conversion with success check."""
        output_path = tmp_dir / "output_pdfa.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.success is True
        assert result.input_path == sample_pdf
        assert result.output_path == output_path
        assert result.level == "2b"
        assert output_path.exists()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode regression")
    def test_new_pdfa_output_uses_normal_creation_mode(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        control_path = tmp_dir / "control"
        control_path.touch()
        output_path = tmp_dir / "output_pdfa.pdf"

        convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.stat().st_mode & 0o7777 == (
            control_path.stat().st_mode & 0o7777
        )

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

    @pytest.mark.parametrize(
        ("branch", "options"),
        [
            ("signed", {"validate": True}),
            ("no-processing", {"pdfa": False}),
        ],
    )
    def test_early_copy_paths_close_check_pdf_before_callback(
        self,
        branch: str,
        options: dict[str, bool],
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Early copy callbacks run after releasing the input PDF."""
        closed = False

        def detect_encryption(check_pdf: pikepdf.Pdf) -> bool:
            original_close = check_pdf.close

            def tracked_close() -> None:
                nonlocal closed
                closed = True
                original_close()

            check_pdf.close = tracked_close
            return False

        def early_callback(*_args: object, **_kwargs: object):
            assert closed
            return None

        with (
            patch(
                "pdftopdfa.converter.is_pdf_encrypted",
                side_effect=detect_encryption,
            ),
            patch(
                "pdftopdfa.converter.count_digital_signatures",
                return_value=1 if branch == "signed" else 0,
            ),
            patch(
                "pdftopdfa.converter._copy_input_to_output",
                side_effect=early_callback,
            ) as mock_callback,
        ):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "output.pdf",
                **options,
            )

        mock_callback.assert_called_once()

    def test_no_processing_partial_copy_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A failed unchanged-copy operation cannot truncate its destination."""
        output_path = tmp_dir / "processed.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        def partial_copy(_source: object, staged: object) -> None:
            staged.write(b"partial copy")
            raise PermissionError("copy interrupted")

        with patch(
            "pdftopdfa.staging.shutil.copyfileobj",
            side_effect=partial_copy,
        ):
            with pytest.raises(PermissionError, match="copy interrupted"):
                convert_to_pdfa(sample_pdf, output_path, pdfa=False)

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".processed_copy_*"))

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL regression")
    @pytest.mark.parametrize("existing_output", [False, True])
    def test_unchanged_copy_uses_destination_windows_acl(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        existing_output: bool,
    ) -> None:
        destination_directory = tmp_dir / "shared"
        destination_directory.mkdir()
        subprocess.run(
            [
                "icacls",
                str(destination_directory),
                "/grant",
                "*S-1-1-0:(OI)(CI)(RX)",
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        output_path = destination_directory / "output.pdf"
        expected_dacl = None
        if existing_output:
            output_path.write_bytes(b"existing output")
            subprocess.run(
                [
                    "icacls",
                    str(output_path),
                    "/grant",
                    "*S-1-1-0:(RX)",
                ],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
            expected_dacl = _windows_dacl_sddl(output_path)
            assert ";;;WD)" in expected_dacl

        result = convert_to_pdfa(sample_pdf, output_path, pdfa=False)

        assert result.success is True
        assert output_path.read_bytes() == sample_pdf.read_bytes()
        published_dacl = _windows_dacl_sddl(output_path)
        assert ";;;WD)" in published_dacl
        if expected_dacl is not None:
            assert published_dacl.replace("D:AI", "D:", 1) == (
                expected_dacl.replace("D:AI", "D:", 1)
            )
        else:
            world_ace = next(
                ace for ace in published_dacl.split("(") if ace.endswith(";;;WD)")
            )
            assert ";ID;" in world_ace

    @pytest.mark.skipif(os.name == "nt", reason="POSIX metadata regression")
    def test_unchanged_copy_preserves_existing_posix_metadata(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        output_path.write_bytes(b"existing output")
        output_path.chmod(0o640)
        original = output_path.stat()

        result = convert_to_pdfa(sample_pdf, output_path, pdfa=False)

        published = output_path.stat()
        assert result.success is True
        assert published.st_mode & 0o7777 == original.st_mode & 0o7777
        assert (published.st_uid, published.st_gid) == (
            original.st_uid,
            original.st_gid,
        )

    def test_no_processing_in_place_skip_preserves_input(
        self, sample_pdf: Path
    ) -> None:
        """An unchanged in-place request is a safe no-op."""
        expected = sample_pdf.read_bytes()

        result = convert_to_pdfa(sample_pdf, sample_pdf, pdfa=False)

        assert result.skipped is True
        assert sample_pdf.read_bytes() == expected
        assert not list(sample_pdf.parent.glob(f".{sample_pdf.stem}_copy_*.pdf"))

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

    def test_no_pdfa_batch_apis_ignore_level_for_empty_inputs(
        self, tmp_dir: Path
    ) -> None:
        """Processing-only batch APIs do not apply the PDF/A level option."""
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()

        assert convert_files([], pdfa=False, level="invalid") == []
        assert (
            convert_directory(
                empty_dir,
                pdfa=False,
                level="invalid",
                show_progress=False,
            )
            == []
        )

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
        staged_path = call_args.args[1]
        assert staged_path != output_path
        assert staged_path.parent.parent == output_path.parent
        assert staged_path.parent.name.startswith(f".{output_path.stem}_pdfa_stage_")
        assert not staged_path.exists()
        assert call_args.args[2] == "2b"
        assert call_args.kwargs["verify"] is True

    def test_partial_pdf_save_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A partial final save never replaces an existing destination."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        def partial_save(
            _pdf: pikepdf.Pdf, staged_path: Path, **_kwargs: object
        ) -> None:
            Path(staged_path).write_bytes(b"partial PDF")
            raise PermissionError("disk full")

        with patch.object(pikepdf.Pdf, "save", partial_save):
            with pytest.raises(PermissionError, match="disk full"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    @pytest.mark.parametrize(
        "failure_target",
        [
            "pdftopdfa.converter._ensure_binary_comment",
            "pdftopdfa.converter._verify_file_structure",
        ],
    )
    def test_final_hardening_failure_preserves_existing_output(
        self,
        failure_target: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Hardening and reopen failures affect only the staged output."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        with patch(failure_target, side_effect=RuntimeError("hardening failed")):
            with pytest.raises(ConversionError, match="hardening failed"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_final_header_read_failure_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A real staged-header read failure prevents publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        real_open = open

        def fail_staged_header(file: object, mode: str = "r", *args, **kwargs):
            path = Path(file) if isinstance(file, (str, os.PathLike)) else None
            if (
                path is not None
                and path.name.startswith(".output_pdfa_")
                and "rb" in mode
            ):
                raise OSError("staged header unreadable")
            return real_open(file, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=fail_staged_header):
            with pytest.raises(ConversionError, match="binary comment check"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_final_reopen_failure_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A real staged-PDF reopen failure prevents publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        real_open = pikepdf.open

        def fail_staged_reopen(file: object, *args, **kwargs):
            path = Path(file) if isinstance(file, (str, os.PathLike)) else None
            if path is not None and path.name.startswith(".output_pdfa_"):
                raise OSError("staged PDF cannot be reopened")
            return real_open(file, *args, **kwargs)

        with patch("pdftopdfa.converter.pikepdf.open", side_effect=fail_staged_reopen):
            with pytest.raises(ConversionError, match="could not reopen"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_final_header_mismatch_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A saved PDF with the wrong header version is never published."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        wrong_version_settings = {
            "linearize": False,
            "force_version": "1.6",
            "deterministic_id": True,
            "preserve_pdfa": True,
            "object_stream_mode": pikepdf.ObjectStreamMode.preserve,
        }

        with patch(
            "pdftopdfa.converter._get_pdfa_save_settings_for_version",
            return_value=wrong_version_settings,
        ):
            with pytest.raises(ConversionError, match="does not start with"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    @pytest.mark.parametrize("invalid_id", [None, "wrong-type"])
    def test_final_invalid_trailer_id_preserves_existing_output(
        self,
        invalid_id: str | None,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A missing or malformed trailer ID prevents publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        real_open = pikepdf.open

        fake_pdf = MagicMock()
        fake_pdf.trailer.get.return_value = (
            None
            if invalid_id is None
            else Array([Name("/NotAString"), pikepdf.String(b"second")])
        )
        fake_context = MagicMock()
        fake_context.__enter__.return_value = fake_pdf

        def reopen_with_invalid_id(file: object, *args, **kwargs):
            path = Path(file) if isinstance(file, (str, os.PathLike)) else None
            if path is not None and path.name.startswith(".output_pdfa_"):
                return fake_context
            return real_open(file, *args, **kwargs)

        with patch(
            "pdftopdfa.converter.pikepdf.open",
            side_effect=reopen_with_invalid_id,
        ):
            with pytest.raises(ConversionError, match="trailer /ID"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_failed_binary_comment_rewrite_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A required but failed binary-comment rewrite blocks publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        original_save = pikepdf.Pdf.save
        final_saved = False

        def save_without_comment_then_fail(
            pdf: pikepdf.Pdf,
            path: Path,
            *args,
            **kwargs,
        ) -> None:
            nonlocal final_saved
            path = Path(path)
            if path.name.startswith(".output_pdfa_"):
                original_save(pdf, path, *args, **kwargs)
                data = path.read_bytes()
                first_newline = data.find(b"\n")
                second_newline = data.find(b"\n", first_newline + 1)
                path.write_bytes(data[: first_newline + 1] + data[second_newline + 1 :])
                final_saved = True
                return
            if final_saved and path.parent.parent == tmp_dir:
                raise OSError("binary-comment rewrite failed")
            original_save(pdf, path, *args, **kwargs)

        with patch.object(pikepdf.Pdf, "save", save_without_comment_then_fail):
            with pytest.raises(ConversionError, match="Could not add binary comment"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_failed_eof_hardening_preserves_existing_output(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """A failed required EOF rewrite blocks publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        original_save = pikepdf.Pdf.save
        original_write_bytes = Path.write_bytes

        def save_with_trailing_data(
            pdf: pikepdf.Pdf,
            path: Path,
            *args,
            **kwargs,
        ) -> None:
            path = Path(path)
            original_save(pdf, path, *args, **kwargs)
            if path.name.startswith(".output_pdfa_"):
                with open(path, "ab") as output:
                    output.write(b"trailing data")

        def fail_staged_rewrite(path: Path, data: bytes) -> int:
            if path.name.startswith(".output_pdfa_"):
                raise OSError("EOF rewrite failed")
            return original_write_bytes(path, data)

        with (
            patch.object(pikepdf.Pdf, "save", save_with_trailing_data),
            patch.object(Path, "write_bytes", fail_staged_rewrite),
        ):
            with pytest.raises(ConversionError, match="Could not truncate"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))

    def test_final_cleanup_failure_does_not_mask_primary_error(
        self, sample_pdf: Path, tmp_dir: Path, caplog
    ) -> None:
        """Staging cleanup warnings preserve the original conversion failure."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        original_unlink = Path.unlink

        def fail_staged_cleanup(path: Path, *args, **kwargs) -> None:
            if path.name.startswith(".output_pdfa_"):
                raise OSError("cleanup failed")
            original_unlink(path, *args, **kwargs)

        with (
            patch("pdftopdfa.converter.save_pdfa", side_effect=RuntimeError("primary")),
            patch.object(Path, "unlink", fail_staged_cleanup),
            caplog.at_level(logging.WARNING),
        ):
            with pytest.raises(ConversionError, match="primary"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_pdfa_stage_*"))
        assert any(
            "Could not delete staged PDF/A output" in r.message for r in caplog.records
        )

    @patch("pdftopdfa.converter.save_pdfa")
    def test_success_atomically_replaces_existing_output_once(
        self,
        mock_save: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A successful pipeline publishes its staged PDF exactly once."""
        output_path = tmp_dir / "output.pdf"
        output_path.write_bytes(b"existing output")
        replacement = b"complete staged PDF"

        def write_staged_pdf(
            _pdf: pikepdf.Pdf,
            staged_path: Path,
            _level: str,
            *,
            verify: bool,
        ) -> None:
            assert verify is True
            staged_path.write_bytes(replacement)

        mock_save.side_effect = write_staged_pdf
        with patch(
            "pdftopdfa.converter.publish_staged_file",
            wraps=publish_staged_file_impl,
        ) as mock_publish:
            result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        staged_path = mock_save.call_args.args[1]
        assert result.success is True
        assert output_path.read_bytes() == replacement
        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[:2] == (staged_path, output_path)
        assert not staged_path.exists()

    @patch("pdftopdfa.converter.save_pdfa")
    def test_validated_final_output_cannot_be_swapped_before_publish(
        self,
        mock_save: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        def write_staged_pdf(
            _pdf: pikepdf.Pdf,
            staged_path: Path,
            _level: str,
            *,
            verify: bool,
        ) -> None:
            assert verify is True
            staged_path.write_bytes(b"validated candidate")

        def swap_validated_path(staged_path: Path, *_args: object) -> bool:
            replacement = Path(staged_path).with_name("replacement.pdf")
            replacement.write_bytes(b"different bytes")
            os.replace(replacement, staged_path)
            return False

        mock_save.side_effect = write_staged_pdf
        with (
            patch(
                "pdftopdfa.converter._validate_pdfa_output",
                side_effect=swap_validated_path,
            ),
            pytest.raises(ConversionError, match="changed after validation"),
        ):
            convert_to_pdfa(
                sample_pdf,
                output_path,
                level="2a",
                validate=True,
            )

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".output_pdfa_stage_*"))

    @patch("pdftopdfa.converter.save_pdfa")
    def test_failed_atomic_publication_preserves_existing_output(
        self,
        mock_save: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A final atomic-replace failure preserves the destination and cleans up."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        def write_staged_pdf(
            _pdf: pikepdf.Pdf,
            staged_path: Path,
            _level: str,
            *,
            verify: bool,
        ) -> None:
            assert verify is True
            staged_path.write_bytes(b"complete staged PDF")

        mock_save.side_effect = write_staged_pdf
        with patch(
            "pdftopdfa.converter.publish_staged_file",
            side_effect=PermissionError("destination is locked"),
        ):
            with pytest.raises(PermissionError, match="destination is locked"):
                convert_to_pdfa(sample_pdf, output_path, level="2b")

        staged_path = mock_save.call_args.args[1]
        assert output_path.read_bytes() == sentinel
        assert not staged_path.exists()

    @pytest.mark.parametrize("existing_output", [False, True])
    def test_post_publication_inspection_failure_is_rolled_back(
        self,
        existing_output: bool,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A post-replace failure restores the old target or removes the new one."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        original = None
        if existing_output:
            output_path.write_bytes(sentinel)
            os.utime(output_path, ns=(1_700_000_000_000_000_000,) * 2)
            original = output_path.stat()

        real_snapshot = staged_file_snapshot

        def fail_published_target(path: Path):
            if Path(path) == output_path:
                raise ConversionError("post-publication inspection failed")
            return real_snapshot(Path(path))

        with (
            patch(
                "pdftopdfa.staging.staged_file_snapshot",
                side_effect=fail_published_target,
            ),
            pytest.raises(ConversionError, match="post-publication inspection"),
        ):
            convert_to_pdfa(sample_pdf, output_path, level="2b")

        if original is None:
            assert not output_path.exists()
        else:
            restored = output_path.stat()
            assert output_path.read_bytes() == sentinel
            assert restored.st_dev == original.st_dev
            assert restored.st_ino == original.st_ino
            assert restored.st_size == original.st_size
            assert restored.st_mtime_ns == original.st_mtime_ns
        assert not list(tmp_dir.glob(".output_pdfa_stage_*"))

    def test_changed_published_candidate_is_rolled_back_by_identity(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Candidate mutation cannot prevent restoration of the retained target."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        original = output_path.stat()
        real_snapshot = staged_file_snapshot
        target_checks = 0

        def mutate_published_target(path: Path):
            nonlocal target_checks
            path = Path(path)
            if path == output_path:
                target_checks += 1
                if target_checks == 1:
                    path.write_bytes(b"changed after publication")
            return real_snapshot(path)

        with (
            patch(
                "pdftopdfa.staging.staged_file_snapshot",
                side_effect=mutate_published_target,
            ),
            pytest.raises(ConversionError, match="differs from validated candidate"),
        ):
            convert_to_pdfa(sample_pdf, output_path, level="2b")

        restored = output_path.stat()
        assert output_path.read_bytes() == sentinel
        assert restored.st_dev == original.st_dev
        assert restored.st_ino == original.st_ino
        assert restored.st_size == original.st_size
        assert restored.st_mtime_ns == original.st_mtime_ns
        assert not list(tmp_dir.glob(".output_pdfa_stage_*"))

    def test_foreign_replacement_after_publication_is_not_removed(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Rollback leaves a concurrently replaced publication target untouched."""
        output_path = tmp_dir / "output.pdf"
        output_path.write_bytes(b"existing output")
        foreign_path = tmp_dir / "foreign.pdf"
        foreign = b"concurrent replacement"
        foreign_path.write_bytes(foreign)
        real_snapshot = staged_file_snapshot
        replaced = False

        def replace_published_target(path: Path):
            nonlocal replaced
            path = Path(path)
            if path == output_path and not replaced:
                replaced = True
                os.replace(foreign_path, output_path)
            return real_snapshot(path)

        with (
            patch(
                "pdftopdfa.staging.staged_file_snapshot",
                side_effect=replace_published_target,
            ),
            pytest.raises(ConversionError, match="differs from validated candidate"),
        ):
            convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert output_path.read_bytes() == foreign
        assert not foreign_path.exists()
        assert not list(tmp_dir.glob(".output_pdfa_stage_*"))

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
        """An encrypted PDF with an empty user password is converted."""
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(encrypted_pdf, output_path)

        assert result.success is True
        assert result.skipped is False
        assert result.level == "3b"
        assert "Encryption removed for PDF/A compliance" in result.warnings
        assert output_path.exists()
        assert output_path.read_bytes() != encrypted_pdf.read_bytes()

        with Pdf.open(output_path) as pdf:
            assert pdf.is_encrypted is False
            metadata = pdf.Root["/Metadata"]
            assert metadata["/Type"] == Name.Metadata
            assert metadata["/Subtype"] == Name.XML
            output_intent = pdf.Root["/OutputIntents"][0]
            assert output_intent["/DestOutputProfile"]["/N"] == 3

    @pytest.mark.parametrize("pdfa", [True, False])
    def test_password_encrypted_pdf_is_copied_unchanged(
        self,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
        pdfa: bool,
    ) -> None:
        """A non-empty user password does not turn a safe skip into failure."""
        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            password_encrypted_pdf,
            output_path,
            level="3a",
            pdfa=pdfa,
        )

        assert result.success is True
        assert result.skipped is True
        assert result.level is None
        assert any("encrypted" in warning for warning in result.warnings)
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()

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

    @pytest.mark.parametrize("level", ["2a", "2b", "2u", "3a", "3b", "3u"])
    def test_convert_all_levels(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Conversion works for all PDF/A levels."""
        output_path = tmp_dir / f"output_{level}.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level=level)

        assert result.success is True
        assert result.level == level
        assert output_path.exists()

    @pytest.mark.parametrize("level", ["2b", "2u", "3b", "3u"])
    def test_pdfua_requires_level_a(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """PDF/UA-1 is only emitted with PDF/A-2a or PDF/A-3a."""
        with pytest.raises(ConversionError, match="PDF/A-2a or PDF/A-3a"):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "output.pdf",
                level=level,
                pdfua=True,
            )

    @pytest.mark.parametrize("level", ["2a", "3a"])
    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_pdfua_adds_identification_and_catalog_requirements(
        self,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        level: str,
    ) -> None:
        """The opt-in output declares PDF/UA-1 and keeps Level A structure."""
        mock_verapdf.side_effect = lambda *, path, flavour: VeraPDFResult(
            compliant=True,
            flavour=flavour,
        )
        output_path = tmp_dir / f"output_{level}_pdfua.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level=level,
            pdfua=True,
        )

        assert result.success is True
        with Pdf.open(output_path) as pdf:
            standards = {
                (standard.standard, standard.version)
                for standard in detect_iso_standards(pdf)
            }
            assert ("PDF/UA", "1") in standards
            assert bool(pdf.Root.MarkInfo.Marked) is True
            assert bool(pdf.Root.ViewerPreferences.DisplayDocTitle) is True
            assert str(pdf.Root.Lang)
            assert pdf.Root.StructTreeRoot.Type == Name.StructTreeRoot
            assert pdf.pages[0].obj.Tabs == Name.S
            assert str(pdf.docinfo.Title) == sample_pdf.stem
            metadata = bytes(pdf.Root.Metadata.read_bytes())
            assert b"pdfaExtension:schemas" in metadata
            assert sample_pdf.stem.encode() in metadata
        assert any("WCAG 2.1 3.1.1" in warning for warning in result.warnings)

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_preserves_tagged_structure(
        self, tagged_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Level A converts an already tagged PDF without replacing its tags."""
        output_path = tmp_dir / f"output_{level}.pdf"

        result = convert_to_pdfa(tagged_pdf, output_path, level=level)

        assert result.success is True
        assert result.level == level
        with Pdf.open(output_path) as pdf:
            assert bool(pdf.Root.MarkInfo.Marked) is True
            assert bool(pdf.Root.ViewerPreferences.DisplayDocTitle) is True
            assert pdf.Root.StructTreeRoot.Type == Name.StructTreeRoot
            assert len(pdf.Root.StructTreeRoot.K) == 1

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_tags_scanned_input(
        self, pdf_with_image: Path, tmp_dir: Path, level: str
    ) -> None:
        """Level A tags an image without inventing an alternative description."""
        output_path = tmp_dir / f"output_{level}.pdf"

        result = convert_to_pdfa(pdf_with_image, output_path, level=level)

        assert result.success is True
        assert any("requires manual review" in warning for warning in result.warnings)
        with Pdf.open(output_path) as pdf:
            assert bool(pdf.Root.MarkInfo.Marked) is True
            assert pdf.Root.StructTreeRoot.Type == Name.StructTreeRoot
            assert "/ParentTree" in pdf.Root.StructTreeRoot
            assert int(pdf.pages[0].StructParents) == 0
            content = bytes(pdf.pages[0].Contents.read_bytes())
            assert b"/Span" in content
            assert b"/MCID 0" in content
            structure = str(pdf.Root.StructTreeRoot)
            assert "/Figure" in structure
            assert "/Alt" not in structure
            assert "/ActualText" not in structure

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_removes_pdfua_claim_when_rebuilding_tags(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """A rebuilt technical tag tree must not retain a PDF/UA claim."""
        input_path = tmp_dir / f"claimed_pdfua_{level}.pdf"
        output_path = tmp_dir / f"rebuilt_{level}.pdf"
        xmp = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}">
    <rdf:Description rdf:about="">
      <pdfuaid:part>1</pdfuaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>""".encode()

        with Pdf.open(sample_pdf) as pdf:
            embed_xmp_metadata(pdf, xmp)
            pdf.save(input_path)

        result = convert_to_pdfa(input_path, output_path, level=level)

        assert result.success is True
        assert any(
            "PDF/UA identification removed" in warning for warning in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            assert all(
                standard.standard != "PDF/UA" for standard in detect_iso_standards(pdf)
            )

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_removes_pdfvt_claim_with_pdfx(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Dropping PDF/X also drops its dependent PDF/VT claim."""
        input_path = tmp_dir / f"claimed_pdfvt_{level}.pdf"
        output_path = tmp_dir / f"without_pdfvt_{level}.pdf"
        xmp = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}"
           xmlns:pdfxid="{NAMESPACES["pdfxid"]}"
           xmlns:pdfvtid="{NAMESPACES["pdfvtid"]}"
           xmlns:pdfe="{NAMESPACES["pdfeid"]}">
    <rdf:Description rdf:about="">
      <pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>
      <pdfvtid:GTS_PDFVTVersion>PDF/VT-1</pdfvtid:GTS_PDFVTVersion>
      <pdfe:ISO_PDFEVersion>PDF/E-1</pdfe:ISO_PDFEVersion>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>""".encode()

        with Pdf.open(sample_pdf) as pdf:
            embed_xmp_metadata(pdf, xmp)
            pdf.save(input_path)

        result = convert_to_pdfa(input_path, output_path, level=level)

        assert result.success is True
        assert any(
            "PDF/VT identification removed" in warning for warning in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            standards = {standard.standard for standard in detect_iso_standards(pdf)}
            assert "PDF/X" not in standards
            assert "PDF/VT" not in standards
            assert "PDF/E" not in standards

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_removes_pdfvt_claim_without_pdfx(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """A stale PDF/VT claim is removed even when PDF/X ID is already absent."""
        input_path = tmp_dir / f"pdfvt_without_pdfx_{level}.pdf"
        output_path = tmp_dir / f"without_pdfvt_{level}.pdf"
        xmp = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}"
           xmlns:pdfvtid="{NAMESPACES["pdfvtid"]}">
    <rdf:Description rdf:about="">
      <pdfvtid:GTS_PDFVTVersion>PDF/VT-1</pdfvtid:GTS_PDFVTVersion>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>""".encode()

        with Pdf.open(sample_pdf) as pdf:
            embed_xmp_metadata(pdf, xmp)
            pdf.save(input_path)

        result = convert_to_pdfa(input_path, output_path, level=level)

        assert result.success is True
        assert any(
            "PDF/VT identification removed" in warning for warning in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            assert all(
                standard.standard != "PDF/VT" for standard in detect_iso_standards(pdf)
            )

    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_level_a_removes_pdfe_claim_after_docinfo_cleanup(
        self, sample_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Dropping PDF/E's required DocInfo marker also drops its XMP claim."""
        input_path = tmp_dir / f"claimed_pdfe_{level}.pdf"
        output_path = tmp_dir / f"without_pdfe_{level}.pdf"
        xmp = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}"
           xmlns:pdfe="{NAMESPACES["pdfeid"]}">
    <rdf:Description rdf:about=""
                     pdfe:ISO_PDFEVersion="PDF/E-1"/>
  </rdf:RDF>
</x:xmpmeta>""".encode()

        with Pdf.open(sample_pdf) as pdf:
            embed_xmp_metadata(pdf, xmp)
            pdf.docinfo["/ISO_PDFEVersion"] = "PDF/E-1"
            pdf.save(input_path)

        result = convert_to_pdfa(input_path, output_path, level=level)

        assert result.success is True
        assert any(
            "PDF/E identification removed" in warning for warning in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            assert "/ISO_PDFEVersion" not in pdf.docinfo
            assert all(
                standard.standard != "PDF/E" for standard in detect_iso_standards(pdf)
            )

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_level_a_preserves_tags_after_noop_ocr(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tagged_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A no-op OCR pass must not replace valid semantic tags."""
        import shutil

        def copy_ocr_output(
            input_path: Path,
            output_path: Path,
            _languages: list[str],
            **_kwargs: object,
        ) -> Path:
            shutil.copy2(input_path, output_path)
            return output_path

        mock_apply_ocr.side_effect = copy_ocr_output
        output_path = tmp_dir / "ocr_output_2a.pdf"

        result = convert_to_pdfa(
            tagged_pdf,
            output_path,
            level="2a",
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        assert not any(
            "Tagged PDF structure generated" in item for item in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            document = pdf.Root.StructTreeRoot.K[0]
            assert document.S == Name.Document
            assert document.K[0].S == Name.P

    @patch("pdftopdfa.converter.ensure_logical_structure")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_level_a_does_not_report_absent_ocr_evidence_or_uncertainty(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_ensure: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """An empty OCR manifest must not produce unsupported warning claims."""
        import shutil

        def copy_ocr_output(
            input_path: Path,
            output_path: Path,
            _languages: list[str],
            **kwargs: object,
        ) -> Path:
            shutil.copy2(input_path, output_path)
            manifest_path = kwargs["_manifest_output_path"]
            assert isinstance(manifest_path, Path)
            manifest_path.write_text('{"pages": []}', encoding="utf-8")
            return output_path

        mock_apply_ocr.side_effect = copy_ocr_output
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_alternatives_review_required": 0,
            "semantic_vector_review_required": 0,
            "semantic_scanned_visual_review_required": 0,
            "semantic_link_review_required": 0,
            "semantic_form_review_required": 0,
            "structure_rebuilt": True,
            "semantic_structure_generated": True,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "empty-ocr-manifest.pdf",
            level="2a",
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        assert (
            "Semantic Tagged PDF structure generated from final digital content"
            in result.warnings
        )
        assert not any("OCR layout evidence" in item for item in result.warnings)
        assert not any(
            "review reported semantic uncertainties" in item for item in result.warnings
        )

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_level_a_preserves_tags_for_new_decorative_ocr_path(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tagged_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A new untagged path is repaired as an artifact without losing tags."""

        def add_untagged_ocr_content(
            input_path: Path,
            output_path: Path,
            _languages: list[str],
            **_kwargs: object,
        ) -> Path:
            with Pdf.open(input_path) as pdf:
                page = pdf.pages[0]
                page.obj.Contents.append(pdf.make_stream(b"0 0 10 10 re f"))
                pdf.save(output_path)
            return output_path

        mock_apply_ocr.side_effect = add_untagged_ocr_content
        output_path = tmp_dir / "ocr_changed_2a.pdf"

        result = convert_to_pdfa(
            tagged_pdf,
            output_path,
            level="2a",
            ocr_languages=["en"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        assert not any(
            "Tagged PDF structure generated" in item for item in result.warnings
        )
        with Pdf.open(output_path) as pdf:
            document = pdf.Root.StructTreeRoot.K[0]
            assert document.S == Name.Document
            assert document.K[0].S == Name.P
            assert b"/Artifact" in bytes(pdf.pages[0].Contents.read_bytes())

    @pytest.mark.parametrize("pdfa", [True, False])
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_ocr_rejects_same_input_and_output_without_modifying_source(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        pdfa: bool,
    ) -> None:
        """OCR cannot bypass the conversion's in-place write protection."""
        original = sample_pdf.read_bytes()

        with pytest.raises(ConversionError, match="Input and output paths must differ"):
            convert_to_pdfa(
                sample_pdf,
                sample_pdf,
                pdfa=pdfa,
                ocr_languages=["en"],
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            )

        assert sample_pdf.read_bytes() == original
        mock_apply_ocr.assert_not_called()

    @pytest.mark.parametrize("pdfa", [True, False])
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_ocr_rejects_hard_link_output_without_modifying_source(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        pdfa: bool,
    ) -> None:
        """OCR rejects a distinct output path that aliases the input file."""
        input_path = tmp_dir / "hard-link-input.pdf"
        output_path = tmp_dir / "hard-link-output.pdf"
        input_path.write_bytes(sample_pdf.read_bytes())
        original = input_path.read_bytes()
        try:
            os.link(input_path, output_path)
        except OSError as exc:
            pytest.skip(f"Hard links are not supported: {exc}")

        with pytest.raises(ConversionError, match="Input and output paths must differ"):
            convert_to_pdfa(
                input_path,
                output_path,
                pdfa=pdfa,
                ocr_languages=["en"],
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            )

        assert input_path.read_bytes() == original
        assert output_path.read_bytes() == original
        mock_apply_ocr.assert_not_called()

    @pytest.mark.skipif(
        not is_verapdf_available(),
        reason="veraPDF is not installed",
    )
    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_scanned_level_a_passes_verapdf(
        self, pdf_with_image: Path, tmp_dir: Path, level: str
    ) -> None:
        """An image-only scan passes the exact veraPDF level A profile."""
        output_path = tmp_dir / f"validated_{level}.pdf"

        convert_to_pdfa(pdf_with_image, output_path, level=level)
        validation = validate_with_verapdf(output_path, flavour=level)

        assert validation.compliant is True
        assert validation.flavour == level
        assert validation.failed_rules == 0

    @pytest.mark.skipif(
        not is_verapdf_available(),
        reason="veraPDF is not installed",
    )
    @pytest.mark.parametrize("level", ["2a", "3a"])
    def test_convert_pdfua_passes_both_verapdf_profiles(
        self, tagged_pdf: Path, tmp_dir: Path, level: str
    ) -> None:
        """Opt-in output passes its PDF/A and PDF/UA-1 machine checks."""
        output_path = tmp_dir / f"validated_{level}_pdfua.pdf"

        convert_to_pdfa(tagged_pdf, output_path, level=level, pdfua=True)

        assert validate_with_verapdf(output_path, flavour=level).compliant is True
        assert validate_with_verapdf(output_path, flavour="ua1").compliant is True

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_with_validation_flag(
        self, mock_verapdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """validate=True runs validation without errors for compliant PDF."""
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="3b")
        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, validate=True)

        assert result.success is True
        # Compliant PDF should have no validation errors
        has_validation_error = any("Validation:" in w for w in result.warnings)
        assert not has_validation_error
        assert result.validation_failed is False
        mock_verapdf.assert_called_once()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_pdfua_always_validates_both_profiles(
        self, mock_verapdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """PDF/UA candidates are checked even without the validation flag."""
        mock_verapdf.side_effect = lambda *, path, flavour: VeraPDFResult(
            compliant=True,
            flavour=flavour,
        )
        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            sample_pdf,
            output_path,
            level="2a",
            pdfua=True,
            validate=False,
        )

        assert result.validation_failed is False
        assert [call.kwargs["flavour"] for call in mock_verapdf.call_args_list] == [
            "2a",
            "ua1",
        ]

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_with_failing_validation_sets_flag(
        self, mock_verapdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Non-compliance is reported without suppressing the output."""
        mock_verapdf.return_value = MagicMock(
            compliant=False,
            errors=["Rule 6.1.2 failed"],
        )
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        result = convert_to_pdfa(sample_pdf, output_path, validate=True)

        assert result.success is False
        assert result.validation_failed is True
        assert result.error == "Validation failed; output candidate was published"
        assert any("Validation: Rule 6.1.2 failed" in w for w in result.warnings)
        assert any("published despite" in warning for warning in result.warnings)
        validated_path = mock_verapdf.call_args.kwargs["path"]
        assert validated_path != output_path
        assert validated_path.parent.parent == output_path.parent
        assert output_path.read_bytes() != sentinel
        assert output_path.read_bytes().startswith(b"%PDF-")
        assert not validated_path.exists()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_with_unavailable_validation_publishes_output(
        self,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """A validator error is reported without suppressing the output."""
        mock_verapdf.side_effect = VeraPDFError("veraPDF crashed")
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        result = convert_to_pdfa(sample_pdf, output_path, validate=True)

        assert result.success is False
        assert result.validation_failed is True
        assert result.error == "Validation failed; output candidate was published"
        assert "Validation: veraPDF could not run: veraPDF crashed" in result.warnings
        assert any("published despite" in warning for warning in result.warnings)
        validated_path = mock_verapdf.call_args.kwargs["path"]
        assert validated_path != output_path
        assert output_path.read_bytes() != sentinel
        assert output_path.read_bytes().startswith(b"%PDF-")
        assert not validated_path.exists()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_encrypted_skip_is_not_validated_when_requested(
        self,
        mock_verapdf: MagicMock,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Explicit validation is skipped when no PDF/A output was created."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)

        result = convert_to_pdfa(password_encrypted_pdf, output_path, validate=True)

        assert result.success is True
        assert result.skipped is True
        assert result.validation_failed is False
        assert result.level is None
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()
        mock_verapdf.assert_not_called()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_pdfua_encrypted_skip_is_not_validated(
        self,
        mock_verapdf: MagicMock,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Mandatory PDF/UA validation is skipped for an unchanged input."""
        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            password_encrypted_pdf,
            output_path,
            level="2a",
            pdfua=True,
        )

        assert result.success is True
        assert result.skipped is True
        assert result.validation_failed is False
        assert result.level is None
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()
        mock_verapdf.assert_not_called()

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
    def test_convert_forwards_layout_configuration(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The public API forwards the layout flag."""
        import shutil

        mock_apply_ocr.side_effect = lambda source, destination, *args, **kwargs: (
            shutil.copy2(source, destination) or destination
        )
        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "layout.pdf",
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            ocr_layout=True,
        )

        assert result.success is True
        assert mock_apply_ocr.call_args.kwargs["layout"] is True

    def test_convert_layout_requires_ocr_models(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Layout processing cannot run without the OCR model pair."""
        with pytest.raises(ValueError, match="OCR requires"):
            convert_to_pdfa(
                sample_pdf,
                tmp_dir / "layout.pdf",
                ocr_layout=True,
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
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    def test_convert_retries_transient_inline_ocr_temp_cleanup_failures(
        self,
        _mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The final cleanup retries transient Windows-style file locks."""
        import shutil
        import tempfile
        from collections import Counter

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

        def copy_ocr_output(
            input_path: Path, output_path: Path, *_args: object, **_kwargs: object
        ) -> Path:
            return Path(shutil.copy2(input_path, output_path))

        mock_apply_ocr.side_effect = copy_ocr_output
        created: list[Path] = []
        real_mkstemp = tempfile.mkstemp
        real_unlink = Path.unlink
        unlink_attempts: Counter[Path] = Counter()

        def tracking_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(Path(path))
            return fd, path

        def transient_unlink(path: Path, *args: object, **kwargs: object) -> None:
            retry_inline = "_sig_" in path.name or "_clean_" in path.name
            unlink_attempts[path] += 1
            if retry_inline and unlink_attempts[path] == 1:
                raise PermissionError("temporary Windows file lock")
            real_unlink(path, *args, **kwargs)

        output_path = tmp_dir / "output.pdf"
        with (
            patch(
                "pdftopdfa.converter.tempfile.mkstemp",
                side_effect=tracking_mkstemp,
            ),
            patch("pathlib.Path.unlink", new=transient_unlink),
        ):
            result = convert_to_pdfa(
                annotated_pdf,
                output_path,
                ocr_languages=["en"],
                ocr_detection_model_dir=_DETECTION_MODEL_DIR,
                ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
            )

        assert result.success is True
        retried = [
            path for path in created if "_sig_" in path.name or "_clean_" in path.name
        ]
        assert len(retried) == 2
        assert all(unlink_attempts[path] == 2 for path in retried)
        assert all(not path.exists() for path in created)

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

    @pytest.mark.parametrize("skip_any_pdfa", [False, True])
    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available")
    def test_requested_ocr_bypasses_compliant_pdfa_skip(
        self,
        mock_is_ocr_available: MagicMock,
        mock_apply_ocr: MagicMock,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        skip_any_pdfa: bool,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Normal OCR still runs for a compliant but potentially image-only PDF/A."""
        import shutil

        mock_is_ocr_available.return_value = True
        mock_detect.return_value = "2a"
        mock_verapdf.return_value = VeraPDFResult(compliant=True, flavour="2a")
        mock_apply_ocr.side_effect = lambda inp, out, *a, **kw: (
            shutil.copy(inp, out) or out
        )

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "ocr.pdf",
            level="2a",
            skip_any_pdfa=skip_any_pdfa,
            ocr_languages=["de"],
            ocr_detection_model_dir=_DETECTION_MODEL_DIR,
            ocr_recognition_model_dir=_RECOGNITION_MODEL_DIR,
        )

        assert result.success is True
        assert result.skipped is False
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
        assert result.level is None
        assert any("digital signatures" in warning for warning in result.warnings)
        assert output_path.read_bytes() == signed_input.read_bytes()
        mock_is_ocr_available.assert_not_called()
        mock_apply_ocr.assert_not_called()

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_signed_skip_is_not_validated_when_requested(
        self,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Explicit validation is skipped when a signature prevents conversion."""
        signed_input = tmp_dir / "signed_input.pdf"
        _write_signed_pdf(sample_pdf, signed_input)
        output_path = tmp_dir / "output.pdf"

        result = convert_to_pdfa(
            signed_input,
            output_path,
            level="3b",
            validate=True,
        )

        assert result.success is True
        assert result.skipped is True
        assert result.validation_failed is False
        assert result.level is None
        assert output_path.read_bytes() == signed_input.read_bytes()
        mock_verapdf.assert_not_called()

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
        validated_path = mock_verapdf.call_args.args[0]
        assert validated_path != sample_pdf
        assert validated_path.parent.parent == output_path.parent
        assert not validated_path.exists()
        assert mock_verapdf.call_args.kwargs == {
            "flavour": "2b",
            "non_compliant_log_level": logging.WARNING,
        }

    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level", return_value="2b")
    def test_already_compliant_skip_publishes_exact_validated_snapshot(
        self,
        _mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """PDF/A skip validation and publication use one immutable snapshot."""
        expected = sample_pdf.read_bytes()

        def validate_snapshot(path: Path, **_kwargs: object) -> VeraPDFResult:
            assert path.read_bytes() == expected
            sample_pdf.write_bytes(b"changed after snapshot")
            return VeraPDFResult(compliant=True, flavour="2b")

        mock_verapdf.side_effect = validate_snapshot
        output_path = tmp_dir / "output.pdf"
        output_path.write_bytes(b"existing output")

        result = convert_to_pdfa(sample_pdf, output_path, level="2b")

        assert result.skipped is True
        assert output_path.read_bytes() == expected

    @pytest.mark.parametrize("level", ["2a", "3a"])
    @patch("pdftopdfa.converter.validate_with_verapdf")
    @patch("pdftopdfa.converter.detect_pdfa_level")
    def test_already_compliant_level_a_is_semantically_processed(
        self,
        mock_detect: MagicMock,
        mock_verapdf: MagicMock,
        level: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Level A inputs are not skipped before semantic inspection and repair."""
        mock_detect.return_value = level

        output_path = tmp_dir / "output.pdf"
        result = convert_to_pdfa(sample_pdf, output_path, level=level)

        assert result.success is True
        assert result.level == level
        assert result.skipped is False
        assert output_path.exists()
        mock_verapdf.assert_not_called()

    @pytest.mark.parametrize(
        ("review_count", "expected"),
        [
            (
                1,
                "1 Figure/Formula element requires manual review: no trustworthy "
                "Alt, ActualText, or Caption is available",
            ),
            (
                2,
                "2 Figure/Formula elements require manual review: no trustworthy "
                "Alt, ActualText, or Caption is available",
            ),
        ],
    )
    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_missing_trustworthy_alternatives(
        self,
        mock_ensure: MagicMock,
        review_count: int,
        expected: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Level A conversion exposes semantic descriptions needing author review."""
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_alternatives_review_required": review_count,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / f"review-{review_count}.pdf",
            level="2a",
        )

        assert result.success is True
        assert expected in result.warnings

    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_unclassified_direct_vector_painting(
        self,
        mock_ensure: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_alternatives_review_required": 0,
            "semantic_vector_review_required": 2,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "vector-review.pdf",
            level="2a",
        )

        assert result.success is True
        assert (
            "2 pages require manual review: unclassified direct vector painting "
            "was retained as a Layout artifact"
        ) in result.warnings

    @pytest.mark.parametrize(
        ("review_count", "expected"),
        [
            (
                1,
                "1 inferred non-rectangular table requires manual review and was "
                "retained as conservative reading structure",
            ),
            (
                2,
                "2 inferred non-rectangular tables require manual review and were "
                "retained as conservative reading structure",
            ),
        ],
    )
    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_conservatively_demoted_tables(
        self,
        mock_ensure: MagicMock,
        review_count: int,
        expected: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_table_review_required": review_count,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / "table-review.pdf",
            level="2a",
        )

        assert expected in result.warnings

    @pytest.mark.parametrize(
        ("review_count", "expected"),
        [
            (
                1,
                "1 OCR page requires manual review: a full-page scan may contain "
                "meaningful non-text visuals that available OCR layout evidence "
                "cannot represent",
            ),
            (
                2,
                "2 OCR pages require manual review: full-page scans may contain "
                "meaningful non-text visuals that available OCR layout evidence "
                "cannot represent",
            ),
        ],
    )
    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_scanned_visual_uncertainty(
        self,
        mock_ensure: MagicMock,
        review_count: int,
        expected: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_scanned_visual_review_required": review_count,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / f"scanned-visual-review-{review_count}.pdf",
            level="2a",
        )

        assert result.success is True
        assert expected in result.warnings

    @pytest.mark.parametrize(
        ("review_count", "expected"),
        [
            (
                1,
                "1 Link annotation requires manual review: the link could not be "
                "safely associated with content owned by a single logical structure "
                "element",
            ),
            (
                2,
                "2 Link annotations require manual review: the link could not be "
                "safely associated with content owned by a single logical structure "
                "element",
            ),
        ],
    )
    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_unassociated_links(
        self,
        mock_ensure: MagicMock,
        review_count: int,
        expected: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_link_review_required": review_count,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / f"link-review-{review_count}.pdf",
            level="3a",
        )

        assert result.success is True
        assert expected in result.warnings

    @pytest.mark.parametrize(
        ("review_count", "expected"),
        [
            (
                1,
                "1 Form field requires manual review: no trustworthy tooltip or "
                "field name is available",
            ),
            (
                2,
                "2 Form fields require manual review: no trustworthy tooltip or "
                "field name is available",
            ),
        ],
    )
    @patch("pdftopdfa.converter.ensure_logical_structure")
    def test_level_a_reports_unnamed_form_fields(
        self,
        mock_ensure: MagicMock,
        review_count: int,
        expected: str,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        mock_ensure.return_value = {
            "semantic_repairs": 0,
            "semantic_form_review_required": review_count,
            "structure_rebuilt": False,
        }

        result = convert_to_pdfa(
            sample_pdf,
            tmp_dir / f"form-review-{review_count}.pdf",
            level="2a",
        )

        assert result.success is True
        assert expected in result.warnings

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
        validated_path = mock_verapdf.call_args.args[0]
        assert validated_path != sample_pdf
        assert validated_path.parent.parent == output_path.parent
        assert not validated_path.exists()
        assert mock_verapdf.call_args.kwargs == {
            "flavour": detected_level,
            "non_compliant_log_level": logging.WARNING,
        }

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

    @patch(
        "pdftopdfa.converter.save_pdfa", side_effect=PermissionError("access denied")
    )
    def test_output_permission_error_is_not_wrapped(
        self,
        _mock_save: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Output permission failures remain distinguishable to API callers."""
        with pytest.raises(PermissionError, match="access denied"):
            convert_to_pdfa(sample_pdf, tmp_dir / "output.pdf")

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

    def test_convert_directory_rejects_invalid_level_when_empty(
        self, tmp_dir: Path
    ) -> None:
        """A global PDF/A configuration error fails before file discovery."""
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()

        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            convert_directory(empty_dir, level="4b", show_progress=False)

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
        assert [input_path for input_path, _ in file_pairs] == [
            source,
            previous_output,
        ]

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
        (input_dir / "doc_processed.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(input_dir, pdfa=False, show_progress=False)

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, input_dir / "doc_processed.pdf")
        ]
        assert mock_convert_files.call_args.kwargs["pdfa"] is False

    @pytest.mark.parametrize(
        ("pdfa", "source_name", "output_name"),
        [
            (True, "archive_pdfa.pdf", "archive_pdfa_pdfa.pdf"),
            (False, "archive_processed.pdf", "archive_processed_processed.pdf"),
        ],
    )
    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_processes_standalone_output_suffix_source(
        self,
        mock_convert_files: MagicMock,
        pdfa: bool,
        source_name: str,
        output_name: str,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """A suffix-like filename is input unless its corresponding source exists."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        source = input_dir / source_name
        source.write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(input_dir, pdfa=pdfa, show_progress=False)

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, input_dir / output_name)
        ]

    @patch("pdftopdfa.converter.convert_files")
    def test_convert_directory_skips_output_for_uppercase_pdf_source(
        self,
        mock_convert_files: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Generated output matching an uppercase-extension source is skipped."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        source = input_dir / "document.PDF"
        source.write_bytes(sample_pdf_bytes)
        (input_dir / "document_pdfa.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(input_dir, show_progress=False)

        assert mock_convert_files.call_args.kwargs["file_pairs"] == [
            (source, input_dir / "document_pdfa.pdf")
        ]

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
        """convert_directory forwards PDF/A and PDF/UA conformance options."""
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(sample_pdf_bytes)
        mock_convert_files.return_value = []

        convert_directory(
            input_dir,
            level="2a",
            pdfua=True,
            show_progress=False,
            skip_any_pdfa=True,
        )

        assert mock_convert_files.call_args.kwargs["skip_any_pdfa"] is True
        assert mock_convert_files.call_args.kwargs["pdfua"] is True

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

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_files_reports_validator_runtime_failure(
        self,
        mock_verapdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Batch results preserve conversion context if veraPDF cannot run."""
        mock_verapdf.side_effect = VeraPDFError("validator crashed")
        output_path = tmp_dir / "output.pdf"

        results = convert_files(
            [(sample_pdf, output_path)],
            level="2a",
            validate=True,
        )

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.validation_failed is True
        assert result.skipped is False
        assert result.level == "2a"
        assert "Validation: veraPDF could not run: validator crashed" in result.warnings
        assert any("published despite" in warning for warning in result.warnings)
        assert output_path.is_file()
        assert output_path.read_bytes().startswith(b"%PDF-")

    @patch("pdftopdfa.converter.validate_with_verapdf")
    def test_convert_files_does_not_validate_encrypted_skip(
        self,
        mock_verapdf: MagicMock,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Batch validation ignores an unchanged encrypted copy."""
        output_path = tmp_dir / "output.pdf"

        results = convert_files(
            [(password_encrypted_pdf, output_path)],
            level="3a",
            validate=True,
        )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.validation_failed is False
        assert result.skipped is True
        assert result.level is None
        assert any("PDF is encrypted" in warning for warning in result.warnings)
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()
        mock_verapdf.assert_not_called()

    def test_convert_files_copies_password_encrypted_input(
        self,
        password_encrypted_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Batch conversion preserves a password-protected input unchanged."""
        output_path = tmp_dir / "output.pdf"

        results = convert_files(
            [(password_encrypted_pdf, output_path)],
            level="2a",
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is True
        assert results[0].level is None
        assert output_path.read_bytes() == password_encrypted_pdf.read_bytes()

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

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_rejects_duplicate_outputs_before_conversion(
        self,
        mock_convert_to_pdfa: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Two batch entries cannot silently overwrite the same output."""
        first_input = tmp_dir / "first.pdf"
        second_input = tmp_dir / "second.pdf"
        output_path = tmp_dir / "output.pdf"
        first_input.write_bytes(sample_pdf_bytes)
        second_input.write_bytes(sample_pdf_bytes)

        with pytest.raises(
            ConversionError, match="Output paths in a batch must be unique"
        ):
            convert_files(
                [(first_input, output_path), (second_input, output_path)],
                force_overwrite=True,
            )

        mock_convert_to_pdfa.assert_not_called()

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_rejects_output_overlapping_later_input(
        self,
        mock_convert_to_pdfa: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """An earlier output cannot replace a later batch input."""
        first_input = tmp_dir / "first.pdf"
        second_input = tmp_dir / "second.pdf"
        final_output = tmp_dir / "final.pdf"
        first_input.write_bytes(sample_pdf_bytes)
        second_input.write_bytes(sample_pdf_bytes)

        with pytest.raises(ConversionError, match="overlaps an input path"):
            convert_files(
                [(first_input, second_input), (second_input, final_output)],
                force_overwrite=True,
            )

        mock_convert_to_pdfa.assert_not_called()

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_rejects_hard_link_output_overlapping_input(
        self,
        mock_convert_to_pdfa: MagicMock,
        tmp_dir: Path,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Batch overlap detection follows file identity through hard links."""
        first_input = tmp_dir / "first.pdf"
        second_input = tmp_dir / "second.pdf"
        linked_output = tmp_dir / "linked-output.pdf"
        final_output = tmp_dir / "final.pdf"
        first_input.write_bytes(sample_pdf_bytes)
        second_input.write_bytes(sample_pdf_bytes)
        try:
            os.link(second_input, linked_output)
        except OSError as exc:
            pytest.skip(f"Hard links are not supported: {exc}")

        with pytest.raises(ConversionError, match="overlaps an input path"):
            convert_files(
                [(first_input, linked_output), (second_input, final_output)],
                force_overwrite=True,
            )

        mock_convert_to_pdfa.assert_not_called()

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

    @patch("pdftopdfa.converter.save_pdfa")
    def test_convert_files_does_not_clobber_target_created_during_conversion(
        self,
        mock_save: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """The no-overwrite check is enforced again by atomic publication."""
        output_path = tmp_dir / "output.pdf"
        sentinel = b"concurrently created output"

        def create_target_then_save_candidate(
            _pdf: pikepdf.Pdf,
            staged_path: Path,
            _level: str,
            *,
            verify: bool,
        ) -> None:
            assert verify is True
            output_path.write_bytes(sentinel)
            staged_path.write_bytes(b"complete staged PDF")

        mock_save.side_effect = create_target_then_save_candidate

        results = convert_files(
            [(sample_pdf, output_path)],
            force_overwrite=False,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "already exists" in results[0].error
        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".output_pdfa_stage_*"))

    def test_copy_skip_does_not_clobber_target_created_during_staging(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """Unchanged-copy branches retain the same race-safe no-clobber rule."""
        output_path = tmp_dir / "processed.pdf"
        sentinel = b"concurrently created copy target"
        real_snapshot = staged_file_snapshot

        def create_target_after_copy(path: Path):
            path = Path(path)
            if path.parent.name.startswith(".processed_copy_"):
                output_path.write_bytes(sentinel)
            return real_snapshot(path)

        with patch(
            "pdftopdfa.converter.staged_file_snapshot",
            side_effect=create_target_after_copy,
        ):
            results = convert_files(
                [(sample_pdf, output_path)],
                pdfa=False,
                force_overwrite=False,
            )

        assert len(results) == 1
        assert results[0].success is False
        assert "already exists" in results[0].error
        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".processed_copy_*"))

    @pytest.mark.parametrize("target_kind", ["directory", "symlink"])
    def test_no_clobber_publication_rejects_nonregular_target(
        self,
        target_kind: str,
        tmp_dir: Path,
    ) -> None:
        """A directory or symlink appearing at the target is never replaced."""
        stage_directory = tmp_dir / "stage"
        stage_directory.mkdir()
        staged = stage_directory / "candidate.pdf"
        staged.write_bytes(b"candidate")
        snapshot = staged_file_snapshot(staged)
        destination = tmp_dir / "output.pdf"
        symlink_target = tmp_dir / "symlink-target.pdf"
        if target_kind == "directory":
            destination.mkdir()
        else:
            symlink_target.write_bytes(b"symlink sentinel")
            try:
                destination.symlink_to(symlink_target)
            except OSError as exc:
                pytest.skip(f"File symlinks are not supported: {exc}")

        with pytest.raises(ConversionError, match="not a regular file"):
            publish_staged_file_impl(
                staged,
                destination,
                snapshot,
                backup=stage_directory / "backup.pdf",
                require_absent=True,
            )

        assert staged.read_bytes() == b"candidate"
        if target_kind == "directory":
            assert destination.is_dir()
        else:
            assert destination.is_symlink()
            assert symlink_target.read_bytes() == b"symlink sentinel"

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
        """convert_files forwards PDF/A and PDF/UA conformance options."""
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
            level="2a",
            pdfua=True,
            skip_any_pdfa=True,
        )

        assert mock_convert_to_pdfa.call_args.kwargs["skip_any_pdfa"] is True
        assert mock_convert_to_pdfa.call_args.kwargs["pdfua"] is True

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

    def test_convert_files_rejects_invalid_level_when_empty(self) -> None:
        """A global PDF/A configuration error fails before iteration."""
        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            convert_files([], level="4b")

    @patch("pdftopdfa.converter.convert_to_pdfa")
    def test_convert_files_normalizes_level_before_forwarding(
        self, mock_convert_to_pdfa: MagicMock, tmp_dir: Path
    ) -> None:
        """The batch API forwards one normalized level to every conversion."""
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        mock_convert_to_pdfa.return_value = ConversionResult(
            success=True,
            input_path=input_path,
            output_path=output_path,
            level="2a",
        )

        convert_files([(input_path, output_path)], level="2A")

        assert mock_convert_to_pdfa.call_args.kwargs["level"] == "2a"


class TestPublicationWithoutHardLinks:
    """Publication must survive filesystems that reject hard links."""

    @staticmethod
    def _unsupported_link(*_args, **_kwargs):
        raise OSError(errno.EPERM, "operation not permitted")

    @staticmethod
    def _stage(tmp_dir: Path, payload: bytes) -> tuple[Path, Path]:
        stage_directory = tmp_dir / f"stage-{payload.decode()}"
        stage_directory.mkdir()
        staged = stage_directory / "candidate.pdf"
        staged.write_bytes(payload)
        return stage_directory, staged

    def test_publishes_exact_bytes_without_hard_links(self, tmp_dir: Path) -> None:
        """An exFAT-style EPERM falls back to an exclusive copy."""
        stage_directory, staged = self._stage(tmp_dir, b"candidate")
        snapshot = staged_file_snapshot(staged)
        destination = tmp_dir / "output.pdf"

        with patch("pdftopdfa.staging.os.link", self._unsupported_link):
            published = publish_staged_file_impl(
                staged,
                destination,
                snapshot,
                backup=stage_directory / "backup.pdf",
                require_absent=True,
            )

        assert destination.read_bytes() == b"candidate"
        assert published.sha256 == snapshot.sha256
        assert published.size == snapshot.size
        assert not staged.exists()

    def test_no_clobber_still_refuses_existing_target(self, tmp_dir: Path) -> None:
        """Losing hard links must not weaken the overwrite protection."""
        stage_directory, staged = self._stage(tmp_dir, b"candidate")
        snapshot = staged_file_snapshot(staged)
        destination = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        destination.write_bytes(sentinel)

        with (
            patch("pdftopdfa.staging.os.link", self._unsupported_link),
            pytest.raises(ConversionError, match="already exists"),
        ):
            publish_staged_file_impl(
                staged,
                destination,
                snapshot,
                backup=stage_directory / "backup.pdf",
                require_absent=True,
            )

        assert destination.read_bytes() == sentinel
        assert staged.read_bytes() == b"candidate"

    def test_overwrite_retains_backup_without_hard_links(self, tmp_dir: Path) -> None:
        """The retained target is copied when it cannot be hard linked."""
        stage_directory, staged = self._stage(tmp_dir, b"replacement")
        snapshot = staged_file_snapshot(staged)
        destination = tmp_dir / "output.pdf"
        destination.write_bytes(b"previous output")
        backup = stage_directory / "backup.pdf"

        with patch("pdftopdfa.staging.os.link", self._unsupported_link):
            publish_staged_file_impl(staged, destination, snapshot, backup=backup)

        assert destination.read_bytes() == b"replacement"
        assert backup.read_bytes() == b"previous output"

    def test_rollback_restores_copied_backup(self, tmp_dir: Path) -> None:
        """A copied backup is a different inode but still rolls back."""
        stage_directory, staged = self._stage(tmp_dir, b"replacement")
        snapshot = staged_file_snapshot(staged)
        destination = tmp_dir / "output.pdf"
        destination.write_bytes(b"previous output")
        backup = stage_directory / "backup.pdf"

        with (
            patch("pdftopdfa.staging.os.link", self._unsupported_link),
            patch("pdftopdfa.staging.os.replace", side_effect=OSError("no rename")),
            pytest.raises(OSError, match="no rename"),
        ):
            publish_staged_file_impl(staged, destination, snapshot, backup=backup)

        assert destination.read_bytes() == b"previous output"


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

    def test_bad_header_fails(self, tmp_dir: Path) -> None:
        """File with the wrong header fails verification."""
        bad_file = tmp_dir / "bad.pdf"
        bad_file.write_bytes(b"%PDF-2.0 garbage data\n%\xe2\xe3\xcf\xd3\n")

        with pytest.raises(ConversionError, match="does not start with"):
            _verify_file_structure(bad_file, "1.7")

    def test_nonexistent_file_fails(self, tmp_dir: Path) -> None:
        """A missing output fails verification."""
        missing = tmp_dir / "missing.pdf"

        with pytest.raises(ConversionError, match="could not read"):
            _verify_file_structure(missing, "1.7")

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

    def test_no_eof_marker_fails(self, tmp_dir: Path) -> None:
        """File without %%EOF fails hardening."""
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.7\nsome content\n")
        with pytest.raises(ConversionError, match="No %%EOF marker"):
            _truncate_trailing_data(f)

    def test_nonexistent_file_fails(self, tmp_dir: Path) -> None:
        """A missing file fails hardening."""
        f = tmp_dir / "missing.pdf"
        with pytest.raises(ConversionError, match="Could not read file"):
            _truncate_trailing_data(f)

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

    def test_nonexistent_file_fails(self, tmp_dir: Path) -> None:
        """A missing file fails binary-comment hardening."""
        f = tmp_dir / "missing.pdf"
        with pytest.raises(ConversionError, match="Could not read header"):
            _ensure_binary_comment(f, "1.7")

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


@pytest.mark.skipif(
    os.environ.get("PDFTOPDFA_RUN_TEST_DOCS") != "1",
    reason="Set PDFTOPDFA_RUN_TEST_DOCS=1 for the local test_docs matrix",
)
def test_local_test_docs_pdfa_level_a_matrix(tmp_path: Path) -> None:
    """Convert every local test document to 2a and 3a and reopen its tags."""
    test_docs = Path(__file__).resolve().parents[1] / "test_docs"
    sources = sorted(test_docs.glob("*.pdf"))
    assert sources, f"No PDFs found in {test_docs}"
    failures = []
    for source_index, source in enumerate(sources):
        for level in ("2a", "3a"):
            output = tmp_path / f"{source_index:03d}-{level}.pdf"
            try:
                result = convert_to_pdfa(
                    source,
                    output,
                    level=level,
                    validate=True,
                )
                assert result.success is True
                assert result.validation_failed is False
                validation = validate_with_verapdf(output, flavour=level)
                assert validation.compliant is True
                with Pdf.open(output) as pdf:
                    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
                    assert isinstance(root, Dictionary)
                    assert root.get("/Type") == Name.StructTreeRoot
                    assert "/ParentTree" in root
                    original_root = root.objgen
                    preserved = ensure_logical_structure(pdf, semantic=True)
                    assert preserved["structure_preserved"] is True
                    assert preserved["structure_rebuilt"] is False
                    assert pdf.Root["/StructTreeRoot"].objgen == original_root
            except Exception as exc:
                failures.append(f"{source.name} PDF/A-{level}: {exc}")
    assert not failures, "\n".join(failures)
