# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for ocr.py."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf
from PIL import Image

from pdftopdfa.exceptions import OCRError
from pdftopdfa.ocr import (
    _PREPROCESS_QUALITIES,
    OCR_SETTINGS,
    OcrQuality,
    _detect_consistent_text_skew,
    _normalize_best_quality_skipped_text_pages,
    _normalize_best_quality_text_page_rotations,
    _normalize_best_quality_text_page_skew,
    _OrientationResult,
    _page_has_images,
    _page_has_text,
    _parse_tesseract_osd,
    _should_clear_page_rotate,
    apply_ocr,
    is_ocr_available,
    needs_ocr,
)
from pdftopdfa.ocr_rotation_fix import (
    _compose_page_rotation,
    _should_swap_visible_page_axis,
    filter_pdf_page,
    rasterize_pdf_page,
)


class TestIsOcrAvailable:
    """Tests for is_ocr_available."""

    def test_is_ocr_available_returns_bool(self) -> None:
        """Checks that is_ocr_available returns a boolean value."""
        result = is_ocr_available()

        assert isinstance(result, bool)


class TestNeedsOcr:
    """Tests for needs_ocr."""

    def test_empty_pdf_returns_false(self, empty_pdf_obj: Pdf) -> None:
        """Empty PDF (without pages) doesn't need OCR."""
        result = needs_ocr(empty_pdf_obj)

        assert result is False

    def test_pdf_with_text_returns_false(self, pdf_with_text_obj: Pdf) -> None:
        """PDF with text doesn't need OCR."""
        result = needs_ocr(pdf_with_text_obj)

        assert result is False

    def test_pdf_with_image_only_returns_true(self, pdf_with_image_obj: Pdf) -> None:
        """PDF with image only (without text) needs OCR."""
        result = needs_ocr(pdf_with_image_obj)

        assert result is True

    def test_threshold_parameter_low(self, pdf_with_image_obj: Pdf) -> None:
        """Low threshold (0.0) detects OCR need."""
        result = needs_ocr(pdf_with_image_obj, threshold=0.0)

        assert result is True

    def test_threshold_parameter_high(self, pdf_with_image_obj: Pdf) -> None:
        """High threshold (1.0) with one page: 100% required."""
        # With one page with image without text: ratio = 1.0, so >= 1.0
        result = needs_ocr(pdf_with_image_obj, threshold=1.0)

        assert result is True

    def test_threshold_above_ratio_returns_false(self, tmp_dir: Path) -> None:
        """Threshold above actual ratio returns False."""
        # Create PDF with 2 pages: one with text, one with image
        pdf = new_pdf()

        # Page 1: With text
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )
        page1_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )
        content1 = pdf.make_stream(b"BT /F1 12 Tf 100 700 Td (Text) Tj ET")
        page1_dict[Name.Contents] = content1
        pdf.pages.append(pikepdf.Page(page1_dict))

        # Page 2: With image without text
        image_data = b"\x80"
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceGray
        image_stream[Name.BitsPerComponent] = 8

        page2_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream)),
        )
        content2 = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q")
        page2_dict[Name.Contents] = content2
        pdf.pages.append(pikepdf.Page(page2_dict))

        # ratio = 1/2 = 0.5, threshold = 0.6 -> False
        result = needs_ocr(pdf, threshold=0.6)

        assert result is False

    def test_simple_pdf_without_images_returns_false(self, sample_pdf_obj: Pdf) -> None:
        """Simple PDF without images doesn't need OCR."""
        result = needs_ocr(sample_pdf_obj)

        assert result is False


class TestPageHasImages:
    """Tests for _page_has_images."""

    def test_page_without_resources(self, tmp_dir: Path) -> None:
        """Page without Resources has no images."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_images(pdf.pages[0])

        assert result is False

    def test_page_with_empty_resources(self, tmp_dir: Path) -> None:
        """Page with empty Resources has no images."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_images(pdf.pages[0])

        assert result is False

    def test_page_with_image_xobject(self, pdf_with_image_obj: Pdf) -> None:
        """Page with image XObject is detected."""
        result = _page_has_images(pdf_with_image_obj.pages[0])

        assert result is True

    def test_page_with_form_xobject_no_image(self, tmp_dir: Path) -> None:
        """Page with Form XObject (no image) has no images."""
        pdf = new_pdf()

        # Create Form XObject (not Image)
        form_stream = pdf.make_stream(b"")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_images(pdf.pages[0])

        assert result is False


class TestPageHasText:
    """Tests for _page_has_text."""

    def test_page_without_contents(self, tmp_dir: Path) -> None:
        """Page without content stream has no text."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_text(pdf.pages[0])

        assert result is False

    def test_page_with_text_operators_tj(self, pdf_with_text_obj: Pdf) -> None:
        """Page with Tj operator is detected as text."""
        result = _page_has_text(pdf_with_text_obj.pages[0])

        assert result is True

    def test_page_with_text_operators_tj_array(self, tmp_dir: Path) -> None:
        """Page with TJ operator (array) is detected as text."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
        )

        # Content stream with TJ operator
        content_data = b"BT /F1 12 Tf 100 700 Td [(He) 10 (llo)] TJ ET"
        content_stream = pdf.make_stream(content_data)
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_text(pdf.pages[0])

        assert result is True

    def test_page_with_graphics_only(self, pdf_with_image_obj: Pdf) -> None:
        """Page with only graphics operators has no text."""
        result = _page_has_text(pdf_with_image_obj.pages[0])

        assert result is False

    def test_page_with_content_array(self, tmp_dir: Path) -> None:
        """Page with content array (multiple streams) is checked correctly."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        # Two content streams
        stream1 = pdf.make_stream(b"q 1 0 0 1 0 0 cm Q")  # Only graphics
        stream2 = pdf.make_stream(b"BT /F1 12 Tf (Text) Tj ET")  # With text

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font_dict)),
            Contents=Array([stream1, stream2]),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = _page_has_text(pdf.pages[0])

        assert result is True

    def test_page_with_text_in_form_xobject(self, tmp_dir: Path) -> None:
        """Text inside a Form XObject is detected."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        # Create Form XObject containing text
        form_stream = pdf.make_stream(b"BT /F1 12 Tf 100 700 Td (Text in Form) Tj ET")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 612, 792])
        form_stream[Name.Resources] = Dictionary(Font=Dictionary(F1=font_dict))

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        # Page content only invokes the Form XObject, no direct text
        content = pdf.make_stream(b"q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        pdf.pages.append(pikepdf.Page(page_dict))

        result = _page_has_text(pdf.pages[0])

        assert result is True

    def test_page_with_nested_form_xobject_text(self, tmp_dir: Path) -> None:
        """Text inside a nested Form XObject is detected."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        # Inner Form XObject with text
        inner_form = pdf.make_stream(b"BT /F1 12 Tf 50 50 Td (Nested text) Tj ET")
        inner_form[Name.Type] = Name.XObject
        inner_form[Name.Subtype] = Name.Form
        inner_form[Name.BBox] = Array([0, 0, 200, 200])
        inner_form[Name.Resources] = Dictionary(Font=Dictionary(F1=font_dict))

        # Outer Form XObject that references inner
        outer_form = pdf.make_stream(b"q /Fm1 Do Q")
        outer_form[Name.Type] = Name.XObject
        outer_form[Name.Subtype] = Name.Form
        outer_form[Name.BBox] = Array([0, 0, 612, 792])
        outer_form[Name.Resources] = Dictionary(XObject=Dictionary(Fm1=inner_form))

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=outer_form)),
        )
        content = pdf.make_stream(b"q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        pdf.pages.append(pikepdf.Page(page_dict))

        result = _page_has_text(pdf.pages[0])

        assert result is True

    def test_page_with_form_xobject_no_text(self, tmp_dir: Path) -> None:
        """Form XObject without text operators is not detected as text."""
        pdf = new_pdf()

        # Form XObject with only graphics
        form_stream = pdf.make_stream(b"q 1 0 0 1 0 0 cm 0 0 100 100 re f Q")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        content = pdf.make_stream(b"q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        pdf.pages.append(pikepdf.Page(page_dict))

        result = _page_has_text(pdf.pages[0])

        assert result is False

    def test_needs_ocr_detects_text_in_form_xobject(self, tmp_dir: Path) -> None:
        """needs_ocr returns False when text exists inside Form XObjects."""
        pdf = new_pdf()

        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )

        # Image XObject
        image_data = b"\x80"
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceGray
        image_stream[Name.BitsPerComponent] = 8

        # Form XObject with OCR text layer
        form_stream = pdf.make_stream(b"BT /F1 12 Tf 100 700 Td (OCR text) Tj ET")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 612, 792])
        form_stream[Name.Resources] = Dictionary(Font=Dictionary(F1=font_dict))

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream, Fm0=form_stream)),
        )
        content = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q /Fm0 Do")
        page_dict[Name.Contents] = content
        pdf.pages.append(pikepdf.Page(page_dict))

        # Page has images AND text (in Form XObject), so should NOT need OCR
        result = needs_ocr(pdf)

        assert result is False


class TestApplyOcr:
    """Tests for apply_ocr."""

    def test_apply_ocr_raises_when_not_available(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """OCRError when OCR is not installed."""
        output_path = tmp_dir / "output.pdf"

        with patch("pdftopdfa.ocr.HAS_OCR", False):
            with pytest.raises(OCRError, match="OCR not available"):
                apply_ocr(sample_pdf, output_path, ["deu"])

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_calls_ocrmypdf(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr calls ocrmypdf.ocr with correct default parameters."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"])

        mock_ocrmypdf.ocr.assert_called_once_with(
            sample_pdf,
            output_path,
            language=["eng"],
            output_type="pdf",
            rasterizer="pypdfium",
            plugins=["pdftopdfa.ocr_rotation_fix"],
            **OCR_SETTINGS[OcrQuality.DEFAULT],
        )

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    @patch("pdftopdfa.ocr.EncryptedPdfError", Exception)
    def test_apply_ocr_handles_encrypted_pdf(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """EncryptedPdfError is converted to OCRError."""
        output_path = tmp_dir / "output.pdf"

        # Simulate EncryptedPdfError
        mock_ocrmypdf.ocr.side_effect = Exception("encrypted")

        with patch("pdftopdfa.ocr.EncryptedPdfError", Exception):
            # Since we patch Exception as EncryptedPdfError, it gets caught
            # but handled as a general error
            with pytest.raises(OCRError, match="OCR failed"):
                apply_ocr(sample_pdf, output_path, ["deu"])

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    @patch("pdftopdfa.ocr.shutil.copy2")
    def test_apply_ocr_handles_prior_ocr(
        self,
        mock_copy: MagicMock,
        mock_ocrmypdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """PriorOcrFoundError leads to copying the file."""
        output_path = tmp_dir / "output.pdf"

        # Create mock exception
        class MockPriorOcrFoundError(Exception):
            pass

        # Patch the exception class
        with patch("pdftopdfa.ocr.PriorOcrFoundError", MockPriorOcrFoundError):
            mock_ocrmypdf.ocr.side_effect = MockPriorOcrFoundError()

            result = apply_ocr(sample_pdf, output_path, ["deu"])

            mock_copy.assert_called_once_with(sample_pdf, output_path)
            assert result == output_path

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_returns_output_path(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr returns the output path."""
        output_path = tmp_dir / "output.pdf"

        result = apply_ocr(sample_pdf, output_path, ["deu"])

        assert result == output_path

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_default_language(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr uses English as default language."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["language"] == ["eng"]

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_multi_language(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr supports multiple languages."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["deu", "eng"])

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["language"] == ["deu", "eng"]

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_blank_exception_uses_exception_type(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Exceptions without a message still produce a useful OCRError."""
        output_path = tmp_dir / "output.pdf"
        mock_ocrmypdf.ocr.side_effect = RuntimeError()

        with pytest.raises(OCRError, match=r"OCR failed: RuntimeError"):
            apply_ocr(sample_pdf, output_path, ["deu"])

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    @patch("pdftopdfa.ocr.MissingDependencyError", Exception)
    def test_apply_ocr_handles_missing_dependency(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """MissingDependencyError is converted to OCRError preserving the message."""
        output_path = tmp_dir / "output.pdf"

        # Simulate MissingDependencyError
        class MockMissingDependencyError(Exception):
            pass

        with patch("pdftopdfa.ocr.MissingDependencyError", MockMissingDependencyError):
            mock_ocrmypdf.ocr.side_effect = MockMissingDependencyError(
                "tesseract is not installed"
            )

            with pytest.raises(OCRError, match="tesseract is not installed"):
                apply_ocr(sample_pdf, output_path, ["deu"])

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_tesseract_path_modifies_path(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """TESSERACT_PATH prepends its directory to PATH during OCR."""
        output_path = tmp_dir / "output.pdf"
        original_path = os.environ.get("PATH", "")

        captured_path = {}

        def capture_path(*args: object, **kwargs: object) -> None:
            captured_path["during"] = os.environ.get("PATH", "")

        mock_ocrmypdf.ocr.side_effect = capture_path

        with patch.dict(
            "os.environ",
            {"TESSERACT_PATH": "/opt/tesseract/bin/tesseract"},
        ):
            apply_ocr(sample_pdf, output_path, ["eng"])

        expected_dir = str(Path("/opt/tesseract/bin/tesseract").parent)
        assert captured_path["during"].startswith(expected_dir + os.pathsep)
        # PATH is restored after the call
        assert os.environ.get("PATH", "") == original_path

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_tesseract_path_accepts_directory(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_path: Path
    ) -> None:
        """TESSERACT_PATH pointing to a directory uses it directly (not its parent)."""
        output_path = tmp_path / "output.pdf"
        tesseract_dir = tmp_path / "tesseract" / "bin"
        tesseract_dir.mkdir(parents=True)

        captured_path = {}

        def capture_path(*args: object, **kwargs: object) -> None:
            captured_path["during"] = os.environ.get("PATH", "")

        mock_ocrmypdf.ocr.side_effect = capture_path

        with patch.dict(
            "os.environ",
            {"TESSERACT_PATH": str(tesseract_dir)},
        ):
            apply_ocr(sample_pdf, output_path, ["eng"])

        assert captured_path["during"].startswith(str(tesseract_dir) + os.pathsep)

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_tesseract_path_not_set_leaves_path_unchanged(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """PATH remains unchanged when TESSERACT_PATH is not set."""
        output_path = tmp_dir / "output.pdf"
        original_path = os.environ.get("PATH", "")

        captured_path = {}

        def capture_path(*args: object, **kwargs: object) -> None:
            captured_path["during"] = os.environ.get("PATH", "")

        mock_ocrmypdf.ocr.side_effect = capture_path

        with patch.dict("os.environ", {}, clear=False):
            # Ensure TESSERACT_PATH is not set
            os.environ.pop("TESSERACT_PATH", None)
            apply_ocr(sample_pdf, output_path, ["eng"])

        assert captured_path["during"] == original_path

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_tesseract_path_restored_on_error(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """PATH is restored even when ocrmypdf raises an exception."""
        output_path = tmp_dir / "output.pdf"
        original_path = os.environ.get("PATH", "")

        mock_ocrmypdf.ocr.side_effect = RuntimeError("OCR crash")

        with patch.dict(
            "os.environ",
            {"TESSERACT_PATH": "/opt/tesseract/bin/tesseract"},
        ):
            with pytest.raises(OCRError):
                apply_ocr(sample_pdf, output_path, ["eng"])

        assert os.environ.get("PATH", "") == original_path


class TestOcrQuality:
    """Tests for OCR quality presets."""

    def test_ocr_quality_enum_values(self) -> None:
        """OcrQuality enum has the expected values."""
        assert OcrQuality.FAST.value == "fast"
        assert OcrQuality.DEFAULT.value == "default"
        assert OcrQuality.BEST.value == "best"

    def test_ocr_quality_enum_from_string(self) -> None:
        """OcrQuality can be created from string values."""
        assert OcrQuality("fast") is OcrQuality.FAST
        assert OcrQuality("default") is OcrQuality.DEFAULT
        assert OcrQuality("best") is OcrQuality.BEST

    def test_ocr_settings_has_all_presets(self) -> None:
        """OCR_SETTINGS contains entries for all quality presets."""
        for quality in OcrQuality:
            assert quality in OCR_SETTINGS

    def test_ocr_settings_fast_preset(self) -> None:
        """Fast preset uses minimal parameters."""
        settings = OCR_SETTINGS[OcrQuality.FAST]
        assert settings["skip_text"] is True
        assert settings["deskew"] is False
        assert settings["rotate_pages"] is False
        assert settings["optimize"] == 0
        assert settings["progress_bar"] is False
        assert "oversample" not in settings
        assert "clean" not in settings

    def test_ocr_settings_default_preset(self) -> None:
        """Default preset uses quality parameters without visual changes."""
        settings = OCR_SETTINGS[OcrQuality.DEFAULT]
        assert settings["skip_text"] is True
        assert settings["deskew"] is False
        assert settings["rotate_pages"] is False
        assert settings["oversample"] == 300
        assert settings["optimize"] == 0
        assert settings["progress_bar"] is False
        assert "clean" not in settings

    def test_ocr_settings_best_preset(self) -> None:
        """Best preset uses all quality parameters including visual changes."""
        settings = OCR_SETTINGS[OcrQuality.BEST]
        assert settings["skip_text"] is True
        assert settings["deskew"] is True
        assert settings["rotate_pages"] is True
        assert settings["rotate_pages_threshold"] == 5.0
        assert settings["oversample"] == 200
        assert settings["optimize"] == 0
        assert settings["progress_bar"] is False
        assert "clean" not in settings

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_fast_quality(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr with FAST quality passes correct parameters."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.FAST)

        mock_ocrmypdf.ocr.assert_called_once_with(
            sample_pdf,
            output_path,
            language=["eng"],
            output_type="pdf",
            rasterizer="pypdfium",
            **OCR_SETTINGS[OcrQuality.FAST],
        )

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_default_quality(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr with DEFAULT quality passes correct parameters."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.DEFAULT)

        mock_ocrmypdf.ocr.assert_called_once_with(
            sample_pdf,
            output_path,
            language=["eng"],
            output_type="pdf",
            rasterizer="pypdfium",
            plugins=["pdftopdfa.ocr_rotation_fix"],
            **OCR_SETTINGS[OcrQuality.DEFAULT],
        )

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    @patch("pdftopdfa.ocr._normalize_best_quality_skipped_text_pages")
    def test_apply_ocr_best_quality(
        self,
        mock_normalize_pages: MagicMock,
        mock_ocrmypdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
    ) -> None:
        """apply_ocr with BEST quality passes correct parameters."""
        output_path = tmp_dir / "output.pdf"
        mock_ocrmypdf.ocr.side_effect = lambda *args, **kwargs: output_path.touch()

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST)

        mock_ocrmypdf.ocr.assert_called_once_with(
            sample_pdf,
            output_path,
            language=["eng"],
            output_type="pdf",
            rasterizer="pypdfium",
            plugins=["pdftopdfa.ocr_rotation_fix"],
            **OCR_SETTINGS[OcrQuality.BEST],
        )
        mock_normalize_pages.assert_called_once_with(output_path)

    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_default_quality_when_omitted(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr uses DEFAULT quality when quality parameter is omitted."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"])

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        expected = OCR_SETTINGS[OcrQuality.DEFAULT]
        for key, value in expected.items():
            assert call_kwargs[key] == value


class TestApplyOcrForce:
    """Tests for apply_ocr(force=True) behaviour."""

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_force_sets_redo_ocr(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """force=True sets redo_ocr=True in ocrmypdf call."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], force=True)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["redo_ocr"] is True

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_force_removes_skip_text(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """force=True removes skip_text from ocrmypdf call."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], force=True)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert "skip_text" not in call_kwargs

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_no_force_uses_skip_text(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """force=False (default) still uses skip_text=True."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"])

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["skip_text"] is True
        assert "redo_ocr" not in call_kwargs

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_force_with_best_quality(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """force=True with BEST quality removes redo_ocr conflicts only."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST, force=True)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["redo_ocr"] is True
        assert "skip_text" not in call_kwargs
        assert "deskew" not in call_kwargs
        assert call_kwargs["rotate_pages"] is True

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_force_removes_all_redo_ocr_incompatible_options(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """force=True strips all ocrmypdf options incompatible with redo_ocr."""
        output_path = tmp_dir / "output.pdf"
        forced_best_settings = {
            **OCR_SETTINGS[OcrQuality.BEST],
            "clean_final": True,
            "remove_background": True,
        }

        with patch.dict(
            "pdftopdfa.ocr.OCR_SETTINGS",
            {OcrQuality.BEST: forced_best_settings},
            clear=False,
        ):
            apply_ocr(
                sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST, force=True
            )

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert "deskew" not in call_kwargs
        assert "clean_final" not in call_kwargs
        assert "remove_background" not in call_kwargs
        assert call_kwargs["rotate_pages"] is True

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_force_logs_removed_incompatible_options(
        self,
        mock_ocrmypdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """force=True logs when redo_ocr-incompatible options are disabled."""
        output_path = tmp_dir / "output.pdf"

        with caplog.at_level(logging.INFO, logger="pdftopdfa.ocr"):
            apply_ocr(
                sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST, force=True
            )

        assert (
            "force=True disables redo_ocr-incompatible OCR options: deskew"
            in caplog.text
        )


class TestOpenCVPlugin:
    """Tests for OpenCV preprocessing plugin integration."""

    def test_preprocess_qualities_contains_default_and_best(self) -> None:
        """_PREPROCESS_QUALITIES includes DEFAULT and BEST."""
        assert OcrQuality.DEFAULT in _PREPROCESS_QUALITIES
        assert OcrQuality.BEST in _PREPROCESS_QUALITIES
        assert OcrQuality.FAST not in _PREPROCESS_QUALITIES

    @patch("pdftopdfa.ocr.HAS_OPENCV", True)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_uses_opencv_plugin(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Plugins kwarg is set when OpenCV is available and quality supports it."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.DEFAULT)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["plugins"] == [
            "pdftopdfa.ocr_rotation_fix",
            "pdftopdfa.ocr_preprocess",
        ]

    @patch("pdftopdfa.ocr.HAS_OPENCV", True)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_best_uses_opencv_plugin(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """BEST quality also uses the OpenCV plugin."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["plugins"] == [
            "pdftopdfa.ocr_rotation_fix",
            "pdftopdfa.ocr_preprocess",
        ]

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_no_opencv_no_plugin(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Rotation fix stays enabled even when OpenCV preprocessing is unavailable."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.DEFAULT)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert call_kwargs["plugins"] == ["pdftopdfa.ocr_rotation_fix"]

    @patch("pdftopdfa.ocr.HAS_OPENCV", True)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_fast_no_plugin(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """FAST quality never uses the OpenCV plugin."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.FAST)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert "plugins" not in call_kwargs

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_no_opencv_logs_warning(
        self,
        mock_ocrmypdf: MagicMock,
        sample_pdf: Path,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning is logged when OpenCV is not available."""
        output_path = tmp_dir / "output.pdf"

        with caplog.at_level(logging.WARNING, logger="pdftopdfa.ocr"):
            apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.DEFAULT)

        assert "OpenCV not available" in caplog.text


class TestFilterOcrImage:
    """Tests for the filter_ocr_image plugin hook."""

    def test_filter_ocr_image_color_input(self) -> None:
        """Color image is converted to grayscale and binarized."""
        from pdftopdfa.ocr_preprocess import filter_ocr_image

        # Create a color image (RGB)
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = filter_ocr_image(page=None, image=img)

        assert isinstance(result, Image.Image)
        assert result.mode == "L"  # Grayscale output
        assert result.size == (100, 100)

    def test_filter_ocr_image_grayscale_input(self) -> None:
        """Grayscale image is processed without color conversion."""
        from pdftopdfa.ocr_preprocess import filter_ocr_image

        img = Image.new("L", (100, 100), color=128)
        result = filter_ocr_image(page=None, image=img)

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == (100, 100)

    def test_filter_ocr_image_binarizes_output(self) -> None:
        """Output image contains only black and white pixels."""
        import numpy as np

        from pdftopdfa.ocr_preprocess import filter_ocr_image

        img = Image.new("L", (100, 100), color=128)
        result = filter_ocr_image(page=None, image=img)

        pixels = np.array(result)
        unique_values = set(np.unique(pixels))
        assert unique_values <= {0, 255}


class TestBestQualityTextRotationNormalization:
    """Tests for post-OCR normalization of skipped text-page rotations."""

    def test_parse_tesseract_osd(self) -> None:
        """OSD output is parsed into rotation and confidence."""
        result = _parse_tesseract_osd(
            "Rotate: 180\nOrientation confidence: 12.5\nScript: Latin\n"
        )

        assert result == _OrientationResult(rotate=180, confidence=12.5)

    def test_should_clear_page_rotate_when_cleared_preview_is_better(self) -> None:
        """Clearing /Rotate is preferred when it removes the needed correction."""
        current = _OrientationResult(rotate=180, confidence=8.0)
        cleared = _OrientationResult(rotate=0, confidence=9.5)

        assert _should_clear_page_rotate(180, current, cleared) is True

    def test_should_not_clear_page_rotate_without_improvement(self) -> None:
        """Pages stay untouched when the cleared preview is not better."""
        current = _OrientationResult(rotate=0, confidence=9.0)
        cleared = _OrientationResult(rotate=180, confidence=10.0)

        assert _should_clear_page_rotate(180, current, cleared) is False

    @patch("pdftopdfa.ocr._run_tesseract_orientation")
    @patch("pdftopdfa.ocr._render_pdf_page_preview")
    @patch("pdftopdfa.ocr._write_single_page_with_rotate")
    def test_normalize_best_quality_text_page_rotations_clears_rotate(
        self,
        mock_write_single_page: MagicMock,
        mock_render_preview: MagicMock,
        mock_run_orientation: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """A text-only page with better orientation after clearing is normalized."""
        pdf_path = tmp_dir / "rotated.pdf"

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            page.Rotate = 180
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
            page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            page.obj[Name.Contents] = pdf.make_stream(
                b"BT /F1 12 Tf 100 700 Td (Rotated text) Tj ET"
            )
            pdf.save(pdf_path)

        mock_write_single_page.side_effect = lambda *args, **kwargs: kwargs[
            "output_path"
        ]
        mock_run_orientation.side_effect = [
            _OrientationResult(rotate=180, confidence=8.0),
            _OrientationResult(rotate=0, confidence=9.0),
        ]

        changed_pages = _normalize_best_quality_text_page_rotations(pdf_path)

        assert changed_pages == [1]
        with Pdf.open(pdf_path) as pdf:
            page = pdf.pages[0]
            assert int(page.obj.get("/Rotate", 0)) == 0
            assert [float(value) for value in page.obj["/MediaBox"]] == pytest.approx(
                [0.0, 0.0, 595.0, 842.0]
            )

    @patch("pdftopdfa.ocr._run_tesseract_orientation")
    @patch("pdftopdfa.ocr._render_pdf_page_preview")
    @patch("pdftopdfa.ocr._write_single_page_with_rotate")
    def test_normalize_best_quality_text_page_rotations_preserves_visible_a4(
        self,
        mock_write_single_page: MagicMock,
        mock_render_preview: MagicMock,
        mock_run_orientation: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Clearing /Rotate keeps the original page box geometry unchanged."""
        pdf_path = tmp_dir / "rotated_visible_a4.pdf"
        original_box = [0.0, 0.0, 842.0, 595.0]

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(842.0, 595.0))
            page.Rotate = 270
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
            page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            page.obj[Name.CropBox] = Array(original_box)
            page.obj[Name.TrimBox] = Array(original_box)
            page.obj[Name.Contents] = pdf.make_stream(
                b"BT /F1 12 Tf 100 450 Td (Portrait A4) Tj ET"
            )
            pdf.save(pdf_path)

        mock_write_single_page.side_effect = lambda *args, **kwargs: kwargs[
            "output_path"
        ]
        mock_run_orientation.side_effect = [
            _OrientationResult(rotate=270, confidence=8.0),
            _OrientationResult(rotate=0, confidence=9.0),
        ]

        changed_pages = _normalize_best_quality_text_page_rotations(pdf_path)

        assert changed_pages == [1]
        with Pdf.open(pdf_path) as pdf:
            page = pdf.pages[0]
            assert int(page.obj.get("/Rotate", 0)) == 0
            assert [float(value) for value in page.obj["/MediaBox"]] == pytest.approx(
                original_box
            )
            assert [float(value) for value in page.obj["/CropBox"]] == pytest.approx(
                original_box
            )
            assert [float(value) for value in page.obj["/TrimBox"]] == pytest.approx(
                original_box
            )

    @patch("pdftopdfa.ocr._run_tesseract_orientation")
    @patch("pdftopdfa.ocr._render_pdf_page_preview")
    @patch("pdftopdfa.ocr._write_single_page_with_rotate")
    def test_normalize_best_quality_text_page_rotations_keeps_rotate_when_needed(
        self,
        mock_write_single_page: MagicMock,
        mock_render_preview: MagicMock,
        mock_run_orientation: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Rotation is preserved when clearing it would not improve orientation."""
        pdf_path = tmp_dir / "rotated_keep.pdf"

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            page.Rotate = 90
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
            page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            page.obj[Name.Contents] = pdf.make_stream(
                b"BT /F1 12 Tf 100 700 Td (Keep rotate) Tj ET"
            )
            pdf.save(pdf_path)

        mock_write_single_page.side_effect = lambda *args, **kwargs: kwargs[
            "output_path"
        ]
        mock_run_orientation.side_effect = [
            _OrientationResult(rotate=0, confidence=9.0),
            _OrientationResult(rotate=90, confidence=9.5),
        ]

        changed_pages = _normalize_best_quality_text_page_rotations(pdf_path)

        assert changed_pages == []
        with Pdf.open(pdf_path) as pdf:
            assert int(pdf.pages[0].obj.get("/Rotate", 0)) == 90

    def test_detect_consistent_text_skew(self, tmp_dir: Path) -> None:
        """Consistent small text-matrix skew is detected."""
        pdf_path = tmp_dir / "skew_detect.pdf"

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
            page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            page.obj[Name.Contents] = pdf.make_stream(
                b"BT /F1 12 Tf 12.133 1.0125 -1.0125 12.133 100 700 Tm (A) Tj "
                b"12.133 1.0125 -1.0125 12.133 100 680 Tm (B) Tj ET"
            )
            pdf.save(pdf_path)

        with Pdf.open(pdf_path) as pdf:
            angle = _detect_consistent_text_skew(pdf.pages[0])

        assert angle is not None
        assert angle == pytest.approx(4.77, abs=0.1)

    def test_normalize_best_quality_text_page_skew(self, tmp_dir: Path) -> None:
        """Text-only pages with dominant skew keep their original page size."""
        pdf_path = tmp_dir / "skew_page.pdf"
        original_box = [0.0, 0.0, 595.0, 842.0]

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
            page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            page.obj[Name.CropBox] = Array(original_box)
            page.obj[Name.TrimBox] = Array(original_box)
            page.obj[Name.Contents] = pdf.make_stream(
                b"BT /F1 12 Tf "
                b"12.133 1.0125 -1.0125 12.133 100 700 Tm (A) Tj "
                b"12.133 1.0125 -1.0125 12.133 100 680 Tm (B) Tj ET"
            )
            pdf.save(pdf_path)

        normalized = _normalize_best_quality_text_page_skew(pdf_path)

        assert normalized
        with Pdf.open(pdf_path) as pdf:
            page = pdf.pages[0]
            contents = page.obj["/Contents"]
            assert isinstance(contents, Array)
            prefix = bytes(contents[0].read_bytes()).decode("ascii")
            assert " cm" in prefix
            assert [float(value) for value in page.obj["/MediaBox"]] == pytest.approx(
                original_box
            )
            assert [float(value) for value in page.obj["/CropBox"]] == pytest.approx(
                original_box
            )
            assert [float(value) for value in page.obj["/TrimBox"]] == pytest.approx(
                original_box
            )

    @patch("pdftopdfa.ocr._normalize_best_quality_text_page_rotations")
    @patch("pdftopdfa.ocr._normalize_best_quality_text_page_skew")
    def test_normalize_best_quality_skipped_text_pages_runs_both_steps(
        self,
        mock_normalize_skew: MagicMock,
        mock_normalize_rotations: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Best-quality skipped text normalization runs rotation then skew."""
        pdf_path = tmp_dir / "combined.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        _normalize_best_quality_skipped_text_pages(pdf_path)

        mock_normalize_rotations.assert_called_once_with(pdf_path)
        mock_normalize_skew.assert_called_once_with(pdf_path)


class TestVisiblePageRotationFix:
    """Tests for visible-page rotation normalization during OCR."""

    def _make_page_context(
        self,
        *,
        width_points: float,
        height_points: float,
        rotation: int,
    ) -> SimpleNamespace:
        media_box = [0.0, 0.0, width_points, height_points]
        pageinfo = SimpleNamespace(
            width_inches=width_points / 72.0,
            height_inches=height_points / 72.0,
            rotation=rotation,
            mediabox=media_box,
            cropbox=media_box,
            trimbox=media_box,
            artbox=media_box,
            bleedbox=media_box,
        )
        return SimpleNamespace(pageinfo=pageinfo)

    @pytest.mark.parametrize(
        ("page_size", "image_size", "expected_swap"),
        [
            ((595.0, 842.0), (300, 200), True),
            ((842.0, 595.0), (200, 300), True),
            ((842.0, 595.0), (300, 200), False),
        ],
    )
    def test_should_swap_visible_page_axis(
        self,
        page_size: tuple[float, float],
        image_size: tuple[int, int],
        expected_swap: bool,
    ) -> None:
        """Axis swapping follows the actual rendered image orientation."""
        assert (
            _should_swap_visible_page_axis(
                page_size[0],
                page_size[1],
                image_size[0],
                image_size[1],
            )
            is expected_swap
        )

    def test_filter_pdf_page_fixes_rotate_270_plus_180_regression(
        self, tmp_dir: Path
    ) -> None:
        """A /Rotate=270 page stays in the rendered orientation after 180° OCR fix."""
        image_path = tmp_dir / "rendered.png"
        output_pdf = tmp_dir / "visible.pdf"

        # Represents the rendered bitmap after OCRmyPDF applied /Rotate=270 and
        # then an additional 180 degree autorotation correction.
        Image.new("RGB", (300, 200), color="white").save(image_path)
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(595.0, 842.0))
            pdf.save(output_pdf)

        page_context = self._make_page_context(
            width_points=842.0,
            height_points=595.0,
            rotation=270,
        )

        filter_pdf_page(page_context, image_path, output_pdf)

        with Pdf.open(output_pdf) as pdf:
            mediabox = [float(value) for value in pdf.pages[0].mediabox]

        assert mediabox == [0.0, 0.0, 842.0, 595.0]

    @pytest.mark.parametrize(
        ("page_size", "image_size", "expected_mediabox"),
        [
            ((595.0, 842.0), (300, 200), [0.0, 0.0, 842.0, 595.0]),
            ((842.0, 595.0), (200, 300), [0.0, 0.0, 595.0, 842.0]),
        ],
    )
    def test_filter_pdf_page_preserves_90_and_270_autorotate_cases(
        self,
        tmp_dir: Path,
        page_size: tuple[float, float],
        image_size: tuple[int, int],
        expected_mediabox: list[float],
    ) -> None:
        """Regular 90°/270° autorotation still produces the correct page geometry."""
        image_path = tmp_dir / f"rendered_{image_size[0]}x{image_size[1]}.png"
        output_pdf = tmp_dir / f"visible_{page_size[0]}x{page_size[1]}.pdf"

        Image.new("RGB", image_size, color="white").save(image_path)
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=page_size)
            pdf.save(output_pdf)

        page_context = self._make_page_context(
            width_points=page_size[0],
            height_points=page_size[1],
            rotation=0,
        )

        filter_pdf_page(page_context, image_path, output_pdf)

        with Pdf.open(output_pdf) as pdf:
            mediabox = [float(value) for value in pdf.pages[0].mediabox]

        assert mediabox == expected_mediabox


class TestPypdfiumRotationFix:
    """Tests for pypdfium raster rotation composition."""

    def test_compose_page_rotation_preserves_existing_rotate(self) -> None:
        """The requested OCR correction is composed with the existing /Rotate."""
        assert _compose_page_rotation(270, 90) == 180
        assert _compose_page_rotation(270, 180) == 90
        assert _compose_page_rotation(270, 270) == 0
        assert _compose_page_rotation(270, 0) == 270
        assert _compose_page_rotation(0, 90) == 270

    @patch("ocrmypdf.builtin_plugins.pypdfium.rasterize_pdf_page")
    def test_rasterize_pdf_page_composes_existing_rotate_for_pypdfium(
        self, mock_rasterize: MagicMock, tmp_dir: Path
    ) -> None:
        """A /Rotate=270 page plus 180° OCR correction is rendered as /Rotate=90."""
        input_pdf = tmp_dir / "input.pdf"
        output_png = tmp_dir / "output.png"

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            page.Rotate = 270
            pdf.save(input_pdf)

        captured: dict[str, Path] = {}

        def capture_temp_pdf(*args, **kwargs):
            captured["temp_pdf"] = Path(args[0])
            return output_png

        mock_rasterize.side_effect = capture_temp_pdf
        options = SimpleNamespace(rasterizer="pypdfium", keep_temporary_files=True)

        result = rasterize_pdf_page(
            input_pdf,
            output_png,
            "png16m",
            SimpleNamespace(),
            1,
            None,
            180,
            False,
            True,
            options,
            False,
        )

        assert result == output_png
        assert mock_rasterize.call_args.args[6] == 0
        with Pdf.open(captured["temp_pdf"]) as pdf:
            assert int(pdf.pages[0].obj.get("/Rotate", 0)) == 90

    @patch("ocrmypdf.builtin_plugins.pypdfium.rasterize_pdf_page")
    def test_rasterize_pdf_page_applies_90_degree_correction_against_rotate(
        self, mock_rasterize: MagicMock, tmp_dir: Path
    ) -> None:
        """A /Rotate=270 page plus 90° OCR correction is rendered as /Rotate=180."""
        input_pdf = tmp_dir / "input_90.pdf"
        output_png = tmp_dir / "output_90.png"

        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(595.0, 842.0))
            page.Rotate = 270
            pdf.save(input_pdf)

        captured: dict[str, Path] = {}

        def capture_temp_pdf(*args, **kwargs):
            captured["temp_pdf"] = Path(args[0])
            return output_png

        mock_rasterize.side_effect = capture_temp_pdf
        options = SimpleNamespace(rasterizer="pypdfium", keep_temporary_files=True)

        result = rasterize_pdf_page(
            input_pdf,
            output_png,
            "png16m",
            SimpleNamespace(),
            1,
            None,
            90,
            False,
            True,
            options,
            False,
        )

        assert result == output_png
        assert mock_rasterize.call_args.args[6] == 0
        with Pdf.open(captured["temp_pdf"]) as pdf:
            assert int(pdf.pages[0].obj.get("/Rotate", 0)) == 180

    @patch("ocrmypdf.builtin_plugins.pypdfium.rasterize_pdf_page")
    def test_rasterize_pdf_page_skips_when_no_existing_rotate(
        self, mock_rasterize: MagicMock, tmp_dir: Path
    ) -> None:
        """Pages without /Rotate fall back to OCRmyPDF's default pypdfium hook."""
        input_pdf = tmp_dir / "input.pdf"
        output_png = tmp_dir / "output.png"

        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(595.0, 842.0))
            pdf.save(input_pdf)

        options = SimpleNamespace(rasterizer="pypdfium", keep_temporary_files=False)

        result = rasterize_pdf_page(
            input_pdf,
            output_png,
            "png16m",
            SimpleNamespace(),
            1,
            None,
            180,
            False,
            True,
            options,
            False,
        )

        assert result is None
        mock_rasterize.assert_not_called()
