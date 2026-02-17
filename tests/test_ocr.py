# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for ocr.py."""

import logging
import os
from pathlib import Path
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
    _page_has_images,
    _page_has_text,
    apply_ocr,
    is_ocr_available,
    needs_ocr,
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
        assert settings["oversample"] == 300
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
            **OCR_SETTINGS[OcrQuality.DEFAULT],
        )

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_best_quality(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """apply_ocr with BEST quality passes correct parameters."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.BEST)

        mock_ocrmypdf.ocr.assert_called_once_with(
            sample_pdf,
            output_path,
            language=["eng"],
            output_type="pdf",
            rasterizer="pypdfium",
            **OCR_SETTINGS[OcrQuality.BEST],
        )

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
        assert call_kwargs["plugins"] == ["pdftopdfa.ocr_preprocess"]

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
        assert call_kwargs["plugins"] == ["pdftopdfa.ocr_preprocess"]

    @patch("pdftopdfa.ocr.HAS_OPENCV", False)
    @patch("pdftopdfa.ocr.HAS_OCR", True)
    @patch("pdftopdfa.ocr.ocrmypdf")
    def test_apply_ocr_no_opencv_no_plugin(
        self, mock_ocrmypdf: MagicMock, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """No plugins kwarg when OpenCV is not available."""
        output_path = tmp_dir / "output.pdf"

        apply_ocr(sample_pdf, output_path, ["eng"], quality=OcrQuality.DEFAULT)

        call_kwargs = mock_ocrmypdf.ocr.call_args[1]
        assert "plugins" not in call_kwargs

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
