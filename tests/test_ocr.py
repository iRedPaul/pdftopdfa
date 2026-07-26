# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the public OCR integration."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Dictionary, Name, Pdf
from PIL import Image

from pdftopdfa.exceptions import OCRError
from pdftopdfa.ocr import (
    _finalize_ocr_output,
    _ocr_form_names,
    _page_has_images,
    _page_has_text,
    _strip_invisible_text_from_form,
    apply_ocr,
    is_ocr_available,
    needs_ocr,
    validate_ocr_languages,
)
from pdftopdfa.ocr_rotation_fix import (
    _should_swap_visible_page_axis,
    filter_pdf_page,
)


@pytest.fixture(autouse=True)
def _mock_paddle_orientation():
    """Keep OCR unit tests independent from the orientation model."""
    with patch("pdftopdfa.ocr.normalize_pdf_orientation") as mock_normalize:

        def copy_input(input_path: Path, output_path: Path) -> list[object]:
            shutil.copy2(input_path, output_path)
            return []

        mock_normalize.side_effect = copy_input
        yield mock_normalize


@pytest.fixture
def model_dirs(tmp_dir: Path) -> tuple[Path, Path]:
    """Return explicit model paths for integration-boundary tests."""
    return tmp_dir / "detection", tmp_dir / "recognition"


@pytest.fixture
def validate_models(model_dirs: tuple[Path, Path]):
    """Bypass artifact hashing in tests that exercise only OCRmyPDF options."""
    with patch(
        "pdftopdfa.ocr_paddle.validate_model_directories",
        return_value=model_dirs,
    ) as mock_validate:
        yield mock_validate


def _copy_ocr_input(input_path: Path, output_path: Path, **_kwargs: object) -> None:
    """Model OCRmyPDF's output contract in option-boundary tests."""
    shutil.copy2(input_path, output_path)


class TestOcrDetection:
    """Tests for public OCR availability and page analysis helpers."""

    def test_is_ocr_available_returns_bool(self) -> None:
        assert isinstance(is_ocr_available(), bool)

    def test_empty_pdf_does_not_need_ocr(self, empty_pdf_obj: Pdf) -> None:
        assert needs_ocr(empty_pdf_obj) is False

    def test_text_pdf_does_not_need_ocr(self, pdf_with_text_obj: Pdf) -> None:
        assert needs_ocr(pdf_with_text_obj) is False

    def test_image_pdf_needs_ocr(self, pdf_with_image_obj: Pdf) -> None:
        assert needs_ocr(pdf_with_image_obj) is True

    def test_threshold_applies_to_page_ratio(self, tmp_dir: Path) -> None:
        pdf = new_pdf()
        text_page = pdf.add_blank_page(page_size=(100, 100))
        text_page.obj[Name.Contents] = pdf.make_stream(b"BT (Text) Tj ET")
        image_page = pdf.add_blank_page(page_size=(100, 100))
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 8
        image_page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Im0=image))
        image_page.obj[Name.Contents] = pdf.make_stream(
            b"q 100 0 0 100 0 0 cm /Im0 Do Q"
        )

        assert needs_ocr(pdf, threshold=0.5) is True
        assert needs_ocr(pdf, threshold=0.6) is False

    def test_page_image_and_text_detection(
        self,
        pdf_with_image_obj: Pdf,
        pdf_with_text_obj: Pdf,
    ) -> None:
        assert _page_has_images(pdf_with_image_obj.pages[0]) is True
        assert _page_has_text(pdf_with_image_obj.pages[0]) is False
        assert _page_has_images(pdf_with_text_obj.pages[0]) is False
        assert _page_has_text(pdf_with_text_obj.pages[0]) is True


class TestOcrLanguageMetadata:
    """Tests for catalog language assignment after OCR."""

    def test_sets_primary_language_when_missing(self, tmp_dir: Path) -> None:
        path = tmp_dir / "language.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(path)

        _finalize_ocr_output(path, ["de", "en"], _ocr_form_names(path))

        with Pdf.open(path) as pdf:
            assert str(pdf.Root[Name.Lang]) == "de"

    def test_preserves_existing_language(self, tmp_dir: Path) -> None:
        path = tmp_dir / "language.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.Root[Name.Lang] = "fr-FR"
            pdf.save(path)

        _finalize_ocr_output(path, ["de"], _ocr_form_names(path))

        with Pdf.open(path) as pdf:
            assert str(pdf.Root[Name.Lang]) == "fr-FR"

    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("ch", "zh-Hans"),
            ("chinese_cht", "zh-Hant"),
            ("french", "fr"),
            ("german", "de"),
            ("japan", "ja"),
            ("rs_latin", "sr-Latn"),
        ],
    )
    def test_maps_paddle_alias_to_bcp47(
        self,
        tmp_dir: Path,
        language: str,
        expected: str,
    ) -> None:
        path = tmp_dir / f"{language}.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(path)

        _finalize_ocr_output(path, [language], _ocr_form_names(path))

        with Pdf.open(path) as pdf:
            assert str(pdf.Root[Name.Lang]) == expected


class TestRotatedOcrFormBoxes:
    """Tests for rotated OCR text-layer clipping repair."""

    @pytest.mark.parametrize("rotation", [90, 270])
    def test_swaps_ocr_form_box_axes(
        self,
        tmp_dir: Path,
        rotation: int,
    ) -> None:
        path = tmp_dir / f"rotated-{rotation}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(576, 432))
            page.obj[Name.Rotate] = rotation
            form = pdf.make_stream(b"")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 576, 432])
            xobjects = Dictionary()
            xobjects[Name("/OCR-pdf-0")] = form
            page.obj[Name.Resources] = Dictionary(XObject=xobjects)
            pdf.save(path)

        _finalize_ocr_output(path, ["en"], [frozenset()])

        with Pdf.open(path) as pdf:
            form = pdf.pages[0].Resources.XObject["/OCR-pdf-0"]
            assert [float(value) for value in form.BBox] == [0, 0, 432, 576]

    @pytest.mark.parametrize("inherited", [False, True])
    def test_leaves_preexisting_ocr_form_unchanged(
        self,
        tmp_dir: Path,
        inherited: bool,
    ) -> None:
        path = tmp_dir / f"rotated-inherited-{inherited}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(576, 432))
            form = pdf.make_stream(b"")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 576, 432])
            xobjects = Dictionary()
            xobjects[Name("/OCR-existing")] = form
            container = page.obj.Parent if inherited else page.obj
            container[Name.Rotate] = 90
            container[Name.Resources] = Dictionary(XObject=xobjects)
            if inherited:
                del page.obj[Name.Resources]
            pdf.save(path)

        existing_names = _ocr_form_names(path)
        _finalize_ocr_output(path, ["en"], existing_names)

        with Pdf.open(path) as pdf:
            form = pdf.pages[0].resources.XObject["/OCR-existing"]
            assert [float(value) for value in form.BBox] == [0, 0, 576, 432]


class TestLanguages:
    """Tests for the PP-OCRv6 language contract."""

    @pytest.mark.parametrize("languages", [["en"], ["de"], ["de", "en"]])
    def test_supported_languages(self, languages: list[str]) -> None:
        assert validate_ocr_languages(languages) == languages

    @pytest.mark.parametrize("languages", [[], ["eng"], ["deu"], ["unknown"]])
    def test_unsupported_languages(self, languages: list[str]) -> None:
        with pytest.raises(ValueError, match="OCR language|PaddleOCR"):
            validate_ocr_languages(languages)


@pytest.mark.parametrize(
    ("content", "operator"),
    [
        (b"BT 3 Tr (hidden) Tj ET", b"Tj"),
        (b"BT 3 Tr [(hid) 20 (den)] TJ ET", b"TJ"),
        (b"BT 3 Tr (hidden) ' ET", b"'"),
        (b'BT 3 Tr 1 2 (hidden) " ET', b'"'),
    ],
)
def test_invisible_form_cleanup_preserves_text_show_operator(
    content: bytes,
    operator: bytes,
) -> None:
    with Pdf.new() as pdf:
        form = pdf.make_stream(content)

        assert _strip_invisible_text_from_form(form) is True

        rewritten = form.read_bytes()
        assert b"hid" not in rewritten
        assert operator in rewritten
        assert b"3 Tr" in rewritten


class TestApplyOcr:
    """Tests for the fixed PaddleOCR/OCRmyPDF boundary."""

    def test_passes_fixed_offline_configuration(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output = tmp_dir / "output.pdf"
        detection, recognition = model_dirs

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            result = apply_ocr(
                sample_pdf,
                output,
                ["de", "en"],
                detection_model_dir=detection,
                recognition_model_dir=recognition,
            )

        assert result == output
        validate_models.assert_called_once_with(detection, recognition)
        mock_ocr.assert_called_once()
        args, kwargs = mock_ocr.call_args
        assert args == (sample_pdf, output)
        assert kwargs == {
            "language": ["de", "en"],
            "ocr_engine": "paddle",
            "pdf_renderer": "fpdf2",
            "rasterizer": "pypdfium",
            "output_type": "pdf",
            "oversample": 600,
            "optimize": 0,
            "jobs": 1,
            "skip_text": True,
            "deskew": False,
            "rotate_pages": False,
            "progress_bar": False,
            "plugins": [
                "pdftopdfa.ocr_paddle",
                "pdftopdfa.ocr_rotation_fix",
            ],
            "paddle_detection_model_dir": detection,
            "paddle_recognition_model_dir": recognition,
        }

    def test_defaults_to_english_metadata(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output = tmp_dir / "output.pdf"
        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                sample_pdf,
                output,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

        assert mock_ocr.call_args.kwargs["language"] == ["en"]
        with Pdf.open(output) as pdf:
            assert str(pdf.Root[Name.Lang]) == "en"

    def test_force_uses_redo_without_skip_text(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                ["en"],
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                force=True,
            )

        kwargs = mock_ocr.call_args.kwargs
        assert kwargs["redo_ocr"] is True
        assert "skip_text" not in kwargs

    def test_force_removes_only_invisible_text_from_existing_ocr_forms(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            hidden = pdf.make_stream(b"BT 3 Tr (old OCR) Tj ET")
            hidden[Name.Type] = Name.XObject
            hidden[Name.Subtype] = Name.Form
            hidden[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            visible = pdf.make_stream(
                b"1 0 0 rg 0 0 10 10 re f BT 0 Tr (visible text) Tj ET"
            )
            visible[Name.Type] = Name.XObject
            visible[Name.Subtype] = Name.Form
            visible[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            separate = pdf.make_stream(
                b"BT 3 Tr (separate hidden) Tj ET BT 0 Tr (separate visible) Tj ET"
            )
            separate[Name.Type] = Name.XObject
            separate[Name.Subtype] = Name.Form
            separate[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            visible_first = pdf.make_stream(
                b"BT 0 Tr (visible first) Tj 3 Tr (hidden last) Tj ET"
            )
            visible_first[Name.Type] = Name.XObject
            visible_first[Name.Subtype] = Name.Form
            visible_first[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            hidden_first = pdf.make_stream(
                b"BT 3 Tr (hidden first) Tj 0 Tr (visible last) Tj ET"
            )
            hidden_first[Name.Type] = Name.XObject
            hidden_first[Name.Subtype] = Name.Form
            hidden_first[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            user_form = pdf.make_stream(b"BT 3 Tr (user hidden text) Tj ET")
            user_form[Name.Type] = Name.XObject
            user_form[Name.Subtype] = Name.Form
            user_form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            page.obj[Name.Resources] = Dictionary(
                XObject=Dictionary(
                    {
                        "/OCR-hidden": hidden,
                        "/OCR-visible": visible,
                        "/OCR-separate": separate,
                        "/OCR-visible-first": visible_first,
                        "/OCR-hidden-first": hidden_first,
                        "/User-form": user_form,
                    }
                )
            )
            page.obj[Name.Contents] = pdf.make_stream(
                b"/OCR-hidden Do /OCR-visible Do /OCR-separate Do "
                b"/OCR-visible-first Do /OCR-hidden-first Do /User-form Do"
            )
            pdf.save(input_path)

        def add_new_ocr_form(
            source: Path,
            destination: Path,
            **_kwargs: object,
        ) -> None:
            shutil.copy2(source, destination)
            with Pdf.open(destination, allow_overwriting_input=True) as pdf:
                page = pdf.pages[0]
                new = pdf.make_stream(b"BT 3 Tr (new OCR) Tj ET")
                new[Name.Type] = Name.XObject
                new[Name.Subtype] = Name.Form
                new[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
                page.Resources.XObject["/OCR-new"] = new
                page.Contents = pikepdf.Array(
                    [page.Contents, pdf.make_stream(b"/OCR-new Do")]
                )
                pdf.save(destination)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=add_new_ocr_form,
        ):
            apply_ocr(
                input_path,
                output_path,
                ["en"],
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                force=True,
            )

        with Pdf.open(input_path) as pdf:
            assert (
                b"(old OCR)"
                in pdf.pages[0].Resources.XObject["/OCR-hidden"].read_bytes()
            )
        with Pdf.open(output_path) as pdf:
            xobjects = pdf.pages[0].Resources.XObject
            hidden_bytes = xobjects["/OCR-hidden"].read_bytes()
            assert b"(old OCR)" not in hidden_bytes
            assert b"BT" in hidden_bytes
            assert b"3 Tr" in hidden_bytes
            assert b"(visible text)" in xobjects["/OCR-visible"].read_bytes()
            assert b" re" in xobjects["/OCR-visible"].read_bytes()
            separate_bytes = xobjects["/OCR-separate"].read_bytes()
            assert b"(separate hidden)" not in separate_bytes
            assert b"(separate visible)" in separate_bytes
            visible_first_bytes = xobjects["/OCR-visible-first"].read_bytes()
            assert b"(visible first)" in visible_first_bytes
            assert b"(hidden last)" in visible_first_bytes
            hidden_first_bytes = xobjects["/OCR-hidden-first"].read_bytes()
            assert b"(hidden first)" in hidden_first_bytes
            assert b"(visible last)" in hidden_first_bytes
            assert b"(new OCR)" in xobjects["/OCR-new"].read_bytes()
            assert b"(user hidden text)" in xobjects["/User-form"].read_bytes()

    def test_force_and_deskew_are_incompatible(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
    ) -> None:
        with pytest.raises(OCRError, match="Deskew cannot"):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                force=True,
                deskew=True,
            )

    def test_annotations_disable_deskew(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        with (
            patch("pdftopdfa.ocr._pdf_has_annotations", return_value=True),
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ) as mock_ocr,
        ):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        assert mock_ocr.call_args.kwargs["deskew"] is False

    def test_rotation_preflight_supplies_temporary_pdf(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        _mock_paddle_orientation: MagicMock,
    ) -> None:
        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                rotate_pages=True,
            )

        _mock_paddle_orientation.assert_called_once()
        assert mock_ocr.call_args.args[0] != sample_pdf

    def test_invalid_language_is_ocr_error(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
    ) -> None:
        with pytest.raises(OCRError, match="Unsupported PaddleOCR"):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                ["eng"],
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

    def test_unavailable_dependency_is_fail_closed(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
    ) -> None:
        with (
            patch("pdftopdfa.ocr.HAS_OCR", False),
            pytest.raises(OCRError, match="OCR not available"),
        ):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

    def test_engine_error_is_wrapped(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=RuntimeError("inference failed"),
            ),
            pytest.raises(OCRError, match="inference failed"),
        ):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )


class TestVisiblePageRotationFix:
    """Tests for visible-page rotation normalization during OCR."""

    @staticmethod
    def _page_context(
        width_points: float,
        height_points: float,
        rotation: int = 0,
    ) -> SimpleNamespace:
        media_box = [0.0, 0.0, width_points, height_points]
        return SimpleNamespace(
            pageinfo=SimpleNamespace(
                width_inches=width_points / 72.0,
                height_inches=height_points / 72.0,
                rotation=rotation,
                mediabox=media_box,
                cropbox=media_box,
                trimbox=media_box,
                artbox=media_box,
                bleedbox=media_box,
            )
        )

    @pytest.mark.parametrize(
        ("page_size", "image_size", "expected"),
        [
            ((595.0, 842.0), (300, 200), True),
            ((842.0, 595.0), (200, 300), True),
            ((842.0, 595.0), (300, 200), False),
        ],
    )
    def test_axis_swap_detection(
        self,
        page_size: tuple[float, float],
        image_size: tuple[int, int],
        expected: bool,
    ) -> None:
        assert _should_swap_visible_page_axis(*page_size, *image_size) is expected

    @pytest.mark.parametrize(
        ("page_size", "image_size", "expected_mediabox"),
        [
            ((595.0, 842.0), (300, 200), [0.0, 0.0, 842.0, 595.0]),
            ((842.0, 595.0), (200, 300), [0.0, 0.0, 595.0, 842.0]),
        ],
    )
    def test_filter_preserves_visible_orientation(
        self,
        tmp_dir: Path,
        page_size: tuple[float, float],
        image_size: tuple[int, int],
        expected_mediabox: list[float],
    ) -> None:
        image_path = tmp_dir / "page.png"
        output_pdf = tmp_dir / "page.pdf"
        Image.new("RGB", image_size, color="white").save(image_path)
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=page_size)
            pdf.save(output_pdf)

        filter_pdf_page(
            self._page_context(*page_size),
            image_path,
            output_pdf,
        )

        with pikepdf.open(output_pdf) as pdf:
            assert [float(value) for value in pdf.pages[0].mediabox] == (
                expected_mediabox
            )
