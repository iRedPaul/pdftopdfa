# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the public OCR integration."""

import ctypes
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest
from conftest import new_pdf
from ocrmypdf.exceptions import PriorOcrFoundError
from ocrmypdf.pdfinfo import PdfInfo
from pikepdf import Dictionary, Name, Pdf
from PIL import Image

import pdftopdfa._ocr_runtime as ocr_runtime
from pdftopdfa import recognize_image
from pdftopdfa._ocr_runtime import (
    _DXGIAdapterDesc1,
    _is_directml_adapter,
    _parse_execution_provider,
    execution_provider_base,
    list_directml_devices,
    onnxruntime_engine_config,
    require_execution_provider,
    validate_ocr_execution_provider,
)
from pdftopdfa.exceptions import OCRError
from pdftopdfa.ocr import (
    _finalize_ocr_output,
    _find_deskew_pages,
    _ocr_form_names,
    _page_has_images,
    _page_has_text,
    _prepare_deskew_input,
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

        def copy_input(
            input_path: Path,
            output_path: Path,
            **_kwargs: object,
        ) -> list[object]:
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
    """Bypass model validation in tests that exercise only OCRmyPDF options."""
    with patch(
        "pdftopdfa.ocr_paddle.validate_model_directories",
        return_value=model_dirs,
    ) as mock_validate:
        yield mock_validate


def _copy_ocr_input(input_path: Path, output_path: Path, **_kwargs: object) -> None:
    """Model OCRmyPDF's output contract in option-boundary tests."""
    shutil.copy2(input_path, output_path)


def _enumerate_mock_directml_devices(
    descriptions: list[_DXGIAdapterDesc1],
) -> list[ocr_runtime.DirectMLDevice]:
    def create_factory(_iid: object, factory: object) -> int:
        factory._obj.value = 1
        return 0

    def enum_adapters(_factory: object, index: int, adapter: object) -> int:
        if index == len(descriptions):
            return ocr_runtime._DXGI_ERROR_NOT_FOUND
        adapter._obj.value = index + 2
        return 0

    def bind_com_method(
        pointer: object,
        method_index: int,
        *_signature: object,
    ) -> object:
        if method_index == ocr_runtime._IDXGI_FACTORY1_ENUM_ADAPTERS1:
            return enum_adapters
        if method_index == ocr_runtime._IDXGI_ADAPTER1_GET_DESC1:
            description = descriptions[pointer.value - 2]

            def get_description(_adapter: object, output: object) -> int:
                ctypes.memmove(
                    output,
                    ctypes.byref(description),
                    ctypes.sizeof(description),
                )
                return 0

            return get_description
        raise AssertionError(f"Unexpected COM method index {method_index}")

    create_factory_mock = MagicMock(side_effect=create_factory)
    dxgi = SimpleNamespace(CreateDXGIFactory1=create_factory_mock)
    with (
        patch.object(sys, "platform", "win32"),
        patch.object(ocr_runtime.ctypes, "WinDLL", return_value=dxgi, create=True),
        patch.object(ocr_runtime, "_com_method", side_effect=bind_com_method),
        patch.object(ocr_runtime, "_com_release"),
    ):
        return list_directml_devices()


def test_recognize_image_is_exposed_by_public_api(
    model_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "line.png"
    with (
        patch("pdftopdfa.ocr.onnxruntime_engine_config"),
        patch(
            "pdftopdfa.ocr_paddle.recognize_image",
            return_value=[("123", 0.9)],
        ) as recognize_with_paddle,
        patch("pdftopdfa.ocr._release_ocr_models") as release_models,
    ):
        result = recognize_image(
            image_path,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            layout="single_line",
            allowed_characters="0123456789",
        )

    assert result == [("123", 0.9)]
    recognize_with_paddle.assert_called_once_with(
        image_path,
        detection_model_dir=model_dirs[0],
        recognition_model_dir=model_dirs[1],
        ocr_execution_provider="cpu",
        layout="single_line",
        allowed_characters="0123456789",
    )
    release_models.assert_called_once_with()


class TestOcrExecutionProvider:
    """Tests for strict ONNX Runtime provider selection."""

    def test_cpu_is_the_default_configuration(self) -> None:
        assert onnxruntime_engine_config("cpu") == {
            "providers": ["CPUExecutionProvider"]
        }

    def test_directml_uses_required_session_options(self) -> None:
        fake_onnxruntime = SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
        )

        with patch.dict("sys.modules", {"onnxruntime": fake_onnxruntime}):
            config = onnxruntime_engine_config("directml")

        assert config == {
            "providers": ["DmlExecutionProvider"],
            "execution_mode": "sequential",
            "enable_mem_pattern": False,
        }

    def test_unavailable_directml_is_fail_closed(self) -> None:
        fake_onnxruntime = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        )

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_onnxruntime}),
            pytest.raises(
                OCRError,
                match=r"DmlExecutionProvider.*pdftopdfa\[directml\]",
            ),
        ):
            onnxruntime_engine_config("directml")

    def test_native_error_bytes_survive_invalid_utf8(self) -> None:
        message = b"ORT: Ger\xe4t nicht verf\xfcgbar"

        def get_available_providers() -> list[str]:
            raise UnicodeDecodeError(
                "utf-8",
                message,
                8,
                9,
                "invalid start byte",
            )

        fake_onnxruntime = SimpleNamespace(
            get_available_providers=get_available_providers
        )

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_onnxruntime}),
            pytest.raises(
                OCRError,
                match=r"could not be loaded: ORT: Ger.+nicht verf.+gbar",
            ),
        ):
            onnxruntime_engine_config("directml")

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported OCR execution provider"):
            onnxruntime_engine_config("cuda")

    def test_initialized_cpu_fallback_is_rejected(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]

        with pytest.raises(OCRError, match="refusing CPU fallback"):
            require_execution_provider(session, "directml")

        session.disable_fallback.assert_not_called()


class TestDirectMLDeviceIndex:
    """Tests for the optional ``directml:<index>`` device suffix."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("cpu", ("cpu", None)),
            ("directml", ("directml", None)),
            ("directml:0", ("directml", 0)),
            ("directml:1", ("directml", 1)),
            ("directml:63", ("directml", 63)),
        ],
    )
    def test_supported_providers_are_parsed(
        self,
        value: str,
        expected: tuple[str, int | None],
    ) -> None:
        assert _parse_execution_provider(value) == expected

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("cpu:0", "does not support a device index"),
            ("directml:", "Invalid DirectML device index"),
            ("directml:-1", "Invalid DirectML device index"),
            ("directml:+1", "Invalid DirectML device index"),
            ("directml:x", "Invalid DirectML device index"),
            ("directml:١", "Invalid DirectML device index"),
            ("directml:64", "out of range"),
            (None, "expected a string"),
        ],
    )
    def test_malformed_providers_are_rejected(
        self,
        value: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _parse_execution_provider(value)

    def test_device_index_is_canonicalized(self) -> None:
        """Padded indices collapse so one device maps to one cache key."""
        assert validate_ocr_execution_provider("directml:01") == "directml:1"
        assert validate_ocr_execution_provider("directml:0") == "directml:0"
        assert validate_ocr_execution_provider("directml") == "directml"

    def test_base_name_strips_the_device_index(self) -> None:
        assert execution_provider_base("directml:1") == "directml"
        assert execution_provider_base("directml") == "directml"
        assert execution_provider_base("cpu") == "cpu"

    def test_device_index_becomes_a_provider_option(self) -> None:
        fake_onnxruntime = SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
        )

        with patch.dict("sys.modules", {"onnxruntime": fake_onnxruntime}):
            config = onnxruntime_engine_config("directml:1")

        assert config == {
            "providers": ["DmlExecutionProvider"],
            "execution_mode": "sequential",
            "enable_mem_pattern": False,
            "provider_options": [{"device_id": 1}],
        }

    def test_indexed_directml_still_rejects_cpu_fallback(self) -> None:
        """The guard must not go silent because of the ``:1`` suffix."""
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]

        with pytest.raises(OCRError, match="DirectML.*refusing CPU fallback"):
            require_execution_provider(session, "directml:1")

        session.disable_fallback.assert_not_called()

    def test_indexed_directml_disables_fallback(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

        require_execution_provider(session, "directml:1")

        session.disable_fallback.assert_called_once_with()


class TestDirectMLDeviceEnumeration:
    """Tests for the DXGI adapter listing behind ``directml:<index>``."""

    def test_hardware_adapter_is_listed(self) -> None:
        desc = _DXGIAdapterDesc1(
            Description="Test GPU",
            VendorId=0x10DE,
            DedicatedVideoMemory=8 * 1024**3,
            Flags=0,
        )

        assert _is_directml_adapter(desc) is True

    def test_software_adapter_is_filtered(self) -> None:
        """WARP is flagged as software and never becomes a device index."""
        desc = _DXGIAdapterDesc1(
            Description="Microsoft Basic Render Driver",
            VendorId=0x1414,
            DedicatedVideoMemory=0,
            Flags=0x2,
        )

        assert _is_directml_adapter(desc) is False

    def test_basic_render_driver_is_filtered(self) -> None:
        """The Microsoft fallback adapter reports no dedicated video memory."""
        desc = _DXGIAdapterDesc1(
            Description="Microsoft Basic Render Driver",
            VendorId=0x1414,
            DedicatedVideoMemory=0,
            Flags=0,
        )

        assert _is_directml_adapter(desc) is False

    def test_device_ids_preserve_raw_dxgi_indices(self) -> None:
        descriptions = [
            _DXGIAdapterDesc1(
                Description="Software Adapter",
                VendorId=0x1414,
                Flags=0x2,
            ),
            _DXGIAdapterDesc1(
                Description="GPU A",
                VendorId=0x10DE,
                DeviceId=0x1F95,
                SubSysId=1,
                Revision=1,
                DedicatedVideoMemory=4 * 1024**3,
            ),
            _DXGIAdapterDesc1(
                Description="GPU B",
                VendorId=0x1002,
                DeviceId=0x73DF,
                SubSysId=2,
                Revision=2,
                DedicatedVideoMemory=8 * 1024**3,
            ),
        ]

        devices = _enumerate_mock_directml_devices(descriptions)

        assert [device.description for device in devices] == ["GPU A", "GPU B"]
        assert [device.device_id for device in devices] == [1, 2]
        assert [device.execution_provider for device in devices] == [
            "directml:1",
            "directml:2",
        ]

    def test_duplicate_pci_adapters_keep_lowest_dxgi_index(self) -> None:
        descriptions = [
            _DXGIAdapterDesc1(
                Description="GPU",
                VendorId=0x10DE,
                DeviceId=0x1F95,
                SubSysId=0x3A3E17AA,
                Revision=161,
                DedicatedVideoMemory=4 * 1024**3,
            )
            for _ in range(3)
        ]

        devices = _enumerate_mock_directml_devices(descriptions)

        assert [device.device_id for device in devices] == [0]

    def test_distinct_device_ids_are_not_deduplicated(self) -> None:
        descriptions = [
            _DXGIAdapterDesc1(
                Description=f"GPU {device_id}",
                VendorId=0x10DE,
                DeviceId=device_id,
                SubSysId=1,
                Revision=1,
                DedicatedVideoMemory=4 * 1024**3,
            )
            for device_id in (1, 2)
        ]

        devices = _enumerate_mock_directml_devices(descriptions)

        assert [device.device_id for device in devices] == [0, 1]

    @pytest.mark.skipif(sys.platform == "win32", reason="DXGI is available")
    def test_enumeration_is_windows_only(self) -> None:
        with pytest.raises(OCRError, match="only be enumerated on Windows"):
            list_directml_devices()

    @pytest.mark.skipif(sys.platform != "win32", reason="requires DXGI")
    def test_device_ids_are_ordered_and_usable(self) -> None:
        """Every listed adapter yields a provider string the API accepts."""
        devices = list_directml_devices()

        device_ids = [device.device_id for device in devices]
        assert device_ids == sorted(set(device_ids))
        for device in devices:
            provider = device.execution_provider
            assert provider == f"directml:{device.device_id}"
            assert validate_ocr_execution_provider(provider) == provider


def _add_content_page(
    pdf: Pdf,
    *,
    image_scale: int = 100,
    visible_text: bool = False,
    hidden_form_text: bool = False,
    visible_form_text: bool = False,
    vector: bool = False,
) -> None:
    """Add a page with controlled raster, text, and vector content."""
    page = pdf.add_blank_page(page_size=(100, 100))
    image = pdf.make_stream(bytes([128]) * 100)
    image[Name.Type] = Name.XObject
    image[Name.Subtype] = Name.Image
    image[Name.Width] = 10
    image[Name.Height] = 10
    image[Name.ColorSpace] = Name.DeviceGray
    image[Name.BitsPerComponent] = 8

    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type1,
        BaseFont=Name.Helvetica,
    )
    xobjects = Dictionary(Im0=image)
    page.obj[Name.Resources] = Dictionary(
        XObject=xobjects,
        Font=Dictionary(F1=font),
    )
    content = [f"q {image_scale} 0 0 {image_scale} 0 0 cm /Im0 Do Q".encode("ascii")]
    if visible_text:
        content.append(b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET")
    if hidden_form_text or visible_form_text:
        render_mode = 0 if visible_form_text else 3
        form = pdf.make_stream(
            f"BT /F1 12 Tf {render_mode} Tr 10 50 Td (Form text) Tj ET".encode()
        )
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
        form[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
        xobjects["/HiddenText"] = form
        content.append(b"/HiddenText Do")
    if vector:
        content.append(b"0 0 10 10 re f")
    page.obj[Name.Contents] = pdf.make_stream(b"\n".join(content))


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

    def test_whitespace_only_text_does_not_suppress_ocr_detection(self) -> None:
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            page.Contents.write(
                page.Contents.read_bytes() + b"\nBT /F1 11 Tf [( ) 10.5 ( )] TJ ET"
            )

            assert _page_has_text(page) is False
            assert needs_ocr(pdf) is True


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

    def test_deskew_selects_full_page_scan_for_one_targeted_call(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.args == (input_path, output_path)
        kwargs = mock_ocr.call_args.kwargs
        assert kwargs["deskew"] is True
        assert kwargs["pages"] == "1"
        assert kwargs["jobs"] == 1
        assert kwargs["skip_text"] is True
        assert "force_ocr" not in kwargs
        assert "redo_ocr" not in kwargs

    def test_deskew_copies_digital_page_without_ocrmypdf(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "digital.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_text=True)
            pdf.save(input_path)

        with patch("pdftopdfa.ocr.ocrmypdf.ocr") as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        mock_ocr.assert_not_called()
        with Pdf.open(input_path) as source, Pdf.open(output_path) as output:
            assert source.pages[0].Contents.read_bytes() == (
                output.pages[0].Contents.read_bytes()
            )

    def test_deskew_accepts_scan_with_invisible_form_text(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "ocr-scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, hidden_form_text=True)
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]
        prepared_page_has_text = []

        def inspect_and_copy(
            source: Path,
            destination: Path,
            **_kwargs: object,
        ) -> None:
            with Pdf.open(source) as pdf:
                prepared_page_has_text.append(_page_has_text(pdf.pages[0]))
            shutil.copy2(source, destination)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=inspect_and_copy,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.kwargs["pages"] == "1"
        assert mock_ocr.call_args.kwargs["skip_text"] is True
        assert "force_ocr" not in mock_ocr.call_args.kwargs
        assert prepared_page_has_text == [False]

    def test_prior_ocr_fallback_preserves_deskew_text_layer(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "ocr-scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, hidden_form_text=True)
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=PriorOcrFoundError(),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        with Pdf.open(output_path) as pdf:
            assert b"(Form text)" in (
                pdf.pages[0].Resources.XObject.HiddenText.read_bytes()
            )

    def test_deskew_handles_mixed_pdf_page_by_page_in_disjoint_calls(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "mixed.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            _add_content_page(pdf, hidden_form_text=True)
            _add_content_page(pdf, vector=True)
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        assert mock_ocr.call_count == 2
        regular_call, deskew_call = mock_ocr.call_args_list
        assert regular_call.kwargs["pages"] == "4"
        assert regular_call.kwargs["deskew"] is False
        assert deskew_call.kwargs["pages"] == "1,3"
        assert deskew_call.kwargs["deskew"] is True
        assert all(
            call.kwargs["skip_text"] is True and "force_ocr" not in call.kwargs
            for call in mock_ocr.call_args_list
        )

    def test_deskew_rejects_vector_and_small_image_pages(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "not-scans.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, vector=True)
            _add_content_page(pdf, image_scale=50)
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    @pytest.mark.parametrize(
        "unsafe_content",
        [
            "shading",
            "clip",
            "off-page",
            "optional-content",
            "embedded-alpha",
            "invalid-blend",
        ],
    )
    def test_deskew_rejects_unsafe_image_coverage(
        self,
        tmp_dir: Path,
        unsafe_content: str,
    ) -> None:
        input_path = tmp_dir / f"{unsafe_content}.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            if unsafe_content == "shading":
                page.Contents.write(page.Contents.read_bytes() + b"\n/Sh0 sh")
            elif unsafe_content == "clip":
                page.Contents.write(b"0 0 10 100 re W n q 100 0 0 100 0 0 cm /Im0 Do Q")
            elif unsafe_content == "off-page":
                page.Contents.write(b"q 100 0 0 100 50 0 cm /Im0 Do Q")
            elif unsafe_content == "optional-content":
                ocg = Dictionary(Type=Name.OCG, Name="Hidden scan")
                pdf.Root[Name.OCProperties] = Dictionary(
                    OCGs=pikepdf.Array([ocg]),
                    D=Dictionary(OFF=pikepdf.Array([ocg])),
                )
                page.Resources[Name.Properties] = Dictionary(MC0=ocg)
                page.Contents.write(
                    b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET\n"
                    b"/OC /MC0 BDC q 100 0 0 100 0 0 cm /Im0 Do Q EMC"
                )
            elif unsafe_content == "embedded-alpha":
                page.Resources.XObject.Im0[Name.SMaskInData] = 1
                page.Contents.write(
                    b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET\n"
                    b"q 100 0 0 100 0 0 cm /Im0 Do Q"
                )
            else:
                page.Resources[Name.ExtGState] = Dictionary(
                    GS0=Dictionary(BM=pikepdf.Array([]))
                )
                page.Contents.write(
                    b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET\n"
                    b"/GS0 gs q 100 0 0 100 0 0 cm /Im0 Do Q"
                )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_accepts_full_page_inline_scan(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "inline-scan.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].Contents.write(
                b"q 100 0 0 100 0 0 cm "
                b"BI /W 10 /H 10 /CS /G /BPC 8 ID\n" + bytes([128]) * 100 + b"\nEI Q"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    @pytest.mark.parametrize("image_kind", ["xobject", "inline"])
    def test_deskew_rejects_pattern_painted_image_masks(
        self,
        tmp_dir: Path,
        image_kind: str,
    ) -> None:
        input_path = tmp_dir / f"{image_kind}-image-mask.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            pattern = pdf.make_stream(b"1 0 0 rg 0 0 5 10 re f 0 0 1 rg 5 0 5 10 re f")
            pattern[Name.Type] = Name.Pattern
            pattern[Name.PatternType] = 1
            pattern[Name.PaintType] = 1
            pattern[Name.TilingType] = 1
            pattern[Name.BBox] = pikepdf.Array([0, 0, 10, 10])
            pattern[Name.XStep] = 10
            pattern[Name.YStep] = 10
            pattern[Name.Resources] = Dictionary()
            page.Resources[Name.Pattern] = Dictionary(P1=pattern)

            if image_kind == "xobject":
                image_mask = pdf.make_stream(b"\x00")
                image_mask[Name.Type] = Name.XObject
                image_mask[Name.Subtype] = Name.Image
                image_mask[Name.Width] = 1
                image_mask[Name.Height] = 1
                image_mask[Name.ImageMask] = True
                image_mask[Name.BitsPerComponent] = 1
                page.Resources.XObject[Name.Mask] = image_mask
                mask_paint = b"q 50 0 0 50 25 25 cm /Mask Do Q"
            else:
                mask_paint = (
                    b"q 50 0 0 50 25 25 cm BI /W 1 /H 1 /IM true /BPC 1 ID \x00 EI Q"
                )
            page.Contents.write(
                page.Contents.read_bytes() + b"\n/Pattern cs /P1 scn " + mask_paint
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    @pytest.mark.parametrize(
        "image_kind",
        [
            "xobject-separation",
            "xobject-separation-alias",
            "xobject-devicen-alias",
            "xobject-short-alias",
            "xobject-defaultgray",
            "xobject-indexed",
            "xobject-missing-color-space",
            "inline-separation-alias",
        ],
    )
    def test_deskew_rejects_nonmarking_full_page_images(
        self,
        tmp_dir: Path,
        image_kind: str,
    ) -> None:
        input_path = tmp_dir / f"{image_kind}.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            tint = Dictionary(
                FunctionType=2,
                Domain=pikepdf.Array([0, 1]),
                C0=pikepdf.Array([0]),
                C1=pikepdf.Array([1]),
                N=1,
            )
            separation = pikepdf.Array(
                [Name.Separation, Name("/None"), Name.DeviceGray, tint]
            )
            if image_kind == "xobject-devicen-alias":
                color_space = pikepdf.Array(
                    [
                        Name.DeviceN,
                        pikepdf.Array([Name("/None")]),
                        Name.DeviceGray,
                        tint,
                    ]
                )
            elif image_kind == "xobject-indexed":
                color_space = pikepdf.Array(
                    [Name.Indexed, separation, 0, pikepdf.String(b"\x00")]
                )
            else:
                color_space = separation
            native_text = b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET\n"

            if image_kind == "inline-separation-alias":
                page.Resources[Name.ColorSpace] = Dictionary(CS0=color_space)
                page.Contents.write(
                    native_text + b"q 100 0 0 100 0 0 cm "
                    b"BI /W 1 /H 1 /CS /CS0 /BPC 8 ID \x80 EI Q"
                )
            else:
                image = page.Resources.XObject.Im0
                if image_kind == "xobject-separation":
                    image[Name.ColorSpace] = color_space
                elif image_kind == "xobject-short-alias":
                    page.Resources[Name.ColorSpace] = Dictionary()
                    page.Resources.ColorSpace[Name("/G")] = color_space
                    image[Name.ColorSpace] = Name("/G")
                elif image_kind == "xobject-defaultgray":
                    page.Resources[Name.ColorSpace] = Dictionary(
                        DefaultGray=color_space
                    )
                    image[Name.ColorSpace] = Name.DeviceGray
                elif image_kind == "xobject-indexed":
                    image[Name.ColorSpace] = color_space
                elif image_kind == "xobject-missing-color-space":
                    del image[Name.ColorSpace]
                else:
                    page.Resources[Name.ColorSpace] = Dictionary(CS0=color_space)
                    image[Name.ColorSpace] = Name.CS0
                page.Contents.write(native_text + b"q 100 0 0 100 0 0 cm /Im0 Do Q")
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_resolves_marking_color_space_alias(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "marking-color-space-alias.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            page.Resources[Name.ColorSpace] = Dictionary(CS0=Name.DeviceGray)
            page.Resources.XObject.Im0[Name.ColorSpace] = Name.CS0
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    @pytest.mark.parametrize("mask_kind", ["nonempty", "missing-group"])
    def test_deskew_rejects_unsafe_soft_mask_group(
        self,
        tmp_dir: Path,
        mask_kind: str,
    ) -> None:
        input_path = tmp_dir / f"{mask_kind}-soft-mask.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            mask_group = pdf.make_stream(
                b"0 0 100 100 re f" if mask_kind == "nonempty" else b""
            )
            mask_group[Name.Type] = Name.XObject
            mask_group[Name.Subtype] = Name.Form
            mask_group[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            if mask_kind == "nonempty":
                mask_group[Name.Group] = Dictionary(
                    S=Name.Transparency,
                    CS=Name.DeviceGray,
                )
            page.Resources[Name.ExtGState] = Dictionary(
                GS0=Dictionary(SMask=Dictionary(S=Name.Alpha, G=mask_group))
            )
            page.Contents.write(
                page.Contents.read_bytes() + b"\n/GS0 gs q 10 0 0 10 0 0 cm /Im0 Do Q"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_allows_unsupported_soft_mask_reset_before_image(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "reset-soft-mask.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            mask_group = pdf.make_stream(b"0 0 100 100 re f")
            mask_group[Name.Type] = Name.XObject
            mask_group[Name.Subtype] = Name.Form
            mask_group[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            mask_group[Name.Group] = Dictionary(
                S=Name.Transparency,
                CS=Name.DeviceGray,
            )
            page.Resources[Name.ExtGState] = Dictionary(
                GS0=Dictionary(SMask=Dictionary(S=Name.Alpha, G=mask_group)),
                GS1=Dictionary(SMask=Name("/None")),
            )
            page.Contents.write(b"/GS0 gs /GS1 gs q 100 0 0 100 0 0 cm /Im0 Do Q")
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    @pytest.mark.parametrize("hidden_kind", ["render-mode-3", "alpha-zero"])
    def test_deskew_accepts_invisible_text_with_unsupported_soft_mask(
        self,
        tmp_dir: Path,
        hidden_kind: str,
    ) -> None:
        input_path = tmp_dir / f"{hidden_kind}-soft-mask.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            mask_group = pdf.make_stream(b"0 0 100 100 re f")
            mask_group[Name.Type] = Name.XObject
            mask_group[Name.Subtype] = Name.Form
            mask_group[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            mask_group[Name.Group] = Dictionary(
                S=Name.Transparency,
                CS=Name.DeviceGray,
            )
            graphics_state = Dictionary(SMask=Dictionary(S=Name.Alpha, G=mask_group))
            render_mode = 3
            if hidden_kind == "alpha-zero":
                graphics_state[Name("/ca")] = 0
                render_mode = 0
            page.Resources[Name.ExtGState] = Dictionary(GS0=graphics_state)
            page.Contents.write(
                page.Contents.read_bytes()
                + f"\n/GS0 gs BT /F1 12 Tf {render_mode} Tr "
                "10 50 Td (OCR) Tj ET".encode()
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    def test_deskew_accepts_empty_alpha_soft_mask_ocr_text(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "empty-soft-mask.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            mask_group = pdf.make_stream(b"")
            mask_group[Name.Type] = Name.XObject
            mask_group[Name.Subtype] = Name.Form
            mask_group[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            mask_group[Name.Group] = Dictionary(
                S=Name.Transparency,
                CS=Name.DeviceGray,
            )
            page.Resources[Name.ExtGState] = Dictionary(
                GS0=Dictionary(SMask=Dictionary(S=Name.Alpha, G=mask_group))
            )
            page.Contents.write(
                page.Contents.read_bytes()
                + b"\n/GS0 gs BT /F1 12 Tf 0 Tr 10 50 Td (OCR) Tj ET"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    @pytest.mark.parametrize("hidden_kind", ["render-mode-7", "alpha-zero"])
    def test_deskew_accepts_other_invisible_ocr_text(
        self,
        tmp_dir: Path,
        hidden_kind: str,
    ) -> None:
        input_path = tmp_dir / f"{hidden_kind}.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            if hidden_kind == "render-mode-7":
                hidden_text = b"BT /F1 12 Tf 7 Tr 10 50 Td (OCR) Tj ET"
            else:
                page.Resources[Name.ExtGState] = Dictionary(GS0=Dictionary(ca=0))
                hidden_text = b"/GS0 gs BT /F1 12 Tf 0 Tr 10 50 Td (OCR) Tj ET"
            page.Contents.write(page.Contents.read_bytes() + b"\n" + hidden_text)
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    @pytest.mark.parametrize(
        "hidden_kind",
        [
            "render-mode-3",
            "alpha-zero",
            "optional-form",
            "extgstate-font",
            "malformed-extgstate-font",
        ],
    )
    def test_deskew_rejects_type3_text_even_with_hidden_outer_state(
        self,
        tmp_dir: Path,
        hidden_kind: str,
    ) -> None:
        input_path = tmp_dir / f"type3-{hidden_kind}.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            glyph = pdf.make_stream(b"500 0 d0 0 0 500 500 re f")
            type3_font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                FontBBox=pikepdf.Array([0, 0, 500, 500]),
                FontMatrix=pikepdf.Array([0.001, 0, 0, 0.001, 0, 0]),
                CharProcs=Dictionary(A=glyph),
                Encoding=Dictionary(
                    Type=Name.Encoding,
                    Differences=pikepdf.Array([65, Name.A]),
                ),
                FirstChar=65,
                LastChar=65,
                Widths=pikepdf.Array([500]),
                Resources=Dictionary(),
            )
            page.Resources.Font[Name.F3] = type3_font

            if hidden_kind == "render-mode-3":
                type3_text = b"BT /F3 12 Tf 3 Tr 10 50 Td (A) Tj ET"
            elif hidden_kind == "alpha-zero":
                page.Resources[Name.ExtGState] = Dictionary(GS0=Dictionary(ca=0, CA=0))
                type3_text = b"/GS0 gs BT /F3 12 Tf 0 Tr 10 50 Td (A) Tj ET"
            elif hidden_kind == "optional-form":
                ocg = Dictionary(Type=Name.OCG, Name="Hidden Type3 text")
                pdf.Root[Name.OCProperties] = Dictionary(
                    OCGs=pikepdf.Array([ocg]),
                    D=Dictionary(OFF=pikepdf.Array([ocg])),
                )
                form = pdf.make_stream(b"BT /F3 12 Tf 3 Tr 10 50 Td (A) Tj ET")
                form[Name.Type] = Name.XObject
                form[Name.Subtype] = Name.Form
                form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
                form[Name.OC] = ocg
                form[Name.Resources] = Dictionary(Font=Dictionary(F3=type3_font))
                page.Resources.XObject[Name.Type3Form] = form
                type3_text = b"/Type3Form Do"
            else:
                font_setting = (
                    pikepdf.Array([type3_font])
                    if hidden_kind == "malformed-extgstate-font"
                    else pikepdf.Array([type3_font, 12])
                )
                page.Resources[Name.ExtGState] = Dictionary(
                    GSFont=Dictionary(Font=font_setting)
                )
                type3_text = b"BT /F1 12 Tf ET /GSFont gs BT 3 Tr 10 50 Td (A) Tj ET"

            page.Contents.write(page.Contents.read_bytes() + b"\n" + type3_text)
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_accepts_optional_content_hidden_ocr_text(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "optional-hidden-ocr.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            ocg = Dictionary(Type=Name.OCG, Name="Hidden OCR")
            pdf.Root[Name.OCProperties] = Dictionary(
                OCGs=pikepdf.Array([ocg]),
                D=Dictionary(OFF=pikepdf.Array([ocg])),
            )
            page.Resources[Name.Properties] = Dictionary(OCRLayer=ocg)
            page.Contents.write(
                page.Contents.read_bytes() + b"\n/OC /OCRLayer BDC "
                b"BT /F1 12 Tf 3 Tr 10 50 Td (OCR) Tj ET EMC"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    def test_deskew_accepts_optional_form_with_alpha_zero_ocr_text(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "optional-form-hidden-ocr.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            ocg = Dictionary(Type=Name.OCG, Name="Hidden OCR")
            pdf.Root[Name.OCProperties] = Dictionary(
                OCGs=pikepdf.Array([ocg]),
                D=Dictionary(OFF=pikepdf.Array([ocg])),
            )
            form = pdf.make_stream(b"/GS0 gs BT /F1 12 Tf 0 Tr 10 50 Td (OCR) Tj ET")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.OC] = ocg
            form[Name.Resources] = Dictionary(
                Font=page.Resources.Font,
                ExtGState=Dictionary(GS0=Dictionary(ca=0)),
            )
            page.Resources.XObject[Name.HiddenOCR] = form
            page.Contents.write(page.Contents.read_bytes() + b"\n/HiddenOCR Do")
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    def test_deskew_rejects_optional_form_with_visible_text(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "optional-form-visible-text.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            ocg = Dictionary(Type=Name.OCG, Name="Optional text")
            pdf.Root[Name.OCProperties] = Dictionary(
                OCGs=pikepdf.Array([ocg]),
                D=Dictionary(OFF=pikepdf.Array([ocg])),
            )
            form = pdf.make_stream(b"BT /F1 12 Tf 0 Tr 10 50 Td (Text) Tj ET")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.OC] = ocg
            form[Name.Resources] = Dictionary(Font=page.Resources.Font)
            page.Resources.XObject[Name.OptionalText] = form
            page.Contents.write(page.Contents.read_bytes() + b"\n/OptionalText Do")
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_accepts_text_occluded_by_full_page_image(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "occluded-text.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].Contents.write(
                b"BT /F1 12 Tf 0 Tr 10 50 Td (OCR) Tj ET\n"
                b"q 100 0 0 100 0 0 cm /Im0 Do Q"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == [1]

    def test_deskew_does_not_hide_text_behind_partial_page_image(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "partial-occlusion.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].Contents.write(
                b"BT /F1 12 Tf 0 Tr 10 50 Td (Native text) Tj ET\n"
                b"q 99.9 0 0 100 0 0 cm /Im0 Do Q"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_rejects_text_clip_that_affects_later_image(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "text-clip.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].Contents.write(
                b"q 100 0 0 100 0 0 cm /Im0 Do Q\n"
                b"BT /F1 12 Tf 7 Tr 10 50 Td (clip) Tj ET\n"
                b"q 10 0 0 10 0 0 cm /Im0 Do Q"
            )
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_regular_and_deskew_pages_are_ocrd_once_in_disjoint_runs(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "two-raster-pages.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, image_scale=50)
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        assert mock_ocr.call_count == 2
        regular_call, deskew_call = mock_ocr.call_args_list
        assert regular_call.kwargs["pages"] == "2"
        assert regular_call.kwargs["deskew"] is False
        assert deskew_call.kwargs["pages"] == "1"
        assert deskew_call.kwargs["deskew"] is True
        assert all(
            call.kwargs["skip_text"] is True
            and call.kwargs["jobs"] == 1
            and "force_ocr" not in call.kwargs
            for call in mock_ocr.call_args_list
        )

    def test_deskew_preserves_tagging_on_unselected_digital_page(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "tagged-mixed.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            pdf.Root[Name.StructTreeRoot] = Dictionary(
                Type=Name.StructTreeRoot,
                K=pikepdf.Array([]),
            )
            pdf.Root[Name.MarkInfo] = Dictionary(Marked=True)
            pdf.pages[1].obj[Name.StructParents] = 0
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        with Pdf.open(output_path) as pdf:
            assert Name.StructTreeRoot in pdf.Root
            assert bool(pdf.Root.MarkInfo.Marked)
            assert int(pdf.pages[1].obj.StructParents) == 0

    def test_deskew_text_preparation_clones_shared_form(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "shared-form.pdf"
        output_path = tmp_dir / "prepared.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            shared = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 10 50 Td (shared OCR) Tj ET")
            shared[Name.Type] = Name.XObject
            shared[Name.Subtype] = Name.Form
            shared[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            for page in pdf.pages:
                page.Resources.XObject[Name.Shared] = shared
                page.Contents.write(page.Contents.read_bytes() + b"\n/Shared Do")
            pdf.save(input_path)

        _prepare_deskew_input(input_path, output_path, (1,))

        with Pdf.open(output_path) as pdf:
            selected = pdf.pages[0].Resources.XObject.Shared
            digital = pdf.pages[1].Resources.XObject.Shared
            assert b"Tj" not in selected.read_bytes()
            assert b"Tj" in digital.read_bytes()

    def test_deskew_rejects_visible_form_text_with_inherited_resources(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "inherited-resources.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_form_text=True)
            page = pdf.pages[0]
            page.obj.Parent[Name.Resources] = page.obj[Name.Resources]
            del page.obj[Name.Resources]
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

    def test_deskew_rejects_form_without_bbox(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "form-without-bbox.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, hidden_form_text=True)
            del pdf.pages[0].Resources.XObject.HiddenText[Name.BBox]
            pdf.save(input_path)

        assert _find_deskew_pages(input_path) == []

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
            "paddle_execution_provider": "cpu",
        }

    def test_prepares_whitespace_only_image_page_for_regular_ocr(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "whitespace-text.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            page = pdf.pages[0]
            page.Contents.write(
                page.Contents.read_bytes() + b"\nBT /F1 11 Tf [( )] TJ ET"
            )
            pdf.save(input_path)

        assert [page.has_text for page in PdfInfo(input_path, max_workers=1).pages] == [
            True,
            True,
        ]
        prepared_text = []

        def inspect_and_copy(
            source: Path,
            destination: Path,
            **_kwargs: object,
        ) -> None:
            assert source != input_path
            prepared_text.append(
                [page.has_text for page in PdfInfo(source, max_workers=1).pages]
            )
            with Pdf.open(source) as pdf:
                assert b"Native text" in pdf.pages[1].Contents.read_bytes()
            shutil.copy2(source, destination)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=inspect_and_copy,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

        assert prepared_text == [[False, True]]
        assert mock_ocr.call_args.kwargs["skip_text"] is True

    def test_passes_directml_to_ocrmypdf_plugin(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output = tmp_dir / "output.pdf"
        with (
            patch("pdftopdfa.ocr.onnxruntime_engine_config"),
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ) as mock_ocr,
        ):
            apply_ocr(
                sample_pdf,
                output,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                ocr_execution_provider="directml",
            )

        assert mock_ocr.call_args.kwargs["paddle_execution_provider"] == "directml"

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

    def test_prior_ocr_fallback_preserves_existing_ocr_form(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page()
            existing_ocr = pdf.make_stream(b"BT 3 Tr (old OCR) Tj ET")
            existing_ocr[Name.Type] = Name.XObject
            existing_ocr[Name.Subtype] = Name.Form
            existing_ocr[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            page.obj[Name.Resources] = Dictionary(
                XObject=Dictionary({"/OCR-existing": existing_ocr})
            )
            page.obj[Name.Contents] = pdf.make_stream(b"/OCR-existing Do")
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=PriorOcrFoundError(),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                force=True,
            )

        with Pdf.open(output_path) as pdf:
            assert b"(old OCR)" in (
                pdf.pages[0].Resources.XObject["/OCR-existing"].read_bytes()
            )

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

    def test_annotations_disable_deskew_only_for_annotated_scan(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "annotated-scan.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].obj[Name.Annots] = pikepdf.Array(
                [
                    Dictionary(
                        Type=Name.Annot,
                        Subtype=Name.Link,
                        Rect=pikepdf.Array([0, 0, 10, 10]),
                    )
                ]
            )
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        assert mock_ocr.call_args.kwargs["deskew"] is False
        assert mock_ocr.call_args.kwargs["pages"] == "1"

    def test_annotation_on_digital_page_does_not_disable_scan_deskew(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "mixed-annotation.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            pdf.pages[1].obj[Name.Annots] = pikepdf.Array(
                [
                    Dictionary(
                        Type=Name.Annot,
                        Subtype=Name.Link,
                        Rect=pikepdf.Array([0, 0, 10, 10]),
                    )
                ]
            )
            pdf.save(input_path)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=_copy_ocr_input,
        ) as mock_ocr:
            apply_ocr(
                input_path,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.kwargs["deskew"] is True
        assert mock_ocr.call_args.kwargs["pages"] == "1"

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
        assert _mock_paddle_orientation.call_args.kwargs["execution_provider"] == "cpu"
        assert mock_ocr.call_args.args[0] != sample_pdf

    def test_rotation_preflight_uses_directml(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        _mock_paddle_orientation: MagicMock,
    ) -> None:
        with (
            patch("pdftopdfa.ocr.onnxruntime_engine_config"),
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
        ):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                rotate_pages=True,
                ocr_execution_provider="directml",
            )

        assert (
            _mock_paddle_orientation.call_args.kwargs["execution_provider"]
            == "directml"
        )

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
            patch("pdftopdfa.ocr._release_ocr_models") as release_models,
            pytest.raises(OCRError, match="inference failed"),
        ):
            apply_ocr(
                sample_pdf,
                tmp_dir / "output.pdf",
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

        release_models.assert_called_once_with()


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
