# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the public OCR integration."""

import ctypes
import json
import os
import shutil
import sys
import zlib
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
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
import pdftopdfa.digital_layout as digital_layout
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
    _OCR_MAX_PAGE_RASTER_PIXELS,
    _cleanup_ocr_resources,
    _finalize_ocr_output,
    _find_deskew_pages,
    _image_color_space_marks,
    _ocr_form_names,
    _page_has_images,
    _page_has_text,
    _page_paint_analysis,
    _preflight_ocr_input,
    _prepare_deskew_input,
    _read_ocr_run_sidecars,
    _strip_invisible_text_from_form,
    _validate_ocr_document_manifest,
    _validate_ocr_page_manifest,
    _write_ocr_document_manifest,
    apply_ocr,
    is_ocr_available,
    needs_ocr,
    validate_ocr_languages,
)
from pdftopdfa.ocr_rotation_fix import (
    _should_swap_visible_page_axis,
    filter_pdf_page,
)
from pdftopdfa.staging import StagedFileSnapshot
from pdftopdfa.staging import publish_staged_file as publish_staged_file_impl
from pdftopdfa.staging import (
    rollback_staged_publication as rollback_staged_publication_impl,
)
from pdftopdfa.staging import verify_staged_file as verify_staged_file_snapshot


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


def _add_ocr_form(
    pdf: Pdf,
    page_index: int,
    name: str,
    mcids: Sequence[int],
) -> None:
    """Attach an OCR Form whose content marks exactly *mcids*."""
    content = b" ".join(
        f"/Span <</MCID {mcid}>> BDC BT ET EMC".encode() for mcid in mcids
    )
    form = pdf.make_stream(content)
    form[Name.Type] = Name.XObject
    form[Name.Subtype] = Name.Form
    form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
    form[Name.Resources] = Dictionary()
    resources = pdf.pages[page_index].Resources
    if Name.XObject not in resources:
        resources[Name.XObject] = Dictionary()
    resources.XObject[Name(name)] = form


def _ocr_page_manifest(page_index: int, text: str) -> dict[str, object]:
    bbox = {"left": 10.0, "top": 10.0, "right": 90.0, "bottom": 30.0}
    polygon = [[10.0, 10.0], [90.0, 10.0], [90.0, 30.0], [10.0, 30.0]]
    return {
        "schema_version": 1,
        "type": "pdftopdfa-ocr-page",
        "page_index": page_index,
        "raster": {"width": 100, "height": 100, "dpi": 300.0},
        "coordinates": {
            "width": 100.0,
            "height": 100.0,
            "dpi": 300.0,
            "scale_from_raster": 1.0,
        },
        "languages": ["de", "en"],
        "layout": {
            "reading_order_applied": True,
            "sections": [],
            "selected_columns": [
                {"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 100.0}
            ],
        },
        "lines": [
            {
                "mcid": 0,
                "ocr_class": "ocr_line",
                "text": text,
                "confidence": 0.95,
                "bbox": bbox,
                "polygon": polygon,
                "language": "de+en",
                "direction": "ltr",
                "baseline": {"slope": 0.0, "intercept": 0.0},
                "text_angle": 0.0,
                "words": [
                    {
                        "index": 0,
                        "ocr_class": "ocrx_word",
                        "text": text,
                        "confidence": 0.95,
                        "bbox": bbox,
                        "polygon": polygon,
                        "language": "de+en",
                        "direction": "ltr",
                        "baseline": None,
                        "text_angle": None,
                    }
                ],
            }
        ],
    }


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
    backend = MagicMock()
    backend.recognize_image.return_value = [("123", 0.9)]
    with (
        patch("pdftopdfa.ocr.onnxruntime_engine_config"),
        patch(
            "pdftopdfa.ocr_paddle._ImageOCRSession",
            return_value=backend,
        ) as backend_class,
        patch("pdftopdfa.ocr.gc.collect") as collect,
    ):
        result = recognize_image(
            image_path,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            layout="single_line",
            allowed_characters="0123456789",
        )

    assert result == [("123", 0.9)]
    backend_class.assert_called_once_with(
        detection_model_dir=model_dirs[0],
        recognition_model_dir=model_dirs[1],
        ocr_execution_provider="cpu",
    )
    backend.recognize_image.assert_called_once_with(
        image_path,
        layout="single_line",
        allowed_characters="0123456789",
    )
    backend.close.assert_called_once_with()
    collect.assert_called_once_with()


def test_recognize_image_preserves_error_and_release_behavior(
    model_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    backend = MagicMock()
    backend.recognize_image.side_effect = OCRError("inference failed")
    with (
        patch("pdftopdfa.ocr.onnxruntime_engine_config"),
        patch(
            "pdftopdfa.ocr_paddle._ImageOCRSession",
            return_value=backend,
        ),
        patch("pdftopdfa.ocr.gc.collect") as collect,
        pytest.raises(OCRError, match="inference failed"),
    ):
        recognize_image(
            tmp_path / "page.png",
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        )

    backend.close.assert_called_once_with()
    collect.assert_called_once_with()


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


def _add_scan_page(
    pdf: Pdf,
    *,
    page_size: tuple[int, int] = (1700, 2338),
    image_size: tuple[int, int] = (1700, 2338),
) -> None:
    """Add a full-page RGB JPEG scan with controlled geometry and native DPI."""
    page = pdf.add_blank_page(page_size=page_size)
    encoded = BytesIO()
    with Image.new("RGB", image_size, color="white") as image:
        image.save(encoded, format="JPEG")
    image = pdf.make_stream(encoded.getvalue())
    image[Name.Type] = Name.XObject
    image[Name.Subtype] = Name.Image
    image[Name.Width] = image_size[0]
    image[Name.Height] = image_size[1]
    image[Name.ColorSpace] = Name.DeviceRGB
    image[Name.BitsPerComponent] = 8
    image[Name.Filter] = Name.DCTDecode
    page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Im0=image))
    page.obj[Name.Contents] = pdf.make_stream(
        f"q {page_size[0]} 0 0 {page_size[1]} 0 0 cm /Im0 Do Q".encode()
    )


def _nested_form_chain(
    pdf: Pdf,
    leaf: pikepdf.Object,
    count: int = 1200,
) -> pikepdf.Object:
    """Wrap a Form XObject in a deeply nested Form chain."""
    child = leaf
    for _ in range(count - 1):
        parent = pdf.make_stream(b"/Fm Do")
        parent[Name.Type] = Name.XObject
        parent[Name.Subtype] = Name.Form
        parent[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
        parent[Name.Resources] = Dictionary(XObject=Dictionary(Fm=child))
        child = parent
    return child


def _write_pdf_with_declared_content_length(
    path: Path,
    content: bytes,
    declared_length: int,
) -> None:
    """Write a minimal PDF whose stream length can intentionally be wrong."""
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
        b"/Resources <<>> /Contents 4 0 R >>",
        f"<< /Length {declared_length} >>\nstream\n".encode("ascii")
        + content
        + b"\nendstream",
    )
    raw_pdf = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, value in enumerate(objects, start=1):
        offsets.append(len(raw_pdf))
        raw_pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        raw_pdf.extend(value)
        raw_pdf.extend(b"\nendobj\n")
    xref_offset = len(raw_pdf)
    raw_pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    raw_pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        raw_pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    raw_pdf.extend(
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    path.write_bytes(raw_pdf)


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

    def test_needs_ocr_operator_budget_stops_before_public_walk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[Name.Contents] = pdf.make_stream(b"")
            consumed = 0

            def many_operators(_owner: object):
                nonlocal consumed
                for _ in range(200):
                    consumed += 1
                    yield SimpleNamespace(operator=pikepdf.Operator("q"), operands=())

            monkeypatch.setattr(pikepdf, "parse_content_stream", many_operators)
            monkeypatch.setattr(
                digital_layout,
                "_MAX_DIGITAL_OPERATORS_PER_PAGE",
                100,
            )
            with (
                patch("pdftopdfa.ocr._page_has_images") as page_has_images,
                patch("pdftopdfa.ocr._page_has_text") as page_has_text,
                pytest.raises(OCRError, match="page operator budget exceeded"),
            ):
                needs_ocr(pdf)

        assert consumed == digital_layout._MAX_DIGITAL_OPERATORS_PER_PAGE + 1
        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_rejects_underdeclared_encoded_content_before_analysis(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "needs-ocr-underdeclared-content.pdf"
        _write_pdf_with_declared_content_length(path, b"q Q q Q", 1)
        original = path.read_bytes()
        monkeypatch.setattr(
            digital_layout,
            "_MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER",
            5,
        )

        with (
            Pdf.open(path) as pdf,
            patch("pdftopdfa.ocr._page_has_images") as page_has_images,
            patch("pdftopdfa.ocr._page_has_text") as page_has_text,
            pytest.raises(OCRError, match="encoded content container byte budget"),
        ):
            needs_ocr(pdf)

        assert path.read_bytes() == original
        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_rejects_decoded_content_before_analysis(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            stream = pdf.make_stream(b"")
            stream.write(
                zlib.compress(b" " * 4_096 + b"q Q"),
                filter=Name.FlateDecode,
            )
            page.obj[Name.Contents] = stream
            monkeypatch.setattr(
                digital_layout,
                "_MAX_DECODED_CONTENT_BYTES_PER_CONTAINER",
                128,
            )

            with (
                patch("pdftopdfa.ocr._page_has_images") as page_has_images,
                patch("pdftopdfa.ocr._page_has_text") as page_has_text,
                pytest.raises(OCRError, match="content container byte budget exceeded"),
            ):
                needs_ocr(pdf)

        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_rejects_form_nesting_before_analysis(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            leaf = pdf.make_stream(b"q Q")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary()
            root = _nested_form_chain(pdf, leaf, count=66)
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=root))
            page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")

            with (
                patch("pdftopdfa.ocr._page_has_images") as page_has_images,
                patch("pdftopdfa.ocr._page_has_text") as page_has_text,
                pytest.raises(OCRError, match="nesting depth budget exceeded"),
            ):
                needs_ocr(pdf)

        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_rejects_recursive_form_before_analysis(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"/Self Do")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary(XObject=Dictionary(Self=form))
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")

            with (
                patch("pdftopdfa.ocr._page_has_images") as page_has_images,
                patch("pdftopdfa.ocr._page_has_text") as page_has_text,
                pytest.raises(OCRError, match="Form XObject is recursive"),
            ):
                needs_ocr(pdf)

        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_rejects_repeated_form_invocations_before_analysis(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"q Q")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary()
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Fm=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Fm Do /Fm Do /Fm Do")
            monkeypatch.setattr(
                digital_layout,
                "_MAX_FORM_INVOCATIONS_PER_RESOURCE_PER_PAGE",
                2,
            )

            with (
                patch("pdftopdfa.ocr._page_has_images") as page_has_images,
                patch("pdftopdfa.ocr._page_has_text") as page_has_text,
                pytest.raises(OCRError, match="invocation budget exceeded"),
            ):
                needs_ocr(pdf)

        page_has_images.assert_not_called()
        page_has_text.assert_not_called()

    def test_needs_ocr_accepts_bounded_invoked_form_text(self) -> None:
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_form_text=True)

            assert needs_ocr(pdf) is False

    def test_needs_ocr_accepts_opened_encrypted_pdf_without_mutation(
        self,
        encrypted_pdf: Path,
    ) -> None:
        original = encrypted_pdf.read_bytes()

        with Pdf.open(encrypted_pdf, password="testpassword") as pdf:
            assert needs_ocr(pdf) is False

        assert encrypted_pdf.read_bytes() == original

    def test_page_image_and_text_detection(
        self,
        pdf_with_image_obj: Pdf,
        pdf_with_text_obj: Pdf,
    ) -> None:
        assert _page_has_images(pdf_with_image_obj.pages[0]) is True
        assert _page_has_text(pdf_with_image_obj.pages[0]) is False
        assert _page_has_images(pdf_with_text_obj.pages[0]) is False
        assert _page_has_text(pdf_with_text_obj.pages[0]) is True

    def test_unreferenced_form_text_is_not_page_text(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"BT (Unused) Tj ET")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary()
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Unused=form))
            page.obj[Name.Contents] = pdf.make_stream(b"q Q")

            assert _page_has_text(page) is False

    @pytest.mark.parametrize("order", [b"/A Do /B Do", b"/B Do /A Do"])
    @pytest.mark.parametrize("empty_shared_resources", [False, True])
    def test_form_text_detection_tracks_effective_resource_context(
        self,
        order: bytes,
        empty_shared_resources: bool,
    ) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))

            def form(content: bytes, resources: Dictionary | None) -> pikepdf.Stream:
                value = pdf.make_stream(content)
                value[Name.Type] = Name.XObject
                value[Name.Subtype] = Name.Form
                value[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
                if resources is not None:
                    value[Name.Resources] = resources
                return value

            shared = form(
                b"/Nested Do",
                Dictionary() if empty_shared_resources else None,
            )
            empty = form(b"q Q", Dictionary())
            text = form(b"BT (Visible) Tj ET", Dictionary())
            first = form(
                b"/Shared Do",
                Dictionary(XObject=Dictionary(Shared=shared, Nested=empty)),
            )
            second = form(
                b"/Shared Do",
                Dictionary(XObject=Dictionary(Shared=shared, Nested=text)),
            )
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(A=first, B=second))
            page.obj[Name.Contents] = pdf.make_stream(order)

            assert _page_has_text(page) is True

    def test_resource_less_self_referential_form_text_detection_terminates(
        self,
    ) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"/Self Do")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Self=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Self Do")

            assert _page_has_text(page) is False

    def test_shared_form_with_own_resources_still_detects_text(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"BT (Shared) Tj ET")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary(ProcSet=pikepdf.Array([Name.PDF]))
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Shared=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Shared Do /Shared Do")

            assert _page_has_text(page) is True

    def test_whitespace_only_text_does_not_suppress_ocr_detection(self) -> None:
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            page = pdf.pages[0]
            page.Contents.write(
                page.Contents.read_bytes() + b"\nBT /F1 11 Tf [( ) 10.5 ( )] TJ ET"
            )

            assert _page_has_text(page) is False
            assert needs_ocr(pdf) is True

    def test_text_detection_handles_1200_nested_forms(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            leaf = pdf.make_stream(b"BT 3 Tr (deep OCR text) Tj ET")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary()
            root = _nested_form_chain(pdf, leaf)
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(DeepForm=root))
            page.obj[Name.Contents] = pdf.make_stream(b"/DeepForm Do")

            assert _page_has_text(page) is True

    @pytest.mark.parametrize("color_space_kind", ["alias", "indexed"])
    def test_color_space_analysis_handles_1200_nested_values(
        self,
        color_space_kind: str,
    ) -> None:
        with Pdf.new():
            resources = Dictionary()
            if color_space_kind == "alias":
                color_spaces = Dictionary()
                for index in range(1199):
                    color_spaces[Name(f"/CS{index}")] = Name(f"/CS{index + 1}")
                color_spaces[Name("/CS1199")] = Name.DeviceGray
                resources[Name.ColorSpace] = color_spaces
                color_space = Name("/CS0")
            else:
                color_space = Name.DeviceGray
                for _ in range(1200):
                    color_space = pikepdf.Array(
                        [
                            Name.Indexed,
                            color_space,
                            0,
                            pikepdf.String(b"\x00"),
                        ]
                    )

            assert _image_color_space_marks(color_space, resources) is True

    def test_color_space_analysis_rejects_indexed_cycle(self) -> None:
        with Pdf.new() as pdf:
            color_space = pdf.make_indirect(
                pikepdf.Array(
                    [
                        Name.Indexed,
                        Name.DeviceGray,
                        0,
                        pikepdf.String(b"\x00"),
                    ]
                )
            )
            color_space[1] = color_space

            assert _image_color_space_marks(color_space, Dictionary()) is False


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

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_zero_bases_ocr_form_box_for_every_rotation(
        self,
        tmp_dir: Path,
        rotation: int,
    ) -> None:
        path = tmp_dir / f"rotated-{rotation}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(576, 432))
            page.obj[Name.MediaBox] = pikepdf.Array([10, 20, 586, 452])
            page.obj[Name.Rotate] = rotation
            page.obj[Name.UserUnit] = 2.5
            form = pdf.make_stream(b"")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([10, 20, 586, 452])
            xobjects = Dictionary()
            xobjects[Name("/OCR-pdf-0")] = form
            page.obj[Name.Resources] = Dictionary(XObject=xobjects)
            pdf.save(path)

        new_forms = _finalize_ocr_output(path, ["en"], [frozenset()])

        with Pdf.open(path) as pdf:
            form = pdf.pages[0].Resources.XObject["/OCR-pdf-0"]
            expected = [0, 0, 432, 576] if rotation in {90, 270} else [0, 0, 576, 432]
            assert [float(value) for value in form.BBox] == expected
        assert new_forms == {0: ("/OCR-pdf-0",)}

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
        new_forms = _finalize_ocr_output(path, ["en"], existing_names)

        with Pdf.open(path) as pdf:
            form = pdf.pages[0].resources.XObject["/OCR-existing"]
            assert [float(value) for value in form.BBox] == [0, 0, 576, 432]
        assert new_forms == {}


class TestOcrResourcePreflight:
    """Tests for fail-closed page geometry and raster limits."""

    @pytest.mark.parametrize(
        ("page_size", "expected_oversample"),
        [
            (758.0, 600),
            (760.0, 300),
        ],
    )
    def test_plans_safe_dpi_with_rotation_and_user_unit(
        self,
        tmp_dir: Path,
        page_size: float,
        expected_oversample: int,
    ) -> None:
        path = tmp_dir / f"raster-{page_size}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(page_size, page_size))
            page.obj[Name.Rotate] = 90
            page.obj[Name.UserUnit] = 2.5
            pdf.save(path)

        assert _preflight_ocr_input(path) == expected_oversample

    def test_rejects_page_that_exceeds_limit_at_300_dpi(
        self,
        tmp_dir: Path,
    ) -> None:
        path = tmp_dir / "too-large-at-300.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(3800, 3800))
            pdf.save(path)

        with pytest.raises(
            OCRError,
            match=(
                "OCR page 1.*at 300 dpi.*"
                f"safety limit is {_OCR_MAX_PAGE_RASTER_PIXELS:,}"
            ),
        ):
            _preflight_ocr_input(path)

    def test_allows_a0_page_at_fallback_dpi(self, tmp_dir: Path) -> None:
        path = tmp_dir / "a0.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(2384, 3370))
            pdf.save(path)

        assert _preflight_ocr_input(path) == 300

    def test_native_dpi_is_capped_at_planned_fallback(
        self,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_dir / "high-native-dpi.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(1500, 1500))
            pdf.save(path)
        pdfinfo_value = SimpleNamespace(
            pages=[
                SimpleNamespace(
                    has_text=False,
                    images=[],
                    dpi=SimpleNamespace(x=800.0, y=800.0),
                )
            ]
        )

        with patch("ocrmypdf.pdfinfo.PdfInfo", return_value=pdfinfo_value):
            assert _preflight_ocr_input(path) == 300

        assert "preferred 600 dpi oversampling (effective 800 dpi)" in caplog.text
        assert "using 300 dpi" in caplog.text
        assert "39,062,500 pixels expected" in caplog.text

    def test_allows_large_document_with_safe_individual_pages(
        self,
        tmp_dir: Path,
    ) -> None:
        path = tmp_dir / "large-raster-document.pdf"
        with Pdf.new() as pdf:
            for _ in range(29):
                pdf.add_blank_page(page_size=(595, 842))
            pdf.save(path)

        assert _preflight_ocr_input(path) == 600

    def test_enforces_document_page_count_before_pdfinfo_analysis(
        self,
        tmp_dir: Path,
    ) -> None:
        path = tmp_dir / "too-many-pages.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(10, 10))
            pdf.add_blank_page(page_size=(10, 10))
            pdf.save(path)

        with (
            patch("pdftopdfa.ocr._OCR_MAX_DOCUMENT_PAGES", 1),
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="more than 1 pages"),
        ):
            _preflight_ocr_input(path)

        pdfinfo.assert_not_called()

    def test_operator_budget_stops_before_consuming_unbounded_iterator(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "operator-budget.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[Name.Contents] = pdf.make_stream(b"")
            pdf.save(path)

        consumed = 0

        def many_operators(_owner: object):
            nonlocal consumed
            for _ in range(200):
                consumed += 1
                yield SimpleNamespace(operator=pikepdf.Operator("q"), operands=())

        monkeypatch.setattr(pikepdf, "parse_content_stream", many_operators)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_DIGITAL_OPERATORS_PER_PAGE",
            100,
        )
        with (
            patch("pdftopdfa.ocr._page_has_text") as page_has_text,
            patch("pdftopdfa.ocr._page_paint_analysis") as paint_analysis,
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="page operator budget exceeded"),
        ):
            _preflight_ocr_input(path)

        assert consumed == digital_layout._MAX_DIGITAL_OPERATORS_PER_PAGE + 1
        assert consumed < 200
        page_has_text.assert_not_called()
        paint_analysis.assert_not_called()
        pdfinfo.assert_not_called()

    def test_document_operator_budget_runs_before_ocr_walkers(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "document-operator-budget.pdf"
        with Pdf.new() as pdf:
            for _ in range(2):
                page = pdf.add_blank_page(page_size=(100, 100))
                page.obj[Name.Contents] = pdf.make_stream(b"")
            pdf.save(path)
        consumed = 0

        def two_operators(_owner: object):
            nonlocal consumed
            for _ in range(2):
                consumed += 1
                yield SimpleNamespace(operator=pikepdf.Operator("q"), operands=())

        monkeypatch.setattr(pikepdf, "parse_content_stream", two_operators)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_DIGITAL_OPERATORS_PER_DOCUMENT",
            3,
        )

        with (
            patch("pdftopdfa.ocr._page_has_text") as page_has_text,
            patch("pdftopdfa.ocr._page_paint_analysis") as paint_analysis,
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="document operator budget exceeded"),
        ):
            _preflight_ocr_input(path)

        assert consumed == 4
        page_has_text.assert_not_called()
        paint_analysis.assert_not_called()
        pdfinfo.assert_not_called()

    def test_rejects_decoded_content_bomb_before_ocr_walkers(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "decoded-content-bomb.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            stream = pdf.make_stream(b"")
            stream.write(
                zlib.compress(b" " * 4_096 + b"q Q"),
                filter=Name.FlateDecode,
            )
            page.obj[Name.Contents] = stream
            pdf.save(path)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_DECODED_CONTENT_BYTES_PER_CONTAINER",
            128,
        )

        with (
            patch("pdftopdfa.ocr._page_has_text") as page_has_text,
            patch("pdftopdfa.ocr._page_paint_analysis") as paint_analysis,
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="content container byte budget exceeded"),
        ):
            _preflight_ocr_input(path)

        page_has_text.assert_not_called()
        paint_analysis.assert_not_called()
        pdfinfo.assert_not_called()

    def test_rejects_cumulative_encoded_content_before_ocr_walkers(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "encoded-content-budget.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[Name.Contents] = pikepdf.Array(
                [pdf.make_stream(b"q Q"), pdf.make_stream(b"q Q")]
            )
            pdf.save(path, compress_streams=False)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_ENCODED_CONTENT_BYTES_PER_PAGE",
            5,
        )

        with (
            patch("pdftopdfa.ocr._page_has_text") as page_has_text,
            patch("pdftopdfa.ocr._page_paint_analysis") as paint_analysis,
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="page encoded-content byte budget"),
        ):
            _preflight_ocr_input(path)

        page_has_text.assert_not_called()
        paint_analysis.assert_not_called()
        pdfinfo.assert_not_called()

    def test_canonical_copy_repairs_underdeclared_length_before_raw_read(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "underdeclared-content-length.pdf"
        _write_pdf_with_declared_content_length(path, b"q Q q Q", 1)
        observed_lengths: list[tuple[int, int]] = []
        original_sizes = digital_layout._content_stream_sizes

        def inspect_canonical_length(
            stream: pikepdf.Stream,
            decoded_limit: int,
        ) -> tuple[int, int]:
            declared_length = int(stream.get("/Length"))
            raw_length = len(stream.get_raw_stream_buffer())
            observed_lengths.append((declared_length, raw_length))
            return original_sizes(stream, decoded_limit)

        monkeypatch.setattr(
            digital_layout,
            "_content_stream_sizes",
            inspect_canonical_length,
        )
        monkeypatch.setattr(
            digital_layout,
            "_MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER",
            5,
        )

        with (
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="encoded content container byte budget"),
        ):
            _preflight_ocr_input(path)

        assert observed_lengths
        assert observed_lengths[0][0] == observed_lengths[0][1]
        assert observed_lengths[0][0] > 5
        pdfinfo.assert_not_called()

    def test_rejects_form_nesting_depth_before_pdfinfo(
        self,
        tmp_dir: Path,
    ) -> None:
        path = tmp_dir / "deep-forms.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            leaf = pdf.make_stream(b"q Q")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary()
            root = _nested_form_chain(pdf, leaf, count=66)
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=root))
            page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")
            pdf.save(path)

        with (
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="nesting depth budget exceeded"),
        ):
            _preflight_ocr_input(path)

        pdfinfo.assert_not_called()

    def test_rejects_repeated_form_invocations_before_pdfinfo(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "repeated-form.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"q Q")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary()
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Fm=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Fm Do /Fm Do /Fm Do")
            pdf.save(path)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_FORM_INVOCATIONS_PER_RESOURCE_PER_PAGE",
            2,
        )

        with (
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="invocation budget exceeded"),
        ):
            _preflight_ocr_input(path)

        pdfinfo.assert_not_called()

    def test_rejects_recursive_form_before_pdfinfo(self, tmp_dir: Path) -> None:
        path = tmp_dir / "recursive-form.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"/Self Do")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary(XObject=Dictionary(Self=form))
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=form))
            page.obj[Name.Contents] = pdf.make_stream(b"/Root Do")
            pdf.save(path)

        with (
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="Form XObject is recursive"),
        ):
            _preflight_ocr_input(path)

        pdfinfo.assert_not_called()

    def test_forced_ocr_budgets_unreferenced_existing_ocr_forms(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_dir / "unreferenced-existing-ocr-form.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            form = pdf.make_stream(b"q Q q Q")
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            form[Name.Resources] = Dictionary()
            xobjects = Dictionary()
            xobjects[Name("/OCR-existing")] = form
            page.obj[Name.Resources] = Dictionary(XObject=xobjects)
            page.obj[Name.Contents] = pdf.make_stream(b"")
            pdf.save(path)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_DIGITAL_OPERATORS_PER_PAGE",
            3,
        )

        with (
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            pytest.raises(OCRError, match="page operator budget exceeded"),
        ):
            _preflight_ocr_input(path, force=True)

        pdfinfo.assert_not_called()

    def test_accepts_bounded_nested_and_reused_forms(self, tmp_dir: Path) -> None:
        path = tmp_dir / "bounded-complex-forms.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            font = Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name.Helvetica,
            )
            leaf = pdf.make_stream(b"BT /F1 10 Tf 10 10 Td (Text) Tj ET")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
            root = _nested_form_chain(pdf, leaf, count=5)
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Root=root))
            page.obj[Name.Contents] = pdf.make_stream(b"/Root Do /Root Do")
            pdf.save(path)
        pdfinfo_value = SimpleNamespace(
            pages=[SimpleNamespace(has_text=True, images=[])]
        )

        with patch("ocrmypdf.pdfinfo.PdfInfo", return_value=pdfinfo_value) as pdfinfo:
            _preflight_ocr_input(path)

        pdfinfo.assert_called_once_with(path, max_workers=1)

    def test_content_preflight_preserves_existing_output_and_manifest(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_path = tmp_dir / "oversized-content.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[Name.Contents] = pikepdf.Array(
                [pdf.make_stream(b"q Q"), pdf.make_stream(b"q Q")]
            )
            pdf.save(input_path, compress_streams=False)
        output_sentinel = b"existing output"
        manifest_sentinel = b"existing manifest"
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)
        monkeypatch.setattr(
            digital_layout,
            "_MAX_ENCODED_CONTENT_BYTES_PER_PAGE",
            5,
        )

        with (
            patch("pdftopdfa.ocr._require_ocr_runtime") as require_runtime,
            patch("pdftopdfa.ocr_paddle.validate_model_directories") as validate_models,
            patch("pdftopdfa.ocr.private_staging_directory") as stage_output,
            patch("ocrmypdf.pdfinfo.PdfInfo") as pdfinfo,
            patch("pdftopdfa.ocr.ocrmypdf.ocr") as run_ocr,
            pytest.raises(OCRError, match="page encoded-content byte budget"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        require_runtime.assert_called_once_with("cpu")
        validate_models.assert_not_called()
        stage_output.assert_not_called()
        pdfinfo.assert_not_called()
        run_ocr.assert_not_called()
        assert output_path.read_bytes() == output_sentinel
        assert manifest_path.read_bytes() == manifest_sentinel

    @pytest.mark.parametrize(
        ("box_name", "coordinates", "message"),
        [
            (Name.MediaBox, [0, 0, 2, 100], "page-boundary limits"),
            (Name.MediaBox, [0, 0, 14_401, 100], "page-boundary limits"),
            (Name.MediaBox, [100, 0, 0, 100], "non-normalized MediaBox"),
            (Name.CropBox, [-1, 0, 90, 90], "outside its MediaBox"),
        ],
    )
    def test_rejects_page_boxes_that_pdfa_sanitization_would_change(
        self,
        tmp_dir: Path,
        box_name: Name,
        coordinates: list[int],
        message: str,
    ) -> None:
        path = tmp_dir / f"invalid-{str(box_name)[1:]}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[box_name] = pikepdf.Array(coordinates)
            pdf.save(path)

        with pytest.raises(OCRError, match=message):
            _preflight_ocr_input(path)

    @pytest.mark.parametrize(
        ("page_size", "rotate_pages", "exceeds_limit"),
        [
            (4216.0, True, False),
            (4217.0, True, True),
            (4217.0, False, False),
        ],
    )
    def test_enforces_orientation_render_pixel_limit_on_text_pages(
        self,
        tmp_dir: Path,
        page_size: float,
        rotate_pages: bool,
        exceeds_limit: bool,
    ) -> None:
        path = tmp_dir / f"orientation-{page_size}-{rotate_pages}.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_text=True)
            page = pdf.pages[0]
            page.obj[Name.MediaBox] = pikepdf.Array(
                [10, 20, 10 + page_size, 20 + page_size]
            )
            page.obj[Name.CropBox] = page.obj[Name.MediaBox]
            page.obj[Name.Rotate] = 90
            page.obj[Name.UserUnit] = 2.5
            pdf.save(path)

        if exceeds_limit:
            with pytest.raises(
                OCRError,
                match="Paddle orientation page 1.*at 108 dpi",
            ):
                _preflight_ocr_input(path, rotate_pages=rotate_pages)
        else:
            _preflight_ocr_input(path, rotate_pages=rotate_pages)

    @pytest.mark.parametrize("rotate_pages", [False, True])
    def test_user_unit_cannot_bypass_raster_preflight(
        self,
        tmp_dir: Path,
        rotate_pages: bool,
    ) -> None:
        path = tmp_dir / f"large-user-unit-{rotate_pages}.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj[Name.UserUnit] = 1000
            pdf.save(path)

        with pytest.raises(OCRError, match="safety limit"):
            _preflight_ocr_input(path, rotate_pages=rotate_pages)

    def test_rejects_fractional_inherited_rotation(self, tmp_dir: Path) -> None:
        path = tmp_dir / "fractional-rotation.pdf"
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            page.obj.Parent[Name.Rotate] = 90.5
            if Name.Rotate in page.obj:
                del page.obj[Name.Rotate]
            pdf.save(path)

        with pytest.raises(OCRError, match="rotation is not a multiple of 90"):
            _preflight_ocr_input(path)


def test_ocr_cleanup_reports_failures_after_attempting_every_resource() -> None:
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()
    first.cleanup.side_effect = RuntimeError("locked")

    with patch(
        "pdftopdfa.ocr._release_ocr_models",
        side_effect=RuntimeError("close failed"),
    ) as release_models:
        with pytest.raises(OCRError, match="OCR cleanup failed"):
            _cleanup_ocr_resources(
                (
                    ("first", first),
                    ("second", second),
                    ("third", third),
                )
            )

    release_models.assert_called_once_with()
    first.cleanup.assert_called_once_with()
    second.cleanup.assert_called_once_with()
    third.cleanup.assert_called_once_with()


def test_ocr_cleanup_preserves_primary_error(caplog: pytest.LogCaptureFixture) -> None:
    first = MagicMock()
    second = MagicMock()
    first.cleanup.side_effect = RuntimeError("locked")

    with patch(
        "pdftopdfa.ocr._release_ocr_models",
        side_effect=RuntimeError("close failed"),
    ):
        _cleanup_ocr_resources(
            (("first", first), ("second", second)),
            preserve_primary_error=True,
        )

    first.cleanup.assert_called_once_with()
    second.cleanup.assert_called_once_with()
    assert "Could not release OCR models" in caplog.text
    assert "Could not clean up first" in caplog.text


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
    @pytest.mark.parametrize("manifest_alias", ["input", "output"])
    def test_rejects_manifest_path_aliasing_pdf_paths(
        self,
        tmp_dir: Path,
        manifest_alias: str,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        output_sentinel = b"existing output"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        input_sentinel = input_path.read_bytes()
        output_path.write_bytes(output_sentinel)
        manifest_path = input_path if manifest_alias == "input" else output_path

        with pytest.raises(OCRError, match="manifest path must differ"):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=tmp_dir / "unused-detection-model",
                recognition_model_dir=tmp_dir / "unused-recognition-model",
                _manifest_output_path=manifest_path,
            )

        assert input_path.read_bytes() == input_sentinel
        assert output_path.read_bytes() == output_sentinel

    """Tests for the fixed PaddleOCR/OCRmyPDF boundary."""

    def test_deskew_analysis_handles_scan_in_1200_nested_forms(self) -> None:
        with Pdf.new() as pdf:
            page = pdf.add_blank_page(page_size=(100, 100))
            image = pdf.make_stream(bytes([128]) * 100)
            image[Name.Type] = Name.XObject
            image[Name.Subtype] = Name.Image
            image[Name.Width] = 10
            image[Name.Height] = 10
            image[Name.ColorSpace] = Name.DeviceGray
            image[Name.BitsPerComponent] = 8
            leaf = pdf.make_stream(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary(XObject=Dictionary(Im0=image))
            root = _nested_form_chain(pdf, leaf)
            page.obj[Name.Resources] = Dictionary(XObject=Dictionary(DeepForm=root))
            page.obj[Name.Contents] = pdf.make_stream(b"/DeepForm Do")

            analysis = _page_paint_analysis(page)

            assert analysis.unsafe is False
            assert analysis.image_candidates == [(1.0, 1)]

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
        source, staged_output = mock_ocr.call_args.args
        assert source == input_path
        assert staged_output.parent.parent == output_path.parent
        assert staged_output.parent.name.startswith(".output_ocr_")
        assert staged_output != output_path
        assert output_path.exists()
        kwargs = mock_ocr.call_args.kwargs
        assert kwargs["deskew"] is True
        assert kwargs["pages"] == "1"
        assert kwargs["jobs"] == 1
        assert kwargs["skip_text"] is True
        assert "force_ocr" not in kwargs
        assert "redo_ocr" not in kwargs

    @pytest.mark.parametrize("deskew", [False, True])
    def test_mixed_full_page_scan_uses_redo_and_preserves_native_page_content(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        deskew: bool,
    ) -> None:
        input_path = tmp_dir / "mixed-scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_text=True)
            page = pdf.pages[0]
            page.obj[Name.CropBox] = pikepdf.Array([5, 10, 95, 90])
            page.obj[Name.Rotate] = 90
            page.obj[Name.UserUnit] = 2
            page.obj[Name.Annots] = pikepdf.Array(
                [
                    Dictionary(
                        Type=Name.Annot,
                        Subtype=Name.Link,
                        Rect=pikepdf.Array([10, 20, 30, 40]),
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
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=deskew,
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.kwargs["pages"] == "1"
        assert mock_ocr.call_args.kwargs["redo_ocr"] is True
        assert mock_ocr.call_args.kwargs["deskew"] is False
        assert "skip_text" not in mock_ocr.call_args.kwargs
        with Pdf.open(input_path) as source, Pdf.open(output_path) as output:
            assert source.pages[0].Contents.read_bytes() == (
                output.pages[0].Contents.read_bytes()
            )
            assert list(output.pages[0].CropBox) == [5, 10, 95, 90]
            assert int(output.pages[0].Rotate) == 90
            assert int(output.pages[0].UserUnit) == 2
            assert list(output.pages[0].Annots[0].Rect) == [10, 20, 30, 40]

    @pytest.mark.parametrize("deskew", [False, True])
    @pytest.mark.parametrize("in_form", [False, True])
    def test_mixed_scan_filling_visible_cropbox_uses_redo(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        deskew: bool,
        in_form: bool,
    ) -> None:
        input_path = tmp_dir / "cropped-mixed-scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, visible_text=True)
            page = pdf.pages[0]
            image_name = b"/Im0 Do"
            if in_form:
                form = pdf.make_stream(b"/Im0 Do")
                form[Name.Type] = Name.XObject
                form[Name.Subtype] = Name.Form
                form[Name.BBox] = pikepdf.Array([0, 0, 1, 1])
                form[Name.Resources] = Dictionary(
                    XObject=Dictionary(Im0=page.Resources.XObject.Im0)
                )
                page.Resources.XObject[Name.CropScan] = form
                image_name = b"/CropScan Do"
            page.Contents.write(
                b"q 80 0 0 80 10 10 cm "
                + image_name
                + b" Q\nBT /F1 12 Tf 0 Tr 20 50 Td (Native text) Tj ET"
            )
            page.obj[Name.CropBox] = pikepdf.Array([10, 10, 90, 90])
            page.obj[Name.Rotate] = 90
            page.obj[Name.UserUnit] = 2
            assert _page_paint_analysis(page).image_candidates == [(1.0, 1)]
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
                deskew=deskew,
            )

        mock_ocr.assert_called_once()
        kwargs = mock_ocr.call_args.kwargs
        assert kwargs["pages"] == "1"
        assert kwargs["redo_ocr"] is True
        assert kwargs["deskew"] is False
        assert "skip_text" not in kwargs
        with Pdf.open(output_path) as pdf:
            assert list(pdf.pages[0].CropBox) == [10, 10, 90, 90]
            assert int(pdf.pages[0].Rotate) == 90
            assert int(pdf.pages[0].UserUnit) == 2

    def test_deskew_copies_digital_text_with_decorative_image_without_ocrmypdf(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "digital.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, image_scale=50, visible_text=True)
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

    def test_clipped_text_after_full_page_image_fails_closed(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "clipped-text.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            pdf.pages[0].Contents.write(
                b"q 100 0 0 100 0 0 cm /Im0 Do Q\n"
                b"0 0 1 1 re W n\n"
                b"BT /F1 12 Tf 0 Tr 50 50 Td (Clipped text) Tj ET"
            )
            pdf.save(input_path)

        with (
            patch("pdftopdfa.ocr.ocrmypdf.ocr") as mock_ocr,
            pytest.raises(OCRError, match="ambiguous scan-like page"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        mock_ocr.assert_not_called()
        assert not output_path.exists()

    def test_manifest_failure_preserves_existing_output_atomically(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        sentinel = b"existing output must survive"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        output_path.write_bytes(sentinel)

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=OCRError("manifest failed"),
            ),
            pytest.raises(OCRError, match="manifest failed"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert output_path.read_bytes() == sentinel
        assert not manifest_path.exists()
        assert not list(tmp_dir.glob(".output_ocr_*.pdf"))

    def test_finalized_ocr_output_cannot_be_swapped_before_publish(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        sentinel = b"existing output"
        output_path.write_bytes(sentinel)
        swapped = False

        def swap_before_verification(
            path: Path,
            snapshot: StagedFileSnapshot,
        ) -> None:
            nonlocal swapped
            if not swapped:
                swapped = True
                replacement = Path(path).with_name("replacement.pdf")
                replacement.write_bytes(b"different bytes")
                os.replace(replacement, path)
            verify_staged_file_snapshot(path, snapshot)

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.staging.verify_staged_file",
                side_effect=swap_before_verification,
            ),
            pytest.raises(OCRError, match="publish OCR output atomically"),
        ):
            apply_ocr(
                sample_pdf,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".output_ocr_*"))

    def test_failed_publish_recovery_retains_original_backup(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"original output"
        manifest_sentinel = b"original manifest"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)

        def publish_with_failed_recovery(
            staged: Path,
            destination: Path,
            expected: StagedFileSnapshot,
            **kwargs: object,
        ) -> StagedFileSnapshot:
            if destination == manifest_path:
                raise PermissionError("manifest locked")
            return publish_staged_file_impl(
                staged,
                destination,
                expected,
                **kwargs,
            )

        def fail_output_recovery(
            destination: Path,
            candidate: StagedFileSnapshot,
            **kwargs: object,
        ) -> None:
            if Path(destination) == output_path:
                raise PermissionError("output restore locked")
            rollback_staged_publication_impl(
                destination,
                candidate,
                **kwargs,
            )

        def write_staged_manifest(destination: Path, *_args: object) -> None:
            Path(destination).write_text("{}", encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=write_staged_manifest,
            ),
            patch(
                "pdftopdfa.ocr.publish_staged_file",
                side_effect=publish_with_failed_recovery,
            ),
            patch(
                "pdftopdfa.ocr.rollback_staged_publication",
                side_effect=fail_output_recovery,
            ),
            pytest.raises(OCRError, match="recovery copy retained"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        backups = list(tmp_dir.glob(".output_ocr_*/backup.pdf"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == output_sentinel
        assert manifest_path.read_bytes() == manifest_sentinel

    def test_manifest_publish_failure_restores_existing_targets(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"original output"
        manifest_sentinel = b"original manifest"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)
        output_peer = tmp_dir / "output-peer.pdf"
        try:
            os.link(output_path, output_peer)
        except OSError as exc:
            pytest.skip(f"Hard links are not supported: {exc}")

        def fail_manifest_publish(
            staged: Path,
            destination: Path,
            expected: StagedFileSnapshot,
            **kwargs: object,
        ) -> StagedFileSnapshot:
            if Path(destination) == manifest_path:
                raise PermissionError("manifest locked")
            return publish_staged_file_impl(
                staged,
                destination,
                expected,
                **kwargs,
            )

        def write_staged_manifest(destination: Path, *_args: object) -> None:
            Path(destination).write_text("{}", encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=write_staged_manifest,
            ),
            patch(
                "pdftopdfa.ocr.publish_staged_file",
                side_effect=fail_manifest_publish,
            ),
            pytest.raises(OCRError, match="publish OCR output atomically"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert output_path.read_bytes() == output_sentinel
        assert os.path.samefile(output_path, output_peer)
        assert manifest_path.read_bytes() == manifest_sentinel
        assert not list(tmp_dir.glob(".*_ocr_*"))
        assert not list(tmp_dir.glob(".*_backup_*"))

    def test_manifest_failure_restores_target_created_during_publication(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        concurrent_output = b"concurrent output"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)

        def publish_with_concurrent_target(
            staged: Path,
            destination: Path,
            expected: StagedFileSnapshot,
            **kwargs: object,
        ) -> StagedFileSnapshot:
            if Path(destination) == output_path:
                output_path.write_bytes(concurrent_output)
            else:
                raise PermissionError("manifest locked")
            return publish_staged_file_impl(
                staged,
                destination,
                expected,
                **kwargs,
            )

        def write_staged_manifest(destination: Path, *_args: object) -> None:
            Path(destination).write_text("{}", encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=write_staged_manifest,
            ),
            patch(
                "pdftopdfa.ocr.publish_staged_file",
                side_effect=publish_with_concurrent_target,
            ),
            pytest.raises(OCRError, match="publish OCR output atomically"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert output_path.read_bytes() == concurrent_output
        assert not manifest_path.exists()
        assert not list(tmp_dir.glob(".*_ocr_*"))

    def test_missing_hardlink_support_fails_before_publication(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        sentinel = b"original output"
        output_path.write_bytes(sentinel)

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.staging.os.link",
                side_effect=OSError("hard links unavailable"),
            ),
            pytest.raises(OCRError, match="retain publication target"),
        ):
            apply_ocr(
                sample_pdf,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
            )

        assert output_path.read_bytes() == sentinel
        assert not list(tmp_dir.glob(".*_ocr_*"))

    @pytest.mark.parametrize("existing_targets", [False, True])
    def test_keyboard_interrupt_after_manifest_publish_restores_both_targets(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        existing_targets: bool,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"original output"
        manifest_sentinel = b"original manifest"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        if existing_targets:
            output_path.write_bytes(output_sentinel)
            manifest_path.write_bytes(manifest_sentinel)

        def publish_then_interrupt(
            staged: Path,
            destination: Path,
            expected: StagedFileSnapshot,
            **kwargs: object,
        ) -> StagedFileSnapshot:
            published = publish_staged_file_impl(
                staged,
                destination,
                expected,
                **kwargs,
            )
            if Path(destination) == manifest_path:
                raise KeyboardInterrupt
            return published

        def write_staged_manifest(destination: Path, *_args: object) -> None:
            Path(destination).write_text("{}", encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=write_staged_manifest,
            ),
            patch(
                "pdftopdfa.ocr.publish_staged_file",
                side_effect=publish_then_interrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        if existing_targets:
            assert output_path.read_bytes() == output_sentinel
            assert manifest_path.read_bytes() == manifest_sentinel
        else:
            assert not output_path.exists()
            assert not manifest_path.exists()
        assert not list(tmp_dir.glob(".*_ocr_*"))

    def test_cleanup_failure_does_not_mask_publish_error(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"original output"
        manifest_sentinel = b"original manifest"
        with Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(input_path)
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)
        real_cleanup = TemporaryDirectory.cleanup
        manifest_publish_failed = False

        def fail_manifest_publish(
            staged: Path,
            destination: Path,
            expected: StagedFileSnapshot,
            **kwargs: object,
        ) -> StagedFileSnapshot:
            nonlocal manifest_publish_failed
            if Path(destination) == manifest_path:
                manifest_publish_failed = True
                raise PermissionError("manifest locked")
            return publish_staged_file_impl(
                staged,
                destination,
                expected,
                **kwargs,
            )

        def fail_staged_manifest_cleanup(
            directory: TemporaryDirectory[str],
        ) -> None:
            if manifest_publish_failed and Path(directory.name).name.startswith(
                ".manifest_ocr_"
            ):
                raise PermissionError("staged manifest locked")
            real_cleanup(directory)

        def write_staged_manifest(destination: Path, *_args: object) -> None:
            Path(destination).write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            TemporaryDirectory,
            "cleanup",
            fail_staged_manifest_cleanup,
        )
        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=_copy_ocr_input,
            ),
            patch(
                "pdftopdfa.ocr._write_ocr_document_manifest",
                side_effect=write_staged_manifest,
            ),
            patch(
                "pdftopdfa.ocr.publish_staged_file",
                side_effect=fail_manifest_publish,
            ),
            pytest.raises(OCRError, match="publish OCR output atomically"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert output_path.read_bytes() == output_sentinel
        assert manifest_path.read_bytes() == manifest_sentinel
        assert not list(tmp_dir.glob(".*_backup_*"))
        assert "Could not remove private OCR staging directory" in caplog.text

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

    def test_regular_ocr_replaces_invisible_scan_text_with_paddle_output(
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
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.kwargs["pages"] == "1"
        assert mock_ocr.call_args.kwargs["skip_text"] is True
        assert prepared_page_has_text == [False]

    @pytest.mark.parametrize("deskew", [False, True])
    def test_ambiguous_scan_with_foreign_text_fails_closed(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        deskew: bool,
    ) -> None:
        input_path = tmp_dir / "ambiguous-ocr-scan.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, hidden_form_text=True, vector=True)
            pdf.save(input_path)

        with (
            patch("pdftopdfa.ocr.ocrmypdf.ocr") as mock_ocr,
            pytest.raises(OCRError, match="ambiguous scan-like page"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=deskew,
            )

        mock_ocr.assert_not_called()
        assert not output_path.exists()

    def test_prior_ocr_aborts_deskew_without_publishing_foreign_text(
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

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=PriorOcrFoundError(),
            ),
            pytest.raises(OCRError, match="already contains an OCR text layer"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
            )

        assert not output_path.exists()

    def test_prior_ocr_aborts_regular_run_and_preserves_atomic_targets(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"existing output"
        manifest_sentinel = b"existing manifest"
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=PriorOcrFoundError(),
            ),
            pytest.raises(OCRError, match="already contains an OCR text layer"),
        ):
            apply_ocr(
                sample_pdf,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert output_path.read_bytes() == output_sentinel
        assert manifest_path.read_bytes() == manifest_sentinel
        assert not list(tmp_dir.glob(".*_ocr_*"))

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

        assert mock_ocr.call_count == 3
        regular_call, deskew_call, redo_call = mock_ocr.call_args_list
        assert regular_call.kwargs["pages"] == "4"
        assert regular_call.kwargs["deskew"] is False
        assert deskew_call.kwargs["pages"] == "1,3"
        assert deskew_call.kwargs["deskew"] is True
        assert redo_call.kwargs["pages"] == "2"
        assert redo_call.kwargs["deskew"] is False
        assert redo_call.kwargs["redo_ocr"] is True
        assert "skip_text" not in redo_call.kwargs
        assert all(
            call.kwargs["skip_text"] is True and "force_ocr" not in call.kwargs
            for call in (regular_call, deskew_call)
        )

    def test_regular_and_redo_pages_are_disjoint_without_deskew(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "mixed-without-deskew.pdf"
        output_path = tmp_dir / "output.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            _add_content_page(pdf, image_scale=50, visible_text=True)
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
            )

        assert mock_ocr.call_count == 2
        regular_call, redo_call = mock_ocr.call_args_list
        assert regular_call.kwargs["pages"] == "1"
        assert regular_call.kwargs["skip_text"] is True
        assert regular_call.kwargs["deskew"] is False
        assert redo_call.kwargs["pages"] == "2"
        assert redo_call.kwargs["redo_ocr"] is True
        assert redo_call.kwargs["deskew"] is False
        assert "skip_text" not in redo_call.kwargs

    def test_prior_ocr_in_second_stage_does_not_publish_partial_result(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "mixed-without-deskew.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "manifest.json"
        output_sentinel = b"existing output"
        manifest_sentinel = b"existing manifest"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, visible_text=True)
            pdf.save(input_path)
        output_path.write_bytes(output_sentinel)
        manifest_path.write_bytes(manifest_sentinel)
        calls = 0

        def copy_then_reject_prior_ocr(
            source: Path,
            destination: Path,
            **_kwargs: object,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                shutil.copy2(source, destination)
                return
            raise PriorOcrFoundError()

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=copy_then_reject_prior_ocr,
            ) as mock_ocr,
            pytest.raises(OCRError, match="already contains an OCR text layer"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert mock_ocr.call_count == 2
        assert output_path.read_bytes() == output_sentinel
        assert manifest_path.read_bytes() == manifest_sentinel
        assert not list(tmp_dir.glob(".*_ocr_*"))

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

    def test_level_a_manifest_merges_all_isolated_runs_by_physical_page(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        _mock_paddle_orientation: MagicMock,
    ) -> None:
        input_path = tmp_dir / "two-raster-pages.pdf"
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "ocr-manifest.json"
        with Pdf.new() as pdf:
            _add_content_page(pdf)
            _add_content_page(pdf, image_scale=50)
            _add_content_page(pdf, visible_text=True)
            pdf.save(input_path)

        run_directories: list[Path] = []

        def emit_sidecars_and_forms(
            source: Path,
            destination: Path,
            **kwargs: object,
        ) -> None:
            shutil.copy2(source, destination)
            run_directory = Path(kwargs["paddle_manifest_dir"])
            run_number = len(run_directories)
            run_directories.append(run_directory)
            page_numbers = [int(value) - 1 for value in str(kwargs["pages"]).split(",")]
            for page_index in page_numbers:
                text = {
                    0: "Förderung",
                    1: "Kärchow",
                    2: "Straße",
                }[page_index]
                sidecar = run_directory / f"page-{page_index:06d}.json"
                sidecar.write_text(
                    json.dumps(
                        _ocr_page_manifest(page_index, text),
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            with Pdf.open(destination, allow_overwriting_input=True) as pdf:
                for page_index in page_numbers:
                    form = pdf.make_stream(b"/Span <</MCID 0>> BDC BT ET EMC")
                    form[Name.Type] = Name.XObject
                    form[Name.Subtype] = Name.Form
                    form[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
                    form[Name.Resources] = Dictionary()
                    name = Name(f"/OCR-run-{run_number}-page-{page_index}")
                    pdf.pages[page_index].Resources.XObject[name] = form
                pdf.save(destination)

        with patch(
            "pdftopdfa.ocr.ocrmypdf.ocr",
            side_effect=emit_sidecars_and_forms,
        ) as mock_ocr:
            result = apply_ocr(
                input_path,
                output_path,
                ["de", "en"],
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                deskew=True,
                rotate_pages=True,
                _manifest_output_path=manifest_path,
            )

        assert result == output_path
        assert mock_ocr.call_count == 3
        assert len(set(run_directories)) == 3
        assert run_directories[0].parent == run_directories[1].parent
        assert all(not directory.exists() for directory in run_directories)
        assert all(
            call.kwargs["paddle_manifest_dir"] == run_directory
            for call, run_directory in zip(
                mock_ocr.call_args_list,
                run_directories,
                strict=True,
            )
        )
        _mock_paddle_orientation.assert_called_once()
        regular_call, deskew_call, redo_call = mock_ocr.call_args_list
        assert regular_call.kwargs["pages"] == "2"
        assert deskew_call.kwargs["pages"] == "1"
        assert deskew_call.kwargs["deskew"] is True
        assert redo_call.kwargs["pages"] == "3"
        assert redo_call.kwargs["redo_ocr"] is True
        assert "skip_text" not in redo_call.kwargs

        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        assert b"\xef\xbf\xbd" not in manifest_bytes
        assert manifest["schema_version"] == 1
        assert manifest["type"] == "pdftopdfa-ocr-document"
        assert manifest["page_count"] == 3
        assert manifest["languages"] == ["de", "en"]
        assert [page["page_index"] for page in manifest["pages"]] == [0, 1, 2]
        assert [page["lines"][0]["text"] for page in manifest["pages"]] == [
            "Förderung",
            "Kärchow",
            "Straße",
        ]
        assert [page["form_name"] for page in manifest["pages"]] == [
            "/OCR-run-1-page-0",
            "/OCR-run-0-page-1",
            "/OCR-run-2-page-2",
        ]
        with Pdf.open(output_path) as pdf:
            assert b"(Native text)" in pdf.pages[2].Contents.read_bytes()
            assert "/OCR-run-2-page-2" in pdf.pages[2].Resources.XObject

    def test_level_a_manifest_rejects_malformed_run_sidecar(
        self,
        sample_pdf: Path,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        output_path = tmp_dir / "output.pdf"
        manifest_path = tmp_dir / "ocr-manifest.json"
        run_directories: list[Path] = []

        def emit_malformed_sidecar(
            source: Path,
            destination: Path,
            **kwargs: object,
        ) -> None:
            shutil.copy2(source, destination)
            run_directory = Path(kwargs["paddle_manifest_dir"])
            run_directories.append(run_directory)
            (run_directory / "page-000000.json").write_text("{}", encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=emit_malformed_sidecar,
            ),
            pytest.raises(OCRError, match="Invalid OCR manifest"),
        ):
            apply_ocr(
                sample_pdf,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                _manifest_output_path=manifest_path,
            )

        assert not manifest_path.exists()
        assert len(run_directories) == 1
        assert not run_directories[0].exists()

    @pytest.mark.parametrize(
        "mutation",
        [
            "bbox_outside",
            "polygon_mismatch",
            "layout_outside",
            "word_outside_line",
        ],
    )
    def test_level_a_sidecar_rejects_inconsistent_geometry(
        self,
        tmp_dir: Path,
        mutation: str,
    ) -> None:
        run_directory = tmp_dir / mutation
        run_directory.mkdir()
        manifest = _ocr_page_manifest(0, "geometry")
        line = manifest["lines"][0]
        if mutation == "bbox_outside":
            line["bbox"]["left"] = -1.0
            line["polygon"][0][0] = -1.0
            line["polygon"][3][0] = -1.0
        elif mutation == "polygon_mismatch":
            line["polygon"][1][0] = 80.0
            line["polygon"][2][0] = 80.0
        elif mutation == "layout_outside":
            manifest["layout"]["selected_columns"][0]["right"] = 101.0
        else:
            word = line["words"][0]
            word["bbox"] = deepcopy(word["bbox"])
            word["polygon"] = deepcopy(word["polygon"])
            word["bbox"]["left"] = 5.0
            word["polygon"][0][0] = 5.0
            word["polygon"][3][0] = 5.0
        (run_directory / "page-000000.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with pytest.raises(OCRError, match="Invalid OCR manifest"):
            _read_ocr_run_sidecars(run_directory)

    def test_document_manifest_validation_is_exact(self) -> None:
        page = _ocr_page_manifest(0, "strict")
        page["form_name"] = "/OCR-0"
        document = {
            "schema_version": 1,
            "type": "pdftopdfa-ocr-document",
            "page_count": 1,
            "languages": ["de", "en"],
            "pages": [page],
        }

        assert _validate_ocr_document_manifest(document, "test") is document

        malformed = deepcopy(document)
        malformed["unexpected"] = True
        with pytest.raises(OCRError, match="unexpected keys"):
            _validate_ocr_document_manifest(malformed, "test")

    def test_document_manifest_is_revalidated_after_atomic_write(
        self,
        tmp_dir: Path,
    ) -> None:
        pdf_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "manifest.json"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(100, 100))
            _add_ocr_form(pdf, 0, "/OCR-0", [0])
            pdf.save(pdf_path)
        pages = {0: _ocr_page_manifest(0, "strict")}

        def write_corrupt_manifest(path: Path, _value: object) -> None:
            path.write_text('{"schema_version":1}', encoding="utf-8")

        with (
            patch(
                "pdftopdfa.ocr._write_json_atomic",
                side_effect=write_corrupt_manifest,
            ),
            pytest.raises(OCRError, match="Invalid OCR manifest"),
        ):
            _write_ocr_document_manifest(
                output_path,
                pdf_path,
                ["de", "en"],
                pages,
                {0: ("/OCR-0",)},
            )

    @staticmethod
    def _multi_line_page(page_index: int, count: int) -> dict[str, object]:
        page = _ocr_page_manifest(page_index, "line")
        line = page["lines"][0]
        page["lines"] = []
        for mcid in range(count):
            clone = deepcopy(line)
            clone["mcid"] = mcid
            clone["text"] = f"line {mcid}"
            clone["words"][0]["text"] = f"line{mcid}"
            page["lines"].append(clone)
        return page

    def test_document_manifest_drops_suppressed_lines(self, tmp_dir: Path) -> None:
        """Lines the renderer discarded are removed instead of aborting."""
        pdf_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "manifest.json"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(100, 100))
            _add_ocr_form(pdf, 0, "/OCR-0", [0, 2])
            pdf.save(pdf_path)
        pages = {0: self._multi_line_page(0, 3)}

        _write_ocr_document_manifest(
            output_path,
            pdf_path,
            ["de", "en"],
            pages,
            {0: ("/OCR-0",)},
        )

        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        assert [line["mcid"] for line in manifest["pages"][0]["lines"]] == [0, 2]
        assert [line["text"] for line in manifest["pages"][0]["lines"]] == [
            "line 0",
            "line 2",
        ]
        # The sidecar is not modified by reconciliation.
        assert len(pages[0]["lines"]) == 3

    def test_document_manifest_keeps_page_without_emitted_lines(
        self,
        tmp_dir: Path,
    ) -> None:
        """A page whose lines were all suppressed stays in the manifest."""
        pdf_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "manifest.json"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(100, 100))
            _add_ocr_form(pdf, 0, "/OCR-0", [])
            pdf.save(pdf_path)

        _write_ocr_document_manifest(
            output_path,
            pdf_path,
            ["de", "en"],
            {0: self._multi_line_page(0, 2)},
            {0: ("/OCR-0",)},
        )

        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        assert manifest["pages"][0]["lines"] == []

    def test_document_manifest_rejects_undeclared_mcids(self, tmp_dir: Path) -> None:
        """Marked content without a manifest line is a defect, not a drop."""
        pdf_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "manifest.json"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(100, 100))
            _add_ocr_form(pdf, 0, "/OCR-0", [0, 5])
            pdf.save(pdf_path)

        with pytest.raises(OCRError, match=r"undeclared MCIDs: \[5\]"):
            _write_ocr_document_manifest(
                output_path,
                pdf_path,
                ["de", "en"],
                {0: _ocr_page_manifest(0, "strict")},
                {0: ("/OCR-0",)},
            )

        assert not output_path.exists()

    def test_run_sidecar_still_requires_contiguous_mcids(self, tmp_dir: Path) -> None:
        """Only reconciled document manifests may contain MCID gaps."""
        run_directory = tmp_dir / "run"
        run_directory.mkdir()
        page = self._multi_line_page(0, 2)
        page["lines"][1]["mcid"] = 2
        (run_directory / "page-000000.json").write_text(
            json.dumps(page),
            encoding="utf-8",
        )

        with pytest.raises(OCRError, match=r"lines\[1\].mcid: expected 1"):
            _read_ocr_run_sidecars(run_directory)

    def test_document_manifest_rejects_unordered_mcids(self) -> None:
        """MCIDs must stay in reading order even with gaps."""
        page = self._multi_line_page(0, 2)
        page["form_name"] = "/OCR-0"
        page["lines"][0]["mcid"] = 3

        with pytest.raises(OCRError, match="strictly increasing MCID"):
            _validate_ocr_page_manifest(page, "test", document_page=True)

    def test_deskew_preserves_tagging_on_mixed_redo_page(
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

    def test_deskew_text_preparation_handles_1200_nested_forms(
        self,
        tmp_dir: Path,
    ) -> None:
        input_path = tmp_dir / "deep-form-text.pdf"
        output_path = tmp_dir / "prepared.pdf"
        with Pdf.new() as pdf:
            selected_page = pdf.add_blank_page(page_size=(100, 100))
            shared_page = pdf.add_blank_page(page_size=(100, 100))
            leaf = pdf.make_stream(b"BT 3 Tr (deep OCR) Tj ET\n0 0 m")
            leaf[Name.Type] = Name.XObject
            leaf[Name.Subtype] = Name.Form
            leaf[Name.BBox] = pikepdf.Array([0, 0, 100, 100])
            leaf[Name.Resources] = Dictionary()
            root = _nested_form_chain(pdf, leaf)
            for page in (selected_page, shared_page):
                page.obj[Name.Resources] = Dictionary(XObject=Dictionary(DeepForm=root))
                page.obj[Name.Contents] = pdf.make_stream(b"/DeepForm Do")
            pdf.save(input_path)

        _prepare_deskew_input(input_path, output_path, (1,))

        with Pdf.open(output_path) as pdf:
            assert _page_has_text(pdf.pages[0]) is False
            assert _page_has_text(pdf.pages[1]) is True

            current = pdf.pages[0].Resources.XObject.DeepForm
            count = 1
            while (xobjects := current.Resources.get("/XObject")) is not None:
                current = xobjects.Fm
                count += 1
            assert count == 1200
            assert b"Tj" not in current.read_bytes()
            assert b"0 0 m" in current.read_bytes()

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

    def test_normal_a4_document_uses_600_dpi(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "a4.pdf"
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(595, 842))
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
            )

        assert mock_ocr.call_args.kwargs["oversample"] == 600

    def test_large_72_dpi_scan_uses_300_dpi_fallback(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_path = tmp_dir / "large-scan.pdf"
        with Pdf.new() as pdf:
            _add_scan_page(pdf)
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
            )

        kwargs = mock_ocr.call_args.kwargs
        assert kwargs["oversample"] == 300
        assert kwargs["max_image_mpixels"] == _OCR_MAX_PAGE_RASTER_PIXELS / 1_000_000
        assert "OCR page 1" in caplog.text
        assert "preferred 600 dpi" in caplog.text
        assert "using 300 dpi" in caplog.text
        assert "69,012,328 pixels expected" in caplog.text

    def test_mixed_text_and_large_scan_document_remains_processable(
        self,
        tmp_dir: Path,
        model_dirs: tuple[Path, Path],
        validate_models: MagicMock,
    ) -> None:
        input_path = tmp_dir / "mixed-large-scan.pdf"
        with Pdf.new() as pdf:
            _add_content_page(pdf, image_scale=50, visible_text=True)
            _add_scan_page(pdf)
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
            )

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args.kwargs["pages"] == "2"
        assert mock_ocr.call_args.kwargs["oversample"] == 300

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
        assert args[0] == sample_pdf
        assert args[1].parent.parent == output.parent
        assert args[1].parent.name.startswith(".output_ocr_")
        assert args[1] != output
        assert kwargs == {
            "language": ["de", "en"],
            "ocr_engine": "paddle",
            "pdf_renderer": "fpdf2",
            "rasterizer": "pypdfium",
            "output_type": "pdf",
            "oversample": 600,
            "max_image_mpixels": _OCR_MAX_PAGE_RASTER_PIXELS / 1_000_000,
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
            "paddle_layout": False,
            "pages": "1",
        }

    def test_passes_layout_configuration(
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
            apply_ocr(
                sample_pdf,
                output,
                detection_model_dir=detection,
                recognition_model_dir=recognition,
                layout=True,
            )

        validate_models.assert_called_once_with(detection, recognition)
        assert mock_ocr.call_args.kwargs["paddle_layout"] is True

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
            _add_content_page(pdf, image_scale=50, visible_text=True)
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

    def test_prior_ocr_aborts_force_without_publishing_foreign_form(
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

        with (
            patch(
                "pdftopdfa.ocr.ocrmypdf.ocr",
                side_effect=PriorOcrFoundError(),
            ),
            pytest.raises(OCRError, match="already contains an OCR text layer"),
        ):
            apply_ocr(
                input_path,
                output_path,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                force=True,
            )

        assert not output_path.exists()

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
            _add_content_page(pdf, image_scale=50, visible_text=True)
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
