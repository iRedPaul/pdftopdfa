# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ONNX Runtime provider selection for PaddleOCR models."""

from __future__ import annotations

import ctypes
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import OCRError

OCR_EXECUTION_PROVIDERS = ("cpu", "directml")
_MODEL_FILENAMES = frozenset({"inference.onnx", "inference.yml"})

_ONNXRUNTIME_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "directml": "DmlExecutionProvider",
}
_EXECUTION_PROVIDER_LABELS = {
    "cpu": "CPU",
    "directml": "DirectML",
}

# Upper bound for "directml:<index>"; far beyond any real adapter count, but
# small enough to keep a typo from reaching ONNX Runtime as a huge device id.
_MAX_OCR_DEVICE_ID = 63


@dataclass(frozen=True)
class _ModelSpec:
    name: str


def _validate_model_directory(
    model_dir: Path,
    spec: _ModelSpec,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise OCRError(f"{spec.name} model directory does not exist: {model_dir}")

    try:
        entries = {entry.name for entry in model_dir.iterdir()}
    except OSError as exc:
        raise OCRError(f"Could not inspect {spec.name} model directory: {exc}") from exc

    if entries != _MODEL_FILENAMES:
        missing = sorted(_MODEL_FILENAMES - entries)
        unexpected = sorted(entries - _MODEL_FILENAMES)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise OCRError(
            f"{spec.name} model directory must contain exactly inference.onnx "
            f"and inference.yml ({'; '.join(details)})"
        )

    fingerprint = []
    for filename in sorted(_MODEL_FILENAMES):
        path = model_dir / filename
        try:
            artifact_stat = path.lstat()
        except OSError as exc:
            raise OCRError(
                f"Could not inspect {spec.name} model artifact: {exc}"
            ) from exc
        if path.is_symlink() or not stat.S_ISREG(artifact_stat.st_mode):
            raise OCRError(f"{spec.name} model artifact is not a regular file: {path}")
        fingerprint.append(
            (
                artifact_stat.st_dev,
                artifact_stat.st_ino,
                artifact_stat.st_mode,
                artifact_stat.st_size,
                artifact_stat.st_mtime_ns,
                artifact_stat.st_ctime_ns,
            )
        )
    return tuple(fingerprint)


def _parse_execution_provider(value: str) -> tuple[str, int | None]:
    """Split a provider name into its base name and optional DirectML device index."""
    if not isinstance(value, str):
        raise ValueError(
            f"Unsupported OCR execution provider {value!r}; expected a string"
        )

    base, separator, suffix = value.partition(":")
    if base not in OCR_EXECUTION_PROVIDERS:
        choices = ", ".join(OCR_EXECUTION_PROVIDERS)
        raise ValueError(
            f"Unsupported OCR execution provider {value!r}; expected one of: {choices}"
        )
    if not separator:
        return base, None
    if base != "directml":
        raise ValueError(
            f"OCR execution provider {value!r} does not support a device index"
        )
    # Rejects "", "-1", "+1" and non-ASCII digits, which int() would accept.
    if not suffix.isascii() or not suffix.isdecimal():
        raise ValueError(f"Invalid DirectML device index in {value!r}")

    device_id = int(suffix)
    if device_id > _MAX_OCR_DEVICE_ID:
        raise ValueError(f"DirectML device index {device_id} is out of range")
    return base, device_id


def execution_provider_base(value: str) -> str:
    """Return the provider name without its optional device index."""
    return _parse_execution_provider(value)[0]


def validate_ocr_execution_provider(value: str) -> str:
    """Validate an OCR execution-provider name and return its canonical spelling."""
    base, device_id = _parse_execution_provider(value)
    if device_id is None:
        return base
    # Canonical so that "directml:01" and "directml:1" share one model cache entry.
    return f"{base}:{device_id}"


def onnxruntime_engine_config(value: str) -> dict[str, object]:
    """Return the strict PaddleX ONNX Runtime config for an OCR provider."""
    base, device_id = _parse_execution_provider(value)
    provider = _ONNXRUNTIME_PROVIDERS[base]
    if base == "cpu":
        return {"providers": [provider]}

    try:
        import onnxruntime

        available_providers = tuple(onnxruntime.get_available_providers())
    except Exception as exc:
        raise OCRError(
            "DirectML execution was requested, but the optional ONNX Runtime "
            f"could not be loaded: {exc}. Install it on Windows 11 with: "
            "pip install pdftopdfa[directml]"
        ) from exc

    if provider not in available_providers:
        raise OCRError(
            "DirectML execution was requested, but ONNX Runtime does not expose "
            "DmlExecutionProvider. Install the optional DirectML runtime on "
            "Windows 11 with: pip install pdftopdfa[directml]. Available "
            f"providers: {list(available_providers)!r}"
        )

    config: dict[str, object] = {
        "providers": [provider],
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
    }
    if device_id is not None:
        # PaddleX forbids extra engine_config keys and types "providers" as a
        # plain list of strings, so the adapter is selected through the separate
        # "provider_options" field it forwards to InferenceSession.
        config["provider_options"] = [{"device_id": device_id}]
    return config


def require_execution_provider(session: Any, value: str) -> None:
    """Reject a session that silently selected a different execution provider."""
    value = validate_ocr_execution_provider(value)
    base = execution_provider_base(value)
    expected = _ONNXRUNTIME_PROVIDERS[base]
    try:
        providers = tuple(session.get_providers())
    except Exception as exc:
        raise OCRError(
            f"Could not verify the ONNX Runtime {value} execution provider: {exc}"
        ) from exc

    if not providers or providers[0] != expected:
        actual = providers[0] if providers else "no execution provider"
        raise OCRError(
            f"{_EXECUTION_PROVIDER_LABELS[base]} execution was requested, but "
            f"ONNX Runtime initialized {actual} instead of {expected}; refusing "
            "CPU fallback"
        )

    if base == "directml":
        try:
            session.disable_fallback()
        except Exception as exc:
            raise OCRError(
                "Could not disable ONNX Runtime fallback for DirectML execution"
            ) from exc


# DXGI adapter enumeration, used to map a "directml:<index>" device id back to a
# physical adapter. ctypes keeps this dependency-free; a full run costs ~25 ms.
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
_DXGI_ERROR_NOT_FOUND = -2005270526  # 0x887A0002 as a signed 32-bit HRESULT
_MICROSOFT_VENDOR_ID = 0x1414

_IUNKNOWN_RELEASE = 2
_IDXGI_ADAPTER1_GET_DESC1 = 10
_IDXGI_FACTORY1_ENUM_ADAPTERS1 = 12


class _LUID(ctypes.Structure):
    _fields_ = (("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32))


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    )


class _DXGIAdapterDesc1(ctypes.Structure):
    _fields_ = (
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    )


_IID_IDXGI_FACTORY1 = _GUID(
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


@dataclass(frozen=True)
class DirectMLDevice:
    """A DirectML-capable adapter and the device index that selects it."""

    device_id: int
    description: str
    vendor_id: int
    dedicated_video_memory: int

    @property
    def execution_provider(self) -> str:
        """The ``ocr_execution_provider`` value that selects this adapter."""
        return f"directml:{self.device_id}"


def _com_method(
    pointer: ctypes.c_void_p,
    index: int,
    restype: Any,
    *argtypes: Any,
) -> Any:
    """Bind the vtable entry at ``index`` of a COM interface pointer."""
    vtable = ctypes.cast(
        pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _com_release(pointer: ctypes.c_void_p) -> None:
    if pointer:
        _com_method(pointer, _IUNKNOWN_RELEASE, ctypes.c_uint32)(pointer)


def _is_directml_adapter(desc: _DXGIAdapterDesc1) -> bool:
    """Report whether DirectML would expose this adapter as a device."""
    if desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
        return False
    # The Basic Render Driver reports the Microsoft vendor id without any
    # dedicated video memory; ONNX Runtime skips it the same way.
    return not (desc.VendorId == _MICROSOFT_VENDOR_ID and not desc.DedicatedVideoMemory)


def _hresult_error(operation: str, hresult: int) -> OCRError:
    return OCRError(f"{operation} failed with HRESULT 0x{hresult & 0xFFFFFFFF:08X}")


def list_directml_devices() -> list[DirectMLDevice]:
    """Return the DirectML adapters in DXGI enumeration order.

    The position in the returned list is the index to use in
    ``"directml:<index>"``.
    """
    if sys.platform != "win32":
        raise OCRError("DirectML devices can only be enumerated on Windows")

    try:
        dxgi = ctypes.WinDLL("dxgi")
        create_factory = dxgi.CreateDXGIFactory1
    except (AttributeError, OSError) as exc:
        raise OCRError(f"Could not load the DXGI library: {exc}") from exc

    create_factory.restype = ctypes.c_int32
    create_factory.argtypes = (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))

    factory = ctypes.c_void_p()
    hresult = create_factory(ctypes.byref(_IID_IDXGI_FACTORY1), ctypes.byref(factory))
    if hresult < 0 or not factory:
        raise _hresult_error("CreateDXGIFactory1", hresult)

    devices: list[DirectMLDevice] = []
    try:
        enum_adapters = _com_method(
            factory,
            _IDXGI_FACTORY1_ENUM_ADAPTERS1,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hresult = enum_adapters(factory, index, ctypes.byref(adapter))
            index += 1
            if hresult == _DXGI_ERROR_NOT_FOUND:
                break
            if hresult < 0 or not adapter:
                raise _hresult_error("IDXGIFactory1::EnumAdapters1", hresult)
            try:
                desc = _DXGIAdapterDesc1()
                get_desc = _com_method(
                    adapter,
                    _IDXGI_ADAPTER1_GET_DESC1,
                    ctypes.c_int32,
                    ctypes.POINTER(_DXGIAdapterDesc1),
                )
                hresult = get_desc(adapter, ctypes.byref(desc))
                if hresult < 0:
                    raise _hresult_error("IDXGIAdapter1::GetDesc1", hresult)
                if _is_directml_adapter(desc):
                    devices.append(
                        DirectMLDevice(
                            device_id=len(devices),
                            description=desc.Description,
                            vendor_id=desc.VendorId,
                            dedicated_video_memory=desc.DedicatedVideoMemory,
                        )
                    )
            finally:
                _com_release(adapter)
    finally:
        _com_release(factory)
    return devices
