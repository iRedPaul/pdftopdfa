# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ONNX Runtime provider selection for PaddleOCR models."""

from __future__ import annotations

import stat
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


def validate_ocr_execution_provider(value: str) -> str:
    """Validate and return an OCR execution-provider name."""
    if value not in OCR_EXECUTION_PROVIDERS:
        choices = ", ".join(OCR_EXECUTION_PROVIDERS)
        raise ValueError(
            f"Unsupported OCR execution provider {value!r}; expected one of: {choices}"
        )
    return value


def onnxruntime_engine_config(value: str) -> dict[str, object]:
    """Return the strict PaddleX ONNX Runtime config for an OCR provider."""
    value = validate_ocr_execution_provider(value)
    provider = _ONNXRUNTIME_PROVIDERS[value]
    if value == "cpu":
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

    return {
        "providers": [provider],
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
    }


def require_execution_provider(session: Any, value: str) -> None:
    """Reject a session that silently selected a different execution provider."""
    value = validate_ocr_execution_provider(value)
    expected = _ONNXRUNTIME_PROVIDERS[value]
    try:
        providers = tuple(session.get_providers())
    except Exception as exc:
        raise OCRError(
            f"Could not verify the ONNX Runtime {value} execution provider: {exc}"
        ) from exc

    if not providers or providers[0] != expected:
        actual = providers[0] if providers else "no execution provider"
        raise OCRError(
            f"{_EXECUTION_PROVIDER_LABELS[value]} execution was requested, but "
            f"ONNX Runtime initialized {actual} instead of {expected}; refusing "
            "CPU fallback"
        )

    if value == "directml":
        try:
            session.disable_fallback()
        except Exception as exc:
            raise OCRError(
                "Could not disable ONNX Runtime fallback for DirectML execution"
            ) from exc
