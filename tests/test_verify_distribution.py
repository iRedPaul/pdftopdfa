# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for built distribution verification."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_verify_script():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_distribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pdftopdfa_verify_distribution", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VERIFY_SCRIPT = _load_verify_script()


def _valid_archive_files() -> dict[str, bytes]:
    root = f"archive/{_VERIFY_SCRIPT.MODEL_ROOT}"
    model_files = {
        "LICENSE": b"license",
        "SOURCE.md": b"source",
        "inference.onnx": b"orientation model",
        "inference.yml": b"orientation config",
    }
    files = {f"{root}/{name}": content for name, content in model_files.items()}
    manifest = {
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in model_files.items()
        }
    }
    files[f"{root}/manifest.json"] = json.dumps(manifest).encode()
    return files


def test_external_model_hash_is_detected_regardless_of_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _valid_archive_files()
    content = b"external model content"
    files["archive/pdftopdfa/resources/models/model.bin"] = content
    monkeypatch.setattr(
        _VERIFY_SCRIPT,
        "EXTERNAL_OCR_MODEL_HASHES",
        {hashlib.sha256(content).hexdigest()},
    )

    with pytest.raises(RuntimeError, match="External PP-OCRv6 model artifact"):
        _VERIFY_SCRIPT._verify_files(list(files), files.__getitem__)


@pytest.mark.parametrize(
    "source",
    [
        b"from ocrmypdf._exec.tesseract import ThresholdingMethod\n",
        b'path = os.environ.get("TESSERACT_PATH")\n',
        b'options = {"tesseract_timeout": 300}\n',
        b'ocr_engine = "tesseract"\n',
    ],
)
def test_active_tesseract_configuration_is_rejected(source: bytes) -> None:
    files = _valid_archive_files()
    files["archive/src/pdftopdfa/ocr.py"] = source

    with pytest.raises(RuntimeError, match="Active Tesseract configuration"):
        _VERIFY_SCRIPT._verify_files(list(files), files.__getitem__)


def test_tesseract_blocker_compatibility_namespace_is_allowed() -> None:
    files = _valid_archive_files()
    files["archive/src/pdftopdfa/ocr_paddle.py"] = b"""
_TESSERACT_PLUGIN = "ocrmypdf.builtin_plugins.tesseract_ocr"

class _TesseractCompatibilityOptions:
    pass

options = {"tesseract": _TesseractCompatibilityOptions}
plugin_manager.set_blocked(_TESSERACT_PLUGIN)
"""

    _VERIFY_SCRIPT._verify_files(list(files), files.__getitem__)
