# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for built distribution verification."""

import hashlib
import importlib.util
import json
import tomllib
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


def test_cpu_and_directml_runtimes_are_separate_extras() -> None:
    """The overlapping ONNX Runtime wheels must never be co-installed."""
    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]

    base_dependencies = project["dependencies"]
    cpu_dependencies = project["optional-dependencies"]["ocr"]
    directml_dependencies = project["optional-dependencies"]["directml"]

    assert not any(
        dependency.startswith("onnxruntime") for dependency in base_dependencies
    )
    assert "onnxruntime==1.27.0" in cpu_dependencies
    assert not any(
        dependency.startswith("onnxruntime-directml") for dependency in cpu_dependencies
    )
    assert (
        "onnxruntime-directml==1.24.4; sys_platform == 'win32'" in directml_dependencies
    )
    assert not any(
        dependency.startswith("onnxruntime==") for dependency in directml_dependencies
    )


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


@pytest.mark.parametrize(
    "model_name",
    ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"],
)
def test_external_model_name_marker_is_detected(model_name: str) -> None:
    files = _valid_archive_files()
    files[f"archive/pdftopdfa/resources/models/{model_name}/inference.onnx"] = (
        b"external model content"
    )

    with pytest.raises(RuntimeError, match="External PP-OCRv6 model files"):
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
