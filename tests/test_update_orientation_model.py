# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the bundled orientation model rebuild script."""

import importlib.util
from pathlib import Path


def _load_update_script():
    """Load the maintenance script without relying on the repository import path."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "update_orientation_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pdftopdfa_update_orientation_model", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UPDATE_SCRIPT = _load_update_script()
BUNDLED_MODEL_DIR = _UPDATE_SCRIPT.BUNDLED_MODEL_DIR
MODEL_METADATA_FILES = _UPDATE_SCRIPT.MODEL_METADATA_FILES
_copy_model_metadata = _UPDATE_SCRIPT._copy_model_metadata


def test_copy_model_metadata_populates_custom_output(tmp_path: Path) -> None:
    """A custom rebuild output includes its license and source notice."""
    output_dir = tmp_path / "model"
    output_dir.mkdir()

    _copy_model_metadata(output_dir)

    for filename in MODEL_METADATA_FILES:
        assert (output_dir / filename).read_bytes() == (
            BUNDLED_MODEL_DIR / filename
        ).read_bytes()
