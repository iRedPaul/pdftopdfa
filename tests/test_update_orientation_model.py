# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the bundled orientation model rebuild script."""

from pathlib import Path

from scripts.update_orientation_model import (
    BUNDLED_MODEL_DIR,
    MODEL_METADATA_FILES,
    _copy_model_metadata,
)


def test_copy_model_metadata_populates_custom_output(tmp_path: Path) -> None:
    """A custom rebuild output includes its license and source notice."""
    output_dir = tmp_path / "model"
    output_dir.mkdir()

    _copy_model_metadata(output_dir)

    for filename in MODEL_METADATA_FILES:
        assert (output_dir / filename).read_bytes() == (
            BUNDLED_MODEL_DIR / filename
        ).read_bytes()
