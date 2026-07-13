# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Rebuild the bundled PP-LCNet document-orientation ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
MODEL_REVISION = "d3b95a6dff5fe8a94f2748e12b61cb26818a0df8"
MODEL_REPOSITORY = f"https://huggingface.co/PaddlePaddle/{MODEL_NAME}"
BUNDLED_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pdftopdfa"
    / "resources"
    / "models"
    / MODEL_NAME
)
MODEL_METADATA_FILES = ("LICENSE", "SOURCE.md")
SOURCE_HASHES = {
    "inference.json": (
        "3580978602f309c3554508dd85d4fe09b73a7e0d80d7f9e63258f5b72c390c69"
    ),
    "inference.pdiparams": (
        "e8d6e7c5d264507e40e58a655779059d616b20d7441ea22047d829eb3931989c"
    ),
    "inference.yml": "9e195eb729a8173588cd0e8a852c8b373aa606e79e77b4ac7d8346f5426caf26",
}
CONVERSION = {
    "opset": 7,
    "paddle2onnx": "2.0.2rc3",
    "paddlepaddle": "3.0.0.dev20250426",
    "paddlex": "3.7.2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_sources(target_dir: Path) -> None:
    for filename, expected_hash in SOURCE_HASHES.items():
        url = f"{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{filename}"
        target = target_dir / filename
        urllib.request.urlretrieve(url, target)
        actual_hash = _sha256(target)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {filename}: {actual_hash} != {expected_hash}"
            )


def _write_manifest(model_dir: Path) -> None:
    model_files = {
        filename: _sha256(model_dir / filename)
        for filename in ("inference.onnx", "inference.yml")
    }
    manifest = {
        "engine": "onnxruntime",
        "files": model_files,
        "model_name": MODEL_NAME,
        "source": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "conversion": CONVERSION,
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_model_metadata(model_dir: Path) -> None:
    """Copy the bundled model license and source notice to an output directory."""
    for filename in MODEL_METADATA_FILES:
        source = BUNDLED_MODEL_DIR / filename
        target = model_dir / filename
        if source.resolve() == target.resolve():
            continue
        if not source.is_file():
            raise RuntimeError(f"Bundled model metadata is missing: {source}")
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paddlex-command",
        default="paddlex",
        help="PaddleX executable with the Paddle2ONNX plugin installed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLED_MODEL_DIR,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pdftopdfa_paddle2onnx_") as temp:
        temp_dir = Path(temp)
        source_dir = temp_dir / "paddle"
        converted_dir = temp_dir / "onnx"
        source_dir.mkdir()
        converted_dir.mkdir()
        _download_sources(source_dir)

        subprocess.run(
            [
                args.paddlex_command,
                "--paddle2onnx",
                "--paddle_model_dir",
                str(source_dir),
                "--onnx_model_dir",
                str(converted_dir),
                "--opset_version",
                str(CONVERSION["opset"]),
            ],
            check=True,
        )

        for filename in ("inference.onnx", "inference.yml"):
            source = converted_dir / filename
            if not source.is_file():
                raise RuntimeError(f"PaddleX did not produce {filename}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("inference.onnx", "inference.yml"):
            shutil.copy2(converted_dir / filename, args.output_dir / filename)
        _copy_model_metadata(args.output_dir)
        _write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
