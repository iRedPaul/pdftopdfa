# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Verify bundled orientation artifacts in built distributions."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

MODEL_ROOT = "pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori"
REQUIRED_FILES = {
    "LICENSE",
    "SOURCE.md",
    "inference.onnx",
    "inference.yml",
    "manifest.json",
}


def _verify_files(names: list[str], read_file) -> None:
    roots = {
        name[: -len("/manifest.json")]
        for name in names
        if name.endswith(f"{MODEL_ROOT}/manifest.json")
    }
    if len(roots) != 1:
        raise RuntimeError(f"Expected one bundled model root, found {sorted(roots)}")
    root = roots.pop()

    missing = [name for name in REQUIRED_FILES if f"{root}/{name}" not in names]
    if missing:
        raise RuntimeError(f"Missing bundled model files: {sorted(missing)}")

    manifest = json.loads(read_file(f"{root}/manifest.json").decode("utf-8"))
    for filename, expected_hash in manifest["files"].items():
        actual_hash = hashlib.sha256(read_file(f"{root}/{filename}")).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Hash mismatch for bundled {filename}")


def _verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        _verify_files(archive.namelist(), archive.read)


def _verify_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()

        def read_file(name: str) -> bytes:
            member = archive.extractfile(name)
            if member is None:
                raise RuntimeError(f"Could not read {name}")
            return io.BytesIO(member.read()).getvalue()

        _verify_files(names, read_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if not wheels or not sdists:
        raise RuntimeError("Both a wheel and source distribution are required")
    for wheel in wheels:
        _verify_wheel(wheel)
    for sdist in sdists:
        _verify_sdist(sdist)


if __name__ == "__main__":
    main()
