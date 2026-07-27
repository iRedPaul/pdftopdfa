# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Verify model contents and OCR configuration in built distributions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

MODEL_ROOT = "pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori"
REQUIRED_FILES = {
    "LICENSE",
    "SOURCE.md",
    "inference.onnx",
    "inference.yml",
    "manifest.json",
}
EXTERNAL_OCR_MODEL_MARKERS = {
    "pp-ocrv6_medium_det",
    "pp-ocrv6_medium_rec",
}
ALLOWED_TESSERACT_IDENTIFIERS = {
    "_TESSERACT_PLUGIN",
    "_TesseractCompatibilityOptions",
}


def _is_package_python_source(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix == ".py" and "pdftopdfa" in path.parts


def _verify_no_active_tesseract(name: str, content: bytes) -> None:
    try:
        tree = ast.parse(content, filename=name)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect package Python source: {name}") from exc

    allowed_literals = {
        id(key)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if (
            isinstance(key, ast.Constant)
            and key.value == "tesseract"
            and isinstance(value, ast.Name)
            and value.id == "_TesseractCompatibilityOptions"
        )
    }

    for node in ast.walk(tree):
        identifiers: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.append(node.name)
        elif isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers.append(node.arg)
        elif isinstance(node, ast.Import):
            identifiers.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                identifiers.append(node.module)
            identifiers.extend(alias.name for alias in node.names)

        forbidden_identifier = next(
            (
                identifier
                for identifier in identifiers
                if "tesseract" in identifier.casefold()
                and identifier not in ALLOWED_TESSERACT_IDENTIFIERS
            ),
            None,
        )
        if forbidden_identifier is not None:
            raise RuntimeError(
                "Active Tesseract configuration must not be distributed: "
                f"{name}:{node.lineno} ({forbidden_identifier})"
            )

        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.casefold()
        if (
            value.startswith("tesseract_") or value in {"tesseract", "tesseract.exe"}
        ) and id(node) not in allowed_literals:
            raise RuntimeError(
                "Active Tesseract configuration must not be distributed: "
                f"{name}:{node.lineno} ({node.value})"
            )


def _verify_files(names: list[str], read_file) -> None:
    external_model_entries = [
        name
        for name in names
        if any(marker in name.lower() for marker in EXTERNAL_OCR_MODEL_MARKERS)
    ]
    if external_model_entries:
        raise RuntimeError(
            "External PP-OCRv6 model files must not be distributed: "
            f"{sorted(external_model_entries)}"
        )

    for name in names:
        if _is_package_python_source(name):
            _verify_no_active_tesseract(name, read_file(name))

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
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        _verify_files(names, archive.read)


def _verify_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]

        def read_file(name: str) -> bytes:
            member = archive.extractfile(name)
            if member is None:
                raise RuntimeError(f"Could not read {name}")
            return member.read()

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
