# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Optional real-model end-to-end tests for semantic OCR tagging."""

from __future__ import annotations

import hashlib
import os
import zlib
from pathlib import Path

import pikepdf
import pytest
from pdfminer.high_level import extract_text
from pikepdf import Array, Dictionary, Name, NumberTree
from PIL import Image, ImageDraw, ImageFont

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.tagging import ensure_logical_structure
from pdftopdfa.utils import resolve_indirect
from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf


def _model_directories() -> tuple[Path, Path]:
    detection = os.environ.get("PDFTOPDFA_TEST_OCR_DETECTION_MODEL_DIR")
    recognition = os.environ.get("PDFTOPDFA_TEST_OCR_RECOGNITION_MODEL_DIR")
    if bool(detection) != bool(recognition):
        pytest.fail("Both real OCR model directories must be configured together")
    if detection is None or recognition is None:
        pytest.skip("Real OCR model directories are not configured")
    return Path(detection), Path(recognition)


def _model_snapshot(model_directories: tuple[Path, Path]) -> tuple[object, ...]:
    entries = []
    roots = (
        (model_directories[0].parent,)
        if model_directories[0].parent == model_directories[1].parent
        else model_directories
    )
    for model_index, root in enumerate(roots):
        for path in (root, *sorted(root.rglob("*"))):
            stat = path.stat()
            digest = None
            if path.is_file():
                with path.open("rb") as model_file:
                    digest = hashlib.file_digest(model_file, "sha256").hexdigest()
            entries.append(
                (
                    model_index,
                    path.relative_to(root).as_posix(),
                    path.is_dir(),
                    stat.st_size,
                    stat.st_mtime_ns,
                    digest,
                )
            )
    return tuple(entries)


def _roles(pdf: pikepdf.Pdf) -> set[str]:
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    pending = [root.get("/K")]
    roles = set()
    while pending:
        value = resolve_indirect(pending.pop())
        if isinstance(value, Array):
            pending.extend(value)
        elif isinstance(value, Dictionary):
            role = resolve_indirect(value.get("/S"))
            if isinstance(role, Name):
                roles.add(str(role))
            if value.get("/Type") == Name.StructElem:
                pending.append(value.get("/K"))
    return roles


def _mixed_ocr_source(path: Path, *, rotation: int) -> None:
    image = Image.new("RGB", (800, 1000), "white")
    font_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pdftopdfa"
        / "resources"
        / "fonts"
        / "LiberationSans-Regular.ttf"
    )
    font = ImageFont.truetype(font_path, 68)
    drawing = ImageDraw.Draw(image)
    drawing.text((50, 55), "NATIVE HDR", fill="black", font=font)
    drawing.text((120, 500), "SCAN BODY 789", fill="black", font=font)

    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(100, 120))
    raster = pdf.make_stream(zlib.compress(image.tobytes()))
    raster["/Type"] = Name.XObject
    raster["/Subtype"] = Name.Image
    raster["/Width"] = image.width
    raster["/Height"] = image.height
    raster["/ColorSpace"] = Name.DeviceRGB
    raster["/BitsPerComponent"] = 8
    raster["/Filter"] = Name.FlateDecode
    font_resource = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
        )
    )
    page.obj["/Resources"] = Dictionary(
        Font=Dictionary(F1=font_resource),
        XObject=Dictionary(Im0=raster),
    )
    page.obj["/Contents"] = pdf.make_stream(
        b"q 80 0 0 100 10 10 cm /Im0 Do Q\nBT /F1 8 Tf 15 100 Td (NATIVE HDR) Tj ET"
    )
    page.obj["/MediaBox"] = Array([0, 0, 100, 120])
    page.obj["/CropBox"] = Array([10, 10, 90, 110])
    page.obj["/Rotate"] = rotation
    page.obj["/UserUnit"] = 2
    link = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([14, 97, 65, 109]),
            Border=Array([0, 0, 0]),
            A=Dictionary(S=Name.URI, URI=pikepdf.String("https://example.test")),
            P=page.obj,
        )
    )
    page.obj["/Annots"] = Array([link])
    pdf.save(path)


def _content_markers(
    container: pikepdf.Page | pikepdf.Stream,
) -> tuple[tuple[str, str | None, int | None], ...]:
    markers = []
    for instruction in pikepdf.parse_content_stream(container):
        if (
            not isinstance(instruction, pikepdf.ContentStreamInstruction)
            or str(instruction.operator) != "BDC"
            or len(instruction.operands) < 2
        ):
            continue
        tag = str(resolve_indirect(instruction.operands[0]))
        properties = resolve_indirect(instruction.operands[1])
        if not isinstance(properties, Dictionary):
            continue
        artifact_type = resolve_indirect(properties.get("/Type"))
        mcid = resolve_indirect(properties.get("/MCID"))
        markers.append(
            (
                tag,
                str(artifact_type) if isinstance(artifact_type, Name) else None,
                int(mcid) if isinstance(mcid, int) else None,
            )
        )
    return tuple(markers)


def _structure_dictionaries(pdf: pikepdf.Pdf) -> tuple[Dictionary, ...]:
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    pending = [root.get("/K")]
    values = []
    while pending:
        value = resolve_indirect(pending.pop())
        if isinstance(value, Array):
            pending.extend(value)
        elif isinstance(value, Dictionary):
            values.append(value)
            if value.get("/Type") == Name.StructElem:
                pending.append(value.get("/K"))
    return tuple(values)


@pytest.mark.parametrize("level", ["2a", "3a"])
def test_real_ocr_semantics_survive_reopen_and_pass_verapdf(
    tmp_path: Path,
    level: str,
) -> None:
    """Exercise recognition, manifest binding, tagging, persistence, and PDF/A."""
    detection, recognition = _model_directories()
    if not is_verapdf_available():
        pytest.skip("veraPDF is not installed")
    model_snapshot = _model_snapshot((detection, recognition))
    image = Image.new("RGB", (1800, 650), "white")
    font_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pdftopdfa"
        / "resources"
        / "fonts"
        / "LiberationSans-Regular.ttf"
    )
    font = ImageFont.truetype(font_path, 96)
    drawing = ImageDraw.Draw(image)
    drawing.text((100, 90), "SEMANTIC OCR TEST 123", fill="black", font=font)
    drawing.text((100, 280), "Foerderung Groesse Rueckgabe", fill="black", font=font)
    source = tmp_path / "semantic-ocr-source.pdf"
    image.save(source, "PDF", resolution=300.0)
    output = tmp_path / f"semantic-ocr-{level}.pdf"

    try:
        result = convert_to_pdfa(
            source,
            output,
            level=level,
            ocr_languages=["de"],
            ocr_detection_model_dir=detection,
            ocr_recognition_model_dir=recognition,
            ocr_force=True,
            ocr_layout=True,
            validate=True,
        )
    finally:
        assert _model_snapshot((detection, recognition)) == model_snapshot

    assert result.success is True
    assert result.validation_failed is False
    assert validate_with_verapdf(output, flavour=level).compliant is True
    extracted = extract_text(output)
    assert "SEMANTIC OCR TEST" in extracted.upper()
    assert "\ufffd" not in extracted

    roundtrip = tmp_path / f"semantic-ocr-{level}-roundtrip.pdf"
    with pikepdf.Pdf.open(output) as pdf:
        root = resolve_indirect(pdf.Root["/StructTreeRoot"])
        forms = [
            item
            for item in pdf.objects
            if isinstance(item, pikepdf.Stream)
            and resolve_indirect(item.get("/Subtype")) == Name.Form
            and "/StructParents" in item
        ]
        assert len(forms) == 1
        form = forms[0]
        parent_tree = NumberTree(root["/ParentTree"])
        owners = resolve_indirect(parent_tree[int(form["/StructParents"])])
        assert isinstance(owners, Array)
        assert owners
        assert {"/Document", "/Div", "/P"} <= _roles(pdf)
        original_root = root.objgen
        preserved = ensure_logical_structure(pdf, semantic=True)
        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert preserved["semantic_repairs"] == 0
        assert pdf.Root["/StructTreeRoot"].objgen == original_root
        pdf.save(roundtrip)

    assert validate_with_verapdf(roundtrip, flavour=level).compliant is True


@pytest.mark.parametrize(
    ("level", "rotation"),
    [("2a", 0), ("3a", 90)],
)
def test_real_mixed_ocr_keeps_native_and_scan_semantics_once(
    tmp_path: Path,
    level: str,
    rotation: int,
) -> None:
    detection, recognition = _model_directories()
    if not is_verapdf_available():
        pytest.skip("veraPDF is not installed")
    model_snapshot = _model_snapshot((detection, recognition))
    source = tmp_path / f"mixed-ocr-source-{rotation}.pdf"
    output = tmp_path / f"mixed-ocr-{level}-{rotation}.pdf"
    _mixed_ocr_source(source, rotation=rotation)

    try:
        result = convert_to_pdfa(
            source,
            output,
            level=level,
            ocr_languages=["en"],
            ocr_detection_model_dir=detection,
            ocr_recognition_model_dir=recognition,
            ocr_layout=True,
            validate=True,
        )
    finally:
        assert _model_snapshot((detection, recognition)) == model_snapshot

    assert result.success is True
    assert result.validation_failed is False
    assert validate_with_verapdf(output, flavour=level).compliant is True
    roundtrip = tmp_path / f"mixed-ocr-{level}-{rotation}-roundtrip.pdf"
    with pikepdf.Pdf.open(output) as pdf:
        page = pdf.pages[0]
        root = resolve_indirect(pdf.Root["/StructTreeRoot"])
        forms = [
            item
            for item in pdf.objects
            if isinstance(item, pikepdf.Stream)
            and resolve_indirect(item.get("/Subtype")) == Name.Form
            and "/StructParents" in item
        ]
        assert len(forms) == 1
        ocr_form = forms[0]
        page_markers = _content_markers(page)
        form_markers = _content_markers(ocr_form)
        assert [marker[2] for marker in page_markers if marker[2] is not None] == [0]
        assert ("/Artifact", "/Layout", None) in page_markers
        assert [marker[2] for marker in form_markers if marker[2] is not None] == [1]
        assert ("/Artifact", "/Layout", None) in form_markers
        assert "/StructParents" in page.obj
        assert "/StructParents" in ocr_form
        assert int(page.obj["/StructParents"]) != int(ocr_form["/StructParents"])
        parent_tree = NumberTree(root["/ParentTree"])
        page_owners = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
        form_owners = resolve_indirect(parent_tree[int(ocr_form["/StructParents"])])
        assert isinstance(page_owners, Array) and len(page_owners) == 1
        assert isinstance(form_owners, Array) and len(form_owners) == 2
        assert resolve_indirect(form_owners[0]) is None
        assert isinstance(resolve_indirect(form_owners[1]), Dictionary)
        structure = _structure_dictionaries(pdf)
        mcrs = [item for item in structure if item.get("/Type") == Name.MCR]
        assert len(mcrs) == 2
        assert sum("/Stm" not in item for item in mcrs) == 1
        assert (
            sum(
                "/Stm" in item
                and resolve_indirect(item["/Stm"]).objgen == ocr_form.objgen
                and int(item["/MCID"]) == 1
                for item in mcrs
            )
            == 1
        )
        annotation = resolve_indirect(page.obj["/Annots"][0])
        assert annotation.get("/Subtype") == Name.Link
        assert "/StructParent" in annotation
        assert page.obj.get("/Tabs") == Name.S
        assert {"/Document", "/Div", "/P", "/Link"} <= _roles(pdf)
        original_root = root.objgen
        preserved = ensure_logical_structure(pdf, semantic=True)
        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert pdf.Root["/StructTreeRoot"].objgen == original_root
        pdf.save(roundtrip)

    assert validate_with_verapdf(roundtrip, flavour=level).compliant is True


@pytest.mark.parametrize("level", ["2a", "3a"])
@pytest.mark.parametrize(
    ("filename", "ocr_options"),
    [
        (
            "200+210 - C1+2 - Crop Bons.pdf",
            {"ocr_force": True},
        ),
        (
            "Rechnung Schräg.pdf",
            {"ocr_deskew": True, "ocr_rotate_pages": True},
        ),
    ],
    ids=["cropped-receipts-force", "skewed-invoice-deskew"],
)
def test_real_test_document_ocr_is_stable_tagged_and_compliant(
    tmp_path: Path,
    level: str,
    filename: str,
    ocr_options: dict[str, bool],
) -> None:
    """Cover representative real scans with the configured immutable models."""
    detection, recognition = _model_directories()
    if not is_verapdf_available():
        pytest.skip("veraPDF is not installed")
    source = Path(__file__).resolve().parents[1] / "test_docs" / filename
    with source.open("rb") as source_file:
        source_digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    model_snapshot = _model_snapshot((detection, recognition))
    output = tmp_path / f"{source.stem}-{level}.pdf"
    roundtrip = tmp_path / f"{source.stem}-{level}-roundtrip.pdf"

    try:
        result = convert_to_pdfa(
            source,
            output,
            level=level,
            ocr_languages=["de", "en"],
            ocr_detection_model_dir=detection,
            ocr_recognition_model_dir=recognition,
            ocr_layout=True,
            validate=True,
            **ocr_options,
        )

        assert result.success is True
        assert result.validation_failed is False
        assert validate_with_verapdf(output, flavour=level).compliant is True
        extracted = extract_text(output)
        assert sum(character.isalnum() for character in extracted) >= 10
        assert "\ufffd" not in extracted

        with pikepdf.Pdf.open(output) as pdf:
            root = resolve_indirect(pdf.Root["/StructTreeRoot"])
            assert isinstance(resolve_indirect(root.get("/ParentTree")), Dictionary)
            assert {"/Document", "/Div", "/P"} <= _roles(pdf)
            structure = _structure_dictionaries(pdf)
            assert any(item.get("/Type") == Name.MCR for item in structure)
            original_root = root.objgen
            preserved = ensure_logical_structure(pdf, semantic=True)
            assert preserved["structure_preserved"] is True
            assert preserved["structure_rebuilt"] is False
            assert pdf.Root["/StructTreeRoot"].objgen == original_root
            pdf.save(roundtrip)

        assert validate_with_verapdf(roundtrip, flavour=level).compliant is True
    finally:
        with source.open("rb") as source_file:
            assert (
                hashlib.file_digest(source_file, "sha256").hexdigest() == source_digest
            )
        assert _model_snapshot((detection, recognition)) == model_snapshot
