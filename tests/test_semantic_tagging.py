# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End-to-end regression tests for semantic tagged-PDF writing."""

from __future__ import annotations

import zlib
from copy import deepcopy
from io import BytesIO

import pikepdf
import pytest
from conftest import register_form_widget
from pikepdf import Array, Dictionary, Name, NumberTree, String

import pdftopdfa.digital_layout as digital_layout
import pdftopdfa.tagging as tagging
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.tagging import (
    _existing_structure_elements,
    _FigureOCRStatus,
    _has_unambiguous_existing_reading_order_inversion,
    _remove_native_text_from_ocr_word,
    ensure_logical_structure,
)
from pdftopdfa.utils import resolve_indirect


def _font(pdf: pikepdf.Pdf, *, bold: bool = False) -> Dictionary:
    return pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica-Bold") if bold else Name.Helvetica,
            Encoding=Name.WinAnsiEncoding,
        )
    )


def _page(
    pdf: pikepdf.Pdf,
    content: bytes,
    resources: Dictionary | None = None,
    *,
    size: tuple[float, float] = (600, 800),
) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=size)
    page.obj["/Contents"] = pdf.make_stream(content)
    page.obj["/Resources"] = resources or Dictionary()
    return page


def _form(
    pdf: pikepdf.Pdf,
    content: bytes,
    resources: Dictionary | None = None,
    *,
    bbox: tuple[float, float, float, float] = (0, 0, 200, 200),
) -> pikepdf.Stream:
    form = pdf.make_stream(content)
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array(bbox)
    form["/Resources"] = resources or Dictionary()
    return form


def _image(
    pdf: pikepdf.Pdf,
    data: bytes = b"\xff\x00\x00",
    *,
    color_space: Name = Name.DeviceRGB,
) -> pikepdf.Stream:
    image = pdf.make_stream(data)
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = color_space
    image["/BitsPerComponent"] = 8
    return image


def _uncertain_soft_mask_image(pdf: pikepdf.Pdf) -> pikepdf.Stream:
    mask = _image(pdf, zlib.compress(b"\x00"), color_space=Name.DeviceGray)
    mask["/Filter"] = Name.FlateDecode
    mask["/DecodeParms"] = Dictionary(Predictor=12)
    image = _image(pdf)
    image["/SMask"] = mask
    return image


def _nested_form_document(pdf: pikepdf.Pdf, depth: int) -> pikepdf.Page:
    invoked = _form(
        pdf,
        b"BT /F1 10 Tf 20 20 Td (Nested text.) Tj ET",
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )
    for _ in range(depth - 1):
        invoked = _form(
            pdf,
            b"/Nested Do",
            Dictionary(XObject=Dictionary(Nested=invoked)),
        )
    return _page(
        pdf,
        b"/Root Do",
        Dictionary(XObject=Dictionary(Root=invoked)),
        size=(200, 200),
    )


def _structure_objects(pdf: pikepdf.Pdf) -> tuple[Dictionary, ...]:
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    objects: list[Dictionary] = []

    def visit(value: object) -> None:
        resolved = resolve_indirect(value)
        if isinstance(resolved, Array):
            for item in resolved:
                visit(item)
            return
        if not isinstance(resolved, Dictionary):
            return
        objects.append(resolved)
        if resolved.get("/Type") == Name.StructElem and "/K" in resolved:
            visit(resolved["/K"])

    visit(root["/K"])
    return tuple(objects)


def _roles(pdf: pikepdf.Pdf) -> list[str]:
    return [
        str(item["/S"])
        for item in _structure_objects(pdf)
        if item.get("/Type") == Name.StructElem
    ]


def _set_optional_content(
    pdf: pikepdf.Pdf,
    groups: tuple[Dictionary, ...],
    *,
    off: tuple[Dictionary, ...] = (),
) -> None:
    off_keys = {group.objgen for group in off}
    pdf.Root["/OCProperties"] = Dictionary(
        OCGs=Array(groups),
        D=Dictionary(
            BaseState=Name.ON,
            Intent=Name.View,
            Order=Array(groups),
            ON=Array([group for group in groups if group.objgen not in off_keys]),
            OFF=Array(off),
        ),
    )


def _marked_paint_stacks(
    owner: pikepdf.Page | pikepdf.Stream,
) -> list[tuple[str, tuple[str, ...]]]:
    stack: list[str] = []
    paints: list[tuple[str, tuple[str, ...]]] = []
    for instruction in pikepdf.parse_content_stream(owner):
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            paints.append(("inline_image", tuple(stack)))
            continue
        if instruction.operator in {pikepdf.Operator("BMC"), pikepdf.Operator("BDC")}:
            stack.append(str(resolve_indirect(instruction.operands[0])))
        elif instruction.operator == pikepdf.Operator("EMC"):
            stack.pop()
        elif str(instruction.operator) in {
            "Tj",
            "TJ",
            "'",
            '"',
            "Do",
            "S",
            "s",
            "f",
            "F",
            "f*",
            "B",
            "B*",
            "b",
            "b*",
            "sh",
        }:
            paints.append((str(instruction.operator), tuple(stack)))
    assert stack == []
    return paints


@pytest.mark.parametrize(
    ("value", "expected_length"),
    [
        ("a" * 40_000, 32_767),
        ("\u6f22" * 20_000, 16_382),
        ("\U0001f600" * 10_000, 8_191),
    ],
    ids=["ascii", "bmp", "non-bmp"],
)
def test_generated_pdf_text_strings_respect_encoded_byte_limit(
    value: str,
    expected_length: int,
) -> None:
    bounded = tagging._bounded_pdf_string(value)

    assert str(bounded) == value[:expected_length]
    assert len(bytes(bounded)) <= tagging._MAX_STRING_BYTES


def test_ocr_actualtext_override_respects_encoded_byte_limit() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, b"/Span <</MCID 0>> BDC EMC")
    value = "\U0001f600" * 10_000

    tagging._rewrite_ocr_form_semantics(
        form,
        1,
        {},
        {tagging._semantic_span_id(1, "ocr", 0): value},
    )

    marker = next(
        instruction
        for instruction in pikepdf.parse_content_stream(form)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and instruction.operator == tagging._BDC
    )
    properties = resolve_indirect(marker.operands[1])
    actual_text = resolve_indirect(properties["/ActualText"])
    assert isinstance(actual_text, String)
    assert str(actual_text) == value[:8_191]
    assert len(bytes(actual_text)) <= tagging._MAX_STRING_BYTES


def test_inherited_widget_tooltip_respects_encoded_byte_limit() -> None:
    value = "\u6f22" * 20_000
    annotation = Dictionary(Parent=Dictionary(T=String(value)))

    assert tagging._ensure_widget_tooltip(annotation) is True

    tooltip = resolve_indirect(annotation["/TU"])
    assert isinstance(tooltip, String)
    assert str(tooltip) == value[:16_382]
    assert len(bytes(tooltip)) <= tagging._MAX_STRING_BYTES


def _k_objects(element: Dictionary) -> list[Dictionary]:
    value = resolve_indirect(element.get("/K"))
    items = list(value) if isinstance(value, Array) else [value]
    return [
        resolved
        for item in items
        if isinstance((resolved := resolve_indirect(item)), Dictionary)
    ]


def _struct_children(element: Dictionary) -> list[Dictionary]:
    return [
        item for item in _k_objects(element) if item.get("/Type") == Name.StructElem
    ]


def _install_two_paragraph_structure(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    logical_order: tuple[int, int],
) -> tuple[Dictionary, tuple[Dictionary, Dictionary]]:
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    paragraphs = tuple(
        pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.P,
                P=document,
                Pg=page.obj,
                K=mcid,
            )
        )
        for mcid in range(2)
    )
    document["/K"] = Array([paragraphs[mcid] for mcid in logical_order])
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array(paragraphs)
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root
    return document, paragraphs


def _install_single_mcid_structure(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
) -> tuple[Dictionary, Dictionary]:
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    paragraph = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.P,
            P=document,
            Pg=page.obj,
            K=0,
        )
    )
    document["/K"] = paragraph
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([paragraph])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root
    return root, paragraph


def _install_figure_structure(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    *,
    alt_text: str | None = None,
) -> Dictionary:
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj, K=0)
    )
    if alt_text is not None:
        figure["/Alt"] = String(alt_text)
    document["/K"] = figure
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([figure])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root
    return root


def _marked_content(
    container: pikepdf.Page | pikepdf.Stream,
) -> list[tuple[str, str | None, str | None, int | None]]:
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
        type_name = resolve_indirect(properties.get("/Type"))
        subtype = resolve_indirect(properties.get("/Subtype"))
        mcid = resolve_indirect(properties.get("/MCID"))
        markers.append(
            (
                tag,
                str(type_name) if type_name is not None else None,
                str(subtype) if subtype is not None else None,
                int(mcid) if isinstance(mcid, int) else None,
            )
        )
    return markers


def _invoked_forms(
    container: pikepdf.Page | pikepdf.Stream,
) -> list[pikepdf.Stream]:
    owner = container.obj if isinstance(container, pikepdf.Page) else container
    resources = resolve_indirect(owner.get("/Resources"))
    xobjects = (
        resolve_indirect(resources.get("/XObject"))
        if isinstance(resources, Dictionary)
        else None
    )
    forms = []
    for instruction in pikepdf.parse_content_stream(container):
        if (
            not isinstance(instruction, pikepdf.ContentStreamInstruction)
            or str(instruction.operator) != "Do"
            or not instruction.operands
        ):
            continue
        name = resolve_indirect(instruction.operands[0])
        invoked = (
            resolve_indirect(xobjects.get(name))
            if isinstance(name, Name) and isinstance(xobjects, Dictionary)
            else None
        )
        if (
            isinstance(invoked, pikepdf.Stream)
            and resolve_indirect(invoked.get("/Subtype")) == Name.Form
        ):
            forms.append(invoked)
    return forms


def _ocr_document() -> tuple[pikepdf.Pdf, pikepdf.Stream, dict[str, object]]:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = pdf.make_stream(
        b"/Span <</MCID 7>> BDC\n"
        b"BT /F1 10 Tf 3 Tr 1 0 0 1 40 220 Tm (placeholder) Tj ET\n"
        b"EMC"
    )
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 400, 300])
    form["/Resources"] = Dictionary(Font=Dictionary(F1=font))
    xobjects = Dictionary()
    xobjects["/OCR-0"] = form
    _page(
        pdf,
        b"q /OCR-0 Do Q",
        Dictionary(XObject=xobjects),
        size=(400, 300),
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "type": "pdftopdfa-ocr-document",
        "page_count": 1,
        "languages": ["de-DE"],
        "pages": [
            {
                "page_index": 0,
                "form_name": "/OCR-0",
                "coordinates": {"width": 400, "height": 300},
                "lines": [
                    {
                        "mcid": 7,
                        "text": "• Förderung, Größe und Rückgabe",
                        "confidence": 0.99,
                        "bbox": {
                            "left": 40,
                            "top": 60,
                            "right": 360,
                            "bottom": 80,
                        },
                    }
                ],
            }
        ],
    }
    return pdf, form, manifest


@pytest.mark.parametrize(
    ("scan_width", "expected"),
    [
        pytest.param(356, 1, id="eighty-nine-percent"),
        pytest.param(288, 1, id="artifact-threshold"),
        pytest.param(287, 0, id="below-artifact-threshold"),
    ],
)
def test_ocr_background_raster_requires_visual_review(
    scan_width: int,
    expected: int,
) -> None:
    pdf, _form, manifest = _ocr_document()
    page = pdf.pages[0]
    image = pdf.make_stream(b"\xff\xff\xff")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    resources = resolve_indirect(page.obj["/Resources"])
    xobjects = resolve_indirect(resources["/XObject"])
    xobjects["/Scan"] = image
    page.obj["/Contents"].write(
        f"q {scan_width} 0 0 300 0 0 cm /Scan Do Q\nq /OCR-0 Do Q".encode("ascii")
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_scanned_visual_review_required"] == expected
    assert result["semantic_content_items"] == 1 + int(not expected)
    assert _roles(pdf).count("/Figure") == int(not expected)


def test_scanned_visual_review_requires_ocr_and_full_page_raster() -> None:
    ocr_pdf, _form, manifest = _ocr_document()

    ocr_result = ensure_logical_structure(
        ocr_pdf,
        semantic=True,
        ocr_manifest=manifest,
    )

    assert ocr_result["semantic_scanned_visual_review_required"] == 0

    digital_pdf = pikepdf.Pdf.new()
    image = digital_pdf.make_stream(b"\xff\xff\xff")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    _page(
        digital_pdf,
        b"q 400 0 0 300 0 0 cm /Scan Do Q",
        Dictionary(XObject=Dictionary(Scan=image)),
        size=(400, 300),
    )

    digital_result = ensure_logical_structure(digital_pdf, semantic=True)

    assert digital_result["semantic_scanned_visual_review_required"] == 0


def test_invisible_ocr_page_rasters_do_not_require_visual_review() -> None:
    pdf, _form, manifest = _ocr_document()
    page = pdf.pages[0]
    image = pdf.make_stream(b"\xff\xff\xff")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    resources = resolve_indirect(page.obj["/Resources"])
    resources["/ExtGState"] = Dictionary(Hidden=Dictionary(ca=0))
    xobjects = resolve_indirect(resources["/XObject"])
    xobjects["/Scan"] = image
    page.obj["/Contents"].write(
        b"q /Hidden gs 400 0 0 300 0 0 cm /Scan Do Q\n"
        b"q 0 0 0 0 re W n 400 0 0 300 0 0 cm /Scan Do Q\n"
        b"q /OCR-0 Do Q"
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_scanned_visual_review_required"] == 0


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize(
    ("rotation", "raw_bbox", "coordinate_size"),
    [
        (0, (30, 70, 50, 80), (200, 100)),
        (90, (20, 30, 30, 50), (100, 200)),
        (180, (150, 20, 170, 30), (200, 100)),
        (270, (70, 150, 80, 170), (100, 200)),
    ],
)
def test_ocr_bbox_uses_rotated_media_then_clips_to_crop(
    rotation: int,
    raw_bbox: tuple[int, int, int, int],
    coordinate_size: tuple[int, int],
    inherited: bool,
) -> None:
    pdf, form, manifest = _ocr_document()
    page = pdf.pages[0]
    owner = resolve_indirect(page.obj["/Parent"]) if inherited else page.obj
    owner["/MediaBox"] = Array([10, 20, 210, 120])
    owner["/CropBox"] = Array([30, 30, 180, 100])
    owner["/Rotate"] = rotation
    if inherited:
        for key in ("/MediaBox", "/CropBox", "/Rotate"):
            if key in page.obj:
                del page.obj[key]
    page.obj["/UserUnit"] = 2.5
    raw_page = manifest["pages"][0]
    raw_page["coordinates"] = {
        "width": coordinate_size[0],
        "height": coordinate_size[1],
    }
    raw_page["lines"][0]["bbox"] = dict(
        zip(("left", "top", "right", "bottom"), raw_bbox, strict=True)
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_content_items"] == 1
    parent_tree = NumberTree(pdf.Root["/StructTreeRoot"]["/ParentTree"])
    owner_element = resolve_indirect(parent_tree[int(form["/StructParents"])][7])
    attributes = resolve_indirect(owner_element["/A"])
    assert tuple(float(value) for value in attributes["/BBox"]) == pytest.approx(
        (40, 40, 60, 50)
    )


@pytest.mark.parametrize(
    ("raw_bbox", "expected_bbox", "artifact"),
    [
        ((10, 70, 40, 80), (30, 40, 50, 50), False),
        ((0, 70, 10, 80), None, True),
    ],
)
def test_ocr_bbox_crop_intersection_excludes_fully_hidden_text(
    raw_bbox: tuple[int, int, int, int],
    expected_bbox: tuple[int, int, int, int] | None,
    artifact: bool,
) -> None:
    pdf, form, manifest = _ocr_document()
    page = pdf.pages[0]
    page.obj["/MediaBox"] = Array([10, 20, 210, 120])
    page.obj["/CropBox"] = Array([30, 30, 180, 100])
    raw_page = manifest["pages"][0]
    raw_page["coordinates"] = {"width": 200, "height": 100}
    raw_page["lines"][0]["bbox"] = dict(
        zip(("left", "top", "right", "bottom"), raw_bbox, strict=True)
    )
    raw_page["lines"][0]["text"] = "HIDDEN" if artifact else "HIDDEN VISIBLE"
    raw_page["lines"][0]["words"] = (
        [
            {
                "text": "HIDDEN",
                "bbox": {"left": 0, "top": 70, "right": 10, "bottom": 80},
            }
        ]
        if artifact
        else [
            {
                "text": "HIDDEN",
                "bbox": {"left": 10, "top": 70, "right": 20, "bottom": 80},
            },
            {
                "text": "VISIBLE",
                "bbox": {"left": 20, "top": 70, "right": 40, "bottom": 80},
            },
        ]
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    if artifact:
        assert result["semantic_content_items"] == 0
        assert result["artifacts_tagged"] == 1
        assert "/StructParents" not in form
        assert _marked_content(form) == [
            ("/Artifact", "/Layout", None, None),
        ]
    else:
        assert result["semantic_content_items"] == 1
        parent_tree = NumberTree(pdf.Root["/StructTreeRoot"]["/ParentTree"])
        owner_element = resolve_indirect(parent_tree[int(form["/StructParents"])][7])
        attributes = resolve_indirect(owner_element["/A"])
        assert tuple(float(value) for value in attributes["/BBox"]) == pytest.approx(
            expected_bbox
        )
        markers = [
            instruction
            for instruction in pikepdf.parse_content_stream(form)
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and str(instruction.operator) == "BDC"
        ]
        properties = resolve_indirect(markers[0].operands[1])
        assert str(properties["/ActualText"]) == "VISIBLE"


def test_digital_tj_tj_and_image_have_stable_parent_tree_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf 20 250 Td (First paragraph.) Tj "
            b"0 -20 Td [(Second) 0 ( line.)] TJ ET "
            b"q 40 0 0 30 300 100 cm /Im0 Do Q"
        ),
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Im0=image),
        ),
        size=(400, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    assert result["semantic_content_items"] == 3
    assert {"/P", "/Figure"} <= set(_roles(pdf))
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert "/Alt" not in figure
    assert result["semantic_alternatives_review_required"] == 1
    markers = [marker for marker in _marked_content(page) if marker[0] == "/Span"]
    assert [marker[3] for marker in markers] == [0, 1, 2]

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    page_key = int(page.obj["/StructParents"])
    parent_array = resolve_indirect(NumberTree(root["/ParentTree"])[page_key])
    assert isinstance(parent_array, Array)
    assert len(parent_array) == 3
    assert all(
        isinstance(resolve_indirect(owner), Dictionary) for owner in parent_array
    )
    assert int(root["/ParentTreeNextKey"]) > page_key

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        reopened_key = int(reopened.pages[0].obj["/StructParents"])
        reopened_array = resolve_indirect(
            NumberTree(reopened_root["/ParentTree"])[reopened_key]
        )
        assert isinstance(reopened_array, Array)
        assert len(reopened_array) == 3
        assert {"/P", "/Figure"} <= set(_roles(reopened))
        preserved = ensure_logical_structure(reopened, semantic=True)
        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert preserved["semantic_alternatives_review_required"] == 1


@pytest.mark.parametrize(
    ("recognized", "expected_actual_text", "ocr_review", "artifact_count"),
    [
        (
            "INFO Professionelle Lösungen für erfolgreiche Arbeit",
            "INFO Professionelle Lösungen für erfolgreiche Arbeit",
            1,
            0,
        ),
        (None, None, 0, 1),
    ],
)
def test_direct_image_figure_uses_review_required_ocr_actualtext(
    recognized: str | None,
    expected_actual_text: str | None,
    ocr_review: int,
    artifact_count: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    _page(
        pdf,
        b"q 50 100 50 80 re W n 100 0 0 80 50 100 cm /Im Do Q",
        Dictionary(XObject=Dictionary(Im=image)),
        size=(400, 300),
    )
    seen: list[tuple[int, int]] = []
    seen_crops: list[tuple[tuple[float, float], ...]] = []

    def recognize(
        candidate: pikepdf.Stream,
        crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        seen.append(candidate.objgen)
        seen_crops.append(crop_polygon)
        return recognized

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    figures = [
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    ]
    assert seen == [image.objgen]
    assert min(x for x, _y in seen_crops[0]) == pytest.approx(0.0)
    assert max(x for x, _y in seen_crops[0]) == pytest.approx(0.5)
    assert min(y for _x, y in seen_crops[0]) == pytest.approx(0.0)
    assert max(y for _x, y in seen_crops[0]) == pytest.approx(1.0)
    if expected_actual_text is None:
        assert not figures
        assert ("/Artifact", "/Layout", None, None) in _marked_content(pdf.pages[0])
    else:
        figure = figures[0]
        assert str(figure["/ActualText"]) == expected_actual_text
        assert "/Alt" not in figure
        layout = resolve_indirect(figure["/A"])
        assert layout["/O"] == Name.Layout
        assert layout["/Placement"] == Name.Block
    assert result["semantic_alternatives_review_required"] == 0
    assert result["semantic_ocr_figure_text_review_required"] == ocr_review
    assert result["semantic_ocr_figure_artifacts"] == artifact_count


@pytest.mark.parametrize("nested", [False, True])
def test_preserved_direct_image_figure_uses_ocr_actualtext(nested: bool) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    if nested:
        form = _form(
            pdf,
            b"q 100 0 0 80 50 100 cm /Im Do Q",
            Dictionary(XObject=Dictionary(Im=image)),
        )
        content = b"/Figure <</MCID 0>> BDC /Fm Do EMC"
        resources = Dictionary(XObject=Dictionary(Fm=form))
    else:
        content = b"/Figure <</MCID 0>> BDC q 100 0 0 80 50 100 cm /Im Do Q EMC"
        resources = Dictionary(XObject=Dictionary(Im=image))
    page = _page(
        pdf,
        content,
        resources,
        size=(400, 300),
    )
    root = _install_figure_structure(pdf, page)
    seen: list[tuple[int, int]] = []

    def recognize(
        candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        seen.append(candidate.objgen)
        return "Existing Figure text"

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert seen == [image.objgen]
    assert str(figure["/ActualText"]) == "Existing Figure text"
    assert result["semantic_alternatives_review_required"] == 0
    assert result["semantic_ocr_figure_text_review_required"] == 1
    assert result["semantic_ocr_figure_artifacts"] == 0


@pytest.mark.parametrize("nested", [False, True])
def test_preserved_direct_image_figure_rejected_by_ocr_becomes_artifact(
    nested: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    if nested:
        form = _form(
            pdf,
            b"q 100 0 0 80 50 100 cm /Im Do Q",
            Dictionary(XObject=Dictionary(Im=image)),
        )
        content = b"/Figure <</MCID 0>> BDC /Fm Do EMC"
        resources = Dictionary(XObject=Dictionary(Fm=form))
    else:
        content = b"/Figure <</MCID 0>> BDC q 100 0 0 80 50 100 cm /Im Do Q EMC"
        resources = Dictionary(XObject=Dictionary(Im=image))
    page = _page(pdf, content, resources, size=(400, 300))

    root = _install_figure_structure(pdf, page)
    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=lambda _image, _crop: None,
    )

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert "/Figure" not in _roles(pdf)
    artifact_markers = [
        marker
        for item in pdf.objects
        if isinstance(item, pikepdf.Stream) and item.get("/Subtype") == Name.Form
        for marker in _marked_content(item)
    ]
    artifact_markers.extend(_marked_content(page))
    assert ("/Artifact", "/Layout", None, None) in artifact_markers
    assert result["semantic_alternatives_review_required"] == 0
    assert result["semantic_ocr_figure_text_review_required"] == 0
    assert result["semantic_ocr_figure_artifacts"] == 1


def test_rejected_figure_preserves_other_author_structure() -> None:
    pdf = pikepdf.Pdf.new()
    described_image = _image(pdf, b"\xff\x00\x00")
    rejected_image = _image(pdf, b"\x00\xff\x00")
    page = _page(
        pdf,
        (
            b"/Illustration <</MCID 0>> BDC "
            b"q 80 0 0 80 30 100 cm /Described Do Q EMC "
            b"/Figure <</MCID 1>> BDC "
            b"q 80 0 0 80 150 100 cm /Rejected Do Q EMC"
        ),
        Dictionary(
            XObject=Dictionary(Described=described_image, Rejected=rejected_image)
        ),
        size=(400, 300),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(
        Dictionary(
            Type=Name.StructTreeRoot,
            RoleMap=Dictionary(Illustration=Name.Figure),
        )
    )
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    described = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name("/Illustration"),
            P=document,
            Pg=page.obj,
            K=0,
            Alt=String("Author description"),
        )
    )
    rejected = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj, K=1)
    )
    document["/K"] = Array([described, rejected])
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([described, rejected])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    seen = []

    def recognize(
        candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> None:
        seen.append(candidate.objgen)

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert seen == [rejected_image.objgen]
    assert _roles(pdf) == ["/Document", "/Illustration"]
    assert str(described["/Alt"]) == "Author description"
    assert str(root["/RoleMap"]["/Illustration"]) == "/Figure"
    assert _marked_content(page) == [
        ("/Illustration", None, None, 0),
        ("/Artifact", "/Layout", None, None),
    ]
    parent_array = resolve_indirect(NumberTree(root["/ParentTree"])[0])
    assert isinstance(parent_array, Array)
    assert len(parent_array) == 1
    assert parent_array[0].objgen == described.objgen
    assert result["semantic_ocr_figure_artifacts"] == 1


def test_ocr_ineligible_figure_remains_structured_for_manual_review() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    _page(
        pdf,
        b"q 100 0 0 80 50 100 cm /Im Do Q",
        Dictionary(XObject=Dictionary(Im=image)),
        size=(400, 300),
    )

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=(lambda _image, _crop: _FigureOCRStatus.INELIGIBLE),
    )

    assert "/Figure" in _roles(pdf)
    assert ("/Artifact", "/Layout", None, None) not in _marked_content(pdf.pages[0])
    assert result["semantic_alternatives_review_required"] == 1
    assert result["semantic_ocr_figure_text_review_required"] == 0
    assert result["semantic_ocr_figure_artifacts"] == 0


def test_existing_image_actualtext_skips_figure_ocr() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    _page(
        pdf,
        (
            b"/Figure <</ActualText (Existing replacement)>> BDC "
            b"q 100 0 0 80 50 100 cm /Im Do Q EMC"
        ),
        Dictionary(XObject=Dictionary(Im=image)),
        size=(400, 300),
    )

    def recognize(
        _candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        raise AssertionError("OCR must not replace existing ActualText")

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Existing replacement"
    assert result["semantic_alternatives_review_required"] == 0
    assert result["semantic_ocr_figure_text_review_required"] == 0


def test_described_nested_figure_is_not_replaced_by_parent_figure_ocr() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    page = _page(
        pdf,
        b"/Figure <</MCID 0>> BDC q 100 0 0 80 50 100 cm /Im Do Q EMC",
        Dictionary(XObject=Dictionary(Im=image)),
        size=(400, 300),
    )
    root = _install_figure_structure(pdf, page, alt_text="Author description")
    document = resolve_indirect(root["/K"])
    inner = resolve_indirect(document["/K"])
    outer = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, K=inner)
    )
    inner["/P"] = outer
    document["/K"] = outer

    def recognize(
        _candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        raise AssertionError("OCR must not replace a nested Figure description")

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    assert result["structure_preserved"] is True
    assert "/ActualText" not in outer
    assert str(inner["/Alt"]) == "Author description"
    assert result["semantic_alternatives_review_required"] == 1
    assert result["semantic_ocr_figure_text_review_required"] == 0


def test_translucent_direct_image_figure_skips_ocr_actualtext() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    _page(
        pdf,
        b"q /Faint gs 100 0 0 80 50 100 cm /Im Do Q",
        Dictionary(
            ExtGState=Dictionary(Faint=Dictionary(ca=0.01)),
            XObject=Dictionary(Im=image),
        ),
        size=(400, 300),
    )

    def recognize(
        _candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        raise AssertionError("OCR must not inspect a translucent image invocation")

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        _figure_text_recognizer=recognize,
    )

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert "/ActualText" not in figure
    assert result["semantic_alternatives_review_required"] == 1
    assert result["semantic_ocr_figure_text_review_required"] == 0


def test_page_ocr_text_is_not_duplicated_as_figure_actualtext() -> None:
    pdf, _form, manifest = _ocr_document()
    page = pdf.pages[0]
    resources = resolve_indirect(page.obj["/Resources"])
    xobjects = resolve_indirect(resources["/XObject"])
    xobjects["/Scan"] = _image(pdf)
    page.obj["/Contents"].write(b"q 100 0 0 80 40 210 cm /Scan Do Q\nq /OCR-0 Do Q")

    def recognize(
        _candidate: pikepdf.Stream,
        _crop_polygon: tuple[tuple[float, float], ...],
    ) -> str | None:
        raise AssertionError("OCR text must not be duplicated on its source Figure")

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        preflight=False,
        ocr_manifest=manifest,
        _figure_text_recognizer=recognize,
    )

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert "/ActualText" not in figure
    assert result["semantic_content_items"] == 2
    assert result["semantic_alternatives_review_required"] == 1
    assert result["semantic_ocr_figure_text_review_required"] == 0


def test_pdfua_does_not_invent_fallback_alt_or_document_language() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    image = _image(pdf)
    _page(
        pdf,
        (
            b"BT /F1 12 Tf 20 250 Td "
            b"(Rechnung Kunde Menge Zahlungshinweis) Tj ET "
            b"q 40 0 0 30 300 100 cm /Im0 Do Q"
        ),
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Im0=image),
        ),
        size=(400, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True, pdfua=True)

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert "/Lang" not in pdf.Root
    assert "/Alt" not in figure
    assert result["semantic_alternatives_review_required"] == 1


def test_overlaid_invisible_digital_text_is_tagged_once() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf 0 Tr 1 0 0 1 72 700 Tm (Hello) Tj "
            b"3 Tr 1 0 0 1 72 700 Tm (Hello) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["artifacts_tagged"] == 1
    assert _marked_content(page) == [
        ("/Span", None, None, 0),
        ("/Artifact", "/Layout", None, None),
    ]


def test_separate_invisible_digital_text_remains_logical_content() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf 0 Tr 1 0 0 1 72 700 Tm (Hello) Tj "
            b"3 Tr 1 0 0 1 72 650 Tm (Hello) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    assert [marker[0] for marker in _marked_content(page)] == ["/Span", "/Span"]


def test_iccbased_text_color_has_deterministic_paint_visibility() -> None:
    pdf = pikepdf.Pdf.new()
    profile = pdf.make_stream(b"")
    profile["/N"] = 1
    page = _page(
        pdf,
        b"/CS0 cs 0 scn BT /F1 12 Tf 20 200 Td (Visible text) Tj ET",
        Dictionary(
            ColorSpace=Dictionary(CS0=Array([Name.ICCBased, profile])),
            Font=Dictionary(F1=_font(pdf)),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _marked_content(page)[0][0] == "/Span"


def test_semantic_link_owns_overlapping_text_and_objr_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([35, 215, 170, 235]),
            A=Dictionary(S=Name.URI, URI=String("https://example.com/docs")),
        )
    )
    page = _page(
        pdf,
        b"BT /F1 12 Tf 1 0 0 1 40 220 Tm (Open documentation) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    page.obj["/Annots"] = Array([annotation])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    assert result["annotations_tagged"] == 1
    assert result["semantic_link_review_required"] == 0
    assert str(annotation["/Contents"]) == "Open documentation"
    assert "/Tabs" not in page.obj
    link = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Link)
    link_kids = _k_objects(link)
    assert [item.get("/Type") for item in link_kids] == [Name.MCR, Name.OBJR]
    mcr, objr = link_kids
    assert int(mcr["/MCID"]) == 0
    assert resolve_indirect(mcr["/Pg"]).objgen == page.obj.objgen
    assert resolve_indirect(objr["/Obj"]).objgen == annotation.objgen
    assert resolve_indirect(objr["/Pg"]).objgen == page.obj.objgen
    assert resolve_indirect(link["/P"]).get("/S") == Name.P

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    page_owners = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
    assert isinstance(page_owners, Array)
    assert resolve_indirect(page_owners[0]).objgen == link.objgen
    assert (
        resolve_indirect(parent_tree[int(annotation["/StructParent"])]).objgen
        == link.objgen
    )

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        root_objgen = reopened_root.objgen
        reopened_page = reopened.pages[0]
        reopened_annotation = resolve_indirect(reopened_page.obj["/Annots"][0])
        assert str(reopened_annotation["/Contents"]) == "Open documentation"
        reopened_link = next(
            item for item in _structure_objects(reopened) if item.get("/S") == Name.Link
        )
        reopened_kids = _k_objects(reopened_link)
        assert [item.get("/Type") for item in reopened_kids] == [
            Name.MCR,
            Name.OBJR,
        ]
        assert resolve_indirect(reopened_kids[1]["/Obj"]).objgen == (
            reopened_annotation.objgen
        )
        reopened_tree = NumberTree(reopened_root["/ParentTree"])
        reopened_owners = resolve_indirect(
            reopened_tree[int(reopened_page.obj["/StructParents"])]
        )
        assert isinstance(reopened_owners, Array)
        assert resolve_indirect(reopened_owners[0]).objgen == reopened_link.objgen
        assert (
            resolve_indirect(
                reopened_tree[int(reopened_annotation["/StructParent"])]
            ).objgen
            == reopened_link.objgen
        )

        preserved = ensure_logical_structure(reopened, semantic=True)

        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert reopened.Root["/StructTreeRoot"].objgen == root_objgen


def test_multi_owner_link_is_preserved_for_review_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    regular = _font(pdf)
    bold = _font(pdf, bold=True)
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([35, 205, 250, 280]),
            A=Dictionary(S=Name.URI, URI=String("https://example.com/topic")),
        )
    )
    page = _page(
        pdf,
        b"BT /FB 22 Tf 40 250 Td (Linked heading) Tj "
        b"/F1 12 Tf 0 -30 Td (Linked paragraph.) Tj ET",
        Dictionary(Font=Dictionary(F1=regular, FB=bold)),
        size=(400, 300),
    )
    page.obj["/Annots"] = Array([annotation])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_link_review_required"] == 1
    heading = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.H1
    )
    paragraph = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.P
    )
    link = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Link)
    assert resolve_indirect(link["/P"]).get("/S") == Name.Div
    assert resolve_indirect(link["/K"]).get("/Type") == Name.OBJR
    assert resolve_indirect(link["/K"])["/Obj"].objgen == annotation.objgen
    assert resolve_indirect(heading["/K"]).get("/Type") == Name.MCR
    assert resolve_indirect(paragraph["/K"]).get("/Type") == Name.MCR

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    page_owners = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
    assert isinstance(page_owners, Array)
    assert resolve_indirect(page_owners[0]).objgen == heading.objgen
    assert resolve_indirect(page_owners[1]).objgen == paragraph.objgen
    assert (
        resolve_indirect(parent_tree[int(annotation["/StructParent"])]).objgen
        == link.objgen
    )

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_page = reopened.pages[0]
        reopened_annotation = resolve_indirect(reopened_page.obj["/Annots"][0])
        reopened_link = next(
            item for item in _structure_objects(reopened) if item.get("/S") == Name.Link
        )
        reopened_tree = NumberTree(reopened.Root["/StructTreeRoot"]["/ParentTree"])

        assert resolve_indirect(reopened_link["/K"])["/Obj"].objgen == (
            reopened_annotation.objgen
        )
        assert (
            resolve_indirect(
                reopened_tree[int(reopened_annotation["/StructParent"])]
            ).objgen
            == reopened_link.objgen
        )


def test_semantic_link_quadpoints_exclude_rect_gap_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([35, 205, 180, 265]),
            QuadPoints=Array(
                [
                    35,
                    265,
                    180,
                    265,
                    35,
                    245,
                    180,
                    245,
                    35,
                    225,
                    180,
                    225,
                    35,
                    205,
                    180,
                    205,
                ]
            ),
            A=Dictionary(S=Name.URI, URI=String("https://example.com/two-lines")),
        )
    )
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf "
            b"1 0 0 1 40 250 Tm (First linked line) Tj "
            b"1 0 0 1 40 230 Tm (Unrelated middle) Tj "
            b"1 0 0 1 40 210 Tm (Second linked line) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    page.obj["/Annots"] = Array([annotation])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 3
    link = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Link)
    link_kids = _k_objects(link)
    assert [item.get("/Type") for item in link_kids] == [
        Name.MCR,
        Name.MCR,
        Name.OBJR,
    ]
    assert [int(item["/MCID"]) for item in link_kids[:-1]] == [0, 2]
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    owners = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
    assert isinstance(owners, Array)
    assert [resolve_indirect(owners[index]).objgen for index in (0, 2)] == [
        link.objgen,
        link.objgen,
    ]
    assert resolve_indirect(owners[1]).objgen != link.objgen
    assert (
        resolve_indirect(parent_tree[int(annotation["/StructParent"])]).objgen
        == link.objgen
    )

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        root_objgen = reopened_root.objgen
        reopened_page = reopened.pages[0]
        reopened_annotation = resolve_indirect(reopened_page.obj["/Annots"][0])
        reopened_link = next(
            item for item in _structure_objects(reopened) if item.get("/S") == Name.Link
        )
        reopened_kids = _k_objects(reopened_link)
        assert [
            int(item["/MCID"])
            for item in reopened_kids
            if item.get("/Type") == Name.MCR
        ] == [0, 2]
        reopened_tree = NumberTree(reopened_root["/ParentTree"])
        reopened_owners = resolve_indirect(
            reopened_tree[int(reopened_page.obj["/StructParents"])]
        )
        assert isinstance(reopened_owners, Array)
        assert resolve_indirect(reopened_owners[1]).objgen != reopened_link.objgen
        assert (
            resolve_indirect(
                reopened_tree[int(reopened_annotation["/StructParent"])]
            ).objgen
            == reopened_link.objgen
        )

        preserved = ensure_logical_structure(reopened, semantic=True)

        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert reopened.Root["/StructTreeRoot"].objgen == root_objgen


@pytest.mark.parametrize(
    ("quad_points", "message"),
    [
        ([35, 265, 180, 265, 35, 245, 180], "is malformed"),
        (
            [35, 265, 180, 265, 35, 245, String("invalid"), 245],
            "is malformed",
        ),
        ([35, 245, 35, 245, 35, 245, 35, 245], "empty quadrilateral"),
    ],
)
def test_semantic_link_quadpoints_fail_closed(
    quad_points: list[object],
    message: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([35, 205, 180, 265]),
            QuadPoints=Array(quad_points),
            A=Dictionary(S=Name.URI, URI=String("https://example.com/malformed")),
        )
    )
    page = _page(pdf, b"q Q", size=(400, 300))
    page.obj["/Annots"] = Array([annotation])
    contents = resolve_indirect(page.obj["/Contents"])
    contents["/StructParents"] = 91
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_annotation_keys = set(annotation.keys())
    original_content_keys = set(contents.keys())
    original_content_objgen = contents.objgen
    original_content = bytes(contents.read_bytes())
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match=message):
        ensure_logical_structure(pdf, semantic=True)

    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert set(annotation.keys()) == original_annotation_keys
    assert set(contents.keys()) == original_content_keys
    assert resolve_indirect(page.obj["/Contents"]).objgen == original_content_objgen
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )
    assert len(pdf.objects) == original_object_count


def test_semantic_late_annotation_failure_is_atomic_across_pages() -> None:
    pdf = pikepdf.Pdf.new()
    first_widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([20, 200, 140, 225]),
            Contents=String("First note"),
        )
    )
    malformed_widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([20, 100, String("bad"), 125]),
            Contents=String("Second note"),
        )
    )
    first_page = _page(pdf, b"q Q", size=(300, 300))
    second_page = _page(pdf, b"q Q", size=(300, 300))
    first_page.obj["/Annots"] = Array([first_widget])
    second_page.obj["/Annots"] = Array([malformed_widget])
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = [set(page.obj.keys()) for page in pdf.pages]
    original_annotation_keys = [
        set(first_widget.keys()),
        set(malformed_widget.keys()),
    ]
    original_contents = [
        (
            resolve_indirect(page.obj["/Contents"]).objgen,
            bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()),
        )
        for page in pdf.pages
    ]
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="annotation /Rect is malformed"):
        ensure_logical_structure(pdf, semantic=True)

    assert set(pdf.Root.keys()) == original_root_keys
    assert [set(page.obj.keys()) for page in pdf.pages] == original_page_keys
    assert [set(first_widget.keys()), set(malformed_widget.keys())] == (
        original_annotation_keys
    )
    assert [
        (
            resolve_indirect(page.obj["/Contents"]).objgen,
            bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()),
        )
        for page in pdf.pages
    ] == original_contents
    assert len(pdf.objects) == original_object_count


def test_semantic_preflight_preserves_extraction_permissions_and_is_atomic() -> None:
    source = pikepdf.Pdf.new()
    font = _font(source)
    _page(
        source,
        b"BT /F1 12 Tf 20 200 Td (Protected text.) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
    )
    serialized = BytesIO()
    source.save(
        serialized,
        encryption=pikepdf.Encryption(
            owner="owner",
            user="user",
            allow=pikepdf.Permissions(extract=False),
            R=6,
        ),
    )
    serialized.seek(0)

    with pikepdf.Pdf.open(serialized, password="owner") as pdf:
        page = pdf.pages[0]
        original_root_keys = set(pdf.Root.keys())
        original_page_keys = set(page.obj.keys())
        original_content = bytes(page.obj["/Contents"].read_bytes())
        original_object_count = len(pdf.objects)

        with pytest.raises(ConversionError, match="extraction is not permitted"):
            ensure_logical_structure(pdf, semantic=True)

        assert pdf.allow.extract is False
        assert set(pdf.Root.keys()) == original_root_keys
        assert set(page.obj.keys()) == original_page_keys
        assert bytes(page.obj["/Contents"].read_bytes()) == original_content
        assert len(pdf.objects) == original_object_count


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_semantic_geometry_honors_crop_rotation_inheritance_and_user_unit(
    rotation: int,
    inherited: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    page = _page(
        pdf,
        b"q 20 0 0 10 40 40 cm /Im0 Do Q",
        Dictionary(XObject=Dictionary(Im0=image)),
        size=(200, 100),
    )
    page.obj["/MediaBox"] = Array([10, 20, 210, 120])
    page.obj["/UserUnit"] = 2.5
    geometry_owner = resolve_indirect(page.obj["/Parent"]) if inherited else page.obj
    geometry_owner["/CropBox"] = Array([30, 30, 180, 100])
    geometry_owner["/Rotate"] = rotation
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([40, 40, 60, 50]),
            QuadPoints=Array([40, 50, 60, 50, 40, 40, 60, 40]),
            A=Dictionary(S=Name.URI, URI=String("https://example.com/image")),
        )
    )
    page.obj["/Annots"] = Array([annotation])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    page_div = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Div
    )
    assert tuple(float(value) for value in figure["/A"]["/BBox"]) == pytest.approx(
        (40, 40, 60, 50)
    )
    assert tuple(float(value) for value in page_div["/A"]["/BBox"]) == pytest.approx(
        (30, 30, 180, 100)
    )
    link = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Link)
    assert resolve_indirect(link["/P"]).objgen == figure.objgen
    link_kids = _k_objects(link)
    assert [item.get("/Type") for item in link_kids] == [Name.MCR, Name.OBJR]
    assert resolve_indirect(link_kids[1]["/Obj"]).objgen == annotation.objgen

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    page_owners = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
    assert isinstance(page_owners, Array)
    assert resolve_indirect(page_owners[0]).objgen == link.objgen
    assert (
        resolve_indirect(parent_tree[int(annotation["/StructParent"])]).objgen
        == link.objgen
    )


def test_digital_text_wholly_outside_cropbox_is_a_layout_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (b"BT /F1 10 Tf 1 0 0 1 10 50 Tm (hidden) Tj 1 0 0 1 70 50 Tm (visible) Tj ET"),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )
    page.obj["/CropBox"] = Array([50, 0, 150, 100])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["artifacts_tagged"] == 1
    assert _roles(pdf).count("/P") == 1
    markers = _marked_content(page)
    assert markers == [
        ("/Artifact", "/Layout", None, None),
        ("/Span", None, None, 0),
    ]


def test_digital_text_wholly_outside_active_clip_is_a_layout_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"q 0 0 20 20 re W n "
            b"BT /F1 5 Tf 5 5 Td (inside) Tj 95 95 Td (outside) Tj ET Q"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["artifacts_tagged"] == 1
    assert _roles(pdf).count("/P") == 1
    assert _marked_content(page) == [
        ("/Span", None, None, 0),
        ("/Artifact", "/Layout", None, None),
    ]


def test_text_clip_only_run_fails_closed_without_mutating_source() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"BT /F1 20 Tf 7 Tr 5 5 Td (X) Tj ET "
            b"BT /F1 5 Tf 0 Tr 100 50 Td (hidden) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )

    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="Cannot create semantic digital PDF"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("render_mode", [4, 5, 6])
def test_painted_text_clip_run_fails_closed(render_mode: int) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            f"BT /F1 20 Tf {render_mode} Tr 5 5 Td (X) Tj ET "
            "BT /F1 5 Tf 0 Tr 100 50 Td (hidden) Tj ET"
        ).encode(),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )

    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="Cannot create semantic digital PDF"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_direct_vector_only_page_is_reported_for_manual_review() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"q 1 0 0 RG 5 w 20 20 m 180 80 l S Q",
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["artifacts_tagged"] == 1
    assert result["semantic_vector_review_required"] == 1
    assert _marked_content(page) == [("/Artifact", "/Layout", None, None)]


def test_semantic_widget_uses_form_role_and_inherited_tooltip() -> None:
    pdf = pikepdf.Pdf.new()
    field = pdf.make_indirect(Dictionary(FT=Name.Tx, T=String("Customer email")))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([40, 220, 240, 245]),
            Parent=field,
            TU=String(""),
        )
    )
    field["/Kids"] = Array([widget])
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([field])))
    page = _page(pdf, b"q Q", size=(400, 300))
    page.obj["/Annots"] = Array([widget])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    assert result["annotations_tagged"] == 1
    assert result["semantic_form_review_required"] == 0
    assert str(widget["/TU"]) == "Customer email"
    assert str(field["/TU"]) == "Customer email"
    form = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Form)
    object_reference = resolve_indirect(form["/K"])
    assert object_reference["/Type"] == Name.OBJR
    assert resolve_indirect(object_reference["/Obj"]).objgen == widget.objgen
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    owner = resolve_indirect(
        NumberTree(root["/ParentTree"])[int(widget["/StructParent"])]
    )
    assert owner.objgen == form.objgen

    second = ensure_logical_structure(pdf, semantic=True)

    assert second["structure_preserved"] is True
    assert second["semantic_repairs"] == 0
    assert second["semantic_form_review_required"] == 0


@pytest.mark.parametrize("key", ["TU", "T"])
@pytest.mark.parametrize("inherited", [False, True])
def test_trusted_widget_names_do_not_require_review(
    key: str,
    inherited: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    field = pdf.make_indirect(Dictionary(FT=Name.Tx))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([40, 220, 240, 245]),
            Parent=field,
        )
    )
    (field if inherited else widget)[f"/{key}"] = String("Customer email")
    field["/Kids"] = Array([widget])
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([field])))
    page = _page(pdf, b"q Q", size=(400, 300))
    page.obj["/Annots"] = Array([widget])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_form_review_required"] == 0
    assert str(widget["/TU"]) == "Customer email"


def test_unlabeled_widget_requires_review_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    field = pdf.make_indirect(Dictionary(FT=Name.Tx))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([40, 220, 240, 245]),
            Parent=field,
        )
    )
    field["/Kids"] = Array([widget])
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([field])))
    page = _page(pdf, b"q Q", size=(400, 300))
    page.obj["/Annots"] = Array([widget])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_form_review_required"] == 1
    assert "/TU" not in widget
    form = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Form)
    parent_tree = NumberTree(pdf.Root["/StructTreeRoot"]["/ParentTree"])
    assert (
        resolve_indirect(parent_tree[int(widget["/StructParent"])]).objgen
        == form.objgen
    )

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_page = reopened.pages[0]
        reopened_widget = resolve_indirect(reopened_page.obj["/Annots"][0])

        preserved = ensure_logical_structure(reopened, semantic=True)

        assert preserved["structure_preserved"] is True
        assert preserved["semantic_repairs"] == 0
        assert preserved["semantic_form_review_required"] == 1
        reopened_tree = NumberTree(reopened.Root["/StructTreeRoot"]["/ParentTree"])
        reopened_owner = resolve_indirect(
            reopened_tree[int(reopened_widget["/StructParent"])]
        )
        assert reopened_owner.get("/S") == Name.Form
        assert "/TU" not in reopened_widget


def test_semantic_widgets_follow_spatial_reading_order_after_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf "
            b"1 0 0 1 40 250 Tm (First label) Tj "
            b"1 0 0 1 40 150 Tm (Second label) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    first = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            T=String("First field"),
            Rect=Array([40, 210, 240, 230]),
        )
    )
    second = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            T=String("Second field"),
            Rect=Array([40, 110, 240, 130]),
        )
    )
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([first, second])))
    page.obj["/Annots"] = Array([second, first])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    assert result["annotations_tagged"] == 2

    def reading_order(document: pikepdf.Pdf) -> list[tuple[str, str | None]]:
        page_div = next(
            item for item in _structure_objects(document) if item.get("/S") == Name.Div
        )
        order = []
        for child in _struct_children(page_div):
            role = str(child["/S"])
            field_name = None
            if child.get("/S") == Name.Form:
                objr = resolve_indirect(child["/K"])
                field_name = str(resolve_indirect(objr["/Obj"])["/T"])
            order.append((role, field_name))
        return order

    expected = [
        ("/P", None),
        ("/Form", "First field"),
        ("/P", None),
        ("/Form", "Second field"),
    ]
    assert reading_order(pdf) == expected

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        root_objgen = root.objgen
        assert reading_order(reopened) == expected

        preserved = ensure_logical_structure(reopened, semantic=True)

        assert preserved["structure_preserved"] is True
        assert preserved["structure_rebuilt"] is False
        assert preserved["semantic_repairs"] == 0
        assert reopened.Root["/StructTreeRoot"].objgen == root_objgen
        assert reading_order(reopened) == expected


def test_semantic_annotation_geometry_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"q Q", size=(400, 300))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            T=String("Malformed field"),
            Rect=Array([40, 210, 240]),
        )
    )
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([widget])))
    page.obj["/Annots"] = Array([widget])

    with pytest.raises(ConversionError, match=r"annotation /Rect is malformed"):
        ensure_logical_structure(pdf, semantic=True)


def test_semantic_annotation_grouping_counts_existing_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pdftopdfa.tagging._MAX_ARRAY_ITEMS", 2)
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"BT /F1 12 Tf "
            b"1 0 0 1 40 250 Tm (First label) Tj "
            b"1 0 0 1 40 150 Tm (Second label) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            T=String("First field"),
            Rect=Array([40, 210, 240, 230]),
        )
    )
    pdf.Root["/AcroForm"] = pdf.make_indirect(Dictionary(Fields=Array([widget])))
    page.obj["/Annots"] = Array([widget])

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["annotations_tagged"] == 1
    page_div = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Div
    )
    groups = _struct_children(page_div)
    assert len(groups) == 2
    assert all(group.get("/S") == Name.Part for group in groups)
    assert all(len(_struct_children(group)) <= 2 for group in groups)
    assert [
        str(child["/S"]) for group in groups for child in _struct_children(group)
    ] == ["/P", "/Form", "/P"]


def test_semantic_writer_emits_heading_paragraph_and_table_roles() -> None:
    pdf = pikepdf.Pdf.new()
    regular = _font(pdf)
    bold = _font(pdf, bold=True)
    content = b"\n".join(
        [
            b"BT",
            b"/FB 24 Tf 1 0 0 1 50 760 Tm (Quarterly report) Tj",
            b"/F1 10 Tf 1 0 0 1 50 700 Tm (Reliable body paragraph.) Tj",
            b"/FB 10 Tf 1 0 0 1 50 650 Tm (Name) Tj",
            b"1 0 0 1 220 650 Tm (Amount) Tj",
            b"/F1 10 Tf 1 0 0 1 50 632 Tm (Alpha) Tj",
            b"1 0 0 1 220 632 Tm (10) Tj",
            b"1 0 0 1 50 614 Tm (Beta) Tj",
            b"1 0 0 1 220 614 Tm (20) Tj",
            b"ET",
        ]
    )
    _page(
        pdf,
        content,
        Dictionary(Font=Dictionary(F1=regular, FB=bold)),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    roles = _roles(pdf)
    assert roles.count("/H1") == 1
    assert "/P" in roles
    assert roles.count("/Table") == 1
    assert roles.count("/TR") == 3
    assert roles.count("/TH") == 2
    assert roles.count("/TD") == 4
    document = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Document
    )
    page_div = _struct_children(document)[0]
    assert [str(child["/S"]) for child in _struct_children(page_div)] == [
        "/H1",
        "/P",
        "/Table",
    ]
    for header in (
        item for item in _structure_objects(pdf) if item.get("/S") == Name.TH
    ):
        attributes = resolve_indirect(header["/A"])
        entries = list(attributes) if isinstance(attributes, Array) else [attributes]
        table_attributes = next(
            entry
            for entry in entries
            if isinstance(entry, Dictionary) and entry.get("/O") == Name.Table
        )
        assert table_attributes["/Scope"] == Name.Column


def test_cross_page_paragraph_omits_pg_but_keeps_mcr_page_owners() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    resources = Dictionary(Font=Dictionary(F1=font))
    first_page = _page(
        pdf,
        b"BT /F1 12 Tf 1 0 0 1 50 40 Tm (A dependent clause,) Tj ET",
        resources,
    )
    second_page = _page(
        pdf,
        b"BT /F1 12 Tf 1 0 0 1 50 730 Tm (which continues on page two.) Tj ET",
        resources,
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    paragraph = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.P
    )
    assert "/Pg" not in paragraph
    mcrs = [item for item in _k_objects(paragraph) if item.get("/Type") == Name.MCR]
    assert len(mcrs) == 2
    assert [resolve_indirect(item["/Pg"]).objgen for item in mcrs] == [
        first_page.obj.objgen,
        second_page.obj.objgen,
    ]

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        paragraph = next(
            item for item in _structure_objects(reopened) if item.get("/S") == Name.P
        )
        assert "/Pg" not in paragraph
        mcrs = [item for item in _k_objects(paragraph) if item.get("/Type") == Name.MCR]
        assert [resolve_indirect(item["/Pg"]).objgen for item in mcrs] == [
            reopened.pages[0].obj.objgen,
            reopened.pages[1].obj.objgen,
        ]


def test_unambiguously_reversed_existing_reading_order_is_rebuilt() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/P <</MCID 0>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 72 700 Tm (Top paragraph) Tj ET EMC\n"
            b"/P <</MCID 1>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 72 100 Tm (Bottom paragraph) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    _install_two_paragraph_structure(pdf, page, (1, 0))

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is False
    assert result["structure_rebuilt"] is True
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    document = resolve_indirect(root["/K"])
    page_div = _struct_children(document)[0]
    paragraphs = [
        child for child in _struct_children(page_div) if child.get("/S") == Name.P
    ]
    assert [int(_k_objects(paragraph)[0]["/MCID"]) for paragraph in paragraphs] == [
        0,
        1,
    ]


def test_irrelevant_page_layout_budget_does_not_hide_reversed_reading_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pdftopdfa.digital_layout._MAX_DIGITAL_OPERATORS_PER_PAGE",
        20,
    )
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/P <</MCID 0>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 72 700 Tm (Top paragraph) Tj ET EMC\n"
            b"/P <</MCID 1>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 72 100 Tm (Bottom paragraph) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    _page(pdf, b"q Q\n" * 11)
    _install_two_paragraph_structure(pdf, page, (1, 0))

    content_references = {}
    elements = _existing_structure_elements(
        pdf,
        content_references_out=content_references,
    )
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    assert elements is not None
    assert isinstance(root, Dictionary)
    assert _has_unambiguous_existing_reading_order_inversion(
        pdf,
        root,
        elements,
        content_references,
    )

    monkeypatch.setattr(
        "pdftopdfa.digital_layout._MAX_DIGITAL_OPERATORS_PER_PAGE",
        10,
    )
    with pytest.raises(ConversionError, match=r"budget exceeded on page 1"):
        _has_unambiguous_existing_reading_order_inversion(
            pdf,
            root,
            elements,
            content_references,
        )
    result = ensure_logical_structure(pdf, semantic=True)
    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False


def test_multicolumn_existing_reading_order_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/P <</MCID 0>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 72 100 Tm (Left column end) Tj ET EMC\n"
            b"/P <</MCID 1>> BDC "
            b"BT /F1 12 Tf 1 0 0 1 360 700 Tm (Right column start) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    document, _paragraphs = _install_two_paragraph_structure(pdf, page, (0, 1))

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert [int(child["/K"]) for child in document["/K"]] == [0, 1]


def test_rich_existing_structure_is_repaired_without_rebuilding() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Figure <</MCID 0>> BDC q Q EMC\n/TH <</MCID 1>> BDC BT ET EMC",
    )
    page.obj["/StructParents"] = 0
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([20, 20, 120, 40]),
            StructParent=1,
            T=String("Customer name"),
        )
    )
    page.obj["/Annots"] = Array([widget])
    register_form_widget(pdf, widget)

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=page.obj,
            K=0,
            T=String("Sales chart"),
            Alt=Name("/BadAlt"),
            ActualText=String(""),
        )
    )
    table = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Table, P=document, Pg=page.obj)
    )
    row = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.TR, P=table, Pg=page.obj)
    )
    header = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.TH,
            P=row,
            Pg=page.obj,
            K=1,
            A=Array(
                [
                    Dictionary(O=Name.Layout, Placement=Name.Block),
                    Dictionary(
                        O=Name.Table,
                        Scope=Name("/Bogus"),
                        Headers=Array([String("missing-header-id")]),
                    ),
                ]
            ),
        )
    )
    form = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Form, P=document, Pg=page.obj)
    )
    form["/K"] = pdf.make_indirect(Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=widget))
    row["/K"] = header
    table["/K"] = row
    document["/K"] = Array([figure, table, form])
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([figure, header])
    parent_tree[1] = form
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 2
    pdf.Root["/StructTreeRoot"] = root
    original_root = root.objgen
    original_elements = [
        item.objgen for item in (document, figure, table, row, header, form)
    ]
    original_content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert result["semantic_repairs"] == 6
    assert result["semantic_alternatives_review_required"] == 1
    assert pdf.Root["/StructTreeRoot"].objgen == original_root
    assert [item.objgen for item in (document, figure, table, row, header, form)] == (
        original_elements
    )
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )
    assert "/Alt" not in figure
    assert "/ActualText" not in figure
    attributes = resolve_indirect(header["/A"])
    assert isinstance(attributes, Array)
    assert resolve_indirect(attributes[0]).get("/O") == Name.Layout
    table_attributes = next(
        resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Table
        and "/Scope" in resolve_indirect(attribute)
    )
    assert table_attributes.get("/O") == Name.Table
    assert table_attributes.get("/Scope") == Name.Column
    assert all(
        "/Headers" not in resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Table
    )
    assert str(widget["/TU"]) == "Customer name"

    second = ensure_logical_structure(pdf, semantic=True)

    assert second["structure_preserved"] is True
    assert second["semantic_repairs"] == 0
    assert second["semantic_alternatives_review_required"] == 1
    assert pdf.Root["/StructTreeRoot"].objgen == original_root


def test_existing_trustworthy_alternatives_and_caption_need_no_review() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/Span <</MCID 0>> BDC q Q EMC "
            b"/Span <</MCID 1>> BDC q Q EMC "
            b"/Span <</MCID 2>> BDC "
            b"BT /F1 10 Tf (Quarterly revenue) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    described_figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=page.obj,
            K=0,
            Alt=String("Revenue chart by quarter"),
        )
    )
    described_formula = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Formula,
            P=document,
            Pg=page.obj,
            K=1,
            ActualText=String("x squared plus y squared"),
        )
    )
    captioned_figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    caption = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Caption,
            P=captioned_figure,
            Pg=page.obj,
            K=2,
        )
    )
    captioned_figure["/K"] = caption
    empty_template = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    document["/K"] = Array(
        [described_figure, described_formula, captioned_figure, empty_template]
    )
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([described_figure, described_formula, caption])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    first = ensure_logical_structure(pdf, semantic=True)
    second = ensure_logical_structure(pdf, semantic=True)

    assert first["structure_preserved"] is True
    assert first["semantic_repairs"] == 0
    assert first["semantic_alternatives_review_required"] == 0
    assert second["structure_preserved"] is True
    assert second["semantic_repairs"] == 0
    assert second["semantic_alternatives_review_required"] == 0
    assert str(described_figure["/Alt"]) == "Revenue chart by quarter"
    assert str(described_formula["/ActualText"]) == "x squared plus y squared"
    assert "/Alt" not in captioned_figure
    assert "/Alt" not in empty_template


def test_existing_empty_caption_still_requires_alternative_review() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"/Span <</MCID 0>> BDC q Q EMC")
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    caption = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Caption,
            P=figure,
            Pg=page.obj,
            K=0,
        )
    )
    figure["/K"] = caption
    document["/K"] = figure
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([caption])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    first = ensure_logical_structure(pdf, semantic=True)
    second = ensure_logical_structure(pdf, semantic=True)

    assert first["structure_preserved"] is True
    assert first["semantic_alternatives_review_required"] == 1
    assert second["structure_preserved"] is True
    assert second["semantic_alternatives_review_required"] == 1


def test_pdfua_preserved_figure_does_not_get_fallback_alt() -> None:
    pdf = pikepdf.Pdf.new()
    pdf.Root["/Lang"] = String("de")
    page = _page(pdf, b"/Span <</MCID 0>> BDC q Q EMC")
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=page.obj,
            K=0,
        )
    )
    document["/K"] = figure
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([figure])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf, semantic=True, pdfua=True)

    assert result["structure_preserved"] is True
    assert result["semantic_alternatives_review_required"] == 1
    assert "/Alt" not in figure


@pytest.mark.parametrize(
    "role",
    [Name.Figure, Name.Formula],
)
def test_pdfua_preserved_element_keeps_language_without_fallback_alt(
    role: Name,
) -> None:
    pdf = pikepdf.Pdf.new()
    pdf.Root["/Lang"] = String("fr")
    page = _page(pdf, b"/Span <</MCID 0>> BDC q Q EMC")
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    element = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=role,
            P=document,
            Pg=page.obj,
            K=0,
            Lang=String("fr"),
        )
    )
    document["/K"] = element
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([element])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf, semantic=True, pdfua=True)

    assert result["structure_preserved"] is True
    assert result["semantic_alternatives_review_required"] == 1
    assert "/Alt" not in element
    assert str(element["/Lang"]) == "fr"


def test_existing_figure_does_not_combine_distinct_inner_actualtext_values() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/Figure <</MCID 0>> BDC\n"
            b"/Span <</ActualText (first fragment)>> BDC "
            b"BT /F1 10 Tf 20 40 Td (x) Tj ET EMC\n"
            b"/Span <</ActualText (second fragment)>> BDC "
            b"BT /F1 10 Tf 30 40 Td (y) Tj ET EMC\n"
            b"EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=page.obj,
            K=0,
        )
    )
    document["/K"] = figure
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([figure])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    first = ensure_logical_structure(pdf, semantic=True)
    second = ensure_logical_structure(pdf, semantic=True)

    assert first["structure_preserved"] is True
    assert first["semantic_repairs"] == 0
    assert first["semantic_alternatives_review_required"] == 0
    assert second["structure_preserved"] is True
    assert second["semantic_alternatives_review_required"] == 0
    assert "/ActualText" not in figure
    assert "/Alt" not in figure


@pytest.mark.parametrize(("textual", "expected_review"), [(False, 1), (True, 0)])
def test_existing_form_caption_requires_real_description_evidence(
    textual: bool,
    expected_review: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = pdf.make_stream(
        b"/Span <</MCID 0>> BDC "
        + (b"BT /F1 10 Tf (Revenue by quarter) Tj ET" if textual else b"q Q")
        + b" EMC"
    )
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 200, 100])
    form["/Resources"] = Dictionary(Font=Dictionary(F1=font))
    form["/StructParents"] = 0
    page = _page(
        pdf,
        b"q /Fm0 Do Q",
        Dictionary(XObject=Dictionary(Fm0=form)),
        size=(200, 100),
    )
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    caption = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Caption,
            P=figure,
            Pg=page.obj,
            K=pdf.make_indirect(
                Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)
            ),
        )
    )
    figure["/K"] = caption
    document["/K"] = figure
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([caption])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    first = ensure_logical_structure(pdf, semantic=True)
    second = ensure_logical_structure(pdf, semantic=True)

    assert first["structure_preserved"] is True
    assert first["semantic_alternatives_review_required"] == expected_review
    assert second["structure_preserved"] is True
    assert second["semantic_alternatives_review_required"] == expected_review


@pytest.mark.parametrize("named_properties", [False, True])
@pytest.mark.parametrize("form_invocation", [False, True])
def test_source_actualtext_describes_generated_figure_and_survives_preserve(
    named_properties: bool,
    form_invocation: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    description = "A red status indicator"
    if form_invocation:
        form = pdf.make_stream(b"q 20 0 0 20 40 40 cm /Im0 Do Q")
        form["/Type"] = Name.XObject
        form["/Subtype"] = Name.Form
        form["/BBox"] = Array([0, 0, 200, 100])
        form["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        resources = Dictionary(XObject=Dictionary(Fm0=form))
        paint = b" q /Fm0 Do Q EMC"
    else:
        resources = Dictionary(XObject=Dictionary(Im0=image))
        paint = b" q 20 0 0 20 40 40 cm /Im0 Do Q EMC"
    if named_properties:
        resources["/Properties"] = Dictionary(
            Desc=Dictionary(ActualText=String(description))
        )
        opener = b"/Figure /Desc BDC"
    else:
        opener = b"/Figure <</ActualText (A red status indicator)>> BDC"
    page = _page(
        pdf,
        opener + paint,
        resources,
        size=(200, 100),
    )

    first = ensure_logical_structure(pdf, semantic=True)

    figures = [
        element
        for element in _structure_objects(pdf)
        if element.get("/S") == Name.Figure
    ]
    assert first["semantic_structure_generated"] is True
    assert first["semantic_alternatives_review_required"] == 0
    assert len(figures) == 1
    assert str(figures[0]["/ActualText"]) == description
    assert any(
        isinstance(instruction, pikepdf.ContentStreamInstruction)
        and instruction.operator == pikepdf.Operator("BDC")
        and resolve_indirect(instruction.operands[0]) == Name.Figure
        for instruction in pikepdf.parse_content_stream(page)
    )

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        second = ensure_logical_structure(reopened, semantic=True)
        reopened_figures = [
            element
            for element in _structure_objects(reopened)
            if element.get("/S") == Name.Figure
        ]
        assert second["structure_preserved"] is True
        assert second["semantic_alternatives_review_required"] == 0
        assert len(reopened_figures) == 1
        assert str(reopened_figures[0]["/ActualText"]) == description


def test_reused_resourceless_form_resolves_named_actualtext_per_page() -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    form = pdf.make_stream(b"/Figure /Desc BDC q 20 0 0 20 40 40 cm /Im0 Do Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 200, 100])

    for description in ("First page diagram", "Second page diagram"):
        resources = Dictionary(
            XObject=Dictionary(Fm0=form, Im0=image),
            Properties=Dictionary(Desc=Dictionary(ActualText=String(description))),
        )
        _page(pdf, b"q /Fm0 Do Q", resources, size=(200, 100))

    result = ensure_logical_structure(pdf, semantic=True)

    figures = [
        element
        for element in _structure_objects(pdf)
        if element.get("/S") == Name.Figure
    ]
    assert result["semantic_structure_generated"] is True
    assert result["semantic_alternatives_review_required"] == 0
    assert [str(figure["/ActualText"]) for figure in figures] == [
        "First page diagram",
        "Second page diagram",
    ]


def test_missing_table_header_scopes_follow_row_and_column_position() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b" ".join(
            f"/TH <</MCID {mcid}>> BDC q Q EMC".encode("ascii") for mcid in range(4)
        ),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    table = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Table, P=document, Pg=page.obj)
    )
    rows: list[Dictionary] = []
    headers: list[Dictionary] = []
    for row_index in range(2):
        row = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=Name.TR, P=table, Pg=page.obj)
        )
        row_headers = []
        for column_index in range(2):
            mcid = row_index * 2 + column_index
            header = pdf.make_indirect(
                Dictionary(
                    Type=Name.StructElem,
                    S=Name.TH,
                    P=row,
                    Pg=page.obj,
                    K=mcid,
                )
            )
            row_headers.append(header)
            headers.append(header)
        row["/K"] = Array(row_headers)
        rows.append(row)
    table["/K"] = Array(rows)
    document["/K"] = table
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array(headers)
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["semantic_repairs"] == 4
    scopes = []
    for header in headers:
        attributes = resolve_indirect(header["/A"])
        assert isinstance(attributes, Dictionary)
        scopes.append(attributes["/Scope"])
    assert scopes == [Name.Both, Name.Column, Name.Row, Name.Column]

    second = ensure_logical_structure(pdf, semantic=True)
    assert second["semantic_repairs"] == 0


@pytest.mark.parametrize(
    "role_chain",
    [
        (Name.TD,),
        (Name.Table, Name.P),
        (Name.L, Name.P),
        (Name.L, Name.LI),
        (Name.L, Name.LI, Name.P),
    ],
)
def test_invalid_existing_role_hierarchy_is_rebuilt(
    role_chain: tuple[Name, ...],
) -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC BT /F1 12 Tf 40 220 Td (Body text.) Tj ET EMC",
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    parent = document
    elements: list[Dictionary] = []
    for role in role_chain:
        element = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=role, P=parent, Pg=page.obj)
        )
        parent["/K"] = element
        elements.append(element)
        parent = element
    parent["/K"] = 0
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([parent])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root
    original_root = root.objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_rebuilt"] is True
    assert result["semantic_structure_generated"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != original_root
    assert "/P" in _roles(pdf)


def test_generic_tagging_is_upgraded_to_semantic_roles() -> None:
    pdf = pikepdf.Pdf.new()
    regular = _font(pdf)
    bold = _font(pdf, bold=True)
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([35, 215, 170, 240]),
        )
    )
    page = _page(
        pdf,
        b"BT /FB 22 Tf 40 250 Td (Document title) Tj "
        b"/F1 12 Tf 0 -30 Td (Open documentation) Tj ET",
        Dictionary(Font=Dictionary(F1=regular, FB=bold)),
        size=(400, 300),
    )
    page.obj["/Annots"] = Array([annotation])

    first = ensure_logical_structure(pdf, semantic=False)
    first_root = pdf.Root["/StructTreeRoot"].objgen
    assert first["semantic_structure_generated"] is False
    assert "/Link" in _roles(pdf)

    second = ensure_logical_structure(pdf, semantic=True)

    assert second["structure_rebuilt"] is True
    assert second["semantic_structure_generated"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != first_root
    assert {"/H1", "/P", "/Link"} <= set(_roles(pdf))


def test_custom_role_mapped_to_div_is_not_treated_as_semantic_content() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        b"/MyBlock <</MCID 0>> BDC BT /F1 12 Tf 40 220 Td (Body text.) Tj ET EMC",
        Dictionary(Font=Dictionary(F1=font)),
        size=(400, 300),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(
        Dictionary(Type=Name.StructTreeRoot, RoleMap=Dictionary(MyBlock=Name.Div))
    )
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    custom = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name("/MyBlock"),
            P=document,
            Pg=page.obj,
            K=0,
        )
    )
    document["/K"] = custom
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([custom])
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_rebuilt"] is True
    assert result["semantic_structure_generated"] is True
    assert "/MyBlock" not in _roles(pdf)
    assert "/P" in _roles(pdf)


def test_existing_heading_sequence_and_annotation_roles_are_repaired() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/H2 <</MCID 0>> BDC q Q EMC\n/H4 <</MCID 1>> BDC q Q EMC",
    )
    page.obj["/StructParents"] = 0
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([20, 20, 120, 40]),
            StructParent=1,
            TU=String("Customer name"),
        )
    )
    link_annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([20, 60, 120, 80]),
            StructParent=2,
        )
    )
    page.obj["/Annots"] = Array([widget, link_annotation])
    register_form_widget(pdf, widget)

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    first_heading = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H2,
            P=document,
            Pg=page.obj,
            K=0,
        )
    )
    second_heading = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H4,
            P=document,
            Pg=page.obj,
            K=1,
        )
    )
    widget_reference = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=widget)
    )
    paragraph = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.P, P=document, Pg=page.obj)
    )
    link_reference = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=link_annotation)
    )
    paragraph["/K"] = link_reference
    document["/K"] = Array([first_heading, second_heading, widget_reference, paragraph])
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([first_heading, second_heading])
    parent_tree[1] = document
    parent_tree[2] = paragraph
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 3
    pdf.Root["/StructTreeRoot"] = root
    original_root = root.objgen
    original_content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["semantic_repairs"] == 4
    assert first_heading["/S"] == Name.H1
    assert second_heading["/S"] == Name.H2
    form = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Form)
    link = next(item for item in _structure_objects(pdf) if item.get("/S") == Name.Link)
    assert resolve_indirect(form["/P"]).objgen == document.objgen
    assert resolve_indirect(link["/P"]).objgen == paragraph.objgen
    assert resolve_indirect(form["/K"]).objgen == widget_reference.objgen
    assert resolve_indirect(link["/K"]).objgen == link_reference.objgen
    repaired_tree = NumberTree(root["/ParentTree"])
    assert resolve_indirect(repaired_tree[1]).objgen == form.objgen
    assert resolve_indirect(repaired_tree[2]).objgen == link.objgen
    assert pdf.Root["/StructTreeRoot"].objgen == original_root
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )

    second = ensure_logical_structure(pdf, semantic=True)

    assert second["structure_preserved"] is True
    assert second["semantic_repairs"] == 0


def test_repeated_page_content_and_paths_become_typed_artifacts() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    for page_number in range(1, 4):
        _page(
            pdf,
            b"\n".join(
                [
                    b"BT /F1 10 Tf",
                    b"1 0 0 1 40 770 Tm (Annual report) Tj",
                    f"1 0 0 1 50 650 Tm (Body page {page_number}.) Tj".encode("ascii"),
                    b"1 0 0 1 40 20 Tm (Internal use) Tj",
                    f"1 0 0 1 295 20 Tm ({page_number}) Tj ET".encode("ascii"),
                    b"0 0 m 10 10 l S",
                ]
            ),
            Dictionary(Font=Dictionary(F1=font)),
        )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_structure_generated"] is True
    assert result["semantic_content_items"] == 3
    assert result["artifacts_tagged"] == 12
    artifact_types = [
        (type_name, subtype)
        for page in pdf.pages
        for tag, type_name, subtype, _mcid in _marked_content(page)
        if tag == "/Artifact"
    ]
    assert artifact_types.count(("/Pagination", "/Header")) == 3
    assert artifact_types.count(("/Pagination", "/Footer")) == 6
    assert artifact_types.count(("/Layout", None)) == 3
    assert all("/Tabs" not in page.obj for page in pdf.pages)


def test_generated_path_artifact_marker_starts_before_path_object() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"q 0 0 m 10 10 l S Q", size=(100, 100))

    ensure_logical_structure(pdf, semantic=True)

    operators = [
        str(instruction.operator)
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
    ]
    assert operators == ["q", "BDC", "m", "l", "S", "EMC", "Q"]


def test_ocr_form_mcid_is_referenced_by_mcr_and_preserves_unicode() -> None:
    pdf, _form, manifest = _ocr_document()

    result = ensure_logical_structure(
        pdf,
        semantic=True,
        ocr_manifest=manifest,
    )

    assert result["semantic_structure_generated"] is True
    assert result["semantic_content_items"] == 1
    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        page = reopened.pages[0]
        xobjects = resolve_indirect(page.obj["/Resources"]["/XObject"])
        form = resolve_indirect(xobjects["/OCR-0"])
        assert isinstance(form, pikepdf.Stream)
        assert "/StructParents" in form
        assert "/StructParents" not in page.obj
        assert "/Tabs" not in page.obj
        assert str(reopened.Root["/Lang"]) == "de-DE"

        mcr = next(
            item
            for item in _structure_objects(reopened)
            if item.get("/Type") == Name.MCR
        )
        assert int(mcr["/MCID"]) == 7
        assert resolve_indirect(mcr["/Stm"]).objgen == form.objgen
        assert resolve_indirect(mcr["/Pg"]).objgen == page.obj.objgen
        assert any(marker[3] == 7 for marker in _marked_content(form))

        actual_text = {
            str(item["/ActualText"])
            for item in _structure_objects(reopened)
            if "/ActualText" in item
        }
        assert "Förderung, Größe und Rückgabe" in actual_text
        form_key = int(form["/StructParents"])
        root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        parent_array = resolve_indirect(NumberTree(root["/ParentTree"])[form_key])
        assert isinstance(parent_array, Array)
        owner = resolve_indirect(parent_array[7])
        assert isinstance(owner, Dictionary)
        mcr_owner = next(
            item
            for item in _structure_objects(reopened)
            if item.get("/Type") == Name.StructElem
            and any(child.get("/Type") == Name.MCR for child in _k_objects(item))
        )
        assert owner.objgen == mcr_owner.objgen


def test_ocr_page_keeps_native_text_and_xobjects_after_target_form() -> None:
    pdf, form, manifest = _ocr_document()
    page = pdf.pages[0]
    resources = resolve_indirect(page.obj["/Resources"])
    assert isinstance(resources, Dictionary)
    xobjects = resolve_indirect(resources["/XObject"])
    assert isinstance(xobjects, Dictionary)
    scan = pdf.make_stream(b"\xff\xff\xff")
    logo = pdf.make_stream(b"\x00\x00\x00")
    for image in (scan, logo):
        image["/Type"] = Name.XObject
        image["/Subtype"] = Name.Image
        image["/Width"] = 1
        image["/Height"] = 1
        image["/ColorSpace"] = Name.DeviceRGB
        image["/BitsPerComponent"] = 8
    xobjects["/Scan"] = scan
    xobjects["/Logo"] = logo
    resources["/Font"] = Dictionary(F2=_font(pdf, bold=True))
    content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(content, pikepdf.Stream)
    content.write(
        b"q 400 0 0 300 0 0 cm /Scan Do Q\n"
        b"BT /F2 18 Tf 30 230 Td (Native heading) Tj ET\n"
        b"q /OCR-0 Do Q\n"
        b"q 24 0 0 24 350 250 cm /Logo Do Q"
    )
    form.write(
        bytes(form.read_bytes())
        + b"\n/Span <</MCID 8>> BDC "
        + b"BT /F1 10 Tf 3 Tr 1 0 0 1 30 230 Tm (duplicate) Tj ET EMC"
    )
    raw_page = manifest["pages"][0]
    raw_line = raw_page["lines"][0]
    raw_line["text"] = "Native headingSCANNED BODY 123"
    raw_line["bbox"] = {
        "left": 30,
        "top": 50,
        "right": 330,
        "bottom": 80,
    }
    raw_line["words"] = [
        {
            "text": "Native",
            "bbox": {"left": 30, "top": 50, "right": 82, "bottom": 80},
        },
        {
            "text": "headingSCANNED",
            "bbox": {"left": 82, "top": 50, "right": 220, "bottom": 80},
        },
        {
            "text": "BODY",
            "bbox": {"left": 225, "top": 50, "right": 270, "bottom": 80},
        },
        {
            "text": "123",
            "bbox": {"left": 275, "top": 50, "right": 310, "bottom": 80},
        },
    ]
    raw_page["lines"].append(
        {
            "mcid": 8,
            "text": "Native heading",
            "confidence": 0.99,
            "bbox": {"left": 30, "top": 50, "right": 190, "bottom": 80},
            "words": [
                {
                    "text": "Native",
                    "bbox": {
                        "left": 30,
                        "top": 50,
                        "right": 82,
                        "bottom": 80,
                    },
                },
                {
                    "text": "heading",
                    "bbox": {
                        "left": 82,
                        "top": 50,
                        "right": 160,
                        "bottom": 80,
                    },
                },
            ],
        }
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_content_items"] == 3
    assert result["artifacts_tagged"] == 2
    page_markers = _marked_content(page)
    assert [marker[3] for marker in page_markers if marker[0] == "/Span"] == [0, 1]
    assert any(marker[0] == "/Artifact" for marker in page_markers)
    form_markers = _marked_content(form)
    assert form_markers[0][3] == 7
    assert form_markers[1] == ("/Artifact", "/Layout", None, None)
    form_marker = next(
        instruction
        for instruction in pikepdf.parse_content_stream(form)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and str(instruction.operator) == "BDC"
        and resolve_indirect(instruction.operands[0]) != Name.Artifact
    )
    form_properties = resolve_indirect(form_marker.operands[1])
    assert str(form_properties["/ActualText"]) == "SCANNED BODY 123"
    assert "/StructParents" in page.obj
    assert "/StructParents" in form
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    page_parent_array = resolve_indirect(parent_tree[int(page.obj["/StructParents"])])
    form_parent_array = resolve_indirect(parent_tree[int(form["/StructParents"])])
    assert isinstance(page_parent_array, Array)
    assert isinstance(form_parent_array, Array)
    assert len(page_parent_array) == 2
    assert isinstance(resolve_indirect(form_parent_array[7]), Dictionary)

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_page = reopened.pages[0]
        reopened_xobjects = resolve_indirect(
            reopened_page.obj["/Resources"]["/XObject"]
        )
        reopened_form = resolve_indirect(reopened_xobjects["/OCR-0"])
        reopened_tree = NumberTree(reopened.Root["/StructTreeRoot"]["/ParentTree"])
        assert isinstance(
            resolve_indirect(reopened_tree[int(reopened_page.obj["/StructParents"])]),
            Array,
        )
        assert isinstance(
            resolve_indirect(reopened_tree[int(reopened_form["/StructParents"])]),
            Array,
        )


def test_ocr_native_dedup_accepts_strong_fuzzy_text_only_with_tight_overlap() -> None:
    evidence = (((15.0, 10.0, 63.5, 20.0), frozenset({"nativehdr"})),)

    assert (
        _remove_native_text_from_ocr_word(
            "NATVEEHRR",
            (15.0, 10.0, 63.5, 20.0),
            evidence,
        )
        is None
    )
    assert _remove_native_text_from_ocr_word(
        "NATVEEHRR",
        (55.0, 10.0, 103.5, 20.0),
        evidence,
    ) == ("NATVEEHRR", (55.0, 10.0, 103.5, 20.0), False)


def test_ocr_native_dedup_uses_rotated_crop_and_userunit_geometry() -> None:
    pdf = pikepdf.Pdf.new()
    native_font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
    )
    ocr_font = _font(pdf)
    form = _form(
        pdf,
        b"/Span <</MCID 0>> BDC "
        b"BT /F2 8 Tf 3 Tr 10 80 Td (duplicate) Tj ET EMC\n"
        b"/Span <</MCID 1>> BDC "
        b"BT /F2 8 Tf 3 Tr 10 40 Td (body) Tj ET EMC",
        Dictionary(Font=Dictionary(F2=ocr_font)),
        bbox=(0, 0, 120, 100),
    )
    scan = pdf.make_stream(b"\xff\xff\xff")
    scan["/Type"] = Name.XObject
    scan["/Subtype"] = Name.Image
    scan["/Width"] = 1
    scan["/Height"] = 1
    scan["/ColorSpace"] = Name.DeviceRGB
    scan["/BitsPerComponent"] = 8
    page = _page(
        pdf,
        b"q 0 1.1988 -1.19904 0 100 0 cm /OCR-0 Do Q\n"
        b"q 80 0 0 100 10 10 cm /Scan Do Q\n"
        b"BT /F1 8 Tf 0 Tr 15 100 Td (NATIVE) Tj ET",
        Dictionary(
            Font=Dictionary(F1=native_font),
            XObject=Dictionary({"/OCR-0": form, "/Scan": scan}),
        ),
        size=(100, 120),
    )
    page.obj["/CropBox"] = Array([10, 10, 90, 110])
    page.obj["/Rotate"] = 90
    page.obj["/UserUnit"] = 2
    manifest = {
        "schema_version": 1,
        "type": "pdftopdfa-ocr-document",
        "page_count": 1,
        "languages": ["en"],
        "pages": [
            {
                "page_index": 0,
                "form_name": "/OCR-0",
                "coordinates": {"width": 500.5, "height": 417},
                "lines": [
                    {
                        "mcid": 0,
                        "text": "NATIVE",
                        "confidence": 0.99,
                        "bbox": {
                            "left": 411.5,
                            "top": 59,
                            "right": 445,
                            "bottom": 185.5,
                        },
                        "words": [
                            {
                                "text": "NATIVE",
                                "bbox": {
                                    "left": 411.5,
                                    "top": 75.5,
                                    "right": 445,
                                    "bottom": 176.5,
                                },
                            }
                        ],
                    },
                    {
                        "mcid": 1,
                        "text": "CROP SCAN BODY 456",
                        "confidence": 0.99,
                        "bbox": {
                            "left": 259,
                            "top": 70,
                            "right": 282.5,
                            "bottom": 322.5,
                        },
                        "words": [
                            {
                                "text": "CROP",
                                "bbox": {
                                    "left": 259,
                                    "top": 81.5,
                                    "right": 282.5,
                                    "bottom": 133,
                                },
                            },
                            {
                                "text": "SCAN",
                                "bbox": {
                                    "left": 259,
                                    "top": 152.5,
                                    "right": 282.5,
                                    "bottom": 200,
                                },
                            },
                            {
                                "text": "BODY",
                                "bbox": {
                                    "left": 259,
                                    "top": 219.5,
                                    "right": 282.5,
                                    "bottom": 271,
                                },
                            },
                            {
                                "text": "456",
                                "bbox": {
                                    "left": 259,
                                    "top": 290.5,
                                    "right": 282.5,
                                    "bottom": 314.5,
                                },
                            },
                        ],
                    },
                ],
            }
        ],
    }

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_content_items"] == 2
    assert result["artifacts_tagged"] == 2
    assert [marker[0] for marker in _marked_content(page)] == [
        "/Artifact",
        "/Span",
    ]
    assert [marker[0] for marker in _marked_content(form)] == [
        "/Artifact",
        "/Span",
    ]
    assert "/StructParents" in page.obj
    assert "/StructParents" in form


def test_ocr_manifest_line_order_is_authoritative_over_geometry() -> None:
    pdf, form, manifest = _ocr_document()
    form.write(
        b"/Span <</MCID 0>> BDC "
        b"BT /F1 10 Tf 3 Tr 1 0 0 1 40 40 Tm (bottom) Tj ET EMC\n"
        b"/Span <</MCID 1>> BDC "
        b"BT /F1 10 Tf 3 Tr 1 0 0 1 40 250 Tm (top) Tj ET EMC"
    )
    pages = manifest["pages"]
    assert isinstance(pages, list)
    pages[0]["lines"] = [
        {
            "mcid": 0,
            "text": "Bottom first.",
            "confidence": 0.99,
            "bbox": {"left": 40, "top": 240, "right": 160, "bottom": 260},
        },
        {
            "mcid": 1,
            "text": "Top second.",
            "confidence": 0.99,
            "bbox": {"left": 40, "top": 30, "right": 160, "bottom": 50},
        },
    ]

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_structure_generated"] is True
    document = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Document
    )
    page_div = _struct_children(document)[0]
    paragraphs = [
        child for child in _struct_children(page_div) if child.get("/S") == Name.P
    ]
    assert len(paragraphs) == 2
    assert [int(_k_objects(paragraph)[0]["/MCID"]) for paragraph in paragraphs] == [
        0,
        1,
    ]


def test_ocr_form_mcids_exactly_match_manifest() -> None:
    pdf, form, manifest = _ocr_document()

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_content_items"] == 1
    assert _marked_content(form) == [("/Span", None, None, 7)]
    parent_tree = NumberTree(pdf.Root["/StructTreeRoot"]["/ParentTree"])
    owners = resolve_indirect(parent_tree[int(form["/StructParents"])])
    assert isinstance(owners, Array)
    assert isinstance(resolve_indirect(owners[7]), Dictionary)


def test_ocr_form_mcid_match_ignores_valid_source_artifacts() -> None:
    pdf, form, manifest = _ocr_document()
    form.write(
        bytes(form.read_bytes())
        + b"\n/Artifact <</Type /Layout>> BDC 10 10 m 20 20 l S EMC"
    )

    result = ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert result["semantic_content_items"] == 1
    assert _marked_content(form) == [
        ("/Span", None, None, 7),
        ("/Artifact", "/Layout", None, None),
    ]


@pytest.mark.parametrize("mismatch", ["missing", "extra"])
def test_ocr_form_mcid_mismatch_fails_atomically(mismatch: str) -> None:
    pdf, form, manifest = _ocr_document()
    if mismatch == "missing":
        form.write(bytes(form.read_bytes()).replace(b"/MCID 7", b"/Lang /en"))
    else:
        form.write(
            bytes(form.read_bytes())
            + b"\n/Span <</MCID 8>> BDC "
            + b"BT /F1 10 Tf 3 Tr 1 0 0 1 40 180 Tm (extra) Tj ET EMC"
        )
    page = pdf.pages[0]
    page_content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(page_content, pikepdf.Stream)
    original_form = bytes(form.read_bytes())
    original_page_content = bytes(page_content.read_bytes())
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_form_keys = set(form.keys())
    original_object_count = len(pdf.objects)

    detail = (
        r"manifest lines without marked content: \[7\]"
        if mismatch == "missing"
        else r"marked content without manifest line: \[8\]"
    )
    with pytest.raises(
        ConversionError,
        match=f"OCR Form MCIDs do not match manifest on page 1 .*{detail}",
    ):
        ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert bytes(form.read_bytes()) == original_form
    assert bytes(page_content.read_bytes()) == original_page_content
    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert set(form.keys()) == original_form_keys
    assert len(pdf.objects) == original_object_count


def test_digital_extraction_failure_leaves_no_partial_structure() -> None:
    pdf = pikepdf.Pdf.new()
    malformed_form = pdf.make_stream(b"BT ET")
    malformed_form["/Type"] = Name.XObject
    malformed_form["/Subtype"] = Name.Form
    malformed_form["/Resources"] = Dictionary()
    page = _page(
        pdf,
        b"q /BrokenForm Do Q",
        Dictionary(XObject=Dictionary(BrokenForm=malformed_form)),
    )
    original_content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(
        ConversionError,
        match="Cannot create semantic digital PDF structure",
    ):
        ensure_logical_structure(pdf, semantic=True)

    assert "/StructTreeRoot" not in pdf.Root
    assert "/StructParents" not in page.obj
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )


@pytest.mark.parametrize("failure", ["schema", "form"])
def test_malformed_ocr_manifest_fails_closed(failure: str) -> None:
    pdf, form, valid_manifest = _ocr_document()
    manifest = deepcopy(valid_manifest)
    if failure == "schema":
        manifest["schema_version"] = 2
    else:
        pages = manifest["pages"]
        assert isinstance(pages, list)
        pages[0]["form_name"] = "/OCR-missing"
    page = pdf.pages[0]
    original_form = bytes(form.read_bytes())
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_form_keys = set(form.keys())
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="Cannot create semantic OCR structure"):
        ensure_logical_structure(pdf, semantic=True, ocr_manifest=manifest)

    assert "/StructTreeRoot" not in pdf.Root
    assert bytes(form.read_bytes()) == original_form
    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert set(form.keys()) == original_form_keys
    assert len(pdf.objects) == original_object_count


def test_semantic_rebuild_preserves_oc_actualtext_points_and_named_properties() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer A")))
    legacy = Dictionary(
        MCID=7,
        ActualText=String("replacement"),
        Lang=String("de-DE"),
    )
    pdf.Root["/OCProperties"] = Dictionary(
        OCGs=Array([ocg]),
        D=Dictionary(Order=Array([ocg]), ON=Array([ocg])),
    )
    page = _page(
        pdf,
        b"",
        Dictionary(
            Font=Dictionary(F1=font),
            Properties=Dictionary(OC1=ocg, Legacy=legacy),
        ),
        size=(300, 300),
    )
    page.obj["/Contents"] = Array(
        [
            pdf.make_stream(
                b"/OC /OC1 BDC\n/P /Legacy BDC\n"
                b"BT /F1 12 Tf 30 200 Td (painted) Tj ET\n"
            ),
            pdf.make_stream(
                b"EMC\nEMC\n/TouchUp_TextEdit MP\n"
                b"/Point <</MCID 9 /ActualText (point)>> DP"
            ),
        ]
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["mcids_removed"] == 2
    assert "/MCID" not in legacy
    assert str(legacy["/ActualText"]) == "replacement"
    instructions = [
        instruction
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
    ]
    oc_marker = next(
        instruction
        for instruction in instructions
        if instruction.operator == pikepdf.Operator("BDC")
        and instruction.operands[0] == Name("/OC")
    )
    assert oc_marker.operands[1] == Name("/OC1")
    legacy_marker = next(
        instruction
        for instruction in instructions
        if instruction.operator == pikepdf.Operator("BDC")
        and instruction.operands[0] == Name.P
    )
    assert legacy_marker.operands[1] == Name("/Legacy")
    assert any(
        instruction.operator == pikepdf.Operator("MP")
        and instruction.operands[0] == Name("/TouchUp_TextEdit")
        for instruction in instructions
    )
    point = next(
        instruction
        for instruction in instructions
        if instruction.operator == pikepdf.Operator("DP")
    )
    point_properties = resolve_indirect(point.operands[1])
    assert isinstance(point_properties, Dictionary)
    assert "/MCID" not in point_properties
    assert str(point_properties["/ActualText"]) == "point"
    assert [marker[3] for marker in _marked_content(page) if marker[3] is not None] == [
        0
    ]

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_page = reopened.pages[0]
        reopened_resources = resolve_indirect(reopened_page.obj["/Resources"])
        reopened_legacy = resolve_indirect(reopened_resources["/Properties"]["/Legacy"])
        assert "/MCID" not in reopened_legacy
        assert str(reopened_legacy["/ActualText"]) == "replacement"
        root = resolve_indirect(reopened.Root["/StructTreeRoot"])
        page_key = int(reopened_page.obj["/StructParents"])
        owners = resolve_indirect(NumberTree(root["/ParentTree"])[page_key])
        assert isinstance(owners, Array)
        assert len(owners) == 1


def test_semantic_rebuild_preserves_nested_cross_stream_scopes() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        b"",
        Dictionary(Font=Dictionary(F1=font)),
        size=(300, 300),
    )
    page.obj["/Contents"] = Array(
        [
            pdf.make_stream(
                b"/ReversedChars BMC\n"
                b"/Span <</ActualText (both)>> BDC\n"
                b"BT /F1 12 Tf 30 200 Td (one) Tj\n"
            ),
            pdf.make_stream(b"0 -20 Td (two) Tj ET\nEMC\nEMC"),
        ]
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    stack: list[tuple[str, Dictionary | None]] = []
    text_show_count = 0
    for instruction in pikepdf.parse_content_stream(page):
        assert isinstance(instruction, pikepdf.ContentStreamInstruction)
        if instruction.operator in {pikepdf.Operator("BMC"), pikepdf.Operator("BDC")}:
            properties = (
                resolve_indirect(instruction.operands[1])
                if instruction.operator == pikepdf.Operator("BDC")
                else None
            )
            stack.append((str(instruction.operands[0]), properties))
        elif instruction.operator == pikepdf.Operator("EMC"):
            stack.pop()
        elif str(instruction.operator) in {"Tj", "TJ", "'", '"'}:
            text_show_count += 1
            assert any(tag == "/ReversedChars" for tag, _properties in stack)
            assert any(
                isinstance(properties, Dictionary)
                and str(properties.get("/ActualText")) == "both"
                for _tag, properties in stack
            )
            assert (
                sum(
                    isinstance(properties, Dictionary) and "/MCID" in properties
                    for _tag, properties in stack
                )
                == 1
            )
    assert text_show_count == 2
    assert stack == []
    assert [marker[3] for marker in _marked_content(page) if marker[3] is not None] == [
        0,
        1,
    ]


def test_semantic_rebuild_keeps_source_artifact_out_of_structure() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/Artifact <</Type /Pagination /Subtype /Footer>> BDC\n"
            b"BT /F1 10 Tf 20 20 Td (Confidential) Tj "
            b"0 -10 Td (Page footer) Tj ET\n"
            b"0 0 m 10 10 l S\nEMC\n"
            b"BT /F1 12 Tf 20 200 Td (Body text.) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["artifacts_tagged"] == 0
    assert result["semantic_alternatives_review_required"] == 0
    artifact_depth = 0
    source_artifact_count = 0
    mcids = []
    artifact_text = 0
    for instruction in pikepdf.parse_content_stream(page):
        assert isinstance(instruction, pikepdf.ContentStreamInstruction)
        if instruction.operator in {pikepdf.Operator("BMC"), pikepdf.Operator("BDC")}:
            tag = resolve_indirect(instruction.operands[0])
            if tag == Name.Artifact:
                artifact_depth += 1
                source_artifact_count += 1
                properties = resolve_indirect(instruction.operands[1])
                assert properties["/Type"] == Name.Pagination
                assert properties["/Subtype"] == Name.Footer
            if instruction.operator == pikepdf.Operator("BDC"):
                properties = resolve_indirect(instruction.operands[1])
                if isinstance(properties, Dictionary) and "/MCID" in properties:
                    assert artifact_depth == 0
                    mcids.append(int(properties["/MCID"]))
        elif instruction.operator == pikepdf.Operator("EMC"):
            if artifact_depth:
                artifact_depth -= 1
        elif str(instruction.operator) in {"Tj", "TJ", "'", '"'} and artifact_depth:
            artifact_text += 1
    assert source_artifact_count == 1
    assert artifact_text == 2
    assert mcids == [0]

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    owners = resolve_indirect(
        NumberTree(root["/ParentTree"])[int(page.obj["/StructParents"])]
    )
    assert isinstance(owners, Array)
    assert len(owners) == 1


@pytest.mark.parametrize(
    "content",
    [
        b"/Artifact BMC /P <</MCID 0>> BDC 0 0 10 10 re f EMC EMC",
        b"/P <</MCID 0>> BDC /Artifact BMC 0 0 10 10 re f EMC EMC",
    ],
    ids=["artifact-contains-mcid", "mcid-contains-artifact"],
)
def test_existing_mcid_with_only_source_artifact_paint_is_rebuilt(
    content: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, content, size=(100, 100))
    old_root, _paragraph = _install_single_mcid_structure(pdf, page)

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert all(marker[3] is None for marker in _marked_content(page))
    assert all("/Artifact" in stack for _operator, stack in _marked_paint_stacks(page))
    assert _existing_structure_elements(pdf) is not None


@pytest.mark.parametrize(
    "content",
    [
        (b"/P <</MCID 0>> BDC 0 0 10 10 re f /Artifact BMC 20 20 10 10 re f EMC EMC"),
        (b"/Artifact BMC 20 20 10 10 re f EMC /P <</MCID 0>> BDC 0 0 10 10 re f EMC"),
    ],
    ids=["nested-artifact", "separate-artifact"],
)
def test_existing_mcid_with_semantic_and_artifact_paint_is_preserved(
    content: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, content, size=(100, 100))
    old_root, _paragraph = _install_single_mcid_structure(pdf, page)
    original_content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )


@pytest.mark.parametrize(
    ("page_content", "form_content"),
    [
        (
            b"/P <</MCID 0>> BDC /Fm Do EMC",
            b"/Artifact BMC 0 0 10 10 re f EMC",
        ),
        (
            b"/Artifact BMC /P <</MCID 0>> BDC /Fm Do EMC EMC",
            b"0 0 10 10 re f",
        ),
        (
            b"/P <</MCID 0>> BDC /Artifact BMC /Fm Do EMC EMC",
            b"0 0 10 10 re f",
        ),
    ],
    ids=["artifact-in-form", "artifact-around-mcid", "artifact-around-form"],
)
def test_existing_ancestor_mcid_with_only_form_artifact_paint_is_rebuilt(
    page_content: bytes,
    form_content: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, form_content, bbox=(0, 0, 20, 20))
    page = _page(
        pdf,
        page_content,
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )
    old_root, _paragraph = _install_single_mcid_structure(pdf, page)

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert all(marker[3] is None for marker in _marked_content(page))
    assert _existing_structure_elements(pdf) is not None


def test_existing_ancestor_mcid_with_visible_form_paint_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, b"0 0 10 10 re f", bbox=(0, 0, 20, 20))
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )
    old_root, _paragraph = _install_single_mcid_structure(pdf, page)

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_existing_empty_form_mcid_is_preserved_after_save_and_reopen() -> None:
    pdf = pikepdf.Pdf.new()
    empty = _form(pdf, b"", bbox=(0, 0, 20, 20))
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC /Empty Do EMC",
        Dictionary(XObject=Dictionary(Empty=empty)),
        size=(100, 100),
    )
    _install_single_mcid_structure(pdf, page)
    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)

    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_page = reopened.pages[0]
        reopened_empty = resolve_indirect(
            reopened_page.obj["/Resources"]["/XObject"]["/Empty"]
        )
        assert bytes(reopened_empty.get_raw_stream_buffer()) == b""
        original_root = resolve_indirect(reopened.Root["/StructTreeRoot"]).objgen

        result = ensure_logical_structure(reopened, semantic=True)

        assert result["structure_preserved"] is True
        assert (
            resolve_indirect(reopened.Root["/StructTreeRoot"]).objgen == original_root
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "class-missing-owner",
        "class-invalid-owner",
        "class-array-missing-owner",
        "attribute-wrong-type",
        "attribute-missing-owner",
        "attribute-leading-revision",
        "attribute-consecutive-revisions",
        "negative-element-revision",
        "boolean-element-revision",
        "invalid-element-revision",
    ],
)
def test_invalid_existing_structure_attributes_are_rebuilt(invalid: str) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC 0 0 10 10 re f EMC",
        size=(100, 100),
    )
    old_root, paragraph = _install_single_mcid_structure(pdf, page)
    if invalid.startswith("class-"):
        if invalid == "class-missing-owner":
            value: object = Dictionary()
        elif invalid == "class-invalid-owner":
            value = pdf.make_stream(b"")
            value["/O"] = 42
        else:
            value = Array([Dictionary(O=Name.Layout), pdf.make_indirect(Dictionary())])
        old_root["/ClassMap"] = Dictionary(Body=value)
        paragraph["/C"] = Name.Body
    elif invalid == "attribute-wrong-type":
        paragraph["/A"] = 42
    elif invalid == "attribute-missing-owner":
        paragraph["/A"] = pdf.make_indirect(Dictionary())
    elif invalid == "attribute-leading-revision":
        paragraph["/A"] = Array([0, Dictionary(O=Name.Layout)])
    elif invalid == "attribute-consecutive-revisions":
        paragraph["/A"] = Array([Dictionary(O=Name.Layout), 0, 1])
    elif invalid == "negative-element-revision":
        paragraph["/R"] = -1
    elif invalid == "boolean-element-revision":
        paragraph["/R"] = True
    else:
        paragraph["/R"] = String("1")

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert _existing_structure_elements(pdf) is not None


@pytest.mark.parametrize(
    "valid",
    [
        "empty-class-attributes",
        "empty-element-classes",
        "empty-element-attributes",
        "revisioned-element-attributes",
        "class-attribute-array",
        "negative-class-revision",
        "indirect-owner-name",
    ],
)
def test_valid_existing_structure_attributes_are_preserved(valid: str) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC 0 0 10 10 re f EMC",
        size=(100, 100),
    )
    old_root, paragraph = _install_single_mcid_structure(pdf, page)
    if valid == "empty-class-attributes":
        old_root["/ClassMap"] = Dictionary(Body=Array())
        paragraph["/C"] = Name.Body
    elif valid == "empty-element-classes":
        paragraph["/C"] = Array()
    elif valid == "empty-element-attributes":
        paragraph["/A"] = Array()
    elif valid == "revisioned-element-attributes":
        stream_attribute = pdf.make_stream(b"")
        stream_attribute["/O"] = Name.Table
        paragraph["/A"] = Array([Dictionary(O=Name.Layout), -1, stream_attribute, 2])
        paragraph["/R"] = 0
    elif valid == "class-attribute-array":
        stream_attribute = pdf.make_stream(b"")
        stream_attribute["/O"] = Name.Table
        old_root["/ClassMap"] = Dictionary(
            Body=Array([Dictionary(O=Name.Layout), stream_attribute])
        )
        paragraph["/C"] = Name.Body
    elif valid == "negative-class-revision":
        old_root["/ClassMap"] = Dictionary(Body=Dictionary(O=Name.Layout))
        paragraph["/C"] = Array([Name.Body, -1])
    else:
        attribute = pdf.make_indirect(Dictionary(O=pdf.make_indirect(Name.Layout)))
        paragraph["/A"] = attribute

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_artifact_only_existing_mcid_rebuild_failure_is_atomic() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Artifact BMC /P <</MCID 0>> BDC 0 0 10 10 re f EMC EMC",
        size=(100, 100),
    )
    annotation = pdf.make_indirect(
        Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 0, 0]))
    )
    page.obj["/Annots"] = Array([annotation])
    old_root, _paragraph = _install_single_mcid_structure(pdf, page)
    original_content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="Link /Rect is empty"):
        ensure_logical_structure(pdf, semantic=True)

    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert (
        bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_content
    )
    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert len(pdf.objects) == original_object_count


def test_semantic_force_rebuild_does_not_accumulate_structural_wrappers() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer A")))
    page = _page(
        pdf,
        (
            b"/OC /OC1 BDC\n"
            b"/P <</MCID 5 /ActualText (kept)>> BDC\n"
            b"BT /F1 12 Tf 20 200 Td (Body text.) Tj ET\n"
            b"EMC\nEMC"
        ),
        Dictionary(
            Font=Dictionary(F1=font),
            Properties=Dictionary(OC1=ocg),
        ),
        size=(300, 300),
    )

    first = ensure_logical_structure(pdf, semantic=True)
    first_instructions = list(pikepdf.parse_content_stream(page))
    second = ensure_logical_structure(pdf, rebuild=True, semantic=True)
    second_instructions = list(pikepdf.parse_content_stream(page))

    assert first["mcids_removed"] == 1
    assert second["mcids_removed"] == 1

    def signature(
        instructions: list[
            pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage
        ],
    ) -> tuple[int, int, list[int], list[str]]:
        bdc_count = 0
        emc_count = 0
        mcids = []
        actual_text = []
        for instruction in instructions:
            assert isinstance(instruction, pikepdf.ContentStreamInstruction)
            if instruction.operator == pikepdf.Operator("BDC"):
                bdc_count += 1
                properties = resolve_indirect(instruction.operands[1])
                if isinstance(properties, Name):
                    properties = resolve_indirect(
                        page.obj["/Resources"]["/Properties"].get(properties)
                    )
                if isinstance(properties, Dictionary):
                    if "/MCID" in properties:
                        mcids.append(int(properties["/MCID"]))
                    if "/ActualText" in properties:
                        actual_text.append(str(properties["/ActualText"]))
            elif instruction.operator == pikepdf.Operator("EMC"):
                emc_count += 1
        return bdc_count, emc_count, mcids, actual_text

    assert signature(first_instructions) == (3, 3, [0], ["kept"])
    assert signature(second_instructions) == (3, 3, [0], ["kept"])


def test_semantic_form_children_get_stream_mcrs_and_page_do_stays_unmarked() -> None:
    pdf = pikepdf.Pdf.new()
    regular = _font(pdf)
    bold = _font(pdf, bold=True)
    image = pdf.make_stream(b"\x00\x80\xff")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    nested = pdf.make_stream(b"BT /F1 10 Tf 0 5 Td (Nested note.) Tj ET")
    nested["/Type"] = Name.XObject
    nested["/Subtype"] = Name.Form
    nested["/BBox"] = Array([0, 0, 80, 20])
    nested["/Resources"] = Dictionary(Font=Dictionary(F1=regular))
    source_form = pdf.make_stream(
        b"/Span <</ActualText (Accessible form heading)>> BDC\n"
        b"BT /FB 22 Tf 30 250 Td (Form heading) Tj ET\nEMC\n"
        b"BT /F1 11 Tf 30 210 Td [(First) 0 ( paragraph.)] TJ ET\n"
        b"BT /F1 11 Tf 30 194 Td 14 TL (Second paragraph.) ' ET\n"
        b'BT /F1 11 Tf 30 164 Td 14 TL 0 0 (Third paragraph.) " ET\n'
        b"q 24 0 0 18 280 180 cm /Im0 Do Q\n"
        b"q 1 0 0 1 280 130 cm /Nested Do Q\n"
        b"q 12 0 0 12 330 130 cm "
        b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI Q\n"
        b"/Artifact <</Type /Pagination /Subtype /Footer>> BDC\n"
        b"BT /F1 8 Tf 30 15 Td (Form footer) Tj ET\nEMC"
    )
    source_form["/Type"] = Name.XObject
    source_form["/Subtype"] = Name.Form
    source_form["/BBox"] = Array([0, 0, 400, 300])
    source_form["/DL"] = len(source_form.read_bytes())
    source_form["/Resources"] = Dictionary(
        Font=Dictionary(F1=regular, FB=bold),
        XObject=Dictionary(Im0=image, Nested=nested),
    )
    page = _page(
        pdf,
        b"q 1 0 0 1 10 20 cm /Fm0 Do Q",
        Dictionary(XObject=Dictionary(Fm0=source_form)),
        size=(440, 360),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 7
    assert {"/H1", "/P", "/Figure"} <= set(_roles(pdf))
    page_instructions = list(pikepdf.parse_content_stream(page))
    assert not any(
        isinstance(instruction, pikepdf.ContentStreamInstruction)
        and instruction.operator in {pikepdf.Operator("BMC"), pikepdf.Operator("BDC")}
        for instruction in page_instructions
    )
    page_do = next(
        instruction
        for instruction in page_instructions
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and str(instruction.operator) == "Do"
    )
    target_name = resolve_indirect(page_do.operands[0])
    assert isinstance(target_name, Name)
    assert str(target_name).startswith("/PdftopdfaSemanticForm1_0")
    xobjects = resolve_indirect(page.obj["/Resources"]["/XObject"])
    clone = resolve_indirect(xobjects[target_name])
    assert isinstance(clone, pikepdf.Stream)
    assert clone.objgen != source_form.objgen
    assert "/DL" not in clone
    assert sorted(
        marker[3] for marker in _marked_content(clone) if marker[3] is not None
    ) == list(range(6))
    nested_clones = _invoked_forms(clone)
    assert len(nested_clones) == 1
    nested_clone = nested_clones[0]
    assert [
        marker[3] for marker in _marked_content(nested_clone) if marker[3] is not None
    ] == [0]
    assert any(
        isinstance(resolve_indirect(instruction.operands[1]), Dictionary)
        and str(resolve_indirect(instruction.operands[1]).get("/ActualText"))
        == "Accessible form heading"
        for instruction in pikepdf.parse_content_stream(clone)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and instruction.operator == pikepdf.Operator("BDC")
    )
    artifact_depth = 0
    for instruction in pikepdf.parse_content_stream(clone):
        if not isinstance(instruction, pikepdf.ContentStreamInstruction):
            continue
        if instruction.operator in {
            pikepdf.Operator("BMC"),
            pikepdf.Operator("BDC"),
        }:
            if resolve_indirect(instruction.operands[0]) == Name.Artifact:
                artifact_depth += 1
            elif instruction.operator == pikepdf.Operator("BDC"):
                properties = resolve_indirect(instruction.operands[1])
                if isinstance(properties, Dictionary) and "/MCID" in properties:
                    assert artifact_depth == 0
        elif instruction.operator == pikepdf.Operator("EMC") and artifact_depth:
            artifact_depth -= 1
    assert artifact_depth == 0
    assert "/StructParents" not in page.obj

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    form_key = int(clone["/StructParents"])
    parent_array = resolve_indirect(NumberTree(root["/ParentTree"])[form_key])
    assert isinstance(parent_array, Array)
    assert len(parent_array) == 6
    nested_form_key = int(nested_clone["/StructParents"])
    nested_parent_array = resolve_indirect(
        NumberTree(root["/ParentTree"])[nested_form_key]
    )
    assert isinstance(nested_parent_array, Array)
    assert len(nested_parent_array) == 1
    mcrs = [item for item in _structure_objects(pdf) if item.get("/Type") == Name.MCR]
    assert len(mcrs) == 7
    assert (
        sum(resolve_indirect(item.get("/Stm")).objgen == clone.objgen for item in mcrs)
        == 6
    )
    assert (
        sum(
            resolve_indirect(item.get("/Stm")).objgen == nested_clone.objgen
            for item in mcrs
        )
        == 1
    )


def test_reused_page_form_gets_per_invocation_clones_and_stable_parent_trees() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    source_form = pdf.make_stream(b"BT /F1 10 Tf 5 10 Td (Repeated.) Tj ET")
    source_form["/Type"] = Name.XObject
    source_form["/Subtype"] = Name.Form
    source_form["/BBox"] = Array([0, 0, 80, 30])
    source_form["/Resources"] = Dictionary(Font=Dictionary(F1=font))
    page = _page(
        pdf,
        (b"q 1 0 0 1 20 220 cm /Fm0 Do Q\nq 2 0 0 2 220 80 cm /Fm0 Do Q"),
        Dictionary(XObject=Dictionary(Fm0=source_form)),
        size=(420, 320),
    )

    first = ensure_logical_structure(pdf, semantic=True)

    assert first["semantic_content_items"] == 2
    first_names = [
        resolve_indirect(instruction.operands[0])
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and str(instruction.operator) == "Do"
    ]
    assert len(first_names) == 2
    assert first_names[0] != first_names[1]
    xobjects = resolve_indirect(page.obj["/Resources"]["/XObject"])
    first_clones = [resolve_indirect(xobjects[name]) for name in first_names]
    assert all(isinstance(clone, pikepdf.Stream) for clone in first_clones)
    assert first_clones[0].objgen != first_clones[1].objgen
    assert all(
        [marker[3] for marker in _marked_content(clone)] == [0]
        for clone in first_clones
    )
    first_parent_keys = [int(clone["/StructParents"]) for clone in first_clones]
    assert first_parent_keys[0] != first_parent_keys[1]
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    parent_tree = NumberTree(root["/ParentTree"])
    assert all(
        len(resolve_indirect(parent_tree[key])) == 1 for key in first_parent_keys
    )
    paragraph_boxes = []
    for item in _structure_objects(pdf):
        if item.get("/S") != Name.P:
            continue
        attributes = resolve_indirect(item.get("/A"))
        attributes = list(attributes) if isinstance(attributes, Array) else [attributes]
        layout = next(
            (
                resolve_indirect(attribute)
                for attribute in attributes
                if isinstance(resolve_indirect(attribute), Dictionary)
                and resolve_indirect(attribute).get("/O") == Name.Layout
            ),
            None,
        )
        if isinstance(layout, Dictionary) and isinstance(layout.get("/BBox"), Array):
            paragraph_boxes.append(tuple(float(value) for value in layout["/BBox"]))
    assert len(paragraph_boxes) == 2
    assert sorted(box[0] for box in paragraph_boxes)[0] < 50
    assert sorted(box[0] for box in paragraph_boxes)[1] > 200

    buffer = BytesIO()
    pdf.save(buffer)
    buffer.seek(0)
    with pikepdf.Pdf.open(buffer) as reopened:
        rebuilt = ensure_logical_structure(reopened, rebuild=True, semantic=True)
        assert rebuilt["semantic_content_items"] == 2
        reopened_page = reopened.pages[0]
        second_names = [
            resolve_indirect(instruction.operands[0])
            for instruction in pikepdf.parse_content_stream(reopened_page)
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and str(instruction.operator) == "Do"
        ]
        assert [str(name) for name in second_names] == [
            str(name) for name in first_names
        ]
        reopened_xobjects = resolve_indirect(
            reopened_page.obj["/Resources"]["/XObject"]
        )
        assert (
            len(
                [
                    name
                    for name in reopened_xobjects.keys()
                    if str(name).startswith("/PdftopdfaSemanticForm1_")
                ]
            )
            == 2
        )
        second_clones = [
            resolve_indirect(reopened_xobjects[name]) for name in second_names
        ]
        assert all(
            [marker[3] for marker in _marked_content(clone)] == [0]
            for clone in second_clones
        )
        assert len({int(clone["/StructParents"]) for clone in second_clones}) == 2


def test_pure_vector_form_invocation_becomes_described_figure() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(pdf, b"0 0 m 50 40 l S", bbox=(0, 0, 50, 40))
    page = _page(
        pdf,
        (
            b"/Span <</ActualText (Revenue trend line)>> BDC\n"
            b"q 1 0 0 1 20 30 cm /Vector Do Q\nEMC"
        ),
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 0
    assert result["semantic_alternatives_review_required"] == 0
    figures = [
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    ]
    assert len(figures) == 1
    assert str(figures[0]["/ActualText"]) == "Revenue trend line"
    assert [marker[3] for marker in _marked_content(page) if marker[3] is not None] == [
        0
    ]


def test_reused_vector_form_cache_includes_inherited_line_width() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        b"20 90 m 90 20 l S",
        bbox=(0, 0, 100, 100),
    )
    _page(
        pdf,
        (b"q 60 60 20 20 re W n 1 w /Vector Do 40 w /Vector Do Q"),
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/Figure") == 1


def test_reused_vector_form_cache_includes_outer_invocation_clip() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        b"20 90 m 90 20 l S",
        bbox=(0, 0, 100, 100),
    )
    _page(
        pdf,
        (b"q 60 60 20 20 re W n /Vector Do Q q 40 40 20 20 re W n /Vector Do Q"),
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/Figure") == 1


@pytest.mark.parametrize(
    "form_content",
    [b"20 90 m 90 20 l S", b"20 20 60 60 re f"],
)
def test_zero_alpha_form_vector_invocation_is_omitted_per_reuse(
    form_content: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(pdf, form_content, bbox=(0, 0, 100, 100))
    _page(
        pdf,
        b"q /Hidden gs /Vector Do Q /Vector Do",
        Dictionary(
            ExtGState=Dictionary(Hidden=Dictionary(CA=0, ca=0)),
            XObject=Dictionary(Vector=vector),
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/Figure") == 1


def test_fully_hidden_described_form_vector_scope_is_omitted() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        (b"/Figure <</ActualText (Hidden line)>> BDC 20 90 m 90 20 l S EMC"),
        bbox=(0, 0, 100, 100),
    )
    _page(
        pdf,
        b"60 60 20 20 re W n /Vector Do",
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert "/Figure" not in _roles(pdf)


def test_partially_clipped_described_form_vector_scope_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        (b"/Figure <</ActualText (Clipped line)>> BDC 20 70 m 90 70 l S EMC"),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"60 60 20 20 re W n /Vector Do",
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(100, 100),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="partially clipped"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_unclassified_vector_form_bbox_uses_visible_paint_union() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        b"50 50 20 20 re f",
        bbox=(0, 0, 200, 200),
    )
    _page(
        pdf,
        b"/Vector Do",
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(200, 200),
    )

    ensure_logical_structure(pdf, semantic=True)

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    attributes = resolve_indirect(figure["/A"])
    attributes = list(attributes) if isinstance(attributes, Array) else [attributes]
    layout = next(
        resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Layout
    )
    assert tuple(float(value) for value in layout["/BBox"]) == pytest.approx(
        (50, 50, 70, 70)
    )


@pytest.mark.parametrize("painting_kind", ["direct", "inline", "form"])
def test_zero_fill_alpha_image_invocation_is_omitted_per_reuse(
    painting_kind: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    extgstate = Dictionary(Hidden=Dictionary(ca=0))
    if painting_kind == "direct":
        painting = b"/Im Do"
        resources = Dictionary(ExtGState=extgstate, XObject=Dictionary(Im=image))
    elif painting_kind == "inline":
        painting = b"BI /W 1 /H 1 /CS /G /BPC 8 ID\n\x80\nEI"
        resources = Dictionary(ExtGState=extgstate)
    else:
        form = _form(
            pdf,
            b"/Im Do",
            Dictionary(XObject=Dictionary(Im=image)),
            bbox=(0, 0, 1, 1),
        )
        painting = b"/Form Do"
        resources = Dictionary(
            ExtGState=extgstate,
            XObject=Dictionary(Form=form),
        )
    _page(
        pdf,
        b"q /Hidden gs " + painting + b" Q " + painting + b"\n",
        resources,
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/Figure") == 1


def test_soft_masked_form_vector_fails_closed_without_mutation() -> None:
    pdf = pikepdf.Pdf.new()
    mask = _form(pdf, b"0 0 10 10 re f", bbox=(0, 0, 10, 10))
    vector = _form(pdf, b"0 0 10 10 re f", bbox=(0, 0, 10, 10))
    page = _page(
        pdf,
        b"/Masked gs /Vector Do",
        Dictionary(
            ExtGState=Dictionary(
                Masked=Dictionary(
                    SMask=Dictionary(S=Name.Luminosity, G=mask),
                )
            ),
            XObject=Dictionary(Vector=vector),
        ),
        size=(100, 100),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="soft mask"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_sheared_form_text_outside_exact_bbox_polygon_is_omitted() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        b"BT /F1 5 Tf -7 8 Td (I) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(0, 0, 20, 10),
    )
    form["/Matrix"] = Array([1, 0, 1, 1, 0, 0])
    _page(
        pdf,
        b"/Form Do",
        Dictionary(XObject=Dictionary(Form=form)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.P for item in _structure_objects(pdf))


def test_form_vector_scope_bbox_uses_invocation_transform() -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(
        pdf,
        b"/Figure <</ActualText (Local region)>> BDC 5 5 10 10 re f EMC",
        bbox=(0, 0, 100, 100),
    )
    _page(
        pdf,
        b"q 2 0 0 2 50 50 cm /Vector Do Q",
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(300, 300),
    )

    ensure_logical_structure(pdf, semantic=True)

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    attributes = resolve_indirect(figure["/A"])
    attributes = list(attributes) if isinstance(attributes, Array) else [attributes]
    layout = next(
        resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Layout
    )
    assert tuple(float(value) for value in layout["/BBox"]) == pytest.approx(
        (60, 60, 80, 80)
    )


@pytest.mark.parametrize("named_properties", [False, True])
def test_page_actualtext_vector_scope_becomes_one_figure(
    named_properties: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    properties = Dictionary(ActualText=String("Two-line chart"))
    resources = (
        Dictionary(Properties=Dictionary(Chart=properties))
        if named_properties
        else Dictionary()
    )
    opener = (
        b"/Figure /Chart BDC"
        if named_properties
        else b"/Figure <</ActualText (Two-line chart)>> BDC"
    )
    page = _page(
        pdf,
        opener + b"\n1 1 m 80 60 l S\n1 60 m 80 1 l S\nEMC",
        resources,
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 0
    assert result["semantic_alternatives_review_required"] == 0
    figures = [
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    ]
    assert len(figures) == 1
    assert str(figures[0]["/ActualText"]) == "Two-line chart"
    assert [marker[3] for marker in _marked_content(page) if marker[3] is not None] == [
        0
    ]
    assert not any(marker[0] == "/Artifact" for marker in _marked_content(page))

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_figures = [
            item
            for item in _structure_objects(reopened)
            if item.get("/S") == Name.Figure
        ]
        assert len(reopened_figures) == 1
        assert str(reopened_figures[0]["/ActualText"]) == "Two-line chart"


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_page_vector_scope_bbox_uses_default_space_with_crop_and_rotation(
    rotation: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Figure <</ActualText (Filled region)>> BDC 30 20 40 20 re f EMC",
        size=(200, 100),
    )
    page.obj["/CropBox"] = Array([20, 10, 180, 90])
    page.obj["/Rotate"] = rotation

    ensure_logical_structure(pdf, semantic=True)

    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    attributes = resolve_indirect(figure["/A"])
    attributes = list(attributes) if isinstance(attributes, Array) else [attributes]
    layout = next(
        resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Layout
    )
    assert tuple(float(value) for value in layout["/BBox"]) == pytest.approx(
        (30, 20, 70, 40)
    )


def test_page_actualtext_scope_mixing_text_and_vector_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    page = _page(
        pdf,
        (
            b"/Figure <</ActualText (Mixed replacement)>> BDC\n"
            b"BT /F1 10 Tf 20 80 Td (Label) Tj ET\n"
            b"1 1 m 80 60 l S\nEMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        size=(200, 200),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(
        ConversionError,
        match="marked text-evidence scope mixing vector and other painting",
    ):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("inside_form", [False, True])
@pytest.mark.parametrize("named_properties", [False, True])
def test_marked_image_alt_is_typed_structure_evidence(
    inside_form: bool,
    named_properties: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    properties = Dictionary(Alt=String("Status indicator"))
    resources = Dictionary(XObject=Dictionary(Im=image))
    opener = b"/Figure <</Alt (Status indicator)>> BDC"
    if named_properties:
        resources["/Properties"] = Dictionary(Description=properties)
        opener = b"/Figure /Description BDC"
    content = opener + b" q 20 0 0 20 20 20 cm /Im Do Q EMC"
    if inside_form:
        form = _form(pdf, content, resources)
        _page(
            pdf,
            b"/Form Do",
            Dictionary(XObject=Dictionary(Form=form)),
            size=(200, 200),
        )
    else:
        _page(pdf, content, resources, size=(200, 200))

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_alternatives_review_required"] == 0
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/Alt"]) == "Status indicator"
    assert "/ActualText" not in figure

    serialized = BytesIO()
    pdf.save(serialized)
    serialized.seek(0)
    with pikepdf.Pdf.open(serialized) as reopened:
        reopened_figure = next(
            item
            for item in _structure_objects(reopened)
            if item.get("/S") == Name.Figure
        )
        assert str(reopened_figure["/Alt"]) == "Status indicator"


def test_marked_vector_alt_scope_becomes_figure_alt() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        b"/Figure <</Alt (Process diagram)>> BDC 1 1 m 80 60 l S EMC",
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_alternatives_review_required"] == 0
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/Alt"]) == "Process diagram"
    assert "/ActualText" not in figure


def test_preserved_figure_recovers_typed_alt_from_marked_content() -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    page = _page(
        pdf,
        b"/Figure /Description BDC q 20 0 0 20 20 20 cm /Im Do Q EMC",
        Dictionary(
            XObject=Dictionary(Im=image),
            Properties=Dictionary(
                Description=Dictionary(Alt=String("Recovered description"))
            ),
        ),
        size=(200, 200),
    )
    ensure_logical_structure(pdf, semantic=True)
    root_key = pdf.Root["/StructTreeRoot"].objgen
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    del figure["/Alt"]

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["semantic_repairs"] == 1
    assert pdf.Root["/StructTreeRoot"].objgen == root_key
    repaired = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(repaired["/Alt"]) == "Recovered description"
    assert int(page.obj["/StructParents"]) >= 0


def test_artifact_actualtext_vector_scope_remains_source_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Artifact <</ActualText (Decoration)>> BDC 0 0 m 80 60 l S EMC",
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["semantic_vector_review_required"] == 0
    assert "/Figure" not in _roles(pdf)
    assert sum(marker[0] == "/Artifact" for marker in _marked_content(page)) == 1


def test_form_text_and_described_vector_scope_remain_separate_leaves() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        (
            b"BT /F1 10 Tf 20 100 Td (Body paragraph) Tj ET\n"
            b"/Figure <</ActualText (Trend chart)>> BDC\n"
            b"1 1 m 80 60 l S\n10 1 m 70 60 l S\nEMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    page = _page(
        pdf,
        b"/Form Do",
        Dictionary(XObject=Dictionary(Form=form)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    assert result["semantic_vector_review_required"] == 0
    assert {"/P", "/Figure"} <= set(_roles(pdf))
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Trend chart"
    form_name = next(
        resolve_indirect(instruction.operands[0])
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and str(instruction.operator) == "Do"
    )
    clone = resolve_indirect(page.obj["/Resources"]["/XObject"][form_name])
    assert isinstance(clone, pikepdf.Stream)
    assert sorted(
        marker[3] for marker in _marked_content(clone) if marker[3] is not None
    ) == [0, 1]


def test_empty_form_invocation_remains_a_layout_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    empty = _form(pdf, b" ", bbox=(0, 0, 50, 40))
    page = _page(
        pdf,
        b"q 1 0 0 1 20 30 cm /Empty Do Q",
        Dictionary(XObject=Dictionary(Empty=empty)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["artifacts_tagged"] == 1
    assert "/Figure" not in _roles(pdf)
    assert ("/Artifact", "/Layout", None, None) in _marked_content(page)


def test_nested_form_summaries_are_cached_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    depth = 20
    _nested_form_document(pdf, depth)
    summary_count = 0
    original_summary = tagging._FormSemanticSummary

    def make_summary(*args, **kwargs):
        nonlocal summary_count
        summary_count += 1
        return original_summary(*args, **kwargs)

    monkeypatch.setattr(tagging, "_FormSemanticSummary", make_summary)

    tagging._digital_semantic_inputs(pdf, {})

    assert summary_count == depth


def test_semantic_form_nesting_limit_is_supported() -> None:
    pdf = pikepdf.Pdf.new()
    _nested_form_document(pdf, digital_layout._MAX_FORM_NESTING_DEPTH)

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/P") == 1


def test_semantic_form_nesting_over_limit_fails_before_spooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _nested_form_document(pdf, digital_layout._MAX_FORM_NESTING_DEPTH + 1)
    root_keys = set(pdf.Root.keys())
    page_keys = set(page.obj.keys())
    content = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())
    object_count = len(pdf.objects)

    def unexpected_spool(*_args, **_kwargs):
        pytest.fail("semantic preflight was spooled before the Form depth check")

    monkeypatch.setattr(tagging, "SpooledTemporaryFile", unexpected_spool)

    with pytest.raises(ConversionError, match="nesting depth budget exceeded"):
        ensure_logical_structure(pdf, semantic=True)

    assert set(pdf.Root.keys()) == root_keys
    assert set(page.obj.keys()) == page_keys
    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == content
    assert len(pdf.objects) == object_count


def test_nested_form_source_artifact_text_is_not_bound() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"/Artifact BMC BT /F1 10 Tf 20 20 Td (Inner footer) Tj ET EMC",
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    page = _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert "/P" not in _roles(pdf)
    assert ("/Artifact", "/Layout", None, None) in _marked_content(page)


def test_nested_form_actualtext_is_propagated_in_execution_order() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        (
            b"/Span <</ActualText (Accessible replacement)>> BDC "
            b"BT /F1 10 Tf 20 20 Td (Raw glyphs) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    paragraphs = [item for item in _structure_objects(pdf) if item.get("/S") == Name.P]
    assert len(paragraphs) == 1
    assert str(paragraphs[0]["/ActualText"]) == "Accessible replacement"


@pytest.mark.parametrize("wrapper_count", [0, 1, 2])
def test_nested_form_preserves_multiple_semantic_text_leaves(
    wrapper_count: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        (
            b"BT /F1 10 Tf 20 140 Td (First paragraph) Tj ET\n"
            b"BT /F1 10 Tf 20 20 Td (Second paragraph) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    invoked = inner
    for _index in range(wrapper_count):
        invoked = _form(
            pdf,
            b"/Nested Do",
            Dictionary(XObject=Dictionary(Nested=invoked)),
        )
    page = _page(
        pdf,
        b"/Root Do",
        Dictionary(XObject=Dictionary(Root=invoked)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    assert _roles(pdf).count("/P") == 2
    chain = []
    container: pikepdf.Page | pikepdf.Stream = page
    for _index in range(wrapper_count + 1):
        forms = _invoked_forms(container)
        assert len(forms) == 1
        container = forms[0]
        chain.append(container)
    assert all(
        not any(marker[3] is not None for marker in _marked_content(form))
        for form in chain[:-1]
    )
    assert [
        marker[3] for marker in _marked_content(chain[-1]) if marker[3] is not None
    ] == [0, 1]
    assert "/StructParents" in chain[-1]
    assert all("/StructParents" not in form for form in chain[:-1])
    mcr_streams = {
        resolve_indirect(item["/Stm"]).objgen
        for element in _structure_objects(pdf)
        for item in _k_objects(element)
        if item.get("/Type") == Name.MCR and "/Stm" in item
    }
    assert mcr_streams == {chain[-1].objgen}
    if wrapper_count == 2:
        serialized = BytesIO()
        pdf.save(serialized)
        serialized.seek(0)
        with pikepdf.Pdf.open(serialized) as reopened:
            rebuilt = ensure_logical_structure(
                reopened,
                rebuild=True,
                semantic=True,
            )
            assert rebuilt["semantic_content_items"] == 2
            assert _roles(reopened).count("/P") == 2
            reopened_container: pikepdf.Page | pikepdf.Stream = reopened.pages[0]
            reopened_chain = []
            for _index in range(wrapper_count + 1):
                forms = _invoked_forms(reopened_container)
                assert len(forms) == 1
                reopened_container = forms[0]
                reopened_chain.append(reopened_container)
            assert [
                marker[3]
                for marker in _marked_content(reopened_chain[-1])
                if marker[3] is not None
            ] == [0, 1]
            for owner in [reopened.pages[0], *reopened_chain[:-1]]:
                owner_object = owner.obj if isinstance(owner, pikepdf.Page) else owner
                owner_resources = resolve_indirect(owner_object["/Resources"])
                owner_xobjects = resolve_indirect(owner_resources["/XObject"])
                assert (
                    sum(
                        str(name).startswith("/PdftopdfaSemanticForm1_")
                        for name in owner_xobjects.keys()
                    )
                    == 1
                )


@pytest.mark.parametrize("wrapper_count", [0, 1, 2])
def test_nested_form_preserves_multiple_image_leaves(wrapper_count: int) -> None:
    pdf = pikepdf.Pdf.new()
    images = Dictionary()
    for name, value in (("First", b"\x00"), ("Second", b"\xff")):
        image = pdf.make_stream(value)
        image["/Type"] = Name.XObject
        image["/Subtype"] = Name.Image
        image["/Width"] = 1
        image["/Height"] = 1
        image["/ColorSpace"] = Name.DeviceGray
        image["/BitsPerComponent"] = 8
        images[f"/{name}"] = image
    inner = _form(
        pdf,
        (
            b"/Figure <</ActualText (First icon)>> BDC "
            b"q 20 0 0 20 20 80 cm /First Do Q EMC\n"
            b"/Figure <</ActualText (Second icon)>> BDC "
            b"q 20 0 0 20 20 20 cm /Second Do Q EMC"
        ),
        Dictionary(XObject=images),
    )
    invoked = inner
    for _index in range(wrapper_count):
        invoked = _form(
            pdf,
            b"/Nested Do",
            Dictionary(XObject=Dictionary(Nested=invoked)),
        )
    page = _page(
        pdf,
        b"/Root Do",
        Dictionary(XObject=Dictionary(Root=invoked)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    figures = [
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    ]
    assert result["semantic_content_items"] == 2
    assert len(figures) == 2
    assert {str(figure["/ActualText"]) for figure in figures} == {
        "First icon",
        "Second icon",
    }
    chain = []
    container: pikepdf.Page | pikepdf.Stream = page
    for _index in range(wrapper_count + 1):
        forms = _invoked_forms(container)
        assert len(forms) == 1
        container = forms[0]
        chain.append(container)
    assert [
        marker[3] for marker in _marked_content(chain[-1]) if marker[3] is not None
    ] == [0, 1]
    assert "/StructParents" in chain[-1]
    assert all("/StructParents" not in form for form in chain[:-1])


def test_nested_form_preserves_artifact_and_semantic_leaves() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        (
            b"/Artifact BMC BT /F1 8 Tf 10 10 Td (Footer) Tj ET EMC\n"
            b"BT /F1 10 Tf 20 80 Td (Semantic body) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    page = _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )
    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert _roles(pdf).count("/P") == 1
    outer_clone = _invoked_forms(page)[0]
    inner_clone = _invoked_forms(outer_clone)[0]
    assert b"/Artifact BMC" in bytes(inner_clone.read_bytes())
    assert [marker[3] for marker in _marked_content(inner_clone)] == [0]
    assert "/StructParents" in inner_clone
    assert "/StructParents" not in outer_clone


def test_reused_nested_form_gets_per_invocation_leaf_clones() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"BT /F1 10 Tf 5 10 Td (Repeated nested text.) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(0, 0, 100, 30),
    )
    outer = _form(
        pdf,
        (b"q 1 0 0 1 20 140 cm /Inner Do Q\nq 1 0 0 1 20 20 cm /Inner Do Q"),
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    page = _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(240, 240),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    outer_clone = _invoked_forms(page)[0]
    inner_clones = _invoked_forms(outer_clone)
    assert len(inner_clones) == 2
    assert inner_clones[0].objgen != inner_clones[1].objgen
    assert all(
        [marker[3] for marker in _marked_content(clone) if marker[3] is not None] == [0]
        for clone in inner_clones
    )
    assert len({int(clone["/StructParents"]) for clone in inner_clones}) == 2


def test_nested_form_leaf_uses_inherited_resources() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"BT /F1 10 Tf 20 40 Td (Inherited font text.) Tj ET",
    )
    del inner["/Resources"]
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Inner=inner),
        ),
    )
    page = _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    outer_clone = _invoked_forms(page)[0]
    inner_clone = _invoked_forms(outer_clone)[0]
    assert [
        marker[3] for marker in _marked_content(inner_clone) if marker[3] is not None
    ] == [0]
    assert "/Resources" not in inner_clone
    assert "/StructParents" in inner_clone


def test_nested_form_text_and_vector_requests_manual_vector_review() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"BT /F1 10 Tf 20 80 Td (Chart label) Tj ET 0 0 m 80 60 l S",
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 1
    assert "/P" in _roles(pdf)


def test_nested_pure_vector_form_uses_invocation_actualtext() -> None:
    pdf = pikepdf.Pdf.new()
    inner = _form(pdf, b"0 0 m 80 60 l S")
    outer = _form(
        pdf,
        b"/Span <</ActualText (Quarterly trend)>> BDC /Inner Do EMC",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 0
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Quarterly trend"


def test_nested_form_scoped_text_clip_keeps_later_visible_text() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        (
            b"q BT /F1 60 Tf 7 Tr 10 150 Td (Clipping title) Tj ET Q\n"
            b"BT /F1 10 Tf 20 40 Td (Visible body) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    paragraphs = [item for item in _structure_objects(pdf) if item.get("/S") == Name.P]
    assert len(paragraphs) == 1


def test_source_artifact_form_does_not_request_nested_vector_review() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        b"BT /F1 10 Tf 20 80 Td (Decoration) Tj ET 0 0 m 80 60 l S",
        Dictionary(Font=Dictionary(F1=font)),
    )
    _page(
        pdf,
        b"/Artifact BMC /Decor Do EMC",
        Dictionary(XObject=Dictionary(Decor=form)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["semantic_vector_review_required"] == 0


@pytest.mark.parametrize("hidden_by", ["path_clip", "text_clip", "cropbox"])
def test_nested_form_hidden_text_does_not_leak_into_structure(hidden_by: str) -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    if hidden_by == "path_clip":
        inner_content = (
            b"0 0 10 10 re W n BT /F1 10 Tf 100 100 Td (Hidden by path clip) Tj ET"
        )
    elif hidden_by == "text_clip":
        inner_content = (
            b"BT /F1 10 Tf 7 Tr 10 10 Td (X) Tj ET "
            b"BT /F1 10 Tf 0 Tr 100 100 Td (Hidden by text clip) Tj ET"
        )
    else:
        inner_content = b"BT /F1 10 Tf 100 100 Td (Outside crop) Tj ET"
    inner = _form(
        pdf,
        inner_content,
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do BT /F1 10 Tf 20 20 Td (Outer visible) Tj ET",
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Inner=inner),
        ),
    )
    page = _page(
        pdf,
        b"/Outer Do BT /F1 10 Tf 20 35 Td (Page visible) Tj ET",
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Outer=outer),
        ),
        size=(200, 200),
    )
    if hidden_by == "cropbox":
        page.obj["/CropBox"] = Array([0, 0, 50, 50])
    if hidden_by == "text_clip":
        original = bytes(page.obj["/Contents"].read_bytes())
        with pytest.raises(ConversionError, match="Cannot create semantic digital PDF"):
            ensure_logical_structure(pdf, semantic=True)
        assert bytes(page.obj["/Contents"].read_bytes()) == original
        assert "/StructTreeRoot" not in pdf.Root
        return

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 2
    outer_name = next(
        resolve_indirect(instruction.operands[0])
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and str(instruction.operator) == "Do"
    )
    outer_clone = resolve_indirect(page.obj["/Resources"]["/XObject"][outer_name])
    assert isinstance(outer_clone, pikepdf.Stream)
    assert ("/Artifact", "/Layout", None, None) in _marked_content(outer_clone)


@pytest.mark.parametrize(
    ("content", "expected_items"),
    [
        (b"0 0 10 10 re W n 100 100 m 150 150 l S", 0),
        (b"0 0 120 120 re W n 100 100 m 150 150 l S", 1),
    ],
)
def test_vector_form_respects_conservative_active_clip(
    content: bytes,
    expected_items: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    vector = _form(pdf, content)
    _page(
        pdf,
        b"/Vector Do",
        Dictionary(XObject=Dictionary(Vector=vector)),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == expected_items
    assert ("/Figure" in _roles(pdf)) is bool(expected_items)
    assert result["semantic_vector_review_required"] == 0


@pytest.mark.parametrize(
    "clip",
    [
        b"0 0 10 10 re 90 90 10 10 re W n",
        b"0 0 m 0 100 l 0 0 l 100 0 l h W n",
    ],
)
def test_described_vector_under_nonrectangular_clip_fails_closed(
    clip: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (b"/Figure <</ActualText (Hidden)>> BDC " + clip + b" 40 40 10 10 re f EMC"),
        size=(200, 200),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="non-rectangular clip"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_described_vector_after_scoped_nonrectangular_clip_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q 0 0 m 20 0 l 10 20 l h W n 0 0 m 20 20 l S Q "
            b"/Figure <</ActualText (Visible)>> BDC "
            b"40 40 10 10 re f EMC"
        ),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Visible"


@pytest.mark.parametrize("operator", [b"f", b"S"])
def test_empty_described_vector_path_does_not_create_figure(operator: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        b"/Figure <</ActualText (Empty path)>> BDC " + operator + b" EMC",
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.Figure for item in _structure_objects(pdf))


@pytest.mark.parametrize(
    "path",
    [
        b"10 10 m S",
        b"10 10 m 20 20 l f",
    ],
)
def test_degenerate_described_vector_path_does_not_create_figure(
    path: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        b"/Figure <</ActualText (Degenerate)>> BDC " + path + b" EMC",
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.Figure for item in _structure_objects(pdf))


def test_ambiguous_compound_described_fill_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"/Figure <</ActualText (Cancelled)>> BDC "
            b"10 10 20 20 re 10 10 20 20 re f* EMC"
        ),
        size=(100, 100),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="ambiguous path geometry"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_described_vector_accepts_implicitly_closed_rectangular_clip() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"0 0 m 100 0 l 100 100 l 0 100 l W n "
            b"/Figure <</ActualText (Visible)>> BDC 40 40 10 10 re f EMC"
        ),
        size=(200, 200),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Visible"


@pytest.mark.parametrize("coords", [(10, 0, 20, 0), (300, 0, 400, 0)])
def test_described_shading_without_exact_geometry_fails_closed(
    coords: tuple[int, int, int, int],
) -> None:
    pdf = pikepdf.Pdf.new()
    function = pdf.make_indirect(
        Dictionary(
            FunctionType=2,
            Domain=Array([0, 1]),
            C0=Array([0, 0, 0]),
            C1=Array([1, 1, 1]),
            N=1,
        )
    )
    shading = pdf.make_indirect(
        Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            Coords=Array(coords),
            Function=function,
            Extend=Array([False, False]),
        )
    )
    page = _page(
        pdf,
        b"/Figure <</ActualText (Gradient)>> BDC /Sh0 sh EMC",
        Dictionary(Shading=Dictionary(Sh0=shading)),
        size=(200, 100),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="described shading"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_described_vector_line_wholly_outside_clip_is_omitted() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"40 40 20 20 re W n "
            b"/Figure <</Alt (Phantom diagonal)>> BDC "
            b"0 70 m 70 0 l S EMC"
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.Figure for item in _structure_objects(pdf))


def test_partially_clipped_described_line_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"40 40 20 20 re W n "
            b"/Figure <</Alt (Clipped line)>> BDC "
            b"0 50 m 70 50 l S EMC"
        ),
        size=(100, 100),
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="partially clipped"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_large_translation_does_not_make_skewed_vector_clip_rectangular() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"q 1 0 0 1 1000000000000 1000000000000 cm "
            b"0 0 m 500 2000 l 2000 1500 l 1500 0 l h W n "
            b"/Figure <</ActualText (Outside)>> BDC "
            b"50 1850 20 20 re f EMC Q"
        ),
        size=(200, 200),
    )
    page.obj["/MediaBox"] = Array(
        [
            1_000_000_000_000,
            1_000_000_000_000,
            1_000_000_003_000,
            1_000_000_003_000,
        ]
    )
    original = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(ConversionError, match="non-rectangular clip"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_described_vector_wholly_inside_rectangular_clip_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"20 20 60 60 re W n "
            b"/Figure <</ActualText (Inside)>> BDC "
            b"30 30 20 20 re f EMC"
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    figure = next(
        item for item in _structure_objects(pdf) if item.get("/S") == Name.Figure
    )
    assert str(figure["/ActualText"]) == "Inside"


def test_nested_form_uses_visible_descendant_bbox_not_full_form_bbox() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        (
            b"/Span <</ActualText (Nested label)>> BDC "
            b"BT /F1 10 Tf 120 120 Td (Raw label) Tj ET EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(200, 200),
    )

    ensure_logical_structure(pdf, semantic=True)

    paragraph = next(
        item
        for item in _structure_objects(pdf)
        if item.get("/S") == Name.P and str(item.get("/ActualText")) == "Nested label"
    )
    attributes = resolve_indirect(paragraph["/A"])
    attributes = list(attributes) if isinstance(attributes, Array) else [attributes]
    layout = next(
        resolve_indirect(attribute)
        for attribute in attributes
        if isinstance(resolve_indirect(attribute), Dictionary)
        and resolve_indirect(attribute).get("/O") == Name.Layout
    )
    bbox = tuple(float(value) for value in layout["/BBox"])
    assert bbox[0] > 100
    assert bbox[2] - bbox[0] < 100


def test_semantic_form_fails_closed_for_untracked_nested_do() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    unsupported = pdf.make_stream(b"")
    unsupported["/Type"] = Name.XObject
    unsupported["/Subtype"] = Name("/PS")
    form = pdf.make_stream(b"BT /F1 10 Tf 5 20 Td (Tracked.) Tj ET /Unsupported Do")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 100, 40])
    form["/Resources"] = Dictionary(
        Font=Dictionary(F1=font),
        XObject=Dictionary(Unsupported=unsupported),
    )
    page = _page(
        pdf,
        b"/Fm0 Do",
        Dictionary(XObject=Dictionary(Fm0=form)),
        size=(200, 100),
    )
    original_page = bytes(resolve_indirect(page.obj["/Contents"]).read_bytes())

    with pytest.raises(
        ConversionError,
        match="XObject subtype is unsupported",
    ):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(resolve_indirect(page.obj["/Contents"]).read_bytes()) == original_page
    assert "/StructTreeRoot" not in pdf.Root


def test_semantic_flate_bomb_is_rejected_before_public_preflight_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    page.obj["/Contents"].write(
        zlib.compress(b" " * 4_096 + b"q Q"),
        filter=Name.FlateDecode,
    )
    monkeypatch.setattr(
        digital_layout,
        "_MAX_DECODED_CONTENT_BYTES_PER_CONTAINER",
        128,
    )

    def unexpected_spool(*_args, **_kwargs):
        pytest.fail("public semantic preflight serialized before the byte guard")

    monkeypatch.setattr(tagging, "SpooledTemporaryFile", unexpected_spool)

    with pytest.raises(ConversionError, match="container byte budget exceeded"):
        ensure_logical_structure(pdf, semantic=True)

    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("case", ["unsupported_xobject", "operandless_text"])
def test_semantic_direct_indices_fail_closed_when_pdfminer_would_ignore_operator(
    case: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    resources = Dictionary(Font=Dictionary(F1=_font(pdf)))
    if case == "unsupported_xobject":
        ignored = pdf.make_stream(b"ignored")
        ignored["/Subtype"] = Name("/PS")
        image = pdf.make_stream(b"\x00\x00\x00")
        image["/Subtype"] = Name.Image
        image["/Width"] = 1
        image["/Height"] = 1
        image["/ColorSpace"] = Name.DeviceRGB
        image["/BitsPerComponent"] = 8
        resources["/XObject"] = Dictionary(PS0=ignored, Im0=image)
        content = b"/Artifact BMC /PS0 Do EMC /Im0 Do"
    else:
        content = b"BT /F1 10 Tf /Artifact BMC Tj EMC 10 20 Td (Visible) Tj ET"
    page = _page(pdf, content, resources, size=(200, 100))
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="provenance|subtype is unsupported"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_empty_form_resources_inherit_page_resources_for_semantics() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"BT /F1 10 Tf 10 20 Td (Inherited resources.) Tj ET",
        Dictionary(),
        bbox=(0, 0, 150, 40),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(),
        bbox=(0, 0, 150, 40),
    )
    _page(
        pdf,
        b"/Outer Do",
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Inner=inner, Outer=outer),
        ),
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    paragraphs = [item for item in _structure_objects(pdf) if item.get("/S") == Name.P]
    assert len(paragraphs) == 1


@pytest.mark.parametrize("render_mode", [4, 5, 6, 7])
def test_described_vector_after_text_clip_fails_closed(render_mode: int) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            f"BT /F1 20 Tf {render_mode} Tr 150 50 Td (X) Tj ET ".encode()
            + b"/Figure <</ActualText (Invisible square)>> BDC "
            b"0 0 20 20 re f EMC"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="non-rectangular clip"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_scoped_text_clip_does_not_affect_later_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q BT /F1 20 Tf 7 Tr 150 50 Td (X) Tj ET Q "
            b"/Figure <</ActualText (Visible square)>> BDC "
            b"20 20 20 20 re f EMC"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert (
        str(
            next(
                item
                for item in _structure_objects(pdf)
                if item.get("/S") == Name.Figure
            )["/ActualText"]
        )
        == "Visible square"
    )


def test_empty_text_show_does_not_change_clip() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"BT /F1 20 Tf 7 Tr 150 50 Td () Tj ET "
            b"/Figure <</ActualText (Visible square)>> BDC "
            b"20 20 20 20 re f EMC"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_form_inherits_text_clipping_mode_for_vector_provenance() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        (
            b"BT /F1 20 Tf 150 50 Td (X) Tj ET "
            b"/Figure <</ActualText (Invisible square)>> BDC "
            b"0 0 20 20 re f EMC"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(0, 0, 200, 100),
    )
    page = _page(
        pdf,
        b"7 Tr /Fm Do",
        Dictionary(Font=Dictionary(F1=font), XObject=Dictionary(Fm=form)),
        size=(200, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="non-rectangular clip"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("dash", [b"[5 100] 10 d", b"[0 100] 0 d"])
def test_described_dashed_stroke_fails_closed(dash: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Figure <</ActualText (Dashed line)>> BDC "
        + dash
        + b" 20 20 m 30 20 l S EMC",
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_definite_fill_can_prove_described_dashed_combined_path() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"/Figure <</ActualText (Filled square)>> BDC "
            b"[5 100] 10 d 20 20 20 20 re B EMC"
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


@pytest.mark.parametrize(
    "stroke",
    [
        b"20 w 0 J 0 5 m 10 5 l S",
        b"10 w 1 J 1 5 m 9 5 l S",
    ],
)
def test_described_stroke_envelope_must_fit_clip(stroke: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"0 0 10 10 re W n /Figure <</ActualText (Clipped stroke)>> BDC "
        + stroke
        + b" EMC",
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="partially clipped"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_solid_stroke_envelope_inside_clip_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"0 0 20 20 re W n /Figure <</ActualText (Contained stroke)>> BDC "
            b"2 w 0 J 5 10 m 15 10 l S EMC"
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_transparent_stroke_does_not_prove_degenerate_combined_fill() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"/TransparentStroke gs /Figure <</ActualText (No paint)>> BDC "
            b"10 10 m 20 10 l B EMC"
        ),
        Dictionary(ExtGState=Dictionary(TransparentStroke=Dictionary(CA=0))),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.Figure for item in _structure_objects(pdf))


def test_transparent_fill_does_not_distort_combined_stroke_geometry() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"0 0 20 20 re W n /TransparentFill gs "
            b"/Figure <</ActualText (Visible line)>> BDC "
            b"2 w 5 10 m 15 10 l B EMC"
        ),
        Dictionary(ExtGState=Dictionary(TransparentFill=Dictionary(ca=0))),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_form_shading_bbox_outside_invocation_clip_is_omitted() -> None:
    pdf = pikepdf.Pdf.new()
    function = pdf.make_indirect(
        Dictionary(
            FunctionType=2,
            Domain=Array([0, 1]),
            C0=Array([0, 0, 0]),
            C1=Array([1, 1, 1]),
            N=1,
        )
    )
    shading = pdf.make_indirect(
        Dictionary(
            ShadingType=2,
            ColorSpace=Name.DeviceRGB,
            Coords=Array([100, 100, 200, 200]),
            Function=function,
            Extend=Array([False, False]),
            BBox=Array([100, 100, 200, 200]),
        )
    )
    form = _form(
        pdf,
        b"/Sh0 sh",
        Dictionary(Shading=Dictionary(Sh0=shading)),
        bbox=(0, 0, 200, 200),
    )
    _page(
        pdf,
        b"/Figure <</ActualText (Off-box gradient)>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(200, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert not any(item.get("/S") == Name.Figure for item in _structure_objects(pdf))


@pytest.mark.parametrize(
    ("parameters", "operator"),
    [
        (Dictionary(op=True, OPM=1), b"10 10 20 20 re f"),
        (Dictionary(OP=True, OPM=1), b"2 w 10 20 m 30 20 l S"),
    ],
)
def test_described_overprint_vector_fails_closed(
    parameters: Dictionary,
    operator: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/GS gs /Figure <</ActualText (No ink)>> BDC " + operator + b" EMC",
        Dictionary(ExtGState=Dictionary(GS=parameters)),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_scoped_overprint_state_is_restored_for_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (b"q /GS gs Q /Figure <</ActualText (Visible fill)>> BDC 10 10 20 20 re f EMC"),
        Dictionary(ExtGState=Dictionary(GS=Dictionary(op=True, OPM=1))),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_form_inherits_overprint_state_for_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"/Figure <</ActualText (No ink)>> BDC 10 10 20 20 re f EMC",
        Dictionary(),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/GS gs /Fm Do",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(op=True, OPM=1)),
            XObject=Dictionary(Fm=form),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_empty_pattern_cannot_prove_described_fill() -> None:
    pdf = pikepdf.Pdf.new()
    pattern = pdf.make_stream(b"")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary()
    page = _page(
        pdf,
        (
            b"/Pattern cs /P0 scn /Figure <</ActualText (Empty pattern)>> BDC "
            b"10 10 20 20 re f EMC"
        ),
        Dictionary(Pattern=Dictionary(P0=pattern)),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_scoped_pattern_color_is_restored_to_device_color() -> None:
    pdf = pikepdf.Pdf.new()
    pattern = pdf.make_stream(b"")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary()
    _page(
        pdf,
        (
            b"q /Pattern cs /P0 scn Q 0 g "
            b"/Figure <</ActualText (Solid fill)>> BDC 10 10 20 20 re f EMC"
        ),
        Dictionary(Pattern=Dictionary(P0=pattern)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


@pytest.mark.parametrize("blend_mode", [Name.Multiply, Name.Screen, Name.Difference])
def test_described_non_normal_blend_vector_fails_closed(
    blend_mode: Name,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (b"/GS gs /Figure <</ActualText (Neutral blend)>> BDC 10 10 20 20 re f EMC"),
        Dictionary(ExtGState=Dictionary(GS=Dictionary(BM=blend_mode))),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_normal_blend_after_scoped_complex_blend_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q /Multiply gs Q /Normal gs "
            b"/Figure <</ActualText (Normal fill)>> BDC 10 10 20 20 re f EMC"
        ),
        Dictionary(
            ExtGState=Dictionary(
                Multiply=Dictionary(BM=Name.Multiply),
                Normal=Dictionary(BM=Name.Normal),
            )
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_form_inherits_complex_blend_for_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"/Figure <</ActualText (Neutral blend)>> BDC 10 10 20 20 re f EMC",
        Dictionary(),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/GS gs /Fm Do",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
            XObject=Dictionary(Fm=form),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_semantic_text_under_overprint_fails_closed_but_artifact_is_allowed() -> None:
    semantic_pdf = pikepdf.Pdf.new()
    semantic_page = _page(
        semantic_pdf,
        b"/GS gs BT /F1 10 Tf 10 20 Td (No ink) Tj ET",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(op=True, OPM=1)),
            Font=Dictionary(F1=_font(semantic_pdf)),
        ),
        size=(100, 100),
    )
    original = bytes(semantic_page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(semantic_pdf, semantic=True)

    assert bytes(semantic_page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in semantic_pdf.Root

    artifact_pdf = pikepdf.Pdf.new()
    _page(
        artifact_pdf,
        b"/Artifact BMC /GS gs BT /F1 10 Tf 10 20 Td (Decoration) Tj ET EMC",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(op=True, OPM=1)),
            Font=Dictionary(F1=_font(artifact_pdf)),
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(artifact_pdf, semantic=True)

    assert result["semantic_content_items"] == 0


def test_described_image_under_complex_blend_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    page = _page(
        pdf,
        (
            b"/GS gs /Figure <</ActualText (Neutral image)>> BDC "
            b"q 20 0 0 20 10 10 cm /Im Do Q EMC"
        ),
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
            XObject=Dictionary(Im=image),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_intrinsically_invisible_image_is_tagged_as_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    mask = _image(pdf, b"\x00", color_space=Name.DeviceGray)
    image = _image(pdf)
    image["/SMask"] = mask
    page = _page(
        pdf,
        b"q 20 0 0 20 10 10 cm /Im Do Q",
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert "/Figure" not in _roles(pdf)
    assert ("/Artifact", "/Layout", None, None) in _marked_content(page)


def test_undescribed_intrinsically_uncertain_image_is_retained_for_review() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    _page(
        pdf,
        b"q 20 0 0 20 10 10 cm /Im Do Q",
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_alternatives_review_required"] == 1
    assert "/Figure" in _roles(pdf)


def test_described_intrinsically_uncertain_image_fails_atomically() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    page = _page(
        pdf,
        (b"/Figure <</Alt (Meaningful image)>> BDC q 20 0 0 20 10 10 cm /Im Do Q EMC"),
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain intrinsic visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_form_description_of_intrinsically_uncertain_image_fails_atomically() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    form = _form(
        pdf,
        b"q 20 0 0 20 10 10 cm /Im Do Q",
        Dictionary(XObject=Dictionary(Im=image)),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Figure <</Alt (Meaningful Form)>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain intrinsic visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_undescribed_image_under_complex_blend_still_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf)
    page = _page(
        pdf,
        b"/GS gs q 20 0 0 20 10 10 cm /Im Do Q",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
            XObject=Dictionary(Im=image),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_intrinsic_uncertainty_does_not_mask_complex_blend_failure() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    page = _page(
        pdf,
        b"/GS gs q 20 0 0 20 10 10 cm /Im Do Q",
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
            XObject=Dictionary(Im=image),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_invisible_image_forces_existing_figure_structure_rebuild() -> None:
    pdf = pikepdf.Pdf.new()
    mask = _image(pdf, b"\x00", color_space=Name.DeviceGray)
    image = _image(pdf)
    image["/SMask"] = mask
    page = _page(
        pdf,
        (b"/Figure <</MCID 0>> BDC q 20 0 0 20 10 10 cm /Im Do Q EMC"),
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )
    original_root = _install_figure_structure(pdf, page).objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is False
    assert result["semantic_structure_generated"] is True
    assert result["semantic_content_items"] == 0
    assert pdf.Root["/StructTreeRoot"].objgen != original_root
    assert "/Figure" not in _roles(pdf)


def test_described_uncertain_existing_figure_fails_atomically() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    page = _page(
        pdf,
        (b"/Figure <</MCID 0>> BDC q 20 0 0 20 10 10 cm /Im Do Q EMC"),
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )
    root = _install_figure_structure(pdf, page, alt_text="Meaningful image")
    original_root = root.objgen
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain intrinsic visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert pdf.Root["/StructTreeRoot"].objgen == original_root


def test_undescribed_uncertain_existing_figure_is_preserved_for_review() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    page = _page(
        pdf,
        (b"/Figure <</MCID 0>> BDC q 20 0 0 20 10 10 cm /Im Do Q EMC"),
        Dictionary(XObject=Dictionary(Im=image)),
        size=(100, 100),
    )
    original_root = _install_figure_structure(pdf, page).objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["semantic_alternatives_review_required"] == 1
    assert pdf.Root["/StructTreeRoot"].objgen == original_root


def test_existing_form_figure_ignores_artifact_intrinsic_uncertainty() -> None:
    pdf = pikepdf.Pdf.new()
    image = _uncertain_soft_mask_image(pdf)
    form = _form(
        pdf,
        (b"BT /F1 10 Tf 10 20 Td (Visible caption) Tj ET /Artifact BMC /Im Do EMC"),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            XObject=Dictionary(Im=image),
        ),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Figure <</MCID 0>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )
    root = _install_figure_structure(pdf, page, alt_text="Captioned figure")
    original_root = root.objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == original_root


@pytest.mark.parametrize(
    "content",
    [
        (
            b"0 0 38.8 100 re W n 1 0 1 0.1 0 0 cm "
            b"/Figure <</ActualText (Sheared intersection)>> BDC "
            b"2 w 0 J 20 20 m 30 10 l S EMC"
        ),
        (
            b"215 0 20 50 re W n 1 0 10 1 0 0 cm "
            b"/Figure <</ActualText (Sheared stroke)>> BDC "
            b"2 w 0 J 20 20 m 30 20 l S EMC"
        ),
    ],
)
def test_sheared_stroke_envelope_must_fit_clip(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, content, size=(300, 100))
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="partially clipped"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("painting_kind", ["text", "image", "form"])
def test_soft_mask_uncertainty_respects_artifact_provenance(
    painting_kind: str,
) -> None:
    def document(*, artifact: bool) -> tuple[pikepdf.Pdf, pikepdf.Page]:
        pdf = pikepdf.Pdf.new()
        mask = _form(pdf, b"0 0 10 10 re f", bbox=(0, 0, 10, 10))
        resources = Dictionary(
            ExtGState=Dictionary(
                Masked=Dictionary(SMask=Dictionary(S=Name.Luminosity, G=mask))
            )
        )
        if painting_kind == "text":
            resources["/Font"] = Dictionary(F1=_font(pdf))
            painting = b"BT /F1 10 Tf 10 20 Td (Masked text) Tj ET"
        elif painting_kind == "image":
            resources["/XObject"] = Dictionary(Im=_image(pdf))
            painting = b"q 20 0 0 20 10 10 cm /Im Do Q"
        else:
            form = _form(
                pdf,
                b"BT /F1 10 Tf 10 20 Td (Masked Form text) Tj ET",
                Dictionary(Font=Dictionary(F1=_font(pdf))),
                bbox=(0, 0, 100, 100),
            )
            resources["/XObject"] = Dictionary(Fm=form)
            painting = b"/Fm Do"
        content = b"/Masked gs " + painting
        if artifact:
            content = b"/Artifact BMC " + content + b" EMC"
        return pdf, _page(pdf, content, resources, size=(100, 100))

    semantic_pdf, semantic_page = document(artifact=False)
    original = bytes(semantic_page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="soft mask|uncertain final-paint"):
        ensure_logical_structure(semantic_pdf, semantic=True)

    assert bytes(semantic_page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in semantic_pdf.Root

    artifact_pdf, _artifact_page = document(artifact=True)
    result = ensure_logical_structure(artifact_pdf, semantic=True)

    assert result["semantic_content_items"] == 0


@pytest.mark.parametrize("painting_kind", ["text", "image", "form"])
def test_inexact_text_clip_respects_artifact_provenance(
    painting_kind: str,
) -> None:
    def document(*, artifact: bool) -> tuple[pikepdf.Pdf, pikepdf.Page]:
        pdf = pikepdf.Pdf.new()
        font = _font(pdf)
        resources = Dictionary(Font=Dictionary(F1=font))
        if painting_kind == "text":
            painting = b"BT /F1 10 Tf 10 40 Td (Clipped text) Tj ET"
        elif painting_kind == "image":
            resources["/XObject"] = Dictionary(Im=_image(pdf))
            painting = b"q 20 0 0 20 10 40 cm /Im Do Q"
        else:
            form = _form(
                pdf,
                b"BT /F1 10 Tf 10 20 Td (Clipped Form text) Tj ET",
                Dictionary(Font=Dictionary(F1=font)),
                bbox=(0, 0, 100, 100),
            )
            resources["/XObject"] = Dictionary(Fm=form)
            painting = b"/Fm Do"
        content = b"BT /F1 20 Tf 7 Tr 10 20 Td (X) Tj ET 0 Tr " + painting
        if artifact:
            content = b"/Artifact BMC " + content + b" EMC"
        return pdf, _page(pdf, content, resources, size=(100, 100))

    semantic_pdf, semantic_page = document(artifact=False)
    original = bytes(semantic_page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint"):
        ensure_logical_structure(semantic_pdf, semantic=True)

    assert bytes(semantic_page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in semantic_pdf.Root

    artifact_pdf, _artifact_page = document(artifact=True)
    result = ensure_logical_structure(artifact_pdf, semantic=True)

    assert result["semantic_content_items"] == 0


@pytest.mark.parametrize("device_state", ["hairline", "stroke_adjust"])
def test_device_dependent_described_stroke_fails_closed(
    device_state: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    if device_state == "hairline":
        state = b"0 w"
        resources = Dictionary()
        line_x = b"10.1"
    else:
        state = b"/Adjusted gs"
        resources = Dictionary(
            ExtGState=Dictionary(Adjusted=Dictionary(SA=True, LW=0.1))
        )
        line_x = b"10.6"
    page = _page(
        pdf,
        (
            b"0 0 10 100 re W n "
            + state
            + b" /Figure <</ActualText (Device stroke)>> BDC "
            + line_x
            + b" 20 m "
            + line_x
            + b" 80 l S EMC"
        ),
        resources,
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("device_state", ["hairline", "stroke_adjust"])
def test_artifact_allows_device_dependent_stroke(device_state: str) -> None:
    pdf = pikepdf.Pdf.new()
    if device_state == "hairline":
        state = b"0 w"
        resources = Dictionary()
    else:
        state = b"/Adjusted gs"
        resources = Dictionary(ExtGState=Dictionary(Adjusted=Dictionary(SA=True)))
    _page(
        pdf,
        b"/Artifact BMC " + state + b" 10 20 m 10 80 l S EMC",
        resources,
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0


@pytest.mark.parametrize("device_state", ["hairline", "stroke_adjust"])
def test_definite_fill_proves_device_dependent_combined_path(
    device_state: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    if device_state == "hairline":
        state = b"0 w"
        resources = Dictionary()
    else:
        state = b"/Adjusted gs"
        resources = Dictionary(ExtGState=Dictionary(Adjusted=Dictionary(SA=True)))
    _page(
        pdf,
        (state + b" /Figure <</ActualText (Filled square)>> BDC 20 20 20 20 re B EMC"),
        resources,
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_scoped_stroke_adjust_is_restored_for_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q /Adjusted gs Q /Figure <</ActualText (Stable line)>> BDC "
            b"2 w 20 20 m 40 20 l S EMC"
        ),
        Dictionary(ExtGState=Dictionary(Adjusted=Dictionary(SA=True))),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


def test_form_inherits_stroke_adjust_for_described_vector() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"/Figure <</ActualText (Adjusted line)>> BDC 20 20 m 40 20 l S EMC",
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Adjusted gs /Fm Do",
        Dictionary(
            ExtGState=Dictionary(Adjusted=Dictionary(SA=True)),
            XObject=Dictionary(Fm=form),
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


@pytest.mark.parametrize("device_state", ["hairline", "stroke_adjust"])
def test_device_dependent_stroked_text_respects_artifact_provenance(
    device_state: str,
) -> None:
    def document(*, artifact: bool) -> tuple[pikepdf.Pdf, pikepdf.Page]:
        pdf = pikepdf.Pdf.new()
        if device_state == "hairline":
            state = b"0 w"
            extgstate = None
        else:
            state = b"/Adjusted gs"
            extgstate = Dictionary(Adjusted=Dictionary(SA=True))
        resources = Dictionary(Font=Dictionary(F1=_font(pdf)))
        if extgstate is not None:
            resources["/ExtGState"] = extgstate
        content = state + b" BT /F1 10 Tf 1 Tr 10 20 Td (Outlined) Tj ET"
        if artifact:
            content = b"/Artifact BMC " + content + b" EMC"
        return pdf, _page(pdf, content, resources, size=(100, 100))

    semantic_pdf, semantic_page = document(artifact=False)
    original = bytes(semantic_page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="uncertain final-paint visibility"):
        ensure_logical_structure(semantic_pdf, semantic=True)

    assert bytes(semantic_page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in semantic_pdf.Root

    artifact_pdf, _artifact_page = document(artifact=True)
    result = ensure_logical_structure(artifact_pdf, semantic=True)

    assert result["semantic_content_items"] == 0


def test_invalid_stroke_adjust_extgstate_fails_atomically() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        b"/Invalid gs 20 20 m 40 20 l S",
        Dictionary(ExtGState=Dictionary(Invalid=Dictionary(SA=1))),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_retraced_even_odd_described_fill_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"/Figure <</ActualText (Cancelled)>> BDC "
            b"10 10 m 30 10 l 30 30 l 10 30 l 10 10 l "
            b"30 10 l 30 30 l 10 30 l 10 10 l f* EMC"
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="ambiguous path geometry"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_self_intersecting_described_fill_fails_closed() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"/Figure <</ActualText (Crossed)>> BDC "
            b"10 10 m 50 40 l 10 50 l 40 10 l f EMC"
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="ambiguous path geometry"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_described_exact_stroke_does_not_hide_ambiguous_fill() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(
        pdf,
        (
            b"/Figure <</ActualText (Outlined cancellation)>> BDC "
            b"10 10 20 20 re 10 10 20 20 re B* EMC"
        ),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="ambiguous path geometry"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_simple_concave_described_fill_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"/Figure <</ActualText (Concave)>> BDC "
            b"10 10 m 50 10 l 50 50 l 30 30 l 10 50 l f* EMC"
        ),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1


@pytest.mark.parametrize(
    "content",
    [
        b"10 10 20 20 re 10 10 20 20 re f*",
        (b"10 10 m 30 10 l 30 30 l 10 30 l 10 10 l 30 10 l 30 30 l 10 30 l 10 10 l f*"),
        b"10 10 m 50 40 l 10 50 l 40 10 l f",
    ],
    ids=["identical-even-odd-rectangles", "retraced-even-odd", "self-intersecting"],
)
def test_ambiguous_unclassified_form_fill_becomes_layout_artifact(
    content: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, content, bbox=(0, 0, 100, 100))
    page = _page(
        pdf,
        b"/Fm Do",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["semantic_vector_review_required"] == 1
    assert result["artifacts_tagged"] == 1
    assert "/Figure" not in _roles(pdf)
    assert _marked_content(page) == [("/Artifact", "/Layout", None, None)]


def test_ambiguous_form_fill_is_not_overridden_by_later_exact_geometry() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"10 10 20 20 re 10 10 20 20 re f* 60 60 20 20 re f",
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Fm Do",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["semantic_vector_review_required"] == 1
    assert "/Figure" not in _roles(pdf)
    assert _marked_content(page) == [("/Artifact", "/Layout", None, None)]


def test_form_text_survives_ambiguous_vector_artifacting() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        (
            b"BT /F1 10 Tf 10 70 Td (Semantic label) Tj ET "
            b"10 10 20 20 re 10 10 20 20 re f*"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Fm Do",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 1
    assert "/P" in _roles(pdf)
    assert "/Figure" not in _roles(pdf)
    clone = _invoked_forms(page)[0]
    paints = _marked_paint_stacks(clone)
    assert "/Span" in next(stack for operator, stack in paints if operator == "Tj")
    assert "/Artifact" in next(stack for operator, stack in paints if operator == "f*")


def test_described_ambiguous_form_fill_still_fails_atomically() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"10 10 20 20 re 10 10 20 20 re f*",
        bbox=(0, 0, 100, 100),
    )
    page = _page(
        pdf,
        b"/Figure <</ActualText (Status icon)>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )
    original = bytes(page.obj["/Contents"].read_bytes())

    with pytest.raises(ConversionError, match="described.*ambiguous path geometry"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page.obj["/Contents"].read_bytes()) == original
    assert "/StructTreeRoot" not in pdf.Root


def test_simple_concave_unclassified_form_fill_is_preserved() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(
        pdf,
        b"10 10 m 50 10 l 50 50 l 30 30 l 10 50 l f*",
        bbox=(0, 0, 100, 100),
    )
    _page(
        pdf,
        b"/Fm Do",
        Dictionary(XObject=Dictionary(Fm=form)),
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 0
    assert "/Figure" in _roles(pdf)


@pytest.mark.parametrize(
    ("policy", "expression", "visible"),
    [
        (Name.AllOn, None, False),
        (Name.AnyOn, None, True),
        (Name.AllOff, None, False),
        (Name.AnyOff, None, True),
        (Name.AnyOn, Name.And, False),
    ],
    ids=["all-on", "any-on", "all-off", "any-off", "ve-precedence"],
)
def test_optional_content_policy_controls_semantic_binding(
    policy: Name,
    expression: Name | None,
    visible: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    enabled = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Enabled")))
    disabled = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Disabled")))
    membership = Dictionary(
        Type=Name.OCMD,
        OCGs=Array([enabled, disabled]),
        P=policy,
    )
    if expression is not None:
        membership["/VE"] = Array([expression, enabled, disabled])
    _set_optional_content(pdf, (enabled, disabled), off=(disabled,))
    page = _page(
        pdf,
        (
            b"/OC /Layer BDC "
            b"BT /F1 12 Tf 20 200 Td (conditional) Tj ET EMC\n"
            b"BT /F1 12 Tf 20 160 Td (always visible) Tj ET"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=membership),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1 + int(visible)
    paints = _marked_paint_stacks(page)
    assert "/OC" in paints[0][1]
    assert ("/Span" in paints[0][1]) is visible
    assert ("/Artifact" in paints[0][1]) is not visible
    assert "/Span" in paints[1][1]
    properties = resolve_indirect(page.obj["/Resources"]["/Properties"])
    assert resolve_indirect(properties["/Layer"]) == membership


def test_optional_content_all_intent_is_active_for_default_view() -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(
        Dictionary(Type=Name.OCG, Name=String("All intents"), Intent=Name.All)
    )
    _set_optional_content(pdf, (group,))
    page = _page(
        pdf,
        b"/OC /Layer BDC BT /F1 12 Tf 20 200 Td (visible) Tj ET EMC",
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert "/Span" in _marked_paint_stacks(page)[0][1]


def test_nested_named_and_inline_optional_content_scopes_are_conjunctive() -> None:
    pdf = pikepdf.Pdf.new()
    disabled = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Disabled")))
    _set_optional_content(pdf, (disabled,), off=(disabled,))
    inline_membership = Dictionary(
        Type=Name.OCMD,
        P=Name.AnyOn,
    )
    inline_marker = pikepdf.unparse_content_stream(
        [
            pikepdf.ContentStreamInstruction(
                [Name.OC, inline_membership],
                pikepdf.Operator("BDC"),
            )
        ]
    )
    page = _page(
        pdf,
        (
            b"/OC /Outer BDC\n"
            b"BT /F1 12 Tf 20 200 Td (outer one) Tj\n"
            + inline_marker
            + b"(inner hidden) Tj\nEMC\n"
            b"(outer two) Tj ET\nEMC\n"
            b"BT /F1 12 Tf 20 150 Td (outside visible) Tj ET"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Outer=disabled),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    paints = _marked_paint_stacks(page)
    assert len(paints) == 4
    assert paints[0][1].count("/OC") == 1
    assert "/Artifact" in paints[0][1]
    assert paints[1][1].count("/OC") == 2
    assert "/Artifact" in paints[1][1]
    assert paints[2][1].count("/OC") == 1
    assert "/Artifact" in paints[2][1]
    assert "/OC" not in paints[3][1]
    assert "/Span" in paints[3][1]


@pytest.mark.parametrize("kind", ["image", "nested-form"])
def test_hidden_xobject_optional_content_is_artifacted(kind: str) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    if kind == "image":
        hidden = _image(pdf)
        hidden["/OC"] = group
        invoked = hidden
    else:
        hidden = _form(
            pdf,
            b"BT /F1 12 Tf 20 100 Td (hidden form text) Tj ET",
            Dictionary(Font=Dictionary(F1=_font(pdf))),
        )
        hidden["/OC"] = group
        invoked = _form(
            pdf,
            (b"BT /F2 12 Tf 20 140 Td (visible outer form text) Tj ET\n/Hidden Do"),
            Dictionary(
                Font=Dictionary(F2=_font(pdf)),
                XObject=Dictionary(Hidden=hidden),
            ),
        )
    page = _page(
        pdf,
        b"BT /F1 12 Tf 20 200 Td (visible text) Tj ET\n/Invoked Do",
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            XObject=Dictionary(Invoked=invoked),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == (1 if kind == "image" else 2)
    assert "/Figure" not in _roles(pdf)
    assert resolve_indirect(hidden["/OC"]).objgen == group.objgen
    if kind == "image":
        do_stack = next(
            stack for operator, stack in _marked_paint_stacks(page) if operator == "Do"
        )
    else:
        page_do = next(
            instruction
            for instruction in pikepdf.parse_content_stream(page)
            if isinstance(instruction, pikepdf.ContentStreamInstruction)
            and instruction.operator == pikepdf.Operator("Do")
        )
        page_resources = resolve_indirect(page.obj["/Resources"])
        outer = resolve_indirect(
            page_resources["/XObject"][resolve_indirect(page_do.operands[0])]
        )
        assert isinstance(outer, pikepdf.Stream)
        do_stack = next(
            stack for operator, stack in _marked_paint_stacks(outer) if operator == "Do"
        )
    assert "/Artifact" in do_stack


def test_hidden_optional_content_vector_paint_gets_a_generated_artifact() -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    page = _page(
        pdf,
        (
            b"/OC /Layer BDC 10 10 80 40 re f EMC\n"
            b"BT /F1 12 Tf 20 200 Td (visible text) Tj ET"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
        ),
        size=(300, 300),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 1
    assert result["semantic_vector_review_required"] == 0
    assert result["artifacts_tagged"] == 1
    vector_stack = next(
        stack for operator, stack in _marked_paint_stacks(page) if operator == "f"
    )
    assert "/OC" in vector_stack
    assert "/Artifact" in vector_stack
    oc_marker = next(
        instruction
        for instruction in pikepdf.parse_content_stream(page)
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
        and instruction.operator == pikepdf.Operator("BDC")
        and resolve_indirect(instruction.operands[0]) == Name.OC
    )
    assert resolve_indirect(oc_marker.operands[1]) == Name("/Layer")


def test_hidden_annotations_are_excluded_before_link_and_widget_review() -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    page = _page(
        pdf,
        b"BT /F1 12 Tf 20 200 Td (visible link text) Tj ET",
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(300, 300),
    )
    hidden_link = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([15, 190, 180, 215]),
            NM=String("shared-name"),
            StructParent=97,
            OC=group,
        )
    )
    hidden_widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([20, 80, 180, 110]),
            StructParent=98,
            OC=group,
        )
    )
    visible_link = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([15, 190, 180, 215]),
            NM=String("shared-name"),
        )
    )
    page.obj["/Annots"] = Array([hidden_link, hidden_widget, visible_link])
    register_form_widget(pdf, hidden_widget)

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["annotations_tagged"] == 1
    assert result["semantic_link_review_required"] == 0
    assert result["semantic_form_review_required"] == 0
    assert "/StructParent" not in hidden_link
    assert "/StructParent" not in hidden_widget
    assert "/TU" not in hidden_widget
    assert str(visible_link["/NM"]) == "shared-name"
    referenced_annotations = {
        resolve_indirect(item["/Obj"]).objgen
        for item in _structure_objects(pdf)
        if item.get("/Type") == Name.OBJR
    }
    assert hidden_link.objgen not in referenced_annotations
    assert hidden_widget.objgen not in referenced_annotations
    assert visible_link.objgen in referenced_annotations


@pytest.mark.parametrize("hidden", [False, True], ids=["visible", "hidden"])
def test_existing_structure_is_preserved_only_for_visible_oc_mcrs(
    hidden: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer")))
    _set_optional_content(pdf, (group,), off=(group,) if hidden else ())
    page = _page(
        pdf,
        (
            b"/OC /Layer BDC /P <</MCID 0>> BDC "
            b"BT /F1 12 Tf 20 200 Td (conditional paragraph) Tj ET EMC EMC\n"
            b"/P <</MCID 1>> BDC "
            b"BT /F1 12 Tf 20 160 Td (visible paragraph) Tj ET EMC"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
        ),
        size=(300, 300),
    )
    document, paragraphs = _install_two_paragraph_structure(pdf, page, (0, 1))
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    original_root = root.objgen
    original_elements = (
        document.objgen,
        *(paragraph.objgen for paragraph in paragraphs),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is not hidden
    assert result["structure_rebuilt"] is hidden
    current_root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    if hidden:
        assert current_root.objgen != original_root
        assert result["semantic_content_items"] == 1
        assert "/Artifact" in _marked_paint_stacks(page)[0][1]
    else:
        assert current_root.objgen == original_root
        assert (
            document.objgen,
            *(paragraph.objgen for paragraph in paragraphs),
        ) == original_elements


@pytest.mark.parametrize("hidden", [False, True], ids=["visible", "hidden"])
def test_existing_objr_is_preserved_only_when_optional_content_is_visible(
    hidden: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer")))
    _set_optional_content(pdf, (group,), off=(group,) if hidden else ())
    page = _page(
        pdf,
        b"/P <</MCID 0>> BDC BT /F1 12 Tf 20 200 Td (body) Tj ET EMC",
        Dictionary(Font=Dictionary(F1=_font(pdf))),
        size=(300, 300),
    )
    page.obj["/StructParents"] = 0
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([15, 190, 180, 215]),
            StructParent=1,
            OC=group,
        )
    )
    page.obj["/Annots"] = Array([annotation])
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    paragraph = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.P, P=document, Pg=page.obj, K=0)
    )
    link = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Link, P=document, Pg=page.obj)
    )
    link["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=annotation)
    )
    document["/K"] = Array([paragraph, link])
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([paragraph])
    parent_tree[1] = link
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 2
    pdf.Root["/StructTreeRoot"] = root
    original_root = root.objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is not hidden
    assert result["structure_rebuilt"] is hidden
    current_root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    if hidden:
        assert current_root.objgen != original_root
        assert result["annotations_tagged"] == 0
        assert "/StructParent" not in annotation
        assert "/Link" not in _roles(pdf)
    else:
        assert current_root.objgen == original_root
        assert int(annotation["/StructParent"]) == 1
        assert link.objgen in {
            item.objgen
            for item in _structure_objects(pdf)
            if item.get("/S") == Name.Link
        }


@pytest.mark.parametrize(
    "failure",
    ["malformed-policy", "cyclic-expression", "expression-budget"],
)
def test_invalid_optional_content_fails_semantic_tagging_atomically(
    failure: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer")))
    _set_optional_content(pdf, (group,))
    membership = pdf.make_indirect(Dictionary(Type=Name.OCMD, OCGs=Array([group])))
    if failure == "malformed-policy":
        membership["/P"] = Name("/Unsupported")
    elif failure == "cyclic-expression":
        expression = pdf.make_indirect(Array([Name.Not]))
        expression.append(expression)
        membership["/VE"] = expression
    else:
        membership["/VE"] = Array([Name.And, *([group] * 8_192)])
    page = _page(
        pdf,
        b"/OC /Layer BDC BT /F1 12 Tf 20 200 Td (conditional) Tj ET EMC",
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=membership),
        ),
        size=(300, 300),
    )
    content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(content, pikepdf.Stream)
    original_content = bytes(content.read_bytes())
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_membership = bytes(membership.unparse())
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="optional.content"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(content.read_bytes()) == original_content
    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert bytes(membership.unparse()) == original_membership
    assert len(pdf.objects) == original_object_count


@pytest.mark.parametrize(
    "painting_kind",
    ["text", "inline-image", "image", "form"],
)
def test_hidden_optional_content_skips_final_paint_uncertainty(
    painting_kind: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    resources = Dictionary(
        ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
        Properties=Dictionary(Layer=group),
    )
    if painting_kind == "text":
        resources["/Font"] = Dictionary(F1=_font(pdf))
        paint = b"BT /F1 10 Tf 10 20 Td (Hidden text) Tj ET"
    elif painting_kind == "inline-image":
        paint = (
            b"q 20 0 0 20 10 10 cm BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI Q"
        )
    elif painting_kind == "image":
        resources["/XObject"] = Dictionary(Im=_image(pdf))
        paint = b"q 20 0 0 20 10 10 cm /Im Do Q"
    else:
        form = _form(
            pdf,
            b"BT /F1 10 Tf 10 20 Td (Hidden Form text) Tj ET",
            Dictionary(Font=Dictionary(F1=_font(pdf))),
            bbox=(0, 0, 100, 100),
        )
        resources["/XObject"] = Dictionary(Fm=form)
        paint = b"/Fm Do"
    page = _page(
        pdf,
        b"/OC /Layer BDC /GS gs " + paint + b" EMC",
        resources,
        size=(100, 100),
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["artifacts_tagged"] == 1
    paints = _marked_paint_stacks(page)
    assert len(paints) == 1
    assert paints[0][1].count("/OC") == 1
    assert paints[0][1].count("/Artifact") == 1


@pytest.mark.parametrize("artifact_outermost", [False, True])
def test_source_artifact_precedes_hidden_optional_content(
    artifact_outermost: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    if artifact_outermost:
        content = (
            b"/Artifact BMC /OC /Layer BDC /GS gs "
            b"BT /F1 10 Tf 10 20 Td (Decoration) Tj ET EMC EMC"
        )
    else:
        content = (
            b"/OC /Layer BDC /Artifact BMC /GS gs "
            b"BT /F1 10 Tf 10 20 Td (Decoration) Tj ET EMC EMC"
        )
    page = _page(
        pdf,
        content,
        Dictionary(
            ExtGState=Dictionary(GS=Dictionary(BM=Name.Multiply)),
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
        ),
        size=(100, 100),
    )
    properties = resolve_indirect(page.obj["/Resources"]["/Properties"])
    original_group = resolve_indirect(properties["/Layer"]).objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["semantic_content_items"] == 0
    assert result["artifacts_tagged"] == 0
    paints = _marked_paint_stacks(page)
    assert len(paints) == 1
    assert paints[0][1].count("/OC") == 1
    assert paints[0][1].count("/Artifact") == 1
    properties = resolve_indirect(page.obj["/Resources"]["/Properties"])
    assert resolve_indirect(properties["/Layer"]).objgen == original_group


@pytest.mark.parametrize("hidden", [False, True], ids=["visible", "hidden"])
@pytest.mark.parametrize("nesting_depth", [1, 2], ids=["direct-form", "nested-form"])
def test_existing_page_mcr_tracks_optional_content_inside_forms(
    hidden: bool,
    nesting_depth: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Layer")))
    _set_optional_content(pdf, (group,), off=(group,) if hidden else ())
    invoked = _form(
        pdf,
        (
            b"/OC /Layer BDC "
            b"BT /F1 10 Tf 10 20 Td (Conditional text) Tj ET EMC "
            b"BT /F1 10 Tf 10 40 Td (Visible text) Tj ET"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
        ),
        bbox=(0, 0, 100, 100),
    )
    for _ in range(nesting_depth - 1):
        invoked = _form(
            pdf,
            b"/Nested Do",
            Dictionary(XObject=Dictionary(Nested=invoked)),
            bbox=(0, 0, 100, 100),
        )
    page = _page(
        pdf,
        b"/Figure <</MCID 0>> BDC /Fm Do EMC",
        Dictionary(XObject=Dictionary(Fm=invoked)),
        size=(100, 100),
    )
    root = _install_figure_structure(pdf, page, alt_text="Form content")
    original_root = root.objgen

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is not hidden
    assert result["structure_rebuilt"] is hidden
    if hidden:
        assert resolve_indirect(pdf.Root["/StructTreeRoot"]).objgen != original_root
        assert result["semantic_content_items"] == 1
    else:
        assert resolve_indirect(pdf.Root["/StructTreeRoot"]).objgen == original_root


@pytest.mark.parametrize(
    "failure",
    ["malformed-type", "cyclic-expression", "expression-budget"],
)
def test_fully_hidden_form_optional_content_is_validated_atomically(
    failure: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    if failure == "malformed-type":
        membership = pdf.make_indirect(Dictionary(Type=Name("/Bad")))
    else:
        membership = pdf.make_indirect(Dictionary(Type=Name.OCMD, OCGs=Array([group])))
        if failure == "cyclic-expression":
            expression = pdf.make_indirect(Array([Name.Not]))
            expression.append(expression)
            membership["/VE"] = expression
        else:
            membership["/VE"] = Array([Name.And, *([group] * 8_192)])
    inner = _form(
        pdf,
        b"/OC /Bad BDC BT /F1 10 Tf 10 20 Td (Invalid hidden text) Tj ET EMC",
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Bad=membership),
        ),
        bbox=(0, 0, 100, 100),
    )
    outer = _form(
        pdf,
        b"/Inner Do",
        Dictionary(XObject=Dictionary(Inner=inner)),
        bbox=(0, 0, 100, 100),
    )
    outer["/OC"] = group
    page = _page(
        pdf,
        b"/Outer Do",
        Dictionary(XObject=Dictionary(Outer=outer)),
        size=(100, 100),
    )
    page_content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(page_content, pikepdf.Stream)
    original_page_content = bytes(page_content.read_bytes())
    original_inner_content = bytes(inner.read_bytes())
    original_outer_content = bytes(outer.read_bytes())
    original_root_keys = set(pdf.Root.keys())
    original_page_keys = set(page.obj.keys())
    original_inner_keys = set(inner.keys())
    original_outer_keys = set(outer.keys())
    original_membership = bytes(membership.unparse())
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="optional.content"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page_content.read_bytes()) == original_page_content
    assert bytes(inner.read_bytes()) == original_inner_content
    assert bytes(outer.read_bytes()) == original_outer_content
    assert set(pdf.Root.keys()) == original_root_keys
    assert set(page.obj.keys()) == original_page_keys
    assert set(inner.keys()) == original_inner_keys
    assert set(outer.keys()) == original_outer_keys
    assert bytes(membership.unparse()) == original_membership
    assert len(pdf.objects) == original_object_count


def test_hidden_existing_mcr_does_not_skip_later_oc_validation() -> None:
    pdf = pikepdf.Pdf.new()
    group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String("Hidden")))
    _set_optional_content(pdf, (group,), off=(group,))
    malformed = pdf.make_indirect(Dictionary(Type=Name("/Bad")))
    invalid_form = _form(
        pdf,
        b"/OC /Bad BDC BT /F2 10 Tf 10 20 Td (Invalid) Tj ET EMC",
        Dictionary(
            Font=Dictionary(F2=_font(pdf)),
            Properties=Dictionary(Bad=malformed),
        ),
        bbox=(0, 0, 100, 100),
    )
    invalid_form["/OC"] = group
    page = _page(
        pdf,
        (
            b"/OC /Layer BDC /Figure <</MCID 0>> BDC "
            b"BT /F1 10 Tf 10 60 Td (Hidden reference) Tj ET EMC EMC "
            b"/Invalid Do"
        ),
        Dictionary(
            Font=Dictionary(F1=_font(pdf)),
            Properties=Dictionary(Layer=group),
            XObject=Dictionary(Invalid=invalid_form),
        ),
        size=(100, 100),
    )
    root = _install_figure_structure(pdf, page, alt_text="Hidden reference")
    page_content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(page_content, pikepdf.Stream)
    original_page_content = bytes(page_content.read_bytes())
    original_form_content = bytes(invalid_form.read_bytes())
    original_root = root.objgen
    original_object_count = len(pdf.objects)

    with pytest.raises(ConversionError, match="optional.content"):
        ensure_logical_structure(pdf, semantic=True)

    assert bytes(page_content.read_bytes()) == original_page_content
    assert bytes(invalid_form.read_bytes()) == original_form_content
    assert resolve_indirect(pdf.Root["/StructTreeRoot"]).objgen == original_root
    assert len(pdf.objects) == original_object_count
