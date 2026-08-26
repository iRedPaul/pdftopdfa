# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for PDF/A level A logical structure handling."""

import re
from io import BytesIO
from unittest.mock import patch

import pikepdf
import pytest
from conftest import new_pdf, open_pdf
from pikepdf import Array, Dictionary, Name, NameTree, NumberTree, String

import pdftopdfa.tagging as tagging
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.semantics import ContentReference, StructureNode
from pdftopdfa.tagging import ensure_logical_structure
from pdftopdfa.utils import resolve_indirect


def _add_page(
    pdf: pikepdf.Pdf,
    content: bytes | list[bytes] | None = None,
    *,
    resources: Dictionary | None = None,
) -> pikepdf.Page:
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
    )
    if content is not None:
        if isinstance(content, bytes):
            page_dict["/Contents"] = pdf.make_stream(content)
        else:
            page_dict["/Contents"] = Array(
                [pdf.make_stream(stream) for stream in content]
            )
    if resources is not None:
        page_dict["/Resources"] = resources
    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)
    return page


def test_pdfua_demotes_inferred_nonrectangular_table() -> None:
    table = StructureNode(
        "Table",
        children=(
            StructureNode(
                "TR",
                children=(
                    StructureNode("TH", content=(ContentReference("h1"),)),
                    StructureNode("TH", content=(ContentReference("h2"),)),
                ),
            ),
            StructureNode(
                "TR",
                children=(StructureNode("TD", content=(ContentReference("v1"),)),),
            ),
        ),
    )
    document = StructureNode("Document", children=(table,))

    normalized, review_required = tagging._normalize_pdfua_plan_tables(document)

    assert review_required == 1
    assert [node.role for node in normalized.walk()] == [
        "Document",
        "Div",
        "Div",
        "P",
        "P",
        "Div",
        "P",
    ]
    assert [
        reference.span_id for node in normalized.walk() for reference in node.content
    ] == ["h1", "h2", "v1"]


def _nested_form_chain(pdf: pikepdf.Pdf, depth: int) -> pikepdf.Stream:
    form = pdf.make_stream(b"/Artifact BMC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    for _ in range(depth):
        parent = pdf.make_stream(b"/Fm Do")
        parent["/Type"] = Name.XObject
        parent["/Subtype"] = Name.Form
        parent["/BBox"] = Array([0, 0, 10, 10])
        parent["/Resources"] = Dictionary(XObject=Dictionary(Fm=form))
        form = parent
    return form


def _install_existing_structure(
    pdf: pikepdf.Pdf,
    *,
    role: Name = Name.Document,
    role_map: Dictionary | None = None,
    language: object | None = None,
) -> tuple[Dictionary, Dictionary]:
    page = pdf.pages[0]
    page_key = int(page.obj.get("/StructParents", 0))
    page.obj["/StructParents"] = page_key
    content = resolve_indirect(page.obj.get("/Contents"))
    if isinstance(content, pikepdf.Stream):
        content.write(
            b"/Document <</MCID 0>> BDC\n" + bytes(content.read_bytes()) + b"\nEMC\n"
        )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=role,
            P=root,
            Pg=page.obj,
            K=0,
        )
    )
    if language is not None:
        document["/Lang"] = language
    parent_tree = NumberTree.new(pdf)
    parent_tree[page_key] = Array([document])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = page_key + 1
    if role_map is not None:
        root["/RoleMap"] = role_map
    pdf.Root["/StructTreeRoot"] = root
    pdf.Root["/MarkInfo"] = Dictionary(Marked=False, UserProperties=True)
    return root, document


def _install_annotation_appearance_structure(
    pdf: pikepdf.Pdf,
) -> tuple[Dictionary, pikepdf.Stream, Dictionary, Dictionary]:
    page = pdf.pages[0]
    appearance = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    appearance["/Type"] = Name.XObject
    appearance["/Subtype"] = Name.Form
    appearance["/BBox"] = Array([0, 0, 10, 10])
    appearance["/StructParents"] = 4
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            AP=Dictionary(N=appearance),
            StructParent=5,
        )
    )
    page.obj["/Annots"] = Array([annotation])

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    appearance_element = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    mcr = pdf.make_indirect(
        Dictionary(
            Type=Name.MCR,
            Pg=page.obj,
            Stm=appearance,
            StmOwn=annotation,
            MCID=0,
        )
    )
    appearance_element["/K"] = mcr
    annotation_element = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Annot, P=document, Pg=page.obj)
    )
    annotation_element["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=annotation)
    )
    document["/K"] = Array([appearance_element, annotation_element])
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([appearance_element])
    parent_tree[5] = annotation_element
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 6
    pdf.Root["/StructTreeRoot"] = root
    return root, appearance, annotation, mcr


def _content_streams(page: pikepdf.Page) -> list[pikepdf.Stream]:
    contents = resolve_indirect(page.obj["/Contents"])
    if isinstance(contents, pikepdf.Stream):
        return [contents]
    return [resolve_indirect(item) for item in contents]


def _generated_parts(
    pdf: pikepdf.Pdf,
) -> tuple[Dictionary, Dictionary, list[Dictionary]]:
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    document = resolve_indirect(root["/K"])
    divs = [resolve_indirect(item) for item in resolve_indirect(document["/K"])]
    return root, document, divs


def _wrapper_mcid(page: pikepdf.Page) -> int:
    prefix = bytes(_content_streams(page)[0].read_bytes())
    match = re.fullmatch(rb"/Div <</MCID (\d+)>> BDC\n", prefix)
    assert match is not None
    return int(match.group(1))


def test_tags_untagged_text_page_and_preserves_native_stream() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT /F1 12 Tf (Hello) Tj ET")
    native = page.obj["/Contents"]
    native_bytes = bytes(native.read_bytes())

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert result["pages_tagged"] == 1
    assert bool(pdf.Root["/MarkInfo"]["/Marked"]) is True
    streams = _content_streams(page)
    assert len(streams) == 3
    assert streams[1].objgen == native.objgen
    assert bytes(streams[1].read_bytes()) == native_bytes
    assert bytes(streams[2].read_bytes()) == b"\nEMC\n"

    root, document, divs = _generated_parts(pdf)
    div = divs[0]
    assert str(document["/S"]) == "/Document"
    assert str(div["/S"]) == "/Div"
    assert div["/Pg"].objgen == page.obj.objgen
    assert int(page.obj["/StructParents"]) == 0
    mcid = _wrapper_mcid(page)
    assert int(div["/K"][0]) == mcid
    parent_entry = NumberTree(root["/ParentTree"])[0]
    assert parent_entry[mcid].objgen == div.objgen


def test_tags_image_only_page() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 8
    image["/StructParents"] = 9
    resources = Dictionary(XObject=Dictionary(Im0=image))
    page = _add_page(pdf, b"q 10 0 0 10 0 0 cm /Im0 Do Q", resources=resources)

    result = ensure_logical_structure(pdf)

    assert bytes(_content_streams(page)[1].read_bytes()).endswith(b"/Im0 Do Q")
    assert _wrapper_mcid(page) == 0
    assert result["stream_structure_keys_removed"] == 2
    assert "/StructParent" not in image
    assert "/StructParents" not in image


def test_tags_empty_page_with_empty_marked_content() -> None:
    pdf = new_pdf()
    page = _add_page(pdf)

    result = ensure_logical_structure(pdf)

    assert result["pages_tagged"] == 1
    assert [bytes(stream.read_bytes()) for stream in _content_streams(page)] == [
        b"/Div <</MCID 0>> BDC\n",
        b"\nEMC\n",
    ]
    assert int(page.obj["/StructParents"]) == 0


def test_tags_document_without_pages_without_empty_k_array() -> None:
    pdf = new_pdf()

    first_result = ensure_logical_structure(pdf)
    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    document = resolve_indirect(root["/K"])
    first_root = pdf.Root["/StructTreeRoot"]
    second_result = ensure_logical_structure(pdf)

    assert first_result["structure_rebuilt"] is True
    assert "/K" not in document
    assert second_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == first_root.objgen


def test_tags_multiple_pages_in_page_and_stream_order() -> None:
    pdf = new_pdf()
    first = _add_page(pdf, [b"q", b"Q"])
    second = _add_page(pdf)
    third = _add_page(pdf, b"BT ET")

    ensure_logical_structure(pdf)

    _, _, divs = _generated_parts(pdf)
    assert len(divs) == 3
    assert [int(page.obj["/StructParents"]) for page in pdf.pages] == [0, 1, 2]
    assert [bytes(stream.read_bytes()) for stream in _content_streams(first)[1:-1]] == [
        b"q",
        b"Q",
    ]
    assert len(_content_streams(second)) == 2
    assert bytes(_content_streams(third)[1].read_bytes()) == b"BT ET"


def test_preserves_valid_existing_structure_and_normalizes_invalid_lang() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    page.obj["/StructParents"] = 7
    root, document = _install_existing_structure(
        pdf,
        role=Name("/CustomDocument"),
        role_map=Dictionary(CustomDocument=Name.Document),
        language=String("not valid!"),
    )
    content = page.obj["/Contents"]
    role_map = root["/RoleMap"]

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_languages_normalized"] == 1
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert page.obj["/Contents"].objgen == content.objgen
    assert root["/RoleMap"] == role_map
    assert str(document["/Lang"]) == "und"
    assert int(page.obj["/StructParents"]) == 7
    assert bool(pdf.Root["/MarkInfo"]["/Marked"]) is True
    assert bool(pdf.Root["/MarkInfo"]["/UserProperties"]) is True


def test_preserves_valid_structure_marked_as_suspect() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    document["/ActualText"] = String("replacement text")
    pdf.Root["/MarkInfo"]["/Marked"] = True
    pdf.Root["/MarkInfo"]["/Suspects"] = True

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert result["mark_info_updated"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert str(document["/ActualText"]) == "replacement text"
    assert bool(pdf.Root["/MarkInfo"]["/Marked"]) is True
    assert "/Suspects" not in pdf.Root["/MarkInfo"]
    assert bool(pdf.Root["/MarkInfo"]["/UserProperties"]) is True


def test_preserves_structure_with_redundant_standard_role_mappings() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    role_map = Dictionary(Document=Name.Document, Span=Name.Span)
    old_root, document = _install_existing_structure(pdf, role_map=role_map)
    document["/ActualText"] = String("replacement text")

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert old_root["/RoleMap"] == role_map
    assert str(document["/ActualText"]) == "replacement text"


def test_pdfua_removes_circular_standard_identity_role_mappings() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, _document = _install_existing_structure(
        pdf,
        role_map=Dictionary(Document=Name.Document, Span=Name.Span),
    )

    result = ensure_logical_structure(pdf, pdfua=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert len(resolve_indirect(old_root["/RoleMap"])) == 0


def test_suspect_valid_structure_actualtext_still_covers_pua_content() -> None:
    from pdftopdfa.sanitizers.pua_actualtext import sanitize_pua_actualtext

    pdf = new_pdf()
    tounicode = pdf.make_stream(
        b"begincmap\n"
        b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        b"1 beginbfchar\n<E0> <E000>\nendbfchar\nendcmap"
    )
    font = Dictionary(
        Type=Name.Font,
        Subtype=Name.TrueType,
        BaseFont=Name("/TestFont"),
        Encoding=Name.WinAnsiEncoding,
        ToUnicode=tounicode,
    )
    page = _add_page(
        pdf,
        b"BT /F1 12 Tf <E0> Tj ET",
        resources=Dictionary(Font=Dictionary(F1=font)),
    )
    root, document = _install_existing_structure(pdf)
    document["/ActualText"] = String("replacement text")
    pdf.Root["/MarkInfo"]["/Suspects"] = True
    original_content = bytes(page.obj["/Contents"].read_bytes())

    pua_result = sanitize_pua_actualtext(pdf)
    tagging_result = ensure_logical_structure(pdf)

    assert pua_result["pua_actualtext_added"] == 0
    assert bytes(page.obj["/Contents"].read_bytes()) == original_content
    assert tagging_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert str(document["/ActualText"]) == "replacement text"
    assert "/Suspects" not in pdf.Root["/MarkInfo"]


def test_empty_struct_element_actualtext_covers_its_content() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    _, document = _install_existing_structure(pdf)
    document["/ActualText"] = String("")

    references = tagging.get_structural_actualtext_references(pdf)

    assert references == frozenset({(pdf.pages[0].obj.objgen, 0)})


def test_struct_tree_root_actualtext_does_not_cover_content() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, _ = _install_existing_structure(pdf)
    root["/ActualText"] = String("root replacement")

    references = tagging.get_structural_actualtext_references(pdf)

    assert references == frozenset()


def test_preserves_valid_structure_language() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    _, document = _install_existing_structure(pdf, language=String("de-DE"))

    result = ensure_logical_structure(pdf)

    assert result["structure_languages_normalized"] == 0
    assert str(document["/Lang"]) == "de-DE"


def test_preserves_structure_with_unstructured_popup_annotation() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    popup = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Popup,
            Rect=Array([0, 0, 10, 10]),
        )
    )
    page.obj["/Annots"] = Array([popup])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert resolve_indirect(root["/K"]).objgen == document.objgen
    assert "/StructParent" not in popup


def test_pdfua_preserves_printer_mark_as_untagged_incidental_artifact() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    printer_mark = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.PrinterMark,
            Rect=Array([0, 0, 10, 10]),
            StructParent=9,
        )
    )
    page.obj["/Annots"] = Array([printer_mark])

    first_result = ensure_logical_structure(
        pdf,
        semantic=True,
        pdfua=True,
        preflight=False,
    )
    first_root = pdf.Root["/StructTreeRoot"]
    second_result = ensure_logical_structure(
        pdf,
        semantic=True,
        pdfua=True,
        preflight=False,
    )

    assert first_result["structure_rebuilt"] is True
    assert first_result["annotations_tagged"] == 0
    assert second_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == first_root.objgen
    assert len(page.obj["/Annots"]) == 1
    assert resolve_indirect(page.obj["/Annots"][0]).objgen == printer_mark.objgen
    assert "/StructParent" not in printer_mark
    assert "/StructParents" not in printer_mark


def test_normalizes_empty_structure_language() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    _, document = _install_existing_structure(pdf, language=String(""))

    result = ensure_logical_structure(pdf)

    assert result["structure_languages_normalized"] == 1
    assert str(document["/Lang"]) == "und"


def test_preserves_standard_role_mapped_to_standard_role() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, _ = _install_existing_structure(
        pdf,
        role_map=Dictionary(Document=Name.Div),
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_pdfua_rebuilds_standard_role_mapped_to_another_standard_role() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, _document = _install_existing_structure(
        pdf,
        role_map=Dictionary(Document=Name.Div),
    )

    result = ensure_logical_structure(pdf, pdfua=True)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


def test_pdfua_assigns_and_registers_missing_note_identifier() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, note = _install_existing_structure(pdf, role=Name.Note)

    result = ensure_logical_structure(pdf, pdfua=True)

    identifier = resolve_indirect(note["/ID"])
    assert result["structure_preserved"] is True
    assert result["semantic_repairs"] == 1
    assert isinstance(identifier, String)
    assert resolve_indirect(NameTree(root["/IDTree"])[str(identifier)]).objgen == (
        note.objgen
    )


def test_pdfua_replaces_empty_note_identifier_in_id_tree() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, note = _install_existing_structure(pdf, role=Name.Note)
    note["/ID"] = String("")
    id_tree = NameTree.new(pdf)
    id_tree[""] = note
    root["/IDTree"] = id_tree.obj

    result = ensure_logical_structure(pdf, pdfua=True)

    identifier = resolve_indirect(note["/ID"])
    repaired_tree = NameTree(root["/IDTree"])
    assert result["structure_preserved"] is True
    assert result["semantic_repairs"] == 1
    assert isinstance(identifier, String)
    assert len(repaired_tree) == 1
    assert resolve_indirect(repaired_tree[str(identifier)]).objgen == note.objgen


def test_widget_tooltips_are_scoped_to_terminal_fields() -> None:
    pdf = new_pdf()
    shared_parent = pdf.make_indirect(Dictionary(FT=Name.Tx))
    fields = []
    widgets = []
    for label in ("First name", "Last name"):
        field = pdf.make_indirect(Dictionary(T=String(label), Parent=shared_parent))
        widget = pdf.make_indirect(
            Dictionary(Type=Name.Annot, Subtype=Name.Widget, Parent=field)
        )
        field["/Kids"] = Array([widget])
        fields.append(field)
        widgets.append(widget)
    shared_parent["/Kids"] = Array(fields)

    assert tagging._ensure_widget_tooltip(widgets[0]) is True
    assert tagging._ensure_widget_tooltip(widgets[1]) is True

    assert "/TU" not in shared_parent
    assert [str(field["/TU"]) for field in fields] == ["First name", "Last name"]
    assert [str(widget["/TU"]) for widget in widgets] == ["First name", "Last name"]


def test_widget_tooltip_uses_parent_field_when_widget_repeats_field_type() -> None:
    pdf = new_pdf()
    field = pdf.make_indirect(Dictionary(FT=Name.Tx, T=String("Customer email")))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Tx,
            Parent=field,
        )
    )
    field["/Kids"] = Array([widget])

    assert tagging._ensure_widget_tooltip(widget) is True

    assert str(field["/TU"]) == "Customer email"
    assert str(widget["/TU"]) == "Customer email"


def test_preserves_long_role_map_alias_chain() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    role_map = Dictionary()
    for index in range(1200):
        target = f"/Alias{index + 1}" if index < 1199 else "/Document"
        role_map[f"/Alias{index}"] = Name(target)
    old_root, _ = _install_existing_structure(
        pdf,
        role=Name("/Alias0"),
        role_map=role_map,
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_preserves_valid_structure_attribute_class() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    root["/ClassMap"] = Dictionary(
        Body=Dictionary(O=Name.Layout),
    )
    document["/C"] = Name.Body

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_preserves_revisioned_structure_attribute_classes() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    stream_attribute = pdf.make_stream(b"")
    stream_attribute["/O"] = Name.Layout
    root["/ClassMap"] = Dictionary(
        Body=Dictionary(O=Name.Layout),
        Emphasis=stream_attribute,
    )
    document["/R"] = 2
    document["/C"] = Array([Name.Body, 1, Name.Emphasis, 2])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


@pytest.mark.parametrize("invalid_class_map", ["wrong-type", "missing", "bad-value"])
def test_rebuilds_invalid_structure_attribute_class(
    invalid_class_map: str,
) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    document["/C"] = Name.Body
    if invalid_class_map == "wrong-type":
        root["/ClassMap"] = 42
    elif invalid_class_map == "bad-value":
        root["/ClassMap"] = Dictionary(Body=42)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != root.objgen


def test_preserves_valid_structure_id_tree() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    document["/ID"] = String("doc-id")
    id_tree = NameTree.new(pdf)
    id_tree["doc-id"] = document
    root["/IDTree"] = id_tree.obj

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_preserves_semantics_with_empty_structure_id_tree() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, figure = _install_existing_structure(pdf, role=Name.Figure)
    figure["/Alt"] = String("Diagram")
    root["/IDTree"] = pdf.make_indirect(Dictionary())

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    preserved_figure = resolve_indirect(root["/K"])
    assert preserved_figure["/S"] == Name.Figure
    assert str(preserved_figure["/Alt"]) == "Diagram"


@pytest.mark.parametrize("malformed", [False, True], ids=["valid", "malformed"])
def test_handles_deep_structure_id_tree(malformed: bool) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    identifier = String("deep-id")
    document["/ID"] = identifier
    node = pdf.make_indirect(
        Dictionary(
            Names=Array([identifier, document]),
            Limits=Array([identifier, identifier]),
        )
    )
    for _ in range(1200):
        node = pdf.make_indirect(
            Dictionary(
                Kids=Array([node]),
                Limits=Array([identifier, identifier]),
            )
        )
    id_tree = pdf.make_indirect(Dictionary(Kids=Array([node])))
    if malformed:
        id_tree["/Limits"] = Array([identifier, identifier])
    root["/IDTree"] = id_tree

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is not malformed
    assert result["structure_rebuilt"] is malformed


@pytest.mark.parametrize(
    "invalid_id_tree",
    ["missing", "duplicate-id", "wrong-target", "orphan-entry"],
)
def test_rebuilds_invalid_structure_id_tree(invalid_id_tree: str) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    if invalid_id_tree != "orphan-entry":
        document["/ID"] = String("doc-id")
    if invalid_id_tree == "duplicate-id":
        child = pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.P,
                P=document,
                ID=String("doc-id"),
            )
        )
        document["/K"] = Array([0, child])
    if invalid_id_tree != "missing":
        id_tree = NameTree.new(pdf)
        id_tree["doc-id"] = root if invalid_id_tree == "wrong-target" else document
        root["/IDTree"] = id_tree.obj

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


@pytest.mark.parametrize(
    "role_map",
    [
        None,
        Dictionary(Custom=Name("/Other"), Other=Name("/Custom")),
        Dictionary(Document=Name("/Custom"), Custom=Name.Div),
    ],
    ids=["missing-mapping", "cycle", "standard-to-custom"],
)
def test_rebuilds_structure_with_unresolvable_roles(
    role_map: Dictionary | None,
) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, _ = _install_existing_structure(
        pdf,
        role=Name("/Custom"),
        role_map=role_map,
    )

    result = ensure_logical_structure(pdf)

    new_root = pdf.Root["/StructTreeRoot"]
    assert result["structure_rebuilt"] is True
    assert new_root.objgen != old_root.objgen
    assert "/RoleMap" not in new_root


@pytest.mark.parametrize(
    "role_map",
    [
        Dictionary(A=Name("/B"), B=Name("/A")),
        Dictionary(P=Name.Div, Div=Name.P),
    ],
    ids=["custom", "standard"],
)
def test_rebuilds_when_role_map_is_cyclic_even_if_cycle_is_unused(
    role_map: Dictionary,
) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, _ = _install_existing_structure(
        pdf,
        role=Name.Document,
        role_map=role_map,
    )

    ensure_logical_structure(pdf)

    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


def test_rebuilds_long_cyclic_role_map() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    role_map = Dictionary()
    for index in range(1200):
        target = f"/Alias{index + 1}" if index < 1199 else "/Alias600"
        role_map[f"/Alias{index}"] = Name(target)
    old_root, _ = _install_existing_structure(
        pdf,
        role=Name.Document,
        role_map=role_map,
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


@pytest.mark.parametrize(
    "invalid_part",
    ["root-type", "element-type", "parent-link"],
)
def test_rebuilds_structurally_inconsistent_existing_tree(
    invalid_part: str,
) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    if invalid_part == "root-type":
        old_root["/Type"] = Name.Catalog
    elif invalid_part == "element-type":
        document["/Type"] = Name.Catalog
    else:
        document["/P"] = pdf.Root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


@pytest.mark.parametrize(
    "child_roles",
    [
        (Name.LBody,),
        (Name.Lbl,),
        (Name.Lbl, Name.Lbl, Name.LBody, Name.LBody),
        (Name.LBody, Name.Lbl, Name.LBody),
    ],
    ids=["body-only", "label-only", "repeated", "mixed-order"],
)
def test_preserves_valid_list_item_child_sequences(
    child_roles: tuple[Name, ...],
) -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        b" ".join(
            f"/Span <</MCID {mcid}>> BDC q Q EMC".encode("ascii")
            for mcid in range(len(child_roles))
        ),
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root, Pg=page.obj)
    )
    list_element = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.L, P=document, Pg=page.obj)
    )
    item = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.LI, P=list_element, Pg=page.obj)
    )
    children: list[Dictionary] = []
    marked_content_references: list[Dictionary] = []
    for mcid, role in enumerate(child_roles):
        marked_content_reference = pdf.make_indirect(
            Dictionary(
                Type=Name.MCR,
                Pg=page.obj,
                MCID=mcid,
            )
        )
        marked_content_references.append(marked_content_reference)
        children.append(
            pdf.make_indirect(
                Dictionary(
                    Type=Name.StructElem,
                    S=role,
                    P=item,
                    Pg=page.obj,
                    K=marked_content_reference,
                )
            )
        )
    item["/K"] = Array(children)
    list_element["/K"] = item
    document["/K"] = list_element
    root["/K"] = document
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array(children)
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert [resolve_indirect(child)["/S"] for child in item["/K"]] == list(child_roles)
    assert [owner.objgen for owner in NumberTree(root["/ParentTree"])[0]] == [
        child.objgen for child in children
    ]
    assert [resolve_indirect(child["/K"]).objgen for child in children] == [
        reference.objgen for reference in marked_content_references
    ]


def test_repairs_wrong_parent_tree_mapping_without_rebuilding_structure() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    wrong_owner = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.P, P=document)
    )
    NumberTree(root["/ParentTree"])[0] = Array([wrong_owner])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert NumberTree(root["/ParentTree"])[0][0].objgen == document.objgen


def test_preserves_struct_element_without_optional_type() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    del document["/Type"]

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_preserves_deep_structure_tree_without_recursion_error() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Artifact BMC 0 0 1 1 re f EMC")
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    parent = root
    for _ in range(1200):
        element = pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.Div,
                P=parent,
                Pg=page.obj,
            )
        )
        parent["/K"] = element
        parent = element
    pdf.Root["/StructTreeRoot"] = root
    pdf.Root["/MarkInfo"] = Dictionary(Marked=True)

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_rebuilds_when_struct_element_k_does_not_match_page_mcid() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    document["/K"] = 1

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


def test_rebuilds_empty_structure_element_k_without_content_reference() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    document["/K"] = Array()

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


def test_preserves_empty_structure_element_k_with_other_content() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    empty_heading = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.H1,
            P=document,
            K=Array(),
        )
    )
    document["/K"] = Array([0, empty_heading])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen
    assert len(empty_heading["/K"]) == 0


def test_preserves_nested_structure_element_k_array() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    old_root, document = _install_existing_structure(pdf)
    document["/K"] = Array([Array([0])])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == old_root.objgen


def test_preserves_child_integer_k_with_inherited_page() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/P <</MCID 0>> BDC q Q EMC")
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=root,
            Pg=page.obj,
        )
    )
    paragraph = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.P,
            P=document,
            K=0,
        )
    )
    document["/K"] = paragraph
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([paragraph])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_rebuilds_when_page_reuses_an_mcid() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    old_root, _ = _install_existing_structure(pdf)
    content = resolve_indirect(page.obj["/Contents"])
    content.write(
        bytes(content.read_bytes()) + b"/P <</MCID 0>> BDC BT (duplicate) Tj ET EMC\n"
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen


def test_rebuilds_nested_structure_content_items() -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        b"/P <</MCID 0>> BDC /Span <</MCID 1>> BDC q Q EMC EMC",
    )
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=root,
            Pg=page.obj,
            K=Array([0, 1]),
        )
    )
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([document, document])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_rebuilds_when_pages_share_structparents_key() -> None:
    pdf = new_pdf()
    pages = [
        _add_page(pdf, b"/P <</MCID 0>> BDC q Q EMC"),
        _add_page(pdf, b"/P <</MCID 0>> BDC q Q EMC"),
    ]
    for page in pages:
        page.obj["/StructParents"] = 3

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    divs = [
        pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.Div,
                P=document,
                Pg=page.obj,
                K=0,
            )
        )
        for page in pages
    ]
    document["/K"] = Array(divs)
    parent_tree = NumberTree.new(pdf)
    parent_tree[3] = Array([divs[0]])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 4
    pdf.Root["/StructTreeRoot"] = root
    old_parent_tree = resolve_indirect(root["/ParentTree"])

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert [int(page.obj["/StructParents"]) for page in pages] == [0, 1]
    assert resolve_indirect(root["/ParentTree"]).objgen == old_parent_tree.objgen
    assert int(root["/ParentTreeNextKey"]) == 4


def test_repairs_parent_tree_with_unreferenced_key() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    NumberTree(root["/ParentTree"])[99] = Array([document])

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert list(NumberTree(root["/ParentTree"]).keys()) == [0]


def test_repairs_orphan_parent_array_slots_and_preserves_actualtext() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    document["/ActualText"] = String("replacement text")
    orphan = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.P, Pg=pdf.pages[0].obj, K=3)
    )
    NumberTree(root["/ParentTree"])[0] = Array([document, None, None, orphan])

    result = ensure_logical_structure(pdf)

    repaired = NumberTree(root["/ParentTree"])[0]
    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert str(document["/ActualText"]) == "replacement text"
    assert len(repaired) == 1
    assert repaired[0].objgen == document.objgen


@pytest.mark.parametrize("malformed", [False, True], ids=["valid", "malformed"])
def test_handles_deep_parent_tree(malformed: bool) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    node = pdf.make_indirect(
        Dictionary(
            Nums=Array([0, Array([document])]),
            Limits=Array([0, 0]),
        )
    )
    for _ in range(1200):
        node = pdf.make_indirect(
            Dictionary(
                Kids=Array([node]),
                Limits=Array([0, 0]),
            )
        )
    parent_tree = pdf.make_indirect(Dictionary(Kids=Array([node])))
    if malformed:
        parent_tree["/Limits"] = Array([0, 0])
    root["/ParentTree"] = parent_tree

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_repairs_parent_tree_with_duplicate_raw_key() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    root["/ParentTree"] = pdf.make_indirect(
        Dictionary(
            Nums=Array(
                [
                    0,
                    Array([document]),
                    0,
                    Array([document]),
                ]
            )
        )
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_repairs_parent_tree_with_direct_kid() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    direct_child = Dictionary(
        Nums=Array([0, Array([document])]),
        Limits=Array([0, 0]),
    )
    root["/ParentTree"] = pdf.make_indirect(Dictionary(Kids=Array([direct_child])))

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


@pytest.mark.parametrize(
    "invalid_tree",
    ["root-limits", "boolean-child-limits", "empty-root"],
)
def test_repairs_parent_tree_with_invalid_number_tree_shape(
    invalid_tree: str,
) -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, document = _install_existing_structure(pdf)
    if invalid_tree == "root-limits":
        root["/ParentTree"] = pdf.make_indirect(
            Dictionary(
                Nums=Array([0, Array([document])]),
                Limits=Array([0, 0]),
            )
        )
    elif invalid_tree == "boolean-child-limits":
        child = pdf.make_indirect(
            Dictionary(
                Nums=Array([0, Array([document])]),
                Limits=Array([False, False]),
            )
        )
        root["/ParentTree"] = pdf.make_indirect(Dictionary(Kids=Array([child])))
    else:
        root["/ParentTree"] = pdf.make_indirect(Dictionary())

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_repairs_invalid_parent_tree_next_key() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    root, _ = _install_existing_structure(pdf)
    root["/ParentTreeNextKey"] = 0

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert result["structure_rebuilt"] is False
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert int(pdf.Root["/StructTreeRoot"]["/ParentTreeNextKey"]) == 1


def test_parent_tree_repair_rolls_back_when_content_is_irreparable() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    root, _ = _install_existing_structure(pdf)
    content = resolve_indirect(page.obj["/Contents"])
    content.write(bytes(content.read_bytes()) + b"/P <</MCID 1>> BDC q Q EMC\n")
    root["/ParentTreeNextKey"] = 17
    old_parent_tree = resolve_indirect(root["/ParentTree"])

    assert tagging._existing_structure_elements(pdf) is None
    assert resolve_indirect(root["/ParentTree"]).objgen == old_parent_tree.objgen
    assert int(root["/ParentTreeNextKey"]) == 17

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != root.objgen
    assert resolve_indirect(root["/ParentTree"]).objgen == old_parent_tree.objgen
    assert int(root["/ParentTreeNextKey"]) == 17


@pytest.mark.parametrize(
    "content",
    [
        b"/Document <</MCID 0>> BDC BT (x) Tj ET",
        b"/Document <</MCID 0>> BDC BT (x) Tj ET EMC EMC",
    ],
    ids=["missing-emc", "extra-emc"],
)
def test_rebuilds_and_balances_malformed_marked_content(content: bytes) -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    _install_existing_structure(pdf)
    resolve_indirect(page.obj["/Contents"]).write(content)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    depth = 0
    for instruction in pikepdf.parse_content_stream(page):
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        if instruction.operator in (
            pikepdf.Operator("BMC"),
            pikepdf.Operator("BDC"),
        ):
            depth += 1
        elif instruction.operator == pikepdf.Operator("EMC"):
            depth -= 1
            assert depth >= 0
    assert depth == 0


def test_preserves_marked_content_spanning_page_content_streams() -> None:
    pdf = new_pdf()
    first = b"/Artifact BMC q Q"
    second = b"q Q EMC"
    page = _add_page(pdf, [first, second])

    ensure_logical_structure(pdf)

    native_streams = _content_streams(page)[1:-1]
    assert [bytes(stream.read_bytes()) for stream in native_streams] == [
        first,
        second,
    ]


@pytest.mark.parametrize(
    "content",
    [
        b"/Document <</MCID 0>> BDC BT (x) Tj EMC ET",
        b"BT /Document <</MCID 0>> BDC (x) Tj ET EMC",
    ],
    ids=["emc-before-et", "et-before-emc"],
)
def test_rebuilds_and_repairs_crossed_text_and_marked_content(
    content: bytes,
) -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    _install_existing_structure(pdf)
    resolve_indirect(page.obj["/Contents"]).write(content)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    nesting: list[str] = []
    for instruction in pikepdf.parse_content_stream(page):
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        if instruction.operator in (
            pikepdf.Operator("BMC"),
            pikepdf.Operator("BDC"),
        ):
            nesting.append("marked")
        elif instruction.operator == pikepdf.Operator("BT"):
            nesting.append("text")
        elif instruction.operator == pikepdf.Operator("EMC"):
            assert nesting.pop() == "marked"
        elif instruction.operator == pikepdf.Operator("ET"):
            assert nesting.pop() == "text"
    assert nesting == []


def test_rebuilds_dummy_tree_without_page_mcid_mapping() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT (untagged) Tj ET")
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    root["/K"] = document
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert _wrapper_mcid(pdf.pages[0]) == 0


def test_preserves_artifact_only_page_without_direct_mcid() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Artifact BMC 0 0 m 10 10 l S EMC")
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    root["/K"] = document
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert len(_content_streams(page)) == 1


def test_preserves_semantics_with_empty_structure_parent_tree() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"/Artifact BMC 0 0 m 10 10 l S EMC")
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=root,
            T=String("Accessible document"),
        )
    )
    root["/K"] = document
    root["/ParentTree"] = pdf.make_indirect(Dictionary())
    root["/ParentTreeNextKey"] = 0
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    preserved_document = resolve_indirect(root["/K"])
    assert preserved_document.objgen == document.objgen
    assert preserved_document["/S"] == Name.Document
    assert str(preserved_document["/T"]) == "Accessible document"


def test_rebuilds_artifact_mapped_as_structural_content() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Artifact <</MCID 0>> BDC q Q EMC")
    page.obj["/StructParents"] = 0
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root, Pg=page.obj, K=0)
    )
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([document])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 1
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    instructions = [
        instruction
        for instruction in pikepdf.parse_content_stream(page)
        if not isinstance(instruction, pikepdf.ContentStreamInlineImage)
    ]
    artifact = next(
        instruction
        for instruction in instructions
        if instruction.operator == pikepdf.Operator("BDC")
        and instruction.operands[0] == Name.Artifact
    )
    assert "/MCID" not in resolve_indirect(artifact.operands[1])


def test_preserves_artifact_only_document_without_k_or_parent_tree() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Artifact BMC 0 0 m 10 10 l S EMC")
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert len(_content_streams(page)) == 1


def test_preserves_valid_form_only_structure() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert int(form["/StructParents"]) == 4


def test_preserves_mcr_for_page_content_stream() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/P <</MCID 0>> BDC q Q EMC")
    content = resolve_indirect(page.obj["/Contents"])
    content["/StructParents"] = 4

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=content, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert int(content["/StructParents"]) == 4


def test_ignores_unrendered_form_resource_context() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P /MC BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/StructParents"] = 4
    _add_page(
        pdf,
        b"q Q",
        resources=Dictionary(
            XObject=Dictionary(Fm0=form),
            Properties=Dictionary(MC=Dictionary()),
        ),
    )
    second_page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(
            XObject=Dictionary(Fm0=form),
            Properties=Dictionary(MC=Dictionary(MCID=0)),
        ),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=second_page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=second_page.obj, Stm=form, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_rebuilds_image_stream_referenced_as_mcr() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParents"] = 4
    page = _add_page(
        pdf,
        b"/Im0 Do",
        resources=Dictionary(XObject=Dictionary(Im0=image)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=image, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert "/StructParents" not in image


def test_rebuilds_form_mcr_with_wrong_page() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    first_page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )
    second_page = _add_page(pdf, b"/Artifact BMC q Q EMC")

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=first_page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=second_page.obj, Stm=form, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert _wrapper_mcid(first_page) == 0
    assert _wrapper_mcid(second_page) == 0


def test_rebuilds_form_using_structparent_and_structparents() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    form["/StructParent"] = 5
    page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    div["/K"] = Array(
        [
            pdf.make_indirect(Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)),
            pdf.make_indirect(Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=form)),
        ]
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    parent_tree[5] = div
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 6
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert "/StructParents" not in form
    assert "/StructParent" not in form


def test_rebuilds_form_with_own_mcid_invoked_inside_page_mcid() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    page = _add_page(
        pdf,
        b"/P <</MCID 0>> BDC /Fm0 Do EMC",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )
    page.obj["/StructParents"] = 0

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    outer = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj, K=0)
    )
    inner = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    inner["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)
    )
    document["/K"] = Array([outer, inner])
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([outer])
    parent_tree[4] = Array([inner])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_rebuilds_form_with_own_mcid_invoked_multiple_times() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    page = _add_page(
        pdf,
        b"/Fm0 Do /Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)
    )
    document["/K"] = div
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_rebuilds_structured_form_also_invoked_from_pattern() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 4
    pattern = pdf.make_stream(b"/Fm0 Do")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
    page = _add_page(
        pdf,
        (b"/Fm0 Do /P <</MCID 0>> BDC /Pattern cs /P1 scn 0 0 10 10 re f EMC"),
        resources=Dictionary(
            XObject=Dictionary(Fm0=form),
            Pattern=Dictionary(P1=pattern),
        ),
    )
    page.obj["/StructParents"] = 0

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    outer = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj, K=0)
    )
    inner = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=page.obj)
    )
    inner["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=page.obj, Stm=form, MCID=0)
    )
    document["/K"] = Array([outer, inner])
    parent_tree = NumberTree.new(pdf)
    parent_tree[0] = Array([outer])
    parent_tree[4] = Array([inner])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_rebuilds_untagged_nested_xobject_in_wrong_page_context() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 5

    form = pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC /Im0 Do")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
    form["/StructParents"] = 4
    first_page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )
    second_page = _add_page(
        pdf,
        b"/Im0 Do",
        resources=Dictionary(XObject=Dictionary(Im0=image)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    div = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Div, P=document, Pg=first_page.obj)
    )
    div["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.MCR, Pg=first_page.obj, Stm=form, MCID=0)
    )
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=second_page.obj,
        )
    )
    figure["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=second_page.obj, Obj=image)
    )
    document["/K"] = Array([div, figure])
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = Array([div])
    parent_tree[5] = figure
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 6
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_preserves_image_tagged_as_complete_xobject() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 4
    page = _add_page(
        pdf,
        b"/Im0 Do",
        resources=Dictionary(XObject=Dictionary(Im0=image)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    figure["/K"] = pdf.make_indirect(Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=image))
    document["/K"] = figure
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = figure
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert int(image["/StructParent"]) == 4


def test_preserves_nested_image_tagged_as_complete_xobject() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 4
    form = pdf.make_stream(b"/Im0 Do")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
    page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=page.obj)
    )
    figure["/K"] = pdf.make_indirect(Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=image))
    document["/K"] = figure
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = figure
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_preserves_nested_form_with_artifact_only_painting() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/Artifact BMC 0 0 10 10 re f EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    root["/K"] = document
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_preserves_deep_nested_form_chain() -> None:
    pdf = new_pdf()
    form = _nested_form_chain(pdf, 1200)
    _add_page(
        pdf,
        b"/Fm Do",
        resources=Dictionary(XObject=Dictionary(Fm=form)),
    )
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    root["/K"] = document
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen


def test_rebuilds_deep_nested_form_chain() -> None:
    pdf = new_pdf()
    form = _nested_form_chain(pdf, 1200)
    page = _add_page(
        pdf,
        b"/Fm Do",
        resources=Dictionary(XObject=Dictionary(Fm=form)),
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert _wrapper_mcid(page) == 0


def test_preserves_complete_xobject_referenced_once_per_page() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 4
    resources = Dictionary(XObject=Dictionary(Im0=image))
    pages = [
        _add_page(pdf, b"/Im0 Do", resources=resources),
        _add_page(pdf, b"/Im0 Do", resources=resources),
    ]

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Alt=String("Logo"),
        )
    )
    figure["/K"] = Array(
        [
            pdf.make_indirect(Dictionary(Type=Name.OBJR, Pg=page.obj, Obj=image))
            for page in pages
        ]
    )
    document["/K"] = figure
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = figure
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert str(figure["/Alt"]) == "Logo"


@pytest.mark.parametrize(
    "keys",
    [
        {"/StructParent": 8},
        {"/StructParents": 9},
        {"/StructParent": 8, "/StructParents": 9},
    ],
    ids=["singular", "plural", "both"],
)
def test_rebuilds_stale_structure_keys_on_image_xobject(
    keys: dict[str, int],
) -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    for key, value in keys.items():
        image[key] = value
    resources = Dictionary(XObject=Dictionary(Im0=image))
    _add_page(pdf, b"/Artifact BMC /Im0 Do EMC", resources=resources)
    pdf.Root["/StructTreeRoot"] = pdf.make_indirect(
        Dictionary(Type=Name.StructTreeRoot)
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert "/StructParent" not in image
    assert "/StructParents" not in image


def test_rebuilds_image_objr_used_on_another_page() -> None:
    pdf = new_pdf()
    image = pdf.make_stream(b"\x80")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image["/StructParent"] = 4
    resources = Dictionary(XObject=Dictionary(Im0=image))
    first_page = _add_page(pdf, b"/Im0 Do", resources=resources)
    second_page = _add_page(pdf, b"/Im0 Do", resources=resources)

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    figure = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Figure, P=document, Pg=first_page.obj)
    )
    figure["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=first_page.obj, Obj=image)
    )
    document["/K"] = figure
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = figure
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert _wrapper_mcid(first_page) == 0
    assert _wrapper_mcid(second_page) == 0


def test_rebuilds_annotation_objr_with_wrong_page() -> None:
    pdf = new_pdf()
    first_page = _add_page(pdf, b"/Artifact BMC q Q EMC")
    second_page = _add_page(pdf, b"/Artifact BMC q Q EMC")
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            StructParent=4,
        )
    )
    first_page.obj["/Annots"] = Array([annotation])

    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    annot_element = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Annot,
            P=document,
            Pg=first_page.obj,
        )
    )
    annot_element["/K"] = pdf.make_indirect(
        Dictionary(Type=Name.OBJR, Pg=second_page.obj, Obj=annotation)
    )
    document["/K"] = annot_element
    parent_tree = NumberTree.new(pdf)
    parent_tree[4] = annot_element
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    root["/ParentTreeNextKey"] = 5
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert int(annotation["/StructParent"]) == 2


def test_preserves_annotation_appearance_mcr_with_stmown() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"/Artifact BMC q Q EMC")
    root, appearance, annotation, _ = _install_annotation_appearance_structure(pdf)

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root.objgen
    assert int(appearance["/StructParents"]) == 4
    assert int(annotation["/StructParent"]) == 5


def test_rebuilds_annotation_appearance_mcr_with_wrong_stmown() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"/Artifact BMC q Q EMC")
    _, _, _, mcr = _install_annotation_appearance_structure(pdf)
    mcr["/StmOwn"] = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
        )
    )

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True


def test_rebuilds_structure_referencing_foreign_page_dictionary() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Artifact BMC q Q EMC")
    foreign_page = pdf.make_indirect(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 10, 10]),
            Contents=pdf.make_stream(b"/P <</MCID 0>> BDC q Q EMC"),
            StructParents=5,
        )
    )
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=root,
            Pg=foreign_page,
            K=0,
        )
    )
    parent_tree = NumberTree.new(pdf)
    parent_tree[5] = Array([document])
    root["/K"] = document
    root["/ParentTree"] = parent_tree.obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert _wrapper_mcid(page) == 0


def test_rebuilds_when_form_mcid_has_no_structure_reference() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/P <</MCID 5>> BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    page = _add_page(
        pdf,
        b"/Fm0 Do",
        resources=Dictionary(XObject=Dictionary(Fm0=form)),
    )
    old_root, _ = _install_existing_structure(pdf)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert b"/MCID" not in bytes(form.read_bytes())
    assert _wrapper_mcid(page) == 0


def test_self_referencing_form_xobject_fails_closed() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/Fm Do")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form_resources = Dictionary()
    form["/Resources"] = form_resources
    form_resources["/XObject"] = Dictionary(Fm=form)
    page = _add_page(
        pdf,
        b"/Fm Do",
        resources=Dictionary(XObject=Dictionary(Fm=form)),
    )

    with pytest.raises(ConversionError, match="Form XObject is recursive"):
        ensure_logical_structure(pdf)

    assert "/StructTreeRoot" not in pdf.Root
    assert "/StructParents" not in page.obj


def test_rebuild_removes_mcid_from_soft_mask_with_inherited_resources() -> None:
    pdf = new_pdf()
    group = pdf.make_stream(b"/Span /MC BDC q Q EMC")
    group["/Type"] = Name.XObject
    group["/Subtype"] = Name.Form
    group["/BBox"] = Array([0, 0, 10, 10])
    group["/Group"] = Dictionary(S=Name.Transparency)
    group["/StructParents"] = 7
    resources = Dictionary(
        ExtGState=Dictionary(
            GS=Dictionary(SMask=Dictionary(S=Name.Luminosity, G=group))
        ),
        Properties=Dictionary(MC=Dictionary(MCID=7)),
    )
    page = _add_page(pdf, b"/GS gs q 0 0 10 10 re f Q", resources=resources)

    first = ensure_logical_structure(pdf)
    second = ensure_logical_structure(pdf)

    assert first["mcids_removed"] == 1
    assert "/StructParents" not in group
    assert b"/MCID" not in bytes(group.read_bytes())
    assert second["structure_rebuilt"] is False
    assert _wrapper_mcid(page) == 0


def test_rebuild_removes_mcid_from_form_inheriting_pattern_resources() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/Span /MC BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/StructParents"] = 7
    pattern = pdf.make_stream(b"/Fm Do")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary(
        XObject=Dictionary(Fm=form),
        Properties=Dictionary(MC=Dictionary(MCID=7)),
    )
    page = _add_page(
        pdf,
        b"/Pattern cs /P1 scn 0 0 10 10 re f",
        resources=Dictionary(Pattern=Dictionary(P1=pattern)),
    )

    first = ensure_logical_structure(pdf)
    second = ensure_logical_structure(pdf)

    assert first["mcids_removed"] == 1
    assert "/StructParents" not in form
    assert b"/MCID" not in bytes(form.read_bytes())
    assert second["structure_rebuilt"] is False
    assert _wrapper_mcid(page) == 0


def test_rebuilds_existing_tree_with_orphan_mcid_in_pattern_form() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/Span /MC BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    pattern = pdf.make_stream(b"/Fm Do")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary(
        XObject=Dictionary(Fm=form),
        Properties=Dictionary(MC=Dictionary(MCID=7)),
    )
    _add_page(
        pdf,
        b"/Pattern cs /P1 scn 0 0 10 10 re f",
        resources=Dictionary(Pattern=Dictionary(P1=pattern)),
    )
    old_root, _ = _install_existing_structure(pdf)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert b"/MCID" not in bytes(form.read_bytes())


def test_rebuilds_existing_tree_with_orphan_mcid_in_tiling_pattern() -> None:
    pdf = new_pdf()
    pattern = pdf.make_stream(b"/Span <</MCID 7>> BDC 0 0 10 10 re f EMC")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary()
    page = _add_page(
        pdf,
        b"/Pattern cs /P1 scn 0 0 10 10 re f",
        resources=Dictionary(Pattern=Dictionary(P1=pattern)),
    )
    old_root, _ = _install_existing_structure(pdf)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert b"/MCID" not in bytes(pattern.read_bytes())
    assert _wrapper_mcid(page) == 0


@pytest.mark.parametrize("font_is_reachable", [True, False])
def test_rebuilds_existing_tree_with_orphan_mcid_in_type3_charproc(
    font_is_reachable: bool,
) -> None:
    pdf = new_pdf()
    charproc = pdf.make_stream(b"/Span <</MCID 7>> BDC 0 0 10 10 re f EMC")
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 10, 10]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(A=charproc),
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([65, Name.A]),
            ),
            FirstChar=65,
            LastChar=65,
            Widths=Array([600]),
            Resources=Dictionary(),
        )
    )
    page = _add_page(
        pdf,
        b"BT /F1 12 Tf (A) Tj ET",
        resources=Dictionary(Font=Dictionary(F1=font)),
    )
    old_root, _ = _install_existing_structure(pdf)
    if not font_is_reachable:
        del page.obj["/Resources"]["/Font"]

    result = ensure_logical_structure(pdf)
    second = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert b"/MCID" not in bytes(charproc.read_bytes())
    assert _wrapper_mcid(page) == 0
    assert second["structure_preserved"] is True


def test_explicit_rebuild_replaces_existing_structure_and_page_key() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    page.obj["/StructParents"] = 99
    old_root, _ = _install_existing_structure(pdf)

    result = ensure_logical_structure(pdf, rebuild=True)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert int(page.obj["/StructParents"]) == 0
    assert len(_content_streams(page)) == 3


def test_tags_annotation_roles_with_objr_and_parent_tree_entries() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    page.obj["/StructParent"] = 98
    first = Dictionary(
        Type=Name.Annot,
        Subtype=Name.Text,
        Rect=Array([0, 0, 10, 10]),
        StructParent=99,
        StructParents=100,
    )
    second = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([10, 10, 20, 20]),
        )
    )
    third = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([20, 20, 30, 30]),
        )
    )
    page.obj["/Annots"] = Array([first, second, third])

    result = ensure_logical_structure(pdf)

    assert result["annotations_tagged"] == 3
    assert page.obj["/Tabs"] == Name.S
    annotations = [resolve_indirect(item) for item in page.obj["/Annots"]]
    assert [int(item["/StructParent"]) for item in annotations] == [1, 2, 3]
    assert "/StructParent" not in page.obj
    assert all("/StructParents" not in item for item in annotations)
    assert all(item.is_indirect for item in annotations)

    root, _, divs = _generated_parts(pdf)
    div_kids = divs[0]["/K"]
    annotation_elements = [resolve_indirect(item) for item in div_kids[1:]]
    assert [str(item["/S"]) for item in annotation_elements] == [
        "/Annot",
        "/Link",
        "/Form",
    ]
    for annotation, element, key in zip(
        annotations,
        annotation_elements,
        [1, 2, 3],
        strict=True,
    ):
        objr = resolve_indirect(element["/K"])
        assert str(objr["/Type"]) == "/OBJR"
        assert objr["/Obj"].objgen == annotation.objgen
        assert NumberTree(root["/ParentTree"])[key].objgen == element.objgen
    assert int(root["/ParentTreeNextKey"]) == 4


def test_clones_annotation_reused_on_multiple_pages() -> None:
    pdf = new_pdf()
    first_page = _add_page(pdf, b"q Q")
    second_page = _add_page(pdf, b"q Q")
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            P=first_page.obj,
        )
    )
    first_page.obj["/Annots"] = Array([annotation])
    second_page.obj["/Annots"] = Array([annotation])

    first_result = ensure_logical_structure(pdf)
    annotations = [resolve_indirect(page.obj["/Annots"][0]) for page in pdf.pages]
    first_root = pdf.Root["/StructTreeRoot"]
    second_result = ensure_logical_structure(pdf)

    assert first_result["structure_rebuilt"] is True
    assert annotations[0].objgen != annotations[1].objgen
    assert [int(item["/StructParent"]) for item in annotations] == [2, 3]
    assert annotations[0]["/P"].objgen == first_page.obj.objgen
    assert annotations[1]["/P"].objgen == second_page.obj.objgen
    assert second_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == first_root.objgen


def test_rejects_widget_reused_in_multiple_annotation_arrays() -> None:
    pdf = new_pdf()
    first_page = _add_page(pdf, b"q Q")
    second_page = _add_page(pdf, b"q Q")
    field = pdf.make_indirect(
        Dictionary(
            FT=Name.Tx,
            T=String("F"),
        )
    )
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([0, 0, 10, 10]),
            Parent=field,
        )
    )
    field["/Kids"] = Array([widget])
    pdf.Root["/AcroForm"] = Dictionary(Fields=Array([field]))
    first_page.obj["/Annots"] = Array([widget])
    second_page.obj["/Annots"] = Array([widget])

    with pytest.raises(ConversionError, match="Widget annotation.*multiple /Annots"):
        ensure_logical_structure(pdf)

    kids = resolve_indirect(field["/Kids"])
    assert len(kids) == 1
    assert resolve_indirect(kids[0]).objgen == widget.objgen
    assert resolve_indirect(second_page.obj["/Annots"][0]).objgen == widget.objgen


def test_rejects_reused_popup_annotation_graph() -> None:
    pdf = new_pdf()
    first_page = _add_page(pdf, b"q Q")
    second_page = _add_page(pdf, b"q Q")
    text = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            P=first_page.obj,
        )
    )
    popup = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Popup,
            Rect=Array([0, 0, 10, 10]),
            Parent=text,
            P=first_page.obj,
        )
    )
    text["/Popup"] = popup
    first_page.obj["/Annots"] = Array([text, popup])
    second_page.obj["/Annots"] = Array([text, popup])

    with pytest.raises(
        ConversionError,
        match=r"annotation with /Popup.*multiple /Annots",
    ):
        ensure_logical_structure(pdf)

    assert resolve_indirect(text["/Popup"]).objgen == popup.objgen
    assert resolve_indirect(popup["/Parent"]).objgen == text.objgen
    assert resolve_indirect(second_page.obj["/Annots"][0]).objgen == text.objgen
    assert resolve_indirect(second_page.obj["/Annots"][1]).objgen == popup.objgen


def test_clones_reused_annotation_without_duplicate_page_name() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            NM=String("duplicate"),
        )
    )
    page.obj["/Annots"] = Array([annotation, annotation])

    first_result = ensure_logical_structure(pdf)
    annotations = [resolve_indirect(item) for item in page.obj["/Annots"]]
    first_root = pdf.Root["/StructTreeRoot"]
    second_result = ensure_logical_structure(pdf)

    assert first_result["structure_rebuilt"] is True
    assert annotations[0].objgen != annotations[1].objgen
    assert str(annotations[0]["/NM"]) == "duplicate"
    assert "/NM" not in annotations[1]
    assert second_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == first_root.objgen


def test_rebuilds_distinct_annotations_with_duplicate_page_names() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    page.obj["/Annots"] = Array(
        [
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([0, 0, 10, 10]),
                )
            ),
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([10, 10, 20, 20]),
                )
            ),
        ]
    )
    ensure_logical_structure(pdf)
    annotations = [resolve_indirect(item) for item in page.obj["/Annots"]]
    for annotation in annotations:
        annotation["/NM"] = String("duplicate")
    old_root = pdf.Root["/StructTreeRoot"]

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != old_root.objgen
    assert sum("/NM" in annotation for annotation in annotations) == 1


def test_removes_stale_direct_and_named_mcids_when_rebuilding() -> None:
    pdf = new_pdf()
    resources = Dictionary(
        Properties=Dictionary(
            MC2=Dictionary(MCID=2, ActualText=String("named text")),
        )
    )
    page = _add_page(
        pdf,
        (
            b"/P <</MCID 0 /ActualText (direct text)>> BDC q Q EMC "
            b"/Span /MC2 BDC BT ET EMC"
        ),
        resources=resources,
    )

    ensure_logical_structure(pdf)

    assert _wrapper_mcid(page) == 0
    native = bytes(_content_streams(page)[1].read_bytes())
    assert b"/MCID" not in native
    assert b"direct text" in native
    assert "/MCID" not in resources["/Properties"]["/MC2"]
    assert str(resources["/Properties"]["/MC2"]["/ActualText"]) == "named text"
    root, _, divs = _generated_parts(pdf)
    parent_entry = NumberTree(root["/ParentTree"])[0]
    assert len(parent_entry) == 1
    assert parent_entry[0].objgen == divs[0].objgen


def test_normalizes_inline_and_named_marked_content_languages() -> None:
    pdf = new_pdf()
    resources = Dictionary(
        Properties=Dictionary(
            SpanProps=Dictionary(Lang=String("not valid!")),
        )
    )
    page = _add_page(
        pdf,
        (
            b"/Span <</Lang <feff0430043d002d00430041>>> BDC q Q EMC "
            b"/Span /SpanProps BDC BT ET EMC"
        ),
        resources=resources,
    )

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 2
    native = _content_streams(page)[1]
    inline_properties = next(
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(native)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("BDC")
            and isinstance(resolve_indirect(instruction.operands[1]), Dictionary)
        )
    )
    assert str(inline_properties["/Lang"]) == "und"
    assert str(resources["/Properties"]["/SpanProps"]["/Lang"]) == "und"


def test_normalizes_equal_direct_form_resource_contexts_separately() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(b"/Span /Props BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    resource_contexts = [
        Dictionary(
            XObject=Dictionary(Fm=form),
            Properties=Dictionary(
                Props=Dictionary(Lang=String("not valid!")),
            ),
        )
        for _ in range(2)
    ]
    for resources in resource_contexts:
        _add_page(pdf, b"/Fm Do", resources=resources)

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 2
    assert all(
        str(resources["/Properties"]["/Props"]["/Lang"]) == "und"
        for resources in resource_contexts
    )


def test_normalizes_empty_marked_content_language() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"/Span <</Lang ()>> BDC q Q EMC")

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 1
    native = _content_streams(page)[1]
    properties = next(
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(native)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("BDC")
            and isinstance(resolve_indirect(instruction.operands[1]), Dictionary)
        )
    )
    assert str(properties["/Lang"]) == "und"


def test_normalizes_inline_language_split_across_page_content_streams() -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        [
            b"/Span <</Lang (not_a_tag)>>",
            b"BDC 0 0 10 10 re f EMC",
        ],
    )

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 1
    properties = next(
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(page)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("BDC")
            and isinstance(resolve_indirect(instruction.operands[1]), Dictionary)
            and "/Lang" in resolve_indirect(instruction.operands[1])
        )
    )
    assert str(properties["/Lang"]) == "und"


def test_combined_rewrite_does_not_mutate_stream_shared_by_another_page() -> None:
    pdf = new_pdf()
    shared = pdf.make_stream(b"q Q")
    tail = pdf.make_stream(b"/Span <</Lang (not_a_tag)>> BDC 0 0 10 10 re f EMC")
    first_page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=Array([shared, tail]),
        )
    )
    second_page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=shared,
        )
    )
    pdf.pages.append(first_page)
    pdf.pages.append(second_page)

    ensure_logical_structure(pdf)

    assert bytes(shared.read_bytes()) == b"q Q"
    second_operators = [
        str(instruction.operator)
        for instruction in pikepdf.parse_content_stream(second_page)
        if not isinstance(instruction, pikepdf.ContentStreamInlineImage)
    ]
    assert "re" not in second_operators
    assert "f" not in second_operators


def test_combined_rewrite_preserves_repeated_stream_occurrences() -> None:
    pdf = new_pdf()
    shared = pdf.make_stream(b"/Span <</Lang (not_a_tag)>> BDC 0 0 10 10 re f EMC")
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=Array([shared, shared]),
        )
    )
    pdf.pages.append(page)

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 2
    assert b"not_a_tag" in bytes(shared.read_bytes())
    operators = [
        str(instruction.operator)
        for instruction in pikepdf.parse_content_stream(page)
        if not isinstance(instruction, pikepdf.ContentStreamInlineImage)
    ]
    assert operators.count("re") == 2
    assert operators.count("f") == 2


def test_page_repair_does_not_mutate_a_shared_content_stream() -> None:
    pdf = new_pdf()
    shared = pdf.make_stream(b"/Artifact BMC q Q")
    closing = pdf.make_stream(b"EMC")
    first_page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=Array([shared, closing]),
        )
    )
    second_page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Contents=shared,
        )
    )
    pdf.pages.append(first_page)
    pdf.pages.append(second_page)

    first_result = ensure_logical_structure(pdf)
    second_result = ensure_logical_structure(pdf)

    assert first_result["structure_rebuilt"] is True
    assert second_result["structure_preserved"] is True
    assert bytes(shared.read_bytes()) == b"/Artifact BMC q Q"
    assert bytes(closing.read_bytes()) == b"EMC"


def test_normalizes_language_in_marked_content_point_property() -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        b"/Span <</Lang (invalid!)>> DP",
    )

    result = ensure_logical_structure(pdf)

    assert result["marked_content_languages_normalized"] == 1
    native = _content_streams(page)[1]
    properties = next(
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(native)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("DP")
        )
    )
    assert str(properties["/Lang"]) == "und"


def test_normalizes_named_language_from_inherited_page_resources() -> None:
    pdf = new_pdf()
    resources = Dictionary(
        Properties=Dictionary(
            SpanProps=Dictionary(Lang=String("invalid!")),
        )
    )
    page = _add_page(pdf, b"/Span /SpanProps BDC q Q EMC")
    page.obj["/Parent"]["/Resources"] = resources
    if "/Resources" in page.obj:
        del page.obj["/Resources"]

    ensure_logical_structure(pdf)

    assert str(resources["/Properties"]["/SpanProps"]["/Lang"]) == "und"


def test_cleans_stale_structure_from_form_xobject() -> None:
    pdf = new_pdf()
    form = pdf.make_stream(
        b"/Span <</MCID 4 /Lang (invalid!) /ActualText (kept)>> BDC q Q EMC"
    )
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = Dictionary()
    form["/StructParents"] = 12
    form["/StructParent"] = 13
    resources = Dictionary(XObject=Dictionary(Fm0=form))
    _add_page(pdf, b"/Fm0 Do", resources=resources)

    result = ensure_logical_structure(pdf)

    assert result["mcids_removed"] == 1
    assert result["stream_structure_keys_removed"] == 2
    assert "/StructParents" not in form
    assert "/StructParent" not in form
    properties = next(
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(form)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("BDC")
        )
    )
    assert "/MCID" not in properties
    assert str(properties["/Lang"]) == "und"
    assert str(properties["/ActualText"]) == "kept"


def test_removes_named_form_mcid_from_inherited_resources() -> None:
    pdf = new_pdf()
    properties = Dictionary(MCID=5, ActualText=String("kept"))
    form = pdf.make_stream(b"/Span /P BDC q Q EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    resources = Dictionary(
        Properties=Dictionary(P=properties),
        XObject=Dictionary(Fm0=form),
    )
    _add_page(pdf, b"/Fm0 Do", resources=resources)

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert "/MCID" not in properties
    assert str(properties["/ActualText"]) == "kept"


def test_removes_named_appearance_mcid_from_page_resources() -> None:
    pdf = new_pdf()
    properties = Dictionary(MCID=7, ActualText=String("kept"))
    appearance = pdf.make_stream(b"/Span /P BDC q Q EMC")
    appearance["/Type"] = Name.XObject
    appearance["/Subtype"] = Name.Form
    appearance["/BBox"] = Array([0, 0, 10, 10])
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([0, 0, 10, 10]),
            AP=Dictionary(N=appearance),
        )
    )
    resources = Dictionary(Properties=Dictionary(P=properties))
    page = _add_page(pdf, b"q Q", resources=resources)
    page.obj["/Annots"] = Array([annotation])

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert "/MCID" not in properties
    assert str(properties["/ActualText"]) == "kept"


def test_second_default_call_preserves_generated_structure_and_wrappers() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    ensure_logical_structure(pdf)
    root_objgen = pdf.Root["/StructTreeRoot"].objgen
    stream_objgens = [stream.objgen for stream in _content_streams(page)]

    result = ensure_logical_structure(pdf)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root_objgen
    assert [stream.objgen for stream in _content_streams(page)] == stream_objgens


def test_split_inline_mcid_is_removed_and_second_call_preserves_structure() -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        [
            b"/Span <</MCID 4 /ActualText (kept)>>",
            b"BDC q Q EMC",
        ],
    )

    first_result = ensure_logical_structure(pdf)
    root_objgen = pdf.Root["/StructTreeRoot"].objgen
    second_result = ensure_logical_structure(pdf)

    assert first_result["mcids_removed"] == 1
    assert second_result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == root_objgen
    inline_properties = [
        resolve_indirect(instruction.operands[1])
        for instruction in pikepdf.parse_content_stream(page)
        if (
            not isinstance(instruction, pikepdf.ContentStreamInlineImage)
            and instruction.operator == pikepdf.Operator("BDC")
            and len(instruction.operands) >= 2
            and isinstance(resolve_indirect(instruction.operands[1]), Dictionary)
        )
    ]
    assert all("/MCID" not in properties for properties in inline_properties[1:])
    assert any(
        str(properties.get("/ActualText")) == "kept" for properties in inline_properties
    )


def test_rebuild_is_idempotent_and_does_not_nest_generated_wrappers() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT ET")
    ensure_logical_structure(pdf)
    native_objgen = _content_streams(page)[1].objgen

    ensure_logical_structure(pdf, rebuild=True)
    first_rebuild_mcid = _wrapper_mcid(page)
    first_rebuild_streams = _content_streams(page)
    ensure_logical_structure(pdf, rebuild=True)
    second_rebuild_streams = _content_streams(page)

    assert first_rebuild_mcid == _wrapper_mcid(page)
    assert len(first_rebuild_streams) == len(second_rebuild_streams) == 3
    assert first_rebuild_streams[1].objgen == native_objgen
    assert second_rebuild_streams[1].objgen == native_objgen


def test_generated_structure_survives_save_and_reopen() -> None:
    pdf = new_pdf()
    _add_page(pdf, b"BT ET")
    ensure_logical_structure(pdf)
    output = BytesIO()
    pdf.save(output)
    output.seek(0)
    reopened = open_pdf(output)

    result = ensure_logical_structure(reopened)

    assert result["structure_preserved"] is True
    assert _wrapper_mcid(reopened.pages[0]) == 0


def test_rejects_malformed_contents_during_rebuild() -> None:
    pdf = new_pdf()
    page = _add_page(pdf)
    page.obj["/Contents"] = Name.Contents

    with pytest.raises(ConversionError, match="page contents are malformed"):
        ensure_logical_structure(pdf)


def test_rejects_malformed_annotations_during_rebuild() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    page.obj["/Annots"] = Array([Name.Annot])

    with pytest.raises(ConversionError, match="malformed annotation"):
        ensure_logical_structure(pdf)


def test_groups_page_structure_at_array_implementation_limit() -> None:
    pdf = new_pdf()
    for _ in range(8_192):
        _add_page(pdf)

    result = ensure_logical_structure(pdf)

    root = resolve_indirect(pdf.Root["/StructTreeRoot"])
    document = resolve_indirect(root["/K"])
    groups = [resolve_indirect(item) for item in document["/K"]]
    assert result["pages_tagged"] == 8_192
    assert [len(group["/K"]) for group in groups] == [8_191, 1]
    assert all(group["/S"] == Name.Part for group in groups)
    assert all(group["/P"].objgen == document.objgen for group in groups)
    assert groups[0]["/K"][0]["/P"].objgen == groups[0].objgen
    assert groups[1]["/K"][0]["/P"].objgen == groups[1].objgen


def test_groups_annotations_at_array_implementation_limit() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"q Q")
    page.obj["/Annots"] = Array(
        [
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([0, 0, 1, 1]),
                )
            )
            for _ in range(8_191)
        ]
    )

    result = ensure_logical_structure(pdf)

    root, _, divs = _generated_parts(pdf)
    div = divs[0]
    group = resolve_indirect(div["/K"][1])
    assert result["annotations_tagged"] == 8_191
    assert len(div["/K"]) == 2
    assert group["/S"] == Name.Part
    assert len(group["/K"]) == 8_191
    assert group["/P"].objgen == div.objgen
    parent_tree = NumberTree(root["/ParentTree"])
    assert parent_tree[1]["/P"].objgen == group.objgen
    assert parent_tree[8_191]["/P"].objgen == group.objgen

    pending = [resolve_indirect(root["/ParentTree"])]
    while pending:
        node = pending.pop()
        if "/Nums" in node:
            assert len(node["/Nums"]) <= 8_191
        if "/Kids" in node:
            assert len(node["/Kids"]) <= 8_191
            pending.extend(resolve_indirect(item) for item in node["/Kids"])


def test_coalesces_content_before_adding_wrapper_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tagging, "_MAX_ARRAY_ITEMS", 3)
    pdf = new_pdf()
    page = _add_page(pdf, [b"q", b"Q"])

    ensure_logical_structure(pdf)

    streams = _content_streams(page)
    assert len(streams) == 3
    assert bytes(streams[1].read_bytes()) == b"q\nQ"


def test_rebuilds_structure_with_oversized_k_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tagging, "_MAX_ARRAY_ITEMS", 3)
    pdf = new_pdf()
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    document["/K"] = Array(
        [
            pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name.Div, P=document))
            for _ in range(4)
        ]
    )
    root["/K"] = document
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != root.objgen


def test_rebuilds_structure_with_oversized_id_tree_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tagging, "_MAX_ARRAY_ITEMS", 3)
    pdf = new_pdf()
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(Type=Name.StructElem, S=Name.Document, P=root)
    )
    first = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Div,
            P=document,
            ID=String("a"),
        )
    )
    second = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Div,
            P=document,
            ID=String("b"),
        )
    )
    document["/K"] = Array([first, second])
    root["/K"] = document
    root["/IDTree"] = pdf.make_indirect(
        Dictionary(Names=Array([String("a"), first, String("b"), second]))
    )
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != root.objgen


def test_rebuilds_structure_with_oversized_id_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tagging, "_MAX_STRING_BYTES", 3)
    pdf = new_pdf()
    root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=root,
            ID=String("four"),
        )
    )
    root["/K"] = document
    root["/IDTree"] = pdf.make_indirect(
        Dictionary(Names=Array([String("four"), document]))
    )
    root["/ParentTree"] = NumberTree.new(pdf).obj
    pdf.Root["/StructTreeRoot"] = root

    result = ensure_logical_structure(pdf)

    assert result["structure_rebuilt"] is True
    assert pdf.Root["/StructTreeRoot"].objgen != root.objgen


def test_preserved_rich_tree_reports_newly_artifactized_vector_path() -> None:
    pdf = new_pdf()
    page = _add_page(pdf, b"BT 10 10 Td (Body) Tj ET")
    _install_existing_structure(pdf, role=Name.P)
    content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(content, pikepdf.Stream)
    content.write(
        bytes(content.read_bytes())
        + b"\n/Artifact BMC 0 0 m 5 5 l S EMC"
        + b"\n10 10 m 20 20 l S"
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert result["path_artifacts_tagged"] == 1
    assert result["semantic_vector_review_required"] == 1
    instructions = list(pikepdf.parse_content_stream(page))
    operators = [
        str(instruction.operator)
        for instruction in instructions
        if isinstance(instruction, pikepdf.ContentStreamInstruction)
    ]
    assert operators[-5:] == ["BMC", "m", "l", "S", "EMC"]
    assert (
        sum(
            isinstance(instruction, pikepdf.ContentStreamInstruction)
            and instruction.operator == pikepdf.Operator("BMC")
            and instruction.operands[0] == Name.Artifact
            for instruction in instructions
        )
        == 2
    )


def test_preserved_rich_tree_artifactizes_only_unprotected_shading() -> None:
    pdf = new_pdf()
    page = _add_page(
        pdf,
        b"BT 10 10 Td (Body) Tj ET",
        resources=Dictionary(Shading=Dictionary(Sh0=Dictionary())),
    )
    root, _document = _install_existing_structure(pdf, role=Name.P)
    original_root = root.objgen
    content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(content, pikepdf.Stream)
    tagged = bytes(content.read_bytes())
    assert tagged.endswith(b"\nEMC\n")
    content.write(
        tagged[: -len(b"\nEMC\n")]
        + b"\n/Sh0 sh\nEMC\n"
        + b"/Artifact BMC /Sh0 sh EMC\n"
        + b"/Sh0 sh"
    )

    result = ensure_logical_structure(pdf, semantic=True)

    assert result["structure_preserved"] is True
    assert pdf.Root["/StructTreeRoot"].objgen == original_root
    assert result["path_artifacts_tagged"] == 1
    assert result["semantic_vector_review_required"] == 1
    instructions = list(pikepdf.parse_content_stream(page))
    assert (
        sum(
            isinstance(instruction, pikepdf.ContentStreamInstruction)
            and instruction.operator == pikepdf.Operator("BMC")
            and instruction.operands[0] == Name.Artifact
            for instruction in instructions
        )
        == 2
    )


def _form_with_named_mcid(
    pdf: pikepdf.Pdf,
    *,
    form_resources: Dictionary,
) -> pikepdf.Stream:
    form = pdf.make_stream(b"/P /MC0 BDC 0 0 m 5 5 l S EMC")
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 10, 10])
    form["/Resources"] = form_resources
    return form


def test_untagged_painting_resolves_named_bdc_in_nested_form_resources() -> None:
    """A nested form's own /Properties protect its marked content."""
    pdf = new_pdf()
    form = _form_with_named_mcid(
        pdf,
        form_resources=Dictionary(Properties=Dictionary(MC0=Dictionary(MCID=0))),
    )
    page_resources = Dictionary(XObject=Dictionary(Fm=form))
    page = _add_page(pdf, b"/Fm Do", resources=page_resources)

    assert tagging._has_untagged_painting(page, page_resources, set()) is False


def test_untagged_painting_ignores_page_properties_inside_nested_form() -> None:
    """Page-level /Properties cannot protect painting inside a nested form."""
    pdf = new_pdf()
    form = _form_with_named_mcid(
        pdf,
        form_resources=Dictionary(Properties=Dictionary(Other=Dictionary())),
    )
    page_resources = Dictionary(
        XObject=Dictionary(Fm=form),
        Properties=Dictionary(MC0=Dictionary(MCID=0)),
    )
    page = _add_page(pdf, b"/Fm Do", resources=page_resources)

    assert tagging._has_untagged_painting(page, page_resources, set()) is True


def test_untagged_painting_reports_unprotected_painting_in_nested_form() -> None:
    """Painting inside a nested form without matching properties is untagged."""
    pdf = new_pdf()
    form = _form_with_named_mcid(
        pdf,
        form_resources=Dictionary(Properties=Dictionary(Other=Dictionary())),
    )
    page_resources = Dictionary(XObject=Dictionary(Fm=form))
    page = _add_page(pdf, b"/Fm Do", resources=page_resources)

    assert tagging._has_untagged_painting(page, page_resources, set()) is True


def test_untagged_painting_resolves_named_bdc_in_page_resources() -> None:
    pdf = new_pdf()
    page_resources = Dictionary(Properties=Dictionary(MC0=Dictionary(MCID=0)))
    page = _add_page(
        pdf,
        b"/MC0 <</MCID 0>> BDC 0 0 m 5 5 l S EMC",
        resources=page_resources,
    )

    assert tagging._has_untagged_painting(page, page_resources, set()) is False


def test_untagged_painting_reports_unprotected_named_bdc_in_page_resources() -> None:
    pdf = new_pdf()
    page_resources = Dictionary(Properties=Dictionary(MC0=Dictionary()))
    page = _add_page(
        pdf,
        b"/P /MC0 BDC 0 0 m 5 5 l S EMC",
        resources=page_resources,
    )

    assert tagging._has_untagged_painting(page, page_resources, set()) is True


def test_marked_content_properties_resolves_named_reference() -> None:
    properties = tagging._marked_content_properties(
        [Name.P, Name("/MC0")],
        Dictionary(Properties=Dictionary(MC0=Dictionary(MCID=3))),
    )

    assert isinstance(properties, Dictionary)
    assert int(properties.MCID) == 3
    assert (
        tagging._marked_content_properties(
            [Name.P, Name("/Missing")],
            Dictionary(Properties=Dictionary()),
        )
        is None
    )
    assert (
        tagging._marked_content_properties(
            [Name.P, Dictionary(MCID=1)],
            None,
        )
        is not None
    )


def test_scan_content_description_evidence_allows_bounded_form_nesting() -> None:
    pdf = new_pdf()
    form = _nested_form_chain(pdf, tagging._MAX_FORM_SCAN_DEPTH)

    text_mcids, actual_text_mcids, has_text, has_actual_text = (
        tagging._scan_content_description_evidence(
            form,
            resolve_indirect(form.get("/Resources")),
            "bounded form chain",
        )
    )

    assert text_mcids == frozenset()
    assert actual_text_mcids == frozenset()
    assert has_text is False
    assert has_actual_text is False


def test_scan_content_description_evidence_rejects_deep_form_nesting() -> None:
    pdf = new_pdf()
    form = _nested_form_chain(pdf, tagging._MAX_FORM_SCAN_DEPTH + 1)

    with pytest.raises(ConversionError, match="nested deeper than"):
        tagging._scan_content_description_evidence(
            form,
            resolve_indirect(form.get("/Resources")),
            "deep form chain",
        )


def test_sanitize_content_stream_removes_named_mcid() -> None:
    pdf = new_pdf()
    stream = pdf.make_stream(b"/P /MC0 BDC q Q EMC")
    mc0 = Dictionary(MCID=0)
    resources = Dictionary(Properties=Dictionary(MC0=mc0))

    languages_removed, mcids_removed = tagging._sanitize_content_stream(
        stream,
        resources,
        remove_mcids=True,
        description="test stream",
    )

    assert (languages_removed, mcids_removed) == (0, 1)
    assert "/MCID" not in mc0


def test_marked_instruction_without_mcid_resolves_named_reference() -> None:
    instruction = pikepdf.ContentStreamInstruction(
        [Name.P, Name("/MC0")],
        pikepdf.Operator("BDC"),
    )
    resources = Dictionary(
        Properties=Dictionary(MC0=Dictionary(MCID=7, Lang="en-US")),
    )

    resolved, removed, has_properties, named = tagging._marked_instruction_without_mcid(
        instruction, resources, 1
    )

    assert removed is True
    assert has_properties is True
    assert isinstance(named, Dictionary)
    assert str(named.Lang) == "en-US"
    assert resolved is instruction
    assert str(resolved.operands[1]) == "/MC0"


def test_artifact_untagged_path_painting_honors_named_bdc_protection() -> None:
    pdf = new_pdf()
    page_resources = Dictionary(Properties=Dictionary(MC0=Dictionary(MCID=1)))
    page = _add_page(pdf, b"BT 10 10 Td (Body) Tj ET", resources=page_resources)
    root, document = _install_existing_structure(pdf, role=Name.P)
    figure = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Figure,
            P=document,
            Pg=page.obj,
            K=1,
        )
    )
    document["/K"] = Array([0, figure])
    NumberTree(root["/ParentTree"])[0] = Array([document, figure])
    content = resolve_indirect(page.obj["/Contents"])
    assert isinstance(content, pikepdf.Stream)
    content.write(
        bytes(content.read_bytes())
        + b"\n/P /MC0 BDC 0 0 m 5 5 l S EMC"
        + b"\n10 10 m 20 20 l S"
    )

    elements, changed = tagging._artifact_untagged_path_painting(pdf)

    assert changed == 1
    assert elements is not None
    instructions = list(pikepdf.parse_content_stream(page))
    assert (
        sum(
            isinstance(instruction, pikepdf.ContentStreamInstruction)
            and instruction.operator == pikepdf.Operator("BMC")
            and list(instruction.operands) == [Name.Artifact]
            for instruction in instructions
        )
        == 1
    )


class TestStructureHierarchyValidation:
    """PDF 1.7 nesting that is legal must not force a structure rebuild."""

    @staticmethod
    def _tree(pdf: pikepdf.Pdf, roles: tuple[str, ...]) -> list[Dictionary]:
        """Build a linear chain of structure elements below the root."""
        root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
        parent: Dictionary = root
        elements: list[Dictionary] = []
        for role in roles:
            element = pdf.make_indirect(
                Dictionary(Type=Name.StructElem, S=Name(role), P=parent)
            )
            elements.append(element)
            parent = element
        return [root, *elements]

    @pytest.mark.parametrize(
        "roles",
        [
            # PDF 32000-1, Table 336: /Lbl labels list and TOC items alike.
            ("/TOC", "/TOCI", "/Lbl"),
            ("/L", "/LI", "/Lbl"),
            # Table 333: grouping elements nest, so /Document is not root-only.
            ("/Part", "/Document"),
            ("/Document", "/Document"),
            ("/Sect", "/Document"),
        ],
    )
    def test_accepts_legal_nesting(self, roles: tuple[str, ...]) -> None:
        pdf = new_pdf()
        root, *elements = self._tree(pdf, roles)

        assert tagging._valid_structure_hierarchy(root, None, elements) is True

    @pytest.mark.parametrize(
        "roles",
        [
            ("/P", "/Lbl"),
            ("/H1", "/Document"),
            ("/Table", "/TD"),
            ("/P", "/LI"),
            ("/Div", "/TR"),
        ],
    )
    def test_rejects_illegal_nesting(self, roles: tuple[str, ...]) -> None:
        pdf = new_pdf()
        root, *elements = self._tree(pdf, roles)

        assert tagging._valid_structure_hierarchy(root, None, elements) is False

    def test_pdfua_rejects_multiple_table_captions(self) -> None:
        pdf = new_pdf()
        root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
        table = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=Name.Table, P=root)
        )
        captions = [
            pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name.Caption, P=table))
            for _ in range(2)
        ]
        row = pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name.TR, P=table))
        cell = pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name.TD, P=row))
        table["/K"] = Array([*captions, row])
        row["/K"] = cell
        elements = [table, *captions, row, cell]

        assert tagging._valid_structure_hierarchy(root, None, elements) is True
        assert (
            tagging._valid_structure_hierarchy(
                root,
                None,
                elements,
                pdfua=True,
            )
            is False
        )

    def test_pdfua_accepts_rectangular_table_with_column_span(self) -> None:
        pdf = new_pdf()
        root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
        table = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=Name.Table, P=root)
        )
        first_row = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=Name.TR, P=table)
        )
        second_row = pdf.make_indirect(
            Dictionary(Type=Name.StructElem, S=Name.TR, P=table)
        )
        first_cell = pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.TD,
                P=first_row,
                A=Dictionary(O=Name.Table, ColSpan=2),
            )
        )
        other_cells = [
            pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name.TD, P=second_row))
            for _ in range(2)
        ]
        table["/K"] = Array([first_row, second_row])
        first_row["/K"] = first_cell
        second_row["/K"] = Array(other_cells)

        assert tagging._valid_structure_hierarchy(
            root,
            None,
            [table, first_row, first_cell, second_row, *other_cells],
            pdfua=True,
        )


class TestPreflightOptOut:
    """`preflight=False` skips the rehearsal without changing the result."""

    def test_default_rehearses_on_an_isolated_copy(self) -> None:
        pdf = new_pdf()
        _add_page(pdf, b"BT ET")
        calls: list[bool] = []
        original = tagging._ensure_logical_structure_in_place

        def record(target, **kwargs):
            calls.append(target is pdf)
            return original(target, **kwargs)

        with patch.object(
            tagging,
            "_ensure_logical_structure_in_place",
            side_effect=record,
        ):
            ensure_logical_structure(pdf, semantic=True)

        assert calls == [False, True]

    def test_opt_out_runs_once_on_the_caller_pdf(self) -> None:
        pdf = new_pdf()
        _add_page(pdf, b"BT ET")
        calls: list[bool] = []
        original = tagging._ensure_logical_structure_in_place

        def record(target, **kwargs):
            calls.append(target is pdf)
            return original(target, **kwargs)

        with patch.object(
            tagging,
            "_ensure_logical_structure_in_place",
            side_effect=record,
        ):
            ensure_logical_structure(pdf, semantic=True, preflight=False)

        assert calls == [True]

    def test_opt_out_produces_the_same_structure(self) -> None:
        results = []
        for preflight in (True, False):
            pdf = new_pdf()
            _add_page(pdf, b"BT ET")
            results.append(
                ensure_logical_structure(pdf, semantic=True, preflight=preflight)
            )

        assert results[0] == results[1]
