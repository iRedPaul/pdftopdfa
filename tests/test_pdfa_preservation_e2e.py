# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Preserve already conforming features through a full PDF/A conversion."""

from pathlib import Path

import pikepdf
import pytest
from conftest import new_pdf
from lxml import etree
from pikepdf import Array, Name, Pdf

from pdftopdfa.color_profile import (
    ColorSpaceType,
    create_output_intent_for_colorspace,
    get_cmyk_profile,
    get_srgb_profile,
)
from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.metadata import sync_metadata
from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf

pytestmark = pytest.mark.skipif(
    not is_verapdf_available(), reason="veraPDF is not installed"
)

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_DC = "http://purl.org/dc/elements/1.1/"
_MM = "http://ns.adobe.com/xap/1.0/mm/"


def _document(level: str, *, cmyk: bool = False) -> Pdf:
    """Create a conforming document with a matching output profile."""
    pdf = new_pdf()
    pdf.add_blank_page(page_size=(200, 200))
    sync_metadata(pdf, level)
    colorspace = ColorSpaceType.DEVICE_CMYK if cmyk else ColorSpaceType.DEVICE_RGB
    profile = get_cmyk_profile() if cmyk else get_srgb_profile()
    intent = create_output_intent_for_colorspace(pdf, colorspace, profile, level)
    pdf.Root.OutputIntents = Array([pdf.make_indirect(intent)])
    return pdf


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_repairs_malformed_preserved_xmp(
    source_level: str, target_level: str, tmp_path: Path
) -> None:
    """Invalid arrays and provenance cannot survive as copied metadata packets."""
    pdf = _document(source_level)
    event = "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#"
    ref = "http://ns.adobe.com/xap/1.0/sType/ResourceRef#"
    tree = etree.fromstring(pdf.Root.Metadata.read_bytes())
    description = tree.find(f".//{{{_RDF}}}Description")
    properties = (
        f'<dc:title xmlns:dc="{_DC}" xmlns:rdf="{_RDF}"><rdf:Alt>'
        '<rdf:li xml:lang="x-default">Report</rdf:li>'
        '<rdf:li xml:lang="x-default">Duplicate</rdf:li>'
        "<rdf:notAnItem>bad</rdf:notAnItem></rdf:Alt></dc:title>",
        f'<dc:creator xmlns:dc="{_DC}" xmlns:rdf="{_RDF}"><rdf:Seq>'
        "<rdf:li>Alice</rdf:li><rdf:notAnItem>bad</rdf:notAnItem>"
        "</rdf:Seq></dc:creator>",
        f'<xmpMM:History xmlns:xmpMM="{_MM}" xmlns:rdf="{_RDF}" '
        f'xmlns:stEvt="{event}"><rdf:Seq><rdf:li rdf:parseType="Resource" '
        'stEvt:notAField="bad"/></rdf:Seq></xmpMM:History>',
        f'<xmpMM:DerivedFrom xmlns:xmpMM="{_MM}" xmlns:rdf="{_RDF}" '
        f'xmlns:stRef="{ref}" rdf:parseType="Literal">'
        "<stRef:instanceID>uuid:source</stRef:instanceID></xmpMM:DerivedFrom>",
    )
    for xml in properties:
        description.append(etree.fromstring(xml.encode()))
    pdf.Root.Metadata.write(etree.tostring(tree))
    page_tree = etree.fromstring(
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="{_RDF}" '
        f'xmlns:xmpMM="{_MM}" xmlns:stEvt="{event}"><rdf:RDF>'
        '<rdf:Description rdf:about=""><xmpMM:History><rdf:Seq>'
        '<rdf:li rdf:parseType="Resource" stEvt:When="yesterday"/>'
        "</rdf:Seq></xmpMM:History></rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    metadata = pdf.make_stream(etree.tostring(page_tree))
    metadata.Type, metadata.Subtype = Name.Metadata, Name.XML
    pdf.pages[0].Metadata = metadata
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")
    assert not validate_with_verapdf(source, flavour=source_level).compliant
    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.success and result.target_produced and not result.validation_failed
    with pikepdf.open(output) as restored:
        for owner in (restored.Root, restored.pages[0].obj):
            actual = etree.fromstring(owner.Metadata.read_bytes())
            assert not actual.xpath(
                "//xmpMM:History | //xmpMM:DerivedFrom", namespaces={"xmpMM": _MM}
            )
        actual = etree.fromstring(restored.Root.Metadata.read_bytes())
        assert actual.xpath(
            "//dc:title/rdf:Alt/rdf:li/text()", namespaces={"dc": _DC, "rdf": _RDF}
        ) == ["Report"]
        assert actual.xpath(
            "//dc:creator/rdf:Seq/rdf:li/text()", namespaces={"dc": _DC, "rdf": _RDF}
        ) == ["Alice"]


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_rejects_binary_custom_docinfo(
    source_level, target_level, tmp_path
):
    """Reject invalid text encodings while retaining valid PDFDoc and Unicode text."""
    pdf = _document(source_level)
    invalid = {
        "/Binary": b"invalid\x9f",
        "/UndefinedControl": b"invalid\x00",
        "/OddUnicode": b"\xfe\xff\x00",
        "/Surrogate": b"\xfe\xff\xd8\x00",
    }
    valid = {
        "/PDFDoc": b"\x80\x95\xa0",
        "/Unicode": b"\xfe\xff\xd8\x3d\xde\x00",
    }
    for key, value in {**invalid, **valid}.items():
        pdf.docinfo[key] = pikepdf.String(value)
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")
    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.target_produced and not result.validation_failed, result
    with pikepdf.open(output) as restored:
        assert not any(key in restored.docinfo for key in invalid)
        assert {key: bytes(restored.docinfo[key]) for key in valid} == valid


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_synchronizes_first_creator(
    source_level, target_level, tmp_path
):
    """An Author found later in the creator list moves to the first entry."""
    pdf = _document(source_level)
    tree = etree.fromstring(pdf.Root.Metadata.read_bytes())
    description = tree.find(f".//{{{_RDF}}}Description")
    creator = etree.SubElement(description, f"{{{_DC}}}creator")
    sequence = etree.SubElement(creator, f"{{{_RDF}}}Seq")
    for author in ("Alice", "Bob"):
        etree.SubElement(sequence, f"{{{_RDF}}}li").text = author
    pdf.Root.Metadata.write(etree.tostring(tree))
    pdf.docinfo["/Author"] = "Bob"
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")

    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.target_produced and not result.validation_failed, result
    with pikepdf.open(output) as restored:
        tree = etree.fromstring(restored.Root.Metadata.read_bytes())
        assert tree.xpath(
            "//dc:creator/rdf:Seq/rdf:li/text()", namespaces={"dc": _DC, "rdf": _RDF}
        ) == ["Bob", "Alice"]
        assert str(restored.docinfo["/Author"]) == "Bob"


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_keeps_conforming_provenance_parts(
    source_level, target_level, tmp_path
):
    """Fields and events PDF/A does not accept cannot cost the conforming rest."""
    xmp_mm = "http://ns.adobe.com/xap/1.0/mm/"
    st_ref = "http://ns.adobe.com/xap/1.0/sType/ResourceRef#"
    st_evt = "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#"
    namespaces = {"xmpMM": xmp_mm, "stRef": st_ref, "stEvt": st_evt, "rdf": _RDF}
    pdf = _document(source_level)
    tree = etree.fromstring(pdf.Root.Metadata.read_bytes())
    description = tree.find(f".//{{{_RDF}}}Description")
    properties = etree.fromstring(
        f'<w xmlns:xmpMM="{xmp_mm}" xmlns:stRef="{st_ref}" xmlns:stEvt="{st_evt}" '
        f'xmlns:rdf="{_RDF}">'
        '<xmpMM:DerivedFrom rdf:parseType="Resource">'
        "<stRef:instanceID>uuid:source</stRef:instanceID>"
        "<stRef:originalDocumentID>uuid:original</stRef:originalDocumentID>"
        "<stRef:filePath>file:///source.pdf</stRef:filePath>"
        "<stRef:manageTo>not a uri</stRef:manageTo>"
        "</xmpMM:DerivedFrom><xmpMM:History><rdf:Seq>"
        '<rdf:li stEvt:action="saved" stEvt:when="2024-01-01T00:00:00Z" '
        'stEvt:softwareAgent="Tool" stEvt:changed="/"/>'
        '<rdf:li stEvt:action="converted" stEvt:parameters="from A to B"/>'
        "</rdf:Seq></xmpMM:History></w>"
    )
    for prop in properties:
        description.append(prop)
    pdf.Root.Metadata.write(etree.tostring(tree))
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")

    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.target_produced and not result.validation_failed, result
    with pikepdf.open(output) as restored:
        tree = etree.fromstring(restored.Root.Metadata.read_bytes())
        assert tree.xpath("//stRef:instanceID/text()", namespaces=namespaces) == [
            "uuid:source"
        ]
        assert not tree.xpath(
            "//stRef:originalDocumentID | //stRef:filePath | //stRef:manageTo",
            namespaces=namespaces,
        )
        assert tree.xpath("//@stEvt:action", namespaces=namespaces) == ["saved"]
        assert not tree.xpath("//@stEvt:changed", namespaces=namespaces)


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_repairs_xmp_language_tags(
    source_level, target_level, tmp_path
):
    """Malformed language qualifiers neither cost values nor conformance."""
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    pdf = _document(source_level)
    tree = etree.fromstring(pdf.Root.Metadata.read_bytes())
    description = tree.find(f".//{{{_RDF}}}Description")
    description.set(xml_lang, "en")
    rights = etree.SubElement(description, f"{{{_DC}}}rights")
    alternatives = etree.SubElement(rights, f"{{{_RDF}}}Alt")
    item = etree.SubElement(alternatives, f"{{{_RDF}}}li")
    item.set(xml_lang, "en_US")
    item.text = "Copyright Acme"
    pdf.Root.Metadata.write(etree.tostring(tree))
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")

    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.target_produced and not result.validation_failed, result
    with pikepdf.open(output) as restored:
        tree = etree.fromstring(restored.Root.Metadata.read_bytes())
        items = tree.xpath(
            "//dc:rights/rdf:Alt/rdf:li", namespaces={"dc": _DC, "rdf": _RDF}
        )
        assert [(item.get(xml_lang), item.text) for item in items] == [
            ("en-US", "Copyright Acme")
        ]
        assert tree.find(f".//{{{_RDF}}}Description").get(xml_lang) is None
