# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Preserve already conforming features through a full PDF/A conversion."""

from pathlib import Path

import pikepdf
import pytest
from conftest import new_pdf
from lxml import etree
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import (
    ColorSpaceType,
    create_output_intent_for_colorspace,
    get_cmyk_profile,
    get_gray_profile,
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
    pdf = new_pdf()
    pdf.add_blank_page(page_size=(200, 200))
    sync_metadata(pdf, level)
    colorspace = ColorSpaceType.DEVICE_CMYK if cmyk else ColorSpaceType.DEVICE_RGB
    profile = get_cmyk_profile() if cmyk else get_srgb_profile()
    intent = create_output_intent_for_colorspace(pdf, colorspace, profile, level)
    pdf.Root.OutputIntents = Array([pdf.make_indirect(intent)])
    return pdf


def _form(pdf: Pdf, data: bytes, size: int = 200):
    form = pdf.make_stream(data)
    form.Type = Name.XObject
    form.Subtype = Name.Form
    form.BBox = Array([0, 0, size, size])
    form.Resources = Dictionary()
    return form


def _add_preserved_features(pdf: Pdf) -> None:
    """Construct the original data directly, without the sanitizers under test."""
    page = pdf.pages[0]
    rgb = Array([Name.ICCBased, pdf.Root.OutputIntents[0].DestOutputProfile])
    gray_profile = pdf.make_stream(get_gray_profile())
    gray_profile.N = 1
    mask = _form(pdf, b"0.25 g 0 0 200 200 re f")
    mask.Group = Dictionary(
        S=Name.Transparency, CS=Array([Name.ICCBased, gray_profile])
    )
    transfer = Dictionary(
        FunctionType=2,
        Domain=Array([0, 1]),
        Range=Array([0, 1]),
        C0=Array([1]),
        C1=Array([0]),
        N=1,
    )
    calibration = Dictionary(
        WhitePoint=Array([0.9505, 1, 1.089]),
        Gamma=Array([1, 1, 1]),
        Matrix=Array(
            [0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505]
        ),
    )
    page.Resources = Dictionary(
        ColorSpace=Dictionary(Cal=Array([Name.CalRGB, calibration])),
        ExtGState=Dictionary(
            Mask=Dictionary(
                Type=Name.ExtGState,
                SMask=Dictionary(S=Name.Luminosity, G=mask, TR=transfer),
            )
        ),
    )
    page.Group = Dictionary(S=Name.Transparency, CS=rgb)
    page.Contents = pdf.make_stream(
        b"q /Mask gs 1 0 0 rg 10 10 80 80 re f Q "
        b"/Cal cs 0.5 0.25 0.1 scn 110 10 80 80 re f"
    )
    appearance = _form(pdf, b"0.5 g 0 0 20 20 re f", size=20)
    square = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Square,
            Rect=Array([50, 150, 70, 170]),
            F=4,
            CA=0.4,
            C=Array([1, 0, 0]),
            IC=Array([1, 1, 0]),
            AP=Dictionary(N=appearance),
        )
    )
    link = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([10, 100, 180, 120]),
            Border=Array([0, 0, 2]),
            C=Array([1, 0, 0]),
            F=4,
            A=Dictionary(S=Name.URI, URI="https://example.com"),
        )
    )
    trap = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.TrapNet,
            Rect=Array([20, 150, 40, 170]),
            F=4,
            LastModified="D:20261106000000Z",
            AP=Dictionary(N=appearance),
        )
    )
    page.Annots = Array([square, link, trap])
    pdf.Root.Collection = Dictionary(Type=Name.Collection, View=Name.D)
    thread = pdf.make_indirect(
        Dictionary(Type=Name.Thread, I=Dictionary(Title="Article A"))
    )
    bead = pdf.make_indirect(
        Dictionary(Type=Name.Bead, T=thread, P=page.obj, R=Array([0, 0, 100, 100]))
    )
    bead.N = bead
    bead.V = bead
    thread.F = bead
    page.B = Array([bead])
    pdf.Root.Threads = Array([thread])
    pdf.Root.OutputIntents.append(
        pdf.make_indirect(
            Dictionary(
                Type=Name.OutputIntent,
                S=Name.GTS_PDFX,
                OutputConditionIdentifier="Shared profile",
                DestOutputProfile=pdf.Root.OutputIntents[0].DestOutputProfile,
            )
        )
    )
    pdf.docinfo["/Company"] = "Acme Corp"
    # Inject the original XMP after sync_metadata, so fixture generation cannot
    # silently reproduce the same stripping bug that the conversion should fix.
    tree = etree.fromstring(pdf.Root.Metadata.read_bytes())
    description = tree.find(f".//{{{_RDF}}}Description")
    assert description is not None
    for xml in (
        f'<xmpMM:History xmlns:xmpMM="{_MM}" xmlns:rdf="{_RDF}" '
        'xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#">'
        '<rdf:Seq><rdf:li rdf:parseType="Resource"><stEvt:action>saved</stEvt:action>'
        "<stEvt:when>2024-01-01T00:00:00Z</stEvt:when></rdf:li></rdf:Seq>"
        "</xmpMM:History>",
        f'<xmpMM:DerivedFrom xmlns:xmpMM="{_MM}" xmlns:rdf="{_RDF}" '
        'xmlns:stRef="http://ns.adobe.com/xap/1.0/sType/ResourceRef#" '
        'rdf:parseType="Resource"><stRef:instanceID>'
        "uuid:00000000-0000-0000-0000-000000000001"
        "</stRef:instanceID></xmpMM:DerivedFrom>",
        f'<dc:title xmlns:dc="{_DC}" xmlns:rdf="{_RDF}"><rdf:Alt>'
        '<rdf:li xml:lang="x-default">Report</rdf:li>'
        '<rdf:li xml:lang="de-DE">Bericht</rdf:li>'
        '<rdf:li xml:lang="fr-FR">Rapport</rdf:li>'
        '<rdf:li xml:lang="en-GB"/>'
        '<rdf:li xml:lang="en-US">None</rdf:li></rdf:Alt></dc:title>',
        f'<dc:creator xmlns:dc="{_DC}" xmlns:rdf="{_RDF}"><rdf:Seq>'
        "<rdf:li>Alice</rdf:li><rdf:li>Bob</rdf:li></rdf:Seq></dc:creator>",
    ):
        description.append(etree.fromstring(xml.encode()))
    pdf.Root.Metadata.write(etree.tostring(tree))


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_preserves_conforming_features(
    source_level: str, target_level: str, tmp_path: Path
) -> None:
    pdf = _document(source_level)
    _add_preserved_features(pdf)
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")
    original = validate_with_verapdf(source, flavour=source_level)
    assert original.compliant, original

    # Change the PDF/A part so the already-conforming shortcut cannot hide the
    # behavior of the actual conversion pipeline.
    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.success and result.target_produced and not result.skipped, result
    assert not result.validation_failed, result
    with pikepdf.open(output) as restored:
        page = restored.pages[0]
        transfer = page.Resources.ExtGState.Mask.SMask.TR
        assert list(transfer.C0) == [1] and list(transfer.C1) == [0]
        assert page.Resources.ColorSpace.Cal == pdf.pages[0].Resources.ColorSpace.Cal
        annotations = {str(annot.Subtype): annot for annot in page.Annots}
        assert set(annotations) == {"/Square", "/Link", "/TrapNet"}
        assert float(annotations["/Square"].CA) == 0.4
        assert list(annotations["/Square"].C) == [1, 0, 0]
        assert list(annotations["/Square"].IC) == [1, 1, 0]
        assert list(annotations["/Link"].C) == [1, 0, 0]
        assert restored.Root.Collection.View == Name.D
        assert str(restored.Root.Threads[0].I.Title) == "Article A"
        assert restored.Root.Threads[0].F.P.objgen == page.objgen
        intents = restored.Root.OutputIntents
        assert [str(intent.S) for intent in intents] == ["/GTS_PDFA1", "/GTS_PDFX"]
        assert (
            intents[0].DestOutputProfile.objgen == intents[1].DestOutputProfile.objgen
        )
        assert str(restored.docinfo["/Company"]) == "Acme Corp"
        tree = etree.fromstring(restored.Root.Metadata.read_bytes())
        assert tree.find(f".//{{{_MM}}}History") is not None
        assert tree.find(f".//{{{_MM}}}DerivedFrom") is not None
        titles = tree.xpath(
            "//dc:title/rdf:Alt/rdf:li", namespaces={"dc": _DC, "rdf": _RDF}
        )
        assert [item.text or "" for item in titles] == [
            "Report",
            "Bericht",
            "Rapport",
            "",
            "None",
        ]
        assert tree.xpath(
            "//dc:creator/rdf:Seq/rdf:li/text()", namespaces={"dc": _DC, "rdf": _RDF}
        ) == ["Alice", "Bob"]


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
def test_full_conversion_preserves_overprint_on_an_unrelated_page(
    source_level: str, target_level: str, tmp_path: Path
) -> None:
    pdf = _document(source_level, cmyk=True)
    first = pdf.pages[0]
    first.Resources = Dictionary(
        ExtGState=Dictionary(
            GS=Dictionary(Type=Name.ExtGState, OP=True, op=True, OPM=1)
        )
    )
    first.Contents = pdf.make_stream(
        b"1 0 0 0 k 20 20 160 160 re f /GS gs 0 0 0 1 k 60 60 80 80 re f"
    )
    second = pdf.add_blank_page(page_size=(200, 200))
    second.Resources = Dictionary(
        ColorSpace=Dictionary(
            ICC=Array([Name.ICCBased, pdf.Root.OutputIntents[0].DestOutputProfile])
        )
    )
    second.Contents = pdf.make_stream(b"/ICC cs 0 1 0 0 scn 20 20 160 160 re f")
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")
    original = validate_with_verapdf(source, flavour=source_level)
    assert original.compliant, original
    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.success and result.target_produced and not result.skipped, result
    assert not result.validation_failed, result
    with pikepdf.open(output) as restored:
        gs = restored.pages[0].Resources.ExtGState.GS
        assert gs.OPM == 1 and gs.OP is True and gs.op is True
        assert restored.pages[0].Contents.read_bytes() == first.Contents.read_bytes()


@pytest.mark.parametrize("source_level,target_level", [("2b", "3b"), ("3b", "2b")])
@pytest.mark.parametrize("kind", ["tiling", "shading"])
def test_full_conversion_preserves_valid_pattern_overprint(
    source_level: str, target_level: str, kind: str, tmp_path: Path
) -> None:
    pdf = _document(source_level, cmyk=True)
    page = pdf.pages[0]
    icc = Array([Name.ICCBased, pdf.Root.OutputIntents[0].DestOutputProfile])
    page.Resources = Dictionary()
    if kind == "tiling":
        pattern = pdf.make_stream(b"/ICC cs 0 1 0 0 scn 0 0 20 20 re f")
        pattern.PatternType = 1
        pattern.PaintType = 1
        pattern.TilingType = 1
        pattern.BBox = Array([0, 0, 20, 20])
        pattern.XStep = pattern.YStep = 20
        pattern.Resources = Dictionary(ColorSpace=Dictionary(ICC=icc))
        page.Resources.ExtGState = Dictionary(GS=Dictionary(OP=True, OPM=1))
        content = b"/GS gs /Pattern cs /P scn 0 0 100 100 re f"
    else:
        pattern = Dictionary(
            PatternType=2,
            ExtGState=Dictionary(OP=True, op=False, OPM=1),
            Shading=Dictionary(
                ShadingType=2,
                ColorSpace=icc,
                Coords=Array([0, 0, 100, 0]),
                Extend=Array([True, True]),
                Function=Dictionary(
                    FunctionType=2,
                    Domain=Array([0, 1]),
                    C0=Array([1, 0, 0, 0]),
                    C1=Array([0, 1, 0, 0]),
                    N=1,
                ),
            ),
        )
        content = b"/Pattern cs /P scn 0 0 100 100 re f"
    page.Resources.Pattern = Dictionary(P=pattern)
    page.Contents = pdf.make_stream(content)
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source, force_version="1.7")
    original = validate_with_verapdf(source, flavour=source_level)
    assert original.compliant, original
    result = convert_to_pdfa(source, output, level=target_level, validate=True)
    assert result.success and result.target_produced and not result.skipped, result
    assert not result.validation_failed, result
    with pikepdf.open(output) as restored:
        assert restored.pages[0].Contents.read_bytes() == content
        actual = restored.pages[0].Resources.Pattern.P
        if kind == "tiling":
            assert actual.read_bytes() == pattern.read_bytes()
        else:
            assert actual.ExtGState.OPM == 1
            assert actual.ExtGState.OP is True and actual.ExtGState.op is False
