# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Retain complete trap networks and flatten or reject malformed ones."""

import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.trapnet import validate_preserved_trapnets


def _trap(pdf, *, version=False):
    """Build a complete single-network annotation with either tracking method."""
    page = pdf.add_blank_page()
    page.Resources = Dictionary()
    page.Contents = pdf.make_stream(b"0.5 g 0 0 100 100 re f")
    appearance = pdf.make_stream(b"0 g 0 0 20 20 re f")
    appearance.Type, appearance.Subtype = Name.XObject, Name.Form
    appearance.BBox = Array([0, 0, 20, 20])
    appearance.Resources = Dictionary()
    appearance.PCM = Name.DeviceGray
    trap = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.TrapNet,
            F=68,
            AS=Name.Normal,
            AP=Dictionary(N=appearance),
            Rect=Array([0, 0, 20, 20]),
        )
    )
    if version:
        trap.Version, trap.AnnotStates = Array([page.Contents]), Array()
    else:
        trap.LastModified = "D:20261106000000Z"
    page.Annots = Array([trap])
    return page, trap


@pytest.mark.parametrize("version", [False, True])
def test_complete_trap_network_is_preserved(version):
    """Both specified tracking methods preserve the annotation and appearance."""
    pdf = new_pdf()
    page, trap = _trap(pdf, version=version)
    before = trap.AP.N.read_bytes()
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == 0
    assert page.Annots[0].objgen == trap.objgen
    assert trap.AP.N.read_bytes() == before
    validate_preserved_trapnets(pdf)


@pytest.mark.parametrize(
    "damage", ["date", "bad_date", "flags", "state", "pcm", "tracking", "version"]
)
def test_malformed_trap_network_appearance_is_flattened(damage):
    """A usable appearance survives even when subtype-specific metadata is invalid."""
    pdf = new_pdf()
    page, trap = _trap(pdf, version=damage == "version")
    appearance = trap.AP.N
    if damage == "date":
        del trap.LastModified
    elif damage == "bad_date":
        trap.LastModified = "D:2024garbage"
    elif damage == "flags":
        trap.F = 4
    elif damage == "state":
        del trap.AS
    elif damage == "pcm":
        del appearance.PCM
    elif damage == "tracking":
        trap.Version = Array()
    else:
        trap.Version = Array()
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == 1
    assert "/Annots" not in page.obj
    assert any(
        obj.objgen == appearance.objgen for obj in page.Resources.XObject.values()
    )


def test_trap_without_usable_appearance_is_removed():
    """An invalid trap cannot survive when its appearance cannot be flattened."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    del trap.AP
    result = sanitize_for_pdfa(pdf)
    assert result["forbidden_annotations_removed"] == 1
    assert "/Annots" not in page.obj


@pytest.mark.parametrize(
    "field,value",
    [
        ("/BBox", Array([0, 0, 20])),
        ("/Matrix", Name.Bad),
        ("/FormType", True),
        ("/Resources", Name.Bad),
    ],
)
def test_malformed_appearance_is_not_reused_as_a_form(field, value):
    """Flattening must not introduce a structurally invalid Form into page content."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    trap.AP.N[field] = value
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == 0
    assert result["forbidden_annotations_removed"] == 1
    assert "/Annots" not in page.obj and "/XObject" not in page.Resources


@pytest.mark.parametrize("damage", ["order", "duplicate"])
def test_trap_order_and_cardinality_are_enforced(damage):
    """There must be one trap network, last in the page annotation array."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    if damage == "order":
        page.Annots.append(
            pdf.make_indirect(
                Dictionary(Subtype=Name.Link, F=4, Rect=Array([0, 0, 20, 20]))
            )
        )
    else:
        page.Annots.append(pdf.make_indirect(Dictionary(trap)))
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == (
        1 if damage == "order" else 2
    )
    assert all(
        annotation.Subtype != Name.TrapNet for annotation in page.get("/Annots", [])
    )


def test_annotation_removal_invalidates_saved_annotation_states():
    """Removing another annotation must not leave stale trap tracking behind."""
    pdf = new_pdf()
    page, trap = _trap(pdf, version=True)
    sound = pdf.make_indirect(
        Dictionary(Subtype=Name.Sound, Rect=Array([0, 0, 20, 20]))
    )
    page.Annots = Array([sound, trap])
    trap.AnnotStates = Array([None])
    result = sanitize_for_pdfa(pdf)
    assert result["forbidden_annotations_removed"] == 1
    assert result["non_compliant_annotations_flattened"] == 1
    assert "/Annots" not in page.obj


def test_late_content_changes_cannot_publish_stale_version():
    """Recheck saved references after conversion rewrites content streams."""
    pdf = new_pdf()
    page, _ = _trap(pdf, version=True)
    validate_preserved_trapnets(pdf)
    page.Contents = pdf.make_stream(page.Contents.read_bytes())
    with pytest.raises(ConversionError, match="invalidated a preserved TrapNet"):
        validate_preserved_trapnets(pdf)
