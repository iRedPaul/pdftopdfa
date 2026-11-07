# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Retain complete trap networks and flatten or reject malformed ones."""

import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.trapnet import (
    capture_trapnet_snapshots,
    validate_preserved_trapnets,
)


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
    snapshots = capture_trapnet_snapshots(pdf)
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == 0
    assert page.Annots[0].objgen == trap.objgen
    assert trap.AP.N.read_bytes() == before
    validate_preserved_trapnets(pdf, snapshots)


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
    "rect",
    [
        None,
        Array([0, 0, 20]),
        Array([0, 0, 20, 20, 20]),
        Array([0, 0, 20, Name.Bad]),
        Array([0, 0, 20, True]),
        Name.Bad,
    ],
)
@pytest.mark.parametrize("version", [False, True])
def test_trap_without_valid_rect_is_removed(rect, version):
    """Without a position, the appearance can be neither kept nor flattened."""
    pdf = new_pdf()
    page, trap = _trap(pdf, version=version)
    before = page.Contents.read_bytes()
    if rect is None:
        del trap.Rect
    else:
        trap.Rect = rect
    result = sanitize_for_pdfa(pdf)
    assert result["non_compliant_annotations_flattened"] == 0
    assert result["forbidden_annotations_removed"] == 1
    assert "/Annots" not in page.obj and "/XObject" not in page.Resources
    assert page.Contents.read_bytes() == before


def test_conversion_removes_trap_without_rect(tmp_path):
    """A trap that cannot be retained does not force the original-input fallback."""
    from pikepdf import Pdf

    import pdftopdfa.converter as converter

    pdf = new_pdf()
    _, trap = _trap(pdf)
    del trap.Rect
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = converter.convert_to_pdfa(source, output, level="3b")
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        assert "/Annots" not in converted.pages[0].obj


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
    snapshots = capture_trapnet_snapshots(pdf)
    validate_preserved_trapnets(pdf, snapshots)
    page.Contents = pdf.make_stream(page.Contents.read_bytes())
    with pytest.raises(ConversionError, match="invalidated a preserved TrapNet"):
        validate_preserved_trapnets(pdf, snapshots)


@pytest.mark.parametrize("page_date", [None, "D:20200101000000Z"])
@pytest.mark.parametrize(
    "change", ["content", "in_place", "resources", "nested", "inherited", "state"]
)
def test_late_page_changes_invalidate_date_tracking(page_date, change):
    """Dates cannot hide changed streams or resource values, including shared ones."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    if page_date is not None:
        page.LastModified = page_date
    state = pdf.make_indirect(Dictionary(CA=1))
    form = pdf.make_stream(b"0 g 0 0 10 10 re f")
    form.Subtype = Name.Form
    form.Resources = Dictionary(ExtGState=Dictionary(GS=state))
    page.Resources.XObject = Dictionary(Form=form)
    if change == "inherited":
        page.Parent.Resources = page.Resources
        del page.Resources
    other = pdf.make_indirect(Dictionary(Subtype=Name.Square, AS=Name.Normal))
    page.Annots = Array([other, trap])
    snapshots = capture_trapnet_snapshots(pdf)
    validate_preserved_trapnets(pdf, snapshots)
    if change == "content":
        page.Contents = pdf.make_stream(b"0 g 0 0 200 200 re f")
    elif change == "in_place":
        page.Contents.write(b"0 g 0 0 200 200 re f")
    elif change in {"nested", "inherited"}:
        state.CA = 0.5
    elif change == "resources":
        page.Resources.ColorSpace = Dictionary(DefaultGray=Name.DeviceGray)
    else:
        other.AS = Name.Changed
    with pytest.raises(ConversionError, match="invalidated a preserved TrapNet"):
        validate_preserved_trapnets(pdf, snapshots)


def test_in_place_content_changes_also_invalidate_version_tracking():
    """Reference identity alone cannot prove that a Version-tracked page is intact."""
    pdf = new_pdf()
    page, _ = _trap(pdf, version=True)
    snapshots = capture_trapnet_snapshots(pdf)
    page.Contents.write(b"1 g 0 0 100 100 re f")
    with pytest.raises(ConversionError, match="invalidated a preserved TrapNet"):
        validate_preserved_trapnets(pdf, snapshots)


def _sibling_with_all_appearances(pdf):
    """Build an annotation that has normal, rollover, and down appearances."""

    def form(content):
        appearance = pdf.make_stream(content)
        appearance.Type, appearance.Subtype = Name.XObject, Name.Form
        appearance.BBox = Array([0, 0, 20, 20])
        appearance.Resources = Dictionary()
        return appearance

    return pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Square,
            F=4,
            Rect=Array([0, 0, 20, 20]),
            AP=Dictionary(
                N=form(b"0 g 0 0 20 20 re f"),
                R=form(b"0.5 g 0 0 20 20 re f"),
                D=form(b"1 g 0 0 20 20 re f"),
            ),
        )
    )


@pytest.mark.parametrize("key", ["/R", "/D"])
def test_non_normal_appearances_do_not_affect_snapshots(key):
    """Only the normal appearance of another annotation can affect trapping."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    sibling = _sibling_with_all_appearances(pdf)
    page.Annots = Array([sibling, trap])
    snapshots = capture_trapnet_snapshots(pdf)
    del sibling.AP[key]
    validate_preserved_trapnets(pdf, snapshots)


def test_changed_normal_appearance_still_invalidates_snapshots():
    """Reducing /AP to /N must not hide a change to the printed appearance."""
    pdf = new_pdf()
    page, trap = _trap(pdf)
    sibling = _sibling_with_all_appearances(pdf)
    page.Annots = Array([sibling, trap])
    snapshots = capture_trapnet_snapshots(pdf)
    sibling.AP.N.write(b"1 g 0 0 20 20 re f")
    with pytest.raises(ConversionError, match="invalidated a preserved TrapNet"):
        validate_preserved_trapnets(pdf, snapshots)


def test_snapshots_ignore_metadata_and_other_page_changes():
    """Unrelated edits cannot invalidate an unchanged page's date-tracked trap."""
    pdf = new_pdf()
    page, _ = _trap(pdf)
    page.Resources.ProcSet = Array([Name.PDF])
    other = pdf.add_blank_page()
    snapshots = capture_trapnet_snapshots(pdf)
    pdf.docinfo["/Title"] = "Updated"
    page.Resources.ProcSet.append(Name.Text)
    other.Contents = pdf.make_stream(b"0 g 0 0 10 10 re f")
    validate_preserved_trapnets(pdf, snapshots)


def test_snapshots_survive_saving_and_object_renumbering(tmp_path):
    """Compression and object numbers alone do not change a page description."""
    from pikepdf import Pdf

    pdf = new_pdf()
    pdf.make_indirect(Dictionary(Unused="Renumber subsequent objects"))
    page, _ = _trap(pdf)
    form = pdf.make_stream(b"0 g 0 0 20 20 re f")
    form.Resources = Dictionary(XObject=Dictionary(Self=form))
    page.Resources.XObject = Dictionary(Form=form)
    snapshots = capture_trapnet_snapshots(pdf)
    path = tmp_path / "roundtrip.pdf"
    pdf.save(path)
    with Pdf.open(path) as restored:
        assert restored.pages[0].objgen != page.objgen
        validate_preserved_trapnets(restored, snapshots)


@pytest.mark.parametrize("change", [None, "content", "resource"])
def test_conversion_rejects_stale_date_traps_after_late_edits(
    change, tmp_path, monkeypatch
):
    """Final conversion checks use the original page, before profile/content edits."""
    import pdftopdfa.converter as converter

    pdf = new_pdf()
    page, trap = _trap(pdf)
    trap.LastModified = "D:20990101000000Z"
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    embed_profiles = converter.embed_color_profiles

    def rewrite_page(pdf, *args, **kwargs):
        """Apply a late edit without updating either stored modification date."""
        result = embed_profiles(pdf, *args, **kwargs)
        if change == "content":
            pdf.pages[0].Contents.write(b"1 g 0 0 100 100 re f")
        elif change == "resource":
            pdf.pages[0].Resources.ExtGState = Dictionary(GS=Dictionary(CA=0.5))
        return result

    monkeypatch.setattr(converter, "embed_color_profiles", rewrite_page)
    result = converter.convert_to_pdfa(source, output, level="3b")
    if change is None:
        assert result.target_produced and not result.skipped, result
    else:
        assert result.skipped and not result.target_produced, result
        assert output.read_bytes() == source.read_bytes()
        assert any("invalidated a preserved TrapNet" in w for w in result.warnings)


def test_conversion_keeps_trap_when_sibling_loses_non_normal_appearances(tmp_path):
    """Required /R and /D removal must not fall back to copying the original input."""
    from pikepdf import Pdf

    import pdftopdfa.converter as converter

    pdf = new_pdf()
    page, trap = _trap(pdf)
    trap.LastModified = "D:20990101000000Z"
    page.Annots = Array([_sibling_with_all_appearances(pdf), trap])
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)

    result = converter.convert_to_pdfa(source, output, level="3b")

    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        sibling, retained = converted.pages[0].Annots
        assert sibling.Subtype == Name.Square
        assert list(sibling.AP.keys()) == ["/N"]
        assert retained.Subtype == Name.TrapNet


@pytest.mark.parametrize("changed", [False, True])
def test_conversion_tracks_date_traps_across_ocr(changed, tmp_path, monkeypatch):
    """OCR changes must be compared with the original, before annotations restore."""
    from pikepdf import Pdf

    import pdftopdfa.converter as converter
    import pdftopdfa.ocr as ocr

    pdf = new_pdf()
    _trap(pdf)
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)

    def apply_ocr(input_path, output_path, *args, **kwargs):
        """Simulate OCR with or without a page-content rewrite."""
        with Pdf.open(input_path) as processed:
            if changed:
                processed.pages[0].Contents.write(b"1 g 0 0 100 100 re f")
            processed.save(output_path)
        kwargs["_processed_pages"].add(1)
        return output_path

    monkeypatch.setattr(ocr, "is_ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "apply_ocr", apply_ocr)
    result = converter.convert_to_pdfa(
        source,
        output,
        level="3b",
        ocr_detection_model_dir=tmp_path / "detection",
        ocr_recognition_model_dir=tmp_path / "recognition",
    )
    if changed:
        assert result.skipped and not result.target_produced, result
        assert output.read_bytes() == source.read_bytes()
        assert any("invalidated a preserved TrapNet" in w for w in result.warnings)
    else:
        assert result.target_produced and not result.skipped, result
