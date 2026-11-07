# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair malformed optional portfolios without corrupting article navigation."""

import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers.catalog import remove_forbidden_catalog_entries


def _article(pdf):
    """Create one complete circular article with reciprocal page references."""
    page = pdf.add_blank_page()
    thread = pdf.make_indirect(Dictionary(I=Dictionary(Title="Article")))
    bead = pdf.make_indirect(
        Dictionary(T=thread, P=page.obj, R=Array([0, 0, 100, 100]))
    )
    bead.N = bead.V = bead
    thread.F = bead
    page.B = Array([bead])
    pdf.Root.Threads = Array([thread])
    return thread, bead, page


@pytest.mark.parametrize("value", [42, Name.Bad, Array([1, 2])])
def test_non_dictionary_collection_is_removed(value):
    """Drop malformed portfolio containers while retaining the rest of the catalog."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = value
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/Collection" not in pdf.Root


def test_collection_repairs_keep_valid_fields():
    """Remove only malformed field settings, schema entries and dependent sorting."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(
        Schema=Dictionary(
            Good=Dictionary(Subtype=Name.S, N="Title", O=True, V=42),
            Bad=Dictionary(Subtype=Name.S),
        ),
        Sort=Dictionary(S=Name.Bad),
        View=Name.Invalid,
        AcmePrivate="keep",
    )
    remove_forbidden_catalog_entries(pdf)
    collection = pdf.Root.Collection
    assert str(collection.Schema.Good.N) == "Title"
    assert "/O" not in collection.Schema.Good and "/V" not in collection.Schema.Good
    assert "/Bad" not in collection.Schema
    assert "/Sort" not in collection and "/View" not in collection
    assert str(collection.AcmePrivate) == "keep"


@pytest.mark.parametrize(
    "schema,fields",
    [
        (None, Name.Foo),
        (None, Array([Name.Foo])),
        (Name.Bad, Name.Foo),
        (Dictionary(), Name.Foo),
        (Dictionary(A=Dictionary(Subtype=Name.S, N="A")), Array()),
        (Dictionary(A=Dictionary(Subtype=Name.S, N="A")), Array([Name.A, Name.Foo])),
        (Dictionary(A=Dictionary(Subtype=Name.S, N="A")), Name.Type),
    ],
)
def test_collection_sort_fields_must_resolve_in_schema(schema, fields):
    """Table 158 sort fields identify schema entries, so none may be left dangling."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(Sort=Dictionary(S=fields), View=Name.D)
    if schema is not None:
        pdf.Root.Collection.Schema = schema
    remove_forbidden_catalog_entries(pdf)
    assert "/Sort" not in pdf.Root.Collection
    assert pdf.Root.Collection.View == Name.D


@pytest.mark.parametrize("ascending", [[], [False], [True, False, True]])
def test_collection_sort_array_lengths_need_not_match(ascending):
    """Table 158 permits extra directions and defaults missing ones to true."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(
        Schema=Dictionary(
            A=Dictionary(Subtype=Name.S, N="A"), B=Dictionary(Subtype=Name.N, N="B")
        ),
        Sort=Dictionary(S=Array([Name.A, Name.B]), A=Array(ascending)),
    )
    expected = pdf.Root.Collection.unparse()
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert pdf.Root.Collection.unparse() == expected


@pytest.mark.parametrize(
    "damage",
    [
        "container",
        "first",
        "thread",
        "next",
        "previous",
        "page",
        "rectangle",
        "page_beads",
        "info",
    ],
)
def test_invalid_article_graph_is_rejected_without_dropping_destinations(damage):
    """Ambiguous article repairs must not renumber or delete navigation targets."""
    pdf = new_pdf()
    thread, bead, page = _article(pdf)
    if damage == "container":
        pdf.Root.Threads = 42
    elif damage == "first":
        del thread.F
    elif damage == "thread":
        del bead.T
    elif damage == "next":
        bead.N = pdf.make_indirect(Dictionary())
    elif damage == "previous":
        bead.V = Name.Bad
    elif damage == "page":
        bead.P = pdf.make_indirect(Dictionary(Type=Name.Page))
    elif damage == "rectangle":
        bead.R = Array([1, 2, True, 4])
    elif damage == "page_beads":
        page.B = Array()
    else:
        thread.I = 42
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread)
    with pytest.raises(ConversionError, match="Malformed /Threads"):
        remove_forbidden_catalog_entries(pdf)
    assert pdf.Root.OpenAction.D.objgen == thread.objgen


def test_valid_article_needs_no_optional_type_entries():
    """Optional Thread/Bead type markers may be omitted on valid graphs."""
    pdf = new_pdf()
    thread, bead, page = _article(pdf)
    thread.I.Trapped = Name("/True")
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert page.B[0].objgen == bead.objgen


def test_article_pages_may_omit_optional_bead_arrays():
    """Table 30 makes page /B optional because /Threads can recreate it."""
    pdf = new_pdf()
    thread, _, page = _article(pdf)
    del page.B
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert "/B" not in page.obj
    assert pdf.Root.Threads[0].objgen == thread.objgen


def test_each_page_decides_whether_its_bead_array_is_checked():
    """A page's listing is only compared with the beads placed on that page."""
    pdf = new_pdf()
    thread, first, _ = _article(pdf)
    other = pdf.add_blank_page()
    second = pdf.make_indirect(
        Dictionary(T=thread, P=other.obj, R=Array([0, 0, 100, 100]))
    )
    first.N = first.V = second
    second.N = second.V = first
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert "/B" not in other.obj

    other.B = Array()
    with pytest.raises(ConversionError, match="missing from its page's /B"):
        remove_forbidden_catalog_entries(pdf)


def test_conversion_keeps_article_without_page_bead_arrays(tmp_path):
    """A conforming article cannot force the original-input fallback."""
    from pikepdf import Pdf

    pdf = new_pdf()
    _, _, page = _article(pdf)
    del page.B
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        assert str(converted.Root.Threads[0].I.Title) == "Article"
        assert "/B" not in converted.pages[0].obj


def test_malformed_article_conversion_preserves_original_file(tmp_path):
    """A malformed graph cannot be published as a claimed PDF/A conversion."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Threads = Array([pdf.make_indirect(Dictionary())])
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.skipped and not result.target_produced
    assert output.read_bytes() == source.read_bytes()
    assert any("Malformed /Threads" in warning for warning in result.warnings)
