# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair malformed optional portfolios without corrupting article navigation."""

from decimal import Decimal

import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers import sanitize_for_pdfa
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


def _portfolio(pdf, schema):
    """Create a portfolio with an item in a nested embedded-file name tree."""
    pdf.add_blank_page()
    spec = pdf.make_indirect(
        Dictionary(
            Type=Name.Filespec,
            F="a.txt",
            UF="a.txt",
            CI=Dictionary(Type=Name.CollectionItem, Good="keep", Foo="dangling"),
        )
    )
    leaf = pdf.make_indirect(Dictionary(Names=Array(["a.txt", spec])))
    pdf.Root.Names = Dictionary(EmbeddedFiles=Dictionary(Kids=Array([leaf])))
    pdf.Root.Collection = Dictionary(Schema=schema)
    return spec


def test_collection_items_lose_only_discarded_schema_fields():
    """A removed field cannot leave item values that no schema entry describes."""
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Good=Dictionary(Subtype=Name.S, N="Good"), Foo=Dictionary(Subtype=Name.S)
        ),
    )
    remove_forbidden_catalog_entries(pdf)
    assert "/Foo" not in pdf.Root.Collection.Schema
    assert sorted(spec.CI.keys()) == ["/Good", "/Type"]
    assert str(spec.CI.Good) == "keep"


def test_collection_items_lose_all_fields_with_a_removed_schema():
    """Without a schema, no item value identifies a field."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Name.Bad)
    remove_forbidden_catalog_entries(pdf)
    assert "/Schema" not in pdf.Root.Collection
    assert list(spec.CI.keys()) == ["/Type"]


def test_collection_items_lose_values_a_valid_schema_does_not_declare():
    """Item keys must identify schema fields, even if no repair was needed."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert sorted(spec.CI.keys()) == ["/Good", "/Type"]
    assert str(spec.CI.Good) == "keep"


def test_collection_items_matching_the_schema_are_unchanged():
    """Declared values stay intact."""
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Good=Dictionary(Subtype=Name.S, N="Good"),
            Foo=Dictionary(Subtype=Name.S, N="Foo"),
        ),
    )
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert sorted(spec.CI.keys()) == ["/Foo", "/Good", "/Type"]


@pytest.mark.parametrize(
    "subtype,value,valid",
    [
        (Name.S, "text", True),
        (Name.S, 42, False),
        (Name.D, "D:20240101000000Z", True),
        (Name.D, "not a date", False),
        (Name.D, 42, False),
        (Name.N, 42, True),
        (Name.N, Decimal("1.5"), True),
        (Name.N, True, False),
        (Name.N, "42", False),
        (Name.S, Dictionary(Type=Name.CollectionSubitem, D="text", P="Prefix "), True),
        (Name.S, Dictionary(P="Prefix "), True),
        (Name.S, Dictionary(D=42), False),
        (Name.S, Dictionary(D="text", P=42), False),
        (Name.S, Dictionary(Type=Name.Bad, D="text"), False),
        (Name.F, "a.txt", True),
        (Name.F, True, False),
        (Name.Desc, "description", True),
        (Name.Desc, Array([1]), False),
        (Name.ModDate, "D:20240101000000Z", True),
        (Name.ModDate, "yesterday", False),
        (Name.CreationDate, "D:20240101000000Z", True),
        (Name.CreationDate, Name.Bad, False),
        (Name.Size, 42, True),
        (Name.Size, Array([1, 2]), False),
        (Name.Size, True, False),
    ],
)
def test_collection_item_values_must_match_their_field_type(subtype, value, valid):
    """A declared key is not enough when its value contradicts the schema."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Field=Dictionary(Subtype=subtype, N="Field")))
    spec.CI = Dictionary(Field=value)
    remove_forbidden_catalog_entries(pdf)
    assert ("/Field" in spec.CI) is valid


def test_direct_collection_item_becomes_indirect_and_keeps_its_values():
    """Making the reference indirect repairs the entry without losing any value."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    spec.CI = Dictionary(Good="keep")
    assert spec.CI.objgen == (0, 0)
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert spec.CI.objgen != (0, 0)
    assert str(spec.CI.Good) == "keep"


@pytest.mark.parametrize("schema", [None, Dictionary()])
def test_collection_item_markers_are_repaired_without_losing_values(schema):
    """An invalid optional type marker or container is repaired on its own."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary() if schema is None else schema)
    if schema is None:
        del pdf.Root.Collection.Schema
    spec.CI = Dictionary(Type=Name.Bad)
    remove_forbidden_catalog_entries(pdf)
    assert "/Type" not in spec.CI

    spec.CI = 42
    remove_forbidden_catalog_entries(pdf)
    assert "/CI" not in spec


@pytest.mark.parametrize(
    "default,kept",
    [("a.txt", True), ("missing.txt", False), (42, False), (Name.Bad, False)],
)
def test_collection_default_must_name_an_embedded_file(default, kept):
    """The initial document has to exist in the embedded-file name tree."""
    pdf = new_pdf()
    _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    pdf.Root.Collection.D = default
    remove_forbidden_catalog_entries(pdf)
    assert ("/D" in pdf.Root.Collection) is kept


def test_collection_default_without_embedded_files_is_removed():
    """Without a name tree no initial document can exist."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(D="a.txt", View=Name.D)
    remove_forbidden_catalog_entries(pdf)
    assert "/D" not in pdf.Root.Collection
    assert pdf.Root.Collection.View == Name.D


def test_collection_default_is_rechecked_after_embedded_file_removal():
    """PDF/A-2 may discard the file that the portfolio opened initially."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    spec.EF = Dictionary(F=pdf.make_stream(b"plain text", Type=Name.EmbeddedFile))
    pdf.Root.Collection.D = "a.txt"
    result = sanitize_for_pdfa(pdf, level="2b")
    assert result["files_removed"] == 1
    assert "/D" not in pdf.Root.Collection


def test_collection_items_lose_all_fields_without_any_schema():
    """Item keys identify schema fields, so no schema leaves no valid value."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary())
    del pdf.Root.Collection.Schema
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert list(spec.CI.keys()) == ["/Type"]


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


def _two_bead_article(pdf):
    """Create an article with two beads on one page, listed in reading order."""
    thread, first, page = _article(pdf)
    second = pdf.make_indirect(
        Dictionary(T=thread, P=page.obj, R=Array([0, 0, 100, 100]))
    )
    first.N = first.V = second
    second.N = second.V = first
    page.B = Array([first, second])
    return first, second, page


def test_page_beads_in_reading_order_are_unchanged():
    """A correctly ordered array needs no repair."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert [bead.objgen for bead in page.B] == [first.objgen, second.objgen]


def test_page_beads_out_of_reading_order_are_repaired():
    """The thread defines the order, so this does not need the original fallback."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    page.B = Array([second, first])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [first.objgen, second.objgen]


def test_duplicate_page_beads_are_removed():
    """A page lists each of its beads once, so the repair keeps all destinations."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    page.B = Array([first, second, first])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [first.objgen, second.objgen]


def test_page_bead_repair_keeps_positions_of_other_threads():
    """No order between separate articles is defined, so they stay in place."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    other_thread = pdf.make_indirect(Dictionary())
    other = pdf.make_indirect(
        Dictionary(T=other_thread, P=page.obj, R=Array([0, 0, 50, 50]))
    )
    other.N = other.V = other
    other_thread.F = other
    pdf.Root.Threads.append(other_thread)
    page.B = Array([second, other, first])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [
        first.objgen,
        other.objgen,
        second.objgen,
    ]


def test_conversion_repairs_page_bead_order(tmp_path):
    """An unordered /B array cannot send a document to the original-input fallback."""
    from pikepdf import Pdf

    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    page.B = Array([second, first])
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        listed = converted.pages[0].B
        assert listed[0].N.objgen == listed[1].objgen
        assert converted.Root.Threads[0].F.objgen == listed[0].objgen


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


@pytest.mark.parametrize("key", ["/CreationDate", "/ModDate"])
@pytest.mark.parametrize("value", ["not a date", 42, Name.Bad])
def test_article_rejection_precedes_collection_repairs(key, value):
    """A rejected article leaves even repairable portfolio settings untouched."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I[key] = value
    pdf.Root.Collection = Dictionary(View=Name.Invalid, Private="keep")
    before = pdf.Root.Collection.unparse()
    with pytest.raises(ConversionError, match="thread information"):
        remove_forbidden_catalog_entries(pdf)
    assert pdf.Root.Collection.unparse() == before


def test_collection_repairs_are_idempotent():
    """A second pass preserves the repaired schema, sorting and item references."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    pdf.Root.Collection.Sort = Dictionary(S=Name.Good, A="invalid")
    assert remove_forbidden_catalog_entries(pdf) == 2
    before = pdf.Root.Collection.unparse(), spec.CI.unparse(), spec.CI.objgen
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert (pdf.Root.Collection.unparse(), spec.CI.unparse(), spec.CI.objgen) == before


@pytest.mark.parametrize("key", ["/N", "/V"])
def test_article_rejects_nonreciprocal_links(key):
    """Every edge must be reciprocal even when both beads belong to the article."""
    pdf = new_pdf()
    first, second, _ = _two_bead_article(pdf)
    second[key] = second
    with pytest.raises(ConversionError, match="inconsistent next/previous links"):
        remove_forbidden_catalog_entries(pdf)
    assert first.N.objgen == second.objgen


def test_article_rejects_a_bead_listed_on_another_page():
    """An otherwise valid cycle cannot leave an unrelated page destination."""
    pdf = new_pdf()
    _, bead, _ = _article(pdf)
    other = pdf.add_blank_page()
    other.B = Array([bead])
    with pytest.raises(ConversionError, match="unrelated or malformed bead"):
        remove_forbidden_catalog_entries(pdf)
