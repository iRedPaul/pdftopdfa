# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair malformed optional portfolios without corrupting article navigation."""

from decimal import Decimal

import pytest
from conftest import new_pdf, pdf_object
from pikepdf import Array, Dictionary, Name, String

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
    pdf.Root.Threads = pdf.make_indirect(Array([thread]))
    return thread, bead, page


@pytest.mark.parametrize("value", [42, Name.Bad, Array([1, 2])])
def test_non_dictionary_collection_is_removed(value):
    """Drop malformed portfolio containers while retaining the rest of the catalog."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = pdf_object(value)
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
    pdf.Root.Collection = Dictionary(Sort=Dictionary(S=pdf_object(fields)), View=Name.D)
    if schema is not None:
        pdf.Root.Collection.Schema = pdf_object(schema)
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


def test_collection_items_are_kept_with_a_removed_schema():
    """Without a schema nothing contradicts a value, and PDF/A does not forbid it."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Name.Bad)
    remove_forbidden_catalog_entries(pdf)
    assert "/Schema" not in pdf.Root.Collection
    assert sorted(spec.CI.keys()) == ["/Foo", "/Good", "/Type"]


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
        # ISO 32000-1, 7.9.2.2: text strings use PDFDocEncoding or UTF-16BE.
        (Name.S, String(b"\xfe\xff\x00A\x00b"), True),
        (Name.S, String(b"caf\xe9"), True),
        (Name.S, String(b"\x01undefined"), False),
        (Name.S, String(b"\xfe\xff\xd8\x00"), False),
        (Name.S, String(b"\xfe\xff\x00A\x00"), False),
        (Name.S, Dictionary(D="text", P=String(b"\x02")), False),
        (Name.S, Dictionary(D=String(b"\xfe\xff\xdc\x00"), P="Prefix "), False),
        (Name.Desc, String(b"\x03"), False),
        (Name.F, String(b"\xfe\xff\x00"), False),
        (Name.D, "D:20240101000000Z", True),
        (Name.D, "D:20240101000000Z00'00'", True),
        (Name.D, "D:20240101000000Z00'00", True),
        (Name.D, "D:20240101000000+05'30'", True),
        (Name.D, "D:20240101000000-08'", True),
        (Name.D, "D:20240101000000-08", True),
        (Name.D, "D:20240101000000+", True),
        (Name.D, "D:2024", True),
        (Name.D, "D:20240229", True),
        (Name.D, "D:2024123123", True),
        (Name.D, "D:20241231235959", True),
        (Name.D, "D:20241231235959+05'30", True),
        # ISO 32000-1, 7.9.4: a field requires every preceding one, so the
        # relationship to UT requires the complete date and time.
        (Name.D, "D:2024Z", False),
        (Name.D, "D:202401+05", False),
        (Name.D, "D:20240101Z", False),
        (Name.D, "D:2024010112-08'00'", False),
        (Name.D, "D:202401011200Z00'00'", False),
        (Name.D, "D:20240101000000+0530", False),
        (Name.D, "D:202", False),
        (Name.D, "D:2024010", False),
        (Name.D, "D:20240001", False),
        (Name.D, "D:20240100", False),
        (Name.D, "D:20230229", False),
        (Name.D, "D:20240431", False),
        (Name.D, "D:2024010124", False),
        (Name.D, "D:202401012360", False),
        (Name.D, "D:20240101235960", False),
        (Name.D, "D:\uff12\uff10\uff12\uff14", False),
        (Name.D, "D:20240101000000+24'00'", False),
        (Name.D, "D:20240101000000+05'60'", False),
        (Name.D, "D:20240101000000+05'30'00", False),
        (Name.D, "D:20240101000000Z0", False),
        # Z denotes equality with UT, so an offset after it can only be zero.
        (Name.D, "D:20240101000000Z00'", True),
        (Name.D, "D:20240101000000Z00", True),
        (Name.D, "D:20240101000000Z05'00'", False),
        (Name.D, "D:20240101000000Z00'30'", False),
        (Name.D, "D:20240101000000Z01", False),
        (Name.D, "D:20240101000000Z23'59'", False),
        (Name.D, "D:20240101000000-23'59'", True),
        (Name.D, "D:20240101000000-00'00'", True),
        (Name.D, "20240101000000Z", False),
        (Name.D, "D:20241301", False),
        (Name.D, "not a date", False),
        (Name.D, 42, False),
        (Name.N, 42, True),
        (Name.N, Decimal("1.5"), True),
        (Name.N, True, False),
        (Name.N, "42", False),
        (Name.S, Dictionary(Type=Name.CollectionSubitem, D="text", P="Prefix "), True),
        # ISO 32000-1, Table 49: every subitem entry, /D included, is optional.
        (Name.S, Dictionary(P="Prefix "), True),
        (Name.S, Dictionary(), True),
        (Name.S, Dictionary(Type=Name.CollectionSubitem), True),
        (Name.D, Dictionary(D="D:20240101000000Z00'00'", P="Sent "), True),
        (Name.D, Dictionary(D="not a date", P="Sent "), False),
        (Name.N, Dictionary(D=Decimal("1.5"), P="#"), True),
        (Name.N, Dictionary(D="1.5"), False),
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
    spec.CI = Dictionary(Field=pdf_object(value))
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
    spec = _portfolio(pdf, Dictionary() if schema is None else pdf_object(schema))
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


@pytest.mark.parametrize(
    "value",
    [
        Dictionary(),
        Dictionary(Type=Name.Bad, F="a.txt"),
        Dictionary(Desc="no file name or stream"),
    ],
)
def test_collection_default_must_name_a_file_specification(value):
    """A name-tree value that is no file specification cannot be the default."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Names = Dictionary(
        EmbeddedFiles=Dictionary(Names=Array(["a.txt", pdf.make_indirect(value)]))
    )
    pdf.Root.Collection = Dictionary(D="a.txt", View=Name.D)
    remove_forbidden_catalog_entries(pdf)
    assert "/D" not in pdf.Root.Collection
    assert pdf.Root.Collection.View == Name.D


@pytest.mark.parametrize("level", ["2b", "3b"])
def test_collection_default_is_rechecked_after_name_tree_cleanup(level):
    """Both PDF/A parts validate /D against the final embedded-file name tree."""
    pdf = new_pdf()
    pdf.add_blank_page()
    spec = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="a.txt", UF="a.txt"))
    pdf.Root.Names = Dictionary(
        EmbeddedFiles=Dictionary(
            Names=Array(
                ["a.txt", spec, "b.txt", pdf.make_indirect(Dictionary(Desc="none"))]
            )
        )
    )
    pdf.Root.Collection = Dictionary(D="b.txt", View=Name.D)
    sanitize_for_pdfa(pdf, level=level)
    assert "/D" not in pdf.Root.Collection
    assert pdf.Root.Collection.View == Name.D


def test_collection_default_survives_name_tree_cleanup():
    """A default naming a kept file specification stays after the final check.

    PDF/A-3 keeps any embedded file, unlike PDF/A-2, which removes this one.
    """
    pdf = new_pdf()
    pdf.add_blank_page()
    spec = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="a.txt", UF="a.txt"))
    pdf.Root.Names = Dictionary(
        EmbeddedFiles=Dictionary(Names=Array(["a.txt", spec, "b.txt", 42]))
    )
    pdf.Root.Collection = Dictionary(D="a.txt")
    sanitize_for_pdfa(pdf, level="3b")
    assert str(pdf.Root.Collection.D) == "a.txt"


def test_collection_default_is_rechecked_after_embedded_file_removal():
    """PDF/A-2 may discard the file that the portfolio opened initially."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    spec.EF = Dictionary(F=pdf.make_stream(b"plain text", Type=Name.EmbeddedFile))
    pdf.Root.Collection.D = "a.txt"
    result = sanitize_for_pdfa(pdf, level="2b")
    assert result["files_removed"] == 1
    assert "/D" not in pdf.Root.Collection


def test_collection_items_are_kept_without_any_schema():
    """Portfolio metadata survives, because veraPDF accepts it without a schema."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary())
    del pdf.Root.Collection.Schema
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert sorted(spec.CI.keys()) == ["/Foo", "/Good", "/Type"]
    assert str(spec.CI.Good) == "keep"


def test_collection_schema_emptied_by_repairs_describes_no_item_values():
    """An emptied schema is well formed, but no item value or sort field remains."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.Bad, N="Good")))
    pdf.Root.Collection.Sort = Dictionary(S=Name.Good)
    pdf.Root.Collection.View = Name.T
    remove_forbidden_catalog_entries(pdf)
    collection = pdf.Root.Collection
    assert list(collection.Schema.keys()) == []
    assert sorted(spec.CI.keys()) == ["/Type"]
    assert "/Sort" not in collection
    assert collection.View == Name.T
    assert remove_forbidden_catalog_entries(pdf) == 0


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


@pytest.mark.parametrize("ascending", [Array([False]), Array(), Array([True, 1])])
def test_single_sort_field_loses_an_array_of_directions(ascending):
    """Table 158 relates /A array elements to an /S array, so a name takes a bool."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(
        Schema=Dictionary(A=Dictionary(Subtype=Name.S, N="A")),
        Sort=Dictionary(S=Name.A, A=ascending),
    )
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert pdf.Root.Collection.Sort.S == Name.A
    assert "/A" not in pdf.Root.Collection.Sort


@pytest.mark.parametrize("fields", [Name.A, Array([Name.A])])
def test_sort_direction_boolean_is_kept(fields):
    """A single boolean direction is valid for a name or an array of names."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.Collection = Dictionary(
        Schema=Dictionary(A=Dictionary(Subtype=Name.S, N="A")),
        Sort=Dictionary(S=fields, A=False),
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
    else:
        bead.R = Array([1, 2, True, 4])
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


def test_inline_threads_array_becomes_indirect():
    """ISO 32000-1, Table 28 requires an indirect /Threads array."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    pdf.Root.Threads = Array([thread])
    assert pdf.Root.Threads.objgen == (0, 0)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert pdf.Root.Threads.objgen != (0, 0)
    assert [item.objgen for item in pdf.Root.Threads] == [thread.objgen]
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_inline_threads_array_is_not_changed_for_a_rejected_article():
    """Rejection leaves the article exactly as the source stored it."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    del thread.F
    pdf.Root.Threads = Array([thread])
    with pytest.raises(ConversionError, match="Malformed /Threads"):
        remove_forbidden_catalog_entries(pdf)
    assert pdf.Root.Threads.objgen == (0, 0)


def test_conversion_writes_an_indirect_threads_array(tmp_path):
    """The saved catalog refers to the /Threads array indirectly."""
    from pikepdf import Pdf

    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    pdf.Root.Threads = Array([thread])
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        assert converted.Root.Threads.objgen != (0, 0)
        assert str(converted.Root.Threads[0].I.Title) == "Article"


def test_omitted_page_bead_array_is_rebuilt():
    """ISO 32000-1, 12.4.3 requires /B on each page on which beads appear."""
    pdf = new_pdf()
    thread, bead, page = _article(pdf)
    del page.B
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [item.objgen for item in page.B] == [bead.objgen]
    assert pdf.Root.Threads[0].objgen == thread.objgen
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_rebuilt_page_bead_arrays_follow_the_thread():
    """Each page lists only its own beads, in the order of their thread."""
    pdf = new_pdf()
    thread, first, page = _article(pdf)
    other = pdf.add_blank_page()
    beads = [first] + [
        pdf.make_indirect(Dictionary(T=thread, P=target, R=Array([0, 0, 10, 10])))
        for target in (other.obj, page.obj, other.obj)
    ]
    for index, bead in enumerate(beads):
        bead.N = beads[(index + 1) % len(beads)]
        bead.V = beads[index - 1]
    del page.B
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert [item.objgen for item in page.B] == [beads[0].objgen, beads[2].objgen]
    assert [item.objgen for item in other.B] == [beads[1].objgen, beads[3].objgen]


@pytest.mark.parametrize(
    "listed",
    ["empty", "malformed", "direct", "unrelated", "another_page"],
)
def test_incomplete_or_wrong_page_bead_arrays_are_rebuilt(listed):
    """The validated bead cycles define the beads of every page unambiguously."""
    pdf = new_pdf()
    thread, first, page = _article(pdf)
    other = pdf.add_blank_page()
    second = pdf.make_indirect(
        Dictionary(T=thread, P=other.obj, R=Array([0, 0, 100, 100]))
    )
    first.N = first.V = second
    second.N = second.V = first
    other.B = Array([second])
    if listed == "empty":
        page.B = Array()
    elif listed == "malformed":
        page.B = 42
    elif listed == "direct":
        page.B = Array([Dictionary(R=Array([0, 0, 1, 1])), first])
    elif listed == "unrelated":
        page.B = Array([pdf.make_indirect(Dictionary(Type=Name.Bead)), first])
    else:
        page.B = Array([second, first])
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread, B=first)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [item.objgen for item in page.B] == [first.objgen]
    assert [item.objgen for item in other.B] == [second.objgen]
    assert pdf.Root.OpenAction.B.objgen == first.objgen


def test_page_without_beads_loses_a_bead_array_listing_other_pages_beads():
    """A page on which no bead appears cannot list a bead of another page."""
    pdf = new_pdf()
    _, bead, page = _article(pdf)
    other = pdf.add_blank_page()
    other.B = Array([bead])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/B" not in other.obj
    assert [item.objgen for item in page.B] == [bead.objgen]


def test_page_without_beads_keeps_an_empty_bead_array():
    """An empty array lists exactly the beads of a page without any."""
    pdf = new_pdf()
    _article(pdf)
    other = pdf.add_blank_page()
    other.B = Array()
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert len(other.B) == 0


def test_conversion_rebuilds_omitted_page_bead_arrays(tmp_path):
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
        listed = converted.pages[0].B
        assert [item.objgen for item in listed] == [converted.Root.Threads[0].F.objgen]


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


def test_page_beads_in_drawing_order_are_unchanged():
    """ISO 32000-1, 12.4.3 lists /B in drawing order, which may differ from /N."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    page.B = pdf.make_indirect(Array([second, first]))
    listed = page.B.objgen
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert page.B.objgen == listed
    assert [bead.objgen for bead in page.B] == [second.objgen, first.objgen]


def test_duplicate_page_beads_are_removed():
    """A page lists each of its beads once, at the place of its first entry."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    page.B = Array([second, first, second])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [second.objgen, first.objgen]


def _visiting_article(pdf):
    """Create an article that visits its first page, another page, then it again.

    The rectangles tell the beads apart once the file has been saved.
    """
    thread, first, page = _article(pdf)
    other = pdf.add_blank_page()
    beads = [first] + [
        pdf.make_indirect(
            Dictionary(T=thread, P=target, R=Array([0, 0, 10 * index, 10 * index]))
        )
        for index, target in enumerate((other.obj, page.obj, page.obj), 2)
    ]
    for index, bead in enumerate(beads):
        bead.N = beads[(index + 1) % len(beads)]
        bead.V = beads[index - 1]
    other.B = Array([beads[1]])
    return beads, page


def test_page_bead_repair_keeps_listed_order_and_appends_omitted_beads():
    """Listed beads keep their order; omitted ones follow in reading order."""
    pdf = new_pdf()
    beads, page = _visiting_article(pdf)
    page.B = Array([beads[3], 42, Dictionary(), beads[1], beads[3]])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [
        beads[3].objgen,
        beads[0].objgen,
        beads[2].objgen,
    ]
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_page_bead_repair_keeps_positions_of_other_threads():
    """Repairs leave the listed beads of separate articles in place."""
    pdf = new_pdf()
    first, second, page = _two_bead_article(pdf)
    other_thread = pdf.make_indirect(Dictionary())
    other = pdf.make_indirect(
        Dictionary(T=other_thread, P=page.obj, R=Array([0, 0, 50, 50]))
    )
    other.N = other.V = other
    other_thread.F = other
    pdf.Root.Threads.append(other_thread)
    page.B = Array([second, other])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [bead.objgen for bead in page.B] == [
        second.objgen,
        other.objgen,
        first.objgen,
    ]


def test_conversion_keeps_page_beads_in_drawing_order(tmp_path):
    """A page that an article visits twice keeps /B in drawing order, not /N order."""
    from pikepdf import Pdf

    pdf = new_pdf()
    beads, page = _visiting_article(pdf)
    # Drawing order: the last bead of the article, then its first and third.
    page.B = Array([beads[3], beads[0], beads[2]])
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        listed = converted.pages[0].B
        assert [int(bead.R[2]) for bead in listed] == [40, 100, 30]
        first = converted.Root.Threads[0].F
        assert first.objgen == listed[1].objgen
        assert first.N.N.objgen == listed[2].objgen
        assert first.N.N.N.objgen == listed[0].objgen
        assert [int(bead.R[2]) for bead in converted.pages[1].B] == [20]


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


def test_article_rejection_precedes_all_repairs():
    """A rejected article leaves even repairable information and portfolios."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.ModDate = "not a date"
    del thread.F
    pdf.Root.Collection = Dictionary(View=Name.Invalid, Private="keep")
    before = pdf.Root.Collection.unparse(), thread.I.unparse()
    with pytest.raises(ConversionError, match="Malformed /Threads"):
        remove_forbidden_catalog_entries(pdf)
    assert (pdf.Root.Collection.unparse(), thread.I.unparse()) == before


@pytest.mark.parametrize(
    "value",
    [
        "D:20240101000000Z00'00'",
        "D:20240101000000Z00'",
        "D:20240101000000Z00",
        "D:20240101000000+05'",
        "D:2024",
    ],
)
def test_thread_information_accepts_every_iso_date_form(value):
    """ISO 32000-1, 7.9.4 permits a zero offset after Z and omitted fields."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.ModDate = value
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert str(thread.I.ModDate) == value


@pytest.mark.parametrize(
    "value",
    [
        "D:2024Z",
        "D:202401+05",
        "D:20240230000000Z",
        "D:20240101000000Z05'00'",
        "D:20240101000000+24'00'",
        "not a date",
        42,
        Name.Bad,
    ],
)
@pytest.mark.parametrize("key", ["/CreationDate", "/ModDate"])
def test_thread_information_loses_invalid_dates(key, value):
    """Like DocInfo, a malformed date is removed without its article or siblings."""
    pdf = new_pdf()
    thread, bead, _ = _article(pdf)
    thread.I[key] = value
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert key not in thread.I
    assert str(thread.I.Title) == "Article"
    assert pdf.Root.Threads[0].F.objgen == bead.objgen
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("/Title", String(b"\x01undefined")),
        ("/Author", String(b"\xfe\xff\xd8\x00")),
        ("/Subject", String(b"\xfe\xff\x00A\x00")),
        ("/Keywords", 42),
        ("/Creator", Name.Bad),
        ("/Producer", Array(["a"])),
        ("/Custom", String(b"\x02")),
        ("/Custom", Dictionary()),
        ("/Trapped", String("True")),
        ("/Trapped", Name.Maybe),
        ("/Trapped", True),
    ],
)
def test_thread_information_loses_invalid_entries(key, value):
    """Each /I entry keeps the syntax that retained DocInfo values have."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.Keep = String(b"\xfe\xff\x00o\x00k")
    thread.I[key] = pdf_object(value)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert sorted(thread.I.keys()) == sorted({"/Keep", "/Title"} - {key})
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("/Title", String(b"caf\xe9")),
        ("/Author", String(b"\xfe\xff\x00A")),
        ("/Custom", "text"),
        ("/Trapped", Name.Unknown),
        ("/Trapped", Name("/False")),
    ],
)
def test_thread_information_keeps_valid_entries(key, value):
    """Valid text strings and /Trapped names are preserved byte for byte."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I[key] = pdf_object(value)
    before = thread.I.unparse()
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert thread.I.unparse() == before


@pytest.mark.parametrize("value", [42, Name.Bad, Array([1]), "Article"])
def test_malformed_thread_information_is_removed_without_the_article(value):
    """Thread information is optional; removing it changes no destination."""
    pdf = new_pdf()
    thread, bead, page = _article(pdf)
    thread.I = pdf_object(value)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/I" not in thread
    assert pdf.Root.OpenAction.D.objgen == thread.objgen
    assert [item.objgen for item in page.B] == [bead.objgen]


def test_conversion_writes_only_valid_thread_information(tmp_path):
    """The converted article keeps its valid information next to removed entries."""
    from pikepdf import Pdf

    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.Author = String(b"\x01undefined")
    thread.I.ModDate = "D:20240101000000Z05'00'"
    thread.I.Subject = String(b"\xfe\xff\x00S\x00u\x00b")
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        info = converted.Root.Threads[0].I
        assert sorted(info.keys()) == ["/Subject", "/Title"]
        assert str(info.Subject) == "Sub"


@pytest.mark.parametrize(
    "name",
    [String(b"\x01undefined"), String(b"\xfe\xff\xd8\x00"), String(b"\xfe\xff\x00")],
)
def test_collection_field_names_need_a_valid_text_encoding(name):
    """Table 156 defines the field name /N as a text string."""
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Good=Dictionary(Subtype=Name.S, N="Good"),
            Foo=Dictionary(Subtype=Name.S, N=name),
        ),
    )
    remove_forbidden_catalog_entries(pdf)
    assert sorted(pdf.Root.Collection.Schema.keys()) == ["/Good"]
    assert sorted(spec.CI.keys()) == ["/Good", "/Type"]


def _unlisted_article(pdf):
    """Create a two-page article that the catalog /Threads array does not list."""
    thread, first, page = _article(pdf)
    del pdf.Root.Threads
    other = pdf.add_blank_page()
    second = pdf.make_indirect(Dictionary(P=other.obj, R=Array([0, 0, 50, 50])))
    first.N = first.V = second
    second.N = second.V = first
    other.B = Array([second])
    return thread, first, second, page, other


def test_unlisted_thread_keeps_its_page_beads():
    """A thread that only a Thread action names still owns the beads of its pages."""
    pdf = new_pdf()
    thread, first, second, page, other = _unlisted_article(pdf)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread, B=second)
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert [item.objgen for item in page.B] == [first.objgen]
    assert [item.objgen for item in other.B] == [second.objgen]
    assert "/Threads" not in pdf.Root


def test_thread_found_through_page_beads_rebuilds_other_pages():
    """A bead without /T leads to its thread, whose other pages regain /B."""
    pdf = new_pdf()
    _, first, second, page, other = _unlisted_article(pdf)
    del page.B
    other.B = Array([second, second])
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert [item.objgen for item in page.B] == [first.objgen]
    assert [item.objgen for item in other.B] == [second.objgen]


def _annotation_action(pdf, page, action):
    annotation = pdf.make_indirect(
        Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 1, 1]))
    )
    annotation.A = action
    page.Annots = Array([annotation])


def _outline_action(pdf, page, action):
    item = pdf.make_indirect(Dictionary(Title="Article"))
    item.A = action
    pdf.Root.Outlines = pdf.make_indirect(Dictionary(First=item, Last=item, Count=1))
    item.Parent = pdf.Root.Outlines


@pytest.mark.parametrize(
    "location", ["open", "document", "page", "annotation", "outline", "next"]
)
def test_threads_of_thread_actions_regain_their_page_beads(location):
    """Pages of a thread that an action names get /B although /Threads omits it."""
    pdf = new_pdf()
    thread, first, second, page, other = _unlisted_article(pdf)
    del page.B, other.B
    action = Dictionary(S=Name.Thread, D=thread)
    if location == "open":
        pdf.Root.OpenAction = action
    elif location == "document":
        pdf.Root.AA = Dictionary(WC=action)
    elif location == "page":
        other.AA = Dictionary(O=action)
    elif location == "annotation":
        _annotation_action(pdf, other, action)
    elif location == "outline":
        _outline_action(pdf, page, action)
    else:
        pdf.Root.OpenAction = Dictionary(
            S=Name.URI, URI="https://example.org", Next=Array([action])
        )
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert [item.objgen for item in page.B] == [first.objgen]
    assert [item.objgen for item in other.B] == [second.objgen]


def test_unlisted_threads_follow_the_listed_ones():
    """Beads of listed and unlisted threads share a page in their drawing order."""
    pdf = new_pdf()
    thread, first, second, page, other = _unlisted_article(pdf)
    listed, bead, _ = _article(pdf)
    bead.P = page.obj
    pdf.Root.Threads = pdf.make_indirect(Array([listed]))
    page.B = Array([first, bead])
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert [item.objgen for item in page.B] == [first.objgen, bead.objgen]
    assert "/B" not in pdf.pages[2].obj
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize("damage", ["next", "owner", "page"])
def test_malformed_unlisted_thread_leaves_only_valid_beads(damage):
    """A cycle no catalog entry needs is not rejected; its entries are not beads."""
    pdf = new_pdf()
    thread, first, second, page, other = _unlisted_article(pdf)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread)
    thread.I.Title = String(b"\x01undefined")
    if damage == "next":
        second.N = second
    elif damage == "owner":
        first.T = pdf.make_indirect(Dictionary(I=Dictionary(Title=String(b"\x01"))))
    else:
        second.P = pdf.make_indirect(Dictionary(Type=Name.Page))
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert "/B" not in page.obj and "/B" not in other.obj
    # Information of a thread that is not valid is left as the source stored it.
    assert bytes(thread.I.Title) == b"\x01undefined"
    assert pdf.Root.OpenAction.D.objgen == thread.objgen


def test_conversion_keeps_beads_of_an_action_thread(tmp_path):
    """A thread outside /Threads keeps its beads and /B arrays in the output."""
    from pikepdf import Pdf

    pdf = new_pdf()
    thread, _, second, page, other = _unlisted_article(pdf)
    del page.B
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread, B=second)
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        action = converted.Root.OpenAction
        first = action.D.F
        assert str(action.D.I.Title) == "Article"
        assert [bead.objgen for bead in converted.pages[0].B] == [first.objgen]
        assert [bead.objgen for bead in converted.pages[1].B] == [first.N.objgen]
        assert action.B.objgen == first.N.objgen


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


def _malformed_kid(pdf, kind):
    """Build one /Kids entry that is not a name-tree node dictionary."""
    return {
        "integer": lambda: 42,
        "null": lambda: None,
        "name": lambda: Name.Bad,
        "string": lambda: "not a node",
        "array": lambda: Array([1, 2]),
        "nested array": lambda: Array([Dictionary(Names=Array(["x.txt", 1]))]),
        "indirect integer": lambda: pdf.make_indirect(42),
        "indirect array": lambda: pdf.make_indirect(Array([1, 2])),
        "stream": lambda: pdf.make_stream(b"not a node"),
    }[kind]()


_MALFORMED_KIDS = [
    "integer",
    "null",
    "name",
    "string",
    "array",
    "nested array",
    "indirect integer",
    "indirect array",
    "stream",
]


@pytest.mark.parametrize("kind", _MALFORMED_KIDS)
def test_portfolio_tolerates_malformed_name_tree_children(kind):
    """A /Kids entry that is no node is skipped instead of aborting the repair."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    pdf.Root.Names.EmbeddedFiles.Kids.append(_malformed_kid(pdf, kind))
    pdf.Root.Collection.D = "a.txt"
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert str(pdf.Root.Collection.D) == "a.txt"
    assert str(spec.CI.Good) == "keep" and "/Foo" not in spec.CI


def _spec_placeholder(pdf, value, spec):
    """Rebuild a parametrized tree, putting the file specification at "SPEC"."""
    if isinstance(value, list | Array):
        return Array([_spec_placeholder(pdf, item, spec) for item in value])
    if isinstance(value, dict | Dictionary):
        return Dictionary(
            {key: _spec_placeholder(pdf, item, spec) for key, item in value.items()}
        )
    return spec if value == "SPEC" else value


@pytest.mark.parametrize(
    "embedded,kept",
    [
        # Odd length: the trailing key has no value.
        ({"/Names": ["a.txt", "SPEC", "orphan"]}, True),
        # Keys that are not strings are skipped with their values.
        ({"/Names": [42, "SPEC", Name.Bad, "SPEC", "a.txt", "SPEC"]}, True),
        ({"/Names": [Name("/a.txt"), "SPEC"]}, False),
        # /Names and /Kids values of the wrong type.
        ({"/Names": {"/A": 1}, "/Kids": [{"/Names": 7}]}, False),
        ({"/Names": "a.txt", "/Kids": 42}, False),
        ({"/Kids": [{"/Kids": Name.Bad}, {"/Names": ["a.txt", "SPEC"]}]}, True),
        # A single node given as /Kids is accepted like the normalization does.
        ({"/Kids": {"/Names": ["a.txt", "SPEC"]}}, True),
    ],
)
def test_portfolio_tolerates_malformed_name_tree_leaves(embedded, kept):
    """Malformed /Names and /Kids values are skipped like the later normalization.

    Only a string key paired with a file specification names the initial document.
    """
    pdf = new_pdf()
    pdf.add_blank_page()
    spec = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="a.txt", UF="a.txt"))
    pdf.Root.Names = Dictionary(EmbeddedFiles=_spec_placeholder(pdf, embedded, spec))
    pdf.Root.Collection = Dictionary(D="a.txt", View=Name.D)
    remove_forbidden_catalog_entries(pdf)
    assert ("/D" in pdf.Root.Collection) is kept
    assert pdf.Root.Collection.View == Name.D


def test_portfolio_tolerates_cyclic_name_tree():
    """Cyclic /Kids terminate, and the files reached before the cycle still count."""
    pdf = new_pdf()
    _portfolio(pdf, None)
    root = pdf.make_indirect(pdf.Root.Names.EmbeddedFiles)
    pdf.Root.Names.EmbeddedFiles = root
    root.Kids.append(root)
    root.Kids[0].Kids = Array([root, root.Kids[0]])
    pdf.Root.Collection.D = "a.txt"
    remove_forbidden_catalog_entries(pdf)
    assert str(pdf.Root.Collection.D) == "a.txt"


@pytest.mark.parametrize("kind", ["integer", "null", "indirect array"])
def test_sanitizing_portfolio_discards_malformed_name_tree_children(kind):
    """The full pipeline keeps the portfolio and flattens away the malformed child.

    PDF/A-3 keeps the attachment, which PDF/A-2 would discard with its /D entry.
    """
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    pdf.Root.Names.EmbeddedFiles.Kids.append(_malformed_kid(pdf, kind))
    pdf.Root.Collection.D = "a.txt"
    sanitize_for_pdfa(pdf, level="3b")
    embedded = pdf.Root.Names.EmbeddedFiles
    assert "/Kids" not in embedded
    assert [str(key) for key in embedded.Names[::2]] == ["a.txt"]
    assert str(pdf.Root.Collection.D) == "a.txt"
    assert str(spec.CI.Good) == "keep"


def test_conversion_of_portfolio_with_malformed_name_tree_child(tmp_path):
    """A malformed /EmbeddedFiles child no longer aborts the conversion."""
    from pikepdf import Pdf

    pdf = new_pdf()
    _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    kids = pdf.Root.Names.EmbeddedFiles.Kids
    kids.append(42)
    kids.append(pdf.make_indirect(Array([1, 2])))
    pdf.Root.Collection.D = "a.txt"
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output, level="3b")
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        embedded = converted.Root.Names.EmbeddedFiles
        assert [str(key) for key in embedded.Names[::2]] == ["a.txt"]
        assert str(converted.Root.Collection.D) == "a.txt"
        assert str(embedded.Names[1].CI.Good) == "keep"


def test_conversion_of_portfolio_with_invalid_subitems(tmp_path):
    """Only the subitem that contradicts its field is dropped from the output."""
    from pikepdf import Pdf

    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Good=Dictionary(Subtype=Name.S, N="Good"),
            Sent=Dictionary(Subtype=Name.D, N="Sent"),
            Size=Dictionary(Subtype=Name.N, N="Size"),
            Note=Dictionary(Subtype=Name.S, N="Note"),
        ),
    )
    spec.CI.Sent = Dictionary(
        Type=Name.CollectionSubitem, D="D:20240101000000Z00'00'", P="Sent "
    )
    spec.CI.Size = Dictionary(Type=Name.CollectionSubitem, D="large")
    spec.CI.Note = Dictionary(Type=Name.CollectionSubitem, P="Prefix only")
    pdf.Root.Collection.Sort = Dictionary(S=Array([Name.Sent, Name.Size]))
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output, level="3b")
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        item = converted.Root.Names.EmbeddedFiles.Names[1].CI
        assert sorted(item.keys()) == ["/Good", "/Note", "/Sent", "/Type"]
        assert str(item.Sent.D) == "D:20240101000000Z00'00'"
        assert str(item.Note.P) == "Prefix only" and "/D" not in item.Note
        assert list(converted.Root.Collection.Sort.S) == [Name.Sent, Name.Size]
