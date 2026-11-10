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
from pdftopdfa.metadata import _is_pdf_text_string
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
        # ISO 32000-1, 7.9.4 requires the D: prefix, which PDF Reference 1.7,
        # 3.8.3 made optional; it is added rather than accepted without it.
        (Name.D, "20240101000000Z", True),
        (Name.D, "2024Z", False),
        (Name.D, "20241301", False),
        (Name.D, "d:2024", False),
        (Name.D, Dictionary(D="20240101000000Z00'00'", P="Sent "), True),
        (Name.D, Dictionary(D="20240230", P="Sent "), False),
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
    ],
)
def test_collection_item_values_must_match_their_field_type(subtype, value, valid):
    """A declared key is not enough when its value contradicts the schema."""
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Field=Dictionary(Subtype=subtype, N="Field")))
    spec.CI = Dictionary(Field=pdf_object(value))
    remove_forbidden_catalog_entries(pdf)
    assert ("/Field" in spec.CI) is valid
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "value,expected",
    [
        ("20240101000000Z", "D:20240101000000Z"),
        ("2024", "D:2024"),
        ("20241231235959+05'30'", "D:20241231235959+05'30'"),
        ("D:20240101000000Z", "D:20240101000000Z"),
    ],
)
def test_collection_dates_gain_the_required_prefix(value, expected):
    """A PDF 1.7 date without D: is kept with the prefix ISO 32000-1 requires."""
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Sent=Dictionary(Subtype=Name.D, N="Sent"),
            Read=Dictionary(Subtype=Name.D, N="Read"),
        ),
    )
    spec.CI = Dictionary(Sent=value, Read=Dictionary(D=value, P="Read "))
    assert remove_forbidden_catalog_entries(pdf) == 2 * (value != expected)
    assert str(spec.CI.Sent) == expected
    assert str(spec.CI.Read.D) == expected and str(spec.CI.Read.P) == "Read "
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "subtype", [Name.F, Name.Desc, Name.ModDate, Name.CreationDate, Name.Size]
)
@pytest.mark.parametrize(
    "value",
    [
        "a.txt",
        "D:20240101000000Z",
        42,
        Dictionary(D="a.txt", P="File "),
        String(b"\x03"),
        Name.Bad,
    ],
)
def test_file_related_fields_take_no_collection_item_value(subtype, value):
    """ISO 32000-1, Table 157 sources these fields from the file specification.

    Only S, D and N identify fields of a collection item or subitem dictionary, so
    a /CI value under a file-related field's key is removed. The field itself and
    the file specification data it displays are unchanged.
    """
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Field=Dictionary(Subtype=subtype, N="Field"),
            Good=Dictionary(Subtype=Name.S, N="Good"),
        ),
    )
    spec.Desc = "description"
    spec.CI = Dictionary(Field=pdf_object(value), Good="keep")
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert sorted(spec.CI.keys()) == ["/Good"]
    assert str(spec.CI.Good) == "keep"
    assert pdf.Root.Collection.Schema.Field.Subtype == subtype
    assert str(spec.F) == "a.txt" and str(spec.Desc) == "description"
    assert remove_forbidden_catalog_entries(pdf) == 0


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


@pytest.mark.parametrize("key", ["/CreationDate", "/ModDate"])
def test_thread_information_dates_gain_the_required_prefix(key):
    """ISO 32000-1, 7.9.4 requires D:, which PDF Reference 1.7 left optional."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I[key] = "20240101000000+05'30'"
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert str(thread.I[key]) == "D:20240101000000+05'30'"
    assert str(thread.I.Title) == "Article"
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "value",
    [
        "D:2024Z",
        "D:202401+05",
        "D:20240230000000Z",
        "2024Z",
        "20240230000000Z",
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
    # Both /B arrays and the Thread action, whose destination is not a valid
    # article (ISO 32000-1, Table 205), are removed.
    assert remove_forbidden_catalog_entries(pdf) == 3
    assert "/B" not in page.obj and "/B" not in other.obj
    # Information of a thread that is not valid is left as the source stored it.
    assert bytes(thread.I.Title) == b"\x01undefined"
    assert "/OpenAction" not in pdf.Root
    assert remove_forbidden_catalog_entries(pdf) == 0


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


def _malformed_unlisted_article(pdf):
    """Create an unlisted two-page article whose second bead breaks the cycle."""
    thread, first, second, page, other = _unlisted_article(pdf)
    second.N = second
    return thread, page, other


def _link(pdf, page, **entries):
    annotation = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 1, 1]), **entries
        )
    )
    page.Annots = Array([annotation])
    return annotation


@pytest.mark.parametrize(
    "location", ["open", "annotation", "additional", "outline", "next", "chain"]
)
def test_actions_naming_a_malformed_unlisted_thread_are_removed(location):
    """ISO 32000-1, Table 205: the action has no valid destination, so it goes.

    Only the reference to that action is removed; the conversion, the container
    of the action and the other actions remain.
    """
    pdf = new_pdf()
    thread, page, other = _malformed_unlisted_article(pdf)
    broken = Dictionary(S=Name.Thread, D=thread)
    goto = pdf.make_indirect(Dictionary(S=Name.GoTo, D=Array([page.obj, Name.Fit])))
    annotation = None
    if location == "open":
        pdf.Root.OpenAction = broken
    elif location == "annotation":
        annotation = _link(pdf, other, A=broken)
    elif location == "additional":
        annotation = _link(pdf, other, AA=Dictionary(E=broken, X=goto))
    elif location == "outline":
        _outline_action(pdf, page, broken)
    elif location == "next":
        pdf.Root.OpenAction = Dictionary(
            S=Name.URI, URI="https://example.org", Next=Array([broken, goto])
        )
    else:
        pdf.Root.OpenAction = Dictionary(S=Name.GoTo, D=goto.D, Next=broken)
    # Both page /B arrays, which list no valid bead, and the action reference.
    assert remove_forbidden_catalog_entries(pdf) == 3
    if location == "open":
        assert "/OpenAction" not in pdf.Root
    elif location == "annotation":
        assert "/A" not in annotation
        assert other.Annots[0].objgen == annotation.objgen
    elif location == "additional":
        assert list(annotation.AA.keys()) == ["/X"]
        assert annotation.AA.X.objgen == goto.objgen
    elif location == "outline":
        assert "/A" not in pdf.Root.Outlines.First
        assert str(pdf.Root.Outlines.First.Title) == "Article"
    elif location == "next":
        assert [item.objgen for item in pdf.Root.OpenAction.Next] == [goto.objgen]
    else:
        assert "/Next" not in pdf.Root.OpenAction
        assert pdf.Root.OpenAction.S == Name.GoTo
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_shared_thread_action_is_removed_from_every_reference():
    """A container left without actions is removed: /AA and /Next."""
    pdf = new_pdf()
    thread, page, other = _malformed_unlisted_article(pdf)
    del page.B, other.B
    broken = pdf.make_indirect(Dictionary(S=Name.Thread, D=thread))
    clicked = _link(pdf, page, A=broken)
    entered = _link(pdf, other, AA=Dictionary(E=broken))
    pdf.Root.OpenAction = Dictionary(
        S=Name.Named, N=Name.NextPage, Next=pdf.make_indirect(Array([broken]))
    )
    assert remove_forbidden_catalog_entries(pdf) == 3
    assert "/A" not in clicked and "/AA" not in entered
    assert sorted(pdf.Root.OpenAction.keys()) == ["/N", "/S"]


@pytest.mark.parametrize(
    "target,kept",
    [
        ("thread", True),
        (0, True),
        ("Article", True),
        (1, False),
        (-1, False),
        ("Missing", False),
        ("indirect", False),
        ("direct", False),
        (None, False),
        (Name.Article, False),
        (True, False),
        (Decimal("0.5"), False),
    ],
)
def test_thread_action_destination_forms(target, kept):
    """Table 205: /D is a thread dictionary, a /Threads index or a thread title."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    action = Dictionary(S=Name.Thread)
    if target == "thread":
        action.D = thread
    elif target == "indirect":
        action.D = pdf.make_indirect(Dictionary(I=Dictionary(Title="Article")))
    elif target == "direct":
        action.D = Dictionary(F=thread.F)
    elif target is not None:
        action.D = target
    pdf.Root.OpenAction = action
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/OpenAction" in pdf.Root) == kept


def test_thread_title_names_the_first_listed_thread():
    """Table 205: of two threads with one title, the first listed one is used."""
    pdf = new_pdf()
    first, _, _ = _article(pdf)
    second, bead, _ = _article(pdf)
    pdf.Root.Threads = pdf.make_indirect(Array([first, second]))
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D="Article", B=bead)
    # The bead of the second thread is not a bead of the first one.
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert sorted(pdf.Root.OpenAction.keys()) == ["/D", "/S"]


@pytest.mark.parametrize("target", [7, "Elsewhere"])
def test_thread_action_into_another_file_is_kept(target):
    """An index or title of a thread in another file (/F) is not checked here."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, F="other.pdf", D=target, B=3)
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert pdf.Root.OpenAction.D == target and pdf.Root.OpenAction.B == 3


@pytest.mark.parametrize(
    "file,kept",
    [
        ("other.pdf", True),
        ("filespec", True),
        ("unicode", True),
        ("indirect", True),
        ("", False),
        (42, False),
        (Name.Other, False),
        (Array([String("other.pdf")]), False),
        ("empty", False),
        ("wrong-type", False),
        ("name-entry", False),
    ],
)
@pytest.mark.parametrize("target", [7, "Elsewhere", "thread"])
def test_thread_action_file_must_be_a_file_specification(target, file, kept):
    """Table 205, 7.11: /F is a file specification string or dictionary.

    Its presence alone does not make the destination a thread in another file,
    so an action whose /F is not a file specification is removed.
    """
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    specs = {
        "filespec": Dictionary(Type=Name.Filespec, F="other.pdf"),
        "unicode": Dictionary(UF="other.pdf"),
        "indirect": pdf.make_indirect(Dictionary(Type=Name.Filespec, F="other.pdf")),
        "empty": Dictionary(),
        "wrong-type": Dictionary(Type=Name.EmbeddedFile, F="other.pdf"),
        "name-entry": Dictionary(Type=Name.Filespec, F=Name.Other),
    }
    action = Dictionary(S=Name.Thread, D=thread if target == "thread" else target)
    action.F = specs.get(file, file) if isinstance(file, str) else file
    pdf.Root.OpenAction = action
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/OpenAction" in pdf.Root) == kept


@pytest.mark.parametrize("target,kept", [(0, True), ("Article", True), (1, False)])
def test_thread_action_with_a_null_file_is_in_the_current_file(target, kept):
    """A null /F is an absent /F, so the destination is checked in this file."""
    from pikepdf import Object

    pdf = new_pdf()
    _article(pdf)
    action = Object.parse(b"<< /S /Thread /F null >>")
    action.D = target
    pdf.Root.OpenAction = action
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/OpenAction" in pdf.Root) == kept


_TITLE_ENCODINGS = [
    (String("Article"), True),
    (String("Caf\xe9"), True),
    (String(b"\xfe\xff" + "Article \U0001f600".encode("utf-16-be")), True),
    # 0x7F is undefined in PDFDocEncoding (ISO 32000-1, Table D.2), as are
    # 0x01-0x08.
    (String(b"Article\x7f"), False),
    (String(b"Article\x01"), False),
    # Odd-length UTF-16BE and a lone high surrogate.
    (String(b"\xfe\xff" + "Article".encode("utf-16-be") + b"\x00"), False),
    (String(b"\xfe\xff\xd8\x3d\x00A"), False),
]


@pytest.mark.parametrize("title,kept", _TITLE_ENCODINGS)
def test_thread_title_into_another_file_must_be_a_text_string(title, kept):
    """Table 205: a title /D is a text string, also for a thread in another file."""
    pdf = new_pdf()
    _article(pdf)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, F="other.pdf", D=title)
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/OpenAction" in pdf.Root) == kept


@pytest.mark.parametrize("title,kept", _TITLE_ENCODINGS)
def test_thread_title_in_this_file_must_be_a_text_string(title, kept):
    """A malformed title /D never matches a thread title, even if it decodes."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    # A malformed /D decodes leniently, for instance odd-length UTF-16BE
    # "Article" to "Article"; the thread carries that title where it is valid.
    decoded = String(str(title))
    if kept:
        thread.I.Title = title
    elif _is_pdf_text_string(decoded):
        thread.I.Title = decoded
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=title)
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/OpenAction" in pdf.Root) == kept


def test_thread_dictionary_with_a_file_specification_is_checked():
    """A thread dictionary is in the current file whatever /F says (Table 205)."""
    pdf = new_pdf()
    thread, _, _ = _malformed_unlisted_article(pdf)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, F="other.pdf", D=thread)
    assert remove_forbidden_catalog_entries(pdf) == 3
    assert "/OpenAction" not in pdf.Root


@pytest.mark.parametrize(
    "bead,kept",
    [
        ("own", True),
        (0, True),
        (1, False),
        (-1, False),
        (True, False),
        ("other", False),
        (Name.Bad, False),
    ],
)
def test_thread_action_bead_must_belong_to_its_thread(bead, kept):
    """Table 205: /B is a bead of the destination thread or its index there.

    An invalid bead is removed, so that the action opens the thread at its
    first bead, rather than the whole action.
    """
    pdf = new_pdf()
    thread, own, _ = _article(pdf)
    other, foreign, _ = _article(pdf)
    pdf.Root.Threads = pdf.make_indirect(Array([thread, other]))
    action = pdf.make_indirect(Dictionary(S=Name.Thread, D=thread))
    action.B = {"own": own, "other": foreign}.get(bead, bead)
    pdf.Root.OpenAction = action
    _link(pdf, pdf.pages[0], A=action)
    assert remove_forbidden_catalog_entries(pdf) == (0 if kept else 1)
    assert ("/B" in action) == kept
    assert pdf.Root.OpenAction.objgen == action.objgen


def test_conversion_removes_actions_naming_a_malformed_unlisted_thread(tmp_path):
    """The output keeps the link and its other actions, without the broken ones."""
    from pikepdf import Pdf

    pdf = new_pdf()
    thread, page, other = _malformed_unlisted_article(pdf)
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread)
    _link(
        pdf,
        other,
        A=Dictionary(
            S=Name.GoTo,
            D=Array([page.obj, Name.Fit]),
            Next=Array([Dictionary(S=Name.Thread, D=thread)]),
        ),
    )
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        assert "/OpenAction" not in converted.Root
        link = converted.pages[1].Annots[0]
        assert link.A.S == Name.GoTo and "/Next" not in link.A
        assert all("/B" not in item.obj for item in converted.pages)


def test_repeated_threads_are_traversed_once(monkeypatch):
    """A thread listed or named many times is not validated again each time."""
    from pdftopdfa.sanitizers import navigation

    calls = []
    original = navigation._thread_beads

    def counting(thread, pages):
        calls.append(thread.objgen)
        return original(thread, pages)

    monkeypatch.setattr(navigation, "_thread_beads", counting)
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    unlisted, _, _, _, _ = _unlisted_article(pdf)
    pdf.Root.Threads = pdf.make_indirect(Array([thread] * 50))
    action = pdf.make_indirect(Dictionary(S=Name.Thread, D=unlisted))
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, D=thread, Next=Array([action] * 50))
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert calls == [thread.objgen, unlisted.objgen]


def test_action_scan_visits_shared_containers_once():
    """Shared /Annots arrays, annotations, /AA dictionaries and /Next arrays."""
    from pdftopdfa.sanitizers import navigation

    pdf = new_pdf()
    leaf = pdf.make_indirect(Dictionary(S=Name.Named, N=Name.NextPage))
    following = pdf.make_indirect(Array([leaf]))
    action = pdf.make_indirect(
        Dictionary(S=Name.Named, N=Name.PrevPage, Next=following)
    )
    annotation = _link(
        pdf, pdf.add_blank_page(), A=action, AA=pdf.make_indirect(Dictionary(E=leaf))
    )
    annots = pdf.make_indirect(Array([annotation] * 5))
    for _ in range(30):
        pdf.add_blank_page().Annots = annots
    pdf.Root.OpenAction = Dictionary(S=Name.Named, N=Name.FirstPage, Next=following)
    checked = []
    actions, removed = navigation._walk_actions(
        pdf, lambda item: checked.append(item.objgen) or False
    )
    # The open action, the annotation /A and /AA entries and the /Next entry.
    assert removed == 0 and len(checked) == 4
    assert len(actions) == 3


def test_page_bead_array_shared_by_pages_is_split():
    """Pages that share one /B array each regain their own beads."""
    pdf = new_pdf()
    _, first, second, page, other = _unlisted_article(pdf)
    page.B = other.B = pdf.make_indirect(Array([first, second]))
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert [item.objgen for item in page.B] == [first.objgen]
    assert [item.objgen for item in other.B] == [second.objgen]
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_shared_thread_information_is_checked_once(monkeypatch):
    """Threads that share one information dictionary share its repaired copy."""
    from pdftopdfa.sanitizers import navigation

    calls = []
    original = navigation._pdf_date

    def counting(value):
        calls.append(str(value))
        return original(value)

    monkeypatch.setattr(navigation, "_pdf_date", counting)
    pdf = new_pdf()
    threads = [_article(pdf)[0] for _ in range(5)]
    info = pdf.make_indirect(Dictionary(Title="Shared", ModDate="20240101"))
    for thread in threads:
        thread.I = info
    pdf.Root.Threads = pdf.make_indirect(Array(threads))
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert calls == ["20240101"] and str(info.ModDate) == "20240101"
    copies = {thread.I.objgen for thread in threads}
    assert len(copies) == 1 and info.objgen not in copies
    assert str(threads[0].I.ModDate) == "D:20240101"


def test_shared_collection_item_is_checked_once(monkeypatch):
    """A file specification listed under many names has its item checked once."""
    from pdftopdfa.sanitizers import navigation

    calls = []
    original = navigation._field_data

    def counting(value, subtype):
        calls.append(str(value))
        return original(value, subtype)

    monkeypatch.setattr(navigation, "_field_data", counting)
    pdf = new_pdf()
    spec = _portfolio(pdf, Dictionary(Good=Dictionary(Subtype=Name.S, N="Good")))
    del spec.CI.Foo
    names = []
    for index in range(20):
        names.extend([f"{index:02}.txt", spec])
    pdf.Root.Names.EmbeddedFiles = Dictionary(Names=Array(names))
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert calls == ["keep"]


def _utf16(text):
    return String(b"\xfe\xff" + text.encode("utf-16-be"))


@pytest.mark.parametrize("text", ["D:20240101000000Z", "20240101000000Z"])
def test_utf16_thread_information_dates_become_ascii(text):
    """ISO 32000-1, Table 34 lists a date as an ASCII string."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.ModDate = _utf16(text)
    thread.I.Title = _utf16("Article")
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert bytes(thread.I.ModDate) == b"D:20240101000000Z"
    # A text string keeps its UTF-16BE encoding.
    assert bytes(thread.I.Title) == bytes(_utf16("Article"))
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize("text", ["D:20240101000000Z", "20240101000000Z"])
def test_utf16_collection_dates_become_ascii(text):
    """Item and subitem dates are rewritten as ASCII strings of the same date."""
    pdf = new_pdf()
    spec = _portfolio(
        pdf,
        Dictionary(
            Sent=Dictionary(Subtype=Name.D, N="Sent"),
            Read=Dictionary(Subtype=Name.D, N="Read"),
        ),
    )
    spec.CI = Dictionary(Sent=_utf16(text), Read=Dictionary(D=_utf16(text)))
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert bytes(spec.CI.Sent) == bytes(spec.CI.Read.D) == b"D:20240101000000Z"
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_utf16_text_that_is_not_a_date_is_removed():
    """The decoded text must still be a PDF date."""
    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    thread.I.ModDate = _utf16("D:2024é")
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/ModDate" not in thread.I


@pytest.mark.parametrize("target", [7, "Elsewhere"])
@pytest.mark.parametrize("bead", [-1, True, Decimal("0.5"), Name.Bad, "bead", "dict"])
def test_thread_action_into_another_file_loses_an_invalid_bead(target, bead):
    """Table 205: a bead of a thread in another file is given only by its index.

    The bead dictionary form requires the thread to be in the current file, so
    an invalid bead is removed on its own and the action opens the first bead.
    """
    pdf = new_pdf()
    pdf.add_blank_page()
    action = Dictionary(S=Name.Thread, F="other.pdf", D=target)
    action.B = pdf.make_indirect(Dictionary(Type=Name.Bead)) if bead == "dict" else bead
    pdf.Root.OpenAction = action
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert sorted(pdf.Root.OpenAction.keys()) == ["/D", "/F", "/S"]
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_conversion_keeps_a_long_utf16_thread_title_valid(tmp_path):
    """A title over the string limit is cut at a UTF-16BE character boundary."""
    from pikepdf import Pdf

    from pdftopdfa.metadata import _is_pdf_text_string

    pdf = new_pdf()
    thread, _, _ = _article(pdf)
    title = "\U0001f600" + "\u00e9" * 20_000
    thread.I.Title = _utf16(title)
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output)
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        written = converted.Root.Threads[0].I.Title
        assert len(bytes(written)) <= 32_767 and _is_pdf_text_string(written)
        assert title.startswith(str(written)) and len(str(written)) > 16_000


def _two_file_portfolio(pdf):
    """Create a portfolio of two file specifications with a schema."""
    pdf.add_blank_page()
    specs = [
        pdf.make_indirect(Dictionary(Type=Name.Filespec, F=name, UF=name))
        for name in ("a.txt", "b.txt")
    ]
    pdf.Root.Names = Dictionary(
        EmbeddedFiles=Dictionary(Names=Array(["a.txt", specs[0], "b.txt", specs[1]]))
    )
    pdf.Root.Collection = Dictionary(
        Schema=Dictionary(Good=Dictionary(Subtype=Name.S, N="Good"))
    )
    return specs


@pytest.mark.parametrize("owner,shared", [(0, 1), (1, 0), (0, 0)])
def test_collection_item_that_is_a_listed_file_specification_is_detached(owner, shared):
    """A file specification is not a collection item, in whichever order.

    The /CI entry is removed rather than the file specification checked as an
    item, which would remove the entries that its embedded file needs.
    """
    pdf = new_pdf()
    specs = _two_file_portfolio(pdf)
    specs[owner].CI = specs[shared]
    expected = {key: repr(specs[shared][key]) for key in ("/Type", "/F", "/UF")}
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/CI" not in specs[owner]
    assert {key: repr(specs[shared][key]) for key in expected} == expected
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_collection_item_typed_as_a_file_specification_is_detached():
    """A file specification outside the name tree is left unchanged as well."""
    pdf = new_pdf()
    specs = _two_file_portfolio(pdf)
    attached = pdf.make_indirect(Dictionary(Type=Name.Filespec, F="c.txt", Good="keep"))
    specs[0].CI = attached
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/CI" not in specs[0]
    assert sorted(attached.keys()) == ["/F", "/Good", "/Type"]


def test_action_that_is_also_an_additional_actions_dictionary_is_checked():
    """An object scanned as an /AA dictionary is still followed as an action."""
    pdf = new_pdf()
    shared = pdf.make_indirect(
        Dictionary(
            S=Name.Named,
            N=Name.NextPage,
            Next=Array([Dictionary(S=Name.Thread, D=99)]),
        )
    )
    pdf.Root.OpenAction = shared
    _link(pdf, pdf.add_blank_page(), AA=shared)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/Next" not in shared
    assert pdf.Root.OpenAction.objgen == shared.objgen


@pytest.mark.parametrize("target", [-1, -7])
def test_thread_action_into_another_file_rejects_a_negative_index(target):
    """Table 205: an integer /D indexes the other file's /Threads array."""
    pdf = new_pdf()
    pdf.add_blank_page()
    pdf.Root.OpenAction = Dictionary(S=Name.Thread, F="other.pdf", D=target)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/OpenAction" not in pdf.Root


def _structures(pdf):
    """Create a valid article, portfolio and DocInfo of indirect objects.

    Returns the thread, the file specification and the objects, by role name,
    that a thread information or collection item reference may alias.
    """
    thread, bead, page = _article(pdf)
    specs = _two_file_portfolio(pdf)
    field = pdf.make_indirect(Dictionary(Subtype=Name.S, N="Good"))
    schema = pdf.make_indirect(Dictionary(Type=Name.CollectionSchema, Good=field))
    sort = pdf.make_indirect(Dictionary(Type=Name.CollectionSort, S=Name.Good))
    collection = pdf.make_indirect(
        Dictionary(Type=Name.Collection, Schema=schema, Sort=sort, View=Name.D)
    )
    pdf.Root.Collection = collection
    action = pdf.make_indirect(Dictionary(S=Name.Thread, D=thread))
    annotation = _link(pdf, pdf.pages[1], A=action)
    pdf.trailer.Info = pdf.make_indirect(Dictionary(Title="Document"))
    item = pdf.make_indirect(Dictionary(Type=Name.CollectionItem, Good="keep"))
    specs[1].CI = item
    targets = {
        "bead": bead,
        "thread": thread,
        "page": page.obj,
        "schema": schema,
        "field": field,
        "collection": collection,
        "sort": sort,
        "filespec": specs[1],
        "catalog": pdf.Root,
        "docinfo": pdf.trailer.Info,
        "action": action,
        "annotation": annotation,
        "item": item,
        "info": pdf.make_indirect(Dictionary(Title="Article")),
    }
    thread.I = targets["info"]
    return thread, specs[0], targets


def _snapshot(pdf):
    """Serialize every object, so that any edit of an existing one is detected."""
    return {
        item.objgen: bytes(item.unparse(resolved=True))
        for item in pdf.objects
        if item is not None
    }


def _unchanged(pdf, before, owner):
    """Check that no object existing before was edited, except the owner.

    Objects added since, such as sanitized copies, are not compared.
    """
    after = _snapshot(pdf)
    return all(
        after.get(objgen) == data
        for objgen, data in before.items()
        if objgen != owner.objgen
    )


def test_alias_fixture_is_valid():
    """Without an alias, nothing in the fixture needs a repair or a copy."""
    pdf = new_pdf()
    thread, _, targets = _structures(pdf)
    before = _snapshot(pdf)
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert _snapshot(pdf) == before
    assert thread.I.objgen == targets["info"].objgen
    assert targets["filespec"].CI.objgen == targets["item"].objgen


@pytest.mark.parametrize(
    "target",
    [
        "bead",
        "thread",
        "page",
        "schema",
        "field",
        "collection",
        "sort",
        "filespec",
        "catalog",
        "action",
        "annotation",
    ],
)
def test_thread_information_aliasing_another_role_is_copied(target):
    """A dictionary with another role is never edited as thread information.

    Checking it in place would remove the entries of that role, such as the /T,
    /N, /V, /P and /R links of a bead, so /I refers to a sanitized copy instead,
    which retains only the entries that the DocInfo syntax allows.
    """
    pdf = new_pdf()
    thread, _, targets = _structures(pdf)
    thread.I = targets[target]
    before = _snapshot(pdf)
    assert remove_forbidden_catalog_entries(pdf) >= 1
    assert _unchanged(pdf, before, thread)
    copy = thread.I
    assert copy.objgen not in ((0, 0), targets[target].objgen)
    assert all(_is_pdf_text_string(copy[key]) for key in copy.keys())
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert thread.I.objgen == copy.objgen


def test_thread_information_may_be_the_document_information():
    """/I has the syntax of DocInfo (Table 160), so the two may be one object.

    A valid one stays shared; a repair goes to a copy, leaving DocInfo itself
    to the conversion of the document information.
    """
    pdf = new_pdf()
    thread, _, targets = _structures(pdf)
    docinfo = targets["docinfo"]
    thread.I = docinfo
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert thread.I.objgen == docinfo.objgen

    docinfo.ModDate = "20240101"
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert thread.I.objgen not in ((0, 0), docinfo.objgen)
    assert str(thread.I.ModDate) == "D:20240101"
    assert str(thread.I.Title) == "Document"
    assert str(docinfo.ModDate) == "20240101"
    assert pdf.trailer.Info.objgen == docinfo.objgen


@pytest.mark.parametrize(
    "target",
    [
        "schema",
        "field",
        "collection",
        "sort",
        "bead",
        "thread",
        "page",
        "catalog",
        "docinfo",
        "action",
        "annotation",
    ],
)
def test_collection_item_aliasing_another_role_is_copied(target):
    """A dictionary with another role is never edited as a collection item.

    Checking it in place would remove its /Type and every entry that the schema
    does not describe, so /CI refers to a sanitized copy instead.
    """
    pdf = new_pdf()
    _, spec, targets = _structures(pdf)
    spec.CI = targets[target]
    before = _snapshot(pdf)
    assert remove_forbidden_catalog_entries(pdf) >= 1
    assert _unchanged(pdf, before, spec)
    copy = spec.CI
    assert copy.objgen not in ((0, 0), targets[target].objgen)
    # Only schema fields remain; the schema of the fixture has one, /Good.
    assert set(copy.keys()) <= {"/Good"}
    assert remove_forbidden_catalog_entries(pdf) == 0
    assert spec.CI.objgen == copy.objgen


def test_collection_item_that_is_a_file_specification_is_still_removed():
    """A file specification is not a collection item, so no copy is made."""
    pdf = new_pdf()
    _, spec, targets = _structures(pdf)
    spec.CI = targets["filespec"]
    before = _snapshot(pdf)
    assert remove_forbidden_catalog_entries(pdf) == 1
    assert "/CI" not in spec
    assert _unchanged(pdf, before, spec)
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "target,kept",
    [
        ("docinfo", True),
        ("bead", False),
        ("page", False),
        ("thread", True),
        ("schema", False),
    ],
)
def test_collection_subitem_aliasing_another_role_is_copied(target, kept):
    """A subitem date is rewritten in a copy, never in a shared dictionary.

    A dictionary typed as another object, or whose /P is no text string (such
    as the page of a bead), is no subitem (Table 49) and is removed from the
    item; any other one is kept, repaired in a copy if needed.
    """
    pdf = new_pdf()
    _, spec, targets = _structures(pdf)
    pdf.Root.Collection.Schema.Sent = Dictionary(Subtype=Name.D, N="Sent")
    targets["info"].D = targets["docinfo"].D = "20240101"
    spec.CI = pdf.make_indirect(Dictionary(Good="keep", Sent=targets[target]))
    before = _snapshot(pdf)
    remove_forbidden_catalog_entries(pdf)
    assert _unchanged(pdf, before, spec)
    assert str(spec.CI.Good) == "keep"
    assert ("/Sent" in spec.CI) is kept
    if kept and target == "docinfo":
        assert spec.CI.Sent.objgen not in ((0, 0), targets[target].objgen)
        assert str(spec.CI.Sent.D) == "D:20240101"
    elif kept:
        assert spec.CI.Sent.objgen == targets[target].objgen
    assert remove_forbidden_catalog_entries(pdf) == 0


@pytest.mark.parametrize(
    "alias", ["/I to item", "/CI to /I", "subitem to /I", "subitem to item"]
)
def test_dictionary_of_two_metadata_roles_gets_a_copy_per_role(alias):
    """Thread information, items and subitems are checked by different rules.

    Each role sanitizes its own copy where its rules require a change, so the
    dictionary that both refer to is left unchanged and valid in the other one.
    """
    pdf = new_pdf()
    thread, spec, targets = _structures(pdf)
    pdf.Root.Collection.Schema.Sent = Dictionary(Subtype=Name.D, N="Sent")
    info, item = targets["info"], targets["item"]
    info.D = "20240101"
    if alias == "/I to item":
        thread.I = shared = item
    elif alias == "/CI to /I":
        spec.CI = shared = info
    else:
        shared = info if alias == "subitem to /I" else item
        spec.CI = pdf.make_indirect(Dictionary(Good="keep", Sent=shared))
    before = bytes(shared.unparse(resolved=True))
    assert remove_forbidden_catalog_entries(pdf) >= 1
    assert bytes(shared.unparse(resolved=True)) == before
    if alias == "/I to item":
        # The /Type name is no DocInfo entry; the item itself stays valid.
        assert sorted(thread.I.keys()) == ["/Good"]
        assert targets["filespec"].CI.objgen == item.objgen
    elif alias == "/CI to /I":
        # Neither /Title nor /D is a field of the schema.
        assert sorted(spec.CI.keys()) == []
        assert thread.I.objgen == info.objgen
    elif alias == "subitem to /I":
        assert str(spec.CI.Sent.D) == "D:20240101" and str(info.D) == "20240101"
        assert thread.I.objgen == info.objgen
    else:
        # A dictionary typed /CollectionItem is no subitem.
        assert sorted(spec.CI.keys()) == ["/Good"]
        assert targets["filespec"].CI.objgen == item.objgen
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_conversion_copies_aliased_thread_information_and_items(tmp_path):
    """The converted article and portfolio stay valid and pass veraPDF."""
    from pikepdf import Pdf

    from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf

    pdf = new_pdf()
    thread, spec, targets = _structures(pdf)
    thread.I = targets["bead"]
    spec.CI = targets["schema"]
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output, level="3b")
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        written = converted.Root.Threads[0]
        bead = written.F
        assert written.I.objgen != bead.objgen and len(written.I.keys()) == 0
        assert sorted(bead.keys()) == ["/N", "/P", "/R", "/T", "/V"]
        assert bead.T.objgen == written.objgen and bead.N.objgen == bead.objgen
        assert [item.objgen for item in bead.P.B] == [bead.objgen]
        schema = converted.Root.Collection.Schema
        first = converted.Root.Names.EmbeddedFiles.Names[1]
        assert first.CI.objgen != schema.objgen
        assert sorted(first.CI.keys()) == ["/Good"]
        assert schema.Type == Name.CollectionSchema
        assert str(schema.Good.N) == "Good"
        assert sorted(converted.Root.Collection.Sort.keys()) == ["/S", "/Type"]
    if is_verapdf_available():
        validation = validate_with_verapdf(output, flavour="3b")
        assert validation.compliant, validation.failed_rules


def _nested_structures(pdf):
    """Create indirect dictionaries nested below the first page and a widget."""
    page = pdf.pages[0]
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
    )
    image = pdf.make_stream(b"\xff", Type=Name.XObject, Subtype=Name.Image)
    image.Width = image.Height = image.BitsPerComponent = 1
    image.ColorSpace = Name.DeviceGray
    resources = pdf.make_indirect(
        Dictionary(Font=Dictionary(F1=font), XObject=Dictionary(Im1=image))
    )
    page.obj.Resources = resources
    appearance = pdf.make_stream(b"", Type=Name.XObject, Subtype=Name.Form)
    appearance.BBox = Array([0, 0, 1, 1])
    characteristics = pdf.make_indirect(Dictionary(BG=Array([1]), CA="x"))
    appearances = pdf.make_indirect(Dictionary(N=appearance))
    widget = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([0, 0, 1, 1]),
            MK=characteristics,
            AP=appearances,
        )
    )
    page.obj.Annots = Array([widget])
    return {
        "resources": resources,
        "font": font,
        "mk": characteristics,
        "ap": appearances,
    }


@pytest.mark.parametrize(
    "role,target",
    [
        ("/CI", "resources"),
        ("/I", "resources"),
        ("/CI", "mk"),
        ("/I", "mk"),
        ("/CI", "ap"),
        ("/I", "ap"),
        ("/CI", "font"),
        ("/I", "font"),
    ],
)
def test_metadata_aliasing_a_nested_structural_dictionary_is_copied(role, target):
    """Page resources, fonts and annotation dictionaries are never edited.

    A /CI or /I reference to them is replaced by a sanitized copy, so the page
    keeps its /Font and /XObject resources and the widget its /MK and /AP.
    """
    pdf = new_pdf()
    thread, spec, _ = _structures(pdf)
    nested = _nested_structures(pdf)
    owner = spec if role == "/CI" else thread
    owner[role] = nested[target]
    before = _snapshot(pdf)
    assert remove_forbidden_catalog_entries(pdf) >= 1
    assert _unchanged(pdf, before, owner)
    assert owner[role].objgen not in ((0, 0), nested[target].objgen)
    resources = pdf.pages[0].Resources
    assert resources.objgen == nested["resources"].objgen
    assert sorted(resources.keys()) == ["/Font", "/XObject"]
    assert resources.Font.F1.objgen == nested["font"].objgen
    assert sorted(nested["font"].keys()) == ["/BaseFont", "/Subtype", "/Type"]
    assert sorted(nested["mk"].keys()) == ["/BG", "/CA"]
    assert sorted(nested["ap"].keys()) == ["/N"]
    assert remove_forbidden_catalog_entries(pdf) == 0


def test_shared_collection_item_copy_is_shared():
    """File specifications that share an item share its one sanitized copy."""
    pdf = new_pdf()
    _, spec, targets = _structures(pdf)
    resources = _nested_structures(pdf)["resources"]
    spec.CI = targets["filespec"].CI = resources
    assert remove_forbidden_catalog_entries(pdf) == 2
    assert spec.CI.objgen == targets["filespec"].CI.objgen != resources.objgen
    assert sorted(resources.keys()) == ["/Font", "/XObject"]


def test_conversion_keeps_page_resources_aliased_by_a_collection_item(tmp_path):
    """A /CI that is the page /Resources leaves the resources intact."""
    from pikepdf import Pdf

    from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf

    pdf = new_pdf()
    _, spec, _ = _structures(pdf)
    resources = _nested_structures(pdf)["resources"]
    del pdf.pages[0].obj["/Annots"]
    pdf.pages[0].Contents = pdf.make_stream(
        b"BT /F1 12 Tf 10 10 Td (Hi) Tj ET q 10 0 0 10 50 50 cm /Im1 Do Q"
    )
    spec.CI = resources
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    pdf.save(source)
    result = convert_to_pdfa(source, output, level="3b")
    assert result.target_produced and not result.skipped, result
    with Pdf.open(output) as converted:
        page = converted.pages[0]
        assert "/F1" in page.Resources.Font and "/Im1" in page.Resources.XObject
        item = converted.Root.Names.EmbeddedFiles.Names[1].CI
        assert item.objgen != page.Resources.objgen
        assert "/Font" not in item and "/XObject" not in item
    if is_verapdf_available():
        validation = validate_with_verapdf(output, flavour="3b")
        assert validation.compliant, validation.failed_rules
