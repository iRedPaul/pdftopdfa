# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate preserved collection settings and circular article navigation."""

import logging
import math
import re
from datetime import datetime
from decimal import Decimal
from itertools import chain

from pikepdf import Array, Dictionary, Name, Pdf, String

from ..exceptions import ConversionError
from ..metadata import _is_pdf_text_string
from .files import _is_filespec, _iter_name_tree_pairs

logger = logging.getLogger(__name__)

_THREAD = Name.Thread


# ISO 32000-1, 7.9.4: D:YYYYMMDDHHmmSSOHH'mm. Each field may be present only
# together with all of the preceding ones, so the relationship to UT (O) needs a
# complete date and time. An hour offset may end with its apostrophe, and the
# trailing apostrophe of PDF 1.7 and earlier versions (HH'mm') is tolerated. Z
# denotes equality with UT, so an offset may follow it only as zero (Z00'00').
_PDF_DATE = re.compile(
    r"D:([0-9]{4})(?:([0-9]{2})(?:([0-9]{2})(?:([0-9]{2})(?:([0-9]{2})(?:([0-9]{2})"
    r"(?:Z(?:00(?:'(?:00'?)?)?)?"
    r"|[+-](?:(?:[01][0-9]|2[0-3])(?:'(?:[0-5][0-9]'?)?)?)?)?"
    r")?)?)?)?)?"
)


def _is_pdf_date(value) -> bool:
    """Require complete PDF date syntax as well as valid calendar components.

    Omitted fields take their 7.9.4 defaults (01 for month and day, 00 otherwise);
    the month, the day of that month, the hour, minute and second must exist.
    """
    match = isinstance(value, String) and _PDF_DATE.fullmatch(str(value))
    if not match:
        return False
    fields = [
        int(field) if field else default
        for field, default in zip(match.groups(), (0, 1, 1, 0, 0, 0), strict=True)
    ]
    try:
        datetime(*fields)
    except ValueError:
        return False
    return True


def _remove_entry(owner, key: str) -> int:
    """Remove one malformed optional entry without discarding its valid siblings."""
    del owner[key]
    logger.warning("Removed malformed navigation entry %s", key)
    return 1


def _optional_type(owner, expected: Name) -> int:
    """Omit an invalid optional type marker; its container defines the object type."""
    value = owner.get("/Type")
    return (
        _remove_entry(owner, "/Type") if value is not None and value != expected else 0
    )


def _embedded_files(pdf: Pdf) -> list[tuple]:
    """Return the name and file specification of each embedded file.

    Values that are not file specifications are skipped, as the name-tree
    normalization of the embedded-file sanitizers later discards them.
    """
    names = pdf.Root.get("/Names")
    embedded = names.get("/EmbeddedFiles") if isinstance(names, Dictionary) else None
    if not isinstance(embedded, Dictionary):
        return []
    return [
        (name, filespec)
        for name, filespec in _iter_name_tree_pairs(embedded)
        if _is_filespec(filespec)
    ]


def _valid_item_value(value, subtype) -> bool:
    """Check a collection item value, or a subitem's data, against its field type.

    ISO 32000-1, Table 49 (and PDF Reference 1.7, Table 3.43) defines every entry
    of a collection subitem dictionary, /D included, as optional with no default,
    so a subitem with only a /P prefix, or none at all, is well formed. A subitem
    whose /Type, /P or /D contradicts the field is removed as one item value.
    A text string, the /P prefix included, needs a valid PDFDocEncoding or
    UTF-16BE encoding (7.9.2.2), as retained DocInfo values do.
    """
    if isinstance(value, Dictionary):
        if value.get("/Type") not in (None, Name.CollectionSubitem):
            return False
        if value.get("/P") is not None and not _is_pdf_text_string(value.P):
            return False
        value = value.get("/D")
        if value is None:
            return True
    if subtype in (Name.S, Name.F, Name.Desc):
        return _is_pdf_text_string(value)
    if subtype in (Name.D, Name.ModDate, Name.CreationDate):
        return _is_pdf_date(value)
    return (
        subtype in (Name.N, Name.Size)
        and not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(value)
    )


def _sanitize_collection_items(pdf: Pdf, schema) -> int:
    """Remove malformed item entries, and values that the schema does not describe.

    Without a schema there is nothing to compare the values with. PDF/A does not
    require their removal, as veraPDF confirms, so a portfolio keeps this data.
    A schema left with no fields (Table 156 makes each of them optional) describes
    no values, so its items keep only their /Type. Only files of the EmbeddedFiles
    name tree belong to the collection (ISO 32000-1, 12.3.5), so the items of other
    file specifications are not checked against its schema.
    """
    removed = 0
    for _, filespec in _embedded_files(pdf):
        item = filespec.get("/CI")
        if item is None:
            continue
        if not isinstance(item, Dictionary):
            removed += _remove_entry(filespec, "/CI")
            continue
        # A file specification refers to its collection item indirectly.
        if item.objgen == (0, 0):
            item = pdf.make_indirect(item)
            filespec["/CI"] = item
        removed += _optional_type(item, Name.CollectionItem)
        if schema is None:
            continue
        for key in list(item.keys()):
            if key == "/Type":
                continue
            field = schema.get(key)
            if field is None or not _valid_item_value(item[key], field.get("/Subtype")):
                removed += _remove_entry(item, key)
    return removed


def remove_dangling_collection_default(pdf: Pdf) -> int:
    """Remove the initial document of a portfolio unless it names an embedded file.

    The sanitizer pipeline repeats this check after every embedded-file cleanup,
    which can remove the file that /D names.
    """
    collection = pdf.Root.get("/Collection")
    if not isinstance(collection, Dictionary) or collection.get("/D") is None:
        return 0
    default = collection.D
    if isinstance(default, String) and any(
        isinstance(name, String) and bytes(name) == bytes(default)
        for name, _ in _embedded_files(pdf)
    ):
        return 0
    return _remove_entry(collection, "/D")


def _sanitize_collection(pdf: Pdf) -> int:
    """Repair optional portfolio settings while preserving valid fields and sorting."""
    collection = pdf.Root.get("/Collection")
    if collection is None:
        return 0
    if not isinstance(collection, Dictionary):
        return _remove_entry(pdf.Root, "/Collection")
    removed = _optional_type(collection, Name.Collection)
    view = collection.get("/View")
    if view is not None and view not in (Name.D, Name.T, Name.H):
        removed += _remove_entry(collection, "/View")
    removed += remove_dangling_collection_default(pdf)
    schema = collection.get("/Schema")
    if schema is not None and not isinstance(schema, Dictionary):
        removed += _remove_entry(collection, "/Schema")
        schema = None
    if isinstance(schema, Dictionary):
        removed += _optional_type(schema, Name.CollectionSchema)
        for key in list(schema.keys()):
            if key == "/Type":
                continue
            field = schema[key]
            if (
                not isinstance(field, Dictionary)
                or field.get("/Subtype")
                not in (
                    Name.S,
                    Name.D,
                    Name.N,
                    Name.F,
                    Name.Desc,
                    Name.ModDate,
                    Name.CreationDate,
                    Name.Size,
                )
                or not _is_pdf_text_string(field.get("/N"))
            ):
                removed += _remove_entry(schema, key)
                continue
            removed += _optional_type(field, Name.CollectionField)
            for flag in ("/V", "/E"):
                if field.get(flag) is not None and not isinstance(field[flag], bool):
                    removed += _remove_entry(field, flag)
            order = field.get("/O")
            if order is not None and (
                isinstance(order, bool) or not isinstance(order, int)
            ):
                removed += _remove_entry(field, "/O")
    removed += _sanitize_collection_items(pdf, schema)
    sorting = collection.get("/Sort")
    if sorting is None:
        return removed
    if not isinstance(sorting, Dictionary):
        return removed + _remove_entry(collection, "/Sort")
    removed += _optional_type(sorting, Name.CollectionSort)
    fields = sorting.get("/S")
    scalar = isinstance(fields, Name)
    fields = [fields] if scalar else fields
    # ISO 32000-1 Table 158: each /S name identifies a field of the collection
    # schema, so sorting cannot be valid without one.
    valid = (
        isinstance(schema, Dictionary)
        and isinstance(fields, (list, Array))
        and len(fields) > 0
        and all(
            isinstance(field, Name) and str(field) != "/Type" and str(field) in schema
            for field in fields
        )
    )
    ascending = sorting.get("/A")
    # ISO 32000-1 Table 158 explicitly permits unequal /S and /A lengths. Each
    # element of an /A array refers to the entry at the same index of an /S
    # array, so a single /S name takes a single boolean; without /A it is sorted
    # in ascending order.
    if ascending is not None and not (
        isinstance(ascending, bool)
        or not scalar
        and isinstance(ascending, Array)
        and all(isinstance(item, bool) for item in ascending)
    ):
        removed += _remove_entry(sorting, "/A")
    if not valid:
        removed += _remove_entry(collection, "/Sort")
    return removed


def _thread_error(detail: str) -> ConversionError:
    """Reject ambiguous article repairs rather than leaving dangling destinations."""
    return ConversionError(f"Malformed /Threads article navigation: {detail}")


def _reference(value, expected: Name):
    """Require an indirect thread/bead dictionary and its optional correct type."""
    if (
        not isinstance(value, Dictionary)
        or value.objgen == (0, 0)
        or value.get("/Type") not in (None, expected)
    ):
        raise _thread_error(f"invalid {expected} reference")
    return value


def _thread_beads(thread, pages: set) -> list[tuple]:
    """Validate the bead cycle of one thread and its page links.

    Returns:
        The id, page id and dictionary of each bead, in reading order.
    """
    thread = _reference(thread, Name.Thread)
    first = _reference(thread.get("/F"), Name.Bead)
    beads, bead, seen = [], first, set()
    while bead.objgen not in seen:
        seen.add(bead.objgen)
        owner = bead.get("/T")
        if owner is not None or bead.objgen == first.objgen:
            if _reference(owner, Name.Thread).objgen != thread.objgen:
                raise _thread_error("bead belongs to a different thread")
        following = _reference(bead.get("/N"), Name.Bead)
        previous = _reference(bead.get("/V"), Name.Bead)
        if (
            _reference(following.get("/V"), Name.Bead).objgen != bead.objgen
            or _reference(previous.get("/N"), Name.Bead).objgen != bead.objgen
        ):
            raise _thread_error("inconsistent next/previous links")
        page_ref = bead.get("/P")
        if not isinstance(page_ref, Dictionary) or page_ref.objgen not in pages:
            raise _thread_error("bead refers to a nonexistent page")
        rectangle = bead.get("/R")
        if (
            not isinstance(rectangle, Array)
            or len(rectangle) != 4
            or not all(
                not isinstance(number, bool)
                and isinstance(number, (int, float, Decimal))
                and math.isfinite(number)
                for number in rectangle
            )
        ):
            raise _thread_error("invalid bead rectangle")
        beads.append((bead.objgen, page_ref.objgen, bead))
        bead = following
    if bead.objgen != first.objgen:
        raise _thread_error("bead cycle does not return to the first bead")
    return beads


def _action_threads(pdf: Pdf):
    """Yield the threads that Thread actions name directly by their dictionary.

    ISO 32000-1, 12.6.4.6 lets a Thread action refer to a thread of the same file
    by its dictionary, so the thread need not appear in the optional catalog
    /Threads array. Actions of the document, its pages, annotations and outline
    items are searched, including the actions that follow them (/Next).

    The keys of each dictionary are listed once: with pikepdf, looking up an
    absent key costs far more, which matters for the many annotations of a large
    document.
    """
    owners = [pdf.Root]
    for page in pdf.pages:
        owners.append(page.obj)
        if "/Annots" in page.obj.keys() and isinstance(page.obj.Annots, Array):
            owners.extend(
                item for item in page.obj.Annots if isinstance(item, Dictionary)
            )
    outlines = pdf.Root.get("/Outlines")
    items = [outlines] if isinstance(outlines, Dictionary) else []
    visited = set()
    while items:
        item = items.pop()
        if not isinstance(item, Dictionary) or item.objgen in visited:
            continue
        if item.objgen != (0, 0):
            visited.add(item.objgen)
        keys = item.keys()
        if item is not outlines:
            owners.append(item)
        items.extend(item[key] for key in ("/First", "/Next") if key in keys)
    pending = [pdf.Root.get("/OpenAction")]
    for owner in owners:
        keys = owner.keys()
        if "/A" in keys:
            pending.append(owner["/A"])
        if "/AA" in keys and isinstance(owner["/AA"], Dictionary):
            pending.extend(owner["/AA"].values())
    visited = set()
    while pending:
        action = pending.pop()
        if not isinstance(action, Dictionary) or action.objgen in visited:
            continue
        if action.objgen != (0, 0):
            visited.add(action.objgen)
        keys = action.keys()
        if (
            "/S" in keys
            and "/D" in keys
            and action["/S"] == _THREAD
            and isinstance(action["/D"], Dictionary)
        ):
            yield action["/D"]
        if "/Next" in keys:
            following = action["/Next"]
            pending.extend(following if isinstance(following, Array) else [following])


def _listed_bead_threads(pdf: Pdf):
    """Yield the thread of each bead cycle that a page /B array refers to.

    Only the first bead of a thread has to name it (/T), so each cycle is followed
    until a bead does; no bead is followed twice.
    """
    visited = set()
    for page in pdf.pages:
        listed = page.obj.get("/B")
        for bead in listed if isinstance(listed, Array) else []:
            while (
                isinstance(bead, Dictionary)
                and bead.objgen != (0, 0)
                and bead.objgen not in visited
            ):
                visited.add(bead.objgen)
                owner = bead.get("/T")
                if isinstance(owner, Dictionary):
                    yield owner
                    break
                bead = bead.get("/N")


def _validate_threads(pdf: Pdf) -> tuple[list, dict[tuple[int, int], tuple]]:
    """Validate bead cycles and page links without renumbering local Thread actions.

    Every thread of the catalog /Threads array must be valid. A thread that a
    Thread action or a page /B array refers to without being listed there (the
    catalog array is optional, Table 28) supplies its beads when it is valid.

    Returns:
        The valid threads, and for each of their beads, in /Threads and reading
        order: its page and bead dictionary.
    """
    threads = pdf.Root.get("/Threads")
    if threads is not None and not isinstance(threads, Array):
        raise _thread_error("expected an array")
    pages = {page.objgen for page in pdf.pages}
    found, beads = {}, {}
    for thread in threads if threads is not None else []:
        for bead_id, page_id, bead in _thread_beads(thread, pages):
            # A thread listed twice keeps the place of its first traversal;
            # separate threads cannot share beads, because the first bead of
            # each one names its own thread.
            beads.setdefault(bead_id, (page_id, bead))
        found.setdefault(thread.objgen, thread)
    tried = set(found)
    for thread in chain(_action_threads(pdf), _listed_bead_threads(pdf)):
        if thread.objgen in tried:
            continue
        tried.add(thread.objgen)
        try:
            cycle = _thread_beads(thread, pages)
        except ConversionError:
            continue
        found[thread.objgen] = thread
        for bead_id, page_id, bead in cycle:
            beads.setdefault(bead_id, (page_id, bead))
    return list(found.values()), beads


def _sanitize_thread_info(threads: list) -> int:
    """Remove the thread information entries that DocInfo could not contain.

    ISO 32000-1, Table 160 gives /I the syntax of the document information
    dictionary (14.3.3). Like the conversion of DocInfo, each malformed entry is
    removed on its own, keeping its valid siblings and the article: a date that
    is not a PDF date, a /Trapped value other than the three names, and a text
    string, standard or custom, whose encoding is invalid.
    """
    removed = 0
    for thread in threads:
        info = thread.get("/I")
        if info is None:
            continue
        if not isinstance(info, Dictionary):
            removed += _remove_entry(thread, "/I")
            continue
        for key in list(info.keys()):
            value = info.get(key)
            if value is None:
                continue
            if key == "/Trapped":
                valid = isinstance(value, Name) and value in (
                    Name("/True"),
                    Name("/False"),
                    Name.Unknown,
                )
            elif key in ("/CreationDate", "/ModDate"):
                valid = _is_pdf_date(value)
            else:
                valid = _is_pdf_text_string(value)
            if not valid:
                removed += _remove_entry(info, key)
    return removed


def _make_threads_indirect(pdf: Pdf) -> int:
    """ISO 32000-1, Table 28: the catalog /Threads array is an indirect object."""
    threads = pdf.Root.get("/Threads")
    if threads is None or threads.objgen != (0, 0):
        return 0
    pdf.Root.Threads = pdf.make_indirect(threads)
    logger.warning("Made the catalog /Threads array an indirect object")
    return 1


def _repair_page_beads(pdf: Pdf, beads: dict[tuple[int, int], tuple]) -> int:
    """List exactly the beads of each page once, keeping the order of the array.

    ISO 32000-1, 12.4.3 requires a /B array on every page on which article beads
    appear, and Table 30 notes that it can be recreated from /Threads. The valid
    bead cycles define the beads of each page, including those of threads that
    only actions or /B arrays refer to, so omitted and duplicated beads are
    repaired and only entries that belong to no valid cycle or to another page
    are removed. The array lists the beads in drawing order (12.4.3),
    which the threads do not record and which can differ from their reading order,
    so the listed beads keep their relative order, and a complete array stays
    unchanged. Beads that the array omitted follow, in /Threads and reading order.
    """
    on_page: dict[tuple[int, int], dict[tuple[int, int], Dictionary]] = {}
    for bead_id, (page_id, bead) in beads.items():
        on_page.setdefault(page_id, {})[bead_id] = bead

    repaired = 0
    for page in pdf.pages:
        expected = on_page.get(page.objgen, {})
        listed = page.get("/B")
        if listed is None and not expected:
            continue
        original = list(listed) if isinstance(listed, Array) else None
        # Direct dictionaries are never beads of a cycle, which requires indirect
        # references, and a dictionary keeps the first occurrence of each bead.
        kept = {
            item.objgen: expected[item.objgen]
            for item in original or []
            if isinstance(item, Dictionary) and item.objgen in expected
        }
        kept.update({key: bead for key, bead in expected.items() if key not in kept})
        ordered = list(kept.values())
        if (
            original is not None
            and len(original) == len(ordered)
            and all(
                isinstance(item, Dictionary) and item.objgen == bead.objgen
                for item, bead in zip(original, ordered, strict=True)
            )
        ):
            continue
        if ordered:
            page.B = Array(ordered)
        else:
            del page.obj["/B"]
        repaired += 1
        logger.warning("Repaired the beads listed in a page /B array")
    return repaired


def sanitize_catalog_navigation(pdf: Pdf) -> int:
    """Preserve portfolios/articles and reject ambiguous repairs of article graphs.

    Returns:
        Number of article structures repaired (an inline /Threads array or a page
        /B array) and malformed thread information and collection entries
        removed, all in place.

    Raises:
        ConversionError: If an article of the catalog /Threads array is malformed.
            Article validation precedes all edits and leaves thread destinations
            unchanged.
    """
    threads, beads = _validate_threads(pdf)
    return (
        _make_threads_indirect(pdf)
        + _repair_page_beads(pdf, beads)
        + _sanitize_thread_info(threads)
        + _sanitize_collection(pdf)
    )
