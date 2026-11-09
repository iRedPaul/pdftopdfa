# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate preserved collection settings and circular article navigation."""

import logging
import math
import re
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Pdf, String

from ..exceptions import ConversionError
from .files import _is_filespec, _iter_name_tree_pairs

logger = logging.getLogger(__name__)


def _pdf_date(value):
    """Require complete PDF date syntax as well as valid calendar components."""
    from ..metadata import _parse_pdf_date

    if not isinstance(value, String) or not re.fullmatch(
        r"D:\d{4}(?:\d{2}){0,5}(?:Z|[+-]\d{2}(?:'\d{2}'?)?)?", str(value)
    ):
        return None
    try:
        return _parse_pdf_date(str(value))
    except (ValueError, OverflowError):
        return None


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
    """Check a collection item value, or a subitem's data, against its field type."""
    if isinstance(value, Dictionary):
        if value.get("/Type") not in (None, Name.CollectionSubitem):
            return False
        if value.get("/P") is not None and not isinstance(value.P, String):
            return False
        value = value.get("/D")
        if value is None:
            return True
    if subtype in (Name.S, Name.F, Name.Desc):
        return isinstance(value, String)
    if subtype in (Name.D, Name.ModDate, Name.CreationDate):
        return _pdf_date(value) is not None
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
                or not isinstance(field.get("/N"), String)
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
    fields = [fields] if isinstance(fields, Name) else fields
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
    # ISO 32000-1 Table 158 explicitly permits unequal /S and /A lengths.
    if ascending is not None and not (
        isinstance(ascending, bool)
        or isinstance(ascending, Array)
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


def _validate_threads(pdf: Pdf) -> dict[tuple[int, int], tuple]:
    """Validate bead cycles and page links without renumbering local Thread actions.

    Returns:
        For each bead, in /Threads and reading order: its thread, position on that
        thread, page and bead dictionary.
    """
    threads = pdf.Root.get("/Threads")
    if threads is None:
        return {}
    if not isinstance(threads, Array):
        raise _thread_error("expected an array")
    pages = {page.objgen for page in pdf.pages}
    beads = {}
    for value in threads:
        thread = _reference(value, Name.Thread)
        info = thread.get("/I")
        if info is not None:
            if not isinstance(info, Dictionary) or any(
                item is not None
                and not (
                    item in (Name("/True"), Name("/False"), Name.Unknown)
                    if key == "/Trapped"
                    else isinstance(item, String)
                )
                for key, item in info.items()
            ):
                raise _thread_error("invalid thread information dictionary")
            if any(
                info.get(key) is not None and _pdf_date(info[key]) is None
                for key in ("/CreationDate", "/ModDate")
            ):
                raise _thread_error("invalid thread information date")
        first = _reference(thread.get("/F"), Name.Bead)
        bead, seen = first, set()
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
            # A thread listed twice in /Threads keeps the positions of its first
            # traversal; separate threads cannot share beads, because the first
            # bead of each one names its own thread.
            beads.setdefault(
                bead.objgen, (thread.objgen, len(seen) - 1, page_ref.objgen, bead)
            )
            bead = following
        if bead.objgen != first.objgen:
            raise _thread_error("bead cycle does not return to the first bead")
    return beads


def _make_threads_indirect(pdf: Pdf) -> int:
    """ISO 32000-1, Table 28: the catalog /Threads array is an indirect object."""
    threads = pdf.Root.get("/Threads")
    if threads is None or threads.objgen != (0, 0):
        return 0
    pdf.Root.Threads = pdf.make_indirect(threads)
    logger.warning("Made the catalog /Threads array an indirect object")
    return 1


def _repair_page_beads(pdf: Pdf, beads: dict[tuple[int, int], tuple]) -> int:
    """List exactly the beads of each page once, in the order of their threads.

    ISO 32000-1, 12.4.3 requires a /B array on every page on which article beads
    appear, and Table 30 notes that it can be recreated from /Threads. The validated
    bead cycles define the beads of each page, so omitted, duplicated or unrelated
    entries are repaired. Only a thread defines an order for its own beads (Table
    30's natural reading order), so other threads keep their positions in the array,
    and beads that the array omitted follow the listed ones.
    """
    on_page: dict[tuple[int, int], dict[tuple[int, int], Dictionary]] = {}
    for bead_id, (_, _, page_id, bead) in beads.items():
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
        slots: dict[tuple[int, int], list[int]] = {}
        for index, bead in enumerate(ordered):
            slots.setdefault(beads[bead.objgen][0], []).append(index)
        for indices in slots.values():
            thread_beads = sorted(
                (ordered[index] for index in indices),
                key=lambda bead: beads[bead.objgen][1],
            )
            for index, bead in zip(indices, thread_beads, strict=True):
                ordered[index] = bead
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
        /B array) and malformed collection entries removed, all in place.

    Raises:
        ConversionError: If article navigation is malformed. Article validation
            precedes collection edits and leaves thread destinations unchanged.
    """
    beads = _validate_threads(pdf)
    return (
        _make_threads_indirect(pdf)
        + _repair_page_beads(pdf, beads)
        + _sanitize_collection(pdf)
    )
