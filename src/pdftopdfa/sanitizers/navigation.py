# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate preserved collection settings and circular article navigation."""

import logging
import math
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Pdf, String

from ..exceptions import ConversionError
from .files import _iter_name_tree_pairs
from .trapnet import _date as _pdf_date

logger = logging.getLogger(__name__)


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
    """Return the name and file specification of each embedded file."""
    names = pdf.Root.get("/Names")
    embedded = names.get("/EmbeddedFiles") if isinstance(names, Dictionary) else None
    if not isinstance(embedded, Dictionary):
        return []
    return [
        (name, filespec)
        for name, filespec in _iter_name_tree_pairs(embedded)
        if isinstance(filespec, Dictionary)
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
    if subtype == Name.S:
        return isinstance(value, String)
    if subtype == Name.D:
        return _pdf_date(value) is not None
    if subtype == Name.N:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float, Decimal))
            and math.isfinite(value)
        )
    # Other field types take their data from the file specification itself.
    return True


def _sanitize_collection_items(pdf: Pdf, schema, check_fields: bool) -> int:
    """Remove malformed item entries, and values that the schema does not describe."""
    removed = 0
    for _, filespec in _embedded_files(pdf):
        item = filespec.get("/CI")
        if item is None:
            continue
        if not isinstance(item, Dictionary):
            removed += _remove_entry(filespec, "/CI")
            continue
        removed += _optional_type(item, Name.CollectionItem)
        if not check_fields:
            continue
        for key in list(item.keys()):
            if key == "/Type":
                continue
            field = schema.get(key) if schema is not None else None
            if field is None or not _valid_item_value(item[key], field.get("/Subtype")):
                removed += _remove_entry(item, key)
    return removed


def remove_dangling_collection_default(pdf: Pdf) -> int:
    """Remove the initial document of a portfolio unless it names an embedded file."""
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
    # Without any schema there is nothing to compare item values with, unless it
    # is removed here and so no longer describes the values that relied on it.
    check_fields = schema is not None
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
    removed += _sanitize_collection_items(pdf, schema, check_fields)
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


def _validate_threads(pdf: Pdf) -> None:
    """Validate bead cycles and page links without renumbering local Thread actions."""
    threads = pdf.Root.get("/Threads")
    if threads is None:
        return
    if not isinstance(threads, Array):
        raise _thread_error("expected an array")
    pages = {page.objgen: page for page in pdf.pages}
    page_bead_ids = {}
    for page_id, page in pages.items():
        listed = page.get("/B")
        if listed is None:
            continue
        if not isinstance(listed, Array) or any(
            not isinstance(bead, Dictionary) or bead.objgen == (0, 0) for bead in listed
        ):
            raise _thread_error("malformed page /B array")
        page_bead_ids[page_id] = {bead.objgen for bead in listed}
    all_beads = {}
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
            # ISO 32000-1, Table 30: /B is optional and can be recovered from
            # /Threads, so only an array that is present must list the bead.
            listed_beads = page_bead_ids.get(page_ref.objgen)
            if listed_beads is not None and bead.objgen not in listed_beads:
                raise _thread_error("bead is missing from its page's /B array")
            all_beads[bead.objgen] = page_ref.objgen
            bead = following
        if bead.objgen != first.objgen:
            raise _thread_error("bead cycle does not return to the first bead")
    for page_id, bead_ids in page_bead_ids.items():
        if any(all_beads.get(bead_id) != page_id for bead_id in bead_ids):
            raise _thread_error("page /B contains an unrelated or malformed bead")


def sanitize_catalog_navigation(pdf: Pdf) -> int:
    """Preserve portfolios/articles and reject ambiguous repairs of article graphs.

    Returns:
        Number of malformed collection entries removed in place.

    Raises:
        ConversionError: If article navigation is malformed. Article validation
            precedes collection edits and leaves thread destinations unchanged.
    """
    _validate_threads(pdf)
    return _sanitize_collection(pdf)
