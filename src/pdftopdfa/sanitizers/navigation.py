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


def _is_pdf_date(text: str) -> bool:
    """Require complete PDF date syntax as well as valid calendar components.

    Omitted fields take their 7.9.4 defaults (01 for month and day, 00 otherwise);
    the month, the day of that month, the hour, minute and second must exist.
    """
    match = _PDF_DATE.fullmatch(text)
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


def _check_date(owner, key: str) -> int | None:
    """Validate a date entry, adding the D: prefix that ISO 32000-1 requires.

    PDF Reference 1.7, 3.8.3 made the prefix optional ("The prefix D:, although
    also optional, is strongly recommended"), whereas ISO 32000-1, 7.9.4 states
    "The prefix D: shall be present". A date that is valid apart from the prefix
    is therefore kept with the prefix added.

    ISO 32000-1, Table 34 lists a date as "Date (ASCII string)", so a date stored
    in another encoding, such as UTF-16BE with a byte order marker, is rewritten
    as the ASCII string of the same date, as DocInfo dates are written.

    Returns:
        The number of repaired entries, or None if the value is not a PDF date.
    """
    value = owner[key]
    if not isinstance(value, String):
        return None
    text = str(value)
    if not text.startswith("D:"):
        text = f"D:{text}"
    # The date syntax admits only ASCII characters.
    if not _is_pdf_date(text):
        return None
    canonical = text.encode("ascii")
    if bytes(value) == canonical:
        return 0
    owner[key] = String(canonical)
    logger.warning("Rewrote navigation date %s as an ASCII string with D:", key)
    return 1


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


def _first_visit(item, seen: set) -> bool:
    """Record an indirect object; a direct object has only the owner it is in."""
    key = item.objgen
    if key == (0, 0):
        return True
    if key in seen:
        return False
    seen.add(key)
    return True


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


def _check_item_value(item, key: str, subtype) -> int | None:
    """Check a collection item value, or a subitem's data, against its field type.

    Returns the number of repaired dates, or None if the value is to be removed.

    ISO 32000-1, Table 49 (and PDF Reference 1.7, Table 3.43) defines every entry
    of a collection subitem dictionary, /D included, as optional with no default,
    so a subitem with only a /P prefix, or none at all, is well formed. A subitem
    whose /Type, /P or /D contradicts the field is removed as one item value.
    A text string, the /P prefix included, needs a valid PDFDocEncoding or
    UTF-16BE encoding (7.9.2.2), as retained DocInfo values do.
    """
    owner, value = item, item[key]
    if isinstance(value, Dictionary):
        if value.get("/Type") not in (None, Name.CollectionSubitem):
            return None
        if value.get("/P") is not None and not _is_pdf_text_string(value.P):
            return None
        owner, key = value, "/D"
        if value.get("/D") is None:
            return 0
        value = value.D
    if subtype == Name.S:
        return 0 if _is_pdf_text_string(value) else None
    if subtype == Name.D:
        return _check_date(owner, key)
    valid = (
        subtype == Name.N
        and not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(value)
    )
    return 0 if valid else None


def _sanitize_collection_items(pdf: Pdf, schema) -> int:
    """Remove malformed item entries, and values that the schema does not describe.

    Without a schema there is nothing to compare the values with. PDF/A does not
    require their removal, as veraPDF confirms, so a portfolio keeps this data.
    A schema left with no fields (Table 156 makes each of them optional) describes
    no values, so its items keep only their /Type. Only files of the EmbeddedFiles
    name tree belong to the collection (ISO 32000-1, 12.3.5), so the items of other
    file specifications are not checked against its schema. A file specification
    or item that several names share is checked once.

    ISO 32000-1, Table 157 lists only S, D and N as "the types of fields in the
    collection item or collection subitem dictionary". The field data of the
    file-related subtypes (F, Desc, ModDate, CreationDate, Size) "shall be" the
    corresponding entry of the file specification or embedded file parameters,
    so an item value under such a field's key is not field data and is removed.
    """
    removed, seen = 0, set()
    for _, filespec in _embedded_files(pdf):
        if not _first_visit(filespec, seen):
            continue
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
        if not _first_visit(item, seen):
            continue
        removed += _optional_type(item, Name.CollectionItem)
        if schema is None:
            continue
        for key in list(item.keys()):
            if key == "/Type":
                continue
            field = schema.get(key)
            subtype = field.get("/Subtype") if field is not None else None
            repaired = (
                _check_item_value(item, key, subtype)
                if subtype in (Name.S, Name.D, Name.N)
                else None
            )
            removed += _remove_entry(item, key) if repaired is None else repaired
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


def _action_owners(pdf: Pdf) -> list:
    """Return the catalog, then the pages, annotations and outline items, each once.

    An /Annots array or annotation that several pages share is listed once.
    """
    owners, seen = [pdf.Root], set()
    for page in pdf.pages:
        if not _first_visit(page.obj, seen):
            continue
        owners.append(page.obj)
        annots = page.obj["/Annots"] if "/Annots" in page.obj.keys() else None
        if isinstance(annots, Array) and _first_visit(annots, seen):
            owners.extend(
                item
                for item in annots
                if isinstance(item, Dictionary) and _first_visit(item, seen)
            )
    outlines = pdf.Root.get("/Outlines")
    items = [outlines] if isinstance(outlines, Dictionary) else []
    visited = set()
    while items:
        item = items.pop()
        if not isinstance(item, Dictionary) or not _first_visit(item, visited):
            continue
        keys = item.keys()
        if item is not outlines and _first_visit(item, seen):
            owners.append(item)
        items.extend(item[key] for key in ("/First", "/Next") if key in keys)
    return owners


def _walk_actions(pdf: Pdf, remove=None) -> tuple[list, int]:
    """Visit each action of the document once, optionally removing references.

    The catalog /OpenAction and the /A and /AA entries of the catalog, pages,
    annotations and outline items are searched, with the actions that follow
    them (/Next, ISO 32000-1, Table 193). Each indirect action, /AA dictionary and
    /Next array is visited once, so the work stays linear in the size of the
    document however often they are referenced.

    The keys of each dictionary are listed once: with pikepdf, looking up an
    absent key costs far more, which matters for the many annotations of a large
    document.

    Args:
        pdf: The document.
        remove: Called with each action dictionary where it is referenced. A true
            result removes that reference together with the actions that follow
            the removed action, which form its subtree (12.6.2, NOTE 1), as
            remove_actions() removes a non-compliant action. A /AA dictionary
            or /Next array that this leaves empty is removed as well.

    Returns:
        The visited actions, and the number of removed references.
    """
    actions, pending, seen, emptied = [], [], set(), set()
    removed = 0

    def prune(holder, keys: list) -> bool:
        """Remove rejected actions from a container; True if that emptied it."""
        nonlocal removed
        kept = []
        for key in keys:
            value = holder[key]
            if remove is not None and isinstance(value, Dictionary) and remove(value):
                del holder[key]
                removed += 1
            else:
                kept.append(value)
        # Keys are given last first, so that array indices stay valid while
        # deleting; the actions are visited in their original order.
        pending.extend(reversed(kept))
        return len(kept) < len(keys) and len(holder) == 0

    def emptied_container(holder, keys: list) -> bool:
        """Prune a shared /AA or /Next container once; report if it was emptied."""
        if not _first_visit(holder, seen):
            return holder.objgen in emptied
        if not prune(holder, keys):
            return False
        emptied.add(holder.objgen)
        return True

    for index, owner in enumerate(_action_owners(pdf)):
        keys = owner.keys()
        slots = ("/OpenAction", "/A") if index == 0 else ("/A",)
        prune(owner, [key for key in reversed(slots) if key in keys])
        additional = owner["/AA"] if "/AA" in keys else None
        if isinstance(additional, Dictionary) and emptied_container(
            additional, sorted(additional.keys(), reverse=True)
        ):
            del owner["/AA"]
    while pending:
        action = pending.pop()
        if not isinstance(action, Dictionary) or not _first_visit(action, seen):
            continue
        actions.append(action)
        if "/Next" not in action.keys():
            continue
        following = action["/Next"]
        if not isinstance(following, Array):
            prune(action, ["/Next"])
        elif emptied_container(following, list(reversed(range(len(following))))):
            del action["/Next"]
    return actions, removed


def _action_threads(actions: list):
    """Yield the threads that Thread actions name directly by their dictionary.

    ISO 32000-1, 12.6.4.6 lets a Thread action refer to a thread of the same file
    by its dictionary, so the thread need not appear in the optional catalog
    /Threads array.
    """
    for action in actions:
        keys = action.keys()
        if (
            "/S" in keys
            and "/D" in keys
            and action["/S"] == _THREAD
            and isinstance(action["/D"], Dictionary)
        ):
            yield action["/D"]


def _listed_bead_threads(pdf: Pdf):
    """Yield the thread of each bead cycle that a page /B array refers to.

    Only the first bead of a thread has to name it (/T), so each cycle is followed
    until a bead does; no bead, and no /B array that pages share, is followed twice.
    """
    visited = set()
    for page in pdf.pages:
        listed = page.obj.get("/B")
        if not isinstance(listed, Array) or not _first_visit(listed, visited):
            continue
        for bead in listed:
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


def _validate_threads(pdf: Pdf) -> tuple[dict, dict[tuple[int, int], tuple]]:
    """Validate bead cycles and page links without renumbering local Thread actions.

    Every thread of the catalog /Threads array must be valid. A thread that a
    Thread action or a page /B array refers to without being listed there (the
    catalog array is optional, Table 28) supplies its beads when it is valid.
    Each thread is traversed once, however often it is listed or referred to.

    Returns:
        For each valid thread, in /Threads and discovery order: its dictionary
        and the ids of its beads. For each of their beads, in the same order and
        then in reading order: its page and bead dictionary.
    """
    threads = pdf.Root.get("/Threads")
    if threads is not None and not isinstance(threads, Array):
        raise _thread_error("expected an array")
    pages = {page.objgen for page in pdf.pages}
    found, beads = {}, {}

    def add(thread, cycle: list) -> None:
        found[thread.objgen] = (thread, {bead_id for bead_id, _, _ in cycle})
        for bead_id, page_id, bead in cycle:
            # Separate threads cannot share beads, because the first bead of
            # each one names its own thread.
            beads.setdefault(bead_id, (page_id, bead))

    for thread in threads if threads is not None else []:
        # A thread listed again keeps the place of its first occurrence; its
        # cycle is not traversed again.
        if not (isinstance(thread, Dictionary) and thread.objgen in found):
            add(thread, _thread_beads(thread, pages))
    tried = set(found)
    actions, _ = _walk_actions(pdf)
    for thread in chain(_action_threads(actions), _listed_bead_threads(pdf)):
        if thread.objgen in tried:
            continue
        tried.add(thread.objgen)
        try:
            cycle = _thread_beads(thread, pages)
        except ConversionError:
            # The Thread actions that name this thread are removed later.
            continue
        add(thread, cycle)
    return found, beads


def _sanitize_thread_info(threads: list) -> int:
    """Remove the thread information entries that DocInfo could not contain.

    ISO 32000-1, Table 160 gives /I the syntax of the document information
    dictionary (14.3.3). Like the conversion of DocInfo, each malformed entry is
    removed on its own, keeping its valid siblings and the article: a date that
    is not a PDF date, a /Trapped value other than the three names, and a text
    string, standard or custom, whose encoding is invalid. A date lacking only the
    D: prefix gains it. An information dictionary that threads share is checked
    once.
    """
    removed, seen = 0, set()
    for thread in threads:
        info = thread.get("/I")
        if info is None:
            continue
        if not isinstance(info, Dictionary):
            removed += _remove_entry(thread, "/I")
            continue
        if not _first_visit(info, seen):
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
                repaired = _check_date(info, key)
                valid = repaired is not None
                removed += repaired or 0
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

    The position of each bead in a /B array that pages share is found once, so
    the work stays linear in the size of the arrays and the number of beads.
    """
    on_page: dict[tuple[int, int], dict[tuple[int, int], Dictionary]] = {}
    for bead_id, (page_id, bead) in beads.items():
        on_page.setdefault(page_id, {})[bead_id] = bead

    shared: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    repaired = 0
    for page in pdf.pages:
        expected = on_page.get(page.objgen, {})
        listed = page.get("/B")
        if listed is None and not expected:
            continue
        positions, length = {}, None
        if isinstance(listed, Array):
            length = len(listed)
            if listed.objgen in shared:
                positions = shared[listed.objgen]
            else:
                # Direct dictionaries are never beads of a cycle, which requires
                # indirect references; the first occurrence of each bead counts.
                for index, item in enumerate(listed):
                    if isinstance(item, Dictionary) and item.objgen != (0, 0):
                        positions.setdefault(item.objgen, index)
                if listed.objgen != (0, 0):
                    shared[listed.objgen] = positions
        kept = sorted(
            (bead_id for bead_id in expected if bead_id in positions),
            key=positions.__getitem__,
        )
        # An array that lists every expected bead and has no other entry holds
        # exactly those beads, once each, in drawing order.
        if length == len(kept) == len(expected):
            continue
        ordered = [expected[bead_id] for bead_id in kept]
        ordered.extend(
            bead for bead_id, bead in expected.items() if bead_id not in positions
        )
        if ordered:
            page.B = Array(ordered)
        else:
            del page.obj["/B"]
        repaired += 1
        logger.warning("Repaired the beads listed in a page /B array")
    return repaired


def _thread_action_filter(pdf: Pdf, found: dict):
    """Return a check that rejects Thread actions without a valid destination.

    ISO 32000-1, Table 205 names the destination thread (/D) of a Thread action
    by its dictionary in the current file, by its index in the catalog /Threads
    array, or by the title of its thread information dictionary, the first
    thread of the catalog array with that title being used. With a file
    specification (/F) only the dictionary form refers to the current file; an
    index or a title in another file is not checked. An action whose destination
    is missing, of another type, out of range, or not a valid thread is rejected.
    The optional bead (/B), a bead dictionary of that thread or its index in the
    thread, is removed if it is neither, so that the action opens the thread at
    its first bead.

    Returns:
        The check, called with each action dictionary, and a one-element list
        that counts the removed /B entries.
    """
    listed = pdf.Root.get("/Threads")
    listed = [thread.objgen for thread in listed] if listed is not None else []
    titles: dict[str, tuple[int, int]] = {}
    for thread_id in listed:
        info = found[thread_id][0].get("/I")
        title = info.get("/Title") if isinstance(info, Dictionary) else None
        if isinstance(title, String):
            titles.setdefault(str(title), thread_id)
    decided: dict[tuple[int, int], bool] = {}
    beads_removed = [0]

    def invalid(action) -> bool:
        keys = action.keys()
        if "/S" not in keys or action["/S"] != _THREAD:
            return False
        target = action["/D"] if "/D" in keys else None
        if isinstance(target, Dictionary):
            thread_id = target.objgen if target.objgen in found else None
        elif isinstance(target, bool) or not isinstance(target, (int, String)):
            thread_id = None
        elif "/F" in keys:
            return False
        elif isinstance(target, int):
            thread_id = listed[target] if 0 <= target < len(listed) else None
        else:
            thread_id = titles.get(str(target))
        if thread_id is None:
            logger.warning("Removed a Thread action without a valid destination")
            return True
        bead = action["/B"] if "/B" in keys else None
        members = found[thread_id][1]
        if bead is not None and not (
            (isinstance(bead, Dictionary) and bead.objgen in members)
            or (
                isinstance(bead, int)
                and not isinstance(bead, bool)
                and 0 <= bead < len(members)
            )
        ):
            beads_removed[0] += _remove_entry(action, "/B")
        return False

    def check(action) -> bool:
        key = action.objgen
        if key == (0, 0):
            return invalid(action)
        if key not in decided:
            decided[key] = invalid(action)
        return decided[key]

    return check, beads_removed


def _repair_thread_actions(pdf: Pdf, found: dict) -> int:
    """Remove the Thread actions whose destination is not a valid article.

    A malformed thread of the catalog /Threads array rejects the conversion, as
    removing it would renumber the threads that actions name by index. A thread
    that only actions name is not part of that array, so removing a Thread
    action that names it, or a thread that does not exist, changes no other
    destination. The action is removed where it is referenced: from the
    /OpenAction, /A or /AA entry, or from the /Next entry of another action.
    """
    check, beads_removed = _thread_action_filter(pdf, found)
    _, removed = _walk_actions(pdf, check)
    return removed + beads_removed[0]


def sanitize_catalog_navigation(pdf: Pdf) -> int:
    """Preserve portfolios/articles and reject ambiguous repairs of article graphs.

    Returns:
        Number of article structures repaired (an inline /Threads array, a page
        /B array, a Thread action without a valid destination or its invalid
        bead) and malformed thread information and collection entries removed,
        all in place.

    Raises:
        ConversionError: If an article of the catalog /Threads array is malformed.
            Article validation precedes all edits and leaves thread destinations
            unchanged.
    """
    found, beads = _validate_threads(pdf)
    return (
        _make_threads_indirect(pdf)
        + _repair_page_beads(pdf, beads)
        + _sanitize_thread_info([thread for thread, _ in found.values()])
        + _repair_thread_actions(pdf, found)
        + _sanitize_collection(pdf)
    )
