# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate preserved collection settings and circular article navigation."""

import copy
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
_TRAPPED = (Name("/True"), Name("/False"), Name.Unknown)
_FILE_NAME_KEYS = ("/F", "/UF", "/DOS", "/Mac", "/Unix")


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


def _pdf_date(value):
    """Validate a date, adding the D: prefix that ISO 32000-1 requires.

    PDF Reference 1.7, 3.8.3 made the prefix optional ("The prefix D:, although
    also optional, is strongly recommended"), whereas ISO 32000-1, 7.9.4 states
    "The prefix D: shall be present". A date that is valid apart from the prefix
    is therefore kept with the prefix added.

    ISO 32000-1, Table 34 lists a date as "Date (ASCII string)", so a date stored
    in another encoding, such as UTF-16BE with a byte order marker, is rewritten
    as the ASCII string of the same date, as DocInfo dates are written.

    Returns:
        The value itself if it is a canonical date, the repaired date string,
        or None if the value is not a PDF date.
    """
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
        return value
    logger.warning("Rewrote a navigation date as an ASCII string with D:")
    return String(canonical)


def _remove_entry(owner, key: str) -> int:
    """Remove a malformed reference from the owner whose entry it is.

    Only an owner's own reference entry is removed this way (the catalog
    /Collection, a thread /I, a file specification /CI); an object reached
    through a reference is never edited, but replaced by a sanitized copy.
    """
    del owner[key]
    logger.warning("Removed malformed navigation entry %s", key)
    return 1


def _first_visit(item, seen: set) -> bool:
    """Record an indirect object; a direct object has only the owner it is in."""
    key = item.objgen
    if key == (0, 0):
        return True
    if key in seen:
        return False
    seen.add(key)
    return True


# The result of a metadata dictionary that needs no change, while memoized.
_UNCHANGED = object()


def _detached(value):
    """Return a value for a copy that shares no direct object with the original.

    pikepdf inserts a direct dictionary or array into another container by
    reference, so a copy that took it as it is would edit the original along
    with it. Its copy (copy.copy) duplicates the direct objects it contains and
    keeps the references to indirect ones.
    """
    if isinstance(value, (Dictionary, Array)) and value.objgen == (0, 0):
        return copy.copy(value)
    return value


def _new_dictionary(pdf: Pdf, entries: dict, indirect: bool) -> Dictionary:
    """Create the copy of a dictionary from its retained entries."""
    result = Dictionary({key: _detached(value) for key, value in entries.items()})
    return pdf.make_indirect(result) if indirect else result


def _copy_on_write(pdf: Pdf, original, memo: dict, context, sanitize):
    """Sanitize a referenced dictionary without ever editing it.

    Thread information (/I), the collection, its schema, fields and sorting,
    collection items (/CI) and subitems are referred to by reference, and a
    malformed file may make that reference any other object, such as a page,
    its resources, a font, an annotation's appearance, a bead or the catalog
    /Names dictionary.
    So the original is never changed: a dictionary that needs no change is kept
    as it is, and otherwise a new dictionary of the retained entries, indirect if
    the original is, replaces it at the reference. The original stays unchanged
    in all of its other roles and is not written if nothing refers to it anymore.

    Each indirect dictionary is sanitized once per role (each role has its own
    memo) and context (a field type), and its references share the result, so
    the work stays linear in the size of the document and shared metadata stays
    shared.

    Args:
        pdf: The document.
        original: The referenced dictionary.
        memo: The results of this role, by object id and context.
        context: What the result depends on besides the dictionary.
        sanitize: Called with the dictionary; returns the entries to retain and
            whether they differ from its own, or None to remove the reference.

    Returns:
        The original itself if it is kept, its sanitized copy, or None.
    """
    key = (original.objgen, context)
    indirect = original.objgen != (0, 0)
    if indirect and key in memo:
        result = memo[key]
    else:
        checked = sanitize(original)
        if checked is None:
            result = None
        elif not checked[1]:
            result = _UNCHANGED
        else:
            result = _new_dictionary(pdf, checked[0], indirect)
        if indirect:
            memo[key] = result
    return original if result is _UNCHANGED else result


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


def _field_data(value, subtype):
    """Check collection item or subitem data against the type of its field.

    A text string needs a valid PDFDocEncoding or UTF-16BE encoding (7.9.2.2),
    as retained DocInfo values do.

    Returns:
        The value itself if it is valid, a repaired date, or None.
    """
    if subtype == Name.S:
        return value if _is_pdf_text_string(value) else None
    if subtype == Name.D:
        return _pdf_date(value)
    valid = (
        subtype == Name.N
        and not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(value)
    )
    return value if valid else None


def _dropped(key: str) -> int:
    """Count one malformed entry that a sanitized copy does not retain."""
    logger.warning("Removed malformed navigation entry %s", key)
    return 1


def _sanitize_collection_items(pdf: Pdf, schema) -> int:
    """Remove malformed item entries, and values that the schema does not describe.

    Without a schema there is nothing to compare the values with. PDF/A does not
    require their removal, as veraPDF confirms, so a portfolio keeps this data.
    A schema left with no fields (Table 156 makes each of them optional) describes
    no values, so its items keep only their /Type. Only files of the EmbeddedFiles
    name tree belong to the collection (ISO 32000-1, 12.3.5), so the items of other
    file specifications are not checked against its schema. A file specification,
    item or subitem that several references share is checked once.

    ISO 32000-1, Table 157 lists only S, D and N as "the types of fields in the
    collection item or collection subitem dictionary". The field data of the
    file-related subtypes (F, Desc, ModDate, CreationDate, Size) "shall be" the
    corresponding entry of the file specification or embedded file parameters,
    so an item value under such a field's key is not field data and is removed.

    ISO 32000-1, Table 49 (and PDF Reference 1.7, Table 3.43) defines every entry
    of a collection subitem dictionary, /D included, as optional with no default,
    so a subitem with only a /P prefix, or none at all, is well formed. A subitem
    whose /Type, /P or /D contradicts the field is removed as one item value.

    Items and subitems are sanitized by copy on write (_copy_on_write), so an
    item that is also another object, such as the page resources or the DocInfo
    dictionary, keeps that object unchanged.
    """
    removed, seen_specs, items, subitems = 0, set(), {}, {}

    def subitem(original, subtype):
        nonlocal removed
        if original.get("/Type") not in (None, Name.CollectionSubitem):
            return None
        prefix = original.get("/P")
        if prefix is not None and not _is_pdf_text_string(prefix):
            return None
        data = original.get("/D")
        if data is None:
            return {}, False
        checked = _field_data(data, subtype)
        if checked is None:
            return None
        if checked is data:
            return {}, False
        removed += 1
        entries = {key: original[key] for key in original.keys()}
        entries["/D"] = checked
        return entries, True

    def item(original):
        nonlocal removed
        entries, changed = {}, False
        for key in original.keys():
            value = original[key]
            if key == "/Type":
                kept = value if value == Name.CollectionItem else None
            elif schema is None:
                kept = value
            else:
                field = schema.get(key)
                subtype = field.get("/Subtype") if field is not None else None
                if subtype not in (Name.S, Name.D, Name.N):
                    kept = None
                elif isinstance(value, Dictionary):
                    kept = _copy_on_write(
                        pdf,
                        value,
                        subitems,
                        subtype,
                        lambda found, subtype=subtype: subitem(found, subtype),
                    )
                else:
                    kept = _field_data(value, subtype)
                    # A repaired date; a subitem counts its own repair once.
                    if kept is not None and kept is not value:
                        removed += 1
            if kept is None:
                removed += _dropped(key)
                changed = True
            else:
                changed = changed or kept is not value
                entries[key] = kept
        return entries, changed

    for _, filespec in _embedded_files(pdf):
        if not _first_visit(filespec, seen_specs):
            continue
        original = filespec.get("/CI")
        if original is None:
            continue
        # A file specification, here also one outside the name tree that is
        # typed as such (Table 44), is not a collection item (Table 155).
        checked = (
            _copy_on_write(pdf, original, items, None, item)
            if isinstance(original, Dictionary)
            and original.get("/Type") != Name.Filespec
            else None
        )
        if checked is None:
            removed += _remove_entry(filespec, "/CI")
            continue
        # A file specification refers to its collection item indirectly.
        if checked.objgen == (0, 0):
            checked = pdf.make_indirect(checked)
        if checked is not original:
            filespec["/CI"] = checked
    return removed


def _dangling_default(pdf: Pdf, collection) -> bool:
    """Check whether the initial document (/D) names no embedded file."""
    default = collection.get("/D")
    if default is None:
        return False
    return not (
        isinstance(default, String)
        and any(
            isinstance(name, String) and bytes(name) == bytes(default)
            for name, _ in _embedded_files(pdf)
        )
    )


def _replace_collection(pdf: Pdf, collection, entries: dict) -> None:
    """Point the catalog to a copy of the collection with the given entries.

    The collection dictionary is reached through the catalog reference and may
    also be another object of a malformed file, so it is never edited itself.
    """
    pdf.Root.Collection = _new_dictionary(pdf, entries, collection.objgen != (0, 0))


def remove_dangling_collection_default(pdf: Pdf) -> int:
    """Remove the initial document of a portfolio unless it names an embedded file.

    The sanitizer pipeline repeats this check after every embedded-file cleanup,
    which can remove the file that /D names. The catalog then refers to a copy
    of the collection without /D, leaving the original dictionary unchanged.
    """
    collection = pdf.Root.get("/Collection")
    if not isinstance(collection, Dictionary) or not _dangling_default(pdf, collection):
        return 0
    entries = {key: collection[key] for key in collection.keys() if key != "/D"}
    _replace_collection(pdf, collection, entries)
    return _dropped("/D")


_FIELD_SUBTYPES = (
    Name.S,
    Name.D,
    Name.N,
    Name.F,
    Name.Desc,
    Name.ModDate,
    Name.CreationDate,
    Name.Size,
)


def _sanitized_entries(original, check) -> tuple[dict, bool, int]:
    """Collect the entries of a dictionary that a check retains or replaces.

    Args:
        original: The dictionary, which is not changed.
        check: Called with each key and value; returns the value to retain,
            which may be a repaired one, or None to omit the entry.

    Returns:
        The retained entries, whether they differ from the original ones, and
        the number of omitted entries.
    """
    entries, changed, removed = {}, False, 0
    for key in original.keys():
        value = original.get(key)
        # A null value is an absent entry (7.3.7), which a copy omits.
        if value is None:
            continue
        kept = check(key, value)
        if kept is None:
            removed += _dropped(key)
            changed = True
        else:
            entries[key] = kept
            changed = changed or kept is not value
    return entries, changed, removed


def _sanitize_schema(pdf: Pdf, schema) -> tuple:
    """Sanitize a collection schema and its fields by copy on write.

    A field is removed unless it has a known subtype (Table 157) and a valid
    text string name; its optional entries are repaired. The schema and each
    field dictionary may also be another object of a malformed file, such as
    the catalog /Names dictionary or the page resources, so they are never
    edited: a sanitized copy replaces them where they need a repair
    (_copy_on_write), shared by every reference to the same field.

    Returns:
        The schema itself or its sanitized copy, and the number of repairs.
    """
    removed, fields = 0, {}

    def field_value(key, value):
        if key == "/Type":
            return value if value == Name.CollectionField else None
        if key in ("/V", "/E"):
            return value if isinstance(value, bool) else None
        if key == "/O":
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool)
                else None
            )
        return value

    def field_entries(field):
        nonlocal removed
        entries, changed, count = _sanitized_entries(field, field_value)
        removed += count
        return entries, changed

    def schema_value(key, value):
        if key == "/Type":
            return value if value == Name.CollectionSchema else None
        if (
            not isinstance(value, Dictionary)
            or value.get("/Subtype") not in _FIELD_SUBTYPES
            or not _is_pdf_text_string(value.get("/N"))
        ):
            return None
        return _copy_on_write(pdf, value, fields, None, field_entries)

    def schema_entries(original):
        nonlocal removed
        entries, changed, count = _sanitized_entries(original, schema_value)
        removed += count
        return entries, changed

    return _copy_on_write(pdf, schema, {}, None, schema_entries), removed


def _sanitize_sort(pdf: Pdf, sorting, schema) -> tuple:
    """Sanitize collection sorting by copy on write.

    Like the schema, the sort dictionary may be another object as well, such as
    a schema field, so it is never edited (_copy_on_write).

    Returns:
        The sort dictionary itself, its sanitized copy, or None if the sorting
        names no field of the schema, and the number of repairs.
    """
    removed = 0
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

    def sort_value(key, value):
        if key == "/Type":
            return value if value == Name.CollectionSort else None
        # ISO 32000-1 Table 158 explicitly permits unequal /S and /A lengths.
        # Each element of an /A array refers to the entry at the same index of
        # an /S array, so a single /S name takes a single boolean; without /A
        # it is sorted in ascending order.
        if key == "/A" and not (
            isinstance(value, bool)
            or not scalar
            and isinstance(value, Array)
            and all(isinstance(item, bool) for item in value)
        ):
            return None
        return value

    def sort_entries(original):
        nonlocal removed
        entries, changed, count = _sanitized_entries(original, sort_value)
        removed += count
        # Invalid sorting is removed together with its own malformed entries.
        return (entries, changed) if valid else None

    return _copy_on_write(pdf, sorting, {}, None, sort_entries), removed


def _sanitize_collection(pdf: Pdf) -> int:
    """Repair optional portfolio settings while preserving valid fields and sorting.

    The collection dictionary, its schema, fields and sorting are sanitized by
    copy on write: the catalog refers to a sanitized copy of the collection
    where it needs a repair, which refers to sanitized copies of its schema,
    fields and sorting, so none of them is edited where it is also another
    object of the document.
    """
    collection = pdf.Root.get("/Collection")
    if collection is None:
        return 0
    if not isinstance(collection, Dictionary):
        return _remove_entry(pdf.Root, "/Collection")
    removed, changed = 0, False
    entries = {key: collection[key] for key in collection.keys()}

    def drop(key: str) -> None:
        nonlocal removed, changed
        del entries[key]
        removed += _dropped(key)
        changed = True

    if entries.get("/Type") not in (None, Name.Collection):
        drop("/Type")
    if entries.get("/View") not in (None, Name.D, Name.T, Name.H):
        drop("/View")
    if _dangling_default(pdf, collection):
        drop("/D")
    schema = entries.get("/Schema")
    if schema is not None and not isinstance(schema, Dictionary):
        drop("/Schema")
        schema = None
    if schema is not None:
        checked, count = _sanitize_schema(pdf, schema)
        removed += count
        if checked is not schema:
            entries["/Schema"] = schema = checked
            changed = True
    removed += _sanitize_collection_items(pdf, schema)
    sorting = entries.get("/Sort")
    if sorting is not None:
        checked, count = (
            _sanitize_sort(pdf, sorting, schema)
            if isinstance(sorting, Dictionary)
            else (None, 0)
        )
        removed += count
        if checked is None:
            drop("/Sort")
        elif checked is not sorting:
            entries["/Sort"] = checked
            changed = True
    if changed:
        _replace_collection(pdf, collection, entries)
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


class _Holder:
    """A scanned dictionary or array whose entries may refer to actions.

    Attributes:
        obj: The original object, which is never edited unless it is an owner.
        node: The id of the action this is (its own /Next entry), or None.
        entries: Each key or index, its value, the id of the action it refers
            to (None for another value), and whether that reference is removed.
        omitted: Keys that a copy of this action leaves out (an invalid /B).
        next_id: The id of the /Next array of this action, if it has one.
    """

    __slots__ = ("obj", "node", "entries", "omitted", "next_id")

    def __init__(self, obj, node=None, omitted=()):
        self.obj, self.node, self.omitted = obj, node, omitted
        self.entries, self.next_id = [], None


def _walk_actions(pdf: Pdf, decide=None) -> tuple[list, int]:
    """Visit each action of the document once, optionally removing references.

    The catalog /OpenAction and the /A and /AA entries of the catalog, pages,
    annotations and outline items are searched, with the actions that follow
    them (/Next, ISO 32000-1, Table 193). Each indirect action, /AA dictionary and
    /Next array is visited once, so the work stays linear in the size of the
    document however often they are referenced.

    The keys of each dictionary are listed once: with pikepdf, looking up an
    absent key costs far more, which matters for the many annotations of a large
    document.

    Repairs are made by copy on write. An action, /AA dictionary or /Next array
    is reached through a reference, and a malformed file may make it any other
    object as well, such as a page, an annotation's /Kids or the catalog
    /Threads array, so it is never edited: where it, or an action that follows
    it, needs a repair, a copy (indirect if the original is) takes its place at
    each reference, shared by all of them, and the original stays unchanged in
    its other roles. Only the reference itself is edited in place: the
    /OpenAction, /A and /AA entries of their owners (the catalog, pages,
    annotations and outline items), whose role these entries are.

    Args:
        pdf: The document.
        decide: Called with each action dictionary where it is referenced;
            returns whether to remove that reference, and the keys that the
            kept action has to leave out. A removed reference takes the actions
            that follow the removed action, which form its subtree (12.6.2,
            NOTE 1), as remove_actions() removes a non-compliant action. A /AA
            dictionary or /Next array that this leaves empty is removed as well.

    Returns:
        The visited actions, and the number of removed references.
    """
    # A dictionary scanned as an /AA dictionary is still followed as an action.
    actions, pending, holders, owners = [], [], {}, []
    seen_actions, removed, direct = {}, 0, 0

    def holder_id(obj, kind: str):
        """Identify an indirect object by itself, and a direct one by its visit.

        A direct object has only the one container it is in, so it is visited
        once for each visit of that container.
        """
        nonlocal direct
        if obj.objgen != (0, 0):
            return (kind, obj.objgen)
        direct += 1
        return (kind, direct)

    def scan(holder: _Holder, keys) -> None:
        """Record the references of a container; kept actions are followed."""
        nonlocal removed
        for key in keys:
            value = holder.obj[key]
            if not isinstance(value, Dictionary):
                holder.entries.append((key, value, None, False))
                continue
            if decide is None:
                drop, omitted = False, ()
            else:
                drop, omitted = decide(value)
            if drop:
                removed += 1
                holder.entries.append((key, value, None, True))
                continue
            if value.objgen in seen_actions:
                node = seen_actions[value.objgen]
            else:
                node = holder_id(value, "action")
                if value.objgen != (0, 0):
                    seen_actions[value.objgen] = node
                pending.append((node, value, omitted))
            holder.entries.append((key, value, node, False))

    def container(obj, keys) -> tuple:
        """Scan a shared /AA dictionary or /Next array once."""
        kind = "array" if isinstance(obj, Array) else "aa"
        key = holder_id(obj, kind)
        if key not in holders:
            holders[key] = holder = _Holder(obj)
            scan(holder, keys)
        return key

    for index, owner in enumerate(_action_owners(pdf)):
        keys = owner.keys()
        slots = ("/OpenAction", "/A") if index == 0 else ("/A",)
        holder = _Holder(owner)
        scan(holder, [key for key in slots if key in keys])
        additional = owner["/AA"] if "/AA" in keys else None
        aa_id = (
            container(additional, sorted(additional.keys()))
            if isinstance(additional, Dictionary)
            else None
        )
        owners.append((holder, aa_id))
    # The actions are visited in their order: the stack is used last first.
    pending.reverse()
    while pending:
        node, action, omitted = pending.pop()
        actions.append(action)
        holder = holders[node] = _Holder(action, node, omitted)
        keys = action.keys()
        if "/Next" not in keys:
            continue
        following = action["/Next"]
        mark = len(pending)
        if isinstance(following, Array):
            holder.next_id = container(following, range(len(following)))
        else:
            scan(holder, ["/Next"])
        pending[mark:] = reversed(pending[mark:])
    if decide is not None:
        _rewrite_actions(pdf, holders, owners)
    return actions, removed


def _rewrite_actions(pdf: Pdf, holders: dict, owners: list) -> None:
    """Replace the actions and containers that need a repair by copies.

    An action needs a copy if it leaves out an entry, or if its /Next entry
    refers to a removed action or to an action or array that needs a copy; a
    /AA dictionary or /Next array needs one if one of its entries does. This is
    propagated backwards from the repairs, so actions that follow each other in
    a cycle are handled as well. A copy of an indirect object is created first,
    and filled in afterwards, so that copies can refer to each other.
    """
    parents: dict = {}
    dirty, queue = set(), []
    for key, holder in holders.items():
        children = [node for _, _, node, _ in holder.entries if node is not None]
        if holder.next_id is not None:
            children.append(holder.next_id)
        for child in children:
            parents.setdefault(child, []).append(key)
        if holder.omitted or any(drop for *_, drop in holder.entries):
            dirty.add(key)
            queue.append(key)
    while queue:
        for parent in parents.get(queue.pop(), ()):
            if parent not in dirty:
                dirty.add(parent)
                queue.append(parent)

    def emptied(holder: _Holder) -> bool:
        """A container whose every entry is removed is removed with them."""
        return (
            holder.node is None
            and bool(holder.entries)
            and all(drop for *_, drop in holder.entries)
        )

    results = {}
    for key in dirty:
        holder = holders[key]
        if emptied(holder):
            results[key] = None
        elif holder.obj.objgen != (0, 0):
            empty = Array() if isinstance(holder.obj, Array) else Dictionary()
            results[key] = pdf.make_indirect(empty)

    def value_of(entry):
        _, value, node, _ = entry
        return results[node] if node in dirty else value

    # A direct object is visited after its container and before the objects it
    # contains, so in reverse order each one is built after its contents.
    order = {key: index for index, key in enumerate(holders)}
    for key in sorted(dirty, key=order.__getitem__, reverse=True):
        holder = holders[key]
        if results.get(key, False) is None:
            continue
        kept = [entry for entry in holder.entries if not entry[3]]
        if isinstance(holder.obj, Array):
            items = [_detached(value_of(entry)) for entry in kept]
            result = results.get(key)
            if result is None:
                results[key] = Array(items)
            else:
                result.extend(items)
            continue
        # An /AA dictionary consists of its scanned entries; an action keeps
        # every entry but the omitted ones, and its /Next entry is rebuilt.
        entries = {}
        if holder.node is not None:
            for name in holder.obj.keys():
                value = holder.obj.get(name)
                if value is not None and name not in holder.omitted:
                    entries[name] = value
            entries.pop("/Next", None)
        entries.update((entry[0], value_of(entry)) for entry in kept)
        if holder.next_id is not None:
            if holder.next_id not in dirty:
                entries["/Next"] = holder.obj["/Next"]
            elif results[holder.next_id] is not None:
                entries["/Next"] = results[holder.next_id]
        result = results.get(key)
        if result is None:
            results[key] = _new_dictionary(pdf, entries, False)
        else:
            for name, value in entries.items():
                result[name] = _detached(value)

    # The owners hold the references; their /A, /OpenAction and /AA entries
    # are their own and are replaced in place.
    for holder, aa_id in owners:
        for name, _, node, drop in holder.entries:
            if drop:
                del holder.obj[name]
            elif node in dirty:
                holder.obj[name] = results[node]
        if aa_id in dirty:
            if results[aa_id] is None:
                del holder.obj["/AA"]
            else:
                holder.obj["/AA"] = results[aa_id]


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


def _sanitize_thread_info(pdf: Pdf, threads: list) -> int:
    """Remove the thread information entries that DocInfo could not contain.

    ISO 32000-1, Table 160 gives /I the syntax of the document information
    dictionary (14.3.3). Like the conversion of DocInfo, each malformed entry is
    removed on its own, keeping its valid siblings and the article: a date that
    is not a PDF date, a /Trapped value other than the three names, and a text
    string, standard or custom, whose encoding is invalid. A date lacking only the
    D: prefix gains it. An information dictionary that threads share is checked
    once, and they share its sanitized copy.

    The dictionary is sanitized by copy on write (_copy_on_write), so an /I
    entry that refers to another object, such as a bead, the page resources, a
    font or the DocInfo dictionary itself, leaves that object unchanged.
    """
    removed, memo = 0, {}

    def sanitize(info):
        nonlocal removed
        entries, changed = {}, False
        for key in info.keys():
            value = info.get(key)
            if value is None:
                continue
            if key == "/Trapped":
                kept = value if isinstance(value, Name) and value in _TRAPPED else None
            elif key in ("/CreationDate", "/ModDate"):
                kept = _pdf_date(value)
            else:
                kept = value if _is_pdf_text_string(value) else None
            if kept is None:
                removed += _dropped(key)
                changed = True
            else:
                entries[key] = kept
                if kept is not value:
                    removed += 1
                    changed = True
        return entries, changed

    for thread in threads:
        info = thread.get("/I")
        if info is None:
            continue
        checked = (
            _copy_on_write(pdf, info, memo, None, sanitize)
            if isinstance(info, Dictionary)
            else None
        )
        if checked is None:
            removed += _remove_entry(thread, "/I")
        elif checked is not info:
            thread.I = checked
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


def _is_file_reference(value) -> bool:
    """Check that a value is a file specification naming a file (7.11).

    A file specification is a string or a dictionary (7.11.1). The dictionary
    names the file by /F, /UF or the deprecated /DOS, /Mac and /Unix strings, at
    least one of which is required without embedded file streams (Table 44);
    its optional /Type is /Filespec.
    """
    if isinstance(value, String):
        return len(bytes(value)) > 0
    if not isinstance(value, Dictionary):
        return False
    keys = value.keys()
    if "/Type" in keys and value["/Type"] != Name.Filespec:
        return False
    return any(
        key in keys and isinstance(value[key], String) and len(bytes(value[key])) > 0
        for key in _FILE_NAME_KEYS
    )


def _thread_action_filter(pdf: Pdf, found: dict):
    """Return a check that rejects Thread actions without a valid destination.

    ISO 32000-1, Table 205 names the destination thread (/D) of a Thread action
    by its dictionary in the current file, by its index in the catalog /Threads
    array, or by the title of its thread information dictionary, the first
    thread of the catalog array with that title being used. With a file
    specification (/F) only the dictionary form refers to the current file; an
    index or a title in another file is not looked up. A null /F is an absent
    one, and an action whose /F is not a file specification, or whose destination
    is missing, of another type, a negative index (in another file as well), a
    title that is not a valid text string, out of range, or not a valid thread,
    is rejected.
    The optional bead (/B), a bead dictionary of that thread or its index in the
    thread, is removed if it is neither, so that the action opens the thread at
    its first bead. In another file the bead can only be an index (a bead
    dictionary requires the thread to be in the current file), which must not be
    negative; its upper bound is not checked. The bead is left out of a copy of
    the action (_walk_actions), never removed from the action itself, which may
    also be another object, such as a page and its own /B array.

    Returns:
        The check, called with each action dictionary, which returns whether to
        remove the action and the keys its copy leaves out, and a one-element
        list that counts the invalid /B entries.
    """
    listed = pdf.Root.get("/Threads")
    listed = [thread.objgen for thread in listed] if listed is not None else []
    titles: dict[str, tuple[int, int]] = {}
    for thread_id in listed:
        info = found[thread_id][0].get("/I")
        title = info.get("/Title") if isinstance(info, Dictionary) else None
        if isinstance(title, String):
            titles.setdefault(str(title), thread_id)
    decided: dict[tuple[int, int], tuple[bool, tuple]] = {}
    beads_removed = [0]
    keep, without_bead = (False, ()), (False, ("/B",))

    def omit_bead() -> tuple[bool, tuple]:
        beads_removed[0] += _dropped("/B")
        return without_bead

    def invalid(action) -> tuple[bool, tuple]:
        keys = action.keys()
        if "/S" not in keys or action["/S"] != _THREAD:
            return keep
        target = action["/D"] if "/D" in keys else None
        file = action["/F"] if "/F" in keys else None
        if file is not None and not _is_file_reference(file):
            logger.warning("Removed a Thread action with an invalid file specification")
            return True, ()
        if isinstance(target, Dictionary):
            thread_id = target.objgen if target.objgen in found else None
        elif (
            isinstance(target, bool)
            or not isinstance(target, (int, String))
            or (isinstance(target, int) and target < 0)
            or (isinstance(target, String) and not _is_pdf_text_string(target))
        ):
            # An index is not negative and a title is a text string (Table
            # 205), here or in another file.
            thread_id = None
        elif file is not None:
            # The bead of a thread in another file can only be its index; the
            # dictionary form requires the thread to be in the current file.
            bead = action["/B"] if "/B" in keys else None
            if bead is not None and (
                isinstance(bead, bool) or not isinstance(bead, int) or bead < 0
            ):
                return omit_bead()
            return keep
        elif isinstance(target, int):
            thread_id = listed[target] if 0 <= target < len(listed) else None
        else:
            thread_id = titles.get(str(target))
        if thread_id is None:
            logger.warning("Removed a Thread action without a valid destination")
            return True, ()
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
            return omit_bead()
        return keep

    def check(action) -> tuple[bool, tuple]:
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
    /OpenAction, /A or /AA entry, or from the /Next entry of another action,
    which is replaced by a copy without it (_walk_actions).
    """
    check, beads_removed = _thread_action_filter(pdf, found)
    _, removed = _walk_actions(pdf, check)
    return removed + beads_removed[0]


def sanitize_catalog_navigation(pdf: Pdf) -> int:
    """Preserve portfolios/articles and reject ambiguous repairs of article graphs.

    Returns:
        Number of article structures repaired (an inline /Threads array, a page
        /B array, a Thread action without a valid destination or its invalid
        bead) and malformed thread information and collection entries removed.
        No object reached through a reference is edited: thread information,
        collection items and subitems, the collection, its schema, fields and
        sorting, actions, /AA dictionaries and /Next arrays are replaced by
        sanitized copies where they need a repair, and a page /B array by a new
        one, so that no other object they may also be is changed. Only the
        entries that hold these references are set or removed, on their owners:
        the catalog (/Collection, /OpenAction, /A, /AA, /Threads), pages (/B,
        /A, /AA), annotations and outline items (/A, /AA), threads (/I) and
        embedded file specifications (/CI).

    Raises:
        ConversionError: If an article of the catalog /Threads array is malformed.
            Article validation precedes all edits and leaves thread destinations
            unchanged.
    """
    found, beads = _validate_threads(pdf)
    return (
        _make_threads_indirect(pdf)
        + _repair_page_beads(pdf, beads)
        + _sanitize_thread_info(pdf, [thread for thread, _ in found.values()])
        + _repair_thread_actions(pdf, found)
        + _sanitize_collection(pdf)
    )
