# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Apply deterministic WCAG 2.1 PDF accessibility requirements."""

import logging
import re

import pikepdf
from pikepdf import Array, Dictionary, Name, NumberTree, Pdf, String

from .accessibility import AccessibilityStrings, accessibility_strings, primary_language
from .tagging import _effective_structure_role, _number_tree_keys
from .utils import resolve_indirect

logger = logging.getLogger(__name__)

_FIELD_REQUIRED = 1 << 1
_MAX_STRING_BYTES = 32_767
_GENERIC_WIDGET_TOOLTIPS = frozenset({"Form field", "Formularfeld"})
_PAGE_LABEL_STYLES = frozenset({"/D", "/R", "/r", "/A", "/a"})


def prepare_pdfua_document(pdf: Pdf) -> dict[str, int]:
    """Keep printer-mark annotations as untagged incidental artifacts."""
    printer_marks_preserved = 0
    for page in pdf.pages:
        annotations = resolve_indirect(page.obj.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        for value in annotations:
            annotation = resolve_indirect(value)
            if isinstance(annotation, Dictionary) and (
                resolve_indirect(annotation.get("/Subtype")) == Name.PrinterMark
            ):
                printer_marks_preserved += 1
                for key in ("/StructParent", "/StructParents"):
                    if key in annotation:
                        del annotation[key]
    return {"printer_mark_annotations_preserved": printer_marks_preserved}


def _field_entry(widget: Dictionary, key: str) -> tuple[Dictionary, object] | None:
    """Return an inheritable field entry without following a cyclic field tree."""
    current: object = widget
    visited: set[tuple[int, int] | int] = set()
    while isinstance(current := resolve_indirect(current), Dictionary):
        objgen = current.objgen
        identity: tuple[int, int] | int = objgen if objgen != (0, 0) else id(current)
        if identity in visited:
            return None
        visited.add(identity)
        if key in current:
            return current, resolve_indirect(current.get(key))
        current = current.get("/Parent")
    return None


def _required_tooltip(label: str, suffix: str) -> String:
    """Append the required suffix without exceeding the PDF string byte limit."""
    result = String(f"{label}{suffix}")
    if len(bytes(result)) <= _MAX_STRING_BYTES:
        return result

    low = 0
    high = len(label)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = String(f"{label[:middle]}{suffix}")
        if len(bytes(candidate)) <= _MAX_STRING_BYTES:
            low = middle
        else:
            high = middle - 1
    return String(f"{label[:low]}{suffix}")


def _bounded_string(value: str) -> String:
    """Return a non-empty PDF string within the encoded string limit."""
    result = String(value)
    if len(bytes(result)) <= _MAX_STRING_BYTES:
        return result
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(bytes(String(value[:middle]))) <= _MAX_STRING_BYTES:
            low = middle
        else:
            high = middle - 1
    return String(value[:low])


def _localize_generic_widget_tooltip(
    widget: Dictionary,
    strings: AccessibilityStrings,
) -> bool:
    tooltip_entry = _field_entry(widget, "/TU")
    if tooltip_entry is None:
        return False
    owner, value = tooltip_entry
    if (
        not isinstance(value, String)
        or str(value).strip() not in _GENERIC_WIDGET_TOOLTIPS
    ):
        return False
    tooltip = String(strings.form_field)
    owner["/TU"] = tooltip
    if "/TU" in widget:
        widget["/TU"] = tooltip
    return True


def _indicate_required_control(
    widget: Dictionary,
    strings: AccessibilityStrings,
) -> bool:
    """Include inherited required status in a trustworthy accessible label."""
    flags_entry = _field_entry(widget, "/Ff")
    flags = flags_entry[1] if flags_entry is not None else None
    try:
        required = bool(int(flags) & _FIELD_REQUIRED) if flags is not None else False
    except (TypeError, ValueError, OverflowError):
        return False
    if not required:
        return False

    tooltip_entry = _field_entry(widget, "/TU")
    field_name_entry = _field_entry(widget, "/T")
    label_entry: tuple[Dictionary, str] | None = None
    for entry in (tooltip_entry, field_name_entry):
        if entry is None:
            continue
        owner, value = entry
        if not isinstance(value, String):
            continue
        label = str(value).strip()
        if label and label not in _GENERIC_WIDGET_TOOLTIPS:
            label_entry = owner, label
            break
    if label_entry is None:
        return False
    label_owner, label = label_entry
    if any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", label, re.IGNORECASE)
        for term in strings.required_terms
    ):
        return False
    if field_name_entry is not None:
        field_owner = field_name_entry[0]
    elif flags_entry is not None:
        field_owner = flags_entry[0]
    else:
        field_owner = label_owner
    tooltip = _required_tooltip(label, strings.required_suffix)
    field_owner["/TU"] = tooltip
    if "/TU" in widget:
        widget["/TU"] = tooltip
    return True


def _valid_page_labels(pdf: Pdf) -> bool:
    root = resolve_indirect(pdf.Root.get("/PageLabels"))
    if not isinstance(root, Dictionary):
        return False
    if _number_tree_keys(root) is None:
        return False
    try:
        entries = list(NumberTree(root).items())
    except (TypeError, ValueError, RuntimeError, pikepdf.PdfError):
        return False
    if not entries or entries[0][0] != 0:
        return False
    previous = -1
    for page_index, raw_label in entries:
        if (
            isinstance(page_index, bool)
            or not isinstance(page_index, int)
            or page_index <= previous
            or not 0 <= page_index < len(pdf.pages)
        ):
            return False
        previous = page_index
        label = resolve_indirect(raw_label)
        if not isinstance(label, Dictionary):
            return False
        style = resolve_indirect(label.get("/S"))
        prefix = resolve_indirect(label.get("/P"))
        if style is not None and str(style) not in _PAGE_LABEL_STYLES:
            return False
        if prefix is not None and not isinstance(prefix, String):
            return False
        if style is None and prefix is None:
            return False
        start = resolve_indirect(label.get("/St"))
        if start is not None and (
            isinstance(start, bool) or not isinstance(start, int) or start < 1
        ):
            return False
    return True


def _object_identity(value: object) -> tuple[int, int] | int:
    resolved = resolve_indirect(value)
    objgen = getattr(resolved, "objgen", (0, 0))
    return objgen if objgen != (0, 0) else id(resolved)


def _structure_children(element: Dictionary) -> list[Dictionary]:
    raw_children = resolve_indirect(element.get("/K"))
    values = list(raw_children) if isinstance(raw_children, Array) else [raw_children]
    children = []
    for value in values:
        child = resolve_indirect(value)
        if isinstance(child, Dictionary) and "/S" in child:
            children.append(child)
    return children


def _content_page_reference(element: Dictionary) -> object | None:
    """Return the first page referenced by a nested MCR or OBJR dictionary."""
    pending = [resolve_indirect(element.get("/K"))]
    visited: set[tuple[int, int] | int] = set()
    while pending:
        value = resolve_indirect(pending.pop())
        if isinstance(value, Array):
            pending.extend(reversed(value))
            continue
        if not isinstance(value, Dictionary):
            continue
        identity = _object_identity(value)
        if identity in visited:
            continue
        visited.add(identity)
        if resolve_indirect(value.get("/Type")) in {Name.MCR, Name.OBJR}:
            page = resolve_indirect(value.get("/Pg"))
            if page is not None:
                return page
        child = resolve_indirect(value.get("/K"))
        if child is not None:
            pending.append(child)
    return None


def _heading_level(role: object) -> int | None:
    match = re.fullmatch(r"/H([1-6])", str(resolve_indirect(role)))
    if match:
        return int(match.group(1))
    return 1 if str(resolve_indirect(role)) == "/H" else None


def _bookmark_title(element: Dictionary) -> str | None:
    for key in ("/T", "/ActualText", "/Alt"):
        value = resolve_indirect(element.get(key))
        if not isinstance(value, String):
            continue
        title = " ".join(str(value).split())
        if title and not any(
            0xE000 <= ord(character) <= 0xF8FF
            or 0xF0000 <= ord(character) <= 0xFFFFD
            or 0x100000 <= ord(character) <= 0x10FFFD
            for character in title
        ):
            return title
    return None


def _heading_bookmarks(
    pdf: Pdf,
) -> list[tuple[int, str, Dictionary]]:
    structure_root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    if not isinstance(structure_root, Dictionary):
        return []
    role_map = resolve_indirect(structure_root.get("/RoleMap"))
    if not isinstance(role_map, Dictionary):
        role_map = None
    page_objects = {_object_identity(page.obj): page.obj for page in pdf.pages}
    bookmarks = []
    raw_root = resolve_indirect(structure_root.get("/K"))
    root_elements = list(raw_root) if isinstance(raw_root, Array) else [raw_root]
    stack: list[tuple[object, object | None]] = [
        (element, None) for element in reversed(root_elements)
    ]
    visited: set[tuple[int, int] | int] = set()
    while stack:
        raw_element, inherited_page = stack.pop()
        element = resolve_indirect(raw_element)
        if not isinstance(element, Dictionary) or "/S" not in element:
            continue
        identity = _object_identity(element)
        if identity in visited:
            continue
        visited.add(identity)
        page_reference = resolve_indirect(element.get("/Pg"))
        if page_reference is None:
            page_reference = inherited_page
        level = _heading_level(_effective_structure_role(element.get("/S"), role_map))
        title = _bookmark_title(element) if level is not None else None
        heading_page_reference = page_reference
        if level is not None and heading_page_reference is None:
            heading_page_reference = _content_page_reference(element)
        page = page_objects.get(_object_identity(heading_page_reference))
        if level is not None and title is not None and page is not None:
            bookmarks.append((level, title, page))
        children = _structure_children(element)
        stack.extend((child, page_reference) for child in reversed(children))
    return bookmarks


def _ensure_heading_bookmarks(pdf: Pdf) -> int:
    existing = resolve_indirect(pdf.Root.get("/Outlines"))
    if isinstance(existing, Dictionary) and isinstance(
        resolve_indirect(existing.get("/First")), Dictionary
    ):
        return 0
    headings = _heading_bookmarks(pdf)
    if len(headings) < 2:
        return 0

    outline_root = pdf.make_indirect(Dictionary(Type=Name.Outlines))
    child_groups: dict[tuple[int, int] | int, list[Dictionary]] = {}
    hierarchy: list[tuple[int, Dictionary]] = []
    for level, title, page in headings:
        while hierarchy and hierarchy[-1][0] >= level:
            hierarchy.pop()
        parent = hierarchy[-1][1] if hierarchy else outline_root
        item = pdf.make_indirect(
            Dictionary(
                Title=_bounded_string(title),
                Parent=parent,
                Dest=Array([page, Name.Fit]),
            )
        )
        child_groups.setdefault(_object_identity(parent), []).append(item)
        hierarchy.append((level, item))

    parents = [
        outline_root,
        *(item for items in child_groups.values() for item in items),
    ]
    for parent in parents:
        children = child_groups.get(_object_identity(parent), [])
        if not children:
            continue
        parent["/First"] = children[0]
        parent["/Last"] = children[-1]
        for index, child in enumerate(children):
            if index:
                child["/Prev"] = children[index - 1]
            if index + 1 < len(children):
                child["/Next"] = children[index + 1]

    def descendant_count(parent: Dictionary) -> int:
        count = 0
        for child in child_groups.get(_object_identity(parent), []):
            descendants = descendant_count(child)
            if descendants:
                child["/Count"] = descendants
            count += 1 + descendants
        return count

    outline_root["/Count"] = descendant_count(outline_root)
    pdf.Root["/Outlines"] = outline_root
    return len(headings)


def _annotation_structure_owner(
    structure_root: Dictionary | None,
    annotation: Dictionary,
) -> Dictionary | None:
    if structure_root is None:
        return None
    parent_key = resolve_indirect(annotation.get("/StructParent"))
    parent_tree = resolve_indirect(structure_root.get("/ParentTree"))
    if (
        not isinstance(parent_key, int)
        or isinstance(parent_key, bool)
        or not isinstance(parent_tree, Dictionary)
    ):
        return None
    try:
        owner = resolve_indirect(NumberTree(parent_tree).get(parent_key))
    except (TypeError, ValueError, RuntimeError, pikepdf.PdfError):
        return None
    return owner if isinstance(owner, Dictionary) and "/S" in owner else None


def apply_wcag_21(pdf: Pdf) -> dict[str, int | bool]:
    """Apply machine-enforceable WCAG 2.1 PDF techniques.

    Semantic tagging supplies text alternatives, headings, lists, tables,
    reading order, links, and form ownership. This final pass handles the
    document properties that are specific to the WCAG-enabled PDF/UA mode.
    Requirements that need human judgement remain conversion warnings.
    """
    tab_orders_set = 0
    required_controls_labeled = 0
    annotation_descriptions_added = 0
    annotation_descriptions_review_required = 0
    seen_widgets: set[tuple[int, int] | int] = set()
    language = str(resolve_indirect(pdf.Root.get("/Lang")) or "").strip().casefold()
    strings = accessibility_strings(language)
    structure_root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    if not isinstance(structure_root, Dictionary):
        structure_root = None

    page_labels_added = 0
    page_labels_repaired = 0
    if not _valid_page_labels(pdf):
        page_labels_repaired = int("/PageLabels" in pdf.Root)
        pdf.Root["/PageLabels"] = pdf.make_indirect(
            Dictionary(Nums=Array([0, Dictionary(S=Name.D, St=1)]))
        )
        page_labels_added = int(not page_labels_repaired)

    bookmarks_added = _ensure_heading_bookmarks(pdf)

    for page in pdf.pages:
        if resolve_indirect(page.obj.get("/Tabs")) != Name.S:
            page.obj["/Tabs"] = Name.S
            tab_orders_set += 1

        annotations = resolve_indirect(page.obj.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        for value in annotations:
            annotation = resolve_indirect(value)
            if not isinstance(annotation, Dictionary):
                continue
            subtype = resolve_indirect(annotation.get("/Subtype"))
            if subtype != Name.Widget:
                contents = resolve_indirect(annotation.get("/Contents"))
                if not isinstance(contents, String) or not str(contents).strip():
                    description = None
                    fallback_generated = False
                    if subtype == Name.Popup:
                        parent = resolve_indirect(annotation.get("/Parent"))
                        candidate = (
                            resolve_indirect(parent.get("/Contents"))
                            if isinstance(parent, Dictionary)
                            else None
                        )
                        if isinstance(candidate, String) and str(candidate).strip():
                            description = str(candidate).strip()
                    elif subtype == Name.FileAttachment:
                        file_spec = resolve_indirect(annotation.get("/FS"))
                        if isinstance(file_spec, Dictionary):
                            for key in ("/Desc", "/UF", "/F"):
                                candidate = resolve_indirect(file_spec.get(key))
                                if (
                                    isinstance(candidate, String)
                                    and str(candidate).strip()
                                ):
                                    description = str(candidate).strip()
                                    break
                    if description is None:
                        subtype_name = (
                            str(subtype).removeprefix("/")
                            if isinstance(subtype, Name)
                            else "PDF"
                        )
                        description = (
                            strings.link
                            if subtype == Name.Link
                            else strings.annotation.format(subtype=subtype_name)
                        )
                        fallback_generated = True
                        annotation_descriptions_review_required += 1
                    annotation["/Contents"] = _bounded_string(description)
                    if (
                        fallback_generated
                        and primary_language(language) != strings.language
                    ):
                        owner = _annotation_structure_owner(structure_root, annotation)
                        if owner is not None and "/Lang" not in owner:
                            owner["/Lang"] = String(strings.language)
                    annotation_descriptions_added += 1
            if subtype != Name.Widget:
                continue
            widget = annotation
            objgen = widget.objgen
            identity: tuple[int, int] | int = objgen if objgen != (0, 0) else id(widget)
            if identity in seen_widgets:
                continue
            seen_widgets.add(identity)
            _localize_generic_widget_tooltip(widget, strings)
            required_controls_labeled += int(
                _indicate_required_control(widget, strings)
            )

    language_review_required = primary_language(language) in {None, "und"}
    logger.info(
        "Applied WCAG 2.1 PDF requirements: %d tab order(s), %d required "
        "control label(s), %d annotation description(s)",
        tab_orders_set,
        required_controls_labeled,
        annotation_descriptions_added,
    )
    return {
        "page_tab_orders_set": tab_orders_set,
        "page_labels_added": page_labels_added,
        "page_labels_repaired": page_labels_repaired,
        "bookmarks_added": bookmarks_added,
        "required_controls_labeled": required_controls_labeled,
        "annotation_descriptions_added": annotation_descriptions_added,
        "annotation_descriptions_review_required": (
            annotation_descriptions_review_required
        ),
        "language_review_required": language_review_required,
        "human_review_required": True,
    }


__all__ = ["apply_wcag_21", "prepare_pdfua_document"]
