# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Apply deterministic WCAG 2.1 PDF accessibility requirements."""

import logging
import re

from pikepdf import Array, Dictionary, Name, Pdf, String

from .utils import resolve_indirect

logger = logging.getLogger(__name__)

_FIELD_REQUIRED = 1 << 1
_GENERIC_WIDGET_TOOLTIP = "Form field"
_MAX_STRING_BYTES = 32_767
_REQUIRED_LABEL = re.compile(r"\brequired\b", re.IGNORECASE)
_REQUIRED_SUFFIX = " (required)"


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


def _required_tooltip(label: str) -> String:
    """Append the required suffix without exceeding the PDF string byte limit."""
    result = String(f"{label}{_REQUIRED_SUFFIX}")
    if len(bytes(result)) <= _MAX_STRING_BYTES:
        return result

    low = 0
    high = len(label)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = String(f"{label[:middle]}{_REQUIRED_SUFFIX}")
        if len(bytes(candidate)) <= _MAX_STRING_BYTES:
            low = middle
        else:
            high = middle - 1
    return String(f"{label[:low]}{_REQUIRED_SUFFIX}")


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


def _indicate_required_control(widget: Dictionary) -> bool:
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
        if label and label != _GENERIC_WIDGET_TOOLTIP:
            label_entry = owner, label
            break
    if label_entry is None:
        return False
    label_owner, label = label_entry
    if _REQUIRED_LABEL.search(label):
        return False
    if field_name_entry is not None:
        field_owner = field_name_entry[0]
    elif flags_entry is not None:
        field_owner = flags_entry[0]
    else:
        field_owner = label_owner
    tooltip = _required_tooltip(label)
    field_owner["/TU"] = tooltip
    if "/TU" in widget:
        widget["/TU"] = tooltip
    return True


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

    page_labels_added = 0
    if not isinstance(resolve_indirect(pdf.Root.get("/PageLabels")), Dictionary):
        pdf.Root["/PageLabels"] = pdf.make_indirect(
            Dictionary(Nums=Array([0, Dictionary(S=Name.D, St=1)]))
        )
        page_labels_added = 1

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
                            "Link"
                            if subtype == Name.Link
                            else f"{subtype_name} annotation"
                        )
                        annotation_descriptions_review_required += 1
                    annotation["/Contents"] = _bounded_string(description)
                    annotation_descriptions_added += 1
            if subtype != Name.Widget:
                continue
            widget = annotation
            objgen = widget.objgen
            identity: tuple[int, int] | int = objgen if objgen != (0, 0) else id(widget)
            if identity in seen_widgets:
                continue
            seen_widgets.add(identity)
            required_controls_labeled += int(_indicate_required_control(widget))

    language = str(resolve_indirect(pdf.Root.get("/Lang")) or "").strip().casefold()
    language_review_required = not language or language == "und"
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
        "required_controls_labeled": required_controls_labeled,
        "annotation_descriptions_added": annotation_descriptions_added,
        "annotation_descriptions_review_required": (
            annotation_descriptions_review_required
        ),
        "language_review_required": language_review_required,
        "human_review_required": True,
    }


__all__ = ["apply_wcag_21", "prepare_pdfua_document"]
