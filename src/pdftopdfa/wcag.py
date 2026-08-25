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
    seen_widgets: set[tuple[int, int] | int] = set()

    for page in pdf.pages:
        if resolve_indirect(page.obj.get("/Tabs")) != Name.S:
            page.obj["/Tabs"] = Name.S
            tab_orders_set += 1

        annotations = resolve_indirect(page.obj.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        for value in annotations:
            widget = resolve_indirect(value)
            if not isinstance(widget, Dictionary) or (
                resolve_indirect(widget.get("/Subtype")) != Name.Widget
            ):
                continue
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
        "control label(s)",
        tab_orders_set,
        required_controls_labeled,
    )
    return {
        "page_tab_orders_set": tab_orders_set,
        "required_controls_labeled": required_controls_labeled,
        "language_review_required": language_review_required,
    }


__all__ = ["apply_wcag_21"]
