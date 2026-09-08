# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font discovery across all nested PDF structures.

Discovers fonts in:
- Page-level Resources/Font
- Form XObjects (Resources/XObject/*/Resources/Font where Subtype=/Form)
- Annotation Appearance Streams (Annots/*/AP/{N,R,D}/Resources/Font)
- Tiling Patterns (Resources/Pattern/*/Resources/Font where PatternType=1)
- Nested combinations of the above (recursive)
"""

import logging
import secrets
from collections.abc import Iterator

import pikepdf

from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

_ObjectKey = tuple[int, int]
_DirectAncestors = tuple[pikepdf.Object, ...]


def _indirect_identity(obj: pikepdf.Object) -> _ObjectKey | None:
    """Return a stable identity for an indirect PDF object."""
    objgen = obj.objgen
    return objgen if objgen != (0, 0) else None


def _check_visited(obj: pikepdf.Object, visited: set[_ObjectKey]) -> bool:
    """Deduplicate indirect objects, whose object numbers are stable."""
    key = _indirect_identity(obj)
    if key is None:
        return False
    if key in visited:
        return True
    visited.add(key)
    return False


def _is_direct_cycle(
    obj: pikepdf.Object,
    ancestors: _DirectAncestors,
    marker: str,
) -> bool:
    """Return whether a direct object already occurs on the current path."""
    if not ancestors:
        return False

    # pikepdf does not expose qpdf's direct-object identity. A temporary key
    # provides the same test because wrappers of one direct object share writes.
    while marker in obj:
        marker += "_"
    obj[marker] = True
    try:
        return any(marker in ancestor for ancestor in ancestors)
    finally:
        del obj[marker]


def _descendant_ancestors(
    container: pikepdf.Object,
    ancestors: _DirectAncestors,
) -> _DirectAncestors:
    """Reset direct ancestors after crossing a stable indirect object."""
    return ancestors if _indirect_identity(container) is None else ()


def get_page_resources(page: pikepdf.Page) -> pikepdf.Object | None:
    """Return a page's local or inherited resources dictionary."""
    current = _resolve_indirect(page.obj)
    seen: set[_ObjectKey] = set()
    direct_ancestors: _DirectAncestors = ()
    cycle_marker = f"/__pdftopdfa_direct_cycle_{secrets.token_hex(32)}"
    while isinstance(current, pikepdf.Dictionary):
        key = _indirect_identity(current)
        if key is None:
            if _is_direct_cycle(current, direct_ancestors, cycle_marker):
                return None
            direct_ancestors += (current,)
        else:
            direct_ancestors = ()
            if key in seen:
                return None
            seen.add(key)

        resources = current.get("/Resources")
        if resources is not None:
            resources = _resolve_indirect(resources)
            return resources if isinstance(resources, pikepdf.Dictionary) else None

        parent = current.get("/Parent")
        if parent is None:
            return None
        current = _resolve_indirect(parent)
    return None


def iter_all_page_fonts(
    page: pikepdf.Page,
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields all (font_key, font_obj) pairs from a page and its nested structures.

    Discovers fonts in page-level Resources, Form XObjects, Tiling Patterns,
    and Annotation Appearance Streams. Uses cycle detection to handle
    recursive structures safely.

    Args:
        page: A pikepdf Page object.

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    visited: set[_ObjectKey] = set()

    # Page-level Resources
    try:
        resources = get_page_resources(page)
    except Exception:
        resources = None

    if resources is not None:
        yield from _iter_fonts_from_resources(resources, visited)

    # Annotation Appearance Streams
    yield from _iter_fonts_from_appearance_streams(page, visited)


def iter_acroform_dr_fonts(
    pdf: pikepdf.Pdf,
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields all (font_key, font_obj) pairs from the AcroForm /DR font dict.

    AcroForm Default Resources fonts may never appear in page resources,
    but they are still used to render widget field appearances and must
    satisfy the same PDF/A font rules as page fonts.

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    try:
        root = pdf.Root
        if root is None or "/AcroForm" not in root:
            return
        acroform = _resolve_indirect(root.AcroForm)
        dr = acroform.get("/DR")
        if dr is None:
            return
        dr = _resolve_indirect(dr)
        font_dict = dr.get("/Font")
        if font_dict is None:
            return
        font_dict = _resolve_indirect(font_dict)
        if not isinstance(font_dict, pikepdf.Dictionary):
            return
        font_keys = list(font_dict.keys())
    except Exception as e:
        log_suppressed_error(logger, e, "Error accessing AcroForm /DR fonts: %s", e)
        return

    for font_key in font_keys:
        try:
            font_obj = _resolve_indirect(font_dict[font_key])
            try:
                key_str = str(font_key)
            except (UnicodeDecodeError, UnicodeEncodeError):
                key_str = repr(font_key)
            yield (key_str, font_obj)
        except Exception:
            continue


def _iter_fonts_from_resources(
    resources: pikepdf.Object,
    visited: set[_ObjectKey],
    direct_ancestors: _DirectAncestors = (),
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yield fonts from a resource graph without using Python recursion.

    Args:
        resources: A PDF Resources dictionary.
        visited: Stable identities already visited for cycle detection.

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    tasks: list[tuple[str, pikepdf.Object, str | None, _DirectAncestors]] = [
        ("resources", resources, None, direct_ancestors)
    ]
    cycle_marker = f"/__pdftopdfa_direct_cycle_{secrets.token_hex(32)}"
    while tasks:
        kind, obj, font_key, current_ancestors = tasks.pop()
        if kind == "font":
            assert font_key is not None
            yield font_key, obj
            if (
                isinstance(obj, pikepdf.Dictionary)
                and str(obj.get("/Subtype")) == "/Type3"
                and not _check_visited(obj, visited)
            ):
                type3_resources = _resolve_indirect(obj.get("/Resources"))
                if isinstance(type3_resources, pikepdf.Dictionary):
                    tasks.append(
                        (
                            "resources",
                            type3_resources,
                            None,
                            _descendant_ancestors(obj, current_ancestors),
                        )
                    )
            continue

        try:
            current_resources = _resolve_indirect(obj)
        except Exception:
            continue
        if not isinstance(current_resources, pikepdf.Dictionary):
            continue
        if _check_visited(current_resources, visited):
            continue
        if _indirect_identity(current_resources) is None:
            if _is_direct_cycle(
                current_resources,
                current_ancestors,
                cycle_marker,
            ):
                continue
            current_ancestors += (current_resources,)
        else:
            current_ancestors = ()

        discovered: list[tuple[str, pikepdf.Object, str | None, _DirectAncestors]] = []

        # 1. Yield fonts from Resources/Font, then traverse Type3 resources.
        try:
            font_dict = _resolve_indirect(current_resources.get("/Font"))
        except Exception:
            font_dict = None
        if isinstance(font_dict, pikepdf.Dictionary):
            for raw_font_key in list(font_dict.keys()):
                try:
                    font_obj = _resolve_indirect(font_dict[raw_font_key])
                    try:
                        key_str = str(raw_font_key)
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        key_str = repr(raw_font_key)
                    discovered.append(("font", font_obj, key_str, current_ancestors))
                except Exception:
                    continue

        # 2. Traverse Form XObjects.
        try:
            xobject_dict = _resolve_indirect(current_resources.get("/XObject"))
        except Exception:
            xobject_dict = None
        if isinstance(xobject_dict, pikepdf.Dictionary):
            for xobj_key in list(xobject_dict.keys()):
                try:
                    xobj = _resolve_indirect(xobject_dict[xobj_key])
                    if _check_visited(xobj, visited):
                        continue
                    if str(xobj.get("/Subtype")) != "/Form":
                        continue
                    nested_resources = _resolve_indirect(xobj.get("/Resources"))
                    if isinstance(nested_resources, pikepdf.Dictionary):
                        discovered.append(
                            (
                                "resources",
                                nested_resources,
                                None,
                                _descendant_ancestors(xobj, current_ancestors),
                            )
                        )
                except Exception:
                    continue

        # 3. Traverse Tiling Patterns.
        try:
            pattern_dict = _resolve_indirect(current_resources.get("/Pattern"))
        except Exception:
            pattern_dict = None
        if isinstance(pattern_dict, pikepdf.Dictionary):
            for pat_key in list(pattern_dict.keys()):
                try:
                    pattern = _resolve_indirect(pattern_dict[pat_key])
                    if _check_visited(pattern, visited):
                        continue
                    if int(pattern.get("/PatternType", 0)) != 1:
                        continue
                    nested_resources = _resolve_indirect(pattern.get("/Resources"))
                    if isinstance(nested_resources, pikepdf.Dictionary):
                        discovered.append(
                            (
                                "resources",
                                nested_resources,
                                None,
                                _descendant_ancestors(pattern, current_ancestors),
                            )
                        )
                except Exception:
                    continue

        # 4. Traverse transparency soft-mask group Form XObjects.
        try:
            extgstate_dict = _resolve_indirect(current_resources.get("/ExtGState"))
        except Exception:
            extgstate_dict = None
        if isinstance(extgstate_dict, pikepdf.Dictionary):
            for gs_key in list(extgstate_dict.keys()):
                try:
                    extgstate = _resolve_indirect(extgstate_dict[gs_key])
                    if not isinstance(extgstate, pikepdf.Dictionary):
                        continue
                    font = _resolve_indirect(extgstate.get("/Font"))
                    if isinstance(font, pikepdf.Array) and len(font) == 2:
                        font_obj = _resolve_indirect(font[0])
                        if isinstance(font_obj, pikepdf.Dictionary):
                            discovered.append(
                                ("font", font_obj, str(gs_key), current_ancestors)
                            )
                    soft_mask = _resolve_indirect(extgstate.get("/SMask"))
                    if not isinstance(soft_mask, pikepdf.Dictionary):
                        continue
                    group = _resolve_indirect(soft_mask.get("/G"))
                    if not isinstance(group, pikepdf.Stream):
                        continue
                    subtype = group.get("/Subtype")
                    if subtype is not None and str(subtype) != "/Form":
                        continue
                    if _check_visited(group, visited):
                        continue
                    nested_resources = _resolve_indirect(group.get("/Resources"))
                    if isinstance(nested_resources, pikepdf.Dictionary):
                        discovered.append(
                            (
                                "resources",
                                nested_resources,
                                None,
                                _descendant_ancestors(group, current_ancestors),
                            )
                        )
                except Exception:
                    continue

        tasks.extend(reversed(discovered))


def _iter_fonts_from_appearance_streams(
    page: pikepdf.Page,
    visited: set[_ObjectKey],
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields fonts from Annotation Appearance Streams on a page.

    Iterates page /Annots, and for each annotation iterates /AP/{N,R,D}.
    Each AP entry can be a stream (Form XObject) directly, or a dictionary
    of sub-state streams.

    Args:
        page: A pikepdf Page object.
        visited: Set of objgen tuples already visited (for cycle detection).

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    annots = page.get("/Annots")
    if annots is None:
        return

    try:
        annots = _resolve_indirect(annots)
    except Exception:
        return

    for annot_ref in annots:
        try:
            annot = _resolve_indirect(annot_ref)
            ap = annot.get("/AP")
            if ap is None:
                continue
            ap = _resolve_indirect(ap)

            # Iterate N (Normal), R (Rollover), D (Down) appearance entries
            for ap_key in ("/N", "/R", "/D"):
                ap_entry = ap.get(ap_key)
                if ap_entry is None:
                    continue

                try:
                    ap_entry = _resolve_indirect(ap_entry)
                except Exception:
                    continue

                # AP entry can be a stream (Form XObject) or a dict of sub-states
                yield from _iter_fonts_from_ap_entry(ap_entry, visited)

        except Exception:
            continue


def _iter_fonts_from_ap_entry(
    ap_entry: pikepdf.Object,
    visited: set[_ObjectKey],
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields fonts from a single AP entry (stream or sub-state dict).

    Args:
        ap_entry: An appearance stream (Form XObject) or a dictionary
            mapping sub-state names to appearance streams.
        visited: Set of objgen tuples already visited (for cycle detection).

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    if isinstance(ap_entry, pikepdf.Stream):
        # Direct Form XObject stream
        yield from _iter_fonts_from_form_xobject(ap_entry, visited)
    elif isinstance(ap_entry, pikepdf.Dictionary):
        # Dictionary of sub-state streams
        for sub_key in list(ap_entry.keys()):
            try:
                sub_stream = ap_entry[sub_key]
                sub_stream = _resolve_indirect(sub_stream)
                if isinstance(sub_stream, pikepdf.Stream):
                    yield from _iter_fonts_from_form_xobject(sub_stream, visited)
            except Exception:
                continue


def _iter_fonts_from_form_xobject(
    xobj: pikepdf.Object,
    visited: set[_ObjectKey],
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields fonts from a Form XObject's Resources.

    Args:
        xobj: A Form XObject (stream with /Subtype /Form).
        visited: Set of objgen tuples already visited (for cycle detection).

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    if _check_visited(xobj, visited):
        return

    resources = xobj.get("/Resources")
    if resources is None:
        return

    try:
        resources = _resolve_indirect(resources)
    except Exception:
        return

    yield from _iter_fonts_from_resources(resources, visited)
