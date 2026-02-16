# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Recursive font discovery across all nested PDF structures.

Discovers fonts in:
- Page-level Resources/Font
- Form XObjects (Resources/XObject/*/Resources/Font where Subtype=/Form)
- Annotation Appearance Streams (Annots/*/AP/{N,R,D}/Resources/Font)
- Tiling Patterns (Resources/Pattern/*/Resources/Font where PatternType=1)
- Nested combinations of the above (recursive)
"""

import logging
from collections.abc import Iterator

import pikepdf

from ..utils import resolve_indirect as _resolve_indirect
from .utils import check_visited as _check_visited

logger = logging.getLogger(__name__)


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
    visited: set[tuple[int, int]] = set()

    # Page-level Resources
    resources = page.get("/Resources")
    if resources is not None:
        try:
            resources = _resolve_indirect(resources)
        except Exception:
            resources = None

    if resources is not None:
        yield from _iter_fonts_from_resources(resources, visited)

    # Annotation Appearance Streams
    yield from _iter_fonts_from_appearance_streams(page, visited)


def _iter_fonts_from_resources(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yields fonts from a Resources dictionary, recursing into XObjects and Patterns.

    Args:
        resources: A PDF Resources dictionary.
        visited: Set of objgen tuples already visited (for cycle detection).

    Yields:
        Tuples of (font_key, dereferenced_font_obj).
    """
    # 1. Yield fonts from Resources/Font
    font_dict = resources.get("/Font")
    if font_dict is not None:
        try:
            font_dict = _resolve_indirect(font_dict)
        except Exception:
            font_dict = None

    if font_dict is not None:
        for font_key in list(font_dict.keys()):
            try:
                font_obj = font_dict[font_key]
                font_obj = _resolve_indirect(font_obj)
                try:
                    key_str = str(font_key)
                except (UnicodeDecodeError, UnicodeEncodeError):
                    key_str = repr(font_key)
                yield (key_str, font_obj)

                # Recurse into Type3 font Resources
                subtype = font_obj.get("/Subtype")
                if subtype is not None and str(subtype) == "/Type3":
                    if not _check_visited(font_obj, visited):
                        type3_resources = font_obj.get("/Resources")
                        if type3_resources is not None:
                            type3_resources = _resolve_indirect(type3_resources)
                            yield from _iter_fonts_from_resources(
                                type3_resources, visited
                            )
            except Exception:
                continue

    # 2. Recurse into Form XObjects in Resources/XObject
    xobject_dict = resources.get("/XObject")
    if xobject_dict is not None:
        try:
            xobject_dict = _resolve_indirect(xobject_dict)
        except Exception:
            xobject_dict = None

    if xobject_dict is not None:
        for xobj_key in list(xobject_dict.keys()):
            try:
                xobj = xobject_dict[xobj_key]
                xobj = _resolve_indirect(xobj)

                if _check_visited(xobj, visited):
                    continue

                # Only recurse into Form XObjects
                subtype = xobj.get("/Subtype")
                if subtype is not None and str(subtype) == "/Form":
                    nested_resources = xobj.get("/Resources")
                    if nested_resources is not None:
                        nested_resources = _resolve_indirect(nested_resources)
                        yield from _iter_fonts_from_resources(nested_resources, visited)
            except Exception:
                continue

    # 3. Recurse into Tiling Patterns in Resources/Pattern
    pattern_dict = resources.get("/Pattern")
    if pattern_dict is not None:
        try:
            pattern_dict = _resolve_indirect(pattern_dict)
        except Exception:
            pattern_dict = None

    if pattern_dict is not None:
        for pat_key in list(pattern_dict.keys()):
            try:
                pattern = pattern_dict[pat_key]
                pattern = _resolve_indirect(pattern)

                if _check_visited(pattern, visited):
                    continue

                # Only recurse into Tiling Patterns (PatternType=1)
                pattern_type = pattern.get("/PatternType")
                if pattern_type is not None and int(pattern_type) == 1:
                    nested_resources = pattern.get("/Resources")
                    if nested_resources is not None:
                        nested_resources = _resolve_indirect(nested_resources)
                        yield from _iter_fonts_from_resources(nested_resources, visited)
            except Exception:
                continue


def _iter_fonts_from_appearance_streams(
    page: pikepdf.Page,
    visited: set[tuple[int, int]],
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
    visited: set[tuple[int, int]],
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
    visited: set[tuple[int, int]],
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
