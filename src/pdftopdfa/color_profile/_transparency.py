# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transparency group color space fixes for PDF/A compliance."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _create_icc_colorspace
from ._types import _DEVICE_NAME_TO_TYPE, ColorSpaceType

logger = logging.getLogger(__name__)


def _fix_transparency_group_cs_in_form(
    form_xobj,
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Replace Device color space in a Form XObject's transparency group /CS.

    ISO 19005-2 clause 6.4 forbids uncalibrated Device color spaces in
    the ``/CS`` entry of a transparency group dictionary.  Default color
    spaces from resources do NOT apply to explicit ``/CS`` references, so
    ALL Device color spaces must be replaced (not just non-dominant ones).

    Args:
        form_xobj: A Form XObject (pikepdf Stream/Dictionary).
        pdf: The document (needed to create ICC streams).
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        1 if the ``/CS`` was replaced, 0 otherwise.
    """
    group = form_xobj.get(Name.Group)
    if group is None:
        return 0

    group = _resolve_indirect(group)

    # Only process Transparency groups
    if group.get(Name.S) != Name.Transparency:
        return 0

    cs = group.get(Name.CS)
    if isinstance(cs, Name):
        cs_type = _DEVICE_NAME_TO_TYPE.get(cs)
    elif isinstance(cs, Array) and len(cs) == 1 and isinstance(cs[0], Name):
        # Array-form device color space, e.g. [/DeviceRGB] from wkhtmltopdf
        cs_type = _DEVICE_NAME_TO_TYPE.get(cs[0])
    else:
        return 0

    if cs_type is None:
        return 0

    group[Name.CS] = _create_icc_colorspace(pdf, cs_type, icc_stream_cache)
    return 1


def _fix_transparency_groups_in_xobjects(
    xobjects,
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
    visited: set[tuple[int, int]],
) -> int:
    """Recursively fix transparency group /CS in an XObject dictionary.

    For each Form XObject found, fixes its own transparency group ``/CS``
    and then recurses into ``/Resources/XObject`` for nested forms.

    Args:
        xobjects: ``/XObject`` Dictionary from resources.
        pdf: The document being converted.
        icc_stream_cache: Shared cache of ICC stream objects.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        Number of transparency groups fixed.
    """
    fixed = 0

    for name in xobjects.keys():
        xobj = _resolve_indirect(xobjects[name])

        # Cycle detection using objgen
        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        if xobj.get(Name.Subtype) != Name.Form:
            continue

        fixed += _fix_transparency_group_cs_in_form(xobj, pdf, icc_stream_cache)

        # Recurse into nested resources
        form_resources = xobj.get(Name.Resources)
        if form_resources is not None:
            form_resources = _resolve_indirect(form_resources)
            nested = form_resources.get(Name.XObject)
            if nested:
                fixed += _fix_transparency_groups_in_xobjects(
                    nested, pdf, icc_stream_cache, visited
                )

    return fixed


def _process_ap_entry(
    ap_value,
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
    visited: set[tuple[int, int]],
) -> int:
    """Process a single AP dictionary entry (/N, /R, or /D).

    The entry may be a Form XObject (stream) directly, or a dictionary
    of sub-state Form XObjects.

    Args:
        ap_value: The value of an AP entry.
        pdf: The document being converted.
        icc_stream_cache: Shared cache of ICC stream objects.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        Number of transparency groups fixed.
    """
    fixed = 0
    ap_value = _resolve_indirect(ap_value)

    if isinstance(ap_value, Stream):
        # Direct AP stream (Form XObject)
        objgen = ap_value.objgen
        if objgen != (0, 0):
            if objgen in visited:
                return 0
            visited.add(objgen)

        fixed += _fix_transparency_group_cs_in_form(ap_value, pdf, icc_stream_cache)

        # Check for nested XObjects within the AP stream
        ap_resources = ap_value.get(Name.Resources)
        if ap_resources is not None:
            ap_resources = _resolve_indirect(ap_resources)
            nested = ap_resources.get(Name.XObject)
            if nested:
                fixed += _fix_transparency_groups_in_xobjects(
                    nested, pdf, icc_stream_cache, visited
                )
    elif isinstance(ap_value, Dictionary):
        # Sub-state dictionary: each value is a Form XObject
        for key in ap_value.keys():
            sub_stream = _resolve_indirect(ap_value[key])
            if isinstance(sub_stream, Stream):
                objgen = sub_stream.objgen
                if objgen != (0, 0):
                    if objgen in visited:
                        continue
                    visited.add(objgen)

                fixed += _fix_transparency_group_cs_in_form(
                    sub_stream, pdf, icc_stream_cache
                )

                sub_resources = sub_stream.get(Name.Resources)
                if sub_resources is not None:
                    sub_resources = _resolve_indirect(sub_resources)
                    nested = sub_resources.get(Name.XObject)
                    if nested:
                        fixed += _fix_transparency_groups_in_xobjects(
                            nested, pdf, icc_stream_cache, visited
                        )

    return fixed


def _fix_transparency_group_colorspaces(
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Fix all transparency group /CS entries in the document.

    Iterates all pages and processes:
    1. Page -> /Group (page-level transparency group)
    2. Page -> /Resources/XObject (recursive)
    3. Page -> /Annots -> each annotation's /AP -> /N, /R, /D entries

    Args:
        pdf: The document being converted.
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        Total number of transparency groups fixed.
    """
    fixed = 0
    visited: set[tuple[int, int]] = set()

    for page in pdf.pages:
        # Fix page-level transparency group /CS (ISO 32000-1, Table 30)
        fixed += _fix_transparency_group_cs_in_form(page, pdf, icc_stream_cache)

        resources = page.get(Name.Resources)
        if resources is not None:
            resources = _resolve_indirect(resources)
            xobjects = resources.get(Name.XObject)
            if xobjects:
                fixed += _fix_transparency_groups_in_xobjects(
                    xobjects, pdf, icc_stream_cache, visited
                )

        # Process annotations
        annots = page.get(Name.Annots)
        if annots is None:
            continue

        annots = _resolve_indirect(annots)
        for i in range(len(annots)):
            try:
                annot = _resolve_indirect(annots[i])
                ap = annot.get(Name.AP)
                if ap is None:
                    continue
                ap = _resolve_indirect(ap)

                for ap_key in (Name.N, Name.R, Name.D):
                    ap_entry = ap.get(ap_key)
                    if ap_entry is not None:
                        fixed += _process_ap_entry(
                            ap_entry, pdf, icc_stream_cache, visited
                        )
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                logger.debug("Error processing annotation AP: %s", e)

    return fixed
