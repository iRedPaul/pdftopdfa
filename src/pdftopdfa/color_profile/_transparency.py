# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transparency group color space fixes for PDF/A compliance."""

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..fonts.glyph_usage import _iter_content_streams_with_resources
from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _create_icc_colorspace
from ._types import (
    _CMYK_OPERATORS,
    _CS_OPERATORS,
    _DEVICE_NAME_TO_TYPE,
    _GRAY_OPERATORS,
    _RGB_OPERATORS,
    ColorSpaceType,
)

logger = logging.getLogger(__name__)


def _gs_has_transparency(gs_dict) -> bool:
    """Check if an ExtGState dictionary uses transparency features.

    Args:
        gs_dict: A resolved ExtGState dictionary.

    Returns:
        True if any transparency feature is detected.
    """
    try:
        bm = gs_dict.get(Name.BM)
        if bm is not None and bm not in (Name.Normal, Name.Compatible):
            return True
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        ca_upper = gs_dict.get(Name.CA)
        if ca_upper is not None and float(ca_upper) < 1.0:
            return True
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        ca_lower = gs_dict.get(Name.ca)
        if ca_lower is not None and float(ca_lower) < 1.0:
            return True
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        smask = gs_dict.get(Name.SMask)
        if smask is not None:
            smask = _resolve_indirect(smask)
            if isinstance(smask, Dictionary):
                return True
    except (AttributeError, TypeError, ValueError):
        pass

    return False


def _resources_have_transparency(
    resources,
    visited: set[tuple[int, int]],
) -> bool:
    """Check nested Form resources for transparency without recursion."""
    stack = [resources]

    while stack:
        try:
            current = _resolve_indirect(stack.pop())

            ext_gstate = current.get(Name.ExtGState)
            if ext_gstate is not None:
                ext_gstate = _resolve_indirect(ext_gstate)
                for key in ext_gstate.keys():
                    try:
                        gs = _resolve_indirect(ext_gstate[key])
                        if _gs_has_transparency(gs):
                            return True
                    except (AttributeError, TypeError, ValueError):
                        continue

            xobjects = current.get(Name.XObject)
            if xobjects is None:
                continue
            xobjects = _resolve_indirect(xobjects)
            for key in xobjects.keys():
                try:
                    xobject = _resolve_indirect(xobjects[key])
                    objgen = xobject.objgen
                    if objgen != (0, 0):
                        if objgen in visited:
                            continue
                        visited.add(objgen)

                    if xobject.get(Name.Subtype) != Name.Form:
                        continue

                    group = xobject.get(Name.Group)
                    if group is not None:
                        group = _resolve_indirect(group)
                        if group.get(Name.S) == Name.Transparency:
                            return True

                    form_resources = xobject.get(Name.Resources)
                    if form_resources is not None:
                        stack.append(form_resources)
                except (AttributeError, TypeError, ValueError):
                    continue
        except (AttributeError, TypeError, ValueError):
            continue

    return False


def _page_uses_transparency(page) -> bool:
    """Check if a page uses transparency features.

    Examines page resources and annotation appearance streams.

    Args:
        page: A pikepdf page object.

    Returns:
        True if the page uses any transparency features.
    """
    visited: set[tuple[int, int]] = set()

    # Check page resources
    resources = page.get(Name.Resources)
    if resources is not None:
        resources = _resolve_indirect(resources)
        if _resources_have_transparency(resources, visited):
            return True

    # Patterns, Type3 CharProcs, and soft-mask groups are content-bearing
    # resource owners too.  The generic content graph walker supplies their
    # effective resources without relying on Python recursion.
    try:
        for owner, nested_resources in _iter_content_streams_with_resources(page):
            owner = _resolve_indirect(owner)
            if isinstance(owner, Stream):
                group = _resolve_indirect(owner.get(Name.Group))
                if (
                    isinstance(group, Dictionary)
                    and group.get(Name.S) == Name.Transparency
                ):
                    return True
            if _resources_have_transparency(nested_resources, visited):
                return True
    except (pikepdf.PdfError, AttributeError, TypeError, ValueError):
        pass

    # Check annotation AP streams
    annots = page.get(Name.Annots)
    if annots is not None:
        try:
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
                        if ap_entry is None:
                            continue
                        ap_entry = _resolve_indirect(ap_entry)

                        if isinstance(ap_entry, Stream):
                            # Check if Form XObject has transparency group
                            group = ap_entry.get(Name.Group)
                            if group is not None:
                                group = _resolve_indirect(group)
                                if group.get(Name.S) == Name.Transparency:
                                    return True
                            # Check AP stream resources
                            ap_res = ap_entry.get(Name.Resources)
                            if ap_res is not None:
                                ap_res = _resolve_indirect(ap_res)
                                if _resources_have_transparency(ap_res, visited):
                                    return True
                        elif isinstance(ap_entry, Dictionary):
                            # Sub-state dict
                            for sub_key in ap_entry.keys():
                                try:
                                    sub = _resolve_indirect(ap_entry[sub_key])
                                    if isinstance(sub, Stream):
                                        group = sub.get(Name.Group)
                                        if group is not None:
                                            group = _resolve_indirect(group)
                                            if group.get(Name.S) == Name.Transparency:
                                                return True
                                        sub_res = sub.get(Name.Resources)
                                        if sub_res is not None:
                                            sub_res = _resolve_indirect(sub_res)
                                            if _resources_have_transparency(
                                                sub_res, visited
                                            ):
                                                return True
                                except (
                                    AttributeError,
                                    TypeError,
                                    ValueError,
                                ):
                                    continue
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
        except (AttributeError, TypeError, ValueError):
            pass

    return False


def _detect_page_dominant_cs(page) -> ColorSpaceType:
    """Detect the dominant color space used on a page.

    Parses the content stream for color operators and checks Image
    XObject color spaces.  Priority: CMYK > RGB > Gray, default RGB.

    Args:
        page: A pikepdf page object.

    Returns:
        The dominant ColorSpaceType for the page.
    """
    has_gray = False
    has_rgb = False
    has_cmyk = False

    # Parse content stream for color operators
    try:
        for _operands, operator in pikepdf.parse_content_stream(page):
            op_name = str(operator)
            if op_name in _CMYK_OPERATORS:
                has_cmyk = True
            elif op_name in _RGB_OPERATORS:
                has_rgb = True
            elif op_name in _GRAY_OPERATORS:
                has_gray = True
            elif op_name in _CS_OPERATORS and _operands:
                cs_name = _operands[0]
                if cs_name == Name.DeviceCMYK:
                    has_cmyk = True
                elif cs_name == Name.DeviceRGB:
                    has_rgb = True
                elif cs_name == Name.DeviceGray:
                    has_gray = True
    except (pikepdf.PdfError, AttributeError, IndexError, TypeError):
        pass

    # Check Image XObject color spaces
    try:
        resources = page.get(Name.Resources)
        if resources is not None:
            resources = _resolve_indirect(resources)
            xobjects = resources.get(Name.XObject)
            if xobjects is not None:
                xobjects = _resolve_indirect(xobjects)
                for key in xobjects.keys():
                    try:
                        xobj = _resolve_indirect(xobjects[key])
                        if xobj.get(Name.Subtype) != Name.Image:
                            continue
                        cs = xobj.get(Name.ColorSpace)
                        if isinstance(cs, Name):
                            cs_type = _DEVICE_NAME_TO_TYPE.get(cs)
                            if cs_type == ColorSpaceType.DEVICE_CMYK:
                                has_cmyk = True
                            elif cs_type == ColorSpaceType.DEVICE_RGB:
                                has_rgb = True
                            elif cs_type == ColorSpaceType.DEVICE_GRAY:
                                has_gray = True
                    except (AttributeError, TypeError, ValueError):
                        continue
    except (pikepdf.PdfError, AttributeError, TypeError, ValueError):
        pass

    # Include Forms, tiling patterns, soft-mask groups, and Type3 CharProcs.
    try:
        for owner, resources in _iter_content_streams_with_resources(page):
            owner = _resolve_indirect(owner)
            if isinstance(owner, Stream):
                try:
                    for operands, operator in pikepdf.parse_content_stream(owner):
                        op_name = str(operator)
                        if op_name in _CMYK_OPERATORS:
                            has_cmyk = True
                        elif op_name in _RGB_OPERATORS:
                            has_rgb = True
                        elif op_name in _GRAY_OPERATORS:
                            has_gray = True
                        elif op_name in _CS_OPERATORS and operands:
                            cs_name = operands[0]
                            if cs_name == Name.DeviceCMYK:
                                has_cmyk = True
                            elif cs_name == Name.DeviceRGB:
                                has_rgb = True
                            elif cs_name == Name.DeviceGray:
                                has_gray = True
                except (
                    pikepdf.PdfError,
                    AttributeError,
                    IndexError,
                    TypeError,
                ):
                    pass

            resources = _resolve_indirect(resources)
            if not isinstance(resources, Dictionary):
                continue
            xobjects = _resolve_indirect(resources.get(Name.XObject))
            if not isinstance(xobjects, Dictionary):
                continue
            for key in list(xobjects.keys()):
                try:
                    image = _resolve_indirect(xobjects[key])
                    if image.get(Name.Subtype) != Name.Image:
                        continue
                    cs_type = _DEVICE_NAME_TO_TYPE.get(image.get(Name.ColorSpace))
                    if cs_type == ColorSpaceType.DEVICE_CMYK:
                        has_cmyk = True
                    elif cs_type == ColorSpaceType.DEVICE_RGB:
                        has_rgb = True
                    elif cs_type == ColorSpaceType.DEVICE_GRAY:
                        has_gray = True
                except (AttributeError, TypeError, ValueError):
                    continue
    except (pikepdf.PdfError, AttributeError, TypeError, ValueError):
        pass

    if has_cmyk:
        return ColorSpaceType.DEVICE_CMYK
    if has_rgb:
        return ColorSpaceType.DEVICE_RGB
    if has_gray:
        return ColorSpaceType.DEVICE_GRAY
    return ColorSpaceType.DEVICE_RGB


def _add_missing_transparency_groups(
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Add /Group to pages that use transparency but lack one.

    ISO 19005-2 rule 6.2.10-2 requires pages with transparency to have
    a /Group entry with a /CS for the blending color space.

    Args:
        pdf: The document being converted.
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        Number of pages where /Group was added.
    """
    added = 0

    for page in pdf.pages:
        # Skip pages that already have /Group
        if page.get(Name.Group) is not None:
            continue

        # Skip pages without transparency
        if not _page_uses_transparency(page):
            continue

        # Detect dominant color space for this page
        cs_type = _detect_page_dominant_cs(page)

        # Create /Group with ICCBased /CS
        icc_cs = _create_icc_colorspace(pdf, cs_type, icc_stream_cache)
        page[Name.Group] = Dictionary(
            S=Name.Transparency,
            CS=icc_cs,
        )
        added += 1
        logger.debug(
            "Added /Group with %s /CS to page (rule 6.2.10-2)",
            cs_type.value,
        )

    return added


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
    """Fix transparency group /CS entries in nested Form XObjects."""
    fixed = 0
    stack = [xobjects]

    while stack:
        try:
            current = _resolve_indirect(stack.pop())
            for name in current.keys():
                try:
                    xobject = _resolve_indirect(current[name])
                    objgen = xobject.objgen
                    if objgen != (0, 0):
                        if objgen in visited:
                            continue
                        visited.add(objgen)

                    if xobject.get(Name.Subtype) != Name.Form:
                        continue

                    fixed += _fix_transparency_group_cs_in_form(
                        xobject,
                        pdf,
                        icc_stream_cache,
                    )

                    resources = xobject.get(Name.Resources)
                    if resources is not None:
                        resources = _resolve_indirect(resources)
                        nested = resources.get(Name.XObject)
                        if nested is not None:
                            stack.append(nested)
                except (AttributeError, TypeError, ValueError):
                    continue
        except (AttributeError, TypeError, ValueError):
            continue

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

        for owner, _resources in _iter_content_streams_with_resources(page):
            owner = _resolve_indirect(owner)
            if isinstance(owner, Stream) and owner.get(Name.Subtype) == Name.Form:
                fixed += _fix_transparency_group_cs_in_form(
                    owner,
                    pdf,
                    icc_stream_cache,
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
