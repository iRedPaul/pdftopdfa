# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Default color space application for PDF/A compliance."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _create_icc_colorspace
from ._types import _DEFAULT_CS_NAMES, _DEVICE_CS_NAMES, ColorSpaceType

logger = logging.getLogger(__name__)


def _add_default_colorspaces(
    resources,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
) -> int:
    """Add Default color space entries to a resource dictionary.

    Inserts ``DefaultGray``, ``DefaultRGB``, or ``DefaultCMYK`` into
    ``resources[/ColorSpace]`` for each *non_dominant* Device space.
    Existing entries are never overwritten.

    Args:
        resources: pikepdf Dictionary (page or Form XObject resources).
        non_dominant: Set of Device color space types needing a Default.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.

    Returns:
        Number of Default entries actually added.
    """
    if not non_dominant:
        return 0

    cs_dict = resources.get(Name.ColorSpace)
    if cs_dict is None:
        cs_dict = Dictionary()
        resources[Name.ColorSpace] = cs_dict
    else:
        cs_dict = _resolve_indirect(cs_dict)

    added = 0
    for cs_type in non_dominant:
        default_name = _DEFAULT_CS_NAMES[cs_type]
        if default_name not in cs_dict:
            cs_dict[default_name] = icc_arrays[cs_type]
            added += 1

    return added


def _replace_device_colorspace_in_images(
    xobjects,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> int:
    """Replace Device color spaces in Image XObjects with ICCBased.

    Default color spaces do NOT apply to Image XObjects (PDF spec 8.6.5.6),
    so images must be fixed individually.  Form XObjects are recursed into:
    their resources get Default entries and nested images are replaced.

    Args:
        xobjects: ``/XObject`` Dictionary from resources.
        non_dominant: Device color space types to replace.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        Number of images whose color space was replaced.
    """
    if xobjects is None:
        return 0

    replaced = 0

    for name in xobjects.keys():
        xobj = _resolve_indirect(xobjects[name])

        # Cycle detection using objgen (safe for pikepdf, see MEMORY.md)
        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        subtype = xobj.get(Name.Subtype)

        if subtype == Name.Image:
            # Only replace bare Device color space names (e.g. /DeviceRGB).
            # Separation and DeviceN arrays are intentionally left untouched:
            # their alternate spaces are resolved by the PDF viewer, and they
            # are PDF/A-conformant with an OutputIntent (ISO 19005-2, 6.2.4.4).
            cs = xobj.get(Name.ColorSpace)
            if isinstance(cs, Name):
                for cs_type in non_dominant:
                    if cs == _DEVICE_CS_NAMES[cs_type]:
                        xobj[Name.ColorSpace] = icc_arrays[cs_type]
                        replaced += 1
                        break
            elif isinstance(cs, Array) and len(cs) >= 2:
                if cs[0] == Name.Indexed:
                    base = cs[1]
                    if isinstance(base, Name):
                        for cs_type in non_dominant:
                            if base == _DEVICE_CS_NAMES[cs_type]:
                                cs[1] = icc_arrays[cs_type]
                                replaced += 1
                                break

        elif subtype == Name.Form:
            form_resources = xobj.get(Name.Resources)
            if form_resources is not None:
                form_resources = _resolve_indirect(form_resources)
                # Add Default color spaces to form resources
                _add_default_colorspaces(form_resources, non_dominant, icc_arrays)
                # Recurse into nested XObjects
                nested = form_resources.get(Name.XObject)
                if nested:
                    replaced += _replace_device_colorspace_in_images(
                        nested, non_dominant, icc_arrays, visited
                    )
                # Process patterns in form resources
                form_patterns = form_resources.get("/Pattern")
                if form_patterns:
                    _d, _r = _apply_defaults_to_patterns(
                        form_patterns, non_dominant, icc_arrays, visited
                    )
                    replaced += _r
                # Process shadings in form resources
                form_shadings = form_resources.get("/Shading")
                if form_shadings:
                    replaced += _replace_device_colorspace_in_shadings(
                        form_shadings, non_dominant, icc_arrays, visited
                    )
                # Process Type3 font resources in form
                replaced += _apply_defaults_to_type3_fonts(
                    form_resources, non_dominant, icc_arrays, visited
                )

    return replaced


def _replace_device_colorspace_in_shadings(
    shadings,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> int:
    """Replace bare Device color spaces in Shading dictionaries.

    Default color spaces do NOT apply to explicit ``/ColorSpace`` entries
    in Shading dictionaries (PDF Spec 8.6.5.6), so non-dominant Device
    color spaces must be replaced directly with ICCBased arrays.

    Separation/DeviceN arrays are naturally skipped by the
    ``isinstance(cs, Name)`` check - they are PDF/A-conformant when an
    OutputIntent is present (ISO 19005-2, 6.2.4.4).

    Args:
        shadings: ``/Shading`` Dictionary from resources.
        non_dominant: Device color space types to replace.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        Number of shadings whose color space was replaced.
    """
    if shadings is None:
        return 0

    replaced = 0
    shadings = _resolve_indirect(shadings)

    for name in shadings.keys():
        try:
            shading = _resolve_indirect(shadings[name])

            # Cycle detection using objgen (safe for pikepdf, see MEMORY.md)
            objgen = shading.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            cs = shading.get(Name.ColorSpace)
            if cs is not None:
                cs = _resolve_indirect(cs)
                if isinstance(cs, Name):
                    for cs_type in non_dominant:
                        if cs == _DEVICE_CS_NAMES[cs_type]:
                            shading[Name.ColorSpace] = icc_arrays[cs_type]
                            replaced += 1
                            break
                elif isinstance(cs, Array) and len(cs) >= 2:
                    if cs[0] == Name.Indexed:
                        base = cs[1]
                        if isinstance(base, Name):
                            for cs_type in non_dominant:
                                if base == _DEVICE_CS_NAMES[cs_type]:
                                    cs[1] = icc_arrays[cs_type]
                                    replaced += 1
                                    break
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error replacing colorspace in shading %s: %s", name, e)

    return replaced


def _apply_defaults_to_type3_fonts(
    resources,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> int:
    """Apply default color spaces to Type3 font resources.

    Creates ``/Resources`` on the font if missing, inserts DefaultGray/
    DefaultRGB/DefaultCMYK, and replaces Device color spaces in Image
    XObjects, Shading dictionaries, and Patterns within the font
    resources.

    Defaults do **not** apply to explicit ``/ColorSpace`` entries in
    Shading dictionaries (PDF Spec 8.6.5.6) or to Tiling-Pattern
    resources (which don't inherit parent defaults), so these must be
    replaced directly.

    Args:
        resources: A resolved Resources dictionary (page or form).
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` tuples for cycle detection.

    Returns:
        Number of replaced color space entries.
    """
    replaced = 0

    for _font_name, font in _iter_type3_fonts(resources, visited):
        font_resources = font.get("/Resources")
        if font_resources is None:
            font_resources = Dictionary()
            font[Name("/Resources")] = font_resources
        else:
            font_resources = _resolve_indirect(font_resources)

        _add_default_colorspaces(font_resources, non_dominant, icc_arrays)

        # Replace Device color spaces in Image XObjects
        font_xobjects = font_resources.get(Name.XObject)
        if font_xobjects:
            replaced += _replace_device_colorspace_in_images(
                font_xobjects, non_dominant, icc_arrays, visited
            )

        # Replace Device color spaces in Shading dictionaries
        font_shadings = font_resources.get("/Shading")
        if font_shadings:
            replaced += _replace_device_colorspace_in_shadings(
                font_shadings, non_dominant, icc_arrays, visited
            )

        # Replace Device color spaces in Patterns
        font_patterns = font_resources.get("/Pattern")
        if font_patterns:
            d, r = _apply_defaults_to_patterns(
                font_patterns, non_dominant, icc_arrays, visited
            )
            replaced += d + r

    return replaced


def _apply_defaults_to_patterns(
    patterns,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> tuple[int, int]:
    """Apply default color spaces and ICC replacements to patterns.

    Tiling patterns (PatternType=1) have their own ``/Resources`` and do
    NOT inherit Default color space entries from their parent page or
    Form XObject.  This function adds the necessary Default entries and
    recurses into nested XObjects, patterns, and shadings.

    PatternType=2 (Shading) patterns have their ``/Shading/ColorSpace``
    replaced directly when it is a bare Device color space - Defaults do
    not apply to explicit ``/ColorSpace`` entries in shadings.

    Args:
        patterns: ``/Pattern`` Dictionary from resources.
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        ``(defaults_added, images_replaced)`` counts.
    """
    patterns = _resolve_indirect(patterns)

    defaults_added = 0
    images_replaced = 0

    for name in patterns.keys():
        try:
            pattern = _resolve_indirect(patterns[name])

            # Cycle detection using objgen (safe for pikepdf, see MEMORY.md)
            obj_key = pattern.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            pattern_type = pattern.get("/PatternType")

            if pattern_type == 1:
                # Tiling pattern: has content stream and own resources
                pat_resources = pattern.get("/Resources")
                if pat_resources is None:
                    pat_resources = Dictionary()
                    pattern[Name.Resources] = pat_resources
                else:
                    pat_resources = _resolve_indirect(pat_resources)
                defaults_added += _add_default_colorspaces(
                    pat_resources, non_dominant, icc_arrays
                )
                # Recurse into nested XObjects
                nested = pat_resources.get(Name.XObject)
                if nested:
                    images_replaced += _replace_device_colorspace_in_images(
                        nested, non_dominant, icc_arrays, visited
                    )
                # Recurse into nested patterns
                nested_patterns = pat_resources.get("/Pattern")
                if nested_patterns:
                    d, r = _apply_defaults_to_patterns(
                        nested_patterns,
                        non_dominant,
                        icc_arrays,
                        visited,
                    )
                    defaults_added += d
                    images_replaced += r
                # Process shadings in tiling pattern resources
                pat_shadings = pat_resources.get("/Shading")
                if pat_shadings:
                    images_replaced += _replace_device_colorspace_in_shadings(
                        pat_shadings, non_dominant, icc_arrays, visited
                    )

            elif pattern_type == 2:
                # Shading pattern: replace Device color space in the
                # shading's /ColorSpace directly (Defaults don't apply).
                shading = pattern.get("/Shading")
                if shading is not None:
                    shading = _resolve_indirect(shading)
                    sh_objgen = shading.objgen
                    if sh_objgen != (0, 0):
                        if sh_objgen in visited:
                            continue
                        visited.add(sh_objgen)
                    cs = shading.get(Name.ColorSpace)
                    if cs is not None:
                        cs = _resolve_indirect(cs)
                        if isinstance(cs, Name):
                            for cs_type in non_dominant:
                                if cs == _DEVICE_CS_NAMES[cs_type]:
                                    shading[Name.ColorSpace] = icc_arrays[cs_type]
                                    images_replaced += 1
                                    break
                        elif isinstance(cs, Array) and len(cs) >= 2:
                            if cs[0] == Name.Indexed:
                                base = cs[1]
                                if isinstance(base, Name):
                                    for cs_type in non_dominant:
                                        if base == _DEVICE_CS_NAMES[cs_type]:
                                            cs[1] = icc_arrays[cs_type]
                                            images_replaced += 1
                                            break

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error applying defaults to pattern %s: %s", name, e)

    return defaults_added, images_replaced


def _apply_defaults_to_ap_entry(
    ap_value,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> tuple[int, int]:
    """Apply default color spaces and ICC replacements to an AP entry.

    The entry may be a Form XObject (stream) directly, or a dictionary
    of sub-state Form XObjects (e.g. On/Off for checkboxes).

    Args:
        ap_value: The value of an AP entry (/N, /R, or /D).
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        ``(defaults_added, images_replaced)`` counts.
    """
    defaults_added = 0
    images_replaced = 0
    ap_value = _resolve_indirect(ap_value)

    if isinstance(ap_value, Stream):
        objgen = ap_value.objgen
        if objgen != (0, 0):
            if objgen in visited:
                return 0, 0
            visited.add(objgen)

        ap_resources = ap_value.get(Name.Resources)
        if ap_resources is None:
            ap_resources = Dictionary()
            ap_value[Name.Resources] = ap_resources
        else:
            ap_resources = _resolve_indirect(ap_resources)
        defaults_added += _add_default_colorspaces(
            ap_resources, non_dominant, icc_arrays
        )
        nested = ap_resources.get(Name.XObject)
        if nested:
            images_replaced += _replace_device_colorspace_in_images(
                nested, non_dominant, icc_arrays, visited
            )
        ap_patterns = ap_resources.get("/Pattern")
        if ap_patterns:
            d, r = _apply_defaults_to_patterns(
                ap_patterns, non_dominant, icc_arrays, visited
            )
            defaults_added += d
            images_replaced += r
        ap_shadings = ap_resources.get("/Shading")
        if ap_shadings:
            images_replaced += _replace_device_colorspace_in_shadings(
                ap_shadings, non_dominant, icc_arrays, visited
            )
        images_replaced += _apply_defaults_to_type3_fonts(
            ap_resources, non_dominant, icc_arrays, visited
        )

    elif isinstance(ap_value, Dictionary):
        # Sub-state dictionary (e.g. /Yes, /Off): each value is a stream
        for key in ap_value.keys():
            sub_stream = _resolve_indirect(ap_value[key])
            if isinstance(sub_stream, Stream):
                objgen = sub_stream.objgen
                if objgen != (0, 0):
                    if objgen in visited:
                        continue
                    visited.add(objgen)

                sub_resources = sub_stream.get(Name.Resources)
                if sub_resources is None:
                    sub_resources = Dictionary()
                    sub_stream[Name.Resources] = sub_resources
                else:
                    sub_resources = _resolve_indirect(sub_resources)
                defaults_added += _add_default_colorspaces(
                    sub_resources, non_dominant, icc_arrays
                )
                nested = sub_resources.get(Name.XObject)
                if nested:
                    images_replaced += _replace_device_colorspace_in_images(
                        nested, non_dominant, icc_arrays, visited
                    )
                sub_patterns = sub_resources.get("/Pattern")
                if sub_patterns:
                    d, r = _apply_defaults_to_patterns(
                        sub_patterns,
                        non_dominant,
                        icc_arrays,
                        visited,
                    )
                    defaults_added += d
                    images_replaced += r
                sub_shadings = sub_resources.get("/Shading")
                if sub_shadings:
                    images_replaced += _replace_device_colorspace_in_shadings(
                        sub_shadings, non_dominant, icc_arrays, visited
                    )
                images_replaced += _apply_defaults_to_type3_fonts(
                    sub_resources, non_dominant, icc_arrays, visited
                )

    return defaults_added, images_replaced


def _apply_defaults_to_smask_groups(
    resources,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> tuple[int, int]:
    """Apply default color spaces to SMask /G Form XObjects.

    ExtGState entries may reference an SMask dict whose ``/G`` value is
    a Form XObject (transparency group).  Default color spaces do not
    propagate into these groups automatically, so we must add them.

    Args:
        resources: A resolved Resources dictionary.
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        ``(defaults_added, images_replaced)`` counts.
    """
    defaults_added = 0
    images_replaced = 0

    ext_gstate = resources.get("/ExtGState")
    if ext_gstate is None:
        return 0, 0
    ext_gstate = _resolve_indirect(ext_gstate)
    if not isinstance(ext_gstate, Dictionary):
        return 0, 0

    for gs_name in ext_gstate.keys():
        try:
            gs = _resolve_indirect(ext_gstate[gs_name])
            if not isinstance(gs, Dictionary):
                continue
            smask = gs.get("/SMask")
            if smask is None:
                continue
            smask = _resolve_indirect(smask)
            if not isinstance(smask, Dictionary):
                continue
            g_form = smask.get("/G")
            if g_form is None:
                continue
            g_form = _resolve_indirect(g_form)
            if not isinstance(g_form, Stream):
                continue

            objgen = g_form.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            form_resources = g_form.get(Name.Resources)
            if form_resources is None:
                form_resources = Dictionary()
                g_form[Name.Resources] = form_resources
            else:
                form_resources = _resolve_indirect(form_resources)

            defaults_added += _add_default_colorspaces(
                form_resources, non_dominant, icc_arrays
            )

            nested = form_resources.get(Name.XObject)
            if nested:
                images_replaced += _replace_device_colorspace_in_images(
                    nested, non_dominant, icc_arrays, visited
                )

            nested_patterns = form_resources.get("/Pattern")
            if nested_patterns:
                d, r = _apply_defaults_to_patterns(
                    nested_patterns, non_dominant, icc_arrays, visited
                )
                defaults_added += d
                images_replaced += r

            nested_shadings = form_resources.get("/Shading")
            if nested_shadings:
                images_replaced += _replace_device_colorspace_in_shadings(
                    nested_shadings, non_dominant, icc_arrays, visited
                )

            images_replaced += _apply_defaults_to_type3_fonts(
                form_resources, non_dominant, icc_arrays, visited
            )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(
                "Error applying defaults to SMask /G for %s: %s",
                gs_name,
                e,
            )

    return defaults_added, images_replaced


def _apply_default_colorspaces(
    pdf: Pdf,
    non_dominant: set[ColorSpaceType],
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> tuple[int, int]:
    """Cover non-dominant Device color spaces for PDF/A compliance.

    For each non-dominant Device space:
    * Page and Form XObject resources get a ``DefaultXxx`` entry that maps
      Device color operators to an ICCBased profile.
    * Image XObjects get their ``/ColorSpace`` replaced with
      ``[/ICCBased <stream>]`` because Defaults do not apply to images.
    * Shading dictionaries get their ``/ColorSpace`` replaced directly
      because Defaults do not apply to explicit entries in shadings.

    Args:
        pdf: The document being converted.
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        ``(defaults_added, images_replaced)`` counts.
    """
    # Build reusable ICCBased arrays
    icc_arrays: dict[ColorSpaceType, Array] = {}
    for cs_type in non_dominant:
        icc_arrays[cs_type] = _create_icc_colorspace(pdf, cs_type, icc_stream_cache)

    defaults_added = 0
    images_replaced = 0
    visited: set[tuple[int, int]] = set()

    for page in pdf.pages:
        resources = page.get(Name.Resources)
        if resources is None:
            resources = Dictionary()
            page[Name.Resources] = resources
        else:
            resources = _resolve_indirect(resources)

        defaults_added += _add_default_colorspaces(resources, non_dominant, icc_arrays)

        xobjects = resources.get(Name.XObject)
        if xobjects:
            images_replaced += _replace_device_colorspace_in_images(
                xobjects, non_dominant, icc_arrays, visited
            )

        patterns = resources.get("/Pattern")
        if patterns:
            d, r = _apply_defaults_to_patterns(
                patterns, non_dominant, icc_arrays, visited
            )
            defaults_added += d
            images_replaced += r

        shadings = resources.get("/Shading")
        if shadings:
            images_replaced += _replace_device_colorspace_in_shadings(
                shadings, non_dominant, icc_arrays, visited
            )

        # Process Type3 font resources
        images_replaced += _apply_defaults_to_type3_fonts(
            resources, non_dominant, icc_arrays, visited
        )

        # Process SMask /G Form XObjects in ExtGState
        d, r = _apply_defaults_to_smask_groups(
            resources, non_dominant, icc_arrays, visited
        )
        defaults_added += d
        images_replaced += r

        # Process annotation appearance streams
        annots = page.get(Name.Annots)
        if annots is not None:
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
                            d, r = _apply_defaults_to_ap_entry(
                                ap_entry,
                                non_dominant,
                                icc_arrays,
                                visited,
                            )
                            defaults_added += d
                            images_replaced += r
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    logger.debug("Error processing annotation AP defaults: %s", e)

    logger.debug(
        "Default color spaces: %d added, %d images replaced",
        defaults_added,
        images_replaced,
    )
    return defaults_added, images_replaced
