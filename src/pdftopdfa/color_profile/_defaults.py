# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Default color space application for PDF/A compliance."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _create_icc_colorspace
from ._types import _DEFAULT_CS_NAMES, ColorSpaceType

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


def _apply_defaults_to_resource_graph(
    tasks: list[tuple[str, object]],
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> tuple[int, int]:
    """Apply defaults across a resource graph without changing its objects."""
    defaults_added = 0
    stack = list(tasks)

    while stack:
        kind, value = stack.pop()
        try:
            value = _resolve_indirect(value)

            if kind == "resources":
                if not isinstance(value, Dictionary):
                    continue
                defaults_added += _add_default_colorspaces(
                    value,
                    non_dominant,
                    icc_arrays,
                )
                for task_kind, key in (
                    ("smask", Name.ExtGState),
                    ("type3", Name.Font),
                    ("patterns", Name.Pattern),
                    ("xobjects", Name.XObject),
                ):
                    child = value.get(key)
                    if child is not None:
                        stack.append(
                            (
                                task_kind,
                                value if task_kind in {"smask", "type3"} else child,
                            )
                        )
                continue

            if kind == "xobjects":
                if not isinstance(value, Dictionary):
                    continue
                for name in reversed(list(value.keys())):
                    try:
                        xobject = _resolve_indirect(value[name])
                        if xobject.get(Name.Subtype) != Name.Form:
                            continue

                        objgen = xobject.objgen
                        if objgen != (0, 0):
                            if objgen in visited:
                                continue
                            visited.add(objgen)

                        resources = xobject.get(Name.Resources)
                        if resources is not None:
                            stack.append(("resources", resources))
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        logger.debug(
                            "Error applying defaults to XObject %s: %s",
                            name,
                            e,
                        )
                continue

            if kind == "patterns":
                if not isinstance(value, Dictionary):
                    continue
                for name in reversed(list(value.keys())):
                    try:
                        pattern = _resolve_indirect(value[name])
                        objgen = pattern.objgen
                        if objgen != (0, 0):
                            if objgen in visited:
                                continue
                            visited.add(objgen)

                        pattern_type = pattern.get("/PatternType")
                        if pattern_type == 1:
                            resources = pattern.get(Name.Resources)
                            if resources is None:
                                resources = Dictionary()
                                pattern[Name.Resources] = resources
                            stack.append(("resources", resources))
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        logger.debug(
                            "Error applying defaults to pattern %s: %s",
                            name,
                            e,
                        )
                continue

            if kind == "type3":
                if not isinstance(value, Dictionary):
                    continue
                for _font_name, font in _iter_type3_fonts(value, visited):
                    resources = font.get(Name.Resources)
                    if resources is None:
                        resources = Dictionary()
                        font[Name.Resources] = resources
                    stack.append(("resources", resources))
                continue

            if kind == "smask":
                if not isinstance(value, Dictionary):
                    continue
                ext_gstate = value.get(Name.ExtGState)
                if ext_gstate is None:
                    continue
                ext_gstate = _resolve_indirect(ext_gstate)
                if not isinstance(ext_gstate, Dictionary):
                    continue
                for name in reversed(list(ext_gstate.keys())):
                    try:
                        graphics_state = _resolve_indirect(ext_gstate[name])
                        soft_mask = graphics_state.get(Name.SMask)
                        if soft_mask is None:
                            continue
                        soft_mask = _resolve_indirect(soft_mask)
                        if not isinstance(soft_mask, Dictionary):
                            continue
                        group = soft_mask.get(Name.G)
                        if group is None:
                            continue
                        group = _resolve_indirect(group)
                        if not isinstance(group, Stream):
                            continue

                        objgen = group.objgen
                        if objgen != (0, 0):
                            if objgen in visited:
                                continue
                            visited.add(objgen)

                        resources = group.get(Name.Resources)
                        if resources is None:
                            resources = Dictionary()
                            group[Name.Resources] = resources
                        stack.append(("resources", resources))
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        logger.debug(
                            "Error applying defaults to SMask /G for %s: %s",
                            name,
                            e,
                        )
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Error applying defaults to %s graph: %s", kind, e)

    return defaults_added, 0


def _apply_defaults_to_ap_entry(
    ap_value,
    non_dominant: set[ColorSpaceType],
    icc_arrays: dict[ColorSpaceType, Array],
    visited: set[tuple[int, int]],
) -> tuple[int, int]:
    """Apply default color spaces to an AP entry.

    The entry may be a Form XObject (stream) directly, or a dictionary
    of sub-state Form XObjects (e.g. On/Off for checkboxes).

    Args:
        ap_value: The value of an AP entry (/N, /R, or /D).
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_arrays: Pre-built ``[/ICCBased <stream>]`` arrays.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.

    Returns:
        ``(defaults_added, colorspaces_replaced)`` counts. The second value
        is zero because Device color spaces remain context-dependent.
    """
    ap_value = _resolve_indirect(ap_value)
    streams: list[Stream] = []

    if isinstance(ap_value, Stream):
        streams.append(ap_value)
    elif isinstance(ap_value, Dictionary):
        for key in ap_value.keys():
            sub_stream = _resolve_indirect(ap_value[key])
            if isinstance(sub_stream, Stream):
                streams.append(sub_stream)

    tasks: list[tuple[str, object]] = []
    for stream in streams:
        objgen = stream.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)
        resources = stream.get(Name.Resources)
        if resources is None:
            resources = Dictionary()
            stream[Name.Resources] = resources
        tasks.append(("resources", resources))

    return _apply_defaults_to_resource_graph(
        tasks,
        non_dominant,
        icc_arrays,
        visited,
    )


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
        ``(defaults_added, colorspaces_replaced)`` counts. The second value
        is zero because Device color spaces remain context-dependent.
    """
    return _apply_defaults_to_resource_graph(
        [("smask", resources)],
        non_dominant,
        icc_arrays,
        visited,
    )


def _apply_default_colorspaces(
    pdf: Pdf,
    non_dominant: set[ColorSpaceType],
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> tuple[int, int]:
    """Cover non-dominant Device color spaces for PDF/A compliance.

    For each non-dominant Device space:
    * Every content-owning resource dictionary gets a ``DefaultXxx`` entry
      that maps the Device color space to an ICCBased profile.
    * Device names on images, shadings, and Indexed bases remain unchanged.
      PDF default color spaces apply to them in their current resource
      context, which also preserves shared objects used with different
      defaults.

    Args:
        pdf: The document being converted.
        non_dominant: Device spaces not covered by the OutputIntent.
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        ``(defaults_added, colorspaces_replaced)`` counts. The second value
        is zero because Device color spaces remain context-dependent.
    """
    # Build reusable ICCBased arrays
    icc_arrays: dict[ColorSpaceType, Array] = {}
    for cs_type in non_dominant:
        icc_arrays[cs_type] = _create_icc_colorspace(pdf, cs_type, icc_stream_cache)

    defaults_added = 0
    visited: set[tuple[int, int]] = set()

    for page in pdf.pages:
        resources = page.get(Name.Resources)
        if resources is None:
            resources = Dictionary()
            page[Name.Resources] = resources
        else:
            resources = _resolve_indirect(resources)

        added, _ = _apply_defaults_to_resource_graph(
            [("resources", resources)],
            non_dominant,
            icc_arrays,
            visited,
        )
        defaults_added += added

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
                            added, _ = _apply_defaults_to_ap_entry(
                                ap_entry,
                                non_dominant,
                                icc_arrays,
                                visited,
                            )
                            defaults_added += added
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    logger.debug("Error processing annotation AP defaults: %s", e)

    logger.debug("Default color spaces: %d added", defaults_added)
    return defaults_added, 0
