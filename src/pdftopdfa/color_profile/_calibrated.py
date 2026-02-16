# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""CalGray/CalRGB -> ICCBased conversion."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect
from ._profiles import _create_icc_colorspace
from ._types import ColorSpaceType

logger = logging.getLogger(__name__)

_CAL_TYPE_MAP: dict[str, ColorSpaceType] = {
    "/CalGray": ColorSpaceType.CAL_GRAY,
    "/CalRGB": ColorSpaceType.CAL_RGB,
}


def _log_calibration_params(cs) -> None:
    """Log the original calibration parameters before replacement."""
    try:
        tag = str(cs[0])
        cal_dict = _resolve_indirect(cs[1])
        if not isinstance(cal_dict, Dictionary):
            return

        params = []
        wp = cal_dict.get("/WhitePoint")
        if wp is not None:
            params.append(f"WhitePoint={list(wp)}")
        bp = cal_dict.get("/BlackPoint")
        if bp is not None:
            params.append(f"BlackPoint={list(bp)}")
        gamma = cal_dict.get("/Gamma")
        if gamma is not None:
            if isinstance(gamma, Array):
                params.append(f"Gamma={list(gamma)}")
            else:
                params.append(f"Gamma={float(gamma)}")
        matrix = cal_dict.get("/Matrix")
        if matrix is not None:
            params.append(f"Matrix={list(matrix)}")

        logger.info(
            "Replacing %s with ICCBased (original calibration: %s)"
            " — calibration-specific rendering may change",
            tag.lstrip("/"),
            ", ".join(params) if params else "none",
        )
    except Exception:
        pass


def _replace_cal_colorspace(
    cs,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
) -> Array | None:
    """Return an ICCBased replacement if *cs* is a CalGray/CalRGB array.

    Returns ``None`` when no replacement is needed (not a Cal* array, or the
    corresponding ICC array is not provided).
    """
    if not isinstance(cs, Array) or len(cs) < 2:
        return None
    try:
        tag = str(cs[0])
    except Exception:
        return None
    if tag == "/CalGray" and cal_gray_icc is not None:
        _log_calibration_params(cs)
        return cal_gray_icc
    if tag == "/CalRGB" and cal_rgb_icc is not None:
        _log_calibration_params(cs)
        return cal_rgb_icc
    return None


def _replace_cal_in_colorspace_dict(
    cs_dict,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
) -> int:
    """Replace CalGray/CalRGB entries in a named ColorSpace dictionary."""
    if cs_dict is None:
        return 0

    replaced = 0
    cs_dict = _resolve_indirect(cs_dict)

    for name in cs_dict.keys():
        try:
            cs = _resolve_indirect(cs_dict[name])
            repl = _replace_cal_colorspace(cs, cal_gray_icc, cal_rgb_icc)
            if repl is not None:
                cs_dict[name] = repl
                replaced += 1
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    return replaced


def _replace_cal_in_xobjects(
    xobjects,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB in Image and Form XObjects (recursive)."""
    if xobjects is None:
        return 0

    replaced = 0

    for name in xobjects.keys():
        xobj = _resolve_indirect(xobjects[name])

        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        subtype = xobj.get(Name.Subtype)

        if subtype == Name.Image:
            cs = xobj.get(Name.ColorSpace)
            if cs is not None:
                cs = _resolve_indirect(cs)
                repl = _replace_cal_colorspace(cs, cal_gray_icc, cal_rgb_icc)
                if repl is not None:
                    xobj[Name.ColorSpace] = repl
                    replaced += 1
                elif isinstance(cs, Array) and len(cs) >= 2:
                    if cs[0] == Name.Indexed:
                        base = _resolve_indirect(cs[1])
                        repl = _replace_cal_colorspace(base, cal_gray_icc, cal_rgb_icc)
                        if repl is not None:
                            cs[1] = repl
                            replaced += 1

        elif subtype == Name.Form:
            replaced += _replace_cal_in_group_cs(xobj, cal_gray_icc, cal_rgb_icc)
            replaced += _replace_cal_in_form_resources(
                xobj, cal_gray_icc, cal_rgb_icc, visited
            )

    return replaced


def _replace_cal_in_form_resources(
    form_xobj,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB throughout a Form XObject's resources."""
    resources = form_xobj.get(Name.Resources)
    if resources is None:
        return 0

    resources = _resolve_indirect(resources)
    return _replace_cal_in_resources(resources, cal_gray_icc, cal_rgb_icc, visited)


def _replace_cal_in_resources(
    resources,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB in all resource sub-dictionaries."""
    replaced = 0

    cs_dict = resources.get("/ColorSpace")
    if cs_dict:
        replaced += _replace_cal_in_colorspace_dict(cs_dict, cal_gray_icc, cal_rgb_icc)

    xobjects = resources.get(Name.XObject)
    if xobjects:
        replaced += _replace_cal_in_xobjects(
            xobjects, cal_gray_icc, cal_rgb_icc, visited
        )

    patterns = resources.get("/Pattern")
    if patterns:
        replaced += _replace_cal_in_patterns(
            patterns, cal_gray_icc, cal_rgb_icc, visited
        )

    shadings = resources.get("/Shading")
    if shadings:
        replaced += _replace_cal_in_shadings(
            shadings, cal_gray_icc, cal_rgb_icc, visited
        )

    replaced += _replace_cal_in_type3_fonts(
        resources, cal_gray_icc, cal_rgb_icc, visited
    )

    return replaced


def _replace_cal_in_shadings(
    shadings,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB color spaces in Shading dictionaries."""
    if shadings is None:
        return 0

    replaced = 0
    shadings = _resolve_indirect(shadings)

    for name in shadings.keys():
        try:
            shading = _resolve_indirect(shadings[name])

            objgen = shading.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            cs = shading.get(Name.ColorSpace)
            if cs is not None:
                cs = _resolve_indirect(cs)
                repl = _replace_cal_colorspace(cs, cal_gray_icc, cal_rgb_icc)
                if repl is not None:
                    shading[Name.ColorSpace] = repl
                    replaced += 1
                elif isinstance(cs, Array) and len(cs) >= 2:
                    if cs[0] == Name.Indexed:
                        base = _resolve_indirect(cs[1])
                        repl = _replace_cal_colorspace(base, cal_gray_icc, cal_rgb_icc)
                        if repl is not None:
                            cs[1] = repl
                            replaced += 1
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error replacing Cal* in shading %s: %s", name, e)

    return replaced


def _replace_cal_in_patterns(
    patterns,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB in Pattern resources."""
    patterns = _resolve_indirect(patterns)
    replaced = 0

    for name in patterns.keys():
        try:
            pattern = _resolve_indirect(patterns[name])

            obj_key = pattern.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            pattern_type = pattern.get("/PatternType")

            if pattern_type == 1:
                # Tiling pattern: recurse into resources
                pat_resources = pattern.get("/Resources")
                if pat_resources is not None:
                    pat_resources = _resolve_indirect(pat_resources)
                    replaced += _replace_cal_in_resources(
                        pat_resources, cal_gray_icc, cal_rgb_icc, visited
                    )

            elif pattern_type == 2:
                # Shading pattern: replace in /Shading/ColorSpace
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
                        repl = _replace_cal_colorspace(cs, cal_gray_icc, cal_rgb_icc)
                        if repl is not None:
                            shading[Name.ColorSpace] = repl
                            replaced += 1
                        elif isinstance(cs, Array) and len(cs) >= 2:
                            if cs[0] == Name.Indexed:
                                base = _resolve_indirect(cs[1])
                                repl = _replace_cal_colorspace(
                                    base, cal_gray_icc, cal_rgb_icc
                                )
                                if repl is not None:
                                    cs[1] = repl
                                    replaced += 1

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error replacing Cal* in pattern %s: %s", name, e)

    return replaced


def _replace_cal_in_type3_fonts(
    resources,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB in Type3 font resources."""
    replaced = 0

    for _font_name, font in _iter_type3_fonts(resources, visited):
        font_resources = font.get("/Resources")
        if font_resources is None:
            continue
        font_resources = _resolve_indirect(font_resources)
        replaced += _replace_cal_in_resources(
            font_resources, cal_gray_icc, cal_rgb_icc, visited
        )

    return replaced


def _replace_cal_in_group_cs(
    form_or_page,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
) -> int:
    """Replace CalGray/CalRGB in a transparency group /CS entry."""
    group = form_or_page.get(Name.Group)
    if group is None:
        return 0

    group = _resolve_indirect(group)
    if group.get(Name.S) != Name.Transparency:
        return 0

    cs = group.get(Name.CS)
    if cs is None:
        return 0

    cs = _resolve_indirect(cs)
    repl = _replace_cal_colorspace(cs, cal_gray_icc, cal_rgb_icc)
    if repl is not None:
        group[Name.CS] = repl
        return 1

    return 0


def _replace_cal_in_ap_entry(
    ap_value,
    cal_gray_icc: Array | None,
    cal_rgb_icc: Array | None,
    visited: set[tuple[int, int]],
) -> int:
    """Replace CalGray/CalRGB in annotation appearance streams."""
    replaced = 0
    ap_value = _resolve_indirect(ap_value)

    if isinstance(ap_value, Stream):
        objgen = ap_value.objgen
        if objgen != (0, 0):
            if objgen in visited:
                return 0
            visited.add(objgen)

        replaced += _replace_cal_in_group_cs(ap_value, cal_gray_icc, cal_rgb_icc)
        replaced += _replace_cal_in_form_resources(
            ap_value, cal_gray_icc, cal_rgb_icc, visited
        )

    elif isinstance(ap_value, Dictionary):
        for key in ap_value.keys():
            sub_stream = _resolve_indirect(ap_value[key])
            if isinstance(sub_stream, Stream):
                objgen = sub_stream.objgen
                if objgen != (0, 0):
                    if objgen in visited:
                        continue
                    visited.add(objgen)

                replaced += _replace_cal_in_group_cs(
                    sub_stream, cal_gray_icc, cal_rgb_icc
                )
                replaced += _replace_cal_in_form_resources(
                    sub_stream, cal_gray_icc, cal_rgb_icc, visited
                )

    return replaced


def _convert_calibrated_colorspaces(
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Convert CalGray/CalRGB color spaces to ICCBased throughout the PDF.

    Iterates all pages and replaces CalGray/CalRGB arrays with ICCBased
    equivalents in Resources (ColorSpace dict, XObjects, Patterns, Shadings,
    Type3 fonts), transparency groups, and annotation appearance streams.

    Lab is intentionally skipped - it is already PDF/A-conformant and there
    is no bundled Lab ICC profile.

    Args:
        pdf: The document being converted.
        icc_stream_cache: Shared cache of ICC stream objects.

    Returns:
        Total number of Cal* color spaces replaced.
    """
    cal_gray_icc = _create_icc_colorspace(
        pdf, ColorSpaceType.CAL_GRAY, icc_stream_cache
    )
    cal_rgb_icc = _create_icc_colorspace(pdf, ColorSpaceType.CAL_RGB, icc_stream_cache)

    replaced = 0
    visited: set[tuple[int, int]] = set()

    for page in pdf.pages:
        # Page-level transparency group
        replaced += _replace_cal_in_group_cs(page, cal_gray_icc, cal_rgb_icc)

        resources = page.get(Name.Resources)
        if resources is not None:
            resources = _resolve_indirect(resources)
            replaced += _replace_cal_in_resources(
                resources, cal_gray_icc, cal_rgb_icc, visited
            )

        # Annotation appearance streams
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
                        replaced += _replace_cal_in_ap_entry(
                            ap_entry, cal_gray_icc, cal_rgb_icc, visited
                        )
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                logger.debug("Error replacing Cal* in annotation AP: %s", e)

    if replaced > 0:
        logger.info("CalGray/CalRGB -> ICCBased replacements: %d", replaced)

    return replaced
