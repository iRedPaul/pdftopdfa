# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Color space detection and analysis."""

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect
from ._types import (
    _CMYK_OPERATORS,
    _CS_OPERATORS,
    _GRAY_OPERATORS,
    _INLINE_CS_TO_DEVICE,
    _RGB_OPERATORS,
    ColorSpaceAnalysis,
    SpecialColorSpace,
)

logger = logging.getLogger(__name__)


def _get_inline_image_device_cs(image) -> Name | None:
    """Return the device color space Name used by an inline image, or *None*.

    Tries ``image.colorspace`` first (handles most cases).  When that raises
    (Indexed, resource-name references, missing /CS), falls back to reading
    the raw ``/ColorSpace`` entry from the parsed inline-image dictionary.
    Indexed arrays are inspected for their base color space.
    """
    # Fast path - pikepdf resolves simple device color spaces reliably.
    try:
        return image.colorspace
    except (NotImplementedError, IndexError, AttributeError):
        pass

    # Fallback: read the raw /ColorSpace value from the inline-image dict.
    # pikepdf normalises *keys* (e.g. /CS -> /ColorSpace) but not all values
    # inside arrays.
    try:
        raw_cs = image.obj.get(Name.ColorSpace)
    except Exception:
        return None

    if raw_cs is None:
        return None

    # Simple Name - either a device name, an abbreviation, or a resource ref.
    if isinstance(raw_cs, Name):
        mapped = _INLINE_CS_TO_DEVICE.get(str(raw_cs))
        if mapped is not None:
            return mapped
        logger.debug("Inline image /ColorSpace is a resource reference: %s", raw_cs)
        return None

    # Array - e.g. [/Indexed base hival lookup] or [/I /G 1 <...>]
    try:
        if len(raw_cs) >= 2:
            cs_type = str(raw_cs[0])
            if cs_type in ("/Indexed", "/I"):
                base = raw_cs[1]
                if isinstance(base, Name):
                    mapped = _INLINE_CS_TO_DEVICE.get(str(base))
                    if mapped is not None:
                        return mapped
                if isinstance(base, Array) and len(base) >= 1:
                    # Recursively resolve array base (e.g. [/ICCBased stream])
                    base_type, _ = _parse_colorspace_array(base)
                    if base_type == "ICCBased" and len(base) >= 2:
                        icc_stream = _resolve_indirect(base[1])
                        try:
                            n = int(icc_stream.get("/N", 0))
                        except Exception:
                            n = 0
                        if n == 1:
                            return Name.DeviceGray
                        elif n == 3:
                            return Name.DeviceRGB
                        elif n == 4:
                            return Name.DeviceCMYK
                    elif base_type in ("DeviceGray", "CalGray"):
                        return Name.DeviceGray
                    elif base_type in ("DeviceRGB", "CalRGB"):
                        return Name.DeviceRGB
                    elif base_type == "DeviceCMYK":
                        return Name.DeviceCMYK
                if isinstance(base, Name):
                    logger.debug(
                        "Indexed inline image base color space not a device CS: %s",
                        str(base),
                    )
    except Exception:
        pass

    return None


def _parse_colorspace_array(cs) -> tuple[str | None, str | None]:
    """
    Parse a color space array and return type and alternate space.

    Handles:
    - Simple names: DeviceGray, DeviceRGB, DeviceCMYK
    - Separation: [/Separation name alternate tint]
    - DeviceN: [/DeviceN [names] alternate tint attrs?]
    - Indexed: [/Indexed base hival lookup]
    - ICCBased: [/ICCBased stream]

    Args:
        cs: Color space object (Name or Array).

    Returns:
        Tuple of (color_space_type, alternate_space).
        For simple types, alternate_space is None.
    """
    if cs is None:
        return None, None

    # Handle simple Name types
    if isinstance(cs, Name):
        cs_str = str(cs)
        if cs_str == "/DeviceGray":
            return "DeviceGray", None
        elif cs_str == "/DeviceRGB":
            return "DeviceRGB", None
        elif cs_str == "/DeviceCMYK":
            return "DeviceCMYK", None
        return cs_str.lstrip("/"), None

    # Handle Array types
    if isinstance(cs, Array) and len(cs) > 0:
        cs_type = str(cs[0])

        if cs_type == "/Separation" and len(cs) >= 3:
            # [/Separation colorName alternateSpace tintTransform]
            alternate = cs[2]
            if isinstance(alternate, Name):
                return "Separation", str(alternate).lstrip("/")
            elif isinstance(alternate, Array) and len(alternate) > 0:
                # Alternate space is also an array (e.g., ICCBased)
                return "Separation", str(alternate[0]).lstrip("/")
            return "Separation", None

        elif cs_type == "/DeviceN" and len(cs) >= 3:
            # [/DeviceN [colorNames] alternateSpace tintTransform attrs?]
            alternate = cs[2]
            if isinstance(alternate, Name):
                return "DeviceN", str(alternate).lstrip("/")
            elif isinstance(alternate, Array) and len(alternate) > 0:
                return "DeviceN", str(alternate[0]).lstrip("/")
            return "DeviceN", None

        elif cs_type == "/Indexed" and len(cs) >= 2:
            # [/Indexed baseColorSpace hival lookup]
            base = cs[1]
            base_type, base_alt = _parse_colorspace_array(base)
            return "Indexed", base_type

        elif cs_type == "/ICCBased":
            return "ICCBased", None

        elif cs_type in ("/CalGray", "/CalRGB", "/Lab"):
            return cs_type.lstrip("/"), None

    return None, None


def _analyze_colorspace(
    cs,
    analysis: ColorSpaceAnalysis,
    location: str,
    obj_ref: object | None = None,
) -> None:
    """
    Analyze a color space and update the analysis.

    Args:
        cs: Color space object (Name or Array).
        analysis: ColorSpaceAnalysis to update.
        location: Description of where this color space was found.
        obj_ref: Reference to the object containing this color space.
    """
    cs_type, alternate = _parse_colorspace_array(cs)

    if cs_type is None:
        return

    # Track simple device color spaces
    if cs_type == "DeviceGray":
        analysis.device_gray_used = True
    elif cs_type == "DeviceRGB":
        analysis.device_rgb_used = True
    elif cs_type == "DeviceCMYK":
        analysis.device_cmyk_used = True

    # Track special color spaces
    elif cs_type == "Separation":
        analysis.separation_used = True
        analysis.special_colorspaces.append(
            SpecialColorSpace(
                type="Separation",
                alternate_space=alternate or "unknown",
                location=location,
                obj_ref=obj_ref,
            )
        )
        # Also track the alternate space as a used device color space.
        # Note: For Separation/DeviceN in Image XObjects, the alternate space
        # is resolved by the PDF viewer at render time - Default color spaces
        # do NOT apply to images (PDF spec 8.6.5.6).  This is PDF/A-conformant
        # when an OutputIntent is present (ISO 19005-2, 6.2.4.4).
        if alternate == "DeviceGray":
            analysis.device_gray_used = True
        elif alternate == "DeviceRGB":
            analysis.device_rgb_used = True
        elif alternate == "DeviceCMYK":
            analysis.device_cmyk_used = True

    elif cs_type == "DeviceN":
        analysis.devicen_used = True
        analysis.special_colorspaces.append(
            SpecialColorSpace(
                type="DeviceN",
                alternate_space=alternate or "unknown",
                location=location,
                obj_ref=obj_ref,
            )
        )
        if alternate == "DeviceGray":
            analysis.device_gray_used = True
        elif alternate == "DeviceRGB":
            analysis.device_rgb_used = True
        elif alternate == "DeviceCMYK":
            analysis.device_cmyk_used = True

    elif cs_type == "Indexed":
        # Check if base is a special color space
        if alternate in ("Separation", "DeviceN"):
            analysis.indexed_with_special_base = True
            analysis.special_colorspaces.append(
                SpecialColorSpace(
                    type="Indexed",
                    alternate_space=alternate,
                    location=location,
                    obj_ref=obj_ref,
                )
            )
        elif alternate == "DeviceGray":
            analysis.device_gray_used = True
        elif alternate == "DeviceRGB":
            analysis.device_rgb_used = True
        elif alternate == "DeviceCMYK":
            analysis.device_cmyk_used = True

    elif cs_type == "CalGray":
        analysis.cal_gray_used = True
    elif cs_type == "CalRGB":
        analysis.cal_rgb_used = True
    elif cs_type == "Lab":
        analysis.lab_used = True
        # Validate Lab structure: [/Lab dict] where dict has /WhitePoint
        if isinstance(cs, Array) and len(cs) >= 2:
            try:
                lab_dict = _resolve_indirect(cs[1])
                if not isinstance(lab_dict, Dictionary):
                    logger.warning(
                        "Lab color space at %s: second element is not a dictionary",
                        location,
                    )
                elif lab_dict.get("/WhitePoint") is None:
                    logger.warning(
                        "Lab color space at %s: missing required /WhitePoint entry",
                        location,
                    )
            except Exception:
                logger.warning(
                    "Lab color space at %s: unable to validate structure",
                    location,
                )


def _detect_colors_in_content_stream(
    stream_or_page,
    analysis: ColorSpaceAnalysis,
) -> None:
    """
    Detect color spaces used in a content stream.

    Parses the content stream for color operators (g, G, rg, RG, k, K, cs, CS)
    and inline images (BI...ID...EI) whose /CS entry references a device color space.

    Args:
        stream_or_page: A page or stream object to parse.
        analysis: ColorSpaceAnalysis to update with detected color spaces.
    """
    try:
        for operands, operator in pikepdf.parse_content_stream(stream_or_page):
            op_name = str(operator)

            if op_name in _GRAY_OPERATORS:
                analysis.device_gray_used = True
            elif op_name in _RGB_OPERATORS:
                analysis.device_rgb_used = True
            elif op_name in _CMYK_OPERATORS:
                analysis.device_cmyk_used = True
            elif op_name in _CS_OPERATORS and operands:
                cs_name = operands[0]
                if cs_name == Name.DeviceGray:
                    analysis.device_gray_used = True
                elif cs_name == Name.DeviceRGB:
                    analysis.device_rgb_used = True
                elif cs_name == Name.DeviceCMYK:
                    analysis.device_cmyk_used = True
            elif op_name == "INLINE IMAGE" and operands:
                cs = _get_inline_image_device_cs(operands[0])
                if cs == Name.DeviceGray:
                    analysis.device_gray_used = True
                elif cs == Name.DeviceRGB:
                    analysis.device_rgb_used = True
                elif cs == Name.DeviceCMYK:
                    analysis.device_cmyk_used = True
    except (pikepdf.PdfError, AttributeError, IndexError, TypeError) as e:
        logger.debug("Error parsing content stream: %s", e)


def _process_colorspace_resources(
    colorspaces,
    analysis: ColorSpaceAnalysis,
    location_prefix: str,
) -> None:
    """
    Process ColorSpace dictionary from page or form resources.

    Args:
        colorspaces: ColorSpace dictionary from resources.
        analysis: ColorSpaceAnalysis to update.
        location_prefix: Prefix for location description.
    """
    if colorspaces is None:
        return

    colorspaces = _resolve_indirect(colorspaces)

    for name in colorspaces.keys():
        try:
            cs = colorspaces[name]
            cs = _resolve_indirect(cs)
            location = f"{location_prefix}/ColorSpace/{name}"
            _analyze_colorspace(cs, analysis, location, colorspaces)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error processing color space %s: %s", name, e)


def _process_xobjects(
    xobjects,
    analysis: ColorSpaceAnalysis,
    visited: set[tuple[int, int]] | None = None,
    location_prefix: str = "",
) -> None:
    """
    Process XObjects including Form XObjects recursively.

    Checks Image XObjects for their color space and recursively processes
    Form XObjects for color operators in their content streams.

    Args:
        xobjects: Dictionary of XObjects from page resources.
        analysis: ColorSpaceAnalysis to update with detected color spaces.
        visited: Set of already visited XObject objgen tuples to prevent cycles.
        location_prefix: Prefix for location descriptions.
    """
    if visited is None:
        visited = set()

    for name in xobjects.keys():
        xobj = _resolve_indirect(xobjects[name])

        obj_key = xobj.objgen
        if obj_key != (0, 0):
            if obj_key in visited:
                continue
            visited.add(obj_key)

        subtype = xobj.get("/Subtype")

        if subtype == Name.Image:
            cs = xobj.get("/ColorSpace")
            location = f"{location_prefix}/XObject/{name}"
            _analyze_colorspace(cs, analysis, location, xobj)

        elif subtype == Name.Form:
            # Parse Form XObject content stream
            _detect_colors_in_content_stream(xobj, analysis)

            # Process nested resources
            form_resources = xobj.get("/Resources")
            if form_resources:
                form_resources = _resolve_indirect(form_resources)

                # Process ColorSpace dictionary in form resources
                form_cs = form_resources.get("/ColorSpace")
                if form_cs:
                    _process_colorspace_resources(
                        form_cs,
                        analysis,
                        f"{location_prefix}/XObject/{name}/Resources",
                    )

                # Process nested XObjects
                nested_xobjects = form_resources.get("/XObject")
                if nested_xobjects:
                    _process_xobjects(
                        nested_xobjects,
                        analysis,
                        visited,
                        f"{location_prefix}/XObject/{name}/Resources",
                    )

                # Process patterns in form resources
                form_patterns = form_resources.get("/Pattern")
                if form_patterns:
                    _process_patterns(
                        form_patterns,
                        analysis,
                        visited,
                        f"{location_prefix}/XObject/{name}/Resources",
                    )

                # Process shadings in form resources
                form_shadings = form_resources.get("/Shading")
                if form_shadings:
                    _process_shadings(
                        form_shadings,
                        analysis,
                        f"{location_prefix}/XObject/{name}/Resources",
                    )

                # Process Type3 font CharProcs in form resources
                _process_type3_charprocs_colors(
                    form_resources,
                    analysis,
                    visited,
                    f"{location_prefix}/XObject/{name}/Resources",
                )


def _process_shadings(
    shadings,
    analysis: ColorSpaceAnalysis,
    location_prefix: str,
) -> None:
    """Process Shading dictionary from page or form resources.

    Each shading dictionary has a /ColorSpace entry that may reference
    a Device color space.

    Args:
        shadings: Shading dictionary from resources.
        analysis: ColorSpaceAnalysis to update.
        location_prefix: Prefix for location description.
    """
    shadings = _resolve_indirect(shadings)

    for name in shadings.keys():
        try:
            shading = _resolve_indirect(shadings[name])
            cs = shading.get("/ColorSpace")
            if cs is not None:
                cs = _resolve_indirect(cs)
                location = f"{location_prefix}/Shading/{name}"
                _analyze_colorspace(cs, analysis, location, shading)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error processing shading %s: %s", name, e)


def _process_type3_charprocs_colors(
    resources,
    analysis: ColorSpaceAnalysis,
    visited: set[tuple[int, int]],
    location_prefix: str,
) -> None:
    """Detect color spaces used inside Type3 font CharProcs streams.

    Type3 fonts define glyphs via ``/CharProcs`` content streams that may
    contain Device color operators.  This function parses each CharProc
    and also checks the font's own ``/Resources`` for ColorSpace,
    XObjects, Patterns, and Shadings.

    Args:
        resources: A resolved Resources dictionary (page or form).
        analysis: ColorSpaceAnalysis to update.
        visited: Set of ``(obj_num, gen)`` tuples for cycle detection.
        location_prefix: Prefix for location descriptions.
    """
    for font_name, font in _iter_type3_fonts(resources, visited):
        charprocs = font.get("/CharProcs")
        if charprocs is None:
            continue
        charprocs = _resolve_indirect(charprocs)
        if not isinstance(charprocs, Dictionary):
            continue

        font_loc = f"{location_prefix}/Font/{font_name}"

        # Parse each CharProc content stream
        for cp_name in charprocs.keys():
            cp_stream = _resolve_indirect(charprocs[cp_name])
            if isinstance(cp_stream, Stream):
                _detect_colors_in_content_stream(cp_stream, analysis)

        # Check font-level resources
        font_resources = font.get("/Resources")
        if font_resources is not None:
            font_resources = _resolve_indirect(font_resources)

            cs_dict = font_resources.get("/ColorSpace")
            if cs_dict:
                _process_colorspace_resources(
                    cs_dict, analysis, f"{font_loc}/Resources"
                )

            nested_xobjects = font_resources.get("/XObject")
            if nested_xobjects:
                _process_xobjects(
                    nested_xobjects,
                    analysis,
                    visited,
                    f"{font_loc}/Resources",
                )

            font_patterns = font_resources.get("/Pattern")
            if font_patterns:
                _process_patterns(
                    font_patterns,
                    analysis,
                    visited,
                    f"{font_loc}/Resources",
                )

            font_shadings = font_resources.get("/Shading")
            if font_shadings:
                _process_shadings(
                    font_shadings,
                    analysis,
                    f"{font_loc}/Resources",
                )


def _process_patterns(
    patterns,
    analysis: ColorSpaceAnalysis,
    visited: set[tuple[int, int]],
    location_prefix: str,
) -> None:
    """Process Pattern dictionary from page or form resources.

    Handles both pattern types:
    - PatternType 1 (Tiling): Has own content stream and resources that
      may contain Device color spaces.
    - PatternType 2 (Shading): Contains a /Shading entry whose
      /ColorSpace is analyzed.

    Args:
        patterns: Pattern dictionary from resources.
        analysis: ColorSpaceAnalysis to update.
        visited: Set of already visited objgen tuples to prevent cycles.
        location_prefix: Prefix for location description.
    """
    patterns = _resolve_indirect(patterns)

    for name in patterns.keys():
        try:
            pattern = _resolve_indirect(patterns[name])

            # Cycle detection using objgen
            obj_key = pattern.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            pattern_type = pattern.get("/PatternType")
            pat_prefix = f"{location_prefix}/Pattern/{name}"

            if pattern_type == 1:
                # Tiling pattern: has content stream and own resources
                _detect_colors_in_content_stream(pattern, analysis)

                pat_resources = pattern.get("/Resources")
                if pat_resources:
                    pat_resources = _resolve_indirect(pat_resources)

                    pat_cs = pat_resources.get("/ColorSpace")
                    if pat_cs:
                        _process_colorspace_resources(
                            pat_cs, analysis, f"{pat_prefix}/Resources"
                        )

                    pat_xobjects = pat_resources.get("/XObject")
                    if pat_xobjects:
                        _process_xobjects(
                            pat_xobjects,
                            analysis,
                            visited,
                            f"{pat_prefix}/Resources",
                        )

                    pat_patterns = pat_resources.get("/Pattern")
                    if pat_patterns:
                        _process_patterns(
                            pat_patterns,
                            analysis,
                            visited,
                            f"{pat_prefix}/Resources",
                        )

                    pat_shadings = pat_resources.get("/Shading")
                    if pat_shadings:
                        _process_shadings(
                            pat_shadings,
                            analysis,
                            f"{pat_prefix}/Resources",
                        )

            elif pattern_type == 2:
                # Shading pattern: has /Shading entry with /ColorSpace
                shading = pattern.get("/Shading")
                if shading is not None:
                    shading = _resolve_indirect(shading)
                    cs = shading.get("/ColorSpace")
                    if cs is not None:
                        cs = _resolve_indirect(cs)
                        _analyze_colorspace(
                            cs,
                            analysis,
                            f"{pat_prefix}/Shading",
                            shading,
                        )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error processing pattern %s: %s", name, e)


def _detect_colors_in_ap_entry(
    ap_value,
    analysis: ColorSpaceAnalysis,
    visited: set[tuple[int, int]],
    location_prefix: str,
) -> None:
    """Detect color spaces in an annotation appearance (AP) entry.

    The entry may be a Form XObject (stream) directly, or a dictionary
    of sub-state Form XObjects (e.g. On/Off for checkboxes).

    Args:
        ap_value: The value of an AP entry (/N, /R, or /D).
        analysis: ColorSpaceAnalysis to update with detected color spaces.
        visited: Set of ``(obj_num, gen)`` pairs for cycle detection.
        location_prefix: Prefix for location descriptions.
    """
    ap_value = _resolve_indirect(ap_value)

    if isinstance(ap_value, Stream):
        objgen = ap_value.objgen
        if objgen != (0, 0):
            if objgen in visited:
                return
            visited.add(objgen)

        _detect_colors_in_content_stream(ap_value, analysis)

        ap_resources = ap_value.get(Name.Resources)
        if ap_resources is not None:
            ap_resources = _resolve_indirect(ap_resources)
            cs_dict = ap_resources.get("/ColorSpace")
            if cs_dict:
                _process_colorspace_resources(cs_dict, analysis, location_prefix)
            nested = ap_resources.get(Name.XObject)
            if nested:
                _process_xobjects(nested, analysis, visited, location_prefix)
            ap_patterns = ap_resources.get("/Pattern")
            if ap_patterns:
                _process_patterns(ap_patterns, analysis, visited, location_prefix)
            ap_shadings = ap_resources.get("/Shading")
            if ap_shadings:
                _process_shadings(ap_shadings, analysis, location_prefix)
            _process_type3_charprocs_colors(
                ap_resources, analysis, visited, location_prefix
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

                _detect_colors_in_content_stream(sub_stream, analysis)

                sub_resources = sub_stream.get(Name.Resources)
                if sub_resources is not None:
                    sub_resources = _resolve_indirect(sub_resources)
                    cs_dict = sub_resources.get("/ColorSpace")
                    if cs_dict:
                        _process_colorspace_resources(
                            cs_dict,
                            analysis,
                            f"{location_prefix}/{key}",
                        )
                    nested = sub_resources.get(Name.XObject)
                    if nested:
                        _process_xobjects(
                            nested,
                            analysis,
                            visited,
                            f"{location_prefix}/{key}",
                        )
                    sub_patterns = sub_resources.get("/Pattern")
                    if sub_patterns:
                        _process_patterns(
                            sub_patterns,
                            analysis,
                            visited,
                            f"{location_prefix}/{key}",
                        )
                    sub_shadings = sub_resources.get("/Shading")
                    if sub_shadings:
                        _process_shadings(
                            sub_shadings,
                            analysis,
                            f"{location_prefix}/{key}",
                        )
                    _process_type3_charprocs_colors(
                        sub_resources,
                        analysis,
                        visited,
                        f"{location_prefix}/{key}",
                    )


def detect_color_spaces(pdf: Pdf) -> ColorSpaceAnalysis:
    """
    Detect color spaces used in a PDF document.

    Scans all pages for:
    - Color operators in page content streams (g, G, rg, RG, k, K, cs, CS)
    - XObject images and their color spaces
    - Form XObjects and their nested content streams
    - ColorSpace dictionaries in page resources
    - Pattern resources (Tiling and Shading patterns)
    - Shading dictionaries and their color spaces
    - Special color spaces (Separation, DeviceN, Indexed)

    Args:
        pdf: pikepdf Pdf object.

    Returns:
        ColorSpaceAnalysis with detected color spaces.
    """
    analysis = ColorSpaceAnalysis()
    visited: set[tuple[int, int]] = set()

    for page_num, page in enumerate(pdf.pages, start=1):
        resources = page.get("/Resources")
        location_prefix = f"Page{page_num}/Resources"

        # Parse page content stream for color operators
        _detect_colors_in_content_stream(page, analysis)

        if resources:
            resources = _resolve_indirect(resources)

            # Process ColorSpace dictionary in page resources
            colorspaces = resources.get("/ColorSpace")
            if colorspaces:
                _process_colorspace_resources(colorspaces, analysis, location_prefix)

            # Check XObjects (images and forms)
            xobjects = resources.get("/XObject")
            if xobjects:
                _process_xobjects(
                    xobjects,
                    analysis,
                    visited,
                    location_prefix=location_prefix,
                )

            # Check patterns
            patterns = resources.get("/Pattern")
            if patterns:
                _process_patterns(
                    patterns,
                    analysis,
                    visited,
                    location_prefix=location_prefix,
                )

            # Check shadings
            shadings = resources.get("/Shading")
            if shadings:
                _process_shadings(
                    shadings,
                    analysis,
                    location_prefix=location_prefix,
                )

            # Check Type3 font CharProcs
            _process_type3_charprocs_colors(
                resources, analysis, visited, location_prefix
            )

        # Check annotation appearance streams
        annots = page.get("/Annots")
        if annots is not None:
            annots = _resolve_indirect(annots)
            for i in range(len(annots)):
                try:
                    annot = _resolve_indirect(annots[i])
                    ap = annot.get("/AP")
                    if ap is None:
                        continue
                    ap = _resolve_indirect(ap)
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry is not None:
                            _detect_colors_in_ap_entry(
                                ap_entry,
                                analysis,
                                visited,
                                f"Page{page_num}/Annot[{i}]/AP{ap_key}",
                            )
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    logger.debug("Error processing annotation AP colors: %s", e)

    logger.debug(
        "Color space detection: Gray=%s, RGB=%s, CMYK=%s, "
        "Separation=%s, DeviceN=%s, Indexed(special)=%s, "
        "CalGray=%s, CalRGB=%s, Lab=%s",
        analysis.device_gray_used,
        analysis.device_rgb_used,
        analysis.device_cmyk_used,
        analysis.separation_used,
        analysis.devicen_used,
        analysis.indexed_with_special_base,
        analysis.cal_gray_used,
        analysis.cal_rgb_used,
        analysis.lab_used,
    )
    return analysis
