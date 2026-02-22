# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Color space validation for PDF/A compliance.

This module validates embedded ICC profiles for PDF/A conformance.
Separation, DeviceN, and Indexed color spaces are preserved as-is,
since PDF/A-2 and PDF/A-3 allow them when a valid OutputIntent is present.
"""

import logging
from collections.abc import Callable
from typing import Any

from pikepdf import Array, Dictionary, Name, Pdf, Stream, String

from ..color_profile import get_cmyk_profile, get_gray_profile, get_srgb_profile
from ..exceptions import ConversionError
from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

# ICC color space signature (bytes 16-19) to expected component count
_ICC_COLORSPACE_COMPONENTS: dict[bytes, int] = {
    b"RGB ": 3,
    b"CMYK": 4,
    b"GRAY": 1,
    b"Lab ": 3,
}

# Map component count to built-in profile getter for repair
_PROFILE_GETTER_BY_N: dict[int, Callable[[], bytes]] = {
    1: get_gray_profile,
    3: get_srgb_profile,
    4: get_cmyk_profile,
}


def _validate_icc_in_resources(resources, location_prefix: str, validate_icc_stream):
    """Validate ICCBased profiles found in a Resources dictionary.

    Checks both /ColorSpace entries and Image XObjects for ICCBased profiles.

    Args:
        resources: A resolved Resources dictionary.
        location_prefix: String prefix for log/warning locations.
        validate_icc_stream: Closure that validates a single ICC stream.
    """
    resources = _resolve_indirect(resources)

    if not isinstance(resources, Dictionary):
        return

    # Check ColorSpace dictionary
    colorspaces = resources.get("/ColorSpace")
    if colorspaces:
        colorspaces = _resolve_indirect(colorspaces)

        for name in colorspaces.keys():
            cs = _resolve_indirect(colorspaces[name])

            if isinstance(cs, Array) and len(cs) >= 2:
                cs_type_str = str(cs[0])
                if cs_type_str == "/ICCBased":
                    location = f"{location_prefix}/ColorSpace/{name}"
                    validate_icc_stream(cs[1], location)
                elif cs_type_str == "/Indexed":
                    _fix_indexed_lookup_size(cs, f"{location_prefix}/ColorSpace/{name}")

    # Check XObjects for Image XObjects with ICCBased ColorSpace
    xobjects = resources.get("/XObject")
    if xobjects:
        xobjects = _resolve_indirect(xobjects)

        for xname in xobjects.keys():
            xobj = _resolve_indirect(xobjects[xname])

            subtype = xobj.get("/Subtype")
            if subtype == Name.Image:
                cs = xobj.get("/ColorSpace")
                if cs:
                    cs = _resolve_indirect(cs)

                    if isinstance(cs, Array) and len(cs) >= 2:
                        cs_type_str = str(cs[0])
                        if cs_type_str == "/ICCBased":
                            location = f"{location_prefix}/XObject/{xname}"
                            validate_icc_stream(cs[1], location)
                        elif cs_type_str == "/Indexed":
                            _fix_indexed_lookup_size(
                                cs,
                                f"{location_prefix}/XObject/{xname}",
                            )


def _validate_icc_in_form_xobjects_recursive(
    resources, visited: set, location_prefix: str, validate_icc_stream
):
    """Recurse into Form XObjects' nested Resources for ICCBased profiles.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of (objnum, gen) tuples for cycle detection.
        location_prefix: String prefix for log/warning locations.
        validate_icc_stream: Closure that validates a single ICC stream.
    """
    resources = _resolve_indirect(resources)

    if not isinstance(resources, Dictionary):
        return

    xobjects = resources.get("/XObject")
    if not xobjects:
        return

    xobjects = _resolve_indirect(xobjects)
    if not isinstance(xobjects, Dictionary):
        return

    for xobj_name in list(xobjects.keys()):
        xobj = _resolve_indirect(xobjects[xobj_name])

        if not isinstance(xobj, Stream):
            continue

        subtype = xobj.get("/Subtype")
        if subtype is None or str(subtype) != "/Form":
            continue

        # Cycle detection using objgen for indirect objects
        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        # Process this Form XObject's own Resources
        form_resources = xobj.get("/Resources")
        if form_resources:
            form_resources = _resolve_indirect(form_resources)
            form_loc = f"{location_prefix}/XObject/{xobj_name}"
            _validate_icc_in_resources(form_resources, form_loc, validate_icc_stream)
            # Recurse into nested Form XObjects
            _validate_icc_in_form_xobjects_recursive(
                form_resources, visited, form_loc, validate_icc_stream
            )


def _validate_icc_in_ap_stream(
    ap_entry, visited: set, location_prefix: str, validate_icc_stream
):
    """Validate ICCBased profiles in an annotation appearance stream entry.

    An AP entry value can be a Form XObject (stream) directly, or a
    dictionary of sub-state Form XObjects.

    Args:
        ap_entry: An appearance entry (N, R, or D value).
        visited: Set of (objnum, gen) tuples for cycle detection.
        location_prefix: String prefix for log/warning locations.
        validate_icc_stream: Closure that validates a single ICC stream.
    """
    ap_entry = _resolve_indirect(ap_entry)

    if isinstance(ap_entry, Stream):
        # Direct Form XObject appearance stream
        form_resources = ap_entry.get("/Resources")
        if form_resources:
            form_resources = _resolve_indirect(form_resources)
            _validate_icc_in_resources(
                form_resources, location_prefix, validate_icc_stream
            )
            _validate_icc_in_form_xobjects_recursive(
                form_resources, visited, location_prefix, validate_icc_stream
            )
    elif isinstance(ap_entry, Dictionary):
        # Sub-state dictionary: keys map to Form XObject streams
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                state_loc = f"{location_prefix}/{state_name}"
                form_resources = state_stream.get("/Resources")
                if form_resources:
                    form_resources = _resolve_indirect(form_resources)
                    _validate_icc_in_resources(
                        form_resources, state_loc, validate_icc_stream
                    )
                    _validate_icc_in_form_xobjects_recursive(
                        form_resources, visited, state_loc, validate_icc_stream
                    )


def _validate_icc_in_type3_fonts(
    resources, visited: set, location_prefix: str, validate_icc_stream
):
    """Validate ICCBased profiles in Type3 font resources.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of (objnum, gen) tuples for cycle detection.
        location_prefix: String prefix for log/warning locations.
        validate_icc_stream: Closure that validates a single ICC stream.
    """
    for font_name, font in _iter_type3_fonts(resources, visited):
        font_resources = font.get("/Resources")
        if font_resources is None:
            continue
        font_resources = _resolve_indirect(font_resources)
        if not isinstance(font_resources, Dictionary):
            continue

        font_loc = f"{location_prefix}/Font/{font_name}"
        _validate_icc_in_resources(font_resources, font_loc, validate_icc_stream)
        _validate_icc_in_form_xobjects_recursive(
            font_resources, visited, font_loc, validate_icc_stream
        )


def _fix_indexed_lookup_size(cs, location: str) -> None:
    """Validate and fix an Indexed color space lookup table size.

    Expected size is ``(hival + 1) * num_base_components`` bytes.
    Truncates overlong tables or pads short tables with ``\\x00`` bytes.

    Args:
        cs: A resolved Array starting with /Indexed.
        location: Description string for logging.
    """
    if not isinstance(cs, Array) or len(cs) < 4:
        return
    if str(cs[0]) != "/Indexed":
        return

    try:
        base = _resolve_indirect(cs[1])
        hival = int(cs[2])
    except Exception as e:
        raise ConversionError(
            f"{location}: Indexed colour space array is malformed: {e}"
        ) from e

    # Determine the number of components of the base color space
    num_components = None
    if isinstance(base, Name):
        base_str = str(base)
        if base_str == "/DeviceGray":
            num_components = 1
        elif base_str == "/DeviceRGB":
            num_components = 3
        elif base_str == "/DeviceCMYK":
            num_components = 4
    elif isinstance(base, Array) and len(base) >= 1:
        base_type = str(base[0])
        if base_type == "/ICCBased" and len(base) >= 2:
            icc_stream = _resolve_indirect(base[1])
            try:
                num_components = int(icc_stream.get("/N", 0))
            except Exception:
                pass
        elif base_type == "/CalGray":
            num_components = 1
        elif base_type in ("/CalRGB", "/Lab"):
            num_components = 3

    if num_components is None or num_components == 0:
        raise ConversionError(
            f"{location}: Cannot determine component count"
            " for Indexed base colour space"
        )

    expected_size = (hival + 1) * num_components
    lookup = _resolve_indirect(cs[3])

    try:
        if isinstance(lookup, Stream):
            data = bytes(lookup.read_bytes())
        else:
            data = bytes(lookup)
        actual_size = len(data)
    except Exception:
        return

    if actual_size != expected_size:
        if actual_size > expected_size:
            fixed = data[:expected_size]
            logger.warning(
                "%s: Indexed lookup table too long: truncated from %d to %d bytes"
                " (lossy)",
                location,
                actual_size,
                expected_size,
            )
        else:
            fixed = data + b"\x00" * (expected_size - actual_size)
            logger.warning(
                "%s: Indexed lookup table too short: padded from %d to %d bytes"
                " with zeros (lossy)",
                location,
                actual_size,
                expected_size,
            )

        if isinstance(lookup, Stream):
            lookup.write(fixed)
        else:
            cs[3] = String(fixed)


def validate_embedded_icc_profiles(
    pdf: Pdf, level: str, *, repair: bool = False
) -> tuple[int, list[str], int]:
    """Validate embedded ICC profiles for PDF/A compatibility.

    Checks:
    - Valid ICC signature ('acsp')
    - Profile class (scnr, mntr, prtr, spac)
    - Profile version (PDF/A-1: only v2, PDF/A-2+: also v4)
    - Component count matches

    When *repair* is True, invalid profiles are replaced with a built-in
    profile that matches the stream's ``/N`` component count.

    Args:
        pdf: pikepdf Pdf object.
        level: PDF/A conformance level ('2b' or '3b').
        repair: If True, replace invalid ICC profiles in-place.

    Returns:
        Tuple of (profiles_validated, list of warning messages,
        profiles_repaired).
    """
    validated = 0
    repaired = 0
    warnings: list[str] = []
    visited: set[tuple[int, int]] = set()

    # PDF/A-2+ allows up to ICC v4
    max_major_version = 4

    def _try_repair(icc_stream, location: str) -> bool:
        """Attempt to replace an invalid ICC stream with a built-in profile.

        Returns True if the stream was repaired successfully.
        """
        nonlocal repaired
        if not repair:
            return False

        try:
            declared_n = int(icc_stream.get("/N", 0))
        except Exception:
            declared_n = 0

        getter = _PROFILE_GETTER_BY_N.get(declared_n)
        if getter is None:
            logger.warning(
                "ICC repair skipped at %s: unsupported /N=%s",
                location,
                declared_n,
            )
            return False

        replacement_data = getter()
        icc_stream.write(replacement_data)

        # Ensure /N matches the replacement profile
        replacement_n = {1: 1, 3: 3, 4: 4}[declared_n]
        icc_stream[Name.N] = replacement_n

        repaired += 1
        logger.info(
            "ICC profile repaired at %s: replaced with built-in profile (N=%d)",
            location,
            replacement_n,
        )
        return True

    def validate_icc_stream(icc_stream, location: str) -> bool:
        nonlocal validated
        try:
            icc_stream = _resolve_indirect(icc_stream)

            obj_key = icc_stream.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    return True
                visited.add(obj_key)

            profile_data = bytes(icc_stream.read_bytes())

            # Check minimum size (128-byte header)
            if len(profile_data) < 128:
                warnings.append(f"{location}: ICC profile too small")
                _try_repair(icc_stream, location)
                return False

            # Check 'acsp' signature at bytes 36-39
            signature = profile_data[36:40]
            if signature != b"acsp":
                warnings.append(f"{location}: Invalid ICC signature")
                _try_repair(icc_stream, location)
                return False

            # Check profile version (bytes 8-11)
            major_version = profile_data[8]
            if major_version > max_major_version:
                warnings.append(
                    f"{location}: ICC v{major_version} not allowed in PDF/A-{level}"
                )
                _try_repair(icc_stream, location)
                return False

            # Check profile class at bytes 12-15
            profile_class = profile_data[12:16]
            valid_classes = (b"scnr", b"mntr", b"prtr", b"spac")
            if profile_class not in valid_classes:
                warnings.append(
                    f"{location}: ICC profile class"
                    f" {profile_class!r} may not be compatible"
                )

            # Validate component count (/N) against ICC color space signature
            icc_colorspace = profile_data[16:20]
            expected_n = _ICC_COLORSPACE_COMPONENTS.get(icc_colorspace)
            if expected_n is not None:
                try:
                    declared_n = int(icc_stream.get("/N", 0))
                    if declared_n == 0:
                        # /N is missing — derive from ICC header and set it
                        icc_stream[Name.N] = expected_n
                        logger.info(
                            "ICC /N was missing at %s: set to %d from"
                            " profile color space %r",
                            location,
                            expected_n,
                            icc_colorspace,
                        )
                    elif declared_n != expected_n:
                        warnings.append(
                            f"{location}: ICC /N={declared_n} doesn't match "
                            f"profile color space {icc_colorspace!r}"
                            f" (expected {expected_n})"
                        )
                        if repair:
                            icc_stream[Name.N] = expected_n
                            logger.info(
                                "ICC /N corrected at %s: %d → %d",
                                location,
                                declared_n,
                                expected_n,
                            )
                except Exception:
                    pass
            else:
                # Unknown ICC color space — check if /N is missing
                try:
                    declared_n = int(icc_stream.get("/N", 0))
                    if declared_n == 0:
                        logger.warning(
                            "ICC /N missing at %s and could not derive"
                            " from unknown profile color space %r",
                            location,
                            icc_colorspace,
                        )
                except Exception:
                    pass

            validated += 1
            return True

        except Exception as e:
            warnings.append(f"{location}: Error reading ICC profile: {e}")
            return False

    # Check all ICCBased color spaces
    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)
            page_loc = f"Page {page_num}"

            # 1. Page → Resources (ColorSpace + Image XObjects)
            resources = page_dict.get("/Resources")
            if resources:
                resources = _resolve_indirect(resources)

                _validate_icc_in_resources(resources, page_loc, validate_icc_stream)

                # 2. Page → Resources → XObject → Form XObjects (recursive)
                _validate_icc_in_form_xobjects_recursive(
                    resources, visited, page_loc, validate_icc_stream
                )

                # 2b. Page → Resources → Font → Type3 font resources
                _validate_icc_in_type3_fonts(
                    resources, visited, page_loc, validate_icc_stream
                )

            # 3. Page → Annots → AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve_indirect(annots)
                if isinstance(annots, Array):
                    for annot in annots:
                        annot = _resolve_indirect(annot)
                        if not isinstance(annot, Dictionary):
                            continue

                        ap = annot.get("/AP")
                        if not ap:
                            continue

                        ap = _resolve_indirect(ap)
                        if not isinstance(ap, Dictionary):
                            continue

                        for ap_key in ("/N", "/R", "/D"):
                            ap_entry = ap.get(ap_key)
                            if ap_entry:
                                ap_loc = f"{page_loc}/Annot/AP{ap_key}"
                                _validate_icc_in_ap_stream(
                                    ap_entry, visited, ap_loc, validate_icc_stream
                                )

        except Exception as e:
            logger.debug("Error validating ICC on page %d: %s", page_num, e)

    if validated > 0:
        logger.debug("%d ICC profile(s) validated", validated)
    if repaired > 0:
        logger.info("%d ICC profile(s) repaired", repaired)
    if warnings:
        for warning in warnings:
            logger.warning("ICC profile issue: %s", warning)

    return validated, warnings, repaired


_DEVICE_DEPENDENT_SPACES = frozenset(
    {
        "/DeviceRGB",
        "/DeviceCMYK",
        "/DeviceGray",
    }
)

_PROCESS_COLOR_NAMES = frozenset(
    {
        "/Cyan",
        "/Magenta",
        "/Yellow",
        "/Black",
        "/Red",
        "/Green",
        "/Blue",
        "/Gray",
        "/All",
        "/None",
    }
)

_STREAM_COMPARE_IGNORED_KEYS = frozenset(
    {
        "/Length",
        "/Filter",
        "/DecodeParms",
        "/DL",
    }
)


def _normalize_pdf_name(value) -> str | None:
    """Return a normalized '/Name' string, or None if not a name."""
    value = _resolve_indirect(value)
    if isinstance(value, Name):
        return str(value)
    if isinstance(value, str):
        return value if value.startswith("/") else f"/{value}"
    return None


def _normalized_object_signature(
    obj,
    seen: set[tuple[int, int]] | None = None,
    *,
    _depth: int = 0,
    _max_depth: int = 50,
):
    """Build a structure-preserving signature for PDF object comparison.

    Indirect/direct representation is normalized by resolving references.
    Stream compression differences are normalized by comparing decoded bytes
    and ignoring compression-related dictionary keys.
    """
    if seen is None:
        seen = set()

    if _depth >= _max_depth:
        return ("truncated",)

    obj = _resolve_indirect(obj)
    objgen = getattr(obj, "objgen", (0, 0))
    if objgen != (0, 0):
        if objgen in seen:
            return ("ref", objgen)
        seen.add(objgen)

    if isinstance(obj, Name):
        return ("name", str(obj))

    next_depth = _depth + 1

    if isinstance(obj, Stream):
        d_items = []
        for k in sorted(obj.keys(), key=str):
            k_str = str(k)
            if k_str in _STREAM_COMPARE_IGNORED_KEYS:
                continue
            d_items.append(
                (
                    k_str,
                    _normalized_object_signature(
                        obj[k],
                        seen,
                        _depth=next_depth,
                        _max_depth=_max_depth,
                    ),
                )
            )
        try:
            data = bytes(obj.read_bytes())
        except Exception:
            data = b""
        return ("stream", tuple(d_items), data)

    if isinstance(obj, Dictionary):
        items = []
        for k in sorted(obj.keys(), key=str):
            items.append(
                (
                    str(k),
                    _normalized_object_signature(
                        obj[k],
                        seen,
                        _depth=next_depth,
                        _max_depth=_max_depth,
                    ),
                )
            )
        return ("dict", tuple(items))

    if isinstance(obj, Array):
        return (
            "array",
            tuple(
                _normalized_object_signature(
                    item,
                    seen,
                    _depth=next_depth,
                    _max_depth=_max_depth,
                )
                for item in obj
            ),
        )

    if isinstance(obj, bytes):
        return ("bytes", obj)

    if isinstance(obj, str):
        return ("str", obj)

    if isinstance(obj, bool):
        return ("bool", obj)

    if isinstance(obj, (int, float)):
        return ("number", obj)

    return ("scalar", str(obj))


def _register_or_fix_separation(
    separation_cs: Array,
    canonical_by_name: dict[str, tuple[object, object, object]],
) -> int:
    """Ensure Separation arrays with same name share alternate+tint."""
    if len(separation_cs) < 4:
        return 0

    sep_name = _normalize_pdf_name(separation_cs[1])
    if not sep_name:
        return 0

    signature = (
        _normalized_object_signature(separation_cs[2]),
        _normalized_object_signature(separation_cs[3]),
    )
    canonical = canonical_by_name.get(sep_name)
    if canonical is None:
        canonical_by_name[sep_name] = (
            separation_cs[2],
            separation_cs[3],
            signature,
        )
        return 0

    canonical_alt, canonical_tint, canonical_sig = canonical
    if signature != canonical_sig:
        separation_cs[2] = canonical_alt
        separation_cs[3] = canonical_tint
        logger.debug(
            "Normalized Separation %s to canonical alternate/tintTransform",
            sep_name,
        )
        return 1

    return 0


def _sanitize_colorspace_array(
    cs,
    canonical_by_name: dict[str, tuple[object, object, object]],
    visited_cs: set[tuple[int, int]],
) -> tuple[int, int, object | None]:
    """Sanitize DeviceN/Separation consistency inside a color space object.

    Returns (colorants_added, separation_arrays_normalized, replacement).
    When DeviceN has > 32 colorants the alternate colour space is returned as
    the third element so the caller can substitute it in place of the DeviceN.
    Otherwise the third element is None.
    """
    colorants_added = 0
    separations_normalized = 0

    cs = _resolve_indirect(cs)
    if not isinstance(cs, Array) or len(cs) == 0:
        return 0, 0, None

    objgen = getattr(cs, "objgen", (0, 0))
    if objgen != (0, 0):
        if objgen in visited_cs:
            return 0, 0, None
        visited_cs.add(objgen)

    cs_type = str(cs[0])

    if cs_type == "/Separation":
        separations_normalized += _register_or_fix_separation(cs, canonical_by_name)
        if len(cs) >= 3:
            alt = _resolve_indirect(cs[2])
            if isinstance(alt, Array):
                a, b, repl = _sanitize_colorspace_array(
                    alt, canonical_by_name, visited_cs
                )
                colorants_added += a
                separations_normalized += b
                if repl is not None:
                    cs[2] = repl
        return colorants_added, separations_normalized, None

    if cs_type == "/DeviceN":
        names = _resolve_indirect(cs[1]) if len(cs) >= 2 else None
        if not isinstance(names, Array):
            return colorants_added, separations_normalized, None

        if len(names) > 32:
            alternate = _resolve_indirect(cs[2]) if len(cs) >= 3 else None
            if alternate is None:
                raise ConversionError(
                    f"PDF contains a DeviceN colour space with {len(names)} colorants "
                    "(exceeds 32 per ISO 19005-2 rule 6.1.13-9) and has no alternate "
                    "colour space; cannot repair"
                )
            if isinstance(alternate, Array):
                if len(alternate) >= 1 and str(alternate[0]) == "/DeviceN":
                    alt_names = (
                        _resolve_indirect(alternate[1]) if len(alternate) >= 2 else None
                    )
                    if isinstance(alt_names, Array) and len(alt_names) > 32:
                        raise ConversionError(
                            f"PDF contains a DeviceN colour space with {len(names)} "
                            "colorants (exceeds 32 per ISO 19005-2 rule 6.1.13-9); "
                            "its alternate is also a DeviceN with too many colorants; "
                            "cannot repair"
                        )
            logger.warning(
                "DeviceN colour space has %d colorants (exceeds 32, ISO 19005-2 "
                "rule 6.1.13-9); replacing with alternate colour space (lossy)",
                len(names),
            )
            return 0, 0, cs[2]

        if len(cs) < 4:
            return colorants_added, separations_normalized, None

        # DeviceN attributes dictionary (index 4) is optional.
        attrs = _resolve_indirect(cs[4]) if len(cs) >= 5 else None
        if not isinstance(attrs, Dictionary):
            attrs = Dictionary()
            if len(cs) >= 5:
                cs[4] = attrs
            else:
                cs.append(attrs)
            colorants_added += 1

        colorants = _resolve_indirect(attrs.get("/Colorants"))
        if not isinstance(colorants, Dictionary):
            colorants = Dictionary()
            attrs[Name.Colorants] = colorants
            colorants_added += 1

        alternate = cs[2]
        tint_transform = cs[3]

        # Add missing Colorants entries for spot components.
        for component in names:
            comp_name = _normalize_pdf_name(component)
            if not comp_name or comp_name in _PROCESS_COLOR_NAMES:
                continue

            if comp_name not in colorants:
                separation = Array(
                    [
                        Name.Separation,
                        Name(comp_name),
                        alternate,
                        tint_transform,
                    ]
                )
                colorants[Name(comp_name)] = separation
                colorants_added += 1
                logger.debug("Added missing DeviceN Colorants entry for %s", comp_name)

        # Consistency: all Separation arrays of same name must match.
        for cname in list(colorants.keys()):
            cspace = _resolve_indirect(colorants[cname])
            if isinstance(cspace, Array):
                a, b, repl = _sanitize_colorspace_array(
                    cspace, canonical_by_name, visited_cs
                )
                colorants_added += a
                separations_normalized += b
                if repl is not None:
                    colorants[Name(cname)] = repl

        alt = _resolve_indirect(alternate)
        if isinstance(alt, Array):
            a, b, repl = _sanitize_colorspace_array(alt, canonical_by_name, visited_cs)
            colorants_added += a
            separations_normalized += b
            if repl is not None:
                cs[2] = repl

        return colorants_added, separations_normalized, None

    if cs_type == "/Indexed" and len(cs) >= 2:
        base = _resolve_indirect(cs[1])
        if isinstance(base, Array):
            a, b, repl = _sanitize_colorspace_array(base, canonical_by_name, visited_cs)
            if repl is not None:
                cs[1] = repl
            return a, b, None
        return 0, 0, None

    if cs_type == "/Pattern" and len(cs) >= 2:
        base = _resolve_indirect(cs[1])
        if isinstance(base, Array):
            a, b, repl = _sanitize_colorspace_array(base, canonical_by_name, visited_cs)
            if repl is not None:
                cs[1] = repl
            return a, b, None
        return 0, 0, None

    return 0, 0, None


def _sanitize_special_colorspaces_in_resources(
    resources,
    canonical_by_name: dict[str, tuple[object, object, object]],
    visited_cs: set[tuple[int, int]],
) -> tuple[int, int]:
    """Sanitize DeviceN/Separation rules in one Resources dictionary."""
    colorants_added = 0
    separations_normalized = 0

    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return 0, 0

    colorspaces = _resolve_indirect(resources.get("/ColorSpace"))
    if isinstance(colorspaces, Dictionary):
        for cs_name in list(colorspaces.keys()):
            a, b, repl = _sanitize_colorspace_array(
                colorspaces[cs_name], canonical_by_name, visited_cs
            )
            colorants_added += a
            separations_normalized += b
            if repl is not None:
                colorspaces[cs_name] = repl

    xobjects = _resolve_indirect(resources.get("/XObject"))
    if isinstance(xobjects, Dictionary):
        for xname in list(xobjects.keys()):
            xobj = _resolve_indirect(xobjects[xname])
            if not isinstance(xobj, Stream):
                continue
            if xobj.get("/Subtype") != Name.Image:
                continue
            cs = xobj.get("/ColorSpace")
            if not cs:
                continue
            a, b, repl = _sanitize_colorspace_array(cs, canonical_by_name, visited_cs)
            colorants_added += a
            separations_normalized += b
            if repl is not None:
                xobj[Name.ColorSpace] = repl

    return colorants_added, separations_normalized


def _sanitize_special_colorspaces_in_forms_recursive(
    resources,
    visited_forms: set[tuple[int, int]],
    canonical_by_name: dict[str, tuple[object, object, object]],
    visited_cs: set[tuple[int, int]],
) -> tuple[int, int]:
    """Recurse into Form XObjects and sanitize their nested resources."""
    colorants_added = 0
    separations_normalized = 0

    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return 0, 0

    xobjects = _resolve_indirect(resources.get("/XObject"))
    if not isinstance(xobjects, Dictionary):
        return 0, 0

    for xobj_name in list(xobjects.keys()):
        xobj = _resolve_indirect(xobjects[xobj_name])
        if not isinstance(xobj, Stream):
            continue
        subtype = xobj.get("/Subtype")
        if subtype is None or str(subtype) != "/Form":
            continue

        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited_forms:
                continue
            visited_forms.add(objgen)

        form_resources = xobj.get("/Resources")
        if not form_resources:
            continue
        form_resources = _resolve_indirect(form_resources)

        a, b = _sanitize_special_colorspaces_in_resources(
            form_resources, canonical_by_name, visited_cs
        )
        colorants_added += a
        separations_normalized += b

        a, b = _sanitize_special_colorspaces_in_forms_recursive(
            form_resources, visited_forms, canonical_by_name, visited_cs
        )
        colorants_added += a
        separations_normalized += b

    return colorants_added, separations_normalized


def _sanitize_special_colorspaces_in_ap_stream(
    ap_entry,
    visited_forms: set[tuple[int, int]],
    canonical_by_name: dict[str, tuple[object, object, object]],
    visited_cs: set[tuple[int, int]],
) -> tuple[int, int]:
    """Sanitize DeviceN/Separation rules in annotation AP stream resources."""
    colorants_added = 0
    separations_normalized = 0

    ap_entry = _resolve_indirect(ap_entry)

    if isinstance(ap_entry, Stream):
        form_resources = ap_entry.get("/Resources")
        if form_resources:
            form_resources = _resolve_indirect(form_resources)
            a, b = _sanitize_special_colorspaces_in_resources(
                form_resources, canonical_by_name, visited_cs
            )
            colorants_added += a
            separations_normalized += b
            a, b = _sanitize_special_colorspaces_in_forms_recursive(
                form_resources, visited_forms, canonical_by_name, visited_cs
            )
            colorants_added += a
            separations_normalized += b
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if not isinstance(state_stream, Stream):
                continue
            form_resources = state_stream.get("/Resources")
            if not form_resources:
                continue
            form_resources = _resolve_indirect(form_resources)
            a, b = _sanitize_special_colorspaces_in_resources(
                form_resources, canonical_by_name, visited_cs
            )
            colorants_added += a
            separations_normalized += b
            a, b = _sanitize_special_colorspaces_in_forms_recursive(
                form_resources, visited_forms, canonical_by_name, visited_cs
            )
            colorants_added += a
            separations_normalized += b

    return colorants_added, separations_normalized


def _sanitize_special_colorspaces_in_type3_fonts(
    resources,
    visited_forms: set[tuple[int, int]],
    canonical_by_name: dict[str, tuple[object, object, object]],
    visited_cs: set[tuple[int, int]],
) -> tuple[int, int]:
    """Sanitize DeviceN/Separation rules in Type3 font resources."""
    colorants_added = 0
    separations_normalized = 0

    for _font_name, font in _iter_type3_fonts(resources, visited_forms):
        font_resources = font.get("/Resources")
        if font_resources is None:
            continue
        font_resources = _resolve_indirect(font_resources)
        if not isinstance(font_resources, Dictionary):
            continue

        a, b = _sanitize_special_colorspaces_in_resources(
            font_resources, canonical_by_name, visited_cs
        )
        colorants_added += a
        separations_normalized += b

        a, b = _sanitize_special_colorspaces_in_forms_recursive(
            font_resources, visited_forms, canonical_by_name, visited_cs
        )
        colorants_added += a
        separations_normalized += b

    return colorants_added, separations_normalized


def sanitize_special_colorspace_consistency(pdf: Pdf) -> tuple[int, int]:
    """Fix DeviceN Colorants completeness and Separation consistency.

    Returns:
        Tuple of (device_n_colorants_added, separation_arrays_normalized).
    """
    colorants_added = 0
    separations_normalized = 0
    visited_forms: set[tuple[int, int]] = set()
    visited_cs: set[tuple[int, int]] = set()
    canonical_by_name: dict[str, tuple[object, object, object]] = {}

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)
            resources = page_dict.get("/Resources")
            if resources:
                resources = _resolve_indirect(resources)

                a, b = _sanitize_special_colorspaces_in_resources(
                    resources, canonical_by_name, visited_cs
                )
                colorants_added += a
                separations_normalized += b

                a, b = _sanitize_special_colorspaces_in_forms_recursive(
                    resources, visited_forms, canonical_by_name, visited_cs
                )
                colorants_added += a
                separations_normalized += b

                a, b = _sanitize_special_colorspaces_in_type3_fonts(
                    resources, visited_forms, canonical_by_name, visited_cs
                )
                colorants_added += a
                separations_normalized += b

            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve_indirect(annots)
                if isinstance(annots, Array):
                    for annot in annots:
                        annot = _resolve_indirect(annot)
                        if not isinstance(annot, Dictionary):
                            continue

                        ap = _resolve_indirect(annot.get("/AP"))
                        if not isinstance(ap, Dictionary):
                            continue

                        for ap_key in ("/N", "/R", "/D"):
                            ap_entry = ap.get(ap_key)
                            if not ap_entry:
                                continue
                            a, b = _sanitize_special_colorspaces_in_ap_stream(
                                ap_entry, visited_forms, canonical_by_name, visited_cs
                            )
                            colorants_added += a
                            separations_normalized += b

        except ConversionError:
            raise
        except Exception as e:
            logger.debug(
                "Error sanitizing special color spaces on page %d: %s", page_num, e
            )

    if colorants_added or separations_normalized:
        logger.info(
            "Special color spaces sanitized: colorants_added=%d, "
            "separations_normalized=%d",
            colorants_added,
            separations_normalized,
        )

    return colorants_added, separations_normalized


def _warn_device_dependent_alternates(pdf: Pdf) -> int:
    """Log warnings for Separation/DeviceN with device-dependent alternates.

    PDF/A best practice is to use ICCBased alternate spaces for
    Separation and DeviceN color spaces. This function is informational
    only — it does not modify the PDF (OutputIntent covers compliance).

    Args:
        pdf: pikepdf Pdf object.

    Returns:
        Number of device-dependent alternates found.
    """
    count = 0

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)
            resources = page_dict.get("/Resources")
            if not resources:
                continue

            resources = _resolve_indirect(resources)
            colorspaces = resources.get("/ColorSpace")
            if not colorspaces:
                continue

            colorspaces = _resolve_indirect(colorspaces)

            for name in colorspaces.keys():
                try:
                    cs = _resolve_indirect(colorspaces[name])
                    if not isinstance(cs, Array) or len(cs) < 3:
                        continue

                    cs_type = str(cs[0])
                    if cs_type == "/Separation":
                        alternate = _resolve_indirect(cs[2])
                        alt_str = str(alternate)
                        if alt_str in _DEVICE_DEPENDENT_SPACES:
                            logger.warning(
                                "Page %d: Separation %s uses device-dependent"
                                " alternate %s",
                                page_num,
                                str(cs[1]),
                                alt_str,
                            )
                            count += 1
                    elif cs_type == "/DeviceN":
                        alternate = _resolve_indirect(cs[2])
                        alt_str = str(alternate)
                        if alt_str in _DEVICE_DEPENDENT_SPACES:
                            logger.warning(
                                "Page %d: DeviceN uses device-dependent alternate %s",
                                page_num,
                                alt_str,
                            )
                            count += 1
                except Exception:
                    continue

        except Exception as e:
            logger.debug("Error checking alternates on page %d: %s", page_num, e)

    return count


def sanitize_colorspaces(pdf: Pdf, level: str = "3b") -> dict[str, Any]:
    """Validate color spaces for PDF/A compliance.

    Separation, DeviceN, and Indexed color spaces are preserved since
    PDF/A-2 and PDF/A-3 allow them with a valid OutputIntent.
    Only ICC profile validation is performed.

    Args:
        pdf: pikepdf Pdf object.
        level: PDF/A conformance level ('2b', '2u', '3b', or '3u').

    Returns:
        Dict with statistics:
        - icc_profiles_validated: int
        - icc_profiles_repaired: int
        - devicen_colorants_added: int
        - separation_arrays_normalized: int
        - device_dependent_alternates: int
    """
    validated, warnings, repaired = validate_embedded_icc_profiles(
        pdf, level, repair=True
    )
    colorants_added, separations_normalized = sanitize_special_colorspace_consistency(
        pdf
    )
    device_dep = _warn_device_dependent_alternates(pdf)

    return {
        "icc_profiles_validated": validated,
        "icc_profiles_repaired": repaired,
        "devicen_colorants_added": colorants_added,
        "separation_arrays_normalized": separations_normalized,
        "device_dependent_alternates": device_dep,
    }
