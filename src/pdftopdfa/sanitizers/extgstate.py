# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Extended Graphics State sanitization for PDF/A compliance.

ISO 19005-2, Section 6.2.5 forbids certain ExtGState entries:
- /TR (transfer function) — always forbidden
- /TR2 — forbidden unless value is /Default
- /HTP (halftone phase) — always forbidden
- /HT (halftone) — allowed, but constrained by ISO 19005-2, 6.2.5:
  HalftoneType must be 1 or 5; HalftoneName forbidden; TransferFunction
  removed for PDF/A-2/3 compliance

ISO 19005-2, Section 6.4 constrains transparency-related entries:
- /BM (blend mode) — must be a valid PDF blend mode name
- /CA and /ca (opacity) — must be numeric in [0.0, 1.0]
- /SMask (soft mask) — must be /None or a valid soft mask dictionary

ISO 32000-1, Table 58 defines /OPM (overprint mode) as the integer 0 or 1.
"""

import logging
import math
import re
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Pdf, PdfError, Stream

from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect
from .overprint import sanitize_overprint
from .rendering_intent import VALID_RENDERING_INTENTS

# Valid blend modes per PDF Reference 1.7, Table 136
VALID_BLEND_MODES = frozenset(
    {
        "/Normal",
        "/Compatible",
        "/Multiply",
        "/Screen",
        "/Overlay",
        "/Darken",
        "/Lighten",
        "/ColorDodge",
        "/ColorBurn",
        "/HardLight",
        "/SoftLight",
        "/Difference",
        "/Exclusion",
        "/Hue",
        "/Saturation",
        "/Color",
        "/Luminosity",
    }
)

logger = logging.getLogger(__name__)

PRIMARY_HALFTONE_COLORANTS = frozenset(
    {
        "Cyan",
        "Magenta",
        "Yellow",
        "Black",
    }
)


def _is_iccbased_cmyk(cs) -> bool:
    """Return True when *cs* is an ICCBased color space with /N=4."""
    cs = _resolve_indirect(cs)
    if not isinstance(cs, Array) or len(cs) < 2:
        return False
    if str(cs[0]) != "/ICCBased":
        return False

    icc_stream = _resolve_indirect(cs[1])
    if not isinstance(icc_stream, Stream):
        return False

    try:
        return int(icc_stream.get("/N", 0)) == 4
    except Exception:
        return False


def _colorspace_uses_iccbased_cmyk(
    cs,
    visited: set[tuple[int, int] | int],
) -> bool:
    """Return True if a color space tree contains ICCBased CMYK."""
    stack = [cs]
    while stack:
        cs = _resolve_indirect(stack.pop())
        objgen = getattr(cs, "objgen", (0, 0))
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        if _is_iccbased_cmyk(cs):
            return True

        if not isinstance(cs, Array) or len(cs) == 0:
            continue

        cs_type = str(cs[0])
        if cs_type == "/Separation" and len(cs) >= 3:
            stack.append(cs[2])
        elif cs_type == "/DeviceN":
            if len(cs) >= 3:
                stack.append(cs[2])
            if len(cs) >= 5:
                attrs = _resolve_indirect(cs[4])
                if isinstance(attrs, Dictionary):
                    colorants = _resolve_indirect(attrs.get("/Colorants"))
                    if isinstance(colorants, Dictionary):
                        stack.extend(colorants.values())
        elif cs_type in {"/Indexed", "/Pattern"} and len(cs) >= 2:
            stack.append(cs[1])

    return False


def _resources_use_iccbased_cmyk(
    resources,
    visited: set[tuple[int, int] | int],
) -> bool:
    """Return True if Resources contain ICCBased CMYK anywhere relevant."""
    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return False

    colorspaces = _resolve_indirect(resources.get("/ColorSpace"))
    if isinstance(colorspaces, Dictionary):
        for cs_name in list(colorspaces.keys()):
            if _colorspace_uses_iccbased_cmyk(colorspaces[cs_name], visited):
                return True

    # Default color spaces can also carry ICCBased definitions.
    for default_key in ("/DefaultGray", "/DefaultRGB", "/DefaultCMYK"):
        default_cs = resources.get(default_key)
        if default_cs and _colorspace_uses_iccbased_cmyk(default_cs, visited):
            return True

    xobjects = _resolve_indirect(resources.get("/XObject"))
    if isinstance(xobjects, Dictionary):
        for xname in list(xobjects.keys()):
            xobj = _resolve_indirect(xobjects[xname])
            if not isinstance(xobj, Stream):
                continue
            if xobj.get("/Subtype") == Name.Image:
                cs = xobj.get("/ColorSpace")
                if cs and _colorspace_uses_iccbased_cmyk(cs, visited):
                    return True

    shadings = resources.get("/Shading")
    if isinstance(shadings, Dictionary):
        for shading in shadings.values():
            if isinstance(
                shading, (Dictionary, Stream)
            ) and _colorspace_uses_iccbased_cmyk(shading.get("/ColorSpace"), visited):
                return True

    patterns = resources.get("/Pattern")
    if isinstance(patterns, Dictionary):
        for pattern in patterns.values():
            if not isinstance(pattern, Dictionary):
                continue
            shading = pattern.get("/Shading")
            if isinstance(
                shading, (Dictionary, Stream)
            ) and _colorspace_uses_iccbased_cmyk(shading.get("/ColorSpace"), visited):
                return True

    return False


def _iter_resource_dictionaries(pdf: Pdf):
    """Yield every reachable Resources dictionary without recursive descent."""
    yielded: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()
    stack = [pdf.Root]
    stack.extend(pdf.objects)

    acroform = _resolve_indirect(pdf.Root.get("/AcroForm"))
    if isinstance(acroform, Dictionary):
        default_resources = _resolve_indirect(acroform.get("/DR"))
        if isinstance(default_resources, Dictionary):
            key = default_resources.objgen
            if key != (0, 0):
                yielded.add(key)
            yield default_resources

    while stack:
        obj = _resolve_indirect(stack.pop())
        objgen = getattr(obj, "objgen", (0, 0))
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        if isinstance(obj, (Dictionary, Stream)):
            resources = _resolve_indirect(obj.get("/Resources"))
            if isinstance(resources, Dictionary):
                resource_key = resources.objgen
                if resource_key == (0, 0) or resource_key not in yielded:
                    if resource_key != (0, 0):
                        yielded.add(resource_key)
                    yield resources
            stack.extend(obj.values())
        elif isinstance(obj, Array):
            stack.extend(obj)


def _pdf_uses_iccbased_cmyk(pdf: Pdf) -> bool:
    """Return True if the PDF uses ICCBased CMYK in relevant resources."""
    visited_cs: set[tuple[int, int] | int] = set()
    return any(
        _resources_use_iccbased_cmyk(resources, visited_cs)
        for resources in _iter_resource_dictionaries(pdf)
    )


def _overprint_enabled(gs_dict: Dictionary, key: str) -> bool:
    """Return True if an ExtGState overprint flag is enabled."""
    val = _resolve_indirect(gs_dict.get(key))
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def _pdf_enables_overprint(pdf: Pdf) -> bool:
    """Return whether reachable state entries can enable overprint with OPM 1."""
    stack = [pdf.Root]
    visited: set[tuple[int, int]] = set()
    overprint = False
    mode_one = False

    while stack:
        try:
            obj = _resolve_indirect(stack.pop())
            objgen = getattr(obj, "objgen", (0, 0))
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            if isinstance(obj, (Dictionary, Stream)):
                if _overprint_enabled(obj, "/OP") or _overprint_enabled(obj, "/op"):
                    overprint = True
                if obj.get("/OPM") == 1:
                    mode_one = True
                if overprint and mode_one:
                    return True
                stack.extend(obj.values())
            elif isinstance(obj, Array):
                stack.extend(obj)
        except Exception as exc:
            log_suppressed_error(
                logger, exc, "Error inspecting object for overprint: %s", exc
            )

    return False


def _clamp_opacity(gs_dict: Dictionary, key: str) -> int:
    """Clamp an opacity value (/CA or /ca) to [0.0, 1.0].

    Args:
        gs_dict: A resolved ExtGState dictionary.
        key: The key to clamp ("/CA" or "/ca").

    Returns:
        1 if the value was clamped, 0 otherwise.
    """
    try:
        val = float(gs_dict[key])
    except (TypeError, ValueError):
        # Non-numeric value — reset to fully opaque
        gs_dict[key] = 1.0
        logger.debug("Reset non-numeric %s to 1.0 in ExtGState", key)
        return 1

    if val < 0.0:
        gs_dict[key] = 0.0
        logger.debug("Clamped %s from %s to 0.0 in ExtGState", key, val)
        return 1
    if val > 1.0:
        gs_dict[key] = 1.0
        logger.debug("Clamped %s from %s to 1.0 in ExtGState", key, val)
        return 1

    return 0


def _function_numbers(value, count: int) -> bool:
    """Check the dimensions and finite numeric entries of a function parameter."""
    return (
        isinstance(value, Array)
        and len(value) == count
        and all(
            not isinstance(number, bool)
            and isinstance(number, (int, float, Decimal))
            and math.isfinite(number)
            for number in value
        )
    )


_CALCULATOR_OPERATORS = frozenset(
    "abs add atan ceiling cos cvi cvr div exp floor idiv ln log mod mul neg round "
    "sin sqrt sub truncate and bitshift eq false ge gt le lt ne not or true xor "
    "if ifelse copy dup exch index pop roll".split()
)


def _calculator_syntax(data: bytes) -> bool:
    """Require a braced calculator program using PDF numbers and legal operators."""
    data = re.sub(rb"%[^\r\n]*", b"", data.replace(b"\x00", b" ")).strip()
    if not data.startswith(b"{") or not data.endswith(b"}"):
        return False
    depth = 0
    tokens = re.findall(rb"[{}]|[^\s{}]+", data)
    for index, token in enumerate(tokens):
        if token == b"{":
            depth += 1
        elif token == b"}":
            depth -= 1
            if depth < 0 or (depth == 0 and index != len(tokens) - 1):
                return False
        elif not (
            token.decode("latin-1") in _CALCULATOR_OPERATORS
            or re.fullmatch(rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token)
        ):
            return False
    return depth == 0


def _is_scalar_transfer_function(function, ancestors=frozenset()) -> bool:
    """Validate one-input, one-output PDF functions without executing their code."""
    if function == Name.Identity:
        return True
    if not isinstance(function, (Dictionary, Stream)) or len(ancestors) >= 64:
        return False
    key = function.objgen if function.objgen != (0, 0) else function.unparse()
    if key in ancestors:
        return False
    ancestors = ancestors | {key}
    kind = function.get("/FunctionType")
    domain = function.get("/Domain")
    limits = function.get("/Range")
    if (
        isinstance(kind, bool)
        or not isinstance(kind, int)
        or not _function_numbers(domain, 2)
        or domain[0] > domain[1]
    ):
        return False
    if "/Range" in function and (
        not _function_numbers(limits, 2) or limits[0] > limits[1]
    ):
        return False
    if kind in (0, 4):
        if not isinstance(function, Stream) or not _function_numbers(limits, 2):
            return False
        try:
            data = function.read_bytes()
        except (PdfError, RuntimeError):
            return False
        if kind == 4:
            return _calculator_syntax(data)
        size = function.get("/Size")
        bits = function.get("/BitsPerSample")
        order = function.get("/Order", 1)
        return (
            isinstance(size, Array)
            and len(size) == 1
            and not isinstance(size[0], bool)
            and isinstance(size[0], int)
            and size[0] > 0
            and not isinstance(bits, bool)
            and isinstance(bits, int)
            and bits in (1, 2, 4, 8, 12, 16, 24, 32)
            and not isinstance(order, bool)
            and isinstance(order, int)
            and order in (1, 3)
            and len(data) >= (size[0] * bits + 7) // 8
            and all(
                key not in function or _function_numbers(function[key], 2)
                for key in ("/Encode", "/Decode")
            )
        )
    if not isinstance(function, Dictionary):
        return False
    if kind == 2:
        exponent = function.get("/N")
        return (
            not isinstance(exponent, bool)
            and isinstance(exponent, (int, float, Decimal))
            and math.isfinite(exponent)
            and (int(exponent) == exponent or domain[0] >= 0)
            and (exponent >= 0 or not domain[0] <= 0 <= domain[1])
            and all(
                key not in function or _function_numbers(function[key], 1)
                for key in ("/C0", "/C1")
            )
        )
    if kind == 3:
        functions = function.get("/Functions")
        if not isinstance(functions, Array) or not functions:
            return False
        count = len(functions)
        bounds = function.get("/Bounds")
        if not _function_numbers(bounds, count - 1) or not _function_numbers(
            function.get("/Encode"), 2 * count
        ):
            return False
        if any(a >= b for a, b in zip([domain[0], *bounds], [*bounds, domain[1]])):
            # The final interval may have zero width (ISO 32000-1, 7.10.4).
            if count != 1 and not (
                bounds[-1] == domain[1]
                and all(a < b for a, b in zip([domain[0], *bounds[:-1]], bounds))
            ):
                return False
        return all(
            isinstance(child, (Dictionary, Stream))
            and _is_scalar_transfer_function(child, ancestors)
            for child in functions
        )
    return False


def _sanitize_smask_dict(gs_dict: Dictionary, smask_dict: Dictionary) -> int:
    """Validate internal entries of a soft mask dictionary.

    Per PDF Reference 1.7 Table 144:
    - /S (required): must be /Alpha or /Luminosity
    - /G (required): must be a Stream (Form XObject)
    - /TR (optional): preserve the soft-mask transfer function
    - /BC (optional): if present, must be an Array

    If a required entry is invalid or missing, the SMask is replaced
    with /None on gs_dict. Otherwise, malformed /TR and non-array /BC entries
    are removed from smask_dict; removing /TR restores the identity mapping.

    Args:
        gs_dict: The parent ExtGState dictionary (modified in place).
        smask_dict: A resolved soft mask dictionary.

    Returns:
        Number of fixes applied.
    """
    fixes = 0

    # /S is required and must be /Alpha or /Luminosity
    s_val = smask_dict.get("/S")
    if s_val is not None:
        s_val = _resolve_indirect(s_val)
    if not isinstance(s_val, Name) or str(s_val) not in ("/Alpha", "/Luminosity"):
        gs_dict["/SMask"] = Name("/None")
        logger.debug(
            "Replaced SMask with /None: missing or invalid /S entry %s",
            s_val,
        )
        return 1

    # /G is required and must be a Stream (Form XObject)
    g_val = smask_dict.get("/G")
    if g_val is not None:
        g_val = _resolve_indirect(g_val)
    if not isinstance(g_val, Stream):
        gs_dict["/SMask"] = Name("/None")
        logger.debug("Replaced SMask with /None: missing or invalid /G entry")
        return 1

    # /G must be a Form XObject (ISO 19005-2, §6.4)
    g_subtype = g_val.get("/Subtype")
    if g_subtype is None or str(g_subtype) != "/Form":
        gs_dict["/SMask"] = Name("/None")
        logger.warning(
            "Replaced SMask with /None: /G is not a Form XObject (Subtype=%s)",
            g_subtype,
        )
        return 1

    # ISO 19005-2/3, 6.2.5 prohibits /TR in an ExtGState dictionary only.
    # A soft mask's /TR is a separate, permitted opacity mapping. Removing
    # it would change the mask and therefore the rendered page appearance.
    if "/TR" in smask_dict and not _is_scalar_transfer_function(smask_dict.TR):
        del smask_dict["/TR"]
        fixes += 1
        logger.warning("Removed malformed soft-mask /TR; using the identity mapping")

    # /BC must be an Array if present
    if "/BC" in smask_dict:
        bc_val = _resolve_indirect(smask_dict["/BC"])
        if not isinstance(bc_val, Array):
            del smask_dict["/BC"]
            fixes += 1
            logger.debug("Removed malformed /BC from SMask dictionary")

    return fixes


def _sanitize_halftone_dict(
    ht_dict: Dictionary,
    colorant_name: str | None = None,
    fallback_transfer_function=None,
) -> tuple[int, bool]:
    """Sanitize a halftone dictionary for PDF/A compliance.

    ISO 19005-2, Clause 6.2.5 requires:
    - /HalftoneType must be 1 or 5
    - /TransferFunction shall be used only as required by ISO 32000-1
    - /HalftoneName must not be present

    For HalftoneType 5 (composite), walks all sub-halftone
    dictionary values including the required /Default entry. TransferFunction
    handling follows the colorant-specific rule used by veraPDF:
    - top-level and CMYK primary colorants: TransferFunction must be absent
    - non-primary colorants: TransferFunction must be present
    - Default: either present or absent is allowed

    Args:
        ht_dict: A resolved halftone dictionary.
        colorant_name: Optional Type 5 colorant key without leading slash.
        fallback_transfer_function: Value copied into non-primary entries
            when they miss /TransferFunction.

    Returns:
        Tuple ``(fixes, is_valid)`` where ``is_valid`` indicates whether the
        dictionary can be kept. Invalid halftone dictionaries are removed by
        the caller.
    """
    fixes = 0
    pending = [(ht_dict, colorant_name, fallback_transfer_function)]
    visited: set[tuple[tuple[int, int], str | None]] = set()
    while pending:
        current, current_colorant, current_fallback = pending.pop()
        current = _resolve_indirect(current)
        try:
            objgen = current.objgen
        except Exception:
            objgen = (0, 0)
        if objgen != (0, 0):
            context_key = (objgen, current_colorant)
            if context_key in visited:
                continue
            visited.add(context_key)

        ht_type_obj = _resolve_indirect(current.get("/HalftoneType"))
        try:
            ht_type = int(ht_type_obj)
        except Exception:
            ht_type = None

        # PDF/A-2/-3 only allow halftone types 1 and 5.
        if ht_type not in (1, 5):
            logger.debug("Invalid HalftoneType %s in halftone dictionary", ht_type_obj)
            return fixes, False

        # TransferFunction usage is colorant dependent for Type 5 sub-halftones.
        has_transfer_function = "/TransferFunction" in current
        transfer_function_required = (
            current_colorant is not None
            and current_colorant != "Default"
            and current_colorant not in PRIMARY_HALFTONE_COLORANTS
        )
        transfer_function_forbidden = (
            current_colorant is None or current_colorant in PRIMARY_HALFTONE_COLORANTS
        )

        if transfer_function_forbidden and has_transfer_function:
            del current["/TransferFunction"]
            fixes += 1
            logger.debug("Removed /TransferFunction from halftone dictionary")
        elif transfer_function_required and not has_transfer_function:
            current["/TransferFunction"] = (
                current_fallback if current_fallback is not None else Name.Identity
            )
            fixes += 1
            logger.debug("Added /TransferFunction to non-primary halftone dictionary")

        # /HalftoneName must not be present
        if "/HalftoneName" in current:
            del current["/HalftoneName"]
            fixes += 1
            logger.debug("Removed forbidden /HalftoneName from halftone dictionary")

        if ht_type != 5:
            continue

        default_halftone = _resolve_indirect(current.get("/Default"))
        default_transfer_function = None
        if isinstance(default_halftone, Dictionary):
            default_transfer_function = default_halftone.get("/TransferFunction")

        children = []
        for key in list(current.keys()):
            if key in ("/Type", "/HalftoneType"):
                continue
            sub_val = _resolve_indirect(current[key])
            if not isinstance(sub_val, Dictionary):
                logger.debug(
                    "Invalid Type 5 halftone entry %s: expected dictionary", key
                )
                return fixes, False
            children.append(
                (
                    sub_val,
                    str(key).lstrip("/"),
                    default_transfer_function,
                )
            )
        pending.extend(reversed(children))

    return fixes, True


def _sanitize_gs_dict(gs_dict: Dictionary) -> int:
    """Check and remove forbidden entries from a single ExtGState dictionary.

    Args:
        gs_dict: A resolved ExtGState dictionary.

    Returns:
        Number of entries removed.
    """
    removed = 0

    # ISO 32000-1, Table 58 defines /OPM as the integer 0 or 1. A real with the
    # same value selects the same mode, so this repair is also safe for shared
    # dictionaries. Any other value selects no defined mode.
    if "/OPM" in gs_dict:
        opm = _resolve_indirect(gs_dict["/OPM"])
        if isinstance(opm, Decimal) and opm in (0, 1):
            gs_dict["/OPM"] = int(opm)
            removed += 1
            logger.debug("Normalized real /OPM to an integer in ExtGState")
        elif isinstance(opm, bool) or not isinstance(opm, int) or opm not in (0, 1):
            del gs_dict["/OPM"]
            removed += 1
            logger.debug("Removed undefined /OPM from ExtGState")

    # /TR is always forbidden
    if "/TR" in gs_dict:
        del gs_dict["/TR"]
        removed += 1
        logger.debug("Removed forbidden /TR from ExtGState")

    # /TR2 is forbidden unless value is /Default
    if "/TR2" in gs_dict:
        tr2_val = _resolve_indirect(gs_dict["/TR2"])
        if not (isinstance(tr2_val, Name) and str(tr2_val) == "/Default"):
            del gs_dict["/TR2"]
            removed += 1
            logger.debug("Removed non-Default /TR2 from ExtGState")

    # /HTP is always forbidden
    if "/HTP" in gs_dict:
        del gs_dict["/HTP"]
        removed += 1
        logger.debug("Removed forbidden /HTP from ExtGState")

    # /HT halftone dictionaries are allowed in PDF/A-2/3, but internal
    # entries must comply with ISO 19005-2, Clause 6.2.5.
    if "/HT" in gs_dict:
        ht_val = _resolve_indirect(gs_dict["/HT"])
        if isinstance(ht_val, Dictionary):
            ht_fixes, ht_valid = _sanitize_halftone_dict(ht_val)
            removed += ht_fixes
            if not ht_valid:
                del gs_dict["/HT"]
                removed += 1
                logger.debug("Removed non-compliant /HT from ExtGState")
        elif isinstance(ht_val, Name):
            if str(ht_val) != "/Default":
                del gs_dict["/HT"]
                removed += 1
                logger.debug("Removed invalid /HT name %s from ExtGState", ht_val)
        else:
            del gs_dict["/HT"]
            removed += 1
            logger.debug("Removed malformed /HT from ExtGState")

    # /RI must be one of the four valid rendering intents
    if "/RI" in gs_dict:
        ri_val = _resolve_indirect(gs_dict["/RI"])
        if isinstance(ri_val, Name) and str(ri_val) not in VALID_RENDERING_INTENTS:
            gs_dict["/RI"] = Name.RelativeColorimetric
            removed += 1
            logger.debug(
                "Replaced invalid /RI %s with /RelativeColorimetric in ExtGState",
                ri_val,
            )

    # /BM must be a valid blend mode (ISO 19005-2, 6.4)
    if "/BM" in gs_dict:
        bm_val = _resolve_indirect(gs_dict["/BM"])
        if isinstance(bm_val, Name):
            if str(bm_val) not in VALID_BLEND_MODES:
                gs_dict["/BM"] = Name.Normal
                removed += 1
                logger.debug(
                    "Replaced invalid /BM %s with /Normal in ExtGState",
                    bm_val,
                )
        elif isinstance(bm_val, Array):
            # /BM can be an array of blend modes; fix individual entries
            invalid_indices = []
            for i, item in enumerate(bm_val):
                item = _resolve_indirect(item)
                if not isinstance(item, Name) or str(item) not in VALID_BLEND_MODES:
                    invalid_indices.append(i)
            if invalid_indices:
                if len(invalid_indices) == len(bm_val):
                    # All entries invalid — replace with single Name
                    gs_dict["/BM"] = Name.Normal
                else:
                    # Replace only invalid entries, preserving valid ones
                    for i in invalid_indices:
                        bm_val[i] = Name.Normal
                removed += 1
                logger.debug("Fixed invalid /BM entries in ExtGState")

    # /CA (stroking opacity) must be in [0.0, 1.0] (ISO 19005-2, 6.4)
    if "/CA" in gs_dict:
        removed += _clamp_opacity(gs_dict, "/CA")

    # /ca (non-stroking opacity) must be in [0.0, 1.0] (ISO 19005-2, 6.4)
    if "/ca" in gs_dict:
        removed += _clamp_opacity(gs_dict, "/ca")

    # /SMask must be /None or a valid soft mask dictionary (ISO 19005-2, 6.4)
    if "/SMask" in gs_dict:
        smask_val = _resolve_indirect(gs_dict["/SMask"])
        if isinstance(smask_val, Name):
            # Only /None is valid as a Name value
            if str(smask_val) != "/None":
                gs_dict["/SMask"] = Name("/None")
                removed += 1
                logger.debug(
                    "Replaced invalid /SMask name %s with /None in ExtGState",
                    smask_val,
                )
        elif isinstance(smask_val, Dictionary):
            # Validate internal entries of the soft mask dictionary
            removed += _sanitize_smask_dict(gs_dict, smask_val)
        else:
            # Must be /None (Name) or a soft mask dictionary
            del gs_dict["/SMask"]
            removed += 1
            logger.debug("Removed malformed /SMask from ExtGState")

    return removed


def _process_extgstate_dict(extgstate_dict) -> int:
    """Iterate all entries in an /ExtGState resource dictionary.

    Args:
        extgstate_dict: A resolved /ExtGState resource dictionary.

    Returns:
        Number of forbidden entries removed.
    """
    removed = 0
    extgstate_dict = _resolve_indirect(extgstate_dict)

    if not isinstance(extgstate_dict, Dictionary):
        return 0

    for gs_name in list(extgstate_dict.keys()):
        gs = _resolve_indirect(extgstate_dict[gs_name])
        if isinstance(gs, Dictionary):
            removed += _sanitize_gs_dict(gs)

    return removed


def _sanitize_shadings_in_resources(resources) -> int:
    """Remove /TR and /TR2 from Shading dictionaries in Resources.

    ISO 19005-2, §6.2.5 forbids transfer functions in PDF/A.  Shading
    dictionaries referenced via ``/Shading`` in Resources (or via
    PatternType 2 shading patterns) may carry explicit ``/TR`` or
    ``/TR2`` keys that must be stripped.  Note: ``/Function`` defines
    the shading colour and is NOT a transfer function.

    Args:
        resources: A resolved Resources dictionary.

    Returns:
        Number of entries removed.
    """
    removed = 0

    # Direct /Shading entries in resources
    shadings = resources.get("/Shading")
    if shadings:
        shadings = _resolve_indirect(shadings)
        if isinstance(shadings, Dictionary):
            for sh_name in list(shadings.keys()):
                sh = _resolve_indirect(shadings[sh_name])
                if not isinstance(sh, (Dictionary, Stream)):
                    continue
                if "/TR" in sh:
                    del sh["/TR"]
                    removed += 1
                    logger.debug("Removed /TR from Shading dictionary %s", sh_name)
                if "/TR2" in sh:
                    del sh["/TR2"]
                    removed += 1
                    logger.debug("Removed /TR2 from Shading dictionary %s", sh_name)

    # PatternType 2 (shading patterns) embed a /Shading dictionary
    patterns = resources.get("/Pattern")
    if patterns:
        patterns = _resolve_indirect(patterns)
        if isinstance(patterns, Dictionary):
            for pat_name in list(patterns.keys()):
                pat = _resolve_indirect(patterns[pat_name])
                if not isinstance(pat, (Dictionary, Stream)):
                    continue
                try:
                    pat_type = int(pat.get("/PatternType", 0))
                except Exception:
                    continue
                if pat_type != 2:
                    continue
                shading = pat.get("/Shading")
                if not shading:
                    continue
                shading = _resolve_indirect(shading)
                if not isinstance(shading, (Dictionary, Stream)):
                    continue
                if "/TR" in shading:
                    del shading["/TR"]
                    removed += 1
                    logger.debug("Removed /TR from Shading in pattern %s", pat_name)
                if "/TR2" in shading:
                    del shading["/TR2"]
                    removed += 1
                    logger.debug("Removed /TR2 from Shading in pattern %s", pat_name)

    return removed


def _process_resources(resources) -> int:
    """Extract /ExtGState from a Resources dictionary and process it.

    Args:
        resources: A resolved Resources dictionary.

    Returns:
        Number of forbidden entries removed.
    """
    removed = 0
    resources = _resolve_indirect(resources)

    if not isinstance(resources, Dictionary):
        return 0

    extgstate = resources.get("/ExtGState")
    if extgstate:
        removed += _process_extgstate_dict(extgstate)

    # Remove /TR and /TR2 from Shading dictionaries (ISO 19005-2, §6.2.5)
    removed += _sanitize_shadings_in_resources(resources)

    return removed


def sanitize_overprint_usage(pdf: Pdf) -> int:
    """Repair actual ICCBased CMYK overprint, including newly installed defaults.

    Modify affected painting streams and their resources in place.

    Returns:
        Number of painting operations repaired, or zero when no repair is needed.

    Raises:
        ConversionError: If content cannot be safely analyzed or shared graphics
            contexts require incompatible repairs.
    """
    if _pdf_uses_iccbased_cmyk(pdf) and _pdf_enables_overprint(pdf):
        return sanitize_overprint(pdf)
    return 0


def sanitize_extgstate(pdf: Pdf) -> dict[str, int]:
    """Sanitize Extended Graphics State dictionaries for PDF/A compliance.

    Removes forbidden entries per ISO 19005-2, Section 6.2.5:
    - /TR (transfer function)
    - /TR2 (unless /Default)
    - /HTP (halftone phase)

    Validates transparency-related entries per ISO 19005-2, Section 6.4:
    - /BM (blend mode) — replaced with /Normal if invalid
    - /CA and /ca (opacity) — clamped to [0.0, 1.0]
    - /SMask (soft mask) — must be /None or a valid dictionary

    Traverses every reachable Resources dictionary, including inherited page
    resources, nested forms and patterns, soft-mask groups, Type3 fonts,
    annotation appearances, and AcroForm default resources.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with key 'extgstate_fixed': total dictionary fixes and
        repaired painting operations.

    Raises:
        ConversionError: If overprint analysis cannot determine a safe repair.
            Resource dictionary cleanup errors are caught and skipped instead.
    """
    total_removed = 0
    for resources in _iter_resource_dictionaries(pdf):
        try:
            total_removed += _process_resources(resources)
        except Exception as e:
            log_suppressed_error(logger, e, "Error sanitizing ExtGState: %s", e)

    # Resource presence is only a cheap preflight. The overprint pass resolves
    # actual graphics state at each paint and never edits shared /OPM entries.
    # Keep ConversionError outside the best-effort dictionary cleanup above.
    total_removed += sanitize_overprint_usage(pdf)

    if total_removed > 0:
        logger.info("ExtGState sanitized: %d forbidden entries removed", total_removed)

    return {"extgstate_fixed": total_removed}
