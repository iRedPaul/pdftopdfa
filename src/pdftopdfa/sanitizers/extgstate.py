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
"""

import itertools
import logging
import math
import re
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Pdf, PdfError, Stream

from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect
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
    """Return True when any reachable graphics state enables overprint."""
    stack = [pdf.Root]
    visited: set[tuple[int, int]] = set()

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


_CALCULATOR_UNARY = frozenset(
    "abs ceiling cos cvi cvr floor ln log neg round sin sqrt truncate".split()
)
_CALCULATOR_BINARY = frozenset("add atan bitshift div exp idiv mod mul sub".split())
_CALCULATOR_COMPARISONS = {
    "eq": lambda a, b: a == b,
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "ne": lambda a, b: a != b,
}
_CALCULATOR_LOGICAL = frozenset("and not or xor".split())
_CALCULATOR_OPERATORS = (
    _CALCULATOR_UNARY
    | _CALCULATOR_BINARY
    | frozenset(_CALCULATOR_COMPARISONS)
    | _CALCULATOR_LOGICAL
    | frozenset("copy dup exch false if ifelse index pop roll true".split())
)
_CALCULATOR_BOOLEAN = object()
# A number known to be an integer, although its value is not known.
_CALCULATOR_INTEGER = object()
_CALCULATOR_INTEGER_ONLY = frozenset("and bitshift idiv mod not or xor".split())
_CALCULATOR_PRESERVES_INTEGER = frozenset(
    "abs add ceiling floor mul neg round sub truncate".split()
)
_CALCULATOR_MAX_NESTING = 32
_CALCULATOR_MAX_STACK = 100
_CALCULATOR_MAX_INTEGER = 2**31 - 1
_CALCULATOR_MIN_INTEGER = -(2**31)


def _calculator_atan(numerator, denominator) -> float:
    """Compute PostScript atan in degrees, in which 0 0 atan has no defined result."""
    if numerator == 0 and denominator == 0:
        raise ValueError("undefined angle")
    return math.degrees(math.atan2(numerator, denominator)) % 360


# Deterministic results of operators whose literal operands are known, which keep
# later copy, index and roll operands provable. Other operators yield unknown numbers.
_CALCULATOR_FOLDS = {
    "abs": abs,
    "neg": lambda a: -a,
    "cvi": int,
    "cvr": float,
    "ceiling": lambda a: type(a)(math.ceil(a)),
    "floor": lambda a: type(a)(math.floor(a)),
    # PostScript rounds a tie to the greater value, so -1.5 becomes -1, but renderers
    # may differ. Such a value is not treated as known in order to prove a stack effect.
    "round": lambda a: None if a < 0 and a % 1 == 0.5 else type(a)(math.floor(a + 0.5)),
    "truncate": lambda a: type(a)(math.trunc(a)),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: float(a) / b,
    # These raise ValueError for an operand without a defined result, such as -1 sqrt.
    "sqrt": math.sqrt,
    "ln": math.log,
    "log": math.log10,
    "exp": math.pow,
    "sin": lambda a: math.sin(math.radians(a)),
    "cos": lambda a: math.cos(math.radians(a)),
    "atan": _calculator_atan,
}


def _calculator_bitshift(value: int, shift: int):
    """Fold bitshift where renderers agree on its 32-bit result.

    A non-negative value shifted right, or left without losing bits or reaching the
    sign bit, has one result. Renderers differ for negative values and long shifts
    (arithmetic or logical shifts, and counts taken modulo 32), so those results
    remain an integer of unknown value.
    """
    if value < 0 or not -31 <= shift <= 31:
        return _CALCULATOR_INTEGER
    if shift < 0:
        return value >> -shift
    result = value << shift
    return result if result <= _CALCULATOR_MAX_INTEGER else _CALCULATOR_INTEGER


_CALCULATOR_INTEGER_FOLDS = {
    # idiv truncates toward zero, and mod takes the sign of the dividend.
    "idiv": lambda a, b: (1 if (a < 0) == (b < 0) else -1) * (abs(a) // abs(b)),
    "mod": lambda a, b: (1 if a >= 0 else -1) * (abs(a) % abs(b)),
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "not": lambda a: ~a,
    "bitshift": _calculator_bitshift,
}


def _parse_calculator(data: bytes) -> list | None:
    """Parse one braced calculator program into nested blocks, rejecting bad syntax."""
    data = re.sub(rb"%[^\r\n]*", b"", data.replace(b"\x00", b" "))
    outer: list = []
    blocks = [outer]
    # Only the PDF white-space characters (ISO 32000-1, Table 1) separate tokens;
    # Python's \s would also accept a vertical tab, which is a regular character.
    for token in re.findall(rb"[{}]|[^\x00\t\n\x0c\r {}]+", data):
        if token == b"{":
            block: list = []
            blocks[-1].append(block)
            blocks.append(block)
            if len(blocks) > _CALCULATOR_MAX_NESTING:
                return None
        elif token == b"}":
            blocks.pop()
            if not blocks:
                return None
        elif len(blocks) > 1 and (
            token.decode("latin-1") in _CALCULATOR_OPERATORS
            or re.fullmatch(rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token)
        ):
            blocks[-1].append(token)
        else:
            return None
    return outer[0] if len(blocks) == 1 and len(outer) == 1 else None


def _calculator_boolean(value) -> bool:
    """Return whether a stack entry is a boolean, with a known value or not."""
    return value is _CALCULATOR_BOOLEAN or isinstance(value, bool)


def _calculator_known_integer(value) -> bool:
    """Return whether a stack entry is an integer of known value."""
    return isinstance(value, int) and not isinstance(value, bool)


def _calculator_integer(value) -> bool:
    """Return whether a stack entry is certainly an integer."""
    return _calculator_known_integer(value) or value is _CALCULATOR_INTEGER


def _fold_calculator_operator(operator: str, operands: list):
    """Return the most specific abstract result that an operator's operands allow."""
    function = _CALCULATOR_FOLDS.get(operator) or _CALCULATOR_INTEGER_FOLDS.get(
        operator
    )
    if function is not None and all(isinstance(a, (int, float)) for a in operands):
        result = function(*operands)
        if isinstance(result, int) and not (
            _CALCULATOR_MIN_INTEGER <= result <= _CALCULATOR_MAX_INTEGER
        ):
            if operator == "cvi" or operator in _CALCULATOR_INTEGER_ONLY:
                # These operators can only return an integer, so a value outside
                # its range is a rangecheck error rather than an unknown number.
                raise ValueError("integer result out of range")
            # Integer overflow produces an implementation-dependent real result.
            return None
        return result
    if (
        operator == "cvi"
        or operator in _CALCULATOR_INTEGER_ONLY
        or (
            operator in _CALCULATOR_PRESERVES_INTEGER
            and all(_calculator_integer(a) for a in operands)
        )
    ):
        return _CALCULATOR_INTEGER
    return None


def _apply_calculator_operator(token: bytes, stack: list) -> bool:
    """Apply one number or non-conditional operator to an abstract operand stack."""
    operator = token.decode("latin-1")
    if operator not in _CALCULATOR_OPERATORS:
        text = token.decode("ascii")
        value = int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        if isinstance(value, int) and not (
            _CALCULATOR_MIN_INTEGER <= value <= _CALCULATOR_MAX_INTEGER
        ):
            # An integer literal beyond the implementation limit is read as a real.
            value = float(value)
        stack.append(value)
        return True
    if operator in ("true", "false"):
        stack.append(operator == "true")
        return True
    single = operator in _CALCULATOR_UNARY or operator in (
        "copy",
        "dup",
        "index",
        "not",
        "pop",
    )
    pops = 1 if single else 2
    if len(stack) < pops:
        return False
    if operator == "dup":
        stack.append(stack[-1])
        return True
    if operator == "exch":
        stack[-1], stack[-2] = stack[-2], stack[-1]
        return True
    operands = stack[-pops:]
    del stack[-pops:]
    booleans = [_calculator_boolean(operand) for operand in operands]
    known = all(isinstance(operand, (int, float)) for operand in operands)
    if operator == "pop":
        return True
    if operator == "copy":
        count = operands[0]
        if not _calculator_known_integer(count) or not 0 <= count <= len(stack):
            return False
        stack.extend(stack[len(stack) - count :])
    elif operator == "index":
        offset = operands[0]
        if not _calculator_known_integer(offset) or not 0 <= offset < len(stack):
            return False
        stack.append(stack[-1 - offset])
    elif operator == "roll":
        count, shift = operands
        if (
            not _calculator_known_integer(count)
            or not _calculator_known_integer(shift)
            or not 0 <= count <= len(stack)
        ):
            return False
        if count:
            shift %= count
            rolled = stack[-count:]
            stack[-count:] = rolled[-shift:] + rolled[:-shift]
    elif operator in _CALCULATOR_COMPARISONS:
        if booleans[0] != booleans[1] or (booleans[0] and operator not in ("eq", "ne")):
            return False
        stack.append(
            _CALCULATOR_COMPARISONS[operator](*operands)
            if known
            else _CALCULATOR_BOOLEAN
        )
    elif all(booleans) and operator in _CALCULATOR_LOGICAL:
        if not known:
            stack.append(_CALCULATOR_BOOLEAN)
        elif operator == "not":
            stack.append(not operands[0])
        else:
            stack.append(_CALCULATOR_INTEGER_FOLDS[operator](*operands))
    else:
        # The function input is a real number, so integer operators need a proof
        # such as cvi. Applying them to anything else is a run-time type error.
        if any(booleans) or (
            operator in _CALCULATOR_INTEGER_ONLY
            and not all(_calculator_integer(a) for a in operands)
        ):
            return False
        stack.append(_fold_calculator_operator(operator, operands))
    return True


def _apply_calculator_block(block: list, stack: list) -> bool:
    """Prove a block's effect on an abstract operand stack along each possible path.

    Entries are a known integer, real or boolean, None for an unknown number, or a
    marker for a boolean or an integer of unknown value. Known integers let copy,
    index and roll be proven; other operands cannot. A known condition takes only
    its own path, and an unknown one has to be valid along both.
    """
    conditional: list[list] = []
    for item in block:
        if isinstance(item, list):
            conditional.append(item)
        elif item in (b"if", b"ifelse"):
            if len(conditional) != (1 if item == b"if" else 2):
                return False
            if not stack or not _calculator_boolean(stack[-1]):
                return False
            condition = stack.pop()
            if condition is not _CALCULATOR_BOOLEAN:
                if condition or item == b"ifelse":
                    taken = conditional[0] if condition else conditional[1]
                    if not _apply_calculator_block(taken, stack):
                        return False
                conditional = []
                continue
            outcomes = [list(stack) for _ in conditional]
            for branch, outcome in zip(conditional, outcomes, strict=True):
                if not _apply_calculator_block(branch, outcome):
                    return False
            first, second = outcomes if len(outcomes) == 2 else (outcomes[0], stack)
            if len(first) != len(second) or any(
                _calculator_boolean(a) != _calculator_boolean(b)
                for a, b in zip(first, second, strict=True)
            ):
                return False
            stack[:] = [
                a
                if type(a) is type(b) and a == b
                else _CALCULATOR_BOOLEAN
                if _calculator_boolean(a)
                else _CALCULATOR_INTEGER
                if _calculator_integer(a) and _calculator_integer(b)
                else None
                for a, b in zip(first, second, strict=True)
            ]
            conditional = []
        elif conditional or not _apply_calculator_operator(item, stack):
            return False
        if len(stack) > _CALCULATOR_MAX_STACK:
            return False
    return not conditional


_CALCULATOR_SAMPLE_STEPS = 255
# Tokens executed across all sampled runs of one program, which bounds the cost of
# a long program with many distinct literals.
_CALCULATOR_SAMPLE_BUDGET = 200_000
# Grid steps from coarse to fine (8-bit reversal), so that a program which can only
# afford a few samples still spreads them over the whole domain.
_CALCULATOR_GRID_ORDER = sorted(
    range(_CALCULATOR_SAMPLE_STEPS + 1), key=lambda step: int(f"{step:08b}"[::-1], 2)
)


def _calculator_literals(block: list) -> set[float]:
    """Collect the numeric literals of a parsed calculator program."""
    literals: set[float] = set()
    for item in block:
        if isinstance(item, list):
            literals |= _calculator_literals(item)
        elif item.decode("latin-1") not in _CALCULATOR_OPERATORS:
            literals.add(float(item))
    return literals


def _calculator_size(block: list) -> int:
    """Count the tokens of a parsed calculator program, including nested blocks."""
    return sum(
        _calculator_size(item) if isinstance(item, list) else 1 for item in block
    )


def _calculator_samples(program: list, low: float, high: float) -> list[float]:
    """Return the inputs at which a calculator program is run with actual values.

    The domain is sampled at 256 evenly spaced points, which include the entries of
    an 8-bit transfer lookup table. A result is commonly undefined where an input
    equals a constant of the program, as in { 0.25 sub 1 exch div }, so each literal
    and its negation inside the domain are sampled as well. The ends and the middle
    of the domain come first; grid points and literals then alternate up to the
    number of runs that _CALCULATOR_SAMPLE_BUDGET allows for the program's size.
    """
    grid = [
        low + (high - low) * step / _CALCULATOR_SAMPLE_STEPS
        for step in _CALCULATOR_GRID_ORDER
    ]
    literals = sorted(
        {
            value
            for literal in _calculator_literals(program)
            for value in (literal, -literal)
            if low <= value <= high
        }
    )
    ordered = [low, high, (low + high) / 2]
    for pair in itertools.zip_longest(grid, literals):
        ordered.extend(value for value in pair if value is not None)
    limit = max(3, _CALCULATOR_SAMPLE_BUDGET // max(1, _calculator_size(program)))
    return list(dict.fromkeys(ordered))[:limit]


def _is_scalar_calculator(data: bytes, domain) -> bool:
    """Require a calculator program that provably maps one number to one number.

    The program must hold for any input. It also runs for actual inputs, where
    conditions select the path, so a guarded { ln } passes but an unguarded one fails
    at zero. These inputs are a sample of the domain (_calculator_samples()), so an
    undefined result elsewhere is not excluded: sound interval reasoning would also
    reject such guarded programs and replace their valid transfer functions.
    """
    program = _parse_calculator(data)
    if program is None:
        return False
    low, high = float(domain[0]), float(domain[1])
    try:
        for value in (None, *_calculator_samples(program, low, high)):
            stack: list = [value]
            if (
                not _apply_calculator_block(program, stack)
                or len(stack) != 1
                or _calculator_boolean(stack[0])
            ):
                return False
    except (ZeroDivisionError, OverflowError, ValueError):
        # An operation without a defined result, such as a division by zero.
        return False
    return True


def _is_scalar_transfer_function(
    function, ancestors=frozenset(), validated: dict | None = None
) -> bool:
    """Validate that a PDF function maps one input to one output, in each type.

    A function's validity does not depend on the path that reaches it, so results
    are memoized in ``validated``. Otherwise a stitching graph that reuses a child
    in every level would be checked an exponential number of times.
    """
    if function == Name.Identity:
        return True
    if not isinstance(function, (Dictionary, Stream)) or len(ancestors) >= 64:
        return False
    key = function.objgen if function.objgen != (0, 0) else function.unparse()
    if key in ancestors:
        return False
    if validated is None:
        validated = {}
    if key not in validated:
        validated[key] = _check_scalar_transfer_function(
            function, ancestors | {key}, validated
        )
    return validated[key]


def _check_scalar_transfer_function(function, ancestors, validated: dict) -> bool:
    """Validate one function object; see _is_scalar_transfer_function()."""
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
            return _is_scalar_calculator(data, domain)
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
            # ISO 32000-1, 7.3.8.2: too little or too much data is an error.
            and len(data) == (size[0] * bits + 7) // 8
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
            and _is_scalar_transfer_function(child, ancestors, validated)
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


def _sanitize_gs_dict(gs_dict: Dictionary, reset_opm: bool) -> int:
    """Check and remove forbidden entries from a single ExtGState dictionary.

    Args:
        gs_dict: A resolved ExtGState dictionary.

    Returns:
        Number of entries removed.
    """
    removed = 0

    # /TR is always forbidden
    if "/TR" in gs_dict:
        del gs_dict["/TR"]
        removed += 1
        logger.debug("Removed forbidden /TR from ExtGState")

    # PDF/A 6.2.4.2:
    # OPM=1 is forbidden with ICCBased CMYK when overprint is enabled.
    # Both the colour space and overprint flags are intentionally checked
    # document-wide. Graphics-state parameters persist independently, so an
    # /OPM value and /OP or /op may be supplied by different ExtGState
    # dictionaries or inherited across a nested content stream.
    if reset_opm and "/OPM" in gs_dict:
        try:
            opm = int(_resolve_indirect(gs_dict["/OPM"]))
        except Exception:
            opm = 0
        if opm == 1:
            gs_dict["/OPM"] = 0
            removed += 1
            logger.debug("Set /OPM to 0 due to ICCBased CMYK with overprint enabled")

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


def _process_extgstate_dict(extgstate_dict, reset_opm: bool) -> int:
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
            removed += _sanitize_gs_dict(gs, reset_opm)

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


def _process_resources(resources, reset_opm: bool) -> int:
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
        removed += _process_extgstate_dict(extgstate, reset_opm)

    # Remove /TR and /TR2 from Shading dictionaries (ISO 19005-2, §6.2.5)
    removed += _sanitize_shadings_in_resources(resources)

    return removed


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
        Dictionary with key 'extgstate_fixed': number of entries removed.
    """
    reset_opm = _pdf_uses_iccbased_cmyk(pdf) and _pdf_enables_overprint(pdf)
    total_removed = 0
    for resources in _iter_resource_dictionaries(pdf):
        try:
            total_removed += _process_resources(resources, reset_opm)
        except Exception as e:
            log_suppressed_error(logger, e, "Error sanitizing ExtGState: %s", e)

    if total_removed > 0:
        logger.info("ExtGState sanitized: %d forbidden entries removed", total_removed)

    return {"extgstate_fixed": total_removed}
