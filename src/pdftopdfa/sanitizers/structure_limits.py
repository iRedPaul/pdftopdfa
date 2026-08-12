# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sanitization for structural PDF implementation limits.

This module handles PDF/A structural limits that commonly appear in corpus
tests:

- Rule 6.1.13 (implementation limits)
- Rule 6.1.6 (hexadecimal string syntax in content streams)
- Rule 6.1.8 (UTF-8 validity of selected name objects)
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import UnsupportedPDFError
from ..fonts.glyph_usage import (
    CharacterCode,
    FontUsageCache,
    _iter_content_streams_with_resources,
    collect_font_usage,
)
from ..fonts.traversal import get_page_resources, iter_all_page_fonts
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

_INT_MAX = 2_147_483_647
_INT_MIN = -2_147_483_648
_MAX_STRING_BYTES = 32_767
_MAX_NAME_BYTES = 127
_MAX_Q_NESTING = 28
_MAX_CID_VALUE = 65_535
_MIN_REAL_MAGNITUDE = Decimal("1.175e-38")
_MAX_REAL_MAGNITUDE = Decimal("3.403e+38")

_TEXT_OPERATORS = frozenset({"Tj", "TJ", "'", '"'})
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
_ASCII_NAME_RE = re.compile(r"[^A-Za-z0-9_.+-]+")
_CIDCHAR_INT_RE = re.compile(r"^<[^>]+>\s+(-?\d+)$")
_CIDS_INT_GROUP_RE = re.compile(
    r"^(?P<src><(?P<src_hex>[^>]+)>)\s+"
    r"(?P<end><(?P<end_hex>[^>]+)>)\s+"
    r"(?P<cid>-?\d+)$"
)
_CIDS_HEX_GROUP_RE = re.compile(
    r"^(?P<src><(?P<src_hex>[^>]+)>)\s+"
    r"(?P<end><(?P<end_hex>[^>]+)>)\s+"
    r"<(?P<cid_hex>[0-9A-Fa-f]+)>$"
)
_BLOCK_BEGIN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<count>\d+)"
    r"(?P<suffix>\s+begincid(?:char|range)\s*)$"
)


def _indirect_objgen(obj: Any) -> tuple[int, int] | None:
    """Return indirect object id, or None for direct objects."""
    objgen = getattr(obj, "objgen", (0, 0))
    if objgen != (0, 0):
        return objgen
    return None


def _iter_name_bytes(literal_bytes: bytes) -> bytes:
    """Decode a PDF name literal body into raw bytes."""
    body = literal_bytes[1:] if literal_bytes.startswith(b"/") else literal_bytes
    out = bytearray()
    i = 0
    while i < len(body):
        if (
            body[i] == 0x23
            and i + 2 < len(body)
            and body[i + 1] in _HEX_DIGITS
            and body[i + 2] in _HEX_DIGITS
        ):
            out.append(int(body[i + 1 : i + 3], 16))
            i += 3
            continue
        out.append(body[i])
        i += 1
    return bytes(out)


def _name_token_to_raw_bytes(token: Any) -> bytes:
    """Extract raw name bytes from a Name token (key/value/operand)."""
    if isinstance(token, Name):
        try:
            literal = token.unparse()
        except Exception:
            literal = str(token).encode("utf-8", "surrogateescape")
    elif isinstance(token, str):
        literal = token.encode("utf-8", "surrogateescape")
    else:
        literal = str(token).encode("utf-8", "surrogateescape")
    return _iter_name_bytes(literal)


def _raw_name_bytes_to_literal(raw: bytes) -> str:
    """Encode raw name bytes as a PDF name literal."""
    parts: list[str] = ["/"]
    for b in raw:
        if (
            48 <= b <= 57
            or 65 <= b <= 90
            or 97 <= b <= 122
            or b in (45, 46, 95, 43)  # - . _ +
        ):
            parts.append(chr(b))
        else:
            parts.append(f"#{b:02X}")
    return "".join(parts)


def _sanitize_raw_name(raw: bytes) -> tuple[bytes | None, bool, bool]:
    """Return replacement bytes if name exceeds limits or invalid UTF-8."""
    too_long = len(raw) > _MAX_NAME_BYTES
    try:
        decoded = raw.decode("utf-8")
        utf8_invalid = False
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", "replace")
        utf8_invalid = True

    if not too_long and not utf8_invalid:
        return None, False, False

    base = _ASCII_NAME_RE.sub("_", decoded).strip("_")
    if not base:
        base = "Name"

    digest = hashlib.sha1(raw).hexdigest()[:10]
    suffix = f"_{digest}"
    max_prefix_len = _MAX_NAME_BYTES - len(suffix.encode("ascii"))
    if max_prefix_len < 1:
        max_prefix_len = 1
    prefix = base[:max_prefix_len]
    replacement = (prefix + suffix).encode("ascii")
    return replacement, too_long, utf8_invalid


def _sanitize_name_token(token: Any) -> tuple[Any, bool, bool]:
    """Sanitize one Name token and return (replacement, long_fixed, utf8_fixed)."""
    raw = _name_token_to_raw_bytes(token)
    replacement_raw, long_fixed, utf8_fixed = _sanitize_raw_name(raw)
    if replacement_raw is None:
        return token, False, False

    literal = _raw_name_bytes_to_literal(replacement_raw)
    if isinstance(token, Name):
        return Name(literal), long_fixed, utf8_fixed
    return literal, long_fixed, utf8_fixed


def _sanitize_integer(value: Any) -> tuple[Any, bool]:
    """Clamp integer values to PDF/A implementation limits."""
    if isinstance(value, bool) or not isinstance(value, int):
        return value, False
    if value > _INT_MAX:
        return _INT_MAX, True
    if value < _INT_MIN:
        return _INT_MIN, True
    return value, False


def _sanitize_real(value: Any) -> tuple[Any, bool]:
    """Normalize out-of-range real values.

    Near-zero (abs < 1.175e-38): clamp to 0.
    Overflow  (abs > 3.403e+38): clamp to ±3.403e+38.
    """
    if isinstance(value, Decimal):
        if value != 0 and abs(value) < _MIN_REAL_MAGNITUDE:
            return Decimal("0"), True
        if abs(value) > _MAX_REAL_MAGNITUDE:
            return (_MAX_REAL_MAGNITUDE if value > 0 else -_MAX_REAL_MAGNITUDE), True
        return value, False
    if isinstance(value, float):
        _min = float(_MIN_REAL_MAGNITUDE)
        _max = float(_MAX_REAL_MAGNITUDE)
        if value != 0.0 and abs(value) < _min:
            return 0.0, True
        if abs(value) > _max:
            return (_max if value > 0 else -_max), True
        return value, False
    return value, False


def _sanitize_string(value: Any) -> tuple[Any, bool]:
    """Truncate string objects to the PDF/A implementation limit."""
    if not isinstance(value, pikepdf.String):
        return value, False
    try:
        raw = bytes(value)
    except Exception:
        return value, False
    if len(raw) <= _MAX_STRING_BYTES:
        return value, False
    return pikepdf.String(raw[:_MAX_STRING_BYTES]), True


def _fix_odd_hex_string(value: pikepdf.String) -> pikepdf.String:
    """Pad odd-length hexadecimal string literals with a trailing zero."""
    try:
        literal = value.unparse()
    except Exception:
        return value

    if not (literal.startswith(b"<") and literal.endswith(b">")):
        return value

    inner = literal[1:-1]
    normalized = re.sub(rb"\s+", b"", inner)
    if not normalized:
        return value
    if any(char not in _HEX_DIGITS for char in normalized):
        return value
    if len(normalized) % 2 == 0:
        return value

    try:
        return pikepdf.Object.parse(b"<" + inner + b"0>")
    except Exception:
        return value


def _sanitize_operand(value: Any, stats: dict[str, int]) -> tuple[Any, bool]:
    """Sanitize an operand in a parsed content stream instruction."""
    result = _resolve(value)
    changed = False
    pending: list[tuple[Any, Dictionary | Stream | Array | None, Any]] = [
        (value, None, None)
    ]
    while pending:
        raw, parent, location = pending.pop()
        current = _resolve(raw)

        if isinstance(current, Name):
            replacement, long_fixed, utf8_fixed = _sanitize_name_token(current)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
            item_changed = bool(long_fixed or utf8_fixed)
        elif isinstance(current, str) and current.startswith("/"):
            replacement, long_fixed, utf8_fixed = _sanitize_name_token(current)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
            item_changed = bool(long_fixed or utf8_fixed)
        else:
            replacement, item_changed = _sanitize_string(current)
            if item_changed:
                stats["strings_truncated"] += 1
            else:
                replacement, item_changed = _sanitize_integer(current)
                if item_changed:
                    stats["integers_clamped"] += 1
                else:
                    replacement, item_changed = _sanitize_real(current)
                    if item_changed:
                        stats["reals_normalized"] += 1

        if item_changed:
            if parent is None:
                result = replacement
                changed = True
            else:
                try:
                    parent[location] = replacement
                except KeyError:
                    # Some stream dictionary keys (e.g. /Length) are immutable.
                    continue
                changed = True
            continue

        if isinstance(current, Array):
            pending.extend(
                (item, current, index)
                for index, item in reversed(list(enumerate(current)))
            )
        elif isinstance(current, (Dictionary, Stream)):
            for old_key in list(current.keys()):
                new_key, long_fixed, utf8_fixed = _sanitize_name_token(old_key)
                if long_fixed:
                    stats["names_shortened"] += 1
                if utf8_fixed:
                    stats["utf8_names_fixed"] += 1
                if new_key != old_key:
                    child = current[old_key]
                    del current[old_key]
                    current[new_key] = child
                    changed = True

            pending.extend(
                (current[key], current, key) for key in reversed(list(current.keys()))
            )

    return result, changed


def _count_odd_hex_string_tokens(stream_data: bytes) -> int:
    """Count odd-length hexadecimal string literals in content stream bytes."""
    odd = 0
    for token in re.findall(rb"(?<!<)<([^<>]*)>(?!>)", stream_data):
        normalized = re.sub(rb"\s+", b"", token)
        if not normalized:
            continue
        if any(c not in _HEX_DIGITS for c in normalized):
            continue
        if len(normalized) % 2 == 1:
            odd += 1
    return odd


def _strip_invalid_hex_chars(stream_data: bytes) -> tuple[bytes, int]:
    """Remove non-hexadecimal characters from hex string tokens.

    Returns (modified_bytes, count_of_fixed_tokens).
    Whitespace inside hex strings is preserved (it is already valid per spec).
    """
    count = 0

    def _fix(m: re.Match[bytes]) -> bytes:
        nonlocal count
        inner = m.group(1)
        cleaned = bytes(b for b in inner if b in _HEX_DIGITS or b in b" \t\n\r")
        if cleaned != inner:
            count += 1
            return b"<" + cleaned + b">"
        return m.group(0)

    return re.sub(rb"(?<!<)<([^<>]*)>(?!>)", _fix, stream_data), count


def _operands_contain_parse_placeholders(operand: Any) -> bool:
    """Return True when qpdf left placeholder values in parsed operands."""
    pending = [operand]
    while pending:
        current = pending.pop()
        if current is None:
            return True
        if isinstance(current, Array):
            pending.extend(current)
    return False


def _instructions_need_hex_repair(instructions: list[Any]) -> bool:
    """Detect when parsed content still contains invalid hex placeholders."""
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        if any(
            _operands_contain_parse_placeholders(operand)
            for operand in instruction.operands
        ):
            return True
    return False


def _clone_form_resources(resources: Dictionary) -> Dictionary:
    """Clone a resource dictionary without linking a new Form to itself."""
    cloned = Dictionary()
    for key in list(resources.keys()):
        cloned[key] = resources[key]

    xobjects = _resolve(cloned.get("/XObject"))
    if isinstance(xobjects, Dictionary):
        cloned_xobjects = Dictionary()
        for key in list(xobjects.keys()):
            cloned_xobjects[key] = xobjects[key]
        cloned[Name.XObject] = cloned_xobjects
    return cloned


def _add_form_resource(resources: Dictionary, form: Stream) -> Name:
    """Add a Form XObject under a collision-free private resource name."""
    xobjects = _resolve(resources.get("/XObject"))
    if not isinstance(xobjects, Dictionary):
        xobjects = Dictionary()
        resources[Name.XObject] = xobjects

    index = len(xobjects)
    resource_name = Name(f"/FmQDepth{index}")
    while xobjects.get(resource_name) is not None:
        index += 1
        resource_name = Name(f"/FmQDepth{index}")
    xobjects[resource_name] = form
    return resource_name


def _validate_form_block(instructions: list[Any]) -> None:
    """Reject operator scopes that cannot legally cross a Form boundary."""
    text_depth = 0
    compatibility_depth = 0
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        operator = str(instruction.operator)
        if operator in {"BMC", "BDC", "EMC", "MP", "DP"}:
            raise UnsupportedPDFError(
                "Cannot safely reduce q/Q nesting across marked-content operators."
            )
        if operator == "BT":
            text_depth += 1
        elif operator == "ET":
            text_depth -= 1
        elif operator == "BX":
            compatibility_depth += 1
        elif operator == "EX":
            compatibility_depth -= 1
        if text_depth < 0 or compatibility_depth < 0:
            break

    if text_depth != 0 or compatibility_depth != 0:
        raise UnsupportedPDFError(
            "Cannot safely reduce q/Q nesting across BT/ET or BX/EX boundaries."
        )


def _concatenate_matrix(
    current: tuple[float, float, float, float, float, float],
    operand: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Concatenate a PDF ``cm`` operand with the current transformation."""
    a, b, c, d, e, f = current
    oa, ob, oc, od, oe, of = operand
    return (
        a * oa + c * ob,
        b * oa + d * ob,
        a * oc + c * od,
        b * oc + d * od,
        a * oe + c * of + e,
        b * oe + d * of + f,
    )


def _bbox_in_current_user_space(
    bbox: Array,
    matrix: tuple[float, float, float, float, float, float],
) -> Array:
    """Map owner bounds through the inverse current transformation."""
    a, b, c, d, e, f = matrix
    determinant = a * d - b * c
    if abs(determinant) < 1e-15:
        raise UnsupportedPDFError(
            "Cannot safely reduce q/Q nesting under a singular transformation."
        )

    def inverse_point(x: float, y: float) -> tuple[float, float]:
        translated_x = x - e
        translated_y = y - f
        return (
            (d * translated_x - c * translated_y) / determinant,
            (-b * translated_x + a * translated_y) / determinant,
        )

    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except Exception as e:
        raise UnsupportedPDFError(
            "Cannot safely reduce q/Q nesting with an invalid content bounding box."
        ) from e

    corners = (
        inverse_point(x0, y0),
        inverse_point(x0, y1),
        inverse_point(x1, y0),
        inverse_point(x1, y1),
    )
    x_values = [point[0] for point in corners]
    y_values = [point[1] for point in corners]
    return Array(
        [
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        ]
    )


@dataclass
class _QWrapFrame:
    """One pending q/Q repair scope for the iterative post-order walk."""

    rewritten: list[Any]
    resources: Dictionary
    bbox: Array
    wrapped: int = 0
    continuation: tuple[int, int] | None = None


def _wrap_excess_q_nesting(
    pdf: Pdf,
    instructions: list[Any],
    resources: Dictionary,
    bbox: Array,
) -> tuple[list[Any], int]:
    """Replace excess q...Q blocks with equivalent Form XObject calls."""
    frames = [_QWrapFrame(list(instructions), resources, bbox)]

    while frames:
        frame = frames[-1]
        matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        matrix_stack: list[tuple[float, float, float, float, float, float]] = []
        text_depth = 0
        marked_content_depth = 0
        compatibility_depth = 0
        path_active = False
        overflow_start: int | None = None
        overflow_matrix = matrix
        for index, instruction in enumerate(frame.rewritten):
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                continue
            operator = str(instruction.operator)
            if operator == "q":
                if len(matrix_stack) >= _MAX_Q_NESTING:
                    if (
                        text_depth != 0
                        or marked_content_depth != 0
                        or compatibility_depth != 0
                        or path_active
                    ):
                        raise UnsupportedPDFError(
                            "Cannot safely reduce q/Q nesting across an active "
                            "text, marked-content, compatibility, or path scope."
                        )
                    overflow_start = index
                    overflow_matrix = matrix
                    break
                matrix_stack.append(matrix)
            elif operator == "Q" and matrix_stack:
                matrix = matrix_stack.pop()
            elif operator == "cm" and len(instruction.operands) == 6:
                try:
                    operand = tuple(float(value) for value in instruction.operands)
                except Exception as e:
                    raise UnsupportedPDFError(
                        "Cannot safely reduce q/Q nesting with an invalid cm operand."
                    ) from e
                matrix = _concatenate_matrix(matrix, operand)
            elif operator == "BT":
                text_depth += 1
            elif operator == "ET":
                text_depth = max(0, text_depth - 1)
            elif operator in {"BMC", "BDC"}:
                marked_content_depth += 1
            elif operator == "EMC":
                marked_content_depth = max(0, marked_content_depth - 1)
            elif operator == "BX":
                compatibility_depth += 1
            elif operator == "EX":
                compatibility_depth = max(0, compatibility_depth - 1)
            elif operator in {"m", "l", "c", "v", "y", "h", "re"}:
                path_active = True
            elif operator in {
                "S",
                "s",
                "f",
                "F",
                "f*",
                "B",
                "B*",
                "b",
                "b*",
                "n",
            }:
                path_active = False

        if overflow_start is None:
            completed = frames.pop()
            if not frames:
                return completed.rewritten, completed.wrapped

            parent = frames[-1]
            assert parent.continuation is not None
            parent_start, parent_end = parent.continuation
            form = pdf.make_stream(pikepdf.unparse_content_stream(completed.rewritten))
            form[Name.Type] = Name.XObject
            form[Name.Subtype] = Name.Form
            form[Name.FormType] = 1
            form[Name.BBox] = completed.bbox
            form[Name.Resources] = completed.resources
            resource_name = _add_form_resource(parent.resources, form)
            invocation = pikepdf.ContentStreamInstruction(
                [resource_name],
                pikepdf.Operator("Do"),
            )
            parent.rewritten[parent_start : parent_end + 1] = [invocation]
            parent.wrapped += completed.wrapped + 1
            parent.continuation = None
            continue

        balance = 1
        overflow_end: int | None = None
        for index in range(overflow_start + 1, len(frame.rewritten)):
            instruction = frame.rewritten[index]
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                continue
            operator = str(instruction.operator)
            if operator == "q":
                balance += 1
            elif operator == "Q":
                balance -= 1
                if balance == 0:
                    overflow_end = index
                    break

        if overflow_end is None:
            raise UnsupportedPDFError(
                "Cannot safely reduce unbalanced q/Q graphics-state nesting."
            )

        inner = frame.rewritten[overflow_start + 1 : overflow_end]
        _validate_form_block(inner)
        form_bbox = _bbox_in_current_user_space(frame.bbox, overflow_matrix)
        form_resources = _clone_form_resources(frame.resources)
        frame.continuation = (overflow_start, overflow_end)
        frames.append(
            _QWrapFrame(
                rewritten=inner,
                resources=form_resources,
                bbox=form_bbox,
            )
        )

    raise AssertionError("q/Q repair stack terminated without a result")


def _content_bbox(owner: Dictionary | Stream) -> Array | None:
    """Return the visible coordinate bounds for a content owner."""
    owner = _resolve(owner)
    if not isinstance(owner, (Dictionary, Stream)):
        return None
    for key in ("/BBox", "/MediaBox", "/CropBox"):
        value = _resolve(owner.get(key))
        if isinstance(value, Array) and len(value) == 4:
            return Array(list(value))
    return None


def _sanitize_content_stream(
    pdf: Pdf,
    stream_obj: Stream,
    stats: dict[str, int],
    resources: Dictionary | None,
    bbox: Array | None,
) -> None:
    """Sanitize one parsed content stream."""
    try:
        raw = stream_obj.read_bytes()
    except Exception as e:
        log_suppressed_error(
            logger, e, "Skipping unreadable content stream %s: %s", stream_obj.objgen, e
        )
        return

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unexpected end of stream", category=UserWarning
            )
            instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        instructions = []

    invalid_hex = 0
    if not instructions or _instructions_need_hex_repair(instructions):
        repaired_raw, invalid_hex = _strip_invalid_hex_chars(raw)
        if invalid_hex > 0:
            stats["hex_invalid_fixed"] += invalid_hex
            stream_obj.write(repaired_raw)
            raw = repaired_raw
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Unexpected end of stream", category=UserWarning
                )
                instructions = list(pikepdf.parse_content_stream(stream_obj))
        except Exception:
            return

    odd_hex = _count_odd_hex_string_tokens(raw)
    if odd_hex > 0:
        stats["hex_odd_fixed"] += odd_hex

    depth = 0
    maximum_depth = 0
    changed = False
    rewritten: list[Any] = []

    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            rewritten.append(instruction)
            continue

        op_name = str(instruction.operator)
        if op_name == "q":
            depth += 1
            maximum_depth = max(maximum_depth, depth)
        elif op_name == "Q":
            if depth > 0:
                depth -= 1

        new_operands = []
        operands_changed = False
        for operand in instruction.operands:
            replacement, operand_changed = _sanitize_operand(operand, stats)
            new_operands.append(replacement)
            operands_changed = operands_changed or operand_changed

        if operands_changed:
            changed = True
            rewritten.append(
                pikepdf.ContentStreamInstruction(new_operands, instruction.operator)
            )
        else:
            rewritten.append(instruction)

    if maximum_depth > _MAX_Q_NESTING:
        if not isinstance(resources, Dictionary) or bbox is None:
            raise UnsupportedPDFError(
                "Cannot safely reduce q/Q nesting without effective resources "
                "and a finite content bounding box."
            )
        rewritten, wrapped = _wrap_excess_q_nesting(
            pdf,
            rewritten,
            resources,
            bbox,
        )
        stats["q_nesting_rebalanced"] += wrapped * 2
        changed = changed or wrapped > 0

    if changed or odd_hex > 0 or invalid_hex > 0:
        stream_obj.write(pikepdf.unparse_content_stream(rewritten))


def _iter_owner_streams(owner: Any) -> list[Stream]:
    """Return stream objects that belong to a stream owner."""
    owner = _resolve(owner)
    if isinstance(owner, Stream):
        return [owner]
    if not isinstance(owner, Dictionary):
        return []

    contents = owner.get("/Contents")
    if contents is None:
        return []
    contents = _resolve(contents)
    if isinstance(contents, Stream):
        return [contents]
    if isinstance(contents, Array):
        streams: list[Stream] = []
        for item in contents:
            resolved = _resolve(item)
            if isinstance(resolved, Stream):
                streams.append(resolved)
        return streams
    return []


def _sanitize_object_graph(
    obj: Any,
    stats: dict[str, int],
    visited: set[tuple[int, int]],
) -> Any:
    """Iteratively sanitize dictionaries/arrays for implementation limits."""
    result = _resolve(obj)
    pending: list[tuple[Any, Dictionary | Stream | Array | None, Any]] = [
        (obj, None, None)
    ]
    while pending:
        raw, parent, location = pending.pop()
        current = _resolve(raw)

        if isinstance(current, (Dictionary, Array, Stream)):
            objgen = _indirect_objgen(current)
            if objgen is not None:
                if objgen in visited:
                    continue
                visited.add(objgen)

        if isinstance(current, (Dictionary, Stream)):
            original_keys = list(current.keys())
            for old_key in original_keys:
                new_key, long_fixed, utf8_fixed = _sanitize_name_token(old_key)
                if long_fixed:
                    stats["names_shortened"] += 1
                if utf8_fixed:
                    stats["utf8_names_fixed"] += 1
                if new_key != old_key:
                    value = current[old_key]
                    del current[old_key]
                    current[new_key] = value

            pending.extend(
                (current[key], current, key) for key in reversed(list(current.keys()))
            )
            continue

        if isinstance(current, Array):
            pending.extend(
                (item, current, index)
                for index, item in reversed(list(enumerate(current)))
            )
            continue

        replacement = current
        if isinstance(current, Name):
            replacement, long_fixed, utf8_fixed = _sanitize_name_token(current)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
        elif isinstance(current, str) and current.startswith("/"):
            replacement, long_fixed, utf8_fixed = _sanitize_name_token(current)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
        else:
            if isinstance(replacement, pikepdf.String):
                fixed_string = _fix_odd_hex_string(replacement)
                if fixed_string is not replacement:
                    stats["hex_odd_obj_fixed"] += 1
                    replacement = fixed_string

            replacement, string_changed = _sanitize_string(replacement)
            if string_changed:
                stats["strings_truncated"] += 1
            else:
                replacement, int_changed = _sanitize_integer(replacement)
                if int_changed:
                    stats["integers_clamped"] += 1
                else:
                    replacement, real_changed = _sanitize_real(replacement)
                    if real_changed:
                        stats["reals_normalized"] += 1

        if parent is None:
            result = replacement
        elif replacement is not raw:
            try:
                parent[location] = replacement
            except KeyError:
                # Some stream dictionary keys (e.g. /Length) are immutable.
                continue
    return result


def _cmap_has_cid_overflow(cmap_stream: Stream) -> bool:
    """Return True if embedded CMap contains CID values greater than 65535."""
    try:
        data = cmap_stream.read_bytes().decode("latin-1")
    except Exception:
        return False

    mode: str | None = None
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if line.endswith("begincidchar"):
            mode = "char"
            continue
        if line.endswith("endcidchar"):
            mode = None
            continue
        if line.endswith("begincidrange"):
            mode = "range"
            continue
        if line.endswith("endcidrange"):
            mode = None
            continue

        if mode == "char":
            match = _CIDCHAR_INT_RE.match(line)
            if match is not None:
                try:
                    if int(match.group(1)) > _MAX_CID_VALUE:
                        return True
                except ValueError:
                    continue
        elif mode == "range":
            match_int = _CIDS_INT_GROUP_RE.match(line)
            if match_int is not None:
                try:
                    start = int(match_int.group("src_hex"), 16)
                    end = int(match_int.group("end_hex"), 16)
                    cid_start = int(match_int.group("cid"))
                    if cid_start + max(0, end - start) > _MAX_CID_VALUE:
                        return True
                except ValueError:
                    continue
            match_hex = _CIDS_HEX_GROUP_RE.match(line)
            if match_hex is not None:
                try:
                    start = int(match_hex.group("src_hex"), 16)
                    end = int(match_hex.group("end_hex"), 16)
                    cid_start = int(match_hex.group("cid_hex"), 16)
                    if cid_start + max(0, end - start) > _MAX_CID_VALUE:
                        return True
                except ValueError:
                    continue
    return False


def _rewrite_cmap_block_begin(line: str, count: int) -> str:
    """Update the mapping count on a begincidchar/begincidrange line."""
    match = _BLOCK_BEGIN_RE.match(line)
    if match is None:
        return line
    return f"{match.group('indent')}{count}{match.group('suffix')}"


def _repair_cid_overflow_entries(
    cmap_stream: Stream,
    used_codes: set[CharacterCode],
) -> tuple[int, bool]:
    """Repair overflowing CMap entries, mapping used invalid codes to CID 0.

    Returns:
        Tuple of (number of modified/removed CMap entries, has_remaining_overflow).
    """
    try:
        data = cmap_stream.read_bytes().decode("latin-1")
    except Exception:
        return 0, False

    used_raw = {code for code in used_codes if isinstance(code, bytes)}
    used_values = {code for code in used_codes if isinstance(code, int)}

    def code_is_used(source: bytes) -> bool:
        return source in used_raw or int.from_bytes(source, "big") in used_values

    lines = data.splitlines()
    had_trailing_newline = data.endswith(("\n", "\r"))
    changed_entries = 0
    remaining_overflow = False
    rewritten: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.endswith("begincidchar") or stripped.endswith("begincidrange"):
            block_mode = "char" if stripped.endswith("begincidchar") else "range"
            end_marker = "endcidchar" if block_mode == "char" else "endcidrange"
            begin_line = line
            begin_match = _BLOCK_BEGIN_RE.match(begin_line)
            declared_count = (
                int(begin_match.group("count")) if begin_match is not None else None
            )
            count_delta = 0
            i += 1

            kept_entries: list[str] = []
            remapped_char_entries: list[str] = []
            while i < len(lines):
                entry_line = lines[i]
                entry_stripped = entry_line.strip()
                if entry_stripped.endswith(end_marker):
                    break

                if block_mode == "char":
                    match = _CIDCHAR_INT_RE.match(entry_stripped)
                    if match is not None:
                        try:
                            source = entry_stripped.split(None, 1)[0]
                            source_bytes = bytes.fromhex(source.strip("<>"))
                            cid_value = int(match.group(1))
                        except ValueError:
                            kept_entries.append(entry_line)
                            i += 1
                            continue

                        if cid_value > _MAX_CID_VALUE:
                            changed_entries += 1
                            if code_is_used(source_bytes):
                                kept_entries.append(f"{source} 0")
                            else:
                                count_delta -= 1
                            i += 1
                            continue

                else:
                    match_int = _CIDS_INT_GROUP_RE.match(entry_stripped)
                    match_hex = _CIDS_HEX_GROUP_RE.match(entry_stripped)

                    if match_int is not None or match_hex is not None:
                        try:
                            if match_int is not None:
                                start_code = int(match_int.group("src_hex"), 16)
                                end_code = int(match_int.group("end_hex"), 16)
                                cid_start = int(match_int.group("cid"))
                            else:
                                assert match_hex is not None
                                start_code = int(match_hex.group("src_hex"), 16)
                                end_code = int(match_hex.group("end_hex"), 16)
                                cid_start = int(match_hex.group("cid_hex"), 16)
                        except ValueError:
                            kept_entries.append(entry_line)
                            i += 1
                            continue

                        safe_limit = start_code + (_MAX_CID_VALUE - cid_start)
                        safe_end = min(end_code, safe_limit)
                        if cid_start > _MAX_CID_VALUE or safe_end < end_code:
                            overflow_start = max(start_code, safe_end + 1)
                            source_width_bytes = (
                                len(
                                    match_int.group("src_hex")
                                    if match_int is not None
                                    else match_hex.group("src_hex")
                                )
                                // 2
                            )
                            used_in_overflow = sorted(
                                {
                                    int.from_bytes(code, "big")
                                    for code in used_raw
                                    if len(code) == source_width_bytes
                                    and overflow_start
                                    <= int.from_bytes(code, "big")
                                    <= end_code
                                }
                                | {
                                    code
                                    for code in used_values
                                    if overflow_start <= code <= end_code
                                }
                            )
                            changed_entries += 1
                            count_delta -= 1
                            if safe_end >= start_code:
                                count_delta += 1
                                end_hex = (
                                    match_int.group("end_hex")
                                    if match_int is not None
                                    else match_hex.group("end_hex")
                                )
                                end_hex_len = len(end_hex)
                                new_end = f"<{safe_end:0{end_hex_len}X}>"
                                if match_int is not None:
                                    src = match_int.group("src")
                                    cid = match_int.group("cid")
                                    kept_entries.append(f"{src} {new_end} {cid}")
                                else:
                                    src = match_hex.group("src")
                                    cid_hex = match_hex.group("cid_hex")
                                    kept_entries.append(
                                        f"{src} {new_end} <{cid_hex.upper()}>"
                                    )
                            source_width = max(
                                len(match_int.group("src_hex"))
                                if match_int is not None
                                else len(match_hex.group("src_hex")),
                                len(match_int.group("end_hex"))
                                if match_int is not None
                                else len(match_hex.group("end_hex")),
                            )
                            remapped_char_entries.extend(
                                f"<{code:0{source_width}X}> 0"
                                for code in used_in_overflow
                            )
                            i += 1
                            continue

                kept_entries.append(entry_line)
                i += 1

            rewritten.append(
                _rewrite_cmap_block_begin(
                    begin_line,
                    max(0, declared_count + count_delta)
                    if declared_count is not None
                    else len(kept_entries),
                )
            )
            rewritten.extend(kept_entries)
            if i < len(lines):
                rewritten.append(lines[i])
            for offset in range(0, len(remapped_char_entries), 100):
                chunk = remapped_char_entries[offset : offset + 100]
                rewritten.append(f"{len(chunk)} begincidchar")
                rewritten.extend(chunk)
                rewritten.append("endcidchar")
            i += 1
            continue

        rewritten.append(line)
        i += 1

    if changed_entries > 0:
        new_data = "\n".join(rewritten)
        if had_trailing_newline:
            new_data += "\n"
        cmap_stream.write(new_data.encode("latin-1"))

    return changed_entries, remaining_overflow


def _ensure_no_cid_overflow(
    pdf: Pdf,
    usage_cache: FontUsageCache | None = None,
) -> int:
    """Repair CMap CID overflows and raise on unparseable remaining ones."""
    if usage_cache is not None:
        font_usage = usage_cache.get()
    else:
        font_usage = collect_font_usage(pdf)
    seen_fonts: set[tuple[int, int]] = set()
    repaired = 0
    for page in pdf.pages:
        for font_name, font_obj in iter_all_page_fonts(page):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue
            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_fonts:
                    continue
                seen_fonts.add(objgen)

            subtype = str(font.get("/Subtype"))
            if subtype != "/Type0":
                continue

            encoding = _resolve(font.get("/Encoding"))
            if not isinstance(encoding, Stream):
                continue

            repaired_here, remaining_overflow = _repair_cid_overflow_entries(
                encoding, font_usage.get(objgen, set())
            )
            repaired += repaired_here
            if repaired_here > 0:
                logger.warning(
                    "Removed, clipped, or remapped %d CID overflow entr(y/ies) "
                    "from CMap "
                    "for font %s",
                    repaired_here,
                    font_name,
                )

            if remaining_overflow or _cmap_has_cid_overflow(encoding):
                raise UnsupportedPDFError(
                    "PDF contains CID values greater than 65535 in an embedded CMap. "
                    "This cannot be repaired safely."
                )
    return repaired


def sanitize_structure_limits(
    pdf: Pdf,
    usage_cache: FontUsageCache | None = None,
) -> dict[str, int]:
    """Sanitize structural implementation-limit violations for PDF/A.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        usage_cache: Optional shared font usage cache. It is used for the
            CID overflow check and invalidated when this pass rewrites
            strings, names, operands or q/Q nesting in content streams.

    Returns:
        Dictionary with counts of fixes applied.
    """
    stats: dict[str, int] = {
        "strings_truncated": 0,
        "names_shortened": 0,
        "utf8_names_fixed": 0,
        "integers_clamped": 0,
        "reals_normalized": 0,
        "q_nesting_rebalanced": 0,
        "hex_odd_fixed": 0,
        "hex_odd_obj_fixed": 0,
        "hex_invalid_fixed": 0,
        "cid_overflow_entries_repaired": 0,
    }

    stats["cid_overflow_entries_repaired"] = _ensure_no_cid_overflow(pdf, usage_cache)

    visited: set[tuple[int, int]] = set()
    for obj in pdf.objects:
        _sanitize_object_graph(obj, stats, visited)

    processed_streams: set[tuple[int, int]] = set()
    for page in pdf.pages:
        contents = _resolve(page.obj.get("/Contents"))
        if isinstance(contents, Array):
            page.contents_coalesce()
        resources = get_page_resources(page)
        if not isinstance(resources, Dictionary):
            resources = Dictionary()
            page.obj[Name.Resources] = resources
        bbox = _content_bbox(page.obj)
        if bbox is None:
            try:
                bbox = Array(list(page.mediabox))
            except Exception:
                bbox = None
        for stream in _iter_owner_streams(page.obj):
            objgen = _indirect_objgen(stream)
            if objgen is not None:
                if objgen in processed_streams:
                    continue
                processed_streams.add(objgen)
            _sanitize_content_stream(pdf, stream, stats, resources, bbox)

    for page in pdf.pages:
        for owner, resources in _iter_content_streams_with_resources(page):
            resources = _resolve(resources)
            if not isinstance(resources, Dictionary):
                resources = None
            bbox = _content_bbox(owner)
            if bbox is None and owner is page.obj:
                try:
                    bbox = Array(list(page.mediabox))
                except Exception:
                    bbox = None
            for stream in _iter_owner_streams(owner):
                objgen = _indirect_objgen(stream)
                if objgen is not None:
                    if objgen in processed_streams:
                        continue
                    processed_streams.add(objgen)
                _sanitize_content_stream(pdf, stream, stats, resources, bbox)

    # The passes above rewrite strings, names, operands and q/Q nesting in
    # content streams, any of which can change the collected font usage.
    if usage_cache is not None and any(
        count > 0
        for key, count in stats.items()
        if key != "cid_overflow_entries_repaired"
    ):
        usage_cache.invalidate()

    logger.info(
        "Structure limits sanitized: %d strings truncated, %d names shortened, "
        "%d UTF-8 names fixed, %d integers clamped, %d out-of-range reals sanitized, "
        "%d q/Q nesting ops rebalanced, %d odd hex strings fixed, "
        "%d odd hex object strings fixed, "
        "%d invalid hex strings repaired",
        stats["strings_truncated"],
        stats["names_shortened"],
        stats["utf8_names_fixed"],
        stats["integers_clamped"],
        stats["reals_normalized"],
        stats["q_nesting_rebalanced"],
        stats["hex_odd_fixed"],
        stats["hex_odd_obj_fixed"],
        stats["hex_invalid_fixed"],
    )

    return stats
