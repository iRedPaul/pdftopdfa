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
from decimal import Decimal
from typing import Any

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import UnsupportedPDFError
from ..fonts.glyph_usage import _iter_content_streams_with_resources
from ..fonts.traversal import iter_all_page_fonts
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
_CIDS_INT_RE = re.compile(r"^<[^>]+>\s+<[^>]+>\s+(-?\d+)$")
_CIDCHAR_INT_RE = re.compile(r"^<[^>]+>\s+(-?\d+)$")
_CIDS_HEX_RE = re.compile(r"^<[^>]+>\s+<[^>]+>\s+<([0-9A-Fa-f]+)>$")


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


def _sanitize_operand(value: Any, stats: dict[str, int]) -> tuple[Any, bool]:
    """Sanitize an operand in a parsed content stream instruction."""
    value = _resolve(value)
    changed = False

    if isinstance(value, Name):
        new_value, long_fixed, utf8_fixed = _sanitize_name_token(value)
        if long_fixed:
            stats["names_shortened"] += 1
        if utf8_fixed:
            stats["utf8_names_fixed"] += 1
        return new_value, bool(long_fixed or utf8_fixed)
    if isinstance(value, str) and value.startswith("/"):
        new_value, long_fixed, utf8_fixed = _sanitize_name_token(value)
        if long_fixed:
            stats["names_shortened"] += 1
        if utf8_fixed:
            stats["utf8_names_fixed"] += 1
        return new_value, bool(long_fixed or utf8_fixed)

    new_value, string_changed = _sanitize_string(value)
    if string_changed:
        stats["strings_truncated"] += 1
        return new_value, True

    new_value, int_changed = _sanitize_integer(value)
    if int_changed:
        stats["integers_clamped"] += 1
        return new_value, True

    new_value, real_changed = _sanitize_real(value)
    if real_changed:
        stats["reals_normalized"] += 1
        return new_value, True

    if isinstance(value, Array):
        items = list(value)
        for idx, item in enumerate(items):
            replacement, item_changed = _sanitize_operand(item, stats)
            if item_changed:
                value[idx] = replacement
                changed = True
        return value, changed

    if isinstance(value, (Dictionary, Stream)):
        keys = list(value.keys())
        for old_key in keys:
            new_key, long_fixed, utf8_fixed = _sanitize_name_token(old_key)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
            if new_key != old_key:
                current = value[old_key]
                del value[old_key]
                value[new_key] = current
                changed = True

        for key in list(value.keys()):
            current = value[key]
            replacement, item_changed = _sanitize_operand(current, stats)
            if item_changed:
                try:
                    value[key] = replacement
                except KeyError:
                    # Some stream dictionary keys (e.g. /Length) are immutable.
                    continue
                changed = True
        return value, changed

    return value, False


def _validate_text_operands(operator_name: str, operands: Any) -> bool:
    """Validate parsed text-showing operator operand structure."""
    if operator_name == "Tj" or operator_name == "'":
        return len(operands) == 1 and isinstance(operands[0], pikepdf.String)

    if operator_name == '"':
        if len(operands) != 3:
            return False
        return isinstance(operands[2], pikepdf.String)

    if operator_name == "TJ":
        if len(operands) != 1 or not isinstance(operands[0], Array):
            return False
        for item in operands[0]:
            if (
                isinstance(item, pikepdf.String)
                or isinstance(item, int)
                or isinstance(item, float)
                or isinstance(item, Decimal)
            ):
                continue
            return False
        return True

    return True


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


def _sanitize_content_stream(stream_obj: Stream, stats: dict[str, int]) -> None:
    """Sanitize one parsed content stream."""
    try:
        raw = stream_obj.read_bytes()
    except Exception as e:
        logger.debug("Skipping unreadable content stream %s: %s", stream_obj.objgen, e)
        return
    odd_hex = _count_odd_hex_string_tokens(raw)
    if odd_hex > 0:
        stats["hex_odd_fixed"] += odd_hex

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unexpected end of stream", category=UserWarning
            )
            instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        return

    inside_bt = False
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        op_name = str(instruction.operator)
        if op_name == "BT":
            inside_bt = True
            continue
        if op_name == "ET":
            inside_bt = False
            continue
        if op_name not in _TEXT_OPERATORS:
            continue
        if not inside_bt and len(instruction.operands) == 0:
            # Cross-stream boundary: operator split across Contents array
            # entries — the operands reside in the preceding stream.
            continue
        if not _validate_text_operands(op_name, instruction.operands):
            raise UnsupportedPDFError(
                "Malformed hexadecimal string in text operator "
                "(contains non-hexadecimal characters) is not safely repairable."
            )

    depth = 0
    suppressed_q = 0
    changed = False
    rewritten: list[Any] = []

    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            rewritten.append(instruction)
            continue

        op_name = str(instruction.operator)
        if op_name == "q":
            if depth >= _MAX_Q_NESTING:
                suppressed_q += 1
                stats["q_nesting_rebalanced"] += 1
                changed = True
                continue
            depth += 1
        elif op_name == "Q":
            if suppressed_q > 0:
                suppressed_q -= 1
                stats["q_nesting_rebalanced"] += 1
                changed = True
                continue
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

    if changed or odd_hex > 0:
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
    """Recursively sanitize dictionaries/arrays for implementation limits."""
    obj = _resolve(obj)

    if isinstance(obj, (Dictionary, Array, Stream)):
        objgen = _indirect_objgen(obj)
        if objgen is not None:
            if objgen in visited:
                return obj
            visited.add(objgen)

    if isinstance(obj, (Dictionary, Stream)):
        original_keys = list(obj.keys())
        for old_key in original_keys:
            new_key, long_fixed, utf8_fixed = _sanitize_name_token(old_key)
            if long_fixed:
                stats["names_shortened"] += 1
            if utf8_fixed:
                stats["utf8_names_fixed"] += 1
            if new_key != old_key:
                value = obj[old_key]
                del obj[old_key]
                obj[new_key] = value

        for key in list(obj.keys()):
            current = obj[key]
            replacement = _sanitize_object_graph(current, stats, visited)
            if replacement is current:
                continue
            try:
                obj[key] = replacement
            except KeyError:
                # Some stream dictionary keys (e.g. /Length) are immutable.
                continue
        return obj

    if isinstance(obj, Array):
        for idx, item in enumerate(list(obj)):
            obj[idx] = _sanitize_object_graph(item, stats, visited)
        return obj

    if isinstance(obj, Name):
        replacement, long_fixed, utf8_fixed = _sanitize_name_token(obj)
        if long_fixed:
            stats["names_shortened"] += 1
        if utf8_fixed:
            stats["utf8_names_fixed"] += 1
        return replacement
    if isinstance(obj, str) and obj.startswith("/"):
        replacement, long_fixed, utf8_fixed = _sanitize_name_token(obj)
        if long_fixed:
            stats["names_shortened"] += 1
        if utf8_fixed:
            stats["utf8_names_fixed"] += 1
        return replacement

    obj, string_changed = _sanitize_string(obj)
    if string_changed:
        stats["strings_truncated"] += 1
        return obj

    obj, int_changed = _sanitize_integer(obj)
    if int_changed:
        stats["integers_clamped"] += 1
        return obj

    obj, real_changed = _sanitize_real(obj)
    if real_changed:
        stats["reals_normalized"] += 1
        return obj

    return obj


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
            match_int = _CIDS_INT_RE.match(line)
            if match_int is not None:
                try:
                    if int(match_int.group(1)) > _MAX_CID_VALUE:
                        return True
                except ValueError:
                    continue
            match_hex = _CIDS_HEX_RE.match(line)
            if match_hex is not None:
                try:
                    if int(match_hex.group(1), 16) > _MAX_CID_VALUE:
                        return True
                except ValueError:
                    continue
    return False


def _ensure_no_cid_overflow(pdf: Pdf) -> None:
    """Raise UnsupportedPDFError for non-repairable CID overflows."""
    seen_fonts: set[tuple[int, int]] = set()
    for page in pdf.pages:
        for _font_name, font_obj in iter_all_page_fonts(page):
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

            if _cmap_has_cid_overflow(encoding):
                raise UnsupportedPDFError(
                    "PDF contains CID values greater than 65535 in an embedded CMap. "
                    "This cannot be repaired safely."
                )


def sanitize_structure_limits(pdf: Pdf) -> dict[str, int]:
    """Sanitize structural implementation-limit violations for PDF/A."""
    stats: dict[str, int] = {
        "strings_truncated": 0,
        "names_shortened": 0,
        "utf8_names_fixed": 0,
        "integers_clamped": 0,
        "reals_normalized": 0,
        "q_nesting_rebalanced": 0,
        "hex_odd_fixed": 0,
    }

    _ensure_no_cid_overflow(pdf)

    visited: set[tuple[int, int]] = set()
    for obj in pdf.objects:
        _sanitize_object_graph(obj, stats, visited)

    processed_streams: set[tuple[int, int]] = set()
    for page in pdf.pages:
        for stream in _iter_owner_streams(page.obj):
            objgen = _indirect_objgen(stream)
            if objgen is not None:
                if objgen in processed_streams:
                    continue
                processed_streams.add(objgen)
            _sanitize_content_stream(stream, stats)

    for page in pdf.pages:
        for owner, _resources in _iter_content_streams_with_resources(page):
            for stream in _iter_owner_streams(owner):
                objgen = _indirect_objgen(stream)
                if objgen is not None:
                    if objgen in processed_streams:
                        continue
                    processed_streams.add(objgen)
                _sanitize_content_stream(stream, stats)

    logger.info(
        "Structure limits sanitized: %d strings truncated, %d names shortened, "
        "%d UTF-8 names fixed, %d integers clamped, %d out-of-range reals sanitized, "
        "%d q/Q nesting ops rebalanced, %d odd hex strings fixed",
        stats["strings_truncated"],
        stats["names_shortened"],
        stats["utf8_names_fixed"],
        stats["integers_clamped"],
        stats["reals_normalized"],
        stats["q_nesting_rebalanced"],
        stats["hex_odd_fixed"],
    )

    return stats
