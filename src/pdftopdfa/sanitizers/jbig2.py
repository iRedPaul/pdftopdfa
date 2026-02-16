# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""JBIG2 external globals sanitizer for PDF/A compliance.

This module detects JBIG2-compressed streams with external globals references,
which are forbidden in PDF/A. It inlines the globals data by prepending the
globals segments to the page data, producing a self-contained JBIG2 bitstream.
"""

import logging
import struct
import zlib

from pikepdf import Array, Name, Pdf, Stream

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

# JBIG2 refinement segment types forbidden by ISO 19005-2, section 6.1.4.2.
# Type 40: Intermediate generic refinement region
# Type 42: Immediate generic refinement region
# Type 43: Immediate lossless generic refinement region
_FORBIDDEN_JBIG2_SEGMENT_TYPES = frozenset({40, 42, 43})


def _has_jbig2_filter(stream: Stream) -> bool:
    """Check if a stream uses JBIG2Decode filter.

    Handles both single filter and filter arrays.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        True if the stream uses JBIG2Decode, False otherwise.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False

        filter_obj = _resolve_indirect(filter_obj)

        # Single filter case
        if isinstance(filter_obj, Name):
            return str(filter_obj) == "/JBIG2Decode"

        # Filter array case
        if isinstance(filter_obj, Array):
            for f in filter_obj:
                f = _resolve_indirect(f)
                if isinstance(f, Name) and str(f) == "/JBIG2Decode":
                    return True

        return False
    except Exception:
        return False


def _has_external_globals(stream: Stream) -> bool:
    """Check if a JBIG2 stream has external globals reference.

    External globals are stored in a separate stream referenced via
    /DecodeParms -> /JBIG2Globals. This is forbidden in PDF/A.

    Args:
        stream: A pikepdf Stream object with JBIG2Decode filter.

    Returns:
        True if the stream references external globals, False otherwise.
    """
    try:
        decode_parms = stream.get("/DecodeParms")
        if decode_parms is None:
            return False

        decode_parms = _resolve_indirect(decode_parms)

        # Handle array of decode params (for filter arrays)
        if isinstance(decode_parms, Array):
            for parm in decode_parms:
                parm = _resolve_indirect(parm)
                if parm is not None and parm.get("/JBIG2Globals") is not None:
                    return True
            return False

        # Single decode params dict
        return decode_parms.get("/JBIG2Globals") is not None

    except Exception:
        return False


def _has_jbig2_filter_single(stream: Stream) -> bool:
    """Check if a stream has exactly ``/Filter /JBIG2Decode`` (not an array).

    Globals inlining only works when JBIG2Decode is the sole filter.
    Filter arrays (e.g. ``[/JBIG2Decode /FlateDecode]``) are not supported.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        True if the stream has a single JBIG2Decode filter, False otherwise.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False
        filter_obj = _resolve_indirect(filter_obj)
        return isinstance(filter_obj, Name) and str(filter_obj) == "/JBIG2Decode"
    except Exception:
        return False


def _get_globals_stream(stream: Stream) -> Stream | None:
    """Extract the JBIG2 globals stream from DecodeParms.

    Args:
        stream: A pikepdf Stream with JBIG2Decode filter.

    Returns:
        The globals Stream, or None if not found or unsupported layout.
    """
    try:
        decode_parms = stream.get("/DecodeParms")
        if decode_parms is None:
            return None
        decode_parms = _resolve_indirect(decode_parms)

        # Array DecodeParms not supported for inlining
        if isinstance(decode_parms, Array):
            return None

        globals_ref = decode_parms.get("/JBIG2Globals")
        if globals_ref is None:
            return None
        globals_obj = _resolve_indirect(globals_ref)
        if isinstance(globals_obj, Stream):
            return globals_obj
        return None
    except Exception:
        return None


def _get_jbig2_filter_index(stream: Stream) -> int | None:
    """Return the index of JBIG2Decode in a filter array.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        The index of JBIG2Decode in the filter array, or None if the
        filter is not an array or does not contain JBIG2Decode.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return None
        filter_obj = _resolve_indirect(filter_obj)
        if not isinstance(filter_obj, Array):
            return None
        for i, f in enumerate(filter_obj):
            f = _resolve_indirect(f)
            if isinstance(f, Name) and str(f) == "/JBIG2Decode":
                return i
        return None
    except Exception:
        return None


def _get_globals_from_array(stream: Stream, jbig2_idx: int) -> Stream | None:
    """Extract the JBIG2 globals stream from an array DecodeParms entry.

    Args:
        stream: A pikepdf Stream with a filter array containing JBIG2Decode.
        jbig2_idx: The index of JBIG2Decode in the filter array.

    Returns:
        The globals Stream, or None if not found.
    """
    try:
        decode_parms = stream.get("/DecodeParms")
        if decode_parms is None:
            return None
        decode_parms = _resolve_indirect(decode_parms)
        if not isinstance(decode_parms, Array):
            return None
        if jbig2_idx >= len(decode_parms):
            return None
        parm = _resolve_indirect(decode_parms[jbig2_idx])
        if parm is None:
            return None
        globals_ref = parm.get("/JBIG2Globals")
        if globals_ref is None:
            return None
        globals_obj = _resolve_indirect(globals_ref)
        if isinstance(globals_obj, Stream):
            return globals_obj
        return None
    except Exception:
        return None


def _strip_preceding_filters(data: bytes, filters: list) -> bytes | None:
    """Decode filters that precede JBIG2Decode to extract the raw bitstream.

    Applies decoding in order so the returned data is the raw JBIG2 bitstream
    ready for globals inlining.

    Args:
        data: The raw stream bytes.
        filters: Resolved filter Name objects preceding JBIG2Decode.

    Returns:
        The decoded JBIG2 bitstream, or None if an unsupported filter is
        encountered.
    """
    for f in filters:
        fname = str(f)
        if fname == "/FlateDecode":
            try:
                data = zlib.decompress(data)
            except zlib.error:
                return None
        elif fname == "/ASCIIHexDecode":
            try:
                hex_str = data.decode("ascii").strip().rstrip(">")
                data = bytes.fromhex(hex_str)
            except (ValueError, UnicodeDecodeError):
                return None
        else:
            return None
    return data


def _has_refinement_segments(data: bytes) -> bool:
    """Check if JBIG2 data contains forbidden refinement coding segments.

    Parses JBIG2 segment headers (without file header, as used in PDF)
    and returns True if any segment has a type in
    ``_FORBIDDEN_JBIG2_SEGMENT_TYPES``.

    ISO 19005-2, section 6.1.4.2 forbids JBIG2 refinement coding as
    defined in Annex D of ISO/IEC 14492.

    Args:
        data: Raw JBIG2 bitstream bytes (no file header).

    Returns:
        True if forbidden refinement segments are detected.
    """
    pos = 0
    length = len(data)

    while pos < length:
        # Minimum: segment number (4) + flags (1) + ref-count byte (1)
        if pos + 6 > length:
            break

        # Segment number (4 bytes, big-endian)
        seg_num = struct.unpack_from(">I", data, pos)[0]
        pos += 4

        # Segment header flags (1 byte)
        flags = data[pos]
        seg_type = flags & 0x3F
        page_assoc_large = bool(flags & 0x40)
        pos += 1

        if seg_type in _FORBIDDEN_JBIG2_SEGMENT_TYPES:
            return True

        # End of file segment (type 51) — no data follows
        if seg_type == 51:
            break

        # Referred-to segment count and retention flags
        if pos >= length:
            break
        count_indicator = (data[pos] >> 5) & 7

        if count_indicator <= 4:
            ref_count = count_indicator
            pos += 1  # short form: 1 byte total
        elif count_indicator == 7:
            # Long form: 1 indicator byte + 4-byte count + retention bytes
            pos += 1
            if pos + 4 > length:
                break
            ref_count = struct.unpack_from(">I", data, pos)[0] & 0x1FFFFFFF
            pos += 4
            pos += (ref_count + 7) // 8  # retention flag bytes
        else:
            # Reserved values 5, 6 — cannot parse further
            break

        # Referred-to segment numbers (size depends on current segment number)
        if seg_num <= 256:
            ref_seg_size = 1
        elif seg_num <= 65536:
            ref_seg_size = 2
        else:
            ref_seg_size = 4
        pos += ref_count * ref_seg_size

        # Page association
        pos += 4 if page_assoc_large else 1

        # Segment data length (4 bytes)
        if pos + 4 > length:
            break
        data_length = struct.unpack_from(">I", data, pos)[0]
        pos += 4

        # Unknown length — cannot skip reliably
        if data_length == 0xFFFFFFFF:
            break

        pos += data_length

    return False


def _reencode_jbig2_to_flatedecode(stream: Stream) -> bool:
    """Re-encode a JBIG2 stream to FlateDecode as lossless fallback.

    Decodes JBIG2 image data to raw pixels via pikepdf/QPDF and writes
    them back under FlateDecode.  Requires QPDF to have JBIG2 decode
    support (jbig2dec library).

    Returns True on success, False on failure.
    """
    try:
        decoded_data = stream.read_bytes()
        stream.write(decoded_data)
        try:
            del stream["/DecodeParms"]
        except (KeyError, AttributeError):
            pass
        return True
    except Exception as e:
        logger.debug("Failed to re-encode JBIG2 to FlateDecode: %s", e)
        return False


def _convert_jbig2_array_stream(stream: Stream, pdf: Pdf) -> bool:
    """Inline external JBIG2 globals in a stream with a filter array.

    For single-element arrays ``[/JBIG2Decode]``, inlines globals the same
    way as a single filter.  For multi-element arrays where JBIG2Decode is
    the last filter (e.g. ``[/FlateDecode /JBIG2Decode]``), strips the
    preceding filters, inlines globals, and writes back with just
    ``/JBIG2Decode``.

    Args:
        stream: A pikepdf Stream with a filter array containing JBIG2Decode.
        pdf: The pikepdf Pdf object (unused but kept for API consistency).

    Returns:
        True if conversion succeeded, False otherwise.
    """
    try:
        filter_obj = _resolve_indirect(stream.get("/Filter"))
        if not isinstance(filter_obj, Array):
            return False

        filters = [_resolve_indirect(f) for f in filter_obj]
        jbig2_idx = next(
            (
                i
                for i, f in enumerate(filters)
                if isinstance(f, Name) and str(f) == "/JBIG2Decode"
            ),
            None,
        )
        if jbig2_idx is None:
            return False

        globals_stream = _get_globals_from_array(stream, jbig2_idx)
        if globals_stream is None:
            return False
        globals_data = globals_stream.read_bytes()

        if len(filters) == 1:
            # [/JBIG2Decode] — equivalent to single filter
            page_data = stream.read_raw_bytes()
            stream.write(globals_data + page_data, filter=Name("/JBIG2Decode"))
            return True

        # Multi-filter: JBIG2Decode must be the last filter
        if jbig2_idx != len(filters) - 1:
            return False

        # Strip preceding filters to get the raw JBIG2 bitstream
        jbig2_data = _strip_preceding_filters(
            stream.read_raw_bytes(), filters[:jbig2_idx]
        )
        if jbig2_data is None:
            return False

        stream.write(globals_data + jbig2_data, filter=Name("/JBIG2Decode"))
        return True

    except Exception as e:
        logger.debug("Cannot convert JBIG2 array stream: %s", e)
        return False


def _convert_jbig2_stream(stream: Stream, pdf: Pdf) -> bool:
    """Inline external JBIG2 globals into the stream.

    Prepends the decoded globals segments to the raw JBIG2 page data,
    producing a self-contained bitstream. The ``/JBIG2Globals`` reference
    is removed.

    Args:
        stream: A pikepdf Stream object with JBIG2 compression.
        pdf: The pikepdf Pdf object (unused but kept for API consistency).

    Returns:
        True if inlining succeeded, False otherwise.
    """
    try:
        globals_stream = _get_globals_stream(stream)
        if globals_stream is None:
            return False
        globals_data = globals_stream.read_bytes()
        page_data = stream.read_raw_bytes()
        combined = globals_data + page_data
        stream.write(combined, filter=Name("/JBIG2Decode"))
        return True
    except Exception as e:
        logger.debug("Cannot inline JBIG2 globals: %s", e)
        return False


def convert_jbig2_external_globals(pdf: Pdf) -> dict[str, int]:
    """Inline external globals and detect forbidden refinement in JBIG2 streams.

    JBIG2 compression with external globals is forbidden in PDF/A.
    This function detects such streams and inlines the globals data
    so the bitstream is self-contained.

    Additionally, ISO 19005-2 section 6.1.4.2 forbids JBIG2 refinement
    coding.  Streams containing refinement segments are re-encoded to
    FlateDecode as a lossless fallback.

    Args:
        pdf: pikepdf Pdf object (modified in place if conversion succeeds).

    Returns:
        Dictionary with counts:
        - converted: Number of streams with external globals inlined
        - reencoded: Number of streams re-encoded to FlateDecode
          (refinement detected)
        - failed: Number of streams that could not be converted
    """
    converted = 0
    reencoded = 0
    failed = 0

    seen: set[tuple[int, int]] = set()
    for obj in pdf.objects:
        try:
            objgen = obj.objgen
            if objgen in seen:
                continue
            seen.add(objgen)
            obj = _resolve_indirect(obj)

            if not isinstance(obj, Stream):
                continue

            if not _has_jbig2_filter(obj):
                continue

            # Step 1: Inline external globals if present
            if _has_external_globals(obj):
                if _has_jbig2_filter_single(obj):
                    if _convert_jbig2_stream(obj, pdf):
                        converted += 1
                        logger.debug("Inlined JBIG2 globals: %s", obj.objgen)
                    else:
                        failed += 1
                        logger.debug("Failed to inline JBIG2 globals: %s", obj.objgen)
                        continue
                else:
                    if _convert_jbig2_array_stream(obj, pdf):
                        converted += 1
                        logger.debug("Converted JBIG2 array stream: %s", obj.objgen)
                    else:
                        failed += 1
                        logger.debug(
                            "Failed to convert JBIG2 array stream: %s",
                            obj.objgen,
                        )
                        continue

            # Step 2: Check for forbidden refinement segments.
            # Only check single-filter streams (after inlining, filter is
            # always single; non-inlined array filters are unusual and the
            # raw bytes would need pre-decoding).
            if not _has_jbig2_filter_single(obj):
                continue
            try:
                raw_data = obj.read_raw_bytes()
                if _has_refinement_segments(raw_data):
                    logger.debug("JBIG2 refinement segments detected: %s", obj.objgen)
                    if _reencode_jbig2_to_flatedecode(obj):
                        reencoded += 1
                        logger.debug("Re-encoded JBIG2 to FlateDecode: %s", obj.objgen)
                    else:
                        failed += 1
                        logger.debug("Failed to re-encode JBIG2: %s", obj.objgen)
            except Exception as e:
                logger.debug("Error checking JBIG2 refinement: %s", e)

        except Exception as e:
            logger.debug("Error processing object: %s", e)

    if converted > 0:
        logger.info("%d JBIG2 stream(s) with external globals inlined", converted)

    if reencoded > 0:
        logger.info(
            "%d JBIG2 stream(s) with refinement re-encoded to FlateDecode",
            reencoded,
        )

    if failed > 0:
        logger.warning(
            "%d JBIG2 stream(s) could not be converted "
            "(unsupported filter configuration or decode failure)",
            failed,
        )

    return {"converted": converted, "reencoded": reencoded, "failed": failed}
