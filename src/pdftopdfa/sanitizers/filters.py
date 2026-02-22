# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Filter conversion for PDF/A compliance.

This module converts LZW-compressed streams to FlateDecode, removes /Crypt
filters (both forbidden per ISO 19005-2, 6.1.8), strips external stream
keys /F, /FFilter, /FDecodeParms (forbidden per ISO 19005-2, 6.1.7.1),
re-encodes non-image streams to fix /Length mismatches (rule 6.1.7.1),
and re-encodes inline images with non-Table-6 filters (rule 6.1.10-1).
"""

import logging
import warnings
import zlib

from pikepdf import Array, Dictionary, Name, Pdf, Stream, parse_content_stream
from pikepdf import unparse_content_stream as _unparse_content_stream

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

_CANONICAL_FILTER_NAMES_BY_LOWER: dict[str, str] = {
    "/ahx": "/ASCIIHexDecode",
    "/asciihexdecode": "/ASCIIHexDecode",
    "/a85": "/ASCII85Decode",
    "/ascii85decode": "/ASCII85Decode",
    "/lzw": "/LZWDecode",
    "/lzwdecode": "/LZWDecode",
    "/fl": "/FlateDecode",
    "/flatedecode": "/FlateDecode",
    "/rl": "/RunLengthDecode",
    "/runlengthdecode": "/RunLengthDecode",
    "/ccf": "/CCITTFaxDecode",
    "/ccittfaxdecode": "/CCITTFaxDecode",
    "/dct": "/DCTDecode",
    "/dctdecode": "/DCTDecode",
    "/jbig2decode": "/JBIG2Decode",
    "/jpxdecode": "/JPXDecode",
    "/crypt": "/Crypt",
}

_INLINE_FILTER_KEYS = frozenset({"/F", "/Filter"})
_INLINE_DECODE_PARMS_KEYS = frozenset({"/DP", "/DecodeParms"})

_INLINE_IMAGE_ALLOWED_FILTER_NAMES: frozenset[str] = frozenset(
    {
        "/ASCIIHexDecode",
        "/ASCII85Decode",
        "/FlateDecode",
        "/RunLengthDecode",
        "/CCITTFaxDecode",
        "/DCTDecode",
    }
)


def _normalize_inline_filter_name(filter_name: str) -> str:
    """Normalize filter names to canonical PDF names."""
    return _CANONICAL_FILTER_NAMES_BY_LOWER.get(
        filter_name.lower(),
        filter_name,
    )


def _normalize_inline_filter_object(
    filter_obj,
) -> tuple[list[str], Name | Array, bool] | None:
    """Normalize inline-image /Filter object to full-name Name/Array."""
    filter_obj = _resolve_indirect(filter_obj)

    if isinstance(filter_obj, Name):
        original_name = str(filter_obj)
        normalized_name = _normalize_inline_filter_name(str(filter_obj))
        return (
            [normalized_name],
            Name(normalized_name),
            normalized_name != original_name,
        )

    if isinstance(filter_obj, Array):
        normalized_names: list[str] = []
        original_names: list[str] = []
        for entry in filter_obj:
            entry = _resolve_indirect(entry)
            if not isinstance(entry, Name):
                return None
            original_name = str(entry)
            original_names.append(original_name)
            normalized_names.append(_normalize_inline_filter_name(original_name))
        return (
            normalized_names,
            Array([Name(name) for name in normalized_names]),
            normalized_names != original_names,
        )

    return None


def _normalize_stream_filter_names(stream: Stream) -> bool:
    """Normalize stream /Filter names to canonical spellings."""
    filter_obj = stream.get("/Filter")
    if filter_obj is None:
        return False

    normalized = _normalize_inline_filter_object(filter_obj)
    if normalized is None:
        return False

    _, normalized_filter_obj, changed = normalized
    if changed:
        stream[Name("/Filter")] = normalized_filter_obj
    return changed


def _unparse_inline_image_token(token) -> bytes:
    """Serialize one inline-image dictionary token to PDF bytes."""
    token = _resolve_indirect(token)

    if isinstance(token, bool):
        return b"true" if token else b"false"
    if isinstance(token, int | float):
        return str(token).encode("ascii")

    try:
        unparsed = token.unparse(resolved=True)
        if isinstance(unparsed, bytes):
            return unparsed
    except (AttributeError, TypeError):
        pass

    raise TypeError(f"Unsupported inline-image token: {type(token)!r}")


def _replace_inline_filter_tokens(
    image_object,
    filter_obj: Name | Array | None,
    decode_parms,
) -> tuple:
    """Replace /F(/Filter) and /DP(/DecodeParms) in inline-image tokens."""
    tokens = list(image_object)
    replaced: list = []
    i = 0

    while i + 1 < len(tokens):
        key = _resolve_indirect(tokens[i])
        value = tokens[i + 1]
        i += 2

        key_name = str(key) if isinstance(key, Name) else None
        if key_name in _INLINE_FILTER_KEYS or key_name in _INLINE_DECODE_PARMS_KEYS:
            continue

        replaced.extend([tokens[i - 2], value])

    if i < len(tokens):
        replaced.extend(tokens[i:])

    if filter_obj is not None:
        replaced.extend([Name("/Filter"), filter_obj])
        if decode_parms is not None:
            replaced.extend([Name("/DecodeParms"), decode_parms])

    return tuple(replaced)


def _build_inline_image_bytes(image_object: tuple, image_data: bytes) -> bytes:
    """Build BI/ID/EI bytes from inline-image metadata tokens and raw payload."""
    metadata = b" ".join(_unparse_inline_image_token(token) for token in image_object)
    if metadata:
        return b"BI\n" + metadata + b"\nID\n" + image_data + b"EI"
    return b"BI\nID\n" + image_data + b"EI"


def _create_inline_image_from_parts(image_object: tuple, image_data: bytes):
    """Create a PdfInlineImage object from metadata tokens and raw payload."""
    with Pdf.new() as temp_pdf:
        temp_stream = temp_pdf.make_stream(
            _build_inline_image_bytes(image_object, image_data)
        )

        for operands, operator in parse_content_stream(temp_stream):
            if str(operator) == "INLINE IMAGE" and operands:
                return operands[0]

    raise ValueError("Failed to parse generated inline image")


def _decode_inline_image_payload(
    encoded_data: bytes,
    filter_obj: Name | Array,
    decode_parms,
) -> bytes:
    """Decode an inline-image payload using its normalized filter chain."""
    stripped_data = encoded_data.rstrip(b"\t\n\f\r ")
    candidates = [stripped_data] if stripped_data != encoded_data else []
    candidates.append(encoded_data)
    last_error: Exception | None = None

    for candidate in candidates:
        with Pdf.new() as temp_pdf:
            temp_stream = Stream(temp_pdf, candidate)
            temp_stream[Name("/Filter")] = filter_obj
            if decode_parms is not None:
                temp_stream[Name("/DecodeParms")] = decode_parms
            try:
                return temp_stream.read_bytes()
            except Exception as e:  # pragma: no cover - error path
                last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError("Unable to decode inline-image payload")


def _strip_crypt_from_filter_chain(
    normalized_filter_names: list[str],
    decode_parms,
) -> tuple[Name | Array | None, Array | Dictionary | None]:
    """Remove /Crypt from a normalized inline-image filter chain."""
    keep_indices = [
        idx
        for idx, filter_name in enumerate(normalized_filter_names)
        if filter_name != "/Crypt"
    ]
    if not keep_indices:
        return None, None

    remaining_names = [normalized_filter_names[idx] for idx in keep_indices]
    if len(remaining_names) == 1:
        filter_obj: Name | Array | None = Name(remaining_names[0])
    else:
        filter_obj = Array([Name(name) for name in remaining_names])

    if decode_parms is None:
        return filter_obj, None

    decode_parms = _resolve_indirect(decode_parms)
    if not isinstance(decode_parms, Array):
        return filter_obj, None
    if len(decode_parms) != len(normalized_filter_names):
        return filter_obj, None

    remaining_parms = [decode_parms[idx] for idx in keep_indices]
    if not remaining_parms:
        return filter_obj, None
    if len(remaining_parms) == 1:
        entry = _resolve_indirect(remaining_parms[0])
        return filter_obj, entry if entry is not None else None
    if all(_resolve_indirect(entry) is None for entry in remaining_parms):
        return filter_obj, None
    return filter_obj, Array(remaining_parms)


def _sanitize_inline_image_filters(
    inline_image,
    *,
    convert_lzw: bool,
    remove_crypt: bool,
    sanitize_nonstandard: bool = False,
):
    """Sanitize forbidden LZW/Crypt/non-Table-6 filters in one inline image."""
    filter_obj = inline_image.obj.get("/Filter")
    if filter_obj is None:
        return None, False, False, False

    normalized = _normalize_inline_filter_object(filter_obj)
    if normalized is None:
        return None, False, False, False

    normalized_names, normalized_filter_obj, normalized_changed = normalized
    has_lzw = "/LZWDecode" in normalized_names
    has_crypt = "/Crypt" in normalized_names
    has_nonstandard = sanitize_nonstandard and any(
        n not in _INLINE_IMAGE_ALLOWED_FILTER_NAMES
        for n in normalized_names
        if n not in ("/LZWDecode", "/Crypt")
    )

    should_process = (
        (convert_lzw and has_lzw) or (remove_crypt and has_crypt) or has_nonstandard
    )
    if not should_process and not normalized_changed:
        return None, False, False, False

    decode_parms = inline_image.obj.get("/DecodeParms")
    # Private pikepdf API (tested with pikepdf 8.x–9.x).
    # _data._inline_image_raw_bytes() is the only way to access the raw
    # encoded payload of an inline image without re-parsing the stream.
    # If this breaks after a pikepdf upgrade, check PdfInlineImage internals.
    try:
        raw_payload = inline_image._data._inline_image_raw_bytes()
    except AttributeError as exc:
        raise AttributeError(
            "pikepdf private API _data._inline_image_raw_bytes() is no longer "
            "available; check pikepdf version compatibility"
        ) from exc

    if not should_process:
        # Private pikepdf API (tested with pikepdf 8.x–9.x): _image_object
        # holds the inline image's metadata key/value token tuple.
        replacement_tokens = _replace_inline_filter_tokens(
            inline_image._image_object,
            normalized_filter_obj,
            decode_parms,
        )
        replacement = _create_inline_image_from_parts(
            replacement_tokens,
            raw_payload,
        )
        return replacement, False, False, False

    try:
        decoded = _decode_inline_image_payload(
            raw_payload, normalized_filter_obj, decode_parms
        )
        rewritten_payload = zlib.compress(decoded) + b"\n"
        replacement_filter: Name | Array | None = Name("/FlateDecode")
        replacement_decode_parms = None
        nonstandard_fixed = has_nonstandard
    except Exception as e:
        if remove_crypt and has_crypt and not has_lzw:
            replacement_filter, replacement_decode_parms = (
                _strip_crypt_from_filter_chain(normalized_names, decode_parms)
            )
            rewritten_payload = raw_payload
            nonstandard_fixed = False
            logger.debug(
                "Removed inline-image Crypt filter without payload rewrite: %s",
                e,
            )
        else:
            logger.warning("Failed to sanitize inline image filters: %s", e)
            return None, False, False, False

    # Private pikepdf API: _image_object (see note above)
    replacement_tokens = _replace_inline_filter_tokens(
        inline_image._image_object,
        replacement_filter,
        replacement_decode_parms,
    )
    replacement = _create_inline_image_from_parts(
        replacement_tokens,
        rewritten_payload,
    )
    return (
        replacement,
        has_lzw and convert_lzw,
        has_crypt and remove_crypt,
        nonstandard_fixed,
    )


def _sanitize_inline_images_in_stream(
    stream: Stream,
    *,
    convert_lzw: bool,
    remove_crypt: bool,
    sanitize_nonstandard: bool = False,
) -> tuple[bool, bool, bool]:
    """Sanitize inline-image filters inside one content stream."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unexpected end of stream", category=UserWarning
            )
            instructions = list(parse_content_stream(stream))
    except Exception:
        return False, False, False

    changed = False
    lzw_changed = False
    crypt_changed = False
    nonstandard_changed = False

    for index, (operands, operator) in enumerate(instructions):
        if str(operator) != "INLINE IMAGE" or not operands:
            continue

        replacement, replaced_lzw, replaced_crypt, replaced_nonstandard = (
            _sanitize_inline_image_filters(
                operands[0],
                convert_lzw=convert_lzw,
                remove_crypt=remove_crypt,
                sanitize_nonstandard=sanitize_nonstandard,
            )
        )
        if replacement is None:
            continue

        instructions[index] = ([replacement], operator)
        changed = True
        lzw_changed = lzw_changed or replaced_lzw
        crypt_changed = crypt_changed or replaced_crypt
        nonstandard_changed = nonstandard_changed or replaced_nonstandard

    if changed:
        stream.write(_unparse_content_stream(instructions))

    return lzw_changed, crypt_changed, nonstandard_changed


def _may_contain_inline_images(stream: Stream) -> bool:
    """Check if a stream could contain inline images.

    Only page content streams (no /Subtype, no /Type) and Form XObjects
    (/Subtype /Form) can contain inline images.  All other streams
    (Image, ICC, Metadata, fonts, XRef, etc.) cannot.
    """
    subtype = stream.get("/Subtype")
    if subtype is not None:
        return str(subtype) == "/Form"
    # No /Subtype — could be a page content stream.
    # Skip if /Type is present (e.g. /Metadata, /XRef).
    if stream.get("/Type") is not None:
        return False
    # Skip font streams (have /Length1, /Length2, or /Length3).
    if stream.get("/Length1") is not None:
        return False
    return True


def _has_lzw_filter(stream: Stream) -> bool:
    """Check if a stream uses LZWDecode filter.

    Handles both single filter and filter arrays.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        True if the stream uses LZWDecode, False otherwise.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False

        filter_obj = _resolve_indirect(filter_obj)

        # Single filter case
        if isinstance(filter_obj, Name):
            return _normalize_inline_filter_name(str(filter_obj)) == "/LZWDecode"

        # Filter array case
        if isinstance(filter_obj, Array):
            for f in filter_obj:
                f = _resolve_indirect(f)
                if (
                    isinstance(f, Name)
                    and _normalize_inline_filter_name(str(f)) == "/LZWDecode"
                ):
                    return True

        return False
    except Exception:
        return False


def _convert_lzw_stream(stream: Stream, pdf: Pdf) -> bool:
    """Convert a single LZW-compressed stream to FlateDecode.

    Reads the decompressed data and writes it back; pikepdf automatically
    compresses with FlateDecode on save.

    Args:
        stream: A pikepdf Stream object with LZW compression.
        pdf: The pikepdf Pdf object (unused but kept for consistency).

    Returns:
        True if conversion succeeded, False otherwise.
    """
    try:
        # Read decompressed data (pikepdf handles LZW decompression)
        data = stream.read_bytes()

        # Write back - pikepdf will compress with FlateDecode on save.
        # stream.write() removes /Filter and /DecodeParms implicitly;
        # we delete /DecodeParms explicitly as a defensive measure.
        stream.write(data)
        if stream.get("/DecodeParms") is not None:
            del stream["/DecodeParms"]

        return True
    except Exception as e:
        logger.warning("Failed to convert LZW stream: %s", e)
        return False


def convert_lzw_streams(pdf: Pdf) -> int:
    """Convert all LZW-compressed streams to FlateDecode.

    LZW compression is forbidden in PDF/A. This function iterates over all
    objects in the PDF and converts any LZW-compressed streams to use
    FlateDecode instead.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Number of streams converted.
    """
    converted = 0

    for obj in pdf.objects:
        try:
            obj = _resolve_indirect(obj)

            if not isinstance(obj, Stream):
                continue

            stream_converted = False

            _normalize_stream_filter_names(obj)

            # Check if it's a stream with LZW filter
            if _has_lzw_filter(obj):
                if _convert_lzw_stream(obj, pdf):
                    converted += 1
                    stream_converted = True
                    logger.debug("Converted LZW stream: %s", obj.objgen)

            if _may_contain_inline_images(obj):
                inline_lzw_changed, _, _ = _sanitize_inline_images_in_stream(
                    obj,
                    convert_lzw=True,
                    remove_crypt=False,
                )
                if inline_lzw_changed and not stream_converted:
                    converted += 1
                    logger.debug(
                        "Converted inline-image LZW filter(s) in stream: %s",
                        obj.objgen,
                    )

        except Exception as e:
            logger.debug("Error processing object: %s", e)

    if converted > 0:
        logger.info("%d LZW stream(s) converted to FlateDecode", converted)

    return converted


def _has_crypt_filter(stream: Stream) -> bool:
    """Check if a stream uses the Crypt filter.

    Handles both single filter and filter arrays.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        True if the stream uses Crypt, False otherwise.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False

        filter_obj = _resolve_indirect(filter_obj)

        # Single filter case
        if isinstance(filter_obj, Name):
            return _normalize_inline_filter_name(str(filter_obj)) == "/Crypt"

        # Filter array case
        if isinstance(filter_obj, Array):
            for f in filter_obj:
                f = _resolve_indirect(f)
                if (
                    isinstance(f, Name)
                    and _normalize_inline_filter_name(str(f)) == "/Crypt"
                ):
                    return True

        return False
    except Exception:
        return False


def _remove_crypt_stream(stream: Stream, pdf: Pdf) -> bool:
    """Remove the Crypt filter from a single stream.

    When pikepdf opens an encrypted PDF it decrypts streams transparently,
    so read_bytes() returns decrypted data and write() re-encodes without
    the Crypt filter.

    Args:
        stream: A pikepdf Stream object with a Crypt filter.
        pdf: The pikepdf Pdf object (unused but kept for consistency).

    Returns:
        True if removal succeeded, False otherwise.
    """
    try:
        # Read decompressed/decrypted data
        data = stream.read_bytes()

        # Write back - pikepdf will compress with FlateDecode on save.
        # stream.write() removes /Filter and /DecodeParms implicitly;
        # we delete /DecodeParms explicitly as a defensive measure.
        stream.write(data)
        if stream.get("/DecodeParms") is not None:
            del stream["/DecodeParms"]

        return True
    except Exception as e:
        logger.warning("Failed to remove Crypt filter from stream: %s", e)
        return False


def remove_crypt_streams(pdf: Pdf) -> int:
    """Remove Crypt filters from all streams.

    The /Crypt filter is forbidden in PDF/A (ISO 19005-2, 6.1.8). This
    function iterates over all objects in the PDF and removes the Crypt
    filter from any streams that use it. Since pikepdf transparently
    decrypts data on read, the stream content is preserved.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Number of streams from which the Crypt filter was removed.
    """
    removed = 0

    for obj in pdf.objects:
        try:
            obj = _resolve_indirect(obj)

            if not isinstance(obj, Stream):
                continue

            stream_removed = False

            _normalize_stream_filter_names(obj)

            if _has_crypt_filter(obj):
                if _remove_crypt_stream(obj, pdf):
                    removed += 1
                    stream_removed = True
                    logger.debug("Removed Crypt filter from stream: %s", obj.objgen)

            if _may_contain_inline_images(obj):
                _, inline_crypt_removed, _ = _sanitize_inline_images_in_stream(
                    obj,
                    convert_lzw=False,
                    remove_crypt=True,
                )
                if inline_crypt_removed and not stream_removed:
                    removed += 1
                    logger.debug(
                        "Removed inline-image Crypt filter(s) in stream: %s",
                        obj.objgen,
                    )

        except Exception as e:
            logger.debug("Error processing object: %s", e)

    if removed > 0:
        logger.info("%d Crypt filter(s) removed from streams", removed)

    return removed


_EXTERNAL_STREAM_KEYS = ("/F", "/FFilter", "/FDecodeParms")


def _has_external_stream_keys(stream: Stream) -> list[str]:
    """Return list of forbidden external stream keys present on a stream.

    ISO 19005-2, 6.1.7.1 forbids /F, /FFilter, and /FDecodeParms in stream
    dictionaries (they reference external file data).

    Args:
        stream: A pikepdf Stream object.

    Returns:
        List of forbidden key names found (e.g. ["/F", "/FFilter"]).
    """
    found = []
    for key in _EXTERNAL_STREAM_KEYS:
        try:
            if stream.get(key) is not None:
                found.append(key)
        except Exception:
            pass
    return found


def remove_external_stream_keys(pdf: Pdf) -> int:
    """Remove /F, /FFilter, /FDecodeParms from all stream dictionaries.

    These keys reference external file alternatives for stream data and are
    forbidden in PDF/A (ISO 19005-2, 6.1.7.1). All stream data must be
    self-contained.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Number of streams from which forbidden keys were removed.
    """
    fixed = 0

    for obj in pdf.objects:
        try:
            obj = _resolve_indirect(obj)

            if not isinstance(obj, Stream):
                continue

            found = _has_external_stream_keys(obj)
            if not found:
                continue

            # Warn if /F is present but inline data is empty
            if "/F" in found:
                try:
                    data = obj.read_raw_bytes()
                    if len(data) == 0:
                        logger.warning(
                            "Stream %s has /F (external file) but no inline "
                            "data — removing /F may cause data loss",
                            obj.objgen,
                        )
                except Exception:
                    logger.warning(
                        "Stream %s has /F but inline data could not be read",
                        obj.objgen,
                    )

            for key in found:
                del obj[key]

            fixed += 1
            logger.debug(
                "Removed external stream keys %s from stream: %s",
                found,
                obj.objgen,
            )

        except Exception as e:
            logger.debug("Error processing object: %s", e)

    if fixed > 0:
        logger.info("%d stream(s) had forbidden external keys removed", fixed)

    return fixed


def sanitize_nonstandard_inline_filters(pdf: Pdf) -> int:
    """Re-encode inline images that use filters not in ISO 32000-1, Table 6.

    ISO 19005-2 rule 6.1.10-1 restricts inline image filters to the six
    filters listed in Table 6. Any other filter (e.g. JBIG2Decode, JPXDecode,
    or an unrecognised name) is re-encoded to FlateDecode.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Number of content streams containing modified inline images.
    """
    fixed = 0
    for obj in pdf.objects:
        try:
            obj = _resolve_indirect(obj)
            if not isinstance(obj, Stream):
                continue
            if not _may_contain_inline_images(obj):
                continue
            _, _, nonstandard_changed = _sanitize_inline_images_in_stream(
                obj,
                convert_lzw=False,
                remove_crypt=False,
                sanitize_nonstandard=True,
            )
            if nonstandard_changed:
                fixed += 1
                logger.debug(
                    "Re-encoded non-standard inline-image filter(s) in stream: %s",
                    obj.objgen,
                )
        except Exception as e:
            logger.debug("Error processing object: %s", e)

    if fixed > 0:
        logger.info(
            "%d stream(s) with non-standard inline-image filters re-encoded",
            fixed,
        )
    return fixed


_IMAGE_FILTERS = frozenset(
    {
        "/DCTDecode",
        "/JPXDecode",
        "/JBIG2Decode",
        "/CCITTFaxDecode",
    }
)


def _has_image_filter(stream: Stream) -> bool:
    """Check if a stream uses an image-specific filter.

    Image filters (DCTDecode, JPXDecode, JBIG2Decode, CCITTFaxDecode) use
    lossy or specialised encoding that must not be re-encoded.

    Handles both single filter and filter arrays.

    Args:
        stream: A pikepdf Stream object.

    Returns:
        True if the stream uses an image filter, False otherwise.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False

        filter_obj = _resolve_indirect(filter_obj)

        if isinstance(filter_obj, Name):
            return _normalize_inline_filter_name(str(filter_obj)) in _IMAGE_FILTERS

        if isinstance(filter_obj, Array):
            for f in filter_obj:
                f = _resolve_indirect(f)
                if (
                    isinstance(f, Name)
                    and _normalize_inline_filter_name(str(f)) in _IMAGE_FILTERS
                ):
                    return True

        return False
    except Exception:
        return False


def fix_stream_lengths(pdf: Pdf) -> int:
    """Re-encode non-image streams to fix /Length mismatches.

    QPDF's copy-optimization copies unchanged streams byte-for-byte from
    the source file, including incorrect /Length values.  Reading and
    writing the data back marks each stream as dirty so that QPDF
    recalculates the correct /Length on save.

    Image streams (DCTDecode, JPXDecode, JBIG2Decode, CCITTFaxDecode) are
    skipped because they use lossy or specialised encoding.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Number of streams re-encoded.
    """
    reencoded = 0

    for obj in pdf.objects:
        try:
            obj = _resolve_indirect(obj)

            if not isinstance(obj, Stream):
                continue

            if _has_image_filter(obj):
                continue

            data = obj.read_bytes()
            obj.write(data)
            reencoded += 1

        except Exception as e:
            logger.debug("Error re-encoding stream: %s", e)

    if reencoded > 0:
        logger.info("%d stream(s) re-encoded to fix /Length", reencoded)

    return reencoded
