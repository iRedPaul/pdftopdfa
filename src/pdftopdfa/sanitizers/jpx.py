# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""JPEG2000 (JPXDecode) sanitizer for PDF/A compliance.

ISO 19005-2, Clause 6.1.4.3 requires JPEG2000 images to have exactly one
colour specification box (colr) in the JP2 header with either METH=1
(enumerated colourspace) or METH=2 (restricted ICC profile).

ISO 19005-2, Clause 6.2.8.3 additionally constrains JPEG2000 channel counts
and per-channel bit depths.

This module detects JPXDecode streams and fixes non-compliant JP2 metadata:
- JP2 files: Parse/fix colr boxes in jp2h header
- JP2 files: Repair ihdr channel count/bit depth from codestream SIZ
- Bare codestreams: Wrap in minimal JP2 container with correct colr box
- Fallback: Re-encode to FlateDecode (lossless pixel-level)
"""

import logging
import struct

from pikepdf import Array, Name, Pdf, Stream

from ..color_profile import get_cmyk_profile
from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

# JP2 file signature: 12 bytes (length=12, type='jP  ', content=0x0D0A870A)
_JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"

# JPEG 2000 codestream SOC marker
_SOC_MARKER = b"\xff\x4f"

# Enumerated colour space values
_ENUM_CS_SRGB = 16
_ENUM_CS_GREYSCALE = 17
_ENUM_CS_SYCC = 18


def _has_jpx_filter(stream: Stream) -> bool:
    """Check if a stream uses JPXDecode filter.

    Handles both single filter and filter arrays.
    """
    try:
        filter_obj = stream.get("/Filter")
        if filter_obj is None:
            return False

        filter_obj = _resolve_indirect(filter_obj)

        if isinstance(filter_obj, Name):
            return str(filter_obj) == "/JPXDecode"

        if isinstance(filter_obj, Array):
            for f in filter_obj:
                f = _resolve_indirect(f)
                if isinstance(f, Name) and str(f) == "/JPXDecode":
                    return True

        return False
    except Exception:
        return False


def _get_num_components(stream: Stream) -> int | None:
    """Derive number of colour components from PDF /ColorSpace.

    Returns:
        1 for DeviceGray, 3 for DeviceRGB, 4 for DeviceCMYK,
        N for ICCBased, or None if undetermined.
    """
    try:
        cs = stream.get("/ColorSpace")
        if cs is None:
            return None

        cs = _resolve_indirect(cs)

        if isinstance(cs, Name):
            cs_name = str(cs)
            if cs_name == "/DeviceGray":
                return 1
            if cs_name == "/DeviceRGB":
                return 3
            if cs_name == "/DeviceCMYK":
                return 4
            return None

        if isinstance(cs, Array) and len(cs) >= 2:
            cs_type = _resolve_indirect(cs[0])
            if isinstance(cs_type, Name) and str(cs_type) == "/ICCBased":
                icc_stream = _resolve_indirect(cs[1])
                if isinstance(icc_stream, Stream):
                    n = icc_stream.get("/N")
                    if n is not None:
                        return int(n)
            return None

        return None
    except Exception:
        return None


# --- JP2 binary helpers ---


def _iter_boxes(data: bytes, start: int, end: int):
    """Iterate JP2 boxes within a byte range.

    Yields (box_type, content_start, content_end, box_end) tuples.
    box_type is a 4-byte bytes value (e.g. b'jp2h').
    """
    pos = start
    while pos < end:
        if pos + 8 > end:
            break
        lbox, tbox = struct.unpack_from(">I4s", data, pos)
        if lbox == 1:
            # Extended box length
            if pos + 16 > end:
                break
            xl = struct.unpack_from(">Q", data, pos + 8)[0]
            content_start = pos + 16
            box_end = pos + xl
        elif lbox == 0:
            # Box extends to end of data
            content_start = pos + 8
            box_end = end
        else:
            content_start = pos + 8
            box_end = pos + lbox

        if box_end > end:
            box_end = end

        yield tbox, content_start, box_end, box_end
        pos = box_end


def _parse_colr_box(data: bytes, content_start: int, content_end: int) -> dict:
    """Parse a colr box and return its properties.

    Returns dict with keys: meth, prec, approx, enum_cs (if METH=1),
    icc_data (if METH=2).
    """
    length = content_end - content_start
    if length < 3:
        return {"meth": 0}

    meth = data[content_start]
    prec = data[content_start + 1]
    approx = data[content_start + 2]

    result = {"meth": meth, "prec": prec, "approx": approx}

    if meth == 1:
        if length >= 7:
            result["enum_cs"] = struct.unpack_from(">I", data, content_start + 3)[0]
    elif meth == 2:
        if length > 3:
            result["icc_data"] = data[content_start + 3 : content_end]

    return result


def _is_valid_colr_box(colr: dict) -> bool:
    """Check if a parsed colr box is PDF/A compliant.

    Valid: METH=1 with EnumCS in {16,17,18} or METH=2 with ICC data.
    """
    meth = colr.get("meth", 0)
    if meth == 1:
        enum_cs = colr.get("enum_cs")
        return enum_cs in (_ENUM_CS_SRGB, _ENUM_CS_GREYSCALE, _ENUM_CS_SYCC)
    if meth == 2:
        icc_data = colr.get("icc_data")
        return icc_data is not None and len(icc_data) > 0
    return False


def _parse_siz_marker(codestream: bytes) -> dict | None:
    """Parse the SIZ marker from a bare JPEG 2000 codestream.

    Returns dict with width, height, num_components, bpc,
    component_bit_depths or None.
    The SIZ marker immediately follows the SOC marker (0xFF4F).
    """
    if len(codestream) < 4:
        return None
    if codestream[0:2] != _SOC_MARKER:
        return None

    # SIZ marker should be at offset 2
    if codestream[2:4] != b"\xff\x51":
        return None

    # SIZ marker segment: Lsiz(2) + Rsiz(2) + Xsiz(4) + Ysiz(4) +
    # XOsiz(4) + YOsiz(4) + XTsiz(4) + YTsiz(4) + XTOsiz(4) + YTOsiz(4) +
    # Csiz(2) = 38 bytes minimum
    if len(codestream) < 4 + 38:
        return None

    lsiz = struct.unpack_from(">H", codestream, 4)[0]
    if lsiz < 38:
        return None
    if len(codestream) < 4 + lsiz:
        return None

    # Skip Rsiz (2 bytes at offset 6)
    xsiz = struct.unpack_from(">I", codestream, 8)[0]
    ysiz = struct.unpack_from(">I", codestream, 12)[0]
    xosiz = struct.unpack_from(">I", codestream, 16)[0]
    yosiz = struct.unpack_from(">I", codestream, 20)[0]
    # Skip tile sizes
    csiz = struct.unpack_from(">H", codestream, 40)[0]

    width = xsiz - xosiz
    height = ysiz - yosiz

    component_bit_depths: list[int] = []
    component_data_start = 42
    component_data_end = 4 + lsiz
    for index in range(csiz):
        ssiz_pos = component_data_start + index * 3
        if ssiz_pos + 2 >= component_data_end:
            return None
        ssiz = codestream[ssiz_pos]
        # Ssiz: bit 7 = signed, bits 0-6 = depth - 1
        component_bit_depths.append((ssiz & 0x7F) + 1)

    if not component_bit_depths:
        return None

    return {
        "width": width,
        "height": height,
        "num_components": csiz,
        "bpc": component_bit_depths[0],
        "component_bit_depths": component_bit_depths,
    }


def _parse_ihdr_box(data: bytes, content_start: int, content_end: int) -> dict | None:
    """Parse an ihdr (Image Header) box.

    Returns dict with height, width, num_components, bpc or None.
    ihdr content: Height(4) + Width(4) + NC(2) + BPC(1) + C(1) + UnkC(1) + IPR(1) = 14
    """
    length = content_end - content_start
    if length < 14:
        return None

    height, width = struct.unpack_from(">II", data, content_start)
    nc = struct.unpack_from(">H", data, content_start + 8)[0]
    bpc = data[content_start + 10]
    # bpc in ihdr: value + 1 gives actual depth (0xFF means variable)
    if bpc == 0xFF:
        actual_bpc = 8  # default fallback
    else:
        actual_bpc = bpc + 1

    return {
        "width": width,
        "height": height,
        "num_components": nc,
        "bpc": actual_bpc,
    }


def _is_valid_jpx_channels(num_components: int) -> bool:
    """Return True when JPX channel count satisfies PDF/A 6.2.8.3."""
    return num_components in (1, 3, 4)


def _is_valid_jpx_bit_depth(bpc: int) -> bool:
    """Return True when JPX bit depth satisfies PDF/A 6.2.8.3."""
    return 1 <= bpc <= 38


def _siz_bit_depths_uniform(siz_info: dict) -> bool:
    """Return True when all SIZ component depths are identical and valid."""
    depths = siz_info.get("component_bit_depths", [])
    if not depths:
        return False
    if not all(_is_valid_jpx_bit_depth(depth) for depth in depths):
        return False
    return len(set(depths)) == 1


def _build_box(box_type: bytes, content: bytes) -> bytes:
    """Construct a JP2 box from type and content."""
    length = 8 + len(content)
    return struct.pack(">I", length) + box_type + content


def _build_colr_box_enum(enum_cs: int) -> bytes:
    """Build a METH=1 colr box with the given EnumCS value."""
    # colr content: METH(1) + PREC(1) + APPROX(1) + EnumCS(4)
    content = struct.pack(">BBBI", 1, 0, 0, enum_cs)
    return _build_box(b"colr", content)


def _build_colr_box_icc(icc_data: bytes) -> bytes:
    """Build a METH=2 colr box with the given ICC profile data."""
    # colr content: METH(1) + PREC(1) + APPROX(1) + ICC_data
    content = struct.pack(">BBB", 2, 0, 0) + icc_data
    return _build_box(b"colr", content)


def _colr_box_for_components(num_components: int) -> bytes:
    """Build an appropriate colr box based on number of colour components."""
    if num_components == 1:
        return _build_colr_box_enum(_ENUM_CS_GREYSCALE)
    if num_components == 3:
        return _build_colr_box_enum(_ENUM_CS_SRGB)
    if num_components == 4:
        return _build_colr_box_icc(get_cmyk_profile())
    # Default to sRGB for unknown
    return _build_colr_box_enum(_ENUM_CS_SRGB)


def _build_jp2_wrapper(
    codestream: bytes,
    width: int,
    height: int,
    num_components: int,
    bpc: int,
    colr_box: bytes,
) -> bytes:
    """Build a complete JP2 file wrapping a bare codestream.

    Structure: Signature + File Type + JP2 Header (ihdr + colr) + Codestream
    """
    # JP2 Signature box
    sig_box = _JP2_SIGNATURE + b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "

    # ihdr box content: Height(4) + Width(4) + NC(2) + BPC(1) + C(1) + UnkC(1) + IPR(1)
    # BPC stored as value - 1 in ihdr
    ihdr_content = struct.pack(
        ">IIHBBBB",
        height,
        width,
        num_components,
        bpc - 1,  # BPC - 1 as per spec
        7,  # C = 7 (JP2 compression)
        0,  # UnkC = 0 (colourspace known)
        0,  # IPR = 0 (no intellectual property)
    )
    ihdr_box = _build_box(b"ihdr", ihdr_content)

    # jp2h superbox containing ihdr + colr
    jp2h_content = ihdr_box + colr_box
    jp2h_box = _build_box(b"jp2h", jp2h_content)

    # jp2c (contiguous codestream) box
    jp2c_box = _build_box(b"jp2c", codestream)

    return sig_box + jp2h_box + jp2c_box


def _build_ihdr_box(width: int, height: int, num_components: int, bpc: int) -> bytes:
    """Build an ihdr box."""
    if not _is_valid_jpx_bit_depth(bpc):
        raise ValueError("Invalid JPX bit depth in ihdr")

    ihdr_content = struct.pack(
        ">IIHBBBB",
        height,
        width,
        num_components,
        bpc - 1,
        7,
        0,
        0,
    )
    return _build_box(b"ihdr", ihdr_content)


def _fix_jp2_colr_boxes(data: bytes, num_components: int | None) -> bytes | None:
    """Fix colr boxes in a JP2 file's jp2h header.

    Returns fixed JP2 bytes if changes were made, None if already valid.
    Raises ValueError if the JP2 structure cannot be parsed.
    """
    if len(data) < len(_JP2_SIGNATURE):
        raise ValueError("Data too short for JP2")

    # Find jp2h superbox and parse top-level jp2c codestream metadata.
    jp2h_start = None
    jp2h_cstart = None
    jp2h_cend = None
    jp2h_end = None
    siz_info = None

    for box_type, cs, ce, be in _iter_boxes(data, 0, len(data)):
        if box_type == b"jp2h":
            jp2h_start = cs - 8  # include box header
            jp2h_cstart = cs
            jp2h_cend = ce
            jp2h_end = be
        elif box_type == b"jp2c" and siz_info is None:
            siz_info = _parse_siz_marker(data[cs:ce])

    if jp2h_start is None:
        raise ValueError("No jp2h box found")

    # Parse sub-boxes within jp2h
    ihdr_info = None
    colr_boxes = []
    other_boxes = []  # raw box bytes (excluding ihdr/colr/bpcc)
    bpcc_raw = None
    bpcc_depths: list[int] | None = None

    for box_type, cs, ce, be in _iter_boxes(data, jp2h_cstart, jp2h_cend):
        raw = data[cs - 8 : be]  # full box including header
        if box_type == b"ihdr":
            ihdr_info = _parse_ihdr_box(data, cs, ce)
        elif box_type == b"colr":
            colr = _parse_colr_box(data, cs, ce)
            colr_boxes.append((colr, raw))
        elif box_type == b"bpcc":
            bpcc_raw = raw
            bpcc_depths = [(b & 0x7F) + 1 for b in data[cs:ce]]
        else:
            other_boxes.append(raw)

    if ihdr_info is None:
        raise ValueError("No ihdr box found")

    target_nc = ihdr_info["num_components"]
    target_bpc = ihdr_info["bpc"]

    # Prefer codestream SIZ for channel count/bit depth when available.
    if siz_info is not None:
        siz_nc = siz_info["num_components"]
        if not _is_valid_jpx_channels(siz_nc):
            raise ValueError("Invalid JPX channel count in codestream")
        if not _siz_bit_depths_uniform(siz_info):
            raise ValueError("Invalid JPX per-channel bit depth in codestream")
        target_nc = siz_nc
        target_bpc = siz_info["bpc"]
    elif num_components is not None:
        target_nc = num_components

    if not _is_valid_jpx_channels(target_nc):
        raise ValueError("Cannot determine valid JPX channel count")
    if not _is_valid_jpx_bit_depth(target_bpc):
        # If no codestream SIZ is available, normalize to a conservative default.
        target_bpc = 8

    ihdr_needs_fix = (
        not _is_valid_jpx_channels(ihdr_info["num_components"])
        or not _is_valid_jpx_bit_depth(ihdr_info["bpc"])
        or ihdr_info["num_components"] != target_nc
        or ihdr_info["bpc"] != target_bpc
    )

    # Validate BPCC box if present (ISO 19005-2, section 6.2.8.3).
    # If bpcc component depths are inconsistent with the target BPC
    # derived from ihdr/codestream, remove the bpcc box.
    bpcc_needs_removal = False
    if bpcc_raw is not None and bpcc_depths is not None:
        if all(d == target_bpc for d in bpcc_depths):
            other_boxes.append(bpcc_raw)  # consistent — keep it
        else:
            bpcc_needs_removal = True

    # Check if already valid: one valid colr box, valid ihdr, valid bpcc.
    if (
        len(colr_boxes) == 1
        and _is_valid_colr_box(colr_boxes[0][0])
        and not ihdr_needs_fix
        and not bpcc_needs_removal
    ):
        return None  # already compliant

    # Determine which colr box to keep
    chosen_colr_raw = None

    # Try to keep the first valid colr box
    for colr, raw in colr_boxes:
        if _is_valid_colr_box(colr):
            chosen_colr_raw = raw
            break

    if chosen_colr_raw is None:
        # No valid colr box found; construct one from image info
        chosen_colr_raw = _colr_box_for_components(target_nc)

    # Reconstruct jp2h: ihdr first, then other boxes, then chosen colr box.
    new_jp2h_content = _build_ihdr_box(
        ihdr_info["width"],
        ihdr_info["height"],
        target_nc,
        target_bpc,
    )
    for raw in other_boxes:
        new_jp2h_content += raw
    new_jp2h_content += chosen_colr_raw

    new_jp2h = _build_box(b"jp2h", new_jp2h_content)

    # Reconstruct full JP2: data before jp2h + new jp2h + data after jp2h
    fixed = data[:jp2h_start] + new_jp2h + data[jp2h_end:]
    return fixed


def _fix_bare_codestream(data: bytes, num_components: int | None) -> bytes | None:
    """Wrap a bare JPEG 2000 codestream in a JP2 container.

    Returns JP2 bytes or None if the codestream cannot be parsed.
    """
    siz = _parse_siz_marker(data)
    if siz is None:
        return None

    colr_box = _colr_box_for_components(siz["num_components"])

    return _build_jp2_wrapper(
        data,
        siz["width"],
        siz["height"],
        siz["num_components"],
        siz["bpc"],
        colr_box,
    )


def _strip_extra_jp2c_boxes(data: bytes) -> bytes | None:
    """Remove extra jp2c (codestream) boxes, keeping only the first.

    ISO 19005-2, section 6.1.4.3 requires exactly one codestream per JP2
    image.  If multiple jp2c boxes exist, all but the first are stripped.

    Args:
        data: Complete JP2 file bytes.

    Returns:
        Fixed JP2 bytes if extra codestreams were removed, None if valid.
    """
    # Collect all top-level box ranges with their types
    box_ranges: list[tuple[int, int, bytes]] = []  # (start, end, type)
    pos = 0
    length = len(data)

    while pos < length:
        if pos + 8 > length:
            box_ranges.append((pos, length, b""))
            break
        lbox, tbox = struct.unpack_from(">I4s", data, pos)
        if lbox == 1:
            if pos + 16 > length:
                box_ranges.append((pos, length, b""))
                break
            xl = struct.unpack_from(">Q", data, pos + 8)[0]
            box_end = pos + xl
        elif lbox == 0:
            box_end = length
        else:
            box_end = pos + lbox

        if box_end > length:
            box_end = length

        box_ranges.append((pos, box_end, tbox))
        pos = box_end

    jp2c_count = sum(1 for _, _, t in box_ranges if t == b"jp2c")
    if jp2c_count <= 1:
        return None

    # Rebuild keeping only the first jp2c box
    result = bytearray()
    first_jp2c_seen = False
    for start, end, tbox in box_ranges:
        if tbox == b"jp2c":
            if first_jp2c_seen:
                continue
            first_jp2c_seen = True
        result.extend(data[start:end])

    return bytes(result)


def _reencode_to_flatedecode(stream: Stream) -> bool:
    """Re-encode a JPXDecode stream to FlateDecode as fallback.

    Decodes JPX image data to raw pixels and writes them back under
    FlateDecode.  Also ensures Width, Height, BitsPerComponent and
    ColorSpace are present in the stream dictionary because JPXDecode
    streams may embed that metadata inside the JPX data itself.

    Returns True on success, False on failure.
    """
    try:
        # Parse image metadata from raw JPX bytes *before* decoding,
        # because JPXDecode streams may omit Width/Height/BPC/ColorSpace
        # from the PDF stream dictionary.
        raw_data = stream.read_raw_bytes()
        image_info = None
        if raw_data[:12] == _JP2_SIGNATURE:
            for box_type, cs, ce, _be in _iter_boxes(raw_data, 0, len(raw_data)):
                if box_type == b"jp2h":
                    for sub_type, scs, sce, _sbe in _iter_boxes(raw_data, cs, ce):
                        if sub_type == b"ihdr":
                            image_info = _parse_ihdr_box(raw_data, scs, sce)
                            break
                    break
        elif raw_data[:2] == _SOC_MARKER:
            image_info = _parse_siz_marker(raw_data)

        # Decode JPX to raw pixels (requires QPDF JPX support)
        decoded_data = stream.read_bytes()

        # Write decoded pixel data — pikepdf applies FlateDecode
        stream.write(decoded_data)

        # Remove /DecodeParms that were specific to JPXDecode
        try:
            del stream["/DecodeParms"]
        except (KeyError, AttributeError):
            pass

        # Ensure required image metadata is in the stream dictionary.
        if image_info is not None:
            if stream.get("/Width") is None:
                stream["/Width"] = image_info["width"]
            if stream.get("/Height") is None:
                stream["/Height"] = image_info["height"]
            if stream.get("/BitsPerComponent") is None:
                stream["/BitsPerComponent"] = image_info["bpc"]
            if stream.get("/ColorSpace") is None:
                nc = image_info["num_components"]
                if nc == 1:
                    stream[Name("/ColorSpace")] = Name("/DeviceGray")
                elif nc == 3:
                    stream[Name("/ColorSpace")] = Name("/DeviceRGB")
                elif nc == 4:
                    stream[Name("/ColorSpace")] = Name("/DeviceCMYK")

        return True
    except Exception as e:
        logger.debug("Failed to re-encode JPX to FlateDecode: %s", e)
        return False


def sanitize_jpx_color_boxes(pdf: Pdf) -> dict[str, int]:
    """Fix JPEG2000 colr boxes for PDF/A compliance.

    Iterates all streams with JPXDecode filter and ensures each has
    exactly one valid colr box in a proper JP2 container.

    Args:
        pdf: pikepdf Pdf object (modified in place).

    Returns:
        Dictionary with counts:
        - jpx_fixed: JP2 files with colr boxes repaired
        - jpx_wrapped: Bare codestreams wrapped in JP2
        - jpx_reencoded: Streams re-encoded to FlateDecode
        - jpx_already_valid: Streams that were already compliant
        - jpx_failed: Streams that could not be fixed
    """
    result = {
        "jpx_fixed": 0,
        "jpx_wrapped": 0,
        "jpx_reencoded": 0,
        "jpx_already_valid": 0,
        "jpx_failed": 0,
    }

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

            if not _has_jpx_filter(obj):
                continue

            raw_data = obj.read_raw_bytes()
            num_components = _get_num_components(obj)

            if raw_data[:12] == _JP2_SIGNATURE:
                current_data = raw_data
                modified = False

                # Strip extra jp2c boxes (ISO 19005-2, §6.1.4.3)
                stripped = _strip_extra_jp2c_boxes(current_data)
                if stripped is not None:
                    current_data = stripped
                    modified = True
                    logger.debug("Stripped extra jp2c boxes: %s", obj.objgen)

                # Fix colr boxes, ihdr, and bpcc
                try:
                    fixed = _fix_jp2_colr_boxes(current_data, num_components)
                    if fixed is not None:
                        current_data = fixed
                        modified = True

                    if modified:
                        obj.write(current_data, filter=Name("/JPXDecode"))
                        result["jpx_fixed"] += 1
                        logger.debug("Fixed JPX stream: %s", obj.objgen)
                    else:
                        result["jpx_already_valid"] += 1
                        logger.debug("JPX stream already valid: %s", obj.objgen)
                except ValueError as e:
                    logger.debug(
                        "JP2 fix failed for %s: %s, attempting FlateDecode re-encode",
                        obj.objgen,
                        e,
                    )
                    if _reencode_to_flatedecode(obj):
                        result["jpx_reencoded"] += 1
                        logger.debug("Re-encoded JPX to FlateDecode: %s", obj.objgen)
                    else:
                        result["jpx_failed"] += 1
                        logger.warning("Failed to fix JPX stream: %s", obj.objgen)
            elif raw_data[:2] == _SOC_MARKER:
                # Bare codestream — wrap in JP2
                wrapped = _fix_bare_codestream(raw_data, num_components)
                if wrapped is not None:
                    obj.write(wrapped, filter=Name("/JPXDecode"))
                    result["jpx_wrapped"] += 1
                    logger.debug("Wrapped bare JPX codestream: %s", obj.objgen)
                else:
                    # Codestream parse failed, try FlateDecode
                    if _reencode_to_flatedecode(obj):
                        result["jpx_reencoded"] += 1
                        logger.debug(
                            "Re-encoded bare JPX to FlateDecode: %s",
                            obj.objgen,
                        )
                    else:
                        result["jpx_failed"] += 1
                        logger.warning("Failed to fix bare JPX stream: %s", obj.objgen)
            else:
                # Unknown format — try FlateDecode fallback
                if _reencode_to_flatedecode(obj):
                    result["jpx_reencoded"] += 1
                    logger.debug(
                        "Re-encoded unknown JPX to FlateDecode: %s",
                        obj.objgen,
                    )
                else:
                    result["jpx_failed"] += 1
                    logger.warning("Failed to fix unknown JPX stream: %s", obj.objgen)

        except Exception as e:
            logger.debug("Error processing JPX object: %s", e)

    total_fixed = result["jpx_fixed"] + result["jpx_wrapped"]
    if total_fixed > 0:
        logger.info(
            "%d JPX stream(s) fixed (%d colr repaired, %d wrapped)",
            total_fixed,
            result["jpx_fixed"],
            result["jpx_wrapped"],
        )
    if result["jpx_reencoded"] > 0:
        logger.info(
            "%d JPX stream(s) re-encoded to FlateDecode",
            result["jpx_reencoded"],
        )
    if result["jpx_failed"] > 0:
        logger.warning("%d JPX stream(s) could not be fixed", result["jpx_failed"])

    return result
