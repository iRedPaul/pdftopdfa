# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for JPEG2000 (JPXDecode) colr box sanitizer."""

import struct

import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.sanitizers.jpx import (
    _build_box,
    _build_colr_box_enum,
    _build_colr_box_icc,
    _build_jp2_wrapper,
    _fix_bare_codestream,
    _fix_jp2_colr_boxes,
    _get_num_components,
    _has_jpx_filter,
    _is_valid_colr_box,
    _iter_boxes,
    _parse_colr_box,
    _parse_ihdr_box,
    _parse_siz_marker,
    _reencode_to_flatedecode,
    _strip_extra_jp2c_boxes,
    sanitize_jpx_color_boxes,
)

# --- JP2 constants ---
_JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"
_SOC_MARKER = b"\xff\x4f"


# --- Test helpers ---


def _build_ihdr_box(
    width: int = 100,
    height: int = 100,
    nc: int = 3,
    bpc: int = 8,
) -> bytes:
    """Build an ihdr box."""
    content = struct.pack(
        ">IIHBBBB",
        height,
        width,
        nc,
        bpc - 1,  # BPC - 1 per spec
        7,  # C = JP2 compression
        0,  # UnkC
        0,  # IPR
    )
    return _build_box(b"ihdr", content)


def _build_colr_box(
    meth: int = 1,
    enum_cs: int = 16,
    icc_data: bytes | None = None,
) -> bytes:
    """Build a colr box with given parameters."""
    if meth == 1:
        content = struct.pack(">BBBi", meth, 0, 0, enum_cs)
    elif meth == 2:
        icc = icc_data or b"\x00" * 128
        content = struct.pack(">BBB", meth, 0, 0) + icc
    else:
        content = struct.pack(">BBB", meth, 0, 0)
    return _build_box(b"colr", content)


def _build_minimal_jp2(
    colr_boxes: list[bytes] | None = None,
    width: int = 100,
    height: int = 100,
    nc: int = 3,
    bpc: int = 8,
) -> bytes:
    """Build a minimal JP2 file with given colr boxes.

    If colr_boxes is None, uses a single valid sRGB colr box.
    """
    if colr_boxes is None:
        colr_boxes = [_build_colr_box(meth=1, enum_cs=16)]

    # Signature
    sig = _JP2_SIGNATURE

    # File type box
    ftyp = b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "

    # ihdr
    ihdr = _build_ihdr_box(width, height, nc, bpc)

    # jp2h superbox
    jp2h_content = ihdr + b"".join(colr_boxes)
    jp2h = _build_box(b"jp2h", jp2h_content)

    # Minimal codestream: just SOC + EOC markers
    codestream = b"\xff\x4f\xff\xd9"
    jp2c = _build_box(b"jp2c", codestream)

    return sig + ftyp + jp2h + jp2c


def _build_bare_codestream(
    width: int = 100,
    height: int = 100,
    nc: int = 3,
    bpc: int = 8,
) -> bytes:
    """Build a minimal bare JPEG 2000 codestream with SIZ marker."""
    # SOC marker
    soc = b"\xff\x4f"

    # SIZ marker
    siz_marker = b"\xff\x51"
    # Lsiz = 38 + 3*nc (header + per-component)
    lsiz = 38 + 3 * nc
    rsiz = 0
    xsiz = width
    ysiz = height
    xosiz = 0
    yosiz = 0
    xtsiz = width
    ytsiz = height
    xtosiz = 0
    ytosiz = 0
    csiz = nc

    siz_content = struct.pack(
        ">HHIIIIIIIIH",
        lsiz,
        rsiz,
        xsiz,
        ysiz,
        xosiz,
        yosiz,
        xtsiz,
        ytsiz,
        xtosiz,
        ytosiz,
        csiz,
    )

    # Per-component: Ssiz(1) + XRsiz(1) + YRsiz(1)
    for _ in range(nc):
        siz_content += struct.pack(">BBB", bpc - 1, 1, 1)

    # EOC marker to complete the codestream
    eoc = b"\xff\xd9"

    return soc + siz_marker + siz_content + eoc


def _make_jpx_image(
    pdf: Pdf,
    data: bytes,
    colorspace: Name | Array | None = None,
) -> Stream:
    """Create a JPXDecode image stream in the PDF."""
    stream_dict = Dictionary(
        Type=Name("/XObject"),
        Subtype=Name("/Image"),
        Filter=Name("/JPXDecode"),
        Width=100,
        Height=100,
        BitsPerComponent=8,
    )
    if colorspace is not None:
        stream_dict.ColorSpace = colorspace

    stream = pdf.make_indirect(Stream(pdf, data, stream_dict))
    return stream


# --- Tests ---


class TestHasJpxFilter:
    """Tests for _has_jpx_filter."""

    def test_single_jpx_filter(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(Filter=Name("/JPXDecode")))
        assert _has_jpx_filter(stream) is True

    def test_jpx_in_filter_array(self):
        pdf = new_pdf()
        stream = Stream(
            pdf,
            b"\x00",
            Dictionary(Filter=Array([Name("/FlateDecode"), Name("/JPXDecode")])),
        )
        assert _has_jpx_filter(stream) is True

    def test_no_filter(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00")
        assert _has_jpx_filter(stream) is False

    def test_other_filter(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(Filter=Name("/FlateDecode")))
        assert _has_jpx_filter(stream) is False


class TestGetNumComponents:
    """Tests for _get_num_components."""

    def test_device_gray(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(ColorSpace=Name("/DeviceGray")))
        assert _get_num_components(stream) == 1

    def test_device_rgb(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(ColorSpace=Name("/DeviceRGB")))
        assert _get_num_components(stream) == 3

    def test_device_cmyk(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(ColorSpace=Name("/DeviceCMYK")))
        assert _get_num_components(stream) == 4

    def test_iccbased(self):
        pdf = new_pdf()
        icc_stream = pdf.make_indirect(Stream(pdf, b"\x00" * 128, Dictionary(N=3)))
        stream = Stream(
            pdf,
            b"\x00",
            Dictionary(ColorSpace=Array([Name("/ICCBased"), icc_stream])),
        )
        assert _get_num_components(stream) == 3

    def test_no_colorspace(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00")
        assert _get_num_components(stream) is None

    def test_unknown_colorspace(self):
        pdf = new_pdf()
        stream = Stream(pdf, b"\x00", Dictionary(ColorSpace=Name("/CalRGB")))
        assert _get_num_components(stream) is None


class TestIterBoxes:
    """Tests for _iter_boxes."""

    def test_single_box(self):
        content = b"hello"
        box = _build_box(b"test", content)
        boxes = list(_iter_boxes(box, 0, len(box)))
        assert len(boxes) == 1
        btype, cs, ce, be = boxes[0]
        assert btype == b"test"
        assert box[cs:ce] == content

    def test_multiple_boxes(self):
        box1 = _build_box(b"aaaa", b"one")
        box2 = _build_box(b"bbbb", b"two")
        data = box1 + box2
        boxes = list(_iter_boxes(data, 0, len(data)))
        assert len(boxes) == 2
        assert boxes[0][0] == b"aaaa"
        assert boxes[1][0] == b"bbbb"

    def test_truncated_data(self):
        # Less than 8 bytes — no boxes
        boxes = list(_iter_boxes(b"\x00\x00", 0, 2))
        assert len(boxes) == 0


class TestParseColrBox:
    """Tests for _parse_colr_box."""

    def test_meth1_srgb(self):
        content = struct.pack(">BBBi", 1, 0, 0, 16)
        box = _build_box(b"colr", content)
        # Parse from content area (skip 8-byte header)
        result = _parse_colr_box(box, 8, len(box))
        assert result["meth"] == 1
        assert result["enum_cs"] == 16

    def test_meth1_greyscale(self):
        content = struct.pack(">BBBi", 1, 0, 0, 17)
        box = _build_box(b"colr", content)
        result = _parse_colr_box(box, 8, len(box))
        assert result["meth"] == 1
        assert result["enum_cs"] == 17

    def test_meth2_icc(self):
        icc_data = b"\x00" * 64
        content = struct.pack(">BBB", 2, 0, 0) + icc_data
        box = _build_box(b"colr", content)
        result = _parse_colr_box(box, 8, len(box))
        assert result["meth"] == 2
        assert result["icc_data"] == icc_data

    def test_invalid_meth(self):
        content = struct.pack(">BBB", 5, 0, 0)
        box = _build_box(b"colr", content)
        result = _parse_colr_box(box, 8, len(box))
        assert result["meth"] == 5

    def test_truncated(self):
        # Less than 3 bytes of content
        result = _parse_colr_box(b"\x00\x00", 0, 2)
        assert result["meth"] == 0


class TestIsValidColrBox:
    """Tests for _is_valid_colr_box."""

    def test_valid_srgb(self):
        assert _is_valid_colr_box({"meth": 1, "enum_cs": 16}) is True

    def test_valid_greyscale(self):
        assert _is_valid_colr_box({"meth": 1, "enum_cs": 17}) is True

    def test_valid_sycc(self):
        assert _is_valid_colr_box({"meth": 1, "enum_cs": 18}) is True

    def test_valid_icc(self):
        assert _is_valid_colr_box({"meth": 2, "icc_data": b"\x00" * 64}) is True

    def test_invalid_meth3(self):
        assert _is_valid_colr_box({"meth": 3}) is False

    def test_meth1_invalid_enum(self):
        assert _is_valid_colr_box({"meth": 1, "enum_cs": 99}) is False

    def test_meth1_no_enum(self):
        assert _is_valid_colr_box({"meth": 1}) is False

    def test_meth2_no_icc(self):
        assert _is_valid_colr_box({"meth": 2}) is False

    def test_meth2_empty_icc(self):
        assert _is_valid_colr_box({"meth": 2, "icc_data": b""}) is False


class TestParseSizMarker:
    """Tests for _parse_siz_marker."""

    def test_valid_codestream(self):
        cs = _build_bare_codestream(200, 150, 3, 8)
        result = _parse_siz_marker(cs)
        assert result is not None
        assert result["width"] == 200
        assert result["height"] == 150
        assert result["num_components"] == 3
        assert result["bpc"] == 8

    def test_single_component(self):
        cs = _build_bare_codestream(50, 50, 1, 16)
        result = _parse_siz_marker(cs)
        assert result is not None
        assert result["num_components"] == 1
        assert result["bpc"] == 16

    def test_truncated(self):
        result = _parse_siz_marker(b"\xff\x4f\xff\x51")
        assert result is None

    def test_not_codestream(self):
        result = _parse_siz_marker(b"\x00\x00\x00\x00")
        assert result is None

    def test_wrong_marker(self):
        # SOC followed by non-SIZ marker
        result = _parse_siz_marker(b"\xff\x4f\xff\x52" + b"\x00" * 100)
        assert result is None


class TestParseIhdrBox:
    """Tests for _parse_ihdr_box."""

    def test_valid(self):
        ihdr = _build_ihdr_box(200, 150, 3, 8)
        result = _parse_ihdr_box(ihdr, 8, len(ihdr))
        assert result is not None
        assert result["width"] == 200
        assert result["height"] == 150
        assert result["num_components"] == 3
        assert result["bpc"] == 8

    def test_truncated(self):
        result = _parse_ihdr_box(b"\x00" * 10, 0, 10)
        assert result is None


class TestBuildHelpers:
    """Tests for box building functions."""

    def test_build_box_roundtrip(self):
        content = b"test content"
        box = _build_box(b"test", content)
        boxes = list(_iter_boxes(box, 0, len(box)))
        assert len(boxes) == 1
        btype, cs, ce, _ = boxes[0]
        assert btype == b"test"
        assert box[cs:ce] == content

    def test_build_colr_box_enum_srgb(self):
        box = _build_colr_box_enum(16)
        # Parse it back
        colr = _parse_colr_box(box, 8, len(box))
        assert colr["meth"] == 1
        assert colr["enum_cs"] == 16
        assert _is_valid_colr_box(colr) is True

    def test_build_colr_box_enum_grey(self):
        box = _build_colr_box_enum(17)
        colr = _parse_colr_box(box, 8, len(box))
        assert colr["meth"] == 1
        assert colr["enum_cs"] == 17

    def test_build_colr_box_icc(self):
        icc = b"\x00" * 128
        box = _build_colr_box_icc(icc)
        colr = _parse_colr_box(box, 8, len(box))
        assert colr["meth"] == 2
        assert colr["icc_data"] == icc

    def test_build_jp2_wrapper(self):
        codestream = b"\xff\x4f\xff\xd9"
        colr = _build_colr_box_enum(16)
        jp2 = _build_jp2_wrapper(codestream, 100, 100, 3, 8, colr)
        # Should start with JP2 signature
        assert jp2[:12] == _JP2_SIGNATURE
        # Should contain jp2h and jp2c boxes
        box_types = [bt for bt, _, _, _ in _iter_boxes(jp2, 0, len(jp2))]
        # Signature box is parsed, then ftyp, jp2h, jp2c
        # Actually the signature is 12 bytes followed by ftyp (20 bytes)
        # which are handled as two separate constructions
        assert b"jp2h" in box_types
        assert b"jp2c" in box_types


class TestFixJP2ColrBoxes:
    """Tests for _fix_jp2_colr_boxes."""

    def test_already_valid_returns_none(self):
        jp2 = _build_minimal_jp2(colr_boxes=[_build_colr_box(meth=1, enum_cs=16)])
        result = _fix_jp2_colr_boxes(jp2, 3)
        assert result is None

    def test_multiple_colr_keeps_first_valid(self):
        colr1 = _build_colr_box(meth=1, enum_cs=16)
        colr2 = _build_colr_box(meth=1, enum_cs=17)
        jp2 = _build_minimal_jp2(colr_boxes=[colr1, colr2])
        result = _fix_jp2_colr_boxes(jp2, 3)
        assert result is not None
        # Should have exactly one colr box now
        colr_count = 0
        for btype, _, _, _ in _iter_boxes(result, 0, len(result)):
            if btype == b"jp2h":
                # Find jp2h and count colr boxes inside
                for btype2, cs2, ce2, _ in _iter_boxes(
                    result,
                    _find_jp2h_content_start(result),
                    _find_jp2h_content_end(result),
                ):
                    if btype2 == b"colr":
                        colr_count += 1
        assert colr_count == 1

    def test_meth_gt2_removed(self):
        colr_bad = _build_colr_box(meth=5, enum_cs=0)
        jp2 = _build_minimal_jp2(colr_boxes=[colr_bad], nc=3)
        result = _fix_jp2_colr_boxes(jp2, 3)
        assert result is not None
        # Verify the colr box in result is valid
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, cs, ce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"colr":
                colr = _parse_colr_box(result, cs, ce)
                assert _is_valid_colr_box(colr)

    def test_invalid_ihdr_channel_count_fixed_from_codestream(self):
        """Invalid ihdr channel count is rewritten from codestream SIZ."""
        jp2 = _build_minimal_jp2(colr_boxes=[_build_colr_box(meth=1, enum_cs=16)])
        jp2_bad = _patch_ihdr_fields(jp2, num_components=5)
        result = _fix_jp2_colr_boxes(jp2_bad, 3)
        assert result is not None
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, cs, ce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"ihdr":
                ihdr = _parse_ihdr_box(result, cs, ce)
                assert ihdr is not None
                assert ihdr["num_components"] == 3

    def test_invalid_ihdr_bit_depth_fixed_from_codestream(self):
        """Invalid ihdr bit depth is rewritten from codestream SIZ."""
        jp2 = _build_minimal_jp2(colr_boxes=[_build_colr_box(meth=1, enum_cs=16)])
        jp2_bad = _patch_ihdr_fields(jp2, bpc=41)
        result = _fix_jp2_colr_boxes(jp2_bad, 3)
        assert result is not None
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, cs, ce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"ihdr":
                ihdr = _parse_ihdr_box(result, cs, ce)
                assert ihdr is not None
                assert ihdr["bpc"] == 8

    def test_missing_colr_constructs_from_ihdr(self):
        # JP2 with no colr boxes at all
        jp2 = _build_minimal_jp2(colr_boxes=[], nc=3)
        result = _fix_jp2_colr_boxes(jp2, None)
        assert result is not None
        # Should have a colr box now
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        found_colr = False
        for btype, cs, ce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"colr":
                colr = _parse_colr_box(result, cs, ce)
                assert _is_valid_colr_box(colr)
                found_colr = True
        assert found_colr

    def test_no_jp2h_raises(self):
        # Craft data that starts with JP2 signature but has no jp2h
        data = _JP2_SIGNATURE + b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "
        with pytest.raises(ValueError, match="No jp2h box found"):
            _fix_jp2_colr_boxes(data, 3)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _fix_jp2_colr_boxes(b"\x00\x00", 3)


class TestFixBareCodestream:
    """Tests for _fix_bare_codestream."""

    def test_3_component_srgb(self):
        cs = _build_bare_codestream(100, 100, 3, 8)
        result = _fix_bare_codestream(cs, 3)
        assert result is not None
        # Should be a valid JP2 file
        assert result[:12] == _JP2_SIGNATURE
        # Verify colr box is sRGB
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, ccs, cce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"colr":
                colr = _parse_colr_box(result, ccs, cce)
                assert colr["meth"] == 1
                assert colr["enum_cs"] == 16

    def test_1_component_grey(self):
        cs = _build_bare_codestream(50, 50, 1, 8)
        result = _fix_bare_codestream(cs, 1)
        assert result is not None
        assert result[:12] == _JP2_SIGNATURE
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, ccs, cce, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            if btype == b"colr":
                colr = _parse_colr_box(result, ccs, cce)
                assert colr["meth"] == 1
                assert colr["enum_cs"] == 17

    def test_num_components_from_siz(self):
        # Don't pass num_components, let it derive from SIZ
        cs = _build_bare_codestream(100, 100, 3, 8)
        result = _fix_bare_codestream(cs, None)
        assert result is not None
        assert result[:12] == _JP2_SIGNATURE

    def test_invalid_codestream_returns_none(self):
        result = _fix_bare_codestream(b"\x00\x00\x00", None)
        assert result is None


class TestReencodeToFlatedecode:
    """Tests for _reencode_to_flatedecode."""

    def test_returns_false_when_jpx_decode_fails(self):
        """read_bytes() cannot decode arbitrary data through JPXDecode."""
        pdf = new_pdf()
        raw_data = b"\xde\xad" * 500
        stream = pdf.make_indirect(Stream(pdf, raw_data))
        stream[Name("/Filter")] = Name("/JPXDecode")
        result = _reencode_to_flatedecode(stream)
        assert result is False

    def test_successful_reencode_unfiltered(self):
        """Unfiltered stream can be re-encoded to FlateDecode."""
        pdf = new_pdf()
        pixel_data = b"\xab\xcd" * 500
        stream = pdf.make_indirect(Stream(pdf, pixel_data))
        result = _reencode_to_flatedecode(stream)
        assert result is True
        decoded = stream.read_bytes()
        assert bytes(decoded) == pixel_data

    def test_decode_parms_removed(self):
        pdf = new_pdf()
        pixel_data = b"\xff" * 300
        stream = pdf.make_indirect(Stream(pdf, pixel_data))
        stream[Name("/DecodeParms")] = Dictionary(ColorTransform=1)
        _reencode_to_flatedecode(stream)
        assert stream.get("/DecodeParms") is None

    def test_populates_missing_metadata_from_jp2(self):
        """Image metadata is extracted from JP2 headers when missing."""
        pdf = new_pdf()
        jp2_data = _build_minimal_jp2(width=50, height=30, nc=3, bpc=8)
        # Create a stream with the JP2 data but no image dict entries.
        # Use no filter so read_bytes() can succeed.
        stream = pdf.make_indirect(Stream(pdf, jp2_data))
        result = _reencode_to_flatedecode(stream)
        assert result is True
        assert int(stream.get("/Width")) == 50
        assert int(stream.get("/Height")) == 30
        assert int(stream.get("/BitsPerComponent")) == 8
        assert str(stream.get("/ColorSpace")) == "/DeviceRGB"

    def test_populates_grayscale_colorspace(self):
        """Single-component JP2 gets DeviceGray colorspace."""
        pdf = new_pdf()
        jp2_data = _build_minimal_jp2(width=10, height=10, nc=1, bpc=8)
        stream = pdf.make_indirect(Stream(pdf, jp2_data))
        result = _reencode_to_flatedecode(stream)
        assert result is True
        assert str(stream.get("/ColorSpace")) == "/DeviceGray"

    def test_populates_cmyk_colorspace(self):
        """Four-component JP2 gets DeviceCMYK colorspace."""
        pdf = new_pdf()
        jp2_data = _build_minimal_jp2(width=10, height=10, nc=4, bpc=8)
        stream = pdf.make_indirect(Stream(pdf, jp2_data))
        result = _reencode_to_flatedecode(stream)
        assert result is True
        assert str(stream.get("/ColorSpace")) == "/DeviceCMYK"

    def test_preserves_existing_metadata(self):
        """Existing Width/Height/BPC/ColorSpace are not overwritten."""
        pdf = new_pdf()
        jp2_data = _build_minimal_jp2(width=50, height=30, nc=3, bpc=8)
        stream = pdf.make_indirect(Stream(pdf, jp2_data))
        stream["/Width"] = 200
        stream["/Height"] = 100
        stream["/BitsPerComponent"] = 16
        stream[Name("/ColorSpace")] = Name("/DeviceGray")
        result = _reencode_to_flatedecode(stream)
        assert result is True
        assert int(stream.get("/Width")) == 200
        assert int(stream.get("/Height")) == 100
        assert int(stream.get("/BitsPerComponent")) == 16
        assert str(stream.get("/ColorSpace")) == "/DeviceGray"

    def test_populates_metadata_from_bare_codestream(self):
        """Image metadata is extracted from bare JPEG2000 codestream."""
        pdf = new_pdf()
        cs_data = _build_bare_codestream(width=64, height=48, nc=3, bpc=8)
        stream = pdf.make_indirect(Stream(pdf, cs_data))
        result = _reencode_to_flatedecode(stream)
        assert result is True
        assert int(stream.get("/Width")) == 64
        assert int(stream.get("/Height")) == 48
        assert int(stream.get("/BitsPerComponent")) == 8
        assert str(stream.get("/ColorSpace")) == "/DeviceRGB"


class TestSanitizeJpxColorBoxes:
    """Integration tests for sanitize_jpx_color_boxes."""

    def test_no_jpx_streams(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_fixed"] == 0
        assert result["jpx_wrapped"] == 0
        assert result["jpx_reencoded"] == 0
        assert result["jpx_already_valid"] == 0
        assert result["jpx_failed"] == 0

    def test_valid_jp2_unchanged(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        jp2_data = _build_minimal_jp2()
        _make_jpx_image(pdf, jp2_data, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_already_valid"] == 1

    def test_fixes_multiple_colr(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        colr1 = _build_colr_box(meth=1, enum_cs=16)
        colr2 = _build_colr_box(meth=1, enum_cs=17)
        jp2_data = _build_minimal_jp2(colr_boxes=[colr1, colr2])
        _make_jpx_image(pdf, jp2_data, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_fixed"] == 1

    def test_wraps_bare_codestream(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        cs = _build_bare_codestream(100, 100, 3, 8)
        _make_jpx_image(pdf, cs, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_wrapped"] == 1

    def test_corrupt_jp2_fails(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        # Start with JP2 signature but no valid jp2h — colr fix fails
        # and QPDF cannot decode JPX, so FlateDecode fallback also fails.
        bad_data = _JP2_SIGNATURE + b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "
        _make_jpx_image(pdf, bad_data, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_failed"] == 1

    def test_handles_non_image_jpx(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        jp2_data = _build_minimal_jp2()
        # Stream without /Subtype /Image — still a JPXDecode stream
        pdf.make_indirect(Stream(pdf, jp2_data, Dictionary(Filter=Name("/JPXDecode"))))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_already_valid"] == 1

    def test_multiple_streams(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        # One valid, one needing fix
        jp2_valid = _build_minimal_jp2()
        colr1 = _build_colr_box(meth=1, enum_cs=16)
        colr2 = _build_colr_box(meth=1, enum_cs=17)
        jp2_multi = _build_minimal_jp2(colr_boxes=[colr1, colr2])
        _make_jpx_image(pdf, jp2_valid, Name("/DeviceRGB"))
        _make_jpx_image(pdf, jp2_multi, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_already_valid"] == 1
        assert result["jpx_fixed"] == 1

    def test_unknown_format_fails(self, make_pdf_with_page):
        pdf = make_pdf_with_page()
        # Random bytes that are neither JP2 nor codestream — QPDF cannot
        # decode JPX, so FlateDecode re-encode fails.
        _make_jpx_image(pdf, b"\xde\xad\xbe\xef" * 10, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_failed"] == 1


# --- Helper to find jp2h range in a JP2 file ---


def _find_jp2h_content_start(data: bytes) -> int:
    """Find the content start of the jp2h box."""
    for btype, cs, _, _ in _iter_boxes(data, 0, len(data)):
        if btype == b"jp2h":
            return cs
    raise ValueError("No jp2h found")


def _find_jp2h_content_end(data: bytes) -> int:
    """Find the content end of the jp2h box."""
    for btype, _, ce, _ in _iter_boxes(data, 0, len(data)):
        if btype == b"jp2h":
            return ce
    raise ValueError("No jp2h found")


def _find_jp2h_range(data: bytes) -> tuple[int, int]:
    """Find the (content_start, content_end) of the jp2h box."""
    for btype, cs, ce, _ in _iter_boxes(data, 0, len(data)):
        if btype == b"jp2h":
            return cs, ce
    raise ValueError("No jp2h found")


def _patch_ihdr_fields(
    data: bytes, *, num_components: int | None = None, bpc: int | None = None
) -> bytes:
    """Patch ihdr fields in a JP2 payload for test-case construction."""
    patched = bytearray(data)
    jp2h_cs, jp2h_ce = _find_jp2h_range(data)
    for btype, cs, _ce, _be in _iter_boxes(data, jp2h_cs, jp2h_ce):
        if btype != b"ihdr":
            continue
        if num_components is not None:
            struct.pack_into(">H", patched, cs + 8, num_components)
        if bpc is not None:
            patched[cs + 10] = bpc - 1
        return bytes(patched)
    raise ValueError("No ihdr found")


def _build_bpcc_box(depths: list[int]) -> bytes:
    """Build a bpcc box with given bit depths.

    Each depth is stored as (depth - 1), unsigned.
    """
    content = bytes(d - 1 for d in depths)
    return _build_box(b"bpcc", content)


def _build_minimal_jp2_with_bpcc(
    colr_boxes: list[bytes] | None = None,
    bpcc_box: bytes | None = None,
    width: int = 100,
    height: int = 100,
    nc: int = 3,
    bpc: int = 8,
) -> bytes:
    """Build a minimal JP2 file with optional bpcc box."""
    if colr_boxes is None:
        colr_boxes = [_build_colr_box(meth=1, enum_cs=16)]

    sig = _JP2_SIGNATURE
    ftyp = b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "
    ihdr = _build_ihdr_box(width, height, nc, bpc)

    jp2h_content = ihdr + b"".join(colr_boxes)
    if bpcc_box is not None:
        jp2h_content += bpcc_box
    jp2h = _build_box(b"jp2h", jp2h_content)

    codestream = b"\xff\x4f\xff\xd9"
    jp2c = _build_box(b"jp2c", codestream)

    return sig + ftyp + jp2h + jp2c


def _build_minimal_jp2_multi_codestream(
    num_codestreams: int = 2,
    width: int = 100,
    height: int = 100,
    nc: int = 3,
    bpc: int = 8,
) -> bytes:
    """Build a JP2 file with multiple jp2c boxes."""
    sig = _JP2_SIGNATURE
    ftyp = b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "
    ihdr = _build_ihdr_box(width, height, nc, bpc)
    colr = _build_colr_box(meth=1, enum_cs=16)
    jp2h = _build_box(b"jp2h", ihdr + colr)

    codestream = b"\xff\x4f\xff\xd9"
    jp2c_boxes = b""
    for _ in range(num_codestreams):
        jp2c_boxes += _build_box(b"jp2c", codestream)

    return sig + ftyp + jp2h + jp2c_boxes


# --- Tests for multi-codestream stripping ---


class TestStripExtraJp2cBoxes:
    """Tests for _strip_extra_jp2c_boxes."""

    def test_single_codestream_returns_none(self):
        """Single jp2c box — already valid."""
        jp2 = _build_minimal_jp2()
        assert _strip_extra_jp2c_boxes(jp2) is None

    def test_no_codestream_returns_none(self):
        """No jp2c box — returns None (nothing to strip)."""
        sig = _JP2_SIGNATURE
        ftyp = b"\x00\x00\x00\x14ftypjp2 \x00\x00\x00\x00jp2 "
        ihdr = _build_ihdr_box()
        colr = _build_colr_box()
        jp2h = _build_box(b"jp2h", ihdr + colr)
        data = sig + ftyp + jp2h
        assert _strip_extra_jp2c_boxes(data) is None

    def test_two_codestreams_strips_second(self):
        """Two jp2c boxes — second is stripped."""
        jp2 = _build_minimal_jp2_multi_codestream(num_codestreams=2)
        result = _strip_extra_jp2c_boxes(jp2)
        assert result is not None
        # Count jp2c boxes in result
        jp2c_count = sum(
            1 for bt, _, _, _ in _iter_boxes(result, 0, len(result)) if bt == b"jp2c"
        )
        assert jp2c_count == 1

    def test_three_codestreams_keeps_only_first(self):
        """Three jp2c boxes — only first is kept."""
        jp2 = _build_minimal_jp2_multi_codestream(num_codestreams=3)
        result = _strip_extra_jp2c_boxes(jp2)
        assert result is not None
        jp2c_count = sum(
            1 for bt, _, _, _ in _iter_boxes(result, 0, len(result)) if bt == b"jp2c"
        )
        assert jp2c_count == 1

    def test_other_boxes_preserved(self):
        """Non-jp2c boxes are preserved when stripping."""
        jp2 = _build_minimal_jp2_multi_codestream(num_codestreams=2)
        result = _strip_extra_jp2c_boxes(jp2)
        assert result is not None
        # jp2h should still be present
        box_types = [bt for bt, _, _, _ in _iter_boxes(result, 0, len(result))]
        assert b"jp2h" in box_types

    def test_stripped_result_starts_with_signature(self):
        """Stripped JP2 still starts with JP2 signature."""
        jp2 = _build_minimal_jp2_multi_codestream(num_codestreams=2)
        result = _strip_extra_jp2c_boxes(jp2)
        assert result is not None
        assert result[:12] == _JP2_SIGNATURE


# --- Tests for BPCC box validation ---


class TestFixJP2ColrBoxesBpcc:
    """Tests for bpcc validation in _fix_jp2_colr_boxes."""

    def test_consistent_bpcc_kept(self):
        """BPCC box consistent with ihdr BPC is kept."""
        bpcc = _build_bpcc_box([8, 8, 8])
        jp2 = _build_minimal_jp2_with_bpcc(bpcc_box=bpcc, nc=3, bpc=8)
        result = _fix_jp2_colr_boxes(jp2, 3)
        # Already valid (colr ok, ihdr ok, bpcc consistent) → None
        assert result is None

    def test_inconsistent_bpcc_removed(self):
        """BPCC box inconsistent with ihdr BPC is removed."""
        bpcc = _build_bpcc_box([8, 16, 8])  # second component differs
        jp2 = _build_minimal_jp2_with_bpcc(bpcc_box=bpcc, nc=3, bpc=8)
        result = _fix_jp2_colr_boxes(jp2, 3)
        assert result is not None
        # Verify no bpcc box in result
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        for btype, _, _, _ in _iter_boxes(result, jp2h_cs, jp2h_ce):
            assert btype != b"bpcc"

    def test_no_bpcc_already_valid(self):
        """No bpcc box — colr/ihdr valid → returns None."""
        jp2 = _build_minimal_jp2()
        assert _fix_jp2_colr_boxes(jp2, 3) is None

    def test_inconsistent_bpcc_with_valid_colr(self):
        """Valid colr + valid ihdr + inconsistent bpcc → fix needed."""
        bpcc = _build_bpcc_box([4, 4, 4])  # all 4-bit, but ihdr says 8
        jp2 = _build_minimal_jp2_with_bpcc(bpcc_box=bpcc, nc=3, bpc=8)
        result = _fix_jp2_colr_boxes(jp2, 3)
        assert result is not None
        # ihdr and colr should still be present
        jp2h_cs, jp2h_ce = _find_jp2h_range(result)
        box_types = [bt for bt, _, _, _ in _iter_boxes(result, jp2h_cs, jp2h_ce)]
        assert b"ihdr" in box_types
        assert b"colr" in box_types
        assert b"bpcc" not in box_types


# --- Integration tests for multi-codestream and bpcc ---


class TestSanitizeJpxMultiCodestream:
    """Integration tests for multi-codestream stripping."""

    def test_multi_codestream_fixed(self, make_pdf_with_page):
        """JP2 with two jp2c boxes is fixed."""
        pdf = make_pdf_with_page()
        jp2 = _build_minimal_jp2_multi_codestream(num_codestreams=2)
        _make_jpx_image(pdf, jp2, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_fixed"] == 1

    def test_single_codestream_already_valid(self, make_pdf_with_page):
        """JP2 with one jp2c box is already valid."""
        pdf = make_pdf_with_page()
        jp2 = _build_minimal_jp2()
        _make_jpx_image(pdf, jp2, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_already_valid"] == 1


class TestSanitizeJpxBpcc:
    """Integration tests for bpcc validation."""

    def test_inconsistent_bpcc_fixed(self, make_pdf_with_page):
        """JP2 with inconsistent bpcc is fixed."""
        pdf = make_pdf_with_page()
        bpcc = _build_bpcc_box([8, 16, 8])
        jp2 = _build_minimal_jp2_with_bpcc(bpcc_box=bpcc, nc=3, bpc=8)
        _make_jpx_image(pdf, jp2, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_fixed"] == 1

    def test_consistent_bpcc_already_valid(self, make_pdf_with_page):
        """JP2 with consistent bpcc is already valid."""
        pdf = make_pdf_with_page()
        bpcc = _build_bpcc_box([8, 8, 8])
        jp2 = _build_minimal_jp2_with_bpcc(bpcc_box=bpcc, nc=3, bpc=8)
        _make_jpx_image(pdf, jp2, Name("/DeviceRGB"))
        result = sanitize_jpx_color_boxes(pdf)
        assert result["jpx_already_valid"] == 1
