# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for JBIG2 external globals inlining and refinement detection."""

import struct
import zlib
from collections.abc import Generator

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.sanitizers.jbig2 import (
    _convert_jbig2_array_stream,
    _convert_jbig2_stream,
    _get_globals_from_array,
    _get_globals_stream,
    _get_jbig2_filter_index,
    _has_jbig2_filter_single,
    _has_refinement_segments,
    _reencode_jbig2_to_flatedecode,
    _strip_preceding_filters,
    convert_jbig2_external_globals,
)


def _make_jbig2_stream(
    pdf: Pdf,
    page_data: bytes,
    globals_data: bytes | None = None,
    *,
    filter_array: bool = False,
) -> Stream:
    """Create a JBIG2 stream with optional external globals.

    Args:
        pdf: The Pdf to own the objects.
        page_data: Raw JBIG2 page segment data.
        globals_data: If provided, creates a globals stream with this data.
        filter_array: If True, wrap the filter in an Array.

    Returns:
        An indirect JBIG2 Stream object.
    """
    jbig2_filter = Name("/JBIG2Decode")

    if filter_array:
        filter_val = Array([jbig2_filter])
    else:
        filter_val = jbig2_filter

    stream_dict = Dictionary(Filter=filter_val)

    if globals_data is not None:
        globals_stream = pdf.make_indirect(Stream(pdf, globals_data))
        if filter_array:
            stream_dict.DecodeParms = Array([Dictionary(JBIG2Globals=globals_stream)])
        else:
            stream_dict.DecodeParms = Dictionary(JBIG2Globals=globals_stream)

    stream = pdf.make_indirect(Stream(pdf, page_data, stream_dict))
    return stream


def _make_jbig2_multifilter_stream(
    pdf: Pdf,
    page_data: bytes,
    globals_data: bytes,
    preceding_filters: list[str] | None = None,
) -> Stream:
    """Create a JBIG2 stream with preceding filters and external globals.

    Args:
        pdf: The Pdf to own the objects.
        page_data: Raw JBIG2 page segment data.
        globals_data: Data for the globals stream.
        preceding_filters: Filter names to prepend (e.g. ["/FlateDecode"]).
            The raw stream data will be encoded through these filters.

    Returns:
        An indirect JBIG2 Stream object.
    """
    if preceding_filters is None:
        preceding_filters = ["/FlateDecode"]

    # Encode page_data through preceding filters in reverse order
    encoded = page_data
    for fname in reversed(preceding_filters):
        if fname == "/FlateDecode":
            encoded = zlib.compress(encoded)
        elif fname == "/ASCIIHexDecode":
            encoded = page_data.hex().encode("ascii") + b">"
        else:
            msg = f"Unsupported test filter: {fname}"
            raise ValueError(msg)

    filter_array = Array([Name(f) for f in preceding_filters] + [Name("/JBIG2Decode")])
    globals_stream = pdf.make_indirect(Stream(pdf, globals_data))

    # Build DecodeParms array: empty dicts for preceding, globals for JBIG2
    parms = Array(
        [Dictionary() for _ in preceding_filters]
        + [Dictionary(JBIG2Globals=globals_stream)]
    )

    stream_dict = Dictionary(Filter=filter_array, DecodeParms=parms)
    return pdf.make_indirect(Stream(pdf, encoded, stream_dict))


class TestGetGlobalsStream:
    """Tests for _get_globals_stream."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_returns_globals_stream(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02\x03"
        stream = _make_jbig2_stream(pdf, b"\x10\x11", globals_data)
        result = _get_globals_stream(stream)
        assert result is not None
        assert result.read_bytes() == globals_data

    def test_returns_none_without_decode_parms(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11")
        result = _get_globals_stream(stream)
        assert result is None

    def test_returns_none_for_array_decode_parms(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11", b"\x00\x01", filter_array=True)
        result = _get_globals_stream(stream)
        assert result is None


class TestHasJbig2FilterSingle:
    """Tests for _has_jbig2_filter_single."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_true_for_single_jbig2(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11")
        assert _has_jbig2_filter_single(stream) is True

    def test_false_for_array_filter(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11", filter_array=True)
        assert _has_jbig2_filter_single(stream) is False

    def test_false_for_other_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x78\x9c", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _has_jbig2_filter_single(stream) is False


class TestConvertJbig2Stream:
    """Tests for _convert_jbig2_stream."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_inlines_globals(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02\x03"
        page_data = b"\x10\x11\x12\x13"
        stream = _make_jbig2_stream(pdf, page_data, globals_data)

        result = _convert_jbig2_stream(stream, pdf)

        assert result is True
        # The raw bytes should be globals + page data
        assert stream.read_raw_bytes() == globals_data + page_data

    def test_filter_remains_jbig2(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", b"\x00")
        _convert_jbig2_stream(stream, pdf)
        assert str(stream.get("/Filter")) == "/JBIG2Decode"

    def test_decode_parms_removed(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", b"\x00")
        # Verify DecodeParms exists before conversion
        assert stream.get("/DecodeParms") is not None
        _convert_jbig2_stream(stream, pdf)
        # stream.write() with filter= removes DecodeParms
        assert stream.get("/DecodeParms") is None

    def test_returns_false_without_globals(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11")
        result = _convert_jbig2_stream(stream, pdf)
        assert result is False


class TestConvertJbig2ExternalGlobals:
    """Tests for convert_jbig2_external_globals."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        yield pdf

    def test_successful_inline(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02"
        page_data = b"\x10\x11\x12"
        _make_jbig2_stream(pdf, page_data, globals_data)

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 1
        assert result["failed"] == 0

    def test_multiple_streams_shared_globals(self, pdf: Pdf) -> None:
        """Multiple JBIG2 streams sharing the same globals stream."""
        globals_data = b"\x00\x01"
        globals_stream = pdf.make_indirect(Stream(pdf, globals_data))

        for page_byte in [b"\x10", b"\x20", b"\x30"]:
            pdf.make_indirect(
                Stream(
                    pdf,
                    page_byte,
                    Dictionary(
                        Filter=Name("/JBIG2Decode"),
                        DecodeParms=Dictionary(JBIG2Globals=globals_stream),
                    ),
                )
            )

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 3
        assert result["failed"] == 0

    def test_single_element_filter_array_converted(self, pdf: Pdf) -> None:
        """[/JBIG2Decode] array is equivalent to single filter."""
        globals_data = b"\x00\x01"
        page_data = b"\x10\x11"
        _make_jbig2_stream(pdf, page_data, globals_data, filter_array=True)

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 1
        assert result["failed"] == 0

    def test_jbig2_without_globals_unchanged(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10\x11")

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 0
        assert result["failed"] == 0
        # Stream should be untouched
        assert stream.read_raw_bytes() == b"\x10\x11"

    def test_mixed_streams(self, pdf: Pdf) -> None:
        """Mix of single filter, single-element array, and no-globals."""
        # Single filter with globals (inlineable)
        _make_jbig2_stream(pdf, b"\x10", b"\x00")
        # Single-element array with globals (also inlineable now)
        _make_jbig2_stream(pdf, b"\x20", b"\x01", filter_array=True)
        # No globals (should be ignored)
        _make_jbig2_stream(pdf, b"\x30")

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 2
        assert result["failed"] == 0

    def test_multifilter_flate_jbig2_converted(self, pdf: Pdf) -> None:
        """[/FlateDecode /JBIG2Decode] with globals is converted."""
        globals_data = b"\x00\x01\x02"
        page_data = b"\x10\x11\x12"
        _make_jbig2_multifilter_stream(pdf, page_data, globals_data)

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 1
        assert result["failed"] == 0

    def test_multifilter_result_has_inlined_globals(self, pdf: Pdf) -> None:
        """Multi-filter conversion inlines globals into JBIG2 bitstream."""
        globals_data = b"\x00\x01\x02"
        page_data = b"\x10\x11\x12"
        stream = _make_jbig2_multifilter_stream(pdf, page_data, globals_data)

        convert_jbig2_external_globals(pdf)

        # After conversion, filter should be single /JBIG2Decode
        assert str(stream.get("/Filter")) == "/JBIG2Decode"
        # Raw bytes should be globals + page data
        assert stream.read_raw_bytes() == globals_data + page_data


class TestGetJbig2FilterIndex:
    """Tests for _get_jbig2_filter_index."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_returns_index_for_single_element_array(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", filter_array=True)
        assert _get_jbig2_filter_index(stream) == 0

    def test_returns_index_for_multifilter(self, pdf: Pdf) -> None:
        stream = _make_jbig2_multifilter_stream(pdf, b"\x10", b"\x00", ["/FlateDecode"])
        assert _get_jbig2_filter_index(stream) == 1

    def test_returns_none_for_single_name_filter(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10")
        assert _get_jbig2_filter_index(stream) is None

    def test_returns_none_for_no_jbig2(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x78\x9c", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _get_jbig2_filter_index(stream) is None


class TestGetGlobalsFromArray:
    """Tests for _get_globals_from_array."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_returns_globals_from_single_element_array(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02"
        stream = _make_jbig2_stream(pdf, b"\x10", globals_data, filter_array=True)
        result = _get_globals_from_array(stream, 0)
        assert result is not None
        assert result.read_bytes() == globals_data

    def test_returns_globals_from_multifilter(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02"
        stream = _make_jbig2_multifilter_stream(
            pdf, b"\x10", globals_data, ["/FlateDecode"]
        )
        result = _get_globals_from_array(stream, 1)
        assert result is not None
        assert result.read_bytes() == globals_data

    def test_returns_none_for_wrong_index(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", b"\x00", filter_array=True)
        assert _get_globals_from_array(stream, 5) is None

    def test_returns_none_without_decode_parms(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", filter_array=True)
        assert _get_globals_from_array(stream, 0) is None


class TestStripPrecedingFilters:
    """Tests for _strip_preceding_filters."""

    def test_strips_flate(self) -> None:
        original = b"\x10\x11\x12\x13"
        compressed = zlib.compress(original)
        result = _strip_preceding_filters(compressed, [Name("/FlateDecode")])
        assert result == original

    def test_strips_asciihex(self) -> None:
        original = b"\x10\x11\x12"
        encoded = original.hex().encode("ascii") + b">"
        result = _strip_preceding_filters(encoded, [Name("/ASCIIHexDecode")])
        assert result == original

    def test_returns_none_for_unsupported_filter(self) -> None:
        result = _strip_preceding_filters(b"\x00", [Name("/LZWDecode")])
        assert result is None

    def test_strips_multiple_filters(self) -> None:
        original = b"\xaa\xbb\xcc"
        # ASCIIHex then Flate: raw data is Flate(ASCIIHex(original))
        hex_encoded = original.hex().encode("ascii") + b">"
        compressed = zlib.compress(hex_encoded)
        result = _strip_preceding_filters(
            compressed,
            [Name("/FlateDecode"), Name("/ASCIIHexDecode")],
        )
        assert result == original

    def test_returns_none_for_invalid_flate(self) -> None:
        result = _strip_preceding_filters(b"\xff\xff\xff", [Name("/FlateDecode")])
        assert result is None

    def test_empty_filters_returns_data(self) -> None:
        data = b"\x10\x11"
        result = _strip_preceding_filters(data, [])
        assert result == data


class TestConvertJbig2ArrayStream:
    """Tests for _convert_jbig2_array_stream."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        yield pdf

    def test_converts_single_element_array(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02"
        page_data = b"\x10\x11\x12"
        stream = _make_jbig2_stream(pdf, page_data, globals_data, filter_array=True)

        result = _convert_jbig2_array_stream(stream, pdf)

        assert result is True
        assert stream.read_raw_bytes() == globals_data + page_data
        assert str(stream.get("/Filter")) == "/JBIG2Decode"

    def test_converts_flate_jbig2_array(self, pdf: Pdf) -> None:
        globals_data = b"\x00\x01\x02"
        page_data = b"\x10\x11\x12"
        stream = _make_jbig2_multifilter_stream(
            pdf, page_data, globals_data, ["/FlateDecode"]
        )

        result = _convert_jbig2_array_stream(stream, pdf)

        assert result is True
        assert stream.read_raw_bytes() == globals_data + page_data
        assert str(stream.get("/Filter")) == "/JBIG2Decode"

    def test_returns_false_for_non_array_filter(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", b"\x00")
        assert _convert_jbig2_array_stream(stream, pdf) is False

    def test_returns_false_without_globals(self, pdf: Pdf) -> None:
        stream = _make_jbig2_stream(pdf, b"\x10", filter_array=True)
        assert _convert_jbig2_array_stream(stream, pdf) is False

    def test_returns_false_for_jbig2_not_last(self, pdf: Pdf) -> None:
        """JBIG2Decode not as last filter is unsupported."""
        stream_dict = Dictionary(
            Filter=Array([Name("/JBIG2Decode"), Name("/FlateDecode")]),
            DecodeParms=Array(
                [
                    Dictionary(JBIG2Globals=pdf.make_indirect(Stream(pdf, b"\x00"))),
                    Dictionary(),
                ]
            ),
        )
        stream = pdf.make_indirect(Stream(pdf, b"\x10", stream_dict))

        assert _convert_jbig2_array_stream(stream, pdf) is False


# --- JBIG2 segment builder helper ---


def _build_jbig2_segment(
    seg_num: int,
    seg_type: int,
    data: bytes = b"",
    *,
    referred_to: list[int] | None = None,
    page_assoc: int = 0,
    page_assoc_large: bool = False,
) -> bytes:
    """Build a JBIG2 segment header + data for testing.

    Creates a properly-formatted segment with short-form referred-to
    count (max 4 referred segments).
    """
    if referred_to is None:
        referred_to = []

    # Segment number (4 bytes)
    header = struct.pack(">I", seg_num)

    # Flags: type in bits 0-5, page_assoc_large in bit 6
    flags = seg_type & 0x3F
    if page_assoc_large:
        flags |= 0x40
    header += struct.pack("B", flags)

    # Referred-to count (short form, count <= 4)
    ref_count = len(referred_to)
    assert ref_count <= 4, "Test helper only supports short-form (<=4 refs)"
    count_byte = (ref_count << 5) & 0xE0
    header += struct.pack("B", count_byte)

    # Referred-to segment numbers
    for ref_seg in referred_to:
        if seg_num <= 256:
            header += struct.pack("B", ref_seg)
        elif seg_num <= 65536:
            header += struct.pack(">H", ref_seg)
        else:
            header += struct.pack(">I", ref_seg)

    # Page association
    if page_assoc_large:
        header += struct.pack(">I", page_assoc)
    else:
        header += struct.pack("B", page_assoc)

    # Data length (4 bytes)
    header += struct.pack(">I", len(data))

    return header + data


class TestHasRefinementSegments:
    """Tests for _has_refinement_segments."""

    def test_no_segments_returns_false(self) -> None:
        assert _has_refinement_segments(b"") is False

    def test_truncated_data_returns_false(self) -> None:
        assert _has_refinement_segments(b"\x00\x01") is False

    def test_normal_segments_returns_false(self) -> None:
        """Page info (48) + end-of-page (49) + end-of-file (51)."""
        data = (
            _build_jbig2_segment(0, 48, b"\x00" * 19, page_assoc=1)
            + _build_jbig2_segment(1, 49, b"", page_assoc=1)
            + _build_jbig2_segment(2, 51, b"")
        )
        assert _has_refinement_segments(data) is False

    def test_type_40_detected(self) -> None:
        """Intermediate generic refinement region (type 40) is forbidden."""
        data = _build_jbig2_segment(
            0, 48, b"\x00" * 19, page_assoc=1
        ) + _build_jbig2_segment(1, 40, b"\xaa\xbb", page_assoc=1)
        assert _has_refinement_segments(data) is True

    def test_type_42_detected(self) -> None:
        """Immediate generic refinement region (type 42) is forbidden."""
        data = _build_jbig2_segment(
            0, 48, b"\x00" * 19, page_assoc=1
        ) + _build_jbig2_segment(1, 42, b"\xcc\xdd", page_assoc=1)
        assert _has_refinement_segments(data) is True

    def test_type_43_detected(self) -> None:
        """Immediate lossless generic refinement region (type 43) is forbidden."""
        data = _build_jbig2_segment(0, 43, b"\xee\xff", page_assoc=1)
        assert _has_refinement_segments(data) is True

    def test_refinement_after_normal_segments(self) -> None:
        """Refinement segment after several normal segments."""
        data = (
            _build_jbig2_segment(0, 48, b"\x00" * 19, page_assoc=1)
            + _build_jbig2_segment(1, 38, b"\x00" * 10, page_assoc=1)
            + _build_jbig2_segment(2, 42, b"\x00" * 5, page_assoc=1)
        )
        assert _has_refinement_segments(data) is True

    def test_end_of_file_stops_scanning(self) -> None:
        """Refinement after end-of-file should not be detected."""
        eof = _build_jbig2_segment(0, 51, b"")
        # Append a refinement segment raw bytes after EOF
        refinement = _build_jbig2_segment(1, 42, b"\x00" * 5, page_assoc=1)
        data = eof + refinement
        assert _has_refinement_segments(data) is False

    def test_segment_with_referred_to(self) -> None:
        """Correctly skips referred-to segment fields."""
        data = (
            _build_jbig2_segment(0, 0, b"\x00" * 20, page_assoc=1)
            + _build_jbig2_segment(1, 6, b"\x00" * 10, referred_to=[0], page_assoc=1)
            + _build_jbig2_segment(2, 51, b"")
        )
        assert _has_refinement_segments(data) is False

    def test_large_page_association(self) -> None:
        """Handles 4-byte page association correctly."""
        data = _build_jbig2_segment(
            0, 48, b"\x00" * 19, page_assoc=1, page_assoc_large=True
        ) + _build_jbig2_segment(1, 51, b"")
        assert _has_refinement_segments(data) is False


class TestReencodeJbig2ToFlatedecode:
    """Tests for _reencode_jbig2_to_flatedecode."""

    def test_returns_false_on_decode_failure(self) -> None:
        """JBIG2 data that QPDF cannot decode."""
        pdf = new_pdf()
        stream = pdf.make_indirect(Stream(pdf, b"\xde\xad" * 100))
        stream[Name("/Filter")] = Name("/JBIG2Decode")
        result = _reencode_jbig2_to_flatedecode(stream)
        assert result is False

    def test_reencodes_unfiltered_stream(self) -> None:
        """Plain stream (no JBIG2 filter) can be re-encoded."""
        pdf = new_pdf()
        pixel_data = b"\x00\xff" * 200
        stream = pdf.make_indirect(Stream(pdf, pixel_data))
        result = _reencode_jbig2_to_flatedecode(stream)
        assert result is True
        assert bytes(stream.read_bytes()) == pixel_data

    def test_removes_decode_parms(self) -> None:
        """DecodeParms are removed after re-encoding."""
        pdf = new_pdf()
        stream = pdf.make_indirect(Stream(pdf, b"\xff" * 100))
        stream[Name("/DecodeParms")] = Dictionary(JBIG2Globals=Name("/Dummy"))
        _reencode_jbig2_to_flatedecode(stream)
        assert stream.get("/DecodeParms") is None


class TestConvertJbig2ExternalGlobalsRefinement:
    """Tests for refinement detection in convert_jbig2_external_globals."""

    @pytest.fixture
    def pdf(self) -> Generator[Pdf]:
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)
        yield pdf

    def test_no_refinement_not_reencoded(self, pdf: Pdf) -> None:
        """JBIG2 without refinement is not re-encoded."""
        normal_data = (
            _build_jbig2_segment(0, 48, b"\x00" * 19, page_assoc=1)
            + _build_jbig2_segment(1, 49, b"", page_assoc=1)
            + _build_jbig2_segment(2, 51, b"")
        )
        pdf.make_indirect(
            Stream(
                pdf,
                normal_data,
                Dictionary(Filter=Name("/JBIG2Decode")),
            )
        )

        result = convert_jbig2_external_globals(pdf)

        assert result["converted"] == 0
        assert result["reencoded"] == 0
        assert result["failed"] == 0

    def test_refinement_reencoded_returns_count(self, pdf: Pdf) -> None:
        """JBIG2 with refinement is counted in reencoded (decode may fail)."""
        refinement_data = _build_jbig2_segment(
            0, 48, b"\x00" * 19, page_assoc=1
        ) + _build_jbig2_segment(1, 42, b"\x00" * 10, page_assoc=1)
        pdf.make_indirect(
            Stream(
                pdf,
                refinement_data,
                Dictionary(Filter=Name("/JBIG2Decode")),
            )
        )

        result = convert_jbig2_external_globals(pdf)

        # Depending on QPDF jbig2dec support, re-encode succeeds or fails.
        # Either way, the result dict should reflect the attempt.
        assert result["reencoded"] + result["failed"] >= 1

    def test_globals_inlined_then_refinement_checked(self, pdf: Pdf) -> None:
        """After globals inlining, refinement check still runs."""
        # Globals contain a normal segment
        globals_data = _build_jbig2_segment(0, 0, b"\x00" * 20, page_assoc=1)
        # Page data contains a refinement segment
        page_data = _build_jbig2_segment(1, 42, b"\x00" * 10, page_assoc=1)

        globals_stream = pdf.make_indirect(Stream(pdf, globals_data))
        pdf.make_indirect(
            Stream(
                pdf,
                page_data,
                Dictionary(
                    Filter=Name("/JBIG2Decode"),
                    DecodeParms=Dictionary(JBIG2Globals=globals_stream),
                ),
            )
        )

        result = convert_jbig2_external_globals(pdf)

        # Globals should be inlined
        assert result["converted"] == 1
        # Refinement should be detected in the combined data
        assert result["reencoded"] + result["failed"] >= 1

    def test_result_dict_has_reencoded_key(self, pdf: Pdf) -> None:
        """Result dict always includes 'reencoded' key."""
        result = convert_jbig2_external_globals(pdf)
        assert "reencoded" in result
