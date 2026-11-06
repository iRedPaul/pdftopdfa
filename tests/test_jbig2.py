# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for preserving JBIG2 images during PDF/A conversion.

The synthetic segment payloads exercise byte preservation and error handling;
they are not decodable images or fixtures for rendering/conformance validation.
"""

import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.sanitizers import sanitize_for_pdfa


def _segment(number: int, segment_type: int, data: bytes, page: int = 1) -> bytes:
    """Build a short JBIG2 segment header without referred-to segments."""
    return struct.pack(">IBBBI", number, segment_type, 0, page, len(data)) + data


def _page_data(segment_type: int) -> bytes:
    page_info = struct.pack(">IIII", 8, 8, 0, 0) + b"\x00\x00\x00"
    return _segment(1, 48, page_info) + _segment(2, segment_type, b"\x00" * 20)


_GLOBALS_DATA = _segment(0, 0, b"\x00" * 20, page=0)


def _add_image(
    pdf: Pdf,
    data: bytes,
    filter_layout: str,
    globals_stream: Stream | None = None,
) -> Stream:
    """Attach an image to a page so it survives saving and reopening."""
    image = pdf.make_stream(data)
    image.Type = Name.XObject
    image.Subtype = Name.Image
    image.Width = 8
    image.Height = 8
    image.BitsPerComponent = 1
    image.ColorSpace = Name.DeviceGray
    if filter_layout == "scalar":
        image.Filter = Name.JBIG2Decode
        if globals_stream is not None:
            image.DecodeParms = Dictionary(JBIG2Globals=globals_stream)
    else:
        filters = [Name.JBIG2Decode]
        parms = (
            [Dictionary(JBIG2Globals=globals_stream)]
            if globals_stream is not None
            else [None]
        )
        if filter_layout == "flate":
            data = zlib.compress(data)
            filters.insert(0, Name.FlateDecode)
            parms.insert(0, None)
        image.write(data, filter=Array(filters))
        if globals_stream is not None:
            image.DecodeParms = Array(parms)

    page = pdf.add_blank_page(page_size=(72, 72))
    page.Resources = Dictionary(XObject=Dictionary(Im0=image))
    page.Contents = pdf.make_stream(b"q 64 0 0 64 4 4 cm /Im0 Do Q")
    return image


def _get_globals(image: Stream) -> Stream | None:
    parms = image.get("/DecodeParms")
    if isinstance(parms, Array):
        parms = parms[-1]
    return parms.get("/JBIG2Globals") if parms is not None else None


@pytest.mark.parametrize("level", ["2b", "3b"])
@pytest.mark.parametrize(
    ("filter_layout", "segment_type", "with_globals"),
    [
        pytest.param("scalar", 6, True, id="scalar-globals"),
        pytest.param("array", 6, True, id="array-globals"),
        pytest.param("flate", 6, True, id="flate-globals"),
        pytest.param("scalar", 40, False, id="scalar-refinement-40"),
        pytest.param("array", 42, False, id="array-refinement-42"),
        pytest.param("flate", 43, False, id="flate-refinement-43"),
        pytest.param("scalar", 42, True, id="scalar-refinement-globals"),
        pytest.param("array", 43, True, id="array-refinement-globals"),
        pytest.param("flate", 40, True, id="flate-refinement-globals"),
    ],
)
def test_sanitization_preserves_jbig2_images(
    tmp_path: Path,
    level: str,
    filter_layout: str,
    segment_type: int,
    with_globals: bool,
) -> None:
    pdf = new_pdf()
    globals_stream = pdf.make_stream(_GLOBALS_DATA) if with_globals else None
    image = _add_image(pdf, _page_data(segment_type), filter_layout, globals_stream)
    original_bytes = image.read_raw_bytes()
    original_filter = image.Filter.unparse()

    result = sanitize_for_pdfa(pdf, level)

    assert result.get("jbig2_failed", 0) == 0
    assert result.get("jbig2_reencoded", 0) == 0
    assert image.read_raw_bytes() == original_bytes
    assert image.Filter.unparse() == original_filter
    if globals_stream is not None:
        assert _get_globals(image).objgen == globals_stream.objgen

    output = tmp_path / "sanitized.pdf"
    pdf.save(output)
    with pikepdf.open(output) as saved:
        image = saved.pages[0].Resources.XObject.Im0
        assert image.read_raw_bytes() == original_bytes
        assert image.Filter.unparse() == original_filter
        saved_globals = _get_globals(image)
        if with_globals:
            assert saved_globals is not None
            assert saved_globals.read_bytes() == _GLOBALS_DATA
        else:
            assert saved_globals is None


@pytest.mark.parametrize("level", ["2b", "3b"])
@pytest.mark.parametrize("filter_layout", ["scalar", "array", "flate"])
def test_shared_jbig2_globals_remain_shared_after_sanitization_and_save(
    tmp_path: Path, level: str, filter_layout: str
) -> None:
    pdf = new_pdf()
    globals_stream = pdf.make_stream(_GLOBALS_DATA)
    images = [
        _add_image(pdf, _page_data(6), filter_layout, globals_stream) for _ in range(2)
    ]
    original_bytes = [image.read_raw_bytes() for image in images]

    sanitize_for_pdfa(pdf, level)
    output = tmp_path / "shared.pdf"
    pdf.save(output)

    with pikepdf.open(output) as saved:
        images = [page.Resources.XObject.Im0 for page in saved.pages]
        assert [image.read_raw_bytes() for image in images] == original_bytes
        globals_streams = [_get_globals(image) for image in images]
        assert all(stream is not None for stream in globals_streams)
        assert globals_streams[0].objgen == globals_streams[1].objgen
        assert globals_streams[0].read_bytes() == _GLOBALS_DATA


@pytest.mark.parametrize("level", ["2b", "3b"])
@pytest.mark.parametrize("segment_type", [40, 42, 43])
def test_conversion_preserves_refinement_array_without_fallback(
    tmp_path: Path, level: str, segment_type: int
) -> None:
    """A refinement header must not require decoding or reject conversion."""
    pdf = new_pdf()
    original_data = _page_data(segment_type)
    _add_image(pdf, original_data, "array")
    source = tmp_path / "source.pdf"
    output = tmp_path / "converted.pdf"
    pdf.save(source)
    source_bytes = source.read_bytes()

    with patch(
        "pikepdf.jbig2.get_decoder",
        side_effect=AssertionError("JBIG2 preservation must not require a decoder"),
    ) as get_decoder:
        result = convert_to_pdfa(source, output, level=level, validate=False)

    assert result.success, result.error
    assert result.target_produced, result.warnings
    assert result.published and not result.skipped
    assert result.level == level
    get_decoder.assert_not_called()
    assert source.read_bytes() == source_bytes
    with pikepdf.open(output) as saved:
        image = saved.pages[0].Resources.XObject.Im0
        assert image.read_raw_bytes() == original_data
        # QPDF's converter save settings may collapse a singleton filter array.
        filter_obj = image.Filter
        filters = list(filter_obj) if isinstance(filter_obj, Array) else [filter_obj]
        assert filters == [Name.JBIG2Decode]
        assert _get_globals(image) is None
