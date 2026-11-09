# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""JBIG2 globals validation and deprecated import compatibility regressions."""

import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import new_pdf, pdf_object
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers.jbig2 import (
    convert_jbig2_external_globals,
    get_jbig2_globals_streams,
    validate_jbig2_globals,
)


def _image(pdf: Pdf, layout: str) -> Stream:
    """Create a referenced image with a synthetic payload and selected filters."""
    image = pdf.make_stream(b"synthetic JBIG2 image")
    image.Type = Name.XObject
    image.Subtype = Name.Image
    image.Width = 8
    image.Height = 8
    image.BitsPerComponent = 1
    image.ColorSpace = Name.DeviceGray
    image.Filter = Name.JBIG2Decode if layout == "scalar" else Array([Name.JBIG2Decode])
    page = pdf.add_blank_page(page_size=(72, 72))
    page.Resources = Dictionary(XObject=Dictionary(Im0=image))
    page.Contents = pdf.make_stream(b"q 64 0 0 64 4 4 cm /Im0 Do Q")
    return image


def _set_globals(image: Stream, layout: str, value: object) -> None:
    """Set the globals entry at the corresponding JBIG2 filter position."""
    parms = Dictionary(JBIG2Globals=value)
    image.DecodeParms = parms if layout == "scalar" else Array([parms])


@pytest.mark.parametrize("layout", ["scalar", "array"])
@pytest.mark.parametrize("value", [42, Name.Invalid, Dictionary()])
def test_non_stream_globals_are_rejected(layout: str, value: object) -> None:
    """Reject globals dictionaries, names, and integers before publication."""
    pdf = new_pdf()
    image = _image(pdf, layout)
    _set_globals(image, layout, pdf_object(value))

    with pytest.raises(ConversionError, match="/JBIG2Globals.*stream reference"):
        validate_jbig2_globals(pdf)


@pytest.mark.parametrize(
    ("layout", "parms"),
    [
        ("scalar", Name.Invalid),
        ("scalar", Array([Dictionary(), Dictionary()])),
        ("array", Array([])),
        ("array", Array([42])),
    ],
)
def test_malformed_jbig2_parameters_are_rejected(layout: str, parms: object) -> None:
    """Reject parameter values that cannot describe the specified filters."""
    pdf = new_pdf()
    image = _image(pdf, layout)
    image.DecodeParms = pdf_object(parms)

    with pytest.raises(ConversionError, match="Invalid /DecodeParms"):
        get_jbig2_globals_streams(pdf)


@pytest.mark.parametrize("layout", ["scalar", "array"])
@pytest.mark.parametrize("parameter_array", [False, True])
def test_single_filter_accepts_equivalent_parameter_forms(
    layout: str, parameter_array: bool
) -> None:
    """A singleton filter accepts its dictionary or equivalent singleton array."""
    pdf = new_pdf()
    image = _image(pdf, layout)
    globals_stream = pdf.make_stream(b"globals")
    parms = Dictionary(JBIG2Globals=globals_stream)
    image.DecodeParms = Array([parms]) if parameter_array else parms

    assert validate_jbig2_globals(pdf) == [globals_stream]


def test_multifilter_chain_requires_matching_parameter_array() -> None:
    """Several filters cannot share one ambiguous DecodeParms dictionary."""
    pdf = new_pdf()
    image = _image(pdf, "array")
    image.Filter = Array([Name.FlateDecode, Name.JBIG2Decode])
    image.DecodeParms = Dictionary(JBIG2Globals=pdf.make_stream(b"globals"))

    with pytest.raises(ConversionError, match="one parameter entry per filter"):
        get_jbig2_globals_streams(pdf)


@pytest.mark.parametrize("layout", ["scalar", "array"])
def test_missing_optional_globals_are_allowed(layout: str) -> None:
    """JBIG2 images need neither decode parameters nor a globals stream."""
    pdf = new_pdf()
    image = _image(pdf, layout)
    assert validate_jbig2_globals(pdf) == []

    image.DecodeParms = Dictionary() if layout == "scalar" else Array([None])
    assert validate_jbig2_globals(pdf) == []


def test_compressed_shared_globals_are_validated_without_image_decoding() -> None:
    """Decode outer Flate only, preserving shared globals and raw image data."""
    pdf = new_pdf()
    encoded = zlib.compress(b"synthetic globals")
    globals_stream = pdf.make_stream(encoded)
    globals_stream.Filter = Name.FlateDecode
    for layout in ["scalar", "array"]:
        _set_globals(_image(pdf, layout), layout, globals_stream)

    with patch(
        "pikepdf.jbig2.get_decoder",
        side_effect=AssertionError("image decoder must not be requested"),
    ) as decoder:
        streams = validate_jbig2_globals(pdf)

    assert [stream.objgen for stream in streams] == [globals_stream.objgen]
    assert globals_stream.read_raw_bytes() == encoded
    assert globals_stream.Filter == Name.FlateDecode
    decoder.assert_not_called()


def test_globals_are_selected_from_the_jbig2_filter_parameters() -> None:
    """Select the JBIG2 entry of a multifilter array, not another filter's entry."""
    pdf = new_pdf()
    image = _image(pdf, "array")
    globals_stream = pdf.make_stream(b"globals")
    image.Filter = Array([Name.FlateDecode, Name.JBIG2Decode])
    image.DecodeParms = Array(
        [Dictionary(JBIG2Globals=42), Dictionary(JBIG2Globals=globals_stream)]
    )

    assert get_jbig2_globals_streams(pdf) == [globals_stream]


@pytest.mark.parametrize("layout", ["scalar", "array"])
def test_undecodable_globals_outer_filter_is_rejected(layout: str) -> None:
    """A corrupt Flate container must not be accepted as readable JBIG2 globals."""
    pdf = new_pdf()
    globals_stream = pdf.make_stream(b"not a flate stream")
    globals_stream.Filter = Name.FlateDecode
    _set_globals(_image(pdf, layout), layout, globals_stream)

    with pytest.raises(ConversionError, match="Cannot decode /JBIG2Globals"):
        validate_jbig2_globals(pdf)


def test_legacy_import_remains_a_deprecated_noop() -> None:
    """Both historic import paths retain the result shape without mutating data."""
    from pdftopdfa.sanitizers import convert_jbig2_external_globals as reexported

    assert reexported is convert_jbig2_external_globals
    pdf = new_pdf()
    image = _image(pdf, "scalar")
    _set_globals(image, "scalar", 42)
    original_data = image.read_raw_bytes()
    original_parms = image.DecodeParms.unparse()

    with pytest.warns(DeprecationWarning, match="no longer modifies"):
        result = reexported(pdf)

    assert result == {"converted": 0, "reencoded": 0, "failed": 0}
    assert image.read_raw_bytes() == original_data
    assert image.DecodeParms.unparse() == original_parms


@pytest.mark.parametrize("layout", ["scalar", "array"])
@pytest.mark.parametrize("invalid_globals", ["non-stream", "corrupt-flate"])
def test_conversion_with_invalid_globals_preserves_original(
    tmp_path: Path, layout: str, invalid_globals: str
) -> None:
    """Malformed globals trigger original-input fallback even without veraPDF."""
    pdf = new_pdf()
    image = _image(pdf, layout)
    if invalid_globals == "corrupt-flate":
        value = pdf.make_stream(b"not a flate stream")
        value.Filter = Name.FlateDecode
    else:
        value = 42
    _set_globals(image, layout, value)
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    pdf.save(source)
    original = source.read_bytes()

    result = convert_to_pdfa(source, output, validate=False)

    assert result.success and result.skipped and result.published
    assert not result.target_produced
    assert result.level is None
    assert any("/JBIG2Globals" in warning for warning in result.warnings)
    assert source.read_bytes() == output.read_bytes() == original
