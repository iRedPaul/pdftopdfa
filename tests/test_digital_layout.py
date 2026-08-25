# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for read-only digital PDF painting provenance."""

import base64
import zlib
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, String

import pdftopdfa.digital_layout as digital_layout
from pdftopdfa.digital_layout import (
    DirectTextSpan,
    DirectXObjectSpan,
    extract_digital_layout,
)
from pdftopdfa.exceptions import ConversionError


def _font(pdf: pikepdf.Pdf) -> Dictionary:
    return pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
            Encoding=Name.WinAnsiEncoding,
        )
    )


def _page(
    pdf: pikepdf.Pdf,
    content: bytes,
    resources: Dictionary | None = None,
) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=(200, 100))
    page.obj["/Contents"] = pdf.make_stream(content)
    page.obj["/Resources"] = resources or Dictionary()
    return page


def _form(
    pdf: pikepdf.Pdf,
    content: bytes,
    resources: Dictionary,
    bbox: tuple[float, float, float, float] = (0, 0, 20, 10),
) -> pikepdf.Stream:
    form = pdf.make_stream(content)
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array(bbox)
    form["/Resources"] = resources
    return form


def _image(
    pdf: pikepdf.Pdf,
    data: bytes,
    *,
    width: int = 1,
    height: int = 1,
    color_space: object = Name.DeviceGray,
    bits_per_component: int = 8,
) -> pikepdf.Stream:
    image = pdf.make_stream(data)
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = width
    image["/Height"] = height
    image["/ColorSpace"] = color_space
    image["/BitsPerComponent"] = bits_per_component
    return image


def test_extracts_direct_text_events_with_effective_style_and_order() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"q 2 0 0 2 0 0 cm "
            b"BT /F1 10 Tf 10 10 Td [(Vis) 0 (ible)] TJ "
            b"3 Tr 0 15 Td (Hidden) Tj ET Q"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    pages = extract_digital_layout(pdf)

    assert len(pages) == 1
    page = pages[0]
    assert (page.page_index, page.width, page.height) == (0, 200.0, 100.0)
    assert all(isinstance(span, DirectTextSpan) for span in page.spans)
    first, second = page.spans
    assert isinstance(first, DirectTextSpan)
    assert isinstance(second, DirectTextSpan)
    assert [first.text, second.text] == ["Visible", "Hidden"]
    assert [first.direct_text_index, second.direct_text_index] == [0, 1]
    assert [first.font_name, second.font_name] == ["Helvetica", "Helvetica"]
    assert first.font_size == pytest.approx(20)
    assert second.font_size == pytest.approx(20)
    assert (first.render_mode, first.invisible) == (0, False)
    assert (second.render_mode, second.invisible) == (3, True)
    assert first.bbox[0] < first.bbox[2]
    assert first.bbox[1] < first.bbox[3]


@pytest.mark.parametrize(
    "content",
    [
        b"BT /F1 10 Tf 10 10 Td 12 Tj ET",
        b"BT /F1 10 Tf 10 10 Td (text) TJ ET",
        b"BT /F1 10 Tf 10 10 Td 12 ' ET",
        b'BT /F1 10 Tf 10 10 Td (spacing) 0 (text) " ET',
    ],
)
def test_rejects_invalid_direct_text_operand_types(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, content, Dictionary(Font=Dictionary(F1=_font(pdf))))

    with pytest.raises(ConversionError, match="direct text provenance is malformed"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize(
    ("content", "expected_text"),
    [
        (b"BT /F1 10 Tf 10 10 Td [(A) 12.25 (B)] TJ ET", "AB"),
        (b'BT /F1 10 Tf 10 10 Td 1.25 2.5 (Text) " ET', "Text"),
    ],
)
def test_accepts_finite_decimal_direct_text_operands(
    content: bytes,
    expected_text: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, content, Dictionary(Font=Dictionary(F1=_font(pdf))))

    spans = extract_digital_layout(pdf)[0].spans

    assert [span.text for span in spans] == [expected_text]


@pytest.mark.parametrize(
    "value",
    [
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
def test_rejects_nonfinite_decimal_pdf_numbers(value: Decimal) -> None:
    assert not digital_layout._is_finite_pdf_number(value)


def test_rejects_finite_decimal_outside_float_range() -> None:
    assert not digital_layout._is_finite_pdf_number(Decimal("1E1000000"))


@pytest.mark.parametrize("value", [True, "1", object()])
def test_rejects_non_numeric_pdf_numbers(value: object) -> None:
    assert not digital_layout._is_finite_pdf_number(value)


def test_recovers_unencoded_base14_cid_text_and_advance() -> None:
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
        )
    )
    _page(
        pdf,
        b"BT /F1 8 Tf 1 0 0 1 15 80 Tm (NATIVE HDR) Tj (X) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
    )

    first, second = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectTextSpan)
    assert isinstance(second, DirectTextSpan)
    assert (first.text, second.text) == ("NATIVE HDR", "X")
    assert first.bbox == pytest.approx((15, 78.344, 63.448, 86.344))
    assert second.bbox == pytest.approx((63.448, 78.344, 68.784, 86.344))


def test_records_image_and_inline_image_calls_in_execution_order() -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\xff\x00\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceRGB
    image["/BitsPerComponent"] = 8
    _page(
        pdf,
        (
            b"q 20 0 0 30 10 15 cm /Im0 Do Q "
            b"q 5 0 0 6 1 2 cm "
            b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI Q"
        ),
        Dictionary(XObject=Dictionary(Im0=image)),
    )

    page = extract_digital_layout(pdf)[0]

    assert all(isinstance(span, DirectXObjectSpan) for span in page.spans)
    external, inline = page.spans
    assert isinstance(external, DirectXObjectSpan)
    assert isinstance(inline, DirectXObjectSpan)
    assert external.kind == "image"
    assert external.bbox == pytest.approx((10, 15, 30, 45))
    assert external.resource_name == "Im0"
    assert external.direct_xobject_index == 0
    assert external.text_runs == ()
    assert inline.kind == "inline_image"
    assert inline.bbox == pytest.approx((1, 2, 6, 8))
    assert inline.resource_name is None
    assert inline.direct_xobject_index == 1


@pytest.mark.parametrize(
    ("alpha", "decode", "invisible"),
    [
        (0, None, True),
        (255, None, False),
        (255, (1, 0), True),
        (0, (1, 0), False),
    ],
)
def test_classifies_intrinsic_soft_mask_visibility(
    alpha: int,
    decode: tuple[int, int] | None,
    invisible: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    mask = _image(pdf, bytes([alpha]))
    if decode is not None:
        mask["/Decode"] = Array(decode)
    image["/SMask"] = mask
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is invisible
    assert span.final_paint_uncertain is False
    assert span.intrinsic_visibility_uncertain is False


@pytest.mark.parametrize(
    ("sample", "decode", "invisible"),
    [
        (0x80, (0, 1), True),
        (0x00, (0, 1), False),
        (0x00, (1, 0), True),
        (0x80, (1, 0), False),
    ],
)
def test_classifies_explicit_stencil_mask_visibility(
    sample: int,
    decode: tuple[int, int],
    invisible: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    mask = _image(pdf, bytes([sample]), bits_per_component=1)
    del mask["/ColorSpace"]
    mask["/ImageMask"] = True
    mask["/Decode"] = Array(decode)
    image["/Mask"] = mask
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is invisible
    assert span.final_paint_uncertain is False


@pytest.mark.parametrize(
    ("data", "width", "mask", "color_space", "invisible"),
    [
        (b"\xff\x00\x00", 1, (255, 255, 0, 0, 0, 0), Name.DeviceRGB, True),
        (
            b"\xff\x00\x00\x00\x00\xff",
            2,
            (255, 255, 0, 0, 0, 0),
            Name.DeviceRGB,
            False,
        ),
        (
            bytes([41]),
            1,
            (41, 41),
            Array([Name.Indexed, Name.DeviceRGB, 255, String(b"\x00" * 768)]),
            True,
        ),
    ],
)
def test_classifies_color_key_mask_visibility(
    data: bytes,
    width: int,
    mask: tuple[int, ...],
    color_space: object,
    invisible: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(
        pdf,
        data,
        width=width,
        color_space=color_space,
    )
    image["/Mask"] = Array(mask)
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is invisible
    assert span.final_paint_uncertain is False


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize(("sample", "invisible"), [(0x80, True), (0x00, False)])
def test_classifies_external_and_inline_stencil_visibility(
    inline: bool,
    sample: int,
    invisible: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    if inline:
        content = b"q BI /W 1 /H 1 /IM true /D [0 1] ID " + bytes([sample]) + b" EI Q"
        resources = Dictionary()
    else:
        image = _image(pdf, bytes([sample]), bits_per_component=1)
        del image["/ColorSpace"]
        image["/ImageMask"] = True
        image["/Decode"] = Array([0, 1])
        content = b"/Im Do"
        resources = Dictionary(XObject=Dictionary(Im=image))
    _page(pdf, content, resources)

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is invisible
    assert span.final_paint_uncertain is False


@pytest.mark.parametrize("smask_in_data", [1, 2])
@pytest.mark.parametrize(("alpha", "invisible"), [(0, True), (255, False)])
def test_classifies_jpx_smask_in_data(
    smask_in_data: int,
    alpha: int,
    invisible: bool,
) -> None:
    from PIL import Image

    encoded = BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, alpha)).save(
        encoded,
        format="JPEG2000",
    )
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, encoded.getvalue(), color_space=Name.DeviceRGB)
    image["/Filter"] = Name.JPXDecode
    image["/SMaskInData"] = smask_in_data
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is invisible
    assert span.final_paint_uncertain is False


def test_intrinsic_mask_budget_exhaustion_is_uncertain(monkeypatch) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    image["/SMask"] = _image(pdf, b"\x00")
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))
    monkeypatch.setattr(digital_layout, "_MAX_IMAGE_VISIBILITY_PIXELS", 0)

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.invisible is False
    assert span.final_paint_uncertain is True
    assert span.intrinsic_visibility_uncertain is True


def test_intrinsic_uncertainty_does_not_mask_complex_blend_uncertainty() -> None:
    pdf = pikepdf.Pdf.new()
    mask = _image(pdf, zlib.compress(b"\x00"))
    mask["/Filter"] = Name.FlateDecode
    mask["/DecodeParms"] = Dictionary(Predictor=12)
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    image["/SMask"] = mask
    _page(
        pdf,
        b"/Blend gs /Im Do",
        Dictionary(
            ExtGState=Dictionary(Blend=Dictionary(BM=Name.Multiply)),
            XObject=Dictionary(Im=image),
        ),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.final_paint_uncertain is True
    assert span.intrinsic_visibility_uncertain is True
    assert span.non_intrinsic_visibility_uncertain is True


def test_intrinsic_image_visibility_is_cached_per_resource(monkeypatch) -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    image["/SMask"] = _image(pdf, b"\x00")
    _page(pdf, b"/Im Do /Im Do", Dictionary(XObject=Dictionary(Im=image)))
    original = digital_layout._decoded_image_samples
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(digital_layout, "_decoded_image_samples", counted)

    spans = extract_digital_layout(pdf)[0].spans

    assert all(isinstance(span, DirectXObjectSpan) and span.invisible for span in spans)
    assert calls == 1


@pytest.mark.parametrize("mask_entry", ["/SMask", "/Mask"])
def test_shared_mask_stream_is_decoded_once_across_4097_parents(
    mask_entry: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    if mask_entry == "/SMask":
        mask = _image(pdf, b"\x00")
    else:
        mask = _image(pdf, b"\x80", bits_per_component=1)
        del mask["/ColorSpace"]
        mask["/ImageMask"] = True
        mask["/Decode"] = Array([0, 1])
    images = []
    for _index in range(4_097):
        image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
        image[mask_entry] = mask
        images.append(image)

    budget = digital_layout._ImageVisibilityBudget()
    visibility = [budget.classify_stream(image) for image in images]

    assert set(visibility) == {"invisible"}
    assert budget.decodes == 1
    assert budget.encoded_bytes == 1
    assert len(budget.encoded_streams) == 1


def test_cumulative_encoded_budget_stops_external_reads_and_decodes(
    monkeypatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    encoded = zlib.compress(b"\x00")
    masks = []
    images = []
    for _index in range(2):
        mask = _image(pdf, encoded)
        mask["/Filter"] = Name.FlateDecode
        image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
        image["/SMask"] = mask
        masks.append(mask)
        images.append(image)
    mask_keys = {mask.objgen for mask in masks}
    object_type = type(masks[0])
    original_read = object_type.get_raw_stream_buffer
    original_decode = digital_layout._bounded_flate_decode
    raw_reads = 0
    decodes = 0

    def counted_read(stream):
        nonlocal raw_reads
        if stream.objgen in mask_keys:
            raw_reads += 1
        return original_read(stream)

    def counted_decode(*args, **kwargs):
        nonlocal decodes
        decodes += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(object_type, "get_raw_stream_buffer", counted_read)
    monkeypatch.setattr(digital_layout, "_bounded_flate_decode", counted_decode)
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT",
        len(encoded),
    )

    budget = digital_layout._ImageVisibilityBudget()
    visibility = [budget.classify_stream(image) for image in images]

    assert visibility == ["invisible", "uncertain"]
    assert raw_reads == 1
    assert decodes == 1
    assert budget.encoded_bytes == len(encoded)
    assert len(budget.encoded_streams) == 1


def test_cumulative_encoded_budget_uses_inline_owner_and_index_keys(
    monkeypatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    inline = b"BI /W 1 /H 1 /IM true /D [0 1] ID \x80 EI"
    _page(pdf, b"q " + b" Q q ".join((inline, inline, inline)) + b" Q")
    object_type = type(pdf.pages[0].obj)
    original_read = object_type.get_raw_stream_buffer
    raw_reads = 0

    def counted_read(stream):
        nonlocal raw_reads
        if stream.get("/Subtype") == Name.Image:
            raw_reads += 1
        return original_read(stream)

    monkeypatch.setattr(object_type, "get_raw_stream_buffer", counted_read)
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT",
        2,
    )

    spans = extract_digital_layout(pdf)[0].spans

    assert len(spans) == 3
    assert all(isinstance(span, DirectXObjectSpan) for span in spans)
    assert [span.invisible for span in spans] == [True, True, False]
    assert [span.intrinsic_visibility_uncertain for span in spans] == [
        False,
        False,
        True,
    ]
    assert raw_reads == 2


@pytest.mark.parametrize(
    ("declared_length", "encoded_limit", "expected_reads"),
    [(2, 1, 0), (0, 0, 1)],
)
def test_checks_declared_and_actual_image_lengths(
    monkeypatch,
    declared_length: int,
    encoded_limit: int,
    expected_reads: int,
) -> None:
    pdf = pikepdf.Pdf.new()
    mask = _image(pdf, b"\x00")
    object_type = type(mask)
    original_get = object_type.get
    original_read = object_type.get_raw_stream_buffer
    reads = 0

    def spoofed_get(stream, name, *args):
        if stream.objgen == mask.objgen and str(name) == "/Length":
            return declared_length
        return original_get(stream, name, *args)

    def counted_read(stream):
        nonlocal reads
        if stream.objgen == mask.objgen:
            reads += 1
        return original_read(stream)

    monkeypatch.setattr(object_type, "get", spoofed_get)
    monkeypatch.setattr(object_type, "get_raw_stream_buffer", counted_read)
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_IMAGE_VISIBILITY_BYTES",
        encoded_limit,
    )

    budget = digital_layout._ImageVisibilityBudget()
    visibility = budget.classify_mask_stream(mask, mode="soft_mask")

    assert visibility == "uncertain"
    assert reads == expected_reads
    assert budget.encoded_bytes == expected_reads


def test_visibility_uses_canonical_stream_lengths_after_serialization(
    monkeypatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    mask = _image(pdf, zlib.compress(b"\x00"))
    mask["/Filter"] = Name.FlateDecode
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    image["/SMask"] = mask
    _page(pdf, b"/Im Do", Dictionary(XObject=Dictionary(Im=image)))
    object_type = type(mask)
    original_get = object_type.get
    original_read = object_type.get_raw_stream_buffer
    original_apply = digital_layout._apply_intrinsic_image_visibility
    visibility_active = False
    visibility_pdf_uses_original = None
    visibility_image_reads: list[bool] = []

    def lying_get(stream, name, *args):
        if (
            stream.objgen == mask.objgen
            and stream.same_owner_as(mask)
            and str(name) == "/Length"
        ):
            return 0
        return original_get(stream, name, *args)

    def counted_read(stream):
        if visibility_active and original_get(stream, "/Subtype") == Name.Image:
            visibility_image_reads.append(stream.same_owner_as(mask))
        return original_read(stream)

    def tracked_apply(candidate, layouts):
        nonlocal visibility_active, visibility_pdf_uses_original
        visibility_pdf_uses_original = candidate.pages[0].obj.same_owner_as(
            pdf.pages[0].obj
        )
        visibility_active = True
        try:
            return original_apply(candidate, layouts)
        finally:
            visibility_active = False

    monkeypatch.setattr(object_type, "get", lying_get)
    monkeypatch.setattr(object_type, "get_raw_stream_buffer", counted_read)
    monkeypatch.setattr(
        digital_layout,
        "_apply_intrinsic_image_visibility",
        tracked_apply,
    )
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_IMAGE_VISIBILITY_BYTES",
        0,
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.intrinsic_visibility_uncertain is True
    assert visibility_pdf_uses_original is False
    assert visibility_image_reads == []


def test_cumulative_sample_budget_limits_one_bit_scanning(monkeypatch) -> None:
    pdf = pikepdf.Pdf.new()
    images = []
    for _index in range(2):
        image = _image(
            pdf,
            b"\xff",
            width=8,
            bits_per_component=1,
        )
        del image["/ColorSpace"]
        image["/ImageMask"] = True
        image["/Decode"] = Array([0, 1])
        images.append(image)
    _page(
        pdf,
        b"/First Do /Second Do",
        Dictionary(XObject=Dictionary(First=images[0], Second=images[1])),
    )
    monkeypatch.setattr(
        digital_layout,
        "_MAX_IMAGE_VISIBILITY_SAMPLES_PER_DOCUMENT",
        8,
    )

    first, second = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectXObjectSpan)
    assert first.invisible is True
    assert isinstance(second, DirectXObjectSpan)
    assert second.invisible is False
    assert second.intrinsic_visibility_uncertain is True


def test_real_invoice_indexed_color_key_images_are_proven_visible() -> None:
    source = Path(__file__).parents[1] / "test_docs" / "180 - R3 - Rechnung.pdf"
    if not source.is_file():
        pytest.skip(f"Real-document regression fixture is missing: {source}")
    with pikepdf.Pdf.open(source) as pdf:
        images = [
            item
            for item in pdf.objects
            if isinstance(item, pikepdf.Stream)
            and item.get("/Subtype") == Name.Image
            and isinstance(digital_layout.resolve_indirect(item.get("/Mask")), Array)
        ]

        assert len(images) == 2
        budget = digital_layout._ImageVisibilityBudget()
        assert [budget.classify_stream(image) for image in images] == [
            "visible",
            "visible",
        ]


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        ("/Width", b"0"),
        ("/Height", b"-1"),
        ("/Width", b"1.5"),
        ("/Height", b"true"),
    ],
)
def test_rejects_invalid_image_xobject_dimensions(
    dimension: str,
    value: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    image = pdf.make_stream(b"\x00")
    image["/Type"] = Name.XObject
    image["/Subtype"] = Name.Image
    image["/Width"] = 1
    image["/Height"] = 1
    image["/ColorSpace"] = Name.DeviceGray
    image["/BitsPerComponent"] = 8
    image[dimension] = pikepdf.Object.parse(value)
    _page(pdf, b"/Im0 Do", Dictionary(XObject=Dictionary(Im0=image)))

    with pytest.raises(ConversionError, match="Could not extract"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (b"0", b"1"),
        (b"1", b"-1"),
        (b"1.5", b"1"),
        (b"1", b"true"),
    ],
)
def test_rejects_invalid_inline_image_dimensions(
    width: bytes,
    height: bytes,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        b"BI /W " + width + b" /H " + height + b" /CS /G /BPC 8 ID \x00 EI",
    )

    with pytest.raises(ConversionError, match="invalid dimensions"):
        extract_digital_layout(pdf)


def test_aggregates_nested_form_text_without_direct_text_provenance() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    inner = _form(
        pdf,
        b"BT /F1 4 Tf 3 Tr 1 2 Td (Inner) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
    )
    outer = _form(
        pdf,
        b"q /Inner Do Q BT /F1 6 Tf 2 12 Td (Outer) Tj ET",
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Inner=inner),
        ),
        bbox=(0, 0, 30, 20),
    )
    _page(
        pdf,
        (
            b"BT /F1 8 Tf 1 2 Td (Before) Tj ET "
            b"q 2 0 0 2 20 30 cm /Outer Do Q "
            b"BT /F1 8 Tf 70 80 Td (After) Tj ET"
        ),
        Dictionary(
            Font=Dictionary(F1=font),
            XObject=Dictionary(Outer=outer),
        ),
    )

    page = extract_digital_layout(pdf)[0]

    assert [type(span) for span in page.spans] == [
        DirectTextSpan,
        DirectXObjectSpan,
        DirectTextSpan,
    ]
    before, invocation, after = page.spans
    assert isinstance(before, DirectTextSpan)
    assert isinstance(invocation, DirectXObjectSpan)
    assert isinstance(after, DirectTextSpan)
    assert [before.text, after.text] == ["Before", "After"]
    assert [before.direct_text_index, after.direct_text_index] == [0, 1]
    assert invocation.kind == "form"
    assert invocation.resource_name == "Outer"
    assert invocation.direct_xobject_index == 0
    assert invocation.bbox == pytest.approx((20, 30, 80, 70))
    assert invocation.text == "InnerOuter"
    assert [run.text for run in invocation.text_runs] == ["Inner", "Outer"]
    assert [run.font_size for run in invocation.text_runs] == pytest.approx([8, 12])
    assert [run.render_mode for run in invocation.text_runs] == [3, 0]
    assert [run.invisible for run in invocation.text_runs] == [True, False]
    nested, outer_text = invocation.children
    assert isinstance(nested, DirectXObjectSpan)
    assert isinstance(outer_text, DirectTextSpan)
    assert nested.kind == "form"
    assert nested.resource_name == "Inner"
    assert nested.direct_xobject_index == 0
    assert [child.text for child in nested.children] == ["Inner"]
    assert outer_text.text == "Outer"
    assert outer_text.direct_text_index == 0


def test_records_each_reused_form_invocation_independently() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        b"BT /F1 5 Tf 6 7 Td (Repeated) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(5, 6, 20, 10),
    )
    _page(
        pdf,
        (b"q 1 0 0 1 10 10 cm /Fm0 Do Q q 2 0 0 2 50 20 cm /Fm0 Do Q"),
        Dictionary(XObject=Dictionary(Fm0=form)),
    )

    page = extract_digital_layout(pdf)[0]

    assert all(isinstance(span, DirectXObjectSpan) for span in page.spans)
    first, second = page.spans
    assert isinstance(first, DirectXObjectSpan)
    assert isinstance(second, DirectXObjectSpan)
    assert [first.direct_xobject_index, second.direct_xobject_index] == [0, 1]
    assert [first.resource_name, second.resource_name] == ["Fm0", "Fm0"]
    assert [first.text, second.text] == ["Repeated", "Repeated"]
    assert first.bbox == pytest.approx((15, 16, 30, 20))
    assert second.bbox == pytest.approx((60, 32, 90, 40))
    assert first.text_runs[0].bbox != second.text_runs[0].bbox


def test_records_active_clipping_path_for_direct_text() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"q 0 0 20 20 re W n "
            b"BT /F1 5 Tf 5 5 Td (inside) Tj 95 95 Td (outside) Tj ET Q"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    inside, outside = extract_digital_layout(pdf)[0].spans

    assert isinstance(inside, DirectTextSpan)
    assert isinstance(outside, DirectTextSpan)
    assert inside.clip_bbox == pytest.approx((0, 0, 20, 20))
    assert outside.clip_bbox == pytest.approx((0, 0, 20, 20))
    assert inside.bbox[0] < 20
    assert outside.bbox[0] > 20


def test_propagates_nonrectangular_text_clipping_uncertainty() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"BT /F1 20 Tf 7 Tr 5 5 Td (X) Tj ET "
            b"BT /F1 5 Tf 0 Tr 100 50 Td (outside) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    clipped, outside = extract_digital_layout(pdf)[0].spans

    assert isinstance(clipped, DirectTextSpan)
    assert isinstance(outside, DirectTextSpan)
    assert clipped.final_paint_uncertain is False
    assert outside.final_paint_uncertain is True


def test_accepts_text_clip_when_saved_graphics_state_is_restored() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"q BT /F1 20 Tf 7 Tr 5 5 Td (X) Tj ET Q "
            b"BT /F1 5 Tf 0 Tr 100 50 Td (visible) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    clipped, visible = extract_digital_layout(pdf)[0].spans

    assert clipped.render_mode == 7
    assert visible.text == "visible"
    assert visible.clip_bbox is None


def test_empty_text_clip_does_not_make_clip_inexact() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"BT /F1 10 Tf 7 Tr 10 10 Td () Tj ET "
            b"BT /F1 10 Tf 0 Tr 20 20 Td (Visible) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
    )

    empty, visible = extract_digital_layout(pdf)[0].spans

    assert empty.text == ""
    assert visible.text == "Visible"
    assert visible.clip_bbox is None


def test_propagates_text_clip_uncertainty_inside_form_only() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        (
            b"BT /F1 20 Tf 7 Tr 5 5 Td (X) Tj ET "
            b"BT /F1 5 Tf 0 Tr 100 50 Td (hidden) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(0, 0, 200, 100),
    )
    _page(
        pdf,
        b"/Fm Do BT /F1 5 Tf 0 Tr 100 50 Td (page) Tj ET",
        Dictionary(Font=Dictionary(F1=font), XObject=Dictionary(Fm=form)),
    )

    form_span, page_span = extract_digital_layout(pdf)[0].spans

    assert isinstance(form_span, DirectXObjectSpan)
    clipped, hidden = form_span.children
    assert isinstance(clipped, DirectTextSpan)
    assert isinstance(hidden, DirectTextSpan)
    assert clipped.final_paint_uncertain is False
    assert hidden.final_paint_uncertain is True
    assert isinstance(page_span, DirectTextSpan)
    assert page_span.final_paint_uncertain is False


@pytest.mark.parametrize(
    ("memory_limit", "expected_rollover"),
    [(16 * 1024 * 1024, False), (1, True)],
)
def test_serializes_to_bounded_spool_without_mutating_pdf(
    monkeypatch: pytest.MonkeyPatch,
    memory_limit: int,
    expected_rollover: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"q Q")
    original_content = bytes(page.obj["/Contents"].read_bytes())
    original_object_count = len(pdf.objects)
    original_root_keys = set(pdf.Root.keys())
    buffers: list[SpooledTemporaryFile] = []

    class TrackingSpooledTemporaryFile(SpooledTemporaryFile):
        def __init__(self, *, max_size: int, mode: str) -> None:
            self.did_rollover = False
            super().__init__(max_size=max_size, mode=mode)
            self.config = (max_size, mode)
            buffers.append(self)

        def rollover(self) -> None:
            self.did_rollover = True
            super().rollover()

    monkeypatch.setattr(
        digital_layout,
        "_SERIALIZED_PDF_MEMORY_LIMIT",
        memory_limit,
    )
    monkeypatch.setattr(
        digital_layout,
        "SpooledTemporaryFile",
        TrackingSpooledTemporaryFile,
    )

    pages = extract_digital_layout(pdf)

    assert len(pages) == 1
    assert pages[0].spans == ()
    assert len(buffers) == 1
    assert buffers[0].config == (memory_limit, "w+b")
    assert buffers[0].did_rollover is expected_rollover
    assert buffers[0].closed is True
    assert bytes(page.obj["/Contents"].read_bytes()) == original_content
    assert len(pdf.objects) == original_object_count
    assert set(pdf.Root.keys()) == original_root_keys


def test_rejects_extraction_blocked_pdf() -> None:
    source = pikepdf.Pdf.new()
    _page(source, b"q Q")
    serialized = BytesIO()
    source.save(
        serialized,
        encryption=pikepdf.Encryption(
            owner="owner",
            user="user",
            allow=pikepdf.Permissions(extract=False),
        ),
    )
    serialized.seek(0)

    with pikepdf.Pdf.open(serialized, password="user") as blocked:
        assert blocked.allow.extract is False
        with pytest.raises(ConversionError, match="not permitted"):
            extract_digital_layout(blocked)


@pytest.mark.parametrize(
    "content",
    [
        b"/Missing Do",
        b"BT /Missing 10 Tf (text) Tj ET",
    ],
)
def test_rejects_invalid_painting_resource(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, content)

    with pytest.raises(ConversionError, match="Digital layout|Could not extract"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize("shading_resource", ["missing", "scalar"])
def test_rejects_missing_or_invalid_shading_resource(
    shading_resource: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    resources = Dictionary()
    if shading_resource == "scalar":
        resources["/Shading"] = Dictionary(S0=1)
    _page(pdf, b"/S0 sh", resources)

    with pytest.raises(ConversionError, match="Could not extract"):
        extract_digital_layout(pdf)


def test_accepts_stream_shading_resource() -> None:
    pdf = pikepdf.Pdf.new()
    vertex_data = (
        b"\x00\x00\x00\x00\x00\xff\x00\x00"
        b"\x00\xff\xff\x00\x00\x00\xff\x00"
        b"\x00\x00\x00\xff\xff\x00\x00\xff"
    )
    shading = pdf.make_stream(vertex_data)
    shading["/ShadingType"] = 4
    shading["/ColorSpace"] = Name.DeviceRGB
    shading["/BitsPerCoordinate"] = 16
    shading["/BitsPerComponent"] = 8
    shading["/BitsPerFlag"] = 8
    shading["/Decode"] = Array([0, 200, 0, 100, 0, 1, 0, 1, 0, 1])
    _page(pdf, b"/S0 sh", Dictionary(Shading=Dictionary(S0=shading)))

    assert extract_digital_layout(pdf)[0].spans == ()


def test_rejects_malformed_page_contents() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page()
    page.obj["/Contents"] = Dictionary()

    with pytest.raises(ConversionError, match="malformed page 1"):
        extract_digital_layout(pdf)


def test_page_filter_preserves_physical_indices_and_skips_malformed_content() -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"q Q")
    ignored = pdf.add_blank_page(page_size=(200, 100))
    ignored.obj["/Contents"] = Dictionary()
    _page(pdf, b"q Q")

    pages = extract_digital_layout(pdf, page_indices=frozenset({0, 2}))

    assert [page.page_index for page in pages] == [0, 2]


def test_page_filter_rejects_invalid_physical_index() -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"q Q")

    with pytest.raises(ConversionError, match="page selection is invalid"):
        extract_digital_layout(pdf, page_indices=frozenset({1}))


def test_rejects_exponential_form_invocation_expansion() -> None:
    pdf = pikepdf.Pdf.new()
    leaf = _form(pdf, b"q Q", Dictionary())
    current = leaf
    for _ in range(12):
        current = _form(
            pdf,
            b"/N Do /N Do",
            Dictionary(XObject=Dictionary(N=current)),
        )
    _page(pdf, b"/Root Do", Dictionary(XObject=Dictionary(Root=current)))

    with pytest.raises(ConversionError, match="invocation budget exceeded"):
        extract_digital_layout(pdf)


def test_rejects_page_operator_budget_before_pdfminer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"q Q q Q q Q")
    monkeypatch.setattr(digital_layout, "_MAX_DIGITAL_OPERATORS_PER_PAGE", 5)

    with pytest.raises(ConversionError, match="page operator budget exceeded"):
        extract_digital_layout(pdf)


def test_rejects_document_operator_budget_before_pdfminer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"q Q q Q")
    _page(pdf, b"q Q q Q")
    monkeypatch.setattr(digital_layout, "_MAX_DIGITAL_OPERATORS_PER_DOCUMENT", 7)

    with pytest.raises(ConversionError, match="document operator budget exceeded"):
        extract_digital_layout(pdf)


def test_rejects_flate_bomb_before_content_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    stream = page.obj["/Contents"]
    stream.write(
        zlib.compress(b" " * 4_096 + b"q Q"),
        filter=Name.FlateDecode,
    )
    monkeypatch.setattr(
        digital_layout,
        "_MAX_DECODED_CONTENT_BYTES_PER_CONTAINER",
        128,
    )

    def unexpected_parse(_owner):
        pytest.fail("content parser ran before the decoded-byte guard")

    monkeypatch.setattr(pikepdf, "parse_content_stream", unexpected_parse)

    with pytest.raises(ConversionError, match="container byte budget exceeded"):
        extract_digital_layout(pdf)


def test_accepts_empty_flate_page_content() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    page.obj["/Contents"].write(b"", filter=Name.FlateDecode)

    assert extract_digital_layout(pdf)[0].spans == ()


@pytest.mark.parametrize("raw", [b"x", zlib.compress(b"q Q")[:-1]])
def test_bounded_flate_decoder_rejects_truncated_data(raw: bytes) -> None:
    with pytest.raises(ConversionError, match="malformed"):
        digital_layout._bounded_flate_decode(raw, 100)


def test_accepts_empty_flate_content_with_declared_length() -> None:
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Resources <<>> /Contents 4 0 R >>",
        b"<< /Length 0 /Filter /FlateDecode >>\nstream\nendstream",
    )
    raw_pdf = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, value in enumerate(objects, start=1):
        offsets.append(len(raw_pdf))
        raw_pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        raw_pdf.extend(value)
        raw_pdf.extend(b"\nendobj\n")
    xref_offset = len(raw_pdf)
    raw_pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    raw_pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        raw_pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    raw_pdf.extend(
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )

    with pikepdf.Pdf.open(BytesIO(raw_pdf)) as pdf:
        stream = pdf.pages[0].obj["/Contents"]
        assert stream.get("/Length") == 0
        assert bytes(stream.get_raw_stream_buffer()) == b""
        assert digital_layout._bounded_flate_decode(b"", 0) == b""
        assert digital_layout._decoded_content_stream_size(stream, 100) == 0
        digital_layout._DecodedContentBudget().charge_once(stream, 0)


@pytest.mark.parametrize(
    ("ascii85_name", "flate_name"),
    [
        (Name.ASCII85Decode, Name.FlateDecode),
        (Name("/A85"), Name("/Fl")),
    ],
)
def test_accepts_bounded_ascii85_flate_filter_chain(
    ascii85_name: Name,
    flate_name: Name,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    encoded = base64.a85encode(zlib.compress(b"q Q")) + b"~>"
    page.obj["/Contents"].write(
        encoded,
        filter=Array([ascii85_name, flate_name]),
        decode_parms=Array([None, Dictionary(Predictor=1)]),
    )

    assert extract_digital_layout(pdf)[0].spans == ()


def test_accepts_flate_png_predictor_content() -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    page.obj["/Contents"].write(
        zlib.compress(b"\x02q Q"),
        filter=Name.FlateDecode,
        decode_parms=Dictionary(
            Predictor=12,
            Colors=1,
            BitsPerComponent=8,
            Columns=3,
        ),
    )

    assert extract_digital_layout(pdf)[0].spans == ()


@pytest.mark.parametrize("filter_name", [Name.LZWDecode, Name.Crypt])
def test_rejects_unbounded_content_filter_chain_before_parser(
    monkeypatch: pytest.MonkeyPatch,
    filter_name: Name,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    page.obj["/Contents"].write(b"encoded", filter=filter_name)

    def unexpected_parse(_owner):
        pytest.fail("content parser ran for an unbounded filter")

    monkeypatch.setattr(pikepdf, "parse_content_stream", unexpected_parse)

    with pytest.raises(ConversionError, match="strict bound"):
        extract_digital_layout(pdf)


def test_checks_declared_encoded_length_before_reading_raw_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedStream:
        def get(self, key: str):
            return 5 if key == "/Length" else None

        def get_raw_stream_buffer(self):
            pytest.fail("raw stream was materialized before the declared-length guard")

    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER",
        4,
    )

    with pytest.raises(ConversionError, match="encoded content container"):
        digital_layout._decoded_content_stream_size(OversizedStream(), 100)


def test_checks_actual_encoded_length_after_reading_raw_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MismatchedStream:
        def get(self, key: str):
            return 1 if key == "/Length" else None

        def get_raw_stream_buffer(self):
            return b"12345"

    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER",
        4,
    )

    with pytest.raises(ConversionError, match="encoded content container"):
        digital_layout._decoded_content_stream_size(MismatchedStream(), 100)


def test_rejects_content_stream_count_and_separator_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = _page(pdf, b"")
    page.obj["/Contents"] = Array([pdf.make_stream(b" ") for _ in range(3)])
    monkeypatch.setattr(digital_layout, "_MAX_CONTENT_STREAMS_PER_PAGE", 2)

    with pytest.raises(ConversionError, match="content-stream budget exceeded"):
        extract_digital_layout(pdf)

    monkeypatch.setattr(digital_layout, "_MAX_CONTENT_STREAMS_PER_PAGE", 3)
    monkeypatch.setattr(digital_layout, "_MAX_DECODED_CONTENT_BYTES_PER_PAGE", 4)
    with pytest.raises(ConversionError, match="decoded-content byte budget exceeded"):
        extract_digital_layout(pdf)


def test_content_budget_enforces_cumulative_encoded_page_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    first = pdf.make_stream(b"q Q")
    second = pdf.make_stream(b"q Q")
    budget = digital_layout._DecodedContentBudget()
    monkeypatch.setattr(digital_layout, "_MAX_ENCODED_CONTENT_BYTES_PER_PAGE", 5)

    budget.charge_once(first, 0)
    with pytest.raises(ConversionError, match="page encoded-content byte budget"):
        budget.charge_once(second, 0)


def test_content_budget_enforces_cumulative_encoded_document_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    first = pdf.make_stream(b"q Q")
    second = pdf.make_stream(b"q Q")
    budget = digital_layout._DecodedContentBudget()
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_CONTENT_BYTES_PER_DOCUMENT",
        5,
    )

    budget.charge_once(first, 0)
    with pytest.raises(ConversionError, match="document encoded-content byte budget"):
        budget.charge_once(second, 1)


def test_content_budget_enforces_cumulative_decoded_document_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    first = pdf.make_stream(b"q Q")
    second = pdf.make_stream(b"q Q")
    budget = digital_layout._DecodedContentBudget()
    monkeypatch.setattr(
        digital_layout,
        "_MAX_DECODED_CONTENT_BYTES_PER_DOCUMENT",
        5,
    )

    budget.charge_once(first, 0)
    with pytest.raises(ConversionError, match="document decoded-content byte budget"):
        budget.charge_once(second, 1)


def test_content_budget_deduplicates_shared_streams_but_charges_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    shared = pdf.make_stream(b"q Q")
    unique_budget = digital_layout._DecodedContentBudget()
    monkeypatch.setattr(digital_layout, "_MAX_ENCODED_CONTENT_BYTES_PER_PAGE", 3)
    monkeypatch.setattr(
        digital_layout,
        "_MAX_ENCODED_CONTENT_BYTES_PER_DOCUMENT",
        3,
    )

    unique_budget.charge_once(shared, 0)
    unique_budget.charge_once(shared, 0)
    unique_budget.charge_once(shared, 1)

    invocation_budget = digital_layout._DecodedContentBudget()
    invocation_budget.charge(shared, 0)
    with pytest.raises(ConversionError, match="page encoded-content byte budget"):
        invocation_budget.charge(shared, 0)


@pytest.mark.parametrize("method_name", ["charge", "charge_once"])
def test_content_budget_stops_materializing_streams_at_aggregate_limit(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(100, 100))
    page.obj[Name.Contents] = Array([pdf.make_stream(b"q Q") for _ in range(100)])
    observed_streams = 0
    original_sizes = digital_layout._content_stream_sizes

    def count_stream(
        stream: pikepdf.Stream,
        decoded_limit: int,
    ) -> tuple[int, int]:
        nonlocal observed_streams
        observed_streams += 1
        return original_sizes(stream, decoded_limit)

    monkeypatch.setattr(digital_layout, "_content_stream_sizes", count_stream)
    monkeypatch.setattr(digital_layout, "_MAX_ENCODED_CONTENT_BYTES_PER_PAGE", 5)
    budget = digital_layout._DecodedContentBudget()

    with pytest.raises(ConversionError, match="page encoded-content byte budget"):
        getattr(budget, method_name)(page, 0)

    assert observed_streams == 2


def test_empty_form_resources_inherit_for_invocation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pikepdf.Pdf.new()
    xobjects = Dictionary()
    leaf = _form(pdf, b"q Q", Dictionary())
    xobjects["/N0"] = leaf
    for level in range(1, 5):
        form = _form(pdf, f"/N{level - 1} Do /N{level - 1} Do".encode(), Dictionary())
        xobjects[f"/N{level}"] = form
    _page(pdf, b"/N4 Do", Dictionary(XObject=xobjects))
    monkeypatch.setattr(
        digital_layout,
        "_MAX_FORM_INVOCATIONS_PER_RESOURCE_PER_PAGE",
        4,
    )

    with pytest.raises(ConversionError, match="invocation budget exceeded"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize(
    "content",
    [
        b"/PS0 Do q Q",
        b"BT /F1 10 Tf Tj (Visible) Tj ET",
    ],
)
def test_rejects_direct_provenance_that_pdfminer_would_ignore(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    resources = Dictionary(Font=Dictionary(F1=_font(pdf)))
    if content.startswith(b"/PS0"):
        postscript = pdf.make_stream(b"ignored")
        postscript["/Subtype"] = Name("/PS")
        resources["/XObject"] = Dictionary(PS0=postscript)
    _page(pdf, content, resources)

    with pytest.raises(ConversionError, match="provenance|subtype is unsupported"):
        extract_digital_layout(pdf)


def test_propagates_disjoint_clipping_path_uncertainty() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (b"0 0 10 10 re 90 90 10 10 re W n BT /F1 10 Tf 50 50 Td (gap) Tj ET"),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectTextSpan)
    assert span.text == "gap"
    assert span.final_paint_uncertain is True


@pytest.mark.parametrize(
    "content",
    [
        b"0 0 l S",
        b"0 0 1 1 2 2 c S",
        b"0 0 1 1 v S",
        b"0 0 1 1 y S",
        b"h S",
    ],
)
def test_rejects_path_continuation_without_current_subpath(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, content)

    with pytest.raises(ConversionError, match="no current subpath"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize(
    "content",
    [
        b"0 0 m 10 10 l q Q S",
        b"0 0 m 10 10 l 1 0 0 1 5 5 cm S",
        b"0 0 m 10 10 l BT ET S",
        b"0 0 m 10 10 l /Missing Do S",
        b"0 0 m 10 10 l BI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI S",
    ],
)
def test_rejects_interrupted_path_object(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, content)

    with pytest.raises(ConversionError, match="path object is interrupted"):
        extract_digital_layout(pdf)


def test_rejects_unpainted_path_object_at_end_of_stream() -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"0 0 m 10 10 l")

    with pytest.raises(ConversionError, match="path-painting terminator"):
        extract_digital_layout(pdf)


def test_accepts_multiple_subpaths_started_by_moveto_or_rectangle() -> None:
    pdf = pikepdf.Pdf.new()
    _page(pdf, b"0 0 m 10 10 l 20 20 5 5 re S")

    assert extract_digital_layout(pdf)[0].spans == ()


def test_accepts_scoped_nonrectangular_clip_without_semantic_paint() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q 0 0 m 20 0 l 10 20 l h W n 0 0 m 20 20 l S Q "
            b"BT /F1 10 Tf 40 40 Td (Visible) Tj ET"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    spans = extract_digital_layout(pdf)[0].spans

    assert [span.text for span in spans] == ["Visible"]
    assert spans[0].clip_bbox is None


def test_rectangular_clip_does_not_hide_nonrectangular_ancestor() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q 0 0 m 20 0 l 10 20 l h W n "
            b"q 0 0 20 20 re W n "
            b"BT /F1 10 Tf 5 5 Td (Ambiguous) Tj ET Q Q"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectTextSpan)
    assert span.final_paint_uncertain is True


def test_propagates_retraced_degenerate_clip_uncertainty() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (b"0 0 m 0 100 l 0 0 l 100 0 l h W n BT /F1 10 Tf 40 40 Td (Hidden) Tj ET"),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectTextSpan)
    assert span.final_paint_uncertain is True


def test_large_translation_does_not_make_skewed_clip_rectangular() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            b"q 1 0 0 1 1000000000000 1000000000000 cm "
            b"0 0 m 500 2000 l 2000 1500 l 1500 0 l h W n "
            b"BT /F1 10 Tf 50 1850 Td (Outside) Tj ET Q"
        ),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectTextSpan)
    assert span.final_paint_uncertain is True


def test_inverse_matrix_accepts_large_translation() -> None:
    inverse = digital_layout._inverse_matrix((1.0, 0.25, -0.5, 1.0, 1.0e18, -1.0e18))

    assert digital_layout.mult_matrix(
        (1.0, 0.25, -0.5, 1.0, 1.0e18, -1.0e18),
        inverse,
    ) == pytest.approx((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), abs=256.0)


def test_inverse_matrix_accepts_small_nonsingular_linear_part() -> None:
    assert digital_layout._inverse_matrix(
        (1.0e-8, 0.0, 0.0, 1.0e-8, 0.0, 0.0)
    ) == pytest.approx((1.0e8, 0.0, 0.0, 1.0e8, 0.0, 0.0))


@pytest.mark.parametrize(
    "matrix",
    [
        (1.0, 0.0, 0.0, 1.0, float("inf"), 0.0),
        (1.0, 1.0, 1.0, 2.0, 1.0e308, -1.0e308),
    ],
)
def test_inverse_matrix_rejects_nonfinite_input_or_result(
    matrix: digital_layout.Matrix,
) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        digital_layout._inverse_matrix(matrix)


def test_obsolete_fill_operator_ends_path_after_clip() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (b"0 0 20 20 re W F 0 0 10 10 re W n BT /F1 10 Tf 5 5 Td (Visible) Tj ET"),
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert span.text == "Visible"
    assert span.clip_bbox == pytest.approx((0, 0, 10, 10))


@pytest.mark.parametrize(
    "content",
    [
        b"W n",
        b"0 0 10 10 re W",
        b"0 0 10 10 re W BT /F1 10 Tf 2 2 Td (Invalid) Tj ET n",
    ],
)
def test_rejects_unterminated_or_delayed_clipping_path(content: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        content,
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    with pytest.raises(ConversionError, match="clipping"):
        extract_digital_layout(pdf)


@pytest.mark.parametrize(
    "clips",
    [
        b"0 0 0 0 re W n 0 0 m 20 0 l 10 20 l h W n",
        b"0 0 m 20 0 l 10 20 l h W n 0 0 0 0 re W n",
    ],
)
def test_empty_clip_absorbs_nonrectangular_intersection(clips: bytes) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        clips + b" BT /F1 10 Tf 5 5 Td (Hidden) Tj ET",
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert span.clip_bbox == (0.0, 0.0, 0.0, 0.0)


def test_redundant_close_after_rectangle_keeps_exact_clip() -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        b"0 0 20 20 re h W n BT /F1 10 Tf 5 5 Td (Visible) Tj ET",
        Dictionary(Font=Dictionary(F1=_font(pdf))),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert span.clip_bbox == pytest.approx((0, 0, 20, 20))


def test_form_inherits_text_state_without_leaking_between_invocations() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        b"BT 5 5 Td (Inherited) Tj ET",
        Dictionary(ProcSet=Array([Name.PDF, Name.Text])),
    )
    _page(
        pdf,
        (b"BT /F1 10 Tf 3 Tr ET /Fm Do BT 0 Tr ET /Fm Do BT (Page) Tj ET"),
        Dictionary(Font=Dictionary(F1=font), XObject=Dictionary(Fm=form)),
    )

    first, second, page_text = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectXObjectSpan)
    assert isinstance(second, DirectXObjectSpan)
    assert isinstance(page_text, DirectTextSpan)
    assert first.text_runs[0].font_name == "Helvetica"
    assert first.text_runs[0].font_size == pytest.approx(10)
    assert first.text_runs[0].render_mode == 3
    assert first.text_runs[0].invisible is True
    assert second.text_runs[0].render_mode == 0
    assert second.text_runs[0].invisible is False
    assert page_text.render_mode == 0


def test_form_records_inherited_line_state_per_invocation() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, b"q Q", Dictionary())
    _page(
        pdf,
        (b"23 w 2 J 1 j 7 M [3 4] 2 d /Fm Do q 5 w 0 J 2 j 11 M [] 0 d /Fm Do Q"),
        Dictionary(XObject=Dictionary(Fm=form)),
    )

    first, second = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectXObjectSpan)
    assert isinstance(second, DirectXObjectSpan)
    assert first.entry_state == digital_layout.InvocationPaintState(
        line_width=23,
        line_cap=2,
        line_join=1,
        miter_limit=7,
        dash_array=(3, 4),
        dash_phase=2,
    )
    assert second.entry_state == digital_layout.InvocationPaintState(
        line_width=5,
        line_cap=0,
        line_join=2,
        miter_limit=11,
    )


def test_form_records_inherited_extgstate_alpha_per_invocation() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, b"q Q", Dictionary())
    _page(
        pdf,
        b"q /Transparent gs /Fm Do Q /Fm Do",
        Dictionary(
            ExtGState=Dictionary(Transparent=Dictionary(CA=0, ca=0)),
            XObject=Dictionary(Fm=form),
        ),
    )

    first, second = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectXObjectSpan)
    assert isinstance(second, DirectXObjectSpan)
    assert (first.entry_state.stroke_alpha, first.entry_state.fill_alpha) == (0, 0)
    assert (second.entry_state.stroke_alpha, second.entry_state.fill_alpha) == (1, 1)


def test_soft_mask_uncertainty_is_propagated_for_text_and_image() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    mask = _form(pdf, b"0 0 10 10 re f", Dictionary())
    image = _image(pdf, b"\x80")
    _page(
        pdf,
        (
            b"q /Masked gs BT /F1 10 Tf 10 20 Td (Masked) Tj ET /Im Do Q "
            b"BT /F1 10 Tf 10 40 Td (Visible) Tj ET"
        ),
        Dictionary(
            ExtGState=Dictionary(
                Masked=Dictionary(SMask=Dictionary(S=Name.Luminosity, G=mask))
            ),
            Font=Dictionary(F1=font),
            XObject=Dictionary(Im=image),
        ),
    )

    masked_text, masked_image, visible_text = extract_digital_layout(pdf)[0].spans

    assert isinstance(masked_text, DirectTextSpan)
    assert masked_text.final_paint_uncertain is True
    assert isinstance(masked_image, DirectXObjectSpan)
    assert masked_image.final_paint_uncertain is True
    assert masked_image.intrinsic_visibility_uncertain is False
    assert isinstance(visible_text, DirectTextSpan)
    assert visible_text.final_paint_uncertain is False


@pytest.mark.parametrize(
    ("color", "graphics_state", "render_mode", "uncertain"),
    [
        (b"0 0 0 1 k", Dictionary(op=True, OPM=1), 0, False),
        (b"/DeviceCMYK cs 0 0 0 1 sc", Dictionary(op=True, OPM=1), 0, False),
        (b"0 0 0 0 k", Dictionary(op=True, OPM=1), 0, True),
        (b"0 g", Dictionary(op=True, OPM=1), 0, True),
        (b"0 0 0 1 K", Dictionary(OP=True, OPM=1), 1, False),
        (b"0 0 0 0 K", Dictionary(OP=True, OPM=1), 1, True),
    ],
)
def test_process_cmyk_ink_proves_overprinted_text_visibility(
    color: bytes,
    graphics_state: Dictionary,
    render_mode: int,
    uncertain: bool,
) -> None:
    pdf = pikepdf.Pdf.new()
    _page(
        pdf,
        (
            color
            + b" /GS gs BT /F1 10 Tf "
            + str(render_mode).encode()
            + b" Tr 10 20 Td (Text) Tj ET"
        ),
        Dictionary(
            ExtGState=Dictionary(GS=graphics_state),
            Font=Dictionary(F1=_font(pdf)),
        ),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectTextSpan)
    assert span.final_paint_uncertain is uncertain


def test_device_dependent_stroked_text_uncertainty_is_scoped() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    _page(
        pdf,
        (
            b"q 0 w BT /F1 10 Tf 1 Tr 10 20 Td (Hairline) Tj ET Q "
            b"q /Adjusted gs BT /F1 10 Tf 1 Tr 10 40 Td (Adjusted) Tj ET Q "
            b"BT /F1 10 Tf 1 Tr 10 60 Td (Stable) Tj ET"
        ),
        Dictionary(
            ExtGState=Dictionary(Adjusted=Dictionary(SA=True)),
            Font=Dictionary(F1=font),
        ),
    )

    hairline, adjusted, stable = extract_digital_layout(pdf)[0].spans

    assert isinstance(hairline, DirectTextSpan)
    assert hairline.final_paint_uncertain is True
    assert isinstance(adjusted, DirectTextSpan)
    assert adjusted.final_paint_uncertain is True
    assert isinstance(stable, DirectTextSpan)
    assert stable.final_paint_uncertain is False


def test_explicit_parallelogram_clip_is_exact_for_image() -> None:
    pdf = pikepdf.Pdf.new()
    image = _image(pdf, b"\xff\x00\x00", color_space=Name.DeviceRGB)
    _page(
        pdf,
        (
            b"q 15 65 m 115 50 l 110 15 l 10 30 l 15 65 l h W* n "
            b"100 -15 5 35 10 30 cm /Im Do Q"
        ),
        Dictionary(XObject=Dictionary(Im=image)),
    )

    span = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(span, DirectXObjectSpan)
    assert span.kind == "image"
    assert span.clip_polygon is not None
    assert len(span.clip_polygon) == 4
    assert span.clip_bbox == pytest.approx((10, 15, 115, 65))
    assert span.final_paint_uncertain is False


def test_form_records_and_restores_complex_final_paint_state() -> None:
    pdf = pikepdf.Pdf.new()
    form = _form(pdf, b"q Q", Dictionary())
    pattern = pdf.make_stream(b"")
    pattern["/Type"] = Name.Pattern
    pattern["/PatternType"] = 1
    pattern["/PaintType"] = 1
    pattern["/TilingType"] = 1
    pattern["/BBox"] = Array([0, 0, 10, 10])
    pattern["/XStep"] = 10
    pattern["/YStep"] = 10
    pattern["/Resources"] = Dictionary()
    _page(
        pdf,
        (b"q /Complex gs BT 7 Tr ET /Pattern cs /P0 scn /Fm Do Q /Fm Do"),
        Dictionary(
            ExtGState=Dictionary(
                Complex=Dictionary(SA=True, OP=True, OPM=1, BM=Name.Multiply)
            ),
            Pattern=Dictionary(P0=pattern),
            XObject=Dictionary(Fm=form),
        ),
    )

    first, second = extract_digital_layout(pdf)[0].spans

    assert isinstance(first, DirectXObjectSpan)
    assert isinstance(second, DirectXObjectSpan)
    assert first.entry_state.stroke_adjust is True
    assert first.entry_state.stroke_overprint is True
    assert first.entry_state.fill_overprint is True
    assert first.entry_state.overprint_mode == 1
    assert first.entry_state.fill_color_complex is True
    assert first.entry_state.blend_mode_complex is True
    assert first.entry_state.text_render_mode == 7
    assert second.entry_state == digital_layout.InvocationPaintState()


def test_sheared_form_bbox_keeps_exact_polygon_clip() -> None:
    pdf = pikepdf.Pdf.new()
    font = _font(pdf)
    form = _form(
        pdf,
        b"BT /F1 5 Tf -7 8 Td (I) Tj ET",
        Dictionary(Font=Dictionary(F1=font)),
        bbox=(0, 0, 20, 10),
    )
    form["/Matrix"] = Array([1, 0, 1, 1, 0, 0])
    _page(
        pdf,
        b"/Fm Do",
        Dictionary(XObject=Dictionary(Fm=form)),
    )

    invocation = extract_digital_layout(pdf)[0].spans[0]

    assert isinstance(invocation, DirectXObjectSpan)
    assert invocation.clip_polygon is not None
    assert len(invocation.clip_polygon) == 4
    bbox_corners = {
        (invocation.bbox[0], invocation.bbox[1]),
        (invocation.bbox[0], invocation.bbox[3]),
        (invocation.bbox[2], invocation.bbox[1]),
        (invocation.bbox[2], invocation.bbox[3]),
    }
    assert set(invocation.clip_polygon) != bbox_corners
    child = invocation.children[0]
    assert isinstance(child, DirectTextSpan)
    assert digital_layout._clip_bbox_to_polygon(child.bbox, child.clip_polygon) is None
