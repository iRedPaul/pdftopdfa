# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for color space management and ICC profiles."""

from decimal import Decimal
from io import BytesIO

import pikepdf
import pytest
from conftest import new_pdf, pdf_object
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import (
    ColorSpaceAnalysis,
    ColorSpaceType,
    _analyze_colorspace,
    _apply_default_colorspaces,
    _apply_defaults_to_ap_entry,
    _convert_calibrated_colorspaces,
    _create_icc_colorspace,
    _parse_colorspace_array,
    _validate_icc_profile,
    create_output_intent_for_colorspace,
    detect_color_spaces,
    embed_color_profiles,
    get_cmyk_profile,
    get_gray_profile,
    get_profile_for_colorspace,
    get_srgb_profile,
    has_output_intent,
)
from pdftopdfa.color_profile._output_intents import (
    _ICC_TAG_TYPES,
    _has_usable_tag_table,
)
from pdftopdfa.exceptions import ConversionError


class TestColorSpaceAnalysis:
    """Tests for the ColorSpaceAnalysis class."""

    def test_empty_analysis(self):
        """Empty analysis has no detected color spaces."""
        analysis = ColorSpaceAnalysis()
        assert analysis.detected_spaces == set()

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (dict(device_gray_used=True), {ColorSpaceType.DEVICE_GRAY}),
            (dict(device_rgb_used=True), {ColorSpaceType.DEVICE_RGB}),
            (dict(device_cmyk_used=True), {ColorSpaceType.DEVICE_CMYK}),
        ],
        ids=["gray", "rgb", "cmyk"],
    )
    def test_single_colorspace(self, kwargs, expected):
        """Individual device color space detected."""
        analysis = ColorSpaceAnalysis(**kwargs)
        assert analysis.detected_spaces == expected

    def test_multiple_colorspaces(self):
        """Multiple color spaces detected."""
        analysis = ColorSpaceAnalysis(
            device_gray_used=True,
            device_rgb_used=True,
            device_cmyk_used=True,
        )
        assert analysis.detected_spaces == {
            ColorSpaceType.DEVICE_GRAY,
            ColorSpaceType.DEVICE_RGB,
            ColorSpaceType.DEVICE_CMYK,
        }


class TestICCProfileLoading:
    """Tests for loading ICC profiles."""

    @pytest.mark.parametrize(
        ("loader",),
        [(get_srgb_profile,), (get_gray_profile,), (get_cmyk_profile,)],
        ids=["sRGB", "Gray", "CMYK"],
    )
    def test_profile_valid(self, loader):
        """ICC profile can be loaded and is valid."""
        profile = loader()
        assert isinstance(profile, bytes)
        assert len(profile) > 0
        assert _validate_icc_profile(profile)

    @pytest.mark.parametrize(
        ("cs_type", "loader"),
        [
            (ColorSpaceType.DEVICE_GRAY, get_gray_profile),
            (ColorSpaceType.DEVICE_RGB, get_srgb_profile),
            (ColorSpaceType.DEVICE_CMYK, get_cmyk_profile),
        ],
        ids=["Gray", "RGB", "CMYK"],
    )
    def test_get_profile_for_colorspace(self, cs_type, loader):
        """get_profile_for_colorspace returns correct profile."""
        assert get_profile_for_colorspace(cs_type) == loader()


def _icc_tag_data(tags: tuple[bytes, ...]) -> bytes:
    """Build an ICC tag table followed by minimal elements of a permitted type."""
    data_start = 132 + 12 * len(tags)
    table = len(tags).to_bytes(4, "big") + b"".join(
        tag + (data_start + 8 * index).to_bytes(4, "big") + (8).to_bytes(4, "big")
        for index, tag in enumerate(tags)
    )
    return table + b"".join(_ICC_TAG_TYPES[tag][0] + bytes(4) for tag in tags)


def _make_icc_profile(**overrides: bytes | int) -> bytes:
    """Build a minimal 128-byte ICC profile with optional field overrides.

    Defaults to a valid v2 mntr RGB profile.  Pass keyword arguments to
    override specific header fields:
        version (int): major version byte at offset 8
        device_class (bytes): 4-byte class at offset 12-15
        colorspace (bytes): 4-byte color space signature at offset 16-19
    """
    profile = bytearray(128) + _icc_tag_data((b"desc", b"cprt", b"wtpt", b"A2B0"))
    profile[0:4] = len(profile).to_bytes(4, "big")
    profile[8] = overrides.get("version", 2)
    profile[12:16] = overrides.get("device_class", b"mntr")
    profile[16:20] = overrides.get("colorspace", b"RGB ")
    profile[36:40] = b"acsp"
    return bytes(profile)


class TestICCProfileValidation:
    """Tests for ICC profile structural validation."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [(2, True), (4, True), (0, False), (3, False), (5, False)],
        ids=["v2-valid", "v4-valid", "v0-reject", "v3-reject", "v5-reject"],
    )
    def test_version_validation(self, version, expected):
        """Validates ICC profile version numbers."""
        assert _validate_icc_profile(_make_icc_profile(version=version)) is expected

    @pytest.mark.parametrize(
        ("device_class", "expected"),
        [
            (b"mntr", True),
            (b"prtr", True),
            (b"scnr", True),
            (b"spac", True),
            (b"nmcl", False),
            (b"link", False),
            (b"abst", False),
        ],
        ids=[
            "mntr",
            "prtr",
            "scnr",
            "spac",
            "nmcl-reject",
            "link-reject",
            "abst-reject",
        ],
    )
    def test_device_class_validation(self, device_class, expected):
        """Validates ICC profile device classes."""
        profile = _make_icc_profile(device_class=device_class)
        assert _validate_icc_profile(profile) is expected

    def test_rejects_too_short(self):
        """Profile shorter than 128 bytes is rejected."""
        assert _validate_icc_profile(b"\x00" * 64) is False

    def test_rejects_bad_signature(self):
        """Profile without 'acsp' signature is rejected."""
        profile = bytearray(_make_icc_profile())
        profile[36:40] = b"\x00\x00\x00\x00"
        assert _validate_icc_profile(bytes(profile)) is False

    def test_rejects_size_mismatch(self):
        """Profile with mismatched declared size is rejected."""
        profile = bytearray(_make_icc_profile())
        profile[0:4] = (256).to_bytes(4, "big")  # declared 256 but only 128
        assert _validate_icc_profile(bytes(profile)) is False


class TestColorSpaceDetection:
    """Tests for color space detection."""

    @pytest.fixture
    def pdf_with_gray_image(self) -> Pdf:
        """PDF with DeviceGray image."""
        pdf = new_pdf()

        image_data = b"\x80"
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceGray
        image_stream[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream)),
        )
        content_stream = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_rgb_image(self) -> Pdf:
        """PDF with DeviceRGB image."""
        pdf = new_pdf()

        image_data = b"\xff\x00\x00"  # Red pixel
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceRGB
        image_stream[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream)),
        )
        content_stream = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_cmyk_image(self) -> Pdf:
        """PDF with DeviceCMYK image."""
        pdf = new_pdf()

        image_data = b"\x00\xff\x00\x00"  # Magenta pixel
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceCMYK
        image_stream[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream)),
        )
        content_stream = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_multiple_colorspaces(self) -> Pdf:
        """PDF with multiple color spaces."""
        pdf = new_pdf()

        # Gray image
        gray_image = pdf.make_stream(b"\x80")
        gray_image[Name.Type] = Name.XObject
        gray_image[Name.Subtype] = Name.Image
        gray_image[Name.Width] = 1
        gray_image[Name.Height] = 1
        gray_image[Name.ColorSpace] = Name.DeviceGray
        gray_image[Name.BitsPerComponent] = 8

        # RGB image
        rgb_image = pdf.make_stream(b"\xff\x00\x00")
        rgb_image[Name.Type] = Name.XObject
        rgb_image[Name.Subtype] = Name.Image
        rgb_image[Name.Width] = 1
        rgb_image[Name.Height] = 1
        rgb_image[Name.ColorSpace] = Name.DeviceRGB
        rgb_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                XObject=Dictionary(
                    Im0=gray_image,
                    Im1=rgb_image,
                )
            ),
        )
        content_stream = pdf.make_stream(
            b"q 100 0 0 100 100 600 cm /Im0 Do Q q 100 0 0 100 200 600 cm /Im1 Do Q"
        )
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    def test_detect_device_gray_in_image(self, pdf_with_gray_image: Pdf):
        """Detects DeviceGray in images."""
        analysis = detect_color_spaces(pdf_with_gray_image)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_rgb_in_image(self, pdf_with_rgb_image: Pdf):
        """Detects DeviceRGB in images."""
        analysis = detect_color_spaces(pdf_with_rgb_image)
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is True
        assert analysis.device_cmyk_used is False

    def test_detect_device_cmyk_in_image(self, pdf_with_cmyk_image: Pdf):
        """Detects DeviceCMYK in images."""
        analysis = detect_color_spaces(pdf_with_cmyk_image)
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is True

    def test_detect_multiple_colorspaces(self, pdf_with_multiple_colorspaces: Pdf):
        """Detects multiple color spaces in the same document."""
        analysis = detect_color_spaces(pdf_with_multiple_colorspaces)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is True
        assert analysis.device_cmyk_used is False

    def test_detects_color_through_1200_nested_forms(self):
        """Deep Form resource graphs are traversed without Python recursion."""
        pdf = new_pdf()
        nested = pdf.make_stream(b"1 0 0 rg 0 0 1 1 re f")
        nested[Name.Type] = Name.XObject
        nested[Name.Subtype] = Name.Form
        nested[Name.BBox] = Array([0, 0, 1, 1])
        nested[Name.Resources] = Dictionary()

        for _ in range(1200):
            parent = pdf.make_stream(b"/Nested Do")
            parent[Name.Type] = Name.XObject
            parent[Name.Subtype] = Name.Form
            parent[Name.BBox] = Array([0, 0, 1, 1])
            parent[Name.Resources] = Dictionary(XObject=Dictionary(Nested=nested))
            nested = parent

        page = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 1, 1]),
            Resources=Dictionary(XObject=Dictionary(Nested=nested)),
            Contents=pdf.make_stream(b"/Nested Do"),
        )
        pdf.pages.append(pikepdf.Page(page))

        analysis = detect_color_spaces(pdf)

        assert analysis.device_rgb_used is True

    def test_detect_empty_pdf(self, sample_pdf_obj: Pdf):
        """Empty PDF has no detected color spaces."""
        analysis = detect_color_spaces(sample_pdf_obj)
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_rgb_in_inline_image(self):
        """Detects DeviceRGB from inline image /CS entry."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_gray_in_inline_image(self):
        """Detects DeviceGray from inline image /CS entry."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_cmyk_in_inline_image(self):
        """Detects DeviceCMYK from inline image /CS entry."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(
            b"BI /W 1 /H 1 /CS /CMYK /BPC 8 ID \x00\xff\x00\x00 EI"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False

    def test_detect_indexed_device_rgb_in_inline_image(self):
        """Detects DeviceRGB from Indexed inline image base color space."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(
            b"BI /W 2 /H 1 /CS [/Indexed /RGB 1 <FF000000FF00>] /BPC 8 ID \x00\x01 EI"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_indexed_abbreviated_base_in_inline_image(self):
        """Detects DeviceGray from Indexed inline image with abbreviated /I /G."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(
            b"BI /W 1 /H 1 /CS [/I /G 1 <FF00>] /BPC 8 ID \x00 EI"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_inline_image_without_cs_entry(self):
        """Inline image without /CS does not crash and detects no device CS."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"BI /W 1 /H 1 /BPC 8 ID \x80 EI")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_inline_image_with_operators(self):
        """Detects both CMYK operator and inline RGB image."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(
            b"0 1 0 0 k 100 100 200 300 re f "
            b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True
        assert analysis.device_rgb_used is True


class TestOutputIntentCreation:
    """Tests for OutputIntent creation."""

    def test_create_output_intent_for_gray(self, sample_pdf_obj: Pdf):
        """Creates correct OutputIntent for Gray."""
        profile = get_gray_profile()
        intent = create_output_intent_for_colorspace(
            sample_pdf_obj, ColorSpaceType.DEVICE_GRAY, profile
        )

        assert intent["/Type"] == Name.OutputIntent
        assert intent["/S"] == Name.GTS_PDFA1
        assert str(intent["/OutputConditionIdentifier"]) == "sGray"
        assert intent["/DestOutputProfile"]["/N"] == 1

    def test_create_output_intent_for_rgb(self, sample_pdf_obj: Pdf):
        """Creates correct OutputIntent for RGB."""
        profile = get_srgb_profile()
        intent = create_output_intent_for_colorspace(
            sample_pdf_obj, ColorSpaceType.DEVICE_RGB, profile
        )

        assert intent["/Type"] == Name.OutputIntent
        assert intent["/S"] == Name.GTS_PDFA1
        assert str(intent["/OutputConditionIdentifier"]) == "sRGB"
        assert intent["/DestOutputProfile"]["/N"] == 3

    def test_create_output_intent_for_cmyk(self, sample_pdf_obj: Pdf):
        """Creates correct OutputIntent for CMYK."""
        profile = get_cmyk_profile()
        intent = create_output_intent_for_colorspace(
            sample_pdf_obj, ColorSpaceType.DEVICE_CMYK, profile
        )

        assert intent["/Type"] == Name.OutputIntent
        assert intent["/S"] == Name.GTS_PDFA1
        assert str(intent["/OutputConditionIdentifier"]) == "FOGRA39"
        assert intent["/DestOutputProfile"]["/N"] == 4


class TestContentStreamColorDetection:
    """Tests for color detection in content streams."""

    @pytest.fixture
    def pdf_with_rgb_fill(self) -> Pdf:
        """PDF with RGB fill color in content stream."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        # Content stream: red fill, draw rectangle
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_gray_stroke(self) -> Pdf:
        """PDF with Gray stroke color in content stream."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"0.5 G 100 100 m 200 200 l S")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_cmyk_fill(self) -> Pdf:
        """PDF with CMYK fill color in content stream."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    @pytest.fixture
    def pdf_with_cs_operator(self) -> Pdf:
        """PDF using cs operator to set DeviceRGB color space."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"/DeviceRGB cs 1 0 0 sc 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    def test_detect_rgb_in_content_stream(self, pdf_with_rgb_fill: Pdf):
        """Detects DeviceRGB from rg operator."""
        analysis = detect_color_spaces(pdf_with_rgb_fill)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is False

    def test_detect_gray_in_content_stream(self, pdf_with_gray_stroke: Pdf):
        """Detects DeviceGray from G operator."""
        analysis = detect_color_spaces(pdf_with_gray_stroke)
        assert analysis.device_gray_used is True

    def test_detect_cmyk_in_content_stream(self, pdf_with_cmyk_fill: Pdf):
        """Detects DeviceCMYK from k operator."""
        analysis = detect_color_spaces(pdf_with_cmyk_fill)
        assert analysis.device_cmyk_used is True

    def test_detect_rgb_from_cs_operator(self, pdf_with_cs_operator: Pdf):
        """Detects DeviceRGB from cs operator."""
        analysis = detect_color_spaces(pdf_with_cs_operator)
        assert analysis.device_rgb_used is True


class TestEmbedColorProfiles:
    """Tests for embed_color_profiles."""

    @pytest.fixture
    def pdf_with_gray_image(self) -> Pdf:
        """PDF with DeviceGray image."""
        pdf = new_pdf()

        image_data = b"\x80"
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = Name.DeviceGray
        image_stream[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image_stream)),
        )
        content_stream = pdf.make_stream(b"q 100 0 0 100 100 600 cm /Im0 Do Q")
        page_dict[Name.Contents] = content_stream

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        return pdf

    def test_embed_default_srgb_when_no_colorspace_detected(self, sample_pdf_obj: Pdf):
        """Embeds sRGB when no color spaces are detected."""
        embedded = embed_color_profiles(sample_pdf_obj, "2b")

        assert ColorSpaceType.DEVICE_RGB in embedded
        assert has_output_intent(sample_pdf_obj)

    def test_embed_gray_profile(self, pdf_with_gray_image: Pdf):
        """Embeds Gray profile for DeviceGray images."""
        embedded = embed_color_profiles(pdf_with_gray_image, "2b")

        assert ColorSpaceType.DEVICE_GRAY in embedded
        assert has_output_intent(pdf_with_gray_image)

    def test_embed_multiple_profiles(self) -> None:
        """Embeds single dominant profile when multiple color spaces detected.

        PDF/A allows only a single OutputIntent with S=GTS_PDFA1.
        Priority: CMYK > RGB > Gray.
        Non-dominant RGB/CMYK spaces get Default entries.
        """
        pdf = new_pdf()

        # Gray image
        gray_image = pdf.make_stream(b"\x80")
        gray_image[Name.Type] = Name.XObject
        gray_image[Name.Subtype] = Name.Image
        gray_image[Name.Width] = 1
        gray_image[Name.Height] = 1
        gray_image[Name.ColorSpace] = Name.DeviceGray
        gray_image[Name.BitsPerComponent] = 8

        # RGB image
        rgb_image = pdf.make_stream(b"\xff\x00\x00")
        rgb_image[Name.Type] = Name.XObject
        rgb_image[Name.Subtype] = Name.Image
        rgb_image[Name.Width] = 1
        rgb_image[Name.Height] = 1
        rgb_image[Name.ColorSpace] = Name.DeviceRGB
        rgb_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=gray_image, Im1=rgb_image)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedded = embed_color_profiles(pdf, "2b")

        # Returns all detected color spaces
        assert ColorSpaceType.DEVICE_GRAY in embedded
        assert ColorSpaceType.DEVICE_RGB in embedded
        assert len(embedded) == 2

        # Only one OutputIntent embedded (PDF/A compliance)
        # Dominant color space is RGB (Gray + RGB -> RGB wins)
        output_intents = pdf.Root.OutputIntents
        assert len(output_intents) == 1
        assert output_intents[0]["/DestOutputProfile"]["/N"] == 3  # RGB

        # DeviceGray is covered by the presence of the OutputIntent.
        page_res = pdf.pages[0]["/Resources"]
        assert Name.ColorSpace not in page_res

        # Device names stay context-dependent instead of overriding Defaults.
        gray_img = page_res["/XObject"]["/Im0"]
        assert gray_img["/ColorSpace"] == Name.DeviceGray

        # RGB image untouched (dominant space)
        rgb_img = page_res["/XObject"]["/Im1"]
        assert rgb_img["/ColorSpace"] == Name.DeviceRGB

    def test_invalid_level_raises_error(self, sample_pdf_obj: Pdf):
        """Invalid level raises ConversionError."""
        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            embed_color_profiles(sample_pdf_obj, "invalid")

    def test_replace_existing_output_intents(self, sample_pdf_obj: Pdf):
        """Replaces existing OutputIntents."""
        # First embedding
        embed_color_profiles(sample_pdf_obj, "2b")

        # Second embedding with replace_existing=True
        embed_color_profiles(sample_pdf_obj, "2b", replace_existing=True)

        # OutputIntents should have been replaced
        assert has_output_intent(sample_pdf_obj)

    def test_skip_when_not_replacing(self, sample_pdf_obj: Pdf):
        """Skips when replace_existing=False and OutputIntents exist."""
        # First embedding
        embed_color_profiles(sample_pdf_obj, "2b")

        # Second embedding with replace_existing=False
        result = embed_color_profiles(sample_pdf_obj, "2b", replace_existing=False)

        assert result == []

    def test_skip_does_not_validate_calibration(self) -> None:
        """The do-nothing mode only rejects calibration once it would edit the PDF."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        embed_color_profiles(pdf, "2b")
        page.Resources.ColorSpace = Dictionary(Bad=Array([Name.CalRGB, Dictionary()]))
        intents = pdf.Root.OutputIntents.unparse()

        assert embed_color_profiles(pdf, "2b", replace_existing=False) == []
        assert pdf.Root.OutputIntents.unparse() == intents

        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf, "2b", replace_existing=True)

    @pytest.mark.parametrize("shared", [True, False])
    def test_redundant_pdfa_intents_collapse_to_the_first(self, shared) -> None:
        """Identical profiles keep one PDF/A intent and every other subtype."""
        pdf = new_pdf()
        pdf.add_blank_page(page_size=(612, 792))

        def profile():
            stream = pdf.make_stream(_make_icc_profile())
            stream[Name.N] = 3
            return pdf.make_indirect(stream)

        def intent(subtype, identifier, destination):
            return pdf.make_indirect(
                Dictionary(
                    Type=Name.OutputIntent,
                    S=subtype,
                    OutputConditionIdentifier=pikepdf.String(identifier),
                    DestOutputProfile=destination,
                )
            )

        first = profile()
        pdf.Root.OutputIntents = Array(
            [
                intent(Name.GTS_PDFX, "Press", first),
                intent(Name.GTS_PDFA1, "First", first),
                intent(Name.GTS_PDFA1, "Second", first if shared else profile()),
                intent(Name.GTS_PDFA1, "Third", first if shared else profile()),
            ]
        )

        embed_color_profiles(pdf, "2b", replace_existing=True)

        intents = pdf.Root.OutputIntents
        assert [
            (str(item.S), str(item.OutputConditionIdentifier)) for item in intents
        ] == [("/GTS_PDFX", "Press"), ("/GTS_PDFA1", "First")]
        assert len({item.DestOutputProfile.objgen for item in intents}) == 1

    def test_valid_existing_srgb_output_intent_is_preserved(self) -> None:
        """Valid existing sRGB PDF/A OutputIntent is not replaced."""
        pdf = new_pdf()
        profile_data = _make_icc_profile()
        stream = pdf.make_stream(profile_data)
        stream[Name.N] = 3
        intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("Custom sRGB"),
            DestOutputProfile=stream,
        )
        pdf.Root.OutputIntents = Array([intent])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        preserved = pdf.Root.OutputIntents[0]
        assert str(preserved["/OutputConditionIdentifier"]) == "Custom sRGB"
        assert bytes(preserved["/DestOutputProfile"].read_bytes()) == profile_data
        assert preserved["/DestOutputProfile"]["/N"] == 3

    def test_valid_existing_cmyk_output_intent_is_preserved(self) -> None:
        """Valid existing CMYK PDF/A OutputIntent is not replaced."""
        pdf = new_pdf()
        profile_data = _make_icc_profile(colorspace=b"CMYK")
        stream = pdf.make_stream(profile_data)
        stream[Name.N] = 4
        intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("Custom CMYK"),
            DestOutputProfile=stream,
        )
        pdf.Root.OutputIntents = Array([intent])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"0 1 0 0 k")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        preserved = pdf.Root.OutputIntents[0]
        assert str(preserved["/OutputConditionIdentifier"]) == "Custom CMYK"
        assert bytes(preserved["/DestOutputProfile"].read_bytes()) == profile_data
        assert preserved["/DestOutputProfile"]["/N"] == 4

    @pytest.mark.parametrize("with_profile", [False, True])
    def test_undefined_intent_subtype_is_removed_alone(self, with_profile) -> None:
        """Other output conditions stay, and its own profile causes no conflict."""
        pdf = new_pdf()
        stream = pdf.make_stream(_make_icc_profile())
        stream[Name.N] = 3
        other = bytearray(_make_icc_profile())
        other[50:54] = b"diff"
        undefined = Dictionary(
            Type=Name.OutputIntent,
            S=Name.Bogus,
            OutputConditionIdentifier=pikepdf.String("Undefined"),
        )
        if with_profile:
            undefined.DestOutputProfile = pdf.make_stream(bytes(other))
        pdf.Root.OutputIntents = Array(
            [
                undefined,
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Keep"),
                    DestOutputProfile=stream,
                ),
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert [str(i.OutputConditionIdentifier) for i in pdf.Root.OutputIntents] == [
            "Keep"
        ]

    def test_only_undefined_intent_subtype_is_replaced(self) -> None:
        """Without any defined condition, the document receives its PDF/A intent."""
        pdf = new_pdf()
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.Bogus,
                    OutputConditionIdentifier=pikepdf.String("Undefined"),
                )
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert [str(i.S) for i in pdf.Root.OutputIntents] == ["/GTS_PDFA1"]
        assert "Undefined" not in [
            str(i.OutputConditionIdentifier) for i in pdf.Root.OutputIntents
        ]

    @pytest.mark.parametrize("load", [get_srgb_profile, get_cmyk_profile])
    def test_bundled_profiles_have_usable_tag_tables(self, load) -> None:
        """Real profiles satisfy the checks made for existing output intents."""
        assert _has_usable_tag_table(load())

    @pytest.mark.parametrize(
        "damage",
        [
            "header_only",
            "count_exceeds_data",
            "offset_in_table",
            "data_past_end",
            "short_tag",
            "missing_description",
            "no_color_model",
            "wrong_element_type",
            "empty_element_type",
        ],
    )
    def test_existing_output_intent_needs_a_usable_tag_table(self, damage) -> None:
        """A plausible header alone cannot retain an existing output condition."""
        profile = bytearray(_make_icc_profile())
        if damage == "header_only":
            profile = profile[:128]
        elif damage == "count_exceeds_data":
            profile[128:132] = (1000).to_bytes(4, "big")
        elif damage == "offset_in_table":
            profile[136:140] = (4).to_bytes(4, "big")
        elif damage == "data_past_end":
            profile[140:144] = (1000).to_bytes(4, "big")
        elif damage == "short_tag":
            profile[140:144] = (4).to_bytes(4, "big")
        elif damage == "missing_description":
            profile[132:136] = b"none"
        elif damage == "no_color_model":
            profile[168:172] = b"none"
        elif damage == "wrong_element_type":
            profile[180:184] = b"curv"
        else:
            profile[180:188] = bytes(8)
        profile[0:4] = len(profile).to_bytes(4, "big")
        pdf = new_pdf()
        stream = pdf.make_stream(bytes(profile))
        stream[Name.N] = 3
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Damaged"),
                    DestOutputProfile=stream,
                )
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        replacement = pdf.Root.OutputIntents[0]
        assert str(replacement.OutputConditionIdentifier) != "Damaged"
        assert replacement.DestOutputProfile.read_bytes() != bytes(profile)

    @pytest.mark.parametrize(
        "colorspace,tags,usable",
        [
            (b"GRAY", (b"kTRC",), True),
            (b"GRAY", (), False),
            (
                b"RGB ",
                (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC"),
                True,
            ),
            (b"RGB ", (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC"), False),
            (b"CMYK", (b"A2B0",), True),
            (b"CMYK", (b"kTRC",), False),
        ],
    )
    def test_output_profile_needs_its_color_model_tags(
        self, colorspace, tags, usable
    ) -> None:
        """A matrix or tone-curve model is an alternative to a lookup table only."""
        profile = bytearray(128) + _icc_tag_data((b"desc", b"cprt", b"wtpt", *tags))
        profile[16:20] = colorspace
        assert _has_usable_tag_table(bytes(profile)) is usable

    @pytest.mark.parametrize(
        "value", [Decimal("3.0"), Decimal("3.5"), True, Name.Three, 3]
    )
    def test_existing_output_intent_n_must_be_an_integer(self, value) -> None:
        """A real /N is rewritten instead of being truncated and retained."""
        pdf = new_pdf()
        stream = pdf.make_stream(_make_icc_profile())
        stream[Name.N] = value
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Keep"),
                    DestOutputProfile=stream,
                )
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        preserved = pdf.Root.OutputIntents[0]
        assert str(preserved.OutputConditionIdentifier) == "Keep"
        components = preserved.DestOutputProfile.N
        assert isinstance(components, int) and not isinstance(components, bool)
        assert components == 3

    def test_existing_output_intent_n_is_repaired(self) -> None:
        """Missing or wrong /N is repaired from the ICC color-space signature."""
        pdf = new_pdf()
        profile_data = _make_icc_profile(colorspace=b"GRAY")
        stream = pdf.make_stream(profile_data)
        stream[Name.N] = 3
        intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("Custom Gray"),
            DestOutputProfile=stream,
        )
        pdf.Root.OutputIntents = Array([intent])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"0.5 g")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        preserved = pdf.Root.OutputIntents[0]
        assert bytes(preserved["/DestOutputProfile"].read_bytes()) == profile_data
        assert preserved["/DestOutputProfile"]["/N"] == 1

    def test_broken_existing_output_intent_is_replaced(self) -> None:
        """Broken ICC stream falls back to generated dominant OutputIntent."""
        pdf = new_pdf()
        broken_profile = bytearray(_make_icc_profile())
        broken_profile[36:40] = b"bad!"
        stream = pdf.make_stream(bytes(broken_profile))
        stream[Name.N] = 3
        intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("Broken"),
            DestOutputProfile=stream,
        )
        pdf.Root.OutputIntents = Array([intent])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        replacement = pdf.Root.OutputIntents[0]
        replacement_profile = bytes(replacement["/DestOutputProfile"].read_bytes())
        assert len(pdf.Root.OutputIntents) == 1
        assert str(replacement["/OutputConditionIdentifier"]) == "sRGB"
        assert replacement_profile != bytes(broken_profile)
        assert replacement_profile[36:40] == b"acsp"

    def test_scanner_profile_output_intent_is_replaced(self) -> None:
        """PDF/A OutputIntents accept only monitor or printer profiles."""
        pdf = new_pdf()
        scanner_profile = _make_icc_profile(device_class=b"scnr")
        stream = pdf.make_stream(scanner_profile)
        stream[Name.N] = 3
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Scanner"),
                    DestOutputProfile=stream,
                )
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        replacement = pdf.Root.OutputIntents[0]
        replacement_profile = bytes(replacement["/DestOutputProfile"].read_bytes())
        assert str(replacement["/OutputConditionIdentifier"]) == "sRGB"
        assert replacement_profile[12:16] == b"mntr"

    def test_conflicting_pdfa_output_intents_are_replaced(self) -> None:
        """Conflicting PDF/A OutputIntents fall back to generated dominant intent."""
        pdf = new_pdf()
        profile_rgb = _make_icc_profile()
        profile_cmyk = _make_icc_profile(colorspace=b"CMYK")

        stream1 = pdf.make_stream(profile_rgb)
        stream1[Name.N] = 3
        stream2 = pdf.make_stream(profile_cmyk)
        stream2[Name.N] = 4
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Custom RGB"),
                    DestOutputProfile=stream1,
                ),
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Custom CMYK"),
                    DestOutputProfile=stream2,
                ),
            ]
        )

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        replacement = pdf.Root.OutputIntents[0]
        assert str(replacement["/OutputConditionIdentifier"]) == "sRGB"
        assert bytes(replacement["/DestOutputProfile"].read_bytes()) != profile_rgb

    def test_multiple_identical_output_intents_reduced(self) -> None:
        """Identical PDF/A OutputIntents reduce to one retained entry."""
        pdf = new_pdf()
        profile_data = _make_icc_profile()

        # Create two OutputIntents referencing identical ICC profiles
        stream1 = pdf.make_stream(profile_data)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        stream2 = pdf.make_stream(profile_data)
        stream2[Name.N] = 3
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        # Add a page with RGB content so embed has something to work on
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        assert pdf.Root.OutputIntents[0].S == Name.GTS_PDFA1

    def test_multiple_different_output_intents_reduced(self) -> None:
        """Conflicting PDF/A and PDF/X profiles require a replacement condition."""
        pdf = new_pdf()

        # Two profiles with different bytes (different content)
        profile_rgb = _make_icc_profile()
        profile_cmyk = bytearray(_make_icc_profile())
        profile_cmyk[16:20] = b"CMYK"
        # Also change a later byte to ensure full-bytes comparison matters
        profile_cmyk[50:54] = b"diff"
        profile_cmyk = bytes(profile_cmyk)

        stream1 = pdf.make_stream(profile_rgb)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        stream2 = pdf.make_stream(profile_cmyk)
        stream2[Name.N] = 4
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name("/GTS_PDFX"),
            OutputConditionIdentifier=pikepdf.String("FOGRA39"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        assert pdf.Root.OutputIntents[0].S == Name.GTS_PDFA1

    def test_different_profiles_triggers_warning(self, caplog) -> None:
        """Different ICC profiles across OutputIntents logs a warning."""
        import logging

        pdf = new_pdf()

        profile_a = _make_icc_profile()
        profile_b = bytearray(_make_icc_profile())
        profile_b[60:64] = b"XXXX"  # Same family but different bytes
        profile_b = bytes(profile_b)

        stream1 = pdf.make_stream(profile_a)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        stream2 = pdf.make_stream(profile_b)
        stream2[Name.N] = 3
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        with caplog.at_level(logging.WARNING, logger="pdftopdfa.color_profile"):
            embed_color_profiles(pdf, "2b", replace_existing=True)

        assert any("different ICC profiles" in r.message for r in caplog.records)

    def test_different_profiles_keeps_first_only(self) -> None:
        """Conflicting output conditions are replaced, not silently unified."""
        pdf = new_pdf()

        profile_a = _make_icc_profile()
        profile_b = bytearray(_make_icc_profile())
        profile_b[60:64] = b"XXXX"
        profile_b = bytes(profile_b)

        stream1 = pdf.make_stream(profile_a)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        stream2 = pdf.make_stream(profile_b)
        stream2[Name.N] = 3
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name("/GTS_PDFX"),
            OutputConditionIdentifier=pikepdf.String("sRGB2"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 1
        assert bytes(pdf.Root.OutputIntents[0].DestOutputProfile.read_bytes()) not in {
            profile_a,
            profile_b,
        }

    def test_additional_output_intent_without_profile_preserved(self) -> None:
        """An additional registered condition need not embed another profile."""
        pdf = new_pdf()

        profile_a = _make_icc_profile()
        stream1 = pdf.make_stream(profile_a)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        # Second OutputIntent with no DestOutputProfile
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name("/GTS_PDFX"),
            OutputConditionIdentifier=pikepdf.String("CGATS TR 001"),
            RegistryName=pikepdf.String("http://www.color.org"),
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert len(pdf.Root.OutputIntents) == 2
        assert pdf.Root.OutputIntents[1].S == Name.GTS_PDFX
        assert "/DestOutputProfile" not in pdf.Root.OutputIntents[1]

    @pytest.mark.parametrize(
        "identifier,registry",
        [
            ("Custom", None),
            ("custom", "http://www.color.org"),
            (" CUSTOM ", None),
            ("", None),
            ("  ", "http://www.color.org"),
            ("NotARegistryEntry", None),
            ("NotARegistryEntry", " "),
            ("FOGRA39 extra", None),
        ],
    )
    def test_additional_intent_without_a_standard_condition_is_removed(
        self, identifier, registry
    ) -> None:
        """A non-standard condition must embed its profile, and others stay intact."""
        pdf = new_pdf()
        stream = pdf.make_stream(_make_icc_profile())
        stream[Name.N] = 3
        unregistered = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFX,
            OutputConditionIdentifier=pikepdf.String(identifier),
        )
        if registry is not None:
            unregistered.RegistryName = pikepdf.String(registry)
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier=pikepdf.String("Keep"),
                    DestOutputProfile=stream,
                ),
                unregistered,
            ]
        )
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert [str(i.OutputConditionIdentifier) for i in pdf.Root.OutputIntents] == [
            "Keep"
        ]

    @pytest.mark.parametrize(
        "identifier,registry",
        [
            ("FOGRA39", None),
            ("fogra 51", None),
            ("IFRA26", None),
            ("CGATS TR 006", None),
            ("JC200103", None),
            ("JCW2003", None),
            ("Regional press condition", "http://www.color.org"),
        ],
    )
    def test_additional_intent_naming_a_standard_condition_is_retained(
        self, identifier, registry
    ) -> None:
        """A known family or a stated registry can be used instead of a profile."""
        pdf = new_pdf()
        pdf.add_blank_page(page_size=(612, 792))
        intent = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFX,
            OutputConditionIdentifier=pikepdf.String(identifier),
        )
        if registry is not None:
            intent.RegistryName = pikepdf.String(registry)
        pdf.Root.OutputIntents = Array([intent])

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert [str(i.S) for i in pdf.Root.OutputIntents] == [
            "/GTS_PDFX",
            "/GTS_PDFA1",
        ]

    def test_same_family_different_content_detected(self, caplog) -> None:
        """Profiles with same color space family but different content are flagged.

        The old code only compared bytes 16-20 (family); the new code
        compares the full profile, so two profiles that share the same
        family but differ elsewhere are now correctly identified.
        """
        import logging

        pdf = new_pdf()

        # Both profiles have RGB family at bytes 16-20
        profile_a = _make_icc_profile()
        profile_b = bytearray(_make_icc_profile())
        # Only change bytes outside the family field (bytes 16-20 stay "RGB ")
        profile_b[80:84] = b"DIFF"
        profile_b = bytes(profile_b)

        stream1 = pdf.make_stream(profile_a)
        stream1[Name.N] = 3
        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        stream2 = pdf.make_stream(profile_b)
        stream2[Name.N] = 3
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        with caplog.at_level(logging.WARNING, logger="pdftopdfa.color_profile"):
            embed_color_profiles(pdf, "2b", replace_existing=True)

        assert any("different ICC profiles" in r.message for r in caplog.records)
        assert has_output_intent(pdf)

    def test_identical_profiles_unified_to_same_object(self) -> None:
        """Identical DestOutputProfile streams must share one indirect object.

        Rule 6.2.3-2 requires that when multiple OutputIntents have a
        DestOutputProfile, they all reference the same indirect object.
        """
        pdf = new_pdf()

        profile_data = _make_icc_profile()
        stream1 = pdf.make_stream(profile_data)
        stream1[Name.N] = 3
        stream2 = pdf.make_stream(profile_data)
        stream2[Name.N] = 3

        # Pre-condition: two separate indirect objects
        assert stream1.objgen != stream2.objgen

        oi1 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream1,
        )
        oi2 = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFX,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        from pdftopdfa.utils import resolve_indirect as _ri

        objgens = set()
        assert len(pdf.Root.OutputIntents) == 2
        for oi in pdf.Root.OutputIntents:
            oi = _ri(oi)
            dest = oi.get("/DestOutputProfile")
            if dest is not None:
                objgens.add(_ri(dest).objgen)

        assert len(objgens) == 1, (
            f"Expected all DestOutputProfile entries to share one"
            f" indirect object, got {objgens}"
        )

    @pytest.mark.parametrize("level", ["2a", "2b", "2u", "3a", "3b", "3u"])
    @pytest.mark.parametrize("subtype", ["/GTS_PDFX", "/ISO_PDFE1"])
    @pytest.mark.parametrize("separate_streams", [False, True])
    def test_compatible_intents_preserve_metadata_after_reopen(
        self, level, subtype, separate_streams
    ):
        """Preserve intents of other subtypes and unify identical ICC references."""
        pdf = new_pdf()
        page = pdf.add_blank_page(page_size=(200, 200))
        page.Contents = pdf.make_stream(b"0.5 g 0 0 200 200 re f")
        profile = pdf.make_stream(get_srgb_profile())
        profile.N = 3
        other_profile = profile
        if separate_streams:
            other_profile = pdf.make_stream(get_srgb_profile())
            other_profile.N = 3
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier="Archive",
                    Info="Archival output condition",
                    DestOutputProfile=profile,
                ),
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name(subtype),
                    OutputConditionIdentifier="Press proof",
                    OutputCondition="Retain this printing condition",
                    RegistryName="https://example.com/conditions",
                    Info="Separate output metadata",
                    DestOutputProfile=other_profile,
                ),
            ]
        )
        embed_color_profiles(pdf, level)
        saved = BytesIO()
        pdf.save(saved)
        with Pdf.open(saved) as restored:
            first, second = restored.Root.OutputIntents
            assert str(first.OutputConditionIdentifier) == "Archive"
            assert str(second.S) == subtype
            assert str(second.OutputConditionIdentifier) == "Press proof"
            assert str(second.OutputCondition) == "Retain this printing condition"
            assert str(second.RegistryName) == "https://example.com/conditions"
            assert str(second.Info) == "Separate output metadata"
            assert first.DestOutputProfile.objgen == second.DestOutputProfile.objgen
            assert first.DestOutputProfile.read_bytes() == get_srgb_profile()

    def test_compatible_pdfx_external_reference_removed_without_dropping_intent(self):
        """Remove only the forbidden profile reference, retaining embedded data."""
        pdf = new_pdf()
        pdf.add_blank_page(page_size=(200, 200))
        intent = create_output_intent_for_colorspace(
            pdf, ColorSpaceType.DEVICE_RGB, get_srgb_profile()
        )
        pdfx = Dictionary(intent)
        pdfx.S = Name.GTS_PDFX
        pdfx.DestOutputProfileRef = Dictionary(URL="https://example.com/profile.icc")
        pdf.Root.OutputIntents = Array([intent, pdfx])
        embed_color_profiles(pdf, "2b")
        assert len(pdf.Root.OutputIntents) == 2
        assert pdf.Root.OutputIntents[1].S == Name.GTS_PDFX
        assert "/DestOutputProfileRef" not in pdf.Root.OutputIntents[1]

    @pytest.mark.parametrize("level", ["2b", "3b"])
    @pytest.mark.parametrize("existing_pdfa", [False, True])
    def test_converter_adds_pdfa_intent_without_losing_pdfx_condition(
        self, tmp_path, level, existing_pdfa
    ):
        """A source PDF/X output condition survives complete PDF/A conversion."""
        from pdftopdfa.converter import convert_to_pdfa
        from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf

        pdf = new_pdf()
        page = pdf.add_blank_page(page_size=(200, 200))
        page.Contents = pdf.make_stream(b"0.5 g 0 0 200 200 re f")
        pdfx = create_output_intent_for_colorspace(
            pdf, ColorSpaceType.DEVICE_RGB, get_srgb_profile()
        )
        pdfx.S = Name.GTS_PDFX
        pdfx.OutputConditionIdentifier = "Original print condition"
        pdfx.Info = "Keep original printing information"
        pdf.Root.OutputIntents = Array([pdf.make_indirect(pdfx)])
        if existing_pdfa:
            pdfa = Dictionary(pdfx)
            pdfa.S = Name.GTS_PDFA1
            pdf.Root.OutputIntents.append(pdf.make_indirect(pdfa))
        source = tmp_path / "source.pdf"
        output = tmp_path / "output.pdf"
        pdf.save(source)
        result = convert_to_pdfa(source, output, level=level)
        assert result.success and result.target_produced and not result.skipped, result
        with Pdf.open(output) as restored:
            first, second = restored.Root.OutputIntents
            assert first.S == Name.GTS_PDFX
            assert str(first.OutputConditionIdentifier) == "Original print condition"
            assert str(first.Info) == "Keep original printing information"
            assert second.S == Name.GTS_PDFA1
            assert first.DestOutputProfile.objgen == second.DestOutputProfile.objgen
            assert first.DestOutputProfile.read_bytes() == get_srgb_profile()
        if is_verapdf_available():
            validation = validate_with_verapdf(output, flavour=level)
            assert validation.compliant, validation.failed_rules

    @pytest.mark.parametrize("level", ["2b", "3b"])
    @pytest.mark.parametrize("subtype", ["/GTS_PDFX", "/ISO_PDFE1"])
    def test_converter_preserves_registered_output_condition(
        self, tmp_path, level, subtype
    ):
        """Adding an embedded profile retains an existing registered condition."""
        from pdftopdfa.converter import convert_to_pdfa
        from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf

        pdf = new_pdf()
        page = pdf.add_blank_page(page_size=(200, 200))
        page.Contents = pdf.make_stream(b"0 1 0 0 k 0 0 200 200 re f")
        pdf.Root.OutputIntents = Array(
            [
                pdf.make_indirect(
                    Dictionary(
                        Type=Name.OutputIntent,
                        S=Name(subtype),
                        OutputConditionIdentifier="FOGRA39",
                        RegistryName="http://www.color.org",
                        Info="Original registered printing condition",
                    )
                )
            ]
        )
        source = tmp_path / "source.pdf"
        output = tmp_path / "output.pdf"
        pdf.save(source)

        result = convert_to_pdfa(source, output, level=level)
        assert result.success and result.target_produced and not result.skipped, result
        with Pdf.open(output) as restored:
            original, added = restored.Root.OutputIntents
            assert str(original.S) == subtype
            assert str(original.OutputConditionIdentifier) == "FOGRA39"
            assert str(original.RegistryName) == "http://www.color.org"
            assert str(original.Info) == "Original registered printing condition"
            assert "/DestOutputProfile" not in original
            assert added.S == Name.GTS_PDFA1
            assert added.DestOutputProfile.N == 4
        if is_verapdf_available():
            validation = validate_with_verapdf(output, flavour=level)
            assert validation.compliant, validation.failed_rules


class TestPreservedColorValidation:
    """Preserved dictionaries and color spaces still have to be well formed."""

    @pytest.mark.parametrize(
        "location", ["catalog", "resources", "calibration", "image"]
    )
    def test_private_arrays_are_not_interpreted_as_color_spaces(self, location):
        """A color-like first name does not change private objects' semantics."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        payload = Array([Name.CalRGB, "Private data, not calibration"])
        if location == "catalog":
            pdf.Root.AcmePrivate = Dictionary(Payload=payload)
        elif location == "resources":
            page.Resources.AcmePrivate = Dictionary(Payload=payload)
        elif location == "calibration":
            cs = TestCalibratedPreservation._colorspace("CalRGB")
            cs[1].AcmePrivate = payload
            page.Resources.ColorSpace = Dictionary(Cal=cs)
        else:
            image = pdf.make_stream(b"\x80")
            image.Subtype = Name.Image
            image.ColorSpace = Name.DeviceGray
            image.AcmePrivate = payload
            image.Resources = Dictionary(ColorSpace=Dictionary(Private=payload))
            page.Resources.XObject = Dictionary(Im=image)
        expected = payload.unparse()
        embed_color_profiles(pdf)
        assert payload.unparse() == expected

    @pytest.mark.parametrize(
        "location",
        ["shading", "type3", "page_group", "appearance_group", "indexed_image", "form"],
    )
    def test_invalid_calibration_is_found_in_typed_contexts(self, location):
        """All actual nested color contexts retain calibration validation."""
        pdf = new_pdf()
        TestCalibratedPreservation._add_colorspace(
            pdf, Array([Name.CalRGB, Dictionary()]), location
        )
        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf)

    @pytest.mark.parametrize("location", ["form", "field", "nested_field"])
    def test_invalid_calibration_in_form_resources_is_rejected(self, location):
        """Default form resources are reachable without any page referring to them."""
        pdf = new_pdf()
        pdf.add_blank_page()
        resources = Dictionary(
            ColorSpace=Dictionary(CS=Array([Name.CalRGB, Dictionary()]))
        )
        field = pdf.make_indirect(Dictionary(T="Field"))
        pdf.Root.AcroForm = Dictionary(Fields=Array([field]))
        if location == "form":
            pdf.Root.AcroForm.DR = resources
        elif location == "field":
            field.DR = resources
        else:
            child = pdf.make_indirect(Dictionary(T="Child", Parent=field, DR=resources))
            field.Kids = Array([child])
        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf)

    def test_valid_form_resources_and_cyclic_fields_are_accepted(self):
        """Traversing the field tree terminates and does not reject valid resources."""
        pdf = new_pdf()
        pdf.add_blank_page()
        field = pdf.make_indirect(Dictionary(T="Field"))
        field.Kids = Array([field])
        pdf.Root.AcroForm = Dictionary(
            Fields=Array([field]),
            DR=Dictionary(
                ColorSpace=Dictionary(
                    CS=TestCalibratedPreservation._colorspace("CalRGB")
                )
            ),
        )
        embed_color_profiles(pdf)

    def test_invalid_calibration_in_shading_pattern_soft_mask_is_rejected(self):
        """A pattern's ExtGState holds its mask group outside any resource table."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        mask = pdf.make_stream(b"0 0 20 20 re f")
        mask.Subtype = Name.Form
        mask.BBox = Array([0, 0, 20, 20])
        mask.Resources = Dictionary(
            ColorSpace=Dictionary(CS=Array([Name.CalRGB, Dictionary()]))
        )
        page.Resources.Pattern = Dictionary(
            P=Dictionary(
                PatternType=2,
                ExtGState=Dictionary(SMask=Dictionary(S=Name.Alpha, G=mask)),
                Shading=Dictionary(ShadingType=2, ColorSpace=Name.DeviceGray),
            )
        )
        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf)

    @staticmethod
    def _inline_image_owner(pdf, page, location, data):
        """Place inline image content in the requested typed painting context."""
        if location == "page":
            page.Contents = pdf.make_stream(data)
            return
        program = pdf.make_stream(data)
        if location == "form":
            program.Subtype = Name.Form
            program.BBox = Array([0, 0, 10, 10])
            page.Resources.XObject = Dictionary(Fm=program)
        elif location == "pattern":
            program.PatternType = 1
            program.PaintType = 1
            program.TilingType = 1
            program.BBox = Array([0, 0, 10, 10])
            program.XStep = program.YStep = 10
            program.Resources = Dictionary()
            page.Resources.Pattern = Dictionary(P=program)
        else:
            page.Resources.Font = Dictionary(
                F=Dictionary(
                    Subtype=Name.Type3,
                    FontBBox=Array([0, 0, 10, 10]),
                    FontMatrix=Array([1, 0, 0, 1, 0, 0]),
                    CharProcs=Dictionary(a=program),
                    Encoding=Dictionary(Differences=Array([97, Name.a])),
                    FirstChar=97,
                    LastChar=97,
                    Widths=Array([10]),
                )
            )

    @pytest.mark.parametrize("location", ["page", "form", "pattern", "type3"])
    @pytest.mark.parametrize(
        "space",
        [
            b"[/CalRGB << >>]",
            b"[/CalGray << >>]",
            b"[/I [/CalRGB << >>] 0 <000000>]",
            b"[/Indexed [/CalRGB << >>] 0 <000000>]",
        ],
    )
    def test_invalid_calibration_in_inline_images_is_rejected(self, location, space):
        """Directly declared inline-image spaces have no resource entry to check."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        self._inline_image_owner(
            pdf,
            page,
            location,
            b"BI /W 1 /H 1 /BPC 8 /CS " + space + b" ID \x80\x80\x80 EI",
        )
        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf)

    def test_inline_image_divided_between_content_streams_is_checked(self):
        """A page's /Contents array is interpreted as one concatenated stream."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        page.Contents = Array(
            [
                pdf.make_stream(b"q BI"),
                pdf.make_stream(
                    b"/W 1 /H 1 /BPC 8 /CS [/CalRGB << >>] ID \x80\x80\x80 EI Q"
                ),
            ]
        )
        with pytest.raises(ConversionError, match="invalid /WhitePoint"):
            embed_color_profiles(pdf)

    @pytest.mark.parametrize(
        "data",
        [
            b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \x80\x80\x80 EI",
            b"BI /W 1 /H 1 /BPC 8 /CS [/I /RGB 0 <000000>] ID \x80 EI",
            b"BI /W 1 /H 1 /BPC 8 /CS [/CalRGB << /WhitePoint [1 1 1] >>]"
            b" ID \x80\x80\x80 EI",
            b"(/CalRGB BI) Tj",
            b"BI /CalRGB ( unterminated",
        ],
    )
    def test_valid_or_unrelated_inline_content_is_accepted(self, data):
        """Unrelated text and content that cannot be parsed are not calibrations."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        page.Contents = pdf.make_stream(data)
        embed_color_profiles(pdf)
        assert page.Contents.read_bytes() == data

    def test_private_array_survives_full_conversion(self, tmp_path):
        """Unrelated extension data cannot cause the original-input fallback."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = new_pdf()
        pdf.add_blank_page()
        pdf.Root.AcmePrivate = Dictionary(Payload=Array([Name.CalGray, 42]))
        source, output = tmp_path / "private.pdf", tmp_path / "converted.pdf"
        pdf.save(source)
        result = convert_to_pdfa(source, output)
        assert result.target_produced and not result.skipped
        with Pdf.open(output) as restored:
            assert list(restored.Root.AcmePrivate.Payload) == [Name.CalGray, 42]

    @pytest.mark.parametrize("profiled", [False, True])
    @pytest.mark.parametrize(
        "key,value",
        [
            ("/Type", Name.Other),
            ("/OutputConditionIdentifier", None),
            ("/OutputConditionIdentifier", 42),
            ("/Info", Name.Bad),
            ("/RegistryName", 42),
        ],
    )
    def test_malformed_additional_intent_is_replaced(
        self, profiled, key, value
    ) -> None:
        """A registered printing condition cannot bypass dictionary validation."""
        pdf = new_pdf()
        pdf.add_blank_page()
        intent = pdf.make_indirect(
            Dictionary(
                Type=Name.OutputIntent,
                S=Name.GTS_PDFX,
                OutputConditionIdentifier="Original",
            )
        )
        if profiled:
            intent.DestOutputProfile = pdf.make_stream(get_srgb_profile())
            intent.DestOutputProfile.N = 3
        if value is None:
            del intent[key]
        else:
            intent[key] = value
        pdf.Root.OutputIntents = Array([intent])
        embed_color_profiles(pdf)
        assert len(pdf.Root.OutputIntents) == 1
        replacement = pdf.Root.OutputIntents[0]
        assert replacement.objgen != intent.objgen
        assert replacement.Type == Name.OutputIntent and replacement.S == Name.GTS_PDFA1
        assert isinstance(replacement.OutputConditionIdentifier, pikepdf.String)

    @pytest.mark.parametrize("profiled", [False, True])
    def test_optional_output_intent_type_can_be_absent(self, profiled) -> None:
        """ISO 32000-1 Table 365 permits an omitted OutputIntent /Type."""
        pdf = new_pdf()
        pdf.add_blank_page()
        intent = pdf.make_indirect(
            Dictionary(S=Name.GTS_PDFX, OutputConditionIdentifier="FOGRA39")
        )
        if profiled:
            intent.DestOutputProfile = pdf.make_stream(get_cmyk_profile())
            intent.DestOutputProfile.N = 4
        pdf.Root.OutputIntents = Array([intent])
        embed_color_profiles(pdf)
        assert len(pdf.Root.OutputIntents) == 2
        assert pdf.Root.OutputIntents[0].objgen == intent.objgen
        assert "/Type" not in pdf.Root.OutputIntents[0]

    @pytest.mark.parametrize(
        "kind,key,value",
        [
            ("CalGray", "/WhitePoint", None),
            ("CalRGB", "/WhitePoint", Array([1, 1])),
            ("CalGray", "/WhitePoint", Array([0, 1, 1])),
            ("CalRGB", "/WhitePoint", Array([1, 2, 1])),
            ("CalGray", "/BlackPoint", Array([-1, 0, 0])),
            ("CalGray", "/Gamma", 0),
            ("CalGray", "/Gamma", True),
            ("CalRGB", "/Gamma", Array([1, 2])),
            ("CalRGB", "/Gamma", Array([0, 1, 1])),
            ("CalRGB", "/Gamma", Array([1, 1, -1])),
            ("CalRGB", "/Matrix", Array([1, 2])),
        ],
    )
    @pytest.mark.parametrize("option", [False, True])
    def test_malformed_calibration_cannot_be_preserved(
        self, kind, key, value, option
    ) -> None:
        """Both option paths reject invalid calibration before preserving it."""
        pdf = new_pdf()
        cs = TestCalibratedPreservation._colorspace(kind)
        if value is None:
            del cs[1][key]
        else:
            cs[1][key] = pdf_object(value)
        page = pdf.add_blank_page()
        page.Resources.ColorSpace = Dictionary(Cal=cs)
        with pytest.raises(ConversionError, match="Malformed /Cal"):
            embed_color_profiles(pdf, convert_calibrated=option)

    @pytest.mark.parametrize("kind", ["CalGray", "CalRGB"])
    def test_invalid_calibration_uses_original_input_fallback(
        self, kind, tmp_path
    ) -> None:
        """Report invalid calibration without publishing a claimed PDF/A target."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = new_pdf()
        page = pdf.add_blank_page()
        page.Resources.ColorSpace = Dictionary(
            Cal=Array([Name("/" + kind), Dictionary()])
        )
        page.Contents = pdf.make_stream(b"/Cal cs 0.5 scn 0 0 100 100 re f")
        source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
        pdf.save(source)
        result = convert_to_pdfa(source, output)
        assert result.skipped and not result.target_produced
        assert output.read_bytes() == source.read_bytes()
        assert any("Malformed /Cal" in warning for warning in result.warnings)

    @pytest.mark.parametrize(
        "key,color,expected",
        [
            ("/C", [0, 1, 0, 0], ColorSpaceType.DEVICE_CMYK),
            ("/IC", [0, 1, 0], ColorSpaceType.DEVICE_RGB),
            ("/C", [0.5], ColorSpaceType.DEVICE_GRAY),
        ],
    )
    def test_annotation_color_without_appearance_is_detected(
        self, key, color, expected
    ) -> None:
        """Border and interior arrays contribute their device color space."""
        pdf = new_pdf()
        page = pdf.add_blank_page()
        annotation = Dictionary(Subtype=Name.Link if key == "/C" else Name.Square)
        annotation[key] = Array(color)
        page.Annots = Array([pdf.make_indirect(annotation)])
        assert expected in detect_color_spaces(pdf).detected_spaces


class TestDefaultColorSpaces:
    """Tests for context-dependent Default color space entries."""

    def test_mixed_rgb_gray_needs_no_default_gray(self) -> None:
        """Any PDF/A OutputIntent covers DeviceGray without DefaultGray."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        # Content stream uses both RGB and Gray operators
        content = pdf.make_stream(
            b"1 0 0 rg 100 100 200 300 re f 0.5 g 50 50 100 100 re f"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedded = embed_color_profiles(pdf, "2b")

        assert ColorSpaceType.DEVICE_RGB in embedded
        assert ColorSpaceType.DEVICE_GRAY in embedded

        # OutputIntent is RGB (dominant)
        assert pdf.Root.OutputIntents[0]["/DestOutputProfile"]["/N"] == 3

        resources = pdf.pages[0].get(Name.Resources)
        assert resources is None or Name.ColorSpace not in resources

    def test_mixed_cmyk_rgb_adds_default_rgb(self) -> None:
        """CMYK dominant + RGB non-dominant → DefaultRGB in page resources."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(
            b"0 1 0 0 k 100 100 200 300 re f 1 0 0 rg 50 50 100 100 re f"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedded = embed_color_profiles(pdf, "2b")

        assert ColorSpaceType.DEVICE_CMYK in embedded
        assert ColorSpaceType.DEVICE_RGB in embedded

        # OutputIntent is CMYK (dominant)
        assert pdf.Root.OutputIntents[0]["/DestOutputProfile"]["/N"] == 4

        cs_dict = pdf.pages[0]["/Resources"]["/ColorSpace"]
        assert Name.DefaultRGB in cs_dict
        assert Name.DefaultCMYK not in cs_dict

    def test_gray_image_uses_output_intent_with_rgb_dominant(self) -> None:
        """A Gray image remains DeviceGray when a PDF/A OutputIntent exists."""
        pdf = new_pdf()

        # RGB content stream + Gray image
        gray_image = pdf.make_stream(b"\x80")
        gray_image[Name.Type] = Name.XObject
        gray_image[Name.Subtype] = Name.Image
        gray_image[Name.Width] = 1
        gray_image[Name.Height] = 1
        gray_image[Name.ColorSpace] = Name.DeviceGray
        gray_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=gray_image)),
        )
        content = pdf.make_stream(
            b"1 0 0 rg 100 100 200 300 re f q 50 0 0 50 10 10 cm /Im0 Do Q"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        img = pdf.pages[0]["/Resources"]["/XObject"]["/Im0"]
        assert img["/ColorSpace"] == Name.DeviceGray
        assert Name.ColorSpace not in pdf.pages[0]["/Resources"]

    def test_indexed_image_base_uses_default_rgb(self) -> None:
        """An Indexed DeviceRGB base stays contextual and uses DefaultRGB."""
        pdf = new_pdf()

        # Create a 1-pixel indexed image: [/Indexed /DeviceRGB 255 <lookup>]
        lookup_data = bytes(range(256)) * 3  # 256 entries × 3 components
        lookup_stream = pdf.make_stream(lookup_data)
        indexed_cs = Array([Name.Indexed, Name.DeviceRGB, 255, lookup_stream])

        indexed_image = pdf.make_stream(b"\x00")
        indexed_image[Name.Type] = Name.XObject
        indexed_image[Name.Subtype] = Name.Image
        indexed_image[Name.Width] = 1
        indexed_image[Name.Height] = 1
        indexed_image[Name.ColorSpace] = indexed_cs
        indexed_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=indexed_image)),
        )
        # CMYK content operator makes CMYK dominant, so RGB is non-dominant
        content = pdf.make_stream(
            b"0 1 0 0 k 100 100 200 300 re f q 50 0 0 50 10 10 cm /Im0 Do Q"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        img = pdf.pages[0]["/Resources"]["/XObject"]["/Im0"]
        img_cs = img["/ColorSpace"]
        # Still an Indexed array
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.Indexed
        assert img_cs[1] == Name.DeviceRGB
        # hival and lookup remain intact
        assert int(img_cs[2]) == 255
        assert img_cs[3].objgen == lookup_stream.objgen
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1][Name.N] == 3

    def test_iccbased_image_not_touched(self) -> None:
        """Image already using ICCBased is not modified."""
        pdf = new_pdf()

        # Create an ICCBased color space for the image
        icc_data = get_srgb_profile()
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 3
        icc_cs = Array([Name.ICCBased, icc_stream])

        rgb_image = pdf.make_stream(b"\xff\x00\x00")
        rgb_image[Name.Type] = Name.XObject
        rgb_image[Name.Subtype] = Name.Image
        rgb_image[Name.Width] = 1
        rgb_image[Name.Height] = 1
        rgb_image[Name.ColorSpace] = icc_cs
        rgb_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=rgb_image)),
        )
        # Add a gray operator to trigger mixed color spaces
        content = pdf.make_stream(
            b"0.5 g 50 50 100 100 re f q 50 0 0 50 10 10 cm /Im0 Do Q"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Image CS is still the original ICCBased array (not replaced)
        img = pdf.pages[0]["/Resources"]["/XObject"]["/Im0"]
        img_cs = img["/ColorSpace"]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
        # The stream should be the original one (N=3 for sRGB)
        assert img_cs[1]["/N"] == 3

    def test_single_colorspace_no_defaults(self) -> None:
        """Single color space → no Default entries added."""
        pdf = new_pdf()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedded = embed_color_profiles(pdf, "2b")
        assert embedded == [ColorSpaceType.DEVICE_RGB]

        resources = pdf.pages[0].get("/Resources")
        if resources is not None:
            cs_dict = resources.get("/ColorSpace")
            # Either no ColorSpace dict or no Default entries
            if cs_dict is not None:
                assert Name.DefaultGray not in cs_dict
                assert Name.DefaultRGB not in cs_dict
                assert Name.DefaultCMYK not in cs_dict

    def test_form_xobject_gets_default_colorspace(self) -> None:
        """Form XObject with RGB operators gets DefaultRGB in its resources."""
        pdf = new_pdf()

        # Create a Form XObject that uses RGB operators
        form_content = b"1 0 0 rg 10 10 80 80 re f"
        form_stream = pdf.make_stream(form_content)
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary()

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        # Page uses CMYK, so RGB requires a DefaultRGB entry.
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Form XObject resources should have DefaultRGB
        form = pdf.pages[0]["/Resources"]["/XObject"]["/Fm0"]
        form_res = form["/Resources"]
        cs_dict = form_res["/ColorSpace"]
        assert Name.DefaultRGB in cs_dict
        default_rgb = cs_dict[Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1]["/N"] == 3

    def test_defaults_traverse_1200_nested_forms_iteratively(self) -> None:
        """Deep Form resource graphs get defaults without mutating images."""
        pdf = new_pdf()
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 8

        nested = pdf.make_stream(b"/Image Do")
        nested[Name.Type] = Name.XObject
        nested[Name.Subtype] = Name.Form
        nested[Name.BBox] = Array([0, 0, 1, 1])
        nested[Name.Resources] = Dictionary(XObject=Dictionary(Image=image))
        leaf = nested

        for _ in range(1200):
            parent = pdf.make_stream(b"/Nested Do")
            parent[Name.Type] = Name.XObject
            parent[Name.Subtype] = Name.Form
            parent[Name.BBox] = Array([0, 0, 1, 1])
            parent[Name.Resources] = Dictionary(XObject=Dictionary(Nested=nested))
            nested = parent

        page = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 1, 1]),
            Resources=Dictionary(XObject=Dictionary(Nested=nested)),
            Contents=pdf.make_stream(b"/Nested Do"),
        )
        pdf.pages.append(pikepdf.Page(page))

        defaults_added, images_replaced = _apply_default_colorspaces(
            pdf,
            {ColorSpaceType.DEVICE_GRAY},
            {},
        )

        assert defaults_added == 1202
        assert images_replaced == 0
        assert (
            leaf[Name.Resources][Name.ColorSpace][Name.DefaultGray][0] == Name.ICCBased
        )
        assert image[Name.ColorSpace] == Name.DeviceGray

    def test_all_three_spaces_cmyk_dominant(self) -> None:
        """CMYK + RGB + Gray → OutputIntent=CMYK, DefaultRGB + DefaultGray added."""
        pdf = new_pdf()

        # Gray image
        gray_image = pdf.make_stream(b"\x80")
        gray_image[Name.Type] = Name.XObject
        gray_image[Name.Subtype] = Name.Image
        gray_image[Name.Width] = 1
        gray_image[Name.Height] = 1
        gray_image[Name.ColorSpace] = Name.DeviceGray
        gray_image[Name.BitsPerComponent] = 8

        # RGB image
        rgb_image = pdf.make_stream(b"\xff\x00\x00")
        rgb_image[Name.Type] = Name.XObject
        rgb_image[Name.Subtype] = Name.Image
        rgb_image[Name.Width] = 1
        rgb_image[Name.Height] = 1
        rgb_image[Name.ColorSpace] = Name.DeviceRGB
        rgb_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=gray_image, Im1=rgb_image)),
        )
        # CMYK operator in content stream
        content = pdf.make_stream(
            b"0 1 0 0 k 100 100 200 300 re f "
            b"q 50 0 0 50 10 10 cm /Im0 Do Q "
            b"q 50 0 0 50 70 10 cm /Im1 Do Q"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embedded = embed_color_profiles(pdf, "2b")

        assert ColorSpaceType.DEVICE_CMYK in embedded
        assert ColorSpaceType.DEVICE_RGB in embedded
        assert ColorSpaceType.DEVICE_GRAY in embedded

        # OutputIntent is CMYK
        assert pdf.Root.OutputIntents[0]["/DestOutputProfile"]["/N"] == 4

        # DeviceGray is covered by the OutputIntent; RGB needs DefaultRGB.
        cs_dict = pdf.pages[0]["/Resources"]["/ColorSpace"]
        assert Name.DefaultRGB in cs_dict
        assert Name.DefaultGray not in cs_dict
        assert Name.DefaultCMYK not in cs_dict

        # Both images keep their context-dependent Device names.
        gray_img = pdf.pages[0]["/Resources"]["/XObject"]["/Im0"]
        assert gray_img["/ColorSpace"] == Name.DeviceGray

        rgb_img = pdf.pages[0]["/Resources"]["/XObject"]["/Im1"]
        assert rgb_img["/ColorSpace"] == Name.DeviceRGB

    def test_existing_default_preserved(self) -> None:
        """A pre-existing DefaultGray remains authoritative for an image."""
        pdf = new_pdf()

        # Create a custom ICCBased array for DefaultGray
        custom_gray = get_gray_profile()
        custom_stream = pdf.make_stream(custom_gray)
        custom_stream[Name.N] = 1
        custom_cs = Array([Name.ICCBased, custom_stream])

        cs_dict = Dictionary()
        cs_dict[Name.DefaultGray] = custom_cs

        gray_image = pdf.make_stream(b"\x80")
        gray_image[Name.Type] = Name.XObject
        gray_image[Name.Subtype] = Name.Image
        gray_image[Name.Width] = 1
        gray_image[Name.Height] = 1
        gray_image[Name.ColorSpace] = Name.DeviceGray
        gray_image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=cs_dict,
                XObject=Dictionary(Im0=gray_image),
            ),
        )
        # Both RGB and Gray operators
        content = pdf.make_stream(
            b"1 0 0 rg 100 100 200 300 re f 0.5 g 50 50 100 100 re f /Im0 Do"
        )
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # DefaultGray should still point to the original custom stream
        cs_dict = pdf.pages[0]["/Resources"]["/ColorSpace"]
        default_gray = cs_dict[Name.DefaultGray]
        assert default_gray[0] == Name.ICCBased
        # Verify it's the same stream object (not replaced)
        assert default_gray[1].objgen == custom_stream.objgen
        assert gray_image[Name.ColorSpace] == Name.DeviceGray

    def test_shared_image_preserves_contextual_defaults(self) -> None:
        """A shared image keeps the DefaultRGB of each resource context."""
        pdf = new_pdf()
        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        defaults = []
        for _ in range(2):
            profile = pdf.make_stream(get_srgb_profile())
            profile[Name.N] = 3
            default_rgb = Array([Name.ICCBased, profile])
            defaults.append(profile)
            pdf.pages.append(
                pikepdf.Page(
                    Dictionary(
                        Type=Name.Page,
                        MediaBox=Array([0, 0, 10, 10]),
                        Resources=Dictionary(
                            ColorSpace=Dictionary(DefaultRGB=default_rgb),
                            XObject=Dictionary(Im0=image),
                        ),
                        Contents=pdf.make_stream(b"/Im0 Do"),
                    )
                )
            )

        _apply_default_colorspaces(pdf, {ColorSpaceType.DEVICE_RGB}, {})

        assert image[Name.ColorSpace] == Name.DeviceRGB
        for page, profile in zip(pdf.pages, defaults, strict=True):
            default_rgb = page[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
            assert default_rgb[1].objgen == profile.objgen

    def test_shared_shading_preserves_contextual_defaults(self) -> None:
        """A shared shading keeps the DefaultRGB of each resource context."""
        pdf = new_pdf()
        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 10, 10]),
            )
        )

        defaults = []
        for _ in range(2):
            profile = pdf.make_stream(get_srgb_profile())
            profile[Name.N] = 3
            default_rgb = Array([Name.ICCBased, profile])
            defaults.append(profile)
            pdf.pages.append(
                pikepdf.Page(
                    Dictionary(
                        Type=Name.Page,
                        MediaBox=Array([0, 0, 10, 10]),
                        Resources=Dictionary(
                            ColorSpace=Dictionary(DefaultRGB=default_rgb),
                            Shading=Dictionary(Sh0=shading),
                        ),
                        Contents=pdf.make_stream(b"/Sh0 sh"),
                    )
                )
            )

        _apply_default_colorspaces(pdf, {ColorSpaceType.DEVICE_RGB}, {})

        assert shading[Name.ColorSpace] == Name.DeviceRGB
        for page, profile in zip(pdf.pages, defaults, strict=True):
            default_rgb = page[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
            assert default_rgb[1].objgen == profile.objgen


class TestAnnotationAPColorSpaceDetection:
    """Tests for color space detection in annotation appearance streams."""

    def test_detect_device_gray_in_ap_content_stream(self) -> None:
        """DeviceGray operator in AP /N stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # Create AP stream with DeviceGray operator
        ap_stream = pdf.make_stream(b"0.5 g 50 50 100 100 re f")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 200, 200])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 200, 200]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is False

    def test_detect_colorspace_in_ap_image(self) -> None:
        """DeviceRGB image inside an AP stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # Image XObject with DeviceRGB
        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        # AP stream that references the image
        ap_stream = pdf.make_stream(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 200, 200])
        ap_stream[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=image),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 200, 200]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_detect_ap_sub_state_dict(self) -> None:
        """AP /N as sub-state dict (On/Off) detects colors in both."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # "Yes" sub-state with DeviceGray
        on_stream = pdf.make_stream(b"0.5 g 10 10 80 80 re f")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 100, 100])

        # "Off" sub-state with DeviceRGB
        off_stream = pdf.make_stream(b"1 0 0 rg 10 10 80 80 re f")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 100, 100])

        sub_state_dict = Dictionary(Yes=on_stream, Off=off_stream)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Widget"),
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=sub_state_dict),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is True

    def test_detect_ap_r_and_d_entries(self) -> None:
        """AP /R and /D entries are also scanned for color spaces."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # /R stream with DeviceRGB
        r_stream = pdf.make_stream(b"0 1 0 rg 10 10 80 80 re f")
        r_stream[Name.Type] = Name.XObject
        r_stream[Name.Subtype] = Name.Form
        r_stream[Name.BBox] = Array([0, 0, 100, 100])

        # /D stream with DeviceCMYK
        d_stream = pdf.make_stream(b"0 0 1 0 k 10 10 80 80 re f")
        d_stream[Name.Type] = Name.XObject
        d_stream[Name.Subtype] = Name.Form
        d_stream[Name.BBox] = Array([0, 0, 100, 100])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(R=r_stream, D=d_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_cmyk_used is True

    def test_detect_device_rgb_inline_image_in_ap_stream(self) -> None:
        """DeviceRGB inline image inside an AP stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        ap_stream = pdf.make_stream(b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 100])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_gray_inline_image_in_ap_stream(self) -> None:
        """DeviceGray inline image inside an AP stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        ap_stream = pdf.make_stream(b"BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 100])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_detect_device_cmyk_inline_image_in_ap_stream(self) -> None:
        """DeviceCMYK inline image inside an AP stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        ap_stream = pdf.make_stream(
            b"BI /W 1 /H 1 /CS /CMYK /BPC 8 ID \x00\xff\x00\x00 EI"
        )
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 100])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False

    def test_detect_inline_image_in_ap_sub_state_dict(self) -> None:
        """Inline image in AP sub-state dictionary is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # "Yes" sub-state with DeviceRGB inline image
        on_stream = pdf.make_stream(b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 100, 100])

        # "Off" sub-state with DeviceGray operator (no inline image)
        off_stream = pdf.make_stream(b"0.5 g 10 10 80 80 re f")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 100, 100])

        sub_state_dict = Dictionary(Yes=on_stream, Off=off_stream)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name("/Widget"),
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=sub_state_dict),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is True

    def test_detect_indexed_inline_image_in_ap_stream(self) -> None:
        """Indexed inline image with DeviceRGB base in AP stream is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        ap_stream = pdf.make_stream(
            b"BI /W 2 /H 1 /CS [/Indexed /RGB 1 <FF000000FF00>] /BPC 8 ID \x00\x01 EI"
        )
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 100])

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 100, 100]),
                AP=Dictionary(N=ap_stream),
            )
        )
        pdf.pages[0][Name.Annots] = Array([annot])

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_gray_used is False
        assert analysis.device_cmyk_used is False


class TestAnnotationAPDefaultColorSpaces:
    """Tests for applying default color spaces to annotation AP streams."""

    def test_ap_stream_gets_default_colorspace(self) -> None:
        """AP Form XObject resources get a required DefaultRGB entry."""
        pdf = new_pdf()

        # AP stream with DeviceRGB operator
        ap_stream = pdf.make_stream(b"1 0 0 rg 50 50 100 100 re f")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 200, 200])
        ap_stream[Name.Resources] = Dictionary()

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 200, 200]),
                AP=Dictionary(N=ap_stream),
            )
        )

        # Page with CMYK content so RGB requires a DefaultRGB entry.
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        # AP stream resources should have DefaultRGB
        ap_res = ap_stream.get(Name.Resources)
        assert ap_res is not None
        cs_dict = ap_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_rgb = cs_dict.get(Name.DefaultRGB)
        assert default_rgb is not None
        assert default_rgb[0] == Name.ICCBased

    def test_ap_image_uses_contextual_default(self) -> None:
        """An AP image keeps DeviceRGB and uses the AP's DefaultRGB."""
        pdf = new_pdf()

        # Image with DeviceRGB
        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        # AP stream referencing the image
        ap_stream = pdf.make_stream(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 200, 200])
        ap_stream[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=image),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 200, 200]),
                AP=Dictionary(N=ap_stream),
            )
        )

        # Page with CMYK content so RGB requires a DefaultRGB entry.
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        assert image[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = ap_stream[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1][Name.N] == 3


class TestPatternColorSpaceDetection:
    """Tests for color space detection in Pattern and Shading resources."""

    def test_detect_devicergb_in_shading_dict(self) -> None:
        """DeviceRGB in /Resources/Shading/Sh0 is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True
        assert analysis.device_cmyk_used is False

    def test_detect_devicecmyk_in_shading_pattern(self) -> None:
        """PatternType=2 (Shading pattern) with DeviceCMYK is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceCMYK,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True
        assert analysis.device_rgb_used is False

    def test_detect_devicegray_in_tiling_pattern_content_stream(self) -> None:
        """Tiling pattern with '0.5 g' operator detects DeviceGray."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        pattern = pdf.make_stream(b"0.5 g 0 0 10 10 re f")
        pattern[Name.Type] = Name.Pattern
        pattern[Name("/PatternType")] = 1
        pattern[Name("/PaintType")] = 1
        pattern[Name("/TilingType")] = 1
        pattern[Name.BBox] = Array([0, 0, 10, 10])
        pattern[Name("/XStep")] = 10
        pattern[Name("/YStep")] = 10

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True

    def test_detect_devicergb_in_tiling_pattern_colorspace(self) -> None:
        """Tiling pattern /Resources/ColorSpace with DeviceRGB is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        pattern = pdf.make_stream(b"/CS0 cs 1 0 0 sc 0 0 10 10 re f")
        pattern[Name.Type] = Name.Pattern
        pattern[Name("/PatternType")] = 1
        pattern[Name("/PaintType")] = 1
        pattern[Name("/TilingType")] = 1
        pattern[Name.BBox] = Array([0, 0, 10, 10])
        pattern[Name("/XStep")] = 10
        pattern[Name("/YStep")] = 10
        pattern[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=Name.DeviceRGB),
        )

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_detect_devicergb_image_in_tiling_pattern(self) -> None:
        """Image XObject in tiling pattern resources is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        pattern = pdf.make_stream(b"q 10 0 0 10 0 0 cm /Im0 Do Q")
        pattern[Name.Type] = Name.Pattern
        pattern[Name("/PatternType")] = 1
        pattern[Name("/PaintType")] = 1
        pattern[Name("/TilingType")] = 1
        pattern[Name.BBox] = Array([0, 0, 10, 10])
        pattern[Name("/XStep")] = 10
        pattern[Name("/YStep")] = 10
        pattern[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=image),
        )

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_detect_shading_inside_tiling_pattern(self) -> None:
        """Shading in tiling pattern resources is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceCMYK,
                Coords=Array([0, 0, 10, 10]),
            )
        )

        pattern = pdf.make_stream(b"/Sh0 sh")
        pattern[Name.Type] = Name.Pattern
        pattern[Name("/PatternType")] = 1
        pattern[Name("/PaintType")] = 1
        pattern[Name("/TilingType")] = 1
        pattern[Name.BBox] = Array([0, 0, 10, 10])
        pattern[Name("/XStep")] = 10
        pattern[Name("/YStep")] = 10
        pattern[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True

    def test_detect_pattern_in_form_xobject(self) -> None:
        """Pattern inside a Form XObject resources is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 50, 50]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        form_stream = pdf.make_stream(b"")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )

        page_dict[Name.Resources] = Dictionary(
            XObject=Dictionary(Fm0=form_stream),
        )
        content = pdf.make_stream(b"q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_detect_shading_in_form_xobject(self) -> None:
        """Shading inside a Form XObject resources is detected."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
                Coords=Array([0, 0, 50, 50]),
            )
        )

        form_stream = pdf.make_stream(b"/Sh0 sh")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

        page_dict[Name.Resources] = Dictionary(
            XObject=Dictionary(Fm0=form_stream),
        )
        content = pdf.make_stream(b"q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True

    def test_tiling_pattern_cycle_detection(self) -> None:
        """Two mutually referencing tiling patterns do not cause infinite loop."""
        pdf = new_pdf()
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )

        # Create two tiling patterns that reference each other
        pat_a = pdf.make_stream(b"0.5 g 0 0 10 10 re f")
        pat_a[Name.Type] = Name.Pattern
        pat_a[Name("/PatternType")] = 1
        pat_a[Name("/PaintType")] = 1
        pat_a[Name("/TilingType")] = 1
        pat_a[Name.BBox] = Array([0, 0, 10, 10])
        pat_a[Name("/XStep")] = 10
        pat_a[Name("/YStep")] = 10

        pat_b = pdf.make_stream(b"1 0 0 rg 0 0 10 10 re f")
        pat_b[Name.Type] = Name.Pattern
        pat_b[Name("/PatternType")] = 1
        pat_b[Name("/PaintType")] = 1
        pat_b[Name("/TilingType")] = 1
        pat_b[Name.BBox] = Array([0, 0, 10, 10])
        pat_b[Name("/XStep")] = 10
        pat_b[Name("/YStep")] = 10

        # Cross-reference: A references B, B references A
        pat_a[Name.Resources] = Dictionary(
            Pattern=Dictionary(PB=pat_b),
        )
        pat_b[Name.Resources] = Dictionary(
            Pattern=Dictionary(PA=pat_a),
        )

        page_dict[Name.Resources] = Dictionary(
            Pattern=Dictionary(PA=pat_a),
        )
        content = pdf.make_stream(b"")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        # Should complete without infinite recursion
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is True


class TestPatternDefaultColorSpaces:
    """Tests for applying default color spaces to tiling pattern resources."""

    @staticmethod
    def _make_tiling_pattern(pdf, content: bytes, resources=None):
        """Create a tiling pattern stream with the given content."""
        pat = pdf.make_stream(content)
        pat[Name.Type] = Name.Pattern
        pat[Name("/PatternType")] = 1
        pat[Name("/PaintType")] = 1
        pat[Name("/TilingType")] = 1
        pat[Name.BBox] = Array([0, 0, 10, 10])
        pat[Name("/XStep")] = 10
        pat[Name("/YStep")] = 10
        if resources is not None:
            pat[Name.Resources] = resources
        else:
            pat[Name.Resources] = Dictionary()
        return pat

    def test_tiling_pattern_gets_default_colorspace(self) -> None:
        """Tiling pattern with RGB operator gets DefaultRGB in resources."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"1 0 0 rg 0 0 10 10 re f")

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        # CMYK on page makes RGB non-dominant.
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_rgb = cs_dict.get(Name.DefaultRGB)
        assert default_rgb is not None
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1]["/N"] == 3

    def test_tiling_pattern_image_uses_contextual_default(self) -> None:
        """An image in a tiling pattern uses that pattern's DefaultRGB."""
        pdf = new_pdf()

        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        pattern = self._make_tiling_pattern(
            pdf,
            b"q 10 0 0 10 0 0 cm /Im0 Do Q",
            resources=Dictionary(XObject=Dictionary(Im0=image)),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert image[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = pattern[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1][Name.N] == 3

    def test_tiling_pattern_in_form_xobject_gets_defaults(self) -> None:
        """Tiling pattern inside a Form XObject gets DefaultRGB."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"1 0 0 rg 0 0 10 10 re f")

        form_stream = pdf.make_stream(b"")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])
        form_stream[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_rgb = cs_dict.get(Name.DefaultRGB)
        assert default_rgb is not None
        assert default_rgb[0] == Name.ICCBased

    def test_tiling_pattern_in_ap_stream_gets_defaults(self) -> None:
        """Tiling pattern inside an AP stream gets DefaultRGB."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"1 0 0 rg 0 0 10 10 re f")

        ap_stream = pdf.make_stream(b"")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 200, 200])
        ap_stream[Name.Resources] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Stamp,
                Rect=Array([0, 0, 200, 200]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_rgb = cs_dict.get(Name.DefaultRGB)
        assert default_rgb is not None
        assert default_rgb[0] == Name.ICCBased

    def test_nested_tiling_pattern_gets_defaults(self) -> None:
        """Nested tiling patterns both get DefaultRGB."""
        pdf = new_pdf()

        inner_pattern = self._make_tiling_pattern(pdf, b"1 0 0 rg 0 0 5 5 re f")
        outer_pattern = self._make_tiling_pattern(
            pdf,
            b"1 0 0 rg 0 0 10 10 re f",
            resources=Dictionary(Pattern=Dictionary(P1=inner_pattern)),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=outer_pattern)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Outer pattern gets DefaultRGB
        outer_res = outer_pattern.get(Name.Resources)
        assert outer_res is not None
        outer_cs = outer_res.get(Name.ColorSpace)
        assert outer_cs is not None
        assert Name.DefaultRGB in outer_cs

        # Inner pattern gets DefaultRGB
        inner_res = inner_pattern.get(Name.Resources)
        assert inner_res is not None
        inner_cs = inner_res.get(Name.ColorSpace)
        assert inner_cs is not None
        assert Name.DefaultRGB in inner_cs

    def test_tiling_pattern_without_resources_gets_defaults(self) -> None:
        """Tiling pattern with no /Resources gets DefaultRGB created."""
        pdf = new_pdf()

        # Create a tiling pattern stream WITHOUT setting /Resources
        pat = pdf.make_stream(b"1 0 0 rg 0 0 10 10 re f")
        pat[Name.Type] = Name.Pattern
        pat[Name("/PatternType")] = 1
        pat[Name("/PaintType")] = 1
        pat[Name("/TilingType")] = 1
        pat[Name.BBox] = Array([0, 0, 10, 10])
        pat[Name("/XStep")] = 10
        pat[Name("/YStep")] = 10
        # Deliberately no /Resources key

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pat)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pat.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_rgb = cs_dict.get(Name.DefaultRGB)
        assert default_rgb is not None
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1]["/N"] == 3

    def test_shading_pattern_uses_contextual_default(self) -> None:
        """A PatternType 2 shading keeps DeviceRGB and uses DefaultRGB."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        # CMYK on page makes RGB non-dominant.
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading_dict[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased


class TestShadingDefaultColorSpaces:
    """Tests for contextual Default color spaces used by shadings."""

    def test_shading_dict_uses_page_default(self) -> None:
        """A page shading keeps DeviceRGB and uses the page DefaultRGB."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=Dictionary(Sh0=shading)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1][Name.N] == 3

    def test_shading_pattern_type2_uses_page_default(self) -> None:
        """A PatternType 2 shading uses its page's DefaultRGB."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading_dict[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    def test_shading_dominant_not_replaced(self) -> None:
        """Shading using dominant color space is left as bare Name."""
        pdf = new_pdf()

        # DeviceRGB shading with RGB dominant — should not be replaced
        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=Dictionary(Sh0=shading)),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert cs == Name.DeviceRGB

    def test_shading_separation_untouched(self) -> None:
        """Separation array in shading is not modified."""
        pdf = new_pdf()

        sep_cs = Array(
            [
                Name.Separation,
                Name("/Spot1"),
                Name.DeviceCMYK,
                pdf.make_stream(b""),  # tint transform placeholder
            ]
        )

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=sep_cs,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=Dictionary(Sh0=shading)),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.Separation

    def test_shading_in_form_xobject_uses_form_default(self) -> None:
        """A Form shading uses the Form's DefaultRGB."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        form_stream = pdf.make_stream(b"")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 200, 200])
        form_stream[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_stream)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = form_stream[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    def test_shading_in_tiling_pattern_uses_pattern_default(self) -> None:
        """A tiling-pattern shading uses the pattern's DefaultRGB."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pat = pdf.make_stream(b"0 0 10 10 re f")
        pat[Name.Type] = Name.Pattern
        pat[Name("/PatternType")] = 1
        pat[Name("/PaintType")] = 1
        pat[Name("/TilingType")] = 1
        pat[Name.BBox] = Array([0, 0, 10, 10])
        pat[Name("/XStep")] = 10
        pat[Name("/YStep")] = 10
        pat[Name.Resources] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pat)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = pat[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    def test_indexed_shading_base_uses_page_default(self) -> None:
        """An Indexed DeviceRGB base remains contextual to DefaultRGB."""
        pdf = new_pdf()

        lookup_data = bytes(range(256)) * 3
        lookup_stream = pdf.make_stream(lookup_data)
        indexed_cs = Array([Name.Indexed, Name.DeviceRGB, 255, lookup_stream])

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=indexed_cs,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=Dictionary(Sh0=shading)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.Indexed
        assert cs[1] == Name.DeviceRGB
        assert int(cs[2]) == 255
        assert cs[3].objgen == lookup_stream.objgen
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    def test_indexed_shading_pattern_type2_uses_page_default(self) -> None:
        """An Indexed shading pattern uses its page's DefaultRGB."""
        pdf = new_pdf()

        lookup_data = bytes(range(256)) * 3
        lookup_stream = pdf.make_stream(lookup_data)
        indexed_cs = Array([Name.Indexed, Name.DeviceRGB, 255, lookup_stream])

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=indexed_cs,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading_dict.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.Indexed
        assert cs[1] == Name.DeviceRGB
        assert int(cs[2]) == 255
        assert cs[3].objgen == lookup_stream.objgen
        default_rgb = pdf.pages[0][Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    @pytest.mark.parametrize("nested", [False, True])
    def test_extgstate_only_type3_font_gets_default_rgb(self, nested) -> None:
        pdf = new_pdf()
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                CharProcs=Dictionary(
                    A=pdf.make_stream(b"1000 0 d0 1 0 0 rg 0 0 100 100 re f")
                ),
                Resources=Dictionary(),
            )
        )
        resources = Dictionary(
            ExtGState=Dictionary(GS=Dictionary(Font=Array([font, 12])))
        )
        page = pdf.add_blank_page()
        if nested:
            form = pdf.make_stream(b"/GS gs BT (A) Tj ET")
            form.Subtype = Name.Form
            form.BBox = Array([0, 0, 100, 100])
            form.Resources = resources
            page.Resources = Dictionary(XObject=Dictionary(Fm=form))
        else:
            page.Resources = resources
        page.Contents = pdf.make_stream(b"0 1 0 0 k 0 0 200 200 re f")

        embed_color_profiles(pdf, "2b")

        default_rgb = font.Resources.ColorSpace.DefaultRGB
        assert default_rgb[0] == Name.ICCBased
        assert default_rgb[1].N == 3

    def test_shading_in_type3_font_uses_font_default(self) -> None:
        """A Type3-font shading uses the font's DefaultRGB."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        charproc_stream = pdf.make_stream(b"0 0 1000 1000 d1")
        font = Dictionary()
        font[Name.Type] = Name.Font
        font[Name.Subtype] = Name.Type3
        font[Name.FontBBox] = Array([0, 0, 1000, 1000])
        font[Name.FontMatrix] = Array([0.001, 0, 0, 0.001, 0, 0])
        font[Name("/CharProcs")] = Dictionary(a=charproc_stream)
        font[Name.Encoding] = Dictionary(
            Type=Name.Encoding,
            Differences=Array([0, Name.a]),
        )
        font[Name("/Resources")] = Dictionary(
            Shading=Dictionary(Sh0=shading),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = font[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased

    def test_pattern_in_type3_font_gets_default(self) -> None:
        """Tiling pattern inside Type3 font /Resources gets defaults."""
        pdf = new_pdf()

        tiling_stream = pdf.make_stream(b"1 0 0 rg 0 0 10 10 re f")
        tiling_stream[Name.Type] = Name.Pattern
        tiling_stream[Name.PatternType] = 1
        tiling_stream[Name.PaintType] = 1
        tiling_stream[Name.TilingType] = 1
        tiling_stream[Name.BBox] = Array([0, 0, 10, 10])
        tiling_stream[Name.XStep] = 10
        tiling_stream[Name.YStep] = 10
        tiling_stream[Name.Resources] = Dictionary()
        tiling = pdf.make_indirect(tiling_stream)

        charproc_stream = pdf.make_stream(b"0 0 1000 1000 d1")
        font = Dictionary()
        font[Name.Type] = Name.Font
        font[Name.Subtype] = Name.Type3
        font[Name.FontBBox] = Array([0, 0, 1000, 1000])
        font[Name.FontMatrix] = Array([0.001, 0, 0, 0.001, 0, 0])
        font[Name("/CharProcs")] = Dictionary(a=charproc_stream)
        font[Name.Encoding] = Dictionary(
            Type=Name.Encoding,
            Differences=Array([0, Name.a]),
        )
        font[Name("/Resources")] = Dictionary(
            Pattern=Dictionary(P0=tiling),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        tiling_res = tiling.get(Name.Resources)
        assert tiling_res is not None
        assert Name.DefaultRGB in tiling_res.get(Name.ColorSpace, {})

    def test_shading_pattern_type2_in_type3_font_uses_font_default(self) -> None:
        """A Type3 shading pattern uses the font's DefaultRGB."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceRGB,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        pattern = pdf.make_indirect(
            Dictionary(
                Type=Name.Pattern,
                PatternType=2,
                Shading=shading_dict,
            )
        )

        charproc_stream = pdf.make_stream(b"0 0 1000 1000 d1")
        font = Dictionary()
        font[Name.Type] = Name.Font
        font[Name.Subtype] = Name.Type3
        font[Name.FontBBox] = Array([0, 0, 1000, 1000])
        font[Name.FontMatrix] = Array([0.001, 0, 0, 0.001, 0, 0])
        font[Name("/CharProcs")] = Dictionary(a=charproc_stream)
        font[Name.Encoding] = Dictionary(
            Type=Name.Encoding,
            Differences=Array([0, Name.a]),
        )
        font[Name("/Resources")] = Dictionary(
            Pattern=Dictionary(P0=pattern),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
        content = pdf.make_stream(b"0 1 0 0 k 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        assert shading_dict[Name.ColorSpace] == Name.DeviceRGB
        default_rgb = font[Name.Resources][Name.ColorSpace][Name.DefaultRGB]
        assert default_rgb[0] == Name.ICCBased


class TestCalibratedColorSpaces:
    """Tests for CalGray, CalRGB, and Lab color space recognition."""

    # --- _parse_colorspace_array tests ---

    def test_parse_calgray_array(self):
        """_parse_colorspace_array recognizes CalGray."""
        cs = Array([Name.CalGray, Dictionary(WhitePoint=Array([1, 1, 1]))])
        cs_type, alternate = _parse_colorspace_array(cs)
        assert cs_type == "CalGray"
        assert alternate is None

    def test_parse_calrgb_array(self):
        """_parse_colorspace_array recognizes CalRGB."""
        cs = Array([Name.CalRGB, Dictionary(WhitePoint=Array([1, 1, 1]))])
        cs_type, alternate = _parse_colorspace_array(cs)
        assert cs_type == "CalRGB"
        assert alternate is None

    def test_parse_lab_array(self):
        """_parse_colorspace_array recognizes Lab."""
        cs = Array([Name.Lab, Dictionary(WhitePoint=Array([1, 1, 1]))])
        cs_type, alternate = _parse_colorspace_array(cs)
        assert cs_type == "Lab"
        assert alternate is None

    def test_parse_1200_nested_indexed_arrays_without_recursion(self):
        """Malformed deep Indexed bases cannot exhaust the Python stack."""
        cs = Array([Name.DeviceRGB])
        for _ in range(1200):
            cs = Array([Name.Indexed, cs])

        cs_type, alternate = _parse_colorspace_array(cs)

        assert cs_type == "Indexed"
        assert alternate == "Indexed"

    # --- _analyze_colorspace tests ---

    def test_analyze_calgray(self):
        """_analyze_colorspace sets cal_gray_used."""
        cs = Array([Name.CalGray, Dictionary(WhitePoint=Array([1, 1, 1]))])
        analysis = ColorSpaceAnalysis()
        _analyze_colorspace(cs, analysis, "test")
        assert analysis.cal_gray_used is True
        assert analysis.cal_rgb_used is False
        assert analysis.lab_used is False

    def test_analyze_calrgb(self):
        """_analyze_colorspace sets cal_rgb_used."""
        cs = Array([Name.CalRGB, Dictionary(WhitePoint=Array([1, 1, 1]))])
        analysis = ColorSpaceAnalysis()
        _analyze_colorspace(cs, analysis, "test")
        assert analysis.cal_gray_used is False
        assert analysis.cal_rgb_used is True
        assert analysis.lab_used is False

    def test_analyze_lab(self):
        """_analyze_colorspace sets lab_used."""
        cs = Array([Name.Lab, Dictionary(WhitePoint=Array([1, 1, 1]))])
        analysis = ColorSpaceAnalysis()
        _analyze_colorspace(cs, analysis, "test")
        assert analysis.cal_gray_used is False
        assert analysis.cal_rgb_used is False
        assert analysis.lab_used is True

    # --- calibrated_spaces property tests ---

    def test_calibrated_spaces_empty(self):
        """calibrated_spaces is empty when no calibrated spaces detected."""
        analysis = ColorSpaceAnalysis()
        assert analysis.calibrated_spaces == set()

    def test_calibrated_spaces_calgray(self):
        """calibrated_spaces includes CAL_GRAY."""
        analysis = ColorSpaceAnalysis(cal_gray_used=True)
        assert analysis.calibrated_spaces == {ColorSpaceType.CAL_GRAY}

    def test_calibrated_spaces_calrgb(self):
        """calibrated_spaces includes CAL_RGB."""
        analysis = ColorSpaceAnalysis(cal_rgb_used=True)
        assert analysis.calibrated_spaces == {ColorSpaceType.CAL_RGB}

    def test_calibrated_spaces_lab(self):
        """calibrated_spaces includes LAB."""
        analysis = ColorSpaceAnalysis(lab_used=True)
        assert analysis.calibrated_spaces == {ColorSpaceType.LAB}

    def test_calibrated_spaces_all(self):
        """calibrated_spaces includes all three when all detected."""
        analysis = ColorSpaceAnalysis(
            cal_gray_used=True,
            cal_rgb_used=True,
            lab_used=True,
        )
        assert analysis.calibrated_spaces == {
            ColorSpaceType.CAL_GRAY,
            ColorSpaceType.CAL_RGB,
            ColorSpaceType.LAB,
        }

    # --- detected_spaces unaffected ---

    def test_detected_spaces_unaffected_by_calibrated(self):
        """detected_spaces does not include calibrated color spaces."""
        analysis = ColorSpaceAnalysis(
            cal_gray_used=True,
            cal_rgb_used=True,
            lab_used=True,
        )
        assert analysis.detected_spaces == set()

    def test_detected_spaces_independent_of_calibrated(self):
        """detected_spaces and calibrated_spaces are independent."""
        analysis = ColorSpaceAnalysis(
            device_rgb_used=True,
            cal_gray_used=True,
        )
        assert analysis.detected_spaces == {ColorSpaceType.DEVICE_RGB}
        assert analysis.calibrated_spaces == {ColorSpaceType.CAL_GRAY}

    # --- Integration: detect_color_spaces with calibrated images ---

    def test_detect_calgray_in_image(self):
        """detect_color_spaces finds CalGray in image XObject."""
        pdf = new_pdf()
        cal_gray_cs = Array(
            [
                Name.CalGray,
                Dictionary(WhitePoint=Array([1, 1, 1])),
            ]
        )
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = cal_gray_cs
        image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image)),
        )
        page_dict[Name.Contents] = pdf.make_stream(
            b"q 100 0 0 100 100 600 cm /Im0 Do Q"
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        analysis = detect_color_spaces(pdf)
        assert analysis.cal_gray_used is True
        assert analysis.detected_spaces == set()

    def test_detect_calrgb_in_image(self):
        """detect_color_spaces finds CalRGB in image XObject."""
        pdf = new_pdf()
        cal_rgb_cs = Array(
            [
                Name.CalRGB,
                Dictionary(WhitePoint=Array([1, 1, 1])),
            ]
        )
        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = cal_rgb_cs
        image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image)),
        )
        page_dict[Name.Contents] = pdf.make_stream(
            b"q 100 0 0 100 100 600 cm /Im0 Do Q"
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        analysis = detect_color_spaces(pdf)
        assert analysis.cal_rgb_used is True
        assert analysis.detected_spaces == set()

    def test_detect_lab_in_image(self):
        """detect_color_spaces finds Lab in image XObject."""
        pdf = new_pdf()
        lab_cs = Array(
            [
                Name.Lab,
                Dictionary(WhitePoint=Array([1, 1, 1])),
            ]
        )
        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = lab_cs
        image[Name.BitsPerComponent] = 8

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=image)),
        )
        page_dict[Name.Contents] = pdf.make_stream(
            b"q 100 0 0 100 100 600 cm /Im0 Do Q"
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        analysis = detect_color_spaces(pdf)
        assert analysis.lab_used is True
        assert analysis.detected_spaces == set()


class TestCalibratedPreservation:
    """PDF/A conversion must retain the actual CalGray/CalRGB calibration."""

    @staticmethod
    def _colorspace(kind):
        """Build a calibrated space with nondefault gamma and colorimetry."""
        params = Dictionary(
            WhitePoint=Array([0.9505, 1, 1.089]),
            BlackPoint=Array([0, 0, 0]),
        )
        if kind == "CalGray":
            params.Gamma = 1.6
        else:
            params.Gamma = Array([1.3, 1.8, 2.2])
            params.Matrix = Array(
                [0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505]
            )
        return Array([Name("/" + kind), params])

    @staticmethod
    def _add_colorspace(pdf, cs, location):
        """Install a calibrated space in the requested nested resource context."""
        page = pdf.add_blank_page()
        if location == "named":
            page.Resources.ColorSpace = Dictionary(CS0=cs)
            return page.Resources.ColorSpace, Name.CS0
        if location in ("image", "indexed_image", "form"):
            image = pdf.make_stream(b"\x80\x40\x20")
            image.Type = Name.XObject
            image.Subtype = Name.Image
            image.Width = 1
            image.Height = 1
            image.BitsPerComponent = 8
            image.ColorSpace = cs
            if location == "indexed_image":
                image.ColorSpace = Array([Name.Indexed, cs, 0, "palette"])
            if location == "form":
                form = pdf.make_stream(b"/Im0 Do")
                form.Type = Name.XObject
                form.Subtype = Name.Form
                form.BBox = Array([0, 0, 100, 100])
                form.Resources = Dictionary(XObject=Dictionary(Im0=image))
                page.Resources.XObject = Dictionary(Fm0=form)
            else:
                page.Resources.XObject = Dictionary(Im0=image)
            if location == "indexed_image":
                return image.ColorSpace, 1
            return image, Name.ColorSpace
        if location == "shading":
            shading = Dictionary(ShadingType=2, ColorSpace=cs)
            page.Resources.Shading = Dictionary(Sh0=shading)
            return page.Resources.Shading.Sh0, Name.ColorSpace
        if location == "type3":
            font = pdf.make_indirect(
                Dictionary(
                    Type=Name.Font,
                    Subtype=Name.Type3,
                    Resources=Dictionary(ColorSpace=Dictionary(CS0=cs)),
                )
            )
            page.Resources.Font = Dictionary(F0=font)
            return font.Resources.ColorSpace, Name.CS0
        if location == "page_group":
            page.Group = Dictionary(S=Name.Transparency, CS=cs)
            return page.Group, Name.CS
        form = pdf.make_stream(b"q Q")
        form.Type = Name.XObject
        form.Subtype = Name.Form
        form.BBox = Array([0, 0, 100, 100])
        form.Group = Dictionary(S=Name.Transparency, CS=cs)
        if location == "form_group":
            page.Resources.XObject = Dictionary(Fm0=form)
        else:
            page.Annots = Array(
                [
                    pdf.make_indirect(
                        Dictionary(
                            Type=Name.Annot,
                            Subtype=Name.Stamp,
                            Rect=Array([0, 0, 100, 100]),
                            AP=Dictionary(N=form),
                        )
                    )
                ]
            )
        return form.Group, Name.CS

    @pytest.mark.parametrize("kind", ["CalGray", "CalRGB"])
    @pytest.mark.parametrize(
        "location",
        [
            "named",
            "image",
            "indexed_image",
            "form",
            "shading",
            "type3",
            "page_group",
            "form_group",
            "appearance_group",
        ],
    )
    @pytest.mark.parametrize("option", [None, False, True])
    def test_embed_preserves_calibration(self, kind, location, option, caplog):
        """No option may replace valid calibration with a generic ICC profile."""
        pdf = new_pdf()
        cs = self._colorspace(kind)
        expected = cs.unparse()
        owner, key = self._add_colorspace(pdf, cs, location)
        kwargs = {} if option is None else {"convert_calibrated": option}

        embed_color_profiles(pdf, "3b", **kwargs)

        assert owner[key].unparse() == expected
        assert ("convert_calibrated=True is deprecated" in caplog.text) is (
            option is True
        )

    def test_legacy_helper_preserves_pdf_and_cache(self, caplog):
        """The re-exported compatibility function reports no replacements."""
        pdf = new_pdf()
        owner, key = self._add_colorspace(pdf, self._colorspace("CalRGB"), "image")
        expected = owner[key].unparse()
        cache = {ColorSpaceType.DEVICE_GRAY: pdf.make_stream(b"existing profile")}
        existing_stream = cache[ColorSpaceType.DEVICE_GRAY].objgen
        object_count = len(pdf.objects)

        assert _convert_calibrated_colorspaces(pdf, cache) == 0

        assert owner[key].unparse() == expected
        assert len(pdf.objects) == object_count
        assert set(cache) == {ColorSpaceType.DEVICE_GRAY}
        assert cache[ColorSpaceType.DEVICE_GRAY].objgen == existing_stream
        assert "are preserved" in caplog.text

    @pytest.mark.parametrize("kind", ["CalGray", "CalRGB"])
    @pytest.mark.parametrize("level", ["2b", "3b"])
    @pytest.mark.parametrize("legacy_opt_in", [False, True])
    def test_converter_preserves_calibration_and_color_values(
        self, tmp_path, kind, level, legacy_opt_in
    ):
        """Exercise the default and legacy opt-in through the full pipeline."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = new_pdf()
        cs = self._colorspace(kind)
        expected = cs.unparse()
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources.ColorSpace = Dictionary(CS0=cs)
        values = b"0.5" if kind == "CalGray" else b"0.5 0.25 0.1"
        page.Contents = pdf.make_stream(
            b"q /CS0 cs " + values + b" sc 0 0 100 100 re f Q"
        )
        original_content = pikepdf.unparse_content_stream(
            pikepdf.parse_content_stream(page)
        )
        source = tmp_path / "calibrated.pdf"
        target = tmp_path / "calibrated-pdfa.pdf"
        pdf.save(source)
        kwargs = {"convert_calibrated": True} if legacy_opt_in else {}

        result = convert_to_pdfa(source, target, level, **kwargs)

        assert result.success
        assert result.target_produced
        assert not result.skipped
        with Pdf.open(target) as converted:
            output_page = converted.pages[0]
            assert output_page.Resources.ColorSpace.CS0.unparse() == expected
            assert (
                pikepdf.unparse_content_stream(
                    pikepdf.parse_content_stream(output_page)
                )
                == original_content
            )


class TestApplyDefaultsToApEntryNoResources:
    """Tests for _apply_defaults_to_ap_entry with missing Resources."""

    def _make_icc_arrays(self, pdf):
        """Build ICC arrays for all device color spaces."""
        cache = {}
        result = {}
        for cs_type in (
            ColorSpaceType.DEVICE_GRAY,
            ColorSpaceType.DEVICE_RGB,
            ColorSpaceType.DEVICE_CMYK,
        ):
            result[cs_type] = _create_icc_colorspace(pdf, cs_type, cache)
        return result

    def test_direct_stream_without_resources_gets_defaults(self):
        """AP stream without /Resources gets Resources + DefaultXxx."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_GRAY, ColorSpaceType.DEVICE_RGB}

        ap_stream = pdf.make_stream(b"0.5 g 1 0 0 rg")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 10, 10])
        # No /Resources set

        visited = set()
        defaults_added, _ = _apply_defaults_to_ap_entry(
            ap_stream,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 2
        resources = ap_stream.get(Name.Resources)
        assert resources is not None
        cs = resources.get(Name.ColorSpace)
        assert Name.DefaultGray in cs
        assert Name.DefaultRGB in cs

    def test_direct_stream_with_existing_resources_unchanged(self):
        """AP stream with existing /Resources keeps them and adds defaults."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_GRAY}

        ap_stream = pdf.make_stream(b"0.5 g")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 10, 10])
        ap_stream[Name.Resources] = Dictionary()

        visited = set()
        defaults_added, _ = _apply_defaults_to_ap_entry(
            ap_stream,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 1
        cs = ap_stream[Name.Resources].get(Name.ColorSpace)
        assert Name.DefaultGray in cs

    def test_sub_state_dict_streams_without_resources(self):
        """Sub-state dict streams without /Resources get defaults."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_GRAY}

        off_stream = pdf.make_stream(b"1 g")
        off_stream[Name.Type] = Name.XObject
        off_stream[Name.Subtype] = Name.Form
        off_stream[Name.BBox] = Array([0, 0, 10, 10])

        on_stream = pdf.make_stream(b"0 g")
        on_stream[Name.Type] = Name.XObject
        on_stream[Name.Subtype] = Name.Form
        on_stream[Name.BBox] = Array([0, 0, 10, 10])

        state_dict = Dictionary()
        state_dict[Name("/Off")] = pdf.make_indirect(off_stream)
        state_dict[Name("/Yes")] = pdf.make_indirect(on_stream)

        visited = set()
        defaults_added, _ = _apply_defaults_to_ap_entry(
            state_dict,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 2
        for stream in (off_stream, on_stream):
            resources = stream.get(Name.Resources)
            assert resources is not None
            cs = resources.get(Name.ColorSpace)
            assert Name.DefaultGray in cs


class TestInlineImageIndexedArrayBase:
    """Tests for inline image Indexed base color space as Array."""

    def test_indexed_iccbased_rgb_base_detected(self):
        """Indexed inline image with [/ICCBased stream] base -> RGB detected."""
        pdf = new_pdf()

        # Create an ICCBased stream with N=3 (RGB)
        from pdftopdfa.color_profile import get_srgb_profile

        icc_data = get_srgb_profile()
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 3

        # Build page with an inline image that uses Indexed with array base
        # We'll test via the detection function on a page with a named CS
        icc_cs = Array([Name.ICCBased, icc_stream])
        indexed_cs = Array(
            [
                Name.Indexed,
                icc_cs,
                1,
                pikepdf.String(b"\xff\x00\x00\x00\xff\x00"),
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        analysis = detect_color_spaces(pdf)
        # The ICCBased base should not flag device color directly
        # but the color space should be detected without error
        assert analysis is not None

    def test_indexed_iccbased_gray_base_inline(self):
        """_get_inline_image_device_cs resolves Indexed [/ICCBased N=1]."""
        from pdftopdfa.color_profile import get_gray_profile
        from pdftopdfa.color_profile._detection import _get_inline_image_device_cs

        pdf = new_pdf()
        icc_data = get_gray_profile()
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 1

        icc_cs = Array([Name.ICCBased, icc_stream])

        # Simulate an inline image object with raw /ColorSpace array
        class FakeInlineImage:
            def __init__(self, cs_array):
                self.obj = Dictionary()
                self.obj[Name.ColorSpace] = cs_array

            @property
            def colorspace(self):
                raise NotImplementedError

        indexed_array = Array([Name.Indexed, icc_cs, 1, pikepdf.String(b"\x00\xff")])
        img = FakeInlineImage(indexed_array)
        result = _get_inline_image_device_cs(img)
        assert result == Name.DeviceGray

    def test_indexed_iccbased_cmyk_base_inline(self):
        """_get_inline_image_device_cs resolves Indexed [/ICCBased N=4]."""
        from pdftopdfa.color_profile import get_cmyk_profile
        from pdftopdfa.color_profile._detection import _get_inline_image_device_cs

        pdf = new_pdf()
        icc_data = get_cmyk_profile()
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 4

        icc_cs = Array([Name.ICCBased, icc_stream])

        class FakeInlineImage:
            def __init__(self, cs_array):
                self.obj = Dictionary()
                self.obj[Name.ColorSpace] = cs_array

            @property
            def colorspace(self):
                raise NotImplementedError

        indexed_array = Array(
            [
                Name.Indexed,
                icc_cs,
                0,
                pikepdf.String(b"\x00\x00\x00\x00"),
            ]
        )
        img = FakeInlineImage(indexed_array)
        result = _get_inline_image_device_cs(img)
        assert result == Name.DeviceCMYK

    def test_indexed_calrgb_base_inline(self):
        """_get_inline_image_device_cs resolves Indexed [/CalRGB dict]."""
        from pdftopdfa.color_profile._detection import _get_inline_image_device_cs

        new_pdf()
        cal_rgb = Array(
            [
                Name.CalRGB,
                Dictionary(WhitePoint=Array([0.9505, 1.0, 1.089])),
            ]
        )

        class FakeInlineImage:
            def __init__(self, cs_array):
                self.obj = Dictionary()
                self.obj[Name.ColorSpace] = cs_array

            @property
            def colorspace(self):
                raise NotImplementedError

        indexed_array = Array(
            [
                Name.Indexed,
                cal_rgb,
                0,
                pikepdf.String(b"\xff\x00\x00"),
            ]
        )
        img = FakeInlineImage(indexed_array)
        result = _get_inline_image_device_cs(img)
        assert result == Name.DeviceRGB


class TestSMaskGFormDefaultColorSpaces:
    """Tests for default color spaces in SMask /G Form XObjects."""

    def _make_icc_arrays(self, pdf):
        """Build ICC arrays for all device color spaces."""
        cache = {}
        result = {}
        for cs_type in (
            ColorSpaceType.DEVICE_GRAY,
            ColorSpaceType.DEVICE_RGB,
            ColorSpaceType.DEVICE_CMYK,
        ):
            result[cs_type] = _create_icc_colorspace(pdf, cs_type, cache)
        return result

    def test_smask_g_form_gets_default_colorspaces(self):
        """SMask /G Form XObject gets DefaultGray added."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_GRAY}

        # Create an SMask /G Form XObject
        g_form = pdf.make_stream(b"0.5 g")
        g_form[Name.Type] = Name.XObject
        g_form[Name.Subtype] = Name.Form
        g_form[Name.BBox] = Array([0, 0, 100, 100])
        g_form[Name.Group] = Dictionary(
            S=Name.Transparency,
            CS=Name.DeviceGray,
        )

        smask_dict = Dictionary(
            S=Name.Luminosity,
            G=g_form,
        )
        gs = Dictionary(SMask=smask_dict)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(ExtGState=Dictionary(GS0=gs)),
            )
        )
        pdf.pages.append(page)

        from pdftopdfa.color_profile._defaults import (
            _apply_defaults_to_smask_groups,
        )

        visited = set()
        defaults_added, _ = _apply_defaults_to_smask_groups(
            pdf.pages[0].Resources,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 1
        form_resources = g_form.get(Name.Resources)
        assert form_resources is not None
        cs = form_resources.get(Name.ColorSpace)
        assert Name.DefaultGray in cs

    def test_smask_g_form_without_resources_gets_them_created(self):
        """SMask /G Form XObject without /Resources gets Resources created."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_RGB}

        g_form = pdf.make_stream(b"1 0 0 rg")
        g_form[Name.Type] = Name.XObject
        g_form[Name.Subtype] = Name.Form
        g_form[Name.BBox] = Array([0, 0, 50, 50])
        # No /Resources set

        smask_dict = Dictionary(S=Name.Alpha, G=g_form)
        gs = Dictionary(SMask=smask_dict)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(ExtGState=Dictionary(GS0=gs)),
            )
        )
        pdf.pages.append(page)

        from pdftopdfa.color_profile._defaults import (
            _apply_defaults_to_smask_groups,
        )

        visited = set()
        defaults_added, _ = _apply_defaults_to_smask_groups(
            pdf.pages[0].Resources,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 1
        form_resources = g_form.get(Name.Resources)
        assert form_resources is not None
        cs = form_resources.get(Name.ColorSpace)
        assert Name.DefaultRGB in cs

    def test_smask_g_form_image_uses_contextual_default(self):
        """An image in an SMask Form uses the Form's DefaultGray."""
        pdf = new_pdf()
        icc_arrays = self._make_icc_arrays(pdf)
        non_dominant = {ColorSpaceType.DEVICE_GRAY}

        # Create an image with DeviceGray inside the /G form
        img = pdf.make_stream(b"\x80")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.BitsPerComponent] = 8
        img[Name.ColorSpace] = Name.DeviceGray

        g_form = pdf.make_stream(b"/Im0 Do")
        g_form[Name.Type] = Name.XObject
        g_form[Name.Subtype] = Name.Form
        g_form[Name.BBox] = Array([0, 0, 100, 100])
        g_form[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=img),
        )

        smask_dict = Dictionary(S=Name.Luminosity, G=g_form)
        gs = Dictionary(SMask=smask_dict)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(ExtGState=Dictionary(GS0=gs)),
            )
        )
        pdf.pages.append(page)

        from pdftopdfa.color_profile._defaults import (
            _apply_defaults_to_smask_groups,
        )

        visited = set()
        defaults_added, replacements = _apply_defaults_to_smask_groups(
            pdf.pages[0].Resources,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert defaults_added == 1
        assert replacements == 0
        assert img[Name.ColorSpace] == Name.DeviceGray
        default_gray = g_form[Name.Resources][Name.ColorSpace][Name.DefaultGray]
        assert default_gray[0] == Name.ICCBased

    def test_output_intent_covers_smask_gray(self):
        """Any PDF/A OutputIntent covers DeviceGray inside an SMask Form."""
        pdf = new_pdf()

        g_form = pdf.make_stream(b"0.5 g")
        g_form[Name.Type] = Name.XObject
        g_form[Name.Subtype] = Name.Form
        g_form[Name.BBox] = Array([0, 0, 100, 100])

        smask_dict = Dictionary(S=Name.Luminosity, G=g_form)
        gs = Dictionary(SMask=smask_dict)

        # Page uses RGB (dominant) and has SMask /G using gray
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    ExtGState=Dictionary(GS0=gs),
                ),
                Contents=pdf.make_stream(b"1 0 0 rg 0.5 g"),
            )
        )
        pdf.pages.append(page)

        embed_color_profiles(pdf, "3b")

        assert len(pdf.Root[Name.OutputIntents]) == 1
        assert g_form.get(Name.Resources) is None

    def test_pdfx_dest_output_profile_ref_removed(self):
        """Rule 6.2.3-3: DestOutputProfileRef removed from PDF/X intents."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=pdf.make_stream(b"1 0 0 rg"),
            )
        )
        pdf.pages.append(page)

        oi = Dictionary(
            Type=Name.OutputIntent,
            S=Name("/GTS_PDFX"),
            OutputConditionIdentifier=pikepdf.String("FOGRA39"),
            DestOutputProfileRef=Dictionary(
                URL=pikepdf.String("http://example.com/profile.icc"),
            ),
        )
        oi_ref = pdf.make_indirect(oi)
        pdf.Root.OutputIntents = Array([oi_ref])

        embed_color_profiles(pdf, "2b", replace_existing=True)

        # The forbidden reference is removed while the registered condition
        # remains alongside the newly added PDF/A intent.
        assert "/DestOutputProfileRef" not in oi_ref
        assert len(pdf.Root.OutputIntents) == 2
        assert pdf.Root.OutputIntents[0].objgen == oi_ref.objgen
        assert pdf.Root.OutputIntents[1].S == Name.GTS_PDFA1

    def test_pdfa_dest_output_profile_ref_not_removed(self):
        """DestOutputProfileRef is kept on non-PDF/X OutputIntents."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=pdf.make_stream(b"1 0 0 rg"),
            )
        )
        pdf.pages.append(page)

        oi = Dictionary(
            Type=Name.OutputIntent,
            S=Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfileRef=Dictionary(
                URL=pikepdf.String("http://example.com/profile.icc"),
            ),
        )
        oi_ref = pdf.make_indirect(oi)
        pdf.Root.OutputIntents = Array([oi_ref])

        embed_color_profiles(pdf, "2b", replace_existing=True)

        # Non-PDF/X intents should not have DestOutputProfileRef stripped.
        assert "/DestOutputProfileRef" in oi_ref


class TestPreservationBoundaries:
    """Exercise preservation across repeated passes and nested color declarations."""

    @pytest.mark.parametrize("kind", [Name.CalGray, Name.CalRGB])
    @pytest.mark.parametrize(
        "wrapper", ["pattern", "separation", "devicen", "process", "colorant", "icc"]
    )
    @pytest.mark.parametrize("valid", [False, True])
    def test_nested_calibration_is_validated_without_rewriting(
        self, kind, wrapper, valid
    ):
        """Alternate spaces and DeviceN attributes are declared color contexts."""
        from pdftopdfa.color_profile._calibrated import validate_calibrated_colorspaces

        pdf = new_pdf()
        page = pdf.add_blank_page()
        params = Dictionary(WhitePoint=Array([1, 1, 1])) if valid else Dictionary()
        calibrated = Array([kind, params])
        tint = Dictionary(FunctionType=2, Domain=Array([0, 1]), N=1)
        if wrapper == "pattern":
            space = Array([Name.Pattern, calibrated])
        elif wrapper == "separation":
            space = Array([Name.Separation, Name.Spot, calibrated, tint])
        elif wrapper == "icc":
            profile = pdf.make_stream(b"profile")
            profile.N = 3
            profile.Alternate = calibrated
            space = Array([Name.ICCBased, profile])
        else:
            attrs = Dictionary()
            alternate = calibrated if wrapper == "devicen" else Name.DeviceRGB
            if wrapper == "process":
                attrs.Process = Dictionary(ColorSpace=calibrated)
            elif wrapper == "colorant":
                attrs.Colorants = Dictionary(
                    Spot=Array([Name.Separation, Name.Spot, calibrated, tint])
                )
            space = Array([Name.DeviceN, Array([Name.Spot]), alternate, tint, attrs])
        page.Resources.ColorSpace = Dictionary(CS=space)
        before = space.unparse()
        if valid:
            validate_calibrated_colorspaces(pdf)
        else:
            with pytest.raises(ConversionError, match="invalid /WhitePoint"):
                validate_calibrated_colorspaces(pdf)
        assert space.unparse() == before

    @pytest.mark.parametrize("existing_pdfa", [False, True])
    def test_output_intent_preparation_is_idempotent(self, existing_pdfa):
        """Repeated preparation preserves intent count and profile identity."""
        from pdftopdfa.color_profile._output_intents import (
            prepare_existing_output_intents,
        )

        pdf = new_pdf()
        pdf.add_blank_page()
        profile_data = get_srgb_profile()
        pdfx = pdf.make_indirect(
            Dictionary(
                S=Name.GTS_PDFX,
                OutputConditionIdentifier="Original printing condition",
                Info="Keep this printing metadata",
                DestOutputProfile=pdf.make_stream(profile_data),
            )
        )
        pdf.Root.OutputIntents = Array([pdfx])
        if existing_pdfa:
            pdf.Root.OutputIntents.append(
                pdf.make_indirect(
                    Dictionary(
                        S=Name.GTS_PDFA1,
                        OutputConditionIdentifier="Archival condition",
                        DestOutputProfile=pdf.make_stream(profile_data),
                    )
                )
            )
        first = prepare_existing_output_intents(pdf)
        assert first.keep_existing
        assert first.document_colorspace == ColorSpaceType.DEVICE_RGB
        assert len(pdf.Root.OutputIntents) == 2
        before = pdf.Root.OutputIntents.unparse()
        second = prepare_existing_output_intents(pdf)
        assert second == first
        assert pdf.Root.OutputIntents.unparse() == before
        pdfa = pdf.Root.OutputIntents[1]
        assert pdfa.S == Name.GTS_PDFA1
        assert pdfx.DestOutputProfile.objgen == pdfa.DestOutputProfile.objgen
        assert pdfa.DestOutputProfile.read_bytes() == profile_data
        assert str(pdfx.Info) == "Keep this printing metadata"
        assert "/Type" not in pdfx
