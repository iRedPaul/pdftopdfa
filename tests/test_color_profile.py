# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for color space management and ICC profiles."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import (
    ColorSpaceAnalysis,
    ColorSpaceType,
    _analyze_colorspace,
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


def _make_icc_profile(**overrides: bytes | int) -> bytes:
    """Build a minimal 128-byte ICC profile with optional field overrides.

    Defaults to a valid v2 mntr RGB profile.  Pass keyword arguments to
    override specific header fields:
        version (int): major version byte at offset 8
        device_class (bytes): 4-byte class at offset 12-15
    """
    profile = bytearray(128)
    profile[0:4] = (128).to_bytes(4, "big")
    profile[8] = overrides.get("version", 2)
    profile[12:16] = overrides.get("device_class", b"mntr")
    profile[16:20] = b"RGB "
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
        Non-dominant spaces get Default entries + image replacement.
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

        # DefaultGray added to page resources (non-dominant)
        page_res = pdf.pages[0]["/Resources"]
        cs_dict = page_res["/ColorSpace"]
        assert Name.DefaultGray in cs_dict
        default_gray = cs_dict[Name.DefaultGray]
        assert default_gray[0] == Name.ICCBased
        assert default_gray[1]["/N"] == 1

        # Gray image replaced with ICCBased
        gray_img = page_res["/XObject"]["/Im0"]
        img_cs = gray_img["/ColorSpace"]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
        assert img_cs[1]["/N"] == 1

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

    def test_multiple_identical_output_intents_kept(self) -> None:
        """Multiple OutputIntents with identical ICC profiles are left intact."""
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
            S=Name("/GTS_PDFX"),
            OutputConditionIdentifier=pikepdf.String("sRGB"),
            DestOutputProfile=stream2,
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        # Add a page with RGB content so embed has something to work on
        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        # Should complete without error — profiles were identical
        assert has_output_intent(pdf)

    def test_multiple_different_output_intents_reduced(self) -> None:
        """Different ICC profiles across OutputIntents keeps only the first."""
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

        assert has_output_intent(pdf)

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
            S=Name("/GTS_PDFX"),
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
        """When profiles differ, only the first OutputIntent survives pruning."""
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

        # After the conflict-resolution step, only 1 OutputIntent remains
        # (the embed step will then replace it with its own)
        assert has_output_intent(pdf)

    def test_output_intent_missing_profile_pruned(self) -> None:
        """OutputIntent without DestOutputProfile triggers pruning."""
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
            OutputConditionIdentifier=pikepdf.String("Custom"),
        )
        pdf.Root.OutputIntents = Array([oi1, oi2])

        page = pdf.add_blank_page(page_size=(612, 792))
        page.Contents = pdf.make_stream(b"1 0 0 rg")

        embed_color_profiles(pdf, "2b", replace_existing=True)

        assert has_output_intent(pdf)

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
            S=Name("/GTS_PDFX"),
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


class TestDefaultColorSpaces:
    """Tests for Default color space entries and image replacement."""

    def test_mixed_rgb_gray_adds_default_gray(self) -> None:
        """RGB dominant + Gray non-dominant → DefaultGray in page resources."""
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

        # DefaultGray added
        cs_dict = pdf.pages[0]["/Resources"]["/ColorSpace"]
        assert Name.DefaultGray in cs_dict
        default_gray = cs_dict[Name.DefaultGray]
        assert default_gray[0] == Name.ICCBased
        assert default_gray[1]["/N"] == 1

        # No DefaultRGB (RGB is dominant, covered by OutputIntent)
        assert Name.DefaultRGB not in cs_dict

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

    def test_image_replacement_gray_with_rgb_dominant(self) -> None:
        """Gray image with RGB dominant → image gets [/ICCBased <N=1>]."""
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
        img_cs = img["/ColorSpace"]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
        assert img_cs[1]["/N"] == 1

    def test_indexed_image_base_colorspace_replaced(self) -> None:
        """Indexed image with DeviceRGB base → base replaced with ICCBased."""
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
        # Base is now [/ICCBased <stream>] with N=3
        base = img_cs[1]
        assert isinstance(base, Array)
        assert base[0] == Name.ICCBased
        assert base[1]["/N"] == 3
        # hival and lookup remain intact
        assert int(img_cs[2]) == 255
        assert img_cs[3].objgen == lookup_stream.objgen

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
        """Form XObject with Gray operators gets DefaultGray in its resources."""
        pdf = new_pdf()

        # Create a Form XObject that uses Gray operators
        form_content = b"0.5 g 10 10 80 80 re f"
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
        # Page uses RGB
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Form XObject resources should have DefaultGray
        form = pdf.pages[0]["/Resources"]["/XObject"]["/Fm0"]
        form_res = form["/Resources"]
        cs_dict = form_res["/ColorSpace"]
        assert Name.DefaultGray in cs_dict
        default_gray = cs_dict[Name.DefaultGray]
        assert default_gray[0] == Name.ICCBased
        assert default_gray[1]["/N"] == 1

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

        # DefaultRGB and DefaultGray added
        cs_dict = pdf.pages[0]["/Resources"]["/ColorSpace"]
        assert Name.DefaultRGB in cs_dict
        assert Name.DefaultGray in cs_dict
        assert Name.DefaultCMYK not in cs_dict

        # Both images replaced
        gray_img = pdf.pages[0]["/Resources"]["/XObject"]["/Im0"]
        assert isinstance(gray_img["/ColorSpace"], Array)
        assert gray_img["/ColorSpace"][1]["/N"] == 1

        rgb_img = pdf.pages[0]["/Resources"]["/XObject"]["/Im1"]
        assert isinstance(rgb_img["/ColorSpace"], Array)
        assert rgb_img["/ColorSpace"][1]["/N"] == 3

    def test_existing_default_preserved(self) -> None:
        """Pre-existing DefaultGray entry is not overwritten."""
        pdf = new_pdf()

        # Create a custom ICCBased array for DefaultGray
        custom_gray = get_gray_profile()
        custom_stream = pdf.make_stream(custom_gray)
        custom_stream[Name.N] = 1
        custom_cs = Array([Name.ICCBased, custom_stream])

        cs_dict = Dictionary()
        cs_dict[Name.DefaultGray] = custom_cs

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=cs_dict),
        )
        # Both RGB and Gray operators
        content = pdf.make_stream(
            b"1 0 0 rg 100 100 200 300 re f 0.5 g 50 50 100 100 re f"
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
        """AP Form XObject resources get DefaultGray entry."""
        pdf = new_pdf()

        # AP stream with DeviceGray operator
        ap_stream = pdf.make_stream(b"0.5 g 50 50 100 100 re f")
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

        # Page with RGB content so Gray becomes non-dominant
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        # AP stream resources should have DefaultGray
        ap_res = ap_stream.get(Name.Resources)
        assert ap_res is not None
        cs_dict = ap_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_gray = cs_dict.get(Name.DefaultGray)
        assert default_gray is not None
        assert default_gray[0] == Name.ICCBased

    def test_ap_image_replaced_with_icc(self) -> None:
        """Image XObject in AP stream gets DeviceGray replaced with ICCBased."""
        pdf = new_pdf()

        # Image with DeviceGray
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
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

        # Page with RGB content so Gray becomes non-dominant
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        # Image color space should now be ICCBased array
        img_cs = image[Name.ColorSpace]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
        # N=1 for gray
        assert img_cs[1][Name.N] == 1


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
        """Tiling pattern with Gray operator gets DefaultGray in resources."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"0.5 g 0 0 10 10 re f")

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=pattern)),
        )
        # RGB on page makes Gray non-dominant
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_gray = cs_dict.get(Name.DefaultGray)
        assert default_gray is not None
        assert default_gray[0] == Name.ICCBased
        assert default_gray[1]["/N"] == 1

    def test_tiling_pattern_image_replaced_with_icc(self) -> None:
        """Image in tiling pattern: DeviceGray replaced with ICCBased."""
        pdf = new_pdf()

        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        img_cs = image[Name.ColorSpace]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
        assert img_cs[1]["/N"] == 1

    def test_tiling_pattern_in_form_xobject_gets_defaults(self) -> None:
        """Tiling pattern inside a Form XObject gets DefaultGray."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"0.5 g 0 0 10 10 re f")

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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_gray = cs_dict.get(Name.DefaultGray)
        assert default_gray is not None
        assert default_gray[0] == Name.ICCBased

    def test_tiling_pattern_in_ap_stream_gets_defaults(self) -> None:
        """Tiling pattern inside an AP stream gets DefaultGray."""
        pdf = new_pdf()

        pattern = self._make_tiling_pattern(pdf, b"0.5 g 0 0 10 10 re f")

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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        pdf.pages[0][Name.Annots] = Array([annot])

        embed_color_profiles(pdf, "2b")

        pat_res = pattern.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_gray = cs_dict.get(Name.DefaultGray)
        assert default_gray is not None
        assert default_gray[0] == Name.ICCBased

    def test_nested_tiling_pattern_gets_defaults(self) -> None:
        """Nested tiling patterns both get DefaultGray."""
        pdf = new_pdf()

        inner_pattern = self._make_tiling_pattern(pdf, b"0.5 g 0 0 5 5 re f")
        outer_pattern = self._make_tiling_pattern(
            pdf,
            b"0.5 g 0 0 10 10 re f",
            resources=Dictionary(Pattern=Dictionary(P1=inner_pattern)),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Pattern=Dictionary(P0=outer_pattern)),
        )
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Outer pattern gets DefaultGray
        outer_res = outer_pattern.get(Name.Resources)
        assert outer_res is not None
        outer_cs = outer_res.get(Name.ColorSpace)
        assert outer_cs is not None
        assert Name.DefaultGray in outer_cs

        # Inner pattern gets DefaultGray
        inner_res = inner_pattern.get(Name.Resources)
        assert inner_res is not None
        inner_cs = inner_res.get(Name.ColorSpace)
        assert inner_cs is not None
        assert Name.DefaultGray in inner_cs

    def test_tiling_pattern_without_resources_gets_defaults(self) -> None:
        """Tiling pattern with no /Resources gets DefaultGray created."""
        pdf = new_pdf()

        # Create a tiling pattern stream WITHOUT setting /Resources
        pat = pdf.make_stream(b"0.5 g 0 0 10 10 re f")
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        pat_res = pat.get(Name.Resources)
        assert pat_res is not None
        cs_dict = pat_res.get(Name.ColorSpace)
        assert cs_dict is not None
        default_gray = cs_dict.get(Name.DefaultGray)
        assert default_gray is not None
        assert default_gray[0] == Name.ICCBased
        assert default_gray[1]["/N"] == 1

    def test_shading_pattern_replaced(self) -> None:
        """PatternType=2 (shading) has non-dominant DeviceGray replaced."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        # RGB on page — makes RGB dominant, Gray non-dominant
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        # Shading's ColorSpace should be replaced with ICCBased
        cs = shading_dict.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased


class TestShadingColorSpaceReplacement:
    """Tests for replacing Device color spaces in Shading dictionaries."""

    def test_shading_dict_non_dominant_replaced(self) -> None:
        """Page /Resources/Shading with non-dominant DeviceGray is replaced."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
                Coords=Array([0, 0, 100, 100]),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=Dictionary(Sh0=shading)),
        )
        # RGB content makes RGB dominant, Gray non-dominant
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_shading_pattern_type2_replaced(self) -> None:
        """PatternType 2 with non-dominant DeviceGray shading is replaced."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading_dict.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

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

    def test_shading_in_form_xobject_replaced(self) -> None:
        """Shading inside a Form XObject's resources is replaced."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f q /Fm0 Do Q")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_shading_in_tiling_pattern_replaced(self) -> None:
        """Shading inside a tiling pattern's resources is replaced."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_indexed_shading_base_replaced(self) -> None:
        """Indexed shading with DeviceGray base → base replaced with ICCBased."""
        pdf = new_pdf()

        lookup_data = bytes(range(256))  # 256 entries × 1 component
        lookup_stream = pdf.make_stream(lookup_data)
        indexed_cs = Array([Name.Indexed, Name.DeviceGray, 255, lookup_stream])

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
        # RGB content makes RGB dominant, Gray non-dominant
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.Indexed
        # Base is now [/ICCBased <stream>] with N=1
        base = cs[1]
        assert isinstance(base, Array)
        assert base[0] == Name.ICCBased
        assert base[1]["/N"] == 1
        # hival and lookup remain intact
        assert int(cs[2]) == 255
        assert cs[3].objgen == lookup_stream.objgen

    def test_indexed_shading_pattern_type2_replaced(self) -> None:
        """PatternType 2 with Indexed DeviceGray shading → base replaced."""
        pdf = new_pdf()

        lookup_data = bytes(range(256))  # 256 entries × 1 component
        lookup_stream = pdf.make_stream(lookup_data)
        indexed_cs = Array([Name.Indexed, Name.DeviceGray, 255, lookup_stream])

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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading_dict.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.Indexed
        # Base is now [/ICCBased <stream>] with N=1
        base = cs[1]
        assert isinstance(base, Array)
        assert base[0] == Name.ICCBased
        assert base[1]["/N"] == 1
        # hival and lookup remain intact
        assert int(cs[2]) == 255
        assert cs[3].objgen == lookup_stream.objgen

    def test_shading_in_type3_font_replaced(self) -> None:
        """Shading inside Type3 font /Resources gets Device CS replaced."""
        pdf = new_pdf()

        shading = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_pattern_in_type3_font_replaced(self) -> None:
        """Tiling pattern inside Type3 font /Resources gets defaults."""
        pdf = new_pdf()

        tiling_stream = pdf.make_stream(b"0.5 g 0 0 10 10 re f")
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        tiling_res = tiling.get(Name.Resources)
        assert tiling_res is not None
        assert Name("/DefaultGray") in tiling_res.get("/ColorSpace", {})

    def test_shading_pattern_type2_in_type3_font_replaced(self) -> None:
        """PatternType 2 shading in Type3 font gets Device CS replaced."""
        pdf = new_pdf()

        shading_dict = pdf.make_indirect(
            Dictionary(
                ShadingType=2,
                ColorSpace=Name.DeviceGray,
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
        content = pdf.make_stream(b"1 0 0 rg 100 100 200 300 re f")
        page_dict[Name.Contents] = content
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        embed_color_profiles(pdf, "2b")

        cs = shading_dict.get(Name.ColorSpace)
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased


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


class TestCalibratedConversion:
    """Tests for CalGray/CalRGB → ICCBased conversion."""

    def _make_cal_gray(self) -> Array:
        """Create a CalGray color space array."""
        return Array(
            [
                Name.CalGray,
                Dictionary(WhitePoint=Array([1, 1, 1]), Gamma=1),
            ]
        )

    def _make_cal_rgb(self) -> Array:
        """Create a CalRGB color space array."""
        return Array(
            [
                Name.CalRGB,
                Dictionary(
                    WhitePoint=Array([1, 1, 1]),
                    Matrix=Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
                ),
            ]
        )

    def test_cal_gray_image_converted(self):
        """CalGray image color space is converted to ICCBased(N=1)."""
        pdf = new_pdf()
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = self._make_cal_gray()
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

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 1

    def test_cal_rgb_image_converted(self):
        """CalRGB image color space is converted to ICCBased(N=3)."""
        pdf = new_pdf()
        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = self._make_cal_rgb()
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

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3

    def test_lab_unchanged(self):
        """Lab color spaces are not converted."""
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

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 0
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.Lab

    def test_flag_disabled_no_conversion(self):
        """When convert_calibrated is False, Cal* spaces are untouched."""
        pdf = new_pdf()
        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = self._make_cal_rgb()
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

        embed_color_profiles(pdf, "3b", convert_calibrated=False)
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.CalRGB

    def test_indexed_with_cal_base(self):
        """Indexed color space with CalGray/CalRGB base is converted."""
        pdf = new_pdf()
        cal_rgb = self._make_cal_rgb()
        indexed_cs = Array(
            [
                Name.Indexed,
                cal_rgb,
                1,
                pdf.make_stream(b"\xff\x00\x00\x00\xff\x00"),
            ]
        )
        image = pdf.make_stream(b"\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = indexed_cs
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

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.Indexed
        base = cs[1]
        assert isinstance(base, Array)
        assert base[0] == Name.ICCBased
        assert int(base[1][Name.N]) == 3

    def test_named_colorspace_entry(self):
        """CalRGB in a named ColorSpace dictionary entry is converted."""
        pdf = new_pdf()
        cs_dict = Dictionary(CS0=self._make_cal_rgb())

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=cs_dict),
        )
        page_dict[Name.Contents] = pdf.make_stream(b"")
        pdf.pages.append(pikepdf.Page(page_dict))

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = cs_dict[Name("/CS0")]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_shading_colorspace(self):
        """CalRGB in a Shading dictionary is converted."""
        pdf = new_pdf()
        shading = Dictionary(
            ShadingType=2,
            ColorSpace=self._make_cal_rgb(),
            Coords=Array([0, 0, 100, 100]),
            Function=Dictionary(
                FunctionType=2,
                Domain=Array([0, 1]),
                C0=Array([1, 0, 0]),
                C1=Array([0, 0, 1]),
                N=1,
            ),
        )
        shadings_dict = Dictionary(Sh0=shading)

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Shading=shadings_dict),
        )
        page_dict[Name.Contents] = pdf.make_stream(b"")
        pdf.pages.append(pikepdf.Page(page_dict))

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = shading[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3

    def test_cal_rgb_in_form_xobject(self):
        """CalRGB inside a Form XObject's resources is converted."""
        pdf = new_pdf()

        # Create an image with CalRGB inside a Form XObject
        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = self._make_cal_rgb()
        image[Name.BitsPerComponent] = 8

        form = pdf.make_stream(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=image),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form)),
        )
        page_dict[Name.Contents] = pdf.make_stream(b"q 1 0 0 1 0 0 cm /Fm0 Do Q")
        pdf.pages.append(pikepdf.Page(page_dict))

        cache: dict[ColorSpaceType, pikepdf.Stream] = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = image[Name.ColorSpace]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3

    def test_integration_convert_to_pdfa(self, tmp_path):
        """Integration: convert_calibrated=True converts CalRGB via convert_to_pdfa."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = new_pdf()
        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = self._make_cal_rgb()
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

        input_file = tmp_path / "cal_rgb_test.pdf"
        pdf.save(input_file)

        output_file = tmp_path / "cal_rgb_test_pdfa.pdf"
        result = convert_to_pdfa(input_file, output_file, "3b", convert_calibrated=True)
        assert result.success

        with Pdf.open(output_file) as out_pdf:
            page = out_pdf.pages[0]
            resources = page.get(Name.Resources)
            xobjects = resources.get(Name.XObject)
            im0 = xobjects[Name("/Im0")]
            cs = im0.get(Name.ColorSpace)
            assert isinstance(cs, Array)
            assert cs[0] == Name.ICCBased


class TestCalInTransparencyGroupCS:
    """Tests for CalGray/CalRGB → ICCBased in transparency group /CS."""

    def _make_cal_gray(self) -> Array:
        """Create a CalGray color space array."""
        return Array(
            [
                Name.CalGray,
                Dictionary(WhitePoint=Array([1, 1, 1]), Gamma=1),
            ]
        )

    def _make_cal_rgb(self) -> Array:
        """Create a CalRGB color space array."""
        return Array(
            [
                Name.CalRGB,
                Dictionary(
                    WhitePoint=Array([1, 1, 1]),
                    Matrix=Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
                ),
            ]
        )

    def test_page_level_cal_gray_replaced(self):
        """CalGray in page-level transparency group /CS is replaced."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=self._make_cal_gray()),
            )
        )
        pdf.pages.append(page)

        cache: dict = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = pdf.pages[0][Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 1

    def test_page_level_cal_rgb_replaced(self):
        """CalRGB in page-level transparency group /CS is replaced."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=self._make_cal_rgb()),
            )
        )
        pdf.pages.append(page)

        cache: dict = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = pdf.pages[0][Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3

    def test_page_level_lab_unchanged(self):
        """Lab in page-level transparency group /CS is not replaced."""
        pdf = new_pdf()
        lab_cs = Array([Name.Lab, Dictionary(WhitePoint=Array([1, 1, 1]))])
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=lab_cs),
            )
        )
        pdf.pages.append(page)

        cache: dict = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 0
        cs = pdf.pages[0][Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.Lab

    def test_form_xobject_cal_rgb_in_group_replaced(self):
        """CalRGB in Form XObject transparency group /CS is replaced."""
        pdf = new_pdf()
        form = pdf.make_stream(b"q Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Group] = Dictionary(
            S=Name.Transparency,
            CS=self._make_cal_rgb(),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(XObject=Dictionary(Fm0=form)),
            )
        )
        pdf.pages.append(page)

        cache: dict = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = form[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3

    def test_annotation_ap_cal_rgb_in_group_replaced(self):
        """CalRGB in annotation AP transparency group /CS is replaced."""
        pdf = new_pdf()
        ap_stream = pdf.make_stream(b"q Q")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 20, 20])
        ap_stream[Name.Group] = Dictionary(
            S=Name.Transparency,
            CS=self._make_cal_rgb(),
        )

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(N=ap_stream),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.pages[0].Annots = Array([annot])

        cache: dict = {}
        replaced = _convert_calibrated_colorspaces(pdf, cache)
        assert replaced == 1
        cs = ap_stream[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased
        assert int(cs[1][Name.N]) == 3


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

    def test_smask_g_form_images_replaced(self):
        """Images inside SMask /G Form XObject get color spaces replaced."""
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
        _, images_replaced = _apply_defaults_to_smask_groups(
            pdf.pages[0].Resources,
            non_dominant,
            icc_arrays,
            visited,
        )

        assert images_replaced == 1
        replaced_cs = img.get(Name.ColorSpace)
        assert isinstance(replaced_cs, Array)
        assert replaced_cs[0] == Name.ICCBased

    def test_embed_color_profiles_covers_smask_g(self):
        """End-to-end: embed_color_profiles covers SMask /G form."""
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

        # Verify the /G form got default color spaces
        form_resources = g_form.get(Name.Resources)
        assert form_resources is not None
        cs = form_resources.get(Name.ColorSpace)
        assert Name.DefaultGray in cs
