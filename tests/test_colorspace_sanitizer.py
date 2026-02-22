# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for color space sanitization for PDF/A compliance."""

from collections.abc import Generator

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.color_profile import (
    ColorSpaceAnalysis,
    ColorSpaceType,
    SpecialColorSpace,
    detect_color_spaces,
    get_srgb_profile,
)
from pdftopdfa.exceptions import ConversionError
from pdftopdfa.sanitizers.colorspaces import (
    sanitize_colorspaces,
    validate_embedded_icc_profiles,
)


class TestColorSpaceAnalysisExtensions:
    """Tests for extended ColorSpaceAnalysis class."""

    def test_separation_detected(self):
        """Separation flag is correctly detected."""
        analysis = ColorSpaceAnalysis(separation_used=True)
        assert analysis.separation_used is True
        assert ColorSpaceType.SEPARATION in analysis.detected_spaces

    def test_devicen_detected(self):
        """DeviceN flag is correctly detected."""
        analysis = ColorSpaceAnalysis(devicen_used=True)
        assert analysis.devicen_used is True
        assert ColorSpaceType.DEVICEN in analysis.detected_spaces

    def test_indexed_special_detected(self):
        """Indexed with special base flag is correctly detected."""
        analysis = ColorSpaceAnalysis(indexed_with_special_base=True)
        assert analysis.indexed_with_special_base is True
        assert ColorSpaceType.INDEXED in analysis.detected_spaces

    def test_has_special_colorspaces_true(self):
        """has_special_colorspaces returns True when special spaces exist."""
        analysis = ColorSpaceAnalysis(separation_used=True)
        assert analysis.has_special_colorspaces is True

    def test_has_special_colorspaces_false(self):
        """has_special_colorspaces returns False for simple spaces only."""
        analysis = ColorSpaceAnalysis(device_rgb_used=True)
        assert analysis.has_special_colorspaces is False

    def test_special_colorspaces_list(self):
        """special_colorspaces list stores details."""
        analysis = ColorSpaceAnalysis()
        analysis.special_colorspaces.append(
            SpecialColorSpace(
                type="Separation",
                alternate_space="DeviceCMYK",
                location="Page1/ColorSpace/CS0",
            )
        )
        assert len(analysis.special_colorspaces) == 1
        assert analysis.special_colorspaces[0].type == "Separation"


class TestSeparationDetection:
    """Tests for Separation color space detection."""

    @pytest.fixture
    def pdf_with_separation_colorspace(self) -> Generator[Pdf, None, None]:
        """PDF with Separation color space in page resources."""
        pdf = new_pdf()

        # Create a simple tint transform function
        tint_func = pdf.make_stream(b"{ dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        # Create Separation color space:
        # [/Separation /ColorName /AlternateSpace tintFunc]
        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotRed"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=separation_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_separation_image(self) -> Generator[Pdf, None, None]:
        """PDF with Separation color space in image XObject."""
        pdf = new_pdf()

        # Create tint function
        tint_func = pdf.make_stream(b"{ dup dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        # Create Separation color space
        separation_cs = Array(
            [
                Name.Separation,
                Name("/PantoneBlue"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        # Create image with Separation color space
        image_data = b"\x80"
        image_stream = pdf.make_stream(image_data)
        image_stream[Name.Type] = Name.XObject
        image_stream[Name.Subtype] = Name.Image
        image_stream[Name.Width] = 1
        image_stream[Name.Height] = 1
        image_stream[Name.ColorSpace] = separation_cs
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
        yield pdf

    def test_detect_separation_in_resources(self, pdf_with_separation_colorspace: Pdf):
        """Detects Separation color space in page resources."""
        analysis = detect_color_spaces(pdf_with_separation_colorspace)
        assert analysis.separation_used is True
        assert analysis.device_cmyk_used is True  # From alternate space

    def test_detect_separation_in_image(self, pdf_with_separation_image: Pdf):
        """Detects Separation color space in image XObject."""
        analysis = detect_color_spaces(pdf_with_separation_image)
        assert analysis.separation_used is True


class TestDeviceNDetection:
    """Tests for DeviceN color space detection."""

    @pytest.fixture
    def pdf_with_devicen_colorspace(self) -> Generator[Pdf, None, None]:
        """PDF with DeviceN color space in page resources."""
        pdf = new_pdf()

        # Create tint function
        tint_func = pdf.make_stream(b"{ pop pop pop 0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1, 0, 1, 0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        # Create DeviceN color space: [/DeviceN [names] /AlternateSpace tintFunc]
        devicen_cs = Array(
            [
                Name.DeviceN,
                Array([Name("/Cyan"), Name("/Magenta"), Name("/Yellow")]),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=devicen_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_detect_devicen_in_resources(self, pdf_with_devicen_colorspace: Pdf):
        """Detects DeviceN color space in page resources."""
        analysis = detect_color_spaces(pdf_with_devicen_colorspace)
        assert analysis.devicen_used is True
        assert analysis.device_cmyk_used is True


class TestIndexedDetection:
    """Tests for Indexed color space with special base detection."""

    @pytest.fixture
    def pdf_with_indexed_separation_base(self) -> Generator[Pdf, None, None]:
        """PDF with Indexed color space using Separation as base."""
        pdf = new_pdf()

        # Create tint function
        tint_func = pdf.make_stream(b"{ dup dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        # Create Separation color space
        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotColor"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        # Create Indexed color space with Separation as base
        lookup_data = bytes(range(256))
        indexed_cs = Array(
            [
                Name.Indexed,
                separation_cs,
                255,
                pdf.make_stream(lookup_data),
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=indexed_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_detect_indexed_with_separation_base(
        self, pdf_with_indexed_separation_base: Pdf
    ):
        """Detects Indexed with Separation base in page resources."""
        analysis = detect_color_spaces(pdf_with_indexed_separation_base)
        assert analysis.indexed_with_special_base is True


class TestICCProfileValidation:
    """Tests for embedded ICC profile validation."""

    @pytest.fixture
    def pdf_with_valid_icc(self) -> Generator[Pdf, None, None]:
        """PDF with valid ICCBased color space."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_invalid_icc(self) -> Generator[Pdf, None, None]:
        """PDF with invalid ICCBased color space (bad signature)."""
        pdf = new_pdf()

        # Create invalid ICC profile (wrong signature)
        invalid_profile = bytearray(get_srgb_profile())
        invalid_profile[36:40] = b"xxxx"  # Replace 'acsp' with invalid
        icc_stream = Stream(pdf, bytes(invalid_profile))
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_valid_icc_profile_passes(self, pdf_with_valid_icc: Pdf):
        """Valid ICC profile passes validation."""
        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf_with_valid_icc, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0
        assert repaired == 0

    def test_invalid_icc_profile_detected(self, pdf_with_invalid_icc: Pdf):
        """Invalid ICC profile is detected."""
        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf_with_invalid_icc, "3b"
        )
        assert len(warnings) > 0
        assert "Invalid ICC signature" in warnings[0]
        assert repaired == 0  # repair=False by default


class TestDeviceDependentAlternateWarning:
    """Tests for Separation/DeviceN device-dependent alternate warnings."""

    def test_separation_with_device_dependent_alternate(self):
        """Separation with DeviceCMYK alternate emits warning."""
        from pdftopdfa.sanitizers.colorspaces import _warn_device_dependent_alternates

        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ dup dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotRed"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=separation_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        count = _warn_device_dependent_alternates(pdf)
        assert count == 1

    def test_separation_with_iccbased_alternate_no_warning(self):
        """Separation with ICCBased alternate emits no warning."""
        from pdftopdfa.sanitizers.colorspaces import _warn_device_dependent_alternates

        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        tint_func = pdf.make_stream(b"{ dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1])

        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotRed"),
                icc_cs,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=separation_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        count = _warn_device_dependent_alternates(pdf)
        assert count == 0

    def test_devicen_with_device_dependent_alternate(self):
        """DeviceN with DeviceRGB alternate emits warning."""
        from pdftopdfa.sanitizers.colorspaces import _warn_device_dependent_alternates

        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ pop pop pop 0 0 0 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1, 0, 1, 0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1])

        devicen_cs = Array(
            [
                Name.DeviceN,
                Array([Name("/Cyan"), Name("/Magenta"), Name("/Yellow")]),
                Name.DeviceRGB,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=devicen_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        count = _warn_device_dependent_alternates(pdf)
        assert count == 1


class TestICCComponentCountValidation:
    """Tests for ICC /N cross-check against color space signature."""

    def test_mismatched_n_generates_warning(self):
        """ICC profile with /N not matching color space generates warning."""
        pdf = new_pdf()

        # sRGB profile has color space "RGB " → expects N=3
        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 4  # Wrong: should be 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b")
        assert validated == 1
        assert any("/N=4" in w and "expected 3" in w for w in warnings)

    def test_correct_n_no_warning(self):
        """ICC profile with correct /N generates no warning."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3  # Correct for RGB

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b")
        assert validated == 1
        assert len(warnings) == 0


class TestFullSanitization:
    """Tests for complete color space sanitization."""

    @pytest.fixture
    def pdf_with_separation(self) -> Generator[Pdf, None, None]:
        """PDF with Separation color space."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ dup dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotColor"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=separation_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_devicen(self) -> Generator[Pdf, None, None]:
        """PDF with DeviceN color space."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ pop pop pop 0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1, 0, 1, 0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        devicen_cs = Array(
            [
                Name.DeviceN,
                Array([Name("/Cyan"), Name("/Magenta"), Name("/Yellow")]),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=devicen_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_indexed_separation_base(self) -> Generator[Pdf, None, None]:
        """PDF with Indexed color space using Separation as base."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ dup dup dup }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        separation_cs = Array(
            [
                Name.Separation,
                Name("/SpotColor"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        lookup_data = bytes(range(256))
        indexed_cs = Array(
            [
                Name.Indexed,
                separation_cs,
                255,
                pdf.make_stream(lookup_data),
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=indexed_cs),
            ),
        )

        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_sanitization_preserves_separation(self, pdf_with_separation: Pdf):
        """Sanitization preserves Separation color space (allowed in PDF/A-2/3)."""
        result = sanitize_colorspaces(pdf_with_separation, "3b")

        assert "separation_converted" not in result
        assert "icc_profiles_validated" in result

        # Verify Separation is still present
        resources = pdf_with_separation.pages[0].Resources
        cs = resources.ColorSpace.CS0
        assert str(cs[0]) == "/Separation"

    def test_sanitization_preserves_devicen(self, pdf_with_devicen: Pdf):
        """Sanitization preserves DeviceN color space (allowed in PDF/A-2/3)."""
        result = sanitize_colorspaces(pdf_with_devicen, "3b")

        assert "devicen_converted" not in result

        # Verify DeviceN is still present
        resources = pdf_with_devicen.pages[0].Resources
        cs = resources.ColorSpace.CS0
        assert str(cs[0]) == "/DeviceN"

    def test_sanitization_preserves_indexed_with_special_base(
        self, pdf_with_indexed_separation_base: Pdf
    ):
        """Sanitization preserves Indexed with Separation base."""
        result = sanitize_colorspaces(pdf_with_indexed_separation_base, "3b")

        assert "indexed_converted" not in result

        # Verify Indexed with Separation base is still present
        resources = pdf_with_indexed_separation_base.pages[0].Resources
        cs = resources.ColorSpace.CS0
        assert str(cs[0]) == "/Indexed"
        base = cs[1]
        assert str(base[0]) == "/Separation"


class TestFormXObjectAndAPStreamTraversal:
    """Tests for ICC profile validation in Form XObjects and AP streams."""

    @pytest.fixture
    def pdf_with_icc_in_form_xobject(self) -> Generator[Pdf, None, None]:
        """PDF with ICCBased color space inside a Form XObject's resources."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        form_xobj = pdf.make_stream(b"q Q")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=icc_cs),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_xobj)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_icc_image_in_form_xobject(self) -> Generator[Pdf, None, None]:
        """PDF with ICCBased Image XObject inside a Form XObject."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        image = pdf.make_stream(b"\x80\x80\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.BitsPerComponent] = 8
        image[Name.ColorSpace] = icc_cs

        form_xobj = pdf.make_stream(b"q /Im0 Do Q")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = Dictionary(
            XObject=Dictionary(Im0=image),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_xobj)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_deeply_nested_form(self) -> Generator[Pdf, None, None]:
        """PDF with ICC in an inner Form nested inside an outer Form."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        inner_form = pdf.make_stream(b"q Q")
        inner_form[Name.Type] = Name.XObject
        inner_form[Name.Subtype] = Name.Form
        inner_form[Name.BBox] = Array([0, 0, 50, 50])
        inner_form[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=icc_cs),
        )

        outer_form = pdf.make_stream(b"q /Inner Do Q")
        outer_form[Name.Type] = Name.XObject
        outer_form[Name.Subtype] = Name.Form
        outer_form[Name.BBox] = Array([0, 0, 100, 100])
        outer_form[Name.Resources] = Dictionary(
            XObject=Dictionary(Inner=inner_form),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Outer=outer_form)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_icc_in_ap_stream(self) -> Generator[Pdf, None, None]:
        """PDF with ICCBased color space in an annotation AP /N stream."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        ap_form = pdf.make_stream(b"q Q")
        ap_form[Name.Type] = Name.XObject
        ap_form[Name.Subtype] = Name.Form
        ap_form[Name.BBox] = Array([0, 0, 20, 20])
        ap_form[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=icc_cs),
        )

        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 20, 20]),
            AP=Dictionary(N=ap_form),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_icc_in_ap_substate(self) -> Generator[Pdf, None, None]:
        """PDF with ICCBased color space in an AP /N sub-state dictionary."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        on_form = pdf.make_stream(b"q Q")
        on_form[Name.Type] = Name.XObject
        on_form[Name.Subtype] = Name.Form
        on_form[Name.BBox] = Array([0, 0, 20, 20])
        on_form[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=icc_cs),
        )

        off_form = pdf.make_stream(b"q Q")
        off_form[Name.Type] = Name.XObject
        off_form[Name.Subtype] = Name.Form
        off_form[Name.BBox] = Array([0, 0, 20, 20])

        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 20, 20]),
            AP=Dictionary(N=Dictionary(On=on_form, Off=off_form)),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(),
            Annots=Array([annot]),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    @pytest.fixture
    def pdf_with_cyclic_form_xobject(self) -> Generator[Pdf, None, None]:
        """PDF with a Form XObject that references itself (cycle)."""
        pdf = new_pdf()

        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 3
        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        form_xobj = pdf.make_stream(b"q /Self Do Q")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = Dictionary(
            ColorSpace=Dictionary(CS0=icc_cs),
            XObject=Dictionary(Self=form_xobj),
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Fm0=form_xobj)),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_icc_in_form_xobject(self, pdf_with_icc_in_form_xobject: Pdf):
        """ICC profile in Form XObject's /Resources/ColorSpace is validated."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_icc_in_form_xobject, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0

    def test_icc_in_image_inside_form_xobject(
        self, pdf_with_icc_image_in_form_xobject: Pdf
    ):
        """ICC profile on Image XObject inside a Form's resources is validated."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_icc_image_in_form_xobject, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0

    def test_icc_in_deeply_nested_form(self, pdf_with_deeply_nested_form: Pdf):
        """ICC in inner Form → outer Form → Page is still found."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_deeply_nested_form, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0

    def test_icc_in_annotation_ap_stream(self, pdf_with_icc_in_ap_stream: Pdf):
        """ICC in AP /N stream resources is validated."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_icc_in_ap_stream, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0

    def test_icc_in_ap_substate_dict(self, pdf_with_icc_in_ap_substate: Pdf):
        """ICC in AP /N sub-state dictionary (On/Off) is validated."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_icc_in_ap_substate, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0

    def test_cycle_detection(self, pdf_with_cyclic_form_xobject: Pdf):
        """Form referencing itself doesn't loop infinitely."""
        validated, warnings, _ = validate_embedded_icc_profiles(
            pdf_with_cyclic_form_xobject, "3b"
        )
        assert validated == 1
        assert len(warnings) == 0


class TestICCProfileRepair:
    """Tests for ICC profile repair functionality."""

    def test_invalid_signature_repaired(self):
        """Invalid ICC signature is repaired with built-in sRGB profile."""
        pdf = new_pdf()

        invalid_profile = bytearray(get_srgb_profile())
        invalid_profile[36:40] = b"xxxx"  # Replace 'acsp'
        icc_stream = Stream(pdf, bytes(invalid_profile))
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert repaired == 1
        assert any("Invalid ICC signature" in w for w in warnings)

        # Verify stream now contains valid sRGB data
        cs = pdf.pages[0].Resources.ColorSpace.CS0
        repaired_stream = cs[1]
        repaired_data = bytes(repaired_stream.read_bytes())
        assert repaired_data == get_srgb_profile()

    def test_too_small_profile_repaired(self):
        """ICC profile < 128 bytes is repaired with built-in profile."""
        pdf = new_pdf()

        icc_stream = Stream(pdf, b"\x00" * 64)
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert repaired == 1
        assert any("too small" in w for w in warnings)

        cs = pdf.pages[0].Resources.ColorSpace.CS0
        repaired_data = bytes(cs[1].read_bytes())
        assert repaired_data == get_srgb_profile()

    def test_wrong_icc_version_repaired(self):
        """ICC profile with major version > 4 is repaired."""
        pdf = new_pdf()

        invalid_profile = bytearray(get_srgb_profile())
        invalid_profile[8] = 9  # Major version 9
        icc_stream = Stream(pdf, bytes(invalid_profile))
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert repaired == 1
        assert any("not allowed" in w for w in warnings)

        cs = pdf.pages[0].Resources.ColorSpace.CS0
        repaired_data = bytes(cs[1].read_bytes())
        assert repaired_data == get_srgb_profile()

    def test_mismatched_n_corrected(self):
        """Mismatched /N is corrected when repair=True."""
        pdf = new_pdf()

        # sRGB profile has color space "RGB " → expects N=3
        profile_data = get_srgb_profile()
        icc_stream = Stream(pdf, profile_data)
        icc_stream.N = 4  # Wrong: should be 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert validated == 1
        assert any("/N=4" in w and "expected 3" in w for w in warnings)

        # /N should now be corrected to 3
        cs = pdf.pages[0].Resources.ColorSpace.CS0
        assert int(cs[1].N) == 3

    def test_unknown_n_skipped(self):
        """Stream with unsupported /N (e.g. 5) is not repaired."""
        pdf = new_pdf()

        icc_stream = Stream(pdf, b"\x00" * 64)
        icc_stream.N = 5  # Unsupported component count

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert repaired == 0
        assert any("too small" in w for w in warnings)

    def test_valid_profile_untouched_with_repair(self):
        """Valid ICC profile is not modified when repair=True."""
        pdf = new_pdf()

        original_data = get_srgb_profile()
        icc_stream = Stream(pdf, original_data)
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert validated == 1
        assert repaired == 0
        assert len(warnings) == 0

        cs = pdf.pages[0].Resources.ColorSpace.CS0
        assert bytes(cs[1].read_bytes()) == original_data

    def test_sanitize_colorspaces_returns_repaired_count(self):
        """sanitize_colorspaces includes icc_profiles_repaired in result."""
        pdf = new_pdf()

        invalid_profile = bytearray(get_srgb_profile())
        invalid_profile[36:40] = b"xxxx"
        icc_stream = Stream(pdf, bytes(invalid_profile))
        icc_stream.N = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=icc_cs),
            ),
        )
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        result = sanitize_colorspaces(pdf, "3b")
        assert "icc_profiles_repaired" in result
        assert result["icc_profiles_repaired"] == 1


class TestSpecialColorSpaceConsistency:
    """Tests for DeviceN Colorants and Separation consistency fixes."""

    def test_missing_devicen_colorants_entry_added(self):
        """Adds missing /Colorants entries for DeviceN spot names."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ pop pop 0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1, 0, 1])
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        spot_a = Array(
            [
                Name.Separation,
                Name("/SpotA"),
                Name.DeviceCMYK,
                tint_func,
            ]
        )

        devicen_cs = Array(
            [
                Name.DeviceN,
                Array([Name("/SpotA"), Name("/SpotB")]),
                Name.DeviceCMYK,
                tint_func,
                Dictionary(Colorants=Dictionary(SpotA=spot_a)),
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=devicen_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        result = sanitize_colorspaces(pdf, "3b")
        assert result["devicen_colorants_added"] == 1

        colorants = pdf.pages[0].Resources.ColorSpace.CS0[4].Colorants
        assert "/SpotA" in colorants
        assert "/SpotB" in colorants
        assert str(colorants["/SpotB"][0]) == "/Separation"

    def test_separation_arrays_with_same_name_are_normalized(self):
        """Normalizes alternate/tintTransform for matching Separation names."""
        pdf = new_pdf()

        tint_a = pdf.make_stream(b"{ dup dup dup }")
        tint_a[Name.FunctionType] = 4
        tint_a[Name.Domain] = Array([0, 1])
        tint_a[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        tint_b = pdf.make_stream(b"{ dup dup 0 0 }")
        tint_b[Name.FunctionType] = 4
        tint_b[Name.Domain] = Array([0, 1])
        tint_b[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        cs0 = Array(
            [
                Name.Separation,
                Name("/SpotBrand"),
                Name.DeviceCMYK,
                tint_a,
            ]
        )
        cs1 = Array(
            [
                Name.Separation,
                Name("/SpotBrand"),
                Name.DeviceRGB,
                tint_b,
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=cs0, CS1=cs1)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        result = sanitize_colorspaces(pdf, "3b")
        assert result["separation_arrays_normalized"] == 1

        cs0_out = pdf.pages[0].Resources.ColorSpace.CS0
        cs1_out = pdf.pages[0].Resources.ColorSpace.CS1
        assert str(cs1_out[2]) == str(cs0_out[2])
        assert cs1_out[3].objgen == cs0_out[3].objgen

    def test_colorants_separation_is_included_in_global_consistency(self):
        """Applies same-name Separation consistency to Colorants dictionaries."""
        pdf = new_pdf()

        tint_a = pdf.make_stream(b"{ dup dup dup }")
        tint_a[Name.FunctionType] = 4
        tint_a[Name.Domain] = Array([0, 1])
        tint_a[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        tint_b = pdf.make_stream(b"{ dup 0 0 0 }")
        tint_b[Name.FunctionType] = 4
        tint_b[Name.Domain] = Array([0, 1])
        tint_b[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        canonical_sep = Array(
            [
                Name.Separation,
                Name("/SpotMix"),
                Name.DeviceCMYK,
                tint_a,
            ]
        )
        mismatched_sep = Array(
            [
                Name.Separation,
                Name("/SpotMix"),
                Name.DeviceRGB,
                tint_b,
            ]
        )

        devicen_cs = Array(
            [
                Name.DeviceN,
                Array([Name("/SpotMix")]),
                Name.DeviceCMYK,
                tint_a,
                Dictionary(Colorants=Dictionary(SpotMix=mismatched_sep)),
            ]
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                ColorSpace=Dictionary(CS0=canonical_sep, CS1=devicen_cs)
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        result = sanitize_colorspaces(pdf, "3b")
        assert result["separation_arrays_normalized"] == 1

        colorant_sep = pdf.pages[0].Resources.ColorSpace.CS1[4].Colorants["/SpotMix"]
        canonical_out = pdf.pages[0].Resources.ColorSpace.CS0
        assert str(colorant_sep[2]) == str(canonical_out[2])
        assert colorant_sep[3].objgen == canonical_out[3].objgen

    def test_devicen_over_32_colorants_replaced_by_alternate(self, caplog):
        """DeviceN > 32 colorants replaced by alternate (lossy, rule 6.1.13-9)."""
        import logging

        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ " + b"pop " * 33 + b"0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1] * 33)
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        spot_names = Array([Name(f"/Spot{i}") for i in range(33)])
        devicen_cs = Array([Name.DeviceN, spot_names, Name.DeviceCMYK, tint_func])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=devicen_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            sanitize_colorspaces(pdf, "3b")

        assert "6.1.13-9" in caplog.text
        cs0 = pdf.pages[0].Resources.ColorSpace.CS0
        assert str(cs0) != "/DeviceN"

    def test_devicen_over_32_colorants_missing_alternate_raises(self):
        """DeviceN > 32 colorants with no alternate raises ConversionError."""
        pdf = new_pdf()

        spot_names = Array([Name(f"/Spot{i}") for i in range(33)])
        # Only 2-element array: [/DeviceN, [names]] — no alternate
        devicen_cs = Array([Name.DeviceN, spot_names])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=devicen_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with pytest.raises(ConversionError, match="6.1.13-9"):
            sanitize_colorspaces(pdf, "3b")

    def test_devicen_over_32_colorants_bad_alternate_raises(self):
        """DeviceN > 32 colorants with DeviceN > 32 alternate raises ConversionError."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ " + b"pop " * 33 + b"0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1] * 33)
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        alt_names = Array([Name(f"/AltSpot{i}") for i in range(33)])
        bad_alternate = Array([Name.DeviceN, alt_names, Name.DeviceCMYK, tint_func])

        spot_names = Array([Name(f"/Spot{i}") for i in range(33)])
        devicen_cs = Array([Name.DeviceN, spot_names, bad_alternate, tint_func])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=devicen_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with pytest.raises(ConversionError, match="6.1.13-9"):
            sanitize_colorspaces(pdf, "3b")

    def test_devicen_exactly_32_colorants_passes(self):
        """DeviceN with exactly 32 colorants does not raise (rule 6.1.13-9 boundary)."""
        pdf = new_pdf()

        tint_func = pdf.make_stream(b"{ " + b"pop " * 32 + b"0 0 0 1 }")
        tint_func[Name.FunctionType] = 4
        tint_func[Name.Domain] = Array([0, 1] * 32)
        tint_func[Name.Range] = Array([0, 1, 0, 1, 0, 1, 0, 1])

        spot_names = Array([Name(f"/Spot{i}") for i in range(32)])
        devicen_cs = Array([Name.DeviceN, spot_names, Name.DeviceCMYK, tint_func])

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=devicen_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        # Should not raise
        sanitize_colorspaces(pdf, "3b")


class TestICCMissingNDerivedFromProfile:
    """Tests for deriving /N from ICC profile header when missing."""

    def _make_icc_profile(self, colorspace: bytes = b"RGB ") -> bytes:
        """Create a minimal valid ICC profile with given color space."""
        data = bytearray(128)
        data[36:40] = b"acsp"  # ICC signature
        data[8] = 2  # version 2
        data[12:16] = b"mntr"  # profile class
        data[16:20] = colorspace  # color space
        return bytes(data)

    def test_missing_n_derived_from_rgb_profile(self):
        """Missing /N is set to 3 for an RGB ICC profile."""
        pdf = new_pdf()
        icc_data = self._make_icc_profile(b"RGB ")
        icc_stream = pdf.make_stream(icc_data)
        # Deliberately do NOT set /N

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=icc_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        validated, warnings, repaired = validate_embedded_icc_profiles(
            pdf, "3b", repair=True
        )
        assert validated == 1
        assert int(icc_stream[Name.N]) == 3

    def test_missing_n_derived_from_gray_profile(self):
        """Missing /N is set to 1 for a GRAY ICC profile."""
        pdf = new_pdf()
        icc_data = self._make_icc_profile(b"GRAY")
        icc_stream = pdf.make_stream(icc_data)

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=icc_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        validate_embedded_icc_profiles(pdf, "3b", repair=True)
        assert int(icc_stream[Name.N]) == 1

    def test_missing_n_derived_from_cmyk_profile(self):
        """Missing /N is set to 4 for a CMYK ICC profile."""
        pdf = new_pdf()
        icc_data = self._make_icc_profile(b"CMYK")
        icc_stream = pdf.make_stream(icc_data)

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=icc_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        validate_embedded_icc_profiles(pdf, "3b", repair=True)
        assert int(icc_stream[Name.N]) == 4

    def test_present_n_not_overwritten(self):
        """/N already present and correct is not changed."""
        pdf = new_pdf()
        icc_data = self._make_icc_profile(b"RGB ")
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 3

        icc_cs = Array([Name.ICCBased, pdf.make_indirect(icc_stream)])
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=icc_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b", repair=False)
        assert validated == 1
        assert len(warnings) == 0
        assert int(icc_stream[Name.N]) == 3


class TestIndexedLookupSizeValidation:
    """Tests for Indexed color space lookup table size validation."""

    def test_correct_size_no_warning(self, caplog):
        """Correctly sized lookup produces no warning."""
        import logging

        pdf = new_pdf()
        # DeviceRGB base, hival=1 -> lookup = (1+1)*3 = 6 bytes
        lookup_data = b"\xff\x00\x00\x00\xff\x00"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                1,
                pikepdf.String(lookup_data),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            validate_embedded_icc_profiles(pdf, "3b", repair=False)
        assert "lookup table size mismatch" not in caplog.text

    def test_wrong_size_logs_warning(self, caplog):
        """Incorrectly sized (too short) lookup is padded and a warning is logged."""
        import logging

        pdf = new_pdf()
        # DeviceRGB base, hival=1 -> expected 6, give 4
        lookup_data = b"\xff\x00\x00\x00"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                1,
                pikepdf.String(lookup_data),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            validate_embedded_icc_profiles(pdf, "3b", repair=False)
        assert "too short" in caplog.text
        assert "padded" in caplog.text
        # Lookup is fixed in-place: should now be 6 bytes
        cs = pdf.pages[0].Resources.ColorSpace.CS0
        assert len(bytes(cs[3])) == 6

    def test_stream_lookup_validated(self, caplog):
        """Lookup as a stream is padded when too short."""
        import logging

        pdf = new_pdf()
        # DeviceGray base, hival=2 -> expected (2+1)*1 = 3 bytes
        lookup_stream = pdf.make_stream(b"\x00\xff")  # only 2 bytes
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceGray,
                2,
                lookup_stream,
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            validate_embedded_icc_profiles(pdf, "3b", repair=False)
        assert "too short" in caplog.text
        assert "padded" in caplog.text
        # Stream data is fixed in-place: should now be 3 bytes
        cs = pdf.pages[0].Resources.ColorSpace.CS0
        assert len(bytes(cs[3].read_bytes())) == 3

    def test_image_xobject_indexed_validated(self, caplog):
        """Indexed color space on Image XObject is padded when too short."""
        import logging

        pdf = new_pdf()
        # DeviceCMYK base, hival=0 -> expected (0+1)*4 = 4, give 2
        lookup_data = b"\xff\x00"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceCMYK,
                0,
                pikepdf.String(lookup_data),
            ]
        )
        img = pdf.make_stream(b"\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.BitsPerComponent] = 8
        img[Name.ColorSpace] = indexed_cs

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=Dictionary(Im0=img)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            validate_embedded_icc_profiles(pdf, "3b", repair=False)
        assert "too short" in caplog.text
        assert "padded" in caplog.text

    def test_too_long_table_is_truncated(self, caplog):
        """Overlong lookup table is truncated to the correct size."""
        import logging

        from pdftopdfa.sanitizers.colorspaces import _fix_indexed_lookup_size

        pdf = new_pdf()
        # DeviceRGB base, hival=1 -> expected 6 bytes, give 10
        lookup_data = b"\xff\x00\x00\x00\xff\x00\xaa\xbb\xcc\xdd"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                1,
                pikepdf.String(lookup_data),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        cs = pdf.pages[0].Resources.ColorSpace.CS0

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            _fix_indexed_lookup_size(cs, "Page 1/ColorSpace/CS0")

        assert "truncated" in caplog.text
        assert len(bytes(cs[3])) == 6

    def test_too_short_table_is_padded(self, caplog):
        """Short lookup table is padded with zero bytes to the correct size."""
        import logging

        from pdftopdfa.sanitizers.colorspaces import _fix_indexed_lookup_size

        pdf = new_pdf()
        # DeviceRGB base, hival=1 -> expected 6 bytes, give 3
        lookup_data = b"\xff\x00\x00"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                1,
                pikepdf.String(lookup_data),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        cs = pdf.pages[0].Resources.ColorSpace.CS0

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            _fix_indexed_lookup_size(cs, "Page 1/ColorSpace/CS0")

        assert "padded" in caplog.text
        fixed = bytes(cs[3])
        assert len(fixed) == 6
        assert fixed[:3] == lookup_data
        assert fixed[3:] == b"\x00\x00\x00"

    def test_malformed_raises_conversion_error(self):
        """Indexed array with unreadable hival raises ConversionError."""
        from pdftopdfa.sanitizers.colorspaces import _fix_indexed_lookup_size

        pdf = new_pdf()
        # hival is a Name, not convertible to int — structurally malformed
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                Name("/bad"),
                pikepdf.String(b"\xff\x00\x00"),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))
        cs = pdf.pages[0].Resources.ColorSpace.CS0

        with pytest.raises(ConversionError, match="malformed"):
            _fix_indexed_lookup_size(cs, "Page 1/ColorSpace/CS0")

    def test_after_fix_lookup_length_equals_expected(self, caplog):
        """After sanitize_colorspaces(), lookup length equals (hival+1)*components."""
        import logging

        pdf = new_pdf()
        # DeviceRGB base, hival=3 -> expected (3+1)*3 = 12 bytes, give 5
        lookup_data = b"\xff\x00\x00\x00\xff"
        indexed_cs = Array(
            [
                Name.Indexed,
                Name.DeviceRGB,
                3,
                pikepdf.String(lookup_data),
            ]
        )
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(ColorSpace=Dictionary(CS0=indexed_cs)),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        with caplog.at_level(logging.WARNING, logger="pdftopdfa"):
            sanitize_colorspaces(pdf, "3b")

        cs = pdf.pages[0].Resources.ColorSpace.CS0
        assert len(bytes(cs[3])) == 12


class TestIntegrationWithSanitizeForPdfa:
    """Integration tests with the main sanitize_for_pdfa function."""

    def test_sanitize_for_pdfa_does_not_include_removed_keys(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa no longer returns colorspace conversion keys."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "separation_converted" not in result
        assert "devicen_converted" not in result
        assert "indexed_converted" not in result
        assert "icc_profiles_replaced" not in result
