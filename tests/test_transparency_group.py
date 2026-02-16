# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for transparency group /CS validation (ISO 19005-2, Clause 6.4)."""

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.color_profile import (
    ColorSpaceType,
    _create_icc_colorspace,
    _fix_transparency_group_colorspaces,
    embed_color_profiles,
)


def _make_form_with_transparency_group(pdf: Pdf, cs: object = Name.DeviceRGB) -> Stream:
    """Create a Form XObject with a transparency group.

    Args:
        pdf: An open pikepdf Pdf.
        cs: The /CS value for the transparency group.

    Returns:
        A Form XObject stream with /Group << /S /Transparency /CS <cs> >>.
    """
    form = pdf.make_stream(b"q Q")
    form[Name.Type] = Name.XObject
    form[Name.Subtype] = Name.Form
    form[Name.BBox] = Array([0, 0, 100, 100])
    group = Dictionary(S=Name.Transparency, CS=cs)
    form[Name.Group] = group
    return form


def _add_page_with_form(pdf: Pdf, form: Stream, form_name: str = "Fm0") -> None:
    """Add a page referencing a Form XObject.

    Args:
        pdf: An open pikepdf Pdf.
        form: The Form XObject to reference.
        form_name: The resource name for the form.
    """
    xobj_dict = Dictionary()
    xobj_dict[Name("/" + form_name)] = form
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(XObject=xobj_dict),
        )
    )
    pdf.pages.append(page)


class TestTransparencyGroupCSFix:
    """Tests for replacing Device /CS in transparency groups."""

    @pytest.mark.parametrize(
        "device_cs",
        [Name.DeviceRGB, Name.DeviceCMYK, Name.DeviceGray],
    )
    def test_device_cs_replaced(self, device_cs):
        """Replaces device color space in transparency group /CS."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = _make_form_with_transparency_group(pdf, device_cs)
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        group = form[Name.Group]
        cs = group[Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_no_cs_no_error(self):
        """No /CS in group dict does not cause errors."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = pdf.make_stream(b"q Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Group] = Dictionary(S=Name.Transparency)
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0

    def test_iccbased_cs_untouched(self):
        """ICCBased /CS is left unchanged."""
        pdf = new_pdf()
        icc_cache: dict = {}
        icc_array = _create_icc_colorspace(pdf, ColorSpaceType.DEVICE_RGB, icc_cache)
        form = _make_form_with_transparency_group(pdf, icc_array)
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0

    def test_no_group_no_error(self):
        """Form XObject without /Group does not cause errors."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = pdf.make_stream(b"q Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0

    @pytest.mark.parametrize(
        "device_cs",
        [Name.DeviceRGB, Name.DeviceCMYK, Name.DeviceGray],
    )
    def test_array_device_cs_replaced(self, device_cs):
        """Replaces [/Device*] array form in transparency group /CS."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = _make_form_with_transparency_group(pdf, Array([device_cs]))
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        group = form[Name.Group]
        cs = group[Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_array_non_device_cs_ignored(self):
        """Array with non-device color space like [/CalRGB ...] is ignored."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = _make_form_with_transparency_group(pdf, Array([Name("/CalRGB")]))
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0

    def test_non_transparency_group_ignored(self):
        """Group with /S other than /Transparency is ignored."""
        pdf = new_pdf()
        icc_cache: dict = {}
        form = pdf.make_stream(b"q Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 100, 100])
        form[Name.Group] = Dictionary(S=Name("/SomeOtherType"), CS=Name.DeviceRGB)
        _add_page_with_form(pdf, form)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0


class TestTransparencyGroupNested:
    """Tests for nested Form XObjects with transparency groups."""

    def test_nested_form_fixed(self):
        """Transparency group in nested Form XObject is fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        inner = _make_form_with_transparency_group(pdf, Name.DeviceRGB)
        outer = pdf.make_stream(b"/Inner Do")
        outer[Name.Type] = Name.XObject
        outer[Name.Subtype] = Name.Form
        outer[Name.BBox] = Array([0, 0, 200, 200])
        outer[Name.Resources] = Dictionary(
            XObject=Dictionary(Inner=inner),
        )
        _add_page_with_form(pdf, outer)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        cs = inner[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_deeply_nested_three_levels(self):
        """Transparency group in 3-level nested Form XObject is fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        level3 = _make_form_with_transparency_group(pdf, Name.DeviceCMYK)

        level2 = pdf.make_stream(b"/L3 Do")
        level2[Name.Type] = Name.XObject
        level2[Name.Subtype] = Name.Form
        level2[Name.BBox] = Array([0, 0, 150, 150])
        level2[Name.Resources] = Dictionary(
            XObject=Dictionary(L3=level3),
        )

        level1 = pdf.make_stream(b"/L2 Do")
        level1[Name.Type] = Name.XObject
        level1[Name.Subtype] = Name.Form
        level1[Name.BBox] = Array([0, 0, 200, 200])
        level1[Name.Resources] = Dictionary(
            XObject=Dictionary(L2=level2),
        )

        _add_page_with_form(pdf, level1)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        cs = level3[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_cycle_detection(self):
        """Cycle in Form XObject references does not cause infinite loop."""
        pdf = new_pdf()
        icc_cache: dict = {}

        form_a = _make_form_with_transparency_group(pdf, Name.DeviceRGB)
        form_b = pdf.make_stream(b"/FormA Do")
        form_b[Name.Type] = Name.XObject
        form_b[Name.Subtype] = Name.Form
        form_b[Name.BBox] = Array([0, 0, 100, 100])

        # Make both indirect so they get real objgen values
        form_a = pdf.make_indirect(form_a)
        form_b = pdf.make_indirect(form_b)

        # Create cycle: A -> B -> A
        form_a[Name.Resources] = Dictionary(
            XObject=Dictionary(FormB=form_b),
        )
        form_b[Name.Resources] = Dictionary(
            XObject=Dictionary(FormA=form_a),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    XObject=Dictionary(FormA=form_a, FormB=form_b),
                ),
            )
        )
        pdf.pages.append(page)

        # Should terminate without infinite loop
        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1


class TestTransparencyGroupInAnnotations:
    """Tests for transparency groups in annotation appearance streams."""

    def test_ap_n_stream_fixed(self):
        """Transparency group in AP /N stream is fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        ap_stream = _make_form_with_transparency_group(pdf, Name.DeviceRGB)

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

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        cs = ap_stream[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_ap_sub_state_dict_fixed(self):
        """Transparency groups in AP sub-state dictionaries are fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        on_stream = _make_form_with_transparency_group(pdf, Name.DeviceRGB)
        off_stream = _make_form_with_transparency_group(pdf, Name.DeviceCMYK)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(
                    N=Dictionary(On=on_stream, Off=off_stream),
                ),
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

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 2
        assert on_stream[Name.Group][Name.CS][0] == Name.ICCBased
        assert off_stream[Name.Group][Name.CS][0] == Name.ICCBased

    def test_ap_r_and_d_entries_fixed(self):
        """Transparency groups in AP /R and /D entries are also fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        r_stream = _make_form_with_transparency_group(pdf, Name.DeviceRGB)
        d_stream = _make_form_with_transparency_group(pdf, Name.DeviceGray)

        annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                AP=Dictionary(R=r_stream, D=d_stream),
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

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 2
        assert r_stream[Name.Group][Name.CS][0] == Name.ICCBased
        assert d_stream[Name.Group][Name.CS][0] == Name.ICCBased


class TestTransparencyGroupOnPage:
    """Tests for page-level transparency group /CS (ISO 32000-1, Table 30)."""

    @pytest.mark.parametrize(
        "device_cs",
        [Name.DeviceRGB, Name.DeviceCMYK, Name.DeviceGray],
    )
    def test_page_level_device_cs_replaced(self, device_cs):
        """Replaces device color space in page-level transparency group /CS."""
        pdf = new_pdf()
        icc_cache: dict = {}

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=device_cs),
            )
        )
        pdf.pages.append(page)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 1
        cs = pdf.pages[0][Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_page_level_iccbased_untouched(self):
        """ICCBased /CS on page-level group is left unchanged."""
        pdf = new_pdf()
        icc_cache: dict = {}
        icc_array = _create_icc_colorspace(pdf, ColorSpaceType.DEVICE_RGB, icc_cache)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=icc_array),
            )
        )
        pdf.pages.append(page)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0

    def test_page_level_and_xobject_both_fixed(self):
        """Both page-level and XObject transparency groups are fixed."""
        pdf = new_pdf()
        icc_cache: dict = {}

        form = _make_form_with_transparency_group(pdf, Name.DeviceCMYK)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Group=Dictionary(S=Name.Transparency, CS=Name.DeviceRGB),
                Resources=Dictionary(XObject=Dictionary(Fm0=form)),
            )
        )
        pdf.pages.append(page)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 2
        page_cs = pdf.pages[0][Name.Group][Name.CS]
        assert isinstance(page_cs, Array)
        assert page_cs[0] == Name.ICCBased
        form_cs = form[Name.Group][Name.CS]
        assert isinstance(form_cs, Array)
        assert form_cs[0] == Name.ICCBased

    def test_page_without_group_no_error(self):
        """Page without /Group entry does not cause errors."""
        pdf = new_pdf()
        icc_cache: dict = {}

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)

        fixed = _fix_transparency_group_colorspaces(pdf, icc_cache)

        assert fixed == 0


class TestTransparencyGroupIntegration:
    """End-to-end tests via embed_color_profiles."""

    def test_end_to_end_via_embed_color_profiles(self):
        """embed_color_profiles fixes transparency group /CS."""
        pdf = new_pdf()
        form = _make_form_with_transparency_group(pdf, Name.DeviceRGB)
        _add_page_with_form(pdf, form)

        # Add RGB content so color detection finds DeviceRGB
        content = pdf.make_stream(b"1 0 0 rg")
        pdf.pages[0][Name.Contents] = content

        embed_color_profiles(pdf, "3b")

        cs = form[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_dominant_color_space_also_replaced(self):
        """Transparency group /CS is replaced even for the dominant space."""
        pdf = new_pdf()
        # Create CMYK document (CMYK is dominant)
        content = pdf.make_stream(b"0 0 0 1 k")
        form = _make_form_with_transparency_group(pdf, Name.DeviceCMYK)

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=content,
                Resources=Dictionary(
                    XObject=Dictionary(Fm0=form),
                ),
            )
        )
        pdf.pages.append(page)

        embed_color_profiles(pdf, "3b")

        cs = form[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

    def test_image_and_transparency_group_both_fixed(self):
        """Image and transparency group in same form are both handled."""
        pdf = new_pdf()

        # Form XObject with transparency group
        form = _make_form_with_transparency_group(pdf, Name.DeviceRGB)

        # Image XObject with DeviceRGB
        image = pdf.make_stream(b"\xff\x00\x00")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 8

        # Put both under the same page, with RGB content to trigger detection
        content = pdf.make_stream(b"1 0 0 rg")

        # Also make a CMYK operator so RGB becomes non-dominant
        content2 = pdf.make_stream(b"0 0 0 1 k")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Contents=Array([content, content2]),
                Resources=Dictionary(
                    XObject=Dictionary(Fm0=form, Im0=image),
                ),
            )
        )
        pdf.pages.append(page)

        embed_color_profiles(pdf, "3b")

        # Transparency group should be fixed
        cs = form[Name.Group][Name.CS]
        assert isinstance(cs, Array)
        assert cs[0] == Name.ICCBased

        # Image should also be fixed (RGB is non-dominant when CMYK present)
        img_cs = image[Name.ColorSpace]
        assert isinstance(img_cs, Array)
        assert img_cs[0] == Name.ICCBased
