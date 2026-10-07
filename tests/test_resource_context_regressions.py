# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression checks for resource inheritance and inline-image shortcuts."""

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.fonts.glyph_usage import (
    find_ambiguous_resource_context_streams,
    stream_uses_named_resources,
    used_resource_names,
)
from pdftopdfa.sanitizers.rendering_intent import sanitize_rendering_intent
from pdftopdfa.sanitizers.xobjects import fix_image_interpolate


def make_form(pdf, body):
    form = pdf.make_stream(body)
    form.Type = Name.XObject
    form.Subtype = Name.Form
    form.BBox = Array([0, 0, 10, 10])
    return form


@pytest.mark.parametrize("body", [b"0.5 g", b"/DeviceGray cs 0.5 sc"])
def test_preserve_implicit_default_gray(body):
    with Pdf.new() as pdf:
        form = make_form(pdf, body + b" 0 0 10 10 re f")
        default_gray = Array(
            [Name.CalGray, Dictionary(WhitePoint=Array([1, 1, 1]), Gamma=2.2)]
        )
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(
            ColorSpace=Dictionary(DefaultGray=default_gray),
            XObject=Dictionary(Fm=form),
        )
        page.Contents = pdf.make_stream(b"/Fm Do")

        sanitize_rendering_intent(pdf)

        spaces = form.Resources.get("/ColorSpace", Dictionary())
        assert spaces.get("/DefaultGray") == default_gray


@pytest.mark.parametrize("name", ["RGB", "G", "CMYK", "I", "Indexed"])
def test_color_space_resource_names_outside_inline_images(name):
    with Pdf.new() as pdf:
        form = make_form(pdf, f"/{name} cs 0.5 sc 0 0 10 10 re f".encode())
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(
            ColorSpace=Dictionary({f"/{name}": Name.DeviceGray}),
            XObject=Dictionary(Fm=form),
        )
        page.Contents = pdf.make_stream(b"/Fm Do")

        sanitize_rendering_intent(pdf)

        spaces = form.Resources.get("/ColorSpace", Dictionary())
        assert spaces.get(f"/{name}") == Name.DeviceGray


def test_preserve_resources_used_by_nested_resourceless_form():
    with Pdf.new() as pdf:
        inner = make_form(pdf, b"/CS0 cs 0.5 sc 0 0 10 10 re f")
        outer = make_form(pdf, b"/Inner Do")
        outer.Resources = Dictionary(XObject=Dictionary(Inner=inner))
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(
            ColorSpace=Dictionary(CS0=Name.DeviceGray),
            XObject=Dictionary(Outer=outer),
        )
        page.Contents = pdf.make_stream(b"/Outer Do")

        sanitize_rendering_intent(pdf)

        spaces = inner.Resources.get("/ColorSpace", Dictionary())
        assert spaces.get("/CS0") == Name.DeviceGray


@pytest.mark.parametrize(
    "entry",
    [b"/I#6eterpolate true", b"/In#74erpolate true", b"/I\x00true"],
)
def test_interpolate_name_escapes_and_pdf_whitespace(entry):
    with Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Contents = pdf.make_stream(
            b"BI /W 1 /H 1 /CS /G /BPC 8 " + entry + b" ID \x80 EI"
        )
        before = list(pikepdf.parse_content_stream(page.Contents))[0].iimage.obj
        assert before["/Interpolate"] is True

        fix_image_interpolate(pdf)

        after = list(pikepdf.parse_content_stream(page.Contents))[0].iimage.obj
        assert after["/Interpolate"] is False


@pytest.mark.parametrize(
    ("body", "default"),
    [
        (b"0.5 g", "/DefaultGray"),
        (b"0.5 G", "/DefaultGray"),
        (b"1 0 0 rg", "/DefaultRGB"),
        (b"1 0 0 RG", "/DefaultRGB"),
        (b"0 1 0 0 k", "/DefaultCMYK"),
        (b"0 1 0 0 K", "/DefaultCMYK"),
        (b"/DeviceGray cs", "/DefaultGray"),
        (b"/DeviceRGB CS", "/DefaultRGB"),
        (b"/DeviceCMYK cs", "/DefaultCMYK"),
        (b"BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI", "/DefaultGray"),
    ],
)
def test_device_colors_depend_on_default_resources(body, default):
    with Pdf.new() as pdf:
        stream = pdf.make_stream(body)
        assert stream_uses_named_resources(stream)
        assert default in used_resource_names([stream])["/ColorSpace"]


@pytest.mark.parametrize("type3", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        b"0.5 g 0 0 10 10 re f",
        b"/DeviceGray cs 0.5 sc 0 0 10 10 re f",
        b"BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI",
    ],
)
def test_shared_streams_retain_each_pages_default_gray(type3, body):
    with Pdf.new() as pdf:
        shared = make_form(pdf, body)
        if type3:
            glyph = pdf.make_stream(b"10 0 d0 " + body)
            shared = pdf.make_indirect(
                Dictionary(
                    Type=Name.Font,
                    Subtype=Name.Type3,
                    FontBBox=Array([0, 0, 10, 10]),
                    FontMatrix=Array([1, 0, 0, 1, 0, 0]),
                    CharProcs=Dictionary(a=glyph),
                    Encoding=Dictionary(Differences=Array([97, Name.a])),
                    FirstChar=97,
                    LastChar=97,
                    Widths=Array([10]),
                )
            )
        for gamma in (1, 2):
            page = pdf.add_blank_page(page_size=(100, 100))
            page.Resources = Dictionary(
                ColorSpace=Dictionary(
                    DefaultGray=Array(
                        [
                            Name.CalGray,
                            Dictionary(WhitePoint=Array([1, 1, 1]), Gamma=gamma),
                        ]
                    )
                )
            )
            if type3:
                page.Resources.Font = Dictionary(T=shared)
                page.Contents = pdf.make_stream(b"BT /T 10 Tf (a) Tj ET")
            else:
                page.Resources.XObject = Dictionary(Fm=shared)
                page.Contents = pdf.make_stream(b"/Fm Do")

        sanitize_rendering_intent(pdf)

        owners = [
            page.Resources.Font.T if type3 else page.Resources.XObject.Fm
            for page in pdf.pages
        ]
        assert owners[0].objgen != owners[1].objgen
        assert [
            owner.Resources.ColorSpace.DefaultGray[1].Gamma for owner in owners
        ] == [
            1,
            2,
        ]
        assert not find_ambiguous_resource_context_streams(pdf)


@pytest.mark.parametrize("local_color", [False, True])
@pytest.mark.parametrize("appearance", [False, True])
def test_nested_forms_keep_nearest_resource_binding(local_color, appearance):
    with Pdf.new() as pdf:
        inner = make_form(pdf, b"/CS0 cs 0.5 sc 0 0 10 10 re f")
        outer = make_form(pdf, b"/Inner Do")
        outer.Resources = Dictionary(XObject=Dictionary(Inner=inner))
        if local_color:
            outer.Resources.ColorSpace = Dictionary(CS0=Name.DeviceGray)
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(ColorSpace=Dictionary(CS0=Name.DeviceRGB))
        if appearance:
            page.Annots = Array(
                [
                    pdf.make_indirect(
                        Dictionary(
                            Type=Name.Annot,
                            Subtype=Name.Stamp,
                            Rect=Array([0, 0, 10, 10]),
                            AP=Dictionary(N=outer),
                        )
                    )
                ]
            )
        else:
            page.Resources.XObject = Dictionary(Outer=outer)
            page.Contents = pdf.make_stream(b"/Outer Do")

        sanitize_rendering_intent(pdf)

        assert inner.Resources.ColorSpace.CS0 == (
            Name.DeviceGray if local_color else Name.DeviceRGB
        )
        assert "/XObject" not in inner.Resources


def test_default_rgb_survives_an_indirect_image_color_selection():
    with Pdf.new() as pdf:
        image = pdf.make_stream(b"\x80\x00\x00")
        image.Type = Name.XObject
        image.Subtype = Name.Image
        image.Width = image.Height = 1
        image.BitsPerComponent = 8
        image.ColorSpace = Name.DeviceRGB
        form = make_form(pdf, b"/Im Do")
        default_rgb = Array(
            [
                Name.CalRGB,
                Dictionary(WhitePoint=Array([1, 1, 1]), Gamma=Array([2, 2, 2])),
            ]
        )
        page = pdf.add_blank_page(page_size=(100, 100))
        page.Resources = Dictionary(
            ColorSpace=Dictionary(DefaultRGB=default_rgb),
            XObject=Dictionary(Fm=form, Im=image),
        )
        page.Contents = pdf.make_stream(b"/Fm Do")

        sanitize_rendering_intent(pdf)

        assert form.Resources.ColorSpace.DefaultRGB == default_rgb
        assert set(form.Resources.XObject.keys()) == {"/Im"}
