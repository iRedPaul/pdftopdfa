# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for sanitizers/xobjects.py."""

import pikepdf
from pikepdf import Array, Dictionary, Name, Stream

from pdftopdfa.sanitizers.xobjects import (
    fix_bits_per_component,
    fix_image_interpolate,
    remove_forbidden_xobjects,
)


def _make_image_xobject(pdf, interpolate=None):
    """Create a minimal Image XObject."""
    stream = pdf.make_stream(b"\x80")
    stream[Name.Type] = Name.XObject
    stream[Name.Subtype] = Name.Image
    stream[Name.Width] = 1
    stream[Name.Height] = 1
    stream[Name.ColorSpace] = Name.DeviceGray
    stream[Name.BitsPerComponent] = 8
    if interpolate is not None:
        stream["/Interpolate"] = interpolate
    return stream


def _make_form_xobject(pdf, resources=None):
    """Create a minimal Form XObject."""
    stream = pdf.make_stream(b"")
    stream[Name.Type] = Name.XObject
    stream[Name.Subtype] = Name.Form
    stream[Name.BBox] = Array([0, 0, 100, 100])
    if resources is not None:
        stream[Name.Resources] = resources
    return stream


def _get_first_inline_image_token_value(stream, token_name: str):
    """Return the value of one token in the first inline image of a stream."""
    for item in pikepdf.parse_content_stream(stream):
        if not isinstance(item, pikepdf.ContentStreamInlineImage):
            continue
        inline = item.operands[0]
        tokens = list(inline._image_object)
        for index in range(0, len(tokens) - 1, 2):
            if str(tokens[index]) == token_name:
                return tokens[index + 1]
    return None


class TestRemoveForbiddenXobjects:
    """Tests for remove_forbidden_xobjects()."""

    def test_no_xobjects(self, make_pdf_with_page):
        """Returns 0 for page without XObjects."""
        pdf = make_pdf_with_page()
        result = remove_forbidden_xobjects(pdf)
        assert result == 0

    def test_removes_postscript_xobject(self, make_pdf_with_page):
        """PostScript XObject (/PS) is removed."""
        pdf = make_pdf_with_page()
        ps_stream = pdf.make_stream(b"% PostScript")
        ps_stream[Name.Type] = Name.XObject
        ps_stream[Name.Subtype] = Name("/PS")
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(PS1=ps_stream))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        xobjects = pdf.pages[0].Resources.XObject
        assert "/PS1" not in xobjects

    def test_removes_ref_xobject(self, make_pdf_with_page):
        """Reference XObject (/Ref) is removed."""
        pdf = make_pdf_with_page()
        ref_stream = pdf.make_stream(b"")
        ref_stream[Name.Type] = Name.XObject
        ref_stream[Name.Subtype] = Name("/Ref")
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Ref1=ref_stream))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        xobjects = pdf.pages[0].Resources.XObject
        assert "/Ref1" not in xobjects

    def test_removes_ref_key_from_form_xobject(self, make_pdf_with_page):
        """Form XObject with /Ref key has /Ref stripped (Reference XObject)."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        form["/Ref"] = Dictionary(
            F=Dictionary(Type=Name.Filespec, F="external.pdf"),
            Page=0,
        )
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        # Form XObject kept, but /Ref stripped
        xobjects = pdf.pages[0].Resources.XObject
        assert "/Fm0" in xobjects
        from pdftopdfa.utils import resolve_indirect

        resolved = resolve_indirect(xobjects["/Fm0"])
        assert "/Ref" not in resolved

    def test_removes_subtype2_ps_key_from_form_xobject(self, make_pdf_with_page):
        """Form XObject with /Subtype2 /PS has the key stripped."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        form["/Subtype2"] = Name("/PS")
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        from pdftopdfa.utils import resolve_indirect

        resolved = resolve_indirect(pdf.pages[0].Resources.XObject["/Fm0"])
        assert "/Subtype2" not in resolved

    def test_removes_ps_key_from_form_xobject(self, make_pdf_with_page):
        """Form XObject with /PS key has the key stripped."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        form["/PS"] = pdf.make_stream(b"% postscript chunk")
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        from pdftopdfa.utils import resolve_indirect

        resolved = resolve_indirect(pdf.pages[0].Resources.XObject["/Fm0"])
        assert "/PS" not in resolved

    def test_keeps_image_xobject(self, make_pdf_with_page):
        """Image XObjects are kept (not forbidden)."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = remove_forbidden_xobjects(pdf)
        assert result == 0
        assert "/Im0" in pdf.pages[0].Resources.XObject

    def test_keeps_form_xobject(self, make_pdf_with_page):
        """Form XObjects are kept (not forbidden)."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = remove_forbidden_xobjects(pdf)
        assert result == 0
        assert "/Fm0" in pdf.pages[0].Resources.XObject

    def test_removes_alternates(self, make_pdf_with_page):
        """Alternates array is removed from XObjects."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        alt_image = _make_image_xobject(pdf)
        image["/Alternates"] = Array([Dictionary(Image=alt_image)])
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        resolved = pdf.pages[0].Resources.XObject["/Im0"]
        assert "/Alternates" not in resolved

    def test_removes_opi(self, make_pdf_with_page):
        """/OPI dictionary is removed from XObjects."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        image["/OPI"] = Dictionary(Version=Name("/2.0"))
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        resolved = pdf.pages[0].Resources.XObject["/Im0"]
        assert "/OPI" not in resolved

    def test_mixed_xobjects(self, make_pdf_with_page):
        """Only forbidden XObjects removed from mixed set."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        ps_stream = pdf.make_stream(b"% PS")
        ps_stream[Name.Type] = Name.XObject
        ps_stream[Name.Subtype] = Name("/PS")
        pdf.pages[0]["/Resources"] = Dictionary(
            XObject=Dictionary(Im0=image, PS1=ps_stream)
        )
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1
        xobjects = pdf.pages[0].Resources.XObject
        assert "/Im0" in xobjects
        assert "/PS1" not in xobjects

    def test_nested_forbidden_xobject_in_form(self, make_pdf_with_page):
        """Forbidden XObjects nested in Form XObjects are removed."""
        pdf = make_pdf_with_page()
        ps_stream = pdf.make_stream(b"% PS")
        ps_stream[Name.Type] = Name.XObject
        ps_stream[Name.Subtype] = Name("/PS")
        form = _make_form_xobject(
            pdf,
            resources=Dictionary(XObject=Dictionary(NestedPS=ps_stream)),
        )
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1

    def test_multiple_pages(self, make_pdf_with_page):
        """XObjects across multiple pages are processed."""
        pdf = make_pdf_with_page()
        page2 = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page2)

        for page in pdf.pages:
            ps = pdf.make_stream(b"% PS")
            ps[Name.Type] = Name.XObject
            ps[Name.Subtype] = Name("/PS")
            page["/Resources"] = Dictionary(XObject=Dictionary(PS1=ps))

        result = remove_forbidden_xobjects(pdf)
        assert result >= 2


class TestFixImageInterpolate:
    """Tests for fix_image_interpolate()."""

    def test_no_xobjects(self, make_pdf_with_page):
        """Returns 0 for page without XObjects."""
        pdf = make_pdf_with_page()
        result = fix_image_interpolate(pdf)
        assert result == 0

    def test_fixes_interpolate_true(self, make_pdf_with_page):
        """Image with Interpolate=true is fixed to false."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf, interpolate=True)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_image_interpolate(pdf)
        assert result == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert bool(img.get("/Interpolate")) is False

    def test_leaves_interpolate_false(self, make_pdf_with_page):
        """Image with Interpolate=false is left alone."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf, interpolate=False)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_image_interpolate(pdf)
        assert result == 0

    def test_leaves_no_interpolate(self, make_pdf_with_page):
        """Image without Interpolate key is left alone."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_image_interpolate(pdf)
        assert result == 0

    def test_skips_form_xobject(self, make_pdf_with_page):
        """Form XObjects (non-image) are not changed."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = fix_image_interpolate(pdf)
        assert result == 0

    def test_nested_image_in_form(self, make_pdf_with_page):
        """Nested images in Form XObjects are fixed."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf, interpolate=True)
        form = _make_form_xobject(
            pdf,
            resources=Dictionary(XObject=Dictionary(Im0=image)),
        )
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = fix_image_interpolate(pdf)
        assert result == 1

    def test_multiple_images(self, make_pdf_with_page):
        """All images with Interpolate=true are fixed."""
        pdf = make_pdf_with_page()
        img1 = _make_image_xobject(pdf, interpolate=True)
        img2 = _make_image_xobject(pdf, interpolate=True)
        img3 = _make_image_xobject(pdf, interpolate=False)
        pdf.pages[0]["/Resources"] = Dictionary(
            XObject=Dictionary(Im0=img1, Im1=img2, Im2=img3)
        )
        result = fix_image_interpolate(pdf)
        assert result == 2

    def test_fixes_inline_image_i_true_in_page_contents(self, make_pdf_with_page):
        """Inline image /I true in page contents is set to false."""
        pdf = make_pdf_with_page()
        pdf.pages[0]["/Contents"] = pdf.make_stream(
            b"q BI /W 1 /H 1 /BPC 8 /CS /G /I true ID \x80 EI Q\n"
        )
        result = fix_image_interpolate(pdf)
        assert result == 1
        token_value = _get_first_inline_image_token_value(pdf.pages[0].Contents, "/I")
        assert bool(token_value) is False

    def test_fixes_inline_image_interpolate_true_in_page_contents(
        self, make_pdf_with_page
    ):
        """Inline image /Interpolate true in page contents is set to false."""
        pdf = make_pdf_with_page()
        pdf.pages[0]["/Contents"] = pdf.make_stream(
            b"q BI /W 1 /H 1 /BPC 8 /CS /G /Interpolate true ID \x80 EI Q\n"
        )
        result = fix_image_interpolate(pdf)
        assert result == 1
        token_value = _get_first_inline_image_token_value(
            pdf.pages[0].Contents, "/Interpolate"
        )
        assert bool(token_value) is False


def _make_annotation_with_ap(pdf, ap_resources):
    """Create an annotation with an AP/N stream containing given resources.

    Args:
        pdf: Parent PDF object.
        ap_resources: Dictionary to use as Resources on the AP/N Form XObject.

    Returns:
        The annotation Dictionary.
    """
    ap_stream = _make_form_xobject(pdf, resources=ap_resources)
    return Dictionary(
        Type=Name.Annot,
        Subtype=Name("/Widget"),
        Rect=Array([0, 0, 100, 100]),
        AP=Dictionary(N=ap_stream),
    )


class TestRemoveForbiddenXobjectsInAPStream:
    """Tests for forbidden XObject removal in annotation AP streams."""

    def test_removes_forbidden_xobject_in_ap_stream(self, make_pdf_with_page):
        """PostScript XObject in AP/N stream is removed."""
        pdf = make_pdf_with_page()
        ps_stream = pdf.make_stream(b"% PostScript")
        ps_stream[Name.Type] = Name.XObject
        ps_stream[Name.Subtype] = Name("/PS")
        annot = _make_annotation_with_ap(
            pdf, Dictionary(XObject=Dictionary(PS1=ps_stream))
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1

    def test_removes_alternates_in_ap_stream(self, make_pdf_with_page):
        """Image with /Alternates in AP stream is cleaned."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        alt_image = _make_image_xobject(pdf)
        image["/Alternates"] = Array([Dictionary(Image=alt_image)])
        annot = _make_annotation_with_ap(pdf, Dictionary(XObject=Dictionary(Im0=image)))
        pdf.pages[0]["/Annots"] = Array([annot])
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1

    def test_ap_stream_substate_dict(self, make_pdf_with_page):
        """AP/N as sub-state dictionary is traversed."""
        pdf = make_pdf_with_page()
        ps_stream = pdf.make_stream(b"% PS")
        ps_stream[Name.Type] = Name.XObject
        ps_stream[Name.Subtype] = Name("/PS")
        yes_form = _make_form_xobject(
            pdf, resources=Dictionary(XObject=Dictionary(PS1=ps_stream))
        )
        off_form = _make_form_xobject(pdf)
        # AP/N is a dictionary with sub-states instead of a single stream
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 100, 100]),
            AP=Dictionary(N=Dictionary(Yes=yes_form, Off=off_form)),
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        result = remove_forbidden_xobjects(pdf)
        assert result >= 1

    def test_ap_stream_no_resources(self, make_pdf_with_page):
        """AP stream without Resources causes no error."""
        pdf = make_pdf_with_page()
        ap_stream = _make_form_xobject(pdf)  # No resources
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 100, 100]),
            AP=Dictionary(N=ap_stream),
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        result = remove_forbidden_xobjects(pdf)
        assert result == 0


class TestFixInterpolateInAPStream:
    """Tests for /Interpolate fix in annotation AP streams."""

    def test_fixes_interpolate_in_ap_stream(self, make_pdf_with_page):
        """Image with Interpolate=true in AP stream is fixed."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf, interpolate=True)
        annot = _make_annotation_with_ap(pdf, Dictionary(XObject=Dictionary(Im0=image)))
        pdf.pages[0]["/Annots"] = Array([annot])
        result = fix_image_interpolate(pdf)
        assert result == 1
        assert bool(image.get("/Interpolate")) is False

    def test_ap_stream_substate_dict_interpolate(self, make_pdf_with_page):
        """AP/N as sub-state dictionary — images are fixed."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf, interpolate=True)
        yes_form = _make_form_xobject(
            pdf, resources=Dictionary(XObject=Dictionary(Im0=image))
        )
        off_form = _make_form_xobject(pdf)
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 100, 100]),
            AP=Dictionary(N=Dictionary(Yes=yes_form, Off=off_form)),
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        result = fix_image_interpolate(pdf)
        assert result == 1
        assert bool(image.get("/Interpolate")) is False

    def test_ap_stream_no_resources_interpolate(self, make_pdf_with_page):
        """AP stream without Resources causes no error."""
        pdf = make_pdf_with_page()
        ap_stream = _make_form_xobject(pdf)
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 100, 100]),
            AP=Dictionary(N=ap_stream),
        )
        pdf.pages[0]["/Annots"] = Array([annot])
        result = fix_image_interpolate(pdf)
        assert result == 0


def _make_image_xobject_with_bpc(
    pdf, bpc, image_mask=False, width=1, height=1, colorspace=None, num_components=1
):
    """Create a minimal Image XObject with correctly-sized pixel data."""
    if bpc > 0:
        samples_per_row = width * num_components
        bits_per_row = samples_per_row * bpc
        bytes_per_row = (bits_per_row + 7) // 8
        total_bytes = bytes_per_row * height
        data = b"\x80" * max(1, total_bytes)
    else:
        data = b"\x80"
    stream = pdf.make_stream(data)
    stream[Name.Type] = Name.XObject
    stream[Name.Subtype] = Name.Image
    stream[Name.Width] = width
    stream[Name.Height] = height
    if colorspace is not None:
        stream[Name.ColorSpace] = colorspace
    else:
        stream[Name.ColorSpace] = Name.DeviceGray
    stream[Name.BitsPerComponent] = bpc
    if image_mask:
        stream["/ImageMask"] = True
        if "/ColorSpace" in stream:
            del stream[Name.ColorSpace]
    return stream


class TestFixBitsPerComponent:
    """Tests for fix_bits_per_component()."""

    def test_no_xobjects(self, make_pdf_with_page):
        """Returns zero counts for page without XObjects."""
        pdf = make_pdf_with_page()
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 0
        assert result["mask_bpc_fixed"] == 0

    def test_valid_bpc_values(self, make_pdf_with_page):
        """All valid BPC values (1, 2, 4, 8, 16) produce no fixes."""
        for bpc in (1, 2, 4, 8, 16):
            pdf = make_pdf_with_page()
            image = _make_image_xobject_with_bpc(pdf, bpc)
            pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
            result = fix_bits_per_component(pdf)
            assert result["invalid_bpc_fixed"] == 0, f"BPC={bpc} should be valid"
            assert result["mask_bpc_fixed"] == 0

    def test_invalid_bpc_value(self, make_pdf_with_page):
        """Invalid BPC value (e.g. 3) is fixed to 8."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 3)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8

    def test_invalid_bpc_value_zero(self, make_pdf_with_page):
        """BPC value of 0 is skipped (cannot unpack 0-bit samples)."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 0)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 0

    def test_invalid_bpc_value_32(self, make_pdf_with_page):
        """BPC value of 32 is fixed to 8."""
        pdf = make_pdf_with_page()
        # 32-bit → 4 bytes per sample for 1×1 grayscale
        image = _make_image_xobject_with_bpc(pdf, 32)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8

    def test_image_mask_bpc_1_valid(self, make_pdf_with_page):
        """Image mask with BPC=1 is valid (no fix needed)."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 1, image_mask=True)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 0
        assert result["mask_bpc_fixed"] == 0

    def test_image_mask_bpc_not_1(self, make_pdf_with_page):
        """Image mask with BPC=8 is fixed to BPC=1."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 8, image_mask=True)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["mask_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 1

    def test_no_bpc_key_no_fix(self, make_pdf_with_page):
        """Image without BitsPerComponent key produces no fix."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject(pdf)
        del image[Name.BitsPerComponent]
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 0
        assert result["mask_bpc_fixed"] == 0

    def test_multiple_images_mixed(self, make_pdf_with_page):
        """Multiple images: only invalid ones are fixed."""
        pdf = make_pdf_with_page()
        img_valid = _make_image_xobject_with_bpc(pdf, 8)
        img_invalid1 = _make_image_xobject_with_bpc(pdf, 3)
        img_invalid2 = _make_image_xobject_with_bpc(pdf, 7)
        pdf.pages[0]["/Resources"] = Dictionary(
            XObject=Dictionary(Im0=img_valid, Im1=img_invalid1, Im2=img_invalid2)
        )
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 2

    def test_nested_image_in_form(self, make_pdf_with_page):
        """Invalid BPC in nested Form XObject image is fixed."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 5)
        form = _make_form_xobject(
            pdf,
            resources=Dictionary(XObject=Dictionary(Im0=image)),
        )
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1

    def test_invalid_bpc_in_ap_stream(self, make_pdf_with_page):
        """Invalid BPC in annotation AP stream is fixed."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 6)
        annot = _make_annotation_with_ap(pdf, Dictionary(XObject=Dictionary(Im0=image)))
        pdf.pages[0]["/Annots"] = Array([annot])
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1

    def test_form_xobject_not_counted(self, make_pdf_with_page):
        """Form XObjects (non-image) are not fixed for BPC."""
        pdf = make_pdf_with_page()
        form = _make_form_xobject(pdf)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Fm0=form))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 0
        assert result["mask_bpc_fixed"] == 0


class TestBpcReencoding:
    """Tests for BPC re-encoding pixel data correctness."""

    def test_12bit_grayscale_to_8bit(self, make_pdf_with_page):
        """12-bit grayscale 2x2 image re-encoded to 8-bit with correct scaling."""
        pdf = make_pdf_with_page()
        # 2x2 DeviceGray, 12-bit: 2 samples/row × 12 bits = 24 bits = 3 bytes/row
        # Row 1: sample 0 = 0xFFF (4095), sample 1 = 0x800 (2048)
        # Row 2: sample 0 = 0x000 (0), sample 1 = 0x555 (1365)
        row1 = bytes([0xFF, 0xF8, 0x00])  # FFF 800
        row2 = bytes([0x00, 0x05, 0x55])  # 000 555
        data = row1 + row2
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 2
        image[Name.Height] = 2
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 12
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8
        new_data = img.read_bytes()
        assert len(new_data) == 4  # 2×2×1 = 4 bytes at 8-bit
        # 4095 → round(4095*255/4095) = 255
        # 2048 → round(2048*255/4095) = round(127.50) = 128
        # 0 → 0
        # 1365 → round(1365*255/4095) = round(84.96) = 85
        assert new_data[0] == 255
        assert new_data[1] == 128
        assert new_data[2] == 0
        assert new_data[3] == 85

    def test_5bit_to_8bit(self, make_pdf_with_page):
        """5-bit grayscale re-encoded to 8-bit."""
        pdf = make_pdf_with_page()
        # 1x1 DeviceGray, 5-bit: sample = 31 (max) → 0xF8 in MSB
        data = bytes([0xF8])  # 11111_000
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 5
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8
        new_data = img.read_bytes()
        # 31 → round(31*255/31) = 255
        assert new_data[0] == 255

    def test_7bit_to_8bit(self, make_pdf_with_page):
        """7-bit grayscale re-encoded to 8-bit."""
        pdf = make_pdf_with_page()
        # 1x1 DeviceGray, 7-bit: sample = 64 (half-range) → 0x80 in MSB
        data = bytes([0x80])  # 1000000_0
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 7
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8
        new_data = img.read_bytes()
        # 64 → round(64*255/127) = round(128.50...) = 128 or 129
        assert 128 <= new_data[0] <= 129

    def test_mask_bpc8_to_1bit(self, make_pdf_with_page):
        """8-bit mask thresholded to 1-bit."""
        pdf = make_pdf_with_page()
        # 4x1 mask, 8-bit: [255, 0, 128, 1]
        data = bytes([255, 0, 128, 1])
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 4
        image[Name.Height] = 1
        image["/ImageMask"] = True
        image[Name.BitsPerComponent] = 8
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["mask_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 1
        new_data = img.read_bytes()
        # 255→1, 0→0, 128→1, 1→0 → bits: 1010_0000 = 0xA0
        assert new_data[0] == 0xA0

    def test_mask_bpc4_to_1bit(self, make_pdf_with_page):
        """4-bit mask re-encoded to 1-bit."""
        pdf = make_pdf_with_page()
        # 2x1 mask, 4-bit: sample0=15 (0xF), sample1=0 → byte 0xF0
        data = bytes([0xF0])
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 2
        image[Name.Height] = 1
        image["/ImageMask"] = True
        image[Name.BitsPerComponent] = 4
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["mask_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 1
        new_data = img.read_bytes()
        # 15→1, 0→0 → bits: 10_000000 = 0x80
        assert new_data[0] == 0x80

    def test_rgb_3component_invalid_bpc(self, make_pdf_with_page):
        """DeviceRGB image with invalid BPC fixed to 8."""
        pdf = make_pdf_with_page()
        # 1x1 DeviceRGB, 3-bit: 3 components × 3 bits = 9 bits → 2 bytes
        # R=7, G=0, B=3 → 111_000_011_0000000 = 0xE180
        data = bytes([0xE1, 0x80])
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceRGB
        image[Name.BitsPerComponent] = 3
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8
        new_data = img.read_bytes()
        assert len(new_data) == 3  # 1×1×3 = 3 bytes at 8-bit
        # R: 7 → round(7*255/7) = 255
        assert new_data[0] == 255
        # G: 0 → 0
        assert new_data[1] == 0
        # B: 3 → round(3*255/7) = round(109.28) = 109
        assert new_data[2] == 109

    def test_cmyk_4component_invalid_bpc(self, make_pdf_with_page):
        """DeviceCMYK image with invalid BPC fixed to 8."""
        pdf = make_pdf_with_page()
        # 1x1 DeviceCMYK, 3-bit: 4 components × 3 bits = 12 bits → 2 bytes
        # C=7, M=0, Y=3, K=5 → 111_000_011_101_0000 = 0xE3A0
        data = bytes([0xE3, 0xA0])
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceCMYK
        image[Name.BitsPerComponent] = 3
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8
        new_data = img.read_bytes()
        assert len(new_data) == 4  # 1×1×4 = 4 bytes at 8-bit

    def test_iccbased_colorspace(self, make_pdf_with_page):
        """ICCBased with /N=3: component count detected and BPC fixed."""
        pdf = make_pdf_with_page()
        # Create an ICCBased colorspace with N=3
        icc_stream = pdf.make_stream(b"\x00" * 128)
        icc_stream["/N"] = 3
        cs = Array([Name("/ICCBased"), icc_stream])
        # 1x1, 3 components, 3-bit: 9 bits → 2 bytes
        data = bytes([0xFF, 0x80])
        image = pdf.make_stream(data)
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = cs
        image[Name.BitsPerComponent] = 3
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        assert result["invalid_bpc_fixed"] == 1
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.BitsPerComponent]) == 8

    def test_dctdecode_skipped(self, make_pdf_with_page):
        """JPEG image with invalid BPC is skipped (not re-encoded)."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 3)
        image[Name.Filter] = Name.DCTDecode
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        # Should not fix — DCTDecode stream cannot be re-encoded
        assert result["invalid_bpc_fixed"] == 0
        assert int(image[Name.BitsPerComponent]) == 3

    def test_indexed_colorspace_skipped(self, make_pdf_with_page):
        """Indexed colorspace image is skipped (cannot re-encode lookup table)."""
        pdf = make_pdf_with_page()
        # [/Indexed /DeviceRGB 255 <lookup>]
        lookup = pdf.make_stream(b"\x00" * 768)
        cs = Array([Name("/Indexed"), Name.DeviceRGB, 255, lookup])
        image = _make_image_xobject_with_bpc(pdf, 3)
        image[Name.ColorSpace] = cs
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        # _get_num_components returns None for Indexed → skip
        assert result["invalid_bpc_fixed"] == 0

    def test_decompression_failure_skipped(self, make_pdf_with_page):
        """Corrupt stream data is skipped gracefully."""
        pdf = make_pdf_with_page()
        # Create a stream with FlateDecode filter but invalid compressed data
        image = Stream(pdf, b"\x00\x01\x02\x03\x04")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 3
        image[Name.Filter] = Name.FlateDecode
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        result = fix_bits_per_component(pdf)
        # Should skip — decompression fails
        assert result["invalid_bpc_fixed"] == 0

    def test_preserves_width_height_colorspace(self, make_pdf_with_page):
        """Width, Height, and ColorSpace are unchanged after BPC fix."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 3, width=4, height=2)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        fix_bits_per_component(pdf)
        img = pdf.pages[0].Resources.XObject["/Im0"]
        assert int(img[Name.Width]) == 4
        assert int(img[Name.Height]) == 2
        assert str(img[Name.ColorSpace]) == "/DeviceGray"

    def test_data_length_correct(self, make_pdf_with_page):
        """After 8-bit fix: len(data) == W * H * N."""
        pdf = make_pdf_with_page()
        image = _make_image_xobject_with_bpc(pdf, 3, width=4, height=3)
        pdf.pages[0]["/Resources"] = Dictionary(XObject=Dictionary(Im0=image))
        fix_bits_per_component(pdf)
        img = pdf.pages[0].Resources.XObject["/Im0"]
        data = img.read_bytes()
        assert len(data) == 4 * 3 * 1  # W * H * 1 component (DeviceGray)
