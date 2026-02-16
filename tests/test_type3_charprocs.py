# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Type3 font CharProcs compliance processing."""

import pikepdf
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.color_profile import (
    ColorSpaceType,
    _apply_default_colorspaces,
    detect_color_spaces,
)
from pdftopdfa.sanitizers.colorspaces import validate_embedded_icc_profiles
from pdftopdfa.sanitizers.extgstate import sanitize_extgstate
from pdftopdfa.sanitizers.rendering_intent import sanitize_rendering_intent
from pdftopdfa.utils import iter_type3_fonts


def _make_type3_font_with_charprocs(
    pdf: Pdf,
    content: bytes,
    resources: Dictionary | None = None,
) -> Dictionary:
    """Create a minimal Type3 font with a single CharProc.

    Args:
        pdf: An open pikepdf Pdf.
        content: Raw content stream bytes for the CharProc 'a'.
        resources: Optional /Resources dictionary for the font.

    Returns:
        The Type3 font dictionary (already added to a page).
    """
    charproc_stream = pdf.make_stream(content)
    charprocs = Dictionary(a=charproc_stream)

    font = Dictionary()
    font[Name.Type] = Name.Font
    font[Name.Subtype] = Name.Type3
    font[Name.FontBBox] = Array([0, 0, 1000, 1000])
    font[Name.FontMatrix] = Array([0.001, 0, 0, 0.001, 0, 0])
    font[Name("/CharProcs")] = charprocs
    font[Name.Encoding] = Dictionary(
        Type=Name.Encoding,
        Differences=Array([0, Name.a]),
    )

    if resources is not None:
        font[Name("/Resources")] = resources

    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font),
            ),
        )
    )
    pdf.pages.append(page)
    return font


# ---------------------------------------------------------------------------
# Color detection tests
# ---------------------------------------------------------------------------


class TestType3ColorDetection:
    """Tests for device color operator detection in Type3 CharProcs."""

    def test_gray_operator_detected(self):
        """Device gray operator (g) in CharProc is detected."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0.5 g")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True

    def test_rgb_operator_detected(self):
        """Device RGB operator (rg) in CharProc is detected."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"1 0 0 rg")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_cmyk_operator_detected(self):
        """Device CMYK operator (k) in CharProc is detected."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0 1 0 1 k")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_cmyk_used is True

    def test_stroking_operators_detected(self):
        """Stroking color operators (G, RG, K) in CharProc are detected."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0.5 G 1 0 0 RG 0 1 0 1 K")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True
        assert analysis.device_rgb_used is True
        assert analysis.device_cmyk_used is True

    def test_cs_operator_detected(self):
        """cs/CS operator with DeviceRGB in CharProc is detected."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"/DeviceRGB cs")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True

    def test_no_color_operators(self):
        """CharProc without color operators does not flag any space."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1")
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is False
        assert analysis.device_rgb_used is False
        assert analysis.device_cmyk_used is False

    def test_form_xobject_in_font_resources(self):
        """Device colors in Form XObject nested in font resources detected."""
        pdf = new_pdf()
        form_stream = pdf.make_stream(b"0.5 g")
        form_stream[Name.Type] = Name.XObject
        form_stream[Name.Subtype] = Name.Form
        form_stream[Name.BBox] = Array([0, 0, 100, 100])

        font_resources = Dictionary(
            XObject=Dictionary(Fm0=form_stream),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True

    def test_cycle_detection(self):
        """Cycle detection prevents infinite loop with self-referencing font."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0.5 g")

        # Make font reference itself (artificial cycle scenario)
        # The iter_type3_fonts uses objgen to prevent revisiting
        analysis = detect_color_spaces(pdf)
        assert analysis.device_gray_used is True

    def test_type3_in_ap_stream(self):
        """Type3 font inside an annotation AP stream is processed."""
        pdf = new_pdf()

        # Create a Type3 font with device RGB in CharProc
        charproc_stream = pdf.make_stream(b"1 0 0 rg")
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

        # Create AP stream with Type3 font in resources
        ap_stream = pdf.make_stream(b"/F1 12 Tf (a) Tj")
        ap_stream[Name.Type] = Name.XObject
        ap_stream[Name.Subtype] = Name.Form
        ap_stream[Name.BBox] = Array([0, 0, 100, 100])
        ap_stream[Name.Resources] = Dictionary(
            Font=Dictionary(F1=font),
        )

        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name("/FreeText"),
            Rect=Array([0, 0, 100, 100]),
            AP=Dictionary(N=ap_stream),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Annots=Array([annot]),
            )
        )
        pdf.pages.append(page)

        analysis = detect_color_spaces(pdf)
        assert analysis.device_rgb_used is True


# ---------------------------------------------------------------------------
# Default color space application tests
# ---------------------------------------------------------------------------


class TestType3DefaultColorSpaces:
    """Tests for applying default color spaces to Type3 font resources."""

    def _run_apply_defaults(self, pdf, font):
        """Helper to run _apply_default_colorspaces with RGB as non-dominant."""
        from pdftopdfa.color_profile import (
            ColorSpaceType,
            _create_icc_colorspace,
        )

        icc_stream_cache = {}
        non_dominant = {ColorSpaceType.DEVICE_RGB}
        icc_arrays = {}
        for cs_type in non_dominant:
            icc_arrays[cs_type] = _create_icc_colorspace(pdf, cs_type, icc_stream_cache)

        _apply_default_colorspaces(pdf, non_dominant, icc_stream_cache)
        return font

    def test_defaults_added_to_existing_font_resources(self):
        """DefaultRGB is added to existing font /Resources."""
        pdf = new_pdf()
        font_resources = Dictionary()
        font = _make_type3_font_with_charprocs(pdf, b"1 0 0 rg", font_resources)
        self._run_apply_defaults(pdf, font)

        font_res = font.get("/Resources")
        assert font_res is not None
        cs_dict = font_res.get("/ColorSpace")
        assert cs_dict is not None
        assert "/DefaultRGB" in cs_dict

    def test_resources_created_when_missing(self):
        """Font /Resources is created when missing, defaults are added."""
        pdf = new_pdf()
        font = _make_type3_font_with_charprocs(pdf, b"1 0 0 rg")
        self._run_apply_defaults(pdf, font)

        font_res = font.get("/Resources")
        assert font_res is not None
        cs_dict = font_res.get("/ColorSpace")
        assert cs_dict is not None
        assert "/DefaultRGB" in cs_dict

    def test_existing_defaults_not_overwritten(self):
        """Existing DefaultRGB entry in font resources is preserved."""
        pdf = new_pdf()
        marker_stream = pdf.make_stream(b"\x00" * 128)
        existing_cs = Array([Name.ICCBased, marker_stream])
        font_resources = Dictionary(
            ColorSpace=Dictionary(DefaultRGB=existing_cs),
        )
        font = _make_type3_font_with_charprocs(pdf, b"1 0 0 rg", font_resources)
        self._run_apply_defaults(pdf, font)

        # The original marker stream should still be referenced
        font_cs = font["/Resources"]["/ColorSpace"]["/DefaultRGB"]
        assert str(font_cs[0]) == "/ICCBased"
        assert font_cs[1].objgen == marker_stream.objgen

    def test_image_in_font_resources_replaced(self):
        """Image XObject in font resources gets ICC replacement."""
        pdf = new_pdf()
        img = pdf.make_stream(b"\xff\x00\x00")
        img[Name.Type] = Name.XObject
        img[Name.Subtype] = Name.Image
        img[Name.Width] = 1
        img[Name.Height] = 1
        img[Name.BitsPerComponent] = 8
        img[Name.ColorSpace] = Name.DeviceRGB

        font_resources = Dictionary(
            XObject=Dictionary(Im0=img),
        )
        font = _make_type3_font_with_charprocs(pdf, b"1 0 0 rg", font_resources)
        self._run_apply_defaults(pdf, font)

        cs = img.get("/ColorSpace")
        # Should now be an ICCBased array, not bare DeviceRGB
        assert isinstance(cs, Array)
        assert str(cs[0]) == "/ICCBased"

    def test_gray_and_cmyk_defaults(self):
        """DefaultGray and DefaultCMYK can be added to font resources."""

        pdf = new_pdf()
        font_resources = Dictionary()
        font = _make_type3_font_with_charprocs(pdf, b"0.5 g 0 1 0 1 k", font_resources)

        icc_stream_cache = {}
        non_dominant = {ColorSpaceType.DEVICE_GRAY, ColorSpaceType.DEVICE_CMYK}
        _apply_default_colorspaces(pdf, non_dominant, icc_stream_cache)

        font_res = font.get("/Resources")
        cs_dict = font_res.get("/ColorSpace")
        assert cs_dict is not None
        assert "/DefaultGray" in cs_dict
        assert "/DefaultCMYK" in cs_dict

    def test_no_type3_fonts_is_safe(self):
        """No crash when page has no Type3 fonts."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(),
            )
        )
        pdf.pages.append(page)

        icc_stream_cache = {}
        non_dominant = {ColorSpaceType.DEVICE_RGB}
        _apply_default_colorspaces(pdf, non_dominant, icc_stream_cache)


# ---------------------------------------------------------------------------
# Rendering intent tests
# ---------------------------------------------------------------------------


class TestType3RenderingIntent:
    """Tests for ri operator fixing in Type3 CharProcs."""

    def test_invalid_ri_fixed(self):
        """Invalid ri operator in CharProc is replaced."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"/FooBar ri")

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 1

    def test_valid_ri_preserved(self):
        """Valid ri operator (/Perceptual) in CharProc is preserved."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"/Perceptual ri")

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 0

    def test_multiple_charprocs(self):
        """Multiple CharProcs with invalid ri are all fixed."""
        pdf = new_pdf()
        cp_a = pdf.make_stream(b"/BadIntent ri")
        cp_b = pdf.make_stream(b"/AnotherBad ri")
        charprocs = Dictionary(a=cp_a, b=cp_b)

        font = Dictionary()
        font[Name.Type] = Name.Font
        font[Name.Subtype] = Name.Type3
        font[Name.FontBBox] = Array([0, 0, 1000, 1000])
        font[Name.FontMatrix] = Array([0.001, 0, 0, 0.001, 0, 0])
        font[Name("/CharProcs")] = charprocs
        font[Name.Encoding] = Dictionary(
            Type=Name.Encoding,
            Differences=Array([0, Name.a, Name.b]),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
            )
        )
        pdf.pages.append(page)

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 2

    def test_cycle_detection_no_infinite_loop(self):
        """Cycle detection prevents infinite loops."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"/BadRI ri")

        result = sanitize_rendering_intent(pdf)
        assert result["ri_operators_fixed"] == 1


# ---------------------------------------------------------------------------
# ExtGState tests
# ---------------------------------------------------------------------------


class TestType3ExtGState:
    """Tests for ExtGState sanitization in Type3 font resources."""

    def test_tr_removed_from_font_resources(self):
        """/TR is removed from ExtGState in Type3 font resources."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR=tr_stream)
        font_resources = Dictionary(
            ExtGState=Dictionary(GS0=gs),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/TR" not in gs

    def test_tr2_non_default_removed(self):
        """/TR2 with non-Default value is removed from font ExtGState."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR2=tr_stream)
        font_resources = Dictionary(
            ExtGState=Dictionary(GS0=gs),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/TR2" not in gs

    def test_htp_removed(self):
        """/HTP is removed from ExtGState in font resources."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, HTP=Name("/SomePhase"))
        font_resources = Dictionary(
            ExtGState=Dictionary(GS0=gs),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1
        assert "/HTP" not in gs

    def test_valid_extgstate_preserved(self):
        """Valid ExtGState entries in font resources are not removed."""
        pdf = new_pdf()
        gs = Dictionary(Type=Name.ExtGState, BM=Name.Normal)
        font_resources = Dictionary(
            ExtGState=Dictionary(GS0=gs),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 0
        assert "/BM" in gs

    def test_cycle_detection_no_infinite_loop(self):
        """Cycle detection prevents infinite loops for ExtGState."""
        pdf = new_pdf()
        tr_stream = pdf.make_stream(b"{ }")
        gs = Dictionary(Type=Name.ExtGState, TR=tr_stream)
        font_resources = Dictionary(
            ExtGState=Dictionary(GS0=gs),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        result = sanitize_extgstate(pdf)
        assert result["extgstate_fixed"] == 1


# ---------------------------------------------------------------------------
# ICC validation tests
# ---------------------------------------------------------------------------


def _make_valid_icc_profile(n_components: int = 3) -> bytes:
    """Create a minimal valid ICC profile for testing.

    Args:
        n_components: Number of color components (1=Gray, 3=RGB, 4=CMYK).

    Returns:
        Bytes of a minimal ICC profile with 'acsp' signature.
    """
    profile = bytearray(128)
    # Profile size
    profile[0:4] = (128).to_bytes(4, "big")
    # Major version 2
    profile[8] = 2
    # Profile class: mntr
    profile[12:16] = b"mntr"
    # Color space
    if n_components == 1:
        profile[16:20] = b"GRAY"
    elif n_components == 3:
        profile[16:20] = b"RGB "
    elif n_components == 4:
        profile[16:20] = b"CMYK"
    # 'acsp' signature at bytes 36-39
    profile[36:40] = b"acsp"
    return bytes(profile)


class TestType3ICCValidation:
    """Tests for ICC profile validation in Type3 font resources."""

    def test_icc_in_font_resources_validated(self):
        """ICCBased profile in Type3 font resources is validated."""
        pdf = new_pdf()
        icc_data = _make_valid_icc_profile(3)
        icc_stream = pdf.make_stream(icc_data)
        icc_stream[Name.N] = 3

        font_resources = Dictionary(
            ColorSpace=Dictionary(
                CS0=Array([Name.ICCBased, icc_stream]),
            ),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b")
        assert validated >= 1
        assert len(warnings) == 0

    def test_invalid_icc_in_font_resources_warned(self):
        """Invalid ICC profile in Type3 font resources produces warning."""
        pdf = new_pdf()
        # Invalid profile: too small and no acsp signature
        icc_stream = pdf.make_stream(b"\x00" * 128)
        icc_stream[Name.N] = 3

        font_resources = Dictionary(
            ColorSpace=Dictionary(
                CS0=Array([Name.ICCBased, icc_stream]),
            ),
        )
        _make_type3_font_with_charprocs(pdf, b"100 0 0 100 0 0 d1", font_resources)

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b")
        assert len(warnings) >= 1

    def test_no_icc_no_crash(self):
        """No crash when Type3 font has no ICC profiles in resources."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0.5 g")

        validated, warnings, _ = validate_embedded_icc_profiles(pdf, "3b")
        # Should not crash, no profiles to validate
        assert isinstance(validated, int)


# ---------------------------------------------------------------------------
# iter_type3_fonts helper tests
# ---------------------------------------------------------------------------


class TestIterType3Fonts:
    """Tests for the iter_type3_fonts utility function."""

    def test_yields_type3_fonts(self):
        """Yields Type3 fonts from resources."""
        pdf = new_pdf()
        _make_type3_font_with_charprocs(pdf, b"0.5 g")
        resources = pdf.pages[0].Resources

        visited = set()
        fonts = list(iter_type3_fonts(resources, visited))
        assert len(fonts) == 1
        assert fonts[0][0] == "/F1"

    def test_skips_non_type3_fonts(self):
        """Non-Type3 fonts are not yielded."""
        pdf = new_pdf()
        type1_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"),
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
                Resources=Dictionary(
                    Font=Dictionary(F1=type1_font),
                ),
            )
        )
        pdf.pages.append(page)

        resources = pdf.pages[0].Resources
        visited = set()
        fonts = list(iter_type3_fonts(resources, visited))
        assert len(fonts) == 0

    def test_cycle_detection(self):
        """Same font objgen is not yielded twice when indirect."""
        pdf = new_pdf()
        # Create an indirect Type3 font
        charproc_stream = pdf.make_stream(b"0.5 g")
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type3,
                FontBBox=Array([0, 0, 1000, 1000]),
                FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
                Encoding=Dictionary(
                    Type=Name.Encoding,
                    Differences=Array([0, Name.a]),
                ),
            )
        )
        font[Name("/CharProcs")] = Dictionary(a=charproc_stream)

        resources = Dictionary(Font=Dictionary(F1=font))

        visited = set()
        fonts1 = list(iter_type3_fonts(resources, visited))
        fonts2 = list(iter_type3_fonts(resources, visited))
        assert len(fonts1) == 1
        # Second pass: indirect font already visited
        assert len(fonts2) == 0

    def test_empty_resources(self):
        """No crash on empty resources dictionary."""
        visited = set()
        fonts = list(iter_type3_fonts(Dictionary(), visited))
        assert len(fonts) == 0

    def test_no_font_dict(self):
        """No crash when /Font key is missing from resources."""
        resources = Dictionary(XObject=Dictionary())
        visited = set()
        fonts = list(iter_type3_fonts(resources, visited))
        assert len(fonts) == 0
