# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for sanitizers/fonts.py (CIDFont sanitizer)."""

import logging
from io import BytesIO

import pikepdf
from conftest import new_pdf, open_pdf, resolve
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.fonts import (
    _STANDARD_CMAP_NAMES,
    _get_cidsysteminfo_from_cmap,
    _strip_subset_prefix,
    sanitize_cidfont_structures,
    sanitize_fontname_consistency,
)


def _make_type0_font(
    pdf: Pdf,
    *,
    encoding: str = "Identity-H",
    cidfont_subtype: str = "/CIDFontType2",
    registry: str = "Adobe",
    ordering: str = "Identity",
    supplement: int = 0,
    add_cidtogidmap: bool = True,
    add_cidset: bool = False,
) -> Dictionary:
    """Create a Type0 font with a CIDFont descendant.

    Returns the font dictionary.
    """
    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestFont"),
            Flags=4,
        )
    )

    if add_cidset:
        cidset_stream = pdf.make_stream(b"\xff")
        font_descriptor[Name.CIDSet] = cidset_stream

    cidfont = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name(cidfont_subtype),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry=registry,
                Ordering=ordering,
                Supplement=supplement,
            ),
            FontDescriptor=font_descriptor,
        )
    )

    if add_cidtogidmap:
        cidfont[Name.CIDToGIDMap] = Name.Identity

    type0_font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestFont"),
            Encoding=Name("/" + encoding),
            DescendantFonts=Array([cidfont]),
        )
    )

    return type0_font


def _make_embedded_simple_font(
    pdf: Pdf,
    *,
    subtype: str = "/Type1",
    add_charset: bool = True,
    embedded: bool = True,
) -> Dictionary:
    """Create a simple Type1/MMType1 font with optional /CharSet."""
    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/TestSimpleFont"),
            Flags=32,
        )
    )

    if embedded:
        # Minimal Type1-like payload; parser validity is irrelevant here.
        font_stream = pdf.make_stream(b"%!PS-AdobeFont-1.0: TestSimpleFont 1.0\n")
        font_descriptor[Name.FontFile] = pdf.make_indirect(font_stream)

    if add_charset:
        font_descriptor[Name("/CharSet")] = "/.notdef/space/A"

    return pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name(subtype),
            BaseFont=Name("/TestSimpleFont"),
            FontDescriptor=font_descriptor,
        )
    )


def _build_pdf_with_type0_font(pdf: Pdf, type0_font: Dictionary) -> None:
    """Add a page with the given Type0 font to the PDF."""
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(
            Font=Dictionary(F1=type0_font),
        ),
    )
    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)


def _roundtrip(pdf: Pdf) -> Pdf:
    """Save and reopen a PDF to get proper indirect references."""
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return open_pdf(buf)


def _get_cidfont(pdf: Pdf) -> Dictionary:
    """Get the first CIDFont dict from the first page's first Type0 font."""
    font = resolve(pdf.pages[0].Resources.Font["/F1"])
    return resolve(font.DescendantFonts[0])


class TestCIDSystemInfoFix:
    """Tests for CIDSystemInfo consistency fix."""

    def test_mismatched_cidsysteminfo_is_corrected(self) -> None:
        """CIDSystemInfo mismatch between CMap and CIDFont is corrected."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            registry="Adobe",
            ordering="Japan1",
            supplement=6,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 1

        # Roundtrip to verify the changes persisted
        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        csi = cidfont.CIDSystemInfo
        assert str(csi.Registry) == "Adobe"
        assert str(csi.Ordering) == "Identity"
        assert int(csi.Supplement) == 0

    def test_cidfonttype0_identity_h_preserves_existing_ordering(self) -> None:
        """CIDFontType0 with Identity-H keeps its original CIDSystemInfo."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Japan1",
            supplement=4,
            add_cidtogidmap=False,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 0

        # Verify CIDSystemInfo was NOT changed to Identity-0
        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        csi = cidfont.CIDSystemInfo
        assert str(csi.Registry) == "Adobe"
        assert str(csi.Ordering) == "Japan1"
        assert int(csi.Supplement) == 4

    def test_correct_cidsysteminfo_not_modified(self) -> None:
        """CIDSystemInfo that already matches the CMap is not modified."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            registry="Adobe",
            ordering="Identity",
            supplement=0,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 0

    def test_named_cmap_unijis(self) -> None:
        """UniJIS-UTF16-H CMap sets CIDSystemInfo to Adobe-Japan1-6."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="UniJIS-UTF16-H",
            registry="Adobe",
            ordering="Identity",
            supplement=0,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 1

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDSystemInfo.Ordering) == "Japan1"
        assert int(cidfont.CIDSystemInfo.Supplement) == 6

    def test_stream_cmap_cidsysteminfo(self) -> None:
        """Stream CMap's CIDSystemInfo is extracted and applied."""
        pdf = new_pdf()

        cmap_data = b"""%!PS-Adobe-3.0 Resource-CMap
%%BeginResource: CMap (TestCMap)
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo 3 dict dup begin
  /Registry (Adobe) def
  /Ordering (Korea1) def
  /Supplement 2 def
end def
endcmap
CMapName currentdict /CMap defineresource pop
end
end
%%EndResource
"""
        cmap_stream = pdf.make_stream(cmap_data)

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",  # will be overridden
            registry="Adobe",
            ordering="Identity",
            supplement=0,
        )
        font_obj = resolve(font)
        font_obj[Name.Encoding] = cmap_stream

        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 1

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDSystemInfo.Registry) == "Adobe"
        assert str(cidfont.CIDSystemInfo.Ordering) == "Korea1"
        assert int(cidfont.CIDSystemInfo.Supplement) == 2


class TestCIDToGIDMapFix:
    """Tests for CIDToGIDMap fix."""

    def test_cidtogidmap_added_to_cidfonttype2(self) -> None:
        """CIDToGIDMap is added to CIDFontType2 fonts that lack it."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=False,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 1

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDToGIDMap) == "/Identity"

    def test_cidtogidmap_not_added_to_cidfonttype0(self) -> None:
        """CIDToGIDMap is NOT added to CIDFontType0 fonts."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType0",
            add_cidtogidmap=False,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 0

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert "/CIDToGIDMap" not in cidfont

    def test_existing_cidtogidmap_not_modified(self) -> None:
        """Existing CIDToGIDMap is not touched."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=True,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 0


class TestCIDSetRemoval:
    """Tests for CIDSet removal."""

    def test_cidset_is_removed(self) -> None:
        """CIDSet in font descriptor is removed."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            add_cidset=True,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidset_removed"] == 1

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        fd = resolve(cidfont.FontDescriptor)
        assert "/CIDSet" not in fd

    def test_no_cidset_no_change(self) -> None:
        """No CIDSet means nothing to remove."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            add_cidset=False,
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidset_removed"] == 0


class TestSimpleFonts:
    """Tests that simple (non-CID) fonts are not affected."""

    def test_type1_font_not_affected(self) -> None:
        """Type1 fonts are ignored by the CIDFont sanitizer."""
        pdf = new_pdf()

        type1_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type1,
                BaseFont=Name("/Helvetica"),
            )
        )

        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=type1_font),
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 0
        assert result["cidtogidmap_fixed"] == 0
        assert result["cidset_removed"] == 0
        assert result["type1_charset_removed"] == 0


class TestType1CharSetRemoval:
    """Tests for Type1/MMType1 /CharSet removal."""

    def test_embedded_type1_charset_removed(self) -> None:
        """Embedded Type1 font with /CharSet has the entry removed."""
        pdf = new_pdf()
        type1_font = _make_embedded_simple_font(
            pdf, subtype="/Type1", add_charset=True, embedded=True
        )
        _build_pdf_with_type0_font(pdf, type1_font)

        result = sanitize_cidfont_structures(pdf)

        assert result["type1_charset_removed"] == 1

        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        fd = resolve(font_obj["/FontDescriptor"])
        assert "/CharSet" not in fd

    def test_embedded_mmtype1_charset_removed(self) -> None:
        """Embedded MMType1 font with /CharSet has the entry removed."""
        pdf = new_pdf()
        mm_font = _make_embedded_simple_font(
            pdf, subtype="/MMType1", add_charset=True, embedded=True
        )
        _build_pdf_with_type0_font(pdf, mm_font)

        result = sanitize_cidfont_structures(pdf)

        assert result["type1_charset_removed"] == 1

    def test_non_embedded_type1_charset_not_removed(self) -> None:
        """Non-embedded Type1 font is skipped for /CharSet removal."""
        pdf = new_pdf()
        type1_font = _make_embedded_simple_font(
            pdf, subtype="/Type1", add_charset=True, embedded=False
        )
        _build_pdf_with_type0_font(pdf, type1_font)

        result = sanitize_cidfont_structures(pdf)

        assert result["type1_charset_removed"] == 0
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        fd = resolve(font_obj["/FontDescriptor"])
        assert "/CharSet" in fd

    def test_type1_charset_removed_in_form_xobject(self) -> None:
        """Embedded Type1 /CharSet inside Form XObject is removed."""
        pdf = new_pdf()
        type1_font = _make_embedded_simple_font(
            pdf, subtype="/Type1", add_charset=True, embedded=True
        )
        _build_page_with_form_xobject_font(pdf, type1_font)

        result = sanitize_cidfont_structures(pdf)

        assert result["type1_charset_removed"] == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_pdf_without_fonts(self) -> None:
        """PDF without any fonts doesn't crash."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 0
        assert result["cidtogidmap_fixed"] == 0
        assert result["cidset_removed"] == 0

    def test_empty_pdf(self) -> None:
        """Empty PDF (no pages) doesn't crash."""
        pdf = new_pdf()

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 0
        assert result["cidtogidmap_fixed"] == 0
        assert result["cidset_removed"] == 0


class TestGetCIDSystemInfoFromCMap:
    """Tests for _get_cidsysteminfo_from_cmap helper."""

    def test_identity_h(self) -> None:
        result = _get_cidsysteminfo_from_cmap(Name("/Identity-H"))
        assert result == ("Adobe", "Identity", 0)

    def test_identity_v(self) -> None:
        result = _get_cidsysteminfo_from_cmap(Name("/Identity-V"))
        assert result == ("Adobe", "Identity", 0)

    def test_unigb(self) -> None:
        result = _get_cidsysteminfo_from_cmap(Name("/UniGB-UTF16-H"))
        assert result == ("Adobe", "GB1", 5)

    def test_unicns(self) -> None:
        result = _get_cidsysteminfo_from_cmap(Name("/UniCNS-UTF16-H"))
        assert result == ("Adobe", "CNS1", 6)

    def test_uniks(self) -> None:
        result = _get_cidsysteminfo_from_cmap(Name("/UniKS-UTF16-H"))
        assert result == ("Adobe", "Korea1", 2)

    def test_unknown_name_returns_none(self) -> None:
        assert _get_cidsysteminfo_from_cmap(Name("/UnknownCMap")) is None


def _build_page_with_form_xobject_font(pdf: Pdf, type0_font: Dictionary) -> None:
    """Add a page with a Form XObject containing the given Type0 font."""
    form_xobj = pdf.make_stream(b"")
    form_xobj[Name.Type] = Name.XObject
    form_xobj[Name.Subtype] = Name.Form
    form_xobj[Name.BBox] = Array([0, 0, 100, 100])
    form_xobj[Name.Resources] = pdf.make_indirect(
        Dictionary(Font=Dictionary(F1=type0_font))
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(
            XObject=Dictionary(Fm1=pdf.make_indirect(form_xobj)),
        ),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


def _build_page_with_annotation_font(pdf: Pdf, type0_font: Dictionary) -> None:
    """Add a page with an Annotation Appearance Stream containing the font."""
    ap_stream = pdf.make_stream(b"")
    ap_stream[Name.Type] = Name.XObject
    ap_stream[Name.Subtype] = Name.Form
    ap_stream[Name.BBox] = Array([0, 0, 50, 50])
    ap_stream[Name.Resources] = pdf.make_indirect(
        Dictionary(Font=Dictionary(F1=type0_font))
    )

    annot = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name("/Widget"),
            Rect=Array([0, 0, 50, 50]),
            AP=Dictionary(N=pdf.make_indirect(ap_stream)),
        )
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Annots=Array([annot]),
    )
    pdf.pages.append(pikepdf.Page(page_dict))


class TestNestedCIDFonts:
    """Tests that CIDFonts in nested structures are found and fixed."""

    def test_cidfont_in_form_xobject_is_fixed(self) -> None:
        """CIDFont inside a Form XObject gets CIDSystemInfo corrected."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            registry="Adobe",
            ordering="Japan1",
            supplement=6,
        )
        _build_page_with_form_xobject_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 1

    def test_cidtogidmap_added_in_form_xobject(self) -> None:
        """CIDFontType2 inside Form XObject gets CIDToGIDMap added."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=False,
        )
        _build_page_with_form_xobject_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 1

    def test_cidset_removed_in_form_xobject(self) -> None:
        """CIDSet inside Form XObject font descriptor is removed."""
        pdf = new_pdf()
        font = _make_type0_font(pdf, add_cidset=True)
        _build_page_with_form_xobject_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidset_removed"] == 1

    def test_cidfont_in_annotation_ap_is_fixed(self) -> None:
        """CIDFont inside Annotation Appearance Stream gets fixed."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            registry="Adobe",
            ordering="Japan1",
            supplement=6,
            add_cidtogidmap=False,
            add_cidset=True,
        )
        _build_page_with_annotation_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidsysteminfo_fixed"] == 1
        assert result["cidtogidmap_fixed"] == 1
        assert result["cidset_removed"] == 1

    def test_shared_font_deduplicated(self) -> None:
        """Same font object in page-level and Form XObject is counted once."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            registry="Adobe",
            ordering="Japan1",
            supplement=6,
        )

        # Create a Form XObject with the same font
        form_xobj = pdf.make_stream(b"")
        form_xobj[Name.Type] = Name.XObject
        form_xobj[Name.Subtype] = Name.Form
        form_xobj[Name.BBox] = Array([0, 0, 100, 100])
        form_xobj[Name.Resources] = pdf.make_indirect(
            Dictionary(Font=Dictionary(F1=font))
        )

        # Page has the font both directly and via Form XObject
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(
                Font=Dictionary(F1=font),
                XObject=Dictionary(Fm1=pdf.make_indirect(form_xobj)),
            ),
        )
        pdf.pages.append(pikepdf.Page(page_dict))

        result = sanitize_cidfont_structures(pdf)

        # Should only fix once despite appearing in two places
        assert result["cidsysteminfo_fixed"] == 1


class TestCIDToGIDMapInvalidValues:
    """Tests for CIDToGIDMap fix with invalid (non-None) values."""

    def test_invalid_name_replaced_with_identity(self) -> None:
        """CIDToGIDMap with invalid Name value is replaced with /Identity."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=False,
        )
        # Set invalid Name value (like /NoIdentity in corpus fail-a)
        cidfont = resolve(resolve(font).DescendantFonts[0])
        cidfont[Name.CIDToGIDMap] = Name("/NoIdentity")
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 1
        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDToGIDMap) == "/Identity"

    def test_wrong_name_replaced_with_identity(self) -> None:
        """CIDToGIDMap with arbitrary wrong Name is replaced with /Identity."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=False,
        )
        cidfont = resolve(resolve(font).DescendantFonts[0])
        cidfont[Name.CIDToGIDMap] = Name("/WrongValue")
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 1
        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDToGIDMap) == "/Identity"

    def test_stream_cidtogidmap_preserved(self) -> None:
        """CIDToGIDMap that is a valid Stream is not replaced."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=False,
        )
        cidfont = resolve(resolve(font).DescendantFonts[0])
        # Create a valid CIDToGIDMap stream (identity mapping for 256 CIDs)
        gid_data = b"".join(i.to_bytes(2, "big") for i in range(256))
        cidfont[Name.CIDToGIDMap] = pdf.make_stream(gid_data)
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 0

    def test_identity_cidtogidmap_preserved(self) -> None:
        """CIDToGIDMap /Identity is not replaced."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            cidfont_subtype="/CIDFontType2",
            add_cidtogidmap=True,  # adds /Identity
        )
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cidtogidmap_fixed"] == 0


def _make_cmap_stream(
    pdf: Pdf,
    *,
    registry: str = "Adobe",
    ordering: str = "Korea1",
    supplement: int = 2,
    wmode: int = 0,
    cmap_name: str = "Adobe-Korea1-2",
) -> pikepdf.Stream:
    """Create a minimal embedded CMap stream."""
    data = f"""%!PS-Adobe-3.0 Resource-CMap
%%BeginResource: CMap ({cmap_name})
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo 3 dict dup begin
  /Registry ({registry}) def
  /Ordering ({ordering}) def
  /Supplement {supplement} def
end def
/CMapName /{cmap_name} def
/WMode {wmode} def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0000> <FFFF> 0
endcidrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
%%EndResource
"""
    stream = pdf.make_stream(data.encode("latin-1"))
    stream[Name.Type] = Name("/CMap")
    stream[Name("/CMapName")] = Name("/" + cmap_name)
    stream[Name("/WMode")] = wmode
    stream[Name.CIDSystemInfo] = Dictionary(
        Registry=registry,
        Ordering=ordering,
        Supplement=supplement,
    )
    return stream


class TestCMapNonStandardName:
    """Tests for non-standard CMap Name replacement (6.2.11.3.3 t01)."""

    def test_nonstandard_cmap_name_replaced_with_identity_h(self) -> None:
        """Non-standard CMap Name is replaced with /Identity-H."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",  # will be overridden
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = Name("/Adobe-Korea1-2")
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        assert str(font_obj.Encoding) == "/Identity-H"

    def test_nonstandard_vertical_cmap_replaced_with_identity_v(self) -> None:
        """Non-standard vertical CMap Name is replaced with /Identity-V."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = Name("/Adobe-Korea1-2-V")
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        assert str(font_obj.Encoding) == "/Identity-V"

    def test_standard_cmap_names_preserved(self) -> None:
        """Standard CMap names from ISO 32000-1 Table 118 are not replaced."""
        for name in [
            "Identity-H",
            "Identity-V",
            "UniJIS-UTF16-H",
            "H",
            "V",
            "90ms-RKSJ-H",
            "KSC-EUC-H",
            "GB-EUC-H",
            "B5pc-H",
        ]:
            pdf = new_pdf()
            font = _make_type0_font(
                pdf,
                encoding=name,
                cidfont_subtype="/CIDFontType0",
                add_cidtogidmap=False,
            )
            _build_pdf_with_type0_font(pdf, font)

            result = sanitize_cidfont_structures(pdf)

            assert result["cmap_encoding_fixed"] == 0, (
                f"Standard CMap /{name} should not be replaced"
            )

    def test_cmap_fix_updates_cidsysteminfo(self) -> None:
        """After CMap Name replacement, CIDSystemInfo is updated to match."""
        pdf = new_pdf()
        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType2",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
        )
        resolve(font)[Name.Encoding] = Name("/Adobe-Korea1-2")
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        assert result["cidsysteminfo_fixed"] == 1

        pdf = _roundtrip(pdf)
        cidfont = _get_cidfont(pdf)
        assert str(cidfont.CIDSystemInfo.Registry) == "Adobe"
        assert str(cidfont.CIDSystemInfo.Ordering) == "Identity"
        assert int(cidfont.CIDSystemInfo.Supplement) == 0


class TestCMapWModeFix:
    """Tests for CMap WMode mismatch fix (6.2.11.3.3 t02)."""

    def test_wmode_mismatch_dict_updated_to_match_stream(self) -> None:
        """Dict WMode=1 but stream WMode=0 → dict updated to 0."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(pdf, wmode=0)
        # Override dict WMode to create mismatch
        cmap[Name("/WMode")] = 1

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_wmode_fixed"] == 1
        # Verify dict WMode was corrected
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        assert int(enc["/WMode"]) == 0

    def test_wmode_mismatch_reverse(self) -> None:
        """Dict WMode=0 but stream WMode=1 → dict updated to 1."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(pdf, wmode=1)
        # Override dict WMode to create mismatch
        cmap[Name("/WMode")] = 0

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_wmode_fixed"] == 1
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        assert int(enc["/WMode"]) == 1

    def test_wmode_matching_not_modified(self) -> None:
        """Matching WMode values are not modified."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(pdf, wmode=0)

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_wmode_fixed"] == 0

    def test_wmode_missing_in_dict_added_from_stream(self) -> None:
        """No /WMode in dict but stream defines it → dict gets WMode."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(pdf, wmode=1)
        # Remove WMode from dict
        del cmap[Name("/WMode")]

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Korea1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_wmode_fixed"] == 1
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        assert int(enc["/WMode"]) == 1


class TestCMapUseCMap:
    """Tests for CMap /UseCMap fix (6.2.11.3.3 t03)."""

    def test_usecmap_stream_triggers_stripping(self) -> None:
        """/UseCMap pointing to embedded stream → /UseCMap stripped."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(
            pdf,
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            cmap_name="Adobe-Japan1-2",
        )
        # Add /UseCMap pointing to an embedded stream
        usecmap_stream = _make_cmap_stream(
            pdf,
            registry="Adobe",
            ordering="Japan1",
            supplement=4,
            cmap_name="Adobe-Japan1-4",
        )
        cmap[Name("/UseCMap")] = usecmap_stream

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        # Encoding is still a Stream (CMap preserved), but /UseCMap is gone
        assert isinstance(enc, pikepdf.Stream)
        assert enc.get("/UseCMap") is None

    def test_usecmap_nonstandard_name_triggers_stripping(self) -> None:
        """/UseCMap with non-standard Name → /UseCMap stripped."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(
            pdf,
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            cmap_name="Adobe-Japan1-2",
        )
        cmap[Name("/UseCMap")] = Name("/NonStandard-CMap")

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        # Encoding is still a Stream (CMap preserved), but /UseCMap is gone
        assert isinstance(enc, pikepdf.Stream)
        assert enc.get("/UseCMap") is None

    def test_usecmap_standard_name_preserved(self) -> None:
        """/UseCMap with standard predefined Name is preserved."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(
            pdf,
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            cmap_name="Adobe-Japan1-2",
        )
        cmap[Name("/UseCMap")] = Name("/H")

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 0

    def test_usecmap_vertical_stripping(self) -> None:
        """Vertical CMap with bad /UseCMap → /UseCMap stripped."""
        pdf = new_pdf()
        cmap = _make_cmap_stream(
            pdf,
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            cmap_name="Adobe-Japan1-2",
            wmode=1,
        )
        usecmap_stream = _make_cmap_stream(pdf, cmap_name="Bad-CMap")
        cmap[Name("/UseCMap")] = usecmap_stream

        font = _make_type0_font(
            pdf,
            encoding="Identity-H",
            cidfont_subtype="/CIDFontType0",
            registry="Adobe",
            ordering="Japan1",
            supplement=2,
            add_cidtogidmap=False,
        )
        resolve(font)[Name.Encoding] = cmap
        _build_pdf_with_type0_font(pdf, font)

        result = sanitize_cidfont_structures(pdf)

        assert result["cmap_encoding_fixed"] == 1
        pdf = _roundtrip(pdf)
        font_obj = resolve(pdf.pages[0].Resources.Font["/F1"])
        enc = resolve(font_obj.Encoding)
        # Encoding is still a Stream (CMap preserved), but /UseCMap is gone
        assert isinstance(enc, pikepdf.Stream)
        assert enc.get("/UseCMap") is None


class TestStandardCMapNames:
    """Tests for the _STANDARD_CMAP_NAMES set."""

    def test_identity_h_v_in_standard_set(self) -> None:
        assert "Identity-H" in _STANDARD_CMAP_NAMES
        assert "Identity-V" in _STANDARD_CMAP_NAMES

    def test_cjk_names_in_standard_set(self) -> None:
        """Representative CJK CMap names are in the standard set."""
        for name in [
            "H",
            "V",
            "90ms-RKSJ-H",
            "UniJIS-UTF16-H",
            "GB-EUC-H",
            "UniGB-UTF16-H",
            "B5pc-H",
            "UniCNS-UTF16-H",
            "KSC-EUC-H",
            "UniKS-UTF16-H",
        ]:
            assert name in _STANDARD_CMAP_NAMES, f"{name} missing"

    def test_nonstandard_names_not_in_set(self) -> None:
        """Non-standard CMap names are not in the standard set."""
        for name in ["Adobe-Korea1-2", "Adobe-Japan1-4", "Custom-CMap"]:
            assert name not in _STANDARD_CMAP_NAMES, f"{name} should not be in set"


class TestStripSubsetPrefix:
    """Tests for _strip_subset_prefix helper."""

    def test_strips_valid_prefix(self) -> None:
        assert _strip_subset_prefix("ABCDEF+Arial") == "Arial"

    def test_strips_random_prefix(self) -> None:
        assert _strip_subset_prefix("XYZABC+TimesNewRoman") == "TimesNewRoman"

    def test_no_prefix_unchanged(self) -> None:
        assert _strip_subset_prefix("Arial") == "Arial"

    def test_short_prefix_unchanged(self) -> None:
        assert _strip_subset_prefix("ABC+Arial") == "ABC+Arial"

    def test_lowercase_prefix_unchanged(self) -> None:
        assert _strip_subset_prefix("abcdef+Arial") == "abcdef+Arial"

    def test_mixed_case_prefix_unchanged(self) -> None:
        assert _strip_subset_prefix("AbCdEf+Arial") == "AbCdEf+Arial"


def _make_font_with_descriptor(
    pdf: Pdf,
    *,
    base_font: str = "Arial",
    font_name: str = "Arial",
    subtype: str = "/TrueType",
) -> Dictionary:
    """Create a font with a FontDescriptor for FontName tests."""
    font_descriptor = pdf.make_indirect(
        Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name("/" + font_name),
            Flags=4,
        )
    )
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name(subtype),
            BaseFont=Name("/" + base_font),
            FontDescriptor=font_descriptor,
        )
    )
    return font


def _add_font_to_page(pdf: Pdf, font: Dictionary) -> None:
    """Add a page with the given font to the PDF."""
    page = pikepdf.Page(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )
    )
    pdf.pages.append(page)


class TestFontNameConsistency:
    """Tests for FontDescriptor /FontName vs /BaseFont (ISO 19005-2 §6.3.5)."""

    def test_matching_names_no_fix(self) -> None:
        """Matching /FontName and /BaseFont require no fix."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(pdf, base_font="Arial", font_name="Arial")
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 0

    def test_matching_names_with_subset_prefix(self) -> None:
        """Matching names with subset prefixes require no fix."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(
            pdf, base_font="ABCDEF+Arial", font_name="ABCDEF+Arial"
        )
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 0

    def test_different_subset_prefixes_same_name(self) -> None:
        """Different subset prefixes but same core name require no fix."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(
            pdf, base_font="ABCDEF+Arial", font_name="XYZABC+Arial"
        )
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 0

    def test_mismatched_names_fixed(self) -> None:
        """Mismatched /FontName is corrected to match /BaseFont."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(pdf, base_font="Arial", font_name="Helvetica")
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 1

        fd = resolve(resolve(font).FontDescriptor)
        assert str(fd.FontName) == "/Arial"

    def test_mismatched_with_subset_prefix(self) -> None:
        """Mismatched name with subset prefix preserves prefix from BaseFont."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(
            pdf,
            base_font="ABCDEF+Arial",
            font_name="XYZABC+Helvetica",
        )
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 1

        fd = resolve(resolve(font).FontDescriptor)
        assert str(fd.FontName) == "/ABCDEF+Arial"

    def test_basefont_no_prefix_fontname_has_prefix(self) -> None:
        """BaseFont without prefix, FontName with prefix but wrong name."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(
            pdf,
            base_font="Arial",
            font_name="ABCDEF+Helvetica",
        )
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 1

        fd = resolve(resolve(font).FontDescriptor)
        # No prefix from BaseFont, so result should be just the name
        assert str(fd.FontName) == "/Arial"

    def test_type0_cidfont_fontname_fixed(self) -> None:
        """FontName mismatch in Type0's CIDFont descendant is fixed."""
        pdf = new_pdf()

        cidfont_descriptor = pdf.make_indirect(
            Dictionary(
                Type=Name.FontDescriptor,
                FontName=Name("/WrongName"),
                Flags=4,
            )
        )
        cidfont = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/CIDFontType2"),
                BaseFont=Name("/ABCDEF+CorrectName"),
                CIDSystemInfo=Dictionary(
                    Registry="Adobe",
                    Ordering="Identity",
                    Supplement=0,
                ),
                CIDToGIDMap=Name.Identity,
                FontDescriptor=cidfont_descriptor,
            )
        )
        type0_font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/Type0"),
                BaseFont=Name("/ABCDEF+CorrectName"),
                Encoding=Name("/Identity-H"),
                DescendantFonts=Array([cidfont]),
            )
        )
        _add_font_to_page(pdf, type0_font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 1

        fd = resolve(cidfont_descriptor)
        assert str(fd.FontName) == "/ABCDEF+CorrectName"

    def test_no_font_descriptor_skipped(self) -> None:
        """Font without FontDescriptor is silently skipped."""
        pdf = new_pdf()
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name("/TrueType"),
                BaseFont=Name("/Arial"),
            )
        )
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 0

    def test_shared_font_counted_once(self) -> None:
        """Same font on two pages is only fixed once."""
        pdf = new_pdf()
        font = _make_font_with_descriptor(pdf, base_font="Arial", font_name="Helvetica")
        _add_font_to_page(pdf, font)
        _add_font_to_page(pdf, font)

        result = sanitize_fontname_consistency(pdf)
        assert result["fontname_fixed"] == 1


def _make_type0_with_cidfont(pdf: Pdf, cidfont: Dictionary) -> Dictionary:
    """Wrap a CIDFont dict in a Type0 font and return the Type0 dict."""
    return Dictionary(
        Type=Name.Font,
        Subtype=Name.Type0,
        BaseFont=Name("/TestFont"),
        Encoding=Name("/Identity-H"),
        DescendantFonts=Array([pdf.make_indirect(cidfont)]),
    )


class TestCIDValuesOver65535:
    """Tests for CID value range validation (ISO 19005-2 rule 6.1.13-10)."""

    def test_w_format1_cid_over_65535_warns(self, caplog) -> None:
        """/W format-1 entry with CID > 65535 triggers a warning."""
        pdf = new_pdf()
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
            W=Array([70000, Array([600, 600])]),  # CID 70000 > 65535
        )
        font = _make_type0_with_cidfont(pdf, cidfont)
        _build_pdf_with_type0_font(pdf, font)

        with caplog.at_level(logging.WARNING):
            result = sanitize_cidfont_structures(pdf)

        assert result["cid_values_over_65535_warned"] == 1
        assert any("6.1.13-10" in r.message for r in caplog.records)

    def test_w_format2_range_over_65535_warns(self, caplog) -> None:
        """/W format-2 range where upper bound > 65535 triggers a warning."""
        pdf = new_pdf()
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
            W=Array([65500, 65540, 600]),  # range includes CIDs > 65535
        )
        font = _make_type0_with_cidfont(pdf, cidfont)
        _build_pdf_with_type0_font(pdf, font)

        with caplog.at_level(logging.WARNING):
            result = sanitize_cidfont_structures(pdf)

        assert result["cid_values_over_65535_warned"] == 1
        assert any("6.1.13-10" in r.message for r in caplog.records)

    def test_cidtogidmap_stream_over_131072_warns(self, caplog) -> None:
        """/CIDToGIDMap stream longer than 131072 bytes implies CIDs > 65535."""
        pdf = new_pdf()
        cidtogidmap_stream = pdf.make_stream(b"\x00" * 131074)
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
            CIDToGIDMap=cidtogidmap_stream,
        )
        font = _make_type0_with_cidfont(pdf, cidfont)
        _build_pdf_with_type0_font(pdf, font)

        with caplog.at_level(logging.WARNING):
            result = sanitize_cidfont_structures(pdf)

        assert result["cid_values_over_65535_warned"] == 1
        assert any("6.1.13-10" in r.message for r in caplog.records)

    def test_cid_at_limit_does_not_warn(self, caplog) -> None:
        """/W entry with CID exactly 65535 should not trigger a warning."""
        pdf = new_pdf()
        cidfont = Dictionary(
            Type=Name.Font,
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name("/TestFont"),
            CIDSystemInfo=Dictionary(
                Registry=pikepdf.String("Adobe"),
                Ordering=pikepdf.String("Identity"),
                Supplement=0,
            ),
            W=Array([65535, Array([600])]),  # exactly at limit — should be fine
        )
        font = _make_type0_with_cidfont(pdf, cidfont)
        _build_pdf_with_type0_font(pdf, font)

        with caplog.at_level(logging.WARNING):
            result = sanitize_cidfont_structures(pdf)

        assert result["cid_values_over_65535_warned"] == 0
        assert not any("6.1.13-10" in r.message for r in caplog.records)
