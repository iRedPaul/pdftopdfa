# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for structural implementation-limit sanitization."""

from decimal import Decimal

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.exceptions import UnsupportedPDFError
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.structure_limits import sanitize_structure_limits

_INT_MAX = 2_147_483_647


def _make_page_pdf(content: bytes, resources: Dictionary | None = None) -> Pdf:
    """Create a single-page PDF with custom content and optional resources."""
    pdf = new_pdf()
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 200, 200]),
        Contents=pdf.make_stream(content),
    )
    if resources is not None:
        page_dict[Name.Resources] = resources
    pdf.pages.append(pikepdf.Page(page_dict))
    return pdf


def _max_q_depth(stream_obj: pikepdf.Stream) -> int:
    """Compute max q/Q nesting depth for a content stream."""
    depth = 0
    max_depth = 0
    for instruction in pikepdf.parse_content_stream(stream_obj):
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        op = str(instruction.operator)
        if op == "q":
            depth += 1
            max_depth = max(max_depth, depth)
        elif op == "Q":
            depth = max(0, depth - 1)
    return max_depth


class TestStructureLimitsSanitizer:
    """Tests for structure limit repairs and unsupported detection."""

    def test_fixes_odd_hex_string_in_text_operator(self) -> None:
        pdf = _make_page_pdf(b"BT <48455> Tj ET")

        result = sanitize_structure_limits(pdf)

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        text_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Tj"
        )
        assert bytes(text_op.operands[0]) == b"HEP"
        assert result["hex_odd_fixed"] == 1

    def test_raises_for_non_hexadecimal_string_in_text_operator(self) -> None:
        pdf = _make_page_pdf(b"BT <48G5> Tj ET")

        with pytest.raises(UnsupportedPDFError, match="Malformed hexadecimal"):
            sanitize_structure_limits(pdf)

    def test_rebalances_q_q_nesting_to_28(self) -> None:
        q_count = 29
        content = (b"q " * q_count) + (b"Q " * q_count)
        pdf = _make_page_pdf(content)

        result = sanitize_structure_limits(pdf)

        assert _max_q_depth(pdf.pages[0].Contents) <= 28
        assert result["q_nesting_rebalanced"] == 2

    def test_shortens_long_name_keys_values_and_operands(self) -> None:
        pdf = new_pdf()
        long_name = "X" * 130

        form = pdf.make_stream(b"q Q")
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 10, 10])
        form[Name.Name] = Name("/" + long_name)

        xobjects = Dictionary()
        xobjects[Name("/" + long_name)] = form
        resources = Dictionary(XObject=xobjects)
        content = f"q /{long_name} Do Q".encode("ascii")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=resources,
                Contents=pdf.make_stream(content),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)

        keys = list(pdf.pages[0].Resources.XObject.keys())
        assert all(len(k.encode("utf-8", "surrogateescape")) - 1 <= 127 for k in keys)

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        do_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Do"
        )
        do_name = str(do_op.operands[0]).lstrip("/")
        assert len(do_name.encode("utf-8", "surrogateescape")) <= 127

        assert len(str(form.get("/Name")).lstrip("/").encode("utf-8")) <= 127
        assert result["names_shortened"] >= 2

    def test_truncates_overlong_strings_in_objects_and_content(self) -> None:
        long_bytes = b"A" * 40_000
        content = b"BT <" + (b"41" * 33_000) + b"> Tj ET"
        pdf = _make_page_pdf(content)
        pdf.Root[Name("/LongString")] = pikepdf.String(long_bytes)

        result = sanitize_structure_limits(pdf)

        root_string = pdf.Root["/LongString"]
        assert isinstance(root_string, pikepdf.String)
        assert len(bytes(root_string)) == 32_767

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        text_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Tj"
        )
        assert len(bytes(text_op.operands[0])) == 32_767
        assert result["strings_truncated"] >= 2

    def test_fixes_utf8_name_and_numeric_limits(self) -> None:
        content = b"2157483648 0 Td 0.00000000000000000000000000000000000001173 g"
        pdf = _make_page_pdf(content)
        pdf.Root["/Cyan" + "\udcc2"] = Name("/ValidName")
        pdf.Root[Name("/BigInt")] = 2_157_483_648
        pdf.Root[Name("/TinyReal")] = Decimal("1.173E-38")

        result = sanitize_structure_limits(pdf)

        assert all("\udcc2" not in key for key in pdf.Root.keys())
        assert int(pdf.Root["/BigInt"]) == _INT_MAX
        assert Decimal(pdf.Root["/TinyReal"]) == Decimal("0")

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        td_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Td"
        )
        g_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "g"
        )
        assert int(td_op.operands[0]) == _INT_MAX
        assert Decimal(g_op.operands[0]) == Decimal("0")

        assert result["utf8_names_fixed"] >= 1
        assert result["integers_clamped"] >= 1
        assert result["reals_normalized"] >= 1

    def test_clamps_overflow_real_values(self) -> None:
        # 3.404e+38 written as full decimal (PDF content streams don't support
        # scientific notation); minus sign prefix for the negative case.
        # 3.404e+38 = 340400000000000000000000000000000000000.0
        _overflow_pos = b"340400000000000000000000000000000000000.0"
        _overflow_neg = b"-340400000000000000000000000000000000000.0"
        # 3.403e+38 = exactly at the boundary (must not be changed)
        _at_limit = b"340300000000000000000000000000000000000.0"
        content = _overflow_pos + b" g " + _overflow_neg + b" w " + _at_limit + b" J"
        pdf = _make_page_pdf(content)
        pdf.Root[Name("/PosOverflow")] = Decimal("3.404e+38")
        pdf.Root[Name("/NegOverflow")] = Decimal("-3.404e+38")
        pdf.Root[Name("/AtLimit")] = Decimal("3.403e+38")

        result = sanitize_structure_limits(pdf)

        # float() comparison because pikepdf converts Decimal to float64 internally;
        # the read-back Decimal is the full float64 expansion, not the short form.
        _max_float = float(Decimal("3.403e+38"))

        # Object graph: overflow values clamped to ±3.403e+38
        assert float(Decimal(pdf.Root["/PosOverflow"])) == _max_float
        assert float(Decimal(pdf.Root["/NegOverflow"])) == -_max_float
        # Exactly at limit: must be left unchanged
        assert float(Decimal(pdf.Root["/AtLimit"])) == _max_float

        # Content stream operands are also detected and counted.
        # Note: we do not re-parse the stream after sanitization because
        # float64 values near 3.403e+38 are exact integers and pikepdf
        # serializes them without a decimal point, causing a 64-bit
        # integer overflow on re-parse. The counter is the reliable signal.
        # At least 4 fixes: 2 object-graph + 2 content-stream operands
        assert result["reals_normalized"] >= 4

    def test_rejects_cmap_cid_overflow(self) -> None:
        pdf = new_pdf()
        cmap_data = b"""
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincidchar
<0001> 65791
endcidchar
endcmap
end
end
"""
        encoding_stream = pdf.make_stream(cmap_data)
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name("/TestCID"),
            Encoding=encoding_stream,
            DescendantFonts=Array([]),
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=pdf.make_stream(b"BT /F1 12 Tf <0001> Tj ET"),
            )
        )
        pdf.pages.append(page)

        with pytest.raises(UnsupportedPDFError, match="CID values greater than 65535"):
            sanitize_structure_limits(pdf)

    def test_pipeline_raises_for_non_hex_text_stream(self) -> None:
        pdf = _make_page_pdf(b"BT <48G5> Tj ET")

        with pytest.raises(UnsupportedPDFError, match="Malformed hexadecimal"):
            sanitize_for_pdfa(pdf, "2b")

    def test_tolerates_cross_stream_boundary_text_operator(self) -> None:
        """A Contents array may split a TJ instruction across streams."""
        pdf = new_pdf()
        # First stream ends with the TJ operand array, second starts with TJ
        stream1 = pdf.make_stream(b"BT [(Hello)] ")
        stream2 = pdf.make_stream(b"TJ ET")
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Contents=Array([stream1, stream2]),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)

        assert result["hex_odd_fixed"] == 0

    def test_skips_unreadable_content_streams(self) -> None:
        pdf = new_pdf()
        content_stream = pdf.make_stream(b"q Q")
        content_stream["/Filter"] = Name("/Flatedecode")

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Contents=content_stream,
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)

        assert result["hex_odd_fixed"] == 0
