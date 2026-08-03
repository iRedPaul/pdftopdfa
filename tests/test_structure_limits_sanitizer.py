# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for structural implementation-limit sanitization."""

from decimal import Decimal
from io import BytesIO

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.exceptions import UnsupportedPDFError
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.structure_limits import (
    _operands_contain_parse_placeholders,
    _sanitize_operand,
    sanitize_structure_limits,
)

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

    @staticmethod
    def _patch_string_unparse(
        monkeypatch: pytest.MonkeyPatch, sentinel: bytes, token: bytes
    ) -> None:
        """Patch Object.unparse so one sentinel string emits a chosen raw token."""
        object_type = type(pikepdf.String(b""))
        original_unparse = object_type.unparse

        def _patched_unparse(obj) -> bytes:
            if isinstance(obj, pikepdf.String):
                try:
                    if bytes(obj) == sentinel:
                        return token
                except Exception:
                    pass
            return original_unparse(obj)

        monkeypatch.setattr(object_type, "unparse", _patched_unparse)

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

    def test_strips_invalid_chars_from_hex_string_in_text_operator(self) -> None:
        # <48G5>: 'G' is not a valid hex digit — strip it → <485> (odd) → <4850> = b"HP"
        pdf = _make_page_pdf(b"BT <48G5> Tj ET")

        result = sanitize_structure_limits(pdf)

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        text_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Tj"
        )
        assert bytes(text_op.operands[0]) == b"HP"
        assert result["hex_invalid_fixed"] == 1
        assert result["hex_odd_fixed"] == 1  # odd after stripping

    def test_strips_all_invalid_chars_leaves_empty_hex_string(self) -> None:
        # <GGG>: all chars invalid — stripped to <> (empty string = b"")
        pdf = _make_page_pdf(b"BT <GGG> Tj ET")

        result = sanitize_structure_limits(pdf)

        instructions = list(pikepdf.parse_content_stream(pdf.pages[0].Contents))
        text_op = next(
            i
            for i in instructions
            if not isinstance(i, pikepdf.ContentStreamInlineImage)
            and str(i.operator) == "Tj"
        )
        assert bytes(text_op.operands[0]) == b""
        assert result["hex_invalid_fixed"] == 1
        assert result["hex_odd_fixed"] == 0  # empty is even-length (0)

    def test_fixes_odd_hex_string_in_dictionary_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = new_pdf()
        sentinel = b"__ODD_HEX_OBJ_SENTINEL__"
        pdf.Root[Name("/HexValue")] = pikepdf.String(sentinel)
        self._patch_string_unparse(monkeypatch, sentinel, b"<ABC>")

        result = sanitize_structure_limits(pdf)

        assert bytes(pdf.Root["/HexValue"]) == b"\xab\xc0"
        assert result["hex_odd_obj_fixed"] == 1

    def test_keeps_even_hex_string_in_dictionary_value_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = new_pdf()
        sentinel = b"__EVEN_HEX_OBJ_SENTINEL__"
        pdf.Root[Name("/HexValue")] = pikepdf.String(sentinel)
        self._patch_string_unparse(monkeypatch, sentinel, b"<ABCD>")

        result = sanitize_structure_limits(pdf)

        assert bytes(pdf.Root["/HexValue"]) == sentinel
        assert result["hex_odd_obj_fixed"] == 0

    def test_keeps_literal_string_in_dictionary_value_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = new_pdf()
        sentinel = b"__LITERAL_OBJ_SENTINEL__"
        pdf.Root[Name("/HexValue")] = pikepdf.String(sentinel)
        self._patch_string_unparse(monkeypatch, sentinel, b"(ABC)")

        result = sanitize_structure_limits(pdf)

        assert bytes(pdf.Root["/HexValue"]) == sentinel
        assert result["hex_odd_obj_fixed"] == 0

    def test_fixes_odd_hex_string_with_embedded_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = new_pdf()
        sentinel = b"__WHITESPACE_HEX_OBJ_SENTINEL__"
        pdf.Root[Name("/HexValue")] = pikepdf.String(sentinel)
        self._patch_string_unparse(monkeypatch, sentinel, b"<41 4>")

        result = sanitize_structure_limits(pdf)

        assert bytes(pdf.Root["/HexValue"]) == b"A@"
        assert result["hex_odd_obj_fixed"] == 1

    def test_rebalances_q_q_nesting_to_28(self) -> None:
        q_count = 29
        content = (b"q " * q_count) + (b"Q " * q_count)
        pdf = _make_page_pdf(content)

        result = sanitize_structure_limits(pdf)

        assert _max_q_depth(pdf.pages[0].Contents) <= 28
        assert result["q_nesting_rebalanced"] == 2
        xobjects = pdf.pages[0].Resources.XObject
        assert len(xobjects) == 1
        form = next(iter(xobjects.values()))
        assert _max_q_depth(form) <= 28

    def test_q_q_form_bbox_accounts_for_current_transformation(self) -> None:
        """A repair Form does not clip content under an existing CTM."""
        content = (
            b"0.1 0 0 0.1 0 0 cm "
            + (b"q " * 29)
            + b"1000 1000 100 100 re f "
            + (b"Q " * 29)
        )
        pdf = _make_page_pdf(content)

        sanitize_structure_limits(pdf)

        xobjects = pdf.pages[0].Resources.XObject
        name, form = next(iter(xobjects.items()))
        assert list(form.BBox) == [0, 0, 2000, 2000]
        form_xobjects = form.Resources.get("/XObject")
        assert form_xobjects is None or form_xobjects.get(name) is None

    def test_q_q_across_page_content_streams_is_repaired_as_one_sequence(
        self,
    ) -> None:
        """Page Contents arrays share one graphics-state stack."""
        pdf = _make_page_pdf(b"")
        pdf.pages[0].obj[Name.Contents] = Array(
            [
                pdf.make_stream(b"q " * 29),
                pdf.make_stream(b"0 0 10 10 re f " + (b"Q " * 29)),
            ]
        )

        result = sanitize_structure_limits(pdf)

        assert isinstance(pdf.pages[0].Contents, pikepdf.Stream)
        assert _max_q_depth(pdf.pages[0].Contents) <= 28
        assert result["q_nesting_rebalanced"] == 2

    def test_rebalances_form_stream_with_inherited_resources(self) -> None:
        """A Form stream's own BBox is available for safe q/Q repair."""
        pdf = new_pdf()
        form = pdf.make_stream((b"q " * 30) + b"0 0 10 10 re f " + (b"Q " * 30))
        form[Name.Type] = Name.XObject
        form[Name.Subtype] = Name.Form
        form[Name.BBox] = Array([0, 0, 1000, 1000])
        resources = Dictionary(XObject=Dictionary(X0=form))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 500, 500]),
                Resources=resources,
                Contents=pdf.make_stream(b"/X0 Do"),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)

        assert _max_q_depth(form) <= 28
        assert result["q_nesting_rebalanced"] == 2
        assert form.get("/Resources") is None
        assert len(resources.XObject) == 2

    def test_repairs_1200_q_levels_without_python_recursion(self) -> None:
        """Deep graphics-state repair uses an iterative post-order walk."""
        pdf = _make_page_pdf((b"q " * 1200) + b"0 0 10 10 re f " + (b"Q " * 1200))

        result = sanitize_structure_limits(pdf)

        pending = [pdf.pages[0].Contents]
        seen: set[tuple[int, int]] = set()
        while pending:
            stream = pending.pop()
            assert _max_q_depth(stream) <= 28
            resources = stream.get("/Resources")
            if not isinstance(resources, Dictionary):
                resources = pdf.pages[0].Resources
            xobjects = resources.get("/XObject")
            if not isinstance(xobjects, Dictionary):
                continue
            for candidate in xobjects.values():
                if not isinstance(candidate, pikepdf.Stream):
                    continue
                if candidate.objgen in seen:
                    continue
                seen.add(candidate.objgen)
                pending.append(candidate)

        assert result["q_nesting_rebalanced"] > 0

    @pytest.mark.parametrize(
        ("prefix", "suffix"),
        [
            (b"BT ", b" ET"),
            (b"/Span BMC ", b" EMC"),
            (b"BX ", b" EX"),
        ],
    )
    def test_q_q_repair_rejects_crossing_operator_scopes(
        self,
        prefix: bytes,
        suffix: bytes,
    ) -> None:
        """Unsafe Form boundaries fail instead of silently changing semantics."""
        content = prefix + (b"q " * 29) + (b"Q " * 29) + suffix
        pdf = _make_page_pdf(content)

        with pytest.raises(UnsupportedPDFError, match="Cannot safely reduce q/Q"):
            sanitize_structure_limits(pdf)

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

    def test_remaps_used_cmap_cid_overflow_to_notdef(self) -> None:
        pdf = new_pdf()
        cmap_data = b"""
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincidchar
<3F00> 65791
endcidchar
endcmap
end
end
"""
        encoding_stream = pdf.make_stream(cmap_data)
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/TestCID"),
                Encoding=encoding_stream,
                DescendantFonts=Array([]),
            )
        )

        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=pdf.make_stream(b"BT /F1 12 Tf <3F00> Tj ET"),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)
        cmap_text = encoding_stream.read_bytes().decode("latin-1")

        assert "<3F00> 0" in cmap_text
        assert "65791" not in cmap_text
        assert result["cid_overflow_entries_repaired"] == 1

    def test_remaps_used_overflow_part_of_cid_range(self) -> None:
        pdf = new_pdf()
        encoding_stream = pdf.make_stream(
            b"""
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<3F00> <3FFF> 65536
endcidrange
"""
        )
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/TestCID"),
                Encoding=encoding_stream,
                DescendantFonts=Array([]),
            )
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=pdf.make_stream(b"BT /F1 12 Tf <3F00> Tj ET"),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)
        cmap_text = encoding_stream.read_bytes().decode("latin-1")

        assert "<3F00> 0" in cmap_text
        assert "65536" not in cmap_text
        assert result["cid_overflow_entries_repaired"] == 1

    def test_removes_unused_cmap_cid_overflow_range(self) -> None:
        pdf = new_pdf()
        cmap_data = b"""
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincidchar
<0001> 1
endcidchar
1 begincidrange
<3F00> <3FFF> 65536
endcidrange
endcmap
end
end
"""
        encoding_stream = pdf.make_stream(cmap_data)
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/TestCID"),
                Encoding=encoding_stream,
                DescendantFonts=Array([]),
            )
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

        result = sanitize_structure_limits(pdf)
        cmap_text = encoding_stream.read_bytes().decode("latin-1")

        assert "65536" not in cmap_text
        assert "<3F00> <3FFF> 65536" not in cmap_text
        assert result["cid_overflow_entries_repaired"] == 1

    def test_splits_mixed_overflow_range_and_preserves_block_count(self) -> None:
        pdf = new_pdf()
        encoding_stream = pdf.make_stream(
            b"""
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
% the comment is not a mapping entry

<0000> <0002> 65534
endcidrange
"""
        )
        font = pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name("/TestCID"),
                Encoding=encoding_stream,
                DescendantFonts=Array([]),
            )
        )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(Font=Dictionary(F1=font)),
                Contents=pdf.make_stream(b"BT /F1 12 Tf <00000002> Tj ET"),
            )
        )
        pdf.pages.append(page)

        result = sanitize_structure_limits(pdf)
        cmap_text = encoding_stream.read_bytes().decode("latin-1")

        assert "<0000> <0001> 65534" in cmap_text
        assert "<0002> 0" in cmap_text
        assert "% the comment is not a mapping entry" in cmap_text
        assert cmap_text.count("1 begincidrange") == 1
        assert cmap_text.count("1 begincidchar") == 1
        assert result["cid_overflow_entries_repaired"] == 1

    def test_pipeline_fixes_non_hex_text_stream(self) -> None:
        pdf = _make_page_pdf(b"BT <48G5> Tj ET")

        result = sanitize_for_pdfa(pdf, "2b")

        assert result["structure_hex_invalid_fixed"] == 1

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

    def test_sanitizes_deep_form_object_graph_without_recursion_error(self) -> None:
        pdf = new_pdf()
        form = pdf.make_stream(b"q Q")
        form["/Type"] = Name.XObject
        form["/Subtype"] = Name.Form
        form["/BBox"] = Array([0, 0, 10, 10])
        form["/Resources"] = Dictionary()
        form["/DeepValue"] = _INT_MAX + 1
        for _ in range(1200):
            parent = pdf.make_stream(b"/Fm Do")
            parent["/Type"] = Name.XObject
            parent["/Subtype"] = Name.Form
            parent["/BBox"] = Array([0, 0, 10, 10])
            parent["/Resources"] = Dictionary(XObject=Dictionary(Fm=form))
            form = parent
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 200, 200]),
                Resources=Dictionary(XObject=Dictionary(Fm=form)),
                Contents=pdf.make_stream(b"/Fm Do"),
            )
        )
        pdf.pages.append(page)
        buffer = BytesIO()
        pdf.save(buffer)
        buffer.seek(0)

        with Pdf.open(buffer) as reopened:
            result = sanitize_structure_limits(reopened)
            deep_form = next(
                item
                for item in reopened.objects
                if isinstance(item, pikepdf.Stream) and "/DeepValue" in item
            )

            assert int(deep_form["/DeepValue"]) == _INT_MAX
            assert result["integers_clamped"] == 1

    def test_sanitizes_deep_array_operand_without_recursion_error(self) -> None:
        operand: object = _INT_MAX + 1
        for _ in range(1200):
            operand = Array([operand])
        stats = {
            "names_shortened": 0,
            "utf8_names_fixed": 0,
            "strings_truncated": 0,
            "integers_clamped": 0,
            "reals_normalized": 0,
        }

        sanitized, changed = _sanitize_operand(operand, stats)

        current = sanitized
        for _ in range(1200):
            assert isinstance(current, Array)
            current = current[0]
        assert int(current) == _INT_MAX
        assert changed is True
        assert stats["integers_clamped"] == 1

    def test_detects_placeholder_in_1200_deep_array_operand(self) -> None:
        """Invalid parsed hex placeholders are detected without recursion."""
        operand: object = None
        for _ in range(1200):
            operand = Array([operand])

        assert _operands_contain_parse_placeholders(operand) is True
