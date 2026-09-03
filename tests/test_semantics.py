# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import pytest

import pdftopdfa.semantics as semantics
from pdftopdfa.semantics import (
    ArtifactKind,
    BoundingBox,
    SemanticPage,
    SemanticSpan,
    SpanKind,
    StructureAttribute,
    StructureNode,
    build_semantic_plan,
)


def _span(
    span_id: str,
    text: str,
    left: float,
    top: float,
    *,
    width: float = 180,
    height: float = 10,
    font_size: float = 10,
    font_name: str = "Body",
    kind: SpanKind = SpanKind.TEXT,
    confidence: float | None = None,
    invisible: bool = False,
) -> SemanticSpan:
    return SemanticSpan(
        span_id,
        text,
        BoundingBox(left, top, left + width, top + height),
        font_size,
        font_name,
        kind,
        confidence,
        invisible,
    )


def _page(
    *spans: SemanticSpan,
    number: int = 1,
    reading_order_hint: tuple[str, ...] | None = None,
    column_gutters: tuple[float, ...] | None = None,
) -> SemanticPage:
    return SemanticPage(
        number,
        600,
        800,
        spans,
        reading_order_hint,
        column_gutters,
    )


def _roles(node: StructureNode) -> list[str]:
    return [descendant.role for descendant in node.walk()]


def _physical_references(node: StructureNode) -> list[str]:
    return [
        reference.span_id
        for descendant in node.walk()
        for reference in descendant.content
    ]


def _role_for_span(node: StructureNode, span_id: str) -> str:
    matches = [
        descendant.role for descendant in node.walk() if span_id in descendant.span_ids
    ]
    assert len(matches) == 1
    return matches[0]


def test_two_column_reading_order_is_segmented_by_full_width_heading() -> None:
    page = _page(
        _span("right-2", "right second.", 330, 94),
        _span("left-2", "left second.", 50, 94),
        _span(
            "heading",
            "Quarterly report",
            50,
            20,
            width=500,
            height=20,
            font_size=20,
            font_name="Body Bold",
        ),
        _span("right-1", "Right first line", 330, 80),
        _span("left-1", "Left first line", 50, 80),
    )

    plan = build_semantic_plan([page])

    assert plan.pages[0].reading_order == (
        "heading",
        "left-1",
        "left-2",
        "right-1",
        "right-2",
    )
    assert [child.role for child in plan.pages[0].structure.children] == [
        "H1",
        "P",
        "P",
    ]
    assert plan.pages[0].structure.children[1].span_ids == ("left-1", "left-2")
    assert plan.pages[0].structure.children[2].span_ids == (
        "right-1",
        "right-2",
    )


def test_full_width_line_restarts_column_flow_mid_page() -> None:
    page = _page(
        _span("left-a", "Left above.", 50, 80, width=180),
        _span("right-a", "Right above.", 330, 80, width=180),
        _span("divider", "Shared divider", 50, 150, width=500),
        _span("left-b", "Left below.", 50, 190, width=180),
        _span("right-b", "Right below.", 330, 190, width=180),
    )

    plan = build_semantic_plan([page])

    assert plan.pages[0].reading_order == (
        "left-a",
        "right-a",
        "divider",
        "left-b",
        "right-b",
    )


def test_nearly_equal_baselines_are_read_left_to_right() -> None:
    page = _page(
        _span("right", "Amount", 430, 100.35, width=70, height=9.65),
        _span("left", "Invoice date", 50, 100.55, width=80, height=9.45),
        _span("middle", "28.05.2015", 230, 100, width=70, height=10),
    )

    plan = build_semantic_plan([page])

    assert plan.pages[0].reading_order == ("left", "middle", "right")
    assert _physical_references(plan.pages[0].structure) == [
        "left",
        "middle",
        "right",
    ]


def test_visual_rows_do_not_chain_through_staggered_members() -> None:
    page = _page(
        _span("bottom-left", "Third", 50, 110, width=60),
        _span("middle", "Second", 230, 105, width=60),
        _span("top-right", "First", 430, 100, width=60),
    )

    plan = build_semantic_plan([page])

    assert plan.pages[0].reading_order == (
        "middle",
        "top-right",
        "bottom-left",
    )


def test_three_columns_are_read_top_to_bottom_then_left_to_right() -> None:
    spans = tuple(
        _span(
            f"c{column}-r{row}",
            f"Column {column} row {row}",
            left,
            80 + row * 14,
            width=120,
        )
        for column, left in enumerate((40, 230, 420))
        for row in range(2)
    )

    plan = build_semantic_plan([_page(*reversed(spans))])

    assert plan.pages[0].reading_order == tuple(span.id for span in spans)


def test_reading_order_hint_overrides_two_column_visual_order() -> None:
    spans = (
        _span("left-1", "Left first line", 50, 80),
        _span("left-2", "Left second line.", 50, 94),
        _span("right-1", "Right first line", 330, 80),
        _span("right-2", "Right second line.", 330, 94),
    )
    hint = ("right-1", "right-2", "left-1", "left-2")

    plan = build_semantic_plan([_page(*reversed(spans), reading_order_hint=hint)])

    assert plan.pages[0].reading_order == hint
    assert _physical_references(plan.pages[0].structure) == list(hint)


def test_reading_order_hint_skips_artifacts_without_losing_physical_order() -> None:
    spans = (
        _span("scan", "", 0, 0, width=600, height=800, kind=SpanKind.IMAGE),
        _span("second", "Second OCR line.", 50, 120, invisible=True),
        _span("first", "First OCR line.", 50, 100, invisible=True),
    )
    hint = ("scan", "first", "second")

    plan = build_semantic_plan([_page(*spans, reading_order_hint=hint)])

    assert [artifact.span_id for artifact in plan.artifacts] == ["scan"]
    assert plan.pages[0].reading_order == ("first", "second")
    assert _physical_references(plan.pages[0].structure) == ["first", "second"]


def test_reading_order_hint_deduplicates_table_and_list_blocks() -> None:
    table_spans = tuple(
        span
        for row, values in enumerate((("Name", "Amount"), ("Alpha", "10")))
        for span in (
            _span(
                f"r{row}c0",
                values[0],
                50,
                100 + row * 18,
                width=80,
                font_name="Body Bold" if row == 0 else "Body",
            ),
            _span(
                f"r{row}c1",
                values[1],
                220,
                100 + row * 18,
                width=80,
                font_name="Body Bold" if row == 0 else "Body",
            ),
        )
    )
    list_spans = (
        _span("item-1", "\u2022 First item", 50, 200, width=250),
        _span("item-2", "\u2022 Second item", 50, 218, width=250),
    )
    hint = ("item-1", "item-2", *(span.id for span in table_spans))

    plan = build_semantic_plan(
        [_page(*table_spans, *list_spans, reading_order_hint=hint)]
    )
    references = _physical_references(plan.pages[0].structure)

    assert [child.role for child in plan.pages[0].structure.children] == ["L", "Table"]
    assert references == list(hint)
    assert len(references) == len(set(references))


@pytest.mark.parametrize(
    "hint",
    [
        ("first",),
        ("first", "first"),
        ("first", "unknown"),
    ],
)
def test_reading_order_hint_requires_every_span_exactly_once(
    hint: tuple[str, ...],
) -> None:
    spans = (
        _span("first", "First", 50, 100),
        _span("second", "Second", 50, 120),
    )

    with pytest.raises(ValueError, match="every page span ID exactly once"):
        _page(*spans, reading_order_hint=hint)


def test_reading_order_hint_requires_a_tuple_of_span_ids() -> None:
    span = _span("first", "First", 50, 100)

    with pytest.raises(TypeError, match="tuple of non-empty span IDs"):
        SemanticPage(1, 600, 800, (span,), ["first"])  # type: ignore[arg-type]


def test_heading_levels_use_document_statistics_and_never_jump() -> None:
    page = _page(
        _span(
            "h1",
            "Document title",
            50,
            20,
            width=400,
            height=24,
            font_size=24,
            font_name="Sans Bold",
        ),
        _span(
            "body-1",
            "A long body line that establishes the document font size.",
            50,
            70,
            width=480,
        ),
        _span(
            "h-small",
            "Detailed subject",
            50,
            120,
            width=260,
            height=14,
            font_size=14,
            font_name="Sans Bold",
        ),
        _span(
            "body-2",
            "Another substantial body line for stable statistics.",
            50,
            150,
            width=480,
        ),
        _span(
            "h-medium",
            "Broader subject",
            50,
            200,
            width=300,
            height=18,
            font_size=18,
            font_name="Sans Bold",
        ),
        _span(
            "body-3",
            "The final body line uses the same ten point body style.",
            50,
            230,
            width=480,
        ),
    )

    plan = build_semantic_plan([page])
    headings = [
        node.role
        for node in plan.pages[0].structure.children
        if node.role.startswith("H")
    ]

    assert headings == ["H1", "H2", "H2"]


def test_bold_body_sized_key_value_row_is_not_a_heading() -> None:
    page = _page(
        _span(
            "section",
            "Payment details",
            50,
            40,
            width=180,
            font_name="Body Bold",
        ),
        _span(
            "body-1",
            "This regular sentence establishes the main document text style.",
            50,
            80,
            width=450,
        ),
        _span(
            "body-2",
            "Another ordinary sentence keeps the body-size statistic stable.",
            50,
            100,
            width=450,
        ),
        _span(
            "key",
            "Invoice date",
            50,
            150.5,
            width=100,
            height=9.5,
            font_name="Body Bold",
        ),
        _span(
            "date",
            "28.05.2015",
            230,
            150,
            width=70,
            font_name="Body Bold",
        ),
        _span(
            "amount",
            "4.098,95 €",
            430,
            150.25,
            width=70,
            height=9.75,
            font_name="Body Bold",
        ),
    )

    plan = build_semantic_plan([page])

    assert _role_for_span(plan.root, "section").startswith("H")
    assert all(
        not _role_for_span(plan.root, span_id).startswith("H")
        for span_id in ("key", "date", "amount")
    )


def test_parallel_column_section_titles_remain_headings() -> None:
    page = _page(
        _span(
            "left-heading",
            "Products",
            50,
            80,
            width=160,
            font_name="Body Bold",
        ),
        _span(
            "right-heading",
            "Services",
            330,
            80,
            width=160,
            font_name="Body Bold",
        ),
        _span("left-1", "First product description.", 50, 150, width=180),
        _span("left-2", "Second product description.", 50, 170, width=180),
        _span("right-1", "First service description.", 330, 150, width=180),
        _span("right-2", "Second service description.", 330, 170, width=180),
    )

    plan = build_semantic_plan([page])

    assert _role_for_span(plan.root, "left-heading") == "H1"
    assert _role_for_span(plan.root, "right-heading") == "H1"


def test_columnized_key_value_pair_remains_paragraph_content() -> None:
    page = _page(
        _span(
            "key",
            "Invoice date",
            50,
            80,
            width=160,
            font_name="Body Bold",
        ),
        _span(
            "value",
            "28.05.2015",
            330,
            80,
            width=160,
            font_name="Body Bold",
        ),
        _span("left-1", "First account note.", 50, 150, width=180),
        _span("left-2", "Second account note.", 50, 170, width=180),
        _span("right-1", "First delivery note.", 330, 150, width=180),
        _span("right-2", "Second delivery note.", 330, 170, width=180),
    )

    plan = build_semantic_plan([page])

    assert _role_for_span(plan.root, "key") == "P"
    assert _role_for_span(plan.root, "value") == "P"


def test_soft_lines_form_paragraph_and_vertical_gap_starts_another() -> None:
    page = _page(
        _span("line-1", "A sentence that", 50, 100, width=300),
        _span("line-2", "continues on the next visual line.", 50, 113, width=350),
        _span("line-3", "A new paragraph.", 50, 150, width=250),
    )

    plan = build_semantic_plan([page])
    paragraphs = plan.pages[0].structure.children

    assert [node.role for node in paragraphs] == ["P", "P"]
    assert paragraphs[0].span_ids == ("line-1", "line-2")
    assert paragraphs[1].span_ids == ("line-3",)


@pytest.mark.parametrize(
    ("left_text", "right_text"),
    [
        ("A long word con-", "tinues at the top of the next page."),
        ("A dependent clause,", "which continues at the top of the next page."),
    ],
    ids=["hyphenated-word", "lowercase-clause"],
)
def test_clear_paragraph_continuation_crosses_page_break(
    left_text: str,
    right_text: str,
) -> None:
    pages = (
        _page(
            _span("page-1", left_text, 50, 735, width=300),
            number=1,
        ),
        _page(
            _span("page-2", right_text, 50, 70, width=350),
            number=2,
        ),
    )

    plan = build_semantic_plan(pages)
    paragraph = plan.pages[0].structure.children[0]

    assert paragraph.role == "P"
    assert paragraph.span_ids == ("page-1", "page-2")
    assert paragraph.bbox is None
    assert paragraph.page_number is None
    assert [page.structure.page_number for page in plan.pages] == [1, 2]
    assert plan.pages[1].structure.children == ()
    assert _physical_references(plan.root) == ["page-1", "page-2"]


def test_paragraphs_well_inside_page_break_bands_remain_separate() -> None:
    pages = (
        _page(
            _span("page-1", "Authorized signature", 50, 580, width=300),
            number=1,
        ),
        _page(
            _span("page-2", "date of birth", 50, 210, width=350),
            number=2,
        ),
    )

    plan = build_semantic_plan(pages)

    assert plan.pages[0].structure.children[0].span_ids == ("page-1",)
    assert plan.pages[1].structure.children[0].span_ids == ("page-2",)


def test_complete_paragraphs_at_page_edges_remain_separate() -> None:
    pages = (
        _page(
            _span("page-1", "This paragraph is complete.", 50, 735, width=300),
            number=1,
        ),
        _page(
            _span("page-2", "Another paragraph starts here.", 50, 70, width=350),
            number=2,
        ),
    )

    plan = build_semantic_plan(pages)

    assert plan.pages[0].structure.children[0].span_ids == ("page-1",)
    assert plan.pages[1].structure.children[0].span_ids == ("page-2",)


def test_lists_have_pdf_roles_and_combined_spans_are_referenced_once() -> None:
    page = _page(
        _span("item-1", "\u2022 First item", 50, 100, width=250),
        _span("continuation", "continued text", 65, 113, width=220),
        _span("item-2", "\u2022 Second item", 50, 128, width=250),
    )

    plan = build_semantic_plan([page])
    list_node = plan.pages[0].structure.children[0]

    assert [child.role for child in list_node.children] == ["LI", "LI"]
    assert [[child.role for child in item.children] for item in list_node.children] == [
        ["Lbl", "LBody"],
        ["Lbl", "LBody"],
    ]
    assert list_node.attributes[0].owner == "List"
    assert list_node.attributes[0].value == "Disc"
    first_label, first_body = list_node.children[0].children
    assert first_label.content == ()
    assert first_label.actual_text == "\u2022"
    assert first_body.span_ids == ("item-1", "continuation")
    assert first_body.actual_text == "First item continued text"
    references = _physical_references(plan.root)
    assert references == ["item-1", "continuation", "item-2"]
    assert len(references) == len(set(references))


def test_bullet_list_and_item_body_continue_across_page_breaks() -> None:
    pages = (
        _page(
            _span("item-1", "• First item", 50, 700, width=250),
            _span("item-2", "• Second item con-", 50, 720, width=250),
            number=1,
        ),
        _page(
            _span("item-2-body", "tinues on page two", 65, 70, width=230),
            _span("item-3", "• Third item", 50, 90, width=250),
            _span("item-4", "• Fourth item", 50, 110, width=250),
            number=2,
        ),
    )

    plan = build_semantic_plan(pages)
    list_node = plan.pages[0].structure.children[0]

    assert list_node.role == "L"
    assert list_node.page_number is None
    assert len(list_node.children) == 4
    assert [item.page_number for item in list_node.children] == [1, None, 2, 2]
    continued_label, continued_body = list_node.children[1].children
    assert continued_label.page_number == 1
    assert continued_body.page_number is None
    assert list_node.children[1].children[1].span_ids == (
        "item-2",
        "item-2-body",
    )
    assert plan.pages[1].structure.children == ()
    assert _physical_references(plan.root) == [
        "item-1",
        "item-2",
        "item-2-body",
        "item-3",
        "item-4",
    ]


@pytest.mark.parametrize(
    ("right_labels", "merged"),
    [(("3.", "4."), True), (("1.", "2."), False)],
    ids=["sequential", "restart"],
)
def test_numbered_lists_cross_pages_only_when_sequential(
    right_labels: tuple[str, str],
    merged: bool,
) -> None:
    pages = (
        _page(
            _span("first-1", "1. First item", 50, 700, width=250),
            _span("first-2", "2. Second item", 50, 720, width=250),
            number=1,
        ),
        _page(
            _span(
                "second-1",
                f"{right_labels[0]} First item on page two",
                50,
                70,
                width=250,
            ),
            _span(
                "second-2",
                f"{right_labels[1]} Second item on page two",
                50,
                90,
                width=250,
            ),
            number=2,
        ),
    )

    plan = build_semantic_plan(pages)

    first_list = plan.pages[0].structure.children[0]
    assert first_list.role == "L"
    if merged:
        assert first_list.page_number is None
        assert len(first_list.children) == 4
        assert [item.page_number for item in first_list.children] == [1, 1, 2, 2]
        assert plan.pages[1].structure.children == ()
    else:
        second_list = plan.pages[1].structure.children[0]
        assert second_list.role == "L"
        assert first_list.page_number == 1
        assert second_list.page_number == 2
        assert len(first_list.children) == 2
        assert len(second_list.children) == 2


def test_separate_list_label_span_owns_its_physical_reference() -> None:
    page = _page(
        _span("label-1", "(1)", 50, 100, width=12),
        _span("body-1", "First", 90, 100, width=120),
        _span("label-2", "(2)", 50, 118, width=12),
        _span("body-2", "Second", 90, 118, width=120),
    )

    plan = build_semantic_plan([page])
    list_node = plan.pages[0].structure.children[0]

    assert list_node.role == "L"
    assert list_node.children[0].children[0].span_ids == ("label-1",)
    assert list_node.children[0].children[1].span_ids == ("body-1",)
    assert list_node.children[0].children[0].actual_text is None
    assert _physical_references(plan.root) == [
        "label-1",
        "body-1",
        "label-2",
        "body-2",
    ]


def test_single_hyphenated_line_uses_paragraph_fallback() -> None:
    plan = build_semantic_plan(
        [_page(_span("dash", "- an isolated dash clause", 50, 100, width=250))]
    )

    assert [child.role for child in plan.pages[0].structure.children] == ["P"]


def test_strong_aligned_grid_becomes_table_with_column_headers() -> None:
    spans = []
    values = (("Name", "Amount"), ("Alpha", "10"), ("Beta", "20"))
    for row, values_for_row in enumerate(values):
        font_name = "Body Bold" if row == 0 else "Body"
        spans.extend(
            (
                _span(
                    f"r{row}c0",
                    values_for_row[0],
                    50,
                    100 + row * 18,
                    width=80,
                    font_name=font_name,
                ),
                _span(
                    f"r{row}c1",
                    values_for_row[1],
                    220,
                    100 + row * 18,
                    width=80,
                    font_name=font_name,
                ),
            )
        )

    plan = build_semantic_plan([_page(*spans)])
    table = plan.pages[0].structure.children[0]

    assert _roles(table) == [
        "Table",
        "TR",
        "TH",
        "TH",
        "TR",
        "TD",
        "TD",
        "TR",
        "TD",
        "TD",
    ]
    headers = table.children[0].children
    assert all(header.attributes[0].owner == "Table" for header in headers)
    assert all(header.attributes[0].name == "Scope" for header in headers)
    assert all(header.attributes[0].value == "Column" for header in headers)
    assert plan.pages[0].reading_order == tuple(span.id for span in spans)


def test_sparse_multiline_table_preserves_logical_cells_and_order() -> None:
    spans = (
        _span("h-item", "Item", 50, 100, width=45, font_name="Body Bold"),
        _span(
            "h-description",
            "Description",
            130,
            100,
            width=90,
            font_name="Body Bold",
        ),
        _span("h-qty", "Qty", 300, 100, width=35, font_name="Body Bold"),
        _span(
            "h-unit",
            "Unit price",
            380,
            100,
            width=55,
            font_name="Body Bold",
        ),
        _span("h-total", "Total", 480, 100, width=55, font_name="Body Bold"),
        _span(
            "h-total-cont",
            "gross",
            480,
            112,
            width=40,
            font_name="Body Bold",
        ),
        _span("r1-item", "A-100", 50, 130, width=45),
        _span("r1-description", "Ergonomic desk", 130, 130, width=120),
        _span("r1-qty", "2", 300, 130, width=10),
        _span("r1-unit", "100,00 €", 380, 130, width=55),
        _span("r1-total", "200,00 €", 480, 130, width=60),
        _span(
            "r1-description-cont-1",
            "with cable tray",
            130,
            142,
            width=110,
        ),
        _span(
            "r1-description-cont-2",
            "and rounded edge",
            130,
            154,
            width=120,
        ),
        _span("r2-item", "B-200", 50, 176, width=45),
        _span("r2-description", "Visitor chair", 130, 176, width=100),
        _span("r2-qty", "3", 300, 176, width=10),
        _span("r2-unit", "50,00 €", 380, 176, width=55),
        _span("r2-total", "150,00 €", 480, 176, width=60),
        _span(
            "r2-description-cont",
            "with armrests",
            130,
            188,
            width=100,
        ),
        _span("summary-label", "Invoice total", 380, 212, width=75),
        _span("summary-amount", "350,00 €", 480, 212, width=60),
    )
    expected_order = (
        "h-item",
        "h-description",
        "h-qty",
        "h-unit",
        "h-total",
        "h-total-cont",
        "r1-item",
        "r1-description",
        "r1-description-cont-1",
        "r1-description-cont-2",
        "r1-qty",
        "r1-unit",
        "r1-total",
        "r2-item",
        "r2-description",
        "r2-description-cont",
        "r2-qty",
        "r2-unit",
        "r2-total",
        "summary-label",
        "summary-amount",
    )

    plan = build_semantic_plan([_page(*reversed(spans))])
    tables = [node for node in plan.root.walk() if node.role == "Table"]

    assert len(tables) == 1
    table = tables[0]
    assert [cell.role for cell in table.children[0].children] == ["TH"] * 5
    assert all(cell.role == "TD" for row in table.children[1:] for cell in row.children)
    assert table.children[0].children[-1].span_ids == (
        "h-total",
        "h-total-cont",
    )
    assert table.children[1].children[1].span_ids == (
        "r1-description",
        "r1-description-cont-1",
        "r1-description-cont-2",
    )
    assert table.children[2].children[1].span_ids == (
        "r2-description",
        "r2-description-cont",
    )
    assert tuple(_physical_references(table)) == expected_order
    assert plan.pages[0].reading_order == expected_order


def test_sparse_header_does_not_absorb_incomplete_first_data_row() -> None:
    spans = (
        _span("h-item", "Item", 50, 100, width=60, font_name="Body Bold"),
        _span(
            "h-description",
            "Description",
            220,
            100,
            width=90,
            font_name="Body Bold",
        ),
        _span("h-total", "Total", 430, 100, width=60, font_name="Body Bold"),
        _span("partial-item", "A-100", 50, 112, width=60),
        _span("partial-total", "10,00 €", 430, 112, width=60),
        _span("item", "B-200", 50, 130, width=60),
        _span("description", "Desk chair", 220, 130, width=90),
        _span("total", "20,00 €", 430, 130, width=60),
    )

    plan = build_semantic_plan([_page(*spans)])
    table = next(node for node in plan.root.walk() if node.role == "Table")

    assert len(table.children) == 3
    assert _physical_references(table.children[0]) == [
        "h-item",
        "h-description",
        "h-total",
    ]
    assert _physical_references(table.children[1]) == [
        "partial-item",
        "partial-total",
    ]


def test_sparse_header_does_not_absorb_nonbold_data_without_record_anchor() -> None:
    spans = (
        _span("h-item", "Item", 50, 100, width=60, font_name="Body Bold"),
        _span(
            "h-description",
            "Description",
            220,
            100,
            width=90,
            font_name="Body Bold",
        ),
        _span("h-total", "Total", 430, 100, width=60, font_name="Body Bold"),
        _span("partial-description", "Replacement part", 220, 112, width=90),
        _span("partial-total", "10,00 €", 430, 112, width=60),
        _span("item", "B-200", 50, 130, width=60),
        _span("description", "Desk chair", 220, 130, width=90),
        _span("total", "20,00 €", 430, 130, width=60),
    )

    plan = build_semantic_plan([_page(*spans)])
    table = next(node for node in plan.root.walk() if node.role == "Table")

    assert len(table.children) == 3
    assert _physical_references(table.children[0]) == [
        "h-item",
        "h-description",
        "h-total",
    ]
    assert _physical_references(table.children[1]) == [
        "partial-description",
        "partial-total",
    ]


def test_large_table_scan_does_not_rescan_growing_row_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 240
    spans = tuple(
        _span(
            f"r{row}c{column}",
            text,
            left,
            50 + row * 2.5,
            width=40,
            height=1.5,
            font_name="Body Bold" if row == 0 else "Body",
        )
        for row in range(row_count)
        for column, (left, text) in enumerate(
            (
                (50, "Item" if row == 0 else f"A-{row}"),
                (250, "Quantity" if row == 0 else str(row)),
                (450, "Amount" if row == 0 else f"{row},00 €"),
            )
        )
    )
    page = _page(*spans)
    original = semantics._compatible_table_rows
    calls = 0
    inspected_rows = 0

    def tracked_compatibility(rows: tuple[object, ...], page_width: float) -> bool:
        nonlocal calls, inspected_rows
        calls += 1
        inspected_rows += len(rows)
        return original(rows, page_width)

    monkeypatch.setattr(semantics, "_compatible_table_rows", tracked_compatibility)

    candidates = semantics._table_candidates(page, list(spans))

    assert candidates
    assert calls <= 2
    assert inspected_rows <= row_count * 2


def test_table_candidate_selection_uses_one_cumulative_ownership_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_count = 80
    spans = [
        _span(
            f"t{table}-r{row}c{column}",
            text,
            left,
            20 + table * 50 + row * 12,
            width=60,
            font_name="Body Bold" if row == 0 else "Body",
        )
        for table in range(table_count)
        for row, values in enumerate((("Name", "Amount"), (f"Item {table}", "1")))
        for column, (left, text) in enumerate(zip((50, 220), values, strict=True))
    ]
    original = semantics._table_candidate

    class CountingFrozenSet(frozenset):
        calls = 0

        def isdisjoint(self, other: object) -> bool:
            type(self).calls += 1
            return super().isdisjoint(other)

    def tracked_candidate(
        rows: tuple[semantics._TableRow, ...], page_number: int
    ) -> semantics._TableCandidate:
        candidate = original(rows, page_number)
        return semantics._TableCandidate(
            candidate.rows,
            candidate.bbox,
            CountingFrozenSet(candidate.span_ids),
            candidate.node,
        )

    monkeypatch.setattr(semantics, "_table_candidate", tracked_candidate)

    candidates = semantics._table_candidates(_page(*spans), spans)

    assert len(candidates) == table_count
    assert CountingFrozenSet.calls <= table_count * 2


def test_bold_values_before_table_do_not_become_its_header() -> None:
    spans = (
        _span("field-a", "Delivery date:", 50, 100, width=75),
        _span(
            "value-a",
            "See carrier receipt",
            140,
            100,
            width=135,
            font_name="Body Bold",
        ),
        _span("field-b", "Customer:", 300, 100, width=65),
        _span(
            "value-b",
            "1421554",
            400,
            100,
            width=50,
            font_name="Body Bold",
        ),
        _span("header-item", "Item", 50, 144, width=50, font_name="Body Bold"),
        _span(
            "header-amount",
            "Amount",
            400,
            144,
            width=60,
            font_name="Body Bold",
        ),
        _span("item", "A-100", 50, 164, width=50),
        _span("amount", "10,00 €", 400, 164, width=60),
    )

    plan = build_semantic_plan([_page(*spans)])
    table = next(node for node in plan.root.walk() if node.role == "Table")

    assert _physical_references(table.children[0]) == [
        "header-item",
        "header-amount",
    ]
    assert all(
        span_id not in _physical_references(table)
        for span_id in ("field-a", "value-a", "field-b", "value-b")
    )


@pytest.mark.parametrize(
    ("right_header", "merged"),
    [(("Name", "Amount"), True), (("Description", "Amount"), False)],
    ids=["repeated-header", "different-header"],
)
def test_tables_cross_page_break_only_with_matching_repeated_header(
    right_header: tuple[str, str],
    merged: bool,
) -> None:
    def table_spans(
        prefix: str,
        header: tuple[str, str],
        top: int,
        rows: int,
    ) -> tuple[SemanticSpan, ...]:
        values = (header, *((f"Item {row}", str(row)) for row in range(1, rows)))
        return tuple(
            _span(
                f"{prefix}-r{row}c{column}",
                text,
                left,
                top + row * 18,
                width=80,
                font_name="Body Bold" if row == 0 else "Body",
            )
            for row, row_values in enumerate(values)
            for column, (left, text) in enumerate(
                zip((50, 220), row_values, strict=True)
            )
        )

    first_page_spans = table_spans("p1", ("Name", "Amount"), 610, 8)
    second_page_spans = table_spans("p2", right_header, 70, 3)

    plan = build_semantic_plan(
        [
            _page(*first_page_spans, number=1),
            _page(*second_page_spans, number=2),
        ]
    )

    if merged:
        table = plan.pages[0].structure.children[0]
        assert table.role == "Table"
        assert table.bbox is None
        assert table.page_number is None
        assert len(table.children) == 11
        assert [row.page_number for row in table.children] == [1] * 8 + [2] * 3
        assert all(
            cell.page_number == row.page_number
            for row in table.children
            for cell in row.children
        )
        assert (
            sum(cell.role == "TH" for row in table.children for cell in row.children)
            == 4
        )
        assert plan.pages[1].structure.children == ()
    else:
        first_table = plan.pages[0].structure.children[0]
        second_table = plan.pages[1].structure.children[0]
        assert first_table.role == "Table"
        assert second_table.role == "Table"
        assert first_table.page_number == 1
        assert second_table.page_number == 2
    assert _physical_references(plan.root) == [
        *(span.id for span in first_page_spans),
        *(span.id for span in second_page_spans),
    ]


def test_multipage_table_merge_does_not_depend_on_header_figure_artifacting() -> None:
    def table_spans(prefix: str, top: int, rows: int) -> tuple[SemanticSpan, ...]:
        values = (
            ("Name", "Amount"),
            *((f"Item {row}", str(row)) for row in range(1, rows)),
        )
        return tuple(
            _span(
                f"{prefix}-r{row}c{column}",
                text,
                left,
                top + row * 18,
                width=80,
                font_name="Body Bold" if row == 0 else "Body",
            )
            for row, row_values in enumerate(values)
            for column, (left, text) in enumerate(
                zip((50, 220), row_values, strict=True)
            )
        )

    first_page_spans = table_spans("p1", 610, 8)
    second_page_spans = table_spans("p2", 130, 32)
    third_page_spans = table_spans("p3", 130, 3)
    logos = tuple(
        _span(
            f"page-{number}-logo",
            "",
            50,
            20,
            width=100,
            height=100,
            kind=SpanKind.IMAGE,
        )
        for number in (2, 3)
    )

    plan = build_semantic_plan(
        [
            _page(*first_page_spans, number=1),
            _page(logos[0], *second_page_spans, number=2),
            _page(logos[1], *third_page_spans, number=3),
        ]
    )
    table = plan.pages[0].structure.children[0]

    assert table.role == "Table"
    assert table.page_number is None
    assert len(table.children) == 43
    assert [row.page_number for row in table.children] == [1] * 8 + [2] * 32 + [3] * 3
    assert [
        [child.role for child in page.structure.children] for page in plan.pages[1:]
    ] == [["Figure"], ["Figure"]]
    artifact_ids = {artifact.span_id for artifact in plan.artifacts}
    assert all(logo.id not in artifact_ids for logo in logos)


@pytest.mark.parametrize(
    "right_tops",
    [(100, 120, 140), (109, 129, 149)],
    ids=["aligned", "staggered"],
)
def test_side_by_side_tables_are_detected_independently(
    right_tops: tuple[int, int, int],
) -> None:
    spans = [
        _span(
            "heading",
            "Account overview",
            50,
            40,
            width=500,
            height=20,
            font_size=20,
            font_name="Body Bold",
        )
    ]
    values = (("Name", "Amount"), ("Alpha", "10"), ("Beta", "20"))
    for prefix, lefts, tops in (
        ("left", (50, 150), (100, 120, 140)),
        ("right", (330, 450), right_tops),
    ):
        for row, (top, row_values) in enumerate(zip(tops, values, strict=True)):
            spans.extend(
                _span(
                    f"{prefix}-r{row}c{column}",
                    text,
                    lefts[column],
                    top,
                    width=50,
                    font_name="Body Bold" if row == 0 else "Body",
                )
                for column, text in enumerate(row_values)
            )

    plan = build_semantic_plan([_page(*spans)])
    children = plan.pages[0].structure.children

    assert [child.role for child in children] == ["H1", "Table", "Table"]
    left_references = tuple(_physical_references(children[1]))
    right_references = tuple(_physical_references(children[2]))
    assert left_references == tuple(
        f"left-r{row}c{column}" for row in range(3) for column in range(2)
    )
    assert right_references == tuple(
        f"right-r{row}c{column}" for row in range(3) for column in range(2)
    )
    assert plan.pages[0].reading_order == (
        "heading",
        *left_references,
        *right_references,
    )
    references = _physical_references(plan.root)
    assert len(references) == len(set(references)) == len(spans)


def test_explicit_ocr_column_gutter_recovers_one_sided_table() -> None:
    spans = []
    values = (("Name", "Amount"), ("Alpha", "10"), ("Beta", "20"))
    for row, row_values in enumerate(values):
        spans.extend(
            (
                _span(
                    f"table-r{row}c0",
                    row_values[0],
                    50,
                    100 + row * 18,
                    width=50,
                    font_name="Body Bold" if row == 0 else "Body",
                ),
                _span(
                    f"table-r{row}c1",
                    row_values[1],
                    150,
                    100 + row * 18,
                    width=50,
                    font_name="Body Bold" if row == 0 else "Body",
                ),
                _span(
                    f"note-r{row}",
                    f"Note {row}",
                    340,
                    100 + row * 18,
                    width=50,
                ),
            )
        )
    spans.append(_span("extra-note", "extra", 450, 118, width=50))
    hint = tuple(span.id for span in spans)

    plan = build_semantic_plan(
        [
            _page(
                *spans,
                reading_order_hint=hint,
                column_gutters=(300,),
            )
        ]
    )

    tables = [
        child for child in plan.pages[0].structure.children if child.role == "Table"
    ]
    assert len(tables) == 1
    assert _physical_references(tables[0]) == [
        f"table-r{row}c{column}" for row in range(3) for column in range(2)
    ]
    references = _physical_references(plan.root)
    assert len(references) == len(set(references)) == len(spans)


def test_dominant_gap_does_not_split_one_table_with_distinct_columns() -> None:
    values = (
        ("Region", "Owner", "Quarter", "Amount"),
        ("North", "Ada", "Q1", "10"),
        ("South", "Lin", "Q1", "20"),
    )
    spans = tuple(
        _span(
            f"r{row}c{column}",
            text,
            left,
            100 + row * 20,
            width=50,
            font_name="Body Bold" if row == 0 else "Body",
        )
        for row, row_values in enumerate(values)
        for column, (left, text) in enumerate(
            zip((50, 130, 330, 410), row_values, strict=True)
        )
    )

    plan = build_semantic_plan([_page(*spans)])
    tables = [
        child for child in plan.pages[0].structure.children if child.role == "Table"
    ]

    assert len(tables) == 1
    assert _physical_references(tables[0]) == [span.id for span in spans]


def test_ambiguous_aligned_prose_falls_back_to_paragraphs() -> None:
    spans = []
    for row in range(3):
        spans.extend(
            (
                _span(
                    f"left-{row}",
                    f"Left prose line {row}.",
                    50,
                    100 + row * 16,
                    width=180,
                ),
                _span(
                    f"right-{row}",
                    f"Right prose line {row}.",
                    330,
                    100 + row * 16,
                    width=180,
                ),
            )
        )

    plan = build_semantic_plan([_page(*spans)])

    assert "Table" not in _roles(plan.root)
    assert all(child.role == "P" for child in plan.pages[0].structure.children)


def test_running_headers_footers_and_page_numbers_are_artifacts() -> None:
    pages = [
        _page(
            _span(f"header-{number}", "Annual report", 40, 20, width=160),
            _span(f"body-{number}", f"Body page {number}", 50, 100, width=300),
            _span(f"footer-{number}", "Internal use", 40, 775, width=120),
            _span(f"number-{number}", str(number), 295, 775, width=10),
            number=number,
        )
        for number in range(1, 4)
    ]

    plan = build_semantic_plan(reversed(pages))
    artifacts = {artifact.span_id: artifact for artifact in plan.artifacts}

    for number in range(1, 4):
        assert artifacts[f"header-{number}"].kind is ArtifactKind.HEADER
        assert artifacts[f"footer-{number}"].kind is ArtifactKind.FOOTER
        assert artifacts[f"number-{number}"].kind is ArtifactKind.PAGE_NUMBER
        assert plan.pages[number - 1].reading_order == (f"body-{number}",)
    assert plan.root.children == tuple(page.structure for page in plan.pages)


def test_repeated_deep_footer_and_margin_graphic_are_artifacts() -> None:
    footer_lefts = (40, 54, 47)
    graphic_lefts = (516, 530, 523)
    pages = [
        _page(
            _span(f"body-{number}", f"Body page {number}", 50, 100, width=300),
            _span(
                f"footer-{number}",
                "DIN EN ISO 9001 · DIN EN ISO 14001",
                footer_lefts[number - 1],
                714 + number * 0.5,
                width=220,
            ),
            _span(
                f"graphic-{number}",
                "",
                graphic_lefts[number - 1],
                712 + number,
                width=30,
                height=18,
                kind=SpanKind.IMAGE,
            ),
            number=number,
        )
        for number in range(1, 4)
    ]

    plan = build_semantic_plan(pages)
    artifacts = {artifact.span_id: artifact for artifact in plan.artifacts}

    for number in range(1, 4):
        assert artifacts[f"footer-{number}"].kind is ArtifactKind.FOOTER
        assert artifacts[f"graphic-{number}"].kind is ArtifactKind.FOOTER
        assert plan.pages[number - 1].reading_order == (f"body-{number}",)


def test_repeated_large_margin_crossing_images_remain_figures() -> None:
    pages = [
        _page(
            _span(
                f"figure-{number}",
                "",
                50,
                80,
                width=500,
                height=320,
                kind=SpanKind.IMAGE,
            ),
            _span(f"body-{number}", f"Body page {number}.", 50, 500, width=300),
            number=number,
        )
        for number in range(1, 4)
    ]

    plan = build_semantic_plan(pages)
    artifact_ids = {artifact.span_id for artifact in plan.artifacts}

    assert all(f"figure-{number}" not in artifact_ids for number in range(1, 4))
    assert [node.role for node in plan.root.walk() if node.role == "Figure"] == [
        "Figure"
    ] * 3


def test_figure_is_declared_as_block_content() -> None:
    plan = build_semantic_plan(
        [
            _page(
                _span(
                    "image",
                    "",
                    50,
                    100,
                    width=100,
                    height=80,
                    kind=SpanKind.IMAGE,
                )
            )
        ]
    )

    figure = next(node for node in plan.root.walk() if node.role == "Figure")

    assert figure.attributes == (StructureAttribute("Layout", "Placement", "Block"),)


def test_repeated_clusters_enforce_full_geometry_diameter() -> None:
    positions = (
        (40.0, 40.0),
        (75.4, 57.4),
        (92.8, 66.4),
    )
    pages = tuple(
        _page(
            _span(
                f"header-{number}",
                "Running title",
                text_left,
                20,
                width=20,
            ),
            _span(
                f"graphic-{number}",
                "",
                graphic_left,
                70,
                width=30,
                height=18,
                kind=SpanKind.IMAGE,
            ),
            number=number,
        )
        for number, (text_left, graphic_left) in enumerate(positions, 1)
    )

    artifacts = semantics._detect_artifacts(pages)

    assert set(artifacts) == {
        "header-1",
        "header-2",
        "graphic-1",
        "graphic-2",
    }


def test_interleaved_nontext_geometries_match_their_own_clusters() -> None:
    pages = (
        _page(
            _span(
                "target-1",
                "",
                50,
                20,
                width=30,
                height=18,
                kind=SpanKind.IMAGE,
            ),
            _span(
                "distractor",
                "",
                57,
                70,
                width=30,
                height=18,
                kind=SpanKind.IMAGE,
            ),
            number=1,
        ),
        _page(
            _span(
                "target-2",
                "",
                64,
                20,
                width=30,
                height=18,
                kind=SpanKind.IMAGE,
            ),
            number=2,
        ),
    )

    artifacts = semantics._detect_artifacts(pages)

    assert set(artifacts) == {"target-1", "target-2"}


def test_repeated_text_clustering_does_not_recompute_growing_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_count = 180
    pages = tuple(
        _page(
            _span(
                f"header-{number}",
                "Annual report",
                40 + number % 3 * 7,
                20,
                width=160,
            ),
            number=number,
        )
        for number in range(1, page_count + 1)
    )
    original = semantics.statistics.median
    largest_sample = 0

    def tracked_median(values: object) -> float:
        nonlocal largest_sample
        sample = tuple(values)
        largest_sample = max(largest_sample, len(sample))
        return original(sample)

    monkeypatch.setattr(semantics.statistics, "median", tracked_median)

    artifacts = semantics._detect_artifacts(pages)

    assert len(artifacts) == page_count
    assert largest_sample <= 2


def test_page_label_prefix_with_document_number_is_an_artifact() -> None:
    pages = [
        _page(
            _span(f"body-{number}", f"Body page {number}", 50, 100, width=300),
            _span(
                f"page-label-{number}",
                f"Seite {number} von 3 · Dokument 244057509",
                350,
                715,
                width=200,
            ),
            number=number,
        )
        for number in range(1, 4)
    ]

    plan = build_semantic_plan(pages)
    artifacts = {artifact.span_id: artifact for artifact in plan.artifacts}

    for number in range(1, 4):
        assert artifacts[f"page-label-{number}"].kind is ArtifactKind.PAGE_NUMBER
        assert plan.pages[number - 1].reading_order == (f"body-{number}",)


def test_scan_background_is_artifact_but_invisible_ocr_text_is_structured() -> None:
    page = _page(
        _span("scan", "", 0, 0, width=600, height=800, kind=SpanKind.IMAGE),
        _span("ocr", "Recognized text", 50, 100, width=300, invisible=True),
        _span("figure", "", 50, 200, width=100, height=80, kind=SpanKind.IMAGE),
        _span("form", "", 50, 320, width=120, height=50, kind=SpanKind.FORM),
    )

    plan = build_semantic_plan([page])

    assert [(artifact.span_id, artifact.kind) for artifact in plan.artifacts] == [
        ("scan", ArtifactKind.BACKGROUND)
    ]
    assert plan.pages[0].reading_order == ("ocr", "figure", "form")
    assert [child.role for child in plan.pages[0].structure.children] == [
        "P",
        "Figure",
        "Form",
    ]
    assert plan.pages[0].structure.children[0].span_ids == ("ocr",)


def test_full_page_image_with_visible_text_is_not_assumed_to_be_background() -> None:
    page = _page(
        _span("image", "", 0, 0, width=600, height=800, kind=SpanKind.IMAGE),
        _span("visible", "Visible overlay", 50, 100, width=300),
    )

    plan = build_semantic_plan([page])

    assert plan.artifacts == ()
    assert "Figure" in _roles(plan.root)


def test_low_confidence_large_text_uses_safe_paragraph_fallback() -> None:
    page = _page(
        _span(
            "uncertain",
            "Possibly a heading",
            50,
            50,
            width=300,
            height=24,
            font_size=24,
            font_name="Bold",
            confidence=0.2,
        ),
        _span("body", "Reliable body content.", 50, 100, width=300),
    )

    plan = build_semantic_plan([page])

    assert [child.role for child in plan.pages[0].structure.children] == ["P", "P"]


def test_ocr_line_height_uses_conservative_heading_evidence() -> None:
    page = _page(
        _span(
            "title",
            "Invoice 5027425",
            50,
            40,
            width=260,
            height=18,
            font_size=18,
            invisible=True,
        ),
        _span(
            "body-1",
            "A regular body line establishes the OCR text size.",
            50,
            100,
            width=400,
            font_size=10,
            invisible=True,
        ),
        _span(
            "product",
            "Supermicro Mainboard X12STL-IF",
            50,
            130,
            width=300,
            height=15,
            font_size=15,
            invisible=True,
        ),
        _span(
            "logo-fragment",
            "Years",
            500,
            20,
            width=100,
            height=24,
            font_size=24,
            invisible=True,
        ),
        _span(
            "body-2",
            "Another regular body line keeps the statistic stable.",
            50,
            160,
            width=400,
            font_size=10,
            invisible=True,
        ),
    )

    plan = build_semantic_plan([page])
    roles_by_span = {
        span_id: node.role
        for node in plan.pages[0].structure.children
        for span_id in node.span_ids
    }

    assert roles_by_span["title"] == "H1"
    assert roles_by_span["product"] == "P"
    assert roles_by_span["logo-fragment"] == "P"


def test_plan_is_deterministic_and_rejects_duplicate_ids() -> None:
    spans = (
        _span("b", "Second", 50, 120),
        _span("a", "First", 50, 100),
    )
    first = build_semantic_plan([_page(*spans)])
    second = build_semantic_plan([_page(*reversed(spans))])

    assert first == second
    with pytest.raises(ValueError, match="Span IDs must be unique"):
        build_semantic_plan(
            [
                _page(_span("same", "One", 50, 100)),
                _page(_span("same", "Two", 50, 100), number=2),
            ]
        )


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ((0, 0, 0, 1), "positive width"),
        ((0, 0, float("nan"), 1), "finite"),
    ],
)
def test_bounding_box_validation(
    bbox: tuple[float, float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BoundingBox(*bbox)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("page 7", (7, True)),
        ("Seite iv von 9", (4, True)),
        ("3 of 9", (3, True)),
        ("- 3 -", (3, True)),
        ("12", (12, False)),
        ("iv", (4, False)),
        ("not a page", None),
        # _ROMAN matches sequences that no Roman numeral system can value.
        # Those must fall through rather than yield a None page number.
        ("iiiiiiiv", None),
        ("page iiiiiiiv", None),
        ("iiiiiiiv / 9", None),
        ("- iiiiiiiv -", None),
        ("Seite iiiiiiiv von 3", None),
    ],
)
def test_page_number_never_returns_an_unvalued_number(
    text: str,
    expected: tuple[int, bool] | None,
) -> None:
    result = semantics._page_number(text)

    assert result == expected
    assert result is None or isinstance(result[0], int)
