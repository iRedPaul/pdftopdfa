# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pikepdf
import pytest

from pdftopdfa.semantics import (
    ArtifactKind,
    SemanticPage,
    SemanticPlan,
    SemanticSpan,
    StructureNode,
    build_semantic_plan,
)
from pdftopdfa.tagging import _digital_semantic_inputs

_TEST_DOCS = Path(__file__).resolve().parents[1] / "test_docs"


@dataclass(frozen=True, slots=True)
class _RealDocumentPlan:
    pages: tuple[SemanticPage, ...]
    plan: SemanticPlan


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _build_plan(filename: str) -> _RealDocumentPlan:
    path = _TEST_DOCS / filename
    if not path.is_file():
        pytest.skip(f"Real-document regression fixture is missing: {path}")
    with pikepdf.Pdf.open(path) as pdf:
        spans_by_page, _, dimensions, *_ = _digital_semantic_inputs(pdf, {})
        pages = tuple(
            SemanticPage(
                page_index + 1,
                *dimensions[page_index],
                spans_by_page[page_index],
            )
            for page_index in range(len(pdf.pages))
        )
    return _RealDocumentPlan(pages, build_semantic_plan(pages))


@pytest.fixture(scope="module")
def shirtinator() -> _RealDocumentPlan:
    return _build_plan("180 - R3 - Rechnung.pdf")


@pytest.fixture(scope="module")
def hammerbacher() -> _RealDocumentPlan:
    return _build_plan("244057509.2024-12-13T060041.808.pdf")


@pytest.fixture(scope="module")
def delivery_note() -> _RealDocumentPlan:
    return _build_plan("010 - LS1 - Lieferschein.pdf")


@pytest.fixture(scope="module")
def outgoing_delivery_notes() -> _RealDocumentPlan:
    return _build_plan("120-150 - ALS1-4 - Ausgangslieferscheine.pdf")


def _matching_spans(
    document: _RealDocumentPlan,
    text: str,
    *,
    page_number: int | None = None,
) -> list[tuple[int, SemanticSpan]]:
    normalized = _normalized(text)
    matches = [
        (page.number, span)
        for page in document.pages
        if page_number is None or page.number == page_number
        for span in page.spans
        if _normalized(span.text) == normalized
    ]
    assert matches, f"No span found for {text!r} on page {page_number or 'any'}"
    return matches


def _span(
    document: _RealDocumentPlan,
    text: str,
    *,
    page_number: int,
) -> SemanticSpan:
    matches = _matching_spans(document, text, page_number=page_number)
    assert len(matches) == 1, (
        f"Expected one span for {text!r} on page {page_number}, got {len(matches)}"
    )
    return matches[0][1]


def _roles_and_tables_by_span(
    root: StructureNode,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[StructureNode, ...]]]:
    roles: dict[str, tuple[str, ...]] = {}
    tables: dict[str, tuple[StructureNode, ...]] = {}

    def walk(
        node: StructureNode,
        lineage: tuple[str, ...],
        table_lineage: tuple[StructureNode, ...],
    ) -> None:
        lineage = (*lineage, node.role)
        if node.role == "Table":
            table_lineage = (*table_lineage, node)
        for reference in node.content:
            roles[reference.span_id] = lineage
            tables[reference.span_id] = table_lineage
        for child in node.children:
            walk(child, lineage, table_lineage)

    walk(root, (), ())
    return roles, tables


def _reading_positions(
    document: _RealDocumentPlan,
    page_number: int,
    spans: list[SemanticSpan],
) -> list[int]:
    page_plan = next(
        page for page in document.plan.pages if page.page_number == page_number
    )
    positions = {
        span_id: index for index, span_id in enumerate(page_plan.reading_order)
    }
    return [positions[span.id] for span in spans]


def _assert_not_heading(
    document: _RealDocumentPlan,
    texts: tuple[str, ...],
) -> None:
    roles, _ = _roles_and_tables_by_span(document.plan.root)
    for text in texts:
        for page_number, span in _matching_spans(document, text):
            lineage = roles[span.id]
            assert not any(role.startswith("H") for role in lineage), (
                f"{text!r} on page {page_number} is incorrectly structured as "
                f"a heading: {lineage}"
            )


def test_delivery_note_keeps_visible_overprinted_cmyk_text(
    delivery_note: _RealDocumentPlan,
) -> None:
    span = _span(delivery_note, "Sehr geehrter Kunde!", page_number=3)

    assert span.kind.value == "text"


def test_outgoing_delivery_note_keeps_parallelogram_clipped_image(
    outgoing_delivery_notes: _RealDocumentPlan,
) -> None:
    page = outgoing_delivery_notes.pages[2]

    assert any(span.kind.value == "image" for span in page.spans)


def test_shirtinator_field_and_product_reading_order(
    shirtinator: _RealDocumentPlan,
) -> None:
    order_label = _span(shirtinator, "Auftrag vom:", page_number=1)
    invoice_label = _span(shirtinator, "Rechnungsnummer:", page_number=1)
    order_date = min(
        (span for _, span in _matching_spans(shirtinator, "28.05.2015", page_number=1)),
        key=lambda span: abs(span.bbox.center_y - order_label.bbox.center_y),
    )
    invoice_number = _span(shirtinator, "15/05/28-DE0154", page_number=1)
    field_positions = _reading_positions(
        shirtinator,
        1,
        [order_label, order_date, invoice_label, invoice_number],
    )
    assert field_positions == sorted(field_positions)

    product_text = (
        "Classic Fit T-",
        "Shirt Männer",
        "Weiß",
        "XL",
        "Vorne:",
        'M#1: "User", Direktdruck',
        "5",
        "18,90 €",
        "22,49 €",
        "112,45 €",
        "95,58 €",
    )
    product_positions = _reading_positions(
        shirtinator,
        1,
        [_span(shirtinator, text, page_number=1) for text in product_text],
    )
    assert product_positions == sorted(product_positions)


def test_shirtinator_product_and_total_are_under_tables(
    shirtinator: _RealDocumentPlan,
) -> None:
    _, tables = _roles_and_tables_by_span(shirtinator.plan.root)
    product = _span(shirtinator, "Classic Fit T-", page_number=1)
    gross_total = _span(shirtinator, "Rechnungsbetrag brutto", page_number=1)

    product_tables = tables[product.id]
    total_tables = tables[gross_total.id]
    assert product_tables, "The invoice product is not structured under Table"
    assert total_tables, "The gross total is not structured under Table"


def test_shirtinator_dates_and_amounts_are_not_headings(
    shirtinator: _RealDocumentPlan,
) -> None:
    _assert_not_heading(
        shirtinator,
        (
            "28.05.2015",
            "18,90 €",
            "22,49 €",
            "112,45 €",
            "95,58 €",
            "94,60 €",
            "17,97 €",
            "112,57 €",
        ),
    )


def test_hammerbacher_product_table_continues_across_pages(
    hammerbacher: _RealDocumentPlan,
) -> None:
    _, tables = _roles_and_tables_by_span(hammerbacher.plan.root)
    product_spans = (
        _span(hammerbacher, "VXDLR16/5/S", page_number=1),
        _span(hammerbacher, "VHS16/5/S", page_number=1),
        _span(hammerbacher, "VAC30/5/5/BM", page_number=2),
        _span(hammerbacher, "V4550/5/5/BM", page_number=2),
    )
    table_sets = [{id(table) for table in tables[span.id]} for span in product_spans]
    assert all(table_set for table_set in table_sets), (
        "At least one product is not structured under Table"
    )
    assert set.intersection(*table_sets), (
        "The repeated product table was not merged across both pages"
    )


def test_hammerbacher_items_precede_amounts_and_amounts_are_not_headings(
    hammerbacher: _RealDocumentPlan,
) -> None:
    pairs = (
        (1, "VXDLR16/5/S", "644,50 € -15 % 2.191,30 €"),
        (2, "VAC30/5/5/BM", "248,00 € -15 % 843,20 €"),
    )
    for page_number, item_text, amount_text in pairs:
        item = _span(hammerbacher, item_text, page_number=page_number)
        amount = _span(hammerbacher, amount_text, page_number=page_number)
        assert _reading_positions(hammerbacher, page_number, [item, amount]) == sorted(
            _reading_positions(hammerbacher, page_number, [item, amount])
        )

    _assert_not_heading(
        hammerbacher,
        (
            "644,50 € -15 % 2.191,30 €",
            "248,00 € -15 % 843,20 €",
            "4.098,95 €",
            "778,80 €",
            "4.877,75 €",
        ),
    )


def test_hammerbacher_repeated_footers_are_artifacts(
    hammerbacher: _RealDocumentPlan,
) -> None:
    artifacts = {artifact.span_id: artifact for artifact in hammerbacher.plan.artifacts}
    for text in (
        "DIN EN ISO 9001 • DIN EN ISO 14001 • WEE-REG.-Nr. DE 34853279",
        "HypoVereinsbank",
    ):
        matches = _matching_spans(hammerbacher, text)
        assert {page_number for page_number, _ in matches} == {1, 2}
        for page_number, span in matches:
            assert span.id in artifacts, (
                f"Repeated footer {text!r} on page {page_number} is not an artifact"
            )
            assert artifacts[span.id].kind is ArtifactKind.FOOTER
