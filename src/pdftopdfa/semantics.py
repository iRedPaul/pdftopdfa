# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Deterministic semantic layout planning for tagged PDF output.

The planner is deliberately independent of PDF and OCR libraries.  Coordinates
use a top-left origin: ``top`` increases towards the bottom of a page.  Input
spans are expected to be the smallest layout units for which downstream code
can create marked-content references (normally a text run, image, or form).
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import product


class SpanKind(StrEnum):
    """Kinds of page content understood by the semantic planner."""

    TEXT = "text"
    IMAGE = "image"
    FORM = "form"


class ArtifactKind(StrEnum):
    """Reasons why content is excluded from the logical structure tree."""

    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    BACKGROUND = "background"
    LAYOUT = "layout"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned coordinates using a top-left origin."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("Bounding-box coordinates must be numbers")
        converted = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in converted):
            raise ValueError("Bounding-box coordinates must be finite")
        if converted[0] >= converted[2] or converted[1] >= converted[3]:
            raise ValueError("Bounding boxes must have positive width and height")
        object.__setattr__(self, "left", converted[0])
        object.__setattr__(self, "top", converted[1])
        object.__setattr__(self, "right", converted[2])
        object.__setattr__(self, "bottom", converted[3])

    @property
    def width(self) -> float:
        """Return the box width."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Return the box height."""
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        """Return the horizontal center."""
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        """Return the vertical center."""
        return (self.top + self.bottom) / 2


@dataclass(frozen=True, slots=True)
class SemanticSpan:
    """One neutral page-layout item.

    ``invisible`` describes rendering only.  Invisible OCR text remains logical
    content; it is not treated as an artifact merely because it is invisible.
    """

    id: str
    text: str
    bbox: BoundingBox
    font_size: float | None = None
    font_name: str | None = None
    kind: SpanKind = SpanKind.TEXT
    confidence: float | None = None
    invisible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Span IDs must be non-empty strings")
        if not isinstance(self.text, str):
            raise TypeError("Span text must be a string")
        if not isinstance(self.bbox, BoundingBox):
            try:
                bbox = BoundingBox(*self.bbox)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "bbox must be a BoundingBox or four coordinates"
                ) from exc
            object.__setattr__(self, "bbox", bbox)
        try:
            kind = SpanKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported span kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if self.font_size is not None:
            if (
                isinstance(self.font_size, bool)
                or not isinstance(self.font_size, (int, float))
                or not math.isfinite(self.font_size)
                or self.font_size <= 0
            ):
                raise ValueError("font_size must be a positive finite number")
            object.__setattr__(self, "font_size", float(self.font_size))
        if self.font_name is not None and not isinstance(self.font_name, str):
            raise TypeError("font_name must be a string or None")
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(self.confidence)
                or not 0 <= self.confidence <= 1
            ):
                raise ValueError("confidence must be between zero and one")
            object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.invisible, bool):
            raise TypeError("invisible must be a boolean")


@dataclass(frozen=True, slots=True)
class SemanticPage:
    """Semantic input for one one-based document page."""

    number: int
    width: float
    height: float
    spans: tuple[SemanticSpan, ...]
    reading_order_hint: tuple[str, ...] | None = None
    column_gutters: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.number, bool)
            or not isinstance(self.number, int)
            or self.number < 1
        ):
            raise ValueError("Page numbers must be positive integers")
        for name in ("width", "height"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Page {name} must be a positive finite number")
            object.__setattr__(self, name, float(value))
        try:
            spans = tuple(self.spans)
        except TypeError as exc:
            raise TypeError("spans must be iterable") from exc
        if any(not isinstance(span, SemanticSpan) for span in spans):
            raise TypeError("Every page span must be a SemanticSpan")
        object.__setattr__(self, "spans", spans)
        hint = self.reading_order_hint
        if hint is not None:
            if not isinstance(hint, tuple) or any(
                not isinstance(span_id, str) or not span_id for span_id in hint
            ):
                raise TypeError(
                    "reading_order_hint must be a tuple of non-empty span IDs"
                )
            span_ids = {span.id for span in spans}
            if (
                len(hint) != len(spans)
                or len(set(hint)) != len(hint)
                or set(hint) != span_ids
            ):
                raise ValueError(
                    "reading_order_hint must contain every page span ID exactly once"
                )
        gutters = self.column_gutters
        if gutters is not None:
            if not isinstance(gutters, tuple) or any(
                isinstance(gutter, bool)
                or not isinstance(gutter, (int, float))
                or not math.isfinite(gutter)
                for gutter in gutters
            ):
                raise TypeError("column_gutters must be a tuple of finite numbers")
            normalized = tuple(float(gutter) for gutter in gutters)
            if (
                any(not 0 < gutter < self.width for gutter in normalized)
                or tuple(sorted(set(normalized))) != normalized
            ):
                raise ValueError(
                    "column_gutters must be unique, ordered, and inside the page"
                )
            object.__setattr__(self, "column_gutters", normalized)


@dataclass(frozen=True, slots=True)
class ContentReference:
    """A physical reference to one complete input span."""

    span_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ValueError("Content references require a non-empty span ID")


@dataclass(frozen=True, slots=True)
class StructureAttribute:
    """One PDF 1.7 structure attribute entry."""

    owner: str
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class StructureNode:
    """One standard PDF 1.7 logical-structure element."""

    role: str
    content: tuple[ContentReference, ...] = ()
    children: tuple[StructureNode, ...] = ()
    attributes: tuple[StructureAttribute, ...] = ()
    bbox: BoundingBox | None = None
    page_number: int | None = None
    actual_text: str | None = None

    @property
    def span_ids(self) -> tuple[str, ...]:
        """Return IDs referenced directly by this node, without duplicates."""
        return tuple(dict.fromkeys(reference.span_id for reference in self.content))

    def walk(self) -> Iterable[StructureNode]:
        """Yield this node and all descendants in logical order."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Content that must be marked as an artifact instead of structure content."""

    span_id: str
    page_number: int
    kind: ArtifactKind
    pdf_type: str
    pdf_subtype: str | None


@dataclass(frozen=True, slots=True)
class PagePlan:
    """Page anchor structure and flattened physical content order.

    A continuation that begins on this page may reference content on following
    pages; those pages retain their own ``Div`` anchors and reading orders.
    """

    page_number: int
    structure: StructureNode
    reading_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    """A PDF-library-neutral tagged-document plan."""

    root: StructureNode
    pages: tuple[PagePlan, ...]
    artifacts: tuple[ArtifactReference, ...]


@dataclass(frozen=True, slots=True)
class _Cell:
    spans: tuple[SemanticSpan, ...]
    bbox: BoundingBox
    text: str


@dataclass(frozen=True, slots=True)
class _TableRow:
    cells: tuple[_Cell, ...]
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class _TableCandidate:
    rows: tuple[_TableRow, ...]
    bbox: BoundingBox
    span_ids: frozenset[str]
    node: StructureNode


@dataclass(frozen=True, slots=True)
class _Line:
    spans: tuple[SemanticSpan, ...]
    bbox: BoundingBox
    text: str
    font_size: float
    confidence: float
    bold_ratio: float


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str
    bbox: BoundingBox
    span_ids: tuple[str, ...]
    content: tuple[ContentReference, ...]
    text: str = ""
    font_size: float = 0.0
    confidence: float = 1.0
    bold_ratio: float = 0.0
    line: _Line | None = None
    node: StructureNode | None = None


@dataclass(frozen=True, slots=True)
class _OrderedBlock:
    block: _Block
    column: int | None
    segment: int


@dataclass(frozen=True, slots=True)
class _Columns:
    anchors: tuple[float, ...]
    gutters: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ListMatch:
    label: tuple[ContentReference, ...]
    body: tuple[ContentReference, ...]
    numbering: str
    family: str
    label_actual_text: str | None = None
    body_actual_text: str | None = None
    strong_single: bool = False


_WHITESPACE = re.compile(r"\s+")
_BOLD_NAME = re.compile(r"(?:bold|black|heavy|demi|semi)", re.IGNORECASE)
_NUMERIC_CELL = re.compile(
    r"^[\s$€£¥(+-]*\d[\d\s.,%:/-]*\)?"
    r"(?:\s*(?:[$€£¥�]|CHF|EUR|USD))?\s*$",
    re.IGNORECASE,
)
_PLAIN_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_ROMAN = re.compile(r"^[ivxlcdm]{1,8}$", re.IGNORECASE)
_PAGE_LABEL = re.compile(
    r"^(?:page|p\.?|seite)\s+(?P<number>\d{1,4}|[ivxlcdm]{1,8})"
    r"(?:\s*(?:of|von|/)\s*(?:\d{1,4}|[ivxlcdm]{1,8}))?$",
    re.IGNORECASE,
)
_PAGE_LABEL_PREFIX = re.compile(
    r"^(?:page|p\.?|seite)\s+(?P<number>\d{1,4}|[ivxlcdm]{1,8})"
    r"\s*(?:of|von|/)\s*(?:\d{1,4}|[ivxlcdm]{1,8})",
    re.IGNORECASE,
)
_PAGE_FRACTION = re.compile(
    r"^(?P<number>\d{1,4}|[ivxlcdm]{1,8})\s*(?:of|von|/)\s*"
    r"(?:\d{1,4}|[ivxlcdm]{1,8})$",
    re.IGNORECASE,
)
_DECORATED_PAGE_NUMBER = re.compile(
    r"^[-–—]\s*(?P<number>\d{1,4}|[ivxlcdm]{1,8})\s*[-–—]$",
    re.IGNORECASE,
)
_LIST_WITH_BODY = re.compile(
    r"^\s*(?P<label>"
    r"[•◦▪▫‣⁃·*\-–—]"
    r"|(?:\d{1,3}[.)]|\(\d{1,3}\))"
    r"|(?:[A-Za-z][.)]|\([A-Za-z]\))"
    r"|(?:[ivxlcdmIVXLCDM]{1,8}[.)]|\([ivxlcdmIVXLCDM]{1,8}\))"
    r")\s+(?P<body>\S.*)$"
)
_LIST_LABEL_ONLY = re.compile(
    r"^\s*(?P<label>"
    r"[•◦▪▫‣⁃·*\-–—]"
    r"|(?:\d{1,3}[.)]|\(\d{1,3}\))"
    r"|(?:[A-Za-z][.)]|\([A-Za-z]\))"
    r"|(?:[ivxlcdmIVXLCDM]{1,8}[.)]|\([ivxlcdmIVXLCDM]{1,8}\))"
    r")\s*$"
)


def _normalized(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _box_union(boxes: Iterable[BoundingBox]) -> BoundingBox:
    boxes = tuple(boxes)
    return BoundingBox(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def _vertical_overlap(left: BoundingBox, right: BoundingBox) -> float:
    overlap = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
    return overlap / min(left.height, right.height)


def _effective_font_size(span: SemanticSpan) -> float:
    return span.font_size if span.font_size is not None else span.bbox.height * 0.8


def _confidence(span: SemanticSpan) -> float:
    return span.confidence if span.confidence is not None else 1.0


def _is_bold(span: SemanticSpan) -> bool:
    return bool(span.font_name and _BOLD_NAME.search(span.font_name))


def _whole_reference(span: SemanticSpan) -> ContentReference:
    return ContentReference(span.id)


def _repeated_margin_band(page: SemanticPage, box: BoundingBox) -> str | None:
    margin = min(108.0, max(54.0, page.height * 0.12))
    if box.top <= margin:
        return "Header"
    if box.bottom >= page.height - margin:
        return "Footer"
    return None


def _repeated_nontext_margin_band(page: SemanticPage, box: BoundingBox) -> str | None:
    band = _repeated_margin_band(page, box)
    if band is None:
        return None
    margin = min(108.0, max(54.0, page.height * 0.12))
    band_top, band_bottom = (
        (0.0, margin) if band == "Header" else (page.height - margin, page.height)
    )
    overlap = max(0.0, min(box.bottom, band_bottom) - max(box.top, band_top))
    return band if overlap >= box.height * 0.8 else None


def _roman_value(text: str) -> int | None:
    if not _ROMAN.fullmatch(text):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(text.casefold()):
        value = values[character]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total if total > 0 else None


def _labelled_page_number(token: str, *, strong: bool) -> tuple[int, bool] | None:
    """Return a parsed page-label token, or None when it is not a number.

    ``_ROMAN`` also matches sequences that no Roman numeral system can value,
    such as ``iiiiiiiv``. Those must fall through instead of producing a
    ``None`` page number inside an otherwise well-typed result.
    """
    if token.isdigit():
        return int(token), strong
    value = _roman_value(token)
    return (value, strong) if value is not None else None


def _page_number(text: str) -> tuple[int, bool] | None:
    normalized = _normalized(text)
    for pattern in (_PAGE_LABEL, _PAGE_FRACTION, _DECORATED_PAGE_NUMBER):
        match = pattern.fullmatch(normalized)
        if match is not None:
            return _labelled_page_number(match.group("number"), strong=True)
    match = _PAGE_LABEL_PREFIX.match(normalized)
    if match is not None:
        return _labelled_page_number(match.group("number"), strong=True)
    if _PLAIN_PAGE_NUMBER.fullmatch(normalized):
        return int(normalized), False
    if _ROMAN.fullmatch(normalized) and len(normalized) > 1:
        return _labelled_page_number(normalized, strong=False)
    return None


def _artifact_reference(
    span: SemanticSpan,
    page: SemanticPage,
    kind: ArtifactKind,
    subtype: str | None,
) -> ArtifactReference:
    pdf_type = (
        "Pagination"
        if kind
        in {
            ArtifactKind.HEADER,
            ArtifactKind.FOOTER,
            ArtifactKind.PAGE_NUMBER,
        }
        else "Layout"
    )
    return ArtifactReference(span.id, page.number, kind, pdf_type, subtype)


def _detect_artifacts(
    pages: tuple[SemanticPage, ...],
) -> dict[str, ArtifactReference]:
    artifacts: dict[str, ArtifactReference] = {}

    for page in pages:
        text_spans = [
            span
            for span in page.spans
            if span.kind is SpanKind.TEXT and _normalized(span.text)
        ]
        has_ocr_text = any(span.invisible for span in text_spans)
        for span in page.spans:
            if span.kind is SpanKind.TEXT and not _normalized(span.text):
                artifacts[span.id] = _artifact_reference(
                    span, page, ArtifactKind.LAYOUT, None
                )
            elif (
                span.kind is SpanKind.IMAGE
                and has_ocr_text
                and span.bbox.width * span.bbox.height
                >= page.width * page.height * 0.72
            ):
                artifacts[span.id] = _artifact_reference(
                    span, page, ArtifactKind.BACKGROUND, None
                )

    number_candidates: dict[
        tuple[str, int], list[tuple[SemanticPage, SemanticSpan, int, bool]]
    ] = defaultdict(list)
    for page in pages:
        for span in page.spans:
            if span.id in artifacts or span.kind is not SpanKind.TEXT:
                continue
            band = _repeated_margin_band(page, span.bbox)
            parsed = _page_number(span.text)
            if band is None or parsed is None:
                continue
            value, strong = parsed
            horizontal_region = min(2, int(3 * span.bbox.center_x / page.width))
            number_candidates[(band, horizontal_region)].append(
                (page, span, value, strong)
            )
            if strong:
                artifacts[span.id] = _artifact_reference(
                    span,
                    page,
                    ArtifactKind.PAGE_NUMBER,
                    band,
                )

    for candidates in number_candidates.values():
        weak = [candidate for candidate in candidates if not candidate[3]]
        page_offsets = {value - page.number for page, _span, value, _strong in weak}
        has_sequence = (
            len({page.number for page, *_rest in weak}) >= 2 and len(page_offsets) == 1
        )
        for page, span, value, _strong in weak:
            centered = abs(span.bbox.center_x - page.width / 2) <= page.width * 0.12
            if has_sequence or (centered and value == page.number):
                band = _repeated_margin_band(page, span.bbox)
                artifacts[span.id] = _artifact_reference(
                    span,
                    page,
                    ArtifactKind.PAGE_NUMBER,
                    band,
                )

    repeated: dict[tuple[str, str], list[tuple[SemanticPage, SemanticSpan]]] = (
        defaultdict(list)
    )
    for page in pages:
        for span in page.spans:
            if span.id in artifacts or span.kind is not SpanKind.TEXT:
                continue
            text = _normalized(span.text).casefold()
            band = _repeated_margin_band(page, span.bbox)
            if band is None or not 1 < len(text) <= 120:
                continue
            repeated[(band, text)].append((page, span))

    minimum_pages = max(2, math.ceil(len(pages) * 0.6))
    for (band, _text), candidates in repeated.items():
        clusters: list[list[tuple[SemanticPage, SemanticSpan]]] = []
        bounds: list[tuple[float, float]] = []
        for page, span in sorted(
            candidates,
            key=lambda item: (
                item[1].bbox.center_x / item[0].width,
                item[0].number,
                item[1].id,
            ),
        ):
            position = span.bbox.center_x / page.width
            if not clusters:
                clusters.append([(page, span)])
                bounds.append((position, position))
                continue
            lower, upper = bounds[-1]
            if max(upper, position) - min(lower, position) <= 0.06:
                clusters[-1].append((page, span))
                bounds[-1] = (min(lower, position), max(upper, position))
            else:
                clusters.append([(page, span)])
                bounds.append((position, position))
        kind = ArtifactKind.HEADER if band == "Header" else ArtifactKind.FOOTER
        for cluster in clusters:
            if len({page.number for page, _span in cluster}) < minimum_pages:
                continue
            for page, span in cluster:
                artifacts[span.id] = _artifact_reference(span, page, kind, band)

    decorations: dict[tuple[str, SpanKind], list[tuple[SemanticPage, SemanticSpan]]] = (
        defaultdict(list)
    )
    for page in pages:
        for span in page.spans:
            band = _repeated_nontext_margin_band(page, span.bbox)
            if (
                span.id not in artifacts
                and span.kind is not SpanKind.TEXT
                and not _normalized(span.text)
                and band is not None
            ):
                decorations[(band, span.kind)].append((page, span))
    for (band, _kind), candidates in decorations.items():
        clusters: list[list[tuple[SemanticPage, SemanticSpan]]] = []
        bounds: list[
            tuple[
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
        ] = []
        cluster_bins: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        tolerances = (0.03, 0.015, 0.01, 0.01)
        for page, span in sorted(
            candidates,
            key=lambda item: (
                item[1].bbox.center_x / item[0].width,
                item[1].bbox.center_y / item[0].height,
                item[0].number,
                item[1].id,
            ),
        ):
            geometry = (
                span.bbox.center_x / page.width,
                span.bbox.center_y / page.height,
                span.bbox.width / page.width,
                span.bbox.height / page.height,
            )
            geometry_bin = tuple(
                math.floor(value / tolerance)
                for value, tolerance in zip(geometry, tolerances, strict=True)
            )
            nearby = sorted(
                {
                    cluster_index
                    for offsets in product((-1, 0, 1), repeat=4)
                    for cluster_index in cluster_bins.get(
                        tuple(
                            value + offset
                            for value, offset in zip(geometry_bin, offsets, strict=True)
                        ),
                        (),
                    )
                }
            )
            compatible = []
            for cluster_index in nearby:
                lower, upper = bounds[cluster_index]
                if all(
                    max(high, value) - min(low, value) <= tolerance
                    for value, low, high, tolerance in zip(
                        geometry, lower, upper, tolerances, strict=True
                    )
                ):
                    compatible.append(cluster_index)
            if compatible:
                cluster_index = min(
                    compatible,
                    key=lambda index: (
                        sum(
                            abs(value - (low + high) / 2) / tolerance
                            for value, low, high, tolerance in zip(
                                geometry,
                                bounds[index][0],
                                bounds[index][1],
                                tolerances,
                                strict=True,
                            )
                        ),
                        index,
                    ),
                )
                clusters[cluster_index].append((page, span))
                lower, upper = bounds[cluster_index]
                bounds[cluster_index] = (
                    tuple(
                        min(low, value)
                        for low, value in zip(lower, geometry, strict=True)
                    ),
                    tuple(
                        max(high, value)
                        for high, value in zip(upper, geometry, strict=True)
                    ),
                )
            else:
                cluster_index = len(clusters)
                clusters.append([(page, span)])
                bounds.append((geometry, geometry))
                cluster_bins[geometry_bin].append(cluster_index)
        kind = ArtifactKind.HEADER if band == "Header" else ArtifactKind.FOOTER
        for cluster in clusters:
            if len({page.number for page, _span in cluster}) < minimum_pages:
                continue
            for page, span in cluster:
                artifacts[span.id] = _artifact_reference(span, page, kind, band)
    return artifacts


def _cluster_rows(spans: Iterable[SemanticSpan]) -> list[list[SemanticSpan]]:
    rows: list[list[SemanticSpan]] = []
    row_boxes: list[BoundingBox] = []
    row_anchors: list[BoundingBox] = []
    for span in sorted(
        spans,
        key=lambda item: (item.bbox.top, item.bbox.left, item.bbox.bottom, item.id),
    ):
        best_index: int | None = None
        best_distance = math.inf
        for index in range(len(rows) - 1, -1, -1):
            anchor = row_anchors[index]
            if anchor.bottom < span.bbox.top - max(anchor.height, span.bbox.height):
                break
            center_distance = abs(anchor.center_y - span.bbox.center_y)
            tolerance = max(2.0, min(anchor.height, span.bbox.height) * 0.4)
            if (
                _vertical_overlap(anchor, span.bbox) >= 0.45
                or center_distance <= tolerance
            ) and center_distance < best_distance:
                best_index = index
                best_distance = center_distance
        if best_index is None:
            rows.append([span])
            row_boxes.append(span.bbox)
            row_anchors.append(span.bbox)
        else:
            rows[best_index].append(span)
            row_box = row_boxes[best_index]
            row_boxes[best_index] = BoundingBox(
                min(row_box.left, span.bbox.left),
                min(row_box.top, span.bbox.top),
                max(row_box.right, span.bbox.right),
                max(row_box.bottom, span.bbox.bottom),
            )
    for row in rows:
        row.sort(key=lambda item: (item.bbox.left, item.bbox.top, item.id))
    paired_rows = sorted(
        zip(rows, row_boxes, strict=True),
        key=lambda item: (item[1].top, item[0][0].bbox.left),
    )
    return [row for row, _box in paired_rows]


def _cells_for_row(
    spans: list[SemanticSpan],
    *,
    generous: bool,
) -> tuple[_Cell, ...]:
    groups: list[list[SemanticSpan]] = []
    for span in spans:
        if not groups:
            groups.append([span])
            continue
        previous = groups[-1][-1]
        font_size = statistics.median(
            (_effective_font_size(previous), _effective_font_size(span))
        )
        threshold = max(4.0, font_size * (2.2 if generous else 0.8))
        if generous and _LIST_LABEL_ONLY.fullmatch(previous.text):
            threshold = max(threshold, font_size * 6.0)
        if span.bbox.left - previous.bbox.right <= threshold:
            groups[-1].append(span)
        else:
            groups.append([span])
    return tuple(
        _Cell(
            spans=tuple(group),
            bbox=_box_union(span.bbox for span in group),
            text=" ".join(text for span in group if (text := _normalized(span.text))),
        )
        for group in groups
    )


def _compatible_table_rows(
    rows: tuple[_TableRow, ...],
    page_width: float,
) -> bool:
    if len(rows) < 2 or len({len(row.cells) for row in rows}) != 1:
        return False
    tolerance = max(8.0, page_width * 0.025)
    for column in range(len(rows[0].cells)):
        cells = [row.cells[column] for row in rows]
        spreads = (
            max(cell.bbox.left for cell in cells)
            - min(cell.bbox.left for cell in cells),
            max(cell.bbox.right for cell in cells)
            - min(cell.bbox.right for cell in cells),
            max(cell.bbox.center_x for cell in cells)
            - min(cell.bbox.center_x for cell in cells),
        )
        if min(spreads) > tolerance:
            return False
    return all(
        current.bbox.top - previous.bbox.bottom
        <= max(previous.bbox.height, current.bbox.height) * 1.6
        for previous, current in zip(rows, rows[1:], strict=False)
    )


def _weighted_bold(spans: Iterable[SemanticSpan]) -> float:
    weighted = [(max(1, len(_normalized(span.text))), _is_bold(span)) for span in spans]
    total = sum(weight for weight, _bold in weighted)
    return sum(weight for weight, bold in weighted if bold) / total if total else 0.0


def _table_has_header(rows: tuple[_TableRow, ...]) -> bool:
    header_spans = tuple(span for cell in rows[0].cells for span in cell.spans)
    body_spans = tuple(
        span for row in rows[1:] for cell in row.cells for span in cell.spans
    )
    header_bold = _weighted_bold(header_spans)
    body_bold = _weighted_bold(body_spans)
    header_size = statistics.median(_effective_font_size(span) for span in header_spans)
    body_size = statistics.median(_effective_font_size(span) for span in body_spans)
    bold_evidence = header_bold >= 0.6 and header_bold - body_bold >= 0.3
    size_evidence = header_size >= body_size * 1.08
    body_cells = [cell for row in rows[1:] for cell in row.cells]
    numeric_evidence = sum(
        bool(_NUMERIC_CELL.fullmatch(cell.text)) for cell in body_cells
    ) >= max(1, math.ceil(len(body_cells) * 0.5)) and all(
        any(character.isalpha() for character in cell.text) for cell in rows[0].cells
    )
    compact = (
        statistics.median(len(cell.text) for row in rows for cell in row.cells) <= 80
    )
    return compact and (bold_evidence or size_evidence or numeric_evidence)


def _looks_like_marker(text: str) -> bool:
    return _LIST_LABEL_ONLY.fullmatch(text) is not None


def _merge_table_continuation(
    previous: _TableRow,
    continuation: _TableRow,
) -> _TableRow:
    cells = list(previous.cells)
    for continuation_cell in continuation.cells:
        overlaps = [
            max(
                0.0,
                min(cell.bbox.right, continuation_cell.bbox.right)
                - max(cell.bbox.left, continuation_cell.bbox.left),
            )
            for cell in cells
        ]
        if max(overlaps) > 0:
            target = max(
                range(len(cells)),
                key=lambda index: (
                    overlaps[index]
                    / min(cells[index].bbox.width, continuation_cell.bbox.width),
                    overlaps[index],
                    -index,
                ),
            )
        else:
            target = min(
                range(len(cells)),
                key=lambda index: (
                    abs(cells[index].bbox.center_x - continuation_cell.bbox.center_x),
                    index,
                ),
            )
        cell = cells[target]
        spans = (*cell.spans, *continuation_cell.spans)
        cells[target] = _Cell(
            spans,
            _box_union((cell.bbox, continuation_cell.bbox)),
            " ".join(text for span in spans if (text := _normalized(span.text))),
        )
    return _TableRow(
        tuple(cells),
        _box_union((previous.bbox, continuation.bbox)),
    )


def _table_row_alignment(
    header: _TableRow,
    row: _TableRow,
    page_width: float,
) -> int:
    tolerance = max(10.0, page_width * 0.05)
    return sum(
        any(
            abs(cell.bbox.left - header_cell.bbox.left) <= tolerance
            for header_cell in header.cells
        )
        for cell in row.cells
    )


def _strong_table_header(row: _TableRow) -> bool:
    if len(row.cells) < 2:
        return False
    spans = tuple(span for cell in row.cells for span in cell.spans)
    bold_cells = sum(any(_is_bold(span) for span in cell.spans) for cell in row.cells)
    return _weighted_bold(spans) >= 0.6 and bold_cells >= math.ceil(
        len(row.cells) * 0.75
    )


def _sparse_table_continuation(
    header: _TableRow,
    previous: _TableRow,
    row: _TableRow,
    page_width: float,
    *,
    into_header: bool,
) -> bool:
    if len(row.cells) >= len(previous.cells):
        return False
    tolerance = max(6.0, page_width * 0.015)
    if into_header:
        if any(
            abs(cell.bbox.left - header.cells[0].bbox.left) <= tolerance
            for cell in row.cells
        ):
            return False
        spans = tuple(span for cell in row.cells for span in cell.spans)
        if _weighted_bold(spans) < 0.6:
            return False
    return all(
        any(
            abs(cell.bbox.left - previous_cell.bbox.left) <= tolerance
            or max(
                0.0,
                min(cell.bbox.right, previous_cell.bbox.right)
                - max(cell.bbox.left, previous_cell.bbox.left),
            )
            >= min(cell.bbox.width, previous_cell.bbox.width) * 0.5
            for previous_cell in previous.cells
        )
        for cell in row.cells
    )


def _table_candidate(
    rows: tuple[_TableRow, ...],
    page_number: int,
) -> _TableCandidate:
    node = _table_node(rows, page_number)
    span_ids = frozenset(
        span.id for row in rows for cell in row.cells for span in cell.spans
    )
    return _TableCandidate(rows, node.bbox, span_ids, node)


def _table_node(rows: tuple[_TableRow, ...], page_number: int) -> StructureNode:
    row_nodes = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row.cells:
            role = "TH" if row_index == 0 else "TD"
            attributes = (
                (StructureAttribute("Table", "Scope", "Column"),)
                if role == "TH"
                else ()
            )
            cells.append(
                StructureNode(
                    role,
                    tuple(_whole_reference(span) for span in cell.spans),
                    attributes=attributes,
                    bbox=cell.bbox,
                    page_number=page_number,
                )
            )
        row_nodes.append(
            StructureNode(
                "TR",
                children=tuple(cells),
                bbox=row.bbox,
                page_number=page_number,
            )
        )
    return StructureNode(
        "Table",
        children=tuple(row_nodes),
        bbox=_box_union(row.bbox for row in rows),
        page_number=page_number,
    )


def _table_candidates(
    page: SemanticPage,
    spans: list[SemanticSpan],
) -> tuple[_TableCandidate, ...]:
    raw_rows = []
    for row_spans in _cluster_rows(spans):
        cells = _cells_for_row(row_spans, generous=False)
        if 1 <= len(cells) <= 12:
            raw_rows.append(_TableRow(cells, _box_union(cell.bbox for cell in cells)))

    candidates: list[_TableCandidate] = []
    index = 0
    while index < len(raw_rows):
        first = raw_rows[index]
        if len(first.cells) < 2:
            index += 1
            continue
        run = [first]
        minimums = [
            [cell.bbox.left, cell.bbox.right, cell.bbox.center_x]
            for cell in first.cells
        ]
        maximums = [values.copy() for values in minimums]
        tolerance = max(8.0, page.width * 0.025)
        first_is_header = _strong_table_header(first)
        next_index = index + 1
        while next_index < len(raw_rows):
            next_row = raw_rows[next_index]
            if len(next_row.cells) != len(first.cells):
                break
            if not first_is_header and _strong_table_header(next_row):
                break
            previous = run[-1]
            if (
                next_row.bbox.top - previous.bbox.bottom
                > max(previous.bbox.height, next_row.bbox.height) * 1.6
            ):
                break
            proposed_minimums = []
            proposed_maximums = []
            for cell, lower, upper in zip(
                next_row.cells, minimums, maximums, strict=True
            ):
                values = [cell.bbox.left, cell.bbox.right, cell.bbox.center_x]
                proposed_lower = [
                    min(current, value)
                    for current, value in zip(lower, values, strict=True)
                ]
                proposed_upper = [
                    max(current, value)
                    for current, value in zip(upper, values, strict=True)
                ]
                if (
                    min(
                        high - low
                        for low, high in zip(
                            proposed_lower, proposed_upper, strict=True
                        )
                    )
                    > tolerance
                ):
                    break
                proposed_minimums.append(proposed_lower)
                proposed_maximums.append(proposed_upper)
            if len(proposed_minimums) != len(first.cells):
                break
            minimums = proposed_minimums
            maximums = proposed_maximums
            run.append(next_row)
            next_index += 1
        candidate_rows = tuple(run)
        first_column = [row.cells[0].text for row in candidate_rows]
        if (
            len(candidate_rows) >= 2
            and _compatible_table_rows(candidate_rows, page.width)
            and not all(_looks_like_marker(text) for text in first_column)
            and _table_has_header(candidate_rows)
        ):
            candidates.append(_table_candidate(candidate_rows, page.number))
        index = next_index if next_index > index + 1 else index + 1

    index = 0
    while index < len(raw_rows):
        header = raw_rows[index]
        if not _strong_table_header(header):
            index += 1
            continue
        logical_rows = [header]
        last_physical = header
        next_index = index + 1
        while next_index < len(raw_rows):
            row = raw_rows[next_index]
            gap = row.bbox.top - last_physical.bbox.bottom
            height = max(last_physical.bbox.height, row.bbox.height)
            continuation_gap = max(3.0, height * 0.45)
            if (
                gap >= -height * 0.5
                and gap <= continuation_gap
                and _sparse_table_continuation(
                    header,
                    logical_rows[-1],
                    row,
                    page.width,
                    into_header=len(logical_rows) == 1,
                )
            ):
                logical_rows[-1] = _merge_table_continuation(logical_rows[-1], row)
                last_physical = row
                next_index += 1
                continue
            if (
                _strong_table_header(row)
                and len(row.cells) >= len(header.cells)
                and gap > continuation_gap
            ):
                break
            numeric = sum(
                _NUMERIC_CELL.fullmatch(cell.text) is not None for cell in row.cells
            )
            alignment = _table_row_alignment(header, row, page.width)
            required_alignment = 1 if numeric else min(2, len(row.cells))
            if (
                len(row.cells) < 2
                or gap < -height * 0.5
                or gap > max(24.0, height * 4.0)
                or alignment < required_alignment
            ):
                break
            logical_rows.append(row)
            last_physical = row
            next_index += 1
        sparse_rows = tuple(logical_rows)
        if len(sparse_rows) >= 2 and _table_has_header(sparse_rows):
            candidates.append(_table_candidate(sparse_rows, page.number))
        index = next_index if next_index > index + 1 else index + 1

    selected: list[_TableCandidate] = []
    selected_ids: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (-len(item.span_ids), item.bbox.top, item.bbox.left),
    ):
        if candidate.span_ids.isdisjoint(selected_ids):
            selected.append(candidate)
            selected_ids.update(candidate.span_ids)
    return tuple(sorted(selected, key=lambda item: (item.bbox.top, item.bbox.left)))


def _table_gutters(
    page: SemanticPage,
    spans: list[SemanticSpan],
) -> tuple[tuple[float, float], ...]:
    if len(spans) < 8:
        return ()
    tolerance = max(
        18.0,
        statistics.median(_effective_font_size(span) for span in spans) * 2.0,
    )
    clusters: list[list[SemanticSpan]] = []
    anchors: list[float] = []
    for span in sorted(spans, key=lambda item: (item.bbox.left, item.bbox.top)):
        if not clusters:
            clusters.append([span])
            anchors.append(span.bbox.left)
            continue
        if span.bbox.left - anchors[-1] <= tolerance:
            clusters[-1].append(span)
            anchors[-1] += (span.bbox.left - anchors[-1]) / len(clusters[-1])
        else:
            clusters.append([span])
            anchors.append(span.bbox.left)
    clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    minimum_gap = max(36.0, page.width * 0.06)
    gutters = []
    for left_cluster, right_cluster in zip(clusters, clusters[1:], strict=False):
        left_edge = statistics.median(span.bbox.right for span in left_cluster)
        right_edge = statistics.median(span.bbox.left for span in right_cluster)
        if right_edge - left_edge >= minimum_gap:
            gutters.append(((left_edge + right_edge) / 2, right_edge - left_edge))
    return tuple(gutters)


def _same_table_schema(
    left: tuple[_TableCandidate, ...],
    right: tuple[_TableCandidate, ...],
) -> bool:
    left_headers = {
        tuple(_normalized(cell.text).casefold() for cell in candidate.rows[0].cells)
        for candidate in left
    }
    right_headers = {
        tuple(_normalized(cell.text).casefold() for cell in candidate.rows[0].cells)
        for candidate in right
    }
    return bool(left_headers & right_headers)


def _detect_tables(
    page: SemanticPage,
    artifact_ids: set[str],
) -> tuple[_TableCandidate, ...]:
    eligible = [
        span
        for span in page.spans
        if span.id not in artifact_ids
        and span.kind is SpanKind.TEXT
        and _normalized(span.text)
        and _confidence(span) >= 0.6
    ]
    best = _table_candidates(page, eligible)
    best_coverage = (
        len(set().union(*(candidate.span_ids for candidate in best))) if best else 0
    )
    inferred_gutters = _table_gutters(page, eligible)
    gutters = (
        *((gutter, math.inf, True) for gutter in (page.column_gutters or ())),
        *((gutter, width, False) for gutter, width in inferred_gutters),
    )
    widest_gutters = sorted(
        (
            (width, index)
            for index, (_position, width, explicit) in enumerate(gutters)
            if not explicit
        ),
        reverse=True,
    )
    for gutter_index, (gutter, gutter_width, explicit) in enumerate(gutters):
        left = [span for span in eligible if span.bbox.right <= gutter]
        right = [span for span in eligible if span.bbox.left >= gutter]
        left_candidates = _table_candidates(page, left)
        right_candidates = _table_candidates(page, right)
        if not left_candidates and not right_candidates:
            continue
        if not explicit and (not left_candidates or not right_candidates):
            continue
        partitioned = (*left_candidates, *right_candidates)
        partitioned_ids = set().union(
            *(candidate.span_ids for candidate in partitioned)
        )
        combined = (
            *(
                candidate
                for candidate in best
                if candidate.span_ids.isdisjoint(partitioned_ids)
            ),
            *partitioned,
        )
        coverage = len(set().union(*(candidate.span_ids for candidate in combined)))
        if not widest_gutters:
            other_gutter_width = 0.0
        elif widest_gutters[0][1] != gutter_index:
            other_gutter_width = widest_gutters[0][0]
        elif len(widest_gutters) > 1:
            other_gutter_width = widest_gutters[1][0]
        else:
            other_gutter_width = 0.0
        strong_gutter = gutter_width >= max(
            page.width * 0.12,
            other_gutter_width * 1.5,
        )
        replaces_spanning_candidate = any(
            candidate.bbox.left < gutter < candidate.bbox.right for candidate in best
        )
        if coverage > best_coverage or (
            replaces_spanning_candidate
            and (
                explicit
                or (
                    coverage == best_coverage
                    and strong_gutter
                    and _same_table_schema(left_candidates, right_candidates)
                )
            )
        ):
            best = tuple(
                sorted(combined, key=lambda item: (item.bbox.top, item.bbox.left))
            )
            best_coverage = coverage
    return best


def _make_line(spans: tuple[SemanticSpan, ...]) -> _Line:
    text = " ".join(
        normalized for span in spans if (normalized := _normalized(span.text))
    )
    weights = [max(1, len(_normalized(span.text))) for span in spans]
    weight_sum = sum(weights)
    font_size = (
        sum(
            _effective_font_size(span) * weight
            for span, weight in zip(spans, weights, strict=True)
        )
        / weight_sum
    )
    confidence = (
        sum(
            _confidence(span) * weight
            for span, weight in zip(spans, weights, strict=True)
        )
        / weight_sum
    )
    bold_ratio = (
        sum(
            weight
            for span, weight in zip(spans, weights, strict=True)
            if _is_bold(span)
        )
        / weight_sum
    )
    return _Line(
        spans,
        _box_union(span.bbox for span in spans),
        text,
        font_size,
        confidence,
        bold_ratio,
    )


def _text_blocks(
    page: SemanticPage,
    excluded_ids: set[str],
) -> list[_Block]:
    spans = [
        span
        for span in page.spans
        if span.id not in excluded_ids
        and span.kind is SpanKind.TEXT
        and _normalized(span.text)
    ]
    blocks = []
    for row in _cluster_rows(spans):
        for cell in _cells_for_row(row, generous=True):
            line = _make_line(cell.spans)
            blocks.append(
                _Block(
                    "line",
                    line.bbox,
                    tuple(span.id for span in line.spans),
                    tuple(_whole_reference(span) for span in line.spans),
                    line.text,
                    line.font_size,
                    line.confidence,
                    line.bold_ratio,
                    line,
                )
            )
    return blocks


def _page_blocks(
    page: SemanticPage,
    artifacts: dict[str, ArtifactReference],
) -> list[_Block]:
    artifact_ids = set(artifacts)
    tables = _detect_tables(page, artifact_ids)
    table_ids = (
        set().union(*(candidate.span_ids for candidate in tables)) if tables else set()
    )
    blocks = _text_blocks(page, artifact_ids | table_ids)
    for table in tables:
        ordered_ids = tuple(
            span.id for row in table.rows for cell in row.cells for span in cell.spans
        )
        blocks.append(
            _Block(
                "table",
                table.bbox,
                ordered_ids,
                (),
                node=table.node,
            )
        )
    for span in page.spans:
        if (
            span.id in artifact_ids
            or span.id in table_ids
            or span.kind is SpanKind.TEXT
        ):
            continue
        role = "Figure" if span.kind is SpanKind.IMAGE else "Form"
        node = StructureNode(
            role,
            (_whole_reference(span),),
            attributes=(
                (StructureAttribute("Layout", "Placement", "Block"),)
                if role == "Figure"
                else ()
            ),
            bbox=span.bbox,
            page_number=page.number,
        )
        blocks.append(
            _Block(
                role.casefold(),
                span.bbox,
                (span.id,),
                (_whole_reference(span),),
                node=node,
            )
        )
    return blocks


def _anchor_clusters(
    line_blocks: list[_Block],
    tolerance: float,
) -> list[list[_Block]]:
    clusters: list[list[_Block]] = []
    for block in sorted(line_blocks, key=lambda item: (item.bbox.left, item.bbox.top)):
        if not clusters:
            clusters.append([block])
            continue
        representative = statistics.median(item.bbox.left for item in clusters[-1])
        if block.bbox.left - representative <= tolerance:
            clusters[-1].append(block)
        else:
            clusters.append([block])
    return clusters


def _detect_columns(page: SemanticPage, blocks: list[_Block]) -> _Columns | None:
    lines = [
        block
        for block in blocks
        if block.kind == "line" and block.bbox.width < page.width * 0.58
    ]
    if len(lines) < 4:
        return None
    median_font = statistics.median(block.font_size for block in lines)
    tolerance = max(18.0, median_font * 2.0)
    clusters = [
        cluster for cluster in _anchor_clusters(lines, tolerance) if len(cluster) >= 2
    ]
    if not 2 <= len(clusters) <= 3:
        return None
    anchors = tuple(
        statistics.median(block.bbox.left for block in cluster) for cluster in clusters
    )
    if any(
        right - left < max(72.0, page.width * 0.16)
        for left, right in zip(anchors, anchors[1:], strict=False)
    ):
        return None
    for left_cluster, right_anchor in zip(clusters, anchors[1:], strict=False):
        typical_right = statistics.median(block.bbox.right for block in left_cluster)
        if typical_right >= right_anchor - page.width * 0.025:
            return None
    for left_cluster, right_cluster in zip(clusters, clusters[1:], strict=False):
        left_top = min(block.bbox.top for block in left_cluster)
        left_bottom = max(block.bbox.bottom for block in left_cluster)
        right_top = min(block.bbox.top for block in right_cluster)
        right_bottom = max(block.bbox.bottom for block in right_cluster)
        overlap = max(0.0, min(left_bottom, right_bottom) - max(left_top, right_top))
        smaller_extent = min(left_bottom - left_top, right_bottom - right_top)
        if overlap < smaller_extent * 0.25:
            return None
    gutters = tuple(
        (statistics.median(block.bbox.right for block in left_cluster) + right_anchor)
        / 2
        for left_cluster, right_anchor in zip(clusters, anchors[1:], strict=False)
    )
    return _Columns(anchors, gutters)


def _column_for(block: _Block, columns: _Columns) -> int:
    for index, gutter in enumerate(columns.gutters):
        if block.bbox.center_x < gutter:
            return index
    return len(columns.anchors) - 1


def _is_spanning(block: _Block, page: SemanticPage, columns: _Columns) -> bool:
    if block.bbox.width >= page.width * 0.65:
        return True
    padding = max(2.0, block.font_size * 0.25)
    return any(
        block.bbox.left < gutter - padding and block.bbox.right > gutter + padding
        for gutter in columns.gutters
    )


def _visual_key(block: _Block) -> tuple[float, float, float, str]:
    return (block.bbox.top, block.bbox.left, block.bbox.bottom, block.span_ids[0])


def _same_visual_row(left: BoundingBox, right: BoundingBox) -> bool:
    center_distance = abs(left.center_y - right.center_y)
    tolerance = max(2.0, min(left.height, right.height) * 0.4)
    return _vertical_overlap(left, right) >= 0.45 or center_distance <= tolerance


def _visual_rows(blocks: Iterable[_Block]) -> list[list[_Block]]:
    rows: list[list[_Block]] = []
    row_boxes: list[BoundingBox] = []
    row_anchors: list[BoundingBox] = []
    for block in sorted(blocks, key=_visual_key):
        best_index: int | None = None
        best_distance = math.inf
        for index in range(len(rows) - 1, -1, -1):
            anchor = row_anchors[index]
            if anchor.bottom < block.bbox.top - max(anchor.height, block.bbox.height):
                break
            center_distance = abs(anchor.center_y - block.bbox.center_y)
            if _same_visual_row(anchor, block.bbox) and center_distance < best_distance:
                best_index = index
                best_distance = center_distance
        if best_index is None:
            rows.append([block])
            row_boxes.append(block.bbox)
            row_anchors.append(block.bbox)
        else:
            rows[best_index].append(block)
            row_box = row_boxes[best_index]
            row_boxes[best_index] = BoundingBox(
                min(row_box.left, block.bbox.left),
                min(row_box.top, block.bbox.top),
                max(row_box.right, block.bbox.right),
                max(row_box.bottom, block.bbox.bottom),
            )
    for row in rows:
        row.sort(key=lambda block: (block.bbox.left, block.bbox.top, block.span_ids[0]))
    return [
        row
        for row, _row_box in sorted(
            zip(rows, row_boxes, strict=True),
            key=lambda item: (item[1].top, item[1].left),
        )
    ]


def _visual_row_order(blocks: Iterable[_Block]) -> list[_Block]:
    return [block for row in _visual_rows(blocks) for block in row]


def _order_region(
    blocks: list[_Block],
    columns: _Columns,
    segment: int,
) -> list[_OrderedBlock]:
    by_column: dict[int, list[_Block]] = defaultdict(list)
    for block in blocks:
        by_column[_column_for(block, columns)].append(block)
    return [
        _OrderedBlock(block, column, segment)
        for column in range(len(columns.anchors))
        for block in _visual_row_order(by_column[column])
    ]


def _order_page_blocks(
    page: SemanticPage,
    blocks: list[_Block],
) -> list[_OrderedBlock]:
    if page.reading_order_hint is not None:
        block_by_span: dict[str, int] = {}
        for block_index, block in enumerate(blocks):
            for span_id in block.span_ids:
                if span_id in block_by_span:
                    raise RuntimeError(
                        "Semantic blocks contain duplicate physical content"
                    )
                block_by_span[span_id] = block_index
        ordered = []
        seen_blocks: set[int] = set()
        for span_id in page.reading_order_hint:
            block_index = block_by_span.get(span_id)
            if block_index is None or block_index in seen_blocks:
                continue
            seen_blocks.add(block_index)
            ordered.append(_OrderedBlock(blocks[block_index], 0, 0))
        if len(seen_blocks) != len(blocks):
            raise RuntimeError("Semantic block order is incomplete")
        return ordered

    columns = _detect_columns(page, blocks)
    if columns is None:
        return [_OrderedBlock(block, 0, 0) for block in _visual_row_order(blocks)]
    spanning = sorted(
        (block for block in blocks if _is_spanning(block, page, columns)),
        key=_visual_key,
    )
    remaining = [block for block in blocks if block not in spanning]
    ordered: list[_OrderedBlock] = []
    segment = 0
    for spanning_block in spanning:
        before = [
            block
            for block in remaining
            if block.bbox.center_y < spanning_block.bbox.center_y
        ]
        ordered.extend(_order_region(before, columns, segment))
        remaining = [block for block in remaining if block not in before]
        ordered.append(_OrderedBlock(spanning_block, None, segment))
        segment += 1
    ordered.extend(_order_region(remaining, columns, segment))
    return ordered


def _body_font_size(pages: tuple[SemanticPage, ...], artifact_ids: set[str]) -> float:
    weights: Counter[float] = Counter()
    for page in pages:
        for span in page.spans:
            if (
                span.id in artifact_ids
                or span.kind is not SpanKind.TEXT
                or not _normalized(span.text)
            ):
                continue
            bucket = round(_effective_font_size(span) * 2) / 2
            weights[bucket] += max(1, len(_normalized(span.text)))
    if not weights:
        return 10.0
    return min(
        weights,
        key=lambda size: (-weights[size], abs(size - statistics.median(weights)), size),
    )


def _heading_candidate(
    block: _Block,
    body_size: float,
    *,
    has_row_peer: bool,
) -> bool:
    if block.kind != "line" or block.confidence < 0.6:
        return False
    text = block.text.strip()
    if not text or len(text) > 160 or len(text.split()) > 24:
        return False
    if _NUMERIC_CELL.fullmatch(text):
        return False
    body_style = block.font_size < body_size * 1.4
    if body_style and has_row_peer:
        return False
    if body_style and ":" in text and any(character.isdigit() for character in text):
        return False
    if text.endswith((".", ";", ",")) and not re.match(r"^\d+(?:\.\d+)*\s+", text):
        return False
    if _LIST_WITH_BODY.fullmatch(text):
        marker = _LIST_WITH_BODY.fullmatch(text).group("label")
        if block.font_size < body_size * 1.08 and block.bold_ratio < 0.6:
            return False
        if marker in {"-", "–", "—", "•", "◦", "▪", "▫", "‣", "⁃", "·", "*"}:
            return False
    if block.line is not None and all(span.invisible for span in block.line.spans):
        if block.font_size < body_size * 1.8:
            return False
        return len(text.split()) != 1 or (
            block.font_size >= body_size * 2.8
            and block.bbox.width >= block.font_size * 3.0
        )
    return block.font_size >= body_size * 1.4 or (
        block.bold_ratio >= 0.6 and block.font_size >= body_size * 0.98
    )


def _heading_roles(
    ordered_pages: tuple[tuple[_OrderedBlock, ...], ...],
    body_size: float,
) -> dict[str, str]:
    row_peers: set[int] = set()
    for page in ordered_pages:
        ordered_by_block = {id(ordered.block): ordered for ordered in page}
        line_blocks = [
            ordered.block for ordered in page if ordered.block.kind == "line"
        ]
        for row in _visual_rows(line_blocks):
            if len(row) > 1:
                by_region: dict[tuple[int | None, int], list[_Block]] = defaultdict(
                    list
                )
                for block in row:
                    ordered = ordered_by_block[id(block)]
                    by_region[(ordered.column, ordered.segment)].append(block)
                if len(by_region) > 1 and all(
                    len(peers) == 1 for peers in by_region.values()
                ):
                    if not all(
                        _heading_candidate(
                            block,
                            body_size,
                            has_row_peer=False,
                        )
                        for block in row
                    ):
                        row_peers.update(id(block) for block in row)
                    continue
                for peers in by_region.values():
                    if len(peers) > 1:
                        row_peers.update(id(block) for block in peers)
    candidates = [
        ordered.block
        for page in ordered_pages
        for ordered in page
        if _heading_candidate(
            ordered.block,
            body_size,
            has_row_peer=id(ordered.block) in row_peers,
        )
    ]
    styles = sorted(
        {
            (round(block.font_size * 2) / 2, block.bold_ratio >= 0.6)
            for block in candidates
        },
        key=lambda style: (-style[0], -style[1]),
    )
    raw_levels = {style: min(index + 1, 6) for index, style in enumerate(styles)}
    roles: dict[str, str] = {}
    previous_level = 0
    for page in ordered_pages:
        for ordered in page:
            block = ordered.block
            if block not in candidates:
                continue
            raw = raw_levels[(round(block.font_size * 2) / 2, block.bold_ratio >= 0.6)]
            level = 1 if previous_level == 0 else min(raw, previous_level + 1)
            roles[block.span_ids[0]] = f"H{level}"
            previous_level = level
    return roles


def _list_numbering(label: str) -> tuple[str, str]:
    stripped = label.strip("().")
    if label[0] in r"•◦▪▫‣⁃·*\-–—":
        return "Disc", "bullet"
    if stripped.isdigit():
        return "Decimal", "ordered"
    if _ROMAN.fullmatch(stripped) and len(stripped) > 1:
        return ("UpperRoman" if stripped.isupper() else "LowerRoman"), "ordered"
    if stripped.isalpha():
        return ("UpperAlpha" if stripped.isupper() else "LowerAlpha"), "ordered"
    return "None", "bullet"


def _parse_list(block: _Block) -> _ListMatch | None:
    if block.kind != "line" or block.confidence < 0.6 or block.line is None:
        return None
    first = block.line.spans[0]
    match = _LIST_WITH_BODY.fullmatch(first.text)
    if match is not None:
        label_text = match.group("label")
        body_text = " ".join(
            [
                match.group("body"),
                *(
                    text
                    for span in block.line.spans[1:]
                    if (text := _normalized(span.text))
                ),
            ]
        )
        numbering, family = _list_numbering(match.group("label"))
        return _ListMatch(
            (),
            tuple(_whole_reference(span) for span in block.line.spans),
            numbering,
            family,
            label_text,
            body_text,
            label_text in "•◦▪▫‣⁃",
        )

    marker = _LIST_LABEL_ONLY.fullmatch(first.text)
    if marker is None or len(block.line.spans) < 2:
        return None
    label = (_whole_reference(first),)
    body = tuple(_whole_reference(span) for span in block.line.spans[1:])
    label_text = marker.group("label")
    numbering, family = _list_numbering(label_text)
    return _ListMatch(
        label,
        body,
        numbering,
        family,
        strong_single=label_text in "•◦▪▫‣⁃",
    )


def _nearby(previous: _OrderedBlock, current: _OrderedBlock, factor: float) -> bool:
    gap = current.block.bbox.top - previous.block.bbox.bottom
    return (
        previous.column == current.column
        and previous.segment == current.segment
        and gap >= -min(previous.block.bbox.height, current.block.bbox.height) * 0.5
        and gap <= max(previous.block.font_size, current.block.font_size) * factor
    )


def _list_continuation(item: _OrderedBlock, candidate: _OrderedBlock) -> bool:
    if candidate.block.kind != "line" or _parse_list(candidate.block) is not None:
        return False
    return _nearby(
        item, candidate, 1.2
    ) and candidate.block.bbox.left >= item.block.bbox.left + max(
        4.0, item.block.font_size * 0.6
    )


def _consume_list(
    ordered: tuple[_OrderedBlock, ...],
    start: int,
    page_number: int,
    heading_roles: dict[str, str],
) -> tuple[StructureNode, int] | None:
    first_match = _parse_list(ordered[start].block)
    if first_match is None:
        return None
    base = ordered[start]
    matches: list[tuple[_OrderedBlock, _ListMatch, list[_OrderedBlock]]] = []
    index = start
    while index < len(ordered):
        item = ordered[index]
        match = _parse_list(item.block)
        if (
            match is None
            or match.family != first_match.family
            or item.column != base.column
            or item.segment != base.segment
            or abs(item.block.bbox.left - base.block.bbox.left)
            > max(8.0, base.block.font_size * 1.4)
            or item.block.span_ids[0] in heading_roles
        ):
            break
        if matches:
            previous_item, _previous_match, previous_continuations = matches[-1]
            previous_tail = (
                previous_continuations[-1] if previous_continuations else previous_item
            )
            if not _nearby(previous_tail, item, 1.8):
                break
        continuations: list[_OrderedBlock] = []
        next_index = index + 1
        while next_index < len(ordered) and _list_continuation(
            item if not continuations else continuations[-1], ordered[next_index]
        ):
            continuations.append(ordered[next_index])
            next_index += 1
        matches.append((item, match, continuations))
        index = next_index

    if len(matches) < 2 and not first_match.strong_single:
        return None
    item_nodes = []
    for item, match, continuations in matches:
        label = StructureNode(
            "Lbl",
            match.label,
            bbox=item.block.bbox,
            page_number=page_number,
            actual_text=match.label_actual_text,
        )
        body_references = match.body + tuple(
            reference
            for continuation in continuations
            for reference in continuation.block.content
        )
        item_box = _box_union(
            [
                item.block.bbox,
                *(continuation.block.bbox for continuation in continuations),
            ]
        )
        body = StructureNode(
            "LBody",
            body_references,
            bbox=item_box,
            page_number=page_number,
            actual_text=(
                " ".join(
                    [
                        match.body_actual_text,
                        *(
                            continuation.block.text
                            for continuation in continuations
                            if continuation.block.text
                        ),
                    ]
                )
                if match.body_actual_text is not None
                else None
            ),
        )
        item_nodes.append(
            StructureNode(
                "LI",
                children=(label, body),
                bbox=item_box,
                page_number=page_number,
            )
        )
    list_box = _box_union(node.bbox for node in item_nodes if node.bbox is not None)
    node = StructureNode(
        "L",
        children=tuple(item_nodes),
        attributes=(
            StructureAttribute("List", "ListNumbering", first_match.numbering),
        ),
        bbox=list_box,
        page_number=page_number,
    )
    return node, index


def _should_merge_paragraph(
    previous: _OrderedBlock,
    current: _OrderedBlock,
) -> bool:
    left = previous.block
    right = current.block
    if (
        left.kind != "line"
        or right.kind != "line"
        or not _nearby(previous, current, 0.95)
    ):
        return False
    if (
        max(left.font_size, right.font_size)
        > min(left.font_size, right.font_size) * 1.18
    ):
        return False
    gap = right.bbox.top - left.bbox.bottom
    if left.text.rstrip().endswith((".", "!", "?")) and gap > left.font_size * 0.45:
        return False
    return not (
        right.bbox.left - left.bbox.left > left.font_size * 1.5
        and not left.text.rstrip().endswith(("-", "‐", "‑"))
    )


def _paragraph_node(
    blocks: list[_OrderedBlock],
    page_number: int,
) -> StructureNode:
    return StructureNode(
        "P",
        tuple(reference for ordered in blocks for reference in ordered.block.content),
        bbox=_box_union(ordered.block.bbox for ordered in blocks),
        page_number=page_number,
    )


def _structure_for_page(
    page: SemanticPage,
    ordered: tuple[_OrderedBlock, ...],
    heading_roles: dict[str, str],
) -> StructureNode:
    children: list[StructureNode] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        block = current.block
        if block.node is not None:
            children.append(block.node)
            index += 1
            continue
        heading_role = heading_roles.get(block.span_ids[0])
        if heading_role is not None:
            children.append(
                StructureNode(
                    heading_role,
                    block.content,
                    bbox=block.bbox,
                    page_number=page.number,
                )
            )
            index += 1
            continue
        consumed = _consume_list(ordered, index, page.number, heading_roles)
        if consumed is not None:
            node, index = consumed
            children.append(node)
            continue
        paragraph = [current]
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            if (
                candidate.block.node is not None
                or candidate.block.span_ids[0] in heading_roles
                or _parse_list(candidate.block) is not None
                or not _should_merge_paragraph(paragraph[-1], candidate)
            ):
                break
            paragraph.append(candidate)
            next_index += 1
        children.append(_paragraph_node(paragraph, page.number))
        index = next_index
    page_box = BoundingBox(0, 0, page.width, page.height)
    return StructureNode(
        "Div",
        children=tuple(children),
        bbox=page_box,
        page_number=page.number,
    )


def _at_page_break(
    left_page: SemanticPage,
    left: _OrderedBlock,
    right_page: SemanticPage,
    right: _OrderedBlock,
    *,
    require_same_column: bool = True,
) -> bool:
    """Return whether two blocks occupy a compatible physical page break."""
    return (
        right_page.number == left_page.number + 1
        and abs(left_page.width - right_page.width)
        <= max(left_page.width, right_page.width) * 0.05
        and abs(left_page.height - right_page.height)
        <= max(left_page.height, right_page.height) * 0.05
        and left.block.bbox.bottom >= left_page.height * 0.72
        and right.block.bbox.top <= right_page.height * 0.28
        and (not require_same_column or left.column == right.column)
    )


def _same_boundary_text_style(left: _Block, right: _Block) -> bool:
    return (
        left.kind == right.kind == "line"
        and min(left.confidence, right.confidence) >= 0.6
        and max(left.font_size, right.font_size)
        <= min(left.font_size, right.font_size) * 1.12
        and abs(left.bold_ratio - right.bold_ratio) <= 0.3
    )


def _continues_sentence(left: str, right: str) -> bool:
    left = _normalized(left)
    right = _normalized(right)
    if not left or not right:
        return False
    if left.endswith(("-", "‐", "‑")):
        return True
    if left.endswith((".", "!", "?", ":", ";")):
        return False
    first_letter = next((character for character in right if character.isalpha()), None)
    return first_letter is not None and first_letter.islower()


def _aligned_across_pages(
    left_page: SemanticPage,
    left: _Block,
    right_page: SemanticPage,
    right: _Block,
    *,
    tolerance: float = 0.02,
) -> bool:
    return (
        abs(left.bbox.left / left_page.width - right.bbox.left / right_page.width)
        <= tolerance
    )


def _paragraph_continuation(
    left_page: SemanticPage,
    left_block: _OrderedBlock,
    left_node: StructureNode,
    right_page: SemanticPage,
    right_block: _OrderedBlock,
    right_node: StructureNode,
) -> StructureNode | None:
    if (
        left_node.role != right_node.role
        or left_node.role != "P"
        or left_node.children
        or right_node.children
        or left_node.actual_text is not None
        or right_node.actual_text is not None
        or not _at_page_break(left_page, left_block, right_page, right_block)
        or left_block.block.bbox.bottom < left_page.height * 0.82
        or right_block.block.bbox.top > right_page.height * 0.18
        or not _same_boundary_text_style(left_block.block, right_block.block)
        or not _aligned_across_pages(
            left_page,
            left_block.block,
            right_page,
            right_block.block,
        )
        or not _continues_sentence(left_block.block.text, right_block.block.text)
    ):
        return None
    return replace(
        left_node,
        content=(*left_node.content, *right_node.content),
        bbox=None,
        page_number=None,
    )


def _structure_list_numbering(node: StructureNode) -> str | None:
    return next(
        (
            attribute.value
            for attribute in node.attributes
            if attribute.owner == "List" and attribute.name == "ListNumbering"
        ),
        None,
    )


def _valid_list_items(node: StructureNode) -> bool:
    return bool(node.children) and all(
        item.role == "LI"
        and len(item.children) == 2
        and item.children[0].role == "Lbl"
        and item.children[1].role == "LBody"
        for item in node.children
    )


def _content_text(
    content: tuple[ContentReference, ...],
    span_texts: dict[str, str],
) -> str:
    return " ".join(
        text
        for reference in content
        if (text := _normalized(span_texts.get(reference.span_id, "")))
    )


def _ordered_block_for_node(
    node: StructureNode,
    ordered: tuple[_OrderedBlock, ...],
    *,
    last: bool,
) -> _OrderedBlock | None:
    span_ids = {
        reference.span_id
        for descendant in node.walk()
        for reference in descendant.content
    }
    candidates = reversed(ordered) if last else iter(ordered)
    return next(
        (
            candidate
            for candidate in candidates
            if span_ids.intersection(candidate.block.span_ids)
        ),
        None,
    )


def _list_label_text(item: StructureNode, span_texts: dict[str, str]) -> str:
    label = item.children[0]
    return _normalized(label.actual_text or _content_text(label.content, span_texts))


def _ordered_list_value(label: str, numbering: str) -> int | None:
    token = label.strip("().")
    if numbering == "Decimal" and token.isdigit():
        return int(token)
    if numbering in {"UpperAlpha", "LowerAlpha"} and len(token) == 1:
        character = token.casefold()
        return ord(character) - ord("a") + 1 if "a" <= character <= "z" else None
    if numbering in {"UpperRoman", "LowerRoman"}:
        return _roman_value(token)
    return None


def _list_continues_numbering(
    left: StructureNode,
    right: StructureNode,
    span_texts: dict[str, str],
) -> bool:
    numbering = _structure_list_numbering(left)
    if numbering is None or numbering != _structure_list_numbering(right):
        return False
    if numbering == "Disc":
        return True
    left_value = _ordered_list_value(
        _list_label_text(left.children[-1], span_texts), numbering
    )
    right_value = _ordered_list_value(
        _list_label_text(right.children[0], span_texts), numbering
    )
    return left_value is not None and right_value == left_value + 1


def _cross_page_list_continuation(
    left_page: SemanticPage,
    left_block: _OrderedBlock,
    left_node: StructureNode,
    right_page: SemanticPage,
    right_block: _OrderedBlock,
    right_node: StructureNode,
    span_texts: dict[str, str],
) -> StructureNode | None:
    if (
        left_node.role != right_node.role
        or left_node.role != "L"
        or not _valid_list_items(left_node)
        or not _valid_list_items(right_node)
        or left_node.attributes != right_node.attributes
        or not _at_page_break(left_page, left_block, right_page, right_block)
        or not _same_boundary_text_style(left_block.block, right_block.block)
        or not _aligned_across_pages(
            left_page,
            left_block.block,
            right_page,
            right_block.block,
        )
        or not _list_continues_numbering(left_node, right_node, span_texts)
    ):
        return None
    return replace(
        left_node,
        children=(*left_node.children, *right_node.children),
        bbox=None,
        page_number=None,
    )


def _list_body_continuation(
    left_page: SemanticPage,
    left_block: _OrderedBlock,
    left_node: StructureNode,
    right_page: SemanticPage,
    right_block: _OrderedBlock,
    right_node: StructureNode,
    span_texts: dict[str, str],
) -> StructureNode | None:
    if (
        left_node.role != "L"
        or right_node.role != "P"
        or not _valid_list_items(left_node)
        or right_node.children
        or right_node.actual_text is not None
        or not _at_page_break(left_page, left_block, right_page, right_block)
        or not _same_boundary_text_style(left_block.block, right_block.block)
        or not _continues_sentence(left_block.block.text, right_block.block.text)
    ):
        return None
    relative_indent = (
        right_block.block.bbox.left / right_page.width
        - left_block.block.bbox.left / left_page.width
    )
    if not 0 <= relative_indent <= 0.06:
        return None

    last_item = left_node.children[-1]
    label, body = last_item.children
    continuation_text = _content_text(right_node.content, span_texts)
    actual_text = body.actual_text
    if actual_text is not None and continuation_text:
        actual_text = f"{actual_text} {continuation_text}"
    merged_body = replace(
        body,
        content=(*body.content, *right_node.content),
        bbox=None,
        page_number=None,
        actual_text=actual_text,
    )
    merged_item = replace(
        last_item,
        children=(label, merged_body),
        bbox=None,
        page_number=None,
    )
    return replace(
        left_node,
        children=(*left_node.children[:-1], merged_item),
        bbox=None,
        page_number=None,
    )


def _table_header(
    node: StructureNode,
) -> tuple[StructureNode, tuple[StructureNode, ...]] | None:
    if node.role != "Table" or len(node.children) < 2:
        return None
    row = node.children[0]
    if (
        row.role != "TR"
        or len(row.children) < 2
        or any(cell.role != "TH" or cell.bbox is None for cell in row.children)
    ):
        return None
    return row, row.children


def _repeated_table_header(
    left: StructureNode,
    left_page: SemanticPage,
    right: StructureNode,
    right_page: SemanticPage,
    span_texts: dict[str, str],
) -> bool:
    left_header = _table_header(left)
    right_header = _table_header(right)
    if left_header is None or right_header is None:
        return False
    left_cells = left_header[1]
    right_cells = right_header[1]
    left_text = tuple(
        _content_text(cell.content, span_texts).casefold() for cell in left_cells
    )
    right_text = tuple(
        _content_text(cell.content, span_texts).casefold() for cell in right_cells
    )
    if not all(left_text) or left_text != right_text:
        return False
    return all(
        abs(
            left_cell.bbox.left / left_page.width
            - right_cell.bbox.left / right_page.width
        )
        <= 0.025
        and abs(
            left_cell.bbox.right / left_page.width
            - right_cell.bbox.right / right_page.width
        )
        <= 0.025
        for left_cell, right_cell in zip(left_cells, right_cells, strict=True)
    )


def _table_continuation(
    left_page: SemanticPage,
    left_block: _OrderedBlock,
    left_node: StructureNode,
    right_page: SemanticPage,
    right_block: _OrderedBlock,
    right_node: StructureNode,
    span_texts: dict[str, str],
) -> StructureNode | None:
    if (
        left_block.block.kind != right_block.block.kind
        or left_block.block.kind != "table"
        or not _at_page_break(
            left_page,
            left_block,
            right_page,
            right_block,
            require_same_column=False,
        )
        or not _repeated_table_header(
            left_node,
            left_page,
            right_node,
            right_page,
            span_texts,
        )
    ):
        return None
    return replace(
        left_node,
        children=(*left_node.children, *right_node.children),
        bbox=None,
        page_number=None,
    )


def _merge_page_continuations(
    pages: tuple[SemanticPage, ...],
    ordered_pages: tuple[tuple[_OrderedBlock, ...], ...],
    page_nodes: tuple[StructureNode, ...],
) -> tuple[StructureNode, ...]:
    """Merge only strongly evidenced logical structures across page breaks."""
    nodes = list(page_nodes)
    span_texts = {span.id: span.text for page in pages for span in page.spans}
    for index in range(len(pages) - 2, -1, -1):
        left_children = list(nodes[index].children)
        right_children = list(nodes[index + 1].children)
        if not left_children or not right_children:
            continue
        left_ordered = ordered_pages[index]
        right_ordered = ordered_pages[index + 1]
        if not left_ordered or not right_ordered:
            continue
        if left_children[-1].role == "Table":
            right_table_index = 0
            while right_table_index < len(right_children):
                child = right_children[right_table_index]
                if (
                    child.role not in {"Figure", "Form"}
                    or child.bbox is None
                    or child.bbox.top > pages[index + 1].height * 0.15
                ):
                    break
                right_table_index += 1
            if (
                0 < right_table_index < len(right_children)
                and right_children[right_table_index].role == "Table"
            ):
                left_table_block = _ordered_block_for_node(
                    left_children[-1], left_ordered, last=True
                )
                right_table_block = _ordered_block_for_node(
                    right_children[right_table_index], right_ordered, last=False
                )
                if (
                    left_table_block is not None
                    and right_table_block is not None
                    and all(
                        child.bbox is not None
                        and child.bbox.bottom <= right_table_block.block.bbox.top
                        for child in right_children[:right_table_index]
                    )
                ):
                    merged_table = _table_continuation(
                        pages[index],
                        left_table_block,
                        left_children[-1],
                        pages[index + 1],
                        right_table_block,
                        right_children[right_table_index],
                        span_texts,
                    )
                    if merged_table is not None:
                        left_children[-1] = merged_table
                        del right_children[right_table_index]
                        nodes[index] = replace(
                            nodes[index], children=tuple(left_children)
                        )
                        nodes[index + 1] = replace(
                            nodes[index + 1], children=tuple(right_children)
                        )
                        continue
        left_boundary = _ordered_block_for_node(
            left_children[-1], left_ordered, last=True
        )
        right_boundary = _ordered_block_for_node(
            right_children[0], right_ordered, last=False
        )
        if left_boundary is None or right_boundary is None:
            continue
        arguments = (
            pages[index],
            left_boundary,
            left_children[-1],
            pages[index + 1],
            right_boundary,
            right_children[0],
        )
        merged = _table_continuation(*arguments, span_texts)
        if merged is None:
            merged = _cross_page_list_continuation(*arguments, span_texts)
        merged_list_body = False
        if merged is None:
            merged = _list_body_continuation(*arguments, span_texts)
            merged_list_body = merged is not None
        if merged is None:
            merged = _paragraph_continuation(*arguments)
        if merged is None:
            continue
        left_children[-1] = merged
        del right_children[0]
        if merged_list_body and right_children:
            right_boundary = _ordered_block_for_node(
                right_children[0], right_ordered, last=False
            )
            if right_boundary is not None:
                arguments = (
                    pages[index],
                    left_boundary,
                    left_children[-1],
                    pages[index + 1],
                    right_boundary,
                    right_children[0],
                )
                continued_list = _cross_page_list_continuation(*arguments, span_texts)
                if continued_list is not None:
                    left_children[-1] = continued_list
                    del right_children[0]
        nodes[index] = replace(nodes[index], children=tuple(left_children))
        nodes[index + 1] = replace(nodes[index + 1], children=tuple(right_children))
    return tuple(nodes)


def _verify_content_ownership(
    pages: tuple[SemanticPage, ...],
    plan_pages: tuple[PagePlan, ...],
    root: StructureNode,
    artifact_ids: set[str],
) -> None:
    expected = {
        span.id for page in pages for span in page.spans if span.id not in artifact_ids
    }
    ownership = Counter(
        reference.span_id for node in root.walk() for reference in node.content
    )
    if set(ownership) != expected or any(count != 1 for count in ownership.values()):
        raise RuntimeError("Semantic plan has inconsistent physical content ownership")
    for page, page_plan in zip(pages, plan_pages, strict=True):
        expected_page = {span.id for span in page.spans if span.id not in artifact_ids}
        if set(page_plan.reading_order) != expected_page:
            raise RuntimeError("Semantic plan has an incomplete page reading order")


def build_semantic_plan(pages: Iterable[SemanticPage]) -> SemanticPlan:
    """Build a conservative, deterministic PDF 1.7 logical-structure plan.

    Pages and spans may arrive in any iterable order.  Page numbers and span IDs
    must be unique.  The output never mutates the inputs and never opens or writes
    a PDF file.
    """
    try:
        pages = tuple(pages)
    except TypeError as exc:
        raise TypeError("pages must be iterable") from exc
    if any(not isinstance(page, SemanticPage) for page in pages):
        raise TypeError("Every document page must be a SemanticPage")
    pages = tuple(sorted(pages, key=lambda page: page.number))
    page_numbers = [page.number for page in pages]
    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("Page numbers must be unique")
    span_ids = [span.id for page in pages for span in page.spans]
    if len(span_ids) != len(set(span_ids)):
        raise ValueError("Span IDs must be unique across the document")

    artifacts = _detect_artifacts(pages)
    ordered_pages = tuple(
        tuple(_order_page_blocks(page, _page_blocks(page, artifacts))) for page in pages
    )
    body_size = _body_font_size(pages, set(artifacts))
    heading_roles = _heading_roles(ordered_pages, body_size)

    page_nodes = tuple(
        _structure_for_page(page, ordered, heading_roles)
        for page, ordered in zip(pages, ordered_pages, strict=True)
    )
    page_nodes = _merge_page_continuations(pages, ordered_pages, page_nodes)
    page_plans = []
    for page, ordered, structure in zip(pages, ordered_pages, page_nodes, strict=True):
        reading_order = tuple(
            dict.fromkeys(
                span_id
                for ordered_block in ordered
                for span_id in ordered_block.block.span_ids
            )
        )
        page_plans.append(PagePlan(page.number, structure, reading_order))
    root = StructureNode("Document", children=page_nodes)
    span_tops = {span.id: span.bbox.top for page in pages for span in page.spans}
    artifact_order = tuple(
        sorted(
            artifacts.values(),
            key=lambda artifact: (
                artifact.page_number,
                span_tops[artifact.span_id],
                artifact.span_id,
            ),
        )
    )
    page_plans = tuple(page_plans)
    _verify_content_ownership(pages, page_plans, root, set(artifacts))
    return SemanticPlan(root, page_plans, artifact_order)


__all__ = [
    "ArtifactKind",
    "ArtifactReference",
    "BoundingBox",
    "ContentReference",
    "PagePlan",
    "SemanticPage",
    "SemanticPlan",
    "SemanticSpan",
    "SpanKind",
    "StructureAttribute",
    "StructureNode",
    "build_semantic_plan",
]
