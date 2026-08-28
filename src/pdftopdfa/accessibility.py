# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Language-aware text used by generated accessibility metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccessibilityStrings:
    """Accessibility labels for one supported document language."""

    language: str
    figure: str
    formula: str


_ENGLISH = AccessibilityStrings(
    language="en",
    figure="Image",
    formula="Formula",
)
_GERMAN = AccessibilityStrings(
    language="de",
    figure="Bild",
    formula="Formel",
)
_STRINGS = {"de": _GERMAN, "en": _ENGLISH}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LANGUAGE_MARKERS = {
    "de": frozenset(
        {
            "artikel",
            "auftrag",
            "betrag",
            "brutto",
            "garantiebedingungen",
            "gesamt",
            "kunde",
            "kundentext",
            "lieferschein",
            "menge",
            "netto",
            "rechnung",
            "rechnungsnummer",
            "steuer",
            "unterschrift",
            "ware",
            "zahlungshinweis",
        }
    ),
    "en": frozenset(
        {
            "amount",
            "article",
            "customer",
            "delivery",
            "invoice",
            "item",
            "net",
            "order",
            "payment",
            "quantity",
            "signature",
            "subtotal",
            "tax",
            "total",
        }
    ),
}


def primary_language(value: object) -> str | None:
    """Return the primary BCP 47 language subtag, if available."""
    text = str(value or "").strip().replace("_", "-").casefold()
    if not text:
        return None
    return text.split("-", 1)[0]


def accessibility_strings(language: object) -> AccessibilityStrings:
    """Return localized strings, using English for unsupported languages."""
    return _STRINGS.get(primary_language(language), _ENGLISH)


def infer_document_language(texts: Iterable[str]) -> str | None:
    """Infer German or English only when document text gives strong evidence."""
    tokens = {token for text in texts for token in _WORD.findall(text.casefold())}
    scores = {
        language: len(tokens & markers)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    other_score = max(value for key, value in scores.items() if key != language)
    if score < 3 or score < other_score + 2:
        return None
    return language


__all__ = [
    "AccessibilityStrings",
    "accessibility_strings",
    "infer_document_language",
    "primary_language",
]
