# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for language-aware generated accessibility text."""

from pdftopdfa.accessibility import (
    accessibility_strings,
    infer_document_language,
)


def test_infers_german_only_from_strong_document_evidence() -> None:
    assert (
        infer_document_language(["Rechnung", "Kunde", "Menge", "Zahlungshinweis"])
        == "de"
    )
    assert infer_document_language(["Kunde", "total"]) is None


def test_infers_english_only_from_strong_document_evidence() -> None:
    assert infer_document_language(["Invoice", "Customer", "Quantity"]) == "en"
    assert infer_document_language(["total", "total", "total"]) is None


def test_localizes_supported_language_and_tags_english_fallback() -> None:
    assert accessibility_strings("de-DE").figure == "Bild"
    assert accessibility_strings("fr").language == "en"
