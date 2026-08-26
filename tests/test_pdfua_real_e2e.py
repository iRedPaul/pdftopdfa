# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End-to-end PDF/UA validation for representative repository documents."""

from pathlib import Path

import pytest

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.verapdf import is_verapdf_available

_TEST_DOCS = Path(__file__).resolve().parents[1] / "test_docs"


@pytest.mark.skipif(not is_verapdf_available(), reason="veraPDF is not installed")
@pytest.mark.parametrize(
    "filename",
    [
        "180 - R3 - Rechnung.pdf",
        "244057509.2024-12-13T060041.808.pdf",
        "010 - LS1 - Lieferschein.pdf",
        "120-150 - ALS1-4 - Ausgangslieferscheine.pdf",
    ],
)
def test_representative_document_passes_pdfa_and_pdfua(
    filename: str,
    tmp_path: Path,
) -> None:
    source = _TEST_DOCS / filename
    if not source.is_file():
        pytest.skip(f"Real-document regression fixture is missing: {source}")

    result = convert_to_pdfa(
        source,
        tmp_path / "output.pdf",
        level="2a",
        pdfua=True,
    )

    assert result.success is True, result.warnings
    assert result.validation_failed is False
