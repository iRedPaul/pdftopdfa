# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Integration coverage for overprint after default colour spaces are installed."""

from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name

from pdftopdfa.color_profile import get_srgb_profile
from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf


@pytest.mark.parametrize("level", ["2b", "3b"])
def test_default_cmyk_overprint_is_repaired_after_profile_embedding(
    tmp_path: Path, level: str
) -> None:
    """Validate local repairs when profile embedding installs DefaultCMYK."""
    source = tmp_path / "input.pdf"
    target = tmp_path / "output.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(200, 200))
        profile = pdf.make_stream(get_srgb_profile())
        profile.N = 3
        pdf.Root.OutputIntents = Array(
            [
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name.GTS_PDFA1,
                    OutputConditionIdentifier="sRGB",
                    DestOutputProfile=profile,
                )
            ]
        )
        page.Resources = Dictionary(ExtGState=Dictionary(GS=Dictionary(OPM=1, OP=True)))
        page.Contents = pdf.make_stream(
            b"/GS gs 1 0 0 0 k 20 20 160 160 re f 0 0 0 1 k 60 60 80 80 re f"
        )
        pdf.save(source)

    result = convert_to_pdfa(source, target, level=level)
    assert result.success and result.target_produced and not result.skipped, (
        result.warnings
    )
    assert result.sanitization_stats["extgstate_fixed"] == 2
    assert any(
        "after color-profile embedding" in warning for warning in result.warnings
    )
    with pikepdf.open(target) as pdf:
        page = pdf.pages[0]
        assert page.Resources.ColorSpace.DefaultCMYK[0] == Name.ICCBased
        assert page.Resources.ExtGState.GS.OPM == 1
        instructions = list(pikepdf.parse_content_stream(page))
        for index, instruction in enumerate(instructions):
            if str(instruction.operator) != "f":
                continue
            # ISO 32000-1, 8.5.1: the guard precedes the whole path object.
            assert str(instructions[index - 1].operator) == "re"
            before, after = instructions[index - 2], instructions[index + 1]
            assert str(before.operator) == str(after.operator) == "gs"
            assert page.Resources.ExtGState[str(before.operands[0])].OPM == 0
            assert page.Resources.ExtGState[str(after.operands[0])].OPM == 1

    if is_verapdf_available():
        validation = validate_with_verapdf(target, flavour=level)
        assert validation.compliant, validation.failed_rules
