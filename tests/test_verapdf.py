# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for verapdf.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdftopdfa.exceptions import VeraPDFError
from pdftopdfa.verapdf import (
    VALID_FLAVOURS,
    VeraPDFResult,
    _extract_flavour_from_profile,
    _get_verapdf_cmd,
    _normalize_flavour,
    _parse_verapdf_xml,
    get_verapdf_version,
    is_verapdf_available,
    validate_with_verapdf,
)


class TestIsVerapdfAvailable:
    """Tests for is_verapdf_available."""

    def test_returns_bool(self) -> None:
        """Checks that is_verapdf_available returns a boolean value."""
        result = is_verapdf_available()

        assert isinstance(result, bool)

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_returns_true_when_found(self, mock_which: MagicMock) -> None:
        """Returns True when verapdf is found in PATH."""
        mock_which.return_value = "/usr/local/bin/verapdf"

        result = is_verapdf_available()

        assert result is True
        mock_which.assert_called_once_with(_get_verapdf_cmd())

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_returns_false_when_not_found(self, mock_which: MagicMock) -> None:
        """Returns False when verapdf is not in PATH."""
        mock_which.return_value = None

        result = is_verapdf_available()

        assert result is False


class TestGetVerapdfCmd:
    """Tests for _get_verapdf_cmd and VERAPDF_PATH."""

    def test_returns_default_when_env_not_set(self) -> None:
        """Falls back to 'verapdf' when VERAPDF_PATH is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert _get_verapdf_cmd() == "verapdf"

    def test_returns_custom_path_when_env_set(self) -> None:
        """Returns custom path from VERAPDF_PATH."""
        with patch.dict("os.environ", {"VERAPDF_PATH": "/opt/verapdf/bin/verapdf"}):
            assert _get_verapdf_cmd() == "/opt/verapdf/bin/verapdf"

    def test_returns_executable_in_dir_when_dir_set(self, tmp_path: Path) -> None:
        """Returns <dir>/verapdf when VERAPDF_PATH points to a directory."""
        with patch.dict("os.environ", {"VERAPDF_PATH": str(tmp_path)}):
            assert _get_verapdf_cmd() == str(tmp_path / "verapdf")

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_is_available_uses_custom_path(self, mock_which: MagicMock) -> None:
        """is_verapdf_available uses VERAPDF_PATH."""
        mock_which.return_value = "/opt/verapdf/bin/verapdf"
        with patch.dict("os.environ", {"VERAPDF_PATH": "/opt/verapdf/bin/verapdf"}):
            result = is_verapdf_available()

        assert result is True
        mock_which.assert_called_once_with("/opt/verapdf/bin/verapdf")

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_validate_uses_custom_path(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """validate_with_verapdf uses VERAPDF_PATH in the command."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="1" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch.dict("os.environ", {"VERAPDF_PATH": "/opt/verapdf/bin/verapdf"}):
            validate_with_verapdf(pdf_path, flavour="2b")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/opt/verapdf/bin/verapdf"


class TestGetVerapdfVersion:
    """Tests for get_verapdf_version."""

    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_returns_none_when_not_available(self, mock_available: MagicMock) -> None:
        """Returns None when veraPDF is not available."""
        mock_available.return_value = False

        result = get_verapdf_version()

        assert result is None

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_returns_version_string(
        self, mock_available: MagicMock, mock_run: MagicMock
    ) -> None:
        """Returns version string."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(stdout="veraPDF 1.24.1\n")

        result = get_verapdf_version()

        assert result == "veraPDF 1.24.1"

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_returns_none_on_timeout(
        self, mock_available: MagicMock, mock_run: MagicMock
    ) -> None:
        """Returns None on timeout."""
        import subprocess

        mock_available.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("verapdf", 10)

        result = get_verapdf_version()

        assert result is None


class TestNormalizeFlavour:
    """Tests for _normalize_flavour."""

    @pytest.mark.parametrize(
        "input_flavour,expected",
        [
            ("2b", "2b"),
            ("2B", "2b"),
            ("1a", "1a"),
            ("3u", "3u"),
            ("PDF/A-2B", "2b"),
            ("PDFA-2B", "2b"),
            ("PDFA_2_B", "2b"),
            ("PDFA2B", "2b"),
            ("pdf/a-1a", "1a"),
            ("4", "4"),
            ("4e", "4e"),
            ("4f", "4f"),
            ("4E", "4e"),
            ("4F", "4f"),
            ("PDF/A-4", "4"),
            ("PDF/A-4E", "4e"),
            ("PDF/A-4F", "4f"),
        ],
    )
    def test_normalizes_valid_flavours(self, input_flavour: str, expected: str) -> None:
        """Normalizes various notations correctly."""
        result = _normalize_flavour(input_flavour)

        assert result == expected

    def test_raises_for_invalid_flavour(self) -> None:
        """Raises VeraPDFError for invalid flavours."""
        with pytest.raises(VeraPDFError, match="Invalid PDF/A flavour"):
            _normalize_flavour("4x")

    def test_raises_for_empty_flavour(self) -> None:
        """Raises VeraPDFError for empty strings."""
        with pytest.raises(VeraPDFError, match="Invalid PDF/A flavour"):
            _normalize_flavour("")


class TestExtractFlavourFromProfile:
    """Tests for _extract_flavour_from_profile."""

    @pytest.mark.parametrize(
        "profile_name,expected",
        [
            ("PDF/A-2B validation profile", "2b"),
            ("PDF/A-1A validation profile", "1a"),
            ("PDF/A-3U", "3u"),
            ("Some text with PDF/A-2b inside", "2b"),
            ("PDF/A-4 validation profile", "4"),
            ("PDF/A-4E validation profile", "4e"),
            ("PDF/A-4F validation profile", "4f"),
        ],
    )
    def test_extracts_flavour(self, profile_name: str, expected: str) -> None:
        """Extracts flavour from profile name."""
        result = _extract_flavour_from_profile(profile_name)

        assert result == expected

    def test_returns_none_for_invalid_profile(self) -> None:
        """Returns None when no flavour is recognized."""
        result = _extract_flavour_from_profile("Some random text")

        assert result is None


class TestParseVerapdfXml:
    """Tests for _parse_verapdf_xml."""

    def test_parses_compliant_report(self) -> None:
        """Parses a compliant validation report."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <report>
            <jobs>
                <job>
                    <validationReport isCompliant="true"
                        profileName="PDF/A-2B validation profile">
                        <details passedRules="123" failedRules="0">
                        </details>
                    </validationReport>
                </job>
            </jobs>
        </report>"""

        result = _parse_verapdf_xml(xml)

        assert result.compliant is True
        assert result.flavour == "2b"
        assert result.passed_rules == 123
        assert result.failed_rules == 0
        assert len(result.errors) == 0

    def test_parses_non_compliant_report(self) -> None:
        """Parses a non-compliant validation report."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <report>
            <jobs>
                <job>
                    <validationReport isCompliant="false"
                        profileName="PDF/A-2B validation profile">
                        <details passedRules="100" failedRules="5">
                            <rule status="failed" clause="6.2.3">
                                <description>Missing required metadata</description>
                            </rule>
                        </details>
                    </validationReport>
                </job>
            </jobs>
        </report>"""

        result = _parse_verapdf_xml(xml)

        assert result.compliant is False
        assert result.passed_rules == 100
        assert result.failed_rules == 5
        assert len(result.errors) > 0
        assert "6.2.3" in result.errors[0]

    def test_handles_invalid_xml(self) -> None:
        """Handles invalid XML without crashing."""
        xml = "not valid xml <<<"

        result = _parse_verapdf_xml(xml)

        assert result.compliant is False
        assert len(result.errors) > 0

    def test_handles_missing_validation_report(self) -> None:
        """Handles missing validationReport element."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <report>
            <jobs>
                <job>
                </job>
            </jobs>
        </report>"""

        result = _parse_verapdf_xml(xml)

        assert result.compliant is False
        assert len(result.warnings) > 0

    def test_preserves_raw_xml(self) -> None:
        """Stores the raw XML in the result."""
        xml = "<report></report>"

        result = _parse_verapdf_xml(xml)

        assert result.raw_xml == xml


class TestValidateWithVerapdf:
    """Tests for validate_with_verapdf."""

    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_when_not_available(
        self, mock_available: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError when veraPDF is not available."""
        mock_available.return_value = False
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="not installed"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_for_nonexistent_file(
        self, mock_available: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError when file does not exist."""
        mock_available.return_value = True
        pdf_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(VeraPDFError, match="not found"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_builds_correct_command(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Builds the correct veraPDF command."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="1" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        validate_with_verapdf(pdf_path, flavour="2b")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "verapdf" in cmd
        assert "--format" in cmd
        assert "xml" in cmd
        assert "--flavour" in cmd
        assert "2b" in cmd
        assert str(pdf_path) in cmd

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_handles_timeout(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Handles timeout correctly."""
        import subprocess

        mock_available.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("verapdf", 300)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="timeout"):
            validate_with_verapdf(pdf_path, timeout=300)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_returns_result_on_success(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Returns VeraPDFResult on successful validation."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="100" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validate_with_verapdf(pdf_path, flavour="2b")

        assert isinstance(result, VeraPDFResult)
        assert result.compliant is True
        assert result.passed_rules == 100

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_on_empty_output(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError on empty output."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="no output"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_on_stderr_error(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError on error output."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            stdout="", stderr="Error: Invalid PDF file", returncode=0
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="Invalid PDF file"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_on_nonzero_exit_code(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError when veraPDF exits with code >= 2."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            stdout="", stderr="Java heap space", returncode=2
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="exit code 2"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_exit_code_1_is_valid(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Exit code 1 (non-compliant) is treated as a valid result."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="false" profileName="PDF/A-2B">'
            '<details passedRules="90" failedRules="5"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=1)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validate_with_verapdf(pdf_path, flavour="2b")

        assert isinstance(result, VeraPDFResult)
        assert result.compliant is False


class TestVerapdfResult:
    """Tests for the VeraPDFResult data class."""

    def test_default_values(self) -> None:
        """Checks default values."""
        result = VeraPDFResult(compliant=False)

        assert result.compliant is False
        assert result.flavour is None
        assert result.passed_rules == 0
        assert result.failed_rules == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.raw_xml is None

    def test_custom_values(self) -> None:
        """Checks custom values."""
        result = VeraPDFResult(
            compliant=True,
            flavour="2b",
            passed_rules=100,
            failed_rules=5,
            errors=["error1"],
            warnings=["warning1"],
            raw_xml="<xml/>",
        )

        assert result.compliant is True
        assert result.flavour == "2b"
        assert result.passed_rules == 100
        assert result.failed_rules == 5
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
        assert result.raw_xml == "<xml/>"


class TestValidFlavours:
    """Tests for VALID_FLAVOURS constant."""

    def test_contains_all_expected_flavours(self) -> None:
        """Contains all expected flavours."""
        expected = {
            "1a",
            "1b",
            "2a",
            "2b",
            "2u",
            "3a",
            "3b",
            "3u",
            "4",
            "4e",
            "4f",
        }

        assert VALID_FLAVOURS == expected

    def test_is_frozenset(self) -> None:
        """Is an immutable set."""
        assert isinstance(VALID_FLAVOURS, frozenset)


# Integration tests (only when veraPDF is installed)
@pytest.mark.skipif(
    not is_verapdf_available(),
    reason="veraPDF is not installed",
)
class TestVerapdfIntegration:
    """Integration tests with real veraPDF."""

    def test_get_version_returns_string(self) -> None:
        """get_verapdf_version returns a string."""
        version = get_verapdf_version()

        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0

    def test_validate_sample_pdf(self, sample_pdf: Path) -> None:
        """Validates a simple test PDF."""
        # Note: A simple test PDF is probably not
        # PDF/A-compliant, so we expect compliant=False
        result = validate_with_verapdf(sample_pdf)

        assert isinstance(result, VeraPDFResult)
        # The result should be non-compliant (simple test PDF)
        # but it should run without errors
        assert result.raw_xml is not None
