# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for verapdf.py."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdftopdfa.exceptions import VeraPDFError
from pdftopdfa.verapdf import (
    VALID_FLAVOURS,
    VeraPDFResult,
    _extract_flavour_from_profile,
    _get_verapdf_candidates,
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

        with patch.dict("os.environ", {}, clear=True):
            result = is_verapdf_available()

        assert result is True
        mock_which.assert_any_call("verapdf")

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_returns_false_when_not_found(self, mock_which: MagicMock) -> None:
        """Returns False when verapdf is not in PATH."""
        mock_which.return_value = None

        with patch.dict("os.environ", {}, clear=True):
            result = is_verapdf_available()

        assert result is False


class TestGetVerapdfCmd:
    """Tests for _get_verapdf_cmd and VERAPDF_PATH."""

    def test_returns_default_when_env_not_set(self) -> None:
        """Falls back to 'verapdf' when VERAPDF_PATH is not set."""
        with (
            patch("pdftopdfa.verapdf.shutil.which", return_value=None),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert _get_verapdf_cmd() == "verapdf"

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_returns_resolved_path_when_found_in_path(
        self, mock_which: MagicMock, tmp_path: Path
    ) -> None:
        """Returns the launcher path found by shutil.which."""
        launcher = tmp_path / "verapdf.bat"
        launcher.write_text("@echo off\n")
        mock_which.return_value = str(launcher)

        with patch.dict("os.environ", {}, clear=True):
            assert _get_verapdf_cmd() == str(launcher)

        mock_which.assert_called_once_with("verapdf")

    def test_returns_custom_path_when_env_set(self) -> None:
        """Returns custom path from VERAPDF_PATH."""
        with patch.dict("os.environ", {"VERAPDF_PATH": "/opt/verapdf/bin/verapdf"}):
            assert _get_verapdf_cmd() == "/opt/verapdf/bin/verapdf"

    def test_returns_executable_in_dir_when_dir_set(self, tmp_path: Path) -> None:
        """Returns <dir>/verapdf when VERAPDF_PATH points to a directory."""
        with patch.dict("os.environ", {"VERAPDF_PATH": str(tmp_path)}):
            assert _get_verapdf_cmd() == str(tmp_path / "verapdf")

    def test_prefers_windows_launcher_when_present(self, tmp_path: Path) -> None:
        """Returns verapdf.bat from a directory on Windows."""
        launcher = tmp_path / "verapdf.bat"
        launcher.write_text("@echo off\n")

        with (
            patch(
                "pdftopdfa.verapdf._get_verapdf_candidates",
                return_value=_get_verapdf_candidates("nt"),
            ),
            patch.dict("os.environ", {"VERAPDF_PATH": str(tmp_path)}),
        ):
            assert _get_verapdf_cmd() == str(launcher)

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_is_available_uses_custom_path(self, mock_which: MagicMock) -> None:
        """is_verapdf_available uses VERAPDF_PATH."""
        mock_which.return_value = "/opt/verapdf/bin/verapdf"
        with patch.dict("os.environ", {"VERAPDF_PATH": "/opt/verapdf/bin/verapdf"}):
            result = is_verapdf_available()

        assert result is True
        mock_which.assert_any_call("/opt/verapdf/bin/verapdf")

    @patch("pdftopdfa.verapdf.shutil.which")
    def test_is_available_accepts_existing_file_path(
        self, mock_which: MagicMock
    ) -> None:
        """Direct executable paths do not require shutil.which."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("pdftopdfa.verapdf._get_verapdf_cmd") as mock_get_cmd:
                mock_get_cmd.return_value = str(Path(__file__))

                result = is_verapdf_available()

        assert result is True
        mock_which.assert_not_called()

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

    @patch("pdftopdfa.verapdf.shutil.which")
    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_validate_uses_resolved_batch_launcher_from_path(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        tmp_path: Path,
    ) -> None:
        """validate_with_verapdf runs the resolved .bat launcher from PATH."""
        mock_available.return_value = True
        launcher = tmp_path / "verapdf.bat"
        launcher.write_text("@echo off\n")
        mock_which.return_value = str(launcher)
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="1" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch.dict("os.environ", {}, clear=True):
            validate_with_verapdf(pdf_path, flavour="2b")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == str(launcher)
        mock_which.assert_called_once_with("verapdf")


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
        assert mock_run.call_args.kwargs["encoding"] == "utf-8"
        assert mock_run.call_args.kwargs["errors"] == "replace"

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

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_returns_none_on_launch_error(
        self, mock_available: MagicMock, mock_run: MagicMock
    ) -> None:
        """Returns None when the version subprocess cannot launch."""
        mock_available.return_value = True
        mock_run.side_effect = OSError("launch failed")

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

    def test_rejects_invalid_xml(self) -> None:
        """Rejects truncated or otherwise invalid XML."""
        xml = "not valid xml <<<"

        with pytest.raises(VeraPDFError, match="Invalid veraPDF XML report"):
            _parse_verapdf_xml(xml)

    def test_rejects_missing_validation_report(self) -> None:
        """Rejects XML without the expected validationReport element."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <report>
            <jobs>
                <job>
                </job>
            </jobs>
        </report>"""

        with pytest.raises(VeraPDFError, match="found 0"):
            _parse_verapdf_xml(xml)

    def test_rejects_multiple_validation_reports(self) -> None:
        """Rejects ambiguous output for the single requested input."""
        xml = """
        <report>
            <validationReport isCompliant="true">
                <details passedRules="1" failedRules="0"/>
            </validationReport>
            <validationReport isCompliant="false">
                <details passedRules="0" failedRules="1"/>
            </validationReport>
        </report>
        """

        with pytest.raises(VeraPDFError, match="found 2"):
            _parse_verapdf_xml(xml)

    @pytest.mark.parametrize(
        ("validation_report", "error_match"),
        [
            (
                '<validationReport><details passedRules="1" failedRules="0"/>'
                "</validationReport>",
                "isCompliant",
            ),
            (
                '<validationReport isCompliant="unknown">'
                '<details passedRules="1" failedRules="0"/>'
                "</validationReport>",
                "isCompliant",
            ),
            (
                '<validationReport isCompliant="true"></validationReport>',
                "details element",
            ),
            (
                '<validationReport isCompliant="true">'
                '<details passedRules="invalid" failedRules="0"/>'
                "</validationReport>",
                "passedRules",
            ),
        ],
    )
    def test_rejects_broken_validation_report(
        self, validation_report: str, error_match: str
    ) -> None:
        """Rejects incomplete or malformed validationReport content."""
        with pytest.raises(VeraPDFError, match=error_match):
            _parse_verapdf_xml(f"<report>{validation_report}</report>")

    @pytest.mark.parametrize(
        ("is_compliant", "failed_rules"),
        [("true", 1), ("false", 0)],
    )
    def test_rejects_compliance_and_failed_rule_contradictions(
        self, is_compliant: str, failed_rules: int
    ) -> None:
        """Rejects contradictory status and rule counters."""
        xml = (
            "<report>"
            f'<validationReport isCompliant="{is_compliant}">'
            f'<details passedRules="1" failedRules="{failed_rules}"/>'
            "</validationReport>"
            "</report>"
        )

        with pytest.raises(VeraPDFError, match="Inconsistent veraPDF"):
            _parse_verapdf_xml(xml)

    def test_parses_namespaced_report(self) -> None:
        """Parses report, details, and failed rules independent of namespaces."""
        xml = """
        <v:report xmlns:v="urn:verapdf:report">
            <v:jobs>
                <v:job>
                    <v:validationReport
                        isCompliant="false"
                        profileName="PDF/A-3A validation profile">
                        <v:details passedRules="120" failedRules="1">
                            <v:rule status="failed" clause="6.2.2">
                                <v:description>Missing output intent</v:description>
                            </v:rule>
                        </v:details>
                    </v:validationReport>
                </v:job>
            </v:jobs>
        </v:report>
        """

        result = _parse_verapdf_xml(xml)

        assert result.compliant is False
        assert result.flavour == "3a"
        assert result.passed_rules == 120
        assert result.failed_rules == 1
        assert result.errors == ["Rule 6.2.2: Missing output intent"]

    @pytest.mark.parametrize(
        "task_result",
        [
            '<taskResult exceptionMessage="parser crashed"/>',
            "<taskResult><exceptionMessage>parser crashed</exceptionMessage>"
            "</taskResult>",
            "<taskException><exceptionMessage>parser crashed</exceptionMessage>"
            "</taskException>",
        ],
    )
    def test_rejects_task_exceptions(self, task_result: str) -> None:
        """Treats veraPDF task exceptions as execution failures."""
        xml = f"""
        <report>
            <validationReport isCompliant="true">
                <details passedRules="1" failedRules="0"/>
            </validationReport>
            {task_result}
        </report>
        """

        with pytest.raises(VeraPDFError, match="parser crashed"):
            _parse_verapdf_xml(xml)

    @pytest.mark.parametrize(
        "attribute",
        ["failedToParse", "encrypted", "outOfMemory", "veraExceptions"],
    )
    def test_rejects_batch_failures(self, attribute: str) -> None:
        """Treats every veraPDF batch failure counter as an execution failure."""
        xml = f"""
        <report>
            <validationReport isCompliant="true">
                <details passedRules="1" failedRules="0"/>
            </validationReport>
            <batchSummary {attribute}="1"/>
        </report>
        """

        with pytest.raises(VeraPDFError, match=attribute):
            _parse_verapdf_xml(xml)

    @pytest.mark.parametrize("value", ["invalid", "-1"])
    def test_rejects_invalid_batch_counters(self, value: str) -> None:
        """Rejects malformed batch counters instead of ignoring them."""
        xml = f"""
        <report>
            <validationReport isCompliant="true">
                <details passedRules="1" failedRules="0"/>
            </validationReport>
            <batchSummary failedToParse="{value}"/>
        </report>
        """

        with pytest.raises(VeraPDFError, match="failedToParse"):
            _parse_verapdf_xml(xml)

    def test_rejects_failed_batch_jobs(self) -> None:
        """Treats failed validation jobs in the batch summary as failures."""
        xml = """
        <report>
            <validationReport isCompliant="true">
                <details passedRules="1" failedRules="0"/>
            </validationReport>
            <batchSummary>
                <validationReports failedJobs="1"/>
            </batchSummary>
        </report>
        """

        with pytest.raises(VeraPDFError, match="failedJobs=1"):
            _parse_verapdf_xml(xml)

    def test_preserves_raw_xml(self) -> None:
        """Stores the raw XML in the result."""
        xml = (
            "<report>"
            '<validationReport isCompliant="true">'
            '<details passedRules="1" failedRules="0"/>'
            "</validationReport>"
            "</report>"
        )

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
        assert Path(cmd[0]).name.lower().startswith("verapdf")
        assert "--format" in cmd
        assert "xml" in cmd
        assert "--flavour" in cmd
        assert "2b" in cmd
        assert str(pdf_path) in cmd
        assert call_args.kwargs["encoding"] == "utf-8"
        assert call_args.kwargs["errors"] == "replace"

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available", return_value=True)
    def test_decodes_utf8_report_for_unicode_path(
        self,
        _mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Windows locale encoding cannot corrupt a UTF-8 veraPDF report."""
        pdf_path = tmp_path / "卍.pdf"
        pdf_path.touch()
        xml_bytes = (
            "<report>"
            f'<item name="{pdf_path.name}"/>'
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="1" failedRules="0"/>'
            "</validationReport>"
            "</report>"
        ).encode()

        def encoded_result(*_args: object, **kwargs: object) -> MagicMock:
            encoding = kwargs["encoding"]
            errors = kwargs["errors"]
            return MagicMock(
                stdout=xml_bytes.decode(encoding, errors),
                stderr="",
                returncode=0,
            )

        mock_run.side_effect = encoded_result

        result = validate_with_verapdf(pdf_path, flavour="2b")

        assert result.compliant is True
        assert pdf_path.name in result.raw_xml

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_sets_java_stack_only_in_child_environment(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Adds the deep-document stack default without mutating os.environ."""
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

        with patch.dict(os.environ, {"JAVA_OPTS": "-Xmx512m"}, clear=True):
            validate_with_verapdf(pdf_path, flavour="2b")

            assert os.environ["JAVA_OPTS"] == "-Xmx512m"

        child_env = mock_run.call_args.kwargs["env"]
        assert child_env["JAVA_OPTS"] == "-Xmx512m -Xss16m"

    @pytest.mark.parametrize(
        "variable",
        ["JAVA_OPTS", "JDK_JAVA_OPTIONS", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS"],
    )
    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_preserves_explicit_java_stack_option(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        variable: str,
    ) -> None:
        """Keeps an explicit user-provided Java stack size unchanged."""
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
        java_options = "-Xmx512m -Xss8m"

        with patch.dict(os.environ, {variable: java_options}, clear=True):
            validate_with_verapdf(pdf_path, flavour="2b")

        child_env = mock_run.call_args.kwargs["env"]
        assert child_env[variable] == java_options
        assert "-Xss16m" not in child_env.get("JAVA_OPTS", "")

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
    def test_raises_clear_error_when_executable_missing(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError when the resolved executable cannot be started."""
        mock_available.return_value = True
        mock_run.side_effect = FileNotFoundError("missing executable")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="executable not found"):
            validate_with_verapdf(pdf_path)

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_clear_error_when_process_launch_fails(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Raises VeraPDFError when subprocess launch fails."""
        mock_available.return_value = True
        mock_run.side_effect = OSError("launch failed")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="Error launching veraPDF"):
            validate_with_verapdf(pdf_path)

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
    def test_rejects_reported_flavour_mismatch(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Rejects a report for a different flavour than requested."""
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

        with pytest.raises(VeraPDFError, match="requested 2a, reported 2b"):
            validate_with_verapdf(pdf_path, flavour="2a")

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_rejects_missing_reported_flavour(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Rejects a report without a flavour when one was requested."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true">'
            '<details passedRules="100" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(stdout=xml_response, stderr="", returncode=0)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="requested 2a, reported unknown"):
            validate_with_verapdf(pdf_path, flavour="2a")

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_accepts_stderr_with_complete_report(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Does not confuse veraPDF log output with a validation failure."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            '<validationReport isCompliant="true" profileName="PDF/A-2B">'
            '<details passedRules="100" failedRules="0"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(
            stdout=xml_response,
            stderr="WARN: optional font cache was rebuilt",
            returncode=0,
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validate_with_verapdf(pdf_path, flavour="2b")

        assert result.compliant is True

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_raises_on_incomplete_xml_report(
        self, mock_available: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Propagates truncated XML as an execution failure."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            stdout="<report><jobs",
            stderr="java.lang.StackOverflowError",
            returncode=1,
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="Invalid veraPDF XML report"):
            validate_with_verapdf(pdf_path, flavour="2b")

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

    @pytest.mark.parametrize(
        ("returncode", "is_compliant", "failed_rules"),
        [(1, "true", 0), (0, "false", 1)],
    )
    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_rejects_exit_code_and_report_contradictions(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        returncode: int,
        is_compliant: str,
        failed_rules: int,
    ) -> None:
        """Rejects a report whose status contradicts the CLI exit code."""
        mock_available.return_value = True
        xml_response = (
            "<report><jobs><job>"
            f'<validationReport isCompliant="{is_compliant}" profileName="PDF/A-2B">'
            f'<details passedRules="100" failedRules="{failed_rules}"></details>'
            "</validationReport></job></jobs></report>"
        )
        mock_run.return_value = MagicMock(
            stdout=xml_response,
            stderr="",
            returncode=returncode,
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(VeraPDFError, match="Inconsistent veraPDF result"):
            validate_with_verapdf(pdf_path, flavour="2b")

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_logs_compliant_result_as_info(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Compliant validation results are logged at INFO level."""
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

        with caplog.at_level(logging.INFO, logger="pdftopdfa.verapdf"):
            validate_with_verapdf(pdf_path, flavour="2b")

        assert any(
            record.levelno == logging.INFO
            and "veraPDF validation: compliant" in record.message
            for record in caplog.records
        )

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_logs_non_compliant_result_as_error(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-compliant validation results are logged at ERROR level."""
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

        with caplog.at_level(logging.ERROR, logger="pdftopdfa.verapdf"):
            validate_with_verapdf(pdf_path, flavour="2b")

        assert any(
            record.levelno == logging.ERROR
            and "veraPDF validation: non-compliant" in record.message
            for record in caplog.records
        )

    @patch("pdftopdfa.verapdf.subprocess.run")
    @patch("pdftopdfa.verapdf.is_verapdf_available")
    def test_logs_non_compliant_result_as_warning_when_requested(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-compliant results can be logged at WARNING level."""
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

        with caplog.at_level(logging.WARNING, logger="pdftopdfa.verapdf"):
            validate_with_verapdf(
                pdf_path,
                flavour="2b",
                non_compliant_log_level=logging.WARNING,
            )

        assert any(
            record.levelno == logging.WARNING
            and "veraPDF validation: non-compliant" in record.message
            for record in caplog.records
        )


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
        if version is None:
            pytest.skip("veraPDF is present but could not be started")

        assert isinstance(version, str)
        assert len(version) > 0

    def test_validate_sample_pdf(self, sample_pdf: Path) -> None:
        """Validates a simple test PDF."""
        # Note: A simple test PDF is probably not
        # PDF/A-compliant, so we expect compliant=False
        try:
            result = validate_with_verapdf(sample_pdf)
        except VeraPDFError as exc:
            pytest.skip(f"veraPDF is present but could not be started: {exc}")

        assert isinstance(result, VeraPDFResult)
        # The result should be non-compliant (simple test PDF)
        # but it should run without errors
        assert result.raw_xml is not None
