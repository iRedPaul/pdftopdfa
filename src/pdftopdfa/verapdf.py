# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""veraPDF integration for pdftopdfa.

This module provides functions for ISO-compliant PDF/A validation
using veraPDF. veraPDF is a Java-based CLI tool that must be
installed externally: https://verapdf.org/

Example:
    >>> from pdftopdfa.verapdf import is_verapdf_available, validate_with_verapdf
    >>> from pathlib import Path
    >>> if is_verapdf_available():
    ...     result = validate_with_verapdf(Path("document.pdf"), flavour="2b")
    ...     print(f"Compliant: {result.compliant}")
"""

# Standard Library
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

# Local
from .exceptions import VeraPDFError

logger = logging.getLogger(__name__)

# Valid PDF/A flavours
VALID_FLAVOURS = frozenset(
    {
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
)


def _get_verapdf_cmd() -> str:
    """Returns the veraPDF command from VERAPDF_PATH or falls back to 'verapdf'."""
    return os.environ.get("VERAPDF_PATH", "verapdf")


@dataclass
class VeraPDFResult:
    """Result of veraPDF validation.

    Attributes:
        compliant: True if the PDF conforms to the specified flavour.
        flavour: Detected/validated PDF/A flavour (e.g. "2b").
        passed_rules: Number of passed rules.
        failed_rules: Number of failed rules.
        errors: List of critical errors.
        warnings: List of warnings.
        raw_xml: The raw XML result from veraPDF.
    """

    compliant: bool
    flavour: str | None = None
    passed_rules: int = 0
    failed_rules: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_xml: str | None = None


def is_verapdf_available() -> bool:
    """Checks if veraPDF is available in PATH.

    Returns:
        True if verapdf is found and executable.
    """
    return shutil.which(_get_verapdf_cmd()) is not None


def get_verapdf_version() -> str | None:
    """Gets the installed veraPDF version.

    Returns:
        Version string or None if veraPDF is not available.
    """
    if not is_verapdf_available():
        return None

    try:
        result = subprocess.run(
            [_get_verapdf_cmd(), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # veraPDF outputs version on stdout
        output = result.stdout.strip()
        if output:
            # Typical output: "veraPDF 1.24.1"
            return output
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.debug("Error getting veraPDF version: %s", e)
        return None


def _normalize_flavour(flavour: str) -> str:
    """Normalizes a PDF/A flavour for veraPDF.

    Converts various notations to the veraPDF format.

    Args:
        flavour: The flavour to normalize (e.g. "2b", "PDFA_2_B", "PDF/A-2B").

    Returns:
        Normalized flavour (e.g. "2b").

    Raises:
        VeraPDFError: If the flavour is invalid.
    """
    # Remove prefixes and normalize
    normalized = flavour.upper()

    # Remove common prefixes
    for prefix in ("PDF/A-", "PDFA-", "PDFA_", "PDF/A", "PDFA"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    # Remove underscores and hyphens
    normalized = normalized.replace("_", "").replace("-", "")

    # Convert to lowercase
    normalized = normalized.lower()

    # Validate
    if normalized not in VALID_FLAVOURS:
        valid_list = ", ".join(sorted(VALID_FLAVOURS))
        raise VeraPDFError(
            f"Invalid PDF/A flavour: '{flavour}'. Valid values: {valid_list}"
        )

    return normalized


def _parse_verapdf_xml(xml_string: str) -> VeraPDFResult:
    """Parses the XML result from veraPDF.

    Args:
        xml_string: The raw XML from veraPDF.

    Returns:
        VeraPDFResult with the extracted information.
    """
    result = VeraPDFResult(compliant=False, raw_xml=xml_string)

    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(xml_string.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as e:
        logger.warning("Error parsing veraPDF XML: %s", e)
        result.errors.append(f"XML parsing error: {e}")
        return result

    # Search for validationReport
    # veraPDF XML structure: <report><jobs><job><validationReport>...
    validation_report = root.find(".//validationReport")

    if validation_report is None:
        # Alternative path for some veraPDF versions
        validation_report = root.find(
            ".//batchSummary/validationReports/validationReport"
        )

    if validation_report is None:
        logger.warning("No validationReport found in veraPDF XML")
        result.warnings.append("No validation report found in veraPDF result")
        return result

    # Extract compliance status
    is_compliant = validation_report.get("isCompliant", "false").lower() == "true"
    result.compliant = is_compliant

    # Extract flavour (profileName contains e.g. "PDF/A-2B validation profile")
    profile_name = validation_report.get("profileName", "")
    if profile_name:
        result.flavour = _extract_flavour_from_profile(profile_name)

    # Count passed/failed rules
    details = validation_report.find("details")
    if details is not None:
        passed_rules = details.get("passedRules", "0")
        failed_rules = details.get("failedRules", "0")
        try:
            result.passed_rules = int(passed_rules)
            result.failed_rules = int(failed_rules)
        except ValueError:
            pass

        # Extract error messages from failed rules
        for rule in details.findall(".//rule[@status='failed']"):
            clause = rule.get("clause", "")
            description_elem = rule.find("description")
            description = description_elem.text if description_elem is not None else ""

            error_msg = f"Rule {clause}: {description}" if clause else description
            if error_msg:
                result.errors.append(error_msg)

    # Search for taskResult for additional errors
    task_result = root.find(".//taskResult")
    if task_result is not None:
        exception_msg = task_result.get("exceptionMessage")
        if exception_msg:
            result.errors.append(f"veraPDF error: {exception_msg}")

    return result


def _extract_flavour_from_profile(profile_name: str) -> str | None:
    """Extracts the flavour from a veraPDF profile name.

    Args:
        profile_name: Profile name like "PDF/A-2B validation profile".

    Returns:
        Flavour like "2b" or None.
    """
    # Typical formats: "PDF/A-2B validation profile", "PDF/A-1A", "PDF/A-4"
    match = re.search(r"PDF/A-(\d)([ABUEFabuef])?", profile_name, re.IGNORECASE)
    if match:
        part = match.group(1)
        conformance = match.group(2)
        if conformance:
            return f"{part}{conformance.lower()}"
        return part

    return None


def validate_with_verapdf(
    path: Path,
    flavour: str | None = None,
    timeout: int = 300,
) -> VeraPDFResult:
    """Validates a PDF file with veraPDF.

    Args:
        path: Path to the PDF file to validate.
        flavour: Optional PDF/A flavour for validation (e.g. "2b").
            If not specified, veraPDF detects automatically.
        timeout: Timeout in seconds (default: 300).

    Returns:
        VeraPDFResult with the validation result.

    Raises:
        VeraPDFError: If veraPDF is not available or an error occurs.
    """
    if not is_verapdf_available():
        raise VeraPDFError(
            "veraPDF is not installed or not in PATH. "
            "Installation: https://verapdf.org/ — "
            "or set the VERAPDF_PATH environment variable to the "
            "veraPDF executable."
        )

    if not path.exists():
        raise VeraPDFError(f"File not found: {path}")

    # Build command
    cmd = [_get_verapdf_cmd(), "--format", "xml"]

    if flavour:
        normalized_flavour = _normalize_flavour(flavour)
        cmd.extend(["--flavour", normalized_flavour])

    cmd.append(str(path))

    logger.debug("Running veraPDF: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise VeraPDFError(f"veraPDF timeout after {timeout} seconds.") from e
    except subprocess.SubprocessError as e:
        raise VeraPDFError(f"Error running veraPDF: {e}") from e

    # Exit code 0 = compliant, 1 = non-compliant (both are valid results).
    # Any other code means veraPDF itself failed.
    if result.returncode not in (0, 1):
        stderr_msg = result.stderr.strip() if result.stderr else "unknown error"
        raise VeraPDFError(
            f"veraPDF failed with exit code {result.returncode}: {stderr_msg}"
        )

    # veraPDF outputs XML to stdout
    xml_output = result.stdout

    if not xml_output.strip():
        # Check stderr for error messages
        if result.stderr:
            raise VeraPDFError(f"veraPDF error: {result.stderr.strip()}")
        raise VeraPDFError("veraPDF returned no output")

    # Parse XML result
    try:
        verapdf_result = _parse_verapdf_xml(xml_output)
    except Exception as e:
        logger.warning("Error parsing veraPDF result: %s", e)
        # Try to extract at least the basic status
        verapdf_result = VeraPDFResult(
            compliant=False,
            raw_xml=xml_output,
            errors=[f"XML parsing failed: {e}"],
        )

    logger.info(
        "veraPDF validation: %s (flavour: %s, %d/%d rules passed)",
        "compliant" if verapdf_result.compliant else "non-compliant",
        verapdf_result.flavour or "unknown",
        verapdf_result.passed_rules,
        verapdf_result.passed_rules + verapdf_result.failed_rules,
    )

    return verapdf_result
