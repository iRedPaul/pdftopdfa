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

_DEFAULT_JAVA_STACK_OPTION = "-Xss16m"
_JAVA_OPTION_ENV_VARS = (
    "JAVA_OPTS",
    "JDK_JAVA_OPTIONS",
    "JAVA_TOOL_OPTIONS",
    "_JAVA_OPTIONS",
)
_JAVA_STACK_OPTION_RE = re.compile(r"(?<!\S)-Xss(?:=)?\S+")
_BATCH_FAILURE_ATTRIBUTES = (
    "failedToParse",
    "encrypted",
    "outOfMemory",
    "veraExceptions",
)


def _get_verapdf_candidates(platform_name: str | None = None) -> tuple[str, ...]:
    """Returns candidate launcher names for the current platform."""
    if platform_name is None:
        platform_name = os.name

    if platform_name == "nt":
        return ("verapdf.bat", "verapdf.cmd", "verapdf.exe", "verapdf")

    return ("verapdf", "verapdf.sh", "verapdf.bat", "verapdf.cmd")


def _get_verapdf_cmd() -> str:
    """Returns the veraPDF command from VERAPDF_PATH or falls back to 'verapdf'."""
    verapdf_path = os.environ.get("VERAPDF_PATH")
    if not verapdf_path:
        cmd = "verapdf"
    else:
        p = Path(verapdf_path)
        if p.is_dir():
            for candidate in _get_verapdf_candidates():
                candidate_path = p / candidate
                if candidate_path.is_file():
                    cmd = str(candidate_path)
                    break
            else:
                cmd = str(p / "verapdf")
        else:
            cmd = verapdf_path

    return shutil.which(cmd) or cmd


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
    cmd = _get_verapdf_cmd()
    if Path(cmd).is_file():
        return True
    return shutil.which(cmd) is not None


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
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        # veraPDF outputs version on stdout
        output = result.stdout.strip()
        if output:
            # Typical output: "veraPDF 1.24.1"
            return output
        return None
    except FileNotFoundError as e:
        logger.debug("veraPDF executable not found: %s", e)
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
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

    Raises:
        VeraPDFError: If the XML is incomplete or reports a veraPDF failure.
    """
    result = VeraPDFResult(compliant=False, raw_xml=xml_string)

    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(xml_string.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as e:
        raise VeraPDFError(f"Invalid veraPDF XML report: {e}") from e

    task_errors: list[str] = []
    for task_result in root.xpath(
        ".//*[local-name()='taskResult' or local-name()='taskException']"
    ):
        exception_message = task_result.get("exceptionMessage")
        exception_elements = task_result.xpath("./*[local-name()='exceptionMessage']")
        if exception_message is None and exception_elements:
            exception_message = " ".join(exception_elements[0].itertext())
        if exception_message is not None:
            task_errors.append(exception_message.strip() or "unknown task error")
    if task_errors:
        raise VeraPDFError(f"veraPDF task failed: {'; '.join(task_errors)}")

    batch_errors: list[str] = []
    for batch_summary in root.xpath(".//*[local-name()='batchSummary']"):
        for attribute in _BATCH_FAILURE_ATTRIBUTES:
            value = batch_summary.get(attribute)
            if value is None:
                continue
            try:
                count = int(value)
            except ValueError as e:
                raise VeraPDFError(
                    f"Invalid veraPDF batch counter {attribute}={value!r}"
                ) from e
            if count < 0:
                raise VeraPDFError(
                    f"Invalid veraPDF batch counter {attribute}={value!r}"
                )
            if count:
                batch_errors.append(f"{attribute}={count}")

        for report_summary in batch_summary.xpath(".//*[@failedJobs]"):
            value = report_summary.get("failedJobs")
            try:
                count = int(value)
            except (TypeError, ValueError) as e:
                raise VeraPDFError(
                    f"Invalid veraPDF batch counter failedJobs={value!r}"
                ) from e
            if count < 0:
                raise VeraPDFError(
                    f"Invalid veraPDF batch counter failedJobs={value!r}"
                )
            if count:
                batch_errors.append(f"failedJobs={count}")
    if batch_errors:
        raise VeraPDFError(f"veraPDF batch failed: {', '.join(batch_errors)}")

    validation_reports = root.xpath(
        "descendant-or-self::*[local-name()='validationReport']"
    )
    if len(validation_reports) != 1:
        raise VeraPDFError(
            "Expected exactly one validationReport in veraPDF XML, "
            f"found {len(validation_reports)}"
        )
    validation_report = validation_reports[0]

    # Extract compliance status
    compliance_status = validation_report.get("isCompliant", "").strip().lower()
    if compliance_status not in {"true", "false"}:
        raise VeraPDFError(
            "Invalid or missing isCompliant attribute in veraPDF validationReport"
        )
    result.compliant = compliance_status == "true"

    # Extract flavour (profileName contains e.g. "PDF/A-2B validation profile")
    profile_name = validation_report.get("profileName", "")
    if profile_name:
        result.flavour = _extract_flavour_from_profile(profile_name)

    # Count passed/failed rules
    details_elements = validation_report.xpath("./*[local-name()='details']")
    if len(details_elements) != 1:
        raise VeraPDFError(
            "Expected exactly one details element in veraPDF validationReport, "
            f"found {len(details_elements)}"
        )
    details = details_elements[0]
    for attribute in ("passedRules", "failedRules"):
        value = details.get(attribute)
        try:
            count = int(value)
        except (TypeError, ValueError) as e:
            raise VeraPDFError(
                f"Invalid or missing {attribute} in veraPDF validationReport"
            ) from e
        if count < 0:
            raise VeraPDFError(
                f"Invalid or missing {attribute} in veraPDF validationReport"
            )
        if attribute == "passedRules":
            result.passed_rules = count
        else:
            result.failed_rules = count

    expected_compliance = result.failed_rules == 0
    if result.compliant != expected_compliance:
        raise VeraPDFError(
            "Inconsistent veraPDF validationReport: "
            f"isCompliant={compliance_status}, failedRules={result.failed_rules}"
        )

    # Extract error messages from failed rules
    for rule in details.xpath(".//*[local-name()='rule']"):
        if rule.get("status", "").lower() != "failed":
            continue
        clause = rule.get("clause", "")
        description_elements = rule.xpath("./*[local-name()='description']")
        description = (
            " ".join(description_elements[0].itertext()).strip()
            if description_elements
            else ""
        )

        error_msg = f"Rule {clause}: {description}" if clause else description
        if error_msg:
            result.errors.append(error_msg)

    return result


def _get_verapdf_subprocess_env() -> dict[str, str]:
    """Builds an isolated environment suitable for deeply nested PDFs.

    veraPDF's default Java thread stack can be exhausted by valid, deeply nested
    resource graphs. An explicit user stack size in any standard Java option
    variable takes precedence; otherwise only the child process receives 16 MiB.
    """
    env = os.environ.copy()
    if any(
        _JAVA_STACK_OPTION_RE.search(env.get(variable, ""))
        for variable in _JAVA_OPTION_ENV_VARS
    ):
        return env

    java_options = env.get("JAVA_OPTS", "").strip()
    env["JAVA_OPTS"] = f"{java_options} {_DEFAULT_JAVA_STACK_OPTION}".strip()
    return env


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
    non_compliant_log_level: int = logging.ERROR,
) -> VeraPDFResult:
    """Validates a PDF file with veraPDF.

    Args:
        path: Path to the PDF file to validate.
        flavour: Optional PDF/A flavour for validation (e.g. "2b").
            If not specified, veraPDF detects automatically.
        timeout: Timeout in seconds (default: 300).
        non_compliant_log_level: Log level used when veraPDF reports a
            non-compliant result. Defaults to ``logging.ERROR``.

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
            "veraPDF executable or its parent directory."
        )

    if not path.exists():
        raise VeraPDFError(f"File not found: {path}")

    # Build command
    cmd = [_get_verapdf_cmd(), "--format", "xml"]

    normalized_flavour = _normalize_flavour(flavour) if flavour else None
    if normalized_flavour:
        cmd.extend(["--flavour", normalized_flavour])

    cmd.append(str(path))

    logger.debug("Running veraPDF: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_get_verapdf_subprocess_env(),
        )
    except FileNotFoundError as e:
        raise VeraPDFError(
            "veraPDF executable not found. "
            "Set VERAPDF_PATH to the executable itself "
            "or to the installation directory containing it."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise VeraPDFError(f"veraPDF timeout after {timeout} seconds.") from e
    except OSError as e:
        raise VeraPDFError(f"Error launching veraPDF: {e}") from e
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

    # Parse XML result. Incomplete reports and veraPDF execution failures raise
    # VeraPDFError instead of being misclassified as ordinary non-compliance.
    verapdf_result = _parse_verapdf_xml(xml_output)
    expected_returncode = 0 if verapdf_result.compliant else 1
    if result.returncode != expected_returncode:
        raise VeraPDFError(
            "Inconsistent veraPDF result: "
            f"exit code {result.returncode} reports "
            f"isCompliant={str(verapdf_result.compliant).lower()}"
        )
    if normalized_flavour and verapdf_result.flavour != normalized_flavour:
        reported_flavour = verapdf_result.flavour or "unknown"
        raise VeraPDFError(
            "veraPDF validated an unexpected flavour: "
            f"requested {normalized_flavour}, reported {reported_flavour}"
        )

    log_level = logging.INFO if verapdf_result.compliant else non_compliant_log_level
    logger.log(
        log_level,
        "veraPDF validation: %s (flavour: %s, %d/%d rules passed)",
        "compliant" if verapdf_result.compliant else "non-compliant",
        verapdf_result.flavour or "unknown",
        verapdf_result.passed_rules,
        verapdf_result.passed_rules + verapdf_result.failed_rules,
    )

    return verapdf_result
