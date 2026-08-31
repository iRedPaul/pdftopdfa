# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core logic for PDF to PDF/A conversion."""

# Standard Library
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

# Third Party
import pikepdf
from tqdm import tqdm

# Local
from ._ocr_runtime import (
    execution_provider_base,
    onnxruntime_engine_config,
    validate_ocr_execution_provider,
)
from .color_profile import embed_color_profiles
from .exceptions import (
    ConversionError,
    FontEmbeddingError,
    OCRError,
    UnsupportedPDFError,
    VeraPDFError,
)
from .extensions import add_extensions_if_needed
from .fonts import check_font_compliance
from .metadata import (
    NAMESPACES,
    _clean_metadata_text,
    _extract_existing_xmp,
    _extract_lang_alt_xmp_property,
    extract_pdf_info,
    remove_pdfe_identification,
    remove_pdfua_identification,
    remove_pdfvt_identification,
    remove_pdfx_identification,
    sync_metadata,
)
from .ocr import validate_ocr_languages
from .sanitizers import (
    count_digital_signatures,
    ensure_display_doc_title,
    sanitize_for_pdfa,
    sanitize_notdef_usage,
    sanitize_signatures,
    sanitize_structure_limits,
    sanitize_truetype_encoding,
)
from .sanitizers.catalog import _is_valid_bcp47
from .staging import (
    copy_to_private_stage,
    private_staging_directory,
    publish_staged_file,
    staged_file_snapshot,
)
from .tagging import ensure_logical_structure
from .utils import (
    get_required_pdf_version,
    is_pdf_encrypted,
    validate_pdfa_level,
)
from .validator import detect_iso_standards, detect_pdfa_level
from .verapdf import VeraPDFResult, validate_with_verapdf

logger = logging.getLogger(__name__)


class _AnnotationRestoreStatus(Enum):
    """Outcome of restoring annotations after OCR."""

    NO_ANNOTATIONS = "no_annotations"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class _AnnotationRestoreResult:
    """Detailed result of an annotation restoration attempt."""

    status: _AnnotationRestoreStatus
    count: int = 0
    error: str | None = None


# Conformance level ranking: a > u > b
_CONFORMANCE_RANK = {"b": 0, "u": 1, "a": 2}


def _validate_ocr_configuration(
    *,
    ocr_languages: list[str] | None,
    ocr_detection_model_dir: Path | None,
    ocr_recognition_model_dir: Path | None,
    ocr_force: bool,
    ocr_deskew: bool,
    ocr_rotate_pages: bool,
    ocr_execution_provider: str,
    ocr_layout: bool,
) -> tuple[bool, list[str]]:
    """Validate OCR activation and return its effective language list."""
    ocr_execution_provider = validate_ocr_execution_provider(ocr_execution_provider)
    if (ocr_detection_model_dir is None) != (ocr_recognition_model_dir is None):
        raise ValueError(
            "OCR detection and recognition model directories must be provided together"
        )

    ocr_requested = (
        ocr_languages is not None
        or ocr_detection_model_dir is not None
        or ocr_force
        or ocr_deskew
        or ocr_rotate_pages
        or ocr_execution_provider != "cpu"
        or ocr_layout
    )
    if ocr_requested and ocr_detection_model_dir is None:
        raise ValueError(
            "OCR requires both detection and recognition model directories"
        )

    effective_languages = ocr_languages if ocr_languages is not None else ["en"]
    if ocr_requested:
        validate_ocr_languages(effective_languages)
        onnxruntime_engine_config(ocr_execution_provider)
    return ocr_requested, effective_languages


# Sanitization result key -> warning message mappings for convert_to_pdfa().
_SANITIZE_WARNINGS: list[tuple[str, str]] = [
    (
        "non_compliant_annotations_flattened",
        "non-compliant annotation(s) flattened into page content",
    ),
    (
        "forbidden_annotations_removed",
        "forbidden annotation(s) removed (rule 6.3.1)",
    ),
    ("notdef_usage_fixed", ".notdef usage operator(s) fixed"),
    (
        "proprietary_stamps_normalized",
        "proprietary stamp annotation(s) converted to standard PDF Stamp annotation(s)",
    ),
    ("javascript_removed", "JavaScript element(s) removed"),
    ("actions_removed", "non-compliant action(s) removed"),
    ("files_removed", "embedded file(s) removed"),
    ("embedded_files_kept", "conformant embedded file(s) kept"),
    (
        "embedded_pdf_conversions_failed",
        "embedded PDF attachment(s) could not be converted; output is not "
        "PDF/A compliant",
    ),
    ("af_relationships_fixed", "AFRelationship(s) added to embedded file(s)"),
    ("xfa_removed", "XFA form element(s) removed"),
    ("btn_ap_subdicts_fixed", "Btn widget /AP/N stream(s) wrapped in state dict"),
    ("annotation_flags_fixed", "annotation flag(s) fixed"),
    ("crypt_streams_removed", "Crypt filter(s) removed from stream(s)"),
    ("cidsysteminfo_fixed", "CIDSystemInfo entr(y/ies) fixed"),
    ("cidtogidmap_fixed", "CIDToGIDMap entr(y/ies) added"),
    ("cidset_removed", "CIDSet entr(y/ies) removed"),
    ("type1_charset_removed", "Type1 /CharSet entr(y/ies) removed"),
    (
        "cid_values_over_65535_fixed",
        "CIDFont entr(y/ies) above CID 65535 removed or clipped",
    ),
    (
        "cid_values_over_65535_warned",
        "CIDFont(s) with CID value(s) exceeding 65535 detected (rule 6.1.13-10);"
        " cannot fix automatically",
    ),
    (
        "tt_nonsymbolic_cmap_added",
        "non-symbolic TrueType font program(s) had (3,1) cmap added",
    ),
    (
        "tt_nonsymbolic_encoding_fixed",
        "non-symbolic TrueType font encoding(s) fixed",
    ),
    (
        "tt_symbolic_encoding_removed",
        "symbolic TrueType font /Encoding entr(y/ies) removed",
    ),
    ("tt_symbolic_flag_set", "symbolic TrueType font /Flags Symbolic bit(s) set"),
    (
        "tt_symbolic_cmap_added",
        "symbolic TrueType font program(s) had (3,0) cmap added",
    ),
    ("type1_encoding_added", "Type1 font(s) had /WinAnsiEncoding added"),
    ("boxes_normalized", "page box(es) normalized"),
    ("boxes_clipped", "page box(es) clipped to MediaBox"),
    ("malformed_boxes_removed", "malformed page box(es) removed"),
    ("undefined_operators_removed", "undefined content stream operator(s) removed"),
    ("structure_strings_truncated", "overlong string object(s) truncated"),
    ("structure_names_shortened", "overlong name object(s) shortened"),
    ("structure_utf8_names_fixed", "invalid UTF-8 name object(s) repaired"),
    ("structure_integers_clamped", "out-of-range integer operand(s) clamped"),
    ("structure_reals_normalized", "out-of-range real operand(s) clamped/normalized"),
    ("structure_q_nesting_rebalanced", "q/Q graphics-state operator(s) rebalanced"),
    ("structure_hex_odd_fixed", "odd-length hexadecimal string(s) fixed"),
    ("structure_hex_odd_obj_fixed", "odd-length hexadecimal string object(s) fixed"),
    ("structure_hex_invalid_fixed", "invalid hexadecimal string(s) repaired"),
]

# Sanitization keys that indicate fatal failures.
_SANITIZE_ERRORS: list[tuple[str, str]] = [
    (
        "jbig2_failed",
        "JBIG2 stream(s) with external globals "
        "could not be converted (unsupported filter configuration). "
        "The output PDF would not be PDF/A compliant.",
    ),
    (
        "jpx_failed",
        "JPEG2000 stream(s) could not be fixed. "
        "The output PDF would not be PDF/A compliant.",
    ),
    (
        "pua_actualtext_warnings",
        "PUA-mapped character(s) could not be resolved to real Unicode "
        "without losing searchable text. The output would not satisfy "
        "PDF/A Unicode/accessibility requirements.",
    ),
]

# Groups of keys whose counts are summed into a single warning.
_SANITIZE_COMBINED_WARNINGS: list[tuple[list[str], str]] = [
    (
        ["jpx_fixed", "jpx_wrapped", "jpx_reencoded"],
        "JPEG2000 stream(s) fixed for PDF/A compliance",
    ),
    (
        ["resources_dictionaries_added", "resources_entries_merged"],
        "content stream resource mapping(s) made explicit",
    ),
]

# Late structure-limit pass (runs after color profile embedding).
_LATE_STRUCTURE_WARNINGS: list[tuple[str, str]] = [
    ("strings_truncated", "overlong string object(s) truncated"),
    ("names_shortened", "overlong name object(s) shortened"),
    ("utf8_names_fixed", "invalid UTF-8 name object(s) repaired"),
    ("integers_clamped", "out-of-range integer operand(s) clamped"),
    ("reals_normalized", "out-of-range real operand(s) clamped or normalized"),
    ("q_nesting_rebalanced", "q/Q graphics-state operator(s) rebalanced"),
    ("hex_odd_fixed", "odd-length hexadecimal string(s) fixed"),
    ("hex_odd_obj_fixed", "odd-length hexadecimal string object(s) fixed"),
    ("hex_invalid_fixed", "invalid hexadecimal string(s) repaired"),
]

_SIGNATURE_SKIP_WARNING = (
    "Conversion skipped: PDF contains digital signatures; conversion would "
    "invalidate them"
)
_VALIDATION_PUBLICATION_WARNING = (
    "Output was published despite failed or incomplete validation"
)
_VALIDATION_FAILURE_ERROR = "Validation failed; output candidate was published"
_VALIDATION_WITHHELD_WARNING = (
    "Output was not published because validation failed or could not complete"
)
_VALIDATION_WITHHELD_ERROR = "Validation failed; output was not published"


class PublicationPolicy(StrEnum):
    """Controls whether a staged output may replace the requested target."""

    ALWAYS = "always"
    VALIDATED = "validated"


class PDFUAStatus(StrEnum):
    """Truthful PDF/UA outcome without implying human certification."""

    NOT_REQUESTED = "not_requested"
    NOT_PRODUCED = "not_produced"
    VALIDATION_FAILED = "validation_failed"
    REVIEW_REQUIRED = "review_required"
    MACHINE_VALIDATED = "machine_validated"


@dataclass(frozen=True, slots=True)
class PDFUAReviewFinding:
    """One structured reason why a PDF/UA output needs author review."""

    code: str
    message: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class ProfileValidationResult:
    """Evidence from one requested veraPDF profile validation."""

    profile: str
    result: VeraPDFResult | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        """Whether veraPDF returned a parseable validation report."""
        return self.result is not None

    @property
    def compliant(self) -> bool:
        """Whether validation completed and reported conformance."""
        return self.result is not None and self.result.compliant

    def to_dict(self, *, include_raw_xml: bool = False) -> dict[str, Any]:
        """Return JSON-serializable validation evidence."""
        if self.result is None:
            return {
                "profile": self.profile,
                "completed": False,
                "compliant": False,
                "error": self.error,
            }

        result = self.result
        data: dict[str, Any] = {
            "profile": self.profile,
            "completed": True,
            "compliant": result.compliant,
            "detected_flavour": result.flavour,
            "passed_rules": result.passed_rules,
            "failed_rules": result.failed_rules,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "validator_version": result.validator_version,
            "rule_findings": [
                {
                    "specification": finding.specification,
                    "clause": finding.clause,
                    "test_number": finding.test_number,
                    "description": finding.description,
                    "passed_checks": finding.passed_checks,
                    "failed_checks": finding.failed_checks,
                    "contexts": list(finding.contexts),
                }
                for finding in result.rule_findings
            ],
        }
        if include_raw_xml:
            data["raw_xml"] = result.raw_xml
        return data


def _publication_policy(
    value: PublicationPolicy | Literal["always", "validated"] | None,
    *,
    validation_required: bool,
) -> PublicationPolicy:
    """Resolve the publication policy, defaulting validation to fail-closed."""
    if value is None:
        return (
            PublicationPolicy.VALIDATED
            if validation_required
            else PublicationPolicy.ALWAYS
        )
    try:
        return PublicationPolicy(value)
    except ValueError as exc:
        raise ConversionError(
            "publication_policy must be 'validated' or 'always'"
        ) from exc


def _signature_invalidation_warning(count: int, *, pdfa: bool = True) -> str:
    """Build the warning emitted when signature invalidation is explicit."""
    operation = "PDF/A conversion" if pdfa else "PDF processing"
    return f"{count} digital signature(s) will be removed/invalidated for {operation}"


def _compare_pdfa_levels(detected: str, target: str) -> int:
    """Compare two PDF/A levels.

    Compares both the part number (1, 2, 3) and conformance level (a, u, b).
    Different part numbers always return -1 because PDF/A parts are not
    strictly ordered (e.g. PDF/A-3 allows arbitrary embedded files that
    PDF/A-2 does not, so 3 is not a superset of 2).

    Args:
        detected: Detected PDF/A level (e.g., "2b", "3a").
        target: Target PDF/A level (e.g., "2b").

    Returns:
        -1 if detected < target or parts differ
         0 if detected == target
         1 if detected > target (same part, higher conformance)
    """
    detected_part = int(detected[0])
    target_part = int(target[0])

    if detected_part != target_part:
        return -1

    # Same part, compare conformance
    detected_conf_char = detected[1].lower() if len(detected) > 1 else None
    target_conf_char = target[1].lower() if len(target) > 1 else None

    detected_conf = _CONFORMANCE_RANK.get(detected_conf_char, 0)
    target_conf = _CONFORMANCE_RANK.get(target_conf_char, 0)

    if detected_conf < target_conf:
        return -1
    elif detected_conf > target_conf:
        return 1
    return 0


def _validate_pdfa_output(
    output_path: Path,
    level: str,
    warnings: list[str],
    evidence: list[ProfileValidationResult] | None = None,
) -> bool:
    """Validate an output and append all reported compliance errors."""
    try:
        result = validate_with_verapdf(path=output_path, flavour=level)
    except VeraPDFError as exc:
        if evidence is not None:
            evidence.append(ProfileValidationResult(profile=level, error=str(exc)))
        warnings.append(f"Validation: veraPDF could not run: {exc}")
        return True
    if evidence is not None:
        evidence.append(ProfileValidationResult(profile=level, result=result))
    if result.compliant:
        return False
    warnings.extend(f"Validation: {error}" for error in result.errors)
    return True


def _validate_pdfua_output(
    output_path: Path,
    warnings: list[str],
    evidence: list[ProfileValidationResult] | None = None,
) -> bool:
    """Validate PDF/UA-1 conformance and append all reported errors."""
    try:
        result = validate_with_verapdf(path=output_path, flavour="ua1")
    except VeraPDFError as exc:
        if evidence is not None:
            evidence.append(ProfileValidationResult(profile="ua1", error=str(exc)))
        warnings.append(f"PDF/UA validation: veraPDF could not run: {exc}")
        return True
    if evidence is not None:
        evidence.append(ProfileValidationResult(profile="ua1", result=result))
    if result.compliant:
        return False
    warnings.extend(f"PDF/UA validation: {error}" for error in result.errors)
    return True


def _validate_pdfua_options(*, pdfa: bool, level: str, pdfua: bool) -> None:
    """Reject PDF/UA-1 combinations that cannot produce a conforming file."""
    if not pdfua:
        return
    if not pdfa:
        raise ConversionError("PDF/UA-1 cannot be used when pdfa=False")
    if level not in {"2a", "3a"}:
        raise ConversionError("PDF/UA-1 can only be combined with PDF/A-2a or PDF/A-3a")


@dataclass
class ConversionResult:
    """Result of a PDF/A conversion.

    Attributes:
        success: True only if processing and every requested validation succeeded.
        input_path: Path to the input PDF.
        output_path: Requested path for the output PDF.
        level: Requested level for a converted output, detected level for a
            validated compliant skip, or None when PDF/A conversion was
            disabled or an unsupported input was copied unchanged.
        warnings: List of warnings during conversion.
        processing_time: Processing time in seconds.
        error: Error message if success=False.
        validation_failed: True if veraPDF reported non-conformance or could
            not complete, or if a preserved embedded PDF could not be
            converted.
        skipped: True if the original PDF was copied through unchanged.
        published: True if this call wrote the requested output path.
        target_produced: True if the requested conformance target was produced.
        pdfua_status: Machine-verifiable PDF/UA outcome. This never represents
            a human accessibility certification.
        review_findings: Structured accessibility findings requiring author
            review.
        validation_results: Full per-profile veraPDF evidence.
        sanitization_stats: Structured counters from PDF/A sanitization.
        tagging_stats: Structured counters from logical-structure processing.
        metadata_sources: Provenance for accessibility-critical metadata.
        candidate_sha256: SHA-256 of the exact staged bytes submitted to
            validation, whether or not they were published.
        candidate_size: Size of that staged candidate in bytes.
    """

    success: bool
    input_path: Path
    output_path: Path
    level: str | None
    warnings: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    error: str | None = None
    validation_failed: bool = False
    skipped: bool = False
    published: bool = True
    target_produced: bool = True
    pdfua_status: PDFUAStatus = PDFUAStatus.NOT_REQUESTED
    review_findings: tuple[PDFUAReviewFinding, ...] = ()
    validation_results: tuple[ProfileValidationResult, ...] = ()
    sanitization_stats: dict[str, Any] = field(default_factory=dict)
    tagging_stats: dict[str, Any] = field(default_factory=dict)
    metadata_sources: dict[str, str] = field(default_factory=dict)
    candidate_sha256: str | None = None
    candidate_size: int | None = None

    @property
    def review_required(self) -> bool:
        """Whether author review is required before claiming accessibility."""
        return self.pdfua_status is PDFUAStatus.REVIEW_REQUIRED

    def to_dict(self, *, include_raw_validation: bool = False) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable conversion audit record."""
        return {
            "schema_version": 1,
            "success": self.success,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "level": self.level,
            "warnings": list(self.warnings),
            "processing_time": self.processing_time,
            "error": self.error,
            "validation_failed": self.validation_failed,
            "skipped": self.skipped,
            "published": self.published,
            "target_produced": self.target_produced,
            "pdfua_status": self.pdfua_status.value,
            "review_required": self.review_required,
            "review_findings": [
                {
                    "code": finding.code,
                    "message": finding.message,
                    "count": finding.count,
                }
                for finding in self.review_findings
            ],
            "validation_results": [
                evidence.to_dict(include_raw_xml=include_raw_validation)
                for evidence in self.validation_results
            ],
            "sanitization_stats": dict(sorted(self.sanitization_stats.items())),
            "tagging_stats": dict(sorted(self.tagging_stats.items())),
            "metadata_sources": dict(sorted(self.metadata_sources.items())),
            "candidate_sha256": self.candidate_sha256,
            "candidate_size": self.candidate_size,
        }


def generate_output_path(
    input_path: Path,
    output_dir: Path | None = None,
    *,
    pdfa: bool = True,
) -> Path:
    """Generates the output path for a converted PDF.

    Args:
        input_path: Path to the input PDF.
        output_dir: Optional output directory.
        pdfa: If True, generate a PDF/A output name. Otherwise, generate a
            processing-only output name.

    Returns:
        Path for the output PDF.
    """
    suffix = "pdfa" if pdfa else "processed"
    output_name = f"{input_path.stem}_{suffix}.pdf"
    if output_dir is not None:
        return output_dir / output_name
    return input_path.parent / output_name


def _path_identity(path: Path) -> tuple[object, ...]:
    """Return a stable identity for existing files and normalized paths."""
    try:
        stat = path.stat()
    except OSError:
        stat = None
    if stat is not None and stat.st_ino:
        return ("file", stat.st_dev, stat.st_ino)
    return ("path", os.path.normcase(str(path.resolve())))


@contextmanager
def _staged_input_copy(input_path: Path, output_path: Path) -> Iterator[Path]:
    """Yield a same-directory snapshot of an unchanged input."""
    staging = private_staging_directory(
        output_path.parent,
        prefix=f".{output_path.stem}_copy_",
        delete=False,
    )
    staged_output = Path(staging.name) / f"snapshot{output_path.suffix}"
    try:
        copy_to_private_stage(
            input_path,
            Path(staging.name),
            staged_output.name,
        )
        yield staged_output
    finally:
        try:
            staging.cleanup()
        except Exception as cleanup_error:
            logger.warning(
                "Could not delete staged unchanged copy: %s (%s)",
                staged_output,
                cleanup_error,
            )


def _copy_input_to_output(
    input_path: Path,
    output_path: Path,
    *,
    allow_overwrite: bool = True,
) -> None:
    """Atomically copy an unchanged input to the requested output location.

    Overwrites an existing output file, matching the behavior of the
    regular conversion path. Overwrite protection is enforced by the
    callers (CLI and convert_files()).

    Args:
        input_path: Original file path.
        output_path: Destination file path.
    """
    if _path_identity(input_path) == _path_identity(output_path):
        return

    with _staged_input_copy(input_path, output_path) as staged_output:
        snapshot = staged_file_snapshot(staged_output)
        publish_staged_file(
            staged_output,
            output_path,
            snapshot,
            backup=staged_output.with_name(f"backup{output_path.suffix}"),
            require_absent=not allow_overwrite,
        )


def _copy_encrypted_input(
    input_path: Path,
    output_path: Path,
    *,
    pdfa: bool,
    pdfua: bool = False,
    publish_unconverted: bool = False,
    start_time: float,
    allow_overwrite: bool = True,
) -> ConversionResult:
    """Handle an encrypted input without treating it as converted output."""
    warning = (
        "Conversion skipped: PDF is encrypted and cannot be converted"
        if pdfa
        else "Processing skipped: PDF is encrypted and cannot be processed"
    )
    logger.warning("%s: %s", warning, input_path)
    published = not pdfua or publish_unconverted
    warnings = [warning]
    if published:
        _copy_input_to_output(
            input_path,
            output_path,
            allow_overwrite=allow_overwrite,
        )
    else:
        warnings.append(
            "Encrypted input was not published because the PDF/UA target "
            "could not be produced"
        )
    processing_time = time.perf_counter() - start_time
    return ConversionResult(
        success=not pdfua,
        input_path=input_path,
        output_path=output_path,
        level=None,
        warnings=warnings,
        processing_time=processing_time,
        error=(
            "PDF/UA target was not produced because the input is encrypted"
            if pdfua
            else None
        ),
        skipped=True,
        published=published,
        target_produced=not pdfa,
        pdfua_status=(PDFUAStatus.NOT_PRODUCED if pdfua else PDFUAStatus.NOT_REQUESTED),
    )


def _get_pdfa_save_settings_for_version(required_version: str) -> dict[str, object]:
    """Return pikepdf save settings for final PDF/A output."""
    return {
        "linearize": False,
        "force_version": required_version,
        "deterministic_id": True,
        "preserve_pdfa": True,
        "object_stream_mode": pikepdf.ObjectStreamMode.preserve,
    }


def get_pdfa_save_settings(level: str) -> dict[str, object]:
    """Return pikepdf save settings for a PDF/A target level."""
    return _get_pdfa_save_settings_for_version(get_required_pdf_version(level))


def _truncate_trailing_data(output_path: Path) -> bool:
    """Remove data after the last ``%%EOF`` marker (ISO 19005-2, 6.1.3).

    PDF/A requires that no data follows the final ``%%EOF`` marker apart
    from an optional single end-of-line sequence.

    Args:
        output_path: Path to the saved PDF file.

    Returns:
        ``True`` if the file was modified, ``False`` otherwise.
    """
    try:
        data = output_path.read_bytes()
    except Exception as exc:
        raise ConversionError(f"Could not read file for %%EOF check: {exc}") from exc

    eof_marker = b"%%EOF"
    last_eof = data.rfind(eof_marker)
    if last_eof == -1:
        raise ConversionError("No %%EOF marker found in output file")

    # Allow %%EOF + optional single EOL
    cut = last_eof + len(eof_marker)
    if cut < len(data):
        if data[cut : cut + 2] == b"\r\n":
            cut += 2
        elif data[cut : cut + 1] in (b"\n", b"\r"):
            cut += 1

    if cut >= len(data):
        return False  # No trailing data

    trailing = len(data) - cut
    logger.debug("Truncating %d byte(s) after %%%%EOF (ISO 19005-2, 6.1.3)", trailing)
    try:
        output_path.write_bytes(data[:cut])
    except Exception as exc:
        raise ConversionError(f"Could not truncate trailing data: {exc}") from exc

    return True


def _ensure_binary_comment(output_path: Path, required_version: str) -> bool:
    """Ensure the PDF header includes a binary comment line (ISO 19005-2, 6.1.2).

    The PDF/A specification requires a comment containing at least four
    bytes with values > 127 to signal that the file is binary.  If the
    comment is missing, the file is re-saved through pikepdf (which always
    produces a valid binary comment via QPDF).

    Args:
        output_path: Path to the saved PDF file.
        required_version: PDF version string for re-save (e.g. ``"1.7"``).

    Returns:
        ``True`` if the file was modified, ``False`` otherwise.
    """
    try:
        with open(output_path, "rb") as f:
            header = f.read(64)
    except Exception as exc:
        raise ConversionError(
            f"Could not read header for binary comment check: {exc}"
        ) from exc

    # Locate end of first line (%PDF-x.y)
    nl = header.find(b"\n")
    if nl == -1:
        nl = header.find(b"\r")
    if nl == -1:
        raise ConversionError("PDF header has no line ending")

    after = nl + 1
    if after < len(header) and header[after : after + 1] == b"%":
        comment_end = header.find(b"\n", after)
        if comment_end == -1:
            comment_line = header[after + 1 :]
        else:
            comment_line = header[after + 1 : comment_end]
        if comment_line.endswith(b"\r"):
            comment_line = comment_line[:-1]
        if sum(1 for b in comment_line if b > 127) >= 4:
            return False  # Already has valid binary comment

    # Re-save through pikepdf — QPDF always writes a binary comment.
    logger.debug("Re-saving to add binary comment (ISO 19005-2, 6.1.2)")
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=output_path.parent)
    tmp = Path(tmp_path)
    try:
        os.close(fd)
        with pikepdf.open(output_path) as pdf:
            pdf.save(tmp, **_get_pdfa_save_settings_for_version(required_version))
        os.replace(str(tmp), str(output_path))
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception as cleanup_error:
            logger.warning(
                "Could not delete failed binary-comment rewrite: %s (%s)",
                tmp,
                cleanup_error,
            )
        raise ConversionError(f"Could not add binary comment: {exc}") from exc

    return True


def _verify_file_structure(output_path: Path, required_version: str) -> None:
    """Lightweight post-save verification of PDF file structure.

    Checks that the output file has the expected PDF header and a /ID
    array in the trailer. Verification failures prevent publication.

    Args:
        output_path: Path to the saved PDF file.
        required_version: Expected PDF version string (e.g. ``"1.7"``).
    """
    try:
        with open(output_path, "rb") as f:
            header = f.read(20)
    except Exception as exc:
        raise ConversionError(
            f"Post-save verification could not read file: {exc}"
        ) from exc

    # 1. Check header starts with %PDF-<version>
    expected_header = f"%PDF-{required_version}".encode("ascii")
    if not header.startswith(expected_header):
        actual = header[:15].decode("ascii", errors="replace")
        raise ConversionError(
            "Post-save verification: file header "
            f"'{actual}' does not start with expected "
            f"'{expected_header.decode('ascii')}'"
        )

    # 2. Check trailer /ID
    try:
        with pikepdf.open(output_path) as check_pdf:
            id_array = check_pdf.trailer.get("/ID")
            valid_id = (
                isinstance(id_array, pikepdf.Array)
                and len(id_array) == 2
                and all(
                    isinstance(value, pikepdf.String) and bool(bytes(value))
                    for value in id_array
                )
            )
            if not valid_id:
                raise ConversionError(
                    "Post-save verification: trailer /ID must contain two "
                    "non-empty byte strings"
                )
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            f"Post-save verification could not reopen file: {exc}"
        ) from exc


def save_pdfa(
    pdf: pikepdf.Pdf,
    output_path: Path,
    level: str,
    *,
    verify: bool = True,
) -> None:
    """Save final PDF/A output with consistent pikepdf settings and hardening."""
    required_version = get_required_pdf_version(level)
    pdf.save(output_path, **_get_pdfa_save_settings_for_version(required_version))

    # Post-save file structure hardening (ISO 19005-2, 6.1.2/6.1.3).
    _ensure_binary_comment(output_path, required_version)
    _truncate_trailing_data(output_path)

    if verify:
        _verify_file_structure(output_path, required_version)


def _annotated_page_numbers(pdf_path: Path) -> frozenset[int]:
    """Return one-based page numbers with non-empty annotation arrays."""
    annotated_pages = set()
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    annots = page.get("/Annots")
                    if annots is not None and len(annots) > 0:
                        annotated_pages.add(page_number)
                except Exception:
                    continue
    except Exception:
        return frozenset()
    return frozenset(annotated_pages)


def _has_annotations(pdf_path: Path) -> bool:
    """Check whether any page in the PDF contains annotations.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ``True`` if at least one page has a non-empty ``/Annots`` array.
    """
    return bool(_annotated_page_numbers(pdf_path))


def _has_acroform(pdf_path: Path) -> bool:
    """Return whether the PDF declares an interactive form."""
    try:
        with pikepdf.open(pdf_path) as pdf:
            return "/AcroForm" in pdf.Root
    except Exception:
        return False


def _save_with_metadata_fallback(
    pdf: pikepdf.Pdf, output_path: Path, warning_message: str
) -> None:
    """Retry a failed save after removing invalid document metadata."""
    try:
        pdf.save(output_path)
    except Exception as exc:
        metadata_removed = False
        if "/Metadata" in pdf.Root:
            try:
                del pdf.Root["/Metadata"]
                metadata_removed = True
            except Exception:
                metadata_removed = False

        if not metadata_removed:
            raise

        logger.warning(warning_message, exc)
        pdf.save(output_path)


def _strip_annotations_for_ocr(input_path: Path, clean_path: Path) -> bool:
    """Remove all annotations from a PDF for clean OCR processing.

    Strips ``/Annots`` from every page and ``/AcroForm`` from the document
    root so that annotation appearance streams are not rasterized into page
    images during OCR.

    Args:
        input_path: Path to the original PDF.
        clean_path: Path where the cleaned PDF will be saved.

    Returns:
        ``True`` if any annotations were removed, ``False`` otherwise.
    """
    try:
        removed = False
        with pikepdf.open(input_path) as pdf:
            for page in pdf.pages:
                try:
                    annots = page.get("/Annots")
                    if annots is not None and len(annots) > 0:
                        del page["/Annots"]
                        removed = True
                except Exception:
                    continue

            if "/AcroForm" in pdf.Root:
                del pdf.Root["/AcroForm"]
                removed = True

            _save_with_metadata_fallback(
                pdf,
                clean_path,
                "Saving annotation-stripped PDF failed; retrying without "
                "document metadata: %s",
            )
        return removed
    except Exception as exc:
        logger.warning("Could not strip annotations for OCR: %s", exc)
        return False


def _prepare_signed_pdf_for_ocr(input_path: Path, prepared_path: Path) -> dict:
    """Create a temporary OCR source with live signatures neutralized.

    OCR rewrites the PDF and therefore invalidates digital signatures.
    When a signed PDF is converted to PDF/A with OCR enabled, we first
    neutralize live signature values in a temporary copy so ocrmypdf can
    process it instead of aborting early.

    Args:
        input_path: Path to the original PDF.
        prepared_path: Path where the prepared PDF should be saved.

    Returns:
        The ``sanitize_signatures()`` statistics dictionary.
    """
    with pikepdf.open(input_path) as pdf:
        sig_result = sanitize_signatures(pdf)
        if sig_result["signatures_found"] == 0:
            return sig_result

        _save_with_metadata_fallback(
            pdf,
            prepared_path,
            "Saving signature-sanitized OCR source failed; retrying without "
            "document metadata: %s",
        )

    return sig_result


def _restore_annotations_after_ocr(
    original_path: Path, ocr_path: Path, output_path: Path
) -> _AnnotationRestoreResult:
    """Re-inject original annotations into an OCR-processed PDF.

    Copies entire ``/Annots`` arrays (preserving internal cross-references
    like ``/Popup`` and ``/IRT`` chains) from the original PDF into the
    OCR output via ``copy_foreign``.  Also restores ``/AcroForm`` if present.

    Args:
        original_path: Path to the original PDF (with annotations).
        ocr_path: Path to the OCR-processed PDF (without annotations).
        output_path: Path where the merged result will be saved.

    Returns:
        A result that distinguishes no annotations, success, and failure.
    """
    original_pdf = None
    ocr_pdf = None
    try:
        original_pdf = pikepdf.open(original_path)
        ocr_pdf = pikepdf.open(ocr_path)

        if len(original_pdf.pages) != len(ocr_pdf.pages):
            error = (
                "Page count mismatch after OCR "
                f"({len(original_pdf.pages)} vs {len(ocr_pdf.pages)})"
            )
            logger.warning("%s; annotation restoration failed", error)
            return _AnnotationRestoreResult(
                status=_AnnotationRestoreStatus.FAILED,
                error=error,
            )

        annotation_count = 0
        for page in original_pdf.pages:
            try:
                annots = page.get("/Annots")
                if annots is not None:
                    annotation_count += len(annots)
            except Exception:
                continue

        has_acroform = "/AcroForm" in original_pdf.Root
        if annotation_count == 0 and not has_acroform:
            return _AnnotationRestoreResult(
                status=_AnnotationRestoreStatus.NO_ANNOTATIONS
            )

        total_restored = 0
        for orig_page, ocr_page in zip(original_pdf.pages, ocr_pdf.pages):
            try:
                annots = orig_page.get("/Annots")
                if annots is None or len(annots) == 0:
                    continue
            except Exception:
                continue

            # Ensure annots is indirect so copy_foreign can track it.
            annots_ref = original_pdf.make_indirect(annots)
            copied_annots = ocr_pdf.copy_foreign(annots_ref)

            # Remap /P (parent page) references to the target page.
            for annot in copied_annots:
                try:
                    resolved = annot.get_object()
                except (AttributeError, TypeError, ValueError):
                    resolved = annot
                if "/P" in resolved:
                    resolved["/P"] = ocr_page.obj

            ocr_page.obj["/Annots"] = copied_annots
            total_restored += len(copied_annots)

        if total_restored != annotation_count:
            error = (
                "Annotation count mismatch while restoring after OCR "
                f"({total_restored} of {annotation_count} restored)"
            )
            logger.warning("%s", error)
            return _AnnotationRestoreResult(
                status=_AnnotationRestoreStatus.FAILED,
                count=total_restored,
                error=error,
            )

        # Restore /AcroForm if present in original. copy_foreign preserves
        # shared indirect objects, including relationships to copied widgets.
        if has_acroform:
            acroform = original_pdf.Root["/AcroForm"]
            acroform_ref = original_pdf.make_indirect(acroform)
            ocr_pdf.Root["/AcroForm"] = ocr_pdf.copy_foreign(acroform_ref)

        ocr_pdf.save(output_path)
        return _AnnotationRestoreResult(
            status=_AnnotationRestoreStatus.SUCCESS,
            count=total_restored,
        )
    except Exception as exc:
        error = f"Could not restore annotations after OCR: {exc}"
        logger.warning("%s", error)
        return _AnnotationRestoreResult(
            status=_AnnotationRestoreStatus.FAILED,
            error=error,
        )
    finally:
        if original_pdf is not None:
            try:
                original_pdf.close()
            except Exception:
                pass
        if ocr_pdf is not None:
            try:
                ocr_pdf.close()
            except Exception:
                pass


def convert_to_pdfa(
    input_path: Path,
    output_path: Path,
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    validate: bool = False,
    publication_policy: PublicationPolicy
    | Literal["always", "validated"]
    | None = None,
    document_title: str | None = None,
    document_language: str | None = None,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
    _allow_output_overwrite: bool = True,
) -> ConversionResult:
    """Converts a PDF file to PDF/A or applies only requested OCR processing.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the output PDF.
        level: PDF/A conformance level ('2a', '2b', '2u', '3a', '3b',
            or '3u').
        pdfa: If False, skip all PDF/A-specific processing and save only the
            requested OCR processing result. PDF/A-specific options are
            ignored in this mode.
        pdfua: If True, also produce PDF/UA-1 output. This requires PDF/A-2a
            or PDF/A-3a.
        validate: If True, the result is validated. PDF/UA-1 output is always
            submitted to both requested validation profiles.
        publication_policy: ``"validated"`` publishes only when every
            requested validator completed successfully and reported
            conformance. ``"always"`` also publishes non-conforming
            candidates. Validation defaults to fail-closed.
        document_title: Authoritative document title for PDF/UA output. When
            omitted, source metadata is preserved and a filename fallback is
            reported for author review.
        document_language: Authoritative BCP 47 language tag for the document
            catalog (for example ``"de"`` or ``"en-GB"``).
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``). Requires both model directories.
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
            Must be provided with ``ocr_recognition_model_dir``.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
            Must be provided with ``ocr_detection_model_dir``.
        ocr_force: If True, force OCR even on pages that already contain
            text by using ocrmypdf's ``redo_ocr`` mode. This cannot be
            combined with ``ocr_deskew``.
        ocr_deskew: If True, enable OCR and straighten scan-like,
            raster-dominant pages.
        ocr_rotate_pages: If True, enable OCR and normalize page orientation.
        ocr_execution_provider: ONNX Runtime provider for Paddle models:
            ``"cpu"`` (default), ``"directml"`` or ``"directml:<index>"``
            to select a specific adapter.
        ocr_layout: If True, order OCR lines by detected page columns.
        convert_calibrated: If True, convert CalGray/CalRGB to ICCBased.
        preserve_stamps: If True, known proprietary stamp annotations are
            normalized to standard ``/Stamp`` annotations instead of being
            flattened into page content.
        allow_signature_invalidation: If True, convert signed PDFs even though
            conversion removes or invalidates their digital signatures.

    Returns:
        ConversionResult with status and details.

    Raises:
        ConversionError: If conversion fails.
        UnsupportedPDFError: If the PDF is not supported.
        FontEmbeddingError: If fonts cannot be embedded.
    """
    ocr_requested, effective_ocr_languages = _validate_ocr_configuration(
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        ocr_execution_provider=ocr_execution_provider,
        ocr_layout=ocr_layout,
    )
    if pdfa:
        level = validate_pdfa_level(level)
    _validate_pdfua_options(pdfa=pdfa, level=level, pdfua=pdfua)
    if not pdfa and validate:
        raise ConversionError("PDF/A validation cannot be used when pdfa=False")
    if not pdfa and (document_title is not None or document_language is not None):
        raise ConversionError(
            "Document metadata overrides cannot be used when pdfa=False"
        )
    effective_publication_policy = _publication_policy(
        publication_policy,
        validation_required=validate or pdfua,
    )
    cleaned_document_title = None
    if document_title is not None:
        cleaned_document_title = _clean_metadata_text(document_title)
        if cleaned_document_title is None:
            raise ConversionError("document_title must contain visible text")
    cleaned_document_language = None
    if document_language is not None:
        cleaned_document_language = document_language.strip()
        if not _is_valid_bcp47(cleaned_document_language):
            raise ConversionError("document_language must be a valid BCP 47 tag")
    start_time = time.perf_counter()
    warnings: list[str] = []
    validation_results: list[ProfileValidationResult] = []
    review_findings: list[PDFUAReviewFinding] = []
    sanitize_result: dict[str, Any] = {}
    tagging_result: dict[str, Any] = {}
    title_source = "not_applicable"
    language_source = "not_applicable"
    unrequested_pdfua_claim = False
    ocr_temp_file: Path | None = None
    ocr_signature_temp_file: Path | None = None
    ocr_clean_temp_file: Path | None = None
    ocr_merged_temp_file: Path | None = None
    ocr_manifest_temp_file: Path | None = None
    final_output_temp_file: Path | None = None
    final_output_temp_directory: tempfile.TemporaryDirectory[str] | None = None
    pdf: pikepdf.Pdf | None = None
    source_info: dict | None = None
    source_xmp_tree = None

    if ocr_force and ocr_deskew:
        raise OCRError("Deskew cannot be combined with forced OCR")
    explicit_ocr_processing_requested = ocr_requested
    explicit_metadata_update_requested = (
        cleaned_document_title is not None or cleaned_document_language is not None
    )

    if pdfa:
        logger.info(
            "Starting conversion: %s -> %s (PDF/A-%s%s)",
            input_path,
            output_path,
            level,
            " + PDF/UA-1" if pdfua else "",
        )
    else:
        logger.info("Starting PDF processing: %s -> %s", input_path, output_path)

    try:
        # 0. Check if PDF is already PDF/A compliant (before OCR)
        with pikepdf.open(input_path) as check_pdf:
            if is_pdf_encrypted(check_pdf):
                if not pdfa:
                    check_pdf.close()
                    return _copy_encrypted_input(
                        input_path,
                        output_path,
                        pdfa=False,
                        pdfua=False,
                        publish_unconverted=True,
                        start_time=start_time,
                        allow_overwrite=_allow_output_overwrite,
                    )
                warnings.append("Encryption removed for PDF/A compliance")
            if pdfa and len(check_pdf.pages) == 0:
                raise UnsupportedPDFError(
                    "PDF contains no pages and cannot be converted to PDF/A"
                )
            if not pdfa and not ocr_requested:
                processing_time = time.perf_counter() - start_time
                warning = "No processing options requested; input copied unchanged"
                check_pdf.close()
                _copy_input_to_output(
                    input_path,
                    output_path,
                    allow_overwrite=_allow_output_overwrite,
                )
                return ConversionResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    level=None,
                    warnings=[warning],
                    processing_time=processing_time,
                    skipped=True,
                )
            source_info = extract_pdf_info(check_pdf) if pdfa else None
            source_xmp_tree = _extract_existing_xmp(check_pdf) if pdfa else None
            if pdfa and source_info is not None:
                source_title = _clean_metadata_text(source_info.get("title"))
                title_source = (
                    "source_docinfo" if source_title is not None else "missing"
                )
                if source_title is None:
                    source_title = _extract_lang_alt_xmp_property(
                        source_xmp_tree,
                        NAMESPACES["dc"],
                        "title",
                    )
                    if source_title is not None:
                        title_source = "source_xmp"
                if cleaned_document_title is not None:
                    source_info["title"] = cleaned_document_title
                    title_source = "user"
                elif source_title is None and pdfua:
                    title_source = "fallback"

                try:
                    source_catalog_language = str(check_pdf.Root.get("/Lang", ""))
                except (TypeError, ValueError):
                    source_catalog_language = ""
                source_catalog_language = source_catalog_language.strip()
                if cleaned_document_language is not None:
                    language_source = "user"
                elif _is_valid_bcp47(source_catalog_language):
                    language_source = "source_catalog"
            signature_count = count_digital_signatures(check_pdf)
            if signature_count > 0 and not allow_signature_invalidation:
                signature_skip_warning = (
                    _SIGNATURE_SKIP_WARNING
                    if pdfa
                    else (
                        "Processing skipped: PDF contains digital signatures; "
                        "processing would invalidate them"
                    )
                )
                logger.warning("%s: %s", signature_skip_warning, input_path)
                check_pdf.close()
                published = not pdfua or (
                    effective_publication_policy is PublicationPolicy.ALWAYS
                )
                signature_warnings = [signature_skip_warning]
                if published:
                    _copy_input_to_output(
                        input_path,
                        output_path,
                        allow_overwrite=_allow_output_overwrite,
                    )
                else:
                    signature_warnings.append(
                        "Signed input was not published because the PDF/UA target "
                        "could not be produced"
                    )
                processing_time = time.perf_counter() - start_time
                return ConversionResult(
                    success=not pdfua,
                    input_path=input_path,
                    output_path=output_path,
                    level=None,
                    warnings=signature_warnings,
                    processing_time=processing_time,
                    error=(
                        "PDF/UA target was not produced because conversion would "
                        "invalidate a digital signature"
                        if pdfua
                        else None
                    ),
                    skipped=True,
                    published=published,
                    target_produced=not pdfa,
                    pdfua_status=(
                        PDFUAStatus.NOT_PRODUCED if pdfua else PDFUAStatus.NOT_REQUESTED
                    ),
                )
            if signature_count > 0:
                warnings.append(
                    _signature_invalidation_warning(signature_count, pdfa=pdfa)
                )
            detected_level = detect_pdfa_level(check_pdf) if pdfa else None
            if pdfa and not pdfua:
                unrequested_pdfua_claim = any(
                    standard.standard == "PDF/UA"
                    for standard in detect_iso_standards(check_pdf)
                )

        if detected_level is not None and (
            explicit_ocr_processing_requested
            or explicit_metadata_update_requested
            or unrequested_pdfua_claim
        ):
            logger.debug(
                "Bypassing PDF/A pre-check skip because %s",
                (
                    "an unrequested PDF/UA claim must be removed"
                    if unrequested_pdfua_claim
                    else "explicit processing was requested"
                ),
            )

        if detected_level is not None and not (
            explicit_ocr_processing_requested
            or explicit_metadata_update_requested
            or unrequested_pdfua_claim
        ):
            should_validate_for_skip = skip_any_pdfa
            if not should_validate_for_skip and not level.endswith("a"):
                level_cmp = _compare_pdfa_levels(detected_level, level)
                should_validate_for_skip = level_cmp >= 0

            if should_validate_for_skip:
                with _staged_input_copy(input_path, output_path) as staged_input:
                    staged_snapshot = staged_file_snapshot(staged_input)
                    try:
                        verapdf_result = validate_with_verapdf(
                            staged_input,
                            flavour=detected_level,
                            non_compliant_log_level=logging.WARNING,
                        )
                    except VeraPDFError:
                        logger.debug(
                            "veraPDF not available, skipping PDF/A pre-check for %s",
                            input_path,
                        )
                        verapdf_result = None

                    pdfua_result = None
                    if (
                        pdfua
                        and verapdf_result is not None
                        and verapdf_result.compliant
                    ):
                        try:
                            pdfua_result = validate_with_verapdf(
                                staged_input,
                                flavour="ua1",
                                non_compliant_log_level=logging.WARNING,
                            )
                        except VeraPDFError:
                            logger.debug(
                                "veraPDF PDF/UA-1 pre-check unavailable for %s",
                                input_path,
                            )

                    pdfua_compliant = not pdfua or (
                        pdfua_result is not None and pdfua_result.compliant
                    )
                    if (
                        verapdf_result is not None
                        and verapdf_result.compliant
                        and pdfua_compliant
                    ):
                        processing_time = time.perf_counter() - start_time
                        logger.info(
                            "Skipping conversion: PDF is already valid PDF/A-%s",
                            detected_level,
                        )
                        publish_staged_file(
                            staged_input,
                            output_path,
                            staged_snapshot,
                            backup=staged_input.with_name(
                                f"backup{output_path.suffix}"
                            ),
                            require_absent=not _allow_output_overwrite,
                        )
                        return ConversionResult(
                            success=True,
                            input_path=input_path,
                            output_path=output_path,
                            level=detected_level,
                            warnings=[
                                "Conversion skipped: PDF already valid PDF/A "
                                "(veraPDF compliant)"
                            ],
                            processing_time=processing_time,
                            skipped=True,
                            pdfua_status=(
                                PDFUAStatus.MACHINE_VALIDATED
                                if pdfua
                                else PDFUAStatus.NOT_REQUESTED
                            ),
                            validation_results=tuple(
                                ProfileValidationResult(
                                    profile=profile,
                                    result=result,
                                )
                                for profile, result in (
                                    (detected_level, verapdf_result),
                                    ("ua1", pdfua_result),
                                )
                                if result is not None
                            ),
                            metadata_sources={
                                "title": title_source,
                                "language": language_source,
                            },
                            candidate_sha256=staged_snapshot.sha256,
                            candidate_size=staged_snapshot.size,
                        )
                    if verapdf_result is not None:
                        logger.info(
                            "PDF claims PDF/A-%s but validation failed, converting",
                            detected_level,
                        )
            elif not skip_any_pdfa:
                logger.debug(
                    "PDF is PDF/A-%s, converting to PDF/A-%s",
                    detected_level,
                    level,
                )

        # Reject in-place processing before OCR replaces actual_input with its
        # temporary output path.
        if _path_identity(input_path) == _path_identity(output_path):
            raise ConversionError(f"Input and output paths must differ: {input_path}")

        # 1. Optional: Perform OCR
        actual_input = input_path
        if ocr_requested:
            from .ocr import apply_ocr, is_ocr_available

            if not is_ocr_available():
                extra = (
                    "directml"
                    if execution_provider_base(ocr_execution_provider) == "directml"
                    else "ocr"
                )
                raise OCRError(f"OCR not available - pip install pdftopdfa[{extra}]")
            fd, tmp_path = tempfile.mkstemp(
                suffix=".pdf", prefix=f".{input_path.stem}_ocr_"
            )
            os.close(fd)
            ocr_temp_file = Path(tmp_path)
            ocr_source_base = input_path

            if pdfa and level.endswith("a"):
                manifest_fd, manifest_tmp = tempfile.mkstemp(
                    suffix=".json",
                    prefix=f".{input_path.stem}_ocr_semantics_",
                )
                os.close(manifest_fd)
                ocr_manifest_temp_file = Path(manifest_tmp)
                ocr_manifest_temp_file.unlink()

            fd_sig, sig_tmp = tempfile.mkstemp(
                suffix=".pdf",
                prefix=f".{input_path.stem}_sig_",
            )
            os.close(fd_sig)
            ocr_signature_temp_file = Path(sig_tmp)
            sig_result = _prepare_signed_pdf_for_ocr(
                input_path, ocr_signature_temp_file
            )
            if sig_result["signatures_found"] > 0:
                ocr_source_base = ocr_signature_temp_file
                if not any("digital signature" in w for w in warnings):
                    warnings.append(
                        _signature_invalidation_warning(
                            sig_result["signatures_found"], pdfa=pdfa
                        )
                    )
            else:
                try:
                    ocr_signature_temp_file.unlink()
                except Exception:
                    pass
                else:
                    ocr_signature_temp_file = None

            # Strip annotations before OCR so they are not
            # rasterized into page images.
            annotated_pages = _annotated_page_numbers(ocr_source_base)
            preserve_annots = bool(annotated_pages) or _has_acroform(ocr_source_base)
            ocr_source = ocr_source_base
            if preserve_annots:
                fd2, clean_tmp = tempfile.mkstemp(
                    suffix=".pdf",
                    prefix=f".{input_path.stem}_clean_",
                )
                os.close(fd2)
                ocr_clean_temp_file = Path(clean_tmp)
                if _strip_annotations_for_ocr(ocr_source_base, ocr_clean_temp_file):
                    ocr_source = ocr_clean_temp_file
                else:
                    preserve_annots = False

            apply_ocr(
                ocr_source,
                ocr_temp_file,
                effective_ocr_languages,
                detection_model_dir=ocr_detection_model_dir,
                recognition_model_dir=ocr_recognition_model_dir,
                force=ocr_force,
                deskew=ocr_deskew,
                rotate_pages=ocr_rotate_pages,
                ocr_execution_provider=ocr_execution_provider,
                layout=ocr_layout,
                _annotated_pages=annotated_pages,
                _manifest_output_path=ocr_manifest_temp_file,
            )

            # Re-inject original annotations into OCR output.
            if preserve_annots:
                fd3, merged_tmp = tempfile.mkstemp(
                    suffix=".pdf",
                    prefix=f".{input_path.stem}_merged_",
                )
                os.close(fd3)
                ocr_merged_temp_file = Path(merged_tmp)
                restore_result = _restore_annotations_after_ocr(
                    ocr_source_base, ocr_temp_file, ocr_merged_temp_file
                )
                if restore_result.status is not _AnnotationRestoreStatus.SUCCESS:
                    reason = restore_result.error or (
                        "No annotations were available for restoration"
                    )
                    raise OCRError(
                        "OCR annotation restoration failed; conversion aborted "
                        f"to prevent annotation loss: {reason}"
                    )
                try:
                    os.replace(str(ocr_merged_temp_file), str(ocr_temp_file))
                except OSError as exc:
                    raise OCRError(
                        "OCR annotation restoration could not be finalized; "
                        f"conversion aborted to prevent annotation loss: {exc}"
                    ) from exc
                logger.debug(
                    "%d annotation(s) preserved through OCR", restore_result.count
                )
                warnings.append(
                    f"{restore_result.count} annotation(s) preserved through OCR"
                )
                ocr_merged_temp_file = None

            # Clean up the stripped copy.
            if ocr_clean_temp_file is not None:
                try:
                    ocr_clean_temp_file.unlink()
                except Exception:
                    pass
                else:
                    ocr_clean_temp_file = None

            actual_input = ocr_temp_file
            lang_str = "+".join(effective_ocr_languages)
            warnings.append(f"OCR performed (languages: {lang_str})")

        if not pdfa:
            _copy_input_to_output(
                actual_input,
                output_path,
                allow_overwrite=_allow_output_overwrite,
            )
            processing_time = time.perf_counter() - start_time
            logger.info(
                "PDF processing successful: %s (%.2f seconds)",
                output_path,
                processing_time,
            )
            return ConversionResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                level=None,
                warnings=warnings,
                processing_time=processing_time,
            )

        # 2. Open PDF
        logger.debug("Opening PDF: %s", actual_input)
        pdf = pikepdf.open(actual_input)

        # 2.6. Detect other ISO PDF standards (informational)
        iso_standards = detect_iso_standards(pdf)
        if iso_standards:
            for std in iso_standards:
                msg = f"ISO standard detected: {std.standard} {std.version}"
                logger.debug(msg)
                warnings.append(msg)

        # 3. Check font compliance and embed missing fonts.
        # A single FontEmbedder serves all font passes so loaded replacement
        # fonts stay cached across the steps.
        from .fonts import FontEmbedder

        logger.debug("Checking font compliance")
        is_compliant, missing_fonts = check_font_compliance(pdf, raise_on_error=False)
        with FontEmbedder(pdf) as embedder:
            if not is_compliant:
                logger.debug(
                    "Attempting to embed missing fonts: %s",
                    ", ".join(missing_fonts),
                )
                embed_result = embedder.embed_missing_fonts()

                if embed_result.fonts_embedded:
                    logger.debug(
                        "Fonts embedded: %s",
                        ", ".join(embed_result.fonts_embedded),
                    )

                if embed_result.fonts_failed:
                    raise FontEmbeddingError(
                        "Could not embed fonts: "
                        f"{', '.join(embed_result.fonts_failed)}. "
                        "All fonts must be embedded for PDF/A compliance "
                        "(ISO 19005, clause 6.3.1)."
                    )

                warnings.extend(embed_result.warnings)

            # Normalize simple-font lookup before generating ToUnicode or
            # subsetting so all three passes select the same glyphs. The full
            # sanitizer repeats this after subsetting because fontTools may
            # prune cmap subtables.
            sanitize_truetype_encoding(pdf)

            # 3.5. Unicode compliance — always add ToUnicode to all embedded
            # fonts (ISO 19005-2/3, rule 6.2.11.7.2).  veraPDF requires
            # explicit ToUnicode even when Unicode is theoretically derivable.
            logger.debug("Adding ToUnicode to embedded fonts for PDF/A-%s", level)
            tounicode_result = embedder.add_tounicode_to_embedded_fonts()

            if tounicode_result.fonts_embedded:
                logger.debug(
                    "ToUnicode added to fonts: %s",
                    ", ".join(tounicode_result.fonts_embedded),
                )

            if tounicode_result.fonts_failed:
                raise ConversionError(
                    "Could not add ToUnicode mappings to: "
                    f"{', '.join(tounicode_result.fonts_failed)}. "
                    "ToUnicode is required for PDF/A compliance "
                    f"(ISO 19005-2/3, rule 6.2.11.7.2, level {level})."
                )

            warnings.extend(tounicode_result.warnings)

            original_subsetted_standard14_font_ids = (
                embedder.collect_subsetted_standard14_font_ids()
            )

            # 3.7. Subset embedded fonts to reduce file size.
            # Font encoding fixes (ISO 19005-2, 6.2.11.6) happen later in
            # sanitize_truetype_encoding(), which runs after subsetting so
            # the cmaps it adds are not pruned by the subsetter.
            logger.debug("Subsetting embedded fonts")
            subset_result = embedder.subset_embedded_fonts()

            if subset_result.fonts_subsetted:
                logger.debug(
                    "Fonts subsetted: %s (saved %d bytes)",
                    ", ".join(subset_result.fonts_subsetted),
                    subset_result.bytes_saved,
                )

            if subset_result.warnings:
                warnings.extend(subset_result.warnings)

            # Replace embedded subsetted Standard-14 fonts with full bundled
            # replacements after subsetting so incomplete original subsets are
            # not carried forward into the final PDF/A output.
            logger.debug("Refreshing subsetted Standard-14 fonts")
            refresh_result = embedder.replace_subsetted_standard14_fonts(
                original_subsetted_standard14_font_ids
            )

            warnings.extend(refresh_result.warnings)

            dedupe_result = embedder.deduplicate_embedded_font_programs()
            if dedupe_result.programs_deduplicated:
                logger.debug(
                    "Deduplicated %d embedded font program(s) (saved ~%d bytes)",
                    dedupe_result.programs_deduplicated,
                    dedupe_result.bytes_saved_estimate,
                )

        # 4. Sanitize PDF for PDF/A
        logger.debug("Sanitizing PDF for PDF/A-%s", level)
        sanitize_result = sanitize_for_pdfa(pdf, level, preserve_stamps=preserve_stamps)
        if cleaned_document_language is not None:
            pdf.Root["/Lang"] = pikepdf.String(cleaned_document_language)
        if pdfua:
            ensure_display_doc_title(pdf, level)

        # Collect warnings from sanitization
        for key, message in _SANITIZE_WARNINGS:
            count = sanitize_result.get(key, 0)
            if count > 0:
                warnings.append(f"{count} {message}")

        for key, error_msg in _SANITIZE_ERRORS:
            count = sanitize_result.get(key, 0)
            if count > 0:
                raise ConversionError(f"{count} {error_msg}")

        for keys, message in _SANITIZE_COMBINED_WARNINGS:
            count = sum(sanitize_result.get(k, 0) for k in keys)
            if count > 0:
                warnings.append(f"{count} {message}")

        # 5. Synchronize metadata
        logger.debug("Synchronizing XMP metadata")
        sync_metadata(
            pdf,
            level,
            source_info=source_info,
            source_xmp_tree=source_xmp_tree,
            pdfua=pdfua,
            fallback_title=input_path.stem,
        )

        # A PDF/UA claim is valid only for the explicitly requested and
        # subsequently validated output. Never carry an inherited claim into
        # a different processing contract.
        if not pdfua and remove_pdfua_identification(pdf):
            warnings.append(
                "PDF/UA identification removed from XMP metadata "
                "(PDF/UA output was not requested)"
            )

        # 5.5. Add Extensions dictionary for PDF/A-3
        add_extensions_if_needed(pdf, level)

        # 6. Detect color spaces and embed profiles
        logger.debug("Detecting color spaces and embedding ICC profiles")
        embedded_spaces = embed_color_profiles(
            pdf, level, convert_calibrated=convert_calibrated
        )
        if len(embedded_spaces) > 1:
            warnings.append(
                "Multiple color spaces handled: "
                f"{', '.join(cs.value for cs in embedded_spaces)}"
            )

        # embed_color_profiles() keeps at most a single PDF/A OutputIntent,
        # so any PDF/X OutputIntent is gone. Remove a stale PDF/X claim from
        # the XMP so the file does not assert PDF/X without its OutputIntent.
        if remove_pdfx_identification(pdf):
            warnings.append(
                "PDF/X identification removed from XMP metadata "
                "(PDF/X OutputIntent not preserved)"
            )
        if remove_pdfvt_identification(pdf):
            warnings.append(
                "PDF/VT identification removed from XMP metadata "
                "(required PDF/X conformance not preserved)"
            )
        if remove_pdfe_identification(pdf):
            warnings.append(
                "PDF/E identification removed from XMP metadata "
                "(required DocInfo identification not preserved)"
            )

        # Final pass for structural limits:
        # embed_color_profiles() may materialize or rewrite ColorSpace names.
        late_structure_result = sanitize_structure_limits(pdf)
        for key, message in _LATE_STRUCTURE_WARNINGS:
            count = late_structure_result.get(key, 0)
            if count > 0:
                warnings.append(f"{count} {message}")

        # The late structure pass can rewrite malformed text strings and
        # thereby reintroduce bytes that resolve to .notdef. Run the
        # content-level .notdef cleanup once more on the final in-memory PDF.
        late_notdef_result = sanitize_notdef_usage(pdf)
        count = late_notdef_result.get("notdef_usage_fixed", 0)
        if count > 0:
            warnings.append(f"{count} .notdef usage operator(s) fixed")

        if level.endswith("a"):
            ocr_manifest = None
            if ocr_manifest_temp_file is not None and ocr_manifest_temp_file.exists():
                try:
                    ocr_manifest = json.loads(
                        ocr_manifest_temp_file.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise OCRError(
                        f"Could not read OCR semantic manifest: {exc}"
                    ) from exc
            # No preflight rehearsal: any failure below propagates out of this
            # try block, which closes `pdf` unsaved and unlinks the staged
            # output, so a partially built structure tree never reaches disk.
            tagging_result = ensure_logical_structure(
                pdf,
                semantic=True,
                pdfua=pdfua,
                ocr_manifest=ocr_manifest,
                preflight=False,
            )
            if tagging_result.get("semantic_repairs", 0):
                repair_count = tagging_result["semantic_repairs"]
                repair_label = "property" if repair_count == 1 else "properties"
                warnings.append(
                    f"{repair_count} semantic structure {repair_label} repaired"
                )
            alternative_review_count = tagging_result.get(
                "semantic_alternatives_review_required",
                0,
            )
            if alternative_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="missing_alternative_text",
                        message=(
                            "Figure or Formula has no trustworthy Alt, ActualText, "
                            "or Caption"
                        ),
                        count=alternative_review_count,
                    )
                )
                element_label = (
                    "element" if alternative_review_count == 1 else "elements"
                )
                review_verb = "requires" if alternative_review_count == 1 else "require"
                warnings.append(
                    f"{alternative_review_count} Figure/Formula {element_label} "
                    f"{review_verb} manual review: no trustworthy Alt, "
                    "ActualText, or Caption is available"
                )
            vector_review_count = tagging_result.get(
                "semantic_vector_review_required",
                0,
            )
            if vector_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="unclassified_vector_content",
                        message=(
                            "Direct vector painting was retained as a Layout artifact"
                        ),
                        count=vector_review_count,
                    )
                )
                page_label = "page" if vector_review_count == 1 else "pages"
                warnings.append(
                    f"{vector_review_count} {page_label} require manual review: "
                    "unclassified direct vector painting was retained as a "
                    "Layout artifact"
                )
            table_review_count = tagging_result.get(
                "semantic_table_review_required",
                0,
            )
            if table_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="non_rectangular_table",
                        message=(
                            "Inferred non-rectangular table was retained as "
                            "conservative reading structure"
                        ),
                        count=table_review_count,
                    )
                )
                table_label = "table" if table_review_count == 1 else "tables"
                review_verb = "requires" if table_review_count == 1 else "require"
                retain_verb = "was" if table_review_count == 1 else "were"
                warnings.append(
                    f"{table_review_count} inferred non-rectangular {table_label} "
                    f"{review_verb} manual review and {retain_verb} retained as "
                    "conservative reading structure"
                )
            scanned_visual_review_count = tagging_result.get(
                "semantic_scanned_visual_review_required",
                0,
            )
            if scanned_visual_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="unrepresented_scan_visuals",
                        message=(
                            "Full-page scan may contain meaningful non-text visuals"
                        ),
                        count=scanned_visual_review_count,
                    )
                )
                page_label = (
                    "OCR page" if scanned_visual_review_count == 1 else "OCR pages"
                )
                review_verb = (
                    "requires" if scanned_visual_review_count == 1 else "require"
                )
                scan_label = (
                    "a full-page scan"
                    if scanned_visual_review_count == 1
                    else "full-page scans"
                )
                warnings.append(
                    f"{scanned_visual_review_count} {page_label} {review_verb} "
                    f"manual review: {scan_label} may contain meaningful non-text "
                    "visuals that available OCR layout evidence cannot represent"
                )
            link_review_count = tagging_result.get(
                "semantic_link_review_required",
                0,
            )
            if link_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="unassociated_link",
                        message=(
                            "Link could not be safely associated with one logical "
                            "structure element"
                        ),
                        count=link_review_count,
                    )
                )
                link_label = (
                    "Link annotation" if link_review_count == 1 else "Link annotations"
                )
                review_verb = "requires" if link_review_count == 1 else "require"
                warnings.append(
                    f"{link_review_count} {link_label} {review_verb} manual review: "
                    "the link could not be safely associated with content owned by "
                    "a single logical structure element"
                )
            form_review_count = tagging_result.get(
                "semantic_form_review_required",
                0,
            )
            if form_review_count:
                review_findings.append(
                    PDFUAReviewFinding(
                        code="unnamed_form_field",
                        message="Form field has no trustworthy tooltip or field name",
                        count=form_review_count,
                    )
                )
                field_label = "Form field" if form_review_count == 1 else "Form fields"
                review_verb = "requires" if form_review_count == 1 else "require"
                warnings.append(
                    f"{form_review_count} {field_label} {review_verb} manual review: "
                    "no trustworthy tooltip or field name is available"
                )
            if tagging_result["structure_rebuilt"]:
                if tagging_result.get("semantic_structure_generated"):
                    review_findings.append(
                        PDFUAReviewFinding(
                            code="generated_semantic_structure",
                            message=(
                                "Logical reading order and semantics were generated "
                                "automatically"
                            ),
                        )
                    )
                    if ocr_manifest is not None:
                        semantic_generation_warning = (
                            "Semantic Tagged PDF structure generated from final "
                            "digital content"
                        )
                        if ocr_manifest.get("pages"):
                            semantic_generation_warning += (
                                " and available OCR layout evidence"
                            )
                        if any(
                            (
                                alternative_review_count,
                                vector_review_count,
                                table_review_count,
                                scanned_visual_review_count,
                                link_review_count,
                                form_review_count,
                            )
                        ):
                            semantic_generation_warning += (
                                "; review reported semantic uncertainties"
                            )
                        warnings.append(semantic_generation_warning)
                    else:
                        warnings.append(
                            "Semantic Tagged PDF structure generated from PDF layout"
                        )
                else:
                    warnings.append(
                        "Tagged PDF structure generated from page content order"
                    )

        if pdfua:
            if title_source == "fallback":
                review_findings.append(
                    PDFUAReviewFinding(
                        code="fallback_document_title",
                        message="Document title was derived from the input filename",
                    )
                )
            catalog_language = str(pdf.Root.get("/Lang", "")).strip()
            if language_source == "not_applicable":
                language_source = (
                    "fallback" if catalog_language == "und" else "source_xmp"
                )
            if catalog_language == "und":
                review_findings.append(
                    PDFUAReviewFinding(
                        code="undetermined_document_language",
                        message=(
                            "Document language is undetermined; supply an "
                            "authoritative BCP 47 language tag"
                        ),
                    )
                )
            review_findings.append(
                PDFUAReviewFinding(
                    code="human_accessibility_review",
                    message=(
                        "Reading order, semantic meaning, alternatives, visual "
                        "contrast, and usability require human verification"
                    ),
                )
            )

        # 7. Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 8. Save PDF with minimum version.
        #
        # Keep output non-linearized because QPDF linearization can still
        # produce invalid /Length values on generated hint streams
        # (rule 6.1.7.1) for specific inputs.
        final_output_temp_directory = private_staging_directory(
            output_path.parent,
            prefix=f".{output_path.stem}_pdfa_stage_",
            delete=False,
        )
        final_output_temp_file = (
            Path(final_output_temp_directory.name)
            / f".{output_path.stem}_pdfa_output.pdf"
        )
        final_output_temp_file.touch(exist_ok=False)

        logger.debug("Saving staged PDF/A: %s", final_output_temp_file)
        required_version = get_required_pdf_version(level)
        current_version = pdf.pdf_version
        if current_version != required_version:
            direction = (
                "upgraded"
                if tuple(int(x) for x in current_version.split("."))
                < tuple(int(x) for x in required_version.split("."))
                else "downgraded"
            )
            warnings.append(
                f"PDF version {direction} from {current_version} to {required_version}"
            )

        save_pdfa(pdf, final_output_temp_file, level, verify=True)
        final_output_snapshot = staged_file_snapshot(final_output_temp_file)
        pdf.close()
        pdf = None

        # 9. Validate requested profiles before atomically publishing the same
        # candidate. Validation is fail-closed unless the caller explicitly
        # requests publication of a non-conforming candidate.
        validation_failed = (
            sanitize_result.get("embedded_pdf_conversions_failed", 0) > 0
        )
        if validate or pdfua:
            logger.debug("Validating output with veraPDF")
            validation_failed = (
                _validate_pdfa_output(
                    final_output_temp_file,
                    level,
                    warnings,
                    validation_results,
                )
                or validation_failed
            )
        if pdfua:
            validation_failed = (
                _validate_pdfua_output(
                    final_output_temp_file,
                    warnings,
                    validation_results,
                )
                or validation_failed
            )

        if pdfua:
            if validation_failed:
                pdfua_status = PDFUAStatus.VALIDATION_FAILED
            elif review_findings:
                pdfua_status = PDFUAStatus.REVIEW_REQUIRED
            else:
                pdfua_status = PDFUAStatus.MACHINE_VALIDATED
        else:
            pdfua_status = PDFUAStatus.NOT_REQUESTED

        published = not (
            validation_failed
            and effective_publication_policy is PublicationPolicy.VALIDATED
        )
        if published:
            if validation_failed:
                warnings.append(_VALIDATION_PUBLICATION_WARNING)
            publish_staged_file(
                final_output_temp_file,
                output_path,
                final_output_snapshot,
                backup=final_output_temp_file.with_name(f"backup{output_path.suffix}"),
                require_absent=not _allow_output_overwrite,
            )
            final_output_temp_file = None
        else:
            warnings.append(_VALIDATION_WITHHELD_WARNING)

        processing_time = time.perf_counter() - start_time
        if validation_failed:
            logger.error(
                "Conversion validation failed%s: %s (%.2f seconds)",
                " after publication" if published else "",
                output_path,
                processing_time,
            )
        elif pdfua_status is PDFUAStatus.REVIEW_REQUIRED:
            logger.warning(
                "Conversion machine validation passed; author review required: "
                "%s (%.2f seconds)",
                output_path,
                processing_time,
            )
        else:
            logger.info(
                "Conversion successful: %s (%.2f seconds)",
                output_path,
                processing_time,
            )

        return ConversionResult(
            success=not validation_failed,
            input_path=input_path,
            output_path=output_path,
            level=level,
            warnings=warnings,
            processing_time=processing_time,
            error=(
                (_VALIDATION_FAILURE_ERROR if published else _VALIDATION_WITHHELD_ERROR)
                if validation_failed
                else None
            ),
            validation_failed=validation_failed,
            published=published,
            target_produced=not validation_failed,
            pdfua_status=pdfua_status,
            review_findings=tuple(review_findings),
            validation_results=tuple(validation_results),
            sanitization_stats=dict(sanitize_result),
            tagging_stats=dict(tagging_result),
            metadata_sources={
                "title": title_source,
                "language": language_source,
            },
            candidate_sha256=final_output_snapshot.sha256,
            candidate_size=final_output_snapshot.size,
        )

    except pikepdf.PasswordError:
        return _copy_encrypted_input(
            input_path,
            output_path,
            pdfa=pdfa,
            pdfua=pdfua,
            publish_unconverted=(
                effective_publication_policy is PublicationPolicy.ALWAYS
            ),
            start_time=start_time,
            allow_overwrite=_allow_output_overwrite,
        )

    except pikepdf.PdfError as e:
        error_msg = f"PDF processing error: {e}"
        logger.error(error_msg)
        raise ConversionError(error_msg) from e

    except (
        UnsupportedPDFError,
        FontEmbeddingError,
        OCRError,
        VeraPDFError,
        PermissionError,
    ):
        # Re-raise specific errors unchanged
        raise

    except ConversionError:
        # Re-raise ConversionError unchanged
        raise

    except Exception as e:
        error_msg = f"Unexpected error during conversion: {e}"
        logger.error(error_msg)
        raise ConversionError(error_msg) from e

    finally:
        # Cleanup: Close PDF if still open (e.g. after an exception)
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass

        if final_output_temp_file is not None:
            try:
                final_output_temp_file.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete staged PDF/A output: %s (%s)",
                    final_output_temp_file,
                    cleanup_error,
                )

        if final_output_temp_directory is not None:
            try:
                final_output_temp_directory.cleanup()
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete private PDF/A staging directory: %s (%s)",
                    final_output_temp_directory.name,
                    cleanup_error,
                )

        # Cleanup: Delete OCR temporary files
        ocr_cleanup_files = (
            (ocr_temp_file, "OCR temporary file"),
            (ocr_signature_temp_file, "OCR signature temporary file"),
            (ocr_clean_temp_file, "OCR annotation-stripped temporary file"),
            (ocr_merged_temp_file, "OCR annotation-merged temporary file"),
            (ocr_manifest_temp_file, "OCR semantic manifest temporary file"),
        )
        for cleanup_file, cleanup_label in ocr_cleanup_files:
            if cleanup_file is None or not cleanup_file.exists():
                continue
            try:
                cleanup_file.unlink()
                logger.debug("%s deleted: %s", cleanup_label, cleanup_file)
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete %s: %s (%s)",
                    cleanup_label,
                    cleanup_file,
                    cleanup_error,
                )


def convert_files(
    file_pairs: list[tuple[Path, Path]],
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    validate: bool = False,
    publication_policy: PublicationPolicy
    | Literal["always", "validated"]
    | None = None,
    document_language: str | None = None,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    force_overwrite: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]:
    """Converts a list of PDF files to PDF/A.

    Shared base for convert_directory().

    Args:
        file_pairs: List of (input_path, output_path) tuples.
        level: PDF/A conformance level (e.g. '2b', '3b').
        pdfa: If False, apply only requested OCR processing.
        pdfua: If True, also produce PDF/UA-1 output. This requires PDF/A-2a
            or PDF/A-3a.
        validate: If True, results are validated.
        publication_policy: Controls whether validation failures may be
            published. Validation defaults to fail-closed.
        document_language: Authoritative BCP 47 language tag applied to every
            document in the batch.
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``). Requires both model directories.
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
            Must be provided with ``ocr_recognition_model_dir``.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
            Must be provided with ``ocr_detection_model_dir``.
        ocr_force: If True, force OCR even on pages that already contain
            text. This cannot be combined with ``ocr_deskew``.
        ocr_deskew: If True, enable OCR and straighten scan-like,
            raster-dominant pages.
        ocr_rotate_pages: If True, enable OCR and normalize page orientation.
        ocr_execution_provider: ONNX Runtime provider for Paddle models:
            ``"cpu"`` (default), ``"directml"`` or ``"directml:<index>"``
            to select a specific adapter.
        ocr_layout: If True, order OCR lines by detected page columns.
        force_overwrite: If True, existing output files are overwritten.
            If False, existing outputs are skipped with an error result.
        preserve_stamps: If True, known proprietary stamp annotations are
            normalized to standard ``/Stamp`` annotations instead of being
            flattened into page content.
        allow_signature_invalidation: If True, convert signed PDFs even though
            conversion removes or invalidates their digital signatures.
        on_progress: Optional callback(current_idx, total, filename) called
            before each file.
        cancel_event: Optional threading.Event; when set, iteration stops.

    Returns:
        List of ConversionResult for all processed files.
    """
    _validate_ocr_configuration(
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        ocr_execution_provider=ocr_execution_provider,
        ocr_layout=ocr_layout,
    )
    if pdfa:
        level = validate_pdfa_level(level)
    _validate_pdfua_options(pdfa=pdfa, level=level, pdfua=pdfua)
    if not pdfa and validate:
        raise ConversionError("PDF/A validation cannot be used when pdfa=False")

    results: list[ConversionResult] = []
    total = len(file_pairs)

    input_identities = [_path_identity(input_path) for input_path, _ in file_pairs]
    output_identities = [_path_identity(output_path) for _, output_path in file_pairs]
    if len(set(output_identities)) != len(output_identities):
        raise ConversionError("Output paths in a batch must be unique")
    input_identities_set = set(input_identities)
    for (_, output_path), output_identity in zip(file_pairs, output_identities):
        if output_identity in input_identities_set:
            raise ConversionError(
                f"Batch output path overlaps an input path: {output_path}"
            )

    for idx, (input_path, output_path) in enumerate(file_pairs):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Conversion cancelled")
            break

        if on_progress is not None:
            on_progress(idx, total, input_path.name)

        # Overwrite protection
        if output_path.exists() and not force_overwrite:
            logger.warning(
                "Skipping %s: Output file already exists (%s)",
                input_path.name,
                output_path,
            )
            results.append(
                ConversionResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    level=level if pdfa else None,
                    error="Output file already exists",
                    published=False,
                    target_produced=False,
                    pdfua_status=(
                        PDFUAStatus.NOT_PRODUCED if pdfua else PDFUAStatus.NOT_REQUESTED
                    ),
                )
            )
            continue

        try:
            result = convert_to_pdfa(
                input_path=input_path,
                output_path=output_path,
                level=level,
                pdfa=pdfa,
                pdfua=pdfua,
                validate=validate,
                publication_policy=publication_policy,
                document_language=document_language,
                skip_any_pdfa=skip_any_pdfa,
                ocr_languages=ocr_languages,
                ocr_detection_model_dir=ocr_detection_model_dir,
                ocr_recognition_model_dir=ocr_recognition_model_dir,
                ocr_force=ocr_force,
                ocr_deskew=ocr_deskew,
                ocr_rotate_pages=ocr_rotate_pages,
                ocr_execution_provider=ocr_execution_provider,
                ocr_layout=ocr_layout,
                convert_calibrated=convert_calibrated,
                preserve_stamps=preserve_stamps,
                allow_signature_invalidation=allow_signature_invalidation,
                _allow_output_overwrite=force_overwrite,
            )
            results.append(result)

        except (
            ConversionError,
            UnsupportedPDFError,
            FontEmbeddingError,
            OCRError,
            PermissionError,
        ) as e:
            logger.error("Error for %s: %s", input_path.name, e)
            results.append(
                ConversionResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    level=level if pdfa else None,
                    error=str(e),
                    processing_time=0.0,
                    published=False,
                    target_produced=False,
                    pdfua_status=(
                        PDFUAStatus.NOT_PRODUCED if pdfua else PDFUAStatus.NOT_REQUESTED
                    ),
                )
            )

    return results


def convert_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    recursive: bool = False,
    validate: bool = False,
    publication_policy: PublicationPolicy
    | Literal["always", "validated"]
    | None = None,
    document_language: str | None = None,
    skip_any_pdfa: bool = False,
    show_progress: bool = True,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    force_overwrite: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]:
    """Converts all PDFs in a directory to PDF/A.

    Args:
        input_dir: Input directory with PDF files.
        output_dir: Optional output directory. If None, files are saved
            in the same directory as the input.
        level: PDF/A conformance level.
        pdfa: If False, apply only requested OCR processing.
        pdfua: If True, also produce PDF/UA-1 output. This requires PDF/A-2a
            or PDF/A-3a.
        recursive: If True, subdirectories are included.
        validate: If True, results are validated.
        publication_policy: Controls whether validation failures may be
            published. Validation defaults to fail-closed.
        document_language: Authoritative BCP 47 language tag applied to every
            document in the directory.
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        show_progress: If True, a progress bar is shown.
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``). Requires both model directories.
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
            Must be provided with ``ocr_recognition_model_dir``.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
            Must be provided with ``ocr_detection_model_dir``.
        ocr_force: If True, force OCR even on pages that already contain
            text. This cannot be combined with ``ocr_deskew``.
        ocr_deskew: If True, enable OCR and straighten scan-like,
            raster-dominant pages.
        ocr_rotate_pages: If True, enable OCR and normalize page orientation.
        ocr_execution_provider: ONNX Runtime provider for Paddle models:
            ``"cpu"`` (default), ``"directml"`` or ``"directml:<index>"``
            to select a specific adapter.
        ocr_layout: If True, order OCR lines by detected page columns.
        preserve_stamps: If True, known proprietary stamp annotations are
            normalized to standard ``/Stamp`` annotations instead of being
            flattened into page content.
        allow_signature_invalidation: If True, convert signed PDFs even though
            conversion removes or invalidates their digital signatures.
        force_overwrite: If True, existing output files are overwritten.

    Returns:
        List of ConversionResult for all processed files.

    Raises:
        ConversionError: If the input directory does not exist.
    """
    _validate_ocr_configuration(
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        ocr_execution_provider=ocr_execution_provider,
        ocr_layout=ocr_layout,
    )
    if pdfa:
        level = validate_pdfa_level(level)
    _validate_pdfua_options(pdfa=pdfa, level=level, pdfua=pdfua)
    if not pdfa and validate:
        raise ConversionError("PDF/A validation cannot be used when pdfa=False")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve() if output_dir is not None else None

    if not input_dir.is_dir():
        raise ConversionError(f"Directory does not exist: {input_dir}")

    # Find all PDFs with a case-insensitive suffix check on every platform.
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    pdf_files = sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() == ".pdf"
    )

    # A recursive search must not process files from a nested output tree.
    if (
        output_dir is not None
        and output_dir != input_dir
        and output_dir.is_relative_to(input_dir)
    ):
        pdf_files = [p for p in pdf_files if not p.is_relative_to(output_dir)]

    # When output goes to the same directory, exclude a generated output only
    # when its corresponding source is also present. A standalone source whose
    # name happens to end in the output suffix remains a valid input.
    if output_dir is None or output_dir == input_dir:
        output_suffix = "_pdfa" if pdfa else "_processed"
        source_stems = {(path.parent, path.stem) for path in pdf_files}
        pdf_files = [
            path
            for path in pdf_files
            if not (
                path.stem.endswith(output_suffix)
                and (
                    path.parent,
                    path.stem.removesuffix(output_suffix),
                )
                in source_stems
            )
        ]

    if not pdf_files:
        logger.warning("No PDF files found in: %s", input_dir)
        return []

    logger.info(
        "Found: %d PDF file(s) in %s%s",
        len(pdf_files),
        input_dir,
        " (recursive)" if recursive else "",
    )

    # Create output directory
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Compute file pairs (input, output)
    file_pairs: list[tuple[Path, Path]] = []
    for pdf_file in pdf_files:
        if output_dir is not None:
            if recursive:
                rel_path = pdf_file.relative_to(input_dir)
                out_subdir = output_dir / rel_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_path = generate_output_path(pdf_file, out_subdir, pdfa=pdfa)
            else:
                out_path = generate_output_path(pdf_file, output_dir, pdfa=pdfa)
        else:
            out_path = generate_output_path(pdf_file, pdfa=pdfa)
        file_pairs.append((pdf_file, out_path))

    # tqdm progress wrapper
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(file_pairs),
            desc="Converting" if pdfa else "Processing",
            unit="file",
            ncols=80,
        )

    def _on_progress(current_idx: int, total: int, filename: str) -> None:
        if progress_bar is not None:
            # The callback fires before each file, so current_idx equals the
            # number of already completed files.
            progress_bar.n = current_idx
            progress_bar.refresh()
            progress_bar.set_postfix_str(filename)

    results = convert_files(
        file_pairs=file_pairs,
        level=level,
        pdfa=pdfa,
        pdfua=pdfua,
        validate=validate,
        publication_policy=publication_policy,
        document_language=document_language,
        skip_any_pdfa=skip_any_pdfa,
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        ocr_execution_provider=ocr_execution_provider,
        ocr_layout=ocr_layout,
        force_overwrite=force_overwrite,
        on_progress=_on_progress if show_progress else None,
        convert_calibrated=convert_calibrated,
        preserve_stamps=preserve_stamps,
        allow_signature_invalidation=allow_signature_invalidation,
    )

    if progress_bar is not None:
        progress_bar.n = len(results)
        progress_bar.refresh()
        progress_bar.close()

    # Log summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    logger.info(
        "Directory conversion completed: %d successful, %d failed",
        successful,
        failed,
    )

    return results
