# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core logic for PDF to PDF/A conversion."""

# Standard Library
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Third Party
import pikepdf
from tqdm import tqdm

# Local
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
    _extract_existing_xmp,
    extract_pdf_info,
    remove_pdfx_identification,
    sync_metadata,
)
from .ocr import DEFAULT_OCR_FALLBACK_AFTER_SECONDS, DEFAULT_OCR_FALLBACK_QUALITY
from .sanitizers import (
    count_digital_signatures,
    sanitize_for_pdfa,
    sanitize_notdef_usage,
    sanitize_signatures,
    sanitize_structure_limits,
)
from .utils import get_required_pdf_version, is_pdf_encrypted, validate_pdfa_level
from .validator import detect_iso_standards, detect_pdfa_level
from .verapdf import validate_with_verapdf

if TYPE_CHECKING:
    from .ocr import OcrQuality

logger = logging.getLogger(__name__)

# Conformance level ranking: a > u > b
_CONFORMANCE_RANK = {"b": 0, "u": 1, "a": 2}

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


def _signature_invalidation_warning(count: int) -> str:
    """Build the warning emitted when signature invalidation is explicit."""
    return (
        f"{count} digital signature(s) will be removed/invalidated for PDF/A conversion"
    )


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


@dataclass
class ConversionResult:
    """Result of a PDF/A conversion.

    Attributes:
        success: True if the conversion was successful.
        input_path: Path to the input PDF.
        output_path: Path to the output PDF/A.
        level: PDF/A conformance level used.
        warnings: List of warnings during conversion.
        processing_time: Processing time in seconds.
        error: Error message if success=False.
        validation_failed: True if veraPDF validation failed.
        skipped: True if the original PDF was copied through unchanged.
    """

    success: bool
    input_path: Path
    output_path: Path
    level: str
    warnings: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    error: str | None = None
    validation_failed: bool = False
    skipped: bool = False


def generate_output_path(
    input_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Generates the output path for a converted PDF.

    Args:
        input_path: Path to the input PDF.
        output_dir: Optional output directory.

    Returns:
        Path for the output PDF/A.
    """
    output_name = f"{input_path.stem}_pdfa.pdf"
    if output_dir is not None:
        return output_dir / output_name
    return input_path.parent / output_name


def _copy_input_to_output(input_path: Path, output_path: Path) -> None:
    """Copy the input file to the requested output location.

    Overwrites an existing output file, matching the behavior of the
    regular conversion path. Overwrite protection is enforced by the
    callers (CLI and convert_files()).

    Args:
        input_path: Original file path.
        output_path: Destination file path.
    """
    if input_path.resolve() == output_path.resolve():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input_path), str(output_path))


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
    except Exception as e:
        logger.warning("Could not read file for %%%%EOF check: %s", e)
        return False

    eof_marker = b"%%EOF"
    last_eof = data.rfind(eof_marker)
    if last_eof == -1:
        logger.warning("No %%%%EOF marker found in output file")
        return False

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
    except Exception as e:
        logger.warning("Could not truncate trailing data: %s", e)
        return False

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
    except Exception as e:
        logger.warning("Could not read header for binary comment check: %s", e)
        return False

    # Locate end of first line (%PDF-x.y)
    nl = header.find(b"\n")
    if nl == -1:
        nl = header.find(b"\r")
    if nl == -1:
        return False

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
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        with pikepdf.open(output_path) as pdf:
            pdf.save(tmp, **_get_pdfa_save_settings_for_version(required_version))
        os.replace(str(tmp), str(output_path))
    except Exception as e:
        logger.warning("Could not add binary comment: %s", e)
        try:
            tmp.unlink()
        except Exception:
            pass
        return False

    return True


def _verify_file_structure(output_path: Path, required_version: str) -> None:
    """Lightweight post-save verification of PDF file structure.

    Checks that the output file has the expected PDF header and a /ID
    array in the trailer.  Logs warnings on failure but does not
    raise — the file may still be valid.

    Args:
        output_path: Path to the saved PDF file.
        required_version: Expected PDF version string (e.g. ``"1.7"``).
    """
    try:
        with open(output_path, "rb") as f:
            header = f.read(20)
    except Exception as e:
        logger.warning("Post-save verification: could not read file: %s", e)
        return

    # 1. Check header starts with %PDF-<version>
    expected_header = f"%PDF-{required_version}".encode("ascii")
    if not header.startswith(expected_header):
        actual = header[:15].decode("ascii", errors="replace")
        logger.warning(
            "Post-save verification: file header '%s' does not start "
            "with expected '%s'",
            actual,
            expected_header.decode("ascii"),
        )

    # 2. Check trailer /ID
    try:
        with pikepdf.open(output_path) as check_pdf:
            id_array = check_pdf.trailer.get("/ID")
            if id_array is None or len(id_array) != 2:
                logger.warning(
                    "Post-save verification: trailer /ID missing or "
                    "does not have 2 elements"
                )
    except Exception as e:
        logger.warning("Post-save verification: could not reopen file: %s", e)


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


def _has_annotations(pdf_path: Path) -> bool:
    """Check whether any page in the PDF contains annotations.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ``True`` if at least one page has a non-empty ``/Annots`` array.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    annots = page.get("/Annots")
                    if annots is not None and len(annots) > 0:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


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

            try:
                pdf.save(clean_path)
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

                logger.warning(
                    "Saving annotation-stripped PDF failed; retrying without "
                    "document metadata: %s",
                    exc,
                )
                pdf.save(clean_path)
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

        try:
            pdf.save(prepared_path)
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

            logger.warning(
                "Saving signature-sanitized OCR source failed; retrying without "
                "document metadata: %s",
                exc,
            )
            pdf.save(prepared_path)

    return sig_result


def _restore_annotations_after_ocr(
    original_path: Path, ocr_path: Path, output_path: Path
) -> int:
    """Re-inject original annotations into an OCR-processed PDF.

    Copies entire ``/Annots`` arrays (preserving internal cross-references
    like ``/Popup`` and ``/IRT`` chains) from the original PDF into the
    OCR output via ``copy_foreign``.  Also restores ``/AcroForm`` if present.

    Args:
        original_path: Path to the original PDF (with annotations).
        ocr_path: Path to the OCR-processed PDF (without annotations).
        output_path: Path where the merged result will be saved.

    Returns:
        Total number of annotations restored (0 on failure or mismatch).
    """
    original_pdf = None
    ocr_pdf = None
    try:
        original_pdf = pikepdf.open(original_path)
        ocr_pdf = pikepdf.open(ocr_path)

        if len(original_pdf.pages) != len(ocr_pdf.pages):
            logger.warning(
                "Page count mismatch after OCR (%d vs %d), "
                "skipping annotation restoration",
                len(original_pdf.pages),
                len(ocr_pdf.pages),
            )
            return 0

        total_restored = 0
        for i, (orig_page, ocr_page) in enumerate(
            zip(original_pdf.pages, ocr_pdf.pages)
        ):
            try:
                annots = orig_page.get("/Annots")
                if annots is None or len(annots) == 0:
                    continue
            except Exception:
                continue

            # Ensure annots is indirect so copy_foreign can track it.
            annots_ref = original_pdf.make_indirect(annots)
            copied_annots = ocr_pdf.copy_foreign(annots_ref)

            # Remap /P (parent page) references to the target page
            for annot in copied_annots:
                try:
                    resolved = annot.get_object()
                    if "/P" in resolved:
                        resolved["/P"] = ocr_page.obj
                except Exception:
                    continue

            ocr_page.obj["/Annots"] = copied_annots
            total_restored += len(copied_annots)

        # Restore /AcroForm if present in original
        if "/AcroForm" in original_pdf.Root:
            acroform = original_pdf.Root["/AcroForm"]
            acroform_ref = original_pdf.make_indirect(acroform)
            ocr_pdf.Root["/AcroForm"] = ocr_pdf.copy_foreign(acroform_ref)

        ocr_pdf.save(output_path)
        return total_restored
    except Exception as exc:
        logger.warning("Could not restore annotations after OCR: %s", exc)
        return 0
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
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_quality: "OcrQuality | None" = None,
    ocr_fallback_quality: "OcrQuality | None" = DEFAULT_OCR_FALLBACK_QUALITY,
    ocr_fallback_after_seconds: float | None = DEFAULT_OCR_FALLBACK_AFTER_SECONDS,
    ocr_force: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> ConversionResult:
    """Converts a PDF file to the PDF/A format.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the output PDF/A.
        level: PDF/A conformance level ('2b' or '3b').
        validate: If True, the result is validated.
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        ocr_languages: Optional list of Tesseract language codes
            (e.g., ``["deu", "eng"]``).  If specified, OCR is applied to
            image-based pages.
        ocr_quality: OCR quality preset. If None, uses OcrQuality.DEFAULT.
        ocr_fallback_quality: Faster OCR quality to retry with if OCR leaves
            the selected preset takes too long. Use None to disable.
        ocr_fallback_after_seconds: Per-page runtime budget for OCR fallback
            (total threshold is this value times the page count).
            Use None to disable time-based retry.
        ocr_force: If True, force OCR even on pages that already contain
            text by using ocrmypdf's ``redo_ocr`` mode. Options
            incompatible with ``redo_ocr`` are disabled automatically.
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
    level = validate_pdfa_level(level)
    start_time = time.perf_counter()
    warnings: list[str] = []
    ocr_temp_file: Path | None = None
    ocr_signature_temp_file: Path | None = None
    ocr_clean_temp_file: Path | None = None
    ocr_merged_temp_file: Path | None = None
    pdf: pikepdf.Pdf | None = None
    source_info: dict | None = None
    source_xmp_tree = None

    force_ocr_requested = ocr_force and ocr_languages is not None

    logger.info(
        "Starting conversion: %s -> %s (PDF/A-%s)",
        input_path,
        output_path,
        level,
    )

    try:
        # 0. Check if PDF is already PDF/A compliant (before OCR)
        with pikepdf.open(input_path) as check_pdf:
            if is_pdf_encrypted(check_pdf):
                processing_time = time.perf_counter() - start_time
                warning = "Conversion skipped: PDF is encrypted and cannot be converted"
                logger.warning("%s: %s", warning, input_path)
                _copy_input_to_output(input_path, output_path)
                return ConversionResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    level=level,
                    warnings=[warning],
                    processing_time=processing_time,
                    skipped=True,
                )
            source_info = extract_pdf_info(check_pdf)
            source_xmp_tree = _extract_existing_xmp(check_pdf)
            signature_count = count_digital_signatures(check_pdf)
            if signature_count > 0 and not allow_signature_invalidation:
                processing_time = time.perf_counter() - start_time
                logger.warning("%s: %s", _SIGNATURE_SKIP_WARNING, input_path)
                _copy_input_to_output(input_path, output_path)
                return ConversionResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    level=level,
                    warnings=[_SIGNATURE_SKIP_WARNING],
                    processing_time=processing_time,
                    skipped=True,
                )
            if signature_count > 0:
                warnings.append(_signature_invalidation_warning(signature_count))
            detected_level = detect_pdfa_level(check_pdf)

        if detected_level is not None and force_ocr_requested:
            logger.debug(
                "Bypassing PDF/A pre-check skip because forced OCR was requested"
            )

        if detected_level is not None and not force_ocr_requested:
            should_validate_for_skip = skip_any_pdfa
            if not should_validate_for_skip:
                level_cmp = _compare_pdfa_levels(detected_level, level)
                should_validate_for_skip = level_cmp >= 0

            if should_validate_for_skip:
                try:
                    verapdf_result = validate_with_verapdf(
                        input_path,
                        flavour=detected_level,
                        non_compliant_log_level=logging.WARNING,
                    )
                except VeraPDFError:
                    logger.debug(
                        "veraPDF not available, skipping PDF/A pre-check for %s",
                        input_path,
                    )
                    verapdf_result = None

                if verapdf_result is not None and verapdf_result.compliant:
                    processing_time = time.perf_counter() - start_time
                    logger.info(
                        "Skipping conversion: PDF is already valid PDF/A-%s",
                        detected_level,
                    )
                    _copy_input_to_output(input_path, output_path)
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
                    )
                elif verapdf_result is not None:
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

        # 1. Optional: Perform OCR
        actual_input = input_path
        if ocr_languages is not None:
            from .ocr import OcrQuality, apply_ocr, is_ocr_available

            if not is_ocr_available():
                warnings.append("OCR not available - pip install pdftopdfa[ocr]")
            else:
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".pdf", prefix=f".{input_path.stem}_ocr_"
                )
                os.close(fd)
                ocr_temp_file = Path(tmp_path)
                effective_quality = (
                    ocr_quality if ocr_quality is not None else OcrQuality.DEFAULT
                )
                ocr_source_base = input_path

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
                                sig_result["signatures_found"]
                            )
                        )
                else:
                    try:
                        ocr_signature_temp_file.unlink()
                    except Exception:
                        pass
                    ocr_signature_temp_file = None

                # Strip annotations before OCR so they are not
                # rasterized into page images.
                preserve_annots = _has_annotations(ocr_source_base)
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
                    ocr_languages,
                    quality=effective_quality,
                    fallback_quality=ocr_fallback_quality,
                    fallback_after_seconds=ocr_fallback_after_seconds,
                    force=ocr_force,
                )

                # Re-inject original annotations into OCR output.
                if preserve_annots:
                    fd3, merged_tmp = tempfile.mkstemp(
                        suffix=".pdf",
                        prefix=f".{input_path.stem}_merged_",
                    )
                    os.close(fd3)
                    ocr_merged_temp_file = Path(merged_tmp)
                    count = _restore_annotations_after_ocr(
                        ocr_source_base, ocr_temp_file, ocr_merged_temp_file
                    )
                    if count > 0:
                        os.replace(str(ocr_merged_temp_file), str(ocr_temp_file))
                        logger.debug("%d annotation(s) preserved through OCR", count)
                        warnings.append(f"{count} annotation(s) preserved through OCR")
                    else:
                        try:
                            ocr_merged_temp_file.unlink()
                        except Exception:
                            pass
                    ocr_merged_temp_file = None

                # Clean up the stripped copy.
                if ocr_clean_temp_file is not None:
                    try:
                        ocr_clean_temp_file.unlink()
                    except Exception:
                        pass
                    ocr_clean_temp_file = None

                actual_input = ocr_temp_file
                lang_str = "+".join(ocr_languages)
                warnings.append(f"OCR performed (languages: {lang_str})")

        # Validate that input and output are not the same file
        if actual_input.resolve() == output_path.resolve():
            raise ConversionError(f"Input and output paths must differ: {actual_input}")

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

        # 7. Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 8. Save PDF with minimum version.
        #
        # Keep output non-linearized because QPDF linearization can still
        # produce invalid /Length values on generated hint streams
        # (rule 6.1.7.1) for specific inputs.
        logger.debug("Saving PDF/A: %s", output_path)
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

        save_pdfa(pdf, output_path, level, verify=not validate)
        pdf.close()
        pdf = None

        processing_time = time.perf_counter() - start_time

        # 9. Optional: Validate
        validation_failed = False
        if validate:
            logger.debug("Validating output with veraPDF")
            try:
                verapdf_result = validate_with_verapdf(path=output_path, flavour=level)
            except VeraPDFError as e:
                logger.warning("veraPDF validation not available: %s", e)
                warnings.append("Validation skipped: veraPDF not available")
                verapdf_result = None

            if verapdf_result is not None and not verapdf_result.compliant:
                validation_failed = True
                for error in verapdf_result.errors:
                    warnings.append(f"Validation: {error}")

        logger.info(
            "Conversion successful: %s (%.2f seconds)",
            output_path,
            processing_time,
        )

        return ConversionResult(
            success=True,
            input_path=input_path,
            output_path=output_path,
            level=level,
            warnings=warnings,
            processing_time=processing_time,
            validation_failed=validation_failed,
        )

    except pikepdf.PdfError as e:
        error_msg = f"PDF processing error: {e}"
        logger.error(error_msg)
        raise ConversionError(error_msg) from e

    except (UnsupportedPDFError, FontEmbeddingError, OCRError):
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

        # Cleanup: Delete OCR temporary files
        ocr_cleanup_files = (
            (ocr_temp_file, "OCR temporary file"),
            (ocr_signature_temp_file, "OCR signature temporary file"),
            (ocr_clean_temp_file, "OCR annotation-stripped temporary file"),
            (ocr_merged_temp_file, "OCR annotation-merged temporary file"),
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
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_quality: "OcrQuality | None" = None,
    ocr_fallback_quality: "OcrQuality | None" = DEFAULT_OCR_FALLBACK_QUALITY,
    ocr_fallback_after_seconds: float | None = DEFAULT_OCR_FALLBACK_AFTER_SECONDS,
    ocr_force: bool = False,
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
        validate: If True, results are validated.
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        ocr_languages: Optional list of Tesseract language codes
            (e.g., ``["deu", "eng"]``).
        ocr_quality: OCR quality preset.
        ocr_fallback_quality: Faster OCR quality to retry with if OCR leaves
            the selected preset takes too long.
        ocr_fallback_after_seconds: Per-page runtime budget for OCR fallback
            (total threshold is this value times the page count).
        ocr_force: If True, force OCR even on pages that already contain
            text. Options incompatible with ocrmypdf's ``redo_ocr``
            mode are disabled automatically.
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
    results: list[ConversionResult] = []
    total = len(file_pairs)

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
                    level=level,
                    error="Output file already exists",
                )
            )
            continue

        try:
            result = convert_to_pdfa(
                input_path=input_path,
                output_path=output_path,
                level=level,
                validate=validate,
                skip_any_pdfa=skip_any_pdfa,
                ocr_languages=ocr_languages,
                ocr_quality=ocr_quality,
                ocr_fallback_quality=ocr_fallback_quality,
                ocr_fallback_after_seconds=ocr_fallback_after_seconds,
                ocr_force=ocr_force,
                convert_calibrated=convert_calibrated,
                preserve_stamps=preserve_stamps,
                allow_signature_invalidation=allow_signature_invalidation,
            )
            results.append(result)

        except (
            ConversionError,
            UnsupportedPDFError,
            FontEmbeddingError,
            OCRError,
        ) as e:
            logger.error("Error for %s: %s", input_path.name, e)
            results.append(
                ConversionResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    level=level,
                    error=str(e),
                    processing_time=0.0,
                )
            )

    return results


def convert_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    level: str = "3b",
    *,
    recursive: bool = False,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    show_progress: bool = True,
    ocr_languages: list[str] | None = None,
    ocr_quality: "OcrQuality | None" = None,
    ocr_fallback_quality: "OcrQuality | None" = DEFAULT_OCR_FALLBACK_QUALITY,
    ocr_fallback_after_seconds: float | None = DEFAULT_OCR_FALLBACK_AFTER_SECONDS,
    ocr_force: bool = False,
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
        level: PDF/A conformance level ('2b' or '3b').
        recursive: If True, subdirectories are included.
        validate: If True, results are validated.
        skip_any_pdfa: If True, skip conversion for any input that veraPDF
            validates as compliant PDF/A, regardless of target level.
        show_progress: If True, a progress bar is shown.
        ocr_languages: Optional list of Tesseract language codes
            (e.g., ``["deu", "eng"]``).
            If specified, OCR is applied to image-based pages.
        ocr_quality: OCR quality preset.
        ocr_fallback_quality: Faster OCR quality to retry with if OCR leaves
            the selected preset takes too long.
        ocr_fallback_after_seconds: Per-page runtime budget for OCR fallback
            (total threshold is this value times the page count).
        ocr_force: If True, force OCR even on pages that already contain
            text. Options incompatible with ocrmypdf's ``redo_ocr``
            mode are disabled automatically.
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
    if not input_dir.is_dir():
        raise ConversionError(f"Directory does not exist: {input_dir}")

    # Find all PDFs
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(input_dir.glob(pattern))

    # When output goes to the same directory, exclude previous conversion outputs
    if output_dir is None or output_dir.resolve() == input_dir.resolve():
        pdf_files = [p for p in pdf_files if not p.stem.endswith("_pdfa")]

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
                out_path = out_subdir / f"{pdf_file.stem}_pdfa.pdf"
            else:
                out_path = generate_output_path(pdf_file, output_dir)
        else:
            out_path = generate_output_path(pdf_file)
        file_pairs.append((pdf_file, out_path))

    # tqdm progress wrapper
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(file_pairs),
            desc="Converting",
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
        validate=validate,
        skip_any_pdfa=skip_any_pdfa,
        ocr_languages=ocr_languages,
        ocr_quality=ocr_quality,
        ocr_fallback_quality=ocr_fallback_quality,
        ocr_fallback_after_seconds=ocr_fallback_after_seconds,
        ocr_force=ocr_force,
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
