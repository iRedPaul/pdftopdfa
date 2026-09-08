# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Click-based CLI for pdftopdfa.

This module provides the command-line interface for
converting PDF files to the PDF/A format.
"""

# Standard Library
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

# Third Party
import click
from colorama import Fore, Style, init

# Local
from . import __version__
from ._ocr_runtime import validate_ocr_execution_provider
from .converter import (
    ConversionResult,
    PDFUAStatus,
    PublicationPolicy,
    convert_directory,
    convert_to_pdfa,
    generate_output_path,
)
from .exceptions import (
    ConversionError,
    FontEmbeddingError,
    OCRError,
    UnsupportedPDFError,
    VeraPDFError,
)
from .utils import setup_logging

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_CONVERSION_FAILED = 3
EXIT_VALIDATION_FAILED = 4
EXIT_PERMISSION_ERROR = 5
EXIT_REVIEW_REQUIRED = 6

logger = logging.getLogger(__name__)
_VALIDATION_PREFIXES = ("Validation:", "PDF/UA validation:")


def _write_audit_report(
    path: Path,
    results: list[ConversionResult],
    *,
    fatal_error: dict[str, object] | None = None,
) -> None:
    """Atomically persist full conversion and validation evidence as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "results": [result.to_dict(include_raw_validation=True) for result in results],
    }
    if fatal_error is not None:
        payload["fatal_error"] = fatal_error
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _encode_for_console(text: str, *, err: bool = False) -> str:
    """Coerces text to the active console encoding when needed."""
    stream = sys.stderr if err else sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text

    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return text.encode(encoding, errors="replace").decode(encoding)

    return text


def _status_message(prefix: str, msg: str, *, color: str, err: bool = False) -> str:
    """Builds a console-safe status message."""
    return _encode_for_console(f"{color}{prefix}{Style.RESET_ALL} {msg}", err=err)


def print_success(msg: str) -> None:
    """Prints a success message in green.

    Args:
        msg: The message to output.
    """
    click.echo(_status_message("Success:", msg, color=Fore.GREEN))


def print_error(msg: str) -> None:
    """Prints an error message in red.

    Args:
        msg: The error message to output.
    """
    click.echo(_status_message("Error:", msg, color=Fore.RED, err=True), err=True)


def print_warning(msg: str) -> None:
    """Prints a warning in yellow.

    Args:
        msg: The warning to output.
    """
    click.echo(_status_message("Warning:", msg, color=Fore.YELLOW))


def _print_result(result: ConversionResult, quiet: bool) -> None:
    """Prints the conversion result in a formatted way.

    Args:
        result: The conversion result.
        quiet: If True, only output errors.
    """
    if result.validation_failed:
        for warning in result.warnings:
            if warning.startswith(_VALIDATION_PREFIXES):
                print_error(warning)

    if result.success:
        if not quiet:
            if result.level is None:
                conversion_skipped = result.skipped and any(
                    warning.startswith("Conversion skipped:")
                    for warning in result.warnings
                )
                action = (
                    "Skipped"
                    if conversion_skipped
                    else ("Copied unchanged" if result.skipped else "Processed")
                )
                details = f"{result.processing_time:.2f}s"
            else:
                action = "Skipped" if result.skipped else "Converted to PDF/A"
                details = f"PDF/A-{result.level}, {result.processing_time:.2f}s"
            print_success(
                f"{action}: {result.input_path.name} -> "
                f"{result.output_path.name} ({details})"
            )
            for warning in result.warnings:
                if result.validation_failed and warning.startswith(
                    _VALIDATION_PREFIXES
                ):
                    continue
                print_warning(warning)
            if result.review_required:
                print_warning(
                    "PDF/UA machine validation passed, but author review is required"
                )
    else:
        if not quiet:
            for warning in result.warnings:
                if result.validation_failed and warning.startswith(
                    _VALIDATION_PREFIXES
                ):
                    continue
                print_warning(warning)
        print_error(f"{result.input_path.name}: {result.error}")


def _ocr_execution_provider_callback(
    ctx: click.Context,
    param: click.Parameter,
    value: str,
) -> str:
    """Validate --ocr-execution-provider, which click.Choice cannot express."""
    try:
        return validate_ocr_execution_provider(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


@click.command()
@click.argument("input_path", required=False, type=click.Path(exists=True))
@click.argument("output", required=False, type=click.Path())
@click.option(
    "-l",
    "--level",
    type=click.Choice(["2a", "2b", "2u", "3a", "3b", "3u"]),
    default="3b",
    help=("PDF/A conformance level: a=accessible, b=basic, u=Unicode (default: 3b)"),
)
@click.option(
    "-v",
    "--validate",
    "do_validate",
    is_flag=True,
    help=(
        "Validate PDF/A after conversion (PDF/UA is always validated; "
        "-v is not verbose, use --verbose)"
    ),
)
@click.option(
    "--pdfua",
    is_flag=True,
    help=(
        "Also produce PDF/UA-1 (requires --level 2a or 3a; both profiles "
        "are always submitted to veraPDF validation)."
    ),
)
@click.option(
    "--publish-noncompliant",
    is_flag=True,
    help=(
        "Publish the candidate even if requested validation fails. By default, "
        "validated conversions are fail-closed."
    ),
)
@click.option(
    "--document-title",
    help="Authoritative document title (single-file conversion only).",
)
@click.option(
    "--document-language",
    help="Authoritative BCP 47 document language, for example de or en-GB.",
)
@click.option(
    "--audit-report",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write machine-readable conversion and veraPDF evidence as JSON.",
)
@click.option(
    "--no-pdfa",
    "pdfa",
    is_flag=True,
    flag_value=False,
    default=True,
    help="Apply requested OCR processing without converting to PDF/A.",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Process directories recursively",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite existing files",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Only output errors",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Detailed output",
)
@click.option(
    "--ocr",
    "ocr_enabled",
    is_flag=True,
    default=False,
    help="Enable OCR for image-based PDFs "
    "(requires both PP-OCRv6 model-directory options; "
    "uses language from --ocr-lang, default: en; "
    "disables compliant-PDF/A skip checks).",
)
@click.option(
    "--ocr-lang",
    "ocr_lang",
    default="en",
    help="PaddleOCR language and recognition script (default: en). Examples: de, de+en",
)
@click.option(
    "--ocr-detection-model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="PP-OCRv6 Medium detection model directory; "
    "use with --ocr-recognition-model-dir.",
)
@click.option(
    "--ocr-recognition-model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="PP-OCRv6 Medium recognition model directory; "
    "use with --ocr-detection-model-dir.",
)
@click.option(
    "--ocr-execution-provider",
    metavar="[cpu|directml|directml:INDEX]",
    default="cpu",
    show_default=True,
    callback=_ocr_execution_provider_callback,
    help="ONNX Runtime provider for all Paddle OCR models; "
    "directml:INDEX selects a specific GPU.",
)
@click.option(
    "--ocr-layout",
    is_flag=True,
    help="Order OCR lines by detected page columns.",
)
@click.option(
    "--ocr-figure-text",
    is_flag=True,
    help=(
        "Use sufficiently confident OCR text as review-required ActualText for "
        "otherwise undescribed image Figures and mark OCR-rejected Figures as "
        "artifacts; requires PDF/A-2a or PDF/A-3a."
    ),
)
@click.option(
    "--ocr-force",
    "ocr_force",
    is_flag=True,
    default=False,
    help="Force OCR even on pages that already contain text "
    "(removes existing OCR layer and re-applies). Implies --ocr.",
)
@click.option(
    "--deskew",
    is_flag=True,
    default=False,
    help="Straighten scan-like, raster-dominant pages. Implies --ocr.",
)
@click.option(
    "--rotate-pages",
    is_flag=True,
    default=False,
    help="Automatically orient pages with the bundled Paddle model. Implies --ocr.",
)
@click.option(
    "--convert-calibrated/--no-convert-calibrated",
    default=True,
    help="Convert CalGray/CalRGB color spaces to ICCBased (default: enabled)",
)
@click.option(
    "--preserve-stamps",
    is_flag=True,
    help=(
        "Convert known proprietary stamp annotations to standard PDF Stamp "
        "annotations instead of flattening them"
    ),
)
@click.option(
    "--skip-any-pdfa",
    is_flag=True,
    help=(
        "Skip files that veraPDF validates as any compliant PDF/A, "
        "regardless of target level"
    ),
)
@click.option(
    "--allow-signature-invalidation",
    is_flag=True,
    help=(
        "Convert digitally signed PDFs even though conversion removes or "
        "invalidates their signatures"
    ),
)
@click.version_option(version=__version__)
def main(
    input_path: str | None,
    output: str | None,
    level: str,
    do_validate: bool,
    pdfua: bool,
    publish_noncompliant: bool,
    document_title: str | None,
    document_language: str | None,
    audit_report: Path | None,
    pdfa: bool,
    recursive: bool,
    force: bool,
    quiet: bool,
    verbose: bool,
    ocr_enabled: bool,
    ocr_force: bool,
    deskew: bool,
    rotate_pages: bool,
    ocr_lang: str,
    ocr_detection_model_dir: Path | None,
    ocr_recognition_model_dir: Path | None,
    ocr_execution_provider: str,
    ocr_layout: bool,
    ocr_figure_text: bool,
    convert_calibrated: bool,
    preserve_stamps: bool,
    skip_any_pdfa: bool,
    allow_signature_invalidation: bool,
) -> None:
    """Converts PDFs to PDF/A or applies OCR processing with --no-pdfa.

    INPUT is the path to the input PDF or a directory.
    OUTPUT is optionally the path for the output PDF.
    """
    # Initialize colorama for Windows compatibility
    init()

    if input_path is None:
        click.echo(click.get_current_context().get_help())
        sys.exit(EXIT_GENERAL_ERROR)

    # Configure logging
    setup_logging(verbose=verbose, quiet=quiet)

    input_path_obj = Path(input_path)

    if audit_report is not None and audit_report.suffix.lower() != ".json":
        raise click.UsageError("--audit-report must use a .json filename")
    if audit_report is not None:
        protected_paths = {input_path_obj.resolve()}
        if output is not None:
            protected_paths.add(Path(output).resolve())
        elif input_path_obj.is_file():
            protected_paths.add(
                generate_output_path(input_path_obj, pdfa=pdfa).resolve()
            )
        if audit_report.resolve() in protected_paths:
            raise click.UsageError(
                "Audit report path must differ from input and output"
            )

    def raise_usage_error(message: str) -> NoReturn:
        error = click.UsageError(message)
        if audit_report is not None:
            failure: dict[str, object] = {
                "error_type": type(error).__name__,
                "message": message,
                "exit_code": error.exit_code,
                "input_path": str(input_path_obj),
            }
            if output is not None:
                failure["output_path"] = output
            try:
                _write_audit_report(audit_report, [], fatal_error=failure)
            except Exception as audit_error:
                logger.exception("Could not write failure audit report")
                print_error(f"Could not write audit report: {audit_error}")
        raise error

    if pdfua and not pdfa:
        raise_usage_error("--pdfua cannot be combined with --no-pdfa")
    if pdfua and level not in {"2a", "3a"}:
        raise_usage_error("--pdfua requires --level 2a or 3a")
    if not pdfa and do_validate:
        raise_usage_error("--validate cannot be combined with --no-pdfa")
    if not pdfa and (document_title is not None or document_language is not None):
        raise_usage_error("Document metadata options cannot be combined with --no-pdfa")
    if ocr_figure_text and (not pdfa or level not in {"2a", "3a"}):
        raise_usage_error("--ocr-figure-text requires --level 2a or 3a")
    if publish_noncompliant and not (do_validate or pdfua):
        raise_usage_error("--publish-noncompliant requires --validate or --pdfua")
    if input_path_obj.is_dir() and document_title is not None:
        raise_usage_error("--document-title is only valid for a single PDF")

    if deskew and ocr_force:
        raise_usage_error("--deskew cannot be combined with --ocr-force")

    model_pair_complete = (
        ocr_detection_model_dir is not None and ocr_recognition_model_dir is not None
    )
    if (ocr_detection_model_dir is None) != (ocr_recognition_model_dir is None):
        raise_usage_error(
            "--ocr-detection-model-dir and --ocr-recognition-model-dir "
            "must be provided together"
        )
    if (
        ocr_enabled
        or ocr_force
        or deskew
        or rotate_pages
        or ocr_execution_provider != "cpu"
        or ocr_layout
        or ocr_figure_text
    ):
        if not model_pair_complete:
            raise_usage_error(
                "OCR requires --ocr-detection-model-dir and --ocr-recognition-model-dir"
            )
    if model_pair_complete:
        ocr_enabled = True
    ocr_languages = ocr_lang.split("+") if ocr_enabled else None
    if ocr_languages is not None:
        from .ocr import validate_ocr_languages

        try:
            validate_ocr_languages(ocr_languages)
        except ValueError as exc:
            raise_usage_error(str(exc))

    fatal_error: Exception | None = None
    try:
        if input_path_obj.is_file():
            # Convert single file
            exit_code = _convert_single_file(
                input_path_obj,
                output,
                level,
                do_validate,
                force,
                quiet,
                pdfa=pdfa,
                pdfua=pdfua,
                publication_policy=(
                    PublicationPolicy.ALWAYS if publish_noncompliant else None
                ),
                document_title=document_title,
                document_language=document_language,
                audit_report=audit_report,
                ocr_languages=ocr_languages,
                ocr_detection_model_dir=ocr_detection_model_dir,
                ocr_recognition_model_dir=ocr_recognition_model_dir,
                ocr_force=ocr_force,
                ocr_deskew=deskew,
                ocr_rotate_pages=rotate_pages,
                ocr_execution_provider=ocr_execution_provider,
                ocr_layout=ocr_layout,
                ocr_figure_text=ocr_figure_text,
                convert_calibrated=convert_calibrated,
                preserve_stamps=preserve_stamps,
                skip_any_pdfa=skip_any_pdfa,
                allow_signature_invalidation=allow_signature_invalidation,
            )
        elif input_path_obj.is_dir():
            # Convert directory
            exit_code = _convert_directory(
                input_path_obj,
                output,
                level,
                do_validate,
                force,
                recursive,
                quiet,
                pdfa=pdfa,
                pdfua=pdfua,
                publication_policy=(
                    PublicationPolicy.ALWAYS if publish_noncompliant else None
                ),
                document_language=document_language,
                audit_report=audit_report,
                ocr_languages=ocr_languages,
                ocr_detection_model_dir=ocr_detection_model_dir,
                ocr_recognition_model_dir=ocr_recognition_model_dir,
                ocr_force=ocr_force,
                ocr_deskew=deskew,
                ocr_rotate_pages=rotate_pages,
                ocr_execution_provider=ocr_execution_provider,
                ocr_layout=ocr_layout,
                ocr_figure_text=ocr_figure_text,
                convert_calibrated=convert_calibrated,
                preserve_stamps=preserve_stamps,
                skip_any_pdfa=skip_any_pdfa,
                allow_signature_invalidation=allow_signature_invalidation,
            )
        else:
            print_error(f"Invalid path: {input_path}")
            exit_code = EXIT_FILE_NOT_FOUND
            fatal_error = FileNotFoundError(f"Invalid path: {input_path}")

    except FileNotFoundError as e:
        print_error(str(e))
        exit_code = EXIT_FILE_NOT_FOUND
        fatal_error = e
    except PermissionError as e:
        print_error(f"Access denied: {e}")
        exit_code = EXIT_PERMISSION_ERROR
        fatal_error = e
    except (
        ConversionError,
        UnsupportedPDFError,
        FontEmbeddingError,
        OCRError,
        VeraPDFError,
    ) as e:
        print_error(str(e))
        exit_code = EXIT_CONVERSION_FAILED
        fatal_error = e
    except Exception as e:
        logger.exception("Unexpected error")
        print_error(f"Unexpected error: {e}")
        exit_code = EXIT_GENERAL_ERROR
        fatal_error = e

    if audit_report is not None and fatal_error is not None:
        failure = {
            "error_type": type(fatal_error).__name__,
            "message": str(fatal_error),
            "exit_code": exit_code,
            "input_path": str(input_path_obj),
        }
        if output is not None:
            failure["output_path"] = output
        try:
            _write_audit_report(audit_report, [], fatal_error=failure)
        except Exception as audit_error:
            logger.exception("Could not write failure audit report")
            print_error(f"Could not write audit report: {audit_error}")

    sys.exit(exit_code)


def _convert_single_file(
    input_path: Path,
    output: str | None,
    level: str,
    do_validate: bool,
    force: bool,
    quiet: bool,
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    publication_policy: PublicationPolicy | None = None,
    document_title: str | None = None,
    document_language: str | None = None,
    audit_report: Path | None = None,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    ocr_figure_text: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    skip_any_pdfa: bool = False,
    allow_signature_invalidation: bool = False,
) -> int:
    """Converts a single PDF file.

    Args:
        input_path: Path to the input PDF.
        output: Optional output path.
        level: PDF/A conformance level.
        do_validate: Whether to validate after conversion.
        force: Whether to overwrite existing files.
        quiet: Whether to only output errors.
        pdfa: Whether to perform PDF/A conversion after OCR processing.
        pdfua: Whether to also produce PDF/UA-1 output.
        publication_policy: Controls publication after failed validation.
        document_title: Authoritative document title.
        document_language: Authoritative BCP 47 document language.
        audit_report: Optional path for a machine-readable JSON audit record.
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``).
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
        ocr_force: If True, force OCR even on pages with existing text.
        ocr_deskew: If True, straighten scan-like, raster-dominant pages.
        ocr_rotate_pages: If True, normalize page orientation before OCR.
        ocr_execution_provider: ONNX Runtime provider for Paddle models;
            ``"directml:<index>"`` selects a specific adapter.
        ocr_layout: If True, order OCR lines by detected page columns.
        ocr_figure_text: If True, generate review-required Figure ActualText
            from sufficiently confident OCR.
        convert_calibrated: If True, convert CalGray/CalRGB to ICCBased.
        preserve_stamps: If True, convert known proprietary stamp annotations
            to standard PDF Stamp annotations instead of flattening them.
        skip_any_pdfa: If True, skip files that veraPDF validates as
            compliant PDF/A regardless of target level.
        allow_signature_invalidation: If True, convert signed PDFs even though
            conversion removes or invalidates their digital signatures.

    Returns:
        Exit code.
    """
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = generate_output_path(input_path, pdfa=pdfa)

    # Check if output exists
    if output_path.exists() and not force:
        message = (
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )
        print_error(message)
        if audit_report is not None:
            _write_audit_report(
                audit_report,
                [],
                fatal_error={
                    "error_type": "FileExistsError",
                    "message": message,
                    "exit_code": EXIT_GENERAL_ERROR,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                },
            )
        return EXIT_GENERAL_ERROR

    if not quiet:
        if pdfa:
            target = f"PDF/A-{level}"
            if pdfua:
                target += " + PDF/UA-1"
            click.echo(
                _encode_for_console(f"Converting {input_path.name} -> {target}...")
            )
        else:
            click.echo(
                _encode_for_console(
                    f"Processing {input_path.name} without PDF/A conversion..."
                )
            )

    # Perform conversion
    result = convert_to_pdfa(
        input_path=input_path,
        output_path=output_path,
        level=level,
        pdfa=pdfa,
        pdfua=pdfua,
        validate=do_validate,
        publication_policy=publication_policy,
        document_title=document_title,
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
        ocr_figure_text=ocr_figure_text,
        convert_calibrated=convert_calibrated,
        preserve_stamps=preserve_stamps,
        allow_signature_invalidation=allow_signature_invalidation,
        _allow_output_overwrite=force,
    )

    _print_result(result, quiet)
    if audit_report is not None:
        _write_audit_report(audit_report, [result])

    if result.validation_failed:
        return EXIT_VALIDATION_FAILED
    if not result.success:
        return EXIT_CONVERSION_FAILED
    if result.pdfua_status is PDFUAStatus.REVIEW_REQUIRED:
        if quiet:
            click.echo(
                _encode_for_console(
                    "Review required: PDF/UA author review is outstanding",
                    err=True,
                ),
                err=True,
            )
        return EXIT_REVIEW_REQUIRED

    return EXIT_SUCCESS


def _convert_directory(
    input_dir: Path,
    output: str | None,
    level: str,
    do_validate: bool,
    force: bool,
    recursive: bool,
    quiet: bool,
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    publication_policy: PublicationPolicy | None = None,
    document_language: str | None = None,
    audit_report: Path | None = None,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    ocr_figure_text: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    skip_any_pdfa: bool = False,
    allow_signature_invalidation: bool = False,
) -> int:
    """Converts all PDFs in a directory.

    Args:
        input_dir: Input directory.
        output: Optional output directory.
        level: PDF/A conformance level.
        do_validate: Whether to validate after conversion.
        force: Whether to overwrite existing output files.
        recursive: Whether to process recursively.
        quiet: Whether to only output errors.
        pdfa: Whether to perform PDF/A conversion after OCR processing.
        pdfua: Whether to also produce PDF/UA-1 output.
        publication_policy: Controls publication after failed validation.
        document_language: Authoritative BCP 47 document language.
        audit_report: Optional path for a machine-readable JSON audit record.
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``).
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
        ocr_force: If True, force OCR even on pages with existing text.
        ocr_deskew: If True, straighten scan-like, raster-dominant pages.
        ocr_rotate_pages: If True, normalize page orientation before OCR.
        ocr_execution_provider: ONNX Runtime provider for Paddle models;
            ``"directml:<index>"`` selects a specific adapter.
        ocr_layout: If True, order OCR lines by detected page columns.
        ocr_figure_text: If True, generate review-required Figure ActualText
            from sufficiently confident OCR.
        convert_calibrated: If True, convert CalGray/CalRGB to ICCBased.
        preserve_stamps: If True, convert known proprietary stamp annotations
            to standard PDF Stamp annotations instead of flattening them.
        skip_any_pdfa: If True, skip files that veraPDF validates as
            compliant PDF/A regardless of target level.
        allow_signature_invalidation: If True, convert signed PDFs even though
            conversion removes or invalidates their digital signatures.

    Returns:
        Exit code.
    """
    output_dir = Path(output) if output else None

    if not quiet:
        mode = "recursive" if recursive else "non-recursive"
        if pdfa:
            target = f"PDF/A-{level}"
            if pdfua:
                target += " + PDF/UA-1"
            click.echo(
                _encode_for_console(
                    f"Converting directory {input_dir} ({mode}) -> {target}..."
                )
            )
        else:
            click.echo(
                _encode_for_console(
                    f"Processing directory {input_dir} ({mode}) "
                    "without PDF/A conversion..."
                )
            )

    results = convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        level=level,
        pdfa=pdfa,
        pdfua=pdfua,
        recursive=recursive,
        validate=do_validate,
        publication_policy=publication_policy,
        document_language=document_language,
        skip_any_pdfa=skip_any_pdfa,
        show_progress=not quiet,
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        ocr_execution_provider=ocr_execution_provider,
        ocr_layout=ocr_layout,
        ocr_figure_text=ocr_figure_text,
        force_overwrite=force,
        convert_calibrated=convert_calibrated,
        preserve_stamps=preserve_stamps,
        allow_signature_invalidation=allow_signature_invalidation,
    )
    if audit_report is not None:
        _write_audit_report(audit_report, results)

    # Output summary
    successful = [
        r
        for r in results
        if r.success
        and not r.skipped
        and not r.validation_failed
        and not r.review_required
    ]
    skipped = [
        r for r in results if r.success and r.skipped and not r.validation_failed
    ]
    failed = [r for r in results if not r.success and not r.validation_failed]
    validation_failures = [r for r in results if r.validation_failed]
    review_required = [
        r for r in results if r.review_required and not r.validation_failed
    ]

    if not quiet:
        click.echo()
        click.echo("Summary:")
        action = "converted" if pdfa else "processed"
        print_success(f"{len(successful)} file(s) successfully {action}")
        if skipped:
            print_warning(f"{len(skipped)} file(s) skipped and copied unchanged")
            for result in skipped:
                for warning in result.warnings:
                    click.echo(
                        _encode_for_console(f"  - {result.input_path.name}: {warning}")
                    )
        if failed:
            print_error(f"{len(failed)} file(s) failed")
            for result in failed:
                click.echo(
                    _encode_for_console(
                        f"  - {result.input_path.name}: {result.error}",
                        err=True,
                    ),
                    err=True,
                )
        if review_required:
            print_warning(
                f"{len(review_required)} file(s) require PDF/UA author review"
            )
    if validation_failures:
        print_error(f"{len(validation_failures)} file(s) failed validation")
        for result in validation_failures:
            val_warnings = [
                w for w in result.warnings if w.startswith(_VALIDATION_PREFIXES)
            ]
            for w in val_warnings:
                click.echo(
                    _encode_for_console(
                        f"  - {result.input_path.name}: {w}",
                        err=True,
                    ),
                    err=True,
                )
    if quiet and review_required:
        click.echo(
            _encode_for_console(
                f"Review required: {len(review_required)} PDF/UA file(s)",
                err=True,
            ),
            err=True,
        )

    if failed:
        if all(result.error == "Output file already exists" for result in failed):
            return EXIT_GENERAL_ERROR
        return EXIT_CONVERSION_FAILED

    if validation_failures:
        return EXIT_VALIDATION_FAILED

    if review_required:
        return EXIT_REVIEW_REQUIRED

    return EXIT_SUCCESS


if __name__ == "__main__":
    main()
