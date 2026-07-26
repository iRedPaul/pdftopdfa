# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Click-based CLI for pdftopdfa.

This module provides the command-line interface for
converting PDF files to the PDF/A format.
"""

# Standard Library
import logging
import sys
from pathlib import Path

# Third Party
import click
from colorama import Fore, Style, init

# Local
from . import __version__
from .converter import (
    ConversionResult,
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
from .verapdf import VeraPDFResult, validate_with_verapdf

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_CONVERSION_FAILED = 3
EXIT_VALIDATION_FAILED = 4
EXIT_PERMISSION_ERROR = 5

logger = logging.getLogger(__name__)


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
    if result.success:
        if not quiet:
            if result.level is None:
                action = "Copied unchanged" if result.skipped else "Processed"
                details = f"{result.processing_time:.2f}s"
            else:
                action = "Skipped" if result.skipped else "Converted to PDF/A"
                details = f"PDF/A-{result.level}, {result.processing_time:.2f}s"
            print_success(
                f"{action}: {result.input_path.name} -> "
                f"{result.output_path.name} ({details})"
            )
            for warning in result.warnings:
                print_warning(warning)
    else:
        print_error(f"{result.input_path.name}: {result.error}")


def _print_validation_result(
    result: VeraPDFResult,
    file_path: Path,
    quiet: bool,
) -> None:
    """Prints the validation result in a formatted way.

    Args:
        result: The veraPDF validation result.
        file_path: Path to the validated file.
        quiet: If True, only output errors.
    """
    if result.compliant:
        if not quiet:
            print_success(f"Validation successful: PDF/A-{result.flavour}")
    else:
        print_error(f"Validation failed for {file_path.name}")
        for error in result.errors:
            click.echo(f"  - {error}", err=True)

    if not quiet:
        for warning in result.warnings:
            print_warning(warning)


@click.command()
@click.argument("input_path", required=False, type=click.Path(exists=True))
@click.argument("output", required=False, type=click.Path())
@click.option(
    "-l",
    "--level",
    type=click.Choice(["2b", "2u", "3b", "3u"]),
    default="3b",
    help="PDF/A conformance level: b=basic, u=Unicode (default: 3b)",
)
@click.option(
    "-v",
    "--validate",
    "do_validate",
    is_flag=True,
    help="Validate after conversion (note: -v is not verbose; use --verbose)",
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
    "uses language from --ocr-lang, default: en).",
)
@click.option(
    "--ocr-lang",
    "ocr_lang",
    default="en",
    help="PaddleOCR language code (default: en). Examples: de, de+en",
)
@click.option(
    "--ocr-detection-model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Verified PP-OCRv6 Medium detection model directory; "
    "use with --ocr-recognition-model-dir.",
)
@click.option(
    "--ocr-recognition-model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Verified PP-OCRv6 Medium recognition model directory; "
    "use with --ocr-detection-model-dir.",
)
@click.option(
    "--ocr-force",
    "ocr_force",
    is_flag=True,
    default=False,
    help="Force OCR even on pages that already contain text "
    "(removes existing OCR layer and re-applies). Implies --ocr and "
    "bypasses compliant-PDF/A skip checks.",
)
@click.option(
    "--deskew",
    is_flag=True,
    default=False,
    help="Straighten skewed pages. Implies --ocr.",
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

    if not pdfa and do_validate:
        raise click.UsageError("--validate cannot be combined with --no-pdfa")

    # Check veraPDF availability if validation is requested
    if do_validate:
        from .verapdf import is_verapdf_available

        if not is_verapdf_available():
            print_error(
                "Validation requires veraPDF, but it is not installed.\n"
                "Please install veraPDF from https://verapdf.org/ "
                "and ensure it is in your PATH."
            )
            sys.exit(EXIT_GENERAL_ERROR)

    if deskew and ocr_force:
        raise click.UsageError("--deskew cannot be combined with --ocr-force")

    model_pair_complete = (
        ocr_detection_model_dir is not None and ocr_recognition_model_dir is not None
    )
    if (ocr_detection_model_dir is None) != (ocr_recognition_model_dir is None):
        raise click.UsageError(
            "--ocr-detection-model-dir and --ocr-recognition-model-dir "
            "must be provided together"
        )
    if ocr_enabled or ocr_force or deskew or rotate_pages:
        if not model_pair_complete:
            raise click.UsageError(
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
            raise click.UsageError(str(exc)) from exc

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
                ocr_languages=ocr_languages,
                ocr_detection_model_dir=ocr_detection_model_dir,
                ocr_recognition_model_dir=ocr_recognition_model_dir,
                ocr_force=ocr_force,
                ocr_deskew=deskew,
                ocr_rotate_pages=rotate_pages,
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
                ocr_languages=ocr_languages,
                ocr_detection_model_dir=ocr_detection_model_dir,
                ocr_recognition_model_dir=ocr_recognition_model_dir,
                ocr_force=ocr_force,
                ocr_deskew=deskew,
                ocr_rotate_pages=rotate_pages,
                convert_calibrated=convert_calibrated,
                preserve_stamps=preserve_stamps,
                skip_any_pdfa=skip_any_pdfa,
                allow_signature_invalidation=allow_signature_invalidation,
            )
        else:
            print_error(f"Invalid path: {input_path}")
            exit_code = EXIT_FILE_NOT_FOUND

    except FileNotFoundError as e:
        print_error(str(e))
        exit_code = EXIT_FILE_NOT_FOUND
    except PermissionError as e:
        print_error(f"Access denied: {e}")
        exit_code = EXIT_PERMISSION_ERROR
    except (
        ConversionError,
        UnsupportedPDFError,
        FontEmbeddingError,
        OCRError,
        VeraPDFError,
    ) as e:
        print_error(str(e))
        exit_code = EXIT_CONVERSION_FAILED
    except Exception as e:
        logger.exception("Unexpected error")
        print_error(f"Unexpected error: {e}")
        exit_code = EXIT_GENERAL_ERROR

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
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
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
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``).
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
        ocr_force: If True, force OCR even on pages with existing text.
        ocr_deskew: If True, straighten skewed pages during OCR.
        ocr_rotate_pages: If True, normalize page orientation before OCR.
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
        print_error(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )
        return EXIT_GENERAL_ERROR

    if not quiet:
        if pdfa:
            click.echo(f"Converting {input_path.name} -> PDF/A-{level}...")
        else:
            click.echo(f"Processing {input_path.name} without PDF/A conversion...")

    # Perform conversion
    result = convert_to_pdfa(
        input_path=input_path,
        output_path=output_path,
        level=level,
        pdfa=pdfa,
        validate=False,  # Validate manually later
        skip_any_pdfa=skip_any_pdfa,
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        convert_calibrated=convert_calibrated,
        preserve_stamps=preserve_stamps,
        allow_signature_invalidation=allow_signature_invalidation,
    )

    _print_result(result, quiet)

    # Note: convert_to_pdfa() raises on failure instead of returning
    # success=False, so no failure branch is needed here.
    if result.skipped:
        return EXIT_SUCCESS

    if result.validation_failed:
        return EXIT_VALIDATION_FAILED

    # Optional: Validation
    if do_validate:
        if not quiet:
            click.echo("Validating output with veraPDF...")

        try:
            verapdf_result = validate_with_verapdf(
                path=output_path,
                flavour=level,
                timeout=300,
            )
        except VeraPDFError as e:
            if not quiet:
                click.echo(
                    f"  Validation skipped: veraPDF not available ({e})",
                    err=True,
                )
            return EXIT_SUCCESS

        _print_validation_result(verapdf_result, output_path, quiet)

        if not quiet:
            click.echo(
                f"  veraPDF: {verapdf_result.passed_rules} rules passed, "
                f"{verapdf_result.failed_rules} failed"
            )

        if not verapdf_result.compliant:
            return EXIT_VALIDATION_FAILED

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
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
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
        ocr_languages: Optional list of PaddleOCR language codes
            (e.g., ``["de", "en"]``).
        ocr_detection_model_dir: PP-OCRv6 Medium detection model directory.
        ocr_recognition_model_dir: PP-OCRv6 Medium recognition model directory.
        ocr_force: If True, force OCR even on pages with existing text.
        ocr_deskew: If True, straighten skewed pages during OCR.
        ocr_rotate_pages: If True, normalize page orientation before OCR.
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
            click.echo(f"Converting directory {input_dir} ({mode}) -> PDF/A-{level}...")
        else:
            click.echo(
                f"Processing directory {input_dir} ({mode}) without PDF/A conversion..."
            )

    results = convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        level=level,
        pdfa=pdfa,
        recursive=recursive,
        validate=do_validate,
        skip_any_pdfa=skip_any_pdfa,
        show_progress=not quiet,
        ocr_languages=ocr_languages,
        ocr_detection_model_dir=ocr_detection_model_dir,
        ocr_recognition_model_dir=ocr_recognition_model_dir,
        ocr_force=ocr_force,
        ocr_deskew=ocr_deskew,
        ocr_rotate_pages=ocr_rotate_pages,
        force_overwrite=force,
        convert_calibrated=convert_calibrated,
        preserve_stamps=preserve_stamps,
        allow_signature_invalidation=allow_signature_invalidation,
    )

    # Output summary
    successful = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.success and r.skipped]
    failed = [r for r in results if not r.success]
    validation_failures = [r for r in successful if r.validation_failed]

    if not quiet:
        click.echo()
        click.echo("Summary:")
        action = "converted" if pdfa else "processed"
        print_success(f"{len(successful)} file(s) successfully {action}")
        if skipped:
            print_warning(f"{len(skipped)} file(s) skipped and copied unchanged")
            for result in skipped:
                for warning in result.warnings:
                    click.echo(f"  - {result.input_path.name}: {warning}")
        if failed:
            print_error(f"{len(failed)} file(s) failed")
            for result in failed:
                click.echo(f"  - {result.input_path.name}: {result.error}", err=True)
        if validation_failures:
            print_error(f"{len(validation_failures)} file(s) failed validation")
            for result in validation_failures:
                val_warnings = [
                    w for w in result.warnings if w.startswith("Validation:")
                ]
                for w in val_warnings:
                    click.echo(f"  - {result.input_path.name}: {w}", err=True)

    if failed:
        return EXIT_CONVERSION_FAILED

    if validation_failures:
        return EXIT_VALIDATION_FAILED

    return EXIT_SUCCESS


if __name__ == "__main__":
    main()
