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
from typing import TYPE_CHECKING

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
    ValidationError,
    VeraPDFError,
)
from .utils import setup_logging
from .verapdf import VeraPDFResult, validate_with_verapdf

if TYPE_CHECKING:
    from .ocr import OcrQuality

# Exit codes as per CLAUDE.md
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_CONVERSION_FAILED = 3
EXIT_VALIDATION_FAILED = 4
EXIT_PERMISSION_ERROR = 5

logger = logging.getLogger(__name__)


def print_success(msg: str) -> None:
    """Prints a success message in green.

    Args:
        msg: The message to output.
    """
    click.echo(f"{Fore.GREEN}\u2713{Style.RESET_ALL} {msg}")


def print_error(msg: str) -> None:
    """Prints an error message in red.

    Args:
        msg: The error message to output.
    """
    click.echo(f"{Fore.RED}\u2717 Error:{Style.RESET_ALL} {msg}", err=True)


def print_warning(msg: str) -> None:
    """Prints a warning in yellow.

    Args:
        msg: The warning to output.
    """
    click.echo(f"{Fore.YELLOW}\u26a0{Style.RESET_ALL} {msg}")


def _print_result(result: ConversionResult, quiet: bool) -> None:
    """Prints the conversion result in a formatted way.

    Args:
        result: The conversion result.
        quiet: If True, only output errors.
    """
    if result.success:
        if not quiet:
            print_success(
                f"Converted: {result.input_path.name} -> "
                f"{result.output_path.name} (PDF/A-{result.level}, "
                f"{result.processing_time:.2f}s)"
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
    help="Validate after conversion",
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
    help="Enable OCR for image-based PDFs"
    " (uses language from --ocr-lang, default: eng).",
)
@click.option(
    "--ocr-lang",
    "ocr_lang",
    default="eng",
    help="OCR language code (default: eng). Examples: deu, deu+eng",
)
@click.option(
    "--ocr-quality",
    "ocr_quality",
    type=click.Choice(["fast", "default", "best"]),
    default="default",
    help="OCR quality preset (default: default). "
    "fast=minimal processing, default=best quality without visual changes, "
    "best=best quality (may alter document visually).",
)
@click.option(
    "--convert-calibrated/--no-convert-calibrated",
    default=True,
    help="Convert CalGray/CalRGB color spaces to ICCBased (default: enabled)",
)
@click.version_option(version=__version__)
def main(
    input_path: str | None,
    output: str | None,
    level: str,
    do_validate: bool,
    recursive: bool,
    force: bool,
    quiet: bool,
    verbose: bool,
    ocr_enabled: bool,
    ocr_lang: str,
    ocr_quality: str,
    convert_calibrated: bool,
) -> None:
    """Converts PDF files to the archival PDF/A format.

    INPUT is the path to the input PDF or a directory.
    OUTPUT is optionally the path for the output PDF/A.
    """
    # Initialize colorama for Windows compatibility
    init()

    if input_path is None:
        click.echo(click.get_current_context().get_help())
        sys.exit(EXIT_GENERAL_ERROR)

    # Configure logging
    setup_logging(verbose=verbose, quiet=quiet)

    input_path_obj = Path(input_path)

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

    try:
        # Determine OCR languages (None if OCR not enabled)
        ocr_languages = ocr_lang.split("+") if ocr_enabled else None

        # Convert OCR quality string to enum (lazy import to avoid requiring
        # ocrmypdf when OCR is not used)
        ocr_quality_enum = None
        if ocr_enabled:
            from .ocr import OcrQuality

            ocr_quality_enum = OcrQuality(ocr_quality)

        if input_path_obj.is_file():
            # Convert single file
            exit_code = _convert_single_file(
                input_path_obj,
                output,
                level,
                do_validate,
                force,
                quiet,
                ocr_languages=ocr_languages,
                ocr_quality=ocr_quality_enum,
                convert_calibrated=convert_calibrated,
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
                ocr_languages=ocr_languages,
                ocr_quality=ocr_quality_enum,
                convert_calibrated=convert_calibrated,
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
    except ValidationError as e:
        print_error(str(e))
        exit_code = EXIT_VALIDATION_FAILED
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
    ocr_languages: list[str] | None = None,
    ocr_quality: "OcrQuality | None" = None,
    convert_calibrated: bool = True,
) -> int:
    """Converts a single PDF file.

    Args:
        input_path: Path to the input PDF.
        output: Optional output path.
        level: PDF/A conformance level.
        do_validate: Whether to validate after conversion.
        force: Whether to overwrite existing files.
        quiet: Whether to only output errors.
        ocr_languages: Optional list of Tesseract language codes (e.g., ``["deu", "eng"]``).
        ocr_quality: OCR quality preset.
        convert_calibrated: If True, convert CalGray/CalRGB to ICCBased.

    Returns:
        Exit code.
    """
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = generate_output_path(input_path)

    # Check if output exists
    if output_path.exists() and not force:
        print_error(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )
        return EXIT_GENERAL_ERROR

    if not quiet:
        click.echo(f"Converting {input_path.name} -> PDF/A-{level}...")

    # Perform conversion
    result = convert_to_pdfa(
        input_path=input_path,
        output_path=output_path,
        level=level,
        validate=False,  # Validate manually later
        ocr_languages=ocr_languages,
        ocr_quality=ocr_quality,
        convert_calibrated=convert_calibrated,
    )

    _print_result(result, quiet)

    if not result.success:
        return EXIT_CONVERSION_FAILED

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
    ocr_languages: list[str] | None = None,
    ocr_quality: "OcrQuality | None" = None,
    convert_calibrated: bool = True,
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
        ocr_languages: Optional list of Tesseract language codes (e.g., ``["deu", "eng"]``).
        ocr_quality: OCR quality preset.
        convert_calibrated: If True, convert CalGray/CalRGB to ICCBased.

    Returns:
        Exit code.
    """
    output_dir = Path(output) if output else None

    if not quiet:
        mode = "recursive" if recursive else "non-recursive"
        click.echo(f"Converting directory {input_dir} ({mode}) -> PDF/A-{level}...")

    results = convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        level=level,
        recursive=recursive,
        validate=do_validate,
        show_progress=not quiet,
        ocr_languages=ocr_languages,
        ocr_quality=ocr_quality,
        force_overwrite=force,
        convert_calibrated=convert_calibrated,
    )

    # Output summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    validation_failures = [r for r in successful if r.validation_failed]

    if not quiet:
        click.echo()
        click.echo("Summary:")
        print_success(f"{len(successful)} file(s) successfully converted")
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
