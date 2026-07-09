# Usage Guide

This guide covers everyday usage of `pdftopdfa` from the command line and Python.

For OCR-specific setup and tuning, see [OCR Guide](ocr.md).

For the full list of PDF/A-2/3 compliance rules, see the [veraPDF PDF/A-2 and PDF/A-3 rules reference](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules).

## Basic Usage

### Convert One File

```bash
# Creates input_pdfa.pdf next to input.pdf
pdftopdfa input.pdf

# Explicit output file
pdftopdfa input.pdf output.pdf

# Target a specific level
pdftopdfa -l 2b input.pdf

# Validate output with veraPDF
pdftopdfa -v input.pdf

# Overwrite an existing output
pdftopdfa -f input.pdf output.pdf
```

### Batch Processing

```bash
# Convert all PDFs in a directory
pdftopdfa ./input-dir/ ./output-dir/

# Convert recursively
pdftopdfa -r ./documents/

# Recursive, verbose, and overwrite existing outputs
pdftopdfa -r -f --verbose ./documents/ ./output/
```

## Output Paths and Overwrite Rules

- Default output filename is `<input_stem>_pdfa.pdf`.
- Single-file conversion without explicit output writes next to the input file.
- Directory conversion without explicit output writes into the same directory.
- Recursive directory conversion with an explicit output directory preserves subdirectory structure.
- When converting in-place (`output_dir=None`), files already ending in `_pdfa.pdf` are skipped to avoid reconversion loops.
- Existing output files are not overwritten unless `-f/--force` (CLI) or `force_overwrite=True` (API) is used.

## Font Policy

- On Windows, `pdftopdfa` may automatically embed a conservative fixed allowlist of local fonts from `%WINDIR%\Fonts`.
- A Windows system font is only used when the file path stays under `%WINDIR%\Fonts`, its actual PostScript name is allowlisted, and its `fsType` permits outline embedding.
- If `fsType` disallows outline embedding, `pdftopdfa` falls back to the bundled replacement font path.
- On macOS and Linux, system fonts are never auto-embedded.
- `fsType` checks are a technical safeguard only and do not replace the rights holder's EULA or other license terms.
- For auditable deployments, tie the allowlist to validated target systems or golden images.

## CLI Reference

### Arguments

| Argument | Description |
|---|---|
| `input_path` | Input PDF file or input directory |
| `output` | Optional output PDF file or output directory |

### Options

| Option | Description |
|---|---|
| `-l, --level [2b\|2u\|3b\|3u]` | Target PDF/A level (default: `3b`) |
| `-v, --validate` | Validate output with veraPDF. Note: unlike the common CLI convention, `-v` does **not** mean verbose — use `--verbose` for detailed logs |
| `-r, --recursive` | Process directories recursively |
| `-f, --force` | Overwrite existing output files |
| `-q, --quiet` | Show only errors |
| `--verbose` | Enable detailed logs |
| `--ocr` | Enable OCR for scanned/image-based PDFs |
| `--ocr-force` | Force OCR even if text is present (implies `--ocr` and disables compliant-PDF/A skip optimization) |
| `--ocr-lang LANG` | OCR language code (default: `eng`), for example `deu` or `deu+eng` |
| `--ocr-quality [fast\|default\|best]` | OCR quality preset (default: `default`) |
| `--ocr-fallback-quality [none\|fast\|default\|best]` | Faster OCR preset to retry with if OCR takes too long (default: `fast`) |
| `--ocr-fallback-after SECONDS` | Retry threshold for OCR fallback (default: `60`) |
| `--convert-calibrated/--no-convert-calibrated` | Convert CalGray/CalRGB to ICCBased (default: enabled) |
| `--preserve-stamps` | Convert known proprietary stamp annotations to standard PDF Stamp annotations instead of flattening them |
| `--skip-any-pdfa` | Skip inputs that veraPDF validates as any compliant PDF/A, regardless of target level |
| `--allow-signature-invalidation` | Convert digitally signed PDFs even though conversion removes or invalidates their signatures |
| `--version` | Show version and exit |
| `--help` | Show help and exit |

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | General error |
| `2` | Input path not found |
| `3` | Conversion failed |
| `4` | Validation failed |
| `5` | Permission error |

## Python API

### `convert_to_pdfa()`

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    level="2b",
    validate=False,
)

if result.success:
    print("Done")
else:
    print(result.error)
```

Signature:

```python
def convert_to_pdfa(
    input_path: Path,
    output_path: Path,
    level: str = "3b",
    *,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_quality: OcrQuality | None = None,
    ocr_fallback_quality: OcrQuality | None = OcrQuality.FAST,
    ocr_fallback_after_seconds: float | None = 60.0,
    ocr_force: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> ConversionResult
```

By default, non-standard visible annotations are flattened into page content for
maximum archival robustness. Set `preserve_stamps=True` or pass
`--preserve-stamps` to convert known proprietary stamp annotations to standard
`/Stamp` annotations instead.

### `convert_directory()`

```python
from pathlib import Path
from pdftopdfa import convert_directory

results = convert_directory(
    input_dir=Path("./input"),
    output_dir=Path("./output"),
    level="3b",
    recursive=True,
)

for r in results:
    print(r.input_path.name, "OK" if r.success else r.error)
```

Signature:

```python
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
    ocr_quality: OcrQuality | None = None,
    ocr_fallback_quality: OcrQuality | None = OcrQuality.FAST,
    ocr_fallback_after_seconds: float | None = 60.0,
    ocr_force: bool = False,
    force_overwrite: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]
```

### `convert_files()`

```python
from pathlib import Path
from pdftopdfa import convert_files

pairs = [
    (Path("a.pdf"), Path("a_pdfa.pdf")),
    (Path("b.pdf"), Path("b_pdfa.pdf")),
]

results = convert_files(pairs, level="3b", force_overwrite=True)
```

Signature:

```python
def convert_files(
    file_pairs: list[tuple[Path, Path]],
    level: str = "3b",
    *,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_quality: OcrQuality | None = None,
    ocr_fallback_quality: OcrQuality | None = OcrQuality.FAST,
    ocr_fallback_after_seconds: float | None = 60.0,
    ocr_force: bool = False,
    force_overwrite: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]
```

### `needs_ocr()`

Library helper to analyze whether a PDF would benefit from OCR. It is not
called by the conversion pipeline itself (OCR is opt-in via `ocr_languages`
or `--ocr`); use it to decide programmatically whether to enable OCR.

```python
import pikepdf
from pdftopdfa.ocr import needs_ocr

with pikepdf.open("scan.pdf") as pdf:
    if needs_ocr(pdf, threshold=0.5):
        ...  # convert with ocr_languages=["eng"]
```

Signature:

```python
def needs_ocr(pdf: pikepdf.Pdf, *, threshold: float = 0.5) -> bool
```

A page counts as needing OCR when it contains images but no text operators.
`needs_ocr()` returns `True` when at least `threshold` (0.0-1.0) of the pages
need OCR.

### `ConversionResult`

`ConversionResult` is returned by all conversion APIs.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` if conversion succeeded |
| `input_path` | `Path` | Input file path |
| `output_path` | `Path` | Output file path |
| `level` | `str` | Effective PDF/A level |
| `warnings` | `list[str]` | Non-fatal conversion warnings |
| `processing_time` | `float` | Runtime in seconds |
| `error` | `str \\| None` | Error message if failed |
| `validation_failed` | `bool` | `True` if veraPDF reported non-compliance |
| `skipped` | `bool` | `True` if the original PDF was copied unchanged |

## Exceptions

All custom exceptions inherit from `PDFToPDFAError`:

- `ConversionError`
- `ValidationError`
- `FontEmbeddingError`
- `UnsupportedPDFError`
- `OCRError`
- `VeraPDFError`

Example:

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.exceptions import ConversionError, UnsupportedPDFError

try:
    convert_to_pdfa(Path("input.pdf"), Path("output.pdf"))
except UnsupportedPDFError:
    print("Unsupported PDF")
except ConversionError as exc:
    print(f"Conversion failed: {exc}")
```

Encrypted PDFs are copied to the output path unchanged and returned with
`success=True`, `skipped=True`, and a warning that conversion was skipped.

Digitally signed PDFs are also copied unchanged by default, because OCR,
metadata repair, font embedding, and PDF/A rewriting would invalidate the
cryptographic signature. Use `--allow-signature-invalidation` or
`allow_signature_invalidation=True` only when you intentionally want an
unsigned PDF/A copy. For signed archives, the recommended workflow is to
convert to PDF/A first and sign the PDF/A output afterwards.

## PDF/A Levels

| Level | ISO Standard | Attachments | Unicode Required | Recommended For |
|---|---|---|---|---|
| `2b` | ISO 19005-2 | PDF/A attachments only | No | Basic archiving |
| `2u` | ISO 19005-2 | PDF/A attachments only | Yes | Searchable archives |
| `3b` | ISO 19005-3 | Any format | No | Hybrid documents (for example PDF + XML) |
| `3u` | ISO 19005-3 | Any format | Yes | Searchable hybrid archives |

Default level: `3b`.

## Already Compliant PDFs

Before conversion, `pdftopdfa` checks whether a file already claims a PDF/A level.
If veraPDF is available, it validates that claim before deciding to skip conversion.

Default behavior:

| Detected | Behavior |
|---|---|
| Same level (`2b` -> `2b`) | Skip conversion |
| Higher conformance in same part (`2u` -> `2b`) | Skip conversion |
| Lower conformance in same part (`2b` -> `2u`) | Convert |
| Different part (`1b` -> `3b`, `3u` -> `2b`) | Convert |

With `--skip-any-pdfa` or `skip_any_pdfa=True`, any input that veraPDF validates as
compliant PDF/A is skipped, regardless of part or conformance level.

- This broader skip only happens when veraPDF actually confirms compliance.
- If veraPDF is not available, the file is converted normally.
- If the document claims PDF/A but veraPDF reports non-compliance, the file is converted normally.

Notes:

- If the metadata claim fails veraPDF validation, conversion is not skipped.
- If veraPDF is unavailable, conversion is not skipped based only on metadata.
- Skipped files return warning: `Conversion skipped: PDF already valid PDF/A (veraPDF compliant)`.
- Forced OCR (`--ocr-force` or `ocr_force=True` with OCR languages configured) always runs and bypasses this skip logic.
- Forced OCR does not bypass the signed-PDF protection; use `--allow-signature-invalidation` explicitly to convert signed inputs.

## Validation

`pdftopdfa` integrates with [veraPDF](https://verapdf.org/) for PDF/A validation.

- CLI: `pdftopdfa -v input.pdf`
- API: pass `validate=True`

If veraPDF is missing, conversion still runs, and validation is reported as skipped.

## Environment Variables

| Variable | Description |
|---|---|
| `VERAPDF_PATH` | Path to `verapdf` executable or its parent directory |
| `TESSERACT_PATH` | Path to `tesseract` executable or its parent directory |

## Related Docs

- OCR details: [ocr.md](ocr.md)
- PDF/A-2/3 rules reference: [veraPDF validation profiles wiki](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules)
