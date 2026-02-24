# pdftopdfa

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MPL--2.0+-blue.svg)

I built pdftopdfa as a free and open-source alternative to [Ghostscript](https://www.ghostscript.com/)-based PDF/A converters.
Ghostscript uses a dual license (AGPL/commercial) that makes it difficult to use in commercial products without purchasing a license.
pdftopdfa is licensed under the permissive [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) and can be freely used in commercial projects.
Instead of re-rendering via Ghostscript, it modifies the PDF structure directly using [pikepdf](https://pikepdf.readthedocs.io/) (based on [QPDF](https://qpdf.sourceforge.io/)), preserving the original content, fonts, and layout.                                                                                                                                  

## Highlights

- **No Ghostscript required** -- direct PDF manipulation via pikepdf/QPDF
- **PDF/A-2b, 2u, 3b, 3u** -- supports modern PDF/A levels (ISO 19005-2 and 19005-3)
- **Automatic font embedding** -- embeds missing fonts with metrically compatible replacements
- **Font subsetting** -- reduces file size by removing unused glyphs
- **CJK support** -- embeds Noto Sans CJK for Chinese, Japanese, and Korean text
- **ICC color profiles** -- automatically embeds sRGB, CMYK, and grayscale profiles
- **Batch processing** -- converts entire directories, optionally recursive
- **Integrated validation** -- checks conformance via [veraPDF](https://verapdf.org/)
- **OCR support** -- optional text recognition for scanned PDFs via Tesseract
- **Simple API** -- usable as CLI tool or Python library

## How It Works

pdftopdfa applies a multi-step conversion pipeline to make a PDF compliant with the PDF/A standard:

1. **Pre-check** -- detects if the PDF is already a valid PDF/A file (skips conversion if the existing level meets or exceeds the target; see [Usage Guide](docs/usage.md#already-compliant-pdfs) for details)
2. **OCR** (optional) -- runs Tesseract via ocrmypdf on scanned pages without a text layer
3. **Font compliance** -- analyzes all fonts, embeds missing ones, adds ToUnicode mappings, subsets embedded fonts, and fixes encoding issues
4. **Sanitization** -- removes or fixes non-compliant elements (JavaScript, non-standard actions, transparency groups, annotations, optional content, etc.)
5. **Metadata** -- synchronizes XMP metadata with the document info dictionary and sets the PDF/A conformance level
6. **Color profiles** -- detects color spaces and embeds the required ICC profiles (sRGB, CMYK/FOGRA39, sGray)
7. **Save** -- writes the output with the correct PDF version header

## Installation

### Prerequisites

- Python 3.12, 3.13, or 3.14
- macOS, Linux, or Windows

```bash
pip install pdftopdfa
```

### Optional: OCR support

```bash
pip install "pdftopdfa[ocr]"
```

OCR requires a [Tesseract](https://github.com/tesseract-ocr/tesseract) installation on the system. See [docs/ocr.md](docs/ocr.md) for details on OCR usage and quality presets.

## Quick Start

```bash
# Simple conversion (creates document_pdfa.pdf)
pdftopdfa document.pdf

# Specific PDF/A level
pdftopdfa -l 2b document.pdf

# With validation
pdftopdfa -v document.pdf

# Convert an entire directory
pdftopdfa -r ./documents/ ./output/

# OCR for scanned PDFs
pdftopdfa --ocr document.pdf
```

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    level="2b",
)
```

See [docs/usage.md](docs/usage.md) for the full CLI reference, Python API documentation, and examples.

## Limitations

- **No PDF/A-1 support** -- only PDF/A-2 and PDF/A-3 levels are supported
- **Encrypted PDFs** -- password-protected PDFs cannot be converted
- **Font replacement** -- fonts without a suitable metrically compatible replacement produce a warning; the resulting file may not be fully compliant

## Development

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

The test suite contains 2600+ tests covering fonts, color profiles, metadata, sanitization, and end-to-end conversion.

### Code Quality

```bash
ruff check src/ tests/   # Linting
ruff format src/ tests/  # Formatting
```

## Documentation

Additional documentation is available in the [docs/](docs/) folder:

- [Usage Guide (CLI & Python API)](docs/usage.md)
- [OCR Guide](docs/ocr.md)
- [PDF/A-2/3 rules reference (veraPDF)](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules)

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/iredpaul/pdftopdfa/issues) to report bugs or suggest features, or submit a pull request.

## Dependencies

**Core:**

- [pikepdf](https://pikepdf.readthedocs.io/) -- PDF manipulation (based on QPDF)
- [lxml](https://lxml.de/) -- XMP metadata processing
- [fonttools](https://github.com/fonttools/fonttools) -- Font analysis, subsetting, and embedding
- [click](https://click.palletsprojects.com/) -- CLI framework
- [colorama](https://pypi.org/project/colorama/) -- Colored terminal output
- [tqdm](https://tqdm.github.io/) -- Progress bars

**Optional:**

- [ocrmypdf](https://ocrmypdf.readthedocs.io/) -- OCR support (requires [Tesseract](https://github.com/tesseract-ocr/tesseract))
- [pypdfium2](https://github.com/nicfit/pypdfium2) -- PDF page rasterizer for OCR
- [OpenCV](https://opencv.org/) -- improved OCR preprocessing (deskewing, denoising)
- [veraPDF](https://verapdf.org/) -- ISO-compliant PDF/A validation

## Acknowledgments

This project bundles the following resources:

- **[Liberation Fonts](https://github.com/liberationfonts/liberation-fonts)** -- metrically compatible replacements for the PDF Standard 14 fonts (SIL OFL 1.1)
- **[Noto Sans CJK](https://github.com/notofonts/noto-cjk)** -- CJK font coverage (SIL OFL 1.1)
- **[Noto Sans Symbols 2](https://github.com/notofonts/symbols)** -- symbol font replacement (SIL OFL 1.1)
- **[STIX Two Math](https://github.com/stipub/stixfonts)** -- math font replacement (SIL OFL 1.1)
- **[sRGB2014.icc](https://registry.color.org/rgb-registry/srgbprofiles)** -- ICC sRGB profile (ICC)
- **[ISOcoated_v2_300_bas.icc](https://www.eci.org/en/downloads)** -- ICC CMYK profile, FOGRA39 (zlib/libpng license)
- **[sGray](https://github.com/saucecontrol/Compact-ICC-Profiles)** -- compact grayscale ICC profile (CC0-1.0)
- **[Adobe cmap-resources](https://github.com/adobe-type-tools/cmap-resources)** -- CID-to-Unicode mapping data (BSD 3-Clause)

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) or later (MPL-2.0+) -- see [LICENSE](LICENSE) for details.
