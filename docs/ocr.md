# OCR Guide

This guide covers OCR behavior in `pdftopdfa` for scanned and image-based
PDFs. General CLI and API usage is documented in the
[Usage Guide](usage.md).

## Architecture

Text detection, recognition, word positions, and deskew use PP-OCRv6 Medium
from PaddleOCR 3.7 through ONNX Runtime on the CPU. OCRmyPDF is used for page
rasterization, searchable text-layer generation, page merging, and PDF
geometry.

The recognition models are supplied explicitly by the application. There is no
automatic model resolution, runtime download, or secondary recognition engine.
Document orientation uses a separate `PP-LCNet_x1_0_doc_ori` model bundled
with `pdftopdfa`.

The verified Paddle session is reused across pages. OCRmyPDF runs one page job
at a time and Paddle predictions are serialized, avoiding concurrent access to
the PaddleX session. The detector limits its inference image to 1,600 pixels
on the longest side. OCR uses a minimum rasterization resolution of 600 DPI;
higher-resolution source images can therefore still require more memory while
the source raster is prepared.

## Installation

Install the OCR extras:

```bash
pip install "pdftopdfa[ocr]"
```

No additional system OCR executable is required.

## Offline Model Contract

Download the files from these exact revisions:

- Detection:
  [`PP-OCRv6_medium_det_onnx` at `6132380`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a)
- Recognition:
  [`PP-OCRv6_medium_rec_onnx` at `50c7eac`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b)

Create one directory for each model. Each directory must contain exactly
`inference.onnx` and `inference.yml`:

```text
PP-OCRv6_medium_det/
├── inference.onnx
└── inference.yml

PP-OCRv6_medium_rec/
├── inference.onnx
└── inference.yml
```

| Model | File | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| Detection | `inference.onnx` | 62,032,837 | `eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1` |
| Detection | `inference.yml` | 886 | `7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871` |
| Recognition | `inference.onnx` | 76,554,979 | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |
| Recognition | `inference.yml` | 150,580 | `991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129` |

Before the first model initialization, `pdftopdfa` rejects missing or extra
directory entries, symbolic links, non-regular files, unexpected sizes, and
SHA-256 mismatches. A successfully verified unchanged path pair and its
inference session are reused.

The PP-OCRv6 models are intentionally not included in the repository, wheel,
or source distribution. Store them in deployment-managed, preferably read-only
directories and pass both paths explicitly on every OCR invocation.

## Languages

The default language code is `en`. Use `de` for German and `de+en` on the CLI
for mixed German/English metadata:

```bash
pdftopdfa --ocr-lang de+en \
  --ocr-detection-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_languages=["de", "en"]`. Legacy codes such as
`eng` and `deu` are rejected without compatibility aliases.

Accepted PaddleOCR 3.7 codes:

`af`, `az`, `bs`, `ca`, `ch`, `chinese_cht`, `cs`, `cy`, `da`, `de`, `en`,
`es`, `et`, `eu`, `fi`, `fr`, `french`, `ga`, `german`, `gl`, `hr`, `hu`,
`id`, `is`, `it`, `japan`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`,
`nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`,
`sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`.

The unified Medium model performs recognition for these languages. The
primary selection is also written to the PDF catalog `/Lang` entry when the
input does not already declare a document language. Paddle-specific aliases
are written as their corresponding BCP 47 tag, for example `ch` as `zh-Hans`
and `chinese_cht` as `zh-Hant`.

## CLI Usage

Both model options must always be supplied together. Supplying the pair enables
OCR, so `--ocr` is optional in the first example:

```bash
DET_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_det
REC_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_rec

# English (default)
pdftopdfa \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# German
pdftopdfa --ocr --ocr-lang de \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# German and English
pdftopdfa --ocr-lang de+en \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# Apply OCR and page processing without PDF/A conversion
pdftopdfa --no-pdfa --deskew --rotate-pages \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

`--ocr`, `--ocr-force`, `--deskew`, and `--rotate-pages` are rejected unless
both model options are present. Providing only one model option is also
rejected. `--ocr-lang` selects metadata but does not activate OCR by itself.

## Python API

The high-level conversion functions use the optional
`ocr_detection_model_dir` and `ocr_recognition_model_dir` keyword arguments.
Providing both enables OCR:

```python
from pathlib import Path

from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    ocr_languages=["de", "en"],
    ocr_detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    ocr_recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
    ocr_rotate_pages=True,
)
```

The same keywords are available on `convert_files()` and
`convert_directory()`. Supplying only one directory, or supplying
`ocr_languages`, `ocr_force`, `ocr_deskew`, or `ocr_rotate_pages` without the
complete pair, raises `ValueError` before input processing starts.

The lower-level function requires both model directories as keyword-only
arguments:

```python
from pathlib import Path

from pdftopdfa.ocr import apply_ocr

result_path = apply_ocr(
    Path("scan.pdf"),
    Path("scan_processed.pdf"),
    ["de", "en"],
    detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
    deskew=True,
)
```

`apply_ocr()` raises `OCRError` for missing, unreadable, altered, or unsupported
model artifacts and for OCR execution failures.

Use `pdfa=False` with a neutral output name such as `scan_processed.pdf` to
write the OCR result directly without font embedding, PDF/A sanitization,
metadata synchronization, color-profile embedding, or PDF/A validation. The
output is not validated or guaranteed to be PDF/A compliant.

## Page Selection and Force Mode

When OCR is enabled, processing runs for the document:

- Pages without a text layer receive OCR.
- Pages that already contain text are skipped.
- Mixed documents are handled page by page.

Use force mode to replace an existing OCR layer:

```bash
pdftopdfa --ocr-force --ocr-lang de \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

Python uses `ocr_force=True`. Force mode:

- replaces existing OCR text layers;
- bypasses the already-compliant PDF/A skip path;
- does not bypass signed-PDF protection;
- cannot be combined with `--deskew` or `ocr_deskew=True`.

Use `--allow-signature-invalidation` only when an unsigned OCR/PDF/A copy is
intentional.

## Deskew and Page Orientation

`--deskew` and `--rotate-pages` are independent opt-in operations. Both enable
OCR and require the external detection and recognition model directories.

Deskew performs an additional PaddleOCR pass. It derives a correction angle
from at least two sufficiently long text polygons within ±10 degrees. With
insufficient evidence the page is left unchanged. Deskewing may alter page
appearance and increase file size.

PDFs containing annotations are OCRed without deskewing, and a warning is
emitted. Annotation geometry cannot be transformed safely through a raster
deskew operation, so this prevents links and interactive content from becoming
misaligned.

Before OCR starts, `--rotate-pages` renders and classifies every page with the
bundled `PP-LCNet_x1_0_doc_ori` model, including pages later skipped because
they already contain text. Predictions below 0.80 leave the page unchanged and
produce a warning. Missing, corrupt, or unusable bundled model files abort the
conversion.

## Offline Deployment

The external PP-OCRv6 directories are the only recognition-model source.
Document-orientation files are bundled under
`pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori/`. All document orientation,
text detection, recognition, and deskew operations run through explicitly
configured local ONNX files.

Applications frozen with tools such as PyInstaller must collect the
`pdftopdfa` package data as well as PaddleOCR, PaddleX, ONNX Runtime, OpenCV,
NumPy, OCRmyPDF, pypdfium2, and the ONNX Runtime native libraries. Network
access is not required at runtime. PaddleX may create the empty housekeeping
directories `func_ret`, `locks`, and `temp` below its configured cache path,
but it does not store or download model artifacts there when the explicit
model directories are used. Point `PADDLE_PDX_CACHE_HOME` at a writable
scratch directory when the application filesystem is read-only.

## Troubleshooting

### `OCR not available - pip install pdftopdfa[ocr]`

Install the OCR extras:

```bash
pip install "pdftopdfa[ocr]"
```

### Model directories are rejected

Check that:

- both directory options were supplied;
- each directory contains only `inference.onnx` and `inference.yml`;
- neither directories nor files are symbolic links;
- the files match the sizes and SHA-256 values above;
- the process can read all four files.

Model validation is intentionally fail-closed and never downloads a
replacement.

### Unsupported language code

Use `en`, `de`, `de+en`, or another code from the accepted list above. Do not
use legacy three-letter aliases.

### Bundled orientation model is missing or corrupt

Reinstall `pdftopdfa` from an intact wheel. The orientation preflight validates
the packaged files against SHA-256 hashes and does not attempt a network
download.

### OCR did not alter a page

Possible reasons:

- the page already has a text layer and force mode was not enabled;
- deskew found insufficient geometric evidence;
- page-orientation confidence was below 0.80;
- the digitally signed input was copied unchanged to preserve its signature.

Use `--ocr-force` with both model directories only when replacing an existing
OCR layer is intentional.
