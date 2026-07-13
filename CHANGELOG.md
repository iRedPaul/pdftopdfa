# Changelog

## [0.6.0] - 2026-07-13

### Features

- Add bundled offline PaddleOCR document-orientation classification for best-quality OCR

### Changes

- Replace Tesseract OSD page rotation with CPU-only ONNX Runtime inference
- Analyze every page before best-quality OCR and keep low-confidence pages unchanged
- Add PaddleOCR, ONNX Runtime, and NumPy as required dependencies

### Bug Fixes

- Match PDF filename extensions case-insensitively during directory conversion

### Documentation

- Document offline model packaging, confidence handling, and frozen application requirements
- Document macOS 14 and Apple Silicon as the supported macOS platform

### CI / Build

- Bundle the pinned PP-LCNet orientation model, manifest, source notice, and license
- Test macOS exclusively on an explicit Apple Silicon runner

## [0.5.1] - 2026-07-12

### Changes

- Refactor the PDF/A conversion pipeline and improve font, metadata, and sanitizer handling

### Bug Fixes

- Count only live digital signatures when sanitizing PDFs
- Preserve validation failures for embedded PDFs and avoid duplicate attachment validation
- Abort OCR conversion when annotations cannot be restored safely
- Handle Tesseract OSD timeouts and apply OCR fallback thresholds per page
- Exclude nested output directories from recursive conversion

### Documentation

- Clarify validation, OCR fallback, CIDFont replacement, and `needs_ocr()` behavior

## [0.5.0] - 2026-07-07

### Changes

- Resolve veraPDF launcher via PATH lookup
- Harden OCR and veraPDF subprocess stdin handling
- Simplify ExtGState opacity values

### Bug Fixes

- Fix ToUnicode CMap bfrange parsing
- Fix Type3 glyph usage during font subsetting
- Preserve duplicate font entries during analysis
- Fix out-of-range simple font code handling
- Translate Mac Roman cmap codes when building Unicode subtables
- Preserve XMP Dublin Core metadata when DocInfo is missing

### CI / Build

- Pin pikepdf below 11 and update dependency minimums

## [0.4.3] - 2026-06-01

### Features

- Add `--allow-signature-invalidation` for explicit signed PDF conversion

### Changes

- Skip digitally signed PDFs by default to avoid invalidating signatures

### Documentation

- Document signed PDF handling in README and usage/OCR guides

## [0.4.2] - 2026-06-01

### Bug Fixes

- Fix WinAnsi and malformed Type0 font handling
- Fix odd hex string parsing in structure limits sanitizer

## [0.4.1] - 2026-04-30

### Changes

- Improve preservation of valid structured XMP metadata

### Bug Fixes

- Fix empty metadata placeholders and XMP identifier refresh handling

## [0.4.0] - 2026-04-22

### Changes

- Centralize PDF/A save settings and hardening

### Bug Fixes

- Preserve valid OutputIntents in color profile handling

## [0.3.15] - 2026-04-22

### Changes

- Normalize additional GdPicture stamp annotations

### Bug Fixes

- Fix font width validation against exact glyph metrics

## [0.3.14] - 2026-04-21

### Features

- Add OCR time-based fallback options

### Changes

- Add mailmap for contributor attribution

### Documentation

- Document OCR time-based fallback options

## [0.3.13] - 2026-04-21

### Features

- Add stamp preservation flag for proprietary annotations

## [0.3.12] - 2026-04-16

### Bug Fixes

- Preserve non-compliant annotations by flattening appearance streams

## [0.3.11] - 2026-04-16

### Bug Fixes

- Strip EXIF and TIFF `NativeDigest` metadata that fails veraPDF

## [0.3.10] - 2026-04-16

### Changes

- Remove OpenCV preprocessing from the OCR pipeline
- Increase Tesseract timeout for the default and best OCR presets

## [0.3.9] - 2026-04-15

### Bug Fixes

- Fix OCR metadata validation for non-catalog XMP streams

## [0.3.8] - 2026-04-15

### Bug Fixes

- Preserve mapped subset glyphs in `.notdef` usage sanitization

## [0.3.7] - 2026-04-14

### Bug Fixes

- Fix OCR rotation composition for pre-rotated pages
- Make `--ocr-force` bypass compliant PDF/A skip checks

### Documentation

- Clarify `--ocr-force` skip-check behavior in the OCR and usage guides

## [0.3.6] - 2026-04-10

### Bug Fixes

- Fix AGL validation for TrueType encoding differences
- Fix `.notdef` filtering for CIDFontType0 charsets

## [0.3.5] - 2026-04-10

### Bug Fixes

- Handle array-valued `OpenAction` objects safely

## [0.3.4] - 2026-04-09

### Bug Fixes

- Preserve original document metadata after OCR conversion

## [0.3.3] - 2026-04-09

### Bug Fixes

- Suppress malformed UTF-8 name decode logs during filter sanitization

## [0.3.2] - 2026-04-09

### Bug Fixes

- Skip malformed UTF-8 PDF names during filter sanitization to avoid noisy decode errors

## [0.3.1] - 2026-04-09

### Bug Fixes

- Preserve subsetted Standard 14 font code mappings during embedded font refresh
- Avoid unsafe content stream hex repairs unless parsed operands contain placeholders

## [0.3.0] - 2026-04-09

### Changes

- Enforce a Windows system font allowlist and `fsType` embedding policy

### Bug Fixes

- Fix Type3 font width scaling for nonstandard `FontMatrix`
- Preserve WinAnsi superscript glyphs during `.notdef` handling

### Documentation

- Document the Windows font sourcing and embedding policy in the README and usage guide

## [0.2.23] - 2026-04-07

### Bug Fixes

- Fix CIDFont width repair when `W` entries are missing
- Handle signed PDFs before OCR conversion

## [0.2.22] - 2026-04-02

### Bug Fixes

- Fix CID glyph references after `.notdef` insertion

## [0.2.21] - 2026-04-02

### Bug Fixes

- Fix CID font subsetting for explicit CIDToGIDMap

## [0.2.20] - 2026-04-01

### Bug Fixes

- Fix interpolate cleanup for soft mask images

## [0.2.19] - 2026-04-01

### Bug Fixes

- Fix font state restoration across q and Q operators

## [0.2.18] - 2026-04-01

### Bug Fixes

- Fix PDF/A validation after font subsetting

## [0.2.17] - 2026-04-01

### Bug Fixes

- Fix font width checks for AcroForm default resource fonts
- Fix OCR error handling for PDFs with invalid metadata

## [0.2.16] - 2026-03-29

### Features

- Add `--skip-any-pdfa` to skip inputs that veraPDF validates as compliant PDF/A regardless of target level

### Documentation

- Document the broader veraPDF-based PDF/A skip behavior in the README and usage guide

## [0.2.15] - 2026-03-26

### Bug Fixes

- Fix text-page deskewing to preserve the original page size

## [0.2.14] - 2026-03-26

### Features

- Add deskew normalization for OCR-skipped text pages in `best` quality mode

### Changes

- Deduplicate identical embedded font programs to reduce output size

## [0.2.13] - 2026-03-26

### Bug Fixes

- Fix best-quality OCR rotation normalization for text pages skipped by OCR
- Preserve font subsetting when refreshing embedded subsetted Standard 14 fonts

## [0.2.12] - 2026-03-26

### Bug Fixes

- Strip PDF/A-unsafe xmpMM metadata during metadata preservation
- Fix Standard 14 font alias resolution for embedded replacements

## [0.2.11] - 2026-03-26

### Bug Fixes

- Preserve canonical XMP extension schemas during metadata sync
- Fix reused XMP extension schema property declarations
- Fix symbolic TrueType width matching when fonts omit Encoding
- Repair unused CID overflow entries in embedded CMaps

## [0.2.10] - 2026-03-24

### Bug Fixes

- Fix PDF/A validation failures for subsetted embedded Standard-14 fonts by refreshing them to full replacements
- Fix generated XMP metadata to omit `pdf:Trapped` and avoid PDF/A validation failures

## [0.2.9] - 2026-03-24

### Changes

- Skip encrypted PDFs by copying them unchanged

### Documentation

- Document skipped conversion results for encrypted PDFs

## [0.2.8] - 2026-03-20

### Bug Fixes

- Fix forced OCR to avoid conflicting redo OCR options

## [0.2.7] - 2026-03-18

### Changes

- Tighten XMP extension schema validation for PDF/A
- Log non-compliant veraPDF prechecks as warnings

## [0.2.6] - 2026-03-13

### Bug Fixes

- Fix remaining OCR autorotation error for pypdfium when existing `/Rotate` must be composed with a 180 degree OSD correction

### Changes

- Add regression coverage for pypdfium rotation composition with existing page rotation

## [0.2.5] - 2026-03-13

### Bug Fixes

- Fix OCR autorotation for pages with existing `/Rotate` when OSD applies a 180 degree correction
- Fix CLI veraPDF validation detection when veraPDF is configured through `VERAPDF_PATH`

### Changes

- Add visible-page OCR rotation normalization and regression coverage for rotated OCR pages

## [0.2.4] - 2026-03-13

### Bug Fixes

- Fix OCR preprocessing for boolean and non-uint8 image arrays

## [0.2.3] - 2026-03-12

### Changes

- Improve veraPDF launcher discovery when `VERAPDF_PATH` points to an installation directory
- Reduce converter progress logging for OCR, ISO detection, and font repair details to DEBUG level

### Bug Fixes

- Fix console-safe status output on Windows consoles with limited encodings
- Fix veraPDF error reporting when the configured executable cannot be started

## [0.2.2] - 2026-03-11

### Bug Fixes

- Remove invalid non-catalog XMP metadata

### Changes

- Switch OCR activation to per-page behavior
- Log non-compliant veraPDF results as ERROR

## [0.2.1] - 2026-02-24

### Changes

- Reduce `best` OCR preset oversampling target to limit output file size growth
- Update OCR best rotation confidence threshold

### Documentation

- Update OCR documentation and usage explanations
- Update README

### CI / Build

- Add `.pdf` files to `.gitignore`

## [0.2.0] - 2026-02-23

### Features

- Add `--ocr-force` flag to re-OCR documents with existing text
- Add recursive conversion of non-compliant embedded PDFs for ISO 19005-2 rule 6.8-5
- Add PUA ActualText sanitizer for ISO 19005 rule 6.2.11.7.3-1
- Add font structure sanitizer for ISO 19005-2 rules 6.2.11.2-1 through 6.2.11.2-7
- Add TrueType font encoding sanitizer for ISO 19005-2 rules 6.2.11.6-1 through 6.2.11.6-4
- Add non-standard inline filter sanitizer for ISO 19005-2 rule 6.1.10-1
- Add extension schema block sanitizer for ISO 19005-2 rules 6.6.2.3.1–6.6.2.3.3
- Add pdfaSchema valueType validation
- Add pdfaField entry validation

### Bug Fixes

- Fix odd-length hex strings
- Fix zero-size annotation exemption logic for ISO 19005-2 rule 6.3.3
- Fix missing /Widths array for ISO 19005-2 rule 6.2.11.2-6
- Fix invalid BitsPerComponent by re-encoding image pixel data for rules 6.2.8-4 and 6.2.8-5
- Fix Indexed colour space lookup table size mismatch with lossy repair

### Changes

- Sanitize catalog /Perms per spec
- Repair hex strings with invalid characters for ISO 19005-2 rule 6.1.6-2
- Replace ICC repair skip with ordered recovery for unsupported /N
- Replace DeviceN > 32 colorants error with lossy alternate substitution for ISO 19005-2 rule 6.1.13-9
- Add overflow real clamping for ISO 19005-2 rule 6.1.13-2
- Add indirect object count limit check for ISO 19005-2 rule 6.1.13-7
- Add CID value range validation for ISO 19005-2 rule 6.1.13-10
- Exempt Link and zero-size annotations from /AP requirement for ISO 19005-2 rule 6.3.3-1
- Add missing /Group to transparent pages for PDF/A compliance (rule 6.2.10-2)
- Unify DestOutputProfile indirect objects across OutputIntents (rule 6.2.3-2)
- Strip DestOutputProfileRef from PDF/X OutputIntents for PDF/A compliance
- Preserve annotations through OCR by stripping before and restoring after

## [0.1.4] - 2026-02-18

### Bug Fixes

- Preserve image metadata (DPI) in OCR preprocessing so ocrmypdf receives correct resolution

### Changes

- Fix E501 line-too-long lint errors in docstrings and tests

### Documentation

- Simplify installation section in README

### CI / Build

- Add macOS to CI test matrix

## [0.1.3] - 2026-02-17

### Bug Fixes

- Fix OCR language parameter to pass list to ocrmypdf instead of string
- Disable ocrmypdf optimizer to avoid missing tool errors on Windows

## [0.1.2] - 2026-02-17

### Changes

- Use pypdfium rasterizer and plain PDF output for OCR, letting pdftopdfa handle PDF/A compliance exclusively
- Remove `remove_background` parameter from all OCR presets
- Add `pypdfium2` as an OCR dependency

## [0.1.1] - 2026-02-17

### Features

- Replace unpaper-based cleaning with OpenCV image preprocessing
- Accept directory paths for `TESSERACT_PATH` and `VERAPDF_PATH`

### Bug Fixes

- Suppress pikepdf "Unexpected end of stream" warnings during content stream parsing
- Fix formatting in `ocr.py`

### Documentation

- Document pre-check behavior for already PDF/A-compliant files
- Add OpenCV to optional dependencies in README

### CI / Build

- Add PyPI publish workflow for automated releases
- Install ocr extras in CI so OpenCV tests run instead of being skipped
- Skip OpenCV filter tests when `opencv-python-headless` is not installed
- Fix import sorting in `test_ocr.py`

## [0.1.0] - 2026-02-16

- Initial release
