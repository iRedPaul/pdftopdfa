# Changelog

## [0.2.1] - 2026-02-24

### Changes

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
