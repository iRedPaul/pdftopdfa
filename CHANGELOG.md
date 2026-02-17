# Changelog

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
