# PDF/A Conformance Actions

This document describes all transformations and checks that pdftopdfa performs to
produce ISO 19005-2 (PDF/A-2) and ISO 19005-3 (PDF/A-3) compliant files.

Supported conformance levels: **2b**, **2u**, **3b**, **3u**.

---

## Table of Contents

1. [Conversion Pipeline Overview](#1-conversion-pipeline-overview)
2. [Encryption Rejection](#2-encryption-rejection)
3. [Font Processing](#3-font-processing)
4. [Sanitizers](#4-sanitizers)
5. [XMP Metadata](#5-xmp-metadata)
6. [ICC Color Profiles and Color Spaces](#6-icc-color-profiles-and-color-spaces)
7. [Extensions Dictionary](#7-extensions-dictionary)
8. [Post-Save File Structure](#8-post-save-file-structure)
9. [Optional OCR](#9-optional-ocr)
10. [Validation](#10-validation)
11. [Level Differences](#11-level-differences)

---

## 1. Conversion Pipeline Overview

The conversion runs in sequential steps:

| Step | Description |
|------|-------------|
| 0 | Pre-check: detect if already PDF/A via veraPDF; skip if valid |
| 0 | Reject encrypted PDFs |
| 1 | Optional OCR for image-based pages |
| 2 | Open PDF; detect other ISO standards (PDF/X, PDF/UA, etc.) |
| 3 | Embed missing fonts |
| 3.5 | Add ToUnicode CMaps to all embedded fonts |
| 3.7 | Subset embedded fonts |
| 3.8 | Fix font encoding issues (symbolic TrueType) |
| 4 | Run all sanitizers |
| 5 | Synchronize XMP metadata |
| 5.5 | Add Extensions dictionary (PDF/A-3) |
| 6 | Detect color spaces; embed ICC OutputIntent profiles |
| 6+ | Late-pass structure limit sanitization |
| 7 | Create output directory |
| 8 | Save with forced PDF version 1.7 and deterministic /ID |
| 8.2 | Ensure binary comment line; truncate data after %%EOF |
| 8.5 | Verify PDF header version and trailer /ID |
| 9 | Optional veraPDF validation of output |

---

## 2. Encryption Rejection

**ISO 19005 (general)** -- Encryption is forbidden in PDF/A.

Encrypted PDFs are detected and rejected before conversion begins.

---

## 3. Font Processing

### 3.1. Font Embedding

**ISO 19005, 6.3.1** -- All fonts must be embedded.

- Replaces the 14 Standard PDF fonts (Helvetica, Times-Roman, Courier, Symbol,
  ZapfDingbats, and their variants) with metrically compatible Liberation fonts.
- Embeds fonts as CIDFontType2 with Identity-H encoding.
- Builds proper font dictionaries with CIDSystemInfo, FontDescriptor, and all
  required entries.

### 3.2. ToUnicode CMap Generation

**ISO 19005-2/3, 6.2.11.7.2** -- ToUnicode CMap required for text extraction.

- Generates ToUnicode CMaps for all standard encodings: WinAnsi, MacRoman,
  Standard, Symbol, ZapfDingbats.
- Generates CMaps from encoding dictionaries with /Differences arrays.
- Generates CMaps for CIDFonts via CIDToGIDMap + cmap table lookup.
- Generates CMaps for Type3 fonts.
- Resolves glyph names to Unicode via the Adobe Glyph List.
- Filters forbidden Unicode values: U+0000, U+FEFF, U+FFFE, and surrogates
  (U+D800-U+DFFF), replacing them with Private Use Area codepoints.
- For Unicode levels (2u/3u): fills ToUnicode gaps so every used glyph has a
  mapping.

### 3.3. Font Subsetting

- Subsets embedded fonts to include only glyphs actually used in the document.
- Preserves the .notdef glyph.
- Reduces file size.

### 3.4. Symbolic TrueType Encoding Fix

**ISO 19005-2, 6.2.11.6**

- Fixes encoding issues in symbolic TrueType fonts.

---

## 4. Sanitizers

### 4.1. JavaScript Removal

**ISO 19005-2, 6.6.1** -- JavaScript is forbidden.

Removes the `/Root/Names/JavaScript` named tree.

### 4.2. Non-Compliant Actions

**ISO 19005-2, 6.6.1/6.6.2** -- Only specific action types are permitted.

- Removes forbidden action types (Launch, Sound, Movie, ImportData, ResetForm,
  etc.) from OpenAction, page/document /AA, annotations, outlines, and AcroForm
  fields.
- Removes /A and /AA from Widget annotations and form fields (Rule 6.4.1).
- Sanitizes /Next chains on compliant actions to strip non-compliant follow-ups.
- Restricts Named actions to NextPage, PrevPage, FirstPage, LastPage.
- SubmitForm only allowed if flags indicate PDF or XFDF format.
- Validates GoTo destinations and removes those referencing non-existent pages.

### 4.3. XFA Form Removal

**ISO 19005** -- XFA is forbidden in all PDF/A levels.

Removes `/XFA` and `/NeedsRendering` from `/AcroForm`.

### 4.4. Annotation Handling

#### 4.4.1. Forbidden Annotation Subtypes

**ISO 19005-2, Rule 6.3.1** -- Sound, Movie, Screen, 3D, RichMedia, and TrapNet
annotations are forbidden.

Removes annotations with forbidden or undefined subtypes.

#### 4.4.2. Annotation Flags

**ISO 19005-2**

- Sets the Print flag on all annotations.
- Clears Hidden, Invisible, NoView, and ToggleNoView flags.
- Sets NoZoom and NoRotate on Text annotations (6.5.2).

#### 4.4.3. Appearance Streams

**ISO 19005-2/3, 6.5.3** -- All annotations (except Popup) must have /AP/N.

- Creates appearance streams for annotations missing /AP or /AP/N.
- For Widget annotations: generates visible appearance (text, border, background).
- For non-Widget annotations: creates a minimal empty Form XObject.
- Collapses state dictionaries to a single stream for non-Btn widgets (Rule 6.3.3).

#### 4.4.4. Appearance Dictionary Cleanup

**Rule 6.3.3** -- /AP may only contain /N (normal appearance).

Removes /R (rollover) and /D (down) appearance entries.

#### 4.4.5. Button Appearance Subdictionaries

**Rule 6.3.3** -- Btn widget /AP/N must be a state subdictionary.

Wraps bare Stream in state Dictionary for Btn fields.

#### 4.4.6. NeedAppearances Flag

PDF/A forbids `/NeedAppearances=true` (viewers must not regenerate appearances).

Removes /NeedAppearances from /AcroForm.

#### 4.4.7. Annotation Opacity

**ISO 19005-2, 6.5.3** -- Annotation-level /CA must be 1.0.

Sets non-1.0 /CA values to 1.0.

#### 4.4.8. Annotation Colors

**ISO 19005-2** -- Device-dependent colors in annotations are forbidden.

Removes /C (border color) and /IC (interior color) arrays from annotations.

### 4.5. Catalog Sanitization

#### 4.5.1. Forbidden Catalog Entries

**ISO 19005-2, 6.1.10-6.1.13**

Removes /Perms, /Requirements, /Collection, /NeedsRendering, /Threads, and
/SpiderInfo from the Document Catalog.

#### 4.5.2. Catalog /Version

**ISO 19005-2, 6.1.2** -- Effective version must not exceed 1.7.

Removes or overwrites /Version if it exceeds the required version.

#### 4.5.3. Forbidden Viewer Preferences

**ISO 19005-2, 6.1.2**

Removes /ViewArea, /ViewClip, /PrintArea, and /PrintClip from /ViewerPreferences.

#### 4.5.4. Forbidden Name Dictionary Entries

**ISO 19005-2, 6.1.11**

Removes /AlternatePresentations from the /Names dictionary.

#### 4.5.5. Forbidden Page Entries

**Rule 6.10**

Removes /PresSteps and /Duration from page dictionaries.

#### 4.5.6. Language Tag

**ISO 19005-2, 6.7.3** -- /Lang required in the Document Catalog.

Sets /Lang from XMP `dc:language`, or `"und"` (undetermined) as fallback.
Validates BCP 47 syntax.

#### 4.5.7. MarkInfo

**ISO 19005-2, 6.7.1**

Ensures a /MarkInfo dictionary with /Marked key exists.

### 4.6. Stream Filters

#### 4.6.1. LZW to FlateDecode

**ISO 19005-2, 6.1.8** -- LZWDecode is forbidden.

Re-encodes all LZW-compressed streams (including inline images) to FlateDecode.

#### 4.6.2. Crypt Filter Removal

**ISO 19005-2, 6.1.8** -- Crypt filter is forbidden.

Removes Crypt filters from all streams (pikepdf transparently decrypts data).

#### 4.6.3. External Stream Keys

**ISO 19005-2, 6.1.7.1** -- /F, /FFilter, /FDecodeParms are forbidden.

Removes these keys from all stream dictionaries (all data must be self-contained).

#### 4.6.4. Stream Length Repair

**ISO 19005-2, 6.1.7.1** -- /Length must be accurate.

Re-encodes non-image streams to force correct /Length on save.

#### 4.6.5. Filter Name Normalization

Normalizes abbreviated filter names (e.g. /AHx to /ASCIIHexDecode, /Fl to
/FlateDecode).

### 4.7. JBIG2

**ISO 19005-2, 6.1.4.2** -- External globals and refinement coding are forbidden.

- Inlines external JBIG2 globals data into the page stream.
- Detects forbidden refinement segments (types 40, 42, 43) and re-encodes to
  FlateDecode.

### 4.8. JPEG2000 (JPX)

**ISO 19005-2, 6.1.4.3** -- Exactly one colr box with METH=1 or METH=2; channel
count and bit depth constraints.

- Parses JP2 headers; fixes colr boxes (removes extra, ensures valid METH).
- Repairs ihdr channel count and bit depth from codestream SIZ marker.
- Wraps bare JPEG2000 codestreams in a minimal JP2 container with correct colr box.
- Fallback: re-encodes to FlateDecode (lossless).

### 4.9. Color Space Sanitization

**ISO 19005-2, multiple color clauses**

- Validates embedded ICC profiles: signature, class, version (up to v4 for
  PDF/A-2+), and component count (/N).
- Repairs invalid ICC profiles by replacing with built-in profiles matching /N.
- Corrects missing or mismatched /N values.
- Validates Indexed color space lookup table sizes.
- Ensures DeviceN Colorants dictionary completeness (adds missing spot color
  entries).
- Normalizes Separation arrays with the same name to share alternate and
  tintTransform.
- Traverses pages, Form XObjects, annotation AP streams, and Type3 font
  resources.

### 4.10. Extended Graphics State (ExtGState)

**ISO 19005-2, 6.2.8 and 6.4**

- Removes /TR (transfer function).
- Removes /TR2 unless its value is /Default.
- Removes /HTP (halftone phase).
- Sanitizes /HT (halftone) dictionaries: enforces HalftoneType 1 or 5 only,
  removes /HalftoneName, manages /TransferFunction per colorant rules.
- Validates /RI (rendering intent): must be one of RelativeColorimetric,
  AbsoluteColorimetric, Perceptual, or Saturation.
- Validates /BM (blend mode): replaces invalid values with /Normal.
- Clamps /CA and /ca (opacity) to [0.0, 1.0].
- Validates /SMask: must be /None or a valid soft mask dictionary (with /S in
  {Alpha, Luminosity}, /G as Form XObject, no /TR).
- Resets /OPM to 0 when ICCBased CMYK with overprint is enabled (6.2.4.2).
- Removes /TR and /TR2 from Shading dictionaries and PatternType 2 shading
  patterns (6.2.5).
- Traverses pages, Form XObjects, AP streams, and Type3 font resources.

### 4.11. Rendering Intent and Content Streams

**ISO 32000-1, ISO 19005-2**

- Replaces invalid `ri` operator operands with /RelativeColorimetric.
- Removes undefined/non-standard content stream operators (only ISO 32000-1
  defined operators are allowed).
- Ensures content streams have explicit associated /Resources dictionaries (not
  inherited).
- Fixes invalid /Intent on Image XObjects.

### 4.12. Optional Content (Layers)

**ISO 19005-2/3, various OC constraints**

- Removes /AS entries (auto-state triggers) from all OC configurations.
- Fixes OCG /Intent to /View.
- Creates default /D configuration if missing.
- Adds /Name to /D configuration and alternate configurations if missing.
- Fixes /ListMode to /AllPages.
- Fixes /BaseState to /ON.
- Ensures all OCGs are listed in the /OCGs array.
- Fixes /RBGroups (radio button groups) for consistency.
- Adds /Name to OCG dictionaries if missing.
- Ensures all OCGs appear in the /Order array.

### 4.13. Page Boxes

**ISO 32000-1 and ISO 19005-2/3**

- Resolves inherited /MediaBox from parent page tree nodes.
- Normalizes box coordinates (ensures x1 < x2, y1 < y2).
- Clips CropBox, BleedBox, TrimBox, and ArtBox to MediaBox boundaries.
- Removes malformed boxes (wrong element count, non-numeric values, degenerate
  dimensions).
- Validates page dimensions (3.0 to 14,400 points).

### 4.14. XObjects

**ISO 19005-2, 6.2.4/6.2.8/6.2.9**

- Removes PostScript XObjects (/Subtype /PS) and Reference XObjects
  (/Subtype /Ref).
- Removes /Alternates arrays from all XObjects.
- Removes /OPI dictionaries from all XObjects.
- Removes /Ref, /Subtype2=/PS, and /PS keys from Form XObjects (6.2.9).
- Sets /Interpolate to false on all Image XObjects and inline images (6.2.8).
- Validates BitsPerComponent (must be 1, 2, 4, 8, or 16; masks must be 1).
- Recurses into nested Form XObjects and AP streams.

### 4.15. Digital Signatures

**Rule 6.4.3** -- Signatures become invalid after conversion.

- Detects signature dictionaries.
- Neutralizes live signature references (removes /Type, /Filter, /SubFilter,
  /ByteRange, /Contents, /Reference, etc.).
- Fixes /SigFlags in /AcroForm.

### 4.16. CIDFont Structure

**ISO 19005-2, 6.2.11.3.1-6.2.11.4.2, 6.3.5, 6.3.7**

- Fixes CIDSystemInfo in CIDFont dictionaries (must be consistent with CMap
  encoding).
- Adds /CIDToGIDMap to CIDFontType2 fonts if missing (6.2.11.3.2).
- Removes /CIDSet from FontDescriptor (6.2.11.4.2).
- Removes Type1 /CharSet from FontDescriptor (6.3.7).
- Fixes /FontName vs /BaseFont consistency in FontDescriptor (6.3.5).

### 4.17. Font .notdef Glyph

**ISO 19005-2, 6.3.3** -- Every embedded font must contain a .notdef glyph.

Inserts a minimal empty .notdef glyph into embedded fonts that lack one.

### 4.18. Glyph Coverage

**ISO 19005-2, 6.2.11.4.1** -- Embedded fonts must define all referenced glyphs.

Adds minimal empty glyph outlines for referenced but missing glyph IDs.

### 4.19. Font Width Validation

**ISO 19005-2, 6.3.7** -- Declared widths must match the embedded font program.

- Extracts glyph widths from embedded TrueType, OpenType, and CFF font programs.
- Compares against declared /Widths or /W arrays.
- Corrects mismatched widths (tolerance of +/-1).
- Handles simple fonts and CIDFonts separately.

### 4.20. ToUnicode Value Sanitization

**veraPDF Rule 6.2.11.7.2** -- Forbidden Unicode values: U+0000, U+FEFF, U+FFFE,
surrogates (U+D800-U+DFFF).

- Scans all existing ToUnicode CMaps for forbidden values.
- Replaces forbidden values with Private Use Area codepoints.
- For Unicode levels (2u/3u): fills ToUnicode gaps so every used glyph has a
  Unicode mapping.

### 4.21. .notdef Usage Removal

**ISO 19005-2, 6.2.11.8** -- References to .notdef glyph are forbidden in text.

Strips character codes that resolve to .notdef from Tj, TJ, ', and " operators
in all content streams (pages, Form XObjects, Tiling Patterns, AP streams, Type3
CharProcs).

### 4.22. Embedded Files

#### PDF/A-2 (2b/2u):

**ISO 19005-2** -- Embedded files must themselves be PDF/A-1 or PDF/A-2.

Removes non-compliant embedded files (validates via XMP and veraPDF); keeps
compliant ones.

#### All levels:

**ISO 19005-2/3**

- Ensures /AFRelationship key on all FileSpec dictionaries (defaults to
  /Unspecified).
- Builds and maintains /Root/AF array referencing all FileSpecs.
- Ensures /Subtype (MIME type) on all embedded file streams.
- Ensures /Params with /ModDate on all embedded file streams.
- Ensures /UF (Unicode filename) and /F on all FileSpec dictionaries.
- Ensures /Desc (description) on all FileSpec dictionaries.
- Sanitizes LZW/Crypt filters on embedded file streams.

### 4.23. Structure Limits

**Rule 6.1.13, Rule 6.1.6, Rule 6.1.8**

- Truncates overlong string objects (max 32,767 bytes).
- Shortens overlong name objects (max 127 bytes).
- Repairs invalid UTF-8 in name objects.
- Clamps out-of-range integer operands to [-2,147,483,648, 2,147,483,647].
- Normalizes near-zero real operands to 0.
- Rebalances q/Q graphics-state operator nesting (max depth 28).
- Fixes odd-length hexadecimal strings in content streams.

Also runs as a late pass after color profile embedding.

---

## 5. XMP Metadata

**ISO 19005-2, 6.6**

- Extracts metadata from the PDF Info dictionary (Title, Author, Subject,
  Keywords, Creator, Producer, dates, Trapped).
- Creates XMP metadata with:
  - `pdfaid:part` and `pdfaid:conformance` (PDF/A identification).
  - `dc:title`, `dc:creator`, `dc:description`, `dc:format`.
  - `xmp:CreateDate`, `xmp:ModifyDate`, `xmp:MetadataDate`.
  - `pdf:Producer`, `xmp:CreatorTool`, `pdf:Keywords`, `pdf:Trapped`.
- Preserves non-managed XMP properties from existing metadata (PDF/X, PDF/UA,
  PDF/E, PDF/VT identifications, custom namespaces).
- Validates preserved properties against predefined schema types.
- Builds extension schemas (`pdfaExtension:schemas`) for non-predefined
  properties, including those from non-catalog XMP streams (veraPDF Rule
  6.6.2.3.1).
- Normalizes structural properties (e.g. corrects `stEvt:When` to `stEvt:when`).
- Sanitizes XML control characters forbidden in XML 1.0.
- Embeds XMP as an uncompressed stream (no /Filter on metadata stream).
- Synchronizes DocInfo with XMP:
  - Removes non-standard keys from DocInfo.
  - Normalizes /Trapped to True, False, or Unknown.
  - Ensures dates are consistent.
  - Ensures /Author is non-empty.
- Sanitizes non-catalog /Metadata streams: re-serializes valid XMP, removes
  malformed ones, ensures all are uncompressed.

---

## 6. ICC Color Profiles and Color Spaces

### 6.1. Color Space Detection

Analyzes all color spaces used in the PDF: DeviceGray, DeviceRGB, DeviceCMYK,
ICCBased, Separation, DeviceN, Indexed, Pattern, CalGray, and CalRGB.

Traverses pages, Form XObjects, Image XObjects, AP streams, Type3 font
resources, and Tiling Patterns.

### 6.2. OutputIntent Profile Embedding

**ISO 19005-2, 6.2** -- Device-independent color must be ensured via OutputIntent.

- Embeds sRGB, sGray, or FOGRA39 (CMYK) ICC profiles as OutputIntents.
- Only one OutputIntent with `S=GTS_PDFA1` is allowed; selects the dominant color
  space (CMYK > RGB > Gray).
- Validates ICC profiles (signature, class, version, component count).
- Deduplicates multiple OutputIntents referencing different profiles (6.2.3).

### 6.3. Default Color Spaces

**ISO 19005-2, 6.2.4** -- Non-dominant device color spaces need default mappings.

Applies /DefaultGray, /DefaultRGB, and /DefaultCMYK entries to page Resources
pointing to ICC-based color spaces. Also applies defaults to AP stream resources.

### 6.4. Calibrated Color Space Conversion

Converts CalGray and CalRGB color spaces to equivalent ICCBased color spaces.

Traverses pages, Image XObjects, Form XObjects, and AP streams.

### 6.5. Transparency Group Color Spaces

**ISO 19005-2, 6.4** -- Transparency group /CS must not use device-dependent
color spaces.

Replaces DeviceGray, DeviceRGB, and DeviceCMYK in transparency group /CS entries
with ICCBased equivalents.

---

## 7. Extensions Dictionary

**ISO 19005-3** -- PDF/A-3 documents require an ADBE extension entry.

- Adds ADBE extension with ExtensionLevel=3 for PDF/A-3 levels (3b/3u).
- Removes non-ADBE extension entries (e.g. ISO for PDF 2.0) for all levels.

---

## 8. Post-Save File Structure

### 8.1. PDF Version

**ISO 19005-2, 6.1.2** -- Version must be 1.7 for PDF/A-2 and PDF/A-3.

Forces version 1.7 via `force_version` on save.

### 8.2. Deterministic File ID

Saves with `deterministic_id=True` for reproducible /ID values. Verifies the
trailer contains a /ID array with 2 elements.

### 8.3. Binary Comment

**ISO 19005-2, 6.1.2** -- The file must include a comment line with at least 4
bytes with values greater than 127.

Re-saves through pikepdf/QPDF if the binary comment is missing.

### 8.4. Trailing Data Truncation

**ISO 19005-2, 6.1.3** -- No data is allowed after the final `%%EOF` (except an
optional single end-of-line character).

Truncates trailing bytes after the `%%EOF` marker.

---

## 9. Optional OCR

Pre-processing step using ocrmypdf.

- Adds an invisible text layer to image-based (scanned) pages.
- Supports multiple languages and quality presets.
- Checks if the PDF already contains text before applying OCR.

---

## 10. Validation

### 10.1. Pre-Conversion Check

Detects the existing PDF/A level from XMP metadata (`pdfaid:part` +
`pdfaid:conformance`). Skips conversion if the file is already valid.

### 10.2. veraPDF Integration

Integrates with the external veraPDF CLI tool for ISO-compliant validation.

- Pre-conversion validation (checking if input is already PDF/A).
- Post-conversion validation (optional, verifying output).
- Validating embedded PDF files for PDF/A-2 compliance.
- Parses veraPDF XML output: passed/failed rules, errors, and warnings.

---

## 11. Level Differences

| Feature | 2b | 2u | 3b | 3u |
|---|:---:|:---:|:---:|:---:|
| All sanitizers | Yes | Yes | Yes | Yes |
| Font embedding | Yes | Yes | Yes | Yes |
| ToUnicode CMaps | Yes | Yes | Yes | Yes |
| ToUnicode gap filling (every glyph mapped) | No | Yes | No | Yes |
| Non-compliant embedded file removal | Yes | Yes | No | No |
| Embedded file metadata enrichment | Yes | Yes | Yes | Yes |
| ADBE Extension Level 3 | No | No | Yes | Yes |
| ICC OutputIntent profiles | Yes | Yes | Yes | Yes |
| XMP pdfaid:part=2 | Yes | Yes | No | No |
| XMP pdfaid:part=3 | No | No | Yes | Yes |
| PDF version forced to 1.7 | Yes | Yes | Yes | Yes |
| Binary comment and %%EOF truncation | Yes | Yes | Yes | Yes |
