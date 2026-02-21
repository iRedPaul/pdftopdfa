# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Embedded file handling for PDF/A compliance."""

import logging
import mimetypes
import re
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..metadata import _format_pdf_date
from ..utils import resolve_indirect as _resolve_indirect
from ..validator import detect_pdfa_level
from ..verapdf import is_verapdf_available, validate_with_verapdf

logger = logging.getLogger(__name__)

# PDF date format: D:YYYYMMDDHHmmSS±HH'mm' (various optional parts)
_PDF_DATE_RE = re.compile(
    r"^D:\d{4}"  # D:YYYY
    r"(\d{2}"  # MM
    r"(\d{2}"  # DD
    r"(\d{2}"  # HH
    r"(\d{2}"  # mm
    r"(\d{2})?"  # SS
    r")?"
    r")?"
    r")?"
    r")?"
    r"([Z+\-]"  # timezone
    r"(\d{2}'\d{2}')?"  # HH'mm'
    r")?$"
)


def _is_valid_pdf_date(date_str: str) -> bool:
    """Check whether *date_str* matches the PDF date format."""
    return bool(_PDF_DATE_RE.match(date_str))


VALID_AF_RELATIONSHIPS = frozenset(
    {
        "/Source",
        "/Data",
        "/Supplement",
        "/Alternative",
        "/Unspecified",
    }
)


def _is_pdfa_compliant_embedded(filespec: object) -> bool:
    """Checks if an embedded file in a FileSpec is PDF/A-1 or PDF/A-2 compliant.

    Args:
        filespec: A pikepdf FileSpec dictionary object.

    Returns:
        True if the embedded file is a PDF/A-1 or PDF/A-2 document.
    """
    try:
        resolved = _resolve_indirect(filespec)

        # Extract /EF dictionary
        ef = resolved.get("/EF")
        if ef is None:
            return False

        ef = _resolve_indirect(ef)

        # Get embedded file stream: prefer /UF, fall back to /F
        stream = ef.get("/UF")
        if stream is None:
            stream = ef.get("/F")
        if stream is None:
            return False

        stream = _resolve_indirect(stream)

        # Read embedded file data
        data = bytes(stream.read_bytes())
        if not data:
            return False

        # Quick check: must start with PDF magic bytes
        if not data[:5] == b"%PDF-":
            return False

        # Open as PDF and check PDF/A level via XMP
        with pikepdf.open(BytesIO(data)) as embedded_pdf:
            level = detect_pdfa_level(embedded_pdf)
            if level is None:
                return False
            # PDF/A-1 and PDF/A-2 are allowed in PDF/A-2
            if level[0] not in ("1", "2"):
                return False

        # XMP claims PDF/A compliance — verify with veraPDF if available
        if is_verapdf_available():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                result = validate_with_verapdf(tmp_path, flavour=level)
                return result.compliant
            except Exception as e:
                logger.warning(
                    "veraPDF validation failed for embedded file, "
                    "falling back to XMP result: %s",
                    e,
                )
                return True
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        return True

    except Exception as e:
        logger.debug("Error checking embedded file compliance: %s", e)
        return False


def _try_convert_embedded_pdf_to_pdfa2(data: bytes) -> bytes | None:
    """Attempt to convert embedded PDF bytes to PDF/A-2b.

    Uses a deferred import of convert_to_pdfa to avoid a circular dependency
    (converter → sanitizers/__init__ → files → converter).

    Args:
        data: Raw bytes of an embedded PDF file.

    Returns:
        Converted PDF/A-2b bytes on success, or None if conversion failed.
    """
    # Deferred import breaks the circular dependency at module load time.
    from ..converter import convert_to_pdfa  # noqa: PLC0415

    tmp_in_path: Path | None = None
    tmp_out_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
            tmp_in.write(data)
            tmp_in_path = Path(tmp_in.name)

        tmp_out_path = tmp_in_path.with_name(tmp_in_path.stem + "_pdfa2b.pdf")
        result = convert_to_pdfa(tmp_in_path, tmp_out_path, level="2b")
        if result.success:
            converted = tmp_out_path.read_bytes()
            logger.debug(
                "Converted embedded PDF to PDF/A-2b (%d → %d bytes)",
                len(data),
                len(converted),
            )
            return converted
        logger.debug("Embedded PDF conversion failed: %s", result.error)
        return None
    except Exception as e:
        logger.debug("Error converting embedded PDF to PDF/A-2b: %s", e)
        return None
    finally:
        if tmp_in_path is not None:
            tmp_in_path.unlink(missing_ok=True)
        if tmp_out_path is not None:
            tmp_out_path.unlink(missing_ok=True)


def _update_embedded_stream(ef: object, new_data: bytes) -> None:
    """Replace the data in an /EF embedded file stream.

    Updates the /UF (preferred) or /F stream with new_data and refreshes
    /Params size and modification date if the /Params dictionary is present.

    Args:
        ef: A resolved /EF dictionary (embedded file streams dictionary).
        new_data: The replacement bytes (e.g. a converted PDF/A-2b file).
    """
    ef = _resolve_indirect(ef)

    # Prefer /UF (Unicode filename stream), fall back to /F
    stream = ef.get("/UF")
    if stream is None:
        stream = ef.get("/F")
    if stream is None:
        return

    stream = _resolve_indirect(stream)
    stream.write(new_data)  # pikepdf re-compresses and updates /Length automatically

    # Refresh /Params metadata if present
    params = stream.get("/Params")
    if params is not None:
        params = _resolve_indirect(params)
        if isinstance(params, Dictionary):
            params[Name.Size] = len(new_data)
            params[Name.ModDate] = pikepdf.String(_format_pdf_date(datetime.now(UTC)))


_MAX_NAME_TREE_DEPTH = 32


def _iter_name_tree_values(node: object, *, _depth: int = 0) -> Iterator[object]:
    """Yield all values from a PDF Name Tree node.

    Name Trees (PDF spec §7.9.6) can be hierarchical: intermediate nodes
    have /Kids arrays of child nodes, and only leaf nodes have /Names arrays.
    This generator traverses both /Names and /Kids so that PDFs with balanced
    trees are fully handled.

    Args:
        node: A pikepdf Dictionary representing a Name Tree node.
        _depth: Internal recursion depth counter (stops at _MAX_NAME_TREE_DEPTH).
    """
    if _depth >= _MAX_NAME_TREE_DEPTH:
        logger.debug("Name Tree depth limit reached (%d)", _MAX_NAME_TREE_DEPTH)
        return

    try:
        resolved = _resolve_indirect(node)
    except Exception:
        return

    # Leaf node: /Names array with [name, value, name, value, ...]
    names_array = resolved.get("/Names")
    if names_array is not None:
        try:
            for i in range(1, len(names_array), 2):
                yield names_array[i]
        except Exception:
            pass

    # Intermediate node: /Kids array of child nodes
    kids = resolved.get("/Kids")
    if kids is not None:
        try:
            for child in kids:
                yield from _iter_name_tree_values(child, _depth=_depth + 1)
        except Exception:
            pass


def _iter_name_tree_pairs(
    node: object, *, _depth: int = 0
) -> Iterator[tuple[object, object]]:
    """Yield (name, value) pairs from a PDF Name Tree node.

    Same traversal as _iter_name_tree_values but yields both the key and value.
    Used by remove_non_compliant_embedded_files which needs to rebuild the array.

    Args:
        node: A pikepdf Dictionary representing a Name Tree node.
        _depth: Internal recursion depth counter (stops at _MAX_NAME_TREE_DEPTH).
    """
    if _depth >= _MAX_NAME_TREE_DEPTH:
        logger.debug("Name Tree depth limit reached (%d)", _MAX_NAME_TREE_DEPTH)
        return

    try:
        resolved = _resolve_indirect(node)
    except Exception:
        return

    # Leaf node: /Names array with [name, value, name, value, ...]
    names_array = resolved.get("/Names")
    if names_array is not None:
        try:
            for i in range(0, len(names_array), 2):
                if i + 1 < len(names_array):
                    yield (names_array[i], names_array[i + 1])
        except Exception:
            pass

    # Intermediate node: /Kids array of child nodes
    kids = resolved.get("/Kids")
    if kids is not None:
        try:
            for child in kids:
                yield from _iter_name_tree_pairs(child, _depth=_depth + 1)
        except Exception:
            pass


def _iter_all_filespecs_by_scan(pdf: Pdf) -> Iterator[object]:
    """Yield all FileSpec objects by scanning every object in the PDF.

    Finds FileSpecs that may be missed by Name Tree / annotation traversal,
    such as those referenced only from page-level /AF arrays or other objects.

    Args:
        pdf: Opened pikepdf PDF object.
    """
    seen_objgen: set[tuple[int, int]] = set()
    for obj in pdf.objects:
        try:
            resolved = _resolve_indirect(obj)
            if not isinstance(resolved, Dictionary):
                continue
            # FileSpec: has /Type /Filespec or contains /EF
            type_val = resolved.get("/Type")
            is_filespec = type_val is not None and str(type_val) == "/Filespec"
            has_ef = resolved.get("/EF") is not None
            if not is_filespec and not has_ef:
                continue
            # Deduplicate
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    continue
                seen_objgen.add(og)
            yield obj
        except Exception:
            continue


def _iter_all_filespecs(pdf: Pdf) -> Iterator[object]:
    """Yield all FileSpec objects found in the PDF.

    Combines Name Tree traversal, FileAttachment annotation scanning,
    and full pdf.objects scan to find all FileSpec dictionaries.
    The full scan catches indirect FileSpecs in page-level /AF arrays
    and other locations; the traversal catches direct (inline) FileSpecs
    in the Name Tree or annotations.

    Args:
        pdf: Opened pikepdf PDF object.
    """
    seen_objgen: set[tuple[int, int]] = set()

    def _dedup(obj: object) -> object | None:
        try:
            resolved = _resolve_indirect(obj)
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return None
                seen_objgen.add(og)
            return obj
        except Exception:
            return None

    # 1. FileSpecs from /Root/Names/EmbeddedFiles (traverse Name Tree)
    try:
        if "/Names" in pdf.Root:
            names = _resolve_indirect(pdf.Root.Names)
            if "/EmbeddedFiles" in names:
                embedded = _resolve_indirect(names.EmbeddedFiles)
                for obj in _iter_name_tree_values(embedded):
                    result = _dedup(obj)
                    if result is not None:
                        yield result
    except Exception as e:
        logger.debug("Error reading EmbeddedFiles: %s", e)

    # 2. FileSpecs from FileAttachment annotations on all pages
    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue
            for annot in annots:
                try:
                    resolved = _resolve_indirect(annot)
                    subtype = resolved.get("/Subtype")
                    if subtype is not None and str(subtype) == "/FileAttachment":
                        fs = resolved.get("/FS")
                        if fs is not None:
                            result = _dedup(fs)
                            if result is not None:
                                yield result
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Error processing annotations on page %d: %s", page_num, e)

    # 3. Full pdf.objects scan for orphan indirect FileSpecs
    for obj in _iter_all_filespecs_by_scan(pdf):
        result = _dedup(obj)
        if result is not None:
            yield result


def _cleanup_af_arrays(pdf: Pdf, removed_objgens: set[tuple[int, int]]) -> None:
    """Remove entries from /AF arrays whose objgen is in removed_objgens.

    Scans /Root/AF and each page's /AF. Deletes empty /AF arrays.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        removed_objgens: Set of (objnum, gen) tuples to remove.
    """
    if not removed_objgens:
        return

    def _clean_af(owner: object, key: str = "/AF") -> None:
        try:
            af = owner.get(key)
            if af is None:
                return
            af = _resolve_indirect(af)
            indices_to_remove = []
            for i, entry in enumerate(af):
                try:
                    resolved = _resolve_indirect(entry)
                    try:
                        og = resolved.objgen
                    except (AttributeError, ValueError, TypeError):
                        og = (0, 0)
                    if og != (0, 0) and og in removed_objgens:
                        indices_to_remove.append(i)
                except Exception:
                    continue
            for i in reversed(indices_to_remove):
                del af[i]
            if len(af) == 0:
                del owner[key]
        except Exception:
            pass

    # Clean /Root/AF
    _clean_af(pdf.Root)

    # Clean each page's /AF
    for page in pdf.pages:
        _clean_af(page)

    # Clean /AF on annotations
    for page in pdf.pages:
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue
            for annot in annots:
                try:
                    resolved = _resolve_indirect(annot)
                    _clean_af(resolved)
                except Exception:
                    continue
        except Exception:
            continue


def remove_non_compliant_embedded_files(pdf: Pdf) -> dict[str, int]:
    """Removes non-PDF/A-compliant embedded files, keeps compliant ones.

    PDF/A-2 (ISO 19005-2) allows embedded files that are themselves
    PDF/A-1 or PDF/A-2 compliant. This function checks each embedded
    file, first attempting to convert non-compliant PDFs to PDF/A-2b.
    Only if conversion fails (or the file is not a PDF) does it fall back
    to removing the embedded content.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with 'removed', 'kept', and 'converted' counts.
    """
    removed = 0
    kept = 0
    converted = 0
    processed_filespecs: set[tuple[int, int]] = set()

    # 1. Process EmbeddedFiles from Names (traverse full Name Tree)
    try:
        if "/Names" in pdf.Root:
            names = _resolve_indirect(pdf.Root.Names)
            if "/EmbeddedFiles" in names:
                embedded = _resolve_indirect(names.EmbeddedFiles)
                new_names: list[object] = []
                for name, filespec in _iter_name_tree_pairs(embedded):
                    try:
                        fs_resolved = _resolve_indirect(filespec)
                        og = fs_resolved.objgen
                    except (AttributeError, ValueError, TypeError):
                        og = (0, 0)
                    if og != (0, 0):
                        processed_filespecs.add(og)
                    if _is_pdfa_compliant_embedded(filespec):
                        new_names.append(name)
                        new_names.append(filespec)
                        kept += 1
                        logger.debug("Kept compliant embedded file: %s", str(name))
                    else:
                        # Not compliant — try to convert it to PDF/A-2b first
                        _converted = False
                        try:
                            ef_obj = fs_resolved.get("/EF")
                            if ef_obj is not None:
                                ef = _resolve_indirect(ef_obj)
                                stream = ef.get("/UF") or ef.get("/F")
                                if stream is not None:
                                    raw = bytes(_resolve_indirect(stream).read_bytes())
                                    if raw[:5] == b"%PDF-":
                                        new_data = _try_convert_embedded_pdf_to_pdfa2(
                                            raw
                                        )
                                        if new_data is not None:
                                            _update_embedded_stream(ef, new_data)
                                            new_names.append(name)
                                            new_names.append(filespec)
                                            converted += 1
                                            logger.debug(
                                                "Converted non-compliant embedded "
                                                "file: %s",
                                                str(name),
                                            )
                                            _converted = True
                        except Exception:
                            pass
                        if not _converted:
                            try:
                                if "/EF" in fs_resolved:
                                    del fs_resolved["/EF"]
                            except Exception:
                                pass
                            removed += 1
                            logger.debug(
                                "Removed non-compliant embedded file: %s",
                                str(name),
                            )

                if new_names:
                    # Flatten tree: store all entries in root /Names
                    embedded["/Names"] = Array(new_names)
                    # Remove stale /Kids and /Limits from root node
                    for key in ("/Kids", "/Limits"):
                        if key in embedded:
                            del embedded[key]
                else:
                    del names["/EmbeddedFiles"]
                    logger.debug("All embedded files removed, deleted EmbeddedFiles")
    except Exception as e:
        logger.debug("Error processing EmbeddedFiles: %s", e)

    # 2. Process FileAttachment annotations
    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue

            indices_to_remove = []
            for i, annot in enumerate(annots):
                try:
                    resolved = _resolve_indirect(annot)
                    subtype = resolved.get("/Subtype")
                    if subtype is not None and str(subtype) == "/FileAttachment":
                        fs = resolved.get("/FS")
                        if fs is not None:
                            try:
                                fs_resolved = _resolve_indirect(fs)
                                og = fs_resolved.objgen
                            except (AttributeError, ValueError, TypeError):
                                og = (0, 0)
                            if og != (0, 0) and og in processed_filespecs:
                                continue
                            if og != (0, 0):
                                processed_filespecs.add(og)
                        if fs is not None and _is_pdfa_compliant_embedded(fs):
                            kept += 1
                        else:
                            # Not compliant — try to convert it to PDF/A-2b first
                            _converted = False
                            if fs is not None:
                                try:
                                    ef_obj = fs_resolved.get("/EF")
                                    if ef_obj is not None:
                                        ef = _resolve_indirect(ef_obj)
                                        stream = ef.get("/UF") or ef.get("/F")
                                        if stream is not None:
                                            raw = bytes(
                                                _resolve_indirect(stream).read_bytes()
                                            )
                                            if raw[:5] == b"%PDF-":
                                                new_data = (
                                                    _try_convert_embedded_pdf_to_pdfa2(
                                                        raw
                                                    )
                                                )
                                                if new_data is not None:
                                                    _update_embedded_stream(
                                                        ef, new_data
                                                    )
                                                    converted += 1
                                                    logger.debug(
                                                        "Converted non-compliant "
                                                        "FileAttachment: page %d",
                                                        page_num,
                                                    )
                                                    _converted = True
                                except Exception:
                                    pass
                            if not _converted:
                                if fs is not None:
                                    try:
                                        if "/EF" in fs_resolved:
                                            del fs_resolved["/EF"]
                                    except Exception:
                                        pass
                                indices_to_remove.append(i)
                except Exception:
                    continue

            # Remove from back to front
            for i in reversed(indices_to_remove):
                del annots[i]
                removed += 1
                logger.debug("Removed non-compliant FileAttachment: page %d", page_num)

            if len(annots) == 0:
                del page["/Annots"]
        except Exception as e:
            logger.debug("Error processing annotations on page %d: %s", page_num, e)

    # 3. Scan ALL remaining FileSpecs for orphans not in Name Tree or annotations
    removed_objgens: set[tuple[int, int]] = set()
    for obj in _iter_all_filespecs_by_scan(pdf):
        try:
            resolved = _resolve_indirect(obj)
            ef = resolved.get("/EF")
            if ef is None:
                continue
            if _is_pdfa_compliant_embedded(resolved):
                kept += 1
                continue
            # Not compliant — try to convert it to PDF/A-2b first
            _converted = False
            try:
                ef_obj = resolved.get("/EF")
                if ef_obj is not None:
                    ef_r = _resolve_indirect(ef_obj)
                    stream = ef_r.get("/UF") or ef_r.get("/F")
                    if stream is not None:
                        raw = bytes(_resolve_indirect(stream).read_bytes())
                        if raw[:5] == b"%PDF-":
                            new_data = _try_convert_embedded_pdf_to_pdfa2(raw)
                            if new_data is not None:
                                _update_embedded_stream(ef_r, new_data)
                                converted += 1
                                logger.debug(
                                    "Converted orphan non-compliant embedded FileSpec"
                                )
                                _converted = True
            except Exception:
                pass
            if not _converted:
                del resolved["/EF"]
                removed += 1
                try:
                    og = resolved.objgen
                except (AttributeError, ValueError, TypeError):
                    og = (0, 0)
                if og != (0, 0):
                    removed_objgens.add(og)
                logger.debug("Stripped /EF from orphan non-compliant FileSpec")
        except Exception:
            continue

    # Clean up /AF arrays referencing stripped FileSpecs
    _cleanup_af_arrays(pdf, removed_objgens)

    # Clean up /Root/AF when all embedded files have been removed.
    # When some files are kept or converted, ensure_af_relationships() will
    # rebuild /Root/AF.
    if removed > 0 and kept == 0 and converted == 0:
        if "/AF" in pdf.Root:
            del pdf.Root["/AF"]
            logger.debug("Removed /Root/AF (all embedded files were non-compliant)")

    if removed > 0 or converted > 0:
        logger.info(
            "%d non-compliant embedded file(s): %d converted, %d removed, %d kept",
            removed + converted,
            converted,
            removed,
            kept,
        )
    elif kept > 0:
        logger.info("%d compliant embedded file(s) kept", kept)

    return {"removed": removed, "kept": kept, "converted": converted}


def ensure_af_relationships(pdf: Pdf) -> int:
    """Ensures AFRelationship keys on all FileSpec dicts and builds /Root/AF.

    PDF/A-2 (ISO 19005-2) and PDF/A-3 (ISO 19005-3) require:
    1. Every FileSpec dictionary to have an /AFRelationship key.
    2. /Root/AF array referencing all FileSpec dictionaries.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of FileSpec dictionaries that were fixed.
    """
    fixed_count = 0
    # Collect all FileSpec references, deduplicated by objgen.
    # Direct objects (objgen == (0,0)) cannot be deduplicated this way,
    # so we track them separately by list identity.
    seen_objgen: set[tuple[int, int]] = set()
    all_filespecs: list[object] = []

    def _collect_filespec(obj: object) -> None:
        """Add a FileSpec to the collection if not already seen."""
        nonlocal fixed_count
        try:
            resolved = _resolve_indirect(obj)

            # Deduplicate indirect objects via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return
                seen_objgen.add(og)

            # Check/fix AFRelationship
            af_rel = resolved.get("/AFRelationship")
            if af_rel is None:
                resolved["/AFRelationship"] = Name.Unspecified
                fixed_count += 1
                logger.debug("Added /AFRelationship=Unspecified to FileSpec")
            elif str(af_rel) not in VALID_AF_RELATIONSHIPS:
                resolved["/AFRelationship"] = Name.Unspecified
                fixed_count += 1
                logger.debug(
                    "Replaced invalid /AFRelationship=%s with /Unspecified",
                    str(af_rel),
                )

            all_filespecs.append(obj)
        except Exception as e:
            logger.debug("Error processing FileSpec: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _collect_filespec(filespec)

    # Build /Root/AF array or clean up stale one
    if all_filespecs:
        pdf.Root["/AF"] = Array(all_filespecs)
        logger.debug("Set /Root/AF with %d FileSpec(s)", len(all_filespecs))
    elif "/AF" in pdf.Root:
        del pdf.Root["/AF"]
        logger.debug("Removed stale /Root/AF (no filespecs remain)")

    if fixed_count > 0:
        logger.info("%d AFRelationship(s) fixed", fixed_count)
    return fixed_count


def ensure_embedded_file_subtypes(pdf: Pdf) -> int:
    """Ensures /Subtype (MIME type) on all embedded file streams.

    ISO 19005-3 clause 6.8 requires each embedded file stream to contain
    a /Subtype key identifying the MIME media type.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of embedded file streams that were fixed.
    """
    fixed_count = 0
    seen_objgen: set[tuple[int, int]] = set()

    def _guess_mime(filespec: object) -> str:
        """Guess MIME type from /UF or /F filename in a FileSpec."""
        for key in ("/UF", "/F"):
            fname = filespec.get(key)
            if fname is not None:
                fname_str = str(fname)
                mime, _ = mimetypes.guess_type(fname_str)
                if mime:
                    return mime
        return "application/octet-stream"

    def _is_valid_mime_subtype(subtype_name: object) -> bool:
        """Check if a /Subtype Name value is a valid MIME type (type/subtype)."""
        try:
            val = str(subtype_name)
            # Strip leading "/" from PDF Name
            if val.startswith("/"):
                val = val[1:]
            # Valid MIME: exactly one "/" separating non-empty type and subtype
            parts = val.split("/")
            return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0
        except Exception:
            return False

    def _fix_stream(stream_obj: object, mime: str) -> bool:
        """Add or fix /Subtype on a stream if missing or invalid.

        Returns True if fixed.
        """
        try:
            resolved = _resolve_indirect(stream_obj)

            # Deduplicate via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return False
                seen_objgen.add(og)

            existing = resolved.get("/Subtype")
            if existing is not None and _is_valid_mime_subtype(existing):
                return False

            resolved["/Subtype"] = Name("/" + mime)
            if existing is not None:
                logger.debug(
                    "Replaced invalid /Subtype=%s with %s on embedded file stream",
                    str(existing),
                    mime,
                )
            else:
                logger.debug("Set /Subtype=%s on embedded file stream", mime)
            return True
        except Exception as e:
            logger.debug("Error fixing stream /Subtype: %s", e)
            return False

    def _process_filespec(obj: object) -> None:
        """Check /EF entries in a FileSpec and fix missing /Subtype."""
        nonlocal fixed_count
        try:
            resolved = _resolve_indirect(obj)

            ef = resolved.get("/EF")
            if ef is None:
                return

            ef = _resolve_indirect(ef)

            mime = _guess_mime(resolved)

            for key in ("/F", "/UF"):
                stream = ef.get(key)
                if stream is not None:
                    if _fix_stream(stream, mime):
                        fixed_count += 1
        except Exception as e:
            logger.debug("Error processing FileSpec for /Subtype: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _process_filespec(filespec)

    if fixed_count > 0:
        logger.info("%d embedded file stream /Subtype(s) fixed", fixed_count)
    return fixed_count


def ensure_embedded_file_params(pdf: Pdf) -> int:
    """Ensures /Params with /ModDate on all embedded file streams.

    ISO 19005-2/3 requires embedded file streams to carry a /Params
    dictionary with at least a /ModDate entry.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of embedded file streams that were fixed.
    """
    fixed_count = 0
    seen_objgen: set[tuple[int, int]] = set()
    now = datetime.now(UTC)
    mod_date_str = _format_pdf_date(now)

    def _fix_stream(stream_obj: object) -> bool:
        """Add /Params with /ModDate to a stream if missing. Returns True if fixed."""
        try:
            resolved = _resolve_indirect(stream_obj)

            # Deduplicate via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return False
                seen_objgen.add(og)

            params = resolved.get("/Params")
            if params is not None:
                params = _resolve_indirect(params)
                if "/ModDate" in params:
                    # Validate existing /ModDate format
                    existing_date = str(params["/ModDate"])
                    if not _is_valid_pdf_date(existing_date):
                        logger.debug(
                            "Reformatting invalid /ModDate '%s' on "
                            "embedded file stream",
                            existing_date,
                        )
                        params["/ModDate"] = pikepdf.String(mod_date_str)
                        return True
                    return False
                # /Params exists but lacks /ModDate — add it
                params["/ModDate"] = pikepdf.String(mod_date_str)
                logger.debug(
                    "Added /ModDate to existing /Params on embedded file stream"
                )
                return True

            # No /Params at all — create it
            resolved["/Params"] = Dictionary(ModDate=pikepdf.String(mod_date_str))
            logger.debug("Added /Params with /ModDate on embedded file stream")
            return True
        except Exception as e:
            logger.debug("Error fixing stream /Params: %s", e)
            return False

    def _process_filespec(obj: object) -> None:
        """Check /EF entries in a FileSpec and fix missing /Params."""
        nonlocal fixed_count
        try:
            resolved = _resolve_indirect(obj)

            ef = resolved.get("/EF")
            if ef is None:
                return

            ef = _resolve_indirect(ef)

            for key in ("/F", "/UF"):
                stream = ef.get(key)
                if stream is not None:
                    if _fix_stream(stream):
                        fixed_count += 1
        except Exception as e:
            logger.debug("Error processing FileSpec for /Params: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _process_filespec(filespec)

    if fixed_count > 0:
        logger.info("%d embedded file stream /Params fixed", fixed_count)
    return fixed_count


def ensure_filespec_uf_entries(pdf: Pdf) -> int:
    """Ensures /UF (Unicode filename) on all FileSpec dictionaries.

    ISO 19005-3 (PDF/A-3) requires every File Specification Dictionary
    to have a /UF entry.  When only /F is present, this copies its value
    to /UF.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of FileSpec dictionaries that were fixed.
    """
    fixed_count = 0
    seen_objgen: set[tuple[int, int]] = set()

    def _fix_filespec(obj: object) -> None:
        """Ensure both /F and /UF exist; mirror in /EF if needed."""
        nonlocal fixed_count
        try:
            resolved = _resolve_indirect(obj)

            # Deduplicate indirect objects via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return
                seen_objgen.add(og)

            f_val = resolved.get("/F")
            uf_val = resolved.get("/UF")

            # Ensure both /F and /UF exist (ISO requires both)
            if uf_val is None and f_val is not None:
                resolved["/UF"] = f_val
                fixed_count += 1
                logger.debug("Added /UF from /F on FileSpec")
            elif f_val is None and uf_val is not None:
                resolved["/F"] = uf_val
                fixed_count += 1
                logger.debug("Added /F from /UF on FileSpec")
            elif f_val is None and uf_val is None:
                # Neither exists — derive from /EF key names or use fallback
                resolved["/F"] = "embedded_file"
                resolved["/UF"] = "embedded_file"
                fixed_count += 1
                logger.debug("Added /F and /UF fallback on FileSpec")

            # Mirror in /EF dictionary: ensure both /EF/F and /EF/UF exist
            ef = resolved.get("/EF")
            if ef is not None:
                ef = _resolve_indirect(ef)
                if ef is not None:
                    if "/F" in ef and "/UF" not in ef:
                        ef["/UF"] = ef["/F"]
                        logger.debug("Added /UF from /F in /EF dictionary")
                    elif "/UF" in ef and "/F" not in ef:
                        ef["/F"] = ef["/UF"]
                        logger.debug("Added /F from /UF in /EF dictionary")
        except Exception as e:
            logger.debug("Error fixing FileSpec /UF: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _fix_filespec(filespec)

    if fixed_count > 0:
        logger.info("%d FileSpec /UF entr(ies) fixed", fixed_count)
    return fixed_count


def ensure_filespec_desc(pdf: Pdf) -> int:
    """Ensures /Desc on all FileSpec dictionaries.

    PDF/A-3 (ISO 19005-3) recommends each File Specification Dictionary
    to have a /Desc entry describing the embedded file.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of FileSpec dictionaries that were fixed.
    """
    fixed_count = 0
    seen_objgen: set[tuple[int, int]] = set()

    def _fix_filespec(obj: object) -> None:
        nonlocal fixed_count
        try:
            resolved = _resolve_indirect(obj)

            # Deduplicate indirect objects via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return
                seen_objgen.add(og)

            # Skip if /Desc already present
            if "/Desc" in resolved:
                return

            # Derive description from filename
            filename = None
            for key in ("/UF", "/F"):
                val = resolved.get(key)
                if val is not None:
                    filename = str(val)
                    break

            if filename:
                desc = f"Embedded file: {filename}"
            else:
                desc = "Embedded file"

            resolved["/Desc"] = pikepdf.String(desc)
            fixed_count += 1
            logger.debug("Added /Desc to FileSpec: %s", desc)
        except Exception as e:
            logger.debug("Error fixing FileSpec /Desc: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _fix_filespec(filespec)

    if fixed_count > 0:
        logger.info("%d FileSpec /Desc entr(ies) fixed", fixed_count)
    return fixed_count


def sanitize_embedded_file_filters(pdf: Pdf) -> dict[str, int]:
    """Sanitize forbidden filters on embedded file streams.

    ISO 19005-2, §6.1.4 forbids LZWDecode and Crypt filters on ALL
    streams, including embedded file streams inside /EF dictionaries.
    The main filters.py handles page/content streams but may not
    traverse embedded file streams specifically.

    LZWDecode streams are re-encoded with FlateDecode.  Crypt filters
    are removed (pikepdf transparently decrypts on read).

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with 'lzw_converted' and 'crypt_removed' counts.
    """
    lzw_converted = 0
    crypt_removed = 0
    seen_objgen: set[tuple[int, int]] = set()

    def _fix_stream(stream_obj: object) -> None:
        nonlocal lzw_converted, crypt_removed
        try:
            resolved = _resolve_indirect(stream_obj)
            if not isinstance(resolved, Stream):
                return

            # Deduplicate via objgen
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                if og in seen_objgen:
                    return
                seen_objgen.add(og)

            filter_obj = resolved.get("/Filter")
            if filter_obj is None:
                return

            filter_obj = _resolve_indirect(filter_obj)

            # Collect filter names (handle single Name and Array)
            filter_names: list[str] = []
            if isinstance(filter_obj, Name):
                filter_names = [str(filter_obj)]
            elif isinstance(filter_obj, Array):
                for f in filter_obj:
                    f = _resolve_indirect(f)
                    if isinstance(f, Name):
                        filter_names.append(str(f))

            has_lzw = any(n in ("/LZWDecode", "/LZW") for n in filter_names)
            has_crypt = any(n in ("/Crypt", "/crypt") for n in filter_names)

            if not has_lzw and not has_crypt:
                return

            # Re-encode: read decompressed, write back (pikepdf
            # re-compresses with FlateDecode on save)
            data = resolved.read_bytes()
            resolved.write(data)
            if resolved.get("/DecodeParms") is not None:
                del resolved["/DecodeParms"]

            if has_lzw:
                lzw_converted += 1
                logger.debug("Converted LZW filter on embedded file stream: %s", og)
            if has_crypt:
                crypt_removed += 1
                logger.debug("Removed Crypt filter from embedded file stream: %s", og)

        except Exception as e:
            logger.debug("Error fixing embedded file stream filter: %s", e)

    def _process_filespec(obj: object) -> None:
        try:
            resolved = _resolve_indirect(obj)
            ef = resolved.get("/EF")
            if ef is None:
                return
            ef = _resolve_indirect(ef)
            for key in ("/F", "/UF"):
                stream = ef.get(key)
                if stream is not None:
                    _fix_stream(stream)
        except Exception as e:
            logger.debug("Error processing FileSpec for filter sanitization: %s", e)

    for filespec in _iter_all_filespecs(pdf):
        _process_filespec(filespec)

    if lzw_converted > 0:
        logger.info(
            "%d embedded file stream(s) converted from LZW to FlateDecode",
            lzw_converted,
        )
    if crypt_removed > 0:
        logger.info(
            "%d Crypt filter(s) removed from embedded file stream(s)",
            crypt_removed,
        )

    return {"lzw_converted": lzw_converted, "crypt_removed": crypt_removed}


def remove_embedded_files(pdf: Pdf) -> int:
    """Removes embedded files from the PDF.

    Removes:
    - EmbeddedFiles from the Names dictionary
    - FileAttachment Annotations

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of files/attachments removed.
    """
    removed_count = 0

    # 1. Remove EmbeddedFiles from Names (count across full Name Tree)
    try:
        if "/Names" in pdf.Root:
            names = _resolve_indirect(pdf.Root.Names)
            if "/EmbeddedFiles" in names:
                embedded = _resolve_indirect(names.EmbeddedFiles)
                for filespec in _iter_name_tree_values(embedded):
                    removed_count += 1
                    # Strip /EF so full scan in step 3 won't re-count
                    try:
                        fs_resolved = _resolve_indirect(filespec)
                        if "/EF" in fs_resolved:
                            del fs_resolved["/EF"]
                    except Exception:
                        pass

                del names["/EmbeddedFiles"]
                logger.debug("EmbeddedFiles removed from Names dictionary")
    except Exception as e:
        logger.debug("Error removing EmbeddedFiles: %s", e)

    # 2. Remove FileAttachment Annotations
    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue

            # Collect indices of annotations to remove
            indices_to_remove = []
            for i, annot in enumerate(annots):
                try:
                    resolved = _resolve_indirect(annot)
                    subtype = resolved.get("/Subtype")
                    if subtype is not None and str(subtype) == "/FileAttachment":
                        # Strip /EF so full scan in step 3 won't re-count
                        fs = resolved.get("/FS")
                        if fs is not None:
                            try:
                                fs_resolved = _resolve_indirect(fs)
                                if "/EF" in fs_resolved:
                                    del fs_resolved["/EF"]
                            except Exception:
                                pass
                        indices_to_remove.append(i)
                except Exception:
                    continue

            # Remove from back to front to avoid index shifting
            for i in reversed(indices_to_remove):
                del annots[i]
                removed_count += 1
                logger.debug("FileAttachment Annotation removed: page %d", page_num)

            # Remove empty Annots array
            if len(annots) == 0:
                del page["/Annots"]
        except Exception as e:
            logger.debug(
                "Error processing Annotations on page %d: %s",
                page_num,
                e,
            )

    # 3. Scan ALL remaining FileSpecs and strip /EF from any orphans
    removed_objgens: set[tuple[int, int]] = set()
    for obj in _iter_all_filespecs_by_scan(pdf):
        try:
            resolved = _resolve_indirect(obj)
            ef = resolved.get("/EF")
            if ef is None:
                continue
            del resolved["/EF"]
            removed_count += 1
            try:
                og = resolved.objgen
            except (AttributeError, ValueError, TypeError):
                og = (0, 0)
            if og != (0, 0):
                removed_objgens.add(og)
            logger.debug("Stripped /EF from orphan FileSpec")
        except Exception:
            continue

    # Clean up /AF arrays referencing stripped FileSpecs
    _cleanup_af_arrays(pdf, removed_objgens)

    # Clean up /Root/AF since all embedded files have been removed
    if removed_count > 0 and "/AF" in pdf.Root:
        del pdf.Root["/AF"]
        logger.debug("Removed /Root/AF after embedded file removal")

    if removed_count > 0:
        logger.info("%d embedded file(s) removed", removed_count)
    return removed_count
