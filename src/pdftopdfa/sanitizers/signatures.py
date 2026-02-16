# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Digital signature sanitization for PDF/A compliance.

PDF/A conversion modifies document bytes, so existing digital signatures
become cryptographically invalid. This sanitizer removes live signature
references and neutralizes signature dictionaries to avoid post-conversion
validation failures (notably rule 6.4.3).
"""

import logging

from pikepdf import Dictionary, Name, Pdf

from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

_VALID_SUBFILTERS = frozenset(
    {
        "/adbe.pkcs7.detached",
        "/adbe.pkcs7.sha1",
        "/ETSI.CAdES.detached",
        "/ETSI.RFC3161",
    }
)


_SIGNATURE_KEYS_TO_REMOVE = (
    "/Type",
    "/Filter",
    "/SubFilter",
    "/ByteRange",
    "/Contents",
    "/Reference",
    "/Cert",
    "/Prop_Build",
    "/M",
    "/Name",
    "/Reason",
    "/Location",
    "/ContactInfo",
    "/R",
)


def _obj_key(obj) -> tuple[int, int] | None:
    """Get a stable deduplication key for a pikepdf object."""
    try:
        objgen = obj.objgen
    except Exception:
        return None
    if objgen != (0, 0):
        return objgen
    return None


def _is_signature_dictionary(obj) -> bool:
    """Check whether an object is a signature dictionary."""
    if not isinstance(obj, Dictionary):
        return False

    try:
        sig_type = obj.get("/Type")
        if sig_type is not None and Name(sig_type) == Name.Sig:
            return True
    except Exception:
        pass

    # /ByteRange is signature-specific in this context; require additional
    # signature markers to avoid accidental matches.
    try:
        has_byte_range = obj.get("/ByteRange") is not None
        has_filter = obj.get("/Filter") is not None
        has_sub_filter = obj.get("/SubFilter") is not None
        has_contents = obj.get("/Contents") is not None
        if has_byte_range and (has_filter or has_sub_filter or has_contents):
            return True
        if has_contents and has_sub_filter:
            return True
    except Exception:
        return False

    return False


def _log_signature_info(sig_dict: Dictionary, file_size: int | None = None) -> None:
    """Log information about a signature being removed.

    If the signature has a valid SubFilter and ByteRange, logs an info
    message noting that a valid signature was removed for compliance.
    Also validates the ByteRange format and logs if it seems malformed.

    Args:
        sig_dict: Resolved signature dictionary.
        file_size: Approximate file size for ByteRange validation (optional).
    """
    try:
        sub_filter = sig_dict.get("/SubFilter")
        sub_filter_str = str(sub_filter) if sub_filter is not None else None
        has_valid_subfilter = sub_filter_str in _VALID_SUBFILTERS

        # Validate ByteRange
        byte_range = sig_dict.get("/ByteRange")
        has_valid_byte_range = False
        if byte_range is not None:
            try:
                br_values = [int(v) for v in byte_range]
                if len(br_values) == 4 and br_values[0] == 0:
                    has_valid_byte_range = True
                    total = br_values[0] + br_values[1] + br_values[2] + br_values[3]
                    if file_size is not None and abs(total - file_size) > 1024:
                        logger.debug(
                            "Signature ByteRange total %d differs from "
                            "file size %d by more than 1024 bytes",
                            total,
                            file_size,
                        )
                else:
                    logger.debug(
                        "Malformed ByteRange: expected 4 integers with "
                        "first element 0, got %s",
                        br_values,
                    )
            except (TypeError, ValueError):
                logger.debug("Malformed ByteRange: could not parse as integers")

        if has_valid_subfilter and has_valid_byte_range:
            logger.info(
                "Valid digital signature (SubFilter=%s) removed for PDF/A compliance",
                sub_filter_str,
            )
        elif has_valid_subfilter:
            logger.info(
                "Digital signature with SubFilter=%s removed for "
                "PDF/A compliance (ByteRange absent or malformed)",
                sub_filter_str,
            )
    except Exception:
        pass


def _neutralize_signature_dictionary(sig_dict: Dictionary) -> bool:
    """Remove signature-specific keys from a signature dictionary."""
    changed = False
    for key in _SIGNATURE_KEYS_TO_REMOVE:
        try:
            if key in sig_dict:
                del sig_dict[key]
                changed = True
        except Exception:
            continue
    return changed


def _collect_signature_fields(fields, visited=None):
    """Collect signed signature fields from AcroForm /Fields recursively.

    Traverses the field tree via /Kids, collecting fields with /FT /Sig
    that have a /V (signature value) entry.

    Args:
        fields: pikepdf Array of field objects.
        visited: Set of visited object IDs for cycle detection.

    Returns:
        List of (field_dict, sig_dict) tuples.
    """
    if visited is None:
        visited = set()

    result = []

    for field_ref in fields:
        try:
            field = _resolve(field_ref)
        except Exception:
            continue

        field_key = _obj_key(field)
        if field_key is not None:
            if field_key in visited:
                continue
            visited.add(field_key)

        try:
            ft = field.get("/FT")
            if ft is not None and Name(ft) == Name.Sig:
                v = field.get("/V")
                if v is not None:
                    try:
                        sig_dict = _resolve(v)
                        result.append((field, sig_dict))
                    except Exception:
                        pass
        except Exception:
            pass

        # Recurse into /Kids
        try:
            kids = field.get("/Kids")
            if kids is not None:
                result.extend(_collect_signature_fields(kids, visited))
        except Exception:
            pass

    return result


def sanitize_signatures(pdf: Pdf, level: str = "3b") -> dict:
    """Sanitize digital signature structures for PDF/A compliance.

    Removes signature values from signature fields (/FT /Sig with /V),
    removes /Perms signature references, and neutralizes all signature
    dictionaries found in the document to avoid digest-related failures.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level ('2b', '2u', '3b', or '3u').

    Returns:
        Dictionary with statistics:
        - signatures_found: Total signature dictionaries found
        - signatures_removed: Signature structures removed/neutralized
        - sigflags_fixed: Whether SigFlags was modified
        - signatures_type_fixed: Kept for backward compatibility (always 0)
    """
    del level  # Signature removal is currently level-independent.

    stats = {
        "signatures_found": 0,
        "signatures_removed": 0,
        "sigflags_fixed": 0,
        "signatures_type_fixed": 0,
    }

    visited_fields: set[tuple[int, int]] = set()
    visited_sig_dicts: set[tuple[int, int]] = set()
    sig_fields: list[tuple[Dictionary, Dictionary]] = []
    sig_dicts: list[Dictionary] = []

    def _collect_sig_dict(candidate) -> None:
        try:
            resolved = _resolve(candidate)
        except Exception:
            return
        if not _is_signature_dictionary(resolved):
            return
        key = _obj_key(resolved)
        if key is not None:
            if key in visited_sig_dicts:
                return
            visited_sig_dicts.add(key)
        sig_dicts.append(resolved)

    # From AcroForm /Fields
    try:
        acroform = _resolve(pdf.Root.AcroForm) if "/AcroForm" in pdf.Root else None
    except Exception:
        acroform = None

    if acroform is not None:
        try:
            fields = acroform.get("/Fields")
            if fields is not None:
                sig_fields.extend(_collect_signature_fields(fields, visited_fields))
        except Exception:
            pass

    # From page annotations (may find fields not in AcroForm /Fields).
    for page in pdf.pages:
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue
            try:
                annots = annots.get_object()
            except (AttributeError, TypeError, ValueError):
                pass

            for annot_ref in annots:
                try:
                    annot = _resolve(annot_ref)
                except Exception:
                    continue

                annot_key = _obj_key(annot)
                if annot_key is not None:
                    if annot_key in visited_fields:
                        continue
                    visited_fields.add(annot_key)

                try:
                    ft = annot.get("/FT")
                    if ft is not None and Name(ft) == Name.Sig:
                        v = annot.get("/V")
                        if v is not None:
                            try:
                                sig_dict = _resolve(v)
                                sig_fields.append((annot, sig_dict))
                                _collect_sig_dict(sig_dict)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

    # Collect signature dicts from field /V values.
    for field, sig_dict in sig_fields:
        _collect_sig_dict(sig_dict)
        try:
            if "/V" in field:
                del field["/V"]
                stats["signatures_removed"] += 1
        except Exception as e:
            logger.debug("Failed to remove signature /V: %s", e)

    # Remove signature references from Catalog /Perms.
    perms_refs_removed = 0
    try:
        perms = pdf.Root.get("/Perms")
        if perms is not None:
            perms = _resolve(perms)
            for key in ("/DocMDP", "/UR", "/UR3"):
                try:
                    sig_ref = perms.get(key)
                    if sig_ref is not None:
                        _collect_sig_dict(sig_ref)
                        del perms[key]
                        perms_refs_removed += 1
                except Exception:
                    continue
            try:
                if len(perms) == 0:
                    del pdf.Root["/Perms"]
            except Exception:
                pass
    except Exception:
        pass

    # Global scan as fallback (catches orphaned signature dictionaries that
    # may still be validated by tools even when unreferenced).
    for obj in pdf.objects:
        try:
            resolved = _resolve(obj)
        except Exception:
            continue
        _collect_sig_dict(resolved)

    stats["signatures_found"] = len(sig_dicts)

    if stats["signatures_found"] > 0:
        logger.warning(
            "Found %d digital signature dictionary/dictionaries; signatures are "
            "removed for PDF/A conversion",
            stats["signatures_found"],
        )

    for sig_dict in sig_dicts:
        _log_signature_info(sig_dict)
        if _neutralize_signature_dictionary(sig_dict):
            stats["signatures_removed"] += 1

    stats["signatures_removed"] += perms_refs_removed

    # Clear SigFlags bit 1 (SignaturesExist) after removing signatures.
    if acroform is not None:
        try:
            current_flags = acroform.get("/SigFlags")
            if current_flags is not None:
                current_val = int(current_flags)
            else:
                current_val = 0
            new_val = current_val & ~1
            if new_val != current_val:
                if new_val == 0:
                    del acroform["/SigFlags"]
                else:
                    acroform["/SigFlags"] = new_val
                stats["sigflags_fixed"] = 1
        except Exception as e:
            logger.debug("Failed to fix /SigFlags: %s", e)

    return stats
