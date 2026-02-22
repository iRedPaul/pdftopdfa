# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PDF/A sanitization functions.

This module provides functions to remove or modify PDF elements
that are not allowed in PDF/A.
"""

import logging
from typing import Any

from pikepdf import Pdf

from ..exceptions import ConversionError
from ..utils import get_required_pdf_version, validate_pdfa_level
from .actions import remove_actions, validate_destinations
from .annotations import (
    ensure_appearance_streams,
    fix_annotation_flags,
    fix_annotation_opacity,
    fix_button_appearance_subdicts,
    remove_annotation_colors,
    remove_forbidden_annotations,
    remove_needs_appearances,
    remove_non_normal_appearance_keys,
)
from .base import (
    ANNOT_FLAG_HIDDEN,
    ANNOT_FLAG_INVISIBLE,
    ANNOT_FLAG_NOROTATE,
    ANNOT_FLAG_NOVIEW,
    ANNOT_FLAG_NOZOOM,
    ANNOT_FLAG_PRINT,
    COMPLIANT_ACTIONS,
    FORBIDDEN_ANNOTATION_SUBTYPES,
    FORBIDDEN_XOBJECT_SUBTYPES,
    SUBMITFORM_FLAG_EXPORTFORMAT,
    SUBMITFORM_FLAG_SUBMITPDF,
    SUBMITFORM_FLAG_XFDF,
    _is_javascript_action,
    _is_non_compliant_action,
)
from .catalog import (
    ensure_catalog_lang,
    ensure_mark_info,
    remove_catalog_version,
    remove_forbidden_catalog_entries,
    remove_forbidden_name_dictionary_entries,
    remove_forbidden_page_entries,
    remove_forbidden_viewer_preferences,
)
from .colorspaces import sanitize_colorspaces
from .extgstate import sanitize_extgstate
from .files import (
    ensure_af_relationships,
    ensure_embedded_file_params,
    ensure_embedded_file_subtypes,
    ensure_filespec_desc,
    ensure_filespec_uf_entries,
    remove_embedded_files,
    remove_non_compliant_embedded_files,
    sanitize_embedded_file_filters,
)
from .filters import (
    convert_lzw_streams,
    fix_stream_lengths,
    remove_crypt_streams,
    remove_external_stream_keys,
    sanitize_nonstandard_inline_filters,
)
from .font_notdef import sanitize_font_notdef
from .font_structure import sanitize_font_structure
from .font_widths import sanitize_font_widths
from .fonts import sanitize_cidfont_structures, sanitize_fontname_consistency
from .glyph_coverage import sanitize_glyph_coverage
from .javascript import remove_javascript
from .jbig2 import convert_jbig2_external_globals
from .jpx import sanitize_jpx_color_boxes
from .notdef_usage import sanitize_notdef_usage
from .optional_content import sanitize_optional_content
from .page_boxes import sanitize_page_boxes
from .pua_actualtext import sanitize_pua_actualtext
from .rendering_intent import sanitize_rendering_intent
from .signatures import sanitize_signatures
from .structure_limits import sanitize_structure_limits
from .tounicode_values import fill_tounicode_gaps, sanitize_tounicode_values
from .truetype_encoding import sanitize_truetype_encoding
from .xfa import remove_xfa_forms
from .xobjects import (
    fix_bits_per_component,
    fix_image_interpolate,
    remove_forbidden_xobjects,
)

logger = logging.getLogger(__name__)


def sanitize_for_pdfa(pdf: Pdf, level: str = "3b") -> dict[str, Any]:
    """Sanitizes a PDF for PDF/A conformance.

    Performs all necessary sanitization based on the
    conformance level:

    - PDF/A-2b: Remove JavaScript, actions, XFA forms (transparency allowed)
    - PDF/A-3b: Remove JavaScript, actions, XFA forms (embedded files
                and transparency allowed)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level ('2b', '2u', '3b', or '3u').

    Returns:
        Dictionary with statistics about performed sanitizations:
        - javascript_removed: Number of JavaScript elements removed
        - files_removed: Number of embedded files removed
        - actions_removed: Number of non-compliant actions removed

    Raises:
        ConversionError: If an invalid level is specified.
    """
    level = validate_pdfa_level(level)

    if len(pdf.objects) > 8_388_607:
        raise ConversionError(
            "PDF exceeds the maximum number of indirect objects allowed by "
            "ISO 19005-2 rule 6.1.13-7 (limit: 8,388,607)"
        )

    logger.info("Sanitizing PDF for PDF/A-%s conformance", level)

    result: dict[str, Any] = {
        "javascript_removed": 0,
        "files_removed": 0,
        "embedded_files_kept": 0,
        "embedded_files_converted": 0,
        "actions_removed": 0,
        "invalid_destinations_removed": 0,
        "xfa_removed": 0,
        "forbidden_annotations_removed": 0,
        "forbidden_xobjects_removed": 0,
        "appearance_streams_added": 0,
        "annotation_flags_fixed": 0,
        "annotation_opacity_fixed": 0,
        "lzw_streams_converted": 0,
        "crypt_streams_removed": 0,
        "external_stream_keys_removed": 0,
        "stream_lengths_fixed": 0,
        "nonstandard_inline_filters_fixed": 0,
        "jbig2_converted": 0,
        "jbig2_reencoded": 0,
        "jbig2_failed": 0,
        "jpx_fixed": 0,
        "jpx_wrapped": 0,
        "jpx_reencoded": 0,
        "jpx_already_valid": 0,
        "jpx_failed": 0,
        "oc_as_entries_removed": 0,
        "oc_intents_fixed": 0,
        "oc_d_created": False,
        "oc_d_name_added": False,
        "oc_list_mode_fixed": 0,
        "oc_base_state_fixed": 0,
        "oc_config_names_added": 0,
        "oc_missing_ocgs_added": 0,
        "oc_rbgroups_fixed": 0,
        "oc_order_ocgs_added": 0,
        "extgstate_fixed": 0,
        "ri_operators_fixed": 0,
        "undefined_operators_removed": 0,
        "resources_dictionaries_added": 0,
        "resources_entries_merged": 0,
        "af_relationships_fixed": 0,
        "embedded_file_subtypes_fixed": 0,
        "embedded_file_params_fixed": 0,
        "filespec_uf_fixed": 0,
        "filespec_desc_fixed": 0,
        "embedded_file_lzw_converted": 0,
        "embedded_file_crypt_removed": 0,
        "forbidden_catalog_entries_removed": 0,
        "forbidden_name_dict_entries_removed": 0,
        "forbidden_page_entries_removed": 0,
        "viewer_prefs_entries_removed": 0,
        "catalog_lang_set": False,
        "mark_info_added": False,
        "image_interpolate_fixed": 0,
        "invalid_bpc_fixed": 0,
        "mask_bpc_fixed": 0,
        "cidsysteminfo_fixed": 0,
        "cidtogidmap_fixed": 0,
        "cidset_removed": 0,
        "type1_charset_removed": 0,
        "simple_font_widths_fixed": 0,
        "cidfont_widths_fixed": 0,
        "notdef_fixed": 0,
        "glyphs_added": 0,
        "tounicode_values_fixed": 0,
        "tounicode_gaps_filled": 0,
        "pua_actualtext_added": 0,
        "pua_actualtext_warnings": 0,
        "notdef_usage_fixed": 0,
        "signatures_found": 0,
        "signatures_removed": 0,
        "sigflags_fixed": 0,
        "signatures_type_fixed": 0,
        "annotation_colors_removed": 0,
        "annotation_ap_keys_removed": 0,
        "btn_ap_subdicts_fixed": 0,
        "needs_appearances_removed": False,
        "mediabox_inherited": 0,
        "boxes_normalized": 0,
        "boxes_clipped": 0,
        "trimbox_added": 0,
        "malformed_boxes_removed": 0,
        "structure_strings_truncated": 0,
        "structure_names_shortened": 0,
        "structure_utf8_names_fixed": 0,
        "structure_integers_clamped": 0,
        "structure_reals_normalized": 0,
        "structure_q_nesting_rebalanced": 0,
        "structure_hex_odd_fixed": 0,
        "catalog_version_removed": False,
        "oc_ocg_names_added": 0,
        "image_intents_fixed": 0,
        "font_type_added": 0,
        "font_subtype_fixed": 0,
        "font_basefont_added": 0,
        "font_firstchar_added": 0,
        "font_lastchar_added": 0,
        "font_widths_size_fixed": 0,
        "font_stream_subtype_removed": 0,
        "tt_nonsymbolic_cmap_added": 0,
        "tt_nonsymbolic_encoding_fixed": 0,
        "tt_symbolic_encoding_removed": 0,
        "tt_symbolic_flag_set": 0,
        "tt_symbolic_cmap_added": 0,
    }

    # Convert LZW-compressed streams to FlateDecode (all levels)
    result["lzw_streams_converted"] = convert_lzw_streams(pdf)

    # Remove Crypt filters from streams (all levels)
    result["crypt_streams_removed"] = remove_crypt_streams(pdf)

    # Remove forbidden external stream keys /F, /FFilter, /FDecodeParms (6.1.7.1)
    result["external_stream_keys_removed"] = remove_external_stream_keys(pdf)

    # Force re-encoding of non-image streams to repair /Length mismatches (6.1.7.1)
    result["stream_lengths_fixed"] = fix_stream_lengths(pdf)

    # Re-encode inline images with non-Table-6 filters (6.1.10-1)
    result["nonstandard_inline_filters_fixed"] = sanitize_nonstandard_inline_filters(
        pdf
    )

    # Convert JBIG2 streams with external globals (all levels)
    jbig2_result = convert_jbig2_external_globals(pdf)
    result["jbig2_converted"] = jbig2_result["converted"]
    result["jbig2_reencoded"] = jbig2_result["reencoded"]
    result["jbig2_failed"] = jbig2_result["failed"]

    # Sanitize JPEG2000 colr boxes (ISO 19005-2, 6.1.4.3)
    jpx_result = sanitize_jpx_color_boxes(pdf)
    result["jpx_fixed"] = jpx_result["jpx_fixed"]
    result["jpx_wrapped"] = jpx_result["jpx_wrapped"]
    result["jpx_reencoded"] = jpx_result["jpx_reencoded"]
    result["jpx_already_valid"] = jpx_result["jpx_already_valid"]
    result["jpx_failed"] = jpx_result["jpx_failed"]

    # Sanitize page boxes (MediaBox/CropBox/TrimBox/BleedBox/ArtBox)
    pb_result = sanitize_page_boxes(pdf)
    result["mediabox_inherited"] = pb_result["mediabox_inherited"]
    result["boxes_normalized"] = pb_result["boxes_normalized"]
    result["boxes_clipped"] = pb_result["boxes_clipped"]
    result["trimbox_added"] = pb_result["trimbox_added"]
    result["malformed_boxes_removed"] = pb_result["malformed_boxes_removed"]

    # Remove JavaScript (all levels)
    result["javascript_removed"] = remove_javascript(pdf)

    # Remove non-compliant actions (all levels)
    result["actions_removed"] = remove_actions(pdf)

    # Remove invalid destinations (references to non-existent pages)
    result["invalid_destinations_removed"] = validate_destinations(pdf)

    # Remove XFA forms (all levels - XFA is forbidden in all PDF/A)
    result["xfa_removed"] = remove_xfa_forms(pdf)

    # Remove forbidden catalog entries (ISO 19005-2, 6.1.10–6.1.13)
    result["forbidden_catalog_entries_removed"] = remove_forbidden_catalog_entries(pdf)

    # Remove/overwrite catalog /Version if it exceeds the required version
    # (ISO 19005-2, clause 6.1.2 — effective version must not exceed 1.7)
    required_version = get_required_pdf_version(level)
    result["catalog_version_removed"] = remove_catalog_version(pdf, required_version)

    # Remove forbidden entries in /Names dictionary (ISO 19005-2, 6.1.11)
    result["forbidden_name_dict_entries_removed"] = (
        remove_forbidden_name_dictionary_entries(pdf)
    )

    # Remove forbidden page dictionary entries (Rule 6.10)
    result["forbidden_page_entries_removed"] = remove_forbidden_page_entries(pdf)

    # Remove forbidden ViewerPreferences entries (ISO 19005-2, 6.1.2)
    result["viewer_prefs_entries_removed"] = remove_forbidden_viewer_preferences(pdf)

    # Ensure /Lang key in catalog (ISO 19005-2, 6.7.3)
    result["catalog_lang_set"] = ensure_catalog_lang(pdf)

    # Ensure /MarkInfo dictionary in catalog (ISO 19005-2, §6.7.1)
    result["mark_info_added"] = ensure_mark_info(pdf)

    # Sanitize digital signatures for PDF/A compliance (all levels)
    sig_result = sanitize_signatures(pdf, level)
    result["signatures_found"] = sig_result["signatures_found"]
    result["signatures_removed"] = sig_result["signatures_removed"]
    result["sigflags_fixed"] = sig_result["sigflags_fixed"]
    result["signatures_type_fixed"] = sig_result["signatures_type_fixed"]

    # Remove forbidden annotation subtypes (all levels)
    result["forbidden_annotations_removed"] = remove_forbidden_annotations(pdf, level)

    # Remove forbidden XObjects (all levels)
    result["forbidden_xobjects_removed"] = remove_forbidden_xobjects(pdf)

    # Fix /Interpolate on Image XObjects (ISO 19005-2, 6.2.9)
    result["image_interpolate_fixed"] = fix_image_interpolate(pdf)

    # Fix BitsPerComponent on Image XObjects (ISO 19005-2, 6.2.8)
    bpc_result = fix_bits_per_component(pdf)
    result["invalid_bpc_fixed"] = bpc_result["invalid_bpc_fixed"]
    result["mask_bpc_fixed"] = bpc_result["mask_bpc_fixed"]

    # Ensure annotation appearance streams (all levels)
    result["appearance_streams_added"] = ensure_appearance_streams(pdf, level)

    # Rule 6.3.3: /AP dictionary may only contain /N, remove /R and /D
    result["annotation_ap_keys_removed"] = remove_non_normal_appearance_keys(pdf)

    # Fix Btn widget /AP/N: must be a state subdictionary (rule 6.3.3)
    result["btn_ap_subdicts_fixed"] = fix_button_appearance_subdicts(pdf)

    # Remove /NeedAppearances from /AcroForm (must come after AP generation)
    result["needs_appearances_removed"] = remove_needs_appearances(pdf)

    # Fix annotation flags (all levels)
    result["annotation_flags_fixed"] = fix_annotation_flags(pdf, level)

    # Fix annotation-level /CA opacity (ISO 19005-2, 6.5.3)
    result["annotation_opacity_fixed"] = fix_annotation_opacity(pdf, level)

    # Remove Device color arrays /C and /IC from annotations (ISO 19005-2)
    result["annotation_colors_removed"] = remove_annotation_colors(pdf, level)

    # Validate color spaces (Separation/DeviceN preserved; ICC profiles validated)
    sanitize_colorspaces(pdf, level)

    # Sanitize optional content (layers) for PDF/A-2/3 compliance
    oc_result = sanitize_optional_content(pdf)
    result["oc_as_entries_removed"] = oc_result.get("as_entries_removed", 0)
    result["oc_intents_fixed"] = oc_result.get("intents_fixed", 0)
    result["oc_d_created"] = oc_result.get("d_created", False)
    result["oc_d_name_added"] = oc_result.get("d_name_added", False)
    result["oc_list_mode_fixed"] = oc_result.get("list_mode_fixed", 0)
    result["oc_base_state_fixed"] = oc_result.get("base_state_fixed", 0)
    result["oc_config_names_added"] = oc_result.get("config_names_added", 0)
    result["oc_missing_ocgs_added"] = oc_result.get("missing_ocgs_added", 0)
    result["oc_rbgroups_fixed"] = oc_result.get("rbgroups_fixed", 0)
    result["oc_ocg_names_added"] = oc_result.get("ocg_names_added", 0)
    result["oc_order_ocgs_added"] = oc_result.get("order_ocgs_added", 0)

    # Sanitize Extended Graphics State (all levels)
    extgstate_result = sanitize_extgstate(pdf)
    result["extgstate_fixed"] = extgstate_result.get("extgstate_fixed", 0)

    # Sanitize content stream operators/resources and fix invalid ri operands
    ri_result = sanitize_rendering_intent(pdf)
    result["ri_operators_fixed"] = ri_result.get("ri_operators_fixed", 0)
    result["undefined_operators_removed"] = ri_result.get(
        "undefined_operators_removed", 0
    )
    result["resources_dictionaries_added"] = ri_result.get(
        "resources_dictionaries_added", 0
    )
    result["resources_entries_merged"] = ri_result.get("resources_entries_merged", 0)
    result["image_intents_fixed"] = ri_result.get("image_intents_fixed", 0)

    # Sanitize implementation limits and structural name/string constraints
    structure_result = sanitize_structure_limits(pdf)
    result["structure_strings_truncated"] = structure_result.get("strings_truncated", 0)
    result["structure_names_shortened"] = structure_result.get("names_shortened", 0)
    result["structure_utf8_names_fixed"] = structure_result.get("utf8_names_fixed", 0)
    result["structure_integers_clamped"] = structure_result.get("integers_clamped", 0)
    result["structure_reals_normalized"] = structure_result.get("reals_normalized", 0)
    result["structure_q_nesting_rebalanced"] = structure_result.get(
        "q_nesting_rebalanced", 0
    )
    result["structure_hex_odd_fixed"] = structure_result.get("hex_odd_fixed", 0)

    # Sanitize CIDFont structures for PDF/A-2 compliance (all levels)
    cidfont_result = sanitize_cidfont_structures(pdf)
    result["cidsysteminfo_fixed"] = cidfont_result.get("cidsysteminfo_fixed", 0)
    result["cidtogidmap_fixed"] = cidfont_result.get("cidtogidmap_fixed", 0)
    result["cidset_removed"] = cidfont_result.get("cidset_removed", 0)
    result["type1_charset_removed"] = cidfont_result.get("type1_charset_removed", 0)
    result["cid_values_over_65535_warned"] = cidfont_result.get(
        "cid_values_over_65535_warned", 0
    )

    # Fix FontDescriptor /FontName vs /BaseFont mismatch (ISO 19005-2, 6.3.5)
    fontname_result = sanitize_fontname_consistency(pdf)
    result["fontname_fixed"] = fontname_result.get("fontname_fixed", 0)

    # Fix broken font dictionary structure (ISO 19005-2, 6.2.11.2)
    # Must run before notdef/glyph_coverage/font_widths
    font_structure_result = sanitize_font_structure(pdf)
    result["font_type_added"] = font_structure_result.get("font_type_added", 0)
    result["font_subtype_fixed"] = font_structure_result.get("font_subtype_fixed", 0)
    result["font_basefont_added"] = font_structure_result.get("font_basefont_added", 0)
    result["font_firstchar_added"] = font_structure_result.get(
        "font_firstchar_added", 0
    )
    result["font_lastchar_added"] = font_structure_result.get("font_lastchar_added", 0)
    result["font_widths_size_fixed"] = font_structure_result.get(
        "font_widths_size_fixed", 0
    )
    result["font_stream_subtype_removed"] = font_structure_result.get(
        "font_stream_subtype_removed", 0
    )

    # Fix TrueType font encoding issues (ISO 19005-2, 6.2.11.6)
    # Must run AFTER font_structure (ensures /Subtype and /FontDescriptor exist)
    # and BEFORE notdef/glyph_coverage/font_widths
    tt_enc_result = sanitize_truetype_encoding(pdf)
    result["tt_nonsymbolic_cmap_added"] = tt_enc_result.get(
        "tt_nonsymbolic_cmap_added", 0
    )
    result["tt_nonsymbolic_encoding_fixed"] = tt_enc_result.get(
        "tt_nonsymbolic_encoding_fixed", 0
    )
    result["tt_symbolic_encoding_removed"] = tt_enc_result.get(
        "tt_symbolic_encoding_removed", 0
    )
    result["tt_symbolic_flag_set"] = tt_enc_result.get("tt_symbolic_flag_set", 0)
    result["tt_symbolic_cmap_added"] = tt_enc_result.get("tt_symbolic_cmap_added", 0)

    # Ensure .notdef glyph in all embedded fonts (ISO 19005-2, 6.3.3)
    # Must run BEFORE width validation — adds .notdef which changes font programs
    notdef_result = sanitize_font_notdef(pdf)
    result["notdef_fixed"] = notdef_result.get("notdef_fixed", 0)

    # Ensure all referenced glyphs exist in embedded fonts (ISO 19005-2, 6.2.11.4.1)
    # Must run BEFORE width validation — adds glyphs which changes font programs
    glyph_coverage_result = sanitize_glyph_coverage(pdf)
    result["glyphs_added"] = glyph_coverage_result.get("glyphs_added", 0)

    # Validate and fix font widths (ISO 19005-2, 6.3.7)
    # Must run AFTER notdef/glyph_coverage — reads final font program state
    font_widths_result = sanitize_font_widths(pdf)
    result["simple_font_widths_fixed"] = font_widths_result.get(
        "simple_font_widths_fixed", 0
    )
    result["cidfont_widths_fixed"] = font_widths_result.get("cidfont_widths_fixed", 0)

    # Sanitize invalid Unicode values in existing ToUnicode CMaps (rule 6.2.11.7.2)
    tounicode_values_result = sanitize_tounicode_values(pdf)
    result["tounicode_values_fixed"] = tounicode_values_result.get(
        "tounicode_values_fixed", 0
    )

    # Fill ToUnicode gaps for Unicode levels (2u/3u) — rule 6.2.11.7.2
    # Every glyph used in content streams must be mappable to Unicode.
    # At 'b' levels veraPDF only checks ToUnicode existence, not coverage;
    # at 'u' levels it checks every used glyph individually.
    if level.endswith("u"):
        tounicode_gaps_result = fill_tounicode_gaps(pdf)
        result["tounicode_gaps_filled"] = tounicode_gaps_result.get(
            "tounicode_gaps_filled", 0
        )

        # Wrap PUA-mapped characters in /ActualText (rule 6.2.11.7.3-1)
        pua_at_result = sanitize_pua_actualtext(pdf)
        result["pua_actualtext_added"] = pua_at_result.get("pua_actualtext_added", 0)
        result["pua_actualtext_warnings"] = pua_at_result.get(
            "pua_actualtext_warnings", 0
        )

    # Remove .notdef glyph references from content streams (ISO 19005-2, 6.2.11.8)
    notdef_usage_result = sanitize_notdef_usage(pdf)
    result["notdef_usage_fixed"] = notdef_usage_result.get("notdef_usage_fixed", 0)

    # Remove non-compliant embedded files (only 2b/2u)
    # PDF/A-2 allows embedded files that are themselves PDF/A-1 or PDF/A-2
    if level in ("2b", "2u"):
        embed_result = remove_non_compliant_embedded_files(pdf)
        result["files_removed"] = embed_result["removed"]
        result["embedded_files_kept"] = embed_result["kept"]
        result["embedded_files_converted"] = embed_result.get("converted", 0)

    # Ensure AF relationships and embedded file metadata (2b/2u and 3b/3u)
    # PDF/A-2 (ISO 19005-2) and PDF/A-3 (ISO 19005-3) both require:
    # - /AFRelationship on FileSpec dicts and /Root/AF array
    # - /Subtype on embedded file streams
    # - /Params with /ModDate on embedded file streams
    # - /UF on FileSpec dicts
    # These are no-ops when no embedded files exist.
    result["af_relationships_fixed"] = ensure_af_relationships(pdf)
    result["embedded_file_subtypes_fixed"] = ensure_embedded_file_subtypes(pdf)
    result["embedded_file_params_fixed"] = ensure_embedded_file_params(pdf)
    result["filespec_uf_fixed"] = ensure_filespec_uf_entries(pdf)
    result["filespec_desc_fixed"] = ensure_filespec_desc(pdf)

    # Sanitize forbidden filters on embedded file streams (ISO 19005-2, 6.1.4)
    ef_filter_result = sanitize_embedded_file_filters(pdf)
    result["embedded_file_lzw_converted"] = ef_filter_result["lzw_converted"]
    result["embedded_file_crypt_removed"] = ef_filter_result["crypt_removed"]

    logger.info(
        "Sanitization completed: %d JS, %d actions, %d invalid dests, "
        "%d files removed, "
        "%d embedded files converted, %d embedded files kept, "
        "%d XFA, %d forbidden catalog entries, "
        "%d viewer prefs entries removed, "
        "catalog /Lang set: %s, "
        "%d forbidden annots, %d forbidden XObjects removed, "
        "%d AP streams added, %d annot flags fixed, %d annot /CA fixed, "
        "%d annot colors removed, %d annot AP keys removed, "
        "%d Btn AP subdicts fixed, "
        "%d LZW streams converted, %d Crypt filters removed, "
        "%d external stream keys removed, "
        "%d non-standard inline filters fixed, "
        "%d JBIG2 streams converted, %d JBIG2 reencoded, %d JBIG2 failed, "
        "%d JPX colr fixed, %d JPX wrapped, %d JPX reencoded, %d JPX failed, "
        "%d OC AS entries removed, %d OC intents fixed, "
        "OC D /Name added: %s, %d OC ListMode fixed, "
        "%d OC BaseState fixed, %d OC config names added, "
        "%d OC missing OCGs added, %d OC RBGroups fixed, "
        "%d OC order OCGs added, "
        "%d ExtGState entries fixed, %d ri operators fixed, "
        "%d undefined operators removed, "
        "%d resources dicts added, %d resource entries merged, "
        "%d image intents fixed, "
        "%d CIDSystemInfo fixed, %d CIDToGIDMap fixed, %d CIDSet removed, "
        "%d Type1 /CharSet removed, "
        "%d AF relationships fixed, %d embedded file subtypes fixed, "
        "%d embedded file params fixed, "
        "%d FileSpec /UF entries fixed, "
        "%d FileSpec /Desc entries fixed, "
        "%d image interpolate fixed, "
        "%d simple font widths fixed, %d CIDFont widths fixed, "
        "%d .notdef glyphs fixed, "
        "%d glyph coverage glyphs added, "
        "%d .notdef usage operators fixed, "
        "%d ToUnicode values fixed, "
        "%d PUA ActualText added, "
        "%d signatures found, %d signatures removed, "
        "%d sigflags fixed, %d signature /Type fixed, "
        "NeedAppearances removed: %s, "
        "%d MediaBox inherited, %d boxes normalized, "
        "%d boxes clipped, %d TrimBox added, "
        "%d malformed boxes removed, "
        "MarkInfo added: %s, "
        "%d font /Type added, %d font /Subtype fixed, "
        "%d font /BaseFont added, %d font /FirstChar added, "
        "%d font /LastChar added, %d font /Widths size fixed, "
        "%d font stream subtypes removed, "
        "%d TT non-symbolic (3,1) cmaps added, "
        "%d TT non-symbolic encodings fixed, "
        "%d TT symbolic /Encoding entries removed, "
        "%d TT symbolic Symbolic flags set, "
        "%d TT symbolic (3,0) cmaps added",
        result["javascript_removed"],
        result["actions_removed"],
        result["invalid_destinations_removed"],
        result["files_removed"],
        result["embedded_files_converted"],
        result["embedded_files_kept"],
        result["xfa_removed"],
        result["forbidden_catalog_entries_removed"],
        result["viewer_prefs_entries_removed"],
        result["catalog_lang_set"],
        result["forbidden_annotations_removed"],
        result["forbidden_xobjects_removed"],
        result["appearance_streams_added"],
        result["annotation_flags_fixed"],
        result["annotation_opacity_fixed"],
        result["annotation_colors_removed"],
        result["annotation_ap_keys_removed"],
        result["btn_ap_subdicts_fixed"],
        result["lzw_streams_converted"],
        result["crypt_streams_removed"],
        result["external_stream_keys_removed"],
        result["nonstandard_inline_filters_fixed"],
        result["jbig2_converted"],
        result["jbig2_reencoded"],
        result["jbig2_failed"],
        result["jpx_fixed"],
        result["jpx_wrapped"],
        result["jpx_reencoded"],
        result["jpx_failed"],
        result["oc_as_entries_removed"],
        result["oc_intents_fixed"],
        result["oc_d_name_added"],
        result["oc_list_mode_fixed"],
        result["oc_base_state_fixed"],
        result["oc_config_names_added"],
        result["oc_missing_ocgs_added"],
        result["oc_rbgroups_fixed"],
        result["oc_order_ocgs_added"],
        result["extgstate_fixed"],
        result["ri_operators_fixed"],
        result["undefined_operators_removed"],
        result["resources_dictionaries_added"],
        result["resources_entries_merged"],
        result["image_intents_fixed"],
        result["cidsysteminfo_fixed"],
        result["cidtogidmap_fixed"],
        result["cidset_removed"],
        result["type1_charset_removed"],
        result["af_relationships_fixed"],
        result["embedded_file_subtypes_fixed"],
        result["embedded_file_params_fixed"],
        result["filespec_uf_fixed"],
        result["filespec_desc_fixed"],
        result["image_interpolate_fixed"],
        result["simple_font_widths_fixed"],
        result["cidfont_widths_fixed"],
        result["notdef_fixed"],
        result["glyphs_added"],
        result["notdef_usage_fixed"],
        result["tounicode_values_fixed"],
        result["pua_actualtext_added"],
        result["signatures_found"],
        result["signatures_removed"],
        result["sigflags_fixed"],
        result["signatures_type_fixed"],
        result["needs_appearances_removed"],
        result["mediabox_inherited"],
        result["boxes_normalized"],
        result["boxes_clipped"],
        result["trimbox_added"],
        result["malformed_boxes_removed"],
        result["mark_info_added"],
        result["font_type_added"],
        result["font_subtype_fixed"],
        result["font_basefont_added"],
        result["font_firstchar_added"],
        result["font_lastchar_added"],
        result["font_widths_size_fixed"],
        result["font_stream_subtype_removed"],
        result["tt_nonsymbolic_cmap_added"],
        result["tt_nonsymbolic_encoding_fixed"],
        result["tt_symbolic_encoding_removed"],
        result["tt_symbolic_flag_set"],
        result["tt_symbolic_cmap_added"],
    )

    return result


# Re-export for backward compatibility
__all__ = [
    # Main function
    "sanitize_for_pdfa",
    # Individual sanitizers
    "ensure_af_relationships",
    "ensure_embedded_file_params",
    "ensure_embedded_file_subtypes",
    "ensure_filespec_desc",
    "ensure_filespec_uf_entries",
    "ensure_appearance_streams",
    "fix_button_appearance_subdicts",
    "remove_needs_appearances",
    "remove_javascript",
    "remove_actions",
    "validate_destinations",
    "remove_embedded_files",
    "remove_non_compliant_embedded_files",
    "sanitize_embedded_file_filters",
    "ensure_catalog_lang",
    "ensure_mark_info",
    "remove_catalog_version",
    "remove_forbidden_catalog_entries",
    "remove_forbidden_page_entries",
    "remove_forbidden_viewer_preferences",
    "remove_xfa_forms",
    "remove_forbidden_annotations",
    "remove_forbidden_xobjects",
    "fix_image_interpolate",
    "fix_bits_per_component",
    "fix_annotation_flags",
    "fix_annotation_opacity",
    "remove_annotation_colors",
    "remove_non_normal_appearance_keys",
    "sanitize_colorspaces",
    "sanitize_optional_content",
    "sanitize_extgstate",
    "sanitize_rendering_intent",
    "sanitize_structure_limits",
    "sanitize_cidfont_structures",
    "sanitize_fontname_consistency",
    "sanitize_font_structure",
    "sanitize_font_notdef",
    "sanitize_truetype_encoding",
    "sanitize_font_widths",
    "sanitize_glyph_coverage",
    "sanitize_notdef_usage",
    "sanitize_pua_actualtext",
    "sanitize_tounicode_values",
    "sanitize_page_boxes",
    "sanitize_signatures",
    "convert_lzw_streams",
    "remove_crypt_streams",
    "remove_external_stream_keys",
    "fix_stream_lengths",
    "sanitize_nonstandard_inline_filters",
    "convert_jbig2_external_globals",
    "sanitize_jpx_color_boxes",
    # Constants
    "COMPLIANT_ACTIONS",
    "FORBIDDEN_ANNOTATION_SUBTYPES",
    "FORBIDDEN_XOBJECT_SUBTYPES",
    "ANNOT_FLAG_INVISIBLE",
    "ANNOT_FLAG_PRINT",
    "ANNOT_FLAG_HIDDEN",
    "ANNOT_FLAG_NOROTATE",
    "ANNOT_FLAG_NOVIEW",
    "ANNOT_FLAG_NOZOOM",
    "SUBMITFORM_FLAG_EXPORTFORMAT",
    "SUBMITFORM_FLAG_XFDF",
    "SUBMITFORM_FLAG_SUBMITPDF",
    # Helper functions (for testing)
    "_is_javascript_action",
    "_is_non_compliant_action",
]
