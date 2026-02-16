# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Document Catalog sanitization."""

import pikepdf
import pytest
from conftest import new_pdf

from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.catalog import (
    _is_valid_bcp47,
    ensure_catalog_lang,
    ensure_mark_info,
    remove_catalog_version,
    remove_forbidden_catalog_entries,
    remove_forbidden_name_dictionary_entries,
    remove_forbidden_page_entries,
    remove_forbidden_viewer_preferences,
)


class TestRemoveForbiddenCatalogEntries:
    """Tests for remove_forbidden_catalog_entries()."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("/Perms", pikepdf.Dictionary()),
            (
                "/Requirements",
                pikepdf.Array(
                    [pikepdf.Dictionary({"/Type": pikepdf.Name("/Requirement")})]
                ),
            ),
            ("/Collection", pikepdf.Dictionary()),
            ("/NeedsRendering", True),
            ("/Threads", pikepdf.Array([pikepdf.Dictionary()])),
            ("/SpiderInfo", pikepdf.Dictionary()),
        ],
    )
    def test_remove_single_forbidden_entry(self, key, value):
        """Each forbidden catalog entry is individually removed."""
        pdf = new_pdf()
        pdf.Root[key] = value
        assert remove_forbidden_catalog_entries(pdf) == 1
        assert key not in pdf.Root

    def test_remove_all_six(self):
        pdf = new_pdf()
        pdf.Root["/Perms"] = pikepdf.Dictionary()
        pdf.Root["/Requirements"] = pikepdf.Array()
        pdf.Root["/Collection"] = pikepdf.Dictionary()
        pdf.Root["/NeedsRendering"] = True
        pdf.Root["/Threads"] = pikepdf.Array()
        pdf.Root["/SpiderInfo"] = pikepdf.Dictionary()
        assert remove_forbidden_catalog_entries(pdf) == 6
        assert "/Perms" not in pdf.Root
        assert "/Requirements" not in pdf.Root
        assert "/Collection" not in pdf.Root
        assert "/NeedsRendering" not in pdf.Root
        assert "/Threads" not in pdf.Root
        assert "/SpiderInfo" not in pdf.Root

    def test_no_forbidden_keys_present(self):
        pdf = new_pdf()
        assert remove_forbidden_catalog_entries(pdf) == 0

    def test_other_keys_preserved(self):
        pdf = new_pdf()
        pdf.Root["/Perms"] = pikepdf.Dictionary()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary()
        remove_forbidden_catalog_entries(pdf)
        assert "/MarkInfo" in pdf.Root
        assert "/Perms" not in pdf.Root


class TestRemoveCatalogVersion:
    """Tests for remove_catalog_version()."""

    def test_removes_version_higher_than_required(self):
        """Catalog /Version /2.0 is removed when required version is 1.7."""
        pdf = new_pdf()
        pdf.Root["/Version"] = pikepdf.Name("/2.0")
        assert remove_catalog_version(pdf, "1.7") is True
        assert "/Version" not in pdf.Root

    def test_removes_redundant_version(self):
        """Catalog /Version equal to required version is removed (redundant)."""
        pdf = new_pdf()
        pdf.Root["/Version"] = pikepdf.Name("/1.7")
        assert remove_catalog_version(pdf, "1.7") is True
        assert "/Version" not in pdf.Root

    def test_keeps_lower_version(self):
        """Catalog /Version lower than required is left unchanged."""
        pdf = new_pdf()
        pdf.Root["/Version"] = pikepdf.Name("/1.5")
        assert remove_catalog_version(pdf, "1.7") is False
        assert "/Version" in pdf.Root

    def test_no_version_present(self):
        """No-op when /Version is absent."""
        pdf = new_pdf()
        assert remove_catalog_version(pdf, "1.7") is False

    def test_integration_via_sanitize_for_pdfa(self):
        """sanitize_for_pdfa() removes catalog /Version /2.0."""
        pdf = new_pdf()
        pdf.Root["/Version"] = pikepdf.Name("/2.0")
        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["catalog_version_removed"] is True
        assert "/Version" not in pdf.Root

    def test_integration_no_version(self):
        """sanitize_for_pdfa() reports False when no /Version present."""
        pdf = new_pdf()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["catalog_version_removed"] is False


class TestRemoveForbiddenNameDictionaryEntries:
    """Tests for remove_forbidden_name_dictionary_entries()."""

    def test_remove_alternate_presentations(self):
        pdf = new_pdf()
        pdf.Root["/Names"] = pikepdf.Dictionary(
            {"/AlternatePresentations": pikepdf.Dictionary()}
        )
        assert remove_forbidden_name_dictionary_entries(pdf) == 1
        assert "/Names" not in pdf.Root

    def test_keep_other_name_dictionary_entries(self):
        pdf = new_pdf()
        pdf.Root["/Names"] = pikepdf.Dictionary(
            {
                "/AlternatePresentations": pikepdf.Dictionary(),
                "/Dests": pikepdf.Dictionary(),
            }
        )
        assert remove_forbidden_name_dictionary_entries(pdf) == 1
        assert "/Names" in pdf.Root
        assert "/AlternatePresentations" not in pdf.Root["/Names"]
        assert "/Dests" in pdf.Root["/Names"]

    def test_no_names_dictionary(self):
        pdf = new_pdf()
        assert remove_forbidden_name_dictionary_entries(pdf) == 0


class TestRemoveForbiddenPageEntries:
    """Tests for remove_forbidden_page_entries()."""

    def test_remove_pressteps_from_page(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                PresSteps=pikepdf.Array([pikepdf.Dictionary()]),
            )
        )
        pdf.pages.append(page)

        assert remove_forbidden_page_entries(pdf) == 1
        assert "/PresSteps" not in pdf.pages[0].obj

    def test_remove_pressteps_from_multiple_pages(self):
        pdf = new_pdf()
        p1 = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                PresSteps=pikepdf.Array([pikepdf.Dictionary()]),
            )
        )
        p2 = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                PresSteps=pikepdf.Array([pikepdf.Dictionary()]),
            )
        )
        pdf.pages.append(p1)
        pdf.pages.append(p2)

        assert remove_forbidden_page_entries(pdf) == 2
        assert "/PresSteps" not in pdf.pages[0].obj
        assert "/PresSteps" not in pdf.pages[1].obj

    def test_remove_duration_from_page(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                Duration=5,
            )
        )
        pdf.pages.append(page)

        assert remove_forbidden_page_entries(pdf) == 1
        assert "/Duration" not in pdf.pages[0].obj

    def test_remove_duration_and_pressteps(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                PresSteps=pikepdf.Array([pikepdf.Dictionary()]),
                Duration=3,
            )
        )
        pdf.pages.append(page)

        assert remove_forbidden_page_entries(pdf) == 2
        assert "/PresSteps" not in pdf.pages[0].obj
        assert "/Duration" not in pdf.pages[0].obj

    def test_no_pressteps_in_pages(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)

        assert remove_forbidden_page_entries(pdf) == 0


class TestSanitizeForPdfaIntegration:
    """Integration tests via sanitize_for_pdfa()."""

    def test_result_dict_contains_key(self):
        pdf = new_pdf()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert "forbidden_catalog_entries_removed" in result

    def test_forbidden_entries_removed_via_sanitize(self):
        pdf = new_pdf()
        pdf.Root["/Perms"] = pikepdf.Dictionary()
        pdf.Root["/Collection"] = pikepdf.Dictionary()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["forbidden_catalog_entries_removed"] == 2
        assert "/Perms" not in pdf.Root
        assert "/Collection" not in pdf.Root

    def test_forbidden_names_entries_removed_via_sanitize(self):
        pdf = new_pdf()
        pdf.Root["/Names"] = pikepdf.Dictionary(
            {
                "/AlternatePresentations": pikepdf.Dictionary(),
                "/Dests": pikepdf.Dictionary(),
            }
        )
        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["forbidden_name_dict_entries_removed"] == 1
        assert "/Names" in pdf.Root
        assert "/AlternatePresentations" not in pdf.Root["/Names"]
        assert "/Dests" in pdf.Root["/Names"]

    def test_forbidden_page_entries_removed_via_sanitize(self):
        pdf = new_pdf()
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([0, 0, 612, 792]),
                PresSteps=pikepdf.Array([pikepdf.Dictionary()]),
            )
        )
        pdf.pages.append(page)

        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["forbidden_page_entries_removed"] == 1
        assert "/PresSteps" not in pdf.pages[0].obj


class TestEnsureCatalogLang:
    """Tests for ensure_catalog_lang()."""

    def test_lang_already_present(self):
        """Existing /Lang is preserved and not overwritten."""
        pdf = new_pdf()
        pdf.Root["/Lang"] = pikepdf.String("de-DE")
        assert ensure_catalog_lang(pdf) is False
        assert str(pdf.Root["/Lang"]) == "de-DE"

    def test_no_xmp_sets_und(self):
        """/Lang is set to 'und' when no XMP metadata exists."""
        pdf = new_pdf()
        assert "/Lang" not in pdf.Root
        assert ensure_catalog_lang(pdf) is True
        assert str(pdf.Root["/Lang"]) == "und"

    def test_lang_from_xmp_dc_language(self):
        """/Lang is extracted from dc:language in XMP metadata."""
        pdf = new_pdf()
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b"<rdf:Description"
            b' xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:language><rdf:Bag><rdf:li>fr-FR</rdf:li>"
            b"</rdf:Bag></dc:language>"
            b"</rdf:Description></rdf:RDF></x:xmpmeta>"
            b'<?xpacket end="w"?>'
        )
        metadata_stream = pikepdf.Stream(pdf, xmp)
        metadata_stream["/Type"] = pikepdf.Name("/Metadata")
        metadata_stream["/Subtype"] = pikepdf.Name("/XML")
        pdf.Root["/Metadata"] = pdf.make_indirect(metadata_stream)

        assert ensure_catalog_lang(pdf) is True
        assert str(pdf.Root["/Lang"]) == "fr-FR"

    def test_xmp_without_dc_language_sets_und(self):
        """Falls back to 'und' when XMP exists but has no dc:language."""
        pdf = new_pdf()
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:title><rdf:Alt><rdf:li>Test</rdf:li></rdf:Alt></dc:title>"
            b"</rdf:Description></rdf:RDF></x:xmpmeta>"
            b'<?xpacket end="w"?>'
        )
        metadata_stream = pikepdf.Stream(pdf, xmp)
        metadata_stream["/Type"] = pikepdf.Name("/Metadata")
        metadata_stream["/Subtype"] = pikepdf.Name("/XML")
        pdf.Root["/Metadata"] = pdf.make_indirect(metadata_stream)

        assert ensure_catalog_lang(pdf) is True
        assert str(pdf.Root["/Lang"]) == "und"

    def test_integration_via_sanitize_for_pdfa(self):
        """sanitize_for_pdfa() includes catalog_lang_set in result."""
        pdf = new_pdf()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert "catalog_lang_set" in result
        assert result["catalog_lang_set"] is True
        assert str(pdf.Root["/Lang"]) == "und"

    def test_integration_lang_already_set(self):
        """sanitize_for_pdfa() reports False when /Lang already exists."""
        pdf = new_pdf()
        pdf.Root["/Lang"] = pikepdf.String("en")
        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["catalog_lang_set"] is False
        assert str(pdf.Root["/Lang"]) == "en"

    def test_invalid_xmp_lang_falls_back_to_und(self):
        """Invalid BCP 47 tag in XMP causes fallback to 'und'."""
        pdf = new_pdf()
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b"<rdf:Description"
            b' xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:language><rdf:Bag><rdf:li>not a valid tag!</rdf:li>"
            b"</rdf:Bag></dc:language>"
            b"</rdf:Description></rdf:RDF></x:xmpmeta>"
            b'<?xpacket end="w"?>'
        )
        metadata_stream = pikepdf.Stream(pdf, xmp)
        metadata_stream["/Type"] = pikepdf.Name("/Metadata")
        metadata_stream["/Subtype"] = pikepdf.Name("/XML")
        pdf.Root["/Metadata"] = pdf.make_indirect(metadata_stream)

        assert ensure_catalog_lang(pdf) is True
        assert str(pdf.Root["/Lang"]) == "und"


class TestIsValidBcp47:
    """Tests for _is_valid_bcp47()."""

    @pytest.mark.parametrize(
        "tag",
        [
            "en",
            "fr",
            "de",
            "und",
            "en-US",
            "de-DE",
            "fr-FR",
            "zh-CN",
            "pt-BR",
            "en-Latn",
            "zh-Hans",
            "sr-Cyrl",
            "zh-Hans-CN",
            "sr-Cyrl-RS",
            "en-US-x-custom",
            "sl-rozaj-1994",
            "x-private",
            "x-a-b",
            "i-klingon",
            "zh-min-nan",
            "art-lojban",
            "en-GB-oed",
        ],
    )
    def test_valid_tags(self, tag):
        assert _is_valid_bcp47(tag) is True

    @pytest.mark.parametrize(
        "tag",
        [
            "",
            "e",
            "toolonglanguage",
            "en_US",
            "en US",
            "123",
            "en-",
            "-en",
            "not a valid tag!",
            "en--US",
            "a",
        ],
    )
    def test_invalid_tags(self, tag):
        assert _is_valid_bcp47(tag) is False


class TestRemoveForbiddenViewerPreferences:
    """Tests for remove_forbidden_viewer_preferences()."""

    @pytest.mark.parametrize(
        "key",
        ["/ViewArea", "/ViewClip", "/PrintArea", "/PrintClip"],
    )
    def test_remove_single_forbidden_vp(self, key):
        """Each forbidden ViewerPreferences key is individually removed."""
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {key: pikepdf.Name("/CropBox")}
        )
        assert remove_forbidden_viewer_preferences(pdf) == 1

    def test_remove_all_four(self):
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {
                "/ViewArea": pikepdf.Name("/CropBox"),
                "/ViewClip": pikepdf.Name("/CropBox"),
                "/PrintArea": pikepdf.Name("/BleedBox"),
                "/PrintClip": pikepdf.Name("/BleedBox"),
            }
        )
        assert remove_forbidden_viewer_preferences(pdf) == 4

    def test_no_viewer_preferences(self):
        pdf = new_pdf()
        assert remove_forbidden_viewer_preferences(pdf) == 0

    def test_other_keys_preserved(self):
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {
                "/HideToolbar": True,
                "/ViewArea": pikepdf.Name("/CropBox"),
            }
        )
        assert remove_forbidden_viewer_preferences(pdf) == 1
        vp = pdf.Root["/ViewerPreferences"]
        assert "/HideToolbar" in vp
        assert "/ViewArea" not in vp

    def test_empty_vp_removed_after_cleanup(self):
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {"/ViewArea": pikepdf.Name("/CropBox")}
        )
        remove_forbidden_viewer_preferences(pdf)
        assert "/ViewerPreferences" not in pdf.Root

    def test_nonempty_vp_kept_after_cleanup(self):
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {
                "/HideToolbar": True,
                "/ViewArea": pikepdf.Name("/CropBox"),
            }
        )
        remove_forbidden_viewer_preferences(pdf)
        assert "/ViewerPreferences" in pdf.Root

    def test_integration_via_sanitize_for_pdfa(self):
        pdf = new_pdf()
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
            {
                "/ViewArea": pikepdf.Name("/CropBox"),
                "/PrintClip": pikepdf.Name("/BleedBox"),
            }
        )
        result = sanitize_for_pdfa(pdf, level="3b")
        assert result["viewer_prefs_entries_removed"] == 2
        assert "/ViewerPreferences" not in pdf.Root


class TestEnsureMarkInfo:
    """Tests for ensure_mark_info()."""

    def test_adds_markinfo_when_missing(self):
        """Creates /MarkInfo with /Marked false when absent."""
        pdf = new_pdf()
        assert "/MarkInfo" not in pdf.Root
        assert ensure_mark_info(pdf) is True
        assert "/MarkInfo" in pdf.Root
        mark_info = pdf.Root["/MarkInfo"]
        assert bool(mark_info.get("/Marked")) is False

    def test_adds_marked_to_existing_markinfo(self):
        """/MarkInfo exists but has no /Marked -> /Marked false added."""
        pdf = new_pdf()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary()
        assert ensure_mark_info(pdf) is True
        assert bool(pdf.Root["/MarkInfo"].get("/Marked")) is False

    def test_preserves_marked_true(self):
        """Existing /Marked true is NOT changed to false."""
        pdf = new_pdf()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary(Marked=True)
        assert ensure_mark_info(pdf) is False
        assert bool(pdf.Root["/MarkInfo"]["/Marked"]) is True

    def test_preserves_marked_false(self):
        """Existing /Marked false is kept unchanged."""
        pdf = new_pdf()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary(Marked=False)
        assert ensure_mark_info(pdf) is False

    def test_preserves_other_markinfo_keys(self):
        """/MarkInfo with other keys but no /Marked -> /Marked added, others kept."""
        pdf = new_pdf()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary(
            UserProperties=True,
        )
        assert ensure_mark_info(pdf) is True
        mark_info = pdf.Root["/MarkInfo"]
        assert bool(mark_info.get("/Marked")) is False
        assert bool(mark_info.get("/UserProperties")) is True

    def test_integration_via_sanitize_for_pdfa(self):
        """sanitize_for_pdfa() includes mark_info_added in result."""
        pdf = new_pdf()
        result = sanitize_for_pdfa(pdf, level="3b")
        assert "mark_info_added" in result
        assert result["mark_info_added"] is True
        assert "/MarkInfo" in pdf.Root

    def test_integration_markinfo_already_present(self):
        """sanitize_for_pdfa() reports False when /MarkInfo already has /Marked."""
        pdf = new_pdf()
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary(Marked=True)
        result = sanitize_for_pdfa(pdf, level="2b")
        assert result["mark_info_added"] is False
