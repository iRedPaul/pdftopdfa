# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Optional Content (Layers) sanitization for PDF/A compliance."""

from collections.abc import Generator

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.optional_content import sanitize_optional_content
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


class TestASEntryRemoval:
    """Tests for /AS entry removal from OCProperties."""

    @pytest.fixture
    def pdf_with_as_in_default_config(self) -> Generator[Pdf, None, None]:
        """PDF with /AS entry in default OCProperties config."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create an OCG
        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )

        # Create OCProperties with /AS in default config
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(
                Name="Default",
                AS=Array(
                    [
                        Dictionary(
                            Event=Name.View,
                            OCGs=Array([ocg]),
                            Category=Array([Name.View]),
                        )
                    ]
                ),
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_as_in_alternate_configs(self) -> Generator[Pdf, None, None]:
        """PDF with /AS entries in alternate OCProperties configs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCGs
        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )

        # Create OCProperties with /AS in alternate configs
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(Name="Default"),
            Configs=Array(
                [
                    Dictionary(
                        Name="Config1",
                        AS=Array(
                            [
                                Dictionary(Event=Name.View, OCGs=Array([ocg1])),
                            ]
                        ),
                    ),
                    Dictionary(
                        Name="Config2",
                        AS=Array(
                            [
                                Dictionary(Event=Name.Print, OCGs=Array([ocg2])),
                            ]
                        ),
                    ),
                ]
            ),
        )

        yield pdf

    def test_remove_as_from_default_config(self, pdf_with_as_in_default_config: Pdf):
        """Removes /AS entry from default OCProperties config."""
        result = sanitize_optional_content(pdf_with_as_in_default_config)

        assert result["as_entries_removed"] == 1
        assert "/AS" not in pdf_with_as_in_default_config.Root.OCProperties.D

    def test_remove_as_from_alternate_configs(
        self, pdf_with_as_in_alternate_configs: Pdf
    ):
        """Removes /AS entries from alternate OCProperties configs."""
        result = sanitize_optional_content(pdf_with_as_in_alternate_configs)

        assert result["as_entries_removed"] == 2
        for config in pdf_with_as_in_alternate_configs.Root.OCProperties.Configs:
            assert "/AS" not in config


class TestIntentCorrection:
    """Tests for OCG /Intent correction."""

    @pytest.fixture
    def pdf_with_design_intent(self) -> Generator[Pdf, None, None]:
        """PDF with OCG having /Intent /Design."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCG with /Design intent
        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="DesignLayer", Intent=Name.Design)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_intent_array(self) -> Generator[Pdf, None, None]:
        """PDF with OCG having /Intent as array with mixed values."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCG with mixed intent array
        ocg = pdf.make_indirect(
            Dictionary(
                Type=Name.OCG,
                Name="MixedIntentLayer",
                Intent=Array([Name.View, Name.Design]),
            )
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_design_only_array(self) -> Generator[Pdf, None, None]:
        """PDF with OCG having /Intent as array with only non-View values."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCG with design-only intent array
        ocg = pdf.make_indirect(
            Dictionary(
                Type=Name.OCG,
                Name="DesignOnlyLayer",
                Intent=Array([Name.Design]),
            )
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    def test_fix_design_intent(self, pdf_with_design_intent: Pdf):
        """Corrects OCG /Intent from /Design to /View."""
        result = sanitize_optional_content(pdf_with_design_intent)

        assert result["intents_fixed"] == 1
        assert result["ocgs_processed"] == 1

        ocg = pdf_with_design_intent.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert str(ocg.Intent) == "/View"

    def test_fix_intent_array_with_view(self, pdf_with_intent_array: Pdf):
        """Corrects OCG /Intent array containing View and Design to just /View."""
        result = sanitize_optional_content(pdf_with_intent_array)

        assert result["intents_fixed"] == 1
        assert result["ocgs_processed"] == 1

        ocg = pdf_with_intent_array.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert str(ocg.Intent) == "/View"

    def test_fix_intent_array_design_only(self, pdf_with_design_only_array: Pdf):
        """Corrects OCG /Intent array with only /Design to /View."""
        result = sanitize_optional_content(pdf_with_design_only_array)

        assert result["intents_fixed"] == 1

        ocg = pdf_with_design_only_array.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert str(ocg.Intent) == "/View"


class TestNoChangesNeeded:
    """Tests for PDFs that don't need optional content changes."""

    def test_pdf_without_ocproperties(self, sample_pdf_obj: Pdf):
        """PDF without OCProperties returns zero counts."""
        result = sanitize_optional_content(sample_pdf_obj)

        assert result["as_entries_removed"] == 0
        assert result["intents_fixed"] == 0
        assert result["ocgs_processed"] == 0

    @pytest.fixture
    def pdf_with_compliant_oc(self) -> Generator[Pdf, None, None]:
        """PDF with already-compliant optional content."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create compliant OCG (Intent=View, no AS)
        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="CompliantLayer", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    def test_already_compliant_no_changes(self, pdf_with_compliant_oc: Pdf):
        """Already-compliant OCProperties don't get modified."""
        result = sanitize_optional_content(pdf_with_compliant_oc)

        assert result["as_entries_removed"] == 0
        assert result["intents_fixed"] == 0
        assert result["ocgs_processed"] == 1

    @pytest.fixture
    def pdf_with_no_intent(self) -> Generator[Pdf, None, None]:
        """PDF with OCG that has no /Intent (defaults to View per spec)."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCG without Intent (defaults to View)
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="NoIntentLayer"))

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    def test_no_intent_not_modified(self, pdf_with_no_intent: Pdf):
        """OCG without /Intent is not modified (default is View)."""
        result = sanitize_optional_content(pdf_with_no_intent)

        assert result["intents_fixed"] == 0
        assert result["ocgs_processed"] == 1

        ocg = pdf_with_no_intent.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert "/Intent" not in ocg


class TestMultipleOCGs:
    """Tests for PDFs with multiple OCGs."""

    @pytest.fixture
    def pdf_with_multiple_ocgs(self) -> Generator[Pdf, None, None]:
        """PDF with multiple OCGs with various intents."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        # Create OCGs with different intents
        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="ViewLayer", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="DesignLayer", Intent=Name.Design)
        )
        ocg3 = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="NoIntentLayer"))
        ocg4 = pdf.make_indirect(
            Dictionary(
                Type=Name.OCG,
                Name="MixedLayer",
                Intent=Array([Name.View, Name.Design]),
            )
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2, ocg3, ocg4]),
            D=Dictionary(
                Name="Default",
                AS=Array([Dictionary(Event=Name.View, OCGs=Array([ocg1]))]),
            ),
        )

        yield pdf

    def test_multiple_ocgs_sanitized(self, pdf_with_multiple_ocgs: Pdf):
        """Multiple OCGs are processed correctly."""
        result = sanitize_optional_content(pdf_with_multiple_ocgs)

        assert result["as_entries_removed"] == 1
        assert result["intents_fixed"] == 2  # ocg2 and ocg4
        assert result["ocgs_processed"] == 4


class TestDNameEntry:
    """Tests for /D config /Name entry (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_d_no_name(self) -> Generator[Pdf, None, None]:
        """PDF with /D config that has no /Name entry."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_d_empty_name(self) -> Generator[Pdf, None, None]:
        """PDF with /D config whose /Name is an empty string."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name=""),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_d_existing_name(self) -> Generator[Pdf, None, None]:
        """PDF with /D config that already has a /Name entry."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="MyConfig"),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_d_no_name_with_as(self) -> Generator[Pdf, None, None]:
        """PDF with /D config that has /AS but no /Name."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(
                AS=Array(
                    [
                        Dictionary(
                            Event=Name.View,
                            OCGs=Array([ocg]),
                            Category=Array([Name.View]),
                        )
                    ]
                ),
            ),
        )

        yield pdf

    def test_adds_name_to_d_config_without_name(self, pdf_with_d_no_name: Pdf):
        """Adds /Name to /D config when missing."""
        result = sanitize_optional_content(pdf_with_d_no_name)

        assert result["d_name_added"] is True
        d_config = pdf_with_d_no_name.Root.OCProperties.D
        assert "/Name" in d_config
        assert str(d_config.Name) == "Default"

    def test_preserves_existing_name_in_d_config(self, pdf_with_d_existing_name: Pdf):
        """Does not modify /D config when /Name already exists."""
        result = sanitize_optional_content(pdf_with_d_existing_name)

        assert result["d_name_added"] is False
        d_config = pdf_with_d_existing_name.Root.OCProperties.D
        assert str(d_config.Name) == "MyConfig"

    def test_replaces_empty_name_in_d_config(self, pdf_with_d_empty_name: Pdf):
        """Replaces empty /Name in /D config with a non-empty fallback."""
        result = sanitize_optional_content(pdf_with_d_empty_name)

        assert result["d_name_added"] is True
        d_config = pdf_with_d_empty_name.Root.OCProperties.D
        assert "/Name" in d_config
        assert str(d_config.Name) == "Default"

    def test_d_config_missing_name_with_as(self, pdf_with_d_no_name_with_as: Pdf):
        """Both /Name is added and /AS is removed when both issues exist."""
        result = sanitize_optional_content(pdf_with_d_no_name_with_as)

        assert result["d_name_added"] is True
        assert result["as_entries_removed"] == 1
        d_config = pdf_with_d_no_name_with_as.Root.OCProperties.D
        assert "/Name" in d_config
        assert "/AS" not in d_config


class TestIntegration:
    """Integration tests with sanitize_for_pdfa."""

    def test_sanitize_for_pdfa_includes_oc_results(self, sample_pdf_obj: Pdf):
        """sanitize_for_pdfa returns optional content results."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "oc_as_entries_removed" in result
        assert "oc_intents_fixed" in result
        assert "oc_d_created" in result
        assert "oc_d_name_added" in result
        assert "oc_list_mode_fixed" in result
        assert "oc_base_state_fixed" in result
        assert "oc_config_names_added" in result
        assert "oc_missing_ocgs_added" in result
        assert "oc_rbgroups_fixed" in result
        assert "oc_ocg_names_added" in result
        assert "oc_order_ocgs_added" in result


class TestListMode:
    """Tests for /ListMode validation (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_visible_pages_listmode(self) -> Generator[Pdf, None, None]:
        """PDF with /ListMode /VisiblePages in default config."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", ListMode=Name.VisiblePages),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_allpages_listmode(self) -> Generator[Pdf, None, None]:
        """PDF with /ListMode /AllPages in default config (compliant)."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", ListMode=Name.AllPages),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_listmode_in_configs(self) -> Generator[Pdf, None, None]:
        """PDF with /ListMode /VisiblePages in alternate configs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
            Configs=Array(
                [
                    Dictionary(Name="Alt1", ListMode=Name.VisiblePages),
                    Dictionary(Name="Alt2", ListMode=Name.AllPages),
                ]
            ),
        )

        yield pdf

    def test_removes_visible_pages_listmode(self, pdf_with_visible_pages_listmode: Pdf):
        """Removes /ListMode /VisiblePages from default config."""
        result = sanitize_optional_content(pdf_with_visible_pages_listmode)

        assert result["list_mode_fixed"] == 1
        assert "/ListMode" not in (pdf_with_visible_pages_listmode.Root.OCProperties.D)

    def test_keeps_allpages_listmode(self, pdf_with_allpages_listmode: Pdf):
        """/ListMode /AllPages is compliant and kept."""
        result = sanitize_optional_content(pdf_with_allpages_listmode)

        assert result["list_mode_fixed"] == 0
        d = pdf_with_allpages_listmode.Root.OCProperties.D
        assert str(d.ListMode) == "/AllPages"

    def test_fixes_listmode_in_alternate_configs(
        self, pdf_with_listmode_in_configs: Pdf
    ):
        """Fixes /ListMode in alternate configs, keeps /AllPages."""
        result = sanitize_optional_content(pdf_with_listmode_in_configs)

        assert result["list_mode_fixed"] == 1
        configs = pdf_with_listmode_in_configs.Root.OCProperties.Configs
        assert "/ListMode" not in configs[0]
        assert str(configs[1].ListMode) == "/AllPages"


class TestBaseState:
    """Tests for /BaseState validation (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_basestate_off(self) -> Generator[Pdf, None, None]:
        """PDF with /BaseState /OFF in default config."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", BaseState=Name.OFF),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_basestate_on(self) -> Generator[Pdf, None, None]:
        """PDF with /BaseState /ON in default config (compliant)."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", BaseState=Name.ON),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_basestate_off_in_configs(self) -> Generator[Pdf, None, None]:
        """PDF with /BaseState /OFF in alternate configs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
            Configs=Array(
                [
                    Dictionary(Name="Alt1", BaseState=Name.OFF),
                    Dictionary(Name="Alt2", BaseState=Name.ON),
                ]
            ),
        )

        yield pdf

    def test_fixes_basestate_off(self, pdf_with_basestate_off: Pdf):
        """/BaseState /OFF is corrected to /ON."""
        result = sanitize_optional_content(pdf_with_basestate_off)

        assert result["base_state_fixed"] == 1
        d = pdf_with_basestate_off.Root.OCProperties.D
        assert str(d.BaseState) == "/ON"

    def test_keeps_basestate_on(self, pdf_with_basestate_on: Pdf):
        """/BaseState /ON is compliant and kept."""
        result = sanitize_optional_content(pdf_with_basestate_on)

        assert result["base_state_fixed"] == 0
        d = pdf_with_basestate_on.Root.OCProperties.D
        assert str(d.BaseState) == "/ON"

    def test_fixes_basestate_in_alternate_configs(
        self, pdf_with_basestate_off_in_configs: Pdf
    ):
        """Fixes /BaseState /OFF in alternate configs, keeps /ON."""
        result = sanitize_optional_content(pdf_with_basestate_off_in_configs)

        assert result["base_state_fixed"] == 1
        configs = pdf_with_basestate_off_in_configs.Root.OCProperties.Configs
        assert str(configs[0].BaseState) == "/ON"
        assert str(configs[1].BaseState) == "/ON"


class TestConfigNames:
    """Tests for /Name on alternate configs (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_configs_missing_names(self) -> Generator[Pdf, None, None]:
        """PDF with alternate configs missing /Name entries."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
            Configs=Array(
                [
                    Dictionary(),
                    Dictionary(Name="HasName"),
                    Dictionary(),
                ]
            ),
        )

        yield pdf

    def test_adds_names_to_configs_without_name(
        self, pdf_with_configs_missing_names: Pdf
    ):
        """Adds /Name to alternate configs that are missing it."""
        result = sanitize_optional_content(pdf_with_configs_missing_names)

        assert result["config_names_added"] == 2
        configs = pdf_with_configs_missing_names.Root.OCProperties.Configs
        assert str(configs[0].Name) == "Config0"
        assert str(configs[1].Name) == "HasName"
        assert str(configs[2].Name) == "Config2"

    def test_makes_duplicate_config_names_unique(self) -> None:
        """Ensures /D and /Configs names are unique across all configs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="OCConfigName0"),
            Configs=Array(
                [
                    Dictionary(Name="OCConfigName0"),
                    Dictionary(Name="OCConfigName0"),
                ]
            ),
        )

        result = sanitize_optional_content(pdf)

        assert result["config_names_added"] == 2
        d = pdf.Root.OCProperties.D
        configs = pdf.Root.OCProperties.Configs
        assert str(d.Name) == "OCConfigName0"
        assert str(configs[0].Name) == "OCConfigName0_1"
        assert str(configs[1].Name) == "OCConfigName0_2"


class TestMissingOCGs:
    """Tests for OCGs missing from /OCGs array (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_ocg_in_page_not_in_array(self) -> Generator[Pdf, None, None]:
        """PDF with an OCG referenced in page resources but not in /OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        registered_ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Registered", Intent=Name.View)
        )
        unregistered_ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Unregistered", Intent=Name.View)
        )

        # Add unregistered OCG to page resources but not to /OCGs
        page_dict = pdf.pages[0].obj
        page_dict["/Resources"] = Dictionary(
            Properties=Dictionary(OC1=unregistered_ocg)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([registered_ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_all_ocgs_registered(self) -> Generator[Pdf, None, None]:
        """PDF where all OCGs used in pages are in /OCGs array."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Registered", Intent=Name.View)
        )

        page_dict = pdf.pages[0].obj
        page_dict["/Resources"] = Dictionary(Properties=Dictionary(OC1=ocg))

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    def test_adds_missing_ocg_to_array(self, pdf_with_ocg_in_page_not_in_array: Pdf):
        """Adds OCG referenced in page but missing from /OCGs array."""
        result = sanitize_optional_content(pdf_with_ocg_in_page_not_in_array)

        assert result["missing_ocgs_added"] == 1
        ocgs = pdf_with_ocg_in_page_not_in_array.Root.OCProperties.OCGs
        assert len(ocgs) == 2

    def test_no_addition_when_all_registered(self, pdf_with_all_ocgs_registered: Pdf):
        """Does not add OCGs when all are already registered."""
        result = sanitize_optional_content(pdf_with_all_ocgs_registered)

        assert result["missing_ocgs_added"] == 0
        ocgs = pdf_with_all_ocgs_registered.Root.OCProperties.OCGs
        assert len(ocgs) == 1


class TestOCGName:
    """Tests for /Name on individual OCG dictionaries (ISO 19005-2, 6.8)."""

    def test_ocg_missing_name_gets_default(self) -> None:
        """Adds /Name to OCG dictionary when missing."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Intent=Name.View))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["ocg_names_added"] == 1
        ocg = _resolve_indirect(pdf.Root.OCProperties.OCGs[0])
        assert "/Name" in ocg
        assert str(ocg.Name) == "Unnamed OCG"

    def test_ocg_existing_name_preserved(self) -> None:
        """Does not modify OCG dictionary when /Name already exists."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="MyLayer", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["ocg_names_added"] == 0
        ocg = _resolve_indirect(pdf.Root.OCProperties.OCGs[0])
        assert str(ocg.Name) == "MyLayer"

    def test_multiple_ocgs_missing_name(self) -> None:
        """Adds /Name to multiple OCGs that are missing it."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(Dictionary(Type=Name.OCG, Intent=Name.View))
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="HasName", Intent=Name.View)
        )
        ocg3 = pdf.make_indirect(Dictionary(Type=Name.OCG))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2, ocg3]),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["ocg_names_added"] == 2
        ocgs = pdf.Root.OCProperties.OCGs
        assert "/Name" in _resolve_indirect(ocgs[0])
        assert str(_resolve_indirect(ocgs[1]).Name) == "HasName"
        assert "/Name" in _resolve_indirect(ocgs[2])


class TestRBGroups:
    """Tests for /RBGroups validation (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_valid_rbgroups(self) -> Generator[Pdf, None, None]:
        """PDF with /RBGroups referencing only registered OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )
        ocg3 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer3", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2, ocg3]),
            D=Dictionary(
                Name="Default",
                RBGroups=Array([Array([ocg1, ocg2]), Array([ocg3])]),
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_invalid_rbgroup_refs(self) -> Generator[Pdf, None, None]:
        """PDF with /RBGroups containing references to unregistered OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )
        # unregistered OCG - not added to /OCGs array
        unregistered = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Ghost", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(
                Name="Default",
                RBGroups=Array([Array([ocg1, unregistered, ocg2])]),
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_all_invalid_rbgroup(self) -> Generator[Pdf, None, None]:
        """PDF with /RBGroups where all refs in a group are invalid."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        bad1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Bad1", Intent=Name.View)
        )
        bad2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Bad2", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1]),
            D=Dictionary(
                Name="Default",
                RBGroups=Array(
                    [
                        Array([bad1, bad2]),
                        Array([ocg1]),
                    ]
                ),
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_rbgroups_in_configs(self) -> Generator[Pdf, None, None]:
        """PDF with /RBGroups in alternate configs with invalid refs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        unregistered = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Ghost", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1]),
            D=Dictionary(Name="Default"),
            Configs=Array(
                [
                    Dictionary(
                        Name="Alt1",
                        RBGroups=Array([Array([ocg1, unregistered])]),
                    ),
                ]
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_empty_rbgroups_result(self) -> Generator[Pdf, None, None]:
        """PDF where cleaning /RBGroups leaves it completely empty."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        bad1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Bad1", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1]),
            D=Dictionary(
                Name="Default",
                RBGroups=Array([Array([bad1])]),
            ),
        )

        yield pdf

    def test_valid_rbgroups_unchanged(self, pdf_with_valid_rbgroups: Pdf):
        """Valid /RBGroups are not modified."""
        result = sanitize_optional_content(pdf_with_valid_rbgroups)

        assert result["rbgroups_fixed"] == 0
        d = pdf_with_valid_rbgroups.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups) == 2
        assert len(d.RBGroups[0]) == 2
        assert len(d.RBGroups[1]) == 1

    def test_removes_invalid_refs_from_rbgroup(
        self, pdf_with_invalid_rbgroup_refs: Pdf
    ):
        """Removes unregistered OCG references from /RBGroups."""
        result = sanitize_optional_content(pdf_with_invalid_rbgroup_refs)

        assert result["rbgroups_fixed"] == 1
        d = pdf_with_invalid_rbgroup_refs.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups[0]) == 2

    def test_removes_empty_group_after_cleanup(self, pdf_with_all_invalid_rbgroup: Pdf):
        """Removes empty inner arrays from /RBGroups after cleanup."""
        result = sanitize_optional_content(pdf_with_all_invalid_rbgroup)

        assert result["rbgroups_fixed"] == 2
        d = pdf_with_all_invalid_rbgroup.Root.OCProperties.D
        assert "/RBGroups" in d
        # Only the valid group [ocg1] remains
        assert len(d.RBGroups) == 1

    def test_fixes_rbgroups_in_alternate_configs(
        self, pdf_with_rbgroups_in_configs: Pdf
    ):
        """Removes invalid OCG references from /RBGroups in alternate configs."""
        result = sanitize_optional_content(pdf_with_rbgroups_in_configs)

        assert result["rbgroups_fixed"] == 1
        config = pdf_with_rbgroups_in_configs.Root.OCProperties.Configs[0]
        assert "/RBGroups" in config
        assert len(config.RBGroups[0]) == 1

    def test_removes_rbgroups_when_all_empty(self, pdf_with_empty_rbgroups_result: Pdf):
        """Removes /RBGroups entirely when all groups become empty."""
        result = sanitize_optional_content(pdf_with_empty_rbgroups_result)

        assert result["rbgroups_fixed"] == 1
        d = pdf_with_empty_rbgroups_result.Root.OCProperties.D
        assert "/RBGroups" not in d


class TestOrderArray:
    """Tests for /Order array validation (ISO 19005-2, 6.8)."""

    @pytest.fixture
    def pdf_with_no_order(self) -> Generator[Pdf, None, None]:
        """PDF with /D config that has no /Order array."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(Name="Default"),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_complete_order(self) -> Generator[Pdf, None, None]:
        """PDF with /Order array listing all OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(Name="Default", Order=Array([ocg1, ocg2])),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_partial_order(self) -> Generator[Pdf, None, None]:
        """PDF with /Order array missing some OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )
        ocg3 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer3", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2, ocg3]),
            D=Dictionary(Name="Default", Order=Array([ocg1])),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_nested_order(self) -> Generator[Pdf, None, None]:
        """PDF with /Order array with nested sub-arrays."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )
        ocg3 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer3", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2, ocg3]),
            D=Dictionary(
                Name="Default",
                Order=Array([ocg1, Array([ocg2])]),
            ),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_order_in_alt_config(self) -> Generator[Pdf, None, None]:
        """PDF with /Order in alternate config missing an OCG."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(Name="Default", Order=Array([ocg1, ocg2])),
            Configs=Array(
                [
                    Dictionary(Name="Alt1", Order=Array([ocg1])),
                ]
            ),
        )

        yield pdf

    def test_creates_order_when_missing(self, pdf_with_no_order: Pdf):
        """Creates /Order array with all OCGs when missing from /D."""
        result = sanitize_optional_content(pdf_with_no_order)

        assert result["order_ocgs_added"] == 2
        d = pdf_with_no_order.Root.OCProperties.D
        assert "/Order" in d
        assert len(d.Order) == 2

    def test_complete_order_unchanged(self, pdf_with_complete_order: Pdf):
        """Complete /Order array is not modified."""
        result = sanitize_optional_content(pdf_with_complete_order)

        assert result["order_ocgs_added"] == 0
        d = pdf_with_complete_order.Root.OCProperties.D
        assert len(d.Order) == 2

    def test_adds_missing_ocgs_to_partial_order(self, pdf_with_partial_order: Pdf):
        """Appends missing OCGs to existing /Order array."""
        result = sanitize_optional_content(pdf_with_partial_order)

        assert result["order_ocgs_added"] == 2
        d = pdf_with_partial_order.Root.OCProperties.D
        assert len(d.Order) == 3

    def test_handles_nested_order(self, pdf_with_nested_order: Pdf):
        """Recognises OCGs in nested /Order sub-arrays and only adds missing."""
        result = sanitize_optional_content(pdf_with_nested_order)

        assert result["order_ocgs_added"] == 1
        d = pdf_with_nested_order.Root.OCProperties.D
        # ocg1 at top, [ocg2] nested, ocg3 appended
        assert len(d.Order) == 3

    def test_fixes_order_in_alternate_config(self, pdf_with_order_in_alt_config: Pdf):
        """Adds missing OCGs to /Order in alternate configs."""
        result = sanitize_optional_content(pdf_with_order_in_alt_config)

        assert result["order_ocgs_added"] == 1
        config = pdf_with_order_in_alt_config.Root.OCProperties.Configs[0]
        assert len(config.Order) == 2


class TestMissingDConfig:
    """Tests for /D default configuration creation (ISO 19005-2, 6.9)."""

    @pytest.fixture
    def pdf_with_ocprops_no_d(self) -> Generator[Pdf, None, None]:
        """PDF with /OCProperties but no /D config."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )

        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
        )

        yield pdf

    @pytest.fixture
    def pdf_with_ocprops_no_d_no_ocgs(self) -> Generator[Pdf, None, None]:
        """PDF with /OCProperties without /D and without /OCGs."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        pdf.Root.OCProperties = Dictionary()

        yield pdf

    @pytest.fixture
    def pdf_with_d_already_present(self) -> Generator[Pdf, None, None]:
        """PDF with /OCProperties that already has a /D config."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Existing"),
        )

        yield pdf

    def test_creates_d_when_missing(self, pdf_with_ocprops_no_d: Pdf):
        """Creates /D config when missing from /OCProperties."""
        result = sanitize_optional_content(pdf_with_ocprops_no_d)

        assert result["d_created"] is True
        oc_props = pdf_with_ocprops_no_d.Root.OCProperties
        assert "/D" in oc_props
        d = oc_props.D
        assert str(d.Name) == "Default"
        assert str(d.BaseState) == "/ON"

    def test_d_has_order_with_all_ocgs(self, pdf_with_ocprops_no_d: Pdf):
        """Created /D config has /Order referencing all OCGs."""
        sanitize_optional_content(pdf_with_ocprops_no_d)

        d = pdf_with_ocprops_no_d.Root.OCProperties.D
        assert "/Order" in d
        assert len(d.Order) == 2

    def test_d_created_without_ocgs(self, pdf_with_ocprops_no_d_no_ocgs: Pdf):
        """Creates /D config even when /OCGs is absent."""
        result = sanitize_optional_content(pdf_with_ocprops_no_d_no_ocgs)

        assert result["d_created"] is True
        oc_props = pdf_with_ocprops_no_d_no_ocgs.Root.OCProperties
        assert "/D" in oc_props
        d = oc_props.D
        assert str(d.Name) == "Default"
        assert str(d.BaseState) == "/ON"
        # No /Order since there are no OCGs
        assert "/Order" not in d

    def test_no_creation_when_d_exists(self, pdf_with_d_already_present: Pdf):
        """Does not create /D when it already exists."""
        result = sanitize_optional_content(pdf_with_d_already_present)

        assert result["d_created"] is False
        d = pdf_with_d_already_present.Root.OCProperties.D
        assert str(d.Name) == "Existing"

    def test_d_created_with_configs(self):
        """Creates /D when missing, even if /Configs exist."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page)

        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            Configs=Array([Dictionary(Name="Alt1")]),
        )

        result = sanitize_optional_content(pdf)

        assert result["d_created"] is True
        assert "/D" in pdf.Root.OCProperties
        d = pdf.Root.OCProperties.D
        assert str(d.Name) == "Default"
        assert str(d.BaseState) == "/ON"
        assert "/Order" in d
        assert len(d.Order) == 1
