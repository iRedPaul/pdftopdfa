# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Optional Content (Layers) sanitization for PDF/A compliance."""

from collections.abc import Generator
from io import BytesIO

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

import pdftopdfa.sanitizers.optional_content as optional_content
from pdftopdfa.sanitizers.optional_content import (
    _default_optional_content_visibility,
    sanitize_optional_content,
)
from pdftopdfa.utils import resolve_indirect as _resolve_indirect


def _visibility_pdf() -> tuple[Pdf, Dictionary, Dictionary]:
    pdf = new_pdf()
    pdf.pages.append(
        pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 100, 100])))
    )
    on = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="On"))
    off = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Off"))
    pdf.Root.OCProperties = Dictionary(
        OCGs=Array([on, off]),
        D=Dictionary(Name="Default", BaseState=Name.ON, OFF=Array([off])),
    )
    return pdf, on, off


class TestDefaultVisibility:
    """Tests for the read-only default OCG/OCMD visibility evaluator."""

    def test_without_ocproperties_optional_content_is_ignored(self) -> None:
        pdf = new_pdf()

        visibility = _default_optional_content_visibility(pdf)

        assert visibility.is_visible(None) is True

    def test_evaluates_registered_ocg_states_without_mutation(self) -> None:
        pdf, on, off = _visibility_pdf()
        original_root_keys = set(pdf.Root.keys())
        original_config_keys = set(pdf.Root.OCProperties.D.keys())
        original_object_count = len(pdf.objects)

        visibility = _default_optional_content_visibility(pdf)

        assert visibility.is_visible(on) is True
        assert visibility.is_visible(off) is False
        assert set(pdf.Root.keys()) == original_root_keys
        assert set(pdf.Root.OCProperties.D.keys()) == original_config_keys
        assert len(pdf.objects) == original_object_count

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            (Name.AnyOn, True),
            (Name.AllOn, False),
            (Name.AnyOff, True),
            (Name.AllOff, False),
        ],
    )
    def test_evaluates_all_membership_policies(
        self,
        policy: Name,
        expected: bool,
    ) -> None:
        pdf, on, off = _visibility_pdf()
        membership = Dictionary(
            Type=Name.OCMD,
            OCGs=Array([on, off]),
            P=policy,
        )

        assert (
            _default_optional_content_visibility(pdf).is_visible(membership) is expected
        )

    def test_empty_membership_has_no_visibility_effect(self) -> None:
        pdf, _on, _off = _visibility_pdf()

        assert _default_optional_content_visibility(pdf).is_visible(
            Dictionary(Type=Name.OCMD, OCGs=Array(), P=Name.AllOn)
        )

    def test_nested_ve_takes_precedence_over_policy(self) -> None:
        pdf, on, off = _visibility_pdf()
        membership = Dictionary(
            Type=Name.OCMD,
            OCGs=Array([on, off]),
            P=Name.AllOff,
            VE=Array([Name.And, on, Array([Name.Not, off])]),
        )

        assert _default_optional_content_visibility(pdf).is_visible(membership)

    def test_valid_ve_takes_precedence_over_malformed_policy(self) -> None:
        pdf, on, _off = _visibility_pdf()
        membership = Dictionary(
            Type=Name.OCMD,
            OCGs=Array(),
            P=Name("/Unknown"),
            VE=Array([Name.And, on]),
        )

        assert _default_optional_content_visibility(pdf).is_visible(membership)

    def test_null_membership_entries_are_ignored(self) -> None:
        pdf, on, _off = _visibility_pdf()
        visibility = _default_optional_content_visibility(pdf)

        assert visibility.is_visible(
            Dictionary(Type=Name.OCMD, OCGs=Array([None, on]), P=Name.AllOn)
        )
        assert visibility.is_visible(
            Dictionary(Type=Name.OCMD, OCGs=Array([None]), P=Name.AllOff)
        )

    def test_nonmatching_group_intent_has_no_visibility_effect(self) -> None:
        pdf, on, off = _visibility_pdf()
        off.Intent = Name.Design
        visibility = _default_optional_content_visibility(pdf)

        assert visibility.is_visible(off)
        assert visibility.is_visible(
            Dictionary(
                Type=Name.OCMD,
                OCGs=Array([off, on]),
                P=Name.AllOn,
            )
        )
        assert visibility.is_visible(
            Dictionary(Type=Name.OCMD, VE=Array([Name.Not, off]))
        )

    @pytest.mark.parametrize(
        ("configuration_intent", "group_intent"),
        [(Name.View, Name.All), (Name.All, Name.View)],
    )
    def test_all_intent_matches_symmetrically(
        self,
        configuration_intent: Name,
        group_intent: Name,
    ) -> None:
        pdf, _on, off = _visibility_pdf()
        pdf.Root.OCProperties.D.Intent = configuration_intent
        off.Intent = group_intent

        assert _default_optional_content_visibility(pdf).is_visible(off) is False

    def test_direct_memberships_do_not_reuse_stale_cached_results(self) -> None:
        pdf, on, off = _visibility_pdf()
        visibility = _default_optional_content_visibility(pdf)

        for index in range(200):
            expected = index % 2 == 0
            membership = Dictionary(
                Type=Name.OCMD,
                OCGs=Array([on if expected else off]),
            )
            assert visibility.is_visible(membership) is expected

    def test_shared_large_indirect_ve_is_evaluated_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 520)
        pdf, on, _off = _visibility_pdf()
        expression = pdf.make_indirect(Array([Name.And, *([on] * 512)]))
        first = pdf.make_indirect(Dictionary(Type=Name.OCMD, VE=expression))
        second = pdf.make_indirect(Dictionary(Type=Name.OCMD, VE=expression))
        visibility = _default_optional_content_visibility(pdf)

        assert visibility.is_visible(first)
        assert visibility.is_visible(second)

    def test_membership_work_budget_is_global(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 6)
        pdf, on, _off = _visibility_pdf()
        visibility = _default_optional_content_visibility(pdf)

        for _ in range(3):
            assert visibility.is_visible(Dictionary(Type=Name.OCMD, OCGs=on))
        with pytest.raises(ValueError, match="evaluation limit exceeded"):
            visibility.is_visible(Dictionary(Type=Name.OCMD, OCGs=on))

    def test_intent_elements_use_evaluator_work_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 3)
        pdf = new_pdf()
        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Intent=Array([Name.View, Name.View]))
        )
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Intent=Array([Name.View, Name.View])),
        )

        with pytest.raises(ValueError, match="evaluation limit exceeded"):
            _default_optional_content_visibility(pdf)

    @pytest.mark.parametrize(
        "membership",
        [
            Dictionary(Type=Name.OCMD, P=Name("/Unknown")),
            Dictionary(Type=Name.OCMD, VE=Array([Name.Not])),
            Dictionary(Type=Name.OCMD, VE=Array([Name("/Unknown"), True])),
        ],
    )
    def test_rejects_malformed_membership(self, membership: Dictionary) -> None:
        pdf, _on, _off = _visibility_pdf()

        with pytest.raises(ValueError, match="Optional-content"):
            _default_optional_content_visibility(pdf).is_visible(membership)

    def test_rejects_unregistered_ocg(self) -> None:
        pdf, _on, _off = _visibility_pdf()
        unregistered = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Missing"))

        with pytest.raises(ValueError, match="unregistered"):
            _default_optional_content_visibility(pdf).is_visible(unregistered)

    def test_rejects_cyclic_ve(self) -> None:
        pdf, _on, _off = _visibility_pdf()
        expression = pdf.make_indirect(Array([Name.Not]))
        expression.append(expression)
        membership = Dictionary(Type=Name.OCMD, VE=expression)

        with pytest.raises(ValueError, match="cyclic"):
            _default_optional_content_visibility(pdf).is_visible(membership)

    def test_rejects_excessively_deep_ve(self) -> None:
        pdf, on, _off = _visibility_pdf()
        expression: object = on
        for _ in range(65):
            expression = Array([Name.Not, expression])

        with pytest.raises(ValueError, match="deeply nested"):
            _default_optional_content_visibility(pdf).is_visible(
                Dictionary(Type=Name.OCMD, VE=expression)
            )

    def test_rejects_ve_work_over_budget(self) -> None:
        pdf, on, _off = _visibility_pdf()
        expression = Array([Name.Or, *([on] * 8_192)])

        with pytest.raises(ValueError, match="limit exceeded"):
            _default_optional_content_visibility(pdf).is_visible(
                Dictionary(Type=Name.OCMD, VE=expression)
            )

    def test_requires_sanitized_default_basestate(self) -> None:
        pdf, _on, _off = _visibility_pdf()
        pdf.Root.OCProperties.D.BaseState = Name.OFF

        with pytest.raises(ValueError, match="BaseState"):
            _default_optional_content_visibility(pdf)


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
    """Tests for preserving and validating optional-content intents."""

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

    def test_preserves_design_intent(self, pdf_with_design_intent: Pdf):
        """An OCG's /Design intent is not changed to another semantic intent."""
        result = sanitize_optional_content(pdf_with_design_intent)

        assert result["ocgs_processed"] == 1

        ocg = pdf_with_design_intent.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert ocg.Intent == Name.Design

    def test_preserves_intent_array_with_view(self, pdf_with_intent_array: Pdf):
        """A mixed OCG intent array remains intact."""
        result = sanitize_optional_content(pdf_with_intent_array)

        assert result["ocgs_processed"] == 1

        ocg = pdf_with_intent_array.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert list(ocg.Intent) == [Name.View, Name.Design]

    def test_preserves_design_only_intent_array(self, pdf_with_design_only_array: Pdf):
        """A design-only OCG intent array remains intact."""
        sanitize_optional_content(pdf_with_design_only_array)

        ocg = pdf_with_design_only_array.Root.OCProperties.OCGs[0]
        ocg = _resolve_indirect(ocg)
        assert list(ocg.Intent) == [Name.Design]

    @pytest.mark.parametrize(
        "intent",
        [Name.Design, Array([Name.View, Name.Design]), Array([Name.Design])],
    )
    def test_preserves_non_view_default_intent(self, intent: object) -> None:
        """ISO 19005-2, 6.9 places no intent restriction on configurations."""
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", Intent=intent),
        )

        result = sanitize_optional_content(pdf)

        assert result["default_intent_fixed"] == 0
        default_intent = pdf.Root.OCProperties.D.Intent
        if isinstance(intent, Array):
            assert list(default_intent) == list(intent)
        else:
            assert default_intent == intent

    @pytest.mark.parametrize(
        "intent",
        [Array([Name.View]), Array([Name.View, Name.View])],
    )
    def test_canonicalizes_view_only_default_intent_array(
        self,
        intent: Array,
    ) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer", Intent=Name.Design)
        )
        alternate = Dictionary(Name="Alternate", Intent=Name.Design)
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", Intent=intent),
            Configs=Array([alternate]),
        )

        result = sanitize_optional_content(pdf)

        assert result["default_intent_fixed"] == 1
        assert pdf.Root.OCProperties.D.Intent == Name.View
        assert ocg.Intent == Name.Design
        assert alternate.Intent == Name.Design

    @pytest.mark.parametrize("explicit", [False, True])
    def test_keeps_default_view_intent_default(self, explicit: bool) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        default = Dictionary(Name="Default")
        if explicit:
            default.Intent = Name.View
        pdf.Root.OCProperties = Dictionary(OCGs=Array([ocg]), D=default)

        result = sanitize_optional_content(pdf)

        assert result["default_intent_fixed"] == 0
        if explicit:
            assert default.Intent == Name.View
        else:
            assert "/Intent" not in default

    def test_shared_intent_array_has_normal_bounded_behavior(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 8)
        pdf = new_pdf()
        shared = pdf.make_indirect(Array([Name.View]))
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer", Intent=shared))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default", Intent=shared),
        )

        result = sanitize_optional_content(pdf)

        assert result["default_intent_fixed"] == 1
        assert pdf.Root.OCProperties.D.Intent == Name.View
        assert list(ocg.Intent) == [Name.View]


class TestNoChangesNeeded:
    """Tests for PDFs that don't need optional content changes."""

    def test_pdf_without_ocproperties(self, sample_pdf_obj: Pdf):
        """PDF without OCProperties returns zero counts."""
        result = sanitize_optional_content(sample_pdf_obj)

        assert result["as_entries_removed"] == 0
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
        assert result["ocgs_processed"] == 4
        ocgs = pdf_with_multiple_ocgs.Root.OCProperties.OCGs
        assert _resolve_indirect(ocgs[1]).Intent == Name.Design
        assert list(_resolve_indirect(ocgs[3]).Intent) == [Name.View, Name.Design]


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
        assert "oc_d_created" in result
        assert "oc_d_name_added" in result
        assert "oc_base_state_fixed" in result
        assert "oc_default_intent_fixed" in result
        assert "oc_config_names_added" in result
        assert "oc_missing_ocgs_added" in result
        assert "oc_ocg_names_added" in result
        assert "oc_order_ocgs_added" in result

    def test_sanitize_for_pdfa_reports_no_dead_oc_metrics(self, sample_pdf_obj: Pdf):
        """Repairs the sanitizer no longer performs are not reported."""
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        result = sanitize_for_pdfa(sample_pdf_obj, "3b")

        assert "oc_intents_fixed" not in result
        assert "oc_list_mode_fixed" not in result
        assert "oc_rbgroups_fixed" not in result

    def test_sanitize_for_pdfa_reports_malformed_oc_as_conversion_error(self):
        """A defective document is not reported as an unexpected error."""
        from pdftopdfa.exceptions import ConversionError
        from pdftopdfa.sanitizers import sanitize_for_pdfa

        pdf = new_pdf()
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=Dictionary(Name="Default", BaseState=Name.Unchanged),
        )

        with pytest.raises(
            ConversionError,
            match="Optional content cannot be made PDF/A compliant",
        ):
            sanitize_for_pdfa(pdf, "3b")


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

    def test_preserves_visible_pages_listmode(
        self, pdf_with_visible_pages_listmode: Pdf
    ):
        """Preserves a valid default configuration presentation preference."""
        sanitize_optional_content(pdf_with_visible_pages_listmode)

        assert (
            pdf_with_visible_pages_listmode.Root.OCProperties.D.ListMode
            == Name.VisiblePages
        )

    def test_keeps_allpages_listmode(self, pdf_with_allpages_listmode: Pdf):
        """/ListMode /AllPages is compliant and kept."""
        sanitize_optional_content(pdf_with_allpages_listmode)

        d = pdf_with_allpages_listmode.Root.OCProperties.D
        assert str(d.ListMode) == "/AllPages"

    def test_preserves_listmode_in_alternate_configs(
        self, pdf_with_listmode_in_configs: Pdf
    ):
        """Preserves valid presentation preferences in alternate configs."""
        sanitize_optional_content(pdf_with_listmode_in_configs)

        configs = pdf_with_listmode_in_configs.Root.OCProperties.Configs
        assert configs[0].ListMode == Name.VisiblePages
        assert configs[1].ListMode == Name.AllPages


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
        """/BaseState /OFF is normalized without turning its OCG on."""
        result = sanitize_optional_content(pdf_with_basestate_off)

        assert result["base_state_fixed"] == 1
        d = pdf_with_basestate_off.Root.OCProperties.D
        assert str(d.BaseState) == "/ON"
        assert "/ON" not in d
        assert len(d.OFF) == 1
        assert (
            _default_optional_content_visibility(pdf_with_basestate_off).is_visible(
                pdf_with_basestate_off.Root.OCProperties.OCGs[0]
            )
            is False
        )

    def test_keeps_basestate_on(self, pdf_with_basestate_on: Pdf):
        """/BaseState /ON is compliant and kept."""
        result = sanitize_optional_content(pdf_with_basestate_on)

        assert result["base_state_fixed"] == 0
        d = pdf_with_basestate_on.Root.OCProperties.D
        assert str(d.BaseState) == "/ON"

    def test_preserves_basestate_in_alternate_configs(
        self, pdf_with_basestate_off_in_configs: Pdf
    ):
        """Alternate configurations are not semantically rewritten."""
        result = sanitize_optional_content(pdf_with_basestate_off_in_configs)

        assert result["base_state_fixed"] == 0
        configs = pdf_with_basestate_off_in_configs.Root.OCProperties.Configs
        assert configs[0].BaseState == Name.OFF
        assert configs[1].BaseState == Name.ON

    def test_basestate_off_normalization_respects_on_then_off_precedence(
        self,
    ) -> None:
        pdf = new_pdf()
        on = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="On"))
        overlap = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Overlap"))
        off = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Off"))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([on, overlap, off]),
            D=Dictionary(
                Name="Default",
                BaseState=Name.OFF,
                ON=Array([on, overlap]),
                OFF=Array([overlap]),
            ),
        )

        result = sanitize_optional_content(pdf)

        assert result["base_state_fixed"] == 1
        d = pdf.Root.OCProperties.D
        assert d.BaseState == Name.ON
        assert "/ON" not in d
        assert {item.objgen for item in d.OFF} == {overlap.objgen, off.objgen}
        visibility = _default_optional_content_visibility(pdf)
        assert visibility.is_visible(on)
        assert visibility.is_visible(overlap) is False
        assert visibility.is_visible(off) is False

    def test_basestate_on_off_array_wins_over_on_array(self) -> None:
        pdf, on, _off = _visibility_pdf()
        pdf.Root.OCProperties.D.ON = Array([on])
        pdf.Root.OCProperties.D.OFF = Array([on])

        result = sanitize_optional_content(pdf)

        assert result["base_state_fixed"] == 0
        assert _default_optional_content_visibility(pdf).is_visible(on) is False

    def test_rejects_default_unchanged_atomically(self) -> None:
        pdf, on, _off = _visibility_pdf()
        d = pdf.Root.OCProperties.D
        d.BaseState = Name.Unchanged
        d.AS = Array([Dictionary(Event=Name.View, OCGs=Array([on]))])
        original_keys = set(d.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="cannot be normalized safely"):
            sanitize_optional_content(pdf)

        assert set(d.keys()) == original_keys
        assert d.BaseState == Name.Unchanged
        assert "/AS" in d
        assert len(pdf.objects) == original_object_count


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
        saved = BytesIO()
        pdf_with_configs_missing_names.save(saved)
        saved.seek(0)
        with Pdf.open(saved) as reopened:
            names = [str(config.Name) for config in reopened.Root.OCProperties.Configs]
            assert names == ["Config0", "HasName", "Config2"]
            assert len(names) == len(set(names))

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

    def test_collects_ocg_from_image_xobject_oc(self) -> None:
        """An Image XObject /OC membership is registered."""
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Intent=Name.Design))
        image = pdf.make_stream(b"\x00")
        image["/Type"] = Name.XObject
        image["/Subtype"] = Name.Image
        image["/Width"] = 1
        image["/Height"] = 1
        image["/ColorSpace"] = Name.DeviceGray
        image["/BitsPerComponent"] = 8
        image["/OC"] = ocg
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(XObject=Dictionary(Im=image)),
            )
        )
        pdf.pages.append(page)
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 1
        assert pdf.Root.OCProperties.OCGs[0].objgen == ocg.objgen
        assert str(ocg.Name) == "Unnamed OCG"
        assert ocg.Intent == Name.Design

    def test_collects_ocgs_from_annotation_ocmd_policy_and_ve(self) -> None:
        """Annotation OCMD memberships register policy and /VE OCGs."""
        pdf = new_pdf()
        policy_ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Policy"))
        expression_ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Expression"))
        membership = pdf.make_indirect(
            Dictionary(
                Type=Name.OCMD,
                OCGs=Array([policy_ocg]),
                VE=Array([Name.Not, expression_ocg]),
            )
        )
        annotation = pdf.make_indirect(
            Dictionary(Type=Name.Annot, Subtype=Name.Text, Rect=Array([0, 0, 10, 10]))
        )
        annotation["/OC"] = membership
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Annots=Array([annotation]),
            )
        )
        pdf.pages.append(page)
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 2
        assert {item.objgen for item in pdf.Root.OCProperties.OCGs} == {
            policy_ocg.objgen,
            expression_ocg.objgen,
        }

    def test_collects_ocg_through_cyclic_nested_form_resources(self) -> None:
        """Nested Form resource cycles terminate and retain their /OC OCGs."""
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Nested form"))
        outer = pdf.make_stream(b"/Inner Do")
        inner = pdf.make_stream(b"/Outer Do")
        for form in (outer, inner):
            form["/Type"] = Name.XObject
            form["/Subtype"] = Name.Form
            form["/BBox"] = Array([0, 0, 10, 10])
        inner["/OC"] = ocg
        outer["/Resources"] = Dictionary(XObject=Dictionary(Inner=inner))
        inner["/Resources"] = Dictionary(XObject=Dictionary(Outer=outer))
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(XObject=Dictionary(Outer=outer)),
            )
        )
        pdf.pages.append(page)
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=Dictionary(Name="Default"),
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 1
        assert pdf.Root.OCProperties.OCGs[0].objgen == ocg.objgen

    def test_collects_ocg_from_default_off_array(self) -> None:
        """Configuration-only OCG references are registered before validation."""
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Initially hidden"))
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=Dictionary(Name="Default", OFF=Array([ocg])),
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 1
        visibility = _default_optional_content_visibility(pdf)
        assert visibility.is_visible(ocg) is False

    def test_repairs_missing_ocgs_array_from_configuration(self) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Configured"))
        pdf.Root.OCProperties = Dictionary(
            D=Dictionary(Name="Default", OFF=Array([ocg]))
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 1
        assert [item.objgen for item in pdf.Root.OCProperties.OCGs] == [ocg.objgen]
        assert _default_optional_content_visibility(pdf).is_visible(ocg) is False

    def test_registers_all_configuration_reference_locations(self) -> None:
        pdf = new_pdf()
        groups = [
            pdf.make_indirect(Dictionary(Type=Name.OCG, Name=f"Layer {index}"))
            for index in range(6)
        ]
        alternate = Dictionary(
            Name="Alternate",
            Intent=Name.Design,
            Locked=Array([groups[5]]),
        )
        pdf.Root.OCProperties = Dictionary(
            D=Dictionary(
                Name="Default",
                ON=Array([groups[0]]),
                OFF=Array([groups[1]]),
                Order=Array([groups[2]]),
                RBGroups=Array([Array([groups[3], groups[4]])]),
            ),
            Configs=Array([alternate]),
        )

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 6
        assert {item.objgen for item in pdf.Root.OCProperties.OCGs} == {
            item.objgen for item in groups
        }
        assert alternate.Intent == Name.Design
        assert [item.objgen for item in alternate.Locked] == [groups[5].objgen]

    def test_collects_ocgs_from_annotation_appearance_streams(self) -> None:
        pdf = new_pdf()
        normal_ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Normal"))
        state_ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="State"))
        normal = pdf.make_stream(b"")
        normal["/OC"] = normal_ocg
        state = pdf.make_stream(b"")
        state["/OC"] = state_ocg
        annotation = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 10, 10]),
                AP=Dictionary(
                    N=normal,
                    R=Dictionary(Active=state),
                ),
            )
        )
        pdf.pages.append(
            pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 100, 100]),
                    Annots=Array([annotation]),
                )
            )
        )
        pdf.Root.OCProperties = Dictionary(D=Dictionary(Name="Default"))

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 2
        assert {item.objgen for item in pdf.Root.OCProperties.OCGs} == {
            normal_ocg.objgen,
            state_ocg.objgen,
        }

    def test_registers_200_direct_ocmd_resource_memberships(self) -> None:
        pdf = new_pdf()
        properties = Dictionary()
        expected: set[tuple[int, int]] = set()
        for index in range(200):
            ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=f"Layer {index}"))
            expected.add(ocg.objgen)
            properties[f"/MC{index}"] = Dictionary(
                Type=Name.OCMD,
                OCGs=Array([ocg]),
            )
        pdf.pages.append(
            pikepdf.Page(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 100, 100]),
                    Resources=Dictionary(Properties=properties),
                )
            )
        )
        pdf.Root.OCProperties = Dictionary(D=Dictionary(Name="Default"))

        result = sanitize_optional_content(pdf)

        assert result["missing_ocgs_added"] == 200
        assert {item.objgen for item in pdf.Root.OCProperties.OCGs} == expected

    def test_missing_ocproperties_with_controls_fails_atomically(self) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        properties = Dictionary(MC0=ocg)
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(Properties=properties),
            )
        )
        pdf.pages.append(page)
        original_root_keys = set(pdf.Root.keys())
        original_page_keys = set(page.obj.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="require a catalog /OCProperties"):
            sanitize_optional_content(pdf)

        assert set(pdf.Root.keys()) == original_root_keys
        assert "/OCProperties" not in pdf.Root
        assert set(page.obj.keys()) == original_page_keys
        assert page.obj.Resources.Properties.MC0.objgen == ocg.objgen
        assert len(pdf.objects) == original_object_count


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
        sanitize_optional_content(pdf_with_valid_rbgroups)

        d = pdf_with_valid_rbgroups.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups) == 2
        assert len(d.RBGroups[0]) == 2
        assert len(d.RBGroups[1]) == 1

    def test_registers_missing_refs_from_rbgroup(
        self, pdf_with_invalid_rbgroup_refs: Pdf
    ):
        """Registers OCGs referenced only from /RBGroups without rewriting it."""
        result = sanitize_optional_content(pdf_with_invalid_rbgroup_refs)

        assert result["missing_ocgs_added"] == 1
        d = pdf_with_invalid_rbgroup_refs.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups[0]) == 3
        assert len(pdf_with_invalid_rbgroup_refs.Root.OCProperties.OCGs) == 3

    def test_registers_every_group_in_rbgroup(self, pdf_with_all_invalid_rbgroup: Pdf):
        """Every valid OCG reference in /RBGroups becomes registered."""
        result = sanitize_optional_content(pdf_with_all_invalid_rbgroup)

        assert result["missing_ocgs_added"] == 2
        d = pdf_with_all_invalid_rbgroup.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups) == 2
        assert len(d.RBGroups[0]) == 2

    def test_registers_rbgroups_in_alternate_configs(
        self, pdf_with_rbgroups_in_configs: Pdf
    ):
        """Registers alternate-config OCGs without rewriting /RBGroups."""
        result = sanitize_optional_content(pdf_with_rbgroups_in_configs)

        assert result["missing_ocgs_added"] == 1
        config = pdf_with_rbgroups_in_configs.Root.OCProperties.Configs[0]
        assert "/RBGroups" in config
        assert len(config.RBGroups[0]) == 2

    def test_preserves_rbgroups_when_references_can_be_registered(
        self, pdf_with_empty_rbgroups_result: Pdf
    ):
        """A valid group is registered rather than silently discarded."""
        result = sanitize_optional_content(pdf_with_empty_rbgroups_result)

        assert result["missing_ocgs_added"] == 1
        d = pdf_with_empty_rbgroups_result.Root.OCProperties.D
        assert "/RBGroups" in d
        assert len(d.RBGroups[0]) == 1


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

    def test_order_beyond_python_recursion_limit(self):
        """Recognises an OCG in an exceptionally deep /Order array."""
        pdf = new_pdf()
        pdf.pages.append(
            pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
        )
        ocg1 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer1", Intent=Name.View)
        )
        ocg2 = pdf.make_indirect(
            Dictionary(Type=Name.OCG, Name="Layer2", Intent=Name.View)
        )
        order = Array([ocg1])
        for _ in range(1200):
            order = Array([order])
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg1, ocg2]),
            D=Dictionary(Name="Default", Order=order),
        )

        result = sanitize_optional_content(pdf)
        assert result["order_ocgs_added"] == 1
        assert len(pdf.Root.OCProperties.D.Order) == 2

    def test_rejects_indirect_order_cycle_atomically(self) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        order = pdf.make_indirect(Array())
        order.append(order)
        default = Dictionary(
            BaseState=Name.OFF,
            AS=Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))]),
            Order=order,
        )
        pdf.Root.OCProperties = Dictionary(OCGs=Array([ocg]), D=default)
        original_keys = set(default.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="Order is cyclic"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_keys
        assert default.BaseState == Name.OFF
        assert "/AS" in default
        assert len(pdf.objects) == original_object_count

    def test_direct_order_cycle_is_bounded_and_atomic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 32)
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        order = Array()
        order.append(order)
        default = Dictionary(BaseState=Name.OFF, Order=order)
        pdf.Root.OCProperties = Dictionary(OCGs=Array([ocg]), D=default)
        original_keys = set(default.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="collection limit exceeded"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_keys
        assert default.BaseState == Name.OFF
        assert len(pdf.objects) == original_object_count
        del order[0]

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


class TestAtomicPreflight:
    """Validation failures occur before any optional-content mutation."""

    @pytest.mark.parametrize("has_as", [False, True])
    def test_rejects_default_config_alias_atomically(self, has_as: bool) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        shared = Dictionary(Name="Shared", BaseState=Name.ON)
        if has_as:
            shared.AS = Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))])
        shared = pdf.make_indirect(shared)
        properties = Dictionary(
            OCGs=Array([ocg]),
            D=shared,
            Configs=Array([shared]),
        )
        pdf.Root.OCProperties = properties
        original_keys = set(shared.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="configuration is aliased"):
            sanitize_optional_content(pdf)

        assert set(shared.keys()) == original_keys
        assert ("/AS" in shared) is has_as
        assert "/Order" not in shared
        assert len(properties.OCGs) == 1
        assert len(pdf.objects) == original_object_count

    @pytest.mark.parametrize("has_as", [False, True])
    def test_rejects_repeated_alternate_config_alias_atomically(
        self,
        has_as: bool,
    ) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        default = Dictionary(BaseState=Name.OFF)
        alternate = Dictionary(Intent=Name.Design)
        if has_as:
            alternate.AS = Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))])
        alternate = pdf.make_indirect(alternate)
        properties = Dictionary(
            OCGs=Array([ocg]),
            D=default,
            Configs=Array([alternate, alternate]),
        )
        pdf.Root.OCProperties = properties
        original_default_keys = set(default.keys())
        original_alternate_keys = set(alternate.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="configuration is aliased"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_default_keys
        assert default.BaseState == Name.OFF
        assert "/Name" not in default
        assert set(alternate.keys()) == original_alternate_keys
        assert ("/AS" in alternate) is has_as
        assert "/Name" not in alternate
        assert len(properties.OCGs) == 1
        assert len(pdf.objects) == original_object_count

    def test_rejects_direct_config_alias_atomically(self) -> None:
        pdf = new_pdf()
        shared = Dictionary()
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=shared,
            Configs=Array([shared]),
        )
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="configuration is aliased"):
            sanitize_optional_content(pdf)

        assert not shared
        assert len(pdf.objects) == original_object_count

    def test_rejects_repeated_empty_direct_config_alias_atomically(self) -> None:
        pdf = new_pdf()
        shared = Dictionary()
        default = Dictionary(Name="Default")
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array(),
            D=default,
            Configs=Array([shared, shared]),
        )
        original_default_keys = set(default.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="configuration is aliased"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_default_keys
        assert not shared
        assert len(pdf.objects) == original_object_count

    def test_repairs_equal_but_independent_direct_configs(self) -> None:
        """Identical content is not aliasing: renameable configs are repaired."""
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Layer"))
        first = Dictionary(Name="Alternate", BaseState=Name.ON)
        second = Dictionary(Name="Alternate", BaseState=Name.ON)
        pdf.Root.OCProperties = Dictionary(
            OCGs=Array([ocg]),
            D=Dictionary(Name="Default"),
            Configs=Array([first, second]),
        )

        result = sanitize_optional_content(pdf)

        assert result["config_names_added"] == 1
        configs = pdf.Root.OCProperties.Configs
        names = [str(config.Name) for config in configs]
        assert names == ["Alternate", "Alternate_1"]
        assert all("/PdftopdfaAliasProbe0" not in config for config in configs)

    def test_late_malformed_alternate_config_keeps_all_planned_changes(self) -> None:
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG))
        default = Dictionary(
            BaseState=Name.OFF,
            AS=Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))]),
        )
        alternate = Dictionary(ListMode=Name("/Invalid"))
        properties = Dictionary(
            OCGs=Array([ocg]),
            D=default,
            Configs=Array([alternate]),
        )
        pdf.Root.OCProperties = properties
        original_default_keys = set(default.keys())
        original_alternate_keys = set(alternate.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="ListMode"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_default_keys
        assert default.BaseState == Name.OFF
        assert "/AS" in default
        assert "/Name" not in ocg
        assert "/Order" not in default
        assert set(alternate.keys()) == original_alternate_keys
        assert "/Name" not in alternate
        assert len(properties.OCGs) == 1
        assert len(pdf.objects) == original_object_count

    def test_collection_budget_failure_keeps_document_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 8)
        pdf = new_pdf()
        ocg = pdf.make_indirect(Dictionary(Type=Name.OCG))
        resource_properties = Dictionary()
        for index in range(8):
            resource_properties[f"/MC{index}"] = Dictionary(
                Type=Name.OCMD,
                OCGs=Array([ocg]),
            )
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 100, 100]),
                Resources=Dictionary(Properties=resource_properties),
            )
        )
        pdf.pages.append(page)
        default = Dictionary(
            BaseState=Name.OFF,
            AS=Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))]),
        )
        properties = Dictionary(OCGs=Array([ocg]), D=default)
        pdf.Root.OCProperties = properties
        original_default_keys = set(default.keys())
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="collection limit exceeded"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_default_keys
        assert default.BaseState == Name.OFF
        assert "/AS" in default
        assert "/Name" not in ocg
        assert len(properties.OCGs) == 1
        assert len(resource_properties) == 8
        assert len(pdf.objects) == original_object_count

    def test_intent_elements_use_collector_budget_atomically(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 5)
        pdf = new_pdf()
        ocg = pdf.make_indirect(
            Dictionary(
                Type=Name.OCG,
                Intent=Array([Name.View, Name.View, Name.View]),
            )
        )
        default = Dictionary(
            BaseState=Name.OFF,
            Intent=Array([Name.View, Name.View]),
            AS=Array([Dictionary(Event=Name.View, OCGs=Array([ocg]))]),
        )
        properties = Dictionary(OCGs=Array([ocg]), D=default)
        pdf.Root.OCProperties = properties
        original_default_keys = set(default.keys())
        original_intent = list(default.Intent)
        original_object_count = len(pdf.objects)

        with pytest.raises(ValueError, match="collection limit exceeded"):
            sanitize_optional_content(pdf)

        assert set(default.keys()) == original_default_keys
        assert default.BaseState == Name.OFF
        assert list(default.Intent) == original_intent
        assert "/AS" in default
        assert "/Name" not in ocg
        assert len(properties.OCGs) == 1
        assert len(pdf.objects) == original_object_count


class TestResourceBudget:
    """The resource ceiling must scale with the document, not a fixed count."""

    @staticmethod
    def _pdf_with_distinct_page_resources(page_count: int) -> Pdf:
        pdf = new_pdf()
        for _index in range(page_count):
            page = pdf.make_indirect(
                Dictionary(
                    Type=Name.Page,
                    MediaBox=Array([0, 0, 612, 792]),
                    Resources=pdf.make_indirect(Dictionary(ProcSet=Array([Name.PDF]))),
                )
            )
            pdf.pages.append(pikepdf.Page(page))
        return pdf

    def test_large_document_without_optional_content_is_not_rejected(self) -> None:
        """A page count above the item limit is not an optional-content defect."""
        page_count = optional_content._MAX_OPTIONAL_CONTENT_ITEMS + 10
        pdf = self._pdf_with_distinct_page_resources(page_count)
        assert "/OCProperties" not in pdf.Root

        result = sanitize_optional_content(pdf)

        assert result["ocgs_processed"] == 0
        assert "/OCProperties" not in pdf.Root

    def test_budget_still_bounds_the_traversal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ceiling tracks the object count, so it stays finite."""
        pdf = self._pdf_with_distinct_page_resources(4)
        monkeypatch.setattr(optional_content, "_MAX_OPTIONAL_CONTENT_ITEMS", 1)

        assert optional_content._resource_budget(pdf) == len(pdf.objects)
