"""Tests for compliance standard name aliasing.

The bundled EU AI Act and OSFI E-23 standards register themselves
under compact names (``euaiact`` / ``osfie23``). STRATEGY.md and the
README always referred to them in snake_case (``eu_ai_act`` /
``osfi_e23``). The registry exposes both forms; the snake_case form
emits a one-shot deprecation warning.
"""

from __future__ import annotations

import logging

import pytest

from mrm.compliance.registry import (
    ComplianceRegistry,
    compliance_registry,
)


# ---------------------------------------------------------------------------
# Bundled aliases
# ---------------------------------------------------------------------------


def test_eu_ai_act_alias_resolves_to_euaiact():
    compliance_registry.load_builtin_standards()
    canonical = compliance_registry.get("euaiact")
    aliased = compliance_registry.get("eu_ai_act")
    assert aliased is canonical


def test_osfi_e23_alias_resolves_to_osfie23():
    compliance_registry.load_builtin_standards()
    canonical = compliance_registry.get("osfie23")
    aliased = compliance_registry.get("osfi_e23")
    assert aliased is canonical


def test_alias_emits_one_shot_deprecation_warning(caplog):
    """First use of an alias logs a deprecation warning; subsequent
    uses are silent so we don't spam logs."""
    compliance_registry.load_builtin_standards()
    # Use a fresh registry so the warned-set is empty.
    fresh = ComplianceRegistry()
    fresh._standards["target"] = compliance_registry.get("euaiact")
    fresh.register_alias("legacy", "target")

    with caplog.at_level(logging.WARNING, logger="mrm.compliance.registry"):
        fresh.get("legacy")
        warnings_after_first = [r for r in caplog.records if "deprecated" in r.getMessage()]
        fresh.get("legacy")
        warnings_after_second = [r for r in caplog.records if "deprecated" in r.getMessage()]

    assert len(warnings_after_first) == 1
    # The second .get() should NOT add another warning.
    assert len(warnings_after_second) == 1


# ---------------------------------------------------------------------------
# Aliases do not pollute list_standards()
# ---------------------------------------------------------------------------


def test_list_standards_excludes_aliases():
    """``list_standards`` returns the canonical name set only -- aliases
    are documentation, not catalog entries."""
    compliance_registry.load_builtin_standards()
    names = set(compliance_registry.list_standards())
    assert "euaiact" in names
    assert "osfie23" in names
    # Aliases must NOT appear in the canonical listing.
    assert "eu_ai_act" not in names
    assert "osfi_e23" not in names


# ---------------------------------------------------------------------------
# Lookup behaviour for unknown names
# ---------------------------------------------------------------------------


def test_unknown_name_still_raises_keyerror():
    compliance_registry.load_builtin_standards()
    with pytest.raises(KeyError, match="not found"):
        compliance_registry.get("does_not_exist")


def test_alias_to_unknown_canonical_raises():
    fresh = ComplianceRegistry()
    fresh.register_alias("anything", "nowhere")
    with pytest.raises(KeyError):
        fresh.get("anything")


# ---------------------------------------------------------------------------
# Report generator dispatch via alias
# ---------------------------------------------------------------------------


def test_generate_compliance_report_accepts_alias(tmp_path):
    """The thin ``generate_compliance_report`` dispatcher must route an
    alias through to the canonical implementation."""
    from mrm.compliance.report_generator import generate_compliance_report

    compliance_registry.load_builtin_standards()

    model_cfg = {
        "model": {
            "name": "credit_scorecard",
            "version": "1.0.0",
            "owner": "credit-team",
            "risk_tier": "tier_1",
            "ai_materiality": "high",
        }
    }

    class _R:
        def __init__(self, passed, score=None):
            self.passed = passed
            self.score = score
            self.details = {}
            self.failure_reason = None

    results = {"tabular.MissingValues": _R(True, 0.02)}

    aliased_report = generate_compliance_report(
        standard_name="eu_ai_act",  # alias
        model_name="credit_scorecard",
        model_config=model_cfg,
        test_results=results,
    )
    canonical_report = generate_compliance_report(
        standard_name="euaiact",
        model_name="credit_scorecard",
        model_config=model_cfg,
        test_results=results,
    )
    # Trim the report-date header line which contains a timestamp.
    def _strip_ts(r: str) -> str:
        return "\n".join(
            line for line in r.splitlines() if "Report Date" not in line
        )

    assert _strip_ts(aliased_report) == _strip_ts(canonical_report)
