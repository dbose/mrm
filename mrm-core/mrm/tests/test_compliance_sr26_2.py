"""Tests for the SR 26-2 bundled compliance standard."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mrm.compliance.builtin.sr26_2 import SR26_2Standard
from mrm.compliance.registry import compliance_registry


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_sr26_2_is_registered_after_builtin_load():
    compliance_registry.load_builtin_standards()
    assert "sr26_2" in compliance_registry.list_standards()
    cls = compliance_registry.get("sr26_2")
    assert cls is SR26_2Standard


def test_sr26_2_metadata():
    s = SR26_2Standard()
    assert s.name == "sr26_2"
    assert s.jurisdiction == "US"
    assert s.version == "2026"
    assert "SR 26-2" in s.display_name


# ---------------------------------------------------------------------------
# Paragraphs and test mapping
# ---------------------------------------------------------------------------


def test_paragraphs_include_traditional_and_ai_specific_sections():
    paragraphs = SR26_2Standard().get_paragraphs()
    # Traditional MRM sections carried over from SR 11-7
    assert "Section I.A" in paragraphs
    assert "Section II.A" in paragraphs
    assert "Section III.A" in paragraphs
    assert "Section IV.A" in paragraphs
    # AI-specific additions
    assert "Section II.AI.A" in paragraphs  # Activity logging
    assert "Section II.AI.B" in paragraphs  # Replay
    assert "Section II.AI.C" in paragraphs  # AI risk testing
    assert "Section II.AI.D" in paragraphs  # RAG context integrity
    assert "Section III.AI.A" in paragraphs  # Tamper-evident audit trail
    assert "Section III.AI.B" in paragraphs  # Chain-of-custody
    assert "Section III.AI.C" in paragraphs  # AI materiality
    assert "Section IV.AI.C" in paragraphs  # Per-decision audit trail
    # New Section V (vendor AI)
    assert "Section V.A" in paragraphs
    assert "Section V.B" in paragraphs


def test_each_paragraph_has_title_and_requirement():
    paragraphs = SR26_2Standard().get_paragraphs()
    for key, body in paragraphs.items():
        assert "title" in body, f"{key} missing title"
        assert "requirement" in body, f"{key} missing requirement"
        assert body["title"].strip()
        assert body["requirement"].strip()


def test_genai_tests_map_into_ai_subsections():
    mapping = SR26_2Standard().get_test_mapping()
    # GenAI risk tests should anchor to II.AI.C
    for t in [
        "genai.HallucinationRate",
        "genai.PromptInjection",
        "genai.JailbreakResistance",
        "genai.PIIDetection",
        "genai.OutputBias",
        "genai.ToxicityRate",
    ]:
        assert mapping.get(t) == "Section II.AI.C", t


def test_traditional_tests_still_map():
    mapping = SR26_2Standard().get_test_mapping()
    assert mapping["ccr.MCConvergence"] == "Section II.A"
    assert mapping["tabular.DataDrift"] == "Section II.B"
    assert mapping["compliance.GovernanceCheck"] == "Section III.A"


# ---------------------------------------------------------------------------
# Governance checks
# ---------------------------------------------------------------------------


def test_governance_checks_include_ai_specific():
    checks = SR26_2Standard().get_governance_checks()
    assert "ai_activity_logging_enabled" in checks
    assert "ai_materiality_classified" in checks
    assert "tamper_evident_evidence" in checks
    assert "third_party_ai_assessed" in checks
    # Each should reference an SR 26-2 section
    for name, body in checks.items():
        assert body["paragraph_ref"].startswith("SR 26-2"), name
        assert "description" in body
        assert "config_key" in body


# ---------------------------------------------------------------------------
# Replay / evidence anchoring (the SR-26-2-specific helpers)
# ---------------------------------------------------------------------------


def test_replay_anchored_clauses_cover_ai_logging_and_replay():
    s = SR26_2Standard()
    assert s.is_replay_anchored("Section II.AI.A") is True
    assert s.is_replay_anchored("Section II.AI.B") is True
    assert s.is_replay_anchored("Section IV.AI.C") is True


def test_replay_anchored_returns_false_for_non_ai_clauses():
    s = SR26_2Standard()
    assert s.is_replay_anchored("Section II.A") is False
    assert s.is_replay_anchored("Section III.A") is False


def test_evidence_vault_anchored_clauses():
    s = SR26_2Standard()
    assert s.is_evidence_vault_anchored("Section III.AI.A") is True
    assert s.is_evidence_vault_anchored("Section III.AI.B") is True
    assert s.is_evidence_vault_anchored("Section II.A") is False


def test_required_evidence_types_for_anchored_clauses():
    s = SR26_2Standard()
    assert s.required_evidence_types("Section II.AI.A") == ["replay:decision_record"]
    assert s.required_evidence_types("Section III.AI.A") == ["evidence:hash_chained_packet"]
    assert s.required_evidence_types("Section II.A") == []


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_model_config():
    return {
        "model": {
            "name": "rag_customer_service",
            "version": "1.0.0",
            "owner": "ai-risk@bank.example",
            "risk_tier": "tier_1",
            "ai_materiality": "high",
            "materiality": "high",
            "complexity": "high",
            "use_case": "Customer support assistant",
            "methodology": "RAG over policy corpus",
            "validation_frequency": "annual",
            "third_party": "Yes",
            "replay_backend": "s3",
            "evidence_backend": "s3_object_lock",
            "third_party_assessment": "SOC 2 + vendor attestation",
        }
    }


@pytest.fixture
def sample_test_results():
    class _Result:
        def __init__(self, passed, score=None, details=None, failure_reason=None):
            self.passed = passed
            self.score = score
            self.details = details or {}
            self.failure_reason = failure_reason

    return {
        "genai.HallucinationRate": _Result(True, 0.02),
        "genai.PromptInjection":  _Result(True, 0.98),
        "genai.PIIDetection":     _Result(True, 1.0),
        "tabular.DataDrift":      _Result(False, 0.31, failure_reason="KS > threshold"),
    }


def test_generate_report_contains_required_sections(sample_model_config, sample_test_results):
    report = SR26_2Standard().generate_report(
        model_name="rag_customer_service",
        model_config=sample_model_config,
        test_results=sample_test_results,
    )
    # Required document anatomy
    assert "# SR 26-2 Model Validation Report" in report
    assert "## 1. Executive Summary" in report
    assert "## 2. Model Inventory Card" in report
    assert "## 3. SR 26-2 AI Evidence Posture" in report
    assert "## 4. SR 26-2 Compliance Matrix" in report
    assert "## 5. Detailed Validation Test Results" in report
    assert "## 7. Findings and Recommendations" in report
    # AI-specific clause appears in matrix
    assert "Section II.AI.A" in report
    assert "Section III.AI.A" in report


def test_report_marks_ai_evidence_posture_as_configured(sample_model_config, sample_test_results):
    report = SR26_2Standard().generate_report(
        model_name="rag_customer_service",
        model_config=sample_model_config,
        test_results=sample_test_results,
    )
    # All three SR-26-2 AI evidence rows should be CONFIGURED in this fixture
    posture_section = report.split("## 3. SR 26-2 AI Evidence Posture", 1)[1].split("## 4.", 1)[0]
    assert posture_section.count("CONFIGURED") >= 3
    assert "NOT CONFIGURED" not in posture_section


def test_report_marks_ai_evidence_posture_as_not_configured_when_absent():
    bare_config = {
        "model": {
            "name": "old_school_pd_model",
            "version": "0.9",
            "risk_tier": "tier_2",
        }
    }
    report = SR26_2Standard().generate_report(
        model_name="old_school_pd_model",
        model_config=bare_config,
        test_results={},
    )
    posture = report.split("## 3. SR 26-2 AI Evidence Posture", 1)[1].split("## 4.", 1)[0]
    assert posture.count("NOT CONFIGURED") >= 3


def test_report_flags_failed_tests_in_findings(sample_model_config, sample_test_results):
    report = SR26_2Standard().generate_report(
        model_name="rag_customer_service",
        model_config=sample_model_config,
        test_results=sample_test_results,
    )
    findings = report.split("## 7. Findings and Recommendations", 1)[1]
    assert "REQUIRES REMEDIATION" in findings
    assert "tabular.DataDrift" in findings


def test_report_writes_to_disk_when_output_path_provided(tmp_path: Path, sample_model_config, sample_test_results):
    out = tmp_path / "sr26_2.md"
    SR26_2Standard().generate_report(
        model_name="rag_customer_service",
        model_config=sample_model_config,
        test_results=sample_test_results,
        output_path=out,
    )
    assert out.exists()
    assert "SR 26-2" in out.read_text()


# ---------------------------------------------------------------------------
# Crosswalk integration
# ---------------------------------------------------------------------------


def test_crosswalk_includes_sr26_2_transition_block():
    crosswalk_path = (
        Path(__file__).parent.parent / "compliance" / "crosswalks" / "standards.yaml"
    )
    data = yaml.safe_load(crosswalk_path.read_text())
    # Top-level keys
    assert "sr_11_7_to_sr_26_2_transition" in data
    transition = data["sr_11_7_to_sr_26_2_transition"]
    assert "carried_over_sections" in transition
    assert "new_in_sr_26_2" in transition
    # The renumbering rule we documented (III.D -> III.C) must survive.
    renumbered = [
        m for m in transition["carried_over_sections"]
        if m["sr117"] == "Section III.D"
    ]
    assert renumbered and renumbered[0]["sr26_2"] == "Section III.C"


def test_crosswalk_lists_ai_specific_new_clauses():
    crosswalk_path = (
        Path(__file__).parent.parent / "compliance" / "crosswalks" / "standards.yaml"
    )
    data = yaml.safe_load(crosswalk_path.read_text())
    new_sections = {
        item["section"]
        for item in data["sr_11_7_to_sr_26_2_transition"]["new_in_sr_26_2"]
    }
    # The four most important wedge clauses
    assert "Section II.AI.A" in new_sections
    assert "Section II.AI.B" in new_sections
    assert "Section III.AI.A" in new_sections
    assert "Section V.A" in new_sections


def test_crosswalk_metadata_records_sr26_2_as_a_standard():
    crosswalk_path = (
        Path(__file__).parent.parent / "compliance" / "crosswalks" / "standards.yaml"
    )
    data = yaml.safe_load(crosswalk_path.read_text())
    names = [s["name"] for s in data["metadata"]["standards_covered"]]
    assert "SR 26-2" in names
