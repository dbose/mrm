"""Federal Reserve SR 26-2 -- Model Risk Management for AI / GenAI Systems.

Bundled compliance standard for the United States Federal Reserve's
2026 supervisory letter that supersedes SR 11-7 and SR 21-8 for banking
organizations with total assets above $30B.

Compared to SR 11-7 (2011), SR 26-2 adds explicit expectations for:

1. AI / GenAI activity logging -- "tamper-evident, integrity-protected,
   immutable, and complete" records of every model decision. ``mrm-core``
   anchors this to the P7 DecisionRecord primitive (see
   ``mrm/replay/record.py`` and ``docs/spec/replay-record-v1.md``).

2. AI-specific risk tiering -- materiality classification that includes
   GenAI-specific risks (hallucination, prompt injection, RAG context
   manipulation, drift).

3. Independent validation cadence for AI / GenAI models -- shorter than
   the traditional-model cadence in SR 11-7 Section III.D.

4. Tamper-evident audit-trail integrity -- anchored to the P5/P9
   Evidence Vault hash chain (see ``mrm/evidence/packet.py`` and
   ``docs/spec/evidence-vault-v1.md``).

5. Vendor / third-party AI model governance -- extends SR 11-7 III.E
   for foundation models and managed LLM endpoints.

Usage::

    mrm docs generate ccr_monte_carlo --compliance standard:sr26_2

Status note (2026-05): SR 26-2 paragraph identifiers below follow the
public-comment draft naming convention. The plugin structure is
finalised; identifiers will be re-keyed against the final published
letter as part of a maintenance release if Fed numbering differs.
Mappings are kept stable by maintaining the paragraph IDs as keys --
re-keying touches only this file.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mrm.compliance.base import ComplianceStandard
from mrm.compliance.registry import register_standard

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers (result introspection -- same shape as sr117.py)
# -----------------------------------------------------------------------

def _result_passed(result) -> bool:
    if hasattr(result, "passed"):
        return result.passed
    if isinstance(result, dict):
        return result.get("passed", False)
    return False


def _result_score(result) -> Optional[float]:
    if hasattr(result, "score"):
        return result.score
    if isinstance(result, dict):
        return result.get("score")
    return None


def _result_details(result) -> Dict:
    if hasattr(result, "details"):
        return result.details or {}
    if isinstance(result, dict):
        return result.get("details", {})
    return {}


def _result_failure(result) -> Optional[str]:
    if hasattr(result, "failure_reason"):
        return result.failure_reason
    if isinstance(result, dict):
        return result.get("failure_reason")
    return None


def _get_compliance_ref(details: Dict, standard_name: str = "sr26_2") -> str:
    """Read compliance reference from test result details."""
    refs = details.get("compliance_references", {})
    if isinstance(refs, dict) and standard_name in refs:
        return refs[standard_name]
    return details.get("sr26_2_reference", "")


# =======================================================================
# SR 26-2 Standard Implementation
# =======================================================================

@register_standard
class SR26_2Standard(ComplianceStandard):
    """Federal Reserve SR 26-2 -- Model Risk Management for AI Systems."""

    name = "sr26_2"
    display_name = (
        "Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk "
        "Management for AI Systems"
    )
    description = (
        "SR 26-2 supersedes SR 11-7 and SR 21-8 for banking organizations "
        "with total assets above $30B. It modernises model risk management "
        "expectations to cover AI and GenAI systems explicitly, mandating "
        "tamper-evident decision logging, AI-specific risk tiering, and "
        "independent validation cadences appropriate to the model class."
    )
    jurisdiction = "US"
    version = "2026"

    # ------------------------------------------------------------------
    # Identifiers tying SR 26-2 clauses to mrm-core primitives
    # ------------------------------------------------------------------

    #: Clauses that REQUIRE replay-record evidence (P7 DecisionRecord).
    REPLAY_ANCHORED_CLAUSES = frozenset({
        "Section II.AI.A",   # AI activity logging requirement
        "Section II.AI.B",   # Decision reconstruction (replay-by-default)
        "Section IV.AI.C",   # AI inventory: per-decision audit trail
    })

    #: Clauses that REQUIRE evidence-vault hash-chain evidence (P5/P9).
    EVIDENCE_VAULT_ANCHORED_CLAUSES = frozenset({
        "Section III.AI.A",  # Tamper-evident audit trail
        "Section III.AI.B",  # Chain-of-custody for validation evidence
    })

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        """Map SR 26-2 sections to requirements.

        Section structure parallels SR 11-7 for traditional models, with
        AI-specific subsections appended:

            Section I.*     -- Model development & implementation
            Section II.*    -- Model validation (with II.AI for AI/GenAI)
            Section III.*   -- Governance, policies, controls (with III.AI)
            Section IV.*    -- Model inventory (with IV.AI)
            Section V.*     -- Third-party / vendor AI models (NEW vs SR 11-7)
        """
        return {
            # === Section I: Development & implementation ============
            "Section I.A": {
                "title": "Model Development -- Sound Theory and Design",
                "requirement": (
                    "Models, including AI and GenAI systems, must be based "
                    "on sound theory consistent with their intended use. "
                    "For AI systems this includes the rationale for the "
                    "selected foundation model, prompt design, retrieval "
                    "augmentation, and fine-tuning strategy."
                ),
            },
            "Section I.B": {
                "title": "Model Use and Intended Purpose",
                "requirement": (
                    "The intended use of every model must be documented. "
                    "For AI systems, documentation must include in-scope "
                    "and out-of-scope use cases, expected user populations, "
                    "and guardrails preventing use beyond design intent."
                ),
            },
            "Section I.C": {
                "title": "Data Quality, Representativeness, and Lineage",
                "requirement": (
                    "Training, validation, and grounding data must be "
                    "assessed for quality, bias, and representativeness. "
                    "Lineage of all data feeding an AI system, including "
                    "retrieval corpora, must be auditable."
                ),
            },

            # === Section II: Validation =============================
            "Section II.A": {
                "title": "Conceptual Soundness Review",
                "requirement": (
                    "Independent review of model theory, assumptions, "
                    "and design choices, conducted by personnel "
                    "independent of model development."
                ),
            },
            "Section II.B": {
                "title": "Ongoing Monitoring and Process Verification",
                "requirement": (
                    "Models must be subject to ongoing monitoring of "
                    "inputs, outputs, drift, and stability. Materially "
                    "deviating behaviour must trigger re-validation."
                ),
            },
            "Section II.C": {
                "title": "Outcomes Analysis and Backtesting",
                "requirement": (
                    "Where applicable, model outputs must be compared "
                    "against realised outcomes; performance degradation "
                    "must trigger documented action."
                ),
            },
            "Section II.D": {
                "title": "Sensitivity and Stress Testing",
                "requirement": (
                    "Model behaviour must be tested under stressed and "
                    "adverse scenarios appropriate to the model class."
                ),
            },

            # --- AI-specific validation subsections (NEW vs SR 11-7) ---
            "Section II.AI.A": {
                "title": "AI Activity Logging (Decision Records)",
                "requirement": (
                    "For every inference produced by an AI or GenAI "
                    "system, the institution must capture a tamper-"
                    "evident decision record containing (i) the exact "
                    "input state, (ii) the model identity and version, "
                    "(iii) the inference parameters, and (iv) the raw "
                    "model output prior to downstream post-processing. "
                    "Records must be retained for the applicable "
                    "regulator retention window."
                ),
            },
            "Section II.AI.B": {
                "title": "Decision Reconstruction (Replay)",
                "requirement": (
                    "The institution must be able to reconstruct any "
                    "AI decision from its captured record on demand. "
                    "Replay must produce byte-for-byte equivalent output "
                    "for deterministic models, and output within a "
                    "documented tolerance for stochastic models."
                ),
            },
            "Section II.AI.C": {
                "title": "AI-Specific Risk Testing",
                "requirement": (
                    "AI systems must be tested for hallucination rate, "
                    "prompt-injection resilience, PII leakage, bias and "
                    "fairness, and adversarial robustness, with thresholds "
                    "appropriate to the model's risk tier."
                ),
            },
            "Section II.AI.D": {
                "title": "RAG Context Integrity",
                "requirement": (
                    "Where AI systems use retrieval augmentation, the "
                    "retrieved context must be captured as part of the "
                    "decision record. Retrieval corpus drift must be "
                    "monitored on a defined frequency."
                ),
            },

            # === Section III: Governance ============================
            "Section III.A": {
                "title": "Policies, Procedures, and Controls",
                "requirement": (
                    "A board-approved model risk management policy "
                    "must establish standards for model development, "
                    "validation, use, and retirement, including "
                    "AI-specific provisions."
                ),
            },
            "Section III.B": {
                "title": "Roles, Responsibilities, and Independence",
                "requirement": (
                    "Model owners, developers, validators, and users "
                    "must have clearly designated roles. Validation "
                    "must be conducted independently of development."
                ),
            },
            "Section III.C": {
                "title": "Risk-Tier-Commensurate Validation Cadence",
                "requirement": (
                    "Validation frequency must be commensurate with the "
                    "model's risk tier. For AI systems classified as "
                    "high-impact, the default cadence is no less than "
                    "annual, with event-driven re-validation on material "
                    "change."
                ),
            },

            # --- AI-specific governance subsections (NEW vs SR 11-7) ---
            "Section III.AI.A": {
                "title": "Tamper-Evident Audit Trail",
                "requirement": (
                    "Validation evidence for AI systems must be stored "
                    "in a tamper-evident, integrity-protected, immutable, "
                    "and complete form. Cryptographic chain-of-custody "
                    "(e.g. hash-chained or Merkle-aggregated evidence) "
                    "satisfies this expectation."
                ),
            },
            "Section III.AI.B": {
                "title": "Chain-of-Custody for Evidence",
                "requirement": (
                    "Each evidence artefact must be linked to its "
                    "predecessor by a cryptographic hash. Verification "
                    "of the chain must detect any retroactive "
                    "modification."
                ),
            },
            "Section III.AI.C": {
                "title": "AI Materiality Classification",
                "requirement": (
                    "AI systems must be classified by materiality and "
                    "complexity. Classification must consider downstream "
                    "decision impact, customer exposure, regulatory "
                    "exposure, and reputational risk."
                ),
            },

            # === Section IV: Inventory ==============================
            "Section IV.A": {
                "title": "Comprehensive Model Inventory",
                "requirement": (
                    "All models, including AI systems and foundation-"
                    "model-based applications, must be tracked in a "
                    "comprehensive inventory."
                ),
            },
            "Section IV.B": {
                "title": "Version Control and Change Management",
                "requirement": (
                    "All material changes to models, including prompt "
                    "templates, retrieval corpora, and fine-tuning "
                    "weights, must be version-controlled."
                ),
            },
            "Section IV.AI.C": {
                "title": "Per-Decision Audit Trail Linkage",
                "requirement": (
                    "The inventory entry for each AI system must link "
                    "to its decision-record store, enabling regulator "
                    "sample-on-demand of historical inferences."
                ),
            },

            # === Section V: Third-party AI (NEW vs SR 11-7) =========
            "Section V.A": {
                "title": "Third-Party / Vendor AI Model Governance",
                "requirement": (
                    "Foundation models and managed AI endpoints "
                    "operated by third parties must be subject to "
                    "model risk management commensurate with the "
                    "institution's reliance on them."
                ),
            },
            "Section V.B": {
                "title": "Vendor-Provided Evidence",
                "requirement": (
                    "Where the institution relies on vendor-provided "
                    "model evidence (e.g. SOC reports, model cards), "
                    "the institution must independently assess the "
                    "sufficiency of that evidence against this "
                    "guidance."
                ),
            },
        }

    def get_test_mapping(self) -> Dict[str, str]:
        """Map test names to SR 26-2 sections."""
        return {
            # CCR and traditional quant tests
            "ccr.MCConvergence": "Section II.A",
            "ccr.EPEReasonableness": "Section II.A",
            "ccr.PFEBacktest": "Section II.C",
            "ccr.CVASensitivity": "Section II.D",
            "ccr.WrongWayRisk": "Section II.A",
            "ccr.ExposureProfileShape": "Section II.A",
            "ccr.CollateralEffectiveness": "Section II.C",
            "ccr.CPS230GovernanceCheck": "Section III.A",
            "compliance.GovernanceCheck": "Section III.A",
            # Tabular
            "tabular.MissingValues": "Section I.C",
            "tabular.DataDrift": "Section II.B",
            "tabular.OutlierDetection": "Section I.C",
            # GenAI -- the AI-specific subsections
            "genai.HallucinationRate": "Section II.AI.C",
            "genai.FactualAccuracy": "Section II.AI.C",
            "genai.PromptInjection": "Section II.AI.C",
            "genai.JailbreakResistance": "Section II.AI.C",
            "genai.AdversarialPerturbation": "Section II.AI.C",
            "genai.PIIDetection": "Section II.AI.C",
            "genai.DemographicParity": "Section II.AI.C",
            "genai.OutputBias": "Section II.AI.C",
            "genai.ToxicityRate": "Section II.AI.C",
            "genai.SafetyClassifier": "Section II.AI.C",
            "genai.OutputConsistency": "Section II.B",
            "genai.SemanticDrift": "Section II.B",
            "genai.LatencyBound": "Section II.B",
            "genai.CostBound": "Section II.B",
        }

    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        """Define SR 26-2 governance requirements.

        Note: the ``ai_*`` checks are the SR 26-2-specific additions
        beyond SR 11-7.
        """
        return {
            "risk_tier_assigned": {
                "description": "Risk tier assigned based on materiality and complexity",
                "paragraph_ref": "SR 26-2 Section III.C",
                "config_key": "risk_tier",
            },
            "owner_designated": {
                "description": "Model owner / accountability designated",
                "paragraph_ref": "SR 26-2 Section III.B",
                "config_key": "owner",
            },
            "validation_frequency_set": {
                "description": "Validation frequency defined commensurate with risk",
                "paragraph_ref": "SR 26-2 Section III.C",
                "config_key": "validation_frequency",
            },
            "use_case_documented": {
                "description": "Model purpose and intended use documented",
                "paragraph_ref": "SR 26-2 Section I.B",
                "config_key": "use_case",
            },
            "methodology_documented": {
                "description": "Model methodology and theory documented",
                "paragraph_ref": "SR 26-2 Section I.A",
                "config_key": "methodology",
            },
            "assumptions_documented": {
                "description": "Key assumptions documented and justified",
                "paragraph_ref": "SR 26-2 Section I.C",
                "config_key": "assumptions",
            },
            "model_inventory_maintained": {
                "description": "Model tracked in comprehensive inventory",
                "paragraph_ref": "SR 26-2 Section IV.A",
                "config_key": "name",
            },
            "version_controlled": {
                "description": "Version control and change management in place",
                "paragraph_ref": "SR 26-2 Section IV.B",
                "config_key": "version",
            },
            # --- AI-specific (NEW vs SR 11-7) ---------------------------
            "ai_activity_logging_enabled": {
                "description": (
                    "AI activity logging via DecisionRecord enabled for "
                    "AI / GenAI models"
                ),
                "paragraph_ref": "SR 26-2 Section II.AI.A",
                "config_key": "replay_backend",
            },
            "ai_materiality_classified": {
                "description": (
                    "AI materiality classification documented (high / "
                    "medium / low impact)"
                ),
                "paragraph_ref": "SR 26-2 Section III.AI.C",
                "config_key": "ai_materiality",
            },
            "tamper_evident_evidence": {
                "description": (
                    "Validation evidence stored in tamper-evident "
                    "(hash-chained) form"
                ),
                "paragraph_ref": "SR 26-2 Section III.AI.A",
                "config_key": "evidence_backend",
            },
            "third_party_ai_assessed": {
                "description": (
                    "Third-party AI / foundation models assessed for "
                    "vendor governance"
                ),
                "paragraph_ref": "SR 26-2 Section V.A",
                "config_key": "third_party_assessment",
            },
        }

    # ------------------------------------------------------------------
    # SR-26-2-specific helpers exposed to consumers
    # ------------------------------------------------------------------

    def is_replay_anchored(self, paragraph_ref: str) -> bool:
        """True if a paragraph requires DecisionRecord evidence (P7)."""
        return paragraph_ref in self.REPLAY_ANCHORED_CLAUSES

    def is_evidence_vault_anchored(self, paragraph_ref: str) -> bool:
        """True if a paragraph requires evidence-vault chain (P5/P9)."""
        return paragraph_ref in self.EVIDENCE_VAULT_ANCHORED_CLAUSES

    def required_evidence_types(self, paragraph_ref: str) -> List[str]:
        """The mrm-core evidence types satisfying a paragraph.

        Returns a list (possibly empty) of the canonical evidence-type
        identifiers a regulator would expect for the given clause.
        """
        evidence: List[str] = []
        if self.is_replay_anchored(paragraph_ref):
            evidence.append("replay:decision_record")
        if self.is_evidence_vault_anchored(paragraph_ref):
            evidence.append("evidence:hash_chained_packet")
        return evidence

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        test_results: Dict[str, Any],
        trigger_events: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate SR 26-2 validation report."""
        now = datetime.now(timezone.utc)
        model_info = model_config.get("model", model_config)

        # Read compliance mapping (under standards.sr26_2 or legacy key).
        sr26_2_mapping = (
            model_config
            .get("compliance", {})
            .get("standards", {})
            .get("sr26_2", {})
            .get("mapping", model_config.get("sr26_2_mapping", {}))
        )
        triggers_cfg = model_config.get("triggers", [])

        paragraphs = self.get_paragraphs()
        test_mapping = self.get_test_mapping()

        sections = [
            self._header(model_name, model_info, now),
            self._executive_summary(model_name, model_info, test_results, now),
            self._model_inventory_card(model_info, model_config),
            self._ai_evidence_status(model_config),
            self._compliance_matrix(test_results, sr26_2_mapping, paragraphs, test_mapping),
            self._detailed_test_results(test_results, paragraphs, test_mapping),
            self._trigger_section(triggers_cfg, trigger_events),
            self._findings_and_recommendations(test_results, test_mapping),
            self._approval_section(model_info, now),
        ]

        report = "\n\n".join(sections)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report written to {output_path}")

        return report

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, model_name: str, model_info: Dict, now: datetime) -> str:
        return f"""# SR 26-2 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | {model_name} |
| **Version** | {model_info.get('version', 'N/A')} |
| **Report Date** | {now.strftime('%Y-%m-%d %H:%M UTC')} |
| **Regulatory Framework** | {self.display_name} |
| **Standard Version** | {self.version} |
| **Risk Tier** | {model_info.get('risk_tier', 'N/A')} |
| **AI Materiality** | {model_info.get('ai_materiality', 'N/A')} |
| **Owner** | {model_info.get('owner', 'N/A')} |
| **Validation Frequency** | {model_info.get('validation_frequency', 'N/A')} |
| **Independent Validator** | MRM Team |

---"""

    def _executive_summary(self, model_name, model_info, test_results, now):
        total = len(test_results)
        passed = sum(1 for r in test_results.values() if _result_passed(r))
        failed = total - passed
        pass_rate = f"{passed / total * 100:.1f}%" if total > 0 else "N/A"
        status_marker = "APPROVED" if failed == 0 and total > 0 else "REQUIRES REMEDIATION"
        status = "PASS" if failed == 0 and total > 0 else "FAIL"

        return f"""## 1. Executive Summary

This report presents the independent validation results for the
**{model_name}** model (v{model_info.get('version', '1.0.0')})
conducted on {now.strftime('%Y-%m-%d')}.

The validation was performed in accordance with **{self.display_name}**
which supersedes SR 11-7 and SR 21-8 for banking organizations with
total assets above $30B. The model is classified as
**{model_info.get('risk_tier', 'tier_1').replace('_', ' ').title()}**
({model_info.get('materiality', 'high')} materiality,
{model_info.get('complexity', 'high')} complexity), requiring
{model_info.get('validation_frequency', 'annual')} validation.

### Overall Result: **{status_marker}**

| Metric | Value |
|--------|-------|
| Validation Tests Executed | {total} |
| Tests Passed | {passed} |
| Tests Failed | {failed} |
| Pass Rate | {pass_rate} |
| Validation Status | **{status}** |

**Model Purpose:** {model_info.get('description', 'N/A')}

**Methodology:** {model_info.get('methodology', 'N/A')}

### Validation Scope

Per SR 26-2 Section II, this validation includes:

- **Conceptual Soundness** (II.A): Review of model theory, assumptions, and structure
- **Ongoing Monitoring** (II.B): Analysis of model inputs, outputs, and stability
- **Outcomes Analysis** (II.C): Backtesting where applicable
- **Sensitivity & Stress Testing** (II.D): Behaviour under adverse scenarios
- **AI Activity Logging** (II.AI.A): Decision-record capture per inference
- **Decision Reconstruction** (II.AI.B): Replay of historical decisions
- **AI-Specific Risk Testing** (II.AI.C): Hallucination, injection, PII, bias, robustness
- **RAG Context Integrity** (II.AI.D): Retrieval corpus drift monitoring"""

    def _model_inventory_card(self, model_info, model_config):
        params = model_info.get("parameters", model_config.get("parameters", {}))
        param_rows = "\n".join(
            f"| {k} | {v} |" for k, v in params.items()
        ) if params else "| N/A | N/A |"

        assumptions = model_info.get("assumptions", "Not documented")
        if isinstance(assumptions, list):
            assumptions = "; ".join(assumptions)

        return f"""## 2. Model Inventory Card

### 2.1 Identification (SR 26-2 Section IV.A)

| Field | Value |
|-------|-------|
| Model Name | {model_info.get('name', 'N/A')} |
| Version | {model_info.get('version', 'N/A')} |
| Owner | {model_info.get('owner', 'N/A')} |
| Business Line | {model_info.get('business_line', 'N/A')} |
| Use Case | {model_info.get('use_case', 'N/A')} |
| Methodology | {model_info.get('methodology', 'N/A')} |
| Risk Tier | {model_info.get('risk_tier', 'N/A')} |
| AI Materiality | {model_info.get('ai_materiality', 'N/A')} |
| Materiality | {model_info.get('materiality', 'N/A')} |
| Complexity | {model_info.get('complexity', 'N/A')} |
| Validation Frequency | {model_info.get('validation_frequency', 'N/A')} |
| Third-Party Model | {model_info.get('third_party', 'No')} |

### 2.2 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
{param_rows}

### 2.3 Key Assumptions (SR 26-2 Section I.C)

{assumptions}

### 2.4 Risk Tier Classification Rationale (SR 26-2 Section III.AI.C)

Per **SR 26-2 Section III.C**, models must be validated on a frequency
commensurate with their risk and materiality. AI materiality
classification additionally considers downstream decision impact,
customer exposure, regulatory exposure, and reputational risk."""

    def _ai_evidence_status(self, model_config: Dict) -> str:
        """SR-26-2-specific section reporting the AI evidence posture."""
        model_info = model_config.get("model", model_config)
        replay_backend = model_info.get("replay_backend") or model_config.get("replay_backend")
        evidence_backend = model_info.get("evidence_backend") or model_config.get("evidence_backend")
        third_party = model_info.get("third_party_assessment") or model_config.get("third_party_assessment")

        def _ok(val) -> str:
            return "CONFIGURED" if val else "NOT CONFIGURED"

        rows = [
            f"| AI Activity Logging (II.AI.A) | {_ok(replay_backend)} | "
            f"{replay_backend or '-'} |",
            f"| Tamper-Evident Evidence Vault (III.AI.A) | "
            f"{_ok(evidence_backend)} | {evidence_backend or '-'} |",
            f"| Third-Party AI Assessment (V.A) | {_ok(third_party)} | "
            f"{third_party or '-'} |",
        ]

        rows_str = "\n".join(rows)

        return f"""## 3. SR 26-2 AI Evidence Posture

This section is unique to SR 26-2 and reports the institution's
configuration for the AI-specific evidence expectations introduced in
this guidance.

| Expectation | Status | Configuration |
|-------------|--------|---------------|
{rows_str}

### 3.1 Replay-Anchored Clauses

The following clauses are satisfied by per-decision replay records
(captured via the ``mrm-core`` ``replay/`` primitive):

- **Section II.AI.A** -- AI Activity Logging
- **Section II.AI.B** -- Decision Reconstruction (Replay)
- **Section IV.AI.C** -- Per-Decision Audit Trail Linkage

### 3.2 Evidence-Vault-Anchored Clauses

The following clauses are satisfied by hash-chained evidence packets:

- **Section III.AI.A** -- Tamper-Evident Audit Trail
- **Section III.AI.B** -- Chain-of-Custody for Evidence"""

    def _compliance_matrix(self, test_results, sr26_2_mapping, paragraphs, test_mapping):
        rows = []

        for para_key, para_info in paragraphs.items():
            mapped_tests = [t for t, p in test_mapping.items() if p == para_key]

            statuses = []
            evidence_items = []
            for test_name in mapped_tests:
                if test_name in test_results:
                    r = test_results[test_name]
                    p = _result_passed(r)
                    statuses.append(p)
                    details = _result_details(r)
                    ref = _get_compliance_ref(details, self.name)
                    evidence_items.append(
                        f"{test_name}: {'PASS' if p else 'FAIL'}"
                        + (f" -- {ref}" if ref else "")
                    )

            for section_items in sr26_2_mapping.values():
                if isinstance(section_items, list):
                    for item in section_items:
                        if item.get("section") == para_key:
                            evidence_items.append(f"Config: {item.get('evidence', 'N/A')}")

            # AI-specific clauses can be satisfied by replay / evidence-vault.
            anchors = self.required_evidence_types(para_key)
            if anchors:
                evidence_items.extend(f"Anchor: {a}" for a in anchors)

            if statuses:
                overall = "COMPLIANT" if all(statuses) else "NON-COMPLIANT"
            elif evidence_items:
                overall = "DOCUMENTED"
            else:
                overall = "NOT ASSESSED"

            evidence_str = "; ".join(evidence_items) if evidence_items else "No tests mapped"
            rows.append(
                f"| {para_key} | {para_info['title']} | {overall} | {evidence_str} |"
            )

        rows_str = "\n".join(rows)

        return f"""## 4. SR 26-2 Compliance Matrix

The following matrix maps each SR 26-2 requirement to the validation
evidence demonstrating compliance. AI-specific clauses (Section II.AI.*,
III.AI.*, IV.AI.C, V.*) are NEW relative to SR 11-7.

| SR 26-2 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
{rows_str}

### 4.1 Compliance Summary

Per SR 26-2, validation activities must provide a critical and
independent assessment of model performance. For AI systems, the
institution must additionally demonstrate **per-decision replay
capability** (II.AI.A / II.AI.B) and **tamper-evident evidence**
(III.AI.A / III.AI.B)."""

    def _detailed_test_results(self, test_results, paragraphs, test_mapping):
        sections = ["## 5. Detailed Validation Test Results\n"]

        for i, (test_name, result) in enumerate(test_results.items(), 1):
            passed = _result_passed(result)
            score = _result_score(result)
            details = _result_details(result)
            failure = _result_failure(result)
            para_ref = test_mapping.get(test_name, "N/A")
            para_info = paragraphs.get(para_ref, {})

            status_str = "PASS" if passed else "FAIL"

            section = f"""### 5.{i} {test_name}

| Field | Value |
|-------|-------|
| Status | **{status_str}** |
| Score | {f'{score:.4f}' if score is not None else 'N/A'} |
| SR 26-2 Reference | {para_ref}: {para_info.get('title', 'N/A')} |
"""

            if failure:
                section += f"| Failure Reason | {failure} |\n"

            if details:
                section += "\n**Validation Evidence:**\n\n"
                section += "```json\n"
                filtered = {
                    k: v for k, v in details.items()
                    if k not in ("sr26_2_reference", "compliance_references")
                }
                section += json.dumps(filtered, indent=2)
                section += "\n```\n"

            if para_info:
                section += f"\n**SR 26-2 Requirement:** {para_info.get('requirement', 'N/A')}\n"

            sections.append(section)

        return "\n".join(sections)

    def _trigger_section(self, triggers_cfg, trigger_events):
        if not triggers_cfg and not trigger_events:
            return """## 6. Validation Triggers

No validation triggers configured for this model."""

        cfg_summary = []
        if triggers_cfg:
            for trig in triggers_cfg:
                cfg_summary.append(
                    f"- **{trig.get('type', 'unknown')}**: {trig.get('description', 'N/A')}"
                )

        cfg_text = "\n".join(cfg_summary) if cfg_summary else "No triggers configured."

        event_text = "No trigger events at this time."
        if trigger_events:
            event_rows = []
            for evt in trigger_events:
                event_rows.append(
                    f"| {evt.get('trigger_type', 'N/A')} | "
                    f"{evt.get('timestamp', 'N/A')} | "
                    f"{evt.get('severity', 'N/A')} | "
                    f"{evt.get('description', 'N/A')} |"
                )
            event_text = (
                "| Trigger Type | Timestamp | Severity | Description |\n"
                "|--------------|-----------|----------|-------------|\n"
                + "\n".join(event_rows)
            )

        return f"""## 6. Validation Triggers (SR 26-2 Section II.B)

Per SR 26-2, models must be subject to ongoing monitoring with
triggers for re-validation when material changes occur, performance
degrades, or AI-specific risks (drift, prompt-injection, retrieval-
corpus change) breach configured thresholds.

### 6.1 Configured Triggers

{cfg_text}

### 6.2 Recent Trigger Events

{event_text}"""

    def _findings_and_recommendations(self, test_results, test_mapping):
        failed_tests = [
            (name, result) for name, result in test_results.items()
            if not _result_passed(result)
        ]

        if not failed_tests:
            return """## 7. Findings and Recommendations

### 7.1 Key Findings

All validation tests passed. The model demonstrates:

- Sound conceptual basis aligned with theory and intended use
- Accurate and stable outputs within expected ranges
- Appropriate sensitivity to key assumptions and parameters
- Effective ongoing monitoring and controls
- Where applicable, AI-specific risk thresholds satisfied

### 7.2 Recommendations

**Status: APPROVED FOR USE**

The model is approved for use subject to:

1. Ongoing monitoring per configured triggers
2. Re-validation at the scheduled frequency
3. Immediate re-validation if material changes occur
4. Documentation of any model overrides or adjustments
5. For AI systems: continued capture of decision records and
   periodic verification of evidence-chain integrity"""

        findings = []
        for i, (test_name, result) in enumerate(failed_tests, 1):
            failure = _result_failure(result)
            para_ref = test_mapping.get(test_name, "N/A")
            findings.append(
                f"{i}. **{test_name}** (SR 26-2 {para_ref}): {failure or 'Test failed'}"
            )

        findings_text = "\n".join(findings)

        return f"""## 7. Findings and Recommendations

### 7.1 Key Findings

The following validation tests failed and require remediation:

{findings_text}

### 7.2 Recommendations

**Status: REQUIRES REMEDIATION**

The model requires the following actions before approval:

1. **Immediate**: Address all failed validation tests
2. **Short-term**: Conduct root cause analysis for failures
3. **Medium-term**: Enhance model documentation per SR 26-2 Section IV
4. **Ongoing**: Implement enhanced monitoring per SR 26-2 Section II.B
5. **For AI systems**: ensure decision records are being captured for
   every inference (Section II.AI.A) and evidence packets are stored
   in a tamper-evident form (Section III.AI.A)

### 7.3 Compensating Controls

Until remediation is complete, the following compensating controls
should be implemented:

- Enhanced model output review by senior model risk personnel
- Restrictions on model use for high-impact decisions
- Increased monitoring frequency
- Documentation of all model overrides
- For AI systems: human-in-the-loop review of in-scope decisions"""

    def _approval_section(self, model_info, now):
        return f"""## 8. Approval and Sign-Off

Per SR 26-2 Section III, model validation results must be reported to
senior management and the board of directors. Model owners must
address validation findings and document any remediation actions.

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Owner | {model_info.get('owner', '_______________')} | _______________ | {now.strftime('%Y-%m-%d')} |
| Independent Validator | MRM Team | _______________ | {now.strftime('%Y-%m-%d')} |
| Model Risk Manager | _______________ | _______________ | _______________ |
| Chief Risk Officer | _______________ | _______________ | _______________ |

---

*This report was generated by* ``mrm-core`` *in accordance with
{self.display_name}. AI-specific clauses are anchored to the
DecisionRecord (`docs/spec/replay-record-v1.md`) and EvidencePacket
(`docs/spec/evidence-vault-v1.md`) primitives.*"""
