"""OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management.

Bundled compliance standard for the Office of the Superintendent of
Financial Institutions (Canada). Implements ``ComplianceStandard`` so
the MRM framework can generate OSFI E-23 reports, run governance checks,
and map tests to specific regulatory sections.

Usage::

    mrm docs generate ccr_monte_carlo --compliance standard:osfie23
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mrm.compliance.base import ComplianceStandard
from mrm.compliance.registry import register_standard

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Numpy type conversion helper
# -----------------------------------------------------------------------

def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# -----------------------------------------------------------------------
# Helpers (result introspection)
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


def _get_compliance_ref(details: Dict, standard_name: str = "osfie23") -> str:
    """Read compliance reference from test result details.

    Supports both the new generic key and the legacy key:
      - ``details["compliance_references"]["osfie23"]``  (new)
      - ``details["osfie23_reference"]``                 (legacy)
    """
    refs = details.get("compliance_references", {})
    if isinstance(refs, dict) and standard_name in refs:
        return refs[standard_name]
    return details.get("osfie23_reference", "")


# =======================================================================
# OSFI E-23 Standard Implementation
# =======================================================================

@register_standard
class OSFIE23Standard(ComplianceStandard):
    """OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management."""

    name = "osfie23"
    display_name = "OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management"
    description = (
        "OSFI E-23 provides guidance on enterprise-wide model risk management "
        "for federally regulated financial institutions (FRFIs) in Canada. "
        "It establishes expectations for model identification, governance, "
        "development, validation, monitoring, and documentation."
    )
    jurisdiction = "CA"
    version = "2023"

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        """Map OSFI E-23 sections to requirements.
        
        Structure follows OSFI's seven main sections:
        1. Introduction and Scope
        2. Model Identification and Risk Tiering
        3. Model Governance
        4. Model Development
        5. Model Validation
        6. Ongoing Model Monitoring
        7. Model Documentation and Inventory
        """
        return {
            "Section 2.1": {
                "title": "Model Identification",
                "requirement": (
                    "FRFIs should identify all models used in business operations, "
                    "including those used for decision-making, risk management, "
                    "capital adequacy assessment, and financial reporting. The "
                    "identification process should be comprehensive and systematic."
                ),
            },
            "Section 2.2": {
                "title": "Risk Tiering and Materiality Assessment",
                "requirement": (
                    "Models should be assigned to risk tiers based on their potential "
                    "impact, complexity, and uncertainty. Higher risk models require "
                    "more rigorous validation and oversight. Risk tiering should "
                    "consider quantitative and qualitative factors including model "
                    "complexity, data availability, and potential financial impact."
                ),
            },
            "Section 3.1": {
                "title": "Governance Framework",
                "requirement": (
                    "FRFIs should establish a comprehensive model risk management "
                    "framework with clear policies, procedures, and controls. The "
                    "framework should cover all aspects of the model lifecycle "
                    "from development through retirement."
                ),
            },
            "Section 3.2": {
                "title": "Roles and Accountability",
                "requirement": (
                    "Clear roles, responsibilities, and accountabilities should be "
                    "defined for model owners, developers, validators, and users. "
                    "Model owners are accountable for model performance, proper use, "
                    "and adherence to policies. Independent validation function "
                    "should be separate from model development and use."
                ),
            },
            "Section 3.3": {
                "title": "Senior Management and Board Oversight",
                "requirement": (
                    "Senior management and the Board should provide effective oversight "
                    "of model risk. This includes understanding material model risks, "
                    "reviewing model risk management practices, and ensuring adequate "
                    "resources are allocated to model risk management activities."
                ),
            },
            "Section 3.4": {
                "title": "Three Lines of Defence",
                "requirement": (
                    "FRFIs should implement a three lines of defence approach for "
                    "model risk management: (1) model owners and users (first line), "
                    "(2) independent model validation and risk management (second line), "
                    "and (3) internal audit (third line)."
                ),
            },
            "Section 4.1": {
                "title": "Model Design and Theory",
                "requirement": (
                    "Models should be based on sound theoretical foundations appropriate "
                    "for their intended use. Design choices should be well-justified "
                    "and documented. Model assumptions should be reasonable, clearly "
                    "stated, and their impact understood."
                ),
            },
            "Section 4.2": {
                "title": "Data Quality and Governance",
                "requirement": (
                    "Model development should use high-quality data that is accurate, "
                    "complete, and representative of the intended application. Data "
                    "limitations should be identified and their impact assessed. FRFIs "
                    "should have robust data governance processes supporting model use."
                ),
            },
            "Section 4.3": {
                "title": "Model Implementation and Testing",
                "requirement": (
                    "Models should be implemented correctly with appropriate controls "
                    "and testing. Initial development testing should verify model "
                    "logic, calculations, and outputs are accurate and perform as "
                    "intended under various scenarios."
                ),
            },
            "Section 5.1": {
                "title": "Independent Validation",
                "requirement": (
                    "All material models should undergo independent validation before "
                    "use and periodically thereafter. Validation should be performed "
                    "by qualified staff independent of model development and use, with "
                    "sufficient expertise to assess model soundness."
                ),
            },
            "Section 5.2": {
                "title": "Validation Scope and Activities",
                "requirement": (
                    "Validation should be comprehensive and include: (1) evaluation "
                    "of conceptual soundness, (2) ongoing monitoring and process "
                    "verification, (3) outcomes analysis and backtesting, and "
                    "(4) sensitivity analysis and stress testing."
                ),
            },
            "Section 5.3": {
                "title": "Conceptual Soundness Review",
                "requirement": (
                    "Validation should assess whether model theory, assumptions, and "
                    "methodology are appropriate for the intended use. This includes "
                    "review of mathematical construction, logical framework, and "
                    "alignment with sound practices and academic research."
                ),
            },
            "Section 5.4": {
                "title": "Outcomes Analysis and Backtesting",
                "requirement": (
                    "Where feasible, model predictions should be compared with actual "
                    "outcomes to assess accuracy and identify systematic biases. "
                    "Backtesting should evaluate model performance over time and under "
                    "various market conditions."
                ),
            },
            "Section 5.5": {
                "title": "Sensitivity Analysis and Scenario Testing",
                "requirement": (
                    "Validation should include comprehensive sensitivity analysis to "
                    "evaluate model behavior under different assumptions, inputs, and "
                    "scenarios. Stress testing should assess model performance under "
                    "adverse conditions."
                ),
            },
            "Section 6.1": {
                "title": "Ongoing Monitoring Framework",
                "requirement": (
                    "FRFIs should establish ongoing monitoring processes to track "
                    "model performance, validate continuing appropriateness, and "
                    "identify emerging risks. Monitoring should be proportionate to "
                    "model risk tier and materiality."
                ),
            },
            "Section 6.2": {
                "title": "Performance Monitoring and Alerting",
                "requirement": (
                    "Model performance should be monitored on an ongoing basis with "
                    "clear metrics, thresholds, and escalation procedures. Monitoring "
                    "should cover inputs, processing logic, and outputs, with alerts "
                    "for deviations from expectations."
                ),
            },
            "Section 6.3": {
                "title": "Triggers for Revalidation",
                "requirement": (
                    "FRFIs should define triggers that require model revalidation, "
                    "including: material changes to model logic or assumptions, "
                    "significant market or business changes, breaches of performance "
                    "thresholds, and regulatory changes affecting model use."
                ),
            },
            "Section 7.1": {
                "title": "Model Documentation Standards",
                "requirement": (
                    "All models should have comprehensive documentation covering "
                    "model design, theory, assumptions, limitations, and appropriate "
                    "use. Documentation should be sufficient for a qualified third "
                    "party to understand and evaluate the model."
                ),
            },
            "Section 7.2": {
                "title": "Model Inventory and Tracking",
                "requirement": (
                    "FRFIs should maintain a complete model inventory that tracks all "
                    "models, their risk tiers, validation status, ownership, and key "
                    "metadata. The inventory should be kept current and accessible to "
                    "relevant stakeholders."
                ),
            },
            "Section 7.3": {
                "title": "Version Control and Change Management",
                "requirement": (
                    "Model changes should be subject to formal change management "
                    "processes with appropriate documentation, testing, and approval. "
                    "Version control should track model changes and their impact on "
                    "model performance and risk."
                ),
            },
        }

    def get_test_mapping(self) -> Dict[str, str]:
        """Map test names to OSFI E-23 sections."""
        return {
            # CCR tests mapping to OSFI E-23 sections
            "ccr.MCConvergence": "Section 4.3",
            "ccr.EPEReasonableness": "Section 5.3",
            "ccr.PFEBacktest": "Section 5.4",
            "ccr.CVASensitivity": "Section 5.5",
            "ccr.WrongWayRisk": "Section 5.3",
            "ccr.ExposureProfileShape": "Section 5.3",
            "ccr.CollateralEffectiveness": "Section 5.4",
            "ccr.CPS230GovernanceCheck": "Section 3.1",
            "compliance.GovernanceCheck": "Section 3.1",
            # Tabular tests mapping
            "tabular_dataset.MissingValues": "Section 4.2",
            "tabular_dataset.ClassImbalance": "Section 4.2",
            "tabular_dataset.OutlierDetection": "Section 4.2",
            "tabular_dataset.FeatureDistribution": "Section 6.2",
            # Model performance tests
            "model.Accuracy": "Section 5.4",
            "model.ROCAUC": "Section 5.4",
            "model.Precision": "Section 5.4",
            "model.Recall": "Section 5.4",
            "model.F1Score": "Section 5.4",
            "model.Gini": "Section 5.4",
        }

    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        """Define OSFI E-23 governance requirements."""
        return {
            "model_identified": {
                "description": "Model identified in comprehensive inventory",
                "paragraph_ref": "OSFI E-23 Section 2.1",
                "config_key": "name",
            },
            "risk_tier_assigned": {
                "description": "Risk tier assigned based on impact, complexity, and uncertainty",
                "paragraph_ref": "OSFI E-23 Section 2.2",
                "config_key": "risk_tier",
            },
            "owner_designated": {
                "description": "Model owner designated and accountable",
                "paragraph_ref": "OSFI E-23 Section 3.2",
                "config_key": "owner",
            },
            "use_case_documented": {
                "description": "Model purpose and intended use documented",
                "paragraph_ref": "OSFI E-23 Section 7.1",
                "config_key": "use_case",
            },
            "methodology_documented": {
                "description": "Model theory and methodology documented",
                "paragraph_ref": "OSFI E-23 Section 4.1",
                "config_key": "methodology",
            },
            "assumptions_documented": {
                "description": "Key assumptions documented and justified",
                "paragraph_ref": "OSFI E-23 Section 4.1",
                "config_key": "assumptions",
            },
            "validation_frequency_set": {
                "description": "Validation frequency commensurate with risk tier",
                "paragraph_ref": "OSFI E-23 Section 5.1",
                "config_key": "validation_frequency",
            },
            "version_controlled": {
                "description": "Version control and change management in place",
                "paragraph_ref": "OSFI E-23 Section 7.3",
                "config_key": "version",
            },
        }

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
        """Generate OSFI E-23 validation report."""
        now = datetime.now(timezone.utc)
        model_info = model_config.get("model", model_config)

        # Read compliance mapping
        osfie23_mapping = (
            model_config
            .get("compliance", {})
            .get("standards", {})
            .get("osfie23", {})
            .get("mapping", model_config.get("osfie23_mapping", {}))
        )
        triggers_cfg = model_config.get("triggers", [])

        paragraphs = self.get_paragraphs()
        test_mapping = self.get_test_mapping()

        sections = [
            self._header(model_name, model_info, now),
            self._executive_summary(model_name, model_info, test_results, now),
            self._model_inventory_card(model_info, model_config),
            self._compliance_matrix(test_results, osfie23_mapping, paragraphs, test_mapping),
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
        return f"""# OSFI E-23 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | {model_name} |
| **Version** | {model_info.get('version', 'N/A')} |
| **Report Date** | {now.strftime('%Y-%m-%d %H:%M UTC')} |
| **Regulatory Framework** | {self.display_name} |
| **Jurisdiction** | Canada (All FRFIs) |
| **Risk Tier** | {model_info.get('risk_tier', 'N/A')} |
| **Owner** | {model_info.get('owner', 'N/A')} |
| **Validation Frequency** | {model_info.get('validation_frequency', 'N/A')} |
| **Independent Validator** | MRM Validation Team |
| **Effective Date** | May 1, 2027 |

---"""

    def _executive_summary(self, model_name, model_info, test_results, now):
        total = len(test_results)
        passed = sum(1 for r in test_results.values() if _result_passed(r))
        failed = total - passed
        pass_rate = f"{passed / total * 100:.1f}%" if total > 0 else "N/A"
        status_marker = "APPROVED FOR USE" if failed == 0 and total > 0 else "REQUIRES REMEDIATION"
        status = "PASS" if failed == 0 and total > 0 else "FAIL"

        return f"""## 1. Executive Summary

This report presents the independent validation results for the
**{model_name}** model (v{model_info.get('version', '1.0.0')})
conducted on {now.strftime('%Y-%m-%d')}.

The validation was performed in accordance with **{self.display_name}**
(effective May 1, 2027). This guideline applies to all federally
regulated financial institutions (FRFIs) in Canada, including banks,
insurance companies, and trust and loan companies.

The model is classified as
**{model_info.get('risk_tier', 'tier_1').replace('_', ' ').title()}**
based on OSFI E-23 Section 2.2 risk tiering criteria (impact, complexity,
and uncertainty), requiring {model_info.get('validation_frequency', 'annual')}
independent validation.

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

**Business Line:** {model_info.get('business_line', 'N/A')}

### Validation Scope (OSFI E-23 Section 5.2)

Per OSFI E-23, this independent validation includes:
- **Conceptual Soundness (Section 5.3)**: Review of model theory, assumptions, and methodology
- **Outcomes Analysis (Section 5.4)**: Backtesting and comparison with actual outcomes
- **Sensitivity Analysis (Section 5.5)**: Testing under various assumptions and stress scenarios
- **Ongoing Monitoring (Section 6)**: Review of performance monitoring framework

### Three Lines of Defence (OSFI E-23 Section 3.4)

This validation represents the **second line of defence** review:
- **First Line**: Model owner ({model_info.get('owner', 'N/A')}) responsible for model performance and proper use
- **Second Line**: Independent validation team (MRM) conducted this review
- **Third Line**: Internal audit review pending"""

    def _model_inventory_card(self, model_info, model_config):
        params = model_info.get("parameters", model_config.get("parameters", {}))
        param_rows = "\n".join(
            f"| {k} | {v} |" for k, v in params.items()
        ) if params else "| N/A | N/A |"

        assumptions = model_info.get("assumptions", "Not documented")
        if isinstance(assumptions, list):
            assumptions = "; ".join(assumptions)

        limitations = model_info.get("limitations", "Not explicitly documented")
        if isinstance(limitations, list):
            limitations = "; ".join(limitations)

        return f"""## 2. Model Inventory Card

### 2.1 Model Identification (OSFI E-23 Section 2.1 & 7.2)

| Field | Value |
|-------|-------|
| Model Name | {model_info.get('name', 'N/A')} |
| Version | {model_info.get('version', 'N/A')} |
| Model Owner | {model_info.get('owner', 'N/A')} |
| Business Line | {model_info.get('business_line', 'N/A')} |
| Use Case | {model_info.get('use_case', 'N/A')} |
| Methodology | {model_info.get('methodology', 'N/A')} |
| Model Type | {model_info.get('model_type', 'Quantitative')} |

### 2.2 Risk Tiering (OSFI E-23 Section 2.2)

| Field | Value |
|-------|-------|
| Risk Tier | {model_info.get('risk_tier', 'N/A')} |
| Materiality | {model_info.get('materiality', 'N/A')} |
| Complexity | {model_info.get('complexity', 'N/A')} |
| Validation Frequency | {model_info.get('validation_frequency', 'N/A')} |

**Risk Tier Rationale**: This model is classified as Tier 1 (highest risk)
based on its potential impact on capital adequacy, financial reporting, and
risk management decisions. The classification considers:
- **Impact**: Material financial and regulatory impact
- **Complexity**: Sophisticated methodology requiring specialized expertise
- **Uncertainty**: Significant model risk from assumptions and data limitations

### 2.3 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
{param_rows}

### 2.4 Key Assumptions (OSFI E-23 Section 4.1)

{assumptions}

### 2.5 Known Limitations (OSFI E-23 Section 7.1)

{limitations}

### 2.6 Governance and Accountability (OSFI E-23 Section 3.2)

| Role | Name/Team |
|------|-----------|
| Model Owner | {model_info.get('owner', 'N/A')} |
| Model Developer | {model_info.get('developer', 'N/A')} |
| Independent Validator | MRM Validation Team |
| Senior Management Sponsor | {model_info.get('sponsor', 'N/A')} |"""

    def _compliance_matrix(self, test_results, osfie23_mapping, paragraphs, test_mapping):
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

            for section_items in osfie23_mapping.values():
                if isinstance(section_items, list):
                    for item in section_items:
                        if item.get("section") == para_key:
                            evidence_items.append(f"Config: {item.get('evidence', 'N/A')}")

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

        return f"""## 3. OSFI E-23 Compliance Matrix

The following matrix maps each OSFI E-23 requirement to the validation
evidence demonstrating compliance.

| Section | Requirement | Status | Evidence |
|---------|-------------|--------|----------|
{rows_str}

**Status Legend:**
- **COMPLIANT**: All tests passed, requirement fully met
- **NON-COMPLIANT**: One or more tests failed
- **DOCUMENTED**: Evidence provided through configuration, no quantitative tests
- **NOT ASSESSED**: Requirement not tested in this validation cycle"""

    def _detailed_test_results(self, test_results, paragraphs, test_mapping):
        if not test_results:
            return "## 4. Detailed Test Results\n\nNo tests executed."

        lines = ["## 4. Detailed Test Results"]
        lines.append("")
        lines.append("Per OSFI E-23 Section 5.2, validation activities include conceptual")
        lines.append("soundness review, outcomes analysis, and sensitivity testing.")
        lines.append("")

        for test_name, result in test_results.items():
            section = test_mapping.get(test_name, "General")
            section_title = paragraphs.get(section, {}).get("title", "General Validation")

            passed = _result_passed(result)
            score = _result_score(result)
            details = _result_details(result)
            failure = _result_failure(result)

            status_marker = "✅ PASS" if passed else "❌ FAIL"
            score_str = f" (score: {score:.4f})" if score is not None else ""

            lines.append(f"### {test_name} {status_marker}{score_str}")
            lines.append(f"**OSFI E-23 Reference:** {section} -- {section_title}")
            lines.append("")

            if not passed and failure:
                lines.append(f"**Failure Reason:** {failure}")
                lines.append("")

            if details:
                lines.append("**Test Details:**")
                lines.append("```json")
                # Convert numpy types to JSON-serializable types
                serializable_details = _convert_to_json_serializable(details)
                lines.append(json.dumps(serializable_details, indent=2))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _trigger_section(self, triggers_cfg, trigger_events):
        lines = ["## 5. Revalidation Triggers (OSFI E-23 Section 6.3)"]
        lines.append("")
        lines.append("OSFI E-23 Section 6.3 requires FRFIs to define triggers that")
        lines.append("require model revalidation, including:")
        lines.append("- Material changes to model logic or assumptions")
        lines.append("- Significant market or business changes")
        lines.append("- Breaches of performance thresholds")
        lines.append("- Regulatory changes affecting model use")
        lines.append("")

        if not triggers_cfg:
            lines.append("**No triggers configured for this model.**")
            return "\n".join(lines)

        lines.append("### Configured Triggers")
        lines.append("")
        lines.append("| Trigger Type | Description | Threshold | Status |")
        lines.append("|--------------|-------------|-----------|--------|")

        for trigger in triggers_cfg:
            trig_type = trigger.get("type", "N/A")
            desc = trigger.get("description", "N/A")
            threshold = trigger.get("threshold", "N/A")

            fired = False
            if trigger_events:
                fired = any(
                    e.get("type") == trig_type
                    for e in trigger_events
                    if e.get("status") == "fired"
                )

            status = "🔥 FIRED" if fired else "✓ Active"
            lines.append(f"| {trig_type} | {desc} | {threshold} | {status} |")

        if trigger_events:
            lines.append("")
            lines.append("### Fired Trigger Events")
            lines.append("")
            for event in trigger_events:
                if event.get("status") == "fired":
                    lines.append(f"- **{event.get('type')}**: {event.get('description')}")
                    lines.append(f"  - Compliance Reference: {event.get('compliance_reference', 'N/A')}")

        return "\n".join(lines)

    def _findings_and_recommendations(self, test_results, test_mapping):
        lines = ["## 6. Findings and Recommendations"]
        lines.append("")

        failed_tests = [name for name, r in test_results.items() if not _result_passed(r)]

        if not failed_tests:
            lines.append("### Summary")
            lines.append("")
            lines.append("All validation tests passed. The model demonstrates:")
            lines.append("- Sound theoretical foundations (OSFI E-23 Section 4.1)")
            lines.append("- Adequate data quality and governance (OSFI E-23 Section 4.2)")
            lines.append("- Satisfactory performance and accuracy (OSFI E-23 Section 5.4)")
            lines.append("- Appropriate sensitivity under stress scenarios (OSFI E-23 Section 5.5)")
            lines.append("")
            lines.append("**Validation Conclusion:** Model is **APPROVED FOR USE** subject to")
            lines.append("ongoing monitoring per OSFI E-23 Section 6.")
        else:
            lines.append("### Critical Findings")
            lines.append("")
            lines.append(f"**{len(failed_tests)} validation test(s) failed**, indicating")
            lines.append("potential model risk issues requiring remediation:")
            lines.append("")

            for test_name in failed_tests:
                result = test_results[test_name]
                section = test_mapping.get(test_name, "General")
                failure = _result_failure(result)
                lines.append(f"- **{test_name}** (OSFI E-23 {section})")
                if failure:
                    lines.append(f"  - Issue: {failure}")
                lines.append("")

            lines.append("### Recommendations")
            lines.append("")
            lines.append("Per OSFI E-23 Section 5.1, the following actions are recommended:")
            lines.append("")
            lines.append("1. **Model owners should address all failed tests** within the")
            lines.append("   remediation period specified in the FRFI's model risk policy.")
            lines.append("2. **Revalidation required** after remediation to confirm issues")
            lines.append("   are resolved (OSFI E-23 Section 6.3).")
            lines.append("3. **Model use should be restricted** pending remediation, with")
            lines.append("   escalation to senior management if issues are material.")
            lines.append("4. **Enhanced monitoring** of model outputs during the remediation")
            lines.append("   period (OSFI E-23 Section 6.2).")
            lines.append("")
            lines.append("**Validation Conclusion:** Model **REQUIRES REMEDIATION** before")
            lines.append("approval for production use.")

        return "\n".join(lines)

    def _approval_section(self, model_info, now):
        return f"""## 7. Validation Sign-Off (OSFI E-23 Section 5.1)

Per OSFI E-23 Section 5.1, independent validation should be performed by
qualified staff with sufficient expertise to assess model soundness.

| Role | Name | Date | Signature |
|------|------|------|-----------|
| **Independent Validator** | [Name] | {now.strftime('%Y-%m-%d')} | _______________ |
| **Senior Model Risk Officer** | [Name] | __________ | _______________ |
| **Model Owner** | {model_info.get('owner', '[Name]')} | __________ | _______________ |

---

**Report Generated By:** mrm-core v1.0.0  
**Framework:** {self.display_name}  
**Report Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}

---

### Regulatory References

- OSFI Guideline E-23: Enterprise-Wide Model Risk Management (January 2023)
- OSFI Effective Date: May 1, 2027 (expanded scope to all FRFIs)
- NIST AI Risk Management Framework (cross-reference where applicable)
- AMF Guideline on Sound Business and Financial Practices (Quebec)"""
