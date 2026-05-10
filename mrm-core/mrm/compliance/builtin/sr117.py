"""Federal Reserve SR 11-7 -- Guidance on Model Risk Management.

Bundled compliance standard for the United States Federal Reserve.
Implements ``ComplianceStandard`` so the MRM framework can generate
SR 11-7 reports, run governance checks, and map tests to specific
regulatory sections.

Usage::

    mrm docs generate ccr_monte_carlo --compliance standard:sr117
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


def _get_compliance_ref(details: Dict, standard_name: str = "sr117") -> str:
    """Read compliance reference from test result details.

    Supports both the new generic key and the legacy key:
      - ``details["compliance_references"]["sr117"]``  (new)
      - ``details["sr117_reference"]``                 (legacy)
    """
    refs = details.get("compliance_references", {})
    if isinstance(refs, dict) and standard_name in refs:
        return refs[standard_name]
    return details.get("sr117_reference", "")


# =======================================================================
# SR 11-7 Standard Implementation
# =======================================================================

@register_standard
class SR117Standard(ComplianceStandard):
    """Federal Reserve SR 11-7 -- Guidance on Model Risk Management."""

    name = "sr117"
    display_name = "Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management"
    description = (
        "SR 11-7 provides guidance on model risk management for banking "
        "organizations supervised by the Federal Reserve. It establishes "
        "comprehensive model risk management practices covering model "
        "development, validation, and governance."
    )
    jurisdiction = "US"
    version = "2011"

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        """Map SR 11-7 sections to requirements.
        
        Structure follows the Federal Reserve's four main sections:
        - Section I: Model Development, Implementation, and Use
        - Section II: Model Validation  
        - Section III: Governance, Policies, and Controls
        - Section IV: Model Inventory
        """
        return {
            "Section I.A": {
                "title": "Model Development and Implementation",
                "requirement": (
                    "Models should be based on sound theory that is consistent with "
                    "their intended use. Design choices should be aligned with the "
                    "model's purpose and appropriately balance complexity with "
                    "interpretability. Development should include rigorous assessment "
                    "of data quality and representativeness."
                ),
            },
            "Section I.B": {
                "title": "Model Use and Purpose",
                "requirement": (
                    "The intended use of the model should be clearly documented, "
                    "including the business purpose, key assumptions, and intended "
                    "application. Users should understand the model's limitations "
                    "and appropriate use constraints."
                ),
            },
            "Section I.C": {
                "title": "Model Assumptions and Data",
                "requirement": (
                    "Model assumptions should be reasonable and supportable. Data used "
                    "for model development and ongoing use should be comprehensive, "
                    "accurate, and appropriate for the model's purpose. Data limitations "
                    "should be identified and their impact assessed."
                ),
            },
            "Section I.D": {
                "title": "Model Outputs and Reporting",
                "requirement": (
                    "Model outputs should be subject to appropriate analysis including "
                    "post-model adjustments, if any. Outputs should be reported with "
                    "sufficient detail and context to support informed decision-making."
                ),
            },
            "Section II.A": {
                "title": "Validation Scope and Objectives",
                "requirement": (
                    "Model validation should be comprehensive and include evaluation of "
                    "conceptual soundness, ongoing monitoring, and outcomes analysis. "
                    "Validation should be independent of model development and use."
                ),
            },
            "Section II.B": {
                "title": "Evaluation of Conceptual Soundness",
                "requirement": (
                    "Validation should assess whether the model design and construction "
                    "are consistent with sound theory and judgment, and appropriate for "
                    "the intended use. This includes review of model assumptions, "
                    "mathematical construction, and theoretical basis."
                ),
            },
            "Section II.C": {
                "title": "Ongoing Monitoring",
                "requirement": (
                    "Models should be subject to ongoing monitoring to evaluate model "
                    "performance and stability. Monitoring should include tracking of "
                    "model inputs, processing logic, and outputs, with comparison to "
                    "expectations and benchmarks."
                ),
            },
            "Section II.D": {
                "title": "Outcomes Analysis and Backtesting",
                "requirement": (
                    "Model outcomes should be compared with actual outcomes to assess "
                    "model accuracy and identify potential issues. Backtesting should "
                    "be performed where feasible, with analysis of errors and systematic "
                    "biases."
                ),
            },
            "Section II.E": {
                "title": "Sensitivity Analysis and Stress Testing",
                "requirement": (
                    "Validation should include comprehensive sensitivity analysis and "
                    "stress testing to evaluate model behavior under a range of "
                    "assumptions and scenarios, particularly adverse conditions."
                ),
            },
            "Section III.A": {
                "title": "Model Risk Management Framework",
                "requirement": (
                    "Banks should have a sound model risk management framework that "
                    "provides well-defined and consistent policies, procedures, and "
                    "controls covering all aspects of the model risk management process."
                ),
            },
            "Section III.B": {
                "title": "Roles and Responsibilities",
                "requirement": (
                    "Clear roles and responsibilities should be established for model "
                    "ownership, development, validation, and approval. Model owners "
                    "should be accountable for model performance and proper use."
                ),
            },
            "Section III.C": {
                "title": "Board and Senior Management Oversight",
                "requirement": (
                    "The board of directors and senior management should understand the "
                    "risks inherent in models, the institution's model risk management "
                    "framework, and provide effective challenge to the model risk "
                    "management process."
                ),
            },
            "Section III.D": {
                "title": "Validation Frequency and Coverage",
                "requirement": (
                    "Model validation should occur on a frequency commensurate with the "
                    "model's risk tier and materiality. Material models should be "
                    "validated prior to use and revalidated periodically or when changes "
                    "occur."
                ),
            },
            "Section IV": {
                "title": "Model Inventory and Documentation",
                "requirement": (
                    "Banks should maintain a comprehensive model inventory that tracks "
                    "all models subject to model risk management. Documentation should "
                    "be comprehensive and provide sufficient detail for effective "
                    "validation and ongoing use."
                ),
            },
        }

    def get_test_mapping(self) -> Dict[str, str]:
        """Map test names to SR 11-7 sections."""
        return {
            # CCR tests mapping to SR 11-7 sections
            "ccr.MCConvergence": "Section II.B",
            "ccr.EPEReasonableness": "Section II.B",
            "ccr.PFEBacktest": "Section II.D",
            "ccr.CVASensitivity": "Section II.E",
            "ccr.WrongWayRisk": "Section II.B",
            "ccr.ExposureProfileShape": "Section II.B",
            "ccr.CollateralEffectiveness": "Section II.D",
            "ccr.CPS230GovernanceCheck": "Section III.A",
            "compliance.GovernanceCheck": "Section III.A",
            # Tabular tests mapping
            "tabular.MissingValues": "Section I.C",
            "tabular.DataDrift": "Section II.C",
            "tabular.OutlierDetection": "Section I.C",
        }

    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        """Define SR 11-7 governance requirements."""
        return {
            "risk_tier_assigned": {
                "description": "Risk tier must be assigned based on materiality and complexity",
                "paragraph_ref": "SR 11-7 Section III.D",
                "config_key": "risk_tier",
            },
            "owner_designated": {
                "description": "Model owner / accountability designated",
                "paragraph_ref": "SR 11-7 Section III.B",
                "config_key": "owner",
            },
            "validation_frequency_set": {
                "description": "Validation frequency defined commensurate with risk",
                "paragraph_ref": "SR 11-7 Section III.D",
                "config_key": "validation_frequency",
            },
            "use_case_documented": {
                "description": "Model purpose and intended use documented",
                "paragraph_ref": "SR 11-7 Section I.B",
                "config_key": "use_case",
            },
            "methodology_documented": {
                "description": "Model methodology and theory documented",
                "paragraph_ref": "SR 11-7 Section I.A",
                "config_key": "methodology",
            },
            "assumptions_documented": {
                "description": "Key assumptions documented and justified",
                "paragraph_ref": "SR 11-7 Section I.C",
                "config_key": "assumptions",
            },
            "model_inventory_maintained": {
                "description": "Model tracked in comprehensive inventory",
                "paragraph_ref": "SR 11-7 Section IV",
                "config_key": "name",
            },
            "version_controlled": {
                "description": "Version control and change management in place",
                "paragraph_ref": "SR 11-7 Section IV",
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
        """Generate SR 11-7 validation report."""
        now = datetime.now(timezone.utc)
        model_info = model_config.get("model", model_config)

        # Read compliance mapping
        sr117_mapping = (
            model_config
            .get("compliance", {})
            .get("standards", {})
            .get("sr117", {})
            .get("mapping", model_config.get("sr117_mapping", {}))
        )
        triggers_cfg = model_config.get("triggers", [])

        paragraphs = self.get_paragraphs()
        test_mapping = self.get_test_mapping()

        sections = [
            self._header(model_name, model_info, now),
            self._executive_summary(model_name, model_info, test_results, now),
            self._model_inventory_card(model_info, model_config),
            self._compliance_matrix(test_results, sr117_mapping, paragraphs, test_mapping),
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
        return f"""# SR 11-7 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | {model_name} |
| **Version** | {model_info.get('version', 'N/A')} |
| **Report Date** | {now.strftime('%Y-%m-%d %H:%M UTC')} |
| **Regulatory Framework** | {self.display_name} |
| **Risk Tier** | {model_info.get('risk_tier', 'N/A')} |
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
requirements. The model is classified as
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

Per SR 11-7 Section II, this validation includes:
- **Conceptual Soundness**: Review of model theory, assumptions, and structure
- **Ongoing Monitoring**: Analysis of model inputs, outputs, and stability
- **Outcomes Analysis**: Backtesting where applicable
- **Sensitivity Analysis**: Testing under various scenarios and stress conditions"""

    def _model_inventory_card(self, model_info, model_config):
        params = model_info.get("parameters", model_config.get("parameters", {}))
        param_rows = "\n".join(
            f"| {k} | {v} |" for k, v in params.items()
        ) if params else "| N/A | N/A |"

        assumptions = model_info.get("assumptions", "Not documented")
        if isinstance(assumptions, list):
            assumptions = "; ".join(assumptions)

        return f"""## 2. Model Inventory Card

### 2.1 Identification (SR 11-7 Section IV)

| Field | Value |
|-------|-------|
| Model Name | {model_info.get('name', 'N/A')} |
| Version | {model_info.get('version', 'N/A')} |
| Owner | {model_info.get('owner', 'N/A')} |
| Business Line | {model_info.get('business_line', 'N/A')} |
| Use Case | {model_info.get('use_case', 'N/A')} |
| Methodology | {model_info.get('methodology', 'N/A')} |
| Risk Tier | {model_info.get('risk_tier', 'N/A')} |
| Materiality | {model_info.get('materiality', 'N/A')} |
| Complexity | {model_info.get('complexity', 'N/A')} |
| Validation Frequency | {model_info.get('validation_frequency', 'N/A')} |

### 2.2 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
{param_rows}

### 2.3 Key Assumptions (SR 11-7 Section I.C)

{assumptions}

### 2.4 Risk Tier Classification Rationale

Per **SR 11-7 Section III.D**, models should be validated on a frequency
commensurate with their risk and materiality. This model is classified as
Tier 1 because:

- It produces material financial impacts (capital requirements, risk metrics)
- It uses complex methodologies requiring specialized expertise
- Errors could result in material misstatement of risk or capital positions
- The model is used in decision-making with significant consequences"""

    def _compliance_matrix(self, test_results, sr117_mapping, paragraphs, test_mapping):
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

            for section_items in sr117_mapping.values():
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

        return f"""## 3. SR 11-7 Compliance Matrix

The following matrix maps each SR 11-7 requirement to the validation
evidence demonstrating compliance.

| SR 11-7 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
{rows_str}

### 3.1 Compliance Summary

Each row above corresponds to a specific section of Federal Reserve SR 11-7.
Tests are designed to provide quantitative evidence that the model satisfies
the supervisory expectations for model risk management. Where a requirement
is marked "COMPLIANT", the corresponding validation test has passed with
results within acceptable thresholds.

Per SR 11-7, validation activities should provide a critical and independent
assessment of model performance. This validation was conducted by personnel
independent of model development and use."""

    def _detailed_test_results(self, test_results, paragraphs, test_mapping):
        sections = ["## 4. Detailed Validation Test Results\n"]

        for i, (test_name, result) in enumerate(test_results.items(), 1):
            passed = _result_passed(result)
            score = _result_score(result)
            details = _result_details(result)
            failure = _result_failure(result)
            para_ref = test_mapping.get(test_name, "N/A")
            para_info = paragraphs.get(para_ref, {})

            status_str = "PASS" if passed else "FAIL"

            section = f"""### 4.{i} {test_name}

| Field | Value |
|-------|-------|
| Status | **{status_str}** |
| Score | {f'{score:.4f}' if score is not None else 'N/A'} |
| SR 11-7 Reference | {para_ref}: {para_info.get('title', 'N/A')} |
"""

            if failure:
                section += f"| Failure Reason | {failure} |\n"

            if details:
                section += "\n**Validation Evidence:**\n\n"
                section += "```json\n"
                filtered = {
                    k: v for k, v in details.items()
                    if k not in ("sr117_reference", "compliance_references")
                }
                section += json.dumps(filtered, indent=2)
                section += "\n```\n"

            if para_info:
                section += f"\n**SR 11-7 Requirement:** {para_info.get('requirement', 'N/A')}\n"

            sections.append(section)

        return "\n".join(sections)

    def _trigger_section(self, triggers_cfg, trigger_events):
        if not triggers_cfg and not trigger_events:
            return """## 5. Validation Triggers

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

        return f"""## 5. Validation Triggers (SR 11-7 Section II.C)

Per SR 11-7, models should be subject to ongoing monitoring with triggers
for re-validation when material changes occur or performance degrades.

### 5.1 Configured Triggers

{cfg_text}

### 5.2 Recent Trigger Events

{event_text}"""

    def _findings_and_recommendations(self, test_results, test_mapping):
        failed_tests = [
            (name, result) for name, result in test_results.items()
            if not _result_passed(result)
        ]

        if not failed_tests:
            return """## 6. Findings and Recommendations

### 6.1 Key Findings

All validation tests passed. The model demonstrates:
- Sound conceptual basis aligned with theory and intended use
- Accurate and stable outputs within expected ranges
- Appropriate sensitivity to key assumptions and parameters
- Effective ongoing monitoring and controls

### 6.2 Recommendations

**Status: APPROVED FOR USE**

The model is approved for use subject to:
1. Ongoing monitoring per configured triggers
2. Re-validation at the scheduled frequency
3. Immediate re-validation if material changes occur
4. Documentation of any model overrides or adjustments"""

        findings = []
        for i, (test_name, result) in enumerate(failed_tests, 1):
            failure = _result_failure(result)
            para_ref = test_mapping.get(test_name, "N/A")
            findings.append(
                f"{i}. **{test_name}** (SR 11-7 {para_ref}): {failure or 'Test failed'}"
            )

        findings_text = "\n".join(findings)

        return f"""## 6. Findings and Recommendations

### 6.1 Key Findings

The following validation tests failed and require remediation:

{findings_text}

### 6.2 Recommendations

**Status: REQUIRES REMEDIATION**

The model requires the following actions before approval:

1. **Immediate**: Address all failed validation tests
2. **Short-term**: Conduct root cause analysis for failures
3. **Medium-term**: Enhance model documentation per SR 11-7 Section IV
4. **Ongoing**: Implement enhanced monitoring per SR 11-7 Section II.C

### 6.3 Compensating Controls

Until remediation is complete, the following compensating controls should be
implemented:

- Enhanced model output review by senior model risk personnel
- Restrictions on model use for high-impact decisions
- Increased monitoring frequency
- Documentation of all model overrides"""

    def _approval_section(self, model_info, now):
        return f"""## 7. Approval and Sign-Off

Per SR 11-7 Section III, model validation results should be reported to
senior management and the board of directors. Model owners should address
validation findings and document any remediation actions.

### 7.1 Validation Team

| Role | Name | Date |
|------|------|------|
| Lead Validator | [To be completed] | {now.strftime('%Y-%m-%d')} |
| Independent Reviewer | [To be completed] | [Pending] |
| Model Risk Manager | [To be completed] | [Pending] |

### 7.2 Model Owner Acknowledgment

| Role | Name | Date |
|------|------|------|
| Model Owner | {model_info.get('owner', '[To be completed]')} | [Pending] |
| Business Line Head | [To be completed] | [Pending] |

### 7.3 Approval Status

- [ ] Model approved for use without restrictions
- [ ] Model approved with conditions/compensating controls
- [ ] Model requires remediation before approval
- [ ] Model disapproved / retired

### 7.4 Next Validation Due

Per SR 11-7 Section III.D, models should be validated on a frequency
commensurate with their risk. Next validation due:
**{model_info.get('validation_frequency', 'annual')} from approval date**

---

*This report was generated using the MRM framework in accordance with
Federal Reserve SR 11-7 supervisory guidance.*"""
