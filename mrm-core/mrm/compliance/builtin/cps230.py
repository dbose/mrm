"""APRA CPS 230 -- Operational Risk Management.

Bundled compliance standard for the Australian Prudential Regulation
Authority.  Implements ``ComplianceStandard`` so the MRM framework can
generate CPS 230 reports, run governance checks, and map tests to
specific regulatory paragraphs.

Usage::

    mrm docs generate ccr_monte_carlo --compliance standard:cps230
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
# Helpers (result introspection -- unchanged from original report module)
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


def _get_compliance_ref(details: Dict, standard_name: str = "cps230") -> str:
    """Read compliance reference from test result details.

    Supports both the new generic key and the legacy key:
      - ``details["compliance_references"]["cps230"]``  (new)
      - ``details["cps230_reference"]``                 (legacy)
    """
    refs = details.get("compliance_references", {})
    if isinstance(refs, dict) and standard_name in refs:
        return refs[standard_name]
    return details.get("cps230_reference", "")


# =======================================================================
# CPS 230 Standard Implementation
# =======================================================================

@register_standard
class CPS230Standard(ComplianceStandard):
    """APRA CPS 230 -- Operational Risk Management."""

    name = "cps230"
    display_name = "APRA CPS 230 -- Operational Risk Management"
    description = (
        "Prudential Standard CPS 230 requires APRA-regulated entities "
        "to manage operational risk, including model risk, in a manner "
        "commensurate with the size, complexity and risk profile of "
        "their operations."
    )
    jurisdiction = "AU"
    version = "2024"

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        return {
            "Para 8-10": {
                "title": "Risk Identification and Classification",
                "requirement": (
                    "An APRA-regulated entity must identify, assess and manage its "
                    "operational risks, including model risk, in a manner commensurate "
                    "with the size, complexity and risk profile of its operations."
                ),
            },
            "Para 11": {
                "title": "Accountability and Ownership",
                "requirement": (
                    "The Board and senior management must ensure clear accountability "
                    "for operational risk management, including designation of model owners."
                ),
            },
            "Para 12-14": {
                "title": "Validation Frequency and Scope",
                "requirement": (
                    "Models must be subject to independent validation at a frequency "
                    "commensurate with their risk tier and materiality."
                ),
            },
            "Para 15-18": {
                "title": "Risk Assessment Methodology",
                "requirement": (
                    "The entity must have a sound methodology for assessing operational "
                    "risks, including model risk, that is integrated into its overall "
                    "risk management framework."
                ),
            },
            "Para 19-23": {
                "title": "Concentration and Interconnection Risk",
                "requirement": (
                    "The entity must identify and manage concentration risks, including "
                    "wrong-way risk where exposure and counterparty credit quality are "
                    "adversely correlated."
                ),
            },
            "Para 24-27": {
                "title": "Scenario Analysis and Stress Testing",
                "requirement": (
                    "The entity must conduct scenario analysis and stress testing of "
                    "its models to assess sensitivity to key assumptions and parameters."
                ),
            },
            "Para 28-29": {
                "title": "Model Adequacy and Fitness-for-Purpose",
                "requirement": (
                    "Models must be fit for their intended purpose, with outputs that "
                    "are consistent with theoretical expectations and market behaviour."
                ),
            },
            "Para 30-33": {
                "title": "Operational Risk Controls",
                "requirement": (
                    "The entity must have controls in place to ensure the integrity, "
                    "accuracy and reliability of its model computations and IT systems."
                ),
            },
            "Para 34-37": {
                "title": "Ongoing Monitoring and Reporting",
                "requirement": (
                    "The entity must monitor model performance on an ongoing basis, "
                    "with triggers for re-validation when material changes occur."
                ),
            },
            "Para 38-42": {
                "title": "Risk Mitigation and Controls",
                "requirement": (
                    "The entity must maintain effective controls to mitigate identified "
                    "risks, including collateral management and netting arrangements."
                ),
            },
        }

    def get_test_mapping(self) -> Dict[str, str]:
        return {
            "ccr.MCConvergence": "Para 30-33",
            "ccr.EPEReasonableness": "Para 15-18",
            "ccr.PFEBacktest": "Para 34-37",
            "ccr.CVASensitivity": "Para 24-27",
            "ccr.WrongWayRisk": "Para 19-23",
            "ccr.ExposureProfileShape": "Para 28-29",
            "ccr.CollateralEffectiveness": "Para 38-42",
            "ccr.CPS230GovernanceCheck": "Para 8-10",
            "compliance.GovernanceCheck": "Para 8-10",
        }

    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        return {
            "risk_tier_assigned": {
                "description": "Risk tier must be assigned",
                "paragraph_ref": "CPS 230 Para 8-10",
                "config_key": "risk_tier",
            },
            "owner_designated": {
                "description": "Owner / accountability designated",
                "paragraph_ref": "CPS 230 Para 11",
                "config_key": "owner",
            },
            "validation_frequency_set": {
                "description": "Validation frequency defined",
                "paragraph_ref": "CPS 230 Para 12-14",
                "config_key": "validation_frequency",
            },
            "use_case_documented": {
                "description": "Model use case documented",
                "paragraph_ref": "CPS 230 Para 15",
                "config_key": "use_case",
            },
            "methodology_documented": {
                "description": "Methodology documented",
                "paragraph_ref": "CPS 230 Para 16",
                "config_key": "methodology",
            },
            "version_controlled": {
                "description": "Version control in place",
                "paragraph_ref": "CPS 230 Para 28",
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
        now = datetime.now(timezone.utc)
        model_info = model_config.get("model", model_config)

        # Read compliance mapping (new path first, legacy fallback)
        cps_mapping = (
            model_config
            .get("compliance", {})
            .get("standards", {})
            .get("cps230", {})
            .get("mapping", model_config.get("cps230_mapping", {}))
        )
        triggers_cfg = model_config.get("triggers", [])

        paragraphs = self.get_paragraphs()
        test_mapping = self.get_test_mapping()

        sections = [
            self._header(model_name, model_info, now),
            self._executive_summary(model_name, model_info, test_results, now),
            self._model_inventory_card(model_info, model_config),
            self._compliance_matrix(test_results, cps_mapping, paragraphs, test_mapping),
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
        return f"""# CPS 230 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | {model_name} |
| **Version** | {model_info.get('version', 'N/A')} |
| **Report Date** | {now.strftime('%Y-%m-%d %H:%M UTC')} |
| **Regulatory Framework** | {self.display_name} |
| **Risk Tier** | {model_info.get('risk_tier', 'N/A')} |
| **Owner** | {model_info.get('owner', 'N/A')} |
| **Validation Frequency** | {model_info.get('validation_frequency', 'N/A')} |

---"""

    def _executive_summary(self, model_name, model_info, test_results, now):
        total = len(test_results)
        passed = sum(1 for r in test_results.values() if _result_passed(r))
        failed = total - passed
        pass_rate = f"{passed / total * 100:.1f}%" if total > 0 else "N/A"
        status_marker = "PASSED" if failed == 0 and total > 0 else "FAILED"
        status = "PASS" if failed == 0 and total > 0 else "FAIL"

        return f"""## 1. Executive Summary

This report presents the independent validation results for the
**{model_name}** model (v{model_info.get('version', '1.0.0')})
conducted on {now.strftime('%Y-%m-%d')}.

The validation was performed in accordance with **{self.display_name}**
requirements.  The model is classified as
**{model_info.get('risk_tier', 'tier_1').replace('_', ' ').title()}**
({model_info.get('materiality', 'high')} materiality,
{model_info.get('complexity', 'high')} complexity), requiring
{model_info.get('validation_frequency', 'quarterly')} validation.

### Overall Result: **{status_marker}**

| Metric | Value |
|--------|-------|
| Tests Executed | {total} |
| Tests Passed | {passed} |
| Tests Failed | {failed} |
| Pass Rate | {pass_rate} |
| Validation Status | **{status}** |

**Model Purpose:** {model_info.get('description', 'N/A')}

**Methodology:** {model_info.get('methodology', 'N/A')}"""

    def _model_inventory_card(self, model_info, model_config):
        params = model_info.get("parameters", model_config.get("parameters", {}))
        param_rows = "\n".join(
            f"| {k} | {v} |" for k, v in params.items()
        ) if params else "| N/A | N/A |"

        return f"""## 2. Model Inventory Card

### 2.1 Identification

| Field | Value |
|-------|-------|
| Model Name | {model_info.get('name', 'N/A')} |
| Version | {model_info.get('version', 'N/A')} |
| Owner | {model_info.get('owner', 'N/A')} |
| Use Case | {model_info.get('use_case', 'N/A')} |
| Methodology | {model_info.get('methodology', 'N/A')} |
| Risk Tier | {model_info.get('risk_tier', 'N/A')} |
| Materiality | {model_info.get('materiality', 'N/A')} |
| Complexity | {model_info.get('complexity', 'N/A')} |
| Validation Frequency | {model_info.get('validation_frequency', 'N/A')} |

### 2.2 Model Parameters

| Parameter | Value |
|-----------|-------|
{param_rows}

### 2.3 CPS 230 Classification Rationale

Per **CPS 230 Para 8-10**, models must be classified by materiality and
complexity.  This model is classified as Tier 1 because:

- It computes regulatory capital metrics (EAD, CVA) used in prudential returns
- It uses Monte Carlo simulation requiring careful convergence control
- Errors in exposure estimates directly impact capital adequacy ratios
- The model covers OTC derivative portfolios with material notional exposure"""

    def _compliance_matrix(self, test_results, cps_mapping, paragraphs, test_mapping):
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

            for section_items in cps_mapping.values():
                if isinstance(section_items, list):
                    for item in section_items:
                        if item.get("paragraph") == para_key:
                            evidence_items.append(f"Config: {item.get('evidence', 'N/A')}")

            if statuses:
                overall = "SATISFIED" if all(statuses) else "NOT SATISFIED"
            elif evidence_items:
                overall = "DOCUMENTED"
            else:
                overall = "NOT ASSESSED"

            evidence_str = "; ".join(evidence_items) if evidence_items else "No tests mapped"
            rows.append(
                f"| {para_key} | {para_info['title']} | {overall} | {evidence_str} |"
            )

        rows_str = "\n".join(rows)

        return f"""## 3. CPS 230 Compliance Matrix

The following matrix maps each CPS 230 requirement to the validation
evidence demonstrating compliance.

| CPS 230 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
{rows_str}

### 3.1 Compliance Summary

Each row above corresponds to a specific paragraph of APRA CPS 230.
Tests are designed to provide quantitative evidence that the model
satisfies the operational risk management requirements of the standard.
Where a requirement is marked "SATISFIED", the corresponding validation
test has passed with results within acceptable thresholds."""

    def _detailed_test_results(self, test_results, paragraphs, test_mapping):
        sections = ["## 4. Detailed Test Results\n"]

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
| CPS 230 Reference | {para_ref}: {para_info.get('title', 'N/A')} |
"""

            if failure:
                section += f"| Failure Reason | {failure} |\n"

            if details:
                section += "\n**Evidence Details:**\n\n"
                section += "```json\n"
                filtered = {
                    k: v for k, v in details.items()
                    if k not in ("cps230_reference", "compliance_references")
                }
                section += json.dumps(filtered, indent=2, default=str)
                section += "\n```\n"

                ref = _get_compliance_ref(details, self.name)
                if ref:
                    section += f"\n**Regulatory Mapping:** {ref}\n"

            sections.append(section)

        return "\n".join(sections)

    def _trigger_section(self, triggers_cfg, trigger_events):
        trigger_rows = []
        for t in triggers_cfg:
            ttype = t.get("type", "unknown")
            desc = t.get("description", "")
            ref = t.get("compliance_reference", t.get("cps230_reference", ""))
            threshold = t.get("threshold") or t.get("schedule_days") or "N/A"
            trigger_rows.append(f"| {ttype} | {desc} | {threshold} | {ref} |")

        trigger_rows_str = "\n".join(trigger_rows) if trigger_rows else "| N/A | N/A | N/A | N/A |"

        event_section = ""
        if trigger_events:
            event_rows = []
            for e in trigger_events:
                event_rows.append(
                    f"| {e.get('trigger_id', 'N/A')} | {e.get('trigger_type', 'N/A')} | "
                    f"{e.get('fired_at', 'N/A')[:19]} | {e.get('reason', 'N/A')} | "
                    f"{e.get('status', 'N/A')} |"
                )
            event_rows_str = "\n".join(event_rows)
            event_section = f"""
### 5.2 Active Trigger Events

| Trigger ID | Type | Fired At | Reason | Status |
|------------|------|----------|--------|--------|
{event_rows_str}
"""

        return f"""## 5. Validation Triggers (CPS 230 Para 34-37)

Re-validation is triggered automatically when any of the following
conditions are met.  This implements the CPS 230 requirement for
ongoing monitoring and timely response to material changes.

### 5.1 Configured Triggers

| Type | Description | Threshold | Compliance Ref |
|------|-------------|-----------|----------------|
{trigger_rows_str}
{event_section}
### 5.3 Re-validation Schedule

Per CPS 230 Para 12-14, Tier 1 models require quarterly validation.
The trigger system supplements scheduled validation with event-driven
re-validation when:

- Back-test breaches exceed the defined threshold
- Model output drift is detected beyond tolerance
- Portfolio composition changes materially
- Regulatory amendments require model review"""

    def _findings_and_recommendations(self, test_results, test_mapping):
        findings = []
        recommendations = []

        for test_name, result in test_results.items():
            if not _result_passed(result):
                failure = _result_failure(result)
                findings.append(f"- **{test_name}**: {failure}")
                ref = test_mapping.get(test_name, "")
                recommendations.append(
                    f"- Remediate {test_name} failure "
                    f"({'(' + ref + ')' if ref else ''})"
                )

        if not findings:
            findings_str = "No material findings.  All validation tests passed."
            recs_str = (
                "- Continue quarterly monitoring per CPS 230 schedule\n"
                "- Review trigger thresholds annually"
            )
        else:
            findings_str = "\n".join(findings)
            recs_str = "\n".join(recommendations)
            recs_str += "\n- Escalate findings to model owner and risk committee per CPS 230 Para 11"

        return f"""## 6. Findings, Limitations, and Recommendations

### 6.1 Findings

{findings_str}

### 6.2 Model Limitations

- The model uses a simplified Vasicek rate process; more complex rate
  dynamics (e.g., Hull-White, LMM) may be warranted for exotic products
- Collateral modelling assumes instantaneous margin calls; margin period
  of risk is approximated
- Wrong-way risk detection is based on portfolio-level correlation;
  name-specific wrong-way risk requires additional analysis
- The model does not currently support multi-currency netting sets

### 6.3 Recommendations

{recs_str}"""

    def _approval_section(self, model_info, now):
        return f"""## 7. Approval and Sign-off

### CPS 230 Para 11: Accountability

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Model Owner | {model_info.get('owner', '_______________')} | ___/___/______ | _______________ |
| Independent Validator | _______________ | ___/___/______ | _______________ |
| Chief Risk Officer | _______________ | ___/___/______ | _______________ |
| Head of Model Risk | _______________ | ___/___/______ | _______________ |

### Attestation

I confirm that this validation has been conducted in accordance with
APRA CPS 230 requirements and the institution's Model Risk Management
Policy.  The findings and recommendations above are a true and accurate
representation of the validation outcomes.

---

*Report generated: {now.strftime('%Y-%m-%d %H:%M UTC')}*
*MRM Framework Version: 0.1.0*
*Regulatory Framework: {self.display_name}*"""
