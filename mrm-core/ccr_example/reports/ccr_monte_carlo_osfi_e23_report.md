# OSFI E-23 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-16 05:32 UTC |
| **Regulatory Framework** | OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management |
| **Jurisdiction** | Canada (All FRFIs) |
| **Risk Tier** | tier_1 |
| **Owner** | ccr-team |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Validation Team |
| **Effective Date** | May 1, 2027 |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-16.

The validation was performed in accordance with **OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management**
(effective May 1, 2027). This guideline applies to all federally
regulated financial institutions (FRFIs) in Canada, including banks,
insurance companies, and trust and loan companies.

The model is classified as
**Tier 1**
based on OSFI E-23 Section 2.2 risk tiering criteria (impact, complexity,
and uncertainty), requiring quarterly
independent validation.

### Overall Result: **REQUIRES REMEDIATION**

| Metric | Value |
|--------|-------|
| Validation Tests Executed | 7 |
| Tests Passed | 6 |
| Tests Failed | 1 |
| Pass Rate | 85.7% |
| Validation Status | **FAIL** |

**Model Purpose:** N/A

**Methodology:** Monte Carlo simulation

**Business Line:** N/A

### Validation Scope (OSFI E-23 Section 5.2)

Per OSFI E-23, this independent validation includes:
- **Conceptual Soundness (Section 5.3)**: Review of model theory, assumptions, and methodology
- **Outcomes Analysis (Section 5.4)**: Backtesting and comparison with actual outcomes
- **Sensitivity Analysis (Section 5.5)**: Testing under various assumptions and stress scenarios
- **Ongoing Monitoring (Section 6)**: Review of performance monitoring framework

### Three Lines of Defence (OSFI E-23 Section 3.4)

This validation represents the **second line of defence** review:
- **First Line**: Model owner (ccr-team) responsible for model performance and proper use
- **Second Line**: Independent validation team (MRM) conducted this review
- **Third Line**: Internal audit review pending

## 2. Model Inventory Card

### 2.1 Model Identification (OSFI E-23 Section 2.1 & 7.2)

| Field | Value |
|-------|-------|
| Model Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Model Owner | ccr-team |
| Business Line | N/A |
| Use Case | Counterparty credit risk exposure |
| Methodology | Monte Carlo simulation |
| Model Type | Quantitative |

### 2.2 Risk Tiering (OSFI E-23 Section 2.2)

| Field | Value |
|-------|-------|
| Risk Tier | tier_1 |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | quarterly |

**Risk Tier Rationale**: This model is classified as Tier 1 (highest risk)
based on its potential impact on capital adequacy, financial reporting, and
risk management decisions. The classification considers:
- **Impact**: Material financial and regulatory impact
- **Complexity**: Sophisticated methodology requiring specialized expertise
- **Uncertainty**: Significant model risk from assumptions and data limitations

### 2.3 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
| N/A | N/A |

### 2.4 Key Assumptions (OSFI E-23 Section 4.1)

Not documented

### 2.5 Known Limitations (OSFI E-23 Section 7.1)

Not explicitly documented

### 2.6 Governance and Accountability (OSFI E-23 Section 3.2)

| Role | Name/Team |
|------|-----------|
| Model Owner | ccr-team |
| Model Developer | N/A |
| Independent Validator | MRM Validation Team |
| Senior Management Sponsor | N/A |

## 3. OSFI E-23 Compliance Matrix

The following matrix maps each OSFI E-23 requirement to the validation
evidence demonstrating compliance.

| Section | Requirement | Status | Evidence |
|---------|-------------|--------|----------|
| Section 2.1 | Model Identification | NOT ASSESSED | No tests mapped |
| Section 2.2 | Risk Tiering and Materiality Assessment | NOT ASSESSED | No tests mapped |
| Section 3.1 | Governance Framework | COMPLIANT | compliance.GovernanceCheck: PASS |
| Section 3.2 | Roles and Accountability | NOT ASSESSED | No tests mapped |
| Section 3.3 | Senior Management and Board Oversight | NOT ASSESSED | No tests mapped |
| Section 3.4 | Three Lines of Defence | NOT ASSESSED | No tests mapped |
| Section 4.1 | Model Design and Theory | NOT ASSESSED | No tests mapped |
| Section 4.2 | Data Quality and Governance | NOT ASSESSED | No tests mapped |
| Section 4.3 | Model Implementation and Testing | COMPLIANT | ccr.MCConvergence: PASS |
| Section 5.1 | Independent Validation | NOT ASSESSED | No tests mapped |
| Section 5.2 | Validation Scope and Activities | NOT ASSESSED | No tests mapped |
| Section 5.3 | Conceptual Soundness Review | COMPLIANT | ccr.EPEReasonableness: PASS; ccr.WrongWayRisk: PASS; ccr.ExposureProfileShape: PASS |
| Section 5.4 | Outcomes Analysis and Backtesting | NON-COMPLIANT | ccr.PFEBacktest: FAIL |
| Section 5.5 | Sensitivity Analysis and Scenario Testing | COMPLIANT | ccr.CVASensitivity: PASS |
| Section 6.1 | Ongoing Monitoring Framework | NOT ASSESSED | No tests mapped |
| Section 6.2 | Performance Monitoring and Alerting | NOT ASSESSED | No tests mapped |
| Section 6.3 | Triggers for Revalidation | NOT ASSESSED | No tests mapped |
| Section 7.1 | Model Documentation Standards | NOT ASSESSED | No tests mapped |
| Section 7.2 | Model Inventory and Tracking | NOT ASSESSED | No tests mapped |
| Section 7.3 | Version Control and Change Management | NOT ASSESSED | No tests mapped |

**Status Legend:**
- **COMPLIANT**: All tests passed, requirement fully met
- **NON-COMPLIANT**: One or more tests failed
- **DOCUMENTED**: Evidence provided through configuration, no quantitative tests
- **NOT ASSESSED**: Requirement not tested in this validation cycle

## 4. Detailed Test Results

Per OSFI E-23 Section 5.2, validation activities include conceptual
soundness review, outcomes analysis, and sensitivity testing.

### ccr.MCConvergence ✅ PASS (score: 0.9991)
**OSFI E-23 Reference:** Section 4.3 -- Model Implementation and Testing

### ccr.EPEReasonableness ✅ PASS (score: 0.9800)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

### ccr.PFEBacktest ❌ FAIL (score: 0.7400)
**OSFI E-23 Reference:** Section 5.4 -- Outcomes Analysis and Backtesting

**Failure Reason:** Backtest p-value below 0.05

### ccr.CVASensitivity ✅ PASS (score: 0.9200)
**OSFI E-23 Reference:** Section 5.5 -- Sensitivity Analysis and Scenario Testing

### ccr.WrongWayRisk ✅ PASS (score: 0.8800)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

### ccr.ExposureProfileShape ✅ PASS (score: 0.9500)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

### compliance.GovernanceCheck ✅ PASS (score: 1.0000)
**OSFI E-23 Reference:** Section 3.1 -- Governance Framework


## 5. Revalidation Triggers (OSFI E-23 Section 6.3)

OSFI E-23 Section 6.3 requires FRFIs to define triggers that
require model revalidation, including:
- Material changes to model logic or assumptions
- Significant market or business changes
- Breaches of performance thresholds
- Regulatory changes affecting model use

**No triggers configured for this model.**

## 6. Findings and Recommendations

### Critical Findings

**1 validation test(s) failed**, indicating
potential model risk issues requiring remediation:

- **ccr.PFEBacktest** (OSFI E-23 Section 5.4)
  - Issue: Backtest p-value below 0.05

### Recommendations

Per OSFI E-23 Section 5.1, the following actions are recommended:

1. **Model owners should address all failed tests** within the
   remediation period specified in the FRFI's model risk policy.
2. **Revalidation required** after remediation to confirm issues
   are resolved (OSFI E-23 Section 6.3).
3. **Model use should be restricted** pending remediation, with
   escalation to senior management if issues are material.
4. **Enhanced monitoring** of model outputs during the remediation
   period (OSFI E-23 Section 6.2).

**Validation Conclusion:** Model **REQUIRES REMEDIATION** before
approval for production use.

## 7. Validation Sign-Off (OSFI E-23 Section 5.1)

Per OSFI E-23 Section 5.1, independent validation should be performed by
qualified staff with sufficient expertise to assess model soundness.

| Role | Name | Date | Signature |
|------|------|------|-----------|
| **Independent Validator** | [Name] | 2026-05-16 | _______________ |
| **Senior Model Risk Officer** | [Name] | __________ | _______________ |
| **Model Owner** | ccr-team | __________ | _______________ |

---

**Report Generated By:** mrm-core v1.0.0  
**Framework:** OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management  
**Report Date:** 2026-05-16 05:32 UTC

---

### Regulatory References

- OSFI Guideline E-23: Enterprise-Wide Model Risk Management (January 2023)
- OSFI Effective Date: May 1, 2027 (expanded scope to all FRFIs)
- NIST AI Risk Management Framework (cross-reference where applicable)
- AMF Guideline on Sound Business and Financial Practices (Quebec)