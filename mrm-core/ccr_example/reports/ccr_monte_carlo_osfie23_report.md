# OSFI E-23 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-10 08:59 UTC |
| **Regulatory Framework** | OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management |
| **Jurisdiction** | Canada (All FRFIs) |
| **Risk Tier** | tier_1 |
| **Owner** | market_risk_team |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Validation Team |
| **Effective Date** | May 1, 2027 |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-10.

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
| Validation Tests Executed | 8 |
| Tests Passed | 5 |
| Tests Failed | 3 |
| Pass Rate | 62.5% |
| Validation Status | **FAIL** |

**Model Purpose:** Vanilla Monte Carlo simulation engine for Counterparty Credit Risk. Simulates interest rate swap mark-to-market paths under Vasicek rate dynamics to compute EPE, PFE, CVA, and EAD for OTC derivative portfolios.


**Methodology:** monte_carlo_simulation

**Business Line:** N/A

### Validation Scope (OSFI E-23 Section 5.2)

Per OSFI E-23, this independent validation includes:
- **Conceptual Soundness (Section 5.3)**: Review of model theory, assumptions, and methodology
- **Outcomes Analysis (Section 5.4)**: Backtesting and comparison with actual outcomes
- **Sensitivity Analysis (Section 5.5)**: Testing under various assumptions and stress scenarios
- **Ongoing Monitoring (Section 6)**: Review of performance monitoring framework

### Three Lines of Defence (OSFI E-23 Section 3.4)

This validation represents the **second line of defence** review:
- **First Line**: Model owner (market_risk_team) responsible for model performance and proper use
- **Second Line**: Independent validation team (MRM) conducted this review
- **Third Line**: Internal audit review pending

## 2. Model Inventory Card

### 2.1 Model Identification (OSFI E-23 Section 2.1 & 7.2)

| Field | Value |
|-------|-------|
| Model Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Model Owner | market_risk_team |
| Business Line | N/A |
| Use Case | counterparty_credit_risk |
| Methodology | monte_carlo_simulation |
| Model Type | Quantitative |

### 2.2 Risk Tiering (OSFI E-23 Section 2.2)

| Field | Value |
|-------|-------|
| Risk Tier | tier_1 |
| Materiality | high |
| Complexity | high |
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
| n_simulations | 5000 |
| n_time_steps | 60 |
| dt | 0.0833 |
| confidence_level | 0.95 |
| rate_model | vasicek |
| kappa | 0.15 |
| theta | 0.03 |
| sigma | 0.01 |
| r0 | 0.025 |

### 2.4 Key Assumptions (OSFI E-23 Section 4.1)

Not documented

### 2.5 Known Limitations (OSFI E-23 Section 7.1)

Not explicitly documented

### 2.6 Governance and Accountability (OSFI E-23 Section 3.2)

| Role | Name/Team |
|------|-----------|
| Model Owner | market_risk_team |
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
| Section 5.3 | Conceptual Soundness Review | NON-COMPLIANT | ccr.EPEReasonableness: FAIL; ccr.WrongWayRisk: PASS; ccr.ExposureProfileShape: FAIL |
| Section 5.4 | Outcomes Analysis and Backtesting | NON-COMPLIANT | ccr.PFEBacktest: FAIL; ccr.CollateralEffectiveness: PASS |
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

### ccr.MCConvergence ✅ PASS (score: 0.9720)
**OSFI E-23 Reference:** Section 4.3 -- Model Implementation and Testing

**Test Details:**
```json
{
  "epe_sample_a": 258694.90280232587,
  "epe_sample_b": 266051.82981546986,
  "relative_difference": 0.02804,
  "threshold": 0.05,
  "n_simulations": 5000,
  "compliance_references": {
    "cps230": "CPS 230 Para 30-33: Operational risk controls for model computation"
  }
}
```

### ccr.EPEReasonableness ❌ FAIL (score: 0.7800)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

**Failure Reason:** 22% of EPE/notional ratios outside [0.001, 0.1]

**Test Details:**
```json
{
  "mean_epe_notional_ratio": 0.017477,
  "min_ratio_observed": 0.0,
  "max_ratio_observed": 0.077223,
  "outlier_count": 11,
  "outlier_pct": 0.22,
  "bounds": [
    0.001,
    0.1
  ],
  "n_counterparties": 50,
  "compliance_references": {
    "cps230": "CPS 230 Para 15-18: Risk identification and assessment"
  }
}
```

### ccr.PFEBacktest ❌ FAIL (score: 0.8600)
**OSFI E-23 Reference:** Section 5.4 -- Outcomes Analysis and Backtesting

**Failure Reason:** PFE breach rate 14.00% exceeds 10% threshold

**Test Details:**
```json
{
  "breach_rate": 0.14,
  "n_breaches": 7,
  "n_observations": 50,
  "max_breach_rate": 0.1,
  "mean_pfe": 599091.71,
  "mean_realised": 72349.92,
  "confidence_level": 0.95,
  "compliance_references": {
    "cps230": "CPS 230 Para 34-37: Ongoing monitoring and reporting"
  }
}
```

### ccr.CVASensitivity ✅ PASS (score: 0.8926)
**OSFI E-23 Reference:** Section 5.5 -- Sensitivity Analysis and Scenario Testing

**Test Details:**
```json
{
  "pd_bump_pct": 50.0,
  "mean_sensitivity": 0.8926,
  "sensitivities": [
    1.0091,
    0.7494,
    0.9193
  ],
  "base_cvas": [
    5775.66,
    11350.68,
    469.41
  ],
  "shocked_cvas": [
    8689.82,
    15603.56,
    685.18
  ],
  "bounds": [
    0.1,
    3.0
  ],
  "compliance_references": {
    "cps230": "CPS 230 Para 24-27: Scenario analysis and stress testing"
  }
}
```

### ccr.WrongWayRisk ✅ PASS (score: 0.7752)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

**Test Details:**
```json
{
  "pd_exposure_correlation": -0.2248,
  "max_correlation": 0.6,
  "risk_level": "LOW",
  "n_counterparties": 50,
  "compliance_references": {
    "cps230": "CPS 230 Para 19-23: Concentration and interconnection risk"
  }
}
```

### ccr.ExposureProfileShape ❌ FAIL (score: 0.6000)
**OSFI E-23 Reference:** Section 5.3 -- Conceptual Soundness Review

**Failure Reason:** Exposure profile shape anomaly detected

**Test Details:**
```json
{
  "has_peak": false,
  "peak_time_step": 0,
  "coefficient_of_variation": 0.3074,
  "not_flat": true,
  "no_negatives": true,
  "profile_length": 60,
  "peak_ee": 449583.33,
  "terminal_ee": 127748.66,
  "compliance_references": {
    "cps230": "CPS 230 Para 28-29: Model adequacy and fitness-for-purpose"
  }
}
```

### ccr.CollateralEffectiveness ✅ PASS (score: 0.2836)
**OSFI E-23 Reference:** Section 5.4 -- Outcomes Analysis and Backtesting

**Test Details:**
```json
{
  "mean_epe_collateralised": 70849.09,
  "mean_epe_uncollateralised": 209868.45,
  "epe_notional_ratio_coll": 0.014945,
  "epe_notional_ratio_uncoll": 0.020862,
  "effective_reduction_pct": 28.36,
  "n_collateralised": 29,
  "n_uncollateralised": 21,
  "compliance_references": {
    "cps230": "CPS 230 Para 38-42: Risk mitigation and controls"
  }
}
```

### compliance.GovernanceCheck ✅ PASS (score: 1.0000)
**OSFI E-23 Reference:** Section 3.1 -- Governance Framework

**Test Details:**
```json
{
  "checks": {
    "risk_tier_assigned": true,
    "owner_designated": true,
    "validation_frequency_set": true,
    "use_case_documented": true,
    "methodology_documented": true,
    "version_controlled": true
  },
  "checks_passed": 6,
  "checks_total": 6,
  "standard": "cps230",
  "compliance_references": {
    "cps230": {
      "risk_tier_assigned": "CPS 230 Para 8-10",
      "owner_designated": "CPS 230 Para 11",
      "validation_frequency_set": "CPS 230 Para 12-14",
      "use_case_documented": "CPS 230 Para 15",
      "methodology_documented": "CPS 230 Para 16",
      "version_controlled": "CPS 230 Para 28"
    }
  }
}
```


## 5. Revalidation Triggers (OSFI E-23 Section 6.3)

OSFI E-23 Section 6.3 requires FRFIs to define triggers that
require model revalidation, including:
- Material changes to model logic or assumptions
- Significant market or business changes
- Breaches of performance thresholds
- Regulatory changes affecting model use

### Configured Triggers

| Trigger Type | Description | Threshold | Status |
|--------------|-------------|-----------|--------|
| scheduled | Quarterly scheduled re-validation | N/A | ✓ Active |
| breach | PFE back-test breach rate exceeds 10% | 0.1 | ✓ Active |
| drift | Monte Carlo output drift exceeds 15% | 0.15 | ✓ Active |
| materiality | Portfolio notional or counterparty count changes > 20% | 0.2 | ✓ Active |
| regulatory | APRA CPS 230 amendment or prudential guidance update | N/A | ✓ Active |

### Fired Trigger Events

- **None**: None
  - Compliance Reference: CPS 230 Para 34: Periodic review frequency
- **None**: None
  - Compliance Reference: CPS 230 Para 36: Breach-driven re-validation

## 6. Findings and Recommendations

### Critical Findings

**3 validation test(s) failed**, indicating
potential model risk issues requiring remediation:

- **ccr.EPEReasonableness** (OSFI E-23 Section 5.3)
  - Issue: 22% of EPE/notional ratios outside [0.001, 0.1]

- **ccr.PFEBacktest** (OSFI E-23 Section 5.4)
  - Issue: PFE breach rate 14.00% exceeds 10% threshold

- **ccr.ExposureProfileShape** (OSFI E-23 Section 5.3)
  - Issue: Exposure profile shape anomaly detected

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
| **Independent Validator** | [Name] | 2026-05-10 | _______________ |
| **Senior Model Risk Officer** | [Name] | __________ | _______________ |
| **Model Owner** | market_risk_team | __________ | _______________ |

---

**Report Generated By:** mrm-core v1.0.0  
**Framework:** OSFI E-23 -- Guideline on Enterprise-Wide Model Risk Management  
**Report Date:** 2026-05-10 08:59 UTC

---

### Regulatory References

- OSFI Guideline E-23: Enterprise-Wide Model Risk Management (January 2023)
- OSFI Effective Date: May 1, 2027 (expanded scope to all FRFIs)
- NIST AI Risk Management Framework (cross-reference where applicable)
- AMF Guideline on Sound Business and Financial Practices (Quebec)