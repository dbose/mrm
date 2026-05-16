# CPS 230 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-16 07:03 UTC |
| **Regulatory Framework** | APRA CPS 230 -- Operational Risk Management |
| **Risk Tier** | tier_1 |
| **Owner** | ccr-team |
| **Validation Frequency** | quarterly |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-16.

The validation was performed in accordance with **APRA CPS 230 -- Operational Risk Management**
requirements.  The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
quarterly validation.

### Overall Result: **FAILED**

| Metric | Value |
|--------|-------|
| Tests Executed | 7 |
| Tests Passed | 6 |
| Tests Failed | 1 |
| Pass Rate | 85.7% |
| Validation Status | **FAIL** |

**Model Purpose:** N/A

**Methodology:** Monte Carlo simulation

## 2. Model Inventory Card

### 2.1 Identification

| Field | Value |
|-------|-------|
| Model Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Owner | ccr-team |
| Use Case | Counterparty credit risk exposure |
| Methodology | Monte Carlo simulation |
| Risk Tier | tier_1 |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | quarterly |

### 2.2 Model Parameters

| Parameter | Value |
|-----------|-------|
| N/A | N/A |

### 2.3 CPS 230 Classification Rationale

Per **CPS 230 Para 8-10**, models must be classified by materiality and
complexity.  This model is classified as Tier 1 because:

- It computes regulatory capital metrics (EAD, CVA) used in prudential returns
- It uses Monte Carlo simulation requiring careful convergence control
- Errors in exposure estimates directly impact capital adequacy ratios
- The model covers OTC derivative portfolios with material notional exposure

## 3. CPS 230 Compliance Matrix

The following matrix maps each CPS 230 requirement to the validation
evidence demonstrating compliance.

| CPS 230 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
| Para 8-10 | Risk Identification and Classification | SATISFIED | compliance.GovernanceCheck: PASS |
| Para 11 | Accountability and Ownership | NOT ASSESSED | No tests mapped |
| Para 12-14 | Validation Frequency and Scope | NOT ASSESSED | No tests mapped |
| Para 15-18 | Risk Assessment Methodology | SATISFIED | ccr.EPEReasonableness: PASS |
| Para 19-23 | Concentration and Interconnection Risk | SATISFIED | ccr.WrongWayRisk: PASS |
| Para 24-27 | Scenario Analysis and Stress Testing | SATISFIED | ccr.CVASensitivity: PASS |
| Para 28-29 | Model Adequacy and Fitness-for-Purpose | SATISFIED | ccr.ExposureProfileShape: PASS |
| Para 30-33 | Operational Risk Controls | SATISFIED | ccr.MCConvergence: PASS |
| Para 34-37 | Ongoing Monitoring and Reporting | NOT SATISFIED | ccr.PFEBacktest: FAIL |
| Para 38-42 | Risk Mitigation and Controls | NOT ASSESSED | No tests mapped |

### 3.1 Compliance Summary

Each row above corresponds to a specific paragraph of APRA CPS 230.
Tests are designed to provide quantitative evidence that the model
satisfies the operational risk management requirements of the standard.
Where a requirement is marked "SATISFIED", the corresponding validation
test has passed with results within acceptable thresholds.

## 4. Detailed Test Results

### 4.1 ccr.MCConvergence

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9991 |
| CPS 230 Reference | Para 30-33: Operational Risk Controls |

### 4.2 ccr.EPEReasonableness

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9800 |
| CPS 230 Reference | Para 15-18: Risk Assessment Methodology |

### 4.3 ccr.PFEBacktest

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.7400 |
| CPS 230 Reference | Para 34-37: Ongoing Monitoring and Reporting |
| Failure Reason | Backtest p-value below 0.05 |

### 4.4 ccr.CVASensitivity

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9200 |
| CPS 230 Reference | Para 24-27: Scenario Analysis and Stress Testing |

### 4.5 ccr.WrongWayRisk

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.8800 |
| CPS 230 Reference | Para 19-23: Concentration and Interconnection Risk |

### 4.6 ccr.ExposureProfileShape

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9500 |
| CPS 230 Reference | Para 28-29: Model Adequacy and Fitness-for-Purpose |

### 4.7 compliance.GovernanceCheck

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| CPS 230 Reference | Para 8-10: Risk Identification and Classification |


## 5. Validation Triggers (CPS 230 Para 34-37)

Re-validation is triggered automatically when any of the following
conditions are met.  This implements the CPS 230 requirement for
ongoing monitoring and timely response to material changes.

### 5.1 Configured Triggers

| Type | Description | Threshold | Compliance Ref |
|------|-------------|-----------|----------------|
| N/A | N/A | N/A | N/A |

### 5.3 Re-validation Schedule

Per CPS 230 Para 12-14, Tier 1 models require quarterly validation.
The trigger system supplements scheduled validation with event-driven
re-validation when:

- Back-test breaches exceed the defined threshold
- Model output drift is detected beyond tolerance
- Portfolio composition changes materially
- Regulatory amendments require model review

## 6. Findings, Limitations, and Recommendations

### 6.1 Findings

- **ccr.PFEBacktest**: Backtest p-value below 0.05

### 6.2 Model Limitations

- The model uses a simplified Vasicek rate process; more complex rate
  dynamics (e.g., Hull-White, LMM) may be warranted for exotic products
- Collateral modelling assumes instantaneous margin calls; margin period
  of risk is approximated
- Wrong-way risk detection is based on portfolio-level correlation;
  name-specific wrong-way risk requires additional analysis
- The model does not currently support multi-currency netting sets

### 6.3 Recommendations

- Remediate ccr.PFEBacktest failure ((Para 34-37))
- Escalate findings to model owner and risk committee per CPS 230 Para 11

## 7. Approval and Sign-off

### CPS 230 Para 11: Accountability

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Model Owner | ccr-team | ___/___/______ | _______________ |
| Independent Validator | _______________ | ___/___/______ | _______________ |
| Chief Risk Officer | _______________ | ___/___/______ | _______________ |
| Head of Model Risk | _______________ | ___/___/______ | _______________ |

### Attestation

I confirm that this validation has been conducted in accordance with
APRA CPS 230 requirements and the institution's Model Risk Management
Policy.  The findings and recommendations above are a true and accurate
representation of the validation outcomes.

---

*Report generated: 2026-05-16 07:03 UTC*
*MRM Framework Version: 0.1.0*
*Regulatory Framework: APRA CPS 230 -- Operational Risk Management*