# SR 11-7 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-16 07:03 UTC |
| **Regulatory Framework** | Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management |
| **Risk Tier** | tier_1 |
| **Owner** | ccr-team |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Team |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-16.

The validation was performed in accordance with **Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management**
requirements. The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
quarterly validation.

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

### Validation Scope

Per SR 11-7 Section II, this validation includes:
- **Conceptual Soundness**: Review of model theory, assumptions, and structure
- **Ongoing Monitoring**: Analysis of model inputs, outputs, and stability
- **Outcomes Analysis**: Backtesting where applicable
- **Sensitivity Analysis**: Testing under various scenarios and stress conditions

## 2. Model Inventory Card

### 2.1 Identification (SR 11-7 Section IV)

| Field | Value |
|-------|-------|
| Model Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Owner | ccr-team |
| Business Line | N/A |
| Use Case | Counterparty credit risk exposure |
| Methodology | Monte Carlo simulation |
| Risk Tier | tier_1 |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | quarterly |

### 2.2 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
| N/A | N/A |

### 2.3 Key Assumptions (SR 11-7 Section I.C)

Not documented

### 2.4 Risk Tier Classification Rationale

Per **SR 11-7 Section III.D**, models should be validated on a frequency
commensurate with their risk and materiality. This model is classified as
Tier 1 because:

- It produces material financial impacts (capital requirements, risk metrics)
- It uses complex methodologies requiring specialized expertise
- Errors could result in material misstatement of risk or capital positions
- The model is used in decision-making with significant consequences

## 3. SR 11-7 Compliance Matrix

The following matrix maps each SR 11-7 requirement to the validation
evidence demonstrating compliance.

| SR 11-7 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
| Section I.A | Model Development and Implementation | NOT ASSESSED | No tests mapped |
| Section I.B | Model Use and Purpose | NOT ASSESSED | No tests mapped |
| Section I.C | Model Assumptions and Data | NOT ASSESSED | No tests mapped |
| Section I.D | Model Outputs and Reporting | NOT ASSESSED | No tests mapped |
| Section II.A | Validation Scope and Objectives | NOT ASSESSED | No tests mapped |
| Section II.B | Evaluation of Conceptual Soundness | COMPLIANT | ccr.MCConvergence: PASS; ccr.EPEReasonableness: PASS; ccr.WrongWayRisk: PASS; ccr.ExposureProfileShape: PASS |
| Section II.C | Ongoing Monitoring | NOT ASSESSED | No tests mapped |
| Section II.D | Outcomes Analysis and Backtesting | NON-COMPLIANT | ccr.PFEBacktest: FAIL |
| Section II.E | Sensitivity Analysis and Stress Testing | COMPLIANT | ccr.CVASensitivity: PASS |
| Section III.A | Model Risk Management Framework | COMPLIANT | compliance.GovernanceCheck: PASS |
| Section III.B | Roles and Responsibilities | NOT ASSESSED | No tests mapped |
| Section III.C | Board and Senior Management Oversight | NOT ASSESSED | No tests mapped |
| Section III.D | Validation Frequency and Coverage | NOT ASSESSED | No tests mapped |
| Section IV | Model Inventory and Documentation | NOT ASSESSED | No tests mapped |

### 3.1 Compliance Summary

Each row above corresponds to a specific section of Federal Reserve SR 11-7.
Tests are designed to provide quantitative evidence that the model satisfies
the supervisory expectations for model risk management. Where a requirement
is marked "COMPLIANT", the corresponding validation test has passed with
results within acceptable thresholds.

Per SR 11-7, validation activities should provide a critical and independent
assessment of model performance. This validation was conducted by personnel
independent of model development and use.

## 4. Detailed Validation Test Results

### 4.1 ccr.MCConvergence

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9991 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.2 ccr.EPEReasonableness

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9800 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.3 ccr.PFEBacktest

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.7400 |
| SR 11-7 Reference | Section II.D: Outcomes Analysis and Backtesting |
| Failure Reason | Backtest p-value below 0.05 |

**SR 11-7 Requirement:** Model outcomes should be compared with actual outcomes to assess model accuracy and identify potential issues. Backtesting should be performed where feasible, with analysis of errors and systematic biases.

### 4.4 ccr.CVASensitivity

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9200 |
| SR 11-7 Reference | Section II.E: Sensitivity Analysis and Stress Testing |

**SR 11-7 Requirement:** Validation should include comprehensive sensitivity analysis and stress testing to evaluate model behavior under a range of assumptions and scenarios, particularly adverse conditions.

### 4.5 ccr.WrongWayRisk

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.8800 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.6 ccr.ExposureProfileShape

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9500 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.7 compliance.GovernanceCheck

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| SR 11-7 Reference | Section III.A: Model Risk Management Framework |

**SR 11-7 Requirement:** Banks should have a sound model risk management framework that provides well-defined and consistent policies, procedures, and controls covering all aspects of the model risk management process.


## 5. Validation Triggers

No validation triggers configured for this model.

## 6. Findings and Recommendations

### 6.1 Key Findings

The following validation tests failed and require remediation:

1. **ccr.PFEBacktest** (SR 11-7 Section II.D): Backtest p-value below 0.05

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
- Documentation of all model overrides

## 7. Approval and Sign-Off

Per SR 11-7 Section III, model validation results should be reported to
senior management and the board of directors. Model owners should address
validation findings and document any remediation actions.

### 7.1 Validation Team

| Role | Name | Date |
|------|------|------|
| Lead Validator | [To be completed] | 2026-05-16 |
| Independent Reviewer | [To be completed] | [Pending] |
| Model Risk Manager | [To be completed] | [Pending] |

### 7.2 Model Owner Acknowledgment

| Role | Name | Date |
|------|------|------|
| Model Owner | ccr-team | [Pending] |
| Business Line Head | [To be completed] | [Pending] |

### 7.3 Approval Status

- [ ] Model approved for use without restrictions
- [ ] Model approved with conditions/compensating controls
- [ ] Model requires remediation before approval
- [ ] Model disapproved / retired

### 7.4 Next Validation Due

Per SR 11-7 Section III.D, models should be validated on a frequency
commensurate with their risk. Next validation due:
**quarterly from approval date**

---

*This report was generated using the MRM framework in accordance with
Federal Reserve SR 11-7 supervisory guidance.*