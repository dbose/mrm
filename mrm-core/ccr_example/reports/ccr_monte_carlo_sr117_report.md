# SR 11-7 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-10 06:22 UTC |
| **Regulatory Framework** | Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management |
| **Risk Tier** | tier_1 |
| **Owner** | market_risk_team |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Team |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-10.

The validation was performed in accordance with **Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management**
requirements. The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
quarterly validation.

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
| Owner | market_risk_team |
| Business Line | N/A |
| Use Case | counterparty_credit_risk |
| Methodology | monte_carlo_simulation |
| Risk Tier | tier_1 |
| Materiality | high |
| Complexity | high |
| Validation Frequency | quarterly |

### 2.2 Model Parameters and Configuration

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
| Section II.B | Evaluation of Conceptual Soundness | NON-COMPLIANT | ccr.MCConvergence: PASS; ccr.EPEReasonableness: FAIL; ccr.WrongWayRisk: PASS; ccr.ExposureProfileShape: FAIL |
| Section II.C | Ongoing Monitoring | NOT ASSESSED | No tests mapped |
| Section II.D | Outcomes Analysis and Backtesting | NON-COMPLIANT | ccr.PFEBacktest: FAIL; ccr.CollateralEffectiveness: PASS |
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
| Score | 0.9720 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**Validation Evidence:**

```json
{
  "epe_sample_a": 258694.90280232587,
  "epe_sample_b": 266051.82981546986,
  "relative_difference": 0.02804,
  "threshold": 0.05,
  "n_simulations": 5000
}
```

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.2 ccr.EPEReasonableness

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.7800 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |
| Failure Reason | 22% of EPE/notional ratios outside [0.001, 0.1] |

**Validation Evidence:**

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
  "n_counterparties": 50
}
```

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.3 ccr.PFEBacktest

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.8600 |
| SR 11-7 Reference | Section II.D: Outcomes Analysis and Backtesting |
| Failure Reason | PFE breach rate 14.00% exceeds 10% threshold |

**Validation Evidence:**

```json
{
  "breach_rate": 0.14,
  "n_breaches": 7,
  "n_observations": 50,
  "max_breach_rate": 0.1,
  "mean_pfe": 599091.71,
  "mean_realised": 72349.92,
  "confidence_level": 0.95
}
```

**SR 11-7 Requirement:** Model outcomes should be compared with actual outcomes to assess model accuracy and identify potential issues. Backtesting should be performed where feasible, with analysis of errors and systematic biases.

### 4.4 ccr.CVASensitivity

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.8926 |
| SR 11-7 Reference | Section II.E: Sensitivity Analysis and Stress Testing |

**Validation Evidence:**

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
  ]
}
```

**SR 11-7 Requirement:** Validation should include comprehensive sensitivity analysis and stress testing to evaluate model behavior under a range of assumptions and scenarios, particularly adverse conditions.

### 4.5 ccr.WrongWayRisk

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.7752 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |

**Validation Evidence:**

```json
{
  "pd_exposure_correlation": -0.2248,
  "max_correlation": 0.6,
  "risk_level": "LOW",
  "n_counterparties": 50
}
```

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.6 ccr.ExposureProfileShape

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.6000 |
| SR 11-7 Reference | Section II.B: Evaluation of Conceptual Soundness |
| Failure Reason | Exposure profile shape anomaly detected |

**Validation Evidence:**

```json
{
  "has_peak": false,
  "peak_time_step": 0,
  "coefficient_of_variation": 0.3074,
  "not_flat": true,
  "no_negatives": true,
  "profile_length": 60,
  "peak_ee": 449583.33,
  "terminal_ee": 127748.66
}
```

**SR 11-7 Requirement:** Validation should assess whether the model design and construction are consistent with sound theory and judgment, and appropriate for the intended use. This includes review of model assumptions, mathematical construction, and theoretical basis.

### 4.7 ccr.CollateralEffectiveness

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.2836 |
| SR 11-7 Reference | Section II.D: Outcomes Analysis and Backtesting |

**Validation Evidence:**

```json
{
  "mean_epe_collateralised": 70849.09,
  "mean_epe_uncollateralised": 209868.45,
  "epe_notional_ratio_coll": 0.014945,
  "epe_notional_ratio_uncoll": 0.020862,
  "effective_reduction_pct": 28.36,
  "n_collateralised": 29,
  "n_uncollateralised": 21
}
```

**SR 11-7 Requirement:** Model outcomes should be compared with actual outcomes to assess model accuracy and identify potential issues. Backtesting should be performed where feasible, with analysis of errors and systematic biases.

### 4.8 compliance.GovernanceCheck

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| SR 11-7 Reference | Section III.A: Model Risk Management Framework |

**Validation Evidence:**

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
  "standard": "cps230"
}
```

**SR 11-7 Requirement:** Banks should have a sound model risk management framework that provides well-defined and consistent policies, procedures, and controls covering all aspects of the model risk management process.


## 5. Validation Triggers (SR 11-7 Section II.C)

Per SR 11-7, models should be subject to ongoing monitoring with triggers
for re-validation when material changes occur or performance degrades.

### 5.1 Configured Triggers

- **scheduled**: Quarterly scheduled re-validation
- **breach**: PFE back-test breach rate exceeds 10%
- **drift**: Monte Carlo output drift exceeds 15%
- **materiality**: Portfolio notional or counterparty count changes > 20%
- **regulatory**: APRA CPS 230 amendment or prudential guidance update

### 5.2 Recent Trigger Events

| Trigger Type | Timestamp | Severity | Description |
|--------------|-----------|----------|-------------|
| scheduled | N/A | N/A | N/A |
| breach | N/A | N/A | N/A |

## 6. Findings and Recommendations

### 6.1 Key Findings

The following validation tests failed and require remediation:

1. **ccr.EPEReasonableness** (SR 11-7 Section II.B): 22% of EPE/notional ratios outside [0.001, 0.1]
2. **ccr.PFEBacktest** (SR 11-7 Section II.D): PFE breach rate 14.00% exceeds 10% threshold
3. **ccr.ExposureProfileShape** (SR 11-7 Section II.B): Exposure profile shape anomaly detected

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
| Lead Validator | [To be completed] | 2026-05-10 |
| Independent Reviewer | [To be completed] | [Pending] |
| Model Risk Manager | [To be completed] | [Pending] |

### 7.2 Model Owner Acknowledgment

| Role | Name | Date |
|------|------|------|
| Model Owner | market_risk_team | [Pending] |
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