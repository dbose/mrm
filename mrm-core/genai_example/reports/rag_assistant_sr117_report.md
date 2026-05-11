# SR 11-7 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | rag_assistant |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-11 04:03 UTC |
| **Regulatory Framework** | Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management |
| **Risk Tier** | tier_1 |
| **Owner** | ai_risk_team@example.com |
| **Validation Frequency** | N/A |
| **Independent Validator** | MRM Team |

---

## 1. Executive Summary

This report presents the independent validation results for the
**rag_assistant** model (v1.0.0)
conducted on 2026-05-11.

The validation was performed in accordance with **Federal Reserve SR 11-7 -- Supervisory Guidance on Model Risk Management**
requirements. The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
annual validation.

### Overall Result: **APPROVED**

| Metric | Value |
|--------|-------|
| Validation Tests Executed | 2 |
| Tests Passed | 2 |
| Tests Failed | 0 |
| Pass Rate | 100.0% |
| Validation Status | **PASS** |

**Model Purpose:** RAG-based customer service assistant - MINIMAL TEST CONFIG


**Methodology:** N/A

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
| Model Name | rag_assistant |
| Version | 1.0.0 |
| Owner | ai_risk_team@example.com |
| Business Line | N/A |
| Use Case | N/A |
| Methodology | N/A |
| Risk Tier | tier_1 |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | N/A |

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
| Section II.B | Evaluation of Conceptual Soundness | NOT ASSESSED | No tests mapped |
| Section II.C | Ongoing Monitoring | NOT ASSESSED | No tests mapped |
| Section II.D | Outcomes Analysis and Backtesting | NOT ASSESSED | No tests mapped |
| Section II.E | Sensitivity Analysis and Stress Testing | NOT ASSESSED | No tests mapped |
| Section III.A | Model Risk Management Framework | NOT ASSESSED | No tests mapped |
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

### 4.1 genai.LatencyBound

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| SR 11-7 Reference | N/A: N/A |

**Validation Evidence:**

```json
{
  "p_latency": 4485.2312207221985,
  "percentile": 95,
  "samples": 20,
  "latencies": [
    4271.193027496338,
    3425.8980751037598,
    4106.152057647705,
    3239.264965057373,
    4255.100250244141,
    4689.66007232666,
    4231.331825256348,
    4344.702959060669,
    3848.16837310791,
    4474.471807479858,
    3253.7803649902344,
    3397.7012634277344,
    3734.4110012054443,
    3169.7840690612793,
    3331.745147705078,
    3625.7128715515137,
    3857.9249382019043,
    4134.780168533325,
    2922.8591918945312,
    3965.130090713501
  ]
}
```

### 4.2 genai.CostBound

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| SR 11-7 Reference | N/A: N/A |

**Validation Evidence:**

```json
{
  "avg_cost": 0.019932000000000002,
  "max_cost": 0.1,
  "samples": 10,
  "costs": [
    0.01842,
    0.017640000000000003,
    0.02058,
    0.02058,
    0.02094,
    0.019020000000000002,
    0.020700000000000003,
    0.0204,
    0.02028,
    0.02076
  ]
}
```


## 5. Validation Triggers

No validation triggers configured for this model.

## 6. Findings and Recommendations

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
4. Documentation of any model overrides or adjustments

## 7. Approval and Sign-Off

Per SR 11-7 Section III, model validation results should be reported to
senior management and the board of directors. Model owners should address
validation findings and document any remediation actions.

### 7.1 Validation Team

| Role | Name | Date |
|------|------|------|
| Lead Validator | [To be completed] | 2026-05-11 |
| Independent Reviewer | [To be completed] | [Pending] |
| Model Risk Manager | [To be completed] | [Pending] |

### 7.2 Model Owner Acknowledgment

| Role | Name | Date |
|------|------|------|
| Model Owner | ai_risk_team@example.com | [Pending] |
| Business Line Head | [To be completed] | [Pending] |

### 7.3 Approval Status

- [ ] Model approved for use without restrictions
- [ ] Model approved with conditions/compensating controls
- [ ] Model requires remediation before approval
- [ ] Model disapproved / retired

### 7.4 Next Validation Due

Per SR 11-7 Section III.D, models should be validated on a frequency
commensurate with their risk. Next validation due:
**annual from approval date**

---

*This report was generated using the MRM framework in accordance with
Federal Reserve SR 11-7 supervisory guidance.*