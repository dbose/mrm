# CPS 230 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | rag_assistant |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-11 04:03 UTC |
| **Regulatory Framework** | APRA CPS 230 -- Operational Risk Management |
| **Risk Tier** | tier_1 |
| **Owner** | ai_risk_team@example.com |
| **Validation Frequency** | N/A |

---

## 1. Executive Summary

This report presents the independent validation results for the
**rag_assistant** model (v1.0.0)
conducted on 2026-05-11.

The validation was performed in accordance with **APRA CPS 230 -- Operational Risk Management**
requirements.  The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
quarterly validation.

### Overall Result: **PASSED**

| Metric | Value |
|--------|-------|
| Tests Executed | 2 |
| Tests Passed | 2 |
| Tests Failed | 0 |
| Pass Rate | 100.0% |
| Validation Status | **PASS** |

**Model Purpose:** RAG-based customer service assistant - MINIMAL TEST CONFIG


**Methodology:** N/A

## 2. Model Inventory Card

### 2.1 Identification

| Field | Value |
|-------|-------|
| Model Name | rag_assistant |
| Version | 1.0.0 |
| Owner | ai_risk_team@example.com |
| Use Case | N/A |
| Methodology | N/A |
| Risk Tier | tier_1 |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | N/A |

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
| Para 8-10 | Risk Identification and Classification | NOT ASSESSED | No tests mapped |
| Para 11 | Accountability and Ownership | NOT ASSESSED | No tests mapped |
| Para 12-14 | Validation Frequency and Scope | NOT ASSESSED | No tests mapped |
| Para 15-18 | Risk Assessment Methodology | NOT ASSESSED | No tests mapped |
| Para 19-23 | Concentration and Interconnection Risk | NOT ASSESSED | No tests mapped |
| Para 24-27 | Scenario Analysis and Stress Testing | NOT ASSESSED | No tests mapped |
| Para 28-29 | Model Adequacy and Fitness-for-Purpose | NOT ASSESSED | No tests mapped |
| Para 30-33 | Operational Risk Controls | NOT ASSESSED | No tests mapped |
| Para 34-37 | Ongoing Monitoring and Reporting | NOT ASSESSED | No tests mapped |
| Para 38-42 | Risk Mitigation and Controls | NOT ASSESSED | No tests mapped |

### 3.1 Compliance Summary

Each row above corresponds to a specific paragraph of APRA CPS 230.
Tests are designed to provide quantitative evidence that the model
satisfies the operational risk management requirements of the standard.
Where a requirement is marked "SATISFIED", the corresponding validation
test has passed with results within acceptable thresholds.

## 4. Detailed Test Results

### 4.1 genai.LatencyBound

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| CPS 230 Reference | N/A: N/A |

**Evidence Details:**

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
| CPS 230 Reference | N/A: N/A |

**Evidence Details:**

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

No material findings.  All validation tests passed.

### 6.2 Model Limitations

- The model uses a simplified Vasicek rate process; more complex rate
  dynamics (e.g., Hull-White, LMM) may be warranted for exotic products
- Collateral modelling assumes instantaneous margin calls; margin period
  of risk is approximated
- Wrong-way risk detection is based on portfolio-level correlation;
  name-specific wrong-way risk requires additional analysis
- The model does not currently support multi-currency netting sets

### 6.3 Recommendations

- Continue quarterly monitoring per CPS 230 schedule
- Review trigger thresholds annually

## 7. Approval and Sign-off

### CPS 230 Para 11: Accountability

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Model Owner | ai_risk_team@example.com | ___/___/______ | _______________ |
| Independent Validator | _______________ | ___/___/______ | _______________ |
| Chief Risk Officer | _______________ | ___/___/______ | _______________ |
| Head of Model Risk | _______________ | ___/___/______ | _______________ |

### Attestation

I confirm that this validation has been conducted in accordance with
APRA CPS 230 requirements and the institution's Model Risk Management
Policy.  The findings and recommendations above are a true and accurate
representation of the validation outcomes.

---

*Report generated: 2026-05-11 04:03 UTC*
*MRM Framework Version: 0.1.0*
*Regulatory Framework: APRA CPS 230 -- Operational Risk Management*