# SR 26-2 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Report Date** | 2026-05-16 05:32 UTC |
| **Regulatory Framework** | Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems |
| **Standard Version** | 2026 |
| **Risk Tier** | tier_1 |
| **AI Materiality** | N/A |
| **Owner** | ccr-team |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Team |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.0.0)
conducted on 2026-05-16.

The validation was performed in accordance with **Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems**
which supersedes SR 11-7 and SR 21-8 for banking organizations with
total assets above $30B. The model is classified as
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

Per SR 26-2 Section II, this validation includes:

- **Conceptual Soundness** (II.A): Review of model theory, assumptions, and structure
- **Ongoing Monitoring** (II.B): Analysis of model inputs, outputs, and stability
- **Outcomes Analysis** (II.C): Backtesting where applicable
- **Sensitivity & Stress Testing** (II.D): Behaviour under adverse scenarios
- **AI Activity Logging** (II.AI.A): Decision-record capture per inference
- **Decision Reconstruction** (II.AI.B): Replay of historical decisions
- **AI-Specific Risk Testing** (II.AI.C): Hallucination, injection, PII, bias, robustness
- **RAG Context Integrity** (II.AI.D): Retrieval corpus drift monitoring

## 2. Model Inventory Card

### 2.1 Identification (SR 26-2 Section IV.A)

| Field | Value |
|-------|-------|
| Model Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Owner | ccr-team |
| Business Line | N/A |
| Use Case | Counterparty credit risk exposure |
| Methodology | Monte Carlo simulation |
| Risk Tier | tier_1 |
| AI Materiality | N/A |
| Materiality | N/A |
| Complexity | N/A |
| Validation Frequency | quarterly |
| Third-Party Model | No |

### 2.2 Model Parameters and Configuration

| Parameter | Value |
|-----------|-------|
| N/A | N/A |

### 2.3 Key Assumptions (SR 26-2 Section I.C)

Not documented

### 2.4 Risk Tier Classification Rationale (SR 26-2 Section III.AI.C)

Per **SR 26-2 Section III.C**, models must be validated on a frequency
commensurate with their risk and materiality. AI materiality
classification additionally considers downstream decision impact,
customer exposure, regulatory exposure, and reputational risk.

## 3. SR 26-2 AI Evidence Posture

This section is unique to SR 26-2 and reports the institution's
configuration for the AI-specific evidence expectations introduced in
this guidance.

| Expectation | Status | Configuration |
|-------------|--------|---------------|
| AI Activity Logging (II.AI.A) | NOT CONFIGURED | - |
| Tamper-Evident Evidence Vault (III.AI.A) | NOT CONFIGURED | - |
| Third-Party AI Assessment (V.A) | NOT CONFIGURED | - |

### 3.1 Replay-Anchored Clauses

The following clauses are satisfied by per-decision replay records
(captured via the ``mrm-core`` ``replay/`` primitive):

- **Section II.AI.A** -- AI Activity Logging
- **Section II.AI.B** -- Decision Reconstruction (Replay)
- **Section IV.AI.C** -- Per-Decision Audit Trail Linkage

### 3.2 Evidence-Vault-Anchored Clauses

The following clauses are satisfied by hash-chained evidence packets:

- **Section III.AI.A** -- Tamper-Evident Audit Trail
- **Section III.AI.B** -- Chain-of-Custody for Evidence

## 4. SR 26-2 Compliance Matrix

The following matrix maps each SR 26-2 requirement to the validation
evidence demonstrating compliance. AI-specific clauses (Section II.AI.*,
III.AI.*, IV.AI.C, V.*) are NEW relative to SR 11-7.

| SR 26-2 Ref | Requirement | Status | Evidence |
|-------------|-------------|--------|----------|
| Section I.A | Model Development -- Sound Theory and Design | NOT ASSESSED | No tests mapped |
| Section I.B | Model Use and Intended Purpose | NOT ASSESSED | No tests mapped |
| Section I.C | Data Quality, Representativeness, and Lineage | NOT ASSESSED | No tests mapped |
| Section II.A | Conceptual Soundness Review | COMPLIANT | ccr.MCConvergence: PASS; ccr.EPEReasonableness: PASS; ccr.WrongWayRisk: PASS; ccr.ExposureProfileShape: PASS |
| Section II.B | Ongoing Monitoring and Process Verification | NOT ASSESSED | No tests mapped |
| Section II.C | Outcomes Analysis and Backtesting | NON-COMPLIANT | ccr.PFEBacktest: FAIL |
| Section II.D | Sensitivity and Stress Testing | COMPLIANT | ccr.CVASensitivity: PASS |
| Section II.AI.A | AI Activity Logging (Decision Records) | DOCUMENTED | Anchor: replay:decision_record |
| Section II.AI.B | Decision Reconstruction (Replay) | DOCUMENTED | Anchor: replay:decision_record |
| Section II.AI.C | AI-Specific Risk Testing | NOT ASSESSED | No tests mapped |
| Section II.AI.D | RAG Context Integrity | NOT ASSESSED | No tests mapped |
| Section III.A | Policies, Procedures, and Controls | COMPLIANT | compliance.GovernanceCheck: PASS |
| Section III.B | Roles, Responsibilities, and Independence | NOT ASSESSED | No tests mapped |
| Section III.C | Risk-Tier-Commensurate Validation Cadence | NOT ASSESSED | No tests mapped |
| Section III.AI.A | Tamper-Evident Audit Trail | DOCUMENTED | Anchor: evidence:hash_chained_packet |
| Section III.AI.B | Chain-of-Custody for Evidence | DOCUMENTED | Anchor: evidence:hash_chained_packet |
| Section III.AI.C | AI Materiality Classification | NOT ASSESSED | No tests mapped |
| Section IV.A | Comprehensive Model Inventory | NOT ASSESSED | No tests mapped |
| Section IV.B | Version Control and Change Management | NOT ASSESSED | No tests mapped |
| Section IV.AI.C | Per-Decision Audit Trail Linkage | DOCUMENTED | Anchor: replay:decision_record |
| Section V.A | Third-Party / Vendor AI Model Governance | NOT ASSESSED | No tests mapped |
| Section V.B | Vendor-Provided Evidence | NOT ASSESSED | No tests mapped |

### 4.1 Compliance Summary

Per SR 26-2, validation activities must provide a critical and
independent assessment of model performance. For AI systems, the
institution must additionally demonstrate **per-decision replay
capability** (II.AI.A / II.AI.B) and **tamper-evident evidence**
(III.AI.A / III.AI.B).

## 5. Detailed Validation Test Results

### 5.1 ccr.MCConvergence

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9991 |
| SR 26-2 Reference | Section II.A: Conceptual Soundness Review |

**SR 26-2 Requirement:** Independent review of model theory, assumptions, and design choices, conducted by personnel independent of model development.

### 5.2 ccr.EPEReasonableness

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9800 |
| SR 26-2 Reference | Section II.A: Conceptual Soundness Review |

**SR 26-2 Requirement:** Independent review of model theory, assumptions, and design choices, conducted by personnel independent of model development.

### 5.3 ccr.PFEBacktest

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.7400 |
| SR 26-2 Reference | Section II.C: Outcomes Analysis and Backtesting |
| Failure Reason | Backtest p-value below 0.05 |

**SR 26-2 Requirement:** Where applicable, model outputs must be compared against realised outcomes; performance degradation must trigger documented action.

### 5.4 ccr.CVASensitivity

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9200 |
| SR 26-2 Reference | Section II.D: Sensitivity and Stress Testing |

**SR 26-2 Requirement:** Model behaviour must be tested under stressed and adverse scenarios appropriate to the model class.

### 5.5 ccr.WrongWayRisk

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.8800 |
| SR 26-2 Reference | Section II.A: Conceptual Soundness Review |

**SR 26-2 Requirement:** Independent review of model theory, assumptions, and design choices, conducted by personnel independent of model development.

### 5.6 ccr.ExposureProfileShape

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9500 |
| SR 26-2 Reference | Section II.A: Conceptual Soundness Review |

**SR 26-2 Requirement:** Independent review of model theory, assumptions, and design choices, conducted by personnel independent of model development.

### 5.7 compliance.GovernanceCheck

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| SR 26-2 Reference | Section III.A: Policies, Procedures, and Controls |

**SR 26-2 Requirement:** A board-approved model risk management policy must establish standards for model development, validation, use, and retirement, including AI-specific provisions.


## 6. Validation Triggers

No validation triggers configured for this model.

## 7. Findings and Recommendations

### 7.1 Key Findings

The following validation tests failed and require remediation:

1. **ccr.PFEBacktest** (SR 26-2 Section II.C): Backtest p-value below 0.05

### 7.2 Recommendations

**Status: REQUIRES REMEDIATION**

The model requires the following actions before approval:

1. **Immediate**: Address all failed validation tests
2. **Short-term**: Conduct root cause analysis for failures
3. **Medium-term**: Enhance model documentation per SR 26-2 Section IV
4. **Ongoing**: Implement enhanced monitoring per SR 26-2 Section II.B
5. **For AI systems**: ensure decision records are being captured for
   every inference (Section II.AI.A) and evidence packets are stored
   in a tamper-evident form (Section III.AI.A)

### 7.3 Compensating Controls

Until remediation is complete, the following compensating controls
should be implemented:

- Enhanced model output review by senior model risk personnel
- Restrictions on model use for high-impact decisions
- Increased monitoring frequency
- Documentation of all model overrides
- For AI systems: human-in-the-loop review of in-scope decisions

## 8. Approval and Sign-Off

Per SR 26-2 Section III, model validation results must be reported to
senior management and the board of directors. Model owners must
address validation findings and document any remediation actions.

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Owner | ccr-team | _______________ | 2026-05-16 |
| Independent Validator | MRM Team | _______________ | 2026-05-16 |
| Model Risk Manager | _______________ | _______________ | _______________ |
| Chief Risk Officer | _______________ | _______________ | _______________ |

---

*This report was generated by* ``mrm-core`` *in accordance with
Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems. AI-specific clauses are anchored to the
DecisionRecord (`docs/spec/replay-record-v1.md`) and EvidencePacket
(`docs/spec/evidence-vault-v1.md`) primitives.*