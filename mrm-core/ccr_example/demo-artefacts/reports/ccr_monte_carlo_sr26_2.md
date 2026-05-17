# SR 26-2 Model Validation Report

| Field | Value |
|-------|-------|
| **Model** | ccr_monte_carlo |
| **Version** | 1.4.0 |
| **Report Date** | 2026-05-17 06:48 UTC |
| **Regulatory Framework** | Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems |
| **Standard Version** | 2026 |
| **Risk Tier** | tier_1 |
| **AI Materiality** | high |
| **Owner** | ccr-validation |
| **Validation Frequency** | quarterly |
| **Independent Validator** | MRM Team |

---

## 1. Executive Summary

This report presents the independent validation results for the
**ccr_monte_carlo** model (v1.4.0)
conducted on 2026-05-17.

The validation was performed in accordance with **Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems**
which supersedes SR 11-7 and SR 21-8 for banking organizations with
total assets above $30B. The model is classified as
**Tier 1**
(high materiality,
high complexity), requiring
quarterly validation.

### Overall Result: **APPROVED**

| Metric | Value |
|--------|-------|
| Validation Tests Executed | 7 |
| Tests Passed | 7 |
| Tests Failed | 0 |
| Pass Rate | 100.0% |
| Validation Status | **PASS** |

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
| Version | 1.4.0 |
| Owner | ccr-validation |
| Business Line | N/A |
| Use Case | Counterparty credit risk exposure |
| Methodology | Monte Carlo simulation |
| Risk Tier | tier_1 |
| AI Materiality | high |
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
| AI Activity Logging (II.AI.A) | CONFIGURED | local |
| Tamper-Evident Evidence Vault (III.AI.A) | CONFIGURED | local + Merkle root |
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
| Section II.C | Outcomes Analysis and Backtesting | COMPLIANT | ccr.PFEBacktest: PASS |
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
| Status | **PASS** |
| Score | 0.9400 |
| SR 26-2 Reference | Section II.C: Outcomes Analysis and Backtesting |

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

All validation tests passed. The model demonstrates:

- Sound conceptual basis aligned with theory and intended use
- Accurate and stable outputs within expected ranges
- Appropriate sensitivity to key assumptions and parameters
- Effective ongoing monitoring and controls
- Where applicable, AI-specific risk thresholds satisfied

### 7.2 Recommendations

**Status: APPROVED FOR USE**

The model is approved for use subject to:

1. Ongoing monitoring per configured triggers
2. Re-validation at the scheduled frequency
3. Immediate re-validation if material changes occur
4. Documentation of any model overrides or adjustments
5. For AI systems: continued capture of decision records and
   periodic verification of evidence-chain integrity

## 8. Approval and Sign-Off

Per SR 26-2 Section III, model validation results must be reported to
senior management and the board of directors. Model owners must
address validation findings and document any remediation actions.

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Owner | ccr-validation | _______________ | 2026-05-17 |
| Independent Validator | MRM Team | _______________ | 2026-05-17 |
| Model Risk Manager | _______________ | _______________ | _______________ |
| Chief Risk Officer | _______________ | _______________ | _______________ |

---

*This report was generated by* ``mrm-core`` *in accordance with
Federal Reserve SR 26-2 -- Supervisory Guidance on Model Risk Management for AI Systems. AI-specific clauses are anchored to the
DecisionRecord (`docs/spec/replay-record-v1.md`) and EvidencePacket
(`docs/spec/evidence-vault-v1.md`) primitives.*