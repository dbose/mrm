# EU AI Act Annex IV Technical Documentation

## High-Risk AI System Technical Documentation

| Field | Value |
|-------|-------|
| **AI System Name** | ccr_monte_carlo |
| **Version** | 1.0.0 |
| **Documentation Date** | 2026-05-16 05:28 UTC |
| **Regulatory Framework** | EU AI Act Annex IV -- Technical Documentation for High-Risk AI Systems |
| **Provider** | ccr-team |
| **Risk Classification** | High-Risk AI System |
| **Intended Purpose** | Counterparty credit risk exposure |

---

**Legal Basis:** Regulation (EU) 2024/1689 of the European Parliament and of the 
Council on artificial intelligence (AI Act), Article 11 and Annex IV.

**Applicable From:** 2 August 2026 (general provisions), 2 August 2027 (high-risk obligations)

---

## Executive Summary

This technical documentation has been prepared in accordance with **EU AI Act Annex IV -- Technical Documentation for High-Risk AI Systems**
for the AI system **ccr_monte_carlo** (v1.0.0).

### System Classification

This AI system is classified as a **high-risk AI system** under Article 6 of the EU AI Act.
Specific classification rationale: N/A

### Conformity Assessment Status: **NON-COMPLIANT**

| Metric | Value |
|--------|-------|
| Technical Tests Executed | 7 |
| Tests Passed | 6 |
| Tests Failed | 1 |
| Compliance Rate | 85.7% |
| Overall Status | **FAIL** |

**Intended Purpose:** N/A

**Methodology:** Monte Carlo simulation

### Conformity Assessment

This documentation demonstrates compliance with the essential requirements set out in 
Chapter 2 of the EU AI Act. The system has been subject to the conformity assessment 
procedure based on internal control as defined in Annex VI.

## 1. General Description of the AI System (Annex IV.1)

### 1.1 System Identification

| Field | Value |
|-------|-------|
| System Name | ccr_monte_carlo |
| Version | 1.0.0 |
| Provider | ccr-team |
| Intended Purpose | Counterparty credit risk exposure |
| Date of Documentation | 2026-05-16 |

### 1.2 Intended Purpose

**Primary Use Case:** N/A

**Target Users:** Financial institutions, credit risk teams

**Operating Environment:** Production banking systems

### 1.3 Reasonably Foreseeable Misuse

The following misuse scenarios have been identified:
- Use of the system outside its intended operating domain
- Application to populations or jurisdictions not covered in training data
- Use without appropriate human oversight and review
- Reliance on model outputs without consideration of limitations

### 1.4 System Architecture

**Deployment Form:** Standalone model component

**Hardware Requirements:** Standard computational resources

**Software Dependencies:** Python ML stack (scikit-learn, pandas, numpy)

### 1.5 Version Management

Current version: 1.0.0

Version updates are managed through:
- Version control system (Git)
- Change management process
- Re-validation after material changes

## 2. System Elements and Development Process (Annex IV.2)

### 2.1 Development Methodology

**Methodology:** Monte Carlo simulation

**Development Steps:**
1. Data collection and preparation
2. Feature engineering and selection
3. Model architecture design
4. Training and hyperparameter tuning
5. Validation and testing
6. Performance evaluation
7. Documentation and deployment

### 2.2 System Architecture

**Model Type:** Monte Carlo simulation

**Architecture Description:** The AI system is implemented as a Monte Carlo simulation 
model that processes structured input features to generate predictions. The system follows standard 
ML pipeline architecture with data preprocessing, feature transformation, model inference, and 
output post-processing stages.

### 2.3 Data Requirements (Datasheets)

#### Training Data

No datasets documented

**Data Quality Requirements:**
- Completeness: < 5% missing values per feature
- Representativeness: Balanced across key demographic groups
- Temporal coverage: Recent data within model refresh cycle
- Data governance: Appropriate consent and legal basis for processing

**Training Methodology:**
- Supervised learning with labeled historical data
- Train/test split methodology
- Cross-validation for hyperparameter tuning
- Holdout validation set for final performance assessment

### 2.4 Computational Resources

**Training Resources:** Standard CPU/GPU compute resources

**Inference Resources:** Low-latency prediction engine

**Update Frequency:** quarterly re-validation and retraining as needed

## 3. Monitoring, Functioning and Control (Annex IV.3)

### 3.1 System Capabilities

The AI system is capable of:
- Processing structured tabular input data
- Generating probability estimates or classification predictions
- Operating within specified accuracy thresholds
- Providing explainable outputs

### 3.2 System Limitations

**Known Limitations:**
- Performance may degrade on out-of-distribution data
- Accuracy varies across different population subgroups
- Requires periodic retraining to maintain performance
- Limited to the specific use case and domain defined in intended purpose

**Performance Boundaries:**
- Intended population: Defined in training data distribution
- Geographic scope: As per training data jurisdiction
- Temporal validity: quarterly validation cycle

### 3.3 Performance Metrics

| Metric | Expected Performance | Actual Performance |
|--------|---------------------|-------------------|
| Overall Accuracy | ≥ 70% | N/A |
| ROC-AUC | ≥ 70% | N/A |
| Gini Coefficient | ≥ 40% | N/A |
| Precision | ≥ 65% | N/A |
| Recall | ≥ 65% | N/A |

### 3.4 Human Oversight Measures

**Human-in-the-Loop Requirements:**
- All high-impact predictions subject to human review
- Model outputs presented with confidence intervals and explanations
- Override mechanisms available for human decision-makers
- Regular review of model performance by qualified personnel
- Escalation procedures for anomalous or uncertain predictions

**Oversight Roles:**
- Model validators: Independent review and challenge
- Business users: Application of model outputs with understanding of limitations
- Risk managers: Ongoing monitoring and governance
- Compliance officers: Regulatory adherence verification

## 4. Risk Management System (Annex IV.4)

### 4.1 Risk Identification and Analysis

Per Article 9 of the EU AI Act, the following risks have been identified:

**Risk Category: Discriminatory Outcomes**
- Risk: Model may produce biased predictions across protected demographic groups
- Mitigation: Fairness testing, bias monitoring, subgroup performance analysis
- Residual Risk: Low (with ongoing monitoring)

**Risk Category: Performance Degradation**
- Risk: Model accuracy may decline due to data drift or changing patterns
- Mitigation: Ongoing performance monitoring, periodic retraining, drift detection
- Residual Risk: Low (with validation triggers)

**Risk Category: Misuse**
- Risk: System used outside intended purpose or on inappropriate populations
- Mitigation: Clear documentation, user training, access controls, human oversight
- Residual Risk: Medium (requires operational controls)

**Risk Category: Data Privacy**
- Risk: Processing of personal data in model training and inference
- Mitigation: GDPR compliance, data minimization, anonymization where appropriate
- Residual Risk: Low (with privacy controls)

### 4.2 Risk Assessment Methodology

**Risk Classification:** Tier 1

**Assessment Criteria:**
- Impact: Materiality of decisions influenced by the AI system
- Likelihood: Probability of risk scenarios occurring
- Detectability: Ability to identify risk events through monitoring
- Controllability: Effectiveness of mitigation measures

### 4.3 Risk Mitigation Measures

| Risk | Mitigation Measure | Responsible Party |
|------|-------------------|-------------------|
| Discrimination | Fairness testing and bias monitoring | Model Risk Team |
| Performance drift | Ongoing validation and retraining | Model Owners |
| Misuse | User training and access controls | Business Line |
| Data privacy | GDPR compliance framework | Data Protection Officer |

### 4.4 Post-Market Monitoring

**Monitoring Plan:**
- Continuous performance tracking against key metrics
- Regular validation testing per defined schedule
- Incident reporting and investigation procedures
- User feedback collection and analysis
- Periodic risk reassessment

## 5. EU AI Act Annex IV Compliance Matrix

The following matrix maps each Annex IV technical documentation requirement to the
validation evidence demonstrating compliance.

| Annex IV Ref | Requirement | Status | Evidence |
|--------------|-------------|--------|----------|
| Annex IV.1 | General Description of the AI System | NOT ASSESSED | No tests mapped |
| Annex IV.2 | Detailed Description of System Elements | NOT ASSESSED | No tests mapped |
| Annex IV.3 | Monitoring, Functioning and Control of the AI System | COMPLIANT | ccr.CVASensitivity: PASS |
| Annex IV.4 | Risk Management System | COMPLIANT | ccr.WrongWayRisk: PASS; compliance.GovernanceCheck: PASS |
| Annex IV.5 | Changes to the AI System Throughout Its Lifecycle | NOT ASSESSED | No tests mapped |
| Annex IV.6 | Harmonised Standards and Common Specifications | NOT ASSESSED | No tests mapped |
| Annex IV.7 | EU Declaration of Conformity | NOT ASSESSED | No tests mapped |
| Annex IV.8 | Detailed Description of System Performance Assessment | NON-COMPLIANT | ccr.MCConvergence: PASS; ccr.EPEReasonableness: PASS; ccr.PFEBacktest: FAIL; ccr.ExposureProfileShape: PASS |
| Annex IV.9 | Cybersecurity Measures | NOT ASSESSED | No tests mapped |

### 5.1 Compliance Summary

Each row corresponds to one of the nine Annex IV technical documentation requirements.
Technical tests provide quantitative evidence that the AI system satisfies the essential
requirements of the EU AI Act. Where a requirement is marked "COMPLIANT", corresponding
validation tests have passed with results within acceptable conformity thresholds.

Per Article 16 of the EU AI Act, technical documentation shall be drawn up in such a way
that it demonstrates that the high-risk AI system complies with the requirements set out
in Chapter 2 of this Regulation and provides the necessary information for authorities
to assess the system's compliance.

## 6. Detailed Technical Test Results

### 6.1 ccr.MCConvergence

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9991 |
| Annex IV Reference | Annex IV.8: Detailed Description of System Performance Assessment |

**Annex IV Requirement:** A detailed description of the system for assessment of the AI system performance in the post-market phase, including: validation and testing procedures; metrics used to measure accuracy, robustness, cybersecurity and compliance with other requirements; identification of any known limitations regarding the AI system's performance; and any other relevant assessment of the AI system's performance.

### 6.2 ccr.EPEReasonableness

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9800 |
| Annex IV Reference | Annex IV.8: Detailed Description of System Performance Assessment |

**Annex IV Requirement:** A detailed description of the system for assessment of the AI system performance in the post-market phase, including: validation and testing procedures; metrics used to measure accuracy, robustness, cybersecurity and compliance with other requirements; identification of any known limitations regarding the AI system's performance; and any other relevant assessment of the AI system's performance.

### 6.3 ccr.PFEBacktest

| Field | Value |
|-------|-------|
| Status | **FAIL** |
| Score | 0.7400 |
| Annex IV Reference | Annex IV.8: Detailed Description of System Performance Assessment |
| Non-Conformity Reason | Backtest p-value below 0.05 |

**Annex IV Requirement:** A detailed description of the system for assessment of the AI system performance in the post-market phase, including: validation and testing procedures; metrics used to measure accuracy, robustness, cybersecurity and compliance with other requirements; identification of any known limitations regarding the AI system's performance; and any other relevant assessment of the AI system's performance.

### 6.4 ccr.CVASensitivity

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9200 |
| Annex IV Reference | Annex IV.3: Monitoring, Functioning and Control of the AI System |

**Annex IV Requirement:** Detailed information about the monitoring, functioning and control of the AI system, in particular with regard to: its capabilities and limitations in performance, including the degrees of accuracy for specific persons or groups of persons on which the system is intended to be used and the overall expected level of accuracy in relation to its intended purpose; the foreseeable circumstances which may lead to risks to the health and safety or fundamental rights; and the human oversight measures needed.

### 6.5 ccr.WrongWayRisk

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.8800 |
| Annex IV Reference | Annex IV.4: Risk Management System |

**Annex IV Requirement:** A detailed description of the risk management system as referred to in Article 9, including: (a) the identification and analysis of the known and foreseeable risks to health, safety or fundamental rights; (b) the estimation and evaluation of the risks that may emerge when the AI system is used in accordance with its intended purpose and under conditions of reasonably foreseeable misuse; (c) the evaluation of other possibly arising risks based on the analysis of data gathered from the post-market monitoring system; (d) the adoption of suitable risk management measures as appropriate to address the specific risks.

### 6.6 ccr.ExposureProfileShape

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 0.9500 |
| Annex IV Reference | Annex IV.8: Detailed Description of System Performance Assessment |

**Annex IV Requirement:** A detailed description of the system for assessment of the AI system performance in the post-market phase, including: validation and testing procedures; metrics used to measure accuracy, robustness, cybersecurity and compliance with other requirements; identification of any known limitations regarding the AI system's performance; and any other relevant assessment of the AI system's performance.

### 6.7 compliance.GovernanceCheck

| Field | Value |
|-------|-------|
| Status | **PASS** |
| Score | 1.0000 |
| Annex IV Reference | Annex IV.4: Risk Management System |

**Annex IV Requirement:** A detailed description of the risk management system as referred to in Article 9, including: (a) the identification and analysis of the known and foreseeable risks to health, safety or fundamental rights; (b) the estimation and evaluation of the risks that may emerge when the AI system is used in accordance with its intended purpose and under conditions of reasonably foreseeable misuse; (c) the evaluation of other possibly arising risks based on the analysis of data gathered from the post-market monitoring system; (d) the adoption of suitable risk management measures as appropriate to address the specific risks.


## 7. System Performance Assessment (Annex IV.8)

### 7.1 Validation and Testing Procedures

**Validation Methodology:**
- Holdout validation dataset separate from training data
- Cross-validation during model development
- Independent testing on representative data samples
- Ongoing performance monitoring in production

**Testing Procedures:**
- Automated test suite executed at validation frequency
- Manual review and challenge by independent validators
- Benchmark comparison against baseline models
- Sensitivity analysis and stress testing

### 7.2 Performance Metrics

The system is assessed using the following standard metrics for classification models:

| Metric | Definition | Target Threshold |
|--------|-----------|------------------|
| Accuracy | Overall correctness of predictions | ≥ 70% |
| ROC-AUC | Area under receiver operating characteristic curve | ≥ 70% |
| Gini | Discriminatory power (2*AUC - 1) | ≥ 40% |
| Precision | Positive predictive value | ≥ 65% |
| Recall | Sensitivity / true positive rate | ≥ 65% |

### 7.3 Known Limitations

**Performance Limitations:**
- Model performance validated only on populations similar to training data
- Accuracy may vary for edge cases or rare scenarios
- Performance subject to data quality and availability
- Temporal stability requires periodic retraining

**Operational Limitations:**
- Requires appropriate feature inputs for reliable predictions
- Subject to computational resource constraints for real-time inference
- Dependent on data pipeline availability and integrity

### 7.4 Post-Market Monitoring Plan

**Monitoring Activities:**
- Continuous tracking of key performance metrics
- Regular validation against new data samples
- User feedback collection and analysis
- Incident and error tracking
- Periodic comprehensive re-validation

**Monitoring Frequency:** Quarterly validation with continuous operational monitoring

## 8. EU Declaration of Conformity (Annex IV.7)

### 8.1 Conformity Statement

This EU declaration of conformity is issued under the sole responsibility of the provider.

**Provider:** ccr-team

**AI System:** ccr_monte_carlo version 1.0.0

**Declaration:** The AI system described above is in conformity with Regulation (EU) 2024/1689 
(Artificial Intelligence Act) and, where applicable, with the following harmonised standards:

- EN ISO/IEC 23894:2023 (AI Risk Management)
- EN ISO/IEC 25059:2023 (AI System Quality Requirements)
- [Additional harmonised standards as applicable]

### 8.2 Conformity Assessment Procedure

**Procedure Applied:** Internal control based on Annex VI of the EU AI Act

**Conformity Assessment Body:** [If applicable]

**Certificate Number:** [If applicable]

### 8.3 Signatory Information

| Role | Name | Date |
|------|------|------|
| Provider Representative | [To be completed] | 2026-05-16 |
| Technical Documentation Preparer | [To be completed] | 2026-05-16 |
| Conformity Assessment Reviewer | [To be completed] | [Pending] |

---

## 9. Cybersecurity Measures (Annex IV.9)

### 9.1 Security Controls

**Access Controls:**
- Authentication and authorization for system access
- Role-based access control for model operations
- Audit logging of all system interactions

**Data Security:**
- Encryption of data at rest and in transit
- Secure storage of training and operational data
- Data integrity verification mechanisms

**System Integrity:**
- Version control and change management
- Model artifact signing and verification
- Secure deployment pipeline

### 9.2 Vulnerability Management

**Protective Measures:**
- Regular security assessments and penetration testing
- Vulnerability scanning and patch management
- Incident response procedures
- Business continuity and disaster recovery plans

**Training Data Protection:**
- Access controls for training datasets
- Data provenance tracking
- Protection against data poisoning attacks

### 9.3 Resilience Measures

**System Resilience:**
- Monitoring for adversarial attacks or manipulation attempts
- Anomaly detection in model inputs and outputs
- Fallback procedures for system failures
- Regular security reviews and updates

---

*This technical documentation has been prepared in accordance with Article 11 and 
Annex IV of Regulation (EU) 2024/1689 on artificial intelligence (EU AI Act).*

*Documentation Version: 1.0*

*Last Updated: 2026-05-16*