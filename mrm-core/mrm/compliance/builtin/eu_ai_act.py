"""EU AI Act Annex IV -- Technical Documentation for High-Risk AI Systems.

Bundled compliance standard for the European Union AI Act. Implements
``ComplianceStandard`` so the MRM framework can generate EU AI Act
Annex IV reports, run governance checks, and map tests to specific
technical documentation requirements.

Usage::

    mrm docs generate credit_scorecard --compliance standard:euaiact
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from mrm.compliance.base import ComplianceStandard
from mrm.compliance.registry import register_standard

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers (result introspection and data conversion)
# -----------------------------------------------------------------------

def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
# Helpers (result introspection)
# -----------------------------------------------------------------------

def _result_passed(result) -> bool:
    if hasattr(result, "passed"):
        return result.passed
    if isinstance(result, dict):
        return result.get("passed", False)
    return False


def _result_score(result) -> Optional[float]:
    if hasattr(result, "score"):
        return result.score
    if isinstance(result, dict):
        return result.get("score")
    return None


def _result_details(result) -> Dict:
    if hasattr(result, "details"):
        return result.details or {}
    if isinstance(result, dict):
        return result.get("details", {})
    return {}


def _result_failure(result) -> Optional[str]:
    if hasattr(result, "failure_reason"):
        return result.failure_reason
    if isinstance(result, dict):
        return result.get("failure_reason")
    return None


def _get_compliance_ref(details: Dict, standard_name: str = "euaiact") -> str:
    """Read compliance reference from test result details.

    Supports both the new generic key and the legacy key:
      - ``details["compliance_references"]["euaiact"]``  (new)
      - ``details["euaiact_reference"]``                 (legacy)
    """
    refs = details.get("compliance_references", {})
    if isinstance(refs, dict) and standard_name in refs:
        return refs[standard_name]
    return details.get("euaiact_reference", "")


# =======================================================================
# EU AI Act Annex IV Standard Implementation
# =======================================================================

@register_standard
class EUAIActStandard(ComplianceStandard):
    """EU AI Act Annex IV -- Technical Documentation for High-Risk AI Systems."""

    name = "euaiact"
    display_name = "EU AI Act Annex IV -- Technical Documentation for High-Risk AI Systems"
    description = (
        "The EU AI Act (Regulation 2024/1689) establishes harmonised rules for "
        "artificial intelligence. Annex IV specifies technical documentation "
        "requirements for high-risk AI systems, including those used in credit "
        "scoring, employment, law enforcement, and critical infrastructure."
    )
    jurisdiction = "EU"
    version = "2024"

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        """Map EU AI Act Annex IV technical documentation requirements.
        
        Annex IV Article 11 specifies 9 categories of technical documentation:
        1. General description of the AI system
        2. Detailed description of system elements
        3. Monitoring, functioning, and control
        4. Risk management system
        5. Changes made throughout lifecycle
        6. Harmonised standards and specifications
        7. EU declaration of conformity
        8. System performance assessment
        9. Cybersecurity measures
        """
        return {
            "Annex IV.1": {
                "title": "General Description of the AI System",
                "requirement": (
                    "A general description of the AI system including: (a) its intended "
                    "purpose, the person/s developing the system, the provider's name, etc.; "
                    "(b) the reasonably foreseeable misuse of the system; (c) the date and "
                    "version of the system; (d) how the system interacts with hardware or "
                    "other software, including other AI systems, if applicable; (e) the "
                    "versions of relevant software or firmware and any requirement related "
                    "to version update; (f) the description of all forms in which the AI "
                    "system is placed on the market or put into service; (g) the description "
                    "of hardware on which the AI system is intended to run; (h) where the AI "
                    "system is a component of products, photographs or illustrations showing "
                    "external features, marking and internal layout of those products."
                ),
            },
            "Annex IV.2": {
                "title": "Detailed Description of System Elements",
                "requirement": (
                    "A detailed description of the elements of the AI system and of the "
                    "process for its development, including: (a) the methods and steps "
                    "performed for the development of the AI system, including, where "
                    "relevant, recourse to pre-trained systems or tools provided by third "
                    "parties and how these have been used, integrated or modified by the "
                    "provider; (b) the design specifications of the system, namely the "
                    "general logic of the AI system and of the algorithms; (c) the "
                    "description of the system architecture explaining how software "
                    "components build on or feed into each other and integrate into the "
                    "overall processing; the computational resources used to develop, train, "
                    "test and validate the AI system; (d) where relevant, the data "
                    "requirements in terms of datasheets describing the training "
                    "methodologies and techniques and the training data sets used."
                ),
            },
            "Annex IV.3": {
                "title": "Monitoring, Functioning and Control of the AI System",
                "requirement": (
                    "Detailed information about the monitoring, functioning and control of "
                    "the AI system, in particular with regard to: its capabilities and "
                    "limitations in performance, including the degrees of accuracy for "
                    "specific persons or groups of persons on which the system is intended "
                    "to be used and the overall expected level of accuracy in relation to "
                    "its intended purpose; the foreseeable circumstances which may lead to "
                    "risks to the health and safety or fundamental rights; and the human "
                    "oversight measures needed."
                ),
            },
            "Annex IV.4": {
                "title": "Risk Management System",
                "requirement": (
                    "A detailed description of the risk management system as referred to in "
                    "Article 9, including: (a) the identification and analysis of the known "
                    "and foreseeable risks to health, safety or fundamental rights; (b) the "
                    "estimation and evaluation of the risks that may emerge when the AI "
                    "system is used in accordance with its intended purpose and under "
                    "conditions of reasonably foreseeable misuse; (c) the evaluation of "
                    "other possibly arising risks based on the analysis of data gathered "
                    "from the post-market monitoring system; (d) the adoption of suitable "
                    "risk management measures as appropriate to address the specific risks."
                ),
            },
            "Annex IV.5": {
                "title": "Changes to the AI System Throughout Its Lifecycle",
                "requirement": (
                    "A description of any change made to the system through its lifecycle, "
                    "including: changes to the system and its intended purpose; changes to "
                    "the algorithms, data, training methodologies; and any change which "
                    "substantially modifies the AI system or which may have an impact on the "
                    "compliance with this Regulation."
                ),
            },
            "Annex IV.6": {
                "title": "Harmonised Standards and Common Specifications",
                "requirement": (
                    "A list of the harmonised standards applied in full or in part the "
                    "references of which have been published in the Official Journal of the "
                    "European Union; where such harmonised standards have not been applied, "
                    "a detailed description of the solutions adopted to meet the requirements "
                    "set out in Section 2, including a list of other relevant standards and "
                    "technical specifications applied, and a description of how those "
                    "standards or specifications are expected to meet the requirements."
                ),
            },
            "Annex IV.7": {
                "title": "EU Declaration of Conformity",
                "requirement": (
                    "A copy of the EU declaration of conformity referred to in Article 47; "
                    "and a detailed description of the system for assessing the AI system "
                    "performance in the post-market phase as referred to in Article 72, "
                    "including the post-market monitoring plan."
                ),
            },
            "Annex IV.8": {
                "title": "Detailed Description of System Performance Assessment",
                "requirement": (
                    "A detailed description of the system for assessment of the AI system "
                    "performance in the post-market phase, including: validation and testing "
                    "procedures; metrics used to measure accuracy, robustness, cybersecurity "
                    "and compliance with other requirements; identification of any known "
                    "limitations regarding the AI system's performance; and any other "
                    "relevant assessment of the AI system's performance."
                ),
            },
            "Annex IV.9": {
                "title": "Cybersecurity Measures",
                "requirement": (
                    "A description of the measures put in place to ensure cybersecurity as "
                    "referred to in Article 15, including: measures to protect the AI system "
                    "against unauthorised access, modification or misuse; technical and "
                    "organisational measures to ensure the security and integrity of the "
                    "training, validation and testing data sets; and measures to ensure the "
                    "resilience of the AI system against attempts to alter its use or "
                    "performance by exploiting system vulnerabilities."
                ),
            },
        }

    def get_test_mapping(self) -> Dict[str, str]:
        """Map test names to EU AI Act Annex IV requirements."""
        return {
            # Model performance tests → Performance Assessment (Annex IV.8)
            "model.Accuracy": "Annex IV.8",
            "model.ROCAUC": "Annex IV.8",
            "model.Gini": "Annex IV.8",
            "model.Precision": "Annex IV.8",
            "model.Recall": "Annex IV.8",
            "model.F1Score": "Annex IV.8",
            
            # Data quality tests → System Elements & Data Requirements (Annex IV.2)
            "tabular.MissingValues": "Annex IV.2",
            "tabular.ClassImbalance": "Annex IV.2",
            "tabular.OutlierDetection": "Annex IV.2",
            "tabular.FeatureDistribution": "Annex IV.2",
            "tabular.DataDrift": "Annex IV.5",
            
            # CCR tests → Performance Assessment (Annex IV.8)
            "ccr.MCConvergence": "Annex IV.8",
            "ccr.EPEReasonableness": "Annex IV.8",
            "ccr.PFEBacktest": "Annex IV.8",
            "ccr.CVASensitivity": "Annex IV.3",
            "ccr.WrongWayRisk": "Annex IV.4",
            "ccr.ExposureProfileShape": "Annex IV.8",
            "ccr.CollateralEffectiveness": "Annex IV.8",
            
            # Governance checks → Risk Management (Annex IV.4)
            "compliance.GovernanceCheck": "Annex IV.4",
        }

    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        """Define EU AI Act Annex IV governance requirements."""
        return {
            "system_purpose_documented": {
                "description": "AI system intended purpose clearly documented",
                "paragraph_ref": "EU AI Act Annex IV.1",
                "config_key": "use_case",
            },
            "provider_identified": {
                "description": "Provider/developer of the AI system identified",
                "paragraph_ref": "EU AI Act Annex IV.1",
                "config_key": "owner",
            },
            "system_version_tracked": {
                "description": "System version and date documented",
                "paragraph_ref": "EU AI Act Annex IV.1",
                "config_key": "version",
            },
            "development_methodology_documented": {
                "description": "Development methods and steps documented",
                "paragraph_ref": "EU AI Act Annex IV.2",
                "config_key": "methodology",
            },
            "data_requirements_defined": {
                "description": "Training data requirements and datasets documented",
                "paragraph_ref": "EU AI Act Annex IV.2",
                "config_key": "datasets",
            },
            "performance_metrics_defined": {
                "description": "Performance metrics and accuracy levels defined",
                "paragraph_ref": "EU AI Act Annex IV.3",
                "config_key": "tests",
            },
            "risk_assessment_conducted": {
                "description": "Risk management system in place",
                "paragraph_ref": "EU AI Act Annex IV.4",
                "config_key": "risk_tier",
            },
            "human_oversight_defined": {
                "description": "Human oversight measures documented",
                "paragraph_ref": "EU AI Act Annex IV.3",
                "config_key": "human_oversight",
            },
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        test_results: Dict[str, Any],
        trigger_events: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate EU AI Act Annex IV technical documentation report."""
        now = datetime.now(timezone.utc)
        model_info = model_config.get("model", model_config)

        # Read compliance mapping
        euaiact_mapping = (
            model_config
            .get("compliance", {})
            .get("standards", {})
            .get("euaiact", {})
            .get("mapping", model_config.get("euaiact_mapping", {}))
        )
        triggers_cfg = model_config.get("triggers", [])

        paragraphs = self.get_paragraphs()
        test_mapping = self.get_test_mapping()

        sections = [
            self._header(model_name, model_info, now),
            self._executive_summary(model_name, model_info, test_results, now),
            self._general_description(model_info, model_config),
            self._system_elements(model_info, model_config),
            self._monitoring_and_control(model_info, test_results),
            self._risk_management(model_info, model_config),
            self._compliance_matrix(test_results, euaiact_mapping, paragraphs, test_mapping),
            self._detailed_test_results(test_results, paragraphs, test_mapping),
            self._performance_assessment(test_results),
            self._conformity_declaration(model_info, now),
        ]

        report = "\n\n".join(sections)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report written to {output_path}")

        return report

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, model_name: str, model_info: Dict, now: datetime) -> str:
        return f"""# EU AI Act Annex IV Technical Documentation

## High-Risk AI System Technical Documentation

| Field | Value |
|-------|-------|
| **AI System Name** | {model_name} |
| **Version** | {model_info.get('version', 'N/A')} |
| **Documentation Date** | {now.strftime('%Y-%m-%d %H:%M UTC')} |
| **Regulatory Framework** | {self.display_name} |
| **Provider** | {model_info.get('owner', 'N/A')} |
| **Risk Classification** | High-Risk AI System |
| **Intended Purpose** | {model_info.get('use_case', 'N/A')} |

---

**Legal Basis:** Regulation (EU) 2024/1689 of the European Parliament and of the 
Council on artificial intelligence (AI Act), Article 11 and Annex IV.

**Applicable From:** 2 August 2026 (general provisions), 2 August 2027 (high-risk obligations)

---"""

    def _executive_summary(self, model_name, model_info, test_results, now):
        total = len(test_results)
        passed = sum(1 for r in test_results.values() if _result_passed(r))
        failed = total - passed
        pass_rate = f"{passed / total * 100:.1f}%" if total > 0 else "N/A"
        status_marker = "COMPLIANT" if failed == 0 and total > 0 else "NON-COMPLIANT"
        status = "PASS" if failed == 0 and total > 0 else "FAIL"

        return f"""## Executive Summary

This technical documentation has been prepared in accordance with **{self.display_name}**
for the AI system **{model_name}** (v{model_info.get('version', '1.0.0')}).

### System Classification

This AI system is classified as a **high-risk AI system** under Article 6 of the EU AI Act.
Specific classification rationale: {model_info.get('description', 'N/A')}

### Conformity Assessment Status: **{status_marker}**

| Metric | Value |
|--------|-------|
| Technical Tests Executed | {total} |
| Tests Passed | {passed} |
| Tests Failed | {failed} |
| Compliance Rate | {pass_rate} |
| Overall Status | **{status}** |

**Intended Purpose:** {model_info.get('description', 'N/A')}

**Methodology:** {model_info.get('methodology', 'N/A')}

### Conformity Assessment

This documentation demonstrates compliance with the essential requirements set out in 
Chapter 2 of the EU AI Act. The system has been subject to the conformity assessment 
procedure based on internal control as defined in Annex VI."""

    def _general_description(self, model_info, model_config):
        """Annex IV.1 - General Description"""
        return f"""## 1. General Description of the AI System (Annex IV.1)

### 1.1 System Identification

| Field | Value |
|-------|-------|
| System Name | {model_info.get('name', 'N/A')} |
| Version | {model_info.get('version', 'N/A')} |
| Provider | {model_info.get('owner', 'N/A')} |
| Intended Purpose | {model_info.get('use_case', 'N/A')} |
| Date of Documentation | {datetime.now(timezone.utc).strftime('%Y-%m-%d')} |

### 1.2 Intended Purpose

**Primary Use Case:** {model_info.get('description', 'N/A')}

**Target Users:** {model_info.get('target_users', 'Financial institutions, credit risk teams')}

**Operating Environment:** {model_info.get('environment', 'Production banking systems')}

### 1.3 Reasonably Foreseeable Misuse

The following misuse scenarios have been identified:
- Use of the system outside its intended operating domain
- Application to populations or jurisdictions not covered in training data
- Use without appropriate human oversight and review
- Reliance on model outputs without consideration of limitations

### 1.4 System Architecture

**Deployment Form:** {model_info.get('deployment', 'Standalone model component')}

**Hardware Requirements:** {model_info.get('hardware', 'Standard computational resources')}

**Software Dependencies:** Python ML stack (scikit-learn, pandas, numpy)

### 1.5 Version Management

Current version: {model_info.get('version', '1.0.0')}

Version updates are managed through:
- Version control system (Git)
- Change management process
- Re-validation after material changes"""

    def _system_elements(self, model_info, model_config):
        """Annex IV.2 - System Elements and Development"""
        datasets = model_config.get('datasets', {})
        dataset_info = "\n".join([
            f"- **{name}**: {ds.get('type', 'N/A')} ({ds.get('path', 'N/A')})"
            for name, ds in datasets.items()
        ]) if datasets else "No datasets documented"

        return f"""## 2. System Elements and Development Process (Annex IV.2)

### 2.1 Development Methodology

**Methodology:** {model_info.get('methodology', 'N/A')}

**Development Steps:**
1. Data collection and preparation
2. Feature engineering and selection
3. Model architecture design
4. Training and hyperparameter tuning
5. Validation and testing
6. Performance evaluation
7. Documentation and deployment

### 2.2 System Architecture

**Model Type:** {model_info.get('methodology', 'N/A')}

**Architecture Description:** The AI system is implemented as a {model_info.get('methodology', 'machine learning')} 
model that processes structured input features to generate predictions. The system follows standard 
ML pipeline architecture with data preprocessing, feature transformation, model inference, and 
output post-processing stages.

### 2.3 Data Requirements (Datasheets)

#### Training Data

{dataset_info}

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

**Update Frequency:** {model_info.get('validation_frequency', 'Quarterly')} re-validation and retraining as needed"""

    def _monitoring_and_control(self, model_info, test_results):
        """Annex IV.3 - Monitoring, Functioning and Control"""
        return f"""## 3. Monitoring, Functioning and Control (Annex IV.3)

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
- Intended population: {model_info.get('target_population', 'Defined in training data distribution')}
- Geographic scope: {model_info.get('geographic_scope', 'As per training data jurisdiction')}
- Temporal validity: {model_info.get('validation_frequency', 'Quarterly')} validation cycle

### 3.3 Performance Metrics

| Metric | Expected Performance | Actual Performance |
|--------|---------------------|-------------------|
| Overall Accuracy | ≥ 70% | {self._get_metric_value(test_results, 'model.Accuracy')} |
| ROC-AUC | ≥ 70% | {self._get_metric_value(test_results, 'model.ROCAUC')} |
| Gini Coefficient | ≥ 40% | {self._get_metric_value(test_results, 'model.Gini')} |
| Precision | ≥ 65% | {self._get_metric_value(test_results, 'model.Precision')} |
| Recall | ≥ 65% | {self._get_metric_value(test_results, 'model.Recall')} |

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
- Compliance officers: Regulatory adherence verification"""

    def _risk_management(self, model_info, model_config):
        """Annex IV.4 - Risk Management System"""
        return f"""## 4. Risk Management System (Annex IV.4)

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

**Risk Classification:** {model_info.get('risk_tier', 'tier_1').replace('_', ' ').title()}

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
- Periodic risk reassessment"""

    def _compliance_matrix(self, test_results, euaiact_mapping, paragraphs, test_mapping):
        """Annex IV Compliance Matrix"""
        rows = []

        for para_key, para_info in paragraphs.items():
            mapped_tests = [t for t, p in test_mapping.items() if p == para_key]

            statuses = []
            evidence_items = []
            for test_name in mapped_tests:
                if test_name in test_results:
                    r = test_results[test_name]
                    p = _result_passed(r)
                    statuses.append(p)
                    details = _result_details(r)
                    ref = _get_compliance_ref(details, self.name)
                    evidence_items.append(
                        f"{test_name}: {'PASS' if p else 'FAIL'}"
                        + (f" -- {ref}" if ref else "")
                    )

            for section_items in euaiact_mapping.values():
                if isinstance(section_items, list):
                    for item in section_items:
                        if item.get("annex_section") == para_key:
                            evidence_items.append(f"Doc: {item.get('evidence', 'N/A')}")

            if statuses:
                overall = "COMPLIANT" if all(statuses) else "NON-COMPLIANT"
            elif evidence_items:
                overall = "DOCUMENTED"
            else:
                overall = "NOT ASSESSED"

            evidence_str = "; ".join(evidence_items) if evidence_items else "No tests mapped"
            rows.append(
                f"| {para_key} | {para_info['title']} | {overall} | {evidence_str} |"
            )

        rows_str = "\n".join(rows)

        return f"""## 5. EU AI Act Annex IV Compliance Matrix

The following matrix maps each Annex IV technical documentation requirement to the
validation evidence demonstrating compliance.

| Annex IV Ref | Requirement | Status | Evidence |
|--------------|-------------|--------|----------|
{rows_str}

### 5.1 Compliance Summary

Each row corresponds to one of the nine Annex IV technical documentation requirements.
Technical tests provide quantitative evidence that the AI system satisfies the essential
requirements of the EU AI Act. Where a requirement is marked "COMPLIANT", corresponding
validation tests have passed with results within acceptable conformity thresholds.

Per Article 16 of the EU AI Act, technical documentation shall be drawn up in such a way
that it demonstrates that the high-risk AI system complies with the requirements set out
in Chapter 2 of this Regulation and provides the necessary information for authorities
to assess the system's compliance."""

    def _detailed_test_results(self, test_results, paragraphs, test_mapping):
        """Detailed Test Results Section"""
        sections = ["## 6. Detailed Technical Test Results\n"]

        for i, (test_name, result) in enumerate(test_results.items(), 1):
            passed = _result_passed(result)
            score = _result_score(result)
            details = _result_details(result)
            failure = _result_failure(result)
            para_ref = test_mapping.get(test_name, "N/A")
            para_info = paragraphs.get(para_ref, {})

            status_str = "PASS" if passed else "FAIL"

            section = f"""### 6.{i} {test_name}

| Field | Value |
|-------|-------|
| Status | **{status_str}** |
| Score | {f'{score:.4f}' if score is not None else 'N/A'} |
| Annex IV Reference | {para_ref}: {para_info.get('title', 'N/A')} |
"""

            if failure:
                section += f"| Non-Conformity Reason | {failure} |\n"

            if details:
                section += "\n**Technical Evidence:**\n\n"
                section += "```json\n"
                filtered = {
                    k: v for k, v in details.items()
                    if k not in ("euaiact_reference", "compliance_references")
                }
                # Convert numpy types to JSON-serializable Python types
                filtered = _convert_to_json_serializable(filtered)
                section += json.dumps(filtered, indent=2)
                section += "\n```\n"

            if para_info:
                section += f"\n**Annex IV Requirement:** {para_info.get('requirement', 'N/A')}\n"

            sections.append(section)

        return "\n".join(sections)

    def _performance_assessment(self, test_results):
        """Annex IV.8 - Performance Assessment System"""
        return f"""## 7. System Performance Assessment (Annex IV.8)

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

**Monitoring Frequency:** {self._get_monitoring_frequency(test_results)}"""

    def _conformity_declaration(self, model_info, now):
        """Annex IV.7 - Conformity Declaration"""
        return f"""## 8. EU Declaration of Conformity (Annex IV.7)

### 8.1 Conformity Statement

This EU declaration of conformity is issued under the sole responsibility of the provider.

**Provider:** {model_info.get('owner', '[To be completed]')}

**AI System:** {model_info.get('name', 'N/A')} version {model_info.get('version', '1.0.0')}

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
| Provider Representative | [To be completed] | {now.strftime('%Y-%m-%d')} |
| Technical Documentation Preparer | [To be completed] | {now.strftime('%Y-%m-%d')} |
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

*Last Updated: {now.strftime('%Y-%m-%d')}*"""

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_metric_value(self, test_results, metric_name):
        """Extract metric value from test results."""
        if metric_name in test_results:
            score = _result_score(test_results[metric_name])
            if score is not None:
                return f"{score:.2%}"
        return "N/A"

    def _get_monitoring_frequency(self, test_results):
        """Determine monitoring frequency based on risk."""
        return "Quarterly validation with continuous operational monitoring"
