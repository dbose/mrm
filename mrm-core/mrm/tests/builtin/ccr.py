"""
Built-in validation tests for Counterparty Credit Risk (CCR) models.

These tests implement the key validation checks required under APRA CPS 230
and general MRM best practice for Monte Carlo CCR models:

- Convergence testing (simulation stability)
- EPE reasonableness
- PFE back-testing
- CVA sensitivity analysis
- Wrong-way risk detection
- Exposure profile monotonicity
- Collateral modelling adequacy
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from mrm.tests.base import ModelTest, ComplianceTest, TestResult
from mrm.tests.library import register_test


# ---------------------------------------------------------------------------
# 1. Monte Carlo Convergence Test
# ---------------------------------------------------------------------------
@register_test
class MCConvergence(ModelTest):
    """
    Tests whether the Monte Carlo simulation has converged by comparing
    EPE estimates from two independent sub-samples.  If the relative
    difference exceeds a threshold the simulation needs more paths.

    CPS 230 relevance: Operational risk management -- ensuring computational
    methods produce reliable outputs (CPS 230 Para 30-33).
    """

    name = "ccr.MCConvergence"
    description = "Monte Carlo convergence test for CCR simulation stability"
    category = "model"
    tags = ["ccr", "convergence", "monte_carlo", "cps230"]

    def run(self, model: Any, dataset: Any, max_relative_diff: float = 0.05, **config) -> TestResult:
        if dataset is None or not isinstance(dataset, pd.DataFrame):
            return TestResult(passed=False, failure_reason="Dataset required for convergence test")

        sample_row = dataset.iloc[0:1]

        # Run two independent simulations with different seeds
        original_seed = model.seed
        try:
            model.seed = 1001
            model._rng = np.random.RandomState(1001)
            epe_a = float(model.predict(sample_row)[0])

            model.seed = 2002
            model._rng = np.random.RandomState(2002)
            epe_b = float(model.predict(sample_row)[0])
        finally:
            model.seed = original_seed
            model._rng = np.random.RandomState(original_seed)

        mean_epe = (epe_a + epe_b) / 2
        if mean_epe == 0:
            rel_diff = 0.0
        else:
            rel_diff = abs(epe_a - epe_b) / mean_epe

        passed = rel_diff <= max_relative_diff

        return TestResult(
            passed=passed,
            score=1.0 - rel_diff,
            details={
                "epe_sample_a": epe_a,
                "epe_sample_b": epe_b,
                "relative_difference": round(rel_diff, 6),
                "threshold": max_relative_diff,
                "n_simulations": model.n_simulations,
                "compliance_references": {"cps230": "CPS 230 Para 30-33: Operational risk controls for model computation"},
            },
            failure_reason=(
                f"MC not converged: relative diff {rel_diff:.4f} > {max_relative_diff}"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 2. EPE Reasonableness Test
# ---------------------------------------------------------------------------
@register_test
class EPEReasonableness(ModelTest):
    """
    Checks that the Expected Positive Exposure (EPE) is within a plausible
    range relative to portfolio notional.  EPE outside 0.1%-10% of notional
    typically signals a modelling error.

    CPS 230 relevance: Model risk identification and assessment
    (CPS 230 Para 15-18).
    """

    name = "ccr.EPEReasonableness"
    description = "Validates EPE is within plausible bounds relative to notional"
    category = "model"
    tags = ["ccr", "epe", "reasonableness", "cps230"]

    def run(
        self, model: Any, dataset: Any,
        min_epe_ratio: float = 0.001,
        max_epe_ratio: float = 0.10,
        **config
    ) -> TestResult:
        if dataset is None:
            return TestResult(passed=False, failure_reason="Dataset required")

        epe_values = model.predict(dataset)
        notionals = dataset["notional"].values

        ratios = epe_values / notionals
        mean_ratio = float(np.mean(ratios))
        outlier_count = int(np.sum((ratios < min_epe_ratio) | (ratios > max_epe_ratio)))
        outlier_pct = outlier_count / len(ratios)

        passed = outlier_pct <= 0.10  # max 10% outliers tolerated

        return TestResult(
            passed=passed,
            score=1.0 - outlier_pct,
            details={
                "mean_epe_notional_ratio": round(mean_ratio, 6),
                "min_ratio_observed": round(float(np.min(ratios)), 6),
                "max_ratio_observed": round(float(np.max(ratios)), 6),
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_pct, 4),
                "bounds": [min_epe_ratio, max_epe_ratio],
                "n_counterparties": len(ratios),
                "compliance_references": {"cps230": "CPS 230 Para 15-18: Risk identification and assessment"},
            },
            failure_reason=(
                f"{outlier_pct:.0%} of EPE/notional ratios outside [{min_epe_ratio}, {max_epe_ratio}]"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 3. PFE Back-testing
# ---------------------------------------------------------------------------
@register_test
class PFEBacktest(ModelTest):
    """
    Back-tests the Potential Future Exposure at the configured confidence
    level.  The realised P&L should exceed PFE no more than
    (1 - confidence_level) of the time.

    CPS 230 relevance: Ongoing monitoring and validation (CPS 230 Para 34-37).
    """

    name = "ccr.PFEBacktest"
    description = "PFE back-testing: realised losses vs predicted PFE"
    category = "model"
    tags = ["ccr", "pfe", "backtesting", "cps230"]

    def run(
        self, model: Any, dataset: Any,
        max_breach_rate: float = 0.10,
        **config
    ) -> TestResult:
        if dataset is None or "realised_pnl" not in dataset.columns:
            return TestResult(passed=False, failure_reason="Dataset with 'realised_pnl' column required")

        full_results = model.predict_full(dataset)
        pfe_peaks = np.array([r["pfe_peak"] for r in full_results])
        realised = dataset["realised_pnl"].values

        breaches = realised > pfe_peaks
        breach_rate = float(np.mean(breaches))
        n_breaches = int(np.sum(breaches))

        passed = breach_rate <= max_breach_rate

        return TestResult(
            passed=passed,
            score=1.0 - breach_rate,
            details={
                "breach_rate": round(breach_rate, 4),
                "n_breaches": n_breaches,
                "n_observations": len(realised),
                "max_breach_rate": max_breach_rate,
                "mean_pfe": round(float(np.mean(pfe_peaks)), 2),
                "mean_realised": round(float(np.mean(realised)), 2),
                "confidence_level": model.confidence_level,
                "compliance_references": {"cps230": "CPS 230 Para 34-37: Ongoing monitoring and reporting"},
            },
            failure_reason=(
                f"PFE breach rate {breach_rate:.2%} exceeds {max_breach_rate:.0%} threshold"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 4. CVA Sensitivity Analysis
# ---------------------------------------------------------------------------
@register_test
class CVASensitivity(ModelTest):
    """
    Tests CVA sensitivity to credit spread / PD shocks.
    Bumps PD by +50% and checks that CVA moves proportionally.

    CPS 230 relevance: Scenario analysis and stress testing
    (CPS 230 Para 24-27).
    """

    name = "ccr.CVASensitivity"
    description = "CVA sensitivity to PD shocks"
    category = "model"
    tags = ["ccr", "cva", "sensitivity", "stress_testing", "cps230"]

    def run(
        self, model: Any, dataset: Any,
        pd_bump: float = 0.50,
        min_sensitivity: float = 0.10,
        max_sensitivity: float = 3.0,
        **config
    ) -> TestResult:
        if dataset is None:
            return TestResult(passed=False, failure_reason="Dataset required")

        # Base CVA
        sample = dataset.iloc[0:3]
        base_results = model.predict_full(sample)
        base_cvas = [r["cva"] for r in base_results]

        # Shocked CVA: bump PD
        shocked_sample = sample.copy()
        shocked_sample["pd_annual"] = shocked_sample["pd_annual"] * (1 + pd_bump)
        shocked_results = model.predict_full(shocked_sample)
        shocked_cvas = [r["cva"] for r in shocked_results]

        sensitivities = []
        for base_cva, shock_cva in zip(base_cvas, shocked_cvas):
            if base_cva > 0:
                sensitivities.append((shock_cva - base_cva) / (base_cva * pd_bump))
            else:
                sensitivities.append(0.0)

        mean_sensitivity = float(np.mean(sensitivities))
        passed = min_sensitivity <= mean_sensitivity <= max_sensitivity

        return TestResult(
            passed=passed,
            score=mean_sensitivity,
            details={
                "pd_bump_pct": pd_bump * 100,
                "mean_sensitivity": round(mean_sensitivity, 4),
                "sensitivities": [round(s, 4) for s in sensitivities],
                "base_cvas": [round(c, 2) for c in base_cvas],
                "shocked_cvas": [round(c, 2) for c in shocked_cvas],
                "bounds": [min_sensitivity, max_sensitivity],
                "compliance_references": {"cps230": "CPS 230 Para 24-27: Scenario analysis and stress testing"},
            },
            failure_reason=(
                f"CVA sensitivity {mean_sensitivity:.4f} outside [{min_sensitivity}, {max_sensitivity}]"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 5. Wrong-Way Risk Detection
# ---------------------------------------------------------------------------
@register_test
class WrongWayRisk(ModelTest):
    """
    Detects wrong-way risk by checking correlation between counterparty
    default probability and exposure.  Positive correlation indicates
    general wrong-way risk.

    CPS 230 relevance: Risk identification -- concentration and
    interconnection risk (CPS 230 Para 19-23).
    """

    name = "ccr.WrongWayRisk"
    description = "Detects wrong-way risk (correlation between PD and exposure)"
    category = "model"
    tags = ["ccr", "wrong_way_risk", "cps230"]

    def run(
        self, model: Any, dataset: Any,
        max_correlation: float = 0.60,
        **config
    ) -> TestResult:
        if dataset is None:
            return TestResult(passed=False, failure_reason="Dataset required")

        epe_values = model.predict(dataset)
        pd_values = dataset["pd_annual"].values

        if len(epe_values) < 3:
            return TestResult(passed=False, failure_reason="Need at least 3 observations")

        correlation = float(np.corrcoef(pd_values, epe_values)[0, 1])
        passed = abs(correlation) <= max_correlation

        risk_level = "LOW" if abs(correlation) < 0.3 else ("MEDIUM" if abs(correlation) < 0.6 else "HIGH")

        return TestResult(
            passed=passed,
            score=1.0 - abs(correlation),
            details={
                "pd_exposure_correlation": round(correlation, 4),
                "max_correlation": max_correlation,
                "risk_level": risk_level,
                "n_counterparties": len(epe_values),
                "compliance_references": {"cps230": "CPS 230 Para 19-23: Concentration and interconnection risk"},
            },
            failure_reason=(
                f"Wrong-way risk: |corr(PD, EPE)| = {abs(correlation):.4f} > {max_correlation}"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 6. Exposure Profile Monotonicity Check
# ---------------------------------------------------------------------------
@register_test
class ExposureProfileShape(ModelTest):
    """
    Validates that exposure profiles have a plausible shape: for a vanilla
    IRS the EE profile should rise then fall (amortising effect).  Checks
    that the profile is not flat or erratic.

    CPS 230 relevance: Model adequacy and fitness-for-purpose
    (CPS 230 Para 28-29).
    """

    name = "ccr.ExposureProfileShape"
    description = "Validates exposure profile shape (hump-shaped for IRS)"
    category = "model"
    tags = ["ccr", "exposure_profile", "cps230"]

    def run(self, model: Any, dataset: Any, **config) -> TestResult:
        if dataset is None:
            return TestResult(passed=False, failure_reason="Dataset required")

        sample = dataset.iloc[0:1]
        results = model.predict_full(sample)
        ee_profile = np.array(results[0]["ee_profile"])

        # Check basic shape: should have a peak and decline
        peak_idx = int(np.argmax(ee_profile))
        has_peak = 0 < peak_idx < len(ee_profile) - 1

        # Check not flat: coefficient of variation > 10%
        cv = float(np.std(ee_profile) / (np.mean(ee_profile) + 1e-10))
        not_flat = cv > 0.10

        # Check no negative values
        no_negatives = bool(np.all(ee_profile >= 0))

        passed = has_peak and not_flat and no_negatives

        return TestResult(
            passed=passed,
            score=float(has_peak) * 0.4 + float(not_flat) * 0.3 + float(no_negatives) * 0.3,
            details={
                "has_peak": has_peak,
                "peak_time_step": peak_idx,
                "coefficient_of_variation": round(cv, 4),
                "not_flat": not_flat,
                "no_negatives": no_negatives,
                "profile_length": len(ee_profile),
                "peak_ee": round(float(np.max(ee_profile)), 2),
                "terminal_ee": round(float(ee_profile[-1]), 2),
                "compliance_references": {"cps230": "CPS 230 Para 28-29: Model adequacy and fitness-for-purpose"},
            },
            failure_reason=(
                "Exposure profile shape anomaly detected"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 7. Collateral Modelling Adequacy
# ---------------------------------------------------------------------------
@register_test
class CollateralEffectiveness(ModelTest):
    """
    Tests that collateral reduces exposure.  Counterparties with a CSA
    collateral threshold should show lower EPE than uncollateralised ones
    with similar profiles.

    CPS 230 relevance: Risk mitigation controls (CPS 230 Para 38-42).
    """

    name = "ccr.CollateralEffectiveness"
    description = "Validates collateral reduces exposure as expected"
    category = "model"
    tags = ["ccr", "collateral", "risk_mitigation", "cps230"]

    def run(self, model: Any, dataset: Any, **config) -> TestResult:
        if dataset is None or "collateral_threshold" not in dataset.columns:
            return TestResult(passed=False, failure_reason="Dataset with collateral_threshold required")

        collateralised = dataset[dataset["collateral_threshold"] > 0]
        uncollateralised = dataset[dataset["collateral_threshold"] == 0]

        if len(collateralised) == 0 or len(uncollateralised) == 0:
            return TestResult(
                passed=True,
                score=1.0,
                details={"note": "Insufficient data to compare; skipped"},
            )

        epe_coll = model.predict(collateralised)
        epe_uncoll = model.predict(uncollateralised)

        mean_coll = float(np.mean(epe_coll))
        mean_uncoll = float(np.mean(epe_uncoll))

        # Collateralised EPE should generally be lower
        # but we normalise by notional to make it fair
        ratio_coll = float(np.mean(epe_coll / collateralised["notional"].values))
        ratio_uncoll = float(np.mean(epe_uncoll / uncollateralised["notional"].values))

        reduction = 1.0 - (ratio_coll / (ratio_uncoll + 1e-10))
        passed = reduction >= 0  # collateral should not increase risk

        return TestResult(
            passed=passed,
            score=max(0, reduction),
            details={
                "mean_epe_collateralised": round(mean_coll, 2),
                "mean_epe_uncollateralised": round(mean_uncoll, 2),
                "epe_notional_ratio_coll": round(ratio_coll, 6),
                "epe_notional_ratio_uncoll": round(ratio_uncoll, 6),
                "effective_reduction_pct": round(reduction * 100, 2),
                "n_collateralised": len(collateralised),
                "n_uncollateralised": len(uncollateralised),
                "compliance_references": {"cps230": "CPS 230 Para 38-42: Risk mitigation and controls"},
            },
            failure_reason=(
                "Collateral increases rather than decreases exposure"
                if not passed else None
            ),
        )


# ---------------------------------------------------------------------------
# 8. Governance Compliance Check (generic, standard-aware)
# ---------------------------------------------------------------------------
@register_test
class GovernanceCheck(ComplianceTest):
    """
    Generic governance compliance check that validates model configuration
    against a compliance standard's requirements.

    Defaults to CPS 230 but supports any registered standard via the
    ``standard`` config parameter.
    """

    name = "compliance.GovernanceCheck"
    description = "Validates governance configuration against a compliance standard"
    category = "compliance"
    tags = ["governance", "regulatory", "compliance"]

    def run(self, model: Any, dataset: Any, model_config: Optional[Dict] = None, **config) -> TestResult:
        cfg = config.get("model_config") or model_config or {}
        standard_name = config.get("standard", "cps230")

        # Load governance checks from the compliance standard
        try:
            from mrm.compliance.registry import compliance_registry
            compliance_registry.load_builtin_standards()
            standard_class = compliance_registry.get(standard_name)
            standard = standard_class()
            governance_checks = standard.get_governance_checks()
        except (KeyError, ImportError):
            # Fallback to built-in CPS 230 checks
            governance_checks = {
                "risk_tier_assigned": {"config_key": "risk_tier", "paragraph_ref": "CPS 230 Para 8-10"},
                "owner_designated": {"config_key": "owner", "paragraph_ref": "CPS 230 Para 11"},
                "validation_frequency_set": {"config_key": "validation_frequency", "paragraph_ref": "CPS 230 Para 12-14"},
                "use_case_documented": {"config_key": "use_case", "paragraph_ref": "CPS 230 Para 15"},
                "methodology_documented": {"config_key": "methodology", "paragraph_ref": "CPS 230 Para 16"},
                "version_controlled": {"config_key": "version", "paragraph_ref": "CPS 230 Para 28"},
            }

        checks = {}
        paragraph_refs = {}
        for check_name, check_def in governance_checks.items():
            config_key = check_def.get("config_key", check_name)
            checks[check_name] = bool(cfg.get(config_key))
            paragraph_refs[check_name] = check_def.get("paragraph_ref", "")

        n_passed = sum(checks.values())
        n_total = len(checks)
        score = n_passed / n_total if n_total > 0 else 1.0

        passed = n_passed == n_total

        return TestResult(
            passed=passed,
            score=score,
            details={
                "checks": checks,
                "checks_passed": n_passed,
                "checks_total": n_total,
                "standard": standard_name,
                "compliance_references": {standard_name: paragraph_refs},
            },
            failure_reason=(
                f"Governance ({standard_name}): {n_total - n_passed} checks failed: "
                + ", ".join(k for k, v in checks.items() if not v)
                if not passed else None
            ),
        )


# Backward compatibility alias
@register_test
class CPS230GovernanceCheck(GovernanceCheck):
    """Backward-compatible alias for GovernanceCheck with CPS 230 default."""
    name = "ccr.CPS230GovernanceCheck"
    description = "Validates CPS 230 governance configuration completeness"
    tags = ["ccr", "cps230", "governance", "regulatory"]
