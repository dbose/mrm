"""
End-to-end CCR validation runner.

Exercises the full MRM pipeline:
  1. Load the CCR Monte Carlo model and datasets
  2. Run all 8 CCR validation tests
  3. Evaluate validation triggers
  4. Generate compliance report (via pluggable standard framework)
  5. Print summary to console

Usage:
    cd ccr_example && python run_validation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# MRM imports
from mrm.tests.library import registry
from mrm.tests.base import TestResult
from mrm.core.triggers import ValidationTriggerEngine
from mrm.compliance.report_generator import generate_compliance_report
from mrm.utils.yaml_utils import load_yaml

ROOT = Path(__file__).parent

def main():
    print("=" * 72)
    print("  CCR MONTE CARLO MODEL -- COMPLIANCE VALIDATION")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # 1. Load model & data
    # ------------------------------------------------------------------
    print("[1/5] Loading model and datasets...")

    model_path = ROOT / "models" / "ccr_monte_carlo.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    val_data = pd.read_csv(ROOT / "data" / "validation.csv")
    print(f"  Model:      {model.__class__.__name__}")
    print(f"  Simulations: {model.n_simulations}")
    print(f"  Time steps:  {model.n_time_steps}")
    print(f"  Dataset:     {len(val_data)} counterparties")

    # Load model config
    model_config = load_yaml(ROOT / "models" / "ccr" / "ccr_monte_carlo.yml")
    model_info = model_config.get("model", {})

    # ------------------------------------------------------------------
    # 2. Run all CCR validation tests
    # ------------------------------------------------------------------
    print("\n[2/5] Running CCR validation tests...\n")

    registry.load_builtin_tests()

    test_names = [
        "ccr.MCConvergence",
        "ccr.EPEReasonableness",
        "ccr.PFEBacktest",
        "ccr.CVASensitivity",
        "ccr.WrongWayRisk",
        "ccr.ExposureProfileShape",
        "ccr.CollateralEffectiveness",
        "compliance.GovernanceCheck",
    ]

    test_results = {}
    all_passed = True

    for test_name in test_names:
        test_class = registry.get(test_name)
        test_instance = test_class()

        # Special config for governance check
        if test_name == "compliance.GovernanceCheck":
            result = test_instance.run(
                model=model,
                dataset=val_data,
                model_config={
                    "risk_tier": model_info.get("risk_tier"),
                    "owner": model_info.get("owner"),
                    "validation_frequency": model_info.get("validation_frequency"),
                    "use_case": model_info.get("use_case"),
                    "methodology": model_info.get("methodology"),
                    "version": model_info.get("version"),
                },
            )
        else:
            result = test_instance.run(model=model, dataset=val_data)

        test_results[test_name] = result
        status = "PASS" if result.passed else "FAIL"
        score_str = f" (score: {result.score:.4f})" if result.score is not None else ""
        print(f"  [{status}] {test_name}{score_str}")

        if not result.passed:
            all_passed = False
            if result.failure_reason:
                print(f"         Reason: {result.failure_reason}")

    total = len(test_results)
    passed_count = sum(1 for r in test_results.values() if r.passed)
    failed_count = total - passed_count

    print(f"\n  Summary: {passed_count}/{total} passed, {failed_count} failed")

    # ------------------------------------------------------------------
    # 3. Evaluate triggers
    # ------------------------------------------------------------------
    print("\n[3/5] Evaluating validation triggers...\n")

    triggers_cfg = model_config.get("triggers", [])
    trigger_engine = ValidationTriggerEngine()

    fired_events = trigger_engine.evaluate(
        model_name="ccr_monte_carlo",
        trigger_configs=triggers_cfg,
        test_results=test_results,
    )

    if fired_events:
        for e in fired_events:
            print(f"  [FIRED] {e.trigger_type.value}: {e.reason}")
            print(f"          Compliance: {e.compliance_reference}")
    else:
        print("  No triggers fired.")

    # ------------------------------------------------------------------
    # 4. Generate compliance report
    # ------------------------------------------------------------------
    print("\n[4/5] Generating compliance regulatory report...\n")

    report_path = ROOT / "reports" / "ccr_monte_carlo_cps230_report.md"
    trigger_event_dicts = [e.to_dict() for e in fired_events]

    report_text = generate_compliance_report(
        standard_name="cps230",
        model_name="ccr_monte_carlo",
        model_config=model_config,
        test_results=test_results,
        trigger_events=trigger_event_dicts,
        output_path=report_path,
    )

    print(f"  Report written to: {report_path}")
    print(f"  Report size: {len(report_text):,} characters")

    # ------------------------------------------------------------------
    # 5. Save test results as JSON evidence
    # ------------------------------------------------------------------
    print("\n[5/5] Saving test evidence...\n")

    evidence = {
        "model": "ccr_monte_carlo",
        "version": model_info.get("version", "1.0.0"),
        "validation_date": datetime.now(timezone.utc).isoformat(),
        "compliance_standard": "cps230",
        "overall_status": "PASS" if all_passed else "FAIL",
        "tests_run": total,
        "tests_passed": passed_count,
        "tests_failed": failed_count,
        "results": {
            name: result.to_dict() for name, result in test_results.items()
        },
        "triggers_fired": len(fired_events),
        "trigger_events": trigger_event_dicts,
    }

    evidence_path = ROOT / "reports" / "validation_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"  Evidence saved to: {evidence_path}")

    # Resolve triggers after successful validation
    if all_passed:
        trigger_engine.resolve_model("ccr_monte_carlo")
        print("  Triggers resolved after successful validation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    overall = "PASSED" if all_passed else "FAILED"
    print(f"  VALIDATION {overall} -- {passed_count}/{total} tests passed")
    print("=" * 72)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
