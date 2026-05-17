"""Tests for the new tabular.DataDrift + tabular.ConceptDrift mrm
tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrm.tests.library import registry


@pytest.fixture(autouse=True)
def _load_builtins():
    registry.load_builtin_tests()


def _instance(name: str):
    return registry.get(name)()


def test_data_drift_registered():
    assert "tabular.DataDrift" in registry.list_tests()


def test_concept_drift_registered():
    assert "tabular.ConceptDrift" in registry.list_tests()


def test_data_drift_passes_when_no_shift():
    test = _instance("tabular.DataDrift")
    rng = np.random.default_rng(seed=42)
    result = test.run(
        dataset={
            "reference": rng.normal(0, 1, size=400),
            "current": rng.normal(0, 1, size=400),
        }
    )
    assert result.passed is True
    assert result.details["detector"] == "ks"
    assert "p_value" in result.details


def test_data_drift_fails_when_shifted():
    test = _instance("tabular.DataDrift")
    rng = np.random.default_rng(seed=42)
    result = test.run(
        dataset={
            "reference": rng.normal(0, 1, size=400),
            "current": rng.normal(2.5, 1, size=400),
        }
    )
    assert result.passed is False
    assert "flagged drift" in (result.failure_reason or "")


def test_data_drift_accepts_pandas_dataframe_with_column():
    test = _instance("tabular.DataDrift")
    rng = np.random.default_rng(seed=42)
    ref_df = pd.DataFrame({"feature": rng.normal(0, 1, size=300), "other": [0] * 300})
    cur_df = pd.DataFrame({"feature": rng.normal(0, 1, size=300), "other": [0] * 300})
    result = test.run(dataset={"reference": ref_df, "current": cur_df}, column="feature")
    assert result.passed is True


def test_concept_drift_passes_for_stationary_stream():
    test = _instance("tabular.ConceptDrift")
    rng = np.random.default_rng(seed=42)
    result = test.run(
        dataset={
            "reference": rng.normal(0, 0.1, size=150),
            "current": rng.normal(0, 0.1, size=150),
        },
        threshold=50.0,
    )
    assert result.passed is True


def test_concept_drift_fires_on_mean_shift():
    test = _instance("tabular.ConceptDrift")
    rng = np.random.default_rng(seed=42)
    cur = np.concatenate([rng.normal(0, 0.1, size=100), rng.normal(1.5, 0.1, size=100)])
    result = test.run(
        dataset={"reference": rng.normal(0, 0.1, size=150), "current": cur},
        threshold=10.0,
    )
    assert result.passed is False
    assert "concept drift" in (result.failure_reason or "").lower()


def test_data_drift_can_choose_wasserstein():
    test = _instance("tabular.DataDrift")
    rng = np.random.default_rng(seed=42)
    result = test.run(
        dataset={
            "reference": rng.normal(0, 1, size=300),
            "current": rng.normal(3.0, 1, size=300),
        },
        detector="wasserstein",
    )
    assert result.details["detector"] == "wasserstein"
    assert result.passed is False  # very shifted -> Wasserstein > heuristic threshold
