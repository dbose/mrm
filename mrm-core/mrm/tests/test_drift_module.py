"""Tests for the drift detection module (P10)."""

from __future__ import annotations

import numpy as np
import pytest

from mrm.drift import (
    DriftKind,
    DriftResult,
    available_backends,
    get_detector,
    list_detectors,
)


# ---------------------------------------------------------------------------
# Registry + capability surface
# ---------------------------------------------------------------------------


def test_registry_contains_at_least_one_scipy_or_numpy_backend_per_detector():
    detectors = list_detectors()
    by_name: dict[str, list[str]] = {}
    for entry in detectors:
        by_name.setdefault(entry["name"], []).append(entry["backend"])
    for name in ("ks", "wasserstein", "page_hinkley", "mmd"):
        backends = by_name.get(name, [])
        # Every detector must ship at least one non-frouros backend so
        # the air-gapped install retains drift coverage.
        assert any(b in ("scipy", "numpy") for b in backends), (
            f"{name} has no scipy/numpy backend; only {backends}"
        )


def test_available_backends_always_includes_scipy_and_numpy():
    backends = set(available_backends())
    assert {"scipy", "numpy"}.issubset(backends)


# ---------------------------------------------------------------------------
# KS detector
# ---------------------------------------------------------------------------


def test_ks_no_drift_for_same_distribution():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=500)
    cur = rng.normal(0, 1, size=500)
    detector = get_detector("ks", prefer_backend="scipy")
    result = detector.fit_detect(ref, cur)
    assert result.kind is DriftKind.DATA
    assert result.drifted is False
    assert result.p_value is not None and result.p_value > 0.05


def test_ks_detects_shifted_distribution():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=500)
    cur = rng.normal(2, 1, size=500)  # shifted mean
    detector = get_detector("ks", prefer_backend="scipy")
    result = detector.fit_detect(ref, cur)
    assert result.drifted is True
    assert result.p_value is not None and result.p_value < 0.01


def test_ks_threshold_override():
    """Caller can override the p-value cutoff."""
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=500)
    cur = rng.normal(0, 1, size=500)
    detector = get_detector("ks", prefer_backend="scipy")
    # A very large threshold should flag noise as drift.
    result = detector.fit_detect(ref, cur, threshold=0.99)
    assert result.drifted is True


# ---------------------------------------------------------------------------
# Wasserstein detector
# ---------------------------------------------------------------------------


def test_wasserstein_no_drift_for_similar_distributions():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=500)
    cur = rng.normal(0, 1, size=500)
    result = get_detector("wasserstein", prefer_backend="scipy").fit_detect(ref, cur)
    assert result.score < 0.5  # distance should be small for same distribution


def test_wasserstein_detects_shifted_distribution():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=500)
    cur = rng.normal(3, 1, size=500)
    result = get_detector("wasserstein", prefer_backend="scipy").fit_detect(ref, cur)
    assert result.drifted is True
    assert result.score > 2.0


# ---------------------------------------------------------------------------
# Page-Hinkley concept-drift
# ---------------------------------------------------------------------------


def test_page_hinkley_no_drift_when_stream_stationary():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 0.1, size=200)
    cur = rng.normal(0, 0.1, size=200)
    result = get_detector("page_hinkley", prefer_backend="numpy").fit_detect(
        ref, cur, threshold=50.0
    )
    assert result.kind is DriftKind.CONCEPT
    assert result.drifted is False


def test_page_hinkley_detects_mean_shift_in_stream():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 0.1, size=200)
    # Inject a clear mean-shift partway through the current stream.
    cur = np.concatenate([
        rng.normal(0, 0.1, size=100),
        rng.normal(1.5, 0.1, size=100),
    ])
    result = get_detector("page_hinkley", prefer_backend="numpy").fit_detect(
        ref, cur, threshold=10.0
    )
    assert result.drifted is True
    assert result.score > 10.0


# ---------------------------------------------------------------------------
# MMD detector (semantic / embedding drift)
# ---------------------------------------------------------------------------


def test_mmd_no_drift_for_same_embedding_distribution():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=(80, 8))
    cur = rng.normal(0, 1, size=(80, 8))
    result = get_detector("mmd", prefer_backend="numpy").fit_detect(ref, cur)
    assert result.kind is DriftKind.SEMANTIC
    assert result.drifted is False


def test_mmd_detects_shifted_embedding_distribution():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=(80, 8))
    cur = rng.normal(2.5, 1, size=(80, 8))
    result = get_detector("mmd", prefer_backend="numpy").fit_detect(ref, cur)
    assert result.drifted is True


def test_mmd_accepts_threshold_override():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=(40, 4))
    cur = rng.normal(0, 1, size=(40, 4))
    # A tiny threshold should flag noise as drift (no permutation test).
    result = get_detector("mmd", prefer_backend="numpy").fit_detect(
        ref, cur, threshold=0.0
    )
    assert result.drifted is True


# ---------------------------------------------------------------------------
# Fallback when prefer_backend is unavailable
# ---------------------------------------------------------------------------


def test_prefer_backend_falls_back_to_installed_alternative():
    """Requesting a missing backend transparently picks the next best."""
    detector = get_detector("ks", prefer_backend="nonexistent-backend")
    # scipy is always installed; that's the fallback.
    assert detector.backend == "scipy"


def test_unknown_detector_raises_keyerror():
    with pytest.raises(KeyError):
        get_detector("not-a-real-detector")


# ---------------------------------------------------------------------------
# DriftResult serialisation
# ---------------------------------------------------------------------------


def test_drift_result_serialises_cleanly():
    rng = np.random.default_rng(seed=42)
    ref = rng.normal(0, 1, size=200)
    cur = rng.normal(0, 1, size=200)
    result = get_detector("ks").fit_detect(ref, cur)
    payload = result.to_dict()
    assert set(payload).issuperset(
        {"detector", "backend", "kind", "drifted", "score", "sample_sizes"}
    )
    # All scalars must be JSON-serialisable (i.e. native Python types).
    import json
    json.dumps(payload)
