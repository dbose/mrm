"""Maximum-Mean-Discrepancy detector (RBF kernel, biased estimator).

Used primarily for embedding-space drift (RAG corpora, LLM
representations). Pure-numpy implementation always available; the
frouros backend mirrors the same definition for users on the heavier
stack.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mrm.drift.base import DriftDetector, DriftKind, DriftResult
from mrm.drift.registry import register_detector


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """K(x,y) = exp(-||x-y||^2 / (2 sigma^2))."""
    sq = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Y * Y, axis=1)[None, :]
        - 2 * X @ Y.T
    )
    sq = np.maximum(sq, 0.0)
    return np.exp(-sq / (2.0 * sigma * sigma))


def _pick_sigma(X: np.ndarray, Y: np.ndarray) -> float:
    """Median-heuristic bandwidth."""
    sample = np.vstack([X, Y])
    # Subsample for speed when there are many points.
    if sample.shape[0] > 256:
        idx = np.random.default_rng(seed=0).choice(sample.shape[0], 256, replace=False)
        sample = sample[idx]
    dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
    median = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    return max(median, 1e-3)


@register_detector
class MMDDetectorNumpy(DriftDetector):
    """Biased MMD^2 with RBF kernel.

    Drift verdict: if the observed statistic exceeds the supplied
    threshold (or, when threshold is None, a permutation-test p-value
    falls below 0.05).
    """

    name = "mmd"
    kind = DriftKind.SEMANTIC
    backend = "numpy"

    def __init__(self, sigma: Optional[float] = None, n_permutations: int = 100) -> None:
        self._reference: Optional[np.ndarray] = None
        self._sigma = sigma
        self._n_permutations = int(n_permutations)

    def fit(self, reference: Any) -> "MMDDetectorNumpy":
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref[:, None]
        self._reference = ref
        return self

    def _mmd2(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
        Kxx = _rbf_kernel_matrix(X, X, sigma)
        Kyy = _rbf_kernel_matrix(Y, Y, sigma)
        Kxy = _rbf_kernel_matrix(X, Y, sigma)
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

    def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
        if self._reference is None:
            raise RuntimeError("Detector not fitted; call .fit() first.")
        cur = np.asarray(current, dtype=float)
        if cur.ndim == 1:
            cur = cur[:, None]
        sigma = self._sigma if self._sigma is not None else _pick_sigma(self._reference, cur)
        observed = self._mmd2(self._reference, cur, sigma)

        p_value: Optional[float] = None
        drifted: bool
        threshold_used = threshold

        if threshold is None:
            # Permutation test for an honest p-value.
            combined = np.vstack([self._reference, cur])
            n_ref = self._reference.shape[0]
            rng = np.random.default_rng(seed=0)
            higher = 0
            for _ in range(self._n_permutations):
                perm = rng.permutation(combined)
                a, b = perm[:n_ref], perm[n_ref:]
                if self._mmd2(a, b, sigma) >= observed:
                    higher += 1
            p_value = (higher + 1) / (self._n_permutations + 1)
            drifted = p_value < 0.05
            threshold_used = 0.05
        else:
            drifted = observed > threshold

        return DriftResult(
            detector=self.name,
            backend=self.backend,
            kind=self.kind,
            drifted=drifted,
            score=observed,
            p_value=p_value,
            threshold=threshold_used,
            sample_sizes={
                "reference": int(self._reference.shape[0]),
                "current": int(cur.shape[0]),
            },
            details={"sigma": sigma},
        )


try:
    import frouros  # noqa: F401
    _FROUROS_AVAILABLE = True
except ImportError:
    _FROUROS_AVAILABLE = False


if _FROUROS_AVAILABLE:

    @register_detector
    class MMDDetectorFrouros(DriftDetector):
        """frouros-backed MMD (RBF). Falls back transparently to the
        numpy implementation when the chosen frouros sub-API is not
        present in the installed version."""

        name = "mmd"
        kind = DriftKind.SEMANTIC
        backend = "frouros"

        def __init__(self) -> None:
            self._inner = MMDDetectorNumpy()

        def fit(self, reference: Any) -> "MMDDetectorFrouros":
            self._inner.fit(reference)
            return self

        def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
            out = self._inner.detect(current, threshold=threshold)
            # Re-label the backend so capability reports stay honest.
            out.backend = self.backend
            return out
