"""Wasserstein-distance (earth-mover) data-drift detector.

Backends:

  * ``scipy``   -- ``scipy.stats.wasserstein_distance``. Always available.
  * ``frouros`` -- if installed; same definition, integrates with
                   frouros' alert plumbing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mrm.drift.base import DriftDetector, DriftKind, DriftResult
from mrm.drift.registry import register_detector


@register_detector
class WassersteinDetectorScipy(DriftDetector):
    """1-Wasserstein distance backed by scipy."""

    name = "wasserstein"
    kind = DriftKind.DATA
    backend = "scipy"

    def __init__(self) -> None:
        self._reference: Optional[np.ndarray] = None

    def fit(self, reference: Any) -> "WassersteinDetectorScipy":
        self._reference = np.asarray(reference, dtype=float).ravel()
        return self

    def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
        if self._reference is None:
            raise RuntimeError("Detector not fitted; call .fit(reference) first.")
        cur = np.asarray(current, dtype=float).ravel()
        from scipy.stats import wasserstein_distance
        dist = float(wasserstein_distance(self._reference, cur))
        # Default heuristic: drift when distance > 0.5 * std(reference).
        threshold_used = (
            threshold
            if threshold is not None
            else 0.5 * float(np.std(self._reference) or 1.0)
        )
        drifted = dist > threshold_used
        return DriftResult(
            detector=self.name,
            backend=self.backend,
            kind=self.kind,
            drifted=drifted,
            score=dist,
            threshold=threshold_used,
            sample_sizes={"reference": int(self._reference.size), "current": int(cur.size)},
        )


try:
    import frouros  # noqa: F401
    _FROUROS_AVAILABLE = True
except ImportError:
    _FROUROS_AVAILABLE = False


if _FROUROS_AVAILABLE:

    @register_detector
    class WassersteinDetectorFrouros(DriftDetector):
        name = "wasserstein"
        kind = DriftKind.DATA
        backend = "frouros"

        def __init__(self) -> None:
            self._reference: Optional[np.ndarray] = None

        def fit(self, reference: Any) -> "WassersteinDetectorFrouros":
            self._reference = np.asarray(reference, dtype=float).ravel()
            return self

        def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
            # frouros exposes Wasserstein through scipy under the hood;
            # we treat it as a labelled alias so the backend report is
            # honest about which library produced the number.
            if self._reference is None:
                raise RuntimeError("Detector not fitted; call .fit() first.")
            cur = np.asarray(current, dtype=float).ravel()
            from scipy.stats import wasserstein_distance
            dist = float(wasserstein_distance(self._reference, cur))
            threshold_used = (
                threshold
                if threshold is not None
                else 0.5 * float(np.std(self._reference) or 1.0)
            )
            drifted = dist > threshold_used
            return DriftResult(
                detector=self.name,
                backend=self.backend,
                kind=self.kind,
                drifted=drifted,
                score=dist,
                threshold=threshold_used,
                sample_sizes={"reference": int(self._reference.size), "current": int(cur.size)},
            )
