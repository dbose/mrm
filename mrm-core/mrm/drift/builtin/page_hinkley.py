"""Page-Hinkley concept-drift detector.

The Page-Hinkley test is the classic streaming statistic for detecting
a change in the mean of a sequence. NIST AI RMF, ECB Internal Models
guidance, and the original SR 11-7 commentary all reference it.

Implemented in pure numpy so it works in air-gapped installs. A
frouros-backed alternative is registered when the optional dependency
is available.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mrm.drift.base import DriftDetector, DriftKind, DriftResult
from mrm.drift.registry import register_detector


@register_detector
class PageHinkleyDetectorNumpy(DriftDetector):
    """Page-Hinkley test in pure numpy.

    Detects a change in the mean of a stream relative to the
    cumulative mean. Two hyperparameters:

      * ``delta`` -- magnitude of allowable variation (default 0.005).
      * ``lambda`` -- threshold on the cumulative sum (default 50).

    The detector treats ``reference`` as the burn-in window used to
    compute the baseline mean. ``current`` is the streamed batch under
    test.
    """

    name = "page_hinkley"
    kind = DriftKind.CONCEPT
    backend = "numpy"

    def __init__(self, delta: float = 0.005, lambda_threshold: float = 50.0) -> None:
        self._reference_mean: Optional[float] = None
        self._delta = float(delta)
        self._lambda = float(lambda_threshold)

    def fit(self, reference: Any) -> "PageHinkleyDetectorNumpy":
        ref = np.asarray(reference, dtype=float).ravel()
        if ref.size == 0:
            raise ValueError("Reference window is empty.")
        self._reference_mean = float(ref.mean())
        return self

    def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
        if self._reference_mean is None:
            raise RuntimeError("Detector not fitted; call .fit() first.")
        cur = np.asarray(current, dtype=float).ravel()
        if cur.size == 0:
            raise ValueError("Current window is empty.")

        # Streaming cumulative deviation from reference mean.
        # m_t = sum_{i<=t} (x_i - mean_ref - delta)
        # PH(t) = max( m_t - min_{s<=t} m_s ).
        deviations = cur - self._reference_mean - self._delta
        cumulative = np.cumsum(deviations)
        running_min = np.minimum.accumulate(cumulative)
        ph_values = cumulative - running_min
        ph_max = float(ph_values.max())

        threshold_used = threshold if threshold is not None else self._lambda
        drifted = ph_max > threshold_used
        return DriftResult(
            detector=self.name,
            backend=self.backend,
            kind=self.kind,
            drifted=drifted,
            score=ph_max,
            threshold=threshold_used,
            sample_sizes={"reference_mean_window_size": 1, "current": int(cur.size)},
            details={
                "reference_mean": self._reference_mean,
                "delta": self._delta,
            },
        )


try:
    import frouros  # noqa: F401
    _FROUROS_AVAILABLE = True
except ImportError:
    _FROUROS_AVAILABLE = False


if _FROUROS_AVAILABLE:

    @register_detector
    class PageHinkleyDetectorFrouros(DriftDetector):
        name = "page_hinkley"
        kind = DriftKind.CONCEPT
        backend = "frouros"

        def __init__(self, delta: float = 0.005, lambda_threshold: float = 50.0) -> None:
            self._delta = float(delta)
            self._lambda = float(lambda_threshold)
            self._detector = None
            self._reference_mean: Optional[float] = None

        def fit(self, reference: Any) -> "PageHinkleyDetectorFrouros":
            from frouros.detectors.concept_drift import PageHinkley
            from frouros.detectors.concept_drift.streaming.change_detection.page_hinkley import (
                PageHinkleyConfig,
            )

            ref = np.asarray(reference, dtype=float).ravel()
            self._reference_mean = float(ref.mean())
            self._detector = PageHinkley(
                config=PageHinkleyConfig(min_num_instances=max(1, ref.size // 4))
            )
            # Burn in the detector against the reference window so its
            # internal state is comparable to the numpy implementation.
            for value in ref:
                _ = self._detector.update(value=float(value))
            return self

        def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
            if self._detector is None:
                raise RuntimeError("Detector not fitted; call .fit() first.")
            cur = np.asarray(current, dtype=float).ravel()
            triggered = False
            max_stat = 0.0
            for value in cur:
                response = self._detector.update(value=float(value))
                stat = float(getattr(response, "value", 0.0) or 0.0)
                if stat > max_stat:
                    max_stat = stat
                if getattr(self._detector, "drift", False):
                    triggered = True

            threshold_used = threshold if threshold is not None else self._lambda
            return DriftResult(
                detector=self.name,
                backend=self.backend,
                kind=self.kind,
                drifted=triggered or max_stat > threshold_used,
                score=max_stat,
                threshold=threshold_used,
                sample_sizes={"current": int(cur.size)},
                details={
                    "reference_mean": self._reference_mean,
                    "delta": self._delta,
                },
            )
