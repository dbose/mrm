"""Kolmogorov-Smirnov data-drift detector.

Two backends:

  * ``scipy``   -- ``scipy.stats.ks_2samp``. Always available.
  * ``frouros`` -- ``frouros.detectors.data_drift.KSTest``. Optional.

Both compute the two-sample KS statistic and a p-value. The result
shape is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from mrm.drift.base import DriftDetector, DriftKind, DriftResult
from mrm.drift.registry import register_detector


@register_detector
class KSDetectorScipy(DriftDetector):
    """KS two-sample test backed by scipy."""

    name = "ks"
    kind = DriftKind.DATA
    backend = "scipy"

    def __init__(self) -> None:
        self._reference: Optional[np.ndarray] = None

    def fit(self, reference: Any) -> "KSDetectorScipy":
        self._reference = np.asarray(reference, dtype=float).ravel()
        return self

    def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
        if self._reference is None:
            raise RuntimeError("Detector not fitted; call .fit(reference) first.")
        cur = np.asarray(current, dtype=float).ravel()
        from scipy.stats import ks_2samp
        result = ks_2samp(self._reference, cur)
        stat = float(result.statistic)
        pval = float(result.pvalue)
        # Default to p<0.05 if the caller didn't pass an explicit cutoff.
        threshold_used = threshold if threshold is not None else 0.05
        drifted = pval < threshold_used
        return DriftResult(
            detector=self.name,
            backend=self.backend,
            kind=self.kind,
            drifted=drifted,
            score=stat,
            p_value=pval,
            threshold=threshold_used,
            sample_sizes={"reference": int(self._reference.size), "current": int(cur.size)},
        )


# ---------------------------------------------------------------------------
# Optional frouros-backed alternative
# ---------------------------------------------------------------------------

try:
    import frouros  # noqa: F401
    _FROUROS_AVAILABLE = True
except ImportError:
    _FROUROS_AVAILABLE = False


if _FROUROS_AVAILABLE:

    @register_detector
    class KSDetectorFrouros(DriftDetector):
        """KS test backed by frouros (matches scipy mathematically but
        plugs into frouros' streaming + alerting hooks)."""

        name = "ks"
        kind = DriftKind.DATA
        backend = "frouros"

        def __init__(self) -> None:
            self._reference: Optional[np.ndarray] = None
            self._detector = None

        def fit(self, reference: Any) -> "KSDetectorFrouros":
            from frouros.detectors.data_drift import KSTest

            self._reference = np.asarray(reference, dtype=float).ravel()
            self._detector = KSTest()
            self._detector.fit(self._reference)
            return self

        def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
            if self._detector is None:
                raise RuntimeError("Detector not fitted; call .fit() first.")
            cur = np.asarray(current, dtype=float).ravel()
            stat, pval = self._detector.compare(cur)
            threshold_used = threshold if threshold is not None else 0.05
            drifted = pval < threshold_used
            return DriftResult(
                detector=self.name,
                backend=self.backend,
                kind=self.kind,
                drifted=drifted,
                score=float(stat),
                p_value=float(pval),
                threshold=threshold_used,
                sample_sizes={"reference": int(self._reference.size), "current": int(cur.size)},
            )
