"""Drift detection -- abstract base class + result schema.

Drift is the single most-cited monitoring expectation across MRM
regulators (CPS 230, SR 11-7 §II.C, SR 26-2 §II.AI.D, EU AI Act
post-market monitoring). This module is the pluggable foundation:

* ``DriftDetector``  -- ABC every detector implements.
* ``DriftResult``    -- structured output every detector returns.
* ``DriftKind``      -- enumeration distinguishing data drift from
                        concept drift from semantic drift.

Implementations live under ``mrm/drift/builtin/``. Each implementation
declares a backend (``"scipy"`` or ``"frouros"``) so callers can opt
in to the heavier dependency without breaking the air-gapped install.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DriftKind(str, Enum):
    DATA = "data"        # input feature distribution
    CONCEPT = "concept"  # input -> output relationship
    SEMANTIC = "semantic"  # embedding-space drift (text / RAG)


@dataclass
class DriftResult:
    """Structured drift result returned by every detector.

    The shape is deliberately small so the CLI can pretty-print it
    without per-detector special cases.
    """

    detector: str                # detector name, e.g. "ks", "page_hinkley"
    backend: str                 # "scipy" | "frouros" | "numpy"
    kind: DriftKind
    drifted: bool                # final pass/fail
    score: float                 # detector-native score (KS stat, MMD^2, etc.)
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "backend": self.backend,
            "kind": self.kind.value,
            "drifted": self.drifted,
            "score": float(self.score),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "sample_sizes": dict(self.sample_sizes),
            "details": dict(self.details),
        }


class DriftDetector(abc.ABC):
    """Abstract base class for drift detectors.

    The contract:

      * ``fit(reference)``  -- snapshot the reference distribution or
        streaming state.
      * ``detect(current, threshold) -> DriftResult``  -- compare and
        return a verdict.

    Subclasses MUST declare ``name`` and ``kind`` class attributes.
    Optionally they declare ``backend`` (defaults to ``"numpy"``); the
    registry uses this to surface capability information in
    ``mrm doctor``.
    """

    name: str = ""
    kind: DriftKind = DriftKind.DATA
    backend: str = "numpy"

    @abc.abstractmethod
    def fit(self, reference: Any) -> "DriftDetector":
        """Snapshot the reference data. Returns self for chaining."""

    @abc.abstractmethod
    def detect(self, current: Any, threshold: Optional[float] = None) -> DriftResult:
        """Compare ``current`` against the fitted reference."""

    def fit_detect(
        self, reference: Any, current: Any, threshold: Optional[float] = None
    ) -> DriftResult:
        """Convenience: fit + detect in one call."""
        return self.fit(reference).detect(current, threshold=threshold)
