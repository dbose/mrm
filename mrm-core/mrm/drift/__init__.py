"""Drift detection -- pluggable detectors with scipy fallbacks and an
opt-in frouros backend (``pip install 'mrm-core[drift]'``).

Public surface:

  * ``DriftDetector``      -- ABC every detector implements.
  * ``DriftResult``        -- structured result schema.
  * ``DriftKind``          -- DATA | CONCEPT | SEMANTIC.
  * ``register_detector``  -- decorator for plugin authors.
  * ``get_detector``       -- factory used by tests + CLI.
  * ``available_backends`` -- capability report for ``mrm doctor``.
  * ``list_detectors``     -- full registry dump.

The package import is side-effect-free aside from registering the
builtin detectors. Heavy imports (``scipy.stats``, ``frouros``) happen
inside ``detect()`` so a misconfigured env doesn't crash ``mrm test``
on unrelated commands.
"""

from mrm.drift.base import DriftDetector, DriftKind, DriftResult
from mrm.drift.registry import (
    available_backends,
    get_detector,
    list_detectors,
    register_detector,
)

# Import the builtin detectors so they self-register.
from mrm.drift import builtin  # noqa: F401

__all__ = [
    "DriftDetector",
    "DriftKind",
    "DriftResult",
    "available_backends",
    "get_detector",
    "list_detectors",
    "register_detector",
]
