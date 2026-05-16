"""Drift detector registry.

The registry is small on purpose. It does three things:

  1. ``register_detector`` -- decorator used by builtin + plugin
     detector classes.
  2. ``get_detector(name, prefer_backend=...)`` -- pick the best
     available implementation for the given logical detector.
  3. ``available_backends()`` -- capability report consumed by
     ``mrm doctor``.

Backends are *implementations*, not detectors. A single logical
detector like ``ks`` can have a scipy implementation (always
available) and a frouros implementation (only when the optional
dependency is installed). The registry treats them as alternatives
and prefers the configured backend when it can.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Type

from mrm.drift.base import DriftDetector

logger = logging.getLogger(__name__)


_REGISTRY: Dict[Tuple[str, str], Type[DriftDetector]] = {}
#: Set of backend names that are *installed* (verified at import-time
#: by each backend module).
_INSTALLED_BACKENDS: set = {"numpy", "scipy"}  # scipy is a hard core dep.


def register_detector(cls: Type[DriftDetector]) -> Type[DriftDetector]:
    """Register a drift detector class.

    Keyed by ``(name, backend)`` so multiple implementations of the
    same detector coexist; ``get_detector`` picks one at runtime.
    """
    if not cls.name:
        raise ValueError(f"Detector {cls.__name__} must set .name")
    _REGISTRY[(cls.name, cls.backend)] = cls
    return cls


def mark_backend_available(backend: str) -> None:
    """Backends call this from their module when their dependency is
    importable. ``mrm doctor`` reads the resulting set."""
    _INSTALLED_BACKENDS.add(backend)


def available_backends() -> List[str]:
    """Return the list of currently installed backends."""
    return sorted(_INSTALLED_BACKENDS)


def get_detector(
    name: str,
    prefer_backend: Optional[str] = None,
) -> DriftDetector:
    """Return an instance of the best available detector for ``name``.

    Order:
      1. ``prefer_backend`` if installed and that detector ships there.
      2. ``frouros`` (highest fidelity) if installed.
      3. ``scipy`` (universal fallback).
      4. ``numpy`` (last resort, pure-Python).

    Raises ``KeyError`` when no implementation is registered.
    """
    candidates: List[str] = []
    if prefer_backend:
        candidates.append(prefer_backend)
    candidates.extend(b for b in ["frouros", "scipy", "numpy"] if b not in candidates)

    for backend in candidates:
        if backend not in _INSTALLED_BACKENDS:
            continue
        cls = _REGISTRY.get((name, backend))
        if cls is not None:
            return cls()

    raise KeyError(
        f"No drift detector registered for name='{name}'. "
        f"Installed backends: {sorted(_INSTALLED_BACKENDS)}; "
        f"available detectors: {sorted({n for n, _ in _REGISTRY})}."
    )


def list_detectors() -> List[Dict[str, str]]:
    """Capability report consumed by ``mrm doctor`` and the README."""
    return [
        {"name": name, "backend": backend, "kind": cls.kind.value, "installed":
            backend in _INSTALLED_BACKENDS}
        for (name, backend), cls in sorted(_REGISTRY.items())
    ]


# ---------------------------------------------------------------------------
# Eager backend probes
# ---------------------------------------------------------------------------

def _probe_frouros() -> None:
    try:
        import frouros  # noqa: F401
        mark_backend_available("frouros")
    except ImportError:
        # Silent -- this is expected on most installs.
        pass


_probe_frouros()
