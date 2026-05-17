"""Built-in drift detectors -- imported eagerly so the registry is
populated by the time application code asks for a detector."""

from mrm.drift.builtin import ks, wasserstein, page_hinkley, mmd  # noqa: F401
