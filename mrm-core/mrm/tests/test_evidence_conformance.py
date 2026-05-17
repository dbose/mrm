"""Run the shipped evidence-vault conformance corpus from pytest.

The same corpus is also runnable from the CLI via
``mrm evidence conformance run``.
"""

from __future__ import annotations

import pytest

from mrm.evidence import _conformance


def test_conformance_vectors_all_pass():
    results = _conformance.run_all()
    assert results["total"] > 0, (
        "No conformance vectors found -- did the corpus directory "
        "get deleted?"
    )
    failures = [d for d in results["details"] if not d["passed"]]
    assert not failures, "\n".join(f"{f['name']}: {f['summary']}" for f in failures)


def test_conformance_corpus_has_positive_and_negative_vectors():
    results = _conformance.run_all()
    names = [d["name"] for d in results["details"]]
    assert any(n.startswith("p0") for n in names), "Need at least one positive vector"
    assert any(n.startswith("n0") for n in names), "Need at least one negative vector"
