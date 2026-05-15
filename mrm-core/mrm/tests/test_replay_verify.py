"""Tests for replay reconstruction and verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.capture import capture
from mrm.replay.record import DecisionRecord, ModelIdentity
from mrm.replay.verify import reconstruct, verify


def _seed_record(backend, output, inputs=None):
    record = DecisionRecord(
        model_identity=ModelIdentity(name="m", version="1"),
        input_state=inputs or {"x": 2},
        output=output,
    )
    return backend.append(record)


def test_verify_matches_when_predictor_is_deterministic(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = _seed_record(backend, output=4, inputs={"x": 2})

    diff = verify(record, predictor=lambda x: x * x)
    assert diff.matched is True
    assert diff.differences == []


def test_verify_detects_drift(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = _seed_record(backend, output=4, inputs={"x": 2})

    diff = verify(record, predictor=lambda x: x * x + 1)
    assert diff.matched is False
    assert any("!=" in d for d in diff.differences)


def test_verify_numeric_tolerance(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = _seed_record(backend, output=1.0, inputs={"x": 1.0})

    diff_tight = verify(record, predictor=lambda x: x + 1e-6, tolerance=1e-12)
    assert diff_tight.matched is False

    diff_loose = verify(record, predictor=lambda x: x + 1e-6, tolerance=1e-3)
    assert diff_loose.matched is True


def test_reconstruct_handles_positional_predictors(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = _seed_record(backend, output=5, inputs={"a": 2, "b": 3})

    def add(a, b):
        return a + b

    assert reconstruct(record, add) == 5


def test_verify_diffs_nested_structures(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = _seed_record(backend, output={"score": 0.5, "label": "A"}, inputs={"x": 1})

    def predictor(x):
        return {"score": 0.5, "label": "B"}

    diff = verify(record, predictor)
    assert diff.matched is False
    assert any("label" in d for d in diff.differences)


def test_end_to_end_capture_then_verify(tmp_path: Path):
    """Capture a real call, then verify it reproduces exactly."""
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="cube", version="1")

    @capture(backend=backend, model_identity=identity)
    def cube(x):
        return x ** 3

    cube(4)
    [recorded] = list(backend.iter_model("cube"))
    diff = verify(recorded, predictor=lambda x: x ** 3)
    assert diff.matched is True
