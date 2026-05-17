"""Tests for the capture decorator and context manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.capture import CaptureContext, capture
from mrm.replay.record import InferenceParams, ModelIdentity


def test_capture_decorator_records_each_call(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="square", version="1")

    @capture(backend=backend, model_identity=identity)
    def square(x):
        return x * x

    assert square(3) == 9
    assert square(4) == 16

    records = list(backend.iter_model("square"))
    assert len(records) == 2
    assert records[0].input_state == {"x": 3}
    assert records[0].output == 9
    assert records[1].input_state == {"x": 4}
    assert records[1].output == 16


def test_capture_decorator_passes_through_return_value(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="m", version="1")

    @capture(backend=backend, model_identity=identity)
    def f(a, b=5):
        return a + b

    assert f(2) == 7
    assert f(2, b=10) == 12


def test_capture_decorator_chains_records(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="m", version="1")

    @capture(backend=backend, model_identity=identity)
    def f(x):
        return x

    f(1); f(2); f(3)

    records = list(backend.iter_model("m"))
    assert records[0].prior_record_hash is None
    assert records[1].prior_record_hash == records[0].content_hash
    assert records[2].prior_record_hash == records[1].content_hash


def test_capture_context_manager_records_explicitly(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="m", version="1")
    params = InferenceParams(seed=42, temperature=0.0)

    with CaptureContext(backend, model_identity=identity, inference_params=params) as ctx:
        record = ctx.record(input_state={"q": "hello"}, output="world")

    assert record.input_state == {"q": "hello"}
    assert record.output == "world"
    assert record.inference_params.seed == 42
    assert backend.tail("m").record_id == record.record_id


def test_capture_context_propagates_metadata(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="m", version="1")
    with CaptureContext(
        backend, model_identity=identity, metadata={"trigger": "scheduled"}
    ) as ctx:
        record = ctx.record(
            input_state={"x": 1},
            output=2,
            extra_metadata={"run_id": "abc"},
        )
    assert record.metadata["trigger"] == "scheduled"
    assert record.metadata["run_id"] == "abc"


def test_capture_handles_non_jsonable_inputs(tmp_path: Path):
    """numpy/pandas-like objects should be coerced, not crash."""
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name="m", version="1")

    class HasToList:
        def tolist(self):
            return [1, 2, 3]

    @capture(backend=backend, model_identity=identity)
    def f(arr):
        return sum(arr.tolist())

    out = f(HasToList())
    assert out == 6
    records = list(backend.iter_model("m"))
    assert records[0].input_state == {"arr": [1, 2, 3]}
