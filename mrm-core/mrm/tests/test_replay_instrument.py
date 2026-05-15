"""Tests for the universal predictor instrumentation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.instrument import (
    ReplayContext,
    instrument_predictor,
    record_llm_call,
)
from mrm.replay.record import InferenceParams, ModelIdentity


def _ctx(tmp_path: Path, model_name: str = "m") -> ReplayContext:
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name=model_name, version="1")
    return ReplayContext(backend=backend, model_identity=identity)


# ---------------------------------------------------------------------------
# instrument_predictor (universal — sklearn-shaped, callable-shaped, etc.)
# ---------------------------------------------------------------------------


def test_instrument_predict_emits_record(tmp_path: Path):
    ctx = _ctx(tmp_path, "sk_model")

    class FakeSklearn:
        def predict(self, X):
            return [x * 2 for x in X]

    wrapped = instrument_predictor(FakeSklearn(), ctx)
    assert wrapped.predict([1, 2, 3]) == [2, 4, 6]

    records = list(ctx.backend.iter_model("sk_model"))
    assert len(records) == 1
    assert records[0].input_state == {"features": [1, 2, 3]}
    assert records[0].output == [2, 4, 6]
    assert records[0].model_identity.name == "sk_model"


def test_instrument_callable_emits_record(tmp_path: Path):
    ctx = _ctx(tmp_path, "fn_model")

    def fn(x, y=2):
        return x + y

    wrapped = instrument_predictor(fn, ctx)
    assert wrapped(3, y=4) == 7

    records = list(ctx.backend.iter_model("fn_model"))
    assert len(records) == 1
    assert records[0].input_state == {"features": 3, "y": 4}
    assert records[0].output == 7


def test_instrument_chains_records_per_invocation(tmp_path: Path):
    ctx = _ctx(tmp_path, "chain")

    class Adder:
        def predict(self, x):
            return x + 1

    wrapped = instrument_predictor(Adder(), ctx)
    for i in range(3):
        wrapped.predict(i)

    records = list(ctx.backend.iter_model("chain"))
    assert records[0].prior_record_hash is None
    assert records[1].prior_record_hash == records[0].content_hash
    assert records[2].prior_record_hash == records[1].content_hash


def test_instrument_passes_through_other_attributes(tmp_path: Path):
    ctx = _ctx(tmp_path)

    class HasExtras:
        params = {"alpha": 0.5}

        def predict(self, X):
            return X

        def score(self, X, y):
            return 0.99

    wrapped = instrument_predictor(HasExtras(), ctx)
    assert wrapped.params == {"alpha": 0.5}
    assert wrapped.score([1], [1]) == 0.99


def test_instrument_with_none_context_is_noop(tmp_path: Path):
    """Passing replay_context=None must return the original object."""

    class M:
        def predict(self, x):
            return x

    m = M()
    assert instrument_predictor(m, None) is m


# ---------------------------------------------------------------------------
# record_llm_call (LLM-shaped capture)
# ---------------------------------------------------------------------------


def test_record_llm_call_captures_prompt_and_retrieval(tmp_path: Path):
    ctx = _ctx(tmp_path, "rag_chat")
    retrieved = [
        {"id": "doc1", "text": "Capital of France is Paris."},
        {"id": "doc2", "text": "France is in Europe."},
    ]
    record = record_llm_call(
        replay_context=ctx,
        prompt="Where is Paris?",
        response="Paris is the capital of France.",
        metadata={"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        system_prompt="You are a helpful assistant.",
        retrieved_docs=retrieved,
        inference_params=InferenceParams(temperature=0.0, top_p=1.0, max_tokens=100),
    )

    assert record is not None
    assert record.prompt == "Where is Paris?"
    assert record.system_prompt == "You are a helpful assistant."
    assert record.retrieved_context == retrieved
    assert record.output == "Paris is the capital of France."
    assert record.inference_params.temperature == 0.0
    assert record.inference_params.max_tokens == 100
    assert record.metadata["llm"]["total_tokens"] == 19


def test_record_llm_call_returns_none_when_replay_off(tmp_path: Path):
    assert (
        record_llm_call(
            replay_context=None,
            prompt="hi",
            response="hello",
        )
        is None
    )


def test_record_llm_call_chains_across_invocations(tmp_path: Path):
    ctx = _ctx(tmp_path, "chat")
    r1 = record_llm_call(ctx, prompt="q1", response="a1")
    r2 = record_llm_call(ctx, prompt="q2", response="a2")
    r3 = record_llm_call(ctx, prompt="q3", response="a3")
    assert r2.prior_record_hash == r1.content_hash
    assert r3.prior_record_hash == r2.content_hash
