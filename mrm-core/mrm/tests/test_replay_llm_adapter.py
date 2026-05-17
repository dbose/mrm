"""Tests for LLM adapter replay capture.

We exercise the base-class ``LLMEndpoint`` capture path via a tiny
concrete subclass whose ``generate`` returns canned responses. This
avoids depending on real provider SDKs while proving the contract:

  * prompt and system_prompt are captured
  * retrieved_context is captured when a retriever is configured
  * temperature / top_p / max_tokens land on inference_params
  * the LLM metadata (tokens, latency, model) lands on the record
  * replay_context=None is a true no-op
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mrm.backends.llm_endpoints import LLMEndpoint
from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.instrument import ReplayContext
from mrm.replay.record import ModelIdentity


class _FakeRetriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        return list(self._docs)


class _FakeLLM(LLMEndpoint):
    """Concrete LLMEndpoint that returns a canned response."""

    def __init__(self, config: Dict[str, Any], canned_response: str = "ok"):
        super().__init__(config)
        self._canned = canned_response
        self.calls: List[Tuple[str, Optional[str], Dict[str, Any]]] = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append((prompt, system_prompt, dict(kwargs)))
        return self._canned, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 12.5,
            "model": "fake-llm",
            "finish_reason": "stop",
        }

    def health_check(self):  # required by ABC
        return True


def _ctx(tmp_path: Path, model_name: str = "llm") -> ReplayContext:
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    identity = ModelIdentity(name=model_name, version="1", provider="fake")
    return ReplayContext(backend=backend, model_identity=identity)


def test_complete_emits_record_with_prompt_and_params(tmp_path: Path):
    config = {"model_name": "fake-llm", "temperature": 0.3, "top_p": 0.9, "max_tokens": 256}
    llm = _FakeLLM(config, canned_response="hello world")
    llm.replay_context = _ctx(tmp_path, "chat1")

    response, metadata = llm.complete("Hi there", system_prompt="You are helpful.")

    assert response == "hello world"
    records = list(llm.replay_context.backend.iter_model("chat1"))
    assert len(records) == 1
    r = records[0]
    assert r.prompt == "Hi there"
    assert r.system_prompt == "You are helpful."
    assert r.output == "hello world"
    assert r.inference_params.temperature == 0.3
    assert r.inference_params.top_p == 0.9
    assert r.inference_params.max_tokens == 256
    assert r.metadata["llm"]["total_tokens"] == 15


def test_complete_captures_retrieval_context(tmp_path: Path):
    docs = [
        {"id": "a", "text": "Doc A content"},
        {"id": "b", "text": "Doc B content"},
    ]
    config = {"model_name": "fake-llm"}
    llm = _FakeLLM(config, canned_response="answer")
    llm.retriever = _FakeRetriever(docs)
    llm.replay_context = _ctx(tmp_path, "rag")

    llm.complete("What does Doc A say?")

    [r] = list(llm.replay_context.backend.iter_model("rag"))
    assert r.retrieved_context == docs
    # retrieval_k must reflect the number of retrieved docs.
    assert r.inference_params.retrieval_k == 2
    # The prompt stored on the record is the *user* query, not the
    # augmented one — that's the regulator-relevant artefact.
    assert r.prompt == "What does Doc A say?"


def test_complete_is_noop_without_replay_context(tmp_path: Path):
    config = {"model_name": "fake-llm"}
    llm = _FakeLLM(config, canned_response="x")
    llm.replay_context = None  # default; making it explicit.

    response, _ = llm.complete("q")
    assert response == "x"
    # No backend was even constructed; verify no spurious files appeared.
    assert list(tmp_path.glob("**/*.jsonl")) == []


def test_chain_links_across_consecutive_calls(tmp_path: Path):
    config = {"model_name": "fake-llm"}
    llm = _FakeLLM(config, canned_response="ok")
    llm.replay_context = _ctx(tmp_path, "chain")

    for i in range(3):
        llm.complete(f"q{i}")

    records = list(llm.replay_context.backend.iter_model("chain"))
    assert records[0].prior_record_hash is None
    assert records[1].prior_record_hash == records[0].content_hash
    assert records[2].prior_record_hash == records[1].content_hash


def test_kwargs_override_defaults_in_inference_params(tmp_path: Path):
    config = {"model_name": "fake-llm", "temperature": 0.0, "top_p": 0.5, "max_tokens": 32}
    llm = _FakeLLM(config, canned_response="ok")
    llm.replay_context = _ctx(tmp_path, "params")

    llm.complete("q", temperature=0.7, max_tokens=128, seed=42)

    [r] = list(llm.replay_context.backend.iter_model("params"))
    assert r.inference_params.temperature == 0.7
    assert r.inference_params.max_tokens == 128
    assert r.inference_params.seed == 42
