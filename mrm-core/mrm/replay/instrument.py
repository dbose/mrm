"""Universal predictor instrumentation.

This module provides one helper per model archetype so that every model
invocation in mrm-core emits a DecisionRecord without callers having
to touch their own code paths.

  * ``instrument_predictor``   wraps any object exposing ``.predict``
                               or ``__call__`` (sklearn, HF wrapper,
                               pickled model, MLflow pyfunc).
  * ``record_llm_call``        emits an LLM-shaped record (prompt,
                               system_prompt, retrieved_context,
                               decoding params, token usage).

Both rely on ``CaptureContext`` so the hash-chain semantics from
``replay/record.py`` are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from mrm.replay.backends.base import ReplayBackend
from mrm.replay.capture import CaptureContext, _to_jsonable
from mrm.replay.record import DecisionRecord, InferenceParams, ModelIdentity


@dataclass
class ReplayContext:
    """Bundle of (backend, model_identity, optional OTLP exporter).

    Carried through the runner/adapters so any code path that wants to
    auto-capture a record only needs this one object.
    """

    backend: ReplayBackend
    model_identity: ModelIdentity
    otlp_exporter: Optional[Any] = None
    default_metadata: Optional[Dict[str, Any]] = None


def instrument_predictor(predictor: Any, replay_context: ReplayContext) -> Any:
    """Wrap a predictor so each `.predict(...)` / `__call__(...)`
    invocation appends a DecisionRecord.

    The original object is returned untouched if ``replay_context`` is
    None — callers can therefore opt-out by passing None and the
    overhead is exactly zero.
    """
    if replay_context is None:
        return predictor

    return _InstrumentedPredictor(predictor, replay_context)


class _InstrumentedPredictor:
    """Transparent proxy that captures every predict/__call__."""

    def __init__(self, inner: Any, replay_context: ReplayContext) -> None:
        # Use object.__setattr__ to avoid recursion through __setattr__.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_replay_context", replay_context)

    # Capture helper -------------------------------------------------------

    def _emit(self, input_state: Dict[str, Any], output: Any) -> None:
        ctx = CaptureContext(
            backend=self._replay_context.backend,
            model_identity=self._replay_context.model_identity,
            otlp_exporter=self._replay_context.otlp_exporter,
            metadata=self._replay_context.default_metadata or {},
        )
        ctx.record(input_state=input_state, output=_to_jsonable(output))

    # Predictor surface ----------------------------------------------------

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        output = self._inner.predict(*args, **kwargs)
        self._emit(_inputs_dict(args, kwargs), output)
        return output

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._inner(*args, **kwargs)
        self._emit(_inputs_dict(args, kwargs), output)
        return output

    # Pass-through for everything else (model.fit, .score, .params, etc.) --

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __repr__(self) -> str:
        return f"<InstrumentedPredictor wrapping {self._inner!r}>"


def _inputs_dict(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce args/kwargs into a stable input_state mapping."""
    payload: Dict[str, Any] = {}
    if args:
        if len(args) == 1:
            payload["features"] = _to_jsonable(args[0])
        else:
            payload["args"] = [_to_jsonable(a) for a in args]
    if kwargs:
        payload.update({k: _to_jsonable(v) for k, v in kwargs.items()})
    return payload


def record_llm_call(
    replay_context: ReplayContext,
    prompt: str,
    response: str,
    metadata: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    inference_params: Optional[InferenceParams] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[DecisionRecord]:
    """Emit an LLM-shaped DecisionRecord.

    Returns the persisted record, or None if ``replay_context`` is
    None (i.e. replay is opted-out).
    """
    if replay_context is None:
        return None

    merged_metadata: Dict[str, Any] = {}
    if metadata:
        merged_metadata["llm"] = _to_jsonable(metadata)
    if extra_metadata:
        merged_metadata.update(extra_metadata)
    if replay_context.default_metadata:
        for k, v in replay_context.default_metadata.items():
            merged_metadata.setdefault(k, v)

    ctx = CaptureContext(
        backend=replay_context.backend,
        model_identity=replay_context.model_identity,
        inference_params=inference_params or InferenceParams(),
        otlp_exporter=replay_context.otlp_exporter,
        metadata=merged_metadata,
    )
    return ctx.record(
        input_state={"prompt": prompt},
        output=response,
        prompt=prompt,
        system_prompt=system_prompt,
        retrieved_context=retrieved_docs,
    )
