"""Capture decorator + context manager for recording model decisions.

Usage as a decorator:

    @capture(backend=backend, model_identity=ModelIdentity(...))
    def predict(features):
        return model.predict(features)

Usage as a context manager:

    with CaptureContext(backend, model_identity=mid) as ctx:
        out = model.predict(features)
        ctx.record(input_state={"features": features}, output=out)
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, Optional

from mrm.replay.backends.base import ReplayBackend
from mrm.replay.record import DecisionRecord, InferenceParams, ModelIdentity


def _coerce_inputs(args: tuple, kwargs: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    """Best-effort serialisation of call args into a JSON-able dict."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: _to_jsonable(v) for k, v in bound.arguments.items()}
    except (TypeError, ValueError):
        return {
            "args": [_to_jsonable(a) for a in args],
            "kwargs": {k: _to_jsonable(v) for k, v in kwargs.items()},
        }


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    # numpy / pandas — best effort without hard imports.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _to_jsonable(to_list())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(to_dict())
        except Exception:
            pass
    return repr(value)


class CaptureContext:
    """Context manager that emits a DecisionRecord on ``record(...)``."""

    def __init__(
        self,
        backend: ReplayBackend,
        model_identity: ModelIdentity,
        inference_params: Optional[InferenceParams] = None,
        metadata: Optional[Dict[str, Any]] = None,
        otlp_exporter: Optional[Any] = None,
    ) -> None:
        self.backend = backend
        self.model_identity = model_identity
        self.inference_params = inference_params or InferenceParams()
        self.metadata = dict(metadata or {})
        self.otlp_exporter = otlp_exporter
        self.last_record: Optional[DecisionRecord] = None

    def __enter__(self) -> "CaptureContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def record(
        self,
        input_state: Dict[str, Any],
        output: Any,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retrieved_context: Optional[list] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DecisionRecord:
        merged_metadata = dict(self.metadata)
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        record = DecisionRecord(
            model_identity=self.model_identity,
            input_state=_to_jsonable(input_state),
            inference_params=self.inference_params,
            output=_to_jsonable(output),
            prompt=prompt,
            system_prompt=system_prompt,
            retrieved_context=retrieved_context,
            metadata=merged_metadata,
        )
        record = self.backend.append(record)
        self.last_record = record
        if self.otlp_exporter is not None:
            try:
                self.otlp_exporter.export(record)
            except Exception:
                pass
        return record


def capture(
    backend: ReplayBackend,
    model_identity: ModelIdentity,
    inference_params: Optional[InferenceParams] = None,
    metadata: Optional[Dict[str, Any]] = None,
    otlp_exporter: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: every call of the wrapped function emits a record.

    The function's bound arguments form ``input_state``; the return
    value becomes ``output``. The original return value is passed
    through unchanged so the decorator is non-invasive.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            inputs = _coerce_inputs(args, kwargs, func)
            output = func(*args, **kwargs)
            ctx = CaptureContext(
                backend=backend,
                model_identity=model_identity,
                inference_params=inference_params,
                metadata=metadata,
                otlp_exporter=otlp_exporter,
            )
            ctx.record(input_state=inputs, output=output)
            return output

        return wrapper

    return decorator
