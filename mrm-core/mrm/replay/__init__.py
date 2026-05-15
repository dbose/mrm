"""1:1 Decision Replay for mrm-core (P7)

Every model invocation emits a DecisionRecord capturing the four
required components for replay: input state, model identity, inference
parameters, and raw output. Records are append-only, hash-chained per
model, and exportable in OTLP wire format.

Public surface:
    DecisionRecord            - pydantic schema for one decision
    capture                   - decorator / context manager
    LocalReplayBackend        - JSONL hash-chained store (dev)
    S3ReplayBackend           - S3 + Object Lock (production)
    OTLPExporter              - push records to an OTel collector
    reconstruct, verify       - replay re-run + diff
"""

from mrm.replay.record import DecisionRecord, ModelIdentity, InferenceParams
from mrm.replay.capture import capture, CaptureContext
from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.verify import reconstruct, verify, ReplayDiff
from mrm.replay.instrument import (
    ReplayContext,
    instrument_predictor,
    record_llm_call,
)

__all__ = [
    "DecisionRecord",
    "ModelIdentity",
    "InferenceParams",
    "capture",
    "CaptureContext",
    "LocalReplayBackend",
    "reconstruct",
    "verify",
    "ReplayDiff",
    "ReplayContext",
    "instrument_predictor",
    "record_llm_call",
]
