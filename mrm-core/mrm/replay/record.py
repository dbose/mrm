"""DecisionRecord schema — the four-component replay primitive.

A DecisionRecord captures everything needed to reconstruct a single
model invocation:

  1. input_state       exact inputs at decision time
  2. model_identity    URI, version, checkpoint hash, config hash
  3. inference_params  temperature, top-p, seed, retrieval-k, etc.
  4. output            raw output BEFORE downstream post-processing

Records are hash-chained per model (each record references the prior
record's content_hash). Serialisation is canonical JSON so the hash is
deterministic across processes and platforms.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class ModelIdentity(BaseModel):
    """Component 2 — model identity.

    Same model + same identity hash means same weights. For LLM
    endpoints the artifact_hash is a hash of (provider, model_name,
    config) since there is no local artifact.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str
    version: str
    uri: Optional[str] = None
    artifact_hash: Optional[str] = None
    checkpoint_hash: Optional[str] = None
    config_hash: Optional[str] = None
    framework: Optional[str] = None
    provider: Optional[str] = None


class InferenceParams(BaseModel):
    """Component 3 — inference-time parameters.

    Tabular models: preprocessing pipeline hash, seed.
    LLM endpoints: temperature, top_p, max_tokens, seed, retrieval_k.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    retrieval_k: Optional[int] = None
    preprocessing_hash: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    """A single replayable model decision.

    Hash semantics:
        content_hash       = sha256(canonical_json(record - {content_hash}))
        prior_record_hash  = the content_hash of the immediately
                             preceding record for the SAME model
                             (None for the first record).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_utc_now_iso)
    model_identity: ModelIdentity
    input_state: Dict[str, Any]
    inference_params: InferenceParams = Field(default_factory=InferenceParams)
    output: Any
    retrieved_context: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    prior_record_hash: Optional[str] = None
    content_hash: Optional[str] = None
    schema_version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", self.compute_content_hash())

    def compute_content_hash(self) -> str:
        payload = self.model_dump(exclude={"content_hash"})
        return _sha256_hex(_canonical_json(payload))

    def verify_hash(self) -> bool:
        return self.content_hash == self.compute_content_hash()

    def verify_chain(self, prior: Optional["DecisionRecord"]) -> bool:
        if prior is None:
            return self.prior_record_hash is None
        return self.prior_record_hash == prior.content_hash

    def to_json(self, indent: Optional[int] = None) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "DecisionRecord":
        return cls.model_validate_json(payload)

    @staticmethod
    def hash_inputs(inputs: Any) -> str:
        """Stable hash of any JSON-serialisable input payload."""
        return _sha256_hex(_canonical_json({"inputs": inputs}))
