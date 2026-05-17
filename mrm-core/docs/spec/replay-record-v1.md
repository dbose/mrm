# Decision Record Specification — v1

- **Status:** Public Review (PRD-2)
- **Version:** v1
- **Last updated:** 2026-05-15
- **Reference implementation:** [`mrm/replay/record.py`](../../mrm/replay/record.py)

## 1. Scope

This specification defines the wire format and integrity semantics of
a **DecisionRecord**: a single, replayable record of one model
invocation. Implementations of MRM tooling that wish to interoperate
with `mrm-core` SHOULD adopt this schema verbatim.

The conformance keywords (MUST, SHOULD, MAY) follow RFC 2119.

## 2. Object shape

A DecisionRecord is a JSON object with the following fields.

| Field | Type | Required | Notes |
|---|---|---|---|
| `record_id` | string (UUIDv4) | yes | Globally unique. |
| `schema_version` | string | yes | `"1.0"` for this spec. |
| `timestamp` | string (RFC 3339, UTC, trailing `Z`) | yes | When the inference occurred. |
| `model_identity` | object | yes | See §3. |
| `input_state` | object | yes | The exact inputs at decision time. Schema is producer-defined. |
| `inference_params` | object | yes | See §4. May be empty. |
| `output` | any | yes | Raw model output **before** any post-processing. |
| `prompt` | string\|null | no | The user prompt for LLM invocations. |
| `system_prompt` | string\|null | no | The system prompt for LLM invocations. |
| `retrieved_context` | array of objects \| null | no | Retrieval documents for RAG. |
| `prior_record_hash` | string (hex SHA-256) \| null | yes (may be null) | See §5. |
| `content_hash` | string (hex SHA-256) | yes | See §5. |
| `metadata` | object | yes | Producer-defined. May be empty. |

Implementations MUST NOT add unspecified top-level fields. Producer-
specific data lives under `metadata`.

## 3. `model_identity`

```json
{
  "name": "ccr_monte_carlo",
  "version": "1.4.0",
  "uri": "s3://bank-models/ccr/1.4.0/model.pkl",
  "artifact_hash": "sha256:...",
  "checkpoint_hash": null,
  "config_hash": null,
  "framework": "scikit-learn",
  "provider": null
}
```

| Field | Type | Required |
|---|---|---|
| `name` | string | yes |
| `version` | string | yes |
| `uri` | string\|null | no |
| `artifact_hash` | string\|null | no |
| `checkpoint_hash` | string\|null | no |
| `config_hash` | string\|null | no |
| `framework` | string\|null | no |
| `provider` | string\|null | no |

For LLM endpoints (no local artefact), `artifact_hash` SHOULD be the
SHA-256 of the canonical JSON of `{provider, model_name, config,
prompt_version, embedding_model}`. This produces a stable identity
for replay purposes without a binary artefact.

## 4. `inference_params`

Free-form object with these well-known keys:

| Field | Type | Applies to |
|---|---|---|
| `seed` | int\|null | Any |
| `temperature` | float\|null | LLM |
| `top_p` | float\|null | LLM |
| `top_k` | int\|null | LLM |
| `max_tokens` | int\|null | LLM |
| `retrieval_k` | int\|null | RAG |
| `preprocessing_hash` | string\|null | Tabular / vision |

Implementations MAY include additional keys; consumers MUST tolerate
unknown keys.

## 5. Hash semantics

### 5.1 `content_hash`

`content_hash` MUST equal:

```
sha256( canonical_json(record_without_content_hash) )
```

where `canonical_json` is JSON serialisation with:

- UTF-8 encoding
- Sorted object keys
- Separators `","` and `":"` (no whitespace)
- No trailing newline
- The `content_hash` field excluded from the input object

### 5.2 `prior_record_hash`

For the **first** record produced for a given `model_identity.name`,
`prior_record_hash` MUST be `null`. For every subsequent record,
`prior_record_hash` MUST equal the `content_hash` of the immediately
preceding record for the same `model_identity.name`.

### 5.3 Verification

A record is **valid** iff:

1. `content_hash` equals the recomputed hash per §5.1.
2. Either `prior_record_hash` is `null` AND the record is the
   genesis record, OR `prior_record_hash` equals the `content_hash`
   of the preceding record.

A chain is **valid** iff every record in the chain is valid.

## 6. Transport

Records SHOULD be exported over **OTLP/HTTP-JSON** as `LogRecord`
entries, with:

- `body.stringValue` = the canonical JSON of the record
- Attributes hoisted: `mrm.model.name`, `mrm.model.version`,
  `mrm.replay.record_id`, `mrm.replay.content_hash`,
  `mrm.replay.prior_record_hash`, `mrm.replay.schema_version`
- `service.name = "mrm-core"`, `service.namespace = "mrm.replay"`

See [`mrm/replay/otlp.py`](../../mrm/replay/otlp.py).

## 7. Storage backends

Storage substrate is implementation-defined; the canonical
chain-of-custody is the hash chain, not the backend. Backends that
provide regulator-accepted WORM (e.g. S3 Object Lock COMPLIANCE mode)
SHOULD be preferred for production.

## 8. Conformance

A conformance test vector suite is published under
[`test-vectors/replay/`](test-vectors/replay/). Implementations
claiming conformance MUST pass every positive vector and reject every
negative vector.

## 9. Changelog

| Version | Date | Change |
|---|---|---|
| v1 (PRD-2) | 2026-05-15 | Initial public review draft. |
