# ADR-0003: Decision Replay as a first-class primitive

- **Status:** Accepted
- **Date:** 2026-05-14
- **Deciders:** core maintainers

## Context

Four distinct actors will demand the ability to reconstruct any model
decision from its constituent parts:

| Actor | Demand | Risk if missing |
|---|---|---|
| **Regulator** | Sampled decision records on demand | Consent orders, deployment restrictions |
| **Plaintiff attorney** | Adverse-decision discovery | Spoliation liability, class actions |
| **Board / audit committee** | Loss-event reconstruction | Officer accountability findings |
| **Acquirer** | AI governance documentation | Deal discounts, transaction failure |

Fed **SR 26-2** (2026, superseding SR 11-7) explicitly mandates
"tamper-evident, integrity-protected, immutable, and complete" logs
for AI activity. Brian Pelow's *AI Governance Framework* makes the
sharper claim that **systems without 1:1 replay capability represent
unmanaged regulatory and legal liability**.

We surveyed adjacent open-source projects:

- ValidMind: model validation, no decision-level replay.
- Deepchecks / Evidently: validation suites, no replay.
- MLflow: tracking, not designed for per-decision reconstruction.
- OpenMRM: governance scaffolding, no replay.

No open-source MRM tool ships replay as a first-class primitive. This
is a wedge.

## Decision

Every model invocation in `mrm-core` — traditional predictor, LLM
endpoint, or anything in between — emits a **`DecisionRecord`** that
captures the four required components:

1. **Input state** — exact inputs at decision time (features,
   prompt, retrieved context, system prompt).
2. **Model identity** — name, version, URI, artifact hash, config
   hash, checkpoint hash, provider, framework.
3. **Inference parameters** — temperature, top-p, top-k, max tokens,
   seed, retrieval-k, preprocessing pipeline hash.
4. **Output** — the raw model output **before** any downstream
   modification, formatting, or post-processing.

Records are **append-only**, **hash-chained per model**, and exportable
in **OTLP/HTTP-JSON** so banks pipe them into existing observability
without deploying a new agent (see [ADR-0004](0004-otlp-wire-format-for-replay.md)).

Capture is offered three ways:

- **`@capture` decorator** — wrap a function; every call emits a record.
- **`CaptureContext` context manager** — for explicit control.
- **Automatic via `TestRunner`** — when `replay_backend` is configured
  on the runner, every model invocation during a test run is captured
  with zero caller changes. LLM adapters get a `replay_context` slot
  set; everything else is wrapped via `instrument_predictor`.

Replay is **opt-in**. Passing `replay_backend=None` to the runner (or
not setting `replay_context` on an adapter) keeps the runtime cost at
exactly zero.

## Consequences

- **Easier:** "show me how this decision was made" is a one-CLI
  command (`mrm replay reconstruct <record-id>`).
- **Easier:** drift detection is built in (`mrm replay verify`
  re-runs the captured inputs and diffs against the recorded output
  within a numeric tolerance).
- **Easier:** SR 26-2 evidence requirements for AI activity logging
  have a concrete reference implementation, not a slide.
- **Harder:** record schema is now a public contract. Versioned via
  [docs/spec/replay-record-v1.md](../spec/replay-record-v1.md).
  Backwards-incompatible changes require a v2 spec.
- **Trade-off accepted:** capturing every inference adds storage
  cost. We bias toward "capture-by-default in tests, opt-in in
  production"; the production trade-off is a deployment decision, not
  a code decision.

## Alternatives considered

- **Capture only failing test cases** — rejected. Regulators sample
  successes too; only capturing failures looks like cherry-picking.
- **Outsource to MLflow tracking** — rejected. MLflow tracks
  experiments; it has no schema for prompt/retrieval/system-prompt and
  no hash-chain integrity guarantee.
- **Capture in the database backend, not the runner** — rejected. The
  runner is the only place that sees *every* model archetype
  uniformly; pushing capture into the backend means N×M wiring (every
  backend × every adapter).

## References

- [mrm/replay/](../../mrm/replay/)
- [mrm/replay/record.py](../../mrm/replay/record.py)
- [mrm/replay/instrument.py](../../mrm/replay/instrument.py)
- [docs/spec/replay-record-v1.md](../spec/replay-record-v1.md)
- [ADR-0002](0002-evidence-vault-hash-chain.md) — chain-of-custody primitive
- Pelow, B. *The AI Governance Framework* (2026). https://github.com/brianpelow/ai-governance-framework
- Federal Reserve SR 26-2 (2026), superseding SR 11-7 and SR 21-8
