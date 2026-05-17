# ADR-0004: OTLP/HTTP-JSON wire format for replay export

- **Status:** Accepted
- **Date:** 2026-05-14
- **Deciders:** core maintainers

## Context

`DecisionRecord` instances need to leave the producing process and
arrive somewhere durable, queryable, and ideally already integrated
with the bank's existing observability stack. The choices for the
wire format were:

1. **Custom JSON shipped over the project's own collector.** Forces
   banks to deploy yet another agent inside a tightly-controlled VDI.
   Lowest probability of getting through change control.
2. **A new gRPC protocol.** Strong typing but adds a hard dependency
   on `protobuf` + a code generator. Banks audit transitive deps.
3. **OTLP/HTTP-JSON** — the OpenTelemetry Protocol over HTTP, JSON
   encoding. Already terminating in every modern observability stack
   (Datadog, Splunk, Grafana, Elastic, Honeycomb, AWS X-Ray, vendor-
   neutral collectors). No new agent required. JSON is human-auditable.

## Decision

`mrm-core` exports `DecisionRecord` as **OTLP/HTTP-JSON log records**:

- One record → one `LogRecord` in an `OTLP/HTTP-JSON` envelope.
- The full canonical JSON of the record is placed in
  `body.stringValue` (the entire artefact is preserved verbatim).
- High-cardinality fields a SIEM would filter on are hoisted into
  `attributes`: `mrm.model.name`, `mrm.model.version`,
  `mrm.model.provider`, `mrm.replay.record_id`,
  `mrm.replay.content_hash`, `mrm.replay.prior_record_hash`,
  `mrm.replay.schema_version`.
- `service.name = "mrm-core"`, `service.namespace = "mrm.replay"`.

We deliberately **do not** depend on `opentelemetry-*` SDKs. The
payload shape is stable, well-documented, and small enough to emit
from `urllib` with no third-party dependency. Banks frequently audit
transitive deps; the minimum-deps stance preserves install-footprint
goodwill.

## Consequences

- **Easier:** any OTLP-aware collector ingests records on day one.
- **Easier:** SIEMs filter, alert, and retain decision records using
  the queries they already write for application logs.
- **Easier:** export errors are non-fatal — replay capture must never
  break the host application if the collector is down.
- **Harder:** if OTLP changes the JSON shape across a major spec
  revision, our `record_to_otlp_log` function changes too. Mitigation:
  the shape is fixed at the v0.20.0 spec and exercised by tests.

## Alternatives considered

- **Pull in `opentelemetry-sdk`** — rejected for transitive-deps
  reasons. Worth revisiting if a paying enterprise customer demands
  it.
- **CloudEvents** — viable, but less universally accepted in
  bank observability stacks than OTLP.
- **Custom Kafka schema** — rejected. Couples replay to a transport
  choice the bank may not be running.

## References

- [mrm/replay/otlp.py](../../mrm/replay/otlp.py)
- OpenTelemetry Protocol Specification (OTLP/HTTP-JSON encoding)
- [ADR-0003](0003-replay-as-first-class-primitive.md)
