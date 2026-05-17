# ADR-0002: Evidence Vault uses content-addressed hash chains

- **Status:** Accepted
- **Date:** 2026-04-22
- **Deciders:** core maintainers

## Context

Regulators (APRA, Fed, OSFI, EU AI Office) expect validation evidence
to be **tamper-evident, integrity-protected, immutable, and complete**
across multi-year retention windows. The 2026 Fed SR 26-2 supersedes
SR 11-7 specifically to anchor these expectations for AI models.

We had three options for the on-disk shape of an evidence packet:

1. **Plain files in a folder.** Easy. No integrity guarantee. Anyone
   with filesystem access can rewrite history.
2. **A database with an audit table.** Strong audit. But couples
   evidence to a database; banks running air-gapped VDIs need
   filesystem-shaped artefacts.
3. **Content-addressed hash-chained JSON packets.** Each packet
   contains the SHA-256 of the prior packet, making any retroactive
   edit detectable by walking the chain. Storage substrate is
   independent: filesystem, S3 Object Lock, Databricks UC, Azure
   Immutable Blob — all interchangeable.

## Decision

Evidence packets are content-addressed JSON with a `prior_packet_hash`
field that points to the immediately preceding packet's `content_hash`
for the same model. Backends are pluggable; the **canonical**
chain-of-custody is the hash chain itself, not the backend.

Storage substrates that already carry regulator assessments (SEC 17a-4
via S3 Object Lock Compliance mode) get a thin adapter. We never
reinvent WORM.

## Consequences

- **Easier:** any tamper is detectable with `mrm evidence verify`,
  regardless of backend.
- **Easier:** institutions migrate between substrates by replaying the
  chain into a new backend.
- **Easier:** the same hash-chain primitive is reused for replay
  (see [ADR-0003](0003-replay-as-first-class-primitive.md)).
- **Harder:** the schema is now a public contract. Adding fields is
  fine; reordering or renaming is a v2 spec change.
- **Trade-off accepted:** content hashes are computed from canonical
  JSON (sorted keys, no whitespace). This is portable but slower than
  binary frame hashing; acceptable at MRM scale.

## Alternatives considered

- **Database-backed audit log only** — rejected. Couples to DB
  vendor; doesn't survive air-gapped deployment.
- **Git-as-evidence-store** — tempting but rejected. Git history is
  rewritable; regulators won't accept a `git push --force`-able
  artefact as immutable.
- **Blockchain anchoring** — not rejected, deferred. The hash-chain
  *is* a Merkle-friendly structure; periodic root anchoring to an
  external timestamping service is on the roadmap (P9 — cryptographic
  hardening).

## References

- [mrm/evidence/packet.py](../../mrm/evidence/packet.py)
- [mrm/evidence/backends/s3_object_lock.py](../../mrm/evidence/backends/s3_object_lock.py)
- [docs/spec/evidence-vault-v1.md](../spec/evidence-vault-v1.md)
- SEC 17a-4(f); Cohasset Associates assessment of S3 Object Lock
