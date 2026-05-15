# Architecture Decision Records (ADRs)

This directory holds the load-bearing design decisions behind
`mrm-core`. Each ADR is a small, dated, immutable document that
captures:

- **Context** — what forced the decision
- **Decision** — what we chose
- **Consequences** — what we accept by choosing it
- **Alternatives** — what we rejected and why

ADRs are append-only. We don't edit history. If a decision is
superseded, we write a new ADR that references and overrides the old
one, and update the old one's *Status* line to `superseded by ADR-NNNN`.

This directory deliberately mirrors the layout used by other
regulator-facing reference implementations
(e.g. [SR-26.2-MRM](https://github.com/mmpworks/SR-26.2-Model-Risk-Management),
[Pelow AI Governance Framework](https://github.com/brianpelow/ai-governance-framework))
so that auditors can navigate this repository without learning a new
convention.

## Index

| ID | Title | Status |
|---|---|---|
| [0001](0001-pluggable-compliance-standards.md) | Pluggable compliance standards via decorator registry | Accepted |
| [0002](0002-evidence-vault-hash-chain.md) | Evidence Vault uses content-addressed hash chains | Accepted |
| [0003](0003-replay-as-first-class-primitive.md) | Decision Replay as a first-class primitive | Accepted |
| [0004](0004-otlp-wire-format-for-replay.md) | OTLP/HTTP-JSON wire format for replay export | Accepted |
| [0005](0005-oss-vs-cloud-split.md) | OSS/Cloud feature split mirrors dbt-core / dbt Cloud | Accepted |

## Writing a new ADR

1. Copy [`template.md`](template.md).
2. Number it sequentially (`0006-...`).
3. Open a pull request. The Status starts as `Proposed`.
4. Once merged, change Status to `Accepted` in a follow-up commit (or
   in the same PR if the discussion happened in an issue).
