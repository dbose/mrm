# ADR-0005: OSS / Cloud feature split mirrors dbt-core / dbt Cloud

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** core maintainers

## Context

The product is split across two distributions:

- **`mrm-core`** — free, Apache 2.0, the CLI and library.
- **`<brand>` Cloud** — commercial, hosted, the monetisation path.

The split has to be defensible to three audiences simultaneously:

1. **Open-source contributors** — must believe core is genuinely
   useful on its own, not crippleware.
2. **Risk officers and validators** — must be able to run core in an
   air-gapped VDI without ever talking to a hosted service.
3. **Commercial buyers** — must see a clear set of capabilities only
   available in Cloud, justifying the contract.

dbt Labs has run this split successfully for years. We deliberately
adopt the same shape rather than inventing a new one.

## Decision

Anything that runs on a single developer's machine, in an air-gapped
VDI, or in a CI pipeline stays in `mrm-core`. The Cloud product is
**operations over the OSS CLI**, never a fork.

| `mrm-core` (free, Apache 2.0) | `<brand>` Cloud (commercial) |
|---|---|
| CLI, DAG, tests, compliance plugins | Hosted scheduled run orchestration |
| All bundled jurisdictions | Web UI for risk officers / validators / auditors |
| All worked examples (CCR, XVA, IRB, RAG) | RBAC, SSO, workspace audit log |
| Local + S3 Object Lock evidence backends | Evidence vault as managed service |
| GPG/age root signing | HSM-backed root signing (FIPS 140-2 L3+) |
| `mrm replay record / verify / sample` | Regulator-portal export, 7-year retention SLA |
| Markdown reports | Workflow engine (validator approves → owner re-runs) |
| OSS test packs | Hosted GRC integrations (OpenPages, ServiceNow, Workiva) |
| Open governance, ADRs, specs | Premium support, SLAs, custom standard authoring |
| Community contributions | Customer-managed keys (BYOK), VPC deploy |

## Consequences

- **Easier:** every paid feature has an OSS anchor it is built on
  (e.g. Cloud HSM signing extends OSS GPG/age signing in
  [ADR-0002](0002-evidence-vault-hash-chain.md); Cloud regulator-portal
  export uses the OSS `mrm replay sample` command in
  [ADR-0003](0003-replay-as-first-class-primitive.md)). The OSS primitive
  is always the contract.
- **Easier:** acquisition narrative is clean. An acquirer sees the OSS
  surface and knows what they would inherit; the Cloud surface is
  recurring revenue with a clear delineation.
- **Harder:** must resist the urge to move OSS features into Cloud
  for monetisation reasons. The contract is: capabilities that work
  on a single machine stay free. Anything multi-user / multi-tenant /
  managed-key / hosted is paid.

## Alternatives considered

- **All-free, services-revenue model** — rejected. The SaaS thesis
  requires recurring revenue tied to a product, not consulting time.
- **Open-core with paid "advanced" tests** — rejected. Test breadth
  must be a community gravity well; locking tests behind a paywall
  kills contributor energy.
- **Source-available license** — rejected. Banks reject licences they
  can't put through legal review in 24 hours. Apache 2.0 is the
  shortest path through procurement.

## References

- [STRATEGY.md](../../../STRATEGY.md) — full backlog with each
  feature's OSS / Cloud tier.
- dbt Labs commercial strategy (publicly documented in S-1 filings).
