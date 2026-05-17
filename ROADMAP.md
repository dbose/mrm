# Roadmap

`mrm-core` is being developed openly. The roadmap below is non-binding
— priorities shift as the regulatory landscape evolves and as the
community contributes.

## Shipped

- CLI with dbt-style ergonomics + DAG + pluggable test framework
- 5 bundled compliance standards: APRA CPS 230 · Fed SR 11-7 ·
  Fed SR 26-2 · EU AI Act Annex IV · OSFI E-23
- Cross-standard crosswalk (27 concepts × 5 standards) including the
  SR 11-7 → SR 26-2 transition map
- Validation triggers (6 types: scheduled / drift / breach /
  materiality / regulatory / manual)
- Evidence vault — hash-chained packets + local + S3 Object Lock
  backends; SEC 17a-4-shape immutability
- Cryptographic vault hardening — HMAC-chained event log, RFC-6962
  Merkle daily root, pluggable signer (Local / GPG / age / AWS KMS),
  conformance test-vector suite
- 1:1 Decision Replay — DecisionRecord schema, capture decorator,
  OTLP exporter, replay verification
- Replay capture for every model archetype — sklearn, HF, MLflow,
  LiteLLM and the legacy LLM adapters
- GenAI test pack (14 tests across 7 categories) + LiteLLM unified
  endpoint adapter
- Drift detection — pluggable detector framework (KS, Wasserstein,
  Page-Hinkley, MMD) with scipy/numpy fallbacks; optional frouros
  backend via `pip install 'mrm-core[drift]'`
- dbt-style project/profile config split + unified resolver
- Specs PRD-2: Decision Record (v1), Evidence Vault Chain (v1),
  Compliance Plugin Contract (v1)
- Architecture Decision Records (6) + GOVERNANCE.md + MAINTAINERS.md
- Test coverage: 158 pytest + 59 end-to-end acceptance checks

## In progress

- LLM adversarial red-team pack — 50+ attack templates across
  fiduciary-bypass / PII-extraction / jailbreak / system-prompt
  override / regulatory-claim fabrication; financial-F1 entity-weighted
  accuracy
- ADR + spec governance posture extensions for community contributions

## Planned

- GRC platform connectors (OpenPages, ServiceNow IRM, Workiva)
- Quant worked examples: XVA via ORE, IRB credit risk (PD/LGD/EAD)
- Crosswalk auto-update from authoritative regulator sources

## Contributing

PRs welcome — see [CONTRIBUTING.md](mrm-core/CONTRIBUTING.md) and
[GOVERNANCE.md](mrm-core/GOVERNANCE.md). Architecture-significant
changes need an ADR under
[docs/adr/](mrm-core/docs/adr/).
