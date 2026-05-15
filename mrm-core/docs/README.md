# Documentation

This directory holds everything that doesn't belong at the repo root.

## Layout

| Directory | Purpose |
|---|---|
| [`guides/`](guides/) | User-facing walkthroughs (quick start, lifecycle, model references). |
| [`adr/`](adr/) | Architecture Decision Records — load-bearing design choices. |
| [`spec/`](spec/) | Normative specifications for public contracts (replay record, evidence vault, compliance plugin). |
| [`framework_guides/`](framework_guides/) | Per-jurisdiction compliance guides. |
| [`internal/`](internal/) | Historical delivery / project summaries. Kept for archaeology; not user-facing. |
| [`CROSSWALK.md`](CROSSWALK.md) | Cross-standard requirements mapping (CPS 230 ↔ SR 11-7 ↔ EU AI Act ↔ OSFI E-23 ↔ NIST AI RMF). |

## Start here

- New to `mrm-core`? Read [guides/QUICKSTART.md](guides/QUICKSTART.md).
- Designing a new feature? Read [adr/README.md](adr/README.md) and the
  existing ADRs first.
- Implementing against an `mrm-core` artefact? Read the relevant spec
  under [`spec/`](spec/).

Repo-root documents live at the top of the tree:

- [`README.md`](../README.md) — what `mrm-core` is and why.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — system overview.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — how to send changes.
- [`GOVERNANCE.md`](../GOVERNANCE.md) — who decides, how specs evolve.
- [`MAINTAINERS.md`](../MAINTAINERS.md) — current maintainers.
