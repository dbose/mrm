# Governance

This document describes how `mrm-core` is governed: who decides what,
how specifications evolve, and how the project intends to transition
to a neutral foundation once it has reached sufficient adoption.

## Project structure

`mrm-core` is currently a maintainer-led open-source project under the
Apache License 2.0. There is one tier of authority:

- **Maintainers.** Approve PRs, cut releases, accept or reject spec
  changes, and steward the public roadmap (see
  [STRATEGY.md](../STRATEGY.md)). The current maintainer set is
  recorded in [MAINTAINERS.md](MAINTAINERS.md).

As external contributors arrive, additional roles (committers,
reviewers, working-group leads) will be introduced via a follow-up
governance amendment proposed as a PR against this document.

## Specifications

The normative contracts of `mrm-core` are versioned in
[`docs/spec/`](docs/spec/) and follow a Public Review Draft (PRD)
lifecycle:

1. **PRD-1 — Draft.** Maintainer-authored. Public for comment.
2. **PRD-2 — Public Review.** At least two independent implementers
   have acknowledged building against it. Open issue tracker for
   gap-class findings.
3. **Final.** PRD-2 closed with zero open gap-class findings.
4. **Superseded.** A successor PRD has reached Final.

Spec-affecting changes require:

- A pull request that updates the spec document AND the reference
  implementation AND the conformance test vectors.
- Public comment via GitHub issues for at least 14 days.
- Approval from at least one maintainer who did not author the change.

Breaking changes to a Final spec require a new major version
(`v1` → `v2`). The prior version remains discoverable and supported
for at least one release cycle.

## Code contributions

See [CONTRIBUTING.md](CONTRIBUTING.md). In summary:

- Open an issue for non-trivial changes before writing code.
- All PRs require green CI and at least one maintainer approval.
- Architecture-significant changes require an
  [ADR](docs/adr/README.md).
- Tests are mandatory for new features.

## Architecture Decision Records

Load-bearing design decisions are captured as ADRs in
[`docs/adr/`](docs/adr/). ADRs are append-only. Superseding an ADR
requires writing a new one that references the prior decision and
explains the reversal.

## Security

Vulnerability reports are private. See
[SECURITY.md](SECURITY.md) for the disclosure channel. The project
follows a 90-day responsible-disclosure window by default.

## Conduct

Contributors are expected to follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
Code of Conduct.

## Intent to transition to a neutral foundation

`mrm-core` is designed from the outset to be relicensable to a neutral
open-source foundation once the project reaches sufficient adoption.
Candidate foundations include:

- **OpenSSF** (best fit for the cryptographic evidence-vault work)
- **CNCF** (best fit for the OTLP-native replay work)
- **FINOS** (best fit for the financial-services positioning)
- **Linux Foundation AI & Data** (alternative for AI-governance focus)

The transition would include:

- Trademark transfer
- Domain transfer
- A neutral Technical Steering Committee
- Foundation-standard CLA / DCO

The maintainer set commits to evaluating this transition no later than
the point at which there are five named institutional users, an active
contributor graph of more than three independent organisations, or a
Final v1 of any spec — whichever comes first.

## Amending this document

Amendments are proposed as pull requests. A 14-day public comment
window applies. Approval requires two maintainer signoffs.

---

*Last updated: 2026-05-15.*
