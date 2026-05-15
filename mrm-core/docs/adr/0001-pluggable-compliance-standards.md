# ADR-0001: Pluggable compliance standards via decorator registry

- **Status:** Accepted
- **Date:** 2026-04-12
- **Deciders:** core maintainers

## Context

Model Risk Management regulations differ by jurisdiction (APRA CPS 230,
Fed SR 11-7 / SR 26-2, EU AI Act Annex IV, OSFI E-23, NIST AI RMF,
ECB Guide to Internal Models, MAS, …). A regulated institution
typically maps a single model against two or three of these
simultaneously.

The naïve options were:

1. **Hard-code one standard** — fastest to ship, but every new
   jurisdiction is a fork.
2. **A single YAML schema** — flexible, but the schema becomes the
   union of every standard's quirks, and report generation has nowhere
   sensible to live.
3. **A plugin model** — every standard owns its paragraph mappings,
   governance checks, and report generator behind a stable contract.

Incumbents (OpenPages, ServiceNow IRM) lock buyers into the
jurisdictions their consultants happen to know. That is a wedge for an
open, plugin-driven tool.

## Decision

Compliance standards are plugins, discovered via three tiers:

1. **Bundled** — `mrm/compliance/builtin/<standard>.py` ships with
   `mrm-core`.
2. **External pip packages** — discovered via the
   `mrm.compliance` Python entry-point group.
3. **Local custom** — paths declared in
   `mrm_project.yml > compliance_paths` are scanned at startup.

Each plugin registers via:

```python
@register_standard("sr117")
class FedSR117(ComplianceStandard):
    ...
```

The contract requires paragraph mappings, test mappings, governance
checks, and a report generator. The registry is the single source of
truth for `mrm docs list-standards`.

## Consequences

- **Easier:** adding a jurisdiction is a single file under
  `mrm/compliance/builtin/`. Four standards bundled today
  (CPS 230, SR 11-7, EU AI Act, OSFI E-23) all use the same shape.
- **Easier:** community can ship a standard as a pip package without
  forking the core repo.
- **Harder:** the contract is now part of the public API. Changing
  the `ComplianceStandard` ABC is an SDK-version breaking change.
- **Trade-off accepted:** some standards (e.g. ECB IRB) want
  numerical evidence; the contract has a generic `extra_artifacts`
  escape hatch so we don't extend the ABC every quarter.

## Alternatives considered

- **Single YAML schema across all standards** — rejected. Report
  generation needs Jinja2 templates and per-standard logic that YAML
  cannot express without becoming a programming language.
- **A standards-as-data approach with a giant JSON** — rejected.
  Couples every release to every regulation update; provides no
  governance hook for institutions that want to ship private
  standards.

## References

- [mrm/compliance/registry.py](../../mrm/compliance/registry.py)
- [mrm/compliance/builtin/cps230.py](../../mrm/compliance/builtin/cps230.py)
- [mrm/compliance/builtin/sr117.py](../../mrm/compliance/builtin/sr117.py)
