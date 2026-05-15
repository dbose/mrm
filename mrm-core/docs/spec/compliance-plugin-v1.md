# Compliance Plugin Contract — v1

- **Status:** Public Review (PRD-2)
- **Version:** v1
- **Last updated:** 2026-05-15
- **Reference implementation:** [`mrm/compliance/registry.py`](../../mrm/compliance/registry.py)

## 1. Scope

This specification defines the contract a Python module must satisfy
to register itself as a compliance standard with `mrm-core`.

Conformance keywords follow RFC 2119.

## 2. Registration

A plugin MUST register exactly one subclass of `ComplianceStandard`
via the `@register_standard("<id>")` decorator. The `<id>` is the
public identifier used by CLI commands (`mrm docs generate ...
--compliance standard:<id>`).

```python
from mrm.compliance.registry import register_standard, ComplianceStandard

@register_standard("sr26_2")
class FedSR26_2(ComplianceStandard):
    ...
```

## 3. Required attributes

The class MUST expose:

| Attribute | Type | Purpose |
|---|---|---|
| `name` | str | Human-readable name (e.g. "Fed SR 26-2"). |
| `version` | str | Standard's own version (e.g. "2026"). |
| `jurisdiction` | str | Two-letter ISO + descriptor ("US/Federal Reserve"). |
| `paragraphs` | dict | Paragraph ID → description. |
| `test_mappings` | dict | Test name → list of paragraph IDs. |
| `governance_checks` | list[callable] | Functions returning (passed, reason). |

## 4. Required methods

```python
def generate_report(
    self,
    model_results: dict,
    evidence: list[EvidencePacket] | None = None,
    output_format: str = "markdown",
) -> str: ...
```

Returns a regulator-shaped report keyed against this standard's
paragraphs.

```python
def evidence_artifacts(
    self,
    model_results: dict,
) -> dict[str, list[str]]:
```

Returns a `{paragraph_id: [artifact_path, ...]}` mapping. The result
is consumed by the evidence vault when freezing packets.

## 5. Discovery

`mrm-core` discovers plugins in three tiers:

1. **Bundled.** Anything in `mrm/compliance/builtin/`.
2. **External pip package.** Declared under the `mrm.compliance`
   Python entry-point group.
3. **Local project plugins.** Paths listed under
   `mrm_project.yml > compliance_paths`.

A plugin MUST be usable from all three tiers without code change.

## 6. Crosswalks

A plugin MAY contribute mappings to the cross-standard crosswalk file
at `mrm/compliance/crosswalks/standards.yaml`. Cross-mappings live in
the crosswalk file, not in the plugin, so each pair of standards has
exactly one canonical mapping.

## 7. Conformance

The reference implementation bundles four conformant plugins (CPS 230,
SR 11-7, EU AI Act, OSFI E-23). Test vectors under
[`test-vectors/compliance/`](test-vectors/compliance/) include a
minimal example plugin that exercises every required attribute and
method.

## 8. Changelog

| Version | Date | Change |
|---|---|---|
| v1 (PRD-2) | 2026-05-15 | Initial public review draft. |
