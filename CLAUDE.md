# CLAUDE.md

Operational context for Claude Code sessions on this repo. Read this before
making non-trivial changes. For strategic context (positioning, roadmap,
acquirer landscape) see `STRATEGY.md` in the repo root.

---

## What this project is

`mrm` is an open-source CLI for Model Risk Management and AI Governance,
positioned as **"dbt for Model Risk."** It brings software-engineering
discipline (version control, tests, lineage, docs-as-code) to MRM workflows
that today live in GUI-driven tools (OpenPages, RiskSpan, ServiceNow GRC).

Primary buyer persona: **MRM teams at Tier 1/2 banks and regulated financial
institutions.** Secondary: AI governance functions at any regulated AI shop.

---

## Architecture at a glance

```
mrm-core/
├── mrm/
│   ├── cli/                  Typer-based CLI entry point
│   ├── core/                 Project loading, DAG, catalog, triggers
│   │   └── catalog_backends/ Databricks UC + MLflow integration
│   ├── compliance/           Pluggable regulatory standards framework
│   │   └── builtin/cps230.py APRA CPS 230 (currently the only bundled standard)
│   ├── tests/                Pluggable test framework
│   │   └── builtin/          Tabular + CCR + governance tests
│   ├── engine/runner.py      Test runner with model loading
│   └── backends/             Storage backends (local, MLflow)
├── ccr_example/              Worked CCR Monte Carlo example (canonical demo)
└── credit_risk_example/      Credit risk worked example
```

The two extension points that matter most:

- **`@register_test`** decorator in `mrm/tests/library.py` — adds a new
  validation test
- **`@register_standard`** decorator in `mrm/compliance/registry.py` — adds
  a new regulatory framework

Standards are discovered three ways: bundled (in `compliance/builtin/`),
external pip package (entry point `mrm.compliance`), or local custom
(via `compliance_paths` in `mrm_project.yml`).

---

## Conventions

### YAML over code where possible

Model definitions, project config, compliance mappings all live in YAML.
Keep the bar high for moving things from YAML to code — the dbt analogy
relies on declarative configs being readable by risk officers, not just
Python developers.

### dbt parallels are intentional

When designing a feature, ask "how would dbt do this?" first. Examples:
`ref()` for model references, graph operators (`+model`, `model+`,
`+model+`), `mrm test --select tier:tier_1`, `mrm docs generate`. If a
proposed feature breaks the dbt mental model without good reason, push
back.

### Compliance evidence is append-only

Reports are regenerable; **evidence is not.** When implementing the
evidence vault (see `STRATEGY.md` priority 2), treat evidence packets as
immutable, hashed, optionally signed artefacts. Never overwrite. Never
silently mutate.

### Test naming

Tests are namespaced: `ccr.MCConvergence`, `tabular.MissingValues`,
`compliance.GovernanceCheck`. Keep namespaces tight and domain-aligned.
New regulatory test packs go under their domain prefix
(e.g. `xva.CCRSensitivity`, `credit.PDBacktest`).

---

## How to run things

```bash
# Install (editable)
cd mrm-core && pip install -e .

# Run the canonical CCR example end-to-end
cd ccr_example
python setup_ccr_example.py    # generate synthetic data + pickle model
python run_validation.py       # run 8 tests, evaluate triggers, emit report

# Via the CLI
mrm docs generate ccr_monte_carlo --compliance standard:cps230
mrm docs list-standards
mrm triggers check ccr_monte_carlo

# Publish to Databricks UC (requires credentials)
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
mrm publish ccr_monte_carlo
```

The CCR example is the regression suite for the framework as a whole.
**If a change breaks `python run_validation.py`, the change is broken.**

---

## What "done" looks like for a feature

A feature is done when:

1. It works in the CCR example (or a new worked example exists demonstrating it)
2. Its behaviour is testable from the CLI
3. YAML config is documented in the relevant example's `*.yml`
4. If it touches compliance: at least one paragraph mapping shows it generates
   regulator-shaped evidence
5. The README's CLI Commands section is updated if a new command was added

---

## What NOT to do

- Don't add a dependency to satisfy a one-off. The install footprint matters
  — risk teams run this in locked-down environments.
- Don't bake Databricks-specific assumptions into core. UC is one backend
  among several (MLflow OSS, local, future SageMaker / Vertex). Keep the
  abstraction at `catalog_backends/` clean.
- Don't fork the dbt vocabulary unnecessarily. If dbt calls it `ref()`, we
  call it `ref()`.
- Don't write code that only the author can validate. MRM teams are the
  audience — readability beats cleverness.
- Don't add LLM-based "AI features" to the core CLI without a clear
  governance story. The whole point of the tool is to govern AI; baking
  ungoverned AI into the tool is a credibility hit.

---

## Current priorities

See `STRATEGY.md` for the full 90-day plan. In short, the next features
that matter most:

1. **SR 11-7 (US Fed) and EU AI Act Annex IV** as bundled compliance
   standards alongside CPS 230. Without a second and third jurisdiction,
   this reads as a personal project rather than a platform.
2. **Evidence vault** — append-only, hashed, optionally signed evidence
   packets. Currently reports are regenerable markdown; evidence needs to
   be immutable to satisfy regulator workflows.
3. **XVA worked example** alongside CCR. Broadens the tool's surface from
   "one quant model type" to "platform for quant model risk."

Lower priority but on the list: agent / compound-system versioning
(prompts + retrievers + guardrails as a single releasable unit), more
catalog backends (SageMaker Model Registry, Vertex AI), and a small
mrm-cloud companion for hosted scheduled runs.

---

## Style

- Python 3.9+, type hints throughout, pydantic for config models
- `rich` for CLI output (already a dependency)
- Tests: pytest, kept fast; the CCR example is the integration test
- Docstrings on public APIs; internal helpers can stay terse

---

## When in doubt

Lean toward the dbt analogy and toward what a senior model validator at
APRA, the Fed, or the ECB would find credible. The audience is regulators
and the people who answer to them, not the AI/ML hype cycle.
