# STRATEGY.md

Strategic context for **<brand>** (the product family) and `mrm-core`
(the open-source CLI inside it). This document is the source of truth
for the roadmap. `CLAUDE.md` references it, and Claude Code sessions
can be driven by asking *"read STRATEGY.md and build the next
feature."*

> **Brand placeholder.** Every occurrence of `<brand>` in this document
> is intentional. Once a name is confirmed (WHOIS, PyPI, GitHub org,
> trademark all clear), do a single find-and-replace across this file
> and `CLAUDE.md`. Until then, `<brand>` is the product-family name and
> `mrm-core` is the open-source CLI package name.

Last updated: May 2026

---

## How to use this document with Claude Code

The **Feature Backlog** section below is ordered by priority. Each
feature has:

- A `STATUS` field (`done`, `in-progress`, `next`, `backlog`)
- A `WEDGE` field (what gap this closes versus competitors)
- A clear definition of done

To pick up work in a Claude Code session, say:

> Read STRATEGY.md. Find the next feature with STATUS: next. Build it.
> When done, mark STATUS: done and commit.

Or to pick a specific item:

> Read STRATEGY.md. Build feature P3 (XVA worked example via ORE).

Only edit STATUS fields and the "What's done" log when work completes.
Don't restructure the document for tactical changes.

---

## Brand and product structure

The product family is **<brand>**. The OSS CLI is published as
`mrm-core` and lives under the `<brand>/` GitHub org once the rebrand
ships.

Two tiers under the <brand> umbrella, deliberately mirroring the
dbt-core / dbt Cloud split:

| `mrm-core` (free, Apache 2.0) | <brand> Cloud (commercial) |
|---|---|
| CLI, DAG, tests, compliance plugins | Hosted multi-user platform |
| All bundled jurisdictions | Scheduled run orchestration |
| All worked examples (CCR, XVA, IRB) | Evidence vault as a managed service (S3 Object Lock you don't have to set up) |
| Local evidence backend | RBAC, SSO, audit log of who ran what |
| Markdown reports | Web UI for risk officers / validators / auditors |
| Single-user use | Workflow engine: validator approves, owner re-runs, etc. |
| Open governance | Hosted GRC integrations (push to OpenPages, ServiceNow with credentials managed) |
| Community contributions | Premium support, SLAs, custom standard authoring |

**Repository plan.** `mrm-core` stays at the existing path
(`dbose/mrm`) while it ramps; the rebrand to `<brand>/mrm-core` happens
once SR 11-7, EU AI Act, and OSFI E-23 plugins ship and the README can
credibly describe a multi-jurisdiction platform.

**Domain.** Secure `<brand>.io` (and `.com` if available) before any
further public posting.

---

## One-line positioning

**<brand> is dbt for Model Risk Management.**

A declarative, version-controlled, plugin-extensible toolchain that
brings software-engineering discipline to a domain that today runs on
Excel, SharePoint, and GUI-driven GRC suites. The OSS CLI (`mrm-core`)
runs anywhere; <brand> Cloud is the hosted layer for risk officers and
audit teams who need a UI, workflow, and managed evidence storage.

---

## Why this exists

Three forces converging in 2026:

1. **Regulatory tailwinds.** APRA CPS 230 fully effective July 2025,
   with APRA's 2025-26 supervisory program targeting significant
   financial institutions. APRA published an industry-wide AI letter
   and CPS 230 amendments on 30 April 2026. OSFI E-23 effective May
   2027 (expanded to all FRFIs including insurers). EU AI Act fully
   applicable August 2026, high-risk obligations extended to August
   2027. Fed SR 11-7 being actively reinterpreted to cover GenAI.
   Colorado SB205 and similar US state laws creating evidence-
   collection burden.

2. **GenAI in regulated workflows.** Banks, insurers, and super funds
   are deploying LLMs into customer-facing and decisioning workflows.
   SR 11-7 wasn't written for stochastic models. Incumbents (OpenPages,
   RiskSpan, ServiceNow GRC) are slow to adapt.

3. **dbt has trained the buyers.** Tier 1/2 banks now have data teams
   fluent in declarative-config, version-controlled, test-driven
   workflows. Cultural readiness for "dbt for X" is high.

---

## Primary competitor: ValidMind

ValidMind is the closest direct competitor and has already adopted the
OSS-library + commercial-cloud split. <brand>'s wedge sits in the gaps
they leave open.

### Where ValidMind is ahead

- **GenAI / LLM testing depth** — closed by P6 below
- **Documentation automation** — partial parity via P6
- **GRC integration breadth** — closed by P7 below
- **Sales motion and capital** — Series A backed, Experian partnership
- **Test library scale** — broader catalogue of built-in tests

### Where <brand> has structural wedge (don't lose these)

- **dbt-shaped workflow** — YAML + CLI + Git, not Jupyter notebooks
- **Pluggable compliance standards** — `@register_standard` three-tier
  discovery; ValidMind hard-codes jurisdictions
- **Local-first / air-gapped friendly** — `mrm-core` runs in locked-
  down VDIs; ValidMind always pushes to their cloud
- **DAG and dependency management** — `ref()`, graph operators,
  `--select`, topological sort
- **Validation triggers as first-class** — scheduled, drift, breach,
  materiality, regulatory, manual
- **Deep quant model worked examples** — CCR Monte Carlo today; XVA
  via ORE and IRB credit risk planned

---

## Integration philosophy

<brand> integrates with existing enterprise infrastructure rather
than replacing it. Three layers to keep distinct:

- **MLOps stack** (MLflow, Unity Catalog, SageMaker Model Registry,
  Vertex AI, DVC, W&B) — <brand> reads from these.
- **Immutable storage** (S3 Object Lock, Azure Immutable Blob,
  Databricks UC) — <brand> writes evidence to these.
- **GRC platforms** (IBM OpenPages, ServiceNow IRM, Archer,
  MetricStream, Workiva) — <brand> pushes results into these.

---

## What's done

- ✅ CLI with dbt-style ergonomics (`ref()`, graph operators,
  `--select`, `docs generate`)
- ✅ DAG management with topological sort and parallel execution
- ✅ Pluggable test framework (`@register_test`) with 30+ built-ins
- ✅ Pluggable compliance framework (`@register_standard`) with
  three-tier discovery (bundled / pip / local)
- ✅ Validation trigger engine (scheduled, drift, breach, materiality,
  regulatory, manual)
- ✅ Databricks Unity Catalog + MLflow integration with auto signature
  inference
- ✅ Worked CCR Monte Carlo example end-to-end
- ✅ APRA CPS 230 bundled standard with paragraph-level evidence mapping
- ✅ Federal Reserve SR 11-7 bundled standard with section-level evidence mapping
- ✅ EU AI Act Annex IV bundled standard with 9 technical documentation requirements
- ✅ OSFI E-23 bundled standard for Canadian FRFIs with 7-section framework
- ✅ Federal Reserve SR 26-2 bundled standard (supersedes SR 11-7/SR 21-8) with AI-activity-logging clauses anchored to P7 DecisionRecord and tamper-evident clauses anchored to P5 EvidencePacket
- ✅ Cross-standard crosswalk mapping 27 concepts across 5 standards (AU/US-SR11-7/US-SR26-2/EU/CA) including explicit SR 11-7 → SR 26-2 transition map
- ✅ Evidence vault with hash-chained packets, local + S3 Object Lock backends, SEC 17a-4 compliance
- ✅ Cryptographic vault hardening: HMAC-chained fast-path event log + RFC-6962 Merkle daily root + pluggable Signer (LocalSigner / GpgSigner / AgeSigner / KmsSigner OSS; CloudHsmSigner paid-tier stub) + 6 conformance vectors
- ✅ GenAI / LLM testing depth: 14 tests across 7 categories (hallucination, bias, robustness, safety, drift, PII, operational)
- ✅ LLM endpoint adapters: **LiteLLM** unified interface to 100+ providers (OpenAI, Anthropic, Bedrock, Azure, Cohere, etc.) + legacy adapters for backward compatibility
- ✅ RAG customer service worked example with FAISS retrieval + comprehensive GenAI validation
- ✅ Drift detection module — pluggable detector ABC + registry; builtin detectors (KS, Wasserstein, Page-Hinkley, MMD) with scipy/numpy fallbacks; opt-in frouros backend via `pip install 'mrm-core[drift]'`; `tabular.DataDrift`, `tabular.ConceptDrift`, `genai.SemanticDrift` routed through the registry; `mrm doctor` reports installed backends

---

## Feature backlog

Ordered by priority. Each feature is self-contained enough that Claude
Code can pick one up given the context here plus the repo.

---

### P1. SR 11-7 bundled compliance standard

- **STATUS:** done
- **WEDGE:** Without a US jurisdiction, the tool reads as Australia-
  only. SR 11-7 is the canonical bank MRM standard globally.
- **EFFORT:** 3-5 days config + 2 days feature-parity testing against
  ValidMind/ModelOp public materials

#### Approach

Pure plugin work using existing `@register_standard` framework. Mirror
the structure of `mrm/compliance/builtin/cps230.py`. Reference NIST
AI RMF as a crosswalk for AI/ML-specific provisions. Emit evidence in
JSON shaped for OpenPages / ServiceNow ingestion.

#### Definition of done

- `mrm/compliance/builtin/sr117.py` shipped with paragraph mappings,
  test mappings, governance checks, report generator
- Listed in `mrm docs list-standards`
- CCR example produces a valid SR 11-7 report when run with
  `--compliance standard:sr117`
- README updated to reflect two jurisdictions

---

### P2. EU AI Act Annex IV bundled compliance standard

- **STATUS:** done
- **WEDGE:** Loudest 2026 enforcement narrative. EU AI Act fully
  applicable August 2026, high-risk obligations extended to August
  2027.
- **EFFORT:** 5-7 days; revisit Q3 2026 when CEN/CENELEC harmonised
  standards land

#### Approach

Plugin maps to nine Annex IV technical documentation requirements.
Track EU AI Office support instruments (due Q2 2026). Align test
outputs to harmonised standards as they publish. Export packet format
consumable by GRC platforms — IBM OpenPages already has an EU AI Act
module to mirror.

#### Definition of done

- `mrm/compliance/builtin/eu_ai_act.py` shipped with Annex IV mappings
- Generated report aligns with the nine Annex IV technical doc
  requirements
- Listed in `mrm docs list-standards`
- README updated to reflect three jurisdictions

---

### P3. OSFI E-23 bundled compliance standard

- **STATUS:** done
- **WEDGE:** Cheap geographic spread. OSFI E-23 effective 1 May 2027,
  expanded to all FRFIs including insurers.
- **EFFORT:** 2-3 days (most logic reused from SR 11-7)

#### Approach

Plugin reuses paragraph framework from SR 11-7 (structurally similar).
Cross-reference NIST AI RMF and AMF (Quebec) where they converge.

#### Definition of done

- `mrm/compliance/builtin/osfi_e23.py` shipped
- Passes the same shape of acceptance tests as SR 11-7

---

### P4. Cross-standard crosswalk

- **STATUS:** done
- **WEDGE:** Cross-framework compliance mapping is a stated 2026+ MRM
  ask that no incumbent ships. Also the natural artefact for SSRN /
  LinkedIn credibility.
- **EFFORT:** 3-4 days

#### Approach

A single YAML/JSON crosswalk file in
`mrm/compliance/crosswalks/standards.yaml` mapping requirements across
CPS 230 ↔ SR 11-7 ↔ OSFI E-23 ↔ EU AI Act Annex IV ↔ NIST AI RMF.
Loadable via a new CLI command `mrm docs crosswalk`.

#### Definition of done

- Crosswalk file shipped covering all four standards
- `mrm docs crosswalk --from cps230 --to sr117` outputs paragraph-level
  mapping for the named pair
- Public-friendly version published as a markdown table in `docs/`
  (this becomes the SSRN/LinkedIn artefact)

---

### P5. Evidence vault — packet primitive + S3 Object Lock backend

- **STATUS:** done
- **WEDGE:** Today reports are regenerable markdown. Regulators want
  immutable, hashed, append-only evidence. This is the single biggest
  technical gap between "personal project" and "institution-grade
  tool."
- **EFFORT:** 7-10 days total (3-4 packet primitive + 2-3 S3 backend
  + 2-3 Databricks UC backend)

#### Architecture

Thin abstraction with pluggable backends, mirroring existing
`catalog_backends/` pattern. Vendor-neutral packet format. Backend
handles immutability via substrates already Cohasset-assessed for
SEC 17a-4 — <brand> doesn't reinvent WORM.

```
mrm-core/
├── evidence/
│   ├── base.py                # EvidenceBackend ABC + EvidencePacket + hash chain
│   ├── packet.py              # Packet schema, signing, verification
│   └── backends/
│       ├── local.py           # Local filesystem (dev only) — JSONL hash chain
│       ├── s3_object_lock.py  # AWS S3 + Object Lock Compliance mode
│       └── databricks_uc.py   # UC governance + audit log delivery to S3
```

#### CLI shape

```bash
mrm evidence freeze ccr_monte_carlo \
  --backend s3 \
  --bucket my-bank-evidence \
  --mode compliance \
  --retention 2555

mrm evidence verify <packet-uri>
mrm evidence list --model ccr_monte_carlo
```

#### Backend integration map

| Substrate | Bank adoption | Regulator acceptance | This sprint? |
|---|---|---|---|
| **AWS S3 + Object Lock (Compliance)** | Near-universal in AWS-shop banks | Cohasset-assessed for SEC 17a-4, CFTC, FINRA | ✅ default |
| **Databricks UC + audit log to S3** | High in modern bank/super stacks | UC governance + S3 audit log | ✅ |
| **Local filesystem + JSONL** | Dev only | Not for regulated use | ✅ |
| Azure Immutable Blob | Microsoft-shop banks | SEC 17a-4 / FINRA compliant | Defer |
| GCS Bucket Lock | Fintechs | Compliance-grade WORM | Defer |
| On-prem WORM (Cloudian, Scality, NetApp) | Tier 1 air-gapped | All SEC 17a-4-assessed | Likely works via s3 backend |
| ~~AWS Audit Manager~~ | Going to maintenance April 2026 | — | Skip — dying product |

#### Definition of done

- `EvidencePacket` class with content hashes, model artefact hash,
  test results, paragraph mappings, optional GPG signature
- Hash-chain semantics (each packet references the prior packet's hash)
- S3 Object Lock backend with Compliance-mode retention
- Databricks UC backend writing to UC-governed Delta tables + audit log
- Local filesystem backend for dev (clearly labelled as such)
- `mrm evidence freeze`, `verify`, `list` commands
- CCR example demonstrates the freeze workflow

---

### P6. ValidMind parity — GenAI / LLM testing depth

- **STATUS:** done
- **WEDGE:** Without credible LLM/GenAI test coverage, <brand> can't
  be considered a serious 2026 MRM tool. This is a parity ask, not a
  differentiator.
- **EFFORT:** 10-14 days

#### Approach

Add a new test namespace `genai.*` covering the categories regulators
and risk officers care about:

| Category | Tests to ship | Notes |
|---|---|---|
| **Hallucination / factual accuracy** | `genai.FactualAccuracy`, `genai.HallucinationRate` | Use ground-truth Q&A datasets; integrate with RAGAS-style metrics |
| **Bias and fairness** | `genai.DemographicParity`, `genai.OutputBias`, `genai.PromptBias` | Reuse fairness primitives from existing tabular tests where possible |
| **Robustness / adversarial** | `genai.PromptInjection`, `genai.JailbreakResistance`, `genai.AdversarialPerturbation` | Curated attack prompt sets; pass/fail thresholds |
| **Toxicity and safety** | `genai.ToxicityRate`, `genai.SafetyClassifier` | HuggingFace toxicity classifiers as default |
| **Drift and consistency** | `genai.OutputConsistency`, `genai.SemanticDrift` | Embedding-based drift detection |
| **PII leakage** | `genai.PIIDetection` | Microsoft Presidio (same library ValidMind uses) |
| **Cost and latency** | `genai.LatencyBound`, `genai.CostBound` | Operational risk tests |

Add a new model source type `model.location.type: llm_endpoint`
supporting OpenAI-compatible APIs, Anthropic, Bedrock, Databricks
Model Serving, and HuggingFace Inference Endpoints.

Ship a worked example: a RAG-based customer-service assistant
validated end-to-end against CPS 230 and EU AI Act mappings.

#### Definition of done

- `mrm/tests/builtin/genai.py` with at least 10 LLM tests covering all
  seven categories above
- New model source type `llm_endpoint` with adapters for OpenAI,
  Anthropic, Bedrock, Databricks Model Serving
- Worked example `genai_example/` showing a RAG assistant validated
  with `mrm test --models customer_service_rag`
- README updated to position <brand> as covering traditional, AI,
  and GenAI models
- Evidence packet format extends cleanly to LLM test outputs (token
  counts, prompt versions captured)

---

### P7. 1:1 Decision Replay — the acquirer wedge

- **STATUS:** done
- **WEDGE:** The single most defensible positioning sentence in the
  category. Regulators, plaintiff attorneys, boards, and acquirers will
  all demand the ability to reconstruct any AI/model decision from its
  constituent parts. No incumbent ships this as a first-class primitive.
  Combined with Fed **SR 26-2** (which supersedes SR 11-7 and explicitly
  expects "tamper-evident, integrity-protected, immutable, complete"
  decision logs), this is the wedge that turns `mrm-core` from "another
  governance CLI" into the de-facto open-source replay reference
  implementation. Inspired by Pelow's *AI Governance Framework* thesis.
- **EFFORT:** 12-16 days

#### Approach

Every model invocation (traditional model `.predict()` or LLM endpoint
call) emits a **Decision Record** capturing the four required
components for replay:

1. **Input state** — exact inference inputs at decision moment
   (features, prompt, retrieved context, system prompt)
2. **Model identity** — model URI, version, checkpoint hash, config
   hash (deterministic — same model + same hash means same weights)
3. **Inference parameters** — temperature, top-p, retrieval-k,
   seed, any decoding params; for tabular: preprocessing pipeline hash
4. **Output record** — raw model output *before* any downstream
   modification, formatting, or post-processing

Decision Records are append-only, hash-chained to the previous record
for the same model, and emitted in an OpenTelemetry-native wire format
(OTLP) so banks can pipe them into existing observability pipelines
without deploying a new agent.

```
mrm-core/
├── replay/
│   ├── record.py            # DecisionRecord schema (pydantic)
│   ├── capture.py           # Context manager + decorator for capture
│   ├── otlp.py              # OTLP export adapter
│   ├── verify.py            # Replay verification — re-run + diff
│   └── backends/
│       ├── local.py         # JSONL hash chain (dev only)
│       ├── s3.py            # S3 + Object Lock (production)
│       └── otlp_collector.py # Push to OTel collector
```

#### CLI shape

```bash
# Wrap a prediction with capture
mrm replay record ccr_monte_carlo --inputs trade_book.csv

# Reconstruct a specific decision
mrm replay reconstruct <record-id>

# Verify replay matches (catches non-determinism, drifted deps)
mrm replay verify <record-id> --tolerance 1e-6

# Sample-on-demand for regulator
mrm replay sample --model ccr_monte_carlo --since 2026-01-01 --n 50
```

#### Definition of done

- `DecisionRecord` pydantic schema with the four required components
- `@mrm.replay.capture` decorator + context manager for instrumented
  inference
- OTLP exporter so records flow to any OTel collector
- Hash-chain semantics across records for the same model
- `mrm replay record / reconstruct / verify / sample` commands
- LLM endpoint adapter (P6) updated to auto-capture prompt + retrieval
  context as part of the record
- CCR example and RAG example both demonstrate end-to-end replay
- README adds "Replay-by-default" as a top-line positioning bullet

---

### P8. Fed SR 26-2 bundled compliance standard

- **STATUS:** done
- **WEDGE:** SR 26-2 supersedes SR 11-7 and SR 21-8 for US banks >$30B
  assets. Without it, the SR 11-7 plugin reads as outdated to Tier-1
  US buyers in 2026-2027. SR 26-2 explicitly mandates AI activity
  logging — natural anchor for P7 (Replay).
- **EFFORT:** 4-6 days

#### Approach

Mirror `mrm/compliance/builtin/sr117.py`. Add explicit mappings to:

- AI/ML activity logging requirements (anchored to P7 Decision Records)
- Tamper-evident audit trail requirements (anchored to P5 Evidence
  Vault hash-chain)
- Risk-tiering for AI models (model materiality classification)
- Independent model validation cadence for AI/GenAI models

Cross-reference NIST AI RMF, FFIEC IT Examination Handbook, OCC
2011-12 in the crosswalk (P4 extension).

#### Definition of done

- `mrm/compliance/builtin/sr26_2.py` shipped with paragraph mappings
- Crosswalk (P4) extended to map SR 11-7 ↔ SR 26-2 transition
- Listed in `mrm docs list-standards`
- CCR example produces a valid SR 26-2 report
- Replay (P7) records are referenced as the evidence type for the
  AI-activity-logging clauses

---

### P9. Cryptographic evidence vault hardening

- **STATUS:** done
- **WEDGE:** P5 ships hash-chained packets. Banks under SR 26-2 will
  ask: "Who signed the chain root? How do you prove integrity across
  the year?" The SR-26.2-MRM reference repo answers this with
  **HMAC-chained event capture, daily Merkle tree aggregation, HSM-
  rooted signatures, OTLP wire protocol.** Bringing this design into
  `mrm-core` (basic) and <brand> Cloud (HSM-backed) is the cleanest
  OSS/paid line in the entire product.
- **EFFORT:** 8-12 days OSS + ongoing for Cloud HSM

#### Approach

Extend the existing `evidence/` module:

1. **HMAC-chained events** — each event in a session signed with a
   session-scoped HMAC key (rotated daily)
2. **Daily Merkle tree** — at end of each UTC day, aggregate all
   evidence packets + decision records into a Merkle tree; publish
   the root
3. **Root signature** — root signed with a long-lived key. In OSS:
   GPG / age. In <brand> Cloud: HSM-backed (FIPS 140-2 Level 3+).
4. **Conformance test vectors** — ship positive + negative test
   corpus (mirror SR-26.2-MRM's `spec/test-vectors/`) so third parties
   can prove their implementation conforms.

```
mrm-core/
├── evidence/
│   ├── chain.py             # HMAC-chained event log
│   ├── merkle.py            # Daily Merkle aggregation + root publication
│   ├── sign.py              # Root signing (GPG/age in OSS; HSM in Cloud)
│   └── test_vectors/        # Conformance corpus
```

#### CLI shape

```bash
mrm evidence root publish --date 2026-05-15
mrm evidence root verify <root-hash>
mrm evidence conformance run    # runs the test-vector suite
```

#### OSS vs SaaS split

| Capability | mrm-core (OSS) | <brand> Cloud (paid) |
|---|---|---|
| HMAC-chained events | ✅ | ✅ |
| Daily Merkle aggregation | ✅ | ✅ |
| GPG/age root signing | ✅ | — |
| **HSM-backed root signing (FIPS 140-2 L3+)** | — | ✅ |
| Long-term retention SLA (7yr) | — | ✅ |
| Regulator-portal sample export | — | ✅ |
| Customer-managed keys (BYOK) | — | ✅ (Enterprise) |

#### Definition of done

- `evidence/chain.py`, `merkle.py`, `sign.py` shipped
- Daily Merkle root publication command + verification command
- GPG / age signature support in OSS
- Conformance test vector suite (positive + negative cases)
- Spec document under `docs/spec/evidence-vault-v1.md` published with
  PRD-style lifecycle (mirror SR-26.2-MRM governance pattern)
- README adds "Cryptographic chain-of-custody" as a top-line bullet

---

### P10. Drift detection — pluggable detectors + frouros optional install

- **STATUS:** done
- **WEDGE:** Drift is the *one* MRM test family every regulator cites
  (CPS 230 ongoing monitoring, SR 11-7 §II.C, SR 26-2 §II.AI.D, EU AI
  Act post-market monitoring). STRATEGY.md already claims "frouros
  integration for statistical drift detection on LLM outputs" — the
  documented-but-undelivered gap reads worse than missing features.
  Plus: banks evaluate `mrm-core` on whether it installs cleanly in
  air-gapped VDIs. A conditional-install `frouros` integration with
  pure-numpy / scipy fallbacks is genuinely differentiating against
  ValidMind (which bundles its own monitoring SDK).
- **EFFORT:** 5-7 days

#### Approach

A new `mrm/drift/` module mirroring the pluggable patterns of
`compliance/`, `evidence/sign.py`, and `replay/backends/`:

```
mrm-core/
├── drift/
│   ├── base.py           # DriftDetector ABC + DriftResult schema
│   ├── registry.py       # @register_detector + lazy backend selection
│   ├── builtin/
│   │   ├── ks.py         # Kolmogorov-Smirnov (data drift)
│   │   ├── wasserstein.py# Wasserstein distance (data drift)
│   │   ├── mmd.py        # Maximum Mean Discrepancy (embeddings / RAG)
│   │   └── page_hinkley.py # Page-Hinkley (concept drift, streaming)
│   └── backends/
│       ├── scipy.py      # Pure scipy.stats fallbacks
│       └── frouros.py    # frouros-backed implementations (opt-in)
```

The detector ABC exposes `fit(reference)` + `score(current)` +
`detect(current, threshold) -> DriftResult`. The registry picks the
frouros backend when available, falls back to scipy/numpy otherwise.
Lazy imports inside `run()` so the package import doesn't crash on
air-gapped installs.

Tests that consume detectors live in the existing test namespaces
(no separate `drift.*` namespace -- regulators read these as
performance-monitoring tests):

| Test | Frouros backend | Pure-scipy fallback |
|---|---|---|
| `tabular.DataDrift` | KS / Wasserstein / MMD | `scipy.stats.ks_2samp` |
| `tabular.ConceptDrift` | DDM / EDDM / ADWIN | Page-Hinkley in pure numpy |
| `genai.SemanticDrift` | MMD over embeddings | cosine-distance KS test |
| `genai.OutputConsistency` | sliding-window divergence | rolling-std fallback |

#### OSS install footprint

- `frouros` moves from the existing `[genai]` extra to a new
  `[drift]` extra in `pyproject.toml`.
- `pip install mrm-core` works on every airgapped bank VDI.
- `pip install 'mrm-core[drift]'` enables the frouros backends.
- `mrm doctor` reports which drift backends are available so users
  know exactly what they have without reading source.

#### CLI surface

```bash
mrm doctor                          # capability report (drift backends + signers)
mrm test --select tier:tier_1      # drift tests run as part of the normal pack
```

Drift tests integrate with P7 (Decision Replay) so the *reference*
and *current* windows used in a drift computation are captured in
the DecisionRecord -- regulators can replay exactly what drifted.

#### Definition of done

- `mrm/drift/` module shipped with detector ABC + registry +
  scipy fallbacks + frouros-backed implementations
- `tabular.DataDrift`, `tabular.ConceptDrift`, `genai.SemanticDrift`,
  `genai.OutputConsistency` all routed through the registry
- `pyproject.toml` exposes `drift` as a standalone extra; `frouros`
  removed from the `genai` bundle
- `mrm doctor` command lists available drift backends + crypto signers
- Pytest suite covers fallback path (no frouros) AND frouros path
  (skipped on environments without it)
- README: drift coverage matrix + a "no extra deps required" callout

---

### P11. LLM adversarial red-team pack + RAG context capture

- **STATUS:** next (extends P6 and P10)
- **WEDGE:** P6 ships hallucination + bias + toxicity tests. The
  `llm_eval` repo ships a **50+-template adversarial library** (PII
  exposure, fiduciary-bypass, system-prompt override) and **financial-
  F1 entity-weighted accuracy** (severity-weighted errors: $10B vs
  $10M weighted higher than grammar). Both are domain-specific and
  directly applicable to bank LLM use-cases.
- **EFFORT:** 6-8 days

#### Approach

- Ship `mrm/tests/builtin/genai_adversarial.py` with at least 50 attack
  templates across categories: fiduciary bypass, PII extraction,
  jailbreak chains, system-prompt override, regulatory-claim
  fabrication
- Ship `mrm/tests/builtin/genai_financial.py` with entity-weighted F1
  for tickers, ISINs, currencies, monetary values
- Templates live in `mrm/tests/data/adversarial/*.json` so banks can
  add their own without forking
- All adversarial test runs auto-emit Decision Records (P7) — every
  attack attempt is replayable evidence
- Explainability hooks: optional SHAP/LIME attribution captured in the
  Decision Record for tabular models (regulator interpretability ask
  under SR 11-7 §3.3 / SR 26-2)

#### Definition of done

- 50+ adversarial templates shipped, namespaced under
  `genai.adversarial.*`
- Financial-F1 test shipped with severity weighting config
- RAG example (from P6) extended to demonstrate adversarial sweep
- Decision Records emitted for every adversarial run
- README updated: "50+ adversarial templates, regulator-shaped reports"

---

### P12. Regulator engagement + spec governance

- **STATUS:** next (non-code; runs in parallel)
- **WEDGE:** Acquirers in this category pay for **regulator mindshare**
  more than for revenue at small scale. The SR-26.2-MRM repo's
  governance posture (`GOVERNANCE.md`, PRD lifecycle, intent to
  transfer to neutral foundation) is the template. Pelow's framework
  positions replay capability via comment-letter strategy. Both
  imitable in weeks, not months.
- **EFFORT:** 5-7 days docs + ongoing relationship work

#### Approach

1. Add `docs/adr/` directory with Architecture Decision Records for
   every load-bearing design choice (replay schema, evidence chain,
   compliance plugin contract). Mirror Pelow's `docs/adr/` pattern.
2. Add `GOVERNANCE.md` documenting the spec lifecycle for `mrm-core`
   itself: PRD drafts, public comment via GitHub issues, convergence
   criteria, neutral-foundation transition statement (CNCF / OpenSSF /
   Linux Foundation FinOS as candidates).
3. Add `docs/spec/` with versioned normative specs for:
   - Decision Record schema (v1)
   - Evidence Vault chain format (v1)
   - Compliance Plugin contract (v1)
4. Submit public comment to **SR 26-2** docket referencing `mrm-core`
   as a reference implementation. Free; disproportionately valuable
   for acquirer narrative.
5. Submit `mrm-core` to **FINOS** AI Governance Framework as an
   implementation pattern.

#### Definition of done

- `docs/adr/` with ≥5 ADRs (replay, evidence chain, plugin contract,
  OTLP format, OSS/Cloud split)
- `GOVERNANCE.md` with PRD lifecycle + foundation-transfer intent
- `docs/spec/` with ≥3 versioned specs
- One public comment submitted to a regulator docket citing `mrm-core`
- One application/PR to FINOS AI Governance Framework

---

### P13. ValidMind parity — GRC platform integration

- **STATUS:** backlog (after P7-P12)
- **WEDGE:** Banks live in OpenPages/ServiceNow/Workiva. Without push
  connectors, `mrm-core` outputs sit in a developer's filesystem
  unread by the people who actually do governance.
- **EFFORT:** 10-14 days

#### Approach

A new `grc/` module with pluggable connectors, mirroring the
`evidence/backends/` pattern.

```
mrm-core/
├── grc/
│   ├── base.py                # GRCConnector ABC
│   └── connectors/
│       ├── openpages.py       # IBM OpenPages — REST API
│       ├── servicenow_irm.py  # ServiceNow IRM — Table API
│       ├── workiva.py         # Workiva Wdesk — Wdata API
│       └── archer.py          # Archer IRM — REST API (later)
```

Push semantics:
- Validation results → "Test Result" or "Issue" records
- Evidence packets → "Evidence" attachments with link-back to the
  immutable store URI
- Trigger events → "Workflow Task" or "Re-validation Required" records
- Compliance reports → "Document" records with paragraph-level
  metadata

CLI shape:

```bash
mrm grc push ccr_monte_carlo --to openpages --workspace mrm-prod
mrm grc list-connectors
mrm grc test-connection openpages
```

Credentials stay out of YAML — read from env vars or a credentials
file the CLI never logs.

#### Priority of connectors

| Connector | Bank adoption | Build first? |
|---|---|---|
| **IBM OpenPages** | Largest installed base in Tier 1 banks; Gartner MQ Leader | ✅ yes |
| **ServiceNow IRM** | Aggressive consolidator; high in modern stacks | ✅ yes |
| **Workiva** | Strong in reporting/audit functions | ✅ yes |
| Archer IRM | Legacy enterprise | Defer |
| MetricStream | Large enterprise | Defer |

#### Definition of done

- `GRCConnector` ABC with `push_validation`, `push_evidence`,
  `push_trigger`, `push_report` methods
- OpenPages, ServiceNow IRM, and Workiva connectors implemented
- `mrm grc push`, `list-connectors`, `test-connection` commands
- CCR example demonstrates push to a mock OpenPages endpoint
- Credentials handled via env vars / credentials file — never logged

---

### P14. Quant model worked example — XVA via ORE

- **STATUS:** backlog (parallel-safe with P5-P13; pick up when capacity
  allows)
- **WEDGE:** Broadens <brand> from "one quant model" to "platform for
  quant model risk." XVA is the founder's stated research interest.
- **EFFORT:** 7-10 days

#### Approach

Don't build an XVA engine. Wrap the **Open Source Risk Engine (ORE)**
— QuantLib-based, sponsored by Acadia / Post Trade Solutions, the
canonical open-source XVA library. The positioning becomes sharp:
*"<brand> validates ORE-priced XVA models against regulator-accepted
standards."*

#### Definition of done

- Worked example `xva_example/` with ORE-priced XVA model registered
  as an `mrm-core` model
- CVA/DVA/FVA validation tests + Chebyshev tensor approximation
  accuracy tests
- Publishes to UC, generates SR 11-7 + EU AI Act evidence
- README update positioning <brand> as quant-model-aware

---

### P15. Quant model worked example — IRB credit risk (PD/LGD/EAD)

- **STATUS:** backlog (after P14)
- **WEDGE:** Closes the third quant model archetype (counterparty,
  derivatives, credit). Opens Finalyse channel partnership.
- **EFFORT:** 5-7 days

#### Approach

Worked example builds a PD model (synthetic data, scikit-learn or
xgboost), runs IRB-style backtests (calibration, discriminatory power,
stability, MoC). Map to ECB Guide to Internal Models 2025 paragraphs
as a separate compliance plugin (`ecb_irb` or similar).

Reach out to Finalyse early — their SAS/R/Python ECB validation
toolkit is consultancy-delivered. Productising it via <brand> is
mutually beneficial.

#### Definition of done

- Worked example `irb_example/`
- IRB-flavoured tests in `mrm/tests/builtin/credit.py`
- ECB IRB compliance plugin
- Finalyse outreach attempted (separate from code work)

---

### P16. <brand> Cloud — minimum viable hosted layer

- **STATUS:** backlog (start when first design partner signed)
- **WEDGE:** The commercial monetisation path. Without it the OSS
  doesn't pay for itself.
- **EFFORT:** weeks-to-months; not single-sprint

#### Scope (minimum viable)

- Hosted scheduled run orchestration (<brand> Cloud reads `mrm-core`
  configs from a connected Git repo and runs them on schedule)
- RBAC, SSO (Okta, Azure AD), audit log of who ran what
- Web UI for risk officers / validators / auditors to read reports,
  approve runs, see triggers
- Evidence vault as managed service (<brand> operates the S3 Object
  Lock bucket; customer doesn't have to)
- Hosted GRC integrations with credentials managed by <brand> Cloud
- Workflow engine: validator approves, owner re-runs, escalation paths

Build choice: thin wrapper around `mrm-core`. Anything in the OSS CLI
must remain in the OSS CLI. Cloud is *operations* over the CLI, not a
fork.

**Premium SaaS-only features anchored to OSS primitives:**

| Tier | Feature | Anchored to OSS primitive |
|---|---|---|
| Team | Hosted scheduled `mrm test` runs | `mrm-core` CLI |
| Team | Web UI for lineage, reports, evidence diff | DAG + evidence packets |
| Team | SSO + RBAC + workspace audit log | — |
| Business | **HSM-backed evidence signing (FIPS 140-2 L3+)** | P9 chain root signing |
| Business | **Long-term replay storage (7-yr regulator retention)** | P7 Decision Records |
| Business | **Regulator-portal export (sample-on-demand)** | P7 `mrm replay sample` |
| Enterprise | Customer-managed keys (BYOK), VPC deploy, on-prem agent | — |
| Enterprise | **Certified conformance program** (per P9 test vectors) | P9 conformance suite |

#### Definition of done

Out of scope to specify in detail until P1-P12 are done and a design
partner is signed. This entry exists to keep the commercial layer
visible in the backlog.

---

### P17. Crosswalk auto-update via authoritative source sync

- **STATUS:** backlog (after P5-P13; requires LLM API + human review
  workflow)
- **WEDGE:** Standards evolve (CPS 230 updated Nov 2024, EU AI Act
  harmonised standards due Q3 2026). Manual crosswalk maintenance
  doesn't scale beyond 10 standards. Auto-sync with human-in-the-loop
  approval is a differentiator no incumbent ships.
- **EFFORT:** 10-14 days (5-7 LLM integration + 3-5 human review
  workflow + 2-3 PDF/web/RDF parsers)

#### Approach

Add authoritative source metadata to each standard (PDF URL, web page,
SPARQL endpoint). New CLI command `mrm docs crosswalk --standard
<name> --update` fetches the source, extracts structure via Claude/LLM
API, compares against existing crosswalk, generates a diff proposal
(new/changed/deprecated paragraphs with suggested mappings), and
presents to user for approval.

Human review is mandatory — per CLAUDE.md governance principle: "Don't
add LLM-based AI features without a clear governance story." This HAS
a governance story: human approves all changes, audit trail, versioned
updates become evidence artefacts.

Add authoritative source metadata in each standard class or separate
YAML:

```yaml
standards:
  cps230:
    authoritative_source:
      type: pdf_url
      url: https://www.apra.gov.au/.../CPS_230.pdf
      checksum: sha256:abc123...
    last_updated: 2024-11-07
    version: "2024"
  
  sr117:
    authoritative_source:
      type: web_page
      url: https://www.federalreserve.gov/.../sr1107.htm
      selector: ".content"
    last_updated: 2011-04-04
    version: "2011"
  
  euaiact:
    authoritative_source:
      type: rdf_sparql
      endpoint: https://op.europa.eu/sparql-endpoint
      query: "SELECT ?text WHERE { <eur-lex:32024R1689> ?p ?text }"
    last_updated: 2024-08-01
    version: "2024"
```

CLI workflow:

```bash
$ mrm docs crosswalk --standard cps230 --update

Fetching: https://www.apra.gov.au/.../CPS_230.pdf
Checksum: unchanged since last fetch
Extracting paragraphs using Claude API...
Extracted 42 paragraph groups

Comparing with crosswalk (current: 10 paragraphs mapped)...

==== PROPOSED UPDATES ====

[NEW] Para 43 "Third-party risk management"
  Suggested mapping: → SR 11-7 III.E (vendor management)
  Action: [A]dd / [S]kip / [E]dit

[CHANGED] Para 30-33 "Controls and Mitigation"
  Old: "establish and maintain controls"
  New: "must establish, maintain, and test controls"
  Current mappings: SR 11-7 III.A, EU AI Act IV.4, OSFI E-23 3.1
  Action: [K]eep / [R]eview / [S]kip

[DEPRECATED] Para 5-7 (removed in 2024 version)
  Currently mapped to: SR 11-7 I.A
  Action: [R]emove / [K]eep / [A]rchive

Generate update report? [Y/n]
```

Human review UI options:
- CLI interactive prompts (above)
- Web UI (if <brand> Cloud exists)
- VS Code extension with diff view
- Git-style patch file for review/approval

Cost model: ~$0.10-0.50 per standard update via Claude API (couple
hundred KB of regulatory text).

#### Definition of done

- Standard metadata format with `authoritative_source` fields
- PDF, web page, and RDF/SPARQL parsers implemented
- Claude API integration for structure extraction + similarity
  inference
- CLI command `mrm docs crosswalk --standard <name> --update` with
  interactive approval workflow
- Diff output showing new/changed/deprecated paragraphs with suggested
  cross-mappings
- Versioned update metadata written to `standards.yaml` (becomes
  evidence artefact)
- Human approval audit trail (who approved what, when)
- README / docs explain the workflow and governance story

---

## Acquirer landscape

| Acquirer | Why they'd buy | Recent signals |
|---|---|---|
| **Moody's** | Already aggressive in MRM tech, wants regulator-facing tooling | Acquired RMS (2021), continuing model-risk-adjacent acquisitions |
| **MSCI** | Risk analytics platform play | Active in factor / ESG / climate risk M&A |
| **SAS** | Legacy MRM incumbent looking for modern UX | Slow but well-capitalised |
| **Workiva** | GRC + reporting platform, missing the model-risk vertical | Active GRC consolidator |
| **ServiceNow** | Aggressive GRC buyer, missing the modelling depth | Multiple recent governance acquisitions |
| **S&P Global** | Risk tech footprint via Market Intelligence | Periodic regtech acquisitions |
| **IBM** | OpenPages dominance; missing developer-first MRM tooling | Aggressive on AI governance acquisitions |
| **Bloomberg** | Quant tech adjacency, would be a stretch | Less likely but possible |

### Realistic deal range

Governance / regtech acquisitions in 2025-2026 cluster in the
**$15-50M range**, with outliers higher when team or
regulator-relationship is exceptional. Realistic exit window is
18-36 months from now in the **$10-40M range**, with upside if EU AI
Act enforcement creates a buying panic in bank-MRM specifically.

### What kills the deal

- Single-jurisdiction tool → fixed by P1-P3 ✅
- Mutable evidence → fixed by P5 ✅
- No GenAI parity → fixed by P6 ✅
- **No replay primitive** → fixed by P7 (the wedge)
- **No US 2026+ jurisdiction** → fixed by P8 (SR 26-2)
- **No cryptographic chain-of-custody** → fixed by P9 ✅
- **No drift monitoring** → fixed by P10 ✅
- **No regulator mindshare** → fixed by P12 (comment letters,
  FINOS submission, ADR/GOVERNANCE.md posture)
- No external customers → fixed by design partner work (see channels)
- No GRC integration → fixed by P13
- Founder-only contributor graph — needs ≥1 external contributor
- Reinventing infrastructure already owned by the buyer (e.g. building
  proprietary evidence storage instead of writing to S3 Object Lock)

---

## Design partner channels

Pilot conversations that change trajectory. Macquarie excluded —
politics, contracting via Boson constrains commercial freedom.

| Channel | Targets | Mechanism | Lead time |
|---|---|---|---|
| **Smaller AU banks/super funds** | Judo Bank, Athena, Hostplus, Aware Super, AMP, Latitude, Plenti — all CPS 230 obligated | Direct outreach to model risk leads; free pilot for paragraph mapping validation; AFR / RMA / FINSIA conferences | 4-8 weeks per pilot |
| **Big-4 risk consulting** | KPMG AU, Deloitte AU, EY AU, PwC AU. KPMG AU has published 5+ CPS 230 advisory pieces in last 6 months — hungry for tooling | Channel partnership; co-marketing CPS 230 reference implementation; PRMIA Sydney speaking | 6-12 weeks |
| **Quant validation channel** | Finalyse (EU credit risk validators); Acadia/Post Trade Solutions (ORE sponsors) | Contribute IRB tests upstream to ORE community; pitch Finalyse on open-source interop | 4-8 weeks |

Big-4 is highest leverage — one consulting partnership puts <brand>
in front of every Tier-2 AU institution at once.

Anthony Hough at Macquarie is a relationship to nurture for advice
and introductions, not as a customer.

---

## "Why now" deck — primary sources

| Date | Event | Source |
|---|---|---|
| 1 July 2025 | APRA CPS 230 fully effective | APRA |
| 30 April 2026 | APRA industry-wide AI letter + CPS 230 amendments | APRA |
| 2 August 2026 | EU AI Act fully applicable | EU AI Office |
| 2 August 2027 | EU AI Act high-risk obligations (embedded systems) | EU AI Office |
| 1 May 2027 | OSFI E-23 effective (expanded scope) | OSFI |
| Through 2026 | Fed SR 11-7 reinterpretation for GenAI | Fed/OCC |

The 30 April 2026 APRA AI letter is the lead slide.

Commercial-signal proof points: MAS MindForge AI RMT (March 2026),
Experian-ValidMind launch (July 2025), FIS Insurance Risk Suite AI
Assistant (Feb 2026), Solytics in RegTech100/RiskTech100 2026.

---

## Constraints to respect

- **Time sovereignty.** Founder is contracting at Macquarie ~full-time.
  Build cadence assumes evenings and weekends. Features that require
  sustained focus blocks should be batched. The feature numbering
  above is designed so each P-item is a self-contained sprint.
- **Australian residence.** Travel for pilots / conferences feasible
  but expensive. Prefer remote-first design-partner relationships.
- **No outside capital yet.** Bootstrap-friendly choices throughout —
  no managed-service dependencies, no per-seat commercial licences in
  the core.
- **Credential ceiling.** Lack of quant PhD blocks certain in-house
  MRM career paths but is **not** a constraint as a vendor. Vendors
  are judged on the product.

---

## When this document changes

Edit this file when:

- A feature changes STATUS (`next` → `in-progress` → `done`)
- A new priority needs to be inserted into the backlog
- Story A is replaced by another (rare — flag as a major pivot)
- Acquirer landscape shifts materially (named buyer enters or exits)
- A pilot is signed or fails
- A named integration target shifts (e.g. ORE deprecation, Finalyse
  acquired, Databricks UC pricing changes materially)
- The brand name is confirmed (do find-and-replace `<brand>` →
  chosen name across this file and `CLAUDE.md`, then remove this
  bullet)

Don't edit it for tactical changes, single-feature decisions, or
bugfixes — those go in commit messages and `CLAUDE.md`.
