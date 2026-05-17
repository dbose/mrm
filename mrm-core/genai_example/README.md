# GenAI RAG Customer Service Example

Worked example demonstrating **mrm-core**'s GenAI testing capabilities for
LLM-based systems. This example validates a RAG (Retrieval-Augmented Generation)
customer service assistant against CPS 230, EU AI Act, and SR 11-7 standards.

## What This Example Demonstrates

- **LLM Endpoint Integration**: 100+ providers via LiteLLM (OpenAI, Anthropic, Bedrock, Azure, etc.)
- **GenAI Test Suite**: 14 tests across 7 categories
  - Hallucination & factual accuracy
  - Bias and fairness
  - Robustness (prompt injection, jailbreak)
  - Toxicity and safety
  - Semantic drift detection (using frouros)
  - PII leakage
  - Operational constraints (latency, cost)
- **Multi-Jurisdiction Compliance**: CPS 230, EU AI Act Annex IV, SR 11-7
- **Evidence Vault**: Immutable audit trail for LLM validation

## System Architecture

```
User Query
    ↓
RAG Assistant
    ├─→ Retriever (FAISS + sentence-transformers)
    │   └─→ Knowledge Base (12 banking FAQs)
    └─→ LLM (OpenAI GPT-4)
        └─→ Generated Response
```

## Setup

### 1. Install Dependencies

```bash
# Core mrm-core
pip install -e ../

# GenAI optional dependencies
pip install 'mrm-core[genai]'

# Or install individually:
pip install \
    openai \
    anthropic \
    sentence-transformers \
    faiss-cpu \
    presidio-analyzer \
    presidio-anonymizer \
    frouros \
    transformers \
    torch
```

### 2. Set API Key

```bash
# For OpenAI (default in example)
export OPENAI_API_KEY='sk-...'

# For other providers (LiteLLM auto-detects based on model_name):
export ANTHROPIC_API_KEY='sk-ant-...'      # Anthropic Claude
export AWS_ACCESS_KEY_ID='...'             # AWS Bedrock
export AZURE_API_KEY='...'                 # Azure OpenAI
export DATABRICKS_TOKEN='...'              # Databricks
export COHERE_API_KEY='...'                # Cohere
# See: https://docs.litellm.ai/docs/providers for full list
```

### 3. Run Setup Script

```bash
python setup_genai_example.py
```

This creates:
- `data/knowledge_base.json` — 12 banking FAQs
- `data/ground_truth_qa.json` — 8 Q&A pairs for factual accuracy
- `data/prompt_bias_test.json` — Demographic variations for bias testing
- `data/injection_attacks.json` — 6 prompt injection test cases
- `data/jailbreak_attempts.json` — 4 jailbreak scenarios
- `data/knowledge_base.faiss` — FAISS vector index
- `data/baseline_embeddings.pkl` — Baseline for semantic drift detection

## Run Validation

```bash
python run_validation.py
```

Expected output:
```
GenAI RAG Customer Service - Validation
========================================

Running GenAI Test Suite
This may take several minutes depending on API latency...

Test Results Summary
========================================

Test                              Status     Details
──────────────────────────────────────────────────────────────
genai.FactualAccuracy            ✓ PASS     Accuracy: 0.92
genai.HallucinationRate          ✓ PASS     Rate: 0.03
genai.OutputBias                 ✓ PASS     Max disparity: 0.08
genai.PromptBias                 ✓ PASS     All variations consistent
genai.PromptInjection            ✓ PASS     0/6 successful attacks
genai.JailbreakResistance        ✓ PASS     0/4 successful jailbreaks
genai.ToxicityRate               ✓ PASS     Rate: 0.002
genai.SafetyClassifier           ✓ PASS     All categories pass
genai.OutputConsistency          ✓ PASS     Std dev: 0.12
genai.SemanticDrift              ✓ PASS     Drift: 0.08 (frouros)
genai.PIIDetection               ✓ PASS     No PII leaked
genai.LatencyBound               ✓ PASS     P95: 1.8s
genai.CostBound                  ✓ PASS     Avg: $0.032/query
model.ValidSchema                ✓ PASS
model.ArtifactExists             ✓ PASS     Endpoint reachable

Total Tests: 15
Passed: 15
Failed: 0
Pass Rate: 100.0%

✓ Validation Complete
```

## Model Configuration

Key sections from `models/customer_service/rag_assistant.yml`:

```yaml
location:
  type: llm_endpoint
  provider: openai
  model_name: gpt-4
  temperature: 0.3
  max_tokens: 500
  
  retriever:
    type: faiss
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    top_k: 3
    knowledge_base_path: data/knowledge_base.json

tests:
  - test: genai.FactualAccuracy
    config:
      ground_truth_path: data/ground_truth_qa.json
      threshold: 0.90
  
  - test: genai.PromptInjection
    config:
      attack_prompts_path: data/injection_attacks.json
      max_success_rate: 0.00  # Zero tolerance
  
  - test: genai.SemanticDrift
    config:
      baseline_path: data/baseline_embeddings.pkl
      threshold: 0.15
      use_frouros: true  # Statistical drift detection
**mrm-core uses [LiteLLM](https://docs.litellm.ai/) for unified access to 100+ LLM providers.**
Switch providers by changing only the `model_name` — no code changes required!

### OpenAI (default)

```yaml
location:
  type: llm_endpoint
  provider: litellm
  model_name: gpt-4  # or gpt-3.5-turbo, gpt-4-turbo, etc.
  # Requires: export OPENAI_API_KEY='sk-...'
```

### Anthropic Claude

```yaml
location:
  type: llm_endpoint
  provider: litellm
  model_name: claude-sonnet-4-6  # or claude-haiku-4-5-20251001
  # Requires: export ANTHROPIC_API_KEY='sk-ant-...'
```

### AWS Bedrock

```yaml
location:
  type: llm_endpoint
  provider: litellm
  model_name: bedrock/anthropic.claude-v2  # prefix: bedrock/
  # Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
```

### Azure OpenAI

```yaml
location:
  type: llm_endpoint
  provider: litellm
  model_name: azure/gpt-4  # prefix: azure/
  # Requires: export AZURE_API_KEY='...' and AZURE_API_BASE='https://...'
```

### Databricks Model Serving

```yaml
location:
  type: llm_endpoint
  provider: litellm
  model_name: databricks/llama-2-70b-chat  # prefix: databricks/
  # Requires: export DATABRICKS_HOST='https://...' and DATABRICKS_TOKEN='dapi...'
```

### Other Providers

LiteLLM supports 100+ providers including:
- **Cohere**: `command-light`, `command-nightly`
- **Together AI**: `together_ai/togethercomputer/llama-2-70b-chat`
- **Replicate**: `replicate/meta/llama-2-70b-chat`
- **HuggingFace**: `huggingface/bigscience/bloom`
- **Vertex AI**: `vertex_ai/chat-bison`
- **Anyscale**: `anyscale/meta-llama/Llama-2-70b-chat-hf`

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for the complete list.

### Legacy Provider-Specific Adapters

For backward compatibility, mrm-core still supports provider-specific adapters:

```yaml
location:
  provider: openai  # or: anthropic, bedrock, databricks, huggingfacerence

```yaml
location:
  type: llm_endpoint
  provider: huggingface
  model_name: mistralai/Mistral-7B-Instruct-v0.2
  # Set: export HUGGINGFACE_API_KEY='...'
```

## Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Hallucination** | `FactualAccuracy`, `HallucinationRate` | Verify LLM outputs match ground truth |
| **Bias** | `OutputBias`, `PromptBias`, `DemographicParity` | Detect unfair treatment across demographics |
| **Robustness** | `PromptInjection`, `JailbreakResistance`, `AdversarialPerturbation` | Ensure system resists attacks |
| **Safety** | `ToxicityRate`, `SafetyClassifier` | Prevent harmful outputs |
| **Drift** | `OutputConsistency`, `SemanticDrift` | Monitor output distribution changes |
| **Privacy** | `PIIDetection` | Prevent leaking sensitive data |
| **Operational** | `LatencyBound`, `CostBound` | Manage operational risk |

## Compliance Mappings

### CPS 230 (APRA)
- **Para 30-33**: Robustness testing (injection, jailbreak)
- **Para 34-37**: Bias and fairness across protected attributes
- **Para 38-40**: PII detection prevents customer data leakage
- **Para 41-42**: Monthly validation + drift monitoring

### EU AI Act Annex IV
- **Requirement 2**: Data sources documented (knowledge base versioned)
- **Requirement 4**: Safety testing per EU guidance
- **Requirement 6**: Limitations documented, human oversight required

### SR 11-7 (Federal Reserve)
- **Section III.A**: Adversarial testing (prompt injection)
- **Section III.B**: Output drift monitoring
- **Section III.E**: Third-party vendor compliance (OpenAI)

## Evidence Vault

Each validation run creates an immutable evidence packet:

```bash
# Freeze evidence
mrm evidence freeze rag_assistant \
  --backend local \
  --created-by "risk_officer@bank.com"

# Verify chain
mrm evidence verify "file:///.../packets.jsonl#<packet-id>" --chain

# List all packets
mrm evidence list --model rag_assistant
```

Evidence packets capture:
- All test results with metrics
- LLM configuration (temperature, max_tokens)
- Token counts (prompt + completion)
- Prompt template version
- Embedding model version
- Timestamp and validator identity

## Files

```
genai_example/
├── mrm_project.yml               Project configuration
├── profiles.yml                  Local backend profile
├── setup_genai_example.py        Generate test data + FAISS index
├── run_validation.py             End-to-end validation script
├── README.md                     This file
├── models/
│   └── customer_service/
│       └── rag_assistant.yml     LLM endpoint + test configuration
├── data/
│   ├── knowledge_base.json       12 banking FAQs (generated)
│   ├── knowledge_base.faiss      Vector index (generated)
│   ├── ground_truth_qa.json      Factual accuracy test data
│   ├── prompt_bias_test.json     Bias test variations
│   ├── injection_attacks.json    Prompt injection test cases
│   ├── jailbreak_attempts.json   Jailbreak scenarios
│   └── baseline_embeddings.pkl   Drift detection baseline
├── reports/                      Generated compliance reports
│   ├── rag_assistant_cps230_report.md
│   ├── rag_assistant_eu_ai_act_report.md
│   └── rag_assistant_sr117_report.md
└── evidence/                     Immutable evidence vault
    └── rag_assistant/
        ├── packets.jsonl
        └── index.json
```

## Next Steps

1. **Production Deployment**: Switch to S3 Object Lock evidence backend
2. **Continuous Monitoring**: Set up scheduled validation triggers
3. **Custom Tests**: Add domain-specific tests in `tests/custom/`
4. **GRC Integration**: Push results to OpenPages/ServiceNow
5. **Multi-Model**: Validate multiple LLM versions in parallel

## References

- mrm-core GenAI tests: `mrm/tests/builtin/genai.py`
- LLM endpoint adapters: `mrm/backends/llm_endpoints.py`
- Frouros drift detection: https://github.com/IFCA-Advanced-Computing/frouros
- Microsoft Presidio (PII): https://microsoft.github.io/presidio/
