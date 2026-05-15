# Conformance test vectors

This directory will hold the positive- and negative-case test corpora
that implementations claim conformance against. Layout:

```
test-vectors/
├── replay/        # DecisionRecord (replay-record-v1)
├── evidence/      # EvidencePacket (evidence-vault-v1)
└── compliance/    # Compliance plugin contract (compliance-plugin-v1)
```

Each directory contains:

- `positive/` — well-formed artefacts; implementations MUST accept.
- `negative/` — malformed artefacts (tampered hashes, broken chains,
  schema-invalid fields); implementations MUST reject.
- `README.md` — one line per vector explaining what it tests.

**Status:** initial corpus ships in P9 (Cryptographic evidence vault
hardening) in [STRATEGY.md](../../../../STRATEGY.md). The directory
structure is reserved here so the specs can reference it stably from
v1.
