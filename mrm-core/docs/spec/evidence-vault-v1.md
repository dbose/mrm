# Evidence Vault Chain Specification — v1

- **Status:** Public Review (PRD-2)
- **Version:** v1
- **Last updated:** 2026-05-16
- **Reference implementation:** [`mrm/evidence/`](../../mrm/evidence/)
  (`packet.py`, `chain.py`, `merkle.py`, `sign.py`)

## 1. Scope

This specification defines the wire format and integrity semantics of
an `mrm-core` **EvidencePacket** and the **hash chain** they form.

Conformance keywords follow RFC 2119.

## 2. EvidencePacket

An EvidencePacket is a JSON object with the following fields.

| Field | Type | Required |
|---|---|---|
| `packet_id` | string (UUIDv4) | yes |
| `model_name` | string | yes |
| `model_version` | string | yes |
| `model_artifact_hash` | string (hex SHA-256) | yes |
| `test_results` | object | yes |
| `compliance_mappings` | object (standard → list of paragraphs) | yes |
| `timestamp` | string (RFC 3339, UTC, trailing `Z`) | yes |
| `created_by` | string | yes |
| `prior_packet_hash` | string (hex SHA-256) \| null | yes (may be null) |
| `content_hash` | string (hex SHA-256) | yes |
| `signature` | string \| null | no |
| `metadata` | object | yes (may be empty) |

For LLM endpoints, see [`replay-record-v1.md`](replay-record-v1.md)
§3 for `model_artifact_hash` derivation.

## 3. Hash semantics

### 3.1 `content_hash`

```
sha256( canonical_json(packet_without_content_hash_or_signature) )
```

The signature field is excluded so signing does not invalidate the
hash. Canonical JSON rules match those in
[`replay-record-v1.md`](replay-record-v1.md) §5.1.

### 3.2 `prior_packet_hash`

For the first packet for a `(model_name)` chain, `prior_packet_hash`
MUST be `null`. Every subsequent packet MUST set
`prior_packet_hash` to the immediately preceding packet's
`content_hash`.

### 3.3 Signature

The `signature` field, if present, MUST be a detached signature over
`content_hash`. OSS implementations SHOULD support GPG and `age`.
Commercial implementations MAY use HSM-backed signatures (e.g. FIPS
140-2 Level 3+).

## 4. Storage substrates

The canonical chain-of-custody is the hash chain. Storage substrate
is independent. Substrates that already carry SEC 17a-4 / FINRA /
CFTC assessments (S3 Object Lock COMPLIANCE mode, Azure Immutable
Blob, on-prem WORM) SHOULD be preferred for production. Local
filesystem MUST be labelled as development-only.

## 5. Retention

For US bank deployments, the retention window MUST default to **2555
days (7 years)** to satisfy typical regulator expectations. The
window is configurable; implementations MUST refuse to delete packets
inside their retention window.

## 6. Verification

A packet is valid iff:

1. `content_hash` equals the recomputed hash per §3.1.
2. `prior_packet_hash` is consistent with the preceding packet
   (§3.2).
3. If `signature` is present, it MUST verify against the configured
   public key for the chain.

## 7. Fast-path event log (HMAC-chained)

Implementations SHOULD also expose a *fast-path* event log so every
inference or validation event can be captured at application-server
speed without an HSM round-trip. The reference implementation lives
in [`mrm/evidence/chain.py`](../../mrm/evidence/chain.py).

### 7.1 ChainedEvent

A ChainedEvent is a JSON object with the following fields:

| Field | Type | Required |
|---|---|---|
| `event_id` | string (hex) | yes |
| `timestamp` | RFC 3339, UTC `Z` | yes |
| `session_id` | string | yes |
| `epoch` | UTC date `YYYY-MM-DD` | yes |
| `event_type` | string | yes |
| `payload_hash` | hex SHA-256 of canonical-JSON payload | yes |
| `prior_event_hash` | hex \| null | yes |
| `event_hash` | hex HMAC-SHA256 over body | yes |
| `metadata` | object | yes (may be empty) |

### 7.2 Session keys

Session keys are derived from a long-lived **chain secret** as

```
session_key = HMAC-SHA256(chain_secret, "mrm-core/v1/session/{epoch}/{session_id}")
```

This is a NIST SP 800-108 HMAC-KDF construction. The chain secret
file MUST be created mode 0600 and contain at least 256 bits of
entropy.

### 7.3 `event_hash`

```
event_hash = HMAC-SHA256(session_key, canonical_json(event_body))
```

where `event_body` is the ChainedEvent JSON object with the
`event_hash` field removed. Canonical JSON rules match §3.1.

### 7.4 Verification

A chain is **valid** for one `epoch` iff, for every session log under
that epoch's directory, walking the file from top to bottom: (a) each
event's `event_hash` recomputes correctly, and (b) each event's
`prior_event_hash` equals the immediately-preceding event's
`event_hash` (or is `null` for the first event in the session).

## 8. Daily Merkle root (lockdown path)

Implementations SHOULD aggregate the day's chained events into a
deterministic Merkle tree once per UTC day and sign the root with a
configured `Signer`. Reference implementation in
[`mrm/evidence/merkle.py`](../../mrm/evidence/merkle.py).

### 8.1 Hash construction

Per RFC 6962 (Certificate Transparency):

```
leaf_hash(e)  = SHA-256(0x00 || bytes.fromhex(e.event_hash))
node_hash(l,r) = SHA-256(0x01 || bytes.fromhex(l) || bytes.fromhex(r))
```

An unpaired leaf at the right edge of a level is **promoted**
unchanged (RFC 6962 §2.1); it MUST NOT be duplicated.

### 8.2 Leaf ordering

Leaves are taken in `(session_id ascending, in-file order)`. The
ordering is deterministic so any auditor can re-derive the root from
the events.

### 8.3 `DailyMerkleRoot`

| Field | Type | Required |
|---|---|---|
| `epoch` | UTC date `YYYY-MM-DD` | yes |
| `root_hash` | hex SHA-256 | yes |
| `leaf_count` | int | yes |
| `sessions` | list of session_id strings | yes |
| `spec_version` | string (`"evidence-vault-v1"`) | yes |
| `published_at` | RFC 3339, UTC `Z` | populated when signed |
| `signature` | string | populated by Signer |
| `signer` | short name of the signer (`"local"`, `"gpg"`, `"age"`, `"kms"`, `"cloud-hsm"`) | populated by Signer |
| `metadata` | object | may be empty |

### 8.4 Signed bytes

The exact byte string a `Signer` signs / verifies is
`canonical_json(root)` with `signature`, `signer`, `published_at`, and
`metadata` removed. Resigning or relabelling MUST NOT invalidate
existing signatures.

## 9. Signers

| Name | OSS / paid | Notes |
|---|---|---|
| `local` | OSS | HMAC-SHA256 over a long-lived root key (file, mode 0600). Hash-equivalent to GPG for audit, no external deps. |
| `gpg` | OSS | Detached GPG signature; requires `python-gnupg`. |
| `age` | OSS | `age` binary (Filippo Sottile); for users without GPG infrastructure. |
| `kms` | OSS | Cloud KMS envelope sign (AWS KMS today; GCP KMS, Azure Key Vault follow the same plug-point). Software-protected keys managed by the cloud provider. |
| `cloud-hsm` | **paid** | FIPS 140-2 Level 3+ HSM-backed signing (AWS CloudHSM, GCP Cloud HSM, Azure Dedicated HSM). The plug-point is in the OSS surface; the implementation ships in `<brand>` Cloud. |

This split deliberately mirrors HashiCorp Vault's open-source vs.
enterprise tiering: cloud-KMS sealing is free; dedicated HSM is the
enterprise edition.

## 10. Conformance

Test vectors are published under
[`test-vectors/evidence/`](test-vectors/evidence/). The directory
contains both positive and negative corpora; conforming
implementations MUST accept every positive vector and reject every
negative one. Run them via:

```bash
mrm evidence conformance run
```

## 11. Changelog

| Version | Date | Change |
|---|---|---|
| v1 (PRD-2) | 2026-05-16 | Added §7-§9 (HMAC chain, Merkle root, signers). |
| v1 (PRD-2) | 2026-05-15 | Initial public review draft. |
