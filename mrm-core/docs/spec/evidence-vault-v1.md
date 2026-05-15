# Evidence Vault Chain Specification — v1

- **Status:** Public Review (PRD-2)
- **Version:** v1
- **Last updated:** 2026-05-15
- **Reference implementation:** [`mrm/evidence/packet.py`](../../mrm/evidence/packet.py)

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

## 7. Conformance

Test vectors are published under
[`test-vectors/evidence/`](test-vectors/evidence/).

## 8. Changelog

| Version | Date | Change |
|---|---|---|
| v1 (PRD-2) | 2026-05-15 | Initial public review draft. |
