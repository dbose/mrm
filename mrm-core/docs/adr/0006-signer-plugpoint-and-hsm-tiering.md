# ADR-0006: Signer plug-point and HSM tier line

- **Status:** Accepted
- **Date:** 2026-05-16
- **Deciders:** core maintainers

## Context

Banks under SR 26-2, CPS 230 amendments, and EU AI Act enforcement
need a **defensible cryptographic chain-of-custody** over their model
evidence. The hash-chained `EvidencePacket` from
[ADR-0002](0002-evidence-vault-hash-chain.md) is necessary but not
sufficient: regulators also ask *"who signed the root?"* and the
answer has to be a hardware-protected, FIPS-validated path for
Tier-1 institutions.

We surveyed the production pattern banks expect:

| Layer | What it does | Frequency | Touches HSM? |
|---|---|---|---|
| **Fast path** | App servers HMAC-chain every event | per inference | no |
| **Aggregator** | Build Merkle tree per UTC day | once per day | no |
| **Lockdown** | Sign root via cloud HSM / KMS | once per day | yes |

Banks reject any design that puts HSM round-trips on the hot path
(latency, cost, ops complexity). So the daily Merkle root is the only
artefact that hits hardware.

We also surveyed comparable open-core / commercial splits:

- **dbt Cloud / dbt Core.** Core is Apache 2.0; Cloud sells orchestration,
  web UI, SSO, audit log. Used as the canonical "open + commercial"
  model in this repo (see [ADR-0005](0005-oss-vs-cloud-split.md)).
- **HashiCorp Vault.** Cloud-KMS auto-unseal (AWS KMS / GCP KMS / Azure
  Key Vault) is in the open-source edition; HSM auto-unseal is
  Enterprise-only. This is the precise tier line we want for signers.

## Decision

`mrm-core` ships a `Signer` ABC with a small registry. Each
implementation declares whether it `requires_hsm`. The OSS surface
includes the four signers an unpaid user needs:

1. `LocalSigner` — HMAC-SHA256 over a root key file (dev / air-gapped).
2. `GpgSigner` — detached GPG signature.
3. `AgeSigner` — `age` binary for users without GPG.
4. `KmsSigner` — cloud KMS envelope-sign (AWS today; GCP and Azure
   follow the same plug-point). Software-protected keys.

The `CloudHsmSigner` plug-point exists in the OSS surface but raises
`NotImplementedError` if instantiated. The full implementation
(against AWS CloudHSM / GCP Cloud HSM / Azure Dedicated HSM, FIPS
140-2 Level 3+) ships in `<brand>` Cloud.

## Consequences

- **Easier:** the architecture banks expect (fast path / aggregator /
  HSM-only lockdown) is the architecture we ship.
- **Easier:** the OSS / paid line is clear and survives diligence
  scrutiny — software-protected sealing free, hardware-protected
  sealing paid, exactly as HashiCorp did.
- **Easier:** third-party integrators can subclass `CloudHsmSigner`
  to bind to on-prem HSMs (Thales Luna, Entrust nShield) without
  forking `mrm-core`.
- **Trade-off accepted:** the OSS `KmsSigner` only ships AWS today.
  GCP KMS and Azure Key Vault are one-file additions; deferred until
  the first design partner asks.
- **Trade-off accepted:** `LocalSigner` uses HMAC, not asymmetric
  signing. That's deliberately a regulator-shaped artefact: HMAC is
  faster and the regulator can audit the chain with the same secret.
  Institutions that want public-key verifiability use `GpgSigner`,
  `KmsSigner`, or upgrade to `CloudHsmSigner`.

## Alternatives considered

- **Bundle a full HSM implementation in OSS.** Rejected. HSM client
  libraries (AWS CloudHSM PKCS#11, Google Cloud HSM SDK) are non-trivial
  install footprints, and the value to a bank is the *managed* HSM —
  i.e. the paid-tier offering.
- **Treat KMS as paid-tier.** Rejected. HashiCorp Vault tried this in
  2018 and reversed course; cloud KMS is now table-stakes for OSS
  crypto tooling.
- **Outsource signing to an external tool (cosign, sigstore).**
  Rejected for the daily-root use case. Sigstore is excellent for
  artefact signing but doesn't fit the *time-anchored, single-root
  per-day* model expected by financial regulators.

## References

- [`mrm/evidence/chain.py`](../../mrm/evidence/chain.py)
- [`mrm/evidence/merkle.py`](../../mrm/evidence/merkle.py)
- [`mrm/evidence/sign.py`](../../mrm/evidence/sign.py)
- [docs/spec/evidence-vault-v1.md §7-§9](../spec/evidence-vault-v1.md)
- [docs/spec/test-vectors/evidence/](../spec/test-vectors/evidence/)
- HashiCorp Vault auto-unseal:
  https://developer.hashicorp.com/vault/docs/concepts/seal
- AWS CloudHSM audit-log architecture:
  https://docs.aws.amazon.com/cloudhsm/latest/userguide/cloudhsm-audit-log-reference.html
- [ADR-0002](0002-evidence-vault-hash-chain.md), [ADR-0005](0005-oss-vs-cloud-split.md)
