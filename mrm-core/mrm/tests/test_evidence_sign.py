"""Tests for the pluggable Signer abstraction (P9 lockdown path)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from mrm.evidence.merkle import DailyMerkleRoot
from mrm.evidence.sign import (
    CloudHsmSigner,
    KmsSigner,
    LocalSigner,
    build_signer,
    get_signer_cls,
    list_signers,
)


def _root() -> DailyMerkleRoot:
    return DailyMerkleRoot(
        epoch="2026-05-01",
        root_hash="ab" * 32,
        leaf_count=3,
        sessions=["s1"],
    )


# ---------------------------------------------------------------------------
# LocalSigner
# ---------------------------------------------------------------------------


def test_local_signer_round_trip(tmp_path):
    signer = LocalSigner(tmp_path / "root.key")
    root = signer.sign(_root())
    assert root.signature
    assert root.signer == "local"
    assert signer.verify(root) is True


def test_local_signer_rejects_tampered_root(tmp_path):
    signer = LocalSigner(tmp_path / "root.key")
    root = signer.sign(_root())
    root.root_hash = "ff" * 32
    assert signer.verify(root) is False


def test_local_signer_rejects_swapped_signature(tmp_path):
    a = LocalSigner(tmp_path / "a.key")
    b = LocalSigner(tmp_path / "b.key")
    signed_by_a = a.sign(_root())
    # Signer B refuses A's signature.
    assert b.verify(signed_by_a) is False


def test_local_signer_persists_key_with_strict_permissions(tmp_path):
    key_path = tmp_path / "root.key"
    LocalSigner(key_path)
    import os, stat
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


def test_registry_lists_oss_signers():
    names = set(list_signers().keys())
    # OSS signers
    assert {"local", "kms"}.issubset(names)
    # paid stub is registered but flagged
    assert "cloud-hsm" in names
    assert list_signers()["cloud-hsm"]["requires_hsm"] is True
    assert list_signers()["local"]["requires_hsm"] is False


def test_get_signer_cls_returns_class():
    assert get_signer_cls("local") is LocalSigner


def test_build_signer_factory(tmp_path):
    signer = build_signer({"name": "local", "key_path": str(tmp_path / "k")})
    assert isinstance(signer, LocalSigner)


def test_build_signer_rejects_missing_name():
    with pytest.raises(ValueError):
        build_signer({})


def test_build_signer_rejects_unknown_signer():
    with pytest.raises(KeyError):
        build_signer({"name": "nope"})


# ---------------------------------------------------------------------------
# CloudHsmSigner (paid stub)
# ---------------------------------------------------------------------------


def test_cloud_hsm_signer_raises_paid_tier_message():
    with pytest.raises(NotImplementedError, match="paid tier"):
        CloudHsmSigner()


def test_cloud_hsm_signer_flagged_requires_hsm():
    assert CloudHsmSigner.requires_hsm is True


# ---------------------------------------------------------------------------
# KmsSigner -- exercised against a fake AWS KMS client
# ---------------------------------------------------------------------------


class _FakeKmsClient:
    """A boto3-shaped stand-in for AWS KMS."""

    def __init__(self):
        self.sign_calls = []
        self.verify_calls = []

    def sign(self, KeyId, Message, MessageType, SigningAlgorithm):
        # Deterministic "signature": SHA-256 of message prefixed by key id.
        import hashlib
        sig = hashlib.sha256(KeyId.encode() + Message).digest()
        self.sign_calls.append(
            {"KeyId": KeyId, "MessageType": MessageType, "Alg": SigningAlgorithm}
        )
        return {"Signature": sig}

    def verify(self, KeyId, Message, MessageType, Signature, SigningAlgorithm):
        import hashlib
        expected = hashlib.sha256(KeyId.encode() + Message).digest()
        ok = Signature == expected
        self.verify_calls.append({"KeyId": KeyId, "ok": ok})
        return {"SignatureValid": ok}


def test_kms_signer_round_trip_with_fake_client():
    client = _FakeKmsClient()
    signer = KmsSigner("aws-kms://us-east-1/alias/mrm-root", client=client)
    root = signer.sign(_root())
    assert root.signer == "kms"
    assert root.signature
    # Stored as base64 so signature survives canonical JSON.
    base64.b64decode(root.signature)
    assert signer.verify(root) is True
    # And one of each verb hit the fake client.
    assert client.sign_calls and client.verify_calls


def test_kms_signer_rejects_tampered_root():
    client = _FakeKmsClient()
    signer = KmsSigner("aws-kms://us-east-1/alias/mrm-root", client=client)
    root = signer.sign(_root())
    root.leaf_count = 999
    assert signer.verify(root) is False


def test_kms_signer_rejects_unknown_provider():
    signer = KmsSigner("gcp-kms://projects/x/locations/global", client=None)
    with pytest.raises(NotImplementedError):
        signer.sign(_root())
