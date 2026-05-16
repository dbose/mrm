"""Tests for the daily Merkle aggregator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mrm.evidence.chain import ChainWriter
from mrm.evidence.merkle import (
    LEAF_PREFIX,
    NODE_PREFIX,
    DailyMerkleRoot,
    aggregate_epoch,
    leaf_hash,
    merkle_root,
    node_hash,
    read_root,
    reproduce_root_from_chain,
    write_root,
)


SECRET = bytes.fromhex("aa" * 32)
EPOCH = "2026-05-01"


def _seed_chain(tmp_path: Path, n_events: int = 3, session_id: str = "s1") -> Path:
    writer = ChainWriter(
        tmp_path / "chain", session_id=session_id, chain_secret=SECRET, epoch=EPOCH
    )
    for i in range(n_events):
        writer.append("evidence_packet", {"i": i})
    return tmp_path / "chain"


# ---------------------------------------------------------------------------
# Primitive hash functions (RFC 6962)
# ---------------------------------------------------------------------------


def test_leaf_hash_uses_domain_separator():
    h = leaf_hash("ab" * 32)
    assert h == hashlib.sha256(LEAF_PREFIX + bytes.fromhex("ab" * 32)).hexdigest()


def test_node_hash_concatenates_with_node_prefix():
    left, right = "ab" * 32, "cd" * 32
    h = node_hash(left, right)
    expected = hashlib.sha256(
        NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()
    assert h == expected


def test_merkle_root_with_single_leaf():
    h = "ab" * 32
    assert merkle_root([h]) == leaf_hash(h)


def test_merkle_root_with_two_leaves():
    a, b = "ab" * 32, "cd" * 32
    expected = node_hash(leaf_hash(a), leaf_hash(b))
    assert merkle_root([a, b]) == expected


def test_merkle_root_promotes_unpaired_leaf():
    """RFC 6962: unpaired leaf is promoted, not duplicated."""
    a, b, c = "11" * 32, "22" * 32, "33" * 32
    pair = node_hash(leaf_hash(a), leaf_hash(b))
    # c is promoted unchanged at the next level.
    expected = node_hash(pair, leaf_hash(c))
    assert merkle_root([a, b, c]) == expected


def test_merkle_root_rejects_empty():
    with pytest.raises(ValueError):
        merkle_root([])


# ---------------------------------------------------------------------------
# aggregate_epoch
# ---------------------------------------------------------------------------


def test_aggregate_epoch_produces_a_root(tmp_path):
    _seed_chain(tmp_path, n_events=4)
    root = aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)
    assert root.epoch == EPOCH
    assert root.leaf_count == 4
    assert len(root.root_hash) == 64
    assert root.sessions == ["s1"]


def test_aggregate_epoch_is_deterministic(tmp_path):
    _seed_chain(tmp_path, n_events=3)
    a = aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)
    b = aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)
    assert a.root_hash == b.root_hash


def test_aggregate_epoch_refuses_corrupt_chain(tmp_path):
    _seed_chain(tmp_path, n_events=3)
    path = tmp_path / "chain" / EPOCH / "s1.jsonl"
    lines = path.read_text().splitlines()
    # Replace one line with junk.
    lines[1] = '{"event_hash": "deadbeef"}'
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(Exception):
        aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)


def test_root_writeread_roundtrip(tmp_path):
    _seed_chain(tmp_path, n_events=2)
    root = aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)
    write_root(tmp_path / "roots", root)
    loaded = read_root(tmp_path / "roots", EPOCH)
    assert loaded.root_hash == root.root_hash
    assert loaded.leaf_count == root.leaf_count


def test_reproduce_root_from_chain_matches_aggregator(tmp_path):
    _seed_chain(tmp_path, n_events=5)
    expected = aggregate_epoch(tmp_path / "chain", EPOCH, chain_secret=SECRET)
    rederived = reproduce_root_from_chain(
        tmp_path / "chain", EPOCH, chain_secret=SECRET
    )
    assert rederived == expected.root_hash


# ---------------------------------------------------------------------------
# DailyMerkleRoot.signed_bytes
# ---------------------------------------------------------------------------


def test_signed_bytes_excludes_signature_and_metadata():
    r = DailyMerkleRoot(
        epoch=EPOCH, root_hash="ab" * 32, leaf_count=1, sessions=["s1"]
    )
    base = r.signed_bytes()
    r.signature = "deadbeef"
    r.signer = "local"
    r.metadata = {"note": "ignore me"}
    r.published_at = "2026-05-02T00:00:00Z"
    assert r.signed_bytes() == base
