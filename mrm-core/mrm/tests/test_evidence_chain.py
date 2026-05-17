"""Tests for the HMAC-chained event log (P9 fast path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrm.evidence.chain import (
    ChainReader,
    ChainWriter,
    ChainedEvent,
    derive_session_key,
)


SECRET = bytes.fromhex("aa" * 32)
EPOCH = "2026-05-01"


def _writer(tmp_path: Path, session_id: str = "s1") -> ChainWriter:
    return ChainWriter(
        tmp_path / "chain", session_id=session_id, chain_secret=SECRET, epoch=EPOCH
    )


def test_first_event_has_null_prior_hash(tmp_path):
    w = _writer(tmp_path)
    ev = w.append("evidence_packet", {"x": 1})
    assert ev.prior_event_hash is None
    assert ev.event_hash and len(ev.event_hash) == 64


def test_events_form_a_link(tmp_path):
    w = _writer(tmp_path)
    e1 = w.append("evidence_packet", {"x": 1})
    e2 = w.append("evidence_packet", {"x": 2})
    e3 = w.append("evidence_packet", {"x": 3})
    assert e2.prior_event_hash == e1.event_hash
    assert e3.prior_event_hash == e2.event_hash


def test_event_hash_is_hmac_of_body(tmp_path):
    w = _writer(tmp_path)
    ev = w.append("evidence_packet", {"hello": "world"})
    session_key = derive_session_key(SECRET, "s1", EPOCH)
    assert ev.verify(session_key) is True


def test_event_hash_rejects_tamper(tmp_path):
    w = _writer(tmp_path)
    ev = w.append("evidence_packet", {"x": 1})
    session_key = derive_session_key(SECRET, "s1", EPOCH)
    # Mutate body and expect verify to fail.
    bad = ChainedEvent.from_dict({**ev.to_dict(), "payload_hash": "0" * 64})
    assert bad.verify(session_key) is False


def test_session_key_changes_with_epoch(tmp_path):
    k1 = derive_session_key(SECRET, "s1", "2026-05-01")
    k2 = derive_session_key(SECRET, "s1", "2026-05-02")
    assert k1 != k2


def test_session_key_changes_with_session_id(tmp_path):
    k1 = derive_session_key(SECRET, "s1", EPOCH)
    k2 = derive_session_key(SECRET, "s2", EPOCH)
    assert k1 != k2


def test_writer_persists_to_jsonl(tmp_path):
    w = _writer(tmp_path)
    w.append("evidence_packet", {"x": 1})
    w.append("evidence_packet", {"x": 2})
    path = tmp_path / "chain" / EPOCH / "s1.jsonl"
    assert path.exists()
    lines = [l for l in path.read_text().splitlines() if l]
    assert len(lines) == 2
    # Each line is canonical JSON.
    json.loads(lines[0])
    json.loads(lines[1])


def test_writer_resumes_from_existing_tail(tmp_path):
    w1 = _writer(tmp_path)
    e1 = w1.append("evidence_packet", {"x": 1})

    w2 = _writer(tmp_path)
    e2 = w2.append("evidence_packet", {"x": 2})
    assert e2.prior_event_hash == e1.event_hash


def test_chainreader_verifies_clean_chain(tmp_path):
    w = _writer(tmp_path)
    for i in range(5):
        w.append("evidence_packet", {"i": i})
    reader = ChainReader(tmp_path / "chain", chain_secret=SECRET)
    assert reader.verify_epoch(EPOCH) is True


def test_chainreader_detects_tampered_payload_hash(tmp_path):
    w = _writer(tmp_path)
    for i in range(3):
        w.append("evidence_packet", {"i": i})
    path = tmp_path / "chain" / EPOCH / "s1.jsonl"
    lines = path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["payload_hash"] = "0" * 64
    lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    reader = ChainReader(tmp_path / "chain", chain_secret=SECRET)
    assert reader.verify_epoch(EPOCH) is False


def test_chainreader_detects_deleted_event(tmp_path):
    w = _writer(tmp_path)
    for i in range(3):
        w.append("evidence_packet", {"i": i})
    path = tmp_path / "chain" / EPOCH / "s1.jsonl"
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    reader = ChainReader(tmp_path / "chain", chain_secret=SECRET)
    assert reader.verify_epoch(EPOCH) is False
