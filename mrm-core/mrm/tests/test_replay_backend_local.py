"""Tests for the LocalReplayBackend JSONL hash-chain store."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.record import DecisionRecord, ModelIdentity


def _new_record(output, prior_hash=None):
    return DecisionRecord(
        model_identity=ModelIdentity(name="m1", version="1"),
        input_state={"x": output},
        output=output,
        prior_record_hash=prior_hash,
    )


def test_append_first_record_has_no_prior(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    record = backend.append(_new_record(output=1))
    assert record.prior_record_hash is None
    assert record.content_hash is not None


def test_append_chains_records_via_prior_hash(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    r1 = backend.append(_new_record(output=1))
    r2 = backend.append(_new_record(output=2))
    r3 = backend.append(_new_record(output=3))
    assert r2.prior_record_hash == r1.content_hash
    assert r3.prior_record_hash == r2.content_hash


def test_tail_returns_last_record(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    assert backend.tail("m1") is None
    r1 = backend.append(_new_record(output=1))
    assert backend.tail("m1").record_id == r1.record_id
    r2 = backend.append(_new_record(output=2))
    assert backend.tail("m1").record_id == r2.record_id


def test_iter_model_preserves_append_order(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    ids = [backend.append(_new_record(output=i)).record_id for i in range(5)]
    assert [r.record_id for r in backend.iter_model("m1")] == ids


def test_get_finds_record_across_models(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    r_a = backend.append(
        DecisionRecord(
            model_identity=ModelIdentity(name="a", version="1"),
            input_state={},
            output=1,
        )
    )
    r_b = backend.append(
        DecisionRecord(
            model_identity=ModelIdentity(name="b", version="1"),
            input_state={},
            output=2,
        )
    )
    assert backend.get(r_a.record_id).model_identity.name == "a"
    assert backend.get(r_b.record_id).model_identity.name == "b"
    assert backend.get("missing-id") is None


def test_verify_chain_walks_every_link(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    for i in range(4):
        backend.append(_new_record(output=i))
    assert backend.verify_chain("m1") is True


def test_verify_chain_detects_tampered_file(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    backend.append(_new_record(output=1))
    backend.append(_new_record(output=2))
    records_file = tmp_path / "m1" / "records.jsonl"
    text = records_file.read_text()
    tampered = text.replace('"output":2', '"output":999')
    records_file.write_text(tampered)
    assert backend.verify_chain("m1") is False


def test_sample_applies_filters(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    r1 = backend.append(_new_record(output=1))
    r2 = backend.append(_new_record(output=2))
    backend.append(_new_record(output=3))
    sampled = backend.sample(model_name="m1", n=2)
    assert len(sampled) == 2
    assert sampled[0].record_id == r1.record_id
    assert sampled[1].record_id == r2.record_id


def test_sample_respects_time_window(tmp_path: Path):
    backend = LocalReplayBackend(tmp_path, warn_on_use=False)
    backend.append(_new_record(output=1))
    backend.append(_new_record(output=2))
    future = "2999-01-01T00:00:00Z"
    assert backend.sample(model_name="m1", since=future) == []
