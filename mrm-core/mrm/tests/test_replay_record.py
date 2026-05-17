"""Tests for the DecisionRecord schema and hash semantics."""

from __future__ import annotations

import json

import pytest

from mrm.replay.record import DecisionRecord, InferenceParams, ModelIdentity


def _make_record(prior_hash=None, output=42):
    return DecisionRecord(
        model_identity=ModelIdentity(name="m", version="1"),
        input_state={"x": 1, "y": [1, 2, 3]},
        inference_params=InferenceParams(seed=7, temperature=0.0),
        output=output,
        prior_record_hash=prior_hash,
    )


def test_content_hash_is_deterministic():
    r1 = _make_record()
    r2 = DecisionRecord(
        record_id=r1.record_id,
        timestamp=r1.timestamp,
        model_identity=r1.model_identity,
        input_state=r1.input_state,
        inference_params=r1.inference_params,
        output=r1.output,
        prior_record_hash=r1.prior_record_hash,
    )
    assert r1.content_hash == r2.content_hash


def test_content_hash_changes_when_output_changes():
    r1 = _make_record(output=1)
    r2 = _make_record(output=2)
    assert r1.content_hash != r2.content_hash


def test_verify_hash_round_trip():
    r = _make_record()
    serialised = r.to_json()
    restored = DecisionRecord.from_json(serialised)
    assert restored.content_hash == r.content_hash
    assert restored.verify_hash() is True


def test_verify_hash_detects_tamper():
    r = _make_record()
    blob = json.loads(r.to_json())
    blob["output"] = 999
    tampered = DecisionRecord(**blob)
    # Newly constructed record will compute a fresh content_hash that
    # differs from the embedded one — so we force the OLD hash to test
    # tamper detection.
    object.__setattr__(tampered, "content_hash", r.content_hash)
    assert tampered.verify_hash() is False


def test_verify_chain_first_record_has_no_prior():
    r = _make_record(prior_hash=None)
    assert r.verify_chain(None) is True


def test_verify_chain_links_to_prior_content_hash():
    r1 = _make_record()
    r2 = _make_record(prior_hash=r1.content_hash, output="next")
    assert r2.verify_chain(r1) is True


def test_verify_chain_detects_broken_link():
    r1 = _make_record()
    r2 = _make_record(prior_hash="0" * 64, output="next")
    assert r2.verify_chain(r1) is False


def test_inputs_hash_is_stable_regardless_of_key_order():
    a = DecisionRecord.hash_inputs({"a": 1, "b": 2})
    b = DecisionRecord.hash_inputs({"b": 2, "a": 1})
    assert a == b


def test_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        DecisionRecord(
            model_identity=ModelIdentity(name="m", version="1"),
            input_state={},
            output=None,
            not_a_field="boom",
        )
