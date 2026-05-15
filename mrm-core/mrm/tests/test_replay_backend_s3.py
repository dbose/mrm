"""Tests for the S3ReplayBackend using an in-memory fake S3 client."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pytest

from mrm.replay.backends.s3 import S3ReplayBackend
from mrm.replay.record import DecisionRecord, ModelIdentity


class FakeS3Client:
    """A minimal in-memory stand-in for boto3 S3 used by the backend."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}
        self.put_kwargs: List[Dict[str, Any]] = []

    def put_object(self, Bucket, Key, Body, ContentType=None, **kwargs):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.put_kwargs.append({"Bucket": Bucket, "Key": Key, **kwargs})
        return {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key]

        class _Stream:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Stream(body)}

    def get_paginator(self, name):
        objects = self.objects
        class _P:
            def paginate(self, Bucket, Prefix):
                contents = [
                    {"Key": k} for k in sorted(objects) if k.startswith(Prefix)
                ]
                yield {"Contents": contents}
        return _P()


def _new_record(output):
    return DecisionRecord(
        model_identity=ModelIdentity(name="m1", version="1"),
        input_state={"x": output},
        output=output,
    )


def test_s3_append_writes_with_object_lock_retention():
    client = FakeS3Client()
    backend = S3ReplayBackend(
        bucket="evidence", prefix="replay/", retention_days=30, client=client
    )
    backend.append(_new_record(1))
    # The records put call (first one) must carry Object Lock kwargs.
    record_puts = [
        k for k in client.put_kwargs if "/records/" in k["Key"]
    ]
    assert len(record_puts) == 1
    assert record_puts[0]["ObjectLockMode"] == "COMPLIANCE"
    assert "ObjectLockRetainUntilDate" in record_puts[0]


def test_s3_chains_via_tail_pointer():
    client = FakeS3Client()
    backend = S3ReplayBackend(bucket="b", client=client)
    r1 = backend.append(_new_record(1))
    r2 = backend.append(_new_record(2))
    assert r2.prior_record_hash == r1.content_hash
    assert backend.tail("m1").record_id == r2.record_id


def test_s3_iter_model_walks_chain_in_order():
    client = FakeS3Client()
    backend = S3ReplayBackend(bucket="b", client=client)
    appended_ids = [backend.append(_new_record(i)).record_id for i in range(4)]
    walked = [r.record_id for r in backend.iter_model("m1")]
    assert walked == appended_ids


def test_s3_get_returns_record_by_id():
    client = FakeS3Client()
    backend = S3ReplayBackend(bucket="b", client=client)
    r = backend.append(_new_record(1))
    assert backend.get(r.record_id).output == 1
    assert backend.get("missing") is None


def test_s3_verify_chain_returns_true_for_clean_chain():
    client = FakeS3Client()
    backend = S3ReplayBackend(bucket="b", client=client)
    for i in range(3):
        backend.append(_new_record(i))
    assert backend.verify_chain("m1") is True
