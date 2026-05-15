"""S3 + Object Lock replay backend (production).

Layout in the bucket:
    {prefix}/{model_name}/records/{record_id}.json   # one record/object
    {prefix}/{model_name}/tail.json                  # latest record_id pointer

Records are written with Object Lock in COMPLIANCE mode using the
configured retention. The tail pointer is a normal mutable object
(rewritten on each append) — it is purely a hint; the canonical chain
is reconstructed by walking ``prior_record_hash``.

Requires ``boto3``; the import is deferred so the OSS install stays
slim.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional

from mrm.replay.backends.base import ReplayBackend
from mrm.replay.record import DecisionRecord

logger = logging.getLogger(__name__)


class S3ReplayBackend(ReplayBackend):
    """S3 + Object Lock replay backend.

    Args:
        bucket: target bucket (must have Object Lock enabled).
        prefix: key prefix, e.g. ``replay/``.
        retention_days: Object Lock retention in days.
        mode: ``COMPLIANCE`` (default) or ``GOVERNANCE``.
        client: optional pre-built boto3 S3 client (for tests).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "replay/",
        retention_days: int = 2555,
        mode: str = "COMPLIANCE",
        client: Optional[object] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.retention_days = retention_days
        self.mode = mode
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "S3ReplayBackend requires boto3: pip install boto3"
                ) from exc
            client = boto3.client("s3")
        self.client = client

    def _record_key(self, model_name: str, record_id: str) -> str:
        return f"{self.prefix}{model_name}/records/{record_id}.json"

    def _tail_key(self, model_name: str) -> str:
        return f"{self.prefix}{model_name}/tail.json"

    def append(self, record: DecisionRecord) -> DecisionRecord:
        model_name = record.model_identity.name
        tail = self.tail(model_name)
        prior_hash = tail.content_hash if tail else None

        record_dict = record.model_dump()
        record_dict["prior_record_hash"] = prior_hash
        record_dict["content_hash"] = None
        finalised = DecisionRecord(**record_dict)

        retain_until = datetime.now(timezone.utc) + timedelta(days=self.retention_days)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._record_key(model_name, finalised.record_id),
            Body=finalised.to_json().encode("utf-8"),
            ContentType="application/json",
            ObjectLockMode=self.mode,
            ObjectLockRetainUntilDate=retain_until,
        )
        # Tail pointer is mutable; safe to overwrite.
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._tail_key(model_name),
            Body=json.dumps(
                {
                    "record_id": finalised.record_id,
                    "content_hash": finalised.content_hash,
                }
            ).encode("utf-8"),
            ContentType="application/json",
        )
        return finalised

    def get(self, record_id: str) -> Optional[DecisionRecord]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(f"/{record_id}.json"):
                    body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                    return DecisionRecord.from_json(body.decode("utf-8"))
        return None

    def tail(self, model_name: str) -> Optional[DecisionRecord]:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self._tail_key(model_name))
        except Exception:
            return None
        pointer = json.loads(resp["Body"].read())
        record_id = pointer.get("record_id")
        if not record_id:
            return None
        body = self.client.get_object(
            Bucket=self.bucket, Key=self._record_key(model_name, record_id)
        )["Body"].read()
        return DecisionRecord.from_json(body.decode("utf-8"))

    def iter_model(self, model_name: str) -> Iterator[DecisionRecord]:
        records: List[DecisionRecord] = []
        paginator = self.client.get_paginator("list_objects_v2")
        prefix = f"{self.prefix}{model_name}/records/"
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                body = self.client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                records.append(DecisionRecord.from_json(body.decode("utf-8")))
        # Reconstruct chain order by walking prior_record_hash.
        by_hash = {r.content_hash: r for r in records}
        first = next((r for r in records if r.prior_record_hash is None), None)
        if first is None:
            yield from sorted(records, key=lambda r: r.timestamp)
            return
        cur: Optional[DecisionRecord] = first
        seen = set()
        while cur is not None and cur.record_id not in seen:
            seen.add(cur.record_id)
            yield cur
            cur = next(
                (r for r in records if r.prior_record_hash == cur.content_hash),
                None,
            )

    def sample(
        self,
        model_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[DecisionRecord]:
        out: List[DecisionRecord] = []
        prefix = f"{self.prefix}{model_name}/records/" if model_name else self.prefix
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                if not obj["Key"].endswith(".json") or obj["Key"].endswith("/tail.json"):
                    continue
                body = self.client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                record = DecisionRecord.from_json(body.decode("utf-8"))
                if since and record.timestamp < since:
                    continue
                if until and record.timestamp > until:
                    continue
                out.append(record)
                if n is not None and len(out) >= n:
                    return out
        return out
