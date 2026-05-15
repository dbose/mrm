"""OTLP-compatible exporter for DecisionRecords.

Emits records as OTLP/HTTP-JSON log payloads. Any OpenTelemetry
collector with the OTLP/HTTP receiver enabled can ingest these without
additional configuration.

We deliberately avoid pulling in the ``opentelemetry-*`` SDK as a hard
dependency. Banks frequently audit transitive deps; the JSON payload
shape is stable and well-documented in the OTLP spec.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from mrm.replay.record import DecisionRecord

logger = logging.getLogger(__name__)


def _ts_to_unix_ns(iso_ts: str) -> int:
    """ISO-8601 (UTC, trailing Z) -> unix nanoseconds.

    Falls back to current time if parsing fails.
    """
    from datetime import datetime, timezone

    try:
        clean = iso_ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return int(time.time() * 1_000_000_000)


def record_to_otlp_log(record: DecisionRecord) -> Dict[str, Any]:
    """Convert a DecisionRecord to one OTLP/HTTP-JSON log record.

    The body is the full record JSON; attributes hoist the fields a
    bank's SIEM would want to filter on without parsing the body.
    """
    ts_ns = _ts_to_unix_ns(record.timestamp)
    attrs = {
        "mrm.replay.record_id": record.record_id,
        "mrm.replay.schema_version": record.schema_version,
        "mrm.model.name": record.model_identity.name,
        "mrm.model.version": record.model_identity.version,
        "mrm.replay.content_hash": record.content_hash or "",
        "mrm.replay.prior_record_hash": record.prior_record_hash or "",
    }
    if record.model_identity.provider:
        attrs["mrm.model.provider"] = record.model_identity.provider
    if record.model_identity.artifact_hash:
        attrs["mrm.model.artifact_hash"] = record.model_identity.artifact_hash

    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "mrm-core"}},
                        {"key": "service.namespace", "value": {"stringValue": "mrm.replay"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "mrm.replay", "version": record.schema_version},
                        "logRecords": [
                            {
                                "timeUnixNano": str(ts_ns),
                                "observedTimeUnixNano": str(ts_ns),
                                "severityText": "INFO",
                                "body": {"stringValue": record.to_json()},
                                "attributes": [
                                    {"key": k, "value": {"stringValue": str(v)}}
                                    for k, v in attrs.items()
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


class OTLPExporter:
    """Exports DecisionRecords to an OTLP/HTTP collector.

    Args:
        endpoint: full URL of the OTLP/HTTP logs endpoint, e.g.
            ``http://otel-collector:4318/v1/logs``.
        headers: optional auth/tenant headers.
        sender: optional callable for tests; defaults to urllib.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        sender: Optional[Callable[[str, bytes, Dict[str, str]], None]] = None,
        timeout: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout
        self.sender = sender or self._default_sender
        self.last_payload: Optional[Dict[str, Any]] = None

    def _default_sender(self, endpoint: str, body: bytes, headers: Dict[str, str]) -> None:
        import urllib.request

        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"OTLP exporter got HTTP {resp.status}")

    def export(self, record: DecisionRecord) -> Dict[str, Any]:
        payload = record_to_otlp_log(record)
        self.last_payload = payload
        body = json.dumps(payload).encode("utf-8")
        try:
            self.sender(self.endpoint, body, self.headers)
        except Exception as exc:
            logger.warning("OTLP export failed: %s", exc)
        return payload

    def export_batch(self, records: List[DecisionRecord]) -> None:
        for record in records:
            self.export(record)
