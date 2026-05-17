"""Tests for the OTLP exporter."""

from __future__ import annotations

import json

import pytest

from mrm.replay.otlp import OTLPExporter, record_to_otlp_log
from mrm.replay.record import DecisionRecord, ModelIdentity


def _make_record():
    return DecisionRecord(
        model_identity=ModelIdentity(name="m", version="1", provider="openai"),
        input_state={"prompt": "hi"},
        output="hello",
    )


def test_record_to_otlp_log_shape():
    record = _make_record()
    payload = record_to_otlp_log(record)
    assert "resourceLogs" in payload
    log = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert log["severityText"] == "INFO"
    body = json.loads(log["body"]["stringValue"])
    assert body["record_id"] == record.record_id
    attrs = {a["key"]: a["value"]["stringValue"] for a in log["attributes"]}
    assert attrs["mrm.model.name"] == "m"
    assert attrs["mrm.model.provider"] == "openai"
    assert attrs["mrm.replay.content_hash"] == record.content_hash


def test_otlp_log_timestamp_is_unix_nanos():
    record = _make_record()
    payload = record_to_otlp_log(record)
    log = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    ts = int(log["timeUnixNano"])
    # 2020-01-01 in unix-nanos, roughly.
    assert ts > 1_577_836_800_000_000_000


def test_exporter_invokes_sender_with_json_body():
    record = _make_record()
    calls = []

    def sender(endpoint, body, headers):
        calls.append((endpoint, body, headers))

    exporter = OTLPExporter(
        endpoint="http://collector:4318/v1/logs",
        headers={"Authorization": "Bearer x"},
        sender=sender,
    )
    payload = exporter.export(record)
    assert len(calls) == 1
    endpoint, body, headers = calls[0]
    assert endpoint == "http://collector:4318/v1/logs"
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer x"
    decoded = json.loads(body.decode("utf-8"))
    assert decoded == payload


def test_exporter_swallows_sender_errors():
    """Replay must not break the host application if the collector is down."""
    record = _make_record()

    def bad_sender(*_):
        raise RuntimeError("collector unreachable")

    exporter = OTLPExporter(endpoint="http://x", sender=bad_sender)
    exporter.export(record)  # should not raise


def test_export_batch_processes_all_records():
    sent = []

    def sender(endpoint, body, headers):
        sent.append(body)

    exporter = OTLPExporter(endpoint="http://x", sender=sender)
    records = [
        DecisionRecord(
            model_identity=ModelIdentity(name="m", version="1"),
            input_state={"i": i},
            output=i,
        )
        for i in range(3)
    ]
    exporter.export_batch(records)
    assert len(sent) == 3
