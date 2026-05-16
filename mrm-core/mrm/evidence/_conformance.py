"""Evidence-vault conformance test runner.

Walks ``docs/spec/test-vectors/evidence/`` (or a user-supplied
directory) and runs each vector. Used by both ``mrm evidence
conformance run`` and the pytest suite.

Vector layout::

    docs/spec/test-vectors/evidence/
        positive/
            <vector-name>/
                meta.json            # {"kind": "...", "summary": "..."}
                chain.jsonl          # the events
                chain.secret         # hex chain secret (testing only)
                root.json            # expected DailyMerkleRoot (unsigned)
        negative/
            <vector-name>/
                meta.json            # {"kind": "...", "reason": "..."}
                chain.jsonl
                chain.secret
                # no root.json — should fail to verify

For each positive vector we require:
  - chain verification succeeds
  - aggregator output's ``root_hash`` equals the expected root_hash

For each negative vector we require:
  - chain verification FAILS (the corruption must be detectable)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from mrm.evidence.chain import ChainReader, load_or_create_chain_secret
from mrm.evidence.merkle import aggregate_epoch


def _default_vectors_dir() -> Path:
    # docs/spec/test-vectors/evidence relative to the package root.
    pkg_root = Path(__file__).parent.parent.parent
    return pkg_root / "docs" / "spec" / "test-vectors" / "evidence"


@dataclass
class VectorResult:
    name: str
    passed: bool
    summary: str


def _run_positive(vector_dir: Path) -> VectorResult:
    meta = json.loads((vector_dir / "meta.json").read_text())
    epoch = meta["epoch"]
    summary = meta.get("summary", "")
    expected_root = meta.get("expected_root_hash")

    try:
        secret = bytes.fromhex(
            (vector_dir / "chain.secret").read_text().strip()
        )
    except Exception as exc:
        return VectorResult(vector_dir.name, False, f"bad chain.secret: {exc}")

    chain_root_dir = vector_dir / "chain"
    # Verify the underlying chain.
    reader = ChainReader(chain_root_dir, chain_secret=secret)
    if not reader.verify_epoch(epoch):
        return VectorResult(vector_dir.name, False, "chain verification failed")

    try:
        aggregated = aggregate_epoch(chain_root_dir, epoch=epoch, chain_secret=secret)
    except Exception as exc:
        return VectorResult(vector_dir.name, False, f"aggregator raised: {exc}")

    if expected_root and aggregated.root_hash != expected_root:
        return VectorResult(
            vector_dir.name,
            False,
            f"root mismatch (got {aggregated.root_hash}, expected {expected_root})",
        )
    return VectorResult(vector_dir.name, True, summary or "positive vector verified")


def _run_negative(vector_dir: Path) -> VectorResult:
    meta = json.loads((vector_dir / "meta.json").read_text())
    epoch = meta["epoch"]
    reason = meta.get("reason", "")
    try:
        secret = bytes.fromhex(
            (vector_dir / "chain.secret").read_text().strip()
        )
    except Exception as exc:
        return VectorResult(vector_dir.name, False, f"bad chain.secret: {exc}")

    chain_root_dir = vector_dir / "chain"
    reader = ChainReader(chain_root_dir, chain_secret=secret)
    # Negative vectors MUST fail verification.
    if reader.verify_epoch(epoch):
        return VectorResult(
            vector_dir.name,
            False,
            f"negative vector ({reason}) should have failed verification but passed",
        )
    return VectorResult(vector_dir.name, True, f"correctly rejected: {reason}")


def run_all(vectors_dir: Optional[str] = None) -> Dict:
    """Run every positive + negative vector. Returns an aggregate dict."""
    base = Path(vectors_dir) if vectors_dir else _default_vectors_dir()
    if not base.exists():
        return {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "details": [],
            "warning": f"No vectors directory at {base}",
        }

    details: List[VectorResult] = []
    for pos in sorted((base / "positive").glob("*/")) if (base / "positive").exists() else []:
        details.append(_run_positive(pos))
    for neg in sorted((base / "negative").glob("*/")) if (base / "negative").exists() else []:
        details.append(_run_negative(neg))

    passed = sum(1 for d in details if d.passed)
    failed = sum(1 for d in details if not d.passed)
    return {
        "passed": passed,
        "failed": failed,
        "total": len(details),
        "details": [{"name": d.name, "passed": d.passed, "summary": d.summary} for d in details],
    }
