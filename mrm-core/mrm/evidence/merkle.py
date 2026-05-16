"""Daily Merkle aggregator (Fast Path -> Lockdown Path bridge).

At end of each UTC day the chain writer's events are aggregated into a
deterministic Merkle tree. The root is then signed by a ``Signer``
(``sign.py``) — that's the only hot path that touches the HSM in the
production architecture.

The tree construction follows the RFC 6962 (Certificate Transparency)
convention with domain-separation bytes (``0x00`` for leaves, ``0x01``
for internal nodes) so any third party — or a regulator audit script —
can re-derive the root from the events without using ``mrm-core``.

This module is intentionally small and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mrm.evidence.chain import ChainedEvent, ChainReader

logger = logging.getLogger(__name__)


LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def _canonical_json(payload: Dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# RFC-6962-style Merkle root
# ---------------------------------------------------------------------------

def leaf_hash(event_hash_hex: str) -> str:
    """``H(0x00 || bytes.fromhex(event_hash))`` — RFC 6962 leaf hash."""
    return _sha256(LEAF_PREFIX + bytes.fromhex(event_hash_hex))


def node_hash(left_hex: str, right_hex: str) -> str:
    """``H(0x01 || left || right)`` — RFC 6962 internal node."""
    return _sha256(NODE_PREFIX + bytes.fromhex(left_hex) + bytes.fromhex(right_hex))


def merkle_root(leaves: List[str]) -> str:
    """Compute the Merkle root over hex-encoded leaf hashes.

    Empty input is a defined error: regulators are unlikely to accept
    a root over zero events and we'd rather surface that explicitly.
    """
    if not leaves:
        raise ValueError("Cannot compute Merkle root over empty leaf set")
    level = [leaf_hash(h) for h in leaves]
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(node_hash(level[i], level[i + 1]))
            else:
                # RFC 6962: promote unpaired leaf (no duplication).
                nxt.append(level[i])
        level = nxt
    return level[0]


# ---------------------------------------------------------------------------
# DailyMerkleRoot artefact
# ---------------------------------------------------------------------------

@dataclass
class DailyMerkleRoot:
    """The published artefact for one UTC day.

    ``signature`` and ``signer`` are populated once the root has been
    passed through a ``Signer`` (``sign.py``). They are deliberately
    null until then so we can audit the unsigned root if needed.
    """

    epoch: str
    root_hash: str
    leaf_count: int
    sessions: List[str]
    spec_version: str = "evidence-vault-v1"
    published_at: Optional[str] = None
    signature: Optional[str] = None
    signer: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict) -> "DailyMerkleRoot":
        return cls(**data)

    def signed_bytes(self) -> bytes:
        """The exact byte string a Signer must sign / verify.

        Excludes ``signature``, ``signer``, ``published_at`` and
        ``metadata`` so resigning or relabelling doesn't break
        verification.
        """
        body = self.to_dict()
        for k in ("signature", "signer", "published_at", "metadata"):
            body.pop(k, None)
        return _canonical_json(body).encode("utf-8")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate_epoch(
    chain_dir: Path,
    epoch: str,
    chain_secret: Optional[bytes] = None,
) -> DailyMerkleRoot:
    """Build the daily Merkle root for ``epoch`` from a chain directory.

    Leaves are taken in (session_id, file-order) sorted order so the
    root is reproducible from the same files on any host.
    """
    reader = ChainReader(Path(chain_dir), chain_secret=chain_secret)
    if not reader.verify_epoch(epoch):
        raise ValueError(
            f"Chain integrity verification failed for epoch {epoch}; "
            f"refusing to publish a root."
        )

    leaves: List[str] = []
    sessions: List[str] = []
    epoch_dir = Path(chain_dir) / epoch
    for path in sorted(epoch_dir.glob("*.jsonl")):
        sessions.append(path.stem)
        for event in _iter_chain_file(path):
            leaves.append(event.event_hash)

    root = merkle_root(leaves)
    return DailyMerkleRoot(
        epoch=epoch,
        root_hash=root,
        leaf_count=len(leaves),
        sessions=sessions,
    )


def _iter_chain_file(path: Path) -> Iterable[ChainedEvent]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield ChainedEvent.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Root publication / loading
# ---------------------------------------------------------------------------

def root_path(roots_dir: Path, epoch: str) -> Path:
    return Path(roots_dir) / f"{epoch}.root.json"


def write_root(roots_dir: Path, root: DailyMerkleRoot) -> Path:
    roots_dir = Path(roots_dir)
    roots_dir.mkdir(parents=True, exist_ok=True)
    target = root_path(roots_dir, root.epoch)
    if root.published_at is None:
        root.published_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    target.write_text(root.to_json())
    return target


def read_root(roots_dir: Path, epoch: str) -> DailyMerkleRoot:
    path = root_path(Path(roots_dir), epoch)
    data = json.loads(path.read_text())
    return DailyMerkleRoot.from_dict(data)


# ---------------------------------------------------------------------------
# Verification helpers (no Signer dependency — pure data)
# ---------------------------------------------------------------------------

def reproduce_root_from_chain(
    chain_dir: Path,
    epoch: str,
    chain_secret: Optional[bytes] = None,
) -> str:
    """Re-derive the Merkle root from the chain alone.

    Auditors use this to confirm the published root matches the events
    on disk without trusting ``mrm-core`` to have done it correctly.
    """
    return aggregate_epoch(chain_dir, epoch, chain_secret=chain_secret).root_hash
