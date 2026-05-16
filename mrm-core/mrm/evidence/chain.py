"""HMAC-chained event log (Fast Path).

This is the *fast path* of the cryptographic evidence vault. It runs
entirely in software on application servers; the HSM is only involved
once a day, when the Merkle root is signed (see ``merkle.py`` and
``sign.py``).

Design goals
------------

1. **Speed.** Capturing an event is one HMAC-SHA256 over a canonical
   JSON encoding plus a single fsync. No network calls.
2. **Session-scoped keys.** Each session has its own HMAC key, derived
   from the long-lived chain secret. Compromise of a session key
   reveals only that session.
3. **Daily rotation.** The session key is rotated on every UTC date
   boundary by re-deriving from the chain secret with a fresh epoch
   label.
4. **Append-only.** Events are written as JSONL; rewriting a past line
   breaks the HMAC of every subsequent line.

The output of this layer is a stream of ``ChainedEvent`` JSON lines.
At UTC midnight, ``merkle.py`` aggregates the day's events into a
Merkle tree whose root is then signed (``sign.py``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical encoding (must match the rest of the evidence/replay specs)
# ---------------------------------------------------------------------------

def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_today_label() -> str:
    """YYYY-MM-DD in UTC. Used as the daily key epoch."""
    return datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def derive_session_key(chain_secret: bytes, session_id: str, epoch: str) -> bytes:
    """Derive a per-session, per-day HMAC key.

    ``chain_secret`` is the long-lived secret (never leaves the host).
    ``epoch`` is the UTC date (YYYY-MM-DD); ``session_id`` is a random
    identifier per producer process.

    The construction is HMAC-SHA256 with the chain secret as key and
    ``f"mrm-core/v1/session/{epoch}/{session_id}"`` as data. This is a
    standard NIST SP 800-108 KDF pattern in HMAC mode.
    """
    label = f"mrm-core/v1/session/{epoch}/{session_id}".encode("utf-8")
    return hmac.new(chain_secret, label, hashlib.sha256).digest()


def load_or_create_chain_secret(secret_path: Path) -> bytes:
    """Load the long-lived chain secret, or create one with 256 bits
    of entropy if missing. The file is created mode 0600.
    """
    secret_path = Path(secret_path)
    if secret_path.exists():
        data = secret_path.read_bytes().strip()
        if len(data) >= 32:
            return bytes.fromhex(data.decode("ascii")) if all(
                c in b"0123456789abcdefABCDEF" for c in data
            ) else data
        # Fall through to regeneration if the file is malformed.
    secret = secrets.token_bytes(32)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically with 0600 permissions.
    tmp = secret_path.with_suffix(secret_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(secret.hex().encode("ascii"))
    os.chmod(tmp, 0o600)
    os.replace(tmp, secret_path)
    return secret


# ---------------------------------------------------------------------------
# ChainedEvent
# ---------------------------------------------------------------------------

@dataclass
class ChainedEvent:
    """One entry in the fast-path event log.

    ``event_hash`` is HMAC(session_key, canonical_json(body)). Body
    includes ``prior_event_hash``, so any retroactive edit to a past
    event breaks the HMAC of every subsequent event.
    """

    event_id: str
    timestamp: str
    session_id: str
    epoch: str                       # UTC date this event belongs to
    event_type: str                  # "evidence_packet" | "decision_record" | ...
    payload_hash: str                # sha256(canonical_json(payload))
    prior_event_hash: Optional[str]
    event_hash: str                  # HMAC(session_key, body without event_hash)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_key: bytes,
        session_id: str,
        epoch: str,
        event_type: str,
        payload_hash: str,
        prior_event_hash: Optional[str],
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ChainedEvent":
        if event_id is None:
            event_id = secrets.token_hex(16)
        body = {
            "event_id": event_id,
            "timestamp": _utc_now_iso(),
            "session_id": session_id,
            "epoch": epoch,
            "event_type": event_type,
            "payload_hash": payload_hash,
            "prior_event_hash": prior_event_hash,
            "metadata": metadata or {},
        }
        h = hmac.new(session_key, _canonical_json(body).encode("utf-8"), hashlib.sha256)
        return cls(event_hash=h.hexdigest(), **body)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChainedEvent":
        return cls(**data)

    def verify(self, session_key: bytes) -> bool:
        """Recompute the HMAC and constant-time compare."""
        body = self.to_dict()
        expected = body.pop("event_hash")
        h = hmac.new(session_key, _canonical_json(body).encode("utf-8"), hashlib.sha256)
        return hmac.compare_digest(h.hexdigest(), expected)


# ---------------------------------------------------------------------------
# ChainWriter — the producer
# ---------------------------------------------------------------------------

class ChainWriter:
    """Append-only writer for one session.

    Layout::

        chain_dir/
            {epoch}/{session_id}.jsonl   # the event log
            chain.secret                 # long-lived secret (0600)

    Each line is a canonical-JSON ``ChainedEvent``. The writer keeps the
    tail hash in memory so the chain links correctly across appends in
    one process.
    """

    def __init__(
        self,
        chain_dir: Path,
        session_id: Optional[str] = None,
        chain_secret: Optional[bytes] = None,
        epoch: Optional[str] = None,
    ) -> None:
        """Create a chain writer.

        Args:
            chain_dir: Root directory for chain files.
            session_id: Optional explicit session id (random if None).
            chain_secret: Optional explicit chain secret (loaded from
                ``chain_dir/chain.secret`` if None).
            epoch: Optional explicit UTC epoch (YYYY-MM-DD). When set,
                the writer pins to this epoch and **does not auto-
                rotate**. Used for deterministic test vector generation
                and replay backfill. In normal operation leave as None.
        """
        self.chain_dir = Path(chain_dir)
        self.chain_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or secrets.token_hex(8)
        self._pinned_epoch = epoch is not None
        self.epoch = epoch or _utc_today_label()
        secret_path = self.chain_dir / "chain.secret"
        if chain_secret is not None:
            # Caller supplied a secret. Persist it to disk if no secret
            # file exists yet, so verifiers / aggregators in the same
            # chain_dir can recover it. Never overwrite an existing one.
            self._chain_secret = chain_secret
            if not secret_path.exists():
                secret_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = secret_path.with_suffix(secret_path.suffix + ".tmp")
                with open(tmp, "wb") as fh:
                    fh.write(chain_secret.hex().encode("ascii"))
                os.chmod(tmp, 0o600)
                os.replace(tmp, secret_path)
        else:
            self._chain_secret = load_or_create_chain_secret(secret_path)
        self._session_key = derive_session_key(
            self._chain_secret, self.session_id, self.epoch
        )
        self._tail_hash: Optional[str] = self._load_tail()

    # ----- public API --------------------------------------------------

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChainedEvent:
        """Hash the payload, build a ChainedEvent, fsync to disk."""
        self._maybe_rotate()
        payload_hash = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        event = ChainedEvent.create(
            session_key=self._session_key,
            session_id=self.session_id,
            epoch=self.epoch,
            event_type=event_type,
            payload_hash=payload_hash,
            prior_event_hash=self._tail_hash,
            metadata=metadata,
        )
        path = self._log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(event.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._tail_hash = event.event_hash
        return event

    def session_id_str(self) -> str:
        return self.session_id

    def current_epoch(self) -> str:
        return self.epoch

    # ----- internals ---------------------------------------------------

    def _log_path(self) -> Path:
        return self.chain_dir / self.epoch / f"{self.session_id}.jsonl"

    def _load_tail(self) -> Optional[str]:
        path = self._log_path()
        if not path.exists() or path.stat().st_size == 0:
            return None
        last = ""
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return None
        try:
            return ChainedEvent.from_dict(json.loads(last)).event_hash
        except Exception:
            return None

    def _maybe_rotate(self) -> None:
        """Roll over to a new epoch + session_key when the UTC day flips."""
        if self._pinned_epoch:
            return  # explicit epoch pin -- never rotate.
        now = _utc_today_label()
        if now != self.epoch:
            logger.info("Rotating chain key: %s -> %s", self.epoch, now)
            self.epoch = now
            self._session_key = derive_session_key(
                self._chain_secret, self.session_id, self.epoch
            )
            self._tail_hash = None  # new epoch -> new chain head


# ---------------------------------------------------------------------------
# ChainReader — verifier
# ---------------------------------------------------------------------------

class ChainReader:
    """Reads and verifies a chain directory.

    Verification is the canonical, regulator-relevant operation. We
    walk every JSONL file under ``{epoch}/`` and check:

      1. Each event's HMAC verifies under its session key.
      2. ``prior_event_hash`` for each event equals the prior event's
         ``event_hash`` (per session).
    """

    def __init__(self, chain_dir: Path, chain_secret: Optional[bytes] = None) -> None:
        self.chain_dir = Path(chain_dir)
        secret_path = self.chain_dir / "chain.secret"
        if chain_secret is None:
            if not secret_path.exists():
                raise FileNotFoundError(
                    f"Chain secret missing at {secret_path}; cannot verify."
                )
            chain_secret = load_or_create_chain_secret(secret_path)
        self._chain_secret = chain_secret

    def iter_epoch(self, epoch: str) -> Iterator[ChainedEvent]:
        for path in sorted((self.chain_dir / epoch).glob("*.jsonl")):
            yield from self._iter_file(path)

    def list_epochs(self) -> List[str]:
        return sorted(p.name for p in self.chain_dir.iterdir() if p.is_dir())

    def verify_epoch(self, epoch: str) -> bool:
        """Verify every session log under ``epoch/``."""
        for path in sorted((self.chain_dir / epoch).glob("*.jsonl")):
            session_id = path.stem
            session_key = derive_session_key(self._chain_secret, session_id, epoch)
            prior: Optional[str] = None
            for event in self._iter_file(path):
                if not event.verify(session_key):
                    logger.warning("HMAC failed on event %s", event.event_id)
                    return False
                if event.prior_event_hash != prior:
                    logger.warning(
                        "Chain link broken at %s (expected prior=%s, got=%s)",
                        event.event_id, prior, event.prior_event_hash,
                    )
                    return False
                prior = event.event_hash
        return True

    def _iter_file(self, path: Path) -> Iterator[ChainedEvent]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield ChainedEvent.from_dict(json.loads(line))
