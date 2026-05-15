"""Local JSONL hash-chain replay backend (DEV ONLY).

Layout:
    replay_dir/
        {model_name}/
            records.jsonl   # append-only, one DecisionRecord per line
            index.json      # record_id -> byte offset (lazy-built)

NOT FOR REGULATED PRODUCTION USE. Files are mutable on disk. Use the
S3 + Object Lock backend for production.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

from mrm.replay.backends.base import ReplayBackend
from mrm.replay.record import DecisionRecord

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_dirname(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned or "_unnamed_"


class LocalReplayBackend(ReplayBackend):
    """JSONL-on-disk replay store. Dev-only.

    Per-model files are append-only; index is rebuilt on demand.
    """

    def __init__(self, replay_dir: Path, warn_on_use: bool = True) -> None:
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        if warn_on_use:
            logger.warning(
                "LocalReplayBackend is for development only - files are mutable "
                "and provide no regulatory chain-of-custody guarantees."
            )

    def _model_dir(self, model_name: str) -> Path:
        path = self.replay_dir / _safe_dirname(model_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _records_file(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "records.jsonl"

    def append(self, record: DecisionRecord) -> DecisionRecord:
        model_name = record.model_identity.name
        tail = self.tail(model_name)
        prior_hash = tail.content_hash if tail else None

        # Rebuild content_hash now that prior_record_hash is set.
        record_dict = record.model_dump()
        record_dict["prior_record_hash"] = prior_hash
        record_dict["content_hash"] = None
        finalised = DecisionRecord(**record_dict)

        with self._records_file(model_name).open("a", encoding="utf-8") as fh:
            fh.write(finalised.to_json() + "\n")
        return finalised

    def get(self, record_id: str) -> Optional[DecisionRecord]:
        for path in self.replay_dir.glob("*/records.jsonl"):
            for record in self._iter_file(path):
                if record.record_id == record_id:
                    return record
        return None

    def tail(self, model_name: str) -> Optional[DecisionRecord]:
        path = self._records_file(model_name)
        if not path.exists() or path.stat().st_size == 0:
            return None
        last_line = ""
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        return DecisionRecord.from_json(last_line)

    def iter_model(self, model_name: str) -> Iterator[DecisionRecord]:
        path = self._records_file(model_name)
        if not path.exists():
            return
        yield from self._iter_file(path)

    def _iter_file(self, path: Path) -> Iterator[DecisionRecord]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield DecisionRecord.from_json(line)
                except Exception as exc:
                    logger.warning("Skipping malformed record in %s: %s", path, exc)

    def sample(
        self,
        model_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[DecisionRecord]:
        out: List[DecisionRecord] = []
        if model_name:
            paths = [self._records_file(model_name)]
        else:
            paths = list(self.replay_dir.glob("*/records.jsonl"))
        for path in paths:
            if not path.exists():
                continue
            for record in self._iter_file(path):
                if since and record.timestamp < since:
                    continue
                if until and record.timestamp > until:
                    continue
                out.append(record)
                if n is not None and len(out) >= n:
                    return out
        return out
