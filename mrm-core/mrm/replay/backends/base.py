"""Abstract base class for replay backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from mrm.replay.record import DecisionRecord


class ReplayBackend(ABC):
    """A store of hash-chained DecisionRecords, append-only per model."""

    @abstractmethod
    def append(self, record: DecisionRecord) -> DecisionRecord:
        """Append a record. Implementations must set prior_record_hash
        based on the current tail for ``record.model_identity.name``
        BEFORE the record's content_hash is finalised."""

    @abstractmethod
    def get(self, record_id: str) -> Optional[DecisionRecord]:
        """Return a record by ID, or None if not found."""

    @abstractmethod
    def tail(self, model_name: str) -> Optional[DecisionRecord]:
        """Return the most recent record for a model, or None."""

    @abstractmethod
    def iter_model(self, model_name: str) -> Iterator[DecisionRecord]:
        """Iterate records for a model in append order."""

    @abstractmethod
    def sample(
        self,
        model_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[DecisionRecord]:
        """Return records matching the filter, capped at ``n``."""

    def verify_chain(self, model_name: str) -> bool:
        """Walk a model's chain and verify every link + content hash."""
        prior: Optional[DecisionRecord] = None
        for record in self.iter_model(model_name):
            if not record.verify_hash():
                return False
            if not record.verify_chain(prior):
                return False
            prior = record
        return True
