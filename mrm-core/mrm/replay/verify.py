"""Replay verification — reconstruct and diff a recorded decision.

``reconstruct(record_id, predictor)`` re-invokes ``predictor`` with the
captured input_state and compares the output to the recorded output.

``verify(record_id, predictor, tolerance)`` returns a ``ReplayDiff``
indicating whether the replay matches within the tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from mrm.replay.backends.base import ReplayBackend
from mrm.replay.record import DecisionRecord


@dataclass
class ReplayDiff:
    record_id: str
    matched: bool
    recorded_output: Any
    replayed_output: Any
    differences: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "matched": self.matched,
            "recorded_output": self.recorded_output,
            "replayed_output": self.replayed_output,
            "differences": list(self.differences),
        }


def _compare(
    recorded: Any,
    replayed: Any,
    tolerance: float,
    path: str = "$",
) -> List[str]:
    """Recursively compare with numeric tolerance. Returns difference paths."""
    diffs: List[str] = []

    if isinstance(recorded, float) or isinstance(replayed, float):
        try:
            a = float(recorded)
            b = float(replayed)
        except (TypeError, ValueError):
            diffs.append(f"{path}: type mismatch ({type(recorded).__name__} vs {type(replayed).__name__})")
            return diffs
        if math.isnan(a) and math.isnan(b):
            return diffs
        if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
            diffs.append(f"{path}: {a!r} != {b!r} (tol={tolerance})")
        return diffs

    if isinstance(recorded, (int, bool)) and isinstance(replayed, (int, bool)):
        if recorded != replayed:
            diffs.append(f"{path}: {recorded!r} != {replayed!r}")
        return diffs

    if isinstance(recorded, str) and isinstance(replayed, str):
        if recorded != replayed:
            diffs.append(f"{path}: string mismatch")
        return diffs

    if isinstance(recorded, dict) and isinstance(replayed, dict):
        all_keys = set(recorded) | set(replayed)
        for key in sorted(all_keys):
            if key not in recorded:
                diffs.append(f"{path}.{key}: missing in recorded")
            elif key not in replayed:
                diffs.append(f"{path}.{key}: missing in replayed")
            else:
                diffs.extend(_compare(recorded[key], replayed[key], tolerance, f"{path}.{key}"))
        return diffs

    if isinstance(recorded, (list, tuple)) and isinstance(replayed, (list, tuple)):
        if len(recorded) != len(replayed):
            diffs.append(f"{path}: length {len(recorded)} != {len(replayed)}")
            return diffs
        for i, (a, b) in enumerate(zip(recorded, replayed)):
            diffs.extend(_compare(a, b, tolerance, f"{path}[{i}]"))
        return diffs

    if recorded != replayed:
        diffs.append(f"{path}: {recorded!r} != {replayed!r}")
    return diffs


def reconstruct(
    record: DecisionRecord,
    predictor: Callable[..., Any],
) -> Any:
    """Re-invoke ``predictor`` with the captured input_state.

    The convention matches ``capture`` — input_state keys are passed as
    keyword arguments. If that fails (e.g. predictor takes positional
    args), input_state values are passed positionally.
    """
    inputs = record.input_state
    try:
        return predictor(**inputs)
    except TypeError:
        return predictor(*list(inputs.values()))


def verify(
    record: DecisionRecord,
    predictor: Callable[..., Any],
    tolerance: float = 1e-9,
) -> ReplayDiff:
    """Verify replay matches the recorded output within tolerance."""
    replayed = reconstruct(record, predictor)
    diffs = _compare(record.output, replayed, tolerance=tolerance)
    return ReplayDiff(
        record_id=record.record_id,
        matched=len(diffs) == 0,
        recorded_output=record.output,
        replayed_output=replayed,
        differences=diffs,
    )


def verify_chain_integrity(backend: ReplayBackend, model_name: str) -> bool:
    """Verify content_hash + prior_record_hash for every record."""
    return backend.verify_chain(model_name)


def find_record(backend: ReplayBackend, record_id: str) -> Optional[DecisionRecord]:
    return backend.get(record_id)
