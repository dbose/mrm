"""
Validation Trigger System for MRM.

Defines conditions under which model validation must be re-run.
Triggers are evaluated against model metadata, test history, and
external signals.  Supports regulatory requirements for ongoing
monitoring and timely re-validation (e.g. CPS 230 Para 34-37).

Trigger types:
  - SCHEDULED:    Calendar-based (quarterly, semi-annual, annual)
  - DRIFT:        Data/model drift exceeds threshold
  - BREACH:       Back-test breach rate exceeds limit
  - MATERIALITY:  Portfolio notional or counterparty count changes
  - REGULATORY:   Regulation or policy change mandates re-validation
  - MANUAL:       Ad-hoc trigger by model owner
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    SCHEDULED = "scheduled"
    DRIFT = "drift"
    BREACH = "breach"
    MATERIALITY = "materiality"
    REGULATORY = "regulatory"
    MANUAL = "manual"


class TriggerStatus(str, Enum):
    ACTIVE = "active"           # trigger is armed
    FIRED = "fired"             # condition met, validation needed
    ACKNOWLEDGED = "acknowledged"  # owner has seen it
    RESOLVED = "resolved"       # re-validation completed


@dataclass
class TriggerCondition:
    """A single trigger condition."""
    trigger_type: TriggerType
    description: str
    threshold: Optional[float] = None
    schedule_days: Optional[int] = None
    compliance_reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerEvent:
    """A fired trigger event with evidence."""
    trigger_id: str
    trigger_type: TriggerType
    model_name: str
    fired_at: str  # ISO timestamp
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: TriggerStatus = TriggerStatus.FIRED
    compliance_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trigger_type"] = self.trigger_type.value
        d["status"] = self.status.value
        return d


def _read_compliance_ref(cfg: Dict) -> str:
    """Read compliance reference from config, with legacy fallback."""
    return cfg.get("compliance_reference", cfg.get("cps230_reference", ""))


class ValidationTriggerEngine:
    """
    Evaluates trigger conditions and manages the trigger lifecycle.

    Usage in model YAML::

        triggers:
          - type: scheduled
            schedule_days: 90
            compliance_reference: "CPS 230 Para 34"
          - type: breach
            threshold: 0.10
            compliance_reference: "CPS 230 Para 36"
    """

    def __init__(self, trigger_store_path: Optional[Path] = None):
        self.store_path = trigger_store_path or Path.home() / ".mrm" / "triggers"
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._events: List[TriggerEvent] = []
        self._load_events()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_name: str,
        trigger_configs: List[Dict[str, Any]],
        test_results: Optional[Dict[str, Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TriggerEvent]:
        """Evaluate all trigger conditions for a model."""
        fired = []
        now = datetime.now(timezone.utc)

        for cfg in trigger_configs:
            ttype = TriggerType(cfg["type"])
            compliance_ref = _read_compliance_ref(cfg)

            if ttype == TriggerType.SCHEDULED:
                event = self._check_scheduled(
                    model_name, cfg.get("schedule_days", 90), now, compliance_ref
                )
            elif ttype == TriggerType.BREACH:
                event = self._check_breach(
                    model_name, cfg.get("threshold", 0.10),
                    test_results, compliance_ref
                )
            elif ttype == TriggerType.DRIFT:
                event = self._check_drift(
                    model_name, cfg.get("threshold", 0.15),
                    test_results, compliance_ref
                )
            elif ttype == TriggerType.MATERIALITY:
                event = self._check_materiality(
                    model_name, cfg.get("threshold", 0.20),
                    model_metadata, compliance_ref
                )
            elif ttype == TriggerType.REGULATORY:
                event = self._check_regulatory(model_name, cfg, compliance_ref)
            elif ttype == TriggerType.MANUAL:
                event = None
            else:
                event = None

            if event is not None:
                fired.append(event)
                self._events.append(event)

        if fired:
            self._save_events()

        return fired

    # ------------------------------------------------------------------
    # Individual trigger checks
    # ------------------------------------------------------------------

    def _check_scheduled(
        self, model_name: str, schedule_days: int,
        now: datetime, compliance_ref: str
    ) -> Optional[TriggerEvent]:
        last_validation = self._get_last_resolved(model_name, TriggerType.SCHEDULED)
        if last_validation:
            last_dt = datetime.fromisoformat(last_validation.fired_at)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if (now - last_dt).days < schedule_days:
                return None

        return TriggerEvent(
            trigger_id=f"SCHED-{model_name}-{now.strftime('%Y%m%d')}",
            trigger_type=TriggerType.SCHEDULED,
            model_name=model_name,
            fired_at=now.isoformat(),
            reason=f"Scheduled re-validation: {schedule_days} days since last run",
            evidence={"schedule_days": schedule_days},
            compliance_reference=compliance_ref,
        )

    def _check_breach(
        self, model_name: str, threshold: float,
        test_results: Optional[Dict], compliance_ref: str
    ) -> Optional[TriggerEvent]:
        if not test_results:
            return None

        for test_name, result in test_results.items():
            if "PFEBacktest" in test_name or "Backtest" in test_name:
                details = result.details if hasattr(result, "details") else result.get("details", {})
                breach_rate = details.get("breach_rate", 0)
                if breach_rate > threshold:
                    now = datetime.now(timezone.utc)
                    return TriggerEvent(
                        trigger_id=f"BREACH-{model_name}-{now.strftime('%Y%m%d%H%M')}",
                        trigger_type=TriggerType.BREACH,
                        model_name=model_name,
                        fired_at=now.isoformat(),
                        reason=f"PFE breach rate {breach_rate:.2%} exceeds {threshold:.0%}",
                        evidence={"breach_rate": breach_rate, "threshold": threshold},
                        compliance_reference=compliance_ref,
                    )
        return None

    def _check_drift(
        self, model_name: str, threshold: float,
        test_results: Optional[Dict], compliance_ref: str
    ) -> Optional[TriggerEvent]:
        if not test_results:
            return None

        for test_name, result in test_results.items():
            if "Convergence" in test_name:
                details = result.details if hasattr(result, "details") else result.get("details", {})
                rel_diff = details.get("relative_difference", 0)
                if rel_diff > threshold:
                    now = datetime.now(timezone.utc)
                    return TriggerEvent(
                        trigger_id=f"DRIFT-{model_name}-{now.strftime('%Y%m%d%H%M')}",
                        trigger_type=TriggerType.DRIFT,
                        model_name=model_name,
                        fired_at=now.isoformat(),
                        reason=f"MC drift {rel_diff:.4f} exceeds {threshold}",
                        evidence={"relative_difference": rel_diff, "threshold": threshold},
                        compliance_reference=compliance_ref,
                    )
        return None

    def _check_materiality(
        self, model_name: str, threshold: float,
        model_metadata: Optional[Dict], compliance_ref: str
    ) -> Optional[TriggerEvent]:
        if not model_metadata:
            return None

        pct_change = model_metadata.get("portfolio_change_pct", 0)
        if abs(pct_change) > threshold:
            now = datetime.now(timezone.utc)
            return TriggerEvent(
                trigger_id=f"MAT-{model_name}-{now.strftime('%Y%m%d%H%M')}",
                trigger_type=TriggerType.MATERIALITY,
                model_name=model_name,
                fired_at=now.isoformat(),
                reason=f"Portfolio change {pct_change:.1%} exceeds {threshold:.0%}",
                evidence={"portfolio_change_pct": pct_change, "threshold": threshold},
                compliance_reference=compliance_ref,
            )
        return None

    def _check_regulatory(
        self, model_name: str, cfg: Dict, compliance_ref: str
    ) -> Optional[TriggerEvent]:
        if cfg.get("policy_change_flag"):
            now = datetime.now(timezone.utc)
            return TriggerEvent(
                trigger_id=f"REG-{model_name}-{now.strftime('%Y%m%d%H%M')}",
                trigger_type=TriggerType.REGULATORY,
                model_name=model_name,
                fired_at=now.isoformat(),
                reason=cfg.get("description", "Regulatory change requires re-validation"),
                evidence={"policy": cfg.get("policy_name", "unknown")},
                compliance_reference=compliance_ref,
            )
        return None

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def acknowledge(self, trigger_id: str):
        for e in self._events:
            if e.trigger_id == trigger_id:
                e.status = TriggerStatus.ACKNOWLEDGED
        self._save_events()

    def resolve(self, trigger_id: str):
        for e in self._events:
            if e.trigger_id == trigger_id:
                e.status = TriggerStatus.RESOLVED
        self._save_events()

    def resolve_model(self, model_name: str):
        """Resolve all fired triggers for a model (after successful re-validation)."""
        for e in self._events:
            if e.model_name == model_name and e.status in (TriggerStatus.FIRED, TriggerStatus.ACKNOWLEDGED):
                e.status = TriggerStatus.RESOLVED
        self._save_events()

    def get_active_triggers(self, model_name: Optional[str] = None) -> List[TriggerEvent]:
        events = [e for e in self._events if e.status in (TriggerStatus.FIRED, TriggerStatus.ACKNOWLEDGED)]
        if model_name:
            events = [e for e in events if e.model_name == model_name]
        return events

    def get_all_events(self, model_name: Optional[str] = None) -> List[TriggerEvent]:
        if model_name:
            return [e for e in self._events if e.model_name == model_name]
        return list(self._events)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_last_resolved(self, model_name: str, ttype: TriggerType) -> Optional[TriggerEvent]:
        resolved = [
            e for e in self._events
            if e.model_name == model_name
            and e.trigger_type == ttype
            and e.status == TriggerStatus.RESOLVED
        ]
        if resolved:
            return max(resolved, key=lambda e: e.fired_at)
        return None

    def _save_events(self):
        path = self.store_path / "trigger_events.json"
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

    def _load_events(self):
        path = self.store_path / "trigger_events.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._events = [
                TriggerEvent(
                    trigger_id=d["trigger_id"],
                    trigger_type=TriggerType(d["trigger_type"]),
                    model_name=d["model_name"],
                    fired_at=d["fired_at"],
                    reason=d["reason"],
                    evidence=d.get("evidence", {}),
                    status=TriggerStatus(d["status"]),
                    compliance_reference=d.get(
                        "compliance_reference",
                        d.get("cps230_reference", ""),
                    ),
                )
                for d in data
            ]
