"""
Counterparty Credit Risk (CCR) Monte Carlo Simulation Engine

Implements a vanilla Monte Carlo simulation for computing:
- Expected Positive Exposure (EPE)
- Potential Future Exposure (PFE) at various confidence levels
- Credit Valuation Adjustment (CVA)
- Exposure at Default (EAD)
- Expected Exposure (EE) profiles over time

The model simulates mark-to-market paths for a portfolio of OTC derivatives
(interest rate swaps) under risk-neutral GBM dynamics, then aggregates
netting-set level exposures to produce regulatory CCR metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class CounterpartyProfile:
    """Describes a single counterparty and its trade portfolio."""
    counterparty_id: str
    credit_rating: str  # e.g. "AA", "A", "BBB"
    pd_annual: float  # annual probability of default
    lgd: float  # loss given default (recovery = 1 - lgd)
    netting_set_id: str = ""
    collateral_threshold: float = 0.0  # CSA threshold
    min_transfer_amount: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)


class CCRMonteCarloModel:
    """
    Vanilla Monte Carlo engine for Counterparty Credit Risk.

    Simulates mark-to-market values of interest rate swap portfolios
    using Geometric Brownian Motion for the short rate, then computes
    netting-set level positive exposure profiles.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        n_time_steps: int = 60,  # monthly steps over 5 years
        dt: float = 1 / 12,
        confidence_level: float = 0.95,
        seed: Optional[int] = 42,
    ):
        self.n_simulations = n_simulations
        self.n_time_steps = n_time_steps
        self.dt = dt
        self.confidence_level = confidence_level
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Outputs populated after simulation
        self.exposure_profiles: Optional[np.ndarray] = None  # (n_sims, n_steps)
        self.time_grid: Optional[np.ndarray] = None
        self.ee_profile: Optional[np.ndarray] = None
        self.epe: Optional[float] = None
        self.pfe_profile: Optional[np.ndarray] = None
        self.cva: Optional[float] = None
        self.ead: Optional[float] = None
        self.simulation_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(self, counterparty: CounterpartyProfile) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation for a counterparty.

        Returns a results dict suitable for downstream validation.
        """
        self.time_grid = np.arange(1, self.n_time_steps + 1) * self.dt

        # 1. Simulate short-rate paths (Vasicek-like mean-reverting process)
        rate_paths = self._simulate_rate_paths()

        # 2. Value the netting set on each path / time step
        mtm_paths = self._value_netting_set(rate_paths, counterparty.trades)

        # 3. Apply netting & collateral
        exposure_paths = self._apply_netting_and_collateral(
            mtm_paths, counterparty
        )

        # 4. Compute exposure metrics
        self.exposure_profiles = exposure_paths
        self.ee_profile = np.mean(exposure_paths, axis=0)
        self.epe = float(np.mean(self.ee_profile))
        self.pfe_profile = np.percentile(
            exposure_paths, self.confidence_level * 100, axis=0
        )
        self.ead = float(np.max(self.ee_profile)) * 1.4  # alpha multiplier

        # 5. Compute CVA
        self.cva = self._compute_cva(counterparty)

        self.simulation_metadata = {
            "n_simulations": self.n_simulations,
            "n_time_steps": self.n_time_steps,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "counterparty_id": counterparty.counterparty_id,
            "n_trades": len(counterparty.trades),
        }

        return self._build_results()

    def _simulate_rate_paths(self) -> np.ndarray:
        """
        Simulate interest-rate paths using a Vasicek model:
            dr = kappa * (theta - r) * dt + sigma * dW
        """
        kappa = 0.15  # mean-reversion speed
        theta = 0.03  # long-term mean rate
        sigma = 0.01  # volatility
        r0 = 0.025  # initial rate

        paths = np.zeros((self.n_simulations, self.n_time_steps))
        paths[:, 0] = r0

        dW = self._rng.normal(
            0, np.sqrt(self.dt), (self.n_simulations, self.n_time_steps - 1)
        )

        for t in range(1, self.n_time_steps):
            paths[:, t] = (
                paths[:, t - 1]
                + kappa * (theta - paths[:, t - 1]) * self.dt
                + sigma * dW[:, t - 1]
            )

        return paths

    def _value_netting_set(
        self, rate_paths: np.ndarray, trades: List[Dict]
    ) -> np.ndarray:
        """
        Value each trade in the netting set across all paths and time steps.
        Simplified IRS valuation: PV of fixed vs floating legs.
        """
        mtm = np.zeros((self.n_simulations, self.n_time_steps))

        for trade in trades:
            notional = trade.get("notional", 1_000_000)
            fixed_rate = trade.get("fixed_rate", 0.03)
            maturity_years = trade.get("maturity_years", 5)
            direction = 1 if trade.get("pay_fixed", True) else -1

            for t_idx in range(self.n_time_steps):
                time_t = self.time_grid[t_idx]
                remaining = max(maturity_years - time_t, 0)
                if remaining <= 0:
                    continue

                # Approximate swap PV: (floating - fixed) * annuity * notional
                floating_rate = rate_paths[:, t_idx]
                annuity = remaining  # simplified
                swap_pv = direction * (floating_rate - fixed_rate) * annuity * notional
                mtm[:, t_idx] += swap_pv

        return mtm

    def _apply_netting_and_collateral(
        self, mtm_paths: np.ndarray, counterparty: CounterpartyProfile
    ) -> np.ndarray:
        """
        Apply close-out netting (take max(V, 0)) and collateral (CSA threshold).
        """
        # Positive exposure after netting
        positive_exposure = np.maximum(mtm_paths, 0)

        # Apply collateral: exposure reduced by collateral above threshold
        if counterparty.collateral_threshold > 0:
            collateral = np.maximum(
                positive_exposure - counterparty.collateral_threshold, 0
            )
            positive_exposure = np.minimum(
                positive_exposure, counterparty.collateral_threshold
            ) + collateral * 0.1  # residual exposure from margin period of risk

        return positive_exposure

    def _compute_cva(self, counterparty: CounterpartyProfile) -> float:
        """
        Unilateral CVA = LGD * sum_t [ DiscountedEE(t) * dPD(t) ]
        """
        lgd = counterparty.lgd
        pd_annual = counterparty.pd_annual
        discount_rate = 0.025

        cva = 0.0
        survival_prob = 1.0

        for t_idx in range(self.n_time_steps):
            time_t = self.time_grid[t_idx]
            ee_t = self.ee_profile[t_idx]
            discount_factor = np.exp(-discount_rate * time_t)

            # Marginal default probability for this period
            hazard_rate = -np.log(1 - pd_annual)
            marginal_pd = survival_prob * (1 - np.exp(-hazard_rate * self.dt))
            survival_prob *= np.exp(-hazard_rate * self.dt)

            cva += lgd * discount_factor * ee_t * marginal_pd

        return float(cva)

    def _build_results(self) -> Dict[str, Any]:
        return {
            "epe": self.epe,
            "ead": self.ead,
            "cva": self.cva,
            "pfe_peak": float(np.max(self.pfe_profile)),
            "pfe_profile": self.pfe_profile.tolist(),
            "ee_profile": self.ee_profile.tolist(),
            "time_grid": self.time_grid.tolist(),
            "metadata": self.simulation_metadata,
        }

    # ------------------------------------------------------------------
    # sklearn-compatible interface for MRM framework
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        sklearn-like predict interface.

        X is expected to have columns describing counterparty parameters.
        Returns per-row EPE predictions.
        """
        results = []
        for _, row in X.iterrows():
            cp = CounterpartyProfile(
                counterparty_id=str(row.get("counterparty_id", "CP_UNKNOWN")),
                credit_rating=str(row.get("credit_rating", "BBB")),
                pd_annual=float(row.get("pd_annual", 0.02)),
                lgd=float(row.get("lgd", 0.45)),
                netting_set_id=str(row.get("netting_set_id", "NS_1")),
                collateral_threshold=float(row.get("collateral_threshold", 0)),
                trades=[
                    {
                        "notional": float(row.get("notional", 1_000_000)),
                        "fixed_rate": float(row.get("fixed_rate", 0.03)),
                        "maturity_years": float(row.get("maturity_years", 5)),
                        "pay_fixed": bool(row.get("pay_fixed", True)),
                    }
                ],
            )
            sim = self.simulate(cp)
            results.append(sim["epe"])

        return np.array(results)

    def predict_full(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return full simulation results per counterparty row."""
        results = []
        for _, row in X.iterrows():
            cp = CounterpartyProfile(
                counterparty_id=str(row.get("counterparty_id", "CP_UNKNOWN")),
                credit_rating=str(row.get("credit_rating", "BBB")),
                pd_annual=float(row.get("pd_annual", 0.02)),
                lgd=float(row.get("lgd", 0.45)),
                netting_set_id=str(row.get("netting_set_id", "NS_1")),
                collateral_threshold=float(row.get("collateral_threshold", 0)),
                trades=[
                    {
                        "notional": float(row.get("notional", 1_000_000)),
                        "fixed_rate": float(row.get("fixed_rate", 0.03)),
                        "maturity_years": float(row.get("maturity_years", 5)),
                        "pay_fixed": bool(row.get("pay_fixed", True)),
                    }
                ],
            )
            results.append(self.simulate(cp))
        return results
