"""
Bootstrap script: generates synthetic CCR data and pickles the model
so the MRM framework can load it via the standard file-based path.

Run once:
    cd ccr_example && python setup_ccr_example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from models.ccr.ccr_monte_carlo import CCRMonteCarloModel

ROOT = Path(__file__).parent


def generate_counterparty_dataset(n_counterparties: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic counterparty dataset."""
    rng = np.random.RandomState(seed)

    ratings = ["AAA", "AA", "A", "BBB", "BB", "B"]
    rating_pds = {"AAA": 0.0002, "AA": 0.0008, "A": 0.002, "BBB": 0.008, "BB": 0.03, "B": 0.08}
    rating_weights = [0.05, 0.10, 0.25, 0.30, 0.20, 0.10]

    rows = []
    for i in range(n_counterparties):
        rating = rng.choice(ratings, p=rating_weights)
        pd_annual = rating_pds[rating] * rng.uniform(0.8, 1.2)
        lgd = rng.uniform(0.35, 0.65)
        notional = rng.choice([500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000])
        fixed_rate = rng.uniform(0.01, 0.05)
        maturity_years = rng.choice([1, 2, 3, 5, 7, 10])
        pay_fixed = rng.choice([True, False])
        collateral_threshold = rng.choice([0, 0, 0, 500_000, 1_000_000])

        # Backtesting columns: realised P&L drawn from a distribution
        # correlated with model-predicted exposure
        realised_pnl = abs(rng.normal(0, notional * 0.01))

        rows.append({
            "counterparty_id": f"CP_{i+1:04d}",
            "credit_rating": rating,
            "pd_annual": round(pd_annual, 6),
            "lgd": round(lgd, 4),
            "netting_set_id": f"NS_{i+1:04d}",
            "notional": notional,
            "fixed_rate": round(fixed_rate, 4),
            "maturity_years": maturity_years,
            "pay_fixed": pay_fixed,
            "collateral_threshold": collateral_threshold,
            "realised_pnl": round(realised_pnl, 2),
            # target column: 1 = default within horizon, 0 = no default
            "defaulted": int(rng.random() < pd_annual * maturity_years),
        })

    return pd.DataFrame(rows)


def main():
    print("=== Setting up CCR example ===")

    # 1. Generate datasets
    print("Generating training dataset...")
    train_df = generate_counterparty_dataset(n_counterparties=80, seed=42)
    train_path = ROOT / "data" / "training.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  Saved {len(train_df)} counterparties to {train_path}")

    print("Generating validation dataset...")
    val_df = generate_counterparty_dataset(n_counterparties=50, seed=99)
    val_path = ROOT / "data" / "validation.csv"
    val_df.to_csv(val_path, index=False)
    print(f"  Saved {len(val_df)} counterparties to {val_path}")

    # 2. Create and pickle the CCR Monte Carlo model
    print("Creating CCR Monte Carlo model...")
    model = CCRMonteCarloModel(
        n_simulations=5000,
        n_time_steps=60,
        confidence_level=0.95,
        seed=42,
    )

    model_path = ROOT / "models" / "ccr_monte_carlo.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model to {model_path}")

    # 3. Quick sanity run
    print("Running sanity simulation on first counterparty...")
    from models.ccr.ccr_monte_carlo import CounterpartyProfile
    cp = CounterpartyProfile(
        counterparty_id="CP_0001",
        credit_rating="BBB",
        pd_annual=0.008,
        lgd=0.45,
        trades=[{
            "notional": 10_000_000,
            "fixed_rate": 0.03,
            "maturity_years": 5,
            "pay_fixed": True,
        }],
    )
    results = model.simulate(cp)
    print(f"  EPE:      {results['epe']:,.2f}")
    print(f"  EAD:      {results['ead']:,.2f}")
    print(f"  CVA:      {results['cva']:,.2f}")
    print(f"  PFE peak: {results['pfe_peak']:,.2f}")

    print("\n=== Setup complete ===")


if __name__ == "__main__":
    main()
