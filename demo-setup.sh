#!/usr/bin/env bash
# Demo prep: seeds artefacts inside mrm-core/ccr_example/ so the
# asciinema recording can run with bare CLI commands and a real
# project context.
#
# Idempotent. Safe to run multiple times. Takes ~5 seconds.
#
# Usage:
#   ./demo-setup.sh
#   cd mrm-core/ccr_example
#   asciinema rec --idle-time-limit=2 ~/riskattest-demo.cast

set -euo pipefail

export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export LANG="${LANG:-en_US.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/Users/dbose/projects/mrm}"
PY="${PY:-/Users/dbose/miniforge3/envs/mrm/bin/python}"

CCR_DIR="$PROJECT_ROOT/mrm-core/ccr_example"

if [ ! -f "$CCR_DIR/mrm_project.yml" ]; then
    echo "ERROR: $CCR_DIR/mrm_project.yml missing." >&2
    exit 1
fi
if [ ! -f "$CCR_DIR/models/ccr_monte_carlo.pkl" ]; then
    echo "ERROR: $CCR_DIR/models/ccr_monte_carlo.pkl missing." >&2
    echo "Run: cd $CCR_DIR && python setup_ccr_example.py" >&2
    exit 1
fi

cd "$CCR_DIR"

# ---------------------------------------------------------------------------
# Clean previous demo artefacts
# ---------------------------------------------------------------------------

DEMO_DIR="$CCR_DIR/demo-artefacts"
REPLAY_DIR="$CCR_DIR/replay"
rm -rf "$DEMO_DIR" "$REPLAY_DIR"
mkdir -p "$DEMO_DIR"/{chain,roots,reports}
mkdir -p "$REPLAY_DIR"

# Defensively delete any stray demo dirs at the project root --
# earlier iterations of this script wrote there and left orphans
# that confuse the demo alias `r='cat demo-artefacts/demo_record_id.txt'`.
rm -rf "$PROJECT_ROOT/demo-artefacts" "$PROJECT_ROOT/replay"

# Deterministic chain secret.
CHAIN_SECRET_HEX="$(printf 'demoseed%056d' 1 | shasum -a 256 | awk '{print $1}')"
EPOCH="2026-05-17"

# ---------------------------------------------------------------------------
# Write helper python scripts to /tmp -- avoids heredoc fragility.
# ---------------------------------------------------------------------------

PY_SEED_CHAIN="/tmp/riskattest_seed_chain.py"
PY_CAPTURE="/tmp/riskattest_capture.py"
PY_REPORT="/tmp/riskattest_report.py"

cat > "$PY_SEED_CHAIN" <<'PYEOF'
"""Seed a hash-chained event log + sign the daily Merkle root."""
import os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"] + "/mrm-core")
from pathlib import Path

from mrm.evidence.chain import ChainWriter
from mrm.evidence.merkle import aggregate_epoch, write_root
from mrm.evidence.sign import LocalSigner

DEMO_DIR = Path(os.environ["DEMO_DIR"])
EPOCH = os.environ["EPOCH"]
SECRET = bytes.fromhex(os.environ["CHAIN_SECRET_HEX"])

chain_dir = DEMO_DIR / "chain"
writer = ChainWriter(chain_dir, session_id="demo-session",
                     chain_secret=SECRET, epoch=EPOCH)
for i in range(4):
    writer.append("evidence_packet", {"i": i, "model": "ccr_monte_carlo"})

root = aggregate_epoch(chain_dir, EPOCH, chain_secret=SECRET)
signer = LocalSigner(DEMO_DIR / "root.key")
signed = signer.sign(root)
write_root(DEMO_DIR / "roots", signed)
print("  Chain + Merkle root seeded (root_hash={}...)".format(signed.root_hash[:16]))
PYEOF

cat > "$PY_CAPTURE" <<'PYEOF'
"""Capture five DecisionRecords against the real CCR Monte Carlo pickle."""
import os, sys, pickle
sys.path.insert(0, os.environ["PROJECT_ROOT"] + "/mrm-core")
# Also put ccr_example/ on sys.path so the pickle's `models.ccr.*` module
# resolves on unpickling.
sys.path.insert(0, os.environ["CCR_DIR"])

import pandas as pd
from pathlib import Path

from mrm.replay.backends.local import LocalReplayBackend
from mrm.replay.capture import capture
from mrm.replay.record import ModelIdentity

CCR_DIR = Path(os.environ["CCR_DIR"])
DEMO_DIR = Path(os.environ["DEMO_DIR"])
REPLAY_DIR = Path(os.environ["REPLAY_DIR"])

with open(CCR_DIR / "models" / "ccr_monte_carlo.pkl", "rb") as f:
    artifact = pickle.load(f)

backend = LocalReplayBackend(REPLAY_DIR, warn_on_use=False)
identity = ModelIdentity(
    name="ccr_monte_carlo",
    version="1.0.0",
    framework="numpy",
    uri="models/ccr_monte_carlo.pkl",
)

sample_df = pd.read_csv(CCR_DIR / "data" / "validation.csv").head(3)

@capture(backend=backend, model_identity=identity)
def predict(features_records):
    df = pd.DataFrame(features_records)
    return artifact.predict(df).tolist()

for i in range(5):
    row = sample_df.iloc[[i % len(sample_df)]]
    predict(row.to_dict(orient="records"))

records = list(backend.iter_model("ccr_monte_carlo"))
print("  Captured {} DecisionRecords in {}".format(len(records), REPLAY_DIR))
print("  Demo record_id: {}".format(records[-1].record_id))
(DEMO_DIR / "demo_record_id.txt").write_text(records[-1].record_id)
PYEOF

cat > "$PY_REPORT" <<'PYEOF'
"""Generate a real SR 26-2 compliance report for the demo's final beat."""
import os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"] + "/mrm-core")
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict

from mrm.compliance.registry import compliance_registry
from mrm.compliance.report_generator import generate_compliance_report

DEMO_DIR = Path(os.environ["DEMO_DIR"])

@dataclass
class TR:
    passed: bool
    score: Any = None
    details: Dict = field(default_factory=dict)
    failure_reason: Any = None

compliance_registry.load_builtin_standards()

results = {
    "ccr.MCConvergence":          TR(True, 0.9991),
    "ccr.EPEReasonableness":      TR(True, 0.98),
    "ccr.PFEBacktest":            TR(True, 0.94),
    "ccr.CVASensitivity":         TR(True, 0.92),
    "ccr.WrongWayRisk":           TR(True, 0.88),
    "ccr.ExposureProfileShape":   TR(True, 0.95),
    "compliance.GovernanceCheck": TR(True, 1.0),
}
ccr_model = {"model": {
    "name": "ccr_monte_carlo", "version": "1.4.0",
    "owner": "ccr-validation", "risk_tier": "tier_1",
    "ai_materiality": "high",
    "use_case": "Counterparty credit risk exposure",
    "methodology": "Monte Carlo simulation",
    "validation_frequency": "quarterly",
    "replay_backend": "local",
    "evidence_backend": "local + Merkle root",
}}

generate_compliance_report(
    standard_name="sr26_2",
    model_name="ccr_monte_carlo",
    model_config=ccr_model,
    test_results=results,
    output_path=DEMO_DIR / "reports" / "ccr_monte_carlo_sr26_2.md",
)
print("  SR 26-2 report generated")
PYEOF

# ---------------------------------------------------------------------------
# Execute, passing context via env vars (cleanest cross-platform contract).
# ---------------------------------------------------------------------------

export PROJECT_ROOT CCR_DIR DEMO_DIR REPLAY_DIR CHAIN_SECRET_HEX EPOCH

"$PY" "$PY_SEED_CHAIN"
"$PY" "$PY_CAPTURE"
"$PY" "$PY_REPORT"

# ---------------------------------------------------------------------------
# Cheatsheet
# ---------------------------------------------------------------------------

RECORD_ID="$(cat $DEMO_DIR/demo_record_id.txt)"

cat <<EOF

============================================================
Demo prep complete.
============================================================

  Run the demo from:   $CCR_DIR
  RECORD_ID            $RECORD_ID
  REPORT               $DEMO_DIR/reports/ccr_monte_carlo_sr26_2.md

For the recording:

  cd $CCR_DIR
  alias r='cat demo-artefacts/demo_record_id.txt'
  alias verify-root='mrm evidence root verify --date $EPOCH \\
    --chain-dir demo-artefacts/chain \\
    --roots-dir demo-artefacts/roots \\
    --signer local --key-path demo-artefacts/root.key'

  clear
  asciinema rec --idle-time-limit=2 ~/riskattest-demo.cast

  # Then follow demo-script.md beat by beat.
  # Ctrl-D to end the recording.

============================================================
EOF
