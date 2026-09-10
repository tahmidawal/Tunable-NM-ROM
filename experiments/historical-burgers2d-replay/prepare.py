"""Build an isolated, hash-manifested payload; does not submit anything."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = "/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907"


def main():
    attempt = sys.argv[1]
    assert attempt.startswith("historical_") and attempt.replace("_", "").isalnum()
    stage = HERE / "cluster" / "stage" / attempt
    stage.mkdir(parents=True)
    for directory in ("code", "in"):
        shutil.copytree(HERE / directory, stage / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_ladder.py", "HISTORICAL-SOURCES.json"):
        shutil.copy2(HERE / name, stage / name)
    (stage / "logs").mkdir()
    (stage / "out").mkdir()
    commit = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    historical = json.loads((HERE / "HISTORICAL-SOURCES.json").read_text())
    provenance = dict(replay_commit=commit, historical_commit=historical["original_commit"],
                      attempt=attempt, historical_sources=historical,
                      source_files={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((HERE / "code").rglob("*.py"))})
    (stage / "REPLAY-PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    remote = f"{BASE}/{attempt}"
    batch = f'''#!/bin/bash
#SBATCH --job-name=mr_b2d_historical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1
cd "$TASK_ROOT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}', flush=True); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c MANIFEST.sha256 --quiet
"$PY" run_ladder.py
echo ALL-DONE
'''
    (stage / "run.sbatch").write_text(batch)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(stage)}"
             for p in sorted(stage.rglob("*")) if p.is_file()]
    (stage / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(stage=str(stage), remote=remote, commit=commit, manifest_files=len(lines))))


if __name__ == "__main__":
    main()
