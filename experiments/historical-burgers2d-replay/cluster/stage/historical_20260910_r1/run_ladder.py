"""Replay the archived per-mesh Burgers configurations in one GPU allocation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PY = "/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python"


def main():
    assert sys.executable == PY, sys.executable
    provenance = json.loads((HERE / "REPLAY-PROVENANCE.json").read_text())
    records = []
    shared = dict(
        K="16", R="64", EQ_M="64", EQ_MQ="256", EQ_CAND_CAP="65536",
        SEED="0", SEED0="0", TEST_SEED="1", N_TEST="8", TRAIN="0",
        TR_FACTOR="0.01", STEP_TOL="1e-9", STALL="1e-3", EXTRAP="1.0",
        IC_ENC_BUDGET="50", ENC_STEPS="12000", TIME_REPS="5", BURN="2",
        FOM_REPS="5", PAIR_REPS="3", WARM="2", ARMS="full,ex,tensor",
        NEWTON_TOLS="3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4", LIN_FRACS="0.05,0.5",
        JAX_DEFAULT_MATMUL_PRECISION="highest", JAX_ENABLE_X64="true",
        COMMIT=provenance["replay_commit"], HISTORICAL_COMMIT=provenance["historical_commit"],
    )
    for n in (64, 256, 512, 1024):
        out = HERE / "out" / f"n{n}"
        out.mkdir()
        ckpt = HERE / "in" / (f"sep_burgers_N{n}_K16_R64.pkl" if n != 512
                              else "sep_b2d_tensor_n512_ckpt.pkl")
        nodes = (HERE / "in" / ("sep_codesign_n_m64_nodes.npz" if n == 64
                                else f"sep_codesign_n_N{n}_m64_nodes.npz"))
        env = dict(os.environ, **shared, N=str(n),
                   N_TRAIN="96" if n == 1024 else "512", N_VAL="8" if n == 1024 else "64",
                   GEN_CHUNK="8" if n == 1024 else ("16" if n == 512 else "64"),
                   CKPT=str(ckpt), NODES_NPZ=str(nodes) if nodes.exists() else "",
                   CHECKPOINT_SHA256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                   OUT=str(out / f"sep_b2d_tensor_n{n}.json"),
                   CKPT_OUT=str(out / f"sep_b2d_tensor_n{n}_ckpt.pkl"))
        print(f"BEGIN N={n} checkpoint={ckpt.name}", flush=True)
        start = time.time()
        with (HERE / "logs" / f"n{n}.out").open("w") as stdout, \
                (HERE / "logs" / f"n{n}.err").open("w") as stderr:
            result = subprocess.run([PY, str(HERE / "code" / "sep_b2d_tensor.py")],
                                    cwd=HERE / "code", env=env, stdout=stdout, stderr=stderr)
        record = dict(N=n, returncode=result.returncode, seconds=time.time() - start,
                      environment={k: env[k] for k in sorted(set(shared) | {
                          "N", "N_TRAIN", "N_VAL", "GEN_CHUNK", "CKPT", "NODES_NPZ",
                          "CHECKPOINT_SHA256", "OUT", "CKPT_OUT"})})
        records.append(record)
        (HERE / "out" / "LADDER.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"END N={n} returncode={result.returncode} seconds={record['seconds']:.1f}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        artifact = json.loads(Path(env["OUT"]).read_text())
        assert artifact["complete"] and artifact["config"]["backend"] == "gpu"
        assert artifact["config"]["x64"] and artifact["config"]["matmul_precision"] == "highest"
    (HERE / "out" / "COMPLETE").write_text("Four original mesh configurations completed on one GPU allocation.\n")


if __name__ == "__main__":
    main()
