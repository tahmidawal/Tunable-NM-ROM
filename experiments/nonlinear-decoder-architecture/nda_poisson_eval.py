"""Evaluate any Poisson decoder checkpoint with its recorded architecture.

``pro_common`` deliberately rejects a checkpoint when the process environment
does not match its architecture.  Comparison jobs need to evaluate the frozen
FiLM control and a compact decoder sequentially on one GPU, so this small
launcher reads the manifest first, configures the environment, and only then
imports ``pro_colloc``.

Usage: python nda_poisson_eval.py <checkpoint.pkl> <output.json>
"""
from __future__ import annotations

import os
import pickle
import sys


def configure(checkpoint: str) -> dict:
    with open(checkpoint, "rb") as f:
        saved = pickle.load(f)
    cfg = saved["config"]
    dc = cfg.get("decoder_config", {
        "name": "film", "hidden": cfg["hidden"],
        "n_layers": cfg["n_layers"], "z_ff": cfg.get("z_ff", 0)})
    env = {
        "N": cfg["N"],
        "N_TRAIN": cfg["n_train"],
        "N_VAL": cfg["n_val"],
        "HIDDEN": cfg["hidden"],
        "N_LAYERS": cfg["n_layers"],
        "Z_FF": dc.get("z_ff", cfg.get("z_ff", 0)),
        "HARD_BC": cfg.get("hard_bc", 0),
        "DECODER_ARCH": dc.get("name", "film"),
        "FILM_GROUP_SIZE": dc.get("group_size", 8),
        "FILM_START": dc.get("film_start", 1),
        "Z_WIDTH": dc.get("z_width", 64),
        "WARP_MAX_SHIFT": dc.get("warp_max_shift", 0.15),
        "WARP_MAX_LOG_SCALE": dc.get("warp_max_log_scale", 0.25),
    }
    if "residual_scale" in dc:
        env["RESIDUAL_SCALE"] = dc["residual_scale"]
    for key, value in env.items():
        os.environ[key] = str(value)
    os.environ["PKL"] = os.path.abspath(checkpoint)
    return cfg


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: nda_poisson_eval.py <checkpoint.pkl> <output.json>")
    checkpoint, output = sys.argv[1:]
    cfg = configure(checkpoint)
    # pro_colloc reads its output path at import time.
    sys.argv = ["pro_colloc.py", output]
    import pro_colloc

    print(f"CHECKPOINT_CONFIG {cfg}", flush=True)
    pro_colloc.main()


if __name__ == "__main__":
    main()
