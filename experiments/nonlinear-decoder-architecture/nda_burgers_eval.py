"""Evaluate a Burgers decoder checkpoint with its recorded architecture.

The shared Burgers module binds its architecture at import time.  This launcher
loads the checkpoint manifest first and exports the matching settings before it
imports ``blat_rom``.  It is used by same-GPU trust-region validation jobs.

Usage: python nda_burgers_eval.py <checkpoint.pkl> <output-directory>
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
        "name": "film", "hidden": cfg["ad_hidden"],
        "n_layers": cfg["ad_layers"], "z_ff": 0})
    env = {
        "N": cfg["N"],
        "N_TRAIN": cfg["n_train"],
        "N_VAL": cfg["n_val"],
        "K_LAT": saved["k_lat"],
        "AD_HIDDEN": cfg["ad_hidden"],
        "AD_LAYERS": cfg["ad_layers"],
        "BC_MODE": cfg["bc_mode"],
        "GN_BUDGET": cfg.get("gn_budget", 30),
        "GN_TOL": cfg.get("gn_tol", 1e-9),
        "IC_BUDGET": cfg.get("ic_budget", 100),
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
    return cfg


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: nda_burgers_eval.py <checkpoint.pkl> <output-directory>")
    checkpoint, output_dir = sys.argv[1:]
    cfg = configure(checkpoint)
    # blat_rom binds its two positional paths at import time.
    sys.argv = ["blat_rom.py", os.path.abspath(checkpoint), os.path.abspath(output_dir)]
    import blat_rom

    print(f"CHECKPOINT_CONFIG {cfg}", flush=True)
    blat_rom.main()


if __name__ == "__main__":
    main()
