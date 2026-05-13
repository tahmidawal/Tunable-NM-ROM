"""
generate_sweep.py
-----------------
Build per-run directories under sweeps/ for the Poisson-2D Pareto sweep.

Sweep design (per resolution):
  baseline + one-knob-at-a-time variants, EQ on (mode=nnls) for the main
  family and EQ off (mode=full) for companion baselines.

Strategy: fresh training per run (no cross-run checkpoint sharing for
safety). If a given sweep becomes expensive, we can add checkpoint
reuse later via a fingerprint match.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SWEEPS = ROOT / 'sweeps'
SWEEPS.mkdir(exist_ok=True)


def base_cfg(N, k_dim, rank, num_epochs, n_train, n_val):
    return dict(
        N=N,
        k_dim=k_dim,
        rank=rank,
        hidden_dim=256,
        patch_size=8 if N <= 128 else 16,
        embed_dim=64,
        num_heads=4,
        num_enc_layers=4,
        num_epochs=num_epochs,
        batch_size=32,
        peak_lr=1e-3,
        weight_decay=5e-4,
        n_train=n_train,
        n_val=n_val,
        max_iters=8,
        gn_tol=1e-6,
        cg_tol=1e-3,
        cg_iters=0,
        eq_mode='nnls',
        n_eq_snaps=80,
        min_eq_points=300,
        seed=42,
    )


def variants_for(res_name, base):
    """Return list of (run_label, override_dict)."""
    V = []

    # ---- Baselines ----
    V.append((f"{res_name}_base_eq",   dict()))                              # EQ-on baseline
    V.append((f"{res_name}_base_full", dict(eq_mode='full')))                # non-EQ baseline

    # ---- max_iters sweep (solver knob; nnls) ----
    for mi in [4, 6, 10]:
        V.append((f"{res_name}_maxiters{mi}_eq", dict(max_iters=mi)))

    # ---- CP rank sweep ----
    for R in [128, 256, 768]:
        V.append((f"{res_name}_rank{R}_eq", dict(rank=R)))

    # ---- latent dim sweep ----
    for k in [4, 8, 16]:
        if k == base['k_dim']:
            continue
        V.append((f"{res_name}_k{k}_eq", dict(k_dim=k)))

    # ---- CG tol sweep ----
    for ct in [1e-2, 1e-4]:
        tag = f"cgtol{ct:.0e}".replace('-0','-').replace('+0','')
        V.append((f"{res_name}_{tag}_eq", dict(cg_tol=ct)))

    # ---- encoder depth ----
    V.append((f"{res_name}_enc6_eq", dict(num_enc_layers=6)))

    # ---- EQ sample count sweep (min_eq_points) ----
    for mp in [100, 500, 1000]:
        V.append((f"{res_name}_eqmin{mp}", dict(min_eq_points=mp)))

    return V


def main():
    # Per-resolution defaults match the existing baselines in results.tsv
    resolutions = [
        ("res64",  dict(N=64,  k_dim=8,  rank=512, num_epochs=100_000, n_train=700, n_val=140)),
        ("res128", dict(N=128, k_dim=12, rank=512, num_epochs=100_000, n_train=700, n_val=140)),
        ("res256", dict(N=256, k_dim=16, rank=512, num_epochs=80_000,  n_train=700, n_val=140)),
    ]

    all_runs = []
    for res_name, kw in resolutions:
        base = base_cfg(**kw)
        # res256 uses patch_size 16
        for label, override in variants_for(res_name, base):
            cfg = dict(base)
            cfg.update(override)
            cfg['label'] = label
            run_dir = SWEEPS / label
            run_dir.mkdir(exist_ok=True)
            with open(run_dir / 'config.json', 'w') as f:
                json.dump(cfg, f, indent=2)
            all_runs.append(label)

    print(f"Generated {len(all_runs)} sweep configs under {SWEEPS}")
    for r in all_runs:
        print(f"  {r}")

    # Manifest
    with open(SWEEPS / 'manifest.txt', 'w') as f:
        for r in all_runs:
            f.write(r + '\n')


if __name__ == '__main__':
    main()
