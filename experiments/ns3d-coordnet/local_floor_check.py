"""Local recomputation (GB10, small) of a bank job's 32^3 development floors from the pulled
bank file, with an independent NumPy centring/shift/projection (no diag_floor code).

The bank is re-sampled from the stored network parameters and T with the same mesh_bank
(the object under test is the floor number, not the sampler). Truth is regenerated from
the development seed with the parent FOM. Writes results/local_floor_check_<job>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder",
              "experiments/ns3d-grok", "experiments/ns3d-shift"):
    sys.path.insert(0, str(ROOT / extra))
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import ns3d_fom as F  # noqa: E402
import bankio as BI  # noqa: E402


def centroid_np(u):
    w = np.sum(u * u, axis=0)
    n = u.shape[-1]
    ang = 2 * np.pi * np.arange(n) / n
    c = []
    for ax in range(3):
        m = w.sum(axis=tuple(i for i in range(3) if i != ax))
        c.append((np.arctan2((m * np.sin(ang)).sum(), (m * np.cos(ang)).sum()) / (2 * np.pi)) % 1.0)
    return np.asarray(c)


def shift_np(u, frac):
    """u(x) -> u(x - frac) on the periodic unit cube (frac in units of the box)."""
    n = u.shape[-1]
    s = np.fft.fftn(u, axes=(1, 2, 3))
    k = np.fft.fftfreq(n) * n
    for ax in range(3):
        ph = np.exp(-2j * np.pi * k * frac[ax])
        shape = [1, 1, 1, 1]
        shape[ax + 1] = n
        s = s * ph.reshape(shape)
    return np.fft.ifftn(s, axes=(1, 2, 3)).real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cases", type=int, default=16)
    args = ap.parse_args()
    run = HERE / "runs" / args.job / "output"
    summary = json.loads((run / "summary.json").read_text())
    bank = run / "bank_selected.npz"
    p = BI.load_params(bank)
    T = np.load(bank)["T"]
    n = args.n
    G, info = BI.mesh_bank(p, T, n)
    info.pop("lowdin_factor")
    par = F.parameters(int(summary["config"]["dev_seed"]), args.cases)
    steps = 200
    solver = F.make_solver(0.001, steps, steps // 5)
    geom = F.geometry(n)
    R = G.shape[1]
    prefixes = sorted({int(x) for x in summary["config"]["prefixes"] if int(x) <= R} | {R})
    errs = {Rp: np.zeros((len(par), 6)) for Rp in prefixes}
    for c, row in enumerate(par):
        fr = np.asarray(solver(jnp.asarray(F.initial(n, row)), float(row[-1]), geom))
        den = np.sqrt(np.sum(fr[0] ** 2))
        for t in range(fr.shape[0]):
            cc = centroid_np(fr[t])
            v = shift_np(fr[t], -cc).ravel()
            a = G.T @ v
            for Rp in prefixes:
                rec = shift_np((G[:, :Rp] @ a[:Rp]).reshape(3, n, n, n), cc)
                errs[Rp][c, t] = np.sqrt(np.sum((rec - fr[t]) ** 2)) / den
    out = {}
    for Rp in prefixes:
        mine = float(errs[Rp][:, 1:].max())
        theirs = summary["floors"][str(n)][f"coordnet_R{Rp}"]["evolved_worst"]
        out[f"R{Rp}"] = dict(local=mine, job=theirs, relative_gap=abs(mine - theirs) / theirs)
    payload = dict(job=args.job, n=n, cases=args.cases, bank_info=info, floors=out,
                   passed=bool(all(v["relative_gap"] < 1e-6 for v in out.values())))
    (HERE / "results" / f"local_floor_check_{args.job}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
