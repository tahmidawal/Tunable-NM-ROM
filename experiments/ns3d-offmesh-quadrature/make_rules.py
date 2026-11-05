"""Generate this lane's rule ladder (DESIGN.md section 2) with the vendored, byte-identical
generators of exp/2026-10-01-quadrature-burgers3d (vendor/quadrature-burgers3d/rules.py).

Writes rules/rules.npz ({name}_X (m, 3), {name}_w (m,)) and rules/rules.json (family, size,
generating vector, P_2 merit, weight sum, per-rule sha256). Run once locally; the cluster only
reads the committed files. The shifted trapezoid rules (`trap`) are generated here too (and by
`trapezoid()` in-job for the large continuum targets, which are never stored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "quadrature-burgers3d"))
import rules as RU  # noqa: E402  (vendored, unchanged)

POW2 = [2 ** e for e in range(9, 17)]
KOR_PRIMES = [509, 1021, 2039, 4093, 8191, 16381, 32749, 65521]
P_TENSOR = [8, 10, 13, 16, 20, 25, 32, 40]


def trapezoid(p, seed):
    """Shifted periodic trapezoid p^3 on [0,1)^3, uniform random shift in [0, 1/p)^3."""
    shift = np.random.default_rng(seed).uniform(size=3) / p
    s = np.arange(p, dtype=np.float64) / p
    X = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3) + shift[None, :]
    return X % 1.0, np.full(p ** 3, 1.0 / p ** 3), shift


def is_prime(v):
    return v > 1 and all(v % d for d in range(2, int(v ** 0.5) + 1))


def build(log=print):
    rules, meta = {}, {}

    def add(name, family, X, w, role, **extra):
        X = np.ascontiguousarray(X, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 3 and len(w) == len(X)
        assert np.all((X >= 0) & (X <= 1)), name
        rules[name] = (X, w)
        meta[name] = dict(family=family, m=int(len(w)), role=role, weight_sum=float(w.sum()),
                          w_min=float(w.min()),
                          sha256=hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest(), **extra)
        log(f"{name}: m={len(w)} wsum={w.sum():.15f} {extra}")

    z, P = RU.cbc_lattice_vector3(64)
    add("lat64", "lat", *RU.rank1_lattice3(64, z, seed=0), "must-fail control",
        z=[int(v) for v in z], p2=P, shift_seed=0)
    for n in POW2:
        z, P = RU.cbc_lattice_vector3(n)
        if n in RU.HARI_CBC:
            assert tuple(int(v) for v in z) == RU.HARI_CBC[n], (n, z)
        add(f"lat{n}", "lat", *RU.rank1_lattice3(n, z, seed=0), "ladder",
            z=[int(v) for v in z], p2=P, p2_check=RU.p2_merit3(z, n), shift_seed=0)
    for n in KOR_PRIMES:
        assert is_prime(n), n
        z, P = RU.korobov_search3(n)
        add(f"kor{n}", "kor", *RU.rank1_lattice3(n, z, seed=0), "ladder",
            z=[int(v) for v in z], p2=P, p2_check=RU.p2_merit3(z, n), shift_seed=0)
    for n in POW2:
        add(f"sob{n}", "sob", *RU.sobol3(n, seed=0), "ladder", scramble_seed=0)
    for p in P_TENSOR:
        add(f"gl{p}", "gl", *RU.gauss_tensor3(p), "ladder", p=p)
    for p in P_TENSOR:
        X, w, shift = trapezoid(p, 0)
        add(f"trap{p}", "trap", X, w, "ladder (supplementary)", p=p, shift=shift.tolist(),
            shift_seed=0)
    return rules, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "rules" / "rules.npz"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rules, meta = build()
    np.savez(out, **{f"{k}_X": v[0] for k, v in rules.items()},
             **{f"{k}_w": v[1] for k, v in rules.items()})
    (out.parent / "rules.json").write_text(json.dumps(dict(
        generator="experiments/ns3d-offmesh-quadrature/make_rules.py",
        vendored_rules_sha256=hashlib.sha256(
            (HERE / "vendor" / "quadrature-burgers3d" / "rules.py").read_bytes()).hexdigest(),
        rules=meta, npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest()), indent=1) + "\n")


if __name__ == "__main__":
    main()
