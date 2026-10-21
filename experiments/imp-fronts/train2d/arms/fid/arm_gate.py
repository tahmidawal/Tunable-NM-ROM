"""imp-fronts arm (a) 'fid': the R0 replacement gate (DESIGN B1(a), Amendment 1), called by the arm's smoothtrain.py
at the production data size (256 nodes, 576 trajectories) after the training data are built, before any training.

Replaced gate (fails closed): (1) every training state finite (the generator's residual sentinel already aborts on a
non-finite state; checked again here on S_tr); (2) C-B3: on all six dev6 cases the PS fields of THIS generator
(N = 128, dt = 0.001), synthesised on the 257^2 scoring lattice x = i/256 at t = 0.05 k, k = 1..5, are within 1e-4
(0.01 %, project metric max_k ||v - u|| / ||u(t0)||) of the E0 references (inputs/fid_gate_dev6.npz, built from the E0
ST store by arms/fid/make_gate_inputs.py, sha256 in PROVENANCE.json); (3) PS self-convergence: N = 128, dt = 0.001
vs N = 192, dt = 0.0005 distance <= 1e-4 (same metric) on the same six cases.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

INPUT = Path(__file__).resolve().parent.parent / 'inputs' / 'fid_gate_dev6.npz'
BAR_E0 = 1e-4
BAR_SELF = 1e-4
X_LAT = np.arange(257) / 256.0


def metric(v, u, n0):
    """max over t1..t5 of ||v - u|| / n0 (inf if non-finite); v, u (5, 257, 257) at t = 0.05 k, k = 1..5."""
    d = np.linalg.norm((np.asarray(v) - np.asarray(u)).reshape(5, -1), axis=1) / n0
    return float(d.max()) if np.isfinite(d).all() else float('inf')


def lattice_fields(descs, N, dt):
    import burgers2d_film as bf
    every = int(round(0.05 / dt))
    assert abs(every * dt - 0.05) < 1e-12
    return bf.ps_fields(descs, X_LAT, N=N, dt=dt, every=every, n_out=5)[:, 1:]       # (B, 5, 257, 257)


def check_dev6(ref_path=INPUT):
    import burgers2d_film as bf
    z = np.load(ref_path)
    descs = [tuple(r) for r in z['desc']]                  # (cx, cy, w, a, nu)
    t0 = time.time()
    u128 = lattice_fields(descs, bf.PS_N, bf.PS_DT)
    t1 = time.time()
    u192 = lattice_fields(descs, 192, bf.PS_DT / 2)
    t2 = time.time()
    rows = []
    for j in range(len(descs)):
        rows.append(dict(case=int(z['case'][j]), eligible=bool(z['eligible'][j]), eps_ref=float(z['eps_ref'][j]),
                         err_vs_E0=metric(u128[j], z['ref'][j], float(z['n0'][j])),
                         self_conv=metric(u128[j], u192[j], float(z['n0'][j]))))
    worst_e0 = max(r['err_vs_E0'] for r in rows)
    worst_self = max(r['self_conv'] for r in rows)
    return dict(rows=rows, worst_err_vs_E0=worst_e0, worst_self_conv=worst_self,
                C_B3_pass=bool(worst_e0 <= BAR_E0), self_conv_pass=bool(worst_self <= BAR_SELF),
                ps_config=dict(N=bf.PS_N, dt=bf.PS_DT, every=bf.PS_EVERY, self_conv=dict(N=192, dt=bf.PS_DT / 2)),
                seconds=dict(ps128=t1 - t0, ps192=t2 - t1), u128=u128)


def r0_replacement(S_tr, worst, fingerprint):
    import burgers2d_film as bf
    assert bf.PS_N == 128 and bf.PS_DT == 0.001 and bf.PS_EVERY == 5, 'arm fid: solver configuration changed'
    finite = bool(np.isfinite(np.asarray(S_tr)).all()) and bool(np.isfinite(worst))
    r = check_dev6()
    r.pop('u128')
    passed = finite and r['C_B3_pass'] and r['self_conv_pass']
    return dict(arm='fid', passed=bool(passed), all_states_finite=finite, sentinel_worst=float(worst),
                fingerprint=fingerprint, bars=dict(C_B3=BAR_E0, self_conv=BAR_SELF), **r,
                summary=dict(finite=finite, worst_err_vs_E0=r['worst_err_vs_E0'], worst_self_conv=r['worst_self_conv']),
                note='replaces R0 (r3a fingerprint) and the upwind residual gate for PS data (DESIGN Amendment 1 B1(a))')
