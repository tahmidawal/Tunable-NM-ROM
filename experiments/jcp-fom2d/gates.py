"""jcp-fom2d gates (DESIGN §6). Each function returns a JSON-able dict with 'passed' (gates) or 'fires' (controls).

  g_fd1()          refcore.py / pscore.py byte-identical to the E0 sources (sha256 from INHERITED.json, re-hashed)
  g_fd2(...)       F-UB == vendor engines.make_fom (BE) at equal tolerances, 1e-12 relative
  g_e0(...)        F-CC at L=2048, dt=0.00125, (1e-11, 1e-9) == stored E0 level a0_L2048 lattice fields, 1e-10 relative
  g_ps1(...)       psjax2d == pscore.make_ps (1e-12), half-step control (> 1e-9)
  g_ps2(...)       psjax2d reproduces the stored E0 cross-check PS run val32|0 N1024_dt0_a0 (1e-10)
  g_int(...)       sine interpolation exact at coincident nodes / for band-limited sine fields; PS FFT == matrix synthesis
  g_ct(...)        lattice restriction of a full-field output == lattice-contract output (1e-13), each family
  k_swap(...)      case swap / time shift change the score by > 1e-6
Relative differences are max |a - b| / max |b| over all output times.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import fomrun2 as FR
import psjax2d
import pscore as PS
import refstore as RS
import cohorts as CH

HERE = Path(__file__).resolve().parent


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def g_fd1():
    inh = json.loads((HERE / 'INHERITED.json').read_text())
    rows = []
    for f in inh['files']:
        h = hashlib.sha256((HERE / f['path']).read_bytes()).hexdigest()
        rows.append(dict(path=f['path'], sha256=h, source_sha256=f['source_sha256'], equal=h == f['source_sha256']))
    return dict(files=rows, passed=all(r['equal'] for r in rows))


def g_fd2(desc, L=256, dt=0.005, ntol=1e-8, ltol=1e-9):
    """F-UB (refcore upwind, BE) vs the vendor engines.make_fom at equal tolerances: same scheme arithmetic, same Newton
    rule and BiCGStab limit (200), same preconditioner; outputs on mesh L (walls zero)."""
    foms = FR.FOMs()
    sp = ('fd', 'UB', L, dt, '1')
    x, _ = FR.prepare(sp, desc)
    F, o = foms.run(sp, x, float(desc[4]), ('mesh', L))
    q, _ = FR.E2.make_fom(L, dt)
    G, it, rn = q(jnp.asarray(FR.E2.initial(L, desc)), float(desc[4]), ntol, ltol)
    r = rel(F, G)
    return dict(L=L, dt=dt, ntol=ntol, ltol=ltol, rel=r, passed=bool(r <= 1e-12),
                newton_equal=bool(np.array_equal(np.asarray(o['nwt'])[:int(o['stages'])], np.asarray(it))))


def g_e0(cohort, case, desc):
    """Reproduce a trusted FOM: F-CC at E0's coarsest level of this case (L, dt, ntol, ltol, scheme read from the stored
    attempt metadata; asserted to be central CN-R) vs the stored lattice field."""
    arr, meta = RS._load_raw(2, cohort, case, 'ST')
    att = {a['id']: a for a in meta['attempts']}
    a0 = att[meta['final'][0]]
    assert a0['adv'] == 'central' and a0['scheme'] == 'cnr' and a0['max_newton'] == 20, a0
    L, dt, ntol, ltol = int(a0['L']), float(a0['dt']), float(a0['ntol']), float(a0['ltol'])
    s = FR.FC.make_solver(2, L, adv='central')
    o = s(FR.fd_input(desc, L), float(desc[4]), dt, 'cnr', ntol, ltol)
    F = FR.fd_map(L, 'lattice')(o['out'])
    ref = np.asarray(arr[meta['final'][0]])
    r = rel(F, ref)
    return dict(cohort=cohort, case=case, attempt=a0['id'], L=L, dt=dt, ntol=ntol, ltol=ltol, rel=r, passed=bool(r <= 1e-10),
                newton_total=int(np.asarray(o['nwt'])[:int(o['stages'])].sum()), stored_newton_total=a0.get('newton_total'))


def g_ps1(descs):
    return psjax2d.gate(descs, [float(d[4]) for d in descs])


def g_ps2(desc, cohort='val32', case=0, run_id='N1024_dt0_a0'):
    man = RS._manifest()
    rels = sorted(k for k in man if k.startswith(f'2d/{cohort}/PS/c{case:03d}_a') and k.endswith('.json'))
    stem = rels[-1][:-5]
    meta = json.loads(RS._read(stem + '.json'))
    arr = np.load(io.BytesIO(RS._read(stem + '.npz')))
    run = next(r for r in meta['runs'] if r['id'] == run_id)
    N, dt = int(run['N']), float(run['dt'])
    q = psjax2d.make_ps_query(N, dt)
    F = q(psjax2d.ic(desc, N), float(desc[4]))
    r = rel(F, arr[run_id])
    return dict(stem=stem, run=run_id, N=N, dt=dt, rel=r, passed=bool(r <= 1e-10))


def g_int(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for L, M in ((64, 256), (128, 256), (128, 1024)):
        U = jnp.asarray(rng.standard_normal((2, L - 1, L - 1)))
        V = np.asarray(FR.sine_interp(U, L, M))
        s = M // L
        out[f'coincident_{L}_{M}'] = rel(V[:, s - 1::s, s - 1::s], U)
        # band-limited: u = sum_{k,l<L} a_kl sin(k pi x) sin(l pi y) sampled on both meshes
        k = np.arange(1, L)
        a = rng.standard_normal((L - 1, L - 1)) / (1 + k[:, None] ** 2 + k[None, :] ** 2)
        xs = lambda n: np.sin(np.pi * np.outer(np.arange(1, n) / n, k))
        uL = xs(L) @ a @ xs(L).T
        uM = xs(M) @ a @ xs(M).T
        out[f'bandlimited_{L}_{M}'] = rel(np.asarray(FR.sine_interp(jnp.asarray(uL), L, M)), uM)
    b = jnp.asarray(rng.standard_normal((47, 47)) / 50)
    lat_mat = np.asarray(PS.synth_lattice(b, PS.lattice_S(48, 256), 2))
    lat_fft = np.asarray(psjax2d.synth_mesh(b, 256))
    out['ps_fft_vs_matrix'] = rel(lat_fft, lat_mat)
    passed = all(v <= (1e-14 if k_.startswith('coincident') else 1e-12) for k_, v in out.items())
    # the coincident-node test compares nodes j*s (1-based) of the fine interior with node j of the coarse interior
    return dict(values=out, passed=bool(passed))


def g_ct(foms, specs, desc, M=1024):
    """lattice restriction of the full-field (mesh M) output == lattice-contract output."""
    out = {}
    for sp in specs:
        x, _ = FR.prepare(sp, desc)
        nu = float(desc[4])
        A, _ = foms.run(sp, x, nu, 'lattice')
        B, _ = foms.run(sp, x, nu, ('mesh', M))
        out[FR.key(sp)] = rel(np.asarray(B)[:, ::M // 256, ::M // 256], A)
    return dict(values=out, passed=all(v <= 1e-13 for v in out.values()))


def k_swap(F, cohort, case, other):
    """case swap: the field of `case` scored against `other`'s reference; time shift: outputs rolled by one."""
    s0 = RS.score(F, 2, cohort, case)['e']
    sw = RS.score(F, 2, cohort, other)['e']
    sh = RS.score(np.concatenate((F[:1], F[:1], F[1:5])), 2, cohort, case)['e']
    return dict(e=s0, e_swap=sw, e_shift=sh, fires=bool(abs(sw - s0) > 1e-6 and abs(sh - s0) > 1e-6))


def descs(cohort):
    return CH.Cohort(2, cohort).D
