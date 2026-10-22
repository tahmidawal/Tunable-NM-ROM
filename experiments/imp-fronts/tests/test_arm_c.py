"""imp-fronts arm (c) 'sine_even' (arm name 'bnd'): JAX-side identity tests, frozen by DESIGN Amendments 19-21.

    JAX_PLATFORMS=cpu bash .../campaign/jaxslot.sh 3600 /home/tahmid/Dev/.venv/bin/python \
        experiments/imp-fronts/tests/test_arm_c.py [--ckpt <trained bnd checkpoint>] [--json <path>] [--root <dir>]

Bank under test: the seed-1 initial sine_even bank, built exactly as smoothtrain's train() builds it
(key = PRNGKey(1); key, kz, kp = split(key, 3); init_separable(kp, 16, 512, n_ff=128, ff_scale=4, g_hidden=1024,
h_hidden=256, out_scale=1, arch='sine_even')), or a trained checkpoint with --ckpt. Code under test: the arm's
sep_common copies (train2d/arms/bnd/sep_common_stage{A,B}.py; Stage B's copy is also the rotation copy) and the NumPy
reference train2d/arms/bnd/sine_even_ref.py.

Fixtures (Amendment 20): 256 points from default_rng(20261022) uniform in [-2, 3]^2 plus every interior wall node of
the 257^2 lattice on all four walls (4 x 255). Aggregation: per column max |statistic| / (that column's RMS on the 257^2
lattice), then max over the 512 columns. A zero RMS makes the test unavailable (counts as a failure).
  T1 oddness       |u(-x,y)+u(x,y)|, |u(x,-y)+u(x,y)|                        <= 1e-12
  T2 periodicity   |u(x+2,y)-u(x,y)|, |u(x,y+2)-u(x,y)|                       <= 1e-12
  T3 wall value    |u| on the wall nodes                                       <= 1e-14
  T4 JAX vs NumPy  values (/ value RMS) and x/y gradients (JAX forward-mode autodiff vs the reference's hand
                   gradient, / that gradient component's lattice RMS)          <= 1e-12
  T5 wall d_n^2    JAX autodiff (forward-over-forward) on the wall nodes / RMS of the column's 5-point interior
                   Laplacian on the lattice (255^2 interior, h = 1/256)          <= 1e-10
  T6 wall d_n^2    A2 gradient difference (d_n u(w + delta n) - d_n u(w - delta n)) / (2 delta), delta = 2.5e-5,
                   same normalisation                                          <= 1e-6
  C+ positive control: 16x(1-x)y(1-y) through T5 and T6 must FAIL (statistic above the tolerance)
  C- must-fail bank control (Amendment 21): the K1 checkpoint (jcp-robust s1) decoded with the coefficients of
     POL-K1's scored terminal run on FI test case 4 at t5; wall d_n^2 by the A2 gradient difference on all interior wall
     nodes; statistic = RMS over the wall nodes / RMS of the 5-point Laplacian of the E0-style reference of FI|4 at t5
     over the 255^2 interior; must exceed 1e-3.
  Extra (load guard, audit codex-code-C-1): load_pkl of both arm copies refuses cfg['arch'] / Kc disagreement and a
     malformed Kc (shape, dtype, non-integer), and accepts valid sine_even and incumbent checkpoints.
  Extra (backward compatibility): with no 'Kc' the arm copies reproduce the original sep_common bit for bit (K1
     checkpoint values; default init); Stage A and Stage B copies agree bit for bit on the arm bank; d loss / d Kc and
     d loss / d B are exactly zero (Kc and B cannot train); Kc = |round(2 B0)|.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

LANE = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = LANE
ROBUST = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust')
TOL = dict(odd=1e-12, per=1e-12, wall=1e-14, jaxnp=1e-12, hess=1e-10, gd=1e-6, bank_control=1e-3)
DELTA = 2.5e-5
fails, out = [], {}


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        fails.append(msg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--json')
    ap.add_argument('--root', default=str(ROOT_DEFAULT), help='lane root holding train2d/ and runs/ (job dir on the cluster)')
    ap.add_argument('--k1-dir', default=str(ROBUST / 'inputs/banks/s1'), help='K1 bank dir (ckpt.pkl, rotation_R512.npz)')
    a = ap.parse_args()
    root = Path(a.root)
    k1 = Path(a.k1_dir)
    T2 = root / 'train2d'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    scA = load('sc_bnd_A', T2 / 'arms/bnd/sep_common_stageA.py')
    scB = load('sc_bnd_B', T2 / 'arms/bnd/sep_common_stageB.py')
    sc0 = load('sc_orig_B', T2 / 'stageB/code/sep_common.py')
    sc0A = load('sc_orig_A', T2 / 'stageA/deps/sep_common.py')
    ref = load('sine_even_ref', T2 / 'arms/bnd/sine_even_ref.py')
    print(f'jax_backend={jax.default_backend()}', flush=True)
    out['backend'] = jax.default_backend()

    # ---------------------------------------------------------------- the bank under test
    key = jax.random.PRNGKey(1)
    key, kz, kp = jax.random.split(key, 3)
    kw = dict(n_ff=128, ff_scale=4.0, g_hidden=1024, h_hidden=256, out_scale=1.0)
    p_init = scA.init_separable(kp, 16, 512, arch='sine_even', **kw)
    p_inc = sc0A.init_separable(kp, 16, 512, **kw)
    p_def = scA.init_separable(kp, 16, 512, **kw)
    same_def = all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree_util.tree_leaves(p_def), jax.tree_util.tree_leaves(p_inc)))
    check(same_def and 'Kc' not in p_def, 'default init (no arch) = original sep_common init, bit for bit, no Kc')
    check(bool(jnp.array_equal(p_init['Kc'], jnp.abs(jnp.round(2 * p_inc['B'])))) and bool(jnp.array_equal(p_init['B'], p_inc['B'])),
          'Kc = |round(2 B0)| of the seed-1 init draw; B kept unchanged')
    nd = [int(len(np.unique(np.asarray(p_init['Kc'][i])))) for i in (0, 1)]
    out['Kc_distinct_per_axis'] = nd
    print(f'  Kc distinct values per axis: {nd}; max {int(np.max(np.asarray(p_init["Kc"])))}', flush=True)
    if a.ckpt:
        params, _, cfg = scB.load_pkl(a.ckpt)
        check(cfg.get('arch') == 'sine_even' and 'Kc' in params, f"checkpoint cfg['arch'] = {cfg.get('arch')!r} and Kc present")
        out['bank'] = a.ckpt
    else:
        params = p_init
        out['bank'] = 'seed-1 initial sine_even bank'
    pn = jax.tree_util.tree_map(np.asarray, params)

    def fJ(X):
        return scB.features(params, jnp.asarray(X))

    def chunked(fn, X, n=4096):
        return np.concatenate([np.asarray(fn(X[s:s + n])) for s in range(0, len(X), n)])

    XL = np.arange(257) / 256.
    LX, LY = np.meshgrid(XL, XL, indexing='ij')
    lat = np.stack([LX.ravel(), LY.ravel()], 1)
    G = chunked(fJ, lat)
    rms = np.sqrt(np.mean(G ** 2, axis=0))
    check(bool(np.all(rms > 0)), f'column RMS on the lattice positive (min {rms.min():.3e})')

    def lap_rms(F):                                   # F (257, 257, R) -> per-column RMS of the 5-point interior Laplacian
        L = (F[2:, 1:-1] + F[:-2, 1:-1] + F[1:-1, 2:] + F[1:-1, :-2] - 4 * F[1:-1, 1:-1]) * 256. ** 2
        return np.sqrt(np.mean(L.reshape(-1, F.shape[-1]) ** 2, axis=0))
    lrms = lap_rms(G.reshape(257, 257, -1))

    rng = np.random.default_rng(20261022)
    Prand = rng.uniform(-2.0, 3.0, (256, 2))
    i = np.arange(1, 256) / 256.
    walls = dict(x0=np.stack([0 * i, i], 1), x1=np.stack([0 * i + 1, i], 1), y0=np.stack([i, 0 * i], 1), y1=np.stack([i, 0 * i + 1], 1))
    W_all = np.concatenate(list(walls.values()))
    P = np.concatenate([Prand, W_all])

    def agg(D, scale):
        return float(np.max(np.max(np.abs(D), axis=0) / scale))

    fP = chunked(fJ, P)
    s = dict(odd_x=agg(chunked(fJ, P * [-1, 1]) + fP, rms), odd_y=agg(chunked(fJ, P * [1, -1]) + fP, rms),
             per_x=agg(chunked(fJ, P + [2, 0]) - fP, rms), per_y=agg(chunked(fJ, P + [0, 2]) - fP, rms),
             wall=agg(chunked(fJ, W_all), rms))
    # T4: values and gradients, JAX autodiff vs the NumPy reference
    ex, ey = np.array([1., 0.]), np.array([0., 1.])

    def dJ(X, e):
        X = jnp.asarray(X)
        return jax.jvp(lambda Y: scB.features(params, Y), (X,), (jnp.broadcast_to(jnp.asarray(e), X.shape),))[1]
    Gx_lat = chunked(lambda X: dJ(X, ex), lat)
    Gy_lat = chunked(lambda X: dJ(X, ey), lat)
    rx, ry = np.sqrt(np.mean(Gx_lat ** 2, 0)), np.sqrt(np.mean(Gy_lat ** 2, 0))
    Rv, Rx, Ry = ref.features_grad(pn, P)
    s['jaxnp_val'] = agg(Rv - fP, rms)
    s['jaxnp_gx'] = agg(Rx - chunked(lambda X: dJ(X, ex), P), rx)
    s['jaxnp_gy'] = agg(Ry - chunked(lambda X: dJ(X, ey), P), ry)

    # T5 / T6: wall normal second derivative
    def d2J(f, X, e):
        X = jnp.asarray(X)
        E = jnp.broadcast_to(jnp.asarray(e), X.shape)
        g = lambda Y: jax.jvp(f, (Y,), (E,))[1]
        return jax.jvp(g, (X,), (E,))[1]

    def gdJ(f, X, e):
        X = jnp.asarray(X)
        E = jnp.broadcast_to(jnp.asarray(e), X.shape)
        g = lambda Y: jax.jvp(f, (Y,), (E,))[1]
        return (g(X + DELTA * E) - g(X - DELTA * E)) / (2 * DELTA)

    nrm = dict(x0=ex, x1=ex, y0=ey, y1=ey)
    fB = lambda Y: scB.features(params, Y)
    s['hess'] = max(agg(np.asarray(d2J(fB, walls[w], nrm[w])), lrms) for w in walls)
    s['gd'] = max(agg(np.asarray(gdJ(fB, walls[w], nrm[w])), lrms) for w in walls)
    out['stats'] = s
    for k, t in (('odd_x', 'odd'), ('odd_y', 'odd'), ('per_x', 'per'), ('per_y', 'per'), ('wall', 'wall'),
                 ('jaxnp_val', 'jaxnp'), ('jaxnp_gx', 'jaxnp'), ('jaxnp_gy', 'jaxnp'), ('hess', 'hess'), ('gd', 'gd')):
        check(np.isfinite(s[k]) and s[k] <= TOL[t], f'{k}: {s[k]:.3e} <= {TOL[t]:g}')

    # bit-for-bit: Stage A copy = Stage B copy on the arm bank; incumbent path unchanged (K1 checkpoint)
    check(np.array_equal(np.asarray(scA.features(params, jnp.asarray(P))), fP), 'Stage A copy = Stage B/rotation copy (arm bank), bit for bit')
    pk, _, ck = sc0.load_pkl(k1 / 'ckpt.pkl')
    v0 = np.asarray(sc0.features(pk, jnp.asarray(P)))
    check(np.array_equal(np.asarray(scB.features(pk, jnp.asarray(P))), v0) and
          np.array_equal(np.asarray(scA.features(pk, jnp.asarray(P))), v0),
          'no Kc (K1 checkpoint): arm copies = original sep_common, bit for bit')
    # load_pkl architecture guard (audit codex-code-C-1), both arm copies
    import pickle as _pk
    import tempfile
    base = jax.tree_util.tree_map(np.asarray, p_init)
    cases = dict(valid=(base, {'arch': 'sine_even'}, True),
                 kc_no_arch=(base, {}, False),
                 kc_wrong_arch=(base, {'arch': 'bc16poly'}, False),
                 arch_no_kc=({k: v for k, v in base.items() if k != 'Kc'}, {'arch': 'sine_even'}, False),
                 kc_shape=(dict(base, Kc=base['Kc'][:, :64]), {'arch': 'sine_even'}, False),
                 kc_dtype=(dict(base, Kc=base['Kc'].astype(np.float32)), {'arch': 'sine_even'}, False),
                 kc_nonint=(dict(base, Kc=base['Kc'] + 0.5), {'arch': 'sine_even'}, False),
                 incumbent=({k: v for k, v in base.items() if k != 'Kc'}, {}, True))
    guard = {}
    with tempfile.TemporaryDirectory() as td:
        for nm, (pp, cfg, want) in cases.items():
            f = Path(td) / f'{nm}.pkl'
            f.write_bytes(_pk.dumps(dict(params=pp, Z_tr=np.zeros((1, 16)), cfg=cfg)))
            for tag, mod in (('A', scA), ('B', scB)):
                try:
                    mod.load_pkl(str(f)); got = True
                except ValueError:
                    got = False
                guard[f'{nm}/{tag}'] = got == want
    out['load_guard'] = guard
    check(all(guard.values()), f"load_pkl guard (Stage A and Stage B/rotation copies): accepts valid sine_even and incumbent, "
          f"refuses Kc/arch mismatches, bad shape, dtype, non-integer Kc ({sum(guard.values())}/{len(guard)})")
    check(scB.load_pkl(str(k1 / 'ckpt.pkl'))[2].get('arch') is None, 'K1 checkpoint loads through the guarded load_pkl')
    # Kc and B cannot train
    loss = lambda p: jnp.sum(scB.features(p, jnp.asarray(P[:64])) ** 2)
    gr = jax.grad(loss)(params)
    check(float(jnp.max(jnp.abs(gr['Kc']))) == 0.0 and float(jnp.max(jnp.abs(gr['B']))) == 0.0,
          'd loss / d Kc = d loss / d B = 0 exactly (fixed frequencies; B unused)')

    # C+: polynomial positive control (scalar field, same statistics)
    poly = lambda Y: (16. * Y[..., 0] * (1 - Y[..., 0]) * Y[..., 1] * (1 - Y[..., 1]))[..., None]
    PL = np.asarray(poly(jnp.asarray(lat))).reshape(257, 257, 1)
    pl = lap_rms(PL)
    ph = max(agg(np.asarray(d2J(poly, walls[w], nrm[w])), pl) for w in walls)
    pg = max(agg(np.asarray(gdJ(poly, walls[w], nrm[w])), pl) for w in walls)
    out['positive_control'] = dict(hess=ph, gd=pg)
    check(ph > TOL['hess'] and pg > TOL['gd'], f'C+ polynomial must fail: hess {ph:.3e} > 1e-10, gd {pg:.3e} > 1e-6')

    # C-: K1 bank, POL-K1 scored terminal run of FI|4 at t5 (Amendment 21)
    try:
        polroot = root / 'runs/rom/POL-K1-out/output/fpolicy_pol1'
        res = json.loads((polroot / 'result.json').read_text())
        prog = json.loads((polroot / 'progress.json').read_text())
        case = next(c for c in prog['cases'] if c['case'] == 'FI|4')
        assert res['bank']['ckpt_sha256'] == hashlib.sha256((k1 / 'ckpt.pkl').read_bytes()).hexdigest() == '244e1107c33ae38a23d17517fc198d7f2f307f12d2bb20006d3ed7aada188954'
        assert res['bank']['rotation_sha256'] == hashlib.sha256((k1 / 'rotation_R512.npz').read_bytes()).hexdigest()
        coh, j, lev, rule, f = case['scored'].split('|')
        fname = f"{coh}__{j}__{lev}__{rule}__{f.replace('.', 'p').rstrip('0').rstrip('p') if '.' in f else f}.npz"
        z = np.load(polroot / 'runs' / fname)
        w5 = np.asarray(z['W'], float)[5]
        Trot = np.load(k1 / 'rotation_R512.npz')['T'][:, :384]
        coef = jnp.asarray(Trot @ w5)
        fK = lambda Y: (sc0.features(pk, Y) @ coef)[..., None]
        refz = np.load(root / 'runs/refT/pulled/refslim/2d/FI/ST/c004.npz')['a2_L8192']
        u5 = refz[5]
        Lref = (u5[2:, 1:-1] + u5[:-2, 1:-1] + u5[1:-1, 2:] + u5[1:-1, :-2] - 4 * u5[1:-1, 1:-1]) * 256. ** 2
        den = float(np.sqrt(np.mean(Lref ** 2)))
        dec = chunked(fK, lat).reshape(257, 257)
        dec_err = float(np.linalg.norm(dec - u5) / np.linalg.norm(refz[0]))
        gd = np.concatenate([np.asarray(gdJ(fK, walls[w], nrm[w]))[:, 0] for w in walls])
        stat = float(np.sqrt(np.mean(gd ** 2)) / den)
        out['bank_control'] = dict(run=fname, terminal=case['terminal'], stat=stat, denominator=den, decode_vs_ref_t5=dec_err)
        print(f'  C- K1 decoded FI|4 t5 ({fname}): decode-vs-reference {dec_err:.3e}; wall gd RMS / ref Laplacian RMS {stat:.3e}')
        check(dec_err < 0.05, f'C- decode sanity: K1 decoded field within 5 % of the reference at t5 ({dec_err:.3e})')
        check(stat > TOL['bank_control'], f'C- K1 bank must fail: {stat:.3e} > 1e-3')
    except Exception as e:                          # missing evidence blocks (counts as a failure)
        check(False, f'C- bank control unavailable: {type(e).__name__}: {e}')

    out['fails'] = fails
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(f'\n{len(fails)} failure(s)')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
