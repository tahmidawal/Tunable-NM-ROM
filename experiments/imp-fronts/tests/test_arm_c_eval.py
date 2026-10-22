"""imp-fronts arm (c) evaluator-side identity tests (DESIGN Amendments 19-20). JAX on CPU (run through the slot guard):

    JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest bash <jaxslot.sh> 3600 /home/tahmid/Dev/.venv/bin/python \
        -m pytest -q -p no:cacheprovider experiments/imp-fronts/tests/test_arm_c_eval.py

Subjects: the ROM path's sep_common (rom/deps, = the training side's committed copy), the NumPy evaluator npbank with
its hand gradient patched by rom/npbank_arch, and the reference rom/deps/sine_even_ref. A synthetic 'sine_even'
checkpoint is built with the training side's own init_separable(arch='sine_even') (n_ff = 128, the registered input
width 256; a narrow MLP keeps CPU time small; the formula does not depend on the width).

Frozen fixtures (Amendment 20): 256 points from default_rng(20261022) uniform in [-2, 3]^2 plus all interior wall nodes
of the 257^2 lattice on the four walls; values aggregated as max |.| / RMS of the bank output on the 257^2 lattice, per
column then max over columns; gradients normalised by the gradient RMS; wall d_n^2 by JAX autodiff Hessian (<= 1e-10 of
the interior 5-point Laplacian RMS) and by the A2 gradient difference at delta = 2.5e-5 (<= 1e-6). Positive control:
the incumbent envelope 16x(1-x)y(1-y) with the same MLP must fail the curvature test. Must-fail bank control: K1 (rom
inputs/banks/s1) with the coefficients of POL-K1's terminal run on FI test case 4 at t5, same statistic, > 1e-3.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

T = Path(__file__).resolve().parent
LANE = T.parent
ROM = LANE / 'rom'
for p_ in (str(ROM), str(ROM / 'deps'), str(ROM / 'vendor' / 'quad2d')):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import sep_common as SC  # noqa: E402   (rom/deps: the arch-dispatching copy)
import sine_even_ref as SE  # noqa: E402
import npbank as NB  # noqa: E402
import npbank_arch  # noqa: E402

npbank_arch.install(NB)
DELTA = 2.5e-5
POLK1 = LANE / 'runs' / 'rom' / 'POL-K1-out' / 'output' / 'fpolicy_pol1'


def lattice(n=257):
    x = np.arange(n) / (n - 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.stack([X.ravel(), Y.ravel()], 1)


def wall_nodes(n=257):
    i = np.arange(1, n - 1) / (n - 1)
    z, o = np.zeros_like(i), np.ones_like(i)
    pts = np.concatenate([np.stack([z, i], 1), np.stack([o, i], 1), np.stack([i, z], 1), np.stack([i, o], 1)])
    nrm = np.concatenate([np.tile([1., 0.], (len(i), 1)), np.tile([-1., 0.], (len(i), 1)),
                          np.tile([0., 1.], (len(i), 1)), np.tile([0., -1.], (len(i), 1))])
    return pts, nrm


def fixture_points():
    P = np.random.default_rng(20261022).uniform(-2., 3., size=(256, 2))
    return np.concatenate([P, wall_nodes()[0]])


@pytest.fixture(scope='module')
def ckpt(tmp_path_factory):
    key = jax.random.PRNGKey(1)
    p = SC.init_separable(key, 16, 16, n_ff=128, ff_scale=4.0, g_hidden=48, g_layers=2, h_hidden=16, h_layers=1,
                          out_scale=0.7, arch='sine_even')
    # untrained MLPs are smooth and small: scale the last layer so the field is O(1) and non-trivial
    host = jax.tree_util.tree_map(np.asarray, p)
    f = tmp_path_factory.mktemp('c') / 'ckpt.pkl'
    pickle.dump(dict(params=host, Z_tr=np.zeros((4, 16)), cfg=dict(arch='sine_even', seed=1)), open(f, 'wb'))
    inc = dict(host)
    inc.pop('Kc')
    fi = f.parent / 'inc.pkl'
    pickle.dump(dict(params=inc, Z_tr=np.zeros((4, 16)), cfg=dict(seed=1)), open(fi, 'wb'))
    return f, fi


def jax_feat(params):
    P = jax.tree_util.tree_map(jnp.asarray, params)
    return lambda X: SC.features(P, jnp.asarray(X))


def rms_cols(V):
    return np.sqrt(np.mean(V ** 2, axis=0))


def interior_lap_rms(V, n=257):
    U = V.reshape(n, n, -1)
    h = 1. / (n - 1)
    L = (U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]) / h ** 2
    return np.sqrt(np.mean(L ** 2, axis=(0, 1)))


def dnn_graddiff(fg, P, pts, nrm, delta=DELTA):
    gp = fg(P, pts + delta * nrm)
    gm = fg(P, pts - delta * nrm)
    dp = gp[1] * nrm[:, :1] + gp[2] * nrm[:, 1:]
    dm = gm[1] * nrm[:, :1] + gm[2] * nrm[:, 1:]
    return (dp - dm) / (2 * delta)


def test_dispatch_and_checkpoint_arch(ckpt):
    f, fi = ckpt
    P = NB.load(str(f))
    assert P['arch'] == 'sine_even' and 'Kc' in P and np.array_equal(P['Kc'], np.round(P['Kc']))
    assert NB.load(str(fi))['arch'] == 'bc16poly'
    bad = pickle.load(open(f, 'rb'))
    bad['cfg'] = dict(seed=1)                                  # Kc present, cfg says incumbent -> refused
    pickle.dump(bad, open(f.parent / 'bad.pkl', 'wb'))
    with pytest.raises(ValueError):
        NB.load(str(f.parent / 'bad.pkl'))


def test_jax_numpy_values_and_gradients(ckpt):
    f, _ = ckpt
    ck = pickle.load(open(f, 'rb'))
    P = NB.load(str(f))
    X = np.concatenate([fixture_points(), lattice()[::37]])
    fj = jax_feat(ck['params'])
    Vj = np.asarray(fj(X))
    gx = np.asarray(jax.jvp(fj, (jnp.asarray(X),), (jnp.tile(jnp.array([1., 0.]), (len(X), 1)),))[1])
    gy = np.asarray(jax.jvp(fj, (jnp.asarray(X),), (jnp.tile(jnp.array([0., 1.]), (len(X), 1)),))[1])
    Vn, Gxn, Gyn = NB.features_grad(P, X)
    Vr = SE.features(ck['params'], X)
    VL, GxL, GyL = NB.features_grad(P, lattice())         # normalisations on the 257^2 lattice, per column and component
    s, sgx, sgy = rms_cols(VL), rms_cols(GxL), rms_cols(GyL)
    assert np.all(s > 0) and np.all(sgx > 0) and np.all(sgy > 0)
    assert np.max(np.abs(Vj - Vn) / s) <= 1e-12 and np.max(np.abs(Vr - Vn) / s) <= 1e-12
    assert np.max(np.abs(gx - Gxn) / sgx) <= 1e-12 and np.max(np.abs(gy - Gyn) / sgy) <= 1e-12


def test_oddness_periodicity_wall_values(ckpt):
    f, _ = ckpt
    P = NB.load(str(f))
    X = fixture_points()
    V = NB.features_grad(P, X)[0]
    s = rms_cols(NB.features_grad(P, lattice())[0])
    assert np.all(s > 0)
    for Xs, sign in ((X * [-1, 1], -1), (X * [1, -1], -1), (X + [2, 0], 1), (X + [0, 2], 1)):
        W = NB.features_grad(P, Xs)[0]
        assert np.max(np.max(np.abs(W - sign * V), axis=0) / s) <= 1e-12
    Vw = NB.features_grad(P, wall_nodes()[0])[0]
    assert np.max(np.max(np.abs(Vw), axis=0) / s) <= 1e-14


def test_wall_curvature_hessian_and_graddiff_and_positive_control(ckpt):
    f, fi = ckpt
    ck = pickle.load(open(f, 'rb'))
    P = NB.load(str(f))
    pts, nrm = wall_nodes()
    L = interior_lap_rms(NB.features_grad(P, lattice())[0])
    assert np.all(L > 0)
    fj = jax_feat(ck['params'])
    one = lambda x: fj(x[None])[0]
    H = np.asarray(jax.vmap(jax.jacfwd(jax.jacfwd(one)))(jnp.asarray(pts)))      # (n, R, 2, 2)
    dnn_h = np.einsum('nrij,ni,nj->nr', H, nrm, nrm)
    assert np.max(np.max(np.abs(dnn_h), axis=0) / L) <= 1e-10
    dnn_g = dnn_graddiff(NB.features_grad, P, pts, nrm)
    assert np.max(np.max(np.abs(dnn_g), axis=0) / L) <= 1e-6
    # positive control: the incumbent envelope with the same network must FAIL the same test
    Pi = NB.load(str(fi))
    Li = interior_lap_rms(NB.features_grad(Pi, lattice())[0])
    di = dnn_graddiff(NB.features_grad, Pi, pts, nrm)
    assert np.max(np.max(np.abs(di), axis=0) / Li) > 1e-3


REFT = LANE / 'runs' / 'refT' / 'pulled' / 'refslim' / '2d' / 'FI' / 'ST' / 'c004.npz'


def test_must_fail_bank_control_k1_terminal_fi4():
    """Amendment 21: K1 + POL-K1's scored terminal coefficients on FI|4 at t5; the decoded field's wall d_n^2 by the A2
    gradient difference (delta = 2.5e-5) on all interior wall nodes; statistic = RMS over the wall nodes / RMS of the
    5-point Laplacian of the E0-style REFERENCE of FI|4 at t5 over the 255^2 interior; must exceed 1e-3. Missing
    evidence FAILS (it blocks), it is never skipped."""
    res = POLK1 / 'result.json'
    assert res.exists() and REFT.exists(), 'must-fail control evidence unavailable (POL-K1 output / FI|4 reference)'
    r = json.loads(res.read_text())
    key = next(p['scored'] for p in r['policy'] if p['case'] == 'FI|4')
    z = np.load(POLK1 / 'runs' / f"{key.replace('|', '__').replace('.', 'p')}.npz")
    w = np.asarray(z['W'])[5]
    bank = ROM / 'inputs' / 'banks' / 's1'
    P = NB.load(str(bank / 'ckpt.pkl'))
    c = np.asarray(np.load(bank / 'rotation_R512.npz')['T'])[:, :len(w)] @ w
    u5 = np.asarray(np.load(REFT)['a2_L8192'])[5]
    Lref = (u5[2:, 1:-1] + u5[:-2, 1:-1] + u5[1:-1, 2:] + u5[1:-1, :-2] - 4 * u5[1:-1, 1:-1]) * 256. ** 2
    den = float(np.sqrt(np.mean(Lref ** 2)))
    pts, nrm = wall_nodes()
    d = np.concatenate([dnn_graddiff(NB.features_grad, P, pts[s:s + 255], nrm[s:s + 255]) @ c for s in range(0, len(pts), 255)])
    stat = float(np.sqrt(np.mean(d ** 2)) / den)
    X = lattice()
    dec = np.concatenate([NB.features_grad(P, X[s:s + 4096])[0] @ c for s in range(0, len(X), 4096)]).reshape(257, 257)
    assert np.linalg.norm(dec - u5) / np.linalg.norm(np.asarray(np.load(REFT)['a2_L8192'])[0]) < .05   # decode sanity
    print('K1 must-fail statistic', stat)
    assert stat > 1e-3, stat


def test_rom_path_resolves_the_dispatching_sep_common():
    import subprocess
    code = ("import sys; sys.argv=['x']; sys.path.insert(0, %r); import fpolicy, archrun; archrun.pin(); "
            "import p1common as P; import sep_common as sc, engines; "
            "print(sc.__file__); print(P.Q.sc.__file__); print(engines.sc.__file__)") % str(ROM)
    out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=True,
                         env=dict(__import__('os').environ, JAX_PLATFORMS='cpu')).stdout.split()
    assert all(o.startswith(str(ROM / 'deps')) for o in out), out
