"""jcp-smooth-bank evaluation of one or more banks (DESIGN.md §5 as amended by A1). One process = one GPU.

For every bank variant (checkpoint + rotation; `deployed` = the 2D lane's own Model/rotation/trust, else the lane
procedure of A1.2) and every linear setting (acc: R'=384, M=1536; fast: R'=128, M=512), at mesh L (1024):

  P1 rollouts of every arm on every dev6/val32 case (vendored qcore.make_linear_query, unchanged): worst evolved
     error vs the ST and S references on the 257^2 shared nodes, distance from the bank's own continuum rollout
     `gref` (Gauss 640^2), LM iterations/exits/residuals, internal coefficients (fields are their decode).
  P2 projection metrics (discrete, 257^2 nodes): LS projection of the S (and ST) reference fields on the rotated span,
     value error, gradient error vs central differences D2 (and D4), the FD-target uncertainty ||D2-D4||/||D4||.
  P3 rho ladders (continuum target Gauss 640^2, point form) on three populations: own `gref` rollout states,
     `lat64` rollout states (R1 replication), common states = the projections of P2; per-state rho for the full M
     and for the first 320 tests; target checks (Gauss 768^2, flux form). m*(b) with monotone confirmation.
  P4 rollouts of the m*_G(0.116) and m*_G(0.06) rules when not already run.
  P5 Chebyshev spectra of u and f = u (u_x + u_y) on 64 fixed own-population states (n = 256 and 512).
  C5a analytic vs finite-difference bank gradient. Controls C2, C3, C5b once per process.

    python bankeval.py --config jobs/<file>.json --task <i> --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / 'experiments/separable-decoder', ROOT / 'experiments/mr-burgers2d', HERE / 'vendor/quad2d/vendor',
           HERE / 'vendor/quad2d'):
    sys.path.insert(0, str(_p))

import numpy as np
import scipy.fft as sfft
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import qcore as Q          # vendored, unchanged (vendor/quad2d/qcore.py)
import qstudy as QS        # vendored, unchanged: Mesh, burn, sha, clean, evolved_max

SETTINGS = ('acc', 'fast')
GAUSS = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256)
FIB = (987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368)
E2E_FIXED = ('gauss32', 'gauss48', 'gauss64', 'gauss96', 'fib1597', 'fib4181', 'fib6765')
BARS = (0.116, 0.06, 0.01)
GREF, GCHK = 'gauss640', 'gauss768'
DT = .005


def log(*a):
    print(*a, flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------ model ----

class LaneModel(Q.Model):
    """Q.Model with the lane rotation (A1.2): T, L from the lane npz; trust radius from its LS training coefficients
    C (rows c_i) instead of h(Z_tr). Everything else (bank evaluation, gradients) is Q.Model's."""

    def __init__(self, ckpt, rotation):
        super().__init__(ckpt=ckpt, rotation=rotation)
        self.codes = np.asarray(np.load(rotation)['C'])

    def trust_linear(self, Rp):
        codes = self.codes @ self.Lrot[:Rp].T
        return .01 * float(np.max(np.linalg.norm(codes - codes.mean(0), axis=1))), None


def cold_codes(mdl, Rp, deployed):
    """Candidate codes carried by the cold tuple (unused by the linear initial fit, which is closed-form)."""
    if deployed:
        Z = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
        H = np.asarray(jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))(jnp.asarray(Z)))
        return H @ mdl.Lrot[:Rp].T
    return mdl.codes[::max(1, len(mdl.codes) // 8192)] @ mdl.Lrot[:Rp].T


# ------------------------------------------------------------------- cohorts ----

def cohorts_checked():
    """dev6 + val32 (qcore.cohort), asserted disjoint from both training draws (seed 0 x 576, seed 1000 x 4032)."""
    train = np.concatenate([Q.e.params_draw(0, 576), Q.e.params_draw(1000, 4032)])
    out = {}
    for k in ('dev6', 'val32'):
        ph = Q.cohort(k)
        for t in ph:
            assert not np.any(np.all(np.isclose(train, t[None], rtol=0, atol=1e-12), axis=1)), f'{k} overlaps training'
        out[k] = ph
    return out


def load_refs(rdir, cases, coh_sha, waive_cohort_hash=False):
    man = json.loads((Path(rdir) / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted'), 'reference job incomplete'
    assert int(man['config']['mesh']) == 8192
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    refs = {}
    for coh, c, _ in cases:
        assert waive_cohort_hash or man['cohort_sha256'][coh] == coh_sha[coh], ('reference cohort', coh)
        for tag in ('ST', 'S'):
            ent = idx[(coh, c, tag)]
            assert ent['accepted'] and ent['max_relative_residual'] <= 2e-11, ent
            r = np.load(Path(rdir) / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert r.shape == (6, 257, 257) and QS.sha(r) == ent['f257_sha256'], (coh, c, tag)
            refs[(coh, c, tag)] = r
    return refs, dict(dir=str(rdir), job_id=man.get('job_id'), commit=man.get('commit'))


# ---------------------------------------------------------- finite differences ----

def d2(F, h):
    """Central differences of fields F (..., n, n) incl. boundary rows; returns (Dx, Dy) on the interior."""
    return ((F[..., 2:, 1:-1] - F[..., :-2, 1:-1]) / (2 * h), (F[..., 1:-1, 2:] - F[..., 1:-1, :-2]) / (2 * h))


def d4_inner(F, h):
    """Fourth-order central differences at nodes >= 2 from the wall: (Dx, Dy) on [2:-2, 2:-2]."""
    dx = (-F[..., 4:, 2:-2] + 8 * F[..., 3:-1, 2:-2] - 8 * F[..., 1:-3, 2:-2] + F[..., :-4, 2:-2]) / (12 * h)
    dy = (-F[..., 2:-2, 4:] + 8 * F[..., 2:-2, 3:-1] - 8 * F[..., 2:-2, 1:-3] + F[..., 2:-2, :-4]) / (12 * h)
    return dx, dy


# ------------------------------------------------------------------ chebyshev ----

def cheb_points(n):
    return .5 * (1. + np.cos(np.pi * (np.arange(n) + .5) / n))


def cheb_envelope(F):
    """F (n, n) on the first-kind Chebyshev tensor points -> envelope E_j = max_{max(j',k)=j} |a_j'k| / max |a|."""
    n = F.shape[0]
    A = sfft.dctn(F, type=2) / (n * n)          # a_jk up to the j=0 / k=0 halving (does not change decay)
    A[0, :] *= .5
    A[:, 0] *= .5
    A = np.abs(A)
    j = np.maximum.outer(np.arange(n), np.arange(n))
    E = np.zeros(n)
    np.maximum.at(E, j.ravel(), A.ravel())
    return E / max(E.max(), 1e-300)


def n_eps(E, eps):
    """Smallest degree beyond which the envelope stays <= eps; None if it never falls below eps (unresolved)."""
    above = np.nonzero(E > eps)[0]
    if above.size and above.max() >= len(E) - 1:
        return None
    return int(above.max() + 1) if above.size else 0


def classify(E):
    """A2.3 (as amended in A3): block envelope B_k = max_{4k<=j<4k+4} E_j; fit ln B_k against the block's centre degree
    (geometric) and against its log (algebraic) over the blocks with j >= 4 and B_k > 1e-13; < 4 blocks -> unresolved."""
    nb = len(E) // 4
    B = np.array([E[4 * k:4 * k + 4].max() for k in range(nb)])
    jc = 4 * np.arange(nb) + 1.5
    keep = (jc >= 4) & (B > 1e-13)
    if keep.sum() < 4:
        return dict(kind='unresolved', blocks=int(keep.sum()))
    j, y = jc[keep], np.log(B[keep])
    rg = np.polyfit(j, y, 1, full=True)
    ra = np.polyfit(np.log(j), y, 1, full=True)
    sg = float(rg[1][0]) if len(rg[1]) else 0.
    sa = float(ra[1][0]) if len(ra[1]) else 0.
    kind = 'geometric' if sg < .5 * sa else ('algebraic' if sa < .5 * sg else 'inconclusive')
    return dict(kind=kind, blocks=int(keep.sum()), window=[float(j.min()), float(j.max())], rss_geometric=sg,
                rss_algebraic=sa, geometric_rate=float(np.exp(-rg[0][0])), algebraic_exponent=float(-ra[0][0]))


def spectra(fields_by_n):
    """fields_by_n: {n: (F_u (n,n), F_f (n,n))} -> envelopes, n_eps (with the 256-vs-512 agreement rule), class."""
    out = {}
    for q, name in ((0, 'u'), (1, 'f')):
        Es = {n: cheb_envelope(v[q]) for n, v in fields_by_n.items()}
        ent = {}
        for eps, key in ((1e-4, '1e-04'), (1e-8, '1e-08')):
            a, b = n_eps(Es[256], eps), n_eps(Es[512], eps)
            ent[f'n_{key}'] = b if (a is not None and b is not None and abs(a - b) <= 2) else None
            ent[f'n_{key}_256_512'] = [a, b]
        ent['class'] = classify(Es[512])
        ent['envelope_512'] = Es[512].tolist()
        out[name] = ent
    return out


def controls_once():
    """C2, C3 (Chebyshev estimator) and C5b (central-difference response) -- independent of any bank."""
    res = {}
    env = {}
    for nm, fn in (('bump_w0.2', lambda X, Y: np.exp(-((X - .43) ** 2 + (Y - .57) ** 2) / (2 * .2 ** 2))),
                   ('bump_w0.05', lambda X, Y: np.exp(-((X - .43) ** 2 + (Y - .57) ** 2) / (2 * .05 ** 2))),
                   ('kink', lambda X, Y: np.abs(X - .4) * np.exp(-((X - .5) ** 2 + (Y - .5) ** 2) / .02))):
        by_n = {}
        for n in (256, 512):
            x = cheb_points(n)
            X, Y = np.meshgrid(x, x, indexing='ij')
            F = fn(X, Y)
            by_n[n] = (F, F)
        env[nm] = spectra(by_n)['u']
    res['C2'] = dict(n8_w02=env['bump_w0.2']['n_1e-08'], n8_w005=env['bump_w0.05']['n_1e-08'],
                     class_w02=env['bump_w0.2']['class']['kind'], class_w005=env['bump_w0.05']['class']['kind'])
    res['C2']['passed'] = bool(res['C2']['n8_w02'] is not None and res['C2']['n8_w005'] is not None and
                               res['C2']['n8_w02'] < res['C2']['n8_w005'] and
                               res['C2']['class_w02'] == 'geometric' and res['C2']['class_w005'] == 'geometric')
    res['C3'] = dict(class_kink=env['kink']['class'], passed=bool(env['kink']['class']['kind'] == 'algebraic'))
    h = 1 / 256
    x = np.arange(257) * h
    X, Y = np.meshgrid(x, x, indexing='ij')
    F = np.sin(5 * np.pi * X) * np.sin(3 * np.pi * Y)
    Dx, Dy = d2(F, h)
    ex = (np.sin(5 * np.pi * h) / h) * np.cos(5 * np.pi * X) * np.sin(3 * np.pi * Y)
    ey = (np.sin(3 * np.pi * h) / h) * np.sin(5 * np.pi * X) * np.cos(3 * np.pi * Y)
    err = max(np.abs(Dx - ex[1:-1, 1:-1]).max(), np.abs(Dy - ey[1:-1, 1:-1]).max())
    res['C5b'] = dict(max_abs_error=float(err), passed=bool(err <= 1e-12))
    return res


# ------------------------------------------------------------------- one bank ----

def evaluate(var, L, cases, refs, out, cfg):
    tag = var['label']
    t0 = time.perf_counter()
    deployed = bool(var.get('deployed'))
    ckpt, rot = HERE / var['ckpt'] if not Path(var['ckpt']).is_absolute() else Path(var['ckpt']), var['rotation']
    rot = HERE / rot if not Path(rot).is_absolute() else Path(rot)
    mdl = Q.Model(ckpt=ckpt, rotation=rot) if deployed else LaneModel(ckpt, rot)
    rep = dict(label=tag, deployed=deployed, checkpoint=str(ckpt), checkpoint_sha256=sha_file(ckpt),
               rotation=str(rot), rotation_sha256=sha_file(rot), mesh=L, settings={}, complete=False)
    save = lambda: (out / f'eval_{tag}.json').write_text(json.dumps(QS.clean(rep), indent=1))
    mesh = QS.Mesh(mdl, L, list(SETTINGS))
    rep['bank_build_seconds'] = mesh.bank_seconds
    s256 = L // 256
    # C5a: analytic gradient vs central difference of the bank (step 1e-5) at 1000 random points, R' = 384
    Xr = np.random.default_rng(0).uniform(.02, .98, (1000, 2))
    v, gx, gy = (np.asarray(a) for a in mdl.values_grads(Xr, 384))
    errs = {}
    for dd in (1e-3, 1e-4, 1e-5):
        fdx = (np.asarray(mdl.values_grads(Xr + [dd, 0], 384)[0]) - np.asarray(mdl.values_grads(Xr - [dd, 0], 384)[0])) / (2 * dd)
        fdy = (np.asarray(mdl.values_grads(Xr + [0, dd], 384)[0]) - np.asarray(mdl.values_grads(Xr - [0, dd], 384)[0])) / (2 * dd)
        errs[f'{dd:g}'] = float(np.sqrt((np.linalg.norm(gx - fdx) ** 2 + np.linalg.norm(gy - fdy) ** 2)
                                        / (np.linalg.norm(gx) ** 2 + np.linalg.norm(gy) ** 2)))
    c5a = min(errs.values())
    rep['C5a'] = dict(rel_error_by_step=errs, passed=bool(c5a <= 1e-6))
    # interior nodes of the 257^2 reference grid and the bank there (for P2)
    xg = np.arange(1, 256) / 256
    Xg = np.stack(np.meshgrid(xg, xg, indexing='ij'), -1).reshape(-1, 2)

    for s in SETTINGS:
        st = Q.SETTINGS[s]
        Rp, M = st['Rp'], st['M']
        trust, _ = mdl.trust_linear(Rp)
        cold = Q.build_cold_linear(mdl, Rp, cold_codes(mdl, Rp, deployed))
        o = mesh.ops[M]
        lam_sorted = bool(np.all(np.diff(np.asarray(o['lam'])) >= -1e-9))
        sv_A = np.linalg.svd(np.asarray(mesh.base(s)['A']), compute_uv=False)
        R = dict(R_prime=Rp, M=M, trust=trust, tests_sorted_by_eigenvalue=lam_sorted, rows=[], rho={}, mstar={},
                 A_singular_values=dict(max=float(sv_A[0]), min=float(sv_A[-1])))
        rep['settings'][s] = R
        cache = {}

        def blocks(kind, rule):
            key = (kind, rule)
            if key not in cache:
                if kind == 'mesh':
                    ij, w = Q.mesh_rule(rule, L)
                    cache[key] = (Q.mesh_data(mdl, Rp, ij, w, L, o['kx'], o['ky']), len(ij))
                else:
                    X, w = Q.offmesh_rule(rule)
                    cache[key] = (Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], kind), len(X))
            return cache[key]

        def build(name):
            gtol = 1e-4 if name == 'gref_tight' else 1e-3
            rule_name = GREF if name in ('gref', 'gref_tight') else name
            kind, rule = ('mesh', 'lat64') if name == 'lat64' else ('point', rule_name)
            d, m = blocks(kind, rule)
            data = dict(mesh.base(s), **d)
            fq, dec = Q.make_linear_query(kind, Rp, L, DT, trust, step_budget=600, gtol=gtol)
            return dict(call=lambda u, nu, fq=fq, data=data: fq(u, nu, data, cold), m=m, kind=kind, rule=rule, gtol=gtol)

        pops, labels, coeffs = dict(gref=[], lat64=[]), dict(gref=[], lat64=[]), {}
        keep_gref, keep_full = {}, {}

        def rollout_arm(name, case_list):
            arm = build(name)
            for coh, c, ph in case_list:
                u0 = jnp.asarray(Q.e.initial(L, ph))
                t1 = time.perf_counter()
                v = arm['call'](u0, float(ph[4]))
                jax.block_until_ready(v['fields'])
                secs = time.perf_counter() - t1
                fr = np.asarray(v['fields'][:, ::s256, ::s256])
                full_finite = bool(jnp.all(jnp.isfinite(v['fields'])) & jnp.all(jnp.isfinite(v['internal'])))
                it, reason, rn = (np.asarray(v[k]) for k in ('it', 'reason', 'rn'))
                n0r = float(np.linalg.norm(np.asarray(u0)[::s256, ::s256]))
                row = dict(arm=name, kind=arm['kind'], rule=arm['rule'], m=arm['m'], cohort=coh, case=c, gtol=arm['gtol'],
                           finite=full_finite, seconds_first=secs, iterations_total=int(it.sum()),
                           iterations_max=int(it.max()), budget_exits=int(np.sum(reason == 0)),
                           nonaccepted_exits=int(np.sum((reason == 0) | (reason == 2) | (reason == 3))),
                           exits={QS.REASONS[k]: int(np.sum(reason == k)) for k in QS.REASONS},
                           residual_max=float(np.max(rn)), restricted_sha256=QS.sha(fr))
                n0c = float(np.linalg.norm(np.asarray(u0)[::2 * s256, ::2 * s256]))
                for tg in ('ST', 'S'):
                    pe = [float(np.linalg.norm(a - b)) / n0r for a, b in zip(fr, refs[(coh, c, tg)])]
                    row[f'ref_{tg}_per_time'] = pe
                    row[f'ref_{tg}_evolved'] = QS.evolved_max(pe)
                    pc = [float(np.linalg.norm(a[::2, ::2] - b[::2, ::2])) / n0c for a, b in zip(fr, refs[(coh, c, tg)])]
                    row[f'ref_{tg}_evolved_129'] = QS.evolved_max(pc)       # A2.6 common 129-node restriction
                if name == 'gref':
                    keep_gref[(coh, c)] = fr
                    if coh == 'dev6':
                        keep_full[(coh, c)] = v['fields']
                elif (coh, c) in keep_gref:
                    pt = [float(np.linalg.norm(a - b)) / n0r for a, b in zip(fr, keep_gref[(coh, c)])]
                    row['vs_gref_restricted_per_time'] = pt
                    row['vs_gref_restricted_evolved'] = QS.evolved_max(pt)
                    if (coh, c) in keep_full:                                  # qstudy's primary: full fields, full n0
                        n0 = float(jnp.linalg.norm(u0))
                        pf = [float(x) for x in jnp.linalg.norm((v['fields'] - keep_full[(coh, c)]).reshape(6, -1), axis=1) / n0]
                        row['vs_gref_per_time'] = pf
                        row['vs_gref_evolved'] = QS.evolved_max(pf)
                R['rows'].append(row)
                W = np.asarray(v['internal'])
                coeffs[f'{name}|{coh}|{c}'] = W
                if name in pops:
                    pops[name].append(W[1:])
                    labels[name] += [(coh, c, k + 1) for k in range(len(W) - 1)]
                del v
            del arm
            cache.clear()                     # bound device memory: rule blocks are rebuilt if needed again
            gc.collect()

        dev6 = [x for x in cases if x[0] == 'dev6']
        for name in ('gref', 'lat64') + E2E_FIXED:
            rollout_arm(name, cases)
            log(tag, s, 'ROLLOUT', name, round(time.perf_counter() - t0))
        rollout_arm(GCHK, dev6)
        rollout_arm('gref_tight', dev6)
        save()

        # ---------------------------------------------- P2: projections ----
        Gv, Gx, Gy = (np.asarray(a) for a in mdl.values_grads(Xg, Rp))
        Qg, Rg = np.linalg.qr(Gv)
        h = 1 / 256
        proj = {}
        common = []
        for tg in ('S', 'ST'):
            ev, eg, eg2c, eg4, fdu, lab = [], [], [], [], [], []
            for coh, c, _ in cases:
                F = refs[(coh, c, tg)]
                for t in range(F.shape[0]):
                    u = F[t, 1:-1, 1:-1].reshape(-1)
                    cf = np.linalg.solve(Rg, Qg.T @ u)
                    up = Gv @ cf
                    ev.append(np.linalg.norm(up - u) / np.linalg.norm(u))
                    Dx, Dy = d2(F[t], h)
                    gpx, gpy = (Gx @ cf).reshape(255, 255), (Gy @ cf).reshape(255, 255)
                    eg.append(np.sqrt(np.sum((gpx - Dx) ** 2 + (gpy - Dy) ** 2) / np.sum(Dx ** 2 + Dy ** 2)))
                    D4x, D4y = d4_inner(F[t], h)
                    Dxc, Dyc = Dx[1:-1, 1:-1], Dy[1:-1, 1:-1]
                    eg2c.append(np.sqrt(np.sum((gpx[1:-1, 1:-1] - Dxc) ** 2 + (gpy[1:-1, 1:-1] - Dyc) ** 2)
                                        / np.sum(Dxc ** 2 + Dyc ** 2)))
                    eg4.append(np.sqrt(np.sum((gpx[1:-1, 1:-1] - D4x) ** 2 + (gpy[1:-1, 1:-1] - D4y) ** 2)
                                       / np.sum(D4x ** 2 + D4y ** 2)))
                    fdu.append(np.sqrt(np.sum((Dx[1:-1, 1:-1] - D4x) ** 2 + (Dy[1:-1, 1:-1] - D4y) ** 2)
                                       / np.sum(D4x ** 2 + D4y ** 2)))
                    lab.append((coh, c, t))
                    if tg == 'S':
                        common.append(cf)
            q = lambda a: dict(median=float(np.median(a)), max=float(np.max(a)), p90=float(np.quantile(a, .9)))
            proj[tg] = dict(states=len(ev), value=q(ev), grad_D2=q(eg), grad_D2_common=q(eg2c), grad_D4=q(eg4),
                            fd_target_uncertainty=q(fdu), value_per_state=ev, grad_D2_per_state=eg,
                            grad_D2_common_per_state=eg2c, grad_D4_per_state=eg4, fd_per_state=fdu, labels=lab)
        R['projection'] = proj
        C_common = np.asarray(common)
        del Gv, Gx, Gy, Qg
        log(tag, s, 'PROJ value S med/max', proj['S']['value']['median'], proj['S']['value']['max'],
            'grad', proj['S']['grad_D2']['median'])
        save()

        # ---------------------------------------------- P3: rho ladders ----
        Cg = np.concatenate(pops['gref'])
        Cl = np.concatenate(pops['lat64'])
        popn = dict(gref=Cg, lat64=Cl, common=C_common)
        Call = np.concatenate([Cg, Cl, C_common])
        sl = dict(gref=slice(0, len(Cg)), lat64=slice(len(Cg), len(Cg) + len(Cl)),
                  common=slice(len(Cg) + len(Cl), len(Call)))
        Cj = jnp.asarray(Call)

        def values(kind, rule, chunk=64):
            f = jax.jit(jax.vmap(Q.tested_value(kind, L), in_axes=(0, None)))
            X, w = Q.offmesh_rule(rule)
            acc = np.zeros((len(Call), M))
            for q0 in range(0, len(X), 32768):
                d = Q.offmesh_data(mdl, Rp, X[q0:q0 + 32768], w[q0:q0 + 32768], L, o['kx'], o['ky'], kind)
                acc += np.concatenate([np.asarray(f(Cj[i:i + chunk], d)) for i in range(0, len(Call), chunk)])
                del d
            return acc, len(X)

        def rho(V, T_, k=None):
            V, T_ = (V, T_) if k is None else (V[:, :k], T_[:, :k])
            return np.linalg.norm(V - T_, axis=1) / np.maximum(np.linalg.norm(T_, axis=1), 1e-300)

        Tc, _ = values('point', GREF)
        Tk, _ = values('point', GCHK)
        Tf, _ = values('flux', GREF)
        tn = np.linalg.norm(Tc, axis=1)
        tn320 = np.linalg.norm(Tc[:, :320], axis=1)
        bad_ = lambda x: int(np.sum(~np.isfinite(x) | (x <= 0)))
        R['target_norm'] = {p: dict(min=float(tn[sl[p]].min()), zero_or_nonfinite=bad_(tn[sl[p]]),
                                    min_320=float(tn320[sl[p]].min()), zero_or_nonfinite_320=bad_(tn320[sl[p]]),
                                    states=int(sl[p].stop - sl[p].start)) for p in sl}
        np.savez_compressed(out / f'target_norms_{tag}_{s}.npz', full=tn, first320=tn320)
        R['C4'] = {p: dict(check_rho_max=float(rho(Tc, Tk)[sl[p]].max()), flux_rho_max=float(rho(Tf, Tc)[sl[p]].max()))
                   for p in sl}
        perstate = {}
        rules = [f'gauss{p}' for p in GAUSS] + [f'fib{n}' for n in FIB]
        for rule in rules:
            V, m = values('point', rule)
            r_full, r_320 = rho(V, Tc), rho(V, Tc, 320)
            perstate[rule] = np.stack((r_full, r_320))
            R['rho'][rule] = {p: dict(m=m, max=float(r_full[sl[p]].max()), median=float(np.median(r_full[sl[p]])),
                                      p95=float(np.quantile(r_full[sl[p]], .95)),
                                      max_320=float(r_320[sl[p]].max()), median_320=float(np.median(r_320[sl[p]])))
                              for p in sl}
        np.savez_compressed(out / f'rho_{tag}_{s}.npz', rules=np.array(rules), **perstate,
                            labels_gref=np.array([f'{a}|{b}|{k}' for a, b, k in labels['gref']]),
                            labels_lat64=np.array([f'{a}|{b}|{k}' for a, b, k in labels['lat64']]))
        np.savez_compressed(out / f'pop_{tag}_{s}.npz', **popn)
        for p in sl:
            for fam, ladder in (('G', [f'gauss{q}' for q in GAUSS]), ('F', [f'fib{n}' for n in FIB])):
                for b in BARS:
                    ms = None
                    for i, r_ in enumerate(ladder):
                        if all(R['rho'][x][p]['max'] <= b for x in ladder[i:]):
                            ms = R['rho'][r_][p]['m']
                            break
                    R['mstar'][f'{p}|{fam}|{b}'] = ms        # None = right-censored (> largest rung)
            # tail rate (A1.4): Gauss rungs with worst rho < 0.5 and median > 1e-12, >= 3 rungs
            pts = [(2 * q, R['rho'][f'gauss{q}'][p]['median']) for q in GAUSS
                   if R['rho'][f'gauss{q}'][p]['max'] < .5 and R['rho'][f'gauss{q}'][p]['median'] > 1e-12]
            if len(pts) >= 3:
                x_, y_ = np.array([a for a, _ in pts], float), np.log([b for _, b in pts])
                sl_, ic = np.polyfit(x_, y_, 1)
                r2 = 1 - np.sum((y_ - (sl_ * x_ + ic)) ** 2) / max(np.sum((y_ - y_.mean()) ** 2), 1e-300)
                R.setdefault('tail', {})[p] = dict(rungs=len(pts), slope=float(sl_), varrho=float(np.exp(-sl_)), r2=float(r2))
            else:
                R.setdefault('tail', {})[p] = dict(rungs=len(pts), resolved=False)
        log(tag, s, 'RHO done', {k: v for k, v in R['mstar'].items() if k.startswith('gref|G')}, round(time.perf_counter() - t0))
        save()

        # ------------------------------------------- P4: m* rule rollouts ----
        done_arms = {r_['arm'] for r_ in R['rows']}
        for b in (0.116, 0.06):
            ms = R['mstar'][f'gref|G|{b}']
            if ms is not None:
                name = f'gauss{int(round(np.sqrt(ms)))}'
                if name not in done_arms:
                    rollout_arm(name, cases)
                    done_arms.add(name)
        np.savez_compressed(out / f'coeffs_{tag}_{s}.npz', **coeffs)

        # ----------------------------------------------- P5: spectra ----
        pick = np.sort(np.random.default_rng(0).choice(len(Cg), min(64, len(Cg)), replace=False))
        spec_states = []
        by_state = {}
        for n in (256, 512):
            x = cheb_points(n)
            Xc = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
            Gv, Gx, Gy = (np.asarray(a) for a in mdl.values_grads(Xc, Rp))
            U = Cg[pick] @ Gv.T
            Fv = U * (Cg[pick] @ (Gx + Gy).T)
            for i in range(len(pick)):
                by_state.setdefault(i, {})[n] = (U[i].reshape(n, n), Fv[i].reshape(n, n))
            del Gv, Gx, Gy
        for i in range(len(pick)):
            spec_states.append(spectra(by_state[i]))
        agg = {}
        for q in ('u', 'f'):
            for eps in ('1e-04', '1e-08'):
                vals = [sp[q][f'n_{eps}'] for sp in spec_states]
                fin = [v_ for v_ in vals if v_ is not None]
                agg[f'{q}|n_{eps}'] = dict(median=float(np.median(fin)) if fin else None, max=max(fin) if fin else None,
                                           unresolved=int(sum(v_ is None for v_ in vals)))
            kinds = [sp[q]['class']['kind'] for sp in spec_states]
            agg[f'{q}|class'] = {k: kinds.count(k) for k in ('geometric', 'algebraic', 'inconclusive', 'unresolved')}
            agg[f'{q}|envelope_median'] = np.median([sp[q]['envelope_512'] for sp in spec_states], axis=0).tolist()
        R['spectra'] = dict(states=[labels['gref'][i] for i in pick], aggregate=agg,
                            note='bandwidth aggregates are over the states with a resolved n_eps (count of unresolved given)',
                            per_state=[{q: {k: v for k, v in sp[q].items() if k != 'envelope_512'} for q in ('u', 'f')}
                                       for sp in spec_states])
        np.savez_compressed(out / f'spectra_{tag}_{s}.npz',
                            **{f'{q}_{i}': np.asarray(sp[q]['envelope_512']) for i, sp in enumerate(spec_states) for q in ('u', 'f')})
        log(tag, s, 'SPECTRA', {k: v for k, v in agg.items() if 'envelope' not in k}, round(time.perf_counter() - t0))
        cache.clear()
        save()
        gc.collect()

    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - t0
    save()
    del mesh, mdl
    gc.collect()


def fom_comparators(cases, refs, nodes_list=(129, 257)):
    """A2.6: the training generator (deps/burgers2d_film.py, loaded by path so it cannot shadow other modules) at 129
    and 257 nodes per axis, scored on its own nodes and on the common 129-node restriction against the references."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bf_lane', HERE / 'deps/burgers2d_film.py')
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    out = []
    for n in nodes_list:
        rollout, _ = bf.make_rollout(n)
        st = 256 // (n - 1)
        for coh, c, ph in cases:
            U0 = bf.blob_ic(n, ph[0], ph[1], ph[2], ph[3])[None]
            snaps, res = rollout(jnp.asarray(U0), jnp.asarray([ph[4]]))
            F = np.asarray(snaps)[::10, 0].reshape(6, n, n)
            row = dict(nodes=n, cohort=coh, case=c, max_rel_residual=float(np.max(np.asarray(res))),
                       finite=bool(np.isfinite(F).all()))
            for tg in ('ST', 'S'):
                Rf = refs[(coh, c, tg)][:, ::st, ::st]
                n0 = float(np.linalg.norm(Rf[0]))
                row[f'ref_{tg}_evolved_own'] = QS.evolved_max([float(np.linalg.norm(a - b)) / n0 for a, b in zip(F, Rf)])
                k = (n - 1) // 128
                Fc, Rc = F[:, ::k, ::k], refs[(coh, c, tg)][:, ::2, ::2]
                n0c = float(np.linalg.norm(Rc[0]))
                row[f'ref_{tg}_evolved_129'] = QS.evolved_max([float(np.linalg.norm(a - b)) / n0c for a, b in zip(Fc, Rc)])
            out.append(row)
            log('FOM', n, coh, c, row['ref_S_evolved_129'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    log(f'jax_backend={jax.default_backend()} x64=True precision=highest gpu={jax.devices()[0].device_kind}')
    L = int(cfg['mesh'])
    allc = cohorts_checked()
    cases = []
    coh_sha = {}
    for coh in cfg['cohorts']:
        ph = allc[coh]
        coh_sha[coh] = QS.sha(ph)
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    refs, rinfo = load_refs(cfg['refs'], cases, coh_sha, bool(cfg.get('local_smoke_waives_cohort_hash')))
    meta = dict(config=cfg, task=a.task, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), jax_version=jax.__version__,
                cohort_sha256=coh_sha, references=rinfo, cases=len(cases), controls=controls_once())
    (out / f'meta_task{a.task}.json').write_text(json.dumps(QS.clean(meta), indent=1))
    log('CONTROLS', json.dumps(QS.clean({k: v.get('passed') for k, v in meta['controls'].items()})))
    if cfg['tasks'][a.task].get('fom_comparators'):
        meta['fom_comparators'] = fom_comparators(cases, refs)
        (out / f'meta_task{a.task}.json').write_text(json.dumps(QS.clean(meta), indent=1))
    for var in cfg['tasks'][a.task]['variants']:
        log('VARIANT', var['label'])
        evaluate(var, L, cases, refs, out, cfg)
    (out / f'COMPLETE_task{a.task}').write_text('complete\n')
    log('BANKEVAL COMPLETE')


if __name__ == '__main__':
    main()
