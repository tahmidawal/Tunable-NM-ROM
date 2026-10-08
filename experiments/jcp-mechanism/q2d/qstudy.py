"""quadrature-study driver: one allocation per (mesh, cohort set). DESIGN.md is the specification.

jcp-mechanism copy (2026-10-08) of vendor/quad2d/qstudy.py. Changes, and nothing else: (1) Q.offmesh_rule receives the
mesh (for the A1 rule `nodes`); (2) the `nodes` gates G1-G3 of the lane DESIGN section 5 run once per setting before
the rollouts when a `nodes` arm is configured; (3) the timing phase is skipped when timing.skip is set (A1 does not
time); (4) the `nodes` arm's reached states are kept and rho of every configured rule is evaluated on those of the first
`nodes_reached_rho_cases` cases (DESIGN amendment A2-5); (5) the cohort-overlap check is unchanged.

Phases (setting-major, so only one setting's rule blocks are resident at a time)
  0  setup: cohorts (expected hashes, pairwise disjointness), refined references (validated against the reference job's
     manifest: complete, every case accepted, sha256 of every restricted field), frozen model, rotated bank on the mesh
     (nested column blocks 128 | 384 | 512), operators for M in {1536, 512, 64}; optional extra timing meshes
  T  same-grid truth per case: `fft_tight` through the historical path (iterative_paths.make_fom 'fft', ntol 1e-6,
     ltol 1e-8 -- the truth of the parity source jobs), kept on the host
  per setting:
  1  every rollout arm on every case: same-grid error (full mesh and the 257^2 shared nodes), refined-reference errors
     (ST, S) on the shared nodes, distance from the dense rollout and from the continuum rollout `gref` (full mesh and
     shared nodes), LM iterations / exits / rejections, query seconds. lat64's and gref's internal states k = 1..50
     are kept (rho population; gref-trajectory convergence check).
  2  rho ladder on the lat64 population against the continuum target (gref rule) and the mesh target (dense);
     continuum-target convergence (G6) on both populations
  3  timing: every subject compiled and warmed first; randomised order; a pre-compiled 0.1 s burn before every
     invocation; block_until_ready; each timed ROM output's restricted sha256 compared with phase 1, each timed FOM
     output scored against the truth. Subjects: the setting's timed arms, the decode alone, two FOM settings, and the
     same arms at the extra timing meshes (cross-mesh cost panel on ONE GPU, DESIGN B4).
Outputs result.json (+ npz side files) in --out. Acceptance (G1-G8) is decided by audit_qs.py, not here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import qcore as Q
import iterative_paths as IP

EDGES = [0, 128, 384, 512]
REASONS = {0: 'budget', 1: 'tol', 2: 'tiny_step', 3: 'damping_limit', 4: 'stationary'}
EXPECTED = {'test64': 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}
_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    """engines.burn with a cached, pre-compiled kernel (the clock starts after compilation)."""
    a = jnp.ones((512, 512), jnp.float64) * .01
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = _BURN(a)
        jax.block_until_ready(a)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def evolved_max(x):
    return float(np.max(np.asarray(x)[1:]))


class Mesh:
    """Everything mesh-dependent: rotated bank, operators, per-setting base data."""

    def __init__(self, mdl, L, settings, nrb=None):
        self.L = L
        Rb = max(Q.SETTINGS[s]['Rp'] for s in settings)
        self.edges = [e_ for e_ in EDGES if e_ <= Rb]
        t0 = time.perf_counter()
        self.Grot = Q.build_rotated_bank(mdl, L, self.edges, nrb=nrb)
        self.bank_seconds = time.perf_counter() - t0
        self.bank_bytes = int(sum(c.nbytes for rb in self.Grot for c in rb))
        self.ops = {}
        for s in settings:
            M = Q.SETTINGS[s]['M']
            if M not in self.ops:
                self.ops[M] = Q.operators(self.Grot, self.edges, L, M)

    def base(self, s):
        Rp, M = Q.SETTINGS[s]['Rp'], Q.SETTINGS[s]['M']
        o = self.ops[M]
        return dict(A=o['Arot'][:, :Rp], lam=o['lam'], G=Q.BK.prefix(self.Grot, self.edges, Rp))


def nodes_gates(mdl, mesh, s, Rp, M, blocks):
    """DESIGN section 5 gates for the `nodes` rule at this mesh and setting (relative max-abs differences).
    G1 analytic x+y derivative vs centred differences at 32 nodes; G2a nodes values G(x_i) c vs the mesh bank decode
    (4 states); G2b L w psi vs Phi at 4096 nodes; G2c the off-mesh GEMM N(c) vs the separable Phi^T (u (u_x + u_y));
    G3 the analytic Jacobian vs jax.jacfwd of the value."""
    L = mesh.L
    o = mesh.ops[M]
    d, m = blocks(mesh, 'point', 'nodes')
    X, w = Q.offmesh_rule('nodes', L)
    def rel(a, b):
        """Amendment A3-2: inf unless both finite and max|b| >= 1e-8; else max|a - b| / max|b|."""
        a, b = np.asarray(a), np.asarray(b)
        if not (np.isfinite(a).all() and np.isfinite(b).all() and np.abs(b).max() >= 1e-8):
            return float('inf')
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    g = {}
    sel = np.linspace(0, len(X) - 1, 32).astype(int)
    hfd = 1e-5
    _, gx, gy = mdl.values_grads(X[sel], Rp)
    vxp = mdl.values_grads(X[sel] + [hfd, 0.], Rp)[0]
    vxm = mdl.values_grads(X[sel] - [hfd, 0.], Rp)[0]
    vyp = mdl.values_grads(X[sel] + [0., hfd], Rp)[0]
    vym = mdl.values_grads(X[sel] - [0., hfd], Rp)[0]
    g['G1_derivative_vs_fd'] = rel((vxp - vxm + vyp - vym) / (2 * hfd), gx + gy)
    rng = np.random.default_rng(20261008)
    cs = jnp.asarray(rng.normal(size=(4, Rp)) / np.sqrt(Rp))
    base = mesh.base(s)
    off, dmax, rmax = 0, 0., 0.                         # G2a: the nodes value block equals the mesh bank, block by block
    for rb in base['G']:
        blk = jnp.concatenate(rb, axis=1)               # (rows, R') of the nested rotated mesh bank
        dmax = max(dmax, float(jnp.max(jnp.abs(d['Gq'][off:off + blk.shape[0]] - blk))))
        rmax = max(rmax, float(jnp.max(jnp.abs(blk))))
        off += blk.shape[0]
    assert off == d['Gq'].shape[0], (off, d['Gq'].shape)
    g['G2a_values_vs_mesh_bank'] = dmax / rmax if (rmax >= 1e-8 and np.isfinite(dmax)) else float('inf')
    u_nodes = cs @ d['Gq'].T
    ii = np.linspace(0, len(X) - 1, 4096).astype(int)
    ij = np.stack(np.unravel_index(ii, (L - 1, L - 1)), 1) + 1
    g['G2b_Psi_vs_Phi'] = rel(d['Psi'][jnp.asarray(ii)], Q.H.phi_rows(L, o['kx'], o['ky'], ij))
    val = jax.jit(jax.vmap(Q.tested_value('point', L), in_axes=(0, None)))(cs, d)
    sep = Q.H.sep_project((u_nodes * (cs @ d['Gs'].T)).reshape(-1, L - 1, L - 1), o['sx'], o['sy'], L)
    g['G2c_gemm_vs_separable'] = rel(val, sep)
    J_f = Q.tested_jac('point', L)(cs[0], d, None)[1]
    J_ad = jax.jacfwd(lambda c_: Q.tested_value('point', L)(c_, d))(cs[0])
    g['G3_jacobian_vs_jacfwd'] = rel(J_f, J_ad)
    g['m'] = int(m)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    L, dt = int(cfg['mesh']), .005
    s256 = max(1, L // 256)     # stride to the 257^2 nodes shared by every mesh (L >= 256)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, jax_version=jax.__version__, mesh=L,
               checkpoint_sha256=sha_file(Q.CKPT), rotation_sha256=sha_file(Q.ROTATION), eq_rule_sha256=sha_file(Q.EQ_RULE),
               cohorts={}, references={}, setup={}, rows=[], fom_rows=[], rho={}, gates={}, timing={}, complete=False)
    save = lambda: json.dump(clean(rep), open(out / 'result.json', 'w'), indent=1)
    burn(.05)

    # ------------------------------------------------------------- cohorts ----
    train = Q.e.params_draw(0, 128)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32', 'test64')}
    for k1 in allc:
        assert not any(np.allclose(t, s) for t in train for s in allc[k1]), f'{k1} overlaps training'
        for k2 in allc:
            if k1 < k2:
                assert not any(np.allclose(t, s) for t in allc[k1] for s in allc[k2]), f'{k1}/{k2} overlap'
    cases = []
    for coh in cfg['cohorts']:
        ph = allc[coh]
        h = sha(ph)
        if coh in EXPECTED and not cfg.get('local_smoke_waives_cohort_hash'):
            assert h == EXPECTED[coh], (coh, h)
        rep['cohorts'][coh] = dict(n=len(ph), physical_sha256=h, physical=ph.tolist())
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    print('CASES', len(cases), el(), flush=True)

    # ---------------------------------------------------------- references ----
    refs = {}
    if cfg.get('refs'):
        rdir = Path(cfg['refs'])
        if not rdir.is_absolute():                      # jcp-mechanism: staged references live under TASK_ROOT
            rdir = Path(os.environ['TASK_ROOT']) / rdir
        man = json.loads((rdir / 'result.json').read_text())
        assert man.get('complete') and man.get('all_accepted'), 'reference job incomplete or rejected'
        want = cfg['ref_contract']           # DESIGN 4: mesh 8192; ST dt/16, S dt; ntol 1e-11, ltol 1e-9, accept 2e-11
        assert int(man['config']['mesh']) == want['mesh'], ('reference mesh', man['config']['mesh'])
        for tag, w_ in want['refs'].items():
            got = man['config']['refs'][tag]
            assert all(abs(float(got[k_]) - float(w_[k_])) <= 1e-15 * max(1., abs(float(w_[k_]))) for k_ in w_), (tag, got)
        keys_ = [(x['cohort'], x['case'], x['ref']) for x in man['cases']]
        assert len(keys_) == len(set(keys_)), 'duplicate reference entries'
        idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
        for coh, c, ph in cases:
            assert man['cohort_sha256'][coh] == rep['cohorts'][coh]['physical_sha256'], ('reference cohort', coh)
            for tag in ('ST', 'S'):
                ent = idx.get((coh, c, tag))
                assert ent is not None and ent['accepted'], ('reference missing or not accepted', coh, c, tag)
                assert ent['mesh'] == want['mesh'] and abs(ent['dt'] - want['refs'][tag]['dt']) <= 1e-15, ent
                assert ent['max_relative_residual'] <= want['refs'][tag]['accept_residual'], ent
                r = np.load(rdir / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
                assert r.shape == (6, 257, 257) and np.isfinite(r).all(), ('reference shape', coh, c, tag, r.shape)
                assert sha(r) == ent['f257_sha256'], ('reference sha', coh, c, tag)
                refs[(coh, c, tag)] = r
                rep['references'].setdefault(tag, {})[f'{coh}{c}'] = dict(file=str(rdir / f'ref_{tag}_{coh}_{c:03d}.npz'),
                                                                         f257_sha256=ent['f257_sha256'])
        rep['references_job'] = dict(dir=str(rdir), job_id=man.get('job_id'), commit=man.get('commit'))
    else:
        assert cfg.get('allow_missing_refs'), 'references required'
    print('REFS', len(refs), el(), flush=True)

    # ---------------------------------------------------------- model, meshes ----
    mdl = Q.Model()
    mesh = Mesh(mdl, L, cfg['settings'], nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(columns=mesh.edges[-1], edges=mesh.edges, seconds=mesh.bank_seconds, bytes=mesh.bank_bytes)
    extra = {Lx: Mesh(mdl, Lx, cfg['settings']) for Lx in cfg.get('timing_meshes', [])}
    print('BANK+OPS', el(), flush=True)
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))

    # --------------------------------------------------------------- truth ----
    fom_hist, fom_pre = IP.make_fom(L, dt, 'fft')
    fom_lean = Q.e.make_fom(L, dt)[0]
    tr = cfg['fom']['truth']
    truth = {}
    au = cfg.get('audit', {})
    for ci, (coh, c, ph) in enumerate(cases):
        u0 = jnp.asarray(Q.e.initial(L, ph))
        t1 = time.perf_counter()
        v = fom_hist(u0, float(ph[4]), tr['ntol'], tr['ltol'], *fom_pre)
        f = np.asarray(jax.block_until_ready(v[0]))
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and rn.max() <= tr['ntol'] * (1 + 1e-9), ('truth', coh, c)
        u0n = np.asarray(u0)
        fr = f[:, ::s256, ::s256]
        n0, n0r = float(np.linalg.norm(u0n)), float(np.linalg.norm(u0n[::s256, ::s256]))
        truth[(coh, c)] = dict(full=f, r=fr, n0=n0, n0r=n0r)
        row = dict(cohort=coh, case=c, name=tr['name'], seconds=time.perf_counter() - t1,
                   max_relative_residual=float(rn.max()), newton_iterations=int(np.sum(np.asarray(v[1]))),
                   restricted_sha256=sha(fr))
        for tag in ('ST', 'S'):
            r = refs.get((coh, c, tag))
            if r is not None:
                pe = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, r)]
                row[f'ref_{tag}_per_time'] = pe
                row[f'ref_{tag}_evolved'] = evolved_max(pe)
        rep['fom_rows'].append(row)
        if coh == au.get('cohort') and c in au.get('cases', []):
            full_ok = L <= au.get('full_max_mesh', 256) or (L <= au.get('full_case0_max_mesh', 0) and c == au['cases'][0])
            np.savez_compressed(out / f'audit_truth_{coh}{c}.npz', f257=fr, **({'full': f} if full_ok else {}))
    rep['phases'] = dict(truth=el())
    print('TRUTH', el(), flush=True)
    save()

    # ------------------------------------------------------- per setting ----
    tcfg = cfg['timing']
    rng = np.random.default_rng(int(tcfg.get('seed', 20261001)))
    tcases = [x for x in cases if x[0] == tcfg['cohort']][:tcfg['cases']]
    inv_all = []

    def run_setting(s):
        st = Q.SETTINGS[s]
        Rp, M = st['Rp'], st['M']
        if st['kind'] == 'lin':
            trust, _ = mdl.trust_linear(Rp)
            cold = Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T)
            pf = None
            cmap = lambda W: W
        else:
            trust = mdl.trust_head()
            pf = mdl.head_params(Rp)
            cold = Q.build_cold_head(mdl, pf, Zsub)
            hf = jax.jit(jax.vmap(lambda z, pf=pf: Q.sc.head(pf, z)))
            cmap = lambda W, hf=hf: np.asarray(hf(jnp.asarray(W)))
        rep['setup'][s] = dict(R_prime=Rp, M=M, kind=st['kind'], trust=trust)
        cache = {}

        def blocks(msh, kind, rule):
            key = (msh.L, kind, rule)
            if key not in cache:
                o = msh.ops[M]
                if kind == 'mesh':
                    ij, w = Q.mesh_rule(rule, msh.L)
                    cache[key] = (Q.mesh_data(mdl, Rp, ij, w, msh.L, o['kx'], o['ky']), len(ij))
                elif kind in ('point', 'flux'):
                    X, w = Q.offmesh_rule(rule, msh.L)
                    cache[key] = (Q.offmesh_data(mdl, Rp, X, w, msh.L, o['kx'], o['ky'], kind), len(X))
                elif kind == 'dense':
                    cache[key] = (dict(sx=o['sx'], sy=o['sy']), (msh.L - 1) ** 2)
                else:
                    raise ValueError(kind)
            return cache[key]

        def build(msh, spec):
            t1 = time.perf_counter()
            d, m = blocks(msh, spec['kind'], spec.get('rule'))
            data = dict(msh.base(s), **d)
            gtol = spec.get('gtol', cfg['gtol'])
            chunks = cfg.get('tangent_chunks', {}).get(s, 1) if spec['kind'] == 'dense' else 1
            if st['kind'] == 'lin':
                fq, dec = Q.make_linear_query(spec['kind'], Rp, msh.L, dt, trust, step_budget=cfg['step_budget'],
                                              gtol=gtol, tangent_chunks=chunks)
                call = lambda u, nu, fq=fq, data=data: fq(u, nu, data, cold)
                dcall = (lambda W, dec=dec, data=data: dec(W, data), Rp)
            else:
                fq, dec = Q.make_head_query(spec['kind'], pf, mdl.K, msh.L, dt, trust, step_budget=cfg['step_budget'],
                                            gtol=gtol, tangent_chunks=chunks)
                tab = Q.head_tables(pf, mdl.K, data, cold)
                call = lambda u, nu, fq=fq, data=data, tab=tab: fq(u, nu, data, cold, tab)
                dcall = (lambda W, dec=dec, data=data, tab=tab: dec(W, data, tab), mdl.K)
            rep['setup'].setdefault('arms', []).append(dict(setting=s, mesh=msh.L, arm=spec['name'], kind=spec['kind'],
                                                            rule=spec.get('rule'), m=m, gtol=gtol,
                                                            build_seconds=time.perf_counter() - t1))
            return dict(spec=spec, call=call, dcall=dcall, m=m, gtol=gtol)

        arms = {spec['name']: build(mesh, spec) for spec in cfg['arms'][s]}
        print('ARMS', s, len(arms), el(), flush=True)
        if any(sp.get('rule') == 'nodes' for sp in cfg['arms'][s]):
            g = nodes_gates(mdl, mesh, s, Rp, M, blocks)
            rep['gates'][f'nodes_{s}'] = g
            print('NODES GATES', s, g, flush=True)
            save()
            assert g['G1_derivative_vs_fd'] < 1e-6 and g['G2a_values_vs_mesh_bank'] < 1e-12, g
            assert g['G2b_Psi_vs_Phi'] < 1e-12 and g['G2c_gemm_vs_separable'] < 1e-11, g
            assert g['G3_jacobian_vs_jacfwd'] < 1e-12, g

        # ---------------------------------------------- phase 1: rollouts ----
        pops = dict(lat64=[], gref=[], nodes=[])
        labels = dict(lat64=[], gref=[], nodes=[])
        sha1 = {}
        order = ([n for n in arms if n == 'gref'] + [n for n in arms if n == 'dense'] +
                 [n for n in arms if n not in ('gref', 'dense')])
        for ci, (coh, c, ph) in enumerate(cases):
            u0 = jnp.asarray(Q.e.initial(L, ph))
            nu = float(ph[4])
            T = truth[(coh, c)]
            n0, n0r = T['n0'], T['n0r']
            Tf = jnp.asarray(T['full'])
            dense_ok = (cfg['dense_cases'].get(coh) == 'all' or
                        (isinstance(cfg['dense_cases'].get(coh), list) and c in cfg['dense_cases'][coh]))
            keepf = {}
            for name in order:
                arm = arms[name]
                if arm['spec'].get('cohorts') and coh not in arm['spec']['cohorts']:
                    continue
                if name == 'dense' and not dense_ok:
                    continue
                t2 = time.perf_counter()
                v = arm['call'](u0, nu)
                jax.block_until_ready(v['fields'])
                secs = time.perf_counter() - t2
                f = v['fields']
                fr = np.asarray(f[:, ::s256, ::s256])
                it, reason, rej = (np.asarray(v[k_]) for k_ in ('it', 'reason', 'rej'))
                sha1[(name, coh, c)] = sha(fr)
                row = dict(setting=s, arm=name, kind=arm['spec']['kind'], rule=arm['spec'].get('rule'), m=arm['m'],
                           gtol=arm['gtol'], cohort=coh, case=c, finite=bool(jnp.all(jnp.isfinite(f))),
                           seconds_first=secs, iterations_total=int(it.sum()), iterations_max=int(it.max()),
                           exits={REASONS[k_]: int(np.sum(reason == k_)) for k_ in REASONS},
                           rejected_total=int(rej.sum()), restricted_sha256=sha1[(name, coh, c)])
                sg = [float(x) for x in jnp.linalg.norm((f - Tf).reshape(6, -1), axis=1) / n0]
                row['same_grid_per_time'] = sg
                row['same_grid_evolved'] = evolved_max(sg)
                row['same_grid_restricted_evolved'] = evolved_max(
                    [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, T['r'])])
                for tag in ('ST', 'S'):
                    r = refs.get((coh, c, tag))
                    if r is not None:
                        pe = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, r)]
                        row[f'ref_{tag}_per_time'] = pe
                        row[f'ref_{tag}_evolved'] = evolved_max(pe)
                for other in ('dense', 'gref'):
                    if other in keepf and other != name:
                        pt = [float(x) for x in jnp.linalg.norm((f - keepf[other][0]).reshape(6, -1), axis=1) / n0]
                        row[f'vs_{other}_per_time'] = pt
                        row[f'vs_{other}_evolved'] = evolved_max(pt)
                        row[f'vs_{other}_restricted_evolved'] = evolved_max(
                            [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, keepf[other][1])])
                if name in ('dense', 'gref'):
                    keepf[name] = (f, fr)
                rep['rows'].append(row)
                if name in pops:
                    W = np.asarray(v['internal'])[1:]
                    pops[name].append(cmap(W))
                    labels[name] += [(coh, c, k + 1) for k in range(len(W))]
                if coh == au.get('cohort') and c in au.get('cases', []):
                    full_ok = name in au.get('full_arms', []) and (
                        L <= au.get('full_max_mesh', 256) or
                        (L <= au.get('full_case0_max_mesh', 0) and c == au['cases'][0] and s == 'acc'))
                    np.savez_compressed(out / f'audit_{s}_{name}_{coh}{c}.npz', f257=fr,
                                        internal=np.asarray(v['internal']), **({'full': np.asarray(f)} if full_ok else {}))
                del v, f
            keepf.clear()
            del Tf
            print('CASE', s, ci, coh, c, el(), flush=True)
            if ci % 8 == 7:
                save()
        save()

        # -------------------------------------------------- phase 2: rho ----
        t1 = time.perf_counter()
        C = np.concatenate(pops['lat64'])
        Cg = np.concatenate(pops['gref']) if pops['gref'] else None
        np.savez_compressed(out / f'population_{s}.npz', C=C, labels=np.array([f'{a_}|{b_}|{k_}' for a_, b_, k_ in labels['lat64']]),
                            **({'Cgref': Cg, 'labels_gref': np.array([f'{a_}|{b_}|{k_}' for a_, b_, k_ in labels['gref']])}
                               if Cg is not None else {}))

        def values(Cs, kind, rule, chunk):
            """Tested advection of every state; off-mesh rules streamed over point chunks (sum split, rule unchanged)."""
            Cj = jnp.asarray(Cs)
            f = jax.jit(jax.vmap(Q.tested_value(kind, L), in_axes=(0, None)))
            if kind in ('point', 'flux'):
                X, w = Q.offmesh_rule(rule, L)
                o = mesh.ops[M]
                acc = np.zeros((len(Cs), M))
                pc = cfg.get('point_chunk', 32768)
                for q0 in range(0, len(X), pc):
                    d = Q.offmesh_data(mdl, Rp, X[q0:q0 + pc], w[q0:q0 + pc], L, o['kx'], o['ky'], kind)
                    acc += np.concatenate([np.asarray(f(Cj[i:i + chunk], d)) for i in range(0, len(Cs), chunk)])
                    del d
                return acc, len(X)
            d, m = blocks(mesh, kind, rule)
            data = dict(mesh.base(s), **d)
            return np.concatenate([np.asarray(f(Cj[i:i + chunk], data)) for i in range(0, len(Cs), chunk)]), m

        rho = lambda V, T_: np.linalg.norm(V - T_, axis=1) / np.maximum(np.linalg.norm(T_, axis=1), 1e-300)
        Tm, _ = values(C, 'dense', None, cfg.get('dense_chunk', 8))
        Tc, _ = values(C, 'point', cfg['gref'], 64)
        g6 = {}
        for pname, Cs, Tref in (('lat64', C, Tc), ('gref', Cg, None)):
            if Cs is None:
                continue
            Tref = Tref if Tref is not None else values(Cs, 'point', cfg['gref'], 64)[0]
            Tk = values(Cs, 'point', cfg['gref_check'], 64)[0]
            Tf_ = values(Cs, 'flux', cfg['gref'], 64)[0]
            ck, cf = rho(Tref, Tk), rho(Tf_, Tref)
            g6[pname] = dict(states=len(Cs), check_rho_max=float(ck.max()), check_rho_median=float(np.median(ck)),
                             flux_rho_max=float(cf.max()), flux_rho_median=float(np.median(cf)))
        rep['gates'][f'continuum_target_converged_{s}'] = dict(
            gref=cfg['gref'], check=cfg['gref_check'], bar=cfg['gref_bar'], populations=g6,
            passed=bool(all(v_['check_rho_max'] <= cfg['gref_bar'] and v_['flux_rho_max'] <= cfg['gref_bar'] for v_ in g6.values())))
        res, perstate = {}, {}
        lab = labels['lat64']
        for spec in cfg['rho_rules'][s]:
            if spec['kind'] == 'dense':
                V, m = Tm, (L - 1) ** 2
            else:
                V, m = values(C, spec['kind'], spec.get('rule'), 64 if spec['kind'] != 'mesh' else 16)
            r_c, r_m = rho(V, Tc), rho(V, Tm)
            perstate[spec['name']] = np.stack((r_c, r_m))
            ent = dict(kind=spec['kind'], rule=spec.get('rule'), m=m)
            for tag, r_ in (('cont', r_c), ('mesh', r_m)):
                i = int(np.argmax(r_))
                ent[tag] = dict(max=float(r_.max()), p95=float(np.quantile(r_, .95)), median=float(np.median(r_)),
                                argmax=list(lab[i]), pass_primary=bool(r_.max() <= .116), pass_tight=bool(r_.max() <= .06))
            res[spec['name']] = ent
        np.savez_compressed(out / f'rho_per_state_{s}.npz', **perstate)
        # jcp-mechanism A2-5: rho of every configured rule on the nodes-reached states of the first k cases
        if pops['nodes'] and cfg.get('nodes_reached_rho_cases'):
            kc = cfg['nodes_reached_rho_cases']
            Cn = np.concatenate(pops['nodes'][:kc])
            Tcn = values(Cn, 'point', cfg['gref'], 64)[0]
            Tmn = values(Cn, 'dense', None, cfg.get('dense_chunk', 8))[0]
            Tkn = values(Cn, 'point', cfg['gref_check'], 64)[0]
            rn = dict(states=len(Cn), cases=kc, check_rho_max=float(rho(Tkn, Tcn).max()), rules={})
            for spec in cfg['rho_rules'][s]:
                Vn = Tmn if spec['kind'] == 'dense' else values(Cn, spec['kind'], spec.get('rule'),
                                                                64 if spec['kind'] != 'mesh' else 16)[0]
                rc_, rm_ = rho(Vn, Tcn), rho(Vn, Tmn)
                rn['rules'][spec['name']] = dict(cont_max=float(rc_.max()), cont_median=float(np.median(rc_)),
                                                 mesh_max=float(rm_.max()))
            rn['check_valid'] = bool(rn['check_rho_max'] <= cfg['gref_bar'])
            rep.setdefault('rho_nodes_reached', {})[s] = rn
            print('RHO-NODES-REACHED', s, {k_: v_['cont_max'] for k_, v_ in rn['rules'].items()}, flush=True)
        rep['rho'][s] = dict(states=len(C), population_arm='lat64', seconds=time.perf_counter() - t1,
                             mesh_vs_continuum_gap=dict(max=float(rho(Tm, Tc).max()), median=float(np.median(rho(Tm, Tc)))),
                             rules=res)
        print('RHO', s, len(C), round(time.perf_counter() - t1, 1), el(), flush=True)
        save()

        # ------------------------------------------------ phase 3: timing ----
        if tcfg.get('skip'):
            return
        subjects = []
        for name in tcfg['arms'][s]:
            if name in arms:
                subjects.append(('rom', L, name, arms[name]))
        subjects.append(('decode', L, None, next(iter(arms.values()))))
        xarms = tcfg.get('extra_mesh_arms', {}).get(s, [])
        for Lx, mx in extra.items():
            built_x = None
            for name in xarms:
                spec = next(x for x in cfg['arms'][s] if x['name'] == name)
                built_x = build(mx, spec)
                subjects.append(('rom', Lx, name, built_x))
            if built_x is not None:
                subjects.append(('decode', Lx, None, built_x))
        for fs in cfg['fom']['timed']:
            subjects.append(('fom', L, fs['name'], fs))
        inputs = {Lx: [(jnp.asarray(Q.e.initial(Lx, ph)), float(ph[4]), (coh, c)) for coh, c, ph in tcases]
                  for Lx in [L] + list(extra)}

        def fn_of(sj, x):
            kind, Lx, name, obj = sj
            u0, nu, _ = inputs[Lx][x]
            if kind == 'rom':
                return lambda: obj['call'](u0, nu)
            if kind == 'decode':
                dfn, dd = obj['dcall']
                Wd = jnp.asarray(.1 * np.random.default_rng(x).normal(size=(6, dd)))
                return lambda: dfn(Wd)
            return lambda: fom_lean(u0, nu, obj['ntol'], obj['ltol'])

        base_sha = {}
        for j, sj in enumerate(subjects):    # compile + warm every signature before any timed invocation
            for x in range(len(tcases)):
                o = fn_of(sj, x)()
                jax.block_until_ready(o)
                if sj[0] == 'rom':
                    sx_ = max(1, sj[1] // 256)
                    base_sha[(j, x)] = sha(np.asarray(o['fields'][:, ::sx_, ::sx_]))
                del o
        inv = []
        for rep_i in range(tcfg['reps']):
            orderT = [(j, x) for j in range(len(subjects)) for x in range(len(tcases))]
            rng.shuffle(orderT)
            for j, x in orderT:
                sj = subjects[j]
                fn = fn_of(sj, x)
                burn(tcfg['burn'])
                t1 = time.perf_counter()
                o = fn()
                jax.block_until_ready(o)
                secs = time.perf_counter() - t1
                coh, c = inputs[sj[1]][x][2]
                ent = dict(setting=s, kind=sj[0], mesh=sj[1], name=sj[2], case=f'{coh}{c}', rep=rep_i, seconds=secs)
                if sj[0] == 'rom':
                    sx_ = max(1, sj[1] // 256)
                    h_ = sha(np.asarray(o['fields'][:, ::sx_, ::sx_]))
                    ent['output_matches_warmup'] = bool(h_ == base_sha[(j, x)])
                    if sj[1] == L:
                        ent['output_matches_phase1'] = bool(h_ == sha1.get((sj[2], coh, c)))
                elif sj[0] == 'fom':
                    T = truth[(coh, c)]
                    f = np.asarray(o[0])
                    ent['same_grid_evolved'] = evolved_max([float(np.linalg.norm(a_ - b_)) / T['n0'] for a_, b_ in zip(f, T['full'])])
                    ent['converged'] = bool(np.asarray(o[2]).max() <= sj[3]['ntol'] * (1 + 1e-9))
                inv.append(ent)
                del o
        inv_all.extend(inv)
        med = {}
        for d_ in inv_all:
            med.setdefault(f"{d_['setting']}|{d_['kind']}|{d_['mesh']}|{d_['name']}", []).append(d_['seconds'])
        rep['timing'] = dict(cases=[f'{a_}{b_}' for a_, b_, _ in tcases], reps=tcfg['reps'], burn=tcfg['burn'],
                             invocations=inv_all, median_ms={k: 1e3 * float(np.median(v)) for k, v in med.items()})
        print('TIMING', s, el(), flush=True)
        save()

    for s in cfg['settings']:
        run_setting(s)               # function scope: every rule block / compiled arm of s is released afterwards
        gc.collect()

    # jcp-mechanism: `complete` means execution completed; `targets_valid` records the required target checks
    rep['targets_valid'] = bool(all(v['passed'] for k, v in rep['gates'].items()
                                    if k.startswith('continuum_target_converged')) and
                                all(v.get('check_valid', True) for v in rep.get('rho_nodes_reached', {}).values()))
    rep['elapsed_seconds'] = el()
    rep['checkpoint_sha256_after'] = sha_file(Q.CKPT)
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('QSTUDY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
