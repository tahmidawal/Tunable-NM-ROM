"""jcp-wide-bank J1: the 2D dial (1a) and the test-count trim (1d) with off-mesh rules. DESIGN.md is the specification.

The frozen 2D model dn256b (K=16, R=512, rotation_R512), linear rung only, on ONE mesh (config 'mesh'), for a grid of
settings (R', M). Everything model-side is the vendored, byte-identical code of the 2026-10-01 quadrature lane
(vendor/quad2d/qcore.py and its vendor/ modules), imported unchanged: the rotated nested bank, the exact discrete
linear terms, Hari's off-mesh 'point' form and the Table-1 linear-rung LM query `make_linear_query`. This driver only
changes WHICH (R', M) and WHICH rules are run, and what is measured:

  setup   cohorts (pairwise-disjoint, disjoint from training), refined references (manifest + contract + sha256),
          rotated bank on the mesh with nested column blocks at every R' of the config, operators at M_max (the
          M lowest modes are nested: M-prefixes of the M_max table are exactly modes_lean(L, M)).
  per R' (largest first; rule blocks built once at (R', M_max(R')) and column-sliced for smaller M):
    per M:
      1  rollouts of every arm on every case: refined errors vs ST and S on the 257^2 shared nodes, distance from the
         converged continuum rollout `gref` (Gauss 640^2) on the full mesh, LM iterations / exits / rejections; gref's
         reached states k=1..50 are the rho population; `gref_check` (Gauss 768^2) on its cohort.
      2  rho of every arm on the population against the continuum target (gref rule); target checks (gref_check,
         flux form).
      3  gates (K-conv on all cases, K-target, controls reported per criterion), in-job m* per family at both tau
         (primary and secondary) and the named diagnostics m_d, m_rho (DESIGN section 4, A0-3).
      4  A-B-A timing panel of the setting's arms (A0-8): A1 ROM arms, B decode only, A2 ROM arms.
    projection floor of the first R' columns against the ST / S references on the 257^2 nodes (thin QR, A0-6).
  final   cross-setting A-B-A panel of every setting's selected arms (primary tau), the confirmation block of H3.
Outputs result.json (+ npz side files) in --out.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VQ = HERE / 'vendor' / 'quad2d'
# the vendored qcore resolves its own vendor/ dir; engines/iterative_paths/sep_common live in the repository
for _p in (VQ, VQ / 'vendor', ROOT / 'experiments' / 'mr-burgers2d', ROOT / 'experiments' / 'separable-decoder'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import qcore as Q   # vendor/quad2d/qcore.py, unchanged (its CKPT constant is not used: the path is passed explicitly)

CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
REASONS = {0: 'budget', 1: 'tol', 2: 'tiny_step', 3: 'damping_limit', 4: 'stationary'}
EXPECTED_COHORT = {'dev6': None, 'val32': None}
_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
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
    """max over the evolved output times t >= 1 (index 0 is the initial state)."""
    return float(np.max(np.asarray(x)[1:]))


def eligible(row, budget_frac):
    """DESIGN A0-3: finite fields, zero damping-limit exits, at most `budget_frac` of the steps ending on the budget."""
    steps = sum(row['exits'].values())
    return bool(row['finite'] and row['exits']['damping_limit'] == 0 and row['exits']['budget'] <= budget_frac * steps)


def select_mstar(rows_by_arm, rho_by_arm, arms, family, tau, rho_bar, budget_frac, use_d=True, use_rho=True):
    """DESIGN section 4 + A0-3: the smallest-m non-control arm of `family` whose every case is eligible and (use_d)
    worst distance from gref <= tau and (use_rho) worst continuum rho <= rho_bar. use_d/use_rho = False gives the
    named diagnostics m_rho / m_d. Returns (name, m) or (None, None) = no tested ladder member qualifies."""
    cands = sorted([a for a in arms if a['family'] == family and not a.get('control')], key=lambda a: a['m'])
    for a in cands:
        rs = rows_by_arm.get(a['name'], [])
        if not rs or not all(eligible(r, budget_frac) for r in rs):
            continue
        if use_d:
            d = [r.get('vs_gref_evolved') for r in rs]
            if not all(v is not None and np.isfinite(v) for v in d) or max(d) > tau:
                continue
        if use_rho:
            rh = rho_by_arm.get(a['name'])
            if rh is None or not np.isfinite(rh) or rh > rho_bar:
                continue
        return a['name'], a['m']
    return None, None


def aba_panel(subjects, ncases, reps, burn_s, rng, log):
    """DESIGN A0-8: A1 (every 'A' subject x case x reps, seeded random order) -> B ('B' subjects, same cases) -> A2
    (the A set again, fresh order). Each subject is {key, phase: 'A'|'B', make(x) -> zero-arg callable,
    check(x, out) -> bool | None}; every subject is compiled and warmed on every case first; a pre-compiled burn
    precedes every timed call; block_until_ready on the whole output. Returns (invocations, summary)."""
    for sj in subjects:
        for x in range(ncases):
            jax.block_until_ready(sj['make'](x)())
    inv = []
    for ph, want in (('A1', 'A'), ('B', 'B'), ('A2', 'A')):
        order = [(j, x, r) for j, sj in enumerate(subjects) if sj['phase'] == want for x in range(ncases)
                 for r in range(reps)]
        rng.shuffle(order)
        for j, x, r in order:
            sj = subjects[j]
            fn = sj['make'](x)
            burn(burn_s)
            t = time.perf_counter()
            o = fn()
            jax.block_until_ready(o)
            secs = time.perf_counter() - t
            ent = dict(key=sj['key'], phase=ph, case=x, rep=r, seconds=secs)
            if sj.get('check') is not None:
                ent['output_matches'] = sj['check'](x, o)
            inv.append(ent)
            del o
        log(f'timing phase {ph}: {len(order)} invocations')
    summ = {}
    for sj in subjects:
        xs = [d_ for d_ in inv if d_['key'] == sj['key']]
        a1 = [d_['seconds'] for d_ in xs if d_['phase'] == 'A1']
        a2 = [d_['seconds'] for d_ in xs if d_['phase'] == 'A2']
        summ[sj['key']] = dict(median_ms=1e3 * float(np.median([d_['seconds'] for d_ in xs])), n=len(xs),
                               drift=(float(np.median(a2) / np.median(a1)) if a1 and a2 else None))
    dr = [v['drift'] for v in summ.values() if v['drift'] is not None]
    det = [d_['output_matches'] for d_ in inv if 'output_matches' in d_]
    gates = dict(drift_worst=float(max(max(dr), 1 / min(dr))) if dr else None,
                 deterministic=bool(all(det)), checked=len(det))
    gates['drift_pass'] = bool(gates['drift_worst'] is not None and gates['drift_worst'] <= 1.10)
    return inv, dict(subjects=summ, gates=gates)


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
    assert jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda m_: print(f'[{el()}s] {m_}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    L, dt = int(cfg['mesh']), .005
    assert L % 256 == 0
    s256 = L // 256                 # stride to the 257^2 nodes shared with the 8192^2 references
    taus = dict(primary=float(cfg['tau']), secondary=float(cfg['tau_secondary']))
    rho_bar, bfrac = float(cfg['rho_bar']), float(cfg['budget_fraction'])
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, jax_version=jax.__version__, mesh=L, x64=True,
               precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               checkpoint_sha256=sha_file(CKPT), rotation_sha256=sha_file(Q.ROTATION),
               qcore_sha256=sha_file(Path(Q.__file__)),
               cohorts={}, references={}, setup={}, settings={}, rows=[], rho={}, gates={}, timing={}, floors={},
               selection={}, final_timing={}, complete=False)
    assert cfg.get('expected_checkpoint_sha256') in (None, rep['checkpoint_sha256']), rep['checkpoint_sha256']
    assert cfg.get('expected_rotation_sha256') in (None, rep['rotation_sha256']), rep['rotation_sha256']
    assert cfg.get('expected_qcore_sha256') in (None, rep['qcore_sha256']), rep['qcore_sha256']
    save = lambda: json.dump(clean(rep), open(out / 'result.json', 'w'), indent=1)

    def peak():
        try:
            ms = jax.devices()[0].memory_stats()
            return dict(peak_bytes_in_use=int(ms.get('peak_bytes_in_use', -1)), bytes_in_use=int(ms.get('bytes_in_use', -1)),
                        host_maxrss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        except Exception:  # noqa: BLE001
            return dict(host_maxrss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    burn(.05)

    # ------------------------------------------------------------- cohorts ----
    train = Q.e.params_draw(0, 128)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32', 'test64')}
    for k1 in allc:
        assert not any(np.allclose(t, s_) for t in train for s_ in allc[k1]), f'{k1} overlaps training'
        for k2 in allc:
            if k1 < k2:
                assert not any(np.allclose(t, s_) for t in allc[k1] for s_ in allc[k2]), f'{k1}/{k2} overlap'
    assert 'test64' not in cfg['cohorts'], 'test64 is not used in this lane (DESIGN section 3)'
    cases = []
    for coh in cfg['cohorts']:
        ph = allc[coh]
        h = sha(ph)
        want = cfg.get('expected_cohort_sha256', {}).get(coh)
        assert want in (None, h), (coh, h)
        rep['cohorts'][coh] = dict(n=len(ph), physical_sha256=h)
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    log(f'CASES {len(cases)}')

    # ---------------------------------------------------------- references ----
    refs = {}
    rdir = Path(cfg['refs'])
    man = json.loads((rdir / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted'), 'reference job incomplete or rejected'
    want = cfg['ref_contract']
    assert int(man['config']['mesh']) == want['mesh'], ('reference mesh', man['config']['mesh'])
    for tag, w_ in want['refs'].items():
        got = man['config']['refs'][tag]
        assert all(abs(float(got[k_]) - float(w_[k_])) <= 1e-15 * max(1., abs(float(w_[k_]))) for k_ in w_), (tag, got)
    keys_ = [(x['cohort'], x['case'], x['ref']) for x in man['cases']]
    assert len(keys_) == len(set(keys_)), 'duplicate reference entries'
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    for coh, c, ph in cases:
        if not cfg.get('local_waive_cohort_hash'):      # local ARM libm may differ from the cluster in the last bit
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
    rep['references'] = dict(dir=str(rdir), job_id=man.get('job_id'), commit=man.get('commit'), count=len(refs),
                             manifest_sha256=sha_file(rdir / 'result.json'))
    log(f'REFS {len(refs)}')

    # --------------------------------------------------------- model, mesh ----
    mdl = Q.Model(ckpt=CKPT, rotation=Q.ROTATION)
    assert mdl.R == 512 and mdl.K == 16, (mdl.R, mdl.K)
    Rps = sorted({int(s_['Rp']) for s_ in cfg['settings']}, reverse=True)
    edges = [0] + sorted(Rps)
    if edges[-1] != mdl.R:
        edges.append(mdl.R)
    t0 = time.perf_counter()
    Grot = Q.build_rotated_bank(mdl, L, edges, nrb=cfg.get('bank_blocks'))
    Mmax = max(int(s_['M']) for s_ in cfg['settings'])
    ops = Q.operators(Grot, edges, L, Mmax)
    lam_np = np.asarray(ops['lam'])
    for s_ in cfg['settings']:     # nested modes: the M-prefix of the M_max table is exactly modes_lean(L, M)
        M_ = int(s_['M'])
        kx_, ky_, lam_ = Q.H.modes_lean(L, M_)
        assert np.array_equal(kx_, ops['kx'][:M_]) and np.array_equal(ky_, ops['ky'][:M_]), s_
        assert np.array_equal(lam_, lam_np[:M_]), s_
    rep['setup']['bank'] = dict(edges=edges, seconds=time.perf_counter() - t0,
                                bytes=int(sum(c.nbytes for rb in Grot for c in rb)), M_max=Mmax, memory=peak())
    log('BANK+OPS')
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))
    x257 = np.arange(1, 256) / 256           # bank on the 255^2 interior of the 257^2 shared nodes (floor, A0-6)
    xy257 = np.stack(np.meshgrid(x257, x257, indexing='ij'), -1).reshape(-1, 2)
    G257 = jax.block_until_ready(mdl.base.at(xy257, chunk=8192) @ jnp.asarray(mdl.T))      # (255^2, 512)
    save()

    U0 = {}
    for coh, c, ph in cases:
        u0n = Q.e.initial(L, ph)
        U0[(coh, c)] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n)),
                        float(np.linalg.norm(u0n[::s256, ::s256])))
    tcfg = cfg['timing']
    rng = np.random.default_rng(int(tcfg.get('seed', 20261008)))
    tcases = [x for x in cases if x[0] == tcfg['cohort']][:tcfg['cases']]
    final = []                    # (setting key, family, arm name, call, base, dec, Rp, sha-per-timing-case)

    for Rp in Rps:
        trust, _ = mdl.trust_linear(Rp)
        cold = Q.build_cold_linear(mdl, Rp, Hsub @ mdl.Lrot[:Rp].T)
        Gpre = Q.BK.prefix(Grot, edges, Rp)
        settings = sorted([s_ for s_ in cfg['settings'] if int(s_['Rp']) == Rp], key=lambda s_: -int(s_['M']))
        MR = int(settings[0]['M'])
        kxR, kyR = ops['kx'][:MR], ops['ky'][:MR]
        specs = list(cfg['arms']) + [dict(name='gref', family='ref', rule=cfg['gref']),
                                     dict(name='gref_check', family='ref', rule=cfg['gref_check'])]
        t1 = time.perf_counter()
        blocks = {}
        for spec in specs:
            X, w = Q.offmesh_rule(spec['rule'])
            blocks[spec['name']] = (Q.offmesh_data(mdl, Rp, X, w, L, kxR, kyR, 'point',
                                                   chunk=cfg.get('point_chunk', 32768)), len(X))
        rep['setup'][f'blocks_R{Rp}'] = dict(seconds=time.perf_counter() - t1, M=MR, memory=peak(),
                                             m={k: v[1] for k, v in blocks.items()})
        log(f'BLOCKS R{Rp}')

        for st in settings:
            M = int(st['M'])
            key = f'R{Rp}_M{M}'
            assert M >= Rp, key
            ts = time.perf_counter()
            base = dict(A=ops['Arot'][:M, :Rp], lam=ops['lam'][:M], G=Gpre)
            sv = np.linalg.svd(np.asarray(base['A']), compute_uv=False)
            fq, dec = Q.make_linear_query('point', Rp, L, dt, trust, step_budget=cfg['step_budget'], gtol=cfg['gtol'])
            arms = {}
            for spec in specs:
                d, m = blocks[spec['name']]
                data = dict(base, Gq=d['Gq'], Gs=d['Gs'], Psi=d['Psi'][:, :M])
                arms[spec['name']] = dict(spec=spec, m=m, data=data,
                                          call=(lambda u, nu, data=data: fq(u, nu, data, cold)))
            rep['settings'][key] = dict(
                Rp=Rp, M=M, kappa=M / Rp, kappa_nominal=st.get('kappa'), trust=trust, tensor_bytes=8 * M * Rp * Rp,
                A_singular_max=float(sv[0]), A_singular_min=float(sv[-1]), A_condition=float(sv[0] / sv[-1]),
                A_rank=int(np.sum(sv > 1e-12 * sv[0])),
                arms={n_: dict(rule=v['spec']['rule'], family=v['spec']['family'], m=v['m'],
                               control=bool(v['spec'].get('control')), bytes=8 * v['m'] * (2 * Rp + M))
                      for n_, v in arms.items()})
            # ------------------------------------------ phase 1: rollouts ----
            pop, labels, sha1, rows_by_arm, gref257 = [], [], {}, {}, {}
            order = ['gref'] + [n_ for n_ in arms if n_ != 'gref']
            for ci, (coh, c, ph) in enumerate(cases):
                u0, nu, n0, n0r = U0[(coh, c)]
                gref_f = None
                for name in order:
                    arm = arms[name]
                    t2 = time.perf_counter()
                    v = arm['call'](u0, nu)
                    jax.block_until_ready(v['fields'])
                    secs = time.perf_counter() - t2
                    f = v['fields']
                    fr = np.asarray(f[:, ::s256, ::s256])
                    it, reason, rej = (np.asarray(v[k_]) for k_ in ('it', 'reason', 'rej'))
                    sha1[(name, coh, c)] = sha(fr)
                    row = dict(setting=key, Rp=Rp, M=M, arm=name, rule=arm['spec']['rule'],
                               family=arm['spec']['family'], control=bool(arm['spec'].get('control')), m=arm['m'],
                               cohort=coh, case=c, finite=bool(jnp.all(jnp.isfinite(f))), seconds_first=secs,
                               iterations_total=int(it.sum()), iterations_max=int(it.max()),
                               exits={REASONS[k_]: int(np.sum(reason == k_)) for k_ in REASONS},
                               rejected_total=int(rej.sum()), restricted_sha256=sha1[(name, coh, c)], n0=n0, n0r=n0r)
                    row['eligible'] = eligible(row, bfrac)
                    for tag in ('ST', 'S'):
                        r = refs[(coh, c, tag)]
                        pe = [float(np.linalg.norm(x - y)) / n0r for x, y in zip(fr, r)]
                        row[f'ref_{tag}_per_time'] = pe
                        row[f'ref_{tag}_evolved'] = evolved_max(pe)
                    if name == 'gref':
                        gref_f = f
                        gref257[f'{coh}{c}'] = fr
                        W = np.asarray(v['internal'])[1:]
                        pop.append(W)
                        labels += [f'{coh}|{c}|{k + 1}' for k in range(len(W))]
                    else:
                        pt = [float(x) for x in jnp.linalg.norm((f - gref_f).reshape(6, -1), axis=1) / n0]
                        row['vs_gref_per_time'] = pt
                        row['vs_gref_evolved'] = evolved_max(pt)
                    rep['rows'].append(row)
                    rows_by_arm.setdefault(name, []).append(row)
                    if coh == cfg.get('audit_cohort') and c in cfg.get('audit_cases', []):
                        np.savez_compressed(out / f'audit_{key}_{name}_{coh}{c}.npz', f257=fr,
                                            internal=np.asarray(v['internal']),
                                            **({'full': np.asarray(f)} if name in cfg.get('audit_full_arms', []) else {}))
                    del v, f
                del gref_f
                if ci % 8 == 7:
                    log(f'CASE {key} {ci}')
            np.savez_compressed(out / f'gref_f257_{key}.npz', **gref257)
            save()

            # ------------------------------- identifiability at reached states (A2-10) ----
            nlJ = Q.tested_jac('point', L)
            gd = arms['gref']['data']
            S_ = 1. / (1 + dt * cases[0][2][4] * np.asarray(base['lam']))
            jc = {}
            for k_ in cfg.get('jacobian_states', [1, 25, 50]):
                cst = jnp.asarray(pop[0][k_ - 1])
                _, dN = nlJ(cst, gd, None)
                Jm = S_[:, None] * (np.asarray(base['A']) + dt * (np.asarray(dN) + cases[0][2][4] *
                                                                  np.asarray(base['lam'])[:, None] * np.asarray(base['A'])))
                svj = np.linalg.svd(Jm, compute_uv=False)
                jc[str(k_)] = dict(sigma_max=float(svj[0]), sigma_min=float(svj[-1]), condition=float(svj[0] / svj[-1]),
                                   rank=int(np.sum(svj > 1e-12 * svj[0])))
            rep['settings'][key]['jacobian_reached'] = dict(case=f'{cases[0][0]}{cases[0][1]}', states=jc)

            # ------------------------------------------------ phase 2: rho ----
            t1 = time.perf_counter()
            C = np.concatenate(pop)
            np.savez_compressed(out / f'population_{key}.npz', C=C, labels=np.array(labels))
            fval = jax.jit(jax.vmap(Q.tested_value('point', L), in_axes=(0, None)))
            ffl = jax.jit(jax.vmap(Q.tested_value('flux', L), in_axes=(0, None)))
            Cj = jnp.asarray(C)
            ch = 64

            def values(rule, form='point'):
                X, w = Q.offmesh_rule(rule)
                acc = np.zeros((len(C), M))
                pc = cfg.get('point_chunk', 32768)
                fn = fval if form == 'point' else ffl
                for q0 in range(0, len(X), pc):
                    d = Q.offmesh_data(mdl, Rp, X[q0:q0 + pc], w[q0:q0 + pc], L, kxR[:M], kyR[:M], form)
                    acc += np.concatenate([np.asarray(fn(Cj[i:i + ch], d)) for i in range(0, len(C), ch)])
                    del d
                return acc
            rho = lambda V, T_: np.linalg.norm(V - T_, axis=1) / np.linalg.norm(T_, axis=1)
            Tc = values(cfg['gref'])
            tn = np.linalg.norm(Tc, axis=1)
            assert tn.min() > 0, 'zero continuum target on the population'
            ck, cf = rho(values(cfg['gref_check']), Tc), rho(values(cfg['gref'], 'flux'), Tc)
            rep['gates'][f'continuum_target_{key}'] = dict(
                states=len(C), target_norm_min=float(tn.min()), target_norm_median=float(np.median(tn)),
                check_rho_max=float(ck.max()), flux_rho_max=float(cf.max()), bar=cfg['target_bar'],
                passed=bool(ck.max() <= cfg['target_bar'] and cf.max() <= cfg['target_bar']))
            res, rho_max = {}, {}
            for name, arm in arms.items():
                if name == 'gref':
                    continue
                V = np.concatenate([np.asarray(fval(Cj[i:i + ch], arm['data'])) for i in range(0, len(C), ch)])
                r_c = rho(V, Tc)
                i = int(np.argmax(r_c))
                res[name] = dict(m=arm['m'], max=float(r_c.max()), p95=float(np.quantile(r_c, .95)),
                                 median=float(np.median(r_c)), argmax=labels[i])
                rho_max[name] = float(r_c.max())
                if name == cfg.get('audit_rho_arm'):
                    sub_ = slice(None, None, max(1, len(C) // 64))
                    np.savez_compressed(out / f'audit_rho_{key}_{name}.npz', C=C[sub_], V=V[sub_], Tc=Tc[sub_])
            rep['rho'][key] = dict(states=len(C), population='gref', seconds=time.perf_counter() - t1, rules=res)
            del Cj
            log(f'RHO {key}')

            # ----------------------------------- phase 3: gates and selection ----
            gref_ok = all(r['eligible'] for r in rows_by_arm['gref'])
            gchk = max(r['vs_gref_evolved'] for r in rows_by_arm['gref_check'])
            chk_ok = all(r['eligible'] for r in rows_by_arm['gref_check'])
            rep['gates'][f'converged_{key}'] = dict(
                gref_all_eligible=gref_ok, check_all_eligible=chk_ok, gref_check_worst_distance=gchk,
                bar=cfg['conv_bar'], cases=len(rows_by_arm['gref_check']),
                passed=bool(gref_ok and chk_ok and gchk <= cfg['conv_bar']))
            arm_list = [dict(name=n_, family=v['spec']['family'], m=v['m'], control=v['spec'].get('control'))
                        for n_, v in arms.items()]
            ctrl = {}
            for name, arm in arms.items():
                if arm['spec'].get('control'):
                    dmax = max(r['vs_gref_evolved'] for r in rows_by_arm[name])
                    ctrl[name] = dict(worst_distance=dmax, rho_max=rho_max[name],
                                      all_eligible=all(r['eligible'] for r in rows_by_arm[name]),
                                      fails_distance={t_: bool(dmax > tv) for t_, tv in taus.items()},
                                      fails_rho=bool(rho_max[name] > rho_bar))
                    ctrl[name]['would_be_selected'] = {
                        t_: bool(ctrl[name]['all_eligible'] and not ctrl[name]['fails_distance'][t_]
                                 and not ctrl[name]['fails_rho']) for t_ in taus}
            disc = {t_: bool(ctrl and not any(v['would_be_selected'][t_] for v in ctrl.values())) for t_ in taus}
            rep['gates'][f'controls_{key}'] = dict(arms=ctrl, discriminating=disc)
            valid = bool(rep['gates'][f'converged_{key}']['passed'] and rep['gates'][f'continuum_target_{key}']['passed'])
            sel = {}
            for t_, tv in taus.items():
                for fam in cfg['families']:
                    nm, m = select_mstar(rows_by_arm, rho_max, arm_list, fam, tv, rho_bar, bfrac)
                    nd, md = select_mstar(rows_by_arm, rho_max, arm_list, fam, tv, rho_bar, bfrac, use_rho=False)
                    sel[f'{t_}|{fam}'] = dict(tau=tv, family=fam,
                                              available=bool(valid and disc[t_]),
                                              arm=nm if (valid and disc[t_]) else None,
                                              m=m if (valid and disc[t_]) else None,
                                              arm_raw=nm, m_raw=m, m_d=md, arm_d=nd)
            for fam in cfg['families']:
                nr, mr = select_mstar(rows_by_arm, rho_max, arm_list, fam, 0., rho_bar, bfrac, use_d=False)
                sel[f'rho_only|{fam}'] = dict(m_rho=mr, arm_rho=nr)
            rep['selection'][key] = dict(valid=valid, discriminating=disc, entries=sel)
            save()

            # --------------------------------------------- phase 4: timing ----
            tn_ = [n_ for n_ in arms if n_ != 'gref_check' and not arms[n_]['spec'].get('control')]

            def mk_rom(n_):
                def make(x):
                    coh, c, _ = tcases[x]
                    u0, nu, _, _ = U0[(coh, c)]
                    return lambda: arms[n_]['call'](u0, nu)

                def check(x, o):
                    coh, c, _ = tcases[x]
                    return bool(sha(np.asarray(o['fields'][:, ::s256, ::s256])) == sha1[(n_, coh, c)])
                return dict(key=n_, phase='A', make=make, check=check)

            def mk_dec(x):
                Wd = jnp.asarray(.1 * np.random.default_rng(x).normal(size=(6, Rp)))
                return lambda: dec(Wd, base)
            subjects = [mk_rom(n_) for n_ in tn_] + [dict(key='decode', phase='B', make=mk_dec, check=None)]
            inv, summ = aba_panel(subjects, len(tcases), tcfg['reps'], tcfg['burn'], rng, log)
            rep['timing'][key] = dict(cases=[f'{a_}{b_}' for a_, b_, _ in tcases], reps=tcfg['reps'], burn=tcfg['burn'],
                                      invocations=inv, **summ)
            rep['settings'][key]['memory'] = peak()
            rep['settings'][key]['seconds'] = time.perf_counter() - ts
            # deployed family (A2-8): the primary-tau m* arm with the lower median in THIS panel, fixed before the final
            dep = [(summ['subjects'][sv_['arm']]['median_ms'], sv_['family'], sv_['arm']) for k_, sv_ in sel.items()
                   if k_.startswith('primary|') and sv_['arm'] is not None]
            if dep:
                ms_, fam_, arm_ = min(dep)
                rep['selection'][key]['deployed'] = dict(family=fam_, arm=arm_, m=arms[arm_]['m'], median_ms=ms_)
                shas = [sha1[(arm_, coh, c)] for coh, c, _ in tcases]
                final.append((key, fam_, arm_, arms[arm_]['call'], base, dec, Rp, shas))
            else:
                rep['selection'][key]['deployed'] = None
            log(f'TIMING {key}')
            save()
            del arms, pop, C
            jax.clear_caches()
            gc.collect()

        # ------------------------------------------ projection floor (per R') ----
        Uf, sf, _ = jnp.linalg.svd(G257[:, :Rp], full_matrices=False)
        sf = np.asarray(sf)
        rk = int(np.sum(sf > 1e-12 * sf[0]))
        Uf = Uf[:, :rk]
        fl = {}
        for ii, (coh, c, ph) in enumerate(cases):
            n0r = U0[(coh, c)][3]
            for tag in ('ST', 'S'):
                r = jnp.asarray(refs[(coh, c, tag)][:, 1:-1, 1:-1].reshape(6, -1))
                res_ = r - (r @ Uf) @ Uf.T
                pe = [float(x) / n0r for x in jnp.linalg.norm(res_, axis=1)]
                fl.setdefault(tag, []).append(dict(cohort=coh, case=c, per_time=pe, evolved=evolved_max(pe)))
                if ii == 0 and tag == 'ST':
                    sol = np.linalg.lstsq(np.asarray(G257[:, :Rp]), np.asarray(r[-1]), rcond=1e-12)[0]   # A3-6
                    lres = float(np.linalg.norm(np.asarray(G257[:, :Rp]) @ sol - np.asarray(r[-1]))) / n0r
                    resid_check = (abs(lres - pe[-1]) / pe[-1]) if pe[-1] > 1e-12 else abs(lres - pe[-1])
        rep['floors'][f'R{Rp}'] = dict(
            rank=rk, condition=float(sf[0] / sf[rk - 1]), lstsq_vs_svd_rel=resid_check,
            **{tag: dict(worst=max(x['evolved'] for x in v), median=float(np.median([x['evolved'] for x in v])),
                         cases=v) for tag, v in fl.items()})
        del Uf, blocks
        gc.collect()
        save()

    # ------------------------------------------------- final cross-setting panel ----
    if final and cfg.get('final_panel', True):
        subjects = []
        for key, fam, arm, call, base, dec, Rp, shas in final:
            def make(x, call=call):
                coh, c, _ = tcases[x]
                u0, nu, _, _ = U0[(coh, c)]
                return lambda: call(u0, nu)

            def check(x, o, shas=shas):
                return bool(sha(np.asarray(o['fields'][:, ::s256, ::s256])) == shas[x])
            subjects.append(dict(key=f'{key}|{fam}|{arm}', phase='A', make=make, check=check))
        seen = set()
        for key, fam, arm, call, base, dec, Rp, shas in final:
            if Rp in seen:
                continue
            seen.add(Rp)

            def mkd(x, dec=dec, base=base, Rp=Rp):
                Wd = jnp.asarray(.1 * np.random.default_rng(x).normal(size=(6, Rp)))
                return lambda: dec(Wd, base)
            subjects.append(dict(key=f'decode|R{Rp}', phase='B', make=mkd, check=None))
        inv, summ = aba_panel(subjects, len(tcases), tcfg['reps'], tcfg['burn'], rng, log)
        rep['final_timing'] = dict(invocations=inv, **summ)
        log('FINAL TIMING')

    rep['elapsed_seconds'] = el()
    rep['peak_bytes'] = peak()
    rep['checkpoint_sha256_after'] = sha_file(CKPT)
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('W2D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
