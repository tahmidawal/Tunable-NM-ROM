"""burgers-eqcert: certification of q=256 quadrature rules at 256^2 / 512^2 / 1024^2 on five held-out draws,
then the frozen Burgers NM-ROM against Newton-BiCGStab in ONE allocation, ONE GPU (DESIGN.md).

Adapted from experiments/hires-burgers/hires.py @ 0ab60014. Changes (DESIGN.md sections 3-4): populations from a
fresh parameter seed split into several held-out DRAWS, every state tagged by its step index k; certificates per
draw and per k-threshold; per-arm status (confirmed / marginal / fails / exact); exact-first-steps arms (xfast);
a timed exact-residual arm; per-rule tolerance lists; 'lattice': 'half' = every second node.

Phases, each written incrementally to `result.json` so a truncated job is collectable:

  1 cohort, same-grid full-order reference (`fft_tight`) for every development case
  2 bank on the L-grid, reachable populations (audited dense query at the population mesh)
  3 per rung (q, M): test projections without Phi, candidate quadrature rules (transferred
    b-eqtop support with weights refit, uniform sub-lattice), held-out rho certificate on the
    dense-query population AND on the states the deployed query itself visits
  4 QUICK ANSWER: every deployed arm once per case against `fft_tight` (printed at once)
  5 dense truth arms (exact advection, same solver) once per case
  6 timed panel: randomised order, burn-in, synchronised, >= 5 repetitions, ROM and FOM paired
  7 profile of the accurate rung (parts and micro-kernels)
  8 refined reference and physical error of every kept field (last; time-guarded)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import ladder as LD
import topfix as TF
import fast as F
import ladders as FL
import hops as H
import hfast as HF
import xfast as XF

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = np.array([0., .05, .1, .15, .2, .25])


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def is_oom(exc):
    s = str(exc)
    return 'RESOURCE_EXHAUSTED' in s or 'out of memory' in s.lower()


def rel_per_time(f, truth, n0):
    return [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, truth)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(a.inputs)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:                               # noqa: BLE001
        smi = []

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt = int(cfg['intervals']), cfg['dt']
    st = cfg['strict']
    bar, tight_bar = cfg['rho_bar'], cfg['rho_tight']
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), x64=True,
               matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt,
               weights_frozen=True, final_cohort_unopened=True, output_times=TIMES.tolist(),
               timing_contract=('gpu_seconds: supplied dense initial field resident on the GPU to six dense GPU '
                                'output fields, block_until_ready on both sides; host_seconds: the same invocation '
                                'plus the host upload of the input and the host copy of the six outputs; identical '
                                'for every subject'),
               populations={}, rules=[], arm_status={}, quick=[], dense_truth=[], arm_setup=[], invocations=[], parity=[],
               profile=[], reference=[], physical=[], dropped=[], gates={}, phases={}, complete=False)
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None      # a non-finite diagnostic is recorded as null
        if isinstance(x, np.integer):
            return int(x)
        return x

    save = lambda: dump(out / 'result.json', clean(rep))
    save()

    # ------------------------------------------------------------------ cohort --
    if cfg.get('eval_draws'):
        # another cohort, e.g. bank-floor's hold64 = params_draw(20260916, 64); named in the config and the report
        physical = np.concatenate([e.params_draw(int(sd), int(n_)) for sd, n_ in cfg['eval_draws']])
    else:
        physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                                   e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    rep['cohort_name'] = cfg.get('cohort_name', 'dev6: params_draw(7090702,4) + params_draw(911702,2), opened development cases')
    rep['physical_sha256'] = sha_array(physical)
    want = cfg.get('expected_physical_sha256')
    rep['gates']['evaluation_cohort_matches_b_panel'] = dict(expected=want, got=rep['physical_sha256'],
                                                             passed=(None if want is None else want == rep['physical_sha256']))
    # GB10 and cluster NumPy differ by 1 ulp in exp() (lab-log landmine), so only the local smoke may waive this
    assert want is None or want == rep['physical_sha256'] or cfg.get('local_smoke_waives_cohort_hash'), 'evaluation cohort differs from b-panel'
    if cfg.get('case_subset') is not None:
        physical = physical[[int(i) for i in cfg['case_subset']]]
    ncase = len(physical)
    rep['physical_cases'] = physical.tolist()
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    pc = cfg['population']
    pop_physical = e.params_draw(*pc['source_draw'])
    hold64 = e.params_draw(20260916, 64)
    for other, nm in ((physical, 'evaluation cohort'), (hold64, 'hold64'), (train_physical, 'train_physical')):
        assert not any(np.any(np.all(np.isclose(o, pop_physical), axis=1)) for o in other), f'population/{nm} overlap'
    draws = [list(d) for d in pc['cert_draws']]
    ncert = len(draws)
    if pc.get('confirm_draw'):
        draws.append(list(pc['confirm_draw']))     # draw index ncert: used ONLY after selection (DESIGN A1)
    flat = [i for d in draws for i in d]
    assert len(flat) == len(set(flat)) and max(flat) < len(pop_physical), 'held-out draws must be disjoint'
    rep['population_source'] = dict(source_draw=pc['source_draw'], sha256=sha_array(pop_physical), draws=draws,
                                    certification_draws=ncert, confirmation_draw_index=(ncert if pc.get('confirm_draw') else None),
                                    disjoint_from=['evaluation cohort', 'hold64 = params_draw(20260916,64)',
                                                   'train_physical'])
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = [float(np.linalg.norm(u)) for u in inputs_u]
    sub = max(1, L // cfg.get('restrict_to', 256))

    # the bank first: the largest single allocation (64 GiB at 4096^2) goes into an unfragmented pool
    bank = A.CoordBank(params, K, R)
    t0 = time.perf_counter()
    G = H.build_bank(bank, L, nblocks=cfg.get('bank_blocks'))
    single = len(G) == 1
    rep['phases']['bank'] = dict(shape=[H.bank_rows(G), R], blocks=len(G), seconds=time.perf_counter() - t0, bytes=int(sum(g.nbytes for g in G)))
    print('BANK', H.bank_rows(G), 'rows in', len(G), 'blocks', el(), flush=True)

    # ---------------------------------------------------- full-order solvers ----
    foms = {}

    def fom_for(fs):
        key = (fs.get('mesh', L), fs['dt'], fs.get('impl', 'audited'))
        if key not in foms:
            if key[0] == L and key[2] == 'audited':
                foms[key] = ip.make_fom(L, fs['dt'], 'fft')
            else:
                foms[key] = (e.make_fom(key[0], fs['dt'], target=L)[0], None)
        return foms[key]

    def run_fom(fs, u, nu):
        fn, pre = fom_for(fs)
        if pre is None:
            return fn(u, nu, fs['ntol'], fs['ltol'])
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    tight = next(fs for fs in cfg['fom_settings'] if fs['name'] == cfg['same_grid_reference'])
    truth = {}
    for c in range(ncase):
        t0 = time.perf_counter()
        v = run_fom(tight, jnp.asarray(inputs_u[c]), float(physical[c, 4]))
        jax.block_until_ready(v)
        f = np.asarray(v[0])
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and np.isfinite(rn).all() and rn.max() <= tight['ntol'] * (1 + 1e-9), ('truth not converged', c)
        truth[c] = f
        rep['phases'].setdefault('truth', []).append(dict(case=c, seconds=time.perf_counter() - t0,
                                                          max_relative_residual=float(rn.max()),
                                                          converged=bool(rn.max() <= tight['ntol'] * (1 + 1e-9)),
                                                          newton_total=int(np.sum(np.asarray(v[1])))))
        print('TRUTH', c, el(), flush=True)
    save()

    def score(f, c):
        sg = rel_per_time(f, truth[c], n0[c])
        cur = [float(np.linalg.norm(a_ - b_) / max(np.linalg.norm(b_), 1e-300)) for a_, b_ in zip(f, truth[c])]
        return dict(same_grid_per_time=sg, same_grid_all=float(max(sg)), same_grid_evolved=float(max(sg[1:])),
                    t0_compression=float(sg[0]), current_relative_per_time=cur,
                    current_relative_evolved=float(max(cur[1:])))

    # -------------------------------------------------------- bank, directions --
    dfile = inputs / cfg['directions_file']
    Cnp = np.ascontiguousarray(np.load(dfile)['C'])
    rep['directions'] = dict(file=cfg['directions_file'], sha256=sha_file(dfile), expected=cfg.get('directions_sha256'))
    assert cfg.get('directions_sha256') in (None, rep['directions']['sha256'])
    Cfull = jnp.asarray(Cnp)
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // cfg['decoder_code_subsample'])])
    colds, heads = {}, {}

    def cold_for(q):
        if q not in colds:
            heads[q] = TF.corrected_head(params, Cfull[:, :q], K)
            colds[q] = A.build_cold(bank, heads[q], np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1),
                                    cfg['cold_axis_points'])[0]
        return colds[q], heads[q]

    lin = lambda d: 'gj' if d <= cfg['gauss_jordan_max'] else 'lu'
    save()

    # ------------------------------------- operators without Phi, parity-gated ----
    pc = cfg['population']
    Lp = int(pc['mesh'])
    Gp = bank.on_grid(Lp) if Lp != L else jnp.concatenate(G, 0)
    ops_M = {}

    def operators(M):
        if M not in ops_M:
            kx, ky, lam = H.modes_lean(L, M)
            sx, sy = H.sine_tables(L, kx, ky)
            sxd, syd = jnp.asarray(sx), jnp.asarray(sy)
            ops_M[M] = dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sxd, sy=syd, A=H.project_bank(G, sxd, syd, L))
            jax.block_until_ready(ops_M[M]['A'])
        return ops_M[M]

    pg = []
    for Mg in sorted({r['M'] for r in cfg['rungs']}):          # every deployed test count, at the population mesh
        Phig, lamg, _ = e.modes(Lp, Mg)
        kxg, kyg, lamg2 = H.modes_lean(Lp, Mg)
        sxg, syg = H.sine_tables(Lp, kxg, kyg)
        Ag = H.project_bank((Gp,), jnp.asarray(sxg), jnp.asarray(syg), Lp)
        Aref = jnp.asarray(Phig).T @ Gp
        pick = np.arange(0, (Lp - 1) ** 2, 97)
        g0 = dict(mesh=Lp, M=Mg, lam_identical=bool(np.array_equal(lamg, lamg2)),
                  A_relative=float(jnp.linalg.norm(Ag - Aref) / jnp.linalg.norm(Aref)),
                  rows_max_abs=float(np.max(np.abs(H.phi_rows(Lp, kxg, kyg, H.unravel_nodes(pick, Lp)) - Phig[pick]))))
        g0['passed'] = bool(g0['lam_identical'] and g0['A_relative'] <= 1e-12 and g0['rows_max_abs'] <= 1e-14)
        pg.append(g0)
        del Phig, Aref, Ag
    rep['gates']['phi_free_operator_parity'] = dict(passed=all(x['passed'] for x in pg), per_M=pg,
                                                    note='algebraically equal, different summation order; toleranced')
    assert rep['gates']['phi_free_operator_parity']['passed'], pg
    save()

    # --------------------------------------------- rules, arms, the quick answer ----
    assert not any(rs.get('refit') for rg in cfg['rungs'] for rs in rg['rules']), 'no fitted rules in this lane'
    rules, built, subjects, kept = {}, {}, [], {}
    pops, targets = {}, {}

    def targets_for(q, M, tag):
        if (q, M, tag) not in targets:
            o = operators(M)
            targets[(q, M, tag)] = H.dense_targets(G, pops[q][tag], o['sx'], o['sy'], L, chunk=cfg['target_chunk'])
        return targets[(q, M, tag)]

    def build_rule(rung, rs):
        q, M, name = rung['q'], rung['M'], rs['name']
        o = operators(M)
        t0 = time.perf_counter()
        parts_, src = [], []
        for part in rs['parts']:
            if 'lattice' in part:
                s_ = L // 2 if part['lattice'] == 'half' else int(part['lattice'])
                ij, w = H.lattice_rule(L, s_)
                src.append(dict(kind='lattice', s=s_, spec=part['lattice'], m=int(len(ij))))
            elif 'random' in part:
                rr = np.random.default_rng(part['seed'])
                ij = H.unravel_nodes(np.sort(rr.choice((L - 1) ** 2, part['random'], replace=False)), L)
                w = np.full(len(ij), (L - 1) ** 2 / len(ij))
                src.append(dict(kind='random', m=int(len(ij)), seed=part['seed']))
            else:
                z = np.load(inputs / part['file'])
                assert sha_file(inputs / part['file']) == part['sha256'], part['file']
                ij = H.transfer_nodes(z['nodes'].astype(int), part['mesh'], L)
                w = np.asarray(z['weights'], float) * (L / part['mesh']) ** 2
                src.append(dict(kind='transfer', file=part['file'], sha256=part['sha256'], source_mesh=part['mesh'],
                                source_m=int(len(ij)), source_status=part.get('status')))
            parts_.append((ij, w))
        assert len(parts_) == 1 or rs.get('refit'), 'a union support needs a weight refit'
        ij = np.concatenate([x[0] for x in parts_])
        w = np.concatenate([x[1] for x in parts_])
        _, uniq = np.unique(H.ravel_nodes(ij, L), return_index=True)
        ij, w = ij[np.sort(uniq)], w[np.sort(uniq)]
        finfo = None
        if rs.get('refit'):
            G5 = bank.stencil(ij, L)
            w, finfo = H.refit_weights(H.phi_rows(L, o['kx'], o['ky'], ij), H.sampled_advection(G5, pops[q]['fit'], L),
                                       targets_for(q, M, 'fit'), scaling=rs['refit'])
            del G5
        ops, ij, w = H.rule_ops(bank, L, o['kx'], o['ky'], ij, w)
        np.savez_compressed(out / f'rule_L{L}_q{q}_M{M}_{name}.npz', ij=ij, weights=w)
        info = dict(q=q, M=M, rule=name, m=int(len(ij)), source=src, refit=finfo, control=bool(rs.get('control')),
                    rho_bar=bar, rho_tight=tight_bar, build_seconds=time.perf_counter() - t0, deployed={})
        rules[(q, M, name)] = (ops, info)
        rep['rules'].append(info)

    KTHR = cfg.get('k_thresholds', [0, 1, 2])

    def draw_summaries(r, meta):
        """rho per state -> {k>=j: {draw d: summary}} plus the union; meta columns = (draw, trajectory, k)."""
        out_ = {}
        for j in KTHR:
            sel = meta[:, 2] >= j
            per = {}
            for d in range(len(draws)):
                sd = sel & (meta[:, 0] == d)
                sm = H.rho_summary(r[sd], bar, tight_bar)
                am = np.flatnonzero(sd)[int(np.argmax(r[sd]))]
                sm.update(argmax_trajectory=int(meta[am, 1]), argmax_k=int(meta[am, 2]))
                per[str(d)] = sm
            un = H.rho_summary(r[sel & (meta[:, 0] < ncert)], bar, tight_bar)       # certification draws only
            un['draws_certified_primary'] = int(sum(per[str(d)]['certified_primary'] for d in range(ncert)))
            out_[f'k>={j}'] = dict(union=un, per_draw=per)
        return out_

    def certify_heldout(q, M, name):
        ops, info = rules[(q, M, name)]
        r = H.rho(ops['Pq'], H.sampled_advection(ops['G5'], pops[q]['cert'], L), targets_for(q, M, 'cert'))
        info['heldout_population'] = draw_summaries(r, pops[q]['meta'])
        info['heldout_rho_per_state'] = [float(x) for x in r]
        u0_ = info['heldout_population']['k>=0']['union']
        print('RULE', q, M, name, 'm', info['m'], 'rho_max(k>=0)', f"{u0_['rho_max']:.4f}", 'draws', u0_['draws_certified_primary'],
              '| k>=1', f"{info['heldout_population']['k>=1']['union']['rho_max']:.4f}",
              info['heldout_population']['k>=1']['union']['draws_certified_primary'], el(), flush=True)
        save()

    def add(name, **kw):
        built[name] = dict(name=name, **kw)
        rep['arm_setup'].append({k: v for k, v in kw.items() if k in (
            'family', 'q', 'M', 'm', 'rule', 'gtol', 'kernel', 'solver', 'setting', 'quadrature', 'parity_twin', 'variant',
            'exact_steps', 'exact', 'control')} | dict(arm=name))
        return name

    def register(rung, rs):
        """Deployed arms of one (rung, rule): hfast / xfast only (q0_through_hfast is required)."""
        q, M = rung['q'], rung['M']
        o = operators(M)
        cold, head = cold_for(q)
        C = Cfull[:, :q]
        exact = bool(rs.get('exact'))
        if exact:
            data = dict(A=o['A'], lam=o['lam'], G=G, sx=o['sx'], sy=o['sy'])
            m_ = int((L - 1) ** 2)
        else:
            ops, info = rules[(q, M, rs['name'])]
            data = dict(A=o['A'], lam=o['lam'], G=G, G5=ops['G5'], Pq=ops['Pq'], sx=o['sx'], sy=o['sy'])
            m_ = info['m']
        tab = HF.build_tables(params, C, K, data, cold)
        tag = f"q{q}_M{M}_{rs['name']}"
        d = K + q
        chunks = next(k for k in range(1, d + 1) if d % k == 0 and d // k <= cfg['dense_tangent_group'])
        names = []
        for var in rs.get('variants', rung['variants']):
            for g in var.get('gtols', rs.get('gtols', rung['gtols'])):
                solver = var.get('solver', 'lu')
                j = int(var.get('exact_steps', 0))
                if exact and j:
                    continue
                sfx = ''.join(f'_{k_}' for k_ in ([solver] if solver != 'lu' else []) +
                              [k_ for k_ in ('clip', 'lamcarry', 'pred2') if var.get(k_)])
                if j:
                    sfx += f'_x{j}'
                fq, parts = XF.make_query(params, C, K, q, L, dt, trust, 'dense' if exact else 'eq', exact_steps=j,
                                          ic_budget=st['ic_budget'], step_budget=st['step_budget'], gtol=g,
                                          ic_gtol=var.get('ic_gtol', cfg['ic_gtol']), ridge=cfg['inner_damping'],
                                          solver=solver, tangent_chunks=chunks, parts=True, clip=bool(var.get('clip')),
                                          lam_carry=bool(var.get('lamcarry')),
                                          predictor='quad' if var.get('pred2') else 'lin')
                names.append(add(f'{tag}_{gt(g)}_fast{sfx}', family='rom', kind='rom', q=q, M=M, m=m_, rule=rs['name'],
                                 gtol=g, kernel=('xfast exact residual' if exact else
                                                 f'xfast EQ, first {j} step(s) exact' if j else 'hfast EQ') +
                                 ' (algorithmic variant: not a parity arm)',
                                 solver=solver, quadrature='exact' if exact else 'eq', data=data, cold=cold, tab=tab,
                                 parts=parts, fastarm=True, hfast=True, parity_twin=None, variant=var, exact_steps=j,
                                 exact=exact, control=bool(rs.get('control')),
                                 query=(lambda u, nu, d_, c, _t=tab, _f=fq: _f(u, nu, d_, c, _t))))
        return names

    def invoke(b, u, c):
        nu = float(physical[c, 4])
        if b['kind'] == 'fom':
            return run_fom(b['setting'], u, nu)
        return b['query'](u, nu, b['data'], b['cold'])

    def rom_row(b, v):
        reasons = np.asarray(v[3]).tolist()
        it = np.asarray(v[1])
        gj = np.asarray(v[12] if len(v) > 12 else v[8], float)
        row = dict(q=b['q'], M=b['M'], m=b.get('m'), rule=b.get('rule'), gtol=b['gtol'], stop_reasons=reasons,
                   iterations=it.tolist(), total_iterations=int(it.sum()), median_iterations=float(np.median(it)),
                   max_iterations=int(it.max()), ic_iterations=int(v[5]), ic_reason=int(v[6]),
                   budget_exits=int(sum(r == 0 for r in reasons)), rejected_exits=int(sum(r == 3 for r in reasons)),
                   residual_exits=int(sum(r == 1 for r in reasons)), tiny_step_exits=int(sum(r == 2 for r in reasons)),
                   gradient_exits=int(sum(r == 4 for r in reasons)),
                   worst_joint_stationarity=float(np.max(gj)) if np.isfinite(gj).all() else None,
                   stalled_exits=int(sum(r in (0, 2, 3) for r in reasons)))
        if len(v) > 15:
            row['damping_retries_total'] = int(np.sum(np.asarray(v[15])))
        return row

    def fom_row(b, v):
        fs = b['setting']
        rn = np.asarray(v[2])
        okstep = np.isfinite(rn) & (rn <= fs['ntol'] * (1 + 1e-9))
        return dict(dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'], mesh=fs.get('mesh', L), impl=fs.get('impl', 'audited'),
                    newton_iterations_total=int(np.sum(np.asarray(v[1]))), steps=int(len(rn)),
                    stalled_steps=int(np.sum(~okstep)), nonlinear_converged=bool(okstep.all()),
                    max_relative_residual=float(np.nanmax(rn)))

    PBAR = cfg['parity_bar']
    keep_pat = cfg.get('keep_for_reference')

    def parity_and_prune(names):
        """Parity of every optimised arm in `names` against its audited twin (same rule, same
        tolerance), then release full fields that no later phase needs (4096^2 host memory)."""
        for name in names:
            b = built.get(name)
            if b is None or not b.get('fastarm') or name not in subjects:
                continue
            twin = b.get('parity_twin')
            if twin is None or twin not in subjects:
                rep['parity'].append(dict(fast=name, base=None, covered=False, passed=None,
                                          note='no audited twin in this job; the arm stands on its directly measured error'))
                continue
            if kept[(name, 0)][0] is None or kept[(twin, 0)][0] is None:
                continue
            per = []
            for c in range(ncase):
                (fa, ia), (fb, ib) = kept[(name, c)], kept[(twin, c)]
                per.append(dict(case=c, relative=float(np.linalg.norm(fa - fb) / np.linalg.norm(fb)),
                                iterations_identical=bool(np.array_equal(ia[0], ib[0])),
                                reasons_identical=bool(np.array_equal(ia[2], ib[2])),
                                iterations_fast=int(np.sum(ia[0])), iterations_base=int(np.sum(ib[0]))))
            worst = max(x['relative'] for x in per)
            ints = all(x['iterations_identical'] and x['reasons_identical'] for x in per)
            rep['parity'].append(dict(fast=name, base=twin, covered=True, worst_relative=worst, integers_identical=ints,
                                      bar=PBAR, passed=bool(worst <= PBAR and ints), fields_within_bar=bool(worst <= PBAR),
                                      cases=per))
            print('PARITY', name, f'{worst:.3e}', 'integers', ints, flush=True)
        if keep_pat is not None:
            for name in names:
                if not any(s_ in name for s_ in keep_pat):
                    for c in range(ncase):
                        if (name, c) in kept:
                            kept[(name, c)] = (None, kept[(name, c)][1])
        save()

    def run_quick(names):
        for name in names:
            b = built[name]
            t0 = time.perf_counter()
            try:
                for c in range(ncase):
                    v = invoke(b, jnp.asarray(inputs_u[c]), c)
                    jax.block_until_ready(v)
                    f = np.asarray(v[0])
                    assert np.isfinite(f).all(), name
                    vh = (None,) + tuple(host(v[1:]))
                    row = dict(name=name, case=c, family=b['family'], field_sha256=sha_array(f), **score(f, c))
                    row.update(rom_row(b, vh) if b['kind'] == 'rom' else fom_row(b, vh))
                    rep['quick'].append(row)
                    kept[(name, c)] = (f, vh[1:4] if b['kind'] == 'rom' else None)
                    extra = dict(internal_latents=np.asarray(v[7])) if b['kind'] == 'rom' else {}
                    np.savez_compressed(out / f'restricted_{name}_case{c}.npz', fields=f[:, ::sub, ::sub], **extra)
                    if c in cfg['audit_cases'] and any(s_ == name for s_ in cfg['audit_arms']):
                        np.save(out / f'full_{name}_case{c}.npy', f)
            except Exception as exc:                     # noqa: BLE001
                if not is_oom(exc):
                    raise
                rep['dropped'].append(dict(name=name, phase='quick', reason=str(exc)[:300]))
                del built[name]
                jax.clear_caches()
                continue
            subjects.append(name)
            if cfg.get('prune_each_arm') and keep_pat is not None and not any(s_ in name for s_ in keep_pat):
                for c in range(ncase):                   # large cohorts: never hold more than one arm's fields
                    kept[(name, c)] = (None, kept[(name, c)][1])
            rows = [x for x in rep['quick'] if x['name'] == name]
            print('QUICK', name, 'evolved%', round(100 * max(x['same_grid_evolved'] for x in rows), 4),
                  'all%', round(100 * max(x['same_grid_all'] for x in rows), 4),
                  'stalled', sum(x.get('stalled_exits', x.get('stalled_steps', 0)) for x in rows),
                  round(time.perf_counter() - t0, 1), el(), flush=True)
            save()
        parity_and_prune(names)

    # (a) full-order arms, (b) every rule that needs NO fit -> the early answer, before any population work
    run_quick([add(fs['name'], family='fom', kind='fom', setting=fs) for fs in cfg['fom_settings'] + cfg.get('coarse_fom', [])])
    for rung in cfg['rungs']:
        for rs in rung['rules']:
            if not rs.get('refit'):
                if not rs.get('exact'):
                    build_rule(rung, rs)
                run_quick(register(rung, rs))
    rep['phases']['early_answer_seconds'] = el()
    # hb4k02 lost EVERY ROM arm to a mislabelled 'OOM' and carried on for ten minutes: never again
    assert any(built[n]['kind'] == 'rom' for n in subjects), ('no ROM arm survived the early answer', rep['dropped'])

    # (c) held-out populations: the audited dense query at the target mesh on every trajectory of every draw,
    #     each per-step state tagged (draw, trajectory, k)
    assert int(pc['mesh']) == L, 'this lane certifies at the target mesh'
    steps_total = int(round(.25 / dt))

    def tagged(v, d, i):
        n_ = len(np.asarray(v[7]))
        assert n_ == steps_total + 1
        return np.stack((np.full(n_, d), np.full(n_, i), np.arange(n_)), 1)

    for q in sorted({r['q'] for r in cfg['rungs']}):
        t0 = time.perf_counter()
        M = 4 * (K + q)
        Phi, lam, _ = e.modes(Lp, M)
        P = jnp.asarray(Phi)
        dd = dict(A=P.T @ Gp, lam=jnp.asarray(lam), G=Gp, Phi=P)
        cold, head = cold_for(q)
        qf = TF.make_query(params, Cfull[:, :q], K, q, Lp, dt, trust, 'dense', 'base', ic_budget=st['ic_budget'],
                           step_budget=st['step_budget'], gtol=st['gtol'], ic_gtol=cfg['ic_gtol'], linear=lin(K + q),
                           inner_damping=cfg['inner_damping'], tau_y=cfg['tau_y'])
        hv = jax.jit(jax.vmap(head))
        rows, metas, stalls = [], [], 0
        for d, dr in enumerate(draws):
            for i in dr:
                ph = pop_physical[i]
                v = qf(jnp.asarray(e.initial(Lp, ph)), float(ph[4]), dd, cold)
                wv = np.asarray(v[7])
                assert np.isfinite(wv).all() and np.linalg.norm(wv[-1] - wv[0]) > 0, ('frozen rollout', q, i)
                stalls += int(sum(r in (0, 2, 3) for r in np.asarray(v[3]).tolist()))
                rows.append(np.asarray(hv(jnp.asarray(wv))))
                metas.append(tagged(v, d, i))
        pops[q] = dict(cert=np.concatenate(rows), meta=np.concatenate(metas))
        np.savez_compressed(out / f'population_q{q}.npz', coefficients=pops[q]['cert'], meta=pops[q]['meta'])
        rep['populations'][str(q)] = dict(mesh=Lp, M=M, cert_states=int(len(pops[q]['cert'])),
                                          cert_sha256=sha_array(pops[q]['cert']), stalled_steps=stalls,
                                          seconds=time.perf_counter() - t0, draws=draws,
                                          source='audited dense query (topfix base, gtol %g) at the target mesh on '
                                                 'params_draw%s; every per-step state incl. the initial fit, tagged '
                                                 '(draw, trajectory, k); stalled steps counted, not removed'
                                                 % (st['gtol'], tuple(pc['source_draw'])))
        print('POPULATION q', q, 'states', len(pops[q]['cert']), 'stalled', stalls, el(), flush=True)
        del dd, P, Phi, qf
        save()

    # (d) held-out certificates (dense-query population) for every built rule
    for (q, M, name) in list(rules):
        certify_heldout(q, M, name)

    # (f) deployed-state certificate, PER ARM, on every draw: rho on the states that arm's own query visits
    for name in subjects:
        b = built[name]
        if not b.get('fastarm') or b.get('exact'):
            continue
        t0 = time.perf_counter()
        q, M = b['q'], b['M']
        _, head = cold_for(q)
        hv = jax.jit(jax.vmap(head))
        rows, metas, stall = [], [], 0
        for d, dr in enumerate(draws):
            for i in dr:
                ph = pop_physical[i]
                v = b['query'](jnp.asarray(e.initial(L, ph)), float(ph[4]), b['data'], b['cold'])
                stall += int(sum(r in (0, 2, 3) for r in np.asarray(v[3]).tolist()))
                rows.append(np.asarray(hv(jnp.asarray(np.asarray(v[7])))))
                metas.append(tagged(v, d, i))
        co, meta = np.concatenate(rows), np.concatenate(metas)
        o = operators(M)
        tg = H.dense_targets(G, co, o['sx'], o['sy'], L, chunk=cfg['target_chunk'])
        ops, info = rules[(q, M, b['rule'])]
        r = H.rho(ops['Pq'], H.sampled_advection(ops['G5'], co, L), tg)
        np.savez_compressed(out / f'deployed_{name}.npz', coefficients=co, meta=meta, rho=r)
        dep = draw_summaries(r, meta)
        dep.update(stalled_steps=stall, seconds=time.perf_counter() - t0)
        info['deployed'][name] = dep
        j = b['exact_steps']
        kk = f'k>={j}'
        per = [bool(info['heldout_population'][kk]['per_draw'][str(dd_)]['certified_primary'] and
                    dep[kk]['per_draw'][str(dd_)]['certified_primary']) for dd_ in range(len(draws))]
        npass = int(sum(per[:ncert]))
        status = 'confirmed' if npass == ncert else ('fails' if npass == 0 else f'marginal ({npass}/{ncert})')
        rep['arm_status'][name] = dict(rule=b['rule'], q=q, M=M, exact_steps=j, k_threshold=kk, draws_passed=npass,
                                       draws=ncert, per_draw_pass=per[:ncert],
                                       confirmation_pass=(per[ncert] if len(per) > ncert else None),
                                       status=status, control=b['control'],
                                       heldout_rho_max=info['heldout_population'][kk]['union']['rho_max'],
                                       deployed_rho_max=dep[kk]['union']['rho_max'],
                                       heldout_rho_max_all_k=info['heldout_population']['k>=0']['union']['rho_max'],
                                       deployed_rho_max_all_k=dep['k>=0']['union']['rho_max'],
                                       heldout_rho_max_k1=info['heldout_population']['k>=1']['union']['rho_max'],
                                       deployed_rho_max_k1=dep['k>=1']['union']['rho_max'])
        print('ARM-STATUS', name, status, 'heldout', f"{rep['arm_status'][name]['heldout_rho_max']:.4f}",
              'deployed', f"{rep['arm_status'][name]['deployed_rho_max']:.4f}", el(), flush=True)
        save()
    for name in subjects:
        if built[name].get('exact'):
            rep['arm_status'][name] = dict(rule='exact', q=built[name]['q'], M=built[name]['M'], exact_steps=None,
                                           status='exact residual', control=False)
    save()
    ctrl = [v_ for v_ in rep['arm_status'].values() if v_.get('control')]
    rep['gates']['control_rule_fails_certificate'] = dict(
        passed=(all(v_['status'] != 'confirmed' for v_ in ctrl) if ctrl else None),
        controls=[dict(q=v_['q'], M=v_['M'], rule=v_['rule'], status=v_['status'], heldout_rho_max=v_['heldout_rho_max'])
                  for v_ in ctrl],
        note='a control that PASSES the certificate means the historical regression did not persist at this mesh; '
             'the control argument is then void here and is reported so, nothing is tuned until it fails')
    save()

    # --------------------------------------------------- 5. dense truth arms ----
    for rung in cfg['rungs']:
        if not rung.get('dense'):
            continue
        q, M = rung['q'], rung['M']
        if time.perf_counter() - begin > cfg['dense_deadline_seconds']:
            rep['dropped'].append(dict(name=f'dense_q{q}_M{M}', phase='dense', reason='deadline'))
            continue
        o = operators(M)
        cold, head = cold_for(q)
        C = Cfull[:, :q]
        data = dict(A=o['A'], lam=o['lam'], G=G, sx=o['sx'], sy=o['sy'])
        tab = HF.build_tables(params, C, K, data, cold)
        d = K + q
        chunks = next(k for k in range(1, d + 1) if d % k == 0 and d // k <= cfg['dense_tangent_group'])
        fq = HF.make_query(params, C, K, q, L, dt, trust, 'dense', ic_budget=st['ic_budget'],
                           step_budget=st['step_budget'], gtol=st['gtol'], ic_gtol=cfg['ic_gtol'],
                           ridge=cfg['inner_damping'], tangent_chunks=chunks)
        try:
            for c in rung.get('dense_cases', list(range(ncase))):
                t0 = time.perf_counter()
                v = fq(jnp.asarray(inputs_u[c]), float(physical[c, 4]), data, cold, tab)
                jax.block_until_ready(v)
                f = np.asarray(v[0])
                b = dict(q=q, M=M, gtol=st['gtol'])
                row = dict(name=f'q{q}_M{M}_dense', case=c, seconds=time.perf_counter() - t0, **score(f, c),
                           **rom_row(b, (None,) + tuple(host(v[1:]))))
                # the deployed arms against their own dense truth, same case
                row['deployed_vs_dense'] = {n: float(np.linalg.norm(kept[(n, c)][0] - f) / np.linalg.norm(f))
                                            for n in subjects if built[n]['kind'] == 'rom' and built[n]['q'] == q
                                            and built[n]['M'] == M and kept.get((n, c), (None,))[0] is not None}
                rep['dense_truth'].append(row)
                np.savez_compressed(out / f'restricted_q{q}_M{M}_dense_case{c}.npz', fields=f[:, ::sub, ::sub],
                                    internal_latents=np.asarray(v[7]))
                print('DENSE', q, M, 'case', c, 'evolved%', round(100 * row['same_grid_evolved'], 4), 'iters',
                      row['total_iterations'], round(row['seconds'], 1), el(), flush=True)
                save()
        except Exception as exc:                     # noqa: BLE001
            if not is_oom(exc):
                raise
            rep['dropped'].append(dict(name=f'dense_q{q}_M{M}', phase='dense', reason=str(exc)[:300]))
            jax.clear_caches()
        del fq

    # ------------------------------------------------------- 6. timed panel ----
    timed = [n for n in subjects if n not in cfg.get('untimed', [])]
    for n in timed:                                   # re-warm after clear_caches
        jax.block_until_ready(invoke(built[n], jnp.asarray(inputs_u[0]), 0))
    order_rng = np.random.default_rng(cfg['order_seed'])
    quick_sha = {(x['name'], x['case']): x['field_sha256'] for x in rep['quick']}
    for r_ in range(cfg['reps']):
        for c in range(ncase):
            for i in order_rng.permutation(len(timed)):
                name = timed[int(i)]
                b = built[name]
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs_u[c], copy=True))
                jax.block_until_ready(u)
                g0_ = time.perf_counter()
                v = invoke(b, u, c)
                jax.block_until_ready(v)
                gs = time.perf_counter() - g0_
                f = np.asarray(v[0])
                hs = time.perf_counter() - ht
                same = sha_array(f) == quick_sha[(name, c)]       # full-field SHA256 against the untimed quick run
                row = dict(name=name, case=c, rep=r_, family=b['family'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), identical_to_first_rep=same, **score(f, c))
                vh = (None,) + tuple(host(v[1:]))
                row.update(rom_row(b, vh) if b['kind'] == 'rom' else fom_row(b, vh))
                rep['invocations'].append(row)
        print('TIMED rep', r_, el(), flush=True)
        save()
    cover = {(n, c) for n in timed for c in range(ncase)}
    counts = {k: sum(1 for x in rep['invocations'] if (x['name'], x['case']) == k) for k in cover}
    rep['gates']['repetition_output_identical'] = dict(
        passed=bool(rep['invocations']) and all(x['identical_to_first_rep'] for x in rep['invocations']),
        basis='full-field SHA256 of every timed repetition against the untimed quick run of the same arm and case')
    rep['gates']['five_retained_repetitions_everywhere'] = dict(
        passed=bool(counts) and min(counts.values()) >= cfg.get('required_reps', 5), minimum=min(counts.values()) if counts else 0)

    # ------------------------------------------------------------ 7. profile ----
    def timeit(fn, reps=7):
        jax.block_until_ready(fn())
        ts = []
        for _ in range(reps):
            e.burn(.05)
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    for name in cfg.get('profile_arms', []):
        if name not in built or not built[name].get('hfast'):
            continue
        b = built[name]
        pr = b['parts']
        u = jnp.asarray(inputs_u[0])
        nu = float(physical[0, 4])
        ini = pr['initialize'](u, b['data'], b['cold'], b['tab'])
        w0, scale = ini[0], ini[1]
        ev = pr['evolve'](w0, nu, scale, b['data'], b['tab'])
        W = jnp.concatenate((w0[None], ev[0]))[::int(round(.05 / dt))]
        its = int(np.sum(np.asarray(ev[2])))
        prev = b['data']['A'] @ (heads[b['q']](w0))
        args = (prev, nu, b['data'], b['tab'])
        r, J = pr['evalJ'](w0, args)
        normal = jax.jit(lambda J, r: jnp.linalg.solve(J.T @ J + 1e-6 * jnp.diag(jnp.diag(J.T @ J)), -(J.T @ r)))
        gram = jax.jit(lambda J: J.T @ J)
        entry = dict(arm=name, case=0, iterations=its, steps=int(len(np.asarray(ev[2]))),
                     retries=int(np.sum(np.asarray(ev[5]))), J_shape=list(J.shape), m=b['m'],
                     whole_query_ms=1e3 * timeit(lambda: b['query'](u, nu, b['data'], b['cold'])),
                     initialize_ms=1e3 * timeit(lambda: pr['initialize'](u, b['data'], b['cold'], b['tab'])),
                     evolve_ms=1e3 * timeit(lambda: pr['evolve'](w0, nu, scale, b['data'], b['tab'])),
                     decode_ms=1e3 * timeit(lambda: pr['decode'](W, b['data'], b['tab'])),
                     evalJ_ms=1e3 * timeit(lambda: pr['evalJ'](w0, args), 30),
                     residual_ms=1e3 * timeit(lambda: pr['res'](w0, *args), 30),
                     gram_ms=1e3 * timeit(lambda: gram(J), 30),
                     gram_plus_solve_ms=1e3 * timeit(lambda: normal(J, r), 30))
        entry['evolve_ms_per_iteration'] = entry['evolve_ms'] / max(its, 1)
        rep['profile'].append(entry)
        print('PROFILE', json.dumps(entry), flush=True)
        save()

    # ------------------------------------- 8. refined reference, physical error ----
    rc = cfg.get('reference')
    if rc:
        rf, rt = rc['mesh'], rc['dt']
        # everything the panel produced is already on disk; release the ROM side before the big solve
        for b_ in built.values():
            for k_ in ('data', 'tab', 'query', 'parts', 'cold'):
                b_.pop(k_, None)
        rules.clear(); ops_M.clear(); targets.clear(); bank._grid.clear(); colds.clear()
        del G, Gp
        jax.clear_caches()
        try:
            q_ref, _ = e.make_fom(rf, rt)
            for c in rc['cases']:
                if time.perf_counter() - begin > rc['deadline_seconds']:
                    rep['dropped'].append(dict(name=f'reference_case{c}', phase='reference', reason='deadline'))
                    continue
                t0 = time.perf_counter()
                fr, it, rn = q_ref(jnp.asarray(e.initial(rf, physical[c])), float(physical[c, 4]), rc['ntol'], rc['ltol'])
                fr = np.array(np.asarray(fr)[:, ::rf // L, ::rf // L], copy=True)
                rn = np.asarray(rn)
                accepted = bool(np.isfinite(fr).all() and np.isfinite(rn).all() and rn.max() <= rc['accept_residual'])
                rep['reference'].append(dict(case=c, mesh=rf, dt=rt, max_relative_residual=float(np.nanmax(rn)),
                                             accepted=accepted, accept_residual=rc['accept_residual'],
                                             seconds=time.perf_counter() - t0, restricted_to=L, field_sha256=sha_array(fr)))
                np.savez_compressed(out / f'reference_restricted_case{c}.npz', fields=fr[:, ::sub, ::sub])
                if accepted:
                    nr = float(np.linalg.norm(fr[0]))
                    for (name, cc), (f, _) in kept.items():
                        if cc != c or f is None:
                            continue
                        pe = rel_per_time(f, fr, nr)
                        rep['physical'].append(dict(name=name, case=c, reference_per_time=pe, reference_all=float(max(pe)),
                                                    reference_evolved=float(max(pe[1:]))))
                    pe = rel_per_time(truth[c], fr, nr)
                    rep['physical'].append(dict(name='__truth__', case=c, reference_per_time=pe, reference_all=float(max(pe)),
                                                reference_evolved=float(max(pe[1:]))))
                print('REFERENCE', c, 'accepted', accepted, round(time.perf_counter() - t0, 1), el(), flush=True)
                save()
        except Exception as exc:                         # noqa: BLE001 - the panel above is already saved
            rep['dropped'].append(dict(name='reference', phase='reference', reason=str(exc)[:400]))
            print('REFERENCE FAILED', str(exc)[:200], flush=True)
            save()

    rep['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert rep['checkpoint_sha256'] == rep['checkpoint_sha256_after']
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HIRES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
