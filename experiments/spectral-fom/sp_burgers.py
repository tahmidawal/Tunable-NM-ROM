"""spectral-fom, Burgers 2D: DST fixed-point backward-Euler solver vs the burgers-bank-knob settings.

ROM arms: the lane's selected accurate and fast arms, built by a verbatim copy of the setup of
`experiments/burgers-bank-knob/bankknob.py` main() at commit b393fa55 (cohort, rotated nested bank, Phi-free
operators, rules, models, cold start, build_arm, invoke), with three deviations that do not change any arm's
arithmetic: no parent (unrotated) bank, only the column blocks up to the largest selected R' are kept (a truncated
arm reads only its prefix blocks, bkfast.prefix), and the certificate / quick / ladder phases are not run.  A
parity gate checks each re-run arm's per-case evolved error against the lane's recorded value (same fields).

Reference (the paper's): `fft_tight` = mr-burgers2d iterative_paths.make_fom(L, .005, 'fft') at ntol 1e-6,
ltol 1e-8.  Error = max over t = .05..0.25 of ||u - u_ref|| / ||u0||, worst over the six dev6 cases.

Spectral subjects: spec_core.make_burgers (Picard / modal-Helmholtz fixed point on the paper's residual, and the
one-sweep IMEX), a (dt, ntol) ladder.  The DST implementation inside H^{-1} (fft | mm) is chosen per mesh by an
untimed micro-benchmark before the timed phases (recorded).
Validation (untimed): case 0, the paper's Newton-BiCGStab at ntol 1e-10 / ltol 1e-12 vs spectral Picard at ntol
1e-10 (must agree to <= 1e-8 of ||u0||); CONTROL: the same Picard on a residual with nu -> 1.01 nu must NOT agree.
"""
from __future__ import annotations

import argparse
import hashlib
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
import hops as H
import hfast as HF
import xfast as XF
import bkfast as BK            # patches hops.bank_apply for the nested rotated bank
from bankknob import arm_name, rel_per_time
import spec_core as SC
import spec_timing as ST

sha_array, sha_file, host = LD.sha_array, LD.sha_file, LD.host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    lane = json.loads(Path(cfg['lane_config']).read_text())          # the lane's committed per-mesh config
    sel = json.loads(Path(cfg['lane_selection']).read_text())         # the lane's committed selection
    if a.smoke:
        cfg.update(repetitions=2, burn_seconds=0.001, cooldown_seconds=0.1, phase_dummy_seconds=0.1)
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    uuid0 = ST.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory
    except FileNotFoundError:
        inventory = None
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)

    ck = pickle.load(open(cfg['checkpoint'], 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = int(lane['intervals'])
    dt = lane['dt']
    if a.smoke:
        L = int(cfg.get('smoke_intervals', L))
    st = lane['strict']
    edges = [0] + list(lane['ladder'])
    names = dict(rom_accurate=sel['accurate'], rom_fast=sel['fast'])
    specs = {}
    for s in lane['arms']:
        s2 = dict(s)
        s2.setdefault('q', 0)
        if arm_name(s2) in names.values():
            specs[arm_name(s2)] = s2
    assert set(specs) == set(names.values()), (sorted(specs), names)
    Rmax = max(int(s.get('Rp', R)) for s in specs.values())
    assert Rmax in edges
    rep = dict(problem='burgers2d', config=cfg, lane_config=lane, lane_selection=sel, commit=os.environ.get('SOURCE_COMMIT'),
               job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               gpu_uuid=uuid0, nvidia_smi=inventory, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               jax_version=jax.__version__, smoke=bool(a.smoke), intervals=L, dt=dt, K=K, R=R, selected=names,
               checkpoint_sha256=sha_file(cfg['checkpoint']),
               error_convention='max over t=.05..0.25 of ||u-u_ref||/||u0|| (full (L+1)^2 nodes) vs the paper fft_tight reference; worst over dev6',
               complete=False)
    save = lambda: LD.dump(out / 'result.json', SC_clean(rep))

    # ---- cohort (bankknob.py verbatim)
    physical = np.concatenate((e.params_draw(lane['eval_seed'], lane['eval_cases']),
                               e.params_draw(lane['eval_fresh_seed'], lane['eval_fresh_cases'])))
    rep['physical_sha256'] = sha_array(physical)
    assert a.smoke or rep['physical_sha256'] == lane['expected_physical_sha256']   # local numpy differs in the last ulp
    if a.smoke:
        physical = physical[:2]
    ncase = len(physical)
    train_physical = e.params_draw(lane['train_seed'], lane['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical)
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = [float(np.linalg.norm(u)) for u in inputs_u]
    sub = max(1, L // 256)

    # ---- rotation, bank (bankknob.py verbatim, no parent; blocks beyond Rmax dropped)
    rz = np.load(cfg['rotation'])
    T, Lrot = np.asarray(rz['T']), np.asarray(rz['L'])
    assert sha_file(cfg['rotation']) == lane['rotation_sha256']
    Tj = jnp.asarray(T)
    base = A.CoordBank(params, K, R)
    t0 = time.perf_counter()
    n = (L - 1) ** 2
    nrb = int(lane.get('bank_blocks') or np.ceil(n * R / H.MAX_GEMM_ELEMENTS))
    x = np.arange(1, L) / L
    redges = np.linspace(0, L - 1, nrb + 1).astype(int)
    nkeep = edges.index(Rmax)
    rotate = jax.jit(lambda g, t: tuple((g @ t)[:, a_:b_] for a_, b_ in zip(edges[:-1], edges[1:])))
    Grot = []
    for i0, i1 in zip(redges[:-1], redges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        g = jax.block_until_ready(base.at(xy, chunk=8192))
        blocks = jax.block_until_ready(rotate(g, Tj))
        Grot.append(tuple(blocks[:nkeep]))
        del g, blocks
    Grot = tuple(Grot)
    rep['bank'] = dict(rows=n, row_blocks=nrb, column_edges=edges, kept_blocks=nkeep, seconds=time.perf_counter() - t0,
                       rotated_bytes=int(sum(c.nbytes for rb in Grot for c in rb)))
    print('BANK', rep['bank'], el(), flush=True)

    # ---- reference (the paper's fft_tight) and the paper's FOM for validation
    tight = next(fs for fs in lane['fom_settings'] if fs['name'] == lane['same_grid_reference'])
    fom_fn, fom_pre = ip.make_fom(L, tight['dt'], 'fft')
    truth = {}
    for c in range(ncase):
        v = fom_fn(jnp.asarray(inputs_u[c]), float(physical[c, 4]), tight['ntol'], tight['ltol'], *fom_pre)
        jax.block_until_ready(v)
        f = np.asarray(v[0])
        assert np.isfinite(f).all() and np.asarray(v[2]).max() <= tight['ntol'] * (1 + 1e-9)
        truth[c] = f
    np.save(out / 'fields' / 'truth_case0.npy', truth[0])
    for c in range(ncase):
        np.save(out / 'fields' / f'truth_restricted_case{c}.npy', truth[c][:, ::sub, ::sub])
    print('TRUTH', el(), flush=True)

    def score(f, c):
        sg = rel_per_time(f, truth[c], n0[c])
        fr, tr = f[:, ::sub, ::sub], truth[c][:, ::sub, ::sub]
        rr = [float(np.linalg.norm(x_ - y_)) for x_, y_ in zip(fr, tr)]
        return dict(same_grid_evolved=float(max(sg[1:])), same_grid_per_time=sg,
                    restricted_abs_per_time=rr)

    # ---- operators, rules, models, arms (bankknob.py verbatim, parent-free)
    dfile = Path(cfg['inputs']) / lane['directions_file']
    Cfull = jnp.asarray(np.ascontiguousarray(np.load(dfile)['C']))
    assert lane['directions_sha256'] == sha_file(dfile)
    ops_M = {}

    def operators(M):
        if M not in ops_M:
            kx, ky, lam = H.modes_lean(L, M)
            sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
            o = dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sx, sy=sy,
                     Arot=BK.project_nested(Grot, edges, Rmax, sx, sy, L))
            ops_M[M] = jax.block_until_ready(o)
        return ops_M[M]
    rule_nodes, G5 = {}, {}

    def rule(name):
        if name in rule_nodes:
            return rule_nodes[name]
        rs = lane['rules'][name]
        if 'lattice' in rs:
            ij, w = H.lattice_rule(L, int(rs['lattice']))
        else:
            fp = Path(cfg['inputs']) / rs['file']
            assert sha_file(fp) == rs['sha256'], rs['file']
            z = np.load(fp)
            ij = H.transfer_nodes(z['nodes'].astype(int), rs['mesh'], L)
            w = np.asarray(z['weights'], float) * (L / rs['mesh']) ** 2
        keep = w > 0
        ij, w = ij[keep], w[keep]
        g5 = base.stencil(ij, L)
        G5[name] = dict(rot=jnp.einsum('msr,rp->msp', g5, Tj))
        rule_nodes[name] = (ij, w)
        return rule_nodes[name]
    Pq_cache = {}

    def Pq(name, M):
        if (name, M) not in Pq_cache:
            ij, w = rule(name)
            o = operators(M)
            Pq_cache[(name, M)] = jnp.asarray(H.phi_rows(L, o['kx'], o['ky'], ij) * w[:, None])
        return Pq_cache[(name, M)]
    trust0 = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // lane['decoder_code_subsample'])])
    hv0 = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))
    Hall = np.concatenate([np.asarray(hv0(jnp.asarray(Zold[s:s + 8192]))) for s in range(0, len(Zold), 8192)])
    Hsub = np.asarray(hv0(jnp.asarray(Zsub)))

    def model(kind, Rp):
        if kind == 'trunc':
            m_ = dict(params=BK.fold_head(params, Lrot, Rp), C=jnp.asarray(Lrot[:Rp]) @ Cfull,
                      bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust0, codes=Zsub,
                      A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm]['rot'][:, :, :Rp])
        else:
            assert kind == 'lin'
            codes_all = Hall @ Lrot[:Rp].T
            trust = .01 * float(np.max(np.linalg.norm(codes_all - codes_all.mean(0), axis=1)))
            m_ = dict(params=None, C=None, bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust,
                      codes=Hsub @ Lrot[:Rp].T, A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm]['rot'][:, :, :Rp])
        return m_

    def chunks_for(d):
        return next(k for k in range(1, d + 1) if d % k == 0 and d // k <= lane['dense_tangent_group'])

    def build_arm(s):
        v = s['variant']
        j = int(v.get('exact_steps', 0))
        kind, Rp, q, M = s['model'], int(s.get('Rp', R)), int(s['q']), int(s['M'])
        m_ = model(kind, Rp)
        o = operators(M)
        if kind == 'lin':
            head, codes = (lambda w_: w_), m_['codes']
        else:
            head = TF.corrected_head(m_['params'], m_['C'][:, :q], K)
            codes = np.concatenate((m_['codes'], np.zeros((len(m_['codes']), q))), 1)
        cold = A.build_cold(m_['bank'], head, codes, lane['cold_axis_points'])[0]
        rule(s['rule'])
        data = dict(A=m_['A'](M), lam=o['lam'], G=m_['G'], G5=m_['G5'](s['rule']), Pq=Pq(s['rule'], M),
                    sx=o['sx'], sy=o['sy'])
        common = dict(step_budget=st['step_budget'], gtol=s['gtol'], ridge=lane['inner_damping'],
                      solver=v.get('solver', 'lu'), clip=bool(v.get('clip')), lam_carry=bool(v.get('lamcarry')),
                      predictor='quad' if v.get('pred2') else 'lin')
        if kind == 'lin':
            fq = BK.make_linear_query(Rp, L, dt, m_['trust'], exact_steps=j, tangent_chunks=chunks_for(Rp), **common)
            tab = {}
        else:
            C = m_['C'][:, :q]
            tab = HF.build_tables(m_['params'], C, K, data, cold)
            fq = XF.make_query(m_['params'], C, K, q, L, dt, m_['trust'], 'eq', exact_steps=j,
                               ic_budget=st['ic_budget'], ic_gtol=lane['ic_gtol'], tangent_chunks=chunks_for(K + q),
                               **common)
        return lambda u, nu, _t=tab, _f=fq, _d=data, _c=cold: _f(u, nu, _d, _c, _t)

    rom = []
    for role, name in names.items():
        qf = build_arm(specs[name])

        def call(c, qf=qf):
            nu = float(physical[c, 4])
            return ST.query(lambda u: qf(u, nu), inputs_u[c])
        rom.append(dict(name=name, role=role, call=call))
    print('ARMS built', el(), flush=True)

    # ---- spectral subjects: DST variant chosen by an untimed micro-benchmark
    bench = {}
    for variant in cfg.get('dst_candidates', ['fft', 'mm']):
        fn = SC.make_burgers(L, dt, e.residual, max_iter=1, dst=variant)
        u0j = jnp.asarray(inputs_u[0])
        jax.block_until_ready(fn(u0j, float(physical[0, 4]), 0.0))
        ts = []
        for _ in range(5):
            t_ = time.perf_counter()
            jax.block_until_ready(fn(u0j, float(physical[0, 4]), 0.0))
            ts.append(time.perf_counter() - t_)
        bench[variant] = float(np.median(ts))
    dst_variant = min(bench, key=bench.get)
    rep['dst_variant_benchmark'] = dict(seconds_one_sweep_trajectory=bench, chosen=dst_variant)
    spec = []
    for sp in cfg['spectral_settings']:
        fn = SC.make_burgers(L, sp['dt'], e.residual, max_iter=sp.get('max_iter', 400), dst=dst_variant,
                             predictor=bool(sp.get('predictor', False)))
        nm = sp['name']

        def call(c, fn=fn, ntol=sp['ntol']):
            return ST.query(lambda u: fn(u, float(physical[c, 4]), ntol), inputs_u[c])
        spec.append(dict(name=nm, role='spectral', call=call, setting=sp))
    save()

    # ---- V: validation (case 0)
    V = {}
    tt = time.perf_counter()
    vf = fom_fn(jnp.asarray(inputs_u[0]), float(physical[0, 4]), cfg['validation_ntol'], cfg['validation_ltol'], *fom_pre)
    vf = [np.asarray(x_) for x_ in jax.device_get(vf)]
    V['paper_fom_tight'] = dict(ntol=cfg['validation_ntol'], ltol=cfg['validation_ltol'], seconds=time.perf_counter() - tt,
                                max_relative_residual=float(vf[2].max()), newton_total=int(vf[1].sum()),
                                converged=bool(vf[2].max() <= cfg['validation_ntol'] * (1 + 1e-6)))
    pic = SC.make_burgers(L, dt, e.residual, max_iter=2000, dst=dst_variant)
    vs = [np.asarray(x_) for x_ in jax.device_get(pic(jnp.asarray(inputs_u[0]), float(physical[0, 4]), cfg['validation_ntol']))]
    V['spectral_tight'] = dict(ntol=cfg['validation_ntol'], max_relative_residual=float(vs[2].max()),
                               iterations_total=int(vs[1].sum()), max_iterations_per_step=int(vs[1].max()),
                               converged=bool(vs[2].max() <= cfg['validation_ntol'] * (1 + 1e-6)))
    agree = max(rel_per_time(vs[0], vf[0], n0[0]))
    V['agreement_over_u0'] = agree
    wrong = lambda u, prev, nu, dt_, L_: e.residual(u, prev, 1.01 * nu, dt_, L_)
    ctl = SC.make_burgers(L, dt, wrong, max_iter=2000, dst=dst_variant)
    vc = np.asarray(ctl(jnp.asarray(inputs_u[0]), float(physical[0, 4]), cfg['validation_ntol'])[0])
    V['control'] = dict(name='picard_on_residual_with_1.01nu_CONTROL', agreement_over_u0=max(rel_per_time(vc, vf[0], n0[0])),
                        must_exceed=cfg['validation_agreement_limit'])
    V['control']['failed_as_required'] = bool(V['control']['agreement_over_u0'] > cfg['validation_agreement_limit'])
    V['gates'] = dict(paper_fom_tight_converged=V['paper_fom_tight']['converged'],
                      spectral_tight_converged=V['spectral_tight']['converged'],
                      same_discrete_solution=bool(agree <= cfg['validation_agreement_limit']),
                      control_failed=V['control']['failed_as_required'])
    rep['validation'] = V
    print('VALIDATION', V['gates'], agree, V['control']['agreement_over_u0'], el(), flush=True)
    save()

    # ---- T: A-B-A timing
    saved = {}
    full_keep = set(names.values()) | set(cfg['full_field_subjects'])

    def record(sub_, c, rep_, phase, field, row, raw):
        assert np.isfinite(field).all(), sub_['name']
        h = sha_array(field)
        key = (sub_['name'], c)
        if key not in saved:
            np.save(out / 'fields' / f"{sub_['name']}_restricted_case{c}.npy", field[:, ::sub, ::sub])
            if c == 0 and sub_['name'] in full_keep:
                np.save(out / 'fields' / f"{sub_['name']}_case0.npy", field)
            saved[key] = h
        d = dict(field_sha256=h, matches_saved_field=bool(h == saved[key]), **score(field, c))
        if sub_['role'] == 'spectral':
            it, rr = np.asarray(raw[1]), np.asarray(raw[2])
            d.update(iterations_total=int(it.sum()), max_iterations_per_step=int(it.max()),
                     max_relative_residual=float(rr.max()),
                     converged=bool(sub_['setting'].get('max_iter', 400) == 1 or rr.max() <= sub_['setting']['ntol'] * (1 + 1e-9)))
        return d

    for s_ in rom + spec:
        for _ in range(2):
            s_['call'](0)
    harness = ST.ABA(cfg, uuid0, record, log=lambda s_: print(s_, el(), flush=True), save=save)
    rep['invocations'] = harness.invocations
    gates = harness.run(rom, spec, list(range(ncase)), cfg['repetitions'], cfg['order_seed'])
    rep['phase_breaks'] = harness.breaks
    rep['timing_gates'] = gates
    rep['timings'] = harness.timings(rom, spec)
    errs = {}
    for s_ in rom + spec:
        ev = [x_['same_grid_evolved'] for x_ in harness.invocations if x_['name'] == s_['name'] and x_['rep'] == 0
              and x_['phase'] in ('romA1', 'spec')]
        errs[s_['name']] = dict(worst=float(max(ev)), median=float(np.median(ev)), per_case=ev)
    rep['errors'] = errs
    # parity with the lane's recorded errors for the same arms (same code, same cohort, same reference)
    lane_err = {}
    lr = cfg.get('lane_quick_errors')
    if lr:
        lane_err = json.loads(Path(lr).read_text())
    par = []
    for s_ in rom:
        if s_['name'] in lane_err:
            mine = errs[s_['name']]['per_case']
            theirs = lane_err[s_['name']]
            dev_ = max(abs(x_ - y_) / y_ for x_, y_ in zip(mine, theirs))
            par.append(dict(name=s_['name'], worst_relative_deviation=dev_, passed=bool(dev_ <= 1e-8)))
    rep['lane_error_parity'] = par
    rep['gates'] = dict(**V['gates'], deterministic=all(x_['matches_saved_field'] for x_ in harness.invocations),
                        spectral_converged=all(x_.get('converged', True) for x_ in harness.invocations),
                        lane_error_parity=bool(par) and all(x_['passed'] for x_ in par),
                        drift=gates['drift']['passed'], neighbour=gates['neighbour']['passed'],
                        device_guard=ST.gpu_uuid() == uuid0)
    rep['checkpoint_sha256_after'] = sha_file(cfg['checkpoint'])
    print('GATES', rep['gates'], flush=True)
    print('TIMINGS', {k: round(v['median_ms'], 2) for k, v in rep['timings'].items()}, flush=True)
    print('ERRORS%', {k: round(100 * v['worst'], 4) for k, v in errs.items()}, flush=True)
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    save()
    (out / ('COMPLETE' if all(rep['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(rep['gates']) + '\n')


def SC_clean(x):
    if isinstance(x, dict):
        return {k: SC_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [SC_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == '__main__':
    main()
