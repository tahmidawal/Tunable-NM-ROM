"""tensor-crossover, 3D driver (PROTOCOL.md). One allocation, all widths given in the config; per width: qualify rules on
dev4, build the tensor of the selected rules, check trajectory agreement, time full queries and the nonlinear term, record
bytes/memory.

The query is the vendored quadrature-burgers3d `offmesh.make_fsc_rule` unchanged, with `contract_offmesh` (quadrature) or
`contract_tensor` (precomputed tensor, data['Ts']). python tc3d.py --config configs/<job>.json --out output
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / 'vendor' / 'tree' / 'experiments'
sys.path.insert(0, str(V / 'quadrature-burgers3d'))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

import offmesh as OM  # noqa: E402
from offmesh import C, TB  # noqa: E402
import tcommon as TC  # noqa: E402

block = jax.block_until_ready
COHORTS = {'dev4': (923651, 4)}


def cfg_path(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest' and jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda s: print(f'[{el()}s] {s}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    n = int(cfg['mesh'])
    widths = [int(r) for r in cfg['widths']]
    mp = cfg_path(cfg['model'])
    rep = dict(config=cfg, dim=3, mesh=n, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               jax_version=jax.__version__, x64=True, precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               model_sha256=sha_file(mp), gates={}, widths={}, complete=False)
    assert rep['model_sha256'] == cfg['expected_model_sha256'], rep['model_sha256']
    save = lambda: json.dump(TC.clean(rep), open(out / 'result.json', 'w'), indent=1)
    save()
    burn = TC.make_burn()
    b = pickle.loads(mp.read_bytes())
    bank = jax.tree_util.tree_map(jnp.asarray, b['params'])
    T = np.asarray(b['rotation'])
    spread = b['coefficient_rms_spread']
    R = int(T.shape[1])

    # ---------------------------------------------------------------- rules (G-R) ----
    fams = cfg['families']
    man = json.loads(cfg_path(cfg['rules_manifest']).read_text())
    rp = cfg_path(cfg['rules_npz'])
    assert sha_file(rp) == man['npz_sha256']
    z = np.load(rp)
    RULES, bad = {}, []
    for fam, f in fams.items():
        for nm in f['ladder']:
            X, w = np.asarray(z[f'{nm}_X']), np.asarray(z[f'{nm}_w'])
            if hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest() != man['rules'][nm]['sha256']:
                bad.append(nm)
            RULES[nm] = (X, w)
    rep['gates']['G_R'] = dict(checked=len(RULES), mismatched=bad, passed=not bad)
    assert not bad, bad
    rrec = {nm: dict(m=int(len(w))) for nm, (X, w) in RULES.items()}
    rep['rules'] = rrec

    # ---------------------------------------------------------------- tables ----
    order_, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order_, lam_all)
    t0 = time.perf_counter()
    tb = TB.build_tables(n, bank, T, sorted(set(widths) | {R}), M_of, log=log, tensor=False)
    rep['setup'] = dict(table_seconds=time.perf_counter() - t0, gram_condition=tb['gram_cond'] ** 2, M_max=int(tb['M_max']),
                        bank_bytes=TC.tree_bytes(tb['GTb']))
    assert rep['setup']['gram_condition'] <= 1e8
    kx = tb['kxyz']
    t0 = time.perf_counter()
    OMT = {nm: OM.offmesh_tables(n, bank, T, *RULES[nm], kx) for nm in RULES}
    rep['setup']['offmesh_table_seconds'] = time.perf_counter() - t0
    seed, count = COHORTS[cfg['cohort']]
    tab = C.table(seed, count)
    rep['cohort'] = dict(name=cfg['cohort'], seed=seed, count=count, sha256=tab['sha256'])
    cases = list(cfg.get('cases', range(count)))
    U0 = [block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases]
    NU = [float(tab['nu'][j]) for j in cases]
    log(f'tables built; cases {cases}')
    save()
    K = int(cfg['kernel_evals'])
    tcfg = cfg['timing']
    frozen_w = int(cfg['frozen_width'])
    order_w = [frozen_w] + [r for r in widths if r != frozen_w] if frozen_w in widths else list(widths)
    TAU = dict(cfg.get('tau', {}))          # a split job (frozen width elsewhere) passes tau from the frozen-width job
    jac_q = lambda c, d: (lambda J: (0.5 * (J @ c), J))(OM.contract_offmesh(d, c))
    jac_t = lambda c, d: (lambda J: (0.5 * (J @ c), J))(OM.contract_tensor(d, c))

    for r in order_w:
        W = dict(r=r)
        rep['widths'][str(r)] = W
        tw = time.perf_counter()
        M = int(M_of(r))
        base = dict(GTb=TB.rows_prefix(tb['GTb'], r), L=tb['L'][:r, :r], A=tb['A'][:M, :r], lam=tb['lam'][:M])
        base = jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), base)
        trust = float(cfg['trust_fraction'] * spread[str(r)])
        W.update(M=M, trust=trust, shared_bytes=dict(A=TC.nbytes(base['A']), lam=TC.nbytes(base['lam']),
                                                     L=TC.nbytes(base['L']), bank_prefix=TC.tree_bytes(base['GTb'])))
        fq = OM.make_fsc_rule(n, r, M, OM.contract_offmesh, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust)
        ft = OM.make_fsc_rule(n, r, M, OM.contract_tensor, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust)

        def qdata(nm):
            o = OMT[nm]
            d = dict(base, B=o['B'][:, :r], D=o['D'][:, :r], P=o['P'][:, :M])
            return jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), d)

        def run(f, data, j):
            v = f(U0[j], NU[j], data, {})
            block(v)
            return v

        # ------------------------------------------------ qualification (dev cases) ----
        qual = {}
        for fam, f in fams.items():
            conv = f['conv']
            dconv = qdata(conv)
            FC = [np.asarray(run(fq, dconv, j)[0]) for j in range(len(cases))]
            del dconv
            rows = []
            for nm in f['ladder']:
                if nm == conv:
                    rows.append(dict(rule=nm, m=rrec[nm]['m'], dist=0.0, finite=True))
                    continue
                d = qdata(nm)
                ds, fin, its = [], True, []
                for j in range(len(cases)):
                    v = run(fq, d, j)
                    F = np.asarray(v[0])
                    ok = bool(np.isfinite(F).all())
                    fin &= ok
                    ds.append(TC.rel_field_dist(F, FC[j]) if ok else np.inf)
                    its.append(int(np.asarray(v[2]).sum()))
                rows.append(dict(rule=nm, m=rrec[nm]['m'], dist=float(max(ds)), dist_cases=ds, finite=fin, lm_its=its))
                del d
            if r == frozen_w:
                TAU[fam] = next(x['dist'] for x in rows if x['rule'] == f['frozen'])
            sel, floor = TC.select_rung([x['rule'] for x in rows], [x['m'] for x in rows], [x['dist'] for x in rows], TAU[fam])
            if r == frozen_w:
                sel_used, how = f['frozen'], 'frozen'
            else:
                sel_used, how = sel, ('qualified-ladder-floor' if floor else 'qualified')
            qual[fam] = dict(rows=rows, tau=TAU[fam], suffix_selection=sel, ladder_floor=floor, selected=sel_used, how=how)
            log(f'r={r} {fam}: tau={TAU[fam]:.3e} selected {sel_used} ({how}); suffix rule gives {sel}')
        W['qualification'] = qual
        save()

        # ------------------------------------------------ paths for this width ----
        P_fam, S_fam = cfg['primary'], cfg['secondary']
        quad_rules = {'P': qual[P_fam]['selected'], 'S': qual[S_fam]['selected'],
                      'FP': fams[P_fam]['frozen'], 'FS': fams[S_fam]['frozen']}
        W['rules_used'] = quad_rules
        QD = {k: dict(rule=nm, data=qdata(nm)) for k, nm in quad_rules.items() if nm is not None}
        for k, v in QD.items():
            v['op_bytes'] = TC.nbytes(v['data']['B'], v['data']['D'], v['data']['P'])
        TD, oom = {}, None
        for k in ('P', 'S'):
            if k not in QD:
                continue
            d = QD[k]['data']
            try:
                t1 = time.perf_counter()
                Ts = TC.build_tensor(d['P'], d['B'], d['D'])
                TD[k] = dict(rule=QD[k]['rule'], data=dict(base, Ts=Ts), build_seconds=time.perf_counter() - t1,
                             op_bytes=TC.nbytes(Ts))
            except Exception as ex:  # noqa: BLE001
                if not TC.is_oom(ex):
                    raise
                oom = dict(path=f'tensor_{k}', attempted_bytes=8 * M * r * r, error=repr(ex)[:400])
                log(f'r={r} tensor {k} OOM: {oom}')
                break
        W['oom'] = oom
        W['op_bytes'] = {f'quad_{k}': v['op_bytes'] for k, v in QD.items()}
        W['op_bytes'].update({f'tensor_{k}': v['op_bytes'] for k, v in TD.items()})
        W['op_bytes_tensor_formula'] = 8 * M * r * r
        W['tensor_build_seconds'] = {k: v['build_seconds'] for k, v in TD.items()}
        W['m'] = {k: rrec[v['rule']]['m'] for k, v in QD.items()}

        rng = np.random.default_rng(20261115 + r)
        gt = {}
        for k in TD:
            dd = []
            for _ in range(4):
                c = jnp.asarray(rng.normal(size=r) / np.sqrt(r))
                vq, Jq = jac_q(c, QD[k]['data'])
                vt, Jt = jac_t(c, TD[k]['data'])
                dd += [TC.rel_max(Jt, Jq), TC.rel_max(vt, vq)]
            gt[k] = max(dd)
        W['G_tab'] = dict(max_rel=gt, passed=bool(all(v <= 1e-12 for v in gt.values())))

        OUT, geq = {}, {}
        for k, v in QD.items():
            for j in range(len(cases)):
                o_ = run(fq, v['data'], j)
                OUT[(f'quad_{k}', j)] = (np.asarray(o_[0]), np.asarray(o_[1]), np.asarray(o_[2]))
        for k, v in TD.items():
            fd, cd, itd = [], [], []
            for j in range(len(cases)):
                o_ = run(ft, v['data'], j)
                Ft, Ct, It = np.asarray(o_[0]), np.asarray(o_[1]), np.asarray(o_[2])
                OUT[(f'tensor_{k}', j)] = (Ft, Ct, It)
                Fq, Cq, Iq = OUT[(f'quad_{k}', j)]
                fd.append(TC.rel_field_dist(np.concatenate([Ft[:1], Ft]), np.concatenate([Fq[:1], Fq])))
                cd.append(TC.rel_max(Ct, Cq))
                itd.append(int(np.abs(It - Iq).sum()))
            geq[k] = dict(field_max_rel=max(fd), coef_max_rel=max(cd), lm_iteration_diff_total=int(sum(itd)),
                          per_case_field=fd, passed=bool(max(fd) <= float(cfg['eq_tol'])))
        W['G_eq'] = geq
        log(f'r={r} G_tab {W["G_tab"]} G_eq ' + str({k: (v["field_max_rel"], v["passed"]) for k, v in geq.items()}))
        save()

        Cq0 = OUT[('quad_P', 0)][1]
        c0 = jnp.asarray(Cq0[len(Cq0) // 2])
        kq, kt = TC.make_kernel(jac_q, K), TC.make_kernel(jac_t, K)
        qk = lambda d: {x: d[x] for x in ('B', 'D', 'P')}
        mem = {}
        for k, v in QD.items():
            mem[f'query_quad_{k}'] = TC.mem_analysis(fq, U0[0], NU[0], v['data'], {})
            mem[f'kernel_quad_{k}'] = TC.mem_analysis(kq, qk(v['data']), c0)
        for k, v in TD.items():
            mem[f'query_tensor_{k}'] = TC.mem_analysis(ft, U0[0], NU[0], v['data'], {})
            mem[f'kernel_tensor_{k}'] = TC.mem_analysis(kt, {'Ts': v['data']['Ts']}, c0)
        W['memory_analysis'] = mem

        skip = lambda k: k in ('FP', 'FS') and quad_rules[k] == quad_rules[k[1]]
        subj = {f'quad_{k}': [(lambda j=j, d=v['data']: fq(U0[j], NU[j], d, {})) for j in range(len(cases))]
                for k, v in QD.items() if not skip(k)}
        if 'P' in TD:
            subj['tensor_P'] = [(lambda j=j, d=TD['P']['data']: ft(U0[j], NU[j], d, {})) for j in range(len(cases))]

        def check(nm, j, o_):
            return max(TC.rel_max(np.asarray(o_[0]), OUT[(nm, j)][0]), TC.rel_max(np.asarray(o_[1]), OUT[(nm, j)][1]))
        inv, summ = TC.panel(subj, tcfg['reps'], burn, tcfg['burn'], tcfg['initial_burn'], seed=tcfg['seed'] + r,
                             check=check, label='query')
        W['timing_query'] = dict(summary=summ, invocations=inv, max_output_diff=max(d['diff'] for d in inv))
        log(f'r={r} query ms ' + str({k: round(v['median_ms'], 2) for k, v in summ.items()}))
        save()

        ksubj = {f'quad_{k}': [lambda dd=qk(v['data']): kq(dd, c0)] for k, v in QD.items() if not skip(k)}
        ksubj.update({f'tensor_{k}': [lambda dd={'Ts': v['data']['Ts']}: kt(dd, c0)] for k, v in TD.items()})
        kinv, ksumm = TC.panel(ksubj, tcfg['kernel_reps'], burn, tcfg['burn'], tcfg['initial_burn'] / 3,
                               seed=tcfg['seed'] + 7 * r, label='kernel')
        for v in ksumm.values():
            v['per_eval_us'] = 1e3 * v['median_ms'] / K
            v['per_eval_q1_us'] = 1e3 * v['q1_ms'] / K
            v['per_eval_q3_us'] = 1e3 * v['q3_ms'] / K
        W['timing_kernel'] = dict(K=K, summary=ksumm, invocations=kinv)
        log(f'r={r} kernel us/eval ' + str({k: round(v['per_eval_us'], 2) for k, v in ksumm.items()}))
        W['device_after_width'] = TC.device_peak()
        W['seconds'] = time.perf_counter() - tw
        save()
        del QD, TD, OUT, base, fq, ft, kq, kt
        gc.collect()
        jax.clear_caches()

    rep['tau'] = TAU
    rep['gates']['G_tab_all'] = all(rep['widths'][str(r)]['G_tab']['passed'] for r in widths)
    rep['gates']['G_eq_all'] = all(v['passed'] for r in widths for v in rep['widths'][str(r)]['G_eq'].values())
    rep['gates']['timed_outputs'] = max(rep['widths'][str(r)]['timing_query']['max_output_diff'] for r in widths)
    rep['device_final'] = TC.device_peak()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('TC3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
