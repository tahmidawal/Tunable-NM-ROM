"""tensor-crossover, 2D driver (PROTOCOL.md). One allocation, all widths; per width: qualify rules on dev6, build the
tensor of the selected rules, check trajectory agreement, time full queries and the nonlinear term, record bytes/memory.

The query is the vendored quadrature-study `qcore.make_linear_query('point', ...)` (the `acc` solver settings) unchanged;
the tensor path is the same function with `kind='tensor'`, added here by wrapping qcore.tested_value / tested_jac (the
vendored file is not edited). python tc2d.py --config configs/<job>.json --out output
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
V = HERE / 'vendor' / 'tree' / 'experiments'
for _p in (V / 'mr-burgers2d', V / 'separable-decoder', V / 'quadrature-study' / 'vendor', V / 'quadrature-study'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

import qcore as Q  # noqa: E402
import tcommon as TC  # noqa: E402

block = jax.block_until_ready
DT = 0.005
DEV6_SHA = None   # recorded, not asserted (dev cohort)

# ---- tensor kind for qcore (wrapping, not editing, the vendored dispatchers) ----
_tv, _tj = Q.tested_value, Q.tested_jac


def _tensor_tv(kind, L):
    if kind == 'tensor':
        return lambda c, d: 0.5 * (jnp.einsum('mij,j->mi', d['Ts'], c) @ c)
    return _tv(kind, L)


def _tensor_tj(kind, L):
    if kind == 'tensor':
        def f(c, d, Tm):
            assert Tm is None
            J = jnp.einsum('mij,j->mi', d['Ts'], c)
            return 0.5 * (J @ c), J
        return f
    return _tj(kind, L)


Q.tested_value, Q.tested_jac = _tensor_tv, _tensor_tj


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
    L = int(cfg['mesh'])
    widths = [int(r) for r in cfg['widths']]
    rep = dict(config=cfg, dim=2, mesh=L, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               jax_version=jax.__version__, x64=True, precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               checkpoint_sha256=sha_file(cfg_path(cfg['checkpoint'])), rotation_sha256=sha_file(cfg_path(cfg['rotation'])),
               gates={}, widths={}, complete=False)
    save = lambda: json.dump(TC.clean(rep), open(out / 'result.json', 'w'), indent=1)
    save()
    burn = TC.make_burn()

    # ---------------------------------------------------------------- rules (G-R) ----
    fams = cfg['families']                          # {fam: {ladder: [...], conv: name, frozen: name}}
    man = json.loads(cfg_path(cfg['rules_manifest']).read_text())['rules']
    ra = np.load(cfg_path(cfg['rules_arrays']))
    RULES, rrec = {}, {}
    for fam, f in fams.items():
        for nm in f['ladder']:
            X, w = Q.offmesh_rule(nm)
            X0, w0 = ra[f'{nm}_X'], ra[f'{nm}_w']
            dX = float(np.abs(X - X0).max()) if X.shape == X0.shape else None
            dw = float(np.abs(w - w0).max() / np.abs(w0).max()) if w.shape == w0.shape else None
            RULES[nm] = (X, w)
            rrec[nm] = dict(m=int(len(w)), max_dX=dX, max_dw_rel=dw, manifest_sha=man.get(nm))
    bad = [k for k, v in rrec.items() if v['max_dX'] is None or v['max_dX'] > 1e-14 or v['max_dw_rel'] > 1e-12]
    rep['gates']['G_R'] = dict(checked=len(rrec), mismatched=bad, passed=not bad)
    rep['rules'] = rrec
    assert not bad, bad

    # ---------------------------------------------------------------- model, bank ----
    mdl = Q.Model(ckpt=cfg_path(cfg['checkpoint']), rotation=cfg_path(cfg['rotation']))
    edges = sorted(set([0] + widths + [512]))
    t0 = time.perf_counter()
    Grot = Q.build_rotated_bank(mdl, L, edges)
    rep['setup'] = dict(bank_seconds=time.perf_counter() - t0, bank_bytes=TC.tree_bytes(Grot), edges=edges)
    hv = jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))
    Hall = np.concatenate([np.asarray(hv(jnp.asarray(mdl.Zold[s:s + 8192]))) for s in range(0, len(mdl.Zold), 8192)])
    Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
    Hsub = np.asarray(hv(jnp.asarray(Zsub)))
    ph_all = Q.cohort(cfg['cohort'])
    rep['cohort'] = dict(name=cfg['cohort'], sha256=hashlib.sha256(np.ascontiguousarray(ph_all).tobytes()).hexdigest(),
                         count=int(len(ph_all)))
    assert cfg['cohort'] in ('dev6', 'val32'), 'development/validation cohorts only'
    cases = list(cfg.get('cases', range(len(ph_all))))
    U0 = [block(jnp.asarray(Q.e.initial(L, ph_all[c]))) for c in cases]
    NU = [float(ph_all[c][4]) for c in cases]
    log(f'bank built ({rep["setup"]["bank_seconds"]:.1f}s), cases {cases}')
    save()
    BK = Q.BK
    K = int(cfg['kernel_evals'])
    tcfg = cfg['timing']
    frozen_w = int(cfg['frozen_width'])

    # tolerance tau per family is calibrated at the frozen width first (PROTOCOL §4)
    order_w = [frozen_w] + [r for r in widths if r != frozen_w]
    TAU = {}
    for r in order_w:
        W = dict(r=r)
        rep['widths'][str(r)] = W
        tw = time.perf_counter()
        M = 4 * r
        o = Q.operators(Grot, edges, L, M)
        base = dict(A=o['Arot'][:, :r], lam=o['lam'], G=BK.prefix(Grot, edges, r))
        codes = Hall @ mdl.Lrot[:r].T
        trust = .01 * float(np.max(np.linalg.norm(codes - codes.mean(0), axis=1)))
        cold = Q.build_cold_linear(mdl, r, Hsub @ mdl.Lrot[:r].T)
        W.update(M=M, trust=trust, shared_bytes=dict(A=TC.nbytes(base['A']), lam=TC.nbytes(base['lam']),
                                                     bank_prefix=TC.tree_bytes(base['G']), cold=TC.tree_bytes(cold)))
        fq, _ = Q.make_linear_query('point', r, L, DT, trust, step_budget=cfg['step_budget'], gtol=cfg['gtol'])
        ft, _ = Q.make_linear_query('tensor', r, L, DT, trust, step_budget=cfg['step_budget'], gtol=cfg['gtol'])

        def qdata(nm):
            X, w = RULES[nm]
            return dict(base, **Q.offmesh_data(mdl, r, X, w, L, o['kx'], o['ky'], 'point'))

        def run(f, data, j):
            v = f(U0[j], NU[j], data, cold)
            block(v['fields'])
            return v

        # ------------------------------------------------ qualification (dev cases) ----
        qual = {}
        for fam, f in fams.items():
            conv = f['conv']
            dconv = qdata(conv)
            FC = []
            for j in range(len(cases)):
                v = run(fq, dconv, j)
                FC.append(np.asarray(v['fields']))
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
                    F = np.asarray(v['fields'])
                    fin &= bool(np.isfinite(F).all())
                    ds.append(TC.rel_field_dist(F, FC[j]) if np.isfinite(F).all() else np.inf)
                    its.append(int(np.asarray(v['it']).sum()))
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
        QD = {}
        for k, nm in quad_rules.items():
            if nm is None:
                continue
            QD[k] = dict(rule=nm, data=qdata(nm))
        QD = {k: v for k, v in QD.items()}
        # operator bytes
        for k, v in QD.items():
            d = v['data']
            v['op_bytes'] = TC.nbytes(d['Gq'], d['Gs'], d['Psi'])
        TD, oom = {}, None
        for k in ('P', 'S'):
            if k not in QD:
                continue
            d = QD[k]['data']
            want = 8 * M * r * r
            try:
                t1 = time.perf_counter()
                Ts = TC.build_tensor(d['Psi'], d['Gq'], d['Gs'])
                TD[k] = dict(rule=QD[k]['rule'], data=dict(base, Ts=Ts), build_seconds=time.perf_counter() - t1,
                             op_bytes=TC.nbytes(Ts))
            except Exception as ex:  # noqa: BLE001
                if not TC.is_oom(ex):
                    raise
                oom = dict(path=f'tensor_{k}', attempted_bytes=want, error=repr(ex)[:400])
                log(f'r={r} tensor {k} OOM: {oom}')
                break
        W['oom'] = oom
        W['op_bytes'] = {f'quad_{k}': v['op_bytes'] for k, v in QD.items()}
        W['op_bytes'].update({f'tensor_{k}': v['op_bytes'] for k, v in TD.items()})
        W['op_bytes_tensor_formula'] = 8 * M * r * r
        W['tensor_build_seconds'] = {k: v['build_seconds'] for k, v in TD.items()}
        W['m'] = {k: rrec[v['rule']]['m'] for k, v in QD.items()}

        # G-tab: Jacobians agree on random vectors
        rng = np.random.default_rng(20261115 + r)
        nlq, nlt = Q.tested_jac('point', L), Q.tested_jac('tensor', L)
        gt = {}
        for k in TD:
            dd = []
            for _ in range(4):
                c = jnp.asarray(rng.normal(size=r) / np.sqrt(r))
                vq, Jq = nlq(c, QD[k]['data'], None)
                vt, Jt = nlt(c, TD[k]['data'], None)
                dd += [TC.rel_max(Jt, Jq), TC.rel_max(vt, vq)]
            gt[k] = max(dd)
        W['G_tab'] = dict(max_rel=gt, passed=bool(all(v <= 1e-12 for v in gt.values())))

        # G-eq: same trajectory on every dev case; outputs kept for the timed-output check
        OUT, geq = {}, {}
        for k, v in QD.items():
            for j in range(len(cases)):
                o_ = run(fq, v['data'], j)
                OUT[(f'quad_{k}', j)] = (np.asarray(o_['fields']), np.asarray(o_['internal']), np.asarray(o_['it']))
        for k, v in TD.items():
            fd, cd, itd = [], [], []
            for j in range(len(cases)):
                o_ = run(ft, v['data'], j)
                Ft, Ct, It = np.asarray(o_['fields']), np.asarray(o_['internal']), np.asarray(o_['it'])
                OUT[(f'tensor_{k}', j)] = (Ft, Ct, It)
                Fq, Cq, Iq = OUT[(f'quad_{k}', j)]
                fd.append(TC.rel_field_dist(np.concatenate([Ft[:1], Ft]), np.concatenate([Fq[:1], Fq])))  # all 6 times
                cd.append(TC.rel_max(Ct, Cq))
                itd.append(int(np.abs(It - Iq).sum()))
            geq[k] = dict(field_max_rel=max(fd), coef_max_rel=max(cd), lm_iteration_diff_total=int(sum(itd)),
                          per_case_field=fd, passed=bool(max(fd) <= float(cfg['eq_tol'])))
        W['G_eq'] = geq
        log(f'r={r} G_tab {W["G_tab"]} G_eq ' + str({k: (v["field_max_rel"], v["passed"]) for k, v in geq.items()}))
        save()

        # ------------------------------------------------ memory analysis ----
        mem = {}
        c0 = jnp.asarray(OUT[(f'quad_P', 0)][1][len(OUT[('quad_P', 0)][1]) // 2])
        kq = TC.make_kernel(lambda c, d: nlq(c, d, None), K)
        kt = TC.make_kernel(lambda c, d: nlt(c, d, None), K)
        for k, v in QD.items():
            mem[f'query_quad_{k}'] = TC.mem_analysis(fq, U0[0], NU[0], v['data'], cold)
            mem[f'kernel_quad_{k}'] = TC.mem_analysis(kq, {x: v['data'][x] for x in ('Gq', 'Gs', 'Psi')}, c0)
        for k, v in TD.items():
            mem[f'query_tensor_{k}'] = TC.mem_analysis(ft, U0[0], NU[0], v['data'], cold)
            mem[f'kernel_tensor_{k}'] = TC.mem_analysis(kt, {'Ts': v['data']['Ts']}, c0)
        W['memory_analysis'] = mem

        # ------------------------------------------------ timing: full query ----
        subj = {}
        for k, v in QD.items():
            if k in ('FP', 'FS') and quad_rules[k] == quad_rules[k[1]]:
                continue                                    # frozen rule == selected rule: not timed twice
            subj[f'quad_{k}'] = [(lambda j=j, d=v['data']: fq(U0[j], NU[j], d, cold)) for j in range(len(cases))]
        if 'P' in TD:
            subj['tensor_P'] = [(lambda j=j, d=TD['P']['data']: ft(U0[j], NU[j], d, cold)) for j in range(len(cases))]

        def check(nm, j, o_):
            return max(TC.rel_max(np.asarray(o_['fields']), OUT[(nm, j)][0]),
                       TC.rel_max(np.asarray(o_['internal']), OUT[(nm, j)][1]))
        inv, summ = TC.panel(subj, tcfg['reps'], burn, tcfg['burn'], tcfg['initial_burn'], seed=tcfg['seed'] + r,
                             check=check, label='query')
        W['timing_query'] = dict(summary=summ, invocations=inv, max_output_diff=max(d['diff'] for d in inv))
        log(f'r={r} query ms ' + str({k: round(v['median_ms'], 2) for k, v in summ.items()}))
        save()

        # ------------------------------------------------ timing: nonlinear term alone ----
        ksubj = {}
        for k, v in QD.items():
            if k in ('FP', 'FS') and quad_rules[k] == quad_rules[k[1]]:
                continue
            dd = {x: v['data'][x] for x in ('Gq', 'Gs', 'Psi')}
            ksubj[f'quad_{k}'] = [lambda dd=dd: kq(dd, c0)]
        for k, v in TD.items():
            dd = {'Ts': v['data']['Ts']}
            ksubj[f'tensor_{k}'] = [lambda dd=dd: kt(dd, c0)]
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
        del QD, TD, OUT, base, o, cold, fq, ft, kq, kt
        gc.collect()
        jax.clear_caches()

    rep['gates']['G_tab_all'] = all(rep['widths'][str(r)]['G_tab']['passed'] for r in widths)
    rep['gates']['G_eq_all'] = all(v['passed'] for r in widths for v in rep['widths'][str(r)]['G_eq'].values())
    rep['gates']['timed_outputs'] = max(rep['widths'][str(r)]['timing_query']['max_output_diff'] for r in widths)
    rep['device_final'] = TC.device_peak()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('TC2D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
