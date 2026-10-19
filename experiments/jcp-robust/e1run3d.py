"""jcp-robust E1 3D driver (DESIGN 2; A1.2-A1.4, A2.1, A3.2-A3.3, A4.2). One job, one GPU, cohort 923801, n = 65.

Uses speed3d's context (s3ctx.Ctx) after jcp-dims's d3common.install (lane rule names incl. lat<n>s<k>, badlat<n>,
sob<m>s<k>); both copied unchanged into this lane. Per configuration {setting, spec, base n, successor n}:
  target   gl64 production rollouts on every case; target check gl56 on check cases (A3.3)
  draws    for s in draws: rule lat<n>s<s>: production (in-job stationarity certificate), ADT tight on every case and
           tighter on tighter cases (Q1), successor lat<succ>s<s> production on every case (Q2), rho of the own six output
           states against the successor and against the target table, lattice distances to successor and target
  extra    diagnostic rules (Sobol scrambles; level 'prod': production + target quantities only)
  bad      badlat<n> production + target quantities (evidence only) and the in-job fixture rho_tgt (C-E1.1, A1.3)
  permute  draw 0 replayed on the reversed case order (G-E1.3)
All coefficient arrays are saved (W_e1.npz, keys <tag>|<spec>, 'cases|...'); offline analysis recomputes every distance.

    python e1run3d.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import d3common as D3
import c3common as C3
import s3common as S
from s3ctx import Ctx
from d3run import P1rho

HERE = Path(__file__).resolve().parent
GAMMA = 1e-3


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rho_rows(cx, st, rule_a, rule_b, W):
    """rho of the own output states (cases, 6) of table rule_a against rule_b."""
    da, db = st.data(rule_a), st.data(rule_b)
    out = np.full((len(W), 6), np.nan)
    for ci in range(len(W)):
        for kk in range(6):
            c_ = jnp.asarray(W[ci, kk])
            out[ci, kk] = P1rho(C3.nl3(c_, da), C3.nl3(c_, db))
    return out


def adt(cx, sname, rule, sp, cases, tcs, tag):
    at, att = ('ad', sp[1], sp[2], 'tight'), ('ad', sp[1], sp[2], 'tighter')
    ra = cx.rollouts(sname, rule, [at], cases, tag=tag)[S.spec_key(at)]
    rb = cx.rollouts(sname, rule, [att], tcs, tag=tag)[S.spec_key(att)] if tcs else None
    idx = [cases.index(j) for j in tcs]
    return dict(W=ra['W'], valid=[bool(x) for x in ra['valid']],
                tt=(cx.dist_cases(sname, ra['W'][idx], rb['W'], tcs)[:, 1:].max(1).tolist() if rb else []),
                valid_tighter=[bool(x) for x in rb['valid']] if rb else [])


def evaluate(cx, cf, rule, succ, Wt, level, cases, tcs):
    sname, sp = cf['setting'], S.parse_key(cf['spec'])
    st = cx.ST[sname]
    key = S.spec_key(sp)
    tag = f"{cf['name']}|{rule}"
    r = cx.rollouts(sname, rule, [sp], cases, tag=tag)[key]
    W = r['W']
    out = dict(rule=rule, succ=succ, level=level, valid=[bool(x) for x in r['valid']], nonstationary=int(sum(r['nonst'])),
               steps=int(sum(r['steps'])), nonst_per_case=[int(x) for x in r['nonst']], rule_sha256=st._d[rule][1]['sha256'],
               m=st._d[rule][1]['m'])
    if Wt is not None:
        out['dist_target'] = cx.dist_cases(sname, W, Wt, cases)[:, 1:].max(1).tolist()
        out['rho_tgt'] = rho_rows(cx, st, rule, cf['target'], W).tolist()
    if level == 'full':
        a = adt(cx, sname, rule, sp, cases, tcs, tag)
        out.update(dist_adt=cx.dist_cases(sname, W, a['W'], cases)[:, 1:].max(1).tolist(), adt_valid=a['valid'],
                   adt_tight_tighter=a['tt'], adt_tighter_valid=a['valid_tighter'])
        if succ:
            rs = cx.rollouts(sname, succ, [sp], cases, tag=f"{cf['name']}|{rule}|succ|{succ}")[key]
            out.update(succ_valid=[bool(x) for x in rs['valid']],
                       dist_succ=cx.dist_cases(sname, W, rs['W'], cases)[:, 1:].max(1).tolist(),
                       rho_succ=rho_rows(cx, st, rule, succ, W).tolist(), succ_sha256=st._d[succ][1]['sha256'])
            st.drop(succ)
    if rule != cf['target']:
        st.drop(rule)
    gc.collect()
    return out, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    assert int(cfg['cohort_seed']) == 923801, 'E1 3D runs on the validation cohort only'
    D3.install(cfg)
    cx = Ctx(cfg, a.out)
    cx.rep['peak_after_setup'] = C3.peak_bytes()
    cx.rep['settings_M'] = {k: dict(M=v.M, Rp=v.Rp, mode_hash=v.mode_hash) for k, v in cx.ST.items()}
    cx.rep['configs'] = {}
    cases = list(cx.cases)
    for cf in cfg['configs']:
        t0 = time.perf_counter()
        sname, sp = cf['setting'], S.parse_key(cf['spec'])
        st = cx.ST[sname]
        tcs = [j for j in cf.get('tighter_cases', list(range(8))) if j in cases]
        chk = [j for j in cf.get('check_cases', list(range(8))) if j in cases]
        res = cx.rep['configs'][cf['name']] = dict(setting=sname, spec=cf['spec'], target=cf['target'], check=cf['check'],
                                                   M=st.M, Rp=st.Rp, draws={}, extra={}, controls={})
        # ---------------- target and target check (A3.3)
        key = S.spec_key(sp)
        rt = cx.rollouts(sname, cf['target'], [sp], cases, tag=f"{cf['name']}|TARGET|{cf['target']}")[key]
        Wt = rt['W']
        rc = cx.rollouts(sname, cf['check'], [sp], chk, tag=f"{cf['name']}|CHECK|{cf['check']}")[key]
        idx = [cases.index(j) for j in chk]
        res['target_run'] = dict(valid=[bool(x) for x in rt['valid']], check_valid=[bool(x) for x in rc['valid']],
                                 check_cases=chk, dist_check=cx.dist_cases(sname, Wt[idx], rc['W'], chk)[:, 1:].max(1).tolist(),
                                 target_sha256=st._d[cf['target']][1]['sha256'])
        st.drop(cf['check'])
        cx.save()
        # ---------------- draws
        for s in cf['draws']:
            rule, succ = f"lat{cf['n']}s{s}", f"lat{cf['succ_n']}s{s}"
            res['draws'][str(s)], W = evaluate(cx, cf, rule, succ, Wt, 'full', cases, tcs)
            if s == cf.get('permute_draw', 0) and cf.get('permute'):      # G-E1.3: reversed case order replay
                perm = list(reversed(cases))
                rp = cx.rollouts(sname, rule, [sp], perm, tag=f"{cf['name']}|{rule}|PERMUTED")[key]
                dev = []
                for i, j in enumerate(perm):
                    w0 = W[cases.index(j)]
                    dev.append(float(np.max(np.abs(rp['W'][i] - w0)) / max(float(np.max(np.abs(w0))), 1e-300)))
                st.drop(rule)
                res['controls']['permute'] = dict(rule=rule, max_rel_dev=max(dev), passed=bool(max(dev) <= 1e-12))
            print('DRAW', cf['name'], s, cx.el(), flush=True)
            cx.save()
            cx.dump()
        # ---------------- identity gate / control (G-E1.1, C-E1.2) on the in-job rule tables
        n = cf['n']
        X0, w0 = cx.rules[f'lat{n}']
        Xs0, ws0 = cx.rules[f'lat{n}s0']
        X1, _ = cx.rules[f'lat{n}s1']
        sh = (X1 - X0) % 1.0
        res['controls']['identity'] = dict(s0_equals_deployed=bool(np.array_equal(X0, Xs0) and np.array_equal(w0, ws0)),
                                           s1_pure_shift=bool(np.ptp((sh - sh[0] + .5) % 1.0, axis=0).max() < 1e-12),
                                           s1_differs=bool(not np.array_equal(X1, X0)))
        # ---------------- diagnostics (Sobol)
        for rl in cf.get('extra', []):
            res['extra'][rl], _ = evaluate(cx, cf, rl, None, Wt, 'prod', cases, tcs)
            cx.save()
        # ---------------- bad generating vector (C-E1.1)
        if cf.get('bad'):
            bad = cf['bad']
            fx = HERE / cf['fixture']['path']
            assert sha_file(fx) == cf['fixture']['sha256'], 'fixture differs from its pin'
            C = np.asarray(np.load(fx)['W'])[:, :st.Rp]
            Nb = [np.asarray(C3.nl3(jnp.asarray(c), st.data(bad))) for c in C]
            Nt = [np.asarray(C3.nl3(jnp.asarray(c), st.data(cf['target']))) for c in C]
            res['controls']['fixture'] = dict(rule=bad, rho_tgt=[P1rho(x, y) for x, y in zip(Nb, Nt)],
                                              target_norm=[float(np.linalg.norm(y)) for y in Nt],
                                              finite=bool(all(np.isfinite(x).all() for x in Nb + Nt)),
                                              fixture=cf['fixture']['path'], fixture_sha256=cf['fixture']['sha256'])
            res['controls']['bad_rollout'], _ = evaluate(cx, cf, bad, None, Wt, 'prod', cases, tcs)
        st.drop(cf['target'])
        gc.collect()
        res['seconds'] = time.perf_counter() - t0
        res['peak_bytes'] = C3.peak_bytes()
        cx.save()
        cx.dump()
    cx.rep['peak_bytes_job'] = C3.peak_bytes()
    cx.rep['elapsed_seconds'] = cx.el()
    cx.rep['complete'] = True
    cx.save()
    cx.dump()
    (cx.out / 'COMPLETE').write_text('complete\n')
    print('E1RUN3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
