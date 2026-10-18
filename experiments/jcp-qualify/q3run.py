"""jcp-qualify Q2-3D job (DESIGN 2.4, A1.1): quadrature qualification of speed3d front members on their own reached states.

Uses speed3d's job context (s3ctx.Ctx, copied unchanged) at n = 65 on cohort 923801. Per member (setting|rule|spec):
  q      rerun with the member's own rule on every case (own reached states; reproduction gate against speed3d's archive
         is evaluated offline), certified as speed3d does (fixed-sweep / QN runs)
  q+     replay with the successor rule (same setting, spec, everything else), every case
  q++    replay with the check rule on the qpp cases (finer-rule convergence)
  rho    ||N_q(c) - N_q+(c)|| / ||N_q+(c)|| of the tested advection (c3common.nl3) at the six output states of every case
Distances (A2.3 lattice metric) are computed in-job (Ctx.dist_cases) and recomputed offline. Optional 'pilot' block:
build the listed (setting, rule) tables and record the peak device memory. Optional 'control': the 3D under-resolved
rule fixture (DESIGN 7), run through the same code.

    python q3run.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import c3common as C3
import s3common as S
from s3ctx import Ctx


def rho_of(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('inf')
    nb = float(np.linalg.norm(b))
    if nb == 0.:
        return float('nan') if float(np.linalg.norm(a)) == 0. else float('inf')
    return float(np.linalg.norm(a - b) / nb)


def member(cx, m, cases, qpp_cases, label):
    sname, rule, *spec = m['key'].split('|')
    sp = S.parse_key('|'.join(spec))
    key = S.spec_key(sp)
    res = dict(key=m['key'], setting=sname, rule=rule, spec=key, qp=m['qp'], qpp=m.get('qpp'), label=label)
    ra = cx.rollouts(sname, rule, [sp], cases)[key]
    rq = cx.rollouts(sname, m['qp'], [sp], cases)[key]
    res['valid'] = [bool(x) for x in ra['valid']]
    res['nonstationary'], res['steps'] = int(sum(ra['nonst'])), int(sum(ra['steps']))
    res['valid_qp'] = [bool(x) for x in rq['valid']]
    res['dist_qp'] = cx.dist_cases(sname, ra['W'], rq['W'], cases)[:, 1:].max(1).tolist()
    st = cx.ST[sname]
    dq, dp = st.data(rule), st.data(m['qp'])
    rho = np.full((len(cases), 6), np.nan)
    for i in range(len(cases)):
        for k in range(6):
            c = jnp.asarray(ra['W'][i, k])
            rho[i, k] = rho_of(C3.nl3(c, dq), C3.nl3(c, dp))
    res['rho'] = rho.tolist()
    if m.get('qpp'):
        idx = [cases.index(j) for j in qpp_cases]
        rqq = cx.rollouts(sname, m['qpp'], [sp], qpp_cases)[key]
        res['valid_qpp'] = [bool(x) for x in rqq['valid']]
        res['dist_qp_qpp'] = cx.dist_cases(sname, rq['W'][idx], rqq['W'], qpp_cases)[:, 1:].max(1).tolist()
        res['qpp_cases'] = list(qpp_cases)
    res['cases'] = list(cases)
    print('MEMBER', label, m['key'], 'worst dist_qp', max(res['dist_qp']), 'rho_max', float(np.nanmax(rho)), cx.el(), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    assert int(cfg['cohort_seed']) == 923801, 'Q2-3D runs on the validation cohort only'
    cx = Ctx(cfg, a.out)
    cx.rep['peak_after_setup'] = C3.peak_bytes()
    cases = list(cx.cases)
    qpp_cases = [j for j in cfg.get('qpp_cases', list(range(8))) if j in cases]
    cx.rep['members'] = {}
    if cfg.get('pilot'):
        pil = []
        for p_ in cfg['pilot']:
            cx.ST[p_['setting']].data(p_['rule'])
            pil.append(dict(p_, peak_bytes=C3.peak_bytes()))
        cx.rep['pilot'] = pil
        cx.save()
    for m in cfg['members']:
        cx.rep['members'][m['key']] = member(cx, m, cases, qpp_cases, 'member')
        cx.save()
        cx.dump()
    if cfg.get('control'):
        m = cfg['control']
        cx.rep['control'] = member(cx, m, cases, [], 'control')
    cx.rep['peak_bytes_job'] = C3.peak_bytes()
    cx.rep['elapsed_seconds'] = cx.el()
    cx.rep['complete'] = True
    cx.save()
    cx.dump()
    (cx.out / 'COMPLETE').write_text('complete\n')
    print('Q3RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
