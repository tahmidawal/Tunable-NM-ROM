"""jcp-smooth-bank paired timing (DESIGN §5.8, A1.4): every bank variant at its own m*_G(0.116) and m*_G(0.06) Gauss
rules (read from the evaluation outputs), all on ONE GPU in ONE process. Every subject is compiled and warmed on every
case first; then for each repetition and each dev6 case the subjects run in forward then reverse order (A-B-..-B-A),
each call preceded by a 0.1 s pre-compiled burn and closed by block_until_ready. Every invocation is saved, with the
sha256 of its restricted output compared to the evaluation's phase-1 rollout of the same arm and case.

    python banktime.py --config jobs/<file>.json --evals <dir with eval_<label>.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import bankeval as B
Q, QS = B.Q, B.QS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--evals', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu' or cfg.get('allow_cpu_smoke'), jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    L = int(cfg['mesh'])
    s256 = L // 256
    dev6 = [(c, ph) for c, ph in enumerate(Q.cohort('dev6'))]
    variants = [v for t in cfg['tasks'] for v in t['variants']]
    rep = dict(gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), commit=os.environ.get('SOURCE_COMMIT'),
               job_id=os.environ.get('SLURM_JOB_ID'), mesh=L, reps=a.reps, burn=0.1, cases=[f'dev6{c}' for c, _ in dev6],
               subjects=[], invocations=[], complete=False)
    QS.burn(.05)
    for s in B.SETTINGS:
        st = Q.SETTINGS[s]
        Rp, M = st['Rp'], st['M']
        subjects = []
        for var in variants:
            ev = json.loads((Path(a.evals) / f"eval_{var['label']}.json").read_text())
            S = ev['settings'][s]
            rules = []
            for b in (0.116, 0.06):
                ms = S['mstar'].get(f'gref|G|{b}')
                if ms is not None:
                    nm = f'gauss{int(round(np.sqrt(ms)))}'
                    if nm not in rules:
                        rules.append(nm)
            if not rules:
                continue
            deployed = bool(var.get('deployed'))
            ck = B.HERE / var['ckpt']
            rt = B.HERE / var['rotation']
            mdl = Q.Model(ckpt=ck, rotation=rt) if deployed else B.LaneModel(ck, rt)
            mesh = QS.Mesh(mdl, L, [s])
            o = mesh.ops[M]
            trust, _ = mdl.trust_linear(Rp)
            cold = Q.build_cold_linear(mdl, Rp, B.cold_codes(mdl, Rp, deployed))
            phase1 = {(r['arm'], r['case']): r['restricted_sha256'] for r in S['rows'] if r['cohort'] == 'dev6'}
            assert ev.get('complete'), f"evaluation of {var['label']} incomplete"
            cz = np.load(Path(a.evals) / f"coeffs_{var['label']}_{s}.npz")
            phase1W = {(k.split('|')[0], int(k.split('|')[2])): cz[k] for k in cz.files if k.split('|')[1] == 'dev6'}
            for nm in rules:
                X, w = Q.offmesh_rule(nm)
                d = Q.offmesh_data(mdl, Rp, X, w, L, o['kx'], o['ky'], 'point')
                data = dict(mesh.base(s), **d)
                fq, _ = Q.make_linear_query('point', Rp, L, B.DT, trust, step_budget=600, gtol=1e-3)
                subjects.append(dict(label=var['label'], rule=nm, m=len(X),
                                     call=lambda u, nu, fq=fq, data=data, cold=cold: fq(u, nu, data, cold),
                                     phase1=phase1, phase1W=phase1W))
            rep['subjects'].append(dict(setting=s, label=var['label'], rules=rules))
        inputs = [(jnp.asarray(Q.e.initial(L, ph)), float(ph[4]), c) for c, ph in dev6]
        for sj in subjects:                       # compile + warm every subject on every case
            for u0, nu, c in inputs:
                jax.block_until_ready(sj['call'](u0, nu)['fields'])
        for r_ in range(a.reps):
            for u0, nu, c in inputs:
                order = list(range(len(subjects))) + list(range(len(subjects)))[::-1]
                for pos, j in enumerate(order):
                    sj = subjects[j]
                    QS.burn(.1)
                    t1 = time.perf_counter()
                    v = sj['call'](u0, nu)
                    jax.block_until_ready(v['fields'])
                    secs = time.perf_counter() - t1
                    h = QS.sha(np.asarray(v['fields'][:, ::s256, ::s256]))
                    W, W1 = np.asarray(v['internal']), sj['phase1W'].get((sj['rule'], c))
                    dW = float(np.linalg.norm(W - W1) / np.linalg.norm(W1)) if W1 is not None else None
                    reason = np.asarray(v['reason'])
                    rep['invocations'].append(dict(setting=s, label=sj['label'], rule=sj['rule'], m=sj['m'], case=c,
                                                   rep=r_, position=pos, seconds=secs,
                                                   iterations_total=int(np.asarray(v['it']).sum()),
                                                   exits={QS.REASONS[k]: int(np.sum(reason == k)) for k in QS.REASONS},
                                                   sha256=h, phase1_sha256=sj['phase1'].get((sj['rule'], c)),
                                                   output_matches_phase1=bool(h == sj['phase1'].get((sj['rule'], c))),
                                                   coeff_rel_diff_vs_phase1=dW))
                    del v
        (out / 'timing.json').write_text(json.dumps(QS.clean(rep), indent=1))
        del subjects
    med = {}
    for d_ in rep['invocations']:
        med.setdefault(f"{d_['setting']}|{d_['label']}|{d_['rule']}", []).append(d_['seconds'])
    rep['median_ms'] = {k: 1e3 * float(np.median(v)) for k, v in med.items()}
    diffs = [d_['coeff_rel_diff_vs_phase1'] for d_ in rep['invocations']]
    rep['max_coeff_rel_diff_vs_phase1'] = max((x for x in diffs if x is not None), default=None)
    # valid only if every timed rollout reproduces the evaluation's (a different process/compile may differ by round-off)
    rep['valid'] = bool(all(x is not None and x <= 1e-6 for x in diffs))
    rep['complete'] = True
    (out / 'timing.json').write_text(json.dumps(QS.clean(rep), indent=1))
    print('BANKTIME COMPLETE', json.dumps(rep['median_ms'], indent=1), flush=True)


if __name__ == '__main__':
    main()
