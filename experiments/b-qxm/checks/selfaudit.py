"""Self-audit: recompute every headline number of the b-qxm report from the audit JSONs by a
code path that shares nothing with `reports/generate_xm.py`, and re-derive the per-cell errors
from the archived fields for a sample of cells. Codex was unavailable (quota), so this stands
in for the cross-family audit and is deliberately written to disagree if the generator is wrong.

    python checks/selfaudit.py --audits checks/*-audit.json --analysis reports/analysis.json \
        --summary reports/summary.json --out checks/selfaudit.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audits', nargs='+', required=True)
    p.add_argument('--analysis', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--field-cells', default='q0_M64_dense,q256_M1088_dense,q128_M1088_dense')
    a = p.parse_args()
    au = {}
    for f in a.audits:
        j = json.loads(Path(f).read_text())
        au[j['attempt']] = j
    an = json.loads(Path(a.analysis).read_text())
    summ = json.loads(Path(a.summary).read_text())
    out, bad = {}, []

    def check(name, mine, theirs, tol=1e-12):
        ok = (mine is None and theirs is None) or (
            theirs is not None and mine is not None and abs(mine - theirs) <= tol * max(1., abs(theirs)))
        out[name] = dict(mine=mine, report=theirs, agree=bool(ok))
        if not ok:
            bad.append(name)

    # --- flat cell table, built independently from the audits -------------------
    cells = {}
    for att, j in au.items():
        for r in j['arms']:
            if r['family'] != 'rom' or 'solver-control' in (r.get('labels') or []):
                continue
            cells.setdefault((r['q'], r['M']), []).append((att, j['question'], r))
    E = {}
    for k, lst in cells.items():
        pref = [x for x in lst if x[1] == ('G1' if k[0] <= 32 else 'G2')] or lst
        E[k] = pref[0][2]['worst_evolved_percent']

    # --- spans -----------------------------------------------------------------
    for M in (256, 1088):
        qs = sorted(q for (q, MM) in E if MM == M)
        vals = [E[(q, M)] for q in qs]
        check(f'span_q_fixed_M{M}', vals[0] / vals[-1],
              an['spans']['worst_evolved_percent']['fixed_M'][str(M)]['span'])
        out[f'monotone_fixed_M{M}'] = dict(mine=all(b <= a_ for a_, b in zip(vals, vals[1:])),
                                           report=an['spans']['worst_evolved_percent']['fixed_M'][str(M)]['monotone'])
    sched = [(0, 64), (16, 128), (32, 192), (64, 320), (128, 576), (256, 1088)]
    sv = [E[c] for c in sched]
    check('span_scheduled_4x', sv[0] / sv[-1], an['spans']['worst_evolved_percent']['scheduled']['4x']['span'])

    # --- corner decomposition, recomputed from scratch --------------------------
    dM = math.log(E[(0, 64)]) - math.log(E[(0, 1088)])
    dq = math.log(E[(0, 1088)]) - math.log(E[(256, 1088)])
    tot = math.log(E[(0, 64)]) - math.log(E[(256, 1088)])
    c = an['decomposition']['worst_evolved_percent']['corner']
    check('corner_delta_M', dM, c['delta_M'])
    check('corner_delta_q', dq, c['delta_q'])
    check('corner_identity_holds', dM + dq, tot)
    check('corner_share_M', dM / tot, c['share_M'])
    check('corner_share_q', dq / tot, c['share_q'])

    # --- rung decomposition -----------------------------------------------------
    rm, rq = 0., 0.
    for (q0, M0), (q1, M1) in zip(sched, sched[1:]):
        rm += math.log(E[(q0, M0)]) - math.log(E[(q0, M1)])
        rq += math.log(E[(q0, M1)]) - math.log(E[(q1, M1)])
    r = an['decomposition']['worst_evolved_percent']['rung']
    check('rung_delta_M', rm, r['delta_M'])
    check('rung_share_M', rm / (rm + rq), r['share_M'])
    check('rung_total_equals_corner_total', rm + rq, tot)

    # --- two-way variance shares ------------------------------------------------
    qs, Ms = [0, 16, 32, 64, 128], [256, 1088]
    A = np.array([[math.log(E[(q, M)]) for M in Ms] for q in qs])
    mu = A.mean()
    al, be = A.mean(1) - mu, A.mean(0) - mu
    inter = A - mu - al[:, None] - be[None, :]
    sst = float(((A - mu) ** 2).sum())
    anv = an['decomposition']['worst_evolved_percent']['anova']
    check('anova_frac_q', len(Ms) * float((al ** 2).sum()) / sst, anv['frac_q'])
    check('anova_frac_M', len(qs) * float((be ** 2).sum()) / sst, anv['frac_M'])
    check('anova_frac_interaction', float((inter ** 2).sum()) / sst, anv['frac_interaction'])

    # --- within-job tunability ladder ------------------------------------------
    g2 = au['bqx201']
    rows = {r['q']: r for r in g2['arms'] if r['family'] == 'rom' and r['M'] == 1088}
    qs2 = sorted(rows)
    w = an['verdict']['fixed1088_within_job']
    check('within_job_error_span', rows[qs2[0]]['worst_evolved_percent'] / rows[qs2[-1]]['worst_evolved_percent'], w['error_span'])
    check('within_job_cost_span', rows[qs2[-1]]['median_gpu_ms'] / rows[qs2[0]]['median_gpu_ms'], w['cost_span'])
    out['within_job_same_job'] = dict(mine=True, report=(w['job_id'] == g2['job_id']),
                                      agree=bool(w['job_id'] == g2['job_id']))
    if w['job_id'] != g2['job_id']:
        bad.append('within_job_same_job')

    # --- saturation -------------------------------------------------------------
    s1 = au['bqx301']
    for q in (0, 64):
        byM = {r['M']: r for r in s1['arms'] if r['family'] == 'rom' and r['q'] == q}
        Ms_ = sorted(byM)
        star = None
        for i, M in enumerate(Ms_[:-1]):
            nxt = byM[Ms_[i + 1]]
            if (byM[M]['worst_evolved_percent'] - nxt['worst_evolved_percent']) / byM[M]['worst_evolved_percent'] < .05:
                star = M
                break
        rep = an['saturation']['per_q'][str(q)]
        out[f'saturation_M_star_q{q}'] = dict(mine=star, report=rep['M_star'], agree=bool(star == rep['M_star']))
        if star != rep['M_star']:
            bad.append(f'saturation_M_star_q{q}')
        ref = byM.get(4 * (16 + q))
        check(f'saturation_cost_ratio_q{q}', byM[star]['median_gpu_ms'] / ref['median_gpu_ms'], rep['cost_ratio_at_M_star'], 1e-9)

    # --- every summary.json cell row must equal its audit row -------------------
    mism = []
    idx = {(att, r['arm']): r for att, j in au.items() for r in j['arms']}
    for row in summ['rows']:
        if row.get('attempt') and row.get('arm') and row.get('q') is not None and row['metric'] in (
                'worst_evolved_percent', 'worst_all_times_percent', 'worst_t0_compression_percent',
                'median_gpu_ms', 'median_iterations', 'converged'):
            src = idx.get((row['attempt'], row['arm']))
            if src is None or src[row['metric']] != row['value']:
                mism.append(row['arm'] + '/' + row['metric'])
    out['summary_rows_match_audits'] = dict(mine=len(mism), report=0, agree=not mism, mismatches=mism[:5])
    if mism:
        bad.append('summary_rows_match_audits')

    # --- re-derive a sample of cells from the ARCHIVED FIELDS -------------------
    field = {}
    for att, want in (('bqx101', 'q0_M64_dense'), ('bqx201', 'q256_M1088_dense'), ('bqx201', 'q128_M1088_dense')):
        art = HERE / 'artifacts' / att
        meta = json.loads((art / 'archive.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            tar = Path(td) / 'c.tar.gz'
            with tar.open('wb') as o:
                for ch in meta['chunks']:
                    o.write((art / ch['path']).read_bytes())
            with tarfile.open(tar) as tf:
                names = [n for n in tf.getnames() if n.startswith('output/') and n.endswith('.npz')]
                tf.extractall(td, members=[tf.getmember(n) for n in names])
            od = Path(td) / 'output'
            res = json.loads((HERE / 'artifacts' / att / 'result.json').read_text())
            refs = {x['case']: np.load(od / x['artifact'])['fields'] for x in res['reference']}
            base = {x['case']: x['artifact'] for x in res['invocations'] if x['name'] == 'fft_tight'}
            worst = 0.
            for x in res['invocations']:
                if x['name'] != want:
                    continue
                f = np.load(od / x['artifact'])['fields']
                b = np.load(od / base[x['case']])['fields']
                n0 = np.linalg.norm(refs[x['case']][0])
                sg = np.linalg.norm((f - b).reshape(len(f), -1), axis=1) / n0
                worst = max(worst, float(sg[1:].max()))
            field[f'{att}/{want}'] = worst * 100
    for k, v in field.items():
        att, arm = k.split('/')
        check(f'field_rederived_{arm}', v, idx[(att, arm)]['worst_evolved_percent'], 1e-12)

    res = dict(checks=out, disagreements=sorted(set(bad)),
               n_checks=len(out), all_agree=not bad,
               note=('independent recomputation of the report headline numbers and a re-derivation of three '
                     'cells from the archived fields; shares no code with reports/generate_xm.py'))
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(dict(n_checks=len(out), all_agree=not bad, disagreements=sorted(set(bad))), indent=2))


if __name__ == '__main__':
    main()
