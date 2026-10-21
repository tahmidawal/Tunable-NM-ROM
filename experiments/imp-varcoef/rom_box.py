"""Inherited ROM box cost per worst-error target (imp-varcoef, DESIGN §5).

Reads the qualified 2D front (jcp-fom2d inputs/FRONT-2D.json, 4096^2 members + other_qualified, both numerical-qualification labels,
errors = conditional E0 estimates on dev6 + val32) and the jcp-fom2d C2 paired-timing result at 4096^2 (H200, job
5456288, copied to inputs/c24096-result.json). For each primary (LSPG-BE / LSPG-CN-R) member: worst / median error,
stage-2 median ratio to the C2 baseline A, median absolute seconds. For each target tau in TARGETS: the cheapest
member (by median ratio) with worst <= tau. Writes checks/rom-box.json.

    python rom_box.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRONT = HERE.parent / 'jcp-fom2d/inputs/FRONT-2D.json'
C2 = HERE / 'inputs/c24096-result.json'
TARGETS = (0.01, 0.03, 0.10)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    fr = json.loads(FRONT.read_text())
    c2 = json.loads(C2.read_text())
    st2 = [t for t in c2['stage2'] if t.get('valid')]
    rows = []
    assert c2.get('complete') is True, 'C2 4096 job not complete'
    veto = set(c2.get('veto') or [])
    M4 = fr['meshes']['4096']
    for src, m in [('members', x) for x in M4['members']] + [('other_qualified', x) for x in M4['other_qualified']]:
        mf, lab = m['manifest'], m['labels']
        if mf['form'] != 'LSPG':
            continue
        if not (lab.get('numerically_qualified_accuracy_job') and lab.get('numerically_qualified_timing_job')):
            continue
        sch = 'CNR' if mf['scheme'] in ('CNR', 'CN-R') else mf['scheme']
        label = f"ckpt:18f0266ae6f0|R{mf['Rp']}|{mf['rule']}|{mf.get('policy', 'frozen')}|{sch}|{mf['dt']:g}|LSPG|{mf['level']}"
        tr = [t for t in st2 if t['B'] == label]
        if not tr:
            raise SystemExit(f'no C2 stage-2 triplets for {label}')
        cases = sorted({t['case'] for t in tr})
        assert label not in veto, ('vetoed in C2', label)
        assert len(cases) == 6 and all(sum(t['case'] == c for t in tr) == 3 for c in cases), ('C2 coverage', label)
        # C2 cost = median over cases of the per-case median ratio (jcp-fom2d DESIGN §5)
        ratio = float(np.median([np.median([t['ratio'] for t in tr if t['case'] == c]) for c in cases]))
        secs = float(np.median([np.median([t['tB'] for t in tr if t['case'] == c]) for c in cases]))
        rows.append(dict(label=label, front_config=m['config'], front_list=src, Rp=mf['Rp'], rule=mf['rule'], scheme=sch, dt=mf['dt'],
                         worst=m['errors']['worst'], median=m['errors']['median'], n=m['errors']['n'],
                         ratio_to_A=ratio, seconds=secs, n_triplets=len(tr), n_cases=len(cases)))
    tA = float(np.median([.5 * (t['tA1'] + t['tA2']) for t in st2]))
    by = {}
    for tau in TARGETS:
        ok = [r for r in rows if r['worst'] <= tau]
        by[f'{tau:g}'] = min(ok, key=lambda r: r['ratio_to_A']) if ok else None
    out = dict(source=dict(front=str(FRONT.relative_to(HERE.parent.parent.parent)), front_sha256=sha(FRONT),
                           c2=str(C2.relative_to(HERE)), c2_sha256=sha(C2), c2_job=c2['job_id'], c2_gpu=c2['gpu'],
                           c2_commit=c2['commit'], c2_backend=c2['backend']),
               A=c2['config']['A'] if 'A' in c2['config'] else None, A_seconds_median=tA,
               labels='E0 A10 (post hoc) and combined A10/A11 and the retracted A15 carry forward unchanged',
               note='errors: conditional E0 estimates (dev6+val32, constant-viscosity family); costs: C2 stage-2 '
                    '4096^2 full-field fused query, H200; ratio = t_B / mean(t_A1, t_A2)',
               members=rows, cheapest_by_target=by)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks/rom-box.json').write_text(json.dumps(out, indent=1))
    for k, r in by.items():
        print(k, None if r is None else (r['label'], round(r['worst'], 5), round(r['ratio_to_A'], 4), round(r['seconds'], 5)))


if __name__ == '__main__':
    main()
