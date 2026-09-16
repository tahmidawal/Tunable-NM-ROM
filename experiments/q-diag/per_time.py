"""Stage 1: recompute per-case, per-output-time same-grid errors for every archived arm.

Reads the restored collection archives (fields + result.json) of btq101/btq102/btq201
(b-ladder-top), cclad01 (cheap-corrections) and qlad01 (head-ablation) and writes one
JSON with the full (arm, case, time) error cube plus the archived aggregates it must
reproduce. Pure NumPy; nothing is re-solved.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def cube(root: Path):
    r = json.loads((root / 'output' / 'result.json').read_text())
    out = root / 'output'
    inv = r['invocations']
    refs = {e['case']: np.load(out / e['artifact'])['fields'] for e in r['reference']}
    base = {x['case']: x['artifact'] for x in inv if x['name'] in ('fft_tight',)}
    cache = {}

    def fields(name):
        if name not in cache:
            cache[name] = np.load(out / name)['fields']
        return cache[name]

    rows = {}
    for x in inv:
        key = (x['name'], x['case'])
        if key in rows:
            continue
        f = fields(x['artifact'])
        n0 = np.linalg.norm(refs[x['case']][0])
        g = fields(base[x['case']])
        sg = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
        ref = np.linalg.norm((f - refs[x['case']]).reshape(len(f), -1), axis=1) / n0
        d = np.load(out / x['artifact'])
        rows[key] = dict(
            arm=x['name'], case=x['case'], kind=x['kind'], artifact=x['artifact'],
            same_grid_per_time=sg.tolist(), reference_per_time=ref.tolist(),
            archived_reference_per_time=x['error']['fixed_initial_per_time'],
            residuals=x.get('residuals'), iterations=x.get('iterations'),
            latent_norms=(np.linalg.norm(d['internal_latents'], axis=1).tolist()
                          if 'internal_latents' in d.files else None),
            correction_norms=None)
        if 'internal_latents' in d.files:
            W = d['internal_latents']
            K = int(r['K'])
            rows[key]['correction_norms'] = np.linalg.norm(W[:, K:], axis=1).tolist() if W.shape[1] > K else None
            rows[key]['z_norms'] = np.linalg.norm(W[:, :K], axis=1).tolist()
    setup = {s['arm']: s for s in r.get('arm_setup', []) if 'arm' in s}
    meta = dict(job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'], K=r['K'], R=r['R'],
                intervals=r['intervals'], dt=r['dt'] if 'dt' in r else r['config']['dt'],
                output_times=r['output_times'], cohort_roles=r['cohort_roles'],
                physical_cases=r['physical_cases'],
                directions_sha256=r.get('directions', {}).get('directions_sha256'),
                residual_energy_captured=r.get('directions', {}).get('residual_energy_captured'))
    arms = {k: dict(q=v.get('q'), M=v.get('M'), m=v.get('m'), quadrature=v.get('quadrature'),
                    rule=v.get('rule'), fix=v.get('fix'), gtol=v.get('gtol'),
                    variant=v.get('variant'), solved_dimension=v.get('solved_dimension'),
                    eq_relative_fit=v.get('eq_relative_fit'),
                    eq_rule_valid=v.get('eq_rule_valid'),
                    quadrature_truncated=(v.get('eq_fit') or {}).get('truncated'))
            for k, v in setup.items()}
    return meta, arms, list(rows.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restore', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    R = Path(a.restore)
    doc = {}
    for job in ('btq101', 'btq102', 'btq201', 'cclad01', 'qlad01'):
        meta, arms, rows = cube(R / job)
        doc[job] = dict(meta=meta, arms=arms, rows=rows)
        print(job, 'arms', len(arms), 'rows', len(rows), flush=True)
    Path(a.out).write_text(json.dumps(doc))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
