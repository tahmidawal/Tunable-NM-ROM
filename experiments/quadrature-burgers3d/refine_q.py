"""quadrature-burgers3d: errors against the refined reference, in NumPy, from a pulled panel (DESIGN.md sections 5, 9).

    refine_q.py --panels runs/val65/code/output runs/val129/code/output ... --ref <ref_<seed>.npz> --out <json>

For every panel: every ROM arm's fields on the 65-node lattice are decoded from the saved output coefficients with an
independent NumPy bank (npcore.bank_np), the FOM's from its saved f65 fields; worst/median over cases of the
evolved-time maximum of ||u - u_ref|| / ||u_ref(0)||. Where the panel computed the same numbers in-job, they must agree
to 1e-10 (recorded). Cross-mesh distances between consecutive panels (same arm, same case) on the lattice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import npcore as N

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panels', nargs='+', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--model', default=str(HERE / 'inputs' / 'model_M2'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--richardson', default=None,
                    help='R5 diagnostic: panel dir whose 257-node same-grid reference gives u_R = 2 u_513 - u_257')
    a = ap.parse_args()
    p, T = N.load_bank(a.model)
    G65 = N.bank_np(p, T, N.lattice65_coords())                                   # (63^3, R)
    done = json.loads(Path(a.ref).with_suffix('.done').read_text())
    ref_sha = hashlib.sha256(Path(a.ref).read_bytes()).hexdigest()
    assert done['sha256'] == ref_sha and done['accepted'] is True, done
    z = np.load(a.ref)
    assert int(z['n']) == done['n'] and float(z['dt']) == done['dt'] and float(z['ntol']) == done['ntol'], done
    out = dict(done=done, ref=a.ref, ref_sha256=hashlib.sha256(Path(a.ref).read_bytes()).hexdigest(), seed=int(z['seed']),
               panels={}, cross_mesh={})
    fields, n0_all = {}, {}
    RICH = None
    if a.richardson:
        rr = json.loads((Path(a.richardson) / 'result.json').read_text())
        assert rr['mesh'] == 257 and rr['cohort']['seed'] == int(z['seed'])
        sg257 = np.load(Path(a.richardson) / 'fields' / 'same_grid_ref65.npz')
        RICH = {j: 2.0 * np.asarray(z[f'c{j}']) - np.asarray(sg257[f'c{j}']) for j in range(rr['cohort']['count'])}
        out['richardson'] = dict(panel=str(a.richardson), note='u_R = 2 u_513(dt .0025) - u_257(dt .005), diagnostic',
                                 ref513_vs_R=[float((np.linalg.norm(np.asarray(z[f'c{j}']) - RICH[j], axis=1) /
                                                     np.linalg.norm(RICH[j][0]))[1:].max()) for j in RICH])
    for pdir in a.panels:
        pdir = Path(pdir)
        rep = json.loads((pdir / 'result.json').read_text())
        n = rep['mesh']
        assert rep['cohort']['seed'] == int(z['seed']), (rep['cohort']['seed'], z['seed'])
        cases = list(range(rep['cohort']['count']))
        RR = {j: np.asarray(z[f'c{j}']) for j in cases}
        n0 = {j: float(np.linalg.norm(RR[j][0])) for j in cases}
        n0_all.update(n0)
        coef = np.load(pdir / 'fields' / 'coefficients.npz')
        for k_, v_ in rep['config']['expected_ref'].items():
            assert done[k_] == v_, (k_, done[k_], v_)
        res = dict(mesh=n, job_id=rep.get('job_id'), arms={}, fom={}, max_abs_diff_vs_in_job=0.0, in_job_compared=0)
        rich_per = {}
        sg = np.load(pdir / 'fields' / 'same_grid_ref65.npz')
        res['same_grid_reference'] = [float(N.worst_evolved(np.asarray(sg[f'c{j}']), RR[j])[1:].max()) for j in cases]
        res['initial_match'] = im = max(float(np.abs(np.asarray(sg[f'c{j}'])[0] - RR[j][0]).max() / np.abs(RR[j][0]).max())
                                   for j in cases)
        assert im <= 1e-12, im
        for nm, arm in rep['arms'].items():
            Rp = arm['spec']['Rp']
            per, curves = [], {}
            for q in arm.get('quick', []):
                j = q['case']
                cj = np.asarray(coef[f'{nm}__c{j}'])
                F = cj @ G65[:, :Rp].T
                fields[(n, nm, j)] = cj
                if RICH is not None:
                    rich_per.setdefault(nm, []).append(float(N.worst_evolved(F, RICH[j])[1:].max()))
                e = N.worst_evolved(F, RR[j])
                curves[j] = e
                per.append(float(e[1:].max()))
                if 'err_refined' in q:
                    res['in_job_compared'] += 1
                    res['max_abs_diff_vs_in_job'] = max(res['max_abs_diff_vs_in_job'],
                                                        float(np.abs(np.asarray(q['err_refined']) - e).max()))
            if per:
                res['arms'][nm] = dict(Rp=Rp, rule=arm['spec']['rule'], family=arm['spec']['family'], cases=len(per),
                                       worst=max(per), median=float(np.median(per)), per_case=per)
                if RICH is not None:
                    res['arms'][nm]['richardson_worst'] = max(rich_per[nm])
                    res['arms'][nm]['richardson_median'] = float(np.median(rich_per[nm]))
        for name, fom in rep['fom'].items():
            per = []
            for q in fom['quick']:
                j = q['case']
                F = np.asarray(np.load(pdir / 'fields' / f'{name}_c{j}.npz')['f65'])
                e = N.worst_evolved(F, RR[j])
                per.append(float(e[1:].max()))
                if RICH is not None:
                    rich_per.setdefault(name, []).append(float(N.worst_evolved(F, RICH[j])[1:].max()))
                if 'err_refined' in q:
                    res['in_job_compared'] += 1
                    res['max_abs_diff_vs_in_job'] = max(res['max_abs_diff_vs_in_job'],
                                                        float(np.abs(np.asarray(q['err_refined']) - e).max()))
            res['fom'][name] = dict(worst=max(per), median=float(np.median(per)), per_case=per)
            if RICH is not None:
                res['fom'][name]['richardson_worst'] = max(rich_per[name])
        out['panels'][str(n)] = res
        print(f"mesh {n}: refined (worst) " + ', '.join(f"{k} {v['worst']:.4%}" for k, v in res['arms'].items()))
        print(f"mesh {n}: FOM refined (worst) " + ', '.join(f"{k} {v['worst']:.4%}" for k, v in res['fom'].items()))
        print(f"mesh {n}: same-grid reference vs refined worst {max(res['same_grid_reference']):.4%}; "
              f"in-job diff {res['max_abs_diff_vs_in_job']:.2e}")
    meshes = sorted({k[0] for k in fields})
    for n1, n2 in zip(meshes[:-1], meshes[1:]):
        arms = sorted({k[1] for k in fields if k[0] == n1} & {k[1] for k in fields if k[0] == n2})
        res = {}
        for nm in arms:
            per = []
            for (m_, a_, j) in fields:
                if m_ == n1 and a_ == nm and (n2, nm, j) in fields:
                    dc = fields[(n1, nm, j)] - fields[(n2, nm, j)]
                    dF = dc @ G65[:, :dc.shape[1]].T
                    per.append(float((np.linalg.norm(dF, axis=1) / n0_all[j])[1:].max()))
            res[nm] = dict(worst=max(per), median=float(np.median(per)), cases=len(per))
        out['cross_mesh'][f'{n1}-{n2}'] = res
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
