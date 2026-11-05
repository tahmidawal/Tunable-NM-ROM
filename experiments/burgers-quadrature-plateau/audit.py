"""Offline audit of one pulled job (DESIGN §9 G-audit, G-par, A1.6 fault injections), NumPy + the vendored refstore.
  python audit.py <job dir containing output/result.json> --out checks/<job>-audit.json
JCPREF_STORE must point at a local E0 store copy (read-only). Writes the audit JSON and the plateau summary computed
by plateau.py from result.json (checks/<job>-summary.json). Exit 1 if any audit check fails.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V = HERE / 'vendor' / 'tree' / 'experiments'
sys.path.insert(0, str(V / 'jcp-references'))
sys.path.insert(0, str(V / 'quadrature-burgers3d'))
sys.path.insert(0, str(HERE))
import refstore as RS  # noqa: E402
import verdict as VD  # noqa: E402
import plateau as PL  # noqa: E402

OLD2D = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/runs')
OLD3D = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs')
OLD3D_JOB = {65: 'ho65', 129: 'ho129', 257: 'ho257'}


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    jd = Path(a.job)
    rep = json.loads((jd / 'output' / 'result.json').read_text())
    cfg = rep['config']
    dim, coh = cfg['dim'], cfg['cohort']
    res = dict(job=str(jd), dim=dim, mesh=rep['mesh'], job_id=rep.get('job_id'), gpu=rep.get('gpu'), checks={})
    ok = True
    assert rep.get('complete') is True, 'job incomplete'
    rows = {(r['arm'], r['case']): r for r in rep['rows']}

    # ---------------------------------------------------------------- G-audit: recompute E0 errors in NumPy
    diffs = []
    recomputed = {}
    if dim == 2:
        for f in sorted((jd / 'output' / 'fields').glob('f257_*_c*.npz')):
            arm, c = f.stem[5:].rsplit('_c', 1)
            c = int(c)
            F = np.load(f)['f257']
            assert RS.store_version() == rep['store_version']
            for kind in ('ST', 'S'):
                s = RS.score(F, 2, coh, c, kind)
                recomputed[(arm, c, kind)] = s['e']
                diffs.append(abs(s['e'] - rows[(arm, c)][kind]['e']) / s['e'])
        for f in sorted((jd / 'output' / 'fields').glob('fom_*_c*.npz')):
            name, c = f.stem[4:].rsplit('_c', 1)
            c = int(c)
            F = np.load(f)['f257']
            r = next(x for x in rep['fom_rows'] if x['fom'] == name and x['case'] == c)
            s = RS.score(F, 2, coh, c, 'ST')
            diffs.append(abs(s['e'] - r['ST']['e']) / s['e'])
    else:
        import npcore as N
        p, T = N.load_bank(HERE / cfg['model'])
        G65 = N.bank_np(p, T, N.lattice65_coords())[:, :cfg['Rp']]          # (63^3, R') independent NumPy bank
        Z = np.load(jd / 'output' / 'fields' / 'coefficients.npz')
        for key in Z.files:
            arm, c = key.split('__c')
            c = int(c)
            F = Z[key] @ G65.T
            for kind in ('ST', 'S'):
                ref, meta = RS.load(3, coh, c, kind)
                e, _ = VD.rel_err(F, ref, meta['judge']['n0'])
                recomputed[(arm, c, kind)] = e
                diffs.append(abs(e - rows[(arm, c)][kind]['e']) / e)
    res['checks']['G_audit'] = dict(compared=len(diffs), max_rel=max(diffs), passed=bool(max(diffs) <= 1e-10))
    ok &= res['checks']['G_audit']['passed']

    # ---------------------------------------------------------------- G-par: parity with the October rollouts
    par = []
    if dim == 2 and coh == 'test64':
        old = OLD2D / f"t{rep['mesh']}" / 'archive' / 'output'
        if old.exists():
            for f in sorted((jd / 'output' / 'fields').glob('f257_*_c*.npz')):
                arm, c = f.stem[5:].rsplit('_c', 1)
                oname = 'ctrl_smolyak8' if arm == 'ctrl_smolyak8' else arm
                o = old / f'audit_acc_{oname}_test64{int(c)}.npz'
                if o.exists():
                    par.append(dict(arm=arm, case=int(c), rel=rel(np.load(f)['f257'], np.load(o)['f257'])))
    if dim == 3 and coh == 'test32' and rep['mesh'] in OLD3D_JOB:
        of = OLD3D / OLD3D_JOB[rep['mesh']] / 'code' / 'output' / 'fields' / 'coefficients.npz'
        if of.exists():
            Zo = np.load(of)
            Z = np.load(jd / 'output' / 'fields' / 'coefficients.npz')
            for key in Z.files:
                arm, c = key.split('__c')
                okey = f"{rep['setup']['arms'][arm]['rule'] if arm != 'tensor' else 'tensor'}_R512__c{c}"
                if okey in Zo.files:
                    par.append(dict(arm=arm, case=int(c), rel=rel(Z[key], Zo[okey])))
    if par:
        worst = max(p_['rel'] for p_ in par)
        res['checks']['G_par'] = dict(compared=len(par), max_rel=worst, passed=bool(worst <= 1e-8),
                                      arms=sorted({p_['arm'] for p_ in par}),
                                      worst_items=sorted(par, key=lambda x: -x['rel'])[:5])
    else:
        res['checks']['G_par'] = dict(compared=0, passed=None, note='no October rollout at this mesh/cohort')
    ok &= res['checks']['G_par']['passed'] is not False          # A1.6: an unexplained parity failure withholds the mesh

    # ---------------------------------------------------------------- A1.6 fault injections
    fi = {}
    if dim == 2:
        f0 = sorted((jd / 'output' / 'fields').glob(f"f257_{cfg['conv_arm']}_c*.npz"))[0]
        c0 = int(f0.stem.rsplit('_c', 1)[1])
        F = np.load(f0)['f257']
        e_ok = RS.score(F, 2, coh, c0, 'ST')['e']
        e_sw = RS.score(F, 2, coh, c0 + 1, 'ST')['e']
    else:
        Z = np.load(jd / 'output' / 'fields' / 'coefficients.npz')
        F = Z[f"{cfg['conv_arm']}__c0"] @ G65.T
        r0, m0 = RS.load(3, coh, 0, 'ST')
        r1, m1 = RS.load(3, coh, 1, 'ST')
        e_ok, e_sw = VD.rel_err(F, r0, m0['judge']['n0'])[0], VD.rel_err(F, r1, m1['judge']['n0'])[0]
    fi['swapped_case'] = dict(e=e_ok, e_swapped=e_sw, fired=bool(abs(e_sw - e_ok) > 1e-3))
    k0 = next(iter(recomputed))                      # the same comparison as G_audit, on a copy with one record altered
    bad = copy.deepcopy(rows[(k0[0], k0[1])])
    bad[k0[2]]['e'] *= 1 + 1e-6
    fi['perturbed_record'] = dict(record=list(k0), fired=bool(abs(recomputed[k0] - bad[k0[2]]['e']) / recomputed[k0] > 1e-10))
    store = Path(os.environ['JCPREF_STORE'])
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
        rel_ = f'{dim}d/{coh}/ST/c000.npz'
        dst = Path(td) / rel_
        dst.parent.mkdir(parents=True)
        b = bytearray((store / rel_).read_bytes())
        b[len(b) // 2] ^= 0xFF
        dst.write_bytes(bytes(b))
        shutil.copy(store / rel_.replace('.npz', '.json'), dst.with_suffix('.json'))
        os.environ['JCPREF_STORE'] = td
        try:
            RS.load(dim, coh, 0, 'ST')
            fired = False
        except RS.ReferenceIntegrityError:
            fired = True
        finally:
            os.environ['JCPREF_STORE'] = str(store)
    fi['perturbed_reference_byte'] = dict(fired=fired)
    res['checks']['fault_injections'] = dict(fi, passed=all(v['fired'] for v in fi.values()))
    ok &= res['checks']['fault_injections']['passed']

    # ---------------------------------------------------------------- in-job gates (copied for the record)
    res['in_job_gates'] = rep['gates']
    res['passed'] = bool(ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    summ = dict(plateau_ST=PL.plateau(rep, 'ST'), plateau_S=PL.plateau(rep, 'S'), matching_ST=PL.matching(rep, 'ST'),
                arm_stats_ST=PL.arm_stats(rep, 'ST'), arm_stats_S=PL.arm_stats(rep, 'S'),
                fom_stats_ST=PL.fom_stats(rep, 'ST'), fom_stats_S=PL.fom_stats(rep, 'S'),
                cheapest=PL.cheapest_plateau_rule(rep))
    Path(a.out.replace('-audit.json', '-summary.json')).write_text(json.dumps(summ, indent=1, default=float) + '\n')
    print(json.dumps({k: v.get('passed') for k, v in res['checks'].items()}), 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
