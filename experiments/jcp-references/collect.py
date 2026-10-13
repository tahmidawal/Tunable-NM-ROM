"""jcp-references collector: pulled job outputs -> store/, STATUS.json, results/summary.json (DESIGN §10, A2.4, A2.5).

    python collect.py                     # scans runs/*/output (pulled, non-sealed) and runs/*/result.json

- FD case files (npz + json) of non-sealed cohorts are hard-linked into store/<dim>d/<cohort>/<kind>/ and listed in
  store/MANIFEST.sha256 (committed; the npz files are git-ignored and tracked by this manifest).
- PS cross-check runs -> store/<dim>d/<cohort>/PS/; the ordered cross-check verdict (verdict.crosscheck) per case,
  using the selected runs (A3.6) and the case's FD R_ST record; dimension status (verdict.dimension_status).
- E0 verdict per dimension (A3.5), sealed/OOD QC aggregates (from the sealed jobs' result.json, never fields),
  calibration, gates, old-vs-new distances (A1.9) -> results/summary.json.
Nothing here opens a sealed cohort.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import cohorts as CH
import verdict as V

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
STORE = HERE / 'store'
RES = HERE / 'results'
WT = HERE.parents[2]
OLD2D = {'dev6': WT / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output',
         'val32': WT / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output',
         'test64': WT / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/reft/archive/output'}
OLD3D = {'val64': WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output/ref_923801.npz',
         'test32': WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output/ref_923901.npz'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(dst) != sha(src):
            raise RuntimeError(f'conflicting store file {dst}')
        return
    os.link(src, dst)


def gather():
    """Link every pulled non-sealed FD / PS file into the store; return {attempt: result.json}."""
    results = {}
    for rd in sorted(RUNS.glob('*/')):
        rj = rd / 'result.json'
        if not rj.exists():
            continue
        results[rd.name] = json.loads(rj.read_text())
        if results[rd.name].get('SMOKE_OVERRIDE'):
            continue                                   # A4.4(v): smoke outputs never enter the store
        out = rd / 'output'
        for f in sorted(out.glob('*d/*/*/c*.*')):
            if f.suffix in ('.npz', '.json') and not f.name.endswith('.tmp'):
                dim, coh = int(f.parts[-4][0]), f.parts[-3]
                if CH.is_sealed(dim, coh):
                    raise RuntimeError(f'sealed file pulled locally: {f}')
                link(f, STORE / f.relative_to(out))
    return results


def write_manifest():
    lines = []
    for f in sorted(STORE.glob('*d/*/*/c*.*')):
        lines.append(f'{sha(f)}  {f.relative_to(STORE)}')
    (STORE / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
    return hashlib.sha256((STORE / 'MANIFEST.sha256').read_bytes()).hexdigest()


def fd_meta(dim, cohort, kind, case):
    p = STORE / f'{dim}d/{cohort}/{kind}/c{case:03d}.json'
    return json.loads(p.read_text()) if p.exists() else None


def fd_finest(dim, cohort, kind, case):
    m = fd_meta(dim, cohort, kind, case)
    a = np.load(STORE / f'{dim}d/{cohort}/{kind}/c{case:03d}.npz')
    return np.asarray(a[m['final'][2]]), m


def crosschecks(dim):
    out = {}
    coh = 'val32' if dim == 2 else 'val64'
    for j in CH.crosscheck_cases(dim):
        pj = STORE / f'{dim}d/{coh}/PS/c{j:03d}.json'
        m = fd_meta(dim, coh, 'ST', j)
        if not pj.exists() or m is None:
            out[f'{coh}/{j}'] = dict(verdict='MISSING')
            continue
        ps = json.loads(pj.read_text())
        a = np.load(STORE / f'{dim}d/{coh}/PS/c{j:03d}.npz')
        u3, _ = fd_finest(dim, coh, 'ST', j)
        Ns = ps['Ns']
        runs = {r['id']: r for r in ps['runs']}
        half = ps['selected_half']
        full = next(i for i in ps['selected_spatial'] if runs[i]['N'] == Ns[-1])
        rec = V.crosscheck(m['judge'], u3, dict(
            spatial=[(runs[i]['N'], np.asarray(a[i])) for i in ps['selected_spatial']], half=np.asarray(a[half]),
            full=np.asarray(a[full]), finite=ps['finite'], gates_ok=ps['gates_ok'], stability_ok=ps['stability_ok']),
            m['judge']['n0'])
        # X2: the old first-order reference in place of u3, judged with the NEW eps_ref (A1.14)
        old = old_reference(dim, coh, j, 'ST')
        if old is not None and rec.get('sigma') is not None:
            x_old = float(np.max(V.tnorm(np.asarray(a[half]) - old)[1:]) / m['judge']['n0'])
            rec['x2_old'] = dict(x_ps=x_old, verdict='FAIL' if x_old > 2 * rec['eps_ref'] else 'PASS')
        rec['t0_ps_vs_canonical'] = float(np.linalg.norm((np.asarray(a['t0_ps']) - np.asarray(a['t0'])).ravel())
                                          / m['judge']['n0'])
        out[f'{coh}/{j}'] = V_clean(rec)
    return out


def old_reference(dim, cohort, case, kind):
    """The old first-order reference on the same lattice (2D f257; 3D c<j>), sha-checked against its job record."""
    if dim == 2:
        d = OLD2D.get(cohort)
        if d is None:
            return None
        p = d / f'ref_{kind}_{cohort}_{case:03d}.npz'
        if not p.exists():
            return None
        f = np.load(p)['f257']
        rj = json.loads((d / 'result.json').read_text())
        rec = [c for c in rj['cases'] if c['cohort'] == cohort and c['case'] == case and c['ref'] == kind]
        assert rec and hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() == rec[0]['f257_sha256'], p
        return f
    if kind != 'ST':
        return None
    p = OLD3D.get(cohort)
    if p is None or not p.exists():
        return None
    done = json.loads(p.with_suffix('.done').read_text())
    assert sha(p) == done['sha256'], p
    return np.asarray(np.load(p)[f'c{case}'])


def old_vs_new(dim):
    rows = []
    for cohort in CH.HISTORICAL[dim]:
        for kind in ('ST', 'S'):
            for case in range(CH.count(dim, cohort)):
                m = fd_meta(dim, cohort, kind, case)
                if m is None:
                    continue
                old = old_reference(dim, cohort, case, kind)
                if old is None:
                    continue
                u3, _ = fd_finest(dim, cohort, kind, case)
                n0 = m['judge']['n0']
                t0dev = float(np.linalg.norm((old[0] - u3[0]).ravel()) / n0)
                e_t = (V.tnorm(old - u3)[1:] / n0).tolist()
                rows.append(dict(cohort=cohort, kind=kind, case=case, delta_old=max(e_t), delta_t=e_t,
                                 eps_ref=m['judge']['eps_ref'], t0_dev=t0dev, outcome=m['outcome'],
                                 fires=bool(m['judge']['eps_ref'] is not None and max(e_t) > m['judge']['eps_ref'])))
    return rows


def V_clean(x):
    if isinstance(x, dict):
        return {str(k): V_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [V_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return V_clean(x.tolist())
    return x


def e0_verdict(dim, dstat, cases, xc):
    """A3.5: FAIL iff suspended/unvalidated; PASS iff every required case-kind eligible; else PASS-WITH-K0-CASES."""
    if dstat != 'validated':
        return 'FAIL', []
    bad = []
    for c in cases:
        x = xc.get(f"{c['cohort']}/{c['case']}")
        if c['outcome'] != 'accepted' or (x is not None and x['verdict'] != 'PASS'):
            bad.append(f"{c['cohort']}/{c['case']}/{c['kind']}")
    return ('PASS' if not bad else 'PASS-WITH-K0-CASES'), bad


def main():
    results = gather()
    version = write_manifest()
    summary = dict(store_version=version, jobs={}, dims={})
    for name, r in results.items():
        summary['jobs'][name] = dict(mode=r['config']['mode'], dim=r['config'].get('dim'),
                                     cohort=r['config'].get('cohort'), complete=r.get('complete'), job=r.get('job'),
                                     gates_all_pass=r.get('gates_all_pass'), seconds=r.get('seconds'),
                                     smoke_override=bool(r.get('SMOKE_OVERRIDE')))
        if r['config']['mode'] == 'gates':
            summary['gates'] = r['gates']
        if r['config']['mode'] == 'calib':
            summary.setdefault('calib', {})[str(r['config']['dim'])] = {k: r.get(k) for k in ('calib', 'd1', 'x1', 'x4')}
        if 'qc' in r:
            summary.setdefault('sealed_qc', {})[f"{r['config']['dim']}d/{r['config']['cohort']}/{name}"] = dict(
                qc=r['qc'], manifest_cipher_sha256=r['sealed_manifest']['cipher_sha256'],
                manifest_plain_sha256=r['sealed_manifest']['plain_sha256'], n_ciphertexts=len(r['sealed_manifest']['ciphertexts']))
    status = dict(store_version=version, dimension_status={}, crosscheck={})
    for dim in (2, 3):
        xc = crosschecks(dim)
        verdicts = [v['verdict'] for v in xc.values()]
        dstat = V.dimension_status(verdicts) if 'MISSING' not in verdicts else 'unvalidated'
        cases = []
        for cohort in CH.HISTORICAL[dim]:
            for kind in ('ST', 'S'):
                for case in range(CH.count(dim, cohort)):
                    m = fd_meta(dim, cohort, kind, case)
                    if m is None:
                        cases.append(dict(cohort=cohort, kind=kind, case=case, outcome='MISSING'))
                        continue
                    jd = m['judge']
                    cases.append(dict(cohort=cohort, kind=kind, case=case, outcome=m['outcome'], eps_ref=jd['eps_ref'],
                                      p=jd['p'], c=jd['c'], d23=jd['d23'], tau=jd['tau'], peclet=jd['peclet'],
                                      refined=m['refined'], tight=m['tight_used'], levels=jd['levels'],
                                      nu=m['desc']['nu'], seconds=m.get('seconds'), gpu=m['job']['gpu'],
                                      job_id=m['job']['job_id'], commit=m['job']['commit'],
                                      newton_total=sum(a['newton_total'] for a in m['attempts'])))
        e0, bad = e0_verdict(dim, dstat, cases, xc)
        status['dimension_status'][str(dim)] = dstat
        status['crosscheck'][str(dim)] = {k: v['verdict'] for k, v in xc.items()}
        summary['dims'][str(dim)] = dict(dimension_status=dstat, e0_verdict=e0, ineligible=bad, crosscheck=xc,
                                         cases=cases, old_vs_new=old_vs_new(dim))
    (STORE / 'STATUS.json').write_text(json.dumps(V_clean(status), indent=1) + '\n')
    RES.mkdir(exist_ok=True)
    (RES / 'summary.json').write_text(json.dumps(V_clean(summary), indent=1) + '\n')
    print('store version', version)
    for dim in (2, 3):
        s = summary['dims'][str(dim)]
        print(dim, s['dimension_status'], s['e0_verdict'], len(s['cases']), 'cases',
              sum(c['outcome'] == 'accepted' for c in s['cases']), 'accepted')


if __name__ == '__main__':
    main()
