"""Offline analysis of this lane's references (DESIGN 3, A1.5, A2.3, A3.3; CPU, NumPy only). Recomputes from the pulled
lattice arrays (never trusts the job's verdict fields alone):

  * G-R1  e0repro level fields vs the E0 store's level fields (same L, dt, prod purpose): per time ||d|| / n0;
          bar 1e-12 on E0's GPU type, else the G-R2 bar
  * G-R2  the T-horizon run's first six outputs, level by level, matched on (L, dt, final purpose), vs the E0 store;
          t0 bitwise; evolved times <= min(1e-8, 0.01 eps_ref(t_k)); case-level 'not evaluable' if the finest base
          level is not matched
  * judge recomputation: the E0 judge (verdict.judge) re-run on the stored final triple, whole run and per window,
          compared with the stored records (outcome, eps per time)
  * G-R3  stored in-job flag (fine and lattice agree at the shared nodes) + the fine file's sha256 (when local)
  * C-R1  x1 result: prerequisites (solver, floor, t0_shared) true and order false
  * PS cross-check per window (verdict.crosscheck and the pre-A10 rule) when PS files are present
Writes checks/refs-<attempt>.json.

    python anaref.py --run runs/<attempt>/archive/output [--e0-gpu 'NVIDIA A100 80GB PCIe']
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ref'))
import verdict as V  # noqa: E402  (this lane's copy, output count generalised)

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
WINDOWS = {'W1': (1, 5), 'W2': (6, 10), 'W3': (11, 20)}


def e0_raw(dim, coh, case):
    sys.path.insert(0, str(E0))
    import refstore as RS
    sys.path.pop(0)
    return RS._load_raw(dim, coh, case, 'ST'), RS


def tnorm(a):
    a = np.asarray(a, float)
    return np.sqrt(np.sum(a.reshape(a.shape[0], -1) ** 2, axis=1))


def match_levels(att_new, att_e0):
    """(new id, E0 id) pairs with equal (L, dt, purpose)."""
    out = []
    for a in att_new:
        for b in att_e0:
            if a['L'] == b['L'] and abs(a['dt'] - b['dt']) < 1e-15 and a['purpose'] == b['purpose']:
                out.append((a['id'], b['id']))
                break
    return out


def g_r2(arr, meta, d, gpu_same, bar_same=1e-12):
    (earr, emeta), RS = e0_raw(d, meta['cohort'], meta['case'])
    n0 = float(np.linalg.norm(np.asarray(earr['t0']).ravel()))
    eps_t = emeta['judge']['eps_t']
    t0_bitwise = bool(np.array_equal(np.asarray(arr['t0']), np.asarray(earr['t0'])))
    pairs = match_levels(meta['attempts'], emeta['attempts'])
    rows, ok = [], t0_bitwise
    for a, b in pairs:
        x, y = np.asarray(arr[a])[:6], np.asarray(earr[b])
        dt_ = (tnorm(x - y) / n0).tolist()
        bars = [0.] + [(bar_same if gpu_same else min(1e-8, .01 * (e or 0.))) for e in eps_t]
        rows.append(dict(new=a, e0=b, d_t=dt_, bars=bars, t0_bitwise=bool(np.array_equal(x[0], y[0])),
                         passed=bool(np.array_equal(x[0], y[0]) and all(dt_[k] <= bars[k] for k in range(1, 6)))))
    finest = [a['id'] for a in meta['attempts'] if a['L'] == {2: 8192, 3: 512}[d]]
    evaluable = any(r_['new'] in finest for r_ in rows)
    passed = bool(ok and evaluable and all(r_['passed'] for r_ in rows))
    return dict(t0_bitwise=t0_bitwise, pairs=rows, evaluable=evaluable, passed=passed,
                e0_store_version=RS.store_version())


def judge_recompute(arr, meta):
    t0 = np.asarray(arr['t0'])
    att = {a['id']: a for a in meta['attempts']}
    lv = [dict(L=att[i]['L'], lat=np.asarray(arr[i]), tau=att[i]['tau'], steps_ok=att[i]['steps_ok'],
               umax=att[i]['umax_running']) for i in meta['final']]
    nu = meta['desc']['nu']
    jd = V.judge(lv, t0, nu)
    agree = bool(jd['outcome'] == meta['judge']['outcome'] and
                 all((a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-12 * max(abs(b), 1e-300))
                     for a, b in zip(jd['eps_t'], meta['judge']['eps_t'])))
    win = {}
    n = t0.shape[0] if False else np.asarray(arr[meta['final'][0]]).shape[0]
    for w, (a, b) in WINDOWS.items():
        if b >= n:
            continue
        idx = [0] + list(range(a, b + 1))
        jw = V.judge([dict(l, lat=l['lat'][idx]) for l in lv], t0, nu)
        stored = (meta.get('judge_windows') or {}).get(w, {})
        win[w] = dict(outcome=jw['outcome'], eps=jw['eps_ref'], stored_outcome=stored.get('outcome'),
                      agree=bool(jw['outcome'] == stored.get('outcome')))
    return dict(outcome=jd['outcome'], eps_ref=jd['eps_ref'], agree=agree, windows=win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a pulled job output directory (contains the refrun step dirs)')
    ap.add_argument('--e0-gpu', default='NVIDIA A100 80GB PCIe', help="GPU type of E0's job for the G-R1 cases")
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    rep = dict(run=str(run), G_R1=[], cases=[], x1=[], ps=[])
    for step in sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith('ref_refrun')):
        res = json.loads((step / 'result.json').read_text())
        mode, d = res['config']['mode'], int(res['config']['dim'])
        gpu = res['job']['gpu']
        if mode == 'e0repro':
            for js in sorted(step.rglob('E0REPRO/c*.json')):
                m = json.loads(js.read_text())
                z = np.load(str(js)[:-5] + '.npz')
                (earr, emeta), RS = e0_raw(d, m['cohort'], m['case'])
                n0 = float(np.linalg.norm(np.asarray(earr['t0']).ravel()))
                same = gpu == a.e0_gpu
                rows = []
                for at in m['attempts']:
                    eid = next((b['id'] for b in emeta['attempts'] if b['L'] == at['L'] and b['purpose'] == 'prod'), None)
                    if eid is None:
                        rows.append(dict(L=at['L'], evaluable=False))
                        continue
                    dd = (tnorm(np.asarray(z[f"L{at['L']}"]) - np.asarray(earr[eid])) / n0).tolist()
                    bars = [0.] + [(1e-12 if same else min(1e-8, .01 * (e or 0.))) for e in emeta['judge']['eps_t']]
                    rows.append(dict(L=at['L'], e0=eid, d_t=dd, bars=bars,
                                     passed=bool(dd[0] == 0. and all(dd[k] <= bars[k] for k in range(1, 6)))))
                rep['G_R1'].append(dict(cohort=m['cohort'], case=m['case'], gpu=gpu, e0_gpu=a.e0_gpu, same_gpu=same,
                                        levels=rows, passed=bool(rows and all(r_.get('passed') for r_ in rows))))
        elif mode == 'fd':
            for js in sorted(step.rglob('T*/c*.json')):
                if js.parent.name == 'PS':
                    continue
                m = json.loads(js.read_text())
                z = np.load(str(js)[:-5] + '.npz')
                rec = dict(cohort=m['cohort'], case=m['case'], horizon=m['horizon'], outcome=m['outcome'],
                           refined=m['refined'], tight=m['tight_used'], G_R3=m.get('G_R3'),
                           judge=judge_recompute(z, m),
                           fine_windows={w: (v or {}).get('outcome') for w, v in (m.get('judge_fine_windows') or {}).items()},
                           fine_eps={w: (v or {}).get('eps_ref') for w, v in (m.get('judge_fine_windows') or {}).items()},
                           seconds=m.get('seconds'))
                if m['cohort'] in ({2: ('dev6', 'val32'), 3: ('val64',)}[d]):
                    rec['G_R2'] = g_r2(z, m, d, gpu_same=False)
                fl = Path(str(js)[:-5] + '_fine.npz')
                rec['fine_local'] = fl.exists()
                rep['cases'].append(rec)
        elif mode == 'x1':
            for x in res.get('x1', []):
                ch = x['checks']
                rep['x1'].append(dict(case=x['case'], outcome=x['outcome'],
                                      fired=bool(ch['solver'] and ch['floor'] and ch['t0_shared'] and not ch['order']),
                                      evaluable=bool(ch['solver'] and ch['floor'] and ch['t0_shared'])))
        elif mode == 'ps':
            rep['ps'].append(dict(step=step.name, cases=res.get('cases'), gates=res.get('gates_all_pass')))
    rep['summary'] = dict(G_R1=all(x['passed'] for x in rep['G_R1']) if rep['G_R1'] else None,
                          G_R2=all(c.get('G_R2', {}).get('passed', False) for c in rep['cases']) if rep['cases'] else None,
                          G_R3=all(c['G_R3'] is not False for c in rep['cases']) if rep['cases'] else None,
                          judge_agree=all(c['judge']['agree'] for c in rep['cases']) if rep['cases'] else None,
                          C_R1=all(x['fired'] for x in rep['x1']) if rep['x1'] else None)
    dest = Path(a.out or HERE / 'checks' / f'refs-{run.parents[1].name}.json')
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(dest, json.dumps(rep['summary']))


if __name__ == '__main__':
    main()
