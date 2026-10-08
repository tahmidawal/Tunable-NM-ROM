"""Independent audit of a pulled fomrun job (DESIGN section 9, A1.10, A2.21, A3.5). NumPy re-scoring of the saved
dev6 fields against the pinned references; inventory/verification/calibration/timing-hash checks; the FOM
reproduction gate against the 2D lane's FOM rows; and an independent local re-run of BE at dt0 with the project's
own engines.make_fom (not fom2.py) on dev6 case 0 (tolerance 1e-8 of ||u0||, cross-GPU).

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/audit_fom.py runs/<attempt>/archive [--no-rerun]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_t2 as AT  # noqa: E402  (NumPy cohort, references, initial field)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--no-rerun', action='store_true')
    a = ap.parse_args()
    arc = Path(a.archive)
    res = json.loads((arc / 'output/result.json').read_text())
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    cfg = res['config']
    ck, info = {}, {}
    ck['complete'] = bool(res.get('complete'))
    ck['backend_gpu'] = res['backend'] == 'gpu' and 'jax_backend=gpu' in logs and 'precision=highest' in logs
    man = json.loads((AT.REFSRC / 'result.json').read_text())
    ent = {(x['cohort'], x['case'], x['ref']): x['f257_sha256'] for x in man['cases']}
    cases = [(coh, c) for coh in cfg['cohorts'] for c in range(dict(dev6=6, val32=32)[coh])]
    runs = [f'fom|{sc}|{f:g}' for sc in cfg['schemes'] for f in cfg['dt_factors']]
    rows = {(r['run'], r['cohort'], r['case']): r for r in res['rows']}
    ck['inventory'] = len(rows) == len(res['rows']) == len(runs) * len(cases) and \
        set(rows) == {(k, coh, c) for k in runs for coh, c in cases}
    ck['all_verified'] = all(r['verified'] for r in res['rows'])
    info['unverified'] = [r['run'] + f"|{r['cohort']}{r['case']}" for r in res['rows'] if not r['verified']]
    # re-score saved dev6 fields
    Fz = np.load(arc / 'output/fom_fields_dev6.npz')
    bad = []
    for k in runs:
        for coh, c in [x for x in cases if x[0] == 'dev6']:
            fr = Fz[f'{k}|{coh}|{c}'.replace('|', '__')]
            ph = AT.cohort(coh)[c]
            n0r = np.linalg.norm(AT.initial257(ph))
            rST = np.load(AT.REFSRC / f'ref_ST_{coh}_{c:03d}.npz')['f257']
            rS = np.load(AT.REFSRC / f'ref_S_{coh}_{c:03d}.npz')['f257']
            assert AT.sha(rST) == ent[(coh, c, 'ST')] and AT.sha(rS) == ent[(coh, c, 'S')]
            for tag, R_ in (('ST', rST), ('S', rS), ('TX', (16 * rST - rS) / 15)):
                e = max(np.linalg.norm(fr[j] - R_[j]) / n0r for j in range(1, 6))
                if not AT.close(e, rows[(k, coh, c)][f'e_{tag}']):
                    bad.append((k, coh, c, tag))
            if np.abs(fr[0] - AT.initial257(ph)).max() > 1e-12:
                bad.append((k, coh, c, 't0_field'))
    ck['dev6_rescored'] = not bad
    info['rescore_mismatches'] = bad[:10]
    # calibration rule re-applied
    cal_bad = []
    for kk, e_ in res['calibration'].items():
        ok = [nt for nt in sorted(cfg['ntols'], reverse=True)
              if all(r_['ref_verified'] and r_['diffs'][str(nt)]['verified'] and r_['diffs'][str(nt)]['diff'] < .01 * r_['e_ST_ref']
                     for r_ in e_['cases'])]
        if e_['chosen_ntol'] != (ok[0] if ok else 1e-10):
            cal_bad.append(kk)
    ck['calibration_rule'] = not cal_bad and len(res['calibration']) == len(runs)
    # timing hashes and inventory
    inv = res['timing']['invocations']
    subj = res['timing']['subjects']
    want = {(b, ci, r_) for b in subj if b != res['timing']['A'] for ci in range(cfg['timing']['cases']) for r_ in range(cfg['timing']['reps'])}
    got = [(i['B'], i['case_index'], i['rep']) for i in inv]
    ck['timing_inventory'] = len(got) == len(set(got)) and set(got) == want
    ck['timing_hashes'] = all(i['A_expected_sha'] and i['B_expected_sha'] and i['A_sha'][0] == i['A_sha'][1] == i['A_expected_sha']
                              and i['B_sha'] == i['B_expected_sha'] for i in inv)
    # FOM reproduction vs the 2D lane (BE dt0, ntol 1e-6, ltol 1e-8 = the lane's truth settings)
    lane = json.loads((AT.LANE_DV / f"dv{res['mesh']}/archive/output/result.json").read_text())
    lr = {(r['cohort'], r['case']): r for r in lane['fom_rows']}
    dd = [max(abs(x['e_ST'] - lr[(x['cohort'], x['case'])]['ref_ST_evolved']), abs(x['e_S'] - lr[(x['cohort'], x['case'])]['ref_S_evolved']))
          for x in res['repro']]
    info['repro_max_abs_diff'] = max(dd)
    ck['fom_reproduction'] = max(dd) <= 1e-6 and len(dd) == len(cases)
    # orders (reported) and the local independent BE re-run
    info['orders'] = {f"{o['scheme']}|{o['cohort']}{o['case']}": o['orders'] for o in res['order']}
    if not a.no_rerun:
        sys.path.insert(0, str(AT.ROOT / 'experiments/mr-burgers2d'))
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import engines as e
        L = res['mesh']
        ph = AT.cohort('dev6')[0]
        q, _ = e.make_fom(L, .005)
        nt = res['timing']['chosen_ntol']['BE|1']
        f = np.asarray(q(jnp.asarray(e.initial(L, ph)), float(ph[4]), nt, 1e-8)[0])[:, ::L // 256, ::L // 256]
        n0r = np.linalg.norm(AT.initial257(ph))
        d = max(np.linalg.norm(f[j] - Fz['fom__BE__1__dev6__0'][j]) / n0r for j in range(6))
        info['independent_BE_rerun_diff'] = float(d)
        ck['independent_BE_rerun'] = d <= 1e-8
    ck_all = all(ck.values())
    out = dict(attempt=cfg['attempt'], job_id=res['job_id'], commit=res['commit'], checks=ck, info=info, all_pass=ck_all)
    (HERE / 'checks' / f"audit-{cfg['attempt']}.json").write_text(json.dumps(out, indent=1, default=str) + '\n')
    for k, v in ck.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print(json.dumps(info, default=str)[:2000])
    print('AUDIT ALL PASS' if ck_all else 'AUDIT FAILED')
    sys.exit(0 if ck_all else 1)


if __name__ == '__main__':
    main()
