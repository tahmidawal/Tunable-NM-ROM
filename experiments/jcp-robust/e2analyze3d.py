"""jcp-robust E2 3D analysis per bank (DESIGN 3.3, 3.4; A1.8-A1.10, A2.6, A2.7, A3.5, A3.6). CPU (offline first).

Inputs: checks/select3d-<b>.json, the two selection job dirs with their E0 scores (e2score3.py), the front timing job dir.
  * H-acc-r over widths 256 -> 512 -> 768 -> 1024 on the FIXED rule gl64 (Q2 vs gl80), specs accurate fs|CN|0.01|K1 and fast
    qn|BDF2|0.025|k4 (p1stats.h_acc_r_transition per transition; per-case qualification), plus the selected-Gauss rows;
  * E0 median / worst error per width and spec (gl64 and selected rule);
  * Q1 + Q2 of the four fixed qualify configurations on this bank;
  * the front of the tested universe: slots (width, spec, family) with their realised rule + the fixed configurations,
    cost = median B/A of the panel (A = W256|gl24|fs|CN|0.01|K1 on this bank), dominance among qualified candidates (A3.5);
  * G-E2.3 (bank 0): incumbent replay of W256|lat8192|qn|BDF2|0.025|k4 and W512|lat16384|qn|BDF2|0.025|k4 (qualify q3d3),
    W256|gl24|fs|CN|0.01|K1 (qualify q3d2), production W of every case, relative max-norm <= 1e-10, hashes pinned.
Writes checks/e2-3d-<b>.json.

    python e2analyze3d.py <bank> --qa <dir> --qb <dir> --sa <scores> --sb <scores> [--front <dir>]
"""
import offline  # noqa: F401

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import e1stats as E
import p1stats as S
from make_configs_e2_3d import FIXED

HERE = Path(__file__).resolve().parent
QR = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')
REPLAY = [('q3d3', 'W256|lat8192|qn|BDF2|0.025|k4'), ('q3d3', 'W512|lat16384|qn|BDF2|0.025|k4'), ('q3d2', 'W256|gl24|fs|CN|0.01|K1')]


def load_job(d, sc):
    res = json.loads((Path(d) / 'result.json').read_text())
    lad = {}
    for blk in res['blocks']:
        if blk['kind'] == 'ladder':
            lad.update(blk['result'])
    return res, lad, json.loads(Path(sc).read_text())['scores'], np.load(Path(d) / f"W_{res['mode']}.npz")


def case_q(x, nxt, d_adt, d_next):
    """per-case qualification of a ladder rung for H-acc-r (code audit 2 #22): the configuration's pooled stationarity
    (<= 1 %), and on the case valid production / ADT / successor runs within their bars (distances recomputed by the
    caller), rho <= bar, and on the tighter cases 0-7 a valid tighter run within the tight/tighter bar."""
    pooled_ok = bool(x['steps'] > 0 and x['nonstationary'] / x['steps'] <= .01)
    tc = list(x['tighter_cases'])
    out = []
    for i, j in enumerate(x['cases']):
        ok = bool(pooled_ok and x['valid'][i] and x['adt_valid'][i] and d_adt[i] <= E.BAR_Q1 and nxt['valid'][i] and
                  d_next[i] <= E.BAR_Q2 and max(E.num(v) for v in x['rho'][i]) <= E.BAR_RHO)
        if j in tc:
            k = tc.index(j)
            ok = ok and bool(x['adt_tighter_valid'][k]) and E.num(x['adt_tight_tighter'][k]) <= E.BAR_TT
        out.append(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank')
    for k in ('qa', 'qb', 'sa', 'sb', 'front'):
        ap.add_argument(f'--{k}')
    a = ap.parse_args()
    b = int(a.bank)
    sel = json.loads((HERE / 'checks' / f'select3d-{b}.json').read_text())
    J = [load_job(a.qa, a.sa), load_job(a.qb, a.sb)]
    lad, scores, Wz_of = {}, {}, {}
    for res, l_, s_, W_ in J:
        lad.update(l_)
        scores.update(s_)
        for st in res['config']['settings']:
            Wz_of[st['name']] = W_
    rep = dict(bank=b, jobs={r['config']['attempt']: dict(job_id=r['job_id'], complete=bool(r.get('complete')), backend=r['backend'])
                             for r, *_ in J}, selection=sel['slots'], configurations={}, h_acc_r={}, errors={})
    rep['integrity'] = bool(all(v['complete'] and v['backend'] == 'gpu' for v in rep['jobs'].values())
                            and all(v.get('ok') for v in sel['jobs'].values()))
    import c3common as C3
    import e2score3 as ES
    import score3 as SC
    ES.install_union([J[0][0]['config'], J[1][0]['config']])
    tab = C3.C.table(923801, 64)
    n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in range(64)}

    def dist3(sname, A_, B_, ids):
        G = SC.glat(sname)
        return [max(float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[j] for k in range(1, 6)) for i, j in enumerate(ids)]
    E_ = lambda key: {int(c): v['e'] for c, v in scores[key].items()}
    EL = lambda key: {int(c): bool(v['eligible']) for c, v in scores[key].items()}
    EP = lambda key: {int(c): E.num(v['eps_ref']) for c, v in scores[key].items()}
    for spec in ('fs|CN|0.01|K1', 'qn|BDF2|0.025|k4'):
        out = {}
        for lo, hi in ((256, 512), (512, 768), (768, 1024)):
            ka, kb = f'W{lo}|gl64|{spec}', f'W{hi}|gl64|{spec}'
            if ka not in lad or kb not in lad or ka not in scores or kb not in scores:
                out[f'{lo}->{hi}'] = dict(outcome='not reached (missing)')
                continue
            qs = []
            for w_, k_ in ((lo, ka), (hi, kb)):
                Wz = Wz_of[f'W{w_}']
                x = lad[k_]
                sp_ = spec.split('|')
                kad = f"W{w_}|gl64|ad|LSPG|{sp_[1]}|{float(sp_[2]):.8g}|tight"
                d_adt = dist3(f'W{w_}', Wz[k_], Wz[kad], x['cases'])
                d_nx = dist3(f'W{w_}', Wz[k_], Wz[f'W{w_}|gl80|{spec}'], x['cases'])
                if any(not (np.isfinite(E.num(j)) and abs(o - E.num(j)) <= 1e-9 * abs(E.num(j)) + 1e-13) for o, j in zip(d_adt, x['dist_adt']) if np.isfinite(o)):
                    rep['integrity'] = False
                qs.append(case_q(x, lad[f'W{w_}|gl80|{spec}'], d_adt, d_nx))
            qa, qb = qs
            ea, eb, ela, elb, ep = E_(ka), E_(kb), EL(ka), EL(kb), EP(ka)
            cs = [c for c in sorted(ea) if c in eb and ela[c] and elb[c]]
            idx = {c: i for i, c in enumerate(lad[ka]['cases'])}
            out[f'{lo}->{hi}'] = S.h_acc_r_transition([eb[c] - ea[c] for c in cs], [ep[c] for c in cs],
                                                      [qa[idx[c]] for c in cs], [qb[idx[c]] for c in cs])
            out[f'{lo}->{hi}']['median_paired_change'] = float(np.median([eb[c] - ea[c] for c in cs])) if cs else None
        out['range_256_1024'] = S.h_acc_r_range([v['outcome'] for v in out.values() if isinstance(v, dict) and 'outcome' in v])
        rep['h_acc_r'][f'gl64|{spec}'] = out
        for wd in (256, 512, 768, 1024):
            k = f'W{wd}|gl64|{spec}'
            if k in scores:
                e = list(E_(k).values())
                rep['errors'][f'gl64|W{wd}|{spec}'] = dict(median=float(np.median(e)), worst=float(np.max(e)))
    for slot in sel['slots']:
        if slot['rule']:
            k = f"W{slot['width']}|{slot['rule']}|{slot['spec']}"
            cell = sel['cells'][f"W{slot['width']}|{slot['spec']}|{slot['family']}"][slot['rule']]
            e = list(E_(k).values()) if k in scores else []
            rep['configurations'][f"slot|W{slot['width']}|{slot['spec']}|{slot['family']}"] = dict(
                rule=slot['rule'], qualified=cell['qualified'], median_e=float(np.median(e)) if e else None,
                worst_e=float(np.max(e)) if e else None)
    for f in FIXED:
        st, rl, *sp = f.split('|')
        spec = '|'.join(sp)
        fam = 'gauss' if rl.startswith('gl') else 'lattice'
        cell = sel['cells'].get(f'{st}|{spec}|{fam}', {}).get(rl)
        k = f'{st}|{rl}|{spec}'
        e = list(E_(k).values()) if k in scores else []
        rep['configurations'][f'fixed|{f}'] = dict(rule=rl, qualified=(cell or {}).get('own'),
                                                   median_e=float(np.median(e)) if e else None, worst_e=float(np.max(e)) if e else None)
    # front
    if a.front:
        fres = json.loads((Path(a.front) / 'result.json').read_text())
        bank_of = lambda c: (sorted({s_['bank'] for s_ in c['settings']}), sorted((c.get('bank_registry') or {}).items()))
        if bank_of(fres['config']) != bank_of(J[0][0]['config']):          # code audit 3 #23: same bank identity
            fres = dict(fres, complete=False)
        tm = next(iter(fres['timing'].values()))
        ba = {}
        if tm.get('valid') is True and fres.get('complete') and fres['backend'] == 'gpu':     # code audit 2 #23: d3run's
            per = {}                                                                          # panel acceptance governs
            for inv in tm['invocations']:
                if inv.get('hashes_ok') and 1 / 1.1 <= inv['drift'] <= 1.1:
                    per.setdefault(inv['B'], {}).setdefault(inv['case'], []).append(inv['ratio'])
            for nm_, pc in per.items():
                if all(len(pc.get(c, [])) >= 2 for c in range(8)):
                    ba[nm_] = [float(np.median(pc[c])) for c in range(8)]
        pts = {}
        for k, v in rep['configurations'].items():
            if k.startswith('slot|'):
                _, w, *sp = k.split('|')
                name = f"{w}|{v['rule']}|{'|'.join(sp[:-1])}"
            else:
                name = k.split('|', 1)[1]
            cost = float(np.median(ba[name])) if name in ba else None
            pts[k] = dict(v, cost=cost)
        q = {k: v for k, v in pts.items() if v['qualified'] is True and v['cost'] is not None and v['median_e'] is not None}
        for k, v in pts.items():
            if v['qualified'] is None or v['cost'] is None or v['median_e'] is None:
                v['state'] = 'missing'
            elif v['qualified'] is False:
                v['state'] = 'unqualified'
            else:
                dom = any(o['cost'] <= v['cost'] and o['median_e'] <= v['median_e'] and o['worst_e'] <= v['worst_e'] and
                          (o['cost'] < v['cost'] or o['median_e'] < v['median_e'] or o['worst_e'] < v['worst_e'])
                          for kk, o in q.items() if kk != k)
                v['state'] = 'dominated' if dom else 'member'
        for slot in sel['slots']:
            k = f"slot|W{slot['width']}|{slot['spec']}|{slot['family']}"
            pts.setdefault(k, dict(state='absent' if slot['outcome'] == 'none within ladder' else 'missing'))
        rep['front_65'] = dict(points=pts, timing_valid=tm.get('valid'), label='front of the tested 3D universe (never complete)')
    if b == 0:
        rows = {}
        for job, key in REPLAY:
            arch = QR / job / 'archive' / 'output' / f'q3run_{job}' / 'W_q2.npz'
            pin = next((l.split()[0] for l in (QR / job / 'archive' / 'OUTPUTS.sha256').read_text().splitlines()
                        if l.split()[1] == f'output/q3run_{job}/W_q2.npz'), None)
            if not arch.exists() or hashlib.sha256(arch.read_bytes()).hexdigest() != pin:
                rows[key] = dict(passed=False, reason='archive unavailable or hash differs from its pin')
                continue
            z = np.load(arch)
            ak = f'{key}#q|' + key.split('|', 2)[2]
            Wa, ida = z[ak], [int(x) for x in z['cases|' + ak]]
            st, rl, *sp = key.split('|')
            nk = f"{st}|{rl}|{'|'.join(sp)}"
            Wz = Wz_of[st]
            Wn, idn = Wz[nk], [int(x) for x in Wz['cases|' + nk]]
            if idn != list(range(64)) or sorted(ida) != list(range(64)):
                rows[key] = dict(passed=False, reason='population differs')
                continue
            if Wn.shape != Wa.shape or not (np.isfinite(Wn).all() and np.isfinite(Wa).all()):
                rows[key] = dict(passed=False, reason='shape or non-finite values')
                continue
            rel = [float(np.max(np.abs(Wn[i] - Wa[ida.index(j)])) / max(float(np.max(np.abs(Wa[ida.index(j)]))), 1e-300))
                   for i, j in enumerate(idn)]
            rows[key] = dict(passed=bool(max(rel) <= 1e-10), worst=max(rel))
        rep['G-E2.3'] = dict(rows=rows, passed=all(v['passed'] for v in rows.values()))
    (HERE / 'checks' / f'e2-3d-{b}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps({k: v.get('range_256_1024') for k, v in rep['h_acc_r'].items()}))


if __name__ == '__main__':
    main()
