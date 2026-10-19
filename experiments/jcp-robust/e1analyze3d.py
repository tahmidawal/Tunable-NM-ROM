"""jcp-robust E1 offline analysis, 3D (DESIGN 2; A1.2-A1.5, A2.1-A2.3, A3.2-A3.4, A4.2). CPU (offline.py first).

Inputs: the e1run3d output directory and its E0 scores (`score3.py --run <dir> --out <scores>`, copied unchanged from
jcp-dims/jcp-speed3d). Every in-job lattice distance used by a verdict (to the target, to the successor, to the ADT anchor,
the target check) is recomputed here from the saved coefficients with an independent decode (score3.glat) and compared
with the in-job value. Writes checks/<tag>.json.

    python e1analyze3d.py <e1run3d output dir> <scores json> [--tag e1-3d] [--archive <qualify q3d3 W_q2.npz>]
"""
from __future__ import annotations

import offline  # noqa: F401  (CPU, highest, x64)

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import e1stats as E
import score3 as SC

HERE = Path(__file__).resolve().parent
QARCH = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d3/'
             'archive/output/q3run_q3d3/W_q2.npz')
QKEY = {'E1-3D-a': 'W256|lat8192|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4',
        'E1-3D-b': 'W512|lat16384|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4'}


def agree(off, job):
    if not (np.isfinite(off) and np.isfinite(job)):
        return (not np.isfinite(off)) and (not np.isfinite(job))
    return bool(abs(off - job) <= 1e-9 * abs(job) + 1e-13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('scores')
    ap.add_argument('--tag', default='e1-3d')
    ap.add_argument('--archive', default=str(QARCH))
    a = ap.parse_args()
    import c3common as C3
    out = Path(a.out)
    res = json.loads((out / 'result.json').read_text())
    assert res.get('complete'), 'job output incomplete'
    sc = json.loads(Path(a.scores).read_text())
    Wz = np.load(out / 'W_e1.npz')
    cfg = res['config']
    tab = C3.C.table(int(cfg['cohort_seed']), int(cfg['cohort_count']))
    cases = [int(x) for x in res['cohort']['cases']]
    n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in cases}
    est = HERE / 'checks' / 'establish-e1.json'
    pan3 = [p for p in json.loads(est.read_text())['panels'] if p['panel'].startswith('3d|')] if est.exists() else []
    rep = dict(source=str(out), job_id=res['job_id'], commit=res['commit'], gpu=res['gpu'], backend=res['backend'], configs={})

    def key_of(tag, spec):
        return f'{tag}|{spec}'

    def dist(sname, A_, B_, ids):
        G = SC.glat(sname)
        return [max(float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[j] for k in range(1, 6)) for i, j in enumerate(ids)]

    for cf in cfg['configs']:
        nm, sname, spec = cf['name'], cf['setting'], cf['spec']
        r = res['configs'][nm]
        kt = key_of(f"{sname}|{nm}|TARGET|{cf['target']}", spec)
        Wt, idt = Wz[kt], [int(x) for x in Wz['cases|' + kt]]
        assert idt == cases
        kc = key_of(f"{sname}|{nm}|CHECK|{cf['check']}", spec)
        Wc, idc = Wz[kc], [int(x) for x in Wz['cases|' + kc]]
        dchk = dist(sname, [Wt[cases.index(j)] for j in idc], Wc, idc)
        tc_ok = bool(idc and all(r['target_run']['check_valid']) and all(np.isfinite(dchk)) and max(dchk) <= E.BAR_TC
                     and all(agree(x, y) for x, y in zip(dchk, r['target_run']['dist_check'])))
        tvalid = r['target_run']['valid']
        tgt_ok = (None if not tc_ok else bool(all(tvalid)))
        e_t = {j: sc['scores'][kt][str(j)]['e'] for j in cases} if kt in sc['scores'] else None
        draws, mism = {}, []
        for s in [str(x) for x in cf['draws']]:
            dr = r['draws'][s]
            rule = dr['rule']
            k = key_of(f"{sname}|{nm}|{rule}", spec)
            W, ids = Wz[k], [int(x) for x in Wz['cases|' + k]]
            assert ids == cases
            d_t = dist(sname, W, Wt, ids)
            ks = key_of(f"{sname}|{nm}|{rule}|succ|{dr['succ']}", spec)
            d_s = dist(sname, W, Wz[ks], ids)
            for nmx, off, job in (('target', d_t, dr['dist_target']), ('succ', d_s, dr['dist_succ'])):
                mism += [(s, nmx, j) for j, (x, y) in zip(ids, zip(off, job)) if not agree(x, y)]
            nonst = sum(dr['nonst_per_case']) / max(1, dr['steps'])
            sc_ = sc['scores'][k]
            e = [sc_[str(j)]['e'] for j in ids]
            qual, own, status = E.draw_qualified(
                dict(valid=dr['valid'], valid_tight=dr['adt_valid'], valid_tighter=dr['adt_tighter_valid'], valid_succ=dr['succ_valid']),
                dict(tight=dr['dist_adt'], tt=dr['adt_tight_tighter'], succ=d_s, rho_succ=[max(x) for x in dr['rho_succ']],
                     target=d_t, rho_tgt=[max(x) for x in dr['rho_tgt']], nonst_frac=nonst), tgt_ok)
            draws[s] = dict(rule=rule, rule_sha256=dr['rule_sha256'], qualified=qual, own_family_qualified=own, status=status,
                            D=(max(d_t) if all(np.isfinite(d_t)) else math.inf), rho_tgt_max=float(np.max(dr['rho_tgt'])),
                            dist_succ_max=max(d_s), dist_adt_max=max(dr['dist_adt']), nonst_frac=nonst, e=e,
                            eps=[sc_[str(j)]['eps_ref'] for j in ids], eligible=[sc_[str(j)]['eligible'] for j in ids],
                            valid=dr['valid'])
        dl = [draws[str(x)] for x in cf['draws']]
        elig = [all(d['eligible'][i] for d in dl) for i in range(len(cases))]
        ok_e = [i for i in range(len(cases)) if elig[i]]
        eps = np.array([dl[0]['eps'][i] if dl[0]['eps'][i] is not None else np.nan for i in range(len(cases))])
        delta = 2 * float(np.nanmax(eps[ok_e])) if ok_e else float('nan')
        agg = lambda fn, d: math.inf if any(not d['valid'][i] for i in ok_e) else float(fn([d['e'][i] for i in ok_e]))
        stats = {}
        if tc_ok:
            stats['D'] = E.rank_interval([d['D'] for d in dl], 0.)
        stats['median_e'] = E.rank_interval([agg(np.median, d) for d in dl], delta)
        stats['worst_e'] = E.rank_interval([agg(np.max, d) for d in dl], delta)
        econv = [(e_t[j] if (e_t and tc_ok and tvalid[i]) else None) for i, j in enumerate(cases)]
        failed = np.array([[not d['valid'][i] for i in range(len(cases))] for d in dl])
        emat = np.array([[d['e'][i] if d['valid'][i] else np.inf for i in range(len(cases))] for d in dl])
        ind = dict(tp=0, fp=0, fn=0, tn=0)
        for d in dl:
            if not tc_ok:
                break
            i_ok, r_ok = d['rho_tgt_max'] <= E.BAR_RHO, d['D'] <= E.BAR_Q2
            ind[('t' if i_ok == r_ok else 'f') + ('p' if i_ok else 'n')] += 1
        verd = dict(H_E1a=E.h_e1a([d['qualified'] for d in dl]), H_E1a_own_family=E.h_e1a([d['own_family_qualified'] for d in dl]),
                    H_E1b=dict(label=E.h_e1b({k_: v[2] for k_, v in stats.items()}), stats={k_: list(v) for k_, v in stats.items()},
                               delta_error=delta),
                    H_E1c=E.h_e1c(emat, emat[0], failed, eps, econv, elig),
                    indicator=ind if tc_ok else 'unavailable (target not certified)')
        ctl = dict(r.get('controls', {}))
        if 'fixture' in ctl:
            ctl['C-E1.1'] = (E.fixture_fired(pan3[0]['states'], ctl['fixture']['rho_tgt']) if pan3 else
                             dict(admitted=False, fired=False, reason='establishment missing (control unavailable)'))
            if pan3:
                ctl['C-E1.1']['M_job'] = res['settings_M'][sname]['M']
                ctl['C-E1.1']['M_cpu'] = pan3[0].get('M')
                ctl['C-E1.1']['fired'] = bool(ctl['C-E1.1']['fired'] and ctl['C-E1.1']['M_job'] == ctl['C-E1.1']['M_cpu'])
        idn = ctl.get('identity', {})
        ctl['G-E1.1'] = dict(idn, passed=bool(idn.get('s0_equals_deployed') and idn.get('s1_pure_shift') and idn.get('s1_differs')))
        # G-E1.2 reproduction against qualify q3d3 (A2.1, A3.6): draw 0 vs archived 'q' production W, every case
        arch = Path(a.archive)
        if arch.exists() and nm in QKEY:
            z = np.load(arch)
            Wa, ida = z[QKEY[nm]], [int(x) for x in z['cases|' + QKEY[nm]]]
            W0 = Wz[key_of(f"{sname}|{nm}|lat{cf['n']}s0", spec)]
            rows = []
            for i, j in enumerate(cases):
                if j not in ida:
                    rows = None
                    break
                A_ = Wa[ida.index(j)]
                den = float(np.max(np.abs(A_)))
                rows.append(float(np.max(np.abs(W0[i] - A_)) / den) if den > 0 and np.isfinite(A_).all() else math.inf)
            ctl['G-E1.2'] = (dict(passed=False, reason='case missing') if rows is None else
                             dict(passed=bool(max(rows) <= 1e-10), worst=max(rows), n=len(rows),
                                  archive_sha256=hashlib.sha256(arch.read_bytes()).hexdigest()))
        else:
            ctl['G-E1.2'] = dict(passed=False, reason='archive unavailable')
        extra = {}
        for rl, ex in r.get('extra', {}).items():
            k = key_of(f"{sname}|{nm}|{rl}", spec)
            extra[rl] = dict(D=max(dist(sname, Wz[k], Wt, cases)), rho_tgt_max=float(np.max(ex['rho_tgt'])),
                             median_e=float(np.median([sc['scores'][k][str(j)]['e'] for j in cases])))
        if 'bad_rollout' in ctl:
            b = ctl['bad_rollout']
            k = key_of(f"{sname}|{nm}|{b['rule']}", spec)
            ctl['bad_rollout_summary'] = dict(D=max(dist(sname, Wz[k], Wt, cases)), any_invalid=not all(b['valid']))
        rep['configs'][nm] = dict(setting=sname, spec=spec, target_cert=dict(certified=tc_ok, dist=dchk, cases=idc,
                                                                               label='checked on the registered subset (cases 0-7)'),
                                  target_valid=tvalid, draws=draws, verdicts=verd, controls=ctl, extra=extra,
                                  mismatches=mism, econv=econv)
        print(nm, {k_: (v['label'] if isinstance(v, dict) and 'label' in v else v) for k_, v in verd.items()}, flush=True)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'{a.tag}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
