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


class MalformedEvidence(ValueError):
    """Evidence of the wrong shape / mapping: an integrity failure, never a numerical outcome (code audit 7 #2, #6)."""


def _safe_max(xs):
    """max with any non-finite term -> +inf (code audit 4 #32)."""
    return math.inf if (not xs or not all(math.isfinite(x) for x in xs)) else max(xs)
QARCH = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d3/'
             'archive/output/q3run_q3d3/W_q2.npz')
QKEY = {'E1-3D-a': 'W256|lat8192|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4',
        'E1-3D-b': 'W512|lat16384|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4'}


def agree(off, job):
    off, job = E.num(off), E.num(job)
    if not (np.isfinite(off) and np.isfinite(job)):
        return (not np.isfinite(off)) and (not np.isfinite(job))
    return bool(abs(off - job) <= 1e-9 * abs(job) + 1e-13)


def check_scores(sc, res, Wz, wpath):
    """code audits 9 #5, 10 #2: the E0 scores come from THIS job (job id, cohort seed, coefficient archive sha256), were
    computed by the bank-aware scorer on the job's own decoder banks, against the pinned E0 store (version, STATUS sha,
    cohort), and hold exactly the job's saved coefficient keys, each covering exactly its saved cases."""
    import hashlib
    import c3common as C3_
    import score3 as SC_
    if str(sc.get('job_id')) != str(res.get('job_id')) or int(sc.get('seed', -1)) != int(res['config']['cohort_seed']):
        raise ValueError(f"scores from job {sc.get('job_id')} / seed {sc.get('seed')} do not belong to job {res.get('job_id')}")
    if sc.get('W_sha256') != hashlib.sha256(Path(wpath).read_bytes()).hexdigest() or sc.get('scorer') != 'e2score3 (bank-aware)':
        raise ValueError('scores not computed by the bank-aware scorer on this coefficient archive')
    if (sc.get('store_version'), sc.get('status_sha'), sc.get('cohort')) != \
            (SC_.STORE_VERSION, SC_.STATUS_SHA, SC_.COHORT[int(res['config']['cohort_seed'])]):
        raise ValueError('scores against another E0 store / status / cohort')
    reg = res['config'].get('bank_registry') or {}
    for s_ in res['config']['settings']:
        want = reg[s_['bank']]['sha256'] if s_['bank'] in reg else C3_.BANKS[s_['bank']][1]
        if want != 'SMOKE-RUNTIME' and sc.get('decoder_banks', {}).get(s_['name']) != [s_['bank'], want]:
            raise ValueError(f"scores of {s_['name']} decoded on another bank")
    keys = [k for k in Wz.files if not k.startswith(('cases|', 'wall|', 'fomfield|'))]
    if sorted(sc['scores']) != sorted(keys):
        raise ValueError('score keys differ from the saved coefficient keys')
    for k in keys:
        want = sorted(int(c) for c in Wz['cases|' + k])
        if sorted(int(c) for c in sc['scores'][k]) != want:
            raise ValueError(f'scores of {k} do not cover its {len(want)} saved cases')


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
    Wz0 = np.load(out / 'W_e1.npz')
    check_scores(sc, res, Wz0, out / 'W_e1.npz')
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
        A_, B_ = np.asarray(A_), np.asarray(B_)
        if not (A_.ndim == 3 and A_.shape == B_.shape and A_.shape[0] == len(ids) and A_.shape[1] == 6
                and A_.shape[2] == np.asarray(G).shape[0]):
            raise MalformedEvidence(f'coefficient shapes {A_.shape} vs {B_.shape} for {len(ids)} cases')   # audits 6 #3, 7 #2
        return [_safe_max([float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[j] for k in range(1, 6)]) for i, j in enumerate(ids)]

    for cf in cfg['configs']:
        try:
            nm, sname, spec = cf['name'], cf['setting'], cf['spec']
            r = res['configs'][nm]
            mism = []
            kt = key_of(f"{sname}|{nm}|TARGET|{cf['target']}", spec)
            Wt, idt = Wz[kt], [int(x) for x in Wz['cases|' + kt]]
            assert idt == cases
            kc = key_of(f"{sname}|{nm}|CHECK|{cf['check']}", spec)
            Wc, idc = Wz[kc], [int(x) for x in Wz['cases|' + kc]]
            dchk = dist(sname, [Wt[cases.index(j)] for j in idc], Wc, idc)
            smoke = str(cfg.get('attempt', '')).startswith('smk')
            # independent contracts (code audit 4 #2): production 64 cases / check 0-7 / draws 0-15; smoke (smk1) 8 / 0-7 / 0-2
            want_cases, want_draws = (list(range(8)), [0, 1, 2]) if smoke else (list(range(64)), list(range(E.NDRAW)))
            if cases != want_cases or idc != list(range(8)) or [int(x) for x in cf['draws']] != want_draws:
                mism.append(('population', 'cases / check cases / draws differ from the registered contract', None))
            if len(r['target_run']['dist_check']) != len(idc) or len(r['target_run']['check_valid']) != len(idc) \
                    or len(r['target_run']['valid']) != len(cases):
                mism.append(('TARGET-CHECK', None, 'length'))
            mism += [('TARGET-CHECK', j, 'dist') for j, (x, y) in zip(idc, zip(dchk, r['target_run']['dist_check'])) if not agree(x, y)]
            tc_ok = bool(idc and idc == [j for j in range(8) if j in cases] and len(r['target_run']['check_valid']) == len(idc)
                         and len(r['target_run']['dist_check']) == len(idc) and all(r['target_run']['check_valid'])
                             and len(r['target_run']['valid']) == len(cases) and all(r['target_run']['valid'][cases.index(j)] for j in idc) and all(np.isfinite(dchk)) and max(dchk) <= E.BAR_TC
                         and all(agree(x, y) for x, y in zip(dchk, r['target_run']['dist_check'])))
            tvalid = r['target_run']['valid']
            tgt_ok = (None if not tc_ok else bool(all(tvalid)))
            e_t = {j: sc['scores'][kt][str(j)]['e'] for j in cases} if kt in sc['scores'] else None
            draws = {}
            sp_ = spec.split('|')
            adk = lambda lev: f"ad|LSPG|{sp_[1]}|{float(sp_[2]):.8g}|{lev}"
            tcs = [j for j in range(8) if j in cases]                                  # registered tighter cases 0-7
            if not smoke and [str(x) for x in cf['draws']] != [str(x) for x in range(E.NDRAW)]:
                mism.append(('population', 'draws', cf['draws']))
            for s in [str(x) for x in cf['draws']]:
                dr = r['draws'][s]
                rule = dr['rule']
                tag = f"{sname}|{nm}|{rule}"
                k = key_of(tag, spec)
                W, ids = Wz[k], [int(x) for x in Wz['cases|' + k]]
                if ids != cases:
                    mism.append((s, 'cases', None))
                d_t = [x if tvalid[i] else math.inf for i, x in enumerate(dist(sname, W, Wt, ids))]
                ks = key_of(f"{tag}|succ|{dr['succ']}", spec)
                d_s = dist(sname, W, Wz[ks], ids)
                ka, kb = f"{tag}|{adk('tight')}", f"{tag}|{adk('tighter')}"          # Q1 recomputed (code audit 1 #5)
                Wa, Wb = Wz[ka], Wz[kb]
                ib = [int(x) for x in Wz['cases|' + kb]]
                for kk_, want_ in ((ks, cases), (ka, cases), (kb, tcs)):            # saved case mappings (code audit 4 #2)
                    if [int(x) for x in Wz['cases|' + kk_]] != want_:
                        mism.append((s, f'case mapping {kk_}', None))
                d_a = dist(sname, W, Wa, ids)
                d_tt = dist(sname, [Wa[cases.index(j)] for j in ib], Wb, ib)
                for nmx, off, job in (('target', d_t, dr['dist_target']), ('succ', d_s, dr['dist_succ']), ('adt', d_a, dr['dist_adt']),
                                      ('tt', d_tt, dr['adt_tight_tighter'])):
                    if len(off) != len(job):
                        mism.append((s, nmx, 'length'))
                        continue
                    mism += [(s, nmx, j) for j, (x, y) in zip(ids, zip(off, job)) if not agree(x, E.num(y))]
                if ib != tcs:
                    mism.append((s, 'tighter cases', ib))
                # stationarity recomputed from the saved fresh certificates (rows gam_cert of the production runs)
                cr = [x for x in res['rows'] if x.get('tag') == tag and x.get('spec') == spec and 'gam_cert' in x]
                S_exp = int(round(.25 / float(sp_[2])))
                if sorted(int(x['case']) for x in cr) != sorted(cases) or any(len(x['gam_cert']) != S_exp for x in cr):
                    mism.append((s, 'certificate rows (cases or lengths)', None))
                g_all = [E.num(v) for x in cr for v in x['gam_cert']]
                nonst = (sum(1 for v in g_all if not (np.isfinite(v) and v <= E.GAMMA)) / len(g_all)) if g_all and len(cr) == len(cases) else None
                sc_ = sc['scores'][k]
                valid = [bool(x) for x in dr['valid']]
                e = [sc_[str(j)]['e'] if valid[i] else math.inf for i, j in enumerate(ids)]
                qual, own, status = E.draw_qualified(
                    dict(valid=valid, valid_tight=dr['adt_valid'], valid_tighter=dr['adt_tighter_valid'], valid_succ=dr['succ_valid']),
                    dict(tight=d_a, tt=d_tt, succ=d_s, rho_succ=[max(E.num(v) for v in x) for x in dr['rho_succ']],
                         target=d_t, rho_tgt=[max(E.num(v) for v in x) for x in dr['rho_tgt']], nonst_frac=nonst),
                    tgt_ok, len(cases), len(tcs))
                D = None if tgt_ok is not True else (math.inf if not all(valid) else max(d_t))
                draws[s] = dict(rule=rule, rule_sha256=dr['rule_sha256'], qualified=qual, own_family_qualified=own, status=status,
                                D=D, rho_tgt_max=max(max(E.num(v) for v in x) for x in dr['rho_tgt']),
                                dist_succ_max=max(d_s), dist_adt_max=max(d_a), nonst_frac=nonst, e=e,
                                eps=[sc_[str(j)]['eps_ref'] for j in ids], eligible=[sc_[str(j)]['eligible'] for j in ids],
                                valid=valid)
            dl = [draws[str(x)] for x in cf['draws']]
            tgt_full = tgt_ok is True
            elig = [all(d['eligible'][i] for d in dl) for i in range(len(cases))]
            ok_e = [i for i in range(len(cases)) if elig[i]]
            h_a = E.h_e1a([d['qualified'] for d in dl]) if tgt_full else \
                dict(E.h_e1a([d['own_family_qualified'] for d in dl]), basis='Q1 + own-family Q2 only (target not certified / failed)')
            if ok_e:
                eps = np.array([E.num(dl[0]['eps'][i]) for i in range(len(cases))])
                delta = 2 * float(np.max(eps[ok_e]))                         # A3.4
                agg = lambda fn, d: math.inf if any(not d['valid'][i] for i in ok_e) else float(fn([d['e'][i] for i in ok_e]))
                stats = {}
                if tgt_full:
                    stats['D'] = E.rank_interval([d['D'] for d in dl], 0.)
                stats['median_e'] = E.rank_interval([agg(np.median, d) for d in dl], delta)
                stats['worst_e'] = E.rank_interval([agg(np.max, d) for d in dl], delta)
                econv = [(e_t[j] if (e_t and tc_ok and tvalid[i]) else None) for i, j in enumerate(cases)]   # per case (A4.2)
                failed = np.array([[not d['valid'][i] for i in range(len(cases))] for d in dl])
                emat = np.array([d['e'] for d in dl], float)
                h_c = E.h_e1c(emat, emat[0], failed, eps, econv, elig)
                h_b = dict(label=E.h_e1b({k_: v[2] for k_, v in stats.items()}), stats={k_: list(v) for k_, v in stats.items()},
                           delta_error=delta)
            else:
                h_c, h_b = dict(label='unavailable', n=0), dict(label='unavailable')
                econv = [None] * len(cases)
            ind = 'unavailable (target not certified / failed)'
            if tgt_full:
                ind = dict(tp=0, fp=0, fn=0, tn=0)
                for d in dl:
                    i_ok, r_ok = d['rho_tgt_max'] <= E.BAR_RHO, d['D'] <= E.BAR_Q2
                    ind[('t' if i_ok == r_ok else 'f') + ('p' if i_ok else 'n')] += 1
                ind['draw0'] = dict(rho_ok=dl[0]['rho_tgt_max'] <= E.BAR_RHO, rollout_ok=dl[0]['D'] <= E.BAR_Q2)
            verd = dict(H_E1a=h_a, H_E1b=h_b, H_E1c=h_c, indicator=ind, eligible_cases=len(ok_e))
            ctl = dict(r.get('controls', {}))
            if cf.get('bad'):
                fx_ = ctl.get('fixture', {})
                ctl['C-E1.1'] = (E.fixture_fired(pan3[0]['states'], fx_.get('rho_tgt')) if pan3 else
                                 dict(admitted=False, fired=False, reason='establishment missing (control unavailable)'))
                if pan3:
                    ctl['C-E1.1']['M_job'] = res['settings_M'][sname]['M']
                    ctl['C-E1.1']['M_cpu'] = pan3[0].get('M')
                    ctl['C-E1.1']['fired'] = bool(ctl['C-E1.1']['fired'] and ctl['C-E1.1']['M_job'] == ctl['C-E1.1']['M_cpu'])
            idn = ctl.get('identity', {})
            ctl['G-E1.1'] = dict(idn, passed=bool(idn.get('s0_equals_deployed') and idn.get('s1_pure_shift') and idn.get('s1_differs')))
            ce = ctl.get('C-E1.2', {})
            ctl['C-E1.2'] = dict(ce, fired=bool(ce.get('raised') and ce.get('true_claim_passes')))
            pm = ctl.get('permute')
            if cf.get('permute'):
                ctl['G-E1.3'] = dict(pm or {}, passed=bool(pm and pm.get('passed')))
            # G-E1.2 reproduction against qualify q3d3 (A2.1, A3.6): draw 0 vs archived 'q' production W, every case
            arch = Path(a.archive)
            pin = next((l_.split()[0] for l_ in (QARCH.parents[2] / 'OUTPUTS.sha256').read_text().splitlines()
                        if l_.split()[1] == 'output/q3run_q3d3/W_q2.npz'), None)
            if arch.exists() and nm in QKEY and hashlib.sha256(arch.read_bytes()).hexdigest() == pin:
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
                if rows is not None and not np.isfinite(W0).all():
                    rows = None
                ctl['G-E1.2'] = (dict(passed=False, reason='case missing or non-finite new output') if rows is None else
                                 dict(passed=bool(max(rows) <= 1e-10), worst=max(rows), n=len(rows),
                                      archive_sha256=hashlib.sha256(arch.read_bytes()).hexdigest()))
            else:
                ctl['G-E1.2'] = dict(passed=False, reason='archive unavailable or hash differs from the qualify pin')
            extra = {}
            for rl, ex in r.get('extra', {}).items():
                k = key_of(f"{sname}|{nm}|{rl}", spec)
                extra[rl] = dict(D=max(dist(sname, Wz[k], Wt, cases)), rho_tgt_max=max(E.num(v) for row in ex['rho_tgt'] for v in row),
                                 median_e=float(np.median([sc['scores'][k][str(j)]['e'] for j in cases])))
            if 'bad_rollout' in ctl:
                b = ctl['bad_rollout']
                k = key_of(f"{sname}|{nm}|{b['rule']}", spec)
                ctl['bad_rollout_summary'] = dict(D=max(dist(sname, Wz[k], Wt, cases)), any_invalid=not all(b['valid']))
            integ = dict(ok=bool(not mism and res.get('complete') and (smoke or len(cases) == 64)), mismatches=mism, smoke=smoke)
            rep['configs'][nm] = dict(integrity=integ, setting=sname, spec=spec, target_cert=dict(certified=tc_ok, dist=dchk, cases=idc,
                                                                                   label='checked on the registered subset (cases 0-7)'),
                                      target_valid=tvalid, draws=draws, verdicts=verd, controls=ctl, extra=extra,
                                      mismatches=mism, econv=econv)
            print(nm, {k_: (v['label'] if isinstance(v, dict) and 'label' in v else v) for k_, v in verd.items()}, flush=True)
        except (KeyError, ValueError, IndexError, OSError) as e_:          # absent evidence -> unavailable, never a crash or a pass
            rep['configs'][cf['name']] = dict(setting=cf['setting'], spec=cf['spec'], integrity=dict(ok=False, reason=f'missing evidence: {e_!r}'),
                                              verdicts={}, controls={})
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'{a.tag}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
