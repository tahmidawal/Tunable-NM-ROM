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


class MalformedEvidence(ValueError):
    """Evidence of the wrong shape / mapping: an integrity failure, never a numerical outcome (code audit 7 #2, #6)."""


def _safe_max(xs):
    """max with any non-finite term -> +inf (code audit 4 #32)."""
    return math.inf if (not xs or not all(math.isfinite(x) for x in xs)) else max(xs)
QR = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')
REPLAY = [('q3d3', 'W256|lat8192|qn|BDF2|0.025|k4'), ('q3d3', 'W512|lat16384|qn|BDF2|0.025|k4'), ('q3d2', 'W256|gl24|fs|CN|0.01|K1')]


def load_job(d, sc):
    res = json.loads((Path(d) / 'result.json').read_text())
    lad = {}
    for blk in res['blocks']:
        if blk['kind'] == 'ladder':
            lad.update(blk['result'])
    S_ = json.loads(Path(sc).read_text())
    Wz = np.load(Path(d) / f"W_{res['mode']}.npz")
    check_scores(S_, res, Wz, Path(d) / f"W_{res['mode']}.npz")
    return res, lad, S_['scores'], Wz


def case_q(x, nxt, d_adt, d_next, d_tt, nonst):
    """per-case qualification of a ladder rung for H-acc-r (code audits 2 #22, 5 #5): pooled stationarity recomputed from
    the saved certificates (nonst, <= 1 %); on the case valid production / ADT / successor runs within their bars
    (distances recomputed by the caller), rho <= bar; on the registered tighter cases 0-7 a valid tighter run within the
    tight/tighter bar (recomputed). Any missing evidence -> every case False is not used: the caller passes None."""
    pooled_ok = bool(nonst is not None and nonst <= .01)
    out = []
    for i, j in enumerate(x['cases']):
        ok = bool(pooled_ok and x['valid'][i] and x['adt_valid'][i] and d_adt[i] <= E.BAR_Q1 and nxt['valid'][i] and
                  d_next[i] <= E.BAR_Q2 and max(E.num(v) for v in x['rho'][i]) <= E.BAR_RHO)
        if j < 8:
            ok = ok and bool(x['adt_tighter_valid'][j]) and d_tt[j] <= E.BAR_TT
        out.append(ok)
    return out


def expected_bank3d(b):
    """(bank name, registry entry) the jobs of bank b must carry (code audit 8 #5)."""
    import json as _j
    if int(b) == 0:
        return 'W1024', None
    meta = _j.loads((HERE / 'inputs' / 'banks3d' / f's{b}' / 'BANK.json').read_text())
    return f'W1024s{b}', dict(path=f'inputs/banks3d/s{b}/bank.pkl', sha256=meta['bank_sha256'])


def check_bank3d(cfg, b):
    name, reg = expected_bank3d(b)
    banks = {s_['bank'] for s_ in cfg['settings']}
    regs = cfg.get('bank_registry') or {}
    ok = banks == {name} and int(cfg.get('bank_seed', -1)) == int(b) and (regs.get(name) == reg if reg else not regs)
    if not ok:
        raise SystemExit(f"job {cfg.get('attempt')} is not on bank {b}: banks {banks}, seed {cfg.get('bank_seed')}, registry {regs}")


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
    ap.add_argument('bank')
    for k in ('qa', 'qb', 'sa', 'sb', 'front'):
        ap.add_argument(f'--{k}')
    a = ap.parse_args()
    b = int(a.bank)
    sel = json.loads((HERE / 'checks' / f'select3d-{b}.json').read_text())
    J = [load_job(a.qa, a.sa), load_job(a.qb, a.sb)]
    for (res_, *_), att_ in zip(J, (f'e2q{b}a', f'e2q{b}b')):     # registered roles (code audit 11 #3, applied to 3D)
        check_bank3d(res_['config'], b)
        if res_['config'].get('attempt') != att_:
            raise SystemExit(f"input in role {att_} is job {res_['config'].get('attempt')}")
    if sorted(sel['jobs']) != sorted(r_['config']['attempt'] for r_, *_ in J) or \
            any(str(sel['jobs'][r_['config']['attempt']]['job_id']) != str(r_['job_id']) for r_, *_ in J):
        raise SystemExit('selection file does not come from these two jobs')
    lad, scores, Wz_of = {}, {}, {}
    for res, l_, s_, W_ in J:
        lad.update(l_)
        clash = set(scores) & set(s_)
        if clash:                                                    # halves must not share score keys (audit 10 #2)
            raise SystemExit(f'score keys present in both job halves: {sorted(clash)[:3]}')
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
        A_, B_ = np.asarray(A_), np.asarray(B_)
        if not (A_.ndim == 3 and A_.shape == B_.shape and A_.shape[0] == len(ids) and A_.shape[1] == 6
                and A_.shape[2] == np.asarray(G).shape[0]):
            raise MalformedEvidence(f'coefficient shapes {A_.shape} vs {B_.shape} for {len(ids)} cases')   # audits 6 #3, 7 #2
        return [_safe_max([float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[j] for k in range(1, 6)]) for i, j in enumerate(ids)]
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
                knx = f'W{w_}|gl80|{spec}'
                cs_ = list(range(64)) if not str(J[0][0]['config'].get('attempt', '')).startswith('smk') else x['cases']
                if any([int(v) for v in Wz['cases|' + kk_]] != cs_ for kk_ in (k_, kad, knx)) or list(x['cases']) != cs_ \
                        or len(x['dist_adt']) != len(cs_) or len(x['valid']) != len(cs_):
                    rep['integrity'] = False                                    # mappings / lengths (code audit 4 #7)
                    qs.append([False] * len(x['cases']))
                    continue
                ktt = f"W{w_}|gl64|ad|LSPG|{sp_[1]}|{float(sp_[2]):.8g}|tighter"
                if list(x['tighter_cases']) != list(range(8)) or len(x['adt_tighter_valid']) != 8 or \
                        [int(v) for v in Wz['cases|' + ktt]] != list(range(8)):
                    rep['integrity'] = False                                    # registered tighter population 0-7
                    qs.append([False] * len(x['cases']))
                    continue
                try:
                    d_adt = dist3(f'W{w_}', Wz[k_], Wz[kad], cs_)
                    d_nx = dist3(f'W{w_}', Wz[k_], Wz[knx], cs_)
                    d_tt = dist3(f'W{w_}', Wz[kad][:8], Wz[ktt], list(range(8)))
                except MalformedEvidence:
                    rep['integrity'] = False                                    # code audit 7 #2
                    qs.append([False] * len(x['cases']))
                    continue
                res_ = J[0][0] if any(st_['name'] == f'W{w_}' for st_ in J[0][0]['config']['settings']) else J[1][0]
                cr = [r_ for r_ in res_['rows'] if r_.get('tag') == f'W{w_}|gl64' and r_.get('spec') == spec and 'gam_cert' in r_]
                S_exp = int(round(.25 / float(sp_[2])))
                ok_c = sorted(int(r_['case']) for r_ in cr) == cs_ and all(len(r_['gam_cert']) == S_exp for r_ in cr)
                gam = [E.num(v) for r_ in cr for v in r_['gam_cert']]
                nonst = (sum(1 for v in gam if not (np.isfinite(v) and v <= E.GAMMA)) / len(gam)) if (ok_c and gam) else None
                if nonst is None:
                    rep['integrity'] = False
                agree_ = lambda o, j: ((not np.isfinite(o)) and not np.isfinite(E.num(j))) or \
                    (np.isfinite(E.num(j)) and abs(o - E.num(j)) <= 1e-9 * abs(E.num(j)) + 1e-13)
                if any(not agree_(o, j) for o, j in zip(d_adt, x['dist_adt'])) or \
                        any(not agree_(o, j) for o, j in zip(d_tt, x['adt_tight_tighter'])) or \
                        any(not agree_(o, j) for o, j in zip(d_nx, x['dist_next'])) or \
                        len(x['adt_tight_tighter']) != 8 or len(x['dist_next']) != len(cs_):
                    rep['integrity'] = False
                qs.append(case_q(x, lad[f'W{w_}|gl80|{spec}'], d_adt, d_nx, d_tt, nonst))
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
    QUAL = 'target unverified (diagnostic pick, A2.5)'
    for slot in sel['slots']:
        if slot['rule']:
            k = f"W{slot['width']}|{slot['rule']}|{slot['spec']}"
            cell = sel['cells'][f"W{slot['width']}|{slot['spec']}|{slot['family']}"][slot['rule']]
            e = list(E_(k).values()) if k in scores else []
            diag = slot['outcome'] != 'selected'            # A2.5 (job audit e2q-1 #4): own-family qualification, labelled
            ep_ = [E.num(v_) for v_ in EP(k).values()] if k in scores else []
            rep['configurations'][f"slot|W{slot['width']}|{slot['spec']}|{slot['family']}"] = dict(
                rule=slot['rule'], qualified=cell['own'] if diag else cell['qualified'],
                median_e=float(np.median(e)) if e else None, worst_e=float(np.max(e)) if e else None,
                eps=float(np.median(ep_)) if ep_ else None,
                **({'qualifier': QUAL} if diag else {}))
    # A1.9 selected-rule H-acc-r rows (job audit e2q-1 #5): per spec and family, widths 256 -> 512 -> 768 -> 1024 on each
    # width's realised rule; per-case qualification = the slot's (all-case) qualification; no rule -> unavailable
    slot_of = {(s_['width'], s_['spec'], s_['family']): s_ for s_ in sel['slots']}
    for spec in sorted({s_['spec'] for s_ in sel['slots']}):
        for fam in ('gauss', 'lattice'):
            out = {}
            for lo, hi in ((256, 512), (512, 768), (768, 1024)):
                sa, sb = slot_of.get((lo, spec, fam)), slot_of.get((hi, spec, fam))
                unv_ = any(s_ is not None and s_['outcome'] != 'selected' and s_.get('target_certified') is not True
                           for s_ in (sa, sb))                         # a target-unverified slot on either side (A2.5)
                if not (sa and sb and sa['rule'] and sb['rule']):
                    out[f'{lo}->{hi}'] = dict(outcome='unavailable (no rule)', **({'qualifier': QUAL} if unv_ else {}))
                    continue
                ka, kb = f"W{lo}|{sa['rule']}|{spec}", f"W{hi}|{sb['rule']}|{spec}"
                ca = rep['configurations'][f'slot|W{lo}|{spec}|{fam}']
                cb = rep['configurations'][f'slot|W{hi}|{spec}|{fam}']
                if ka not in scores or kb not in scores or ca['qualified'] is None or cb['qualified'] is None:
                    out[f'{lo}->{hi}'] = dict(outcome='not reached (missing)', **({'qualifier': QUAL} if unv_ else {}))
                    continue
                ea, eb, ela, elb, ep = E_(ka), E_(kb), EL(ka), EL(kb), EP(ka)
                cs = [c for c in sorted(ea) if c in eb and ela[c] and elb[c]]
                tr = S.h_acc_r_transition([eb[c] - ea[c] for c in cs], [ep[c] for c in cs],
                                          [bool(ca['qualified'])] * len(cs), [bool(cb['qualified'])] * len(cs))
                tr['median_paired_change'] = float(np.median([eb[c] - ea[c] for c in cs])) if cs else None
                tr['rules'] = [sa['rule'], sb['rule']]
                if 'qualifier' in ca or 'qualifier' in cb or unv_:
                    tr['qualifier'] = QUAL
                out[f'{lo}->{hi}'] = tr
            out['range_256_1024'] = S.h_acc_r_range([v['outcome'] for v in out.values() if isinstance(v, dict) and 'outcome' in v])
            rep['h_acc_r'][f'sel-{fam}|{spec}'] = out
            if any(isinstance(v, dict) and v.get('qualifier') for v in out.values()):   # the aggregate keeps the A2.5 label
                rep.setdefault('h_acc_r_qualifiers', {})[f'sel-{fam}|{spec}|range_256_1024'] = QUAL     # (re-audit e2q-2 #2)
    for f in FIXED:
        st, rl, *sp = f.split('|')
        spec = '|'.join(sp)
        fam = 'gauss' if rl.startswith('gl') else 'lattice'
        cell = sel['cells'].get(f'{st}|{spec}|{fam}', {}).get(rl)
        k = f'{st}|{rl}|{spec}'
        e = list(E_(k).values()) if k in scores else []
        ep_ = [E.num(v_) for v_ in EP(k).values()] if k in scores else []
        rep['configurations'][f'fixed|{f}'] = dict(rule=rl, qualified=(cell or {}).get('own'),
                                                   median_e=float(np.median(e)) if e else None, worst_e=float(np.max(e)) if e else None,
                                                   eps=float(np.median(ep_)) if ep_ else None)
    # front
    if a.front:
        fres = json.loads((Path(a.front) / 'result.json').read_text())
        bank_of = lambda c: (sorted({s_['bank'] for s_ in c['settings']}), sorted((c.get('bank_registry') or {}).items()))
        if bank_of(fres['config']) != bank_of(J[0][0]['config']):          # code audit 3 #23: same bank identity
            fres = dict(fres, complete=False)
        check_bank3d(fres['config'], b)
        import hashlib as _h
        if fres['config'].get('attempt') != f'e2f{b}' or (fres['config'].get('selection') or {}).get('file') != f'checks/select3d-{b}.json':
            raise SystemExit('front job is not the registered e2f job of this selection')
        tm = next(iter(fres['timing'].values()))
        ba = {}
        if tm.get('valid') is True and fres.get('complete') and fres['backend'] == 'gpu':     # code audit 2 #23: d3run's
            per = {}                                                                          # panel acceptance governs
            seen, bad = set(), set()
            for inv in tm['invocations']:                               # vetoes over ALL records first (code audit 5 #4)
                k_ = (inv['B'], inv['case'], inv['rep'])
                if k_ in seen or inv['rep'] not in range(3) or inv['case'] not in range(8) or not inv.get('hashes_ok'):
                    bad.add(inv['B'])                                   # duplicate / unregistered key / identity failure
                seen.add(k_)
            for inv in tm['invocations']:
                if inv['B'] in bad:
                    continue
                if 1 / 1.1 <= inv['drift'] <= 1.1 and np.isfinite(inv['ratio']) and inv['ratio'] > 0:
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
        # A17: qualify's registered dominance (qstats.dominates) on separate median / worst fronts among qualified points
        import qstats as QS
        q = {k: v for k, v in pts.items() if v['qualified'] is True and v['cost'] is not None and v['median_e'] is not None
             and v.get('eps') is not None}
        for stat, key in (('median_e', 'state'), ('worst_e', 'state_worst')):
            P_ = {k: dict(c=v['cost'], e=v[stat], eps=v['eps']) for k, v in q.items()}
            for k, v in pts.items():
                if v['qualified'] is None or v['cost'] is None or v['median_e'] is None or v.get('eps') is None:
                    v[key] = 'missing'
                elif v['qualified'] is False:
                    v[key] = 'unqualified'
                else:
                    me = dict(c=v['cost'], e=v[stat], eps=v['eps'])
                    v[key] = 'dominated' if any(QS.dominates(o, me) for kk, o in P_.items() if kk != k) else 'member'
        for slot in sel['slots']:
            k = f"slot|W{slot['width']}|{slot['spec']}|{slot['family']}"
            st_ = 'absent' if (slot['outcome'] == 'none within ladder' and slot.get('target_certified')) else 'missing'
            pts.setdefault(k, dict(state=st_, state_worst=st_))                # target-unverified search: not definitive
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
