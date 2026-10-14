"""3D rule selection from `c3desc*` escalation results (DESIGN 3.3 as amended by A1.7, A2.4, A2.5, A6.2, A6.3, A7.1).

Per setting:
  converged rule r_inf and the escalation statuses (from the job, where the rule was decided in-job);
  for every ladder rule and both selection arms (fixed-sweep BE dt0 K=1; fixed-sweep LSPG-CN 2.5 dt0 K=2): A6.2
  eligibility (saved by the job) and the worst distance to the r_inf run of the same arm, RECOMPUTED here independently
  by decoding the coefficients through the bank at the 63^3 lattice coordinates (max over evolved times, relative to the
  physical ||u0|| at the lattice);
  m* per family (lattice: lat8192 ... lat131072; Gauss: gl24 ... gl48) = smallest rule with both arms eligible and worst
  distance <= tau = 2.5e-4 on every case; the control lat256 must NOT qualify (else the selection is withheld:
  non-discriminating);
  deployed family = smaller sum of the two timed fixed-sweep arms' median times tB in the family panels (A1.7 as amended:
  raw median times, not ratios; the panels' A subject is the job's ladder baseline, i.e. the converged rule's fixed BE
  dt0, or the finest escalation rule run when the job found no converged rule; panels must be valid and complete);
  ties -> smaller m; fallbacks = the next two larger rungs of the deployed family, then 'unavailable'.
A setting without a converged rule is 'unavailable' (no selection; the diagnostic-baseline ladder is reported only).
Writes checks/SELECTION-3D.json (or --out). Smoke inputs are refused unless --allow-smoke.

--posthoc (DESIGN A10, POST HOC, labelled as such everywhere downstream): the second selection arm is the verified
adaptive tight LSPG-CN at 2.5 dt0 (`ad|LSPG|CN|0.025|tight`, eligible iff every run of the rule is valid, A5.1) in place
of the fixed-sweep LSPG-CN K=2 arm; the converged rule is recomputed from the saved per-arm escalation statuses of the
fixed-sweep BE arm and that adaptive CN arm (same A2.4 order: gl56 if (gl56, gl64) passes both, else gl64 if (gl64, gl80)
passes both, else none). Everything else (tau, ladders, control, cost = sum of the timed fixed-sweep arms' median times,
fallbacks) is unchanged. Writes checks/SELECTION-3D-POSTHOC.json by default.

    python select3d.py --runs runs/c3descW1024/archive/output runs/c3descW768/archive/output runs/c3descB/archive/output
"""
from __future__ import annotations

import offline  # noqa: F401  (A8.1: CPU, highest, x64 - before any JAX import)

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAU = 2.5e-4
FAMILIES = dict(lattice=['lat8192', 'lat16384', 'lat32768', 'lat65536', 'lat131072'], gauss=['gl24', 'gl32', 'gl40', 'gl48'])
CONTROL = 'lat256'
SEL = ['fs|BE|0.01|K1', 'fs|CN|0.025|K2']
SEL_POSTHOC = ['fs|BE|0.01|K1', 'ad|LSPG|CN|0.025|tight']
TIMED = ['fs|BE|0.01|K1', 'fs|CN|0.025|K2']                 # the arms timed in the family panels (both modes)


def ids_of(res, Wz, sname, rule, sk):
    """Case ids of the rows of W_esc[sname|rule|sk]: the saved ids when present (jobs after A8.2), else the order in which
    the job appended its result rows for that run (rows and W rows are written in the same case loop, c3run.rollouts);
    the row count must equal the array's first dimension."""
    k = f'{sname}|{rule}|{sk}'
    if 'cases|' + k in Wz.files:
        return [int(x) for x in Wz['cases|' + k]]
    ids = [int(r['case']) for r in res['rows'] if r['setting'] == sname and r['rule'] == rule and r['spec'] == sk]
    n = Wz[k].shape[0]
    # a rule that is both an escalation and a ladder rule (gl48) is run twice; the saved W is the LAST run, and every
    # block of rows must have the same case order
    assert n and len(ids) % n == 0 and len(set(ids[-n:])) == n, k
    assert all(ids[i:i + n] == ids[-n:] for i in range(0, len(ids), n)), k
    return ids[-n:]


def posthoc_converged(pairs):
    """A10.2: A2.4 order on the fixed-sweep BE status and the adaptive CN status of each escalation pair."""
    def st(p):
        if p not in pairs:
            return 'absent'
        v = pairs[p]
        a, b = v['selection'][SEL_POSTHOC[0]]['status'], v['diagnostic'][SEL_POSTHOC[1]]['status']
        return 'unresolved' if 'unresolved' in (a, b) else ('pass' if a == b == 'pass' else 'fail')
    sts = {p: st(p) for p in ('gl48|gl56', 'gl56|gl64', 'gl64|gl80') if p in pairs}
    conv = 'gl56' if sts.get('gl56|gl64') == 'pass' else ('gl64' if sts.get('gl64|gl80') == 'pass' else None)
    return conv, sts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out')
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--posthoc', action='store_true', help='DESIGN A10 post-hoc selection (adaptive CN arm)')
    ap.add_argument('--ext', nargs='*', default=[], help='DESIGN A11 escalation-extension archives (with --posthoc only)')
    a = ap.parse_args()
    assert a.posthoc or not a.ext, 'the A11 extension is part of the post-hoc selection only'
    EXT = {}
    for xp in map(Path, a.ext):
        xr = json.loads((xp / 'result.json').read_text())
        if xr['config'].get('smoke') and not a.allow_smoke:
            raise SystemExit(f'{xp}: smoke result refused')
        assert xr.get('complete') and xr['mode'] == 'esc', xp
        for s_ in xr['config']['settings']:
            assert s_.get('ext_rules') and s_['name'] not in EXT, s_['name']
            EXT[s_['name']] = (xp, xr, np.load(xp / 'W_esc.npz'))
    sel = SEL_POSTHOC if a.posthoc else SEL
    import c3common as C3
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    i65 = C3.lattice65_index(65)
    out = dict(smoke_inputs=False, tau=TAU, families=FAMILIES, control=CONTROL, selection_arms=sel, timed_arms=TIMED,
               posthoc=bool(a.posthoc),
               label=('POST HOC (DESIGN A10' + (', A11 extension' if EXT else '') + ')') if a.posthoc else 'pre-registered',
               extension_inputs=[dict(path=str(v[0]), job_id=v[1].get('job_id'), gpu=v[1].get('gpu'),
                                      result_sha256=hashlib.sha256((v[0] / 'result.json').read_bytes()).hexdigest())
                                 for v in EXT.values()],
               inputs=[], settings={})
    for rp in map(Path, a.runs):
        res = json.loads((rp / 'result.json').read_text())
        if res['config'].get('smoke') and not a.allow_smoke:
            raise SystemExit(f'{rp}: smoke result refused')
        out['smoke_inputs'] = out['smoke_inputs'] or bool(res['config'].get('smoke'))
        assert res.get('complete') and res['mode'] == 'esc', rp
        out['inputs'].append(dict(path=str(rp), job_id=res.get('job_id'), gpu=res.get('gpu'),
                                  result_sha256=hashlib.sha256((rp / 'result.json').read_bytes()).hexdigest()))
        cfg = res['config']
        cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
        tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
        n0 = np.array([float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65])) for j in cases])
        Wz = np.load(rp / 'W_esc.npz')
        for s in cfg['settings']:
            sname = s['name']
            S = res['settings'][sname]
            esc = S['escalation']
            ent = dict(bank=s['bank'], Rp=s['Rp'], escalation_status=esc['status'], converged=esc['converged'],
                       pairs={k: dict(selection=v['selection_pair_status'], adaptive=v['adaptive_diagnostic_status'],
                                      worst={kk: vv['worst'] for kk, vv in {**v['selection'], **v['diagnostic']}.items()})
                              for k, v in esc['pairs'].items()},
                       operator_check=esc['operator_check'], fixed_vs_adaptive={k: v['worst'] for k, v in esc['fixed_vs_adaptive'].items()})
            conv = esc['converged']
            if a.posthoc:
                conv, ph = posthoc_converged(esc['pairs'])
                ent.update(posthoc_pair_status=ph, posthoc_converged=conv)
            bk = C3.load_bank(s['bank'])
            G = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
            src = {}                                   # rule -> (result, W archive) holding its trajectories

            def W_of(rule, sk):
                r_, z_ = src.get(rule, (res, Wz))
                assert ids_of(r_, z_, sname, rule, sk) == list(cases), (sname, rule, sk)
                return z_[f'{sname}|{rule}|{sk}']
            if a.posthoc and conv is None and sname in EXT:
                # A11: extension pairs in the A2.4 order continued (gl80 if (gl80, gl96) passes the A10 axis, else gl96 if
                # (gl96, gl112) passes). Accepted only if (codex-a11-1 items 6, 7): the numerical contract matches this
                # job's setting; the population is this job's (production: all 64 cases of 923801); every arm's gl80
                # trajectories reproduce this job's to <= 1e-10 (all four arms, complete); and every extension pair status
                # is RECOMPUTED here from the trajectories, the rows and this job's bar and agrees with the job's.
                xp, xr, xz = EXT[sname]
                xc = xr['config']
                xs_cfg = next(t for t in xc['settings'] if t['name'] == sname)
                for k_ in ('bank', 'Rp', 'a'):
                    assert xs_cfg[k_] == s[k_], ('extension contract', k_)
                assert sname == 'W1024', 'A11 extends the W1024 escalation only'
                for k_ in ('mesh', 'gtol', 'trust_fraction', 'conv_bar', 'cohort_seed', 'cohort_count'):
                    assert xc[k_] == cfg[k_], ('extension contract', k_)
                # reference identity by content (the staged path is job-specific), cohort table by hash
                assert {k_: xr['reference'][k_] for k_ in ('sha256', 'n', 'dt')} == \
                    {k_: res['reference'][k_] for k_ in ('sha256', 'n', 'dt')}, 'extension reference differs'
                # the in-job cohort-table sha256 is NOT node-stable: its peak normaliser s_star is computed on the GPU and
                # differs in the last bit between H200 nodes (observed 2026-10-13: two hashes among the 3D jobs, one case's
                # initial coefficients differing by 1 ulp). Bind seed and count, and require the initial coefficients of
                # every case to agree to 1e-12 relative instead (offline scoring checks the CPU table hash, score.py).
                assert {k_: xr['cohort'][k_] for k_ in ('seed', 'count')} == {k_: res['cohort'][k_] for k_ in ('seed', 'count')}
                w0x, w0r = xz[f'w0|{sname}'], Wz[f'w0|{sname}']
                assert w0x.shape == w0r.shape and float(np.max(np.max(np.abs(w0x - w0r), axis=1) /
                                                               np.max(np.abs(w0r), axis=1))) <= 1e-12, \
                    'extension initial coefficients differ'                          # per case
                assert (xc.get('case_subset') or list(range(xc['cohort_count']))) == list(cases)
                assert bool(xc.get('smoke')) == bool(cfg.get('smoke')), 'smoke and production archives cannot be mixed'
                if not cfg.get('smoke'):
                    assert cfg['cohort_seed'] == 923801 and list(cases) == list(range(64)), 'production population'
                xs = xr['settings'][sname]['extension']
                ARMS = SEL + ['ad|LSPG|BE|0.01|tight', SEL_POSTHOC[1]]

                def xW(rule, sk):
                    assert ids_of(xr, xz, sname, rule, sk) == list(cases), ('extension case ids', rule, sk)
                    w_ = xz[f'{sname}|{rule}|{sk}']
                    assert w_.shape == (len(cases), 6, int(s['Rp'])), ('extension shape', rule, sk, w_.shape)
                    return w_

                def fd(A_, B_):
                    d_ = np.linalg.norm(np.einsum('cki,ip->ckp', A_ - B_, G), axis=2) / n0[:, None]
                    per_ = d_[:, 1:].max(1)
                    return float(per_.max()) if np.isfinite(per_).all() else float('nan')
                repro = {}
                for sk in ARMS:
                    A_, B_ = xW('gl80', sk), W_of('gl80', sk)
                    assert A_.shape == B_.shape == (len(cases), 6, int(s['Rp'])), sk
                    repro[sk] = fd(A_, B_)
                rep_ok = all(np.isfinite(v) and v <= 1e-10 for v in repro.values()) and len(repro) == len(ARMS)

                def xelig(rule, sk):
                    rows_ = [r for r in xr['rows'] if r['setting'] == sname and r['rule'] == rule and r['spec'] == sk]
                    ok_ = sorted(r['case'] for r in rows_) == sorted(cases) and all(r['valid'] is True for r in rows_)
                    if sk.startswith('fs|'):
                        adx = 'ad|LSPG|BE|0.01|tight' if sk == SEL[0] else SEL_POSTHOC[1]
                        ok_ = ok_ and xelig(rule, adx) and fd(xW(rule, sk), xW(rule, adx)) <= 2.5e-4
                    return bool(ok_)
                recomputed, agree = {}, True
                for pk, pv in xs['pairs'].items():
                    r1, r2 = pk.split('|')
                    sts = []
                    for sk in (SEL[0], SEL_POSTHOC[1]):
                        w_ = fd(xW(r1, sk), xW(r2, sk))
                        sts.append('unresolved' if not (xelig(r1, sk) and xelig(r2, sk)) or not np.isfinite(w_)
                                   else ('pass' if w_ <= cfg['conv_bar'] else 'fail'))
                    st_ = 'unresolved' if 'unresolved' in sts else ('pass' if all(x_ == 'pass' for x_ in sts) else 'fail')
                    recomputed[pk] = st_
                    agree = agree and st_ == pv['posthoc_status']
                xconv = None
                if rep_ok and agree:
                    xconv = 'gl80' if recomputed.get('gl80|gl96') == 'pass' else \
                        ('gl96' if recomputed.get('gl96|gl112') == 'pass' else None)
                ent['extension'] = dict(path=str(xp), job_id=xr.get('job_id'), pair_status_job=
                                        {k: v['posthoc_status'] for k, v in xs['pairs'].items()},
                                        pair_status_recomputed=recomputed, statuses_agree=agree,
                                        reproduction_gl80=repro, reproduction_passed=rep_ok, converged=xconv,
                                        worst={k: {kk: vv['worst'] for kk, vv in v['posthoc'].items()} for k, v in xs['pairs'].items()})
                conv = xconv
                ent['posthoc_converged'] = conv
                for r_ in xs['rules']:
                    if r_ != 'gl80':
                        src[r_] = (xr, xz)
            base = conv or S['ladder_baseline']['rule']
            elig = dict(S.get('escalation_rule_eligibility', {}), **S.get('ladder_eligibility', {}))
            if a.posthoc:                     # adaptive arm: eligible iff every run of the rule is valid (A5.1), from the rows
                rows = [r for r in res['rows'] if r['setting'] == sname and r['spec'] == SEL_POSTHOC[1]]
                for rule in {r['rule'] for r in rows}:
                    rr_ = [r for r in rows if r['rule'] == rule]
                    nb = len(rr_) // max(len(cases), 1)          # repeated runs (gl48: escalation + ladder) all count
                    ok = (nb >= 1 and sorted(r['case'] for r in rr_) == sorted(cases * nb)
                          and all(r['valid'] is True for r in rr_))
                    elig.setdefault(rule, {})[SEL_POSTHOC[1]] = dict(eligible=bool(ok), runs=len(rr_),
                                                                      invalid=sum(r['valid'] is not True for r in rr_))

            def dist(r1, r2, sk):
                A_, B_ = W_of(r1, sk), W_of(r2, sk)     # A8.2: joined by case ids (saved ids, else the job's row order)
                d = np.linalg.norm(np.einsum('cki,ip->ckp', A_ - B_, G), axis=2) / n0[:, None]
                per = d[:, 1:].max(1)
                return float(per.max()) if np.isfinite(per).all() else float('nan')
            rules = {}
            for rule in FAMILIES['lattice'] + FAMILIES['gauss'] + [CONTROL]:
                if any(f'{sname}|{rule}|{sk}' not in Wz.files for sk in sel):
                    continue
                rr = {}
                for sk in sel:
                    e = elig.get(rule, {}).get(sk, {})
                    dd = dist(rule, base, sk) if rule != base else 0.
                    rr[sk] = dict(eligible=bool(e.get('eligible')), solver_distance=e.get('solver_distance'),
                                  nonstationary=e.get('nonstationary'), steps=e.get('steps'), dist_to_conv=dd,
                                  within_tau=bool(np.isfinite(dd) and dd <= TAU))
                rr['qualifies'] = all(rr[sk]['eligible'] and rr[sk]['within_tau'] for sk in sel)
                rules[rule] = rr
            ent['rules'] = rules
            ent['ladder_baseline'] = S['ladder_baseline']
            if conv is None:
                ent.update(status='unavailable (no converged rule)', deployed=None)
                out['settings'][sname] = ent
                continue
            absent = [r for r in FAMILIES['lattice'] + FAMILIES['gauss'] if r not in rules and r != base]
            if absent:
                ent.update(status=f'unavailable (ladder rules not evaluated: {absent})', deployed=None)
                out['settings'][sname] = ent
                continue
            if CONTROL not in rules:
                ent.update(status='unavailable (control lat256 not evaluated)', deployed=None)
                out['settings'][sname] = ent
                continue
            ctrl_q = rules[CONTROL]['qualifies']
            ent['control_qualifies'] = ctrl_q
            mstar = {}
            for fam, lad in FAMILIES.items():
                q = [r for r in lad if rules.get(r, {}).get('qualifies')]
                mstar[fam] = q[0] if q else None
            ent['mstar'] = mstar
            if ctrl_q:
                ent.update(status='withheld (control lat256 qualifies: non-discriminating)', deployed=None)
                out['settings'][sname] = ent
                continue
            cost = {}
            for fam, r in mstar.items():
                if r is None:
                    continue
                p = res['timing'].get(f'family|{sname}|{r}')
                if not p or not p.get('valid'):
                    cost[fam] = None
                    continue
                med, slots = {}, set()
                for inv in p['invocations']:
                    med.setdefault(inv['B'], []).append(inv['tB'])               # A1.7: sum of median TIMES
                    slots.add((inv['B'], inv['case'], inv['rep']))
                want = {f'{sname}|{r}|{sk}' for sk in TIMED}
                expected = {(b, c, rp) for b in want for c in p['cases'] for rp in range(int(p['reps']))}
                if set(med) != want or slots != expected or len(p['invocations']) != len(expected):
                    cost[fam] = None                                                # incomplete panel -> unavailable
                    continue
                cost[fam] = float(sum(np.median(v) for v in med.values()))
            ent['family_cost'] = cost
            cands = [(c_, __import__('inventory').rule_points(mstar[f]), f) for f, c_ in cost.items() if c_ is not None]
            if any(mstar[f] is not None and cost.get(f) is None for f in mstar):
                cands = []                                                          # a qualifying family lacks evidence
            if not cands:
                ent.update(status='unavailable (no qualifying rule with a valid panel)', deployed=None)
            else:
                _, _, fam = min(cands)
                lad = FAMILIES[fam]
                i = lad.index(mstar[fam])
                ent.update(status='selected', deployed=mstar[fam], deployed_family=fam,
                           fallbacks=lad[i + 1:i + 3] + ['unavailable'])
            out['settings'][sname] = ent
    dest = Path(a.out or HERE / 'checks' / ('SELECTION-3D-POSTHOC.json' if a.posthoc else 'SELECTION-3D.json'))
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    print(json.dumps({k: (v.get('status'), v.get('converged'), v.get('posthoc_converged'), v.get('mstar'), v.get('deployed'))
                      for k, v in out['settings'].items()}))


if __name__ == '__main__':
    main()
