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
  deployed family = smaller sum of the two arms' median paired ratios in the family panels (A = r_inf fixed BE dt0; panels
  must be valid); ties -> smaller m; fallbacks = the next two larger rungs of the deployed family, then 'unavailable'.
A setting without a converged rule is 'unavailable' (no selection; the diagnostic-baseline ladder is reported only).
Writes checks/SELECTION-3D.json (or --out). Smoke inputs are refused unless --allow-smoke.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out')
    ap.add_argument('--allow-smoke', action='store_true')
    a = ap.parse_args()
    import c3common as C3
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    i65 = C3.lattice65_index(65)
    out = dict(tau=TAU, families=FAMILIES, control=CONTROL, selection_arms=SEL, inputs=[], settings={})
    for rp in map(Path, a.runs):
        res = json.loads((rp / 'result.json').read_text())
        if res['config'].get('smoke') and not a.allow_smoke:
            raise SystemExit(f'{rp}: smoke result refused')
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
            base = esc['converged'] or S['ladder_baseline']['rule']
            bk = C3.load_bank(s['bank'])
            G = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
            elig = dict(S.get('escalation_rule_eligibility', {}), **S.get('ladder_eligibility', {}))

            def dist(r1, r2, sk):
                A_, B_ = Wz[f'{sname}|{r1}|{sk}'], Wz[f'{sname}|{r2}|{sk}']
                d = np.linalg.norm(np.einsum('cki,ip->ckp', A_ - B_, G), axis=2) / n0[:, None]
                per = d[:, 1:].max(1)
                return float(per.max()) if np.isfinite(per).all() else float('nan')
            rules = {}
            for rule in FAMILIES['lattice'] + FAMILIES['gauss'] + [CONTROL]:
                if f'{sname}|{rule}|{SEL[0]}' not in Wz.files:
                    continue
                rr = {}
                for sk in SEL:
                    e = elig.get(rule, {}).get(sk, {})
                    dd = dist(rule, base, sk) if rule != base else 0.
                    rr[sk] = dict(eligible=bool(e.get('eligible')), solver_distance=e.get('solver_distance'),
                                  nonstationary=e.get('nonstationary'), steps=e.get('steps'), dist_to_conv=dd,
                                  within_tau=bool(np.isfinite(dd) and dd <= TAU))
                rr['qualifies'] = all(rr[sk]['eligible'] and rr[sk]['within_tau'] for sk in SEL)
                rules[rule] = rr
            ent['rules'] = rules
            ent['ladder_baseline'] = S['ladder_baseline']
            if esc['converged'] is None:
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
                want = {f'{sname}|{r}|{sk}' for sk in SEL}
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
    dest = Path(a.out or HERE / 'checks' / 'SELECTION-3D.json')
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    print(json.dumps({k: (v.get('status'), v.get('converged'), v.get('mstar'), v.get('deployed')) for k, v in out['settings'].items()}))


if __name__ == '__main__':
    main()
