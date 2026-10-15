"""jcp-speed3d driver (DESIGN §3–§9, A1, A2, A3). Modes (one per job kind):

  grid  routes 1+2 on the validation cohort (g1, g2, g3): per setting and rule (deployed + reduced) the verified adaptive
        solves ADT (tight on every case; tighter on cases 0-7 at the deployed rule, A1.4), every solver spec of the plan,
        certification of every fixed-sweep / QN run (exact Step 2 ratio per step), lattice distance to ADT, the H1.1 replay
        of the failing FS-K2 CN 2.5 dt0 cases (deployed rule, A2.2), controls K1-K3, and one screening A-B-A panel per
        setting with A = C_old (M2_512, gl32, FS-K1 BE dt0) and the T1 control.
  fom   (f1) FOM ladder accuracy on the validation cohort (s3modes.mode_fom)
  prof  (p1) profile, mesh-free qualification, batching, confirmation panels, quadrature sensitivity (s3modes.mode_prof)
  mesh  (m1) ROM at meshes n = L+1 and ROM vs FOM timing per mesh (s3modes.mode_mesh)
  rep   (r1) replication on 923901 after the freeze (s3modes.mode_rep)
Every ROM run is a single query (batch 1) except in the batching study. Saved: result.json, W_<mode>.npz.

    python s3run.py --config configs/<attempt>.json --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import c3common as C3
import replay as RP
import s3common as S
from s3ctx import Ctx, COLD, COLD_SPEC, DT0

SCHEMES = ('BE', 'CN', 'BDF2')


def grid_plan(s, rule, cfg):
    """(adt specs, solver specs) for one setting and rule (DESIGN §2, §3, H2; A1.10 controls on the deployed rule)."""
    dts = cfg.get('dts', S.DTS)
    sv = []
    if rule == s['rule']:
        for dt in dts:
            for sch in SCHEMES:
                sv.append(('fs', sch, dt, 1))
                sv.append(('fsp', sch, dt, 1))
                for kr in (1, 4, 0):
                    sv.append(('qn', sch, dt, kr))
            for sch in ('CN', 'BDF2'):
                sv.append(('fs', sch, dt, 2))
        sv += [('qn', 'CN', .025, 1, 'q0'), ('qn', 'CN', .025, 1, 'noJuT'), ('qn', 'BE', DT0, 1, 'noJuT')]   # K2, K3
    else:                                     # route 2 reduced rules: FS-K1, FS-K2, QN-4; CN, BDF2 (+ BE)
        for dt in dts:
            for sch in SCHEMES:
                sv.append(('fs', sch, dt, 1))
                sv.append(('qn', sch, dt, 4))
            for sch in ('CN', 'BDF2'):
                sv.append(('fs', sch, dt, 2))
    adt = sorted({('ad', sp[1], sp[2], 'tight') for sp in sv}, key=S.spec_key)
    return adt, sv


def inventory(cfg):
    inv = dict(mode=cfg['mode'], settings={})
    nc = len(cfg.get('case_subset') or range(cfg['cohort_count']))
    tc = cfg['timing']
    if cfg['mode'] == 'grid':
        for s in cfg['settings']:
            runs, nsub = [], 0
            for rule in [s['rule']] + s.get('reduced', []):
                adt, sv = grid_plan(s, rule, cfg)
                runs += [(rule, S.spec_key(x)) for x in adt + sv]
                nsub += len([x for x in sv if len(x) <= 4])
            inv['settings'][s['name']] = dict(runs=runs, n_runs=len(runs), case_runs=len(runs) * nc + 8 * len(adt),
                                              timing_invocations=3 * (nsub + 1) * tc['cases'] * tc['reps'])
    else:
        import s3modes
        inv.update(s3modes.inventory(cfg))
    return inv


def mode_grid(cx):
    cfg, tc = cx.cfg, cx.cfg['timing']
    cases = cx.cases
    cold_tag = f"{COLD['name']}|{COLD['rule']}"
    if COLD['name'] not in [s['name'] for s in cfg['settings']]:
        cx.rollouts(COLD['name'], COLD['rule'], [COLD_SPEC], cases[:tc['cases']])
    for k_set, s in enumerate(cfg['settings']):
        sname = s['name']
        st = cx.ST[sname]
        S_rep = cx.rep['settings'][sname] = dict(bank=s['bank'], Rp=st.Rp, M=st.M, trust=st.trust, a=st.a, rules={})
        subjects = []
        for rule in [s['rule']] + s.get('reduced', []):
            adt, sv = grid_plan(s, rule, cfg)
            cond = st.mf_cond(rule)
            R_rep = S_rep['rules'][rule] = dict(m=cond['m'], mf_gram_cond=cond['cond'])
            ra = cx.rollouts(sname, rule, adt, cases)
            if rule == s['rule']:                    # A1.4 ADT tight vs tighter on cases 0-7
                tt = [('ad', x[1], x[2], 'tighter') for x in adt]
                rt = cx.rollouts(sname, rule, tt, cases[:8])
                R_rep['adt_tight_tighter'] = {
                    S.spec_key(x): dict(worst=float(cx.dist_cases(sname, ra[S.spec_key(x)]['W'][:8], rt[S.spec_key(y)]['W'],
                                                                  cases[:8])[:, 1:].max()),
                                        tighter_valid=all(rt[S.spec_key(y)]['valid']))
                    for x, y in zip(adt, tt)}
            k25 = S.spec_key(('fs', 'CN', .025, 2))
            rs = cx.rollouts(sname, rule, sv, cases, keep_wall={k25} if rule == s['rule'] else set())
            dist = {}
            for sp in sv:
                key = S.spec_key(sp)
                ak = S.spec_key(('ad', sp[1], sp[2], 'tight'))
                dist[key] = cx.dist_cases(sname, rs[key]['W'], ra[ak]['W'], cases).tolist()
            R_rep['dist_to_adt'] = dist
            R_rep['adt_valid'] = {k: v['valid'] for k, v in ra.items()}
            if rule == s['rule'] and k25 in rs:       # H1.1 replay (A2.2)
                d = st.data(rule)
                tag = f'{sname}|{rule}'
                rows = {r['case']: r for r in cx.rep['rows'] if r['tag'] == tag and r['spec'] == k25}
                a_eff = C3.fs3.a_effective('CN', .025, DT0, st.a, 10)
                wall = cx.Wstore[f'wall|{tag}|{k25}']
                rp = {}
                for ci, j in enumerate(cases):
                    rr_ = rows[j]
                    dd = np.asarray(dist[k25][ci])
                    if rr_['valid'] and rr_['nonstationary'] == 0 and dd[1:].max() <= 2.5e-4:
                        continue
                    n_star, kind = RP.pick_step(rr_['steps'], rr_['gam_cert'], rr_['valid'], dd, a_eff, 2)
                    if n_star is None:
                        rp[j] = dict(label=kind)
                        continue
                    try:
                        rp[j] = dict(RP.replay_case(st, d, 'CN', .025, float(cx.NU[j][0]), wall[ci], rr_['steps'], n_star),
                                     kind=kind)
                    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
                        rp[j] = dict(label='replay error', error=repr(e))
                R_rep['h11_replay'] = rp
                print('REPLAY', sname, {j: v.get('label') for j, v in rp.items()}, cx.el(), flush=True)
            subjects += [(rule, sp) for sp in sv if len(sp) <= 4]          # controls (mutations) are not timed
            cx.save()
            cx.dump()
        A = cx.rom_subject(COLD['name'], COLD['rule'], COLD_SPEC)
        Bs = [cx.rom_subject(sname, rule, sp) for rule, sp in subjects] + [cx.t1_subject()]
        cx.time_panel(f'screen|{sname}', A, Bs, int(tc['seed']) + k_set, tc['cases'], tc['reps'])
        S_rep['peak_bytes'] = C3.peak_bytes()
        cx.dump()
        cx.save()
    assert cold_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.dry_run:
        print(json.dumps(inventory(cfg), indent=1))
        return
    cx = Ctx(cfg, a.out)
    cx.rep['inventory'] = inventory(cfg)
    if cfg['mode'] == 'grid':
        mode_grid(cx)
    else:
        import s3modes
        getattr(s3modes, f"mode_{cfg['mode']}")(cx)
    cx.rep['peak_bytes_job'] = C3.peak_bytes()
    cx.rep['elapsed_seconds'] = cx.el()
    cx.rep['complete'] = True
    cx.save()
    cx.dump()
    (cx.out / 'COMPLETE').write_text('complete\n')
    print('S3RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
