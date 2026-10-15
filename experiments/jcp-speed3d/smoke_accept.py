"""jcp-speed3d smoke acceptance (DESIGN A4.1; audits code-2 #1, code-3, code-4, code-5). Reads the smoke outputs of THIS
execution and writes <out>/ACCEPTANCE.json (one entry per gate, the overall verdict, the offline and deferred gates).

Principles (audit code-5): no stored boolean is trusted. Every predicate is recomputed from raw evidence with explicit
thresholds; every population (settings, rules, specs, panel subjects, cases, repetitions, FOM ladder, profiles, batch sizes,
meshes) is derived from the committed configs and required EXACTLY (no missing, extra or duplicate entries); every timing is
finite and positive and every ratio / drift is recomputed; every artifact carries this job id and source commit; every
section is evaluated even if another raises (a raising section is a failed gate).

    python smoke_accept.py --out <smoke output dir>
"""
import argparse
import itertools
import json
import os
import traceback
from pathlib import Path

import numpy as np

import analyze as AN
import c3common as C3
import s3common as S
import s3run as SR
import timing as TM

HERE = Path(__file__).resolve().parent
BAR = 2.5e-4
PROFILE_SUBJECTS = ['project', 'evolve', 'decode', 'contraction', 'normal+cholesky', 'qn_iteration', 'qn_step_with_refresh',
                    'dispatch_floor']


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fin(x, n=None, pos=False):
    a = np.asarray(x, float) if x is not None else np.asarray([], float)
    ok = a.size > 0 and np.isfinite(a).all() and (n is None or a.size == n)
    return bool(ok and (not pos or (a > 0).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    O = Path(a.out)
    job, commit = os.environ.get('SLURM_JOB_ID'), os.environ.get('SOURCE_COMMIT')
    G = {}

    def gate(name, ok, **ev):
        G[name] = dict(result='pass' if bool(ok) else 'fail', **ev)

    def section(name, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            gate(f'{name} (section raised)', False, error=traceback.format_exc()[-1500:])

    def ident(obj, what):
        gate(f'{what}: bound to job {job} / commit {commit}', job is not None and commit is not None and obj is not None
             and obj.get('job_id') == job and obj.get('commit') == commit,
             found=None if obj is None else [obj.get('job_id'), obj.get('commit')])

    def panel(name, tm, subjects, cases, reps):
        """Exact A-B-A panel: subjects, (subject, case, rep) product without duplicates, finite positive times, recomputed
        ratio / drift / validity (timing.panel_validity), every hash check true."""
        ok, why = True, []
        if tm is None:
            gate(name, False, why='missing')
            return
        inv = tm.get('invocations', [])
        if sorted(tm.get('subjects', [])) != sorted(subjects):
            ok, why = False, why + ['subjects differ']
        want = set(itertools.product(subjects, cases, range(reps)))
        got = [(x.get('B'), x.get('case'), x.get('rep')) for x in inv]
        if len(got) != len(set(got)) or set(got) != want:
            ok, why = False, why + ['invocation population differs']
        for x in inv:
            t = [x.get('tA1'), x.get('tB'), x.get('tA2')]
            if not fin(t, 3, pos=True):
                ok, why = False, why + ['non-finite or non-positive time']
                break
            if abs(x['ratio'] - x['tB'] / (.5 * (x['tA1'] + x['tA2']))) > 1e-12 * x['ratio'] or \
                    abs(x['drift'] - x['tA2'] / x['tA1']) > 1e-12 * x['drift'] or x.get('hashes_ok') is not True:
                ok, why = False, why + ['ratio / drift / hash inconsistent']
                break
        if ok:
            pv = TM.panel_validity(inv)
            if not pv['valid']:
                ok, why = False, why + ['recomputed panel validity false']
        gate(name, ok, why=why)

    def warm(name, w, cases, reps):
        ok = w is not None and len(w.get('times', [])) == len(cases) * reps
        if ok:
            from collections import Counter
            cnt = Counter(x.get('case') for x in w['times'])
            ok = set(cnt) == set(cases) and all(v == reps for v in cnt.values()) and \
                fin([x.get('seconds') for x in w['times']], len(cases) * reps, pos=True) and \
                all(x.get('hash_ok') is True for x in w['times'])
        gate(name, ok)

    def steps():
        lines = (O / 'STEPS.txt').read_text().splitlines()
        gate('STEPS.txt bound to this job', job is not None and bool(lines) and lines[0] == f'job={job} commit={commit}', head=lines[:1])
        rc = {}
        for l in lines[1:]:
            p = l.split()
            if len(p) >= 2 and p[1].startswith('rc='):
                rc.setdefault(p[0], []).append(int(p[1][3:]))
        for w in ['test_s3_M2', 'test_s3_W1024', 'test_fom', 'smkgrid', 'smkfom', 'smkprof', 'smkmesh']:
            gate(f'step {w} ran once, exit 0', rc.get(w) == [0], rc=rc.get(w))

    def bound(r, cfgname):
        cfg = json.loads((HERE / 'configs' / f'{cfgname}.json').read_text())
        gate(f'{cfgname}: config identical', r is not None and r.get('config') == cfg)
        ident(r, cfgname)
        gate(f'{cfgname}: complete', r is not None and r.get('complete') is True)
        return cfg

    # ------------------------------------------------------------------------------------------- standalone tests
    def tests():
        for s_, rule in (('M2_512', 'gl32'), ('W1024', 'lat65536')):
            t = load(O / f'test_s3-{s_}.json')
            ident(t, f'test_s3 {s_}')
            gate(f'test_s3 {s_}: setting, rule, cases 0 5 7', t is not None and t.get('setting') == s_ and t.get('rule') == rule
                 and t.get('cases') == [0, 5, 7])
            c = {} if t is None else t.get('checks', {})
            gate(f'test_s3 {s_}: G0 bitwise (3 cases)', c.get('G0_bitwise') == [True, True, True])
            for k in ('G1_CN_0.025', 'G1_BDF2_0.0125'):
                gate(f'test_s3 {s_}: {k}', c.get(k, {}).get('bitwise') is True and c.get(k, {}).get('extra_finite') is True)
            for k in ('G2_CN_0.025', 'G2_BDF2_0.0125'):
                w = c.get(k, {}).get('worst_excess')
                gate(f'test_s3 {s_}: {k} (recomputed worst excess <= 0)', w is not None and np.isfinite(w) and w <= 0, worst=w)
            q1 = c.get('qn|CN|0.025|k1', {})
            gate(f'test_s3 {s_}: QN-1 CN 2.5 valid, certified stationary, within the bar',
                 fin(q1.get('dist'), 3) and max(q1['dist']) <= BAR and q1.get('valid') == [True] * 3
                 and q1.get('nonstationary') == [0, 0, 0], dist=q1.get('dist'))
            k2 = c.get('qn|CN|0.025|k1|q0', {})
            gate(f'test_s3 {s_}: K2 fires', fin(k2.get('nonstationary'), 3) and min(k2['nonstationary']) > 0)
            k3 = c.get('qn|CN|0.025|k1|noJuT', {}).get('dist', []) + c.get('qn|BE|0.01|k1|noJuT', {}).get('dist', [])
            gate(f'test_s3 {s_}: K3 fires', fin(k3, 6) and max(k3) > BAR)
            gate(f'test_s3 {s_}: G4 <= 1e-8', fin(c.get('G4_dist'), 3) and max(c['G4_dist']) <= 1e-8, G4=c.get('G4_dist'))
            gate(f'test_s3 {s_}: K6 fires', fin(c.get('K6_dist'), 3) and max(c['K6_dist']) > 2.5e-5)
            cnd = c.get('mf', {}).get('cond', {})
            gate(f'test_s3 {s_}: mf Gram conditioned, rank (m >= R\')', fin([cnd.get('cond')]) and cnd['cond'] < 1e10
                 and cnd.get('m', 0) >= (512 if s_ == 'M2_512' else 1024))
        t = load(O / 'test_fom.json')
        ident(t, 'test_fom')
        c = {} if t is None else t.get('checks', {})
        for k in ('run_UB', 'run_UC', 'run_CC'):
            cv = c.get(k, {}).get('converged', [False, {}])
            gate(f'test_fom: {k} converged with complete evidence', cv[0] is True and cv[1].get('complete') is True)
        r = c.get('G6v', {}).get('rel')
        gate('test_fom: G6v (vendor FOM) rel <= 1e-12', r is not None and np.isfinite(r) and r <= 1e-12, rel=r)
        d = c.get('K8', {}).get('diff')
        gate('test_fom: K8 fires (diff > 1e-6)', d is not None and np.isfinite(d) and d > 1e-6, diff=d)
        g7 = c.get('G7', {})
        per = [v for k, v in g7.items() if k not in ('pass', 'control_fires')]
        gate('test_fom: G7 rel <= 1e-12 and half-step control > 1e-9 on 2 cases', len(per) == 2 and
             all(fin([v.get('rel'), v.get('half_dt_diff')]) and v['rel'] <= 1e-12 and v['half_dt_diff'] > 1e-9 for v in per))
        g5 = c.get('G5_M2_512', {})
        gate('test_fom: G5/K7 M2 recomputed', g5.get('M') == g5.get('M_ref') and g5.get('kxyz_equal') is True and
             g5.get('lam_equal') is True and fin([g5.get('A_rel'), g5.get('K7_rel')]) and g5['A_rel'] <= 1e-11 and g5['K7_rel'] > 1e-4)
        t = load(HERE / 'checks' / 'test_fom-local-cpu.json')
        gate('local evidence G6 (bitwise vs E0 refcore, UB UC CC)', t is not None and
             all(t['checks'].get(k) is True for k in ('G6_UB', 'G6_UC', 'G6_CC')))
        t = load(HERE / 'checks' / 'test_g8.json')
        gate('local evidence G8 (lattice = Gram distance)', t is not None and fin([t.get('max_rel')]) and t['max_rel'] <= 1e-12)

    # ------------------------------------------------------------------------------------------------------ grid
    def grid():
        r = load(O / 'smkgrid' / 'result.json')
        cfg = bound(r, 'smkgrid')
        cases = cfg['case_subset']
        tc = cfg['timing']
        names = [s['name'] for s in cfg['settings']]
        gate('smkgrid: settings exactly', sorted(r['settings']) == sorted(names))
        rows = {}
        for x in r['rows']:
            rows.setdefault((x['tag'], x['spec']), []).append(x)
        ctl = AN.controls(r, None)
        for s in cfg['settings']:
            sn = s['name']
            subj = []
            for rule in [s['rule']] + s.get('reduced', []):
                adt, sv = SR.grid_plan(s, rule, cfg)
                for sp in adt + sv:
                    rs = rows.get((f'{sn}|{rule}', S.spec_key(sp)), [])
                    ok = sorted(x['case'] for x in rs) == sorted(cases) and all(fin(x.get('seconds_first'), 1, pos=True) for x in rs)
                    if sp[0] != 'ad':
                        ok = ok and all(len(x.get('gam_cert', [])) == x.get('steps_n') == S.nsteps(sp[2])[0] for x in rs)
                    gate(f'smkgrid {sn}|{rule}|{S.spec_key(sp)}: rows on exactly the cases (+ certification)', ok)
                d = r['settings'][sn]['rules'][rule]['dist_to_adt']
                gate(f'smkgrid {sn}|{rule}: distances for every spec, finite', sorted(d) == sorted(S.spec_key(x) for x in sv) and
                     all(np.asarray(v, float).shape == (len(cases), 6) and np.isfinite(v).all() for v in d.values()))
                subj += [f'{sn}|{rule}|{S.spec_key(x)}' for x in sv if len(x) <= 4]
            panel(f'smkgrid panel screen|{sn}', r['timing'].get(f'screen|{sn}'), subj + ['T1|C_old twice'], cases[:tc['cases']], tc['reps'])
            tm = r['timing'].get(f'screen|{sn}')
            t1 = [x['tB'] / (.5 * (x['tA1'] + x['tA2'])) for x in (tm or {}).get('invocations', []) if x['B'].startswith('T1')]
            gate(f'smkgrid {sn}: T1 recomputed median in [1.8, 2.2]', bool(t1) and 1.8 <= float(np.median(t1)) <= 2.2,
                 t1=float(np.median(t1)) if t1 else None)
            for k in ('K1', 'K2', 'K3_CN', 'K3_BE'):
                gate(f'smkgrid {sn}: {k} evaluated', k in ctl.get(sn, {}))
            c = ctl.get(sn, {})
            gate(f'smkgrid {sn}: K1 fires', c.get('K1', {}).get('eligible') is False)
            gate(f'smkgrid {sn}: K2 fires', c.get('K2', {}).get('eligible') is False)
            gate(f'smkgrid {sn}: K3 fires', c.get('K3_CN', {}).get('dist_cases_above', 0) + c.get('K3_BE', {}).get('dist_cases_above', 0) >= 1)
            dep = s['rule']
            att = r['settings'][sn]['rules'][dep].get('adt_tight_tighter', {})
            wv = [v.get('worst') for v in att.values()]
            gate(f'smkgrid {sn}: ADT tight/tighter <= 2.5e-5 for every (scheme, step)', len(att) == len(SR.grid_plan(s, dep, cfg)[0])
                 and fin(wv) and max(wv) <= 2.5e-5 and all(v.get('tighter_valid') is True for v in att.values()), worst=wv)
            rp = r['settings'][sn]['rules'][dep].get('h11_replay', {})
            gate(f'smkgrid {sn}: replay records, no error', all(v.get('label') not in (None, 'replay error') for v in rp.values()),
                 labels={k: v.get('label') for k, v in rp.items()})

    # ------------------------------------------------------------------------------------------------------- fom
    def fom():
        r = load(O / 'smkfom' / 'result.json')
        cfg = bound(r, 'smkfom')
        cases, tc = cfg['case_subset'], cfg['timing']
        want = [f'{fam}|L{L}|{dt:.8g}' for fam, L, dts in cfg['fom']['fd'] for dt in dts]
        gate('smkfom: FD population exactly', sorted(r['fom']['fd']) == sorted(want))
        for k, v in r['fom']['fd'].items():
            cal = v.get('calibration', {})
            lev0 = cal.get('levels', {}).get('0', {})
            gate(f'smkfom {k}: level 0 converged on the calibration cases', sorted(int(j) for j in lev0) == sorted(cfg['fom']['calib_cases'])
                 and all(x.get('converged') is True for x in lev0.values()))
            gate(f'smkfom {k}: calibrated level chosen', cal.get('chosen') in (0, 1, 2))
            cs = v.get('cases', {})
            gate(f'smkfom {k}: converged + finite scores on exactly the cases', sorted(int(j) for j in cs) == sorted(cases) and
                 all(c.get('converged') is True and c.get('complete') is True and fin([c.get('e')]) and fin(c.get('e_t'), 6) for c in cs.values()))
            warm(f'smkfom {k}: warm timing exact', v.get('warm'), cases[:tc['cases']], tc['reps'])
        wantp = [f'ps|N{N}|{dt:.8g}' for N in cfg['fom']['ps']['Ns'] for dt in cfg['fom']['ps']['dts']]
        gate('smkfom: PS population exactly', sorted(r['fom']['ps']) == sorted(wantp))
        for k, v in r['fom']['ps'].items():
            cs = v.get('cases', {})
            gate(f'smkfom {k}: finite + scored on exactly the cases', sorted(int(j) for j in cs) == sorted(cases) and
                 all(c.get('converged') is True and fin([c.get('e')]) and fin(c.get('e_t'), 6) for c in cs.values()))
            warm(f'smkfom {k}: warm timing exact', v.get('warm'), cases[:tc['cases']], tc['reps'])

    # ------------------------------------------------------------------------------------------------------ prof
    def prof():
        r = load(O / 'smkprof' / 'result.json')
        cfg = bound(r, 'smkprof')
        cases, tc, b = cfg['case_subset'], cfg['timing'], cfg['prof']
        lab2 = {c['label']: c for c in b['configs']}
        for lab in b['profile']:
            sp = S.parse_key(lab2[lab]['spec'])
            subj = PROFILE_SUBJECTS + (['adaptive_start'] if sp[0] in ('fs', 'fsp') else [])
            panel(f'smkprof profile {lab}', r['timing'].get(f'profile|{lab}'), subj, cases[:tc['cases']], tc['reps'])
        for la, lb in b['confirm']:
            panel(f'smkprof confirm {la}|{lb}', r['timing'].get(f'confirm|{la}|{lb}'), [lb], cases[:b['confirm_cases']], b['confirm_reps'])
        P = r['prof']
        gate('smkprof: batch labels exactly', sorted(P['batch']) == sorted(b['batch']['labels']))
        for lab in b['batch']['labels']:
            gate(f'smkprof batch {lab}: sizes exactly', sorted(int(x) for x in P['batch'].get(lab, {})) == sorted(b['batch']['sizes']))
            for B in b['batch']['sizes']:
                x = P['batch'].get(lab, {}).get(str(B), {})
                gate(f'smkprof batch {lab} B={B}: feasible, 3 finite timings, hashes', 'infeasible' not in x and
                     fin(x.get('seconds'), 3, pos=True) and x.get('hashes_ok') is True, error=x.get('infeasible'))
                gate(f'smkprof batch {lab} B={B}: G4 <= 2.5e-5 (recomputed)', fin([x.get('G4_max')]) and x['G4_max'] <= 2.5e-5,
                     G4=x.get('G4_max'))
                gate(f'smkprof batch {lab} B={B}: K6+ (permutation) <= 2.5e-5', fin([x.get('K6plus_max')]) and x['K6plus_max'] <= 2.5e-5)
                if B > 1:
                    gate(f'smkprof batch {lab} B={B}: K6 fires (> 2.5e-5)', fin([x.get('K6_max')]) and x['K6_max'] > 2.5e-5)
        for lab in b['mf']:
            m = P['mf'].get(lab, {})
            d = np.asarray(m.get('dist_to_adt', []), float)
            gate(f'smkprof mf {lab}: every case, ADT valid on every case, finite', d.shape == (len(cases), 6) and np.isfinite(d).all()
                 and m.get('adt_valid') == [True] * len(cases), info=m.get('invalid'))
            k = P['k5'].get(lab, {})
            cnd = k.get('cond', {})
            gate(f'smkprof K5 {lab}: gl12 full rank and conditioned (qualification deferred, A4.5)',
                 fin([cnd.get('cond')]) and cnd['cond'] < 1e10 and cnd.get('m', 0) >= 512, cond=cnd)
            rows = [x for x in r['rows'] if x['tag'].endswith('mfproj_gl12') and x['setting'] == lab2[lab]['setting']]
            gate(f'smkprof K5 {lab}: runs on exactly the cases', sorted(x['case'] for x in rows) == sorted(cases))
        for lab in b['conv']:
            dq = np.asarray(P['conv'].get(lab, {}).get('delta_quad', []), float)
            gate(f'smkprof conv {lab}: delta_quad on every case, finite', dq.shape == (len(cases), 6) and np.isfinite(dq).all())
        for lab in b['reduced_adt']:
            x = P['reduced_adt'].get(lab, {})
            gate(f'smkprof reduced ADT {lab}: tight/tighter <= 2.5e-5', fin([x.get('worst')]) and x['worst'] <= 2.5e-5, worst=x.get('worst'))

    # ------------------------------------------------------------------------------------------------------ mesh
    def mesh():
        r = load(O / 'smkmesh' / 'result.json')
        cfg = bound(r, 'smkmesh')
        cases, tc, b = cfg['case_subset'], cfg['timing'], cfg['mesh']
        M = r['mesh']
        for key in ('M2|512', 'W1024|256', 'W1024|1024'):
            g = M['G5'].get(key, {})
            gate(f'smkmesh G5/K7 {key} recomputed', g.get('M') == g.get('M_ref') and g.get('kxyz_equal') is True and
                 g.get('lam_equal') is True and fin([g.get('A_rel'), g.get('K7_rel')]) and g['A_rel'] <= 1e-11 and g['K7_rel'] > 1e-4)
        for L in b['Ls']:
            E = M['per_L'].get(str(L))
            gate(f'smkmesh L{L}: present', E is not None)
            if E is None:
                continue
            fom_keys = []
            for rom in b['roms']:
                v = E['rom'].get(rom['label'], {})
                d = np.asarray(v.get('dist_to_adt', []), float)
                ok = (d.shape == (len(cases), 6) and np.isfinite(d).all() and v.get('adt_valid') == [True] * len(cases)
                      and v.get('adt_tighter_valid') == [True] * min(8, len(cases)) and fin([v.get('adt_tight_tighter')]))
                gate(f'smkmesh L{L} ROM {rom["label"]}: complete, ADT valid (pilot)', ok, info=v.get('infeasible'), tables=v.get('tables'))
            for fam, dt in b['same']:
                v = E['fom'].get(f'{fam}|{dt:.8g}', {})
                cs = v.get('cases', {})
                gate(f'smkmesh L{L} FOM {fam} {dt}: converged + scored on exactly the cases (pilot)',
                     sorted(int(j) for j in cs) == sorted(cases[:b['acc_cases']]) and
                     all(c.get('converged') is True and fin([c.get('e')]) for c in cs.values()),
                     info=v.get('infeasible') or (v.get('calibration') or {}).get('status'))
                if 'key' in v:
                    fom_keys.append(v['key'])
            for rl, fk in b['matched']:
                cs = E['fom'].get(fk, {}).get('cases', {})
                gate(f'smkmesh L{L} matched {fk}: converged + scored on exactly the timing cases',
                     sorted(int(j) for j in cs) == sorted(cases[:tc['cases']]) and
                     all(c.get('converged') is True and fin([c.get('e')]) for c in cs.values()))
                fom_keys.append(fk)
            for rom in b['roms']:
                panel(f"smkmesh L{L} panel {rom['label']}", r['timing'].get(f"mesh|L{L}|{rom['label']}"), sorted(set(fom_keys)),
                      cases[:tc['cases']], tc['reps'])

    for name, fn in (('steps', steps), ('tests', tests), ('grid', grid), ('fom', fom), ('prof', prof), ('mesh', mesh)):
        section(name, fn)
    ok = len(G) > 0 and all(v['result'] == 'pass' for v in G.values())
    out = dict(verdict='PASS' if ok else 'FAIL', failing=[k for k, v in G.items() if v['result'] != 'pass'], n_gates=len(G), gates=G,
               offline_after_collection=['K4 (score3.py on smkgrid)', 'K9 (analyze.k9 on smkgrid)'],
               deferred_A4_5=['K10 (FOM matcher)', 'G3', 'H3.2/K5 qualification verdict'])
    (O / 'ACCEPTANCE.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(out['verdict'], out['n_gates'], out['failing'][:20])


if __name__ == '__main__':
    main()
