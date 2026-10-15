"""jcp-speed3d smoke acceptance (DESIGN A4.1; audits code-2 #1, code-3 #1-#2). Reads the smoke outputs of THIS attempt and
writes <out>/ACCEPTANCE.json: one entry per gate (pass / fail, with evidence), the overall verdict, and the gates that are
checked offline after collection (K4 by score3.py, K9 by analyze.py) or deferred by A4.5 (K10, G3, H3.2/K5 qualification).
Exception-safe: every section is evaluated even if another raises (a raising section is a failed gate). Coverage is checked
against the smoke configs (settings, panels, FOM ladder, profiles, batch sizes, meshes); every step's exit status must be 0;
every result.json must name this attempt's config and job id.

    python smoke_accept.py --out <smoke output dir>
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

import analyze as AN
import c3common as C3

HERE = Path(__file__).resolve().parent


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    O = Path(a.out)
    job = os.environ.get('SLURM_JOB_ID')
    commit = os.environ.get('SOURCE_COMMIT')
    G = {}
    BAR = 2.5e-4

    def ident(obj, what):
        """Every artifact must carry this execution's job id and source commit (both required)."""
        gate(f'{what} bound to job {job} / commit {commit}', job is not None and commit is not None and obj is not None
             and obj.get('job_id') == job and obj.get('commit') == commit,
             found=None if obj is None else [obj.get('job_id'), obj.get('commit')])

    def panel_ok(tm, ncases, reps):
        if tm is None or not tm.get('valid') or tm.get('missing_expected'):
            return False
        inv = tm['invocations']
        if len(inv) != len(tm['subjects']) * ncases * reps or not inv:
            return False
        return all(x['hashes_ok'] and np.isfinite(x['ratio']) and x['ratio'] > 0 for x in inv)

    def gate(name, ok, **ev):
        G[name] = dict(result='pass' if ok else 'fail', **ev)

    def section(name, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            gate(f'{name} (section raised)', False, error=traceback.format_exc()[-1500:])

    def steps():
        lines = (O / 'STEPS.txt').read_text().splitlines()
        gate('STEPS.txt bound to this job', bool(lines) and lines[0] == f'job={job} commit={commit}', head=lines[:1])
        rc = {l.split()[0]: int(l.split()[1].split('=')[1]) for l in lines if ' rc=' in l}
        want = ['test_s3_M2', 'test_s3_W1024', 'test_fom', 'smkgrid', 'smkfom', 'smkprof', 'smkmesh']
        for w in want:
            gate(f'step {w} exit 0', rc.get(w) == 0, rc=rc.get(w))

    def bound(r, cfgname):
        cfg = json.loads((HERE / 'configs' / f'{cfgname}.json').read_text())
        gate(f'{cfgname} config identical', r is not None and r['config'] == cfg)
        ident(r, cfgname)
        gate(f'{cfgname} complete', r is not None and bool(r.get('complete')))
        return cfg

    def tests():
        for s_ in ('M2_512', 'W1024'):
            t = load(O / f'test_s3-{s_}.json')
            ident(t, f'test_s3 {s_}')
            c = {} if t is None else t['checks']
            gate(f'test_s3 {s_} cases 0 5 7', t is not None and t['cases'] == [0, 5, 7])
            gate(f'test_s3 {s_} G0 bitwise', c.get('G0_bitwise') == [True, True, True])
            for k in ('G1_CN_0.025', 'G1_BDF2_0.0125'):
                gate(f'test_s3 {s_} {k}', bool(c.get(k, {}).get('bitwise')) and bool(c.get(k, {}).get('extra_finite')))
            for k in ('G2_CN_0.025', 'G2_BDF2_0.0125'):
                gate(f'test_s3 {s_} {k}', c.get(k, {}).get('ok') is True)
            q1 = c.get('qn|CN|0.025|k1', {})
            gate(f'test_s3 {s_} QN-1 CN 2.5 within the bar, valid', len(q1.get('dist', [])) == 3 and max(q1['dist']) <= BAR
                 and all(q1.get('valid', [])), dist=q1.get('dist'))
            k2 = c.get('qn|CN|0.025|k1|q0', {})
            gate(f'test_s3 {s_} K2 fires', len(k2.get('nonstationary', [])) == 3 and min(k2['nonstationary']) > 0)
            k3 = c.get('qn|CN|0.025|k1|noJuT', {}).get('dist', []) + c.get('qn|BE|0.01|k1|noJuT', {}).get('dist', [])
            gate(f'test_s3 {s_} K3 fires', len(k3) == 6 and max(k3) > BAR)
            g4 = c.get('G4_dist', [])
            gate(f'test_s3 {s_} G4', len(g4) == 3 and np.isfinite(g4).all() and max(g4) <= 1e-8, G4=g4)
            k6 = c.get('K6_dist', [])
            gate(f'test_s3 {s_} K6 fires', len(k6) == 3 and max(k6) > 2.5e-5)
            mf = c.get('mf', {})
            gate(f'test_s3 {s_} mf projection conditioned', mf.get('cond', {}).get('cond', np.inf) < 1e10)
        t = load(O / 'test_fom.json')
        ident(t, 'test_fom')
        c = {} if t is None else t['checks']
        for k in ('run_UB', 'run_UC', 'run_CC'):
            gate(f'test_fom {k} converged', bool(c.get(k, {}).get('converged', [False])[0]))
        gate('test_fom G6v (vendor FOM)', bool(c.get('G6v', {}).get('pass_')), rel=c.get('G6v', {}).get('rel'))
        gate('test_fom K8 fires', bool(c.get('K8', {}).get('fires')))
        gate('test_fom G7 (PS port) + control', bool(c.get('G7', {}).get('pass')) and bool(c.get('G7', {}).get('control_fires')))
        gate('test_fom G5/K7 M2', bool(c.get('G5_M2_512', {}).get('pass')) and bool(c.get('G5_M2_512', {}).get('K7_fires')))
        for f, need in (('checks/test_fom-local-cpu.json', ['G6_UB', 'G6_UC', 'G6_CC']), ('checks/test_g8.json', None)):
            t = load(HERE / f)
            ok = t is not None and (all(t['checks'].get(k) is True for k in need) if need else bool(t.get('pass_')))
            gate(f'local evidence {f} (staged, committed copy)', ok, sha256=C3.sha_file(HERE / f) if (HERE / f).exists() else None)

    def grid():
        r = load(O / 'smkgrid' / 'result.json')
        cfg = bound(r, 'smkgrid')
        names = [s['name'] for s in cfg['settings']]
        gate('smkgrid settings', sorted(r['settings']) == sorted(names))
        for sn in names:
            tm = r['timing'].get(f'screen|{sn}')
            gate(f'smkgrid panel screen|{sn} complete, valid, hashes', panel_ok(tm, cfg['timing']['cases'], cfg['timing']['reps']),
                 missing_expected=None if tm is None else tm.get('missing_expected'))
            gate(f'smkgrid panel screen|{sn} T1 in [1.8, 2.2]', tm is not None and AN.t1_ok(tm))
            n_inv = 0 if tm is None else len(tm['invocations'])
            gate(f'smkgrid panel screen|{sn} invocation count', tm is not None and n_inv == len(tm['subjects']) *
                 cfg['timing']['cases'] * cfg['timing']['reps'], n=n_inv)
        ctl = AN.controls(r, None)
        for sn in names:
            for k in ('K1_fires', 'K2_fires', 'K3_fires'):
                gate(f'smkgrid {sn} {k}', bool(ctl.get(sn, {}).get(k)))
            sv = r['settings'][sn]
            dep = next(iter(sv['rules']))
            worst = max(v['worst'] for v in sv['rules'][dep]['adt_tight_tighter'].values())
            gate(f'smkgrid {sn} ADT tight/tighter <= 2.5e-5', worst <= 2.5e-5, worst=worst)
            rp = sv['rules'][dep].get('h11_replay', {})
            gate(f'smkgrid {sn} replay ran without error', all(v.get('label') != 'replay error' for v in rp.values()),
                 labels={k: v.get('label') for k, v in rp.items()})
            rows = [x for x in r['rows'] if x['setting'] == sn]
            gate(f'smkgrid {sn} rows cover the cases', sorted({x['case'] for x in rows}) == sorted(cfg['case_subset']))

    def fom():
        r = load(O / 'smkfom' / 'result.json')
        cfg = bound(r, 'smkfom')
        want = [f'{fam}|L{L}|{dt:.8g}' for fam, L, dts in cfg['fom']['fd'] for dt in dts]
        gate('smkfom FD population', sorted(r['fom']['fd']) == sorted(want))
        for k, v in r['fom']['fd'].items():
            gate(f'smkfom {k} calibrated', v['calibration'].get('chosen') is not None, cal_status=v['calibration'].get('status'))
            gate(f'smkfom {k} converged + scored on exactly the cases', 'cases' in v and
                 sorted(int(x) for x in v['cases']) == sorted(cfg['case_subset']) and
                 all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
            nw = cfg['timing']['cases'] * cfg['timing']['reps']
            gate(f'smkfom {k} warm timing complete + hashes', 'warm' in v and len(v['warm']['times']) == nw and
                 bool(v['warm']['hashes_ok']) and all(np.isfinite(x['seconds']) for x in v['warm']['times']))
        wantp = [f'ps|N{N}|{dt:.8g}' for N in cfg['fom']['ps']['Ns'] for dt in cfg['fom']['ps']['dts']]
        gate('smkfom PS population', sorted(r['fom']['ps']) == sorted(wantp))
        for k, v in r['fom']['ps'].items():
            gate(f'smkfom {k} finite + scored on exactly the cases', sorted(int(x) for x in v['cases']) == sorted(cfg['case_subset'])
                 and all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
            nw = cfg['timing']['cases'] * cfg['timing']['reps']
            gate(f'smkfom {k} warm timing complete + hashes', len(v['warm']['times']) == nw and bool(v['warm']['hashes_ok']))

    def prof():
        r = load(O / 'smkprof' / 'result.json')
        cfg = bound(r, 'smkprof')
        b = cfg['prof']
        for lab in b['profile']:
            tm = r['timing'].get(f'profile|{lab}')
            gate(f'smkprof profile {lab} panel complete + valid', panel_ok(tm, cfg['timing']['cases'], cfg['timing']['reps']),
                 missing_expected=None if tm is None else tm.get('missing_expected'))
        for la, lb in b['confirm']:
            tm = r['timing'].get(f'confirm|{la}|{lb}')
            gate(f'smkprof confirm {la}|{lb} complete + valid', panel_ok(tm, b['confirm_cases'], b['confirm_reps']))
        P = r['prof']
        for lab in b['batch']['labels']:
            for B in b['batch']['sizes']:
                x = P['batch'].get(lab, {}).get(str(B))
                if x is None or 'infeasible' in x:
                    gate(f'smkprof batch {lab} B={B} feasible', False, error=None if x is None else x['infeasible'])
                    continue
                gate(f'smkprof batch {lab} B={B} G4', x['G4'] in ('exact', 'within tolerance'), G4_max=x['G4_max'])
                gate(f'smkprof batch {lab} B={B} hashes', bool(x['hashes_ok']))
                gate(f'smkprof batch {lab} B={B} K6+ (permutation)', bool(x['K6plus_pass']))
                if B > 1:
                    gate(f'smkprof batch {lab} B={B} K6 fires', bool(x['K6_fires']))
        for lab in b['mf']:
            m = P['mf'].get(lab, {})
            d = np.asarray(m.get('dist_to_adt', []), float)
            gate(f'smkprof mf {lab} ran on every case, ADT valid, finite', d.shape == (len(cfg['case_subset']), 6) and
                 np.isfinite(d).all() and all(m.get('adt_valid', [])), info=m.get('invalid'))
            k = P['k5'].get(lab, {})
            gate(f'smkprof K5 {lab} gl12 rank / condition ok (qualification deferred, A4.5)', bool(k.get('rank_ok')), cond=k.get('cond'))
        for lab in b['conv']:
            gate(f'smkprof conv {lab} ran', lab in P['conv'])
        for lab in b['reduced_adt']:
            x = P['reduced_adt'].get(lab)
            gate(f'smkprof reduced ADT {lab} tight/tighter <= 2.5e-5', x is not None and x['worst'] <= 2.5e-5, worst=None if x is None else x['worst'])

    def mesh():
        r = load(O / 'smkmesh' / 'result.json')
        cfg = bound(r, 'smkmesh')
        M = r['mesh']
        gate('smkmesh G5/K7 all banks incl. W256 and M2', bool(M.get('G5_all')) and 'W1024|256' in M['G5'] and 'M2|512' in M['G5'],
             G5={k: v.get('gate') for k, v in M['G5'].items()})
        b = cfg['mesh']
        nc, tcs = len(cfg['case_subset']), cfg['timing']
        for L in b['Ls']:
            E = M['per_L'].get(str(L))
            gate(f'smkmesh L{L} present', E is not None)
            if E is None:
                continue
            for rom in b['roms']:
                v = E['rom'].get(rom['label'], {})
                d = np.asarray(v.get('dist_to_adt', []), float)
                ok = (d.shape == (nc, 6) and np.isfinite(d).all() and len(v.get('adt_valid', [])) == nc and all(v['adt_valid'])
                      and len(v.get('adt_tighter_valid') or []) == min(8, nc) and all(v['adt_tighter_valid'])
                      and np.isfinite(v.get('adt_tight_tighter', np.nan)))
                gate(f'smkmesh L{L} ROM {rom["label"]} complete (pilot)', ok, info=v.get('infeasible'), tables=v.get('tables'))
            for fam, dt in b['same']:
                v = E['fom'].get(f'{fam}|{dt:.8g}', {})
                cs = v.get('cases', {})
                gate(f'smkmesh L{L} FOM {fam} {dt} complete (pilot)', sorted(int(x) for x in cs) == sorted(cfg['case_subset'][:b['acc_cases']])
                     and all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in cs.values()),
                     info=v.get('infeasible') or (v.get('calibration') or {}).get('status'))
            for rl, fk in b['matched']:
                cs = E['fom'].get(fk, {}).get('cases', {})
                gate(f'smkmesh L{L} matched {fk} complete', sorted(int(x) for x in cs) == sorted(cfg['case_subset'][:tcs['cases']])
                     and all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in cs.values()))
            for rom in b['roms']:
                tm = r['timing'].get(f"mesh|L{L}|{rom['label']}")
                gate(f"smkmesh L{L} panel {rom['label']} complete + valid", panel_ok(tm, tcs['cases'], tcs['reps']))

    for name, fn in (('steps', steps), ('tests', tests), ('grid', grid), ('fom', fom), ('prof', prof), ('mesh', mesh)):
        section(name, fn)
    ok = all(v['result'] == 'pass' for v in G.values()) and len(G) > 0
    out = dict(verdict='PASS' if ok else 'FAIL', failing=[k for k, v in G.items() if v['result'] != 'pass'], gates=G,
               offline_after_collection=['K4 (score3.py on smkgrid)', 'K9 (analyze.k9 on smkgrid)'],
               deferred_A4_5=['K10 (FOM matcher)', 'G3', 'H3.2/K5 qualification verdict'])
    (O / 'ACCEPTANCE.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(out['verdict'], out['failing'])


if __name__ == '__main__':
    main()
