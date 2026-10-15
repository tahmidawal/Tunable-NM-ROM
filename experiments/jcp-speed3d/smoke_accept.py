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
    G = {}

    def gate(name, ok, **ev):
        G[name] = dict(result='pass' if ok else 'fail', **ev)

    def section(name, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            gate(f'{name} (section raised)', False, error=traceback.format_exc()[-1500:])

    def steps():
        lines = (O / 'STEPS.txt').read_text().splitlines()
        rc = {l.split()[0]: int(l.split()[1].split('=')[1]) for l in lines if ' rc=' in l}
        want = ['test_s3_M2', 'test_s3_W1024', 'test_fom', 'smkgrid', 'smkfom', 'smkprof', 'smkmesh']
        for w in want:
            gate(f'step {w} exit 0', rc.get(w) == 0, rc=rc.get(w))

    def bound(r, cfgname):
        cfg = json.loads((HERE / 'configs' / f'{cfgname}.json').read_text())
        gate(f'{cfgname} bound to this attempt', r is not None and r['config'] == cfg and (job is None or r.get('job_id') == job),
             job=None if r is None else r.get('job_id'))
        gate(f'{cfgname} complete', r is not None and bool(r.get('complete')))
        return cfg

    def tests():
        for s in ('M2_512', 'W1024'):
            t = load(O / f'test_s3-{s}.json')
            need = ['G0_bitwise', 'G1_CN_0.025', 'G1_BDF2_0.0125', 'G2_CN_0.025', 'G2_BDF2_0.0125', 'qn|CN|0.025|k1',
                    'qn|CN|0.025|k1|q0', 'qn|CN|0.025|k1|noJuT', 'G4_dist', 'K6_dist', 'mf']
            gate(f'test_s3 {s}: every gate present, cases 0 5 7, no failures',
                 t is not None and t.get('mandatory_failures') == [] and all(k in t['checks'] for k in need)
                 and t['cases'] == [0, 5, 7], failures=None if t is None else t.get('mandatory_failures'))
        t = load(O / 'test_fom.json')
        need = ['run_UB', 'run_UC', 'run_CC', 'G6v', 'K8', 'G7', 'G5_M2_512']
        gate('test_fom: G6v, K8, G7, G5/K7 M2, FOM runs converge; no failures',
             t is not None and t.get('mandatory_failures') == [] and all(k in t['checks'] for k in need),
             failures=None if t is None else t.get('mandatory_failures'))
        for f, need in (('checks/test_fom-local-cpu.json', ['G6_UB', 'G6_UC', 'G6_CC']), ('checks/test_g8.json', None)):
            t = load(HERE / f)
            ok = t is not None and (all(t['checks'].get(k) is True for k in need) if need else bool(t.get('pass_')))
            gate(f'local evidence {f} (staged copy)', ok, sha256=C3.sha_file(HERE / f) if (HERE / f).exists() else None)

    def grid():
        r = load(O / 'smkgrid' / 'result.json')
        cfg = bound(r, 'smkgrid')
        names = [s['name'] for s in cfg['settings']]
        gate('smkgrid settings', sorted(r['settings']) == sorted(names))
        for sn in names:
            tm = r['timing'].get(f'screen|{sn}')
            gate(f'smkgrid panel screen|{sn} present + valid + hashes', tm is not None and bool(tm.get('valid')),
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
            gate(f'smkfom {k} converged + scored (all cases)', 'cases' in v and len(v['cases']) == len(cfg['case_subset']) and
                 all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
            gate(f'smkfom {k} warm hashes', 'warm' in v and bool(v['warm']['hashes_ok']))
        wantp = [f'ps|N{N}|{dt:.8g}' for N in cfg['fom']['ps']['Ns'] for dt in cfg['fom']['ps']['dts']]
        gate('smkfom PS population', sorted(r['fom']['ps']) == sorted(wantp))
        for k, v in r['fom']['ps'].items():
            gate(f'smkfom {k} finite + scored', all(c['converged'] and np.isfinite(c.get('e', np.nan)) for c in v['cases'].values()))
            gate(f'smkfom {k} warm hashes', bool(v['warm']['hashes_ok']))

    def prof():
        r = load(O / 'smkprof' / 'result.json')
        cfg = bound(r, 'smkprof')
        b = cfg['prof']
        for lab in b['profile']:
            tm = r['timing'].get(f'profile|{lab}')
            gate(f'smkprof profile {lab} panel valid', tm is not None and bool(tm.get('valid')),
                 missing_expected=None if tm is None else tm.get('missing_expected'))
        for la, lb in b['confirm']:
            tm = r['timing'].get(f'confirm|{la}|{lb}')
            gate(f'smkprof confirm {la}|{lb} valid', tm is not None and bool(tm.get('valid')))
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
            gate(f'smkprof mf {lab} ran (cond gate passed)', 'dist_to_adt' in m, info=m.get('invalid'))
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
        for L in b['Ls']:
            E = M['per_L'].get(str(L))
            gate(f'smkmesh L{L} present', E is not None)
            if E is None:
                continue
            for rom in b['roms']:
                v = E['rom'].get(rom['label'], {})
                ok = 'dist_to_adt' in v and all(v['adt_valid']) and v.get('adt_tighter_valid') is not None and all(v['adt_tighter_valid'])
                gate(f'smkmesh L{L} ROM {rom["label"]} (ADT valid, pilot)', ok, info=v.get('infeasible'), tables=v.get('tables'))
            for fam, dt in b['same']:
                v = E['fom'].get(f'{fam}|{dt:.8g}', {})
                gate(f'smkmesh L{L} FOM {fam} {dt} (pilot)', 'cases' in v and all(c['converged'] for c in v['cases'].values()),
                     info=v.get('infeasible') or (v.get('calibration') or {}).get('status'))
            for rl, fk in b['matched']:
                gate(f'smkmesh L{L} matched {fk}', fk in E['fom'])
                for rom in b['roms']:
                    tm = r['timing'].get(f"mesh|L{L}|{rom['label']}")
                    gate(f"smkmesh L{L} panel {rom['label']} valid", tm is not None and bool(tm.get('valid')))
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
