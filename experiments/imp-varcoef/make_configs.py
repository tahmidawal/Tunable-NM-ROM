"""Write imp-varcoef job configs (DESIGN §6-8, A1-A5).

    python make_configs.py base          ladder keys + reference configs + smoke configs
    python make_configs.py ladder        production ladder config (needs checks/eligibility.json)
    python make_configs.py time          production timing config (needs runs/<ladder>/.../candidates.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
NS = '/cluster/tufts/paralab/tawal01/imp-varcoef'
LEVELS = [[512, 0.005], [1024, 0.0025], [2048, 0.00125]]          # A7.3 (registered smoke levels: LEVELS_FALLBACK)
LEVELS_FALLBACK = [[1024, 0.0025], [2048, 0.00125], [4096, 0.000625]]
TAUS = [0.01, 0.03, 0.1]


def ladder_keys():
    k = []
    Ns = [16, 24, 32, 48, 64, 96, 128]
    for N in Ns:
        for dt in (0.025, 0.0125, 0.00625, 0.003125):
            for tol in (1e-3, 1e-5, 1e-8):
                for sch in ('SBDF2', 'CNAB2'):
                    k.append(f'psi|N{N}|{dt:g}|{sch}|{tol:g}')
        for dt in (0.025, 0.0125, 0.00625, 0.003125, 0.0015625):
            for kb in ('max', 'mid'):
                k.append(f'etd|N{N}|{dt:g}|{kb}')
    for L in (64, 128, 256, 512):
        for dt in (0.0125, 0.00625, 0.003125, 0.0015625):
            for tol in (1e-4, 1e-7):
                for pr in ('MG', 'DST'):
                    k.append(f'fdmg|L{L}|{dt:g}|{pr}|{tol:g}')
        for dt in (0.0125, 0.00625, 0.003125):
            for lv in ('1', '2', '3'):
                k.append(f'ccv|L{L}|{dt:g}|t{lv}')
    return k


def rom_entries():
    """A (C2 baseline) and the frozen per-target denominators (checks/rom-box.json, A5.12) as C2 ROM specs with
    target_e on dev6 (from FRONT-2D per-case errors)."""
    rb = json.loads((HERE / 'checks/rom-box.json').read_text())
    c2 = json.loads((HERE / 'inputs/c24096-result.json').read_text())
    specs = {r['label']: r for r in c2['config']['roms']}
    front = json.loads((HERE.parent / 'jcp-fom2d/inputs/FRONT-2D.json').read_text())['meshes']['4096']
    fm = {m['config']: m for m in front['members'] + front['other_qualified']}
    out = []
    for tau, r in rb['cheapest_by_target'].items():
        if r is None:
            continue
        sp = dict(specs[r['label']])
        pc = fm[r['front_config']]['errors']['per_case']
        sp['target_e'] = {c: pc[c]['e'] for c in pc if c.startswith('dev6|')}
        sp['tau'] = float(tau)
        out.append(sp)
    return c2['config']['A'], out


def dev_by_nu():
    sys.path.insert(0, str(HERE))
    import vcoef as VC
    d = VC.cohort('dev')
    return sorted(range(len(d)), key=lambda j: d[j]['desc'][4])


def base():
    CFG.mkdir(exist_ok=True)
    keys = ladder_keys()
    (CFG / 'ladder-keys.json').write_text(json.dumps(keys, indent=0) + '\n')
    pooled = [['dev', j] for j in range(8)] + [['val', j] for j in range(24)]
    for C in (100, 10):
        (CFG / f'ref{C}.json').write_text(json.dumps(dict(contrasts=[C], cases=pooled, levels=LEVELS,
                                                          crosscheck=[['dev', j] for j in range(4)], controls=True),
                                                     indent=1) + '\n')
    # ---- smoke: real sizes, two dev cases (smallest nu = hardest, and dev 0) -----------------------------------
    hard = dev_by_nu()[0]
    sm = sorted({0, hard})
    if 'refs-only' in sys.argv:                                   # A7.3: production reference configs only
        return
    (CFG / 'smk-impl.json').write_text(json.dumps(dict(small=False)) + '\n')
    (CFG / 'smk-ref.json').write_text(json.dumps(dict(contrasts=[100, 10], cases=[['dev', j] for j in sm], levels=LEVELS,
                                                      crosscheck=[['dev', hard]], controls=True), indent=1) + '\n')
    skeys = ['psi|N48|0.0125|SBDF2|1e-05', 'psi|N128|0.003125|CNAB2|1e-08', 'psi|N32|0.025|SBDF2|0.001',
             'etd|N48|0.0125|mid', 'etd|N128|0.0015625|max',
             'fdmg|L128|0.00625|MG|1e-07', 'fdmg|L512|0.0015625|MG|1e-07', 'fdmg|L256|0.003125|DST|0.0001',
             'ccv|L128|0.00625|t2', 'ccv|L512|0.003125|t1']
    ids = dict(dev=[f'dev|{j}' for j in sm], val=[])
    refdir = '../../output/vref_smk-ref'
    (CFG / 'smk-ladder.json').write_text(json.dumps(dict(
        contrasts=[100], keys=skeys, refdir=refdir, eligibility_from=refdir + '/result.json', ids=ids, taus=TAUS,
        screen=dict(mesh=4096, reps=2, burn=0.1), ntop=1, extensions=True, qualify=True), indent=1) + '\n')
    A, roms = rom_entries()
    fk = ['psi|N48|0.0125|SBDF2|1e-05@C100', 'fdmg|L128|0.00625|MG|1e-07@C100', 'etd|N48|0.0125|mid@C100',
          'ccv|L128|0.00625|t2@C100']
    foms = {k: dict(family=k.split('|')[0].upper()) for k in fk}
    foms['KERNEL|psi|N48|0.0125|SBDF2|1e-05@C100'] = dict(family='PSI')
    foms['SETUP|psi|N48|0.0125|SBDF2|1e-05@C100'] = dict(family='PSI')
    groups = [dict(id=f'{f}|smk', target='smk', family=f, criterion='worst', candidates=[k]) for k, f in
              ((k, k.split('|')[0].upper()) for k in fk)]
    groups.append(dict(id='union|smk', target='smk', family='union', criterion='worst', candidates=fk, pool_by_family=True))
    (CFG / 'smk-time.json').write_text(json.dumps(dict(
        attempt='smk-time', mesh=4096, edges=[0, 128, 256, 384, 512], cases=[['dev6', s] for s in range(6)], A=A,
        roms=[{k: v for k, v in r.items() if k != 'tau'} for r in roms], foms=foms, groups=groups,
        slots={'100': [f'dev|{sm[s % len(sm)]}' for s in range(6)]}, refdir=[refdir],
        slot_e_from='../../output/vladder_smk-ladder/result.json', reps=1, burn=0.1, seed=7,
        controls=dict(doubled_cases=[['dev6', 0]], fault_subject=roms[0]['label'],
                      identity_fault_subject='psi|N48|0.0125|SBDF2|1e-05@C100'),
        pilot_foms=['psi|N128|0.003125|CNAB2|1e-08@C100']), indent=1) + '\n')
    print('keys', len(keys), 'smoke cases', sm, 'roms', [r['label'] for r in roms])


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fallback_test(ref_test_result):
    """A7.3 / A8.3 test fallback: every test case not accepted (or eps_ref > 1e-3) in the test reference result is rerun at
    LEVELS_FALLBACK, with the same test-opening bindings as configs/ref-test.json."""
    r = json.loads(Path(ref_test_result).read_text())
    base = json.loads((CFG / 'ref-test.json').read_text())
    bad = sorted(int(k.split('|')[2]) for k, v in r['cases'].items()
                 if not (v['judge']['outcome'] == 'accepted' and v['eps_ref'] is not None and v['eps_ref'] <= 1e-3))
    if not bad:
        print('no test fallback needed')
        return
    cfg = dict(base, cases=[['test', j] for j in bad], levels=LEVELS_FALLBACK)
    (CFG / 'reffb-test.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('test fallback', bad)


def fallback(job_results):
    """A7.3 fallback references (pooled dev + val): every ineligible case of the eligibility mask built from the first
    reference jobs (checks/eligibility.json, before freezing the final mask) rerun at LEVELS_FALLBACK, one config per contrast."""
    el = json.loads((HERE / 'checks/eligibility.json').read_text())
    for C, v in el['contrasts'].items():
        bad = sorted(v['excluded'])
        xc = [c for c in v.get('crosscheck_missing', [])] + [c for c, x in v['crosscheck'].items() if not x['passed']]
        if not bad and not xc:
            continue
        cases = [[c.split('|')[0], int(c.split('|')[1])] for c in sorted(set(bad) | set(xc))]
        (CFG / f'reffb{C}.json').write_text(json.dumps(dict(contrasts=[float(C)], cases=cases, levels=LEVELS_FALLBACK,
                                                            crosscheck=[c for c in cases if c[0] == 'dev' and c[1] < 4],
                                                            controls=False), indent=1) + '\n')
        print('fallback', C, cases)


def ladder(refdirs, contrasts=(100, 10)):
    """Production ladder config; refdirs = cluster output dirs of the reference jobs (fallback dirs last)."""
    el = HERE / 'checks/eligibility.json'
    fro = el.with_suffix('.FROZEN').read_text().split()[0]
    assert sha(el) == fro
    keys = [k for k in json.loads((CFG / 'ladder-keys.json').read_text()) if not k.startswith('ccv|L512')]   # A7.4 defer
    (CFG / 'ladder.json').write_text(json.dumps(dict(
        contrasts=list(contrasts), keys=keys, refdir=list(refdirs), eligibility='checks/eligibility.json',
        eligibility_sha256=fro, taus=TAUS, screen=dict(mesh=4096, reps=3, burn=0.1), ntop=3, max_tries=8, extensions=True,
        qualify=True, dev_only={'10': True}), indent=1) + '\n')


def time_cfg(refdirs, contrasts=(100,)):                 # A7.4: C = 10 timing deferred
    """Production timing config from checks/candidates.json (copied from the ladder output) and the frozen mask."""
    el = json.loads((HERE / 'checks/eligibility.json').read_text())
    ca = json.loads((HERE / 'checks/candidates.json').read_text())
    A, roms = rom_entries()
    foms, groups = {}, []
    for C in contrasts:
        for tau in TAUS:
            union = []
            for f in ('PSI', 'ETD', 'FDMG', 'CCV'):
                g = ca['groups'][f'{C:g}'][f'{f}|{tau:g}|C{C:g}']
                cands = [f'{k}@C{C:g}' for k in g['candidates']]
                for k in cands:
                    foms[k] = dict(family=f, slot_e=ca['candidates'][k]['slot_e'])
                    foms['KERNEL|' + k] = dict(family=f, slot_e=ca['candidates'][k]['slot_e'])
                    foms['SETUP|' + k] = dict(family=f, slot_e={})
                    if f == 'PSI' and C == 100:
                        foms[k.replace('@C100', '@C1')] = dict(family=f, slot_e={})
                if cands:
                    groups.append(dict(id=f'{f}|{tau:g}|C{C:g}', target=f'{tau:g}', family=f, criterion='worst',
                                       candidates=cands))
                union += cands
            if union:
                groups.append(dict(id=f'union|{tau:g}|C{C:g}', target=f'{tau:g}', family='union', criterion='worst',
                                   candidates=sorted(set(union)), pool_by_family=True))
    slots = {f'{C:g}': [x['fom'] for x in el['contrasts'][f'{C:g}']['slots']] for C in contrasts}
    q = [k for k in foms if not k.startswith(('KERNEL|', 'SETUP|')) and not k.endswith('@C1')]
    files = {n: sha(HERE / n) for n in ('checks/eligibility.json', 'checks/candidates.json', 'checks/rom-box.json')}
    (CFG / 'time.json').write_text(json.dumps(dict(
        attempt='time', mesh=4096, edges=[0, 128, 256, 384, 512], cases=[['dev6', s] for s in range(6)], A=A,
        roms=[{k: v for k, v in r.items() if k != 'tau'} for r in roms], foms=foms, groups=groups, slots=slots,
        refdir=list(refdirs), reps=3, burn=0.1, seed=20261021,
        bindings=dict(files=files, eligibility='checks/eligibility.json', candidates='checks/candidates.json'),
        controls=dict(doubled_cases=[['dev6', 0], ['dev6', 1]], fault_subject=roms[0]['label'],
                      identity_fault_subject=sorted(q)[0]),
        pilot_foms=[]), indent=1) + '\n')
    print('timing subjects', len(foms), 'groups', len(groups))


def test_cfg(refdirs_test, frozen_src):
    """Test-once: copy the timing job's FROZEN-SELECTION into checks/, write the one-time opening record, configs."""
    import shutil
    import time as _t
    sys.path.insert(0, str(HERE))
    import testgate
    dst = HERE / 'checks/FROZEN-SELECTION-4096.json'
    rec = HERE / 'checks/TEST-OPENED.json'
    assert not rec.exists(), 'the test split has already been opened (A2.18: once)'
    shutil.copyfile(frozen_src, dst)
    keys = testgate.permitted(json.loads(dst.read_text()))
    with open(rec, 'x') as fh:                                   # exclusive create: the test split opens once (A2.18)
        fh.write(json.dumps(dict(frozen_selection_sha256=sha(dst), permitted_keys=keys, purpose='Phase-1 FOM test-once',
                                   utc=_t.strftime('%Y-%m-%dT%H:%M:%SZ', _t.gmtime())), indent=1) + '\n')
    bind = dict(test_once=True, frozen_selection='checks/FROZEN-SELECTION-4096.json', frozen_selection_sha256=sha(dst),
                opening_record='checks/TEST-OPENED.json', opening_record_sha256=sha(rec))
    (CFG / 'ref-test.json').write_text(json.dumps(dict(contrasts=[100], cases=[['test', j] for j in range(16)], levels=LEVELS,
                                                       crosscheck=[], controls=False, **bind), indent=1) + '\n')
    (CFG / 'test-run.json').write_text(json.dumps(dict(contrast=100, keys=keys, refdir=list(refdirs_test), **bind),
                                                  indent=1) + '\n')
    print('test keys', keys)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'base':
        base()
    elif cmd == 'fallback':
        fallback(sys.argv[2:])
    elif cmd == 'fallback-test':
        fallback_test(sys.argv[2])
    elif cmd == 'ladder':
        ladder(sys.argv[2:])
    elif cmd == 'time':
        time_cfg(sys.argv[2:])
    elif cmd == 'test':
        test_cfg(sys.argv[3:], sys.argv[2])
    else:
        raise SystemExit(cmd)
