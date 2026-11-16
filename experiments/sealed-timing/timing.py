"""Post-closeout timing panel on validation inputs (PROTOCOL.md is the specification).

Runs the pinned plateau{2,3}d driver unchanged on a validation cohort with a config derived from the frozen closeout
config, and runs the registered measurement panel inside the driver's in-job selection hook, using the driver's own
query closures, prepared inputs and output checks.

python timing.py --base <upstream experiments/burgers-quadrature-plateau> --dim 2 --mesh 256 --out output
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = {2: 'val32', 3: 'val64'}
RULE = {2: 'fib987', 3: 'lat4096'}
MATCHED = {(2, 256): 'PS_N48', (2, 512): 'PS_N48', (2, 1024): 'PS_N48', (2, 2048): 'PS_N48', (2, 4096): 'PS_N48',
           (3, 65): 'PS_N16', (3, 129): 'PS_N20', (3, 257): 'PS_N20'}
CASES = list(range(6))
SCHEDULE = dict(warm_passes=3, initial_burn=60.0, discard_reps=1, main_reps=3, rewarm_passes=1, reburn=60.0,
                confirm_discard_reps=1, confirm_reps=5, burn=0.1, seed=20261105)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def frozen_config(dim, mesh):
    """Closeout config, checked against F1 and its review, then the PROTOCOL.md §3 substitutions only."""
    v = HERE / 'vendor'
    review = json.loads((v / 'F1-REVIEW.json').read_text())
    f1b = (v / 'F1.json').read_bytes()
    assert review['decision'] == 'ACCEPT' and sha(f1b) == review['freeze_sha256'], 'F1 freeze'
    f1 = json.loads(f1b)
    name = f'configs/d{dim}m{mesh}.json'
    raw = (v / name).read_bytes()
    assert sha(raw) == f1['files'][name], name
    assert sha((v / 'upstream.json').read_bytes()) == f1['files']['upstream.json']
    cfg = json.loads(raw)
    arm = [a for a in cfg['arms'] if a['name'] == RULE[dim]]
    fom = [d for d in cfg['foms'] if d['name'] == MATCHED[(dim, mesh)]]
    assert len(arm) == 1 and len(fom) == 1
    new = dict(cfg, cohort=VALIDATION[dim], cases=CASES, arms=arm, foms=fom, conv_arm=RULE[dim],
               anchor_check_arm=RULE[dim],
               timing=dict(cfg['timing'], reps=1, confirm_reps=1, initial_burn=0.0))
    if dim == 2:
        new['audit_cases'] = [0, 2]
    changed = sorted(k for k in set(cfg) | set(new) if cfg.get(k) != new.get(k))
    assert set(changed) <= {'cohort', 'cases', 'arms', 'foms', 'conv_arm', 'anchor_check_arm', 'timing',
                            'audit_cases'}, changed
    return cfg, new, dict(source_config=name, source_config_sha256=sha(raw), freeze_sha256=review['freeze_sha256'],
                          changed_keys=changed)


def check_upstream(base):
    files = json.loads((HERE / 'vendor' / 'upstream.json').read_text())['files']
    for name, want in files.items():
        assert sha((base / name).read_bytes()) == want, f'upstream mismatch: {name}'


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def measurement(dim, solver, frame, out, log):
    """The registered panel (PROTOCOL.md §3), inside the driver process, on the driver's own objects."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    block = jax.block_until_ready
    L = frame
    arm = RULE[dim]
    fname = L['cfg']['foms'][0]['name']
    tcases = list(L['tcases'])
    assert tcases == CASES, tcases
    if dim == 2:
        Q, F2, ph_all, Lm = solver.Q, solver.F2, L['ph_all'], L['L']
        sp = dict(L['fspecs'])[fname]
        u0s = {c: u for (u, nu, c) in L['inputs']}
        nus = {c: nu for (u, nu, c) in L['inputs']}
        call_lat = L['arms'][arm]['call_lat']
        foms = L['foms']
        PREP = L['PREP']
        xidx = {c: i for i, c in enumerate(tcases)}
        check = L['check']

        def rom_prep(c):
            return block(jnp.asarray(Q.e.initial(Lm, ph_all[c])))

        def fom_prep(c):
            return F2.prepare(sp, ph_all[c])[0]

        def rom_q(c, u0=None):
            return call_lat(u0s[c] if u0 is None else u0, nus[c])

        def fom_q(c, x=None):
            return foms.callable(sp, PREP[(fname, c)] if x is None else x, nus[c], 'lattice')()

        def chk_rom(c, o):
            return check(('rom_lat', arm), xidx[c], o)

        def chk_fom(c, o):
            return check(('fom', fname), xidx[c], o)
        burn = solver.QS.burn
    else:
        C, psjax, FR, tab, n = solver.C, solver.psjax, solver.FR, L['tab'], L['n']
        d = L['fdef'][fname]
        assert d['kind'] == 'ps'
        sp = ('ps', int(d['N']), FR.S_canon(d['dt']))
        u0s, nus = L['U0'], L['NU']
        s_ = L['arms'][arm]
        foms, PREP, check = L['foms'], L['PREP'], L['check']

        def rom_prep(c):
            return block(jnp.asarray(C.initial_interior(n, tab, c)))

        def fom_prep(c):
            return block(psjax.ic(tab, c, sp[1]))

        def rom_q(c, u0=None):
            return s_['ql'](u0s[c] if u0 is None else u0, nus[c], s_['dl'], {})

        def fom_q(c, x=None):
            return foms.run(sp, PREP[(fname, c)] if x is None else x, nus[c])[0]

        def chk_rom(c, o):
            return check(('rom_lat', arm), c, o)

        def chk_fom(c, o):
            return check(('fom', fname), c, o)
        burn = L['burn']

    subjects = {
        'rom_lat': (lambda c: rom_q(c), chk_rom),
        'fom': (lambda c: fom_q(c), chk_fom),
        'rom_prep': (rom_prep, lambda c, o: rel(o, u0s[c])),
        'fom_prep': (fom_prep, lambda c, o: rel(o, PREP[(fname, c)])),
        'rom_e2e': (lambda c: rom_q(c, rom_prep(c)), chk_rom),
        'fom_e2e': (lambda c: fom_q(c, fom_prep(c)), chk_fom),
    }
    bitwise = {'rom_prep': lambda c, o: bool(np.array_equal(np.asarray(o), np.asarray(u0s[c]))),
               'fom_prep': lambda c, o: bool(np.array_equal(np.asarray(o), np.asarray(PREP[(fname, c)])))}
    names = list(subjects)
    rng = np.random.default_rng(SCHEDULE['seed'])
    inv = []
    smi_t0 = time.time()

    def warm(passes):
        for _ in range(passes):
            for s in names:
                for c in tcases:
                    block(subjects[s][0](c))

    def rep(block_name, r):
        order = [(s, c) for s in names for c in tcases]
        rng.shuffle(order)
        for s, c in order:
            fn, ck = subjects[s]
            burn(SCHEDULE['burn'])
            la = os.getloadavg()[0]
            t0w = time.time()
            t1 = time.perf_counter()
            o = fn(c)
            block(o)
            secs = time.perf_counter() - t1
            row = dict(block=block_name, subject=s, case=c, rep=r, seconds=secs, diff=ck(c, o), loadavg1=la,
                       wall=t0w - smi_t0)
            if s in bitwise:
                row['bitwise'] = bitwise[s](c, o)
            inv.append(row)
            del o
        log(f'measurement {block_name} rep {r} done')

    log('measurement: warm-up')
    warm(SCHEDULE['warm_passes'])
    burn(SCHEDULE['initial_burn'])
    for r in range(SCHEDULE['discard_reps']):
        rep('discard_main', r)
    for r in range(SCHEDULE['main_reps']):
        rep('main', r)
    warm(SCHEDULE['rewarm_passes'])
    burn(SCHEDULE['reburn'])
    for r in range(SCHEDULE['confirm_discard_reps']):
        rep('discard_confirm', r)
    for r in range(SCHEDULE['confirm_reps']):
        rep('confirm', r)
    res = dict(schedule=SCHEDULE, subjects=names, matched_fom=fname, rule=arm, cases=tcases,
               measurement_start_unix=smi_t0, invocations=inv)
    (out / 'measurement.json').write_text(json.dumps(res, indent=1))
    log('measurement written')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--mesh', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert 'JCP_OPEN_REGISTRY' not in os.environ and 'JCP_APPROVED_REVIEW_SHA256' not in os.environ
    orig, cfg, prov = frozen_config(a.dim, a.mesh)
    check_upstream(a.base)
    a.out.mkdir(parents=True, exist_ok=True)
    cpath = a.out / 'derived-config.json'
    cpath.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    prov['derived_config_sha256'] = sha(cpath.read_bytes())
    (a.out / 'config-provenance.json').write_text(json.dumps(prov, indent=1))
    sys.path.insert(0, str(a.base))
    solver = importlib.import_module(f'plateau{a.dim}d')
    if a.dim == 3:
        solver.COHORT['val64'] = (923801, 64, None)      # refstore val64 = seed 923801; G_coh checks draws vs E0
    begin = time.perf_counter()
    log = lambda s: print(f'[sealed-timing {time.perf_counter() - begin:.1f}s] {s}', flush=True)
    state = dict(done=False)

    def hook(rep, *args, **kwargs):
        frame = sys._getframe(1).f_locals
        assert not state['done'], 'hook called twice'
        measurement(a.dim, solver, frame, a.out, log)
        state['done'] = True
        return {'arm': RULE[a.dim], 'selection': 'frozen-closeout'}
    solver.PL.cheapest_plateau_rule = hook
    sys.argv = [str(solver.__file__), '--config', str(cpath.resolve()), '--out', str(a.out.resolve())]
    solver.main()
    assert state['done'], 'measurement panel did not run'
    print('SEALED-TIMING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
