"""jcp-regimes resource certificates and the production plan (DESIGN A2.6, A3.3, A6; pattern jcp-speed3d/resources.py).
CPU only; reads the smoke acceptance and the committed configs.

    python resources.py plan          -> regenerates the production configs (make_configs.prod with the reductions the
                                         certificate requires), writes checks/resources-<attempt>.json for every
                                         production attempt and checks/resources-plan.json
Rates (seconds, H200, measured by the smoke): ordinary references = max of the dev6 / A+ / B2 pilots; low-nu references
(A-, C1, C2) = the C2 pilot; x1.2 retries/compilation, + 180 s per refjob step (unit gates + JIT); the A+ job adds the
tight reserve (4.5 GPU-h, A2.6); a conditional calibration probe (A1.4) costs the largest measured probe; ROM: the
smoke's qacc time per (case x arm) x production (case x arm) x 1.3, + 300 s; fverify: the smoke's measured fverify time
scaled by the number of production fields, x2; PS: 600 s per case. Time limit = 1.5 x projection.
Reduction order if the certified projection exceeds 50 GPU-h (A3.3): (1) FOM item (not implemented: deferred), (2) PS,
(3) C2 -> c0-c7; if still above, the plan refuses (stop and amend).
"""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOWNU = ('Am', 'C1', 'C2')
TIGHT_RESERVE_H = 4.5
REGIMES = ['N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2']
LIMIT = 50.


def rom_runs(cfg):
    """Enumerated qacc workload (A3.3): per arm, prod + tight per case, tighter per tighter case, successor replay per
    Q2 case, check-rule replay per case in both sets."""
    import rscore as RS
    n = 0
    for st in cfg['settings']:
        for a in st['arms']:
            ex = RS.expected_populations(cfg, a)
            n += 2 * len(ex['cases']) + len(ex['dev']) + len(ex['q2']) + len(set(ex['dev']) & set(ex['q2']))
    return n


def n_fields(cfg):
    import rscore as RS
    return sum(len(RS.expected_populations(cfg, a)['cases']) for st in cfg['settings'] for a in st['arms'])


def basis():
    """Measured rates from the accepted smoke; a missing measurement refuses certification (A6)."""
    sm = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text())
    assert sm['all_blocking_pass'], 'smoke not accepted'
    pil = sm['pilots']
    smk_rom = json.loads((HERE / 'configs' / 'smk1-rom.json').read_text())
    st_ = sm['step_seconds']
    need = ('qacc:configs/smk1-rom.json', 'fverify:configs/smk1-fverify.json')
    missing = [k for k in need if k not in st_] + ([] if sm.get('probe_seconds') else ['probe_seconds']) + \
        ([] if 'calib_agree' in sm else ['calib_agree'])
    if missing:
        raise SystemExit(f'unmeasured inputs, cannot certify: {missing}')
    return dict(ordinary=max(pil[k]['seconds'] for k in ('dev6|0', 'Ap|13', 'B2|15')), lownu=pil['C2|9']['seconds'],
                probe=max(sm['probe_seconds'].values()), rom_per_run=st_[need[0]] / rom_runs(smk_rom),
                fverify_per_field=st_[need[1]] / n_fields(smk_rom), calib_agree=bool(sm['calib_agree']), smoke=sm['run'])


def cost(runs, b):
    total, parts = 0., []
    for r in runs:
        drv, cf = r.split(':', 1)
        cfg = json.loads((HERE / cf).read_text())
        if drv == 'refjob' and cfg['mode'] == 'fd':
            rate = b['lownu'] if cfg['cohort'] in LOWNU else b['ordinary']
            s = len(cfg['cases']) * rate * 1.2 + 180 + TIGHT_RESERVE_H * 3600 * cfg.get('tight_allowance', 0)
            s += b['probe'] * len(cfg.get('probes', {}).get('tight_coarse', []))
        elif drv == 'refjob' and cfg['mode'] == 'ps':
            s = len(cfg['cases']) * 600. + 180
        elif drv == 'qacc':
            s = rom_runs(cfg) * b['rom_per_run'] * 1.3 + 300
        elif drv == 'fverify':
            rom = json.loads((HERE / 'configs' / 'rom1024.json').read_text())
            s = max(300., n_fields(rom) * b['fverify_per_field'] * 2)
        elif drv == 'refslim':
            s = 120.
        else:
            raise SystemExit(f'no rate for {r}')
        parts.append(dict(run=r, seconds=s))
        total += s
    return total / 3600, parts


def attempts(ps=True):
    A = {f'ref{g}': [f'refjob:configs/ref-{g}.json', f'refslim:configs/slim-{g}.json'] for g in REGIMES}
    A['rom1024'] = ['qacc:configs/rom1024.json', 'fverify:configs/fverify-rom1024.json']
    if ps:
        A['ps'] = [f'refjob:configs/ps-{g}.json' for g in REGIMES if g != 'N0s']
    return A


def plan():
    import make_configs as MC
    b = basis()
    calib = not b['calib_agree']
    steps = [dict(ps=True, reduce_c2=False), dict(ps=False, reduce_c2=False), dict(ps=False, reduce_c2=True)]
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    committed = led['spent_gpu_hours'] + led['reserved_gpu_hours']
    seen = {j['attempt'] for j in led['jobs']}
    for k, opt in enumerate(steps):
        MC.prod(reduce_c2=opt['reduce_c2'], calib_probes=calib)
        A = attempts(opt['ps'])
        assert not (set(A) & seen), f'attempt names already used: {sorted(set(A) & seen)} (a retry needs a new name)'
        certs = {a: cost(r, b) for a, r in A.items()}
        proj = sum(c[0] for c in certs.values())
        limits = sum(math.ceil(1.5 * c[0] * 4) / 4 for c in certs.values())
        if proj <= LIMIT and committed + limits <= 60.:      # both the certified projection and the reservation cap
            break
    else:
        raise SystemExit(f'mandatory remainder {proj:.1f} GPU-h (reservations {limits:.1f} + committed {committed:.1f}) '
                         f'does not fit: stop and amend (A3.3)')
    out = {}
    for a, runs in A.items():
        p_, parts = certs[a]
        h = math.ceil(1.5 * p_ * 4) / 4
        cfgs = {r.split(':', 1)[1]: hashlib.sha256((HERE / r.split(':', 1)[1]).read_bytes()).hexdigest() for r in runs}
        c_ = dict(attempt=a, projection_hours=round(p_, 4), time_limit_hours=h, configs=cfgs, runs=runs, parts=parts, basis=b)
        (HERE / 'checks' / f'resources-{a}.json').write_text(json.dumps(c_, indent=1) + '\n')
        out[a] = dict(projection_hours=round(p_, 4), time_limit_hours=h, runs=runs)
    pl = dict(attempts=out, reductions=dict(step=k, **steps[k], calib_probes=calib, fom='deferred (A3.3 step 1)'),
              projection_hours=round(proj, 3), total_time_limit_hours=sum(v['time_limit_hours'] for v in out.values()),
              remaining_reservation_hours=sum(v['time_limit_hours'] for v in out.values()), basis=b)
    (HERE / 'checks' / 'resources-plan.json').write_text(json.dumps(pl, indent=1) + '\n')
    print(json.dumps({k: v for k, v in pl.items() if k != 'basis'}, indent=1))


if __name__ == '__main__':
    {'plan': plan}[sys.argv[1]]()
