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


def basis():
    sm = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text())
    assert sm['all_blocking_pass'], 'smoke not accepted'
    pil = sm['pilots']
    smk_rom = json.loads((HERE / 'configs' / 'smk1-rom.json').read_text())
    n_sm = sum(len(s['arms']) for s in smk_rom['settings']) * len(smk_rom['settings'][0]['arms'][0]['cases'])
    return dict(ordinary=max(pil[k]['seconds'] for k in ('dev6|0', 'Ap|13', 'B2|15')), lownu=pil['C2|9']['seconds'],
                probe=max(sm.get('probe_seconds', {}).values() or [900.]), rom=sm['qacc_elapsed'] / n_sm,
                fverify_per_field=(sm['step_seconds'].get('fverify:configs/smk1-fverify.json', n_sm * 1.0) / n_sm),  # 1 s/field if unmeasured calib_agree=sm.get('calib_agree'),
                smoke=sm['run'])


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
            import rscore as RS
            n = sum(len(RS.expected_populations(cfg, a)['cases']) for st in cfg['settings'] for a in st['arms'])
            s = n * b['rom'] * 1.3 + 300
        elif drv == 'fverify':
            rom = json.loads((HERE / 'configs' / 'rom1024.json').read_text())
            import rscore as RS
            n = sum(len(RS.expected_populations(rom, a)['cases']) for st in rom['settings'] for a in st['arms'])
            s = max(300., n * b['fverify_per_field'] * 2)
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
    for k, opt in enumerate(steps):
        MC.prod(reduce_c2=opt['reduce_c2'], calib_probes=calib)
        A = attempts(opt['ps'])
        certs = {a: cost(r, b) for a, r in A.items()}
        proj = sum(c[0] for c in certs.values())
        if proj <= LIMIT:
            break
    else:
        raise SystemExit(f'mandatory remainder {proj:.1f} GPU-h > {LIMIT}: stop and amend (A3.3)')
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
