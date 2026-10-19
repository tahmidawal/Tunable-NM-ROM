"""jcp-regimes resource certificate (DESIGN A2.6, A3.3; pattern jcp-speed3d/resources.py). CPU, reads only the smoke
acceptance and the committed configs.

    python resources.py <attempt> --run refjob:configs/ref-N0.json [--run refslim:configs/slim-N0.json ...]
        -> checks/resources-<attempt>.json  {projection_hours, time_limit_hours, configs{path: sha256}, runs, basis}

Rates (seconds per case, H200) come from the smoke's pilot references: ordinary = max over the dev6, A+ and B2 pilots;
low-nu (A-, C1, C2) = the C2 pilot. x1.2 for retries/compilation, + 120 s job overhead per refjob step (unit gates, JIT).
The A+ reference job adds the tight reserve (4.5 GPU-h, A2.6). The ROM job scales the smoke's measured qacc time per
(case x arm) to the production population, x1.3. PS: 600 s per case (E0 A100 rate; conservative on H200). Time limit =
1.5 x projection (rounded up to 15 min by the stager).
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOWNU = ('Am', 'C1', 'C2')
TIGHT_RESERVE_H = 4.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('attempt')
    ap.add_argument('--run', action='append', required=True)
    a = ap.parse_args()
    sm = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text())
    assert sm['all_blocking_pass'], 'smoke not accepted'
    pil = sm['pilots']
    ordinary = max(pil[k]['seconds'] for k in ('dev6|0', 'Ap|13', 'B2|15'))
    lownu = pil['C2|9']['seconds']
    smk_rom = json.loads((HERE / 'configs' / 'smk1-rom.json').read_text())
    n_smoke_arms = sum(len(s['arms']) for s in smk_rom['settings'])
    n_smoke_cases = len(smk_rom['settings'][0]['arms'][0]['cases'])
    rom_rate = sm['qacc_elapsed'] / (n_smoke_arms * n_smoke_cases)       # seconds per (case x arm), all run kinds
    total_s, parts, configs = 0., [], {}
    for r in a.run:
        drv, cf = r.split(':', 1)
        cfg = json.loads((HERE / cf).read_text())
        configs[cf] = hashlib.sha256((HERE / cf).read_bytes()).hexdigest()
        if drv == 'refjob' and cfg['mode'] == 'fd':
            rate = lownu if cfg['cohort'] in LOWNU else ordinary
            s = len(cfg['cases']) * rate * 1.2 + 120 + (TIGHT_RESERVE_H * 3600 * cfg.get('tight_allowance', 0))
            parts.append(dict(run=r, cases=len(cfg['cases']), rate_s=rate, seconds=s))
        elif drv == 'refjob' and cfg['mode'] == 'ps':
            s = len(cfg['cases']) * 600. + 120
            parts.append(dict(run=r, cases=len(cfg['cases']), rate_s=600., seconds=s))
        elif drv == 'qacc':
            from p1common import case_list, select
            n_arms = sum(len(s_['arms']) for s_ in cfg['settings'])
            n_cases = len(select(case_list(cfg), cfg['settings'][0]['arms'][0]['cases']))
            s = n_arms * n_cases * rom_rate * 1.3 + 300
            parts.append(dict(run=r, arms=n_arms, cases=n_cases, rate_s=rom_rate, seconds=s))
        elif drv == 'refslim':
            s = 120.
            parts.append(dict(run=r, seconds=s))
        else:
            raise SystemExit(f'no rate for {r}')
        total_s += s
    proj = total_s / 3600
    cert = dict(attempt=a.attempt, projection_hours=round(proj, 4), time_limit_hours=round(1.5 * proj, 4),
                configs=configs, runs=a.run, parts=parts,
                basis=dict(ordinary_s=ordinary, lownu_s=lownu, rom_s_per_case_arm=rom_rate, smoke=sm['run']))
    (HERE / 'checks' / f'resources-{a.attempt}.json').write_text(json.dumps(cert, indent=1) + '\n')
    print(json.dumps(dict(attempt=a.attempt, projection_hours=cert['projection_hours'],
                          time_limit_hours=cert['time_limit_hours'])))


if __name__ == '__main__':
    main()
