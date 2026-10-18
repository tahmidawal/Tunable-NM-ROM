"""jcp-fom2d production ladder configs (DESIGN §3 as amended by A1.1). Writes configs/<attempt>.json.

  fcc    F-CC: L in {32..4096} x dt in {0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025}
  fub    F-UB: L in {64..2048} x dt in {0.05, 0.025, 0.01, 0.005, 0.0025, 0.00125, 0.000625}
  fub4k  F-UB: L = 4096 x the same steps (first item of the defer order, its own job)
  fps    F-PS: N in {8, 12, ..., 256} x dt in {0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.001, 0.0005}
Groups are ordered cheapest first (resolution, then step count), so a timed-out job leaves the cheap part complete and a
resubmission resumes (ladder.py skips finished groups/configurations).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CC_L = [32, 64, 128, 256, 512, 1024, 2048, 4096]
CC_DT = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025]
UB_L = [64, 128, 256, 512, 1024, 2048, 4096]
UB_DT = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
PS_N = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
PS_DT = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.001, 0.0005]
COMMON = dict(cases='all38', calibration_cases=[['dev6', j] for j in range(4)], warm_cases=[['dev6', 0], ['dev6', 1]],
              warm_reps=3, burn=0.1, levels=['0', '1', '2', '3'])


def fd_groups(fam, Ls, dts):
    return [[fam, L, dt] for L in Ls for dt in sorted(dts, reverse=True)]


def main():
    cfgs = {
        'fcc': dict(COMMON, attempt='fcc', fd=fd_groups('CC', CC_L, CC_DT), ps=[]),
        'fub': dict(COMMON, attempt='fub', fd=fd_groups('UB', [L for L in UB_L if L < 4096], UB_DT), ps=[]),
        'fub4k': dict(COMMON, attempt='fub4k', fd=fd_groups('UB', [4096], UB_DT), ps=[]),
        'fps': dict(COMMON, attempt='fps', fd=[], ps=[[N, dt] for N in PS_N for dt in sorted(PS_DT, reverse=True)]),
    }
    for k, c in cfgs.items():
        (HERE / 'configs' / f'{k}.json').write_text(json.dumps(c, indent=1) + '\n')
        print(k, len(c['fd']), 'FD groups', len(c['ps']), 'PS configurations')


if __name__ == '__main__':
    main()
