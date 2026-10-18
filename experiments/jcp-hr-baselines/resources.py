"""Resource certificate for the phase-II tuning jobs (DESIGN 12, A19.4c, A24.2/A24.4), regenerated from the smk2
measurements (runs/smk2/archive/output/smoke.json, job 5422908, H200). Writes checks/RESOURCES-tune.json.

Model (all inputs measured in smk2 unless marked "model"):
- per-run warm seconds of the `none` arm (native) at the smoke rank r0 (N-time A-B-A triples, median of tA);
  other ranks scale as (r/r0)^2 for r >= r0 (QR of the n x r Jacobian dominates; model) and linearly below r0 (model);
  a certified run (prod/tight/tighter) costs CERT_FACTOR x a warm run (model, from the smk2 grid phase);
- HR runs: the smk2 grid phase seconds per (kind, option) block divided by its 8 runs (2 cases x 2 solvers + 2 tight
  + 2 tighter), including compilation, used per run unchanged (conservative: compilation is per block, not per run);
- construction: smk2 basis/adequacy/adv_pod/qdeim/ecsw_states measured; the ECSW path per rank from A24.2's memory-
  bound model (bytes / 2 TB/s effective; model) with the A16.5 retry doubling;
- step-count factor for the 3D dt ladder: mean(steps)/steps(dt0) over the 12 options (exact);
- contingency 1.5 on the total.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'runs/smk2/archive/output/smoke.json').read_text())
CERT_FACTOR = 1.6
BW = 2.0e12                     # effective HBM bytes/s for the NNLS gradient/append passes (model; H200 peak 4.8e12)
CONT = 1.5


def ntime(c):
    T = S[c]['timing']['N-time']['cases']
    return float(np.median([t['tA1'] for v in T.values() for t in v['triples']]))


def none_run(c, r, r0):
    t0 = ntime(c)
    return t0 * ((r / r0) ** 2 if r >= r0 else max(r / r0, 0.25))


def nnls_path(m, N, s):
    it = 1.2 * s
    byts = 32 * m * s * s + 8 * m * N * it                 # appends (K ~ s) + gradient passes reading G
    fl_ref = sum(6 * m * (1024 * j) ** 2 for j in range(1, s // 1024 + 1))
    return 2 * (byts / BW + fl_ref / 3e13)                  # x2: A16.5 ban-free retry (worst case)


def cell(c, d, ranks, r0, options_steps, n_cases=6, n_blocks_smk=None, rows_per_rank=None, fom_runs=0, fom_s=0.):
    C = S[c]
    ph = {k: v['seconds'] for k, v in C['phases'].items()}
    steps0 = options_steps[0]
    sf = float(np.mean(options_steps)) / steps0
    nopt = len(options_steps)
    hr_block = ph['grid'] / n_blocks_smk / 8                # seconds per HR or none run in the smoke grid (incl. compile)
    runs_cert, runs_warm = 2 * n_cases + n_cases + 2, 2 * n_cases     # prod x2 solvers + tight + tighter; warm x2
    out = dict(construction={}, stageA={}, other={})
    out['construction'] = dict(basis=ph['basis'], adequacy=ph['adequacy'] * 2, probe_basis=ph['basis'],
                               adv_pod=ph['adv_pod'] * 1.5, qdeim=ph['qdeim_max_shape'] * 3, ecsw_states=ph['ecsw_states'],
                               ecsw_paths={r: nnls_path(rows_per_rank * r, 32768, 16 * r) for r in ranks})
    for r in ranks:
        tn = none_run(c, r, r0) * sf
        # native: prod + tight on every case and tighter on 2 (certified), warm on every case; common: prod + warm at
        # 0.1 x native (smk2: 0.22 s vs 3.1 s in 2D, 0.08 s vs 1.2 s in 3D)
        none = nopt * ((2 * n_cases + 2) * CERT_FACTOR * tn + n_cases * tn + 0.1 * n_cases * (CERT_FACTOR + 1) * tn)
        hr = 6 * nopt * (runs_cert + runs_warm) * hr_block * sf
        out['stageA'][r] = dict(none=none, hr=hr)
    out['other'] = dict(probe_runs=sum(nopt * n_cases * none_run(c, r, r0) * sf for r in ranks),
                        fom=fom_runs * fom_s, ours=600.)
    tot = (sum(v for k, v in out['construction'].items() if k != 'ecsw_paths') + sum(out['construction']['ecsw_paths'].values())
           + sum(v['none'] + v['hr'] for v in out['stageA'].values()) + sum(out['other'].values()))
    out['total_hours'] = tot / 3600
    out['projected_hours'] = CONT * tot / 3600
    return out


def main():
    R2, R3 = [32, 64, 128, 256, 384], [64, 128, 256, 512, 768, 1024]
    st3 = [25, 20, 15, 10] * 3
    c1024 = cell('2d_L1024', 2, R2, 128, [50, 50], n_blocks_smk=6, rows_per_rank=120, fom_runs=76, fom_s=1.8)
    c3 = cell('3d_L64', 3, R3, 256, st3, n_blocks_smk=9, rows_per_rank=102, fom_runs=768, fom_s=1.5)
    c256 = json.loads(json.dumps(c1024))                  # 2D 256^2: 1/16 of the state; 1024^2 x 0.25 is an upper bound
    c256['total_hours'] *= 0.25
    c256['projected_hours'] = c1024['projected_hours'] * 0.25 + CONT * 0.1   # + saturation (all-step POD, 29,376 states)
    led = json.loads((HERE / 'runs/GPU-LEDGER.json').read_text())
    spent = float(led['spent_gpu_hours'])
    tune = c256['projected_hours'] + c1024['projected_hours'] + c3['projected_hours']
    stageB = 0.5 * tune                                    # model: full Q1 of the Stage-B members + A-B-A pairs
    wall = dict(t2d256=2, t2d1024=5, t3d64=12)            # A26.4 hard bound: Slurm wall limits (1 GPU each)
    reserve_stageB = 7.5
    bound = spent + sum(wall.values()) + reserve_stageB
    res = dict(source='runs/smk2 (job 5422908, H200)', model=__doc__,
               hard_bound=dict(wall_hours=wall, stageB_reserve=reserve_stageB, spent=spent, total=bound,
                               within_cap=bool(bound <= 30.0),
                               note='GPU-hours cannot exceed the wall limits; the model projections below are '
                                    'conditional estimates, not certified'), cells=dict(t2d256=c256, t2d1024=c1024, t3d64=c3),
               spent=spent, tuning_projected=tune, stageB_projection=stageB, projected_total=spent + tune + stageB,
               cap=30.0, within_cap=bool(spent + tune + stageB <= 30.0),
               walltime_request_hours=wall,
               memory=dict(smk2_device_peak_2d1024=S['2d_L1024']['phases']['qdeim_max_shape']['device_peak'],
                           note='3D peak masked by the 2D cell in smk2 (process-cumulative); measured in t3d64 (A24.4)'))
    (HERE / 'checks' / 'RESOURCES-tune.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    for k, v in res['cells'].items():
        print(k, f"{v['projected_hours']:.2f} GPU-h projected (model, not certified)")
    print(f"spent {spent:.3f} + tuning {tune:.2f} + stageB {stageB:.2f} = {spent + tune + stageB:.2f} (cap 30)")
    print('hard bound', res['hard_bound'])


if __name__ == '__main__':
    main()
