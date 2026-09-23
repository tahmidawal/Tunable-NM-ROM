"""Writes every panel config (pre-registered in DESIGN.md). Run once; configs are committed."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
CG = {'fom_cncg_dt0.025_rtol1e-6_NAMED': dict(dt=.025, tolerance=1e-6)}
for dt in (.025, .05, .1):
    for tol in (1e-4, 1e-3, 1e-2):
        CG[f'fom_cncg_dt{dt:g}_rtol{tol:.0e}'.replace('e-0', 'e-')] = dict(dt=dt, tolerance=tol)
DEF = dict(init='field', stepping='cn', dt=.025, starts=4, fit_budget=600, step_budget=120, tolerance=1e-6)
BF = dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True)

D2 = dict(dim=2, model=dict(kind='mlp', bank='wide2d/bank.pkl', head='wide2d/head_K8.pkl'), prep='prep_2d.npz',
          times=[0., .1, .2, .3, .4, .5], diffusivity=.02, tests=dict(count=256),
          edges=[0, 16, 32, 48, 64, 96, 128], ladder=[128, 96, 64, 48, 32, 16],
          q_by_Rp={'128': [0, 32, 120], '96': [0, 32, 88], '64': [0, 32, 56], '48': [0, 32, 40], '32': [0, 24], '16': [0, 8]},
          rom_defaults=DEF, families=dict(cn=dict(init='moments'), bf=BF),   # paper h2d-final04 conventions: CN moments init, batched fit field init
          parity=dict(meshes=[1024, 2048, 4096], q=[0, 32, 120]),
          cohorts=[['validation_791001', 791001, 16], ['heldout_sealed_791099', 791099, 16]],
          meshes=[1024, 2048, 4096], cases_by_mesh={}, repetitions=5, burn_seconds=.1, audit_intervals=256,
          full_audit_cases=2, full_audit_max_unknowns=300000, cg_arms=CG,
          neighbour=dict(cg='fom_cncg_dt0.05_rtol1e-3', cohort='heldout_sealed_791099', cases=3, reps=3),
          profile=dict(meshes=[4096]), paper_fast={'cn': 'nmrom_R128_q0_cn', 'bf': 'nmrom_R128_q0_bf'})
D3 = dict(dim=3, model=dict(kind='mlp', bank='vp_R320/bank.pkl', head='vp_R320/head_K32.pkl'), prep='prep_3d.npz',
          times=[0., .1, .2, .3, .4, .5], diffusivity=.02, tests=dict(count=1024),
          edges=[0, 32, 64, 128, 192, 256, 320], ladder=[320, 256, 192, 128, 64, 32],
          q_by_Rp={'320': [0, 144, 288], '256': [0, 112, 224], '192': [0, 80, 160], '128': [0, 48, 96], '64': [0, 16, 32], '32': [0]},
          rom_defaults=DEF, families=dict(cn=dict(init='field'), bf=BF),   # paper final01/final256 panel A conventions
          parity=dict(meshes=[32, 64, 128], q=[0, 144, 288]),
          cohorts=[['validation_921777', 921777, 16], ['heldout_sealed_921099', 921099, 64]],
          meshes=[32, 64, 128], cases_by_mesh={}, repetitions=5, burn_seconds=.1, audit_intervals=32,
          full_audit_cases=2, full_audit_max_unknowns=300000, cg_arms=CG,
          neighbour=dict(cg='fom_cncg_dt0.05_rtol1e-3', cohort='heldout_sealed_921099', cases=3, reps=3),
          profile=dict(meshes=[128]), paper_fast={'cn': 'nmrom_R320_q0_cn', 'bf': 'nmrom_R320_q0_bf'})
# 2026-09-23 addendum (before any 256^3 data): the 3D linear rung at R'=R (0.0695 % at 32^3) is more accurate than every CN-CG
# setting of the grid (the tight dt 0.025 rtol 1e-6 is CN-time-error bound at ~0.078 %), so the rule had no comparator; the 256^3
# grid adds dt 0.0125 (rtol 1e-6, 1e-4). Rule unchanged.
CG256 = dict(CG, **{'fom_cncg_dt0.0125_rtol1e-6': dict(dt=.0125, tolerance=1e-6), 'fom_cncg_dt0.0125_rtol1e-4': dict(dt=.0125, tolerance=1e-4)})
cfgs = {'h2d.json': D2, 'h3d.json': D3, 'h3d256.json': dict(D3, meshes=[256], parity=dict(meshes=[], q=[0, 144, 288]), profile=dict(meshes=[256]), cg_arms=CG256)}
sm_cg = {k: CG[k] for k in ('fom_cncg_dt0.025_rtol1e-6_NAMED', 'fom_cncg_dt0.05_rtol1e-3')}
cfgs['smoke2d.json'] = dict(D2, local_smoke=True, meshes=[64, 128], ladder=[128, 32], q_by_Rp={'128': [0, 32], '32': [0, 24]},
                            parity=dict(meshes=[64, 128], q=[0, 32]), cohorts=[['validation_791001', 791001, 1], ['dev_790711', 790711, 2]],
                            repetitions=1, burn_seconds=.01, audit_intervals=16, cg_arms=sm_cg,
                            neighbour=dict(cg='fom_cncg_dt0.05_rtol1e-3', cohort='dev_790711', cases=1, reps=1), profile=dict(meshes=[128]))
cfgs['smoke3d.json'] = dict(D3, local_smoke=True, meshes=[16], ladder=[320, 64], q_by_Rp={'320': [0, 288], '64': [0, 32]},
                            parity=dict(meshes=[16], q=[0, 288]), cohorts=[['validation_921777', 921777, 1], ['dev_920311', 920311, 1]],
                            repetitions=1, burn_seconds=.01, audit_intervals=8, cg_arms=sm_cg,
                            neighbour=dict(cg='fom_cncg_dt0.05_rtol1e-3', cohort='dev_920311', cases=1, reps=1), profile=dict(meshes=[16]))
for k, v in cfgs.items():
    (here / 'configs' / k).write_text(json.dumps(v, indent=1) + '\n'); print(k)
