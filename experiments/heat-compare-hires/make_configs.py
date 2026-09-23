"""Writes configs/pn{1024,2048,4096}.json and configs/smoke64.json (committed; never hand-edit)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
NS = '/cluster/tufts/paralab/tawal01/hcmp_20260923'
fom = {
    'fom_cncg_dt0.1_rtol1e-2': (0.1, 1e-2), 'fom_cncg_dt0.05_rtol1e-2': (0.05, 1e-2), 'fom_cncg_dt0.025_rtol1e-2': (0.025, 1e-2),
    'fom_cncg_dt0.1_rtol1e-3': (0.1, 1e-3), 'fom_cncg_dt0.05_rtol1e-3': (0.05, 1e-3), 'fom_cncg_dt0.025_rtol1e-3': (0.025, 1e-3),
    'fom_cncg_dt0.05_rtol1e-4': (0.05, 1e-4), 'fom_cncg_dt0.025_rtol1e-4': (0.025, 1e-4),
    'fom_cncg_dt0.025_rtol1e-6_NAMED': (0.025, 1e-6), 'fom_cncg_dt0.0125_rtol1e-6': (0.0125, 1e-6),
    'fom_cncg_dt0.00625_rtol1e-8_TIGHT': (0.00625, 1e-8)}
base = dict(
    times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], diffusivity=0.02, tests=dict(count=256),
    cohorts=[[791099, 16]], cohort_name='sealed (hires-heat h2d-final04, seed 791099, 16 draws; second opening, no choice made on it)',
    train=[791000, 512], validation=[791001, 16], repetitions=5, burn_seconds=0.15, idle_seconds=1.0, sentinel_reps=15,
    order_tolerance=0.10, audit_intervals=128, random_audit_cases=16, random_audit_nodes=50000, audit_sample_seed=20260923,
    factor_gate_draws=[0, 257, 511], control_matmul_size=8192,
    nmrom=dict(model=dict(kind='mlp', bank='wide2d/bank.pkl', head='wide2d/head_K8.pkl'),
               defaults=dict(init='moments', stepping='cn', dt=0.025, starts=4, fit_budget=600, step_budget=120, tolerance=1e-6),
               arms=[dict(name='nmrom_q0_cn', q=0), dict(name='nmrom_q32_cn', q=32),
                     dict(name='nmrom_q0_field_direct_tol1e-4_chol', q=0, opt=dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True)),
                     dict(name='nmrom_q32_field_direct_tol1e-4_chol', q=32, opt=dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True))],
               linear_bank_inits=['field', 'moments'], linear_bank_cn=True),
    pod=dict(ranks=[8, 32, 128, 256], methods=['galerkin_cn', 'lspg_cn', 'galerkin_exact'], dt=0.025),
    qm=dict(ranks=[8, 16, 32], split_seed=20260923, holdout_fraction=0.2,
            arms=[dict(name='cn'), dict(name='field_direct_tol1e-4_chol', opt=dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True))]),
    cg_arms={k: dict(dt=v[0], tolerance=v[1]) for k, v in fom.items()}, fom_order=list(fom),
    coarse_intervals=[64], coarse_cg='fom_cncg_dt0.05_rtol1e-3',
    retime=dict(nmrom='nmrom_q32_field_direct_tol1e-4_chol', pod=128, operator='op_unet'), carryover_tolerance=0.10, retime_tolerance=0.10)
ops = ['fno', 'unet', 'transolver', 'deeponet']
TRAIN = {1024: 'tr1024a', 2048: 'tr2048'}
for n in (1024, 2048, 4096):
    c = dict(base, mesh=n)
    c['operators'] = {} if n == 4096 else {f'op_{o}': dict(checkpoint=f'{NS}/{TRAIN[n]}/out/{o}/best.pt', family=o) for o in ops}
    (HERE / 'configs' / f'pn{n}.json').write_text(json.dumps(c, indent=1) + '\n')
smoke = dict(base, mesh=64, cohorts=[[791099, 2]], train=[791000, 32], repetitions=2, sentinel_reps=3, idle_seconds=0.1, local_smoke=True,
             factor_gate_draws=[0, 31], retime=dict(nmrom='nmrom_q32_field_direct_tol1e-4_chol', pod=32, operator='op_unet'), pod=dict(base['pod'], ranks=[8, 32]), qm=dict(base['qm'], ranks=[8]),
             audit_intervals=32, random_audit_nodes=500, control_matmul_size=1024,
             operators={f'op_{o}': dict(checkpoint=f'/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6777170d-0e90-4670-b7cb-4ee006705422/scratchpad/opsmoke1/{o}/best.pt', family=o) for o in ops})
(HERE / 'configs' / 'smoke64.json').write_text(json.dumps(smoke, indent=1) + '\n')
