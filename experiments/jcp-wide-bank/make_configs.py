"""Write the lane's job configs (DESIGN.md + A0/A1). `python make_configs.py` regenerates configs/*.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'configs'
REF2D_CLUSTER = '/cluster/tufts/paralab/tawal01/jcpwide/refs2d'
REF2D_LOCAL = str(HERE.parents[2] / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output')

GAUSS2D = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
FIB2D = [987, 1597, 4181, 6765, 17711, 46368]


def arms2d(gauss=GAUSS2D, fib=FIB2D):
    a = [dict(name=f'gauss{p}', family='gauss', rule=f'gauss{p}') for p in gauss]
    a += [dict(name=f'fib{n}', family='fib', rule=f'fib{n}') for n in fib]
    a += [dict(name='ctrl_gauss8', family='ctrl', rule='gauss8', control=True),
          dict(name='ctrl_smolyak8', family='ctrl', rule='smolyak8', control=True)]
    return a


def base2d(**kw):
    c = dict(mesh=1024, cohorts=['dev6', 'val32'],
             settings=[dict(Rp=r, M=k * r, kappa=k) for r in (128, 256, 384, 512) for k in (2, 3, 4)],
             arms=arms2d(), families=['gauss', 'fib'], gref='gauss640', gref_check='gauss768',
             tau=2.5e-4, tau_secondary=1e-3, rho_bar=0.116, budget_fraction=0.01, conv_bar=2.5e-5, target_bar=1e-5,
             step_budget=600, gtol=1e-3, point_chunk=32768,
             refs=REF2D_CLUSTER,
             ref_contract=dict(mesh=8192, refs=dict(ST=dict(dt=0.0003125, ntol=1e-11, ltol=1e-9, accept_residual=2e-11),
                                                    S=dict(dt=0.005, ntol=1e-11, ltol=1e-9, accept_residual=2e-11))),
             expected_checkpoint_sha256='18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589',
             expected_rotation_sha256='51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772',
             expected_qcore_sha256='36d6551ad0f12746e96d409bb11878a78ef1c05c6cdbc9d5f58b3c666c7c0e0b',
             timing=dict(cohort='dev6', cases=6, reps=2, burn=0.1, seed=20261008),
             audit_cohort='dev6', audit_cases=[0, 2], audit_full_arms=[], audit_rho_arm='gauss64')
    c.update(kw)
    return c


def main():
    OUT.mkdir(exist_ok=True)
    cfgs = {
        # J1: the full 2D grid (1a + 1d), validation cohorts, 1024^2
        'j1_2d.json': base2d(attempt='j1'),
        # real-size smoke on the cluster: the largest and the smallest setting, two dev6 cases, every arm (A0-11)
        'smoke_2d.json': base2d(attempt='s1', settings=[dict(Rp=512, M=2048, kappa=4), dict(Rp=128, M=256, kappa=2)],
                                cohorts=['dev6'], case_subset=dict(dev6=[0, 1]),
                                timing=dict(cohort='dev6', cases=2, reps=1, burn=0.1, seed=20261008)),
        # local functional test (GB10, jaxrun): small mesh, small converged rules, NOT a result
        'local_2d.json': base2d(attempt='l1', mesh=256, settings=[dict(Rp=128, M=512, kappa=4), dict(Rp=128, M=256, kappa=2)],
                                cohorts=['dev6'], case_subset=dict(dev6=[0]), arms=arms2d([16, 32, 64], [987, 4181]),
                                gref='gauss160', gref_check='gauss192', refs=REF2D_LOCAL,
                                timing=dict(cohort='dev6', cases=1, reps=1, burn=0.05, seed=1), audit_cases=[0],
                                local_waive_cohort_hash=True, allow_cpu_smoke=True),
    }
    for name, c in cfgs.items():
        (OUT / name).write_text(json.dumps(c, indent=1) + '\n')
        print(name, len(c['settings']), 'settings')


if __name__ == '__main__':
    main()
