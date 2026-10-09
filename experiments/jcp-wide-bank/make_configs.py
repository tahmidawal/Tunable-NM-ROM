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


def ref_shas():
    import hashlib
    d = Path(REF2D_LOCAL)
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir()) if f.is_file()}


def base2d(**kw):
    c = dict(mesh=1024, cohorts=['dev6', 'val32'],
             settings=[dict(Rp=r, M=k * r, kappa=k) for r in (128, 256, 384, 512) for k in (2, 3, 4)],
             arms=arms2d(), families=['gauss', 'fib'], gref='gauss640', gref_check='gauss768',
             tau=2.5e-4, tau_secondary=1e-3, rho_bar=0.116, budget_fraction=0.01, conv_bar=2.5e-5, target_bar=1e-5,
             step_budget=600, gtol=1e-3, point_chunk=32768,
             refs=REF2D_CLUSTER, refs_files_sha256=ref_shas(),
             ref_contract=dict(mesh=8192, refs=dict(ST=dict(dt=0.0003125, ntol=1e-11, ltol=1e-9, accept_residual=2e-11),
                                                    S=dict(dt=0.005, ntol=1e-11, ltol=1e-9, accept_residual=2e-11))),
             expected_checkpoint_sha256='18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589',
             expected_rotation_sha256='51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772',
             expected_qcore_sha256='36d6551ad0f12746e96d409bb11878a78ef1c05c6cdbc9d5f58b3c666c7c0e0b',
             timing=dict(cohort='dev6', cases=6, reps=2, burn=0.1, seed=20261008),
             audit_cohort='dev6', audit_cases=[0, 2], audit_full_arms=[], audit_rho_arm='gauss64')
    c.update(kw)
    return c


REF3D_CLUSTER = '/cluster/tufts/paralab/tawal01/jcpwide/refs3d/ref_923801.npz'
RULES3D = [['experiments/jcp-wide-bank/vendor/quad3d/rules/rules.npz',
            '099a79976d03e993dc3fbdcc153a57dae0a2dbc5f917d7ca3674bef28daf859b'],
           ['experiments/jcp-wide-bank/rules3d/rules_extra.npz', None]]       # sha filled from rules_extra.json
M2 = dict(name='M2', model='experiments/jcp-wide-bank/vendor/quad3d/inputs/model_M2',
          expected_sha256='6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9')


def ladder3d(lats, gauss):
    a = [dict(name=f'lat{n}', family='lat') for n in lats] + [dict(name=f'gl{p}', family='gauss') for p in gauss]
    return a + [dict(name='lat256', family='ctrl', control=True), dict(name='smol8', family='ctrl', control=True)]


def base3d(**kw):
    ex = HERE / 'rules3d' / 'rules_extra.json'
    rules = [list(r) for r in RULES3D]
    import hashlib
    meta = json.loads(ex.read_text())                    # fails closed if the rules were not generated
    npz = HERE / 'rules3d' / 'rules_extra.npz'
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == meta['npz_sha256'], 'rules_extra.npz differs from its metadata'
    rules[1][1] = meta['npz_sha256']
    c = dict(meshes=[65], banks=[dict(M2, Rps=[512, 256], kappas=[4, 3, 2], tensor=True)], rules_files=rules,
             ladder=ladder3d([2048, 4096, 8192, 16384, 32768, 65536], [12, 16, 20, 24, 32, 40]),
             converged='gl48', check='gl40', target='gl80', target_check='gl64', families=['lat', 'gauss'],
             cohort_seed=923801, cohort_count=64, cert_draws=[[923811, 8], [923812, 8], [923813, 8]],
             refined_ref=REF3D_CLUSTER, expected_ref=dict(n=513, dt=0.0025, ntol=1e-06, ltol=1e-07, accepted=True),
             tau=2.5e-4, tau_secondary=1e-3, rho_bar=0.116, nonstat_fraction=0.01, conv_bar=2.5e-5, target_bar=1e-5,
             dt=0.01, gtol=1e-3, trust_fraction=0.05, fom_timing=[0.01, 0.1], timing_cases=16, reps=2,
             timing_seed=20261008, audit_cases=[0, 1], audit_rho_arm='lat4096', jacobian_states=[1, 12, 25],
             expected_offmesh_sha256='c46656e48cec27c4d78d1a00f3d93e19b7911b6d34bae6bdf9e264d10d45cbb4')
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
    cfgs['j2_3d.json'] = base3d(attempt='j2')
    cfgs['smoke_3d.json'] = base3d(attempt='s2', banks=[dict(M2, Rps=[512], kappas=[4], tensor=True)],
                                   case_subset=[0, 1, 2, 3], cert_draws=[[923811, 4]], timing_cases=2, reps=1)
    # local functional test of w3d.py (GB10, jaxrun): tiny rules and cohort, NOT a result
    cfgs['local_3d.json'] = base3d(attempt='l3', banks=[dict(M2, Rps=[256], kappas=[2], tensor=True)],
                                   ladder=ladder3d([2048, 4096], [12, 16]), converged='gl24', check='gl16',
                                   target='gl32', target_check='gl24', case_subset=[0, 1], cert_draws=[[923811, 2]],
                                   timing_cases=1, reps=1, audit_cases=[0], local_smoke=True,
                                   refined_ref=str(HERE.parents[2] / '2026-10-01-quadrature-burgers3d/experiments/'
                                                   'quadrature-burgers3d/runs/ref1/code/output/ref_923801.npz'))
    wb = HERE / 'inputs' / 'model_W1024' / 'bank.pkl'
    if wb.exists():                                       # J4 only after J3's bank is committed (DESIGN A1)
        import hashlib
        W = dict(name='W1024', model='experiments/jcp-wide-bank/inputs/model_W1024',
                 expected_sha256=hashlib.sha256(wb.read_bytes()).hexdigest())
        j4 = dict(meshes=[65, 129], banks=[dict(W, Rps=[1024, 768, 512, 256], kappas=[4], tensor=True),
                                           dict(M2, Rps=[512, 256], kappas=[4], tensor=True)],
                  ladder=ladder3d([4096, 8192, 16384, 32768, 65536, 131072], [16, 20, 24, 32, 40, 48]),
                  converged='gl56', check='gl48', audit_rho_arm='lat8192', adaptive_first=6)          # DESIGN A8
        cfgs['j4_3d.json'] = base3d(attempt='j4', **j4)
        cfgs['smoke4_3d.json'] = base3d(attempt='s4', **dict(j4, meshes=[129], banks=[dict(W, Rps=[1024], kappas=[4], tensor=False)],
                                                              case_subset=[0, 1, 2, 3], cert_draws=[[923811, 4]], timing_cases=2, reps=1))
    for name, c in cfgs.items():
        (OUT / name).write_text(json.dumps(c, indent=1) + '\n')
        print(name)


if __name__ == '__main__':
    main()
