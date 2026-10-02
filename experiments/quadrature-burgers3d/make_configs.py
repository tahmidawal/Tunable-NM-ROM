"""Write every job config of DESIGN.md section 9 into configs/ (deterministic; rerun to regenerate)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = '/cluster/tufts/paralab/tawal01/quad3d_20261001'
RULES_SHA = hashlib.sha256((HERE / 'rules' / 'rules.npz').read_bytes()).hexdigest()
BASE = dict(
    Rps=[512, 256], dt=0.01, gtol=1e-3, trust_fraction=0.05,
    model='experiments/quadrature-burgers3d/inputs/model_M2',
    expected_model_sha256='6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9',
    rules_npz='experiments/quadrature-burgers3d/rules/rules.npz', expected_rules_sha256=RULES_SHA,
    rules=['lat4096', 'lat8192', 'lat16384', 'lat32768', 'kor16381', 'gl16', 'gl24', 'gl32', 'sob16384',
           'lat256', 'smol8'],
    controls=['lat256', 'smol8'],
    continuum_target='gl80', continuum_check='gl64', converged_rule='lat32768', audit_rule='lat4096',
    cert_draws=[[923811, 8], [923812, 8], [923813, 8]],
    ref_ntol=1e-10, ref_ltol=1e-11,
    fom_grid=[[0.005, 0.01, 0.1], [0.005, 0.001, 0.1], [0.01, 0.01, 0.1], [0.01, 0.001, 0.1],
              [0.025, 0.01, 0.1], [0.025, 0.001, 0.1]],
    timing_cases=16, reps=3, refined_wait_hours=1.5)
DENSE = {65: 'all', 129: 4, 257: 0}


def write(name, cfg):
    (HERE / 'configs' / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')


for n in (65, 129, 257):
    write(f'val_n{n}', dict(BASE, mesh=n, dense_cases=DENSE[n], cohort_seed=923801, cohort_count=64,
                            refined_ref=f'{NS}/ref1/code/output/ref_923801.npz'))
write('smoke_n65', dict(BASE, mesh=65, dense_cases=2, cohort_seed=923651, cohort_count=4,
                        cert_draws=[[923811, 2]], timing_cases=2, reps=2, refined_wait_hours=0))
write('ref_probe', dict(mode='probe', n_ref=513, dt_ref=0.0025, probe_seed=923651,
                        tolerances=[[1e-6, 1e-7], [1e-8, 1e-9], [1e-10, 1e-11]], compare_meshes=[257]))
