"""jcp-mechanism: write the job configs (DESIGN.md section 6) from the frozen 2026-10-01 lane outputs (read-only).

  configs/a1d3_n65.json, a1d3_n129.json   A1 3D (a1_3d.py), validation 923801 x 64, R' = 512 and 256
  configs/a1d2_L256.json, a1d2_L1024.json A1 2D (q2d/qstudy.py), dev6 + val32, acc + fast
  configs/a2q.json                        A2 (a2_gap.py), 3D n in {33, 65, 129, 257}, 2D L in {128, ..., 4096}
  configs/smoke_*.json                    tiny local smoke configs (code paths only, never read as results)

Staged reference paths are relative to the job's TASK_ROOT (refs/..., refs2d/...), so a retry under another job name works.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
Q3 = WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d'
Q2 = WT / '2026-10-01-quadrature-study/experiments/quadrature-study'
NS = '/cluster/tufts/paralab/tawal01/jcpmech'
LANE = 'experiments/jcp-mechanism'
V3 = f'{LANE}/vendor/quad3d'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, cfg):
    (HERE / 'configs').mkdir(exist_ok=True)
    (HERE / 'configs' / name).write_text(json.dumps(cfg, indent=1) + '\n')


def a1d3(n, job='a1d3'):
    old = json.loads((Q3 / f'runs/val{n}/code/output/result.json').read_text())
    oc = old['config']
    exp_rho = {str(Rp): {nm: old['rho'][str(Rp)]['rules'][nm]['cont']['worst']
                         for nm in ('tensor', 'dense_upwind', 'gl24', 'lat4096', 'lat32768')} for Rp in (512, 256)}
    return dict(
        mesh=n, Rps=[512, 256], dt=oc['dt'], gtol=oc['gtol'], trust_fraction=oc['trust_fraction'],
        model=f'{V3}/inputs/model_M2', expected_model_sha256=oc['expected_model_sha256'],
        rules_npz=f'{V3}/rules/rules.npz', expected_rules_sha256=oc['expected_rules_sha256'],
        rules=['gl24', 'lat4096', 'lat32768'], converged_rule='lat32768', continuum_target='gl80',
        continuum_check='gl64', cert_draws=oc['cert_draws'], cohort_seed=923801, cohort_count=64,
        expected_cohort_sha256=old['cohort']['table_sha256'],
        refined_ref='refs/ref_923801.npz',
        expected_refined_sha256=sha(Q3 / 'runs/ref1/code/output/ref_923801.npz'),
        same_grid_ref65=f'refs/same_grid_ref65_n{n}.npz',
        expected_same_grid_sha256=sha(Q3 / f'runs/val{n}/code/output/fields/same_grid_ref65.npz'),
        expected_rho=exp_rho, adaptive_check_cases=8 if n == 65 else 0,
        source_job=dict(name=f'val{n}', job_id=old['job_id'], gpu=old['gpu']))


def a1d2(L, job='a1d2'):
    old = json.loads((Q2 / f'configs/dv{L}.json').read_text())
    cfg = copy.deepcopy(old)
    cfg['attempt'] = f'a1d2_L{L}'
    cfg['cohorts'] = ['dev6', 'val32']
    cfg['settings'] = ['acc', 'fast']
    sel = {'acc': 'gauss96', 'fast': 'fib1597'}
    cfg['arms'] = {s: [dict(name='dense', kind='dense'), dict(name='lat64', kind='mesh', rule='lat64'),
                       dict(name=sel[s], kind='point', rule=sel[s]), dict(name='gref', kind='point', rule='gauss640'),
                       dict(name='nodes', kind='point', rule='nodes')] for s in cfg['settings']}
    cfg['rho_rules'] = {s: [dict(name='dense', kind='dense'), dict(name='lat64', kind='mesh', rule='lat64'),
                            dict(name=sel[s], kind='point', rule=sel[s]),
                            dict(name='fib121393', kind='point', rule='fib121393'),
                            dict(name='gref', kind='point', rule='gauss640'),
                            dict(name='nodes', kind='point', rule='nodes')] for s in cfg['settings']}
    cfg['nodes_reached_rho_cases'] = 8
    cfg['refs'] = 'refs2d'
    cfg['timing'] = dict(skip=True, cohort='dev6', cases=1, reps=1, burn=0.1, seed=20261001, arms={})
    cfg['tangent_chunks'] = {k: v for k, v in old['tangent_chunks'].items() if k in cfg['settings']}
    cfg['role'] = 'jcp-mechanism A1 2D (dev/val only)'
    cfg['source_job'] = f'dv{L}'
    return cfg


def a2q():
    return dict(
        model3d=f'{V3}/inputs/model_M2/bank.pkl', expected_model3d_sha256=sha(Q3 / 'inputs/model_M2/bank.pkl'),
        rules3d=f'{V3}/rules/rules.npz', expected_rules3d_sha256=sha(Q3 / 'rules/rules.npz'),
        Rps3d=[512, 256], meshes3d=[33, 65, 129, 257],
        settings2d=['acc', 'fast'], meshes2d=[128, 256, 512, 1024, 2048, 4096],
        expected_ckpt2d_sha256=sha(WT / '2026-10-06-jcp-mechanism/experiments/separable-decoder/runs/dn256b/out/'
                                        'sep_hfit_dense_mid_N256_dense.pkl'),
        expected_states_sha256=sha(HERE / 'inputs/a2_states.npz'),
        fit_window3d=[65, 129, 257], fit_window2d=[256, 512, 1024, 2048, 4096])


def main():
    for n in (65, 129):
        dump(f'a1d3_n{n}.json', a1d3(n))
    for L in (256, 1024):
        dump(f'a1d2_L{L}.json', a1d2(L))
    dump('a2q.json', a2q())
    # local smoke configs (tiny; code paths only)
    s3 = a1d3(65)
    s3.update(mesh=33, Rps=[256], cohort_count=2, expected_cohort_sha256=None, cert_draws=[[923811, 2]], adaptive_check_cases=2, local_smoke=True,
              refined_ref=str(Q3 / 'runs/ref1/code/output/ref_923801.npz'),
              same_grid_ref65=str(Q3 / 'runs/val65/code/output/fields/same_grid_ref65.npz'),
              expected_same_grid_sha256=sha(Q3 / 'runs/val65/code/output/fields/same_grid_ref65.npz'))
    dump('smoke_a1d3.json', s3)
    s2 = a2q()
    s2.update(meshes3d=[17, 33], Rps3d=[256], meshes2d=[64, 128], settings2d=['fast'], local_smoke=True)
    dump('smoke_a2q.json', s2)
    s2d = a1d2(256)
    s2d['case_subset'] = {'dev6': [0, 1], 'val32': []}
    s2d['settings'] = ['fast']
    s2d['allow_cpu_smoke'] = True
    s2d['refs'] = None                   # local numpy draws differ in the last bits from the cluster's: no refs locally
    s2d['allow_missing_refs'] = True
    s2d['nodes_reached_rho_cases'] = 2
    s2d['audit'] = dict(cohort='dev6', cases=[0], full_max_mesh=256, full_case0_max_mesh=256, full_arms=[])
    dump('smoke_a1d2.json', s2d)


if __name__ == '__main__':
    main()
