"""Write config-{1024,2048}.json and config-smoke64.json for t3-burgers-test (edit this generator, never the JSON).

Every value starts from the parent's generator (burgers-compare-hires/make_configs.py @ 07c0e3e8, the configs that
produced paper Table 3's 1024^2 / 2048^2 cells: p1024 job 4204019, p2048e job 4218390) and only these keys change
(DESIGN.md section 3):
  * cohort: dev6 -> test64 = params_draw(20260916, 64), expected physical sha256 cd058fd2...
  * arms restricted to the Table-3 rows: NM-ROM head q=0 (Phase F), bank-span R'=384 `lat64` (Phase S), POD-LSPG
    k=16, quadratic manifold r=16 (+ their untimed parity twins); the 15-setting Newton-BiCGStab grid unchanged.
  * pod_fit_kmax = 512 (the parent's POD eigenproblem size, so the basis can be compared with the parent's record).
  * audit_cases, the Phase-S deadline, the attempt name.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('parent_mc', HERE.parent / 'burgers-compare-hires' / 'make_configs.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

TEST64 = dict(cohort_draws=[[20260916, 64]],
              cohort_name='test64 = hold64: params_draw(20260916, 64), the held-out Burgers 2D TEST cases',
              expected_physical_sha256='cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c')
DROP = ('eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases')


def config(L):
    c = parent.config(L)
    for k in DROP:
        c.pop(k)
    c.update(TEST64)
    c['attempt'] = f't3{L}'
    c['rungs'] = [r for r in c['rungs'] if r['q'] == 0]           # head only in Phase F; no q=256 arms
    c['pod_ranks'] = [16]
    c['pod_fit_kmax'] = 512
    c['qman_ranks'] = [16]
    c['grid_parity_ranks'] = [16]
    c['bank_span'] = dict(c['bank_span'], ranks=[384], rules=['lat64'],
                          note='held-out rho population = burgers-eqcert source draw rows 0-7, disjoint from test64/train')
    c['roles'] = dict(nmrom_fast=c['roles']['nmrom_fast'], nmrom_accurate='bank384_M1536_lat64_g1em06')
    c['reference_roles'] = []
    c['audit_cases'] = [0, 1]
    c['phase_s_deadline_seconds'] = (9 if L >= 2048 else 6) * 3600
    c['table3_rule'] = ('one FOM per cell: the fastest Phase-F-timed setting of the full grid whose worst evolved error '
                        '<= the worst evolved error of the NM-ROM accurate row (roles.nmrom_accurate), same job')
    c['purpose'] = f'experiments/t3-burgers-test/DESIGN.md: paper Table 3 Burgers 2D cells at {L}^2 on test64, one allocation.'
    return c


def smoke():
    """64^2 local smoke: 3 cases, 1 repetition, reduced grids; exercises every code path. Not a result."""
    c = config(64)
    c.update(attempt='smoke64', intervals=64, cohort_draws=[[20260916, 3]], local_smoke_waives_cohort_hash=True,
             reps=1, required_reps=1, bracket_reps=1, burn_seconds=0., train_trajectories=10,
             fom_settings=[f for f in c['fom_settings'] if f['name'] in ('fft_tight', 'nt1e-2_dt01', 'lean_nt3e-3_l3e-3_dt005')],
             pod_ranks=[4], pod_fit_kmax=16, qman_ranks=[4], grid_parity_ranks=[4], grid_tangent_chunk=4,
             fit_block_bytes=8 * 260 * 97, qman_holdout=.3, phase_s_deadline_seconds=3600, parity_mesh=64, audit_cases=[0])
    c['rungs'] = [dict(q=0, M=64, rules=[dict(name='lat16', parts=[dict(lattice=16)], phase='F', status='smoke',
                                              variants=[dict(solver='lu', clip=True, lamcarry=True, pred2=True, gtols=[1e-3])])])]
    c['bank_span'] = dict(c['bank_span'], ranks=[32], rules=['lat16'], population_rows=[0, 2])
    c['roles'] = dict(nmrom_fast='q0_M64_lat16_g0p001_fast_clip_lamcarry_pred2', nmrom_accurate='bank32_M128_lat16_g1em06')
    kept = {f['name'] for f in c['fom_settings']}
    c['fom_subsets'] = {k: [n for n in v if n in kept] for k, v in c['fom_subsets'].items()}
    return c


if __name__ == '__main__':
    for L in (1024, 2048):
        (HERE / f'config-{L}.json').write_text(json.dumps(config(L), indent=1) + '\n')
    (HERE / 'config-smoke64.json').write_text(json.dumps(smoke(), indent=1) + '\n')
    print('written')
