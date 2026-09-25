"""Write config-t{256,1024,2048}.json, config-d256.json and config-smoke64.json (edit this generator, never the JSON).

Every config is burgers-compare-hires/make_configs.config(L) @ 07c0e3e8 (the driver config of the Table 2
1024^2 / 2048^2 cells, jobs 4204019 / 4218390) with ONLY these changes (DESIGN.md section 4):
  * cohort = test64 (params_draw(20260916, 64), cluster sha256 cd058fd2...) -- d256: dev6, the development cohort;
  * arms reduced to the two Table 2 NM-ROM settings: the head-only q=0 (k=16, M=64, `scaled` rule, g 1e-3) in
    Phase F and the bank-span R'=384 (M=1536, lat64, g 1e-6) block in Phase S; q=256, the other R', lat128,
    POD-LSPG and the quadratic manifold are not run (they are not in Table 2);
  * field storage: sub-grid for every case, full field for case 0 only;
  * 256^2 (not in the source lane): intervals 256, parity mesh 256, q=0 rule used at its own mesh.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'burgers-compare-hires'))
import make_configs as MC            # noqa: E402  (the source lane's generator, unchanged)

TEST64 = dict(name='test64 = hold64: params_draw(20260916, 64), the 64 held-out test cases (burgers2d-test lane)',
              split='test', draws=[[20260916, 64]],
              expected_physical_sha256='cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c')
DEV6 = dict(name='dev6: params_draw(7090702,4) + params_draw(911702,2), the development cases of Table 2',
            split='development', draws=[[7090702, 4], [911702, 2]],
            expected_physical_sha256='108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a')
ACC = 'bank384_M1536_lat64_g1em06'
FAST = 'q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2'
STATUS256 = 'b-eqtop q=0 rule, m=1024, built at 256^2; confirmed (3/3) at 256^2'


def config(L, cohort, attempt):
    c = MC.config(L if L in (1024, 2048) else 1024)
    c.update(intervals=L, attempt=attempt, parity_mesh=(512 if L >= 2048 else L))
    r0 = MC.rungs(L)[0]
    if L == 256:
        r0['rules'][0]['status'] = STATUS256
    c['rungs'] = [r0]
    c['cohort'] = cohort
    c['expected_physical_sha256'] = cohort['expected_physical_sha256']
    for k in ('eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases'):
        c.pop(k)
    c['disjoint_from'] = ([[7090702, 4, 'dev6_a'], [911702, 2, 'dev6_b']] if cohort['split'] == 'test'
                          else [[20260916, 64, 'test64']])
    c['operator_split_indices'] = ['../burgers-compare-hires/inputs/pinned/train-index.json',
                                   '../burgers-compare-hires/inputs/pinned/validation-index.json']
    c.update(pod_ranks=[], qman_ranks=[], grid_parity_ranks=[], sub_points=64, full_field_cases=[0])
    c['bank_span'] = dict(c['bank_span'], ranks=[384], rules=['lat64'])
    c['roles'] = dict(nmrom_fast=FAST, nmrom_accurate=ACC)
    c['reference_roles'] = []
    c['purpose'] = (f't2-burgers-test DESIGN.md: the Table 2 Burgers 2D cell at {L}^2 re-run on {cohort["split"]} '
                    f'cases in ONE allocation (burgers-compare-hires driver).')
    return c


def smoke():
    """64^2, one development case, one repetition: exercises every code path locally. Not a result."""
    c = config(64, DEV6, 'smoke64')
    s = MC.smoke()
    c.update(case_subset=[0], reps=1, required_reps=1, bracket_reps=1, burn_seconds=0., local_smoke_waives_cohort_hash=True,
             train_trajectories=10, fom_settings=s['fom_settings'], fom_subsets=s['fom_subsets'],
             phase_s_deadline_seconds=3600, sub_points=16)
    c['rungs'] = [MC.rungs(64, smoke=True)[0]]
    c['bank_span'] = dict(c['bank_span'], ranks=[32], rules=['lat16'], population_rows=[0, 2])
    c['roles'] = dict(nmrom_fast=FAST.replace('_scaled_', '_lat16_'), nmrom_accurate='bank32_M128_lat16_g1em06')
    return c


if __name__ == '__main__':
    for L in (256, 1024, 2048):
        (HERE / f'config-t{L}.json').write_text(json.dumps(config(L, TEST64, f't{L}'), indent=1) + '\n')
    (HERE / 'config-d256.json').write_text(json.dumps(config(256, DEV6, 'd256'), indent=1) + '\n')
    (HERE / 'config-smoke64.json').write_text(json.dumps(smoke(), indent=1) + '\n')
    print('written')
