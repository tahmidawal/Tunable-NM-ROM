"""Write config-{1024,2048}.json and config-smoke64.json (edit this generator, never the JSON).

The meshes differ only in `intervals`, the rule-status labels, the parity mesh for the Phi-free operator gate
(an explicit Phi at 2048^2 and M = 1088 is 36 GB, so that gate runs at 512^2 there, as burgers-eqcert did) and the
job's bank blocking. Everything else -- cohort, checkpoint, arms, FOM grid, repetitions, fits, gates -- is identical.
"""
import json
from pathlib import Path

Q0 = dict(file='rules-eqtop/rule_q0_m1024_qrg304_reachable.npz',
          sha256='6a32568f7b176b8774b31c8c96abf3da61b88dcf8b5ad657ced3f7c840ebe165', mesh=256,
          status='b-eqtop q=0 rule, m=1024; confirmed (3/3) at 256^2')
OPT = dict(solver='chol', clip=True, lamcarry=True, pred2=True)

FOM = [dict(name='fft_tight', dt=.005, ntol=1e-6, ltol=1e-8),
       dict(name='nt1e-4_dt005', dt=.005, ntol=1e-4, ltol=1e-6),
       dict(name='nt1e-3_dt005', dt=.005, ntol=1e-3, ltol=1e-5),
       dict(name='nt1e-2_dt005', dt=.005, ntol=1e-2, ltol=.5),
       dict(name='nt1e-4_dt01', dt=.01, ntol=1e-4, ltol=1e-6),
       dict(name='nt1e-3_dt01', dt=.01, ntol=1e-3, ltol=1e-5),
       dict(name='nt1e-2_dt01', dt=.01, ntol=1e-2, ltol=.5),
       dict(name='lean_tight', dt=.005, ntol=1e-6, ltol=1e-8, impl='lean'),
       dict(name='lean_nt1e-4_dt005', dt=.005, ntol=1e-4, ltol=1e-6, impl='lean'),
       dict(name='lean_nt1e-3_l1e-3_dt005', dt=.005, ntol=1e-3, ltol=1e-3, impl='lean'),
       dict(name='lean_nt3e-3_l3e-3_dt005', dt=.005, ntol=3e-3, ltol=3e-3, impl='lean'),
       dict(name='lean_nt1e-2_l1e-2_dt005', dt=.005, ntol=1e-2, ltol=1e-2, impl='lean'),
       dict(name='lean_nt1e-3_l1e-3_dt01', dt=.01, ntol=1e-3, ltol=1e-3, impl='lean'),
       dict(name='lean_nt3e-3_l3e-3_dt01', dt=.01, ntol=3e-3, ltol=3e-3, impl='lean'),
       dict(name='lean_nt1e-2_l1e-2_dt01', dt=.01, ntol=1e-2, ltol=1e-2, impl='lean')]

STATUS = {
    1024: dict(
        q0='confirmed at 1024^2 (burgers-eqcert bc1024: confirmation 0.0250, bar 0.116)',
        lat64='CERTIFIED at 1024^2 (burgers-eqcert bc1024): 5/5 held-out draws, rho_max 0.0958, confirmation 0.1157 '
              'against a bar of 0.116 -- thin; population-dependent (0.222 on other trajectories, unaudited exploration)',
        lat64_x1='confirmed at 1024^2 (burgers-eqcert bc1024, the robust alternative): rho_max 0.0358, confirmation 0.0498'),
    2048: dict(
        q0='confirmed at 2048^2 (hires-burgers hb2k02: held-out rho 0.0404, deployed 0.0424, bar 0.116)',
        lat64='MARGINAL at 2048^2: passes burgers-eqcert bc2048b draws (rho_max 0.0794, confirmation 0.1156 vs bar 0.116 '
              '-- thin) but reaches 0.183 on other trajectories (unaudited exploration); labelled marginal',
        lat64_x1='confirmed at 2048^2 (burgers-eqcert bc2048b: j=1 confirmation 0.051)'),
}


def rungs(L, smoke=False):
    lat = 'lat16' if smoke else 'lat64'
    latp = [dict(lattice=16 if smoke else 64)]
    st = STATUS.get(L, STATUS[1024])
    q0rule = dict(name='lat16', parts=[dict(lattice=16)]) if smoke else dict(name='scaled', parts=[Q0])
    r0 = dict(q=0, M=64, rules=[dict(q0rule, phase='F', status=st['q0'],
                                     variants=[dict(solver='lu', clip=True, lamcarry=True, pred2=True, gtols=[1e-3])])])
    r256 = dict(q=256, M=1088, rules=[
        dict(name=lat, parts=latp, phase='F', status=st['lat64'], variants=[dict(OPT, gtols=[1e-3])]),
        dict(name=lat + 'x', parts=latp, phase='S', status=st['lat64_x1'],
             variants=[dict(OPT, exact_steps=1, gtols=[1e-2])])])
    return [r0, r256]


def config(L):
    return dict(
        attempt=f'p{L}', intervals=L, dt=.005,
        eval_seed=7090702, eval_cases=4, eval_fresh_seed=911702, eval_fresh_cases=2,
        expected_physical_sha256='108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a',
        train_seed=0, train_trajectories=128, train_state_stride=2, snapshot_ntol=1e-9, snapshot_ltol=1e-7,
        decoder_code_subsample=8192, gauss_jordan_max=64, cold_axis_points=48,
        strict=dict(ic_budget=400, step_budget=600, gtol=1e-6), ic_gtol=1e-6, inner_damping=1e-10,
        directions_file='directions_qtd02.npz',
        directions_sha256='79d794580dad15c03630470940361d8ba9a7779673be9697c90d28e5c3d81535',
        dense_tangent_group=34, parity_mesh=(512 if L >= 2048 else L), bank_blocks=None,
        rungs=rungs(L), fom_settings=FOM, same_grid_reference='fft_tight',
        reps=5, required_reps=5, bracket_reps=2, order_seed=20260923, burn_seconds=.25,
        pod_ranks=[16, 64, 256, 512], qman_ranks=[16, 32, 64],
        qman_gammas=[0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.], qman_seed=20260922, qman_holdout=.2,
        qman_cold_axis_points=96, fit_block_bytes=2e9, grid_tangent_chunk=64, grid_parity_ranks=[16],
        rho_bar=.116, target_chunk=4, phase_s_deadline_seconds=(15 if L >= 2048 else 7) * 3600,
        bank_span=dict(ranks=[512, 384, 256, 128], rules=['lat64', 'lat128'], tests_per_unknown=4, gtol=1e-6,
                       rotation_file='burgers-compare-hires/inputs/rotation_R512.npz',
                       rotation_sha256='51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772',
                       rotation_source=('worktrees/2026-09-23-burgers-bank-knob/experiments/burgers-bank-knob/inputs/'
                                        'rotation_R512.npz (sibling lane, make_rotation.py; training codes only)'),
                       population_draw=[20260921, 56], population_rows=[0, 8],
                       note='held-out rho population = burgers-eqcert source draw rows 0-7, disjoint from dev6/train'),
        roles=dict(nmrom_fast='q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2',
                   nmrom_accurate='q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2',
                   nmrom_accurate_robust='q256_M1088_lat64x_g0p01_fast_chol_clip_lamcarry_pred2_x1'),
        reference_roles=['nmrom_accurate', 'nmrom_accurate_robust'],
        fom_subsets=dict(b_panel=[f['name'] for f in FOM if not f['name'].startswith('lean_')],
                         lean=[f['name'] for f in FOM if f['name'].startswith('lean_')],
                         full=[f['name'] for f in FOM]),
        purpose=f'DESIGN.md: the Burgers 2D comparison panel at {L}^2 in ONE allocation.')


def smoke():
    """64^2, one case, one repetition, reduced grids: exercises every code path (both phases, fits, twins, bracket).
    Local only; not a result."""
    c = config(64)
    c.update(attempt='smoke64', intervals=64, case_subset=[0], reps=1, required_reps=1, bracket_reps=1, burn_seconds=0.,
             local_smoke_waives_cohort_hash=True, rungs=rungs(64, smoke=True), train_trajectories=10,
             fom_settings=[f for f in FOM if f['name'] in ('fft_tight', 'nt1e-2_dt01', 'lean_nt3e-3_l3e-3_dt005')],
             pod_ranks=[4, 16], qman_ranks=[4, 8], grid_parity_ranks=[4, 8], grid_tangent_chunk=4,
             fit_block_bytes=8 * 260 * 97, qman_holdout=.3, phase_s_deadline_seconds=3600)
    c['bank_span'] = dict(c['bank_span'], ranks=[512, 32], rules=['lat16', 'lat32'], population_rows=[0, 2])
    c['roles'] = {k: v.replace('_scaled_', '_lat16_').replace('lat64', 'lat16') for k, v in c['roles'].items()}
    kept = {f['name'] for f in c['fom_settings']}
    c['fom_subsets'] = {k: [n for n in v if n in kept] for k, v in c['fom_subsets'].items()}
    return c


if __name__ == '__main__':
    here = Path(__file__).parent
    for L in (1024, 2048):
        (here / f'config-{L}.json').write_text(json.dumps(config(L), indent=1) + '\n')
    (here / 'config-smoke64.json').write_text(json.dumps(smoke(), indent=1) + '\n')
    print('written')
