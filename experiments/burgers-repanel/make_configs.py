"""Write config-{256,512,1024}.json. One generator so the three meshes differ only where they must.

    python make_configs.py

Differences between meshes, and only these:
  * `intervals`;
  * which quadrature rule the OPTIMISED accurate arm uses (the burgers-eqcert lane's confirmed rule at that
    mesh: lat64 with one exact first step at 256^2/512^2, lat64 with none at 1024^2) and at which tolerance;
  * whether the dense (exact-residual) pre-optimisation arm is timed (1024^2 only: that is the arm the paper
    prints there).
Everything else -- cohort, checkpoint, rules, FOM grid, repetitions, gates -- is identical.
"""
import json
from pathlib import Path

Q256 = dict(file='rules-eqtop/rule_q256_m2560_bet101_fs64.npz',
            sha256='3603d6e18cc799dd4c81c9f92843b607aefc65537665d93622e62d1936adc12e', mesh=256,
            status='b-eqtop q=256 rule, m=2560; the rule family the paper\'s current 256^2/512^2 accurate rows use')
Q0 = dict(file='rules-eqtop/rule_q0_m1024_qrg304_reachable.npz',
          sha256='6a32568f7b176b8774b31c8c96abf3da61b88dcf8b5ad657ced3f7c840ebe165', mesh=256,
          status='b-eqtop q=0 rule, m=1024; confirmed (3/3) at 256^2')

OPT = dict(solver='chol', clip=True, lamcarry=True, pred2=True)      # the full optimisation stack

# The full-order candidate grid: the b-panel settings the current rows were selected from (audited
# implementation, `nt*`) UNION the lean settings the 2048^2/4096^2 rows are selected from. Both are timed in
# this one job, so the paper's FOM rule can be applied like-for-like on either grid.
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

# Certificate status of every deployed rule, carried verbatim from the burgers-eqcert lane
# (reports/2026-09-21-burgers-eqcert.md, jobs bc256 4139288 / bc256b 4143154 / bc512 4140818 / bc1024 4139290).
# This lane RE-TIMES those frozen rules; it does not re-certify or re-select them.
STATUS = {
    256: {'lat64_x1_g0p01': 'CERTIFIED at 256^2 under addendum A2.1: passes 12/12 held-out draws across bc256 '
                            '(rho_max 0.063) and bc256b (0.073), confirmation draw 0.054/0.068, bar 0.116',
          'scaled': 'NOT certified: selected in bc256 on draws 1-5 (rho_max 0.0994) but FAILED the confirmation '
                    'draw (0.165); marginal 3/5 in bc256b. This is the rule the paper\'s current 256^2 row uses.',
          'q0_scaled': 'confirmed at 256^2 (bc256 0.0336 / bc256b 0.0571 confirmation, bar 0.116)',
          'exact': 'no rule: dense residual over all interior nodes'},
    512: {'lat64_x1_g0p01': 'CERTIFIED at 512^2 (bc512): passes 5/5 held-out draws, rho_max 0.0416 held-out / '
                            '0.0400 deployed, confirmation draw 0.0518, bar 0.116',
          'scaled': 'NOT certified at 512^2 (burgers-eqcert verdict). The paper\'s current 512^2 row uses a '
                    'weight-refit (eqtopxfer) of this same support; this job times the no-refit transfer of it.',
          'q0_scaled': 'confirmed at 512^2 (bc512 confirmation 0.0248, bar 0.116)',
          'exact': 'no rule: dense residual over all interior nodes'},
    1024: {'lat64_g0p001': 'CERTIFIED at 1024^2 (bc1024): passes 5/5 held-out draws, rho_max 0.0958 held-out / '
                           '0.0958 deployed, confirmation draw 0.1157 against a bar of 0.116 -- thin',
           'scaled': 'NOT certified at 1024^2 (burgers-eqcert verdict)',
           'q0_scaled': 'confirmed at 1024^2 (bc1024 confirmation 0.0250, bar 0.116)',
           'exact': 'no rule: dense residual over all interior nodes; the arm the paper\'s current 1024^2 '
                    'accurate row is timed on'},
}


def rungs(L):
    """q=256 accurate rung and q=0 fast rung. `base` = the audited pre-optimisation arm (topfix, LU)."""
    x1 = L in (256, 512)                       # the eqcert-confirmed arm at this mesh uses one exact first step
    acc = dict(OPT, exact_steps=1, gtols=[1e-2]) if x1 else dict(OPT, gtols=[1e-3])
    r256 = dict(q=256, M=1088, gtols=[1e-6], variants=[dict(solver='lu')], rules=[
        # (1) the paper's rule family: pre-optimisation control + the parity chain + the optimised stack
        dict(name='scaled', parts=[Q256], base=[1e-6],
             variants=[dict(solver='lu', gtols=[1e-6]),           # kernel fold only  -> parity vs base
                       dict(solver='chol', gtols=[1e-6]),         # + Cholesky        -> parity vs base
                       dict(OPT, gtols=[1e-6, 1e-3])]),           # + clip/lamcarry/pred2 (algorithmic)
        # (2) the rule the burgers-eqcert lane certified at this mesh: the corrected accurate row
        dict(name='lat64', parts=[dict(lattice=64)], variants=[acc]),
        # (3) the dense residual: at 1024^2 this is the arm the paper currently prints
        # the exact arm carries the SAME tolerance as the accurate arm, so the switching-parity check
        # (the first exact step of an x1 arm IS the exact-residual arm's first step) has a twin
        dict(name='exact', exact=True, parts=[], base=([1e-6] if L == 1024 else []),
             variants=[dict(OPT, gtols=[1e-2 if x1 else 1e-3])])])
    r0 = dict(q=0, M=64, gtols=[1e-6], variants=[dict(solver='lu')], rules=[
        dict(name='scaled', parts=[Q0], base=[1e-6],
             variants=[dict(solver='lu', gtols=[1e-6]),                             # parity vs base
                       dict(solver='lu', clip=True, lamcarry=True, pred2=True, gtols=[1e-3])])])
    return [r0, r256]


def config(L):
    x1 = L in (256, 512)
    acc = ('q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2_x1' if x1 else
           'q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2')
    return dict(
        attempt=f'br{L}', intervals=L, dt=.005,
        eval_seed=7090702, eval_cases=4, eval_fresh_seed=911702, eval_fresh_cases=2,
        expected_physical_sha256='108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a',
        train_seed=0, train_trajectories=128, decoder_code_subsample=8192, gauss_jordan_max=64,
        cold_axis_points=48, strict=dict(ic_budget=400, step_budget=600, gtol=1e-6), ic_gtol=1e-6,
        inner_damping=1e-10, tau_y=.1, rho_bar=.116, rho_tight=.06,
        directions_file='directions_qtd02.npz',
        directions_sha256='79d794580dad15c03630470940361d8ba9a7779673be9697c90d28e5c3d81535',
        skip_certificates=True,
        population=dict(mesh=L, source_draw=[20260921, 56], cert_draws=[[0, 1, 2, 3, 4, 5, 6, 7]],
                        note='unused: skip_certificates is set. Kept so the cohort-disjointness assertions run.'),
        rule_status=STATUS[L],
        target_chunk=4, restrict_to=256, dense_tangent_group=34, dense_deadline_seconds=0,
        rungs=rungs(L), fom_settings=FOM, same_grid_reference='fft_tight',
        audit_cases=[0], audit_arms=['fft_tight', acc, 'q256_M1088_scaled_g1em06_base',
                                     'q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2'],
        reps=5, required_reps=5, order_seed=20260922, burn_seconds=.25, parity_bar=1e-9,
        # named roles, so the audit and the report never have to guess which arm is which
        roles=dict(optimised_accurate=acc, optimised_fast='q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2',
                   preoptimisation_accurate=('q256_M1088_exact_g1em06_base' if L == 1024 else
                                             'q256_M1088_scaled_g1em06_base'),
                   preoptimisation_fast='q0_M64_scaled_g1em06_base',
                   optimised_on_the_papers_own_rule='q256_M1088_scaled_g1em06_fast_chol_clip_lamcarry_pred2'),
        # which FOM names belong to which candidate grid, so the paper rule can be applied on either
        fom_subsets=dict(b_panel=[f['name'] for f in FOM if not f['name'].startswith('lean_')],
                         lean=[f['name'] for f in FOM if f['name'].startswith('lean_')],
                         full=[f['name'] for f in FOM]),
        profile_arms=[acc, 'q256_M1088_scaled_g1em06_fast_chol_clip_lamcarry_pred2'],
        keep_for_reference=['_base', '_fast', 'fft_tight', 'lean_', 'nt1e'],
        purpose=('DESIGN.md: re-time the paper\'s Burgers %d^2 rows on the optimised solver path against the '
                 'pre-optimisation arm they were timed on, in ONE allocation. Frozen checkpoint, frozen rules, '
                 'no certification, no retraining.' % L),
        rule_notes=dict(scaled='b-eqtop nodes at the same physical points, weights x (L/256)^2, NO refit',
                        lat64='uniform 63x63 interior sub-lattice, equal weights',
                        exact='no rule: dense advection over all (L-1)^2 interior nodes'),
        fom_notes=('audited = iterative_paths.make_fom, the implementation the 256^2/512^2/1024^2 rows in the '
                   'paper were selected against (it records a per-Newton linear-residual diagnostic costing one '
                   'extra Jacobian-vector product). lean = engines.make_fom, the same discretisation and solver '
                   'without that diagnostic, the implementation the 2048^2/4096^2 rows use. Both grids are timed '
                   'here so the paper rule can be applied on either.'),
        q0_through_hfast=True)


def smoke():
    """64^2, one case, one repetition: exercises every code path this lane adds (audited base arms for an EQ
    rule and for the dense residual, the parity gate against them, the optimised stack, one exact first step,
    skip_certificates). Local only; not a result."""
    c = config(1024)
    c.update(attempt='smoke64', intervals=64, case_subset=[0], reps=1, required_reps=1, burn_seconds=0.,
             restrict_to=64, local_smoke_waives_cohort_hash=True, allow_cpu_smoke=False, target_chunk=8,
             population=dict(mesh=64, source_draw=[20260921, 20], cert_draws=[[0]],
                             note='unused: skip_certificates is set'),
             fom_settings=[f for f in FOM if f['name'] in ('fft_tight', 'lean_nt3e-3_l3e-3_dt005')],
             audit_arms=['fft_tight', 'q256_M1088_lat16_g0p001_fast_chol_clip_lamcarry_pred2'],
             profile_arms=['q256_M1088_lat16_g0p001_fast_chol_clip_lamcarry_pred2'],
             purpose='local 64^2 smoke of every code path (not a result)')
    for rg in c['rungs']:                        # the transferred b-eqtop supports do not map onto a 64^2 grid
        for rs in rg['rules']:
            if rs['name'] == 'scaled':
                rs['name'], rs['parts'] = 'lat16', [dict(lattice=16)]
    c['rule_status'] = {'smoke': 'local smoke: no certificate, no result'}
    return c

if __name__ == '__main__':
    here = Path(__file__).parent
    for L in (256, 512, 1024):
        (here / f'config-{L}.json').write_text(json.dumps(config(L), indent=1) + '\n')
        print(f'config-{L}.json')
    (here / 'config-smoke64.json').write_text(json.dumps(smoke(), indent=1) + '\n')
    print('config-smoke64.json')
