"""Generate `config-j1.json` (bet1xx: top rungs + rebuilt ladder) and `config-j2.json`
(bet2xx: the rho-vs-m curve at q <= 64). Every inherited constant is READ from the parent
lane's committed `config-r3.json` and from the qrg304 archive, never typed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'experiments/q-ridge/config-r3.json'
QRG304 = ROOT / 'experiments/q-ridge/artifacts/qrg304/result.json'
MANIFEST = HERE / 'rules/qrg304/MANIFEST.json'

INHERIT = ('intervals', 'dt', 'eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases',
           'reference_mesh', 'reference_dt', 'train_seed', 'train_trajectories',
           'train_state_stride', 'snapshot_ntol', 'snapshot_ltol', 'residual_snapshots',
           'residual_starts', 'residual_budget', 'residual_seed', 'decoder_code_subsample',
           'quadrature_multiplier', 'quadrature_cap', 'eq_blocks', 'eq_seed', 'candidate_cap',
           'fit_states', 'max_fit_rows', 'gauss_jordan_max', 'cold_axis_points', 'strict',
           'ic_gtol', 'inner_damping', 'reps', 'order_seed', 'burn_seconds',
           'step_budget_provenance', 'q_ladder', 'expected_physical_sha256', 'fom_settings',
           'collect_iters', 'collect_seed', 'fit_trajectories', 'cert_trajectories',
           'cert_states', 'certify_chunk', 'incumbent_candidate_cap', 'incumbent_fit_states',
           'incumbent_max_fit_rows', 'incumbent_cap', 'rule_fitter', 'rho_bar',
           'rho_bar_provenance')


def main():
    parent = json.loads(PARENT.read_text())
    r = json.loads(QRG304.read_text())
    man = json.loads(MANIFEST.read_text())
    assert parent['strict']['step_budget'] == 600 and parent['eq_blocks'] == 16
    assert parent['rho_bar'] == 0.116 and man['rho_bar'] == 0.116
    shared = {k: parent[k] for k in INHERIT}
    shared.update(
        expected_directions_sha256=r['directions']['directions_sha256'],
        expected_pool_sha256={str(c['q']): dict(fit=c['fit_pool_sha256'],
                                                cert=c['certification_pool_sha256'])
                              for c in r['collection']},
        qrg304=dict(job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'],
                    archive_sha256=man['archive_sha256']),
        bars=dict(primary=parent['rho_bar'], tight=0.06),
        bars_provenance=('primary: q-ridge DESIGN A3.4 / q-diag rho of the q=0 incumbent rule '
                         '(0.1158); tight: the coordinator\'s tighter bar, declared before this '
                         'job ran, to test whether the primary bar is sufficient at the top rung'),
        recertify_tolerance=1e-6,
        candidate_pool=16384, pool_seed=20260917,
        eq_seconds=7200., fit_workers=8, fit_threads=1,
        fit_submission_deadline_seconds=11 * 3600.,
        max_rom_arms=16,
    )
    arms = [
        dict(name='std', fit_states='incumbent', scaling='row', compress=False,
             m_grid=[1024, 2048, 2560, 3072, 4096, 6144], adaptive=True, confirm=1,
             note='the qrg304 construction with a 16384 candidate pool'),
        dict(name='fs64', fit_states=64, scaling='row', compress=True,
             m_grid=[2048, 2560, 3072, 4096, 6144], adaptive=True, confirm=1,
             note='64 fit states at every rung, exact QR compression to 16384 rows'),
        dict(name='rhow64', fit_states=64, scaling='state', compress=True,
             m_grid=[2048, 2560, 3072, 4096, 6144], adaptive=True, confirm=1,
             note='64 fit states, rows scaled per state so the objective is sum_s rho(u_s)^2'),
    ]
    j1 = dict(shared, attempt='bet101', question='EQTOP', timed_phase=True,
              new_rungs=[128, 256], arms=arms, dense_twins=[0, 64, 128, 256],
              reproduction_arms=[dict(name='q128_eq_qrg304_m2048', q=128, m=2048),
                                 dict(name='q256_eq_qrg304_m2048', q=256, m=2048)],
              purpose=('Certify a reachable-state EQ rule at q = 128 and q = 256 on the primary '
                       'bar rho_max <= 0.116 (and the tight bar 0.06) by growing m to 6144 under '
                       'three declared fit arms, then rebuild the six-rung EQ ladder with the '
                       'cheapest certified rule per rung, dense twins, the two qrg304 '
                       'reproduction arms and same-job full-order controls.'),
              expectations={
                  'q0_eq_primary': dict(source='qrg304', arm='q0_m4_eqcert', tolerance=1e-9,
                                        note='q = 0 carries no directions: unconditional'),
                  'q16_eq_primary': dict(source='qrg304', arm='q16_m4_eqcert', tolerance=1e-3,
                                         tolerance_if_directions_bitwise=1e-9),
                  'q32_eq_primary': dict(source='qrg304', arm='q32_m4_eqcert', tolerance=1e-3,
                                         tolerance_if_directions_bitwise=1e-9),
                  'q64_eq_primary': dict(source='qrg304', arm='q64_m4_eqcert', tolerance=1e-3,
                                         tolerance_if_directions_bitwise=1e-9),
                  'q128_eq_qrg304_m2048': dict(source='qrg304', arm='q128_m4_eqcert',
                                               tolerance=1e-3, tolerance_if_directions_bitwise=1e-9),
                  'q256_eq_qrg304_m2048': dict(source='qrg304', arm='q256_m4_eqcert',
                                               tolerance=1e-3, tolerance_if_directions_bitwise=1e-9),
                  'q0_dense': dict(source='qrg304', arm='q0_m4_dense', tolerance=1e-9),
                  'q64_dense': dict(source='qrg304', arm='q64_m4_dense', tolerance=1e-3,
                                    tolerance_if_directions_bitwise=1e-9),
                  'q256_dense': dict(source='qrg304', arm='q256_m4_dense', tolerance=1e-3,
                                     tolerance_if_directions_bitwise=1e-9)})
    curve_arms = [
        dict(name='std', fit_states='incumbent', scaling='row', compress=False,
             m_grid=[256, 512, 1024, 2048, 3072, 4096], adaptive=False, confirm=0),
        dict(name='fs64', fit_states=64, scaling='row', compress=True,
             m_grid=[1024, 2048], adaptive=False, confirm=0),
        dict(name='rhow64', fit_states=64, scaling='state', compress=True,
             m_grid=[1024, 2048], adaptive=False, confirm=0),
    ]
    j2 = dict(shared, attempt='bet201', question='CURVE', timed_phase=False,
              new_rungs=[0, 16, 32, 64], arms=curve_arms, dense_twins=[],
              reproduction_arms=[], fit_workers=12, fit_threads=1,
              fit_submission_deadline_seconds=12 * 3600.,
              purpose=('The held-out rho against m at every rung q <= 64 on a common grid, so '
                       'the empirical law rho_max ~ m^-alpha and the m needed for a target rho '
                       'can be stated per rung; plus the fit-state and scaling arms at m = 1024 '
                       'and 2048 to see whether they matter below the top rungs. No timed phase.'),
              expectations={})
    # ---- job 3 (A2): how much of rho_max is the draw? ------------------------
    # Four independent draws of (candidate pool, fit-state subset) at fixed (q, m, states,
    # scaling), at the three configurations the top-rung verdict turns on. No timed phase.
    def rep_arms(states, scaling, compress, m, n=4):
        return [dict(name=f'rep{scaling[:3]}{states}m{m}s{i + 1}', fit_states=states,
                     scaling=scaling, compress=compress, m_grid=[m], adaptive=False,
                     confirm=0, seed_offset=i + 1,
                     note=f'replication draw {i + 1} of {n} at m = {m}') for i in range(n)]
    j3 = dict(shared, attempt='bet301', question='REPLICATION', timed_phase=False,
              new_rungs=[64, 128, 256], dense_twins=[], reproduction_arms=[],
              fit_workers=12, fit_threads=1,
              fit_submission_deadline_seconds=12 * 3600.,
              arms=(rep_arms('incumbent', 'row', False, 1024)
                    + rep_arms(64, 'row', True, 2048)),
              purpose=('Replication: bet201 showed that two independent draws of the candidate '
                       'pool and the fit-state subset change rho_max at fixed (q, m, states) by '
                       '0.11x to 2.3x, enough to flip certification at q = 64, m = 1024. This '
                       'job measures that spread directly: four independent draws at each of '
                       'q = 64, 128, 256, for the incumbent construction at m = 1024 and the '
                       '64-fit-state construction at m = 2048, so the certification verdict can '
                       'be reported with its draw-to-draw variability.'),
              expectations={})
    for name, cfg in (('config-j1.json', j1), ('config-j2.json', j2), ('config-j3.json', j3)):
        (HERE / name).write_text(json.dumps(cfg, indent=2) + '\n')
        print(name, 'rungs', cfg['new_rungs'], 'arms', [a['name'] for a in cfg['arms']])


if __name__ == '__main__':
    main()
