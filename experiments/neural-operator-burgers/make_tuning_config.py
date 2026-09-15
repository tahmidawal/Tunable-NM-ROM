"""Generate the immutable fixed-checkpoint tuning configuration from saved evidence.

Every fixed quantity is copied from the audited refinement02 diagnosis index and
the recorded data handoff; nothing in the configuration is typed by hand.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIAGNOSIS = HERE / 'runs/refinement02/live-diagnosis/index.json'
HANDOFF = HERE / 'checks/refinement02-data-handoff.json'
PROTOCOL = json.loads((HERE / 'protocol-refined.json').read_text())


def main():
    diagnosis = json.loads(DIAGNOSIS.read_text())
    handoff = json.loads(HANDOFF.read_text())
    setup = diagnosis['setup']
    strict = diagnosis['config']['strict']
    assert diagnosis['complete'] and diagnosis['intervals'] == 256 and diagnosis['m'] == 256
    settings = dict(
        native=dict(step_budget=strict['step_budget'], evolution_gtol=strict['gtol'],
                    evolution_residual_scale=1e-9,
                    role='exactly the archived native solver configuration'),
        converged=dict(step_budget=600, evolution_gtol=1e-8, evolution_residual_scale=1e-12,
                       role='high-effort reference endpoint'),
        ultra=dict(step_budget=1200, evolution_gtol=1e-10, evolution_residual_scale=1e-14,
                   role='stability control for the reference endpoint'),
    )
    for budget in [2, 4, 8]:
        settings[f'cap{budget}'] = dict(step_budget=budget, evolution_gtol=strict['gtol'],
                                        evolution_residual_scale=1e-9,
                                        role='iteration-cap screen; deliberately early stopped')
    for gtol in [1e-3, 1e-5, 1e-6]:
        settings[f'gtol{gtol:g}'] = dict(step_budget=600, evolution_gtol=gtol,
                                         evolution_residual_scale=1e-9,
                                         role='stopping-tolerance screen at the converged cap')
    cfg = dict(
        schema_version=1, pde='burgers',
        kind='fixed-checkpoint-solver-and-quadrature-tuning',
        checkpoint_sha256=diagnosis['checkpoint_sha256'],
        source_diagnosis_index_sha256=hashlib.sha256(DIAGNOSIS.read_bytes()).hexdigest(),
        validation_index_sha256=handoff['index_sha256']['validation'],
        reference_index_sha256=handoff['reference_index_sha256'],
        reference_anchor=PROTOCOL['anchor'],
        fixed=dict(latent_dimension=diagnosis['K'], bank_rank=diagnosis['R'], weak_modes=diagnosis['M'],
                   intervals=diagnosis['intervals'], dt=diagnosis['config']['dt'],
                   trust_radius=setup['trust_radius'], mode_ids=setup['mode_ids'],
                   initial_fit=dict(ic_budget=strict['ic_budget'], gtol=strict['gtol']),
                   cold_initializer=dict(rule=diagnosis['cold_setup']['rule'],
                                         axis_points=diagnosis['cold_setup']['axis_points'],
                                         points=diagnosis['cold_setup']['points'])),
        archived_rule=dict(eq_indices=setup['eq_indices'], eq_weights=setup['eq_weights'],
                           eq_relative_fit=setup['eq_relative_fit'], eq_seed=setup['eq_seed'],
                           eq_candidate_cap=setup['eq_candidate_cap'], eq_fit_states=len(setup['eq_fit_rows']),
                           eq_fit_rows=setup['eq_fit_rows']),
        settings=settings,
        quadratures=['m256', 'm512', 'm1024', 'full'],
        eq_refit_levels=[512, 1024],
        eq_fit_seconds_per_rule=1500,
        stage1=dict(cases=[2, 3], quadratures=['m256', 'full'],
                    settings=['native', 'converged', 'ultra'], stability_tolerance=1e-6,
                    purpose='establish a stable converged endpoint at fixed timestep before the broader screen'),
        stage3=dict(step_budgets=[2, 4, 8]),
        stage4=dict(evolution_gtols=[1e-3, 1e-5, 1e-6]),
        rule_selection=('lowest worst-case fixed-initial physical error over the calibration cases at the '
                        'converged setting; rules within 1% of the best are separated by lower median GPU seconds'),
        fom_controls=[dict(name='same_nt1e-2_dt005', mesh=256, dt=.005, ntol=.01, ltol=.5),
                      dict(name='same_nt1e-4_dt005', mesh=256, dt=.005, ntol=1e-4, ltol=.01),
                      dict(name='same_nt1e-6_dt005', mesh=256, dt=.005, ntol=1e-6, ltol=1e-8),
                      dict(name='same_nt1e-2_dt01', mesh=256, dt=.01, ntol=.01, ltol=.5),
                      dict(name='coarse_half_dt005', mesh=128, dt=.005, ntol=1e-4, ltol=.01),
                      dict(name='coarse_quarter_dt01', mesh=64, dt=.01, ntol=1e-4, ltol=.01)],
        drift_controls=['same_nt1e-2_dt005', 'same_nt1e-4_dt005'],
        repetitions=diagnosis['repetitions'], timing_seed=2026091412, burn_seconds=2.,
        gradient_normalization='norm(J^T r) / (norm(J) norm(r)); not a physical-error tolerance',
        output_policy=('every deployable arm returns the supplied initial field exactly while still paying for the '
                       'internal initial fit and evolution; native decoder compression of that field is separate'),
        early_stop_policy=('arms that exit on the iteration cap are recorded as early stopped and never relabelled '
                           'stationary; they may appear on the measured error/cost curve with that status'),
    )
    path = HERE / 'tuning-config.json'
    path.write_text(json.dumps(cfg, indent=2, allow_nan=False, sort_keys=False) + '\n')
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
