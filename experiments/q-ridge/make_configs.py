"""Generate `config-r1.json` and `config-r2.json`.

Every cross-job constant — the evaluation cohort hash, the comparator jobs' direction
hashes, the comparator jobs' reference field hashes, the inherited contract constants — is
READ from the retained archives rather than typed, so the configs cannot drift from the
cells they claim to reproduce. Run it, then commit the configs it writes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCES = {
    'btq101': ROOT / 'experiments/b-ladder-top/artifacts/btq101/result.json',
    'btq102': ROOT / 'experiments/b-ladder-top/artifacts/btq102/result.json',
    'btq201': ROOT / 'experiments/b-ladder-top/artifacts/btq201/result.json',
    'cclad01': ROOT / 'experiments/cheap-corrections/artifacts/cclad01/result.json',
    'abl01': ROOT / 'experiments/head-ablation/artifacts/abl01/result.json',
}
LAMS = [0., 1e-4, 1e-3, 1e-2, 1e-1, 1.]


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    src = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    base = src['btq102']['config']          # the budget-600 contract this lane inherits
    cohort = sha_array(np.asarray(src['abl01']['physical_cases']))
    assert cohort == sha_array(np.asarray(src['cclad01']['physical_cases']))
    assert cohort == sha_array(np.asarray(src['btq101']['physical_cases']))

    shared = {k: base[k] for k in (
        'intervals', 'dt', 'eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases',
        'reference_mesh', 'reference_dt', 'train_seed', 'train_trajectories',
        'train_state_stride', 'snapshot_ntol', 'snapshot_ltol', 'residual_snapshots',
        'residual_starts', 'residual_budget', 'residual_seed', 'decoder_code_subsample',
        'quadrature_multiplier', 'quadrature_cap', 'eq_blocks', 'eq_seed', 'candidate_cap',
        'fit_states', 'max_fit_rows', 'gauss_jordan_max', 'cold_axis_points', 'strict',
        'ic_gtol', 'inner_damping', 'recon_starts', 'recon_budget', 'reps', 'order_seed',
        'burn_seconds')}
    # btq102's frozen `strict` still carries the retained 180; the budget-600 contract this
    # lane inherits is the one its `_b600` sweep block declared and its audit found converged.
    b600 = next(b for b in base['sweep'] if b.get('tag') == '_b600')
    shared['strict'] = dict(shared['strict'], step_budget=int(b600['step_budget']))
    shared['step_budget_provenance'] = (
        "btq102 sweep block '_b600' (q256_m2_dense_base_b600): zero budget exits, worst joint "
        'gradient 9.71e-07, 0.988x the budget-180 control cost')
    assert shared['strict']['step_budget'] == 600, shared['strict']
    shared['eq_seconds'] = 3600.
    shared['q_ladder'] = [0, 16, 64, 256]      # consumed by directions.audited for reporting
    shared['expected_physical_sha256'] = cohort
    shared['expected_directions_sha256'] = base['expected_directions_sha256']
    shared['source_directions_sha256'] = {k: src[k]['directions']['directions_sha256']
                                          for k in ('btq101', 'btq102', 'btq201', 'cclad01')}
    shared['source_reference_sha256'] = {
        k: {str(x['case']): x['field_sha256'] for x in src[k]['reference']}
        for k in ('btq101', 'btq102', 'btq201', 'cclad01')}
    shared['source_job_ids'] = {k: src[k].get('job_id') for k in
                                ('btq101', 'btq102', 'btq201', 'cclad01')}
    shared['fom_settings'] = [
        dict(name='fft_tight', preconditioner='fft', dt=base['dt'], ntol=1e-6, ltol=1e-8),
        dict(name='fft_loose', preconditioner='fft', dt=base['dt'], ntol=1e-4, ltol=1e-6),
        dict(name='nt1e-2_dt01', preconditioner='fft', dt=2 * base['dt'], ntol=1e-2, ltol=.5)]
    shared['cohort_note'] = (
        'bitwise abl01 / cclad01 / btq101: the six physical cases hash to '
        f'{cohort}. The 4096-interval reference FIELD hashes are GPU-model dependent across '
        'jobs (btq101 and btq201 disagree with each other), so they are recorded as probes, '
        'not gates.')

    tight = 1e-9
    loose = 1e-3
    common_exp = {
        'q0_m4_dense_l0': dict(source='btq101', arm='q0_m4_dense_base', tolerance=tight,
                               note='q = 0 carries no correction directions, so this '
                                    'reproduction is unconditional'),
        'q0_m4_eq_l0_ret': dict(source='btq101', arm='q0_m4_eq_base', tolerance=tight,
                                note='built with the RETAINED NNLS fitter, which is what '
                                     'btq101 used for this gate arm'),
        'q16_m4_dense_l0': dict(source='cclad01', arm='q16_m4_dense_block', tolerance=loose,
                                tolerance_if_directions_bitwise=tight),
        'q64_m4_dense_l0': dict(source='cclad01', arm='q64_m4_dense_block', tolerance=loose,
                                tolerance_if_directions_bitwise=tight),
    }

    r1 = dict(shared, attempt='qrg101', question='R1', purpose=(
        'R1: a field-metric ridge lambda ||y||^2 on the correction block, lambda = '
        'lambda_rel * sigma_q^2 with sigma_q = ||Phi^T G C_q||_2. Six lambda_rel at q = 16, '
        '64, 256 in BOTH quadratures at the incumbent rule M = 4(K+q), with q = 0 as the '
        "ladder's base point. Tests whether the evolved-times regression is the extra "
        'unknowns overfitting the M test equations.'))
    r1['gate_arms'] = [
        dict(q=0, rule='m4', quadrature='eq', lam_rel=0., fitter='retained', tag='_ret'),
        dict(q=256, rule='m2', quadrature='dense', lam_rel=0.)]
    r1['sweep'] = ([dict(q=0, rule='m4', quadrature=qd, lams=[0.]) for qd in ('dense', 'eq')]
                   + [dict(q=q, rule='m4', quadrature=qd, lams=LAMS)
                      for q in (16, 64, 256) for qd in ('dense', 'eq')])
    r1['expectations'] = dict(common_exp, **{
        'q256_m2_dense_l0': dict(source='btq102', arm='q256_m2_dense_base_b600',
                                 tolerance=loose, tolerance_if_directions_bitwise=tight,
                                 note='the budget-600 contract this lane inherits')})

    r2 = dict(shared, attempt='qrg201', question='R2', purpose=(
        'R2: more tests at fixed q. M in {4, 8, 16} x (K+q) at q = 0, 16, 64 in BOTH '
        'quadratures at lambda = 0, with the empirical rule refit per (q, M) at m = 4M '
        'target support, capped at quadrature_cap. The same hypothesis from the other side: '
        'if the regression is test-space overfitting, more tests must remove it with no '
        'penalty at all.'))
    r2['q_ladder'] = [0, 16, 64]
    r2['gate_arms'] = [dict(q=0, rule='m4', quadrature='eq', lam_rel=0., fitter='retained',
                            tag='_ret')]
    r2['sweep'] = [dict(q=q, rule=rule, quadrature=qd, lams=[0.])
                   for q in (0, 16, 64) for rule in ('m4', 'm8', 'm16')
                   for qd in ('dense', 'eq')]
    r2['expectations'] = dict(common_exp)

    for name, cfg in (('config-r1.json', r1), ('config-r2.json', r2)):
        (HERE / name).write_text(json.dumps(cfg, indent=2) + '\n')
        arms = len(cfg['gate_arms']) + sum(len(b['lams']) for b in cfg['sweep'])
        print(name, 'declared arms (before de-duplication):', arms)


if __name__ == '__main__':
    main()
