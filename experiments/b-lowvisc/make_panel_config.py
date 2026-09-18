"""Write config-panel.json (the lvp01 stage-3 panel, DESIGN A3/A8) and config-smoke-panel64.json.

Every shared key is read from the incumbent panel's own config (comparators/bpn301-config-256.json,
job 3780638) so the solver, budgets, tolerances, timing contract, POD ranks, free bank and the
tuned full-order grid are the incumbent's verbatim; the cohort and reference hashes come from
this lane's gate job lvg01; the residual-direction settings are b-seeds' dev configuration
(qtd02's). Nothing is typed that exists in a JSON already.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
base = json.loads((HERE / 'comparators/bpn301-config-256.json').read_text())
gate = json.loads((HERE / 'runs/lvg01/archive/output/gate/result.json').read_text())
gcfg = gate['config']
seeds_dev = json.loads((ROOT.parent / '2026-09-17-b-seeds/experiments/b-seeds/config-dev-seed1.json').read_text())

SHARED = ['intervals', 'dt', 'eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases',
          'reference_mesh', 'reference_dt', 'train_seed', 'train_trajectories', 'train_state_stride',
          'snapshot_ntol', 'snapshot_ltol', 'decoder_code_subsample', 'gauss_jordan_max', 'cold_axis_points',
          'strict', 'ic_gtol', 'inner_damping', 'tau_y', 'recon_starts', 'recon_budget', 'reps', 'order_seed',
          'burn_seconds', 'rules_mesh', 'pod_ranks', 'free_bank', 'free_bank_M', 'same_grid_reference',
          'direct_control', 'fom_settings']
cfg = {k: base[k] for k in SHARED}
assert cfg['fom_settings'] == gcfg['fom_settings'], 'the gate job timed a different full-order grid'
cfg.update(
    attempt='lvp01', lane='b-lowvisc',
    nu_bounds=[gate['families']['lowvisc']['nu_lo'], gate['families']['lowvisc']['nu_hi']],
    expected_physical_sha256=gate['families']['lowvisc']['physical_sha256'],
    expected_physical_sha256_source='lvg01 (job %s) families.lowvisc.physical_sha256' % gate['job_id'],
    expected_reference_sha256={str(r['case']): r['field_sha256'] for r in gate['reference'] if r['family'] == 'lowvisc'},
    reference_sha_note='lvg01 4096-interval low-viscosity reference field hashes; GPU-model dependent, recorded as a probe, not a gate',
    snapshot_residual_bar=gcfg['snapshot_residual_bar'],
    # stage 1 (lv_directions.py): the incumbent `old` residual rule, qtd02 / b-seeds settings verbatim
    directions_file='directions_lowvisc.npz',
    q_ladder=seeds_dev['q_ladder'],
    residual_snapshots=seeds_dev['residual_snapshots'], residual_starts=seeds_dev['residual_starts'],
    residual_budget=seeds_dev['residual_budget'], residual_seed=seeds_dev['residual_seed'],
    # stage 2 (lv_panel.py): DESIGN A3 -- fixed-M cells only, dense quadrature, no EQ rule, no fast kernel, no FNO
    dense_q=[], eq_q=[], eq_gtols=[], rules={}, extra_rule_sets=[], extra_dense_q=[], extra_dense_M=None,
    extra_dense_cells=[[q, 1088] for q in (0, 16, 32, 64, 128, 256)] + [[q, 256] for q in (0, 16, 32, 64, 128)] + [[256, 2176]],
    fidelity_expectations={},
    priority_override=([f'q{q}_M1088_dense_g1em06' for q in (0, 16, 32, 64, 128, 256)]
                       + [f'pod{k}_M{4 * k}_dense' for k in base['pod_ranks']]
                       + [f'q{q}_M256_dense_g1em06' for q in (0, 16, 32, 64, 128)]
                       + ['q256_M2176_dense_g1em06', f'free512_M{base["free_bank_M"]}_dense']),
    priority_override_note='DESIGN A3/A8: the headline fixed-M = 1088 column is built first, then POD-LSPG, then the M = 256 control column, the (256, 2176) cell and the free bank; the OOM rule drops from the end',
    source_job_ids=dict(bpn301=base['source_job_ids'] and '3780638', lvg01=str(gate['job_id']), lvt01='3804337', qtd02='3757505'),
    purpose=('DESIGN A3/A8: the low-viscosity stage-3 panel in ONE allocation on ONE 80 GB A100 -- the '
             'checkpoint\'s own residual directions built in-job, the fixed M = 1088 correction ladder '
             'q in {0..256}, the fixed M = 256 control column q <= 128, the (256, 2176) cell, POD-LSPG at '
             'k in {16..512} through the same weak objective/solver/budget, the unrestricted R = 512 bank, '
             'and the tuned full-order Newton grid of job 3780638, three timed repetitions; every row '
             'carries the same-grid error AND the error against the 4096-interval reference (F4)'))
(HERE / 'config-panel.json').write_text(json.dumps(cfg, indent=1) + '\n')

smoke = dict(cfg, attempt='smoke-panel64', intervals=64, reference_mesh=128, reference_dt=0.0025,
             train_trajectories=4, case_subset=[0, 5], reps=1, burn_seconds=0.05, recon_budget=100, recon_starts=4,
             pod_ranks=[8, 16], order_seed=1, expected_physical_sha256=None, expected_reference_sha256=None,
             q_ladder=[0, 4], residual_snapshots=64, residual_starts=2, residual_budget=50,
             extra_dense_cells=[[0, 64], [4, 64], [0, 32], [4, 96]], priority_override=None,
             purpose='local smoke, 64 intervals: both stages run end to end; validates no number')
(HERE / 'config-smoke-panel64.json').write_text(json.dumps(smoke, indent=1) + '\n')
print('cells', cfg['extra_dense_cells'], '\nsubjects', len(cfg['priority_override']) + len(cfg['fom_settings']))
