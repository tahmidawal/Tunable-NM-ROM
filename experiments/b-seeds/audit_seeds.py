"""Independent NumPy audit of one b-seeds ladder invocation. No JAX, no GPU, no driver import.

Derived from q-trajdirs/audit_qtd.py (DESIGN.md D3) with: a cohort mode (`--cohort dev`:
bitwise abl01; `--cohort sealed`: values equal to the declared draw to <= 1 ulp, disjointness
recomputed on the values, `final_cohort_unopened` must be false), the qtd02 comparator for the
incumbent fidelity gate (`qtd02_expectations`), the three-layer decomposition at q = 0, and
the training gates of DESIGN.md section 7 when `--train` names the job's training output.

The original docstring follows.


Recomputes every reported error from the retained output fields, measures every subject
against the same-job converged full-order solve on three metrics (t = 0 compression,
worst over all output times, worst over evolved times only), checks the cross-job
fidelity gates against `cclad01` and the `b-ladder-top` jobs, walks each declared ladder
for monotonicity on both metrics, derives the non-dominated sets, and evaluates the
pre-registered pass of `DESIGN.md` section 6.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = {
    'qtd02': ROOT / 'experiments/b-seeds/comparators/qtd02-result.json',
    'cclad01': ROOT / 'experiments/cheap-corrections/artifacts/cclad01/result.json',
    'btq101': ROOT / 'experiments/b-ladder-top/artifacts/btq101/result.json',
    'btq201': ROOT / 'experiments/b-ladder-top/artifacts/btq201/result.json',
}
COMPARATOR_AUDITS = {
    'qtd02': ROOT / 'experiments/b-seeds/comparators/qtd02-audit.json',
    'cclad01': ROOT / 'experiments/cheap-corrections/checks/cclad01-audit.json',
    'btq101': ROOT / 'experiments/b-ladder-top/checks/btq101-audit.json',
    'btq201': ROOT / 'experiments/b-ladder-top/checks/btq201-audit.json',
}


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def nondominated(rows, cost, err):
    pts = [r for r in rows if r.get(cost) is not None and r.get(err) is not None]
    out = []
    for r in pts:
        if not any((o[cost] <= r[cost] and o[err] <= r[err] and
                    (o[cost] < r[cost] or o[err] < r[err])) for o in pts):
            out.append(r['arm'])
    return sorted(set(out))


def mono(vals):
    return bool(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])))


def comparator_rows(path):
    """Per-arm aggregates of a comparator job, recomputed here from its own JSON."""
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    table = {}
    for x in r['invocations']:
        t = table.setdefault(x['name'], dict(ref={}, t0={}, evolved={}, gpu_ms=[]))
        t['ref'][x['case']] = x['error']['fixed_initial_max']
        t['t0'][x['case']] = x['error']['fixed_initial_per_time'][0]
        t['evolved'][x['case']] = max(x['error']['fixed_initial_per_time'][1:])
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
    return {k: dict(worst_reference=float(max(v['ref'].values())),
                    worst_t0_vs_reference=float(max(v['t0'].values())),
                    worst_evolved_vs_reference=float(max(v['evolved'].values())),
                    median_gpu_ms=median(v['gpu_ms'])) for k, v in table.items()}


def comparator_same_grid(path):
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    rows = d.get('arms') or d.get('checks', {}).get('arm_table') or []
    out = {}
    for x in rows:
        out[x['arm']] = dict(
            same_grid_all=x.get('worst_all_times_percent', x.get('worst_same_grid_percent')),
            same_grid_evolved=x.get('worst_evolved_percent'),
            t0=x.get('worst_t0_compression_percent'),
            median_gpu_ms=x.get('median_gpu_ms'), converged=x.get('converged'))
    return out


def relerr(x, y):
    return abs(float(x) - float(y)) / max(abs(float(y)), 1e-300)


def training_gates(tdir, r, gate, a):
    """Stage A/B/C gates from the job's own JSONs, the emitted checkpoint (a NumPy pickle)
    and the incumbent's recorded job JSONs. Value gates throughout: FOM last bits are
    GPU-model dependent, so sums and sums of squares are compared at 1e-9 relative."""
    import pickle
    r3 = json.loads(next(tdir.glob('sep_burgers_r3_N*_K16_R512.json')).read_text())
    co = json.loads(next(tdir.glob('sep_coeff_N*_K16_R512.json')).read_text())
    hf = json.loads((tdir / 'hfit_full.json').read_text())
    inc_r3 = json.loads(Path(a.incumbent_r3).read_text())
    inc_co = json.loads(Path(a.incumbent_coeff).read_text())
    inc_hf = json.loads(Path(a.incumbent_hfit).read_text())
    seed = int(r3['config']['seed'])
    ck_path = tdir / f'sep_hfit_seed{seed}.pkl'
    ck_bytes = ck_path.read_bytes()
    ck_sha = hashlib.sha256(ck_bytes).hexdigest()
    ck = pickle.loads(ck_bytes)
    pick = np.asarray(ck['cfg']['hfit_pick'])
    T = int(co['data']['T'])
    early = int(np.sum((pick % T) <= int(co['config']['t_early'])))
    recorded = {ln.split()[1].split('/')[-1]: ln.split()[0]
                for ln in (tdir / 'TRAIN-SHA256.txt').read_text().splitlines() if ln.strip()}
    tr = r3['train']
    gate('training_complete', r3.get('complete') is True and co.get('complete') is True
         and hf.get('complete') is True and r3.get('train_only') is True)
    gate('bank_steps_done_300000', tr['steps_done'] == 300000 and tr['steps'] == 300000
         and tr['time_capped'] is False, dict(steps_done=tr['steps_done'], capped=tr['time_capped']))
    mid = hf['arms']['mid']['train']
    gate('head_steps_done_200000', mid['steps_done'] == 200000 and mid['steps'] == 200000,
         dict(steps_done=mid['steps_done'], seconds=mid['seconds']))
    gate('seed_is_the_declared_one',
         seed == int(co['config']['seed']) == int(hf['config']['seed']) and seed in (1, 2, 3),
         dict(r3=r3['config']['seed'], coeff=co['config']['seed'], hfit=hf['config']['seed']))
    gate('recipe_values_match_incumbent', all(
        r3['config'][k] == inc_r3['config'][k] for k in (
            'N', 'k', 'r', 'steps', 'lr', 'p_sub', 'wd', 'ema_decay', 'full_last', 'lam_orth',
            'snap_norm', 'max_snaps', 't_early', 'n_traj', 'pool', 'data_seed', 'test_seed',
            'arch_overrides')) and all(
        co['config'][k] == inc_co['config'][k] for k in (
            'N', 'k', 'r', 'max_snaps', 't_early', 'n_traj', 'extra_seed', 'extra_traj',
            'n_canonical', 'n_test', 'data_seed', 'test_seed', 'loose')) and all(
        hf['config'][k] == inc_hf['config'][k] for k in ('steps', 'batch', 'lr', 'oracle_iters',
                                                          'enc_steps', 'K_ckpt', 'R', 'S'))
        and hf['arms']['mid']['spec'] == inc_hf['arms']['mid']['spec'],
        dict(r3=r3['config'].get('arch_overrides'), hfit_spec=hf['arms']['mid']['spec']))
    gate('pick_is_131072_with_27648_early', len(pick) == 131072 and early == 27648
         and co['data']['n_states_trained'] == 131072 and int(np.asarray(ck['Z_tr']).shape[0]) == 131072,
         dict(pick=len(pick), early=early, T=T))
    fpA, fpA0 = r3['data']['fingerprint'], inc_r3['data']['fingerprint']
    fpB, fpB0 = co['data']['fingerprint'], inc_co['data']['fingerprint']
    fp = dict(bank_sum=relerr(fpA['sum'], fpA0['sum']), bank_sumsq=relerr(fpA['sumsq'], fpA0['sumsq']),
              extract_sum=relerr(fpB['sum'], fpB0['sum']),
              extract_sumsq=relerr(fpB['sumsq'], fpB0['sumsq']),
              shapes_equal=(fpA['shape'] == fpA0['shape'] and fpB['shape'] == fpB0['shape']))
    gate('training_data_fingerprint_matches_incumbent',
         fp['shapes_equal'] and max(v for k, v in fp.items() if k != 'shapes_equal') < 1e-9, fp)
    gate('fom_residuals_converged',
         r3['data']['max_fom_rel_residual'] <= 1e-8 and r3['data']['max_fom_rel_residual_test'] <= 1e-8
         and co['data']['max_fom_rel_residual'] <= 1e-8 and co['data']['max_fom_rel_residual_test'] <= 1e-8)
    gate('gram_identity_and_whitening',
         co['gates']['gram_identity_rel_dev_mean'] < 1e-6 and co['gates']['gram_identity_rel_dev_max'] < 1e-4
         and co['gates']['span_floor_direct_vs_gram_rel_dev'] < 1e-6
         and hf['gates']['whitening_round_trip'] < 1e-10 and hf['gates']['q_equals_LT_h'] < 1e-10,
         dict(co=co['gates'], hf=hf['gates']))
    gate('checkpoint_sha256_consistent',
         ck_sha == r['checkpoint_sha256'] == recorded.get(ck_path.name),
         dict(file=ck_sha, ladder=r['checkpoint_sha256'], recorded=recorded.get(ck_path.name)))
    gate('checkpoint_finite_and_shaped',
         all(np.isfinite(np.asarray(v)).all() for v in (ck['params']['h_lin'], ck['params']['B'], ck['Z_tr']))
         and all(np.isfinite(np.asarray(w)).all() and np.isfinite(np.asarray(b)).all()
                 for w, b in list(ck['params']['g']) + list(ck['params']['h']))
         and np.asarray(ck['params']['h_lin']).shape == (16, 512)
         and np.asarray(ck['Z_tr']).shape == (131072, 16))
    o = hf['arms']['mid']['oracle_test']
    o0 = inc_hf['arms']['mid']['oracle_test']
    return dict(seed=seed, checkpoint_sha256=ck_sha, bank_sha256=recorded.get('sep_burgers_r3_N256_K16_R512.pkl'),
                npz_sha256=recorded.get('sep_coeff_N256_K16_R512.npz'),
                gpu=dict(bank=r3['config'].get('gpu'), extract=co['config'].get('gpu'), hfit=hf['config'].get('gpu')),
                job_ids=dict(bank=r3['config'].get('slurm_job'), extract=co['config'].get('slurm_job'),
                             hfit=hf['config'].get('slurm_job')),
                bank=dict(steps_done=tr['steps_done'], seconds=tr['seconds'], time_capped=tr['time_capped'],
                          used_ema=tr['used_ema'], final_rel_mse=tr['final_rel_mse'],
                          recon_train_mean_percent=tr['recon_rel_l2_mean'] * 100,
                          recon_train_max_percent=tr['recon_rel_l2_max'] * 100,
                          recon_fullgrid_subset_mean_percent=tr.get('recon_fullgrid_subset_mean', float('nan')) * 100,
                          cond_G=co['span']['cond_G'], numerical_rank=co['span']['numerical_rank_1e8'],
                          incumbent=dict(recon_train_mean_percent=inc_r3['train']['recon_rel_l2_mean'] * 100,
                                         seconds=inc_r3['train']['seconds'], cond_G=inc_co['span']['cond_G'])),
                span_floor=dict(train_mean_percent=co['train_span_floor']['mean'] * 100,
                                train_max_percent=co['train_span_floor']['max'] * 100,
                                test_mean_percent=co['test_span_floor']['mean'] * 100,
                                test_max_percent=co['test_span_floor']['max'] * 100,
                                incumbent=dict(train_mean_percent=inc_co['train_span_floor']['mean'] * 100,
                                               test_mean_percent=inc_co['test_span_floor']['mean'] * 100,
                                               test_max_percent=inc_co['test_span_floor']['max'] * 100)),
                head=dict(steps_done=mid['steps_done'], seconds=mid['seconds'], final_loss=mid['final_loss'],
                          recon_train_mean_percent=hf['arms']['mid']['recon_train']['mean'] * 100,
                          recon_train_max_percent=hf['arms']['mid']['recon_train']['max'] * 100,
                          oracle_test_mean_percent=o['mean'] * 100, oracle_test_max_percent=o['max'] * 100,
                          oracle_test_t0_percent=o['t0'] * 100,
                          oracle_over_span_floor=hf['arms']['mid']['oracle_over_span_floor'],
                          code_refit_gain=hf['arms']['mid']['oracle_train_from_codes']['gain_vs_recon'],
                          incumbent=dict(final_loss=inc_hf['arms']['mid']['train']['final_loss'],
                                         recon_train_mean_percent=inc_hf['arms']['mid']['recon_train']['mean'] * 100,
                                         oracle_test_mean_percent=o0['mean'] * 100,
                                         oracle_test_max_percent=o0['max'] * 100,
                                         oracle_over_span_floor=inc_hf['arms']['mid']['oracle_over_span_floor'])),
                pick=dict(states=int(len(pick)), early=early, early_fraction=early / len(pick)),
                fingerprint_relative_differences=fp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', required=True)
    p.add_argument('--partial', action='store_true',
                   help='audit a job that died before its timed phase: run the environment, '
                        'cohort, direction and reconstruction checks and record that the '
                        'timed ladders do not exist, instead of failing on empty tables')
    p.add_argument('--cohort', choices=('dev', 'sealed'), default='dev')
    p.add_argument('--sealed', default=str(ROOT / 'experiments/b-seeds/checks/sealed-cohort.json'),
                   help='the declared sealed cohort (values, seed, disjointness record)')
    p.add_argument('--train', default=None,
                   help='the job\'s output/train directory: enables the training gates')
    p.add_argument('--incumbent-r3', default=str(
        ROOT / 'experiments/separable-decoder/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.json'))
    p.add_argument('--incumbent-coeff', default=str(
        ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_coeff_N256_K16_R512.json'))
    p.add_argument('--incumbent-hfit', default=str(
        ROOT / 'experiments/separable-decoder/runs/dn256b/out/hfit_dn256b_full.json'))
    p.add_argument('--expect-checkpoint-sha256', default=None,
                   help='gate the evaluated checkpoint against a recorded SHA256')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    fields_dir = Path(a.fields)
    checks, fail = {}, []

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def info(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note, blocking=False)

    cfg = r['config']
    K = r['K']
    qlad = list(cfg['q_ladder'])

    # ------------------------------------------------------- environment gates
    partial = bool(a.partial) or not r['invocations']
    if partial:
        checks['complete'] = dict(
            passed=False, detail=dict(complete=r.get('complete'),
                                      invocations=len(r['invocations'])),
            note='PARTIAL audit: this job did not reach its timed phase, so it has no ladders, '
                 'no frontier and no verdict. Everything below that does not need a timed '
                 'invocation is still checked.')
        fail.append('complete')
    else:
        gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    if not partial:
        gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_flag_matches_cohort',
         r['final_cohort_unopened'] is (a.cohort == 'dev'),
         dict(flag=r['final_cohort_unopened'], cohort=a.cohort),
         'development jobs must record the final cohort as unopened; the sealed job as opened')
    want_ck = a.expect_checkpoint_sha256 or cfg.get('expected_checkpoint_sha256')
    if want_ck:
        gate('checkpoint_is_the_recorded_one', r['checkpoint_sha256'] == want_ck,
             dict(expected=want_ck, got=r['checkpoint_sha256']))
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    info('reference_fields_bitwise_match_comparator',
         all(bool(x.get('bitwise_matches_comparator')) for x in r['reference']),
         [x.get('bitwise_matches_comparator') for x in r['reference']],
         'a cross-job bitwise probe; informative, not required')

    # ------------------------------------------------------------ cohort gates
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    mine_pc = np.array(r['physical_cases'], dtype=float)
    if a.cohort == 'dev':
        abl = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'
        recomputed = None
        if abl.exists():
            pc = np.array(json.loads(abl.read_text())['physical_cases'])
            recomputed = hashlib.sha256(np.ascontiguousarray(pc).tobytes()).hexdigest()
            cg = dict(cg, abl01_recomputed_sha256=recomputed,
                      bitwise_equal=bool(np.array_equal(mine_pc, pc)))
        gate('evaluation_cohort_bitwise_abl01',
             bool(cg.get('bitwise_equal', cg.get('passed'))), cg)
        gate('cohort_roles_are_development',
             all(v in ('opened development', 'fresh development') for v in r['cohort_roles']),
             r['cohort_roles'])
    else:
        sd = json.loads(Path(a.sealed).read_text())
        want = np.array(sd['physical_cases'], dtype=float)
        same_shape = want.shape == mine_pc.shape
        ulps = (np.abs(mine_pc - want) / np.spacing(np.abs(want))) if same_shape else None
        gate('sealed_cohort_values_match_declared',
             bool(same_shape and np.all(ulps <= 1.0)),
             dict(seed=sd['seed'], cases=sd['cases'], config_seed=cfg['eval_seed'],
                  config_cases=cfg['eval_cases'], fresh_cases=cfg['eval_fresh_cases'],
                  max_ulp=(float(ulps.max()) if same_shape else None),
                  declared_sha256_local=sd['sha256_local'],
                  job_sha256=r['gates']['evaluation_cohort_bitwise_abl01'].get('got')),
             'values to <= 1 ulp: the viscosity column passes through np.exp, which differs by '
             'one ulp between the GB10 and the cluster NumPy, so the byte hash is a probe only')
        gate('sealed_cohort_seed_and_count_declared',
             cfg['eval_seed'] == sd['seed'] and cfg['eval_cases'] == sd['cases']
             and cfg['eval_fresh_cases'] == 0)
        # disjointness recomputed on the job's own values against every training-family draw
        def draw(seed, count):
            rr = np.random.default_rng(seed)
            return np.stack([rr.uniform(.15, .85, count), rr.uniform(.15, .85, count),
                             rr.uniform(.05, .20, count), rr.uniform(.5, 2., count),
                             np.exp(rr.uniform(np.log(.01), np.log(.1), count))], axis=1)
        dist = {}
        for name, (sd_, n_) in {'seed0_576': (0, 576), 'seed1000_4032': (1000, 4032),
                                'seed1_8': (1, 8), 'seed0_128': (0, 128),
                                'dev_7090702_4': (7090702, 4), 'dev_911702_2': (911702, 2),
                                'holdout_20260916_64': (20260916, 64)}.items():
            other = draw(sd_, n_)
            dist[name] = float(np.linalg.norm(mine_pc[:, None, :] - other[None], axis=2).min())
        gate('sealed_cohort_disjoint', min(dist.values()) > 1e-6, dist)
        gate('cohort_roles_are_sealed', all(v == 'sealed final' for v in r['cohort_roles']),
             r['cohort_roles'])
    dc = r['gates']['direction_cohorts_disjoint_from_evaluation']
    gate('direction_cohorts_disjoint_from_evaluation', bool(dc['passed']), dc['detail'])

    # -------------------------------------------------------- direction gates
    ds = r['direction_sets']
    gate('directions_hashed_and_saved',
         all((fields_dir / v['artifact']).exists() for v in ds.values()),
         {k: v['artifact'] for k, v in ds.items()})
    gate('directions_rank_covers_ladder',
         all(v['available_rank'] >= max(qlad) for v in ds.values()),
         {k: v['available_rank'] for k, v in ds.items()})
    prefix_ok, prefix_detail = True, {}
    for name, v in ds.items():
        C = np.load(fields_dir / v['artifact'])['C']
        got = {str(q): hashlib.sha256(
            np.ascontiguousarray(C[:, :int(q)]).tobytes()).hexdigest() for q in qlad}
        same = got == {str(q): v['prefix_sha256'][str(q)] for q in qlad}
        prefix_detail[name] = dict(matches=bool(same),
                                   whole_matrix_sha256=hashlib.sha256(
                                       np.ascontiguousarray(C).tobytes()).hexdigest(),
                                   reported=v['directions_sha256'])
        prefix_ok = prefix_ok and same
    gate('direction_prefixes_recomputed_from_artifact', prefix_ok, prefix_detail,
         'the hash of the first q columns of each SAVED matrix equals the hash the job '
         'recorded for the slice it handed rung q')
    # The job writes this gate at the very end, so a job that died earlier has the same
    # claim to check but not the job's own answer; the audit then derives it from the
    # arm setups and the saved matrices, which is the stronger check anyway.
    npc = r['gates'].get('nested_prefix_consistent')
    if npc is None:
        ok = all(s2['directions_prefix_sha256']
                 == r['direction_sets']['old' if s2['dirset'] == 'shared' else s2['dirset']
                                        ]['prefix_sha256'][str(s2['q'])]
                 for s2 in r['arm_setup'])
        npc = dict(passed=ok, note='derived by the audit; the job died before writing it')
    gate('nested_prefix_consistent', bool(npc['passed']), npc)
    info('old_directions_hash_matches_comparator',
         bool(ds['old'].get('bitwise_matches_comparator')),
         dict(expected=ds['old'].get('expected_sha256'), got=ds['old']['directions_sha256']),
         'bitwise across jobs; qlad01, cclad01 and the b-ladder-top jobs already disagree '
         'with each other, so this is a probe, not a requirement')
    dir_bitwise = bool(ds['old'].get('bitwise_matches_comparator'))

    # ------------------------------------ what a partial job can still report ---
    if partial:
        recon = {(x['dirset'], x['q']): x for x in r['reconstruction']}
        gate('every_declared_rung_has_a_reconstruction',
             all((s2['dirset'], s2['q']) in recon for s2 in r['arm_setup']),
             sorted({(s2['dirset'], s2['q']) for s2 in r['arm_setup']}
                    - set(recon)))
        eqp = [s2 for s2 in r['arm_setup'] if s2.get('quadrature') == 'eq']
        gate('every_eq_rule_reports_validity',
             all('eq_rule_valid' in s2 or
                 (s2.get('eq_fit') or {}).get('fitter') == 'retained_nnls_capped' for s2 in eqp),
             [s2['arm'] for s2 in eqp if 'eq_rule_valid' not in s2][:5])
        gate('no_eq_rule_truncated',
             all(not (s2.get('eq_fit') or {}).get('truncated', False) for s2 in eqp),
             [s2['arm'] for s2 in eqp if (s2.get('eq_fit') or {}).get('truncated')])
        gate('overdetermined_weak_system',
             all(s2['M'] > s2['solved_dimension'] for s2 in r['arm_setup']),
             [(s2['arm'], s2['M'], s2['solved_dimension']) for s2 in r['arm_setup']
              if s2['M'] <= s2['solved_dimension']][:5])
        # The job never wrote its own elapsed time, so the last phase timestamp it DID
        # write is reported as a floor, and the terminating error is lifted verbatim out
        # of the collected stderr rather than retyped.
        logs = Path(a.fields).parent / 'logs'
        err_lines = []
        for f2 in sorted(logs.glob('*.err')) if logs.exists() else []:
            err_lines = [ln for ln in f2.read_text(errors='replace').splitlines() if ln.strip()]
        terminating = next((ln for ln in reversed(err_lines)
                            if 'Error' in ln or 'error' in ln), None)
        out = dict(result=str(Path(a.result).resolve()),
                   result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
                   partial=True, job_id=r.get('job_id'), commit=r.get('commit'),
                   gpu=r.get('gpu'), elapsed_seconds=r.get('elapsed_seconds'),
                   last_recorded_phase_seconds=max(
                       [v.get('seconds', 0) for v in r['direction_sets'].values()]
                       + [r.get('compile_warmup', {}).get('seconds', 0)] + [0]),
                   terminating_error=terminating,
                   arms_built=len(r['arm_setup']),
                   subjects_warmed=None,
                   output_times=r['output_times'], checks=checks, failed=sorted(fail),
                   arms=[], ladders={}, frontier={}, verdict=None,
                   arm_setup=[{k2: v2 for k2, v2 in s2.items()
                               if k2 not in ('eq_indices', 'eq_weights', 'eq_fit_rows')}
                              for s2 in r['arm_setup']],
                   reconstruction=[dict(dirset=x['dirset'], q=x['q'],
                                        worst_bank_projection_percent=x['worst_bank_projection'] * 100,
                                        worst_best_found_percent=x['worst_best_found'] * 100)
                                   for x in r['reconstruction']],
                   direction_sets={k: {kk: vv for kk, vv in v.items()
                                       if kk not in ('singular_values',)}
                                   for k, v in r['direction_sets'].items()},
                   direction_comparison=r['direction_comparison'],
                   direction_cohorts=r['direction_cohorts'],
                   direction_rollout=r.get('direction_rollout', {}),
                   comparators={})
        Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(dict(partial=True, failed=sorted(fail),
                              arm_setups=len(r['arm_setup']),
                              reconstructions=len(r['reconstruction'])), indent=2))
        print('PARTIAL AUDIT WROTE', a.out)
        return

    # --------------------------------------------------------- bookkeeping ---
    inv = r['invocations']
    reps = cfg['reps']
    counts = {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps},
         sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired',
         all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms),
         [(x['name'], x['M'], x['solved_dimension']) for x in roms
          if x['M'] <= x['solved_dimension']][:5])
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x and 'worst_joint_stationarity' in x)
             for x in roms))
    gate('step_budget_is_600',
         all(x.get('step_budget') == 600 for x in roms), cfg['strict'])
    eq = [s for s in r['arm_setup'] if s.get('quadrature') == 'eq']
    gate('every_eq_rule_reports_validity',
         all(('eq_rule_valid' in s or s.get('eq_fit', {}).get('fitter') == 'retained_nnls_capped')
             for s in eq),
         [s['arm'] for s in eq if 'eq_rule_valid' not in s][:5])
    gate('no_eq_rule_truncated',
         all(not (s.get('eq_fit') or {}).get('truncated', False) for s in eq),
         [s['arm'] for s in eq if (s.get('eq_fit') or {}).get('truncated')])

    # ------------------------------------------- errors recomputed from fields
    cache = {}

    def fields_of(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    bad = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not bad, bad[:5])
    refs = {x['case']: fields_of(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
    gate('same_grid_baseline_present', len(base) == len(refs), sorted(base))
    worst = 0.
    for x in inv:
        truth = refs[x['case']]
        f = fields_of(x['artifact'])
        n0 = np.linalg.norm(truth[0])
        err = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        worst = max(worst, abs(float(np.max(err)) - x['error']['fixed_initial_max'])
                    / max(x['error']['fixed_initial_max'], 1e-300))
        g = fields_of(base[x['case']])
        sg = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
        x['same_grid_per_time'] = sg.tolist()
        x['same_grid_all'] = float(np.max(sg))
        x['same_grid_evolved'] = float(np.max(sg[1:]))
        x['t0_compression'] = float(sg[0])
    gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # ------------------------------------------------------ per-arm aggregates
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    recon = {(x['dirset'], x['q']): x for x in r['reconstruction']}
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), dirset=x.get('dirset'),
            q=x.get('q'), solved_dimension=x.get('solved_dimension'), M=x.get('M'),
            m=x.get('m'), rule=x.get('rule'), quadrature=x.get('quadrature'),
            fitter=x.get('fitter'), dt=x.get('dt'), gpu_ms=[], host_ms=[],
            case_ref={}, case_all={}, case_evolved={}, case_t0={}, case_per_time={},
            iterations=[], stationarity=[], stationary=[], completed=[], converged=[],
            budget_exits=[], exit_reasons=[], ic_reasons=[], failing={}))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_evolved'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['case_per_time'][x['case']] = x['same_grid_per_time']
        t['iterations'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['converged'].append(x['converged'])
            t['budget_exits'].append(x['budget_exits'])
            t['exit_reasons'] += x['stop_reasons']
            t['ic_reasons'].append(x['ic_reason'])
            bd = [i for i, s in enumerate(x['stop_reasons']) if s == 0]
            if bd:
                t['failing'].setdefault(x['case'], sorted(bd))

    rows = []
    for name, t in table.items():
        st = setup.get(name, {})
        eqf = st.get('eq_fit') or {}
        rc = recon.get((t['dirset'], t['q']))
        per_time = np.array([t['case_per_time'][c] for c in sorted(t['case_per_time'])])
        row = dict(
            arm=name, kind=t['kind'], family=t['family'] or 'fom', dirset=t['dirset'],
            q=t['q'], solved_dimension=t['solved_dimension'], M=t['M'], m=t['m'],
            rule=t['rule'], quadrature=t['quadrature'], fitter=t['fitter'], dt=t['dt'],
            worst_reference_percent=float(np.max(list(t['case_ref'].values())) * 100),
            worst_all_times_percent=float(np.max(list(t['case_all'].values())) * 100),
            worst_evolved_percent=float(np.max(list(t['case_evolved'].values())) * 100),
            median_evolved_percent=float(np.median(list(t['case_evolved'].values())) * 100),
            worst_t0_compression_percent=float(np.max(list(t['case_t0'].values())) * 100),
            worst_percent_per_time=(np.max(per_time, axis=0) * 100).tolist(),
            per_case_evolved_percent={str(c): v * 100 for c, v in sorted(t['case_evolved'].items())},
            per_case_all_times_percent={str(c): v * 100 for c, v in sorted(t['case_all'].items())},
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']), gpu_ms_all=sorted(t['gpu_ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            max_iterations=(float(np.max(t['iterations'])) if t['iterations'] else None),
            max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in
                                 zip(*np.unique(t['exit_reasons'], return_counts=True))}
                                if t['exit_reasons'] else None),
            failing_cases={str(c): v for c, v in sorted(t['failing'].items())},
            quadrature_fit_seconds=eqf.get('seconds'),
            quadrature_fit_relative=st.get('eq_relative_fit'),
            quadrature_support=eqf.get('support'), quadrature_truncated=eqf.get('truncated'),
            eq_rule_valid=st.get('eq_rule_valid'),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100)
                                           if rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            setup_seconds=st.get('total_setup_seconds'))
        rows.append(row)
    by_arm = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['dirset'] or '', x['q'] if x['q'] is not None else -1,
                             x['M'] or 0, x['arm']))

    # -------------------------------------------------------- fidelity gates -
    comp = {k: comparator_rows(v) for k, v in COMPARATORS.items()}
    comp_sg = {k: comparator_same_grid(v) for k, v in COMPARATOR_AUDITS.items()}
    fid = {}
    for arm, spec in cfg.get('fidelity_expectations', {}).items():
        mine = by_arm.get(arm)
        theirs = comp.get(spec['job'], {}).get(spec['reproduces'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail='arm or comparator missing')
            gate(f'reproduces_{arm}', False, fid[arm])
            continue
        rel_ref = abs(mine['worst_reference_percent'] / 100 - theirs['worst_reference']) / max(
            theirs['worst_reference'], 1e-300)
        sg = (comp_sg.get(spec['job'], {}).get(spec['reproduces']) or {}).get('same_grid_all')
        rel_sg = (abs(mine['worst_all_times_percent'] - sg) / max(sg, 1e-300)
                  if sg is not None else None)
        tol = spec['tolerance']
        second = spec.get('second_tier_tolerance')
        d = dict(job=spec['job'], comparator=spec['reproduces'], declared_tolerance=tol,
                 second_tier_tolerance=second, directions_bitwise=dir_bitwise,
                 relative_worst_reference_difference=rel_ref,
                 relative_worst_same_grid_difference=rel_sg,
                 ours_worst_reference_percent=mine['worst_reference_percent'],
                 theirs_worst_reference_percent=theirs['worst_reference'] * 100,
                 ours_worst_same_grid_percent=mine['worst_all_times_percent'],
                 theirs_worst_same_grid_percent=sg,
                 ours_median_gpu_ms=mine['median_gpu_ms'],
                 theirs_median_gpu_ms=theirs['median_gpu_ms'])
        worst_rel = max([rel_ref] + ([rel_sg] if rel_sg is not None else []))
        d['worst_relative_difference'] = worst_rel
        d['passes_declared'] = bool(worst_rel <= tol)
        d['passes_second_tier'] = (None if second is None else bool(worst_rel <= second))
        fid[arm] = dict(passed=d['passes_declared'], detail=d)
        gate(f'reproduces_{arm}', d['passes_declared'], d)
    checks['cross_job_fidelity'] = dict(
        passed=(all(v['passed'] for v in fid.values()) if fid else None), detail=fid)

    # ---------------------------- the incumbent must reproduce qtd02 (job 3757505) ----
    q2 = {}
    for arm, spec in cfg.get('qtd02_expectations', {}).items():
        mine = by_arm.get(arm)
        theirs = comp.get('qtd02', {}).get(spec['reproduces'])
        their_sg = comp_sg.get('qtd02', {}).get(spec['reproduces']) or {}
        if mine is None or theirs is None or their_sg.get('same_grid_all') is None:
            q2[arm] = dict(passed=False, detail='arm or qtd02 comparator missing')
            gate(f'reproduces_qtd02_{arm}', False, q2[arm])
            continue
        rels = dict(
            worst_reference=abs(mine['worst_reference_percent'] / 100 - theirs['worst_reference'])
            / max(theirs['worst_reference'], 1e-300),
            same_grid_all=abs(mine['worst_all_times_percent'] - their_sg['same_grid_all'])
            / max(their_sg['same_grid_all'], 1e-300),
            same_grid_evolved=abs(mine['worst_evolved_percent'] - their_sg['same_grid_evolved'])
            / max(their_sg['same_grid_evolved'], 1e-300))
        worst_rel = max(rels.values())
        q2[arm] = dict(passed=bool(worst_rel <= spec['tolerance']), tolerance=spec['tolerance'],
                       probe_tolerance=spec.get('probe_tolerance'),
                       passes_probe=(None if spec.get('probe_tolerance') is None
                                     else bool(worst_rel <= spec['probe_tolerance'])),
                       worst_relative_difference=worst_rel, relative_differences=rels,
                       ours=dict(reference=mine['worst_reference_percent'],
                                 all_times=mine['worst_all_times_percent'],
                                 evolved=mine['worst_evolved_percent'],
                                 gpu_ms=mine['median_gpu_ms'], converged=mine['converged']),
                       qtd02=dict(reference=theirs['worst_reference'] * 100,
                                  all_times=their_sg['same_grid_all'],
                                  evolved=their_sg['same_grid_evolved'],
                                  gpu_ms=theirs['median_gpu_ms'], converged=their_sg['converged']))
        gate(f'reproduces_qtd02_{arm}', q2[arm]['passed'], q2[arm])
    checks['incumbent_reproduces_qtd02'] = dict(
        passed=(all(v['passed'] for v in q2.values()) if q2 else None), detail=q2,
        note='errors only; GPU ms are same-job quantities and are never compared across jobs')
    if q2:
        info('incumbent_reproduces_qtd02_at_1e-9', all(v.get('passes_probe') for v in q2.values()),
             {k: v['worst_relative_difference'] for k, v in q2.items()},
             'the 1e-9 tier qtd02 itself did not meet against cclad01 on three arms; a probe')

    # A probe, not a gate: the same rows against the b-ladder-top envelope job.
    btq = {}
    for arm, other in cfg.get('btq201_expectations', {}).items():
        mine = by_arm.get(arm)
        theirs = comp_sg.get('btq201', {}).get(other)
        if mine is None or theirs is None:
            continue
        btq[arm] = dict(comparator=other, ours_all=mine['worst_all_times_percent'],
                        theirs_all=theirs['same_grid_all'],
                        ours_evolved=mine['worst_evolved_percent'],
                        theirs_evolved=theirs['same_grid_evolved'],
                        ours_gpu_ms=mine['median_gpu_ms'], theirs_gpu_ms=theirs['median_gpu_ms'])
    info('btq201_envelope_probe', True, btq,
         'same-grid comparison against the b-ladder-top envelope job; the EQ rules are '
         'refitted per job so these are not expected to be bitwise')

    # ------------------------------------------------------------- ladders ---
    ladders = {}
    for lid, lad in r['ladders'].items():
        rungs = []
        for rung in lad['rungs']:
            x = by_arm.get(rung['arm'])
            if x is None:
                continue
            rungs.append(dict(q=rung['q'], arm=rung['arm'], M=x['M'], m=x['m'],
                              all_times=x['worst_all_times_percent'],
                              evolved=x['worst_evolved_percent'],
                              t0=x['worst_t0_compression_percent'],
                              reference=x['worst_reference_percent'],
                              best_found=x['worst_best_found_percent'],
                              gpu_ms=x['median_gpu_ms'], converged=x['converged'],
                              budget_exits=x['total_budget_exits'],
                              max_joint_stationarity=x['max_joint_stationarity'],
                              per_time=x['worst_percent_per_time']))
        conv = [u for u in rungs if u['converged']]
        ladders[lid] = dict(
            id=lid, label=lad['label'], dirset=lad['dirset'], rule=lad['rule'],
            quadrature=lad['quadrature'], rungs=rungs,
            all_converged=bool(rungs and all(u['converged'] for u in rungs)),
            monotone_evolved=mono([u['evolved'] for u in rungs]),
            monotone_all_times=mono([u['all_times'] for u in rungs]),
            monotone_evolved_converged_only=mono([u['evolved'] for u in conv]),
            monotone_t0=mono([u['t0'] for u in rungs]),
            regressions_evolved=[dict(from_q=a['q'], to_q=b['q'],
                                      from_percent=a['evolved'], to_percent=b['evolved'])
                                 for a, b in zip(rungs, rungs[1:]) if b['evolved'] > a['evolved']],
            regressions_all_times=[dict(from_q=a['q'], to_q=b['q'],
                                        from_percent=a['all_times'], to_percent=b['all_times'])
                                   for a, b in zip(rungs, rungs[1:])
                                   if b['all_times'] > a['all_times']],
            error_span_evolved=(max(u['evolved'] for u in rungs) / max(
                min(u['evolved'] for u in rungs), 1e-300) if rungs else None),
            cost_span=(max(u['gpu_ms'] for u in rungs) / max(
                min(u['gpu_ms'] for u in rungs), 1e-300) if rungs else None))

    # ------------------------------------------------------- frontier sets ---
    conv_rom = [x for x in rows if x['family'] == 'rom' and x['converged']]
    frontier = dict(
        all_subjects_all_times=nondominated(rows, 'median_gpu_ms', 'worst_all_times_percent'),
        all_subjects_evolved=nondominated(rows, 'median_gpu_ms', 'worst_evolved_percent'),
        converged_rom_all_times=nondominated(conv_rom, 'median_gpu_ms', 'worst_all_times_percent'),
        converged_rom_evolved=nondominated(conv_rom, 'median_gpu_ms', 'worst_evolved_percent'))

    def spans(names, key):
        sel = [by_arm[n] for n in names]
        if len(sel) < 2:
            return dict(points=len(sel), cost_span=None, error_span=None)
        c = [x['median_gpu_ms'] for x in sel]
        v = [x[key] for x in sel]
        return dict(points=len(sel), cost_span=max(c) / max(min(c), 1e-300),
                    error_span=max(v) / max(min(v), 1e-300))

    # Restricted to the pre-registered subject: that config's own primary ladder.
    pc = dict(cfg.get('pass_criteria')
              or {'ladder': 'traj_primary', 'error_span': 2.0, 'cost_span': None})
    primary = ladders.get(pc['ladder'], {})
    prim_names = [u['arm'] for u in primary.get('rungs', []) if u['converged']]
    prim_front = nondominated([by_arm[n] for n in prim_names],
                              'median_gpu_ms', 'worst_evolved_percent')
    verdict = dict(
        ladder=pc['ladder'], criteria=pc,
        criterion_1_evolved_monotone_every_rung_converged=bool(
            primary.get('monotone_evolved') and primary.get('all_converged')),
        criterion_2_all_times_monotone=bool(primary.get('monotone_all_times')),
        criterion_3_error_span=spans(prim_front, 'worst_evolved_percent'),
        converged_nondominated_evolved=prim_front,
        monotone_evolved=primary.get('monotone_evolved'),
        monotone_all_times=primary.get('monotone_all_times'),
        all_converged=primary.get('all_converged'),
        regressions_evolved=primary.get('regressions_evolved'),
        regressions_all_times=primary.get('regressions_all_times'))
    sp = verdict['criterion_3_error_span']
    verdict['criterion_3_passes'] = bool(
        (sp.get('error_span') or 0) >= pc['error_span']
        and (pc.get('cost_span') is None or (sp.get('cost_span') or 0) >= pc['cost_span']))
    verdict['passes'] = bool(verdict['criterion_1_evolved_monotone_every_rung_converged']
                             and verdict['criterion_2_all_times_monotone']
                             and verdict['criterion_3_passes'])
    # Falsification F1: the q = 16 regression on the evolved metric.
    verdict['q16_regression'] = {}
    for lid, lad in ladders.items():
        rg = {u['q']: u['evolved'] for u in lad['rungs']}
        if 0 in rg and 16 in rg:
            verdict['q16_regression'][lid] = dict(
                q0=rg[0], q16=rg[16], regression=bool(rg[16] > rg[0]),
                delta_percentage_points=rg[16] - rg[0])

    # ---------------------------------------------- the three layers at q = 0 ----
    q0 = by_arm.get('q0_M64_dense')
    inc_q0 = (comp_sg.get('qtd02', {}) or {}).get('q0_M64_dense') or {}
    inc_rows = {x['arm']: x for x in json.loads(COMPARATOR_AUDITS['qtd02'].read_text())['arms']} \
        if COMPARATOR_AUDITS['qtd02'].exists() else {}
    inc_q0_row = inc_rows.get('q0_M64_dense', {})
    three_layer = (dict(bank_floor_percent=q0['worst_bank_projection_percent'],
                        best_found_percent=q0['worst_best_found_percent'],
                        solved_all_times_percent=q0['worst_all_times_percent'],
                        solved_evolved_percent=q0['worst_evolved_percent'],
                        t0_compression_percent=q0['worst_t0_compression_percent'],
                        median_gpu_ms=q0['median_gpu_ms'], converged=q0['converged'],
                        incumbent_reference=dict(
                            bank_floor=inc_q0_row.get('worst_bank_projection_percent'),
                            best_found=inc_q0_row.get('worst_best_found_percent'),
                            solved_all_times=inc_q0.get('same_grid_all'),
                            solved_evolved=inc_q0.get('same_grid_evolved'),
                            source='comparators/qtd02-audit.json arm q0_M64_dense (job 3757505)'))
                   if q0 else None)

    # ------------------------------------------------------- the training gates ----
    training = None
    if a.train:
        training = training_gates(Path(a.train), r, gate, a)

    out = dict(result=str(Path(a.result).resolve()),
               result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               elapsed_seconds=r.get('elapsed_seconds'), output_times=r['output_times'],
               cohort=a.cohort, checkpoint_label=r.get('checkpoint_label'),
               checkpoint_sha256=r.get('checkpoint_sha256'), three_layer=three_layer,
               training=training,
               checks=checks, failed=sorted(fail), arms=rows, ladders=ladders,
               frontier=frontier, verdict=verdict,
               direction_sets={k: {kk: vv for kk, vv in v.items()
                                   if kk not in ('singular_values',)}
                               for k, v in r['direction_sets'].items()},
               direction_comparison=r['direction_comparison'],
               direction_cohorts=r['direction_cohorts'],
               comparators=comp_sg)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows),
                          verdict=verdict['passes'],
                          frontier={k: len(v) for k, v in frontier.items()}), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
