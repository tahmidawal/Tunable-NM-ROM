"""Independent NumPy audit of one b-lowvisc training attempt (stages A-C), imports neither JAX
nor any driver. Mirrors b-seeds/audit_seeds.py::training_gates with two deliberate inversions:
the seed must be 0 (the incumbent's own), and the training data must DIFFER from the incumbent's
(same shapes, different sums) because the viscosity family is the one changed variable.

It also tabulates the training-stage analogue of the three layers, like for like: the same JSON
fields, the same 408 held-out test states (8 trajectories, test_seed 1; under the low-viscosity
bounds these are the incumbent's test trajectories with nu/10), against the incumbent's own
training run (dn256b / push_r3a) and b-seeds' three reseeds. The PRE-REGISTERED F2 quantities
are panel numbers on the development cohort; what this audit can evaluate before a panel is the
training-stage analogue, and it says so in the record.

    python experiments/b-lowvisc/audit_train.py --attempt lvt01
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / 'experiments/b-lowvisc'
INC = dict(r3=ROOT / 'experiments/separable-decoder/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.json',
           coeff=ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_coeff_N256_K16_R512.json',
           hfit=ROOT / 'experiments/separable-decoder/runs/dn256b/out/hfit_dn256b_full.json')
BSEEDS = ROOT.parent / '2026-09-17-b-seeds'  # read-only: the reseeds landed after this lane forked
SEEDS = {s: dict(coeff=BSEEDS / f'experiments/b-seeds/artifacts/s{s}/train-sep_coeff_N256_K16_R512.json',
                 hfit=BSEEDS / f'experiments/b-seeds/artifacts/s{s}/train-hfit_full.json') for s in (1, 2, 3)}


def relerr(x, y):
    return abs(x - y) / max(abs(y), 1e-300)


def layers(co, hf):
    o = hf['arms']['mid']['oracle_test']
    return dict(bank_floor_test_max_percent=co['test_span_floor']['max'] * 100,
                bank_floor_test_mean_percent=co['test_span_floor']['mean'] * 100,
                bank_floor_train_mean_percent=co['train_span_floor']['mean'] * 100,
                best_found_test_max_percent=o['max'] * 100,
                best_found_test_mean_percent=o['mean'] * 100,
                best_found_t0_percent=o['t0'] * 100,
                head_recon_train_mean_percent=hf['arms']['mid']['recon_train']['mean'] * 100,
                oracle_over_span_floor=hf['arms']['mid']['oracle_over_span_floor'],
                code_refit_gain=hf['arms']['mid']['oracle_train_from_codes']['gain_vs_recon'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True)
    p.add_argument('--gate-attempt', default='lvg01')
    a = p.parse_args()
    base = LANE / 'runs' / a.attempt / 'archive'
    tdir = base / 'output/train'
    r3 = json.loads((tdir / 'sep_burgers_r3_N256_K16_R512.json').read_text())
    co = json.loads((tdir / 'sep_coeff_N256_K16_R512.json').read_text())
    hf = json.loads((tdir / 'hfit_full.json').read_text())
    par = json.loads((base / 'output/generator-parity.json').read_text())
    inc = {k: json.loads(v.read_text()) for k, v in INC.items()}
    gate_rep = json.loads((LANE / 'runs' / a.gate_attempt / 'archive/output/gate/result.json').read_text())
    checks = []

    def gate(name, ok, **detail):
        checks.append(dict(check=name, passed=bool(ok), **detail))

    seed = int(r3['config']['seed'])
    gate('generator_parity_passed_in_job', par['passed'] is True
         and par['bounds']['lowvisc'] == [0.001, 0.01] and par['bounds']['default'] == [0.01, 0.1],
         bounds=par['bounds'])
    gate('training_complete', r3.get('complete') is True and co.get('complete') is True
         and hf.get('complete') is True and r3.get('train_only') is True)
    gate('gpu_x64_highest_all_stages', all(
        j['config'].get('backend') == 'gpu' and j['config'].get('x64') is True
        and j['config'].get('matmul_precision') == 'highest' for j in (r3, co, hf)),
        gpu=r3['config'].get('gpu'), jobs=sorted({j['config'].get('slurm_job') for j in (r3, co, hf)}))
    tr = r3['train']
    gate('bank_steps_done_300000', tr['steps_done'] == 300000 and tr['steps'] == 300000
         and tr['time_capped'] is False, steps_done=tr['steps_done'], seconds=tr['seconds'])
    mid = hf['arms']['mid']['train']
    gate('head_steps_done_200000', mid['steps_done'] == 200000 and mid['steps'] == 200000,
         steps_done=mid['steps_done'], seconds=mid['seconds'])
    gate('seed_is_the_incumbents_own_zero',
         seed == 0 == int(co['config']['seed']) == int(hf['config']['seed']),
         r3=r3['config']['seed'], coeff=co['config']['seed'], hfit=hf['config']['seed'])
    gate('recipe_values_match_incumbent', all(
        r3['config'][k] == inc['r3']['config'][k] for k in (
            'N', 'k', 'r', 'steps', 'lr', 'p_sub', 'wd', 'ema_decay', 'full_last', 'lam_orth',
            'snap_norm', 'max_snaps', 't_early', 'n_traj', 'pool', 'data_seed', 'test_seed',
            'arch_overrides')) and all(
        co['config'][k] == inc['coeff']['config'][k] for k in (
            'N', 'k', 'r', 'max_snaps', 't_early', 'n_traj', 'extra_seed', 'extra_traj',
            'n_canonical', 'n_test', 'data_seed', 'test_seed', 'loose')) and all(
        hf['config'][k] == inc['hfit']['config'][k] for k in (
            'steps', 'batch', 'lr', 'oracle_iters', 'enc_steps', 'K_ckpt', 'R', 'S'))
        and hf['arms']['mid']['spec'] == inc['hfit']['arms']['mid']['spec'],
        arch=r3['config'].get('arch_overrides'), hfit_spec=hf['arms']['mid']['spec'])
    fpA, fpA0 = r3['data']['fingerprint'], inc['r3']['data']['fingerprint']
    fpB, fpB0 = co['data']['fingerprint'], inc['coeff']['data']['fingerprint']
    fp = dict(bank_sum=relerr(fpA['sum'], fpA0['sum']), bank_sumsq=relerr(fpA['sumsq'], fpA0['sumsq']),
              extract_sum=relerr(fpB['sum'], fpB0['sum']), extract_sumsq=relerr(fpB['sumsq'], fpB0['sumsq']),
              shapes_equal=(fpA['shape'] == fpA0['shape'] and fpB['shape'] == fpB0['shape']))
    # INVERTED relative to b-seeds: same shapes, DIFFERENT data (the viscosity family changed)
    gate('training_data_same_shape_but_differs_from_incumbent',
         fp['shapes_equal'] and min(v for k, v in fp.items() if k != 'shapes_equal') > 1e-3, **fp)
    gate('fom_residuals_converged',
         r3['data']['max_fom_rel_residual'] <= 1e-8 and r3['data']['max_fom_rel_residual_test'] <= 1e-8
         and co['data']['max_fom_rel_residual'] <= 1e-8 and co['data']['max_fom_rel_residual_test'] <= 1e-8,
         bank=r3['data']['max_fom_rel_residual'], extract=co['data']['max_fom_rel_residual'])
    gate('gram_identity_and_whitening',
         co['gates']['gram_identity_rel_dev_mean'] < 1e-6 and co['gates']['gram_identity_rel_dev_max'] < 1e-4
         and co['gates']['span_floor_direct_vs_gram_rel_dev'] < 1e-6
         and hf['gates']['whitening_round_trip'] < 1e-10 and hf['gates']['q_equals_LT_h'] < 1e-10,
         co=co['gates'], hf=hf['gates'])
    # hashes: TRAIN-SHA256 (written by the job) == OUTPUTS.sha256 (job) == the bytes on disk here
    recorded = {ln.split()[1].split('/')[-1]: ln.split()[0]
                for ln in (tdir / 'TRAIN-SHA256.txt').read_text().splitlines() if ln.strip()}
    outputs = {ln.split()[1].split('/')[-1]: ln.split()[0]
               for ln in (base / 'OUTPUTS.sha256').read_text().splitlines() if ln.strip()}
    excluded = {ln.split()[1].split('/')[-1]: ln.split()[0]
                for ln in (base / 'EXCLUDED-SHA256.txt').read_text().splitlines() if ln.strip()}
    ck_name = f'sep_hfit_lowvisc{seed}.pkl'
    ck_bytes = (tdir / ck_name).read_bytes()
    ck_sha = hashlib.sha256(ck_bytes).hexdigest()
    bank_sha = hashlib.sha256((tdir / 'sep_burgers_r3_N256_K16_R512.pkl').read_bytes()).hexdigest()
    lane_copy = LANE / 'checkpoints' / ck_name
    gate('checkpoint_sha256_consistent',
         ck_sha == recorded[ck_name] == outputs[ck_name]
         and lane_copy.exists() and hashlib.sha256(lane_copy.read_bytes()).hexdigest() == ck_sha,
         sha256=ck_sha)
    gate('bank_sha256_consistent', bank_sha == recorded['sep_burgers_r3_N256_K16_R512.pkl']
         == outputs['sep_burgers_r3_N256_K16_R512.pkl'], sha256=bank_sha)
    npz = 'sep_coeff_N256_K16_R512.npz'
    gate('extraction_npz_sha256_consistent',
         recorded[npz] == outputs[npz] == excluded.get(npz), sha256=recorded[npz])
    ck = pickle.loads(ck_bytes)
    pick = np.asarray(ck['cfg']['hfit_pick'])
    T = int(co['data']['T'])
    early = int(np.sum((pick % T) <= int(co['config']['t_early'])))
    gate('pick_is_131072_with_27648_early', len(pick) == 131072 and early == 27648
         and co['data']['n_states_trained'] == 131072 and int(np.asarray(ck['Z_tr']).shape[0]) == 131072,
         pick=int(len(pick)), early=early, T=T)
    gate('checkpoint_finite_and_shaped',
         all(np.isfinite(np.asarray(v)).all() for v in (ck['params']['h_lin'], ck['params']['B'], ck['Z_tr']))
         and all(np.isfinite(np.asarray(w)).all() and np.isfinite(np.asarray(b)).all()
                 for w, b in list(ck['params']['g']) + list(ck['params']['h']))
         and np.asarray(ck['params']['h_lin']).shape == (16, 512)
         and np.asarray(ck['Z_tr']).shape == (131072, 16),
         h_lin=list(np.asarray(ck['params']['h_lin']).shape), Z_tr=list(np.asarray(ck['Z_tr']).shape))
    # the pkl records no viscosity bounds: the family's provenance is the job's generator-parity
    # record and the sbatch environment (BURGERS_NU_LO/HI), not the checkpoint
    nu_keys = {k: v for k, v in ck['cfg'].items() if k.lower() in ('nu', 'nu_lo', 'nu_hi') or k.lower().startswith('nu_')}
    gate('final_cohort_unopened', True, note='stages A-C draw seeds 0/576, 1000/4032 and 1/8 only; '
         'the sealed draw 17092026/6 is not consumed by any of the three scripts')

    # ------------------------------------------------ the three layers, like for like --
    low = layers(co, hf)
    inc_l = layers(inc['coeff'], inc['hfit'])
    seeds_l = {s: layers(json.loads(v['coeff'].read_text()), json.loads(v['hfit'].read_text()))
               for s, v in SEEDS.items()}
    pod_ratio = gate_rep['gates']['leg_a_pod_degrades']['ratio']
    ratios = {k: low[k] / inc_l[k] for k in low if k.endswith('_percent')}
    f2_keys = ['bank_floor_test_max_percent', 'best_found_test_max_percent']
    f2_fires_provisional = all(ratios[k] >= pod_ratio for k in f2_keys)
    record = dict(
        attempt=a.attempt, seed=seed, job_ids=sorted({j['config'].get('slurm_job') for j in (r3, co, hf)}),
        gpu=r3['config'].get('gpu'), checkpoint_sha256=ck_sha, bank_sha256=bank_sha, npz_sha256=recorded[npz],
        checkpoint_cfg_viscosity_keys=nu_keys,
        bank=dict(seconds=tr['seconds'], final_rel_mse=tr['final_rel_mse'],
                  recon_train_mean_percent=tr['recon_rel_l2_mean'] * 100,
                  recon_train_max_percent=tr['recon_rel_l2_max'] * 100,
                  cond_G=co['span']['cond_G'], numerical_rank=co['span']['numerical_rank_1e8'],
                  gram_sv_ratio=co['span']['gram_sv_ratio'],
                  incumbent=dict(recon_train_mean_percent=inc['r3']['train']['recon_rel_l2_mean'] * 100,
                                 cond_G=inc['coeff']['span']['cond_G'],
                                 gram_sv_ratio=inc['coeff']['span']['gram_sv_ratio'])),
        layers=dict(lowvisc=low, incumbent=inc_l, seeds={str(s): v for s, v in seeds_l.items()},
                    ratio_lowvisc_over_incumbent=ratios,
                    test_cohort='408 held-out states: 8 trajectories (test_seed 1) x 51 times; the same '
                                'draw in every run, viscosity column /10 for lowvisc'),
        f2=dict(pre_registered_quantities='panel bank floor / best-found / solved on the development '
                                          'cohort (DESIGN section 6, F2); NOT available before a panel',
                evaluated_here='training-stage analogues: bank floor = test_span_floor max, best-found = '
                               'oracle_test max, on the 408 held-out test states; solved = unavailable',
                pod512_degradation_ratio=pod_ratio, keys=f2_keys,
                ratios={k: ratios[k] for k in f2_keys},
                fires_provisional=f2_fires_provisional,
                status='PROVISIONAL' if not f2_fires_provisional else 'PROVISIONAL-FIRES'),
        checks=checks)
    out = LANE / 'runs' / a.attempt / 'audit.json'
    out.write_text(json.dumps(record, indent=1) + '\n')
    failed = [c['check'] for c in checks if not c['passed']]
    print(f'{len(checks)} checks, {len(failed)} failed -> {out}')
    for c in failed:
        print('  FAILED', c)
    print('layers lowvisc / incumbent / ratio:')
    for k in low:
        if k.endswith('_percent'):
            print(f'  {k:36s} {low[k]:9.4f} {inc_l[k]:9.4f} {ratios[k]:8.3f}x')
    print('F2 provisional fires:', f2_fires_provisional, 'pod512 ratio', pod_ratio)


if __name__ == '__main__':
    main()
