"""End-to-end local smoke of the whole seed pipeline at 64 intervals, from a STAGED tree.

Stages attempt `s1` into scratch with the real `cluster/stage.py` (so the staged layout,
the relocated generator dependencies and the byte-check against Git are exercised), then
runs the incumbent's three training scripts with tiny sizes, the ladder driver on the
emitted checkpoint on a development and on a sealed configuration, and the audit on both.
Durations and the audit gate sets are written to `checks/smoke-chain.json`.

Run under the slot helper:
  JAX_DEFAULT_MATMUL_PRECISION=highest <jaxrun-slot> jaxrun /home/tahmid/Dev/.venv/bin/python \
      experiments/b-seeds/smoke_chain.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SCRATCH = Path(os.environ.get('SMOKE_SCRATCH', '/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-'
                              'Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/'
                              'scratchpad/b-seeds-smoke'))
ENV = dict(os.environ, JAX_ENABLE_X64='true', JAX_DEFAULT_MATMUL_PRECISION='highest')


def run(cmd, cwd, env=None, log=None):
    t = time.perf_counter()
    with open(log, 'w') as fh:
        subprocess.run(cmd, cwd=cwd, env=env or ENV, check=True, stdout=fh, stderr=subprocess.STDOUT)
    return time.perf_counter() - t


SMOKE_SEALED_SEED = 424242   # a throwaway: the real sealed seed is never exercised before the final job


def smoke_config(base, sealed):
    c = json.loads(Path(base).read_text())
    if sealed:
        c['eval_seed'] = SMOKE_SEALED_SEED
        c['cohort_note'] = 'SMOKE ONLY: a throwaway sealed-mode cohort, not the declared one'
    c.update(intervals=64, reference_mesh=256, reference_dt=0.00125, residual_snapshots=64,
             residual_starts=2, residual_budget=80, decoder_code_subsample=512,
             q_ladder=[0, 4, 8], comparison_q=[4, 8], fixed_test_count=64, cold_axis_points=24,
             reps=2, recon_starts=4, recon_budget=100, expected_reference_sha256={},
             fidelity_expectations={}, btq201_expectations={}, expected_directions_sha256=None)
    c.pop('qtd02_expectations', None)
    c['ladders'] = [
        dict(id='dense_m4', label='smoke dense M=4(K+q)', dirset='old', rule='m4',
             quadrature='dense', q=[0, 4, 8]),
        dict(id='dense_fixedM', label='smoke dense fixed M=64', dirset='old', rule='mfix',
             quadrature='dense', q=[0, 4])]
    c['attempt'] = 'smoke-' + ('sealed' if sealed else 'dev')
    return c


def main():
    t_all = time.perf_counter()
    reuse = os.environ.get('SMOKE_REUSE') == '1' and (SCRATCH / 's1/output/train/TRAIN-SHA256.txt').exists()
    if SCRATCH.exists() and not reuse:
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    stage = SCRATCH / 's1'
    out = {}
    if reuse:
        # re-stage the current sources over the retained outputs (the staging byte-check still runs)
        shutil.rmtree(stage / 'experiments', ignore_errors=True)
        tmp = SCRATCH / 'restage'
        shutil.rmtree(tmp, ignore_errors=True)
        out['stage_s'] = run([PY, 'experiments/b-seeds/cluster/stage.py', 's1'], ROOT,
                             env=dict(ENV, STAGE_ROOT=str(tmp)), log=SCRATCH / 'stage.log')
        shutil.copytree(tmp / 's1/experiments', stage / 'experiments')
        for nm in ('COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch'):
            shutil.copy2(tmp / 's1' / nm, stage / nm)
        for d in ('ladder_dev', 'ladder_sealed'):
            shutil.rmtree(stage / 'output' / d, ignore_errors=True)
    else:
        out['stage_s'] = run([PY, 'experiments/b-seeds/cluster/stage.py', 's1'], ROOT,
                             env=dict(ENV, STAGE_ROOT=str(SCRATCH)), log=SCRATCH / 'stage.log')
    sd = stage / 'experiments/separable-decoder'
    train = stage / 'output/train'
    train.mkdir(parents=True, exist_ok=True)
    out['reused_training_outputs'] = reuse
    common = dict(ENV, JAX_DEFAULT_MATMUL_PRECISION='highest')
    a = dict(ROUND='3', K='16', LR='1e-3', P_SUB='4096', WD='1e-5', EMA_DECAY='0.999', LAM_ORTH='1e-4',
             MAX_SNAPS='512', T_EARLY='5', FULLROWS='8', N_FF='128', FF_SCALE='4.0', H_HIDDEN='256',
             N_TEST='2', SEED0='1', N='64', R='512', G_HIDDEN='1024', SNAP_NORM='0', STEPS='300',
             TIME_CAP='0', FULL_LAST='50', POOL='0', N_TRAJ='16', TRAIN_ONLY='1',
             OUT_PREFIX=str(train) + '/')
    bank = train / 'sep_burgers_r3_N64_K16_R512.pkl'
    if not reuse:
        out['stageA_s'] = run([PY, 'sep_burgers_r3.py'], sd, env=dict(common, **a), log=SCRATCH / 'A.log')
    assert bank.exists()
    b = dict(N='64', K='16', R='512', MAX_SNAPS='600', T_EARLY='5', N_TEST='2', SEED0='1', LOOSE='1',
             EXTRA_SEED='1000', EXTRA_TRAJ='8', N_TRAJ='24', GEN_CHUNK='8', PROJ_CHUNK='256',
             IDENT_ROWS='8', CKPT=str(bank), OUT_PREFIX=str(train) + '/')
    npz = train / 'sep_coeff_N64_K16_R512.npz'
    if not reuse:
        out['stageB_s'] = run([PY, 'sep_coeff_extract.py'], sd, env=dict(common, **b), log=SCRATCH / 'B.log')
    assert npz.exists()
    ck = train / 'sep_hfit_seed1.pkl'
    c = dict(NPZ=str(npz), CKPT=str(bank), OUT=str(train / 'hfit_full.json'), ARMS='mid', STEPS='300',
             BATCH='4096', LR='1e-3', TIME_CAP='0', ORACLE_ITERS='20', CODEDIAG_N='64',
             CODEDIAG_ITERS='10', ENC_STEPS='200', SEED0='1', EMIT='mid', EMIT_PATH=str(ck))
    if not reuse:
        out['stageC_s'] = run([PY, 'sep_hfit_run.py'], sd, env=dict(common, **c), log=SCRATCH / 'C.log')
        subprocess.run(f'sha256sum {bank} {npz} {ck} > {train}/TRAIN-SHA256.txt', shell=True, check=True)
    assert ck.exists()
    pp = ':'.join(str(stage / f'experiments/{d}') for d in (
        'mr-burgers2d', 'head-ablation', 'cheap-corrections', 'b-ladder-top', 'q-ridge', 'b-seeds'))
    lenv = dict(common, PYTHONPATH=pp, SOURCE_COMMIT=(stage / 'COMMIT.txt').read_text().strip())
    cfgs = {}
    for name, base, sealed in (('dev', 'config-dev-seed1.json', False),
                               ('sealed', 'config-sealed-seed1.json', True)):
        cfg = smoke_config(ROOT / 'experiments/b-seeds' / base, sealed)
        p = stage / f'experiments/b-seeds/config-smoke-{name}.json'
        p.write_text(json.dumps(cfg, indent=2) + '\n')
        cfgs[name] = p
        out[f'ladder_{name}_s'] = run(
            [PY, 'experiments/b-seeds/seeds_run.py', '--config', str(p.relative_to(stage)),
             '--checkpoint', str(ck), '--out', f'output/ladder_{name}'], stage, env=lenv,
            log=SCRATCH / f'ladder_{name}.log')
        assert (stage / f'output/ladder_{name}/COMPLETE').exists()
        audit = SCRATCH / f'audit_{name}.json'
        cmd = [PY, 'experiments/b-seeds/audit_seeds.py', str(stage / f'output/ladder_{name}/result.json'),
               '--fields', str(stage / f'output/ladder_{name}'), '--out', str(audit),
               '--cohort', name, '--train', str(train)]
        if name == 'sealed':
            # a throwaway declared-cohort file for the throwaway seed, so the real one is untouched
            import numpy as np
            rr = np.random.default_rng(SMOKE_SEALED_SEED)
            pc = np.stack([rr.uniform(.15, .85, 6), rr.uniform(.15, .85, 6), rr.uniform(.05, .20, 6),
                           rr.uniform(.5, 2., 6), np.exp(rr.uniform(np.log(.01), np.log(.1), 6))], axis=1)
            sfile = SCRATCH / 'smoke-sealed-cohort.json'
            sfile.write_text(json.dumps(dict(seed=SMOKE_SEALED_SEED, cases=6, physical_cases=pc.tolist(),
                                             sha256_local='smoke')) + '\n')
            cmd += ['--sealed', str(sfile)]
        r = subprocess.run(cmd, cwd=ROOT, env=dict(ENV, JAX_PLATFORMS='cpu'), capture_output=True, text=True)
        (SCRATCH / f'audit_{name}.log').write_text(r.stdout + r.stderr)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        aj = json.loads(audit.read_text())
        # the training gates that MUST fail at smoke sizes, and nothing else may fail
        expected_fail = {'bank_steps_done_300000', 'head_steps_done_200000', 'recipe_values_match_incumbent',
                         'pick_is_131072_with_27648_early', 'training_data_fingerprint_matches_incumbent',
                         'checkpoint_finite_and_shaped'}
        unexpected = sorted(set(aj['failed']) - expected_fail)
        out[f'audit_{name}'] = dict(failed=aj['failed'], unexpected_failures=unexpected,
                                    verdict=aj['verdict']['passes'],
                                    three_layer=aj['three_layer'],
                                    ladders={k: dict(monotone_evolved=v['monotone_evolved'],
                                                     all_converged=v['all_converged'],
                                                     rungs=[(u['q'], round(u['evolved'], 4), round(u['gpu_ms'], 2))
                                                            for u in v['rungs']])
                                             for k, v in aj['ladders'].items()},
                                    gates_passed=sorted(k for k, v in aj['checks'].items() if v['passed']))
        assert not unexpected, unexpected
        assert aj['checks']['final_cohort_flag_matches_cohort']['passed']
        assert aj['checks']['checkpoint_sha256_consistent']['passed']
        if name == 'sealed':
            assert aj['checks']['sealed_cohort_values_match_declared']['passed']
            assert aj['checks']['sealed_cohort_disjoint']['passed']
        else:
            assert aj['checks']['evaluation_cohort_bitwise_abl01']['passed']
    out['seconds'] = time.perf_counter() - t_all
    out['note'] = ('64 intervals, 300 training steps per stage, 16/24 trajectories: a plumbing '
                   'smoke of the staged layout, the three training scripts, the ladder driver on '
                   'both cohorts and the audit; no accuracy claim. The size-dependent training '
                   'gates are expected to fail here and are listed as such.')
    (ROOT / 'experiments/b-seeds/checks/smoke-chain.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('audit')}, indent=1))
    print('B-SEEDS CHAIN SMOKE OK', round(out['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
