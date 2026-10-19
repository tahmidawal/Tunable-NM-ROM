"""jcp-robust E2 training code: CPU-only checks of train2d/ and train3d/ (no GPU, no cluster).

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-robust/train_tests.py            # static checks, seconds
    ... train_tests.py --dry3d          # + 3D chaining run (smoke-size config, seed 1 seeds), ~1 min CPU
    ... train_tests.py --dry2d          # + 2D chaining run Stage A -> B -> rotation (N=64, 20 steps), ~5 min CPU
    ... train_tests.py --rotation-repro # + incumbent rotation reproduced from the dn256b dense checkpoint, ~15 min CPU

Static checks: (1) the per-seed 3D configs differ from the incumbent config only in model_seed/subset_seed (the pilot
also in bank_steps/warmup) with the registered values; (2) every FILES.txt entry exists; (3) every PROVENANCE.json
sha256 matches the file, and every 2D Stage-B file matches the dn256b job's MANIFEST.sha256, every 3D vendored file
the j3b job's MANIFEST.sha256 (when those manifests are present); (4) both run_seed.sh parse (bash -n);
(5) no runtime file names an absolute /home or /cluster path; (6) the dn256b sbatch's frozen extraction/head-fit
environment appears verbatim in train2d/run_seed.sh's production branch.
The dry runs copy exactly the FILES.txt entries into a scratch job root under /tmp/claude-1002/jcp-robust-train-scratch/
(deleted afterwards unless --keep) and run run_seed.sh there with JAX_PLATFORMS=cpu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parent                    # experiments/jcp-robust
REPO = LANE.parents[1]
T2, T3 = LANE / 'train2d', LANE / 'train3d'
PY = '/home/tahmid/Dev/.venv/bin/python'
SCRATCH = Path('/tmp/claude-1002/jcp-robust-train-scratch')
DN256B_MANIFEST = REPO / 'experiments/separable-decoder/runs/dn256b/MANIFEST.sha256'
DN256B_SBATCH = REPO / 'experiments/separable-decoder/cluster/run_dn256b.sbatch'
J3B_CONFIG_SHA = '1a3c479f636a732bf57a9f8f955d3ba1a0aaf354da4cbeded9c17591cc05241c'   # j3b MANIFEST entry
J3B_TRAIN2W_SHA = '13c45b8719e7dee445ca15b56391befee9b52eec6a53ac4ed46a3d3dafa5ba3b'
INC_ROT_SHA = dict(T='bedc86d9260a3e41a2d663e7338ff135c9d01f20d43d4af79a6bdbfcd7a9327d')   # rotation_R512.json T_sha256

fails = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        fails.append(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files_list(d: Path):
    return [l.strip() for l in (d / 'FILES.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]


def static():
    print('[1] 3D configs differ only in the seeds')
    base = json.loads((T3 / 'configs/train_r1024.json').read_text())
    check(sha(T3 / 'configs/train_r1024.json') == J3B_CONFIG_SHA, 'train_r1024.json = the j3b job config (sha256)')
    check(sha(T3 / 'train2w.py') == J3B_TRAIN2W_SHA, 'train2w.py = the j3b job script (sha256)')
    for s in (1, 2, 3):
        c = json.loads((T3 / f'configs/train_r1024_s{s}.json').read_text())
        diff = sorted(k for k in base if c.get(k) != base[k])
        check(set(c) == set(base) and diff == ['model_seed', 'subset_seed']
              and c['model_seed'] == 923715 + 1000 * s and c['subset_seed'] == 923002 + 1000 * s,
              f's{s}: only model_seed={c["model_seed"]}, subset_seed={c["subset_seed"]} changed')
    p = json.loads((T3 / 'configs/train_r1024_pilot.json').read_text())
    diff = sorted(k for k in base if p.get(k) != base[k])
    check(set(p) == set(base) and diff == ['bank_steps', 'model_seed', 'subset_seed', 'warmup']
          and p['bank_steps'] == 200 and p['warmup'] == 100 and p['model_seed'] == 924715,
          'pilot: seed-1 seeds, bank_steps=200, warmup=100, nothing else')
    check(p['warmup'] < p['bank_steps'], 'pilot warmup < bank_steps (optax cosine schedule needs positive decay steps)')
    # make_config reproduces the committed files
    for s in (1, 2, 3):
        out = SCRATCH / f'cfg_s{s}.json'
        subprocess.run([PY, '-I', str(T3 / 'make_config.py'), '--seed', str(s), '--out', str(out)], check=True,
                       capture_output=True)
        check(out.read_bytes() == (T3 / f'configs/train_r1024_s{s}.json').read_bytes(), f'make_config.py regenerates s{s}')
        out.unlink()

    print('[2] FILES.txt entries exist')
    for d in (T2, T3):
        fl = files_list(d)
        missing = [f for f in fl if not (REPO / f).is_file()]
        check(not missing, f'{d.name}/FILES.txt: {len(fl)} entries, missing {missing}')

    print('[3] PROVENANCE sha256')
    man = {}
    if DN256B_MANIFEST.exists():
        man = {l.split()[1]: l.split()[0] for l in DN256B_MANIFEST.read_text().splitlines()}
    for d, root in ((T2, T2), (T3, LANE)):
        pv = json.loads((d / 'PROVENANCE.json').read_text())['files']
        bad = [f['path'] for f in pv if sha(root / f['path']) != f['sha256']]
        check(not bad, f'{d.name}/PROVENANCE.json: {len(pv)} files, sha mismatches {bad}')
        unedited = [f for f in pv if not f['edited']]
        listed = set(files_list(d))
        notlisted = [f['path'] for f in pv if str((root / f['path']).relative_to(REPO)) not in listed]
        check(not notlisted, f'{d.name}: every PROVENANCE file is in FILES.txt (missing {notlisted})')
        check(all('source_commit' in f or f.get('already_in_lane') for f in unedited), f'{d.name}: sources recorded')
    if man:
        pv = json.loads((T2 / 'PROVENANCE.json').read_text())['files']
        mk = [f for f in pv if 'matches_dn256b_MANIFEST' in f]
        bad = [f['path'] for f in mk if man[f['matches_dn256b_MANIFEST'].split(':: ')[1]] != f['sha256']]
        check(len(mk) >= 10 and not bad, f'{len(mk)} Stage-B/rotation files equal their dn256b MANIFEST entries {bad}')
    else:
        print('  skip dn256b MANIFEST not present')

    print('[4] bash -n')
    for d in (T2, T3):
        r = subprocess.run(['bash', '-n', str(d / 'run_seed.sh')], capture_output=True, text=True)
        check(r.returncode == 0, f'{d.name}/run_seed.sh parses {r.stderr.strip()}')

    print('[5] no absolute /home or /cluster paths in runtime python files')
    for d in (T2, T3):
        hits = []
        for f in files_list(d):
            if f.endswith('.py') and re.search(r'["\'](/home/|/cluster/)', (REPO / f).read_text()):
                hits.append(f)
        check(not hits, f'{d.name}: {hits}')

    print('[6] the frozen dn256b environment is in run_seed.sh')
    rs = (T2 / 'run_seed.sh').read_text()
    for frag in ('K=16 R=512', 'T_EARLY=5 SEED0=0 LOOSE=1 EXTRA_SEED=1000 GEN_CHUNK=64 PROJ_CHUNK=256',
                 'MAX_SNAPS=131072 EXTRA_TRAJ=4032 N_TEST=8', 'IDENT_ROWS=64',
                 'STEPS=200000 ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000',
                 'ARMS=mid,wide,k32', 'BATCH=4096 LR=1e-3', 'TIME_CAP=1500 SEED0="$SEED_S" EMIT=mid',
                 '--arm base --sigma 4 --lam 0 --seed "$SEED_S"', 'A_EXTRA=(--nodes 256)'):
        check(frag in rs, f'run_seed.sh contains {frag!r}')
    if DN256B_SBATCH.exists():
        sb = DN256B_SBATCH.read_text()
        for frag in ('N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1',
                     'EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64',
                     'ARMS=mid,wide,k32 STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500',
                     'ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0'):
            check(frag in sb, f'(reference) dn256b sbatch contains {frag!r}')


def jobroot(name, d):
    root = SCRATCH / name
    shutil.rmtree(root, ignore_errors=True)
    for f in files_list(d):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, root / f)
    return root


def run(cmd, cwd, env, log):
    with open(log, 'w') as fh:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode


def dry2d(keep):
    print('[dry2d] Stage A -> Stage B -> rotation, N=64, CPU')
    root = jobroot('dry2d', T2)
    env = dict(os.environ, JAX_PLATFORMS='cpu', TRAIN2D_DRYRUN='1', PY=PY)
    rc = run(['bash', 'experiments/jcp-robust/train2d/run_seed.sh', '1', str(root / 'out')], root, env, root / 'log.txt')
    check(rc == 0 and (root / 'out/SEED-COMPLETE').exists(), f'2D chain completes (rc={rc}, log {root}/log.txt)')
    if rc == 0:
        rec = json.loads((root / 'out/run_record.json').read_text())
        check(rec['stageB_mid']['steps_done'] == 20 and rec['rotation']['L_times_T_identity_deviation'] < 1e-8,
              f"mid steps_done={rec['stageB_mid']['steps_done']}, ||LT-I||={rec['rotation']['L_times_T_identity_deviation']:.1e}")
        ck = root / 'out/stageB/sep_hfit_dense_mid_N64_dense.pkl'
        check(rec['rotation']['checkpoint_sha256'] == sha(ck), 'rotation was computed from the emitted dense mid checkpoint')
        check((root / 'out/stageA/DIAGNOSTIC-ONLY_ls_rotation.npz').exists() and not (root / 'out/stageA/rotation.npz').exists(),
              "smoothtrain's own rotation is quarantined as DIAGNOSTIC-ONLY")
        ex = json.loads((root / 'out/stageB/sep_coeff_N64_K16_R512.json').read_text())
        check(ex['config']['seed'] == 0 and ex['config']['extra_seed'] == 1000, 'extraction data stream SEED0=0, EXTRA_SEED=1000')
        hf = json.loads((root / 'out/stageB/hfit_full.json').read_text())
        check(hf['config']['seed'] == 1, 'head fit SEED0 = recipe seed')
    if not keep:
        shutil.rmtree(root, ignore_errors=True)


def dry3d(keep):
    print('[dry3d] train2w.py smoke-size config with seed-1 seeds, CPU')
    root = jobroot('dry3d', T3)
    cfg = root / 'smoke_s1.json'
    subprocess.run([PY, '-I', str(T3 / 'make_config.py'), '--seed', '1', '--base',
                    str(T3 / 'configs/train_smoke_local.json'), '--out', str(cfg)], check=True, capture_output=True)
    env = dict(os.environ, JAX_PLATFORMS='cpu', TRAIN3D_CONFIG=str(cfg), PY=PY)
    rc = run(['bash', 'experiments/jcp-robust/train3d/run_seed.sh', '1', str(root / 'out')], root, env, root / 'log.txt')
    check(rc == 0 and (root / 'out/SEED-COMPLETE').exists(), f'3D chain completes (rc={rc}, log {root}/log.txt)')
    if rc == 0:
        t = json.loads((root / 'out/training.json').read_text())
        rec = json.loads((root / 'out/run_record.json').read_text())
        check(t['config']['model_seed'] == 924715 and t['config']['subset_seed'] == 924002, 'seeds reached train2w.py')
        check(rec['selected_step'] == t['bank']['selected_step'] is not None, f"selected step recorded ({rec['selected_step']})")
        check('compare_bank_floors_full' in t, 'compare_bank resolved through the job-root link')
    if not keep:
        shutil.rmtree(root, ignore_errors=True)


def rotation_repro(keep):
    print('[rotation-repro] incumbent rotation from the dn256b dense checkpoint (CPU, ~15 min)')
    ck = REPO / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
    if not ck.exists():
        print('  skip dn256b dense checkpoint not present')
        return
    import numpy as np
    out = SCRATCH / 'rotrepro'
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest', PYTHONPATH=str(T2 / 'rotation'))
    rc = run([PY, 'make_rotation.py', '--checkpoint', str(ck), '--directions', 'inputs/directions_qtd02.npz',
              '--out', str(out)], T2 / 'rotation', env, SCRATCH / 'rotrepro.log')
    check(rc == 0, 'make_rotation.py ran on the incumbent checkpoint')
    if rc == 0:
        inc = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                       'exp/2026-09-23-burgers-bank-knob:experiments/burgers-bank-knob/inputs/rotation_R512.npz'])
        (out / 'inc.npz').write_bytes(inc)
        a, b = np.load(out / 'rotation_R512.npz'), np.load(out / 'inc.npz')
        sv = float(np.abs(a['singular_values'] - b['singular_values']).max() / b['singular_values'].max())
        M = a['L'] @ b['T']
        off = float(np.abs(M[:384, :384] - np.diag(np.diag(M[:384, :384]))).max())
        dg = float(np.abs(np.abs(np.diag(M[:384, :384])) - 1).max())
        check(sv < 1e-10 and off < 1e-8 and dg < 1e-8,
              f'reproduces the incumbent rotation: sv rel {sv:.1e}, |L_new T_inc| off-diag {off:.1e}, |diag|-1 {dg:.1e} (r<=384)')
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
        (SCRATCH / 'rotrepro.log').unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry2d', action='store_true')
    ap.add_argument('--dry3d', action='store_true')
    ap.add_argument('--rotation-repro', action='store_true')
    ap.add_argument('--keep', action='store_true', help='keep scratch job roots')
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    static()
    if a.dry3d:
        dry3d(a.keep)
    if a.dry2d:
        dry2d(a.keep)
    if a.rotation_repro:
        rotation_repro(a.keep)
    print(f'\n{"ALL OK" if not fails else f"{len(fails)} FAILED"}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
