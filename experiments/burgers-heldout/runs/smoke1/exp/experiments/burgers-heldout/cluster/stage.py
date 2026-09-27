"""Stage one burgers-heldout job into its own directory (one directory per job).

    python cluster/stage.py <attempt> build            [--gpu a100|a100-80G|h100|h200] [--mem 128G]
    python cluster/stage.py <attempt> hires <config>   [--gpu h200] [--mem 320G]

Every staged source file is checked byte for byte against the Git blob at HEAD (the job runs a
committed tree). Model files that are too large for Git (the bank-floor `cat1024` blocks and this
lane's own trained model) are staged from local git-ignored copies and verified against the SHA256
recorded in the COMMITTED `CKPT-MANIFEST.json` (or, for bank-floor's file, its committed manifest).

Layouts on the cluster (mirroring the two committed pipelines the jobs reuse unchanged):
  code/  the incumbent head-fit job's flat layout (`separable-decoder/runs/dn256b/MANIFEST.sha256`),
         including its exact-Helmholtz patch of `burgers2d_film.py` (hash asserted after patching);
  exp/   the hires-burgers layout (experiments/<name>/*.py on PYTHONPATH).
Raw sbatch: gpu partition, paralab venv, preflight exit 42, f64, JAX_DEFAULT_MATMUL_PRECISION=highest,
all output under the lane namespace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/burgers-heldout'
NAMESPACE = '/cluster/tufts/paralab/tawal01/bheld_20260921'
INCUMBENT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
BANKFLOOR = ROOT.parent / '2026-09-20-bank-floor/experiments/bank-floor'
CAT1024 = dict(path=BANKFLOOR / 'ckpt/bfb03/ckpt/burgers2d_cat1024.pkl',
               sha256='a722b29eac393c929ed63c545c9e4da38d105a9ef5c95c9580e6df56580c53ab')
FILM_PATCHED_SHA = 'c521b36a9bae5471f0f3c1644d03451109c84688cf3808f9c5ea6ffea18493e4'

# the incumbent job's code/ layout: destination -> committed source (hashes matched 2026-09-21)
SD = 'experiments/separable-decoder'
CODE = {f'code/{n}': f'{SD}/{n}' for n in (
    'pod_floor_n256.py', 'sep_burgers.py', 'sep_burgers_r1.py', 'sep_burgers_r2.py', 'sep_burgers_r3.py',
    'sep_burgers_r4.py', 'sep_burgers_r5.py', 'sep_coeff_extract.py', 'sep_common.py', 'sep_hfit.py',
    'sep_hfit_run.py', 'sep_poisson.py', 'sep_poisson_r1.py', 'sep_poisson_r2.py', 'sep_solvers.py',
    'sep_speed_r4.py', 'sep_speed_r5.py')}
MSP = 'experiments/wave2d-rom-latent-stepping/deps/multistage-precision'
CODE.update({
    'code/ctol_eq.py': 'experiments/cost-to-tolerance/ctol_eq.py',
    'code/ctol_tol.py': 'experiments/cost-to-tolerance/ctol_tol.py',
    'code/deps/burgers2d-rom-latent-stepping/blat_common.py': 'experiments/burgers2d-rom-latent-stepping/blat_common.py',
    'code/deps/burgers2d-rom-latent-stepping/blat_train_ad.py': 'experiments/burgers2d-rom-latent-stepping/blat_train_ad.py',
    'code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py':
        'experiments/wave2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py',
    'code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_autodecoder.py': f'{MSP}/ms_autodecoder.py',
    'code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py': f'{MSP}/ms_parametric.py',
    'code/deps/burgers2d-rom-latent-stepping/followup/fu_common.py': 'experiments/burgers2d-rom-latent-stepping/followup/fu_common.py',
    'code/deps/burgers2d-rom-latent-stepping/followup/fu_style.py': 'experiments/burgers2d-rom-latent-stepping/followup/fu_style.py',
    'code/deps/nonlinear-decoder-architecture/nda_arch.py': 'experiments/nonlinear-decoder-architecture/nda_arch.py',
    'code/deps/poisson2d-rom-objective/deps/ms_autodecoder.py': f'{MSP}/ms_autodecoder.py',
    'code/deps/poisson2d-rom-objective/deps/ms_parametric.py': f'{MSP}/ms_parametric.py',
    'code/deps/poisson2d-rom-objective/followup/fu_eq.py': 'experiments/poisson2d-rom-objective/followup/fu_eq.py',
    'code/deps/poisson2d-rom-objective/followup/fu_style.py': 'experiments/burgers2d-rom-latent-stepping/followup/fu_style.py',
    'code/deps/poisson2d-rom-objective/followup/fu_train.py': 'experiments/poisson2d-rom-objective/followup/fu_train.py',
    'code/deps/poisson2d-rom-objective/pro_common.py': 'experiments/poisson2d-rom-objective/pro_common.py',
})
EXP = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
       'experiments/mr-burgers2d/accuracy_paths.py', 'experiments/head-ablation/arms.py',
       'experiments/head-ablation/ladder.py', 'experiments/head-ablation/ablation.py',
       'experiments/cheap-corrections/varpro.py', 'experiments/cheap-corrections/directions.py',
       'experiments/b-ladder-top/topfix.py', 'experiments/b-panel/speed/fast.py',
       'experiments/b-panel/speed/ladders.py', f'{SD}/sep_common.py',
       'experiments/hires-burgers/hops.py', 'experiments/hires-burgers/hfast.py',
       f'{LANE}/bh_bank.py', f'{LANE}/bh_compress.py', f'{LANE}/bh_dirs.py', f'{LANE}/cluster/stage.py']
PYPATH = ['experiments/mr-burgers2d', SD, 'experiments/head-ablation', 'experiments/cheap-corrections',
          'experiments/b-ladder-top', 'experiments/b-panel/speed', 'experiments/hires-burgers', LANE]
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}

HEAD = '''#!/bin/bash
#SBATCH --job-name=bh_{attempt}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres={gres}
{constraint}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={memfrac}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
EXPPATH="{exppath}"
'''

BUILD = '''
# ---- 1. seed checkpoint: cat1024 exactly, as ONE separable parameter set (R = 1024)
PYTHONPATH="$EXPPATH" "$PY" exp/{lane}/bh_compress.py seed1024 --blocks in/burgers2d_cat1024.pkl \\
  --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out in/cat1024_seed.pkl
# ---- 2. the incumbent's extraction, unchanged, on cat1024 (131072 states of 4608 trajectories)
( cd code && env N=256 K=16 R=1024 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \\
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \\
  CKPT="$TASK_ROOT/in/cat1024_seed.pkl" OUT_PREFIX="$TASK_ROOT/output/x1024_" "$PY" sep_coeff_extract.py )
# ---- 3. POD inside the cat1024 span, floors on dev6/sel32, pre-registered selection, R = 512 seed
PYTHONPATH="$EXPPATH" "$PY" exp/{lane}/bh_compress.py compress --npz output/x1024_sep_coeff_N256_K16_R1024.npz \\
  --blocks in/burgers2d_cat1024.pkl --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out output/compress
# ---- 4. the incumbent's extraction, unchanged, on cpod512
( cd code && env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \\
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \\
  CKPT="$TASK_ROOT/output/compress/cpod512_seed.pkl" OUT_PREFIX="$TASK_ROOT/output/x512_" "$PY" sep_coeff_extract.py )
# ---- 5. the incumbent's head fit, unchanged recipe (arm mid, from scratch), emitted as the new model
( cd code && env NPZ="$TASK_ROOT/output/x512_sep_coeff_N256_K16_R512.npz" \\
  CKPT="$TASK_ROOT/output/compress/cpod512_seed.pkl" OUT="$TASK_ROOT/output/hfit_bh.json" \\
  ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 ORACLE_ITERS=150 CODEDIAG_N=2048 \\
  CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 EMIT=mid EMIT_PATH="$TASK_ROOT/output/bh_model.pkl" \\
  "$PY" sep_hfit_run.py )
# ---- 6. correction directions by the incumbent's rule (qtd02 configuration)
PYTHONPATH="$EXPPATH" "$PY" exp/{lane}/bh_dirs.py --checkpoint output/bh_model.pkl --out output/dirs
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''

HIRES = '''
PYTHONPATH="$EXPPATH" "$PY" exp/{lane}/bh_hires.py --config exp/{lane}/{config} --checkpoint in/bh_model.pkl \\
  --inputs in --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('kind', choices=['build', 'hires'])
    p.add_argument('config', nargs='?')
    p.add_argument('--gpu', default='a100', choices=sorted(GRES))
    p.add_argument('--mem', default='128G')
    p.add_argument('--hours', type=int, default=12)
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []

    def put(dest, src, content=None, want=None):
        if content is None:
            content = (ROOT / src).read_bytes()
            assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{src}']), \
                f'not committed: {src}'
        if want is not None:
            assert sha(content) == want, (dest, sha(content), want)
        d = out / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(content)
        proof.append(dict(dest=dest, source=str(src), bytes=len(content), sha256=sha(content), commit=commit))

    exp = list(EXP)
    if a.kind == 'build':
        for dest, src in CODE.items():
            put(dest, src)
        film = 'code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py'
        subprocess.run(['python3', str(ROOT / SD / 'cluster/patch_bf_precond.py'), str(out / film)], check=True)
        got = sha((out / film).read_bytes())
        assert got == FILM_PATCHED_SHA, ('patched film differs from the incumbent job', got)
        proof.append(dict(dest=film, patched_by=f'{SD}/cluster/patch_bf_precond.py', sha256=got,
                          equals_incumbent_job_manifest=True))
        put('in/burgers2d_cat1024.pkl', CAT1024['path'], CAT1024['path'].read_bytes(), CAT1024['sha256'])
        put('in/sep_hfit_dense_mid_N256_dense.pkl', INCUMBENT)
        body = BUILD
    else:
        assert a.config, 'hires needs a config'
        cfg = json.loads((ROOT / LANE / a.config).read_text())
        exp += [f'{LANE}/bh_hires.py', f'{LANE}/{a.config}']
        man = json.loads((ROOT / LANE / 'CKPT-MANIFEST.json').read_text())
        subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/CKPT-MANIFEST.json'])
        for key in (cfg['model_key'], cfg['directions_key']):
            m = man[key]
            put(f"in/{Path(m['staged_as']).name}", m['local_path'], (ROOT / m['local_path']).read_bytes(), m['sha256'])
        body = HIRES
    for f in exp:
        put(f'exp/{f}', f)
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = HEAD.format(attempt=a.attempt, gres=gres, constraint=('#SBATCH ' + constraint) if constraint else '',
                         mem=a.mem, hours=a.hours, remote=remote, memfrac=a.mem_fraction,
                         exppath=':'.join('$TASK_ROOT/exp/' + x for x in PYPATH))
    script += body.format(lane=LANE, config=a.config)
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{sha(p_.read_bytes())}  {p_.relative_to(out)}' for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
