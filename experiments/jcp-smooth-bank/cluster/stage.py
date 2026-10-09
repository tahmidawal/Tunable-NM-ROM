"""Stage one jcp-smooth-bank job into runs/<attempt>/ from the committed tree (every file from `git show HEAD:<path>`,
working copy must equal HEAD), write a multi-GPU sbatch (one process per GPU, own output dir, own log) and a manifest.

    python cluster/stage.py jobs/<attempt>.json [--gpu a100-80G|h100]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/jcp-smooth-bank'
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcpsmooth'
BASE_FILES = [f'{LANE}/smoothtrain.py', f'{LANE}/deps/burgers2d_film.py', f'{LANE}/deps/sep_common.py',
              f'{LANE}/deps/sep_solvers_reference.py', f'{LANE}/cluster/stage.py',
              'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
GRES = {'a100-80G': ('a100', '#SBATCH --constraint=a100-80G'), 'a100': ('a100', ''), 'h100': ('h100', '')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job')
    ap.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    a = ap.parse_args()
    job = json.loads((ROOT / LANE / a.job).read_text())
    att = job['attempt']
    assert att.isalnum(), att
    files = (BASE_FILES + [f'{LANE}/{a.job}'] + [f'{LANE}/{f}' for f in job.get('extra_files', [])]
             + list(job.get('extra_repo_files', [])))
    out = ROOT / LANE / 'runs' / att
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{att}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l[3:].startswith(f'{LANE}/runs')], dirty
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    for d in job.get('data', []):                   # non-git inputs (references, trained banks), copied with sha256
        src = Path(d['src']) if Path(d['src']).is_absolute() else ROOT / LANE / d['src']
        srcs = sorted(p_ for p_ in src.rglob('*') if p_.is_file()) if src.is_dir() else [src]
        for f in srcs:
            rel = f.relative_to(src) if src.is_dir() else Path(f.name)
            dest = out / LANE / d['dest'] / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = f.read_bytes()
            dest.write_bytes(content)
            proof.append(dict(source=str(f), dest=str(dest.relative_to(out)), bytes=len(content),
                              sha256=hashlib.sha256(content).hexdigest(), kind='data'))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gtype, constraint = GRES[a.gpu]
    lines = []
    tags = []
    for k, t in enumerate(job['tasks']):
        tag = t['cmd'].split('--arm')[1].split()[0] if '--arm' in t['cmd'] else f"task{k}"
        tags.append(tag)
        cmd = t['cmd'].replace('OUT/', '"$TASK_ROOT/output/"')
        lines.append(f'CUDA_VISIBLE_DEVICES=${{DEVS[{k}]}} "$PY" {cmd} > "$TASK_ROOT/logs/{tag}.log" 2>&1 &\n'
                     f'PIDS[{k}]=$!; NAMES[{k}]={tag}')
    nproc = len(job['tasks'])
    post = ''
    for k, cmd in enumerate(job.get('post', [])):     # after every task succeeded, sequentially on the first device
        cmd = cmd.replace('OUT/', '"$TASK_ROOT/output/"')
        post += (f'if [ $FAIL -eq 0 ]; then CUDA_VISIBLE_DEVICES=${{DEVS[0]}} "$PY" {cmd} > "$TASK_ROOT/logs/post{k}.log" 2>&1; '
                 f'rc=$?; echo "post{k} $rc" >> "$TASK_ROOT/output/TASK_STATUS.txt"; [ $rc -eq 0 ] || FAIL=1; fi\n')
    script = f'''#!/bin/bash
#SBATCH --job-name=jcps_{att}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{gtype}:{job["gpus"]}
{constraint}
#SBATCH --nodes=1
{('#SBATCH --exclude=' + job['exclude']) if job.get('exclude') else ''}
#SBATCH --cpus-per-task={job.get("cpus", 32)}
#SBATCH --mem={job.get("mem", "320G")}
#SBATCH --time={int(job["hours"]):02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -uo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet || exit 2
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
IFS=, read -ra DEVS <<< "${{CUDA_VISIBLE_DEVICES}}"
[ "${{#DEVS[@]}}" -ge {nproc} ] || {{ echo "allocated devices ${{CUDA_VISIBLE_DEVICES}} < {nproc}"; exit 3; }}
for i in $(seq 0 {nproc - 1}); do
  CUDA_VISIBLE_DEVICES=${{DEVS[$i]}} "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'dev ${{DEVS[$i]}} jax_backend={{b}} {{jax.devices()[0].device_kind}}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
done
cd {LANE}
declare -a PIDS NAMES
{chr(10).join(lines)}
FAIL=0
for i in $(seq 0 {nproc - 1}); do
  wait ${{PIDS[$i]}}; rc=$?
  echo "task ${{NAMES[$i]}} exit=$rc"; echo "${{NAMES[$i]}} $rc" >> "$TASK_ROOT/output/TASK_STATUS.txt"
  [ $rc -eq 0 ] || FAIL=1
done
{post}
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || {{ echo "CHECKSUM FAILED"; FAIL=1; }}
[ $FAIL -eq 0 ] || {{ echo "SOME TASKS FAILED"; exit 1; }}
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file() and p_.name != 'MANIFEST.sha256']
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
