"""Stage one t2-burgers-test attempt into its own directory (one directory per job, never reused).

    python cluster/stage.py <attempt> --config config-t1024.json --operators operators-t1024.json [--gpu a100-80G|h200]

Adapted from burgers-compare-hires/cluster/stage.py + panel_body.py @ 07c0e3e8. Every staged file is checked byte for
byte against the Git blob at HEAD (the job runs a committed tree); operator checkpoints (not committed) are re-hashed
against the committed operators-<cell>.json and refused on mismatch. Raw sbatch with the protocol's requirements:
gpu partition, paralab venv, preflight exit 42, f64, JAX_DEFAULT_MATMUL_PRECISION=highest, output under the namespace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/t2btest_20260925'
LANE = 'experiments/t2-burgers-test'
SRC = 'experiments/burgers-compare-hires'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h200': ('gpu:h200:1', None), 'h100': ('gpu:h100:1', None)}
LIBS = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
        'experiments/mr-burgers2d/accuracy_paths.py', 'experiments/separable-decoder/sep_common.py',
        'experiments/head-ablation/arms.py', 'experiments/head-ablation/ladder.py',
        'experiments/head-ablation/ablation.py', 'experiments/b-ladder-top/topfix.py',
        'experiments/cheap-corrections/varpro.py', 'experiments/b-panel/speed/fast.py',
        'experiments/b-panel/speed/ladders.py', 'experiments/quadratic-manifold/qman.py',
        f'{SRC}/lib/hops.py', f'{SRC}/lib/hfast.py', f'{SRC}/lib/xfast.py', f'{SRC}/sfit.py', f'{SRC}/gridarm.py',
        f'{SRC}/inputs/rotation_R512.npz', f'{SRC}/inputs/rotation_R512.json',
        f'{SRC}/inputs/pinned/train-index.json', f'{SRC}/inputs/pinned/validation-index.json',
        'experiments/b-panel/inputs/directions_qtd02.npz', CHECKPOINT]
OWN = [f'{LANE}/{f}' for f in ('tcmp.py', 't2audit.py', 'ops/optime.py', 'ops/dataset.py', 'ops/model.py',
                               'ops/families.py', 'ops/spectral_conv_f64.py', 'ops/NEURALOPERATOR-LICENSE')]
PYPATH = ['experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
          'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/b-panel/speed',
          'experiments/quadratic-manifold', f'{SRC}/lib', SRC]

HEADER = '''#!/bin/bash
#SBATCH --job-name=t2b_{attempt}
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="{pypath}"
cd {lane}
"$PY" tcmp.py --config {config} --checkpoint "$TASK_ROOT/{ckpt}" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output || echo "CMP FAILED (operators and audit still run)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
mkdir -p output/optiming
cp {operators} output/operators-runtime.json
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--operators', required=True)
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--hours', type=int, default=8)
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    cfg = json.loads((ROOT / LANE / a.config).read_text())
    assert cfg['attempt'] == a.attempt, (cfg['attempt'], a.attempt)
    files = LIBS + OWN + [f'{LANE}/{a.config}', f'{LANE}/{a.operators}']
    for rung in cfg['rungs']:
        for rs in rung['rules']:
            files += [f"experiments/b-panel/inputs/{part['file']}" for part in rs['parts'] if 'file' in part]
    ops = json.loads((ROOT / LANE / a.operators).read_text())['operators']
    extra = []
    for op in ops:
        got = sha(op['local_path'])
        assert got == op['sha256'], (op['name'], got, op['sha256'])
        extra.append((op['local_path'], f"{LANE}/opckpt/{op['name']}.pt"))
    files = list(dict.fromkeys(files))
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'not committed: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    for src, name in extra:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(src).read_bytes())
        proof.append(dict(source=str(src), staged_as=name, bytes=dest.stat().st_size, sha256=sha(dest), commit=None,
                          note=f'operator checkpoint, verified against the committed {a.operators}'))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    body = HEADER.format(attempt=a.attempt, gres=gres, constraint=('#SBATCH ' + constraint) if constraint else '',
                         mem=a.mem, hours=a.hours, remote=remote, lane=LANE, config=a.config, ckpt=CHECKPOINT,
                         operators=a.operators, pypath=':'.join('$TASK_ROOT/' + x for x in PYPATH))
    sp, fc = int(cfg['sub_points']), ','.join(str(c) for c in cfg['full_field_cases'])
    for op in ops:
        body += (f'"$PY" ops/optime.py --checkpoint opckpt/{op["name"]}.pt --index output/opcohort/index.json '
                 f'--out output/optiming --fields output/fields --name {op["name"]} --role "{op["role"]}" '
                 f'--repetitions 5 --burn-in 20 --sub-points {sp} --full-cases {fc} || echo "OPERATOR FAILED {op["name"]}"\n')
    body += '''rm -f output/opcohort/*.npz
"$PY" t2audit.py output --operators output/operators-runtime.json --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(body)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
