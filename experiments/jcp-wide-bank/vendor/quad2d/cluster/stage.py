"""Stage one quadrature-study attempt into its own directory (burgers2d-test's stage.py pattern). Every staged file is
taken from the Git blob at HEAD (`git show`), so the job runs a committed tree; files present in the checkout must
equal their blob byte for byte.

    python cluster/stage.py <attempt> <config> [--driver qstudy|refjob] [--gpu a100-80G|a100|h100|h200]
                            [--mem 128G] [--hours 12] [--after <jobid>]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/quad2d_20261001'
LANE = 'experiments/quadrature-study'
FILES = [f'{LANE}/qcore.py', f'{LANE}/qstudy.py', f'{LANE}/refjob.py', f'{LANE}/cluster/stage.py',
         f'{LANE}/vendor/arms.py', f'{LANE}/vendor/hops.py', f'{LANE}/vendor/hfast.py', f'{LANE}/vendor/bkfast.py',
         f'{LANE}/vendor/hari_quadrature.py', f'{LANE}/inputs/rotation_R512.npz',
         f'{LANE}/inputs/rule_q0_m1024_qrg304_reachable.npz',
         'experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--driver', default='qstudy', choices=['qstudy', 'refjob'])
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='128G')
    p.add_argument('--hours', type=int, default=12)
    p.add_argument('--mem-fraction', default='0.90')
    p.add_argument('--after', default=None, help='afterok dependency job id')
    p.add_argument('--retry-reason', default=None, help='infrastructure retry of a test attempt (recorded)')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    files = FILES + [f'{LANE}/{a.config}']
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l[3:].startswith(f'{LANE}/runs')], dirty
    cfgj = json.loads((ROOT / LANE / a.config).read_text())
    if a.driver == 'qstudy' and 'test64' in cfgj['cohorts']:
        # DESIGN 8/9 + Codex audit 2 (B6): test64 is evaluated only against a committed freeze manifest whose selection
        # hash matches the committed selection, and only once per mesh (an infrastructure retry needs --retry-reason)
        fz = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/checks/FROZEN-SELECTION.json']))
        sel = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/{fz["selection_file"]}'])
        assert hashlib.sha256(sel).hexdigest() == fz['selection_sha256'], 'selection differs from the freeze manifest'
        prior = []
        for d in (ROOT / LANE / 'runs').glob('*'):
            for f in (d / LANE / 'configs').glob('*.json') if d.name != a.attempt else []:
                c_ = json.loads(f.read_text())
                if c_.get('mesh') == cfgj['mesh'] and 'test64' in c_.get('cohorts', []):
                    prior.append(d.name)
        assert not prior or a.retry_reason, f'test64 at this mesh already staged in {prior}; pass --retry-reason'
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    if a.retry_reason:
        (out / 'RETRY-REASON.txt').write_text(a.retry_reason + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    dep = f'#SBATCH --dependency=afterok:{a.after}' if a.after else ''
    script = f'''#!/bin/bash
#SBATCH --job-name=q2d_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
{dep}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={a.hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={a.mem_fraction}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd {LANE}
"$PY" {a.driver}.py --config {a.config} --out "$TASK_ROOT/output"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
