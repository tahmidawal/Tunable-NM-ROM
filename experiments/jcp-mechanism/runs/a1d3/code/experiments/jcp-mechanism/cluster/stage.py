"""jcp-mechanism: stage one cluster job into runs/<job>/ from COMMITTED files only (pattern of vendor/quad3d/cluster).

    python cluster/stage.py <job> --kind a1d3|a1d2|a2q [--gpu h200|h100|a100-80G] [--mem 240G] [--hours 8]

Each kind has a fixed file list and command list (DESIGN.md section 6). Large reference inputs are NOT staged here:
they are listed in runs/<job>/REFS.json (source path, cluster destination, sha256) and copied by submit.sh; every
config asserts their sha256 in the job. Protocol (CLAUDE.md, lane rules): gpu partition, paralab venv, f64,
JAX_DEFAULT_MATMUL_PRECISION=highest, jax_backend=gpu preflight (exit 42), SHA256 manifest checked on the node,
all output under /cluster/tufts/paralab/tawal01/jcpmech/<job>/.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NS = '/cluster/tufts/paralab/tawal01/jcpmech'
LANE = 'experiments/jcp-mechanism'
V3 = f'{LANE}/vendor/quad3d'
V2 = f'{LANE}/vendor/quad2d'
WT = ROOT.parent
Q3 = WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d'
Q2 = WT / '2026-10-01-quadrature-study/experiments/quadrature-study'
F3 = [f'{V3}/offmesh.py', f'{V3}/vendor/burgers3d-span/common.py', f'{V3}/vendor/burgers3d-retry/tables.py',
      f'{V3}/vendor/paper-b3d/vendor/b3d_common.py', f'{V3}/inputs/model_M2/bank.pkl', f'{V3}/rules/rules.npz']
F2 = [f'{LANE}/q2d/qcore.py', f'{LANE}/q2d/qstudy.py', f'{V2}/vendor/arms.py', f'{V2}/vendor/hops.py',
      f'{V2}/vendor/hfast.py', f'{V2}/vendor/bkfast.py', f'{V2}/vendor/hari_quadrature.py',
      f'{V2}/inputs/rotation_R512.npz', f'{V2}/inputs/rule_q0_m1024_qrg304_reachable.npz',
      'experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
      'experiments/separable-decoder/sep_common.py',
      'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
KINDS = {
    'a1d3': dict(files=F3 + [f'{LANE}/a1_3d.py', f'{LANE}/configs/a1d3_n65.json', f'{LANE}/configs/a1d3_n129.json'],
                 cmds=['cd experiments/jcp-mechanism',
                       '"$PY" a1_3d.py --config configs/a1d3_n65.json --out "$TASK_ROOT/code/output/n65"',
                       '"$PY" a1_3d.py --config configs/a1d3_n129.json --out "$TASK_ROOT/code/output/n129"'],
                 refs=[(Q3 / 'runs/ref1/code/output/ref_923801.npz', 'refs/ref_923801.npz'),
                       (Q3 / 'runs/val65/code/output/fields/same_grid_ref65.npz', 'refs/same_grid_ref65_n65.npz'),
                       (Q3 / 'runs/val129/code/output/fields/same_grid_ref65.npz', 'refs/same_grid_ref65_n129.npz')]),
    'a1d2': dict(files=F2 + [f'{LANE}/configs/a1d2_L256.json', f'{LANE}/configs/a1d2_L1024.json'],
                 cmds=['cd experiments/jcp-mechanism/q2d',
                       '"$PY" qstudy.py --config ../configs/a1d2_L256.json --out "$TASK_ROOT/code/output/L256"',
                       '"$PY" qstudy.py --config ../configs/a1d2_L1024.json --out "$TASK_ROOT/code/output/L1024"'],
                 refs=[(p, f'refs2d/{p.name}') for p in sorted((Q2 / 'runs/refdv/archive/output').glob('*'))]),
    'a2q': dict(files=F3 + F2 + [f'{LANE}/a2_gap.py', f'{LANE}/inputs/a2_states.npz', f'{LANE}/configs/a2q.json'],
                cmds=['cd experiments/jcp-mechanism',
                      '"$PY" a2_gap.py --config configs/a2q.json --out "$TASK_ROOT/code/output"'],
                refs=[]),
}
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None),
        'h200': ('gpu:h200:1', None)}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    p.add_argument('--kind', required=True, choices=sorted(KINDS))
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='240G')
    p.add_argument('--hours', type=int, default=8)
    a = p.parse_args()
    assert a.job.isalnum(), a.job
    k = KINDS[a.kind]
    out = ROOT / LANE / 'runs' / a.job
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NS}/{a.job}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in dict.fromkeys(k['files'] + [f'{LANE}/cluster/stage.py']):
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'not committed: {name}'
        dest = out / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'REFS.json').write_text(json.dumps([dict(source=str(s), dest=d, sha256=sha(s), bytes=s.stat().st_size)
                                               for s, d in k['refs']], indent=1) + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    cmds = '\n'.join(k['cmds'])
    script = f'''#!/bin/bash
#SBATCH --job-name=jm_{a.job}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={a.hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
if [ -s REFS.sha256 ]; then sha256sum -c REFS.sha256 --quiet; fi
export TASK_ROOT
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
{cmds}
cd "$TASK_ROOT"
find code/output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'REFS.sha256').write_text(''.join(f'{sha(s)}  {d}\n' for s, d in k['refs']))
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
