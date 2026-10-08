"""Stage one jcp-wide-bank job into runs/<attempt>/ (pattern of vendor/quad2d/cluster/stage.py). Every staged file is
taken from the Git blob at HEAD (`git show`), so the job runs a committed tree; files present in the checkout must equal
their blob byte for byte. The lane tree must be clean (runs/ excepted).

    python cluster/stage.py <attempt> <config> --driver w2d|w3d|train [--gpu a100-80G|a100|h100|h200]
                            [--mem 128G] [--hours 10]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcpwide'
LANE = 'experiments/jcp-wide-bank'
V2, V3 = f'{LANE}/vendor/quad2d', f'{LANE}/vendor/quad3d'
FILES = {
    'w2d': [f'{LANE}/w2d.py', f'{V2}/qcore.py', f'{V2}/vendor/arms.py', f'{V2}/vendor/hops.py', f'{V2}/vendor/hfast.py',
            f'{V2}/vendor/bkfast.py', f'{V2}/vendor/hari_quadrature.py', f'{V2}/inputs/rotation_R512.npz',
            f'{V2}/inputs/rule_q0_m1024_qrg304_reachable.npz',
            'experiments/mr-burgers2d/engines.py', 'experiments/separable-decoder/sep_common.py',
            'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'],
    'w3d': [f'{LANE}/w3d.py', f'{V3}/offmesh.py', f'{V3}/vendor/burgers3d-span/common.py',
            f'{V3}/vendor/paper-b3d/vendor/b3d_common.py',
            f'{V3}/vendor/burgers3d-retry/tables.py', f'{V3}/rules/rules.npz', f'{V3}/inputs/model_M2/bank.pkl',
            f'{V3}/inputs/model_M2/training.json', f'{LANE}/rules3d/rules_extra.npz'],
    'train': [f'{LANE}/train3d/train2w.py', f'{V3}/vendor/burgers3d-span/common.py',
              f'{V3}/vendor/paper-b3d/vendor/b3d_common.py', f'{V3}/inputs/model_M2/bank.pkl'],
}
DRIVER_CMD = {'w2d': f'cd {LANE} && "$PY" w2d.py --config {{config}} --out "$TASK_ROOT/output"',
              'w3d': f'cd {LANE} && "$PY" w3d.py --config {{config}} --out "$TASK_ROOT/output"',
              # DESIGN A4-J3: whole-process maxima by /usr/bin/time -v (train.err); a supervised sidecar samples the
              # child's RSS every 30 s together with the last training log line (phase alignment without touching the
              # trainer); stdout is unbuffered (-u) and echoed into the Slurm log at the end
              # /usr/bin/time is absent on the compute nodes (job 5012802 failed with exit 127): used only if present
              'train': ('TIMECMD=""; [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v"\n'
                        '$TIMECMD "$PY" -u ' + f'{LANE}/train3d/train2w.py --config {LANE}/{{config}} '
                        '--out "$TASK_ROOT/output" > "$TASK_ROOT/output/train.out" 2> "$TASK_ROOT/output/train.err" &\n'
                        'TPID=$!\n'
                        '( set +e; while kill -0 $TPID 2>/dev/null; do C=$(pgrep -P $TPID | head -1); '
                        'if [ -n "$C" ]; then echo "RSS_SAMPLE $(date -Is) $(ps -o rss= -p $C) kB | '
                        '$(tail -n 1 "$TASK_ROOT/output/train.out" | cut -c1-200)" >> "$TASK_ROOT/output/rss.log"; fi; '
                        'sleep 30; done ) &\n'
                        'SPID=$!\n'
                        'set +e; wait $TPID; RC=$?; set -e\n'
                        'kill $SPID 2>/dev/null || true\n'
                        'cat "$TASK_ROOT/output/train.out"; cat "$TASK_ROOT/output/train.err" >&2\n'
                        '[ $RC -eq 0 ] || exit $RC')}
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--driver', required=True, choices=sorted(FILES))
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='128G')
    p.add_argument('--hours', type=int, default=10)
    p.add_argument('--mem-fraction', default='0.90')
    p.add_argument('--extra', action='append', default=[], help='additional committed files (data inputs)')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    if a.driver == 'train':      # DESIGN A4-J3: H200 and >= 420 GB host memory
        assert a.gpu == 'h200' and a.mem.endswith('G') and int(a.mem[:-1]) >= 420, (a.gpu, a.mem)
    files = FILES[a.driver] + a.extra + [f'{LANE}/{a.config}']
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l[3:].startswith(f'{LANE}/runs')], dirty
    cfgj = json.loads((ROOT / LANE / a.config).read_text())
    assert 'test64' not in cfgj.get('cohorts', []) and cfgj.get('cohort_seed') != 923901, 'test cohorts are not used'
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
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = f'''#!/bin/bash
#SBATCH --job-name=jw_{a.attempt}
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
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={a.mem_fraction}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
{DRIVER_CMD[a.driver].format(config=a.config)}
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
