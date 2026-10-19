"""jcp-robust stage one attempt (pattern of jcp-dims / jcp-qualify cluster/stage.py, rewritten for job specs).

Every staged file is taken from the Git blob at HEAD (`git show`), so the job runs a committed tree; files present in the
checkout must equal their blob byte for byte. A job spec (jobs/<attempt>.json, committed) gives:

  {"attempt", "gpu": "h200"|"a100-80G"|"a100"|"l40s", "gpus": 1, "hours", "mem", "cpus", "mem_fraction",
   "production": bool,                      (production jobs need checks/resources-<attempt>.json, hours >= 1.5 x projection)
   "groups": [[step, ...], ...],            (one group per GPU, groups run in parallel with CUDA_VISIBLE_DEVICES=i; the
                                             steps of a group run in order)
   "extra_files": [repo-relative paths],    (beyond the lane; e.g. the incumbent checkpoint, training sources)
   "cluster_inputs": [{"src": abs cluster path, "dst": repo-relative path, "sha256"}]   (copied and verified in-job)}
  step = {"driver": "r2acc"|"e1run3d"|"d3run"|"dtime"|"e5time", "config": "configs/x.json"}
       | {"sh": "train2d/run_seed.sh", "args": ["1", "OUT/s1"], "complete": "OUT/s1/SEED-COMPLETE"}
                                            (OUT is replaced by $TASK_ROOT/output; "complete" = the step's completion marker)

    python cluster/stage.py <attempt>
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-robust'
LANE = 'experiments/jcp-robust'
INCUMBENT = ['experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
             'experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
             'experiments/separable-decoder/sep_common.py']
DRIVERS = ('r2acc', 'e1run3d', 'd3run', 'e2run3d', 'dtime', 'e5time')
GRES = {'a100-80G': ('a100', '--constraint=a100-80G'), 'a100': ('a100', None), 'h100': ('h100', None),
        'h200': ('h200', None), 'l40s': ('l40s', None)}
SKIP = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json')


def lane_files(commit):
    out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, LANE + '/'], text=True)
    return [f for f in out.split() if not any(f[len(LANE) + 1:].startswith(s) for s in SKIP)]


def step_cmd(s):
    if 'driver' in s:
        assert s['driver'] in DRIVERS, s
        stem = Path(s['config']).stem
        return (f'echo "STEP {s["driver"]} {s["config"]} $(date -u +%FT%TZ)"\n'
                f'"$PY" {s["driver"]}.py --config {s["config"]} --out "$TASK_ROOT/output/{s["driver"]}_{stem}"')
    args = ' '.join(x.replace('OUT', '$TASK_ROOT/output') for x in s.get('args', []))
    return f'echo "STEP sh {s["sh"]} {args} $(date -u +%FT%TZ)"\nbash {s["sh"]} {args}'


def main():
    att = sys.argv[1]
    assert att.isalnum(), att
    spec_p = ROOT / LANE / 'jobs' / f'{att}.json'
    spec = json.loads(spec_p.read_text())
    assert spec['attempt'] == att
    gpus = int(spec.get('gpus', 1))
    groups = spec['groups']
    assert 1 <= len(groups) <= gpus, (len(groups), gpus)
    if spec.get('production'):
        cert = json.loads((ROOT / LANE / 'checks' / f'resources-{att}.json').read_text())
        assert spec['hours'] >= 1.5 * cert['projection_hours'] - 1e-9, (spec['hours'], cert['projection_hours'])
        assert cert['gpus'] == gpus, 'certificate GPU count differs from the job spec'
        inv_p = ROOT / LANE / 'checks' / f'inventory-{att}.json'
        assert hashlib.sha256(inv_p.read_bytes()).hexdigest() == cert['inventory_sha256'], 'inventory differs from the certificate'
        for g in groups:
            for s in g:
                if 'config' in s:
                    assert cert['configs'].get(s['config']) == hashlib.sha256((ROOT / LANE / s['config']).read_bytes()).hexdigest(), s
    out = ROOT / LANE / 'runs' / att
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{att}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits', 'checks'))]
    assert not bad, bad
    files = lane_files(commit) + INCUMBENT + list(spec.get('extra_files', []))
    for g in groups:
        for s in g:
            if 'config' in s:
                assert f'{LANE}/{s["config"]}' in files, f'config not committed: {s["config"]}'
            if 'sh' in s:
                assert f'{LANE}/{s["sh"]}' in files, f'script not committed: {s["sh"]}'
                assert s.get('complete', '').startswith('OUT/'), 'a shell step needs its completion marker'
    proof = []
    for name in sorted(set(files)):
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'JOBSPEC.json').write_text(json.dumps(spec, indent=1) + '\n')
    if spec.get('production'):                      # A4.2: the certificate and its inventory travel with the job
        for nm in (f'resources-{att}.json', f'inventory-{att}.json'):
            (out / nm).write_bytes((ROOT / LANE / 'checks' / nm).read_bytes())
    (out / 'logs').mkdir()
    gname, constraint = GRES[spec['gpu']]
    h = math.ceil(float(spec['hours']) * 4) / 4
    tl = f'{int(h):02d}:{int(round((h - int(h)) * 60)):02d}:00'
    cin = '\n'.join(f'mkdir -p "$(dirname "$TASK_ROOT/{c["dst"]}")" && cp "{c["src"]}" "$TASK_ROOT/{c["dst"]}" && '
                    f'echo "{c["sha256"]}  $TASK_ROOT/{c["dst"]}" | sha256sum -c --quiet'
                    for c in spec.get('cluster_inputs', []))
    blocks = []
    for i, g in enumerate(groups):
        body = '\n'.join(step_cmd(s) for s in g)
        blocks.append(f'( export CUDA_VISIBLE_DEVICES={i}\nset -euo pipefail\n{body}\n) > "$TASK_ROOT/logs/group{i}.log" 2>&1 &\n'
                      f'PIDS+=($!)')
    script = f'''#!/bin/bash
#SBATCH --job-name=jr_{att}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{gname}:{gpus}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task={int(spec.get('cpus', 8))}
#SBATCH --mem={spec.get('mem', '200G')}
#SBATCH --time={tl}
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PY
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={spec.get('mem_fraction', '0.92')}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
{cin}
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
for i in $(seq 0 $(({gpus} - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}} gpu_index=$i',flush=True); sys.exit(0 if b=='gpu' else 42)"
done
cd {LANE}
PIDS=()
{chr(10).join(blocks)}
RC=0
for p in "${{PIDS[@]}}"; do wait "$p" || RC=1; done
cat "$TASK_ROOT"/logs/group*.log
[ "$RC" = 0 ] || {{ echo "A GROUP FAILED"; exit 1; }}
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=att, gpu=spec['gpu'], gpus=gpus, time_limit=h,
                                                          time_limit_hours=h * gpus, groups=groups, commit=commit), indent=1) + '\n')
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote, 'time limit', tl, 'gpus', gpus, 'reserved GPU-h', h * gpus)


if __name__ == '__main__':
    main()
