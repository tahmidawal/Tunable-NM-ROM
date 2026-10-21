"""Stage one imp-fronts reference job (pattern of jcp-regimes cluster/stage.py; never submits, never ssh).

    python cluster/refs/stage_refs.py <attempt> --kind test|smoke [--gpu a100-80G|h100|h200] [--render-only <file>]

Every staged file is taken from the Git blob at HEAD (`git show`) and must equal the working copy byte for byte, so the
job runs a committed tree. Output: experiments/imp-fronts/runs/<attempt>/ with run.sbatch (rendered from
refs_<kind>.sbatch.in), MANIFEST.sha256, PROVENANCE.json, COMMIT.txt, RESERVATION.json (GPU-hour ceiling = GPUs x wall).
Remote directory: /cluster/tufts/paralab/tawal01/imp-fronts/<attempt>/ (one job per directory).
--render-only writes just the rendered sbatch to <file> (dry run; no git, no staging).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE_DIR = HERE.parents[1]
ROOT = HERE.parents[3]
LANE = 'experiments/imp-fronts'
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-fronts'
KINDS = {'test': dict(template='refs_test.sbatch.in', gpus=2, wall_h=1.4, ceiling_gpu_h=2.8),
         'smoke': dict(template='refs_smoke.sbatch.in', gpus=1, wall_h=40 / 60, ceiling_gpu_h=40 / 60)}
GRES = {'a100-80G': ('a100', '#SBATCH --constraint=a100-80G'), 'h100': ('h100', ''), 'h200': ('h200', '')}
LANE_FILES = ('phaseb/frefs/', 'phaseb/fresh_cohorts.json', 'cluster/refs/')
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/jcp-regimes/vendor/quad3d/vendor/burgers3d-span/common.py',
         'experiments/jcp-regimes/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py']
FORBIDDEN = ('sealed/', 'custody/')


def render(kind, attempt, gpu):
    k = KINDS[kind]
    gt, constraint = GRES[gpu]
    s = (HERE / k['template']).read_text()
    s = (s.replace('@ATTEMPT@', attempt).replace('@REMOTE@', f'{NAMESPACE}/{attempt}').replace('@GPUTYPE@', gt)
         .replace('@CONSTRAINT@', constraint))
    assert '@' not in ''.join(l for l in s.splitlines() if l.startswith('#SBATCH')), 'unrendered placeholder'
    assert '--partition=gpu' in s and 'preempt' not in s.replace('never preempt', '')
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--kind', required=True, choices=sorted(KINDS))
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--render-only')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    script = render(a.kind, a.attempt, a.gpu)
    if a.render_only:
        Path(a.render_only).write_text(script)
        print('rendered', a.render_only)
        return
    out = LANE_DIR / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    ls = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, LANE + '/'], text=True).split()
    files = [f for f in ls if any(f[len(LANE) + 1:].startswith(x) for x in LANE_FILES)] + EXTRA
    assert not any(x in f for f in files for x in FORBIDDEN), 'forbidden path in the staged set'
    assert f'{LANE}/phaseb/fresh_cohorts.json' in files, 'fresh_cohorts.json must be committed before any reference job'
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    (out / 'run.sbatch').write_text(script)
    k = KINDS[a.kind]
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt, kind=a.kind, gpu=a.gpu, gpus=k['gpus'],
                                                          wall_hours=k['wall_h'], gpu_hour_ceiling=k['ceiling_gpu_h'],
                                                          commit=commit), indent=1) + '\n')
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file() and p_.parent.name != 'logs']
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(f'{NAMESPACE}/{a.attempt}', 'gpus', k['gpus'], 'wall_h', k['wall_h'], 'ceiling GPU-h', k['ceiling_gpu_h'])


if __name__ == '__main__':
    main()
