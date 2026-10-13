"""Stage one jcp-combined attempt into runs/<attempt>/ (pattern of C2 cluster/stage.py). Code files are taken from the Git
blob at HEAD (the checkout must match byte for byte). References: 2D from the 2D quadrature lane's pulled reference job,
3D from the 3D quadrature lane's refined reference (read-only, outside this branch), checked against their manifests.

    python experiments/jcp-combined/cluster/stage.py <attempt> [--gpu a100-80G|a100|h100|h200] [--mem 128G] [--hours 12]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/jcp-combined'
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-comb'
REFSRC = ROOT.parent / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output'
REF3 = ROOT.parent / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output'
VQ, V3 = f'{LANE}/vendor/quad2d', f'{LANE}/vendor/quad3d'
COMMON = [f'{LANE}/t2core.py', f'{LANE}/fs3.py', f'{LANE}/checks/test_fs.json', f'{LANE}/test_fs.py', f'{LANE}/test_lmm.py']
FILES2 = COMMON + [f'{LANE}/c2common.py', f'{LANE}/c2grid.py', f'{LANE}/fom2.py', f'{VQ}/qcore.py', f'{VQ}/qstudy.py',
                   f'{VQ}/vendor/arms.py', f'{VQ}/vendor/hops.py', f'{VQ}/vendor/hfast.py', f'{VQ}/vendor/bkfast.py',
                   f'{VQ}/vendor/hari_quadrature.py', f'{VQ}/inputs/rotation_R512.npz',
                   f'{VQ}/inputs/rule_q0_m1024_qrg304_reachable.npz', 'experiments/mr-burgers2d/engines.py',
                   'experiments/mr-burgers2d/iterative_paths.py', 'experiments/separable-decoder/sep_common.py',
                   'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
FILES3 = COMMON + [f'{LANE}/c3common.py', f'{LANE}/c3run.py', f'{V3}/offmesh.py', f'{V3}/qpanel.py',
                   f'{V3}/vendor/burgers3d-span/common.py', f'{V3}/vendor/burgers3d-retry/tables.py',
                   f'{V3}/vendor/paper-b3d/vendor/b3d_common.py', f'{V3}/inputs/model_M2/bank.pkl', f'{V3}/rules/rules.npz',
                   f'{LANE}/inputs/model_W1024/bank.pkl', f'{LANE}/rules3d/rules_extra.npz']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def git_show(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='128G')
    p.add_argument('--hours', type=int, default=12)
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    assert not out.exists(), f'{out} exists'
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l[3:].startswith(f'{LANE}/runs')], dirty
    cert = json.loads(git_show(commit, f'{LANE}/checks/test_fs.json'))
    assert cert['all_pass'], 'manufactured tests (G-FS2) did not pass'
    for f, h in cert['source_sha256'].items():
        assert hashlib.sha256(git_show(commit, f'{LANE}/{f}')).hexdigest() == h, f'G-FS2 certificate is for a different {f}'
    if not a.attempt.startswith('smk'):           # A2.12 resource gate (code audit 1 item 10)
        inv = json.loads(git_show(commit, f'{LANE}/checks/inventory-{a.attempt}.json'))
        assert inv['fits'] and inv['requested_hours'] <= a.hours, ('resource gate', inv)
    cfgname = f'{LANE}/configs/{a.attempt}.json'
    cfg0 = json.loads((ROOT / cfgname).read_text())
    assert cfg0['attempt'] == a.attempt
    driver = cfg0['driver']
    files = FILES3 if driver == 'c3run' else FILES2
    for name in files + [cfgname]:
        assert (ROOT / name).read_bytes() == git_show(commit, name), name
    import numpy as np
    reffiles = []
    if driver == 'c3run':
        assert cfg0['refined_ref'] == 'REF3D'
        r3 = REF3 / f"ref_{cfg0['cohort_seed']}.npz"
        dn = json.loads(r3.with_suffix('.done').read_text())
        assert hashlib.sha256(r3.read_bytes()).hexdigest() == dn['sha256'] and dn['accepted']
        reffiles = [r3, r3.with_suffix('.done')]
    else:
        assert cfg0['refs'] == 'REFS_DIR'
        man = json.loads((REFSRC / 'result.json').read_text())
        ent = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
        for coh in cfg0['cohorts']:
            sub = cfg0.get('case_subset', {}).get(coh)
            for c in (range({'dev6': 6, 'val32': 32}[coh]) if sub is None else sub):
                for tag in ('ST', 'S'):
                    f = REFSRC / f'ref_{tag}_{coh}_{c:03d}.npz'
                    z = np.load(f)['f257']
                    assert hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest() == ent[(coh, c, tag)]['f257_sha256'], f
                    reffiles.append(f)
    out.mkdir(parents=True)
    proof = []
    for name in files + [cfgname]:
        content = git_show(commit, name)
        if name == cfgname:
            cfg = json.loads(content)
            if driver == 'c3run':
                cfg['refined_ref'] = f"{remote}/refs/ref_{cfg['cohort_seed']}.npz"
            else:
                cfg['refs'] = f'{remote}/refs'
            content = (json.dumps(cfg, indent=1) + '\n').encode()
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'refs').mkdir()
    if driver != 'c3run':
        shutil.copy2(REFSRC / 'result.json', out / 'refs/result.json')
    for f in reffiles:
        shutil.copy2(f, out / 'refs' / f.name)
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = f'''#!/bin/bash
#SBATCH --job-name=cb_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
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
"$PY" {driver}.py --config configs/{a.attempt}.json --out "$TASK_ROOT/output"
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
