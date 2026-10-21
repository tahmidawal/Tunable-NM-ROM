#!/bin/bash
# imp-fronts Phase B: lay out one single-change retrain job tree (DESIGN B1, Amendment 1).
#
#   bash experiments/imp-fronts/train2d/stage_arm.sh <arm> <jobdir>        arm = base | fid | cov
#
# Writes <jobdir>/experiments/imp-fronts/train2d/ = exactly the FILES.txt entries of the jcp-robust train2d copy next to
# this script (run_seed.sh, PROVENANCE.json and the vendored code, paths mapped experiments/jcp-robust/train2d/ ->
# experiments/imp-fronts/train2d/), then overlays arms/<arm>/ARM.json's files (every source sha256 must equal its
# ARM.json entry) and rewrites PROVENANCE.json for exactly those paths (edited files: edited=true, new sha256, bytes,
# reason, original_sha256; added files: their own entry). run_seed.sh's vendored-file guard then checks the overlay.
# Arm 'base' overlays nothing, so its tree is byte-identical to jcp-robust's. Also writes <jobdir>/ARM-STAGED.json
# (arm, declared paths, sha256 of every staged file) outside the train2d tree. Refuses an existing non-empty <jobdir>.
# The job then runs:  PY=<python> bash experiments/imp-fronts/train2d/run_seed.sh 1 <outdir>   (from <jobdir>)
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <base|fid|cov> <jobdir>" >&2; exit 2; }
ARM=$1; JOB=$2
case "$ARM" in base|fid|cov) ;; *) echo "unknown arm $ARM" >&2; exit 2;; esac
PYI="${PY:-/home/tahmid/Dev/.venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -e "$JOB" ] && [ -n "$(ls -A "$JOB" 2>/dev/null)" ]; then echo "$JOB exists and is not empty: refusing" >&2; exit 3; fi
mkdir -p "$JOB"
JOB="$(cd "$JOB" && pwd)"
"$PYI" -I - "$HERE" "$ARM" "$JOB" <<'PYEOF'
import hashlib, json, shutil, sys
from pathlib import Path
here, arm, job = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PRE = 'experiments/jcp-robust/train2d/'
dst_root = job / 'experiments/imp-fronts/train2d'
entries = [l.strip() for l in (here / 'FILES.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
assert all(e.startswith(PRE) for e in entries), 'FILES.txt entry outside the train2d tree'
rels = [e[len(PRE):] for e in entries]
prov = json.loads((here / 'PROVENANCE.json').read_text())
bad = [f['path'] for f in prov['files'] if sha(here / f['path']) != f['sha256']]
assert not bad, f'lane copy differs from its PROVENANCE.json: {bad}'
for r in rels:
    (dst_root / r).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(here / r, dst_root / r)
spec = json.loads((here / 'arms' / arm / 'ARM.json').read_text())
assert spec['arm'] == arm
declared = []
if spec['files']:
    byp = {f['path']: f for f in prov['files']}
    for f in spec['files']:
        src = here / f['src']
        assert sha(src) == f['sha256'], f'{f["src"]}: sha256 differs from ARM.json'
        d = dst_root / f['dst']
        if f['kind'] == 'edited':
            assert f['dst'] in byp and d.exists(), f'edited file not in the base tree: {f["dst"]}'
            e = byp[f['dst']]
            e.update(original_sha256=e['sha256'], original_bytes=e['bytes'], sha256=f['sha256'],
                     bytes=src.stat().st_size, edited=True, edit_source=f'experiments/imp-fronts/train2d/{f["src"]}',
                     edit_reason=f['reason'], arm=arm)
        else:
            assert f['kind'] == 'added' and f['dst'] not in byp and not d.exists(), f'added file collides: {f["dst"]}'
            e = dict(path=f['dst'], sha256=f['sha256'], bytes=src.stat().st_size, edited=bool(f.get('edited', True)),
                     arm=arm, added_by='imp-fronts', reason=f['reason'],
                     source_path=f.get('source_path', f'experiments/imp-fronts/train2d/{f["src"]}'))
            for k in ('source_branch', 'source_commit'):
                if k in f:
                    e[k] = f[k]
            prov['files'].append(e)
            byp[f['dst']] = e
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        declared.append(f['dst'])
    prov['note'] = (prov['note'] + f' || imp-fronts arm {arm!r} ({spec["design"]}): {spec["single_change"]}. '
                    f'Overlaid paths: {sorted(set(declared))}; every other entry unchanged.')
    (dst_root / 'PROVENANCE.json').write_text(json.dumps(prov, indent=1) + '\n')
    declared.append('PROVENANCE.json')
staged = sorted(str(p.relative_to(dst_root)) for p in dst_root.rglob('*') if p.is_file())
(job / 'ARM-STAGED.json').write_text(json.dumps(dict(arm=arm, arm_json_sha256=sha(here / 'arms' / arm / 'ARM.json'),
    declared=sorted(set(declared)), files={p: sha(dst_root / p) for p in staged}), indent=1) + '\n')
print(f'staged arm {arm}: {len(staged)} files under {dst_root}; declared changes: {sorted(set(declared))}')
PYEOF
chmod +x "$JOB/experiments/imp-fronts/train2d/run_seed.sh"
