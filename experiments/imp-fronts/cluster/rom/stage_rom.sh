#!/bin/bash
# imp-fronts Phase B ROM jobs: stage one job directory locally and (with --push) rsync it to the cluster. NEVER submits.
# Modelled on cluster/train/stage_job.sh.
#
#   bash experiments/imp-fronts/cluster/rom/stage_rom.sh <smr|sel-b|pol-K1|pol-b|sel-c|pol-c|sme|pol-e10s> [--push]
#
# Job names (= render.py's remote directory names): smr -> SM-r, sel-b -> SEL-b, pol-K1 -> POL-K1, pol-b -> POL-b,
# sel-c -> SEL-c, pol-c -> POL-c (arm (c), bank label c = rom/inputs/banks/sc), sme -> SM-e, pol-e10s -> POL-e10s (E10).
# Local:  experiments/imp-fronts/runs/rom/<job name>/ (refuses if it exists) = a tree mirroring the repository layout:
#           experiments/imp-fronts/rom/                (driver closure, configs, registrations, banks: K1 = s1; arm b = sb)
#           experiments/imp-fronts/phaseb/frefs/ + phaseb/fresh_cohorts.json + cluster/refs/   (SM-r dev references)
#           experiments/mr-burgers2d/{engines,iterative_paths}.py, experiments/separable-decoder/sep_common.py,
#           the jcp-regimes vendor files of cluster/refs/stage_refs.py EXTRA
#         + run.sbatch (cluster/rom/render.py) + COMMIT.txt + MANIFEST.sha256.
#         Every staged file equals its blob at HEAD (commit first). Exception: bank files too large for git
#         (ckpt.pkl / rotation_R512.npz not tracked) are copied from the working tree and verified against the sha256 in
#         their committed BANK.json. Job prerequisites are checked before anything is written:
#           sel-b / sel-c  rom/inputs/banks/s<b|c>/BANK.json and rom/configs/e2s<b|c>.json committed
#           pol-K1 / pol-b / pol-c / pol-e10s  rom/configs/pol<1|b|c|e10s>.json committed, valid for its schedule
#                                              (E10-s: schedule.check_e10s_config), its test-once registration committed
#           sme    rom/configs/smke10s.json committed
# --push: refuses if /cluster/tufts/paralab/tawal01/imp-fronts/<job name> exists on tufts-login; rsyncs; verifies
#         MANIFEST.sha256 remotely. Then submit by hand (duplicate-submit rule: squeue before and after).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <smr|sel-b|pol-K1|pol-b|sel-c|pol-c|sme|pol-e10s> [--push]" >&2; exit 2; }
KIND=$1; PUSH=${2:-}
case "$KIND" in
  smr) JOB=SM-r; RENDER="smr";;
  sel-b) JOB=SEL-b; RENDER="sel b";;
  pol-K1) JOB=POL-K1; RENDER="pol K1 pol1";;
  pol-b) JOB=POL-b; RENDER="pol b polb";;
  sel-c) JOB=SEL-c; RENDER="sel c";;
  pol-c) JOB=POL-c; RENDER="pol c polc";;
  sme) JOB=SM-e; RENDER="sme";;
  pol-e10s) JOB=POL-e10s; RENDER="pole10s";;
  *) echo "unknown job $KIND" >&2; exit 2;;
esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # experiments/imp-fronts/cluster/rom
LANE="$(cd "$HERE/../.." && pwd)"                               # experiments/imp-fronts
REPO="$(git -C "$LANE" rev-parse --show-toplevel)"
NS=/cluster/tufts/paralab/tawal01/imp-fronts
PY=/home/tahmid/Dev/.venv/bin/python
LOCAL="$LANE/runs/rom/$JOB"
[ ! -e "$LOCAL" ] || { echo "$LOCAL exists: refusing" >&2; exit 3; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain -- "$LANE/rom" "$LANE/phaseb/frefs" "$LANE/phaseb/fresh_cohorts.json" "$LANE/cluster/refs" "$HERE" \
        | grep -v ' experiments/imp-fronts/cluster/rom/stage_rom.sh$' || true)    # this stager is not staged itself
[ -z "$DIRTY" ] || { echo "uncommitted changes in the staged paths: commit first" >&2; echo "$DIRTY" >&2; exit 4; }
"$PY" -I - "$REPO" "$COMMIT" "$KIND" "$LOCAL" <<'PYEOF'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, commit, kind, local = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
L = 'experiments/imp-fronts'
show = lambda n: subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{n}'])
tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', commit, L + '/'], text=True).split()
T = set(tracked)
need = []
arm = {'sel-b': 'b', 'pol-b': 'b', 'sel-c': 'c', 'pol-c': 'c'}.get(kind)
if kind in ('sel-b', 'sel-c'):
    need = [f'{L}/rom/inputs/banks/s{arm}/BANK.json', f'{L}/rom/configs/e2s{arm}.json']
elif kind == 'sme':
    need = [f'{L}/rom/configs/smke10s.json']
elif kind.startswith('pol-'):
    lab = {'pol-K1': '1', 'pol-b': 'b', 'pol-c': 'c', 'pol-e10s': 'e10s'}[kind]
    need = [f'{L}/rom/configs/pol{lab}.json'] + ([f'{L}/rom/inputs/banks/s{arm}/BANK.json'] if arm else [])
for n in need:
    if n not in T:
        sys.exit(f'prerequisite not committed: {n}')
if kind.startswith('pol-'):
    sys.path.insert(0, str(repo / L / 'rom'))
    import schedule as SCH, testonce as TO, bankguard as BG, fcases as FC
    cfg = json.loads(show(f'{L}/rom/configs/pol{lab}.json'))
    (SCH.check_e10s_config if kind == 'pol-e10s' else SCH.check_policy_config)(cfg)
    bank = BG.verify(cfg['bank'], repo / L / 'rom', expect_label=cfg.get('bank_label'))
    k, _ = TO.key_of(bank, cfg, FC.load_table(repo / L / cfg['table'])[0]['sha256'])
    if f'{L}/rom/registrations/{k}.json' not in T:
        sys.exit(f'test-once registration {k[:12]} not committed (python rom/testonce.py register rom/configs/pol{lab}.json)')
pref = ('rom/', 'phaseb/frefs/', 'phaseb/fresh_cohorts.json', 'cluster/refs/', 'cluster/rom/')
skip = ('/__pycache__/', '.pyc')
files = [f for f in tracked if f[len(L) + 1:].startswith(pref) and not any(s in f for s in skip)]
for a_ in ('b', 'c'):                              # an arm bank only travels with that arm's jobs
    if arm != a_:
        files = [f for f in files if f'/rom/inputs/banks/s{a_}/' not in f]
files += ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
          'experiments/separable-decoder/sep_common.py',
          'experiments/jcp-regimes/vendor/quad3d/vendor/burgers3d-span/common.py',
          'experiments/jcp-regimes/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py']
assert not any(x in f for f in files for x in ('sealed/', 'custody/')), 'forbidden path'
assert f'{L}/phaseb/fresh_cohorts.json' in files
local.mkdir(parents=True)
proof = []
for n in sorted(set(files)):
    c = show(n)
    if (repo / n).exists() and (repo / n).read_bytes() != c:
        sys.exit(f'working copy differs from HEAD: {n}')
    d = local / n
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_bytes(c)
    proof.append(dict(source=n, sha256=hashlib.sha256(c).hexdigest(), bytes=len(c), origin='git'))
# bank files not tracked (too large for git): from the working tree, verified against the committed BANK.json
for bj in [f for f in files if f.endswith('/BANK.json') and '/rom/inputs/banks/' in f]:
    meta = json.loads(show(bj))
    for key, nm in (('ckpt_sha256', meta.get('ckpt', 'ckpt.pkl')), ('rotation_sha256', meta.get('rotation', 'rotation_R512.npz'))):
        rel = str(Path(bj).parent / nm)
        if rel in T:
            continue
        src = repo / rel
        b = src.read_bytes()
        if hashlib.sha256(b).hexdigest() != meta[key]:
            sys.exit(f'{rel}: working-tree file does not match BANK.json {key}')
        (local / rel).write_bytes(b)
        proof.append(dict(source=rel, sha256=meta[key], bytes=len(b), origin='working tree, verified against BANK.json'))
(local / 'PROVENANCE.json').write_text(json.dumps(dict(commit=commit, kind=kind, files=proof), indent=1) + '\n')
print(len(proof), 'files staged')
PYEOF
"$PY" "$HERE/render.py" $RENDER --out "$LOCAL/run.sbatch" >/dev/null
grep -q -- "--partition=gpu" "$LOCAL/run.sbatch" && ! grep -q preempt "$LOCAL/run.sbatch"
grep -q "imf_$JOB" "$LOCAL/run.sbatch"
echo "$COMMIT" > "$LOCAL/COMMIT.txt"
mkdir -p "$LOCAL/logs"
( cd "$LOCAL" && find . -type f ! -name MANIFEST.sha256 ! -path './logs/*' -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256 )
echo "staged $KIND as $JOB (commit $COMMIT) -> $LOCAL ($(wc -l < "$LOCAL/MANIFEST.sha256") files, $(du -sh "$LOCAL" | cut -f1))"
if [ "$PUSH" = "--push" ]; then
  timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$JOB && mkdir -p $NS/$JOB/logs"
  rsync -a "$LOCAL/" "tufts-login:$NS/$JOB/"
  timeout 300 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$JOB && sha256sum -c MANIFEST.sha256 --quiet && echo REMOTE-MANIFEST-OK"
  echo "pushed to tufts-login:$NS/$JOB (not submitted)"
fi
