"""Stage the J2 probe (DESIGN A2.2, A3.2): the byte-identical code trees of separable-decoder jobs 2835788 (r3a bank
fit) and 2837431 (dn256b extraction + head fit), each file resolved by the sha256 in that job's MANIFEST from the Git
tree at HEAD (or one named earlier commit), plus the r3a checkpoint. Writes runs/<attempt>/ with run.sbatch.

    python experiments/imp-multimesh/cluster/stage_probe.py <attempt> [--minutes 75]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/imp-multimesh'
SEP = 'experiments/separable-decoder'
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-multimesh'
EXTRA_SOURCES = ['5ae420414b80189fe5606d3edfb8660ad7919eb7:experiments/separable-decoder/sep_solvers.py']
R3A_ENV = ('ROUND=3 K=16 LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 LAM_ORTH=1e-4 MAX_SNAPS=16384 T_EARLY=5 FULLROWS=64 '
           'N_FF=128 FF_SCALE=4.0 H_HIDDEN=256 EQ_MS=64,256 EQ_CAND_CAP=65536 STEP_TOLS=1e-9,1e-6 N_TEST=8 REPS=5 WARM=2 '
           'PAIR_REPS=3 TR_FACTOR=0.01 SEED0=0 IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 ORACLE_BUDGET=150 '
           'SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 '
           'MAX_NEWTON=20 BATCHED=1 N=256 R=512 G_HIDDEN=1024 SNAP_NORM=0 TIME_CAP=12600 POOL=0 MESH_ARM=1')
DN_EXTRACT_ENV = ('N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 EXTRA_SEED=1000 EXTRA_TRAJ=4032 '
                  'GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64')
DN_HFIT_ENV = ('STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 '
               'ENC_STEPS=12000 SEED0=0')


def sh(*a):
    return subprocess.check_output(['git', '-C', str(ROOT)] + list(a))


def head_index():
    out = sh('ls-tree', '-r', '-l', 'HEAD', 'experiments/').decode().splitlines()
    by = {}
    for ln in out:
        meta, path = ln.split('\t')
        if '/runs/' in path:
            continue
        by.setdefault(Path(path).name, []).append(path)
    return by


def resolve(manifest, by, cache):
    files = {}
    for ln in sh('show', f'HEAD:{manifest}').decode().splitlines():
        h, p = ln.split()
        if p.startswith('in/'):
            continue
        cands = [f'HEAD:{x}' for x in by.get(Path(p).name, [])] + [x for x in EXTRA_SOURCES if x.endswith('/' + Path(p).name)]
        hit = None
        for c in cands:
            if c not in cache:
                cache[c] = sh('show', c)
            if hashlib.sha256(cache[c]).hexdigest() == h:
                hit = c
                break
        assert hit, f'no source with sha256 {h} for {p}'
        files[p] = (hit, h)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('attempt')
    ap.add_argument('--minutes', type=int, default=75)
    a = ap.parse_args()
    out = ROOT / LANE / 'runs' / a.attempt
    assert not out.exists()
    remote = f'{NAMESPACE}/{a.attempt}'
    by, cache = head_index(), {}
    trees = {'code_r3a': resolve(f'{SEP}/runs/push_r3a/MANIFEST.sha256', by, cache),
             'code_dn': resolve(f'{SEP}/runs/dn256b/MANIFEST.sha256', by, cache)}
    prov = {}
    for t, files in trees.items():
        for p, (src, h) in files.items():
            dest = out / t / p[len('code/'):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(cache[src])
            prov[f'{t}/{p}'] = dict(source=src, sha256=h)
    ck = sh('show', f'HEAD:{SEP}/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.pkl')
    assert hashlib.sha256(ck).hexdigest().startswith('acc959e388b7e6d5c024')
    (out / 'in').mkdir()
    (out / 'in' / 'sep_burgers_r3_N256_K16_R512.pkl').write_bytes(ck)
    commit = sh('rev-parse', 'HEAD').decode().strip()
    (out / 'PROVENANCE.json').write_text(json.dumps(dict(commit=commit, files=prov), indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    hh, mm = divmod(a.minutes, 60)
    script = f'''#!/bin/bash
#SBATCH --job-name=imm_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time={hh:02d}:{mm:02d}:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -uo pipefail
ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XDG_CACHE_HOME="$ROOT/cache" TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROOT/out/r3a" "$ROOT/out/dn"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
# ---- (1) r3a bank fit, STEPS=25000 FULL_LAST=10000, stopped after the 'done:' line
cd "$ROOT/code_r3a"
echo "=== r3a start $(date -u +%FT%TZ)"
env {R3A_ENV} STEPS=25000 FULL_LAST=10000 OUT_PREFIX="$ROOT/out/r3a/" "$PY" -u sep_burgers_r3.py > "$ROOT/logs/r3a.log" 2>&1 &
PID=$!
for i in $(seq 1 2700); do
  if grep -q "done: recon" "$ROOT/logs/r3a.log"; then echo "r3a done line seen"; break; fi
  kill -0 $PID 2>/dev/null || {{ echo "r3a exited early"; break; }}
  sleep 2
done
kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null
echo "=== r3a stop $(date -u +%FT%TZ)"
grep -E "gen: trajectories 576|step +(1|5000|15000|20000|25000)/|done: recon" "$ROOT/logs/r3a.log"
# ---- (2) dn256b extraction, (3) head fit, producing arm only
cd "$ROOT/code_dn"
echo "=== extract start $(date -u +%FT%TZ)"
env {DN_EXTRACT_ENV} CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" OUT_PREFIX="$ROOT/out/dn/" timeout 2400 "$PY" -u sep_coeff_extract.py > "$ROOT/logs/extract.log" 2>&1; echo "extract_exit=$?"
tail -3 "$ROOT/logs/extract.log"
echo "=== hfit start $(date -u +%FT%TZ)"
env NPZ="$ROOT/out/dn/sep_coeff_N256_K16_R512.npz" CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" OUT="$ROOT/out/dn/hfit_mid.json" ARMS=mid {DN_HFIT_ENV} EMIT=mid EMIT_PATH="$ROOT/out/dn/emit_mid.pkl" timeout 1800 "$PY" -u sep_hfit_run.py > "$ROOT/logs/hfit.log" 2>&1; echo "hfit_exit=$?"
tail -3 "$ROOT/logs/hfit.log"
echo "=== end $(date -u +%FT%TZ)"
rm -f "$ROOT/out/dn/sep_coeff_N256_K16_R512.npz"
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    man = subprocess.check_output(['bash', '-c', 'find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum'],
                                  cwd=out, text=True)
    (out / 'MANIFEST.sha256').write_text(man)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt, time_limit_hours=a.minutes / 60)) + '\n')
    print('staged', out, sum(len(v) for v in trees.values()), 'files')


if __name__ == '__main__':
    main()
