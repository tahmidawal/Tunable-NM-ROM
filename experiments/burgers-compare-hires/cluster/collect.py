"""Checksum-collect one completed attempt (train or panel) into runs/<attempt>/archive (git-ignored).

    python cluster/collect.py <attempt> [--partial]

train attempts: pulls out/ (checkpoints, histories, results, worker.json), the data indices and summaries, logs;
verifies every file against the job's OUTPUTS.sha256; then writes the committed record operators-<L>.json.
panel attempts: pulls everything under output/ (full fields included) and verifies it.
Remote cleanup is a separate, explicit step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/burgers-compare-hires'
NAMESPACE = '/cluster/tufts/paralab/tawal01/bcmp_20260923'
FAMILY = {'fno-large': 'fno', 'unet-refine': 'unet', 'tsol-refine': 'transolver', 'don-small': 'deeponet'}
FNO256 = dict(name='fno-large-256', family='fno', role='the published 256^2-trained fno-large checkpoint, evaluated '
              'zero-shot at this mesh (FNO is discretisation-agnostic); NOT trained at this mesh',
              local_path='../2026-09-17-b-panel/experiments/b-panel/runs/bpn301/fnockpt/best.pt',
              sha256='208d9002cd8e856760407bd21b1355d017bf52c744fcfa7ea4c249f933852a88',
              source_job='3710846 (no-audit), staged in b-panel bpn301', trained_at='256^2', epochs=None,
              stop_reason=None)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    local = ROOT / LANE / 'runs' / a.attempt / 'archive'
    local.mkdir(parents=True, exist_ok=False)
    train = a.attempt.startswith('t')
    lane_r = f'{remote}/{LANE}'
    if train:
        pre = (f'cd {shlex.quote(lane_r)} && ' +
               ("find out data -name '*.json' -o -name 'best.pt' | sort | xargs sha256sum > OUTPUTS.sha256 && " if a.partial else '') +
               'sha256sum -c OUTPUTS.sha256 --quiet && '
               "tar -cf ../../collection.tar OUTPUTS.sha256 logs out data/opdata-summary.json data/train/index.json "
               "data/validation/index.json -C ../.. logs run.sbatch COMMIT.txt PROVENANCE.json MANIFEST.sha256 && "
               'cd ../.. && sha256sum collection.tar > collection.tar.sha256')
    else:
        pre = (f'cd {shlex.quote(lane_r)} && ' +
               ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && ' if a.partial else '') +
               'sha256sum -c OUTPUTS.sha256 --quiet && '
               'tar -cf ../../collection.tar OUTPUTS.sha256 output -C ../.. logs run.sbatch COMMIT.txt PROVENANCE.json '
               'MANIFEST.sha256 && cd ../.. && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(['ssh', 'tufts-login', pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['rsync', '-a', f'tufts-login:{remote}/{name}', str(local / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=local, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=local, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=local, check=True)
    (local / 'collection.tar').unlink()
    if train:
        w = json.loads((local / 'out/worker.json').read_text())
        od = json.loads((local / 'data/opdata-summary.json').read_text())
        L = int(od['mesh'])
        ops = []
        for rec in w['arms']:
            ck = local / 'out' / rec['arm'] / 'best.pt'
            if rec['exit_code'] != 0 or not ck.exists():
                ops.append(dict(name=rec['arm'], family=FAMILY[rec['arm']], failed=True, exit_code=rec['exit_code']))
                continue
            r = json.loads((local / 'out' / rec['arm'] / 'result.json').read_text())
            assert r['best_checkpoint_sha256'] == sha(ck)
            ops.append(dict(name=rec['arm'], family=FAMILY[rec['arm']],
                            role=f"the 256^2 panel's validation-selected {FAMILY[rec['arm']]} configuration, retrained at {L}^2",
                            local_path=str(ck.relative_to(ROOT)), sha256=r['best_checkpoint_sha256'],
                            source_job=rec['job_id'], trained_at=f'{L}^2', epochs=r['epochs_completed'],
                            best_epoch=r['best_epoch'], stop_reason=r['stop_reason'],
                            wall_budget_seconds=r['wall_budget_seconds'], training_seconds=r['training_seconds'],
                            micro_batch_final=r.get('micro_batch_final'),
                            validation_mean_case_max=r['validation']['mean_case_max'],
                            validation_worst_case_max=r['validation']['worst_case_max'],
                            real_parameter_count=r['real_parameter_count']))
        good = [o for o in ops if not o.get('failed')]
        rec = dict(mesh=L, train_attempt=a.attempt, data=od, operators=good + [FNO256],
                   failed=[o for o in ops if o.get('failed')],
                   note='local_path is relative to the repository root of this worktree (checkpoints are git-ignored)')
        (ROOT / LANE / f'operators-{L}.json').write_text(json.dumps(rec, indent=1) + '\n')
        print('operators record', f'operators-{L}.json', [(o['name'], o.get('epochs'), o.get('stop_reason')) for o in ops])
    print(local)
    print('Checksums verified; remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
