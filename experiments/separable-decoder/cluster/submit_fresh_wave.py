"""Submit one already staged fresh-wave attempt, recording both queue checks."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

NAMESPACE = '/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906'


def ssh(command):
    return subprocess.check_output(['ssh', 'tufts-login', command], text=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}', args.label):
        parser.error('invalid label')
    cell = Path(__file__).resolve().parents[1]
    stage = cell/'cluster/stage'/f'fresh_{args.label}'
    record = cell/'runs/fresh_wave_campaign'/args.label
    cfg = json.loads((record/'submission.json').read_text())
    remote = f'{NAMESPACE}/{args.label}'
    if cfg['remote'] != remote or 'job_id' in cfg:
        raise RuntimeError('Unexpected namespace or already submitted attempt')
    subprocess.run(['sha256sum', '-c', 'MANIFEST.sha256', '--quiet'], cwd=stage, check=True)
    quoted = shlex.quote(remote)
    disk = ssh(f'df -h {shlex.quote(NAMESPACE)}')
    print(disk, flush=True)
    ssh(f'test ! -e {quoted} && mkdir -p {quoted}')
    subprocess.run(['scp', '-q', '-r', str(stage)+'/.', f'tufts-login:{remote}/'], check=True)
    ssh(f'cd {quoted} && sha256sum -c MANIFEST.sha256 --quiet')
    cfg['queue_before'] = ssh('squeue -u tawal01 -o "%.18i %.30j %.8T %.20R"')
    submitted = ssh(f'cd {quoted} && sbatch --parsable job.sbatch').strip()
    if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?', submitted):
        raise RuntimeError(f'Inspect ambiguous submit output before retry: {submitted!r}')
    cfg['job_id'] = submitted.split(';')[0]
    cfg['submitted_at_utc'] = datetime.now(timezone.utc).isoformat()
    # Persist the ID before any follow-up network operation can fail.
    (record/'submission.json').write_text(json.dumps(cfg, indent=2)+'\n')
    cfg['queue_after'] = ssh('squeue -u tawal01 -o "%.18i %.30j %.8T %.20R"')
    (record/'submission.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print(json.dumps({k:cfg[k] for k in ('job_id','source_commit','remote','queue_before','queue_after')}, indent=2))


if __name__ == '__main__':
    main()
