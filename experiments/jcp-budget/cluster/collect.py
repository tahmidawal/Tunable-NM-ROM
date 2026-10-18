"""Checksum-collect one completed attempt into runs/<attempt>/archive; then (separately) clean the remote dir.

    python cluster/collect.py <attempt> [--partial]
    python cluster/collect.py <attempt> --cleanup     (after the archive is verified and committed/manifested)
"""
import argparse
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-budget'
SSH = ['timeout', '600', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--cleanup', action='store_true')
    p.add_argument('--promote', action='store_true', help='copy reference trees of the job output into NAMESPACE/refs')
    p.add_argument('--no-fine', action='store_true', help='leave *_fine.npz on the cluster (A1.8); local check skips them')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = HERE / 'runs' / a.attempt / 'archive'
    if a.cleanup:
        assert (out / 'OUTPUTS.sha256').exists(), 'collect first'
        assert not (out / 'PARTIAL').exists(), 'a partial collection never authorises cleanup'
        steps = [d for d in (out / 'output').iterdir() if d.is_dir()]
        assert steps and all((d / 'COMPLETE').exists() for d in steps), 'a staged step lacks its COMPLETE marker'
        log = '\n'.join(p.read_text() for p in (out / 'logs').glob('*.out'))
        assert 'ALL-DONE' in log, 'job log lacks ALL-DONE'
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'] + (['--ignore-missing'] if (out / 'FINE-ON-CLUSTER').exists() else []), cwd=out, check=True)
        q = subprocess.run(SSH + [f'squeue -h -n jb_{a.attempt} -o %i'], capture_output=True, text=True, check=True).stdout
        assert not q.strip(), f'job jb_{a.attempt} still queued/running: {q}'
        st = subprocess.run(SSH + [f'sacct -n -X -P --name jb_{a.attempt} -S now-7days -o State'], capture_output=True,
                            text=True, check=True).stdout.split()
        assert st and st[-1].split()[0] == 'COMPLETED', f'accounting state {st}'
        subprocess.run(SSH + [f'rm -rf {shlex.quote(remote)}'], check=True)
        print('removed', remote)
        return
 
    if a.promote:                                       # references: output/<step>/<d>d/... -> refs/<d>d/... (no overwrite)
        cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && for s in output/*/; do '
               f'for d in "$s"2d "$s"3d; do [ -d "$d" ] || continue; '
               f'(cd "$s" && find "$(basename $d)" -type f -name "c*" | while read f; do '
               f'test ! -e {NAMESPACE}/refs/"$f" || {{ echo "exists: $f"; exit 9; }}; '
               f'mkdir -p {NAMESPACE}/refs/"$(dirname $f)"; cp -p "$f" {NAMESPACE}/refs/"$f"; chmod 444 {NAMESPACE}/refs/"$f"; done); done; done '
               f'&& cd {NAMESPACE}/refs && find . -type f -name "c*" | sort | xargs sha256sum > REFS-MANIFEST.sha256')
        subprocess.run(SSH + [cmd], check=True)
        print('promoted into', f'{NAMESPACE}/refs')
        return
    out.mkdir(parents=True, exist_ok=False)
    excl = " --exclude='*_fine.npz'" if a.no_fine else ''
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && tar' + excl + ' -cf collection.tar COMMIT.txt PROVENANCE.json MANIFEST.sha256 '
           'run.sbatch logs OUTPUTS.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(SSH + [pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['timeout', '3600', 'scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'] + (['--ignore-missing'] if a.no_fine else []), cwd=out,
                   check=True)
    if a.no_fine:
        (out / 'FINE-ON-CLUSTER').write_text('*_fine.npz left in the cluster refs/ directory (listed in OUTPUTS.sha256)\n')
    (out / 'collection.tar').unlink()
    if a.partial:
        (out / 'PARTIAL').write_text('partial collection\n')
    print(out)


if __name__ == '__main__':
    main()
