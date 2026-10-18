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
        fine_remote = (out / 'FINE-ON-CLUSTER').exists()
        subprocess.run(['sha256sum', '-c', 'OUTPUTS-local.sha256' if fine_remote else 'OUTPUTS.sha256', '--quiet'], cwd=out,
                       check=True)
        if fine_remote:      # every fine file must already be in refs/ with its recorded hash before the job dir goes
            fl = [l for l in (out / 'OUTPUTS.sha256').read_text().splitlines() if l.endswith('_fine.npz')]
            chk = '\n'.join(f"{h}  {NAMESPACE}/refs/{f.split('/', 2)[2]}" for h, f in (l.split(None, 1) for l in fl))
            r = subprocess.run(SSH + ['sha256sum -c --quiet -'], input=chk + '\n', text=True, capture_output=True)
            assert r.returncode == 0, f'fine files not verified in refs/: {r.stdout} {r.stderr}'
        q = subprocess.run(SSH + [f'squeue -h -n jb_{a.attempt} -o %i'], capture_output=True, text=True, check=True).stdout
        assert not q.strip(), f'job jb_{a.attempt} still queued/running: {q}'
        st = subprocess.run(SSH + [f'sacct -n -X -P --name jb_{a.attempt} -S now-7days -o State'], capture_output=True,
                            text=True, check=True).stdout.split()
        assert st and st[-1].split()[0] == 'COMPLETED', f'accounting state {st}'
        subprocess.run(SSH + [f'rm -rf {shlex.quote(remote)}'], check=True)
        print('removed', remote)
        return
 
    if a.promote:       # references: output/<step>/<d>d/... -> refs/<d>d/...; check all, then copy, verify, fail fast
        refs = f'{NAMESPACE}/refs'
        script = f"""set -euo pipefail
cd {shlex.quote(remote)}
sha256sum -c OUTPUTS.sha256 --quiet
grep -E '  output/[^/]+/(2d|3d)/' OUTPUTS.sha256 > PROMOTE.list || {{ echo 'nothing to promote'; exit 8; }}
while read -r h f; do rel=${{f#output/*/}}; if [ -e {refs}/"$rel" ]; then echo "exists: $rel"; exit 9; fi; done < PROMOTE.list
while read -r h f; do rel=${{f#output/*/}}; mkdir -p {refs}/"$(dirname "$rel")"; cp -p "$f" {refs}/"$rel"
  echo "$h  {refs}/$rel" | sha256sum -c --quiet; chmod 444 {refs}/"$rel"; done < PROMOTE.list
cd {refs} && find . -type f -name 'c*' | sort | xargs sha256sum > REFS-MANIFEST.sha256.tmp && mv REFS-MANIFEST.sha256.tmp REFS-MANIFEST.sha256
echo PROMOTED $(wc -l < {shlex.quote(remote)}/PROMOTE.list)
"""
        r = subprocess.run(SSH + ['bash -s'], input=script, text=True, capture_output=True)
        print(r.stdout, r.stderr)
        if r.returncode != 0 or 'PROMOTED' not in r.stdout:
            raise SystemExit(f'promotion failed (exit {r.returncode})')
        (HERE / 'runs' / a.attempt / 'PROMOTED').write_text(r.stdout)
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
    if a.no_fine:            # only *_fine.npz may be absent locally; every other output is checked
        keep = [l for l in (out / 'OUTPUTS.sha256').read_text().splitlines() if l.strip() and not l.endswith('_fine.npz')]
        (out / 'OUTPUTS-local.sha256').write_text('\n'.join(keep) + '\n')
        subprocess.run(['sha256sum', '-c', 'OUTPUTS-local.sha256', '--quiet'], cwd=out, check=True)
    else:
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    if a.no_fine:
        (out / 'FINE-ON-CLUSTER').write_text('*_fine.npz not pulled; they must be promoted into refs/ before cleanup\n')
    (out / 'collection.tar').unlink()
    if a.partial:
        (out / 'PARTIAL').write_text('partial collection\n')
    print(out)


if __name__ == '__main__':
    main()
