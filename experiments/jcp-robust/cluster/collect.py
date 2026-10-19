"""jcp-robust (copied from jcp-dims; namespace, prefix jr_; brief section 16: files > 100 MB stay on the cluster in burgers-archive). Checksum-collect one completed attempt into runs/<attempt>/archive; then (separately) clean the remote dir.

    python cluster/collect.py <attempt> [--partial]
    python cluster/collect.py <attempt> --cleanup     (after the archive is verified and committed/manifested)
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-robust'
ARCHIVE = '/cluster/tufts/paralab/tawal01/burgers-archive/jcp-robust'
SSH = ['timeout', '600', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--cleanup', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = HERE / 'runs' / a.attempt / 'archive'
    if a.cleanup:
        assert (out / 'OUTPUTS.sha256').exists(), 'collect first'
        assert not (out / 'PARTIAL').exists(), 'a partial collection never authorises cleanup'
        spec = json.loads((out / 'JOBSPEC.json').read_text())          # completion contract per staged step (code audit 1 #13)
        need = []
        for g in spec['groups']:
            for st_ in g:
                if 'driver' in st_:
                    need.append(out / 'output' / f"{st_['driver']}_{Path(st_['config']).stem}" / 'COMPLETE')
                else:
                    need.append(out / st_['complete'].replace('OUT', 'output'))
        assert need and all(p_.exists() for p_ in need), ('a staged step lacks its completion marker', [str(p_) for p_ in need if not p_.exists()])
        log = '\n'.join(p.read_text() for p in (out / 'logs').glob('*.out'))
        assert 'ALL-DONE' in log, 'job log lacks ALL-DONE'
        subprocess.run(['sha256sum', '-c', 'OUTPUTS-LOCAL.sha256', '--quiet'], cwd=out, check=True)
        if (out / 'LARGE.txt').read_text().strip():          # empty large-file list: nothing archived (code audit 1 #14)
            chk = subprocess.run(SSH + [f'cd {ARCHIVE}/{a.attempt} && sha256sum -c {shlex.quote(remote)}/LARGE.sha256 --quiet'],
                                 capture_output=True, text=True)
            assert chk.returncode == 0, ('archive copy does not verify', chk.stderr)
        q = subprocess.run(SSH + [f'squeue -h -n jr_{a.attempt} -o %i'], capture_output=True, text=True, check=True).stdout
        assert not q.strip(), f'job jr_{a.attempt} still queued/running: {q}'
        st = subprocess.run(SSH + [f'sacct -n -X -P --name jr_{a.attempt} -S now-7days -o State'], capture_output=True,
                            text=True, check=True).stdout.split()
        assert st and st[-1].split()[0] == 'COMPLETED', f'accounting state {st}'
        subprocess.run(SSH + [f'rm -rf {shlex.quote(remote)}'], check=True)
        print('removed', remote)
        return
    out.mkdir(parents=True, exist_ok=False)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && '
           # brief 16: every output file > 100 MB is copied to the archive (verified) and left out of the pulled tar
           f'mkdir -p {ARCHIVE}/{a.attempt} && find output -type f -size +100M | sort > LARGE.txt && '
           f'(while read f; do mkdir -p "{ARCHIVE}/{a.attempt}/$(dirname "$f")" && cp "$f" "{ARCHIVE}/{a.attempt}/$f"; done < LARGE.txt) && '
           f'(cd {ARCHIVE}/{a.attempt} && (cat {shlex.quote(remote)}/LARGE.txt | xargs -r sha256sum)) > LARGE.sha256 && '
           f'(xargs -r sha256sum < LARGE.txt | cmp - LARGE.sha256) && '
           'tar -cf collection.tar --exclude-from=LARGE.txt COMMIT.txt PROVENANCE.json MANIFEST.sha256 JOBSPEC.json '
           'run.sbatch logs OUTPUTS.sha256 LARGE.txt LARGE.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(SSH + [pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['timeout', '3600', 'scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    large = set((out / 'LARGE.txt').read_text().split())
    lines = [l for l in (out / 'OUTPUTS.sha256').read_text().splitlines() if l.split(None, 1)[1].strip() not in large]
    (out / 'OUTPUTS-LOCAL.sha256').write_text('\n'.join(lines) + '\n')
    subprocess.run(['sha256sum', '-c', 'OUTPUTS-LOCAL.sha256', '--quiet'], cwd=out, check=True)
    (out / 'ARCHIVE-LOCATION.txt').write_text(f'{ARCHIVE}/{a.attempt}  (files listed in LARGE.txt / LARGE.sha256)\n')
    (out / 'collection.tar').unlink()
    if a.partial:
        (out / 'PARTIAL').write_text('partial collection\n')
    print(out)


if __name__ == '__main__':
    main()
