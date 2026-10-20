"""Checksum-collect one completed attempt into runs/<attempt>/archive (git-ignored except what is committed
explicitly). Remote cleanup is a separate explicit step (cluster/cleanup.sh)."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcpsmooth'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--job', required=True, help='jobs/<attempt>.json: its "expect" list of output JSONs must all exist with complete=true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/jcp-smooth-bank/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && tar -cf collection.tar COMMIT.txt PROVENANCE.json MANIFEST.sha256 '
           'run.sbatch logs OUTPUTS.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(['ssh', 'tufts-login', pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'collection.tar').unlink()
    import json
    expect = json.loads((ROOT / 'experiments/jcp-smooth-bank' / a.job).read_text())['expect']
    st_lines = (out / 'output' / 'TASK_STATUS.txt').read_text().split() if (out / 'output' / 'TASK_STATUS.txt').exists() else []
    bad_status = [st_lines[i] for i in range(0, len(st_lines), 2) if st_lines[i + 1] != '0']
    if a.partial or bad_status or not st_lines:
        (out / 'INCOMPLETE').write_text(f'partial={a.partial} failed_tasks={bad_status} status_present={bool(st_lines)}\n')
    bad = [e for e in expect if not (out / 'output' / e).exists() or not json.loads((out / 'output' / e).read_text()).get('complete')]
    st = (out / 'output' / 'TASK_STATUS.txt')
    print('TASK_STATUS:', st.read_text() if st.exists() else 'missing')
    if bad or bad_status or a.partial or not st_lines:
        with open(out / 'INCOMPLETE', 'a') as f_:
            f_.write('\n'.join(bad) + '\n')
        raise SystemExit(f'INCOMPLETE: missing/incomplete {bad}, failed tasks {bad_status}, partial={a.partial}')
    print(out)
    print('Checksums verified; remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
