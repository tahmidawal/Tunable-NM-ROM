"""Short ABBA probe of supported SFTP windows on an immutable saved output."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from cluster_wave import CELL, CONNECTION, NAMESPACE, SSH, sha


if __name__ == '__main__':
    out = CELL/'checks/transport-probe01'
    out.mkdir(parents=True, exist_ok=False)
    folder = NAMESPACE+'/evaluation_reflective01/out/comparison'
    code = ("import json,pathlib; p=pathlib.Path("+repr(folder)+"); "
            "d=json.loads((p/'handoff.json').read_text()); "
            "r=max((r for r in d['invocations'] if r['case']==0 and r['rep']==0),key=lambda r:r['intervals']); "
            "print(p/r['field_artifact'])")
    source = subprocess.check_output(SSH+['/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -c '+shlex.quote(code)], text=True, timeout=30).strip()
    expected = subprocess.check_output(SSH+['sha256sum '+shlex.quote(source)], text=True, timeout=30).split()[0]
    windows = {'default': [], 'large': ['-X', 'nrequests=128', '-X', 'buffer=262144']}
    results = []
    for i, name in enumerate(('default', 'large', 'large', 'default')):
        target = out/f'probe-{i}.npz'
        command = ['scp', *CONNECTION, *windows[name], '-q', 'tufts-login:'+source, str(target)]
        start = time.perf_counter()
        subprocess.run(command, check=True, timeout=60)
        elapsed = time.perf_counter()-start
        actual = sha(target)
        assert actual == expected
        results.append({'window': name, 'seconds': elapsed, 'bytes': target.stat().st_size,
                        'sha256': actual, 'megabytes_per_second': target.stat().st_size/1e6/elapsed})
        target.unlink()
        print(results[-1], flush=True)
    report = {'interpretation': 'Implementation transport probe only; no PDE solve or scientific timing was modified.',
              'source': source, 'source_sha256': expected, 'ordering': 'default,large,large,default',
              'default_documented_window': {'nrequests': 64, 'buffer': 32768},
              'large_window': {'nrequests': 128, 'buffer': 262144}, 'transfers': results}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
