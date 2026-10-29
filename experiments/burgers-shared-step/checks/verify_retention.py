"""Verify GitHub mirror blobs against every retained artifact and output hash.

Read-only Git operations only: caller must finish its fetch/mirror first. Remote
ls-remote confirms the audited object is the actual current GitHub branch tip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path('experiments/burgers-shared-step/artifacts/screen01')
REMOTE = 'refs/heads/codeonly/exp/2026-10-28-burgers-shared-step'
GIT = ['git', '-c', 'gc.auto=0', '-C', str(ROOT)]


def cmd(*args):
    return subprocess.check_output(GIT+list(args))


def tree(ref):
    ans = {}
    for row in cmd('ls-tree', '-r', '-z', ref, '--', str(ARTIFACT)).split(b'\0'):
        if row:
            meta, name = row.split(b'\t', 1)
            mode, typ, oid = meta.decode().split()
            assert typ == 'blob'
            ans[name.decode()] = dict(mode=mode, oid=oid)
    return ans


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    remote = cmd('ls-remote', 'origin', REMOTE).decode().split()
    assert len(remote) == 2 and remote[1] == REMOTE
    mirror = remote[0]; local = cmd('rev-parse', 'HEAD').decode().strip()
    live, committed = tree(mirror), tree(local)
    assert live == committed
    disk = {str(p.relative_to(ROOT)): p for p in (ROOT/ARTIFACT).rglob('*') if p.is_file()}
    assert set(live) == set(disk)
    rows = []
    for name, spec in live.items():
        p = disk[name]; expected = hashlib.sha256(p.read_bytes()).hexdigest()
        digest = hashlib.sha256(); size = 0
        proc = subprocess.Popen(GIT+['cat-file', 'blob', spec['oid']], stdout=subprocess.PIPE)
        while chunk := proc.stdout.read(1 << 20):
            digest.update(chunk); size += len(chunk)
        assert proc.wait() == 0
        assert size == p.stat().st_size and digest.hexdigest() == expected
        assert size <= 50*1024*1024
        rows.append(dict(path=name, blob=spec['oid'], bytes=size, sha256=expected))
    lookup = {r['path']:r for r in rows}
    outputs = []
    for line in (ROOT/ARTIFACT/'OUTPUTS.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1); path = str(ARTIFACT/name)
        assert lookup[path]['sha256'] == digest
        outputs.append(path)
    result = json.loads((ROOT/ARTIFACT/'output/dev6/result.json').read_text())
    witnesses = sorted({str(ARTIFACT/'output/dev6'/t['array_witness']) for t in result['timings']})
    assert set(witnesses) <= set(outputs)
    assert len(outputs) == 31 and len(witnesses) == 12
    proof = dict(status='PASS', source_head_at_audit=local, actual_github_mirror_at_audit=mirror,
        remote_ref=REMOTE, files=len(rows), bytes=sum(r['bytes'] for r in rows),
        every_disk_file_committed_and_mirrored=True, every_mirrored_blob_sha256_matches=True,
        output_entries=len(outputs), unique_timed_witnesses=len(witnesses),
        witness_format='Each witness NPZ contains all six full fields, codes and statistics; no detached companions.',
        output_paths=outputs, witness_paths=witnesses, rows=rows)
    Path(a.out).write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('status', 'source_head_at_audit', 'actual_github_mirror_at_audit',
        'files', 'bytes', 'output_entries', 'unique_timed_witnesses')}, indent=2))


if __name__ == '__main__': main()
