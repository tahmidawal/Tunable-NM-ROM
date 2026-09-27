"""g2d: verify and stage the operator checkpoints of one panel job (runs inside the job).

    python stage_ops.py <manifest.json> <dest dir>   -> prints "<name> <path>" per verified checkpoint

Manifest entries: {"name", "path" (absolute, or relative to $ROOT), and either "sha256" (a checkpoint copied from a
source lane's local archive, hashed at staging time) or "result_json" (a g2d training job: the SHA256 its training
run recorded)}. A missing file, a missing training record or a hash mismatch is written to <dest>/staging.json and
the checkpoint is skipped (reported, never silently).
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    man = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ.get('ROOT', '.'))
    log = []
    for e in man['operators']:
        p = Path(e['path']) if e['path'].startswith('/') else root / e['path']
        rec = dict(e)
        if not p.exists():
            rec.update(ok=False, reason='checkpoint missing (training failed or did not run)')
        else:
            want = e.get('sha256')
            if want is None:
                rj = Path(e['result_json']) if e['result_json'].startswith('/') else root / e['result_json']
                want = json.loads(rj.read_text())['best_checkpoint_sha256'] if rj.exists() else None
            got = sha(p)
            if want is None:
                rec.update(ok=False, reason='no training record (result.json) for the checkpoint')
            elif got != want:
                rec.update(ok=False, reason=f'sha256 mismatch {got} != {want}')
            else:
                rec.update(ok=True, sha256=got)
                if e.get('copy', True):
                    shutil.copyfile(p, dest / f"{e['name']}.pt")
                    print(e['name'], dest / f"{e['name']}.pt")
                else:
                    print(e['name'], p)
        log.append(rec)
    (dest / 'staging.json').write_text(json.dumps(log, indent=1) + '\n')


if __name__ == '__main__':
    main()
