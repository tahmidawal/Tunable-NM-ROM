"""Register bh1's trained model and directions: copy them from the checksum-verified archive into the
git-ignored ckpt/ directory and record size, SHA256 and origin in the COMMITTED CKPT-MANIFEST.json, which
cluster/stage.py verifies before staging any evaluation job.

    python cluster/register_ckpt.py bh1
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
NS = '/cluster/tufts/paralab/tawal01/bheld_20260921'
FILES = {'bh_model': 'output/bh_model.pkl', 'directions_bh': 'output/dirs/directions_bh.npz',
         'cpod512_seed': 'output/compress/cpod512_seed.pkl', 'cpod512_V': 'output/compress/cpod512_V.npz'}


def main():
    att = sys.argv[1]
    arc = LANE / 'runs' / att / 'archive'
    man_p = LANE / 'CKPT-MANIFEST.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    commit = (arc / 'COMMIT.txt').read_text().strip()
    for key, rel in FILES.items():
        src = arc / rel
        dst = LANE / 'ckpt' / att / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        want = next(l.split()[0] for l in (arc / 'OUTPUTS.sha256').read_text().splitlines() if l.split()[1] == rel)
        assert h == want, (key, h, want)
        man[key] = dict(local_path=str(dst.relative_to(LANE.parents[1])), staged_as=f'in/{dst.name}', bytes=dst.stat().st_size,
                        sha256=h, produced_by=att, source_commit=commit, remote_origin=f'{NS}/{att}/{rel}')
        print(key, h[:16], dst.stat().st_size)
    man_p.write_text(json.dumps(man, indent=1) + '\n')


if __name__ == '__main__':
    main()
