"""Turn a completed lshape training run into the solve jobs' model list.

Reads only the training `result.json`: every head arm (the two primaries on the selected
bank and the K=16 comparison heads on the other banks) with its checkpoint and correction
basis, hash-checked, copied under stable flat names for staging.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='collected training output directory (holds result.json)')
    ap.add_argument('--stage', type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((a.run / 'result.json').read_text())
    assert d['complete']
    a.stage.mkdir(parents=True, exist_ok=True)
    models, extra = [], []
    for arm in d['head_arms']:
        ck = next(x for x in d['checkpoints'] if x['id'] == arm['arm'])
        src = a.run / ck['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['sha256'], ck['id']
        bsrc = a.run / arm['basis']['path']
        assert hashlib.sha256(bsrc.read_bytes()).hexdigest() == arm['basis']['sha256'], arm['arm']
        name, bname = f"{arm['arm']}.pkl", f"{arm['arm']}-basis.npz"
        shutil.copy2(src, a.stage / name)
        shutil.copy2(bsrc, a.stage / bname)
        models.append(dict(id=arm['arm'], primary=bool(arm['primary']), checkpoint=name, basis=bname,
                           bank=arm['bank'], factor=arm['factor'], n_enrich=arm['n_enrich'], K=arm['K'],
                           R=arm['R'], R_total=arm['R_total'], sources=arm['S'],
                           checkpoint_sha256=ck['sha256'], basis_sha256=arm['basis']['sha256'],
                           training_job=d['job_id'], training_commit=d['commit']))
        extra += [str(a.stage / name), str(a.stage / bname)]
    (a.stage / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    extra.append(str(a.stage / 'models.json'))
    print(' '.join(extra))


if __name__ == '__main__':
    main()
