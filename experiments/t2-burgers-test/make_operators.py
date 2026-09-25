"""Write operators-{t256,t1024,t2048,d256,smoke64}.json: the frozen operator checkpoints of each cell (no retraining).

256^2: the thirteen checkpoints timed by ops-timing-panel opt201 (job 4181372; records in that lane's
operators.json). 1024^2 / 2048^2: the checkpoints timed by burgers-compare-hires p1024 (4204019) / p2048e (4218390)
(records in that lane's operators-<L>.json). `table2_row` marks the checkpoint printed in the paper's Table 2 for
that cell (the most accurate size per family on the development cases, as printed); the others are reported, never
used for a Table 2 number. Every file is re-hashed here and must equal its source lane's recorded SHA256.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')
OTP = WT / '2026-09-22-ops-timing-panel/experiments/ops-timing-panel/operators.json'
BCH = WT / '2026-09-23-burgers-compare-hires'
T2_256 = {'fno-large', 'unet-large', 'tsol-refine', 'don-refine'}          # Table 2, Burgers 256^2 (opt201)
T2_HI = {'fno-large', 'unet-refine', 'tsol-refine', 'don-small'}           # Table 2, Burgers 1024^2 / 2048^2


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def fam(f):
    return {'transolver': 'tsol'}.get(f, f)


def ops256():
    rec = json.loads(OTP.read_text())
    out = []
    for c in rec['checkpoints']:
        p = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude') / c['path']
        out.append(dict(name=c['name'], family=c['family'], role=f"256^2 checkpoint ({c['role']}; source job {c['source_job']})",
                        local_path=str(p), sha256=c['sha256'], source_job=c['source_job'], trained_at='256^2',
                        source_record=str(OTP), table2_row=c['name'] in T2_256, epochs=None, stop_reason=c.get('stopped_by')))
    return out


def opshi(L):
    rec = json.loads((BCH / f'experiments/burgers-compare-hires/operators-{L}.json').read_text())
    out = []
    for c in rec['operators']:
        lp = c['local_path']
        p = (BCH / lp) if not lp.startswith('..') else (BCH / lp).resolve()
        out.append(dict(c, local_path=str(p.resolve()), source_record=f'burgers-compare-hires operators-{L}.json',
                        table2_row=c['name'] in T2_HI))
    return out


def write(tag, ops):
    for o in ops:
        got = sha(o['local_path'])
        assert got == o['sha256'], (tag, o['name'], got, o['sha256'])
    (HERE / f'operators-{tag}.json').write_text(json.dumps(dict(
        cell=tag, operators=ops, failed=[], note='frozen checkpoints; sha256 re-verified by make_operators.py and '
        'again by cluster/stage.py; no operator is trained in this lane'), indent=1) + '\n')
    print(tag, [(o['name'], o['table2_row']) for o in ops])


if __name__ == '__main__':
    o256 = ops256()
    write('t256', o256)
    write('d256', o256)
    write('t1024', opshi(1024))
    write('t2048', opshi(2048))
    write('smoke64', [o for o in o256 if o['name'] in T2_256])
