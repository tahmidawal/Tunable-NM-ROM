"""Collect, audit and archive one finished attempt, then print what still needs doing.

    python experiments/b-seeds/cluster/finish.py s1            # a seed attempt
    python experiments/b-seeds/cluster/finish.py final         # the sealed-cohort attempt

Runs, in order: `collect.py` (checksum collection, the seed checkpoint into `checkpoints/`),
the NumPy audit of every ladder block the attempt produced, the q-ridge EQ-certification audit
if stage F left one, and `preserve_archive.py`. Nothing is deleted remotely: that stays an
explicit separate step, as in the parent lanes.

Every audit is run with the cohort the block belongs to, so a sealed block is graded against
the declared sealed draw and a development block against abl01.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'experiments/b-seeds'
PY = sys.executable


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--skip-collect', action='store_true',
                   help='the archive is already collected locally')
    a = p.parse_args()
    arch = HERE / 'runs' / a.attempt / 'archive'
    if not a.skip_collect:
        run([PY, HERE / 'cluster/collect.py', a.attempt])
    out = arch / 'output'
    assert out.is_dir(), out
    train = out / 'train'
    audits = []
    # ---- ladder blocks: <block> -> cohort ------------------------------------
    blocks = {}
    for d in sorted(out.iterdir()):
        if not (d / 'result.json').exists():
            continue
        if d.name in ('ladder_seed', 'ladder_incumbent'):
            blocks[d.name] = 'dev'
        elif d.name.startswith('sealed_'):
            blocks[d.name] = 'sealed'
    assert blocks, f'no ladder result.json under {out}'
    for block, cohort in blocks.items():
        dest = HERE / f'checks/{a.attempt}-{block}-audit.json'
        cmd = [PY, HERE / 'audit_seeds.py', out / block / 'result.json',
               '--fields', out / block, '--cohort', cohort, '--out', dest]
        # the training gates belong to the block that evaluates the checkpoint this job trained
        if block == 'ladder_seed' and train.is_dir():
            cmd += ['--train', train]
        run(cmd)
        audits.append(dest)
    # ---- the optional EQ certification --------------------------------------
    eq = out / 'eqcert_seed'
    if (eq / 'result.json').exists():
        dest = HERE / f'checks/{a.attempt}-eqcert-audit.json'
        cmd = [PY, ROOT / 'experiments/q-ridge/audit_eqcert.py', eq / 'result.json',
               '--fields', eq, '--out', dest]
        bank = eq / 'bank_G.npz'
        if bank.exists():
            cmd += ['--bank', bank]
        try:
            run(cmd)
            audits.append(dest)
        except subprocess.CalledProcessError as exc:
            print(f'EQ-certification audit returned {exc.returncode}; the ladder blocks are '
                  f'unaffected and stage F is optional (DESIGN.md section 5)', flush=True)
    elif (out / 'EQCERT-FAILED').exists():
        print('stage F recorded EQCERT-FAILED:', (out / 'EQCERT-FAILED').read_text().strip())
    run([PY, HERE / 'cluster/preserve_archive.py', a.attempt])

    print('\n--- gates ---')
    for d in audits:
        r = json.loads(Path(d).read_text())
        failed = r.get('failed', [])
        n = len(r.get('checks', {}))
        print(f'{Path(d).name}: {n - len(failed)} of {n} passed'
              + (f'; FAILED: {", ".join(failed)}' if failed else ''))
    print('\nNext: verify the gates above, then')
    print(f'  ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/b_seeds_20260917/{a.attempt}"')
    print('  git add -A experiments/b-seeds && git commit')


if __name__ == '__main__':
    main()
