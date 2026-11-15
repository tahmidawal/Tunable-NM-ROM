"""Run the pinned November plateau solver with frozen choices and the original E0 cohorts."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()

def check_upstream(base):
    files = json.loads((HERE/'upstream.json').read_text())['files']
    for name, want in files.items():
        if digest(base/name) != want:
            raise RuntimeError(f'Upstream content mismatch: {name}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    d=cfg['dim']
    check_upstream(args.base)
    sys.path.insert(0,str(args.base))
    solver=importlib.import_module(f'plateau{d}d')
    name={2:'fib987',3:'lat4096'}[d]
    # Historical panel used an in-job performance selection for its extra timing panel.
    # Use the historical selection on every job; never select from confirmation cases.
    solver.PL.cheapest_plateau_rule=lambda *args, **kwargs: {'arm':name, 'selection':'preopening'}
    if cfg['cohort'] in ('sealed128','sealed64'):
        from published import Opened
        opened=Opened(HERE, args.config)
        solver.RS = opened.refstore(solver.RS)
        row = opened.regenerated(d, solver)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'cohort-identity.json').write_text(json.dumps(opened.descriptor_receipt,sort_keys=True)+'\n')
        if d==2:
            solver.Q.cohort=lambda cohort: row
            solver.COHORT_SHA[cfg['cohort']] = solver.QS.sha(row)
        else:
            # Set the archived in-range seed; C.table uses the original family generator.
            seed=opened.seed(d)
            original_table=solver.C.table
            solver.C.table=lambda s,n: row if (s,n)==(seed,len(cfg['cases'])) else original_table(s,n)
            solver.COHORT[cfg['cohort']] = (seed, len(cfg['cases']), row['sha256'])
    elif cfg['cohort'] not in ('dev6','dev4'):
        raise ValueError('Only the two development or two sealed in-range cohorts are admitted')
    sys.argv = [str(solver.__file__), '--config', str(args.config.resolve()), '--out', str(args.out.resolve())]
    solver.main()
if __name__=='__main__':
    main()
