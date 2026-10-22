"""imp-fronts: run the unchanged select2d.py on an R384-group-only selection result (DESIGN Amendment 9). select2d.main
iterates the registered ranks (128, 256, 384) and indexes the config's group of each; the only change here is that rank
tuple, set to (384,) before main() runs. Everything else (bank binding to inputs/banks/s<label>/BANK.json, target
certification, three-state candidate search, the registered candidate order) is select2d's own code.

Incomplete selection (Amendment 9 / 4; code audit B-1): if the job's result.json is missing or not complete (self-timer
`timeout`, kill), select2d would refuse it; this wrapper instead PERSISTS the explicit outcome
checks/select2d-<label>.json = {"selected": {"384": {"rule": null, "outcome": "incomplete selection: no certified rule"}}}
so the policy config gets selected = null (uncertified start at fib4181) and the m* route is FAIL.

    JAX_PLATFORMS=cpu python select2d_r384.py <e2s output dir> <label>     -> checks/select2d-<label>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMPLETE = 'incomplete selection: no certified rule'


def incomplete(out, label):
    p = Path(out) / 'result.json'
    if not p.exists():
        return 'result.json missing'
    try:
        r = json.loads(p.read_text())
    except (ValueError, OSError) as e:                       # truncated / malformed write (job audit SEL-b-1)
        return f'result.json unreadable ({type(e).__name__})'
    if not isinstance(r, dict):
        return 'result.json malformed'
    return None if r.get('complete') is True else 'result.json not complete (self-timer or kill)'


if __name__ == '__main__':
    out, label = sys.argv[1], sys.argv[2]
    why = incomplete(out, label)
    if why:
        (HERE / 'checks').mkdir(exist_ok=True)
        rep = dict(source=str(out), bank=label, selected={'384': dict(rule=None, outcome=INCOMPLETE, reason=why,
                                                                      target_certified=False)}, cells={})
        (HERE / 'checks' / f'select2d-{label}.json').write_text(json.dumps(rep, indent=1) + '\n')
        print('384', rep['selected']['384'])
        sys.exit(0)
    import npcheck as NC
    import npbank_arch
    npbank_arch.install(NC.NB)             # arm (c): NumPy evaluator dispatches on the checkpoint arch
    import select2d
    select2d.RANKS = (384,)
    select2d.main()
