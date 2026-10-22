"""imp-fronts collector for a KILLED fpolicy job (Amendment 4 B8): from the job's persisted progress.json (rewritten after
every solve and every round) it writes the exhaustion records: every case still open becomes BUDGET-CAP, scored at its
latest valid prod, and the summary is computed by the same code as the driver's (fpolicy.summarise). CPU only, no JAX.

    python fcollect.py <fpolicy out dir> [<config.json>] [--claim <claim file>]   (--claim required for test-once runs)   -> <out>/result.json (killed=true, complete=false) unless COMPLETE
exists. Without progress.json the cases are rebuilt from RUN.json (or the given config): every case BUDGET-CAP, unscored.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy_sm as SM  # noqa: E402
import fpolicy as FP  # noqa: E402
import testonce as TO  # noqa: E402


def restore(d):
    c = SM.Case.__new__(SM.Case)
    c.cid, c.terminal, c.scored, c.rounds = d['case'], d['terminal'], d['scored'], d['rounds']
    c.uncertified_start, c.history = d['uncertified_start'], d['history']
    c.rule, c.f = d['final_state'][0], float(d['final_state'][1])
    c.latest_valid_prod = d.get('latest_valid_prod')
    c.fmin = float(d.get('fmin', SM.F_MIN))
    return c


def main():
    args = sys.argv[1:]
    claim = args[args.index('--claim') + 1] if '--claim' in args else None
    pos = [x for i, x in enumerate(args) if x != '--claim' and (i == 0 or args[i - 1] != '--claim')]
    out = Path(pos[0])
    if (out / 'COMPLETE').exists():
        print('job completed normally: nothing to collect')
        return
    run = json.loads((out / 'RUN.json').read_text()) if (out / 'RUN.json').exists() else {}
    if run.get('testonce_key') or (run.get('config') or {}).get('test_once'):   # test-once: only the bound dir / job
        assert claim, 'test-once recovery needs --claim <claim file>'
        TO.bound(out / 'RUN.json', claim)
    if (out / 'progress.json').exists():
        pg = json.loads((out / 'progress.json').read_text())
    else:                                  # died before the driver wrote anything: rebuild the cases from the config
        cfg = json.loads((out / 'RUN.json').read_text())['config'] if (out / 'RUN.json').exists() else \
            json.loads(Path(pos[1]).read_text())
        fmin = SM.factor_set(cfg.get('factors'))[0]
        pg = dict(cases=[SM.Case(c, cfg.get('selected'), fmin).as_dict() for c in cfg['cases']], records={}, rows=[],
                  anchors=[], elapsed=None)
    store = pg['records']
    cases = [restore(d) for d in pg['cases']]
    for c in cases:
        if c.terminal is None:
            SM.budget_cap(c, store)
    res = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
    FP.summarise(res, cases, store, set(pg.get('anchors', [])), pg['rows'], True)
    res.update(killed=True, complete=False, collected_from=('progress.json' if (out / 'progress.json').exists() else 'config'),
               elapsed_seconds=pg.get('elapsed'))
    out.mkdir(parents=True, exist_ok=True)
    FP.dump(res, out / 'result.json')
    print('collected', res['counts'])


if __name__ == '__main__':
    main()
