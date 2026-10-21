"""imp-fronts collector for a KILLED fpolicy job (Amendment 4 B8): from the job's persisted progress.json (rewritten after
every solve and every round) it writes the exhaustion records: every case still open becomes BUDGET-CAP, scored at its
latest valid prod, and the summary is computed by the same code as the driver's (fpolicy.summarise). CPU only, no JAX.

    python fcollect.py <fpolicy out dir>          -> <out>/result.json (killed=true, complete=false) unless COMPLETE exists
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy_sm as SM  # noqa: E402
import fpolicy as FP  # noqa: E402


def restore(d):
    c = SM.Case.__new__(SM.Case)
    c.cid, c.terminal, c.scored, c.rounds = d['case'], d['terminal'], d['scored'], d['rounds']
    c.uncertified_start, c.history = d['uncertified_start'], d['history']
    c.rule, c.f = d['final_state'][0], float(d['final_state'][1])
    c.latest_valid_prod = d.get('latest_valid_prod')
    return c


def main():
    out = Path(sys.argv[1])
    if (out / 'COMPLETE').exists():
        print('job completed normally: nothing to collect')
        return
    pg = json.loads((out / 'progress.json').read_text())
    store = pg['records']
    cases = [restore(d) for d in pg['cases']]
    for c in cases:
        if c.terminal is None:
            SM.budget_cap(c, store)
    res = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
    FP.summarise(res, cases, store, set(pg.get('anchors', [])), pg['rows'], True)
    res.update(killed=True, complete=False, collected_from='progress.json', elapsed_seconds=pg.get('elapsed'))
    FP.dump(res, out / 'result.json')
    print('collected', res['counts'])


if __name__ == '__main__':
    main()
