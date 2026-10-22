"""imp-box cost certificate (DESIGN §10, A1.8, A2.10, A5.8). Smoke-derived, conservative upper bounds per cell type:

  cell (W, A1 or A2) <= measured elapsed of smoke cell smk-A2 (W40, A2: bank + trans + every arm)
  cell (W, A3 or A4) <= measured elapsed of smoke cell smk-A4 (W40, A4)
  (W28 / W16 cells have no bank and no trans arm, so the W40 measurement bounds them too.)
  test-once job per cell  <= TEST_FRAC x the matching smoke elapsed (references + frozen re-evaluation; bounded by the
                             measured ref + time + controls stage seconds of that smoke)

  release iff  spent + committed (running/queued, at their bound) + remaining planned (at their bound) x 1.15
               + one failed-job allowance per wave (largest bound)  <=  cap (87)

  python resources.py <plan.json>    plan = {"spent_jobs": [...], "committed": ["W28_A1", ...], "remaining": [...]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 87.0
POLICY_FACTOR = 1.5           # A12: cost factor when the policy sources changed after the smoke (A10, A11)
SMOKES = {'A2': 'smk2-A2', 'A4': 'smk2-A4'}      # the A9 re-smokes (smk-A2 / smk-A4 were cancelled, A9)
HEADROOM = 1.15


def stage_hours(cell_dir):
    rows = [json.loads(l) for l in (cell_dir / 'stage_seconds.jsonl').read_text().splitlines()]
    out = {}
    for r in rows:
        out[r['stage']] = out.get(r['stage'], 0.0) + r['seconds'] / 3600.0
    return out


def bounds():
    notes = []
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    el = {j['name']: j['gpu_hours'] for j in led['jobs']}
    b = {}
    for smk, levs in ((SMOKES['A2'], ('A1', 'A2')), (SMOKES['A4'], ('A3', 'A4'))):
        factor = [1.0]
        if smk not in el:
            continue
        if not (HERE / 'runs' / 'cells' / smk / 'out' / 'cell' / 'final.json').exists():
            notes.append(dict(smoke=smk, cannot_certify='incomplete smoke (no final.json)'))
            continue
        cd = HERE / 'runs' / 'cells' / smk / 'out' / 'cell'
        job = [j for j in led['jobs'] if j['name'] == smk][0]
        ok = dict(h200=(job.get('gpu_type') or '').lower().startswith('h200'), completed=job.get('state') == 'COMPLETED',
                  scorable=json.loads((cd / 'ref.json').read_text()).get('scorable', False),
                  controls=json.loads((cd / 'controls.json').read_text()).get('all_fire', False)
                  if (cd / 'controls.json').exists() else False)
        coh = json.loads((cd / 'cohorts.json').read_text())['n']
        bk = json.loads((cd / 'bank.json').read_text()) if (cd / 'bank.json').exists() else {}
        ok['full_size'] = bool(coh.get('train') == 96 and coh.get('dev') == 8 and coh.get('val') == 32
                               and bk.get('info', {}).get('steps') == 300000)
        ok['all_stages'] = bool(all((cd / f'.done_{x}').exists() for x in ('ref', 'data', 'pod', 'bank', 'ladder', 'time',
                                                                           'controls')))
        import geom as G
        import timing as TM
        lev0 = json.loads((cd / 'cohorts.json').read_text())['lev']
        ok['N_ref'] = json.loads((cd / 'ref.json').read_text())['N_ref'] == G.LEVELS[lev0][1]
        ok['coverage_file'] = (cd / 'ladder_coverage.json').exists() and (cd / 'ladder_terms.jsonl').exists()
        cj = json.loads((cd / 'cohorts.json').read_text())
        ok['identity'] = (cj['W'], cj['lev']) == {SMOKES['A2']: ('W40', 'A2'), SMOKES['A4']: ('W40', 'A4')}[smk]
        env = json.loads((cd / 'ref.json').read_text())['env']
        ok['default_settings'] = all(k in env and env[k] is None for k in ('IGEO_NCAP', 'IMPBOX_TINY', 'IMPBOX_BANK_STEPS'))
        fz = json.loads((cd / 'final.json').read_text())
        ok['coverage_complete'] = bool(fz.get('scorable') and not any(
            v['status'] in ('search-incomplete', 'missing-construction', 'finalist-partial')
            or v.get('construction_failures', 0) for v in fz['selection'].values()))
        terms = [json.loads(l) for l in (cd / 'ladder_terms.jsonl').read_text().splitlines()]
        ok['terminations_recorded'] = bool(terms) and all(t['reason'] in ('converged', 'pruned',
                                                                           'invalid-rungs', 'cap', 'never-valid') for t in terms)
        pol = json.loads((cd / 'frozen.json').read_text()).get('policy')
        ok['policy_compatible'] = True                  # A12: a policy change after the smoke is allowed only with
        if pol != TM.policy_identity():                 # the registered cost factor below (A10/A11 add FOM rows)
            changed = sorted(k for k in pol if pol[k] != TM.policy_identity().get(k))
            factor[0] = POLICY_FACTOR
            notes.append(dict(smoke=smk, policy_changed=changed, factor=POLICY_FACTOR))
        src = json.loads((cd / 'frozen.json').read_text()).get('source')
        ok['source_compatible'] = True                  # A12: query-source changes after the smoke take the same factor
        if src != TM.source_identity():
            factor[0] = POLICY_FACTOR
            notes.append(dict(smoke=smk, source_changed=sorted(k for k in src if src[k] != TM.source_identity().get(k)),
                              factor=POLICY_FACTOR))
        if not all(ok.values()):
            notes.append(dict(smoke=smk, cannot_certify={k: v for k, v in ok.items() if not v}))
            continue
        st = stage_hours(cd)
        st = {k: v * factor[0] for k, v in st.items()}
        test = st.get('ref', 0) + st.get('time', 0) + st.get('controls', 0)
        for lev in levs:
            b[lev] = dict(cell=el[smk] * factor[0], test=test, stages=st, source=smk, factor=factor[0])
    return b, led, notes


def certify(plan):
    b, led, notes = bounds()
    spent = led['total_gpu_hours']

    def cost(item):
        """'W28_A1' (cell) | 'test:W40_A2' (test-once job) | 'data384:W40_A2' (second data size: registered
        conservative factor 2 x the matching cell bound, A7)."""
        kind, cell = item.split(':') if ':' in item else ('cell', item)
        lev = cell.split('_')[1]
        if kind == 'data384':
            return 2.0 * b[lev]['cell']
        return b[lev][kind]
    committed = sum(cost(x) for x in plan.get('committed', [])) + float(plan.get('committed_hours', 0.0))
    remaining = sum(cost(x) for x in plan.get('remaining', []))
    waves = plan.get('waves') or [plan.get('remaining', []) + plan.get('committed', [])]
    fail = sum(max([cost(x) for x in w] or [0.0]) for w in waves)          # one failed-job allowance per wave
    total = spent + committed + HEADROOM * remaining + fail
    return dict(spent=spent, committed=committed, remaining=remaining, headroom=HEADROOM, failed_job_allowance=fail,
                projected_total=total, cap=CAP, release=bool(total <= CAP), bounds=b, notes=notes)


if __name__ == '__main__':
    print(json.dumps(certify(json.loads(Path(sys.argv[1]).read_text())), indent=1))
