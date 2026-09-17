"""Write the lane's dated lab-log entry from the generated report's summary.json and the audits.

    python experiments/b-seeds/make_lab_entry.py --summary experiments/b-seeds/reports/summary.json \
        --report experiments/b-seeds/reports/2026-09-18-b-seeds.md --out experiments/b-seeds/checks/lab-entry.md \
        --retractions experiments/b-seeds/checks/retractions.md

No number is typed here: every value is looked up in summary.json rows written by
`generate_b_seeds.py`. The retractions file is prose maintained by hand (it is a record of what
went wrong, not a number), and is included verbatim.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = [0, 16, 32, 64, 128, 256]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--retractions', default=None)
    p.add_argument('--date', default='2026-09-18')
    a = p.parse_args()
    rows = json.loads(Path(a.summary).read_text())

    def get(**kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    def val(**kw):
        m = get(**kw)
        return m[0]['value'] if m else None

    verdict = {r['metric']: r['value'] for r in get(cohort='verdict')}
    seeds = sorted({r['checkpoint'] for r in rows if r.get('checkpoint', '').startswith('seed') and r.get('cohort') == 'dev'})
    jobs = sorted({(r['attempt'], r['job_id'], r['gpu']) for r in rows if r.get('job_id')}, key=lambda t: str(t[1]))
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    rep = Path(a.report)
    rsha = hashlib.sha256(rep.read_bytes()).hexdigest()

    def fmt(x, d=4):
        return '—' if x is None else (('yes' if x else 'no') if isinstance(x, bool) else f'{x:.{d}f}')

    L = []
    w = L.append
    c1 = verdict.get('C1_monotone_converged_on_at_least_2_of_3')
    c2 = verdict.get('C2_sealed_over_dev_ratio_le_1p5')
    c2n = verdict.get('C2n_normalised_ratio_le_1p5')
    w(f'## {a.date}')
    w('')
    w(f'### b-seeds — the dense Burgers correction ladder across three retrained seeds and a sealed cohort: '
      f'C1 (monotone + converged on ≥ 2 of 3 seeds) {"HOLDS" if c1 else "FAILS"} ({verdict.get("C1_count")} of {len(seeds)}); '
      f'sealed/dev ratio ≤ 1.5 at every rung {"holds" if c2 else ("fails" if c2 is False else "not run")}'
      f' (difficulty-normalised: {"holds" if c2n else ("fails" if c2n is False else "not run")})')
    w('')
    w(f'Branch `exp/2026-09-17-b-seeds` at `{commit}`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`; '
      f'namespace `/cluster/tufts/paralab/tawal01/b_seeds_20260917/`, one attempt directory per job. Jobs: '
      + ', '.join(f'`{att}` {jid} ({gpu})' for att, jid, gpu in jobs)
      + '. Every job printed `jax_backend=gpu`, ran float64 at highest matmul precision, was checksum-collected, '
        'independently NumPy-audited and Git-archived before its exact remote attempt directory was removed. '
        'Predeclared protocol and amendments: `experiments/b-seeds/DESIGN.md`.')
    w('')
    w('**T12 — development cohort, `dense_m4`, worst evolved % (seed mean ± std; incumbent re-run in the same job):**')
    w('')
    w('| q | ' + ' | '.join(seeds) + ' | incumbent | converged (seeds) |')
    w('|---|' + '---|' * len(seeds) + '---|---|')
    inc_att = next((r['attempt'] for r in rows if r.get('checkpoint') == 'incumbent' and r.get('cohort') == 'dev'), None)
    for q in Q:
        vs = [val(checkpoint=s, cohort='dev', ladder='dense_m4', q=q, metric='evolved') for s in seeds]
        inc = val(checkpoint='incumbent', cohort='dev', attempt=inc_att, ladder='dense_m4', q=q, metric='evolved')
        cv = [val(checkpoint=s, cohort='dev', ladder='dense_m4', q=q, metric='converged') for s in seeds]
        w(f'| {q} | ' + ' | '.join(fmt(v) for v in vs) + f' | {fmt(inc)} | {sum(1 for c in cv if c)} of {len(cv)} |')
    w('')
    w('**Three layers at q = 0 (bank floor / best-found / solved all-times %):** '
      + '; '.join(f'{s}: {fmt(val(checkpoint=s, cohort="dev", metric="three_layer_bank_floor_percent"))} / '
                  f'{fmt(val(checkpoint=s, cohort="dev", metric="three_layer_best_found_percent"))} / '
                  f'{fmt(val(checkpoint=s, cohort="dev", metric="three_layer_solved_all_times_percent"))}' for s in seeds)
      + f'; incumbent: {fmt(val(checkpoint="incumbent", cohort="dev", attempt=inc_att, metric="three_layer_bank_floor_percent"))} / '
        f'{fmt(val(checkpoint="incumbent", cohort="dev", attempt=inc_att, metric="three_layer_best_found_percent"))} / '
        f'{fmt(val(checkpoint="incumbent", cohort="dev", attempt=inc_att, metric="three_layer_solved_all_times_percent"))}.')
    w('')
    if verdict.get('sealed_blocks_complete'):
        w('**T13 — sealed cohort `params_draw(17092026, 6)`, opened in the final job only; worst evolved % (sealed / dev):**')
        w('')
        w('| q | ' + ' | '.join(seeds) + ' | incumbent |')
        w('|---|' + '---|' * len(seeds) + '---|')
        for q in Q:
            cells = []
            for s in seeds + ['incumbent']:
                se = val(checkpoint=s, cohort='sealed', ladder='dense_m4', q=q, metric='evolved')
                de = (val(checkpoint=s, cohort='dev', ladder='dense_m4', q=q, metric='evolved') if s != 'incumbent'
                      else val(checkpoint='incumbent', cohort='dev', attempt=inc_att, ladder='dense_m4', q=q, metric='evolved'))
                cells.append(f'{fmt(se)} / {fmt(de)}')
            w(f'| {q} | ' + ' | '.join(cells) + ' |')
        w('')
    w(f'Monotone-on-evolved counts: {json.dumps(verdict.get("monotone_evolved_counts"))}; monotone-on-all-times: '
      f'{json.dumps(verdict.get("monotone_all_times_counts"))}; converged on the development cohort: '
      f'{verdict.get("converged_dev_count")} of {len(seeds)}; knob bar on {sum(1 for s in seeds if val(checkpoint=s, cohort="dev", metric="gate_complete") is not None)} seeds evaluated. '
      f'TR (recipe reproduced): F3 = {fmt(verdict.get("F3_recipe_not_reproduced"))}.')
    w('')
    if a.retractions and Path(a.retractions).exists():
        w(Path(a.retractions).read_text().strip())
        w('')
    w(f'Source-generated report: `experiments/b-seeds/reports/{rep.name}` (SHA256 `{rsha}`), with `summary.json` and the '
      'generator beside it; every number above is read from `summary.json`. Raw archives are Git-tracked as bounded chunks '
      'under `experiments/b-seeds/artifacts/`. Not pushed; not merged.')
    Path(a.out).write_text('\n'.join(L) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
