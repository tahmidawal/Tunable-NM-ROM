"""imp-fronts reference outcome ledger (job audit codex-job-refT-1): one explicit record per scheduled case. CPU only.

    python ref_ledger.py <output dir> <steps.tsv> [<steps.tsv> ...] --configs <cfg.json> ... --out <ledger.json>
                         [--split test|dev-smoke]

Per case (the union of the step configs, which must equal the scheduled set: fresh.admissible_cases(cohort, split)):
  accepted                 reference .json present, outcome 'accepted'               (finished, eligible)
  ineligible               reference .json present, any other outcome                (finished, not eligible)
  unfinished-at-deadline   no .json; its step was skipped at the self-timer cutoff or terminated by it (rc 124 / 137)
  failed                   no .json otherwise (step rc 0 without the file, any other rc, or no step record)
Unfinished-at-deadline is a recorded budget outcome (case ineligible, counted), not a failure. Exit status 0 iff no case
is 'failed' and the configured set equals the scheduled set; 1 otherwise. The ledger is written in every case.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'phaseb' / 'frefs'))
import fresh as FR  # noqa: E402

DEADLINE_RC = {'124', '137', 'skipped-deadline'}


def read_steps(paths):
    st = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                st[parts[0]] = dict(rc=parts[1], seconds=parts[2] if len(parts) > 2 else None)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output')
    ap.add_argument('steps', nargs='+')
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    out = Path(a.output)
    steps = read_steps(a.steps)
    cases, problems = [], []
    seen = set()
    for cp in a.configs:
        cfg = json.loads(Path(cp).read_text())
        stem = Path(cp).stem
        rec = steps.get(Path(cp).name) or steps.get(stem + '.json')
        for j in cfg['cases']:
            key = (cfg['cohort'], int(j))
            if key in seen:
                problems.append(f'case {key} in more than one config')
            seen.add(key)
            js = out / f'refs_{stem}' / '2d' / cfg['cohort'] / 'ST' / f'c{int(j):03d}.json'
            row = dict(cohort=cfg['cohort'], case=int(j), config=Path(cp).name,
                       step_rc=rec['rc'] if rec else None, step_seconds=rec['seconds'] if rec else None)
            if js.exists():
                try:
                    meta = json.loads(js.read_text())
                    row.update(reference=str(js.relative_to(out)), reference_outcome=meta.get('outcome'),
                               eps_ref=(meta.get('judge') or {}).get('eps_ref'), seconds=meta.get('seconds'),
                               outcome='accepted' if meta.get('outcome') == 'accepted' else 'ineligible')
                except (ValueError, OSError) as e:
                    row.update(outcome='failed', why=f'unreadable reference json: {e}')
            elif rec and rec['rc'] in DEADLINE_RC:
                row.update(outcome='unfinished-at-deadline')
            else:
                row.update(outcome='failed', why='no reference file' + ('' if rec else ' and no step record'))
            cases.append(row)
    sched = {(c, j) for c in FR.NAMES for j in FR.admissible_cases(c, a.split)}
    if seen != sched:
        problems.append(f'configured cases differ from the scheduled set: missing {sorted(sched - seen)}, '
                        f'extra {sorted(seen - sched)}')
    kinds = ('accepted', 'ineligible', 'unfinished-at-deadline', 'failed')
    counts = {k: sum(r['outcome'] == k for r in cases) for k in kinds}
    counts['finished'] = counts['accepted'] + counts['ineligible']
    counts['scheduled'] = len(sched)
    ok = counts['failed'] == 0 and not problems
    led = dict(split=a.split, table_sha256=FR.table()['sha256'], counts=counts,
               by_cohort={c: {k: sum(r['outcome'] == k for r in cases if r['cohort'] == c) for k in kinds}
                          for c in FR.NAMES},
               cases=sorted(cases, key=lambda r: (FR.NAMES.index(r['cohort']), r['case'])), problems=problems, ok=ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(led, indent=1) + '\n')
    print('REF-LEDGER', json.dumps(counts), 'ok' if ok else f'NOT OK {problems}', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
