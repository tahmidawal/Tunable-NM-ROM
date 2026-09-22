"""Every published number this lane compares against, pulled from its own audit JSON.

Nothing here is typed. The report generator imports this module so that a comparison row
cannot drift from the job that produced it, which is the failure mode this project has hit
most often. Each source is named with the job that measured it and the cohort it is on,
because two of them are on cohorts that are NOT this lane's and must be labelled as such.

    python reports/sources.py          # print what resolves, and from where
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

WORKTREES = Path(__file__).resolve().parents[4]

SOURCES = {
    # The published U-Net and Transolver screens: validation-32 and diagnosis-8, the SAME
    # cohorts and metric this lane reports on, so these rows are directly comparable.
    'unet01': WORKTREES / '2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json',
    'tsol01': WORKTREES / '2026-09-17-no-second/experiments/no-second/runs/tsol01/audit.json',
    # The published FNO screen, same cohorts.
    'fno_burgers02': WORKTREES / '2026-09-14-no-audit/experiments/neural-operator-audit/runs/fno_burgers02/field-audit.json',
    # The same-allocation panel: a DIFFERENT cohort (6 development cases) and, for the
    # operator rows, a different reference (a same-job converged 256-grid solve). It is the
    # source of the NM-ROM comparison arms and of the 4.03 % discretisation figure.
    'panel': WORKTREES / '2026-09-22-ops-timing-panel/experiments/ops-timing-panel/reports/summary.json',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name):
    path = SOURCES[name]
    return json.loads(path.read_text()), dict(source=str(path), sha256=sha256(path))


def published_arms():
    """Every published Burgers operator arm: validation-32 and diagnosis-8 statistics, the
    budget it ran under and what ended it. Keyed by the arm name the reports use."""
    arms, provenance = {}, {}
    for name in ('unet01', 'tsol01', 'fno_burgers02'):
        audit, where = load(name)
        provenance[name] = dict(where, job_id=audit.get('job_id'))
        cohort = (audit.get('cohort') or {}).get('models', {})
        for arm, row in (audit.get('arms') or audit.get('models') or {}).items():
            if not row.get('complete', True):
                continue
            short = arm.split('-', 1)[1] if arm.count('-') and arm.split('-')[0] in ('unet', 'tsol') and \
                arm.startswith(('unet-', 'tsol-')) else arm
            key = arm if arm.startswith(('unet-', 'tsol-', 'fno-')) else f'{name}-{arm}'
            arms[key] = dict(
                job=name, job_id=audit.get('job_id'),
                family=row.get('family'), real_parameters=row.get('real_parameter_count'),
                epochs_completed=row.get('epochs_completed'), best_epoch=row.get('best_epoch'),
                stop_reason=row.get('stop_reason'), training_seconds=row.get('training_seconds'),
                wall_budget_seconds=row.get('wall_budget_seconds'),
                training_cases=row.get('training_cases'),
                validation=row.get('fixed_initial'),
                cohort=(cohort.get(arm) or {}).get('fixed_initial'),
                _short=short)
    return arms, provenance


def panel_rows():
    """The same-allocation panel's own rows, on its own 6 development cases. The operator
    percentages there are against a same-job converged 256-grid solve, NOT against the fine
    reference, so they are not subtractable from the discretisation figure below."""
    summary, where = load('panel')
    rows = {r['arm']: r for r in summary['rows']} if summary.get('rows') and \
        isinstance(summary['rows'][0], dict) and 'arm' in summary['rows'][0] else {}
    return dict(rows=rows, job_id=summary.get('job_id'), gpu=summary.get('gpu'),
                cases='6 development cases',
                discretisation_percent=summary.get('fom_discretisation_error_percent', {}),
                provenance=where)


if __name__ == '__main__':
    arms, provenance = published_arms()
    print(json.dumps(dict(provenance=provenance, arms={k: dict(
        family=v['family'], parameters=v['real_parameters'], epochs=v['epochs_completed'],
        stop=v['stop_reason'], cases=v['training_cases'],
        validation_worst=(v['validation'] or {}).get('maximum'),
        validation_median=(v['validation'] or {}).get('median'),
        cohort_worst=(v['cohort'] or {}).get('maximum')) for k, v in sorted(arms.items())}), indent=2))
    panel = panel_rows()
    print(json.dumps(dict(panel_job=panel['job_id'], gpu=panel['gpu'], cases=panel['cases'],
                          discretisation_percent=panel['discretisation_percent'],
                          panel_arms=sorted(panel['rows'])), indent=2))
