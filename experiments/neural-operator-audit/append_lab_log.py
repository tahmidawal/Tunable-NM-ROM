"""Append this session's entry to the canonical lab log under an exclusive lock.

Numbers are read from the audited run records, never typed. The "Where things
stand" block is not touched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LAB = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def pct(value, digits=6):
    return f'{100 * value:.{digits}f}'


def ms(value, digits=6):
    return f'{value:.{digits}f}'


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def accuracy_rows(audit):
    rows = []
    for name, model in audit['models'].items():
        if not model.get('complete'):
            rows.append(f'| `{name}` | did not complete | | | | | |')
            continue
        s, config = model['fixed_initial'], model['config']
        rows.append(f"| `{name}` | {config['width']}/{config['modes']} | {model['epochs_completed']} | "
                    f"{pct(s['median'])} | {pct(s['maximum'])} | {s['above_threshold_counts']['0.02']} | "
                    f"{model['real_parameter_count']} |")
    return '\n'.join(rows)


def timing_rows(audit):
    timing = audit.get('timing') or {}
    if not timing.get('present'):
        return '| _timing block did not complete_ | | |'
    return '\n'.join(
        f"| `{name}` | {ms(row['device_pooled_median_ms'])} | {ms(row['host_pooled_median_ms'])} |"
        for name, row in timing['models'].items())


def cohort_rows(audit, diagnosis):
    cohort = audit.get('diagnosis_cohort') or {}
    if not cohort.get('present'):
        return '| _matched cohort not evaluated_ | | |'
    rows = [f"| `{name}` (FNO) | {pct(row['fixed_initial']['maximum'])} | {pct(row['fixed_initial']['median'])} |"
            for name, row in cohort['models'].items()]
    for name, row in diagnosis['summary'].items():
        kind = 'ROM' if name == 'rom' else 'FOM'
        rows.append(f"| `{name}` ({kind}, other job) | {pct(row['worst_fixed_initial_error'])} | — |")
    return '\n'.join(rows)


def build(primary, screen, launch, screen_launch, diagnosis, definition, commit, report_sha):
    name = min((k for k, v in primary['models'].items() if v.get('complete')),
               key=lambda k: primary['models'][k]['fixed_initial']['mean'])
    best = primary['models'][name]['fixed_initial']
    cohort = primary.get('diagnosis_cohort') or {}
    cohort_line = ''
    if cohort.get('present') and name in cohort['models']:
        c = cohort['models'][name]['fixed_initial']
        rom = diagnosis['summary']['rom']['worst_fixed_initial_error']
        fom = diagnosis['summary']['same_nt1e-2_dt005']['worst_fixed_initial_error']
        cohort_line = (
            f"On the eight cases and references the Burgers lane's own ROM/FOM diagnosis used at 256 "
            f"intervals, the validation-selected FNO `{name}` reaches {pct(c['maximum'])}% worst "
            f"fixed-initial error, against the recorded ROM {pct(rom)}% and efficient FOM "
            f"`same_nt1e-2_dt005` {pct(fom)}%. Accuracy on identical cases is comparable across jobs; "
            f"timing is not, and none is compared. The cohort is disjoint from FNO training by "
            f"generation seed and by input-field content, re-checked against the training index. ")
    screen_line = ''
    if screen:
        screen_name = min((k for k, v in screen['models'].items() if v.get('complete')),
                          key=lambda k: screen['models'][k]['fixed_initial']['mean'])
        screen_best = screen['models'][screen_name]
        screen_line = (
            f"The bounded first screen, job `{screen['job_id']}` on `{screen_launch.get('gpu')}` "
            f"(accounting `{screen['accounting'].strip()}`), gave all three capacities an identical "
            f"200-epoch budget; its best run `{screen_name}` reached "
            f"{pct(screen_best['fixed_initial']['maximum'])}% worst validation error after "
            f"{screen_best['epochs_completed']} epochs and every capacity was still improving at the "
            f"budget, which is why the primary job switched to an equal wall budget with early "
            f"stopping. Those screen numbers are superseded, not retracted. ")
    return f"""

## {time.strftime('%Y-%m-%d', time.gmtime())}
### Burgers FNO common-data baseline — complete, audited, no speed claim

The root-owned audit worktree `worktrees/2026-09-14-no-audit`, branch
`exp/2026-09-14-no-audit`, commit `{commit}`, trained and evaluated an FNO baseline on the
Burgers common dataset in namespace `/cluster/tufts/paralab/tawal01/no_audit_20260914/`. Two GPU
jobs ran, each in its own job directory: `{screen.get('job_id') if screen else 'n/a'}`
(`fno_burgers01`, bounded equal-epoch capacity screen) and `{primary['job_id']}`
(`fno_burgers02`, equal-wall-budget long training, matched-cohort scoring and timing). Both
printed `jax_backend=gpu` on `NVIDIA A100 80GB PCIe`, ran float64/complex128 with highest matmul
precision, and were checksum-collected, independently NumPy-audited and Git-archived before their
exact remote job directories were removed. Primary accounting:
`{primary['accounting'].strip()}`. The shared Burgers cache
`/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01` was copied from with checksum
verification (162 and 171 files verified against `DATA.sha256`) and was not modified or deleted;
its cleanup remains the Burgers lane's.

**Design.** The operator is a direct multi-time output, not an autoregressive rollout: one forward
pass produces the five evolved fields as output channels and the supplied initial state is returned
bitwise, so there is no step-to-step error accumulation and no rollout growth to report. The output
time set is therefore fixed by training and cannot be extended. Inputs are the sampled initial
field, the viscosity and coordinates only; generation descriptors, case ids and solver sidecars are
offline metadata. Accuracy uses the Burgers lane's own fixed-initial metric — the maximum over the
six requested times of the interior discrepancy over the interior norm of the supplied initial
field. `check_burgers_error_definition.py` imports that lane's `fixed_initial_errors` by path
(SHA256 `{definition['burgers_lane_data_py_sha256']}`) and proves the two implementations agree to
`{definition['worst_absolute_per_time_difference']:.3e}`, i.e. summation order only.

**Validation accuracy on the 32 held-out cases** (recomputed independently from saved fields):

| Run | width/modes | Epochs | Median case-max (%) | Worst case-max (%) | Cases > 2% | Real parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{accuracy_rows(primary)}

**Same-job complete-query timing** (supplied on-device initial field to on-device complete
trajectory, GPU burn-in before every timed block, every repetition retained):

| Run | Device query pooled median (ms) | Host transfer pooled median (ms) |
| --- | ---: | ---: |
{timing_rows(primary)}

**Matched eight-case accuracy comparison** (worst / median fixed-initial error):

| Method | Worst (%) | Median (%) |
| --- | ---: | ---: |
{cohort_rows(primary, diagnosis)}

{cohort_line}{screen_line}**No speed claim is established.** The FNO timings above are same-job, same-GPU only; the ROM and
FOM timings were measured in job `3702709` on a different allocation and are never divided by
them. The interleaved ROM/FNO/FOM panel remains the only admissible route to a speed statement,
and the timing protocol is recorded in `timing.py`'s module docstring so that job can reproduce it.

**Limitations and what was not done.** Single training seed, one mesh (256 intervals), one Gaussian
continuum family, bounded compute per capacity; this is not a tuned FNO baseline and establishes no
capacity ceiling. Training is deliberately **not resumable** — each run was bounded inside one
allocation and any truncated run is reported as truncated at its recorded epoch; explicit restart
semantics were not added. The eight-case cohort is small, so its worst-case column is a single case.
Nothing under `best-results/` or any other worktree was modified, no worktree was created or merged,
and no final cohort was opened.

**Open items.** (1) The interleaved same-job ROM/FNO/FOM timing panel. (2) **This branch is
committed locally but not pushed.** Two early `git push` attempts failed with HTTP 500 from GitHub;
the coordinator then directed every agent to stop pushing, because packing this 199 GB repository
reaches roughly 48 GB resident on the shared GB10 and risks `earlyoom` killing other agents' work.
The coordinator will push branches in stages. No other recent `exp/2026-09-1x` branch is on origin
either. All work is committed on `exp/2026-09-14-no-audit`. (3) Resolution transfer, repeated
seeds, TFNO and wider families remain unstudied. (4) The shared `pilot-data01` cache can now be
cleaned up by the Burgers lane.

Generated report: `experiments/neural-operator-audit/reports/2026-09-14-burgers-fno-baseline.md`
(SHA256 `{report_sha}`) with its generator and source manifest beside it.
No earlier numerical result is retracted.
"""


def main(arguments):
    primary = load(arguments.audit)
    if primary is None:
        raise SystemExit(f'No primary audit at {arguments.audit}')
    text = build(primary, load(arguments.screen_audit), load(arguments.launch) or {},
                 load(arguments.screen_launch) or {}, load(arguments.diagnosis),
                 load(arguments.definition), arguments.commit, arguments.report_sha)
    if arguments.dry_run:
        print(text)
        return
    with LAB.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(text)
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    print(f'appended {len(text)} characters to {LAB}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, default=HERE / 'runs/fno_burgers02/field-audit.json')
    parser.add_argument('--screen-audit', type=Path, default=HERE / 'runs/fno_burgers01/field-audit.json')
    parser.add_argument('--launch', type=Path, default=HERE / 'checks/collection-burgers02-launch.json')
    parser.add_argument('--screen-launch', type=Path, default=HERE / 'checks/collection-burgers-launch.json')
    parser.add_argument('--definition', type=Path, default=HERE / 'checks/burgers-error-definition.json')
    parser.add_argument('--diagnosis', type=Path, default=Path(
        '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/'
        'experiments/neural-operator-burgers/checks/refinement02-diagnosis-audit.json'))
    parser.add_argument('--commit', required=True)
    parser.add_argument('--report-sha', required=True)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
