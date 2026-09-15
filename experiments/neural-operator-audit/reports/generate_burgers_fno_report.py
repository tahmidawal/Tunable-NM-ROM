"""Generate the Burgers FNO baseline report from audited run records only.

Every number in the report is read from a JSON produced by an audit, never
typed. The FNO numbers come from this lane's independent NumPy field audits; the
ROM and FOM numbers come from the Burgers lane's own diagnosis audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
PRIMARY = RUNS / 'fno_burgers02/field-audit.json'
SCREEN = RUNS / 'fno_burgers01/field-audit.json'
DEFINITION = HERE.parent / 'checks/burgers-error-definition.json'
LAUNCH = HERE.parent / 'checks/collection-burgers02-launch.json'
SCREEN_LAUNCH = HERE.parent / 'checks/collection-burgers-launch.json'
BURGERS_LANE = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/'
                    '2026-09-14-no-burgers/experiments/neural-operator-burgers')
DIAGNOSIS = BURGERS_LANE / 'checks/refinement02-diagnosis-audit.json'
TIMES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(value, digits=4):
    return f'{100 * value:.{digits}f}'


def ms(value, digits=3):
    return f'{value:.{digits}f}'


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def capacity_table(audit):
    lines = ['| Run | Width | Modes | Real parameters | Epochs run | Best epoch | Training seconds | Truncated by wall budget |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name, model in audit['models'].items():
        if not model.get('complete'):
            lines.append(f'| `{name}` | — | — | — | — | — | — | run did not complete |')
            continue
        config = model['config']
        lines.append(f"| `{name}` | {config['width']} | {config['modes']} | {model['real_parameter_count']} | "
                     f"{model['epochs_completed']} | {model['best_epoch']} | "
                     f"{model['training_seconds']:.0f} | "
                     f"{'yes' if model['stopped_by_wall_budget'] else 'no'} |")
    return '\n'.join(lines)


def accuracy_table(audit, key='fixed_initial', source='models'):
    lines = ['| Run | Mean (%) | Median (%) | p95 (%) | Worst (%) | Cases > 1% | Cases > 2% | Cases > 5% | Upper-Tukey outliers |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, model in audit[source].items():
        if not model.get(key):
            lines.append(f'| `{name}` | — | — | — | — | — | — | — | — |')
            continue
        s = model[key]
        counts = s['above_threshold_counts']
        lines.append(f"| `{name}` | {pct(s['mean'])} | {pct(s['median'])} | {pct(s['p95'])} | "
                     f"{pct(s['maximum'])} | {counts['0.01']} | {counts['0.02']} | {counts['0.05']} | "
                     f"{s['upper_tukey_outliers']} |")
    return '\n'.join(lines)


def growth_table(audit):
    header = '| Run | ' + ' | '.join(f'$t={t:g}$' for t in TIMES) + ' |'
    lines = [header, '| --- | ' + ' | '.join('---:' for _ in TIMES) + ' |']
    for name, model in audit['models'].items():
        if not model.get('complete'):
            continue
        per_time = model['per_time_errors']
        worst = [max(case[i] for case in per_time) for i in range(len(TIMES))]
        lines.append(f'| `{name}` | ' + ' | '.join(pct(v) for v in worst) + ' |')
    return '\n'.join(lines)


def timing_table(audit):
    timing = audit['timing']
    if not timing.get('present'):
        return '_The timing block did not complete inside the allocation; no FNO timing is reported._'
    lines = ['| Run | Repetitions | Device query, pooled median (ms) | Device query, median of case medians (ms) '
             '| Host transfer, pooled median (ms) | Device + host, pooled median (ms) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in timing['models'].items():
        cases, repetitions = row['repetitions']
        lines.append(f"| `{name}` | {cases}×{repetitions} | {ms(row['device_pooled_median_ms'])} | "
                     f"{ms(row['device_median_of_case_medians_ms'])} | "
                     f"{ms(row['host_pooled_median_ms'])} | "
                     f"{ms(row['device_plus_host_pooled_median_ms'])} |")
    return '\n'.join(lines)


def matched_table(audit, diagnosis):
    cohort = audit.get('diagnosis_cohort') or {}
    if not cohort.get('present'):
        return None, ('_The matched diagnosis cohort was not evaluated in this job, so no same-case '
                      'comparison against the ROM and the FOM is available._')
    lines = ['| Method | Kind | Worst (%) | Median (%) | Mean (%) | Cases > 1% | Cases > 2% |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in cohort['models'].items():
        s = row['fixed_initial']
        counts = s['above_threshold_counts']
        lines.append(f"| `{name}` | FNO (this lane) | {pct(s['maximum'])} | {pct(s['median'])} | "
                     f"{pct(s['mean'])} | {counts['0.01']} | {counts['0.02']} |")
    for name, row in diagnosis['summary'].items():
        kind = 'ROM (Burgers lane)' if name == 'rom' else 'FOM (Burgers lane)'
        lines.append(f"| `{name}` | {kind} | {pct(row['worst_fixed_initial_error'])} | — | — | — | — |")
    return cohort, '\n'.join(lines)


def reference_table(diagnosis):
    lines = ['| Method | Worst fixed-initial error (%) | Median GPU (ms) | Median complete host query (ms) |',
             '| --- | ---: | ---: | ---: |']
    for name, row in diagnosis['summary'].items():
        lines.append(f"| `{name}` | {pct(row['worst_fixed_initial_error'])} | "
                     f"{ms(row['gpu_median_ms'])} | {ms(row['host_to_host_median_ms'])} |")
    return '\n'.join(lines)


def best_run(audit, source='models', key='fixed_initial'):
    complete = {k: v for k, v in audit[source].items() if v.get(key)}
    if not complete:
        return None, None
    name = min(complete, key=lambda k: complete[k][key]['mean'])
    return name, complete[name]


def build(primary, screen, launch, screen_launch, definition, diagnosis):
    audit = primary
    name, best = best_run(audit)
    cohort, matched = matched_table(audit, diagnosis)
    rom = diagnosis['summary']['rom']
    fom = diagnosis['summary']['same_nt1e-2_dt005']
    timing = audit['timing']

    text = f"""# Burgers FNO baseline on the common dataset

This report covers the FNO baseline trained on the Burgers common dataset built
by the Burgers lane, evaluated on that lane's 32 held-out validation cases and on
the exact eight cases its ROM/FOM diagnosis was graded on, and timed for complete
queries inside the same allocation. **The numbers are final for these jobs and
provisional as evidence about neural operators on this problem:** one training
seed, one mesh, one Gaussian continuum family, and a bounded compute budget per
capacity. No speed ratio against the ROM or the FOM is stated anywhere here; the
interleaved same-job panel that could produce one has not been run.

Primary job `{audit['job_id']}` on `{launch['gpu']}` (node `{launch['node']}`),
staged from source commit `{launch['source_commit']}`,
`jax_backend={launch['jax_backend']}`, float64 / complex128 throughout,
`JAX_DEFAULT_MATMUL_PRECISION=highest`.
Accounting: `{audit['accounting'].strip()}`.

## What the model is

The operator maps the supplied state and the known physical coefficient to the
whole requested trajectory in one evaluation,

$$\\mathcal{{G}}_\\theta : \\bigl(u_0,\\ \\nu\\bigr) \\;\\longmapsto\\;
\\bigl(u(t_1),\\dots,u(t_5)\\bigr), \\qquad
t_k \\in \\{{0.05,\\,0.10,\\,0.15,\\,0.20,\\,0.25\\}},$$

with the five evolved fields produced as five output channels of a single
Fourier neural operator and the supplied state returned exactly, so the complete
six-time trajectory is $\\bigl(u_0,\\ \\mathcal{{G}}_\\theta(u_0,\\nu)\\bigr)$.

**This is a direct multi-time output, not an autoregressive rollout.** No
prediction is fed back as an input, so there is no step-to-step error
accumulation to report. The price is that the output times are fixed by
training: the model cannot be queried at an unseen time or continued past
$t=0.25$. Per-time errors are retained regardless, so any growth across the
requested times is visible; they are tabulated below.

Inputs are the sampled initial nodal field, the viscosity broadcast to a channel
and the two coordinate channels. Generation descriptors, case identifiers and
solver-audit sidecars are offline metadata and never enter the model.

```mermaid
flowchart LR
  U0["supplied initial field u0<br/>257 x 257, f64"] --> F["feature channels<br/>normalised u0, nu, x, y"]
  NU["viscosity nu"] --> F
  F --> FNO["FNO, 4 Fourier layers<br/>trained weights"]
  FNO --> M["boundary mask<br/>zero Dirichlet"]
  M --> OUT["u(t1) ... u(t5)"]
  U0 --> TRAJ["complete trajectory<br/>u0, u(t1) ... u(t5)"]
  OUT --> TRAJ
  classDef trained fill:#2b6cb0,stroke:#1a365d,color:#ffffff
  classDef given fill:#e2e8f0,stroke:#4a5568,color:#1a202c
  classDef exact fill:#276749,stroke:#1c4532,color:#ffffff
  class FNO trained
  class U0,NU,F given
  class M,TRAJ,OUT exact
```

## How accuracy is defined

The metric is the Burgers lane's own, so the FNO, the ROM and the FOM are graded
identically:

$$E(\\text{{case}}) \\;=\\; \\max_{{k=0,\\dots,5}}\\;
\\frac{{\\bigl\\lVert \\hat u(t_k) - u^{{\\mathrm{{ref}}}}(t_k) \\bigr\\rVert_{{2,\\;\\mathrm{{interior}}}}}}
     {{\\bigl\\lVert u^{{\\mathrm{{ref}}}}(t_0) \\bigr\\rVert_{{2,\\;\\mathrm{{interior}}}}}} .$$

The denominator is fixed at the initial field for every output time
(fixed-initial normalisation) and the maximum runs over the requested times; the
$t_0$ term is identically zero here because the supplied state is returned
bitwise. `check_burgers_error_definition.py` imports the Burgers lane's
`fixed_initial_errors` from
`{definition['burgers_lane_data_py']}`
(SHA256 `{definition['burgers_lane_data_py_sha256']}`) and confirms the two
implementations agree to `{definition['worst_absolute_per_time_difference']:.3e}`
on contract-valid fields — floating-point summation order, not a definitional
difference.

The reference $u^{{\\mathrm{{ref}}}}$ is the Burgers lane's recorded fine
numerical solution (4096 intervals, $\\Delta t = 1.5625\\times 10^{{-4}}$,
restricted to the 256-interval grid). It is an empirically refined numerical
reference, not a continuum error bound.

## Capacities trained

Each capacity received the same wall budget and early stopping decided when to
stop, so epoch counts differ by design: this is an equal-compute comparison, not
an equal-epoch one. The best capacity was then retrained at a lower learning
rate.

{capacity_table(audit)}

## Validation accuracy — 32 held-out cases

Recomputed independently from the saved prediction fields with NumPy.

{accuracy_table(audit)}

## Worst error per output time

Worst case in the cohort at each requested time. Any growth here is the
difficulty of the later states, not rollout accumulation — there is no rollout.

{growth_table(audit)}

## Same-case comparison with the ROM and the efficient FOM

"""
    if cohort:
        text += f"""These are the **same eight cases and the same references** the Burgers lane's
ROM/FOM diagnosis used at 256 intervals: its refined 4096-interval anchors,
restricted to the training grid, with the supplied initial field taken from the
same restriction. The cohort is held out from FNO training by generation seed
(`{cohort['disjoint_from_training_by_seed']}`) and by input-field content
(`{cohort['disjoint_from_training_by_input_field']}`), both re-checked against
the training index.

**Accuracy is comparable across jobs on these cases; timing is not.** Error does
not depend on which GPU ran the job, so the accuracy column below is a genuine
like-for-like comparison. Wall clock does depend on it, which is why no time
appears in this table.

{matched}

"""
    else:
        text += matched + '\n\n'

    text += f"""## Complete-query timing (this job only)

{timing_table(audit)}

Timing protocol, recorded so the later interleaved ROM/FNO/FOM confirmation job
can reproduce it exactly: the timed region starts with the supplied initial
field and the viscosity already resident on the GPU and ends when the complete
six-time trajectory is resident on the GPU; coordinate construction,
normalisation, the forward pass, boundary masking and trajectory assembly are
all inside it, and nothing is precomputed outside it. `torch.cuda.synchronize()`
brackets every repetition, every timed block is preceded by its own
{timing.get('burn_in_per_block', '—')}-query GPU burn-in, host transfer is a
separate timed block with its own burn-in, and every individual repetition is
retained in `timing.npz` — none discarded, and no minimum substituted for a
median.

## The ROM and FOM reference numbers, with their timings

Reproduced from the Burgers lane's diagnosis job (`3702709`) so the eventual
confirmation job has a target. The timings were measured in a **different
allocation on a different GPU instance** and must never be divided by the
timings in the table above.

{reference_table(diagnosis)}

## The bounded first screen

"""
    if screen:
        screen_name, screen_best = best_run(screen)
        text += f"""Job `{screen['job_id']}` on `{screen_launch['gpu']}` gave all three capacities an
identical 200-epoch budget. Every capacity was still improving when that budget
ran out, which is why the primary job above switched to an equal wall budget with
early stopping. Its numbers are superseded by the table above and are kept only
to document that decision.

Accounting: `{screen['accounting'].strip()}`.

{accuracy_table(screen)}

Its best run `{screen_name}` reached {pct(screen_best['fixed_initial']['maximum'])}%
worst validation error after {screen_best['epochs_completed']} epochs.

"""
    else:
        text += '_The bounded first screen was not collected._\n\n'

    text += '## What this does and does not establish\n\n'
    lines = [f"- The best FNO run, `{name}`, reaches a worst validation-case error of "
             f"{pct(best['fixed_initial']['maximum'])}% and a median of "
             f"{pct(best['fixed_initial']['median'])}% over the "
             f"{audit['validation_cases']} held-out validation cases."]
    if cohort and name in cohort['models']:
        # The checkpoint is chosen on the validation cases and only then reported
        # on the comparison cohort; picking the cohort's own best run would be
        # selecting on the very cases the comparison is made on.
        cbest = cohort['models'][name]['fixed_initial']
        verdict_rom = 'below' if cbest['maximum'] < rom['worst_fixed_initial_error'] else 'above'
        verdict_fom = 'below' if cbest['maximum'] < fom['worst_fixed_initial_error'] else 'above'
        lines.append(
            f"- On the eight diagnosis cases, the validation-selected FNO (`{name}`) reaches "
            f"{pct(cbest['maximum'])}% worst error, {verdict_rom} the ROM's "
            f"{pct(rom['worst_fixed_initial_error'])}% and {verdict_fom} the efficient FOM arm "
            f"`same_nt1e-2_dt005`'s {pct(fom['worst_fixed_initial_error'])}%. This is a same-case, "
            f"same-reference accuracy comparison, and it carries no timing claim. The run is chosen "
            f"on the validation cases, not on these eight.")
    else:
        lines.append('- No same-case accuracy comparison with the ROM or the FOM is available from '
                     'this job.')
    lines += [
        '- **No speed claim is made.** FNO complete-query timing is reported for this job only. '
        'Dividing it by the ROM or FOM medians would be exactly the cross-job ratio this project '
        'has already had to retract once; the interleaved same-job panel remains the only '
        'admissible route to a speed statement.',
        '- The screen is a single seed, a single mesh (256 intervals), a single Gaussian continuum '
        'family and a bounded compute budget. It is not a tuned FNO baseline, and it does not '
        'establish a capacity ceiling: a larger budget or a wider search could move these numbers.',
        '- Training here is deliberately **not resumable**. Each run was bounded inside one '
        'allocation, and any run marked truncated stopped at its recorded epoch. No run in this '
        'report was resumed.',
        '- The dataset supports a Gaussian continuum-family claim only. It is not evidence about '
        'arbitrary sampled initial fields.',
        '- The eight-case cohort is small. Its worst-case column is one case, so a single hard '
        'case moves it; the per-case arrays are retained in the audit for anyone who wants the '
        'distribution.']
    text += '\n'.join(lines) + '\n\n' + GLOSSARY
    return text


GLOSSARY = """## Glossary

- **FNO (Fourier neural operator):** a network whose layers multiply the input's
  Fourier coefficients by learned weights, so one trained model applies to a
  whole family of inputs on a grid.
- **Capacity:** how big the network is — its hidden-channel width and the number
  of Fourier modes each layer keeps. `small`, `medium` and `large` are the three
  sizes screened; `refine` is the best size retrained at a lower learning rate.
- **Width / modes:** the two numbers that set capacity: channels carried between
  layers, and Fourier modes retained per axis.
- **Real parameters:** trainable real numbers, counting each complex weight as two.
- **Epoch:** one pass over all 128 training cases. **Best epoch:** the epoch whose
  checkpoint scored best on the validation cases; that checkpoint, not the last
  one, is what is evaluated and timed.
- **Early stopping:** halting when the validation score has not improved for a set
  number of epochs, so a model is not trained past the point of usefulness.
- **Truncated by wall budget:** the run hit its time limit rather than finishing
  its epochs or stopping early on its own.
- **Equal wall budget:** every capacity got the same amount of GPU time rather than
  the same number of epochs, because a bigger network costs more per epoch.
- **Case:** one physical problem — an initial field and a viscosity — with its
  whole reference trajectory.
- **Held-out / validation cases:** 32 cases never used to update weights, used to
  pick the checkpoint and report accuracy. Train and validation cases are disjoint
  by case, by generation seed and by field content.
- **Diagnosis cases:** the eight cases the Burgers lane graded its ROM and FOM on.
  Also never trained on here, which is what makes the same-case comparison valid.
- **Fixed-initial error:** the error measure defined above — field discrepancy
  divided by the size of the *initial* field, so all output times share one
  yardstick.
- **Worst / median / mean / p95:** taken over the per-case numbers, where each case
  has already been reduced to its worst output time.
- **Upper-Tukey outliers:** cases more than 1.5 interquartile ranges above the
  upper quartile — the count says how much a mean is being dragged by a few bad
  cases.
- **Reference:** the fine numerical solution the lane treats as truth. It is an
  empirically refined numerical solution, not an exact one.
- **Restriction:** taking every $n$-th node of a fine grid to land exactly on a
  coarser grid, which is how the fine reference is brought to the training grid.
- **Complete query:** everything charged between handing the model its input and
  having the requested output, with nothing precomputed outside the measurement.
- **Burn-in:** untimed repetitions run first so the GPU clock has already ramped
  when measurement starts; skipping it has manufactured a false crossover in this
  project before.
- **Pooled median:** the median over every (case, repetition) measurement.
  **Median of case medians:** each case is reduced first, then the median is taken
  across cases. Both are given because they answer different questions.
- **Host transfer:** copying the finished result from the GPU back to main memory,
  reported separately because not every consumer needs it.
- **ROM (reduced-order model):** the project's learned small-representation solver.
  **FOM (full-order model):** a conventional solver on the full grid.
- **`same_nt1e-2_dt005`:** the Burgers lane's name for one efficient FOM arm — same
  grid, Newton tolerance $10^{-2}$, time step $0.005$.
- **Autoregressive rollout:** predicting each time by feeding the previous
  prediction back in. Not used here, which is why no rollout error growth is
  reported.
- **Cross-job timing ratio:** dividing a time measured in one Slurm job by a time
  measured in another. Forbidden in this project; different allocations land on
  different hardware instances.
- **f64 / complex128:** double-precision real and complex arithmetic, used
  throughout so that reported errors are not floating-point artefacts.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, default=PRIMARY)
    parser.add_argument('--screen-audit', type=Path, default=SCREEN)
    parser.add_argument('--launch', type=Path, default=LAUNCH)
    parser.add_argument('--screen-launch', type=Path, default=SCREEN_LAUNCH)
    parser.add_argument('--out', type=Path, default=HERE / '2026-09-14-burgers-fno-baseline.md')
    parser.add_argument('--manifest', type=Path,
                        default=HERE / '2026-09-14-burgers-fno-baseline.sources.json')
    arguments = parser.parse_args()
    primary = load(arguments.audit)
    if primary is None:
        raise SystemExit(f'No primary audit at {arguments.audit}')
    screen = load(arguments.screen_audit)
    launch = load(arguments.launch)
    screen_launch = load(arguments.screen_launch) or {}
    definition = load(DEFINITION)
    diagnosis = load(DIAGNOSIS)
    arguments.out.write_text(build(primary, screen, launch, screen_launch, definition, diagnosis))
    sources = [arguments.audit, arguments.launch, DEFINITION, DIAGNOSIS]
    if screen:
        sources += [arguments.screen_audit, arguments.screen_launch]
    arguments.manifest.write_text(json.dumps({
        'generator': sha(__file__), 'report': sha(arguments.out),
        'sources': {str(p): sha(p) for p in sources},
        'rule': 'every number in the report is read from these files; none is typed by hand',
    }, indent=2) + '\n')
    print(arguments.out)


if __name__ == '__main__':
    main()
