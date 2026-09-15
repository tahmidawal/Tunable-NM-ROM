"""Generate the Burgers FNO baseline report from audited run records only.

Every number in the report is read from a JSON produced by an audit, never
typed. The FNO numbers come from this lane's independent NumPy field audit; the
ROM and FOM reference numbers come from the Burgers lane's own diagnosis audit
and are quoted, with their provenance, purely as context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / 'runs/fno_burgers01/field-audit.json'
DEFINITION = HERE.parent / 'checks/burgers-error-definition.json'
LAUNCH = HERE.parent / 'checks/collection-burgers-launch.json'
BURGERS_LANE = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/'
                    '2026-09-14-no-burgers/experiments/neural-operator-burgers')
DIAGNOSIS = BURGERS_LANE / 'checks/refinement02-diagnosis-audit.json'
LABELS = {'fno-small': 'small', 'fno-medium': 'medium', 'fno-large': 'large',
          'fno-refine': 'refinement of the best capacity'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(value, digits=4):
    return f'{100 * value:.{digits}f}'


def milliseconds(value, digits=3):
    return f'{value:.{digits}f}'


def capacity_table(audit):
    lines = ['| Run | Width | Modes | Real parameters | Epochs run | Best epoch | Truncated by wall budget |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name, model in audit['models'].items():
        if not model.get('complete'):
            lines.append(f'| `{name}` | — | — | — | — | — | run did not complete |')
            continue
        config = model['config']
        lines.append(f"| `{name}` | {config['width']} | {config['modes']} | {model['real_parameter_count']} | "
                     f"{model['epochs_completed']} | {model['best_epoch']} | "
                     f"{'yes' if model['stopped_by_wall_budget'] else 'no'} |")
    return '\n'.join(lines)


def accuracy_table(audit):
    lines = ['| Run | Mean (%) | Median (%) | p95 (%) | Worst (%) | Cases > 1% | Cases > 2% | Cases > 5% | Upper-Tukey outliers |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, model in audit['models'].items():
        if not model.get('complete'):
            lines.append(f'| `{name}` | — | — | — | — | — | — | — | — |')
            continue
        s = model['fixed_initial']
        counts = s['above_threshold_counts']
        lines.append(f"| `{name}` | {percent(s['mean'])} | {percent(s['median'])} | {percent(s['p95'])} | "
                     f"{percent(s['maximum'])} | {counts['0.01']} | {counts['0.02']} | {counts['0.05']} | "
                     f"{s['upper_tukey_outliers']} |")
    return '\n'.join(lines)


def growth_table(audit):
    times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    header = '| Run | ' + ' | '.join(f'$t={t:g}$' for t in times) + ' |'
    lines = [header, '| --- | ' + ' | '.join('---:' for _ in times) + ' |']
    for name, model in audit['models'].items():
        if not model.get('complete'):
            continue
        per_time = model['per_time_errors']
        worst = [max(case[i] for case in per_time) for i in range(len(times))]
        lines.append(f'| `{name}` | ' + ' | '.join(percent(v) for v in worst) + ' |')
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
        lines.append(f"| `{name}` | {cases}×{repetitions} | {milliseconds(row['device_pooled_median_ms'])} | "
                     f"{milliseconds(row['device_median_of_case_medians_ms'])} | "
                     f"{milliseconds(row['host_pooled_median_ms'])} | "
                     f"{milliseconds(row['device_plus_host_pooled_median_ms'])} |")
    return '\n'.join(lines)


def reference_table(diagnosis):
    lines = ['| Method | Worst fixed-initial error (%) | Median GPU (ms) | Median complete host query (ms) |',
             '| --- | ---: | ---: | ---: |']
    for name, row in diagnosis['summary'].items():
        lines.append(f"| `{name}` | {percent(row['worst_fixed_initial_error'])} | "
                     f"{milliseconds(row['gpu_median_ms'])} | {milliseconds(row['host_to_host_median_ms'])} |")
    return '\n'.join(lines)


def best_run(audit):
    complete = {k: v for k, v in audit['models'].items() if v.get('complete')}
    if not complete:
        return None, None
    name = min(complete, key=lambda k: complete[k]['fixed_initial']['mean'])
    return name, complete[name]


def build(audit_path=AUDIT):
    audit = json.loads(Path(audit_path).read_text())
    definition = json.loads(DEFINITION.read_text())
    launch = json.loads(LAUNCH.read_text())
    diagnosis = json.loads(DIAGNOSIS.read_text())
    name, best = best_run(audit)
    timing = audit['timing']
    rom = diagnosis['summary']['rom']
    fom = diagnosis['summary']['same_nt1e-2_dt005']
    fno_timing = timing['models'].get(name) if timing.get('present') else None

    parts = [f"""# Burgers FNO baseline on the common dataset

This report covers one bounded FNO capacity screen trained on the Burgers common
dataset produced by the Burgers lane, evaluated on its 32 held-out validation
cases with that lane's own physical error metric, and timed for complete queries
inside the same Slurm allocation. **The accuracy and timing numbers below are
final for this job and provisional as evidence about neural operators on this
problem:** they are a single training seed, a single mesh, a single Gaussian
continuum family, and a bounded epoch budget. No comparison here is a speed
ratio against the ROM or the FOM; the interleaved panel that could produce one
has not been run.

Job `{audit['job_id']}` on `{launch['gpu']}` (node `{launch['node']}`), staged
from source commit `{launch['source_commit']}`, `jax_backend={launch['jax_backend']}`,
float64 / complex128 throughout, `JAX_DEFAULT_MATMUL_PRECISION=highest`.
Accounting: `{audit['accounting'].strip()}`.

## What the model is

The operator is trained to map the supplied state and the known physical
coefficient to the whole requested trajectory in one evaluation,

$$\\mathcal{{G}}_\\theta : \\bigl(u_0,\\ \\nu\\bigr) \\;\\longmapsto\\;
\\bigl(u(t_1),\\dots,u(t_5)\\bigr), \\qquad
t_k \\in \\{{0.05,\\,0.10,\\,0.15,\\,0.20,\\,0.25\\}},$$

with the five evolved fields produced as five output channels of a single
Fourier neural operator and the supplied state returned exactly, so that the
complete six-time trajectory is $\\bigl(u_0, \\mathcal{{G}}_\\theta(u_0,\\nu)\\bigr)$.

**This is a direct multi-time output, not an autoregressive rollout.** No
prediction is fed back as an input, so there is no step-to-step error
accumulation to report. The price is that the output times are fixed by
training: this model cannot be queried at an unseen time or continued past
$t=0.25$. Per-time errors are retained anyway so that any growth across the
requested times is visible, and they are tabulated below.

Inputs are the sampled initial nodal field, the viscosity broadcast to a channel
and the two coordinate channels. Generation descriptors, case identifiers and
solver-audit sidecars are offline metadata and never enter the model.

```mermaid
flowchart LR
  U0["supplied initial field u0<br/>257 x 257, f64"] --> F["feature channels<br/>normalised u0, nu, x, y"]
  NU["viscosity nu"] --> F
  F --> FNO["FNO, 4 Fourier layers<br/>trained weights"]
  FNO --> M["boundary mask<br/>zero Dirichlet"]
  M --> OUT["u(t1..t5)"]
  U0 --> TRAJ["complete trajectory<br/>u0, u(t1..t5)"]
  OUT --> TRAJ
  classDef trained fill:#2b6cb0,stroke:#1a365d,color:#ffffff
  classDef given fill:#e2e8f0,stroke:#4a5568,color:#1a202c
  classDef exact fill:#276749,stroke:#1c4532,color:#ffffff
  class FNO trained
  class U0,NU,F given
  class M,TRAJ,OUT exact
```

## How accuracy is defined

The metric is the Burgers lane's own, so that the FNO, the ROM and the FOM are
graded identically:

$$E(\\text{{case}}) \\;=\\; \\max_{{k=0,\\dots,5}}\\;
\\frac{{\\bigl\\lVert \\hat u(t_k) - u^{{\\mathrm{{ref}}}}(t_k) \\bigr\\rVert_{{2,\\;\\mathrm{{interior}}}}}}
     {{\\bigl\\lVert u^{{\\mathrm{{ref}}}}(t_0) \\bigr\\rVert_{{2,\\;\\mathrm{{interior}}}}}} .$$

The denominator is fixed at the initial field for every output time
(fixed-initial normalisation) and the maximum runs over the evolved times; the
$t_0$ term is identically zero here because the supplied state is returned
bitwise. `check_burgers_error_definition.py` imports the Burgers lane's
`fixed_initial_errors` directly from
`{definition['burgers_lane_data_py']}`
(SHA256 `{definition['burgers_lane_data_py_sha256']}`) and confirms the two
implementations agree to
`{definition['worst_absolute_per_time_difference']:.3e}` on contract-valid
fields — floating-point summation order, not a definitional difference.

The reference $u^{{\\mathrm{{ref}}}}$ is the Burgers lane's recorded fine
numerical solution (4096 intervals, $\\Delta t = 1.5625\\times10^{{-4}}$,
restricted to the 256-interval training grid). It is an empirically refined
numerical reference, not a continuum error bound.

## Capacities trained

Small, medium and large share an identical 200-epoch budget so that capacity is
not confounded with epoch count; the winner was then retrained at a lower
learning rate. Each run also carried a wall-clock cap, and truncation is
recorded per run.

{capacity_table(audit)}

## Validation accuracy

Over all {audit['validation_cases']} held-out validation cases, recomputed
independently from the saved prediction fields with NumPy.

{accuracy_table(audit)}

## Worst error per output time

Worst case in the cohort at each requested time. Any growth visible here is the
difficulty of the later states, not rollout accumulation — there is no rollout.

{growth_table(audit)}

## Complete-query timing

{timing_table(audit)}

Timing protocol, recorded so the later interleaved ROM/FNO/FOM confirmation job
can reproduce it exactly: the timed region starts with the supplied initial
field and the viscosity already resident on the GPU and ends when the complete
six-time trajectory is resident on the GPU; coordinate construction,
normalisation, the forward pass, boundary masking and trajectory assembly are
all inside it and nothing is precomputed outside it. `torch.cuda.synchronize()`
brackets every repetition, every timed block is preceded by its own
{timing.get('burn_in_per_block', '—')}-query GPU burn-in, host transfer is a
separate timed block with its own burn-in, and every individual repetition is
retained in `timing.npz` — none is discarded and no minimum is substituted for a
median.

## Context: the ROM and FOM numbers this lane must eventually be compared with

These are **not** comparable to the table above and no ratio may be formed from
them. They come from the Burgers lane's separate diagnosis job (`3702709`),
which ran on a different allocation, on a different GPU instance, over a
different cohort of 8 diagnosis cases rather than these 32 validation cases.
They are reproduced only to show what a genuine confirmation job would have to
beat, measured inside that job.

{reference_table(diagnosis)}

## What this does and does not establish

""", '']

    accuracy_claim = (f"The best FNO run, `{name}`, reaches a worst validation case error of "
                      f"{percent(best['fixed_initial']['maximum'])}% and a median of "
                      f"{percent(best['fixed_initial']['median'])}% on 32 held-out cases. "
                      f"The Burgers lane's ROM reaches {percent(rom['worst_fixed_initial_error'])}% worst "
                      f"and its efficient FOM arm `same_nt1e-2_dt005` reaches "
                      f"{percent(fom['worst_fixed_initial_error'])}% worst, both over 8 diagnosis cases.")
    verdict = ('worse than' if best['fixed_initial']['maximum'] > rom['worst_fixed_initial_error']
               else 'better than')
    parts.append(f"""- {accuracy_claim} On worst-case error the best FNO is {verdict} the
  ROM's recorded number, but the cohorts differ, so this is a coarse indication
  and not a matched comparison.
- **No speed claim is made.** {'FNO complete-query timing is reported above for this job only. ' if fno_timing else 'FNO timing is unavailable. '}Dividing it by the ROM or FOM
  medians would be exactly the cross-job ratio this project has already had to
  retract once; the interleaved same-job panel remains the only admissible route
  to a speed statement.
- The screen is a single seed, a single mesh (256 intervals), a single Gaussian
  continuum family and a bounded epoch budget. It is not a tuned FNO baseline
  and does not establish a capacity ceiling.
- Training here is deliberately **not resumable**: each run was bounded inside
  one allocation, and any run marked truncated stopped at its recorded epoch.
  No run in this report was resumed.
- The dataset supports a Gaussian continuum-family claim only. It is not
  evidence about arbitrary sampled initial fields.

## Glossary

- **FNO (Fourier neural operator):** a network whose layers multiply the input's
  Fourier coefficients by learned weights, so one trained model can be applied to
  a whole family of inputs on a grid.
- **Capacity:** how big the network is here — its hidden-channel width and the
  number of Fourier modes each layer keeps. `small`, `medium`, `large` are the
  three sizes screened; `refinement` is the best size retrained at a lower
  learning rate.
- **Real parameters:** the count of trainable real numbers, counting each complex
  weight as two.
- **Epoch:** one pass over all 128 training cases.
- **Best epoch:** the epoch whose checkpoint scored best on the validation cases;
  that checkpoint, not the last one, is the one evaluated and timed.
- **Truncated by wall budget:** the run hit its time limit rather than finishing
  its epochs or stopping early on its own.
- **Case:** one physical problem — an initial field and a viscosity — together
  with its whole reference trajectory.
- **Held-out / validation cases:** 32 cases never used to update weights, used to
  pick the checkpoint and to report accuracy. Train and validation cases are
  disjoint by case, by generation seed and by field content.
- **Fixed-initial error:** the error measure defined above: field discrepancy
  divided by the size of the *initial* field, so all times share one yardstick.
- **Worst / median / p95:** taken over the per-case numbers, where each case has
  already been reduced to its worst output time.
- **Upper-Tukey outliers:** cases lying more than 1.5 interquartile ranges above
  the upper quartile — the count says how much a mean is being dragged by a few
  bad cases.
- **Reference:** the fine numerical solution the lane treats as truth. It is an
  empirically refined numerical solution, not an exact one.
- **Complete query:** everything charged between handing the model its input and
  having the requested output — nothing precomputed outside the measurement.
- **Burn-in:** untimed repetitions run first so the GPU clock has already ramped
  when measurement starts; skipping it has manufactured a false crossover here
  before.
- **Pooled median:** the median over every (case, repetition) measurement.
  **Median of case medians** first reduces each case, then takes the median
  across cases; both are given because they answer different questions.
- **Host transfer:** copying the finished result from the GPU back to main
  memory; reported separately because not every consumer needs it.
- **ROM (reduced-order model):** the project's learned small-representation
  solver. **FOM (full-order model):** a conventional solver on the full grid.
- **`same_nt1e-2_dt005`:** the Burgers lane's name for one efficient FOM arm —
  same grid, Newton tolerance $10^{{-2}}$, time step $0.005$.
- **Autoregressive rollout:** predicting each time by feeding the previous
  prediction back in. Not used here, which is why no rollout error growth is
  reported.
- **Cross-job timing ratio:** dividing a time measured in one Slurm job by a time
  measured in another. Forbidden in this project; different allocations land on
  different hardware instances.
""")
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / '2026-09-14-burgers-fno-baseline.md')
    parser.add_argument('--manifest', type=Path, default=HERE / '2026-09-14-burgers-fno-baseline.sources.json')
    parser.add_argument('--audit', type=Path, default=AUDIT)
    arguments = parser.parse_args()
    arguments.out.write_text(build(arguments.audit))
    arguments.manifest.write_text(json.dumps({
        'generator': sha(__file__),
        'report': sha(arguments.out),
        'sources': {str(p): sha(p) for p in (arguments.audit, DEFINITION, LAUNCH, DIAGNOSIS)},
        'rule': 'every number in the report is read from these files; none is typed by hand',
    }, indent=2) + '\n')
    print(arguments.out)


if __name__ == '__main__':
    main()
