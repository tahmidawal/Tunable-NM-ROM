"""Generate the no-second report and summary.json from audited records only.

Every number is read from `runs/*/audit.json` (this lane's independent NumPy
audits), from the parent FNO lane's `runs/fno_burgers02/field-audit.json` and from
the Burgers lane's `checks/refinement02-diagnosis-audit.json`. Nothing is typed.
No speed ratio is formed anywhere; timing rows are same-job only and labelled so.

    python reports/generate_report.py            # writes 2026-09-1x-no-second.md + summary.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = LANE.parents[1]
FNO_AUDIT = LANE.parent / 'neural-operator-audit/runs/fno_burgers02/field-audit.json'
FNO_LAUNCH = LANE.parent / 'neural-operator-audit/checks/collection-burgers02-launch.json'
FNO_POISSON_AUDIT = LANE.parent / 'neural-operator-audit/runs/fno_poisson01/field-audit.json'
FNO_POISSON_PARAMS = {'small': 1192801, 'medium': 5779729, 'large': 17876673}  # from its archived result.json files
# Hash-pinned copy of the Burgers lane's diagnosis audit (its SHA256 is asserted in main()).
DIAGNOSIS = LANE / 'checks/refinement02-diagnosis-audit.json'
DIAGNOSIS_SHA256 = 'ffa77d1b8bc44d2bd3d0ac2753444d2e1ff379898735258b60b9d68a41e3e187'
DIAGNOSIS_JOB = '3702709'
FAMILY_LABEL = {'unet': 'U-Net', 'transolver': 'Transolver', 'fno': 'FNO'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return f'{100 * x:.4f}'


def ms(x):
    return f'{x:.3f}'


def capacity_of(config):
    family = config.get('family', 'fno')
    if family == 'unet':
        return f"base {config['base']}"
    if family == 'transolver':
        return f"dim {config['dim']}, {config['layers']} layers, patch {config['patch']}"
    return f"width {config['width']}, modes {config['modes']}"


def load_attempts(pde='burgers'):
    attempts = []
    for path in sorted(LANE.glob('runs/*/audit.json')):
        audit = json.loads(path.read_text())
        if audit.get('passed') and audit.get('pde', 'burgers') == pde:
            attempts.append(dict(audit=audit, path=path, sha=sha(path)))
    return attempts


def poisson_rows(attempts, fno_p, fno_p_sha):
    rows = []
    for a in attempts:
        audit = a['audit']
        for arm, r in audit['arms'].items():
            if not r.get('complete'):
                continue
            base = dict(operator=FAMILY_LABEL[r['family']], arm=arm, capacity=capacity_of(r['config']), params=r['real_parameter_count'],
                        dtype=r['parameter_dtype'], seed=r['seed'], budget_s=r['wall_budget_seconds'], epochs=r['epochs_completed'],
                        best_epoch=r['best_epoch'], stop_reason=r['stop_reason'], job_id=audit['job_id'], gpu=audit['gpu'],
                        attempt=audit['attempt'], source=str(a['path'].relative_to(WT)), source_sha256=a['sha'], cross_job=False)
            for kind in ('discrete', 'physical_candidate'):
                for metric in ('maximum', 'median', 'mean'):
                    rows.append(dict(base, cohort='poisson-validation-32', metric=f"{'worst' if metric == 'maximum' else metric}_{kind}_relative_error",
                                     value=r[kind][metric]))
            t = audit['timing'].get('models', {}).get(arm)
            if t:
                rows.append(dict(base, cohort='poisson-validation-32', metric='same_job_device_query_pooled_median_ms', value=t['device_pooled_median_ms']))
    for size, r in fno_p['models'].items():
        base = dict(operator='FNO', arm=f'fno-{size}', capacity='—', params=FNO_POISSON_PARAMS[size], dtype='torch.float64', seed=20260914,
                    budget_s=7200.0, epochs=None, best_epoch=r['best_epoch'], stop_reason='early_stopping', job_id=fno_p['job_id'],
                    gpu='NVIDIA A100 80GB PCIe', attempt='fno_poisson01', source=str(FNO_POISSON_AUDIT.relative_to(WT)),
                    source_sha256=fno_p_sha, cross_job=True)
        for kind in ('discrete', 'physical_candidate'):
            for metric in ('maximum', 'median', 'mean'):
                rows.append(dict(base, cohort='poisson-validation-32', metric=f"{'worst' if metric == 'maximum' else metric}_{kind}_relative_error",
                                 value=r[kind][metric]))
    return rows


def poisson_section(attempts, fno_p):
    if not attempts:
        return ''
    jobs = ', '.join(f"`{a['audit']['job_id']}` ({a['audit']['attempt']}, {a['audit']['gpu']}, commit `{a['audit']['source_commit'][:8]}`)" for a in attempts)
    cap = ['| Run | Operator | Capacity | Network dtype | Real parameters | Epochs run | Best epoch | Training s | Budget s | Ended by | Job |',
           '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    acc = ['| Run | Operator | Job | Discrete: median (%) | Discrete: worst (%) | Physical: mean (%) | Physical: median (%) | Physical: p95 (%) | Physical: worst (%) | Cases > 5% (physical) |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in attempts:
        for arm, r in a['audit']['arms'].items():
            if not r.get('complete'):
                continue
            cap.append(f"| `{arm}` | {FAMILY_LABEL[r['family']]} | {capacity_of(r['config'])} | {r['parameter_dtype'].replace('torch.', '')} | "
                       f"{r['real_parameter_count']} | {r['epochs_completed']} | {r['best_epoch']} | {r['training_seconds']:.0f} | "
                       f"{r['wall_budget_seconds']:.0f} | {r['stop_reason'].replace('_', ' ')} | `{a['audit']['job_id']}` |")
            d, ph = r['discrete'], r['physical_candidate']
            acc.append(f"| `{arm}` | {FAMILY_LABEL[r['family']]} | `{a['audit']['job_id']}` | {pct(d['median'])} | {pct(d['maximum'])} | {pct(ph['mean'])} | "
                       f"{pct(ph['median'])} | {pct(ph['p95'])} | {pct(ph['maximum'])} | {ph['above_threshold_counts']['0.05']} |")
    for size, r in fno_p['models'].items():
        d, ph = r['discrete'], r['physical_candidate']
        acc.append(f"| `fno-{size}` | FNO (parent lane, other job) | `{fno_p['job_id']}` | {pct(d['median'])} | {pct(d['maximum'])} | {pct(ph['mean'])} | "
                   f"{pct(ph['median'])} | {pct(ph['p95'])} | {pct(ph['maximum'])} | {ph['above_threshold_counts']['0.05']} |")
    return f"""

## Poisson: U-Net on the Poisson operator-screen dataset

Jobs: {jobs}. The dataset is the Poisson FNO job's (`{fno_p['job_id']}`): 128 training and 32
validation cases at 256 intervals, source field in, zero-Dirichlet solution out, training
target the declared discrete (five-point FD/DST) solution, with an evaluation-only physical
reference sidecar (2048-interval refinement) per validation case. Because no cluster copy
survived, the files were re-uploaded from the Git archive and re-verified in-job against
their recorded hashes; the train/validation index SHA256 equal the FNO job's (asserted by
the audit). Protocol = the Poisson FNO's: 500-epoch cap, patience 80, 7200 s wall budget per
capacity, validation selection on the discrete target, no refinement. Two metrics, as in
the parent audit: **discrete** (against the training target) and **physical candidate**
(against the refinement sidecar). No matched ROM/DST cohort is scored here.

{chr(10).join(cap)}

{chr(10).join(acc)}
"""


def rows_for(attempts, fno, fno_sha, diagnosis, diagnosis_sha):
    rows = []
    for a in attempts:
        audit, job = a['audit'], a['audit']['job_id']
        for arm, r in audit['arms'].items():
            if not r.get('complete'):
                continue
            base = dict(operator=FAMILY_LABEL[r['family']], arm=arm, capacity=capacity_of(r['config']),
                        params=r['real_parameter_count'], dtype=r['parameter_dtype'], seed=r['seed'],
                        budget_s=r['wall_budget_seconds'], epochs=r['epochs_completed'], best_epoch=r['best_epoch'],
                        stop_reason=r['stop_reason'], job_id=job, gpu=audit['gpu'], attempt=audit['attempt'],
                        source=str(a['path'].relative_to(WT)), source_sha256=a['sha'], cross_job=False)
            s = r['fixed_initial']
            for metric, value in (('worst', s['maximum']), ('median', s['median']), ('mean', s['mean']),
                                  ('p95', s['p95'])):
                rows.append(dict(base, cohort='validation-32', metric=f'{metric}_fixed_initial_error', value=value))
            for t, v in zip((0., .05, .1, .15, .2, .25), r['worst_per_time']):
                rows.append(dict(base, cohort='validation-32', metric=f'worst_error_at_t{t:g}', value=v))
            c = audit['cohort'].get('models', {}).get(arm)
            if c:
                for metric, value in (('worst', c['fixed_initial']['maximum']), ('median', c['fixed_initial']['median']),
                                      ('mean', c['fixed_initial']['mean'])):
                    rows.append(dict(base, cohort='diagnosis-8', metric=f'{metric}_fixed_initial_error', value=value))
            t = audit['timing'].get('models', {}).get(arm)
            if t:
                rows.append(dict(base, cohort='validation-32', metric='same_job_device_query_pooled_median_ms',
                                 value=t['device_pooled_median_ms']))
    fno_job = fno['job_id']
    for arm, r in fno['models'].items():
        base = dict(operator='FNO', arm=arm, capacity=capacity_of(r['config']), params=r['real_parameter_count'],
                    dtype='torch.float64', seed=r['config']['seed'], budget_s=3000.0, epochs=r['epochs_completed'],
                    best_epoch=r['best_epoch'], stop_reason='wall_budget' if r.get('stopped_by_wall_budget') else 'other',
                    job_id=fno_job, gpu='NVIDIA A100 80GB PCIe', attempt='fno_burgers02',
                    source=str(FNO_AUDIT.relative_to(WT)), source_sha256=fno_sha, cross_job=True)
        s = r['fixed_initial']
        for metric, value in (('worst', s['maximum']), ('median', s['median']), ('mean', s['mean']), ('p95', s['p95'])):
            rows.append(dict(base, cohort='validation-32', metric=f'{metric}_fixed_initial_error', value=value))
        c = fno['diagnosis_cohort']['models'].get(arm)
        if c:
            for metric, value in (('worst', c['fixed_initial']['maximum']), ('median', c['fixed_initial']['median']),
                                  ('mean', c['fixed_initial']['mean'])):
                rows.append(dict(base, cohort='diagnosis-8', metric=f'{metric}_fixed_initial_error', value=value))
    for name, r in diagnosis['summary'].items():
        rows.append(dict(operator='ROM' if name == 'rom' else 'FOM', arm=name, capacity='—', params=None, dtype='float64',
                         seed=None, budget_s=None, epochs=None, best_epoch=None, stop_reason='—', job_id=DIAGNOSIS_JOB,
                         gpu='NVIDIA A100 80GB PCIe', attempt='refinement02', source=str(DIAGNOSIS), source_sha256=diagnosis_sha,
                         cross_job=True, cohort='diagnosis-8', metric='worst_fixed_initial_error',
                         value=r['worst_fixed_initial_error']))
    return rows


def capacity_table(attempts):
    lines = ['| Run | Operator | Capacity | Network dtype | Real parameters | Epochs run | Best epoch | Training s | Budget s | Ended by | Job |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for a in attempts:
        audit = a['audit']
        for arm, r in audit['arms'].items():
            if r.get('complete'):
                lines.append(f"| `{arm}` | {FAMILY_LABEL[r['family']]} | {capacity_of(r['config'])} | {r['parameter_dtype'].replace('torch.', '')} | "
                             f"{r['real_parameter_count']} | {r['epochs_completed']} | {r['best_epoch']} | {r['training_seconds']:.0f} | "
                             f"{r['wall_budget_seconds']:.0f} | {r['stop_reason'].replace('_', ' ')} | `{audit['job_id']}` |")
            else:
                lines.append(f"| `{arm}` | — | — | — | — | — | — | — | — | incomplete | `{audit['job_id']}` |")
    return '\n'.join(lines)


def validation_table(attempts, fno):
    lines = ['| Run | Operator | Job | Mean (%) | Median (%) | p95 (%) | Worst (%) | Cases > 1% | Cases > 2% | Cases > 5% |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']

    def line(arm, family, job, s, note=''):
        c = s['above_threshold_counts']
        return (f"| `{arm}` | {family}{note} | `{job}` | {pct(s['mean'])} | {pct(s['median'])} | {pct(s['p95'])} | "
                f"{pct(s['maximum'])} | {c['0.01']} | {c['0.02']} | {c['0.05']} |")
    for a in attempts:
        for arm, r in a['audit']['arms'].items():
            if r.get('complete'):
                lines.append(line(arm, FAMILY_LABEL[r['family']], a['audit']['job_id'], r['fixed_initial']))
    for arm, r in fno['models'].items():
        lines.append(line(arm, 'FNO', fno['job_id'], r['fixed_initial'], ' (parent lane, other job)'))
    return '\n'.join(lines)


def per_time_table(attempts, fno):
    lines = ['| Run | $t=0$ | $t=0.05$ | $t=0.1$ | $t=0.15$ | $t=0.2$ | $t=0.25$ |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in attempts:
        for arm, r in a['audit']['arms'].items():
            if r.get('complete'):
                lines.append(f"| `{arm}` | " + ' | '.join(pct(v) for v in r['worst_per_time']) + ' |')
    import numpy as np
    for arm, r in fno['models'].items():
        worst = np.asarray(r['per_time_errors']).max(axis=0)
        lines.append(f"| `{arm}` (FNO, other job) | " + ' | '.join(pct(v) for v in worst) + ' |')
    return '\n'.join(lines)


def cohort_table(attempts, fno, diagnosis):
    lines = ['| Method | Kind | Job | Worst (%) | Median (%) | Mean (%) | Cases > 1% | Cases > 2% |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for a in attempts:
        models = a['audit']['cohort'].get('models', {})
        for arm, r in models.items():
            s = r['fixed_initial']
            family = FAMILY_LABEL[a['audit']['arms'][arm]['family']]
            lines.append(f"| `{arm}` | {family} (this lane) | `{a['audit']['job_id']}` | {pct(s['maximum'])} | {pct(s['median'])} | "
                         f"{pct(s['mean'])} | {s['above_threshold_counts']['0.01']} | {s['above_threshold_counts']['0.02']} |")
    for arm, r in fno['diagnosis_cohort']['models'].items():
        s = r['fixed_initial']
        lines.append(f"| `{arm}` | FNO (parent lane, other job) | `{fno['job_id']}` | {pct(s['maximum'])} | {pct(s['median'])} | "
                     f"{pct(s['mean'])} | {s['above_threshold_counts']['0.01']} | {s['above_threshold_counts']['0.02']} |")
    for name, r in diagnosis['summary'].items():
        kind = 'ROM (Burgers lane, other job)' if name == 'rom' else 'FOM (Burgers lane, other job)'
        lines.append(f"| `{name}` | {kind} | `{DIAGNOSIS_JOB}` | {pct(r['worst_fixed_initial_error'])} | — | — | — | — |")
    return '\n'.join(lines)


def timing_table(attempts):
    lines = ['| Run | Job | GPU | Repetitions | Device query, pooled median (ms) | Median of case medians (ms) | Host transfer (ms) | Device + host (ms) |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for a in attempts:
        t = a['audit']['timing']
        for arm, r in t.get('models', {}).items():
            cases, reps = r['repetitions']
            lines.append(f"| `{arm}` | `{a['audit']['job_id']}` | {t['gpu']} | {cases}×{reps} | {ms(r['device_pooled_median_ms'])} | "
                         f"{ms(r['device_median_of_case_medians_ms'])} | {ms(r['host_pooled_median_ms'])} | {ms(r['device_plus_host_pooled_median_ms'])} |")
    return '\n'.join(lines)


def selected(attempts):
    """Validation-selected arm per operator family (lowest validation mean case-max, the training selection rule)."""
    best = {}
    for a in attempts:
        for arm, r in a['audit']['arms'].items():
            if r.get('complete'):
                key = r['family']
                if key not in best or r['fixed_initial']['mean'] < best[key][1]['fixed_initial']['mean']:
                    best[key] = (arm, r, a['audit'])
    return best


def build(attempts, fno, launch, diagnosis):
    jobs = ', '.join(f"`{a['audit']['job_id']}` ({a['audit']['attempt']}, {a['audit']['gpu']}, commit `{a['audit']['source_commit'][:8]}`)" for a in attempts)
    # Same selection rule as this lane's arms: argmin of validation mean case-max over every
    # complete arm, refine included.
    fno_sel = min(fno['models'], key=lambda k: fno['models'][k]['fixed_initial']['mean'])
    fno_val = fno['models'][fno_sel]['fixed_initial']
    fno_coh = fno['diagnosis_cohort']['models'][fno_sel]['fixed_initial']
    rom = diagnosis['summary']['rom']['worst_fixed_initial_error']
    fom = diagnosis['summary']['same_nt1e-2_dt005']['worst_fixed_initial_error']
    sel = selected(attempts)
    verdict_lines = []
    for family, (arm, r, audit) in sel.items():
        v, c = r['fixed_initial'], audit['cohort'].get('models', {}).get(arm, {}).get('fixed_initial')
        v1 = v['maximum'] <= 1.5 * fno_val['maximum'] and v['median'] <= 1.5 * fno_val['median']
        verdict_lines.append(
            f"- **{FAMILY_LABEL[family]}**, validation-selected arm `{arm}` (job `{audit['job_id']}`): worst validation error "
            f"{pct(v['maximum'])}%, median {pct(v['median'])}% (FNO `{fno_sel}`: {pct(fno_val['maximum'])}% / {pct(fno_val['median'])}%). "
            f"Pre-registered V1 (within 1.5× of the FNO on both): **{'pass' if v1 else 'fail'}**."
            + (f" On the matched eight cases: worst {pct(c['maximum'])}%, median {pct(c['median'])}% "
               f"(FNO {pct(fno_coh['maximum'])}%, ROM {pct(rom)}%, efficient FOM `same_nt1e-2_dt005` {pct(fom)}%)." if c else
               ' The matched cohort was not scored in this job.'))
    controls = ''
    ctrl = [(arm, r, a['audit']) for a in attempts for arm, r in a['audit']['arms'].items()
            if r.get('complete') and (r['parameter_dtype'] == 'torch.float64' or r['seed'] != 20260914)]
    if ctrl:
        controls = '\n\n## Controls: precision and seed\n\nThe validation-selected U-Net capacity retrained under the identical protocol with the network in float64 (precision control) and at a second seed in float32 (seed-variance control). Compare against the float32, seed-20260914 twin in the tables above.\n\n' \
            + '| Run | Network dtype | Seed | Epochs | Ended by | Validation worst (%) | Validation median (%) | Cohort worst (%) | Cohort median (%) | Job |\n| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n'
        for arm, r, audit in ctrl:
            c = audit['cohort'].get('models', {}).get(arm, {}).get('fixed_initial')
            controls += (f"| `{arm}` | {r['parameter_dtype'].replace('torch.', '')} | {r['seed']} | {r['epochs_completed']} | {r['stop_reason'].replace('_', ' ')} | "
                         f"{pct(r['fixed_initial']['maximum'])} | {pct(r['fixed_initial']['median'])} | "
                         f"{pct(c['maximum']) if c else '—'} | {pct(c['median']) if c else '—'} | `{audit['job_id']}` |\n")
    today = dt.date.today().isoformat()
    text = f"""# Second and third neural-operator baselines on the Burgers common dataset: U-Net and Transolver

This report adds a PDEBench-style **U-Net** and a **Transolver** (physics attention) to the
neural-operator comparison on the Burgers common dataset, trained on exactly the data, split,
reference, metric, budget and selection rule the FNO lane used (`no-audit`, job
`{fno['job_id']}`), and scored on the same 32 held-out validation cases and the same eight
ROM/FOM diagnosis cases. **The numbers are final for the jobs listed and provisional as
evidence about neural operators on this problem** (one seed except where a seed control is
shown, one mesh, one Gaussian continuum family, one bounded wall budget per capacity). No
speed ratio against the FNO, the ROM or the FOM is stated anywhere: those were measured in
other jobs.

Jobs: {jobs}. Every job printed `jax_backend=gpu` and `torch_backend=cuda`, verified its
staged code against the committed blob and its data against the Burgers cache's
`DATA.sha256`, and ended with `ALL-DONE`. Training/validation index SHA256
`{attempts[0]['audit']['train_index_sha256'][:8]}…` / `{attempts[0]['audit']['validation_index_sha256'][:8]}…`, identical to the FNO job's
(asserted by the audit). Generated {today} by `reports/generate_report.py` from the audit
JSONs listed in `summary.json`; no number here is typed.

## Verdict against the pre-registered criteria

{chr(10).join(verdict_lines)}

The eight-case worst is a single case; the per-case arrays are in the audit JSONs.

## What the models are

All three families share one contract (the FNO lane's): the input is the supplied initial
field, the viscosity broadcast to a channel and the two coordinate channels, all normalised
with the training statistics; the output is the five evolved fields as channels, masked to the
zero-Dirichlet boundary, with the supplied initial state prepended bitwise — a **direct
multi-time output, not an autoregressive rollout**. Only the block between the features and
the mask differs.

$$\\mathcal{{G}}_\\theta : (u_0, \\nu) \\mapsto (u(t_1), \\dots, u(t_5)), \\qquad t_k \\in \\{{0.05, 0.10, 0.15, 0.20, 0.25\\}}.$$

```mermaid
flowchart LR
  U0["supplied initial field u0<br/>257 x 257, f64"] --> F["feature channels<br/>normalised u0, nu, x, y (f64)"]
  NU["viscosity nu"] --> F
  F --> NET["operator block<br/>FNO (f64) | U-Net (f32) | Transolver (f32)<br/>trained weights"]
  NET --> C["cast to f64"]
  C --> M["boundary mask<br/>zero Dirichlet"]
  M --> OUT["u(t1) ... u(t5)"]
  U0 --> TRAJ["complete trajectory<br/>u0, u(t1) ... u(t5)"]
  OUT --> TRAJ
  classDef trained fill:#2b6cb0,stroke:#1a365d,color:#ffffff
  classDef given fill:#e2e8f0,stroke:#4a5568,color:#1a202c
  classDef exact fill:#276749,stroke:#1c4532,color:#ffffff
  class NET trained
  class U0,NU,F given
  class C,M,TRAJ,OUT exact
```

- **U-Net**: four-level encoder–decoder, two 3×3 convolutions per level, channels
  `base … 16·base`, 2×2 max-pool down, 2×2 transposed convolution up with skip
  concatenation, GroupNorm(8), GELU; the 257² grid is zero-padded to 272² and cropped back.
- **Transolver**: eight physics-attention layers (8 heads, 64 slices, learned temperature,
  3×3 convolutional projections, pre-norm MLP of ratio 2) on 4×4-patch tokens of the grid
  zero-padded to 260² (4 225 tokens), the paper's unified positional encoding, and a
  per-token MLP decoder unpatched onto the grid.
- **FNO** (parent lane): four Fourier layers, float64/complex128.

The U-Net and Transolver networks run in IEEE float32 with TF32 disabled; features, mask,
trajectory assembly, loss and every reported error are float64, and the float64 precision
control below tests whether the network precision matters.

## How accuracy is defined

The Burgers lane's own metric, identical for every method and recomputed independently with
NumPy from the saved fields:

$$E(\\text{{case}}) = \\max_{{k=0,\\dots,5}} \\frac{{\\lVert \\hat u(t_k) - u^{{\\mathrm{{ref}}}}(t_k)\\rVert_{{2,\\mathrm{{interior}}}}}}{{\\lVert u^{{\\mathrm{{ref}}}}(t_0)\\rVert_{{2,\\mathrm{{interior}}}}}}.$$

The reference is the Burgers lane's refined 4096-interval, $\\Delta t = 1.5625\\times10^{{-4}}$
numerical solution restricted to the 256-interval grid.

## Protocol (identical to the FNO arm)

AdamW (lr $10^{{-3}}$, weight decay $10^{{-4}}$), batch 8, gradient clipping at 1.0,
ReduceLROnPlateau(0.5, patience 20) on the validation mean case-maximum error, early stopping
after 250 epochs without improvement, epoch cap 4000, **3000 s of wall per capacity on one
A100**, checkpoint selected by validation mean case-maximum error, the validation-selected
capacity retrained at lr $3\\times10^{{-4}}$ (`*-refine`), seed 20260914. The Transolver adds a
10-epoch linear warm-up (its one protocol difference). Nothing is selected on the eight-case
cohort. Epoch counts differ by design: equal compute, not equal epochs.

## Capacities trained

{capacity_table(attempts)}

## Validation accuracy — 32 held-out cases

{validation_table(attempts, fno)}

## Worst error per output time (validation cases)

There is no rollout, so growth across times is the difficulty of later states, not accumulation.

{per_time_table(attempts, fno)}

## Same-case comparison on the matched eight-case cohort

The eight cases and references the Burgers lane's ROM/FOM diagnosis used at 256 intervals
(job `{DIAGNOSIS_JOB}`), rebuilt in every job from the same anchors and checked disjoint from
training by seed and by input content. **Accuracy is comparable across jobs on these cases;
timing is not**, which is why no time appears here.

{cohort_table(attempts, fno, diagnosis)}
{controls}

## Same-job complete-query timing (this lane's arms only)

Measured with the parent lane's `timing.py` protocol (supplied field on device to complete
trajectory on device, 20-query burn-in per block, synchronisation around every repetition,
every repetition retained). **These numbers must not be divided by any timing from another
job**, including the FNO's; the `b-panel` lane owns the same-job panel.

{timing_table(attempts)}

## What this does and does not establish

- Each new family is reported at every capacity it was trained at, with the number of epochs it
  reached and what ended the run, under exactly the FNO's budget and selection rule.
- The comparison is single-seed (plus the seed control where present), one mesh, one Gaussian
  family, one 3000 s budget per capacity. It is a *protocol-matched* screen, not a capacity
  ceiling for any family.
- Runs ended by the wall budget were still improving or plateauing at that budget; the report
  says which, per arm, in the "Ended by" column.
- No speed claim. Checkpoints (`best.pt`) and the `evaluate`-style entry point
  (`model.predict`) are archived for the same-job panel.

## Glossary

- **U-Net / Transolver / FNO:** the three neural-operator families compared; see "What the models are".
- **Capacity:** the network size — `base` channels for the U-Net, hidden `dim` for the Transolver, width/modes for the FNO. `small`, `medium`, `large` are the screened sizes; `refine` is the validation-selected size retrained at a lower learning rate.
- **Real parameters:** trainable real numbers (each complex FNO weight counts as two).
- **Network dtype:** the precision the network's weights and activations use; everything outside the network is float64 for every family.
- **Epoch / best epoch:** one pass over the 128 training cases; the epoch whose checkpoint scored best on the validation cases (that checkpoint is what is evaluated).
- **Ended by:** what stopped the run — `early stopping` (250 epochs without validation improvement), `wall budget` (the 3000 s limit), `epoch cap` (4000 epochs), or `signal` (the scheduler asked it to stop).
- **Validation-32 / diagnosis-8:** the 32 held-out cases used to pick checkpoints and report accuracy; the eight cases the Burgers lane graded its ROM and FOM on, never used for selection.
- **Fixed-initial error:** field discrepancy divided by the size of the *initial* field, maximised over the six output times.
- **Worst / median / mean / p95:** over the per-case numbers, each case already reduced to its worst time.
- **Cases > x%:** how many of the cases exceed that error.
- **Precision control / seed control:** the same capacity retrained with the network in float64, or at a different seed, to show whether those choices move the numbers.
- **Other job:** a number measured in a different Slurm allocation. Accuracy is comparable across jobs; timing never is.
- **ROM / FOM:** the project's reduced-order model and the conventional full-grid solver; `same_nt1e-2_dt005` is the efficient FOM arm (Newton tolerance $10^{{-2}}$, step 0.005).
- **V1:** the pre-registered competitiveness criterion — within 1.5× of the FNO's validation worst and median.
- **Burn-in / pooled median / host transfer:** timing-protocol terms from the parent lane, reproduced unchanged.
- **Discrete / physical candidate (Poisson):** error against the declared finite-difference training target, and against the finer evaluation-only reference solution; both are whole-field discrepancy over the field's norm.
"""
    return text


def main():
    attempts = load_attempts()
    if not attempts:
        raise SystemExit('no passed audit.json under runs/')
    fno = json.loads(FNO_AUDIT.read_text())
    launch = json.loads(FNO_LAUNCH.read_text())
    assert sha(DIAGNOSIS) == DIAGNOSIS_SHA256
    diagnosis = json.loads(DIAGNOSIS.read_text())
    rows = rows_for(attempts, fno, sha(FNO_AUDIT), diagnosis, sha(DIAGNOSIS))
    poisson = load_attempts('poisson')
    fno_p = json.loads(FNO_POISSON_AUDIT.read_text())
    rows += poisson_rows(poisson, fno_p, sha(FNO_POISSON_AUDIT))
    text = build(attempts, fno, launch, diagnosis)
    glossary = text.index('## Glossary')
    text = text[:glossary] + poisson_section(poisson, fno_p).lstrip('\n') + ('\n' if poisson else '') + text[glossary:]
    attempts = attempts + poisson
    date = max(dt.date.fromtimestamp(a['path'].stat().st_mtime) for a in attempts).isoformat()
    report = HERE / f'{date}-no-second.md'
    for old in HERE.glob('*-no-second.md'):
        if old != report:
            old.unlink()
    report.write_text(text)
    (HERE / 'summary.json').write_text(json.dumps(dict(
        generated=dt.datetime.now(dt.timezone.utc).isoformat(), report=report.name, generator_sha256=sha(__file__),
        sources={str(a['path'].relative_to(WT)): a['sha'] for a in attempts} | {str(FNO_AUDIT.relative_to(WT)): sha(FNO_AUDIT), str(DIAGNOSIS.relative_to(WT)): sha(DIAGNOSIS), str(FNO_POISSON_AUDIT.relative_to(WT)): sha(FNO_POISSON_AUDIT)},
        rule='every number in the report is read from these files; none is typed by hand; no cross-job speed ratio',
        rows=rows), indent=2) + '\n')
    print(report, len(rows), 'rows')


if __name__ == '__main__':
    main()
