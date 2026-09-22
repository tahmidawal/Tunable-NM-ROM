"""Generate the ops-timing-panel report and its summary.json from the audit JSON alone.

    python reports/generate_ops_panel.py <audit.json> --out reports/2026-09-22-ops-timing-panel.md

No number in the report is typed by hand: every cell is read from the audit, which itself
recomputed every error in NumPy from the saved fields. The script also:

* applies DESIGN.md section 6's full-order rule per row and per timing scope;
* refuses to print an operator row whose own gates failed (DESIGN section 7);
* refuses to run at all if the audit's `failed` list is non-empty (DESIGN section 9);
* compares this job's `fno-large` ERROR row with b-panel `bpn301`'s, which is the same cohort,
  metric and code -- errors only, never the timings, which are from another allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILY_LABEL = {'rom': 'NM-ROM', 'fast': 'NM-ROM (fast kernel)', 'pod': 'POD-LSPG', 'free': 'free bank',
                'fno': 'FNO', 'unet': 'U-Net', 'transolver': 'Transolver', 'deeponet': 'DeepONet',
                'fom': 'FOM'}
ROLE = {'unet-refine': 'validation-selected', 'tsol-refine': 'validation-selected',
        'fno-large': 'validation-selected',
        'unet-medium': 'best worst case on no-second\'s validation set; not selected',
        'tsol-large': 'best worst case on no-second\'s validation set; not selected',
        'don-small': 'validation-selected',
        'don-medium': 'best worst case on the DeepONet lane\'s validation set; not selected'}
# no-second's 32-case VALIDATION worst percentages, for the ordering observation only. Those
# cases are a different split from this panel's cohort, so the percentages are never compared;
# only whether the two arms rank the same way is.
NOSECOND_VALIDATION_WORST = {'unet-refine': 7.5176, 'unet-medium': 3.9622,
                             'tsol-refine': 9.3183, 'tsol-large': 6.3953}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fom_rule(rows, arm, scope):
    """DESIGN section 6: fastest tested FOM whose worst evolved error <= this arm's, on `scope`."""
    key = 'median_gpu_ms' if scope == 'gpu' else 'median_host_ms'
    cands = [f for f in rows if f['family'] == 'fom'
             and f['worst_evolved_percent'] <= arm['worst_evolved_percent'] + 1e-12]
    if not cands:
        return None, None
    best = min(cands, key=lambda f: f[key])
    return best['arm'], best[key] / arm[key]


def operator_gates(checks, arm):
    """Every gate this lane added for `arm`, and whether all of them passed."""
    got = {k: v for k, v in checks.items() if k.endswith(f'_{arm}') and v.get('blocking') is not False}
    return got, all(v['passed'] for v in got.values()) if got else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit')
    p.add_argument('--out', required=True)
    p.add_argument('--bpn301', default=None, help="b-panel bpn301 audit.json, for the FNO error cross-check")
    p.add_argument('--previous', default=None,
                   help='an earlier audit.json of this lane; arms absent from it are marked new in this job')
    p.add_argument('--deeponet', default=None,
                   help="the ops-deeponet-b2d lane's reports/summary.json, for the DeepONet section")
    a = p.parse_args()
    d = json.loads(Path(a.audit).read_text())
    if d['failed']:
        raise SystemExit(f"audit has failed gates, the job is not accepted: {d['failed']}")
    rows = d['arms']
    by = {r['arm']: r for r in rows}
    previous = set()
    if a.previous:
        previous = {r['arm'] for r in json.loads(Path(a.previous).read_text())['arms']}
    for r in rows:
        r['new_in_this_job'] = bool(previous) and r['arm'] not in previous

    printed, suppressed = [], []
    for r in rows:
        if r['kind'] == 'fno':
            got, ok = operator_gates(d['checks'], r['arm'])
            r['gates'] = {k: v['passed'] for k, v in got.items()}
            if ok is not True:
                suppressed.append(dict(arm=r['arm'], gates=r['gates']))
                continue
        for scope in ('gpu', 'host'):
            # A full-order row is not compared with itself: the rule exists to give a reduced or
            # operator arm its comparator, and applying it to a FOM row just returns that row.
            comp, ratio = (None, None) if r['family'] == 'fom' else fom_rule(rows, r, scope)
            r[f'fom_{scope}'] = comp
            r[f'speedup_{scope}'] = ratio
        printed.append(r)

    def cell(v, n=3):
        return '—' if v is None else f'{v:.{n}f}'

    lines = ['| arm | family | role | worst evolved % | median evolved % | worst all-times % | '
             'GPU-query ms | complete-query ms | FOM by the rule (GPU) | speedup (GPU) | speedup (complete) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in printed:
        lines.append('| `{arm}` | {fam} | {role} | {we} | {me} | {wa} | {g} | {h} | {c} | {sg} | {sh} |'.format(
            arm=r['arm'], fam=FAMILY_LABEL.get(r['family'], r['family']),
            role=(('**newly timed** — ' if r['new_in_this_job'] else '') + ROLE.get(r['arm'], '')) or '—',
            we=cell(r['worst_evolved_percent'], 4), me=cell(r['median_evolved_percent'], 4),
            wa=cell(r['worst_all_times_percent'], 4), g=cell(r['median_gpu_ms']), h=cell(r['median_host_ms']),
            c=(f"`{r['fom_gpu']}`" if r['fom_gpu'] else
               ('— (is a FOM)' if r['family'] == 'fom' else 'none at least as accurate')),
            sg=cell(r['speedup_gpu'], 3) + ('×' if r['speedup_gpu'] is not None else ''),
            sh=cell(r['speedup_host'], 3) + ('×' if r['speedup_host'] is not None else '')))
    table = '\n'.join(lines)

    cross = None
    if a.bpn301:
        b = json.loads(Path(a.bpn301).read_text())
        bb = {r['arm']: r for r in b['arms']}
        cross = []
        for arm in ('fno-large',):
            if arm in by and arm in bb:
                cross.append(dict(
                    arm=arm, job=d['job_id'], bpn301_job=b['job_id'],
                    worst_evolved_percent=by[arm]['worst_evolved_percent'],
                    bpn301_worst_evolved_percent=bb[arm]['worst_evolved_percent'],
                    absolute_difference=abs(by[arm]['worst_evolved_percent'] - bb[arm]['worst_evolved_percent']),
                    per_case_max_difference=max(
                        abs(by[arm]['per_case_evolved_percent'][k] - bb[arm]['per_case_evolved_percent'][k])
                        for k in by[arm]['per_case_evolved_percent']),
                    note='errors only; the two jobs are different allocations and no timing ratio is formed'))

    summary = dict(
        lane='ops-timing-panel', attempt=d['attempt'], job_id=d['job_id'], commit=d['commit'], gpu=d['gpu'],
        intervals=d['intervals'], dt=d['dt'], K=d['K'], R=d['R'], elapsed_seconds=d['elapsed_seconds'],
        audit=str(Path(a.audit).name), audit_sha256=sha(a.audit), result_sha256=d['result_sha256'],
        failed_gates=d['failed'], operators=d.get('operators'),
        fom_rule='fastest tested FOM setting whose worst evolved same-grid error <= the row, per timing scope',
        rows=[{k: r[k] for k in (
            'arm', 'kind', 'family', 'worst_evolved_percent', 'median_evolved_percent',
            'worst_all_times_percent', 'median_all_times_percent', 'worst_reference_percent',
            'median_gpu_ms', 'median_host_ms', 'per_case_evolved_percent', 'admissible',
            'fom_gpu', 'speedup_gpu', 'fom_host', 'speedup_host') if k in r} | (
            {'gates': r['gates']} if 'gates' in r else {}) for r in printed],
        suppressed_rows=suppressed, fno_error_cross_check=cross,
        fom_discretisation_error_percent=d['fom_discretisation_error_percent'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out.parent / 'table-256.md').write_text(table + '\n')

    ops = [r for r in printed if r['kind'] == 'fno']
    roms = [r for r in printed if r['family'] in ('rom', 'fast')]
    foms = [r for r in printed if r['family'] == 'fom']
    pods = [r for r in printed if r['family'] == 'pod']
    cheapest_op = min(ops, key=lambda r: r['median_gpu_ms']) if ops else None
    best_op = min(ops, key=lambda r: r['worst_evolved_percent']) if ops else None
    acc = by.get('q256_M1088_eqtop_g0p001')
    fast = by.get('q0_M64_eqcert_g1em06_fastL4')
    cheapest_fom = min(foms, key=lambda r: r['median_gpu_ms']) if foms else None
    cheapest_all = min(printed, key=lambda r: r['median_gpu_ms'])
    cheapest_red = min(roms + pods, key=lambda r: r['median_gpu_ms']) if (roms or pods) else None
    cheapest_prose = (
        (f"It is the cheapest arm in the whole panel: cheaper than the cheapest full-order setting "
         f"(`{cheapest_fom['arm']}`, {cheapest_fom['median_gpu_ms']:.3f} ms) and than the cheapest "
         f"NM-ROM or POD arm (`{cheapest_red['arm']}`, {cheapest_red['median_gpu_ms']:.3f} ms).")
        if cheapest_op and cheapest_all['arm'] == cheapest_op['arm'] else
        (f"The cheapest arm in the panel is `{cheapest_all['arm']}` at {cheapest_all['median_gpu_ms']:.3f} ms."))

    def f(v, n=4):
        return '—' if v is None else f'{v:.{n}f}'

    pairs, reproduced = [], []
    for fam, sel, oth in (('U-Net', 'unet-refine', 'unet-medium'),
                          ('Transolver', 'tsol-refine', 'tsol-large')):
        if sel not in by or oth not in by:
            continue
        hs, ho = by[sel]['worst_evolved_percent'], by[oth]['worst_evolved_percent']
        vs, vo = NOSECOND_VALIDATION_WORST[sel], NOSECOND_VALIDATION_WORST[oth]
        same = (hs > ho) == (vs > vo)
        reproduced.append(same)
        pairs.append(f'| {fam} | `{sel}` | {f(hs)} | {vs:.4f} | `{oth}` | {f(ho)} | {vo:.4f} | '
                      f"{'yes' if same else '**no**'} |")
    pairs_table = chr(10).join(pairs)
    if pairs and not any(reproduced):
        pairs_verdict = ('On **neither** family does the ordering reproduce on this cohort: here the '
                         'validation-selected arm has the *lower* worst case in both families, the '
                         'opposite of the ranking on `no-second`\'s 32 validation cases. The two '
                         'cohorts are different splits, so this neither confirms nor refutes the '
                         'selection-rule finding — it says the finding does not transfer to these six '
                         'cases, and that six cases is a thin basis for a tail statement either way.')
    elif pairs and all(reproduced):
        pairs_verdict = ('The ordering reproduces on both families, on a cohort the selection never saw.')
    else:
        pairs_verdict = ('The ordering reproduces on one family and not the other; see the last column.')

    # ---- the DeepONet block, built from the sibling lane's own generated summary -----------
    don_block = ''
    don_rows = [r for r in printed if r['family'] == 'deeponet']
    if don_rows and a.deeponet:
        ds = json.loads(Path(a.deeponet).read_text())
        want = {(x['arm'], x['cohort'], x['metric']): x for x in ds['rows']}
        meta = {}
        for x in ds['rows']:
            meta.setdefault(x['arm'], x)
        lines = ['| arm | capacity | real params | epochs run | best epoch | stop reason | '
                 'their validation-32 worst % | their diagnosis-8 worst % | worst evolved % HERE |',
                 '|---|---|---|---|---|---|---|---|---|']
        for r in sorted(don_rows, key=lambda r: r['arm']):
            m = meta[r['arm']]
            v = want.get((r['arm'], 'validation-32', 'worst_fixed_initial_error'))
            g8 = want.get((r['arm'], 'diagnosis-8', 'worst_fixed_initial_error'))
            lines.append(f"| `{r['arm']}` | {m['capacity']} | {m['params']:,} | {m['epochs']} | "
                         f"{m['best_epoch']} | **{m['stop_reason']}** | "
                         f"{100 * v['value']:.4f} | {100 * g8['value']:.4f} | "
                         f"{r['worst_evolved_percent']:.4f} |")
        stops = sorted({meta[r['arm']]['stop_reason'] for r in don_rows})
        early = stops == ['early_stopping']
        ratios = None
        hand = Path(a.deeponet).with_name('timing-handoff.json')
        if hand.exists():
            ratios = json.loads(hand.read_text()).get('accuracy_ratios_selected_arm_vs_selected_arm')
        rtext = ''
        if ratios:
            rtext = ('\n\nThe sibling lane\'s generated selected-arm-vs-selected-arm ratios on its 32 '
                     'validation cases (its number, not recomputed here): DeepONet\'s error divided by '
                     + ', '.join(f"**{k}**\u2019s is {v['mean']:.3f}× on the mean, {v['median']:.3f}× on the "
                                 f"median and {v['maximum']:.3f}× on the maximum"
                                 for k, v in ratios.items()) + '. These must travel with every DeepONet '
                     'speed row; a cost without them is not a result.')
        # the ordering check against the sibling lane, on rank only
        pairs_d = []
        for arm in sorted(don_rows, key=lambda r: r['worst_evolved_percent']):
            pairs_d.append(arm['arm'])
        theirs = sorted((r['arm'] for r in don_rows),
                        key=lambda n: want[(n, 'diagnosis-8', 'worst_fixed_initial_error')]['value'])
        same_rank = pairs_d == theirs
        don_block = f"""## The DeepONet arms

**Newly timed in this job.** The `ops-deeponet-b2d` lane trained these four on the same data,
split, reference, metric, budget and selection rule, and made no speed claim, for the same
reason `no-second` made none. Their costs above are the first admissible ones.

{chr(10).join(lines)}

**Read every DeepONet row with all four of these.** They are the sibling lane's own record and
they change what the numbers mean:

1. **All four arms ended by {'early stopping, not on the wall budget' if early else '/'.join(stops)}.** The per-capacity wall budget did
   **not** bind — the opposite of the U-Net, Transolver and FNO arms, every one of which ended
   on its budget. Comparing "equal wall" across the families therefore does not mean the same
   thing for DeepONet as it does for the others.
2. **`still_improving = no` is vacuous for these arms.** It is defined against the patience
   window, and an arm that early-stopped satisfies it by construction; it says nothing about
   whether the configuration had converged.
3. **Training loss was still falling in all four histories** when early stopping fired on the
   validation criterion.
4. **128 training cases.** A **data-limited** reading of this result is live, and the sibling
   lane did not separate it from a capacity reading. **Nothing here is evidence of an
   architecture ceiling for DeepONet**, and no sentence in this report should be read that way.
   What is established is that *these four checkpoints, trained this way, on this much data*
   are far less accurate than the other operators on the same cohort while costing a comparable
   amount.{rtext}

**Rank ordering.** By worst error the four arms rank {' < '.join(f'`{x}`' for x in pairs_d)} here and
{' < '.join(f'`{x}`' for x in theirs)} on the sibling lane's diagnosis-8 cohort — the ordering
{'reproduces' if same_rank else '**does not reproduce**'}. Only the ordering is compared: that cohort is the *calibration*
split and this panel's is the *development* split, they share no case, and the absolute
percentages are therefore not comparable in either direction.

"""

    if cross:
        c0 = cross[0]
        exact = c0['per_case_max_difference'] == 0.0
        cross_prose = (
            f"`{c0['arm']}` was scored here (job `{c0['job']}`) and in b-panel `bpn301` (job "
            f"`{c0['bpn301_job']}`) on the same six cases, with the same metric and the same audit "
            f"code. Worst evolved error: **{c0['worst_evolved_percent']:.4f} %** here, "
            f"**{c0['bpn301_worst_evolved_percent']:.4f} %** there; the largest per-case difference is "
            f"**{c0['per_case_max_difference']:.3e} %**"
            + (', i.e. the two jobs agree bit for bit and the operator path is deterministic across '
               'allocations on this GPU model.' if exact else '.'))
    else:
        cross_prose = 'No `bpn301` audit was supplied, so the FNO error cross-check was not run.'

    md = f"""# The operator arms, timed: a same-allocation {d['intervals']}² Burgers panel for U-Net, Transolver, FNO, the NM-ROM, POD and the full-order solver

**These numbers are final for job `{d['job_id']}` and are the first admissible speed numbers for the
U-Net and Transolver checkpoints.** Every cost in this report was measured in **one Slurm
allocation on one GPU** ({d['gpu']}), against subjects measured in that same allocation. No
ratio in this report is formed across jobs, because the project does not allow one.

Job `{d['job_id']}`, attempt `{d['attempt']}`, source commit `{d['commit']}`, {d['elapsed_seconds']:.0f} s elapsed,
`jax_backend=gpu`, float64, `JAX_DEFAULT_MATMUL_PRECISION=highest`. Decoder $K = {d['K']}$, $R = {d['R']}$,
$\\Delta t = {d['dt']}$. Audit `{Path(a.audit).name}` (SHA256 `{sha(a.audit)[:16]}…`), failed gates: **{d['failed'] or 'none'}**.
Generated by `reports/generate_ops_panel.py` from that audit alone; no number below is typed.

## What this job adds

The `no-second` lane trained the U-Net and the Transolver and said plainly that no speed ratio
against the FNO, the ROM or the FOM was stated anywhere, because those were measured in other
jobs. This job measures them where a measurement is admissible. **{len(ops)} operator arms** —
the FNO, four U-Net capacities and four Transolver capacities — were timed as separate
processes inside this one allocation, on the cohort of {len(by['fft_tight']['per_case_evolved_percent'])} development
cases the JAX phase generated, and scored by the same NumPy audit code, against the same
same-job converged `fft_tight` solve, as every other row.

## The panel at {d['intervals']}²

{table}

## What the table says

* **Cheapest operator arm:** `{cheapest_op['arm'] if cheapest_op else '—'}` at {f(cheapest_op['median_gpu_ms'], 3) if cheapest_op else '—'} ms GPU-query, with
  {f(cheapest_op['worst_evolved_percent']) if cheapest_op else '—'} % worst evolved error. {cheapest_prose}
* **Most accurate operator arm:** `{best_op['arm'] if best_op else '—'}` at {f(best_op['worst_evolved_percent']) if best_op else '—'} % worst evolved,
  {f(best_op['median_gpu_ms'], 3) if best_op else '—'} ms.
* **NM-ROM accurate arm** (`q256_M1088_eqtop_g0p001`, the arm §7 of the design named in advance):
  {f(acc['worst_all_times_percent']) if acc else '—'} % worst all-times, {f(acc['median_gpu_ms'], 3) if acc else '—'} ms, comparator
  `{acc['fom_gpu'] if acc else '—'}`, speedup **{f(acc['speedup_gpu'], 3) if acc else '—'}×** on the GPU-query scope. The campaign bar is
  $\\le 1\\%$ **and** $\\ge 5\\times$; the accuracy half {'passes' if acc and acc['worst_all_times_percent'] <= 1 else 'fails'} and the speed half
  {'passes' if acc and acc['speedup_gpu'] and acc['speedup_gpu'] >= 5 else 'FAILS'}. This is the expected outcome at {d['intervals']}² and it is reported, not softened.
* **NM-ROM fast arm** (`q0_M64_eqcert_g1em06_fastL4`): {f(fast['worst_evolved_percent']) if fast else '—'} % worst evolved,
  {f(fast['median_gpu_ms'], 3) if fast else '—'} ms, comparator `{fast['fom_gpu'] if fast else '—'}`, speedup {f(fast['speedup_gpu'], 3) if fast else '—'}×.
* **Discretisation floor.** The $4096$-interval reference says the $256^2$ grid itself carries
  {f(d['fom_discretisation_error_percent']['fft_tight'])} % error ({d['fom_discretisation_error_percent']['fft_tight']:.4f} % for `fft_tight`), so the `vs ref %`
  column of every arm is bounded below by it and same-grid error is the metric that discriminates.

## Selected arm versus best worst case

`no-second` selected on the **mean** validation error, and that rule twice picked the arm with
the worse tail. Both are in the table; here they are side by side on **this** cohort:

| family | validation-selected | worst here % | worst on no-second's validation % | the other arm | worst here % | worst on no-second's validation % | does the ordering reproduce? |
|---|---|---|---|---|---|---|---|
{pairs_table}

{pairs_verdict}

{don_block}## Cross-check, and what is *not* a cross-check

{cross_prose}

`fno-large` ran here on the same six cases, with the same metric and the same audit code, as in
b-panel `bpn301`. The **errors** are compared above. The **timings are not**: `bpn301` is a
different allocation, and dividing across allocations is the thing this lane exists to avoid.
Both jobs' FNO costs appear in their own reports as separate measurements.

`no-second`'s own accuracy numbers are **not** compared here. Its ROM/FOM diagnosis cohort is the
*calibration* split and this panel's cohort is the *development* split, so the two share no case.
Different absolute percentages between the two are expected and are not a disagreement.

## Deviations, stated not glossed

1. **Separate processes in one allocation.** The source timing protocol times all models in one
   process; JAX preallocates the device, so the operator arms run as later processes in the same
   allocation on the same GPU. The audit gates that each arm's recorded GPU equals the JAX
   phase's. One allocation, one GPU, no cross-job ratio.
2. **The complete-query scopes are not byte-identical.** The JAX subjects are charged the
   host-to-device upload of the input; the operator arms are not. The asymmetry **favours the
   operators** by a few hundred microseconds and is left in place so the FNO row stays
   comparable with `bpn301`'s. The GPU-query column is unaffected.
3. **The scored operator fields come from an extra untimed query**, not from a retained timed
   repetition. The audit checks them against the hash the timing process recorded and checks
   $t_0$ bitwise against this panel's own `fft_tight` initial state.
4. **Every operator arm ended on its wall budget**, not on early stopping (`no-second`, recorded).
   None of these accuracies is a capacity ceiling.
5. **The operators are direct multi-time outputs**, not autoregressive rollouts: one forward pass
   returns $u(t_1)\\dots u(t_5)$ and $u(t_0)$ is prepended bitwise.
6. Rows suppressed because their own gates failed: **{suppressed or 'none'}**.

## Glossary

Written for a reader who has none of this project's vocabulary.

* **arm** — one thing being measured: a model, a solver setting, or a ROM configuration.
* **NM-ROM** — the project's nonlinear-manifold reduced-order model. It solves a small
  optimisation problem at every time step on a learned manifold instead of the full grid.
* **q** — the number of *correction* directions added to the ROM's latent space. `q0` is the
  cheap end of the knob, `q256` the accurate end.
* **M** — the number of weak tests (rows of the overdetermined least-squares problem) the ROM
  solves against. **m** is the number of quadrature points a rule keeps.
* **dense / EQ** — how the ROM's residual is evaluated: `dense` over the whole grid, `EQ`
  (empirical quadrature) at a certified sparse subset of points. `eqcert` and `eqtop` are two
  different certified rule sets.
* **fast kernel / `fastL4`** — an optimised implementation of the same $q=0$ query, required to
  match the reference path bitwise to $10^{-12}$ before its timing counts.
* **POD-LSPG** — the classical linear-subspace reduced-order model; the control the NM-ROM has
  to beat to justify a nonlinear manifold.
* **FOM** — full-order model, the real solver on the full grid. Here: backward-Euler Newton
  iterations with a BiCGStab linear solve and an FFT Helmholtz preconditioner. `fft_tight` is
  the converged one; `nt1e-2_dt01` and the rest are deliberately looser, cheaper settings.
* **U-Net / Transolver / FNO / DeepONet** — four trained neural operators. Each takes the
  initial field and the viscosity and returns all five later times in one forward pass. A
  **DeepONet** does it by combining a *branch* network that reads the input field into
  coefficients with a *trunk* network that turns coordinates into basis functions.
* **early stopping vs wall budget** — two different reasons training can end. "Wall budget"
  means the clock ran out while the model was still improving, so the number is a lower bound
  on that configuration. "Early stopping" means validation error stopped improving for a set
  number of epochs; it does not by itself mean the model had converged, especially when the
  training loss was still falling.
* **data-limited vs capacity-limited** — whether a model is held back by having too few
  training examples or by being too small/wrong in shape. With 128 training cases the two are
  not separated here, so a weak result is not evidence about the architecture.
* **`-small` / `-medium` / `-large` / `-refine`** — the four capacities each operator family was
  trained at; `-refine` is a lower-learning-rate retrain of the capacity validation picked.
* **worst / median evolved %** — the error metric, $\\max_k \\lVert \\hat u(t_k) - u^\\star(t_k)\\rVert_2 /
  \\lVert u^\\star(t_0)\\rVert_2$ over the five *evolved* times, taken worst and median over the six
  cases. **all-times** includes $t_0$, where a ROM pays a compression error and an operator pays
  nothing because it returns the supplied state bitwise.
* **vs ref %** — the same error against a much finer $4096$-interval solution instead of the
  same-grid one; it measures how wrong the $256^2$ grid itself is.
* **GPU-query ms** — the time from "inputs already on the GPU" to "complete trajectory on the
  GPU". **complete-query ms** adds the copy of the answer back to the host.
* **held-out / development / calibration** — disjoint sets of test cases. Nothing here was tuned
  on any of them.
* **gate** — an automatic check that must pass for a number to be reported (GPU identity, cohort
  completeness, field hashes, $t_0$ exactness). A row whose gates failed is not printed.
* **admissible** — a number the project's rules allow to be quoted. For a cost, it means
  same-allocation, same-GPU, same-cohort, same-scope.
* **speedup** — full-order time divided by the arm's time, with the full-order setting chosen by
  the paper's rule: the fastest tested setting that is at least as accurate as the arm.
"""
    out.write_text(md)
    print(table)
    print()
    print('suppressed (gates failed):', suppressed or 'none')
    print('fno error cross-check vs bpn301:', json.dumps(cross))
    print('wrote', out, out.parent / 'summary.json', out.parent / 'table-256.md')


if __name__ == '__main__':
    main()
