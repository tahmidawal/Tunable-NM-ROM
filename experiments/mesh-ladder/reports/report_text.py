"""Prose for the mesh-ladder report. Every number is interpolated from the data."""
from __future__ import annotations


def _flat_sentence(label, summary, key, what):
    flat = summary[f'flatness_{key}']
    meshes = summary['meshes']
    return (f'{what} on {label} runs '
            f'{" -> ".join(f"{v:.3f}" for v in flat["values_ms"])} ms across '
            f'{meshes[0]} to {meshes[-1]} intervals, a factor of '
            f'{flat["ratio_finest_over_coarsest"]:.3f} from the coarsest rung to the finest '
            f'and {flat["maximum_over_minimum"]:.3f} between its own extremes, while the number of '
            f'interior unknowns grows by {flat["unknown_growth"]:.0f}$\\times$.')


def _crossover_sentence(label, summary):
    cross = summary['crossover']
    if not cross['rom_ever_faster']:
        worst = min(r['speedup_fom_over_rom'] for r in cross['rows'])
        best = max(r['speedup_fom_over_rom'] for r in cross['rows'])
        return (f'On {label} there is **no crossover**: the efficient full-order solver is faster '
                f'than the reduced complete device query at every rung of the ladder, by between '
                f'{1 / best:.2f}$\\times$ and {1 / worst:.2f}$\\times$.')
    first = cross['first_mesh_where_rom_is_faster']
    paired = cross['rom_faster_and_both_meet_target']
    tail = ('and the reduced model also meets the accuracy target at '
            + ', '.join(str(m) for m in paired) + ' intervals'
            if paired else
            'but the reduced model does not meet the accuracy target at any rung where it is faster, '
            'so this is a timing observation and not a qualifying speedup')
    return (f'On {label} the reduced complete device query first becomes faster than the efficient '
            f'full-order solver at **{first} intervals**, {tail}.')


def document(summaries, provenance, cost_table, subject_table, efficient_fom,
             setup_total, percent, milliseconds, target, stopping_tables):
    out = []
    add = out.append

    add('# Frozen-checkpoint mesh ladder: cached reduced cost, complete device query, '
        'and the full-order crossover')
    add('')
    add('This report measures what changes, and what does not, when **one frozen checkpoint per PDE** '
        'is transferred across the mesh ladder 64, 128, 256, 512, 1024 intervals per axis, with latent '
        'dimension, bank rank, weak-mode count, quadrature budget, solver policy and stopping rule all '
        'held fixed. Each PDE ran as a single job on a single GPU, so the cost columns within a panel '
        'are directly comparable. **These numbers are development-cohort results and are provisional '
        'for publication**: the final cases remain sealed, one training seed and one checkpoint back '
        'each panel, and the reference margins are empirical refinement evidence rather than continuum '
        'error bounds.')
    add('')
    add('## What was held fixed')
    add('')
    for label, summary in summaries.items():
        add(f'- **{label}** — {provenance[label]["frozen_weight_transfer"]}')
    add('')
    add('New-grid truth is used for evaluation only. Nothing here retrains, reselects or tunes; where '
        'the frozen checkpoint\'s accuracy degrades with the mesh, the degradation is reported as it '
        'stands.')
    add('')
    add('## What was measured, and how the three costs are kept apart')
    add('')
    add('```mermaid')
    add('flowchart LR')
    add('  A["supplied input<br/>(host)"] -->|input transfer| B["input on device"]')
    add('  B --> C["source projection<br/>or initial latent fit"]')
    add('  C --> D["reduced solve<br/>(latent only)"]')
    add('  D --> E["dense decode<br/>to the full field"]')
    add('  E -->|output transfer| F["dense output<br/>(host)"]')
    add('  S["offline per-mesh setup:<br/>bank evaluation, operator rebuild,<br/>'
        'EQ refit or exact preassembly"] -.->|reused by every query| D')
    add('  classDef cached fill:#2a78d6,stroke:#1c5ba3,color:#ffffff;')
    add('  classDef complete fill:#eb6834,stroke:#b84d24,color:#ffffff;')
    add('  classDef setup fill:#1baf7a,stroke:#138059,color:#ffffff;')
    add('  classDef host fill:#f2f1ee,stroke:#d8d7d2,color:#0b0b0b;')
    add('  class D cached; class C,E complete; class S setup; class A,F host;')
    add('```')
    add('')
    add('The blue stage alone is the **cached reduced solve**. The blue and orange stages together, '
        'with the input already resident and the output left resident, are the **complete device '
        'query**. The green box is the **offline per-mesh setup**, charged separately and never '
        'amortised into a query number. Host transfers are timed inside the same invocation as the '
        'complete device query and reported as their own columns.')
    add('')
    add('Each cost is its own completed device computation, obtained by blocking on the result — '
        'never by subtracting one measurement from another. Every timed call is preceded by a GPU '
        'burn-in, the arms are interleaved in a randomized order drawn from a recorded seed inside '
        'one job, and every repetition is retained in the JSON beside this report.')
    add('')
    add('## The common observation grid and its restriction')
    add('')
    first = list(provenance.values())[0]
    checks = first['restriction_checks']
    add(f'Cross-mesh comparison uses a common observation grid of '
        f'{first["restriction"]["common_observation_intervals"]} intervals per axis and the restriction '
        f'operator $R_h$ = **{first["restriction"]["operator"]}**: for a source grid of $N$ intervals '
        f'and a target of $n$ intervals with $N = sn$,')
    add('')
    add('$$(R_h u)_{i,j} \\;=\\; u_{si,\\,sj}, \\qquad 0 \\le i,j \\le n,$$')
    add('')
    add('so no averaging or interpolation enters the comparison and the coarse grid\'s nodes are '
        'exactly a subset of the fine grid\'s nodes. This is well defined only on nested grids, and '
        'the driver refuses a non-nested pair rather than silently flooring the stride.')
    add('')
    add(f'It is validated in-job on a concrete reference field: restricting in one step and '
        f'restricting through every intermediate rung of the ladder give **bitwise identical** results '
        f'(`chained_equals_direct = {checks["chained_equals_direct"]}`, both digests '
        f'`{checks["direct_sha256"][:16]}...`); the target nodes are verified to be source nodes '
        f'(`{checks["nodes_are_source_nodes"]}`); and all four boundary rows and columns survive '
        f'restriction exactly (`{checks["boundary_preserved"]}`). Errors are also reported on each '
        f'requested grid, because a common-grid norm alone omits the finer nodes.')
    add('')
    add('## Reference and its resolution margin')
    add('')
    for label, summary in summaries.items():
        metrics = provenance[label].get('reference_metrics')
        if metrics:
            add(f'**{label}.** The reference is an independently refined full-order solve with spatial '
                f'and temporal refinement controls. Per observation grid, the worst refinement margin '
                f'and whether it resolves the {100 * target:.0f}% target within a tenth of it:')
            add('')
            add('| Observation intervals | Worst margin | Margin budget | Refinement decreases | Resolved |')
            add('| --- | --- | --- | --- | --- |')
            for grid, block in sorted(metrics.items(), key=lambda kv: int(kv[0])):
                decrease = all(r['decrease'] for r in block['rows'])
                add(f'| {grid} | {block["worst_margin"]:.3e} | {block["margin_budget"]:.3e} | '
                    f'{decrease} | {block["resolved"]} |')
            add('')
            add(f'> {block["interpretation"]}.')
        else:
            note = provenance[label].get('reference_note')
            add(f'**{label}.** {note}. The level difference $\\delta$ between the two refined levels '
                f'is charged beside every error as the conservative form $(e + \\delta)/(1 - \\delta)$.')
        add('')

    for label, summary in summaries.items():
        add(f'## {label}: cost against mesh')
        add('')
        add(cost_table(summary))
        add('')
        add(f'`ROM offline setup` is charged once per mesh and is not amortised into any query column. '
            f'`Efficient FOM` is the cheapest full-order arm whose worst error meets the '
            f'{100 * target:.0f}% target on that mesh; where no arm meets it, the cheapest arm is shown '
            f'and labelled. Errors are the worst over every development case and repetition on the '
            f'requested grid.')
        add('')
        add(f'### {label}: every arm')
        add('')
        add(subject_table(summary))
        add('')
        parity = provenance[label]['staged_parity']
        worst_parity = max(p['relative_difference'] for p in parity)
        add(f'The staged solver used for the timing split reproduces the retained selected solver to a '
            f'worst relative difference of {worst_parity:.1e} across the ladder, so the split does not '
            f'change the numerics it measures.')
        add('')
        add(f'### {label}: reduced-solver iteration counts and stopping status, per case per mesh')
        add('')
        add(stopping_tables[label])
        add('')

    add('## The figure')
    add('')
    add('![Cost against mesh size at one frozen checkpoint per PDE]'
        '(2026-09-14-frozen-checkpoint-mesh-ladder.png)')
    add('')
    add('Both panels are log-log, in milliseconds, on one axis. The percentage beside each reduced '
        'complete-query point and each full-order point is the worst physical error on that mesh. '
        'The PDF is `2026-09-14-frozen-checkpoint-mesh-ladder.pdf`.')
    add('')
    add('## Is the cached reduced cost flat?')
    add('')
    for label, summary in summaries.items():
        add('- ' + _flat_sentence(label, summary, 'cached', 'The cached reduced solve'))
    add('')
    for label, summary in summaries.items():
        add('- ' + _flat_sentence(label, summary, 'complete', 'The complete device query'))
    add('')
    add('A fixed-size reduced residual and Jacobian have no explicit full-grid loop, so the reduced '
        'solve carries no mesh dimension once its operators are assembled; the measurement is a check '
        'that no hidden mesh dependence survives, not a surprise. The complete device query is the '
        'quantity that can still grow, because reading an arbitrary dense input and reconstructing an '
        'arbitrary dense output must touch their values even when both stay on the device.')
    add('')
    add('## Where is the crossover?')
    add('')
    for label, summary in summaries.items():
        add('- ' + _crossover_sentence(label, summary))
    add('')
    for label, summary in summaries.items():
        cross = summary['crossover']
        add(f'### {label}: reduced against efficient full-order, per rung')
        add('')
        add('| Intervals | ROM device (ms) | Efficient FOM | FOM device (ms) | FOM/ROM | '
            'ROM meets target | FOM meets target |')
        add('| --- | --- | --- | --- | --- | --- | --- |')
        for row in cross['rows']:
            add(f'| {row["intervals"]} | {row["rom_device_ms"]:.3f} | `{row["fom_subject"]}` | '
                f'{row["fom_device_ms"]:.3f} | {row["speedup_fom_over_rom"]:.3f} | '
                f'{row["rom_meets_target"]} | {row["fom_meets_target"]} |')
        add('')
    add('A ratio above one means the reduced model is faster. A ratio is only a *speedup* where both '
        'the `ROM meets target` and `FOM meets target` columns are true; elsewhere it is a timing '
        'diagnostic, because an unattained accuracy target has no qualifying speedup.')
    add('')
    add('## What this does not establish')
    add('')
    add('- **Not a claim about asymptotics.** Flat cached cost over a finite measured range does not '
        'prove asymptotically constant full-query cost, and the constant-cost claim is with respect to '
        'mesh size only — never to reduced dimension or to a tighter accuracy requirement.')
    add('- **Offline setup stays mesh-dependent.** The bank must be evaluated on the new grid and the '
        'operators rebuilt; those costs are in the table and are not amortised away here.')
    add('- **Development cohorts, one checkpoint, one training seed per PDE.** Final cases remain '
        'sealed and no retraining at any resolution was attempted, by design: this is frozen-weight '
        'transfer, so per-resolution tuning would answer a different question.')
    add('- **Reference margins are empirical.** Space and time refinement differences are development '
        'evidence, not rigorous continuum error bounds.')
    add('- **The comparison is against the named full-order arms only.** A result against these arms '
        'does not transfer to every full-order implementation, coefficient field or geometry — in '
        'particular the direct sine-transform solver exists only because this operator is constant '
        'coefficient on a rectangle.')
    add('- **No cross-job timing.** Every ratio here is within one job on one GPU; times from the two '
        'panels are never divided by each other.')
    add('')
    add('## Provenance')
    add('')
    add('| Panel | Attempt | Job | GPU | Node | Backend | Precision | f64 | Cases | Reps | '
        'Checkpoint SHA-256 | Job seconds |')
    add('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for label, block in provenance.items():
        add(f'| {label} | `{block["attempt"]}` | {block["job_id"]} | {block["gpu"]} | '
            f'{block["node"]} | {block["jax_backend"]} | {block["matmul_precision"]} | '
            f'{block["x64"]} | {block["cases"]} | {block["repetitions"]} | '
            f'`{(block["checkpoint_sha256"] or "")[:16]}...` | {block["elapsed_seconds"]:.1f} |')
    add('')
    add('Regenerate everything in this report, including the figure, with:')
    add('')
    add('```bash')
    add('/home/tahmid/Dev/.venv/bin/python experiments/mesh-ladder/reports/generate_ladder.py')
    add('```')
    add('')
    add('The machine-readable aggregates, including every retained timing repetition, are in '
        '`2026-09-14-frozen-checkpoint-mesh-ladder.json` beside this file.')
    add('')
    add('## Plain-language glossary')
    add('')
    add('- **PDE / full-order model (FOM) / reduced model (ROM):** the equation being solved / solving '
        'it on the full grid / solving a compressed version with far fewer unknowns.')
    add('- **Intervals per axis:** how many cells the unit square is cut into along one direction. A '
        'grid of $N$ intervals has $(N+1)^2$ nodes and $(N-1)^2$ interior unknowns, since the boundary '
        'values are fixed at zero.')
    add('- **Mesh ladder / rung:** the list of grids the same frozen model is run on / one entry in it.')
    add('- **Frozen-weight transfer:** the neural weights are trained once, on one grid, and then used '
        'unchanged on every other grid. The opposite — retraining per grid — is a different experiment '
        'and is not done here.')
    add('- **Checkpoint:** the saved file holding those trained weights. Its SHA-256 is a fingerprint '
        'checked before and after the run so nothing can have changed underneath.')
    add('- **Latent / latent dimension $k$:** the handful of numbers the reduced model actually solves '
        'for. **Bank rank $R$:** how many learned spatial patterns those numbers combine. **Weak modes '
        '$M$:** how many smooth test functions the equation is averaged against.')
    add('- **Cached reduced solve:** the time to solve for the latent numbers alone, with every '
        'precomputed operator already in place. It is the quantity expected not to grow with the mesh.')
    add('- **Complete device query:** the time from the supplied input already sitting in GPU memory to '
        'the full dense answer still sitting in GPU memory — projection or initial fit, the solve, and '
        'the reconstruction of the full field.')
    add('- **Host-to-host:** the same query with the cost of copying the input into GPU memory and the '
        'answer back out added on.')
    add('- **Offline per-mesh setup:** one-time work each new grid needs before any query — evaluating '
        'the learned spatial patterns at the new grid points, rebuilding the discrete operators, and '
        'refitting or preassembling the reduced equations.')
    add('- **Empirical quadrature (EQ) / NNLS:** approximating an expensive sum over all grid points by '
        'a small weighted subset / the nonnegative least-squares fit that chooses those weights.')
    add('- **Quadrature-free preassembly:** the Poisson alternative, where the reduced operator is '
        'formed exactly in advance so no sampling is needed at all.')
    add('- **DST / direct solver:** a discrete sine transform, which diagonalises this particular '
        'operator and so solves it exactly in a couple of transforms. **CG:** conjugate gradients, an '
        'iterative solver that stops at a chosen tolerance.')
    add('- **Newton / BiCGStab / preconditioner:** the nonlinear iteration for the implicit time step / '
        'the iterative linear solver inside it / a transform that makes that linear solve converge fast.')
    add('- **Upwind:** a differencing rule that leans into the direction the solution is travelling, '
        'chosen here by the local sign of the field.')
    add('- **Tolerance:** how small a solver is required to drive its residual before it stops. Looser '
        'tolerances are cheaper and less accurate, which is why every full-order arm states its own.')
    add('- **Stationarity:** a near-zero optimisation gradient at the solver\'s exit. It says the '
        'solver stopped somewhere flat; it does **not** say the answer is physically accurate.')
    add('- **Stall / exit reason:** the solver stopped because it stopped improving or ran out of '
        'budget, rather than because it met its tolerance. Recorded per invocation.')
    add('- **Reference / independently refined reference:** a much finer, separately computed solution '
        'used as truth. **Refinement margin:** how much that truth still moves when refined again — '
        'evidence about its own uncertainty, not a proof.')
    add('- **Common observation grid:** the single coarse grid every method\'s answer is compared on, '
        'so results from different meshes are commensurable. **Restriction (nested-node injection):** '
        'the way a fine answer is put on that grid — by simply keeping the values at the shared nodes.')
    add('- **Requested grid:** the grid the caller actually asked for output on, which is the grid the '
        'method ran on. Errors are reported on both it and the common grid.')
    add('- **Same-grid discrepancy:** how far the reduced answer sits from a fully converged full-order '
        'answer on the *same* grid. It isolates the reduction error from the discretisation error.')
    add('- **Worst / median error:** the largest error over every case, output time and repetition / '
        'the middle one. The worst is what a target has to be met on.')
    add('- **Target:** the accuracy a method must reach for a speed comparison against it to count. '
        f'Here {100 * target:.0f}% relative error.')
    add('- **Development case / final (sealed) case:** a problem instance already used while choosing '
        'methods / one deliberately held back so it can confirm a result later. Only development cases '
        'appear here.')
    add('- **Repetition / pooled median / outlier:** one re-timing of the same work / the middle value '
        'over all case-and-repetition samples / a sample above the upper Tukey fence, counted rather '
        'than discarded.')
    add('- **GPU burn-in:** running throwaway arithmetic before a timed block so the GPU is already at '
        'a steady clock speed, because a cold clock once manufactured a crossover that did not exist.')
    add('- **Crossover:** the mesh at which the reduced model first becomes faster than the full-order '
        'solver it is compared with.')
    add('')
    return '\n'.join(out) + '\n'
