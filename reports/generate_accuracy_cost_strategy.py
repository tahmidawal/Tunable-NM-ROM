"""Render the proposed accuracy/cost strategy from archived development evidence.

No numerical jobs are launched. Measured numbers are loaded from hash-checked
artifacts; proposed gates and algebraic dimensions are explicitly distinguished.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-09-09-accuracy-and-cost-improvement-strategy"
CATALOG = ROOT / "reports/2026-09-07-multiresolution-pilots.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def main():
    catalog_bytes = CATALOG.read_bytes()
    catalog = json.loads(catalog_bytes)
    # Fail on archive drift, including evidence supporting the opening table.
    for key, entry in catalog["artifacts"].items():
        actual = digest((ROOT / entry["path"]).read_bytes())
        if actual != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {key}")

    used = {}

    def read(key):
        entry = catalog["artifacts"][key]
        used[key] = entry
        return json.loads((ROOT / entry["path"]).read_bytes())

    latest = catalog["latest_comparisons"]
    by_pde = {row["pde"]: row for row in latest}
    baseline_table = table(
        ["PDE", "Intervals / cases", "Worst relative error (%)", "ROM / FOM query (ms)", "Norm / FOM scope"],
        [[r["pde"], f'{r["intervals"]} / {r["cases"]}',
          f'{100*r["worst_error"]:.4f}', f'{r["rom_ms"]:.3f} / {r["fom_ms"]:.3f}',
          r["metric"] + "; " + r["baseline"]] for r in latest],
    )

    heat = read("heat_transfer")
    heat_steps = round(heat["config"]["times"][-1] / heat["settings"]["dt"])
    heat_outputs = len(heat["config"]["times"]) - 1

    poisson = read("poisson_training_summary")
    pn = by_pde["Poisson"]["intervals"]
    rep, = [r for r in poisson["representation"]
            if r["intervals"] == pn and r["cohort"] == "all_development"
            and r["model"] == "original_relative"]
    retained_modes, = {r["retained_modes"] for r in poisson["setup"] if r["intervals"] == pn}
    architecture = poisson["config"]["architecture"]
    poisson_table = table(["Diagnostic", "Worst relative error (%)"], [
        ["Unrestricted bank projection, full same-grid norm", f'{100*rep["bank_same_grid_max"]:.8f}'],
        ["Bank projection, common-reference physical diagnostic", f'{100*rep["bank_physical_max"]:.8f}'],
        ["Best recorded stationary head fit, common-reference norm", f'{100*rep["head_physical_max"]:.8f}'],
        ["Deployed weak solve, common-reference norm", f'{100*by_pde["Poisson"]["worst_error"]:.8f}'],
    ])

    burgers = read("burgers_steps_summary")
    bn = by_pde["Burgers"]["intervals"]
    # These are the pre-existing finer/current step controls, not a new search.
    fine, = [r for r in burgers["configurations"]
             if r["method"] == "rom" and r["intervals"] == bn and r["dt"] == 0.005]
    current, = [r for r in burgers["configurations"]
                if r["name"] == by_pde["Burgers"]["setting"]]
    required_saving = 1 - current["ms"] / fine["ms"]
    burgers_table = table(
        ["Step", "Worst full-grid error (%)", "Complete query (ms)", "Evolution / first-step budget exits"],
        [[r["dt"], f'{100*r["dense"]["worst"]:.4f}', f'{r["ms"]:.3f}',
          f'{r["evolution_budget_stops"]} / {r["first_step_budget_stops"]}']
         for r in (fine, current)],
    )

    wave = read("wave_heads")
    wn = max(wave["config"]["meshes"])
    method = f'new_mlp32_seed{wave["config"]["training"]["optimizer_seeds"][0]}'
    step = max(wave["config"]["nonlinear_dts"])
    rows = [r for r in wave["invocations"] if r["intervals"] == wn
            and r["measurement_role"] == "timed_comparison"
            and r["comparison_eligible"] and r["completed"]]
    wave_overhead = []
    for boundary, label, fom, fom_setting in (
        ("dirichlet", "Reflective", "dst", 0.0),
        ("absorbing", "Absorbing", "rk4", max(wave["config"]["fom_cfls"])),
    ):
        groups = defaultdict(list)
        controls = defaultdict(list)
        for r in rows:
            if r["boundary"] != boundary:
                continue
            if r["method"] == method and r["setting"] == step:
                groups[r["case"]].append(r)
            if r["method"] == fom and r["setting"] == fom_setting:
                controls[r["case"]].append(r)
        assert set(groups) == set(controls) and groups
        for case, calls in sorted(groups.items()):
            baseline = controls[case]
            assert len(calls) == len(baseline) == wave["config"]["repetitions"]
            remaining = [r["seconds"]["complete_query"] - r["seconds"]["evolution"] for r in calls]
            assert min(remaining) > 0
            wave_overhead.append({
                "boundary": label, "case": case, "intervals": wn,
                "method": method, "dt": step,
                "non_evolution_seconds": remaining,
                "fom_complete_seconds": [r["seconds"]["complete_query"] for r in baseline],
                "non_evolution_median_ms": 1000 * median(remaining),
                "fom_median_ms": 1000 * median(r["seconds"]["complete_query"] for r in baseline),
                "rom_invocations": [r["invocation_id"] for r in calls],
                "fom_invocations": [r["invocation_id"] for r in baseline],
            })
    wave_table = table(
        ["Boundary", "Case", "ROM query minus evolution, median (ms)", "Same-job FOM query, median (ms)"],
        [[r["boundary"], r["case"], f'{r["non_evolution_median_ms"]:.3f}', f'{r["fom_median_ms"]:.3f}']
         for r in wave_overhead],
    )

    replacements = {
        "BASELINE_TABLE": baseline_table,
        "HEAT_STEPS": str(heat_steps), "HEAT_OUTPUTS": str(heat_outputs),
        "POISSON_TABLE": poisson_table,
        "POISSON_N": str(pn), "POISSON_CASES": str(rep["case_count"]),
        "POISSON_R": str(architecture["features"]),
        "POISSON_K": str(architecture["latent_dimension"]),
        "POISSON_M": str(retained_modes),
        "BURGERS_TABLE": burgers_table, "BURGERS_N": str(bn),
        "REQUIRED_SAVING": f"{100*required_saving:.2f}",
        "WAVE_TABLE": wave_table, "WAVE_N": str(wn),
    }
    rendered = TEMPLATE
    for key, value in replacements.items():
        rendered = rendered.replace("@@" + key + "@@", value)
    assert "@@" not in rendered
    (ROOT / "reports" / (STEM + ".md")).write_text(rendered)
    evidence = {
        "status": "Proposed strategy; no new GPU measurements. Derived diagnostics from provisional archived development results.",
        "catalog": {"path": str(CATALOG.relative_to(ROOT)), "sha256": digest(catalog_bytes)},
        "verified_catalog_artifacts": len(catalog["artifacts"]),
        "direct_sources": used,
        "latest_comparisons": latest,
        "poisson_representation": rep,
        "heat_current_steps": heat_steps, "heat_later_outputs": heat_outputs,
        "burgers_fine_step_saving_to_match_current_cost": required_saving,
        "wave_zero_evolution_counterfactual": wave_overhead,
        "proposed_gates_not_measurements": {"bank_or_initial_worst_error": 0.03, "complete_error": 0.05},
    }
    (ROOT / "reports" / (STEM + ".json")).write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Rendered {STEM}; verified {len(catalog['artifacts'])} archived artifact hashes.")


TEMPLATE = r"""# Improving separable NM-ROM accuracy and complete-query cost

This is a proposed experiment strategy based on three parallel code-and-evidence reviews. All measured numbers below remain provisional development evidence; this review launched no numerical jobs, and none of the proposed gains has been measured.

The strongest near-term combination is better heat initialization with direct weak-mode evolution. Poisson offers a separate fixed-rank representation improvement and an exact reduction of repeated solver work. Burgers needs cheaper corrections and controlled startup. Waves need cheaper initialization as well as evolution, with a specific absorbing-boundary defect tested separately.

## Evidence that determines the priorities

@@BASELINE_TABLE@@

These are the latest recorded representative settings, not independently confirmed paper results. Error norms differ across PDEs. Heat, Burgers and Poisson selections include explicitly empirical reference allowances; their displayed errors are the raw physical errors. Wave comparisons are same-grid controls without a coarse-FOM envelope. Absorbing waves also have large late error relative to the remaining field, which the initial-normalized column does not show. Full definitions, all configurations, repetitions and outlier counts are in the [generated results report](2026-09-07-multiresolution-pilots.md).

Query cost is the median over cases of each case's repetition median, charging input processing, initialization, solving and full requested host outputs. None of these primary nonlinear configurations beats its efficient FOM. Cross-job wall times are not hardware-normalized rankings.

## Heat: change evolution and initial reconstruction in separate controls

**Speed intervention.** The verified discrete sine tests are Laplacian eigenmodes. Their coefficients can evolve as

$$q_j(t)=\exp(-\nu\lambda_j t)q_j(0).$$

The current method performs @@HEAT_STEPS@@ sequential Crank–Nicolson steps with a nonlinear fit after each step. Instead, propagate the coefficients directly and fit the decoder only at the @@HEAT_OUTPUTS@@ requested later output times:

$$z(t)=\arg\min_z\|A h(z)-q(t)\|_2^2.$$

Here $h$ is the nonlinear coefficient head and $A$ maps its output into the tested weak modes. This removes temporal discretization in the retained modal dynamics and repeated projection feedback. It changes the reduced evolution method; it is not an algebraically equivalent acceleration of the current manifold trajectory. Truncation, initial-fit and reconstruction errors remain.

First freeze the head and initial fit, and use $q(0)=A h(z_0)$ in every arm. Compare current stepping, direct modal evolution with sequential output fits, and direct modal evolution with independent batched fits. This isolates evolution from improved knowledge of the initial field. Retain the actual fitted initial output. Screen at the smallest and largest already approved meshes using the existing development cohort.

**Accuracy intervention.** The current worst error is dominated by initial reconstruction. Measure full-bank projection and stationary head-fit errors on the complete current cohort before increasing capacity. The earlier favorable bank diagnostic used a smaller cohort and does not establish a floor for the fresh cases.

Cross a frozen/refined head with existing-library/learned initialization. A starting map should consume sampled supplied-field features and be judged by decoded field error and the corrected endpoint. Matching arbitrary latent-code labels alone is insufficient. Preserve the PDE correction and fallback; charge their actual work.

The current initializer still performs a dense full-input projection. Validate a deterministic positive-quadrature field fit against full-field fitting as its own control before combining it with a learned initializer. This is field reconstruction, not pointwise PDE collocation. Refit the quadrature when its bank, test space or mesh changes.

**Proposed gates, not predictions:** lower complete-query cost with no loss of the current empirical 5% qualification for the evolution change; worst initial error below 3% on both screening meshes for the representation change. The refined head must also retain the complete-rollout empirical 5% qualification and avoid a complete-query cost regression before promotion or combination. Require the existing stationarity checks and no new failures. Only combine successful factors. Better modal accuracy without better returned fields is not a pass.

Source: [current heat evolution and initialization](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/heat_core.py), [modal operator construction](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/run_pilot.py).

## Poisson: improve the bank, then remove equivalent residual work

At @@POISSON_N@@ intervals on all @@POISSON_CASES@@ development cases, the selected `original_relative` checkpoint gives:

@@POISSON_TABLE@@

These provisional diagnostics explain why more optimizer effort alone is unlikely to fix accuracy. The unrestricted full same-grid bank projection is a lower bound within that bank and norm. Its common-reference counterpart is a diagnostic, not a rigorous continuum bound. The best recorded stationary head fit is not a proof of the global nonlinear optimum. Its near agreement with the deployed solve nevertheless points toward representation as the main remaining accuracy limitation.

**Accuracy intervention.** Keep the bank rank at @@POISSON_R@@ and the head dimension at @@POISSON_K@@ first. Measure an offline normalized-training-snapshot SVD span at this rank. If it represents the data much better, compare current joint training with staged training: fit the neural spatial bank using unrestricted training coefficients, freeze it, then fit the same nonlinear head. A wider spatial network is a subsequent separate control if it cannot approximate the target span.

Spatial-network evaluation is cached during setup. Better bank training at fixed output rank need not increase steady-query matrix sizes, although it can change conditioning and nonlinear iteration counts. Report setup and training costs separately. SVD optimizes an average, not a worst-case objective; fresh-case and head-compression errors remain necessary gates. If the SVD span misses the worst-case gate, test training-only reweighting or a worst-case-aware bank objective before increasing rank. Failure of an average-optimal span does not prove that no span of the same rank can meet the gate. Increasing coordinate-network width alone cannot repair an inadequate fixed target span. Charge additional output/operator cost if a larger bank is subsequently tested.

**Speed intervention.** Compress the repeated weak residual exactly. For $r(z)=B h(z)-f_m$, compute a thin QR factorization $B=QR$ offline, then

$$b=Q^T f_m,\qquad c_\perp=\|f_m-Qb\|_2^2,$$
$$\|r(z)\|_2^2=\|R h(z)-b\|_2^2+c_\perp.$$

The nonlinear residual row count becomes @@POISSON_R@@ instead of @@POISSON_M@@ while retaining the information from every original weak test. Compute the orthogonal component directly; do not subtract nearly equal squared norms. Keep $c_\perp$ in every residual-based stopping and normalization rule. Check the actual implementation's damping and stationarity definitions before claiming parity.

First compare frozen-checkpoint endpoints, stopping decisions and complete-query timings. Only afterward cross this algebra with a learned field-based starting map, benchmarked against the current nearest-training-code initializer. A faster predictor is useful only when feature construction, correction and fallback together become cheaper.

**Proposed gates:** bank worst projection below 3% on both meshes, followed by corrected-query empirical-adjusted worst error below 5% without a query-cost regression. Pure QR compression must preserve f64 numerical results and improve paired complete-query medians without new failures. These are separate accuracy and speed tests.

If nonlinear cost remains substantial, consider a partially linear decoder $u=G[P a+N(q)]$. For fixed $q$, eliminate the linear coordinates $a$ by an offline-factorized least-squares solve, and optimize only $q$. This applies established [variable projection](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems). Keep total coordinate count fixed and constrain the nonlinear component to a complementary coefficient space. Compare the same partially linear decoder with joint solving and exact elimination to isolate solver algebra. Require adequate stationary reconstruction before timing. This remains an NM-ROM only when the nonlinear component and online weak correction remain active; a fully linear control is labeled separately.

Source: [bank training](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/training_core.py), [weak assembly](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/core.py), [current small solver](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/kernel_solver.py).

## Burgers: use cheaper corrections to recover finer-step accuracy

The same-job finer/current-step controls at @@BURGERS_N@@ intervals are:

@@BURGERS_TABLE@@

These remain provisional development results. Budget-exit counts include every timing repetition; empirical physical qualification is not a stationarity certificate. A @@REQUIRED_SAVING@@% reduction in the finer-step complete-query median would recover its better accuracy at the present coarser-step cost. That is a derived screening target, not a predicted acceleration or an FOM win.

The current solver already extrapolates latent states when useful, caches the Jacobian on rejected trials and precomputes exact weak mass/diffusion. Test reuse across accepted corrections next. Evaluate actual residuals throughout; refresh on poor progress, relevant upwind-sign changes and prospective convergence. Update the damped solve whenever damping changes, and verify termination with a fresh Jacobian. Preserve the FOM-exact upwind advection.

Compare ordinary and refreshed-Jacobian solvers at the existing finer/current steps with the same checkpoint and Gauss initializer. Independent derivative checks and unchanged physical eligibility precede timing. Retain all early stops and failures.

Only if correction work remains dominant, compare startup-controlled backward Euler and BDF2 with identical startup history. Build the coarse-step history from saved equally spaced states; do not apply constant-step BDF2 coefficients directly to unequal history spacings. This separates startup quality from temporal order. The previous larger-step study mixed time error with incomplete solves and does not establish the accuracy of a converged higher-order method.

Neither solver optimization removes the finest-grid initial representation error. Tighter accuracy targets also need bank/head diagnostics with finer-grid and boundary-aware training. An initial encoder alone cannot remove the dominant evolution and full-output costs.

Source: [Burgers solver and weak residual](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/engines.py), [full timestep findings](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md).

## Waves: initialization is part of the bottleneck

For the latest first-seed larger head at @@WAVE_N@@ intervals, subtracting evolution from each recorded complete-query repetition gives:

@@WAVE_TABLE@@

This is a hypothetical zero-evolution diagnostic derived from provisional timings, not a newly measured implementation. The remaining cost is summarized after subtraction per repetition; it is not a sum of independently selected phase medians. Even this hypothetical reflective result loses to its paired same-grid FOM. Faster evolution alone cannot establish a reflective crossover while the rest of the current query stays fixed.

**First speed controls:** share decoder value/velocity/Jacobian/curvature work, compute singular values from the already available triangular QR factor, and compile the complete cold-fit wrapper. Test each separately with frozen checkpoints. Preserve derivative values, rank thresholds, stopping decisions and trajectories, with fallback near rank thresholds. Compiler optimization may already remove duplicate work; gains must be measured on the complete query.

**Absorbing accuracy control:** execute the existing factorial design with unchanged bank, an exact constant direction, initial conserved-moment correction, and both changes. Begin with the linear full-bank control to isolate the mechanism. Conserving the moment must also improve late physical field error without harming initialization or energy behavior. An invariant corrected without better fields is a negative result. See the [existing moment-control design](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/NEXT-MOMENT-CONTROL.md).

**Higher-payoff method candidate:** reduced linear propagation over larger blocks followed by nonlinear weak projection. It could replace many expensive geometry evaluations with fewer corrections, but changes the reduced method. Charge parameter-dependent propagator construction and all projections; check phase, dissipation, physical velocity consistency and full-query cost against both current nonlinear and affine controls.

A protected linear anchor with a nonlinear correction is a possible architecture control, not a new proven solution. The existing [Burgers-3D protected-anchor experiment](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/B3D-ARCH-NOTES.md) improved conditioning while worsening reconstruction and tangent accuracy. Screen representation before wave rollouts. Defer a generic joint displacement–velocity MLP: eliminating explicit curvature can increase Jacobian/factorization sizes and does not automatically preserve kinematics or energy.

## Execution and paper decisions

Use the existing approved worktrees and namespaces. The three reviewers covered heat/Burgers, Poisson, and fresh waves; root checked the shared conclusions and derived diagnostics. This was a read-only scientific review, with documentation as its only output. No branches, jobs or new numerical results were created.

Start with bounded frozen-checkpoint controls and necessary representation diagnostics. Promote independently successful factors into combined methods, then extend the approved mesh ladder and repeat training with separate seeds. Set tolerances and accuracy/cost gates before submission. Preserve final cohorts until choices are frozen. Repeated timing uncertainty must be measured within jobs, with warmed GPUs, raw repetition arrays, per-case medians, outliers and failed cases retained.

Every comparison keeps f64/highest precision, supplied-field inputs, overdetermined smooth weak tests, and complete requested host outputs. Refit decoder-output quadrature when required; no descriptor-conditioned shortcuts or strong-form collocation. Remeasure efficient same/coarse FOM controls in the same job at matched physical accuracy. Report both improved accuracy at fixed cost and improved cost at fixed accuracy, followed by independent confirmation. Record offline training/setup and the many-query break-even if a measured online saving exists.

The linear heat, Poisson and wave equations already admit strong classical solvers. A credible outcome may be a narrower nonlinear-PDE advantage plus well-characterized limits on linear PDEs. Preserve affine ROM controls and explain what the nonlinear manifold adds. Do not claim FOM superiority from a faster old-ROM comparison, reduced-state-only output, or training-selected development examples.

## Reproduction and evidence status

Run `reports/generate_accuracy_cost_strategy.py` with the repository's absolute Python environment. It verifies every artifact hash in the existing results catalog, renders all measured table values, and saves the new subtraction diagnostics and source paths in the [adjacent evidence JSON](2026-09-09-accuracy-and-cost-improvement-strategy.json). Edit the generator's source template and regenerate this report. Proposed gates are explicit design choices, not run results. Only fresh wave evidence after the user's reset is used.

## Plain-language glossary

- **FOM / ROM / NM-ROM:** the full numerical solver, a reduced model, and a reduced model whose possible states form a nonlinear learned surface.
- **Bank, rank, head, latent coordinates:** spatial functions combined to make a field; their count; the network producing their coefficients; the smaller vector entering that network. $G$ denotes the bank and $h$ its nonlinear head.
- **Intervals / cases:** grid subdivisions along each axis / distinct development inputs. A case number identifies an archived input, not a new sample.
- **Relative error, L2, current / initial norm:** discrepancy divided by a reference magnitude; L2 is the square-root sum or integral of squared field values. The divisor is the current reference or its initial value as labeled. Steady fields have no time maximum. Wave state includes displacement and physical velocity with the benchmark's fixed scaling.
- **Full grid / common grid / same grid:** the complete requested output grid / the shared comparison grid / comparison against the reference on the same mesh. Their errors are not interchangeable.
- **Worst error / empirical-adjusted error:** the largest case/time error / that error with the experiment's empirical reference allowance. Refinement evidence is not a rigorous bound.
- **Complete query / median / outlier / envelope:** all charged input-to-host-output work / the middle value / an unusually long retained timing repetition under the original report's rule / the fastest tested FOM configuration meeting the same accuracy target, including coarse-grid interpolation.
- **Weak tests, modes and quadrature:** smooth functions used to average the PDE residual, coefficients against these functions, and weighted sample rules approximating integrals. More tests than active coordinates keeps the fit overdetermined.
- **Projection, residual, stationary, oracle:** a best-fit mapping into a chosen space; mismatch being minimized; a sufficiently small optimization gradient; a reference-informed diagnostic unavailable to an online prediction. A stationary nonlinear fit need not be the global best fit.
- **SVD / QR / Jacobian / curvature:** matrix decompositions revealing important directions or an orthogonal-plus-triangular factorization; decoder/residual first derivatives; decoder second derivatives needed by the current wave geometry.
- **Crank–Nicolson, backward Euler, BDF2, startup:** time-integration rules, with BDF2 using previous states; the first steps that construct the needed history. $\nu$ is heat diffusivity and $\lambda_j$ a discrete positive Laplacian eigenvalue.
- **Damping, budget exit, correction, fallback:** stabilization of a nonlinear update; stopping after allowed work; solving to improve a predicted state; reverting to a more robust solve when required.
- **Affine, protected anchor, variable projection:** a linear map with an offset; a decoder retaining guaranteed linear directions alongside a nonlinear correction; analytically eliminating linear unknowns before optimizing nonlinear ones.
- **Moment, invariant, kinematics, phase, dissipation:** a weighted global field quantity; one preserved by the dynamics; consistency between displacement change and velocity; oscillation timing; loss of energy.
- **Ablation, factorial, gate, parity:** a controlled change; crossed changes revealing their separate and combined effects; a predeclared pass condition; numerical agreement with an unchanged calculation.
- **Development / sealed final cohort / training seed:** examples used to choose a method / untouched independent confirmation inputs / recorded randomness for reproducible training. Repeated development use is not independent confirmation.
- **f64/highest, GPU burn-in, offline, crossover:** required numerical precision settings; warming hardware before timing; work outside a steady query that is still reported; the point where a measured ROM query becomes cheaper at matched accuracy.
"""


if __name__ == "__main__":
    main()
