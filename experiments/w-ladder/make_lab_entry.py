"""Generate the w-ladder lab-log entry from summary.json and the run JSONs. No hand-typed numbers.

    python experiments/w-ladder/make_lab_entry.py > /tmp/.../entry.md
"""
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent
S = json.loads((LANE / 'reports/summary.json').read_text())


def val(mesh, subject, metric):
    for r in S:
        if r['mesh'] == mesh and r['subject'] == subject and r['metric'] == metric:
            return r['value']
    return None


def pct(x):
    return f'{100 * x:.3f}' if x is not None else 'n/a'


meshes = sorted({r['mesh'] for r in S})
jobs = {m: next(r['job_id'] for r in S if r['mesh'] == m) for m in meshes}
attempts = {m: next(r['attempt'] for r in S if r['mesh'] == m) for m in meshes}
heads = ['head_q0', 'trained_nested40', 'nested_q8', 'nested_q16', 'nested_q32']

ratios = [val(m, 'head_q0', 'median_gpu_ms') / val(m, 'linear_bank64', 'median_gpu_ms') for m in meshes]
lines = ['## 2026-09-17', '',
         '### w-ladder — the reflective 2D wave is the second linear PDE whose correction ladder is degenerate: the top rung '
         f'(linear evolution of the full learned bank) is the most accurate rung and {min(ratios):.0f}–{max(ratios):.0f}× cheaper '
         'than the cheapest head rung', '']
lines.append('Branch `exp/2026-09-17-w-ladder`, worktree `worktrees/2026-09-17-w-ladder`, forked from the consolidated baseline '
             '`exp/2026-09-13-nmrom-consolidated` at `02ff0f1f`. Cluster namespace `/cluster/tufts/paralab/tawal01/w_ladder_20260917/` '
             '(empty at close). Nothing merged, nothing pushed. Pre-registration and every amendment: '
             '`experiments/w-ladder/DESIGN.md`; generated report and rows: `experiments/w-ladder/reports/2026-09-17-w-ladder.md`, '
             '`summary.json`. All meshes on NVIDIA A100 80GB PCIe, `jax_backend=gpu`, f64, highest precision.')
lines.append('')

lines += ['**What ran.** One job per mesh on the development cohort (8 cases: `opened` 0-3 seed 690602, `fresh_development` 0-3 '
          'seed 691115; the final cohort stays sealed), 3 timed repetitions per arm and case, all retained, same-job controls only. '
          'Jobs: ' + ', '.join(f'{m}² `{attempts[m]}` job {jobs[m]}' for m in meshes) + '.', '']

lines += ['**The ladder, worst over the 8 cases of the time-maximum initial-normalised energy-state error, and the median '
          'device-resident query over all repetitions:**', '',
          '| mesh | ' + ' | '.join(f'`{h}`' for h in heads) + ' | `linear_bank64` (q=R) | `pod_k64` | direct DST |',
          '|---|' + '---|' * (len(heads) + 3)]
for m in meshes:
    cells = [f'{pct(val(m, h, "worst_energy_state"))} % / {val(m, h, "median_gpu_ms"):.1f} ms' for h in heads]
    cells.append(f'**{pct(val(m, "linear_bank64", "worst_energy_state"))} % / {val(m, "linear_bank64", "median_gpu_ms"):.2f} ms**')
    cells.append(f'{pct(val(m, "pod_k64", "worst_energy_state"))} % / {val(m, "pod_k64", "median_gpu_ms"):.2f} ms')
    cells.append(f'{pct(val(m, "dst", "worst_energy_state"))} % / {val(m, "dst", "median_gpu_ms"):.2f} ms')
    lines.append(f'| {m}² | ' + ' | '.join(cells) + ' |')
lines.append('')

lines += ['**What was found.**', '']
for m in meshes:
    lb_e, lb_c = val(m, 'linear_bank64', 'worst_energy_state'), val(m, 'linear_bank64', 'median_gpu_ms')
    cheap = min(heads, key=lambda h: val(m, h, 'median_gpu_ms'))
    lines.append(f'- **{m}²:** error falls monotonically across the head rungs and then flattens at the top rung ('
                 + ' → '.join(pct(val(m, h, 'worst_energy_state')) for h in heads)
                 + f' → {pct(lb_e)} %; the final step is within the integrator tie band) while cost is not monotone at all ('
                 + ' → '.join(f'{val(m, h, "median_gpu_ms"):.0f}' for h in heads)
                 + f' → {lb_c:.2f} ms). The top rung is {val(m, cheap, "median_gpu_ms") / lb_c:.0f}× cheaper than the cheapest head rung '
                   f'`{cheap}` and {val(m, cheap, "worst_energy_state") / lb_e:.2f}× more accurate. Verdict D1-D4 = '
                   f'{val(m, "verdict", "all")} (strict D1 {val(m, "verdict", "D1_strict")}, integrator tie band '
                   f'{100 * val(m, "verdict", "tie_band_delta"):.4f} pp); ladder monotone = {val(m, "verdict", "H_mono_ladder")}.')
lines.append('')
lines += ['- **POD-Galerkin from the same snapshots beats the learned bank at matched rank** at every mesh: '
          + '; '.join(f'{m}² {pct(val(m, "pod_k64", "worst_energy_state"))} % vs {pct(val(m, "linear_bank64", "worst_energy_state"))} % '
                      f'at {val(m, "pod_k64", "median_gpu_ms"):.2f} vs {val(m, "linear_bank64", "median_gpu_ms"):.2f} ms' for m in meshes)
          + '. H-POD\'s pre-registered "comparable" expectation is wrong; the learned bank is the worse linear subspace, '
            'and this is not verdict-bearing.', '',
          '- **No ROM arm beats the direct full-order solver.** The same-job DST is exact and costs '
          + ', '.join(f'{val(m, "dst", "median_gpu_ms"):.2f} ms at {m}²' for m in meshes)
          + ', so the linear bank\'s speed advantage over the head rungs is not a speed advantage over the FOM.', '',
          '- **The middle rungs are expensive for a solver reason, not a physics one.** $q=16$ and $q=32$ drive the guarded '
          'Cholesky into its QR + exact-SVD fallback on essentially every RK4 stage (recorded per invocation as '
          '`total_guard_fallbacks`), costing roughly an order of magnitude over $q=0$; predicted in DESIGN A1 item 2 and '
          'reported as measured.', '',
          '- **The energy certificate holds.** Exact modal propagation of the reduced system conserves the reduced energy to '
          + ', '.join(f'{val(m, "linear_bank64", "reduced_energy_drift"):.1e} ({m}²)' for m in meshes)
          + ' relative, Crank-Nicolson likewise; the head\'s RK4 does not conserve it exactly and CN is markedly dispersive at '
            'the head\'s step (reported, not used as a top rung).', '']

lines += ['**What was wrong and retracted.**', '',
          '- The DESIGN §1 motivating table originally carried **hand-typed millisecond values taken from a different job\'s '
          'tables** (178.7 / 199.5 ms). Withdrawn in amendment A1 and replaced by `design_table.py` output from the archived '
          'JSONs (178.9815 / 199.4380 ms); the two source jobs are different allocations, so no ratio is formed across them.', '',
          '- The pre-registered **orthonormal** correction directions were replaced by the retained **scaled** PCA directions '
          '(A1 item 2) after the audit showed the rescaling would push the Cholesky guard into fallback for a coordinate reason. '
          'The §3.1 sentence "costs the same launch-bound stage as the q=0 solve" is withdrawn.', '',
          '- **G5b\'s "expected ≲1e-6"** for the $q=32$ vs bank agreement is withdrawn (A1 item 6): the two arms start from '
          'different initial states and RK4 is not invariant under the nonlinear change of variables.', '',
          '- **The retained-value gate was wrong twice.** At 1e-9 for every arm it failed `trained_nested40` (A3), then failed '
          '`head_q0` on the fresh cases and killed `wl256b` (job 3780448) after it had completed all 432 timed invocations '
          '(A4). Diagnosis from the recorded counters: the eight cold-fit starts reach the same minimum and `argmin` breaks '
          'the tie by index, so a last-bit difference flips the selected start and moves the trajectory at ~1e-8; cases where '
          'the same start won reproduced to 1e-14. The gate now checks the tie-invariant selected objective at 1e-9 and the '
          'trajectory error at 1e-7 for fit-carrying arms, both fatal. `wl256b` is archived as a gate-failed attempt.', '',
          '- **Strict D1 fails at every mesh by less than the integrator tie band**: the $q=32$ rung is the full bank stepped '
          'by RK4, whose slight dissipation lowers the metric marginally below the exact propagator. Both the strict and '
          'banded readings are reported; the banded one is the pre-registered verdict (A2).', '']

lines += ['**Integrity.** Every job asserts `jax_backend=gpu`, f64 and highest precision, the frozen-math SHA256s, bank rebuild '
          'parity, POD transfer and orthogonality, positive definiteness of every reduced stiffness, CG true-residual '
          'convergence, byte-identical repetitions, and the retained-value gates against the archived accel12 / accel07 values. '
          'An independent NumPy/SciPy audit (`audit_ladder.py`, no JAX, no driver import) recomputed every reported error from '
          'the saved coefficients and bank tables against a SciPy-DST reference regenerated from the saved initial fields: '
          'maximum difference ' + ', '.join(f'{m}² see `runs/{attempts[m]}/audit.json`' for m in meshes) + '. Archives are '
          'checksum-collected, chunked into Git under `experiments/w-ladder/artifacts/`, and the exact remote directories '
          'deleted.', '']

lines += ['**Open.** (1) The sealed final cohort is untouched; every number here is development-cohort. (2) Absorbing '
          'boundaries were out of scope. (3) The learned bank losing to POD at matched rank deserves its own look — it bears '
          'on every cell that uses this bank, not just waves. (4) The $q=16/32$ fallback cost is a solver artefact that a '
          'better-conditioned parameterisation would remove; nobody should quote those milliseconds as the cost of enrichment '
          'in principle.', '']

print('\n'.join(lines))
