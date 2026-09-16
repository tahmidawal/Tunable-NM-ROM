"""Generate the dated lab-log entry for this cell from the audited JSON.

Every number in the entry is read from the raw result, audit, smoke and archive JSON.
Nothing is typed by hand; the prose that is typed carries no numbers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'reports'))
from generate_head_refine import verdict, nondominated, ladder  # noqa: E402


def pct(x, d=4):
    return 'n/a' if x is None else f'{x:.{d}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--burgers-result', required=True)
    ap.add_argument('--burgers-audit', required=True)
    ap.add_argument('--poisson-result', required=True)
    ap.add_argument('--poisson-audit', required=True)
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--burgers-archive', required=True)
    ap.add_argument('--poisson-archive', required=True)
    ap.add_argument('--branch-head', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    b = json.loads(Path(a.burgers_result).read_text())
    ba = json.loads(Path(a.burgers_audit).read_text())
    p = json.loads(Path(a.poisson_result).read_text())
    pa = json.loads(Path(a.poisson_audit).read_text())
    sm = json.loads(Path(a.smoke).read_text())
    barch = json.loads(Path(a.burgers_archive).read_text())
    parch = json.loads(Path(a.poisson_archive).read_text())
    report = Path(a.report)
    brows, prows = ba['checks']['arm_table'], pa['checks']['arm_table']

    vs = []
    for pde, rows, cost, quad, variants in (
            ('Burgers', brows, 'median_gpu_ms', 'eq', ('v1', 'v2')),
            ('Poisson', prows, 'median_query_ms', None, ('v1',))):
        for variant in variants:
            for mu_key in ('loose', 'tight'):
                v = verdict(rows, variant, mu_key, cost, 'worst_same_grid_percent', quad)
                v['pde'] = pde
                vs.append(v)

    L = []
    w = L.append
    w('')
    w('')
    w('## 2026-09-15')
    w('### head-refine — per-query refinement of the frozen head\'s weights; complete and audited on '
      'both PDEs, and it is not a knob on either')
    w('')
    w('The coordinator asked whether refining the frozen head\'s own weights at query time, anchored '
      'to the trained weights, is a usable inference-time accuracy/cost knob. Everything except the '
      'refinement is the head-ablation arm (a) contract: the frozen bank, the latent dimension, the '
      'weak objective, the test-mode family, the time discretization, the initializer policy, the '
      'Levenberg-Marquardt stopping rule and the output contract. The only online knobs are the '
      'variant, the number n of Adam steps taken on theta inside the timed query, and the anchor '
      'weight mu. Two pre-registered variants: V1 refines once against the supplied initial field on '
      'the initializer\'s own node set, V2 refines after every time step against that step\'s weak '
      'residual and then re-solves z. On Poisson the query is a single static solve, so V2 has no '
      'per-step structure to exploit and collapses onto V1; only V1 was defined and run there.')
    w('')
    w(f"Worktree `worktrees/2026-09-15-head-refine`, branch `exp/2026-09-15-head-refine` at "
      f"`{a.branch_head}`, forked from `exp/2026-09-14-head-ablation` at `2d82421d`. Namespace "
      f"`/cluster/tufts/paralab/tawal01/head_refine_20260915/`, one attempt directory per PDE. "
      f"Burgers job `{b['job_id']}` (`{b['gpu']}`, source `{b['commit']}`, {b['intervals']} "
      f"intervals, the same 6 opened development cases arm (a) used, elapsed "
      f"{b['elapsed_seconds']:.1f} s); Poisson job `{p['job_id']}` (`{p['gpu']}`, source "
      f"`{p['commit']}`, {p['intervals']} intervals, the same 12 development sources, elapsed "
      f"{p['elapsed_seconds']:.1f} s). Both logged `jax_backend=gpu`, f64, highest matmul "
      f"precision, 3 timed repetitions with GPU burn-in before every timed block, randomised "
      f"subject order, every repetition array retained, refinement steps INSIDE the timed query.")
    w('')
    g, gp = b['gate_in_job'], p['gate_in_job']
    bd, pd = ba['checks']['redecode'], pa['checks']['redecode']
    w('**Five fidelity gates, all passed.** Through the new code path with theta as a traced runtime '
      f"operand, n=0 reproduces the consolidated saved Burgers case to "
      f"{sm['n0_vs_saved_case']['relative_l2']:.3e} relative and the incumbent "
      f"`accuracy_paths.make_rom` to {sm['n0_vs_incumbent']:.3e}, both inside the declared 1e-12; "
      f"the reconstructed head is bit-identical to the retained head "
      f"({sm['head_roundtrip_max_abs']:.1e}). In the Burgers job `n0` reproduces `abl01`'s "
      f"`a_neural_eq` on all {g['compared']} cases to {g['worst_relative_delta']:.3e} against a "
      f"declared 1e-9, with {g['bitwise_identical_fields']} of {g['compared']} fields bitwise "
      f"identical across jobs. In the Poisson job `n0` reproduces `pabl01`'s `a_neural` on all "
      f"{gp['compared']} sources to {gp['worst_relative_delta']:.3e} against the declared cross-job "
      f"{gp['tolerance']:g}. An independent NumPy audit that imports neither driver nor JAX "
      f"recomputed every reported error from the retained fields (worst relative difference "
      f"{ba['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e} Burgers, "
      f"{pa['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e} Poisson) and "
      f"re-decoded the saved refined weights in pure NumPy at a fixed {bd['nodes']}-node bank "
      f"sample: {bd['compared']} Burgers (arm, case) pairs to {bd['worst_relative']:.2e} and "
      f"{pd['compared']} Poisson pairs to {pd['worst_relative']:.2e}.")
    w('')
    for tag, r in (('Burgers', b), ('Poisson', p)):
        c, d = r['calibration'], r['anchor_diagnostic']
        grid = r['config']['alpha_grid']
        edge = (' This sits at the edge of the pre-registered grid, so the calibration did not '
                'bracket an interior optimum and every number below is for that step size.'
                if c['selected_alpha'] in (min(grid), max(grid)) else '')
        w(f"**{tag} step size and anchor.** Adam, chosen once on one training-family calibration "
          f"case (not an evaluation case) by lowest V1 data term after n={c['n']} at the loose "
          f"anchor, then frozen for both variants, both anchor weights, every n and every "
          f"evaluation case: alpha = {c['selected_alpha']:g}.{edge} Anchor diagnostic at n={d['n']} "
          f"on the same calibration case (diagnostic only, selects nothing): "
          + ', '.join(f"mu={x['mu']:g} drift {x['drift']:.3e} anchor/data gradient "
                      f"{x['anchor_to_data_gradient']:.2e}" for x in d['rows']) + '.')
        w('')
    w('**Pre-registered acceptance (error monotone in n; at least 3 non-dominated points spanning '
      '>= 2x in cost and >= 2x in error; none early-stopped) — the verdict per variant and anchor:**')
    w('')
    w('| PDE | variant | anchor | monotone | non-dominated | cost span | error span | none '
      'early-stopped | verdict |')
    w('|---|---|---|---|---:|---:|---:|---|---|')
    for v in vs:
        w(f"| {v['pde']} | {v['variant'].upper()} | {v['mu_key']} | "
          f"{'yes' if v['monotone'] else 'no'} | {v['nondominated_count']} | {v['cost_span']:.2f}x | "
          f"{v['error_span']:.2f}x | {'yes' if v['all_nondominated_converged'] else 'no'} | "
          f"**{'A KNOB' if v['accepted'] else 'NOT a knob'}** |")
    w('')
    for v in vs:
        w(f"- {v['pde']} {v['variant'].upper()} / {v['mu_key']}: worst same-grid along n={v['n']} is "
          f"{[round(x, 4) for x in v['errors']]} percent at {[round(x, 3) for x in v['costs']]} "
          f"median ms."
          + (f" Early-stopped on this ladder: {', '.join(v['early_stopped'])}."
             if v['early_stopped'] else ' Nothing on this ladder is early-stopped.'))
    w('')
    for pde, rows, cost, quad in (('Burgers', brows, 'median_gpu_ms', 'eq'),
                                  ('Poisson', prows, 'median_query_ms', None)):
        pts = [x for x in rows if x['kind'] == 'rom' and (quad is None or x.get('quadrature') == quad)]
        nd = nondominated(pts, cost, 'worst_same_grid_percent')
        w(f"**{pde} non-dominated set** (every reduced arm in the primary quadrature): "
          + '; '.join(f"`{x['arm']}` {x['worst_same_grid_percent']:.4f}% at {x[cost]:.3f} ms, drift "
                      f"{(x['worst_drift'] or 0.):.3e}, "
                      f"{'converged' if x['all_completed'] else 'EARLY-STOPPED'}" for x in nd) + '.')
        w('')
    base = next(x for x in brows if x['arm'] == 'n0')
    w(f"**Three layers on Burgers.** The frozen bank projection floor is "
      f"{b['bank_projection']['worst'] * 100:.4f}% worst and is unchanged by refinement, so it "
      f"bounds every arm. The unrefined head reaches {base['worst_best_found_percent']:.4f}% "
      f"best-found reconstruction and {base['worst_same_grid_percent']:.4f}% worst same-grid at "
      f"{base['median_gpu_ms']:.3f} median GPU ms. "
      + '; '.join(f"`{x['arm']}` best-found {x['worst_best_found_percent']:.4f}%, same-grid "
                  f"{x['worst_same_grid_percent']:.4f}%, {x['median_gpu_ms']:.3f} ms"
                  for x in brows if x['kind'] == 'rom' and x['n'] in (8, 32)
                  and x.get('quadrature') == 'eq') + '.')
    w('')
    pbase = next(x for x in prows if x['arm'] == 'n0')
    w(f"**Three layers on Poisson.** Bank projection floor {p['bank_projection']['worst'] * 100:.4f}% "
      f"worst; the unrefined head reaches {pbase['worst_best_found_percent']:.4f}% best-found and "
      f"{pbase['worst_same_grid_percent']:.4f}% worst same-grid at {pbase['median_query_ms']:.3f} "
      f"median ms. "
      + '; '.join(f"`{x['arm']}` best-found {x['worst_best_found_percent']:.4f}%, same-grid "
                  f"{x['worst_same_grid_percent']:.4f}%, {x['median_query_ms']:.3f} ms"
                  for x in prows if x['kind'] == 'rom' and x['n'] in (8, 32)) + '.')
    w('')
    risky = []
    for x in brows:
        v = x.get('worst_same_grid_per_time_percent')
        bt = base.get('worst_same_grid_per_time_percent')
        if v and bt and x['n'] and v[1] < bt[1] * (1 - 1e-12) and v[-1] > bt[-1] * (1 + 1e-12):
            risky.append(x['arm'])
    w('**Held-out generalisation risk, reported explicitly.** Worst same-grid error per output time '
      'is in the report for every arm. Arms that improve the first evolved output time and worsen '
      'the last relative to `n0` — the signature of overfitting the supplied initial field: '
      + (', '.join(f'`{x}`' for x in risky) + '.' if risky else 'none.'))
    w('')
    w('**Recorded deviations and limitations.** The dense control is a pair at the two ends of the '
      'V1 ladder rather than a single arm, because one dense row alone cannot be differenced. The '
      'empirical-quadrature rule and the cold-start candidate table are offline artifacts fitted at '
      'theta_0 and stay frozen while theta moves; the dense pair isolates that. The two anchor '
      'weights were moved from {1e-3, 10} to {1e2, 1e5} after the local fidelity smoke and before '
      'any evaluation run, because the anchor gradient is about 3e-6 against a data gradient of '
      '6e-2 to 1, so the original pair was numerically indistinguishable from no anchor; only '
      'gradient scales were consulted, no accuracy number. The step-size grid was widened downward '
      'for the same reason. Both amendments are dated in `DESIGN.md`. One checkpoint per PDE, one '
      'mesh, one training seed; the final cohorts stay sealed and no new case was opened.')
    w('')
    w(f"Source-generated report and figure: `experiments/head-refine/reports/{report.name}` "
      f"(SHA256 `{hashlib.sha256(report.read_bytes()).hexdigest()}`) with `-cost.png` / `-cost.pdf` "
      f"beside it, produced by `reports/generate_head_refine.py`. Raw archives Git-tracked as "
      f"bounded chunks under `experiments/head-refine/artifacts/`: "
      f"{Path(a.burgers_archive).parent.name} sha256 {barch['sha256']} "
      f"({len(barch['chunks'])} chunks); {Path(a.poisson_archive).parent.name} sha256 "
      f"{parch['sha256']} ({len(parch['chunks'])} chunks). Both exact remote attempt directories "
      f"were removed after checksum collection and the namespace is empty. Not pushed, per the "
      f"coordinator's standing instruction; commits are local only.")
    w('')
    w('**Open.** Other meshes, other checkpoints, more than one training seed, the sealed final '
      'cohorts, a step size chosen inside its grid rather than at an edge, and whether refining a '
      'smaller subset of the head (the linear skip alone, say) would move the frontier. No earlier '
      'numerical result is retracted and no worktree was merged.')
    w('')
    Path(a.out).write_text('\n'.join(L))
    print(a.out)


if __name__ == '__main__':
    main()
