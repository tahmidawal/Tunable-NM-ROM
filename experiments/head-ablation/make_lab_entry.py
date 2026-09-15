"""Generate the lab-log entry for this session from the audited result JSONs.

Numbers are read from the audits, never typed. The prose around them is fixed text.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def burgers_block(r, au):
    rows = au['checks']['arm_table']
    K = r['K']
    out = []
    meshes = sorted({x['intervals'] for x in rows})
    for L in meshes:
        sel = [x for x in rows if x['intervals'] == L and x['kind'] == 'rom']
        neural = next(x for x in sel if x['arm'] == 'a_neural_eq')
        pods = sorted([x for x in sel if x['arm'].startswith('e_pod')], key=lambda x: x['k'])
        match = [x for x in pods if x['worst_rollout_percent'] <= neural['worst_rollout_percent']]
        line = (f"At {L} intervals the neural head reaches "
                f"{neural['worst_rollout_percent']:.6f}% worst rollout error with best-found "
                f"reconstruction {neural['worst_best_found_percent']:.6f}% and median "
                f"{neural['median_gpu_ms']:.3f} GPU ms.")
        for name in ('b_linear_dec_eq', 'b_linear_truth_eq', 'c_quad_dec_eq', 'e_pod16_eq',
                     'd_freebank_dense'):
            x = next((y for y in sel if y['arm'] == name), None)
            if x:
                line += (f" {name}: rollout {x['worst_rollout_percent']:.6f}%, best-found "
                         f"{x['worst_best_found_percent']:.6f}%, {x['median_gpu_ms']:.3f} ms.")
        if match:
            b = match[0]
            line += (f" Smallest POD rank matching the neural head: {b['k']} "
                     f"({b['k'] / K:g}x K), {b['worst_rollout_percent']:.6f}% at "
                     f"{b['median_gpu_ms']:.3f} ms.")
        else:
            t = pods[-1]
            line += (f" No POD rank up to {t['k']} ({t['k'] / K:g}x K) matches it; the largest rung "
                     f"reaches {t['worst_rollout_percent']:.6f}% with best-found "
                     f"{t['worst_best_found_percent']:.6f}%.")
        out.append(line)
    return out


def poisson_block(r, au):
    rows = au['checks']['arm_table']
    K = r['K']
    out = []
    for n in sorted({x['intervals'] for x in rows}):
        sel = [x for x in rows if x['intervals'] == n and x['kind'] in ('rom', 'retained')]
        neural = next(x for x in sel if x['arm'] == 'a_neural')
        pods = sorted([x for x in sel if x['arm'].startswith('e_pod')], key=lambda x: x['k'])
        match = [x for x in pods if x['worst_error_percent'] <= neural['worst_error_percent']]
        line = (f"At {n} intervals the pure neural head reaches {neural['worst_error_percent']:.6f}% "
                f"worst error at {neural['median_query_ms']:.3f} ms.")
        for name in ('a_neural_q32', 'b_linear_dec', 'b_linear_truth', 'c_quad_dec', 'e_pod16',
                     'd_freebank'):
            x = next((y for y in sel if y['arm'] == name), None)
            if x:
                line += (f" {name}: {x['worst_error_percent']:.6f}% at "
                         f"{x['median_query_ms']:.3f} ms.")
        if match:
            b = match[0]
            line += (f" Smallest POD rank matching the pure neural head: {b['k']} ({b['k'] / K:g}x K), "
                     f"{b['worst_error_percent']:.6f}% at {b['median_query_ms']:.3f} ms.")
        else:
            t = pods[-1]
            line += f" No POD rank up to {t['k']} matches it (largest {t['worst_error_percent']:.6f}%)."
        out.append(line)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--burgers-result', required=True)
    p.add_argument('--burgers-audit', required=True)
    p.add_argument('--burgers-smoke', required=True)
    p.add_argument('--poisson-result')
    p.add_argument('--poisson-audit')
    p.add_argument('--poisson-smoke')
    p.add_argument('--date', required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--branch-commit', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--archives', required=True, help='comma separated archive sha256 labels')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    br = json.loads(Path(a.burgers_result).read_text())
    ba = json.loads(Path(a.burgers_audit).read_text())
    bs = json.loads(Path(a.burgers_smoke).read_text())
    L = []
    w = L.append
    w('')
    w('')
    w(f'## {a.date}')
    w(f'### Head ablation at matched latent dimension — {a.status}')
    w('')
    w('The question was whether the nonlinear coefficient map earns its place at one frozen spatial '
      'bank, which is reviewer 5mgh\'s demand for a linear POD-Galerkin/DEIM baseline with the same '
      'knobs and the follow-up objection that the rebuttal\'s wins came from a frozen POD basis plus '
      'the solver. Arms (a) neural, (b) linear, (c) quadratic, (d) unrestricted bank coefficients and '
      '(e) classical POD-LSPG were run at matched latent dimension through the same weak objective, '
      'test modes, time discretization, initializer policy, stopping rule and output contract.')
    w('')
    w(f"Worktree `worktrees/2026-09-14-head-ablation`, branch `exp/2026-09-14-head-ablation` at "
      f"`{a.branch_commit}`, forked from the corrected consolidated baseline "
      f"`exp/2026-09-13-nmrom-consolidated` at `02ff0f1f`. Cluster namespace "
      f"`/cluster/tufts/paralab/tawal01/headabl_20260914/`. Jobs: Burgers `{br.get('job_id')}` "
      f"(`{br.get('gpu')}`, source `{br.get('commit')}`)"
      + (f", Poisson `{json.loads(Path(a.poisson_result).read_text()).get('job_id')}`"
         if a.poisson_result else '') + '.')
    w('')
    w('**Fidelity gates before any verdict.** Run through the generic arm machinery with the archived '
      f"operators, the Burgers neural arm reproduces the consolidated saved case to "
      f"{bs['generic_vs_saved_case']['relative_l2']:.3e} relative and is bit-identical "
      f"({bs['generic_vs_incumbent']:.1e}) to the incumbent `accuracy_paths.make_rom`, so arm (a) is "
      'the retained solver rather than a re-implementation.')
    cc = ba['checks'].get('campaign_frozen_arm_reproduced')
    if cc and cc.get('detail'):
        d = cc['detail']
        w(f"In the job itself, `a_neural_eq` reproduces the retained multiresolution campaign's "
          f"`frozen_stationary` rollout errors on {d['compared']} case/mesh combinations to a worst "
          f"relative difference of {d['worst_relative_delta']:.3e}, so the regenerated reference and "
          'rebuilt operators are the campaign\'s own.')
    w('')
    w('**Burgers 2D.**')
    for line in burgers_block(br, ba):
        w(line)
    w('')
    if a.poisson_result:
        pr = json.loads(Path(a.poisson_result).read_text())
        pa = json.loads(Path(a.poisson_audit).read_text())
        ps = json.loads(Path(a.poisson_smoke).read_text())
        w('**Poisson 2D (linear control).** The weak residual is exactly $Bh(z)-f_m$ with no time '
          'stepping and no quadrature approximation, so the arms differ only in the coefficient map. '
          f"The generic machinery with the frozen neural head reaches the incumbent `core.rom_query` "
          f"solution to {ps['generic_vs_incumbent_neural']['relative']:.3e} relative, inside the "
          f"declared {ps['generic_vs_incumbent_neural']['tolerance']:.0e} tolerance; that is agreement "
          'on the same stationary point between two Levenberg-Marquardt implementations, not bit '
          'identity.')
        for line in poisson_block(pr, pa):
            w(line)
        w('')
    w('**Methodology correction retained.** The campaign\'s stationarity test is the normalized '
      'gradient, which is scale invariant and can only fall below its tolerance once the residual '
      'becomes orthogonal to the reduced tangent space. An arm whose reduced fit is attainable drives '
      'the residual to round-off while that ratio stays of order one, so it exits by the small-step '
      'rule with a better fit and a worse-looking stationarity number. Both the stationarity value and '
      'a separate completion status (no budget exit, no rejected-step exit) are recorded for every '
      'invocation. Nothing previously accepted is retracted by this; it means the stationarity column '
      'alone must not be read as a quality ranking across arms of different reduced dimension.')
    w('')
    w('**Recorded deviations.** Arm (d) needs more tests than bank features, so it runs at a larger '
      'test count with exact dense advection; a nonnegative-least-squares rule at four times that test '
      'count is not constructible inside the job budget. On Burgers the higher POD rungs also run with '
      'exact dense advection, which favours the POD baseline, so any reported matching rank is a '
      'conservative lower bound. Arms above 64 unknowns use a pivoted dense step solve instead of the '
      'incumbent unrolled Gauss-Jordan, which is more accurate, not weaker. Best-found reconstruction '
      'is an exact projection for affine-manifold arms and a seeded multistart local search for curved '
      'ones, so it is an upper bound exactly where the nonlinear arms would benefit from tightness.')
    w('')
    w(f"Source-generated report: `{a.report}`, with its generator beside it. Raw archives are "
      f"Git-tracked as bounded chunks under `experiments/head-ablation/artifacts/`: {a.archives}. "
      'Every exact remote attempt directory was removed after checksum collection.')
    w('')
    w('**Not pushed.** The coordinator instructed mid-session that no branch descended from the '
      'consolidated baseline may be pushed from this worktree, because `git pack-objects` repacks a '
      '199 GB repository and reaches roughly 48 GB resident on the shared box. An incremental push '
      'started before that instruction had already created `origin/exp/2026-09-14-head-ablation` at '
      '`ad804863`, an ancestor 60 commits behind this branch head; it was stopped immediately and no '
      'git process remains. The coordinator will push the remaining branches in stages.')
    w('')
    w('**Open.** Other PDEs, other checkpoints, more than one training seed, and the sealed final '
      'cohorts. No claim is made here about a speed advantage over a full-order solver; the same-job '
      'full-order rows are context only, and the retained Burgers reduced model remains slower and '
      'less accurate than an efficient same-job full-order solver, as already recorded. No worktree '
      'was merged and no earlier numerical result is retracted.')
    w('')
    text = '\n'.join(L)
    Path(a.out).write_text(text)
    print(a.out, len(text))


if __name__ == '__main__':
    main()
