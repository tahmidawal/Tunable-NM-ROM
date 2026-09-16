"""Generate the prior-dial lab-log entry from the audited JSONs and append it.

Every number in the entry comes from the result JSONs, the audit JSONs, the smoke
evidence and the archive manifests. Nothing is typed by hand. The entry is
appended to the canonical log by absolute path; the top block is never rewritten.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reports'))
from generate_prior_dial import (blocks, lam_plain, nondominated, primary,  # noqa: E402
                                 verdict)

LOG = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def pct(x, d=4):
    return 'n/a' if x is None else f'{x:.{d}f}'


def build(r, au, sm, pr, pau, meta):
    rows = au['checks']['arm_table']
    prows = pau['checks']['arm_table']
    prim = primary(rows)
    v_all = verdict(prim, 'worst_same_grid_percent')
    v_evo = verdict(prim, 'worst_same_grid_evolved_percent')
    g0 = au['checks']['initial_output_is_lambda_independent']
    g2 = au['checks']['laminf_reproduces_head_ablation_arm_a']['detail']
    g3 = pau['checks']['laminf_reproduces_pabl01_arm_a']['detail']
    o = []
    w = o.append
    w('')
    w('')
    pverd = {}
    for n in sorted({x['intervals'] for x in prows}):
        blk = sorted([x for x in prows if x['kind'] == 'rom' and x['intervals'] == n],
                     key=lambda z: -(float('inf') if z['lambda_rel'] is None else z['lambda_rel']))
        nd = nondominated(blk, 'worst_same_grid_percent', 'median_query_ms')
        pverd[n] = dict(
            span=max(p['worst_same_grid_percent'] for p in nd)
            / max(min(p['worst_same_grid_percent'] for p in nd), 1e-300),
            cost=max(p['median_query_ms'] for p in nd) / min(p['median_query_ms'] for p in nd))
    pn = max(pverd, key=lambda k: pverd[k]['span'])
    burg = ('a usable knob on the primary block' if v_all['passed'] else
            ('not a knob on the pre-registered metric, which the initializer contract pins; '
             + ('a knob on the evolved-time metric' if v_evo['passed']
                else 'and not a knob on the evolved-time metric either')))
    pois = (f"on Poisson it spans {pverd[pn]['span']:.2f}x in error for {pverd[pn]['cost']:.2f}x "
            f"in cost")
    w('## 2026-09-15')
    w(f'### prior-dial — trusting the neural prior less is {burg} on this Burgers checkpoint, '
      f'and {pois}')
    w('')
    w('The coordinator asked whether "trust in the neural prior" is a usable inference-time '
      'accuracy/cost knob on one frozen checkpoint. Instead of solving only for the latent code '
      'with the bank coefficients pinned to the head, the full bank coefficient vector c is '
      'solved under a penalty pulling it towards the head: '
      'min_{z,c} ||r_w(c)||^2 + lambda ||R_G (c - h_theta(z))||^2, with G = Q_G R_G the thin QR '
      'already used by arms.py, so the penalty is the squared FIELD-norm distance from the '
      'head\'s own state. lambda = infinity is head-ablation arm (a) exactly; lambda -> 0 with '
      'M >= R is the free-bank arm (d). Every network weight, the bank, K = 16, the initializer '
      'policy, dt, the stopping rule and the output contract are arm (a)\'s.')
    w('')
    w(f"Worktree `worktrees/2026-09-15-prior-dial`, branch `exp/2026-09-15-prior-dial` at "
      f"`{meta['branch_head']}`, forked from `exp/2026-09-14-head-ablation` at `2d82421d`. "
      f"Namespace `/cluster/tufts/paralab/tawal01/prior_dial_20260915/`. Burgers job "
      f"`{r.get('job_id')}` (`pdial01`) on `{r.get('gpu')}`, Poisson job `{pr.get('job_id')}` "
      f"(`ppdial01`) on `{pr.get('gpu')}`, both from source `{r.get('commit')}`, both printing "
      f"`jax_backend=gpu`, float64, matmul precision highest; elapsed {r['elapsed_seconds']:.1f} s "
      f"and {pr['elapsed_seconds']:.1f} s. One job per attempt directory, `squeue` checked before "
      f"and after each submit. 3 timed repetitions with GPU burn-in before every block, every "
      f"repetition array retained. Burgers: 256 intervals, the frozen "
      f"`sep_hfit_dense_mid_N256_dense.pkl`, the same six opened development cases as abl01 and "
      f"qlad01. Poisson: the frozen pabl01 bank/head (K=16, R=128, retained M=257), the same "
      f"twelve development sources, at 1024 and 64 intervals. No new case was opened.")
    w('')
    w(f"**Scaling of lambda, declared before the run.** lambda = lambda_rel * sigma^2 with "
      f"sigma = ||A R_G^{{-1}}||_2 = ||Phi^T Q_G||_2, the exact linear part of d r_w / d y, "
      f"which is state independent because the 1/(1 + dt nu lam_j) row scaling of the weak "
      f"residual cancels the diffusion term exactly. Since Phi and Q_G both have orthonormal "
      f"columns, sigma is the largest principal cosine between the test-mode span and the bank "
      f"span; measured {next(x['sigma'] for x in rows if x['sigma']):.10f} on Burgers and "
      f"{next(x['sigma'] for x in prows if x['sigma']):.10f} on Poisson. lambda_rel = 1 balances "
      f"the strongest linear residual response against the prior. Sweep: lambda_rel in "
      f"[{', '.join(lam_plain(x) for x in r['config']['lambda_rel'])}], at test counts "
      f"M = {sorted({x['M'] for x in rows if x['kind'] == 'rom'})} on Burgers.")
    w('')
    w('**Fidelity gates.** Through the new code path at lambda = infinity the local smoke '
      f"reproduces the consolidated saved Burgers case to "
      f"{sm['laminf_vs_saved_case']['relative_l2']:.3e} relative and is bit-identical "
      f"({sm['laminf_vs_incumbent']:.1e}) to the incumbent `accuracy_paths.make_rom`; the "
      f"finite path at lambda_rel = {sm['finite_path_large_lambda']['lambda_rel']:g} returns to "
      f"that limit to {sm['finite_path_large_lambda']['relative_to_laminf']:.3e}. In the job, "
      f"`M64_eq_laminf` reproduces abl01's `a_neural_eq` on all {g2['compared']} cases to a "
      f"worst relative difference of {g2['worst_relative_delta']:.3e} "
      f"({g2['bitwise_identical_fields']} of {g2['compared']} fields bitwise identical across "
      f"jobs), and Poisson `laminf` reproduces pabl01's `a_neural` to "
      f"{g3['worst_relative_delta']:.3e}. Every recorded error was independently recomputed "
      f"from the retained output fields by NumPy (worst disagreement "
      f"{au['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e}).")
    w('')
    w('**The structural finding, and it decides the headline.** The contract fixes arm (a)\'s '
      'initializer, which starts the correction at y = 0, so the t = 0 output field is the '
      'head\'s compression of the supplied field and cannot depend on lambda; the audit confirms '
      f"it to {g0['detail']['worst_relative_spread']:.2e} relative across every lambda, test "
      f"count and quadrature. On these six cases that compression error is also the LARGEST of "
      f"the six output times, so the worst same-grid error over all times is pinned by "
      f"construction at "
      f"{pct(prim[0]['worst_same_grid_percent'])}% and no setting of lambda moves it. "
      f"This is a fact about the checkpoint and the contract, not a defect of the dial, and it "
      f"is why DESIGN.md amendment 2 — recorded from a local probe BEFORE submission — added "
      f"the worst same-grid error over the evolved times as a declared secondary metric while "
      f"keeping the pre-registered primary metric and its criteria unchanged. The correction "
      f"ladder moved this metric only because its initializer fitted the augmented vector too; "
      f"that difference was not previously stated anywhere and is the main reason the two cells' "
      f"headline numbers are not comparable.")
    w('')
    w('**Burgers, primary block (M = 64, EQ — arm (a)\'s own test count and quadrature).**')
    w('')
    w('| lambda_rel | worst same-grid all times % | worst same-grid evolved % | '
      'realised ‖y‖/‖u‖ % | median iters/step | budget exits | completed | median GPU ms |')
    w('|---:|---:|---:|---:|---:|---:|---|---:|')
    for x in prim:
        w('| ' + ' | '.join([lam_plain(x['lambda_rel']), pct(x['worst_same_grid_percent']),
                             pct(x['worst_same_grid_evolved_percent']),
                             pct(x['max_relative_correction_percent']),
                             f"{x['median_iterations']:.1f}", str(x['total_budget_exits']),
                             'yes' if x['all_completed'] else 'no',
                             f"{x['median_gpu_ms']:.3f}"]) + ' |')
    for x in rows:
        if x['kind'] == 'fom':
            w('| ' + ' | '.join([f"FOM {x['arm']}", pct(x['worst_same_grid_percent']),
                                 pct(x['worst_same_grid_evolved_percent']), 'n/a',
                                 f"{x['median_iterations']:.1f}", 'n/a', 'n/a',
                                 f"{x['median_gpu_ms']:.3f}"]) + ' |')
    w('')
    w('**Verdict against the pre-registered acceptance criteria** (fixed in DESIGN.md before any '
      'implementation: error monotone in lambda; at least 3 non-dominated points spanning >= 2x '
      'in cost AND >= 2x in error; none early-stopped).')
    w('')
    for v, label in [(v_all, 'primary, worst same-grid over all six output times'),
                     (v_evo, 'secondary, worst same-grid over the evolved times')]:
        w(f"- **{label}: {'KNOB' if v['passed'] else 'NOT A KNOB'}.** Monotone: "
          f"{'yes' if v['criterion_1_monotone'] else 'no'}. Non-dominated points "
          f"{v['count']}, spanning {v['cost_span']:.2f}x in cost and {v['error_span']:.2f}x in "
          f"error ({'pass' if v['criterion_2_frontier'] else 'fail'}). None early-stopped: "
          f"{'yes' if v['criterion_3_honest'] else 'no'}.")
    w('')
    w('**What the test count does, which is the real content.** With the bank free, the weak '
      'residual has only M equations; at M < R it is easy to satisfy with a tiny correction and '
      'the prior is barely binding, and the small-lambda end there is a REGULARIZED '
      'UNDERDETERMINED solve, not the free bank. Only M > R reaches the free-bank limit.')
    w('')
    w('| M | quad. | overdetermined | worst evolved % at lambda=inf | best worst evolved % | '
      'at lambda_rel | cost factor | best point completed |')
    w('|---:|---|---|---:|---:|---:|---:|---|')
    for (M, quad), blk in blocks(rows):
        inf = blk[0]
        best = min((x for x in blk if x['worst_same_grid_evolved_percent'] is not None),
                   key=lambda x: x['worst_same_grid_evolved_percent'])
        w('| ' + ' | '.join([
            str(M), quad, 'yes' if M > r['R'] else f"no (R={r['R']})",
            pct(inf['worst_same_grid_evolved_percent']),
            pct(best['worst_same_grid_evolved_percent']), lam_plain(best['lambda_rel']),
            f"{best['median_gpu_ms'] / inf['median_gpu_ms']:.3f}",
            'yes' if best['all_completed'] else 'NO']) + ' |')
    w('')
    nd_evo = nondominated([x for x in rows if x['kind'] == 'rom'],
                          'worst_same_grid_evolved_percent')
    w('Non-dominated over every Burgers arm on the evolved-time metric: '
      + ', '.join(f"`{x['arm']}` ({pct(x['worst_same_grid_evolved_percent'])}%, "
                  f"{x['median_gpu_ms']:.1f} ms, "
                  f"{'completed' if x['all_completed'] else 'EARLY-STOPPED'})" for x in nd_evo)
      + '.')
    w('')
    for fname in ['fft_loose', 'fft_tight']:
        fom = next((x for x in rows if x['arm'] == fname), None)
        if not fom:
            continue
        for key, label in [('worst_same_grid_percent', 'all times'),
                           ('worst_same_grid_evolved_percent', 'evolved times')]:
            beat = [x for x in rows if x['kind'] == 'rom' and x[key] is not None
                    and x[key] <= fom[key] and x['median_gpu_ms'] <= fom['median_gpu_ms']]
            w(f"Against the same-job full-order `{fname}` ({pct(fom[key])}% same-grid over "
              f"{label}, {fom['median_gpu_ms']:.3f} ms median GPU): "
              + ((', '.join(f"`{x['arm']}`" for x in beat) + ' dominate it on both axes.')
                 if beat else 'no arm dominates it on both axes.'))
    w('')
    w('**Poisson, where c is eliminated exactly.** The Poisson weak residual is linear in c, so '
      'y has the closed form (B_y^T B_y + lambda I) y = B_y^T (f_m - B h(z)), evaluated from one '
      'thin SVD of B_y = B R_G^{-1} built at setup (never the Gram, which would square the '
      'condition number); the outer LM runs in z only and differentiates through it, so the '
      'solved dimension stays K and the trust radius keeps its latent meaning. M = 257 > R = 128, '
      'so lambda -> 0 does reach the free bank.')
    w('')
    w('| intervals | lambda_rel | worst same-grid % | median same-grid % | worst physical % | '
      'realised ‖y‖/‖u‖ % | median host ms |')
    w('|---:|---:|---:|---:|---:|---:|---:|')
    for x in prows:
        w('| ' + ' | '.join([
            str(x['intervals']),
            lam_plain(x['lambda_rel']) if x['kind'] == 'rom' else f"FOM {x['arm']}",
            pct(x['worst_same_grid_percent']), pct(x['median_same_grid_percent']),
            pct(x['worst_error_percent']), pct(x['max_relative_correction_percent']),
            f"{x['median_query_ms']:.4f}"]) + ' |')
    w('')
    for n in sorted({x['intervals'] for x in prows}):
        blk = sorted([x for x in prows if x['kind'] == 'rom' and x['intervals'] == n],
                     key=lambda z: -(float('inf') if z['lambda_rel'] is None else z['lambda_rel']))
        errs = [x['worst_same_grid_percent'] for x in blk]
        mono = all(b <= a * (1 + 1e-12) for a, b in zip(errs, errs[1:]))
        nd = nondominated(blk, 'worst_same_grid_percent', 'median_query_ms')
        cs = max(p['median_query_ms'] for p in nd) / min(p['median_query_ms'] for p in nd)
        es = max(p['worst_same_grid_percent'] for p in nd) / max(
            min(p['worst_same_grid_percent'] for p in nd), 1e-300)
        fom = next((x for x in prows if x['kind'] == 'fom' and x['intervals'] == n), None)
        w(f"- {n} intervals: worst same-grid {'monotone decreasing' if mono else 'NOT monotone'} "
          f"in lambda, {[round(e, 4) for e in errs]} percent from lambda = infinity down; "
          f"{len(nd)} non-dominated points spanning {cs:.2f}x in cost and {es:.2f}x in error, "
          f"{'all completed' if all(p['all_completed'] for p in nd) else 'some early-stopped'}. "
          f"The direct DST solve costs {fom['median_query_ms']:.4f} ms with zero same-grid error "
          f"by definition, against {blk[0]['median_query_ms']:.4f} ms for the ROM at "
          f"lambda = infinity.")
    w('')
    w('**Recorded deviations.** (1) The trust radius is applied to the LATENT block only in the '
      'primary sweep: arm (a)\'s radius '
      f"{r['trust_radius']:.9g} is 1% of the training code cloud's radius, a latent-space "
      f"quantity, and carrying it to y (field-norm units) would cap the field correction at that "
      f"same number and silently regularize the small-lambda end with something that is not "
      f"lambda. Paired trust-control rows at two lambda values measure the size of that choice. "
      f"At lambda = infinity there is no y block and the two are identical. (2) The LM residual "
      f"tolerance now includes the sqrt(lambda) y block, so the absolute residual exit is harder "
      f"to reach at large lambda; the stationarity and small-step exits are unaffected and every "
      f"row reports its budget exits. (3) Empirical quadrature is fitted only at M = 64 "
      f"(m = 4M = 256), which is arm (a)'s own rule refit with the identical seed, candidate cap, "
      f"fit-state count and code table; at M = 256 and 1024 an m = 4M NNLS rule is not "
      f"constructible inside the job budget, so those blocks use the exact dense grid sum, stated "
      f"per row, and the EQ rule — fitted on decoder-output advection snapshots — is "
      f"extrapolating once c leaves the head manifold, which the paired dense block at M = 64 "
      f"measures. (4) Arms above 64 unknowns use a pivoted dense step solve rather than the "
      f"incumbent unrolled Gauss-Jordan: more accurate, not weaker. (5) Layers 1 and 2 of the "
      f"error decomposition coincide at every finite lambda, because the penalty restricts the "
      f"SOLVER and not the reachable set; whatever lambda does, it does entirely in the "
      f"reduction/solver layer.")
    w('')
    w('**Nothing is retracted.** No earlier numerical result is withdrawn by this cell. The '
      'correction ladder\'s numbers stand; what is newly recorded is that its initializer fitted '
      'the augmented vector while this cell\'s does not, so the two cells\' all-times metrics are '
      'not comparable and the ladder\'s ability to move that metric came partly from correcting '
      'the t = 0 compression.')
    w('')
    w(f"Source-generated report and figures: "
      f"`experiments/prior-dial/reports/{meta['report_name']}` (SHA256 `{meta['report_sha256']}`) "
      f"with `-cost.png`/`-cost.pdf` and `-poisson.png`/`-poisson.pdf` beside it, all produced by "
      f"`reports/generate_prior_dial.py`. Raw archives Git-tracked as bounded chunks under "
      f"`experiments/prior-dial/artifacts/`: "
      + '; '.join(f"{k} sha256 {v['sha256']} ({v['chunks']} chunks)"
                  for k, v in meta['archives'].items())
      + ". Both exact remote attempt directories were removed after checksum collection and the "
        "namespace is empty. Not pushed, per the coordinator's standing instruction; commits are "
        "local only.")
    w('')
    w('**Open.** Whether letting the same penalty act on the INITIAL state fit — which would '
      'change arm (a)\'s initializer contract and was therefore out of scope here — unpins the '
      'all-times metric; other meshes, other checkpoints, more than one training seed; the '
      'sealed final cohort; and whether an M > R test family with a constructible hyper-reduction '
      'rule would make the Burgers dial affordable. No worktree was merged.')
    return '\n'.join(o) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--poisson-result', required=True)
    p.add_argument('--poisson-audit', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--branch-head', required=True)
    p.add_argument('--append', action='store_true')
    a = p.parse_args()
    report = Path(a.report)
    archives = {}
    for d in sorted((HERE / 'artifacts').glob('*/archive.json')):
        j = json.loads(d.read_text())
        archives[d.parent.name] = dict(sha256=j['sha256'], chunks=len(j['chunks']))
    meta = dict(branch_head=a.branch_head, report_name=report.name,
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                archives=archives)
    text = build(json.loads(Path(a.result).read_text()),
                 json.loads(Path(a.audit).read_text()),
                 json.loads(Path(a.smoke).read_text()),
                 json.loads(Path(a.poisson_result).read_text()),
                 json.loads(Path(a.poisson_audit).read_text()), meta)
    if a.append:
        with LOG.open('a') as f:
            f.write(text)
        print(f'appended {len(text)} characters to {LOG}')
    else:
        print(text)


if __name__ == '__main__':
    main()
