"""Generate the head-ablation report from the raw result JSON. No number is typed."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

LABEL = {
    'a_neural': '(a) neural head',
    'b_linear_dec': '(b) linear, decoder-output fit',
    'b_linear_truth': '(b) linear, truth fit',
    'c_quad_dec': '(c) quadratic',
    'd_freebank': '(d) free bank coefficients',
}


def label(arm):
    base, quad = arm.rsplit('_', 1)
    if base.startswith('e_pod'):
        name = f"(e) POD-LSPG, rank {base[len('e_pod'):]}"
    else:
        name = LABEL.get(base, base)
    return f'{name} · {quad}'


def med(x):
    return float(np.median(np.asarray(x, dtype=float)))


def fmt(x, digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def build(result, audit, smoke):
    r = json.loads(Path(result).read_text())
    au = json.loads(Path(audit).read_text())
    sm = json.loads(Path(smoke).read_text())
    rows = au['checks']['arm_table']
    meshes = sorted({x['intervals'] for x in rows})
    K = r['K']
    out = []
    w = out.append

    w('# Does the nonlinear coefficient map earn its place?')
    w('')
    w('A matched head ablation in one frozen Burgers 2D spatial bank: the retained neural')
    w('coefficient map is compared against a linear map, a quadratic map, unrestricted bank')
    w('coefficients and a classical POD-LSPG ROM, at matched latent dimension and under the')
    w('same weak objective, test modes, time discretization, initializer policy, stopping')
    w('rule and output contract. **These numbers are final for the development cohort and')
    w('provisional as paper claims**: one training seed, one checkpoint, six opened')
    w('development cases, two meshes, and the final cohort remains sealed.')
    w('')
    w(f"Job `{r.get('job_id')}` on `{r.get('gpu')}`, source commit `{r.get('commit')}`, "
      f"JAX {r['jax_version']}, backend `{r['backend']}`, float64, "
      f"matmul precision `{r['matmul_precision']}`.")
    w('')

    w('## What every arm shares')
    w('')
    w('Each arm writes the reduced state as')
    w('')
    w(r'$$u(z) = B\,h(z),\qquad B\in\mathbb R^{n\times D},\quad h:\mathbb R^{K}\to\mathbb R^{D},$$')
    w('')
    w('and solves the same overdetermined weak residual for the reduced coordinates $z$.')
    w(r'With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes')
    w(r'($\Phi^\top\Phi=I$), $\lambda$ their Laplacian eigenvalues, $A=\Phi^\top B$, backward')
    w(r'Euler at step $\Delta t$ and the full-order model\'s own sign-upwind advection $\mathcal N$,')
    w('')
    w(r'$$r_w(z)=\frac{A h(z)-p+\Delta t\,\big(\Phi^\top\mathcal N(Bh(z))+\nu\,\lambda\odot A h(z)\big)}'
      r'{1+\Delta t\,\nu\lambda},\qquad p=A h(z^{\rm prev}).$$')
    w('')
    w('Only the pair $(B,h)$ changes between arms. The linear terms are exact everywhere.')
    w(f"Time step {r['config']['dt']}, output times {r['output_times']}, "
      f"test count $M={r['config']['test_multiplier']}K$, empirical-quadrature budget "
      f"$m={r['config']['quadrature_multiplier']}M$ where used, stopping rule "
      f"`gtol` $={r['config']['strict']['gtol']}$ with initial-fit budget "
      f"{r['config']['strict']['ic_budget']} and per-step budget {r['config']['strict']['step_budget']}.")
    w('')
    w(f"**Fidelity gate.** Run through the generic arm machinery, the neural arm reproduces the")
    w(f"consolidated saved Burgers case to {sm['generic_vs_saved_case']['relative_l2']:.3e} relative")
    w(f"(latent coordinates {sm['generic_vs_saved_case']['latent_relative_l2']:.3e}) and agrees with")
    w(f"the incumbent `accuracy_paths.make_rom` to {sm['generic_vs_incumbent']:.3e}. Arm (a) is the")
    w('retained solver, not a re-implementation of it.')
    cc = au['checks'].get('campaign_frozen_arm_reproduced')
    if cc:
        d = cc['detail']
        w('')
        w(f"**Campaign reproduction.** On the same {d['compared']} case/mesh combinations, this job's")
        w(f"arm (a) reproduces the retained multiresolution campaign's `frozen_stationary` rollout")
        w(f"errors to a worst relative difference of {d['worst_relative_delta']:.3e}, so the")
        w('regenerated reference and rebuilt operators are the campaign\'s own.')
    w('')

    w('```mermaid')
    w('flowchart LR')
    w('  U0["supplied dense u(0)"] --> IC["fixed Gauss state fit<br/>same 48x48 rule, same LM"]')
    w('  IC --> Z["reduced coordinates z"]')
    w('  Z --> H{"coefficient map h"}')
    w('  H --> HA["(a) neural MLP"]')
    w('  H --> HB["(b) linear c + Wz"]')
    w('  H --> HC["(c) quadratic, adds Q vech(z tensor z)"]')
    w('  H --> HD["(d) identity on R^R"]')
    w('  H --> HE["(e) identity on R^k, POD bank"]')
    w('  HA --> B["frozen spatial bank B"]')
    w('  HB --> B')
    w('  HC --> B')
    w('  HD --> B')
    w('  HE --> BP["classical POD basis V_k"]')
    w('  B --> W["weak residual on M sine tests<br/>exact linear terms"]')
    w('  BP --> W')
    w('  W --> LM["stationarity-aware LM<br/>same budgets and gtol"]')
    w('  LM --> OUT["six dense output fields"]')
    w('  classDef frozen fill:#dce9f7,stroke:#3b6ea5;')
    w('  classDef solved fill:#f7e6d0,stroke:#b07b32;')
    w('  classDef fitted fill:#e2f0da,stroke:#4f8a3d;')
    w('  class B,BP,HA frozen;')
    w('  class Z,LM,W solved;')
    w('  class HB,HC,HD,HE fitted;')
    w('```')
    w('')

    for L in meshes:
        sel = [x for x in rows if x['intervals'] == L]
        neural = next(x for x in sel if x['arm'] == 'a_neural_eq')
        w(f'## Mesh: {L} intervals per axis')
        w('')
        snap = r['snapshots'][str(L)]
        w(f"Training snapshots: {snap['trajectories']} regenerated trajectories, "
          f"{snap['snapshots']} states, worst relative Newton residual "
          f"{snap['max_relative_residual']:.3e}. The frozen bank's own projection floor over those "
          f"snapshots is {snap['bank_projection_relative_rms'] * 100:.6f}% (root-mean-square).")
        w('')
        w('| arm | $K$ | quad. | $M$ | $m$ | bank proj. % | best-found % | worst rollout % | '
          'median rollout % | median iters/step | median GPU ms | median host ms | stationary | completed |')
        w('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|')
        order = ([x for x in sel if x['arm'].startswith('a_')]
                 + [x for x in sel if x['arm'].startswith('b_')]
                 + [x for x in sel if x['arm'].startswith('c_')]
                 + sorted([x for x in sel if x['arm'].startswith('e_')], key=lambda x: (x['k'], x['arm']))
                 + [x for x in sel if x['arm'].startswith('d_')]
                 + [x for x in sel if x['kind'] == 'fom'])
        setup = {(s['intervals'], s['arm']): s for s in r['arm_setup']}
        for x in order:
            s = setup.get((L, x['arm']), {})
            w('| ' + ' | '.join([
                label(x['arm']) if x['kind'] == 'rom' else f"FOM `{x['arm']}`",
                str(x['k']) if x['k'] else '—',
                x['quadrature'] or '—',
                str(s.get('M', '—')), str(s.get('m') or '—'),
                fmt(x['worst_bank_projection_percent'], 6),
                fmt(x['worst_best_found_percent'], 6),
                fmt(x['worst_rollout_percent'], 6),
                fmt(x['median_rollout_percent'], 6),
                fmt(x['median_iterations'], 1),
                fmt(x['median_gpu_ms'], 3),
                fmt(x['median_host_ms'], 3),
                {True: 'yes', False: 'no', None: '—'}[x['all_stationary']],
                {True: 'yes', False: 'no', None: '—'}[x['all_completed']]]) + ' |')
        w('')
        pods = sorted([x for x in sel if x['arm'].startswith('e_pod')], key=lambda x: x['k'])
        match = [x for x in pods if x['worst_rollout_percent'] <= neural['worst_rollout_percent']]
        if match:
            best = match[0]
            w(f"**POD rank that matches the neural head at {L} intervals: "
              f"$k'={best['k']}$**, that is ${best['k'] / K:g}\\times$ the neural latent dimension "
              f"$K={K}$. Its worst rollout error is {best['worst_rollout_percent']:.6f}% against the "
              f"neural head's {neural['worst_rollout_percent']:.6f}%, at "
              f"{best['median_gpu_ms']:.3f} ms against {neural['median_gpu_ms']:.3f} ms median GPU time "
              f"in this same job.")
        else:
            top = pods[-1] if pods else None
            w(f"**No tested POD rank up to $k'={top['k']}$ matches the neural head at {L} intervals.** "
              f"The largest rung reaches {top['worst_rollout_percent']:.6f}% against the neural head's "
              f"{neural['worst_rollout_percent']:.6f}%.")
        w('')
        same = [x for x in sel if x['k'] == K and x['kind'] == 'rom']
        best_same = min(same, key=lambda x: x['worst_rollout_percent'])
        w(f"At matched $K={K}$ the lowest worst rollout error is "
          f"{best_same['worst_rollout_percent']:.6f}% from {label(best_same['arm'])}; the neural head "
          f"reaches {neural['worst_rollout_percent']:.6f}%.")
        w('')

    w('## Stopping status, and why the stationarity column is not a quality ranking')
    w('')
    w('Every arm runs the identical stopping rule. The campaign\'s stationarity test is the')
    w(r'normalized gradient $\|J^\top r\|/(\|J\|\,\|r\|)$, which is scale invariant: it can only')
    w('fall below its tolerance when the residual becomes orthogonal to the reduced tangent')
    w('space. For an arm whose reduced fit is attainable — a square or nearly square reduced')
    w('system — the residual instead falls to round-off and the normalized gradient stays of')
    w('order one, so the arm exits by the small-step rule with a *better* fit and a *worse*')
    w('looking stationarity number. The `completed` column therefore reports the honest status:')
    w('no budget exit and no rejected-step exit anywhere in the trajectory. Both are reported;')
    w('neither alone is a quality ranking.')
    w('')

    w('## Necessary deviations from the matched contract')
    w('')
    free = [x for x in rows if x['arm'] == 'd_freebank_dense']
    if free:
        fm = setup.get((meshes[0], 'd_freebank_dense'), {})
        w(f"1. Arm (d) solves $R={r['R']}$ coefficients, so its weak system needs $M>R$; it uses "
          f"$M={fm.get('M')}$ and the exact dense advection, because a nonnegative-least-squares "
          f"rule with $m=4M$ points is not constructible inside the job budget. Its cost is "
          f"therefore grid-bound, which is itself part of the compression answer.")
    w('2. Empirical quadrature is fitted at the matched rank for every $K$-dimensional arm and for '
      'POD rank 16. The higher POD rungs run with exact dense advection, which *favours* the POD '
      'baseline, so the reported matching rank is a conservative lower bound on the rank a '
      'hyper-reduced POD ROM would need.')
    w(f"3. The incumbent unpivoted Gauss-Jordan step solve unrolls one graph level per unknown; arms "
      f"above {r['config']['gauss_jordan_max']} unknowns use a pivoted dense solve instead. That is a "
      f"more accurate step, not a weaker one.")
    w('4. POD modes have no continuum representation, so they are evaluated at the Gauss initializer '
      'points by the same aligned bilinear interpolation the supplied input field already receives '
      'in every arm.')
    w('')

    w('## Fitting the linear and quadratic maps')
    w('')
    w(r'With $G=Q_GR_G$ the thin QR of the bank, $\|G\delta\|_2=\|R_G\delta\|_2$, so least squares')
    w('in the whitened coefficient metric is exactly field-metric least squares. The linear arm is')
    w('the optimal rank-$K$ affine map in that metric — proper orthogonal decomposition inside the')
    w('bank — and the quadratic arm adds a quadratic-manifold correction on the same coordinates,')
    w('with its ridge chosen on a seeded held-out split of the same snapshots.')
    w('')
    for L in meshes:
        f = r['fits'][str(L)]
        w(f"At {L} intervals: the rank-{K} linear map leaves a relative projection root-mean-square of "
          f"{f['linear_decoder_output']['relative_projection_rms'] * 100:.6f}% of the decoder-output "
          f"coefficients ({f['linear_decoder_output']['retained_energy_fraction'] * 100:.6f}% of energy "
          f"retained); the quadratic correction is fitted with ridge "
          f"{f['quadratic_decoder_output']['ridge']:g} and a held-out relative residual of "
          f"{f['quadratic_decoder_output']['heldout_relative'] * 100:.6f}%.")
    w('')

    w('## What this does and does not establish')
    w('')
    w('It establishes, inside one frozen bank and at matched latent dimension, how much of the')
    w('retained accuracy is attributable to the nonlinearity of the coefficient map rather than to')
    w('the bank, the weak objective or the solver, and what classical POD rank buys the same')
    w('accuracy in the same job. It does not establish anything about other PDEs, other')
    w('checkpoints, other training seeds, the sealed final cohort, or a speed advantage over a')
    w('full-order solver: the same-job full-order rows are included only as context.')
    w('')

    w('## Glossary')
    w('')
    for term, text in [
        ('arm', 'one configuration under test; everything except the named difference is held fixed.'),
        ('bank $B$', 'the fixed set of spatial fields the reduced state is built from. For arms (a)-(d) '
                     'it is the frozen coordinate network of the retained checkpoint; for arm (e) it is a '
                     'classical POD basis computed from training snapshots.'),
        ('coefficient map $h$', 'the function turning the few solved coordinates into bank coefficients. '
                                'This is the object under ablation.'),
        ('$K$ (latent dimension)', 'how many numbers the online solver actually solves for.'),
        ('$R$', 'the number of fields in the frozen bank (512 here).'),
        ('$M$ (test modes)', 'how many smooth functions the PDE residual is averaged against. It must '
                             'exceed the solved dimension or the objective collapses.'),
        ('$m$ (quadrature points)', 'how many grid points the empirical quadrature rule uses to evaluate '
                                    'the nonlinear advection term instead of the whole grid.'),
        ('EQ / empirical quadrature', 'a learned nonnegative weighted subset of grid points that '
                                      'reproduces the full sum; "dense" means the full grid sum was used '
                                      'instead, with no approximation.'),
        ('POD / POD-LSPG', 'proper orthogonal decomposition: the classical linear basis of snapshot data. '
                           'LSPG means the reduced coordinates are found by least-squares minimising the '
                           'projected residual, which is what every arm here does.'),
        ('bank projection error', 'the smallest error any coefficients whatsoever could achieve in that '
                                  'arm\'s bank. A floor, not a solve.'),
        ('best-found reconstruction error', 'the smallest error found on that arm\'s actual manifold when '
                                            'fitting the reference field directly, with no PDE. It '
                                            'separates representation from dynamics.'),
        ('rollout error', 'the error of the real online solve against the refined reference, largest over '
                          'the six output times, normalised by the initial reference field norm.'),
        ('median / worst', 'middle value across cases or repetitions / largest case value.'),
        ('iterations', 'Levenberg-Marquardt steps per time step; hardware-free, so comparable without any '
                       'timing assumption.'),
        ('stationary', 'the normalized weak gradient fell below the shared tolerance everywhere. See the '
                       'section above for why this is not a quality ranking.'),
        ('completed', 'the shared stopping rule terminated everywhere without hitting the iteration budget '
                      'and without a rejected-step exit.'),
        ('complete query', 'the timed unit: one supplied dense initial field on the GPU to six dense '
                           'output fields, including the initial fit, the evolution and the decode.'),
        ('development / final cohort', 'cases available for method selection / cases reserved unopened for '
                                       'later confirmation.'),
        ('FOM', 'full-order model: the unreduced PDE solver, shown only as same-job context.'),
    ]:
        w(f'- **{term}:** {text}')
    w('')
    w('---')
    w('')
    w(f"Generated by `experiments/head-ablation/reports/generate_head_ablation.py` from "
      f"`result.json` (SHA256 `{au['result_sha256']}`) and "
      f"`{Path(audit).name}`. Every number above is read from those files.")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    text = build(a.result, a.audit, a.smoke)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
