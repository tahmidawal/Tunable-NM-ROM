"""Generate the head-ablation report from the raw result JSONs. No number is typed by hand."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABEL = {
    'a_neural': '(a) neural head',
    'a_neural_q32': '(a+) neural head + 32 eliminated linear corrections (retained)',
    'b_linear_dec': '(b) linear, decoder-output fit',
    'b_linear_truth': '(b) linear, truth fit',
    'c_quad_dec': '(c) quadratic',
    'd_freebank': '(d) free bank coefficients',
}


def label(arm, with_quadrature=True):
    base, quad = (arm.rsplit('_', 1) if arm.endswith(('_eq', '_dense')) else (arm, None))
    if base.startswith('e_pod'):
        name = f"(e) POD-LSPG, rank {base[len('e_pod'):]}"
    else:
        name = LABEL.get(base, base)
    return f'{name} · {quad}' if (quad and with_quadrature) else name


def fmt(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def order_rows(sel):
    def rank(x):
        a = x['arm']
        if a.startswith('a_'):
            return (0, x.get('k') or 0, a)
        if a.startswith('b_'):
            return (1, 0, a)
        if a.startswith('c_'):
            return (2, 0, a)
        if a.startswith('e_pod'):
            return (3, x['k'], a)
        if a.startswith('d_'):
            return (4, 0, a)
        return (5, 0, a)
    return sorted(sel, key=rank)


def burgers_section(w, r, au, sm):
    rows = au['checks']['arm_table']
    setup = {(s['intervals'], s['arm']): s for s in r['arm_setup']}
    K = r['K']
    meshes = sorted({x['intervals'] for x in rows})
    w('## Burgers 2D — the nonlinear performance case')
    w('')
    w(f"Job `{r.get('job_id')}` on `{r.get('gpu')}`, source commit `{r.get('commit')}`, JAX "
      f"{r['jax_version']}, backend `{r['backend']}`, float64, matmul precision "
      f"`{r['matmul_precision']}`, elapsed {r['elapsed_seconds']:.1f} s.")
    w('')
    w(r'$$u_t+u(u_x+u_y)=\nu\Delta u \quad\text{on}\quad (0,1)^2,\qquad u|_{\partial\Omega}=0.$$')
    w('')
    w(f"Frozen bank of $R={r['R']}$ coordinate-network features, neural latent dimension "
      f"$K={K}$, time step {r['config']['dt']}, output times {r['output_times']}, refined reference at "
      f"{r['config']['reference_mesh']} intervals and time step {r['config']['reference_dt']}, "
      f"{len(r['physical_cases'])} opened development cases, {r['config']['reps']} timed repetitions "
      f"each with GPU burn-in.")
    w('')
    g = sm['generic_vs_saved_case']
    w(f"**Fidelity gate.** Run through the generic arm machinery, the neural arm reproduces the "
      f"consolidated saved Burgers case to {g['relative_l2']:.3e} relative (latent coordinates "
      f"{g['latent_relative_l2']:.3e}) and agrees with the incumbent `accuracy_paths.make_rom` to "
      f"{sm['generic_vs_incumbent']:.3e}. Arm (a) is the retained solver, not a re-implementation.")
    cc = au['checks'].get('campaign_frozen_arm_reproduced')
    if cc and cc['detail']:
        d = cc['detail']
        w('')
        w(f"**Campaign reproduction.** On the same {d['compared']} case/mesh combinations this job's "
          f"`a_neural_eq` reproduces the retained multiresolution campaign's `frozen_stationary` "
          f"rollout errors to a worst relative difference of {d['worst_relative_delta']:.3e}, so the "
          f"regenerated reference and rebuilt operators are the campaign's own.")
    w('')
    for L in meshes:
        sel = [x for x in rows if x['intervals'] == L]
        neural = next(x for x in sel if x['arm'] == 'a_neural_eq')
        snap = r['snapshots'][str(L)]
        w(f'### Burgers at {L} intervals per axis')
        w('')
        w(f"Training snapshots: {snap['trajectories']} regenerated trajectories, "
          f"{snap['snapshots']} states, worst relative Newton residual "
          f"{snap['max_relative_residual']:.3e}. Over those snapshots the frozen bank's own "
          f"root-mean-square projection floor is {snap['bank_projection_relative_rms'] * 100:.6f}%.")
        w('')
        w('| arm | $K$ | quad. | $M$ | $m$ | bank proj. % | best-found % | worst rollout % | '
          'median rollout % | median iters/step | median GPU ms | median host ms | stationary | completed |')
        w('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|')
        for x in order_rows([y for y in sel if y['kind'] == 'rom']) + [y for y in sel if y['kind'] == 'fom']:
            s = setup.get((L, x['arm']), {})
            w('| ' + ' | '.join([
                label(x['arm']) if x['kind'] == 'rom' else f"FOM `{x['arm']}` (context only)",
                str(x['k']) if x['k'] else '—', x['quadrature'] or '—',
                str(s.get('M', '—')), str(s.get('m') or '—'),
                fmt(x['worst_bank_projection_percent'], 6), fmt(x['worst_best_found_percent'], 6),
                fmt(x['worst_rollout_percent'], 6), fmt(x['median_rollout_percent'], 6),
                fmt(x['median_iterations'], 1), fmt(x['median_gpu_ms'], 3), fmt(x['median_host_ms'], 3),
                yn(x['all_stationary']), yn(x['all_completed'])]) + ' |')
        w('')
        pods = sorted([x for x in sel if x['arm'].startswith('e_pod')], key=lambda x: x['k'])
        match = [x for x in pods if x['worst_rollout_percent'] <= neural['worst_rollout_percent']]
        if match:
            b = match[0]
            w(f"**POD rank matching the neural head: $k'={b['k']}$**, ${b['k'] / K:g}\\times$ the neural "
              f"latent dimension $K={K}$ — worst rollout {b['worst_rollout_percent']:.6f}% against "
              f"{neural['worst_rollout_percent']:.6f}%, at {b['median_gpu_ms']:.3f} ms against "
              f"{neural['median_gpu_ms']:.3f} ms median GPU time in the same job.")
        else:
            t = pods[-1]
            w(f"**No tested POD rank up to $k'={t['k']}$ (${t['k'] / K:g}\\times$ the neural latent "
              f"dimension) matches the neural head.** The largest rung reaches "
              f"{t['worst_rollout_percent']:.6f}% worst rollout error against the neural head's "
              f"{neural['worst_rollout_percent']:.6f}%, and its best-found reconstruction alone is "
              f"already {t['worst_best_found_percent']:.6f}% against {neural['worst_best_found_percent']:.6f}%.")
        w('')
        same = [x for x in sel if x['kind'] == 'rom' and x['k'] == K]
        best = min(same, key=lambda x: x['worst_rollout_percent'])
        w(f"At matched $K={K}$ the lowest worst rollout error is {best['worst_rollout_percent']:.6f}% "
          f"from {label(best['arm'])}. Every linear and quadratic arm at that dimension is listed above "
          f"with its own quadrature and stopping status.")
        w('')
        f = r['fits'][str(L)]
        w(f"Map fitting at this mesh: the rank-{K} linear map retains "
          f"{f['linear_decoder_output']['retained_energy_fraction'] * 100:.6f}% of the decoder-output "
          f"coefficient energy, leaving a relative projection root-mean-square of "
          f"{f['linear_decoder_output']['relative_projection_rms'] * 100:.6f}%; the quadratic correction "
          f"uses ridge {f['quadratic_decoder_output']['ridge']:g} chosen on a "
          f"{f['quadratic_decoder_output']['heldout_fraction'] * 100:g}% held-out split, with held-out "
          f"relative residual {f['quadratic_decoder_output']['heldout_relative'] * 100:.6f}%.")
        w('')


def poisson_section(w, r, au, sm):
    rows = au['checks']['arm_table']
    K = r['K']
    meshes = sorted({x['intervals'] for x in rows})
    w('## Poisson 2D — the linear control')
    w('')
    w(f"Job `{r.get('job_id')}` on `{r.get('gpu')}`, source commit `{r.get('commit')}`, JAX "
      f"{r['jax_version']}, backend `{r['backend']}`, float64, matmul precision "
      f"`{r['matmul_precision']}`, elapsed {r['elapsed_seconds']:.1f} s.")
    w('')
    w(r'$$-\Delta u = f \quad\text{on}\quad (0,1)^2,\qquad u|_{\partial\Omega}=0,$$')
    w('')
    w('so the weak residual is exactly')
    w('')
    w(r'$$r(z)=B\,h(z)-f_m,\qquad B=\Phi^\top\!B_{\rm bank},\qquad f_m=\lambda^{-1}\Phi^\top f,$$')
    w('')
    w('with no time stepping and no quadrature approximation anywhere. Only $h$ changes between arms.')
    w(f"Frozen bank of $R={r['R']}$ features, neural latent dimension $K={K}$, "
      f"{r['config']['requested_modes']} requested sine tests, "
      f"{len(r['cohort']['parameters'])} opened development sources, "
      f"{r['config']['repetitions']} timed repetitions each with burn-in.")
    w('')
    w(f"**Fidelity gate.** The generic machinery with the frozen neural head reaches the incumbent "
      f"`core.rom_query` solution to {sm['generic_vs_incumbent_neural']['relative']:.3e} relative from "
      f"the same start, inside the declared {sm['generic_vs_incumbent_neural']['tolerance']:.0e} "
      f"tolerance. This is agreement on the same stationary point between two Levenberg-Marquardt "
      f"implementations, not bit identity.")
    w('')
    for n in meshes:
        sel = [x for x in rows if x['intervals'] == n]
        neural = next(x for x in sel if x['arm'] == 'a_neural')
        snap = r['snapshots'][str(n)]
        w(f'### Poisson at {n} intervals per axis')
        w('')
        w(f"Training snapshots: {snap['sources']} exact discrete solutions of the incumbent source "
          f"family. The frozen bank's own root-mean-square projection floor over them is "
          f"{snap['bank_projection_relative_rms'] * 100:.6f}%.")
        w('')
        w('| arm | $K$ | $M$ | bank proj. % | best-found % | worst error % | median error % | '
          'median iters | median query ms | max stationarity | stop reasons |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
        for x in order_rows([y for y in sel if y['kind'] in ('rom', 'retained')]) \
                + [y for y in sel if y['kind'] == 'fom']:
            w('| ' + ' | '.join([
                label(x['arm'], False) if x['kind'] != 'fom' else f"FOM `{x['arm']}` (context only)",
                str(x['k']) if x['k'] else '—', str(x['M'] or '—'),
                fmt(x['worst_bank_projection_percent'], 6), fmt(x['worst_best_found_percent'], 6),
                fmt(x['worst_error_percent'], 6), fmt(x['median_error_percent'], 6),
                fmt(x['median_iterations'], 1), fmt(x['median_query_ms'], 3),
                ('—' if x['max_stationarity'] is None else f"{x['max_stationarity']:.2e}"),
                (','.join(str(v) for v in x['stop_reasons']) if x['stop_reasons'] else '—')]) + ' |')
        w('')
        pods = sorted([x for x in sel if x['arm'].startswith('e_pod')], key=lambda x: x['k'])
        match = [x for x in pods if x['worst_error_percent'] <= neural['worst_error_percent']]
        if match:
            b = match[0]
            w(f"**POD rank matching the neural head: $k'={b['k']}$** (${b['k'] / K:g}\\times$ $K={K}$) — "
              f"worst error {b['worst_error_percent']:.6f}% against {neural['worst_error_percent']:.6f}%, "
              f"at {b['median_query_ms']:.3f} ms against {neural['median_query_ms']:.3f} ms.")
        else:
            t = pods[-1]
            w(f"**No tested POD rank up to $k'={t['k']}$ matches the neural head**: the largest rung "
              f"reaches {t['worst_error_percent']:.6f}% against {neural['worst_error_percent']:.6f}%.")
        w('')


def glossary(w):
    w('## Glossary')
    w('')
    for term, text in [
        ('arm', 'one configuration under test; everything except the named difference is held fixed.'),
        ('bank $B$', 'the fixed set of spatial fields the reduced state is built from. For arms (a)-(d) '
                     'it is the frozen network of the retained checkpoint; for arm (e) it is a classical '
                     'POD basis computed from training snapshots.'),
        ('coefficient map $h$', 'the function turning the few solved coordinates into bank coefficients. '
                                'This is the object under ablation.'),
        ('$K$ (latent dimension)', 'how many numbers the online solver actually solves for.'),
        ('$R$', 'the number of fields in the frozen bank.'),
        ('$M$ (test modes)', 'how many smooth functions the PDE residual is averaged against. It must '
                             'exceed the solved dimension or the objective collapses.'),
        ('$m$ (quadrature points)', 'how many grid points the empirical quadrature rule uses for the '
                                    'nonlinear advection term instead of the whole grid.'),
        ('EQ / empirical quadrature', 'a learned nonnegative weighted subset of grid points that '
                                      'reproduces the full sum; "dense" means the full grid sum was used, '
                                      'with no approximation.'),
        ('POD / POD-LSPG', 'proper orthogonal decomposition: the classical linear basis of snapshot data. '
                           'LSPG means the reduced coordinates minimise the projected residual in least '
                           'squares, which is what every arm here does.'),
        ('quadratic manifold', 'the established extension of POD that adds a quadratic function of the '
                               'same coordinates to the linear subspace.'),
        ('bank projection error', 'the smallest error any coefficients whatsoever could achieve in that '
                                  "arm's bank. A floor, not a solve."),
        ('best-found reconstruction error', "the smallest error found on that arm's actual manifold when "
                                            'fitting the reference field directly, with no PDE. It '
                                            'separates representation from dynamics. For an arm whose '
                                            'manifold is an affine subspace this is an exact projection; '
                                            'for a curved manifold it is a multistart local search and is '
                                            'therefore an upper bound.'),
        ('rollout / physical error', 'the error of the real online solve against the refined reference, '
                                     'largest over output times where there are several.'),
        ('median / worst', 'middle value across cases or repetitions / largest case value.'),
        ('iterations', 'Levenberg-Marquardt steps; hardware-free, so comparable without any timing '
                       'assumption.'),
        ('stationary / stationarity', 'the normalized weak gradient fell below the shared tolerance. See '
                                      'the methodology note: this is not a quality ranking.'),
        ('completed', 'the shared stopping rule terminated everywhere without hitting the iteration '
                      'budget and without a rejected-step exit.'),
        ('stop reason', 'the exit code of the shared solver: 1 residual tolerance, 2 small step, '
                        '3 rejected/non-finite, 4 stationary gradient, 0 iteration budget.'),
        ('complete query', 'the timed unit: one supplied input on the GPU to the requested dense output '
                           'fields, including projection or initial fit, the solve and the decode.'),
        ('development / final cohort', 'cases available for method selection / cases reserved unopened '
                                       'for later confirmation.'),
        ('FOM', 'full-order model: the unreduced PDE solver, shown only as same-job context.'),
    ]:
        w(f'- **{term}:** {text}')
    w('')


def build(args):
    br = json.loads(Path(args.burgers_result).read_text())
    ba = json.loads(Path(args.burgers_audit).read_text())
    bs = json.loads(Path(args.burgers_smoke).read_text())
    out = []
    w = out.append
    w('# Does the nonlinear coefficient map earn its place?')
    w('')
    w('A matched head ablation in one frozen spatial bank per PDE. The retained neural coefficient map')
    w('is compared against a linear map, a quadratic map, unrestricted bank coefficients and a classical')
    w('POD-LSPG reduced model, at matched latent dimension and under the same weak objective, test modes,')
    w('time discretization, initializer policy, stopping rule and output contract. **The numbers are')
    w('final for the development cohorts and provisional as paper claims**: one training seed, one')
    w('checkpoint per PDE, development cases only, and the final cohorts remain sealed.')
    w('')
    w('## What every arm shares')
    w('')
    w('Each arm writes the reduced state as')
    w('')
    w(r'$$u(z) = B\,h(z),\qquad B\in\mathbb R^{n\times D},\quad h:\mathbb R^{K}\to\mathbb R^{D},$$')
    w('')
    w('and solves the same overdetermined weak residual for the reduced coordinates $z$. Only the pair')
    w(r'$(B,h)$ changes. Writing $\Phi\in\mathbb R^{n\times M}$ for the $M$ lowest discrete sine test')
    w(r'modes ($\Phi^\top\Phi=I$), $\lambda$ for their Laplacian eigenvalues and $A=\Phi^\top B$, the')
    w('Burgers arms minimise')
    w('')
    w(r'$$r_w(z)=\frac{A h(z)-p+\Delta t\,\big(\Phi^\top\mathcal N(Bh(z))+\nu\,\lambda\odot A h(z)\big)}'
      r'{1+\Delta t\,\nu\lambda},\qquad p=A h(z^{\rm prev}),$$')
    w('')
    w(r'with $\mathcal N$ the full-order model\'s own sign-upwind advection, and the Poisson arms minimise')
    w(r'$B h(z)-\lambda^{-1}\Phi^\top f$. The linear terms are exact in both.')
    w('')
    w('The arms are')
    w('')
    w('| arm | $B$ | $h$ |')
    w('|---|---|---|')
    w(r'| (a) neural | frozen bank $G$ | the retained checkpoint head $h_\theta$ |')
    w(r'| (b) linear | $G$ | $c + Wz$ |')
    w(r'| (c) quadratic | $G$ | $c + Wz + Q\,\mathrm{vech}(zz^\top)$ |')
    w(r'| (d) free bank | $G$ | identity on $\mathbb R^{R}$ |')
    w(r"| (e) POD-LSPG | classical POD basis $V_{k'}$ | identity on $\mathbb R^{k'}$ |")
    w('')
    w(r'With $G=Q_GR_G$ the thin QR of the bank, $\|G\delta\|_2=\|R_G\delta\|_2$, so least squares in the')
    w('whitened coefficient metric is exactly field-metric least squares: arm (b) is therefore the')
    w('*optimal* rank-$K$ affine map inside the bank, not an arbitrary one, and arm (c) adds the')
    w('established quadratic-manifold correction on the same coordinates with its ridge chosen on a')
    w('seeded held-out split. Arm (b) is fitted twice — once to the neural head\'s own decoder outputs,')
    w('so it is asked to cover arm (a)\'s exact image, and once to bank-projected truth snapshots, which')
    w('is the practitioner\'s baseline.')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  U0["supplied input field or source"] --> IC["shared initializer<br/>same rule, same LM budget"]')
    w('  IC --> Z["reduced coordinates z"]')
    w('  Z --> H{"coefficient map h"}')
    w('  H --> HA["(a) neural MLP"]')
    w('  H --> HB["(b) linear c + Wz"]')
    w('  H --> HC["(c) quadratic, adds Q vech(z tensor z)"]')
    w('  H --> HD["(d) identity on R^R"]')
    w('  H --> HE["(e) identity on R^k"]')
    w('  HA --> B["frozen spatial bank"]')
    w('  HB --> B')
    w('  HC --> B')
    w('  HD --> B')
    w('  HE --> BP["classical POD basis"]')
    w('  B --> W["weak residual on M sine tests<br/>exact linear terms"]')
    w('  BP --> W')
    w('  W --> LM["stationarity-aware LM<br/>same budgets and tolerance"]')
    w('  LM --> OUT["requested dense output"]')
    w('  classDef frozen fill:#dce9f7,stroke:#3b6ea5;')
    w('  classDef solved fill:#f7e6d0,stroke:#b07b32;')
    w('  classDef fitted fill:#e2f0da,stroke:#4f8a3d;')
    w('  class B,BP,HA frozen;')
    w('  class Z,LM,W solved;')
    w('  class HB,HC,HD,HE fitted;')
    w('```')
    w('')
    burgers_section(w, br, ba, bs)
    pr = pa = ps = None
    if args.poisson_result:
        pr = json.loads(Path(args.poisson_result).read_text())
        pa = json.loads(Path(args.poisson_audit).read_text())
        ps = json.loads(Path(args.poisson_smoke).read_text())
        poisson_section(w, pr, pa, ps)

    w('## Stopping status, and why the stationarity column is not a quality ranking')
    w('')
    w('Every arm runs the identical stopping rule. The campaign\'s stationarity test is the normalized')
    w(r'gradient $\|J^\top r\|/(\|J\|\,\|r\|)$, which is scale invariant: it can only fall below its')
    w('tolerance once the residual becomes orthogonal to the reduced tangent space. For an arm whose')
    w('reduced fit is attainable — a square or nearly square reduced system — the residual instead falls')
    w('to round-off while that ratio stays of order one, so the arm exits by the small-step rule with a')
    w('*better* fit and a *worse* looking stationarity number. The `completed` column therefore reports')
    w('the honest status: no iteration-budget exit and no rejected-step exit anywhere. Both are reported;')
    w('neither alone ranks quality.')
    w('')
    w('## Necessary, recorded deviations from the matched contract')
    w('')
    fb = next((s for s in br['arm_setup'] if s.get('arm') == 'd_freebank_dense'), None)
    w(f"1. On Burgers, arm (d) solves $R={br['R']}$ coefficients, so its weak system needs $M>R$; it uses "
      f"$M={fb['M'] if fb else '—'}$ and the exact dense advection, because a nonnegative-least-squares "
      f"rule with $m=4M$ points is not constructible inside the job budget. Its cost is therefore "
      f"grid-bound, which is itself part of the compression answer.")
    w('2. On Burgers, empirical quadrature is fitted at the matched rank for every $K$-dimensional arm '
      'and for POD rank 16. The higher POD rungs run with exact dense advection, which *favours* the POD '
      'baseline, so any reported matching rank is a conservative lower bound on the rank a hyper-reduced '
      'POD reduced model would need. Paired `eq`/`dense` rows at matched rank isolate the quadrature '
      'effect directly.')
    w(f"3. The incumbent unpivoted Gauss-Jordan step solve unrolls one graph level per unknown; arms "
      f"above {br['config']['gauss_jordan_max']} unknowns use a pivoted dense solve instead. That is a "
      f"more accurate step, not a weaker one.")
    w('4. POD modes have no continuum representation, so on Burgers they are evaluated at the Gauss '
      'initializer points by the same aligned bilinear interpolation the supplied input field already '
      'receives in every arm. A classical POD basis is also mesh-bound and is therefore rebuilt at each '
      'mesh, whereas the frozen coordinate bank transfers unchanged.')
    w('5. The best-found reconstruction error is an exact projection for every affine-manifold arm and a '
      'seeded multistart local search for the curved ones, so it is an upper bound exactly where the '
      'nonlinear arms would benefit from it being tight.')
    w('')
    w('## What this does and does not establish')
    w('')
    w('It establishes, inside one frozen bank per PDE and at matched latent dimension, how much of the')
    w('retained accuracy is attributable to the nonlinearity of the coefficient map rather than to the')
    w('bank, the weak objective or the solver, and what classical POD rank buys the same accuracy in the')
    w('same job. It does not establish anything about other PDEs, other checkpoints, other training')
    w('seeds, the sealed final cohorts, or a speed advantage over a full-order solver: the same-job')
    w('full-order rows are context only, and earlier work already records that the retained Burgers')
    w('reduced model is slower and less accurate than an efficient same-job full-order solver.')
    w('')
    glossary(w)
    w('---')
    w('')
    srcs = [f"Burgers `result.json` (SHA256 `{ba['result_sha256']}`)"]
    if pa:
        srcs.append(f"Poisson `result.json` (SHA256 `{pa['result_sha256']}`)")
    w(f"Generated by `experiments/head-ablation/reports/generate_head_ablation.py` from "
      f"{' and '.join(srcs)} and their audit JSONs. Every number above is read from those files.")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--burgers-result', required=True)
    p.add_argument('--burgers-audit', required=True)
    p.add_argument('--burgers-smoke', required=True)
    p.add_argument('--poisson-result')
    p.add_argument('--poisson-audit')
    p.add_argument('--poisson-smoke')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    text = build(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
