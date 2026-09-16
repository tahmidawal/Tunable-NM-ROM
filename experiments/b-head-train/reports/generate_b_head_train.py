"""Generate the b-head-train report from the audited JSONs. No number is typed.

Usage:
  python reports/generate_b_head_train.py \
      --train-result runs/train01/archive/output/result.json \
      --train-audit checks/train01-audit.json \
      --eval-result runs/eval01/archive/output/result.json \
      --eval-audit checks/eval01-audit.json \
      --out reports/2026-09-16-training-the-burgers-head.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def f(x, n=4):
    return '--' if x is None else f'{x:.{n}f}'


def g(x, n=4):
    return '--' if x is None else f'{x:.{n}g}'


def pct(x, n=4):
    return '--' if x is None else f'{100 * x:.{n}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-result', required=True)
    p.add_argument('--train-audit', required=True)
    p.add_argument('--eval-result', required=True)
    p.add_argument('--eval-audit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    tr = json.loads(Path(a.train_result).read_text())
    ta = json.loads(Path(a.train_audit).read_text())
    ev = json.loads(Path(a.eval_result).read_text())
    ea = json.loads(Path(a.eval_audit).read_text())
    rows = ea['arm_table']
    verdicts = {v['arm']: v for v in ea['verdicts']}
    sel = tr['selection']
    tcfg, ecfg = tr['config'], ev['config']
    inc = next(x for x in rows if x['arm'] == ecfg['incumbent_arm'])
    crit = ea['checks']['success_criteria_evaluated']['detail']['criteria']
    out = []
    w = out.append

    w('# Can training alone move the Burgers head off its 2.5 % reconstruction?')
    w('')
    w(f"One frozen separable bank, one architecture, three training levers: how much data the "
      f"head sees, what it is asked to minimise, and how many latent coordinates it has. Every "
      f"number below is **final for the {ecfg['eval_cases'] + ecfg['eval_fresh_cases']} "
      f"development cases at {ecfg['intervals']} intervals** and **provisional as a paper claim**: "
      f"one mesh, one training seed per arm, final cohort sealed. The predeclared protocol is "
      f"`DESIGN.md`; the incumbent checkpoint is arm (a) and was re-run in the evaluation job as "
      f"the control.")
    w('')
    w(f"Training job `{tr.get('job_id')}` on `{tr.get('gpu')}` ({tr['elapsed_seconds'] / 3600:.2f} h) "
      f"and evaluation job `{ev.get('job_id')}` on `{ev.get('gpu')}` "
      f"({ev['elapsed_seconds'] / 3600:.2f} h), source commits `{tr.get('commit')}` and "
      f"`{ev.get('commit')}`, JAX {ev['jax_version']}, backend `{ev['backend']}`, float64, matmul "
      f"precision `{ev['matmul_precision']}`.")
    w('')

    # ------------------------------------------------------------- headline --
    passing = [v for v in ea['verdicts'] if v['success']]
    quad = {x['arm']: x['quadrature'] for x in rows}
    incq = quad.get(ecfg['incumbent_arm'])
    # the headline compares like for like: trained checkpoints in the incumbent's
    # own quadrature. The paired dense arms are quadrature controls, not candidates.
    trained = [v for v in ea['verdicts']
               if v['checkpoint'] != 'incumbent' and quad.get(v['arm']) == incq]
    best = min(trained or ea['verdicts'], key=lambda v: v['best_found_percent'])
    w('## The answer')
    w('')
    w(f"**Pre-registered success** required a frozen checkpoint with best-found reconstruction "
      f"worst $< {crit['best_found']}\\,\\%$, solved worst same-grid $< {crit['same_grid']}\\,\\%$, "
      f"every solve stationary, at $\\le {crit['cost_factor']}\\times$ the incumbent's median GPU "
      f"query cost. "
      + (f"**{len(passing)} of {len(ea['verdicts'])} arms pass**: "
         + ', '.join(f"`{v['arm']}`" for v in passing) + '.'
         if passing else
         f"**No arm passes.** {len(ea['verdicts'])} arms were evaluated."))
    w('')
    w(f"The best best-found reconstruction reached by a TRAINED checkpoint is `{best['arm']}` at "
      f"{best['best_found_percent']:.4f} % against the incumbent's "
      f"{verdicts[ecfg['incumbent_arm']]['best_found_percent']:.4f} % and the frozen bank's "
      f"projection floor of {f(inc['worst_bank_projection_percent'])} %. Its solved worst "
      f"same-grid error is {best['same_grid_percent']:.4f} % against the incumbent's "
      f"{verdicts[ecfg['incumbent_arm']]['same_grid_percent']:.4f} %, at "
      f"{best['cost_factor']:.3f}x the incumbent's cost.")
    w('')
    w(f"**Data-vs-capacity diagnostic: {sel['diagnostic_verdict']}.** At fixed $K = "
      f"{tr['K_incumbent']}$ the held-out representation oracle's worst value along "
      f"$N_{{\\rm traj}} = {[n for n, _ in sel['density_curve']]}$ is "
      f"{[round(v, 6) for _, v in sel['density_curve']]}; monotone decreasing: "
      f"{'yes' if sel['monotone'] else 'no'}; relative improvement from the "
      f"{sel['saturation_rung']}-trajectory rung to {sel['next_rung']} is "
      f"{100 * sel['relative_gain_mid_to_next']:.2f} % and to {sel['top_rung']} is "
      f"{100 * sel['relative_gain_mid_to_top']:.2f} %, against a declared saturation threshold of "
      f"{100 * sel['saturation_fraction']:.0f} %.")
    w('')

    # --------------------------------- what the incumbent was trained on -----
    ick = next(c for c in ev['checkpoints'] if c['arm'] == 'incumbent')
    icfg = ick['cfg']
    n_inc = icfg.get('hfit_n_traj')
    inc_bf = verdicts[ecfg['incumbent_arm']]['best_found_percent']
    w(f"## The incumbent head was trained on {n_inc} trajectories, not 128")
    w('')
    w(f"This has to come before any density number, because the lab log and the framing of this "
      f"question both assumed 128. **128 is the head-ablation configuration's snapshot draw** "
      f"(`config-ablation.json: train_trajectories = 128`), used there to build POD bases, to fit "
      f"the linear and quadratic coefficient maps, and to fit the empirical-quadrature rule. It is "
      f"not what trained the head. The incumbent checkpoint `{ick['file']}` carries its own "
      f"training record: `hfit_n_traj = {n_inc}`, `hfit_extra_seed = "
      f"{icfg.get('hfit_extra_seed')}`, `hfit_extra_traj = {icfg.get('hfit_extra_traj')}`, "
      f"`hfit_arm = {icfg.get('hfit_arm')!r}`, `hfit_source_ckpt = "
      f"{icfg.get('hfit_source_ckpt')!r}` — that is the canonical "
      f"`sample_params(seed={tcfg['canonical_seed']})` draw of "
      f"{n_inc - (icfg.get('hfit_extra_traj') or 0)} plus {icfg.get('hfit_extra_traj')} appended "
      f"from seed {icfg.get('hfit_extra_seed')}, and it carries {ick['codes']} auto-decoder codes, "
      f"one per fitted state.")
    w('')
    w(f"`burgers2d_film.sample_params` and `engines.params_draw` are the *same* sequential RNG draw "
      f"over the *same* ranges, so that data set is reproducible in this lane, and the density "
      f"ladder here is a **nested prefix** of it at {[n for n, _ in sel['density_curve']]} "
      f"trajectories. Two consequences for how the numbers below should be read. First, density is "
      f"the only variable across the rungs, and the top rung is a like-for-like retrain of the "
      f"incumbent inside this pipeline rather than a different experiment. Second, the incumbent's "
      f"{inc_bf:.4f} % best-found reconstruction is **not** the result of a data-starved fit: it is "
      f"what {n_inc} trajectories already bought. Any reading of the head gap that assumed 128 "
      f"trajectories — including the expectation that simply adding data would close it — has to be "
      f"revisited against the ladder below.")
    w('')

    # ------------------------------------------------------------ the shape --
    w('## What is being trained, and what is frozen at query time')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  subgraph OFFLINE["offline - this cell"]')
    w('    D["FOM trajectories<br/>engines.params_draw + make_fom"]:::data')
    w('    G["bank g(x)<br/>random-Fourier MLP"]:::frozen')
    w('    P["span coefficients c*<br/>identity (*)"]:::data')
    w('    H["head h(z) = MLP + linear skip"]:::trained')
    w('    Z["auto-decoder codes Z"]:::trained')
    w('    C["frozen checkpoint<br/>hashed"]:::frozen')
    w('  end')
    w('  subgraph ONLINE["online - unchanged arm (a) contract"]')
    w('    Q["EQ rule, cold initializer,<br/>trust radius, LM stopping rule"]:::frozen')
    w('    S["latent solve<br/>min || r_w(h(z)) ||"]:::solved')
    w('    O["six dense output fields"]:::solved')
    w('  end')
    w('  D --> P')
    w('  G --> P')
    w('  P --> H')
    w('  P --> Z')
    w('  H --> C')
    w('  Z --> C')
    w('  G --> C')
    w('  C --> Q --> S --> O')
    w('  classDef trained fill:#dff0d8,stroke:#3c763d,stroke-width:2px,color:#1b3a1f;')
    w('  classDef frozen fill:#e8e8ef,stroke:#555,stroke-width:2px,color:#222;')
    w('  classDef solved fill:#fde9d9,stroke:#b35c00,stroke-width:2px,color:#4a2600;')
    w('  classDef data fill:#e3f0fb,stroke:#22618f,stroke-width:2px,color:#10314a;')
    w('```')
    w('')
    w('With the bank $G$ frozen, $\\Gamma = G^\\top G = \\Lambda\\Lambda^\\top$ and '
      '$a_i = \\Lambda^\\top c^\\ast_i$, the exact identity')
    w('')
    w('$$\\|G h - u_i\\|_2^2 = \\|\\Lambda^\\top h - a_i\\|_2^2 + f_i^2$$')
    w('')
    w('makes fitting the head against full fields and fitting the whitened head against '
      'precomputed span coefficients the same problem. The trained objective is')
    w('')
    w('$$\\mathcal L = \\mathcal L_{\\rm rec} + \\beta_W \\mathcal L_{\\rm W} '
      '+ \\beta_T \\mathcal L_{\\rm T} + \\gamma\\,\\mathcal L_{\\rm Z},$$')
    w('')
    w('$$\\mathcal L_{\\rm rec}=\\frac1{|\\mathcal B|}\\sum_{i\\in\\mathcal B}'
      '\\frac{\\|q_\\theta(z_i)-a_i\\|_2^2}{\\|u_i\\|_2^2},\\qquad '
      'r_w(c;p,\\nu)=\\frac{Ac-p+\\Delta t\\big(\\Phi^\\top\\mathcal N(Gc)+\\nu\\lambda\\odot Ac\\big)}'
      '{1+\\Delta t\\,\\nu\\lambda},$$')
    w('')
    w('$$\\mathcal L_{\\rm W}=\\frac1{|\\mathcal B_r|}\\sum_i'
      '\\frac{\\|r_w(h_\\theta(z_i);Ac^\\ast_{i-1},\\nu_i)\\|_2^2}{\\|Ac^\\ast_i\\|_2^2},\\qquad '
      '\\mathcal L_{\\rm T}=\\frac1{|\\mathcal B_r|}\\sum_i'
      '\\frac{\\|r_w(h_\\theta(z_i);Ah_\\theta(z_{i-1}),\\nu_i)\\|_2^2}{\\|Ac^\\ast_i\\|_2^2},$$')
    w('')
    w('$$\\mathcal L_{\\rm Z}=\\frac1{|\\mathcal B|}\\sum_i'
      '\\frac{\\|z_i-z_{i-1}\\|_2^2+\\|z_i-z_{\\pi(i)}\\|_2^2}{\\sigma_z^2}.$$')
    w('')
    ww = tr['weights']
    w(f"The weights were fixed once, before any arm ran, by the declared scale rule "
      f"({ww['rule']}): $\\beta_W = \\beta_T = {ww['beta_w']:.6g}$ and "
      f"$\\gamma = {ww['gamma']:.6g}$, from $\\mathcal L_{{\\rm rec}} = {ww['L_rec']:.6g}$, "
      f"$\\mathcal L_{{\\rm W}} = {ww['L_weak']:.6g}$, $\\mathcal L_{{\\rm Z}} = "
      f"{ww['L_smooth']:.6g}$ on {ww['calibration_states']} training-cohort calibration states.")
    w('')

    # ------------------------------------------------------------- gates ----
    w('## Gates')
    w('')
    w('| gate | value | tolerance | passed |')
    w('|---|---:|---:|---|')
    w(f"| whitening round trip $h\\to q\\to h$ | {tr['gates']['whitening_round_trip']:.3e} | 1e-10 | "
      f"{'yes' if ta['checks']['whitening_round_trip']['passed'] else 'NO'} |")
    w(f"| identity $(\\ast)$ against regenerated fields | "
      f"{tr['gates']['identity_star_relative']:.3e} | 1e-9 | "
      f"{'yes' if ta['checks']['identity_star']['passed'] else 'NO'} |")
    ig = ev.get('gates', {}).get('incumbent_reproduces_abl01')
    if ig:
        w(f"| `{ecfg['incumbent_arm']}` reproduces abl01 `{ig['reference']}` on "
          f"{ig['compared']} cases | {ig['worst_relative_delta']:.3e} | "
          f"{ig['tolerance']:g} | {'yes' if ig['passed'] else 'NO'} |")
    for k in ('recorded_errors_recomputed_from_saved_fields', 'same_grid_recomputed_from_saved_fields'):
        c = ea['checks'][k]
        w(f"| {k.replace('_', ' ')} | {c['detail']:.3e} | 1e-9 | "
          f"{'yes' if c['passed'] else 'NO'} |")
    w('')
    w('Every other audit check, both jobs: ' + ', '.join(
        f"`{k}` {'pass' if v['passed'] else '**FAIL**'}"
        for k, v in list(ta['checks'].items()) + list(ea['checks'].items())
        if k not in ('whitening_round_trip', 'identity_star',
                     'recorded_errors_recomputed_from_saved_fields',
                     'same_grid_recomputed_from_saved_fields')) + '.')
    w('')

    # ----------------------------------------------------------- training ---
    w('## The trained checkpoints')
    w('')
    w(f"Data: nested prefixes of the incumbent's own {tcfg['canonical_trajectories']} + "
      f"{tcfg['extra_trajectories']} trajectory draw, state stride {tcfg['state_stride']}, "
      f"{tr['data']['training']['states_per_trajectory']} states per trajectory, worst FOM "
      f"relative residual {tr['data']['training']['max_relative_residual']:.2e}. Held-out "
      f"generalisation cohort: {tcfg['holdout_trajectories']} trajectories from seed "
      f"{tcfg['holdout_seed']}, "
      f"{tr['data']['holdout']['snapshots']} states, span floor "
      f"{100 * tr['data']['holdout_span_floor']['mean']:.4f} % mean / "
      f"{100 * tr['data']['holdout_span_floor']['max']:.4f} % worst.")
    w('')
    w('| arm | traj | states | $K$ | $R$ | bank | objective | steps | recon (train) mean | '
      'held-out oracle mean % | held-out worst % | mean-code-only mean % | '
      '$\\times$ its own span floor | train GPU h |')
    w('|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for t in ta['arm_table']:
        w('| `' + t['arm'] + '` | ' + ' | '.join([
            str(t['trajectories']), str(t['states']), str(t['K']), str(t['R']), t['bank'],
            t['objective'], str(t['steps']), g(t['recon_train_mean'], 4),
            pct(t['holdout_mean']), pct(t['holdout_max']), pct(t['holdout_mean_only']),
            f(t['holdout_over_floor'], 1), f(t['train_gpu_hours'], 3)]) + ' |')
    w('')
    w('The held-out oracle columns use two initialisations (the mean training code and a '
      'training-only encoder) for the frozen-bank arms and the mean training code alone for the '
      'joint arms, which have no whitened training block in their own bank. The **mean-code-only '
      'column is the like-for-like one across the two families**; selection happened among '
      'frozen-bank arms only, where the two-initialisation column exists for all of them. The '
      'joint arms also have their own span floor, because their bank moved.')
    w('')
    w(f"Selection was by the declared rule ({sel['rule']}): best density "
      f"{sel['best_density']}, best $K$ {sel['best_K']}, best objective `{sel['best_objective']}`, "
      f"combination arm `{sel['best_combination_arm']}`"
      + (f"; the joint bank+head arms continue from `{sel['joint_warm_start']}`"
         if sel.get('joint_warm_start') else '') + '.')
    w('')
    w(f"Latent-dimension curve (held-out worst): {[(k, round(v, 6)) for k, v in sel['latent_curve']]}. "
      f"Objective curve: {[(o, round(v, 6)) for o, v in sel['objective_curve']]}.")
    w('')

    # --------------------------------------------------------- three layers --
    w('## The three layers, per checkpoint')
    w('')
    w('The primary metric is the same-grid discrepancy against the converged same-mesh full-order '
      'solve, because the refined-reference metric also contains this mesh\'s discretisation error. '
      'The $t=0$ column is the model\'s own compression of the supplied initial field; on the '
      'incumbent it is the largest of the six output times, which is why the worst-over-all-times '
      'number cannot be moved by any inference-time knob.')
    w('')
    w('| arm | $K$ | $R$ | $M$ | $m$ | quad. | bank floor % | best-found % | best-found $t{=}0$ % | '
      'best-found evolved % | solved same-grid % | solved evolved % | solved $t{=}0$ % | '
      'worst reference % | median iters/step | budget exits | stationary | completed | '
      'median GPU ms | median host ms | cost vs incumbent |')
    w('|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|')
    for x in rows:
        if x['kind'] != 'rom':
            continue
        cf = x['median_gpu_ms'] / inc['median_gpu_ms']
        w('| `' + x['arm'] + '` | ' + ' | '.join([
            str(x['K']), str(x['R']), str(x['M']), str(x['m'] or '--'), x['quadrature'],
            f(x['worst_bank_projection_percent']), f(x['worst_best_found_percent']),
            f(x['worst_best_found_t0_percent']), f(x['worst_best_found_evolved_percent']),
            f(x['worst_same_grid_percent']), f(x['worst_same_grid_evolved_percent']),
            f(x['worst_t0_percent']), f(x['worst_reference_percent']),
            f(x['median_iterations'], 1), str(x['budget_exits']),
            'yes' if x['all_stationary'] else 'NO', 'yes' if x['all_completed'] else 'NO',
            f(x['median_gpu_ms'], 3), f(x['median_host_ms'], 3), f(cf, 3)]) + ' |')
    w('')
    w('Same-job full-order controls (context only; no cross-job ratio is taken):')
    w('')
    w('| method | worst same-grid % | worst reference % | median GPU ms | median host ms |')
    w('|---|---:|---:|---:|---:|')
    for x in rows:
        if x['kind'] == 'rom':
            continue
        w('| `' + x['arm'] + '` | ' + ' | '.join([
            f(x['worst_same_grid_percent']), f(x['worst_reference_percent']),
            f(x['median_gpu_ms'], 3), f(x['median_host_ms'], 3)]) + ' |')
    w('')

    # ------------------------------------------------------------ verdicts ---
    w('## Verdict against the pre-registered criteria')
    w('')
    w(f"All four must hold: best-found $< {crit['best_found']}$ %, solved same-grid "
      f"$< {crit['same_grid']}$ %, every solve stationary and completed, cost "
      f"$\\le {crit['cost_factor']}\\times$ the incumbent.")
    w('')
    w('| arm | best-found % | same-grid % | stationary | completed | cost factor | success |')
    w('|---|---:|---:|---|---|---:|---|')
    for v in ea['verdicts']:
        w('| `' + v['arm'] + '` | ' + ' | '.join([
            f(v['best_found_percent']), f(v['same_grid_percent']),
            'yes' if v['stationary'] else 'NO', 'yes' if v['completed'] else 'NO',
            f(v['cost_factor'], 3), '**PASS**' if v['success'] else 'no']) + ' |')
    w('')

    # --------------------------------------------------------- deviations ----
    w('## Recorded deviations and limitations')
    w('')
    w('- **The density ladder is nested inside the incumbent\'s own draw** rather than three '
      'independent `params_draw(0, n)` draws, and a '
      f"{tcfg['canonical_trajectories'] + tcfg['extra_trajectories']}-trajectory rung was added to "
      'the brief\'s three, so that density is the only variable and the top rung is the '
      'incumbent\'s own data (DESIGN.md deviation D1).')
    w('- **A first training submission, job `3745663`, was CANCELLED while still PENDING.** It '
      'never started, consumed no GPU time and produced no output. It was cancelled because two '
      'defects in the joint bank+head arms were found after submission: the sampled-point data '
      f"term was a factor $n/P \\approx {(tcfg['intervals'] - 1) ** 2 // tcfg['joint_points']}$ "
      'below the relative MSE it claimed to be, which also put every fixed regulariser weight that '
      'factor too high against it; and the orthonormality penalty inherited from '
      '`sep_common.train_autodecoder` exists to condition a *freshly initialised* bank, while these '
      'arms warm start from a trained bank whose Gram is far from the identity, so it drowned the '
      'data term. Both are fixed and recorded as dated amendments A1-A3 in `DESIGN.md`, made before '
      'any evaluation number existed. The executed jobs are the training and evaluation jobs named '
      'at the top of this report.')
    w('- **The joint bank+head arms cannot use identity $(\\ast)$**, because their bank moves. They '
      f"train against a fixed seeded subset of {tcfg['joint_points']} of the "
      f"{(tcfg['intervals'] - 1) ** 2} interior points, at a capped density, continuing from the "
      'selected frozen-bank arm; their orthonormality weight is calibrated at the warm start rather '
      'than inherited, and their held-out oracle uses the mean-code initialisation only. They also '
      'have their own span floor.')
    w('- **The solve-aware training terms use the exact dense advection**, not the '
      'empirical-quadrature rule, because that rule is fitted at a fixed head and would drift as '
      '$\\theta$ moves.')
    w(f"- **The step budget is held fixed at {tcfg['steps']} for every frozen-bank arm**, so the "
      'low-density arms see far more epochs than the high-density ones. That is what isolates '
      'density, and it is also why the low-density arms are the ones most exposed to overfitting.')
    w('- **One mesh, one training seed per arm, development cohort only.** The final cohort stays '
      'sealed and no new case was opened. Nothing here is a speed claim against a full-order '
      'solver: the same-job full-order rows are context only.')
    w('')

    # ----------------------------------------------------------- glossary ----
    w('## Plain-language glossary')
    w('')
    for term, text in [
        ('arm', 'one configuration under test; everything except the named difference is held fixed.'),
        ('bank / $G$', 'the fixed set of spatial fields the reduced state is built from. Here it is '
                       'a coordinate network evaluated once on the grid and cached as a matrix.'),
        ('head / $h_\\theta$', 'the small network turning the few solved coordinates into bank '
                               'coefficients. This is the object being trained.'),
        ('auto-decoder codes / $Z$', 'one latent vector per training snapshot, optimised jointly '
                                     'with the head instead of produced by an encoder.'),
        ('$K$', 'latent dimension: how many numbers the online solver actually solves for.'),
        ('$R$', 'bank rank: how many spatial fields the bank offers.'),
        ('$M$ / test modes', 'the smooth functions the PDE residual is averaged against; there must '
                             'be more of them than solved unknowns or the objective collapses.'),
        ('$m$ / empirical quadrature (EQ)', 'a learned weighted subset of $m$ grid points standing in '
                                           'for a full grid sum; "dense" means no such approximation.'),
        ('span floor / bank projection', 'the best any coefficients at all could do in that bank -- '
                                         'a floor no head can beat.'),
        ('best-found reconstruction', 'the best that arm\'s own manifold can do on the reference '
                                      'field with no PDE involved, found by seeded multistart; it '
                                      'separates representation from dynamics.'),
        ('held-out representation oracle', 'the same quantity on trajectories no arm ever fitted, '
                                           'drawn from a separate seed. It is the generalisation '
                                           'measure and the only selection statistic used.'),
        ('same-grid error', 'difference from the converged full-order solve on the same mesh, so it '
                            'contains no discretisation error.'),
        ('reference error', 'difference from a much finer, much smaller-step full-order solve; it '
                            'contains this mesh\'s discretisation error as well.'),
        ('evolved times', 'the five output times after $t=0$.'),
        ('$t=0$ compression', 'the model\'s own reconstruction error on the supplied initial field, '
                              'before any time stepping.'),
        ('stationary / completed', 'two separate exit statuses: the normalized-gradient stationarity '
                                   'test, and finishing under the shared stopping rule with no '
                                   'iteration-budget or rejected-step exit. Completion is the honest '
                                   'status; stationarity alone is not a quality ranking.'),
        ('cost factor', 'median GPU query milliseconds divided by the incumbent\'s, measured in the '
                        'same job on the same GPU with burn-in before every timed block.'),
        ('development / final cohort', 'cases usable for method selection / cases kept unopened.'),
        ('frozen-bank vs joint arm', 'a frozen-bank arm retrains only the head and the codes, so '
                                     'every such arm shares one span floor; a joint arm also moves '
                                     'the bank, so it has its own floor.'),
    ]:
        w(f'- **{term}:** {text}')
    w('')

    text = '\n'.join(out) + '\n'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
