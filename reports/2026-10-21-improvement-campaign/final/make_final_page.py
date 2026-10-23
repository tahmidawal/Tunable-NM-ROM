"""Plain-language results page (page/index.html) for the 2026-10-21 improvement campaign.

Every number comes from FINAL-EXPORT.json or PROVENANCE.json through common.facts(); the plan's registered
bars come from common.BARS. Figures are the report's own figures, embedded as base64 data URIs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import X, BARS, CSS, FONTS, esc, r3, pct, pct3, gh, img, branches, verdict_class, facts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'page', 'index.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
F = facts()
B = BARS
EXP = X['experiments']


def pill(v):
    return f'<span class="pill {verdict_class(v)}">{esc(v)}</span>'


def ul(items):
    return '<ul>' + ''.join(f'<li>{i}</li>' for i in items) + '</ul>'


def figs(*fs):
    return '<div class="figs">' + ''.join(img(n, c) for n, c in fs) + '</div>'


def tbl(head, rows):
    h = ''.join(f'<th>{esc(x)}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def problem(num, title, question, verdict, tried, numbers, fig_html):
    return (f'<section class="prob" id="p{num}"><h2>Problem {num}: {esc(title)}</h2>'
            f'<p class="q"><b>Question.</b> {question}</p>'
            f'<p class="verdict"><b>Verdict.</b> {pill(verdict)}</p>'
            f'<h3>What was tried</h3>{ul(tried)}<h3>Key numbers</h3>{ul(numbers)}{fig_html}</section>')


answer = (
    f'<p class="lead">Problem 1 is only <b>partly</b> addressed, and Problems 2 and 3 are <b>not fixed</b>. On the plain box, '
    f'a reduced model beats a tuned spectral solver by at least {B["win_S"]}&times; only in a few cells, and our own bank gets '
    f'there only with a new time step, on the easiest family, where the standard POD models do just as well. Without a fast '
    f'transform (an L-shape, a box with a hole) and with a variable coefficient, the full solvers stay cheaper. The tuned '
    f'POD&nbsp;+&nbsp;ECSW model still beats us on cost: a faster step speeds up our evolution by {r3(F["e5_min"])}&ndash;'
    f'{r3(F["e5_max"])}&times; but slightly worsens the hardest case, a cheaper nonlinear term barely helps, and a POD built '
    f'once matches us on every multi-mesh workload at about a tenth of our cost. For fronts steeper than training, no '
    f'retrained bank and no finer time step passes on a single steep test case.</p>')

p1 = problem(
    1, 'a coarse spectral solver is faster at equal accuracy',
    'Can the reduced model beat the best full solver on the box, on complex geometry, or with a variable coefficient?',
    F['v1'],
    [f'<b>E1, box:</b> {len(F["e1_cells"])} (family width &times; steepness) cells, several reduced models against a tuned spectral solver '
     f'(dense-transform version, which was faster than the FFT here).',
     '<b>E2, geometry:</b> an L-shape and a box with a hole, where no fast transform exists, against spectral elements, '
     'finite differences with multigrid and the POD models.',
     f'<b>E3, variable coefficient:</b> full solvers only, at contrast {r3(F["e3_C"])}; a reduced model would be trained only if '
     f'the fastest full solver were at least {B["gate_G"]}&times; slower than our box model.',
     '<b>E4, 3D holes:</b> not run, because it depended on an E2 win.'],
    [f'Box: the largest speed-up on the wide family is {r3(F["e1_maxS"])}&times; over {F["e1_nmatch"]} matched comparisons, '
     f'reached by a translation model, not our method.',
     f'Box: our bank with the new ETDRK4 step reaches {r3(F["e1_bank"])}&times; (test {r3(F["e1_bank_test"])}&times;) only on '
     f'the narrow, low-steepness cell, where POD-Galerkin reaches {r3(F["e1_podg"])}&times; and POD&nbsp;+&nbsp;ECSW '
     f'{r3(F["e1_ecsw"])}&times;. On the wide family it reaches at most {r3(F["e1_e4_wide_max"])}&times;.',
     f'Box: our deployed bank (LSPG step) stays at {r3(F["e1_lspg_min"])}&ndash;{r3(F["e1_lspg_max"])}&times; over its '
     f'{F["e1_lspg_n"]} matched entries.',
     f'Geometry: speed-ups of {r3(F["e2_L_min"])}&ndash;{r3(F["e2_L_max"])}&times; on the L-shape and {r3(F["e2_H_min"])}'
     f'&ndash;{r3(F["e2_H_max"])}&times; on the holed box (bar {B["win_S"]}&times;). The holed-box claim is withdrawn; the '
     f'L-shape is not decidable.',
     f'Variable coefficient: the fastest full solver costs only {r3(F["e3_G03"])} (3 % target) and {r3(F["e3_G10"])} '
     f'(10 % target) of our box model, so no reduced model was trained.'],
    figs(('imp-box__S_vs_W_worst.png', 'Box (E1): speed-up over the spectral solver against family width, worst-error '
          f'targets. Above {B["win_S"]} would be a win; most points are far below it.'),
         ('imp-geometry__S_per_family.png', 'Geometry (E2): our speed-up against each competing solver family; '
          'below 1 means the competitor is faster.'),
         ('imp-varcoef__gate.png', f'Variable coefficient (E3): how much slower the fastest full solver is than our model, '
          f'with its uncertainty. The bar was {B["gate_G"]}; both targets are below 1.')))

e5r = [r for r in F['e5_rows'] if r['mesh'] == '1024']
p2 = problem(
    2, 'tuned POD + ECSW beats us on cost',
    'Can a faster time step, a cheaper nonlinear term, or reuse across meshes close the gap to POD&nbsp;+&nbsp;ECSW?',
    F['v2'],
    [f'<b>E5, faster step:</b> {len(e5r)} Galerkin steppers against our deployed LSPG step; each had to be stable, no worse in '
     f'the worst case, and at least {B["e5_speed"]}&times; faster.',
     f'<b>E6, cheaper nonlinear term:</b> classical quadrature rules with bank test functions (F2) and fitted ECSW-style '
     f'weights (F3); a fitted rule had to pass the quadrature check with at least {B["e6_gain"]}&times; gain.',
     '<b>E7, multi-mesh cost:</b> workloads mixing several meshes and query counts, against a fair control, a continuous POD built once and '
     'reused on every mesh.'],
    [f'Faster step: ETDRK4 makes the evolution {r3(F["e5_min"])}&ndash;{r3(F["e5_max"])}&times; faster, but its worst error '
     f'({pct(F["e5_e4a_worst"])}) is above the deployed step ({pct(F["e5_ref1024"])}), so the rule fires for every arm.',
     f'Even then, a matched POD&nbsp;+&nbsp;ECSW with the same step takes only {r3(F["e5_pod_a"])}&ndash;{r3(F["e5_pod_b"])} '
     f'of our ETDRK4 query time.',
     f'Cheaper nonlinear term: the best fitted-weight gain is {r3(F["e6_best"])}&times;, and {F["e6_npass"]} of '
     f'{F["e6_n_f3"]} fitted-weight arms pass the quadrature check. Halving the points saves little: twice the points cost '
     f'only {r3(F["e6_jac2"])}&times; as much.',
     f'Multi-mesh: the continuous POD matches us on {F["e7_match"]} of {F["e7_n"]} (workload, configuration) pairs at '
     f'{r3(F["e7_min"])}&ndash;{r3(F["e7_max"])} of our cost (median {r3(F["e7_med"])}). The comparative multi-mesh cost '
     f'claim is dropped.'],
    figs(('imp-faststep__worst_error_vs_evolve_time.png', 'Faster step (E5): worst error against evolution time. '
          'ETDRK4 points sit far to the left (faster) but slightly above the deployed step (less accurate in the worst case).'),
         ('imp-hrcost__error-vs-speedup.png', 'Cheaper nonlinear term (E6): worst error against end-to-end gain; '
          'no arm gets far from 1 (no gain) without losing accuracy or failing the check.'),
         ('imp-multimesh__cost_vs_queries.png', 'Multi-mesh (E7): total cost against number of queries; the continuous POD '
          'line stays below ours.')))

p3 = problem(
    3, 'fails on fronts steeper than training',
    'Can a retrained bank, an out-of-regime flag, or better numerics near fronts handle steeper fronts?',
    F['v3'],
    ['<b>E8, retrain a better bank:</b> CPU diagnostics, then single-change retrains against a control retrain (K1): '
     '(b) more steep, low-viscosity training data; (c) a bank that respects the boundary; (a) higher-fidelity data was deferred.',
     '<b>E9, out-of-regime flag:</b> flag queries whose steepness s = a/&nu; is above a threshold frozen before testing.',
     '<b>E10, numerics near fronts:</b> a finer time step (E10-s) and quadrature that refines when the state needs it (E10-q).'],
    [f'More steep data: the median error on the steepest cohort drops only {r3(F["e8_b_ratio"])}&times; '
     f'({pct(F["e8_b_med"])} vs control {pct(F["e8_K1"])}; bar {B["e8_b"]}&times;), and {F["e8_b_q"]} of {F["e8_b_n"]} steep '
     f'cases pass the numerical checks.',
     f'Boundary-respecting bank: it still needs {int(F["e8_c_nodes"])} quadrature points (bar &le; {int(F["e8_c_bar"])}), and '
     f'its high-frequency tail shrinks only {r3(F["e8_c_tail"])}&times; (bar {B["e8_c_tail"]}&times;).',
     f'Finer time step: median {pct(F["e10s_med"])} against a bar of {pct(F["e10s_bar"])}; {F["e10s_q"]} of {F["e10s_n"]} '
     f'steep cases pass and {F["e10s_conv"]} are step-converged.',
     f'Quadrature is not the problem: its indicator never exceeds {r3(F["e10q_rho"])} over {F["e10q_states"]} accepted '
     f'step states ({F["e10q_over"]} over the bar). The binding failure is certifying the time step on steep states.',
     f'Flag: on {F["e9_n"]} test cases it flags {pct(F["e9_flag"], 1)} and separates good from bad perfectly (AUC '
     f'{r3(F["e9_A"]["auc"])}), but only {F["e9_A"]["n_good"]} good / {F["e9_A"]["n_bad"]} bad cases were available '
     f'(minimum {B["e9_good"]} / {B["e9_bad"]}), so it is descriptive only.'],
    figs(('imp-fronts__phaseb_medians.png', 'Steep fronts (E8): median error per test cohort for each retrained arm against '
          'the control K1; the steep cohorts stay high for every arm.'),
         ('imp-fronts__a4_hybrid_floors.png', 'Steep fronts (E8 diagnostics): best error a sine-enriched bank could reach in '
          'principle; it did not justify a retrain.')))

# Table of all experiments.
RULES = {
    'E1': f'stop if every reduced model is below {B["stop_S"]}&times; at every target on the wide family',
    'E2': f'win only if at least {B["win_S"]}&times; faster than the best competitor; otherwise withdraw',
    'E3': f'train a reduced model only if the fastest full solver is at least {B["gate_G"]}&times; slower',
    'E4': 'run only if E2 wins',
    'E5': f'stop if unstable, worse in the worst case than the deployed step, or less than {B["e5_speed"]}&times; faster',
    'E6': f'F2: no more points, no worse, faster; F3: a fitted rule passes the check with at least {B["e6_gain"]}&times;',
    'E7': 'drop the claim if a continuous POD matches us on every workload',
    'E8': f'(b) steep-cohort median at least {B["e8_b"]}&times; lower; (c) stop unless points {B["e8_c_m"]}&times; fewer or '
          f'tail {B["e8_c_tail"]}&times; smaller',
    'E9': 'report coverage and false alarms (no stop rule)',
    'E10': 'only if E8 falls short; single changes against K1',
}
PROB = dict(E1=1, E2=1, E3=1, E4=1, E5=2, E6=2, E7=2, E8=3, E9=3, E10=3)
FIXED = dict(E1='partly', E2='no', E3='no', E4='not run', E5='no', E6='no', E7='no (claim dropped)', E8='no',
             E9='descriptive', E10='no')


def outcome(e):
    v = EXP[e]
    if isinstance(v, dict):
        return '; '.join(f'{k}: {x}' for k, x in v.items())
    return v


exp_rows = [[f'<b>{e}</b>', str(PROB[e]), RULES[e], esc(outcome(e)), pill(FIXED[e])] for e in RULES]

claims_cannot = [
    'A general "faster than a spectral solver" claim.',
    'An advantage on geometry without a fast transform: withdrawn for the holed box, not supported on the L-shape. '
    'The 3D holes case is not claimed.',
    'An advantage with a variable coefficient: the tuned full solver is already about as cheap as our box model.',
    'A comparative multi-mesh cost advantage over POD.',
    'That a faster step or a cheaper nonlinear term closes the gap to POD&nbsp;+&nbsp;ECSW.',
    'Robustness to fronts steeper than training; it stays a stated limitation.',
]
claims_can = [
    f'On the box, a reduced model beats the tuned dense spectral solver by at least {B["win_S"]}&times; only in the few '
    f'listed cells and targets (upper bounds); our bank does so only with the ETDRK4 step on the narrow, low-steepness '
    f'family, where the POD models do as well, and the deployed bank stays at {r3(F["e1_lspg_min"])}&ndash;'
    f'{r3(F["e1_lspg_max"])}&times;.',
    'Mesh-independent evolution cost and mesh-free evaluation remain properties of the method, stated without a '
    'comparative claim.',
    f'The ETDRK4 evolution speed-up ({r3(F["e5_min"])}&ndash;{r3(F["e5_max"])}&times;) as a mechanism result, together '
    'with the finding that tuned POD&nbsp;+&nbsp;ECSW dominates our solve cost.',
    'The steepness s = a/&nu; as a descriptive out-of-regime flag, with fallback to the full solver.',
]

deviations = [
    'One smoke job (E6) was submitted against a do-not-submit audit; the computation was unaffected and accepted as a smoke-only deviation.',
    'One retrain (E8, arm c) was submitted although the budget check refused it; the budget still held. Disclosed as a procedure breach.',
    'E3 used a cross-check rule adopted after seeing reference data (before any solver result); both readings are reported and neither trains a model.',
    'The E2 geometry was redesigned after inspecting smoke runs, so the choice was not outcome-blind.',
    'E1 used spectral reference solutions instead of the second-order recipe, and its timing method changed mid-lane; '
    f'the measured effect is a ratio of {r3(F["e1_a16_min"])}&ndash;{r3(F["e1_a16_max"])}.',
    'Two account usage-limit stops and an expired GitHub credential paused lanes; every lane was re-synced afterwards.',
]
deferred = [
    f'E1: {len(F["e1_deferred"])} of {len(F["e1_deferred"]) + len(F["e1_cells"])} cells (mostly the steepest level), '
    'under the budget cap.',
    'E2: the narrow family, the second steepness level and box controls.',
    'E3: a variable-coefficient reduced model and several stronger full-solver families.',
    'E5: the smaller rank and one control at the finest mesh.',
    'E8: the data-fidelity retrain (over its cap) and the timing of the full-solver fallback.',
    'Next step for steep fronts: an adaptive time integrator or a shifting/local basis (not a single change, so not released).',
]

ledger_rows = [[esc(b['lane']), esc(b['exp']), f'<span class="n">{gh(F["gpu"][b["lane"]])}</span>'] for b in branches()]
ledger_rows.append(['<b>total</b>', '', f'<b>{gh(F["gpu_total"])}</b> of {F["gpu_cap"]} cap'])
branch_rows = [[esc(b['lane']), esc(b['exp']), f'<code>{esc(b["branch"])}</code>', f'<code>{esc(b["head"])}</code>',
                f'<code>{esc(b["mirror"])}</code>'] for b in branches()]

glossary = [
    ('Reduced model (ROM)', 'a small, fast surrogate of the simulation; ours is built from a learned "bank" of basis functions.'),
    ('Full solver (FOM)', 'the conventional, accurate, expensive simulation.'),
    ('Spectral solver', 'a full solver that represents the solution with sines; very fast on a plain box.'),
    ('POD, POD + ECSW', 'the standard competitor: a basis taken from simulation snapshots, with a small weighted set of points for the nonlinear term.'),
    ('Continuous POD', 'a POD basis built once and reused on other meshes; the fair control for multi-mesh cost.'),
    ('Bank', 'our ordered set of learned basis functions.'),
    ('LSPG, Galerkin, ETDRK4', 'ways of stepping the reduced model in time; LSPG is the deployed one, ETDRK4 a fast exponential step.'),
    ('Speed-up S', 'competitor time divided by our time at the same error; above 1 means we are faster. Always an upper bound.'),
    ('Gate ratio G', 'fastest full-solver cost divided by our cost (E3).'),
    ('Worst / median error', 'the largest / middle relative error over the test cases.'),
    ('Target', 'the error level (1, 3 or 10 %) at which speeds are compared.'),
    ('Wide / narrow family', 'how spread out the problem parameters are; wide is the harder, decisive case.'),
    ('Steepness s = a/ν', 'flow strength over viscosity; large s means sharp fronts.'),
    ('Qualified', 'passed the numerical and quadrature checks, so its numbers can be trusted.'),
    ('Stop rule / bar', 'a pass/fail threshold fixed before the experiment ran.'),
    ('Quadrature points m', 'where the nonlinear term is evaluated.'),
    ('K1', 'the control retrain of the existing recipe in E8.'),
    ('AUC', 'how well the flag separates good from bad cases; 1 is perfect.'),
    ('GPU-hours', 'cluster GPU time charged; each lane had a cap.'),
    ('Mirror', 'the filtered GitHub backup branch of a lane.'),
]

PAGE_CSS = CSS + r'''
main{max-width:920px;margin:0 auto;padding:12px max(16px, env(safe-area-inset-right, 0px)) 48px max(16px, env(safe-area-inset-left, 0px))}
.lead{font-size:1.06rem;background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:8px;padding:.9em 1em}
.prob{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:2px 18px 18px;margin:1.6em 0}
.prob h2{border:none}
.prob ul{padding-left:1.2em}.prob li{margin:.3em 0}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:14px;margin-top:1em}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px;margin:1em 0}
.verdicts div{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.7em .9em}
dl.gl{display:grid;grid-template-columns:minmax(0,max-content) minmax(0,1fr);gap:.35em 1em}
dl.gl dt{font-weight:600}dl.gl dd{margin:0}
@media (max-width:560px){dl.gl{grid-template-columns:1fr}dl.gl dd{margin-bottom:.5em}h1{font-size:1.55rem}.prob{padding:2px 12px 14px}}
@media print{.prob,.frame,.tbl{break-inside:avoid}}
'''

vcards = ''.join(f'<div><b>Problem {n}</b><br>{esc(t)}<br>{pill(F["v" + str(n)])}</div>' for n, t in (
    (1, 'Spectral solver is faster'), (2, 'Tuned POD + ECSW beats us'), (3, 'Steep fronts')))

page = f'''<title>Burgers Improvement Campaign Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
<style>{PAGE_CSS}</style>
<main>
<header>
<h1>Burgers improvement campaign: results</h1>
<p class="mute">Can the three known weaknesses of the off-mesh bank reduced model be fixed? Final numbers, compiled
{esc(X["compiled_utc"])}. Speed-ups are upper bounds against the strongest tested competitor; errors are measured against
reference solutions whose own error is estimated, not bounded.</p>
</header>
<section><h2>The answer</h2>{answer}<div class="verdicts">{vcards}</div></section>
{p1}{p2}{p3}
<section><h2>Every experiment and its stop rule</h2>
{tbl(['Exp', 'Problem', 'Rule fixed in advance', 'Outcome', 'Fixed?'], exp_rows)}</section>
<section><h2>What the paper can and cannot claim</h2>
<h3>Can claim</h3>{ul(claims_can)}<h3>Cannot claim</h3>{ul(claims_cannot)}</section>
<section><h2>Deviations and deferred items</h2>
<h3>Deviations (all disclosed)</h3>{ul(deviations)}<h3>Deferred</h3>{ul(deferred)}</section>
<section><h2>GPU-hours</h2>{tbl(['Lane', 'Experiments', 'GPU-hours spent'], ledger_rows)}
<p class="mute">E4 was not run and its reserve was not spent.</p></section>
<section><h2>Branches awaiting your merge decision</h2>
<p>None of the {len(branches())} lane branches was merged in this campaign.</p>
{tbl(['Lane', 'Experiments', 'Branch', 'Last commit', 'GitHub mirror'], branch_rows)}</section>
<section><h2>Words used here</h2><dl class="gl">{"".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in glossary)}</dl></section>
</main>'''

open(OUT, 'w').write(page)
print(OUT, len(page), 'figures:', page.count('data:image/png'))
