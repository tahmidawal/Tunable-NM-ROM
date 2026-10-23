"""Self-contained HTML slide deck (deck/index.html) for the 2026-10-21 improvement campaign.

One slide per question. Every number comes from FINAL-EXPORT.json or PROVENANCE.json via common.facts();
registered bars from common.BARS. On wide screens the deck shows one slide at a time (arrow keys / buttons);
on narrow screens, or without JavaScript, every slide is stacked and visible.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import X, BARS, CSS, FONTS, esc, r3, pct, gh, img, branches, verdict_class, facts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'deck', 'index.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
F = facts()
B = BARS
EXP = X['experiments']


def pill(v):
    return f'<span class="pill {verdict_class(v)}">{esc(v)}</span>'


def slide(question, answer, bullets, fig=None, tag=''):
    body = '<ul>' + ''.join(f'<li>{b}</li>' for b in bullets) + '</ul>'
    figure = img(*fig) if fig else ''
    cls = 'slide two' if fig else 'slide'
    return (f'<section class="{cls}"><div class="txt"><p class="tag">{tag}</p><h2>{question}</h2>'
            f'<p class="ans">{answer}</p>{body}</div>{figure}</section>')


slides = []
slides.append(
    '<section class="slide title"><div class="txt"><p class="tag">Improvement campaign &middot; 2026-10-21</p>'
    '<h1>Can the three known weaknesses of the off-mesh bank be fixed?</h1>'
    f'<p class="ans">Problem 1 (spectral solver faster): {pill(F["v1"])}<br>'
    f'Problem 2 (tuned POD&nbsp;+&nbsp;ECSW beats us): {pill(F["v2"])}<br>'
    f'Problem 3 (steeper fronts): {pill(F["v3"])}</p>'
    f'<p class="mute">Final numbers compiled {esc(X["compiled_utc"])}. {gh(F["gpu_total"])} GPU-hours of a {F["gpu_cap"]} cap. '
    'Speed-ups are upper bounds against the strongest tested competitor.</p></div></section>')

slides.append(slide(
    'Does the reduced model beat a spectral solver on the box?',
    f'Only in a few cells, and our bank only on the easiest family, where POD does as well.',
    [f'Largest wide-family speed-up: {r3(F["e1_maxS"])}&times; over {F["e1_nmatch"]} matched comparisons, by a translation '
     f'model, not our method (bar {B["win_S"]}&times;).',
     f'Our bank with ETDRK4: {r3(F["e1_bank"])}&times; on the narrow, low-steepness cell; POD-Galerkin {r3(F["e1_podg"])}&times;, '
     f'POD&nbsp;+&nbsp;ECSW {r3(F["e1_ecsw"])}&times;; at most {r3(F["e1_e4_wide_max"])}&times; on the wide family.',
     f'Deployed LSPG bank: {r3(F["e1_lspg_min"])}&ndash;{r3(F["e1_lspg_max"])}&times; over {F["e1_lspg_n"]} entries.',
     f'Stop rule (all below {B["stop_S"]}&times;): {esc(EXP["E1"])}.'],
    ('imp-box__S_vs_W_median.png', 'Speed-up over the spectral solver against family width, median-error targets'), 'E1 &middot; Problem 1'))

slides.append(slide(
    'Does the method win on complex geometry?',
    'No: without a fast transform the bank is still slower than the best competitor.',
    [f'L-shape: speed-up {r3(F["e2_L_min"])}&ndash;{r3(F["e2_L_max"])}&times;; holed box: {r3(F["e2_H_min"])}&ndash;'
     f'{r3(F["e2_H_max"])}&times; (bar {B["win_S"]}&times;).',
     f'Holed box: {esc(F["e2_labels"]["H"])}. L-shape: not decidable (reference check failed on some cases).',
     f'Against finite differences with multigrid alone, worst error at 10 %: {r3(F["e2_fdmg"]["L"][1])}&times; (L), '
     f'{r3(F["e2_fdmg"]["H"][1])}&times; (H); the POD models are the strongest competitors.',
     f'E4 (3D holes) was {esc(EXP["E4"])}, since it depended on an E2 win.'],
    ('imp-geometry__S_per_family.png', 'Speed-up of the bank against each competitor family; below 1 the competitor is faster'),
    'E2 &middot; Problem 1'))

slides.append(slide(
    'Does a variable coefficient remove the spectral solver\'s advantage?',
    'No: the tuned full solver is already about as cheap as our box model.',
    [f'At contrast {r3(F["e3_C"])}, the fastest full solver costs {r3(F["e3_G03"])} (3 % target) and {r3(F["e3_G10"])} '
     f'(10 % target) of our model; {B["gate_G"]}&times; was needed.',
     f'95 % intervals: [{r3(F["e3_ci03"][0])}, {r3(F["e3_ci03"][1])}] and [{r3(F["e3_ci10"][0])}, {r3(F["e3_ci10"][1])}].',
     f'Gate: {esc(F["e3_gate"])} (post-hoc rule) / {esc(F["e3_gate_reg"])} (registered rule); either way no reduced model '
     'was trained.'],
    ('imp-varcoef__fom_cost_vs_error.png', 'Full-solver cost against worst error with a variable coefficient'),
    'E3 &middot; Problem 1'))

slides.append(slide(
    'Can a better time step close the gap to POD + ECSW?',
    'No: ETDRK4 is much faster, but slightly worse in the hardest case, and POD stays cheaper.',
    [f'Evolution {r3(F["e5_min"])}&ndash;{r3(F["e5_max"])}&times; faster than the deployed LSPG step.',
     f'Worst error {pct(F["e5_e4a_worst"])} against {pct(F["e5_ref1024"])} for the deployed step, so the worst-error rule '
     'fires for every arm.',
     f'A matched POD&nbsp;+&nbsp;ECSW with the same step takes {r3(F["e5_pod_a"])}&ndash;{r3(F["e5_pod_b"])} of our query time.'],
    ('imp-faststep__worst_error_vs_evolve_time.png', 'Worst error against evolution time per stepper and rank'),
    'E5 &middot; Problem 2'))

slides.append(slide(
    'Can the nonlinear term be made cheaper?',
    'Not usefully: fewer points buy little, and no fitted rule passes the quadrature check.',
    [f'Classical rule with bank tests: {int(F["e6_F2_m"])} points instead of {int(F["e6_R_m"])}, gain {r3(F["e6_F2_e2e"])}&times;, '
     f'worst error {pct(F["e6_F2_worst"])} vs {pct(F["e6_R_worst"])}.',
     f'Fitted ECSW-style weights: best gain {r3(F["e6_best"])}&times; (bar {B["e6_gain"]}&times;); {F["e6_npass"]} of '
     f'{F["e6_n_f3"]} arms pass the check.',
     f'Why: twice the points cost {r3(F["e6_jac2"])}&times;, four times cost {r3(F["e6_jac4"])}&times;.'],
    ('imp-hrcost__error-vs-speedup.png', 'Worst error against end-to-end gain per nonlinear-term arm'),
    'E6 &middot; Problem 2'))

slides.append(slide(
    'Is there a multi-mesh cost advantage?',
    'No: a POD built once and reused matches us everywhere at about a tenth of the cost.',
    [f'Continuous POD matches us on {F["e7_match"]} of {F["e7_n"]} (workload, configuration) pairs.',
     f'Its cost is {r3(F["e7_min"])}&ndash;{r3(F["e7_max"])} of ours (median {r3(F["e7_med"])}).',
     f'Stricter accuracy criteria leave {F["e7_case"]["NO-MATCH"]} and {F["e7_roll"]["NO-MATCH"]} non-matches, all with no '
     'accurate-enough competitor at all, not cost wins.',
     'The comparative claim is dropped; mesh independence stays a property of the method.'],
    ('imp-multimesh__cost_vs_queries.png', 'Total cost against number of queries, ours against continuous POD'),
    'E7 &middot; Problem 2'))

slides.append(slide(
    'Can the bank handle steeper fronts?',
    'No: no retrain or finer step passes on a single steep case; time-step certification is the bottleneck.',
    [f'More steep training data: median {pct(F["e8_b_med"])} vs control {pct(F["e8_K1"])}, only {r3(F["e8_b_ratio"])}&times; '
     f'lower (bar {B["e8_b"]}&times;); {F["e8_b_q"]} of {F["e8_b_n"]} cases qualified.',
     f'Boundary-respecting bank: still {int(F["e8_c_nodes"])} points (bar &le; {int(F["e8_c_bar"])}), tail {r3(F["e8_c_tail"])}&times; '
     f'smaller (bar {B["e8_c_tail"]}&times;).',
     f'Finer step: median {pct(F["e10s_med"])} vs bar {pct(F["e10s_bar"])}; quadrature indicator never above '
     f'{r3(F["e10q_rho"])} over {F["e10q_states"]} states.',
     f'Steepness flag: AUC {r3(F["e9_A"]["auc"])} but only {F["e9_A"]["n_good"]} good / {F["e9_A"]["n_bad"]} bad cases '
     f'(minimum {B["e9_good"]} / {B["e9_bad"]}): descriptive only.'],
    ('imp-fronts__phaseb_medians.png', 'Median error per cohort for each retrained arm against the control K1'),
    'E8 &middot; E9 &middot; E10 &middot; Problem 3'))

slides.append(slide(
    'What can the paper claim?',
    'A narrowed box result, the method\'s mesh properties, and honest limitations.',
    [f'Can: on the box, a reduced model beats the dense spectral solver by &ge; {B["win_S"]}&times; only in a few listed '
     f'cells; our bank only with ETDRK4 on the narrow family, where POD does as well; the deployed bank stays at '
     f'{r3(F["e1_lspg_min"])}&ndash;{r3(F["e1_lspg_max"])}&times;.',
     f'Can: mesh-independent evolution and mesh-free evaluation as properties; the {r3(F["e5_min"])}&ndash;{r3(F["e5_max"])}&times; '
     'ETDRK4 speed-up as a mechanism result; tuned POD&nbsp;+&nbsp;ECSW dominating our solve cost, reported.',
     'Cannot: a general "faster than spectral" claim, a geometry or variable-coefficient advantage, a multi-mesh cost '
     'advantage, or robustness to steeper fronts.'],
    tag='Claims'))

slides.append(slide(
    'What went wrong along the way?',
    'Several procedure deviations, all disclosed; none changes a verdict.',
    ['Two submissions went ahead against a check: an E6 smoke against a do-not-submit audit, an E8 retrain against a '
     'refused budget check (budget still held).',
     'E3 used a cross-check rule adopted after seeing reference data; both readings are reported and neither trains a model. '
     'The E2 geometry was redesigned after smoke runs.',
     f'E1 changed its timing method mid-lane; the measured effect is a ratio of {r3(F["e1_a16_min"])}&ndash;'
     f'{r3(F["e1_a16_max"])} for cross-cell comparisons.',
     'Two usage-limit stops, an expired GitHub credential and a shared-scratchpad collision paused or disturbed lanes; '
     'affected audits were re-run.'],
    tag='Retractions and deviations'))

slides.append(slide(
    'What is left open?',
    'Deferred scope, an untested route for steep fronts, and the merge decision.',
    [f'E1: {len(F["e1_deferred"])} of {len(F["e1_deferred"]) + len(F["e1_cells"])} box cells deferred (mostly the steepest '
     'level); E2: narrow family and second steepness level; E3: a variable-coefficient model.',
     'Steep fronts: an adaptive time integrator or a shifting/local basis, not released because it is not a single change.',
     f'GPU-hours: {gh(F["gpu_total"])} of {F["gpu_cap"]} spent; E4 reserve unspent.',
     f'All {len(branches())} lane branches await your merge decision: '
     + ', '.join(f'<code>{esc(b["branch"])}</code>' for b in branches()) + '.'],
    tag='Open'))

DECK_CSS = CSS + r'''
body{min-height:100vh}
.deck{max-width:1200px;margin:0 auto;padding:12px max(16px, env(safe-area-inset-right, 0px)) 72px max(16px, env(safe-area-inset-left, 0px))}
.slide{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 18px;
display:flex;flex-direction:column;gap:16px}
.slide h1{font-size:1.9rem;margin:.2em 0 .4em}.slide h2{font-size:1.45rem;border:none;margin:.1em 0 .4em;padding:0}
.tag{margin:0;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-weight:700}
.ans{font-size:1.1rem;font-weight:600;margin:.2em 0 .6em}
.slide ul{padding-left:1.2em;margin:0}.slide li{margin:.35em 0}
.slide.title .ans{line-height:2}
.nav{display:none}
@media (min-width:900px){
  .slide.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);align-items:center}
  .slide .frame img{max-height:62vh;width:auto}
  html.js .slide{display:none;min-height:calc(100vh - 110px);margin:0}
  html.js .slide.active{display:flex}
  html.js .slide.two.active{display:grid}
  html.js .slide.title.active{justify-content:center}
  html.js .nav{display:flex;position:fixed;left:0;right:0;bottom:0;justify-content:center;align-items:center;gap:14px;
    padding:10px;background:var(--bg);border-top:1px solid var(--line)}
  .nav button{font:inherit;padding:.35em 1em;border-radius:8px;border:1px solid var(--line);background:var(--surface);
    color:var(--ink);cursor:pointer}
  .nav button:disabled{opacity:.4;cursor:default}
  .nav span{min-width:5em;text-align:center;color:var(--mute)}
}
@media print{html.js .slide{display:flex!important;min-height:0;break-after:page}.nav{display:none!important}}
'''

JS = r'''
(function(){
  var root=document.documentElement; root.classList.add('js');
  var s=[].slice.call(document.querySelectorAll('.slide')), i=0;
  var prev=document.getElementById('prev'), next=document.getElementById('next'), pos=document.getElementById('pos');
  function show(n){ i=Math.max(0,Math.min(s.length-1,n));
    s.forEach(function(e,k){ e.classList.toggle('active',k===i); });
    pos.textContent=(i+1)+' / '+s.length; prev.disabled=i===0; next.disabled=i===s.length-1;
    try{ history.replaceState(null,'','#'+(i+1)); }catch(e){} }
  prev.addEventListener('click',function(){show(i-1);});
  next.addEventListener('click',function(){show(i+1);});
  document.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){ if(window.innerWidth>=900){e.preventDefault();show(i+1);} }
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){ if(window.innerWidth>=900){e.preventDefault();show(i-1);} }
    else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(s.length-1);} });
  var h=parseInt((location.hash||'').slice(1),10); show(isNaN(h)?0:h-1);
})();
'''

page = f'''<title>Improvement Campaign Deck</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
<style>{DECK_CSS}</style>
<main class="deck">
{"".join(slides)}
</main>
<nav class="nav" aria-label="Slide navigation"><button id="prev" type="button">&larr; Previous</button><span id="pos"></span>
<button id="next" type="button">Next &rarr;</button></nav>
<script>{JS}</script>'''

open(OUT, 'w').write(page)
print(OUT, len(page), 'slides:', len(slides), 'figures:', page.count('data:image/png'))
