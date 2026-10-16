"""Build index.html: the JCP campaign progress page (what we did + results).

The 3D rule table is generated from the quadrature-burgers3d lane's select-ho.json (held-out cohort).
Other numbers are quoted from the lanes' generated reports and the notebook (sources listed on the page).
"""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEL = os.path.join(ROOT, "worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/results/select-ho.json")

# ---------------------------------------------------------------- 3D rule table from data
sel = json.load(open(SEL))
RULES = [("tensor", "Old: exact tensor (mesh)", True), ("lat4096", "Lattice", False), ("lat16384", "Lattice", False),
         ("kor16381", "Korobov lattice", False), ("gl16", "Gauss 16³", False), ("gl24", "Gauss 24³ (deployed)", False),
         ("gl32", "Gauss 32³", False), ("sob16384", "Sobol", False), ("lat256", "Lattice (too few points)", False),
         ("smol8", "Smolyak sparse grid", False)]
MESHES = ["65", "129", "257"]


def row3d(key, label, old, R="512"):
    rows = {m: sel["meshes"][m]["R"][R]["rows"].get(f"{key}_R{R}") for m in MESHES}
    r257 = rows["257"]
    m = r257.get("m")
    fail = r257["refined_worst"] > 0.2
    if old:
        err = " → ".join(f"{100*rows[mm]['refined_worst']:.1f}" for mm in MESHES) + " %"
    else:
        err = f"{100*r257['refined_worst']:.1f} / {100*r257['refined_median']:.1f} %"
    cells = []
    for mm in MESHES:
        sp = rows[mm].get("speedup_same_rule")
        cells.append("—" if fail or sp is None else f"{sp:.2f}×" if sp < 1 else f"<b>{sp:.1f}×</b>")
    ref = r257.get("speedup_refined_rule")
    refc = "—" if fail or ref is None or ref == r257.get("speedup_same_rule") else f"{ref:.1f}×"
    ms = f"{r257['ms']:.0f}" if r257.get("ms") else "—"
    cls = ' class="old"' if old else (' class="fail"' if fail else "")
    pts = "—" if m is None else f"{m:,}"
    errc = f'<span class="bad">fails ({err})</span>' if fail else err
    return f"<tr{cls}><td>{label}</td><td>{pts}</td><td>{errc}</td>" + "".join(f"<td>{c}</td>" for c in cells) + f"<td>{refc}</td><td>{ms}</td></tr>"


T3D = "\n".join(row3d(*r) for r in RULES)


def fig(name, alt, cap):
    return f'<figure><img src="figs/{name}" alt="{html.escape(alt)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


def read(txt):
    return f'<details><summary>How to read this plot</summary><p>{txt}</p></details>'


page = f"""<title>JCP Progress Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#f3f5f7;--paper:#ffffff;--ink:#16212a;--muted:#5a6772;--acc:#0b5d7a;--acc2:#b4541a;--soft:#e6f0f4;--line:#d9e0e6;--good:#1f7a4d;--bad:#b3261e;--old:#f6efe9;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0e1418;--paper:#161f26;--ink:#e2e9ee;--muted:#98a6b1;--acc:#6cc0de;--acc2:#f0a06b;--soft:#1b2c35;--line:#2a3640;--good:#5fcf98;--bad:#ff8a80;--old:#2a221c;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0e1418;--paper:#161f26;--ink:#e2e9ee;--muted:#98a6b1;--acc:#6cc0de;--acc2:#f0a06b;--soft:#1b2c35;--line:#2a3640;--good:#5fcf98;--bad:#ff8a80;--old:#2a221c;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 "Source Sans 3",system-ui,-apple-system,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding-inline:16px;padding-block:28px 56px}}
h1,h2,h3{{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance;line-height:1.2}}
h1{{font-size:clamp(1.6rem,4vw,2.2rem);margin:0 0 .4rem;color:var(--acc)}}
h2{{font-size:1.45rem;margin:2.4rem 0 .8rem;padding-top:.6rem;border-top:2px solid var(--line)}}
h3{{font-size:1.12rem;margin:1.8rem 0 .4rem}}
p,li{{max-width:72ch}}
.status{{background:var(--soft);border-left:4px solid var(--acc);padding:.7rem 1rem;border-radius:4px;max-width:none}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:1.2rem 0}}
.tile{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.tile b{{display:block;font-size:1.5rem;color:var(--acc);font-variant-numeric:tabular-nums}}
.tile span{{color:var(--muted);font-size:.9rem}}
nav ol{{columns:2;column-gap:2rem;padding-left:1.2rem;margin:.4rem 0}} @media (max-width:560px){{nav ol{{columns:1}}}}
nav a{{color:var(--acc)}}
figure{{margin:1rem 0;background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:10px}}
figure img{{width:100%;height:auto;display:block;background:#fff;border-radius:4px}}
figcaption{{color:var(--muted);font-size:.9rem;margin-top:.5rem}}
details{{margin:.4rem 0 1rem;max-width:72ch}} summary{{cursor:pointer;color:var(--acc);font-weight:600}}
.take{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--good);border-radius:6px;padding:.6rem .9rem;max-width:72ch}}
.limit{{border-left-color:var(--acc2)}}
.tw{{overflow-x:auto;margin:1rem 0;background:var(--paper);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}}
th,td{{padding:.45rem .6rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft);font-weight:600}}
tr.old td{{background:var(--old)}} .bad{{color:var(--bad);font-weight:600}}
.steps{{counter-reset:s;list-style:none;padding:0;display:grid;gap:10px}}
.steps li{{counter-increment:s;background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:.6rem .9rem .6rem 3rem;position:relative;max-width:none}}
.steps li::before{{content:counter(s);position:absolute;left:.9rem;top:.55rem;width:1.5rem;height:1.5rem;border-radius:50%;background:var(--acc);color:var(--paper);display:grid;place-items:center;font-weight:700;font-size:.85rem}}
.tag{{font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;padding:.1rem .45rem;border-radius:3px;background:var(--soft);color:var(--acc);margin-left:.3rem}}
.formula{{font-family:ui-monospace,Menlo,monospace;background:var(--paper);border:1px solid var(--line);padding:.6rem .9rem;border-radius:6px;display:inline-block}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1rem;max-width:80ch}} dt{{font-weight:600}} dd{{margin:0;color:var(--muted)}}
@media (max-width:560px){{dl{{grid-template-columns:1fr}}}}
</style>
<div class="wrap">
<h1>Tunable bank + off-mesh quadrature: progress report</h1>
<p style="color:var(--muted);margin-top:0">JCP paper campaign · 16 October 2026 · Burgers 2D and 3D</p>
<p class="status"><b>Status of the numbers.</b> Sections 3–7 are scored against the <b>older first-order references</b> and are <b>provisional</b> (3D numbers can move by up to about 2.8 %). Section 8 is scored against the <b>new order-verified second-order references</b>. All results come from validation or held-out cases; the sealed test cohorts have not been opened.</p>

<div class="tiles">
<div class="tile"><b>2.8 %</b><span>3D worst error, flat at every mesh (old method: 10.5 → 3.9 %)</span></div>
<div class="tile"><b>13× less</b><span>memory for the nonlinear term (4.3 GB → 0.34 GB)</span></div>
<div class="tile"><b>5× / 13×</b><span>lower typical error at equal cost, 2D / 3D, against the new references</span></div>
<div class="tile"><b>≈14×</b><span>2D speed-up against a full simulation that is second order in time, at equal accuracy</span></div>
</div>

<nav><b>Contents</b><ol>
<li><a href="#idea">The idea</a></li><li><a href="#did">What we did</a></li>
<li><a href="#acc">Mesh-independent accuracy</a></li><li><a href="#cost">Mesh-independent cost and memory</a></li>
<li><a href="#rules">Which quadrature rules work</a></li><li><a href="#table">Every rule as a full solve: accuracy vs speed</a></li>
<li><a href="#why">Why it works</a></li><li><a href="#now">Where we are now (new references)</a></li>
<li><a href="#limit">The honest limit: 3D speed</a></li><li><a href="#next">Next steps</a></li>
<li><a href="#gloss">Glossary</a></li></ol></nav>

<h2 id="idea">1 · The idea</h2>
<p>Our reduced model writes the solution as a weighted mix of learned shapes, <i>u(x) = G(x)c</i>, and solves only for the few weights <i>c</i>. The bank is ordered, so using the first <i>R′</i> shapes gives a cheaper, rougher model: that is the accuracy-versus-cost dial. The expensive part is the nonlinear term <i>u∇u</i>.</p>
<ul>
<li><b>Before:</b> its integral was computed <b>on the simulation mesh</b>, by a dense sum over all nodes, fitted empirical quadrature, or a precomputed exact tensor. Cost and error depended on the mesh, and the tensor's memory grows with <i>R′</i>².</li>
<li><b>Hari's idea:</b> the bank is a smooth formula, so evaluate it and its <b>exact derivative</b> (automatic differentiation) at a few thousand <b>textbook quadrature points</b>: Gauss, rank-1 lattices or Sobol. Three matrices are precomputed once, and each solver step is then
<br><span class="formula">N(c) = Pᵀ [ (B c) ⊙ (D c) ]</span><br>
with no fitting and the same points at every mesh.</li>
</ul>

<h2 id="did">2 · What we did</h2>
<ol class="steps">
<li><b>Plugged Hari's rules into our model</b> and ran every rule family as a <b>full Burgers solve</b> in 2D (256²–4096²) and 3D (64³–256³): Gauss, Fibonacci/CBC/Korobov lattices, Sobol, Smolyak and the flux form, against the old mesh methods (dense sum, mesh lattice, exact tensor). <span class="tag">Oct 1–2</span></li>
<li><b>Found why it works</b>: a mechanism study comparing the exact derivative with mesh differencing. <span class="tag">Oct 8</span></li>
<li><b>Fixed the next bottleneck</b> with second-order time stepping (CN-R, BDF2, Galerkin). <span class="tag">Oct 8–9</span></li>
<li><b>Tested what else limits accuracy</b>: a wider bank (up to 1024 shapes) and smoother or derivative-trained banks. <span class="tag">Oct 8–9</span></li>
<li><b>Built trustworthy references</b>: order-verified second-order solutions with a spectral cross-check, plus sealed held-out cohorts. <span class="tag">Oct 13–14</span></li>
<li><b>Combined the winners</b>, second-order stepping with the wide bank, and scored them against the new references, with a per-case error budget. <span class="tag">Oct 13–15</span></li>
<li><b>Now running</b>: 3D speed, with a one-pass solver at large steps, a narrower bank, batching, and a large-mesh comparison against a spectral solver. <span class="tag">in progress</span></li>
</ol>
<p style="color:var(--muted)">Every step was pre-registered and adversarially checked by an independent reviewer (Codex) before it ran.</p>

<h2 id="acc">3 · Accuracy no longer depends on the mesh</h2>
{fig("3d_error_vs_mesh.png", "3D error vs mesh", "Burgers 3D, worst error over held-out cases. Left: 512-wide bank; right: 256-wide.")}
{read("Mesh size goes across, from coarse to fine; worst error goes up, and lower is better. Orange is the old exact-tensor method, blue is Hari's off-mesh rule, and grey dashed is the best full simulation at that mesh.")}
<p class="take"><b>Takeaway:</b> the old method improves only as the mesh refines (10.5 → 6.1 → 3.9 %) because it inherits the mesh's error. Off-mesh is <b>flat at 2.83 %</b> and beats the full simulation on the coarse meshes.</p>
{fig("2d_error_vs_mesh.png", "2D error vs mesh", "Burgers 2D, 64 held-out cases. The dense method at 4096² ran on only 6 cases.")}
<p class="take">The same in 2D: flat at 2.6 % (accurate setting) and 4.4 % (fast setting), against 5.6–6.3 % for the mesh methods on the coarse mesh.</p>

<h2 id="cost">4 · Cost and memory don't depend on the mesh either</h2>
{fig("2d_solve_time.png", "2D solve time vs mesh", "Solve time per query, each line one quadrature rule, from 256² to 4096².")}
{read("Every line is horizontal: a 256× larger mesh costs no extra time. Rules with more points sit higher, but mesh size never matters.")}
{fig("c1_3d_j4_memory_vs_Rp.png", "Memory vs bank width", "Memory of the nonlinear term against bank width (log scale). The dashed line is a whole 141 GB GPU.")}
<p class="take"><b>Takeaway:</b> the old tensor needs 4.3 GB at width 512 and 34 GB at 1024, and stops fitting past about 1,600. Off-mesh stays at 0.1–1.6 GB, which is what makes wide banks possible.</p>

<h2 id="rules">5 · Which quadrature rules work</h2>
{fig("3d_rule_ladder.png", "3D rule ladder", "3D: error in the nonlinear integral (ρ) against number of points, at three meshes. The dashed line is the acceptance bar.")}
{fig("2d_rule_ladder.png", "2D rule ladder", "2D at 1024²: ρ against number of points for each rule family.")}
<p class="take"><b>Ranking:</b> lattices and Gauss converge fastest. The lattice is about 5–10× better than Gauss at equal points in 3D, and Gauss is best per point in 2D. Sobol needs about 4× more points. Halton is weaker than Sobol. Smolyak fails at 2–3D for these oscillatory integrands. The three 3D panels look identical: the rules don't care about the mesh.</p>

<h2 id="table">6 · Every rule as a full Burgers solve: accuracy vs speed</h2>
<p><b>3D, held-out cases (64), 512-wide bank.</b> Error is worst / typical; for off-mesh rules it is the same at every mesh. The speed-up is the time of the cheapest full simulation that is at least as accurate, divided by our time, in the same job on the same GPU. Below 1× means slower. The "finer reference" column requires the full simulation to match our accuracy against the finer reference.</p>
<div class="tw"><table>
<tr><th>Method</th><th>Points</th><th>Error</th><th>64³</th><th>128³</th><th>256³</th><th>256³, finer ref.</th><th>ms at 256³</th></tr>
{T3D}
</table></div>
<p><b>2D, held-out cases (64), accurate setting.</b> Worst error at 256² / 1024² / 4096²; speed-up at 4096² against the cheapest equally accurate full simulation. These ratios were worked out from the study's timing table, not taken from a matched panel.</p>
<div class="tw"><table>
<tr><th>Method</th><th>Points</th><th>Worst error</th><th>Speed-up at 4096²</th></tr>
<tr class="old"><td>Full simulation (reference point)</td><td>—</td><td>5.5 / 2.9 / 2.4 %</td><td>1×</td></tr>
<tr class="old"><td>Old: dense mesh sum</td><td>all nodes</td><td>5.6 / 3.3 / 1.6 %</td><td>—</td></tr>
<tr class="old"><td>Old: 63² mesh lattice</td><td>3,969</td><td>5.6 / 3.3 / 1.6 %</td><td>11×</td></tr>
<tr><td>Gauss 64²</td><td>4,096</td><td><b>2.6 / 2.6 / 1.6 %</b></td><td><b>13×</b></td></tr>
<tr><td>Gauss 96² (deployed)</td><td>9,216</td><td>2.6 / 2.6 / 1.6 %</td><td>flat solve, about 55 ms</td></tr>
<tr><td>Gauss 128²</td><td>16,384</td><td>2.6 / 2.6 / 1.6 %</td><td>7.4×</td></tr>
<tr><td>Fibonacci lattice</td><td>6,765</td><td>2.6 / 2.6 / 1.6 %</td><td>10×</td></tr>
<tr><td>Fibonacci lattice</td><td>17,711</td><td>2.6 / 2.6 / 1.6 %</td><td>5.7×</td></tr>
<tr><td>Sobol</td><td>4,096</td><td>2.8 / 2.8 / 1.6 %</td><td>not timed</td></tr>
<tr class="fail"><td>Gauss 32² (too few points)</td><td>1,024</td><td><span class="bad">fails (9.0 / 8.8 / 2.7 %)</span></td><td>—</td></tr>
</table></div>
<p class="take"><b>Takeaway:</b> every working rule gives the same accuracy, so the choice of rule hardly matters once it passes the bar. Speed depends on mesh size: <b>slower on small meshes, about even in the middle, and clearly faster on large ones</b>. That means 2–3× at 256³ (up to 7.8× against the finer reference) and 6–32× at 4096² in 2D. The fast 2D setting reaches 20–32× at 4096². Not yet run as full solves: Halton and Monte Carlo.</p>

<h2 id="why">7 · Why it works: the exact derivative</h2>
{fig("a2_gap_vs_h.png", "Gap vs mesh spacing", "Gap between the mesh-based nonlinear integral and the true integral, against mesh spacing h (finer to the right).")}
{read("Blue is first-order upwind differencing and falls with slope 1. Orange is second-order central differencing and falls with slope 2. Green is mesh nodes used with the exact derivative, thousands of times smaller.")}
{fig("a1_error_vs_mesh.png", "Mechanism arms", "Top row: worst error. Mesh nodes plus the exact derivative (overlapping the off-mesh lines at the bottom) match Hari's rule.")}
<p class="take"><b>Takeaway:</b> the old error was the mesh's own derivative approximation, shrinking exactly at the rate theory predicts. Using the exact derivative removes it, even on mesh nodes. Hari's points make that cheap: about 14k points instead of 250k–2M nodes.</p>

<h3>Then time stepping was the bottleneck</h3>
{fig("c2_error_vs_cost_acc.png", "Accuracy vs cost per time scheme", "2D accurate setting: typical error against time per query, one line per time scheme. The circle is the old backward-Euler setting.")}
{read("Bottom-left is best. The points along a line are different time-step sizes.")}
<p class="take">At equal cost, the second-order schemes (CN-R, BDF2) have about 9× lower typical error than backward Euler. Making the bank wider or smoother changed little until this was fixed.</p>

<h2 id="now">8 · Where we are now, against the new order-verified references</h2>
<p>We built second-order reference solutions and verified their order case by case. 3D passes every case, with observed order 1.99–2.02. 2D passes 191 of 204 cases; the other 13 are flagged and not ranked. A spectral solver agrees on 8 of 8 cross-check cases in each dimension. The old references were off by up to 2.8 % in 3D and 0.35 % in 2D.</p>
{fig("e0_old_vs_new.png", "Old vs new references", "How far the old first-order references were from the new ones (vertical), against the new references' own estimated error (horizontal). Every point sits far above the line, so the old references were the limiting factor.")}
<div class="tw"><table>
<tr><th>Second-order stepping with the wide bank</th><th>Before (backward Euler)</th><th>Now, same cost</th></tr>
<tr><td>2D typical error (accurate setting)</td><td>0.98 %</td><td><b>0.20 %</b></td></tr>
<tr><td>2D worst error (accurate setting)</td><td>3.0 %</td><td><b>1.8 %</b></td></tr>
<tr><td>2D speed-up against a full simulation second order in time, at equal accuracy</td><td>≈4×</td><td><b>≈14×</b></td></tr>
<tr><td>3D typical error (1024-wide bank)</td><td>1.53 %</td><td><b>0.12 %</b></td></tr>
<tr><td>3D worst error (selected arms)</td><td colspan="2">still 4.4–6.7 %, an open problem</td></tr>
</table></div>
{fig("s2_pareto2d.png", "2D error vs cost", "2D: typical error against relative cost. Hollow circles are the old model; filled triangles are the second-order versions; colour is bank width.")}
<p class="take"><b>Takeaway:</b> every old setting has a new setting that is both cheaper and more accurate, and the width dial still works. In 2D, wider banks pay off up to 256–384 shapes; in 3D, 512 → 768 → 1024 keeps improving.</p>

<h2 id="limit">9 · The honest limit: 3D speed</h2>
{fig("3d_time_vs_mesh.png", "3D cost vs mesh", "3D cost per query against mesh size (log scale). Grey dashed is the full simulation tuned to our accuracy.")}
<p class="take limit">We win on speed only on large meshes (2.6× at 256³). In 3D, second-order stepping bought accuracy, not speed: at equal accuracy it is only 4–15 % faster. Early runs that have not yet been audited show a <b>spectral solver beating us clearly at 64³</b> (0.04 % typical error in about 4 ms). The 3D speed lane is now testing a one-pass solver, narrower banks, batching, and comparisons against spectral at 128³–384³.</p>

<h2 id="next">10 · Next steps</h2>
<ul>
<li>Finish the 3D speed lane, including the large-mesh comparison against spectral and the full-simulation ladder.</li>
<li>Standard reduced-model baselines: POD with DEIM, ECSW and GNAT, plus fitted quadrature on our bank.</li>
<li>Re-score the rule comparison (section 6) against the new references, and add Halton and Monte Carlo as full solves.</li>
<li>A harder example where spectral solvers struggle: non-box geometry, sharp fronts, a non-polynomial nonlinearity.</li>
<li>Then open the sealed test cohorts for the final paper numbers.</li>
</ul>
<p><b>Corrections made along the way</b>, kept on record:</p>
<ul>
<li>An early 1.43 % 3D preview became 3.4 % on the full set.</li>
<li>The old-reference error was underestimated: it is up to 2.8 %, not 1.7–2.3 %.</li>
<li>The 3D rule for the wide banks was chosen after seeing data, and is labelled post hoc.</li>
<li>In 2D, one cross-check rule was changed after seeing data; both verdicts are reported.</li>
</ul>

<h2 id="gloss">Glossary</h2>
<dl>
<dt>Bank</dt><dd>The trained set of smooth shapes the solution is built from; R′ = how many are used.</dd>
<dt>Off-mesh quadrature</dt><dd>Computing the nonlinear integral at chosen points, not at mesh nodes (Hari's idea).</dd>
<dt>Gauss rule</dt><dd>The classic best points per axis, repeated on a grid (n³ points in 3D).</dd>
<dt>Lattice (Fibonacci, CBC, Korobov)</dt><dd>An evenly tilted point grid with equal weights; any point count; a random shift gives error bars.</dd>
<dt>Sobol, Halton</dt><dd>Quasi-random space-filling sequences.</dd>
<dt>Smolyak</dt><dd>A sparse grid built for high dimensions; poor for our 2–3D oscillatory integrals.</dd>
<dt>ρ</dt><dd>Error of the nonlinear integral relative to its size; the acceptance bar is 0.116.</dd>
<dt>Worst / typical error</dt><dd>Largest and median error over the cases, as % of the initial field.</dd>
<dt>Speed-up</dt><dd>Time of the cheapest full simulation at least as accurate, divided by our time; same GPU and job.</dd>
<dt>First- vs second-order references</dt><dd>"Truth" solutions; first-order ones carry up to 2.8 % error; second-order ones are order-verified and accurate to about 0.003 % in 3D.</dd>
<dt>Backward Euler, CN-R, BDF2</dt><dd>Time-stepping schemes; backward Euler is first order, the others are second order.</dd>
<dt>Spectral solver</dt><dd>A full simulation using global Fourier or sine modes; very efficient for smooth problems in a box.</dd>
<dt>Provisional</dt><dd>Scored against the old first-order references; will be re-scored.</dd>
</dl>
<p style="color:var(--muted);font-size:.85rem;margin-top:2rem">Sources: lane reports on branches exp/2026-10-01-quadrature-{{study,burgers3d}}, exp/2026-10-06-jcp-mechanism, exp/2026-10-08-jcp-{{time2,wide-bank,smooth-bank}}, exp/2026-10-06-jcp-references, exp/2026-10-13-jcp-combined, and the campaign notebook. The 3D table is generated from select-ho.json by make_page.py.</p>
</div>
"""
open(os.path.join(HERE, "index.html"), "w").write(page)
print("wrote index.html")
