"""Build index.html: projection plots for steps 1-3 with expectation and how-to per plot."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
P = [
 ("Step 1", "p1_cost_vs_mesh.png", "Solve cost stays flat as the mesh grows",
  "Flat lines: the same time at 256², 1024² and 4096² for each k. Bigger k sits higher. The k = 128 and k = 384 lines are measured; k = 256 is projected.",
  "Experiment 1c. Run the 2D model at k = 128, 256, 384 on all three meshes in one GPU job, timing the solve separately from start-up and output, with 6 cases × 3 repeats, side by side."),
 ("Step 1", "p2_cost_vs_k.png", "Cost grows with k, part by part",
  "Straight lines on a log–log plot. The nonlinear term and the Jacobian grow about in proportion to k; the small linear solve grows fastest (about k³) but stays cheap until k is large. Shapes only: no measured values here.",
  "Experiment 1a. Time each solver part separately for k = 32–512 at 1024², with the number of tests and points held fixed, on 12 saved solution states. Fit a slope for each part and compare it with theory."),
 ("Step 1", "p3_heatmap.png", "Cost depends on k, not the mesh",
  "Each column has one colour: changing the mesh (rows) changes nothing, and changing k (columns) changes everything. The numbers are measured relative costs at 1024², repeated down the rows as the projection.",
  "Built from the same runs as plots 1 and 2: cost for every (k, mesh) pair."),
 ("Step 1", "p4_acc_vs_mesh.png", "Accuracy stays flat across meshes",
  "Flat lines for each k. Filled points are measured at 1024² against the trustworthy references (0.45 %, 0.21 %, 0.20 %); the hollow points at other meshes are projected.",
  "Experiment 1d. Run k = 128, 256, 384 at 256² and 4096² (1024² is reused), score against the order-verified references, and keep only settings that pass the soundness checks."),
 ("Step 1", "p5_acc_vs_k.png", "Accuracy improves with k, then levels off",
  "Measured at 1024²: a big gain from 128 to 256, then little or none. Step 1 repeats this at the other meshes.",
  "Experiment 1d, typical error against k at each mesh. Every step from one k to the next is reported, including the ones that don't improve."),
 ("Step 1", "p6_memory.png", "Memory: off-mesh stays small, the tensor explodes",
  "The old exact tensor passes a whole GPU at about k = 1,600. The off-mesh tables stay a few GB. This is arithmetic from the storage formulas; the number of points per k is assumed.",
  "Experiment 1g. Measure peak GPU memory in every Step 1 job, and plot it next to the formulas."),
 ("Step 2", "p7_speedup_vs_mesh.png", "Speed-up against each kind of full simulation",
  "Big filled dot: <b>measured 13.6× at 1024²</b> against a simulation that is second order in time. Shaded bands are projections. We lose on small meshes and win on big ones against standard simulations. Against <b>spectral</b> the band straddles break-even: that's the main risk.",
  "Experiment 2c. For each simulation type, sweep time step × tolerance × simulation mesh. Pick the cheapest setting at least as accurate as ours, chosen on practice cases, then time both side by side on the same GPU."),
 ("Step 2", "p8_vs_pod.png", "Ours vs a standard POD reduced model",
  "Projected: POD needs a bigger size for the same error on Burgers. The shaded band is wide because nothing is built yet; it could also land on our line.",
  "Experiment 2b. Build POD with ECSW and Q-DEIM at the same and smaller sizes, tune each for equal effort on practice cases, and compare error and time on the same cases."),
 ("Step 2", "p9_dial.png", "The accuracy-for-cost dial",
  "Measured at 1024²: the new second-order model (blue) sits far below the old backward-Euler model (grey) at the same cost, for every k. Step 2 repeats this at all meshes, with only settings that pass the soundness checks.",
  "Experiment 2d. Error against cost for each k and time step, after the soundness checks (2a)."),
 ("Step 3", "p10_points_vs_k.png", "Points needed grow with the number of tests",
  "Points needed rise roughly in proportion to k (because tests = 4k). Using fewer mixed-frequency tests could cut points about 3×: an earlier run needed about 3× fewer points with only the first 320 tests.",
  "Experiments 3a/3b. For each k and test count, find the smallest number of points that passes the accuracy bar. Repeat with a reduced test set."),
 ("Step 3", "p11_rules.png", "Every quadrature type, compared",
  "Gauss, Fibonacci and Sobol lines are <b>measured</b> integral errors (2D, 1024²). Monte Carlo is projected: it falls slowest, as 1/√points. Step 3 turns these into full solves and adds Halton and Monte Carlo.",
  "Experiment 3c. Run each rule type at increasing point counts as a full solve; Smolyak is kept as a must-fail check."),
 ("Step 3", "p12_budget.png", "Where the error comes from",
  "Illustrative, shaped after Step 2's error budget: time stepping is the biggest part; the bank's own limit matters most at small k; quadrature and solver errors are tiny.",
  "Experiment 3e. Switch one ingredient at a time to its 'perfect' version (finer rule, smaller time step, tighter solver, more tests) and measure how much the error drops."),
 ("Step 3", "p13_long_time.png", "Error over longer runs",
  "The least known. Most likely slow, steady growth past today's final time of 0.25; the shaded range shows it could grow faster.",
  "Experiment 3f. Extend runs past 0.25. This needs new reference solutions for the longer times."),
]
cards = ""; cur = None
for step, f, title, expect, how in P:
    if step != cur:
        cards += f'<h2>{step}</h2>'; cur = step
    cards += f'''<section class="card"><h3>{title}</h3><img src="figs/{f}" alt="{title}" loading="lazy">
<p><b>Expected:</b> {expect}</p><p class="how"><b>How we'd make it:</b> {how}</p></section>'''
html = f'''<title>Projected Results Steps 1–3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#f3f5f7;--paper:#ffffff;--ink:#16212a;--muted:#5a6772;--acc:#0b5d7a;--acc2:#b4541a;--soft:#e6f0f4;--line:#d9e0e6;--warmsoft:#fbf0e6;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0e1418;--paper:#161f26;--ink:#e2e9ee;--muted:#98a6b1;--acc:#6cc0de;--acc2:#f0a06b;--soft:#1b2c35;--line:#2a3640;--warmsoft:#2c2219;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0e1418;--paper:#161f26;--ink:#e2e9ee;--muted:#98a6b1;--acc:#6cc0de;--acc2:#f0a06b;--soft:#1b2c35;--line:#2a3640;--warmsoft:#2c2219;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:17px/1.6 "Source Sans 3",system-ui,sans-serif}}
.wrap{{max-width:820px;margin:0 auto;padding-inline:16px;padding-block:32px 64px}}
h1,h2,h3{{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance;line-height:1.2}}
h1{{font-size:clamp(1.7rem,5vw,2.3rem);margin:0 0 .3rem;color:var(--acc)}}
h2{{font-size:1.45rem;margin:2.4rem 0 .6rem;color:var(--acc)}} h3{{font-size:1.12rem;margin:0 0 .6rem}}
.sub{{color:var(--muted);margin:0 0 1.2rem}}
.warn{{background:var(--warmsoft);border-left:4px solid var(--acc2);border-radius:6px;padding:.9rem 1.1rem}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}}
.card img{{width:100%;height:auto;display:block;background:#fff;border-radius:4px;border:1px solid var(--line)}}
.card p{{margin:.6rem 0 0;max-width:72ch}} .how{{color:var(--muted);font-size:.95rem}}
</style>
<div class="wrap">
<h1>What steps 1–3 should show</h1>
<p class="sub">Projected plots for the JCP experiment plan · 17 October 2026</p>
<div class="warn"><b>These are projections, not results.</b> Every plot is stamped "PROJECTION". <b>Filled markers are real measurements</b> from earlier runs; dashed lines, hollow markers and shaded bands are our best guess. Each plot says what we expect and how the real version will be made.<br><br><b>Correction:</b> the measured 2D speed-up of about 14× was at the <b>1024²</b> mesh, not 4096². At 4096² it should be larger against the same simulation; that is projected here.</div>
{cards}
<p class="how">Plots generated by <code>reports/2026-10-17-full-experiment-plan/projections/make_projections.py</code>. Measured anchors come from the 2D quadrature-study report (solve times, integral errors) and the combined (Step 2) report (errors, costs, the 13.6× speed-up).</p>
</div>'''
open(os.path.join(HERE, "index.html"), "w").write(html)
print("ok")
