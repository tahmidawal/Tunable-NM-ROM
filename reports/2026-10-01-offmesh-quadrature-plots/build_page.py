"""Build index.html around figs/ (from make_plots.py). Numbers in captions come from captions.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "captions.json")))
f = lambda v, d=2: f"{v:.{d}f}"
e512, e256 = C["3d_err_512"], C["3d_err_256"]
m512 = C["3d_ms_512"]
a2, f2 = C["2d_err_acc"], C["2d_err_fast"]

FIGS = [
 ("3D", "3d_error_vs_mesh", "The new rule's error does not depend on the mesh.",
  f"Red (old tensor) falls from {f(e512['tensor'][0],1)}% to {f(e512['tensor'][2],1)}% only because the mesh gets finer. "
  f"Blue (new off-mesh rule) is flat at {f(e512['off'][0])}% on every mesh. Grey is the best full-solver setting on the same mesh.",
  "Lower is better. Each point is the worst error over 32 held-out test cases, measured against a refined (first-order) reference. "
  "Caveat: at 256³ the full solver is more accurate, and the reference itself is first-order, which the next experiments will fix."),
 ("3D", "3d_rule_ladder", "Lattice points beat Gauss points for the same budget; Sobol and Smolyak fail.",
  "Each line shows how the error of computing the nonlinear term falls as you add points. Yellow (lattice) sits below blue (Gauss) everywhere. "
  "Points above the dashed line fail the project's acceptance bar.",
  "Log–log axes. Measured on validation states; one random lattice shift."),
 ("3D", "3d_time_vs_mesh", "The new rule is cheaper than the old one; the full solver only loses on big meshes.",
  f"At R′ = 512 the new rule (blue) costs {f(m512['off'][0],1)} → {f(m512['off'][2],1)} ms against {f(m512['tensor'][0],1)} → {f(m512['tensor'][2],1)} ms for the tensor (red). "
  f"Grey is the full solver at the same accuracy on the same mesh: far cheaper at 64³ ({f(m512['fom'][0],1)} ms), far more expensive at 256³ ({f(m512['fom'][2],1)} ms).",
  "Log scale. Both reduced models still grow with the mesh, because reading the input and writing the output touch every mesh node."),
 ("3D", "3d_memory", "The new rule needs a small fraction of the memory.",
  "The old tensor stores a table that grows with the square of the bank width; the new rule stores only the bank evaluated at its points.", ""),
 ("2D", "2d_error_vs_mesh", "Same story in 2D: the new rule is flat across meshes and best on coarse meshes.",
  f"Accurate setting: new Gauss 96² stays at {f(a2['new'][0])}% while the dense mesh solve goes {f(a2['dense'][0])}% → {f(a2['dense'][2])}%. "
  f"Fast setting: new Fibonacci 1597 stays at {f(f2['new'][0])}%.",
  f"Worst error over the 64 test cases against a space-and-time refined reference. At 4096² the dense solve was run on only {a2['n_dense'][2]} cases, "
  "so its last point is not directly comparable."),
 ("2D", "2d_solve_time", "Solve time stays flat as the mesh grows from 256² to 4096².",
  "Every rule is a flat line: cost depends on the number of points, not on the mesh. The old 63² lattice (red, dashed) is shown for comparison.",
  "Measured on one H200 in the same job. The full query (including reading input and writing output on the mesh) still grows with the mesh."),
 ("2D", "2d_rule_ladder", "In 2D, Gauss and Fibonacci converge; Sobol is slow and Smolyak fails.",
  "Our 2D bank needs thousands of points before Gauss or Fibonacci pass the bar, far more than Hari's smaller model did.",
  "Measured on the states reached on the 1024² test runs, accurate setting. The red triangle is the old mesh lattice measured against its own (mesh) target, so it is not directly comparable."),
 ("Mechanism", "a1_error_vs_mesh", "New (8 Oct): the accuracy comes from the exact gradient, not from leaving the mesh.",
  "Pink is a control: every mesh node, but the bank's exact derivative instead of the upwind stencil. In 3D it lands on top of the off-mesh rule (yellow/orange) and the converged rollout. Blue/red, the old stencil-based methods, carry the mesh's error. So the exact gradient gives the accuracy, and the classical rule gives the same answer with thousands of points instead of every mesh node.",
  "Top row: worst error vs the refined reference (provisional: the reference is first-order). Bottom row: distance to the converged off-mesh rollout. Preliminary (lane jcp-mechanism, Codex-audited). In 2D the pre-registered label could not be applied, because the gap was below its 1 % threshold, but the distances point the same way."),
 ("Mechanism", "a2_gap_vs_h", "New (8 Oct): the old method's error is exactly the stencil's order.",
  "The difference between the mesh's own sum and the true integral falls with slope 1 for the first-order upwind stencil and slope 2 for a second-order central stencil. With the exact gradient (green) it drops to the noise floor.",
  "Measured on fixed saved states; slopes are fitted over a pre-registered window. Dashed lines: a manufactured test state run through the same code (control)."),
]

cards = []
for sec, fn, head, body, note in FIGS:
    cards.append(f'''<figure class="card" data-sec="{sec}">
  <figcaption><span class="tag">{'Why it works' if sec == 'Mechanism' else 'Burgers ' + sec}</span><h2>{head}</h2><p>{body}</p></figcaption>
  <img src="figs/{fn}.png" alt="{head}" loading="lazy">
  {f'<p class="note"><strong>How to read it:</strong> {note}</p>' if note else ''}
</figure>''')
page = open(os.path.join(HERE, "template.html")).read().replace("{{CARDS}}", "\n".join(cards))
open(os.path.join(HERE, "index.html"), "w").write(page)
print("ok")
