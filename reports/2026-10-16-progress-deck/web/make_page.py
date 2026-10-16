"""Build web/index.html: each deck page as an image with its speaker notes underneath."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = [
 ("Title", "<p>Most plots are scored against the <b>older, rough references</b> and are provisional. Plot 7 uses the <b>new order-verified references</b>. That answers “how do you know the true answer?”</p>"),
 ("The idea", "<p>Our model writes the solution as a mix of learned shapes, u(x)=G(x)c, and solves only for the few weights c. The costly part is the nonlinear term u∇u.</p><ul><li><b>Before:</b> computed on the simulation mesh, so cost and error depended on the mesh.</li><li><b>Hari:</b> the shapes are smooth formulas, so evaluate them and their <b>exact derivative</b> at a few thousand textbook quadrature points.</li><li>Precompute B, D, P once; each step is a few small matrix products.</li></ul>"),
 ("1 · Accuracy no longer depends on the mesh (3D)", "<p><b>Read:</b> mesh size from coarse to fine across the bottom; worst error up the side, lower is better. <span class=k>Orange</span> = old method, <span class=k>blue</span> = Hari's, <span class=k>grey dashed</span> = full simulation. Left panel 512 shapes, right panel 256.</p><p><b>Shows:</b> orange falls 10.5 → 3.9 % because it inherits mesh error. Blue is <b>flat at 2.83 %</b> at every mesh, and beats the full simulation on coarse meshes.</p><p class=say>Say: “With Hari's points, accuracy doesn't care about the mesh.”</p>"),
 ("2 · Neither does cost (2D)", "<p><b>Read:</b> mesh size from 256² to 4096² (256× more points); milliseconds per solve. One line per quadrature rule; accurate setting on the left, fast on the right.</p><p><b>Shows:</b> every line is <b>horizontal</b>. More quadrature points cost more, but mesh size never does.</p><p class=say>Say: “Solve cost is independent of mesh resolution.”</p>"),
 ("3 · Memory: wide banks become possible", "<p><b>Read:</b> bank width (number of shapes) against memory in GB, on a log scale. <span class=k>Orange</span> = old exact tensor, <span class=k>blue</span> = Hari's method, dashed line = a whole 141 GB GPU.</p><p><b>Shows:</b> the old method grows with width squared: 4 GB at 512, 34 GB at 1024, and it no longer fits past about 1600. Hari's stays at 0.1–1.6 GB.</p><p class=say>Say: “The old method can't scale to big banks; Hari's makes wide banks possible.”</p>"),
 ("4 · Which quadrature rules work", "<p><b>Read:</b> number of quadrature points against the error in the nonlinear term (log scale). The dashed line is the acceptance bar.</p><p><b>Shows:</b> Gauss (blue) and the Fibonacci lattice (yellow) drop steeply and pass the bar at about 2–4k points. Sobol (grey) needs about 4× more. Smolyak (pink) never makes it.</p><p class=say>Say: “Classical rules work, with no fitting. Gauss for accuracy, lattices for speed.”</p>"),
 ("5 · Why it works: the exact derivative", "<p><b>Read:</b> mesh spacing, getting finer to the right, against the gap between the mesh calculation and the true integral. <span class=k>Blue</span> = old first-order mesh differences, <span class=k>orange</span> = second-order differences, <span class=k>green</span> = mesh nodes with the exact derivative.</p><p><b>Shows:</b> blue falls with slope 1 and orange with slope 2, exactly as theory predicts. Green is thousands of times lower.</p><p class=say>Say: “The old error was the mesh's own derivative approximation. The exact derivative removes it, and Hari's points make it cheap.”</p>"),
 ("6 · Next bottleneck: time stepping", "<p><b>Read:</b> time per solve against typical error. <b>Bottom-left is best.</b> Each line is one time-stepping scheme; its points are different step sizes. The circle is what we used before (backward Euler).</p><p><b>Shows:</b> the second-order schemes (BDF2, CN-R) pass far below the circle at the same cost, with about 9× less error.</p><p class=say>Say: “Once the nonlinear term was solved, time stepping was the limit. Second-order schemes fix it.”</p>"),
 ("7 · Where we are, against the new accurate references", "<p><b>Read:</b> relative cost across the bottom (1 = our standard setting; left is cheaper) against typical error (lower is better). <b>Hollow circles</b> = old backward-Euler model, <b>filled triangles</b> = new second-order versions. Colour = bank width, from grey (narrow) to pink (512).</p><p><b>Shows:</b> every hollow circle has filled triangles below and to the left of it, so the new versions are both cheaper and more accurate. Wider banks cost more and reach lower error, which is the dial.</p><p class=say>Say: “2D is 5× more accurate at the same cost (0.98 → 0.20 %) and about 14× faster than a second-order full simulation. 3D is 13× more accurate at the same cost.”</p>"),
 ("8 · Honest limit: 3D speed", "<p><b>Read:</b> mesh size against milliseconds per solve (log scale). <span class=k>Grey dashed</span> = full simulation, <span class=k>orange</span> = old tensor, <span class=k>blue</span> = Hari's.</p><p><b>Shows:</b> the full simulation's cost rockets with mesh size while ours barely rises. The lines cross: the simulation is faster at 64³ and 128³, and we're 2.6× faster at 256³.</p><p class=say>Say: “We win on speed only at large meshes. Early runs show a spectral solver is very strong at small meshes. Large-mesh comparisons are running.”</p>"),
 ("Next steps", "<ul><li><b>Running:</b> 3D speed lane (one-pass solver, narrower bank, batching, comparison against spectral at large meshes).</li><li><b>Next:</b> standard reduced-model baselines (POD with DEIM, ECSW and GNAT); a harder example where spectral solvers struggle; opening the held-back test cases.</li><li><b>Done this cycle:</b> accurate reference solutions; second-order stepping combined with the wide bank.</li></ul>"),
]
imgs = sorted(f for f in os.listdir(os.path.join(HERE, "s")) if f.endswith(".png"))
assert len(imgs) == len(NOTES), (len(imgs), len(NOTES))
cards = "\n".join(f'''<section class=card id="s{i+1}"><div class=num>Page {i+1} / {len(NOTES)}</div><h2>{t}</h2>
<img src="s/{f}" alt="{t}" loading="lazy"><div class=notes>{n}</div></section>''' for i, (f, (t, n)) in enumerate(zip(imgs, NOTES)))
html = f'''<title>Progress Deck 16 Oct</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#f4f6f8;--card:#ffffff;--ink:#17222b;--muted:#5b6873;--acc:#0b5d7a;--say:#e8f2f6;--line:#dde3e8;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1519;--card:#172027;--ink:#e3eaef;--muted:#9aa8b3;--acc:#6cc0de;--say:#1c2f38;--line:#2a363f;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f1519;--card:#172027;--ink:#e3eaef;--muted:#9aa8b3;--acc:#6cc0de;--say:#1c2f38;--line:#2a363f;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;line-height:1.5}}
main{{max-width:980px;margin:0 auto;padding-inline:16px;padding-block:24px 48px;display:grid;gap:28px}}
header h1{{margin:0;font-size:1.7rem;color:var(--acc);text-wrap:balance}} header p{{margin:.3rem 0 0;color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}
.num{{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
h2{{margin:.2rem 0 .8rem;font-size:1.2rem;text-wrap:balance}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}}
.notes{{max-width:70ch}} .notes p,.notes ul{{margin:.6rem 0}}
.say{{background:var(--say);border-left:3px solid var(--acc);padding:.5rem .8rem;border-radius:4px}}
.k{{font-weight:600}}
</style>
<main><header><h1>Progress deck: tunable bank + off-mesh quadrature</h1>
<p>11 slides, each with how to read the plot and what to say. Accuracy numbers on plots 1–6 and 8 are provisional (older references); plot 7 uses the new order-verified references.</p></header>
{cards}</main>'''
open(os.path.join(HERE, "index.html"), "w").write(html)
print("wrote", os.path.join(HERE, "index.html"))
