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


page = f"""<title>Off-Mesh Quadrature Results</title>
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
<h1>Off-mesh quadrature: results</h1>
<p style="color:var(--muted);margin-top:0">Burgers 2D and 3D · 16 October 2026</p>
<p class="status">All numbers are from validation or held-out cases, scored against the older first-order references, so they are <b>provisional</b> (3D can move by up to about 2.8 %).</p>
<nav><b>Contents</b><ol>
<li><a href="#rules">Quadrature rules vs what we had</a></li><li><a href="#acc">Accuracy no longer depends on the mesh</a></li>
<li><a href="#table">Every rule as a full Burgers solve: accuracy vs speed</a></li></ol></nav>

<h2 id="rules">1 · Quadrature rules vs what we had</h2>
{fig("3d_rule_ladder.png", "3D rule ladder", "3D: error in the nonlinear integral (ρ) against number of points, at three meshes. The dashed line is the acceptance bar. Lattice and Gauss converge fast; Sobol needs more points; Smolyak and the 256-point lattice fail. The panels are identical: the rules don't depend on the mesh.")}
{fig("2d_rule_ladder.png", "2D rule ladder", "2D at 1024²: ρ by rule. The orange triangle is the old 63² mesh lattice, measured against its own mesh target.")}

<h2 id="acc">2 · Accuracy no longer depends on the mesh</h2>
{fig("3d_error_vs_mesh.png", "3D error vs mesh", "3D, held-out cases: the old exact tensor goes 10.5 → 6.1 → 3.9 %; off-mesh is flat at 2.83 % (512-wide bank) and 4.49 % (256-wide).")}
{fig("2d_error_vs_mesh.png", "2D error vs mesh", "2D, 64 held-out cases: off-mesh is flat at 2.6 % (accurate) and 4.4 % (fast); the mesh methods start at 5.6–6.3 %.")}

<h2 id="table">3 · Every rule as a full Burgers solve: accuracy vs speed</h2>
<p><b>3D, held-out cases (64), 512-wide bank.</b> Error is worst / typical (the same at every mesh for off-mesh rules). The speed-up is the time of the cheapest full simulation that is at least as accurate, divided by our time, in the same job on the same GPU; below 1× is slower. The "finer reference" column requires the full simulation to match our accuracy against the finer reference.</p>
<div class="tw"><table>
<tr><th>Method</th><th>Points</th><th>Error</th><th>64³</th><th>128³</th><th>256³</th><th>256³, finer ref.</th><th>ms at 256³</th></tr>
{T3D}
</table></div>
<p><b>2D, held-out cases (64), accurate setting.</b> Worst error at 256² / 1024² / 4096²; speed-up at 4096² against the cheapest equally accurate full simulation, worked out from the study's timing table.</p>
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
<p class="take">Every working rule gives the same accuracy. Speed depends on mesh size: slower on small meshes, about even in the middle, clearly faster on large ones.</p>

</div>
"""
open(os.path.join(HERE, "index.html"), "w").write(page)
print("wrote index.html")
