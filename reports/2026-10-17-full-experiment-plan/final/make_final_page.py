"""Plain-language final results page (page/index.html) for the Burgers 2D/3D campaign.

Every number shown comes from FINAL-EXPORT.json (verdict key numbers, GPU ledger, lane list); figures are the
report's own figures. The plain-language takeaways contain no numbers.
"""
import base64, html, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
X = json.load(open(os.path.join(HERE, 'FINAL-EXPORT.json')))
V = X['verdicts']
OUT = os.path.join(HERE, 'page', 'index.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def esc(s):
    return html.escape(str(s))


def img(name, cap):
    b = base64.b64encode(open(os.path.join(HERE, 'figs', name), 'rb').read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>'


def cls(v):
    v = v.lower()
    if v.startswith(('pass', 'monotone', 'holds', 'robust', 'agree')):
        return 'pass'
    if v.startswith(('fail', 'fires', 'not ', 'dominated', 'degrades')):
        return 'fail'
    return 'inc'


def rows(hyp, **kw):
    out = [r for r in V if r['hypothesis'] == hyp]
    for k, val in kw.items():
        out = [r for r in out if val in r.get(k, '')]
    return out


def card(title, takeaway, rs):
    lines = ''.join(f'<li><span class="pill {cls(r["verdict"])}">{esc(r["verdict"].split(";")[0][:40])}</span> '
                    f'<span class="scope">{esc(r["scope"].split(",")[0])}</span><br><small>{esc(r["key_number"])}</small></li>'
                    for r in rs)
    return f'<div class="card"><h3>{esc(title)}</h3><p>{esc(takeaway)}</p><ul>{lines}</ul></div>'


hyps = sorted({r['hypothesis'] for r in V})
table = ''.join(f'<tr><td>{esc(r["hypothesis"])}</td><td>{esc(r["scope"])}</td><td><span class="pill {cls(r["verdict"])}">'
                f'{esc(r["verdict"])}</span></td><td>{esc(r["key_number"])}</td></tr>' for r in V)
gh = X['gpu_hours']
ledger = ''.join(f'<tr><td>{esc(k)}</td><td>{v["spent"]:.1f}</td><td>{v["cap"]:g}</td></tr>' for k, v in gh['per_lane'].items())
lanes = ''.join(f'<tr><td>{esc(l["lane"])}</td><td><code>{esc(l["branch"])}</code></td><td><code>{esc(l["head"])}</code></td>'
                f'<td><code>{esc(l["mirror"])}</code></td></tr>' for l in X['lanes'])

core = [
    card('Cost does not depend on the mesh', 'The time to solve the reduced model stays flat as the simulation grid gets finer, in 2D and 3D.',
         rows('H-cost-N')),
    card('Accuracy does not depend on the mesh', 'For a fixed bank width, the error is the same on coarse and fine grids.',
         rows('H-acc-N')),
    card('Accuracy improves with bank width, up to a point', 'A wider bank helps up to r = 384; going to 512 does not help.',
         rows('H-acc-r')),
    card('Cost against bank width', 'Cost grows with r, but not the way operation counts predict, so this registered prediction fails.',
         rows('H-cost-r')),
]
rest_h = [h for h in hyps if h not in ('H-cost-N', 'H-acc-N', 'H-acc-r', 'H-cost-r')]

page = f'''<title>Burgers Campaign Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f5f7f5;--ink:#1c2629;--mute:#5b6a6c;--line:#d6dedb;--card:#ffffff;--acc:#0f6b66;--pass:#2c7a4b;--passbg:#e2f1e7;--fail:#a8392c;--failbg:#f8e4e0;--inc:#8a5a0e;--incbg:#f6ecd8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#111719;--ink:#e3eae8;--mute:#9aaaa8;--line:#2a3537;--card:#182124;--acc:#5ec4bc;--pass:#7fd39d;--passbg:#1d3427;--fail:#f29a8c;--failbg:#3a221e;--inc:#e9bf6e;--incbg:#352b17}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#111719;--ink:#e3eae8;--mute:#9aaaa8;--line:#2a3537;--card:#182124;--acc:#5ec4bc;--pass:#7fd39d;--passbg:#1d3427;--fail:#f29a8c;--failbg:#3a221e;--inc:#e9bf6e;--incbg:#352b17}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;padding-inline:16px;padding-block:24px 48px}}
main{{max-width:780px;margin:0 auto;display:grid;gap:28px}}
h1,h2,h3{{font-family:Newsreader,Georgia,serif;font-weight:600;text-wrap:balance;margin:0}}
h1{{font-size:30px;line-height:1.15}} h2{{font-size:22px}} h3{{font-size:17px}}
.eyebrow{{font:500 11px "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--acc)}}
p{{margin:0;max-width:65ch}} .mute{{color:var(--mute)}}
section{{display:grid;gap:12px;border-top:1px solid var(--line);padding-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;display:grid;gap:8px;align-content:start}}
.card ul{{margin:0;padding-left:0;list-style:none;display:grid;gap:8px}} .card small{{color:var(--mute);font-size:12.5px}}
.scope{{font-size:13px}}
.pill{{display:inline-block;font:500 11.5px "IBM Plex Mono",monospace;padding:2px 7px;border-radius:99px}}
.pass{{color:var(--pass);background:var(--passbg)}} .fail{{color:var(--fail);background:var(--failbg)}} .inc{{color:var(--inc);background:var(--incbg)}}
ul.plain{{margin:0;padding-left:18px;display:grid;gap:6px}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px}}
figure img{{display:block;width:100%;height:auto}} figcaption{{font-size:12px;color:#55635f;padding-top:4px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
.tbl{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
td,th{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{color:var(--mute);font-weight:500}}
code{{font:12px "IBM Plex Mono",monospace}}
details summary{{cursor:pointer;font-weight:500}}
</style>
<main>
<header style="display:grid;gap:8px">
<div class="eyebrow">Viscous Burgers 2D/3D · JCP campaign · final · compiled {esc(X["compiled_utc"])}</div>
<h1>Does the reduced model's cost depend on bank width, and not on the mesh?</h1>
<p><b>Yes for the mesh, partly for the bank width.</b> Cost and accuracy are the same on every mesh we tried, in 2D and 3D, and that holds across retraining seeds and GPU types. Accuracy improves with bank width up to r = 384. But the method is not faster than the best conventional solver, a standard POD reduced model beats it, and it fails on steeper solutions than it was trained on. What the paper can stand on is mesh-independent cost, evaluation without a mesh, and memory.</p>
<p class="mute">{esc(X["label"])}. Total compute: {gh["total_all"]:.0f} GPU-hours over {len(X["lanes"])} lanes.</p>
</header>

<section><h2>The question the advisor asked</h2>
<div class="cards">{"".join(core)}</div>
<div class="figs">{img("summary_cost_vs_mesh.png", "Solve time against mesh size: flat lines mean the cost does not depend on the mesh")}{img("p1_error_vs_r_and_mesh_CNR.png", "Error against bank width, one line per mesh: the lines lie on top of each other")}</div>
</section>

<section><h2>Against other methods</h2>
<div class="cards">
{card("Against the best full solver (K2)", "A spectral solver on a coarse grid reaches the same accuracy faster, in 2D and 3D. The plan's kill criterion fires.", rows("K2") + [r for r in rows("H-comp") if "union" in (r["scope"] + r["key_number"]).lower()][:4])}
{card("Against a standard reduced model (H-base)", "A tuned POD model with hyper-reduction is faster and has lower worst-case error for our 2D settings.", rows("H-base"))}
</div>
<div class="figs">{img("summary_union_speedup.png", "Speed-up over the best full solver at each mesh (below 1 means the full solver is faster)")}{img("hrb_front-c2d1024.png", "2D 1024²: accuracy against cost for POD baselines and our model")}</div>
</section>

<section><h2>How robust it is</h2>
<div class="cards">
{"".join(card({"D1": "New regimes (outside the training range)", "E1": "Random quadrature points", "E2": "Retraining with new seeds", "E5": "Different GPUs"}[h.split()[0]], {"D1": "On steeper fronts than in training, the model is not qualified; the bank itself cannot represent them.", "E1": "Random choices of quadrature points change little; the 2D 'fragile' cells are about rule convergence, not accuracy.", "E2": "Retraining with new seeds keeps mesh-independence and the bank-width trend, but the best quadrature rule changes per bank.", "E5": "A100 and H200 GPUs give the same verdicts."}[h.split()[0]], rows(h)) for h in rest_h if h.startswith(("D", "E")))}
</div>
<div class="figs">{img("regimes_error_vs_s.png", "Error grows with front steepness beyond the training range")}{img("robust_e2_spread.png", "Spread of results across retraining seeds")}</div>
</section>

<section><h2>Where the error comes from</h2>
<p>The time-stepping error dominates for the standard schemes. Results stay stable when runs are extended in time, and the scoring grid is fine enough.</p>
<div class="figs">{img("budget_b2_main_1024.png", "Error budget at 1024²: which ingredient dominates")}{img("dims_h2q_3d.png", "3D: speed-up of the cheapest fully checked setting over our earlier setups (reduced model against reduced model)")}</div>
</section>

<section><h2>Every verdict</h2>
<details><summary>Show all {len(V)} rows</summary><div class="tbl"><table><tr><th>Hypothesis</th><th>Scope</th><th>Verdict</th><th>Key number</th></tr>{table}</table></div></details>
</section>

<section><h2>Corrections made during the campaign</h2>
<ul class="plain">
<li>The first 3D speed-ups over our own earlier setups used settings that failed the quadrature check. Re-timed with fully checked settings, a real gain remains against the W768 setup, but against the original deployment it is below the bar.</li>
<li>An earlier large 2D speed-up held only against a same-mesh upwind solver, not against the best full solver.</li>
<li>Some earlier "flat accuracy across meshes" comparisons used slightly different test functions on the 256² mesh. Fixing them moved errors by hundredths of a percentage point.</li>
<li>Several lanes' first analyses had errors (wrong dominance rule, under-counted budgets, double-counted GPU-hours). Each was caught by an audit, corrected and re-audited. The full list is in the report.</li>
</ul></section>

<section><h2>Compute and lanes</h2>
<div class="figs">{img("summary_gpu_ledger.png", "GPU-hours per lane")}</div>
<details><summary>GPU-hours per lane</summary><div class="tbl"><table><tr><th>Lane</th><th>Spent</th><th>Cap</th></tr>{ledger}</table></div></details>
<details><summary>Branches awaiting your merge decision</summary><div class="tbl"><table><tr><th>Lane</th><th>Branch</th><th>Last commit</th><th>Mirror</th></tr>{lanes}</table></div></details>
</section>

<section><h2>Words used here</h2><ul class="plain">
<li><b>Bank width r</b>: how many basis coefficients the reduced model solves for. This is the "k" in the advisor's question; the other reading, the nonlinear head's size K, is also tested.</li>
<li><b>Mesh</b>: grid points per side of the full simulation.</li>
<li><b>Full solver / spectral solver</b>: the conventional simulation; "spectral" uses fast Fourier-type transforms.</li>
<li><b>K2</b>: the plan's pre-registered outcome "we are not at least 2× faster than the best full solver".</li>
<li><b>POD baseline</b>: the standard reduced model built from simulation snapshots, with hyper-reduction (ECSW or Q-DEIM) to make it fast.</li>
<li><b>Qualified</b>: passed the solver and quadrature checks, so its numbers can be trusted.</li>
<li><b>Seed</b>: the random starting point of training; a seed-robust result holds for every retrained model.</li>
<li><b>Development / validation cases</b>: the test problems used for choosing and checking; the sealed final test set is still unopened.</li>
</ul></section>
</main>'''
open(OUT, 'w').write(page)
print(OUT, len(page))
