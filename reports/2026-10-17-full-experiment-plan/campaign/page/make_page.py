"""Generate the running results page (index.html) from lane EXPORT.json files. No hand-typed numbers."""
import base64, json, os, datetime
ROOT = '/home/tahmid/Dev/Tunable-NM-ROM-Claude'
W = ROOT + '/worktrees'
P1 = W + '/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling'
S3 = W + '/2026-10-14-jcp-speed3d/experiments/jcp-speed3d'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/index.html'
STATUS = json.load(open(os.path.dirname(os.path.abspath(__file__)) + '/status.json'))

p1 = json.load(open(P1 + '/runs/EXPORT.json'))
s3 = json.load(open(S3 + '/runs/EXPORT.json'))
QL = W + '/2026-10-18-jcp-qualify/experiments/jcp-qualify'
F2 = W + '/2026-10-18-jcp-fom2d/experiments/jcp-fom2d'
ql = json.load(open(QL + '/runs/EXPORT.json'))
f2 = json.load(open(F2 + '/runs/EXPORT.json'))
k2m = f2['c2']['k2']['members']
S2 = [v['S'] for v in k2m.values() if v.get('status') == 'ok']
q3 = ql['threeD']['tested']
q3pass = sum(v == 'pass' for v in q3.values())
HB = W + '/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines'
HD = W + '/2026-10-18-jcp-heads/experiments/jcp-heads'
hb = json.load(open(HB + '/runs/EXPORT.json'))['cells']
def domcount(cell):
    X = hb[cell]['X']; return sum(': baseline dominates X' in (': ' + v.get('cell_line','')) for v in X.values()), len(X)
d256, n256 = domcount('c2d256'); d1k, n1k = domcount('c2d1024')
X3 = hb['c3d64']['X']
q3x = [v for v in X3.values() if 'batch 2' in v.get('cell_line','')]
d3, n3 = sum('baseline dominates X' in v['cell_line'] for v in q3x), len(q3x)
dg = [v for v in X3.values() if 'batch 1' in v.get('cell_line','')]
d3g, n3g = sum('baseline dominates X' in v['cell_line'] for v in dg), len(dg)
hd = json.load(open(HD + '/runs/EXPORT.json'))

def img(path, alt):
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'

def pill(v):
    v = v.lower()
    cls = 'pass' if v.startswith(('pass', 'monotone', 'improves')) else 'fail' if v.startswith(('fail', 'not', 'does not', 'fires')) else 'inc'
    return f'<span class="pill {cls}">{v}</span>'

d = p1['detail']['h_cost_n']
Rs = [x['R'] for x in d.values()]
ub = max(x['ci'][1] for x in d.values())
hcn = p1['verdicts']['h_cost_n']; han = p1['verdicts']['h_acc_n']
npass_cn = sum(v == 'pass' for v in hcn.values()); npass_an = sum(v == 'pass' for v in han.values())
hcr = {**{'2D ' + k: v for k, v in p1['verdicts']['h_cost_r_2d'].items()}, **{'3D ' + k: v for k, v in p1['verdicts']['h_cost_r_3d'].items()}}
nfail_cr = sum(v == 'fail' for v in hcr.values())
har = p1['verdicts']['h_acc_r']
m1 = s3['m1']; h41 = m1['H41']
r41 = [x['max_over_min'] for x in h41.values()]
hc = m1['Hcomp_largest_mesh']['cells']
union = [c['union']['speedup'] for c in hc.values()]
h2 = s3['H2']
k2 = m1['K2_fires']
kn = p1['key_numbers']['configurations']

def pct(x): return f'{100*x:.2f} %'

rows_acc = ''.join(
    f'<tr><td>{k.split("|")[0]}² ({16 if k.startswith("4096") else 38} cases)</td><td>{k.split("|")[1]}</td><td>{k.split("|")[2]}</td><td>{pct(v["e0_median"])}</td><td>{pct(v["e0_worst"])}</td></tr>'
    for k, v in kn.items() if k.split('|')[1] in ('R128', 'R256', 'R384', 'R512') and k.split('|')[2] == 'CNR')

lanes = ''.join(f'<li><span class="pill {l["cls"]}">{l["state"]}</span> <b>{l["name"]}</b> — {l["what"]}</li>' for l in STATUS['lanes'])

html = f'''<title>Burgers Scaling Campaign</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f5f7f5;--ink:#1c2629;--mute:#5b6a6c;--line:#d6dedb;--card:#ffffff;--acc:#0f6b66;--pass:#2c7a4b;--passbg:#e2f1e7;--fail:#a8392c;--failbg:#f8e4e0;--inc:#8a5a0e;--incbg:#f6ecd8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#111719;--ink:#e3eae8;--mute:#9aaaa8;--line:#2a3537;--card:#182124;--acc:#5ec4bc;--pass:#7fd39d;--passbg:#1d3427;--fail:#f29a8c;--failbg:#3a221e;--inc:#e9bf6e;--incbg:#352b17}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#111719;--ink:#e3eae8;--mute:#9aaaa8;--line:#2a3537;--card:#182124;--acc:#5ec4bc;--pass:#7fd39d;--passbg:#1d3427;--fail:#f29a8c;--failbg:#3a221e;--inc:#e9bf6e;--incbg:#352b17}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;padding-inline:16px;padding-block:24px 48px}}
main{{max-width:760px;margin:0 auto;display:grid;gap:28px}}
h1,h2{{font-family:Newsreader,Georgia,serif;font-weight:600;text-wrap:balance;margin:0}}
h1{{font-size:30px;line-height:1.15}} h2{{font-size:22px}}
.eyebrow{{font:500 11px "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--acc)}}
p{{margin:0;max-width:65ch}} .mute{{color:var(--mute)}}
section{{display:grid;gap:12px;border-top:1px solid var(--line);padding-top:20px}}
.verdicts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.v{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;display:grid;gap:6px;align-content:start}}
.v b{{font-size:13px}} .v small{{color:var(--mute);font-size:12.5px}}
.pill{{display:inline-block;font:500 11.5px "IBM Plex Mono",monospace;padding:2px 7px;border-radius:99px;white-space:nowrap}}
.pass{{color:var(--pass);background:var(--passbg)}} .fail{{color:var(--fail);background:var(--failbg)}} .inc{{color:var(--inc);background:var(--incbg)}}
ul{{margin:0;padding-left:18px;display:grid;gap:6px}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px}}
figure img{{display:block;width:100%;height:auto}} figcaption{{font-size:12px;color:#55635f;padding-top:4px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
.tbl{{overflow-x:auto}} table{{border-collapse:collapse;font:13px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
td,th{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right}} th{{color:var(--mute);font-weight:500}} td:first-child,th:first-child{{text-align:left}}
</style>
<main>
<header style="display:grid;gap:8px">
<div class="eyebrow">Viscous Burgers 2D/3D · JCP campaign · updated {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M')} UTC</div>
<h1>Does the reduced model's cost depend on bank width, and not on the mesh?</h1>
<p class="mute">{STATUS['headline']}</p>
</header>

<section><h2>The four core questions</h2>
<div class="verdicts">
<div class="v"><b>Cost vs mesh</b>{pill('pass' if npass_cn==len(hcn) else 'mixed')}<small>2D: {npass_cn}/{len(hcn)} settings flat from 256² to 4096² (largest max/min {max(Rs):.3f}, bar 1.25). 3D: flat from 64³ to 384³ (max/min {min(r41):.3f}–{max(r41):.3f}).</small></div>
<div class="v"><b>Accuracy vs mesh</b>{pill('pass' if npass_an==len(han) else 'mixed')}<small>2D: {npass_an}/{len(han)} settings give the same error on every mesh, within margin.</small></div>
<div class="v"><b>Accuracy vs bank width</b>{pill(har['1024|CNR']['range_128_384'])}<small>Error drops from r=128 to 256 to 384 on every mesh. Going to 512 {har['1024|CNR']['384->512']}.</small></div>
<div class="v"><b>Cost vs bank width</b>{pill('fail' if nfail_cr else 'pass')}<small>Cost does grow with r, but {nfail_cr} of {len(hcr)} kernel timings grow more slowly than their operation counts predict.</small></div>
</div></section>

<section><h2>Against standard reduced models (POD)</h2>
<p>H-base fails. A standard POD reduced model with ECSW or Q-DEIM hyper-reduction, tuned on the same validation cases, is both faster and better in the worst case than our model for {d1k} of {n1k} settings at 1024², {d256} of {n256} at 256², and, in 3D, {d3} of the {n3} fully qualified settings (plus {d3g} of {n3g} diagnostic ones). For the other {n3-d3} qualified 3D settings neither model dominated, but that result is held back because the 3D POD basis failed its adequacy check.</p>
<p class="mute">These are development-cohort, in-sample comparisons, and in some pairs our median error is lower. The 2D tuning budgets are close enough that it is unresolved whether they count as equal; in 3D the POD model got less tuning than ours.</p>
<div class="figs">{img(HB+'/plots/front-c2d1024.png','2D 1024²: accuracy against cost, POD baselines and ours')}{img(HB+'/plots/front-c3d64.png','3D 65³: accuracy against cost, POD baselines and ours')}</div>
</section>

<section><h2>The other "k": head size K</h2>
<p>With the nonlinear head (latent size K), cost is still flat in the mesh. At the same online cost, though, the plain linear span with r = 128 is more accurate than the K = 16 head. Too few of the K = 4, 8 and 32 heads passed the solver checks to say how error changes with K.</p>
<div class="figs">{img(HD+'/plots/error_cost_1024.png','Head (K) against linear span (r): error and cost at 1024²')}</div>
</section>

<section><h2>Against the full simulation (2D)</h2>
<p>K2 fires in 2D as well. For every reduced-model setting, the cheapest full solver that is at least as accurate is a spectral solver on a coarse grid (N = 24 or 48). At 4096² the reduced models run at {min(S2):.2f}–{max(S2):.2f}× its speed ({len(S2)} qualified settings). Against a conventional upwind solver, or a central solver on the same 4096² grid, the reduced model is still several times faster, but those are weaker opponents.</p>
<p class="mute">This retracts the earlier 13.6× figure as a claim against the best full solver; it only held against a same-mesh upwind solver.</p>
<div class="figs">{img(F2+'/plots/speedup_vs_mesh.png','2D: reduced-model speed-up over each full-solver family (below 1 = full solver faster)')}{img(QL+'/plots/front2d_median.png','2D accuracy–cost front of the reduced model')}</div>
</section>

<section><h2>Against the full simulation (3D)</h2>
<p>{'The kill criterion K2 fires: ' if k2 else ''}at 384³ a spectral full-order solver reaches the reduced model's accuracy faster. The reduced models run at {min(union):.3f}–{max(union):.2f}× its speed. Against our own earlier 3D setups the new configurations are {h2['C_old']['S_confirmed']:.1f}×, {h2['C_W768']['S_confirmed']:.1f}× and {h2['C_W1024']['S_confirmed']:.1f}× faster.</p>
<p class="mute">Correction: the configurations behind those 2.0–17× speed-ups fail the quadrature check (only {q3pass} of {len(q3)} tested 3D settings pass), so they are not yet speed-ups of fully qualified settings. Priority 3 re-times the qualified ones. Per the plan, the paper reports speed plainly and leads with mesh-independence.</p>
<div class="figs">{img(S3+'/plots/mesh-time.png','3D: time per query against mesh size (flat lines = mesh-independent)')}</div>
</section>

<section><h2>Priority 1 — 2D scaling</h2>
<div class="figs">{img(P1+'/plots/solve_time_vs_mesh.png','2D solve time against mesh, for each bank width')}{img(P1+'/plots/error_vs_r_and_mesh_CNR.png','2D error against bank width, one line per mesh (CN-R scheme)')}{img(P1+'/plots/kernel_cost_vs_r_2D.png','Kernel cost against bank width')}{img(P1+'/plots/p1g_frozen_vs_nominal.png','Recheck: fixing the test functions across meshes barely moves errors')}</div>
<div class="tbl"><table><tr><th>mesh</th><th>r</th><th>scheme</th><th>median error</th><th>worst error</th></tr>{rows_acc}</table></div>
<p class="mute">Errors against the second-order reference (a conditional estimate). Earlier "flat accuracy" claims at 256² compared slightly different test functions; fixing them moves errors by hundredths of a percentage point, so the conclusion stands.</p>
</section>

<section><h2>Lanes</h2><ul>{lanes}</ul></section>
<section><h2>Words used here</h2><ul>
<li><b>Bank width r</b>: how many basis coefficients the reduced model solves for (the "k" in the question).</li>
<li><b>Mesh</b>: grid points per side of the full simulation (256² up to 4096² in 2D).</li>
<li><b>Full-order / spectral solver</b>: the conventional simulation we compare against; "spectral" uses fast Fourier-type transforms.</li>
<li><b>K2</b>: the plan's pre-registered "we do not beat the best full solver" outcome.</li>
<li><b>BE, CN-R</b>: first- and second-order time stepping.</li>
</ul></section>
</main>'''
open(OUT, 'w').write(html)
print(OUT, len(html))
