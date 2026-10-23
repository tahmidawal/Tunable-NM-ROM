"""Builds the 'Research Update' deck (https://claude.ai/artifact/JqsVj9a21N8xFKudoxGHFk) into project/.
Slides 2-4 are copied verbatim from the live method deck (7D1rwb1AthPF4VdGdhtuiU slides ordering, mesh-cost, mesh-iters),
with only their image asset ids and page numbers changed. Numbers on the performance slides are copied from the Codex-audited
reports named in each footer: final report (reports/2026-10-17-full-experiment-plan/final/report.md), jcp-fom2d, jcp-hr-baselines,
jcp-dims, imp-box (E1) report.md."""
import json, html, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LIVE = sys.argv[1] if len(sys.argv) > 1 else 'live_slides'   # folder with the method deck's live slide files
BG="#F7F6F2"; CARD="#FBFBF8"; INK="#14213D"; SOFT="#4A5568"; ACC="#C8641E"; TEAL="#1F6F8B"; PILL="#E3ECF1"; LIGHT="#F7F6F2"
GOOD="#2E6B4F"; BAD="#A8401B"; MIX="#8A6A12"
SERIF="'Source Serif 4', Georgia, serif"; SANS="'IBM Plex Sans', Arial, sans-serif"
B="/_blob/"
F=dict(err=B+'db24dc0167cc8ed61a4464fc66064f6e', costmesh=B+'d1e33df512703feec8514091b843c08b', meshit=B+'8ef6d364d6fdd02c3c06390d6b519bd0',
 fompareto=B+'cc2c9df27cc9f988fe15071c375ae566', s2d=B+'1045285ee59ea88117766f439f8b7ca0', pod2d=B+'1384991a552e7eefa6f06053e505220c',
 front3d=B+'50f6ccebb30a8475519d377f3893f1fc', pod3d=B+'705c17d8e9b74c03d3de6505ce435344', union=B+'49ca0e980cd1f160fcd01bff9b58b226',
 e1=B+'ed71b71f74f58d02203d5c5741e534e2')
REMAP={'f6ae8e4aac27fbe262a7e8e37156120a':F['err'][7:], 'ed0543359adee3808726ee0017acda63':F['costmesh'][7:], '7d54add23cbf07c8401d55f49010bd56':F['meshit'][7:]}
slides=[]
def num(): return len(slides)+1
def foot(src): return f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:24px;color:{SOFT}">{src} · {num()}</p>'
SEC=f'background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:32px'
def head(q,a,verdict=None,vcol=None):
    v=f'<p style="font-size:28px;font-weight:600;color:{LIGHT};background:{vcol};padding:8px 20px;border-radius:8px;white-space:nowrap">{verdict}</p>' if verdict else ''
    return f'''<div style="display:flex;gap:24px;align-items:center">
<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{TEAL};background:{PILL};padding:10px 20px;border-radius:999px;white-space:nowrap">Question</p>
<p style="font-size:32px;color:{SOFT}">{q}</p>
</div>
<div style="display:flex;gap:28px;align-items:center">{v}
<h2 style="font-family:{SERIF};font-size:56px;font-weight:600;line-height:1.1;color:{INK}">{a}</h2>
</div>'''
def fig(id,q,a,img,alt,bullets,src,notes,verdict=None,vcol=None,w=1080):
    li="".join(f'<li>{b}</li>' for b in bullets)
    slides.append((id,f'''<section id="{id}" data-transition="fade" style="{SEC}">
{head(q,a,verdict,vcol)}
<div style="display:flex;gap:44px;align-items:start">
<img src="{img}" alt="{alt}" style="width:{w}px;height:500px;object-fit:contain;background:#FFFFFF;border:1px solid #DDE1E4;border-radius:12px">
<ul style="flex:1;font-size:26px;line-height:1.38;color:{SOFT};display:flex;flex-direction:column;gap:14px">{li}</ul>
</div>
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>'''))
def copied(id):
    s=open(os.path.join(LIVE,f'{id}.html')).read()
    for o,n in REMAP.items(): s=s.replace(o,n)
    s=re.sub(r'· \d+</p>',f'· {num()}</p>',s,count=1)
    slides.append((id,s))
def divider(id,part,title,sub):
    slides.append((id,f'''<section id="{id}" data-transition="fade" style="background:{INK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:36px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">{part}</p>
<h2 style="font-family:{SERIF};font-size:96px;font-weight:600;line-height:1.05;color:{LIGHT}">{title}</h2>
<p style="font-size:36px;line-height:1.4;color:#BFD0DC">{sub}</p>
</section>'''))

slides.append(("cover",f'''<section id="cover" data-transition="fade" style="background:{INK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">JCP paper campaign</p>
<h1 style="font-family:{SERIF};font-size:120px;font-weight:600;line-height:1.05;color:{LIGHT}">Research Update</h1>
<p style="font-size:36px;color:#BFD0DC">Tunable coordinate-network bank with off-mesh quadrature · viscous Burgers 2D and 3D</p>
<p style="font-size:28px;color:#BFD0DC">23 October 2026</p>
<aside>Three parts: the r dial, mesh independence of cost and iterations, and accuracy against speed in 2D and 3D. Accuracy numbers are against the order-verified references on development and validation cases; the sealed test cases are still closed.</aside>
</section>'''))
divider("d1","Part 1","The dial and the mesh","One trained bank; choose how many shapes to use; the solve does not depend on the mesh.")
copied("ordering"); copied("mesh-cost"); copied("mesh-iters")
divider("d2","Part 2","Accuracy against speed","How our model compares with full solvers and with tuned POD reduced models, in 2D and 3D.")
fig("p2d-fom","In 2D, how cheaply do full solvers reach our accuracy?","A coarse spectral solver reaches our accuracy band at far lower cost",F['fompareto'],"Error against cost for three families of full solvers, with the band of our models' accuracy shaded",
 ["Grey band: the accuracy our LSPG models reach at 1024² (median and worst)","Green: sine pseudo-spectral with ETDRK4 reaches that band at the lowest cost","Orange (central + CN-R) and blue (upwind + BE) need much more time","Example at 4096²: our r = 384 CN-R query 97 ms against 11.7 ms for the matched spectral solver"],
 "jcp-fom2d; why-spectral report","Each curve is the cheapest tested member of a full-solver family at each accuracy level, timed on an H200. Smooth Burgers in a box needs only a handful of sine modes, which is why the spectral family wins.")
fig("p2d-s","In 2D, what is our speed-up at matched accuracy?","Below 1× for every configuration, closing as the mesh grows",F['s2d'],"Speed-up over the matched full solver against mesh for every qualified LSPG configuration",
 ["Speed-up S = full-solver cost ÷ our cost at the same accuracy; above 1 means we are faster","At 4096²: S = 0.049–0.71 across 21 configurations; none reaches the 2× bar","S rises with the mesh because our solve cost is flat while the full solver's grows","Smaller r (blue, r = 128) is closest to break-even"],
 "jcp-fom2d; final report §3.4","Every S is an upper bound on our speed-up, because the full-solver ladder is finite. Kill criterion K2 (no 2× win at the largest mesh) fires.",verdict="Slower",vcol=BAD)
fig("p2d-pod","In 2D, how do we compare with a tuned POD reduced model?","POD with ECSW or Q-DEIM matches our accuracy in less time",F['pod2d'],"Worst error against query time at 1024²: our qualified configurations as red stars, POD variants as dots",
 ["Red stars: our qualified configurations; dots: POD + ECSW (blue), Q-DEIM (orange), none (grey)","Dominated (faster and more accurate) in 8 of 8 configurations at 256² and at 1024²","Example: ours r = 384 CN-R 86.8 ms, worst 1.78 %; POD + ECSW 30.7 ms, worst 1.35 %"],
 "jcp-hr-baselines; final report §3.5","Same timing job for both, so the comparison is like for like. Developmental, in-sample, unadjusted evidence. POD snapshots came from the best full solver on the target mesh.",verdict="Slower",vcol=BAD)
fig("p3d-front","In 3D, does the accuracy–cost dial work?","Yes: wider banks buy accuracy, and the qualified model is 2.95× faster than our earlier one",F['front3d'],"3D median error against cost (ratio to our earlier model C_old) for all qualified configurations, with the median front",
 ["Each dot is a numerically qualified 3D configuration (bank width, quadrature, solver)","The black front: the cheapest configuration at each accuracy level","Qualified speed-up 2.95× over our earlier wide model C_W768 at comparable accuracy","Earlier unqualified claims (5.4×, 17×) are withdrawn"],
 "jcp-dims H2q; final report (H2q row)","Cost is relative to C_old, so this is ROM against ROM, not against full solvers. Mesh-path query at 65³ on an H200.",verdict="Works",vcol=GOOD)
fig("p3d-cmp","In 3D, how do we compare with full solvers and POD?","Both are faster: the full solver by 6–130×, POD in most cells",F['pod3d'],"3D worst error against query time at 64³: our configurations as stars, POD variants as dots",
 ["Against the best matched full solver at 384³: S = 0.0079–0.16 (4 models)","Against POD at 65³: dominated in 5 of 5 (diagnostic batch) and 2 of 4 (qualified batch)","POD here failed its own basis-adequacy check, so its results carry that restriction"],
 "jcp-speed3d; jcp-hr-baselines; final report §3.4–3.5","6–130× is the inverse of S = 0.158 to 0.0079. Developmental, in-sample; ECSW coverage incomplete in 3D.",verdict="Slower",vcol=BAD)
fig("p-summary","All together: where do we stand against the best full solver?","Every configuration sits left of 1×; 3D is furthest behind",F['union'],"Speed-up over the best matched full solver for every 2D configuration at 1024² and 4096² and the 3D models",
 ["Blue: 2D at 1024²; orange: 2D at 4096²; green: 3D at 384³","Dashed line: break-even (1×); solid: the 2× bar we registered","2D gets closer at the bigger mesh; 3D stays at 0.008–0.16"],
 "final report §3.4 (union of full-solver families)","The union means the fastest full solver of any family that reaches the same accuracy. These verdicts are on development and validation cases.")
fig("p-e1","Newest: does a faster time stepper change the picture?","With ETDRK4 the bank beats the tuned spectral solver by up to 2×, but so does POD",F['e1'],"Speed-up against family width for each reduced model at three error targets and three steepness levels",
 ["Purple (our bank + ETDRK4): S = 0.81–2.07, above 1 in most cells","Red (our bank, standard LSPG): S = 0.044–0.40","Blue/orange (POD-Galerkin, POD + ECSW): up to 2.14 and 2.01, about the same as ours","Best on narrow families (W = 0.08); falls as the family widens"],
 "imp-box (E1) final report, audited; box, H200, val32","Provisional: errors are against the lane's own spectral references, many points sit at the cheapest model the search tried, and several cells were deferred for budget. ETDRK4 failed the worst-error rule in the 1024² fast-step lane, so it must be qualified here before we claim it.",verdict="New",vcol=MIX)
rows=[["Mesh independence of cost and iterations","2D and 3D","Yes: flat within 1.01–1.09"],["Accuracy dial r","2D 128→512; 3D","Improves to r ≈ 256–384, then flat"],
 ["Against the best full solver (deployed LSPG)","2D 4096²; 3D 384³","Slower: S 0.049–0.71; 0.0079–0.16"],["Against tuned POD + ECSW/Q-DEIM","2D 256², 1024²; 3D 65³","Slower: dominated 16/16 (2D), 7/9 (3D)"],
 ["Against our earlier 3D model","3D 65³","Faster: 2.95× (qualified)"],["Bank + ETDRK4 on the box (new)","2D box, narrow to wide","1–2× over spectral; level with POD (provisional)"]]
th="".join(f'<th style="width:{w}%">{h}</th>' for h,w in zip(["Comparison","Where","Result"],[40,24,36]))
tr="".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)
slides.append(("summary",f'''<section id="summary" data-transition="fade" style="{SEC}">
{head("What does this add up to?","Mesh-independent and tunable; not yet faster than the best alternatives")}
<table style="font-size:27px;color:{INK}"><tr>{th}</tr>{tr}</table>
{foot("final report; jcp-dims; imp-box")}
<aside>The open lever is the time integrator: the deployed LSPG step costs about two Jacobian builds and the decode grows with the mesh. With ETDRK4 the bank reaches parity with POD on the box, which is the strongest result so far, pending qualification.</aside>
</section>'''))
for id,s in slides: open(f'project/slides/{id}.html','w').write(s)
order=[i for i,_ in slides]
json.dump({"v":4,"createdOnFiles":{"v":1,"at":"2026-10-23T09:30:00Z"},"lists":"css","title":"Research Update","order":order,"cover":"cover",
 "sections":{"s0":{"description":"Cover","start":"cover"},"s1":{"description":"The dial and the mesh","start":"d1"},"s2":{"description":"Accuracy against speed, 2D and 3D","start":"d2"}},
 "faces":{"source-serif-4":{"family":"Source Serif 4","href":"https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
  "ibm-plex-sans":{"family":"IBM Plex Sans","href":"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"},
  "jetbrains-mono":{"family":"JetBrains Mono","href":"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"}},"designSystems":[]},open('project/deck.json','w'),indent=1)
print(order)
