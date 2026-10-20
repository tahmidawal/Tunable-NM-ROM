import json, html
BG="#F7F6F2"; INK="#14213D"; SOFT="#4A5568"; ACC="#C8641E"; TEAL="#1F6F8B"; PILL="#E3ECF1"; DARK="#14213D"; LIGHT="#F7F6F2"
SERIF="'Source Serif 4', Georgia, serif"; SANS="'IBM Plex Sans', Arial, sans-serif"
B="/_blob/"
F={'cost':B+'e5b7ed0df1db35bb0ffcfd5ef2251e3e','err':B+'9ceb688270309fa49e99966f98da2b47','kern':B+'d9cf66d1ede64873021e48b078c51440',
   'speed':B+'34f8a7a01c511917e05594ee54abfdc9','pod':B+'54276cfa86aa832d26185aaf348a58b6','heads':B+'dd4281d701f9704c3f2c46c2a9e69cb1',
   'reg':B+'5b3710e5c6594dbf172ce1a678c06cfb','rob':B+'b5a05c85e714355c2e5046eba792e690','front3':B+'bc8c132d5afaaffd776590544d5a44ea'}
slides=[]
def foot(n,src):
    return f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:24px;color:{SOFT}">{src} · {n}</p>'
def q(n,id,question,answer,verdict,vcol,img,alt,bullets,src,notes):
    li="".join(f'<li>{b}</li>' for b in bullets)
    s=f'''<section id="{id}" data-transition="fade" style="background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:32px">
<div style="display:flex;gap:24px;align-items:center">
<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{TEAL};background:{PILL};padding:10px 20px;border-radius:999px">Question</p>
<p style="font-size:32px;color:{SOFT}">{question}</p>
</div>
<div style="display:flex;gap:28px;align-items:center">
<p style="font-size:28px;font-weight:600;color:{LIGHT};background:{vcol};padding:8px 20px;border-radius:8px;white-space:nowrap">{verdict}</p>
<h2 style="font-family:{SERIF};font-size:64px;font-weight:600;line-height:1.1;color:{INK}">{answer}</h2>
</div>
<div style="display:flex;gap:48px;align-items:start">
<img src="{img}" alt="{alt}" style="width:1060px;height:500px;object-fit:contain;background:#FFFFFF;border:1px solid #DDE1E4;border-radius:12px">
<ul style="flex:1;font-size:30px;line-height:1.4;color:{SOFT};display:flex;flex-direction:column;gap:18px">{li}</ul>
</div>
{foot(n,src)}
<aside>{html.escape(notes)}</aside>
</section>'''
    slides.append((id,s))
GOOD="#2E6B4F"; BAD="#A8401B"; MIX="#8A6A12"
# cover
slides.append(("cover",f'''<section id="cover" data-transition="fade" style="background:{DARK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">JCP paper campaign · final results</p>
<h1 style="font-family:{SERIF};font-size:104px;font-weight:600;line-height:1.05;color:{LIGHT}">Viscous Burgers in 2D and 3D: what the experiments answered</h1>
<p style="font-size:36px;color:#BFD0DC">Tunable bank + Hari’s off-mesh quadrature · 10 lanes · 169 GPU-hours · every step Codex-checked</p>
<p style="font-size:28px;color:#BFD0DC">20 October 2026</p>
<aside>Each slide starts with the question it answers, then the answer and the evidence. All accuracy numbers are against the order-verified second-order references, on development and validation cases; the sealed test cases are still closed.</aside>
</section>'''))
# overview
slides.append(("questions",f'''<section id="questions" data-transition="fade" style="background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:40px">
<h2 style="font-family:{SERIF};font-size:64px;font-weight:600;line-height:1.1">The questions, and the short answers</h2>
<table style="font-size:28px;color:{INK}">
<tr><th style="width:62%">Question</th><th style="width:38%">Answer</th></tr>
<tr><td>Does solve cost depend on the mesh?</td><td><b style="color:{GOOD}">No</b> (2D and 3D)</td></tr>
<tr><td>Does accuracy depend on the mesh?</td><td><b style="color:{GOOD}">No</b></td></tr>
<tr><td>Does accuracy improve with k?</td><td><b style="color:{GOOD}">Yes</b>, 128 → 384; not at 512</td></tr>
<tr><td>Does cost follow the textbook k-scaling?</td><td><b style="color:{BAD}">No</b> (grows, but not as predicted)</td></tr>
<tr><td>Does the nonlinear head beat the plain bank?</td><td><b style="color:{BAD}">No</b>, at equal cost</td></tr>
<tr><td>Faster than the best full simulation?</td><td><b style="color:{BAD}">No</b>, spectral wins</td></tr>
<tr><td>Better than a tuned POD reduced model?</td><td><b style="color:{BAD}">No</b>, POD + ECSW wins</td></tr>
<tr><td>Faster than our own earlier 3D model?</td><td><b style="color:{GOOD}">Yes</b>, 2.95×</td></tr>
<tr><td>Does it work on sharper fronts?</td><td><b style="color:{BAD}">No</b>, beyond training</td></tr>
<tr><td>Robust to seeds and hardware?</td><td><b style="color:{GOOD}">Yes</b></td></tr>
</table>
{foot(2,"Final campaign report")}
<aside>Green answers are the paper's strengths; orange ones are limits we report honestly.</aside>
</section>'''))
q(3,"cost-mesh","Does the solve get slower when the mesh gets bigger?","Solve cost does not depend on the mesh","No","#2E6B4F",F['cost'],"Solve time relative to the middle mesh, flat across meshes in 2D and 3D",
  ["<b>2D:</b> flat from 256² to 4096², 8 of 8 settings (ratio 1.01–1.02; bar 1.25)","<b>3D:</b> flat from 64³ to 384³, 4 of 4 models (1.01–1.09)","The full query’s output decode still grows with the mesh"],"Lanes p1-scaling, speed3d",
  "This is the advisor's question. Mesh grew 256 times in 2D; solve time moved by 1 to 2 percent.")
q(4,"acc-mesh","Does accuracy change with the mesh?","Accuracy does not depend on the mesh","No","#2E6B4F",F['err'],"Median error against bank width and against mesh, lines flat across meshes",
  ["8 of 8 settings: every case within the margin across 256², 1024², 4096²","Right panel: each k is a flat line across meshes","Freezing the test set across meshes moved errors only 0.01–0.05 points"],"Lane p1-scaling",
  "R128 with CN-R shows a tiny detectable trend, still within the equivalence margin.")
q(5,"acc-k","Does a bigger bank (k) give better accuracy?","Yes, from 128 to 384, then it levels off","Yes, up to 384","#2E6B4F",F['err'],"Median error falling from k 128 to 384",
  ["Improves at every step 128 → 256 → 384, at every mesh and scheme","Best setting: 0.20 % typical, 1.8 % worst error","384 → 512 does not improve"],"Lane p1-scaling",
  "Left panel. The bank stops being the limit around 384; time stepping takes over.")
q(6,"cost-k","How does cost grow with k?","Cost grows with k, but not at the textbook rates","Not as predicted","#A8401B",F['kern'],"Per-call time of each solver part against bank width",
  ["Predicted exponents 1, 1, 2, 3 for the four solver parts","Measured in 2D: 1.17, 1.08, 1.92, 1.24","At these sizes the GPU is far from its large-matrix regime"],"Lane p1-scaling",
  "Cost does grow with k, which is what we want to show, but the operation-count model doesn't predict the GPU timings. Fail as registered.")
q(7,"heads","Is the nonlinear head (the other “k”) worth it?","At equal cost the plain linear bank wins","No","#A8401B",F['heads'],"Error against cost for linear banks and nonlinear heads",
  ["Linear k = 128: 1.10 % at 0.59× the head’s cost","Nonlinear head K = 16: 1.43 %","Head cost is also mesh-independent (ratio 1.06)"],"Lane heads",
  "Covers the second reading of k. Error versus K is inconclusive because too few heads qualified.")
q(8,"vs-fom","Are we faster than the full simulation at equal accuracy?","A coarse spectral simulation beats us","No","#A8401B",F['speed'],"Speed-up over the best matched full simulation, mostly below 1",
  ["2D 4096²: 0.05–0.71× against the best full solver","3D 384³: 0.008–0.16×","We do beat the standard first-order solver (≥ 2× in 16 of 21)"],"Lanes fom2d, speed3d",
  "Kill criterion K2 fires in both dimensions. The plan's consequence: lead with mesh-independence, not speed.")
q(9,"vs-pod","Are we better than a standard POD reduced model?","A tuned POD + ECSW model dominates ours","No","#A8401B",F['pod'],"Error against time; POD baseline points sit below and left of ours",
  ["2D 256² and 1024²: 8 of 8 of ours dominated","3D: 7 of 9 dominated","We keep lower median error in a few cases"],"Lane hr-baselines",
  "Developmental evidence; POD snapshots from the best full solver, which favours the baseline.")
q(10,"speed3d","In 3D, are we faster than our own earlier model?","Yes: 2.95× faster, with quadrature qualified","Yes","#2E6B4F",F['front3'],"3D accuracy-cost front of qualified configurations",
  ["Qualified speed-up 2.95× over the wide W768 model","Earlier 17× was not quadrature-qualified, so it doesn’t count","Front at the widest banks is too sparse to claim more"],"Lane dims",
  "A ROM-versus-ROM claim; not a claim against full simulations.")
q(11,"regimes","Does it work on sharper fronts than it was trained on?","No: it breaks down beyond the training range","No","#A8401B",F['reg'],"Error rising with front steepness, past the training maximum",
  ["Bank’s own best error 3.5–9.9 % on steeper fronts","3–10× worse in the stiff, low-viscosity corner","Inside the training range it holds"],"Lane regimes",
  "An honest limit, and a reason to pick harder examples carefully.")
q(12,"robust","Are the results robust to retraining and hardware?","Yes: mesh-independence and the k dial hold","Yes","#2E6B4F",F['rob'],"Error spread across four retrained banks, tight per setting",
  ["Four retrained banks give nearly the same errors","Same conclusions on A100 and H200","Quadrature rules must be chosen per bank"],"Lane robust",
  "Seeds and hardware don't change the headline conclusions.")
# stats slide: error source and long runs
slides.append(("budget",f'''<section id="budget" data-transition="fade" style="background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:40px">
<div style="display:flex;gap:24px;align-items:center">
<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{TEAL};background:{PILL};padding:10px 20px;border-radius:999px">Question</p>
<p style="font-size:32px;color:{SOFT}">Where does the remaining error come from, and is it stable over time?</p>
</div>
<h2 style="font-family:{SERIF};font-size:64px;font-weight:600;line-height:1.1">Time stepping dominates; runs stay stable 4× longer</h2>
<div style="display:flex;gap:32px">
<div style="flex:1;display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><p style="font-size:72px;font-weight:600;color:{ACC}">Time</p><p style="font-size:30px;color:{SOFT}">is the largest error source at every mesh; quadrature and solver errors are tiny</p></div>
<div style="flex:1;display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><p style="font-size:72px;font-weight:600;color:{ACC}">T = 1</p><p style="font-size:30px;color:{SOFT}">4× the usual run length: stable, no error growth (provisional)</p></div>
<div style="flex:1;display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><p style="font-size:72px;font-weight:600;color:{ACC}">257²</p><p style="font-size:30px;color:{SOFT}">scoring grid is fine enough; it doesn’t miss detail</p></div>
</div>
{foot(13,"Lane budget")}
<aside>The longer-run verdict is provisional because one reference case failed its own order check.</aside>
</section>'''))
slides.append(("meaning",f'''<section id="meaning" data-transition="fade" style="background:{ACC};color:#FFF8F2;font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:44px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#FFE3CC">What it means for the paper</p>
<h2 style="font-family:{SERIF};font-size:80px;font-weight:600;line-height:1.1;color:#FFF8F2">Lead with mesh-independent cost and accuracy, the k dial, off-mesh evaluation and low memory</h2>
<p style="font-size:36px;line-height:1.4;color:#FFF0E3">Don’t claim speed or accuracy wins over the best full solver or a tuned POD model on Burgers in a box. A speed advantage has to come from harder problems: non-box shapes, sharper physics, Navier–Stokes.</p>
<aside>This follows the plan's written rule for when kill criterion K2 fires.</aside>
</section>'''))
slides.append(("next",f'''<section id="next" data-transition="fade" style="background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:40px">
<h2 style="font-family:{SERIF};font-size:64px;font-weight:600;line-height:1.1">What’s still open</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
<div style="display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">Final numbers</h3><p style="font-size:30px;color:{SOFT}">Freeze every choice, then open the sealed test cases once (128 in 2D, 64 in 3D)</p></div>
<div style="display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">Navier–Stokes 2D and 3D</h3><p style="font-size:30px;color:{SOFT}">Next campaign: where the method may have a real edge</p></div>
<div style="display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">Deferred</h3><p style="font-size:30px;color:{SOFT}">Full POD baseline set, other decoders, 3D new regimes, more seeds</p></div>
<div style="display:flex;flex-direction:column;gap:12px;background:#FFFFFF;padding:40px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">Decisions</h3><p style="font-size:30px;color:{SOFT}">Merging the finished branches; who holds the sealed-case key</p></div>
</div>
{foot(15,"Final campaign report")}
<aside>Nothing here is blocking; the sealed opening is the last step before the paper's numbers are final.</aside>
</section>'''))
for id,s in slides: open(f'project/slides/{id}.html','w').write(s)
order=[i for i,_ in slides]
deck={"v":4,"createdOnFiles":{"v":1,"at":"2026-10-20T17:30:00Z"},"lists":"css","title":"Burgers Campaign Results","order":order,"cover":"cover",
 "sections":{"s1":{"description":"What we set out to answer","start":"cover"},"s2":{"description":"Cost and accuracy against k and the mesh","start":"cost-mesh"},"s3":{"description":"Against other methods","start":"vs-fom"},"s4":{"description":"Limits, robustness and what it means","start":"regimes"}},
 "faces":{"source-serif-4":{"family":"Source Serif 4","href":"https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},"ibm-plex-sans":{"family":"IBM Plex Sans","href":"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"}},"designSystems":[]}
json.dump(deck,open('project/deck.json','w'),indent=1)
print(order)
