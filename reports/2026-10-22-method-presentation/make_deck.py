"""Generates the 'How it works' deck (project/deck.json + project/slides/*.html).
Every number is copied from a cited report (see each slide's footer); none is computed here."""
import json, html, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BG="#F7F6F2"; BG2="#ECEEEA"; CARD="#FBFBF8"; INK="#14213D"; SOFT="#4A5568"; ACC="#C8641E"; TEAL="#1F6F8B"; PILL="#E3ECF1"; LIGHT="#F7F6F2"
GOOD="#2E6B4F"; BAD="#A8401B"; MIX="#8A6A12"
SERIF="'Source Serif 4', Georgia, serif"; SANS="'IBM Plex Sans', Arial, sans-serif"; MONO="'JetBrains Mono', 'Courier New', monospace"
B="/_blob/"
F=dict(quad2d=B+'5a45d65254b84a7dda451a013793f24b', mem3d=B+'a0f89463a6706a3cfed93f38c5ed2650', nsteep=B+'8a843aa86c650ccc866dff5222e89795',
 stepcost=B+'2be447751e6f012d21d24bd8cbafd709', kern=B+'aa2fb685f61bf33bc854ba6de263598a', errr=B+'717e5cf2efc094e662f6b25abdb0e340',
 budget=B+'9995b5caf186ee6332f024a7377b6ca3', order=B+'7d770047426694545ace2b560b30c268', regimes=B+'a7c31457145b86ea55330524c7cbfccc',
 union=B+'34969ed4e26104fa830e31ddaee422cd', pod=B+'31e7f75087b82842a2f5a3c059f02494', heads=B+'0e232166420af711b63fa8f11df9140a',
 costmesh=B+'ed0543359adee3808726ee0017acda63', family=B+'c211e5272035e27b68096d069808345f', seeds=B+'4e4e4a17cebffda7ea6fb34d3e0cb0fe',
 longrun=B+'ca383406263021b285f711eb06a9bfde', h2q=B+'9ba3304b7b68dd9118bb0428a8adbb14')
slides=[]; N=[0]
def add(id,s): slides.append((id,s))
def foot(src):
    return f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:24px;color:{SOFT}">{src} · {len(slides)+1}</p>'
SEC=f'background:{BG};color:{INK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:36px'
def head(question,answer,verdict=None,vcol=None):
    v=f'<p style="font-size:28px;font-weight:600;color:{LIGHT};background:{vcol};padding:8px 20px;border-radius:8px;white-space:nowrap">{verdict}</p>' if verdict else ''
    return f'''<div style="display:flex;gap:24px;align-items:center">
<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{TEAL};background:{PILL};padding:10px 20px;border-radius:999px;white-space:nowrap">Question</p>
<p style="font-size:32px;color:{SOFT}">{question}</p>
</div>
<div style="display:flex;gap:28px;align-items:center">{v}
<h2 style="font-family:{SERIF};font-size:60px;font-weight:600;line-height:1.1;color:{INK}">{answer}</h2>
</div>'''
def ul(items,size=30): 
    li="".join(f'<li>{b}</li>' for b in items)
    return f'<ul style="flex:1;font-size:{size}px;line-height:1.4;color:{SOFT};display:flex;flex-direction:column;gap:16px">{li}</ul>'
def fig(id,question,answer,img,alt,bullets,src,notes,verdict=None,vcol=None,w=1000):
    add(id,f'''<section id="{id}" data-transition="fade" style="{SEC}">
{head(question,answer,verdict,vcol)}
<div style="display:flex;gap:48px;align-items:start">
<img src="{img}" alt="{alt}" style="width:{w}px;height:480px;object-fit:contain;background:#FFFFFF;border:1px solid #DDE1E4;border-radius:12px">
{ul(bullets)}
</div>
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>''')
def eqbox(lines,size=34):
    t="<br>".join(lines)
    return f'<p style="font-family:{MONO};font-size:{size}px;line-height:1.5;color:{INK};background:#FFFFFF;border:1px solid #DDE1E4;border-left:8px solid {ACC};border-radius:12px;padding:28px 36px">{t}</p>'
def step(id,n,title,question,eq,bullets,src,notes,eqsize=34):
    add(id,f'''<section id="{id}" data-transition="fade" style="{SEC}">
<div style="display:flex;gap:24px;align-items:center">
<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{LIGHT};background:{TEAL};padding:10px 20px;border-radius:999px;white-space:nowrap">Step {n} of 6</p>
<p style="font-size:32px;color:{SOFT}">{question}</p>
</div>
<h2 style="font-family:{SERIF};font-size:60px;font-weight:600;line-height:1.1">{title}</h2>
{eqbox(eq,eqsize)}
{ul(bullets,30)}
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>''')
def cards(id,question,answer,items,src,notes,cols=3,verdict=None,vcol=None,big=56):
    cs=""
    for t,b in items:
        cs+=f'<div style="display:flex;flex-direction:column;gap:12px;background:{CARD};padding:32px;border:1px solid #DDE1E4;border-radius:16px"><p style="font-size:{big}px;font-weight:600;line-height:1.1;color:{ACC}">{t}</p><p style="font-size:28px;line-height:1.35;color:{SOFT}">{b}</p></div>'
    add(id,f'''<section id="{id}" data-transition="fade" style="{SEC}">
{head(question,answer,verdict,vcol)}
<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:28px">{cs}</div>
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>''')
def table(id,question,answer,hdr,rows,widths,src,notes,size=28,after=""):
    th="".join(f'<th style="width:{w}%">{h}</th>' for h,w in zip(hdr,widths))
    tr="".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)
    add(id,f'''<section id="{id}" data-transition="fade" style="{SEC}">
{head(question,answer)}
<table style="font-size:{size}px;color:{INK}"><tr>{th}</tr>{tr}</table>
{after}
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>''')
def divider(id,part,title,sub):
    add(id,f'''<section id="{id}" data-transition="fade" style="background:{INK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:36px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">{part}</p>
<h2 style="font-family:{SERIF};font-size:96px;font-weight:600;line-height:1.05;color:{LIGHT}">{title}</h2>
<p style="font-size:36px;line-height:1.4;color:#BFD0DC">{sub}</p>
</section>''')

# ---------------- cover + map
add("cover",f'''<section id="cover" data-transition="fade" style="background:{INK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">JCP paper campaign · the method, end to end</p>
<h1 style="font-family:{SERIF};font-size:100px;font-weight:600;line-height:1.05;color:{LIGHT}">The tunable bank with off-mesh quadrature: how it works, what drives it, where it fails</h1>
<p style="font-size:36px;color:#BFD0DC">Architecture · training · the solver step by step · what accuracy and speed depend on · what failed, and why</p>
<p style="font-size:28px;color:#BFD0DC">22 October 2026 · viscous Burgers 2D and 3D</p>
<aside>One deck for someone who has never seen the method. Each slide starts with the question it answers. All accuracy numbers are against the order-verified second-order references on development and validation cases; the sealed test cases are still closed. Numbers are copied from the Codex-audited lane reports named in each footer.</aside>
</section>''')
cards("map","What does this deck cover?","Five parts, in the order you’d explain the method",
 [("0","How we got here: why off-mesh quadrature, and what it bought us"),("1","The architecture and how the bank is trained"),("2","The online solver, one step at a time"),
  ("3","What accuracy and speed actually depend on, with measured numbers"),("4","What works, what failed, and our best explanation for each failure"),("+","What’s still running and not yet measured")],
 "Deck map","Parts 0 to 3 explain the method; part 4 is the honest scorecard.",cols=3,big=72)

# ---------------- Part 0
divider("p0","Part 0","How we got here","The bank existed before this paper. Hari’s off-mesh quadrature is what made it mesh-independent.")
fig("quad","Why replace the mesh lattice with off-mesh quadrature?","Off-mesh rules make the error the same on every mesh",F['quad2d'],"Worst error against mesh for the dense solve, the mesh lattice and the off-mesh rules",
 ["Off-mesh rules (Gauss, Fibonacci): worst error changes by a factor 1.0001–1.0060 across 256², 1024², 4096²","The old mesh lattice: 1.31–2.19 across the same meshes","Off-mesh evaluates the nonlinear term on the continuum, so it inherits no grid error"],
 "Off-mesh quadrature study 2D (2026-10-01), dev+val, 38 cases",
 "The old method sampled the nonlinear term at mesh nodes, so it carried the mesh's upwind error and changed with the mesh. Off-mesh quadrature evaluates the network and its exact derivative at fixed points in the continuous domain, so the reduced model is literally the same object on every mesh.")
fig("memory","What does off-mesh quadrature cost in memory?","Twenty times less memory than the tensor method in 3D",F['mem3d'],"Memory of the off-mesh rule against the precomputed tensor in 3D",
 ["At r = 1024 (3D): <b>1.61 GB</b> for the off-mesh rule against <b>34.37 GB</b> for the precomputed tensor","At r = 512: 0.34 GB against 4.30 GB","Off-mesh query also faster than the tensor in all six 3D cells"],
 "Lanes jcp-wide-bank and quadrature-burgers3d",
 "The tensor method precomputes a third-order tensor whose size grows like r cubed. Off-mesh quadrature only stores the network values at m points, which grows like m times r.")

# ---------------- Part 1
divider("p1","Part 1","The architecture and how it is trained","A neural network that outputs a whole family of smooth shapes, ordered so that the first r of them are the best r.")
add("idea",f'''<section id="idea" data-transition="fade" style="{SEC}">
{head("What is the reduced model, in one line?","The solution is a short sum of learned shapes")}
{eqbox(["u(x, t)  ≈  c₁(t)·g₁(x) + c₂(t)·g₂(x) + … + c_r(t)·g_r(x)"],44)}
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px">
<div style="display:flex;flex-direction:column;gap:12px;background:{CARD};padding:32px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">The shapes g(x)</h3><p style="font-size:28px;line-height:1.35;color:{SOFT}">Come from one trained network. They are functions of position, so they can be evaluated anywhere, on any mesh.</p></div>
<div style="display:flex;flex-direction:column;gap:12px;background:{CARD};padding:32px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">The numbers c(t)</h3><p style="font-size:28px;line-height:1.35;color:{SOFT}">Are what the solver computes each time step. There are only r of them, against millions of mesh values.</p></div>
<div style="display:flex;flex-direction:column;gap:12px;background:{CARD};padding:32px;border:1px solid #DDE1E4;border-radius:16px"><h3 style="font-size:36px;font-weight:600;color:{TEAL}">The dial r</h3><p style="font-size:28px;line-height:1.35;color:{SOFT}">Picks how many shapes to use, from 32 up to 512 (2D) or 1024 (3D), with one trained network.</p></div>
</div>
{foot("Method summary")}
<aside>Everything else in the deck is about how the shapes are made and ordered, and how the numbers c are computed each step.</aside>
</section>''')
# architecture diagram
def box(t,b,col):
    return f'<div style="flex:1;display:flex;flex-direction:column;gap:8px;background:{CARD};padding:24px;border:2px solid {col};border-radius:14px"><p style="font-size:28px;font-weight:600;color:{col}">{t}</p><p style="font-size:24px;line-height:1.3;color:{SOFT}">{b}</p></div>'
arrow='<x-connector style="width:48px;color:#8A94A6"></x-connector>'
add("arch",f'''<section id="arch" data-transition="fade" style="{SEC}">
{head("What does the 2D network look like?","Position in, 512 shapes out")}
<div style="display:flex;align-items:center;gap:0px">
{box("x = (x, y)","a point anywhere in the unit square",INK)}{arrow}
{box("Fourier features","sin and cos of 2πBx, 128 random frequencies (σ = 4) → 256 numbers",TEAL)}{arrow}
{box("MLP","256 → 1024 → 1024 → 512, SiLU",TEAL)}{arrow}
{box("× envelope","16x(1−x)y(1−y) makes every shape zero on the boundary",ACC)}{arrow}
{box("× rotation T","orders the 512 shapes so any prefix r is useful",ACC)}
</div>
{eqbox(["g(x) = 0.2385 · 16x(1−x)y(1−y) · MLP([sin 2πBx, cos 2πBx]) · T"],34)}
{ul(["Fourier features let a small MLP represent sharp bumps; the envelope builds in the Dirichlet boundary condition exactly; 1.84 M parameters","A K = 16 nonlinear “head” (0.54 M parameters) also exists; it is a side experiment, not the deployed model"],28)}
{foot("separable-decoder sep_common.py; quadrature-study qcore.py")}
<aside>Teal boxes are the trained network, orange are fixed operations. Hidden width 1024 is twice the output width, which avoids a rank cap on the 512 shapes.</aside>
</section>''')
table("arch3d","What do the 3D banks look like?","Same design, wider, with two sizes",
 ["","2D R512","3D M2","3D W1024"],
 [["Shapes out (R)","512","512","1024"],["Hidden layers","1024 × 2","1024 × 3","2048 × 3"],["Fourier features","128, σ = 4","128, σ = 2","128, σ = 2"],
  ["Envelope","16x(1−x)y(1−y)","64·Π x(1−x)","64·Π x(1−x)"],["Parameters","1.84 M","2.89 M","11.0 M"],["Training GPU, fit time","A100 (time not recorded)","H200, 2072 s","H200, 4104 s"],
  ["Worst validation projection error","–","3.13 %","1.31 %"]],[34,22,22,22],
 "experiments/quadrature-burgers3d, jcp-wide-bank configs","The 3D banks also have K = 16/32/64 heads trained, but they did not generalise (K = 16 validation worst 41 %), so they are unused. Only the 2D bank has a head in use, and only as a side experiment.",size=28)
table("train","How is the bank trained?","On short simulations of single and multiple bumps",
 ["","2D","3D"],
 [["Training simulations","576 trajectories, 50 steps each","1536 trajectories, 12 saved steps each"],
  ["Initial conditions","one Gaussian bump; width 0.05–0.2, amplitude 0.5–2","1–3 blobs; width 0.10–0.20, amplitude 0.5–2"],
  ["Viscosity ν","0.01–0.1 (log-uniform)","0.01–0.1 (log-uniform)"],["Solver making the data","upwind + backward Euler, 256², Δt 0.005","upwind + backward Euler on 33³, 65³, 129³"],
  ["Loss","autodecoder: network + one code per snapshot, relative MSE + 1e-4 orthogonality penalty","projection loss: how much of each snapshot the shapes miss, + tail and whitening terms"],
  ["Optimiser","AdamW, lr 1e-3, 300k steps","Adam, lr 1e-3, 12k steps"]],[26,37,37],
 "sep_common.py, sep_solvers.py, train2w.py","The training data were made with a simple first-order solver on a modest grid. A control bank trained on even coarser 129-node data still reached 0.83 % worst error, so fine training data are not what makes the model accurate.",size=24)
fig("ordering","How do we get a dial r from one network?","Reorder the 512 shapes so the first r are the best r",F['errr'],"Error against bank width r at three meshes",
 ["QR of the shapes on a grid makes them orthonormal","SVD of all training coefficients sorts directions by how much they are used","Result: one rotation T; the first r columns are the best r-shape model","Error falls 128 → 256 → 384, then flattens at 512 (the time step takes over)"],
 "make_rotation.py; final report §3 (P1)",
 "This is what makes the bank tunable: smaller r is cheaper, larger r is more accurate, and no retraining is needed. 3D captures 99.97 % of coefficient energy with 128 shapes.")

# ---------------- Part 2
divider("p2","Part 2","The online solver, step by step","Six steps turn an initial field into the solution at any later time, on any mesh.")
flow=[("1 Fit","project u₀ onto the shapes"),("2 Test","residual against 4r sine tests"),("3 Quadrature","nonlinear term at m fixed points"),("4 Solve","Levenberg–Marquardt"),("5 Step","BE / CN-R / BDF2 in time"),("6 Decode","u = G(x)c on any mesh")]
fl=""
for i,(t,b) in enumerate(flow):
    col=TEAL if i in (1,2,3,4) else ACC
    fl+=box(t,b,col)+(arrow if i<5 else '')
add("algo",f'''<section id="algo" data-transition="fade" style="{SEC}">
{head("What happens when we run the model?","Fit once, then repeat steps 2–5 every time step, then decode")}
<div style="display:flex;align-items:center;gap:0px">{fl}</div>
{ul(["Steps 2–5 never touch the mesh: they work on r numbers, M = 4r tests and m quadrature points","The mesh only appears at the start (reading u₀) and the end (writing the output)","So solve cost is flat in the mesh, but reading and writing fields still grow with it"],30)}
{foot("qcore.py, offmesh.py, t2core.py")}
<aside>Orange steps touch the mesh; teal steps don't. This separation is the source of mesh-independent solve cost.</aside>
</section>''')
step("s1",1,"Fit the initial condition to the shapes","How do we get c at time zero?",
 ["c₀ = argmin_c  Σ_q w_q ( G(x_q)·c − u₀(x_q) )²"],
 ["u₀ is sampled at 48 × 48 Gauss points, not on the whole mesh","Closed-form weighted least squares: one small linear solve","Cost about 0.2 ms at r = 384"],
 "quadrature-study arms.py, qcore.py",
 "The initial field can come from any mesh; only 2304 point values are needed.")
step("s2",2,"Ask the solution to satisfy the PDE against sine tests","What equation do we solve each step?",
 ["r(c) = D·( A·c − A·c_old + Δt·( N(c) + ν·Λ·A·c ) )","A = Φᵀ G   (shapes against M = 4r sine tests, exact)","D = (1 + Δt·ν·Λ)⁻¹,  Λ = sine eigenvalues"],
 ["This is one backward-Euler step written in test space; other schemes change only the time weights","Linear terms (time derivative, diffusion) are exact: sines are eigenfunctions of the Laplacian","Only N(c), the nonlinear advection, needs approximating","LSPG: we minimise ‖r(c)‖ because there are more tests (4r) than unknowns (r)"],
 "quadrature-burgers2d report §2; FR:561",
 "Four tests per unknown is a design rule. Galerkin (as many tests as unknowns) was kept only as a diagnostic.",eqsize=32)
step("s3",3,"Evaluate the nonlinear term at m fixed points","How do we compute u·∇u without the mesh?",
 ["N(c) = Pᵀ [ (B·c) ⊙ (D·c) ]","B = G(x_q),  D = (∂x + ∂y) G(x_q),  P_qm = w_q·ψ_m(x_q)"],
 ["(x_q, w_q): a classical quadrature rule (Gauss or Fibonacci lattice), the same on every mesh","Derivatives D come from automatic differentiation of the network: exact, no finite differences","B and D are computed once offline; online it is two matrix–vector products and one multiply","This is Hari’s off-mesh quadrature; m is chosen per bank by a qualification test"],
 "offmesh.py; quadrature-burgers3d DESIGN",
 "m is large: 4181 points at r = 128 up to 36,864 at r = 512 in 2D. That number drives cost, see Part 3.",eqsize=32)
step("s4",4,"Solve the small nonlinear system","How do we find c?",
 ["J = Pᵀ [ diag(D·c)·B + diag(B·c)·D ]","(JᵀJ + λI)·δ = −Jᵀ·r     (Cholesky)","stop when ‖Jᵀr‖ / (‖J‖·‖r‖) ≤ 10⁻³"],
 ["Levenberg–Marquardt: Gauss–Newton with damping λ carried between steps","Warm start from a quadratic prediction of the previous steps","Typical work: about 2 Jacobian builds per step","In 3D, after 3 adaptive steps we run one fixed sweep per step"],
 "qcore.py:279–342; jcp-wide-bank DESIGN",
 "The Jacobian is exact and cheap because N is bilinear in the shapes. The per-step cost is roughly 2·m·M·r per Jacobian.",eqsize=32)
step("s5",5,"March in time","Which time scheme, and how big a step?",
 ["BE: first order    CN-R: Crank–Nicolson after 2 BE start-up steps    BDF2: second order"],
 ["2D: Δt = 0.005, 50 steps to T = 0.25; 3D: Δt = 0.01, 25 steps","Measured order: BE 1.0, CN-R 1.74–1.96, Galerkin-CN 2.0","Time stepping is the largest error source in the deployed model (Part 3)"],
 "t2core.py; FR:168–178",
 "CN-R is the accurate setting. The two backward-Euler start-up steps damp the early transient but cost a little order.",eqsize=30)
step("s6",6,"Decode on whatever mesh you want","How do we get a field back?",
 ["u(x) = G(x)·c     for any set of points x"],
 ["The same c decodes to 256², 4096², or scattered points","At r = 384, 1024²: decode 0.95 ms of a 85 ms query; at 4096²: 13.8 ms of 97 ms","Decoding is the only part that grows with the mesh"],
 "imp-faststep report (query split)",
 "If the user only needs a few point values or integrals, decode cost is negligible.")

# ---------------- Part 3
divider("p3","Part 3","What accuracy and speed depend on","Measured, not modelled: four knobs set the accuracy, three set the cost.")
table("acc-factors","What sets the accuracy?","Four things, in order of importance",
 ["Factor","Effect (2D, 1024², median / worst error)","Source"],
 [["Time step and scheme","BE 0.98 % / 3.0 % against CN-R 0.20 % / 1.8 % at r = 384","P1"],
  ["Bank width r","CN-R: r = 128 0.45 % → 256 0.21 % → 384 0.20 % → 512 0.20 %","P1"],
  ["Steepness a/ν beyond training","in range 0.03 % best fit; steeper fronts 3.5 % and 9.9 %","regimes"],
  ["Quadrature points m","must pass qualification, else the rollout drifts; past it, extra points change nothing","qualify"],
  ["The mesh","no effect: 8 of 8 settings flat from 256² to 4096²","P1"]],[28,52,20],
 "final report §3","The bank stops being the bottleneck around r = 384; past that, the time step dominates. Mesh has no effect.",size=28)
fig("budget","Where does the remaining error come from?","Time stepping dominates the deployed model",F['budget'],"Error budget split into bank, quadrature, solver and time components",
 ["Split at 1024²: bank floor, quadrature, solver, time","LSPG BE and CN-R: time is the largest part","Galerkin-CN at r = 128: the bank floor dominates","Quadrature and solver parts are tiny after qualification"],
 "Lane budget; FR:307–326",
 "This tells us where effort pays: a better time integrator, not a bigger bank. The fast-step lane tried exactly that (Part 4).")
table("cost-factors","What sets the cost?","Points × tests × width, not the mesh",
 ["Factor","Measured","Comment"],
 [["Quadrature points m","4181 (r 128) → 9216 → 16,384 → 36,864 (r 512)","grows faster than r"],
  ["Bank width r","kernel exponents 1.17 / 1.08 / 1.92 / 1.24 (declared 1, 1, 2, 3)","GPU far from large-matrix regime"],
  ["Tests M = 4r","query elasticity 0.53 in M, 0.65 in m, 0.65 in r","all three matter about equally"],
  ["Time steps","50 (2D) or 25 (3D); 84 ms evolve at r = 384","fixed Δt, about 2 Jacobians per step"],
  ["Mesh","solve ratio 4096²/256² = 1.01–1.02","only decode grows"]],[24,48,28],
 "final report §3.1; wide-bank report; imp-faststep report","Work per step ≈ 2·m·M·r per Jacobian. Because m and M both grow with r, cost rises steeply with r even though the GPU hides part of it.",size=27)
fig("kernels","How does each solver part scale with r?","Cost grows with r, but not at the textbook rates",F['kern'],"Per-call time of each kernel against r in 2D",
 ["Declared exponents: nonlinear term 1, Jacobian 1, normal equations 2, Cholesky 3","Measured in 2D: 1.17, 1.08, 1.92, 1.24","3D: 1.01, 1.03, 1.33, 1.03","At r ≤ 1024 the GPU is latency-bound, so Cholesky is much cheaper than r³"],
 "Lane p1-scaling","The registered hypothesis H-cost-r fails as stated, but the practical conclusion is fine: cost is dominated by the m·M·r products, not by the Cholesky.")
fig("rules","Which quadrature rules work?","Gauss and Fibonacci lattices; random points never qualify",F['family'],"Error against points for different rule families",
 ["Gauss reaches qualification at 9216 (r 256) and 16,384 points (r 384)","Sobol, Halton and Monte Carlo never qualify at any tested m","Fewer tests (smaller M) do not reduce the points needed","The rule must be re-selected for each retrained bank"],
 "Lane dims; FR:270–291; robust E2",
 "Random-like rules converge as one over root m; Gauss and lattice rules converge much faster for smooth integrands, which ours are.")
fig("whyspec","Why does a spectral solver beat us on speed?","Smooth Burgers needs very few sine modes",F['stepcost'],"Per-step cost of the ROM against the spectral solver",
 ["The matched spectral solver needs a median of only 11 modes per axis","Each ROM unknown costs 1.07×10⁸ flops per step; each spectral mode 2.3×10³: a factor 4.6×10⁴","The ROM has far fewer unknowns, but each is vastly more expensive"],
 "Spectral investigation, why-spectral report",
 "This is the core reason for the speed result. Our cost is set by m·M per unknown; a spectral method pays an FFT per mode. For smooth solutions in a box, the spectral method wins by construction.")


# ---------------- Part 4
divider("p4","Part 4","What works, what failed, and why","Every failure below was a pre-registered test that fired. For each, we separate what we measured from what we only suspect.")
cards("strengths","What does the method do well?","Six results that hold up",
 [("Flat cost","Solve time 256² → 4096²: ratio 1.01–1.02 (2D); 64³ → 384³: 1.01–1.09 (3D)"),("Flat accuracy","8 of 8 settings within margin across three meshes"),
  ("20× less memory","1.61 GB against 34.37 GB for the tensor method, 3D, r = 1024"),("0.83 %","worst error from a bank trained on coarse 129-node data, beating the 129- and 257-node solvers (7.58 %, 4.12 %)"),
  ("24 of 24","headline accuracy verdicts repeat on three retrained seeds"),("2.95×","faster than our own earlier 3D model, both quadrature-qualified")],
 "final report §3; jcp-wide-bank; jcp-smooth-bank; jcp-robust; jcp-dims",
 "These are the claims the paper can make. Also stable to T = 1, four times the usual horizon, with no error growth on 11 of 11 runs; that verdict is provisional because one cross-check reference failed its own order test.",cols=3,big=52)
def why(items):
    out=""
    for tag,t in items:
        col=TEAL if tag=="Measured" else (ACC if tag=="Hypothesis" else MIX)
        out+=f'<li><b><span style="color:{col}">{tag}:</span></b> {t}</li>'
    return out
def fail(id,question,answer,img,alt,result,whys,src,notes,verdict="Failed"):
    imgel=f'<img src="{img}" alt="{alt}" style="width:820px;height:470px;object-fit:contain;background:#FFFFFF;border:1px solid #DDE1E4;border-radius:12px">' if img else ''
    add(id,f"""<section id="{id}" data-transition="fade" style="{SEC}">
{head(question,answer,verdict,BAD)}
<div style="display:flex;gap:44px;align-items:start">
{imgel}
<div style="flex:1;display:flex;flex-direction:column;gap:18px">
<p style="font-size:26px;line-height:1.35;color:{INK}"><b>Result.</b> {result}</p>
<ul style="font-size:24px;line-height:1.35;color:{SOFT};display:flex;flex-direction:column;gap:10px">{why(whys)}</ul>
</div>
</div>
{foot(src)}
<aside>{html.escape(notes)}</aside>
</section>""")
table("scoreboard","Which big tests failed, and do we know why?","Four headline failures; three causes are measured",
 ["Test","Outcome","Best explanation"],
 [["Faster than best full solver","No: 0.05–0.71× (2D), 0.008–0.16× (3D)","<b>Measured:</b> spectral needs ~11 modes per axis; each of our unknowns costs 4.6×10⁴× more"],
  ["Beat tuned POD + ECSW","No: dominated 8/8 (2D), 7/9 (3D)","<b>Measured:</b> 2 Jacobians per step; 33–43 points per unknown against 4–8"],
  ["Steeper fronts than training","No: 0/8 qualified; bank floor 3.5–9.9 %","<b>Measured:</b> steps can’t be certified on steep states, even at Δt/8"],
  ["Nonlinear head (K)","No: linear r = 128 better at 0.59× cost","<b>Hypothesis:</b> the head moves only inside the frozen span"]],[24,30,46],
 "Failure-explanation sweep over all lane reports","Measured means a diagnostic in the lane shows it; hypothesis means it is plausible but nobody has tested it by ablation. Each of these four gets its own slide next.",size=27)
table("scoreboard2","Which smaller tests failed?","Four more, all tied to the solver rather than the bank",
 ["Test","Outcome","Best explanation"],
 [["r 384 → 512 improves accuracy","No: worse on 22 of 38 cases (CN-R)","<b>Measured:</b> time error dominates; floor barely moves (0.028 → 0.025 %). Link not tested"],
  ["Cost follows textbook exponents","No: 1.17 / 1.08 / 1.92 / 1.24 against 1, 1, 2, 3","<b>Hypothesis:</b> small GPU kernels are latency-bound (not profiled)"],
  ["Early 3D speed-ups (5.4×, 17×)","Withdrawn: winners failed the quadrature check; qualified 2.95×","<b>Measured:</b> cheap rules don’t converge to their finer successors"],
  ["3D fixed-sweep CN at 2.5Δt","Ineligible in 10 of 10 cells","<b>Descriptive:</b> too few sweeps or a binding trust clip at hard steps"]],[24,32,44],
 "p1-scaling; jcp-budget; jcp-qualify; jcp-dims; jcp-speed3d","The 4× fewer points giving only 2.1× speed-up (imp-hrcost) supports the latency-floor hypothesis but doesn't prove it.",size=27)
fail("f-fom","Why are we slower than the best full solver?","Smooth Burgers in a box is the spectral method’s home ground",F['union'],"Speed-up against the best matched full solver, all below 1",
 "Best matched solver is faster in 21 of 21 2D cells (speed-up 0.05–0.71) and all 3D cells (0.008–0.16). At 4096², r = 384: 97 ms against 11.7 ms.",
 [("Measured","a 1 % fit needs only a median 11 sine modes per axis (2D), 9 (3D)"),("Measured","each ROM unknown costs 1.07×10⁸ flops per step against 2.3×10³ per spectral mode"),
  ("Measured","we take 50 implicit steps with ~2.1 Jacobians each; spectral takes 20 explicit ETDRK4 steps"),("Measured","our 4096² decode alone (14.3 ms) is longer than the whole spectral query"),
  ("Derived","to win 2× we’d need ~1000× compression over sines; we have 2–6×")],
 "why-spectral report; final report §3.4",
 "This is structural, not a tuning problem. Even with free evolution, decode caps the r = 384 speed-up near 0.8. A win needs a problem without a fast transform, or with a narrow family; the pilot hinted at 1.7–3× in a narrow family but with a POD stand-in, not our bank.")
fail("f-pod","Why does a tuned POD + ECSW model beat us?","Same accuracy, a third of the per-step work",F['pod'],"Error against time, POD points below and left of ours",
 "Dominated in 8 of 8 2D configurations at 256² and at 1024², and 7 of 9 in 3D. Example: ours 86.8 ms, 1.78 % worst; POD + ECSW 30.7 ms, 1.35 %.",
 [("Measured","per step 1.74 ms (ours) against 0.61 ms; our Jacobian kernel alone is 0.53 ms and runs twice"),("Measured","we use 33–43 quadrature points per unknown; ECSW uses 4–8"),
  ("Measured","POD’s time barely moves with its point count: it sits on the GPU latency floor"),("Hypothesis","POD snapshots come from the target mesh with central differences; the bank learned from upwind 256² data")],
 "hr-baselines; improvement-ideation §2.2",
 "Developmental, in-sample, unadjusted comparison. The two fixes this suggested (fewer Jacobians, fewer points) were tried in the improvement campaign and did not close the gap.")
fail("f-fronts","Why does it fail on steeper fronts?","The time steps can’t be certified, even at an eighth of Δt",F['regimes'],"Error rising with steepness beyond the training range",
 "Steep fronts C1 / C2: 0 of 8 configurations qualified; median error 5–11 % and 13–25 %. Error correlates with steepness (Spearman 0.88–0.93).",
 [("Measured","the bank floor itself rises: 3.5 % (C1), 9.9 % (C2) against 0.03 % in range"),("Measured","retraining with steep data (arm b) helped only 1.4× (needed 2×)"),
  ("Measured","shrinking the step to Δt/8 still leaves 7 of 8 steep cases unconverged"),("Measured","quadrature is not the limit: largest ρ 0.039 against a 0.116 bar"),
  ("Hypothesis","upwind training data smear fronts; the test was inconclusive")],
 "regimes; imp-fronts E8/E10",
 "So the binding constraint on steep fronts is time-step certification, not the shapes or the points. An adaptive embedded integrator or a moving/local basis is the designed next step; not run.")
fail("f-heads","Why doesn’t the nonlinear head help?","At equal cost, more linear shapes beat a nonlinear map",F['heads'],"Error against cost for heads and linear banks",
 "Head K = 16: 1.44 %. Linear r = 128: 1.1 % at 0.59× the head’s cost. Heads K = 4 and 8 never qualified.",
 [("Measured","small heads fail the solver and quadrature checks (worst 12–36 %)"),("Hypothesis","the head only moves within the frozen 512-shape span, so it cannot beat that span’s floor"),
  ("Hypothesis","repair ideas (better initial fit, different solver branch) untested")],
 "jcp-heads; final report §3.10",
 "At matched dimension the head does win (1.44 % against 10.2 % at r = 16), so it is a real compression, just not a cost win.")
table("campaign","Did the improvement campaign rescue speed or accuracy?","Seven lanes so far; none crossed its bar",
 ["Lane","Idea","Outcome"],
 [["E5 fast step","cheaper steppers (Galerkin, ETDRK4)","14–24× faster evolve; worst 1.82–1.93 % vs 1.78 %"],
  ["E6 cheap quadrature","bank-native tests; fitted weights","best qualified 1.07×; fitted weights fail check"],
  ["E7 many meshes","one bank amortised over meshes","POD matches us 126/126 at 0.10–0.13× cost"],
  ["E3 variable coeff.","harder PDE for the full solver","full solver stays cheap (gate 0.86, needs 2)"],
  ["E8/E10 fronts","steep data, wall-safe bank, Δt/8","1.4× better (needs 2×); steps unconverged"],
  ["E2 geometry","no fast transform on an L-shape","SEM and POD faster: S 0.015–0.21 (provisional)"],
  ["E1 box sweep","narrow families in the box","running; bank + ETDRK4 S 0.81–1.38 (interim)"]],[20,34,46],
 "Improvement campaign lane reports, 22 Oct","Every stop rule was pre-registered before results were read. E2 and E1 are still running and their numbers are not yet audited. S is our speed-up against the best competitor; above 1 means we are faster.",size=25)
fail("f-fast","Why didn’t the fast time stepper count?","It is 14–24× faster but slightly worse on the single hardest case",None,"",
 "Median error improves 6× (0.034 % against 0.198 %) and 36 of 38 cases get better. The worst case moves from 1.78 % to 1.82–1.93 %, beyond the ε margin, so the registered rule stops it.",
 [("Measured","the failure is one case (val32 #19), the in-range case with the worst bank floor"),("Hypothesis","Galerkin in this span can’t get the worst case below ~1.5 % (the converged Galerkin endpoint is 1.54 %)"),
  ("Measured","at 4096² decode (13.8 ms) is most of the 18.2 ms ETDRK4 query"),("Measured","a POD + ETDRK4 competitor is still faster (0.62–0.80× our time), though it fails its own quadrature check")],
 "imp-faststep report and EXPORT","This is the closest call. If the paper reports median error, ETDRK4 is a large win; under the registered worst-case rule it is not.",verdict="Stopped")
add("thread",f"""<section id="thread" data-transition="fade" style="background:{ACC};color:#FFF8F2;font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#FFE3CC">The common thread</p>
<h2 style="font-family:{SERIF};font-size:76px;font-weight:600;line-height:1.1;color:#FFF8F2">The shapes are good enough. What costs us is the work per time step, the number of steps, and decoding</h2>
<p style="font-size:34px;line-height:1.4;color:#FFF0E3">Measured in four lanes: two Jacobians per step, 33–43 points per unknown, steps that won’t certify on steep states, and a decode that outweighs a spectral query. Fixing the shapes (wider banks, steep data, wall-compatible envelope) moved errors by 1.1–1.4×.</p>
<aside>The one lever nobody has tested properly is the time integrator together with step certification: an adaptive embedded scheme. That's the most defensible next experiment if we continue on Burgers.</aside>
</section>""")
cards("open","What isn’t measured yet?","Still running, or deliberately not opened",
 [("E1","box-family sweep running: 3 cells collected, 3 running, audit pending"),("E2","geometry: second cell (H-shape) running; L-shape numbers not yet audited"),
  ("Sealed","128 (2D) and 64 (3D) test cases stay closed until every choice is frozen"),("NS","Navier–Stokes 2D/3D: no off-mesh lane yet"),
  ("Deferred","full POD baseline set, more seeds, 3D new regimes"),("Untested","adaptive time integrator; moving or local basis for fronts")],
 "Campaign status 22 Oct","These are gaps in the evidence, not hidden results.",cols=3,big=56)
add("close",f"""<section id="close" data-transition="fade" style="background:{INK};color:{LIGHT};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:28px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#E9A06A">What the paper can say</p>
<h2 style="font-family:{SERIF};font-size:80px;font-weight:600;line-height:1.1;color:{LIGHT}">One trained bank, one dial, the same answer on every mesh, at a fraction of the memory</h2>
<p style="font-size:34px;line-height:1.4;color:#BFD0DC">And, stated plainly: on smooth Burgers it is not faster than a spectral solver or a tuned POD model, and it doesn’t extrapolate to steeper fronts. We know which of those reasons are measured and which are still hypotheses.</p>
<aside>Recommended framing: a characterisation paper of a mesh-independent, tunable, mesh-free reduced model, with the speed and accuracy limits measured and explained.</aside>
</section>""")

for id,s in slides: open(f'project/slides/{id}.html','w').write(s)
order=[i for i,_ in slides]
deck={"v":4,"createdOnFiles":{"v":1,"at":"2026-10-22T20:30:00Z"},"lists":"css","title":"Tunable Bank: How It Works","order":order,"cover":"cover",
 "sections":{"s0":{"description":"Cover and map","start":"cover"},"s1":{"description":"How we got here: off-mesh quadrature","start":"p0"},"s2":{"description":"Architecture and training","start":"p1"},
  "s3":{"description":"The online solver step by step","start":"p2"},"s4":{"description":"What accuracy and speed depend on","start":"p3"},"s5":{"description":"What works, what failed, and why","start":"p4"}},
 "faces":{"source-serif-4":{"family":"Source Serif 4","href":"https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
  "ibm-plex-sans":{"family":"IBM Plex Sans","href":"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"},
  "jetbrains-mono":{"family":"JetBrains Mono","href":"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"}},"designSystems":[]}
json.dump(deck,open('project/deck.json','w'),indent=1)
print(len(slides),[i for i,_ in slides])
