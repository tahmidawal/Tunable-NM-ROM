"""Build the three-week progress page (1-22 Oct 2026) with the lanes' own plots embedded as data URIs."""
import base64, html, pathlib
R = pathlib.Path("/home/tahmid/Dev/Tunable-NM-ROM-Claude")
N, H = R/"notebook/figs", R/"reports/2026-10-17-full-experiment-plan/final/figs"
F, B = R/"reports/2026-10-21-improvement-campaign/final/figs", R/"worktrees/2026-10-22-imp-box/experiments/imp-box/plots"

SECTIONS = [
 ("1–9 October", "Hari's off-mesh quadrature and the accuracy lanes",
  "The off-mesh classical quadrature was put inside our full-scale Burgers models, and four accuracy lanes tested what actually improves the reduced model.",
  [(N/"a1_error_vs_mesh.png", "Off-mesh quadrature makes the error mesh-independent",
    "Top row: worst error against the refined reference. The off-mesh rules (flat lines, about 3.1 % and 4.7 % in 3D, 2.75 % in 2D) give the same error on every mesh; the old tensor model (3D) and the mesh-based rules (2D) change with the mesh. References were first-order then, so the levels are provisional."),
   (N/"c2_fom_vs_rom.png", "Second-order time stepping was the clear accuracy win",
    "Error against cost for full-order solvers (dashed) and reduced models (solid) with the same time schemes at 1024². Second-order steps cut the reduced model's error several-fold at equal cost; against a fair second-order full solver the matched speed-up rose from 1.76× to 3.65× (provisional, first-order references)."),
   (N/"c1_3d_j4_memory_vs_Rp.png", "Memory: the off-mesh rule stays small as the bank widens",
    "Memory of the nonlinear term against bank width R′ in 3D. The off-mesh rule stays near 1.6 GB at R′ = 1024, against about 34 GB for the exact tensor."),
  ]),
 ("13 October", "Second-order references and the combined method",
  "Second-order reference solutions (E0) replaced the old first-order ones, so accuracy numbers stopped being provisional. The combined method (second-order steps plus the wider bank) was scored against them.",
  [(N/"e0_order.png", "The new references converge at second order",
    "Observed convergence order of the E0 references. These references are the yardstick for every accuracy number from here on."),
   (N/"s3_mesh_time.png", "3D: query cost is flat from 64³ to 384³",
    "Solve time per query against mesh size for the 3D reduced models of the speed lane. The cost does not grow with the mesh."),
  ]),
 ("17–20 October", "Burgers 2D/3D campaign (ten lanes, 168.8 GPU-h)",
  "The campaign answered the advisor's question (does cost depend on the bank width, not the mesh?) and compared the method against the strongest alternatives.",
  [(N/"final_cost_vs_mesh.png", "Cost depends on the bank width, not on the mesh",
    "Solve cost across meshes for several bank widths r, in 2D and 3D. Each width keeps the same cost on every mesh; wider banks cost more."),
   (N/"p1_error_cnr.png", "Accuracy is mesh-independent and improves with r up to 384",
    "Worst and median error against r on three meshes. The curves lie on top of each other across meshes; going from 384 to 512 no longer helps."),
   (N/"final_union_speedup.png", "A coarse spectral solver is faster at equal accuracy (K2 fires)",
    "Speed-up of every qualified reduced model over the best accuracy-matched full solver. Every point is below 1 (the full solver is faster): 2D 0.05–0.71×, 3D 0.008–0.16×."),
   (H/"hrb_front-c2d1024.png", "Tuned POD + ECSW beats our configurations (H-base fails)",
    "Error against cost at 1024²: a tuned standard reduced model (POD with ECSW / Q-DEIM hyper-reduction) is both faster and more accurate than our qualified configurations (developmental, in-sample)."),
   (N/"r_error_vs_s.png", "The model fails on fronts steeper than training",
    "Error against front steepness s = a/ν. Inside the training range the error is low; on steeper fronts (C1, C2) the bank cannot represent the solution (floor 3.5–9.9 %)."),
  ]),
 ("21–22 October", "Improvement campaign: trying to fix the three weaknesses",
  "Ten experiments, each with a stop rule written down before it ran. Accuracy is against the E0 references; speed-ups are upper bounds against the tested competitor ladders.",
  [(F/"imp-faststep__worst_error_vs_evolve_time.png", "E5, faster time step: 14–24× faster evolution, but a slightly worse worst case",
    "Worst error against evolution time. The Galerkin ETDRK4 arms are far to the left (much faster) but sit just above the incumbent's worst error (1.82 % against 1.78 %), so the stop rule fires."),
   (F/"imp-hrcost__error-vs-speedup.png", "E6, cheaper nonlinear term: small gains only",
    "Worst error against speed-up for smaller quadrature rules and fitted weights. The best gain is 1.24× (1.5× was needed), and the rules that pass the quadrature check lose accuracy. Both stop rules fire."),
   (F/"imp-multimesh__cost_vs_queries.png", "E7, multi-mesh cost: a plain POD reused across meshes is cheaper",
    "Total cost (setup plus queries) against the number of queries over 1–3 meshes. A continuous POD built once matches our worst error at 0.10–0.13× our cost on every workload, so the comparative multi-mesh cost claim is withdrawn."),
   (F/"imp-varcoef__fom_cost_vs_error.png", "E3, variable diffusion coefficient: the full solver is still as fast",
    "Cost against error of the tuned full-order solvers. The fastest (an exponential integrator) costs 0.86–0.92× our box model, not the 2× slower the plan required, so no model was trained."),
   (F/"imp-fronts__a4_hybrid_floors.png", "E8, steep fronts: adding sines to the bank does not reach the bar",
    "Best-possible (floor) error on the steep cases for the bank, plain sines and a bank+sine hybrid. The hybrid reaches 4.35 % against the 2.93 % needed."),
   (F/"imp-fronts__phaseb_medians.png", "E8, steep fronts: retrained banks help only a little",
    "Median error on the steep test cases for the control retrain (K1) and the two retrained banks. Steeper training data improves the median 1.4× (2× needed), and no steep case passes the solver check. E10's finer time steps also fail; the limit is time-step certification."),
   (F/"imp-geometry__S_per_family.png", "E2, L-shape and box with a hole: the bank is much slower than POD",
    "Speed-up of the bank against each competitor family per error target. Against the best competitor (POD-Galerkin or POD + ECSW) it is 0.013–0.22×; it beats finite differences with multigrid only at the 10 % target. The geometry claim is not supported."),
   (B/"S_vs_W_worst.png", "E1, box family width × steepness (still running, provisional)",
    "Speed-up against family width W at three error targets and four steepness levels, against a dense-matrix spectral solver. The POD arms and the bank with the ETDRK4 step sit around 0.7–1.5×; the bank with the current LSPG stepper is far below. Interim, unaudited for the final report; A3 and A4 panels pending."),
  ]),
]

def img(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

parts = []
for when, title, lede, figs in SECTIONS:
    cards = "".join(
        f'<figure><h3>{html.escape(t)}</h3><div class="frame"><img src="{img(p)}" alt="{html.escape(t)}" loading="lazy"></div>'
        f'<figcaption>{html.escape(c)}</figcaption></figure>' for p, t, c in figs)
    parts.append(f'<section><p class="when">{when}</p><h2>{html.escape(title)}</h2><p class="lede">{html.escape(lede)}</p>{cards}</section>')

tpl = (pathlib.Path(__file__).parent/"template.html").read_text()
(pathlib.Path(__file__).parent/"index.html").write_text(tpl.replace("<!--SECTIONS-->", "\n".join(parts)))
print("ok", sum(len(s[3]) for s in SECTIONS), "figures")
