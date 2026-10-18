#!/home/tahmid/Dev/.venv/bin/python
"""Render a manuscript results extract from the committed paper's generated data.

All table cells and empirical prose numbers come from the existing run-JSON ->
paper/gen_tables.py -> tables-md pipeline. This second rendering selects columns
and writes captions; it neither recomputes scientific verdicts nor reruns jobs.
The manifest preserves the source commit, input hashes and upstream provenance.
Run with /home/tahmid/Dev/.venv/bin/python reports/generate_paper_results_2026_09_19.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_COMMIT = "10498e2e0e264661d702b7606f49133ef7d5be17"
STEM = "2026-09-19-paper-results"
INPUTS = {}
TABLES = []


def read(path):
    data = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PAPER_COMMIT}:paper/{path}"])
    INPUTS[path] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return data.decode()


NUM = json.loads(read("tables-md/numbers.json"))
UPSTREAM = json.loads(read("tables/provenance.json"))
read("gen_tables.py")


def n(key):
    return NUM[key]


def ns_config(attempt):
    rev = UPSTREAM["sources"]["ns2d_summary"]["pinned_commit"]
    path = f"experiments/ns2d/artifacts/{attempt}/result.json"
    data = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{rev}:{path}"])
    INPUTS[f"{rev}:{path}"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return json.loads(data)["config"]


def table(name):
    lines = [s for s in read(f"tables-md/{name}.md").splitlines() if s.startswith("|")]
    cells = lambda line: [v.strip() for v in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
    headers, rows = cells(lines[0]), [cells(line) for line in lines[2:]]
    assert rows and all(len(row) == len(headers) for row in rows), name
    return headers, rows


def render(title, headers, rows, sources, note=""):
    assert rows and all(len(row) == len(headers) for row in rows), title
    number = len(TABLES) + 1
    TABLES.append({"table": number, "title": title, "sources": sources, "rows": len(rows)})
    out = [f"**Table {number}. {title}**", "", "| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    if note:
        out += ["", note]
    links = [f"[{name}](../worktrees/2026-09-16-paper-refresh/paper/tables-md/{name}.md)" for name in sources]
    out += ["", "Generated source: " + ", ".join(links) + "."]
    return "\n".join(out)


def pick(rows, indices):
    return [[row[i] for i in indices] for row in rows]


def main():
    out = ["# Numerical experiments and results",
           "This manuscript-style extract reports the completed campaign using the corrected paper snapshot. "
           "The measurements are collected; reporting remains provisional where independent audits or audit dispositions are outstanding, as noted beside the affected results.",
           "The experiments assess three questions: whether a nonlinear head improves compression at matched dimension; "
           "whether nested corrections provide an accuracy–cost family without retraining; and whether any resulting operating point "
           "is competitive with tuned full-order and linear reduced models. We report negative results alongside the successful cases.",
           r"For the Burgers panels, $E_{\mathrm{evol}}$ is the maximum error over cases and evolved output times, "
           r"$E_{\mathrm{all}}$ includes the initial state, and $E_0$ is initial-state compression error. "
           r"These errors use the initial-field norm and the converged same-grid solution; $E_{\mathrm{ref}}$ instead uses the fine numerical reference. "
           "Other experiments retain their stated metrics, identified in the captions. Costs are medians of repeated measurements after GPU burn-in. "
           "Device time charges resident computation; complete-query time includes host input and output. All cost comparisons are within an allocation. "
           r"For the Burgers panels, only solves meeting the fixed stationarity threshold $10^{-6}$ at every step, or the residual exit rule, "
           "and the applicable quadrature checks are admissible. A nondominated point has no admissible competitor with both lower cost and lower error."]

    _, panel = table("T05m_panel_summary")
    out.append("On the smaller Burgers meshes, tuned full-order solvers dominate every admissible reduced model. "
               "At the largest mesh, the lowest-rank quadrature models reach the evolved-error frontier, while initial-state compression removes "
               "this advantage when all output times are scored. The largest-mesh panel uses different hardware, so the three rows do not establish a hardware-independent crossover.")
    out.append(render("Same-allocation Burgers comparisons.",
                      ["Mesh", "GPU", "Admissible reduced", "Nondominated, evolved", "Nondominated, all times", r"$T_{\mathrm{ROM,min}}/T_{\mathrm{FOM,min}}$"],
                      pick(panel, [0, 2, 4, 5, 6, 7]), ["T05m_panel_summary"],
                      "A cost ratio above unity means the cheapest admissible reduced method costs more than the cheapest full-order setting; those settings need not have equal errors. "
                      "Each row is a separate job. Eligibility follows the corrected fixed-tolerance rule. The intermediate-mesh protocol also has a disclosed post-data interpretation of a transfer gate; "
                      "frontier membership is an empirical result under that disclosed protocol."))

    _, fixed = table("T04_rank_vs_tests")
    fixed = [r for r in fixed if r[0].startswith("fixed")]
    out.append(r"Holding the test count $M$ fixed isolates correction rank $q$. "
               f"The original fixed-$M={n('nQxmFixedM')}$ ladder reduces evolved error by ${n('nQxmErrSpan')}\\times$ "
               f"for ${n('nQxmCostSpan')}\\times$ cost, with all displayed rungs converged. "
               f"The smaller fixed-test ladder spans only ${n('nQxmTwoFiftySixErrSpan')}\\times$. "
               "This supports the correction-rank mechanism on the original registered range of one checkpoint. "
               "The later extension failed its acceptance rule and is excluded from this span. The separate registered hypothesis that both factors pass their individual thresholds is not satisfied; "
               "the qualitative observation that both affect error should not be presented as that formal pass.")
    out.append(render("Correction rank at fixed test count on Burgers.",
                      [r"$M$", r"$q$", r"$E_{\mathrm{evol}}$ (%)", "Device time (ms)"],
                      pick(fixed, [2, 1, 3, 4]), ["T04_rank_vs_tests"],
                      "All displayed rows belong to the same timing job and development checkpoint. This is the fixed-test experiment; the seed replication below uses a scheduled test count. "
                      "Provenance and amendment disclosures elsewhere in this lane remain under audit disposition."))

    _, sched = table("T03m_ladder_main")
    _, split = table("T03_tunability")
    tzero = {r[0]: r[-1] for r in split}
    out.append(r"With $M=4(K+q)$, dense and sampled residual evaluation trace similar error ladders. "
               f"Against the fine reference, the coarse-mesh discretisation error is ${n('nPanelRefDiscretisationTwo')}\\%$; "
               "the reduction error falls substantially while reference error changes much less. "
               "Quadrature reduces evaluation cost, but a passing stored rule and a reproducible construction are different claims.")
    out.append(render("Scheduled Burgers ladder and quadrature costs.",
                      [r"$q$", r"$M$", r"Dense $E_{\mathrm{evol}}$ (%)", r"$E_{\mathrm{all}}$ (%)", r"$E_0$ (%)", r"Dense $E_{\mathrm{ref}}$ (%)", "Dense ms", r"EQ $E_{\mathrm{evol}}$ (%)", "EQ ms", "EQ construction"],
                      [[r[0], r[1], r[2], r[3], tzero[r[0]], r[4], r[5], r[6], r[7], r[8]] for r in sched],
                      ["T03m_ladder_main", "T03_tunability"],
                      "Development cohort; one job. For these selected rules, dense and EQ all-times errors coincide at displayed precision and are set by initial compression. "
                      "The EQ rules are the replication-selected set. Single-draw results are provisional with respect to construction reproducibility. "
                      "The uncorrected development result is not a generalisation estimate; the sealed cohort below exposes its failure."))

    _, sealed = table("T13_sealed")
    _, verdicts = table("T13b_sealed_verdicts")
    out.append(f"The sealed cohort was opened after choices were frozen. All {n('nSealedCheckpoints')} checkpoints yield monotone ladders, "
               f"and {n('nSealedKnobBarCount')} meet the full ladder criterion. Top-rung error ranges from {n('nSealedAllTopMin')}% to {n('nSealedAllTopMax')}%. "
               "The incumbent's uncorrected prediction converges to a wrong branch on one sealed case. The generalisation criterion therefore fails, "
               "and sealed results replace the development baseline as the headline. An unconverged rung in one retrained seed remains reported as run.")
    out.append(render("Sealed-cohort Burgers replication.",
                      [r"$q$", r"$M$", "Incumbent worst evolved (%)", "Retrained seeds: mean ± SD of worst evolved (%)", "Checkpoints converged"],
                      pick(sealed, [0, 1, 2, 4, 8]), ["T13_sealed"],
                      "The seed statistic averages each checkpoint's worst-case error; it is not the median error over individual cases. "
                      "These are dense scheduled-test ladders, so they do not independently replicate the fixed-test experiment or quadrature construction. "
                      "The independent audit supports the ladder verdicts but identifies outstanding check-count and disclosure defects."))
    out.append(render("Sealed-cohort ladder acceptance by checkpoint.",
                      ["Checkpoint", "Monotone, evolved", "Monotone, all times", "Every rung converged", "Error span", "Cost span", "Full criterion"],
                      pick(verdicts, [0, 2, 3, 4, 5, 6, 7]), ["T13b_sealed_verdicts"],
                      "Spans are computed within the sealed-cohort job. The incumbent's large error span includes its failed uncorrected cold start and should not be interpreted as typical improvement."))

    _, head = table("T06a_head_burgers")
    dim = head[0][1]
    selected_head = [r for r in head if r[1] == dim and "dense" not in r[0]]
    out.append("At matched latent dimension, the nonlinear head is substantially more accurate than the tested linear and quadratic alternatives on Burgers. "
               "This comparison isolates compact representation; it does not establish a cost advantage over higher-dimensional POD or a tuned full-order solver.")
    out.append(render("Matched-dimension head ablation on Burgers.",
                      ["Model", "Dimension", "Worst same-grid, all times (%)", "Device time (ms)", "Stationary"],
                      pick(selected_head, [0, 1, 4, 6, 7]), ["T06a_head_burgers"],
                      "A separate development-cohort job. Linear alternatives are the fitted maps actually tested, rather than a universal bound on every possible linear model. "
                      "All-times error can be controlled by initial compression."))

    _, eq = table("T09_eq_ladder")
    out.append("Rules fitted on stored states are evaluated on held-out reachable states and then independently redrawn. "
               "A good fitting residual alone is insufficient. The original timed ladder is monotone for its passing draws, while the higher-rank constructions fail on some redraws. "
               "Construction labels must name the population and certification threshold to which their counts apply.")
    out.append(render("Held-out certification of the original timed quadrature constructions.",
                      [r"$q$", r"$m$", r"$\rho_{\max}$, timed draw", "Primary construction status", "EQ evolved error (%)", "EQ ms", "Dense ms"],
                      pick(eq, [0, 1, 3, 4, 5, 7, 9]), ["T09_eq_ladder"],
                      f"The primary threshold is $\\rho_{{\\max}}\\le {n('nEqtopBar')}$. Timings compare methods within the original ladder job; construction counts include the separate replication experiment. "
                      "This table uses the original timed constructions, whereas the scheduled panel above uses later selected rules; their high-rank statuses must not be pooled. "
                      "Marginal constructions are provisional for reproducibility. Other status-label and retraction disclosures in the lane remain under audit disposition."))

    _, ops = table("T14_operators")
    wanted = ["unet-refine", "tsol-refine", "rom", "fno-large", "same_nt1e-2_dt005"]
    opmap = {r[0].strip('`'): r for r in ops}
    names = {"unet-refine": "U-Net, validation-selected", "tsol-refine": "Transolver, validation-selected", "rom": "ROM comparator", "fno-large": "FNO, validation-selected", "same_nt1e-2_dt005": "Efficient FOM comparator"}
    out.append("On shared Burgers data, the validation-selected U-Net and Transolver are more accurate than the ROM configuration included in the operator comparison on its matched cohort. "
               "The FNO is less accurate there. This conclusion does not compare either operator with the highest correction rank, nor does it assert a uniform ordering of validation tails. "
               "The U-Net advantage also survives the available precision and second-seed controls.")
    out.append(render("Neural operators on shared Burgers data.",
                      ["Method", f"Matched {n('nOpCohortSize')}-case worst (%)", "Validation worst (%)", "Validation median (%)", "Still improving"],
                      [[names[k], opmap[k][5], opmap[k][6], opmap[k][7], opmap[k][4]] for k in wanted], ["T14_operators"],
                      "Error is relative to the fine numerical reference and normalized by the initial-field norm. All methods return the supplied initial state exactly in this comparison. "
                      "The cohort, reference and ROM rollout convention differ from the correction-ladder panels. Operators were selected by validation mean error, not by the displayed matched-cohort tail. "
                      "U-Net and Transolver have no same-allocation ROM timing comparison, so no speed ratios are given. Training-limit and pending-independent-audit qualifications apply."))

    _, p = table("T11b_poisson")
    _, h = table("T11d_heat")
    _, w = table("T11a_waves")
    mesh = p[-1][0]
    pr = [r for r in p if r[0] == mesh and (r[1] in ["$q{=}0$", "linear top rung ($q{=}R$, QR)", "direct DST"] or "k'{=}512" in r[1])]
    hr = [r for r in h if r[0] == mesh and "coarse" not in r[1]]
    wr = [r for r in w if r[0] == mesh and r[1].strip('`') in ["head_q0", "nested_q32", "linear_bank64", "pod_k64", "dst"]]
    lin = [["Poisson", r[0], r[1], "Same-grid", r[3], r[6], r[5]] for r in pr]
    lin += [["Heat", r[0], r[1], "Physical, current-relative", r[2], r[4], r[5]] for r in hr]
    lin += [["Wave", r[0], r[1], "Energy-state", r[3], r[5], r[6]] for r in wr]
    out.append("For the linear PDEs, unrestricted evolution or solution in the learned bank removes the nonlinear optimization cost. "
               "On Poisson the direct linear endpoint is more accurate and cheaper than the head, but the full-order transform remains superior. "
               "On heat it improves both error and cost relative to the head. On waves it attains nearly the best correction-rung accuracy at substantially lower solver cost, "
               "while matched-rank POD is more accurate. The small difference between the wave correction rung and the linear endpoint does not support claiming that the latter is strictly the most accurate rung.")
    out.append(render("Linear-PDE endpoints and competitive controls at the finest reported mesh.",
                      ["PDE", "Mesh", "Method", "Error metric", "Worst error (%)", "Device ms", "Complete-query ms"],
                      lin, ["T11b_poisson", "T11d_heat", "T11a_waves"],
                      "Compare methods within each PDE block only; the blocks use different error norms and jobs. The wave state includes displacement and velocity in its energy norm. "
                      "An error displayed as zero is agreement at printed precision. Poisson and wave report audits remain pending; heat is an earlier independently audited cell. "
                      "Device-only and complete-query costs can lead to different practical comparisons."))

    _, ls = table("T18m_lshape_main")
    out.append("On the L-shaped domain, the loss of a separable full-order transform changes the cost comparison. "
               f"The reported head is ${n('nLshapeNeuralCheaperVsCheapestTwoFiftySix')}\\times$ and "
               f"${n('nLshapeNeuralCheaperVsCheapestFiveTwelve')}\\times$ cheaper than the cheapest measured same-job full-order setting at the two larger meshes. "
               f"POD is cheaper still, with approximately {n('nLshapePodOverHeadErrFiveTwelve')}% more worst-case error at the finest mesh. "
               "This is a reduced-model cost benefit on a linear PDE, not evidence that nonlinearity is necessary.")
    out.append(render("L-shaped Poisson: complete-query cost and same-grid error.",
                      ["Mesh", "Cheapest FOM", "FOM error (%)", "FOM ms", "POD-128 error (%)", "POD ms", r"Head $q=64$ error (%)", "Head ms", "FOM/head cost"],
                      [[r[0], r[3], r[5], r[4], r[6], r[7], r[9], r[10], r[11].split(" / ")[-1]] for r in ls],
                      ["T18m_lshape_main"],
                      "Every ratio uses its row's allocation. The head uses the signed-distance bank and the same reported correction rank across meshes. "
                      "The numerical verifier was repaired and its comparisons match. Reporting remains provisional pending disclosure of post-data gates and fine-reference coverage. "
                      "A separate free-bank experiment selected its test count using development data; it is not included here. No matched-rank POD-512 control was run on this domain."))

    _, ns = table("T11e_ns")
    ns = [r for r in ns if r[0].startswith("$256^2$")]
    ns_configs = [ns_config(a) for a in ["ns203", "ns204", "ns303", "ns302"]]
    assert all(str(cfg["R"]) == row[1] for cfg, row in zip(ns_configs, ns, strict=True))
    out.append("The tested decaying Navier–Stokes family does not establish the required nonlinear representation advantage. "
               f"Across the retained settings, the ratio of POD error to best-found head error is {n('nNsCellRatioMin')}–{n('nNsCellRatioMax')}, "
               "below the registered factor-of-two threshold. Increasing latent dimension, narrowing the family, and increasing training data do not produce a passing head. "
               "Some best-found fits terminate at their iteration budget, so these values are upper bounds on the achievable manifold error. "
               "Training-versus-held-out error ratios use different normalizations in the existing record and are not used here to claim a demonstrated capacity diagnosis.")
    out.append(render("Navier–Stokes representation gate on retained heads.",
                      ["Setting", r"$K$", "Bank rank", "Best-found median (%)", "POD-K median (%)", "POD/head error", "Pass ≥2"],
                      [[r[0], str(cfg["K"]), r[1], r[6], r[7], r[8], r[9]] for cfg, r in zip(ns_configs, ns, strict=True)], ["T11e_ns"],
                      "The unqualified mesh row is the original baseline; other rows identify changes to latent size, family or data. "
                      "These are best-found representation errors, not deployed-trajectory errors. Original failed bank attempts are excluded and retained in the retraction history. "
                      "The head advantage threshold was unchanged; FOM certification depends on separately disclosed amended gates."))

    _, nsl = table("T11g_ns_ladder")
    out.append("An exploratory ladder was run after the head gate failed. It reduces median trajectory error with correction rank, "
               "but the worst-case error increases at the first correction rung. POD-LSPG is more accurate at every matched dimension and cheaper at all but the largest one. "
               "No neural point is nondominated. The matched solve/manifold ratio is used below; the earlier comparison mixed different aggregations and was withdrawn.")
    out.append(render("Exploratory Navier–Stokes correction ladder after the failed representation gate.",
                      [r"$q$", r"Matched dimension $K+q$", "Neural worst (%)", "Neural median (%)", "Neural ms", "Solved/manifold", "POD worst (%)", "POD ms"],
                      pick(nsl, [0, 1, 2, 3, 4, 8, 10, 12]), ["T11g_ns_ladder"],
                      "Worst and median refer to the per-case maximum over evolved times. Solved/manifold uses the median-over-cases worst-evolved statistic on both sides. "
                      "All times are from one job. This is exploratory, not the confirmatory phase on a passing head. "
                      "Zero evolution budget exits in the source do not imply that every initial-state fit converged."))

    _, lv = table("T20_lowvisc_ladder")
    _, lvc = table("T20b_lowvisc_panel")
    lv = [r for r in lv if "M=1088" in r[0]]
    podrows = [r for r in lvc if r[0].startswith("POD")]
    bestpod = min(podrows, key=lambda r: float(r[1]))
    fom = next(r for r in lvc if "nt1e-2_dt01" in r[0])
    out.append("Lower viscosity increases the representation difficulty of the linear models more than that of the nonlinear head. "
               f"The fixed-test ladder spans {n('nLvLadderErrSpan')} in evolved error for {n('nLvLadderCostSpan')} cost, "
               "below the registered accuracy-span bar. Every neural rung is more accurate than the tested POD-LSPG ranks, yet a tuned full-order setting dominates all reduced subjects. "
               f"The converged discrete reference itself has {n('nLvDiscTight')}% error against the fine reference. "
               "Consequently these are under-resolved, same-discretisation comparisons; they do not establish a corresponding continuum-accuracy advantage.")
    out.append(render("Under-resolved low-viscosity Burgers: fixed-test ladder and controls.",
                      ["Method", "Worst evolved (%)", "Worst all times (%)", "Fine-reference error (%)", "Device ms"],
                      pick(lv + [bestpod, fom], [0, 1, 2, 3, 5]), ["T20_lowvisc_ladder", "T20b_lowvisc_panel"],
                      "The POD row is the most accurate tested POD-LSPG subject on evolved error; the FOM row is the cheapest full-order setting dominating all reduced subjects. "
                      "No resolved confirmation panel was run. These conclusions remain provisional pending the independent report audit and resolution confirmation."))

    _, meshrows = table("T10_mesh_ladder")
    out.append(f"At a frozen checkpoint, the Burgers cached solve changes from {n('nMeshBurgersCachedFirst')} to {n('nMeshBurgersCachedLast')} ms "
               f"while the state size increases {n('nMeshBurgersUnknownGrowth')}-fold. Complete-query time increases by ${n('nMeshBurgersCompleteRatio')}\\times$. "
               "This separates mesh-independent reduced work from dense input/output. The inherited configuration does not beat its cheapest full-order comparator; "
               "the later corrected panel above is a different experiment.")
    out.append(render("Frozen-checkpoint mesh transfer.",
                      ["PDE", "Intervals", "Unknowns", "Cached ms", "Device ms", "Complete-query ms", "Worst reference error (%)"],
                      pick(meshrows, [0, 1, 2, 3, 4, 5, 6]), ["T10_mesh_ladder"],
                      "One job per PDE with no retraining across meshes. Cached work excludes dense query input/output; all three cost definitions are preserved."))

    headers, speed = table("T15_speed")
    out.append("Fusing residual and Jacobian work and folding the head output layer into the bank reduces implementation cost while preserving the solved fields. "
               "The measured improvement is below the intended factor-of-two target; kernel optimization alone does not change the representation floor.")
    out.append(render("Implementation speed at numerical parity.", headers, speed, ["T15_speed"],
                      "Speedups use paired timings from each attempt. Parity is a field-disagreement diagnostic, not a comparison with the physical reference. The arm identifiers are source implementation labels."))

    headers, knobs = table("T08_solver_knobs")
    out.append("Stopping tolerances primarily reduce work after the representation error has saturated. Aggressive iteration caps instead cause early termination and large errors. "
               "These observations motivate correction rank as the accuracy control and solver settings as secondary cost controls.")
    out.append(render("Solver controls on the inherited held-out Burgers cohort.", headers, knobs, ["T08_solver_knobs"],
                      "This earlier cohort and reference differ from the main panel. Rows are reported as measured; the loose-tolerance reduced rows do not automatically satisfy the corrected main-panel eligibility rule."))

    headers, train = table("T16_training")
    out.append("The training sweep varies data volume, latent dimension, reconstruction objectives and smoothness terms. "
               "None of these retrains improves on the incumbent's best-found representation error. The nominal recipe reproduction is itself worse than the incumbent, "
               "so this experiment does not establish a general absence of benefit from additional data or model capacity.")
    out.append(render("Burgers head-training study.", headers, train, ["T16_training"],
                      "All query costs are from the evaluation allocation. Bank floor and best-found values describe representation; solved columns describe deployed queries. "
                      "Training schedules do not reproduce the incumbent exactly, limiting causal interpretation. The withdrawn wider joint-bank arm is not included."))

    out += ["The experiments establish a correction-rank family and a strong compact-representation advantage on the tested Burgers family. "
            "They support a narrower computational conclusion: some reduced operating points are useful at the largest Burgers mesh and on the nonseparable Poisson domain, "
            "while linear reduced models or tuned full-order solvers are often preferable. No operator-wide superiority, general nonlinear-manifold speedup, "
            "or successful confirmatory Navier–Stokes ladder follows from these measurements.",
            "**Provenance and reporting status.** All empirical numbers and table cells above are selected from the committed paper's generated tables and number JSON, "
            f"at `{PAPER_COMMIT}`. The paper generator reads lane summary, analysis and audit JSONs and generated reports derived from run artifacts. "
            f"The [extract manifest]({STEM}.manifest.json) preserves every input hash, table mapping and the paper's upstream provenance. "
            "This extract changes presentation only; it is not a new raw-field audit. Independent report audits remain pending for no-second, p-linear, w-ladder and b-lowvisc; "
            "open dispositions in other lanes are stated beside the affected tables. The corrected Burgers admissibility rule and NS matched-statistic comparison are included. "
            "No proposed follow-up experiment is presented as measured evidence.",
            "**Glossary.**",
            "- **FOM / ROM / NM-ROM:** full-order numerical model; reduced-order model; reduced model whose trial states are constrained by a nonlinear decoder.",
            "- **Bank / head / bank floor / best-found:** the learned spatial basis; the small nonlinear map into its coefficients; best linear projection error in that basis; lowest reconstruction error found by the recorded multistart fitting procedure. Best-found is an upper bound, not a certified global optimum.",
            r"- **$K$, dimension, $R$, $q$, $M$, $m$:** head latent dimension; number of solved coordinates; spatial-bank rank; number of added correction coefficients; weak-test count; quadrature-node count. The coupled solve has dimension $K+q$. At $q=R$, the reachable states span the full bank.",
            "- **Rung / ladder / fixed-test / scheduled-test:** one correction-rank setting; a collection of settings; constant weak-test count; weak-test count increased with the number of unknowns.",
            r"- **Worst / median / all times / evolved / initial / vs ref / fine-reference error:** maximum or median under the caption's stated aggregation; including the initial state; excluding it; the initial-state error alone; error against a finer numerical reference. $E_{\mathrm{evol}}$, $E_{\mathrm{all}}$, $E_0$ and $E_{\mathrm{ref}}$ denote those Burgers errors.",
            "- **Same-grid / physical current-relative / energy-state:** comparison with the converged solution of the same discrete equations; heat error normalized by the physical reference at the evaluated time; the wave error norm combining displacement and velocity. These norms are not interchangeable.",
            "- **Development / validation / sealed / checkpoint / seed / SD:** cases used during design; cases used to select trained models; cases opened only after choices were frozen; saved learned parameters; random initialization or sampling identifier; sample standard deviation across retrained checkpoints.",
            "- **Admissible / stationary / converged / valid / conv. / early-stopped / budget exit:** accepted under the specified accuracy and solver gates; meeting the gradient criterion; meeting an accepted stopping rule; count of accepted invocations; abbreviation for convergence; stopping without convergence; reaching an iteration limit. Evolution and initial-state exits are distinct.",
            "- **Nondominated / frontier / full criterion / pass:** no accepted competitor is both cheaper and more accurate; the set of such methods; the stated conjunction of ladder requirements; whether that conjunction or a named gate is met. Frontier membership is defined relative to the measured candidate set and metric.",
            "- **Device ms / cached ms / complete-query ms / incl. host / GPU speedup / cost span:** resident computation time; reduced computation with dense input/output excluded; host-to-host query time; speed ratio including host overhead; old/new device time within an attempt; largest/smallest cost in one ladder. A millisecond is one thousandth of a second.",
            r"- **$T_{\mathrm{ROM,min}}/T_{\mathrm{FOM,min}}$, FOM/head cost, error span, POD/head, solved/manifold, parity:** cheapest reduced/full-order time ratio; full-order/head time ratio; largest/smallest error in the stated ladder; POD representation error divided by best-found head error; matched solved/best-found error ratio; relative disagreement between implementation outputs.",
            r"- **EQ / NNLS / $\rho_{\max}$ / confirmed / marginal / single-draw:** empirical quadrature; nonnegative least-squares fitting of its weights; maximum held-out relative error of its projected nonlinear term; every recorded redraw passed the named bar; some recorded draws failed; only one draw measured. A draw is one sampled rule-construction dataset. A missing table entry means not supplied or not measured, not zero.",
            "- **POD / POD-LSPG / linear map / quadratic map / QR / DST / CG / LM:** proper orthogonal decomposition, a linear basis fitted to training states; least-squares Petrov–Galerkin evolution in that basis; the fitted linear or quadratic coefficient map; orthogonal matrix factorization for a direct reduced solve; discrete sine transform; conjugate gradient; Levenberg–Marquardt nonlinear least squares.",
            "- **U-Net / FNO / Transolver / still improving:** convolutional encoder-decoder operator; Fourier neural operator; attention-based neural operator; validation improvement was continuing at the training limit. Operator architectures and capacities were selected under the original protocol, not reselected for this extract.",
            "- **Signed-distance bank / sparse direct / SuperLU / under-resolved:** basis with a boundary-vanishing distance factor; solution by sparse matrix factorization; the sparse-direct implementation; mesh error large enough to prevent a continuum-accuracy conclusion.",
            "- **Attempt / arm / L4 / C1 / incumbent / trajectory term / weak term / code smoothness / joint bank+head:** one run directory; one tested configuration; recorded implementation variants; original checkpoint; a temporal training objective; projected-residual training term; penalty on variation of latent codes; simultaneous training of both model parts. Their numerical outcomes retain the source configuration labels.",
            "- **Job / allocation / GPU / A100 / H200 / f64 / commit / SHA256:** scheduler execution identifier; assigned compute resources; graphics processor; processor models; double precision; immutable source revision; content hash used to identify an input."]

    target = ROOT / "reports" / f"{STEM}.md"
    target.write_text("\n\n".join(out) + "\n")
    manifest = {"paper_commit": PAPER_COMMIT, "generator": str(Path(__file__).relative_to(ROOT)),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "inputs": INPUTS, "tables": TABLES, "upstream_provenance": UPSTREAM,
                "scope": "Presentation extract; inherited run-derived values, no new raw-field audit."}
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {target} ({len(TABLES)} tables, {sum(t['rows'] for t in TABLES)} rows).")


if __name__ == "__main__":
    main()
