"""Create a session handoff from accepted campaign data and artifact inventories."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/2026-09-11-accuracy-campaign-handoff.md"
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = ROOT / relative
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text())


def link(path, label):
    return f"[{label}]({ROOT / path})"


def git(tree, *args):
    return subprocess.check_output(["git", "-C", str(ROOT / tree), *args], text=True).strip()


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def main():
    data = read("reports/2026-09-11-accuracy-improvements-and-wave-speed.json")
    integration = read("reports/2026-09-11-accuracy-integration.json")
    owners = {key: read(integration[key]["inventory"]) for key in ("poisson", "burgers", "wave")}
    for key, owner in owners.items():
        assert SOURCES[integration[key]["inventory"]] == integration[key]["inventory_sha256"]
    trees = {key: f"worktrees/2026-09-07-mr-{slug}" for key, slug in
             [("Poisson", "poisson2d"), ("Heat", "heat2d"), ("Burgers", "burgers2d"), ("Reflective waves", "wave2d")]}
    tree_state = {key: dict(path=path, commit=git(path, "rev-parse", "HEAD"),
                           branch=git(path, "branch", "--show-current"), dirty=bool(git(path, "status", "--porcelain")))
                  for key, path in trees.items()}
    assert not any(t["dirty"] for t in tree_state.values()), "Record changed worktree state before regenerating a clean handoff."
    report_commit = git(".", "log", "-1", "--format=%H", "--", "reports/2026-09-11-accuracy-improvements-and-wave-speed.json")
    decisions = {row["pde"]: row for row in data["decisions"]}
    p, h, b, w = (decisions[name] for name in trees)
    pgroup = next(g for g in data["poisson_correction"]["groups"] if g["group"] == "all")
    pmesh = next(m for m in pgroup["meshes"] if m["intervals"] == p["intervals"])
    poriginal, pprimary = pmesh["methods"]["original_relative"], pmesh["methods"]["r128_q32"]
    wave_fine = {r["method"]: r for r in data["wave_confirmation"]["combined_panels"] if r["intervals"] == w["intervals"]}
    wave_gain = wave_fine["baseline"]["median_gpu_ms"] / w["gpu_ms"]
    fine_b = {r["name"]: r for r in data["burgers_coverage"]["rows"] if r["intervals"] == b["intervals"] and r["cohort"] == "all"}
    cfg_p = data["poisson_correction"]["config"]
    prefixes = sorted(set(cfg_p["correction_counts"].values()) - {0})
    heat_checkpoint = next(f for f in integration["heat"]["files"] if f["path"].endswith("initial_tail.pkl"))
    p_run = owners["poisson"]["runs"][-1]
    checkpoints = [dict(pde="Heat", role="Selected head", **heat_checkpoint)]
    checkpoints += [dict(pde="Poisson", role=c["id"], path=str(Path(trees["Poisson"])/c["path"]), sha256=c["sha256"])
                    for c in p_run["checkpoints"]]
    basis = owners["poisson"]["correction_basis"]
    checkpoints.append(dict(pde="Poisson", role="Frozen correction basis", path=str(Path(trees["Poisson"])/basis["path"]), sha256=basis["sha256"]))
    bc = owners["burgers"]["accepted_checkpoint"]
    checkpoints.append(dict(pde="Burgers", role="Original retained head", path=str(Path(trees["Burgers"])/bc["path"]), sha256=bc["sha256"]))
    for c in owners["wave"]["files"]["frozen_selected_checkpoint"]:
        if c["path"].endswith(".npz"):
            checkpoints.append(dict(pde="Reflective waves", role=c["role"], path=str(Path(trees["Reflective waves"])/c["path"]), sha256=c["sha256"]))
    for c in checkpoints:
        assert digest(ROOT/c["path"]) == c["sha256"], c["path"]
    lines = ["# Handoff: accuracy campaign and the next NMROM session", "",
             "Prepared September 13 from the finalized, independently audited September 11 development results. This is a handoff snapshot; the canonical lab log remains authoritative for subsequent changes.", "",
             "## Read first", "",
             link("LAB-LOG.md", "Canonical LAB-LOG.md") + " must be read first, followed by " + link("AGENTS.md", "repository operating rules") + ". Read their current versions even when resuming from this document.", "",
             f"Repository: `{ROOT}`. The user is developing the current separable nonlinear-manifold ROM for accuracy and speed across Poisson, heat, Burgers and reflective waves, including transfer across output resolutions and fixed-weight online tuning. The older ViT + CP model is outside this comparison. Absorbing waves are excluded.", "",
             "## What is complete, and what is not", "",
             f"The bounded experiment round, raw-field audits, timing aggregation and source-generated report are complete. The report is committed on main at `{report_commit}`. All experiment source, selected checkpoints, failed alternatives and restorable result archives are retained in the separate PDE worktrees. Cluster run directories were checksum-collected and removed; the queue was empty at campaign closure. Recheck the queue before new work.", "",
             "No consolidated worktree or merge has been created. The latest user request was to write this handoff; it did not approve the previously proposed consolidation. Existing PDF tables and collaborator slides still contain the previous round, while the campaign Markdown/JSON contain the updated results. Final paper cohorts remain sealed. These development results do not establish final publication accuracy or broad PDE-family generalization.", "",
             "The canonical log has local appended entries. The main checkout also has pre-existing changes under `understand/` and unrelated report/build files. Preserve them; do not reset the checkout or stage the whole tree.", "",
             "## Exact worktrees to resume from", ""]
    lines += table(["PDE", "Worktree", "Branch", "Verified clean HEAD", "Retain"],
                   [[name, link(t["path"], Path(t["path"]).name), f"`{t['branch']}`", f"`{t['commit']}`", decisions[name]["choice"]]
                    for name, t in tree_state.items()])
    lines += ["Main carries the combined reports, but its original heat rollout is a frozen, known-broken behavioral baseline. Do not start a new experiment from that implementation merely because it is on main. Each PDE worktree includes its corrected dependencies; the main report alone is not an integrated solver distribution.", "",
              "## Accepted results at the largest mesh", "",
              f"Each row uses {p['intervals']} intervals per spatial axis and the stated full development cohort. Times are pooled medians of complete GPU queries. Each FOM comparator is the fastest tested passing iterative setting from the same job, mesh and cohort. The nominal physical error target is {100*cfg_p['development_target']:g}%, with the recorded reference and numerical checks also required.", ""]
    lines += table(["PDE (cases)", "Relative-error normalization", "Before → retained worst error %", "ROM ms", "Iterative FOM ms", "FOM / ROM", "ROM target"],
                   [[f"{x['pde']} ({x['cases']})", x["norm"], f"{x['before_percent']:.6f} → {x['after_percent']:.6f}",
                     f"{x['gpu_ms']:.6f}", f"{x['fom_ms']:.6f}", f"{x['iterative_fom_over_rom']:.6f}×", "Pass" if x["target_pass"] else "Miss"]
                    for x in data["decisions"]])
    lines += ["Poisson and reflective-wave ratios describe runtime only because their ROMs miss the physical target. Direct DST remains faster for these linear PDEs. Errors with different normalizations cannot be ranked across PDEs. Burgers before/after errors come from the stopping comparison; the displayed timing pair is from the later job, whose original-head fields were checked against that comparison.", "",
              link("reports/2026-09-11-accuracy-improvements-and-wave-speed.md", "Full campaign report: all resolutions and rejected alternatives") + ". Its adjacent JSON retains repetition arrays, individual panels and source hashes.", "",
              "## Architecture and findings to carry forward", "",
              r"The shared decoder structure is $u(X;z)=G(X)h(z)$: a learned spatial bank $G$ and a nonlinear coefficient head $h$. The query supplies physical fields or forcing; solving for latent coordinates remains an online task. Offline operator assembly, caches and compilation are distinct from charged complete-query work. Keep each PDE's audited weak-form operator and input contract.", "",
              "**Poisson.** Fixed-capacity staged training and bank expansion alone failed to improve complete online accuracy. The expanded bank became more expressive, but its original nonlinear head did not use that capacity well. The accepted accuracy-improving research candidate adds training-residual correction directions:", "",
              r"$$D_q(z,y)=G\bigl(h(z)+C_qy\bigr).$$", "",
              r"Here $X$ denotes spatial evaluation points, $z$ the nonlinear latent coordinates, $C_q$ the first $q$ frozen correction directions, and $y$ their linear coefficients. The spatial bank $G$ is evaluated on the requested mesh.", "",
              f"The frozen basis supports prepared prefixes {', '.join(map(str, prefixes))}; `{data['poisson_correction']['primary_model']}` was declared primary before evaluation. The nonlinear optimizer still has {pprimary['nonlinear_optimizer_dimension']} variables, with {pprimary['nominal_latent_dimension']} total coordinates for the primary. Linear coefficients are eliminated exactly by QR projection and recovered after solving for the nonlinear coordinates. Both full and projected stationarity, numerical ranks, initialization and decoding were audited. Source-family descriptors and evaluation truth are not online inputs.", "",
              f"On the finest mesh, target-failing cases drop from {poriginal['failed_target_cases']} to {pprimary['failed_target_cases']} out of {p['cases']}, but primary GPU time increases by {100*(pprimary['gpu_median_ms']/poriginal['gpu_median_ms']-1):.6f}%. All cases in this family were already opened during development. Prefix switching requires offline preparation of each projected operator/cache and compiled solver; it requires no neural retraining. It does not imply zero setup cost for arbitrary unprepared prefixes.", "",
              "**Heat.** Keep `nmrom_initial_tail`. Training emphasizes initial fields and difficult reconstruction examples while retaining the spatial bank and online solver. It improves development accuracy. Its small GPU timing change does not imply an improvement including host transfers. The unrestricted linear-bank control is a different reduced model and remains a useful separate comparison.", "",
              f"**Burgers.** Keep the original checkpoint and the strict stationary solver. The two retrained strict heads reach {100*fine_b['trained576_stationary']['worst_fixed_initial']:.6f}% and {100*fine_b['trained4608_stationary']['worst_fixed_initial']:.6f}% worst trajectory error, both worse than the original's {b['after_percent']:.6f}%. Improving initial reconstruction did not preserve trajectory quality. Keep the FOM-exact upwind term and decoder-output-based fitted quadrature. Because the head, initial-guess library and quadrature weights changed together, these runs do not uniquely identify the source of the rollout degradation.", "",
              f"**Reflective waves.** Shared analytic decoder derivatives and guarded Cholesky solves remove repeated geometry work. The unchanged-step arm has numerical trajectory parity; a larger integration step is a separate change with refinement checks. The final nested decoder preserves the phase-trained parent and adds frozen linear training directions. Its final runtime is {wave_gain:.6f}× faster than the original ROM in the same job. Its pooled energy-state error still misses the target even though the newly introduced development cases pass. Initial-scaled displacement/velocity, energy-state error, current-relative errors and energy-conservation drift are different metrics. Keep the documented initializer limitation and the frozen checkpoint/step; do not reinterpret the correction decoder as the independently retrained larger head.", "",
              "All pre-reset wave experiments remain historical and untrusted under the user's evidence reset. Rejected time steps, unsuccessful larger-head training, failed instrumentation and pre-query attempts remain archived. No accepted numerical result was retracted by this round.", "",
              "## Checkpoints and reproducibility", "",
              link("reports/2026-09-11-accuracy-integration.json", "Combined integration inventory") + " links the source-level owner inventories, archive manifests, retained methods and failed alternatives. Checkpoint paths below are exact; complete SHA256 values are also retained in this handoff's manifest.", ""]
    lines += table(["PDE", "Artifact", "Path", "SHA256"], [[c["pde"], c["role"], link(c["path"], Path(c["path"]).name), f"`{c['sha256']}`"] for c in checkpoints])
    lines += ["The wave decoder also requires its learned spatial bank. Use the owner inventory's `frozen_bank_and_original_head_lineage` for the reflective Dirichlet bank parameters/tables and original training lineage. Copying only the new head file is insufficient. For every PDE, keep the appropriate initializer, projected operators/configuration and referenced source dependencies together.", "",
              "The owner inventories contain restoration procedures and manifests for full predictions, references and timing repetitions. Use the native audits plus the main coordinator audits if integrating code; do not rerun the entire scientific search merely to recover already saved results.", "",
              "To regenerate the accepted campaign report from its retained artifacts:", "", "```bash",
              f"cd {ROOT}",
              "OPENBLAS_NUM_THREADS=1 /home/tahmid/Dev/.venv/bin/python reports/generate_accuracy_campaign.py",
              "```", "",
              "## Proposed starting point for the next session", "",
              "Proposed worktree: `worktrees/2026-09-13-nmrom-consolidated`; proposed branch: `exp/2026-09-13-nmrom-consolidated`.", "",
              f"Recommended base: the corrected heat branch at `{tree_state['Heat']['commit']}`. This preserves the heat rollout corrections. Bring in the selected Poisson, Burgers and reflective-wave implementations and checkpoints from the inventories, plus the current main reports. This is a proposal, not an existing worktree or an approved merge.", "",
              "The repository rules require asking the user to confirm the base and worktree creation before executing that step. Earlier experiment authorization does not resolve the explicitly pending consolidation decision. The handoff itself adds no new approval requirement.", "",
              "Once the user chooses the next scope, a useful sequence is:", "",
              "1. Read the canonical log and check current branch heads, dirty files and any newer work. Confirm the proposed consolidation base/name if consolidation is requested.",
              "2. Integrate only within the chosen new worktree, preserving the original trees and the distinction between selected methods and failed research evidence. Resolve all source/checkpoint/config dependencies before presenting it as runnable.",
              "3. Add a concise entrypoint map and verify checkpoint hashes, imports, bounded smoke execution, and unchanged outputs against retained results. Check executable defaults explicitly; an experiment being archived does not make it the default solver.",
              "4. If requested, regenerate PDF tables/slides from the new campaign JSON. Preserve the user's presentation preference to omit FOM-error and combined-gate columns, while stating missed ROM accuracy targets clearly.",
              "5. For new experiments, predeclare the changed mechanism, controls, cohorts, timing contract and acceptance criteria before evaluation. Candidate directions are improved Poisson correction coverage, trajectory-aware Burgers training with an isolated quadrature-fidelity comparison, and wave displacement-gradient/velocity accuracy. Keep final cohorts sealed until the protocol is ready for final validation.", "",
              "## Runtime and measurement rules", "",
              "The full AGENTS.md rules apply. Locally, use `/home/tahmid/Dev/.venv/bin/python`; on the cluster, use `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python`. Local GPU jobs are bounded smoke tests through `jaxrun`; real experiments use the cluster GPU partition, a private job directory and the mandatory GPU-backend preflight.", "",
              "Require double precision and `JAX_DEFAULT_MATMUL_PRECISION=highest`. Burn in before timing, pair error and cost from the same solver call, preserve repetition arrays, and compare runtime within one GPU job. Retain full fields, seeds, configuration and provenance. Regenerate data from seed on the cluster; verify checksums/restoration before exact remote cleanup. Do not overwrite frozen archives, use bare `scancel`, or launch unapproved additional worktrees. Consult AGENTS.md for the exact concurrency, quadrature and cancellation rules.", "",
              "## Paste into the new session", "", "```text",
              f"Read {ROOT}/LAB-LOG.md first, then AGENTS.md and",
              str(OUT) + ".",
              "The accuracy campaign is complete. The selected code and checkpoints remain",
              "in four separate PDE worktrees; no consolidated worktree or merge exists.",
              "The proposed corrected-heat base and new worktree still need confirmation.",
              "Use the accepted report and integration inventory; keep absorbing waves",
              "excluded and final paper cohorts sealed. Check whether I have supplied",
              "new authorization or a different next task before acting on this snapshot.",
              "```", "",
              "## Plain-language glossary", "",
              "- **Worktree / branch / HEAD:** a separate checked-out directory / its recorded development history / the exact latest commit in that checkout.",
              "- **Bank / head / latent coordinates:** spatial functions / the network choosing their coefficients / small variables fitted during a query.",
              "- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained by a nonlinear decoder.",
              "- **Intervals:** spatial subdivisions along each axis; this is not the total count of grid points.",
              "- **Relative L2 error:** field-error magnitude divided by the stated reference magnitude. Worst means the maximum over the recorded cases and relevant output times.",
              "- **Energy-state error:** combined velocity and displacement-gradient error; it is different from energy-conservation drift.",
              "- **Median GPU ms / FOM over ROM:** middle retained GPU query duration in milliseconds / a timing ratio above one when the ROM is faster.",
              "- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. Burgers uses an iterative nonlinear solver with an FFT preconditioner, not a direct nonlinear solve.",
              "- **Weak form / quadrature / EQ:** residual tested against smooth functions / approximating integrals by weighted samples / fitted empirical quadrature weights.",
              "- **Stationarity / parity / refinement:** sufficiently small objective gradient / agreement with the unchanged method / agreement after refining a numerical discretization.",
              "- **QR / Cholesky:** matrix factorizations used to project or solve small linear systems.",
              "- **Correction prefix / tail emphasis / phase training:** initial columns of a fixed correction basis / giving harder training examples more influence / training displacement and velocity-related quantities together.",
              "- **Checkpoint / initializer / cache:** saved trained parameters / a procedure or model choosing the first solver guess / precomputed values reused by queries.",
              "- **Development / sealed final / target:** cases used for diagnosis and method choice / untouched cases reserved for later final evaluation / the predeclared physical and numerical acceptance criteria.",
              "- **SHA256 / manifest / archive:** a content fingerprint / a record of artifact paths and fingerprints / a retained package that can restore complete result files.", ""]
    OUT.write_text("\n".join(lines))
    manifest = dict(prepared_date="2026-09-13", numbers_finalized="2026-09-11", report_commit=report_commit,
                    canonical_log=str(ROOT/"LAB-LOG.md"), snapshot_only=True, merge_performed=False,
                    tree_state=tree_state, checkpoints=checkpoints, sources=SOURCES,
                    generator_sha256=digest(Path(__file__)), document_sha256=digest(OUT))
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(OUT)


if __name__ == "__main__":
    main()
