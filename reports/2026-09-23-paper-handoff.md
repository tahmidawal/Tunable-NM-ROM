# Paper handoff — ICLR 2027 Tunable NM-ROM (state at 2026-09-23 evening)

This is where the manuscript stands for a new session that continues the editing and writing. The numbers in the paper are final unless this file marks them as pending. Deadline: **2026-09-25 AOE**.

## Files

- **Manuscript:** `paper_latex/main.tex` is the only source; it is a single flattened file. `paper/` was removed (it is recoverable from commit f3230029).
- **Build:** run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in `paper_latex/`. It gives 22 pages, with no undefined references or overfull boxes.
- **Overleaf zip:** `tunable-nmrom-iclr2027-overleaf.zip` at the repo root, holding `main.tex`, `iclr2027_conference.sty`, `.bst`, `fancyhdr.sty` and `natbib.sty`. Rebuild it after every change:
  ```
  rm -f tunable-nmrom-iclr2027-overleaf.zip && (cd paper_latex && zip -q ../tunable-nmrom-iclr2027-overleaf.zip main.tex iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty)
  ```
- **Git:** `paper_latex/main.tex` has uncommitted edits since 7d662f3b. Commit only when the user asks, and don't stage `LAB-LOG.md`, `reports/`, `understand/` or the zip in paper commits.
- **Lab log:** append to `LAB-LOG.md` with `flock .lablog.lock`, because several agents write to it.

## Focus, settled with the user

Focus A + C:
1. **Lead with deployment-time tunability.** One frozen model, an importance-ordered bank of width R′, and a choice of head or span, with no retraining.
2. **Then the nonlinear PDEs at scale:**
   - Burgers 2D up to 4096²: 0.22 % at 8.78×, or 1.9 % at 21.7×.
   - Navier–Stokes 3D at 96³: head 0.15 % at 6.67×.
3. **Then the linear case in one sentence:** on Poisson and heat the ordered span is the best setting.

What else is settled:
- **Framing:**
  - The title stays.
  - The abstract, intro, contributions and conclusion were rewritten in this order today.
  - There is **no "what we don't claim" text**; the user will think about limitations separately.
  - "Train once, deploy on any mesh" was **rejected** as the focus.
- **Reviewer history:** the NeurIPS reviews are in `Older Paper /reviewer_comments `. They explain the framing choices.

## User style rules (must follow)

- **Notation:** §3 uses the old NeurIPS paper's notation and style: u ∈ ℝᴺ, K, F, R(u), ũ(z) = G h(z), J_D, W_dir, N_eq, w, S, S⁺, NNLS. The span is `ũ = Ĝ_{R′} c`. The tests P are the one extra symbol. Add no new ad-hoc symbols.
- **§3.2 shape:** each PDE paragraph gives its residual equation, what is linear or nonlinear, and what gets updated, pointing to eq. `step` and eq. `lm`. The NS paragraph follows that same shape. Its content was verified claim by claim against the NS code; the moving centre is mentioned in one plain sentence.
- **Tables:**
  - Use **no superscript markers of any kind** and **no internal jargon** such as "dev. sources", arm names or q.
  - Row labels are plain problem names.
  - Caveats go in plain words in the appendix, e.g. the status column "development" / "sealed held-out" / "timing provisional".
- **Table 1:** has no FOM-solver column. The caption names the solvers in one sentence. Table 1 order stays.
- **Operator names:** name each operator explicitly; never write "best operator".
- **Writing voice:** it should read like the user's writing, not AI prose.
- **Excluded from the paper (do not include or mention):**
  - C_q correction directions;
  - the spectral / fast-transform FOM comparison (archived, branch `exp/2026-09-23-spectral-fom`);
  - the heat POD comparison, where POD-128 beats our span (the user will check it later).
- **Numbers:** take them only from pinned run records. Ratios may only divide times from the same job. Never type a number without a source.

## Pending work

**In the paper:**
1. **Red `\todo` markers.**
   - (a) §6.1: "64 held-out Burgers cases with these settings: run queued".
   - (b) Done (evening): Table 2 now has Burgers 2048² (fast 2.37 % at 4.55×, bold) and NS 3D 32³/64³ rows (64³ accurate 0.15 % at 3.05×, bold; at 32³ the FNO, 0.40 % at 2.16×, is faster than our accurate setting). There is no 4096² cell, because the operators can't be trained at that mesh.
2. **`\gen{$1024^2$ gain}` placeholder** in §6.3, "Which Knob to Turn", and in the appendix knob table: the solver-path gain on Burgers. It needs a same-job number, or the sentence should be removed.
3. **Page limit:** the main text now ends on p11 (the Conclusion and the references both start on p11). It must end by p9. This is not done yet; the user has not said what to cut.
4. **Conclusion caveat:** the conclusion still says "The named solvers remain more accurate." The user may want it moved to the limitations.
5. **Linear-rows framing:** on Poisson and heat, both Table 1 settings are span solves with no iteration and no Gauss–Newton. The paper states this in §3.2 and §6. The user knows this.
6. **Possible additions if results land:**
   - an NS 3D row in Table 2;
   - a Burgers 3D row in Table 1.

**Running experiments** (each has its own agent, worktree and cluster namespace; they report to the session that launched them, and results land in the lab log and each lane's report):

| lane / job | what | goes to |
|---|---|---|
| `worktrees/2026-09-23-burgers-bank-knob`, job `bbk_bkh64b` (4218386) | Burgers 2D on 64 sealed held-out cases with the Table-1 settings | replaces todo (a) in §6.1 |
| `worktrees/2026-09-23-burgers2d-speed` (new) | Burgers 2D: make the frozen model faster at 256²/512²/1024² with the same settings rule; results by 2026-09-24 18:00 EDT | possibly better Burgers rows in Table 1; any new setting also needs its held-out run |
| `worktrees/2026-09-23-burgers3d-span` (new) | Burgers 3D with the current method at 32³/64³(/128³), pre-registered; stopping rule 2026-09-24 20:00 EDT | possible Burgers 3D row in Table 1, only if it passes |
| `worktrees/2026-09-23-ns3d-operators` | 32³/64³ are done and in Table 2; a 96³ stretch run (32 held-out cases) is running | possible NS 96³ row in Table 2 |

- **Disk incident:** a shared-disk overflow (10:30–11:10 EDT today) may have hit earlier attempts of the two Burgers jobs. Both agents were asked to confirm that no reported number comes from that window.

**How to fill a result:**
- Read the lane's report and summary JSON, and check its sha256.
- Use the same arm as the neighbouring rows. For example, the heat Table 2 rows use `linear_bank_moments_cn`, the span R′ = 128.
- Recompute every speedup against the cell's one full-order solver time from the same job.
- Rebuild, refresh the zip, and append to the lab log.

## Where the numbers come from (for checking)

- **Burgers Table 1:** `worktrees/2026-09-23-burgers-bank-knob/.../checks/bk{256..4096}-summary.json`. The fast speedup is taken against the accurate setting's FOM: 523.13 / 24.06 = 21.7× at 4096².
- **Heat 2D/3D Table 1:** `worktrees/2026-09-23-heat-bank-knob/experiments/heat-bank-knob/report.json`, sha256 bad554ec…
  - Rows use Crank–Nicolson stepping.
  - 32³–128³ and 4096² have provisional timing, because an order-effect gate failed on a setting that isn't reported.
- **Heat Table 2:** 1024² comes from job 4206383; 2048² from `worktrees/2026-09-23-heat-compare-hires/.../runs/pn2048d/summary.json`, sha256 1bf78c38… Accurate is 0.13 % at 289×. Operators: FNO 3.98 % at 0.93×, U-Net 2.76 % at 2.76×, Transolver 3.66 % at 40.5×, DeepONet 10.9 % at 5.11×.
- **NS:** `worktrees/2026-09-23-ns3d-shift-head/experiments/ns3d-shift-head/runs/*/output/summary.json`.
- **Poisson 2D, L-shape and Poisson 3D:** the `poisson-bank-knob` and `poisson-bank-knob-3d` lane summaries.
- **Iteration counts** (not in the paper): Poisson and heat span solves are direct, with no iterations. Burgers takes about 54 (accurate) or 51.5 (fast) LM iterations over 50 steps, flat across meshes. NS takes 10 steps × 3 Gauss–Newton sweeps.
- **Speedup scaling:** our cost grows like N·R′, and only through reading and writing fields. CG iteration counts grow with the mesh; for Poisson 2D, 319 → 5289 from 256² to 4096².

## Glossary

- **NM-ROM:** the nonlinear-manifold reduced-order model of the paper.
- **Bank G:** the frozen spatial functions; **Ĝ** is the same bank ordered by importance.
- **Head h(z):** the small network mapping the latent z to bank coefficients.
- **Span:** a linear solve in the first R′ ordered columns.
- **R′:** the deployment width of the ordered bank, i.e. the accuracy knob.
- **Accurate / fast:** the two settings per row. Accurate is the most accurate tested setting. Fast is the cheapest setting at least as accurate as the earlier default (for NS, the cheapest span below 5 %).
- **FOM:** the full-order solver: CG, CN–CG, Newton–BiCGStab or CNAB2.
- **EQ:** empirical quadrature (NNLS node rule) for Burgers advection.
- **Held-out / development:** sealed cases opened once, versus cases used during development.
- **Provisional timing:** the job's pre-registered timing check failed.
- **Todo / gen:** red placeholders in the PDF.
