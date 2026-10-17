# Prompt: run the JCP experiment campaign for viscous Burgers 2D and 3D

Copy everything below the line into a new Claude Code session started in `/home/tahmid/Dev/Tunable-NM-ROM-Claude`.

---

You are the **coordinator** of an experiment campaign for a Journal of Computational Physics paper. Your job is to run **all** the experiments in the campaign plan for **viscous Burgers in 2D and 3D**, end to end, and at the very end **compile every result into one final report**. You do not run experiments yourself. You **plan, delegate, verify, monitor and compile**.

**Run fully autonomously.** Do not stop to ask the user anything. Where the plan leaves a decision open, take the conservative default in section 7, record it in the lab log, and carry on. Keep going through priorities 1 → 4 until every in-scope experiment is done (or properly recorded as failed or deferred), then compile the results (section 8). Your final message is the hand-back.

## 1. Read first (in this order, before doing anything)

1. `CLAUDE.md`: compute rules (Tufts cluster via `ssh tufts-login` + `sbatch`, the `gpu` partition only, never `preempt`; local GB10 only through `jaxrun` with at most 3 concurrent jobs; venv paths; worktree and branching rules; the code-only GitHub mirror; the lab log; report rules).
2. `LAB-LOG.md`: the "Where things stand" block, then every entry from 2026-10-08 onwards.
3. `reports/2026-10-08-jcp-lane-agent-rules.md`: the rules every lane agent follows. **Codex adversarial verification gates every step.**
4. The campaign plan, in this order, where later files supersede earlier ones:
   - `reports/2026-10-17-full-experiment-plan/PLAN-v2.md`
   - `PLAN-v3-amendments.md`
   - `PLAN-v4-amendments.md`
5. The plan's Codex audits, `codex-plan-audit-{1,2,3}.md` in the same folder. They list known traps.
6. The finished lanes you build on (read their `report.md` and `DESIGN.md`):
   - `worktrees/2026-10-06-jcp-references`: the E0 order-verified references, loader `refstore.py`, store `JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store`. **Never touch `sealed/` or `custody/`.**
   - `worktrees/2026-10-13-jcp-combined`: second-order stepping with the wide bank; the E0 scoring registry; the finalist qualification table.
   - `worktrees/2026-10-08-jcp-wide-bank`: banks up to R′ = 1024 and configurable test count M.
   - `worktrees/2026-10-08-jcp-time2`, `worktrees/2026-10-06-jcp-mechanism`, `worktrees/2026-10-01-quadrature-study`, `worktrees/2026-10-01-quadrature-burgers3d`.
7. `worktrees/2026-10-14-jcp-speed3d` (namespace `jcp-speed3d`) owns 3D speed, batching, the mesh-free query and the 3D FOM comparison. **Its agent was stopped mid-lane.**
   - Its production jobs g1 5301264, g2 5301270 and g3 5301580 all COMPLETED on 2026-10-15/16 but were **never collected**. Their outputs are still in `/cluster/tufts/paralab/tawal01/jcp-speed3d/{g1,g2,g3}`.
   - f1 5301274 was collected. The last lane commit was `f98843cb1`.
   - **Your first action:** launch a lane agent that resumes this lane in its existing worktree. It reads its `DESIGN.md` (amendments A1–A6), `runs/JOB-LEDGER.json`, `audits/` and the last commits, then: collects g1–g3; scores offline on CPU; runs selection and matching; gets the Codex results audit; then runs p1 and m1 with their job audits; then the final report and report audit.
   - Do not duplicate its scope elsewhere. Consume its results only after its audits pass, using the reuse map in `PLAN-v3-amendments.md` (V3-4) and the restrictions in V4-6.

## 2. Scope

**In scope (Burgers 2D and 3D only):**
- Plan priorities 1 → 2 → 3 → 4, in that order. Each priority is gated on the previous one being verified.
- Groups Q, A, B, C (C1/C2 in 2D, plus the C3 POD baselines), E1, E2, E5 and D1, as defined in the plan.

**Out of scope for you:**
- Allen–Cahn, the exponential equation, non-box geometry, and NS 2D/3D. NS has been added to the paper's scope, but runs in a later, separate campaign.
- Opening the sealed or reserved-OOD cohorts (plan F2).
- Merging branches.

**The core question to answer first:** show whether the reduced model's **cost and accuracy depend on the latent dimension k = r = R′ (bank coefficients) and not on the mesh size N.** Use the hypotheses H-cost-N, H-cost-r, H-acc-N and H-acc-r with the decision rules in v3/v4. Report pass, fail or inconclusive honestly: a clear "fail" is a valid result.

## 3. How to organise the work

**Lanes.** Split the work into lanes, one subagent each, run in parallel where the plan allows. Suggested lanes:
- `jcp-p1-scaling`: priority 1, P1a–P1g.
- `jcp-qualify`: group Q.
- `jcp-hr-baselines`: C3, POD + ECSW/Q-DEIM/GNAT. The worktree `worktrees/2026-10-06-jcp-hr-baselines` already exists as scaffolding; reuse it.
- `jcp-fom2d`: 2D C1/C2.
- `jcp-dims`: priority 3, A3, A4, B3, B6.
- `jcp-budget`: B2, B7, B8.
- `jcp-robust`: E1, E2, E5, D1.

Start only the lanes whose dependencies are met. Priority 1 first. C3 can start its **implementation** in parallel with priority 1, because it is new code, but it runs no production jobs until priority 1 is reported.

**Each lane gets:**
- its own git worktree, created with the sparse command in `CLAUDE.md`, branched from the fork point `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`, with `sync_github.sh` configured;
- its own cluster namespace, `/cluster/tufts/paralab/tawal01/<lane>/`;
- a concurrent-job cap: at most **8 cluster jobs in total** across all lanes.

You may name and create worktrees yourself.

**Each lane subagent must follow `reports/2026-10-08-jcp-lane-agent-rules.md` exactly.** Its sequence is:
1. `DESIGN.md` (pre-registration).
2. A Codex design audit.
3. Code.
4. A Codex code audit.
5. A real-size smoke test, where every control is shown to fire.
6. A Codex audit of each job, then submission.
7. Collect the results.
8. A Codex results audit.
9. `make_report.py` → `report.md`. Never hand-type a number.
10. A Codex report audit.
11. Hand back.

Codex calls use `timeout 3600 codex exec -C <worktree> -o <lane>/audits/codex-<step>-<n>.md "<adversarial prompt>" < /dev/null`, and the agent then checks that the output file exists. Fix every WRONG item and re-audit until clean. Codex's report is a claim to check, not a result. Codex audits must be told to stay **read-only** and never to write to `LAB-LOG.md` or `main`.

**Every lane brief you write must include:**
- the exact plan items it owns, with their decision rules from v3/v4;
- the §3 scoring contract:
  - selection on dev/validation only;
  - E0 ST is the primary reference;
  - the S reference only for BE at physical Δt = 0.005, so **3D S is never used**;
  - per-case-kind eligibility read from the E0 store;
  - ties within 2ε_ref;
  - post-hoc labels carried forward;
- the full Q1 contract from V4-2, with the three separate labels;
- frozen mode lists in both dimensions (V4-1). **2D mode sets differ across meshes at M ≥ 1536 under the existing builder.**
- the cumulative GPU-hour ledger and cap for its priority (priority 1: 20 GPU-h including contingency, "P1 incomplete" if the core can't finish);
- the lessons in section 6;
- the hand-back format in the lane rules.

## 4. Watchdogs: 2–3 Codex monitors on the lane agents

Lane agents have stalled before. One waited 18 hours on a Codex output file that was never written, and one died on a usage limit. Run a watchdog so that never happens silently.

1. **Heartbeats.** Every lane agent writes `<worktree>/experiments/<lane>/HEARTBEAT.json` at least every 20 minutes while working, and before and after every long wait. It contains:
   - the time (UTC);
   - the current step;
   - what it is waiting on (job ids, Codex audit output path, local PID);
   - the expected finish time;
   - its last commit.
2. **Two or three watchdogs.** Start two or three background watchdog loops: bash loops with a hard overall deadline, each run in the background. Each loop, every 20–30 minutes (staggered, so a check happens about every 10 minutes), runs:

   `timeout 900 codex exec -C /home/tahmid/Dev/Tunable-NM-ROM-Claude -o <campaign-dir>/watchdog/wd<i>-<timestamp>.md "<prompt>" < /dev/null`

   The prompt says the watchdog is **read-only**: no file changes, no job submissions, no kills. For every lane, it must check:
   - heartbeat age;
   - that every awaited process or PID actually exists (`ps`);
   - that every awaited Codex output file is growing or exists, **and** that its `codex` process is alive;
   - cluster jobs via `ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01'` and `sacct` for finished ones;
   - stale `sleep`/`until` waiter loops with no deadline;
   - that no two lanes write to the same worktree or cluster directory;
   - disk space (`df -h /home`) and the GB10 memory;
   - jobs stuck PENDING or running past their certified time.

   It ends with a per-lane verdict, **OK / STALLED / HUNG / DEAD / AT-RISK**, the evidence, and the exact recommended action.
3. **Acting on watchdog reports.** You act on them; the watchdogs never do.
   - **Read the newest reports from every watchdog** at least every 30 minutes, and whenever a lane notification arrives.
   - **Agreement.** Act when at least two watchdogs agree, or when one reports DEAD/HUNG with direct evidence (a missing process, a heartbeat older than 60 minutes with no live wait).
   - **Allowed actions:**
     - **Resume a stopped or hung agent** with SendMessage to its agent id or name, giving it the situation and the next step.
     - **Kill an orphaned waiter loop** (verify the PID first).
     - **Re-run a dead Codex audit** in the foreground with a timeout.
     - **Escalate to me** if an agent fails to recover after two resumes.
   - **Logging.** Record every intervention in `<campaign-dir>/watchdog/INTERVENTIONS.md`, and summarise them in the lab log.
4. **Watchdog lifecycle.** Each watchdog loop has an overall deadline (for example 24 hours) and is restarted by you when it ends, while lanes are still active. If a watchdog itself stops producing reports, restart it. Stop all watchdogs when no lane is active.

The campaign directory is `reports/2026-10-17-full-experiment-plan/campaign/` on `main`. Watchdog reports and `INTERVENTIONS.md` go there and are committed periodically. They are coordination records, not results.

## 5. Your own duties as coordinator

- **Before launching any lane,** restate the priority-1 lane design yourself from v4 and have Codex audit it, as the plan itself was audited. Only then launch.
- **When a lane hands back:**
  - read its report and audits;
  - spot-check its key numbers against the raw outputs yourself;
  - append a lab-log entry: what ran, the numbers, **what was retracted**, and what's open;
  - add a notebook entry (`notebook/entries/`, then `bash notebook/build.sh`, then commit);
  - commit and push `main`.
- **Keep the lab log's "Where things stand" block current.**
- **Never hand-type a number** into a report: generate it from run outputs.
- **Retractions matter more than successes.** If a result contradicts an earlier claim, say so plainly.
- Never mention or include the heat POD-128 comparison.
- After each priority, update one running results page (Artifact tool, plain language) so the user can glance at progress on a phone. Do not message them otherwise.
- Do not merge branches.

## 6. Lessons from earlier lanes (apply them)

- Every background wait has an **overall deadline** and a **timeout on each remote call**, and checks that the awaited **process is alive**, not just that the file is missing.
- Every Codex call has `timeout` and `< /dev/null`, and its output file is checked afterwards. Prompts that ask Codex to "fake a PASS" can trip its content filter; ask about honest failure modes instead.
- Acceptance checkers recompute every predicate from raw outputs. Forgery-only findings are "noted", not blocking.
- The local GB10 GPU has produced corrupted results inside one process (a "Red zone" error, a wrong condition number). Run analysis and scoring on CPU; every GPU number comes from the cluster, with `jax_backend=gpu` in the log.
- Large quadrature tables run out of memory. Gauss 112³ at R′ = 1024 needed a 42.9 GiB allocation; the 4096² 2D bank alone is about 64 GiB. Memory pilot first, using actual device bytes (143,771 MiB on H200).
- Retrains are not bitwise reproducible: record seeds and hashes.
- A sealed-key custody decision is pending. Sealed and reserved-OOD cohorts stay closed.
- An agent can die on a usage limit. Resume it with SendMessage, telling it where it was.
- Before a large disk write, check that the GB10 disk has headroom (`df -h`). Keep big arrays on the cluster paralab space.

## 7. Defaults for open decisions (do not ask; apply and record)

- **"k" means bank width r = R′.** Run the head-size sweep (C5, K ∈ {4, 8, 16, 32}) in priority 3 as well, so both readings are covered.
- **Sealed and OOD cohorts stay closed.** No retraining beyond the plan. If a priority's cap would be exceeded, defer the lowest-priority items (each lane's defer order) and record them; never exceed a cap.
- **Never merge branches.** List the finished branches in the final report as awaiting the user's merge decision.
- **If a kill criterion fires (governing plan K0–K2), or a result would change the paper's framing** (most likely the spectral FOM beating us), do not stop. Record it prominently, follow the plan's written consequence (for example, report the speed-up honestly and lead with mesh-independence), and continue.
- **Do not message the user mid-campaign.** Progress goes in the lab log and the notebook.

## 8. Final compilation (the deliverable)

When every in-scope experiment is done, failed or deferred:
1. **Generate a final results report** with a script under `reports/2026-10-17-full-experiment-plan/final/` (`make_final.py` → `report.md`, with figures). Every number comes from lane outputs; none is hand-typed. Structure:
   - **Hypothesis verdicts:** for each of H-cost-N, H-cost-r, H-acc-N, H-acc-r, H-comp (per dimension and FOM family) and H-base, give pass, fail or inconclusive, with the key number and plot.
   - **Per-priority results,** with plots: scaling against k, scaling against mesh, the accuracy–cost front, the speed-up against each FOM family, POD baselines, rule families, test space, the error budget, long-horizon runs, robustness.
   - **The P1g recheck:** how much the earlier "flat accuracy" results moved under frozen mode lists.
   - **Retractions and corrections, deferred items, a GPU-hour ledger, and the lane list** with branches and mirror commits.
   - **A plain-language glossary.**
2. **Have Codex audit the final report** (recompute numbers from raw outputs). Fix every issue and re-audit until clean.
3. **Publish it.** Make a plain-language web page (Artifact tool) and a PDF (send with SendUserFile). The user reads on a phone, so it should be plots, one-line takeaways and no jargon. Also add a notebook entry, rebuild `notebook/jcp-notebook.pdf`, and update the lab log's "Where things stand".
4. **Stop all watchdogs.**
5. **Final hand-back message:** the links, the hypothesis verdicts in plain words, anything retracted, the deferred items, the total GPU-hours, and the branches awaiting merge.
