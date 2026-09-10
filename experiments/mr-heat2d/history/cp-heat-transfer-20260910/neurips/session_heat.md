You are continuing work on NeurIPS 2026 submission 25242, "Tunable Non-linear Manifold ROMs
for Elliptic and Parabolic PDEs via Matrix-Free Galerkin Projection". Your scope is the
**heat block**: figure out whether our reduced-order model can be made genuinely faster, and
settle one posted accuracy number that does not reproduce.

## Where you are in the process

The paper got 3/2/2 with a meta-review saying every concern is essential. We posted a
rebuttal on 2026-07-28 and are now in the **discussion period**, delivering on public
commitments. Rules of this phase: replies are **text only** — no file uploads, no links, no
revised PDF. Numbers posted now can still move scores; anything deferred to the revision
cannot.

**Read before doing anything, in this order:**
1. `/home/tahmid/Dev/Neurips/HANDOFF.md` — state, standing decisions, infrastructure
2. `/home/tahmid/Dev/Neurips/COMMITMENTS.md` — what we publicly owe; **your section is §4c**
3. `/home/tahmid/Dev/Neurips/Rebuttal.md` — what reviewers actually read (source of truth)
4. `/home/tahmid/Dev/Neurips/REVIEWS.md` — the four reviews verbatim
5. `/home/tahmid/Dev/Neurips/CLAUDE.md` — GPU/cluster operating rules

## Why this work exists

Reviewer fxe8 attacked our timing methodology: no stated warm-up, no synchronisation, no
repeat counts, and he singled out our claim that treating κ as a runtime JAX argument took
Heat-3D N=128 from 14.7× to 191× as evidence the numbers were tracing artifacts. **He was
right.** We promised him corrected heat timings "within the next few days". That is your
deliverable.

## What has already been found (do not redo)

Re-timing under the corrected protocol is **complete**, and the result is bad:

| cell / arm | rel-L2 now | posted | OLD speedup | NEW speedup |
|---|---|---|---|---|
| Heat-2D N=64 acc / fast | 5.21e-3 ✓ / 1.08e-2 ✓ | matches | 39.6× / 68.0× | **0.079× / 0.198×** |
| Heat-2D N=128 acc / fast | **1.020e-2 ✗** / 1.354e-2 ✓ | **3.75e-3** / 1.35e-2 | 37.8× / 64.2× | **0.197× / 0.424×** |
| Heat-3D N=32 | 9.301e-2 ✓ | matches | 176.4× | **0.068×** |
| Heat-3D N=64 acc / fast | 7.330e-2 / 7.719e-2 (drift 1.7–2.1%) | 7.49e-2 / 7.85e-2 | 197.3× / 269.3× | **0.122× / 0.148×** |
| Heat-3D N=128 acc / fast | 1.812e-1 / 1.896e-1 (drift ≤0.4%) | 1.82e-1 / 1.90e-1 | 126.1× / 191.4× | **0.816× / 0.816×** |

**The ROM is slower than the full-order model on every heat cell.**

**Root cause — get this right, an earlier diagnosis was wrong.** It is not "we forgot to warm
up". Warming the submission-era FOM as written still leaves it at 17–28 s, because that FOM is
an eager Python loop with κ **captured in a closure**, so it re-traces and recompiles on every
call — you cannot discard a compile that happens every time. The real defect is **asymmetry**:
commit `0315b88` ("kappa as runtime arg") rebuilt the **ROM** as one jitted rollout compiled
once and reused across κ, and that treatment was never given to the FOM. Jitting the FOM the
same way — same operator, same CG tolerance 1e-6, same 1000-iteration cap, same 50 steps —
drops it to 15–123 ms, with max relative deviation 3.9e-7 against the cached reference states.
The honest description is: *we optimised the ROM's tracing and compared it against an
un-optimised FOM.*

**A posted accuracy number does not exist.** Heat-2D N=128 `3.75e-3` appears only in
`main.tex` and the rebuttal, nowhere in the result tree. The 37.8× speedup pins the source run
exactly (`slurm-617358`, commit `24dc331`, gn_tol 5e-3, max_iters 20 — the run that
`best-results/Heat-2D/N128/ACC/README.md` names), and that run measured **1.020e-2**,
reproduced to 2.8e-8. The posted accuracy is wrong by ~2.7× in the unfavourable direction.

**Unexplained 3D drift (0.16–2.1%).** Model-val fingerprints reproduce exactly and EQ node
counts are identical, so it is not a checkpoint mismatch. Per-trajectory deltas scatter with
cancelling signs, concentrated in stiff low-κ trajectories. Every affected cell uses EQ with
`patch_nnls`, which truncates NNLS at 3·nrows iterations — path-dependent, so not bitwise
reproducible across nodes. Cells without it reproduced to ~1e-8. **Hypothesis, not proven.**

**Job 1832027 (`hrt_h3d64`) is still PENDING** — one of five exclusive-node runs meant to
satisfy timing-protocol clause (iv) (idle dedicated device). The other four completed. Let it
land, or resubmit it.

## Your goal

1. Determine whether the heat ROM can be made **genuinely faster**, and by how much.
2. Resolve the Heat-2D N=128 accuracy discrepancy definitively — is 1.020e-2 the true number
   for that checkpoint, and where did 3.75e-3 come from?
3. Either explain or bound the 3D drift.
4. Produce numbers we can post to fxe8 and the AC.

**Legitimate levers for making the ROM faster:** EQ node budget, Gauss-Newton iteration cap
and tolerance, float32 vs float64, eliminating host synchronisation inside the rollout,
avoiding recompiles, restructuring the jit boundary, batching across κ, and any genuine
algorithmic improvement.

## HARD RULES — these are not negotiable

1. **The FOM implementation and its treatment are FROZEN.** Do not slow it down, un-jit it,
   loosen its tolerance, reduce its iteration cap, or change its convergence criteria. If an
   optimisation you apply to the ROM would also apply to the FOM, **apply it to the FOM too**.
2. **Report whatever number falls out, including if the ROM stays slower than the FOM.** We
   have already publicly withdrawn three speedup claims that all failed the same way — a ratio
   measured against a reference that was not doing real work (324× vs unpreconditioned CG,
   190× vs a stalled Newton solve, 39–269× vs an un-jitted FOM). A fourth would end the paper.
3. **Every number goes to a JSON in `Tunable-NM-ROM/results/raw/`.** Nothing is quoted from a
   log or from memory. Superseded runs move to a `_superseded_*/` directory, never deleted.
4. **Report per-κ ranges, not just medians.** Speedup is strongly κ-dependent (FOM CG count
   grows with κ, ROM cost does not): per-κ ratios span 0.04–0.17× at 3D N=32 and 0.27–2.6× at
   3D N=128.
5. **Never claim an experiment is running when it is not.** Two reviewers were told SMA runs
   were "currently running" when they had not started; do not repeat that.
6. Record isolation honestly — if a node is shared, set `exclusive_node: false` and log the
   co-resident job IDs, as the existing runs do in `extra.isolation_evidence`.

## Infrastructure

- **Cluster** (preferred for real jobs): `ssh tufts-login`. Python:
  `/cluster/tufts/paralab/tawal01/nmrom-rebuttal/venv/bin/python`. Torch jobs need
  `module load pytorch/2.13.0-cuda12.6-cudnn9`.
- Slurm scripts live in `~/nmrom/slurm/`. **Use `#!/bin/bash -l`.** Never combine `set -u`
  with `source modules.sh` — it silently kills the batch shell and has already cost 3 jobs.
- **A100s on this cluster are 40 GB**; large cells need `--gres=gpu:h200:1` (141 GB).
- Always assert the JAX backend is `gpu` and abort otherwise — a flaky node falling back to
  CPU produces plausible-looking but degraded numbers.
- **Local box** (smoke tests only): `/home/tahmid/Dev/.venv/bin/python`, wrapped in `jaxrun`,
  max 3 concurrent jobs. See `CLAUDE.md`.
- Heat code: `Tunable-NM-ROM/heat/`. Original paper pipeline (the one that produced the
  published numbers): `Tunable-NM-ROM/best-results/Heat-2D|Heat-3D/*/`. Checkpoints on the
  cluster under `~/nmrom/heat-out/`. **Note: `best-results/` contains source only — no
  checkpoints — so the models behind the published heat numbers no longer exist.**

## Coordination

Two other sessions may be active: one on Burgers (writes `COMMITMENTS.md` §4), and a running
job named `chartabl` (the learned-vs-frozen chart ablation). **Write only to §4c of
COMMITMENTS.md**; do not restructure the file or touch other sections. Result JSON filenames
are cell-prefixed so they will not collide.

## Deliverable

When you stop: update `COMMITMENTS.md` §4c with the findings, and give me a short plain-English
summary of (a) the best honest heat speedup and at what accuracy, (b) the resolution of the
3.75e-3 discrepancy, (c) what you recommend we post to fxe8 and the AC, drafted in plain
language — no jargon, no rhetorical flourishes, short sentences. A coauthor flagged an earlier
draft as "LLM-generated", so reviewer-facing text must read like a tired careful researcher
wrote it.

Stop when the ROM's cost is understood and either improved or shown to be at its floor. Work
autonomously; if something fails, diagnose and fix it rather than waiting. Give me
ADHD-friendly progress updates: short lines, bold the punchline, one idea per chunk.

---

## Iteration protocol — work in rounds until a target or a floor

Do not treat this as a single measurement pass. Work in rounds:

**Each round:** state a hypothesis about where the ROM's time is going → make one coherent
change → measure under the frozen protocol → write the JSON → record in a running log what
you changed and what it did. Then decide the next round from the evidence, not from a
pre-planned list.

**Targets, in order of value:**
1. **Minimum useful result:** at least one heat cell with speedup **≥ 1×** at an accuracy
   within 10% of the published number. Below 1× the ROM is slower than simply solving the
   problem, which is not showable.
2. **Good result:** ≥ 1× on the three 3D cells, where reduction should pay most. 3D N=128 is
   closest today at 0.816×.
3. **Strong result:** ≥ 5× anywhere at published accuracy.

**Diagnose before optimising.** Profile where the ROM rollout actually spends time — decoder
evaluation, Jacobian assembly, the k×k solve, EQ gather, host transfers. A ROM at 0.07× of a
15 ms FOM is doing something structurally wrong, not something 10% wrong. Find that first.

**Stop when ANY of these holds:**
- a target above is met and reproduced in a second run;
- **three consecutive rounds produce less than 10% improvement** — then declare the floor,
  report the best honest number, and explain *why* it is the floor (this is a legitimate and
  useful outcome);
- you have spent roughly a day of wall-clock without reaching target 1.

**If the floor is genuinely below 1×, say so plainly and stop.** That is a real finding and we
need it before a reviewer derives it. There is an honest framing available — the method still
buys a residual certificate, a tunable accuracy dial, and mesh-independent cost — but that
framing is only usable if we state the wall-clock result first. Do not keep iterating in
search of a number that the engineering does not support.

Report progress after every 2–3 rounds; do not go silent for hours.
