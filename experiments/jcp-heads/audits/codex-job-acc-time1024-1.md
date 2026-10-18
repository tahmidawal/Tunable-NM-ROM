**acc1024: SUBMIT. time1024: DO-NOT-SUBMIT until timing trajectories are preserved.** The acquisition panels are correct, but the analysis is incomplete and time1024 would discard evidence needed to verify joint accuracy–cost results.

No files changed, generators run, or jobs submitted.

| Item | Verdict | Finding |
|---|---|---|
| **1. Accuracy coverage** | **CORRECT** | 21 deployed arms × BE/CN-R, plus 12 additional fixed-M arms × BE: **54 arm/scheme records**, each covering all38. Successor replay covers sub16; check replay covers dev6. Rules and successors match `RULES.json`. A7 reuse of H32 and L32 is correct; names match analysis. |
| **2. Timing coverage** | **CORRECT** | 42 subject/scheme entries, each with dev6 ×3 and val32 ×1: **2,100 split/fused triplet pairs**. Certification uses dev6. Reference A is independently configured as linear384/CN-R/gauss128/1024². Doubled work targets H4s0/BE/dev6\|0, three repetitions. |
| **3. Staging and sbatch** | **CORRECT** | Both manifests pass. All **138 provenance-listed files per job** match their committed blobs at `59ac81975b443694b20f96c0310fa23f46b7191a`. GPU partition/types, GPU preflight, absolute venv, x64, highest precision, separate paralab namespaces, and certificate time limits agree. |
| **4. Resources** | **NEEDS-RESTATEMENT** | Estimates reproduce and memory is plausible, but “42 arms resident” is inaccurate: there are **22 arm objects including A**, shared between schemes. Estimates remain projections, not bounds. |
| **5. Analysis and registered outputs** | **WRONG as a complete implementation** | Accuracy records support Q1/Q2/Q3, E0 scoring, pooled H-acc-K and fixed-M analysis. Several required timing and comparison outputs are missing; details below. |
| **6. Recoverability** | **WRONG for time1024** | Validation timing trajectories and their actual identity hashes are discarded. Consequently, the saved output cannot establish that the accuracy attached from acc1024 belongs to the measured timing trajectory. |

The **time1024 submission blocker** is in [htime.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htime.py:127). Every case gets an untimed trajectory and an in-memory identity hash, but non-certification cases immediately `continue`. Only dev6 trajectories are archived. Timed records retain `hash_ok` booleans, not the actual hashes or validation outputs.

Meanwhile, [analyze.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:201) combines **A100 accuracy-job errors** with **H200 timing-job costs** without comparing their trajectories. Internal timing repeatability does not establish agreement with acc1024.

**Required before time1024 submission:** persist the scoring trajectory—compact `W` is sufficient with the pinned decoder—and actual identity hash for **every timed case**, with timed-repeat hashes linked to it. Then score those trajectories and verify consistency with the accuracy records before attaching their qualification. This is a small addition; full-grid trajectory storage is unnecessary.

The following **analysis fixes are required before accepting results**, but do not require rerunning acc1024:

- **Finish CN-R and per-seed Q-LH-cost.** [The main analysis](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:335) calls `finish_lh` only for pooled BE. CN-R therefore lacks equal-cost, at-most-cost and bracket results. Per-seed analysis runs BE only and likewise never finalizes Q-LH-cost. Preserve the registered BE-only F5 scope.
- **Implement timing qualification labels.** `timing()` merely copies certification rows; its caller discards them. The `analyze.timing_labels` mentioned in `hscore.py` does not exist. Also consume equivalence and doubled-work controls before accepting cost results.
- **Complete sensitivities and gates.** Per-seed H-acc-K transitions are absent. The per-head Jacobian control is recorded but does not gate `qualified`.
- **Pin and retain E0 provenance.** `runs/MANIFEST-dev.json` is absent, making the store-version assertion optional. Exported per-case errors discard reference hashes/version/metric metadata. The currently inspected store has **all 38 ST cases eligible**, so the present accuracy and cost populations do match.

Resource evidence is reassuring: acc1024 reconstructs to **2.9602 h**, limit **4.5 h**; time1024 to **1.5559 h**, limit **2.5 h**. The 1024² smoke peaked around **17.3 GB**, and the substantially larger 4096² pilot around **98.3 GB**. One minor estimator correction: `time_proj` counts seven burns per split/fused pair; the driver performs eight. Correcting that term alone still rounds to a **2.5 h** limit.

**Final disposition:** **SUBMIT acc1024**; **DO-NOT-SUBMIT time1024** until all-case trajectory/hash persistence is fixed, committed and restaged. Complete the analysis repairs before issuing any scientific verdict.