**Adversarial audit: do not launch this pre-registration as written.**

Audited [DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-22-ops-tune-deeponet/experiments/ops-tune-deeponet/DESIGN.md), SHA256 `8551b66e94b8ac69327d2a4f8c5e750991f78a147d790b66f8158b96bc334221`. Findings concern the **plan and inherited harness**, not certification of a new implementation.

All requested source files were readable. The historical cost-audit JSON was absent from this worktree; I read its canonical copy under `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/`. I read the canonical lab log’s current-state material and relevant dated entries, not its entire chronology. Remote cluster indices were not inspected; their quoted measurements remain unverified against the original files. No files were changed and no jobs were run.

Below, `D` means the target design; inherited filenames refer to `experiments/ops-deeponet-b2d/` unless otherwise specified.

**1. BLOCKER — The claimed generator provenance contradicts the pinned protocol.**

**Where:** D:45–47, 86–98, 134–138; `git show 5169c095:experiments/neural-operator-burgers/protocol.json`.

That protocol specifies anchor `dt=0.0003125`, while the design attributes the pinned data to an anchor at `0.00015625`. Its SHA256 is exactly the design’s `212bc898…`, so this is not an ambiguous filename. The quoted `data.py` hash also matches. Those source pins do not substantiate “exactly the modules that produced the pinned 128 cases” under the stated protocol.

**Fix:** Recover the generating job’s actual protocol, source manifest and recorded amendments. Pin those bytes separately from the proposed generation recipe. Resolve the discrepancy before claiming provenance parity.

**2. BLOCKER — `s-rank` cannot instantiate.**

**Where:** D:303–318; `families.py:238–241`.

The baseline has `trunk_width=384`; `s-rank` changes only `rank` to 512. The inherited constructor asserts `trunk_width >= rank`. The arm crashes before training. Adding `trunk_layers`, the only declared architectural code change, does not fix it.

**Fix:** Either explicitly remove and justify the restriction, or register a coupled rank/width arm and stop calling it a rank-only ablation. Check every proposed and composed configuration before submission.

**3. BLOCKER — The audit changes declared in §3 are grossly insufficient.**

**Where:** D:149; `audit.py`.

Admitting arm names and stop reasons leaves these failures:

| Inherited location | Failure under this plan | Required fix |
|---|---|---|
| 143 | Every extended-data arm must still have the original train-index hash. | Validate each arm against its declared data source and prefix. |
| 73–74, 149–151 | Requires batch 8, epoch patience 250, epoch cap 4000, and LR in `{0.001,0.0003}`. Batch 32 and LR 0.003 fail. | Compare against the frozen arm specification. |
| 147, 174, 186 | History length and indexing assume one entry per epoch. | Store and audit explicit evaluation IDs and optimizer steps. |
| 188 | Counts evaluations as “epochs at minimum LR”; cosine has a different floor. | Derive schedule-specific diagnostics in steps and elapsed time. |
| 270–305 | Supports one old data-manifest layout and one expected training hash. | Audit both mounts and generated-data manifests. |
| 312 | Requires `capacity-selection.json`, whereas the plan names `selection.json`. | Freeze one schema and pathname. |
| 316–320 | Requires every arm directory and rejects signal interruption; does not support declared budget skips. | Reconcile planned, completed, failed and skipped arms explicitly. |
| 329–331 | Requires cohort scoring and timing for every arm in every job. | Make requirements depend on job type; remove timing requirements. |
| 334–336 | Exports the old hash and `identical_split_to_fno_job=True`. | Report actual per-arm data provenance. |

**Fix:** Register the full audit migration, including independent recomputation of selection, composition and T0–T6. Prediction-error recomputation alone does not verify those decisions.

**4. BLOCKER — The unchanged cohort builder conflicts with the inherited cohort-hash gate.**

**Where:** D:129–132; `prepare_diagnosis_cohort.py:49–54, 91–95`; `audit.py:220–225`.

The cohort index embeds the **training-index hash**. Building it against the extended training set changes the index hash even if every evaluation field is identical. The inherited audit then rejects it against `FNO_COHORT_INDEX`.

Building against the original 128 preserves that hash but checks disjointness only against those 128, not all newly generated training cases.

**Fix:** Separate cohort-content identity from training-disjointness evidence. Preserve and verify the original cohort fields, and independently check them against the complete extended training set.

**5. BLOCKER — Symlinked caches will not survive the inherited collection procedure.**

**Where:** D:150–151, 358–359; `cluster/collect.py:29–30, 52–59`; `audit.py:305–306`.

The collector uses ordinary `tar -czf`, which preserves symlinks rather than copying their targets. A symlinked validation cache becomes a link to a cluster-only absolute path in the local archive. Local error recomputation then cannot read its targets. Changing only lane and namespace does not solve this.

The collector also excludes `data/train`, refinement data and solver sidecars. Applied to newly generated data, that can discard evidence that is not already archived elsewhere.

**Fix:** Define an explicit archive inventory. Materialize required validation/cohort evidence, archive the new generator’s records and convergence diagnostics, and preserve the generated cache independently before deleting its producing directory.

**6. BLOCKER — The pinned generator cannot simply be flattened into the parent staging layout.**

**Where:** D:134–150; pinned `data.py:21–26, 70–85`; `engines.py:16–17`; `cluster/stage.py:73–79`.

The generator derives a repository root from its own location and expects:

- `experiments/mr-burgers2d/engines.py`;
- `experiments/separable-decoder/sep_common.py`;
- `protocol.json` beside `data.py`.

It additionally checks the imported engine’s exact path. The parent stager copies lane files beneath flat `code/`. Copying those modules there unchanged breaks the generator’s path assumptions.

**Fix:** Register and stage the original directory structure, or explicitly amend the loader and provenance rules. Test imports from the actual staged layout, not the worktree.

**7. MAJOR — Extending `data.py` requires a new generation protocol, not just a larger count.**

**Where:** D:109–112, 146; pinned `data.py:64–67, 302–342`; pinned `protocol.json`.

`allowed_count()` rejects more than 128 training cases. The protocol’s declared future prefixes are **512 and 1024**, not 2048 and 4608. `generate()` obtains its solver setting from the passing calibration gate; it does not accept a cheaper setting chosen by throughput. Changing the protocol also invalidates `read_calibration()`’s protocol-hash check.

A new wrapper can legitimately reuse low-level functions, but must not inherit the original “empirically calibrated” status for settings that fail its reference budget.

**Fix:** Specify exactly which low-level functions are reused, which old gates are superseded, and the new index schema. Retain convergence checks and solver sidecars; distinguish approximate training labels from accepted evaluation references.

**8. MAJOR — The trajectory arithmetic is correct; the claimed state-parity interpretation is not.**

**Where:** D:64–82.

Verified arithmetic:

| Quantity | Result |
|---|---:|
| Operator evolved targets | \(128\times5=640\) |
| Operator supplied-plus-target time states | \(128\times6=768\) |
| Head source trajectories | \(576+4032=4608\) |
| Available head states | \(4608\times51=235008\) |
| Trajectory-count ratio | \(4608/128=36\) |
| Recorded-code/output-field ratio | \(131072/640=204.8\) |

The JSON supports the configuration counts; the September 16 log supports 131072 retained codes. But **204.8 is not a ratio of equivalent supervised learning examples**. The head reconstructs states using fitted latent codes; the operator learns an initial-condition-to-future-fields map. Time correlation, temporal sampling, spatial sampling and online PDE information differ.

Even at 4608 operator trajectories, there are only 23040 evolved target fields, not 131072. “Parity” would remain trajectory-count parity only.

**Fix:** Report the separate inventories and their roles. Remove “205× more supervised states” as an information-equivalence claim; identify it explicitly as a ratio of unlike recorded quantities.

**9. MAJOR — The bank is incorrectly described as trained on “POD snapshots.”**

**Where:** D:66; `experiments/separable-decoder/sep_burgers_r3.py:321–365`.

The source selects early-time-weighted FOM snapshots and trains a neural auto-decoder. POD is a diagnostic comparator, not the bank-training method. `max_snaps` is a cap, not an audited realized count.

**Fix:** Say “FOM state snapshots used for neural auto-decoder training.” Derive the realized count, retained trajectory coverage and time distribution from the training records. Use `hfit_pick`, whose purpose is explicitly documented in `sep_hfit_run.py:401–408`, to inventory the head’s actual retained states.

**10. MAJOR — “Only the count differs” is false.**

**Where:** D:80–82, 112–113.

The parameter distributions match; that does not make the training datasets equivalent. The NM-ROM checkpoint configuration records `dt=0.005`, 50 steps, early-time-weighted sampling and its own spatial discretization. Operators use five evolved times and refined-reference targets. The extended operator data introduce another discretization.

The checkpoint’s `N=256` convention also needs translating to intervals/nodes before equating it with the operator’s 256 intervals and 257² nodes.

**Fix:** Tabulate both numerical protocols, grids, output times, snapshot selection, target quality, learning objectives and online physics access. Limit the claim to the same parameter-family distribution.

**11. MAJOR — The accounting is not yet source-derived at the level claimed.**

**Where:** D:40–41, 64–98.

The design promises a future report generator but currently gives hand-entered empirical constants without a generated evidence artifact. The historical JSON contains neither 131072 nor the head job ID 2837431; those come from a different source. The reference timings and margins lack local hash-pinned evidence.

Also, `worst_empirical_margin` is **not simply difference from the anchor**: pinned `data.py:204–224` adds the anchor’s spatial and temporal refinement differences to the candidate-anchor discrepancy.

**Fix:** Generate an accounting artifact now, with source hashes and JSON pointers. Distinguish direct discrepancies, empirical margins, configuration caps and realized counts. Mark remote-only measurements as unverified until their metadata are archived.

**12. MAJOR — The three target-protocol safeguards do not establish a bound on the data benefit.**

**Where:** D:114–122, 251–254, 363–366.

The safeguards are useful, but limited:

- Nested prefixes control which physical draws are added.
- Paired targets measure label differences on the first 128 cases.
- The trained control measures one recipe’s response to those differences at 128 cases.

None establishes that target bias has the same effect at 4608 cases or after tuning. Coarser targets can suppress precisely the sharp structures the trunk struggles with. This can help optimization while worsening fidelity elsewhere. Tail cases beyond the first 128 can behave differently.

“If T0 fails, report the magnitude as a bound” has no mathematical basis: no bound or direction is defined.

**Fix:** Remove the bound claim. Report label discrepancies by time and physical regime, alongside paired model-error differences. State explicitly that the ladder estimates the effect of more **approximate-label** data. A transfer claim needs additional higher-fidelity checks beyond the original prefix.

**13. MAJOR — T0’s 10% band is not a measured noise band.**

**Where:** D:121–122, 251–254.

There are no repeated runs establishing training noise. A difference between two validation means is not an equivalence test; cancellation can hide large casewise changes. “Relative” also leaves the denominator unspecified.

**Fix:** Define the signed ratio and denominator, report paired casewise differences and uncertainty, and call 10% a prechosen practical tolerance. Do not interpret passing it as proof that the target protocols are interchangeable.

**14. MAJOR — The throughput-based solver choice is legitimate in principle but insufficiently specified.**

**Where:** D:100–105.

Choosing by resources rather than validation accuracy is not inherently post hoc. However, “measured throughput completes 4608 cases inside the generation budget” leaves open:

- mean versus maximum of the two profiles;
- treatment of first compilation;
- burn-in and output-writing costs;
- reserves for checksums and paired-target comparisons;
- safety margin for case-dependent Newton work;
- handling of failed profiles and overruns.

Those choices can determine which target protocol is used.

**Fix:** Freeze an executable decision formula, profile order, timing boundaries, safety factor, reserves and failure behavior. Record both profiles before the decision. Do not switch settings midway through the accepted prefix.

**15. MAJOR — Generation cost excludes a substantial mandatory overhead if it uses the cited timing field.**

**Where:** D:86–105, 295; pinned `data.py:149–165`.

`wall_seconds_including_first_compile` starts **after** a two-second GPU burn-in. Reusing `solve()` adds at least:

\[
4608\times2\text{ s}=2.56\text{ h}
\]

before writing files, checksums, index updates and comparisons. An eight-hour generation job allows only 6.25 seconds per accepted case including everything.

The 145.92-hour anchor extrapolation is arithmetically correct given 114 seconds, but it does not establish warm throughput for a newly optimized generation path.

**Fix:** Profile end-to-end accepted-case throughput. Separate compilation, burn-in, solve and persistence costs. Treat 146 hours as a historical extrapolation, not proof that every pinned-protocol strategy is infeasible.

**16. MAJOR — `c-pinned128` is simultaneously called a reproduction and admitted not to be one.**

**Where:** D:145, 168–170, 187–190.

The later caveat does not repair the earlier like-for-like claim. This lane changes validation frequency, checkpoint-selection opportunities, stopping patience and LR scheduling. It therefore changes more than “the stopping rule alone.”

Moreover, comparison to a historical job does not isolate these changes from different hardware under a wall-limited schedule.

**Fix:** Preserve `don-small` as the historical recipe-matched row. Label `c-pinned128` as a new-schedule control. If causal attribution matters, rerun the inherited recipe beside it and describe the intervention as the complete validation/scheduler/stopping change.

**17. MAJOR — The patience arithmetic is partly correct and partly wrong.**

**Where:** D:176–185.

Correct at batch 8: 16 and 576 steps per epoch; \(500\times40=20000\) steps; 1250 small-data epochs; five times the old early-stopping patience.

Wrong or unsupported:

- PyTorch `ReduceLROnPlateau` reduces after `num_bad_epochs > patience`. Patience 8 means the ninth bad evaluation, or 4500 steps on this cadence, not 4000.
- Likewise, inherited patience 20 means 21 bad scheduler calls before reduction.
- Twenty top-rung epochs are 11520 steps; “4000 is the closest schedule-shaped equivalent” has no defined derivation.
- Batch 32 changes all epoch/sample-exposure equivalents.

**Fix:** Register scheduler semantics precisely, including its improvement threshold. Separate optimizer steps, evaluations and processed examples. Remove the claimed equivalence unless an actual equivalence criterion is provided.

**18. MAJOR — The stopping diagnostic may again be unreachable within some arms’ budgets.**

**Where:** D:179–185, 311–320.

Early stopping needs at least 20000 non-improving steps after a best score. A 1500-second arm must exceed 13.33 steps/second even before accounting for the initial best, warm-up and validation. The larger trunk, rank and batch arms have no throughput evidence establishing that possibility.

The inherited 4000-epoch cap also remains unspecified in the new design. Keeping it imposes different maximum step counts across data sizes; removing it changes another inherited control.

**Fix:** Define epoch/step caps and require realized steps, examples processed, validation count, LR reductions, warm-up completion and stop reason. If patience could not fire, report it as inactive under the budget—not evidence of continuing improvement or convergence.

**19. MAJOR — Step-based validation requires changes beyond the training loop.**

**Where:** `train.py:147–159, 172–193`; `training_smoke_second.py:63–73`; D:145.

The inherited checkpoint and result schema indexes history by epoch. Multiple evaluations per epoch break that indexing. Conversely, the two-epoch smoke performs only four optimizer steps, so a universal 500-step cadence produces no `best.pt` before the inherited restore.

A production arm interrupted before its first validation has the same problem.

**Fix:** Preserve a genuinely explicit legacy mode, introduce evaluation/step IDs, and define initial/final validation and checkpoint behavior. Add smoke coverage for the new cadence, interruption before the first scheduled evaluation, and checkpoint reload.

**20. MAJOR — The cosine schedule is contradictory and not fully defined.**

**Where:** D:194–199.

“A cosine arm cannot early-stop” directly contradicts the next clause allowing early stopping. Warm-up is measured in steps but decay in wall time; the transition formula is unspecified. A slow arm might spend most of its budget warming up. Validation, checkpoint writes and stalls may advance the cosine without optimization.

Changing from 1500 to 3000 to 9000 seconds also changes the LR trajectory. A longer run is not merely more training of the same schedule.

**Fix:** Specify the exact piecewise schedule and clock, including warm-up-over-budget behavior. State whether final runs restart or resume. Remove the contradictory sentence and report attained schedule progress.

**21. MAJOR — The sweep does not perform the advertised one-factor comparisons.**

**Where:** D:281–286, 296, 310–326.

`base-top` receives 3000 seconds; every sweep arm receives 1500. Thus `s-cos` versus `base` changes both schedule and budget.

If cosine wins, subsequent arms change their named knob **and** the schedule relative to the plateau `base`. Comparing them to `base` does not identify the knob’s contribution. The LR rows explicitly say cosine while the later paragraph makes their schedule conditional.

Composition can consequently import a knob because the schedule helped, not because that knob helped.

**Fix:** Compare plateau and cosine at the same budget. Then freeze a schedule-matched control at the sweep budget and evaluate every subsequent knob against it. Specify separately how the selected schedule enters `tuned`.

**22. MAJOR — The final selection surface is not actually frozen.**

**Where:** D:227–229, 281–286, 296–298.

“Arms of the stage being selected” does not define the stage membership. Open choices include:

- whether both 128-case controls compete for final selection;
- whether selection spans all ladder rungs;
- whether `base` and `base-top` are aliases;
- whether 1500-second sweep checkpoints compete with 3000-second ladder checkpoints;
- whether a winning lower-rung recipe is forcibly moved to the top rung;
- whether shorter-budget checkpoints remain eligible after 9000-second retraining;
- tie handling and missing-arm handling.

`fin01`’s “at the top rung” can change the selected configuration/data combination.

**Fix:** Freeze a decision graph with candidate IDs, dataset, budget, score field, tie-breaks and fallback rules at each stage. Distinguish choosing a recipe from choosing a checkpoint.

**23. MAJOR — T2 and the composition fallback can issue a false negative about tuning.**

**Where:** D:260–261, 281–286.

A single-knob arm can improve substantially while composition fails; the plan selects that single-knob arm but declares “tuning helped” only if the composed arm passes. Those statements answer different questions.

The no-winner branch is also not necessarily “fails by construction”: a fresh `base` run in another job can differ because its realized wall-limited optimization differs. It is deterministic only if the same result is reused.

**Fix:** Separate “composition helped” from “best tuned recipe helped,” evaluated at matched budgets. Define whether an unchanged composition reuses a checkpoint or triggers a new run.

**24. MAJOR — T1 cannot establish data saturation.**

**Where:** D:255–259, 367–369.

At fixed wall budget, more data can mean fewer passes per example. A flat result can reflect optimization limits, schedule interaction or approximate-target bias—not saturation of data benefit. Any regression also satisfies “improves by less than 10%.”

If generation stops barely above a prescribed rung, the last increment may be too small to test saturation. Below 128 cases, the stated controls cannot even be constructed.

The 0.7× criterion is a practical effect-size bar, not an “iff” definition of data limitation.

**Fix:** Rename the conclusions to observed gains under the allotted compute. Register minimum rung spacing, minimum accepted dataset size, partial-prefix mapping, and an inconclusive outcome. Reserve “saturated” for evidence that also addresses optimization adequacy.

**25. MAJOR — The threshold decisions are uncalibrated, especially the 2% composition trigger.**

**Where:** D:251–261, 281–283.

Pre-registering arbitrary practical thresholds is permissible; calling them noise, equivalence or scientific regime boundaries is not. A 2% change on 32 repeatedly used validation cases is especially vulnerable to selection variability. Carrying a 10% threshold over from a different head-training experiment does not calibrate it here.

**Fix:** State their operational purpose, report continuous effects and paired uncertainty, and avoid binary scientific conclusions near thresholds. Predefine sensitivity reporting without using it to select a different winner.

**26. MAJOR — “Selection bias is conservative for this paper” is not generally true.**

**Where:** D:244–247, 373.

Minimum validation mean is optimistically biased for that selected metric. It does not guarantee better worst-case error, better cohort performance or a stronger independently evaluated baseline.

Adaptive validation overfitting can choose a recipe that generalizes poorly, thereby making the NM-ROM look better on another cohort—the opposite direction. The parent lane already demonstrates disagreement between mean-selected and tail-best arms.

The effective search also exceeds “~12 arms”: checkpoint selection, schedule selection, ladder/composition decisions and final retraining all reuse validation-32.

**Fix:** Restrict the bias statement to the selected validation mean. Count the complete adaptive procedure and qualify all generalization claims. A fresh confirmation cohort is needed for a paper-level selected-baseline claim.

**27. MAJOR — The eight-case cohort is neither fresh nor independent of the reference-design evidence.**

**Where:** D:90–91, 239–242, 265–267; `prepare_diagnosis_cohort.py:1–16, 57–61`.

These are the same eight calibration cases used for the reference comparisons. Their operator and ROM results have already been inspected in earlier lanes. Scoring them only once **in this lane** does not make them an untouched test set.

“Fourth draw” means another operator family, not another independent statistical draw.

**Fix:** Call this reused diagnosis/calibration evidence. Freeze the selected checkpoint before scoring it, but do not claim independent confirmation. If fresh testing is out of scope, state that limitation beside the paper comparison.

**28. MAJOR — The most important data result is scheduled behind the tuning sweep.**

**Where:** D:296–298; brief’s priority order.

The first training job establishes only the two endpoints, then spends 15000 seconds on tuning. The intermediate data ladder—and evidence needed to distinguish a trend from an anomalous endpoint—waits until `lad01`. A sweep crash or deadline loss sacrifices the priority question first.

**Fix:** Complete the base data ladder before optional capacity tuning, or protect its arms with a fixed reserve and explicit priority order. Preserve valid completed controls if later optional arms fail.

**29. MAJOR — The job totals fit nominally, but the runtime guarantee is unsupported.**

**Where:** D:293–301; `worker_second.py:74–82`; `train.py:88–115, 161–169`.

The arithmetic itself is sound:

| Job | Allocated arm training | Nominal remaining wall |
|---|---:|---:|
| `tun01` | 24000 s = 6 h 40 min | 3 h 20 min |
| `lad01`, six arms | 18000 s = 5 h | 2 h |
| `fin01`, two arms | 18000 s = 5 h | 2 h |

But the inherited training timer begins after data loading, normalization and model setup. The worker can silently shorten an arm to `remaining()-RESERVE`, and the trainer checks wall/signal stops only at epoch boundaries.

At the top rung, an epoch is 576 steps. Large models can overrun materially. The widest trunk is evaluated over all 66049 grid nodes every step; its cost cannot be estimated from `don-small`. Batch 32 also needs full-resolution memory validation.

**Fix:** Budget end-to-end wall time; enforce step-boundary stops; reserve final validation/checkpoint writing. Pilot the heaviest registered configurations. Mark shortened arms ineligible for matched-budget conclusions.

**30. MAJOR — “Read 17 GB once per job” is not delivered by adding a pool option alone.**

**Where:** D:147, 209–214, 376–377; `dataset.py:20–74, 115–134`; `train.py:44–55, 88–107`.

The inherited path hashes each archive, loads it for schema validation, loads it again for duplicate checks, loads it again for train/validation overlap checks, and finally loads/stacks it for training. Each arm is a separate subprocess. Normalization then scans full tensors and creates additional large temporaries.

The raw seven-field pool is approximately **17.04 GB**, so the estimate is reasonable; the one-read claim is not. Byte-identical `dataset.py` can coexist with an optimized path only if these repeated calls are replaced by an equally strong verified-pool contract.

**Fix:** Verify source data once, hash the packed arrays themselves, and use a read-only mapped pool. Define prefix-specific normalization and record actual selected case IDs. An index hash alone does not prove that the pool arrays match that index.

**31. MAJOR — Several additional shape and configuration traps need explicit coverage.**

**Where:** D:144–145, 203–214; inherited model/trainer.

| Location | Trap | Fix |
|---|---|---|
| `model.py:94` | A five-element scale vector broadcasts against spatial width, not output channels. | Store per-time scale as `(1,5,1,1)` and test reload/evaluation. |
| `families.py:293–296` | Constructor arguments do not currently pass `trunk_layers`; frequencies use config key `trunk_frequencies`. | Freeze and test exact config keys. |
| `families.py:251–255` | `trunk_width` also changes the branch hidden layer and readout. | Label `s-trunk` a coupled branch/trunk-capacity arm. |
| `dataset.py:58, 75` | Reader preserves index order; it does not assert ascending contiguous `case_index`. | Assert exact expected IDs/seeds for every prefix. |
| `dataset.py:25–26` | A budget-truncated generator index remains unusable if marked incomplete. | Specify validated partial-prefix finalization. |
| `train.py:140` | Validation batch size follows training batch size. Batch-32 arm changes selection batching too. | Fix validation batch size independently. |
| `audit.py:176–178` | The inherited tolerance is **relative**; D’s “1e-5” is ambiguous. | Register the exact inequality and applicable batch sizes. |

Do not rely on tiny default-architecture smokes to catch these changes.

**32. MAJOR — The six-job cap has an unauthorized loophole.**

**Where:** D:290, 300–301.

The brief says **≤6 total**. The design imports a parent-lane exception excluding preamble deaths and ambiguously refers to “the four.” A generation job has zero *training* GPU time even after substantial generation work, making that criterion particularly unsuitable.

**Fix:** Count every submitted job toward six unless the user explicitly changes the cap. Classify failures separately without erasing them from the ledger.

**33. MAJOR — The cheap-reference gate and cleanup lifecycle are incomplete.**

**Where:** D:337–359.

The gates emphasize hashes and schema but omit an explicit acceptance rule for the new generation’s per-step residuals, failed cases, missing records and complete contiguous prefix. A fallback must not skip difficult failed cases and continue, because that changes the distribution.

The generated cache must also remain available to later jobs while its producing job directory is supposed to be deleted after collection.

**Fix:** Require convergence diagnostics for every accepted case; forbid silent replacement/skipping; define a failed-prefix policy. Give the immutable cache a separate, verified lifecycle and archive it before deleting the producer.

**34. MAJOR — The plan lacks the diagnostics needed to answer why DeepONet is weak.**

**Where:** D:12–35, 249–277.

The parent evidence points at the first evolved field and a sizable training/generalization gap. This plan sweeps parameters but does not require:

- comparable training and validation errors at the selected checkpoint;
- per-time errors;
- optimization progress versus steps and examples;
- a learned-trunk projection diagnostic separating spatial representation from branch prediction;
- error and label-discrepancy breakdowns by width, viscosity and amplitude.

Without these, a flat ladder or failed sweep remains ambiguous between representation, optimization, generalization and label quality.

**Fix:** Register these diagnostics, preferably using retained fields and a bounded fixed training subset. Keep any truth-assisted projection diagnostic explicitly separate from admissible predictions.

**35. MINOR — The opening comparison mixes incompatible statistics and cohorts.**

**Where:** D:14–16.

DeepONet means/medians and other operators’ validation means are placed beside the ROM’s **eight-case worst** 1.87%. That sentence invites a numerical comparison the later matched-cohort section does not justify.

**Fix:** Label metric and cohort for every value, or remove the ROM number from that sentence.

**36. MINOR — The discretization watch-item is responsibly caveated but scientifically weak.**

**Where:** D:268–274.

The explicit different-cohort caveat is sound. Nevertheless, crossing a six-case worst-error constant on validation-32 does not establish outperforming the same-grid solver on those validation cases. Also, the parent table already contains `unet-medium` validation worst 3.9622%, below 4.0265%; a broad premise that every operator validation worst exceeds this threshold is already false.

**Fix:** Retain this only as historical context. If the appendix argument depends on beating discretization error, compare operator and converged same-grid FOM on the same cases and times.

**37. MINOR — The comparison scope and precision exception need explicit wording.**

**Where:** D:154–170, 218, 303–306; canonical lab log’s September 22 coordinator entry; repository `AGENTS.md`.

“Only DeepONet” is true relative to the historical rows, but the coordinator simultaneously launched a grid-operator tuning/data lane. It will not necessarily describe the final paper comparison.

The float32 network is inherited and openly declared, but conflicts with the repository’s blanket f64 wording; wrapping outputs in f64 does not make the network computation f64.

**Fix:** Name the historical comparison panel and freeze how sibling updates enter the report. Document the inherited operator-network precision exception explicitly and retain TF32/dtype checks.

The fixed-initial error definition, exact supplied-state return, exact boundary checks, independent NumPy prediction audit, and prohibition on cross-job speed ratios are sound. The nested-prefix strategy is also sound **once its ordering, partial-run behavior and target protocol are made explicit**.