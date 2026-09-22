# Second independent report audit, 2026-09-22

Written by an independent auditor (a Claude subagent, read-only, with no part in writing the
lane), commissioned in parallel with the Codex pass and reported after it, so it audited the
already-corrected state — and had independently found six of that pass's eight issues, including
the one wrong rendered number. Brief in `report-audit-prompt-subagent.txt`. **Ten findings, all
accepted**; disposition in `DESIGN.md` §A6. Reproduced as received.

---

## Verdict

**No number in the report is wrong.** I re-derived all 99 rendered cells from the pinned sources and every one matches.

Important context: **the report and generator were rewritten at 18:33–18:34, during this audit**, by a concurrent Codex pass (DESIGN §A5, commit `84dae726`). I had independently found six of the eight issues that pass fixed — including the one genuinely wrong rendered number (the timing "repetitions" column printed `sum(shape) = 62` instead of `30 × 32 = 960`) and the cross-job timing surface in `summary.json`. Those are now fixed and verified. Findings below are against the **current** committed state (`84dae726`, tree clean).

## What I verified and found correct

- **End-to-end recompute, 4 DeepONet arms + 2 cohort arms.** From `runs/don01/archive/don01/out/*/*.prediction.npz` and `data/validation/`, using `audit.py:fixed_initial_errors` as written (denominator `‖target[0,0,1:-1,1:-1]‖`, channel 0, interior only). Exact agreement to all 4 printed decimals with both `audit.json` and the rendered table: `don-small` 14.7877 / 11.7522 / 54.7444 / 30; `don-medium` 14.8999 / 12.7297 / 51.7875 / 29; `don-large` 18.2211 / 14.8022 / 60.3540 / 32; `don-refine` 15.7642 / 12.4612 / 58.9848 / 30. Cohort (`data/diagnosis-cohort/`): `don-small` 16.0229, `don-large` 26.6330. Supplied initial returned bitwise and `errors[0] == 0` on every case.
- **All 99 rendered cells** (§1 capacity table, §2 16 rows × 4 metrics, §3 23 rows, §5 12 fields, §4 D1/D2 ratios) reproduce from their stated source.
- **All five source SHA256s + the generator SHA256 re-hash correctly.**
- **"12 of 12 sibling arms ended on their budget"** — genuinely derived and true (4 U-Net + 4 Transolver `stop_reason=wall_budget`; 4 FNO `stopped_by_wall_budget=true`).
- **"All four arms early-stopped"** — true; `epochs − best_epoch = 251` exactly on all four, i.e. patience 250 fired.
- **D1 8.577 / 6.509, D2 0.16023 vs ROM 0.018671, D3 ranking** — all arithmetically correct; D3's ranking-by-selected-arm-worst does yield FNO, U-Net, Transolver, DeepONet.
- **Cohort index hash** `8b8a2ee1…` does equal the FNO job's, and the eight case ids match the ROM/FOM diagnosis audit's.
- **Selection rule** applied unchanged (`capacity_selection.json` scores = the audited means; `refine` refines `small`); the pre-registered tail warning fires correctly and names both arms.
- **No speed claim.** §5 makes none; `summary.json` now carries 8 timing rows, all `job_id=4179556`, `cross_job=false`. The ROM/FOM `gpu_median_ms` present in the pinned diagnosis source is not imported. `timing-handoff.json`: all 4 checkpoint hashes + `families.py`/`model.py` hashes re-verified; its new accuracy ratios (5.875–10.935×) all recompute correctly.
- **Gates:** `jax_backend=gpu`, `torch_backend=cuda`, log ends `ALL-DONE`, `data_verified=395` = 395 manifest lines.

## Findings

### MAJOR 1 — `HANDOFF.md:19-20` retains the claim §A5 just retracted

> "accuracy got monotonically worse with capacity, so neither 'too small' nor 'too little time' explains it"

The Codex pass removed exactly this overreach from the report (§2 now reads "three coupled configurations are not a capacity sweep, so this does not rule out under-capacity"). The handoff — the document that explicitly tells the next session "Read that before touching anything here" — still carries the withdrawn version, plus "Nothing is typed" (`HANDOFF.md:39`), which the report's own scoped preamble now contradicts. **Fix:** replace both sentences with the report's current wording.

### MAJOR 2 — a load-bearing empirical claim is a hard-coded literal

`reports/generate_report.py:218` renders into §1:

> "the training loss was still falling in all four histories"

This is an f-string constant. The generator reads only `audit.json`, which records no training loss, so nothing derives it and "all four" is typed. It is the single sentence that stops the negative being overstated. **I checked it and it is true** (train MSRE final < train MSRE at best epoch for all four: 0.00131<0.00183, 0.00089<0.00200, 0.00419<0.00739, 0.00207<0.00318) — but this is the no-second precedent finding recurring verbatim. **Fix:** have `audit.py` record `train_loss_at_best` / `train_loss_final` per arm (one line in `audit_arm`) and derive the sentence.

### MAJOR 3 — the comparison arms' realised compute is invisible, and §6 says it is not

§6: *"The float32 network gets more epochs per second than the float64 FNO did — favourable to this lane, and the epoch counts are in the table."* The §1 table contains **only this lane's four arms**; no sibling epoch count or training time appears anywhere in the report. The reader therefore cannot see that the realised compute went the other way:

| | training s | epochs |
|---|---|---|
| DeepONet | 543–898 | 384–602 |
| U-Net | 3001–3002 | 908–2327 |
| Transolver | 3001–3006 | 586–1628 |
| FNO | 3003–3005 | 688–1741 |

Equal *budget* was held, as pre-registered — but realised training time differs **3.3–5.5×** and epochs up to 4.3×. Calling that "favourable to this lane" while omitting the numbers that would show otherwise is the no-second "missing epoch/parameter counts for the comparison arms" finding again. **Fix:** add the sibling rows to the §1 table (all fields are in the pinned `unet01`/`tsol01`/FNO audits) and drop or requalify the "favourable" sentence.

### MAJOR 4 — "No arm here was still improving when it stopped" is structurally unreachable, not a finding

`still_improving` requires `best_epoch ≥ 0.95·(epochs−1)`. Under early stopping with patience 250, `best_epoch = epochs − 251`, so the flag needs `epochs ≥ 5001`; the epoch cap is 4000. **No early-stopped arm in this protocol can ever be flagged "still improving."** The §1 sentence and the whole "Still improving?" column are therefore determined by the stop reason, yet §1 presents them as a second, independent fact reinforcing the first. The revised glossary softens the flag but does not say it is unreachable here. **Fix:** either suppress the column when `stop_reason == early_stopping`, or add one clause: "(by construction this flag cannot fire for an early-stopped arm at patience 250)".

### MAJOR 5 — the most likely explanation is never offered: 128 training cases and a large train/validation gap

The training set is **128 cases** (`data/DATA.sha256`: 257 train entries = 128 cases × 2 files + index). The report never states it. Meanwhile the arms reach train MSRE 0.0013–0.0042 (≈3.6–6.5% RMS) against 14.8–18.2% validation. A DeepONet pushing a 257² field through a 4×4×8·width global bottleneck with 2.6–10.3 M parameters on 128 samples is the textbook data-limited-generalisation case, and it is far more exposed to it than the convolutional field-to-field baselines it sits beside. §1 and §2 offer only "the architecture or the inherited schedule". Related, and also omitted although it is already in `audit.json` as `final_learning_rate`: **all four arms sat at the scheduler's 1e-5 floor for 146–209 epochs before stopping** — the strongest available evidence that the schedule was genuinely exhausted, and it is dropped. **Fix:** state the training-set size, add `final_learning_rate` and the train-loss pair to §1, and name data-limited generalisation as a third reading in §2.

### MINOR 6 — the "equal to the FNO job's" split hashes are not traceable to the FNO job

`audit.py:26-27` hard-codes `5333584b…` / `468b9e70…` with the comment "(its provenance.json, reproduced in its report sources)". Neither literal occurs anywhere in `experiments/neural-operator-audit/`, and no FNO `provenance.json` is present (only `field-audit.json` + archive parts). What is actually verified is (a) self-consistency against `don01`'s own archived manifest and (b) equality with `unet01`/`tsol01`, both of which *are* pinned. The FNO leg rests on an untraceable literal, and the new preamble's list of disclosed literals does not include it. **Fix:** either derive them from the FNO field-audit (as the cohort hash already is) or add them to the disclosed-literals sentence.

### MINOR 7 — `DESIGN.md:294` states a wrong number

A3: "`don-small`, ended by early stopping at **561 s** of its 3000 s budget". The audited value is 552.48 s (`training_seconds`), or 551.62 s (last history `wall_seconds`). Probably a transposition of 551. It is the only hand-typed measurement in the pre-registration. **Fix:** 552 s.

### MINOR 8 — D3's "ranking" contradicts §2 at a glance

Ranking by *selected-arm worst* puts FNO first, while §2 shows the U-Net is the best family by mean, median and cohort. Only `deeponet_is_weakest` is load-bearing (and robust under every metric), but the bare JSON array is not explained in §8. **Fix:** one glossary line, or emit the ranking under all three metrics.

### MINOR 9 — `timing-handoff.json` checkpoint paths lack the `worktrees/` prefix the sibling entries carry

`requires.families_py.path` starts `worktrees/2026-09-22-…`; `checkpoints[*].path` starts `2026-09-22-…`. A consumer resolving both against one root fails on the checkpoints. **Fix:** prefix the four checkpoint paths.

### MINOR 10 — the audited artefact was not frozen

The report, generator, `summary.json`, `timing-handoff.json` and `DESIGN.md` all changed at 18:33–18:34, mid-audit. Nothing here is wrong, but an audit whose target moves cannot certify what a reader saw. **Fix (process):** commit and state the report SHA256 before commissioning an audit.

## What a hostile ICLR reviewer would still ask for

Beyond MAJOR 3/5: independent seeds (declared absent); a train-vs-validation curve; a patience/schedule ablation, since the whole negative rests on an inherited stopping rule the family did not choose; a per-output-time breakdown — the error is **largest at the first evolved time and falls thereafter** (`don-small` mean by time: 0, 14.43, 10.71, 9.49, 9.17, 9.12 %), which points at the bottleneck failing to represent the early sharp field rather than at error accumulation, and is a real diagnostic the report has in hand and does not show; and a trivial baseline. On the last, the report gives no floor at all — I computed it: persistence (hold t=0) scores 64.69 % mean / 90.65 % worst, so the DeepONet is learning something substantial, roughly 4.4× better than persistence while 10.9× worse than the U-Net. Including that would make the negative both more credible and harder to misread.

**Direction of the negative:** as now written it does not excuse DeepONet, and after the A5 pass it no longer overstates its weakness in the capacity or early-stopping readings. The residual tilt is toward *overstating*, via MAJOR 3 (a 3.3–5.5× realised-compute deficit presented as a wash, indeed as "favourable"), MAJOR 5 (the 128-case generalisation reading never named), and MAJOR 4 (a tautology doing duty as evidence).
