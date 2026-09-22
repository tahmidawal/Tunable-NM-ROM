# Adversarial design audit — `ops-tune-deeponet/DESIGN.md`, 2026-09-22

Independent, read-only, before any GPU job. Target: `experiments/ops-tune-deeponet/DESIGN.md`
in worktree `worktrees/2026-09-22-ops-tune-deeponet`. Nothing was modified except this file.
No job was submitted; the cluster was touched read-only over `ssh tufts-login` to verify the
pinned index metadata quoted in §2.

**Verdict: do not submit `gen01` as written.** Seven blockers, of which five are the same class
the parent lane's §A2/B1 finding exists to catch — a pre-registered gate that crashes for a
reason unrelated to the science — and two are wrong-artefact errors that would make `gen01`
either refuse to run or generate against the wrong reference protocol. One quoted measurement
(§2.3, 114 s/case) is hand-typed from a single record and is 14 % low. Two pre-registered
thresholds (T0, and §5.3's 2 %) are unfalsifiable or below the noise they are meant to sit
above, and I can show that with numbers from the parent lane's own saved fields.

## What I read, and what I could not

Read in full: the target `DESIGN.md`; `ops-deeponet-b2d/{DESIGN.md, HANDOFF.md,
reports/2026-09-22-ops-deeponet-b2d.md, train.py, model.py, families.py, dataset.py,
worker_second.py, audit.py, prepare_diagnosis_cohort.py, evaluate_cohort.py,
check_inherited.py, cluster/stage.py, cluster/collect.py, configs/deeponet/small.json,
specs/don01.json, runs/don01/audit.json}`; `git show 5169c095:` for
`experiments/neural-operator-burgers/{data.py, protocol.json, protocol-refined.json, refine.py}`;
`reports/2026-09-10-historical-poisson-and-burgers-cost-audit.json`; the LAB-LOG entries that
mention 4608; and, on the cluster, `pilot-data01/{train,validation,refinement}/index.json`.

Could not read, because they do not exist yet: `gen_more.py`, `build_pool.py`, `worker_tune.py`.
Findings about them are about what the plan commits them to.

Could not verify independently: the `131072` decoder-code count (see m4) and the frozen
checkpoint's internals; both exist only as prose in `LAB-LOG.md`.

---

## BLOCKERS

### B1 — `audit.py` pins the train index to the FNO job's hash; every pooled arm fails the audit

`audit.py:143` asserts, **per arm**:

```python
assert provenance['train_index_sha256'] == FNO_TRAIN_INDEX, folder
```

and `audit.py:303` asserts the same at job level against the archived `DATA.sha256`
(`assert from_manifest.get('train') == expected_train`). `FNO_TRAIN_INDEX` is the literal
`5333584b…` (`audit.py:28`), the 128-case index — I confirmed on the cluster that
`pilot-data01/train/index.json` hashes to `5333584b7162df62…`.

Every arm in this lane trained on the extended pool records a different
`train_index_sha256`, so `audit_arm` raises on the first such arm, `audit.json` is never
written, and §8 gates 5 and 6, the collector's cleanup guard and the report generator all go
down with it. §3's table promises `audit.py` changes are "admits the new arm names and the new
`stop_reason` values" — that does not cover this.

**Fix.** `audit.py` must take the expected train hash per arm from the arm's declared data
source (pinned cache → `5333584b…`; extended pool → the hash `gen01`'s manifest recorded), and
§8 gate 3 must be restated that way. Do not simply delete the assertion; it is the only thing
that stops an arm silently training on the wrong split.

### B2 — `audit.py` asserts a timing block that §3 removes

`audit.py:330-331`:

```python
timing = audit_timing(root)
assert timing['present'] and set(timing['models']) == expected_arms
```

`audit_timing` returns `dict(present=False)` when `out/timing/timing.json` is absent
(`audit.py:250-251`). §3 removes `timing.py` from the lane and §7 says no timing block runs.
So **every job in this lane fails its own audit at this line**, unconditionally, after all the
GPU time has been spent. This is byte-for-byte the parent's B1 failure mode.

**Fix.** Make the timing block optional in `audit.py` and record `timing: {present: false,
reason: 'DESIGN §7 — no timing is admissible from this lane'}`. Add it to §3's change list.

### B3 — §5.1 and `audit.py:329` contradict each other on the matched cohort

§5.1: "The matched eight-case ROM/FOM cohort is scored **once, after every selection decision
is fixed**". §6 puts that scoring in `fin01`.

`audit.py:329`:

```python
assert pde == 'poisson' or (cohort['present'] and set(cohort['models']) == expected_arms), 'cohort must be scored for every arm'
```

`tun01` and `lad01` will have no cohort, so their audits abort — again after the GPU time is
spent. And `worker_second.py:89-92` builds the cohort in every Burgers job's preamble, so
"scored once" also has to be enforced in `worker_tune.py`, not just asserted in prose.

**Fix.** Gate the cohort assertion on a spec flag (`score_cohort: true` only in `fin01`), and
say in §5.1 which jobs build the cohort index at all.

### B4 — `audit.py`'s protocol whitelist rejects three of the eleven sweep arms and the whole step-based rule

`audit.py:73-74` and `:149-151`:

```python
PROTOCOL = dict(epochs=4000, patience=250, batch_size=8, weight_decay=0.0001)
LEARNING_RATES = {0.001, 0.0003}
...
for key, value in PROTOCOL.items():
    assert config[key] == value, (folder, key, config[key])
assert config['learning_rate'] in LEARNING_RATES and config['seed'] in (20260914, 20260915), folder
```

Three independent failures:

* `s-batch32` sets `batch_size` 32 → `assert config['batch_size'] == 8` raises.
* `s-lr3e-3` sets 0.003 → not in `LEARNING_RATES` → raises.
* §4.1 replaces `patience` (epochs) with `patience_evaluations` = 40. Either `patience` stays in
  the config at 250 as a dead key (dishonest — the audit would then certify a constant the
  trainer ignores), or it changes and `assert config['patience'] == 250` raises for **every arm
  in the lane**.

**Fix.** Turn `PROTOCOL` into a per-spec declared constant set, with the new step-based keys,
and widen `LEARNING_RATES` to the pre-registered set `{3e-3, 1e-3, 3e-4}`. Record the change in
§3 and in §8 gate 5.

### B5 — `s-rank` crashes at model construction: `assert trunk_width >= rank`

`families.py:241`:

```python
assert trunk_width >= rank, 'a declared rank needs trunk_width >= rank'
```

`configs/deeponet/small.json` has `trunk_width` 384, `rank` 256. §6's `s-rank` arm sets rank
512 as its single knob, leaving `trunk_width` at 384 → 512 > 384 → `AssertionError` before a
single step. The arm dies, `worker` exits non-zero (`worker_second.py:69-70` `sys.exit(code)`
because `train()` does not pass `allow_failure`), and **the whole job stops at that arm**,
losing every arm after it in the spec order.

**Fix.** Either declare `s-rank` as rank 512 *with* `trunk_width` 512 (and say so, since it is
then a two-knob arm), or relax the assertion. Whichever: this arm must be constructed once
locally under `training_smoke_second.py` before submission. The same smoke would catch B4's
config-key problems.

### B6 — the pinned data was **not** produced by `data.py` + `protocol.json`; `gen01` cannot generate as specified

§3 claims `data.py` (`8d491d5e…`) with `protocol.json` (`212bc898…`) "are exactly the modules
that produced the pinned 128 cases". They are not. From the cluster's own
`pilot-data01/train/index.json`:

```
protocol_sha256  = dec2ba4112f9ee49155f35f593d8f8781fc2ec0f10ed1080eaf74658232385d0
provenance.source_sha256 lists six files, including
    experiments/neural-operator-burgers/protocol-refined.json  dec2ba41…
    experiments/neural-operator-burgers/refine.py              6f2e1a24…
```

`dec2ba41…` is `protocol-refined.json` at `5169c095` (I hashed both blobs). `refine.py:12-17`
monkey-patches `d.PROTOCOL_PATH` to `protocol-refined.json`, reloads `PROTOCOL`, extends
`SOURCE_FILES`, and `refine.py:85` is `def generate(a): configure(); d.generate(a)`. **The
pinned train and validation splits were generated through `refine.py generate`.**

Four concrete consequences:

1. `protocol.json` declares `anchor = {intervals: 4096, dt: 0.0003125}`; `protocol-refined.json`
   declares `dt: 0.00015625`, which is what every pinned record carries. Generating through the
   staged `protocol.json` would use a **2× coarser anchor timestep** than the pinned data.
2. `data.py:307-309` (`read_calibration`) refuses to generate at all:
   `if report.get("protocol_sha256") != sha(PROTOCOL_PATH): raise ValueError("Calibration
   protocol hash mismatch")`. The pinned calibration's `protocol_sha256` is `dec2ba41…`; the
   staged `protocol.json` hashes to `212bc898…`. **`gen01` raises immediately.**
3. `data.py:310-311` additionally requires
   `report['provenance']['source_sha256'] == source_hashes()`, and `source_hashes()`
   (`data.py:47`) keys by path **relative to `ROOT = HERE.parents[1]`**. The lane has copied
   `data.py` to `experiments/ops-tune-deeponet/data.py`, so the keys become
   `experiments/ops-tune-deeponet/data.py` / `.../protocol.json` instead of
   `experiments/neural-operator-burgers/…`, and the equality fails for a second, independent
   reason. On the cluster the staged tree is `<remote>/code/`, so `ROOT` resolves to
   `<remote>` and `NATIVE = ROOT/"experiments/mr-burgers2d/engines.py"` (`data.py:23`) does not
   exist at all.
4. `data.py:329-330` calls `allowed_count(args.split, count)`, and
   `allowed_count` (`data.py:64-67`) caps train at `PROTOCOL["counts"]["train"]` = **128** in
   *both* protocol files. 4608 is refused.
5. `data.py:347` takes the reference setting from `read_calibration`'s `gate["selected"]`, i.e.
   always 4096 / 1.5625e-4. **There is no CLI route to a cheaper candidate**, so §2.3's
   "generates at the finest setting whose measured throughput completes 4608 cases" requires new
   code that overrides the gate selection.

**Fix.** §3 must list `protocol-refined.json` (`dec2ba41…`) and `refine.py` (`6f2e1a24…`) as
inherited byte-identical sources and stage them; `gen_more.py` must go through
`refine.configure()`; and §2.3 must state plainly that it deliberately overrides three declared
generator guards (`allowed_count`, the calibration protocol-hash check, the gate-selected
reference setting) and why. Also note that `case_seed` (`data.py:50-57`) depends only on
`dataset_seed`, `pde_seed_code` and `split_codes`, which are identical in both protocol files —
so the nested-prefix claim in §2.3(1) is sound regardless. That is the one part of §2.3 that
survives untouched.

### B7 — the matched cohort's index hash changes the moment the train index changes

`prepare_diagnosis_cohort.py:94` writes `train_index_sha256=sha(args.train_index)` **into the
cohort `index.json`**. `audit.py:220` then asserts

```python
assert sha(index) == FNO_COHORT_INDEX, 'cohort index must be the FNO job\'s'
```

Any job that builds the cohort with the extended 4608-case train index produces different
`index.json` bytes, a different sha256, and the assertion fails — voiding the cohort and, via
`audit.py:329`, the whole audit. §8 says nothing about this.

**Fix.** Either build the cohort against the *pinned* 128-case index (the disjointness check at
`prepare_diagnosis_cohort.py:77-79` then covers only 128 of 4608 training cases — say so), or
build it against the extended index and record the new cohort index hash as a lane constant
while asserting the *case field contents* match the FNO job's, which is what actually matters.
Note the second option is strictly better science: with 4608 training draws, the
seed-and-input-content disjointness check should run against all of them.

---

## MAJOR

### M1 — the 114 s/case figure is one record, and the parity cost is 14 % higher than stated

§2.3: "`wall_seconds_including_first_compile` ≈ **114 s per case** … 4608 cases at that setting
is ≈ 146 GPU-hours". 114.367 s is `burgers-train-00000`'s value and nothing else. Over all 128
pinned train records (read from the cluster index):

| statistic | s |
|---|---:|
| min | 104.08 |
| median | 122.06 |
| **mean** | **130.33** |
| max | 179.51 |
| sum (128 cases) | 16 682 |

4608 × 130.33 s = 600 558 s = **166.8 GPU-hours**, not 146. The design's own arithmetic is
self-consistent with 114.367 (4608 × 114.367 = 146.4 h), which is how I know the number was
read off record 0. This is exactly the "hand-typed where it should be derived" failure the
project's rules name; the parent lane already had one (§A6, "561 s where the audited value is
552 s").

**Fix.** Derive it — mean, median and max over the 128 records — and quote the mean. The
conclusion ("more than this lane's entire budget") is unchanged and gets stronger.

### M2 — T0 cannot fail, and it stops being vacuous exactly where T3 would be met

The measured difference between a candidate reference and the pinned anchor, **in the same
metric the lane grades with** (`protocol.json`: "maximum over requested times of interior l2
difference divided by anchor initial interior l2 norm" ≡ `audit.py:54-59`
`fixed_initial_errors`), worst over the 8 calibration cases at output 256:

| candidate | vs anchor (raw) | design's §2.3 column |
|---|---:|---:|
| 1024, 3.125e-4 | **0.187 %** | 0.278 % |
| 512, 6.25e-4 | **0.434 %** | 0.525 % |
| 256, 1.25e-3 | **0.915 %** | 1.006 % |

T0 declares `c-new128` and `c-pinned128` to agree if their validation means differ by less than
**10 % relative**. At the parent's error scale (14.79 % mean) that band is **1.48 % absolute** —
three times the largest label perturbation the cheapest admissible setting can introduce. T0
**cannot fail**; it is vacuous by construction, which is precisely the failure the brief says
must not recur.

Worse, the failure is not symmetric. A ~0.43 % systematic label offset is negligible against a
15 % error and decisive against a 2 % one. T3 declares the DeepONet competitive if its median is
within 1.5× `fno-large`'s 1.8054 %, i.e. **≤ 2.708 %**. At that error the 512-setting label
offset is ~16 % of the signal, and the arm would be partly fitting the discretisation difference
between its training targets and the pinned validation targets. **T3 and the cheap-protocol
justification are mutually exclusive**: the design is only safe in the regime where it fails.

**Fix.** (a) Re-express T0 as an **absolute** band tied to safeguard 2's measured per-case
difference — e.g. "agree iff |Δ mean| ≤ 2× the measured median per-case target difference". (b)
Pre-register the converse tripwire now: *if any arm's validation mean falls below N× the
measured target-protocol difference (N declared here, e.g. 5), that arm's number is reported as
protocol-contaminated and is not used for T3.* (c) Prefer the 1024 setting if the profile
permits it; its 0.187 % is 2.3× safer and §2.3's own fallback rule already prefers it.

### M3 — §5.3's 2 % composition threshold is below one standard error, measured on this lane's own data

I recomputed the per-case validation errors from `ops-deeponet-b2d/runs/don01/audit.json`
(32 cases, `case_maximum_errors`). Because every arm is scored on the same 32 cases, the
relevant noise for an arm-vs-arm comparison is the **paired** standard error:

| comparison | mean diff (pp) | paired SE (pp) | SE as % of baseline mean |
|---|---:|---:|---:|
| `don-medium` − `don-small` | +0.112 | 0.292 | **2.0 %** |
| `don-refine` − `don-small` | +0.977 | 0.330 | **2.2 %** |
| `don-medium` − `don-refine` | −0.864 | 0.428 | **2.7 %** |
| `don-large` − `don-small` | +3.433 | 0.535 | **3.6 %** |

(Unpaired SE of a single arm's mean is 1.78–2.01 pp, i.e. 10.7–12.7 % relative — relevant to
T0/T2 if anyone quotes them unpaired.)

§5.3 admits a knob into `tuned` if its arm beats `base` by **≥ 2 % relative** — at or below one
paired SE. For a knob with no real effect, P(observed gain ≥ 0.67 SE) ≈ 0.25. With ten
one-factor arms, **the expected number of pure-noise knobs composed into `tuned` is ≈ 2.5**.
The design's own escape hatch ("if the composed arm is worse than the best single-knob arm, the
best single-knob arm is selected") is then the likely outcome, and the lane will have spent
`lad01` running a `tuned` arm built from noise.

T2's 10 % bar (≈ 3–5 paired SE) is defensible. T1's 0.7× is ≈ 8–15 SE and is fine.

**Fix.** Raise §5.3's composition threshold to **≥ 5 % relative** (≈ 1.5–2.5 paired SE) and
state the paired SE in the pre-registration, computed as above from the parent's saved fields,
so the number is derived and not chosen. Alternatively keep 2 % but declare that `tuned` is a
*screening* composition and that T2 is judged only on `tuned` vs `base` at 10 %.

### M4 — the sweep arms get 1500 s and `base` gets 3000 s, so §5.3's comparison is not one-factor

§6: `base-top` runs at 3000 s in `tun01`; the ten one-factor arms run at **1500 s**. §5.3 then
composes "every knob whose one-factor sweep arm improved the selection metric by at least 2 %
relative over `base`". That comparison has two factors: the knob, and half the wall budget. The
budget factor is biased **against** every knob, so the 2 % screen is not merely noisy (M3) but
systematically conservative by an unmeasured amount.

Sizing it: the parent's `don-small` reached 15.6 optimisation steps/s (539 epochs × 16
steps/epoch ÷ 552 s). At the 4608 rung, 3000 s ≈ 47 000 steps ≈ 81 epochs; 1500 s ≈ 23 400 steps
≈ 41 epochs. The stopping rule needs 20 000 stale steps (§4.1), so a 1500 s arm essentially
**cannot early-stop** and is purely budget-limited. The sweep is therefore measuring "which knob
learns fastest in 41 epochs", and `base` is measured after 81.

**Fix.** Add a `base-1500` arm at the top rung (cost: 1500 s, ~4 % of `tun01`'s budget) and make
§5.3 compare each sweep arm against **`base-1500`**, not `base-top`. Keep `base-top` for T2 and
the ladder.

### M5 — the in-job plateau-vs-cosine branch makes every later arm a two-factor arm, and §6's own table contradicts the rule

§6: "`s-lr3e-3`, `s-lr3e-4` and every arm after `s-cos` are run with whichever of `plateau` or
`cosine` won in `s-cos` versus `base`". As a pre-registration mechanism this is legitimate — the
rule, the inputs and the tie-breaking are fixed in advance and recorded in `selection.json`. Two
problems with what it does:

1. **If `cosine` wins, every subsequent arm differs from `base` in two ways** (its knob *and*
   the schedule). §5.3 then composes knobs whose measured gain includes the schedule gain, and
   `tuned` double-counts it — while `s-cos` is *also* admitted as a knob in its own right.
2. §6's arm table hard-codes the schedule as "cosine" for `s-lr3e-3` and `s-lr3e-4`, directly
   contradicting the rule two paragraphs below it.

**Fix.** Compare every post-`s-cos` arm against a `base` run **on the winning schedule** (this is
the same fix as M4: one extra 1500 s reference arm on the winning schedule), and delete
"cosine" from the two learning-rate rows of the §6 table.

### M6 — `c-pinned128` is not "the inherited configuration under this lane's stopping rule"; three things change, and the direction of the bias is the opposite of §3.1's claim

§4.1 declares one change (patience in steps) and says the `c-pinned128` / `don-small` difference
"is the measurement of what the stopping rule alone changed". At 128 cases (16 steps/epoch)
three things change at once:

| | inherited (`don-small`) | this lane (`c-pinned128`) | factor |
|---|---:|---:|---:|
| validation cadence | every epoch | every 500 steps = **31.25 epochs** | 31× coarser |
| stopping patience | 250 epochs | 20 000 steps = **1250 epochs** | 5× longer |
| `ReduceLROnPlateau` patience | 20 epochs (`train.py:111`) | 4000 steps = **250 epochs** | 12.5× longer |

The scheduler change is the one nobody named. The parent's audit found every arm had sat at the
1e-5 learning-rate floor for 146–209 epochs before stopping; under the new cadence the LR can
halve at most ~10 times in a 3000 s run at this rung and will likely never reach the floor. So
`c-pinned128` is a different *optimisation* as well as a different *stopping rule*.

The cadence change also has a sign. Best-of-*k*-evaluations is monotone in *k*: sampling the
validation score 31× less often can only make the best-found score **worse or equal**. §3.1
names `c-pinned128` as "**the like-for-like row**" for the paper and §5.1 claims selection noise
biases "in the baseline's favour, the conservative direction". For this row that is backwards:
the paper's like-for-like DeepONet number would be handicapped by a bookkeeping change, making
the baseline look worse than the published `don-small`, which is the *anti*-conservative
direction for a paper being reviewed on whether it strawmanned its baselines.

**Fix.** Report `don-small` (the published, audited number) as the like-for-like row and
`c-pinned128` as the stopping-rule control beside it; state both in the comparison table. And
either hold the plateau-scheduler patience at its inherited 320 steps for the 128 rung, or
declare the scheduler change as a second named change in §4.1 with its factor.

### M7 — per-epoch checkpoint I/O biases the fixed-compute ladder toward "data-limited"

`train.py:152-159` writes `history.json`, `best.pt` (when improved) and **`last.pt`
unconditionally, once per epoch**. For `don-small` (2 569 349 float32 parameters) a checkpoint is
model + AdamW state ≈ 31 MB; two writes ≈ 62 MB per epoch.

* At 128 cases an epoch is 16 steps ≈ **1.02 s** (539 epochs / 552 s measured) → ≈ 60 MB/s of
  sustained Lustre writes for the whole run.
* At 4608 cases an epoch is 576 steps ≈ **37 s** → ≈ 1.7 MB/s.

Under §5.2's "at the same wall budget", the 128 rung therefore spends **36× more of its budget on
checkpoint I/O per unit of compute** than the top rung. T1 ("data-limited iff the top rung's mean
is ≤ 0.7× the 128 rung's") is measured across exactly that asymmetry, in the direction that makes
data look like the lever. In the parent lane every arm had 128 cases so this cancelled; here it
does not.

**Fix.** Move the `history` / `last.pt` write to the evaluation cadence (every 500 steps) — which
the step-based refactor makes natural anyway — and record measured I/O-excluded training seconds
per arm so the ladder can be read without this confound.

### M8 — the step-based refactor silently breaks epoch-indexed records in two files

`train.py:193` `batched_selection_score=history[restored['epoch']]['validation']['mean_case_max']`
and `train.py:195` `best_epoch=restored['epoch']` index `history` **positionally by epoch**.
`audit.py:93/146` (`assert len(history) == result['epochs_completed']`), `:119`/`:174`
(`best = history[result['best_epoch']]`), `:175` (exact equality of that record's score with the
reported `batched_selection_score`) and `:186-188` (`train_loss_at_best`,
`epochs_at_minimum_learning_rate`) all depend on it.

If `history` becomes per-evaluation while `epoch` stays per-epoch, `history[best_epoch]` returns
the **wrong record** — and `audit.py:175`'s exact-equality assertion is the only thing that would
notice, turning a silent mis-index into a crash at the end of the job. §3's one-line description
of the `audit.py` change does not cover it.

**Fix.** Rename the record index to `evaluation` end to end (`train.py`, `audit.py`, the report
generator), keep `epoch` as a recorded field rather than an index, and pre-register §8 gate 6's
tolerance against the *evaluation* record.

Related: `audit.py:188` `epochs_at_minimum_learning_rate` counts `learning_rate <= 1.0000001e-5`
— under a cosine schedule to 1e-6 that count is meaningless and will read as "spent" for every
cosine arm. Redefine or drop it for cosine arms.

### M9 — the lane does not measure the one number that answers Q2

The parent's own audit identified data limitation from the **train-versus-validation gap**:
4.28–8.59 % RMS on training against 14.79–18.22 % on validation. That gap closing as the rung
grows is the direct evidence for Q2; the validation mean alone cannot separate "more data helped"
from "more steps at this dataset size would have helped too". §5's criteria do not mention it and
§4 does not pre-register recording it.

**Fix.** Pre-register, per rung: the training-set error of the selected checkpoint under the
same batch-1 metric, the validation error, and their ratio. `train.py` already records
`train_mean_squared_relative_error` per epoch, but that is the *training loss* on the last epoch,
not the selected checkpoint's training error under the reported metric — it needs one extra pass.

### M10 — nothing measures how much closer validation-32 gets to the training set as it grows 36×

`LAB-LOG.md` (2026-09-21) records for our own 4608-trajectory draw: "nearest normalised distance
to training 0.053–0.099 vs a 0.131 median training nearest-neighbour spacing". With 4608 operator
training draws instead of 128, each validation case's nearest training neighbour moves in by
roughly 36^(1/5) ≈ 2.0× in a 5-parameter family. A hostile reviewer will say the top rung's gain
is partly a train/test proximity effect in parameter space, not generalisation — and there is no
measurement in the design to answer with.

**Fix.** At each rung, record the distribution of normalised nearest-neighbour distance from each
validation-32 and cohort-8 case to the training set, and report it beside the ladder. The script
already exists: `reports/checks/2026-09-21-burgers-train-eval-overlap.py`.

### M11 — the pooled path drops `dataset.load_pair`'s disjointness checks

§4.4 says `train.py --pool` "asserts that the pool's recorded index sha256 equals the
`--train-index` it was given". That replaces `dataset.load_pair` (`dataset.py:115-135`), which
does considerably more: per-case sha256 verification, duplicate-case-archive detection
(`:53-55`), duplicate-physical-input detection within a split (`:66-69`),
**train/validation case-id, seed and content-hash overlap** (`:121-131`), and field-contract
equality (`:132-134`). An index-hash assert proves the *index* is the one `gen01` produced; it
proves nothing about the 4608 field files or about overlap with validation.

**Fix.** `build_pool.py` must run `dataset.load_pair` once (it already reads everything) and
record the outcome in `cases.json`; `train.py --pool` asserts against that record. Then say in
§4.4 which checks ran once at pool-build time rather than per arm.

Cost note in the same place: without the pool, `dataset.load_index` reads each case through
`sha256` plus `read_case` twice (`:50`, `:57`, `:67`) and `load_pair` a third time (`:127`) —
about four passes over 17.0 GB per arm. I verified the per-case file size on the cluster
(`burgers-train-00000.npz` = 3 699 802 B), so 4608 cases is **17.0 GB**; the design's figure is
right.

### M12 — "205× supervised states" is not the asymmetry a reviewer will attack; the real one is unstated

§2.2 already caveats that the two state counts "are not the same kind of object". It does not
state the two things that actually decide the argument:

1. **The tasks are not comparable.** Our NM-ROM consumes the governing equations at solve time —
   a projected residual with an empirical-quadrature rule — while the operator receives only
   $u(\cdot,0)$ and $\nu$ and must emit the whole trajectory. Giving the operator 4608
   trajectories does not make it a fair fight; it removes one specific confound. A reviewer who
   reads §2 as "at parity the comparison is fair" has been led there by the section's framing.
2. **Generation cost points the other way.** The 128 operator cases cost 16 682 GPU-seconds at
   the 4096 anchor (mean 130.3 s each, measured above). Our 4608 NM-ROM trajectories are 50-step
   solves at 256². The operator side has 36× fewer trajectories that each cost roughly two orders
   of magnitude more to produce. "The operators only got 128 cases" is partly a consequence of
   generating operator data at a 32×-finer reference than our own model's training data — which
   is also the honest justification for §2.3's cheaper protocol, and is stronger than the
   budget argument the design currently gives.

**Fix.** Add both to §2, and make the headline sentence "36× more trajectories, generated ~100×
more cheaply each, for a task that also receives the PDE at solve time".

### M13 — `gen_more.py` does the lane's most load-bearing work and no gate tests it

§8 gate 4 requires `gen01`'s cases 0–127 to reproduce the pinned "generation descriptors and
input fields bitwise". Both are **protocol-independent**: descriptors come from
`engines.params_draw(case_seed(...))`, and `data.py:355-357` already asserts
`np.array_equal(fields[0], engines.initial(args.intervals, physical))` at *any* reference
setting. So gate 4 passes whether the job used 4096/1.5625e-4, 512/6.25e-4, the wrong protocol
file, or a different anchor entirely. The one gate aimed at the new generation path cannot detect
the failure that path is most likely to have.

**Fix.** Add a gate on the recorded reference setting itself: `gen01`'s index must record
`reference_setting` and `protocol_sha256`, the log must print them, and the report must carry
them. And state safeguard 2's measurement as a *gate* with a pre-registered bound (see M2), not
only as a reported number.

### M14 — "the protocol's declared `future_train_prefixes` extended" is wrong

Both `protocol.json` and `protocol-refined.json` declare `future_train_prefixes: [512, 1024]` and
`counts.train: 128`. The rungs 2048 and 4608 are not declared anywhere in the protocol, and 4608
is 4.5× the largest declared prefix. §2.3(1) presents the ladder as sanctioned by the protocol.

**Fix.** Say what is true: the protocol declared prefixes up to 1024 and a hard count cap of 128;
this lane extends past both, deliberately, and records that as a declared deviation.

### M15 — `cluster/collect.py` will try to transfer the 17 GB pool

`cluster/collect.py:30`:

```python
EXCLUDES = ['--exclude=*/last.pt', '--exclude=*.solver.npz', '--exclude=cache', '--exclude=tmp',
            '--exclude=data/train', '--exclude=data/refinement', '--exclude=code/__pycache__']
```

`data/train` is excluded, but §4.4's pool (`input.npy`, `target.npy`, `parameters.npy`) is a new
directory and is not. Collection would tar and scp 17 GB. §3's change list for `collect.py` says
only "lane path and namespace".

**Fix.** Add the pool directory (and `data/pool`, or whatever it is called) to `EXCLUDES`, and
state in §8 gate 9 what the archive is expected to contain. Note also that this lane produces up
to 21 arms against the parent's 4; at 32 saved validation predictions × 3.2 MB per arm the
archive is ~5× the parent's, chunked into 45 MiB Git-tracked parts.

---

## MINOR

**m1 — §2.3's margin column mixes two definitions and its derivation is undeclared.** The three
candidate rows (2.781e-3, 5.251e-3, 1.006e-2) are, exactly, max over the 8 calibration cases of
`difference_from_anchor + anchor_difference_sum` — the candidate's distance from the anchor
*plus* the anchor's own refinement uncertainty. The pinned row (9.588e-4) is the anchor term
alone. The column is labelled "worst empirical margin vs the anchor", which is the raw
`difference_from_anchor`: **1.874e-3 / 4.344e-3 / 9.149e-3**. Quote both, and use the raw column
for the T0 discussion (M2), since that is the quantity safeguard 2 measures.

**m2 — §4.1's justification of the 4000-step scheduler patience is arithmetically false.** "the
inherited 20 epochs at 128 cases is 320 steps … 4000 steps is the closest schedule-shaped
equivalent at the top rung." At the top rung 20 epochs is 20 × 576 = **11 520 steps**, not 4000.
4000 steps is 6.9 epochs there and 250 epochs at 128 cases. The constant is arbitrary (it is 1/5
of the stopping patience, against the inherited 1/12.5); say that instead of inventing an
equivalence. Everything else in §4.1 checks out: 128/8 = 16, 4608/8 = 576, 40 × 500 = 20 000,
20 000/16 = 1250, 1250/250 = 5, 8 × 500 = 4000, 20 × 16 = 320.

**m3 — `s-pool16` is a capacity arm disguised as a bottleneck arm.** With `levels` 3 and width
48, the branch output has 192 channels, so `branch_hidden` (`families.py:251`) becomes
`Linear(192·16·16 = 49152, 384)` ≈ **18.9 M parameters**, against 1.18 M at `pool_bins` 4. The arm
is ~21 M parameters total — larger than the largest arm of any other family in the comparison
(FNO 17.9 M) and outside the parent's declared 2.6–10.3 M range. Declare it as a capacity+
bottleneck arm, or cap it. Separately, `F.adaptive_avg_pool2d(33 → 4 or 16)` does not divide
evenly; `families.py`'s "exact adaptive average pool" docstring is already loose and gets looser
at 16.

**m4 — `131072` cannot be regenerated by the report.** §2 promises "§6 of the report regenerates
it and the report never re-types it". I confirmed `4608`, `4032`, `576`, `50` (`num_steps`) and
`16384` (`max_snaps`) are all in
`reports/2026-09-10-historical-poisson-and-burgers-cost-audit.json`
(`current_burgers_checkpoint_cfg`). **`131072` is not** — it exists only as prose in `LAB-LOG.md`
line 11365, where it is also identified as a *cap*: the incumbent carries 131072 of the 235008
stride-1 codes. Either read it from the checkpoint and pin its sha256, or mark it in the table as
a prose-sourced cap (the bank row already carries a "≤"; the head row should too).

**m5 — §5.3 says "the three learning rates".** There are two learning-rate *arms* (`s-lr3e-3`,
`s-lr3e-4`) plus `base`'s 1e-3.

**m6 — `s-trunk` varies two knobs** (`trunk_width` 384→768 and `trunk_layers` 3→4) inside a
one-factor sweep whose composition rule assumes one knob per arm.

**m7 — §8 gate 6's "batch-8 selection score" is wrong for `s-batch32`.** `train.py:140` passes
`config['batch_size']` to `evaluate`, so that arm's selection score is a batch-32 score. Restate
the gate as "the batched selection score at the arm's own batch size".

**m8 — `check_inherited.py` cannot prove §3's claim and is not present in this lane.** The
inherited copy hard-codes `FORK = 'ea812685'` and `SOURCE = 'experiments/no-second'`
(`check_inherited.py:22-24`) and `blob()` resolves against a single source tree. This lane forks
`ops-deeponet-b2d` at `306c939d` **and** claims byte-identity against a second tree
(`experiments/{neural-operator-burgers,mr-burgers2d,separable-decoder}` at `5169c095`). It needs
a two-source rewrite, and §3's `checks/inherited-sources.json` does not yet exist — `checks/` is
empty. I verified the four claimed generator hashes by hand and they match the staged copies:
`data.py 8d491d5e…`, `protocol.json 212bc898…`, `engines.py 820039e5…`, `sep_common.py
a74f0279…` — but see B6 for why that file *set* is the wrong one.

**m9 — T5's premise is already false.** The campaign brief and T5 rest on "every operator error
currently exceeds the mesh's own discretisation error (4.0265 %)". `unet-medium`'s **worst** case
on validation-32 is **3.9622 %** (parent report §2), already below it, and every non-DeepONet
arm's mean and median are below it. Report that in the same breath as T5, or the appendix's
argument is being defended against a number that has already fallen.

**m10 — namespace and cache constants.** `worker_second.py:22`
(`assert str(ROOT).startswith('/cluster/.../opsdon_20260922/')`), `cluster/stage.py:26-28`
(`LANE`, `NAMESPACE`, `CACHE`) and `cluster/collect.py:28-29` all carry the parent's lane path
and namespace. §3 covers `stage.py` and `collect.py` but not the worker's assertion string; the
parent lane had to record that same edit explicitly.

**m11 — disk.** `/cluster/tufts/paralab` is at 91 % with **441 GB free** (checked). 4608 cases ×
3 699 802 B = **17.0 GB**, plus ~91 MB of solver sidecars, plus the pool's ~17 GB = ~34 GB, plus
per-job archives. Fine, but only because §3 replaces the sbatch `cp -r` of the cache
(`stage.py` preamble) with symlinks — if that change is dropped, each of three jobs copies 17 GB
and the share loses 51 GB. §8 should state the expected footprint.

**m12 — host memory.** `train.py:49-55` builds `raw = torch.cat(...)` (~4.9 GB) and computes
`y.square().mean()` on the 14.6 GB float64 target tensor, materialising a full 14.6 GB temporary:
peak ≈ 37 GB on top of nothing else. The sbatch template requests `--mem=180G`, so this is safe —
do not reduce it, and note that §4.3's `per_time` output scale needs the same temporary.

**m13 — `fin01`'s cross-job cohort scoring needs checkpoints that gate 9 deletes.**
`evaluate_cohort.py:25` scans only the current job's `out/` for `--pattern`. Scoring "every
checkpoint this lane produced" means restaging `tun01`/`lad01` checkpoints into `fin01`'s `out/`
via `stage.py`'s `checkpoint_source` path. Say so in §6 and keep those remote directories until
`fin01` completes, which conflicts with gate 9's "delete the remote job directory afterwards".

---

## What is sound (one line each)

* §2.1's counts: `train/index.json count = 128`, `validation count = 32`, `refinement count = 8`,
  gate `calibrated`, empirical budget 1e-3, index shas `5333584b…` / `468b9e70…` — all verified
  on the cluster. 128×5 = 640, 128×6 = 768, 4608/128 = 36, 131072/640 = 204.8 → "205×",
  4608×51 = 235008: all correct.
* §2.2's provenance: `n_traj 576`, `hfit_n_traj 4608`, `hfit_extra_traj 4032`, `hfit_extra_seed
  1000`, `num_steps 50`, `max_snaps 16384` all present in the cited cost-audit JSON; jobs 2835788
  / 2837431 and the `sample_params` ≡ `params_draw` identity confirmed in `LAB-LOG.md`
  (2026-09-21).
* §2.3's nested-prefix argument (safeguard 1) is correct: `case_seed` depends only on
  `dataset_seed`, `pde_seed_code` and `split_codes`, identical across both protocol files, and
  `data.py:355-357` guarantees the initial field is the analytic one at any reference setting.
* §2.3's work-proxy column is exactly $L^2 n_t$ and matches the cluster's `work_proxy` for the
  anchor (2.684e10) and arithmetically for the three candidates.
* §5.1's selection rule, the "report every arm" and "name the best-worst-case arm" clauses, and
  the sealed-final-cohort discipline are inherited unchanged and are the right call.
* §6's job arithmetic: 3 × 3000 + 10 × 1500 = 24 000 s = 6.7 h of training inside a 10 h wall,
  with ~1 h of pool build, data verification and cohort build — feasible. `lad01` at ≤ 6 arms
  works out **only** if `c-new128` is treated as `base`@128 (512, 2048 for `base`, four rungs for
  `tuned` = 6); say that explicitly or the count is 7.
* Truncation ordering is better than it looks: T1's headline comparison (`c-new128` vs
  `base-top`) is entirely inside `tun01`, so losing the sweep does not lose the lane's
  first-priority question. `gen01` remains the single point of failure for everything.
* §7's timing abstinence is correct and unambiguous.
* T6's honest-negative clause is the right pre-registration.

## The two things I would change before anything else

1. **Fix B1–B7 and run one local smoke that constructs every arm's config** (`s-rank` would fail
   immediately) **and runs `audit.py` against a two-arm synthetic attempt.** Five of the seven
   blockers are gates that crash after the GPU time is spent; all five are catchable in minutes
   on the GB10.
2. **Rewrite T0 as an absolute band and add the converse tripwire (M2).** As written, the one
   criterion guarding the lane's biggest methodological change cannot fail, and it stops being
   vacuous only in the regime the lane is trying to reach.

## Reachability of the pre-registered criteria, stated plainly

| criterion | reachable? |
|---|---|
| T0 (10 % protocol control) | **Cannot fail** — see M2. Vacuous by construction. |
| T1 (0.7× at the top rung) | Reachable and well-sized (≈ 8–15 paired SE); confounded by M4, M7, M10. |
| T2 (10 % for `tuned`) | Reachable, ≈ 3–5 paired SE. Fine. |
| T3 (within 1.5× `fno-large`) | Requires median 11.7522 % → ≤ 2.708 % (4.3×) **and** worst 54.7444 % → ≤ 9.574 % (5.7×). Near-certainly unreachable; say so in §5 rather than letting the reader discover it. And if it *is* reached, M2 says the number is protocol-contaminated. |
| T4 (cohort, < 1.8671 % worst) | Current best DeepONet cohort worst is 16.0229 %. Effectively a formality. |
| T5 (< 4.0265 %) | Premise already false for `unet-medium` (3.9622 % worst) — see m9. |
| T6 (honest negative) | The likely outcome, and correctly pre-registered. |
