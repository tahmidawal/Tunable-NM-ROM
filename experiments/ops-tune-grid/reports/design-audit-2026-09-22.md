# Independent design audit of `ops-tune-grid/DESIGN.md` — 2026-09-22, before `gen01`

Commissioned with the adversarial brief in `design-audit-prompt.txt` after Codex's read-only
sandbox failed to read any file (`codex-design-audit-2026-09-22.md`). Performed by an
independent Claude subagent that built and ran every proposed training arm, re-derived the
seed space and the calibration gate from their own files, and verified the published
comparison numbers.

**Disposition: every finding accepted, none rejected.** The itemised response, with what was
changed for each, is `DESIGN.md` §A3. The auditor's two deliveries are reproduced below
verbatim; the second is the fuller one and supersedes the first where they differ.

Two notes on what this audit is and is not:

- It audited a **moving tree** — the lane's code was being written while the audit ran, and
  the auditor says so. Several of its findings are DESIGN-vs-code contradictions that existed
  because implementation ran ahead of the pre-registration. The right order is to freeze and
  quote a SHA256 before commissioning; that was traded for speed against the 09-25 deadline,
  and it cost the certainty that the audited text is the text a reader will see.
- Three items it **could not verify** are recorded in §A3 as gaps rather than silently
  dropped: the two NM-ROM rows of §3, the seeds of the timing panel's six development cases,
  and A100 epoch rates at the larger training-set sizes.

---


## Auditor's findings, verbatim

### Blockers

**B1 — `gen01` cannot finish 4608 cases in its declared budget, because `data.solve` spins a
2-second GPU burn-in *per call*.** `data.py:155-159` runs `while time.monotonic() < until:`
for **2.0 s** before `perf_counter` starts, on every `solve()`. `gen_bank.py:153` calls
`data.solve` once per case inside the loop. Real cost = 4608 x (2.0 + solve). The calibration
median for `1024/3.125e-4` is 4.802 s. Take 4.8 s: **4608 x 6.8 = 31 334 s = 8.70 h**, plus
the ~130 s reproduction solve. Against that, `specs/gen01.json` set `generation_seconds:
27000` (7.5 h). Result: ~3 300-3 800 cases, and `write_index` skips any prefix > the count
generated, so **`index-04608.json` is never written** and `ladder01`'s four 4608 arms all name
a file that does not exist. DESIGN 5.1's "~7 h" omits 2.56 h of deliberate idle spin.

**B2 — DESIGN 3.2's calibration table is quoted from the superseded, *failed* calibration.**
`artifacts/calibration01/calibration-index.json` records `protocol_sha256 = 212bc898...` =
`protocol.json`, whose anchor is `4096 / 3.125e-4`. Its gate at output 256 is `passing: false,
selected: null` — *no candidate passed*, which is why `protocol-refined.json` exists. The
pinned 128 training cases were generated under `protocol-refined.json` (anchor
`4096 / 1.5625e-4`) via `refine.py`. Every number in the table is correctly transcribed *from
calibration01* — but calibration01 is the wrong file. The right gate is mirrored in
`checks/refinement02-reference-audit.json`. At output 256, against the **actual** pinned
anchor:

| setting | worst dev | median | gate margin |
|---|---:|---:|---:|
| 256, 1.25e-3 | 9.149e-3 | 6.686e-3 | 1.006e-2 |
| 512, 6.25e-4 | 4.344e-3 | 3.154e-3 | 5.251e-3 |
| **1024, 3.125e-4** | **1.874e-3** | **1.352e-3** | **2.781e-3** |
| 4096, 1.5625e-4 (anchor) | 0 | 0 | **9.588e-4** |

The headline bound is **2.781e-3 = 0.28 %**, not 3.06e-3 = 0.31 %; the anchor's own refinement
margin is **9.588e-4**, not 1.386e-3. The row labelled "4096, 3.125e-4 *(the anchor)*" is not
the anchor at all. **The conclusion survives** (1.874e-3 vs 1.873e-3), so this is a sourcing
failure, not a science failure — but the design names a superseded artifact by filename and
DESIGN 9.9 says no number is hand-typed.

**B3 — DESIGN 5.3 and `specs/ladder01.json` contradict each other on which configuration the
ladder trains.** 5.3 says `grid01`-selected; the spec names the published configs and its own
`design` string says so. Both choices are defensible; the pre-registration saying one and the
spec saying the other is not, and it is load-bearing for T2. **And it changes the schedule:**
if the ladder uses published configs it has no dependency on `grid01`, so the two can be
submitted concurrently — collapsing the lane from ~47.5 h serial to ~29 h, roughly **18 hours
recovered** against a 09-25 deadline.

**B4 — the ladder's learning-rate schedule and early stopping are epoch-indexed, so they
differ ~35x across rungs.** `train.py:111` fixes `ReduceLROnPlateau(factor=.5, patience=20,
min_lr=1e-5)` and `train.py:164` early-stops on `config['patience'] = 250`, both counted in
**epochs**. From the published per-epoch rates:

| family | s/step | epochs at 128 in 4000 s | epochs at 4608 in 4000 s |
|---|---:|---:|---:|
| FNO (`fno-large`, 0.271 s/step) | 0.271 | ~920 | **~26** |
| U-Net (`unet-medium`, 0.096 s/step) | 0.096 | ~2600 | **~73** |
| Transolver (`tsol-small`, 0.115 s/step) | 0.115 | ~2200 | **~60** |

The 128 rung anneals through several plateau reductions to the 1e-5 floor; the **4608 rung
fires the plateau at most once and never reaches the floor**, and early stopping is
mathematically unreachable there. So a rung differs from its neighbours in data size *and*
effective LR schedule *and* checkpoint-selection granularity, not "in the data the same number
of gradient steps is drawn from" as 5.3 claims. Combined with the ~25 % step advantage, the
net bias is *unknown*, which is worse than a known one. **This is the ladder's largest
scientific defect and the DESIGN does not mention it.**

### Majors

**M1 — `gen_bank.py` loads `data.py` under `protocol.json`, not `protocol-refined.json`.**
[Resolved on inspection: this lane's `protocol.json` *is* the refined content, sha `dec2ba41`,
which is the `protocol_sha256` the published index records. What remains is that the published
run reached it through `refine.py`, so its `source_sha256` keys the same content under a
different filename. A note is recorded in the generation report.] The two protocol variants
differ **only** in `anchor.dt` and a comment; everything `gen_bank` consumes is identical, and
`case_seed('train',0) = 428288116` under both, matching the published record.

**M2 — the pre-registration and the code disagree about the reproduction gate.** DESIGN
3.2/5.1/9.4/10 say the gate asserts **bit-identity** and that failure voids `gen01`.
`gen_bank.py:138` raises only on `arrays_equal`; `bytes_identical` is recorded and never
enforced. The code is right and the document is wrong. `np.savez` pins the zip `date_time` to
`(1980,1,1,0,0,0)` so npz output *is* byte-reproducible across runs — but the published cache
was written under a different numpy, so a numpy change would void a byte gate for
non-numerical reasons.

**M3 — G1's tripwire is set ~5x above the value already known, so it cannot fire.** The
refined calibration already puts the 1024 setting's worst deviation at **0.187 %**. G1's
threshold is 1.0 %. Either lower it to something the data could plausibly cross, or re-label
G1 as a measurement rather than a gate.

**M4 — G2 has no resolving power at the rung it is run on.** Correctly constructed, but a
single-seed A/B whose expected effect (<=0.28 pp of label noise, most of it common-mode) is
below the measured seed noise in this exact setup: `no-second`'s `ctrl-medium-seed2` moved
`unet-medium`'s validation mean by +0.0435 pp, median by -0.153 pp and worst by -0.321 pp on
one seed change. A null G2 licenses nothing. It also runs at 128 cases, the rung where the
effect matters least, and is then used to license attribution at 4608.

**The strongest reviewer objection the design does not answer:** *"You changed the training
distribution and the data size at the same time, and your only control is at the size where
the change matters least."* The design's answer should be the one it does not make — that
evaluation is against the unchanged pinned reference, so fidelity-induced bias is inside the
reported number, not outside it.

**M5 — the whole grid is confounded with the wall budget, in the direction that will read as
"tuning didn't help".** Every published arm ended `stopped_by_wall_budget` with `still
improving: yes`. A 1.3-2.1x more expensive arm at a fixed 3000 s is scored with proportionally
fewer epochs, and "worse" will mean "less trained". Section 6 discusses this asymmetry only
across dtypes, never within a family's grid.

**M6 — the knobs section 1 promises are mostly not in the grid.** Section 1 names "learning
rate, weight decay, batch size, scheduler and patience". Section 5.4 varies learning rate in
**one family at one value**, scheduler in three, and **never varies weight decay, batch size
or patience at all**. The FNO and the Transolver get **no learning-rate arm**, despite
lower-lr retraining being the only tuning the published lanes did and it having *helped* both
of them while *hurting* the FNO. Learning rate is the highest-value knob and it is missing
from two of three families.

**M7 — `fno-epochmatch` will contaminate the FNO's selection.** Section 4.2 selects `argmin
over that family's arms`, and `fno-epochmatch` sits in the FNO pool with 2.9x the wall. It
will very likely win, and T1 would report a **budget** effect as a **tuning** effect.

**M8 — `tsol-cosine` changes two variables** (`warmup_epochs: 0` where every other Transolver
config sets 10).

**M9 — T3 and T5 are already satisfied by published arms, and T5 violates section 2's own
caveat.** T5 compares against 1.8891 %, the NM-ROM fast arm's worst on the **6 development
cases against a same-job `fft_tight` 256 solve**; section 2 says those are "not directly
subtractable". On diagnosis-8 the **published** arms already clear it: `unet-small` 1.4712 %,
`unet-medium` 1.5189 %, `tsol-refine` 1.5224 %, `unet-refine` 1.7110 %. The same checkpoint
scores 1.7110 % on diagnosis-8 and 4.5529 % on the panel — a 2.7x cohort gap. T3 does not say
**which statistic**; on validation-32 `unet-medium` is already at mean 1.4341 %, median
1.3169 % and worst 3.9622 %, all below 4.03 %.

**M10 — T1 is biased toward "tuning helped" by construction, and the design does not say so.**
The tuned arm is the argmin of a 5-6-wide pool, the published one of a 3-4-wide pool, on the
same 32 cases.

**M11 — section 5.4's "Code changes this requires, and nothing else" is already false**, and
`audit.py` has demoted `epochs`, `patience`, `weight_decay`, `learning_rate` and `schedule`
from asserted to recorded. `patience` and `epochs` govern early stopping and therefore which
checkpoint is selected.

**M12 — the shared cache `<ns>/cache` has no owner in the DESIGN.** `gen01` is not just a data
job, it is the job that produces the cache both later jobs stage from.

**M13 — "equal wall ~ equal steps" is wrong, in the direction that flatters the headline.**
Per-epoch fixed costs amortise over 16 steps at the bottom rung and 576 at the top, so ~20-25 %
of the bottom rung's wall is fixed overhead versus <1 % at the top: the 4608 rung gets up to
~25 % more gradient steps at the same wall.

**M14 — ladder per-arm data loading is unbudgeted.** Each 4608-case arm re-reads its training
set four times (~64 GB of decompression plus 16 GB of hashing, single-threaded), and
`worker_second.train()`'s `wall = min(budget, remaining() - RESERVE)` takes it out of later
arms' wall, silently breaking the equal-wall premise.

**M15 — disk is unbudgeted**, on a share `CLAUDE.md` describes as ~99 % full with a documented
empty-log-on-full-disk failure mode.

### Constructibility — every arm built and run at batch 8, 257<sup>2</sup>

Nothing OOMs and nothing fails to construct. `fno-large` reconstructs to **exactly 17 877 317**
real parameters, matching the published table. `fno-norm` works (`norm` is a valid `Literal` in
neuralop 2.0.0; `to_f64` converts the GroupNorm weights; `check_dtypes` passes). `fno-modes48`
is the FNO's best-value arm: +120 % parameters for **+2 % step cost**. `tsol-patch2` runs at
10.69 GB and 2.12x step cost. `unet-groups16` would fail at base 24 (24 % 16 != 0); the grid
uses base 32.

| arm | params | peak GB | step s | vs baseline |
|---|---:|---:|---:|---|
| `fno-large` (baseline) | 17 877 317 | 2.01 (b=2) | 0.921 | — |
| `fno-modes48` | 39 373 125 | 2.24 | 0.941 | **1.02x** |
| `fno-width96` | 40 222 181 | 3.28 | 1.200 | 1.30x |
| `fno-layers6` | 26 807 045 | 2.94 | 1.301 | 1.41x |
| `fno-norm` | 17 878 341 | 2.58 | 1.040 | 1.13x |
| `tsol-small` (baseline) | 3 108 240 | 2.79 (b=8) | 0.705 | — |
| `tsol-patch2` | 3 094 356 | **10.69** | 1.496 | **2.12x** |
| `tsol-slices128` | 3 116 944 | 4.09 | 0.473 | ~1x |
| `tsol-layers6-d192` | 5 245 952 | 2.71 | 0.327 | — |
| `unet-base64` | 31 038 469 | 4.35 | 0.398 | 1.24x |

### Disjointness — checked computationally, and clean

`case_seed('train', i)` for i in [0, 4608): **4608 distinct uint32 seeds, zero internal
collisions**; zero collisions with validation 0..31 or calibration 0..7; `case_seed('train',
0) = 428288116` = the published `burgers-train-00000` seed. `dataset.load_index:27` refuses
any split outside `{train, validation, calibration, development}`, so final/held-out is
unreachable by construction. The one gap: the 6 development cases the `ops-timing-panel` used
are generated by the ROM lane's own JAX phase, outside this seed space, and could not be
verified directly.

### Verified correct, so as not to be re-litigated

All five section 2 panel numbers match `ops-timing-panel`'s table exactly (0.5129/0.1789,
1.8891/1.0100, 4.4593/2.3195, 4.5529/1.9662, 7.4164/2.5256). Every number in section 3.2's
table is transcribed correctly — from the wrong file. No route to a final or held-out cohort
exists. G2 is correctly *constructed*; its problem is power, not construction. Section 2's
measurement caveat is the right caveat and correct on the reference axis. `gen_bank.py` uses
no numerical path that differs from `data.generate()`. Holding the 3000 s screen budget is the
**right call, not a dodge** — moving it would break comparability with every published arm.
Arithmetic checked: 4608/128 = 36x; 4608 x 51 = 235 008; 4608 x 5 = 23 040; 128 x 5 = 640;
131 072/23 040 = 5.7x; 135/0.19 = 710x.

### Minors

"135 s per trajectory" is not reproducible: the 16 pinned records give mean **129.9 s**, median
**127.2 s**, range 113.4-157.6, so 173 A100-h is 162.8 h on the measurable median. Section 3
item 3 compares a *gate margin* (1.03 %) against a *deviation* (0.19 %); like for like it is
3.6-4.9x, not 5.5x. "At least 5x finer" rests on 256/dt=0.005, which **was never measured**.
`gen_bank.py`'s docstring says "28x more data"; it is 36x. Section 1 says 2.4-2.8x and section
6 says 2.4-3.4x for the same ratio over different arm sets. Prefix indices are written only at
the end although section 5.1 item 4 promises they are incremental. No in-loop duplicate-input
check; `load_index` catches it only at training time, after 16 GB and nine hours. Section 8
says nothing here is timed, but `worker_second.py` runs `timing.py` and `audit.py` asserts it
present for every arm.

### Coverage gaps — UNVERIFIED

1. Jobs `2835788` / `2837431` — the entire NM-ROM side of section 3. No artifact in any
   worktree read.
2. The seeds of the `ops-timing-panel`'s 6 development cases.
3. The `refinement02` calibration index itself; its local mirror was used.
4. Actual A100 epoch rates at 512/2048/4608 cases — all per-step numbers are GB10 measurements
   scaled against published A100 epoch counts.
5. Current free disk on `/cluster/tufts/paralab`.
6. The lane's files were being edited while the audit ran; findings reflect the tree as of
   ~19:45.

### The single change that would most improve the lane

> Make the ladder's learning-rate schedule and early stopping index on gradient steps rather
> than epochs, and pre-register that the ladder is compared on steps rather than wall.
> Everything else on this list is either a text edit or a number in a JSON file. This is the
> only finding that **silently destroys the lane's headline result and cannot be repaired
> after the jobs run**.
