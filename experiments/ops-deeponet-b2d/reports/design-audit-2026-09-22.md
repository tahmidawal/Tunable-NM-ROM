# Independent design audit, 2026-09-22 — before job `don01`

Written by an independent auditor (a Claude subagent, read-only, with no part in writing the
lane) against the brief in `design-audit-prompt.txt`. Codex could not run: see
`codex-design-audit.md`. Reproduced here as received; the disposition of every finding —
all five accepted and fixed before submission — is in `DESIGN.md` §A2.

---

## Verdict

**The job will run to completion — nothing here dies in the preamble, OOMs, or exhausts the
budget.** But do not submit yet: one deviation changes the trained weights (M1), and two
defects will block or misdirect the post-job pipeline (B1, B2). B1/B2 are cheap to fix now
and expensive to discover after a 4-hour allocation.

## BLOCKER

### B1 — `audit.py:149` rejects every DeepONet arm; DESIGN §6 gates 4 and 5 are unmeetable

```python
assert config['family'] in ('unet', 'transolver')
```

`audit.py` is declared byte-identical to `no-second` (DESIGN §2,
`checks/inherited-sources.json`), and every `don-*` arm's `provenance.json` will carry
`family: "deeponet"`. `audit_arm` raises on the first arm, so `audit.json` is never written —
which means DESIGN §6 gates 4 and 5, `collect.py --cleanup` (asserts `audit.json['passed']`),
and `reports/generate_report.py` (reads only `runs/don01/audit.json`) all fail. This is
exactly the §A1 failure mode the brief names: a pre-registered gate that fails for a reason
unrelated to the science.

Everything else `audit.py` checks is satisfiable: `PROTOCOL` (`epochs=4000, patience=250,
batch_size=8, weight_decay=1e-4`), `LEARNING_RATES = {1e-3, 3e-4}`, `seed 20260914`, and the
float32 selection-gap tolerance `1e-5` all match `configs/deeponet/*.json`. `expected_arms =
{don-small, don-medium, don-large, don-refine}` matches what `worker_second.py` produces.

**Fix:** add `'deeponet'` to the tuple at `audit.py:149` (and `:94` for symmetry), add
`audit.py` to `EXPECTED_CHANGED` in `check_inherited.py:21`, and amend DESIGN §2's
byte-identical list. One line of code.

### B2 — `cluster/collect.py:23-24` still points at the `no-second` lane and namespace

```python
LANE = ROOT / 'experiments/no-second'
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_second_20260917'
```

Verified byte-identical to no-second (`checks/inherited-sources.json` lists it under
`identical`), and DESIGN §2 does not mention it at all — it fell through both the
"byte-identical" list and the "changed" table. `python cluster/collect.py don01` will try to
`ssh … cd /cluster/…/no_second_20260917/don01` (that namespace was cleaned to zero entries
after no-second closed, per its `self-audit-controls.md`) and will `mkdir` the archive under
**`experiments/no-second/runs/don01/`** — a write into another lane's tree, violating the
one-session-one-worktree rule. `--cleanup` would `rm -rf` a path in the wrong namespace.

**Fix:** same two-line edit as `stage.py` got (`LANE = ROOT /
'experiments/ops-deeponet-b2d'`, `NAMESPACE = '…/opsdon_20260922'`), plus
`cluster/collect.py` into `EXPECTED_CHANGED`.

## MAJOR

### M1 — `families.py:269` feeds the trunk coordinates on **[0, 1]**; the 3D reference feeds **[-1, 1]**. The explicit Fourier support is halved.

`model.features` (`model.py:86-88`) builds the coordinate channels as `linspace(0, 1)`. The
3D lane builds its coordinate channels as `grid_coords_3d(n)[interior] * 2 - 1`
(`paper-b3d/operator_panel.py:55` and `:93`; `vendor/b3d_common.py:64` is `linspace(0,1)`),
i.e. on **[-1, 1]**, and `_trunk_features` (`extra_models3d.py:50-54`) applies
`sin/cos(π·f·coords)` to that raw range with the same default `f ∈ {1, 2, 4}`.

Consequence: over its domain the 3D trunk gets 1, 2 and 4 full periods; the 2D port gets ½, 1
and 2. Across a 257-point grid the highest explicit basis function completes **two**
oscillations. The coordinate inputs are also no longer zero-centred, which biases the first
`tanh` layer's pre-activations.

This is not cosmetic. DESIGN §2.1 pre-registers the trunk as "the 2D analogue … exactly as in
3D", DESIGN §7 names the coordinate-MLP trunk as the single thing a reviewer will attack, and
§A2's whole purpose is that the implementation matches the claim. Shipping this produces a
number that will be written up as "DeepONet in 2D" when it is "DeepONet with half the
reference's trunk frequency support". A tanh MLP can synthesise higher frequencies, but that
is precisely the argument this project has retracted before.

**Smallest correct fix:** one line at `families.py:269` — `coords = x[0, -2:].permute(1, 2,
0).reshape(-1, 2) * 2 - 1` — which makes the "exactly as in 3D" claim true and leaves every
other family untouched. (The alternative — keep [0,1] and state the deviation in §2.1 and in
the report — is defensible but weaker, since it is a silent handicap on the only
family-specific component.)

Note this changes the trained weights, so it must be decided **before** submission, not after.

## MINOR

### m1 — `families.py:265`: `F.avg_pool2d(..., 2, ceil_mode=True)` uses a different divisor than the 3D `_pool2`

Verified empirically: on a 5×5 field of ones, PyTorch returns `1.0` at the truncated edge
window (divides by the actual element count), while the 3D `jax.lax.reduce_window(...,
'SAME') / 8` returns `0.5` / `0.25` (divides by the full window, counting the zero pad).
Shapes agree exactly (257→129→65→33 either way); only the last row/column of each pooled map
differs. Given the zero-Dirichlet boundary and the subsequent 4×4 adaptive pool, the numerical
effect is negligible, and PyTorch's behaviour is arguably the better one. Worth one sentence
in DESIGN §2.1 rather than a code change.

`F.adaptive_avg_pool2d` and the 3D `_adaptive_pool` are bin-for-bin identical (`floor(i·n/bins)`
to `ceil((i+1)·n/bins)` in both) — no deviation there.

### m2 — `families.py:258-259`: the 0.1 read-out scaling is applied to the weight only

3D's `_dense(..., scale=.1)` (`models3d.py:15-17`) produces `w = 0.1·N(0,1)/√ci` **and `b =
0`**. The port scales `branch_read.weight` but leaves `branch_read.bias` at PyTorch's default
`U(±1/√768) ≈ ±0.036`, ten times larger relative to the scaled weights than in 3D — an
input-independent offset in the coefficient vector at initialisation. Trivially trainable
away; a one-line `self.branch_read.bias.zero_()` inside the existing `no_grad` block makes it
faithful.

### m3 — `check_inherited.py` will now fail on `HANDOFF.md`

`EXPECTED_NEW` (`check_inherited.py:23-24`) omits `HANDOFF.md`, and the rglob skips only
`runs/`, `reports/`, `checks/`. `checks/inherited-sources.json` was generated at 17:15, before
`HANDOFF.md` was written at 17:20, so its `passed: true` is stale. Re-running it — which
`HANDOFF.md` itself instructs ("Run it after any edit") — reports `undeclared new file:
HANDOFF.md` and exits 1. Add `HANDOFF.md` to `EXPECTED_NEW`.

### m4 — pre-registration note, not a defect

The DeepONet compresses a 257² field through a 4×4×192 global bottleneck before the trunk,
while the FNO/U-Net/Transolver arms it is tabulated beside are full-resolution field-to-field
maps. That is what a DeepONet is, and DESIGN §7 / D3 already pre-register the honest negative.
Flagging only so the report does not let the reader read the gap as an implementation failure.

## Checked and found fine

**DeepONet faithfulness (the rest of it).** Channel split `cin - 2` (`families.py:244`)
matches `model.features`' layout and the Transolver's own `(cin-2)` at `:196`. Branch topology
(3 levels × [conv-GELU, conv-GELU] then pool, then adaptive pool, GELU hidden, linear read)
matches `deeponet_coefficients` block for block. `1/√rank` at `:277` matches
`jnp.sqrt(trunk.shape[-1])` (trunk's last axis *is* rank). Additive `bias[None,:,None,None]`
at `:278` matches `+params['bias']` broadcast over the channel axis. tanh on all trunk layers
but the last, GELU on the branch hidden — both match. `assert trunk_width >= rank` mirrors the
3D assertion and holds for all three configs (384≥256, 512≥384, 768≥512).

**The `x[0, -2:]` trunk shortcut is sound.** `model.features:86-90` builds the grid from
`field.shape` alone and `.expand`s it; the values are bitwise identical across the batch,
`Precision`'s elementwise cast preserves that, and no gradient flows to the coordinates. The
guard at `smoke_second.py:93-97` asserts it against the real `adapter.features` on a batch of
3 and runs in-job on every submission — adequate. Coordinate ordering is consistent:
`permute(1,2,0).reshape(-1,2)` is row-major over `(h,w)` and so is the `reshape(b, cout, h, w)`
at `:278` (confirmed by running the forward: `coords[1] = [0, 1/256]`, `coords[257] = [1/256, 0]`).

**Fairness against the other arms.** `configs/deeponet/*.json` match `configs/unet/medium.json`
key-for-key on every protocol field: `dtype float32`, `warmup_epochs 0`, `epochs 4000`,
`patience 250`, `batch_size 8`, `lr 1e-3`, `wd 1e-4`, `seed 20260914`. (Transolver alone got
`warmup_epochs 10`; DeepONet matching U-Net and FNO is the right call.) Identical contract —
same `features`/`predict`/`relative_errors`, same boundary mask, same `output_scale`, same loss
(`errors[:, 1:].square().mean()`), same `ReduceLROnPlateau(0.5, 20, 1e-5)`, same AdamW, same
grad clip, same selection rule, same 3000 s per arm and same 3e-4 refinement as jobs
3780138/3780139. `train.py`, `dataset.py`, `timing.py`, `evaluate_cohort.py`,
`prepare_diagnosis_cohort.py`, `spectral_conv_f64.py` are byte-identical to no-second (`diff`
clean). `timing.py:71` and `evaluate_cohort.py:29` rebuild the network from
`checkpoint['config']` and are family-agnostic. No arm gets an advantage the others did not have.

**Parameter-count table.** Ran `families.py` under `/home/tahmid/Dev/.venv/bin/python`:
2 569 349 / 4 694 789 / 10 258 181 — matches DESIGN §2.1 exactly, and spans the
FNO/U-Net/Transolver range as claimed.

**Job path.** `runs/don01` is already staged and `check_references` passed (26 staged files,
every `"code/…"` literal in `worker_second.py` and `specs/don01.json` resolves; the `configs/`
rglob picks up all three DeepONet configs, so the `pois01` preamble death does not recur).
`worker_second.py:22` asserts the new namespace and `run.sbatch` sets `TASK_ROOT` to the
matching path. `data_dirs` defaults include `refinement`, which `prepare_diagnosis_cohort.py`
needs. The sbatch carries the fixed trap/`wait` USR1 forwarding (no-second's MAJOR #2), the
`jax_backend=gpu` preflight with exit 42, `--partition=gpu`, `--constraint=a100-80G`,
`--exclude=pax007`, output under paralab.

**Budget.** From no-second's actual `worker.json`: smokes + cohort prep = 27 s, four arms at
3010-3013 s, cohort-eval 8 s, timing 61-91 s, total ~12 150 s against `global_seconds 15600`
and `--time 04:40:00` (16 800 s). Adding DeepONet to the precision smoke adds four tiny 65²
cases (~5 s). The refine gate (`remaining - 900 ≥ 1200`) clears by ~4 400 s. Ample.

**Memory and throughput.** Peak 2.73 GiB for `large` against an 80 GB A100 — no OOM risk, and
far below `unet-large`'s observed 11.38 GiB. At 0.088-0.228 s/step × 16 steps/epoch the arms
should reach roughly 1 300-2 300 epochs in 3000 s, bracketing `unet-small`'s observed 2 327 and
`tsol-large`'s 882; the `epochs: 4000` cap will not bind and `stop_reason` will be
`wall_budget` like every sibling. Per-epoch `last.pt`/`best.pt`/`history.json` writes are the
same load U-Net carried at comparable parameter counts. Ran a CPU forward+backward of the
`small` config at the production 257×257 with a hand-built replica of `model.features`: output
`(2, 5, 257, 257)` float64, all finite, all gradients finite.
