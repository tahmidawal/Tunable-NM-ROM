## 2026-09-17

### b-seeds — the dense Burgers correction ladder across three retrained seeds and a sealed cohort: C1 (monotone + converged on ≥ 2 of 3 seeds) HOLDS (2 of 3); sealed/dev ratio ≤ 1.5 at every rung fails (difficulty-normalised: fails)

Branch `exp/2026-09-17-b-seeds` at `e533b48ecd1de7203a63d80769f6e15471b57e08`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`; namespace `/cluster/tufts/paralab/tawal01/b_seeds_20260917/`, one attempt directory per job. Jobs: `s1` 3783776 (NVIDIA A100 80GB PCIe), `s2` 3783777 (NVIDIA A100 80GB PCIe), `s3` 3783778 (NVIDIA A100-PCIE-40GB), `final` 3804465 (NVIDIA A100-PCIE-40GB). Every job printed `jax_backend=gpu`, ran float64 at highest matmul precision, was checksum-collected, independently NumPy-audited and Git-archived before its exact remote attempt directory was removed. Predeclared protocol and amendments: `experiments/b-seeds/DESIGN.md`.

**T12 — development cohort, `dense_m4`, worst evolved % (seed mean ± std; incumbent re-run in the same job):**

| q | seed1 | seed2 | seed3 | incumbent | converged (seeds) |
|---|---|---|---|---|---|
| 0 | 2.5745 | 1.7483 | 1.6778 | 1.8890 | 3 of 3 |
| 16 | 1.6914 | 1.6114 | 1.4935 | 1.3985 | 3 of 3 |
| 32 | 1.2932 | 1.5243 | 1.3006 | 1.2336 | 3 of 3 |
| 64 | 1.1470 | 1.2561 | 1.2358 | 1.0843 | 3 of 3 |
| 128 | 0.9337 | 0.9853 | 0.9495 | 0.8930 | 3 of 3 |
| 256 | 0.6329 | 0.4363 | 0.4268 | 0.5194 | 2 of 3 |

**Three layers at q = 0 (bank floor / best-found / solved all-times %):** seed1: 0.3627 / 2.8600 / 2.8605; seed2: 0.3827 / 2.5361 / 2.5367; seed3: 0.3511 / 2.5697 / 2.6005; incumbent: 0.3918 / 2.5447 / 2.5629.

**T13 — sealed cohort `params_draw(17092026, 6)`, opened in the final job only; worst evolved % (sealed / dev):**

| q | seed1 | seed2 | seed3 | incumbent |
|---|---|---|---|---|
| 0 | 2.1548 / 2.5745 | 2.8268 / 1.7483 | 3.4254 / 1.6778 | 10.1120 / 1.8890 |
| 16 | 1.7129 / 1.6914 | 1.6105 / 1.6114 | 1.8680 / 1.4935 | 1.4617 / 1.3985 |
| 32 | 1.6485 / 1.2932 | 1.3642 / 1.5243 | 1.5006 / 1.3006 | 1.4458 / 1.2336 |
| 64 | 1.4770 / 1.1470 | 1.2686 / 1.2561 | 1.3126 / 1.2358 | 1.3252 / 1.0843 |
| 128 | 1.1174 / 0.9337 | 1.0061 / 0.9853 | 1.0148 / 0.9495 | 1.0964 / 0.8930 |
| 256 | 0.6123 / 0.6329 | 0.5880 / 0.4363 | 0.6562 / 0.4268 | 0.6789 / 0.5194 |

Counts over the three seeds on the development cohort — `dense_m4` monotone on evolved: 3 of 3, on all times: 3 of 3, every rung converged: 2 of 3, knob bar: 2 of 3; `dense_fixedM` monotone on evolved: 3 of 3. C1 = yes (2 of 3); F3 (the recipe is NOT reproduced on >= 2 seeds) = no.

**What was wrong, retracted or substituted (chronological).**

- **Codex audit substituted (DESIGN.md A1).** `codex exec` with `gpt-6-astra` and `gpt-5.6-sol` both returned the account's usage limit ("try again at Sep 19th, 2026 11:33 AM"), so the protocol's pre-job Codex audit could not be obtained. An independent review by a Claude agent with no access to this lane's conversation (same read-only brief) was used instead and is recorded as such — a different context, not a different model family. Its blocker (the incumbent fidelity gate set at a $10^{-9}$ tier qtd02 itself did not meet) and its major findings were accepted before any job.
- **First local parent smoke died at CUDA init** (`CUDA_ERROR_OUT_OF_MEMORY` creating the stream executor) while another lane's processes held most of the shared GB10; it was rerun once the slot freed. No number depended on it.
- **The audit's training gates read the stage A/B JSONs by their `N256` names**; the 64-interval chain smoke exposed it (`FileNotFoundError`). Fixed to glob the mesh before any job; recorded in A1.
- **Local smokes exceeded the sub-minute rule** (the parent smoke ≈ 2 min, the chain smoke ≈ 15 min at 64 intervals), as the parent lanes' smokes did; recorded as a deviation, the rule stands.
- **Seed 3's top rung is not converged and is reported as such.** `old_q256_M1088_dense` on seed 3 exits on the 600-step budget for 2 of the 6 development cases (cases 2 and 3), identically in all three timed repetitions, worst joint stationarity 1.52e-06 / 5.83e-06 against `gtol` 1e-06. It is budget exhaustion, not divergence — that rung still carries seed 3's lowest error. Nothing was tuned in response (the budget is part of the frozen qtd02 configuration), so C1, C3 and C4 read 2 of 3 and every table marks the rung unconverged. No number was withdrawn.
- **The 1e-9 fidelity probe fails on the high-$q$ incumbent arms**, as A1.1 anticipated: worst 1.34e-08 (s3, `old_q256_M1088_dense`), against the amended 1e-3 gate that every one of the 33 incumbent arms passes with five orders of margin. 20 of 33 arms still meet 1e-9. This is reported per arm in the report body, not only in the amendment.

Source-generated report: `experiments/b-seeds/reports/2026-09-17-b-seeds.md` (SHA256 `cf791634c5bf1a1c7511c6db6f77a79bb15afd483eb2d9c9338425b7442371db`), with `summary.json` and the generator beside it; every number above is read from `summary.json`. Raw archives are Git-tracked as bounded chunks under `experiments/b-seeds/artifacts/`. Not pushed; not merged.

Sealed job `final` 3804465 (A100-40GB, 3h37m, `jax_backend=gpu`): collected, audited (34/34 + 3 × 33/33 gates, incumbent hash `18f0266ae6f0…` gated, sealed values ≤ 1 ulp from the declared draw, disjoint, `final_cohort_unopened == false`), archived to `artifacts/final/`, remote directory deleted; `selfaudit_recheck.py` still 360 of 360 on the development rows. Jobs used: 4 of 8; nothing further submitted. Open: the Codex report audit after 2026-09-19 11:33, appended as a dated addendum.
