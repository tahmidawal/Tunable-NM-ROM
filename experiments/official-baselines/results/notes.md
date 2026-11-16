## What was and was not measured

- **Measured once on test64:** CoLoRA-D (the selected $k=16$ configuration in float64 and its float32 twin; the authors'
  default configuration at the released and at the extended step budget), CROM-I (released Gauss-Newton solver on the
  backward-Euler residual of the reference) and CROM-E (as released, explicit). Selection used validation only
  (`results/selection.json`; the earlier per-arm files `selection-colora.json` and `selection-crom-I.json` give the same
  choices, amendment A2).
- **Not measured, because the code is not released:** CoLoRA's equation-driven (Neural Galerkin) variant; CROM with
  implicit time stepping as the authors would implement it (CROM-I is our pre-declared connection, a separate row).
- **Not carried to test, by the frozen rule:** CROM float32 twins (validation worst error far above the float64 setting,
  rows in the validation table); CROM "fast" rows (no CROM setting meets the validation gate); the CROM `C16-s128` explicit
  setting with tight Gauss-Newton thresholds, which was projected beyond the 6 GPU-hour validation budget, so it ran on
  8 validation cases only and was not selectable because fully evaluated settings exist for that arm (PROTOCOL 5.2).
  Its partial validation number is the lowest of the CROM-E settings and is shown, labelled, in the validation table.
- **Training failure, recorded as measured:** CoLoRA-D `K16-w128` did not train at the released learning rate (loss flat);
  it was not selected.
- **Inputs differ by design:** CoLoRA-D receives the five generating parameters of a case (as in its released notebooks);
  CROM and the paper method receive the initial field.
- **Timing:** all official test rows on A100 80GB PCIe, the GPU type of the paper-method and FNO/U-Net rows (cited from
  the b2d-query-speed lane, job 7220630, not re-run). CROM runs in PyTorch eager mode as released; CoLoRA under `jax.jit`;
  the paper method in float32 with its frozen settings. CROM settings slower than 60 s per query are timed by their single
  evaluation pass (one warm-up case per job; the CROM-E test was split into four case-chunk jobs, amendment A2).
- **Training hardware:** CROM `C16-s20` and `C16-s64` trained on A100 40GB, `C16-s128` on A100 80GB (amendment A2 iii).

## Glossary

- **test64**: the 64 held-out Burgers 2D test cases, `params_draw(20260916, 64)`, evaluated once per frozen selection.
- **worst % / median %**: over the 64 cases, of each case's largest relative $L_2$ error over the five output times, relative to the initial field's norm.
- **ms / query**: wall time of one query at batch size 1 on one GPU, from the input on the GPU to the five output fields on the GPU.
- **CoLoRA-D**: the data-driven CoLoRA variant in the authors' repository (a hyper-network maps parameters and time to the low-rank weights of a coordinate network).
- **CROM-E / CROM-I**: CROM's released online loop with its explicit integrator (E), or with the same Gauss-Newton solver minimising the backward-Euler residual (I).
- **GN setting** (`ex`, `def5`, `def50`, `def2`): Gauss-Newton stopping thresholds and iteration cap (PROTOCOL 5.2).
- **accurate / fast / float32 twin**: the frozen selection roles (lowest validation worst error; fastest under the validation gate; the float32 copy of a selected float64 setting, kept only if within 5 % of its validation error).
- **partial**: a validation setting run on 8 instead of 32 trajectories because its projected cost exceeded 6 GPU-hours.
- **width 176 / 192 / 384**: the paper method's span widths, accurate query settings, cited from b2d-query-speed.
- **F0 / U0**: the FNO and U-Net neural operators trained on the same 4,608 trajectories (cited).
- **re-implementation**: the in-house CROM-style and CoLoRA-style baselines of lane nfrom-baseline (cited).
- **integrator floor**: the error of CROM's explicit time stepping applied to the exact field, with no network (PROTOCOL 5.3).
