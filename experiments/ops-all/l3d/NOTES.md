## Notes and caveats (prose only; every number above is generated)

* **Same-job ratios only.** Every speedup divides the FOM median of *this* panel job by the arm's median in the
  same job, on one GPU in one process. The FOM is chosen by the protocol's rule in this job (fastest setting, all
  cases converged, whose worst error is at most the NM-ROM accurate arm's worst error). For every completed cell it
  is the same setting Table 1 names (Poisson CG rtol 1e-2; heat CN–CG Δt 0.025 rtol 1e-4).
* **Errors reproduce Table 1 exactly; times reproduce at 256³ only.** Both NM-ROM arms reproduce the source jobs'
  per-case errors to machine precision at every mesh. At 256³ (H200, as the source jobs) the NM-ROM and Table-1 FOM
  medians also agree with the source jobs to within a few percent. At 32³–128³ (A100-80G) the NM-ROM medians are
  1.2–2.7× the source jobs' and the FOM medians 1.1–1.9× (see "this job vs the Table-1 source job"): at 128³ the source
  jobs ran on an H200 and these panels on an A100; at 32³/64³ both ran on A100-80G, and the sub-millisecond NM-ROM
  queries are host-dispatch bound (three of these panels shared node pax106 with other jobs). Not diagnosed further;
  a local GB10 check showed no slowdown from having PyTorch and the dlpack input views in the process. **For the
  paper's NM-ROM speedups keep the Table-1 source jobs; use these panels for operator-vs-FOM ratios (same job), or
  quote NM-ROM and operators from the same panel with this caveat.**
* **Shared GPUs during training** (DESIGN §2): the four families of a cell trained concurrently on one GPU (eight
  processes for 32³+64³), each with a fixed memory fraction and its own 3000 s budget. All arms stopped on the wall
  budget; at 64³ and 128³ several arms saw only a handful of epochs (printed per arm). The operator errors are
  therefore budget-limited, not converged, and would improve with a dedicated GPU or a longer budget. No size or
  hyper-parameter was tuned or chosen on any panel data.
* **Heat 64³ reproduction gate failed on the FOM, not on the NM-ROM.** Both NM-ROM arms match the source per-case
  errors exactly; the Table-1 CN–CG setting (Δt 0.025, rtol 1e-4) differs in some case by more than the pre-set
  1e-3 relative tolerance (worst-case error equal to the printed digits). Loose-tolerance CG stops at a
  rounding-dependent iteration, and the source job ran on an H200, this one on an A100. The panel is labelled
  PROVISIONAL by its own rule; the ratios are same-job and unaffected.
* **Operator timing scope** = device-resident (n−1)³ float64 input → device-resident float64 output fields,
  including feature construction (padding, coordinate channels), the network, cropping and scaling, synchronised
  with `torch.cuda.synchronize`. U-Net/Transolver/DeepONet run in float32 internally (as in the Table-2 cells),
  FNO in float64; TF32 disabled, cuDNN autotuning not enabled.
* **Cohorts.** Poisson 32³/64³ use the reserved final cohort (64 cases, held-out; this is a further opening of it
  with no choice made on it), 128³/256³ the 16 development cases — as Table 1. Heat uses the 64 sealed held-out
  cases at every mesh (as Table 1). Operators chose their checkpoint on a validation split of their own training
  draws only.
* **Heat error** is the worst over cases of the max over all six output times (t = 0 included, where the operators
  return u0 exactly and the NM-ROM projects it), as the Table-1 heat rows.
* **Audit** (in `make_results.py`, NumPy, local): recomputes every worst error and every median from the raw
  per-case errors and timing samples in the pulled summary, and re-applies the FOM rule; mismatches are printed.
  No full fields were saved or pulled (disk), so there is no independent field-level recomputation.
* Code was not committed (coordinator commits); every job's staged-code manifest `SOURCE.sha256` is pulled into
  `runs/<job>/`, and each panel summary records the sha256 of every script and dependency.
* **256³** (DESIGN A1, A3–A6): first attempt on a shared H200 — FNO-s and DeepONet-s found no micro-batch within
  their shares; Transolver-s and U-Net-s died in the harness (a data-generation batch, fixed). Second attempt
  `tr_256c` (H200): whole-GPU probes show FNO-s and DeepONet-s fit micro-batch 1 only with a whole H200 (peaks in the
  failure strings); their dedicated-GPU runs were cancelled before starting when the group's GPU cap dropped to 2,
  so they are reported as **not trained**. U-Net-s and Transolver-s trained concurrently on one H200; U-Net-s
  completed a single epoch (35 optimisation steps) in 3000 s, so its 256³ numbers are essentially untrained.
* **Transolver** is the only family faster than the FOM at 128³ and 256³ (its 16³-token grid makes its cost nearly
  mesh-independent), at errors of a few percent — one to two orders of magnitude above the NM-ROM and the FOM.
