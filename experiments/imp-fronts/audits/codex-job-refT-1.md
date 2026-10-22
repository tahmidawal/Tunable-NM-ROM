**WRONG — fix finalization and unfinished-case accounting before submission.**

- **CORRECT — Recipe:** numerical solver, refinement and acceptance logic match jcp-regimes; reference core, verdict, gates and calibration files are byte-identical. Adapter changes restrict cohorts and enforce tight allowance 0.
- **CORRECT — Cases:** exactly 24 unique TEST cases: FI 4–9, FN0s 4–11, FC1 4–11, FC2 4–5.
- **CORRECT — Cohort hash:** canonical hash recomputes to `1427a451bfa6f0a8a713c060ef442eac31bf5dc1627c796ca00e841e079aff47`.
- **CORRECT — Provenance:** all staged manifest hashes pass; all provenance-listed files match their recorded Git blobs.
- **CORRECT — Resources:** `gpu` partition, two H200s, `01:37:30`, 3.25 GPU-h; no preempt.
- **CORRECT — Environment:** absolute cluster venv, f64, `highest`; both GPU preflights exit 42 on CPU fallback.
- **CORRECT — Namespace:** outputs, logs and caches target `/cluster/tufts/paralab/tawal01/imp-fronts/refT/`.
- **CORRECT — Work split:** GPU 0 gets FC1/FC2 (10 cases); GPU 1 gets FI/FN0s (14); disjoint output directories.
- **NEEDS-RESTATEMENT — Timer:** shared cutoff is 5,265 seconds from script start; `timeout` terminates active work there, rather than merely stopping new cases. TERM grace permits another 30 seconds.
- **CORRECT — Successful slimming:** retains the selected finest field and `t0`, copies metadata unchanged, records source/slim hashes and preserves raw arrays.
- **WRONG — Empty finalization:** `refslim.py` never creates its output directory when no case finishes; writing `MANIFEST.json` fails despite `allow_empty=True`, preventing `OUTPUTS.sha256`.
- **WRONG — Unfinished cases:** skipped/interrupted cases are inferable from configs and logs, but no explicit 24-case outcome ledger or unfinished count is written. Slimming lists only completed cases; `ALL-DONE` can follow failed steps.
- **NEEDS-RESTATEMENT — Admission:** reservation metadata alone does not establish Amendment 12’s hash-bound passing controls and smoke evidence.

Read-only audit completed; no files changed, jobs run, SSH used, or sealed/OOD cohorts opened.