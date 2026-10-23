Arithmetic and job specification pass, but the remaining-work certificate has a blocking gap.

- **Arithmetic:** independently recomputed **81.940694 GPU-h**, leaving **5.059306** below 87. Adding W16_A2 gives **87.596458**, correctly exceeding the cap. Ledger rounding is immaterial; completed-job accounting matches `sacct`.
- **Production bound:** **4.918056 h**, from W40_A2, is a defensible empirical planning estimate for W16_A1: all four supporting production cells completed at full size with passing controls, GPU/f64/highest precision, and unchanged query-source hashes. W16_A1 omits bank and TRANS and uses the easier A1 level. This is not a guaranteed runtime ceiling.
- **Specification:** body hash matches RELEASE.json; **H200 / 240G / 10:00:00** matches. Full cohorts and reference resolution remain enabled; controls run; bank and TRANS are excluded by the registered width predicates.
- **Blocking budget gap:** [`certify_wave4.py:12`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/certify_wave4.py:12) assigns **0.5 h per test cell without measured or enforced support**. The existing estimator uses measured reference + timing + controls costs; these total **2.2208 h for W40_A2**. That does not prove test-once needs that long, but the reduction needs evidence. Test-once includes fresh references, qualification and paired timing; A3 has no completed timing measurement. Recertify this mandatory remaining work before adding W16_A1.
- **Deferrals:** A15 incorrectly says A3 crossing analysis includes W08 despite deferring W08_A3. A3 retains only W28/W40. Explicitly account for or defer the registered second-data-size work, also absent from this certificate.
- **Queue:** W08_A1 and W40_A3 were running; W28_A3 and W08_A2 remained with the existing releaser. The second same-command shell is a waiter wrapper, not another active release loop.

No files changed or jobs submitted.

**DO-NOT-SUBMIT**