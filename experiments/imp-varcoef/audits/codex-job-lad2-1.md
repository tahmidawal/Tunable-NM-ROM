- **CORRECT — Job request.** `run.sbatch` requests one H200, 128G, and 5.75 h. Certificate ×1.5 is **5.712574 h**, within that limit. GPU, f64, highest-precision and manifest checks are present.

- **CORRECT — Configuration.** `lad2.json` differs from `ladder.json` only by the three resume fields; `stop_after_val` is absent. The cluster/local paths identify L1’s result, whose SHA256 matches `71fa94dfb5e902e0a4cd69ae6f40744d44b3bfd841d45c4ca8cc3b5b9dfdf01f`.

- **CORRECT — Resume workflow.** `vladder.py` checks L1 phase, eligibility binding and result hash, restores runs and carries the existing stage archive. Screening and qualification occur at C=100; C=10 remains development-only.

- **CORRECT — L1 completion.** Result reports `complete=true`, phase `L1`, backend `gpu`, and both contrasts. C=100 has 476 configurations with 232 reaching validation; C=10 has 542 development configurations. Logs contain `jax_backend=gpu`, `x64+highest ok`, and `ALL-DONE`; stderr is empty. Pulled output checksums pass.

- **WRONG — Resource-model change.** [resources.py:195](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:195) assigns a local `run_cost`, making the earlier L1 branch’s call reference an uninitialized enclosing variable. A read-only estimator probe reproduces the `NameError`. Rename the L2 callable.
  
  **NEEDS-RESTATEMENT — Compile exclusion claim.** The estimator removes each key’s largest measurement, without identifying compilation. For **106/204** matching keys, that largest measurement is not the first case. Describe this as a second-largest timing estimate, or explicitly identify compile-bearing measurements. The L2 calculation itself reproduces **3.808382806 h**, with separate compile allowances.

- **CORRECT — Budget arithmetic.** Ledger spend **1.1017 h** plus projected remaining **5.508382806 h** equals **6.610082806 ≤ 8 h**. **NEEDS-RESTATEMENT** if presented as a worst-case reservation guarantee: substituting L2’s requested walltime yields **8.5517 h**.

- **NOTED — Forgery-only concerns.** Hashes establish consistency, not independent authenticity. All 170 staged manifest entries pass; staged config and audited code match.

No files changed or jobs submitted.

**Overall verdict: NOT CLEAN — resource-estimator regression.**