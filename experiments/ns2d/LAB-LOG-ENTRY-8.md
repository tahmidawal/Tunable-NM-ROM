## 2026-09-17

### ns2d — ns304 (job 3808502, EXPLORATORY q-ladder on the ns204 K=32/R=512 manifold, fixed M=2176) COMPLETE and audited: the correction rank closes the manifold gap to the bank floor (only fully at q=R), the solved error falls 11× at 3.9× cost but is non-monotone at the first rung, and every neural rung is dominated by POD-LSPG and by the full-order solve — exploratory after a failed gate, not a paper Phase-3 number

Branch `exp/2026-09-17-ns2d`; DESIGN §A11; A100 `pax105`, 3 h 11 m, exit 0, `jax_backend=gpu`, commit `31e0846f`; complete (416/416 invocations, nothing dropped — the FFT tensor and the shared-M device copy made it fast). Collected with checksums, audited (`artifacts/ns304/audit.json`, 351 checks, all match), chunk-archived (403 MB, 9 parts), remote dir deleted. Jobs used 9 of 12; ns302/ns303 still running (~3 h 20 m).

**Ladder (worst / median evolved over 8 dev cases; per-query median ms of 3 timed reps; POD-LSPG at matched K+q in the same job):**

| q | K+q | ROM worst | ROM median | ms | POD-LSPG worst | POD ms | manifold (layer 2) median |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 0.688 | 0.414 | 13388 | 0.546 | 212 | 0.106 |
| 32 | 64 | 0.708 | 0.368 | 16979 | 0.355 | 431 | 0.103 |
| 64 | 96 | 0.615 | 0.327 | 21290 | 0.277 | 833 | 0.100 |
| 128 | 160 | 0.566 | 0.257 | 29012 | 0.172 | 2398 | 0.089 |
| 256 | 288 | 0.331 | 0.139 | 46268 | 0.095 | 9964 | 0.043 |
| 512 | 544 | 0.060 | 0.024 | 52592 | 0.042 | 53903 | 0.0023 (= bank floor) |

FOM in the same job: ntol 1e-3 → 421 ms, worst evolved 4.1e-5; converged (1e-11) 1178 ms. Non-dominated set on (ms, worst evolved): `pod_k32` and the FOM tolerance ladder only — **no neural rung**. R-LADDER (exploratory): monotone worst = no (q0→q32: 0.688→0.708), monotone median = yes, gain 11.4×, cost 3.9× → FAIL on monotonicity; R-CONV pass (zero budget exits).

**Reading (§A11).** The mechanism exists on a degree-2 residual: q buys back the head restriction (manifold 0.106 → 0.043 at q=256 → bank floor at q=R) and the solved median error falls monotonically; but the solve layer sits 4–10× above the manifold layer at every rung, and on a manifold that fails H-ORACLE POD-LSPG is better and cheaper everywhere — the falsification clause's named outcome for this head. No reduced model beats the FOM on cost (as the design said in advance).

**Retracted / corrected.** None. Train-cohort hash mismatched on `pax105` (passed by the §A4 inferred-from-dev rule, dev reference by value 3.6e-16) — recorded.

**Open.** ns302 (3808495) and ns303 (3808498) running; the reserved Phase-3 job is for whichever passes H-ORACLE (expected ns303 per §A10). Whether a passing head changes the non-dominated set is the remaining question.
