## 2026-09-17

### ns2d — ns301 (job 3808493, head-only data-scaling diagnosis on the frozen ns203 bank) COMPLETE: "ambiguous" under every regime by the §A9 rule, but the slope decelerates from −0.3/−0.4 per doubling to −0.06/−0.12, and regularisation does not move it — leans head-limited; ns302/ns303/ns304 still running

Branch `exp/2026-09-17-ns2d`; DESIGN §A10; A100 `pax052`, 1 h 45 m, exit 0, `jax_backend=gpu`, commit `31e0846f`. Collected with checksums, audited (`artifacts/ns301/audit.json`: 107 checks, 104 match), chunk-archived, remote dir deleted. Jobs used 9 of 12.

**Dev-report oracle median (dev cases 0–31) vs training subset, train-oracle in parentheses; POD-16 of the same subset in the last column:**

| n | plain 512×3 | reg (wd 1e-4 + early stop) | reg_small 128×2 | POD-16 |
|---|---|---|---|---|
| 128 | 0.2749 (0.0167) | 0.2611 (0.0566) | 0.2466 (0.0730) | 0.2358 |
| 256 | 0.2069 (0.0301) | 0.2102 (0.0651) | 0.2134 (0.1193) | 0.2296 |
| 512 | 0.1983 (0.0481) | 0.1939 (0.0629) | 0.2009 (0.1292) | 0.2277 |

Log-log slopes −0.236 / −0.215 / −0.148 (all in the pre-registered ambiguous band −0.25…−0.10); successive slopes −0.41→−0.06, −0.31→−0.12, −0.21→−0.09; reg vs plain at n=512: +2.2 % / −1.3 % (bar 10 %) — regularisation does not move it. At n=128 the head is worse than POD-16 (ratio 0.86–0.96). Frozen-bank caveat (pre-registered): the bank saw all 512 trajectories, so small-n arms are optimistic and the flattening is the stronger reading.

**Implication (stated before ns302/ns303 land).** Extrapolating the last slope to n=2048 predicts POD-16/oracle ≈ 1.25–1.35 for ns302 — the 4× data arm is unlikely to reach 2.0. The diagnosis leans to "needs a different head / a lower-dimensional manifold", so ns303 (8-dim family) is the arm that can pass; the reserved Phase-3 job stays reserved for whichever passes (expected ns303) and is not spent on a failed head.

**Retracted / corrected.** None. **Audit finding recorded:** an independent SciPy LM beats the driver's 8-start oracle on 1/48 audited states in 3 of 9 arms (3.8–16.8 % on that state; medians unchanged to 1e-10) — the oracle is a best-found upper bound, as §A8 already says.

**Open.** ns302 (3808495), ns303 (3808498), ns304 (3808502) running ~2 h at the time of writing.
