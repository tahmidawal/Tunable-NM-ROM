## 2026-09-17

### ns2d — ns303 (job 3808498, 3-mode / 6-amplitude family, K=16/R=256, ns203 recipe) COMPLETE: FAILS H-ORACLE (POD-16/oracle 1.39; bar 2.0, below the 1.5 marker) — no Phase-3 job submitted, bar not lowered; ns302 (4× data) still running

Branch `exp/2026-09-17-ns2d`; DESIGN §A12; A100 `pax105`, 3 h 37 m, exit 0, `jax_backend=gpu`, commit `31e0846f`. Collected with checksums, audited (`artifacts/ns303/audit.json`, 23 checks, all match, against the new family's first-8 dev trajectories regenerated locally with the certified FOM, sha `ae172ad3…`), chunk-archived, remote dir deleted. Jobs used 9 of 12.

**Gates (training mesh 256²):** B-RANKCAP pass; B-DATA recorded-new-family (dev `81615ae4…`, train `184f97d7…`); B-ORTH rank 256, κ 63.8; B-FLOOR pass (bank 0.0640 vs POD-256 0.0623, ratio 1.03; bank median 0.0108); H-TRAIN pass (recon median 0.0139); H-SOLVED pass (0.0557); **H-ORACLE FAIL: oracle median 0.0528 vs POD-16 median 0.0737, ratio 1.39, `passes_at_1p5` false**; 4/384 budget exits.

**Per time (medians over 64 dev cases):** oracle 0.0058 / 0.0226 / 0.0818 / 0.1124 / 0.1119 / 0.1106 at t = 0 … 1; POD-16 0.0106 / 0.0503 / 0.1223 / 0.1362 / 0.1319 / 0.1242; bank floor 0.0005 → 0.0099. POD-16/oracle = 1.83 at t=0, 1.44 evolved. Held-out/training = 3.81.

**Reading.** Halving the family's intrinsic dimension (14 → 8) lowered every error ~4× (POD-16 0.240 → 0.074, oracle 0.202 → 0.053, bank floor 0.012 → 0.003) but not the head-to-linear ratio (1.19 → 1.39) nor the generalisation gap (4.0 → 3.8). With §A7 (K=32: 1.15) and §A10 (data slope decelerating, regularisation inert), this auto-decoder head class does not beat POD-K by 2× on held-out decaying 2D NS at any family dimension or data budget tried. The reserved Phase-3 job stays reserved for a passing head; ns302 (predicted 1.25–1.35 by §A10) runs to completion so the prediction is tested.

**Retracted / corrected.** None.

**Open.** ns302 (3808495) running ~3 h 45 m; when it lands the §A9 cell is closed either way.
