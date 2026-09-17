# Pre-submission design audit of b-qxm (2026-09-17)

Codex (the protocol's independent auditor) was over its usage limit until 19 Sep 2026
(`ERROR: You've hit your usage limit`), so the audit was run by a fresh Claude Opus agent
with no lane context, read-only, over `DESIGN.md`, `q_xm.py`, `make_configs.py`, the three
configs, `audit_xm.py`, `smoke_xm.py`, `probe_memory.py`, `cluster/*.py` and
`checks/comparators.json`, with the parent lane's code as context. Its findings, verbatim in
substance, and what was done with each:

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | should-fix | `q_xm.py` records the fresh-bank-vs-shared-bank check and then shares unconditionally; a failed check would leave `A` formed from one bank and the residual using another. | **accepted** — `assert same` added |
| 2 | note | $t=0$ invariance in $M$ holds in exact arithmetic (IC fit reads only `cold` and `C`, `topfix.py:182-188`, `varpro.py:292`); but each $M$ is its own jitted program and a $10^{-12}$ gate could trip on an ulp-level scheduling difference. | **accepted in part** — gate re-tiered: blocking at $10^{-6}$ (the IC stationarity tolerance), $10^{-12}$ and bitwise as probes. Computing $t=0$ outside the per-$M$ program was **rejected**: it would change the timed contract |
| 3 | should-fix | DESIGN said 26 cells / 17 named cells; the configs run 28 / 16. | **accepted** — counts corrected |
| 4 | note | All 46 comparator pairs resolve; `q0_M64` sources agree to $5.8\times10^{-13}$; tier logic correct. | — |
| 5 | note | DESIGN's "$\le 3\times10^{-9}$ across GPUs" is looser than the $10^{-9}$ gate; the archive supports $6\times10^{-13}$ at $q=0$. | **accepted** — sentence corrected |
| 6 | blocking | §4.3 needs $c(64, 4(K+q)) = c(64, 320)$ within S1; S1 did not run $(64,320)$. | **accepted** — cell added to S1 |
| 7 | should-fix | §4.3 says "doubling" but the sweep steps $512 \to 1088 \to 2048$. | **accepted** — reworded |
| 8 | should-fix | The generator silently dropped non-converged rungs and quoted a span over the shortened ladder, contradicting §5. | **accepted** — span reported *unavailable* if any rung of the column is not converged |
| 9 | should-fix | H(rank)'s lever $S_M(q) \ge 1.5\times$ could be flipped by the near-square $(64,128)$ cell alone. | **accepted** — the lever reads cells with $M \ge 2(K+q)$; the cell stays in the sweep as data |
| 10 | should-fix | A fidelity pair whose comparator metrics are null would pass vacuously and count toward the "two unconditional reproductions" gate. | **accepted** — a pair must compare $\ge 2$ metrics |
| 11 | note | `expected_directions_sha256` inherited from the parent matches no reproducible source. | **accepted** — set to `None`; source hashes kept as probes |
| 12 | blocking | The memory probe had not finished $(64,4096)$; "$\le 10$ GB measured" was an extrapolation; the probe measures in-pool bytes, not the out-of-pool CUBIN side that killed `qtd01`. | **accepted** — `nvidia-smi` process memory recorded (33.3 GB against a 32 GB pool after five subjects, so ~1.3 GB out-of-pool); the wording in `stage.py`/DESIGN now says what was measured; two reserve attempts earmarked for an `h100` resubmit |
| 13 | blocking | Smoke gate 1 failed at $2.14\times10^{-7}$ while the identical construction in `smoke_gate1.py` passed at $1.56\times10^{-14}$ three times; must be root-caused, and nothing was committed yet. | **accepted** — analysed in DESIGN §9 A0 (IC LM path sensitivity to an ulp-level GEMM difference under a contended GPU autotuner); gate 1 recorded with IC iterations and `XLA_FLAGS`, asserted at the end; the cluster's in-job fidelity gates are the enforcement point; committed before staging |
| 14 | should-fix | The Gauss–Jordan $\to$ LU switch at $K+q > 64$ coincides with the G1/G2 split and the $q = 32 \to 64$ rung, with no control. | **accepted** — `(32, 1088)` re-run with LU in G1 as a labelled solver control, excluded from spans |
| 15 | should-fix | The generator formed a "fixed-$M$" span for every $M$ with two rows, including $M = 128$ from a cross-job pair. | **accepted** — only the declared `fixed256`/`fixed1088` columns are spans; the rest is one informational line |
| 16 | note | The all-times metric is $t=0$-dominated, so its $\Delta_M \approx 0$ is an artefact. | **accepted** — caveat printed beside that table |
| 17 | note | DESIGN §7 promised `checks/smoke-xm.json` and `checks/probe-memory.json`; only logs existed at audit time. | **accepted** — both are written when the runs complete |
