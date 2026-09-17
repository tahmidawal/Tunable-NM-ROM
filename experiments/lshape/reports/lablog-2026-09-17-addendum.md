### lshape (addendum, same day) — fifth job carries the q=R free-bank rung at N=256; it runs at M=1024, not the M=513 named, because M=R+1 is ill-posed on the real banks

Coordinator decision after the interim report: the L-shape is where the "top rung is a linear
reduced model and the cheapest point" claim gets its cleanest test, so the $q=R$ rung is needed.
Job `3784910` (`lsh06`, A100 requested, 14 h, N=256, `config-solve-m1024.json`) submitted
~11:05 EDT, one attempt directory, `squeue` zero-before / one-after; pending on Priority behind
`3784662`/`3`/`4`. Five of eight jobs used.

**Found before spending it (`checks/free_rung_M_sweep.py`, CPU NumPy on the real `lsh02` banks,
untimed).** The free rung is the rank-$R$ least-squares solve of $Bc=f_m$ and is head-independent,
so its error can be computed ahead of the job. At $M=513$ against $R=512$ — one equation over
square — it is 6.28 % worst on `sdf_R512` (8.1× its 0.7766 % floor, cond(B) 6.4e6), 8.49 % on
`smooth_R512`, 8.75 % on `enrich_R512`; at $M=640$ within 1.1–1.2× of the floor; at $M=1024$
it *is* the floor (0.7791 / 0.7161 / 0.6719 % against 0.7766 / 0.7123 / 0.6676 %). The job was
therefore staged at $M=1024$ and the deviation from the named 513 is recorded as a deviation in
DESIGN §A6 with the full table. Consequence for reading it: within-job comparisons hold at any
$M$, but $f_m = Pf$ is charged inside every reduced query, so **no cost in `lsh06` is comparable
with the $M=257$ jobs**; the report must present the two blocks separately.

**Code change, confined to a branch the queued jobs never execute.** On the free rung $B_\perp$
is zero to round-off, so the LM's normalised gradient is a round-off cosine and it would spin its
damping ladder to exit 3 on every query, inflating a linear model's cost. The kernel now skips the
LM there (0 iterations; stationarity is the full-residual `full_stationarity`, computed for every
rung). N=32 smoke with $M=40>R=32$: fires on both primaries, exit 4, full stationarity 1.8e-13,
head-independent output, cheapest reduced subject; NumPy solve audit passes. `lsh03`–`lsh05` have
$M=257<512$ and never reach the branch.

**Flagged, not acted on.** POD-LSPG at $k'=256$ in `lsh03`–`lsh05` is solved against $M=257$
modes — the same one-over-square structure. If it looks poor there, blame the test-space count
before the basis; `lsh06` carries POD at all five ranks under $M=1024$ and is the control.
The queued jobs are not resubmitted.

**Carried to the report regeneration.** Validation-vs-development worst gap goes beside the
headline error, not in a caveat; the generator must key timed subjects by job as well as
`(mesh, name)` or a second N=256 job overwrites `lsh04`; no $K=64$ head until the solves land.
