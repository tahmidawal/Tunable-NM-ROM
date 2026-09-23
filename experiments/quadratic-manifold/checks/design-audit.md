# Independent pre-job audit — quadratic-manifold lane (`qmn101`), 2026-09-22

Commissioned before any GPU job of this lane, against `DESIGN.md`, `qman.py`, the `qman` additions
to `panel.py`, `config-256-qman.json`, `COPIED-FROM.json` and `audit_panel.py`, with the parent
harness (`../b-panel/DESIGN.md`, `../head-ablation/{arms,ablation}.py`) as read-only context. The
auditor was read-only and wrote nothing.

**Codex could not run.** `codex exec -m gpt-6-astra -s read-only` was launched first, as the
protocol asks. Its sandbox failed to start on this box — `bwrap: loopback: Failed RTM_NEWADDR:
Operation not permitted` — and it returned an explicit audit-access blocker rather than an
opinion, having read nothing. This is the recorded failure mode of this machine (b-panel DESIGN
§A1 hit the same wall on 2026-09-17 for a different reason). An independent subagent auditor was
commissioned with the identical eight-question brief; its report follows verbatim. Dispositions
are in `DESIGN.md` §A1.

---

## Verdict

> **Do not submit as-is.** One finding is blocking, and it is exactly the failure mode you asked
> me to hunt for: the pre-registered ridge-selection rule systematically hands the baseline its
> weakest possible $W$, and the lane's own smoke already shows the consequence. Everything else is
> sound — the trial map is the papers' method, the harness parity is genuine, there is no
> evaluation-data selection, the timing contract is comparable, and the job fits on an 80 GB A100
> with room to spare.

## 1. Is the implemented trial map the method GWW/BF describe?

**Yes, and where it deviates it deviates in the baseline's favour — with one exception (§5).**

| item | code | vs the papers | direction |
|---|---|---|---|
| $u_{\rm ref}$ = snapshot mean | `qman.py:60` | GWW use a reference state, commonly the mean | neutral |
| $V_r$ = POD of **centred** snapshots | `qman.py:62` | GWW/BF centre before POD | **correct** (and stronger than the uncentred `pod` arms) |
| $W$ fitted against $E=S_c-VA$ | `qman.py:63-64` | algebraically identical to GWW's $\min_W\lVert S_c-VA-W\Pi\rVert_F^2$ | neutral — **not** a deviation |
| $W$ unrestricted in $\mathbb R^{n\times P}$ | `qman.py:87` | BF restrict $W=\bar V\bar W$ to a truncated POD complement | **STRENGTHENS** (BF's form is a special case) |
| `vech` with $W$ absorbing the off-diagonal 2 | `qman.py:44-45` | equivalent to the full Kronecker form | neutral |
| ridge $\gamma s\lVert W\rVert_F^2$, $s=\mathrm{tr}(\Pi\Pi^\top)/P$ | `qman.py:74,80` | GWW use $\lambda\lVert\bar W\rVert_F^2$; scaling only makes $\gamma$ dimensionless | neutral |
| $\gamma$ **chosen by random-snapshot holdout** | `qman.py:68-71` | GWW/BF choose $\lambda$ by L-curve or by ROM performance | **WEAKENS — §5, BLOCKING** |

The `head`/`bank` ordering matches the $\Pi$ ordering (`np.triu_indices` at `qman.py:44` and
`qman.py:65` — the same routine), so the map is self-consistent. No handicap in the *form* of the
map. `demo()` (`qman.py:116-153`) is a real check, and its refusal to assert exact recovery of a
planted $(V,W)$ is correct reasoning, not a weakened test.

## 2. Is the shared contract fair to the quadratic manifold?

**Mostly yes; three items to flag, none blocking.**

- **$M=4r$ test modes** (`panel.py:145`). Parity is by *solved unknowns*, and the NM-ROM arms
  follow the same rule against a bank far wider than $K+q$, so the precedent is consistent. But at
  $r=64$ the `quad` arm has a 2145-column trial space tested by 256 sine modes; the quadratic
  content lives precisely in the POD complement, which is where the low sine modes are blindest.
  **SHOULD-FIX:** add one $M$-sensitivity arm (e.g. `qman32_quad` at $M=512$) so a loss cannot be
  blamed on test-space truncation. Without it, "the quadratic manifold lost" and "$M=4r$ was too
  small for it" are not separable.
- **Gauss-Jordan vs LU at 64 unknowns** (`panel.py:705` → `arms.py:284`, `gauss_jordan_max: 64`).
  `qman64` and `pod64` both get the *unpivoted* GJ solve, so it is parity — but
  $J=V+2W(a\otimes\cdot)$ can be far worse conditioned than POD's constant $J$, so the unpivoted
  solve is a bigger risk for `qman64`. **NOTE:** if `qman64_quad` shows rejected/budget exits that
  `pod64` does not, check this before concluding anything about the method.
- **Dense quadrature only** (`panel.py:146`). Fair against `pod` (also dense), **not** fair against
  the paper's headline NM-ROM rows, which get a certified EQ rule and the `L4` kernel. GWW and BF
  both pair the quadratic manifold with hyper-reduction (DEIM/ECSW); this lane gives it none.
  **SHOULD-FIX (reporting):** the headline cost ratio must be `qman` vs the NM-ROM **dense** arm,
  with the absence of hyper-reduction for the baseline stated as a limitation. §7 does not say this.

Not problems: the trust radius (`panel.py:704`, same `radius()` rule, identical between `quad` and
`lin`); the fixed-Gauss initializer (unbounded trust on the IC fit, `arms.py:329`); the 600-step
budget (shared); the candidate library (the arm's own 3328 snapshot coordinates, same as `pod`).

## 3. Selection on evaluation data?

**None. Clean.**

- $\gamma$ is chosen inside `qman.fit` from `Ut` = `LD.generate_snapshots(..., train_physical, ...)`
  (`panel.py:331,345`); `train_physical` is asserted disjoint from the evaluation cases at
  `panel.py:258`. The split is on snapshot columns (`qman.py:68-71`), seeded, never on fields.
- The $r$ ladder, $M=4r$, the variants, `gtol`, the $\gamma$ grid, the holdout fraction and
  `qman_cold_axis_points` are all in `config-256-qman.json:191-212` and pre-registered in DESIGN
  §3.1/§3.2. `priority_override` names every declared subject; `declare_subjects` asserts it
  (`panel.py:160-164`) and `smoke_panel.py:178-186` asserts the build order.
- The prediction "the quadratic gain is largest at small $r$ and shrinks or reverses by $r=64$" is
  recorded before the job (DESIGN:128-131). Good practice.

**NOTE:** DESIGN §A1 (lines 284-288) says the pre-job audit "was run" and points at
`checks/design-audit.md`, which does not exist. A committed document asserting a check that has not
landed is the same class of problem the lab log exists to prevent.

## 4. Can the timing be trusted?

**Yes.** `gpu_seconds` is measured after `device_put` + `block_until_ready` and closed by
`block_until_ready(value)` (`panel.py:798-803`); `host_seconds` brackets the upload and the field
copy (`panel.py:797-805`); 0.25 s burn-in precedes every invocation (`panel.py:796`); order is
randomised per (rep, case) (`panel.py:793`); every arm is built and warmed outside the loop
(`panel.py:734-784`). `qman` returns the identical 12-tuple and the identical **six** output fields
through the identical `e.output_field` path (`arms.py:354`) — same contract as `pod` — and
`panel.py:825-826` correctly routes the non-`rom` families to `v[8]/v[9]`. No setup leaks into the
timed region.

- **SHOULD-FIX (DESIGN §3.2 is factually wrong).** §3.2:139-142 claims the 96-point initializer rule
  "is offline setup only … the timed cost is unchanged". It is not:
  `ui = e.sample_field(u0, xy, L) * w` and `y = Q.T @ ui` run **inside the jitted, timed query**
  (`arms.py:335-337`), over all 9216 points against a $9216\times D$ factor. The magnitude is small
  (order microseconds at $r=64$, kernel count unchanged), so I would not hold the job for it — but
  the sentence must be corrected, and the cleaner fix is to set the axis rule **per rank** (48
  suffices through $r=32$, since $2\times561<2304$; only $r=64$ needs 96). That makes
  `qman8/16/32` bitwise cost-comparable with `pod8/16/32` and confines the deviation to one rung.

## 5. Correctness of `qman.fit` — **BLOCKING**

The normal equations, the ridge scaling and the refit are algebraically right. Three defects.

**(a) BLOCKING — the held-out split is leaky, and it demonstrably selects $\gamma=0$.**
`qman.py:68-71` splits the 3328 snapshot *columns* uniformly at random. But
`LD.generate_snapshots` (`head-ablation/ladder.py:65-75`) concatenates trajectories:
`keep = arange(0, 51, 2)` gives 26 states per trajectory, so column $j$ belongs to trajectory
$\lfloor j/26\rfloor$. Consecutive columns are $\Delta t = 0.01$ apart in a smooth viscous flow —
near-duplicates. A uniform 20 % column holdout therefore leaves almost every held-out snapshot with
its own temporal neighbours in the 80 % kept. The "held-out" error is close to an in-sample error,
so **the criterion cannot see overfitting and rewards interpolation**. There are 128 genuinely
independent trajectories fitting $P=2080$ coefficients per row at $r=64$.

This is not speculative. The lane's own smoke (`checks/smoke-panel.json`) shows it:

| rung | selected $\gamma$ | in-sample residual | worst evolved | converged | GPU ms vs its own `lin` |
|---|---|---|---|---|---|
| $r=4$ | $10^{-4}$ | 3.8e-2 (lin 1.2e-1) | **67.3 %** vs lin 71.3 % | yes | 1.28× |
| $r=8$ | **0.0** | 4.5e-5 (lin 3.6e-2) | **83.3 %** vs lin 66.4 % | **no** | **9.7×** |

The one rung where the rule picked $\gamma=0$ is the one rung whose snapshot reconstruction is 800×
better, whose ROM error is 25 % **worse than its own linear control**, which costs 9.7× that
control, and which **fails the admissibility gate**. That is the signature of an overfit $W$
producing a manifold LSPG cannot solve on. At $256^2$, $r=32$ ($P=528$) and $r=64$ ($P=2080$) sit in
the same regime — the regime DESIGN:126-129 itself identifies as the one where "the ridge must do
real work". As written, the most likely outcome of `qmn101` is "the quadratic manifold does not
converge and loses to POD", caused by the selection rule, not by the method. That is the
accidentally-handicapped baseline.

*Fix:* split by **trajectory index** — `traj = np.arange(Ns) // states_per_trajectory`, hold out
~26 of the 128 trajectories. `generate_snapshots` already returns `states_per_trajectory`.

*Cheap discriminating check first:* rerun the 64-interval smoke with the $r=8$ $\gamma$ forced to
$10^{-4}$. If `qman8_quad` then converges and beats `qman8_lin`, the diagnosis is confirmed and the
amendment rests on a measurement rather than an argument. Sub-minute and local.

**(b) SHOULD-FIX (high) — a NaN in the $\gamma$ loop is locked in, and it kills the whole job.**
`qman.py:78-85`: `best` is seeded by the *first* gamma, and the grid starts at `0.0`. At $r=64$,
$\Pi\Pi^\top$ is $2080\times2080$ from products spanning many decades — `jnp.linalg.solve` at
$\gamma=0$ can return non-finite. `err` is then `nan`, every later `err < nan` is `False`, and
$\gamma=0$ is locked in with a NaN $W$, which reaches the bank and kills the job at
`assert np.isfinite(f).all()` (`panel.py:807`) — **outside** the OOM-tolerant `try`, in the middle
of the timed loop, after hours of work. Guard the comparison and assert `isfinite(W)`.

**(c) NOTE — two small inconsistencies.** `scale` is computed from the 80 % Gram `Mtr`
(`qman.py:74`) but applied to the full `Pi @ Pi.T` in the refit (`qman.py:87`), so the refit is
regularised ~0.8× more weakly than the selected point. And `gram_condition` divides by `ev[0]`,
which `eigvalsh` returns *negative* for a numerically singular PSD Gram — the diagnostic §7 promises
will "explain why" can come out negative and meaningless. Also **record $\lVert W\rVert_F$ per
rung**: it is the single number that would have made (a) visible in the smoke JSON, and it is
recorded nowhere.

## 6. Correctness of the wiring

**Correct, no finding.** `bank_columns(m, True) @ head(r, True)(a)` matches the stated map
(asserted to $10^{-13}$, `qman.py:143`); `jacfwd` matches the analytic Jacobian (`qman.py:148-151`);
the initial fit minimises $\lVert R\eta(z)-Q^\top u_i\rVert$ (`arms.py:329`), which equals the
in-span part of $\lVert G_i\eta(z)-u_i\rVert$ since $G_i=QR$, so the argmin is unchanged and trust is
$\infty$ there; `GridBank`'s zero padding is correct because the problem is homogeneous-Dirichlet
(`engines.py:37`), so $u_{\rm ref}$, $V$ and $W$ get the same treatment as every POD mode, and the
off-grid bilinear sampling is the interpolation the supplied field already receives.

- **SHOULD-FIX — the missing diagnostic.** `panel.py:600-612` builds a best-found representation
  floor for `pod` and `panel.py:556-598` one for `rom`; there is **none for `qman`**, so
  `audit_panel.py:320-333` leaves `best_found_percent = None` for every quadratic row. If the qman
  arms come back with large errors you will not be able to say whether the manifold cannot
  represent the field or the LSPG solve cannot find it on it — exactly the question finding (a)
  raises. `A.make_reconstruction(head, r_, L, cfg['recon_budget'])` with starts from the arm's own
  coordinate cloud is a few lines, untimed, OOM-tolerant. The `t0` compression metric partly covers
  this but runs through the initializer, so it conflates the two.

## 7. Memory on an 80 GB A100

**It fits — roughly 10 GB steady-state against a 72 GB budget at `--mem-fraction 0.90`.** `qman`
banks 3.2 GB + retained `W`/`V` 1.5 GB; `pod` banks + `Vmodes` 1.2 GB; per-arm $\Phi$ copies
2.35 GB; `dense_data` cache 0.6 GB; colds 0.4 GB (the $r=64$ `quad` cold is the largest single one
at ~250 MB). Transient peak is the **fit**, not the subjects: during the $r=64$ rung
`Ut + Sc + E + E[:,tr] + Ctr + Wg + W` ≈ 10 GB, all before any subject is built. No OOM risk.
`priority_override` is **correct**: the only arms the OOM rule can take are exactly the two DESIGN
§7 permits to be lost. *Optional:* `del qman_maps[r]['W']` after each `quad` arm would return 1.5 GB;
unnecessary at this budget.

## 8. Is DESIGN.md enough, and is §7 honest about a win?

**§7 is honest — unusually so.** It leads with the losing-for-us outcome, commits the row to the
**main table** rather than an appendix, commits to rewriting the positioning claim, and forbids a
second job to improve a number. §3.1 records a directional prediction before the data exists. §3.3
refuses to claim the `lin` arm is a POD-LSPG row. §5 pre-commits the convergence rule verbatim.

Four gaps:

1. **SHOULD-FIX.** §7's cost bullet does not say the `qman` rows carry dense quadrature while the
   NM-ROM's headline rows carry a certified EQ rule and an optimised kernel. A speedup quoted
   against those without that sentence measures the absence of hyper-reduction for the baseline,
   which GWW and BF both supply. Pre-register the dense-vs-dense comparison as the fair one.
2. **SHOULD-FIX.** §5's "Non-convergence … is a finding about the method at that $r$, not a licence
   to retune it" has no carve-out for harness-attributable non-convergence — an over-fitted $W$ from
   (a), the unpivoted GJ solve at $r=64$, an IC stall. The smoke *already* produced a non-converged
   `quad` arm. Without the carve-out, a numerical artifact gets published as a property of
   Geelen–Wright–Willcox.
3. **NOTE.** §6's `qman` gates are asserted **only in the smoke** at $r=4,8$ — not in the driver and
   not in `audit_panel.py` — so nothing checks them at $r=64$, the rung that can actually break.
4. **NOTE.** §3.2's timed-cost claim is wrong (§4), and §A1 cites a file that does not exist.

## Recommended actions before staging `qmn101`

**Blocking:** amend DESIGN §2 and change `qman.py:68-71` to a **trajectory-level** holdout; confirm
with the forced-$\gamma$ smoke check first.
**Before submit, cheap:** NaN guard + `assert isfinite(W)`; record $\lVert W\rVert_F$ and fix the
`gram_condition` sign; move the three §6 `qman` gates into the driver/audit; add the `qman`
best-found reconstruction floor; per-rank cold axis (48 through $r=32$, 96 only at $r=64$); correct
§3.2 and §7's cost bullet; add the §5 carve-out.
**Optional:** one $M$-sensitivity `qman` arm.
