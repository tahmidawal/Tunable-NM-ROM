# Handoff: ideation session continuing from the 2026-08-26 overnight run

Written 2026-08-26 by the session that ran the overnight stages 1–3, for its own compacted
continuation. **The user (Tahmid) wants to KEEP IDEATING — this is a conversation, not an
execution run.** Do not create worktrees, launch jobs, or start experiments without asking;
propose and discuss. If this file disagrees with LAB-LOG.md, the lab log wins.

## Where the project stands (read these if numbers are needed)

1. `LAB-LOG.md` — "Where things stand — 2026-08-26" (rewritten by the overnight session).
2. `reports/2026-08-26-exact-linear-terms-and-gradient-eq.md` — findings F1–F8, generated
   tables T-X1..T-X5. THE source of every number.
3. `understand/2026-08-26-exact-linear-and-gradient-eq-explained.md` — the plain-language
   version (Codex-written, numbers verified).
4. `understand/2026-08-26-overnight-notes.md` — chronology of the night.

One-paragraph state: exact linear terms (`A = ΦᵀG` precomputed; only advection sampled)
are a zero-cost win — rollout error −18% at N=1024, −11% at N=256, paired cost unchanged
(1.87–1.90× at N=1024). Same-target NNLS refits and learned node positions both work ONLY
where the quadrature binds (coarse m=256; learned nodes beat every convex fit there) and
are clean negatives at the fine budget, where all quadratures tie because **`h`'s
generalisation is the binding error** (reaches only ~1/25–1/35 of the span floor,
resolution-independent). One negative-transfer surprise: same-target refits HURT test
rollouts on N=256 dense_mid (+13%) despite big held-out gains — state-conditioned fits
overfit their fit distribution. Branch `exp/2026-08-26-eq-learned` has all code + 14
result sets; cluster namespace `exlin/` is empty; nothing is running.

## The conversation so far (post-run)

- User asked whether the new decoder was trained → no; everything ran on frozen committed
  checkpoints by design (the handoff they approved). Only node positions were learned.
- User then asked: **"Shouldn't training the decoder along with the NM-ROM train the most
  optimized decoder for this case?"** — the joint / residual-aware end-to-end training
  question. I argued against, on four measured grounds: (1) the error budget points at
  `h` generalisation, not the decoder or the quadrature; (2) collusion — training against
  the SAMPLED residual lets the network fool the m nodes (we measured the miniature
  version: stage-2 weights overfit their fit states, +13% test error); (3) the honest
  teacher (full-grid residual) is unaffordable per training step at N=1024; (4) the
  FREEZE_WDEC scar — "optimizing" a component already at its optimum against a different
  loss has destroyed good initializations in this project before. I offered the
  falsifiable cheap version: residual-aware fine-tuning of **h only**, full-grid teacher
  at reduced N, ladder-certified. My stated prediction: null result.
- User said they were confused; I gave the plain-language recap (the three experiments as
  "stop sampling the easy part / pick better points / learn the points"). Then they said:
  **"Actually I wanna keep ideating"** → this handoff.

## Open ideation threads (where to pick up)

The user has NOT chosen a direction. Live options on the table, with my standing
recommendations:

1. **`h` generalisation push** — my recommendation; the measured ~30× headroom. Untried
   levers: capacity + μ-density together; test-time residual refinement of `h` (refine
   `h`'s output at solve time against the now-cheap exact-linear residual); possibly
   μ-conditioned h. Nothing designed in detail yet — good ideation target.
2. **Joint decoder+ROM training** — the user's own suggestion; I argued it's unmotivated
   but offered the h-only residual-aware fine-tune as the safe falsifiable version. If
   the user still wants it, design the arm rather than re-litigating.
3. **Poisson quadrature-free cell** — designed, not run: whole residual becomes
   `(Λ⊙ΦᵀG)h − Φᵀf`, no EQ at all. Small confirmation cell; also makes a nice paper
   point (elliptic case needs NO quadrature at all under this architecture).
4. **Merge decisions** — consolidated / eq-fidelity-ladder / eq-learned branches; the
   user must decide; do not merge unasked. Also: adopt exlin residual as default.
5. **Paper framing** — the advisor asked for L2/integral/gradient/Hessian errors
   separately ("the gradient is the holy grail") and "learn the quadrature"; both are now
   answered with measurements (ladder + F8). A meeting-ready story exists: quadrature
   measured → fixed for free where fixable → learning it pays only at coarse budgets →
   the real frontier is h. Could become slides/report for the advisor.

## How to talk to this user (learned this session)

Plain language first; keep the math but explain it ("stop sampling the easy part" beat
any formula). They pushed back hard on dense/robotic writing once already. Short
paragraphs, lead with the takeaway, numbers only when they carry the point. When they ask
"should we X?", give a recommendation with the measured reason, then a cheap falsifiable
test — not a lecture. Ask before creating any worktree (CLAUDE.md rule, and they care).

## Mechanics reminders (if ideation turns into execution)

Ask first, then: worktree from `exp/2026-08-26-eq-learned` (NOT main, NOT consolidated —
eq-learned has the exlin residual), date-prefixed name, own cluster namespace, the usual
gates; the exlin residual is the default going forward; certify quadrature-anything with
the ladder, never NNLS rel-fit; N=64 smoke locally before any cluster submission.
