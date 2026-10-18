# Lane brief: jcp-dims (priority 3: A3, A4, B3, B6)

Read `briefs/COMMON-BRIEF.md` first; it binds you (note §11 disk hygiene, §12 waiting on PIDs).

## Where
- Worktree `worktrees/2026-10-18-jcp-dims`, branch `exp/2026-10-18-jcp-dims` (created by the coordinator
  from `21c175a1b`; vendored code; `sync_github.sh`). Lane dir `experiments/jcp-dims`. Namespace
  `/cluster/tufts/paralab/tawal01/jcp-dims/`. Cap **3** concurrent jobs, **55 GPU-h** incl. contingency,
  with reservation (reuse the P1/qualify `ledger.py` + `cluster/submit.sh`).

## Build on (read report.md + DESIGN.md; copy code with provenance)
`jcp-p1-scaling` (frozen mode lists, Q1/Q2 certifier, kernel harness, statistics), `jcp-qualify`
(FRONT-2D, 3D Q2 machinery and its incomplete 3D round: 16 exposed / 146 open candidates),
`jcp-wide-bank` (configurable M: `w2d.py`, `w3d.py`), `jcp-speed3d`, the 2026-10-01 quadrature lanes
(rule generators; `qcore.py:75–100`, `rules.py:173–186`), `jcp-mechanism`.

## Items owned (plan V2 A3, A4, B3, B6; V3-3, V3-5; V4-1)
- **A3 factorial r × M × m** (2D at 1024², 3D at 65³): panels at fixed m, fixed M (frozen mode lists),
  and fixed (M, m); cost and accuracy per cell; separate representation size from test-space and
  quadrature growth. Q1/Q2 per cell; unqualified cells reported.
- **A4 m\*(r, M):** smallest qualifying point count per rule family over a pre-declared discrete
  candidate list, on own reached states, with both bars (Q2 rollout ≤ 2.5e-4 and ρ ≤ 0.116); "none" is a
  valid outcome. **Includes the 3D front completion:** finish the 3D Q2 search that jcp-qualify stopped
  at its round limit (the 16 exposed candidates first), so the 3D front has qualified members per width
  W256–W1024; also re-time the qualified 3D configurations against speed3d's comparators (C_old,
  C_W768, C_W1024) in paired panels, so a **qualified** 3D speed-up exists (or its absence is recorded).
- **B3 rule families**, staged: (i) Gauss, lattice, Sobol, Smolyak control, 2D Halton as full solves on own
  reached states; (ii) MC and 3D Halton only after their generators pass a Codex code audit; random
  families with repeated draws (≥ 5), intervals across draws.
- **B6 test space:** M/r ∈ {2, 3, 4, 8} plus **M = r as a deliberately weak control** (must degrade);
  Galerkin is a different solve equation (Q^T R = 0) — note qualify found GAL-CN on the median front, so
  test GAL vs LSPG at each M/r. In 3D the M = r control breaks the shell (explicit lowest-r list), labelled.
  A low mixed-frequency test set only if its selection logic and stability checks pass audit (else defer).
- Decision rules frozen in DESIGN.md before data, in the style of AM-4/AM-7 (three-way outcomes,
  precedence, actual eligible populations).
- **Defer order:** B3 stage (ii) → low-mixed-frequency test set → A3 3D fixed-m panel → B6 M/r = 8.
  Never dropped: A4 2D, A4 3D front completion + qualified 3D re-timing, B6 M/r ∈ {2, 4} and M = r.
