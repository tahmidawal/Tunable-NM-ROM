# Lane brief: jcp-robust (priority 4: E1 randomised rules, E2 seeds, E5 hardware)

Read `briefs/COMMON-BRIEF.md` first; it binds you (§11–§16: disk hygiene, PID/sacct waits, shared local
slots, shared sbatch lock, raw archives on the cluster).

## Where
Worktree `worktrees/2026-10-19-jcp-robust`, branch `exp/2026-10-19-jcp-robust` (coordinator-created from
`21c175a1b`). Lane dir `experiments/jcp-robust`. Namespace `/cluster/tufts/paralab/tawal01/jcp-robust/`.
Cap **3** concurrent jobs, **75 GPU-h** incl. contingency, with reservation.

## Build on
All finished campaign lanes (read their report.md, EXPORT.json): jcp-p1-scaling, jcp-qualify
(FRONT-2D, 3D Q2 subset), jcp-fom2d, jcp-speed3d, jcp-dims (rule families, 3D front), jcp-heads; the bank
training lanes jcp-wide-bank (3D W1024: 1.14 h bank fit, 1.85 h job), jcp-smooth-bank (retrains are not
bitwise reproducible), the 2D R512 bank provenance.

## Items owned (plan V2 E1, E2, E5)
- **E1 randomised rules:** Cranley–Patterson shifts for the lattice rules and Owen scrambles for Sobol
  (not applicable to Gauss/Smolyak), ≥ 16 draws per qualified front configuration that uses such a rule
  (2D and 3D); distribution of rollout distance to the converged rule and of E0 error over draws; whether
  the **deployed single draw** (seed 0) is typical (rank within the draw distribution). Must-fail control:
  a deliberately bad generating vector must be caught by an independent rule family (Gauss reference).
- **E2 seeds (full recipe):** repeat the complete recipe — bank training, ordering/rotation, rule
  selection — for **3 seeds** (5 if budget allows) for the 2D R512 bank (prefixes r = 128, 256, 384)
  and the 3D W1024 bank (widths 256–1024); then re-run the key priority-1/2 verdicts (H-acc-N, H-acc-r,
  qualified fronts at 1024² / 65³) per seed, and report seed spread with intervals. Record seeds and
  hashes (not bitwise reproducible). Memory pilot first for 3D.
- **E5 hardware:** paired A–B–A comparisons within each physical GPU and job, repeated on A100 and H200
  (and L40S if free) for the P1 mesh-cost panel (subset) and the 2D crossover panel (subset); compare
  only dimensionless ratios across hardware; record clocks, precision, software versions.
- Decision rules frozen in DESIGN.md before data (three-way outcomes, precedence, actual populations).
- **Defer order:** E2 5-seed extension → E5 L40S → E1 3D draws beyond 16 → E2 3D seeds 3. Never dropped:
  E1 2D, E2 2D 3 seeds, E5 A100 vs H200 on the P1 panel subset.
