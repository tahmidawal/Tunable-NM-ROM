FIX-FIRST

Three acceptance gaps remain:

- **A14.5’s memory argument is incorrect.** Phase two drops `qp` and retains **main + qpp** ([dtime.py:135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/dtime.py:135)). The smoke exercised `(256,320)` then `(256,384)`, not `(320,384)` as claimed in [DESIGN.md:857](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md:857). Its 22.1465 GB peak therefore does not establish the required production footprint coverage.

- **Injected-fault acceptance is too weak.** [analyze2d.py:412](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:412) tests “not all hashes passed,” rather than “every hash failed.” An in-memory reproduction with only one failed split hash and all fused hashes passing still accepts the control and leaves another subject’s cost usable. Require complete control inventory and failure of every expected hash check.

- **Offline identity does not enforce dev6 coverage.** [analyze2d.py:382](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:382) checks whichever cases appear in `S_*.npz`. An in-memory fixture containing only one case’s `W`, with six-case certification/timing records, remains both usable and qualified. Require exactly the six unique dev6 IDs and matching `W` dimensions before comparison.

Other checks passed:

- Identity uses case-ID matching, the stated normalization and `1e-10` threshold, and prefers F2 archives. All **298 registered subjects** have local accuracy `W` and dev6 coverage; slim copies suffice. Reading smoke archives in place reproduced **7 subjects**, worst identity **1.1773e-15**.
- Split/fused hash failures, failed equivalence, excessive identity distance and failed doubled-work ratio veto costs in synthetic checks. Smoke doubled median is **2.002665**, with iterations and identity passing.
- `_controls` does **not** break the consumers: elasticity and Shapley access explicit subject labels and require `usable` and `qualified` ([analyze2d.py:462](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:462), [analyze2d.py:689](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:689)).
- Independently reconstructed **298 registered + 1 dedicated control**, without displacement. The no-`qp` control follows guarded paths in `dtime`.
- Certificate reproduces **5.286026058655773 h → 8 h**, including config hash. All **180 manifest entries / 176 provenance files** match staged content and HEAD `87092504c`.
- Staging specifies H200, `gpu`, exit-42 backend preflight, f64/highest and paralab paths.
- Budget: **28.6212 + 12 + 8 + 2 = 50.6212 ≤ 55**, leaving **4.3788 h**. Two SSH checks showed only pending `6056422 / jd_t3s`: lane **1 < 3**, account **1 < 8**.

No files changed, jobs submitted, or scratch files created.