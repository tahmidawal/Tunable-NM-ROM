**A6 does not close the compilation finding.** Its unconditional-verdict test is invalid. A6.2 introduces no new blocking defect.

1. **A6.1 projection labelling — CORRECT.** It explicitly identifies charged compilation costs as projections and treats measurements without a backend-compile event as missing.

2. **A6.1 “compile-insensitive” criterion — WRONG.** [DESIGN.md:632](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/DESIGN.md:632) tests only two scenarios: all compilation costs zero, and all compilation costs equal to a common upper scenario value. Agreement does not establish independence from compilation.

   A hypothetical single-mesh counterexample: non-compilation costs are 100 seconds for CP and 110 seconds for ours, with one executable each. Both prescribed scenarios give MATCH because adding the same compile cost preserves the ordering. But CP compilation of 20 seconds and ours of 1 second gives NO-MATCH—even if both lie within the proposed scenario range.

   Moreover, ten times the largest observed compile time is not an established bound on unmeasured executables. Missing entries cannot support unconditional conclusions through this test either.

3. **A6.2 Q1 fallback — CORRECT at design level, pending execution.** The half-candidate mutation matches `mmcore.unsolved_override`. Requiring the unmutated fixture to pass strengthens the control; A3.5 still blocks production unless a mutation fails Q1. No new blocking defect here.

Previously accepted items remain closed; A6 does not alter their governing rules.

Read-only review completed: no files changed, jobs submitted, scratch created, or writes to main or `LAB-LOG.md`.

**Blocking change:** replace A6.1’s unconditional-verdict rule. Either measure each distinct deployed executable as required in round six, or keep affected verdicts explicitly conditional on compilation projections. If retaining a bounded sensitivity claim, test independent adverse assignments for the two sides and label the result conditional on the assumed bounds.