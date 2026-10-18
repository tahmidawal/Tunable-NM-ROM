**Overall: NEEDS-RESTATEMENT.** A3 resolves three prior blockers and most of the fourth. Remaining issues concern fallback-budget transitions and an incompletely specified TH06 schedule.

References: `D` = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/DESIGN.md); `Brief` / `Common` = campaign briefs; `Combined` / `P1` = corresponding lane experiment directories.

| `codex-design-3.md` blocker | A3 disposition | Judgment and evidence |
|---|---|---|
| 1. Reservation / defer semantics | **PARTIALLY** | **NEEDS-RESTATEMENT.** Exact defer order restored; proposed job excluded from outstanding reservations; unknown fallbacks receive an allowance. But releasing that allowance and reserving actual fallbacks is not explicitly connected; see blocker 1 below. `D:521–531`; `Brief:43–44`. |
| 2. Zero-scale independent comparator | **RESOLVED** | **CORRECT.** Recovering solver residual as `tolratio × 1e-300` matches the stored quantity; overflow explicitly fails G-cert. This compares two independently obtained quantities. `D:533–537`; `P1/p1core.py:192–218`. |
| 3. Positive GAL evidence / attribution | **RESOLVED** | **CORRECT.** Actual JSON reports `cases=38`, `verified=38`; producer defines verification using `nfail==0` and finite coefficients. Failure now opens diagnosis instead of uniquely blaming the certifier. `D:539–544`; `Combined/checks/analysis-c2grid-2d_e0.json:4706–4709`; `Combined/c2grid.py:126–130`. |
| 4. TH06 after the smoke gate | **RESOLVED** | **CORRECT.** All six cases now run in smoke, satisfying the sequencing requirement. This resolves placement, not the schedule omission below. `D:546–548`; `Common:18–22`. |

Additional adversarial checks:

- **CORRECT — global accuracy barrier removed.** A2 explicitly restates A1.2 per mesh; A3 acknowledges waiting for that mesh’s optional accuracy work. Other meshes’ accuracy completion is no longer a logical prerequisite. Shared job slots can still cause resource delays. `D:406–411,484–490,530–531`.
- **CORRECT — B4 undefined fits rejected.** Nonpositive/nonfinite fitted errors, nonpositive fine-step error difference, and undefined fits become uninterpretable. Neither integer-order outcome is logically excluded. `D:247–257,550–552`.
- **CORRECT — TH06 cannot pass vacuously.** It requires at least five interpretable cases and order 1 on every interpretable case. Merely failing to establish order 2 is insufficient. `D:428–434`.
- **CORRECT — restricted 3D claim survives.** Only Q1/Q2-passed candidates enter the tested-subset front; missing successors produce unverified status. No completeness claim remains. `D:474–482`.

Ranked remaining blockers:

1. **NEEDS-RESTATEMENT — finish the fallback reservation lifecycle.**  
   A3 releases the 2 GPU-h allowance when Q2 labels become known, including when those labels trigger mandatory fallbacks. Specify that it is **atomically replaced** by reservations for actual required fallback jobs and any enlarged/repeated timing panel. Explicitly classify fallbacks by their parent arm’s defer priority. Otherwise different implementations can release the same allowance while protecting different outstanding work. Also define the outcome when a **non-deferrable** submission fails the reservation check after permitted deferrals are exhausted; A3 specifies only the deferrable-job branch. `D:106–110,409–410,526–529`.

2. **NEEDS-RESTATEMENT — TH06’s full ladder lacks \(L_s\).**  
   The “same B4 procedure” uses \(\{L_s,2,1,\tfrac12,\tfrac14\}\), but the \(L_s\) table defines only BE, CN-R and GAL-CN. Register TH06’s larger step, or explicitly omit that contextual point for the control. This does not invalidate its fitted three-point order test, but leaves the newly mandatory full smoke schedule underspecified. `D:93–97,244–262,546–548`.

Read-only audit: no files changed, scratch created, arrays copied, or cluster jobs submitted.