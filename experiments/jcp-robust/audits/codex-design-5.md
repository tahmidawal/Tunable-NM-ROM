Read-only audit completed. No files changed, scratch created, jobs submitted, or cluster commands run.

**D** = [experiments/jcp-robust/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/DESIGN.md).

1. **CORRECT — IMPLEMENTATION-NOTE — Prior BLOCKING item 19/34: RESOLVED.**  
   A4.1 explicitly withdraws the single-job fallback and requires both jobs of each compared hardware type (`D:722–727`). This is compatible with the every-job intersections in A3.8 (`D:680–683`): an absent required job yields unavailable pooled conclusions, rather than a verdict from the surviving job. The complete-data q4 constructors remain defined (`D:685–690`).  
   **Fix:** None to the design. Verify that missing-job handling executes before pooling.

2. **CORRECT — IMPLEMENTATION-NOTE — Retry requirements remain subordinate to the hard cap.**  
   A4.1 requests a replacement attempt charged to the ledger (`D:725`). It cannot override the binding 75 GPU-hour cap and reservation requirement (`briefs/jcp-robust.md:9`; `COMMON-BRIEF.md:87–92`). The design already defines the terminal outcome when protected work cannot finish: incomplete, no verdict, shortfall reported (`D:327–331,709–713`). Thus exhausted budget does not require an unattainable successful replacement.  
   **Fix:** Implement retries through the existing reservation check; if funding is unavailable, report incomplete with surviving results labelled “per-job only.” Do not interpret “before reporting” as an unlimited retry obligation.

3. **CORRECT — IMPLEMENTATION-NOTE — A4.2 preserves numerical-failure and target-failure distinctions.**  
   Separate validity flags retain simultaneous failures; the displayed-status precedence is now explicit (`D:730–732`). A failed target makes target-dependent evidence unavailable without converting valid production output into a failed draw (`D:733–735`). This agrees with A3.2–A3.3’s distinction between production failure, missing evidence and uncertified targets (`D:628–644`). The target-free H-E1c margin is defined.  
   **Fix:** Derive predicates from the separate flags, not solely from the displayed status. Preserve valid own-family evidence and label target certification as subset-based.

4. **CORRECT — IMPLEMENTATION-NOTE — Population and emergency-front clarifications introduce no contradiction.**  
   A4.2 restores E2’s registered sixteen-case 4096² population while retaining E1’s thirty-eight cases (`D:737`; `D:189,513–515`). Its emergency incumbent rule withholds selected-rule front agreement when the incumbent receives only fixed-configuration evaluation (`D:741–742`), avoiding comparison of different candidate universes (`D:439–447,711–713`).  
   **Fix:** Encode these distinct populations and unavailable comparison states explicitly.

5. **CORRECT — IMPLEMENTATION-NOTE — Resource feasibility remains a code/job-audit obligation.**  
   A4.2 requires a generated execution inventory and smoke-derived reservation certificate, and explicitly denies certified status to the projected contingency (`D:739–740`). It also preserves archive-key mapping and finite, nonempty control admission (`D:738,743–744`). These address round 4’s implementation concerns without changing scientific outcomes.  
   **Fix:** Include all tight/tighter executions, target checks, replay gates and reuse in the inventory; enforce the certificate before submission.

6. **CORRECT — IMPLEMENTATION-NOTE — No remaining BLOCKING design issue found.**  
   Reading later amendments as governing, the registered questions retain defined outcomes: E1 qualification/sensitivity and tie handling (`D:112–125,531–546,628–649`); E2 definitive agreement, disagreement and unavailable states (`D:570–576,651–655`); E5 pooled interval decisions, job disagreement and missing-job outcomes (`D:584–593,680–691,722–727`). A4 introduces no new undefined or unattainable scientific outcome.  
   **Fix:** None at design level; carry the implementation requirements above into the code audit.

**Overall verdict: READY (no BLOCKING item).**