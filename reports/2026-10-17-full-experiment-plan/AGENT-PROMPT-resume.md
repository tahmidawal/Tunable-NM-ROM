# Resume prompt: Burgers 2D/3D campaign (after the 2026-10-19 usage-limit stop)

Paste everything below the line into a new Claude Code session started in `/home/tahmid/Dev/Tunable-NM-ROM-Claude`.

---

You are resuming as **coordinator** of the viscous Burgers 2D/3D JCP campaign. The previous coordinator (and its lane subagents) stopped on **2026-10-19 at about 05:52 UTC** when the account hit its weekly usage limit. No cluster jobs are queued or running. Nothing was lost, but in-flight lanes stopped mid-step.

1. **Read your instructions:** `reports/2026-10-17-full-experiment-plan/AGENT-PROMPT-burgers.md` (everything below its `---` line). Those rules still apply in full: fully autonomous, Codex gates on every step, watchdogs, defaults in its section 7, final compilation in its section 8.
2. **Read the current state:**
   - `LAB-LOG.md`: "Where things stand" and the entries from 2026-10-17 on;
   - `reports/2026-10-17-full-experiment-plan/campaign/`: `LANES.json`, the briefs, `INTERVENTIONS.md`, the latest watchdog reports;
   - every lane's `HEARTBEAT.json`, last commits, `audits/` and ledger.
3. **Lane status at the stop:**
   - **Done:** jcp-speed3d, jcp-p1-scaling, jcp-qualify, jcp-fom2d, jcp-heads, jcp-hr-baselines. Do not reopen them.
   - **jcp-dims** (`worktrees/2026-10-18-jcp-dims`): job audit `codex-job-t3s-3.md` said SUBMIT; t3s not yet submitted; analyze2d/analyze3d were running. Resume from there.
   - **jcp-budget** (`worktrees/2026-10-18-jcp-budget`): analyses plus the last T=1 reference job, then the results re-audit and the report. First check whether any process from that lane is still running locally (`ps`), and do not start a duplicate.
   - **jcp-robust** and **jcp-regimes** (`worktrees/2026-10-19-jcp-*`): just launched in design mode (setup commit only). Restart their design step.
4. **Relaunch** one background lane subagent per unfinished lane, each with a brief telling it exactly where it stopped. Restart the 2–3 Codex watchdog loops.
5. **Before submitting anything,** check the cluster (`squeue`, `sacct` for the lanes' jobs since 2026-10-18) and collect any finished outputs. Check `df -h /home` (about 90 GB free on 10-19); keep big arrays on the cluster and follow the raw-archive offload policy (campaign brief 16).
6. Record the stop and the resume in the lab log. Then carry on through priority 4 and the final compilation.
