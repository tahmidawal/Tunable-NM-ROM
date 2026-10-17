# Watchdog interventions log

Every coordinator action taken on a watchdog report (resume, kill, re-run, escalate), newest last.

| UTC | lane | watchdog reports | finding | action | outcome |
|---|---|---|---|---|---|
| 2026-10-17T23:58Z | (machine) | wd1-20261017T233632Z | 3 orphaned `until [ -s ... ]` waiter loops from an old session (5432ab58, 2026-10-01), 16 days old, targets 0-byte | verified PIDs 57635/92413/112622 via ps (bash until-loops, no campaign link), killed them and their children (single watchdog with direct evidence suffices for an orphaned waiter) | gone; wd1 restarted |
