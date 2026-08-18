# Cancelling jobs — explicit numeric IDs ONLY

**Incident, 2026-08-17 ~14:5x.** A `scancel` issued from this cell with a *name filter*
(`scancel --name=a,b,c -u tawal01`) did not filter as intended and killed **every job on the
account**, including nine jobs belonging to another agent working in a different worktree —
five of which were 10–16 minutes into real compute. Nothing was corrupted and both fleets
resubmitted, but the lesson is cheap to learn now and expensive later: the Burgers panels are
multi-hour, and a blanket cancel at hour three costs a working day.

## The rule

- **NEVER** run `scancel -u tawal01`, `scancel --name=<anything>`, or any `scancel` whose
  argument is not an explicit list of numeric job IDs.
- The account `tawal01` is **shared with other agents**. Their jobs look nothing like
  `wsp_*` / `wsb_*` and die just the same.
- Before any cancel:

  ```bash
  ssh tufts-login "squeue -u tawal01 -o '%.10i %.20j'"
  ```

  Read the list. Confirm **every** ID you are about to kill is one of yours (`wsp_*` /
  `wsb_*`, submitted by this cell). Then:

  ```bash
  ssh tufts-login "scancel 2511166 2511169 2511171"    # explicit IDs, nothing else
  ssh tufts-login "squeue -u tawal01 -o '%.10i %.20j'" # re-check afterwards
  ```

- If a whole round genuinely has to be cleared, build the ID list from `squeue` filtered by
  **job name**, print it, eyeball it, and pass those IDs literally — do not pipe an unchecked
  list into `scancel`, and do not let `scancel` do the filtering.
