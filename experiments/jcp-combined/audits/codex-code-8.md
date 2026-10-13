Reviewed `fc04bb9d9` against DESIGN A1–A9. No files modified, project modules imported, or GPU jobs launched. Reproductions used extracted source and synthetic inputs.

All paths below are relative to `experiments/jcp-combined/`.

| Previous item | Disposition | Evidence |
|---|---|---|
| 5 | **RESOLVED** | Components and orthogonal pairs consume trajectory validity; B2 now requires the configured population valid. `budget2d.py:190–224,301–312`; `budget3d.py:168–195,256–262`. |
| 7 | **RESOLVED under A8.6** | Production requires grid reproduction with configuration-derived counts. Freeze binding is enforced. `budget2d.py:70,335–338`; `budget3d.py:57,295–310`; `cluster/stage.py:68–72`. Conditional fallback jobs are not smoke blockers. |
| 13 | **PARTIALLY** | Budget analyzers now derive required inventories from configuration (`budget2d.py:101–111`; `budget3d.py:96–109`). The generic scorer’s budget branch still derives required case IDs from surviving archives (`score.py:327–329`). A9.6 assigns budget certification to the analyzers, so that residual scorer defect alone does not block these smokes. |
| N7 | **RESOLVED for budget analyzers** | Null per-time scores are handled, and negative controls select valid baselines. `budget2d.py:127–151`; `budget3d.py:121–128,275–286`. |
| N8 | **RESOLVED** | Offline B2 compares control against converged baseline and requires complete valid populations. `budget2d.py:299–312`; `budget3d.py:254–262`. |
| N9 | **RESOLVED** | Finalist-only cases are excluded from budget summaries; incomplete comparisons receive an unavailable classification. `budget3d.py:208–229`. |
| N12 | **RESOLVED** | Both null-baseline probes now produce `all_break=False` with an unavailable reason, without subtraction errors. `budget2d.py:136–150`; `budget3d.py:275–285`. |
| N13 | **RESOLVED** | Repeating the one-valid/two-invalid control fixture produces `[0.001, nan, nan]`, `complete=False`, `discriminating=False`. `budget2d.py:301–312`. |
| N14 | **RESOLVED for the reported false-pass paths** | Inventory mismatch now vetoes reproduction; duplicate JSON rows are counted and expected grid counts remain configuration-derived. `budget2d.py:103–111,330–338`; `budget3d.py:97–109,295–310`. A missing-trajectory crash remains below. |
| N15 | **RESOLVED** | Both extracted classification blocks return `unavailable (incomplete comparison)` when competing components are missing. `budget2d.py:255–259`; `budget3d.py:226–229`. |

**N16 — MAJOR: the prescribed 2D double-failure branch produces an archive that necessarily fails its new inventory gate.**

When both convergence pairs fail, [c2bud.py:181](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:181) clears the saved trajectories and removes their rows. It restores Gauss320/Gauss640 and labels convergence unverified, but never reruns `conv_check()`.

The subsequent ordinary run loop at `c2bud.py:214–216` restores `budget_runs(...)`. That plan contains check-rule runs only for selected arms (`c2bud.py:58–59`). The analyzer nevertheless requires check-rule counterparts of **every** converged-rule trajectory type (`budget2d.py:105`).

Using the actual `smk2bud` configuration and extracted planner:

```text
R512: 27 missing required check rows → inventory mismatch 54
fast: 27 missing required check rows → inventory mismatch 54
```

Thus this explicitly supported failure outcome cannot obtain passing G-score certification, even if every retained trajectory reproduces correctly. Preserve the original failed-pair evidence or regenerate its required check trajectories after restoring Gauss320. This is the convergence-reporting branch, distinct from A8.6 finalist fallback jobs.

**N17 — MAJOR: a missing selected 3D trajectory crashes analysis before the inventory failure is reported.**

The analyzer correctly records an inventory mismatch at `budget3d.py:109`. However, an absent selected deployed trajectory makes `T['a']` unavailable and leaves `per` empty (`budget3d.py:165`). Finalist reduction then executes `max([])` at [budget3d.py:220](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:220).

Extracting and executing the current finalist block with that missing-trajectory state gives:

```text
ValueError: max() iterable argument is empty
```

The analyzer exits before writing its gate results. Handle empty finalist populations explicitly as unavailable/unqualified and retain the failed inventory result. This does not establish that an intact smoke archive will encounter the failure.

No unconditional startup crash was established for the supplied smoke configurations. The remaining supported-path failure and missing-evidence crash warrant correction before running their offline acceptance analyses.

**NOT READY FOR BUDGET SMOKES.**