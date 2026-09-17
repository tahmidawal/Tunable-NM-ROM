# Self-audit of the controls section (`ctrl01`, job 3783831) — 2026-09-17

Substitute for the Codex report audit (quota exhausted until 2026-09-19 11:33; coordinator
notice in LANE-PROTOCOL). Written by the lane agent. Each claim in the report section
"Controls: precision and seed" is listed with the JSON field it rests on and the check that
was run. Every check below was executed in this session; none is asserted from memory.

| # | Claim in the report | Rests on | Check run | Result |
| --- | --- | --- | --- | --- |
| 1 | Job completed on a GPU, all seven worker tasks exit 0, `ALL-DONE` | `runs/ctrl01/archive/ctrl01/logs/3783831.out`; `sacct -j 3783831` | read the log end to end; `sacct` State/ExitCode/Elapsed/NodeList | COMPLETED 0:0, 02:32:52, pax105; `jax_backend=gpu`, `torch … cuda True`, `data_verified=395`, seven task lines each `"exit_code": 0`, `WORKER FINISHED`, `ALL-DONE` |
| 2 | Archive is what the cluster wrote | `OUTPUTS.sha256`, `MANIFEST.sha256`, `ARCHIVE.sha256` | `cluster/collect.py ctrl01`: remote `sha256sum -c` on both manifests, tarball SHA256 compared after `scp`, every manifest entry re-hashed after extraction | pass; only `last.pt` / cache entries absent (excluded by design) |
| 3 | Every validation and cohort error cell | `audit.json: arms.<arm>.fixed_initial`, `cohort.models.<arm>.fixed_initial` | `audit.py ctrl01` recomputes each per-case, per-time error from `out/*/validation/*.prediction.npz` and the cohort fields with NumPy only, asserts equality with the driver's numbers (1e-5 float32 / 1e-11 float64 pre-registered tolerance), bitwise initial state, zero boundary, finite float64 | `passed: true` |
| 4 | Same split and cohort as the screens and the FNO job | `audit.json: train_index_sha256`, `validation_index_sha256`, `identical_split_to_fno_job`, `cohort.cohort_index_sha256` | compared to `unet01`/`tsol01` audits and the FNO literals (cross-checked against the archived `DATA.sha256`) | `5333584b…` / `468b9e70…`; cohort `8b8a2ee1…` equal in all three lane jobs |
| 5 | Each control differs from its twin in exactly one variable | `audit.json: arms.<arm>.{seed,parameter_dtype,real_parameter_count,wall_budget_seconds,config}`; `spec.arms[*].override` | printed and compared with the twin's fields in `runs/unet01/audit.json` / `runs/tsol01/audit.json` | `ctrl-medium-f64`: seed 20260914, float64, 7 763 461 params, 3000 s (twin: float32, otherwise equal); `ctrl-medium-seed2`: seed 20260915, float32, 7 763 461 (twin: 20260914); `ctrl-tsol-small-f64`: float64, 3 108 240 (twin: float32) |
| 6 | Twin identity and "In" column | `spec.arms[*].twin`, `.role` | `twin_of()` reads the spec; the "In" column is derived from `role`/`parameter_dtype` | matches §A4's arm list |
| 7 | Deltas and the three spreads | `arms.<arm>.fixed_initial.{maximum,median,mean}` of control and screen audits | recomputed in a separate script (no generator import) for family-4, family-3-capacity and pooled spreads | identical to the table: e.g. tsol worst +4.5996 vs 3.1385 (outside), unet-f64 median +0.1935 vs 0.3157 (inside) / 0.0794 (outside strict) |
| 8 | The flag: `tsol-small` validation worst is precision-sensitive; both U-Net controls inside | rule = §A4 text ("worst or median"), spread = twin's family screen (4 arms) | read the generator: `outside_a4` is set only from the 4-arm family column on worst/median | one flag, `tsol-small` (network dtype) |
| 9 | The rule's wiring was changed after the result | `git log -S'def controls_section'` (`b7bfdb3c`, 09:35:50) vs `runs/ctrl01/run.sbatch` commit (09:31:49) vs job end (12:08) | compared timestamps | the pre-result draft pooled families and tested mean/worst — a mismatch with §A4's text; replaced by the literal rule; all three readings shown; recorded in DESIGN §A9 |
| 10 | The positive claim survives: both U-Net twins below the ROM's 1.8671 % | `cohort.models.<arm>.fixed_initial.maximum` vs `checks/refinement02-diagnosis-audit.json: summary.rom.worst_fixed_initial_error` (SHA-pinned in the generator) | compared in the separate script and by the generator | 1.6721 % and 1.5479 % < 1.8671 %; margins 0.1950 / 0.3192 pp |
| 11 | `tsol-refine` is below the ROM and has no twin | `runs/tsol01/audit.json: cohort.models.tsol-refine` (1.5224 %); `spec.arms[*].twin` | generated: family arms with cohort worst < ROM not in the twinned set | `tsol-refine` only |
| 12 | Epoch ratios 0.25× / 0.42× | `arms.<arm>.epochs_completed` | 500/1963, 682/1628 | 0.25, 0.42 |
| 13 | `summary.json` gained exactly the control rows and stayed unique | `summary.json: rows[*].key` | counted rows before/after; `(key, cohort, metric)` uniqueness asserted in `main()` | 426 → 468 rows (42 for job 3783831: 3 arms × (4 validation stats + 6 per-time + 3 cohort + 1 timing)); unique |
| 14 | No number typed into the section | `generate_report.py` | read every f-string in `controls_section`; the only literals are the column headings and the words inside/outside | confirmed |
| 15 | Remote state | `ssh tufts-login find <namespace> -mindepth 1 \| wc -l` after `--cleanup` and `rm -rf poisson-data01` | run in-session | 0 entries; `squeue -u tawal01` shows no lane job |

**Findings against my own report.** (a) The first generated wording said "the claim survives
this control" for `ctrl-tsol-small-f64`, whose twin was already above the ROM — no claim rests
on it; fixed before commit so that line now says no positive claim is defended and names
`tsol-refine` as un-twinned. (b) The "One seed" fairness bullet still said the controls were
"a separate job"; now conditional. (c) Timing of the controls is in `audit.json` (`ctrl-medium-f64`
8.639 ms, `ctrl-medium-seed2` 5.822 ms, `ctrl-tsol-small-f64` 8.368 ms pooled device medians)
and in `summary.json`, but not rendered in the controls table — the per-job timing tables
already forbid cross-job comparison and the controls' timings are a third job; left out of the
prose deliberately. (d) The three-capacity spread on the U-Net median (0.0794 pp) is smaller
than the seed-to-seed delta (0.1529 pp), which says the screen's capacities are inside seed
noise on the median — the report's existing fairness bullet already says family differences
are not separated from seed variation; this is the number behind it.

**Not checked here.** Whether a second seed of `tsol-refine` or `unet-refine` (the headline
arms) behaves like the capacity twins; whether float64 at equal *epochs* rather than equal wall
closes the gap; any other PDE or mesh.
