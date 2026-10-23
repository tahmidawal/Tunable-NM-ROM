The recorded arithmetic and stop outcomes pass. **Three reporting fixes remain:** missing test-search qualifications, incorrect export provenance, and incomplete/mislabelled plots. No files were changed.

Paths below are relative to `experiments/imp-box/`.

1. **Restore search qualifications beside test ratios.** `report.md:357` and `report.md:360` omit the incomplete-FOM-search labels attached to their validation counterparts. For example, W40_A3 TRANS prints **3.32→3.30** and **3.27→3.24** without that qualification. Across the complete test results, **9 matched pairs** have incomplete FOM searches. `EXPORT.json:6210` also exports test pairs without FOM-search or ladder-minimum labels. Propagate these labels into both outputs, as required by `DESIGN.md:537`. Numerical matching on test does not complete the competitor search.

2. **Correct eight export source paths.** Every test cell’s `source` points to W40_A3’s test JSON; eight are wrong (`EXPORT.json:6362`, `6515`, `6596`, `6677`, `6758`, `6797`, `6968`, `7151`). The cause is the stale `tf` variable used at `make_report.py:403`. The exported numbers themselves agree with their proper cell files.

3. **Repair the plots.** Both PNGs omit all completed A3 results because `plots.py:25` reads only the original directories, missing `W28_A3t` and `W40_A3t`. The legend also changes meaning between panels: automatic colours depend on which arms are present, while only the first panel supplies the legend (`plots.py:35`, `plots.py:46`, `plots.py:56`). For example, B-LSPG is green in the first panel but red elsewhere. Use the retained-cell mapping and fixed arm colours; distinguish collected A3 from deferred A4 and preserve search qualifications.

**Test-once integrity passes within the recorded evidence.**

- All nine manifest hashes reproduce `runs/frozen/INDEX.json`. Committed-copy, collected-job and original validation manifests agree. All **164 frozen pairs**, including configuration keys and \(S_{\rm val}\), agree with validation `final.json`.
- Test outputs contain exactly the frozen finalist keys and pair keys. The code evaluates frozen configurations and never selects replacements (`teststage.py:177`, `teststage.py:203`).
- Artifact verification precedes descriptor/source checking and test-reference generation (`teststage.py:24`, `teststage.py:133`, `teststage.py:158`). The four current source hashes match the A16 equivalence destinations.
- All nine collected controls have passing `MF_LEAK`, `MF_FROZEN`—including the positive control—and test-matching fixtures.
- All nine batch scripts contain the GPU refusal check at `run.sbatch:23`. Their corresponding Slurm logs record `jax_backend=gpu` at line 3, successful exclusive `.test-opened` creation at line 8, successful archive verification at line 10, and successful test execution at line 12. Environment records show f64 and `highest`.
- Matching reproduces A3.7’s feasibility/qualification rule plus timing validity; A18 releases this existing procedure. I recomputed **532 finalist error-statistic and feasibility records**, **142 matched-pair timing ratios from 5,680 triplets**, and their doubled-query predicates.

The provenance evidence is partly indirect: `check_inputs` does not persist a separate success receipt or list of equivalence exceptions used. Successful execution through its mandatory checks supports acceptance; the collected JSONs alone are not an independent replay of the staged source/artifact verification. Likewise, opening once is supported by the successful exclusive-directory operations, not a separately collected sentinel history.

**All test counts and all 53 printed transitions reproduce.** Every cell has **32/32 eligible** cases:

| Cell | Frozen pairs | Matched | Recomputed printed \(S_{\rm val}\to S_{\rm test}\), in report order |
|---|---:|---:|---|
| W08_A1 | 24 | 24 | 1.06→1.02; 1.54→1.42; 1.14→1.05; 1.14→1.10; 1.14→1.06; 1.14→1.07 |
| W08_A2 | 24 | 24 | 1.46→1.49; 2.14→2.05; 1.60→1.59; 1.46→1.48; 1.19→1.21; 1.60→1.70 |
| W16_A1 | 12 | 12 | 1.03→0.995; 1.34→1.29; 1.12→1.05; 1.04→0.993; 1.12→1.06; 1.12→1.07 |
| W28_A1 | 12 | 12 | 0.927→0.977; 1.18→1.12; 1.02→1.01; 0.927→0.978; 1.09→1.09; 1.04→0.960 |
| W28_A2 | 12 | 12 | 1.29→1.24; 1.49→1.41; 1.51→1.43; 1.29→1.25; 1.14→1.20; 1.51→1.42 |
| W28_A3 | 5 | 5 | 0.769→0.790; 1.62→1.61; 0.531→0.541; 1.42→1.46; 1.62→1.61 |
| W40_A1 | 27 | 12 | 0.927→0.933; 0.801→0.821; 1.09→1.10; 0.850→0.842; 0.764→0.784; 0.989→1.00 |
| W40_A2 | 29 | 27 | 1.18→1.13; 1.13→1.00; 1.07→1.02; 1.04→0.978; 1.13→1.07; 1.15→1.13 |
| W40_A3 | 19 | 14 | 0.540→0.541; 0.421→0.422; 1.27→1.26; 0.886→0.885; 3.32→3.30; 3.27→3.24 |
| **Total** | **164** | **142** | **No numerical mismatches** |

These reproduce `report.md:352–360`. The **22 unmatched pairs** are numerical-feasibility/qualification failures; none is a timing-only rejection. `report.md:348` correctly calls these validation-selected pairs, not test selection evidence.

**The ledger includes all nine jobs.** Recomputed charges total **1.9041667 GPU-hours** for test jobs and **48.9677778 GPU-hours** overall, agreeing with `runs/GPU-LEDGER.json:479`. That leaves **38.0322222** below the 87-hour cap and **23.0958222** below the certificate’s exact **72.0636** projection (`runs/certificate-tests.json:16`). The report’s rounded **49/87** is correct.

**Narrow-cell results agree; stop outcomes are unchanged.**

All **24 W08_A2** and **12 W16_A1** validation ratios agree between their JSONs, export and crossing points. Their per-cell tables agree (`report.md:184`, `report.md:225`), and their plotted positions are consistent with those values, subject to the legend defect above.

Examples: W08_A2’s 1%-worst ratios are **2.14050476 PODG**, **2.00934477 ECSW**, and **2.06872799 B-E4**. Its bank table correctly reports five below-target entries and one unaffordable entry: floors **0.4623806%**, **0.0219327%**, and **1.6359954%** at the applicable ranks (`report.md:236`). W16_A1 PODG’s 1%-median ratio is **1.03491868**, giving the reported crossing between widths **0.16 and 0.28**.

Neither narrow cell belongs in the registered stop predicates: speed uses W28/W40, bank-stop uses W40 (`stoprule.py:22`, `stoprule.py:140`). Recomputing every input entry gives:

- **SPEED-STOP: NOT FIRED** — 126 entries, 95 matches, 80 at least 0.5, maximum **2.1437823699078313**, 27 missing/FOM-censored.
- **BANK-STOP: NOT FIRED** — 3 above, 4 below, 8 unaffordable, 3 missing.

Both exactly match `codex-results-2`.

The previous audit’s three requested report disclosures are present: retained/deferred scope (`report.md:17`), A16 timing evidence (`report.md:40`), and incomplete-search labels in bank/crossing tables (`report.md:124`, `report.md:299`). Conditional reference errors remain labelled estimates, and deferred cells are not presented as measured results. The new test table nevertheless reintroduces the search-label omission described in fix 1.

VERDICT: RESTATE (3 items) — restore test-search labels; correct export source paths; repair plot coverage and arm colours.