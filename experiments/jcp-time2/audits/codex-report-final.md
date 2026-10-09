**Overall: NOT CLEAN.** The numerical results pass; the report needs presentation and wording fixes before hand-back.

CPU-only audit completed. No files modified, including the lab log. I independently recomputed the wide-job summaries from `result.json`, checked all five audit hashes, reproduced the report and analyses in memory with writes blocked, and inspected all nine PNGs.

1. **CORRECT — numerical provenance.**  
   `report.md` reproduces byte-for-byte; all five analysis JSONs reproduce exactly. All five audit SHA-256 values match their raw results. Experimental measurements are generated from data. The literal statement “every number is generated” needs qualification: configuration values, cohort sizes and thresholds also appear as hard-coded generator text.

2. **CORRECT — wide accuracy and anchor summaries.**  
   All **56 production arms × six ST/S/TX summaries = 336 values** match, as do all anchor medians, maxima and resolution counts. All **8,816 recorded trajectories** are verified.

   At $\Delta t_0$, reference errors below are **PROVISIONAL percentages, worst / median**. Anchor values are median discrepancies to the numerical anchor; all these rows resolve **38/38 cases**.

   | Form | Scheme | ST | S | TX | Anchor median |
   |---|---|---|---|---|---|---|
   | LSPG | BE | 2.837 / 0.907 | 1.608 / 0.060 | 3.003 / 0.968 | 0.978 |
   | LSPG | CN | 2.166 / 0.111 | 3.439 / 1.004 | 2.161 / 0.083 | 0.039 |
   | LSPG | CN-R | 2.145 / 0.137 | 3.336 / 0.850 | 2.149 / 0.196 | 0.190 |
   | LSPG | BDF2 | 2.174 / 0.100 | 3.360 / 0.923 | 2.177 / 0.129 | 0.121 |
   | GAL | BE | 2.814 / 0.907 | 1.735 / 0.058 | 2.980 / 0.968 | 0.978 |
   | GAL | CN | 2.361 / 0.110 | 3.607 / 1.005 | 2.352 / 0.083 | 0.034 |
   | GAL | CN-R | 2.334 / 0.138 | 3.503 / 0.850 | 2.334 / 0.196 | 0.190 |
   | GAL | BDF2 | 2.370 / 0.097 | 3.541 / 0.922 | 2.368 / 0.129 | 0.117 |

3. **CORRECT — wide orders, hypotheses and selections.**  
   Every primary and adjacent triple is valid on all 38 cases.

   | Form | Scheme | Primary median $p$ | Adjacent median $p$ | Both in order-two band | Verdict |
   |---|---|---:|---:|---:|---|
   | LSPG | CN | 1.334237 | 1.845382 | 5/38 | Not established |
   | LSPG | CN-R | 1.964919 | 1.948997 | 33/38 | Order 2 |
   | LSPG | BDF2 | 1.953469 | 2.018634 | 30/38 | Not established |
   | GAL | CN | 2.000662 | 2.004256 | 35/38 | Order 2 |
   | GAL | CN-R | 1.983565 | 1.962573 | 38/38 | Order 2 |
   | GAL | BDF2 | 2.024756 | 2.048382 | 38/38 | Order 2 |

   BE and TH06 establish order one for both forms. **H1 passes through all six CN-family/BDF2-family arms at $\Delta t_0$. H2 passes through LSPG CN-R and BDF2 at $2\Delta t_0$.**

   Both ST selection and TX sensitivity select:
   - Fast-equal-accuracy: **LSPG CN-R, $2\Delta t_0$**, paired ratio **0.554903**.
   - Accurate-equal-cost: **LSPG BDF2, $\Delta t_0$**, paired ratio **0.999621**.

4. **WRONG — 203 Markdown table rows contain unescaped pipes.**  
   Timing subjects and FOM-matching identifiers split into extra columns despite their backticks. Rendering confirms that `main|GAL|BDF2|…` displaces the numerical columns and drops trailing values. See [the timing table](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:127).

   **Fix:** escape every identifier’s `|` as `\|` in both generators, then check rendered column alignment.

5. **WRONG — exact vendor-ratio conversion claim.**  
   The comparator paragraphs say every paired ratio becomes exactly **1.137× / 1.062× / 1.069×** larger against vendor code. Dividing separately aggregated median ratios does **not** produce a measured median paired ratio against another comparator. See [the wide paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:490).

   **Fix:** retain the measured vendor/generic ratio; delete the exact conversion claim or explicitly call it an approximate rescaling.

6. **NEEDS-RESTATEMENT — wide summary lacks A7 context.**  
   The detailed section correctly says **secondary setting, unselected Gauss $128^2$ rule**. The summary and standalone plots omit that qualification and the quadrature-sensitivity context required by A7.

   Also, the summary places “H2 passed” immediately after wide **CN at $2\Delta t_0$**, although that arm **fails** the worst-error condition: **3.481% versus 2.837%**, provisionally.

   **Fix:** label wide secondary/unselected in the summary and plots; identify CN-R/BDF2 as the H2 witnesses. Include or explicitly reference their sensitivity indicators and mark fine-step sensitivity unavailable.

7. **NEEDS-RESTATEMENT — opening status and inventory are incomplete.**  
   [The opening](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:3) describes only 2D results and amendments A1–A9; “What ran” lists only three jobs. The report also contains `fom1k` and `b3d65`, with later applicable amendments through A13.

   **Fix:** list all five jobs and distinguish completed audited measurements from provisional reference accuracy. State that this is development/validation evidence, without final test64 or $4096^2$ replication reported.

8. **NEEDS-RESTATEMENT — glossary is incomplete and partly overgeneralised.**  
   Missing definitions include $R'$, $M$, $L$, $\Delta t_0$, bank/rotation, FOM/ROM, Gauss/Fibonacci/lattice rules, LMM, DST, Newton–BiCGStab, `ntol`, calibration statuses, eligibility, and the pipe-separated identifier fields. Define which triples the “outside” counts describe.

   Existing generic anchor and timing definitions describe only 2D; 3D uses a different lattice, resolved-case median and timing cohort. **Fix:** add explicit 2D/3D definitions, including both base time steps.

9. **CORRECT — principal scientific qualifications.**  
   Reference-error sections are labelled provisional; 3D reference errors are explicitly reference-limited. LSPG CN is consistently “order not established.” Ratios above one are not presented as satisfying the equal-cost selection. Anchor-based time-error qualifications are present. Summary values agree with their underlying tables, allowing for different rounding.

   LaTeX and Mermaid show no evident syntax defect on source inspection; I did not validate them with a full math/Mermaid renderer.

10. **NEEDS-RESTATEMENT — plot presentation.**  
    All nine plots have provisional reference-error labels; the 3D panels also carry reference-limited labels. Series and order labels agree with the analyses, and valid fine-step 3D accuracy points are included.

    **Fix:** add the wide qualifications from finding 6; move the long 2D anchor legends away from curves, separate overlapping cost annotations, and shorten the nearly touching `lat4096_R256` panel titles. Explain hollow markers within each standalone plot as **“fewer than half the cases resolved”**. Label cost-curve connections as step-ladder connections, not an interpolated achievable frontier.

**Accept the wide numerical results. Fix findings 4–8 and the plot qualifications before hand-back.**