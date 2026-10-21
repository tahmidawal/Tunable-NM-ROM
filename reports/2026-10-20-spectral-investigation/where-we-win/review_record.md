Three adversarial read-only Codex rounds were run (`codex-review-1.md`, `-2.md`, `-3.md`). Each finding's disposition is recorded as one of three outcomes: **fixed** (the error was corrected), **withdrawn** (the claim was removed or weakened) or **open** (it needs an experiment that has not been run).

**Round 1 (verdict REJECT, 14 findings).**
1. The 3D anchor was unqualified. **Fixed:** the model now uses the qualified C_old, and W_C_W768 is labelled unqualified.
2. Global resolution was treated as implying a win over adaptive FOMs. **Withdrawn:** this is now a uniform-grid scenario only, and locally refined FOMs were added to the competitor sets and kill criteria.
3. The ROM extrapolation was uncalibrated. **Withdrawn as a prediction:** it is now a sensitivity table. **Open:** iteration, conditioning and time-step sensitivity need controlled 3D runs.
4. The wrong panel was used and the speed-up statistic changed. **Fixed:** times are keyed per configuration, and the registered paired ratios are used.
5. The slowdown thresholds were read as evidence. **Fixed:** $S_{\rm new}=S_{\rm box}f_{\rm FOM}/f_{\rm ROM}$ is stated, and the thresholds are now called necessary conditions.
6. FD CG counts were used for spectral trajectories. **Withdrawn:** all columns are labelled synthetic. **Open:** this needs a FOM-only trajectory benchmark, now the first step of #3.
7. The per-iteration cost was invented. **Fixed:** it is now a hypothetical sensitivity.
8. The break-even numbers were inconsistent. **Fixed:** the hole slowdown and per-mesh paired latencies are now included, and "no break-even" is reported when nothing is saved. **Withdrawn:** "lower bound" became "illustrative partial-cost crossover", and the flagship-2 estimate was removed.
9. Penalisation was used to dismiss FFT-based competitors. **Withdrawn:** the 1 % and 0.2 % resolutions are now labelled extrapolations, and active-penalty, immersed-interface and Fourier-continuation methods are competitors in #1 and #2.
10. The smooth control was invalid and convergence was not recorded. **Fixed:** the control was removed, and convergence is now persisted.
11. The L-shape comparison was not established. **Withdrawn:** it is restricted to the manufactured example. **Open:** this needs a coarse-FOM ladder on the lane's own cases.
12. Varying topology was claimed as exclusive. **Withdrawn:** it is now "candidate, not exclusive", and per-query setup is charged to both sides.
13. Workload and accuracy contracts were missing. **Fixed:** they are now attached. **Open:** a common output contract for 3D needs runs.
14. Generation and rules. **Fixed:** regeneration, status and glossary.

**Round 2 (verdict REJECT, 8 findings).**
1. Flagship 2 still predicted a speed-up, and the multi-mesh edge was overstated. **Withdrawn:** the prediction is now "unknown", and F14b is an untested amortisation hypothesis.
2. The 3D scenario mixed output contracts, and its sensitivity range was incomplete. **Withdrawn:** it is now labelled an incomplete algebraic scenario, with the missing factors listed. **Open:** a common-contract measurement.
3. The warm-start column was over-interpreted. **Withdrawn:** all columns are synthetic, $S>1$ is now distinguished from $S\ge2$, and the stage-count caveat was added.
4. Competitors were missing from the decision rules. **Fixed:** the kill criteria now cover every tested competitor. KIOPS/Leja exponential integrators and reused factorisations were added to #3, and Fourier methods to #2. Claims are limited to the tested set.
5. Flagship 2's acceptance rule did not cover all outputs. **Fixed:** field, gradient and flux each get a tolerance, the statistic is the worst case over a held-out geometry cohort, and gradient and flux floors are checked separately.
6. The break-even timing convention was inconsistent. **Fixed:** per-mesh paired latencies are used, and the ROM is described as "approximately mesh-independent".
7. The KL contrast was undefined. **Fixed:** the field is normalised per draw, and the realised extrema are recorded.
8. Rules. **Fixed:** literature figures are registered in `VALUES.json` as quotations, the glossary is completed and this record is itemised. **Not changed:** the filename `report.md` was set by the task brief.

**Round 3 (verdict ACCEPT-WITH-FIXES, as an exploratory research-direction report, not a validated forecast; 5 findings).**
1. The summary still promoted illustrations to evidence. **Withdrawn:** the 3D numbers are now "an arithmetic illustration with mismatched outputs, not a forecast", the trajectory cost for #3 is "unknown", and the ranking is a subjective research priority.
2. The gates were inconsistent. **Fixed:** #3 gates on the fastest accuracy-admissible tested FOM, and its union includes (a') and (a''). #1 separates FOM superiority (gate), POD superiority (reported result) and the method objective.
3. The contrast-1 row was mislabelled. **Fixed:** it now reads 1.0001, nearly constant.
4. The Galerkin definition was wrong. **Fixed:** test space equals trial space.
5. Rules. **Fixed:** more notation is in LaTeX, and the filename exception is stated in the status paragraph.

No fourth round was run (the brief allows three).
