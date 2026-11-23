# Frozen-choice complete-query confirmation

Registered before generating the new cohort's truth trajectories. The November 19 user request authorizes completion of the missing JCP confirmation experiments and plots. This lane studies the existing original-family Navier–Stokes models; it does not resume the separate paused Burgers campaign.

## Frozen scientific question

Do the previously selected learned heads and their previously selected comparators retain accuracy and their complete-query performance on fresh physical cases? No scientific configuration or comparator is selected in this experiment. `configs/fresh16.json` contains the exact inherited configurations, threshold values generated directly from the accepted development result, and comparator identities.

The head settings are the cheaper $M=292,\Delta t=0.02$ and more accurate $M=584,\Delta t=0.01$ variants. Each has its prior POD-head comparator ($M=292$ with the corresponding time step), the same coarse spectral RK4 comparator (mesh size $40$, ten steps), and the same-grid CNAB2 comparator (mesh size $64$, fifty steps). The learned bank, latent dimension, initialization, operator, nonlinear solver and four-start fit remain unchanged. POD is regenerated from the recorded training seed with the original coefficient-probe and sign-parity check; no new training is conducted.

## Cohort and output contract

The registered new seed is `2026111901`, with all sixteen parameter rows in order. The phase-confirmation lane shares this physical cohort. `COHORT_REGISTRATION.json` records the complete parameter tuple hash and nonoverlap checks against the known training and development tuples. Historical sealed cohorts remain unopened.

Every query receives the same original host initial field and viscosity, and returns six full host velocity fields at $t=0,0.04,0.08,0.12,0.16,0.20$ on the original $64^3$ output mesh. Native initialization is recomputed each call. The primary accuracy is maximum error over the five evolved requested fields, normalized by the original initial-field norm. Initial error is reported separately. The stopwatch includes native fitting, transfers, reduced evolution, output recovery and returned solver-diagnostic conversion.

## Qualification and fixed thresholds

References use the inherited step sizes $0.00025$ and $0.000125$. Their discrepancy must be below $10^{-5}$ and below one percent of each method's evolved error; physical input projection must agree to $10^{-12}$. Learned and POD heads additionally receive untimed solver tightening from the inherited residual tolerance $10^{-7}$ and initial-fit tolerance $10^{-9}$ to $10^{-9}$ and $10^{-11}$, retaining the iteration budget. Every nominal and tightened solve must terminate by stationarity. The field difference must be below the smaller of $10^{-6}$ and one percent of the nominal evolved error. These checks qualify the fixed query; their outcome cannot select a replacement query.

Two accuracy questions are reported separately. First, both cohort median and worst error must meet the *fixed development thresholds* stored in the configuration. Second, the unchanged comparator may or may not have no greater median and worst error than the fresh head. The second check does not move the fixed thresholds. A failed accuracy qualification is reported, not repaired by searching new candidates or dropping cases.

## Timing and failure accounting

One GPU allocation measures all subjects with five randomized interleaved repetitions over every subject–case pair. Three repeated sentinels on the first predetermined case run before and after. Each block receives GPU burn-in. Both within-confirmation early/late case-median drift and sentinel drift must remain at most $1.1$. Full repetition arrays, ordering, output hashes, accuracy and solve diagnostics from the same invocation are retained. Failed invocations remain explicit and invalidate complete eligibility. Ratios of cohort medians and the median of paired case ratios are named separately.

This is warm complete-query cost. Cold activation was already measured in the development study and is not repeated here. Absolute times from the development allocation are never pooled with the fresh allocation. No broad candidate search, training or hardware crossover claim is made.

## Finite execution and retention

Initial allocation: thirty minutes, within a lane ceiling of one and a half GPU-hours including failed attempts. Scientific settings and cases cannot change after outcomes. Implementation/retention corrections preserve all earlier attempts and are explicitly documented. All scientific work runs on a cluster GPU in float64 with highest matrix precision and the mandatory GPU-backend preflight. Every attempt has a separate directory. Code and config are committed before staging; results are pulled with checksum verification before removing the owned remote duplicate.

## Plot deliverables

The saved-development figure uses candidate-selection repetitions only for its accuracy–cost frontier. Selected development confirmations are a separate paired-ratio/repetition figure. A new confirmation figure displays fresh error against the fixed thresholds and retains ineligible points visibly. The plots are generated from JSON with source hashes, machine-readable plotted values and reproducible vector/raster outputs.

## Glossary

**Head:** nonlinear coefficient map inside a frozen spatial bank. **POD:** a linear basis obtained from training snapshots; here its head is also frozen. **Comparator:** the previously selected alternative method. **Cohort:** the group of physical input cases. **Confirmation:** evaluating previously fixed choices on fresh cases. **Complete query:** the entire declared input-to-output calculation. **Eligibility:** satisfaction of both median and worst accuracy limits and numerical checks. **Stationarity:** the stopping condition on the residual objective gradient. **Sentinel:** the same predetermined case measured around the timing block to detect clock or load drift. **Paired ratio:** comparator time divided by head time for the same physical case. **Median:** the middle observation; **worst:** the largest observed error. **Reference:** the more resolved numerical calculation used to score error.
