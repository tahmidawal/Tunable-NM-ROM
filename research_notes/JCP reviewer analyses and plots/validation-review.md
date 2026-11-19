# Physical-validation and performance reviewer notes

Condensed faithfully from `/root/reviewer_validation`, November 19, 2026. This was an advisory review of saved evidence and primary literature, not a real journal referee decision. No scientific runs, writes or cohort access were performed by the reviewer.

## Main recommendation

Promote phase/shape and derivative fidelity into the main evidence. Add clear parameter coverage, case variation and failure accounting. Preserve the distinction between candidate-selection measurements and selected confirmation comparisons rather than plotting a falsely uniform timing frontier.

## Recommended analyses

**Complete-query frontier:** essential. Milliseconds versus cohort median and worst maximum evolved error in separate panels; initial error separately. Use selection repetitions for candidate frontiers and confirmation repetitions for selected paired ratios. Same hardware allocation, input/output and declared cohort eligibility are required. A mixed paired comparison qualifies a favorable aggregate. Source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04-selection.json`, with confirmation in `cost-p04.json`.

**Phase/shape and derivative fidelity:** essential physical panel. The actual periodic physical centroid and aligned/ordinary NS errors are saved; alignment must not replace unaligned primary error. Burgers gradient and advection errors diagnose spatial fidelity beyond field error. The broad core docstring is misleading: derivative/operator metrics use time-varying corresponding reference norms, while field error uses initial norm. Inspect denominators. The existing NS drift plot includes a separate manifold-distance panel requiring its fitting caveat. Use shared scales and a stated representative/difficult-case rule for spatial fields.

**Coverage and generalization:** essential for parametric claims. Plot physical parameter locations labeled training/development/untouched confirmation and error versus meaningful parameters. Split whole trajectories. Interpolation, extrapolation and mesh transfer are distinct. Historical held-out results exist, but the newest controlled panels are development data; final frozen-choice confirmation remains additional work.

**Case distributions and failures:** essential. Use empirical error distributions or paired scatter and a configuration/case success matrix. Separate setup failures, invalid references, solver failures and finite inaccurate outputs. Repeated invocations are not independent cases, and failed development repairs are not method failure-rate estimates. Casewise errors and reasons already exist in the head/cost records.

**Energy and qualified horizon:** supporting, stronger if stability is claimed. The checked NS mechanism is unforced viscous decay, not constant-energy evolution. Plot reference/ROM energy and discrepancy using initial reference energy. Saved divergence is the Fourier-norm ratio $\|\nabla\!\cdot u\|/\|\nabla u\|$. Energy can miss translation, so prioritize phase/shape for drift. Historical longer-horizon evidence has reference/rule qualifications. Source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/checks/b7b8-1024.json`.

**Initialization:** supporting. Plot initial error against evolved maximum for common/native arms, with fitting convergence/start information. Saved initial-fit and trajectory records exist. Full-query totals include fitting but do not imply a replicated fitting-cost breakdown. Never estimate it by subtracting separate jobs.

**Training seeds/data and offline cost:** claim-dependent. Learning curves should use training trajectories and independently trained models, not timing repetitions or test cases. Historical seed data exist, but a unified current-method learning curve was not verified. Keep measured offline costs distinct from projected lifetime totals and avoid break-even when online cost is already greater. Seed source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e2-verdicts-2d.json`.

## Verified primary literature

[Kim et al. author text](https://arxiv.org/html/2009.11990), [JCP publisher](https://www.sciencedirect.com/science/article/pii/S0021999121007361): parameter-space error, data and dimension studies. [Barnett–Farhat author text](https://arxiv.org/html/2204.02462), [JCP publisher](https://www.sciencedirect.com/science/article/pii/S0021999122004107): local/integral physical quantities and offline/online resources. [Romor author text](https://arxiv.org/html/2308.03396), [JCP publisher](https://www.sciencedirect.com/science/article/pii/S0021999125000129): parameter-resolved errors and overhead. [SNF-ROM author text](https://arxiv.org/html/2405.14890v2), [JCP publisher](https://www.sciencedirect.com/science/article/pii/S0021999125002402): interpolation/extrapolation, temporal error and sampling/time/resource interaction. Publication metadata was verified; detailed figure references refer to accessible author versions. Proposed failure matrices and paired timing panels are reviewer adaptations.

## Glossary

A complete query covers the declared physical input through requested output. Selection chooses a candidate; confirmation measures the frozen selection again. Case variability concerns physical inputs, seed variability concerns trained models, and repetitions concern measurement variability. A qualification failure is distinct from a finite inaccurate prediction. Generalization claims need an explicit parameter/time domain and fixed data split.
