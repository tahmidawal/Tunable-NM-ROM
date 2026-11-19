# Burgers tree embeddings: common fixed-tree numerical qualification

Pilot qualification is incomplete. Outstanding checks: b2space, b2quad, b2tests, b2time, b2tight.

The direct tree has1376 adjustable coefficients and the stable global POD-plus-Fourier control has960. The common baseline is L8192, Crank–Nicolson dt0.00125, Q512 and22016 weak tests. The spatial check uses L16384, integration Q1024, tests44032, time step0.000625, and the precision check tightens all registered initial/inner/outer tolerances100 times. Each changes only its named numerical control. This new baseline is not a one-factor comparison with the older global768 pilots.

All solves use the shared initial-field callable and the original fine-grid sign-upwind weak formulation. References enter only subsequent scoring. Physical errors use the current-time reference norm; the per-case empirical reference allowance is included in each accuracy decision. Every run returns an actual4096-by4096 final device field. The table combines the largest of five evolved physical errors and the actual-grid final error, and separately audits the complete returned stopping rule. Reported changes include the initial fit as well as all evolved output times.

| Run | Case | Method | Worst physical error | Steps failing complete convergence | Largest field change | Run gate passes |
|---|---|---|---:|---:|---:|---|
| b2base | dev000 | tree_direct24 | 0.281178% | 0/200 | baseline | True |
| b2base | dev000 | global_stable960 | 0.202566% | 0/200 | baseline | True |
| b2base | dev003 | tree_direct24 | 0.021249% | 0/200 | baseline | True |
| b2base | dev003 | global_stable960 | 0.020537% | 0/200 | baseline | True |
| b2base | align002 | tree_direct24 | 0.312845% | 0/200 | baseline | True |
| b2base | align002 | global_stable960 | 0.360317% | 0/200 | baseline | True |



A passing small pilot is not the all44-development-trajectory gate. No deployable adaptation, method freeze, native NM-ROM confirmation, paired runtime win or final-test result is claimed. The nonlinear child16 checkpoint remains stopped after two unsuccessful online solver repairs; its representation result is retained. The original16.5% failure and the improved representation plots remain in [the representation report](2026-11-18-tree-embedding-representation.md).

Pilot clocks charge joint comparison setup to each method and cannot serve as method-specific speed ratios. Initialization, assembly, solving, rejected trials, fallback and requested output remain mandatory components of later complete-query timing. The equivalent packed field implementation has only CPU fixture evidence and is not used in this batch.

Generated 2026-11-19T02:08:40.067005+00:00. Elapsed allocated time is23248 GPU-seconds (6.457778h). The reservation guard currently charges24748 seconds, including full reservations for active jobs. Active owned jobs: b2space. B2 is capped at5400 additional GPU-seconds from22744, for a cumulative ceiling28144 seconds. Source/configurations, scheduler records and verified retention manifests are in the owned lane. The old campaign remains paused; final-test fields remain unopened.
