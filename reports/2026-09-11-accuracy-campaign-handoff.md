# Handoff: accuracy campaign and the next NMROM session

Prepared September 13 from the finalized, independently audited September 11 development results. This is a handoff snapshot; the canonical lab log remains authoritative for subsequent changes.

## Read first

[Canonical LAB-LOG.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md) must be read first, followed by [repository operating rules](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/AGENTS.md). Read their current versions even when resuming from this document.

Repository: `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude`. The user is developing the current separable nonlinear-manifold ROM for accuracy and speed across Poisson, heat, Burgers and reflective waves, including transfer across output resolutions and fixed-weight online tuning. The older ViT + CP model is outside this comparison. Absorbing waves are excluded.

## What is complete, and what is not

The bounded experiment round, raw-field audits, timing aggregation and source-generated report are complete. The report is committed on main at `fa0f0c21b7da0098c12d2a95cd2efa9aa332743c`. All experiment source, selected checkpoints, failed alternatives and restorable result archives are retained in the separate PDE worktrees. Cluster run directories were checksum-collected and removed; the queue was empty at campaign closure. Recheck the queue before new work.

No consolidated worktree or merge has been created. The latest user request was to write this handoff; it did not approve the previously proposed consolidation. Existing PDF tables and collaborator slides still contain the previous round, while the campaign Markdown/JSON contain the updated results. Final paper cohorts remain sealed. These development results do not establish final publication accuracy or broad PDE-family generalization.

The canonical log has local appended entries. The main checkout also has pre-existing changes under `understand/` and unrelated report/build files. Preserve them; do not reset the checkout or stage the whole tree.

## Exact worktrees to resume from

| PDE | Worktree | Branch | Verified clean HEAD | Retain |
| --- | --- | --- | --- | --- |
| Poisson | [2026-09-07-mr-poisson2d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-poisson2d) | `exp/2026-09-07-mr-poisson2d` | `ef2552800756bd9e6be1b9f44776b0406f056d51` | Nested correction family |
| Heat | [2026-09-07-mr-heat2d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-heat2d) | `exp/2026-09-07-mr-heat2d` | `5974d3e0df3f2e0906059564a72511de375ea2a5` | Initial + tail training |
| Burgers | [2026-09-07-mr-burgers2d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-burgers2d) | `exp/2026-09-07-mr-burgers2d` | `ad9e2e7efa7894230cc0494db948ce17fd5f67ee` | Original head + strict stopping |
| Reflective waves | [2026-09-07-mr-wave2d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-wave2d) | `exp/2026-09-07-mr-wave2d` | `277ef65c267dd1fee9090c50b6294dcc83ba2531` | Accelerated nested decoder |

Main carries the combined reports, but its original heat rollout is a frozen, known-broken behavioral baseline. Do not start a new experiment from that implementation merely because it is on main. Each PDE worktree includes its corrected dependencies; the main report alone is not an integrated solver distribution.

## Accepted results at the largest mesh

Each row uses 1024 intervals per spatial axis and the stated full development cohort. Times are pooled medians of complete GPU queries. Each FOM comparator is the fastest tested passing iterative setting from the same job, mesh and cohort. The nominal physical error target is 5%, with the recorded reference and numerical checks also required.

| PDE (cases) | Relative-error normalization | Before → retained worst error % | ROM ms | Iterative FOM ms | FOM / ROM | ROM target |
| --- | --- | --- | --- | --- | --- | --- |
| Poisson (42) | Static field L2 | 7.280248 → 6.110576 | 6.043163 | 89.205305 | 14.761360× | Miss |
| Heat (16) | Current field L2 | 7.595346 → 4.762515 | 11.623313 | 53.854375 | 4.633307× | Pass |
| Burgers (6) | Initial field L2 | 3.907620 → 3.884680 | 55.924140 | 68.086457 | 1.217479× | Pass |
| Reflective waves (4) | Initial energy-state | 6.213781 → 5.145194 | 202.941565 | 601.496361 | 2.963889× | Miss |

Poisson and reflective-wave ratios describe runtime only because their ROMs miss the physical target. Direct DST remains faster for these linear PDEs. Errors with different normalizations cannot be ranked across PDEs. Burgers before/after errors come from the stopping comparison; the displayed timing pair is from the later job, whose original-head fields were checked against that comparison.

[Full campaign report: all resolutions and rejected alternatives](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-accuracy-improvements-and-wave-speed.md). Its adjacent JSON retains repetition arrays, individual panels and source hashes.

## Architecture and findings to carry forward

The shared decoder structure is $u(X;z)=G(X)h(z)$: a learned spatial bank $G$ and a nonlinear coefficient head $h$. The query supplies physical fields or forcing; solving for latent coordinates remains an online task. Offline operator assembly, caches and compilation are distinct from charged complete-query work. Keep each PDE's audited weak-form operator and input contract.

**Poisson.** Fixed-capacity staged training and bank expansion alone failed to improve complete online accuracy. The expanded bank became more expressive, but its original nonlinear head did not use that capacity well. The accepted accuracy-improving research candidate adds training-residual correction directions:

$$D_q(z,y)=G\bigl(h(z)+C_qy\bigr).$$

Here $X$ denotes spatial evaluation points, $z$ the nonlinear latent coordinates, $C_q$ the first $q$ frozen correction directions, and $y$ their linear coefficients. The spatial bank $G$ is evaluated on the requested mesh.

The frozen basis supports prepared prefixes 8, 16, 32; `r128_q32` was declared primary before evaluation. The nonlinear optimizer still has 16 variables, with 48 total coordinates for the primary. Linear coefficients are eliminated exactly by QR projection and recovered after solving for the nonlinear coordinates. Both full and projected stationarity, numerical ranks, initialization and decoding were audited. Source-family descriptors and evaluation truth are not online inputs.

On the finest mesh, target-failing cases drop from 8 to 2 out of 42, but primary GPU time increases by 53.882742%. All cases in this family were already opened during development. Prefix switching requires offline preparation of each projected operator/cache and compiled solver; it requires no neural retraining. It does not imply zero setup cost for arbitrary unprepared prefixes.

**Heat.** Keep `nmrom_initial_tail`. Training emphasizes initial fields and difficult reconstruction examples while retaining the spatial bank and online solver. It improves development accuracy. Its small GPU timing change does not imply an improvement including host transfers. The unrestricted linear-bank control is a different reduced model and remains a useful separate comparison.

**Burgers.** Keep the original checkpoint and the strict stationary solver. The two retrained strict heads reach 5.459554% and 5.277323% worst trajectory error, both worse than the original's 3.884680%. Improving initial reconstruction did not preserve trajectory quality. Keep the FOM-exact upwind term and decoder-output-based fitted quadrature. Because the head, initial-guess library and quadrature weights changed together, these runs do not uniquely identify the source of the rollout degradation.

**Reflective waves.** Shared analytic decoder derivatives and guarded Cholesky solves remove repeated geometry work. The unchanged-step arm has numerical trajectory parity; a larger integration step is a separate change with refinement checks. The final nested decoder preserves the phase-trained parent and adds frozen linear training directions. Its final runtime is 22.064145× faster than the original ROM in the same job. Its pooled energy-state error still misses the target even though the newly introduced development cases pass. Initial-scaled displacement/velocity, energy-state error, current-relative errors and energy-conservation drift are different metrics. Keep the documented initializer limitation and the frozen checkpoint/step; do not reinterpret the correction decoder as the independently retrained larger head.

All pre-reset wave experiments remain historical and untrusted under the user's evidence reset. Rejected time steps, unsuccessful larger-head training, failed instrumentation and pre-query attempts remain archived. No accepted numerical result was retracted by this round.

## Checkpoints and reproducibility

[Combined integration inventory](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-accuracy-integration.json) links the source-level owner inventories, archive manifests, retained methods and failed alternatives. Checkpoint paths below are exact; complete SHA256 values are also retained in this handoff's manifest.

| PDE | Artifact | Path | SHA256 |
| --- | --- | --- | --- |
| Heat | Selected head | [initial_tail.pkl](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10/archive/outputs/checkpoints/initial_tail.pkl) | `cb4786b0f3dcfbcf4eccd9704a60b0180d9d5b6244b46f12594e3afd90d0d369` |
| Poisson | original_relative | [original_relative.pkl](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/original_relative.pkl) | `81f945571da60bbfe9adfba5969727ade137c940b212bc6a6e25c525273d417a` |
| Poisson | r128_joint | [r128_joint.pkl](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/r128_joint.pkl) | `a128e7635c318faae215c3bc3d9885ac479616caeaf81f02936e73f871583b3c` |
| Poisson | Frozen correction basis | [r128_joint-training-correction-basis.npz](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/capacity_accuracy09/r128_joint-training-correction-basis.npz) | `88d088d0665ea6bafc949433a1764b3032e8773a47a45f94e4bb0fc1036f5456` |
| Burgers | Original retained head | [sep_hfit_dense_mid_N256_dense.pkl](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-burgers2d/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl) | `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589` |
| Reflective waves | Selected complete coefficient decoder and fixed training-code candidate library. | [head_trained_nested40.npz](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel10/cluster/out/pilot/head_trained_nested40.npz) | `f481e82f73dbec2f743ca25473bc0b4c7c6c68618370caf1d5cf4fb66687f4e3` |
| Reflective waves | Selected training-only affine initializer; supplied-field fitting still runs online. | [initializer_trained_nested40.npz](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel10/cluster/out/pilot/initializer_trained_nested40.npz) | `154ec67eae3da47f97b16cd4838de52b685c22fd7e6e02c978294e0de45fbc60` |

The wave decoder also requires its learned spatial bank. Use the owner inventory's `frozen_bank_and_original_head_lineage` for the reflective Dirichlet bank parameters/tables and original training lineage. Copying only the new head file is insufficient. For every PDE, keep the appropriate initializer, projected operators/configuration and referenced source dependencies together.

The owner inventories contain restoration procedures and manifests for full predictions, references and timing repetitions. Use the native audits plus the main coordinator audits if integrating code; do not rerun the entire scientific search merely to recover already saved results.

To regenerate the accepted campaign report from its retained artifacts:

```bash
cd /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude
OPENBLAS_NUM_THREADS=1 /home/tahmid/Dev/.venv/bin/python reports/generate_accuracy_campaign.py
```

## Proposed starting point for the next session

Proposed worktree: `worktrees/2026-09-13-nmrom-consolidated`; proposed branch: `exp/2026-09-13-nmrom-consolidated`.

Recommended base: the corrected heat branch at `5974d3e0df3f2e0906059564a72511de375ea2a5`. This preserves the heat rollout corrections. Bring in the selected Poisson, Burgers and reflective-wave implementations and checkpoints from the inventories, plus the current main reports. This is a proposal, not an existing worktree or an approved merge.

The repository rules require asking the user to confirm the base and worktree creation before executing that step. Earlier experiment authorization does not resolve the explicitly pending consolidation decision. The handoff itself adds no new approval requirement.

Once the user chooses the next scope, a useful sequence is:

1. Read the canonical log and check current branch heads, dirty files and any newer work. Confirm the proposed consolidation base/name if consolidation is requested.
2. Integrate only within the chosen new worktree, preserving the original trees and the distinction between selected methods and failed research evidence. Resolve all source/checkpoint/config dependencies before presenting it as runnable.
3. Add a concise entrypoint map and verify checkpoint hashes, imports, bounded smoke execution, and unchanged outputs against retained results. Check executable defaults explicitly; an experiment being archived does not make it the default solver.
4. If requested, regenerate PDF tables/slides from the new campaign JSON. Preserve the user's presentation preference to omit FOM-error and combined-gate columns, while stating missed ROM accuracy targets clearly.
5. For new experiments, predeclare the changed mechanism, controls, cohorts, timing contract and acceptance criteria before evaluation. Candidate directions are improved Poisson correction coverage, trajectory-aware Burgers training with an isolated quadrature-fidelity comparison, and wave displacement-gradient/velocity accuracy. Keep final cohorts sealed until the protocol is ready for final validation.

## Runtime and measurement rules

The full AGENTS.md rules apply. Locally, use `/home/tahmid/Dev/.venv/bin/python`; on the cluster, use `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python`. Local GPU jobs are bounded smoke tests through `jaxrun`; real experiments use the cluster GPU partition, a private job directory and the mandatory GPU-backend preflight.

Require double precision and `JAX_DEFAULT_MATMUL_PRECISION=highest`. Burn in before timing, pair error and cost from the same solver call, preserve repetition arrays, and compare runtime within one GPU job. Retain full fields, seeds, configuration and provenance. Regenerate data from seed on the cluster; verify checksums/restoration before exact remote cleanup. Do not overwrite frozen archives, use bare `scancel`, or launch unapproved additional worktrees. Consult AGENTS.md for the exact concurrency, quadrature and cancellation rules.

## Paste into the new session

```text
Read /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md first, then AGENTS.md and
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-accuracy-campaign-handoff.md.
The accuracy campaign is complete. The selected code and checkpoints remain
in four separate PDE worktrees; no consolidated worktree or merge exists.
The proposed corrected-heat base and new worktree still need confirmation.
Use the accepted report and integration inventory; keep absorbing waves
excluded and final paper cohorts sealed. Check whether I have supplied
new authorization or a different next task before acting on this snapshot.
```

## Plain-language glossary

- **Worktree / branch / HEAD:** a separate checked-out directory / its recorded development history / the exact latest commit in that checkout.
- **Bank / head / latent coordinates:** spatial functions / the network choosing their coefficients / small variables fitted during a query.
- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained by a nonlinear decoder.
- **Intervals:** spatial subdivisions along each axis; this is not the total count of grid points.
- **Relative L2 error:** field-error magnitude divided by the stated reference magnitude. Worst means the maximum over the recorded cases and relevant output times.
- **Energy-state error:** combined velocity and displacement-gradient error; it is different from energy-conservation drift.
- **Median GPU ms / FOM over ROM:** middle retained GPU query duration in milliseconds / a timing ratio above one when the ROM is faster.
- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. Burgers uses an iterative nonlinear solver with an FFT preconditioner, not a direct nonlinear solve.
- **Weak form / quadrature / EQ:** residual tested against smooth functions / approximating integrals by weighted samples / fitted empirical quadrature weights.
- **Stationarity / parity / refinement:** sufficiently small objective gradient / agreement with the unchanged method / agreement after refining a numerical discretization.
- **QR / Cholesky:** matrix factorizations used to project or solve small linear systems.
- **Correction prefix / tail emphasis / phase training:** initial columns of a fixed correction basis / giving harder training examples more influence / training displacement and velocity-related quantities together.
- **Checkpoint / initializer / cache:** saved trained parameters / a procedure or model choosing the first solver guess / precomputed values reused by queries.
- **Development / sealed final / target:** cases used for diagnosis and method choice / untouched cases reserved for later final evaluation / the predeclared physical and numerical acceptance criteria.
- **SHA256 / manifest / archive:** a content fingerprint / a record of artifact paths and fingerprints / a retained package that can restore complete result files.
