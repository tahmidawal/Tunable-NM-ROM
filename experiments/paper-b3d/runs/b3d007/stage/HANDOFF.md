# Burgers 3D confirmation and final-evaluation handoff

B005 is fully collected and independently audited development evidence. B006 is training the bounded, two-seed conditioned DeepONet candidate. Final parameters and fields remain unopened. The original overnight deadline was missed during the session interruption; completion is still pending.

## Active attempt

- Job `4018959`, attempt `b3d006`, source `b1d72e0c4c51624d5dfce0cd80e5e65e75b8c531`; config SHA `04c623ef311c996e91401a711969665bda4e1867eac170e5d5a450064ade9a1e`. Remote: `/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d006`.
- GPU preflight passed on pax051 (A100 40 GB). All original training data were regenerated from seed. The first seed has completed trunk and branch pretraining and entered joint training. The requested wall limit is 75 minutes; estimate 35–55 minutes total once allocated. Do not submit another Burgers job while this one is queued or running.
- Monitor `workflow.log`, `slurm.4018959.err`, `out/seed{0,1}/metadata.json`, `out/seed{0,1}/development.json` and `out/complete.json`.
- Both seeds receive the same unchanged learned DeepONet architecture, train-only relative-error-weighted POD trunk supervision, branch supervision in the learned-trunk Gram metric, then joint training. The original fresh 100k-step DeepONet runs remain retained. Select this candidate recipe only if it improves expanded-development worst error for both seeds. Never select the better seed using final data.

## B005 accepted evidence

Both panel audits pass across 2688 paired invocations and 26 sampled analytic NumPy gradient checks. Source auditing verifies 57 staged files against Git content. Both checksum manifests pass. Raw panels are `runs/b3d005/collected/out/seed{0,1}/result.json`; `runs/b3d005/summary.json` retains every timing repetition. Each model receives the same 512 training trajectories and eight observed times, and both panels use the same 16 opened development trajectories.

| Seed | Method | Worst evolved error (%) | Median GPU ms |
| ---: | --- | ---: | ---: |
| 0 | `rom_q0` | 8.987516 | 223.392410 |
| 0 | `rom_q192` | 2.434515 | 379.655695 |
| 0 | `pod_256` | 1.135743 | 310.113549 |
| 0 | `fno3d` | 2.499541 | 6.065142 |
| 0 | `unet3d` | 2.657453 | 2.174425 |
| 0 | `deeponet3d` | 14.953016 | 2.604709 |
| 0 | `transolver3d` | 2.148732 | 3.575348 |
| 1 | `rom_q0` | 8.228348 | 222.188354 |
| 1 | `rom_q192` | 2.554136 | 404.159067 |
| 1 | `pod_256` | 1.135743 | 310.231638 |
| 1 | `fno3d` | 2.469130 | 6.064135 |
| 1 | `unet3d` | 2.354178 | 2.176164 |
| 1 | `deeponet3d` | 13.880043 | 2.613644 |
| 1 | `transolver3d` | 2.668687 | 3.556121 |

The bank has rank 256 and the head dimension is 64. The actual fixed test count is 642; requested cutoff 640 completes a degenerate sine-eigenvalue shell. Both nonlinear ladders are stationary under the recorded tolerance, but remain slower than stronger classical/operator controls. The independent second seed supports reproducibility of this same-grid correction ladder; it does not establish a manifold advantage.

The streamed opened-case spatial discrepancy is 1.982912% versus the unchanged 0.500000% budget. The time discrepancy is 0.254331% versus 0.100000%. Both gates fail and remain explicit. The independent native residual-pair audit passes. The primary final comparison must therefore remain same-grid.

Full local arrays and optimizer states remain in the checksum-covered collected directory. In addition to the compact checkpoint/replay records, every scientific field from B003, B004 and B005 is now durably retained in committed lossless split archives with complete actual-Git restore/hash audits. `SCIENTIFIC-RETENTION.md` and each run's `SCIENTIFIC-RETENTION.json` link the evidence; earlier local-only retention records remain as history. B005 remote cleanup is recorded separately in `COLLECTED.json` and must be checked there.

## Required next work

After B006 exits, run its exact checksum collector, `audit_conditioned.py` with `config-conditioned.json`, and `audit_source.py`. Retain its complete local archive and selected checkpoint/teacher records before deleting only the verified literal attempt directory. Then:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/prepare_final.py --confirmation b3d005 --conditioned b3d006 --attempt b3d007
```

The preparer does not import a PDE or draw inputs. It freezes both checkpoints, all four operator recipes, every selected POD/bank/direction asset, the solver configuration, K/R architecture and basis shapes, the physical-reference protocol hash, and the final count/seed. If the conditioned recipe fails the common two-seed selection rule, both original 100k DeepONets are retained automatically.

Retain every `assets` entry from the resulting `config-final.json` in Git, then commit source/configuration/freeze and stage with:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/cluster/stage_frozen.py --config config-final.json
```

`final_campaign.py` first replays real opened rows 512 and 519 for both seeds using frozen offline arrays and checkpoints. Both field/iteration/stopping-reason gates must pass before the process opens any final seed. Only then does it generate the prospective 32-case seed 920399 once per frozen panel, compare both seeds/all controls in the same allocation, and run the prospectively frozen streamed physical-reference checks. The physical refinement thresholds must never be loosened after access. No offline fitting occurs during final query evaluation.

Final acceptance requires checksum collection, both `audit.py` field checks, both `audit_stationarity.py` checks, `audit_contract.py` membership/freeze/reference checks and `audit_source.py`, plus actual retention and exact remote cleanup. `audit_contract.py` allows only a fixed machine-roundoff bound for the derived viscosity exp/log; all direct random draws and between-seed memberships remain exact. The expanded-development audit observed that same CPU-library discrepancy before final access.

The coordinator owns canonical LAB-LOG.md and main paper reports. No merge or push is authorized. Untracked `checks/conditioned/{joint,pretraining}` are preserved from the previous owner; do not remove them.

## Glossary

- **Bank/head:** learned spatial functions / nonlinear latent-to-coefficient map.
- **Correction rank:** number of additional online linear coordinates, denoted $q$.
- **POD/FOM:** linear snapshot basis / full numerical PDE solver.
- **Development/final:** opened selection cases / reserved evaluation cases drawn after freezing choices.
- **Same-grid error:** discrepancy against the converged numerical solver on the identical mesh and time step, normalized by the supplied initial-field norm.
- **Median GPU ms:** median full-query device time, with dense requested inputs/outputs and interpolation charged; comparisons are within the allocation.
- **Stationarity:** the recorded normalized local gradient stopping rule; sampled gradients are independently recomputed.
- **Empirical refinement:** change under a finer numerical solve, not a rigorous continuum-error certificate.
- **Frozen offline arrays:** exact precomputed basis and correction data reused during evaluation.
