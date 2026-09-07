## 2026-09-07

### Fresh absorbing and reflective wave head comparison — completed bounded campaign

User-approved wave branch/worktree was created from `3f8ccc4`, using its own cluster namespace. Implementation owner wrote only the wave tree; coordinator wrote only the repair tree plus this canonical log and canonical reports. All old wave code, banks, checkpoints, data and conclusions remain untrusted historical material. Fresh FOM, learned coordinate network, heads and actual reduced evolution were implemented independently. No merge was performed.

Reference attempts `verify01` and `verify02` failed their declared resolution gates and are retained. Before neural training, the localized family was explicitly revised to a Gaussian core with a smooth compact taper and narrower center range; no Gaussian descriptor enters the head. Expanded fresh verification and independent spectral/field audits accepted the bounded pilot. This does not prove uniform accuracy outside its tested family.

Accepted reference job `3338045`, source `c69bf87439fd2d0ad1673c852e95ec0f322b8c26`, passed 334 declared gates. Two cluster full-pipeline preflights and the local GPU component checks also passed before scientific training.

The table below is generated from original primary-run JSONs. Errors are mean / median / worst of each trajectory's maximum energy norm of state error, normalized by the initial energy norm. Outliers include every trajectory above the declared accuracy ceiling; the full acceptance rule also requires displacement, velocity, fit and temporal checks.

| Boundary | Head | Optimizer seed | Energy error mean | Median | Worst | Outliers | Original time check | Full target |
|---|---|---|---|---|---|---|---|---|
| dirichlet | mlp | 691200 | 0.52043063528874112 | 0.5244627633324237 | 0.88221576486922815 | 16 | True | False |
| dirichlet | quadratic | 691200 | 0.94709984212347331 | 0.93365144074857431 | 1.2528528436742281 | 16 | False | False |
| dirichlet | mlp_velocity | 691200 | 0.50232241004698752 | 0.47239933481236079 | 0.85597600987188094 | 16 | True | False |
| dirichlet | quadratic_velocity | 691200 | 0.9915497195193459 | 0.99077625319055607 | 1.4068886627682697 | 16 | False | False |
| dirichlet | mlp | 691201 | 0.53336115702982312 | 0.46880293374953852 | 0.90439581388898693 | 16 | True | False |
| dirichlet | quadratic | 691201 | 1.0334950749737928 | 1.0695393308831231 | 1.4331683722461284 | 16 | False | False |
| dirichlet | mlp_velocity | 691201 | 0.48733557689532581 | 0.40290810993666659 | 1.0425278193826331 | 16 | True | False |
| dirichlet | quadratic_velocity | 691201 | 1.0141569381583451 | 1.0586699084524078 | 1.3726993220554338 | 16 | False | False |
| absorbing | mlp | 691200 | 0.10509177594067969 | 0.084457018934554692 | 0.21006954213665988 | 6 | True | False |
| absorbing | quadratic | 691200 | 0.12862977227683231 | 0.10279751687268794 | 0.22206065980872294 | 8 | True | False |
| absorbing | mlp_velocity | 691200 | 0.10461218910626355 | 0.085338113498490992 | 0.21918078608778394 | 5 | True | False |
| absorbing | quadratic_velocity | 691200 | 0.2200016585691969 | 0.18375949943800587 | 0.50242535752717343 | 15 | True | False |
| absorbing | mlp | 691201 | 0.10900708119733427 | 0.085550434697553149 | 0.22105436024437713 | 6 | True | False |
| absorbing | quadratic | 691201 | 0.12490009290800706 | 0.10332178269388984 | 0.22410752384367633 | 9 | True | False |
| absorbing | mlp_velocity | 691201 | 0.097171063502375365 | 0.082816237012708682 | 0.20148499026615549 | 4 | True | False |
| absorbing | quadratic_velocity | 691201 | 0.17416323677721018 | 0.17705120364890944 | 0.28670165075577031 | 14 | True | False |

All original heads fail the full provisional engineering target. All original trajectories complete; completion and a small energy-balance defect do not establish accurate evolution. MLP results pass the original time check. Reflective quadratic cases require the separately reported continuation below.

`reflective01` job `3338483`, device `['NVIDIA A100-PCIE-40GB']`, source `fdcc6491657363005cc9060a0cae21dac320a974`: 256 intervals, learned rank 64, latent 16, training seed 690601 / count 64, validation seed 690602 / count 16. Unrestricted learned-bank linear energy-error median 0.034991362279093149, worst 0.077060456022600102; all its physical-error outlier counts are zero. This higher-dimensional comparator does not isolate head structure from dimension.
Selected nonstationary snapshot fits: [('quadratic_velocity', 691200, 1), ('quadratic_velocity', 691201, 1)]. These are explicitly retained failures, not discarded observations.
`absorbing02` job `3338658`, device `['NVIDIA H100 PCIe']`, source `fdcc6491657363005cc9060a0cae21dac320a974`: 256 intervals, learned rank 64, latent 16, training seed 690601 / count 64, validation seed 690602 / count 16. Unrestricted learned-bank linear energy-error median 0.030783727900898332, worst 0.046270465240905542; all its physical-error outlier counts are zero. This higher-dimensional comparator does not isolate head structure from dimension.
Selected nonstationary snapshot fits: [('mlp_velocity', 691200, 1), ('quadratic_velocity', 691201, 1)]. These are explicitly retained failures, not discarded observations.

#### Frozen-checkpoint numerical continuation

Exact original latent position and velocity, trained heads, learned bank and operators were retained. The old finest step was repeated for cross-GPU parity before the extra steps. No retraining, refitting or final-cohort opening occurred. The original primary-step verdict was not replaced. All input checkpoint hashes were reconciled with the completed parent archive.

| Attempt / job | Finest pair passing cases | Unresolved cases | Worst finest-pair state difference | Finest-step state error median | Worst | Resolved-subset state error median | Resolved-subset outliers |
|---|---|---|---|---|---|---|---|
| refquad20001 / 3339304 | 15 / 16 | [2] | 0.073045862826560559 | 0.93368537727653866 | 1.2596438808442822 | 0.9317189847626588 | 15 / 15 |
| refquadvel20002 / 3339553 | 16 / 16 | [] | 0.00036791346757167407 | 0.99078602922802927 | 1.4138638857612811 | 0.99078602922802927 | 16 / 16 |
| refquad20101 / 3339615 | 16 / 16 | [] | 0.00040202925729039774 | 1.0695402119037047 | 1.4331285434580505 | 1.0695402119037047 | 16 / 16 |
| refquadvel20101 / 3339647 | 16 / 16 | [] | 0.00048421877407558007 | 1.0589520073731027 | 1.3692798978602272 | 1.0589520073731027 | 16 / 16 |

The continuation remains a bounded numerical diagnostic; unresolved cases are not claimed globally converged. The first continuation used regenerated scales differing only at roundoff; the independent audit recomputed full-field metrics using original stored normalizations. Later attempts used the stored scales exactly.

Independent reviewers checked primary summaries, fit selection and stationarity, phase masks, completion, energy and refinement aggregation. Coordinator NumPy/SciPy audits independently reconstructed the neural bank, mass/edge-stiffness/face-damping operators, head values/Jacobians, reconstruction and tangent field errors, rollout energies, and continuation field metrics. Audit scripts and JSONs live in the repair campaign `review/` directory. Trained checkpoints and native outputs are tracked on the wave branch; full checked archives, logs, submissions, cancellation records and cleanup receipts are tracked on the repair branch.

Every accepted run logged GPU backend, f64 and highest precision. No cross-job wall-clock comparison is made. Original absorbing job `3338489` was canceled before starting through the numeric-ID helper; its staged source was archived and cleaned. Local stage `refquadvel20001` was never submitted and was superseded by its hardened successor. Every executed cluster directory was checksum-pulled and deleted after completion.

Retractions/limits: no earlier wave finding is reinstated; no new head succeeds at the full target; original reflective quadratic primary trajectories are not globally time converged. Optimizer repeats share one data split, and MLP/quadratic parameter counts differ. The final cohort remains unopened. These are scalar two-dimensional waves with separate learned weights per boundary; no three-dimensional wave, Navier–Stokes or grid-independent online-cost result follows.

Next proposed controlled experiment: retain the verified FOM and learned bank, compare a latent-dimension ladder with matching linear dimensions, then assess wave-state/dynamics-aware training. Velocity-tangent fitting alone is insufficient here. Further architecture experiments and merging the wave branch into the repair branch remain open pending the user's choice.

Canonical generated report: `reports/2026-09-07-fresh-wave-head-transfer.md`; source generator, figures and reproducibility manifest are beside it. The report and this closing table derive numerical values directly from immutable JSONs.
