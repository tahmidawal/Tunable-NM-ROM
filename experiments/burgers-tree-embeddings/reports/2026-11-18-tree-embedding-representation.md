# Burgers tree embeddings: representation diagnosis and continuation

This generated report covers the original bounded failure and the user-authorized continuation. Representation results are development evidence; nonlinear compression, dense audits and fixed-tree physics are still being investigated. No complete PDE-runtime win or final-test result is established.

## Why the original best error was still high

The original best tree had 16.5299% worst projection error at 128 adjustable variables. Its actual-grid witness remains 16.5737%; that failure is real, not the subsequently retracted quadrature result.

The evidence supports several contributing limitations. The original parent alone has 40.8695% worst development error. Better-scaled training with a richer frozen feature span reduces selected training error from 28.5599% to 4.5584%, but development error remains 31.7719%. Thus optimization and generalization are both relevant; changing scaling and the feature bank together does not isolate one cause. The old head’s relative departure from its best affine map is 7.9823%, so describing it as exactly linear would be wrong.

A separate frozen-weight audit identifies a definite global-head restriction: a linear skip plus a finite-width final hidden layer confines every output to a fixed affine span. Giving that span unrestricted coefficients produces the bounds below. Their remaining error cannot be fixed by additional code fitting on the same frozen head. The gap above the bound can reflect both the nonlinear image and imperfect fitting; it does not isolate those two causes. The smaller normalized model uses the original cohort; the larger two use the expanded cohort.

| Global latent variables | Frozen features | Output-space rank | Relaxed output-span error | Nonlinear head error |
|---|---:|---:|---:|---:|
| 16 | 512 | 144 | 8.8096% | 31.7719% |
| 64 | 768 | 192 | 5.5359% | 16.7303% |
| 128 | 768 | 256 | 4.0264% | 7.8825% |

![Frozen global head limits](b1-global-head-limits.png)

Attribution on the old fixed trees provisionally improves when parent and local coefficient restrictions are relaxed. Those sampled-integration diagnostics also increase variable counts and are not an additive causal decomposition. Stronger evidence is the resolved richer local span: its worst original-panel error is 0.2282% with 2000 coefficients, confirmed by actual output at 0.2282%. This rules out a universal error floor at the original value for the broader local representation. It does not yet establish efficient nonlinear compression.

![Training and numerical diagnosis](b1-training-and-integration.png)

## Expanded matched-data comparison

Training uses the registered prefix expansion to 128 trajectories. Development has 32 random trajectories and the same 12 alignment cases; every one of the 11 snapshot times is scored in the current-time physical norm. Final-test solution fields remain unopened. Shared local POD uses training coefficients only, fixed windows and retained parents. Its allocations use the known initial field and a sampled speed estimate, not evolved reference fields or a certified propagation bound.

| Representation | Total variables | Original random panel | Added random panel | Alignment panel | Worst all |
|---|---:|---:|---:|---:|---:|
| Direct local 4 | 336 | 2.0593% | 1.2189% | 3.8250% | 3.8250% |
| Direct local 8 | 544 | 1.1658% | 0.7486% | 2.3990% | 2.3990% |
| Direct local 16 | 960 | 0.5889% | 0.3635% | 0.8908% | 0.8908% |
| Direct local 24 | 1376 | 0.3792% | 0.2173% | 0.4229% | 0.4229% |
| Direct local 36 | 2000 | 0.2150% | 0.1327% | 0.2156% | 0.2156% |
| Global nonlinear 64  | 64 | 7.1775% | 10.3528% | 16.7303% | 16.7303% |
| Global nonlinear 128  | 128 | 2.9436% | 6.2896% | 7.8825% | 7.8825% |
| Global nonlinear 128 wide | 128 | 3.6511% | 6.7567% | 8.2982% | 8.2982% |
| Child nonlinear 8  | 544 | 0.8083% | 0.5119% | 1.5059% | 1.5059% |
| Child nonlinear 16  | 960 | 0.4634% | 0.2687% | 0.6520% | 0.6520% |
| Same child16 head, 64 nodes | 1152 | 0.4238% | 0.2573% | 0.4044% | 0.4238% |

The direct local entries use resolved physical Gram integration; actual-grid audits are separate. Spectral-feature nonlinear controls are architecture-inspired NM-ROMs, not faithful native coordinate-MLP Tunable reproductions.

The wider global128 repair uses960 stable features and hidden widths256,256,960. Its selected projection error remains 8.2982%, versus 7.8825% before the combined feature/output-width change. The selected fits have maximum stationarity below1e-9. Widening this recipe did not solve its generalization/coding limitation; the unsuccessful checkpoint and full training history remain. This repair is not an isolated attribution test because both feature span and hidden widths change.

The extended global control reaches 0.3227% with 1280 variables, better than the admitted direct tree at its larger variable count. Retain that result, with an essential qualification: its trailing training singular value is only 3.01e-18 of the largest. These are valid saved orthonormal spaces extended into numerically null training directions, not reliably learned variance directions. Their stability under retraining is unqualified. The complete data-independent sine space provides a separate control without this ambiguity. Neither variable counts nor projection errors establish PDE cost.

Reordering the identical training rows in two prescribed permutations gives global1280 errors 0.3161%, 0.3235%. The accuracy changes modestly in this limited check; the directions remain numerically null. A separate, explicitly constructed global space avoids those arbitrary SVD directions: retain the reliable training POD512 prefix and append orthogonalized low-frequency sine directions in fixed order. It reaches 0.3304% with960 variables, compared with frozen child16 refitting at 0.6492% with the same count. Its measured orthogonality defect is 1.7e-14. This is a POD-plus-Fourier control, not pure POD or a neural method. It currently defeats the child's error-per-variable comparison; cost and actual-output qualification remain separate.

| Global span control | Variables | Worst all | Relative trailing training singular value |
|---|---:|---:|---:|
| Global saved POD prefix 768 | 768 | 1.6145% | 4.79e-07 |
| Global saved POD prefix 960 | 960 | 1.2726% | 2.6e-08 |
| Global saved POD prefix 1024 | 1024 | 1.1579% | 6.67e-09 |
| Global saved POD prefix 1280 | 1280 | 0.3227% | 3.01e-18 |
| Global saved POD prefix 1376 | 1376 | 0.2958% | 9.25e-19 |
| Global saved POD prefix 1408 | 1408 | 0.2882% | 2.07e-20 |
| Complete sine 24 per axis | 576 | 4.1987% | Not trained |
| Complete sine 32 per axis | 1024 | 1.9606% | Not trained |
| Complete sine 36 per axis | 1296 | 1.3369% | Not trained |
| Complete sine 40 per axis | 1600 | 0.9292% | Not trained |
| Complete sine 48 per axis | 2304 | 0.4543% | Not trained |
| Complete sine 64 per axis | 4096 | 0.1139% | Not trained |
| POD512 + prescribed Fourier completion | 768 | 0.5263% | Not a null-tail completion |
| POD512 + prescribed Fourier completion | 960 | 0.3304% | Not a null-tail completion |
| POD512 + prescribed Fourier completion | 1024 | 0.3097% | Not a null-tail completion |
| POD512 + prescribed Fourier completion | 1280 | 0.2367% | Not a null-tail completion |
| POD512 + prescribed Fourier completion | 1376 | 0.2113% | Not a null-tail completion |

![Error against active variables](b1-representation-capacity.png)

![Actual-grid witnesses](b1-actual-field-witnesses.png)

The image display is subsampled for readability from saved actual-grid arrays. Errors use the complete arrays. The rows show different worst cases and are not a matched-case attribution test.

Independent fixed-code scoring confirms local POD24 on every snapshot at the denser grid: worst 0.4229%, with largest change from the resolved physical metric 1.3e-09. The actual-output witness is 0.4229%. All 1248 of 1248 child directions remain numerically independent after removing the parent; their squared singular-value ratio is 1.38e-07.

The following allocation controls use exactly the same local checkpoint and coefficient dimension per node. Uniform and prescribed entries at each matched node count isolate an allocation/depth-distribution effect. The random allocation is a diagnostic control, not the deployed policy.

| Allocation and retained nodes | Total variables | Worst projection |
|---|---:|---:|
| uniform20 | 608 | 2.2911% |
| prescribed20 | 608 | 1.9276% |
| uniform84 | 2144 | 0.9271% |
| prescribed84 | 2144 | 0.2763% |
| random52 | 1376 | 2.3032% |

## Retraction and numerical controls

The preliminary rich-span score 0.0925% was invalid as a gate result: unchanged coefficients gave 296.5150% on denser scoring. Too few integration points per small patch missed oscillatory directions. This apparent pass is retracted; its outputs are retained. The repaired metric resolves every cell-core and halo boundary using tensor integration. The independent integration-rule difference is 7.45e-12, with actual-grid agreement recorded above. A first repair implementation failed before scientific output due to the reference’s interval/mode indexing convention; its allocation and logs are retained.

The historical attribution JSON accidentally calls a selected update number `old_dev_worst`. It is not an error metric; this report reads the actual global error from its source run. The source label is corrected for future reruns, and the original raw file remains unchanged.

## Advancement and remaining work

The direct local POD24 model passes representation and its independent dense audit, so fixed-tree physics has begun. Frozen-weight, longer dense fitting of child8 reduces its worst error to 1.4861%; maximum stationarity remains 0.000136. The small change weakens the explanation that its entire remaining error is a short code fit, but it is not a proof of a global minimum. Shared nonlinear children train against the assembled physical metric, with jointly refitted parent and child variables at multiple depths.

Child16 also finishes its registered training and frozen-weight fitting audit. Longer fitting reaches 0.6492%, with maximum stationarity 1.61e-05. Every random development trajectory clears the representation threshold, but alignment cases still fail. A finite same-checkpoint prescribed-tree growth diagnostic is registered to test that specific allocation/compression bottleneck while preserving parents and windows; it is not yet a qualified online adaptive method.

That frozen-head growth diagnostic now passes: 64 retained nodes give 1152 total variables, worst physical projection 0.4238% and actual4096 witness 0.4238%. Fixed-code2048 scoring covers every snapshot; its largest change from the resolved metric is 1.34e-09. Zero-extension preserves the previous field with maximum relative change 0; every audited geometry retains all 1024 independent child directions after root removal. The head weights did not change. This achieves approximately the direct24 model's reconstruction accuracy with fewer variables, while the global960 control remains more accurate at a smaller count. A coupled nonlinear PDE solve, same-checkpoint allocation controls and complete costs remain required. This is initial-callable prescribed growth, not online residual-driven adaptation.

## Fixed-tree PDE diagnosis

These pilots solve complete trajectories from the shared initial callable with the frozen prescribed tree and global POD768 control. Within each pilot, both use the same exact fine-grid upwind weak operator, time integrator, common test set and integration rule. BE is backward Euler; the registered CN repair is Crank–Nicolson. Q denotes integration intervals; Q=L is dense integration. Smaller Q retains fine stencil spacing1/L and requires independent returned-state refinement. Reference fields are opened for scoring only after both methods finish each case. The table reports all requested evolved times, not only the final output.

| Case | Method | Spatial intervals | Integrator | Tests M | Integration Q | Time step | Worst evolved error | Every step stationary | Physical accuracy pass |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| dev000 | tree_direct24 | 1024 | BE | 5504 | 1024 | 0.0025 | 2.5373% | True | False |
| dev000 | global_POD768 | 1024 | BE | 5504 | 1024 | 0.0025 | 2.4895% | True | False |
| dev003 | tree_direct24 | 1024 | BE | 5504 | 1024 | 0.0025 | 0.2945% | True | True |
| dev003 | global_POD768 | 1024 | BE | 5504 | 1024 | 0.0025 | 0.2944% | True | True |
| align002 | tree_direct24 | 1024 | BE | 5504 | 1024 | 0.0025 | 2.2570% | True | False |
| align002 | global_POD768 | 1024 | BE | 5504 | 1024 | 0.0025 | 2.0991% | True | False |
| dev000 | tree_direct24 | 2048 | BE | 5504 | 2048 | 0.0025 | 2.0921% | True | False |
| dev000 | global_POD768 | 2048 | BE | 5504 | 2048 | 0.0025 | 1.9700% | True | False |
| dev003 | tree_direct24 | 2048 | BE | 5504 | 2048 | 0.0025 | 0.2248% | True | True |
| dev003 | global_POD768 | 2048 | BE | 5504 | 2048 | 0.0025 | 0.2246% | True | True |
| align002 | tree_direct24 | 2048 | BE | 5504 | 2048 | 0.0025 | 1.9630% | True | False |
| align002 | global_POD768 | 2048 | BE | 5504 | 2048 | 0.0025 | 1.6391% | True | False |
| dev000 | tree_direct24 | 4096 | BE | 5504 | 4096 | 0.0025 | 1.9597% | True | False |
| dev000 | global_POD768 | 4096 | BE | 5504 | 4096 | 0.0025 | 1.7218% | True | False |
| dev003 | tree_direct24 | 4096 | BE | 5504 | 4096 | 0.0025 | 0.1932% | True | True |
| dev003 | global_POD768 | 4096 | BE | 5504 | 4096 | 0.0025 | 0.1930% | True | True |
| align002 | tree_direct24 | 4096 | BE | 5504 | 4096 | 0.0025 | 1.8448% | True | False |
| align002 | global_POD768 | 4096 | BE | 5504 | 4096 | 0.0025 | 1.4849% | True | False |
| dev000 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.0025 | 1.4724% | True | False |
| dev000 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.0025 | 0.0824% | True | True |
| dev003 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.0025 | 1.4331% | True | False |
| align002 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.0025 | 0.6482% | True | True |
| dev000 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.00125 | 1.4715% | True | False |
| dev000 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.00125 | 0.7978% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.00125 | 0.0824% | True | True |
| dev003 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.00125 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 5504 | 2048 | 0.00125 | 1.4210% | True | False |
| align002 | global_POD768 | 2048 | CN | 5504 | 2048 | 0.00125 | 0.6518% | True | True |
| dev000 | tree_direct24 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7950% | True | True |
| dev000 | global_POD768 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0822% | True | True |
| dev003 | global_POD768 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6483% | True | True |
| align002 | global_POD768 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6481% | True | True |
| dev000 | tree_direct24 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.7947% | True | True |
| dev000 | global_POD768 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.0822% | True | True |
| dev003 | global_POD768 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.6478% | True | True |
| align002 | global_POD768 | 2048 | CN | 22016 | 2048 | 0.0025 | 0.6481% | True | True |
| dev000 | tree_direct24 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.7946% | True | True |
| dev000 | global_POD768 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.0822% | True | True |
| dev003 | global_POD768 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.6478% | True | True |
| align002 | global_POD768 | 2048 | CN | 44032 | 2048 | 0.0025 | 0.6481% | True | True |
| dev000 | tree_direct24 | 2048 | CN | 5504 | 512 | 0.0025 | 1.4733% | True | False |
| dev000 | global_POD768 | 2048 | CN | 5504 | 512 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_direct24 | 2048 | CN | 5504 | 512 | 0.0025 | 0.0825% | True | True |
| dev003 | global_POD768 | 2048 | CN | 5504 | 512 | 0.0025 | 0.0820% | True | True |
| align002 | tree_direct24 | 2048 | CN | 5504 | 512 | 0.0025 | 1.4328% | True | False |
| align002 | global_POD768 | 2048 | CN | 5504 | 512 | 0.0025 | 0.6482% | True | True |
| dev000 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7887% | False | True |
| dev000 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0825% | True | True |
| dev003 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6479% | False | True |
| align002 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6481% | True | True |
| dev000 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7938% | False | True |
| dev000 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0825% | True | True |
| dev003 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6531% | False | True |
| align002 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6481% | True | True |
| dev000 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7945% | False | True |
| dev000 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.7941% | True | True |
| dev003 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0825% | True | True |
| dev003 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.0820% | True | True |
| align002 | tree_child16_n64 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6524% | False | True |
| align002 | global_stable960 | 2048 | CN | 11008 | 2048 | 0.0025 | 0.6481% | True | True |

![Fixed-tree physics diagnosis](b1-fixed-tree-physics.png)

![Independent numerical sensitivities](b1-numerical-sensitivities.png)

Retained incomplete PDE outcomes: none in the collected pilot records. The trajectory figure emphasizes the dense BE/CN and test-enrichment comparisons; the table retains every completed pilot setting. A low physical error alone does not establish numerical convergence. The change from5504 to11008 tests is large for the tree. Its next enrichment to22016 changes the alignment-case field by0.106838%, just above the0.1% sensitivity limit;11008 is therefore still not admitted. The global control is much less sensitive to these added tests. The subsequent44032-test check has maximum field change0.0115%; its pilot admission check isTrue. This admits22016 tests only for these three direct24/global768 trajectories at the tested setting, pending independent integration, time, spatial and solver-precision checks. It does not admit the nonlinear arm or the full development panel.

The first nonlinear-child PDE pilot reaches physical errors below1% on all three exposed cases, but160 of300 time steps miss stationarity1e-8. The initial alignment fit is also underconverged. Useful zero-child directions and accurate reconstructions do not guarantee an easy nonlinear online solve. The first bounded repair raises only the PDE LM cap from30 to120, keeping the checkpoint, initial fits, equations, integration and tolerances fixed. Its results, including any failures, are recorded below. Initial-fit stationarity is distinct from time-step stationarity; a passing step alone does not qualify initialization.

| Nonlinear PDE run | Case | Worst evolved physical error | Nonstationary steps | Worst step stationarity | Initial-fit stationarity |
|---|---|---:|---:|---:|---:|
| b1childpde | dev000 | 0.7887% | 74/100 | 3.09e-05 | 6.14e-08 |
| b1childpde | dev003 | 0.0825% | 0/100 | 4.62e-11 | 1.29e-08 |
| b1childpde | align002 | 0.6479% | 86/100 | 5.13e-05 | 3.44e-06 |
| b1childpdeR1 | dev000 | 0.7938% | 68/100 | 2.88e-05 | 6.14e-08 |
| b1childpdeR1 | dev003 | 0.0825% | 0/100 | 4.62e-11 | 1.29e-08 |
| b1childpdeR1 | align002 | 0.6531% | 76/100 | 2.88e-05 | 3.44e-06 |
| b1childpdeR2 | dev000 | 0.7945% | 63/100 | 9.81e-06 | 6.14e-08 |
| b1childpdeR2 | dev003 | 0.0825% | 0/100 | 1.97e-09 | 1.29e-08 |
| b1childpdeR2 | align002 | 0.6524% | 60/100 | 1.01e-05 | 3.44e-06 |

![Nonlinear online solver convergence](b1-nonlinear-solver-diagnostics.png)

The second repair is complete and fails:123 of300 time steps miss1e-8 stationarity, and the largest returned-field change from LM120 is0.226685%. Stop further online optimizer tuning for this frozen checkpoint under the two-repair rule. Initial fits also remain underconverged. The representation admission remains valid; this candidate does not advance to adaptation or timing confirmation. Continue the separate direct-coefficient branch through its numerical gates. This is a scoped failure of the tested checkpoint/optimizers, not a universal limitation of local nonlinear decoders.

Pulse errors remain above the physical gate in the initial pilot for both methods despite tightly converged weak solves. Spatial, temporal, test and integration refinements distinguish numerical error from reduced dynamics. The initial pilot omitted explicit burn-in immediately before timing: those original wall times are retained as unqualified diagnostics and support no speed comparison. Follow-ups include burn-in. Pilot clocks also charge the joint comparison setup to each method; they are not valid method-specific speed ratios. Complete timing must measure each method's own initialization, assembly and output. The small pilot is not all-development qualification, deployable adaptation, or a complete-runtime benchmark.

The focused continuation changes the training allocation mixture for its nonlinear compression pilot; it is explicitly documented in B1-SHARED-CHILD.md. No truth-assisted online refinement is allowed. The verified fine-upwind weak operator remains the primary PDE formulation. Initialization, assembly, failed trials, solving, fallback and requested output must all enter later complete-query timing.

The frozen nonlinear head is now compared on identical node budgets. These are best-found reference-assisted fits, not certified manifold minima; stationarity and actual-grid worst witnesses are explicit. No new weights are trained. A newly passing allocation still needs all-snapshot dense scoring before physics admission.

| Allocation | Variables | Worst all | Actual4096 worst witness | Maximum fit stationarity |
|---|---:|---:|---:|---:|
| uniform20 | 448 | 3.5037% | 3.5037% | 0.000113 |
| prescribed20 | 448 | 2.6805% | 2.6805% | 0.000121 |
| uniform84 | 1472 | 1.2126% | 1.2126% | 0.000106 |
| prescribed84 | 1472 | 0.3562% | 0.3562% | 2.45e-05 |

![Same-checkpoint allocation controls](b1-allocation-controls.png)

Original recipes, unsuccessful repairs and the original decision remain in [the retained B0 report](B0-retained-representation.md). CROM, SNF-ROM, hyper-reduced masked autoencoders and conditional equation-driven CoLoRA have not yet been reproduced in confirmation; they are not counted as defeated. The older burgers-quadtree campaign remains paused and untouched. No merge or three-dimensional work.

## Accounting and provenance

Generated 2026-11-19T01:34:21.275495+00:00. Terminal allocations recorded in the current ledger total 21819 GPU-seconds (6.060833 GPU-hours); nonterminal or not-yet-reconciled names: b1cnM16Q512. Outstanding reservations are enforced by the submission helper, not omitted from its admission calculation. The finite B1 ceiling, recipe definitions, source commits, GPU preflights, raw arrays, checkpoint histories and checksum-verified retention records remain in this lane. This report and all figures are generated from saved data; their source hashes are recorded beside them.

## Glossary

- Representation/projection error: best reconstruction found while the reference field is available; this is not a PDE solve.
- Parent: global field retained when local corrections are added.
- Child variables: independently adjustable numbers controlling one local correction.
- POD: a basis learned by singular-value decomposition of training fields or coefficients.
- Numerical-null directions: basis directions with training variation at rounding scale; their orientation is not reliably determined by the data.
- Complete sine space: every tensor-product sine mode up to the stated axis count, independent of training.
- Stationarity: the fitted weak residual's gradient is small; this alone does not guarantee physical accuracy.
- Gram metric: field inner products that retain overlapping-patch interactions.
- Current-time relative L2: physical root-mean-square error divided by the reference norm at that same time.
- Worst all: largest error across every snapshot and trajectory in all development panels.
- Alignment panel: prescribed cases near tree cell edges/corners, reported separately from random cases.
- Aliasing: integration samples miss variation between sample locations and report an incorrect fit error.
- Zero-child transfer: adding zero-valued correction codes leaves the existing field unchanged.
- Allocated GPU-hours: reserved hardware time actually elapsed, including setup and failures.
- Complete query: initialization, adaptation, assembly, solving, fallback and requested reconstruction together.
