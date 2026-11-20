# B8 optional children on the matched expanded parent

Prospective configuration only; no automatic submission. Use one GPU for at most120allocated minutes, respecting shared concurrency/accounting/backup requirements. Parent must first complete and have verified terminal retention; an incomplete parent is not eligible.

New opt-in invocation:

```
train_residual_tree_multidepth.py --out OUTPUT --parent-job b8parent32 --fresh-children --parent-initializer gn-only --assembled-jacobian reverse --qualification-job b8implprofile --train-jobs b8trainrefs --dev-jobs b8devrefs --updates 30000 --family-sampling split-primary --spatial-sampling residual-mixture --sampling-alpha .5 --all-final-refit
```

Default B7 path, warm-child behavior, forward assembled Jacobian and prior initializer semantics remain unchanged. Exactly one warm checkpoint or fresh children is required. The new expanded-parent path requires fresh children,32 root variables, qualified reverse derivatives and GN-only parent fitting; cross-parent feature/head transfer and old code reuse are deliberately unsupported in this first configuration.

Verify completed expanded-global parent result/model retention, selected hash, exact160/52 trajectory order, expanded provenance,1760 training codes and corresponding encoder fields. Recheck deterministic expanded definitions through the common loader. Use all updated1760 root training codes directly. Fit ALL572 development parent codes anew from nearest/interpolated updated training codes and zero, three60-step GN starts plus120-step GN continuation of the best unfinished start. Never inherit B7 development codes. Preserve fit statuses and partial initialization every16fields.

The parent weights remain frozen during child training. Initialize new feature/head weights and independent child codes from the missing residual using the existing tangent solve, clipping and backtracking. Check exact zero-child preservation. Train shared nonlinear children across coarse4/uniform20/prescribed20 with independent code banks,30k updates, five equally weighted families and training-only residual-mixture spatial sampling. Retain the existing final all-time/all-layout code refresh and selection protocol. Reverse derivatives require six retained checks plus matching stable.py/residual_tree.py hashes against the retained SOURCE.sha256; overall PDE-trace failure remains recorded.

Training eligibility is distinct from representation/PDE qualification: a completed retained matched parent may train children even when its standalone error exceeds2%. No gate is silently waived for assembled representation, actual4096 reconstruction or PDE tests.

## Downstream interfaces

Results add parent_job, expanded_parent, parent retention, fresh-child flag, zero-child audit and derivative provenance. Parent checkpoint data_record still describes the original primary cache; explicit train_labels/dev_labels/expanded_sources and1760 training codes describe combined data. Consumers must not assume data_record alone enumerates codes.

evaluate_residual_tree.py, solve_residual_tree.py, analyze_residual_span.py, extend_residual_fits.py and finish_residual_training.py now use neural/parent_provenance.py: absent parent_job means historical B7; explicit expanded parents require completed retained hashes, exact160/52 ordering/provenance and1760 matching training codes/encoder fields. The all-case evaluator refits every expanded-parent development root rather than reading B7 development fits; it keeps its preexisting fit algorithm. The solver initialization uses the selected parent's complete training library. Diagnostic results retain parent provenance. Do not relabel new-parent models as B7. Current source jobs and their interpretation remain unchanged. The final-refresh protocol still refreshes final-update codes before selecting the best weights; selected-bank finishing remains a separate explicit task if an earlier checkpoint wins.
