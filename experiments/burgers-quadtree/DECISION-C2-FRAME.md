# C2 conditional-frame checkpoint decision — 2026-11-18

Close the tested low-capacity frame recipes; preserve every checkpoint. This is a representation failure before any PDE solve, not an operator failure and not a proof that all conditional architectures fail.

All32 family-A development trajectories were scored at the five prescribed evolved times. N128 coefficients were held unchanged when rescoring atN256. No final tests were opened.

| Recipe | Coordinates | Worst evolved projection | Maximum evolved scoring change |
|---|---:|---:|---:|
| Diffusion-envelope frame, transferred local-POD |128|7.2153%|0.008805%|
| Initial-width frame, transferred local-POD |128|5.4090%|0.005766%|
| Initial-width frame, newly aligned hybrid |192|6.8914%|0.004367%|

The aligned teacher used only TRAIN128, ten physically weighted alternating low-rank updates, rank128 and30,000 smooth neural coordinate-regression updates. Its training objective decreased, but development accuracy did not improve over transferred features. All three best recipes fail most strongly on development case9. The best newly learned pure-neural128-plus-coarse64 recipe is worse (17.576%); this negative is retained rather than selecting only the hybrid result.

These recipes miss the1% target by a wide margin independently of rollout. No PDE/cost success is inferred, and no selector is developed for them. Initial-wall scoring effects are separated from evolved-time scoring changes; the latter are too small to explain the failure. Weighted synthetic rank3 fixture passes, so simple weighted-fit plumbing failure is not supported by that fixture, though it is not a guarantee of optimal teacher training or generalization.

Artifacts: `evidence/c2frame-assessment.json`, `evidence/c2trainframe-retention.json`, `evidence/c2aligned-checkpoint-backup.json`, `checkpoints/c2aligned/`, and immutable remote outputs under `c2frame` and `c2trainframe`. Canonical geometry/training registration is in `AMENDMENTS.md` and `CONTINUATION-C2-FRAME.md`.

This closes the registered C2 recipe. C1 compressed accurate models and family-B full physics/output/cost qualification remain separate ongoing work. It does not change the physical accuracy requirement or claim the overall speed objective has been achieved.
