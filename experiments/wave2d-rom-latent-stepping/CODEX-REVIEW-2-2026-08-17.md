## NUMBER MISMATCHES

- **README:26 — “where the same-`dt` FOM is 1.16e-3 from the true FOM at N=64 — roughly 10x below the manifold floor.”**
  - `1.16e-3` is from the six-trajectory verification sample, not the 16-trajectory primary cells: actual primary-cell value is `2.386003e-3`.
  - At K=8, the held-out oracle floor is `0.1718577`; it is 148.5× the six-sample value or 72.0× the primary-cell value, not roughly 10×.
  - Files/keys: `runs/wverify_n64/out/wlat_verify_N64.json`, `config.n_verify`, `V5_samedt_newmark_fom["20"].traj_rel_vs_fom_mean`; `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `samedt_fom.traj_rel_vs_fom_mean`, `oracle_inferred_latent_test.traj_rel_mean`.

- **README:28 — “~440k params.”**
  - Actual decoder parameter counts are 462,657 / 462,913 / 463,425 for K=4/8/16, i.e. approximately 463k, not 440k.
  - Files/keys: `runs/wad_n64_k*/out/wlat_ad_N64_K*_report.json`, `config.n_params`.

- **README:106 — “ratio 2.5 -> 3.0 -> 3.9 -> 4.0 as `dt` halves.”**
  - The directly observed successive half-step ratios are 2.458, 2.486, 3.008, and 3.913 for RS 1→2→4→8→16. The separate RS 20→40 ratio is 4.984. No observed sequence is exactly the one quoted; `4.0` is an extrapolated asymptote.
  - File/key: `runs/wverify_n64/out/wlat_verify_N64.json`, `V5_samedt_newmark_fom`.

- **README:110–111 — “2.4e-3 is ~70x below the ROM errors and ~30x below the manifold floor.”**
  - At K=8, `0.8783078 / 0.0023860 = 368.1`, while `0.1718577 / 0.0023860 = 72.0`. The quoted factors are both wrong.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `rom["lspg:eq256:weak64"].traj_rel_mean`, `oracle_inferred_latent_test.traj_rel_mean`, `samedt_fom.traj_rel_vs_fom_mean`.

- **README:113 — “they agree to 4 digits everywhere.”**
  - They are extremely close—maximum relative discrepancy is about 0.0208%—but the literal four-significant-digit claim fails for several boundary cases. For example POD-64 full-FD is `0.08377818` versus `0.08379557`, rounding to `0.08378` versus `0.08380`; K=16 `eqoff256:weak64` is `0.6831545` versus `0.6831458`, rounding to `0.6832` versus `0.6831`.
  - Files/keys: main `wlat_rom_N64_K*.json`, `traj_rel_mean` versus `traj_rel_vs_samedt_fom_mean`.

- **README:162, 185, 321 — “6.2x” for K=16 ROM/oracle floor.**
  - Actual ratio is `0.68006417 / 0.11060847 = 6.14839`, which rounds to **6.1×** at the stated one-decimal precision. The K=4 and K=8 ratios, 3.27187→3.3 and 5.11067→5.1, are correct.
  - File/keys: `runs/wad_n64_k16/out/wlat_rom_N64_K16.json`, `rom["lspg:eq256:weak64"].traj_rel_mean`, `oracle_inferred_latent_test.traj_rel_mean`.

- **README:166 — POD ratio at k=32 reported as `1.020`.**
  - Using the TEST floor as intended gives `0.140795838 / 0.138130117 = 1.019299`, which rounds to **1.019**, not 1.020. The other four ratios are correctly rounded, and the calculation does use TEST rather than VAL.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `pod_rom["pod_k32:lspg:full:fd"].traj_rel_mean`, `oracle_pod_projection_floor_test["32"]`.

- **README:178 — “The coordinate ROM dissipates 61–84% of the energy.”**
  - For the headline `lspg:eq256:weak64` recipe, the kinematic final ratios are 0.26524 / 0.27224 / 0.39522, corresponding to losses of **73.5% / 72.8% / 60.5%**, not 61–84%.
  - If this instead means all coordinate arms, no such range is valid: some arms grow, including K=16 `weakl64` with kinematic `E_T/E_0=6.259`.
  - Files/keys: the three main ROM JSONs, `rom[*].energy_final_ratio_mean`.

- **README:203 — “Galerkin vs LSPG … identical to 4 significant figures in all 16 variants.”**
  - Only four matched Galerkin/LSPG pairs were run, not 16.
  - Three pairs round identically to four significant figures. Full-FD does not: Galerkin is `0.8762220`→`0.8762`, while LSPG is `0.8766662`→`0.8767`.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, matched entries under `rom`.

- **README:209 — “the 15x-cheaper quadrature.”**
  - It uses 15.016× fewer nodes (`3844/256`), but measured per-step runtime is only 5.523× lower (`13.4935/2.4432` ms). “15× fewer quadrature nodes” is accurate; “15× cheaper” is not as a timing claim.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `timing["lspg:full:weak64"].ms_per_step`, `timing["lspg:eq256:weak64"].ms_per_step`.

- **README:215–216 — “NNLS relative fit residuals of 1.0e-2 at m=256 and 7e-5 at m=1024.”**
  - For the K=8 coordinate decoder, actual values are `1.00743e-2` and **`1.92346e-3`**.
  - `7.24743e-5` belongs to the **POD k=64** `eq1024:weak256` fit, so the prose mixes coordinate and POD cells.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `rom[...].eq_info.rel_fit` and `pod_rom["pod_k64:lspg:eq1024:weak256"].eq_info.rel_fit`.

- **README:252 — “The IC latent solve is a jitted LM (78–97 ms).”**
  - This covers K=4 (`78–80 ms`) and K=8 (`95–97 ms`) but omits K=16, whose recorded IC fits are **831–841 ms**.
  - File/keys: main ROM JSONs, `timing[*].ic_fit_s`.

- **README:270 and 346–347 — “a single f64 FiLM decoder evaluation … with its Gauss-Newton Jacobian costs 25x” the FOM substep.**
  - The entire K=8 latent step costs `2.4432 ms`, or 25.39× the `0.09623 ms` FOM substep. It averages 5.192 warm iterations. Dividing gives approximately `0.4706 ms`, or 4.89× the FOM substep per iteration/Jacobian evaluation, though the JSON does not isolate a Jacobian-only timing.
  - File/keys: `runs/wad_n64_k8/out/wlat_rom_N64_K8.json`, `timing["lspg:eq256:weak64"].ms_per_step`, `rom[...].iters_warm_mean`, `timing.fom_rollout_cn80`.

- **README:316–319 — “Every verification passes at machine precision.”**
  - The residual checks are at `1e-14–1e-16`, but V1 is `2.78066e-8` and the RS=80 Newmark energy drift is `2.25874e-9`. They pass configured tolerances; they are not machine-precision results.
  - File/keys: `runs/wverify_n64/out/wlat_verify_N64.json`, `V1_newmark_rs80_vs_cn_fom`, `V4b_newmark_rs80_energy_drift`, `V2_residual_ops`, `V3_weak_allmodes_vs_strong`.

- **Unverifiable at audit time, not counted as errors:** every N=128 cell in README:279–285 and all conclusions depending on it, including N=128 zero blow-ups and flat cost; no N=128 JSON was supplied. V7/V8 values at README:86–88 are also absent from the supplied verification JSON. The train energy drift `3.2e-11`, cluster job IDs/GPU models, Burgers/Poisson comparison numbers, and published rebuttal numbers are likewise not represented in the supplied JSONs.

## OVERREACHING CLAIMS

- **Reading 1 — partly supported.**  
  Sentence: “**The recipe ports, the machinery is right, and the ROM does not work on wave.**”  
  The N=64 data support the narrow observation that all tested variants finish and coordinate errors remain `0.592–1.024`. They do not establish that all machinery is right; some verification results are not at machine precision, V7/V8 are missing from the supplied artifact, and successful completion is not a correctness proof.  
  Suggested wording: “The supplied operator checks pass their tolerances, all tested N=64 rollouts complete, and the tested coordinate ROMs remain much less accurate than their inferred-latent fits and POD controls.”

- **Reading 2 — not supported as a mechanism.**  
  Sentence: “**The mechanism is amplification, not injection.**”  
  The fitted exponent is defensible: ordinary least-squares regression of `log(excess_mean)` on `log(H)` gives **1.688**, so “roughly `H^1.7`” is correct. But the oracle endpoint is independently refit at every H, and `excess_mean = rom_mean - oracle_mean` is a difference of relative-error norms, not a decomposition of injected and amplified errors. The experiment does not propagate a controlled one-time perturbation or remove repeated per-step forcing. It therefore cannot distinguish amplification from repeated injection, changing oracle-fit difficulty, optimizer effects, or their combination.  
  Suggested wording: “The step-induced excess is small at H=1 and grows approximately as `H^1.69` over H=1–10; this is consistent with accumulation or amplification, but the diagnostic does not separate them.”

- **Reading 2 — the time-step conclusion is too general.**  
  Sentence: “A latent-stepping ROM on a conservative PDE is not a convergent scheme in `dt`.”  
  The three K=8 points for one variant show the time-discretization floor falling 28.90× while its error rises 7.54%. Three non-asymptotic points, one latent dimension, one architecture, one seed, and one objective do not prove nonconvergence of latent-stepping ROMs on conservative PDEs.  
  Suggested wording: “For K=8 `lspg:eq256:weak64` over RS=8,20,40, refinement does not improve the final error and instead raises it by 7.5%.”

- **Reading 3 — the structural derivation is wrong for the reported LSPG arm.**  
  Sentence: “A linear subspace turns the linear Newmark residual back into a Newmark scheme on `Vᵀ L V` — conservative and unconditionally stable by construction.”  
  For strong LSPG, if `A=I-aL`, minimizing `||AVc-b||²` yields `VᵀAᵀAVc=VᵀAᵀb`, not the Galerkin equation `Vᵀ(AVc-b)=0` that produces the standard reduced operator `VᵀLV`. Weak-form weighting adds another `QᵀQ` factor. Only an appropriate Galerkin/full-unweighted formulation has the claimed direct reduction. The headline POD cells are `pod_k*:lspg:full:fd`.  
  Suggested wording: “The tested POD-LSPG recurrence empirically remains close to the instantaneous POD projection floor and preserves the kinematic energy estimate closely; this experiment does not establish that it is exactly Newmark on `VᵀLV`.”

- **Reading 3 — “conserves energy” is estimator-dependent.**  
  Sentences: “**The linear ROM inherits the FOM’s conservation**” and “POD-LSPG duly sits at its projection floor with `E_T/E_0 = 1.0000`.”  
  The latter is correct only for the kinematic estimator and full-FD POD arm. The README itself calls the dynamic estimator the fair long-horizon diagnostic. Dynamic POD ratios are `1.00914, 1.00806, 1.00667, 1.00538, 1.00460`, with max dynamic drift around `0.014–0.018` and velocity-reconstruction defect `0.143–0.180`; that is not conservation to four digits.  
  Suggested wording: “Full-FD POD-LSPG preserves the kinematic energy estimate to roughly `3e-7–3e-6` max drift, while the dynamic estimate differs at the 0.5–0.9% final-energy level.”

- **Reading 3 — the cross-PDE conclusion is unsupported.**  
  Sentence: “**the nonlinear-manifold advantage is contingent on the PDE damping the error the ROM injects.**”  
  This general conclusion rests on one Wave seed, one inherited architecture, one budget per K, and an external Burgers comparison not present in the supplied results. Damping is confounded with PDE, data family, decoder quality, training, and time integration.  
  Suggested wording: “These Wave results contrast with the cited Burgers round and motivate testing whether dissipation affects latent-step error accumulation.”

- **Energy interpretation elsewhere is stronger than the data allow.**  
  Sentences: “The coordinate ROM dissipates 61–84% of the energy” and “the best arm is the one that fights the numerical dissipation.”  
  For the recipe, the kinematic estimator decays, but the designated fair dynamic estimator grows to `50.67× / 14.54× / 28.56×` at K=4/8/16, with velocity defects near one. For `weakl64`, the dynamic ratios are `477.5× / 57.8× / 845.0×`. The robust conclusion is severe energy inconsistency, not specifically dissipation.  
  Suggested wording: “The two velocity reconstructions give contradictory energy trends; both diagnose a large failure of energy consistency.”

- **Reading 4 — supported only as a narrow N=64/K=8 observation.**  
  Sentences: “**The hyper-reduction half of the recipe is nevertheless validated on wave**” and “That part is problem-independent and carries forward.”  
  At K=8,N=64, `eq256` differs from full weak64 by 1.89%, meshfree differs from grid by 0.092%, and runtime is 5.52× lower. That supports this tested cell. N=128 is unavailable, the m=1024 fit residual is misstated, and one PDE cannot establish problem independence.  
  Suggested wording: “At K=8,N=64, the tested 256-node grid and meshfree EQ rules reproduce the full weak64 error within 2% at about one-fifth the per-step runtime.”

- **Reading 5 — result supported; stated cause is not.**  
  Sentence: “the objective’s minimiser is not the true parameter — proven by starting LM at the truth and watching it walk away to an 18% lower objective on 16/16 trajectories.”  
  Finding a lower objective on every trajectory does establish that the truth is not a global minimizer of the implemented objective. It does not establish that LM found “the minimiser”; rows terminate by budget or `lambda_max`. Nor do the JSONs measure the asserted smoothness/lower amplitude or show that the cause is specifically an “undamped, broadband solution.”  
  Suggested wording: “Oracle initialization finds points with a mean objective ratio of 0.817 and lower objective than the truth on 16/16 trajectories, proving that the truth is not a minimizer of the implemented objective; the physical cause remains a hypothesis.”

- **Stage 1 causal language is unsupported.**  
  Sentences: “Two things cause this, and both are wave-specific” and “a smoother, lower-amplitude field is always more PDE-consistent.”  
  The JSON provides `resid_at_true_z_rel=0.005584` and optimization outcomes, but no latent-space objective-variation, amplitude, or smoothness statistics. “Always” and “wave-specific” are not tested. Also, `both.per_time_mean` does not grow monotonically: the IC_W=1 curve has eight downward transitions.  
  Suggested wording: “The true latent has nonzero residual and the optimizer finds lower-objective, inaccurate latents; lower amplitude/smoothness and lack of damping are candidate explanations.”

- **Reading 6 — N=64 rollout timing is supported, but the explanation is misstated.**  
  Sentence: “one f64 FiLM evaluation with its Gauss-Newton Jacobian at 256 quadrature nodes costs 25x that.”  
  The 25× ratio describes an entire latent step averaging about five iterations, not one evaluation. “No coordinate ROM pays” is supported at N=64 for rollout-only timing; the broader “these resolutions” claim depends on unavailable N=128 data. The K=16 IC solve also contradicts the stated 78–97 ms range.  
  Suggested wording: “At N=64, the K=8 EQ rollout step costs 2.443 ms—25.4× one FOM substep—and averages 5.19 warm iterations; its rollout-only speedup is 0.158×.”

- **Published comparison — hedge is adequate initially, but later inference overreaches.**  
  The sentence “These cannot be reproduced or contradicted by this experiment” is appropriately cautious. The later claim that a `4.7e-3` result is “therefore reporting on a substantially narrower problem” and the proposed list of causes are speculative: different metric, rank, basis construction, damping, horizon, or other setup differences could explain it. “structural and transfers” and “a statement about the basis, not about the rollout” are also not established by this experiment.  
  Suggested wording: “Because the cited setup is unavailable here and this experiment’s rank-64 test projection floor is 0.0820, the two reported errors are not directly comparable; the source of the difference cannot be identified from these files.”

- **“What would have to change” — hypotheses, not demonstrated requirements.**  
  The opening caveat says these ideas are untested, but the title and ranking still imply necessity. The `weakl64` correlation does not demonstrate that an energy constraint would help, especially given its enormous dynamic-energy growth. A rollout loss is untested, and the increasing ratio across three separately trained K models does not show that decoder accuracy is the weakest remedy.  
  Suggested wording: “Candidates to test next are an energy-aware integrator, rollout-aware training, and higher decoder accuracy; these results do not rank their likely effectiveness.”

## MISSING CAVEATS

- The N=128 table and every N=128 conclusion come from an unfinished live log and have no auditable JSON.
- The central energy conclusion uses the kinematic estimator even though the README calls the dynamic estimator fairer. The actual coordinate dynamic ratios—up to `845×`—and order-one velocity defects should be disclosed next to the kinematic table.
- “Oracle floor” is a budgeted nonlinear latent inference (`floor_budget=60`; step diagnostic uses 80), not a certified projection minimum. Optimization error can inflate it.
- Strong LSPG and weighted weak LSPG are least-squares normal-equation schemes, not automatically the standard Galerkin/Newmark reduction on `VᵀLV`.
- The step diagnostic uses only eight trajectories with four correlated start times each. Its refitted oracle endpoint and difference-of-errors “excess” do not isolate injection from amplification.
- The `dt` sweep is one K, one coordinate variant, one trained decoder, three time steps, and no asymptotic study; the existing caveat mentions only K=8.
- No confidence intervals or seed-to-seed variability are reported. The 16 test trajectories and 32 correlated diagnostic cases are presented only through point estimates.
- K=4/8/16 use separately trained models. The changing ROM/floor ratio cannot be attributed solely to latent dimension or manifold quality.
- Timing speedups are rollout-only and hardware-specific; the K=16 IC-fit anomaly (`~0.84 s`) is undisclosed.
- V7/V8 numbers and several external comparisons have no corresponding fields in the supplied raw result files.

## VERIFIED

- The complete eight-row verification RS table matches `V5_samedt_newmark_fom`, including all means, maxima, latent-step counts, and `n_verify=6`.
- V1, V2, V2b, V3, FOM drift, and RS=80 Newmark drift match their JSON values: `2.78066e-8`, `2.09e-16/1.67e-16/2.12e-16/2.596e-4`, `2.937e-17`, `0.3095926` with `3.586e-16` relative difference and 3844 modes, `1.607e-12`, and `2.259e-9`.
- The complete Stage 1 table matches the three Stage 1 JSONs to the shown precision. The oracle-init arm has mean objective ratio `0.8172405`, `z_err_mean=0.255864`, and `n_found_below_true=16`.
- Auto-decoder training errors, inferred-latent errors, IC-fit errors, recipe errors, best-variant errors, and kinematic energies for K=4/8/16 all match their JSON fields.
- The actual ROM/oracle ratios are 3.27187, 5.11067, and 6.14839.
- The POD floor row uses the TEST floors, not VAL: `0.4282484 / 0.3417185 / 0.2174194 / 0.1381301 / 0.0820103`. The actual full-FD ratios are `1.001725 / 1.001921 / 1.006125 / 1.019299 / 1.021556`.
- Every K=8 knob-table error cell is numerically correct: M sweep, weighted/unweighted, `weakl`, weak/FD, meshfree/grid, EQ/full, random, and off-grid.
- Every step-diagnostic table cell matches. The H=1 excess/oracle ratio is 4.46%, hold/excess is 10.96×, and the fitted log-log exponent is 1.688.
- The RS=8/20/40 table matches: floors `0.0138759 / 0.00238600 / 0.000480157`, ROM errors `0.842776 / 0.878308 / 0.906299`; the floor falls 28.90× and the ROM error rises 7.54%.
- Every N=64 timing-table row and listed speedup matches the K=8 JSON to the stated precision. Decode times are approximately 10.0–10.7 ms.
- Every N=64 value in the “Flat in N” table matches the K=8 JSON. The N=128 column is unverifiable.
- All 93 main N=64 variants—48 coordinate and 45 POD—have `n_blowup=0`, `n_total=16`, and `n_completed=16`. All eight RS-arm variants also have 0 blow-ups and 16/16 completion; all step-diagnostic H cells have 0/32 blow-ups.
- For the headline LSPG recipe, `iters_warm_mean` is 4.936 / 5.192 / 5.375 at K=4/8/16, supporting “about five.” Its cold-step counts are 3.563 / 4.125 / 4.000, supporting “about four.”
