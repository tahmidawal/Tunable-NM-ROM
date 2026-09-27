# Harsh review of paper_latex/main.tex — six reviewer agents, consolidated (2026-09-23)

Six read-only reviewer agents reviewed `paper_latex/main.tex` as it stood on 2026-09-23. Each took one angle: ICLR significance, numerical methods, experimental rigour, internal consistency, numbers against run records, and clarity plus the old NeurIPS reviews. No file was edited. Line numbers refer to `main.tex` at that state. The ICLR reviewer scored the paper **3 (reject), confidence 4**.

"Verified" means the coordinator re-checked the item against the .tex or the run JSON. Every other item is the reviewers' claim and has not been checked.

## A. Integrity-level (fix before anything else)

1. **The Burgers accurate setting at 1024²–4096² passes only the post-hoc certificate.** (Verified.)
   - `bk4096-summary.json` shows that the paper's row (0.22 % at 8.78×) is `selection/accurate_if_certificates_ignored`, i.e. `R384_lin_M1536`. Its certificate is "not confirmed (3/5 draws)".
   - The lane report labels this arm "Sensitivity (NOT the pre-registered rule)".
   - Under the pre-registered rule the accurate arm is `R384_q256`, at 0.54 % and 4.76×. That arm uses C_q, which is excluded from the paper.
   - §3.3 (l. 2150–2153) does disclose that the check was changed after the results were seen. But l. 2149 "Every rule behind Table 1 is confirmed" and l. 2502 "Every quadrature rule in these rows is confirmed" hold only under the revised check. They should say so.
2. **The held-out Burgers result will probably be much worse than the development result.** (Verified in the lane report, partial run.)
   - The superseded partial `bkh64` run overlapped the disk-full window, so no number from it can be used. On 64 held-out cases it gave R′=384 span 1.35 % worst, against 0.224 % on the 6 development cases.
   - The `bkh64b` rerun is still running.
3. **Main text and appendix disagree about the operators' training data.** (Verified.)
   - l. 2529 says "on the same data, split, optimiser, seed and wall budget".
   - The appendix (after l. 3150) says the operators saw \nParOpTraj trajectories and 640 states, against \nParRomTraj trajectories and 131k states for our model (reviewers: 128 vs 4608).
   - l. 3107 also says the operators were trained on Burgers 2D only, yet Table 2 contains heat operators.
4. **"Fast" means three different settings.** (Verified.)
   - Table 1 fast is span R′=128, except at Burgers 512² where it is `head, R'=512, M=64` (l. 3362).
   - Tables 2 and 3 and l. 2597/2628 use "our fast setting" for the head with k=16 (1.89 % at 256²; 2.29 % at 1024²).
   - Table 1 at 1024² gives fast = 1.83 % at 2.30×. The "matched latent dimension k=16" claim in the abstract therefore compares the head, not the Table 1 fast setting.
5. **"Earlier default" is undefined** (l. 2416, the Table 1 caption). Nothing in the paper defines it. (Verified.)
6. **The same cells show different numbers in Table 1 and Table 2.** This is the same failure that sank the NeurIPS submission (reviewer fxe8).
   - Heat 1024²: 105× vs 87.4×. Heat 2048²: 216× vs 289×. Both rows use the same 16 sealed cases, run in different jobs.
   - Burgers 1024² accurate: 0.89× vs 0.22×, attributed to the "general solver path".
7. **The "sealed, opened once" heat cohort was opened at least twice** (Table 1 job plus the Table 2 jobs; the lane summaries say "second opening").
8. **Placeholders remain:** `\todo` at l. 2504 and l. 2591; `\gen` at l. 2681 and l. 3566.

## B. Speedup fairness (every reviewer raised these)

9. **Fast-column speedups are divided by a FOM matched to the *accurate* setting.**
   - Burgers 4096² fast is 1.9 % against a FOM at 0.050 %. Against the fastest FOM at ≤1.9 % (`lean_nt1e-2_dt01`, 1.69 %) it is **11.2×, not 21.7×** (provenance reviewer).
   - At 1024², `tab:ops1024` already shows the same head at 1.06× against the 1.66 % FOM, but 2.02× in Tables 2 and 3.
   - The abstract and intro quote 21.7× without qualification.
10. **Linear problems are compared against *unpreconditioned* CG** (l. 3321).
    - The tests are sine eigenvectors of K. The scaled Poisson residual equals P(ũ−u*), and heat is exact modal propagation.
    - So the method uses spectral structure while its baseline does not. This was dropped from the paper by decision, but reviewers will raise it (NeurIPS fxe8 and GwrW already did).
11. **Burgers at equal *physical* accuracy.**
    - In the knob table (l. 3590–3597), against the refined reference, FOM Newton 10⁻² at Δt=0.01 reaches 6.81 % in 9.9 ms, and the ROM reaches 6.70 % in 41.6 ms (verified).
    - Same-grid matching at 0.2 % prices accuracy that the discretisation (about 6 %) cannot deliver.
12. **Offline cost** (data, training, ordering, PKG assembly, EQ fitting) is not reported, and there is no break-even query count. l. 2351 promises the appendix has it; the appendix says unrecorded costs are not inferred.
13. **Timing:** medians without dispersion; GPU type varies between cells; some rows are provisional. Provisional flags are applied inconsistently: Poisson 2048²/4096² fail the same neighbour gate as the flagged rows, and heat Table 2 passed only under an amended gate.
14. **The query cost is not "independent of N"** (l. 1936, 2136, 1880). Query times grow 55–76× across meshes for Poisson 2D and heat 3D. The Burgers time *falls* with the mesh because the solver path and the exact first step change between meshes.

## C. Novelty and story

15. **The ordering is POD of the bank-projected training fields** (QR + SVD). A POD basis is nested by construction. Reviewers will call the tunability contribution "POD truncation of a learned basis".
    - The needed ablation: the span of the learned bank against plain snapshot-POD at equal R′, on held-out data.
16. **The nonlinear head does no headline work outside Navier–Stokes.**
    - Every Poisson, heat and Burgers Table 1 setting is a span (51 of 52 non-NS settings).
    - NS uses a moving-frame POD bank, not the coordinate network. So the coordinate bank and the head never produce a headline number together.
    - The abstract, contributions and conclusion ("three pieces") present them as one integrated architecture.
17. **NS "36× more accurate at the same number of unknowns"** (l. 2510): the head with k=8 costs about the same as the R′=64 span (0.62 %), so the equal-cost ratio is about 4×.
    - NS also contradicts §3: "coordinate network, Dirichlet exact", while NS is periodic with a POD bank.
18. **The "operators give one operating point" framing is a straw man.** The paper itself concedes resolution and inference knobs at l. 1907 and l. 3111.
19. **Weak baselines:**
    - DeepONet at 32.5–32.8 %.
    - Kim et al. NM-LSPG at 145–175 % (worse than predicting zero; encoder capped).
    - POD-LSPG with no hyper-reduction, capped at k=16, where the fair comparison is POD-LSPG with EQ at R′=384.
    - Operators with one seed and no 2D training budget recorded.
    - 3D operators trained but not reported.
20. **The operator Pareto claim** "from 1024²" rests on 1 Burgers cell (6 development cases) and 2 heat cells.

## D. Consistency and presentation

21. **The claim that speedup rises with the mesh "in every series"** (l. 2519, conclusion) is false. Heat 3D goes 10.9 → 8.50 → 5.79 → 7.60; L-shape goes 47.1 → 39.4; Burgers accurate goes 0.15 → 0.12.
22. **"The span wins at every width"** (l. 2658) is false. Poisson R′=32/64 gives 9.17/5.33 %, heat R′=16 gives 8.68 %, and both are worse than the head. Write "at the widths used".
23. **Intro "up to 106×/216×"** are not maxima: Table 1 has 358× and 410×.
24. **The Burgers held-out cohort is listed as sealed/opened in the appendix** (l. 3258, 3472) while it is still pending in the text.
25. **Leftover appendix rows:** Wave 2D (l. 3262, 3285); old heat k=8/R=32 (l. 3260); "original" Poisson k=16/R=128 (l. 3259); a 3D-operator budget table with no results, and no entry for the 2D operators.
26. **Heat 3D R=320 bank** has no training record (the appendix lists only the "earlier bank, k=32, R=128").
27. **Symbol clashes:**
    - N vs n (l. 2185, 2208).
    - R = residual / bank width / QR factor.
    - T = rotation / transpose / horizon.
    - S = speedup / singular values / support.
    - λ = damping / eigenvalue; K is −Δ_h, not "the Laplacian".
    - `M>k` should also require M ≥ R′ for the span.
    - "Training fixes the tests" (l. 2313) contradicts "deployment chooses M" (l. 2315).
    - The head arrow in Fig. 1 goes to the *ordered* bank, but the head uses the unordered G.
28. **Job ids, GPU types, "development cases" and "reproduction gate on the third attempt"** appear in main-text captions (l. 2545–2550, 2604–2606).
29. **Table 3 bolds 6.79/7.72,** where bold means "more accurate and faster", but those cells have no times.
30. **The tolerance-knob example** (6.712 → 6.701 %) is measured where discretisation error dominates, so it shows nothing about accuracy.
31. **AI-sounding prose:**
    - "Here the head earns its place" (l. 2507).
    - "This is what makes the NM-ROM tunable…" (l. 2662).
    - "Three pieces had to come together" (l. 2724), the third First/Second/Third triplet.
    - "which kills the wall-clock benefit" (l. 2086).
    - "the lever neural operators do not expose" (l. 2715).
32. **Redundancy:** the abstract numbers are repeated in the intro and again in the conclusion; "one frozen model / no retraining" appears 8 or more times.
33. **Burgers is labelled "Hyperbolic"** but is viscous (ν ∈ [0.01, 0.1]).

## NeurIPS complaints now worse

- Numbers inconsistent between tables (fxe8).
- Direct and spectral solver evidence dropped (GwrW).
- Whether a nonlinear manifold is needed (5mgh): now answered "mostly not".
- Novelty: the tunability knob is now POD truncation (5mgh).
- The SMA baseline replaced by another broken reproduction (Kim et al.).
- No seeds or variance; no code link.

## Checked and fine (provenance reviewer)

- Every Table 1, Table 2 and Table 3 value matches its source JSON after rounding.
- Every speedup divides two times from the same job.
- Every FOM is the fastest tested setting at least as accurate as the accurate setting.
- No reported number comes from the 10:30–11:10 disk-full window.
- Hand-typed prose numbers match their sources: 1.9 %/21.7×, 10.7 %/43.2×, 1.08 %/512×, 0.083 %, 7.60×, 2.29 %/2.02×, 6.79–7.72 %, 145–175 %.
