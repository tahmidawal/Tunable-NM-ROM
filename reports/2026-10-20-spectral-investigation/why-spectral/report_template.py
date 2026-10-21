"""Source text of report.md. Placeholders «key» are filled by make_report.py from values.py; «REVIEW_RECORD» from
review_record.txt. Edit this file, never the generated report.md."""

TEMPLATE = r'''# Why a coarse sine spectral solver beats our reduced model on the Dirichlet box

**Status (2026-10-20; analysis of saved data plus CPU reruns; nothing launched on the cluster; reviewed, see section 6).** This report explains, with numbers, why the campaign's kill criterion K2 fired in 2D and 3D: a sine pseudo-spectral full-order solver with $N=16$–$48$ modes per axis matches our reduced-order model's (ROM's) accuracy faster (published union speed-ups: 2D «k2_2d», 3D «k2_3d»; a value below 1 means the full solver is faster). It also predicts when that would stop. Every number is computed by `values.py` from this folder's analysis JSONs or from lane outputs, and `make_report.py` fills it in; `VALUES.json` gives the source of each.

- **Measured here.**
  - The sine spectra of the second-order E0 and jcp-regimes references (CPU, read-only).
  - The spectral solver's error against those references, for each case and each $N$. These are CPU reruns of E0's own solver; they reproduce the H200 in-job scores to «gate2d_maxrel» in 2D and «gate3d_maxrel» in 3D.
  - The solver's boundedness on steep fronts, in sampled step brackets.
  - Bank-versus-sine floors on the second-order references.
  - One synthetic experiment.
- **Taken as published from the lanes** (H200, f64, `jax_backend=gpu`): every wall time and every ROM error.
- **Modelled, and labelled so where it appears:** operation counts, the cost model, and the crossover predictions. The predictions extrapolate beyond the measured $N$ and steepness.
- **Labels that travel with the numbers.**
  - All errors are conditional E0 estimates (Richardson; E0 A10).
  - The 3D speed3d configurations are not quadrature-qualified (jcp-qualify).
  - The regime ROM errors on C1 and C2 come from unqualified configurations.
  - The local GB10 run is used only for a ratio; its outputs were checked against CPU to «g1_cpu_dev».

## 0. In brief

- **Why the spectral solver wins (measured): it needs few modes, and each mode is very cheap.**
  - *The references are smooth after the first output time.* Their sine coefficients decay approximately exponentially over the fitted range. A 1 % best-approximation floor needs a median $N=$ «E0_2d_Nprojp01_med» per axis in 2D (max «E0_2d_Nprojp01_max») and «E0_3d_Nprojp01_med» in 3D (max «E0_3d_Nprojp01_max»).
  - *Few modes reach the ROM's accuracy.*
    - 2D: at the 20-step setting that the cost comparison charges ($\Delta t=0.0125$), the solver itself reaches every qualified ROM's median *and* worst error at $N=$ «Njointc_min»–«Njointc_max». That is at most «Njointc_max_modes» modes. With a small step the range is $N=$ «Njoint_min»–«Njoint_max».
    - 3D: at 10 steps ($\Delta t=0.025$) it reaches every selected ROM's errors at $N=$ «N3c_joint»; with a small step, at $N=$ «N3_joint» («N3_modes» modes).
  - *A ROM unknown costs far more than a spectral mode.*
    - Work: «rom_flopunk_R384» flops per unknown per step at $r=384$, against «psflopunk_2d_48» per mode at $N=48$.
    - Hardware: the spectral step runs at the GPU's launch-latency floor («ps2_tau_48» per step, «ps_rate_48» TFLOP/s).
    - Steps: the spectral solver takes 20 explicit steps; the ROM takes 50 implicit ones at $\Delta t_0$.
- **What the learned bank buys (measured).**
  - In range, plain sines in Laplacian order match the $r=384$ bank floor with a median «bank_e0_384_med» $r$ (max «bank_e0_384_max»). This reproduces the ideation's first-order figure («bank_idea_384_med»).
  - The ratio grows with steepness (Spearman «sp_bank_s»).
  - But on the steep cohorts the bank reaches a 3 % floor within 512 functions on only «c12_bank512_reach3» of «c12_n» cases.
- **Cost model.** It is fitted to measured times and lies within «model_check_lo»–«model_check_hi» of every checked query.
  - Per step, a spectral solver costs $\tau_0+W_{PS}/P$, with $W_{PS}\propto N^d\log N$.
  - A dense hyper-reduced ROM costs $\tau_0'+W_R/P'$, with $W_R\propto r^3$.
  - The fitted ROM intercept, $\tau_0'=$ «rom_tau0» per step, is above the spectral one («ps2_tau0» in 2D). It is a property of the present implementation, fitted on $r=128$–$384$; it is not a hardware bound.
- **Predicted crossover (model scenarios for the present implementation, extrapolated).**
  - *Free unknowns.* If its unknowns cost nothing but each step still cost the fitted intercept, a 50-step ROM would be 2× faster than a 20-step spectral solver only when the spectral solver needs $N\approx$ «Nc_c2_50_2» in 2D, or $N\approx$ «Nc_c3_50_2» in 3D.
  - *Real unknowns.* A real ROM also needs $\rho=(N-1)^d/r\gtrsim10^3$ there (table in section 4). Our ROMs have $\rho\approx$ «rho_ours_lo»–«rho_ours_hi».
  - *Steep fronts.* They raise the spectral solver's step count (empirical step law $\Delta t\approx$ «c_dt»$/s$, from sampled boundedness brackets). With the ROM still taking a small, fixed number of steps, the free-unknown and $\rho=1000$ scenarios cross around $s\approx$ «scross_fixed_lo»–«scross_fixed_hi»; some $\rho=100$ scenarios also cross.
  - *Equal steps.* If the ROM needs as many steps as the spectral solver, only free-unknown scenarios cross over; «scross_eq_rho_none».
- **The principle.**
  - On a box with smooth, diffusive solutions, the sine basis was about as compact as our learned bank on the median case (more compact at $r=384$), and it is nearly free to apply.
  - In the model scenarios we scanned (section 4), crossovers came from three ingredients:
    - a solution family with far fewer degrees of freedom than its spatial resolution needs, so that $\rho$ reaches $10^2$–$10^3$ (on our synthetic families: sharp features at fixed positions, not freely moving ones);
    - a spectral solver forced to large $N$ or many explicit steps (steep fronts);
    - a ROM whose step count stays small. With equal step counts, only scenarios with free unknowns crossed over.
  - With dense reduced operators, the per-unknown cost grows like $r^2$ and the total like $r^3$. At fixed $N$, step counts and model constants, increasing $r$ only increases the ROM's run time.
  - These are scenario results for this model and implementation, not proven necessary conditions. In our Burgers cohorts, $\rho\approx$ «rho_ours_lo»–«rho_ours_hi» and the spectral solver needs few modes and steps.

## 1. Spectral convergence on our data

**Method.**
- *References.* E0's second-order fields; for the steep cohorts, jcp-regimes' fields, made by the same procedure.
- *Best $N$-mode error.* On the scoring lattice, the orthonormal DST-I is the discrete sine series, so the best $N$-mode error is a tail sum (Parseval):
$$\mathrm{tail}_k(N)=\frac{\lVert u(t_k)-P_Nu(t_k)\rVert}{\lVert u(t_0)\rVert},\qquad N_{proj}(\varepsilon)=\min\{N:\max_{k\ge1}\mathrm{tail}_k(N)\le\varepsilon\},$$
where $P_N$ keeps modes $1..N-1$ per axis, the solver's convention.
- *Decay rate.* $\sigma$ comes from a least-squares fit of $\log\mathrm{tail}(N)\approx c-\sigma N$ over the range $10^{-6}\le\mathrm{tail}\le10^{-2}$ (`a1_spectra.py`).
- *Lattice limit.* $N$ is at most 256 in 2D and 64 in 3D.

| cohort | cases | $s=a/\nu$ | $N_{proj}(1\%)$ median / max | $N_{proj}(0.1\%)$ median / max | $\sigma$ at $t=0.05$ | $\sigma$ at $t=0.25$ |
|---|---|---|---|---|---|---|
| E0 2D (dev6 + val32) | «E0_2d_n» | «E0_2d_smin»–«E0_2d_smax» | «E0_2d_Nprojp01_med» / «E0_2d_Nprojp01_max» | «E0_2d_Nprojp001_med» / «E0_2d_Nprojp001_max» | «E0_2d_sigma_t1_med» | «E0_2d_sigma_t5_med» |
| regimes N0 (in range) | «reg_N0_n» | «reg_N0_smin»–«reg_N0_smax» | «reg_N0_Nprojp01_med» / «reg_N0_Nprojp01_max» | «reg_N0_Nprojp001_med» / «reg_N0_Nprojp001_max» | «reg_N0_sigma_t1_med» | «reg_N0_sigma_t5_med» |
| regimes A− ($\nu$ below range) | «reg_Am_n» | «reg_Am_smin»–«reg_Am_smax» | «reg_Am_Nprojp01_med» / «reg_Am_Nprojp01_max» | «reg_Am_Nprojp001_med» / «reg_Am_Nprojp001_max» | «reg_Am_sigma_t1_med» | «reg_Am_sigma_t5_med» |
| regimes C1 (steeper) | «reg_C1_n» | «reg_C1_smin»–«reg_C1_smax» | «reg_C1_Nprojp01_med» / «reg_C1_Nprojp01_max» | «reg_C1_Nprojp001_med» / «reg_C1_Nprojp001_max» | «reg_C1_sigma_t1_med» | «reg_C1_sigma_t5_med» |
| regimes C2 (steepest) | «reg_C2_n» | «reg_C2_smin»–«reg_C2_smax» | «reg_C2_Nprojp01_med» / «reg_C2_Nprojp01_max» | «reg_C2_Nprojp001_med» / «reg_C2_Nprojp001_max» | «reg_C2_sigma_t1_med» | «reg_C2_sigma_t5_med» |
| E0 3D (dev4 + val64) | «E0_3d_n» | «E0_3d_smin»–«E0_3d_smax» | «E0_3d_Nprojp01_med» / «E0_3d_Nprojp01_max» | «E0_3d_Nprojp001_med» / «E0_3d_Nprojp001_max» | «E0_3d_sigma_t1_med» | «E0_3d_sigma_t5_med» |

In 3D, $s=A/\nu$ uses the peak amplitude $A$. The C2 numbers at 0.1 % sit close to the 2D lattice limit, so they describe the lattice more than the continuum.

![sine tails](figs/fig1_sine_tails.png)

*Figure 1.* Best $N$-mode sine error of six references at $t=0$, $0.05$ and $0.25$ (dotted line: 1 %).

**Against time (measured).**
- *At $t=0$, the 2D initial field is not smooth in the sine sense.* The Gaussian does not vanish on the walls, but the wall nodes are set to zero (E0 DESIGN §2).
  - The tail therefore decays only algebraically. The median slope of $\log\mathrm{tail}$ against $\log N$ between $N=16$ and $128$ is «E0_2d_t0_slope».
  - The fitted exponential rate is «E0_2d_sigma_t0_med».
- *By the first output time, diffusion has removed this.* The decay is approximately exponential over the fitted range. The median $N$ for a 1 % tail is «E0_2d_Nt1_med» at $t=0.05$ and «E0_2d_Nt5_med» at $t=0.25$ (worst cases «E0_2d_Nt1_max» and «E0_2d_Nt5_max»).
- *So the wall incompatibility costs the spectral solver almost nothing,* because the metric scores only $t\ge0.05$. Its one recorded trace: on two 2D cases, E0 observed first-order spectral self-convergence, with self-differences of order $10^{-9}$ (E0 DESIGN A10).

**Against viscosity and steepness (measured; 2D, «fit_Nprojp01_n» cases from E0 and the four regime cohorts).**
- *Steepness controls the required resolution (in these cohorts, at the fixed horizon $T=0.25$).*
  - Spearman correlations of $N_{proj}(1\%)$: «sp_Nprojp01_s» with $s$, «sp_Nprojp01_nu» with $\nu$, «sp_Nprojp01_a» with $a$, and «sp_Nprojp01_w» with the bump width $w$.
  - A multiple regression gives exponents «mr_Nprojp01_a» on $a$ and «mr_Nprojp01_nu» on $\nu$. These are equal and opposite, so the variable is $s$. The exponent on $w$ is «mr_Nprojp01_w».
- *Power laws:* $N_{proj}(1\%)\approx$ «fit_Nprojp01_c»$\,s^{«fit_Nprojp01_alpha»}$ and $N_{proj}(0.1\%)\approx$ «fit_Nprojp001_c»$\,s^{«fit_Nprojp001_alpha»}$.
- *Decay rate:* $\sigma$ falls with steepness (Spearman of $\sigma(t=0.25)$ with $s$: «sp_sigma_s»).
- *Front width:* with the measured front width $\delta=\max|u|/\max|\nabla u|$, the product $N_{proj}(1\%)\cdot\min_t\delta$ has median «Ndelta_med» (interquartile range «Ndelta_iqr»). The sharpest front sets the resolution.

*Hypothesis (not tested here).*
- The exponent is below 1, whereas the textbook viscous shock gives $N\propto s$. A possible reason: by $t=0.25$ most bumps may not have steepened to the equilibrium width $\sim\nu/a$.
- The analyticity-strip argument (Boyd 2001, Thm. 5; Sulem, Sulem & Frisch 1983) predicts $\sigma\propto\nu/a$ for a fully formed front. We did not check whether the fronts are fully formed.

![N versus steepness](figs/fig2_N_vs_steepness.png)

*Figure 2.* Left: $N$ for a 1 % projection floor against $s$, all cohorts (2D fit dashed). Right: smallest $N$ at which E0's spectral solver itself reaches 1 % and 0.1 % error against the reference.

**The spectral solver's own error (measured; CPU; small steps, $\Delta t=0.0025$ in 2D and $0.01$ in 3D, chosen so that the time error is small; at the charged steps of 20 and 10 the matching $N$ below is unchanged).**

| $N$ (2D) | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| median | «ps2_N16_med» | «ps2_N24_med» | «ps2_N32_med» | «ps2_N48_med» | «ps2_N64_med» | «ps2_N96_med» | «ps2_N128_med» |
| worst | «ps2_N16_worst» | «ps2_N24_worst» | «ps2_N32_worst» | «ps2_N48_worst» | «ps2_N64_worst» | «ps2_N96_worst» | «ps2_N128_worst» |

| $N$ (3D) | 8 | 10 | 12 | 14 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| median | «ps3_N8_med» | «ps3_N10_med» | «ps3_N12_med» | «ps3_N14_med» | «ps3_N16_med» | «ps3_N24_med» | «ps3_N32_med» |
| worst | «ps3_N8_worst» | «ps3_N10_worst» | «ps3_N12_worst» | «ps3_N14_worst» | «ps3_N16_worst» | «ps3_N24_worst» | «ps3_N32_worst» |

The plateau near «ps2_N96_med» is consistent with a reference-limited floor (E0's $\epsilon_{ref}$ is of order $10^{-6}$); it was not decomposed further.

**Which $N$ reaches our ROM's accuracy (measured).**
- *ROM errors:* jcp-qualify's FRONT-2D per-case values at $1024^2$, for all «Njoint_n» qualified configurations.
- *$N_{joint}$:* the smallest $N$ whose spectral median *and* worst both tie or beat the ROM's (the campaign's matching rule).
- *$N^*$:* the per-case smallest $N$ with $e_{PS}\le e_{ROM}$.
- *Two step sizes.* "small step" uses $\Delta t=0.0025$ (spatial error only). "20 steps" uses the $\Delta t=0.0125$ that the cost comparison charges, so accuracy and cost refer to the same configuration.
- *Result:* over all configurations, $N_{joint}=$ «Njoint_min»–«Njoint_max» at the small step and «Njointc_min»–«Njointc_max» at 20 steps. The per-case median $N^*$ (small step) is «Ncase_med_lo»–«Ncase_med_hi».

| configuration ($1024^2$; r, form, scheme, step factor) | ROM median | ROM worst | $N_{joint}$, small step | $N_{joint}$, 20 steps | modes $(N_{joint}-1)^2$, small step | $N^*$ median | $N^*$ max |
|---|---|---|---|---|---|---|---|
«tab_match2d»

- The worst case sets the joint match.
  - For $r=384$ LSPG-CN-R, the per-case $N^*$ correlates with steepness (Spearman «sp_Ncase_s»).
  - The hardest case is «Ncase_worst_case» ($s=$ «Ncase_worst_s»).
- In 3D, every selected speed3d ROM is matched at $N=$ «N3_joint» with a small step ($\Delta t=0.01$), below the lanes' smallest tested $N=16$, and at $N=$ «N3c_joint» at 10 steps ($\Delta t=0.025$). The ROMs are scored at $L=384$, with median and worst over val64.

| 3D ROM | median | worst | $N_{joint}$, small step | $N_{joint}$, 10 steps | modes, small step |
|---|---|---|---|---|---|
«tab_match3d»

**Steep regimes (measured; CPU; small steps: $\Delta t=0.0025$ for N0 and A−, $0.002$ for C1, $0.001$ for C2).** Both methods are scored on the same cases: those the regimes lane marks eligible and valid for the ROM.

| regime | cases | $s$ | ROM median / worst | $N_{joint}$ | $N^*(1\%)$ median / max | $N^*(0.1\%)$ median / max |
|---|---|---|---|---|---|---|
«tab_reg»

- *Columns.* "ROM" is $r=384$ LSPG-CN-R on that regime (jcp-regimes per-case errors; C1 and C2 are not qualified). $N_{joint}$ is the smallest $N$ matching that median and worst jointly. $N^*$ is the per-case smallest $N$ reaching 1 % or 0.1 %.
- *Fit over E0 2D and the regimes, at the step the steepness model charges* (per case $\Delta t=\min(0.0125,\,0.05/\lceil0.05\,s/c\rceil)$, section 4): the spectral solver's own $N^*(1\%)\approx$ «fitPS_c»$\,s^{«fitPS_alpha»}$ («fitPS_n» resolved cases).

## 2. Cost per unit accuracy

**Per step: measured times and modelled work.**
- *Spectral step.* Four ETDRK4 stages. Each stage uses $T=7$ (2D) or $T=12$ (3D) sine/cosine transforms on the 3/2-padded grid, so $W_{PS}=4T(M-1)^{d-1}\cdot2.5(2M)\log_2(2M)$ with $M=3N/2$.
- *ROM step.* Dominated by $n_J=$ «rom_nj» assemblies per step (one initial Jacobian plus one per Levenberg–Marquardt iteration; the mean iteration count is borrowed from one $r=384$ CN-R component run and applied to all widths) of the $M\times r$ Jacobian $\Psi^\top[\operatorname{diag}(Dc)B+\operatorname{diag}(Bc)D]$ from $m$ quadrature points, so $W_R=n_J\cdot2mMr$, with $M=4r$ and $m/r=$ «kappa_lo»–«kappa_hi».

| | spectral 2D $N=24$ | spectral 2D $N=48$ | spectral 2D $N=256$ | ROM 2D $r=128$ | ROM 2D $r=256$ | ROM 2D $r=384$ |
|---|---|---|---|---|---|---|
| work per step (flop) | «psW_2d_24» | «psW_2d_48» | «psW_2d_256» | «rom_W_R128» | «rom_W_R256» | «rom_W_R384» |
| flop per unknown per step | «psflopunk_2d_24» | «psflopunk_2d_48» | «psflopunk_2d_256» | «rom_flopunk_R128» | «rom_flopunk_R256» | «rom_flopunk_R384» |
| measured time per step | «ps2_tau_24» | «ps2_tau_48» | «ps2_tau_256» | «rom_tau_R128» | «rom_tau_R256» | «rom_tau_R384» |
| dominant-term-equivalent rate (TFLOP/s, model flop / measured time) | – | «ps_rate_48» | «ps_rate_256» | «rom_rate_R128» | «rom_rate_R256» | «rom_rate_R384» |
| largest admissible step | «ps2_dtmax_24» | «ps2_dtmax_48» | «ps2_dtmax_256» | 0.005 ($\Delta t_0$; implicit) | 0.005 | 0.005 |
| f64 table footprint read per Jacobian (MB, estimate, not measured traffic) | – | – | – | «rom_bytesJ_R128» | «rom_bytesJ_R256» | «rom_bytesJ_R384» |
| per-query overhead, lattice output | «ps2_alpha_24» | «ps2_alpha_48» | «ps2_alpha_256» | init «rom_init_R128» + decode «rom_dec256_R128» | init «rom_init_R256» + decode «rom_dec256_R256» | init «rom_init_R384» + decode «rom_dec256_R384» |
| per-query overhead, $4096^2$ output | «ps2_alpha4k_24» | «ps2_alpha4k_48» | «ps2_alpha4k_256» | decode «rom_dec4k_R128» | decode «rom_dec4k_R256» | decode «rom_dec4k_R384» |

*Sources.*
- Spectral time per step and overhead: a regression of jcp-fom2d's per-configuration costs on the number of steps (cost $=\alpha+n\tau$).
- Admissible step: fom2d's, over 38 cases.
- ROM: jcp-p1-scaling's evolution time over 50 steps, plus its init and decode times. At $1024^2$ the evolution takes «rom_ev_R128» at $r=128$ and «rom_ev_R384» at $r=384$.

**Query decomposition (measured).**
- *2D, $4096^2$ full field (jcp-fom2d stage 2, one job).*
  - The $r=384$ LSPG-CN-R query takes «q4_R384_rom»; its matched spectral comparator takes «q4_R384_fom» ($S=$ «q4_R384_S»).
  - For $r=128$ BE: «q4_R128_rom» against «q4_R128_fom» ($S=$ «q4_R128_S»).
- *Where the time goes (jcp-p1-scaling component profile, a separate job; its components need not sum to the stage-2 total).*
  - ROM at $4096^2$: init «rom_init_R384», evolution «rom_ev_R384» (measured at $1024^2$; evolution is mesh-independent), decode «rom_dec4k_R384». Evolution is «ev_share_R384» of the $r=384$ CN-R profile and «ev_share_R128» of the $r=128$ BE profile.
  - Spectral solver: 20 steps of «ps2_tau_48» each. The rest is set-up and output synthesis: «ps2_alpha_48» on the lattice, «ps2_alpha4k_48» at $4096^2$.
- *3D (speed3d, mesh-free path).*
  - W_C_W768 takes «r3_W_C_W768_fused» = project «r3_W_C_W768_proj» + evolve «r3_W_C_W768_evolve» + decode «r3_W_C_W768_dec». That is «r3_tau» per step over 15 steps.
  - The 3D spectral step costs «ps3_tau_16», «ps3_tau_32» and «ps3_tau_64» at $N=16$, $32$ and $64$, with overheads «ps3_alpha_16», «ps3_alpha_32» and «ps3_alpha_64».

**Why each ROM unknown costs so much more than a spectral mode (ratios measured; mechanism partly modelled).**
1. *Dense coupling.*
   - Every reduced unknown is coupled to every test function through every quadrature point, so forming the Jacobian costs $2mMr=2\kappa\cdot4\,r^3$ flop ($\kappa=$ «kappa_384» at $r=384$).
   - A sine mode is coupled to the others only through FFTs, at $O(\log N)$ cost per mode.
   - Result: «rom_flopunk_R384» against «psflopunk_2d_48» flop per unknown per step, a factor of «flop_ratio».
2. *Hardware regime.*
   - The ROM's dense kernels run at a dominant-term-equivalent «rom_rate_R384» TFLOP/s at $r=384$ (a large fraction of the H200's f64 throughput).
   - The spectral step is latency-bound: «ps_rate_48» TFLOP/s at $N=48$. Its time per step stays nearly flat, «ps2_tau_16»–«ps2_tau_96», from $N=16$ to $96$.
   - So time per unknown differs by less than the flop count: «rom_perunk_R384» µs against «ps_perunk_48» µs per unknown per step. Going from $N=48$ to $N=256$ multiplies the modes by «ps_modes_ratio_256_48» but the time per step only by «ps_tau_ratio_256_48».
3. *Implicit nonlinear solve.*
   - The ROM takes a Levenberg–Marquardt step: one Jacobian at the start and one per iteration («rom_nj» on average in the measured run), and one normal-matrix Cholesky solve per iteration.
   - The spectral solver integrates diffusion exactly (diagonal $\varphi$-functions) and advection explicitly, with no Jacobian.
4. *Fewer, higher-order steps.*
   - ETDRK4 is fourth order and stable at $\Delta t=0.0125$ in range (20 steps).
   - Our LSPG schemes take 50 steps at $\Delta t_0$, and their time error already dominates (jcp-budget; final report).
5. *Memory traffic.*
   - One ROM Jacobian reads tables with a footprint of about «rom_bytesJ_R384» MB of f64 at $r=384$ (an estimate, not measured device traffic).
   - A padded $N=48$ spectral field is «ps_bytes48_kB» kB and stays in cache.
6. *Full-field output.*
   - The ROM decodes to $4096^2$ by reading the stored bank on the output mesh («rom_dec4k_R384»).
   - The spectral solver synthesises its output by FFT. Its whole $4096^2$ overhead, set-up included, is «ps2_alpha4k_48».

![per-step cost](figs/fig3_step_cost.png)

*Figure 3.* Measured time per step against modelled work per step (H200); lines are $\tau_0+W/P$ fits. The ROM is the more hardware-efficient code. It loses because it does $10^3$–$10^4$ times more work per step.

## 3. Representation: how much does the learned bank buy?

**On the references (measured; `a3_bank.py`).**
- *Bases.* The deployed 2D bank, against sines in Laplacian order.
- *Floors.* Orthogonal projections on the scoring lattice, maximised over $t_1..t_5$.
- *$K_{match}/r$.* The smallest sine prefix that reaches the bank's floor at width $r$, divided by $r$. Below 1, the sines are more compressive.

| cohort | cases | $s$ | bank floor $r=384$ (median) | $K_{match}/r$ at $r=128$: median (max) | at $r=384$ | at $r=512$ |
|---|---|---|---|---|---|---|
«tab_bank»

- *In range, sines beat the bank on the median case:* «bank_e0_384_med» $r$ sines match the $r=384$ floor. The bank wins on the worst cases, by up to «bank_e0_384_max»×.
- *At a fixed 1 % floor,* the sine-to-bank size ratio has median «rho_e0_1e2_med» (max «rho_e0_1e2_max»; «rho_e0_1e2_n» cases).
- *The bank's advantage grows with steepness,* with Spearman «sp_bank_s» between $K_{match}/r$ at $r=384$ and $s$.
- *But on C1 and C2 the bank floor itself is several per cent.*
  - Only «c12_bank512_reach3» of «c12_n» cases reach a 3 % floor within 512 functions; sines need a median «c12_Klap3_med» modes for the same floor.
  - So the ratio is large exactly where the bank is too inaccurate to use.

![bank versus sines](figs/fig4_bank_vs_sines.png)

*Figure 4.* Sines needed to match the $r=384$ bank floor, divided by 384, against steepness.

**When would a learned basis need far fewer coefficients? (Synthetic, `a5_synthetic.py`; not Burgers.)**
- *Families.* Four parametric families on the same lattice.
- *Learned basis.* POD of 800 training fields: the standard data-driven linear basis (optimal in mean square on the training set, not for the held-out worst case), a stand-in for a learned linear bank.
- *Sines.* Square blocks (the spectral solver's set) and the best anisotropic block $(N_x-1)(N_y-1)$ (a direction-aware spectral solver).
- *Entries.* POD rank / square-block sine modes / anisotropic-block sine modes needed for a 1 % worst error over 200 held-out fields, with $\rho$ against each sine set in brackets. >400" means POD did not reach 1 % within rank 400, the reliability cutoff we chose for 800 snapshots (censored: no $\rho$ is inferred).

| family | «syn_deltas» |
|«syn_delim»|
«tab_syn»

![synthetic compression](figs/fig5_synthetic_compression.png)

*Figure 5.* $\rho$ against feature width for the four families: solid against square sine blocks, dashed against the best anisotropic block. A missing point means the POD rank exceeded 400.

**Conditions (observed on these sampled families and tested ranks only; consistent with the n-width literature).**
- *A linear learned basis won big only for the fixed-position family* (F4: a bump at a fixed centre with varying width). There $\rho$ was «syn_F4_rho_lo»–«syn_F4_rho_hi» against square blocks and «syn_F4_an_lo»–«syn_F4_an_hi» against the best anisotropic block, growing as $\delta$ shrinks.
- *The one-direction moving front (F1) looked like a large win against square blocks* ($\rho$ up to «syn_F1_rho_hi»), *but not against a direction-aware sine block* ($\rho=$ «syn_F1_an_lo»–«syn_F1_an_hi»). Its apparent advantage came from isotropic blocks wasting modes in the direction the front does not vary.
- *It won little or nothing when the sharp feature translated freely* (F2, F3).
  - Where POD reached 1 %, $\rho=$ «syn_F3_rho_rng» for F3. At sharper $\delta$ POD needed more than 400 modes, beyond the rank cutoff we chose; we infer nothing there, and a sample-size convergence study would be needed to say more.
  - This is consistent with the Kolmogorov-width barrier of transport (Ohlberger & Rave 2016; Arbes, Greif & Urban 2025).
- *Our Burgers cohorts resemble F3 more than F1 or F4.* The bump centre varies in 2 (or 3) directions, along with width, amplitude and viscosity, and the measured in-range $\rho$ is of order 1.
- *Where a learned basis can pay off.*
  - Features at *known* positions (boundary layers, fixed obstacles) can need orders of magnitude fewer coefficients (F4). Low-dimensional motion alone was not enough in F1 once the sine set was direction-aware.
  - For free motion, the literature's routes are nonlinear or transport-aware decoders (registration, shifted POD, autoencoders); we did not test them.

## 4. The general principle and a cost model

**Model (lattice output, mesh-free contract; a description of the present implementations, not of the method class).**
$$T_{PS}(N)=n_{PS}\Big(\tau_0+\frac{W_{PS}(N)}{P}\Big)+\alpha_{PS}(N),\qquad T_{ROM}(r)=n_{ROM}\Big(\tau_0'+\frac{n_J\,2\kappa r\cdot4r\cdot r}{P'}\Big)+\alpha_{ROM},$$
with $\kappa=m/r$ fixed at «kappa_384» (the $r=384$ rule) in every prediction.

- *Fitted constants* (H200, relative least squares):
  - spectral 2D: $\tau_0=$ «ps2_tau0», $P=$ «ps2_P» TFLOP/s (largest relative residual «ps2_resid»);
  - spectral 3D: $\tau_0=$ «ps3_tau0», $P=$ «ps3_P» TFLOP/s (residual «ps3_resid»);
  - ROM 2D: $\tau_0'=$ «rom_tau0», $P'=$ «rom_P» TFLOP/s (residual «rom_resid»), from six runs at only three widths ($r=128$, 256, 384). The intercept $\tau_0'$ is an extrapolation to zero work of this implementation (history terms, predictor, Levenberg–Marquardt control flow, kernel launches). A different solver, quadrature or kernel design could lower it; nothing here bounds it from below.
  - ROM 3D: the 2D intercept, plus a cubic-in-$r$ term anchored on the one measured 3D configuration (W_C_W768, quasi-Newton with Jacobian reuse). No 3D width sweep exists, so this is the weakest part of the model.
- *Overhead $\alpha_{PS}(N)$:* interpolated from the measured overheads, and held at the largest measured value beyond them (a growth variant is in the sensitivity table).
- *Second throughput variant:* $P$ taken from the largest measured $N$, giving «ps2a_P» (2D) and «ps3a_P» (3D) TFLOP/s. The two fits are scenarios, not bounds.
- *Shape check on another GPU* (local GB10; absolute times not comparable). Above the latency floor, the spectral step scales with $W_{PS}$, as the model assumes:
  - 2D: $\tau(1024)/\tau(256)=$ «g1_ratio_2d», against a work ratio of «g1_Wratio_2d»;
  - 3D: $\tau(128)/\tau(64)=$ «g1_ratio_3d», against «g1_Wratio_3d».
- *Spectral step count.* In range, $n_{PS}=20$ (2D) or 10 (3D); section 1 gives the accuracy reached at those steps.
- *Stability on steep fronts* (measured in 2D on N0, C1 and C2, at $N=64$, 128, 256 and $\Delta t\in\{0.0125, 0.00625, 0.003125\}$; "stable" = finite with $\max|u|\le1.5\max|u_0|$, which is boundedness, not accuracy).
  - N0: all «stab_N0_n» cases are stable at 0.0125.
  - C1: «stab_C1_n0125» of «stab_C1_n» cases are.
  - C2: «stab_C2_n0125» of «stab_C2_n» cases are, and «stab_C2_nnone» C2 cases have no stable tested step at some $N$.
  - The largest stable tested step is a lower end of a bracket, not an exact limit. Over the cases whose bracket closes inside the tested range, $\Delta t_{max}\cdot s$ has minimum $c=$ «c_dt» and median «c_dt_med»; the model uses $\Delta t_{max}=c/s$ with the minimum; the median enters only a cost-only sensitivity.
  - Cases with no stable tested step at some $N$ only give upper limits: «stab_excluded». These do not show whether their $c$ lies above or below «c_dt»; they are unresolved.
  - The law is an empirical 2D description of these cohorts at $T=0.25$. Its use in 3D is an assumption. Rescaling time by the amplitude shows that $aT$ is a second parameter; it varies across cases but is not independently controlled here.

**Check against measured query times** (model / measured; in-sample: the fits used these same runs, except the PS3 and ROM3 anchors, which are reproduced by construction):

| query | model (as used in predictions, $\kappa$ fixed) | model with the configuration's own $m/r$ | measured | ratio (first model) |
|---|---|---|---|---|
«tab_check»

With $N_{joint}$ at 20 steps, the model gives lattice-contract speed-ups of «predS_R384» ($r=384$ CN-R) and «predS_R128» ($r=128$ BE). Both lie inside the published K2 range.

**Scenario 1: unknowns at zero cost.**
- The ROM time is then $n_{ROM}\tau_0'+\alpha_{ROM}$, the fitted intercept of this implementation.
- The table gives the smallest $N$ at which the spectral solver becomes $S\times$ slower than that. Ranges span the two throughput fits.

| dim | spectral steps | ROM steps | bar | ROM time at zero work | crossover $N$ |
|---|---|---|---|---|---|
«tab_cross»

*Sensitivity of the 2× zero-cost crossover $N$* to the weakest constants: ROM intercept halved; spectral throughput doubled; spectral overhead growing like $(N-1)^d$ beyond the measured $N$.

| dim | spectral steps | ROM steps | base | $\tau_0'/2$ | $2P$ | growing $\alpha_{PS}$ |
|---|---|---|---|---|---|---|
«tab_sens»

**Scenario 2: compression needed** for a real ROM with $r=(N-1)^d/\rho$ unknowns. Model values from both throughput fits; a single value means the two agree.

| dim | $N$ | sine modes | spectral steps | ROM steps | $T_{PS}$ | $\rho$ for $S=1$ | $\rho$ for $S=2$ | largest $r$ for $S=1$ |
|---|---|---|---|---|---|---|---|---|
«tab_rho»

**Where our configurations sit,** with $N_{joint}$ at 20 steps (2D) or 10 steps (3D) and $\rho=(N_{joint}-1)^d/r$:

| configuration | $N_{joint}$ | modes | $r$ | $\rho$ |
|---|---|---|---|---|
«tab_ours»

![required compression](figs/fig6_required_compression.png)

*Figure 6.* Compression $\rho$ needed for $S=1$ (solid) and $S=2$ (dashed), against the spectral resolution required (model, present implementations). A missing curve means no $r>0$ suffices: the ROM's fitted per-step intercept alone is too slow. Stars mark our ROMs.

**Scenario 3: crossover in steepness (model; extrapolated beyond the measured $s$).**
- *Accuracy relation.* The fit $N^*_{PS}(\varepsilon)=c\,s^\alpha$ over the cases that reach $\varepsilon$. In 2D it is measured at exactly the step the model charges (per case $\Delta t=\min(0.0125,\,0.05/\lceil0.05\,s/c\rceil)$, aligned to the output interval; `a2_ps.py stab`). In 3D it is measured at $\Delta t=0.025$ on in-range cases only, so 3D accuracy at larger $s$ is unmeasured.
  - 2D, 1 %: «sfit_d2_eps0.01_c»$\,s^{«sfit_d2_eps0.01_alpha»}$ («sfit_d2_eps0.01_n» cases, «sfit_d2_eps0.01_cens» censored, largest resolved $s=$ «sfit_d2_eps0.01_smax»). 0.1 %: «sfit_d2_eps0.001_c»$\,s^{«sfit_d2_eps0.001_alpha»}$ («sfit_d2_eps0.001_cens» censored).
  - 3D, 1 %: «sfit_d3_eps0.01_c»$\,s^{«sfit_d3_eps0.01_alpha»}$ («sfit_d3_eps0.01_cens» censored, in-range only up to $s=$ «sfit_d3_eps0.01_smax»: weak). 0.1 %: «sfit_d3_eps0.001_cens» censored.
  - The fit is a typical case. The "envelope" rows multiply it by the largest ratio $N_i/\mathrm{fit}(s_i)$ over *resolved* cases only; censored cases (no tested $N$ reached $\varepsilon$) are not covered, so this is not a cohort-wide worst case: «sfit_d2_eps0.01_env» (2D, 1 %), «sfit_d2_eps0.001_env» (2D, 0.1 %), «sfit_d3_eps0.01_env» (3D, 1 %), «sfit_d3_eps0.001_env» (3D, 0.1 %).
- *Steps.* $n_{PS}=\max(n_0,\,5\lceil0.05\,s/c\rceil)$ (whole steps per output interval). In 2D the accuracy at those steps is measured up to the regimes' largest $s$; beyond it, both are extrapolated.
- *ROM.* Three assumptions about its step count; the unknowns cost as in the model with compression $\rho$.
- *Table.* The first $s$ at which the spectral solver is 2× slower (with $c=$ «c_dt»).
- *Cost-only sensitivity.* With $c=$ «c_dt_med», fewer spectral steps are charged and the fixed-step crossovers move to $s\approx$ «scross_fixed_med_lo»–«scross_fixed_med_hi». The accuracy and stability at those larger steps were not measured, and Codex's round-3 check found one sampled C2 case non-finite at such a step (`codex-review-3.md`). Treat this range as unvalidated.

| dim | $\varepsilon$ | $N(s)$ | ROM step count | $\rho=\infty$ | $\rho=1000$ | $\rho=100$ | $\rho=5$ |
|---|---|---|---|---|---|---|---|
«tab_scross»

**Reading: the regime boundary (scenario results for these implementations).**
1. *Smooth, moderately steep problems on a box.*
   - The deployed ROMs lose in range (measured, K2), and their steep-regime configurations fail qualification.
   - Our ROMs have $\rho\approx$ «rho_ours_lo»–«rho_ours_hi». Even at zero work, a step of this implementation costs $\tau_0'=$ «rom_tau0», against a spectral step floor of «ps2_tau0» (2D) or «ps3_tau0» (3D). The spectral solver also takes 20 steps against our 50 in 2D, and 10 against 15 in 3D.
   - Below the zero-cost crossover $N$ of scenario 1, changing $r$, the quadrature or the bank alone cannot reverse this in the model. Lowering the per-step intercept and the step count can (sensitivity table).
2. *Resolution.*
   - With zero-cost unknowns, a 50-step ROM is 2× faster only once the spectral solver needs $N\approx$ «Nc_c2_50_2» (2D) or «Nc_c3_50_2» (3D). With 20 ROM steps, $N\approx$ «Nc_c2_20_2» or «Nc_c3_20_2».
   - Higher dimension helps the ROM: spectral work grows like $N^d$, while a parametric ROM's need not.
3. *Compression.*
   - At those $N$, a real ROM needs $\rho$ of order «rho_need2» (2D) or «rho_need3» (3D) for $S=2$ (scenario 2).
   - On the sampled synthetic families, $\rho$ that large appeared only for the fixed-position family (F4). It did not appear for freely moving features (F2, F3), which our cohorts resemble, nor for the one-direction front once the sine set is direction-aware (F1).
4. *Steepness.*
   - Steep fronts raise both $N$ and, more strongly, the spectral solver's explicit step count.
   - The model then crosses over at $s\approx$ «scross_fixed_lo»–«scross_fixed_hi» (fit or envelope, $\rho=\infty$ or 1000), *if* the ROM keeps 20–50 steps there. With $\rho=100$ only some rows cross over (table).
   - If the ROM must take as many steps as the spectral solver, the free-unknown crossover moves to $s\approx$ «scross_eq_lo»–«scross_eq_hi»; and «scross_eq_rho_none».
   - Untested: whether a ROM can step across a front much faster than the front's own time scale ($\delta/a\sim\nu/a^2$) at equal accuracy. Our LSPG ROMs are already time-error dominated in range, and they fail qualification on C1 and C2.
5. *Viscosity.* In these cohorts, at the fixed horizon, the resolution depended on $a$ and $\nu$ through $s$ (regression exponents equal and opposite, section 1). This is an empirical statement, not a scaling law: $aT$ is a second parameter that we did not control independently.

**Prediction (conditional).**
- Measured: the deployed linear-span ROMs with dense hyper-reduction lose to the spectral solver everywhere they were compared, and on the steeper-than-training regimes C1 and C2 ($s$ up to «reg_C2_smax») they are not qualified.
- Model, for these implementations (dense reduced operators, cubic work in $r$ throughout):
  1. *Resolution.* With free unknowns, a 2× crossover needs the spectral solver at about $N\approx$ «Nc_c2_20_2» (2D, 20 ROM steps) to «Nc_c2_50_2» (2D, 50 ROM steps), or «Nc_c3_20_2» to «Nc_c3_50_2» in 3D, at 20 (2D) or 10 (3D) spectral steps; or many more explicit spectral steps (steep fronts).
  2. *Compression.* With real unknowns, the crossovers we found needed $\rho$ of order $10^3$; a few rows cross at $\rho=100$, none at $\rho=5$.
  3. *Steps.* Holding the ROM's step count fixed while the spectral one grows makes the steepness crossover far earlier. With equal step counts, only the free-unknown scenarios cross over; «scross_eq_rho_none».
- These are scenario results, not proven necessary conditions for any reduced model. The first is a property of the problem; the second and third are properties the method would have to acquire, and the model is least reliable exactly there (extrapolated $N$, $s$ and ROM intercept).

## 5. Literature

Every source below was checked by a literature subagent. [V] marks sources whose full text was read for the point cited; [A] marks sources checked only by abstract or bibliographic record.

**Comparisons against weak baselines.** In the fluid-related ML-for-PDE papers that claim to beat standard solvers, the baseline is usually weak (McGreivy & Hakim). The projection-ROM speed-ups listed below are quoted against the authors' own full solvers.
- McGreivy & Hakim (2024, *Nature Machine Intelligence* 6, doi:10.1038/s42256-024-00897-5) [V].
  - Of 76 ML-for-PDE papers claiming to beat a standard solver, 60 compared against a weak baseline.
  - They require comparisons at equal accuracy against an efficient solver. For smooth advection-dominated problems, they note that pseudo-spectral methods are usually the state of the art.
  - When they reran 10 papers with a stronger baseline, the baseline beat the ML solver in 7.
- Kochkov et al. (2021, *PNAS*) [A] reported 40–80× against a finite-volume baseline. McGreivy & Hakim report that a pseudo-spectral baseline is over 80× faster than that baseline on GPU.
- Dresdner et al. (2023, arXiv:2207.00556) [A] corrected a spectral solver with ML. Against a spectral baseline at the same resolution, they report 2–4× better accuracy at about 2× the cost.
- Published hyper-reduced Burgers ROMs report speed-ups against their own implicit grid solvers: Kim et al. (2022, *JCP* 451, 110841) [A], 2.6× in 1D and 11.7× in 2D; GNAT (Carlberg et al. 2013, *JCP* 242) [A].
- We found no projection ROM benchmarked against a coarse pseudo-spectral solver at equal accuracy. That is absence of evidence. Our campaign is such a benchmark, and it agrees with McGreivy & Hakim.

**n-width: transport against diffusion.**
- Ohlberger & Rave (2016, arXiv:1511.02021) [V] prove $d_N\ge\tfrac12N^{-1/2}$ for linear advection of a jump.
- Greif & Urban (2019, *Appl. Math. Lett.* 96) [A]: $O(e^{-\beta N})$ for certain parametrized coercive problems, against $O(N^{-1/2})$ for waves with discontinuous data.
- Arbes, Greif & Urban (2025, *Adv. Comput. Math.* 51:13, arXiv:2305.00066) [V] represent the n-width of a linear transport problem exactly through Fourier analysis.
  - Under their assumptions (including a half-wave symmetry of the data), the optimal spaces are built from sorted trigonometric functions. The decay depends on the regularity of the data: specific nonsmooth examples have algebraic rates, and analytic data give exponential decay. We do not extend these results to Burgers.
- Peherstorfer (2022, *Notices AMS* 69(5)) [V] reviews the Kolmogorov barrier.
- Maday, Patera & Turinici (2002, *J. Sci. Comput.* 17) [A] prove exponential convergence for parametric elliptic problems.
- Our synthetic F2/F3 rows are transport-like; F4 is diffusion-like. They illustrate, and do not prove, these results.

**Spectral convergence.**
- Boyd (2001, *Chebyshev and Fourier Spectral Methods*, Dover) [V].
  - Thm. 5: coefficients decay like $e^{-\rho n}$, where $\rho$ is the width of the strip of analyticity.
  - Thm. 4: decay is algebraic when boundary compatibility fails. For a sine series, that means an even derivative that does not vanish at the wall.
  - §11 treats Burgers fronts.
- Sulem, Sulem & Frisch (1983, *JCP* 50) [A] read the strip width from the slope of the spectrum.
- Trefethen (2000, *Spectral Methods in MATLAB*, ch. 4) and Canuto et al. (2006) are standard background, not separately checked.
- ETDRK4: Cox & Matthews (2002, *JCP* 176) [A]; Kassam & Trefethen (2005, *SIAM J. Sci. Comput.* 26) [A].

**Transport-aware and nonlinear reduced models** (routes around the barrier; in the texts and abstracts we checked, we found no wall-clock comparison against a coarse spectral or similarly tuned full solver):
- Lee & Carlberg (2020, *JCP* 404) [V]: autoencoder manifolds, without hyper-reduction, so no speed-up is reported.
- Shifted POD: Reiss et al. (2018, *SIAM J. Sci. Comput.* 40) [A].
- Transport reversal: Rim, Moe & LeVeque (2018, *SIAM/ASA JUQ* 6) [A].
- Registration: Taddei (2020, *SIAM J. Sci. Comput.* 42) [A].
- Black, Schulze & Unger (2020, *ESAIM M2AN* 54) [A].
- Survey: Hesthaven, Peherstorfer & Unger (2026, *Acta Numerica*, arXiv:2602.01397) [A].

**Cost per reduced unknown.**
- Yano & Patera (2019, *CMAME* 344) [V] count $O(N^2K)$ per Newton step for the hyper-reduced residual and Jacobian with $K$ quadrature points, plus a dense $O(N^3)$ solve. A sparse or FFT solver costs $O(1)$ to $O(\log n)$ per unknown.
- DEIM: Chaturantabut & Sorensen (2010, *SIAM J. Sci. Comput.* 32) [A]. Q-DEIM: Drmač & Gugercin (2016) [A]. ECSW: Farhat et al. (2014, 2015) [A].
- Our $W_R=n_J\,2mMr$ is this count, with $K=m$ and $M$ tests.

## 6. Review

«REVIEW_RECORD»

## 7. Files

- `common.py`: paths and read-only loaders. E0 references go through `refstore.py` (sha256-verified); regime references come from the lane store.
- Analyses:
  - `a1_spectra.py`: reference spectra.
  - `a2_ps.py`: spectral-solver accuracy by $N$, with the reproduction gates.
  - `a2b_stab.py`: stability on steep fronts.
  - `a3_bank.py`: bank against sines.
  - `a4_match.py`: the $N$ that matches the ROM.
  - `a5_synthetic.py`: the synthetic families.
  - `a6_cost.py`: measured costs and fits.
  - `a7_model.py`: the cost model, its check and the crossovers.
  - `g1_ps_scaling.py`: the GB10 shape check and its CPU cross-check.
- Outputs: `analysis/*.json`.
- Report build: `values.py`, `make_figs.py`, `report_template.py`, `make_report.py`.
- Running: everything runs on CPU with `JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python <script>`, except the GPU half of G1, which runs under `jaxrun`.

## Glossary

**The two methods**
- **ROM (reduced-order model).** Our method. It writes the solution as a combination $u\approx\sum_{j\le r}c_jG_j$ of $r$ learned functions (the **bank**) and evolves the coefficients $c_j$ by a projected equation.
- **$r$.** The number of bank functions used, which is the number of ROM unknowns.
- **$M$ (test functions).** The number of sine test functions the residual is projected on; here $M=4r$.
- **$m$ (quadrature points).** The number of points where the nonlinear term is evaluated. "gauss128" is a $128^2$ Gauss rule; "fib4181" is a 4181-point Fibonacci lattice.
- **LSPG, GAL.** Least-squares Petrov–Galerkin and Galerkin projection.
- **BE, CN-R, CN.** Backward Euler; Crank–Nicolson with a Rannacher start; Crank–Nicolson.
- **Step factor.** The time step in units of $\Delta t_0=0.005$.
- **LM.** Levenberg–Marquardt, the ROM's nonlinear solver.
- **Spectral FOM (PS).** The full-order sine pseudo-spectral solver. $N$ means it keeps sine modes $1..N-1$ per axis, so it has $(N-1)^d$ unknowns. It integrates diffusion exactly and computes products on a 3/2-padded grid.
- **ETDRK4.** A fourth-order exponential Runge–Kutta time stepper, used by the spectral FOM.

**Accuracy**
- **E0 references.** The project's second-order reference solutions (central differences + Crank–Nicolson), with a Richardson error estimate $\epsilon_{ref}$. "Conditional" means that estimate assumes the asymptotic regime.
- **Error $e$.** $\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$, where $t_k=0.05k$ and the norm is over the scoring lattice: $257^2$ in 2D, $63^3$ in 3D. **Median** and **worst** are taken over a cohort's cases.
- **Floor / tail.** The best error any combination of the given basis functions can reach (an orthogonal projection), in the same metric. $N_{proj}(\varepsilon)$ is the smallest $N$ whose sine floor is at most $\varepsilon$.
- **$\sigma$.** The exponential decay rate of the sine tail as $N$ grows.
- **$s=a/\nu$ (steepness).** Amplitude over viscosity. In 3D it is $A/\nu$, with $A$ the peak amplitude.
- **$\delta$.** Measured front width, $\max|u|/\max|\nabla u|$.
- **Cohorts.** dev6 and val32: 2D validation cases. dev4 and val64: 3D validation cases. N0: fresh in-range cases. A−: viscosity below the training range. C1 and C2: steeper than the training range.
- **$N_{joint}$.** The smallest $N$ whose spectral median and worst errors both tie or beat the ROM's. **$N^*$** is the same test applied to one case.
- **Qualified.** Passed the campaign's numerical checks of solver convergence and quadrature (Q1/Q2).

**Speed**
- **$S$ (speed-up).** Spectral time divided by ROM time at matched accuracy. Below 1, the spectral solver is faster. The **2× bar** is the campaign's registered threshold.
- **K2.** The campaign's kill criterion: "a full-order solver matches the ROM's accuracy faster".
- **Union.** The cheapest matched solver over all solver families.
- **Lattice contract / full field.** Outputs only on the scoring lattice, or on every node of a $4096^2$ mesh.
- **$\tau$, $\alpha$.** Time per step, and per-query overhead (set-up, input, output synthesis or decode).
- **$\tau_0$, $P$.** Latency floor per step and effective throughput, from the fit $\tau=\tau_0+W/P$.
- **$W$.** Modelled floating-point work per step.
- **$n_J$.** Jacobian assemblies per ROM step.
- **$\kappa=m/r$.** Quadrature points per unknown.
- **Crossover.** The problem size or steepness at which the ROM becomes $S\times$ faster than the spectral solver.
- **H200, GB10.** The H200 is the cluster GPU used for every timing. The GB10 is the local GPU, used here only for a ratio.

**Representation**
- **$K_{match}/r$.** The size of the sine prefix that reaches the bank's floor, divided by $r$.
- **$\rho$ (compression).** Sine modes needed divided by reduced unknowns needed, at the same accuracy.
- **POD.** The optimal linear basis computed from snapshots by SVD.
- **n-width (Kolmogorov).** The best worst-case error any $n$-dimensional linear space can achieve on a set of solutions.
'''
