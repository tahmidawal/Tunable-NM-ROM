**BLOCK — the mathematics supports a bounded algebraic pilot, but the design needs the following fixes before launch.**

The weighted residual and Jacobian agree with [p1core.py:121](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-burgers-multilevel/experiments/jcp-dims/p1core.py:121). For fixed fine history, both \(f(Pz)\) and \(f(x+Pe)\) are valid restricted optimization problems. Their gradients and damped Gauss–Newton matrices are correct. No FAS tau term is needed for this exact pullback. Prefix injection is consistent with the shared rotated-bank construction.

Required fixes:

1. **Correct the “same-width … no-op” control.** With \(P=I\), coarse optimization performs additional fine-space optimization; it generally changes the state. Require equivalence to an uninterrupted LM solve only when damping, stopping, iteration budget and evaluation sequence match. Keep zero-budget/zero-correction bypass as the actual no-op control. ([DESIGN.md:21](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-burgers-multilevel/experiments/burgers-multilevel/DESIGN.md:21))

2. **Freeze the complete intervention state machine.** Specify:
   - whether the fine pre-step consumes one of the 600 iterations;
   - coarse initial damping and whether coarse damping affects the fine post-solve;
   - damping after rejected interventions;
   - behavior when the predictor or pre-step already passes fine convergence;
   - whether correction occurs once per timestep or repeats.

   `p1core.py` carries damping between timesteps; resetting it during segmentation changes the baseline. Rejected correction should restore the complete fine solver state, while retaining its incurred cost. ([p1core.py:193](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-burgers-multilevel/experiments/jcp-dims/p1core.py:193))

3. **Make target-agreement gates executable and unambiguous.** Define the field norm, spatial weights, comparison anchor and time population for every threshold. Require the maximum over **all 50 evolved steps**, cases and repetitions, rather than only six decoded outputs. Tight and tighter anchors must each pass fresh stationarity certification. Record objective agreement as well as field agreement; tight/tighter agreement establishes a reproducible local anchor, not a unique/global minimizer. Freeze numerical tolerances for Jacobian, transfer and gradient controls, plus the known failure required from the zero-budget negative control. ([DESIGN.md:21](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-burgers-multilevel/experiments/burgers-multilevel/DESIGN.md:21))

4. **Pin the actual target and acceptance population before submission.** Record checkpoint/rotation hashes, ordered test modes, quadrature arrays, cold-start rule, numeric trust radius, exact dev6 descriptors/seeds and source hashes. Explicitly preserve CNR’s **two full BE startup steps**, as implemented. Coarse construction must retain all 1024 fine residual rows and fine history terms; using a conventional width-128 setting that reduces \(M\) would change the proposed objective. ([p1core.py:41](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-burgers-multilevel/experiments/jcp-dims/p1core.py:41))

Fairness provisions are otherwise sound: retain the strongest predictor baseline, count rejected work and transfers, certify the timed invocation, compare within one allocation, and preserve every failed case. Freeze the outlier definition and report paired per-case timing ratios and repetition variability; three repetitions support pilot evidence only.

The analytic-advection/discrete-diffusion target is acceptable **for the explicitly limited archived-objective acceleration diagnostic**. It does not establish physical accuracy, geometric multigrid behavior, or superiority over POD/FOM. A finer-quadrature check becomes necessary if claims extend to the integrated weak objective.

No files changed, log appended, or jobs launched.