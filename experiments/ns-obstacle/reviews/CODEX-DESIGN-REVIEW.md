# Independent Codex numerical design review

Reviewer: separate read-only `numerical_reviewer` agent; reviewed draft b0cd5eddc without
consulting Claude or other reviewer responses. Date 2026-10-27. No numerical jobs run.

Verdict: release dependency preparation and basic reference-control implementation. Do not
yet release reference acceptance, training, or performance claims. Core DFG geometry,
mean-speed Reynolds convention, natural-outlet pressure level, normal orientation and
inlet-driven energy accounting are sound. Primary sources independently inspected:
https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
and https://epubs.stfc.ac.uk/manifestation/255/dltr-2001001.pdf .

Findings (draft line references retained):
1. Lines 159,172–178,236–250: freeze exact field/residual/operator/Jacobian denominators,
   whitening, common integration and uncertainty estimator. The preliminary lane's scaled
   RMS convention differs from central relative fields. Keep both labelled where useful.
2. Line 194: neural envelope vanishes on exact circle/sphere but FE boundaries approximate
   them. Separate geometry trace from algebraic Dirichlet error and define evaluation.
   ROM discovery line39 must use `(x-x_in)/(x_out-x_in)`, not `x/L` with negative inlet.
3. Line294: separate G2-steady/G2-transient so steady admission does not pretend temporal
   verification. Add nontrivial manufactured three-component control; pipe Poiseuille alone
   does not exercise convection.
4. Line130: output 0.1 with uniform dt0.04 is incompatible. Align steps or verify interpolation;
   do not silently shorten steps with constant-step BDF2 coefficients.
5. Line216: offer exact quadratic tensors to bank as well as POD. A symmetric steady 3D
   reference needs a qualified axisymmetric competitor before a best-FOM claim, or restrict
   comparison explicitly to the tested full-three-dimensional solver ladder.

Appropriate: sparse direct small-pilot control, modest resource budget, disjoint populations,
same-CPU timing, unsampled POD fallback and staged stops. Record central shorter 3D domain
versus draft longer suggestion. This review approves no numerical result or production gate.
