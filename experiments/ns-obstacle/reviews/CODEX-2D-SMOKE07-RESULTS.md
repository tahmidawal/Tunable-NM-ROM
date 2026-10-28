# Independent 2D smoke07 result review

The independent read-only numerical reviewer accepts the frozen G1 controls for source
`262a0ef0546bdbecb1507fd3a48d22c10b137067`, job 6612209. This accepts initial
implementation controls only; no G2 reference, training or ROM admission follows.

The reviewer verified all collected regular files, archive chunks and executed source
inputs, GPU/f64/highest preflights, the pending-job resource-only update, actual returned
normal coefficients, geometry moments, Newton and linear histories, resolved directional
Jacobian checks, analytic manufactured forcing and force signs. Independent NumPy replay
of saved volume and surface arrays reproduces the physical diagnostics at roundoff.
The channel requires no Newton updates; both obstacle cases perform genuine Newton
iterations. The original failed attempts remain retained.

The result is bound to archive SHA256
`962e5cf5ac28760bf4edde5659aaeab4a8cd638c8152df4a1afccd2cd8ce4a80`,
retention commit `a893210f5` and mirror `98b803bd1`. The owner subsequently recorded
checksum-backed remote-duplicate cleanup at `c3a6c8a76`; the coordinator independently
confirmed the exact job directory absent. This closes the operational condition in the
reviewer's verdict. CPU FE phase times are diagnostic and support no speed comparison.

The coarse DFG solution still fails the stricter divergence, lift, pressure-drop and
gradient-versus-symmetric force-consistency criteria. Its force-definition differences
are not explained by increasing the surface quadrature order. Manufactured analytic
pressure and sampled analytic trace also remain distinct from the future prescribed-FE
trace and refinement checks. The manufactured inlet-only mass ratio is inapplicable
because its inlet flux is nearly zero; total-boundary flux is the appropriate control.

Before reference/data admission: complete curved-mesh clean-process reload, native
producer/export checks, spatial and polynomial refinement, tolerance and quadrature
tightening, domain/outlet sensitivity and conditional uncertainty assessment. Preserve
all force definitions and exact pole evaluations. No threshold was relaxed and no 3D
repair is approved by this result. Quantitative tables are generated from retained JSONs
by the coordinator reporting scripts, not transcribed here.
