## What changed, in plain words

The trained model, the reduced basis, the reduced solve and the answer are the same as in the paper.  Only
the way the online query is evaluated changed.

1. **The bank is stored in separated form.**  The paper path keeps the mesh-evaluated bank as a dense
   array with one column per reduced coordinate and one row per mesh node, so writing the output field is a
   product whose cost grows with (mesh nodes) x (reduced size).  That product is 95 % of the paper query at
   4096^2.  The bank is smooth along each mesh direction, so offline it is compressed once into a factor for
   the x direction, a factor for the y direction and a small core (ranks 65 x 66 for the accurate arm,
   50 x 51 for the fast arm, relative truncation 1e-4 in float64).  Online the field is two small
   matrix products, `U (C a) V^T`.
2. **The source is projected through the same factors**, so reading the full nodal source costs two small
   products instead of a product with the dense bank.
3. **The reduced operator is assembled and factorised offline** and the whole query (project, reduced
   solve, core contraction, field products) is one compiled function with one device synchronisation.
4. Heat 2D: the same separated bank; in addition the initial-condition moments are taken with a small sine
   table instead of a full sine transform of the mesh, and the moment fit, the composed Crank-Nicolson
   maps for the six output times and the core are folded offline into one small matrix, so the six output
   fields come from two products.

Offline cost added: one randomised compression of the bank per mesh (6 to 15 s on the GPU; see the
tables), no retraining.  The dense bank is still needed once offline, so offline memory is unchanged
(4096^2 Poisson 2D on A100 80GB needed the host-side variant of the compression, amendment A2).

## What this supports and what it does not

Supports:

- On heat 2D (A100 80GB) the tuned query returns the same six fields as the paper query (deviation at most
  2.1e-7, worst test error identical to six digits) 2.2x to 14.2x faster than the paper path timed in the
  same job, at 1024^2 to 4096^2.
- On Poisson 2D the tuned query returns the same field as the paper query (largest deviation 1.4e-6 of the
  field norm on the accurate arm, 8.8e-6 on the fast arm; worst test error identical to six digits) and is
  faster than the quoted cuDSS reused-factorisation solve at every mesh from 256^2 to 4096^2, on both GPU
  types, with full-field output and the full nodal source as input.
- The query time is now nearly flat in the mesh size on H200 (0.18 ms at 256^2 to 0.45 ms at 4096^2 on the
  accurate arm), because the only mesh-sized work left is reading the source and writing the field.

Does not support:

- cuDSS and AmgX solve an arbitrary right-hand side to solver tolerance.  The reduced model is accurate
  only to its model error (0.84 % worst case accurate arm, 4.06 % fast arm) and only on the trained source
  family.  The comparison is time-to-answer at the reduced model's accuracy, not equivalence.
- The separated evaluation relies on the tensor-product mesh and on the bank being smooth along each
  direction.  The same structure is what makes sine-transform solvers fast on this problem; those were not
  targeted and no claim is made against them.
- Configuration S (source read on a 256-interval sub-lattice) relies on the smoothness of the trained
  source family.  It is a separate column and never the headline.
- "Solve only" (reduced coefficients, no field) is a separate column and never compared with cuDSS.
- The fast-arm deviation (8.8e-6) is close to the 1e-5 development gate; it passed, with little margin.
- 512^2 has no stored paper timing; its "before" is only the in-job paper path.
- At 256^2 to 1024^2 the paper path timed inside this lane's driver is slower and noisier than the stored
  paper values (see "Paper-path timing spread"), so the conservative speed-up at small meshes uses the
  stored paper value as "before".  The cuDSS comparison does not depend on this.
- Heat 2D was measured on A100 80GB only (the H200 job did not start); the stored paper heat timings are
  from H200, so only the in-job paper path is a same-GPU "before".  Heat 2D comparators are the
  sparse-solvers lane's arithmetic estimates for 20 Crank-Nicolson solves, not measured trajectories.
  The paper's heat 2D query already beat those estimates by two orders of magnitude; the new result is
  the before/after gain, not a changed ranking.
- Poisson 3D and heat 3D were not attempted (three-direction compression and new paper-path drivers were
  out of the time budget; both already beat cuDSS by large margins in the paper).

## Failures and cancelled jobs

- A100 80GB job 7220235: the 4096^2 Poisson test failed (compression ran out of device memory); rerun as
  job 7221131 with the host-side compression stage (amendment A2), same frozen pair.
- Jobs 7220222, 7220223 (superseded) and 7221749 (wrong fast heat arm, amendment A3a) were cancelled while
  pending; none started.
- H200 heat 2D job 7221755 never started (queue priority).
- The unmodified paper driver's own timing-stability gate failed on H200 (see above).
- The cause of the paper path's extra variable overhead in this lane's driver at small meshes was not found.
