# Corrected smoke readiness: full gradients at the FE boundary

Status: pre-execution blocker identified by independent source inspection on2026-10-27.
Neither dimension has been submitted; no numerical result or empirical claim is retracted.
The earlier scoped code READY verdicts are superseded until this repair is reviewed.

Pinned NGSolve6.2.2608 registers VectorH1's full Grad differential operator for volume
entities. Plain Integrate with a boundary region iterates boundary transformations; direct
full-gradient evaluation there can return zero or fail instead of giving the adjacent-fluid
trace. This affects the new3D preflight and pre-existing surface stress/force/balance
expressions in both dimensions. It is a runtime/operator-interface defect, not a change to
the continuum PDE or its pressure convention.

Verified supported repair: `BoundaryFromVolumeCF(Grad(uh))` and the corresponding divergence
trace before forming boundary stresses or balance products. Keep normals on the boundary
outside the wrapper. Ordinary continuous velocity/pressure traces remain boundary values.
Do not wrap trial/test functions or the entire traction expression containing a normal.
The wrapper's pinned implementation maps boundary coordinates into the adjacent volume
cell and evaluates the enclosed coefficient function there.

Primary source evidence:
- Export and intended stress use: https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/python_comp_mesh.cpp#L1096
- Adjacent-volume mapping: https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/meshaccess.cpp#L2411
- Mapped-point adapter: https://github.com/NGSolve/ngsolve/blob/v6.2.2608/fem/coefficient.hpp#L535
- Boundary integration overload: https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/python_comp.cpp#L3714

The first runtime fixture must verify affine velocity u=x, full gradient I, divergence equal
to dimension, and each face's known symmetric traction2νn. A net integral over the closed
boundary alone could incorrectly pass with all gradients zero. Both authors are implementing
this exact path and preserving previous unsubmitted staging. Thresholds/equations unchanged;
new source manifests and focused independent diff review precede any numerical submission.
