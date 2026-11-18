# B1 recipe 2: shared local POD and physical coefficient cache

The rich direct span passes the original 128-by-128 screening at roughly 2000 coefficients; independent finer scoring is in progress. Learn a shared local rotation from training coefficients only, then measure direct prefixes 4/8/16/24/36. This is an analytical-coordinate/POD feature architecture, not a coordinate-MLP reproduction.

Use tensor side count 6, 52 retained local nodes, depth 4, the initial-callable swept allocation, and fixed per-cell physical whitening. Reduce the global parent to the first 16 training-only spectral POD fields, refitting all root and local coefficients jointly. Record this changed parent as a distinct direct control; it is not the old nonlinear parent. Check the uncompressed span before interpreting compression. Compute per-case physical Gram matrices on 128-by-128 quadrature and preserve coefficient/field parity records. No altered fine PDE physics.

Use the expanded training128/development32 data plus the same twelve alignment cases. Development is never used to learn the local rotation. Preserve original16 and added16 development panels separately in reports. Qualification of the extended references and dense scoring are required for claims. The local POD learns from all training trajectories and snapshots with current-time normalization. Save regularized full coefficients, geometry, transforms and Gram matrices, so nonlinear head fitting can optimize the assembled physical metric rather than arbitrary coefficient error.

Reserve at most 120 GPU-minutes within B1. Neural child training is the next conditional recipe after this control; dimensions and schedule will be registered separately. No final-test access.

## Resolved-metric parent choice before cache submission

The physically resolved metric gives worst 0.228158% with root128 but 0.588005% with root16 on the original panel; actual4096 confirms the root128 witness. Therefore build the primary cache with root128 from the newly expanded common training span, passed explicitly to the driver. Keep root16 as a failed lower-capacity control. The scalar parent change is prospectively selected from this bracket, not a silent change in the variable count. Compression controls must count 128 root variables plus all local entries.
