# Phase-3 P3-D resource estimate

Request: one H200, 8 CPUs, 64 GB host memory, 2 hours, `gpu` partition.

P3-D performs 128 fixed diagnostic fits for each of two solvers.  The observed
Phase-2 unpreconditioned C fits took median 0.580/0.730/0.953 seconds at
N=64/128/256, so even charging one second for every new fit gives about 4.3
minutes serial.  The two alternatives and fits are intentionally serial, with
OMP/OpenBLAS/MKL fixed to one thread; the eight-CPU request matches the existing
audited H200 job envelope and leaves scheduler/host work headroom, not an
assumed sparse-solver speedup.  A conservative 8x solver-overhead allowance is
35 minutes.

The S0 job's complete 17,136-fit plus live-cost panel finished in 26:29 on H200.
P3-D has only 256 fits, four kernel compilations plus untimed identity probes,
and 20 balanced timing repetitions for four trajectories.  The excluded N=32
driver smoke compiled/executed all four kernel routes and both solvers in 15.9
seconds on GB10.  The two-hour request therefore leaves more than 2x margin over
the conservative solver allowance plus tight-reference generation and timing.

Compiled smoke memory was below 34 MB per mandatory kernel.  Phase-2 C at the
shape-faithful N=1024 output used 624.8 MB at worst, so 64 GB host and the H200
device comfortably cover the full output, the 98.6 MB staged S0 artifact, sparse
R=48 normal matrices/factors, and sequential/block-local decoder temporaries.
The scientific `<=20 GB` compiled-device gate remains independent of this
scheduler request and is recomputed for every kernel.
