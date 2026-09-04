# j4 (job 2827874) — CANCELLED by user decision at 01:15:13

Poisson N=512 capacity-push refinement (K16 R=128 n_ff=256 150k steps;
K24 R=192 n_ff=256 150k steps; K16 ff_scale=8 probe). Cancelled mid-training
of the FIRST cell (108k/150k steps, train rel-MSE 1.75e-5 and still falling —
vs 5.4e-5 at the j1 K16 cell's 100k-step end, so the capacity push was
working). No result JSON or checkpoint was ever written (first save lands
after training), so out/ was empty; only these logs exist. No numbers from
this job are reportable. The N=512 arm was wrapped when the project
redirected to a focused N=256 push; j3 (Burgers refinement) was never
submitted.
