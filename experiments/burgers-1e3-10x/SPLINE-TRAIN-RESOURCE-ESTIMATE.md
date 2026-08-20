# Phase-2 spline trainer resource estimate

Status: planning-only and prospective; written while S0 job 2667808 remained pending,
before any S0 promotion outcome was available.  It licenses no training cell.

The pulled H200 D0 artifact at commit `048e83f` regenerated 576 N=64 trajectories
at all 51 times and completed its data plus representation diagnostics in
21.8588900566 seconds.  Counting target-grid work in N=64-equivalent trajectories,
the locked trainer's train plus selection mix is

```
512 + 4*128 + 16*64 + 64 + 4*32 + 16*16 = 2,496,
2,496 / 576 = 4.3333.
```

Straight N-squared scaling of the complete D0 elapsed time is 94.7 seconds.  This
is a conservative data-generation planning estimate because D0 also performed two
large representation fits.  The trainer regenerates only the locked indices, not
all 704 draws at every mesh.

Raw retained f64 fields occupy 3.423 GB for training (0.856 GB each at N=64 and
N=128, 1.711 GB at N=256) and 0.749 GB for selection (0.107, 0.214, and 0.428 GB).
Coordinates, affine states, features, autolatents, schedules, optimizer states, and
one selection prediction/difference workspace keep the expected live host footprint
well below 16 GB.  A 64 GB request leaves at least a fourfold margin for Python/JAX,
reference-generation temporaries, serialization, and allocator fragmentation.

Two excluded GB10 execution smokes used arm A with the exact scientific update
shapes: manifold/predictor batches 32x512 and each oracle batch 64x512.  The two-
update and 20-update timings were respectively:

| block | 2 updates | 20 updates | positive incremental seconds/update |
|---|---:|---:|---:|
| manifold | 10.7171 s | 9.1597 s | compile noise; use cross-block maximum |
| predictor | 4.4603 s | 4.7119 s | 0.0140 s |
| oracle start 0 | 11.2990 s | 11.4411 s | 0.0079 s |
| oracle start 1 | 3.1794 s | 3.4184 s | 0.0133 s |
| oracle start 2 | 3.1691 s | 3.3754 s | 0.0115 s |

Using 0.0140 seconds for every one of the locked 80,000 updates gives 18.7 minutes
on the GB10; doubling this for the largest R=48 coefficient head gives 37.4 minutes.
The 12 possible full selection-oracle checkpoints total 1.122 billion sparse
16-support point queries, or 143.6 equivalent GFLOP under the preregistered
128-operation screen.  A further 30-minute allowance covers these decodes,
compilation, checkpointing, and audit serialization.  Combined with the 1.6-minute
mixed-data estimate, the planning total is about 70 minutes before margin.

Every licensed arm/seed cell therefore requests one H200, 8 CPUs, 64 GB host memory,
and 04:00:00 on `gpu`.  This is more than three times the conservative planning
total.  The launcher fixes BLAS thread counts to one, uses the cluster venv, and
performs a mandatory GPU-backend preflight.  No timing here is a scientific speed
comparison or an online gate.
