Frozen copies of the validated building blocks this experiment imports (so concurrent
edits in the sibling worktrees cannot change the code under a running experiment, and
so the cluster staging ships exactly what ran locally):

| file | source worktree / commit |
|---|---|
| burgers2d-rom-latent-stepping/blat_common.py | exp/2026-08-16-burgers2d-rom-latent-stepping @ c3d8968 |
| burgers2d-coord-rom/burgers2d_film.py | exp/2026-08-14-burgers2d-coord-rom @ b327915 (imported by blat_common only) |
| multistage-precision/ms_parametric.py, ms_autodecoder.py | exp/2026-08-14-multistage-precision @ b389467 |
| wave2d-coord-rom/wave2d_film.py | exp/2026-08-14-wave2d-coord-rom @ 3c2e078 (the FOM + sweep decoder) |
