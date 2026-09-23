# `gen01` (job 4183329) — FAILED after 45 s, no data generated

A100-80G `pax050`, source commit `2cc1b86c`. The preamble passed every gate
(`jax_backend=gpu`, `data_verified_pinned=395`, `torch 2.11.0+cu128 cuda True`) and the
generator's own `self-check` **passed on the cluster**: the warm-up-free `solve` is bitwise
equal to `data.solve`, and the six vendored generator files hash to exactly what the pinned
cache's index records as its own provenance.

`generate` then died in 2.6 s, in `data.provenance` → `data.source_hashes()`, which builds
`path.relative_to(ROOT)` over paths derived from `data.py`'s own location — paths that do not
exist in the flat staged `code/` tree. This is exactly the class the design audits named
(codex B6, "the pinned generator cannot simply be flattened into the parent staging layout");
`gen_more.gpu_modules` had been rewritten for it, `data.provenance` had not.

Fixed in `gen_more.provenance`, which copies every field of `data.provenance` and takes the
source hashes from the vendored files instead. Resubmitted as `gen02`. A local smoke now runs
the whole generation and pooling path **from a copy of the staged `code/` tree**, not from the
worktree, so this class fails on this machine in future.

45 s of A100 time; no output beyond the logs and `out/gen-self-check.json` kept here. Counts as
one of the lane's six submissions.
