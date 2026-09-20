# hires-poisson — speed and accuracy log

Each entry: hypothesis → change → measured effect (same job, before/after) → parity → kept/reverted.
Numbers below marked *smoke* are local GB10 N=64 mechanics checks, not results.

| # | hypothesis | change | measured | parity | status |
|---|---|---|---|---|---|
| S1 | the retained `correction_core` kernel spends timed device time on validation (jacfwd, SVD of the projected Jacobian, stationarity) that is not part of the answer | `lean64`: diagnostics moved to an untimed post-query jit; decode chunked | *smoke N=64:* q0 7.1 → 1.9 ms device | field 1.5e-16, integers identical (*smoke*) | pending cluster measurement |
| S2 | decode is memory-bandwidth bound at large n, so an f32 bank halves it | `lean32`: bank stored/applied in f32, field returned f64 | pending | *smoke:* 6e-8 (q=0) … 4e-5 (q=R) field; first-guess 1e-5 limit FAILED, limit set to 1e-4 before any GPU job | pending |
| S3 | at q=R the LM iteration is inert | direct thin-QR triangular solve (inherited from p-linear A4) | pending | — | pending |
