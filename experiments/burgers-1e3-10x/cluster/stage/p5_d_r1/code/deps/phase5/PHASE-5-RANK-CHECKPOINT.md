# Phase 5 rank-geometry checkpoint

Status: **read-only exposed-development diagnostic; not a scientific cell or promotion claim**.

The immutable P4-D H1 coefficient artifact and its independent audit pass every hash,
finite, exact-boundary, support, partition-of-unity, and normal-residual check.  The
current dense hyperdecoder has the exact form `q19 -> 32 -> 32 -> 3328`; therefore
`c(q)=b+W*h(q)` lies in one affine coefficient subspace of dimension at most 32.

Using the fixed single-threaded randomized-SVD protocol recorded in
`phase5_rank_diagnostic.json`, the first 32 sketch singular directions capture
`0.9363904875175779` of centered
coefficient Frobenius variation, leaving estimated relative residual
`0.2522092632763953`.  The reported
`s33/s1` is `0.1304454098690537`.  This supports testing
a generator with nonlinear spatial processing after spatialization, but the randomized
quantity is explicitly **not** a rigorous singular-value lower bound, a field-error
lower bound, or an accuracy gate.

The Phase-5 G1/G2 bracket was fixed prospectively before P5-D and may not adapt to
this diagnostic or to P5-D coefficient-SVD results.  Model-validation and confirmation
remain untouched.  No generator was implemented and no scientific job was submitted.

Reproduce/check:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/tahmid/Dev/.venv/bin/python \
  experiments/burgers-1e3-10x/b10_phase5_rank_diagnostic.py --check
```
