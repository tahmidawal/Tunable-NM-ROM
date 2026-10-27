# Independent 2D native ownership repair review

Numerical reviewer: READY for one g1-smoke05 retry at exact source
`f4f90487940c8a1f0b8bb19d8e9e1b9d7505a7e4`. No blocker found.

Named transformations survive every mapped-point access; saved NumPy arrays are copied,
and finite-difference residual vector owners remain alive. Pinned `fem/intrule.hpp` shows
the mapped point's non-owning transformation pointer. Fault reporting and flushed markers
localize any future native failure. Original crash location remains unresolved.

All input hashes and syntax checks pass; installer, scientific configurations, equations,
thresholds and resource caps are unchanged. Coordinator independently rechecked the exact
staged bytes and queue before one release. No numerical execution by the reviewer. This
is a runtime-control retry, not a reference or training release. See job ledger for hashes,
resources, exact allocation and subsequent accounting.
