# Independent 3D normal diagnostic review — 2026-10-28

The first source candidate `5d5ec40fc` was held before staging: its installer required
`DEPENDENCY-REPAIR.json`, which was absent from the candidate. The old binding named
`g1-smoke06` and could not be copied unchanged into the new diagnostic directory. Hardware
logging also needed restoration because the diagnostic bypasses the solver's queries.
The original candidate and binding remain preserved. No failed allocation resulted.

Focused recheck: **READY for exact staging and coordinator release** at
`6db733531ab0a551baa4fa816f0f386927f7d8f8`. All eight candidate inputs match hashes,
sizes and commit bytes. The candidate-owned binding now names `g1-normal01`, and bounded
CPU/GPU/UUID/device-visibility logging is restored. Geometry/facet source is unchanged;
shell syntax passes. Candidate manifest SHA256:
`db2c42fcbc4967066b3e76a9678e4890953b8e4ee647fbc8fabae548b5a8f757`.

The numerical reviewer found no blocking geometry/facet source error. Like the measured
2D diagnostic, the source retains raw boundary normals, adjacent volume witnesses, ordinary
boundary integration and a separate constant-P0 volume-facet linear form. It saves the
fresh mesh before probing. The order-four mapped volume rule may have signed library
weights; these are diagnostic samples, not admitted positive ROM quadrature. The divergence
theorem uses an independently integrated higher-order volume denominator.

Release is limited to the fresh isolated geometry-only allocation, one H200 GPU, sixteen
reserved CPUs, four requested FE threads, 64 GiB and twenty minutes, with the existing
element/sample caps. No sign correction, matrix factorization, PDE solve or training is
permitted by this release. Inspect nested facet status; outer completion alone passes no
G1/reference criterion, and a fresh mesh cannot recover the original unsaved mesh.

The coordinator independently checked the staged bytes before submission: all eight inputs
plus `SOURCE_COMMIT.txt` match, and the directory has no outputs/dependencies. Remote
manifest SHA256 is `1f4ff9e60c44609af52fe8a7c6b7c9bf1d9bf7f976495cf22c74d14bfdcc9ca5`;
source-record SHA256 is `945851c7209512f894fd009c15b37730f24b1165ffb819280fe94c5bc291cb98`.
The exact directory is `/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-normal01`.
Check the actual queue before and after the single submission and retain all evidence.

The reviewer edited no files, submitted no jobs and ran no native FE calculations.
