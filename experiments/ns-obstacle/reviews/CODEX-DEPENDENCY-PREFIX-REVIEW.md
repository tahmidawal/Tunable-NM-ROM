# Independent dependency-layout repair review

Numerical reviewer, read-only, 2026-10-27: READY for one g1-smoke04 retry at
`d345b0b05cd57811e5c698f4fb0d593adef525f2`. No blocking defect found.

The private prefix matches the pinned loader's RECORD-based library lookup and pip's
[documented prefix scheme](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-prefix).
Wheel versions and hashes are unchanged. Installation uses the designated interpreter,
private paths and no dependency resolution. Integrity, import provenance, GPU, f64 and
highest-precision guards precede FE execution. Scientific code/configuration bytes match
`1ac56b2d7`; all 13 input hashes/sizes and syntax checks pass. Resource caps unchanged.
No installation, numerical execution or edit performed by the reviewer.

The coordinator independently checked all staged source hashes, exact source commit,
manifest and checksum file on the cluster before releasing one retry. At release no
campaign job was live. Unrelated campaign work was not changed.

- Manifest SHA: `f03788836f4fd31765b2ac184abc8b5e636c13daa77e0d3691c9a27f2dd2672b`.
- Checksum-file SHA: `6d3360b24a996f78a969da07c58cbd69375525fae9436d91a71f8a222a476b3d`.
- New allocation: see machine-readable job ledger. No reference or training admission.
