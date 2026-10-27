# G1 preparation record — 2026-10-27

The coordinator relayed an independent Codex physics/code review of source commit `485d4252f740d54fccb172f089ecc48a0db88438`, source SHA256 `e730963026228d869d0ff6a7ae138d1eb2f5c13ff0c11d6167558ee9fffc6ef7`, as scoped READY subject to the 2D smoke and exact staging. This lane has not independently read the full review and does not represent this note as a replacement for its original text. The coordinator authorized staging only, with no installation or submission.

Before staging, the account queue was empty and the exact target did not exist. Code was extracted from the reviewed commit and copied directly into `/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke01/`, never through login-node temporary storage. The remote checksum check passed for every entry, including source, configuration, central A3 snapshot, installer, lockfile, dependency metadata, batch script and source commit record. `g1-smoke01.INPUTS.sha256` records those bindings; `g1-smoke01.REMOTE-VERIFICATION.txt` preserves remote verification output. The account queue was still empty after staging. No dependencies have been downloaded or installed, and no job has been submitted.

Implementation qualifications highlighted by the review remain explicit:

- `fe.quadrature_order` is passed as **additional assembly integration order** through `bonus_intorder`; diagnostic `Integrate(order=...)` is an **absolute diagnostic rule order**. These two numerical settings must not be compared as if they were the same rule. A future quadrature study must store the actual rules and distinguish these semantics.
- The case wall-time check is cooperative between native operations. It does not interrupt a mesh generator or factorization already executing. The outer process timeout and Slurm wall limit provide the hard bound.
- The coarse P2/P1 smoke, geometry checks, and manufactured control do not certify physical references. Signed curved-Jacobian checks, the stronger boundary trace checks, mesh/order/tolerance/domain series, restart parity and independent numerical result review remain required before G2.
- The 2D smoke and Claude implementation gate must be resolved by the coordinator before any 3D solver execution. Staging is not execution release.

The 2D implementation agent also supplied a mapped-integration-point API approach, verified against the pinned library's source but not yet runtime tested, for later signed-Jacobian and sampled boundary-maximum controls. No unreviewed source change was made while binding this smoke. A sampled maximum will not be described as a certified continuous supremum.

## Subsequent audit correction

The independent physics review of candidate `c74d39dc6` found unsupported boundary evaluation of the computed vector gradient, also affecting the older diagnostic stress path. The prior readiness record is insufficient to release the staged candidate. No staged 3D job has run; the original package stays unchanged. See `BOUNDARY-TRACE-AUDIT-CORRECTION.md` for the candidate 03 correction and required new review.
