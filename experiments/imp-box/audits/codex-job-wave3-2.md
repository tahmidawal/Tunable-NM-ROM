Round 1 blockers are resolved:

- `ps` confirms all three duplicate releaser PIDs are gone; no live releaser remains.
- Corrected certificate independently recomputes to **80.61135 ≤ 87 GPU-h**, including the **10 h failure allowance**, leaving **6.38865 h**.
- A14 records the allowance correction and W08_A1 ordering deviation. W08_A1 is running; neither A3 job is queued.
- All three proposed job bodies match their release hashes.

Approve starting exactly one `cluster/auto_release.sh 30 W40_A3 W28_A3 W08_A2`.

Minor documentation caveats: A14’s “now runs” is premature; legacy certificate fields and release descriptions retain 76.8. The corrected total governs this approval.

No files changed or jobs submitted.

**SUBMIT**