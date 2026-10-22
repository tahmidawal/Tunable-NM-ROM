**SUBMIT** — audited HEAD `5569471c36e`.

- Corrected commitment: `24 + 2 × 6.2275 = 36.455 GPU-h`. Certificate total reproduces **73.259275 ≤ 87 GPU-h**, leaving **13.740725 GPU-h**.
- Cancellation guard validates numeric ID and expected owner/name, retries empty queries, and checks terminal status after cancellation. PID **2838310** targets **6187596**, confirmed live as `tawal01/ibox-smk2-A4`. Budget approval no longer depends on cancellation at exactly 14 hours.
- All three release body hashes and resource specifications match `RELEASE.json`. Submission retains locked lane/account concurrency checks.
- Approval covers replacing the existing W28-only releaser with **W28_A1 → W08_A1 → W08_A2**, as proposed. That replacement remains pending.

No blocking safety, budget, or correctness defect found. No files changed, jobs submitted, or LAB-LOG writes performed.