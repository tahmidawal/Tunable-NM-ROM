Two issues block release as described:

- **A14 ordering is not implemented.** Three live `auto_release.sh` processes still specify `W28_A1 W08_A1 W08_A2`. The cluster confirms **W08_A1 job 6202478 is already running**, while neither A3 job is submitted. W08_A2 remains eligible to launch ahead of A3. Reconcile the running job and replace the competing release processes with one agreed queue.
- **The failure allowance excludes A3.** `plan-wave3.json` hides both A3 jobs inside `committed_hours`; its `waves` list contains only W08_A1/A2. Consequently, `resources.py:113` reserves **6.2275 h**, rather than the largest wave-job exposure of **10 h**. Correcting this gives **80.61135 ≤ 87 GPU-h**, so it still fits, but the certificate needs correction.

Checks that passed:

- Ledger sum: **23.8330556 GPU-h**, correctly rounded to 23.8331. Published certificate arithmetic: **76.83885 GPU-h**.
- Both A3 body hashes match `RELEASE.json`. They run full-size reference/data/POD/bank/ladder/time/controls stages: no smoke overrides, train/dev/val **96/8/32**, A3 reference **N=384**. W28’s bank omission follows the registered scope.
- The template enforces **one H200, 240G, 10 hours**, GPU preflight, f64 and highest matmul precision.
- A time-stage stall is wall-bounded. Collection handles `TIMEOUT`; absent final results or passing controls produce **missing** entries in `stoprule.py`.
- smk2-A4 cancellation is confirmed at **09:49:23**. Its captured-constant/cache warnings do not establish A3 feasibility; A3 remains an exposure-bounded attempt.

No files changed, jobs submitted, or LAB-LOG entries written.

**DO-NOT-SUBMIT**