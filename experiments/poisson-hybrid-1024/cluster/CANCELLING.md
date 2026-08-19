# Cancelling these jobs

Never run `scancel` directly. Jobs are deliberately named `ctol_hybp_*` so the repository's
guarded explicit-ID script accepts them:

```bash
experiments/cost-to-tolerance/cluster/cancel.sh <numeric-job-id>
```

The script checks the account, current queue membership, and `ctol_*` name before cancelling
anything. It aborts the whole request if any ID fails validation.
