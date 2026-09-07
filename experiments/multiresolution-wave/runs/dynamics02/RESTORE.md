# Complete wave dynamics archive

The split archive retains every byte collected from the cluster, including the
large full-grid reference fields and bank tables. Raw JSON, code and manifests
are also tracked directly. Large extracted NPZ arrays are omitted as duplicate
git blobs; they are recoverable from the checked archive parts.

From the experiment worktree, run:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-wave/restore_archive.py experiments/multiresolution-wave/runs/dynamics02
```

The helper verifies every ordered archive part and the concatenated archive hash,
checks existing files without overwriting them, restores missing files, then
checks all three extracted source/output/pull manifests. A mismatched existing
file aborts restoration. The supplied Python path is the authorized local venv.
