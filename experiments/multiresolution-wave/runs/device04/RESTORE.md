# Complete fresh-wave device-query archive

The checked split archive retains every byte collected from the cluster,
including every full reference, first timed output, accuracy-control output and
learned mesh table. Repeated timed outputs share an artifact only after their
complete displacement and velocity hashes match. Raw JSON, source, checkpoints,
provenance and manifests are also tracked directly; extracted large field arrays
are omitted as duplicate git blobs.

From this experiment worktree, restore missing extracted files with:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-wave/restore_archive.py experiments/multiresolution-wave/runs/device04
```

The helper checks ordered archive-part hashes, the reconstructed archive hash,
existing extracted files and the source/output/pull manifests. It refuses to
overwrite an existing mismatched file. The archive is authoritative only after
the adjacent cleanup record confirms successful checksum collection.
