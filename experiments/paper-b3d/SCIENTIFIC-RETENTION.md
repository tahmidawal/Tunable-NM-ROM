# Burgers 3D scientific evidence retention

This supplement records the completed lossless retention upgrade for the development panels. All scientific prediction/reference fields, raw paired records, source files and selected trained checkpoints are recoverable from committed split archives; explicitly listed training caches and optimizer continuation states remain in the original local collections.

| Attempt | Files restored | Parts | Tar bytes | Restored file bytes | Git source |
| --- | ---: | ---: | ---: | ---: | --- |
| `b3d003` | 281 | 67 | 4464558080 | 4960468563 | `7528fba6` |
| `b3d004` | 494 | 99 | 6598512640 | 8098799891 | `7528fba6` |
| `b3d005` | 1395 | 208 | 13926707200 | 17994878053 | `7528fba6` |

The verifier reconstructs every file byte stream directly from the pinned Git blobs, checks every file SHA256, checks each split part and the concatenated tar checksum, and validates lossless hard links for identical contents. It does not rely on the ignored original local field files as proof. Original `RETENTION.json` records retain their historical scope; each attempt's new `SCIENTIFIC-RETENTION.json` links the correction.

To verify an archive from the Burgers worktree, replace the attempt below as appropriate:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/scientific_archive.py verify b3d005 --commit 7528fba6
```

To materialize all scientific files, concatenate that attempt's `scientific.tar.part*` files in lexical order and extract the resulting tar into a new directory. The manifest uses paths relative to the original collected directory. Verify the reconstructed archive before using it for a report. Regenerable training arrays and optimizer continuation states are explicitly listed in `manifest.json`; they were never deleted from the original local collection and are not claimed to be recoverable from this scientific archive.

## Glossary

- **Scientific field:** an exact saved numerical prediction or reference used to audit a reported result.
- **Paired record:** accuracy and runtime recorded from the same solver invocation.
- **Split part:** a consecutively numbered piece of one lossless tar archive.
- **Restored file bytes:** total sizes of all reconstructed files, including repeated identical files restored from hard links.
- **Hard link:** a tar entry that reuses an earlier identical file's bytes while preserving its own original pathname.
- **SHA256:** a checksum used to detect any change in file content.
- **Git source:** the commit containing the exact archived bytes used by the restore audit.
- **Optimizer continuation state:** training bookkeeping needed to resume optimization; distinct from the selected trained prediction model.
