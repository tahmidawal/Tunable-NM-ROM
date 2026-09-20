# Verified older Burgers field storage plan

This is a storage plan only. Older fields have not been deleted; broader authorization is still pending. Scientific results and the accepted B007 final are unchanged.

| Attempt | Candidate files | Logical bytes | Allocated bytes reclaimable | Actual Git files verified | Archive commit |
|---|---:|---:|---:|---:|---|
| b3d001 | 273 | 3265690066 | 3266293760 | 305 | `bb0fe9ed715b81933a15f44abba6d6c8a6283392` |
| b3d002 | 9 | 1021292599 | 1021313024 | 57 | `bb0fe9ed715b81933a15f44abba6d6c8a6283392` |
| b3d003 | 207 | 4685275094 | 4685721600 | 281 | `7528fba6519ae56fccec8d83fb0e9d5a13ba9514` |
| b3d004 | 384 | 7738942420 | 7739760640 | 494 | `7528fba6519ae56fccec8d83fb0e9d5a13ba9514` |
| b3d005 | 1105 | 15964300726 | 15966568448 | 1395 | `7528fba6519ae56fccec8d83fb0e9d5a13ba9514` |

Total: **1978 files**, **32679657472 allocated bytes** (30.435303 GiB). These are potential savings, not space already reclaimed.

The exact per-path SHA256, size, allocated bytes, inode and modification time are in `PLAN.json`. Fresh, timestamped actual-Git proofs are in the adjacent `b3d00*-GIT-RECHECK.json` files. Every archived file, archive hard-link alias, part and concatenated archive checksum passed. Every proposed local file separately matched its archived SHA256. The helper has no deletion mode.

Candidates are untracked NPZ bundles containing a `fields` array, only under the recorded inactive output folders. All checkpoints, model/offline assets, tracked replay fields, metadata, archived source, local archive parts and Git blobs are excluded. All B006/B007 files and local-only training/optimizer exclusions are untouched. In particular, `.npy` training caches omitted from the archives are not eligible under this plan.

After explicit authorization for this exact list, any remover must recheck local files against the saved hashes and unchanged metadata before removing only those paths. Preserve this plan, the fresh Git proofs and every archive copy. Update the materialization records and handoff only after an authorized removal has actually completed.

To restore one listed file from committed Git bytes into a new directory within this worktree:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/older_field_storage.py restore b3d001 --path out/fom_nt1e-02_case0_rep1.npz --destination experiments/paper-b3d/checks/restored-older-b001
```

Use repeated `--path` arguments for selected fields or `--all-candidates` for the full listed subset of one attempt. The destination contains paths relative to that attempt’s original `collected` directory. The helper verifies selected file hashes and the entire archive before publishing restored files, and writes `restore-audit.json`. It refuses any existing mismatched destination file.

Regenerate this note with `older_field_storage.py describe`. Do not rerun `plan` over the existing manifest; it intentionally refuses to overwrite the audit history.

## Glossary

- **Candidate:** a verified duplicate proposed for removal, still present locally.
- **Allocated bytes:** actual filesystem blocks occupied by a file; the potential disk-space gain.
- **Logical bytes:** the file length, which can differ slightly from allocated storage.
- **SHA256:** a content checksum used to prove byte-for-byte identity.
- **Actual Git proof:** reconstruction directly from committed archive blobs, independent of local field copies.
- **Archive hard-link alias:** a separately named archived file represented by an earlier entry with exactly identical content.
- **Materialization:** an ordinary local file copy of an archived byte stream.
