1. **CORRECT — exact coverage and valid bundles.** Both bundles contain exactly 61 manifested payload files, with all hashes matching, no missing/extra entries, and zero symlinks. Both pass the actual guard. `submit.sh:15` includes non-directory entries, including symlinks, excluding only the root manifest. In-memory injection of extra `SUBMIT.out`, file, and symlink path entries makes each guard exit 8; no fixtures were written.

2. **CORRECT — submission safeguards preserved.** Compared with audit 2’s baseline (`0aefa89f8^`), only the coverage expression and receipt write/read paths changed. Lane cap 2, campaign cap 8, queue checks, ledger `submit-check`/`bind`, remote lock, and `--no-requeue` remain unchanged. Both shells retain `set -euo pipefail`. A failed-command/`tee` simulation exited 42 without reaching the subsequent bind marker. The receipt now lives outside the bundle. The previously disclosed cross-lane race limitation remains.

3. **CORRECT — bundles unchanged since audit 2.** Both root manifests, provenance files, and batch scripts are byte-identical to `0aefa89f8^`. Unchanged manifests plus exact coverage and matching payload hashes establish unchanged bundle contents.

**SUBMIT-t3d64a**  
**SUBMIT-t2d1024a**