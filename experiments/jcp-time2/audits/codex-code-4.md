**Overall verdict for SUBMITTING THE SMOKE: NOT CLEAN** at `0ded20a9e`, solely because the current audit log blocks staging.

- **Item 1: RESOLVED.** The auditor verifies the staged config’s provenance hash and equality with `res['config']`. Both contain the substituted cluster `refs` path, so this introduces no false mismatch.
- **Item 3: Partially resolved.** Config parsing and reference validation now precede `mkdir`. However, untracked `experiments/jcp-time2/audits/codex-code-4.log` triggers the unchanged dirty-check assertion at `stage.py:43`.
- **No new code defect found** in the three reviewed files. Collection retains the staged config; Python syntax and shell syntax checks pass.

Commit or relocate the audit log before staging. No files modified or jobs submitted.