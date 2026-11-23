# Consolidated JCP physical-diagnostics experiments

This branch combines the completed phase/shape, width/deployment and accuracy/cost experiment branches at the user's request. It preserves their measured records and failed qualifications exactly; consolidation does not rerun or reinterpret an experiment.

`SOURCES.json` fixes the authorized source commits. `verify_consolidation.py` checks their ancestry, exact tracked-file identities and the absence of unrelated changed paths, then generates `CONSOLIDATION.json`. Run it with the repository's absolute Python environment path. `sync_github.sh` backs the branch up through its code-only mirror using the canonical consolidated fork and corresponding remote base.

The checkout includes the named scientific directories. The cost lane's large regenerated basis arrays and transport archive remain tracked in the consolidated Git history but are excluded from this additional physical checkout and from the code-only mirror. Their original local copies, hashes and regeneration instructions remain in the source cost worktree. No original worktree or evidence is deleted by this merge.

The canonical project log and generated reports remain on `main`; this directory records only consolidation provenance. New initialization development and untouched-case confirmation must be distinguished from the preserved failed fresh confirmation.

## Glossary

**Source commit:** an exact saved version of a completed experiment. **Ancestor:** a source history retained by the merged branch. **Blob identity:** exact identity of tracked file bytes in Git. **Qualification:** the original numerical checks, whose outcomes are unchanged here. **Code-only mirror:** the remote backup excluding designated large regenerable files. **Sparse checkout:** a local selection of files; excluded files remain in Git history.
