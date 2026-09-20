# Paper supplement evidence snapshot

This snapshot supplies the separately confirmed Poisson representation diagnostic and the provisional three-dimensional development appendix. It is not final-test evidence.

`manifest.json` pins every imported byte string to a repository commit and SHA256 hash. The result records retain per-case errors, all timing repetitions, upstream raw-result paths and audit hashes. The explicit selection lists every displayed method; the manuscript generator does not choose a best run. Figure PDFs are the coordinator's generated, hash-pinned counterparts of these same tables.

Normal manuscript builds read only this local snapshot. To import a later reviewed coordinator commit, run the following from the paper directory, replacing the argument with that exact commit:

```bash
/home/tahmid/Dev/.venv/bin/python gen_campaign_supplement.py --refresh <coordinator-commit>
./build.sh
```

Refresh imports the committed evidence and explicitly selected panels; it does not confer final status. Final-test integration requires reviewing the cohort freeze, scope, remaining qualifications and manuscript wording. The historical Poisson timed results and retraction remain in their original tables and source registry.

## Glossary

- **Snapshot:** an immutable local copy of the evidence used by this manuscript version.
- **Commit / SHA256:** repository revision / content checksum identifying the imported bytes.
- **Development / final test:** cases available during selection / cases held aside until selection is frozen.
- **Oracle / best found:** a truth-informed representation fit; its achieved error is an upper bound on the unknown global minimum.
- **Bank floor:** the lowest error achievable by free coefficients in the learned linear basis.
- **Paired query cost:** runtime from the same invocation that produced the reported prediction.
- **Provisional:** audited measurements whose scientific claim still awaits the stated confirmation.
