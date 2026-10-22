**FIX-FIRST — retry certification remains incomplete.**

[resources.py:40](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/resources.py:40) still budgets only first attempts. Using its own buffered estimates:

- The 256² pass finishes at **1,864.8 s**, allowing a retry under the **2,400 s** gate.
- That retry raises timing work to **9,032.4 s**, exceeding the **9,000 s** subprocess timeout.
- Including compile probes gives **186.54 minutes**: within the **215-minute** allocation, but absent from the **155.46-minute** certificate.

Update retry-aware certification and reconcile the subprocess timeout. An elapsed-time gate alone also does not guarantee the comment’s “at most one” retry.

Other checks pass: per-configuration persistence is connected correctly; J5 identity arrays are cached per configuration; all **157 manifest entries** verify; all **142 provenance files** match HEAD; certificate config hash matches.

No files changed, jobs submitted, large arrays loaded, or LAB-LOG/main writes.