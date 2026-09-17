# Pinned source copies

`b-eqtop-interim-summary.json` — the b-eqtop lane's `experiments/b-eqtop/reports/summary.json` at lane commit `d6071e3c`
(2026-09-17 10:55, "interim report and summary.json with bet301 pending"). It is the only machine-readable record of
the timed primary-rule EQ ladder and its dense twins (job 3780164, 84 `table == "ladder"` rows); the lane's final
summary (commit `fc7ca639`) dropped those rows. `gen_tables.py` reads the ladder rows from the final file when they
exist and from this copy otherwise, and records both SHA256s in `tables/provenance.json`.
