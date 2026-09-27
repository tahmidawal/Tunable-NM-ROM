# g2d / speed2048 — tuned-solver (__eng) NM-ROM panel at Burgers 2D 2048²

Coordinator task, 2026-09-25. The code is burgers2d-speed @ fb4a9ff7 (`COPIED-FROM.json`). `b2speed.py`, `b2fast.py`
and `audit_b2speed.py` are unmodified. `config-2048.json` is the lane's own `make_configs.base(2048, 'b2048')`. The
only edits to `make_configs.py` are the absolute path to the parent lane and reading the 2048 fast bar from
bk2048-summary; both are recorded. Arms and the FOM grid are the same as config-1024, apart from the 1024-only
general-path arm, which `base()` adds only at 1024. Cohort dev6, reps 5/3 A-B-A, one A100-80GB job (sp2048,
4327675). The lane's NumPy audit ran inside the job on the saved fields, and only the summary and result.json were
pulled.
