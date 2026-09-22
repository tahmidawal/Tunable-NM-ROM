### burgers-heldout — milestone 3 (lane closed, 8/8 jobs): bh5 (incumbent with lat64 j=1 at 4096², coordinator task) audited

Job 4153483 (H200, source 7912b033), dev6 + hold64, 5 reps; audited by `audit_bh.py` and burgers-eqcert's `audit_eqcert.py`; remote deleted; namespace `bheld_20260921` empty. Summaries `experiments/burgers-heldout/checks/bh5-summary.json`, `bh5-eqcert-summary.json`; lane `reports/summary.json`.

- **Paper rule lat64 j=0 at 4096²: passes 5/5 held-out draws (ρ_max 0.0717 / deployed 0.1072) but FAILS the confirmation draw (ρ 0.1173 > 0.116).** Numbers reproduce hb4k04/hb4kh64: q256/M1088 g1e-2 dev6 0.604 % @ 107.0 ms = 4.90× lean_nt3e-3 (523.8 ms, 0.050 %); hold64 1.331 % @ 99.6 ms = 5.39× (536.8 ms, 0.138 %). g1e-3: 125.9 ms (4.16×) / 111.3 ms (4.82×).
- **j=1 certifies with margin (draws ≤ 0.037, confirmation 0.051)** at identical error, but the exact first step at 4096² makes the query 5308 ms (dev6) / 6558 ms (hold64): 0.10× / 0.08× the FOM. Not a usable speed row.
- q0 scaled: dev6 2.415 % @ 40.2 ms (10.1× lean_nt1e-3_dt01), hold64 9.03 % (10.1×), certified. Control bad0 fails.

**Consequence for the paper.** The 4096² Burgers headline rule (lat64 j=0) is not certified at 4096² under the eqcert procedure (thin miss on the confirmation draw, 0.1173 vs 0.116); the certified alternative (j=1) is ~50× slower. Flag the 4096² Burgers rows as "rule marginal/uncertified at this mesh". Lane verdict unchanged: held-out ≤1 % with ≥5× not met by any certified arm.
