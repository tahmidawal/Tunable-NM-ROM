## 2026-09-17
### b-panel — CORRECTION to the wording of the entry above: the certified-EQ ladder owns 7 of the 8 non-dominated reduced points on the all-times metric, not "the entire reduced frontier"

The entry above says, in retraction item (2), *"On the all-times metric the certified-EQ ladder still
owns the entire reduced frontier and no POD rank appears on it."* The second clause is correct; the
first is imprecise and is corrected here. No number in the entry, the report, `summary.json` or the
audit changes — only that sentence.

**What the all-times reduced-only frontier actually is** (job `3780638`, 8 points over all 27
admissible reduced subjects, recomputed from `reports/summary.json`):

| subject | family | median GPU ms | worst all-times % |
|---|---|---:|---:|
| `q0_M64_eqcert_g1em06_fastL4` | correction ladder, optimised q=0 kernel | 39.1 | 2.5629 |
| `q0_M64_eqcert_g0p001` | correction ladder | 42.6 | 2.5628 |
| `q16_M128_eqcert_g0p001` | correction ladder | 60.4 | 2.4806 |
| `q32_M192_eqcert_g0p001` | correction ladder | 72.3 | 2.3534 |
| `q64_M320_eqcert_g0p001` | correction ladder | 88.5 | 2.1489 |
| `q128_M576_eqcert_g0p001` | correction ladder | 189.0 | 1.8116 |
| `q256_M1088_eqcert_g0p001` | correction ladder | 429.4 | 1.0324 |
| `free512_M1024_dense` | unrestricted bank (R = 512) | 2542.3 | 0.6027 |

So the accurate statement is: **the certified-EQ ladder owns 7 of the 8 non-dominated reduced points
and every point up to 429 ms; the eighth is the unrestricted-bank endpoint, which is not a ladder
rung; and no POD rank is on this frontier.**

**Why a reader can easily get a different answer, which is the part worth carrying.** The
coordinator re-derived this set from `summary.json` and got seven points — the six certified-EQ
rungs plus POD-512 — and reported POD-512 as non-dominated at the expensive end. That follows if the
**unrestricted-bank arm is left out of the candidate set**, and only then: `free512_M1024_dense` is
both cheaper (2542.3 ms against 2928.9 ms) **and** more accurate (0.6027 % against 0.6125 %) than
`pod512_M2048_dense`, so with the full reduced set POD-512 is dominated. Keep the bank endpoint and
POD-512 is off; drop it and POD-512 is on. The generated report now enumerates **every family** on
each frontier and states this pod512/free512 comparison explicitly, so the ambiguity that produced
two different answers cannot recur from the generated text.

**Unchanged and re-verified.** POD-512 *is* on the **evolved**-times reduced frontier, and the
finding that POD overtakes the correction ladder at ranks the head ablation never ran stands on that
metric: POD-512 reaches 0.2184 % at 2929 ms against the dense q=256 rung's 0.5194 % at 4165 ms,
better on both axes. With the full-order controls and the FNO included, nothing reduced is
non-dominated on either metric, which is the entry's headline and is unaffected.
`checks/recheck_headline.py`, the independent second code path, still reproduces every reported
number to 0.0 relative.
