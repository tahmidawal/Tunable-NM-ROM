**SUBMIT c3grid at HEAD `1f1b16986`.** The common-panel pricing blocker is resolved.

| Check | Finding |
|---|---|
| Common-panel pricing | A correctly uses W512/lat32768 fixed-sweep BE at dt0: **0.299208 s**. Subjects and warm-ups use the most expensive eligible fixed-sweep subject: W1024/lat65536 CN K2 at dt0, **3.237311 s**. |
| Actual execution coverage | `time_panel` executes **174 subjects, 4,176 timed calls and 700 warm-ups**, including burn-in before every call. Their conservative price is **8,093.05 s**. Inventory counts 175 subjects and prices **8,139.48 s**—a harmless **46.44 s overestimate**. |
| Certificate | `verify_certificate` **passes at HEAD** for H200 / 48 h / 200G. Independent projection exactly reproduces **22.46052 h including margin** and **15.76388 GB** peak. |
| A11.4 | Restatement removes the unsupported one-ULP bound and explicitly identifies peak-normaliser attribution as unproven. |
| Per-case `w0` | Normalisation is now per case. Archived arrays pass: only case 1 differs; maximum relative difference **1.17790e−16**, below **1e−12**. |
| Esc/bud impact | Numerical projection logic is unchanged. **Historical certificates remain bound to older source hashes and are not reusable at HEAD**; this does not affect the regenerated c3grid certificate. |

Nonblocking documentation residue: `select3d.py`’s nearby comment still repeats the old one-ULP/definitive-cause wording.

The prior audit’s **8,081.30 s** estimate omitted pricing the baseline warm-ups at the maximum subject cost; **8,093.05 s** is the corrected execution-based bound.

No files were written and no job was submitted.