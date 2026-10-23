The report and export pass the audit. No files were changed.

| Check | Verified result |
|---|---|
| Export | `stage: final`; nine retained cells, seven deferred cells; required sections, qualifications, and correct test source paths |
| Cell results | All 186 exported entries agree with source JSONs; bank tables and stage costs reproduce |
| SPEED-STOP | **NOT FIRED**: 95 matches, 80 ≥ 0.5, 27 missing/FOM-censored; maximum **2.14378237** |
| BANK-STOP | **NOT FIRED**: 3 above, 4 below, 8 unaffordable, 3 missing |
| Crossings | 60 records, 156 points; brackets reproduce; all 136 affected numerical points carry ladder-minimum qualifications |
| A11 | All 27 comparisons reproduce; FFT/dense ratios **1.542507–2.017621**, supporting the prominent screening-qualified finding |
| A16 | All 28 ratios reproduce from timing arrays; generated ranges and cell attribution are correct |
| Test-once | 164 pairs, 142 matched; 9 matched pairs retain incomplete-search qualifications; ratios reproduce from timing triplets |
| GPU-hours | **48.9677778**, consistent with ledger rounding and reported **49/87** |

Both PNGs agree with the retained results, include A3, use consistent arm colours and qualification markers, and mark A4 panels deferred.

The previous audits’ required fixes are present: crossing qualifications, escaped table pipes, generated A16 summary, expanded glossary and mathematical formatting, prominent A11 finding, test labels, source paths, and plot repairs. No new substantive errors were found.

Conditional references remain estimates, speed-ups retain their tested-ladder limitations, and test-once results are confirmation of frozen validation selections. Deferred cells are not reported as results. Neither prohibited historical comparison appears. No hand-typed measured results were found in the generator.

Required fixes: none.

VERDICT: CLEAN