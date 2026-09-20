`analyze_c2.py` (the corrected spectral shift; `analyze_c.py`'s alignment had a missing 2π and is superseded — its raw-POD quadratic rows are still valid). Every field, training and held-out, is translated on the torus so that the Fourier phases of modes (1,0) and (0,1) vanish (exact spectral shift; residual phase 1e-16; the 5th-percentile relative amplitude of those modes is 0.09, so the phase is well defined for ≥ 95 % of states). POD is then recomputed on the aligned training states and the aligned held-out states are projected:

| linear subspace | raw fields (dev median) | aligned fields (dev median) | gain |
|---|---|---|---|
| POD-14 (= 16 online unknowns with the 2 shifts) | — | 0.197 | 1.21× vs raw POD-16 |
| POD-16 | 0.239 | **0.177** | 1.35× |
| POD-18 | — | 0.162 | 1.47× vs raw POD-16 |
| POD-30 (= 32 unknowns with shifts) | — | 0.111 | 1.27× vs raw POD-32 |
| POD-32 | 0.141 | 0.102 | 1.38× |
| energy in 16 POD modes | 0.891 | 0.933 | |
| nearest training state, same time (coverage) | 0.644 | **0.446** | |
| quadratic manifold on aligned POD-16 coordinates with (ν, t) | — | 0.243 (from the partially aligned run; not redone) | none |

Removing the two translation dimensions by exact structure is worth 1.2–1.5× on the *linear* side and shrinks the coverage gap from 0.64 to 0.45 — more than the 4× data of ns302 bought (11 % in coverage, 1.19 → 1.46 in ratio) and at zero data cost. It does not, by itself, reach 2×, and the quadratic manifold still adds nothing on aligned coordinates.
