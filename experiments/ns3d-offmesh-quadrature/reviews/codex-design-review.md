- **MAJOR — `run_mesh.py:499–512`: G5 can miss invalid timed outputs.** `max(agree, NaN)` preserves `agree`, potentially passing a non-finite result. Comparing only error magnitudes also cannot establish field reproduction. **Fix:** explicitly reject non-finite timed fields and differences; compare timed fields directly against the corresponding accuracy-pass fields.

- **MINOR — `run_mesh.py:520–521`: microbenchmark exposes only one Jacobian row.** This permits backend optimization to remove unused rows. CPU compilation retained the full contraction, but GPU behavior remains unverified. **Fix:** make every Jacobian entry contribute to the returned checksum and loop dependency.

- **MINOR — `DESIGN.md:155–158`, `run_mesh.py:67–70`: an endpoint-only pass counts as a plateau.** The suffix algorithm correctly implements the registration, but the largest size can qualify without demonstrating that accuracy is “held.” **Fix:** label this outcome “endpoint reached; plateau unconfirmed,” or require a passing successor. Keep \(M^\ast\) explicitly ladder- and seed-specific.

- **MINOR — `DESIGN.md:139–143`, `run_mesh.py:258–272`: continuum-relative Jacobian measurements are missing.** Only advection is compared against the network continuum target; Jacobians are compared solely against the tensor. **Fix:** stream a continuum Jacobian reference too, or explicitly narrow the registered measurement to continuum-relative advection.

- **MINOR — `qrom.py:309–316`: the positive-control JIT captures mesh-sized geometry.** This contradicts line 19’s explicit-argument guarantee and embeds approximately 21 MB of wave-number constants at \(n=96\). **Fix:** pass wave numbers and offsets into the compiled function explicitly.

**Overall verdict: READY WITH FIXES.**

Step 0, the contraction and scaling, test values, curl indexing, half-space interpolation, shift sign, and registered solver copies are consistent with the references. CPU-only synthetic checks reproduced tensor advection/Jacobians to approximately \(10^{-15}\). A1 is transparently documented and reasonable.

The comparator selection matches the parent. No definite 80 GB memory blocker was found, though the \(96^3\) positive-control tables alone occupy approximately 8.9 GB before preconditioning and workspaces; the small smoke does not establish peak-memory safety.