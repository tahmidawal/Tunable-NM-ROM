**SUBMIT — no blocking fixes found for `s1b`.**

1. **Gram fix is correct.** `mmcore.py:94–116` partitions rows without overlap. At 4096², the basis has 16,769,025 rows: 63 full blocks of 262,144, then 253,953 tail rows. Every `dynamic_slice` satisfies `a + block <= n`, so none clamps. The sum equals the dense Gram mathematically, with ordinary floating-point summation differences. Eight small f64 boundary/tail tests passed; maximum Gram difference was `7.11e-15`. The projection rewrite also passed (`1.42e-14` difference).

2. **No remaining eager full-basis transpose or concatenation found in the five requested files.** Relevant sites:
   - `mmcore.py:47,54,65,67,73`: resampling transforms inside JIT, operating on eight-column blocks.
   - `mmcore.py:274,301`: eager transposes of block/candidate Jacobians, not the full basis.
   - `mmsmoke.py:58`: restricted 256² basis.
   - `mmbuild.py:320–321`: lattice-only basis.
   - `mmtime.py`, `mmcompile.py`: no direct transpose/concatenation sites.

   Other allocations remain: `mmcore.py:340` copies a rank prefix; `mmcore.py:88` can copy the output for a partial final column block. Neither recreates the failed full-basis transpose; S1’s resampling widths divide eight exactly.

3. **Local staging is consistent.** All 146 manifest entries and all provenance hashes pass. The five audited files and `submit.sh` match their staged copies. Certificate config SHA matches `5c334298…61963`; reservation, paths and `S1_ATTEMPT` agree on `s1b`. Production acceptance now reads that attempt.

This clears the narrow audit; GPU peak-memory success still requires the smoke. No files changed or jobs submitted.