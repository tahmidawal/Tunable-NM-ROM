**Do not launch this version yet.** The core quadrature mathematics checks out, but submission blockers and weakened gates remain.

Read-only audit completed; no files changed, no GPU/JAX jobs run. Checks used source inspection, NumPy, NPZ metadata, and an empty-checksum-file reproduction.

| File | Verdict |
|---|---|
| `a1_3d.py` | **WRONG** — gates do not implement the amended specification |
| `q2d/qcore.py` | **CORRECT** — nodes rule and relocated paths |
| `q2d/qstudy.py` | **WRONG** — incomplete gates and target-failure handling |
| `a2_gap.py` | **NEEDS-RESTATEMENT** — mathematics correct; analysis/acceptance remains unimplemented |
| `make_configs.py` | **WRONG** — job-path coupling and broken 3D smoke configuration |
| `make_inputs.py` | **WRONG** — changes the registered state population |
| `cluster/stage.py` | **WRONG** — empty reference manifest blocks A2 |
| `cluster/submit.sh` | **WRONG** — reference-copy stdin hazard, empty-manifest failure, cap race |

**1. `a1_3d.py`**

The nodes operator itself is **correct**:

- Lines 85–91 form columns as `(R′, N)`, project each chunk to `(chunk, M)`, then transpose to **`(M, R′)`**, matching `contract_offmesh`.
- Lines 93–94 return batched values `(S, M)`.
- Interior weights give \(P=\Phi\), without weight renormalisation.
- A NumPy execution of the extracted function, including a padded column chunk, matched explicit GEMM exactly; \(N(c)=\tfrac12J(c)c\) agreed to \(8.9\times10^{-16}\).

Concrete defects:

- **G3 is weakened.** [Lines 251–262](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/a1_3d.py:251) check 32 random directions at **`1e-11`**, whereas DESIGN requires the Jacobian comparison at **`1e-12`**. Random directions are also a replacement for the registered full Jacobian check.  
  **Fix:** compare all columns, using chunked basis JVPs if necessary, at the registered tolerance.

- **G2c has reduced coverage.** Lines 100–108 and 243–250 check values on four states, but Jacobian columns on **only `cs4[0]`**, with only 32 columns at both meshes. Amendment 1 calls for the full check at \(64^3\), and four-state/32-column coverage at \(128^3\).  
  **Fix:** loop over all four states; check all columns at `n=65`.

- **The reference-magnitude guard is absent.** `rel_max` at lines 56–58 and the separate G2a ratio do not enforce Amendment 3’s \(\max|b|\ge10^{-8}\).  
  **Fix:** one gate helper should enforce finiteness, reference magnitude, and `max_error <= tolerance * reference_max`.

- **G1 uses a different finite-difference test.** Lines 190–192 perturb all coordinates simultaneously. Amendment 2 specifies separate coordinate differences, summed afterward. Both approximate the directional derivative, but their truncation errors differ.  
  **Fix:** implement the registered coordinatewise test.

- **Continuum checks cannot invalidate the result.** Lines 330–348 and 400–408 record check errors, but never enforce the `1e-6` bar or mark the configuration invalid. Line 431 still marks execution complete.  
  **Fix:** persist an explicit target-valid flag and make it a prerequisite for any mechanism label.

The remaining requested paths are sound:

- Certification rho uses tensor-reached states and separates initial/evolved times.
- Nodes-reached rho uses the first eight validation cases, evolved states only.
- Refined and same-grid errors use the correct \(63^3\) lattice. I verified both same-grid archives contain all 64 cases, each shaped `(6, 250047)`.
- Per-case incumbent/converged and nodes/converged distances are saved at lines 387–395.
- `adaptive_first=25` is correct: the vendor solver performs 25 adaptive steps and zero fixed-sweep steps. This changes iteration policy, **not `gtol`**.
- Adaptive records should additionally save finite flags, full reason counts, and iterations; lines 417–424 currently retain only distances and reason-3 totals.

**2. `q2d/qcore.py` and `q2d/qstudy.py`**

`qcore.py`’s diff is correct. [Lines 89–96](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/q2d/qcore.py:89) use the correct mesh ordering and weights \(L^{-2}\), so \(Lw\psi=\Phi\). `ROOT=HERE.parents[2]` resolves to `<job>/code`; vendor rotation/EQ paths also resolve correctly.

`qstudy.py` has these defects:

- **G2a checks four decoded combinations, not equality of bank blocks.** [Lines 131–135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/q2d/qstudy.py:131) can miss bank errors outside the four coefficient directions. DESIGN requires `B_nodes == Grot`.  
  **Fix:** compare bank blocks directly, with bounded chunks.

- **All nodes gates omit the Amendment 3 reference-magnitude guard.** Lines 122 and 343–345 need the same robust gate helper described above.

- **A failed continuum gate still produces `COMPLETE`.** Lines 459–461 compute `passed`, but nothing consumes it. The new nodes-reached check at lines 483–485 is likewise descriptive only.  
  **Fix:** explicitly invalidate the configuration on either required target-check failure.

- **Nodes-reached rho omits the converged arm.** The loop at lines 486–490 follows `rho_rules`, whose generated configuration omits `gref`.  
  **Fix:** add `gref` to that list, even though its continuum discrepancy is zero by construction; its mesh-target discrepancy remains meaningful.

Other suspected issues are **not bugs**:

- Timing skip returns from `run_setting`, not `main`; both settings still execute.
- The first eight nodes-reached cases are dev6 followed by the first two val32 cases, with initial states removed.
- `gref` runs first. Consequently, the dense row contains incumbent-to-converged distance, and the nodes row contains nodes-to-converged distance.
- Use **`vs_gref_restricted_evolved`**, not the full-grid distance, for the registered outcome labels.
- Per-case finite flags, reason counts, and iterations are available.

**3. `a2_gap.py`**

The suspected normalisation error is **not present**:

- [Lines 115–123](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/a2_gap.py:115): dividing `phiT` by \((n-1)^{3/2}\) gives the normalised mesh sum. `continuum_adv(n=2)` sets its scaling factor to one.
- Lines 253–265: `sep_project/L` and `offmesh_data(..., L=1)` likewise both approximate \(\int\psi f\).
- Manufactured leading terms at lines 197–198 and 343–344 have the correct signs and factors: \(-u\Delta u/2\) and \(u\sum_j u_{jjj}/6\).
- The manufactured state is positive inside the domain, so backward-difference expansion applies.
- `split_targets`, concatenation order, and result slicing preserve the fixed/own-state correspondence.

**NEEDS-RESTATEMENT:** this is a data collector, not a completed A2 adjudicator. The file explicitly delegates fitting to `make_report.py`, but no lane-level report generator exists. It therefore does not yet implement:

- independent-family target invalidation at `1e-2`;
- the same surviving state population across every fit-window mesh;
- the 25% survival requirement and numerical floor;
- C-pos slope, leading-coefficient, and finest-mesh vector-agreement acceptance;
- C-neg slope acceptance.

The saved arrays contain most required evidence. Implement those checks before interpreting results. Also copy state labels into the output archives: currently interpretation depends on recovering the separately hashed input archive.

**4. Memory and JIT assessment**

These are allocation estimates, **not measured GPU peaks**.

| Path | Principal allocation |
|---|---|
| A1 3D, `n=129`, `R′=512` | Each `(N,R′)` f64 array: **7.81 GiB** |
| A1 nodes `B+D` | **15.63 GiB** |
| A1 mesh bank | **7.81 GiB** |
| A1 tensor, `M=2049` | **4.00 GiB** |
| A2 3D, `n=257`, current 156 states | Host `U+DU`: **38.54 GiB** |
| A2 2D, `L=4096`, current 246 states | Host `U+GS`: **61.47 GiB** |

A1 [line 87](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/a1_3d.py:87) forms the **entire** Jacobian field before chunking its DST. Thus “column chunks of 16” bounds FFT workspaces, not the full temporary. Building the products inside the chunk loop would remove that large intermediate.

Nevertheless, I found **no demonstrated H200 capacity violation** at `n=129`. The forbidden giant `(N,M)` nodes test matrix is avoided.

A2 streams bank evaluation and batches state projections; it never stores the full `n=257` or `L=4096` bank. Its host fields fit the default **240G** request. Reducing `--mem` needs a separate budget check.

Large solver tables are passed as explicit JIT arguments. I found no newly introduced closure embedding the full nodes bank or test matrix. The 2D model captures network parameters, not full-grid bank arrays.

**5. Configuration, input, staging, and submission**

- **Registered A2 populations are silently halved.** [make_inputs.py:21](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/make_inputs.py:21) selects odd 3D steps; line 44 selects even 2D steps. The actual archive contains **104 instead of 200** fixed 3D states per width and **150 instead of 300** fixed 2D states per setting. Own-mesh populations are also subsampled. Labels are unique and indexing is internally correct; the population is wrong against DESIGN.  
  **Fix:** restore every registered evolved step, or formally amend the population before running.

- **Arbitrary stage names conflict with hardcoded reference paths.** [make_configs.py:49](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/make_configs.py:49) and line 72 embed `/jcpmech/a1d3/` and `/jcpmech/a1d2/`. `stage.py` accepts any alphanumeric job name and stages references under that name. A retry named `a1d3b` therefore reads the wrong directory or fails.  
  **Fix:** resolve references from the actual task root, or enforce matching job names.

- **The 3D smoke configuration retains the 64-case cohort hash after changing the count to two.** `make_configs.py` lines 99–103 conflict with `a1_3d.py` lines 199–201.  
  **Fix:** regenerate the smoke cohort hash. This currently fails after expensive table preparation.

- **A2 cannot pass checksum preflight.** [stage.py:46](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/cluster/stage.py:46) specifies no references; line 126 writes an empty `REFS.sha256`. Generated batch line 113 and `submit.sh` line 23 check it unconditionally. I reproduced exit **1** for an empty checksum list.  
  **Fix:** check reference manifests only when nonempty, at both locations.

- **The reference-copy loop can lose entries.** [submit.sh:18](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/cluster/submit.sh:18) runs `ssh` inside a pipeline-fed `while read` without `-n`. SSH can consume subsequent reference lines from its stdin.  
  **Fix:** use `ssh -n` and prevent other child commands from consuming the loop’s input.

- **Lane cap 1 is not atomic.** Lines 11–15 check the queue before potentially lengthy transfers; line 23 submits without rechecking or locking. Two invocations can both see zero jobs and submit.  
  **Fix:** acquire a namespace-wide lock on shared storage around a fresh queue check and `sbatch`.

The repository-path calculations themselves are correct. Committed-byte comparison, code manifests, reference checksums, node-side verification, GPU preflight, and f64/highest-precision settings are otherwise well structured.

**Will the outputs answer DESIGN?**

They provide the essential per-case distances and primary-solve reason counts for A1. Historical reproduction can also be computed offline from saved coefficients/audit fields.

They do **not yet produce the registered conclusions**. The missing report layer must apply the amended decision order, exclusions, 2,000-resample bootstrap, sensitivity threshold, solver-provisional labels, A2 screens/controls, and missing-case accounting. `complete=True` must mean execution completed—not that scientific gates passed.

**Ranked WRONG items**

1. **A2 launch always fails:** empty reference checksum manifest.
2. **Reference copying can skip files:** SSH consumes the loop’s stdin.
3. **A2 evaluates the wrong registered population:** 104/150 fixed states instead of 200/300.
4. **Required gates are weakened or unenforced:** Jacobian coverage/tolerance, bank equality, magnitude guards, continuum validity.
5. **Lane cap 1 can be exceeded:** queue check and submission are not atomic.
6. **Retry job names resolve references incorrectly:** hardcoded config paths disagree with staging.
7. **3D smoke configuration fails its cohort hash check.**