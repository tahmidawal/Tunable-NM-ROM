import sys,os,json
L=os.getcwd()
def headline(p):
    ls=open(p).read().split('\n'); i=[k for k,l in enumerate(ls) if l.startswith('| mesh')][0]
    out=[]
    for l in ls[i:]:
        if not l.startswith('|'): break
        out.append(l)
    return '\n'.join(out)
prose=open(sys.argv[1]).read()
heat=open(sys.argv[2]).read() if len(sys.argv)>2 and os.path.exists(sys.argv[2]) else None
o=[]
o.append("""# Linear-problem query speed: results

Lane `exp/2026-11-14-linear-query-speed`.  Protocol and dated amendments: `PROTOCOL.md`.  Every number
below is read from the JSON under `runs/`; the full tables (all variants, stage medians, gates) are in
`tables/`.  Comparator numbers (cuDSS, AmgX) are quoted from the sparse-solvers lane for the same GPU type
and were not rerun or altered.

## Poisson 2D: the tuned query is faster than the reused-factorisation cuDSS solve at every mesh, at unchanged accuracy

Timed quantity (headline, column F): nodal source of the whole mesh on the device in float64 -> nodal
solution field of the whole mesh on the device in float64, synchronised; median over the paper's 32-case
test cohort.  This is the like-for-like scope with the cuDSS solve.  The configuration was chosen on
development cases only, frozen, and then run once on the test cohort; the paper path was timed in the same
job on the same GPU (before/after).  Worst test error is identical to six digits before and after.

### NVIDIA H200 (job 7220234)

""")
o.append(headline('tables/poisson2d-h200.md'))
o.append("""

### NVIDIA A100 80GB (jobs 7220235 for 256^2 to 2048^2, 7221131 for 4096^2)

""")
o.append(headline('tables/poisson2d-a100-80G.md'))
o.append("""

Columns: "paper stored (A100)" is the value in the paper's stored results (A100 80GB jobs 4319394 to
4319401, so the paper's Poisson 2D timings came from A100 80GB); "paper path, this job" is the paper's
query function timed in this lane's driver in the same job; S reads the source on a 256-interval
sub-lattice (separate measurement, relies on smooth sources); "solve only" returns reduced coefficients
and no field (separate measurement, not comparable with cuDSS).

Stage medians at 4096^2 on H200, accurate arm (separately compiled and synchronised, so they do not sum
to the fused time): paper path projection + solve 0.996 ms, field reconstruction 17.04 ms; tuned path
projection + solve 0.341 ms, core contraction 0.086 ms, field products 0.247 ms.  All meshes and both
arms: `tables/`.

Gates: stored-error reproduction passed at every mesh with a stored value (largest absolute difference
below 1e-16); the development deviation gate (1e-5) passed for every listed variant; every job log shows
`jax_backend=gpu`.

### Paper-path timing spread at small meshes

The paper path timed in this lane's driver is slower and noisier at small meshes than the stored paper
values, while the tuned variants are tight (H200 256^2 accurate: paper path median 1.335 ms with phase
medians 1.259/1.409 and minimum 0.35; F 0.175).  To check, the unmodified paper driver was run in the
same H200 job:

| mesh | arm | unmodified paper driver, H200 (median ms) | paper path in this lane's driver, H200 | stored paper value, A100 80GB | worst test error (matches stored) |
|---|---|---|---|---|---|
| 256^2 | accurate | 0.734 | 1.335 | 0.565 | 0.008379 |
| 256^2 | fast | 0.482 | 1.014 | 0.381 | 0.040668 |
| 1024^2 | accurate | 1.774 | 2.213 | 3.103 | 0.008350 |
| 1024^2 | fast | 0.961 | 1.399 | 1.177 | 0.040575 |

Both drivers call the same query function; the cause of the extra, variable ~0.5 ms in this lane's driver
was not identified.  The unmodified driver's own timing-stability gate ("neighbour", limit 1.1) failed on
H200 at both meshes (worst ratios 1.185 and 1.105); its error audit passed.  Consequence: at 256^2 to
1024^2 the "F vs paper path" ratios in the tables overstate the gain.  The conservative ratios use the
fastest available paper number as "before": H200 256^2 accurate 0.734 -> 0.175 (4.2x), fast 0.482 ->
0.171 (2.8x); H200 1024^2 accurate 1.774 -> 0.199 (8.9x), fast 0.961 -> 0.188 (5.1x); A100 256^2
accurate 0.565 -> 0.152 (3.7x), fast 0.381 -> 0.136 (2.8x).  At 2048^2 and 4096^2 the in-job paper path
agrees with the stored A100 values within 2 to 7 %.  The comparison with cuDSS does not involve the
paper path and is unaffected.

""")
if heat: o.append(heat+'\n\n')
o.append(prose)
o.append(open('LEDGER.md').read().replace('# GPU ledger (linear-query-speed)','\n## Jobs and GPU-hours'))
open('RESULTS.md','w').write(''.join(o))
