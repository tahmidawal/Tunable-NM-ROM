# Phase-9 terminal-recovery pre-registration

This is a prospective, one-cell infrastructure recovery of terminal metrics from
the exact final work checkpoint of failed Phase-9 T1 job 2702357.  It is not a
retry, performs no optimizer update, cannot complete the original Phase-9 audit,
and cannot license G2/T2.  It is independently licensed by valid P4--P7 evidence,
immutable descriptive P8 evidence, and the checksummed post-update failure.

## Immutable provenance

The recovery cell is `p9_terminal_recovery_s11_r1`, on exactly one H200 with 8
CPUs, 96 GiB host memory, f64, highest matmul precision, and a 16-hour limit.
It binds the following P4--P8 artifacts before reading the work checkpoint:

| artifact | SHA-256 |
|---|---|
| P4 JSON / NPZ / AUDIT / manifest | `ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617` / `720c5890b22709c83858f46e43f18fa3d28bb1305544f02ad9aea71625a2228f` / `f41010b72ad9ddae43409a1c1d2073dc2839edea22e57a7d5bafc8e2359e08a3` / `67a3bf8d0d8c755ec39cd4056a0bca0e852b4ba17493e8f052a0b288e63a201a` |
| P5 JSON / NPZ / AUDIT / manifest | `97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96` / `5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54` / `c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff` / `6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6` |
| P6 JSON / NPZ / AUDIT / manifest | `9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a` / `9f0372daba8c12e3aff86efde3201d3ae0612aba8cd297aa37559aa286967381` / `9e017b37709bf37fc8c8b87bbbb70461cba8afb47603a5b65dd2618c901ad2b4` / `f8932a6a4304a14b93bfdf6783e900a47a9a1d45ba03f9915941bb00770bfb40` |
| P7 JSON / NPZ / checkpoint / AUDIT / manifest | `a59e92640aae787003d5753d4614de1b6fe90c68fdba7f2daa81a125a3c0956c` / `fd40d339c0746b48c07408d5d017dff8819595350c365f7af5fc0d40673946ae` / `113101637ef2b4fb75fe2ca0ba0dabe5a90c35d03a5c563b765438385c62db8c` / `35c95ee38dea7622f40b6c199f4164b6c27ec3f37ad5561a51de6cd3b0a79322` / `a95fc4621a90cef13071df1ad1db363187deac0f7fc7c8c774c0fedd7c9c3a19` |
| P8 JSON / NPZ / AUDIT / manifest | `be0ca15e5c36f45a1f9a8fdfe86b79a592b530c069ae6d25c1f573c59c974f4f` / `93f8eabcaeaafd419e868a3a7f74ddbd84429c23345570a5f46d5fa38af2471f` / `d1e9e7bd238ac9e9320afc018fa5ff85afcd51d6ff714a61eb47c3b32900de17` / `1af89411e6a37c1106fa9285e50468bd936e34a7ab15b070b06d1a82846416ad` |

The failed-run bundle is bound file by file: FAILURE `0810269919ab1d4c9dd315f64e666cb258620b11faaea1cf302c5cd9ba88e9db`,
LOCAL `46b3d681fb0927368b8696acc800736691ef8f3ddade65363689b14bfb436ac9`,
original manifest `a403b5facfa4e6cd3b6a14abdb3503c1e85721408182419e2e197a05a51bfa1e`,
REMOTE `2426d43463506065706ca1a76b0ccc064df6a971a02be4063d1c64e72487eaf1`,
SACCT `98292764147d97708b47ebf50d18f16df29fb6db3cc166a09b74ae9c1a474cf7`,
stdout `ea0ad35af5d8b5bfd94176e40ee41b8a91fcba50cf2733d57ade352ddcc73fbb`,
stderr `d5428105ef196548ad927f13e1cb1e6ad8dc7e1fe176dde8ebc938df743ea5ce`,
PROGRESS `1aca12633096dd0dd75fcf5da353b3500b2dac7d77055b3112a8a32f1cae53b5`,
and work checkpoint `dd7b5fecc07e9021c6264febf5c7187004daaa73a50381921d5190eede639ce0`.
The original commit is `4a87fe325ce07ce88cdb0995375a614fa864bf7a` and job is
`2702357|p9_t1_s11_r1|FAILED|1:0`.  The work checkpoint must say predictor
epoch 18 and global update 528768.

## Locked recovery

Only the training populations `(N,start,count)=(64,0,512),(128,0,128),
(256,0,64)`, each with 51 snapshots, are regenerated.  All 16 immutable P5
target chunks, their global order, reference health, physical-to-normalized
affine mapping (`abs<=2e-15`, no relative tolerance), and feature provenance
are rechecked.  Coefficient mean, two head RMS scales, and predictor feature
mean/scales are recomputed from train only.  Selection targets or fields,
model-validation, confirmation, weak/EQ, scaling, and any downstream data stay
sealed.

The input work checkpoint is read-only.  Its generator (30,594 parameters),
encoder (29,811), predictor (2,104), q_raw `(35904,19)`, final target states
`(35904,24)`, evaluation states `(35904,24)`, and predictor optimizer are
required finite and tree/hash bound.  The first five final-target components
must equal regenerated affine states and the remaining 19 must equal
`tanh(q_raw)` exactly.  The train-only folded predictor must reproduce the
stored evaluation states to componentwise absolute tolerance `2e-15` with no
relative tolerance, and its standardized/unstandardized identity must be at
most `1e-12`.  The source checkpoint hash is checked both before and after.
No missing encoder/joint optimizer, update order, preflight, encoder-handoff,
or epoch history is invented.

There are exactly zero optimizer updates, zero schedule/permutation steps, and
zero state changes.  The driver persists `optimizer_updates=0`, before/after
tree hashes, and immutable-source hashes.  Any mismatch makes the recovery
invalid; it does not create an alternative scientific result.

## Metrics, gates, and decision

For every regenerated train snapshot the sole scientific field is the Cox
full-grid decode.  The primary snapshot loss is
`||u_theta-u_FOM||_2^2/max(||u_FOM||_2^2,1e-300)`.  For each trajectory, sum
the 51 snapshot numerators and denominators before taking the square root.  At
each N and pooled across trajectories, report the arithmetic mean and maximum
trajectory error.  K3 is a separately charged terminal identity control only.
Health requires finite arrays, zero exact-boundary violations, reference
residuals no larger than `1e-8`, and K3/Cox relative identity at most `2e-14`.
The unchanged full-train gate requires mean `<=2e-4` and worst `<=7e-4` at
every N and pooled.

The original control flow reached capacity only under healthy train failure.
The recovery is valid only if it independently reproduces `train_health=true`
and `train_pass=false`.  It never evaluates selection, even if a corruption
would appear to pass the train gate.

The fixed final q is also evaluated with the exact Phase-9 full-cohort
stationary/tangent diagnostic, persisting gamma, eta, residual/tangent energy,
bound fraction, CG iteration/relative-residual/breakdown and convergence
classification at every N and pooled.  These are descriptive final-only
numbers.  Because joint epochs 24 and 27 were overwritten before the crash,
`capacity_license_complete=false` and `g2_licensed=false` unconditionally.
No reconstructed late-improvement value, T2 license, promotion, or complete
original Phase-9 result is permitted.

## Independent audit and cap

The independent auditor revalidates source/dependency/manifest/Slurm/backend/
precision provenance, regenerates train data and normalization, recomputes the
fold and full Cox/K3 field metrics from the recovered checkpoint, and
recomputes the full final-only capacity arrays and summaries.  It recomputes
all gates and the forced-negative license rather than trusting labels.  Fixed
capacity floating-work arrays are compared componentwise at `rtol=5e-13,
atol=5e-13`, except that CG-relative residual uses absolute tolerance `1e-12`
and the unscaled CG solution vector uses the fixed global scale-aware bound
`max_abs_delta <= 1e-12*max(1,max_abs_either_delta)`; Boolean
CG classification remains exact, while independently repeated iteration counts
must lie in `[0,38]` and differ by at most one (the excluded synthetic case can
cross the stopping threshold one iteration apart).  Both copies must
independently satisfy the unchanged `1e-12` convergence rule.  These tolerances
are prospectively fixed from excluded synthetic repeats (largest delta globally
scaled difference `6.5e-13`, largest CG-relative absolute difference `6.9e-13`)
and do not relax a scientific or licensing gate.  Fixed
corruption tests must reject work-tree/q/state/fold mutation, nonzero update
count, opened selection, field metric/boundary corruption, capacity-CG
classification corruption, and any true/complete capacity license.  A local
smoke may use only synthetic data under 60 seconds via the mandated GPU wrapper;
it is excluded.  The total cap is one recovery cell.  No training retry, T2,
weak/EQ, scaling, model-validation, or confirmation cell is opened.
