SUBMIT — **f2l**, subject to the shared lane slot.
SUBMIT — **b2m**, subject to the shared lane slot.

No blocking findings. **Only one may be submitted now:** read-only `ssh tufts-login squeue -u tawal01` confirms `t3s 6056422` and `t2 6056624` are pending; adding both would exceed cap 3.

Paths below are relative to `experiments/jcp-dims/`.

- **f2l inventory passes:** 15 settings, matching both LMF manifest ladders exactly; correct successors, LSPG-CNR dt0, all38, and Q1/Q2 populations. Rank/condition recording and pre-scoring stability checks are present. `configs/MANIFEST-cells.json:12303`, `dacc.py:147`, `dacc.py:162`, `analyze2d.py:650`.
- **LMF selection passes:** independently sorted the entire registered mode domain by product, discrete eigenvalue, then \(k_x\). All first 1,536 modes match exactly. The bank builds the policy list and settings take its correct prefix. `p1common.py:115`, `p1common.py:168`, `p1common.py:232`.
- **b2m inventory passes:** 70 settings = two cells × five draws × seven rungs; candidates \(k=11..15\), successors preserve draw identity. Both fixgate blocks include all five finest draws plus target-self evidence. Analysis consumes B3ii and these blocks. `configs/MANIFEST-cells.json:13163`, `configs/b2m.json:1325`, `analyze2d.py:540`.
- **MC generator passes:** explicit `mc<m>s<k>` dispatches to the vendored `uniform_mc` with that seed; recorded generator tests pass. `rules2d.py:27`, `rules2d.py:61`, `checks/test_dims.json:1`.
- **Certificates reproduce exactly:** f2l **0.1464225427 → 0.25 h**; b2m **0.6060593047 → 1.0 h**. Config hashes match; b2m includes **240 seconds for both fixgate blocks**. `resources.py:80`, `resources.py:101`, `checks/resources-f2l.json:6`, `checks/resources-b2m.json:6`.
- **Memory coverage passes:** production table dimensions fit within the smoke envelope, including its largest live successor pair and LMF policy. Recorded smoke peak **29.661 GB**, below 120 GB; production retains memory rejection checks. `runs/smk2/slim/output/dacc_smk2/result.json:10752`, `dacc.py:290`.
- **Staging passes:** all 176 provenance entries per job match HEAD `58125fb6b`; both checksum manifests pass. Scripts specify H200, `gpu`, exit-42 GPU preflight, f64/highest, separate paralab paths, and correct limits. `runs/f2l/PROVENANCE.json:1`, `runs/b2m/PROVENANCE.json:1`, both `run.sbatch:3`.
- **Budget passes:** recomputed spent **28.6212 h**. Including live 20 h and pending t3c 2 h: f2l alone **50.8712**, b2m alone **51.6212**, both **51.8712 ≤ 55 h**. `runs/GPU-LEDGER.json:266`.

No files changed, submissions made, or scratch/array copies created.