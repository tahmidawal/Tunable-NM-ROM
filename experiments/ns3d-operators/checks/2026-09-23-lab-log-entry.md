
### 2026-09-23 — ns3d-operators: NS 3D against FNO / U-Net / Transolver / DeepONet at 32³, 64³, 96³ (lane closed; all three panels final)

**Where.** Branch `exp/2026-09-23-ns3d-operators` (sparse worktree, forked from `exp/2026-09-23-ns3d-shift-head` @708c70fe), head `261715dc`, **local only, never pushed**. Lane dir `experiments/ns3d-operators/`. Pre-registration and amendments A1–A8 are in `DESIGN.md`. Generated report: `reports/2026-09-23-ns3d-operators.md` (sha256 1aa363c5…), written by `make_report.py`. Combined `reports/summary.json` sha256 `560893bd8e70fd916434f2c75bc1e9d05699416e288cbd4abdc318b005084eee`. Cluster namespace `nsops_20260923` is deleted; every output was pulled and checksum-verified first.

**Ran.**
- **Operators.** 3D PyTorch ports of the Table-2 cells' families: NeuralOperator FNO in f64, and U-Net, Transolver and DeepONet in f32. Each gets two sizes, the same 3000 s wall budget, AdamW, batch 8, a wall-time cosine learning rate, and seed 20260914.
- **Operator data and outputs.** The data are the NM-ROM head's own 512 training trajectories (seed 202609201), with the head's split: index 7 mod 8 is validation (64), the rest train (448). The output is Leray-projected, and that projection is charged in the timing.
- **Training jobs.** 24 jobs, one per (arm, mesh). At 32³ (4232609–4232670) and 64³ (4232677–4232722) all ran on A100-80G. At 96³, five arms ran on H200 (4232892–4232944) and three on A100-80G (4239995/8, 4240001), per the coordinator's direction and disclosed as A4.
- **Size selection.** One size per family, chosen on validation only.
- **Panels.** One allocation and one process per mesh, all on A100-80G: pn32 = job 4234771, pn64 = job 4237885, and pn96c = job 4244423 (on the 32 held-out cases, seed 202609221). Each panel ran the NM-ROM accurate setting (head k=8) and fast setting (span R'=16), the CNAB2 step ladder and all 8 operators.
- **Timing.** Protocol A–B–A with drift and order gates at ≤1.10.

**Found** (worst evolved % / median GPU ms / speedup against the rule FOM, which is the fastest CNAB2 setting at least as accurate as the NM-ROM accurate setting).

| Mesh | FOM (CNAB2) | NM-ROM accurate | NM-ROM fast | FNO | U-Net | Transolver | DeepONet |
|---|---|---|---|---|---|---|---|
| **32³** (development, 16 cases) | 50 steps, 0.094 %, 6.76 ms | 0.151 % / 5.79 / 1.17× | 2.96 % / 3.22 / 2.10× | fno-l: 0.396 % / 3.13 / 2.16× | unet-l: 0.479 % / 15.0 / 0.45× | tsol-s: 0.538 % / 16.3 / 0.42× | don-s: 52.6 % / 3.68 / 1.84× |
| **64³** (development, 16 cases) | 50 steps, 0.093 %, 23.6 ms | 0.152 % / 7.76 / 3.05× | 2.96 % / 4.38 / 5.40× | fno-l: 1.34 % / 11.5 / 2.05× | unet-l: 1.11 % / 32.1 / 0.74× | tsol-l: 1.22 % / 33.8 / 0.70× | don-s: 51.7 % / 17.3 / 1.36× |
| **96³** (held-out, 32 cases) | 70 steps, 0.056 %, 126.8 ms | 0.207 % / 15.8 / 8.01× | 3.24 % / 8.82 / 14.4× | fno-l: 1.51 % / 39.2 / 3.23× | unet-l: 2.29 % / 88.3 / 1.44× | tsol-s: 1.16 % / 19.1 / 6.65× | don-s: 51.4 % / 56.2 / 2.26× |

Reading:
- **64³ and 96³.** The accurate setting is more accurate and faster than every operator.
- **32³.** The accurate setting is the most accurate method, but the FNO is faster and 2.6× less accurate. The FNO also dominates our fast setting there. The accurate setting's 1.17× hangs on a near-miss: CNAB2 at 40 steps reaches 1.004× the NM-ROM error, and against it the accurate setting would be 0.95×.
- **DeepONet.** It never learns this translating family, at 51–53 % error.
- **Gates.** Every gate passed on all three panels: reproduction against the shift-head jobs to ≤3e-10, drift, order, the positive control, timed-output parity, coverage, and bank rebuild. The independent NumPy audit passed, and its perturbed control was rejected.
- **Codex.** A design audit and two results audits: `checks/codex-design-audit.md`, `checks/codex-results-audit-32-64.md` and `checks/codex-results-audit-96.md`.

**Data parity.** Both sides use the same 512 trajectories. The operators take 448 for training and 64 for validation, one pair each (u0, ν → five fields). The head was fitted on the 448 at 21 states each. The bank is the POD of the first 128, which includes 16 of the validation 64.

**Wrong / retracted / failed.**
1. pn96 (4243807) hit a PyTorch OOM in warm-up because JAX held 70 % of the card (A6).
2. pn96b (4244288) hit a JAX OOM because a leftover dict held every timed output of the last A2 round on the device (A7).

   Both are archived under `runs/pn96*_crashed/`. Their accuracy passes agree with pn96c to 3.4e-13, and only pn96c is reported.
3. Codex (design audit, A1) caught three problems before any job ran: a memory probe without optimizer state, a directory race, and interrupted runs being marked complete.
4. Codex (results audits, A3/A8) found that:
   - the timed-output gate in pn32/pn64 compared errors, not fields. Field parity on sample points was added from pn96 onward.
   - saved-field coverage ({0, 1, worst} per arm and 8192 sample points) was less than §5.8 promised, and was not disclosed until A3.
   - the 20 % sampled bound was never enforced, and is now labelled diagnostic.
   - the positive control used A1 instead of A2 (fixed).
   - the report's epoch labels were off (fixed).
5. The 32³/64³ cells are development comparisons: k=8 was chosen on those same 16 cases.
6. At 96³ the training was split across H200 and A100, which confounds Transolver size with training hardware. The held-out cohort has now been opened 4 times.

**Open.** The operators are budget-limited (62–624 epochs for the selected arms). Checkpoints (`runs/tr*/output/best.pt`) and 8 GB of audit fields are local only and gitignored; their sha256 is in each run's `OUTPUTS.sha256`. Ask the user whether to merge this worktree.
