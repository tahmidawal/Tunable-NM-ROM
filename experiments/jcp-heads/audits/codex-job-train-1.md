**DO-NOT-SUBMIT as staged.** The resource certificate contradicts its stated estimation rule.

1. **CORRECT — training configuration.** Twelve `mid` heads cover \(K=\{4,8,16,32\}\) × seeds `{0,1,2}`, each for 200,000 steps, plus incumbent identity and the 2,000-step K16/seed0 probe. Architecture, optimizer, initialization and diagnostic settings match the copied recipe. Filenames match downstream `heads/head_mid_k{K}_s{s}.npz` expectations once collected and installed there.

2. **CORRECT — sbatch and staged provenance, subject to item 4.** GPU partition, A100-80G, exit-42 GPU preflight, absolute cluster venv, highest precision, x64, namespace outputs and two-hour certificate agreement are present. All **122 provenance entries match their committed blobs** at `7e60ef6398c63ea3071a1a8edc9b9bdba1523f72`; the manifest verifies. Memory is plausible for sequential training with smoke-tested endpoint dimensions; no production peak-memory bound was established.

3. **CORRECT — input and archive safeguards.** The external coefficient path is SHA256-asserted, as is the incumbent checkpoint. The driver checks all 4,608 training descriptors against dev6/val32 and verifies code/pick alignment. Compact weights are reloaded and checked for zero output difference. Caveat: that comparison uses the in-memory emitted parameters, not a reloaded pickle; it does not independently test all compact metadata.

4. **WRONG — resource projection as documented.** [resources.py:48](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/resources.py:48) divides the smoke rate by two without a documented derivation.

   | Quantity | Value |
   |---|---:|
   | Maximum measured smoke fit rate | 0.00355597 s/step |
   | Certificate rate | 0.00177798 s/step |
   | Current projection / limit | 1.30168 h / 2 h |
   | Unhalved projection / rounded 1.5× limit | **2.48827 h / 3.75 h** |

   The certificate reproduces the implemented arithmetic exactly. The two-hour estimate may be plausible—the historical full fit took 338.59 seconds and short smoke fits include compilation—but the unexplained halving violates “never below the smoke” and does not establish the claimed margin.

5. **NEEDS-RESTATEMENT — outputs versus completed analysis.** The job preserves training metrics, Q-repro oracle inputs and all twelve compact heads; ROM evidence necessarily comes later. However, `analyze.py` does **not report the A1.7 probe/full-budget ratio**. Its Q-repro ROM test also checks only `ratio <= 1.25`, rather than the registered factor band **0.8–1.25**. Both are downstream fixes; the training outputs suffice to repair them without retraining.

6. **NEEDS-RESTATEMENT — recovery and acceptance.** Completed arms are saved individually, but an interrupted arm has no resumable optimizer checkpoint. The unsupported time allowance therefore risks wasted training and an incomplete matrix. Production identity metrics are recorded but not asserted against A1.7’s reference bars; enforce those during acceptance.

**Required before submission:** remove the unexplained `/2`, regenerate the certificate, reserve **3.75 hours**, and regenerate the staged job/manifests—or provide a documented, evidence-based compilation-adjusted estimate and re-audit it. Before accepting results, enforce identity bars and complete the two analysis corrections above.

Audit-process disclosure: I accidentally invoked `make_configs.train()` while checking generation; it rewrote `configs/train.json` with identical content. Git shows no content change, but this violated your no-write instruction. No jobs were submitted; LAB-LOG.md and main were untouched.