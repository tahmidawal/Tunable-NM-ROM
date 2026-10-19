1. **CORRECT — minimal change and terminal exclusion.** At HEAD `870bebfab`, `tight_base=True` preserves existing configs’ solver behavior. Only the base tight retry is gated; refinement logic is unchanged. Case metadata and `anaref` record the flag, defaulting missing historical values to `True`. `anaref.py:155–163` rejects every non-accepted T=1 case in every window; `ana7.py:161–180` propagates terminal exclusion into W1/W2/W3. Isolated, read-only checks confirmed rejection despite otherwise passing eligibility gates.

2. **CORRECT — coverage under the stated cancellation plan.** Partial `ref2t1` supplies dev6 0–5; `ref2t1w` supplies val32 0–9 and 19: exactly **17 unique cases**, no omissions or overlap. The original sbatch still contains the val32 step, so non-overlap depends on the promised cancellation. `cluster/collect.py:57` checks all promotion destinations before copying and refuses existing files with exit 9.

3. **CORRECT — local staging and resources.** All **159 provenance entries** match HEAD byte-for-byte; all **163 manifest entries** verify. The certificate’s config hash matches. Sbatch requests H200, 240G, `gpu`, f64/highest precision, GPU-backend preflight, and dedicated paralab output/cache/tmp paths. **8.25 h** matches the reservation and exceeds **1.5 × 5.435331 h = 8.152997 h**. The retained tight-path reserve is conservative despite disabling that path.

4. **CORRECT — no additional blocker found.** This verdict covers local artifacts and the stated cancellation plan. Remote state and cancellation were not verified, consistent with the no-SSH constraint. No files were changed and nothing was submitted.

**SUBMIT**

Blocking fixes: none.