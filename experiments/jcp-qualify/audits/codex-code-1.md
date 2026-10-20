**NEEDS-FIX.** No files changed, no jobs submitted, no GPU execution. Ran small CPU checks without bytecode writes; all 20 existing fixtures pass, but adversarial checks expose failures below. Paths are relative to `experiments/jcp-qualify/`.

Ranked must-fix items before smoke:

1. **WRONG — timing memory exceeds the pilot’s model.** `qtime.py:201` drops Q2 tables only after **all subjects** finish certification; every setting retains its successors meanwhile. `qpilot.py:55` builds/drops extra rules individually. The pilot therefore does not measure the actual certification footprint required by A2.4.

2. **WRONG — pilot acceptance looks in the wrong directory.** `smoke_accept.py:139` expects `output/qpilot_smkB`; `cluster/stage.py:80` names output by config stem, producing `output/qpilot_pilot` for `configs/pilot.json`. A successful pilot fails archive discovery.

3. **WRONG — mandatory budget reservations are optional and become double-counted.** `ledger.py:57` treats missing `pending_core` as zero; `reserve`/`bind` never removes submitted jobs from that map (`ledger.py:73`). Subsequent submissions count them as both live and pending. No enforced reservation inventory or fallback lifecycle implements A3–A5.

4. **WRONG — non-finite certificate evidence can pass.** `qcert.py:92`, `qscore.py:43`, `qscore.py:46`: finite certificate plus infinite recorded value passes because `inf <= inf`. CPU reproduction confirmed this for both GAL and LSPG offline G-cert. `qcert.py:81` and `qscore.py:29` also certify `-inf`. Require explicit finiteness.

5. **NEEDS-FIX — smoke controls are incomplete or bypass the code they purportedly test.**
   - `smoke_accept.py:165` omits `test_p1.json`: inherited statistical and identity-swap fixtures are not gated.
   - `test_q.py:24` never tests exclusion of an unqualified dominator through `front2d`.
   - `smoke_accept.py:125` checks split invocations but ignores fused-triplet failures and doubled-work `drift_ok`.
   - `smoke_accept.py:62` checks reported under-resolution thresholds, not the actual Q2 classifier.
   - `make_configs.py:101` gives R512 only dev2, contrary to §7’s preliminary dev6 reproduction gate.

6. **WRONG — required gates do not block analysis verdicts.** `analyze.py:157` records combined reproduction failures but still qualifies configurations and builds fronts. `analyze.py:225` records TH06 failure but retains B4 labels; it never checks agreement with smoke TH06. `analyze.py:388` similarly records the 3D control without blocking its front. Numerical-manifest equality is not enforced before inherited/reproduction acceptance.

7. **WRONG — fallback configurations cannot be represented correctly.** `analyze.py:41` omits rule/fallback identity from keys; `analyze.py:144` loads only fixed `acc<N>` attempts. Original/fallback entries collide, and split/repeated attempts are unsupported. `make_configs.py:62` reuses `arms_for`, so fallback mode still skips inherited arms and retains ext22-only populations instead of full38.

8. **WRONG — ext22 Q1 is not the registered pooled predicate.** `analyze.py:104` requires ext22 to pass independently before checking the union. Counterexample: 12 non-certified steps among ext22’s 1,100, zero among P1’s 800: pooled **12/1,900 passes**, but this implementation rejects it.

9. **WRONG — 3D acceptance trusts incomplete evidence.** `analyze.py:405` consumes recorded distances/rho without independent offline recomputation or required population assertions. At `:430`, missing reproduction evidence does not block passing; own-run invalidity is merely reported. A CPU fixture with missing reproduction and an invalid own run returned **pass**. Archive key formats themselves are correct; all 17 registered keys exist.

10. **WRONG — open-candidate list is too small.** `analyze.py:396` requires a candidate to be undominated on **both** median and worst panels. A candidate dominated only on median can still enter the qualified worst front and must remain open.

11. **WRONG — role labels use the wrong uncertainty and timing eligibility.** `analyze.py:367` uses median uncertainty for both statistics, instead of uncertainty of their defining cases. `analyze.py:356` also admits unusable/unqualified timing into role comparisons.

Other gaps:

- **NEEDS-FIX:** `analyze.py:487` never writes the promised `runs/EXPORT.json`.
- **NEEDS-FIX:** `analyze.py:477` omits A1.5’s accompanying cited P1 transition verdicts, including the explicitly labelled 4096² 16-case comparison.
- **NEEDS-FIX:** `PROVENANCE.json` declares `resources.py` unchanged, but its hash differs.

**CORRECT in the inspected paths:**

- `qacc.py:55–76`: form-aware query cache, schedule sizing through 6,400 steps, tolerances and wrong-mesh guard.
- `qcert.py:25–52`: GAL residual/history/start-up/normalisation matches the solver. Independent tiny CPU calculations agreed within \(1.4\times10^{-16}\); zero-scale overflow G-cert rejection works.
- `qacc.py:209–226`, `q3run.py:39–60`: successor replay, own-state rho and qpp subset indexing.
- `qscore.py:178`: ordinary pooled certificate, every-case distance and tight/tighter predicates; Δt swap raises.
- `qstats.py:17–60`: pairwise dominance and B4 classifier/formulas.
- Config JSONs match regeneration: 108 total configurations including inheritance, ext22, registered larger steps, full TH06 dev6 and order ladders; 436 3D candidates and exact round-one selection.
- `cluster/submit.sh:23–24`: lane cap 3/account cap 8; namespace, job prefix, GPU preflight and precision settings are correct.