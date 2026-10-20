**Overall: NEEDS-FIX.** Several paths can accept incomplete evidence; the newly scored adaptive 3D path also exposes a reproducible crash.

No files changed, no jobs submitted, no production results opened, and no scratch files created. Checks used CPU-only Python, synthetic fixtures, and the permitted smoke archive.

| File | Verdict |
|---|---|
| `dscore.py` | **NEEDS-FIX** |
| `analyze2d.py` | **NEEDS-FIX** |
| `analyze3d.py` | **NEEDS-FIX** |
| `score3.py` | **READY** for the requested mapping change |
| `make_configs_late.py` | **NEEDS-FIX** |
| `make_round.py` | **READY** |
| `waitjobs.sh` | **READY** |
| `d3run.py` | **READY** for the requested adaptive `Wstore` change |

Ranked findings follow. Paths below are relative to `experiments/jcp-dims/`.

1. **Adaptive 3D comparisons crash once scores exist.**  
   `analyze3d.py:243` calls `all(entry['valid'])`, but adaptive entries store a scalar Boolean (`d3run.py:209`). Thus the B6 adaptive comparisons raise **`TypeError: 'bool' object is not iterable`**. Reproduced by executing the actual `arr` function with the runner’s adaptive schema. Use the per-case validity array or explicitly handle the scalar schema.

2. **Registered populations are not enforced; missing checks can pass Q1.**  
   `dscore.py:64` constructs the expected populations from the archive being checked. Removing registered cases or tighter checks therefore changes the expectation itself. On smoke `R128M128gauss64/LSPG-CN`, deleting every dev-check entry still returned `Q1.pass=True`.  
   `analyze3d.py:71`, `:78`, `:96`, and `:142` likewise lack independent all64/check-0–7 inventory enforcement; the Q2 check silently uses whichever check cases remain. Also, `q1_ladder` uses Python `max`: a synthetic distance list `[0, NaN, …]` passed every Q1 predicate. Bind populations to configs/manifests and explicitly reject non-finite distances.

3. **2D timing qualification omits G-cert and can qualify a subject without Q2.**  
   `analyze2d.py:195` ignores each certification row’s `gcert_ok`. At `:198`, Q2 starts as `True` and stays true when `qp` is absent. Neither exact dev6 certification coverage nor the registered invocation inventory is enforced. A synthetic record with **all G-certs false and no successor** returned `qualified=True, usable=True`. This violates A1.1. Missing successor/check evidence must remain unverified.

4. **3D H2q ignores fresh timing-job Q1 evidence; invalid screening can still produce a front.**  
   `analyze3d.py:335` takes Q1 from accuracy results or defaults it to `True`; it never evaluates the certification blocks generated inside `t3s`/`t3c`. Panel validity is insufficient: `d3run.py:499` links identity, execution validity and finite fields, but does not enforce the numerical certificate/anchor predicates.  
   At `analyze3d.py:352`, front construction also omits `screen_valid`, allowing costs from an invalid screen into the “qualified” front.  
   `make_configs_late.py:67` and `:70` generate tighter anchors only for case 0, even for confirmation cases 0–7. Generate the required anchor population and consume its evidence before selection/confirmation. Finally, `analyze3d.py:345` associates confirmation evidence only by comparator label; verify its actual A/B subjects match the selected pair.

5. **Required independent checks are recorded without enforcement—or never performed.**  
   `analyze2d.py:80` records `rho_offline.agree`, but `:134` qualifies solely from Q1 and ladder Q2. An A6.2 mismatch therefore does not block qualification.  
   `analyze3d.py:71` trusts stored ADT distances; its offline check at `:136` covers successor distance only. The inherited `score3.py:149` ADT recomputation executes only for mode `grid`, not these lane modes. The requested analysis also lacks the A6.2 feasible two-case 3D ρ recomputation and A2.2 independent fixgate recomputation. These checks need explicit acceptance predicates.

6. **Cost Shapley accepts unusable/unqualified timing; elasticity fits the wrong population and has no bootstrap.**  
   `analyze2d.py:249` requires only nonempty `per_case` timing values. It ignores `usable` and timing-job `qualified`, and permits a reduced intersection instead of requiring the six dev6 cases.  
   `analyze2d.py:414` pools all matching schemes, including GAL-CN, LSPG-CN and BE, into the registered LSPG-CN-R elasticity fit. `:418` produces only the point fit, omitting the required joint case-bootstrap intervals. The audited rank/condition guard itself is correctly called.  
   Accuracy Shapley correctly freezes m*, jointly resamples common eligible cases, and uses the A2.5 threshold, but needs an explicit empty-population outcome.

7. **Missing 3D family/spec evidence can become a successful reduced-population m*.**  
   `analyze3d.py:89` skips missing jobs entirely. At `:285`, a lattice family represented only by `m3a` seed 0 is treated as a one-draw family rather than an incomplete five-draw family. Width intersections at `:214` similarly use only present specs, rather than requiring all three registered specs. Build expected ladders first and represent absent draws/specs as missing.  
   Both analyses also pass `m=None` directly to `point_efficiency`, conflating **“not reached”** with **“none within the candidate list”** (`analyze2d.py:338`, `analyze3d.py:317`).

8. **Bad-vector control does not block B3 verdicts; 3D Gauss gate is permanently unavailable.**  
   `analyze2d.py:328` and `analyze3d.py:311` report the bad-vector gate, but the point-efficiency branches never require fixture admission or control firing. An unavailable/non-firing control can coexist with affirmative point-efficiency verdicts, contrary to A2.2/A4.1.  
   Separately, `analyze3d.py:309` seeks `fg['gl64']`; the production family configs do not include that rule in their fixgate blocks. Hence Gauss receives an unavailable state gate and every Gauss comparison becomes inconclusive. Supply explicit target-self evidence. The call to `cross_family_gate` otherwise preserves A3.2 precedence correctly.

9. **LMF stability is checked too narrowly and too late.**  
   `analyze2d.py:389` checks validity only at gauss256, not every LMF run. LMF E0 scores are already computed at `:61`, before the required stability gate. Efficiency verdicts are populated at `:382` and remain present even when `:397` declares “unstable test space (no verdict).” Gate scoring/verdict generation on the complete stability predicate.

10. **Q2 labels diverge from A1.5, and q++ identity is unchecked.**  
    `dscore.py:72` labels any missing successor “ladder top,” including missing interior successors. At `:91`, a second-to-top rung may receive `fail`; the audited state machine gives ladder-top precedence. `analyze3d.py:123` repeats that exception.  
    `dscore.py:85` indexes q++ positionally without validating its case identities or numerical configuration. A synthetic q++ with a different case ID, M and mode hash still returned **`pass`**. Validate both successors against the manifest and address check cases by ID.

11. **B6 control mapping is correct, but blocking is incomplete.**  
    The per-form 2D controls and adaptive-LSPG control for the 3D fixed-sweep panel match A2.1. However, only range labels are blocked: secondary transitions at `analyze2d.py:407` and GAL-vs-LSPG verdicts at `:369` / `analyze3d.py:270` remain affirmative regardless of the applicable control outcome. Preserve raw statistics, but propagate the required blocked/adverse labels.

12. **Registered 3D outputs remain incomplete.**  
    `analyze3d.py:252` implements primary B6 only; the secondary m* comparisons and full requested range summaries are absent. The H2q block at `:347` handles only deployed comparators, omitting the registered qualified BE comparator \(C_w^q\). The front output at `:357` omits the open-candidate inventory.

Checks that passed:

- Every constructed 2D ladder key matched a committed config key.
- Seed-0 naming, MC suffix handling, and adaptive `|prod` lookup are consistent.
- `score3.py` differs from speed3d only by the documented mapping/docstring changes; explicit-M and smoke-prefix mappings passed.
- Adaptive `Wstore` and `Cstore` keys match scoring and analysis expectations.
- H2q delegates endpoint ties and the 3% selection tie-break to the audited functions.
- `make_round.py:34` checks predecessor-member label coverage before selecting another round.
- `waitjobs.sh` passes `bash -n`; no remote execution was performed.