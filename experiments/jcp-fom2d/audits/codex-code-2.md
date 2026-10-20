**NOT READY to submit `runs/smk1` as the C1 acceptance/resource smoke.** The staged job now imports successfully; I found no remaining deterministic launch failure. However, its offline smoke matching crashes with the supplied P1 targets, and its pilot measurements do not reliably provide the promised compilation/resource evidence.

Audited HEAD **`7e950303146f8176dd936c99c5a57d0355b4965f`**. All paths below are relative to `experiments/jcp-fom2d/`. No files changed, no cluster commands executed, and no scratch files created. Newly appearing untracked implementation files are outside this verdict.

**Ranked findings and fixes**

1. **Smoke matching crashes on the supplied P1 targets.**  
   `match2.py:339` derives a six-case smoke population, but `match_target` passes the unchanged P1 targets to `common`, which indexes `sc[c]` for their `val32` cases (`match2.py:157–163,261`). I reproduced **`KeyError('val32|0')`** using the actual saved target file and smoke case coverage.

   Create explicitly labelled smoke-population targets, or intersect them with the smoke population before matching and running controls. Keep the production 38-case requirement intact.

2. **Compilation evidence from the smoke pilots is incomplete.**  
   `ladder.py:139,162–164,243–309` uses the same cached `FOMs` instance before the pilots. Several pilot solvers have therefore already compiled. Nevertheless, `ladder.py:334–340` estimates compilation as first pilot call minus second pilot call. For those solvers, this measures two warm executions and can record zero compilation.

   This affects actual smoke workloads, including CC resolutions exercised by G-CT and UB512 exercised by production. Record compilation when it actually occurs, keyed by executable, and carry it into the certificate. The inherited solver already exposes compilation duration (`refcore.py:180–190`).

3. **Expected divergent configurations can falsely block the entire ladder.**  
   Nonfinite in-job scores become JSON `null` through `ladder.clean` (`ladder.py:39–45`). `match2.py:78` then assigns infinite offline disagreement whenever the saved score is `None`; `match2.py:341,357` blocks every match.

   Thus an honestly divergent PS configuration—explicitly allowed and expected by DESIGN—poisons otherwise valid comparators. Distinguish “both scoring passes found an invalid/nonfinite result” from a disagreement on valid evidence. Reject the configuration, not the whole campaign.

   This is a code-path finding; the small PS executions I ran remained finite.

4. **Resource certificates can be issued from a rejected smoke and omit required costs.**  
   `resources.py:67` checks `complete`, but never `accepted`. Its projection (`resources.py:40–57`) ignores measured input preparation, explicit burn-in costs, and resource coverage for extension resolutions. The smoke has no CC16, UB32 or PS4 pilot, although those are possible A2.5 extensions; rate lookup would fail for them. `cluster/stage.py:46–50` also does not verify the certificate’s GPU identity.

   Require accepted smoke evidence, validate usable pilot measurements, include measured preparation and burn costs, and define how extension jobs receive certificates. Check certificate/device compatibility during staging.

5. **Production target identity and FRONT ingestion remain incomplete.**  
   `targets.py:101` still exits for FRONT input. `targets.py:40–41` omits bank identity and explicit solver/tolerance identity. Distinct configurations can consequently collide in `with_populations`, which overwrites keys at `targets.py:54,63`.

   This blocks production matching/C2, rather than the numerical smoke itself.

6. **Extension bookkeeping is only partially implemented.**  
   `match2.py:358–364` emits triggers, but does not represent the required added configurations or completion of the single extension round. After extension, it derives triggers from the new winners rather than preserving the original base-ladder decisions. A successful extension can therefore erase the trigger that required it.

   Persist the base decisions and explicit required/completed extension state before building C2 configurations.

7. **Remaining accounting weaknesses.**  
   `ledger.py:130–136` still assumes the last accounting row is the current attempt. An earlier terminal row appearing last can release a live job’s reservation; subsequent reconciliation skips terminal records (`ledger.py:119`). Automatic requeue is disabled, which reduces exposure but does not repair this logic.

   The account-cap mitigation is also still non-atomic (`cluster/submit.sh:41–46`), as A4 acknowledges. Moreover, remote `squeue | wc -l` pipelines can conceal a failed `squeue` (`cluster/submit.sh:25,43`). Establish attempt ordering and make these queries fail closed.

**Staged smoke and its questions**

| Check | Finding |
|---|---|
| Staged provenance | All **106** provenance entries match committed blobs; manifest has no mismatches; staged commit equals HEAD. |
| Imports | Staged `ladder`, matcher, resource and gate imports succeed. `engines` resolves inside the staged tree; missing `sep_common` is fixed. |
| Batch setup | Shell syntax passes. Absolute cluster interpreter, H200, GPU partition, `--no-requeue`, f64/highest precision, GPU preflight and paralab paths are present. Remote environment/store availability was not tested. |
| §6 gates | All seven are requested. Acceptance checks each result and G-PS1’s half-step control (`ladder.py:345–348`). |
| A1.5 controls | K-loose and fixed dev6 `0→1` K-swap are wired. K-syn/K-false remain offline; the smoke/P1 population crash prevents completing that workflow unchanged. |
| G-CT coverage | All five registered relationships/workloads are present (`ladder.py:162–166`). |
| A1.6 pilots | All **60 FD tolerance/resolution pilots**, **11 base PS pilots**, and the informational **N1024 PS pilot** are present. Compilation accounting remains defective. |
| Production configs | Exactly match `make_configs.py` and A1.1: **48 CC groups, 42 UB groups, 7 UB4096 groups, 88 PS configurations**. |

Small CPU checks confirmed:

- G-FD1 passes.
- Nondefault-tolerance G-FD2 at L32 gives **zero field difference** and identical Newton counts.
- K-false fires for all **48 saved P1 target variants**, and for the round-1 heterogeneous-uncertainty counterexample.
- Saved targets use the active E0 store version.

These checks do **not** establish the registered GPU gate outcomes, GPU bitwise reproducibility, runtime or memory sufficiency.

**Every round-1 WRONG / NEEDS-FIX disposition**

| Round-1 item | Status | Current evidence |
|---|---|---|
| Staging: missing `sep_common` | **FIXED** | `cluster/stage.py:18,63`; staged import succeeded |
| Staging: unrestricted smoke exemption | **FIXED per A4** | `cluster/stage.py:43`; now requires smoke-named configs as well as attempt |
| Staging: non-H200 permitted | **FIXED** | `cluster/stage.py:36` |
| FD per-stage evidence missing | **FIXED** | `fomrun2.py:172–176` |
| G-PS1 empty fixtures pass | **FIXED** | `psjax2d.py:80` |
| G-FD2 unequal nondefault tolerances | **FIXED** | `gates.py:50–55`; CPU checked |
| G-CT empty specifications pass | **FIXED** | `gates.py:123` |
| Ladder ignores gate/control failure | **FIXED** | `ladder.py:345–358` records acceptance separately from completion |
| Resume mixes run identities | **FIXED for staged execution** | `ladder.py:114–118` |
| Nonatomic checkpoint / unchecked production archive | **FIXED** | `ladder.py:128–131,238–241` |
| Undeclared numerical identity fallback | **FIXED on supplied configs** | `ladder.py:190,209–214`; configs use bitwise policy |
| Unchecked warm-up / timing coverage | **FIXED on emitted records** | `ladder.py:198–212` |
| K-swap unspecified case pair | **FIXED** | `ladder.py:326–328` |
| Pilot burn/compile/per-stage evidence | **PARTIAL** | `ladder.py:334–340`; compilation already consumed by earlier work |
| Matcher ignores acceptance/disagreement/controls | **PARTIAL** | `match2.py:334–357`; now blocks, but falsely blocks expected nonfinite configurations |
| Matcher trusts convergence/timing summaries | **FIXED for normal producer output** | `match2.py:85–95,141–153` |
| Production population inferred from records | **FIXED** | `match2.py:326,339`; smoke path has the new target-population crash |
| Store versions not compared | **FIXED** | `match2.py:333–338` |
| Minimum-three uses wrong population | **FIXED** | `match2.py:261–263` |
| K-false heterogeneous uncertainty | **FIXED** | `match2.py:220–228`; counterexample and P1 targets checked |
| Empty targets pass / small targets crash printing | **FIXED** | `match2.py:354–355,368–370` |
| “Projected cost” label | **FIXED for PS** | `match2.py:282–283` |
| Step-edge / extension state missing | **PARTIAL** | `match2.py:180–190,358–364`; extension completion still absent |
| C1/C2 selection distinction; PS anchors missing | **PARTIAL** | Distinction fixed at `match2.py:269–270`; anchor exists, but uses adjacent PS initial-field norm instead of reference \(n_0\), `match2.py:247` |
| P1 comparison agreement not asserted | **FIXED** | `targets.py:81–83` |
| Empty eligible target population | **PARTIAL** | Helper handles it at `targets.py:33–37`; P1 caller still subtracts/rounds `None`, `targets.py:83,88` |
| FRONT / complete identity / H-X populations absent | **PARTIAL** | H/X added at `targets.py:44–64`; identity incomplete and FRONT unimplemented |
| Nonpositive/nonfinite reservations | **FIXED** | `ledger.py:79–80` |
| Retry-aware reservation accounting | **PARTIAL** | Pre-reservation queries at `ledger.py:61–75`; no-requeue at `cluster/stage.py:89`; ordering issue remains |
| Accounting query status/order | **PARTIAL** | Return status checked, `ledger.py:124`; last-row assumption remains, `ledger.py:131` |
| Account-cap race | **PARTIAL, disclosed by A4** | Post-submit cancellation mitigation, `cluster/submit.sh:41–46`; no shared atomic cap |
| Collection cannot retry | **FIXED** | `cluster/collect.py:44–48` |
| Cleanup lacks expected-step coverage | **FIXED** | `cluster/collect.py:29–32` |
| Smoke missing FD tolerance pilots | **FIXED** | `configs/smk1.json:103–162` |
| Burn-in adequacy unestablished | **PARTIAL** | Config now uses 0.1 s (`configs/smk1.json:58`); adequacy remains a GPU measurement question |

**C2: obvious defects only, for the later audit**

| Round-1 item | Status |
|---|---|
| G-RR accepts missing target errors | **FIXED for non-A targets**, `c2time.py:196–197` |
| Wrong selection population/ties/union shortlist | **FIXED in the normal six-case path**, `c2time.py:270–298` |
| Identity-vetoed candidates remain selectable | **PARTIAL**: pre-existing vetoes filtered at `c2time.py:273`; vetoes arising during stage 1 are not removed before selection at `c2time.py:288–298` |
| T-id vetoes production subject | **FIXED** through control alias, `c2time.py:343–357` |
| A as a target lacks stage-2 measurements | **NOT FIXED for A’s own label**, `c2time.py:309`; a separate alias would need explicit configuration handling |

Additional obvious later-stage defects:

- `c2analyze.py:53,132,145` retains unqualified “pass” and **“does not fire”**, contradicting A3.1.
- T-fault records separation but acceptance ignores it (`c2time.py:342`; `c2analyze.py:95–97`).
- ROM scoring validity is recorded but not enforced by G-RR (`c2time.py:189–197`).
- FOM inputs/executables accumulate rather than following A1.6’s subject-stage release policy (`c2time.py:127–137`).
- Only the 1× crossover classification is constructed, not the required 2× bracket (`c2analyze.py:151–168`).

Deliberately fabricated acceptance flags, timing coverage fields or amendment-policy settings could bypass further checks. Those are **NOTED, nonblocking**, under the lane’s stated rule.

Fix the smoke-population path and compilation evidence first, repair nonfinite-score handling and resource certification, then restage. **Current verdict: NOT READY.**