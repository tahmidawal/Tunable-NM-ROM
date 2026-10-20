**DO NOT SUBMIT the jobs as currently staged.** A11’s speculative computation with held release is reasonable in principle, but the release implementation does not safely cover recalibration. The 32-case jobs also lack sufficient refinement contingency within 12 hours.

This was read-only: no edits, submissions, or lab-log append. I assessed partial calibration from the committed status and A11’s cited observations; I did not independently verify the `/tmp` evidence.

| Audit item | Verdict | Finding |
|---|---|---|
| Partial calibration evidence | Provisional only | Four of the expected 12 case/kind/level rows are present, all for case 0. Their reported inequalities pass; worst production/proxy ratio is 0.0111. This supports speculative work, not completed calibration. |
| Production gate and tagging | PASS for fresh staged jobs | `calib_status_ok` requires passing proxy status and matching `FLOOR_STAB`. It returns `provisional`; the job result and newly generated case metadata retain that label. |
| Hold before collection | PASS on the normal fresh-job path | The collector skips provisional jobs **before** linking files or adding results. Their sealed QC is also excluded from summary/E0 processing. |
| Final-status transition | **DEFECT** | The calibration writer omits `provisional`; the collector uses `final.get('provisional', True)`. Copying the generated final status leaves production held indefinitely unless someone explicitly adds `false`. |
| Recalibrated release | **BLOCKER** | With explicit `provisional: false`, collection checks only `proxy_ok_all`. It does not compare calibration dimension, factor, evidence identity, or require re-judgment. A final passing factor of 100 releases outputs judged with factor 10 unchanged. |
| Case-level protection | **GAP** | Collection checks the job-level label, not each case’s label. Resumed files retain old metadata while the job receives the current status. Neither STATUS construction nor loader eligibility independently enforces calibration release. |
| Existing-store protection | **GAP** | Refusing a job does not remove already-linked files. STATUS scans the existing store; the loader checks hashes/status/outcome, not calibration validity. Thus the hold is an ingestion gate, not revocation protection. |
| Recompute floor from saved data | PASS, with qualification | Every attempt stores lattice fields, `res_*`, `nwt_*`, stage count and other judgment inputs. These suffice to recompute $\tau=C\sum_n\max(r_n,10^{-14})$, differences, floor checks and outcomes. |
| “No rerun needed” | **Too broad** | No PDE rerun is needed to re-judge existing attempts. But increasing the factor can newly trigger the mandatory tight ladder or alter refinement decisions. Missing attempts require solves. |
| Sealed re-judgment | Feasible, not implemented here | Residuals are inside encrypted artifacts. Re-judgment must run under cluster custody and regenerate QC and integrity records; the locally pulled aggregates are insufficient. |
| Sealed publication | Separate hold gap | `publish_sealed.sh` can publish provisional ciphertexts into the remote sealed archive without checking calibration. This does not itself make them loader-eligible, but contradicts an unconditional “nothing enters the store” guarantee. |

The decisive code is in [the collector release gate](./experiments/jcp-references/collect.py#L67), [case generation and stored evidence](./experiments/jcp-references/refjob.py#L102), [the final calibration writer](./experiments/jcp-references/refjob.py#L407), and [loader eligibility](./experiments/jcp-references/refstore.py#L139). I evaluated the actual collector condition in memory: **current provisional → HOLD; writer-shaped final → HOLD; explicit final with changed factor → RELEASE; wrong-dimension explicit final → RELEASE.**

The calibration margin also does **not** establish that production tight attempts will be rare. Calibration compares production–tight differences with $\tau$; production floor acceptance compares mesh differences with $10(\tau_i+\tau_j)$. Those are different tests.

| Staging/resource item | Verdict | Finding |
|---|---|---|
| Coverage | PASS | Exact, duplicate-free coverage within every cohort: dev6, val32, test64, sealed128 and ood32. Both ST and S are configured throughout: 262 cases, 524 case-kind outputs. |
| Levels and tolerances | PASS | ST: 2048/4096/8192 with steps 0.00125/0.000625/0.0003125; refinement 12288 at $0.05/240$. S: same meshes, fixed 0.005. Production/retry/tight settings match DESIGN; tighter budget is 8 under A9. |
| Sealed configuration | PASS | Only custody paths and committed plaintext/ciphertext hashes appear, alongside public case indices. Hashes match `checks/sealed_seeds.json`; no plaintext seed or key is staged. |
| Provenance/manifests | PASS | All 12 manifests verify. Every provenance entry matches its staged bytes and git content at `8233a3feed4db8dd2826ca022a51229a7a58a3ba`. |
| Batch setup | PASS | All request A100-80G, 120G host RAM, 12 hours, GPU partition; absolute venv, GPU refusal preflight, f64/highest precision, isolated paralab paths and checksum checks are present. |
| Submission controls | PASS by inspection | Helper enforces the four-job lane cap, local locking, exclusive remote directory creation, and queue checks before/after submission. No live submission preflight was performed. |
| 12288 memory | **Plausible, unverified** | One interior state is 1.208 GB; six outputs are 7.247 GB. Those fit easily, but they are not peak memory. Newton/BiCGStab vectors, odd-extension complex FFTs/workspace, cached solver arrays and output padding coexist. The 0.92 allocator fraction further reduces usable capacity. No measured 12288 peak was available to certify the full solve. |

For budgeting, using the supplied ST measurements and assuming S scales approximately with mesh area:

- Base case, **both kinds**: $282+36.5+5.3+100(1+1/4+1/16)=455.05$ seconds.
- Additional refinement, both kinds: approximately $282(1.5^3)+100(1.5^2)=1176.75$ seconds.
- These are execution estimates, **excluding** compilation, gates, IC construction, serialization, retries and tight attempts.
- One tight 8192 attempt adds approximately **3 hours**. A tight ST refinement could plausibly approach **10 hours** under the same scaling; this is an extrapolation, not a measurement.

| Job | Cases | Base hours | Hours if every case refines both kinds | Verdict |
|---|---:|---:|---:|---|
| `fd2ddev6p0` | 0–5 | 0.76 | 2.72 | **DO NOT SUBMIT** — common release blocker |
| `fd2dval32p0` | 0–15 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dval32p1` | 16–31 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dtest64p0` | 0–15 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dtest64p1` | 16–31 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dtest64p2` | 32–47 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dtest64p3` | 48–63 | 2.02 | 7.25 | **DO NOT SUBMIT** — common release blocker |
| `fd2dsealed128p0` | 0–31 | 4.04 | 14.50 | **DO NOT SUBMIT** — release and walltime |
| `fd2dsealed128p1` | 32–63 | 4.04 | 14.50 | **DO NOT SUBMIT** — release and walltime |
| `fd2dsealed128p2` | 64–95 | 4.04 | 14.50 | **DO NOT SUBMIT** — release and walltime |
| `fd2dsealed128p3` | 96–127 | 4.04 | 14.50 | **DO NOT SUBMIT** — release and walltime |
| `fd2dood32p0` | 0–31 | 4.04 | 14.50 | **DO NOT SUBMIT** — release and walltime |

A 16-case job with all refinements plus one tight 8192 attempt reaches about **10.25 hours before overhead**; two such attempts exceed 12 hours. A 32-case job with three tight 8192 attempts exceeds 12 hours even without refinement. Sealed jobs cannot resume after ciphertext appears, making their timeout exposure especially costly.

Before approval: make final calibration release explicit and bound to the factor/evidence used for each judgment; implement recalibration re-judgment with missing-attempt handling and sealed QC regeneration; resolve the 32-case walltime exposure; and establish a credible full-solver 12288 memory budget or measured peak.