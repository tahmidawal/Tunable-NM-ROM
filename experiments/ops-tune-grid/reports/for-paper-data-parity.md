# For the paper lane: the operator training-set disparity, stated plainly

Handed to the coordinator 2026-09-23 from `experiments/ops-tune-grid`. The paragraph below is
meant to be used **verbatim**; everything after it is the sourcing, so a reviewer question can
be answered without coming back to this lane.

**Status of these numbers: final and independently checkable, with one exception marked †.**
They do not depend on any job this lane is still running — they are an accounting of data that
already exists. The error-versus-training-set-size measurements that this lane is producing are
separate and are not needed for the statement below.

---

## The paragraph

> Our neural-operator baselines were trained on 128 trajectories while the reduced-order model
> they are compared against was fitted on 4608. That disparity is an artefact of how much a
> training sample costs on each side, not a choice about how much data each method was allowed,
> and it runs against the baselines' interest rather than in their favour. Both sides draw
> initial conditions from the same Gaussian-bump family through the same generator and are
> evaluated at the same six output times, but the operators' training targets were held to a
> far stricter reference: each operator case is a 4096-interval, $\Delta t = 1.5625\times10^{-4}$
> solution restricted to the 256-interval evaluation grid, costing a median of 127.2 s of A100
> time per trajectory, whereas each trajectory in the reduced-order model's own bank and head
> was solved directly on the 256-interval grid at $\Delta t = 0.005$, costing about 0.19 s — a
> factor of roughly 670. At that rate the operators' 128 cases are what about 4.5 GPU-hours
> buys, and matching the reduced-order model's 4608 trajectories at the same fidelity would
> have cost roughly 163 A100-hours. The two sides also count "data" differently: an operator
> consumes one input–output pair per trajectory and is supervised on the five evolved fields,
> so 128 cases are 640 supervised states, while the decoder head is fitted per state and sees
> 131 072 of the 235 008 states available across its 4608 trajectories. The comparison should
> therefore not be read as a 36× data advantage to the reduced-order model on equal terms: it
> is a difference in what each side's data budget was spent on, and on the axis that governs
> label quality the baselines were given the stronger protocol throughout.

*(If a shorter form is needed, the load-bearing sentences are the second, the third and the
last.)*

---

## What each side actually sees

| | trajectories | states the model is fitted on | training-target solver | seconds per trajectory (A100) |
|---|---:|---:|---|---:|
| NM-ROM bank $g$ † | 576 | 16 384 | 256 intervals, $\Delta t = 0.005$, solved directly | 0.19 † |
| NM-ROM head $h_\theta$ † | **4608** | **131 072** (of 235 008 available, 51 per trajectory) | 256 intervals, $\Delta t = 0.005$, solved directly | 0.19 † |
| Neural operators, as published | **128** | 128 inputs → **640** supervised evolved states (5 per case) | 4096 intervals, $\Delta t = 1.5625\times10^{-4}$, restricted to 256 | **127.2** |

Derived ratios: trajectories 4608 / 128 = **36×**; supervised states 131 072 / 640 = **205×**;
cost per trajectory 127.2 / 0.19 ≈ **670×** the other way. 128 × 127.2 s = **4.5 GPU-hours**;
4608 × 127.2 s = **163 A100-hours**.

## Sources

| number | source |
|---|---|
| 127.2 s per operator trajectory (median; mean 129.9, range 113.4–157.5, n = 16) | `worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/live-train-index.json`, field `records[].reference.wall_seconds_including_first_compile`. Derived by `reports/sources.py:published_generation_seconds()`, not typed. |
| Operator training count 128; target solver 4096 / $1.5625\times10^{-4}$ restricted to 256 | the same index (`count`, `reference_setting`, `mesh`), and `protocol-refined.json` `counts.train` |
| Operator supervised states = 5 per case | the output-time contract $\{0, 0.05, 0.10, 0.15, 0.20, 0.25\}$ with the supplied initial state returned bitwise, so five fields are learned per case |
| Same initial-condition family, both sides | `experiments/mr-burgers2d/engines.py` `params_draw`, verified **byte-identical** between the two lanes' worktrees |
| Same output times, both sides | `engines.make_fom(horizon=0.25, output_spacing=0.05)` and `protocol-refined.json` `times` |
| † NM-ROM bank 576 trajectories / 16 384 states, job `2835788`; head 4608 trajectories / 131 072 of 235 008 states, job `2837431`; both at 256 intervals, $\Delta t = 0.005$; ≈0.19 s per trajectory (880 s for 4608) | those jobs' own records, via `worktrees/2026-09-16-b-head-train/experiments/b-head-train/{DESIGN.md,config-train.json}` and the job log `worktrees/2026-08-25-sepdec-consolidated/experiments/separable-decoder/runs/dn256b/logs/2837431.out` |

† **The two NM-ROM rows are quoted from those jobs' records and could not be independently
re-derived from any artifact reachable in this repository** — an independent audit of this
lane's design looked for them and did not find them. If the paper states the 0.19 s figure, it
should be traceable to job `2837431`'s log, which is where it comes from. Every other number in
this document is derived from a file the lane read and re-derives on demand.

## One qualification the paper should not drop

This lane is separately generating operator training data out to 4608 trajectories at a
**cheaper** solver setting (1024 intervals, $\Delta t = 3.125\times10^{-4}$, 4.8 s per case),
because 163 A100-hours at the published fidelity does not fit. That setting's measured deviation
from the pinned reference is ≤0.28 % — still several times finer than the targets the
reduced-order model's own bank and head were fitted on, and 10–25× below the errors being
compared. Crucially the **evaluation** targets are unchanged, so any bias from the cheaper
training targets lands *inside* the reported operator error rather than outside it. If the paper
quotes the enlarged-data results, it should say which fidelity produced them; if it quotes only
the published 128-case results, the paragraph above stands on its own.
