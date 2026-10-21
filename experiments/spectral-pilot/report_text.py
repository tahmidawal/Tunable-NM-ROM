"""spectral-pilot: the interpretation, proposal, limitations and glossary sections of report.md. Every measured number
is read from runs/ANALYSIS.json or the run JSONs here; planning numbers (budgets, sizes, thresholds) are design
choices and are labelled as such. PILOT: small, not a result."""
from __future__ import annotations

import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = H / 'runs'
TS = ('0.03', '0.01', '0.003', '0.001')


def _l(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def g(x, f='%.3g'):
    return '–' if x is None else (f % x if isinstance(x, (int, float)) else str(x))


def tail(w):
    A = _l(R / 'ANALYSIS.json')
    a = A['A']
    v = A['verdicts']
    t3 = v['0.03']
    b = A.get('B')
    ab = a.get('AB') if a.get('AB', {}).get('status') == 'run' else None
    L3 = ('A1', 'A2', 'A3')
    SA = [a[lv]['targets'][t]['S'] for lv in L3 for t in TS if a[lv]['targets'][t].get('S')]
    SB = [b['targets'][t]['S'] for t in TS if b and b['targets'][t].get('S')]

    w('## 6. What the pilot suggests (PILOT: small, not a result)\n')
    w('All statements in this section are about **this POD–Galerkin implementation, the evaluated ladders, '
      'in-sample matching on 12 test cases per level, and one CPU core**. They do not transfer to the '
      'coordinate-network bank or to other reduced representations (DESIGN A1.1).\n')
    w('### 6.1 (A) Why spectral wins on the box\n')
    if 'H-A3' in t3:
        S = t3['H-A3']['S']
        w(f"- **No crossover observed.** At the only target reached at all three levels (worst error 3 %), the "
          f"spectral FOM is faster at every level: S = {', '.join(g(x, '%.2f') for x in S)} for A1, A2, A3 "
          f"(median s = {', '.join(g(a[lv]['s_median'], '%.0f') for lv in L3)}); the sequence is not monotone and "
          f"the log-slope is {g(t3['H-A3']['slope'], '%.2f')} (PILOT). Over all eligible levels and targets the "
          f"ROM is {g(1 / max(SA), '%.0f')}–{g(1 / min(SA), '%.0f')}× slower. At A3 the ROM reached no tighter "
          "target with r ≤ 512 (censored, not evidence either way).")
        w(f"- **The trained span needs fewer unknowns than the spectral block, increasingly so at A3** (H-A2 "
          f"{t3['H-A2']['verdict']} by the registered endpoint rule; not monotone): $r_{{\\min}}/(N_{{\\min}}-1)^2$ = "
          f"{', '.join(g(x, '%.2f') for x in t3['H-A2']['ratio'])}; at A3, r = "
          f"{a['A3']['targets']['0.03']['r_min']} against {(a['A3']['targets']['0.03']['N_min'] - 1) ** 2} sine modes "
          f"(N = {a['A3']['targets']['0.03']['N_min']}) (PILOT).")
        hp = t3['H-A2p']
        if 'ratio' in hp:
            w(f"- **Per nonlinear evaluation the ROM is near cost parity, so the per-unknown cost is the issue, not "
              f"the count alone** (H-A2′ {hp['verdict']}): with the cheaper measured evaluation form per r, "
              f"$r_{{\\min}}/r_\\times(N_{{\\min}})$ = {', '.join(g(x, '%.2f') for x in hp['ratio'])}; with the "
              f"tensor alone (diagnostic) {', '.join(g(x, '%.2f') for x in t3.get('H-A2p_tensor_only', []))}. "
              "The cheaper form is often a coarse grid rule whose own accuracy was not required here, so these are "
              "optimistic for the ROM (PILOT). Since the matched whole-query ratio is far below these per-evaluation "
              "ratios, the rest of the ROM query (the grid size actually needed, the initial projection, the "
              "coefficient setup, the output decode and the step count) carries the gap; the pilot did not "
              "decompose it.")
        w(f"- H-A1: α = {g(t3['H-A1']['alpha'], '%.2f')} ({t3['H-A1']['verdict']}); $N_{{\\min}}$ = "
          f"{', '.join(str(a[lv]['targets']['0.03']['N_min']) for lv in L3)} at 3 % and "
          f"{', '.join(str(a[lv]['targets']['0.001']['N_min']) for lv in L3)} at 0.1 % (PILOT): the spectral "
          "resolution needed grows with steepness, but from small values.")
    sine = [(lv, t) for lv in L3 for t in TS
            if a[lv]['targets'][t].get('r_min_sine') and a[lv]['targets'][t].get('r_min')
            and a[lv]['targets'][t]['r_min_sine'] <= a[lv]['targets'][t]['r_min']]
    if sine:
        w(f"- At the smooth end the POD span is not better than plain sines: the sine-ROM needs no more unknowns "
          f"than the POD ROM at {', '.join(lv + ' τ=' + t for lv, t in sine)} (PILOT).")
    nat = a['A1']['targets']['0.03'].get('native_S')
    w(f"- The output contract matters: at A1, 3 %, S is {g(nat, '%.2f')} on the native contract (no decode to the "
      f"257² lattice) against {g(a['A1']['targets']['0.03'].get('S'), '%.2f')} on the field contract (PILOT; native "
      "timings are batched, not call-interleaved).")
    d1 = [(a[lv]['targets']['0.03']['r_floor'], a[lv]['targets']['0.03']['r_floor48']) for lv in L3]
    w(f"- Data sensitivity of the *projection floor* (D1): the 3 % floor needs r = "
      f"{', '.join(f'{x} (96 traj.) / {y} (48)' for x, y in d1)} at A1, A2, A3 (PILOT). More data lowered the "
      "floor-based r here; whether it would lower the rollout r or the cost is untested.")

    w('\n### 6.2 (B) A domain without a fast transform\n')
    if b:
        ts = [t for t in TS if b['targets'][t].get('S') is not None]
        w(f"- **Against these two second-order FD solvers the POD–Galerkin ROM is much faster at every target:** "
          f"S = {', '.join(g(b['targets'][t]['S'], '%.0f') + ' (τ=' + t + ')' for t in ts)}; FD needs "
          f"L = {', '.join(str(min(x for x in (b['targets'][t].get('L_min_IMEX'), b['targets'][t].get('L_min_RK4')) if x)) for t in ts)}, "
          f"the ROM r = {', '.join(str(b['targets'][t]['r_min']) for t in ts)} (PILOT; scoped to these FD "
          "implementations; B basis qualification unestablished).")
        rom = _l(R / 'B' / 'rom.json')
        ref = _l(R / 'B' / 'ref.json')
        off = ref['seconds']['train'] + rom['timings']['pod'] + rom['timings']['tensor_full']
        x = b['targets']['0.001']
        if x.get('S'):
            sav = x['fom']['t'] - x['rom']['t']
            w(f"- Component-based break-even estimate: training data + POD + tensor took {g(off / 60, '%.0f')} minutes "
              f"of wall time on the shared multi-core CPU; at τ = 0.1 % each query saves {g(sav * 1e3, '%.0f')} ms "
              f"on one core, i.e. about {g(off / sav, '%.0f')} queries (PILOT; mixes parallel offline wall time "
              "with single-core query time; not a deployment accounting).")
    if ab and b:
        sab = [(t, ab['targets'][t].get('S'), b['targets'][t].get('S')) for t in TS]
        sab = [(t, x, y) for t, x, y in sab if x is not None]
        if sab:
            win = [t for t, x, _ in sab if x >= 1]
            w(f"- **Post-hoc control AB (A6): the identical Part-B cases solved on the box, ROM vs the spectral FOM:** "
              f"S = {', '.join(g(x, '%.2f') + ' (τ=' + t + ')' for t, x, _ in sab)}; the ROM is faster at "
              f"τ ∈ {{{', '.join(win) or 'none'}}} (r = {', '.join(str(ab['targets'][t]['r_min']) for t, _, _ in sab)} "
              f"against N = {', '.join(str(ab['targets'][t]['N_min']) for t, _, _ in sab)}) (PILOT). So with a "
              "narrow family the same ROM machinery can beat even the spectral FOM on the box at loose targets. "
              f"The descriptive cross-panel ratios S_B / S_AB = "
              f"{', '.join(g(y / x, '%.0f') + ' (τ=' + t + ')' for t, x, y in sab if y)} (PILOT) mix several changes "
              "at once (geometry, competitor, scoring norm, basis, initialisation, output cost); they are not an "
              "attribution to the missing fast transform.")
    w(f"- **Caveat that dominates B:** second-order FD is a weak competitor for smooth solutions (on the box the "
      f"spectral FOM beats this ROM by {g(1 / max(SA), '%.0f')}–{g(1 / min(SA), '%.0f')}× with the wide family, "
      f"while the L-shape FD solvers lose to it by {g(min(SB), '%.0f')}–{g(max(SB), '%.0f')}×; PILOT). How a "
      "high-order spectral-element or multi-domain Chebyshev solver on the L-shape would compare is unknown; it "
      "was not run and it is the comparison that decides question (B).\n")

    w('### 6.3 Does a crossover against spectral look reachable, and where?\n')
    if ab:
        sab = [ab['targets'][t].get('S') for t in TS if ab['targets'][t].get('S')]
        w(f"- **The AB result motivates a controlled family-width experiment.** With the Part-A family the ROM lost "
          f"at every level; with the narrow Part-B family on the same box it won at the looser targets (S up to "
          f"{g(max(sab), '%.1f')}, PILOT). AB also changes the viscosity band, the bump-width distribution and the "
          "cases, so the pilot does not establish that family width dominates viscosity or front sharpness; it "
          "makes family width (with the others held fixed) the first axis to test.")
    w("- **On the box with the wide Part-A family: no crossover was observed for this POD implementation**, and "
      "the steepness sweep did not move the comparison toward the ROM. Whether a better-trained span, the "
      "coordinate-network bank, or a transport-aware or nonlinear representation would cross is a hypothesis for "
      "the full experiment, not a pilot finding.")
    w("- **Off the box: a large advantage over second-order FD was observed; against a high-order solver it is "
      "unknown.** That comparison, with a wider family, is where the full experiment should start.\n")

    w('## 7. Proposed full-scale experiment (cluster lane)\n')
    w("All sizes, budgets and thresholds in this section are **planning estimates and design choices, not "
      "measured results**. Both parts are pre-registered and Codex-gated as usual.\n")
    w("**Part A′ — the box: family width × steepness.**")
    w("- Matrix: family width W ∈ {0.08, 0.16, 0.28, 0.4} (range of bump centres; bump width held at the pilot's "
      "U[0.035, 0.05]) × steepness levels {A1, A2, A3, A4} (paired ν, factor 3 per level; A4 needs "
      "$N_{\\rm ref}$ ≈ 768 on GPU) = 16 cells; 32 test cases per cell; one sealed confirmation cohort per cell, "
      "opened once; training 96 and 384 trajectories per cell (data-sufficiency pair).")
    w("- Arms: (1) spectral FOM ETDRK4 (fixed and adaptive step); (2) POD–Galerkin tensor/grid ETDRK4 (this pilot); "
      "(3) the coordinate-network bank retrained per cell (linear span, off-mesh rule, best stepper F1 of the "
      "ideation report) for the 8 cells with W ∈ {0.08, 0.4}; (4) one transport-aware or nonlinear arm on the 4 "
      "cells with W = 0.4; (5) POD-ECSW.")
    w("- Readouts: the family width at which S crosses 1 and 2 per target and level; matched S on GPU (H200) and "
      "one CPU core; per arm a query-cost decomposition (evaluation, initial projection, setup, decode, steps). "
      "The equal-cost size $r_\\times$ is defined per arm from that arm's own measured evaluation cost.")
    w("- Budget (planning): references and training data 16 cells × 2 sizes ≈ 12 GPU-h; bank retrains 8 cells × "
      "2 data sizes × ~3 A100-h ≈ 48 GPU-h; transport/nonlinear arm ≈ 12 GPU-h; ladders, timing, POD-ECSW ≈ 15 "
      "GPU-h. Subtotal ≈ 87 GPU-h.")
    w("- Kill criteria: stop the box line if, in every cell with W ≥ 0.28, every reduced arm has S < 0.5 at every "
      "target reached by both methods; stop the bank arm if its projection floor at its own $r_\\times$ is above "
      "τ in every cell.\n")
    w("**Part B′ — no fast transform, against strong solvers.**")
    w("- Geometries: the L-shape and a box with a circular hole (the bank uses an R-function/distance envelope; "
      "corner caveat of A2.1).")
    w("- Families: the pilot's narrow family and the wide family W = 0.4 (same bump widths as A′), two steepness "
      "levels; 32 test cases and a sealed cohort per cell (2 geometries × 2 families × 2 levels = 8 cells).")
    w("- Competitors (the decisive part): high-order spectral elements or $p$-FEM with static condensation, "
      "Chebyshev multi-domain (L-shape), FD-IMEX with multigrid; each tuned over order, mesh, step and solver "
      "tolerance. Reduced arms: POD–Galerkin, the bank with an envelope, POD-ECSW. Plus the box control of each "
      "cell (as AB), against the spectral FOM.")
    w("- Budget (planning): references ≈ 10 GPU-h; competitor ladders ≈ 20 GPU-h; bank trainings 8 cells × ~3 "
      "A100-h ≈ 24 GPU-h; ladders and timing ≈ 8 GPU-h. Subtotal ≈ 62 GPU-h.")
    w("- Kill criteria: withdraw the Part-B claim if the best high-order solver matches the ROM's worst error with "
      "S < 2 at every target in the wide-family cells; withdraw any geometry attribution if the box control of a "
      "cell gives the ROM the same S against the spectral FOM.\n")
    w("**Total (planning): ≈ 150 GPU-h, plus a 20 % contingency held by the coordinator (≈ 180 GPU-h ceiling).** "
      "Run Part B′ first (its kill test is cheaper and decides the paper's off-box claim).\n")

    w('## 8. Deviations, limitations, audits\n')
    w("- **Amended exploratory design**: references and training for A1–A3 and an A1 debugging ladder ran before "
      "design amendment A3 was final; implementation records A5 (two addenda) and the post-hoc control A6 are in "
      "`DESIGN.md`. A3's first ROM job reached its 30-minute unit cap at r = 512, which then ran as a separate "
      "checkpoint unit; by rule A2.6 (A4 only if all A3 stages finished) **A4 was not run**.")
    w("- Allowance coverage: the A padding gate covers the two steepest test cases per level; the B time allowance "
      "extrapolates from a fine-grid check on 4 cases and an L = 512 check on 12. G6 (cohort disjointness) and D2 "
      "(no run below its floor, Part A and B, and the spectral floor) were asserted in the code. Diagnostics (G9, "
      "H1, coarse references, random spans) are reported, not asserted.")
    w("- Timing: one GB10 CPU core, f64, single-thread flags asserted; sentinels valid in every screen and final. "
      "`JAX_DEFAULT_MATMUL_PRECISION` was not set (recorded as null); on CPU with f64 it is not expected to change "
      "results, but it departs from the project rule. Native-contract timings are batched, not interleaved. "
      "Spreads and outlier counts are in the timing table above.")
    w("- Not implemented (stated limitations): written memory projections, pre-stage benchmarks, D3, B "
      "corner-window and finer-lattice diagnostics, Part-B native timing and H1, rollouts for D1/random span.")
    w("- Audits (`audits/`): design 1–3 NOT-READY, 4 READY; code 1–4 NEEDS-FIX, 5 READY-TO-TRUST; results 1 "
      "NEEDS-FIX (numbers reproduced; interpretation corrected), later rounds listed in the hand-back.\n")

    w('## 9. Glossary\n')
    for k, d in [
        ('s = a/ν', 'nominal steepness: bump amplitude over viscosity; a front of jump a has width of order 4ν/a.'),
        ('levels A1–A3', 'paired viscosity sweep on the box: the same bump parameters, ν divided by 3 per level.'),
        ('AB', 'post-hoc control: the Part-B cases (narrow family) solved on the box.'),
        ('spectral FOM, N', 'full-order sine pseudo-spectral solver with ETDRK4 steps; it keeps the (N−1)² modes '
         '$\\sin(k_1\\pi x)\\sin(k_2\\pi y)$, $k_a<N$.'),
        ('ROM, r', 'reduced model with r unknowns: POD span of training snapshots, Galerkin projection, ETDRK4.'),
        ('ROM-T / ROM-Q', 'nonlinear term evaluated by the exact quadratic tensor (≈ r³ flops) or by a grid rule of '
         'm points (6mr flops).'),
        ('sine-ROM', 'the same reduced machinery on the first r Laplacian-ordered sines (no learning).'),
        ('floor', 'error of the best approximation in a span (orthogonal projection); no model in that span can do '
         'better.'),
        ('worst / median error', 'over the 12 test cases of the largest relative L² error at t = 0.05…0.25.'),
        ('τ', 'error target on the worst error: 3 %, 1 %, 0.3 %, 0.1 %.'),
        ('U, three-way rule', 'conditional allowance for reference error; a run meets τ if error + U ≤ τ, misses if '
         'error − U > τ, otherwise unresolved.'),
        ('$N_{\\min}$, $r_{\\min}$, $L_{\\min}$', 'smallest observed size that meets τ with some step count.'),
        ('matched / cost-optimal', 'the fastest timed configuration of a method that meets τ.'),
        ('S', 'speed ratio t_FOM / t_ROM of the matched configurations (S > 1: the ROM is faster); K2 bar S ≥ 2.'),
        ('field / native contract', 'query ends with the six output fields on the scoring lattice / with the '
         'method\'s own coefficients.'),
        ('$r_\\times(N)$', 'reduced size at which one ROM tensor evaluation costs as much as one FOM evaluation at N.'),
        ('F-IMEX, F-RK4, SBDF2', 'finite-difference solvers on the L-shape: implicit–explicit second-order backward '
         'differentiation with sparse LU solves, or explicit classical Runge–Kutta.'),
        ('G4, SW1–SW4, G7, G9, D1, D2', 'controls: tensor = direct ROM; matcher, reference-gate and tensor-corruption '
         'fault injections; IC quadrature; padding diagnostic; data sensitivity; no run below its floor.'),
        ('censored', 'a method did not reach the target inside its ladder; not evidence either way.'),
    ]:
        w(f'- **{k}**: {d}')
