"""spectral-pilot: the interpretation, proposal, limitations and glossary sections of report.md. Every measured number
is read from runs/ANALYSIS.json or the run JSONs here; planning numbers (budgets, sizes, thresholds) are design
choices and are labelled as such. PILOT: small, not a result."""
from __future__ import annotations

import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = H / 'runs'


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

    SA = [a[lv]['targets'][t]['S'] for lv in ('A1', 'A2', 'A3') for t in ('0.03', '0.01', '0.003', '0.001')
          if a[lv]['targets'][t].get('S')]
    w('## 6. What the pilot suggests\n')
    w('### 6.1 (A) Why spectral wins on the box\n')
    if 'H-A3' in t3:
        S = t3['H-A3']['S']
        rr = t3['H-A2']['ratio']
        rx = t3['H-A2p'].get('ratio')
        w(f"- **No crossover in the pilot range; the comparison does not move toward the ROM.** At the only target "
          f"reached at all three levels (worst error 3 %), the spectral FOM is faster at every level: "
          f"S = {', '.join(g(x, '%.2f') for x in S)} for A1, A2, A3 (median s = "
          f"{', '.join(g(a[lv]['s_median'], '%.0f') for lv in ('A1', 'A2', 'A3'))}), slope of log S on log s "
          f"{g(t3['H-A3']['slope'], '%.2f')}. At A3 the ROM reaches no tighter target with r ≤ 512 (censored). "
          f"PILOT: small, not a result.")
        w(f"- **The trained span does get relatively more compressive as fronts sharpen** (H-A2 {t3['H-A2']['verdict']} at 3 %): "
          f"$r_{{\\min}}/(N_{{\\min}}-1)^2$ = {', '.join(g(x, '%.2f') for x in rr)}. At A3 the POD span needs "
          f"{a['A3']['targets']['0.03']['r_min']} unknowns where the spectral FOM carries "
          f"{(a['A3']['targets']['0.03']['N_min'] - 1) ** 2} (N = {a['A3']['targets']['0.03']['N_min']}).")
        if rx:
            w(f"- **But not fast enough to pay for its unknowns** (H-A2′ {t3['H-A2p']['verdict']}): one tensor evaluation costs "
              f"about $r^3$ flops against the FOM's $N^2\\log N$, so the ROM's per-step work equals the FOM's at "
              f"$r_\\times(N_{{\\min}})$ ≈ {', '.join(g(a[lv]['targets']['0.03']['r_cross']['value'], '%.0f') for lv in ('A1', 'A2', 'A3'))}; "
              f"the needed $r_{{\\min}}/r_\\times$ = {', '.join(g(x, '%.2f') for x in rx)} grows with steepness. "
              "A crossover would need the reduced size to grow more slowly than about $N^{2/3}$, i.e. far more "
              "compression than a linear span achieves for moving fronts.")
        w(f"- H-A1 (spectral resolution grows with steepness): α = {g(t3['H-A1']['alpha'], '%.2f')} "
          f"({t3['H-A1']['verdict']}); $N_{{\\min}}$ = "
          f"{', '.join(str(a[lv]['targets']['0.03']['N_min']) for lv in ('A1', 'A2', 'A3'))} at 3 %, and "
          f"{', '.join(str(a[lv]['targets']['0.001']['N_min']) for lv in ('A1', 'A2', 'A3'))} at 0.1 %. Spectral "
          "resolution grows with steepness, but slowly (the fronts are resolved, so convergence stays "
          "exponential).")
    sine = [(lv, t) for lv in ('A1', 'A2', 'A3') for t in ('0.03', '0.01', '0.003', '0.001')
            if a[lv]['targets'][t].get('r_min_sine') and a[lv]['targets'][t].get('r_min')
            and a[lv]['targets'][t]['r_min_sine'] <= a[lv]['targets'][t]['r_min']]
    if sine:
        w(f"- On the smooth end the learned span is not even better than plain sines: the sine-ROM (first r "
          f"Laplacian-ordered sines, same Galerkin/ETDRK4 machinery) needs no more unknowns than the POD ROM at "
          f"{', '.join(lv + ' τ=' + t for lv, t in sine)}.")
    nat = a['A1']['targets']['0.03'].get('native_S')
    w(f"- Output synthesis is a large part of the ROM's query: on the native contract (no decode to the 257² "
      f"lattice) S at A1, 3 % is {g(nat, '%.2f')} against {g(a['A1']['targets']['0.03'].get('S'), '%.2f')} on the "
      "field contract.")
    d1 = [(lv, a[lv]['targets']['0.03']['r_floor'], a[lv]['targets']['0.03']['r_floor48']) for lv in ('A1', 'A2', 'A3')]
    w(f"- Data sensitivity (D1): the 3 % floor needs r = {', '.join(f'{x} (96 traj.) / {y} (48)' for _, x, y in d1)} "
      f"at A1, A2, A3; doubling the data lowers the needed r at {sum(1 for _, x, y in d1 if x and y and x < y)} of 3 "
      "levels, so the 96-trajectory POD is probably still data-limited and r_min is pessimistic for a "
      "better-trained span. This changes the cost argument only if r falls by several times.")

    w('\n### 6.2 (B) A domain without a fast transform\n')
    if b:
        ts = [t for t in ('0.03', '0.01', '0.003', '0.001') if b['targets'][t].get('S') is not None]
        w(f"- **Against these two FD solvers, the POD–Galerkin ROM is far faster at every target:** S = "
          f"{', '.join(g(b['targets'][t]['S'], '%.0f') + ' (τ=' + t + ')' for t in ts)}. The FD solvers need "
          f"L = {', '.join(str(min(x for x in (b['targets'][t].get('L_min_IMEX'), b['targets'][t].get('L_min_RK4')) if x)) for t in ts)} "
          f"while the ROM needs r = {', '.join(str(b['targets'][t]['r_min']) for t in ts)}. PILOT: small, not a "
          "result; scoped to these FD implementations.")
        rom = _l(R / 'B' / 'rom.json')
        ref = _l(R / 'B' / 'ref.json')
        off = ref['seconds']['train'] + rom['timings']['pod'] + rom['timings']['tensor_full']
        x = b['targets']['0.001']
        if x.get('S'):
            sav = x['fom']['t'] - x['rom']['t']
            w(f"- Offline cost (training data, POD, tensor; wall time on the shared CPU) is {g(off / 60, '%.0f')} "
              f"minutes; at τ = 0.1 % each query saves {g(sav * 1e3, '%.0f')} ms on one core, so the break-even is "
              f"about {g(off / sav, '%.0f')} queries (mixing multi-process offline wall time with single-core "
              "query time; indicative only).")
    if ab and b:
        sab = [(t, ab['targets'][t].get('S'), b['targets'][t].get('S')) for t in ('0.03', '0.01', '0.003', '0.001')]
        sab = [(t, x, y) for t, x, y in sab if x is not None]
        if sab:
            win = [t for t, x, _ in sab if x >= 1]
            w(f"- **Post-hoc control AB (A6): the identical Part-B cases solved on the box, ROM vs the spectral FOM**: "
              f"S = {', '.join(g(x, '%.2f') + ' (τ=' + t + ')' for t, x, _ in sab)}. With this narrow family the ROM "
              f"beats even the spectral FOM on the box at τ ∈ {{{', '.join(win) or 'none'}}} "
              f"(r = {', '.join(str(ab['targets'][t]['r_min']) for t, _, _ in sab)} against the FOM's N = "
              f"{', '.join(str(ab['targets'][t]['N_min']) for t, _, _ in sab)}). So the Part-B advantage has two "
              "parts: the narrow, low-dimensional family (which already gives the ROM a win on the box at loose "
              "targets) and the replacement of the spectral FOM by second-order FD (the remaining factor "
              f"S_B / S_AB = {', '.join(g(y / x, '%.0f') + ' (τ=' + t + ')' for t, x, y in sab if y)}). The pilot "
              "cannot separate 'no fast transform' from 'second-order FD is slow'; a high-order L-shape solver is "
              "needed for that.")
        else:
            w('- Post-hoc control AB: no target reached by both methods (censored).')
    SA = [a[lv]['targets'][t]['S'] for lv in ('A1', 'A2', 'A3') for t in ('0.03', '0.01', '0.003', '0.001')
          if a[lv]['targets'][t].get('S')]
    SB = [b['targets'][t]['S'] for t in b['targets'] if b['targets'][t].get('S')] if b else []
    w(f"- **Caveat that dominates B:** second-order FD is a weak competitor for smooth solutions; on the box the "
      f"spectral FOM beats the same ROM machinery by {g(1 / max(SA), '%.0f')}–{g(1 / min(SA), '%.0f')}×, while on "
      f"the L-shape the FD solvers lose by {g(min(SB), '%.0f')}–{g(max(SB), '%.0f')}×. A "
      "high-order spectral-element or multi-domain Chebyshev solver on the L-shape would sit between these, and it "
      "is the comparison that decides the question. It was not run.\n")

    w('### 6.3 Does a crossover against spectral look reachable, and where?\n')
    if ab:
        sab = [ab['targets'][t].get('S') for t in ('0.03', '0.01', '0.003', '0.001') if ab['targets'][t].get('S')]
        w(f"- **What drives the outcome is the width of the parameter family (its n-width) more than front "
          f"steepness.** With the Part-A family (bump centres anywhere in a 0.4-wide square) the ROM loses at every "
          f"level; with the Part-B family (centres in a 0.08-wide square) on the same box and the same solvers it wins "
          f"at the looser targets (S up to {g(max(sab), '%.1f')}). Steepness raises both methods' sizes; family width "
          "decides whether the reduced size stays below the equal-cost size $r_\\times$. (AB is post hoc and has a "
          "different ν band from A1–A3; PILOT.)")
    w(f"- **On the box with a wide family: not with a linear span.** The pilot's trend runs the wrong way (the ROM's required size "
      f"relative to its equal-cost size grows with steepness), the ROM is {g(1 / max(SA), '%.0f')}–"
      f"{g(1 / min(SA), '%.0f')}× slower at every eligible level and target, and "
      "its cost per unknown (dense $r^3$ tensor or $6mr$ grid rule) is structurally above the FFT's. Steeper fronts "
      "make the linear n-width worse for moving fronts. A crossover on the box would need a nonlinear or "
      "transport-aware reduced representation (registration, shifted POD, manifold ROMs) whose online cost does "
      "not grow with the front resolution — which this pilot did not test.")
    w("- **Off the box: plausible, and it is where the full experiment should go.** Where no fast transform "
      "exists the ROM's advantage over the FD solvers is one to two orders of magnitude in the pilot. Whether it "
      "survives a high-order solver (spectral elements, $p$-FEM with static condensation, Chebyshev multi-domain) "
      "and a wider parameter family is the open question.\n")

    w('## 7. Proposed full-scale experiment (cluster lane; planning numbers)\n')
    w("Two parts, each pre-registered and Codex-gated as usual. Budgets are planning estimates, not results.\n")
    w("**Part A′ (box, steepness, one more chance for the bank).** Purpose: decide whether *any* reduced "
      "representation we can build crosses the spectral FOM on the box as fronts sharpen; stop the box line if not.")
    w("- Primary axis (from the AB control): family width (bump-centre range 0.08 → 0.4 in 4 steps, width range "
      "likewise) crossed with 3 steepness levels; the readout is the family width at which S crosses 1 and 2 for "
      "each target, per arm.")
    w("- Cohorts: the paired viscosity sweep extended to 5 levels (s up to about 2000; $N_{\\rm ref}$ up to 1536–2048 "
      "on H200), training sets 96 / 384 / 1536 trajectories per level (data-sufficiency curve), 32 test cases per "
      "level, plus a sealed confirmation cohort per level opened only once.")
    w("- Arms: (1) spectral FOM ETDRK4 (fixed and adaptive step), GPU and CPU; (2) POD–Galerkin tensor ETDRK4 "
      "(this pilot); (3) the coordinate-network bank retrained per level (incumbent recipe, linear span) with the "
      "off-mesh rule and its best stepper (F1 of the ideation report); (4) a transport-aware arm (registration or "
      "shifted POD) or a nonlinear decoder; (5) POD-ECSW as the standard hyper-reduction control.")
    w("- Metrics: as here (worst/median continuous-L² error, three-way scoring with allowances, matched cost-optimal "
      "configurations, field and native contracts), on GPU (H200) and one CPU core.")
    w("- Budget: references and training data 15 GPU-h; bank retrains 5 levels × 2 data sizes × ~3 A100-h = 30 GPU-h; "
      "ROM ladders and timing 10 GPU-h; transport/nonlinear arm 10 GPU-h; total about **65 GPU-h**.")
    w("- Kill criteria: stop Part A′ if, at every level and every target reached by both methods, the best reduced "
      "arm has S < 0.5 **and** its $r_{\\min}/r_\\times(N_{\\min})$ does not fall with steepness; stop the bank arm "
      "if its floor at $r_\\times(N_{\\min})$ is above τ at every level.\n")
    w("**Part B′ (no fast transform).** Purpose: test the pilot's Part-B signal against strong solvers and a wider "
      "family.")
    w("- Geometries: the L-shape and a box with a circular hole (curved boundary; the bank uses an R-function / "
      "distance envelope, with the corner-singularity caveat of A2.1).")
    w("- Cohorts: Part-B family widened to the Part-A spread (centres over the domain, w 0.035–0.2) and two "
      "steepness levels; 96 / 384 training trajectories; 32 test; sealed confirmation cohort.")
    w("- Competitors: FD + IMEX with geometric/algebraic multigrid (GPU), high-order spectral-element or $p$-FEM "
      "with static condensation (the decisive comparator), Chebyshev multi-domain on the L-shape, explicit RK on "
      "the coarsest adequate high-order mesh; each tuned over order, mesh, step and solver tolerance.")
    w("- Reduced arms: POD–Galerkin tensor (this pilot), the off-mesh bank with an envelope, POD-ECSW.")
    w("- Budget: references (high-order, refined) 15 GPU-h; competitor ladders 20 GPU-h; bank trainings 2 geometries "
      "× 2 levels × ~3 A100-h = 12 GPU-h; ROM ladders and timing 8 GPU-h; total about **55 GPU-h**. Whole proposal "
      "about **120 GPU-h** with contingency.")
    w("- Kill criteria: the Part-B claim is withdrawn if the best high-order solver matches the ROM's worst error "
      "with S < 2 at every target; the 'geometry' attribution is withdrawn if the box control with the same family "
      "(as AB here) gives the ROM the same advantage over the spectral FOM.\n")

    w('## 8. Deviations, limitations, audits\n')
    w("- **Amended exploratory design**: references and training for A1–A3 and an A1 debugging ladder ran before "
      "design amendment A3 was final; implementation records A5 (two addenda) and the post-hoc control A6 are in "
      "`DESIGN.md`. A4 not run (time cap). A3's r = 512 ROM ladder ran as a separate checkpoint unit after the "
      "first A3 job reached its 30-minute cap; the AB control is post hoc.")
    w("- The ROM is a POD–Galerkin feasibility comparator, not the coordinate-network bank; matching is in-sample "
      "and symmetric on 12 test cases per level; errors are against numerical references with conditional "
      "allowances; all timing is one CPU core of the GB10 (no GPU numbers).")
    w("- Not implemented (stated limitations): written memory projections, pre-stage benchmarks, D3, B "
      "corner-window and finer-lattice diagnostics, Part-B native timing and H1, rollouts for D1/random span.")
    w("- Audits (`audits/`): design 1–3 NOT-READY, 4 READY; code 1–4 NEEDS-FIX, 5 READY-TO-TRUST; results audit "
      "listed in the hand-back.\n")

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
