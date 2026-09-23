"""Generate the burgers2d-speed configs (edit this file, never the JSON). DESIGN.md sections 3-5.

config-{256,512,1024}.json   dev6 panels (the pre-registered candidate set, parity twins, FOM grid in both modes)
config-smoke128.json         local smoke (lattice stand-in for the file rule; not a result)
config-h{256,512,1024}.json  held-out hold64 panels, generated ONLY from the committed selection-<L>.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'burgers-bank-knob'
LADDER = [32, 64, 96, 128, 192, 256, 384, 512]
LIN_RP = [384, 256, 192, 128, 96]
K = 16
FOM = [{"name": "fft_tight", "dt": 0.005, "ntol": 1e-06, "ltol": 1e-08},
       {"name": "nt1e-4_dt005", "dt": 0.005, "ntol": 0.0001, "ltol": 1e-06},
       {"name": "nt1e-3_dt005", "dt": 0.005, "ntol": 0.001, "ltol": 1e-05},
       {"name": "nt1e-2_dt005", "dt": 0.005, "ntol": 0.01, "ltol": 0.5},
       {"name": "nt1e-4_dt01", "dt": 0.01, "ntol": 0.0001, "ltol": 1e-06},
       {"name": "nt1e-3_dt01", "dt": 0.01, "ntol": 0.001, "ltol": 1e-05},
       {"name": "nt1e-2_dt01", "dt": 0.01, "ntol": 0.01, "ltol": 0.5},
       {"name": "lean_tight", "dt": 0.005, "ntol": 1e-06, "ltol": 1e-08, "impl": "lean"},
       {"name": "lean_nt1e-4_dt005", "dt": 0.005, "ntol": 0.0001, "ltol": 1e-06, "impl": "lean"},
       {"name": "lean_nt1e-3_l1e-3_dt005", "dt": 0.005, "ntol": 0.001, "ltol": 0.001, "impl": "lean"},
       {"name": "lean_nt3e-3_l3e-3_dt005", "dt": 0.005, "ntol": 0.003, "ltol": 0.003, "impl": "lean"},
       {"name": "lean_nt1e-2_l1e-2_dt005", "dt": 0.005, "ntol": 0.01, "ltol": 0.01, "impl": "lean"},
       {"name": "lean_nt1e-3_l1e-3_dt01", "dt": 0.01, "ntol": 0.001, "ltol": 0.001, "impl": "lean"},
       {"name": "lean_nt3e-3_l3e-3_dt01", "dt": 0.01, "ntol": 0.003, "ltol": 0.003, "impl": "lean"},
       {"name": "lean_nt1e-2_l1e-2_dt01", "dt": 0.01, "ntol": 0.01, "ltol": 0.01, "impl": "lean"}]
assert [f['name'] for f in FOM] == [f['name'] for f in json.loads((PARENT / 'config-256.json').read_text())['fom_settings']]
RULES = json.loads((PARENT / 'config-256.json').read_text())['rules']
ROT_SHA = '51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772'
FAST = dict(solver='lu', clip=True, lamcarry=True, pred2=True)
PARENT_FAST_BAR = {L: json.loads((PARENT / 'checks' / f'bk{L}-summary.json').read_text())['selection'][
    'fast_reference_error_percent'] for L in (256, 512, 1024)}
# default compile mode when an arm does not name one (every candidate names one: both modes are run)
GRAPHS = dict(lin=True, trunc=True)


def accurate(L, x1=None):
    """The parent's ACCURATE variant at this mesh; x1=False drops the exact first step (a knob, <= 512 only)."""
    if L <= 512:
        v = dict(solver='chol', clip=True, lamcarry=True, pred2=True)
        if x1 is not False:
            v['exact_steps'] = 1
        return v, 0.01
    assert x1 is None
    return dict(solver='chol', clip=True, lamcarry=True, pred2=True), 0.001


def lin(L, Rp, x1=None, cap=None, impl='eng', graphs=None, **kw):
    v, g = accurate(L, x1)
    return dict(model='lin', Rp=Rp, M=4 * Rp, rule='lat64', gtol=g, variant=v, cap=cap, impl=impl,
                graphs=(GRAPHS['lin'] if graphs is None and impl == 'eng' else bool(graphs)), family='lin', **kw)


def q0(cap=None, impl='eng', graphs=None, **kw):
    return dict(model='trunc', Rp=512, q=0, M=64, rule='q0scaled', gtol=0.001, variant=FAST, cap=cap, impl=impl,
                graphs=(GRAPHS['trunc'] if graphs is None and impl == 'eng' else bool(graphs)), family='q0', **kw)


def arm_name(s):
    """Mirror of b2speed.arm_name (import-free)."""
    v = s['variant']
    sfx = ''.join(f'_{k}' for k in ([v['solver']] if v.get('solver', 'lu') != 'lu' else []) +
                  [k for k in ('clip', 'lamcarry', 'pred2') if v.get(k)])
    if v.get('exact_steps'):
        sfx += f"_x{v['exact_steps']}"
    g = f"g{s['gtol']:g}".replace('-', 'm').replace('.', 'p')
    body = f"lin_M{s['M']}" if s['model'] == 'lin' else f"q{s.get('q', 0)}_M{s['M']}"
    b = f"R{s['Rp']}_{body}_{s['rule']}_{g}_fast{sfx}"
    if s.get('cap'):
        b += f"_cap{s['cap']}"
    return b + '__' + s.get('impl', 'parent') + ('_graphs' if s.get('graphs') else '')


def candidates(L):
    """The pre-registered candidate set (DESIGN.md section 3): every knob combination, engineered, in BOTH compile
    modes (the knob's time is the faster mode's median; the two modes are parity-gated against each other). Only the
    graphs-mode arm of each knob is certified (the certificate is a property of the knob's iterates)."""
    arms = []
    for Rp in LIN_RP:
        for x1 in ((None, False) if L <= 512 else (None,)):
            for cap in (None, 1):
                for gr in (True, False):
                    arms.append(lin(L, Rp, x1=x1, cap=cap, graphs=gr, certify=gr))
    for cap in (None, 1):
        for gr in (True, False):
            arms.append(q0(cap=cap, graphs=gr, certify=gr))
    return arms


def twins(L):
    """Non-candidate timed arms: the parent's text at the three Table-1 knob settings (default compile). Parity pairs:
    each Table-1 knob's engineered arms (both modes) against the parent twin; every candidate knob's graphs arm
    against its default-compile arm."""
    tw, pairs = [], []
    for mk in (lambda **k: lin(L, 384, **k), lambda **k: lin(L, 128, **k), lambda **k: q0(**k)):
        par = mk(impl='parent', graphs=False, candidate=False, certify=False, role='parent_twin')
        tw.append(par)
        pairs += [[arm_name(mk(graphs=True)), arm_name(par)], [arm_name(mk(graphs=False)), arm_name(par)]]
    cands = candidates(L)
    for s in cands:
        if s['graphs']:
            pairs.append([arm_name(s), arm_name(dict(s, graphs=False))])
    return tw, pairs


def general(L, Rp=128):
    """The replaced GENERAL solver path on the linear rung (coordinator request for paper section 6.3): LU, reject,
    damping reset, linear 2-way guard, jacfwd Jacobian, unfused LM; same residual/rule/gtol/budget."""
    _, g = accurate(L)
    return dict(model='lin', Rp=Rp, M=4 * Rp, rule='lat64', gtol=g, variant=dict(solver='lu'), impl='general',
                graphs=False, family='lin', candidate=False, certify=False, role='general_path_section_6_3')


def base(L, attempt):
    arms = candidates(L)
    tw, pairs = twins(L)
    arms += tw
    if L == 1024:
        arms.append(general(L))
    names = [arm_name(s) for s in arms]
    assert len(names) == len(set(names))
    pop = dict(source_draw=[20260922 if L == 256 else 20260921, 56],
               cert_draws=[list(range(8 * d, 8 * d + 8)) for d in range(5)], confirm_draw=list(range(40, 56)),
               note='burgers-eqcert populations (as the parent lane): 256^2 params_draw(20260922,56), else '
                    'params_draw(20260921,56); 5 certification draws x 8 + 1 confirmation draw x 16')
    audit = ['fft_tight', arm_name(lin(L, 384)), arm_name(lin(L, 128)), arm_name(q0())]
    return dict(
        attempt=attempt, intervals=L, dt=0.005, eval_seed=7090702, eval_cases=4, eval_fresh_seed=911702,
        eval_fresh_cases=2,
        expected_physical_sha256='108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a',
        cohort_name='dev6: params_draw(7090702,4) + params_draw(911702,2), opened development cases',
        train_seed=0, train_trajectories=128, decoder_code_subsample=8192, cold_axis_points=48,
        strict=dict(ic_budget=400, step_budget=600), ic_gtol=1e-6, inner_damping=1e-10, rho_bar=0.116,
        directions_file='directions_qtd02.npz',
        directions_sha256='79d794580dad15c03630470940361d8ba9a7779673be9697c90d28e5c3d81535',
        rotation_sha256=ROT_SHA, ladder=LADDER, population=pop, rules=RULES, arms=arms,
        parity_pairs=pairs, parity_bar=1e-10, fom_settings=FOM, fom_both_modes=True, same_grid_reference='fft_tight',
        audit_cases=[0], audit_arms=audit, restrict_to=256,
        target_chunk={256: 16, 512: 8, 1024: 4}.get(L, 4), dense_tangent_group=34,
        phi_gate_max_mesh=512, rom_reps=5, fom_reps=3, order_seed=20260924, burn_seconds=0.1,
        phase_cooldown_seconds=5.0, phase_dummy_seconds=2.0, gate_limit=1.10,
        parent_fast_bar=dict(percent=PARENT_FAST_BAR.get(L), arm='R512_q0_M64_q0scaled_g0p001_fast_clip_lamcarry_pred2',
                             source=f'burgers-bank-knob checks/bk{L}-summary.json selection.fast_reference_error_percent'),
        purpose='DESIGN.md: deployment knobs + iterate-preserving engineering at this mesh, dev6 selection panel')


def smoke():
    c = base(128, 'smoke128')
    c['rules'] = dict(RULES, q0scaled=dict(lattice=32, note='SMOKE ONLY: stands in for the q0 file rule (L % 256)'))
    arms = [lin(128, 128, impl='parent', graphs=False, certify=False), lin(128, 128, graphs=True),
            lin(128, 128, graphs=False, certify=False),
            lin(128, 64, x1=False, impl='parent', graphs=False, certify=False), lin(128, 64, x1=False, graphs=True),
            lin(128, 64, x1=False, cap=1, graphs=True), q0(impl='parent', graphs=False, certify=False),
            q0(graphs=True), q0(cap=1, graphs=True), dict(general(1024, 64), gtol=0.01)]
    c.update(arms=arms, parity_pairs=[[arm_name(arms[1]), arm_name(arms[0])], [arm_name(arms[2]), arm_name(arms[0])],
                                      [arm_name(arms[4]), arm_name(arms[3])], [arm_name(arms[7]), arm_name(arms[6])]],
             eval_cases=1, eval_fresh_cases=1, expected_physical_sha256=None, local_smoke_waives_cohort_hash=True,
             rom_reps=1, fom_reps=1, required_reps=2, fom_settings=[FOM[0], FOM[10]], burn_seconds=0.01,
             phase_cooldown_seconds=0.1, phase_dummy_seconds=0.1,
             population=dict(c['population'], cert_draws=[[0], [1]], confirm_draw=[2]),
             audit_arms=['fft_tight', arm_name(arms[1])], phi_gate_max_mesh=128)
    return c


def hold(L):
    """hold64 at this mesh, generated only from the committed selection-<L>.json (DESIGN.md section 6)."""
    sel = json.loads((HERE / f'selection-{L}.json').read_text())
    c = base(L, f'h{L}')
    allarms = {arm_name(s): s for s in c['arms']}
    want = list(dict.fromkeys([sel['accurate'], sel['fast']]))
    twins_ = [sel['parent_accurate_twin'], sel['parent_fast_twin']]
    arms = [dict(allarms[n], certify=False) for n in want] + [dict(allarms[n], certify=False) for n in twins_]
    c.update(arms=arms, eval_draws=[[20260916, 64]], expected_physical_sha256=None, skip_certificates=True,
             cohort_name='hold64: params_draw(20260916, 64), held-out; never used for any fit, rule or selection',
             parity_pairs=[p for p in sel['parity_pairs'] if all(x in {arm_name(s) for s in arms} for x in p)],
             audit_arms=['fft_tight'] + want, rom_reps=2, fom_reps=1, required_reps=5, timed_full_sha_every=8,
             selection_source_summary_sha256=sel['source_summary_sha256'],
             purpose='held-out confirmation of the committed dev6 selection at this mesh')
    return c


def main():
    for L in (256, 512, 1024):
        (HERE / f'config-{L}.json').write_text(json.dumps(base(L, f'b{L}'), indent=1) + '\n')
        if (HERE / f'selection-{L}.json').exists():
            (HERE / f'config-h{L}.json').write_text(json.dumps(hold(L), indent=1) + '\n')
    (HERE / 'config-smoke128.json').write_text(json.dumps(smoke(), indent=1) + '\n')


if __name__ == '__main__':
    main()
