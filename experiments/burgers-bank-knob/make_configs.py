"""Generate config-{256,512,1024,2048,4096}.json and config-smoke128.json. Edit this file, never the JSON.

Arms at every mesh (DESIGN.md section 3), R' in {512, 384, 256, 128, 64, 32}:
  (a) head only, q = 0, M = 64, the certified q = 0 rule (`q0scaled`), the paper's FAST solver variant;
  (b) linear rung, head dropped, R' unknowns, M = 4 R', `lat64`, the paper's ACCURATE solver variant at this mesh;
  (c) head + corrections, q = min(256, R' - 16), M = 4 (16 + q), `lat64`, the ACCURATE variant.
Plus, where the unrotated bank fits beside the rotated one (L <= 2048): the unrotated parent's fast and accurate
arms (the paper's current settings) as parity twins of (a) and (c) at R' = 512; and one certificate CONTROL arm
(the `bad0` rule on (c) at R' = 512, untimed) that must fail the rho bar.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = [32, 64, 128, 256, 384, 512]
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
RULES = {
    'q0scaled': dict(file='rules-eqtop/rule_q0_m1024_qrg304_reachable.npz',
                     sha256='6a32568f7b176b8774b31c8c96abf3da61b88dcf8b5ad657ced3f7c840ebe165', mesh=256,
                     note='b-eqtop q=0 rule, m=1024, nodes at the same physical points, weights x (L/256)^2, no refit; '
                          'the paper fast rule (burgers-eqcert: confirmed at 256^2/512^2/1024^2)'),
    'lat64': dict(lattice=64, note='uniform 63x63 interior sub-lattice, equal weights; the paper accurate rule '
                                   '(burgers-eqcert: certified at 256^2 (bc256b, x1), 512^2 (x1), 1024^2; 2048^2 '
                                   'certificate-only; 4096^2 NOT confirmed)'),
    'bad0': dict(file='rules/rule_q256_reachable_m2048.npz',
                 sha256='770470510e21c564eddee82d2685072336e7d5fe0d86c9357ad49265b9dc89d0', mesh=256, control=True,
                 note='q-ridge rule that regressed at 1024^2 in b-panel: certificate CONTROL, must fail'),
}
ROT_SHA = '51149166b53dad386c93fa0682079aec96b61276801d9a6d4f622453d6426772'
FAST = dict(solver='lu', clip=True, lamcarry=True, pred2=True)


def accurate(L):
    """The paper's certified accurate variant at this mesh (burgers-eqcert / hires-burgers)."""
    if L <= 512:
        return dict(solver='chol', clip=True, lamcarry=True, pred2=True, exact_steps=1), 0.01
    return dict(solver='chol', clip=True, lamcarry=True, pred2=True), 0.001


def arms_for(L, keep_parent, ladder=LADDER):
    acc, gacc = accurate(L)
    arms = []
    for Rp in sorted(ladder, reverse=True):
        q = min(256, Rp - K)
        arms.append(dict(model='trunc', Rp=Rp, q=0, M=64, rule='q0scaled', gtol=0.001, variant=FAST, family='a'))
        arms.append(dict(model='lin', Rp=Rp, M=4 * Rp, rule='lat64', gtol=gacc, variant=acc, family='b'))
        arms.append(dict(model='trunc', Rp=Rp, q=q, M=4 * (K + q), rule='lat64', gtol=gacc, variant=acc, family='c'))
    arms.append(dict(model='trunc', Rp=512, q=256, M=1088, rule='bad0', gtol=gacc, variant=acc, family='control',
                     untimed=True))
    if keep_parent:
        arms.append(dict(model='parent', q=0, M=64, rule='q0scaled', gtol=0.001, variant=FAST, family='parent',
                         role='paper_fast_unrotated'))
        arms.append(dict(model='parent', q=256, M=1088, rule='lat64', gtol=gacc, variant=acc, family='parent',
                         role='paper_accurate_unrotated'))
    return arms


def arm_name(s):
    """Mirror of bankknob.arm_name (kept import-free so this runs without JAX)."""
    v = s['variant']
    sfx = ''.join(f'_{k}' for k in ([v['solver']] if v.get('solver', 'lu') != 'lu' else []) +
                  [k for k in ('clip', 'lamcarry', 'pred2') if v.get(k)])
    if v.get('exact_steps'):
        sfx += f"_x{v['exact_steps']}"
    g = f"g{s['gtol']:g}".replace('-', 'm').replace('.', 'p')
    head = 'P' if s['model'] == 'parent' else f"R{s['Rp']}"
    body = f"lin_M{s['M']}" if s['model'] == 'lin' else f"q{s.get('q', 0)}_M{s['M']}"
    return f"{head}_{body}_{s['rule']}_{g}_fast{sfx}"


def base(L, attempt):
    keep_parent = L <= 2048
    arms = arms_for(L, keep_parent)
    names = {(s['family'], s.get('Rp')): arm_name(s) for s in arms}
    pairs = []
    if keep_parent:
        pairs = [[names[('a', 512)], arm_name(arms[-2])], [names[('c', 512)], arm_name(arms[-1])]]
    pop = dict(source_draw=[20260922 if L == 256 else 20260921, 56],
               cert_draws=[list(range(8 * d, 8 * d + 8)) for d in range(5)], confirm_draw=list(range(40, 56)),
               note='burgers-eqcert populations: 256^2 used params_draw(20260922,56) (bc256b), every other mesh '
                    'params_draw(20260921,56); 5 certification draws of 8 trajectories + 1 confirmation draw of 16; '
                    'rho on the states each arm itself reaches, at THIS mesh')
    audit = ['fft_tight', names[('a', 512)], names[('c', 512)], names[('b', 512)], names[('b', 128)]]
    return dict(
        attempt=attempt, intervals=L, dt=0.005, eval_seed=7090702, eval_cases=4, eval_fresh_seed=911702,
        eval_fresh_cases=2,
        expected_physical_sha256='108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a',
        cohort_name='dev6: params_draw(7090702,4) + params_draw(911702,2), opened development cases',
        train_seed=0, train_trajectories=128, decoder_code_subsample=8192, cold_axis_points=48,
        strict=dict(ic_budget=400, step_budget=600), ic_gtol=1e-6, inner_damping=1e-10, rho_bar=0.116,
        directions_file='directions_qtd02.npz',
        directions_sha256='79d794580dad15c03630470940361d8ba9a7779673be9697c90d28e5c3d81535',
        rotation_sha256=ROT_SHA, ladder=LADDER, keep_parent=keep_parent, population=pop, rules=RULES, arms=arms,
        parity_pairs=pairs, parity_bar=1e-10, fom_settings=FOM, same_grid_reference='fft_tight',
        audit_cases=[0], audit_arms=audit, restrict_to=256,
        target_chunk={256: 16, 512: 8, 1024: 4, 2048: 2, 4096: 1}.get(L, 4), dense_tangent_group=34,
        phi_gate_max_mesh=512, reps=5, required_reps=5, order_seed=20260923, burn_seconds=0.25,
        slow_threshold_seconds=1.0, cooldown_after_seconds=0.5, cooldown_max_seconds=2.0,
        fom_subsets=dict(b_panel=[f['name'] for f in FOM if 'impl' not in f],
                         lean=[f['name'] for f in FOM if f.get('impl') == 'lean']),
        purpose='DESIGN.md: nested bank truncation R\' of the frozen R=512 Burgers model at this mesh, one allocation; '
                'dev6 selection panel; per-arm deployed-state rho re-measured at this mesh')


def smoke():
    c = base(128, 'smoke128')
    c['rules'] = dict(RULES, q0scaled=dict(lattice=32, note='SMOKE ONLY: stands in for the q0 file rule, which '
                                                             'needs L % 256 == 0'),
                      bad0=dict(lattice=16, control=True, note='SMOKE ONLY'))
    keep = arms_for(128, True, ladder=[512, 64])
    c['arms'] = keep
    c['ladder'] = LADDER
    c.update(eval_cases=1, eval_fresh_cases=1, expected_physical_sha256=None, local_smoke_waives_cohort_hash=True,
             reps=1, required_reps=1, fom_settings=[FOM[0], FOM[10]], burn_seconds=0.02,
             population=dict(c['population'], cert_draws=[[0], [1]], confirm_draw=[2]),
             audit_arms=['fft_tight', arm_name(keep[0])], phi_gate_max_mesh=128,
             fom_subsets=dict(lean=[FOM[10]['name']]))
    return c


def hold64():
    """The 4096^2 held-out job, generated ONLY from the committed dev6 selection (selection-4096.json): the chosen
    accurate and fast arms, plus the paper's current fast/accurate settings (R'=512 (a) and (c)), one allocation, the
    lean Newton-BiCGStab grid (the implementation the 4096^2 rows use) with fft_tight as the untimed reference."""
    sel = json.loads((HERE / 'selection-4096.json').read_text())
    c = base(4096, 'bkh64')
    allarms = {arm_name(a): a for a in arms_for(4096, False)}
    want = [sel['accurate'], sel['fast'], sel['current_accurate'], sel['current_fast']] + \
        [x for x in (sel.get('sensitivity_accurate'), sel.get('sensitivity_fast')) if x]
    c['arms'] = [dict(allarms[n], certify=False) for n in dict.fromkeys(want)]
    c.update(eval_draws=[[20260916, 64]], expected_physical_sha256=None, skip_certificates=True,
             cohort_name='hold64: params_draw(20260916, 64), held-out; never used to fit the bank, head, directions, '
                         'rotation, rules or the selection',
             fom_settings=[FOM[0]] + [f for f in FOM if f.get('impl') == 'lean'], untimed_fom=['fft_tight'],
             audit_arms=['fft_tight'] + list(dict.fromkeys(want))[:2], selection_source_summary_sha256=sel['source_summary_sha256'],
             fom_subsets=dict(lean=[f['name'] for f in FOM if f.get('impl') == 'lean']))
    return c


def explore(L):
    """EXPLORATORY (DESIGN amendment A2), not an input of the pre-registered selection: can the linear rung at high
    R' be made to pass the lat64 certificate (its failures sit at the initial state k = 0) by (i) fewer weak tests,
    M = 2R', or (ii) an exact first step (x1), and at what cost. References: the rule's picks at this mesh."""
    c = base(L, f'bx{L}')
    acc, gacc = accurate(L)
    x1 = dict(acc, exact_steps=1)
    arms = []
    for Rp in (512, 384, 256):
        arms.append(dict(model='lin', Rp=Rp, M=2 * Rp, rule='lat64', gtol=gacc, variant=acc, family='explore'))
    for Rp in (384, 256):
        arms.append(dict(model='lin', Rp=Rp, M=4 * Rp, rule='lat64', gtol=gacc, variant=x1, family='explore'))
    arms.append(dict(model='lin', Rp=384, M=4 * 384, rule='lat64', gtol=gacc, variant=acc, family='b'))
    arms.append(dict(model='trunc', Rp=384, q=256, M=1088, rule='lat64', gtol=gacc, variant=acc, family='c'))
    arms.append(dict(model='lin', Rp=128, M=512, rule='lat64', gtol=gacc, variant=acc, family='b'))
    arms.append(dict(model='trunc', Rp=512, q=256, M=1088, rule='bad0', gtol=gacc, variant=acc, family='control',
                     untimed=True))
    c.update(arms=arms, keep_parent=False, parity_pairs=[],
             audit_arms=['fft_tight', arm_name(arms[0]), arm_name(arms[3])],
             purpose='EXPLORATORY (DESIGN A2): linear-rung certificate remedies at this mesh; not a selection input')
    return c


def main():
    (HERE / 'config-x2048.json').write_text(json.dumps(explore(2048), indent=1) + '\n')
    if (HERE / 'selection-4096.json').exists():
        (HERE / 'config-h64.json').write_text(json.dumps(hold64(), indent=1) + '\n')
    for L in (256, 512, 1024, 2048, 4096):
        (HERE / f'config-{L}.json').write_text(json.dumps(base(L, f'bk{L}'), indent=1) + '\n')
    (HERE / 'config-smoke128.json').write_text(json.dumps(smoke(), indent=1) + '\n')


if __name__ == '__main__':
    main()
