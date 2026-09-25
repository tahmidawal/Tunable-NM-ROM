"""Generate config-1024.json, config-4096.json and config-smoke128.json for burgers-eq-tol-knobs (DESIGN.md).

Common keys (cohort, checkpoint-side constants, certificate populations, rho bar, FOM grid, timing protocol) are
copied from the configs that produced the Table-1 rows, so nothing but the arm list changes:
  1024^2: experiments/burgers2d-speed/config-1024.json (job b1024 4241031, engineered E1-E4 code)
  4096^2: experiments/burgers-bank-knob/config-4096.json (job bk4096b 4197473, the bank-knob query text = `parent`)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
VARIANT = dict(solver='chol', clip=True, lamcarry=True, pred2=True)
GTOL0 = 1e-3                       # the current (Table-1) stopping tolerance at 1024^2 and 4096^2
GTOLS = [1e-1, 1e-2, 1e-4]         # the ladder's other rungs (1e-3 is the current rule's arm)
RULES = {                          # name -> lattice (sx, sy); lat64 is the Table-1 rule (63 x 63 = 3969 nodes)
    'lat32': [32, 32],             # 31 x 31   =   961 nodes (0.242x)
    'lat32x64': [32, 64],          # 31 x 63   =  1953 nodes (0.492x)
    'lat64': 64,                   # 63 x 63   =  3969 nodes (1x, the current rule, config unchanged)
    'lat64x128': [64, 128],        # 63 x 127  =  8001 nodes (2.016x)
    'lat128': [128, 128],          # 127 x 127 = 16129 nodes (4.064x)
}
RECORDED = {   # worst evolved % of the Table-1 settings in the job that produced the Table-1 row (parity gate)
    1024: {'R384': dict(percent=0.2108198098221833, job='4241031 (b1024)',
                        source='experiments/burgers2d-speed/checks/b1024-summary.json knobs R384 ... __eng'),
           'R96': dict(percent=2.787033939328547, job='4241031 (b1024)',
                       source='experiments/burgers2d-speed/checks/b1024-summary.json knobs R96 ... __eng_graphs')},
    4096: {'R384': dict(percent=0.2239874024132454, job='4197473 (bk4096b)',
                        source='experiments/burgers-bank-knob/checks/bk4096-summary.json table R384_lin_M1536 ...'),
           'R64': dict(percent=4.830435554687164, job='4197473 (bk4096b)',
                       source='experiments/burgers-bank-knob/checks/bk4096-summary.json table R64_lin_M256 ...')},
}


def arm(Rp, rule, gtol, impl, graphs, dense=False, certify=True, role=None):
    v = dict(VARIANT)
    if dense:
        v['exact_steps'] = 50      # every backward-Euler step on the exact (all-node) residual: no quadrature
    return dict(model='lin', Rp=Rp, M=4 * Rp, rule=rule, gtol=gtol, variant=v, cap=None, impl=impl, graphs=graphs,
                family='lin', certify=bool(certify and not dense), role=role)


def arms_for(Rps, impl, modes, dense):
    out = []
    for Rp in Rps:
        specs = [('lat64', GTOL0, False, 'current')]
        specs += [(r, GTOL0, False, 'eq-ladder') for r in RULES if r != 'lat64']
        specs += [('lat64', g, False, 'tol-ladder') for g in GTOLS]
        if dense:
            specs += [('lat64', GTOL0, True, 'dense')]
        for rule, g, dn, role in specs:
            for gr in modes:
                # certify only one arm per knob (the graphs arm where both modes run), as burgers2d-speed
                out.append(arm(Rp, rule, g, impl, gr, dense=dn, certify=(gr == modes[-1]), role=role))
    return out


def rules_block(base):
    r = {k: v for k, v in base['rules'].items() if k == 'lat64'}
    for k, v in RULES.items():
        if k != 'lat64':
            r[k] = dict(lattice=v, note=f'tensor sub-lattice {v[0] - 1} x {v[1] - 1} = {(v[0] - 1) * (v[1] - 1)} nodes, '
                                        f'equal weights (L/{v[0]})(L/{v[1]}); same physical points at every L')
    return r


def build():
    b1024 = json.loads((EXP / 'burgers2d-speed/config-1024.json').read_text())
    bk4096 = json.loads((EXP / 'burgers-bank-knob/config-4096.json').read_text())
    s128 = json.loads((EXP / 'burgers2d-speed/config-smoke128.json').read_text())

    c = dict(b1024)
    c.update(attempt='e1024', rules=rules_block(b1024), arms=arms_for([384, 96], 'eng', [False, True], dense=True),
             purpose='DESIGN.md: EQ node-count ladder, dense residual and gtol ladder at the Table-1 settings, 1024^2',
             recorded_table1=RECORDED[1024], order_seed=20260925)
    c.pop('parent_fast_bar', None)
    c['audit_arms'] = ['fft_tight'] + [n for n in arm_names(c['arms']) if n.endswith('_graphs') and
                                       ('_g0p001_' in n and ('lat64_' in n))]
    c['parity_pairs'] = [[n, n[:-len('_graphs')]] for n in arm_names(c['arms']) if n.endswith('_graphs')]
    (HERE / 'config-1024.json').write_text(json.dumps(c, indent=1) + '\n')

    d = dict(b1024)                       # the b2speed-driver key set, with the bank-knob 4096 values where they differ
    for k in ('intervals', 'ladder', 'population', 'target_chunk', 'dense_tangent_group', 'burn_seconds'):
        d[k] = bk4096[k]
    d.update(attempt='e4096', rules=rules_block(b1024), arms=arms_for([384, 64], 'parent', [False], dense=False),
             purpose='DESIGN.md: EQ node-count ladder and gtol ladder at the Table-1 settings, 4096^2 (bank-knob text)',
             recorded_table1=RECORDED[4096], order_seed=20260926, parity_pairs=[], fom_both_modes=False,
             audit_cases=[], audit_arms=[], attempt_note='e4096 (H200) never started: H200 queue estimate 2 days; '
                                                         'e4096b on A100-80G with bank_columns 384 (DESIGN A1)',
             bank_columns=384, ladder=[32, 64, 128, 256, 384], bank_blocks=24)
    d['attempt'] = 'e4096c'
    d['attempt_note'] += ('; e4096b (4304018) OOM in the bank build (12 GiB row block beside the rotated bank): '
                          'e4096c uses 24 row blocks (DESIGN A2)')
    d.pop('parent_fast_bar', None)
    (HERE / 'config-4096.json').write_text(json.dumps(d, indent=1) + '\n')

    s = dict(s128)
    allarms = arms_for([32], 'eng', [False, True], dense=True)
    s.update(attempt='smk', rules=rules_block(b1024),
             arms=[a for a in allarms if a['role'] in ('current', 'dense') or a['rule'] == 'lat32x64'],
             recorded_table1={}, order_seed=1)
    s['parity_pairs'] = [[n, n[:-len('_graphs')]] for n in arm_names(s['arms']) if n.endswith('_graphs')]
    s['fom_settings'] = [f for f in s['fom_settings'] if f['name'] in ('fft_tight', 'lean_nt3e-3_l3e-3_dt005')]
    s['population'] = dict(s['population'], cert_draws=[[0, 1]], confirm_draw=[2, 3])
    s['audit_arms'] = ['fft_tight']
    s.pop('parent_fast_bar', None)
    (HERE / 'config-smoke128.json').write_text(json.dumps(s, indent=1) + '\n')
    for cc in (c, d, s):
        print(cc['attempt'], len(cc['arms']), 'arms', len(cc['parity_pairs']), 'parity pairs')


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def arm_name(s):
    """eqtol.arm_name (= b2speed.arm_name) verbatim; checked against the job's arm_setup by the audit."""
    v = s['variant']
    sfx = ''.join(f'_{k}' for k in ([v['solver']] if v.get('solver', 'lu') != 'lu' else []) +
                  [k for k in ('clip', 'lamcarry', 'pred2') if v.get(k)])
    if v.get('exact_steps'):
        sfx += f"_x{v['exact_steps']}"
    head = f"R{s['Rp']}"
    body = f"lin_M{s['M']}" if s['model'] == 'lin' else f"q{s.get('q', 0)}_M{s['M']}"
    base = f"{head}_{body}_{s['rule']}_{gt(s['gtol'])}_fast{sfx}"
    if s.get('cap'):
        base += f"_cap{s['cap']}"
    if s.get('budget'):
        base += f"_budget{s['budget']}"
    base += '__' + s.get('impl', 'parent') + ('_graphs' if s.get('graphs') else '')
    return base


def arm_names(arms):
    return [arm_name(a) for a in arms]


if __name__ == '__main__':
    build()
