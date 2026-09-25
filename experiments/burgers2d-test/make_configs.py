"""Generate the burgers2d-test configs (edit this file, never the JSON). DESIGN.md sections 2-4.

config-t{256,512,1024,2048,4096}.json   test64 panels: the frozen Table-1 arms on hold64 = params_draw(20260916, 64)
config-smoke128.json                    local smoke (not a result)

Every arm is built by experiments/burgers2d-speed/make_configs.py (`lin`, `base`) so its spec -- and therefore its
name, variant, gtol, rule, M -- is the one that produced the development and earlier held-out rows. The compile mode of
each engineered (`eng`) arm at 256^2-1024^2 is FROZEN from the development panel (the knob's faster mode in
checks/b<L>-summary.json `knobs[..]['mode']`), exactly as the parent's held-out configs froze the selected mode.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
B2S = EXP / 'burgers2d-speed'
BKL = EXP / 'burgers-bank-knob'
_spec = importlib.util.spec_from_file_location('b2s_make_configs', B2S / 'make_configs.py')
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)

HOLD64 = [[20260916, 64]]
HOLD64_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'   # h256/h512/h1024 and bkh64f
# The Table-1 widths (DESIGN.md section 2): accurate, fast, and the extra width at each mesh
WIDTHS = {256: dict(accurate=384, fast=96, extra=[128]),
          512: dict(accurate=384, fast=96, extra=[128]),
          1024: dict(accurate=384, fast=96, extra=[128]),
          2048: dict(accurate=384, fast=64, extra=[128]),
          4096: dict(accurate=384, fast=64, extra=[128])}
# code path of the PRIMARY arms = the code that produced the development Table-1 row at that mesh
PRIMARY_IMPL = {256: 'eng', 512: 'eng', 1024: 'eng', 2048: 'parent', 4096: 'parent'}
# earlier held-out summaries on the SAME 64 cases: arms they share with this lane must reproduce their worst error
PRIOR = {256: ('experiments/burgers2d-speed/checks/h256-summary.json', '4242040'),
         512: ('experiments/burgers2d-speed/checks/h512-summary.json', '4242063'),
         1024: ('experiments/burgers2d-speed/checks/h1024-summary.json', '4242036'),
         4096: ('experiments/burgers-bank-knob/checks/bkh64-summary.json @ 340627ca', '4239635')}
UNTIMED_4096 = ['fft_tight', 'lean_tight', 'nt1e-4_dt005', 'lean_nt1e-4_dt005']


def dev_mode(L, Rp):
    """The dev6 panel's faster compile mode of the engineered linear-rung knob R' (no cap, no exact step)."""
    s = json.loads((B2S / 'checks' / f'b{L}-summary.json').read_text())
    hit = [v for v in s['knobs'].values() if v['model'] == 'lin' and v['R_prime'] == Rp and not v['cap']
           and not v['exact_steps']]
    assert len(hit) == 1, (L, Rp, len(hit))
    return hit[0]['mode'] == 'graphs'


def arm(L, Rp, role, impl, graphs):
    x1 = False if L <= 512 else None           # Table 1 drops the exact first step at 256^2/512^2 (burgers2d-speed)
    s = MC.lin(L, Rp, x1=x1, impl=impl, graphs=graphs, certify=False)
    s['role'] = role
    return s


def arms_for(L):
    w = WIDTHS[L]
    roles = [(w['accurate'], 'accurate'), (w['fast'], 'fast')] + [(r, 'extra') for r in w['extra']]
    out = []
    for Rp, role in roles:
        if PRIMARY_IMPL[L] == 'eng':
            out.append(dict(arm(L, Rp, role, 'eng', dev_mode(L, Rp)), primary=True))
        else:
            out.append(dict(arm(L, Rp, role, 'parent', False), primary=True))
    if L == 2048:
        # secondary (reported, never a Table-1 number): the engineered path of the 256^2-1024^2 rows, both modes
        for Rp, role in roles:
            for gr in (True, False):
                out.append(dict(arm(L, Rp, role, 'eng', gr), primary=False, candidate=False))
    return out


def test(L):
    c = MC.base(L, f't{L}')                  # the parent's config skeleton; arms / cohort / protocol replaced below
    arms = arms_for(L)
    names = [MC.arm_name(s) for s in arms]
    assert len(names) == len(set(names))
    prim = {s['role']: MC.arm_name(s) for s in arms if s['primary']}
    c.update(attempt=f't{L}', intervals=L, arms=arms, eval_draws=HOLD64, expected_physical_sha256=HOLD64_SHA,
             cohort_name='hold64 = test64: params_draw(20260916, 64); never used for any fit, rule, setting or selection',
             skip_certificates=True, parity_pairs=[], audit_cases=[0],
             audit_arms=['fft_tight'] + [prim['accurate'], prim['fast']] + [MC.arm_name(s) for s in arms
                                                                         if s['primary'] and s['role'] == 'extra'],
             rom_reps=2, fom_reps=1, fom_mode_parity=False, save_restricted_skip_graphs_fom=True,
             timed_full_sha_every={256: 1, 512: 1, 1024: 1, 2048: 4, 4096: 8}[L],
             target_chunk={256: 16, 512: 8, 1024: 4, 2048: 2, 4096: 1}[L],
             parent_fast_bar=dict(percent=None, arm=None, source='not used (test panel: no selection)'),
             test_arms=dict(primary=prim, primary_impl=PRIMARY_IMPL[L],
                            secondary=[MC.arm_name(s) for s in arms if not s['primary']]),
             prior_heldout=dict(source=PRIOR[L][0], job=PRIOR[L][1]) if L in PRIOR else None,
             purpose='DESIGN.md: frozen Table-1 Burgers 2D settings on the 64 test cases, FOM ladder in the same job')
    if L == 4096:
        c.update(ladder=[32, 64, 128, 256, 384], bank_columns=384, bank_blocks=24, fom_both_modes=False,
                 untimed_fom=UNTIMED_4096, burn_seconds=0.25)
    return c


def smoke():
    c = test(1024)
    c = dict(c, attempt='smoke128', intervals=128, expected_physical_sha256=None, local_smoke_waives_cohort_hash=True,
             eval_draws=[[7090702, 2]], rom_reps=1, fom_reps=1, burn_seconds=0.01, phase_cooldown_seconds=0.1,
             phase_dummy_seconds=0.1, fom_settings=[MC.FOM[0], MC.FOM[10]], phi_gate_max_mesh=128,
             bank_columns=384, ladder=[32, 64, 96, 128, 192, 256, 384], bank_blocks=2,
             arms=[arm(128, 96, 'fast', 'eng', True) | dict(primary=True),
                   arm(128, 64, 'fast', 'parent', False) | dict(primary=True)],
             cohort_name='SMOKE ONLY (2 dev cases at 128^2; not a result)')
    c['audit_arms'] = ['fft_tight', MC.arm_name(c['arms'][0])]
    c['test_arms'] = dict(primary=dict(accurate=MC.arm_name(c['arms'][0]), fast=MC.arm_name(c['arms'][1])),
                          primary_impl='mixed', secondary=[])
    c['prior_heldout'] = None
    return c


def main():
    for L in (256, 512, 1024, 2048, 4096):
        (HERE / f'config-t{L}.json').write_text(json.dumps(test(L), indent=1) + '\n')
    (HERE / 'config-smoke128.json').write_text(json.dumps(smoke(), indent=1) + '\n')
    for p in sorted(HERE.glob('config-*.json')):
        print(hashlib.sha256(p.read_bytes()).hexdigest()[:16], p.name)


if __name__ == '__main__':
    main()
