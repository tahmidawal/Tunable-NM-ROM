"""b2d-query-speed: job configs = iso-error config-sweep256.json with the fields listed in CHANGED replaced.
    python make_config.py <name>     ->  configs/<name>.json
Development configs evaluate dev38 only (PROTOCOL.md section 3). The final configs read frozen.json.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = [32, 64, 96, 128, 144, 160, 176, 192, 256, 384, 512]
PRIMARY_GRAPHS = {96: True, 128: True, 192: False, 256: False, 384: True, 512: True}   # iso-error PROTOCOL section 2
DEV38 = [[7090702, 4], [911702, 2], [2026111401, 32]]
TEST64 = [[20260916, 64]]
FRESH64 = [[2026111402, 64]]
TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
CHANGED = ['arms', 'attempt', 'audit_arms', 'audit_cases', 'batch', 'cohort_name', 'diagnostics', 'disjoint_draws',
           'eval_draws', 'expected_physical_sha256', 'fom_both_modes', 'fom_settings', 'ladder', 'purpose',
           'recon_reps', 'rom_reps', 'rules', 'test_arms', 'prior_heldout', 'allow_cpu_smoke', 'case_subset',
           'local_smoke_waives_cohort_hash', 'frozen']


def base_arm(Rp, M=None, rule='lat64', gtol=0.01, graphs=None, proto=None):
    s = copy.deepcopy(proto)
    s.update(Rp=Rp, M=M or 4 * Rp, rule=rule, gtol=gtol, graphs=PRIMARY_GRAPHS.get(Rp, True) if graphs is None else graphs,
             role='baseline', primary=True)
    return s


def qs_arm(Rp, tag, graphs=True, M=None, rule='lat64', **qs):
    return dict(model='lin', Rp=Rp, M=M or 4 * Rp, rule=rule, impl='qs', graphs=graphs, tag=tag, qs=qs,
                family='lin', certify=False, role='qs')


def arm_name(a):
    return f"R{a['Rp']}_lin_M{a['M']}_{a['rule']}__qs_{a['tag']}" + ('_graphs' if a['graphs'] else '')


def qs_grid(Rp):
    a = []
    for k in (1, 2, 3):
        for s in (1, 5, 10, 50):
            a.append(qs_arm(Rp, f'c_k{k}_s{s}_P', k=k, s=s))
    for k in (1, 2):
        for s in (10, 50):
            a.append(qs_arm(Rp, f'c_k{k}_s{s}_fused', k=k, s=s, form='fused'))
        for s in (1, 50):
            a.append(qs_arm(Rp, f'c_k{k}_s{s}_chol', k=k, s=s, form='chol'))
    for k in (1, 2, 3, 4):
        a.append(qs_arm(Rp, f'pic_k{k}_P', k=k, s=50, jref='A'))
    for k in (1, 2):
        a.append(qs_arm(Rp, f'pic_k{k}_fused', k=k, s=50, jref='A', form='fused'))
    for k in (1, 2, 3):
        for s in (10, 50):
            a.append(qs_arm(Rp, f'g_k{k}_s{s}_P', k=k, s=s, mode='grad'))
    for k in (1, 2):
        a.append(qs_arm(Rp, f'c_k{k}_s50_P_f32', k=k, s=50, stencil32=True))
        a.append(qs_arm(Rp, f'c_k{k}_s50_fused_f32', k=k, s=50, form='fused', stencil32=True))
    a.append(qs_arm(Rp, 'c_k2_s50_P_unroll', k=2, s=50, emit='unroll'))
    a.append(qs_arm(Rp, 'c_k1_s50_fused_unroll', k=1, s=50, form='fused', emit='unroll'))
    a.append(qs_arm(Rp, 'c_k2_s50_fused_unroll', k=2, s=50, form='fused', emit='unroll'))
    a.append(qs_arm(Rp, 'c_k2_s50_P', graphs=False, k=2, s=50))
    a.append(qs_arm(Rp, 'c_k2_s50_fused', graphs=False, k=2, s=50, form='fused'))
    a.append(qs_arm(Rp, 'c_k2_s50_P_lin', k=2, s=50, pred='lin'))
    a.append(qs_arm(Rp, 'c_k3_s50_P_nopred', k=3, s=50, pred='none'))
    a.append(qs_arm(Rp, 'c_k1_s50_P_first3', k=1, s=50, kfirst=3))
    a.append(qs_arm(Rp, 'c_k2_s50_P_first4', k=2, s=50, kfirst=4))
    a.append(qs_arm(Rp, 'c_k2_s50_P_ridge1em6', k=2, s=50, ridge=1e-6))
    return a


def shortlist(Rp):
    """Round 2 (after explore128/explore384, dev38): one refresh per query (s = 50) throughout."""
    a = []
    for form in ('P', 'fused'):
        for k in (1, 2):
            a.append(qs_arm(Rp, f'c_k{k}_s50_{form}', k=k, s=50, form=form))
            a.append(qs_arm(Rp, f'c_k{k}_s50_{form}_f32', k=k, s=50, form=form, stencil32=True))
    for n in (5, 10, 25):
        a.append(qs_arm(Rp, f'c_k1_s50_fused_f32_first{n}x2', k=1, s=50, form='fused', stencil32=True, kfirst=2, nfirst=n))
    a.append(qs_arm(Rp, 'c_k1_s50_fused_f32_first2x3', k=1, s=50, form='fused', stencil32=True, kfirst=3, nfirst=2))
    a.append(qs_arm(Rp, 'c_k1_s50_fused_f32_first5x3', k=1, s=50, form='fused', stencil32=True, kfirst=3, nfirst=5))
    a.append(qs_arm(Rp, 'c_k3_s50_fused_f32', k=3, s=50, form='fused', stencil32=True))
    a.append(qs_arm(Rp, 'c_k1_s25_fused_f32', k=1, s=25, form='fused', stencil32=True))
    return a


def build(name):
    base = json.loads((HERE / '../iso-error/config-sweep256.json').read_text())
    proto = base['arms'][0]
    cfg = copy.deepcopy(base)
    for k in ('test_arms', 'prior_heldout'):
        cfg.pop(k, None)
    cfg.update(attempt=name, ladder=LADDER, audit_arms=[], audit_cases=[], recon_reps=0, rom_reps=2,
               eval_draws=DEV38, expected_physical_sha256=None, disjoint_draws=TEST64 + FRESH64,
               cohort_name='dev38 = params_draw(7090702,4) ++ params_draw(911702,2) ++ params_draw(2026111401,32)',
               fom_settings=[fs for fs in base['fom_settings'] if fs['name'] == 'fft_tight'], fom_both_modes=False)
    cfg['rules'] = dict(lat64=base['rules']['lat64'],
                        lat128=dict(lattice=128, note='uniform 127x127 interior sub-lattice (diagnosis only)'))
    B = lambda *a_, **k_: base_arm(*a_, proto=proto, **k_)
    if name == 'smoke':
        cfg.update(allow_cpu_smoke=True, eval_draws=[[7090702, 2]], ladder=[32, 64, 512], rom_reps=2,
                   diagnostics=dict(rule='lat64', width_M=[[32, 128], [64, 256]], jac_cases=[0]),
                   purpose='local CPU smoke of the driver and qsfast (no number from it is reported)')
        arms = [B(32), B(64)]
        for Rp in (32, 64):
            arms += [qs_arm(Rp, 'c_k2_s50_P', k=2, s=50), qs_arm(Rp, 'c_k2_s10_fused', k=2, s=10, form='fused'),
                     qs_arm(Rp, 'c_k3_s1_chol', k=3, s=1, form='chol'), qs_arm(Rp, 'g_k3_s5_P', k=3, s=5, mode='grad'),
                     qs_arm(Rp, 'g_k3_s5_chol', k=3, s=5, mode='grad', form='chol'),
                     qs_arm(Rp, 'pic_k3_fused', k=3, s=50, jref='A', form='fused'),
                     qs_arm(Rp, 'c_k2_s50_P_f32', k=2, s=50, stencil32=True),
                     qs_arm(Rp, 'c_k1_s25_P_first3_unroll', k=1, s=25, kfirst=3, emit='unroll'),
                     qs_arm(Rp, 'c_k1_s25_P_first3', k=1, s=25, kfirst=3)]
        cfg['batch'] = dict(arms=['R32_lin_M128_lat64__qs_c_k2_s50_P_graphs'], sizes=[1, 3], reps=2)
    elif name in ('explore128', 'explore384'):
        Rp = int(name[7:])
        arms = [B(Rp)] + qs_grid(Rp)
        cfg['purpose'] = f'dev38 exploration and profile of the qs family at width {Rp} (PROTOCOL.md section 4)'
    elif name == 'diag512':
        arms = [B(Rp) for Rp in (96, 128, 144, 160, 176, 192, 256)]
        for Rp in (384, 512):
            arms += [B(Rp), B(Rp, gtol=1e-4), B(Rp, gtol=1e-6), B(Rp, M=3 * Rp), B(Rp, M=2 * Rp),
                     B(Rp, rule='lat128'), B(Rp, rule='lat128', gtol=1e-6), B(Rp, M=3 * Rp, gtol=1e-6)]
        cfg['diagnostics'] = dict(rule='lat64', width_M=[[192, 768], [256, 1024], [384, 1536], [512, 2048], [512, 1536],
                                                         [384, 1152]], jac_cases=[0, 1, 2, 3, 4, 5])
        cfg['purpose'] = 'dev38: baseline at all widths; width-512 diagnosis arms (PROTOCOL.md section 6)'
    elif name in ('sello', 'selhi'):
        arms = []
        for Rp in ((96, 128, 144, 160, 176) if name == 'sello' else (192, 256, 384, 512)):
            arms += [B(Rp)] + shortlist(Rp)
        cfg['purpose'] = 'dev38 selection run: baseline and the shortlisted qs settings per width (PROTOCOL.md section 5)'
    elif name == 'diag2':
        arms = []
        for Rp in (256, 384, 512):
            arms.append(B(Rp))
            for f in (6, 8):
                if f * Rp <= 3969:
                    arms.append(B(Rp, M=f * Rp))
                arms.append(B(Rp, M=f * Rp, rule='lat128'))
            arms.append(qs_arm(Rp, 'c_k2_s50_P', k=2, s=50))
            if 6 * Rp <= 3969:
                arms.append(qs_arm(Rp, 'c_k2_s50_P', M=6 * Rp, k=2, s=50))
        cfg['diagnostics'] = dict(rule='lat64', width_M=[[384, 2304], [384, 3072], [512, 3072]], jac_cases=[0])
        cfg['purpose'] = 'dev38: second width-512 diagnosis, more tests than 4R\' (PROTOCOL.md section 9, amendment A1)'
    elif name in ('ftest64', 'ffresh64', 'fdev'):
        fz = json.loads((HERE / 'frozen.json').read_text())
        arms, batch = [], []
        for Rp in ((128, fz['equal_error_width'], 384) if name == 'fdev' else fz['final_widths']):
            w = fz['widths'][str(Rp)]
            arms.append(B(Rp))
            for kind in ('accurate', 'fast'):
                if w[kind]:
                    a = qs_arm(Rp, w[kind]['tag'], graphs=w[kind]['graphs'], **w[kind]['qs'])
                    a['role'] = kind
                    arms.append(a)
                    assert arm_name(a) == w[kind]['name'], (arm_name(a), w[kind]['name'])
                    if kind == 'accurate' and Rp in (128, fz['equal_error_width'], 384):
                        batch.append(w[kind]['name'])
        test = name == 'ftest64'
        cfg.update(eval_draws=TEST64 if test else FRESH64, expected_physical_sha256=TEST_SHA if test else None,
                   disjoint_draws=DEV38 + (FRESH64 if test else TEST64),
                   cohort_name='test64 = params_draw(20260916,64)' if test else 'fresh64 = params_draw(2026111402,64)',
                   batch=dict(arms=batch, sizes=[1, 8, 64], reps=7), frozen=fz,
                   purpose='FINAL single evaluation of the frozen settings (PROTOCOL.md section 7)')
        if name == 'fdev':
            cfg.update(eval_draws=DEV38[:2], expected_physical_sha256=None, disjoint_draws=TEST64 + FRESH64,
                       cohort_name='dev6 = params_draw(7090702,4) ++ params_draw(911702,2) (first six cases of dev38)',
                       purpose='rehearsal of the final job on six dev38 cases (no sealed cohort); not a reported result')
    else:
        raise SystemExit(f'unknown config {name}')
    cfg['arms'] = arms
    extra = sorted(k for k in cfg if cfg[k] != base.get(k) and k not in CHANGED)
    assert not extra, extra
    return cfg


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        cfg = build(nm)
        (HERE / 'configs').mkdir(exist_ok=True)
        (HERE / f'configs/{nm}.json').write_text(json.dumps(cfg, indent=1) + '\n')
        print(nm, len(cfg['arms']), 'arms')
