"""Writes every heat3d-bank config (single source of truth for the pre-registered settings). See DESIGN.md."""
import copy, json
from pathlib import Path
HERE = Path(__file__).resolve().parent / 'configs'
TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def training(rank):
    return dict(d=3, family='h3d', train_intervals=32, times=TIMES, diffusivity=0.02,
                train_seed=921000, train_count=2048, validation_seed=921777, validation_count=256, model_seed=921002,
                fourier_features=128, fourier_scale=2.0, bank_width=512, bank_depth=3, bank_rank=rank,
                modes=1536, pod_reference_ranks=[128, 256, 320, 384], bank_steps=40000, bank_batch_states=512,
                bank_learning_rate=1e-3, bank_gradient_clip=1.0, warmup=500, tail_power=4, tail_weight=0.5, white_weight=1e-3,
                checkpoint_every=2000, floor_grids=[64, 128],
                head_width=512, latent_dimensions=[16, 32], head_steps=120000, head_batch_states=256, head_learning_rate=5e-4,
                head_checkpoint_every=10000)

CG = {'fom_cncg_dt0.025_rtol1e-6_NAMED': dict(dt=0.025, tolerance=1e-6)}
for dt in (0.025, 0.05, 0.1):
    for tol in (1e-4, 1e-3, 1e-2):
        CG[f'fom_cncg_dt{dt}_rtol{tol:.0e}'.replace('e-0', 'e-')] = dict(dt=dt, tolerance=tol)

def rom_arms(ladder, extra_cn_dt_q):
    arms = []
    for q in ladder:
        arms.append(dict(name=f'nmrom_q{q}_field_cn', q=q, opt=dict(init='field')))
        arms.append(dict(name=f'nmrom_q{q}_field_direct_tol1e-4_chol', q=q, opt=dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True)))
    for q in extra_cn_dt_q:
        for dt in (0.05, 0.1):
            arms.append(dict(name=f'nmrom_q{q}_field_cn_dt{dt}', q=q, opt=dict(init='field', dt=dt)))
    return arms

def panel(name, k, ladder, extra, cohorts, meshes, cases, timed, save, audit_intervals):
    return dict(model=dict(kind='mlp', bank=f'{name}/bank.pkl', head=f'{name}/head_K{k}.pkl'), times=TIMES, diffusivity=0.02,
                tests=dict(count=1024), cohorts=[[s, c] for _, s, c in cohorts], cohort_names=[[nm, c] for nm, _, c in cohorts],
                meshes=meshes, cases_by_mesh={str(m): cases for m in meshes}, timed_cases_by_mesh={str(m): timed for m in meshes},
                save_fields_cases=save, repetitions=5, burn_seconds=0.15, audit_intervals=audit_intervals, full_audit_cases=1,
                reference_budget=1e-4,
                rom_defaults=dict(init='field', stepping='cn', dt=0.025, starts=4, fit_budget=600, step_budget=120, tolerance=1e-6),
                rom_arms=rom_arms(ladder, extra), linear_bank_inits=['moments', 'field'], cg_arms=copy.deepcopy(CG),
                coarse_intervals=[32, 64], coarse_cg='fom_cncg_dt0.05_rtol1e-4')

LADDER = {256: dict(ladder=[0, 32, 64, 96, 128, 160, 192, 224], extra=[128, 192]),
          320: dict(ladder=[0, 32, 64, 96, 128, 192, 256, 288], extra=[128, 192])}
SPEED_Q = {256: [0, 128, 160, 192, 224], 320: [0, 128, 192, 256, 288]}
if __name__ == '__main__':
    for r in (256, 320):
        name = f'vp_R{r}'
        (HERE / f'train_{name}.json').write_text(json.dumps(dict(training=training(r)), indent=1) + '\n')
        for k in (16, 32):
            cfg = panel(name, k, LADDER[r]['ladder'], LADDER[r]['extra'], [('validation', 921777, 256)], [64, 128], 256, 8, 32, 16)
            cfg['save_selection_arms'] = '_field_cn'   # every validation case keeps the selection-driving arms for the NumPy audit
            (HERE / f'val_{name}_K{k}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    # ADDENDUM 1 (speed loop, validation only; see DESIGN.md addendum 1): CN stepping at LM stationarity tolerance 1e-4 with
    # Cholesky normal equations (hires-heat's accepted speed variant), same field init; direct arms repeated in the same allocation.
    for r in (256, 320):
        name = f'vp_R{r}'
        for k in (16, 32):
            qs = [q for q in SPEED_Q[r] if q + k <= r]
            cfg = panel(name, k, [], [], [('validation', 921777, 256)], [64, 128], 256, 16, 32, 32)
            cfg['save_selection_arms'] = '_field_cn'
            cfg['rom_arms'] = [a for q in qs for a in (
                dict(name=f'nmrom_q{q}_field_cn_tol1e-4_chol', q=q, opt=dict(init='field', tolerance=1e-4, cholesky=True)),
                dict(name=f'nmrom_q{q}_field_direct_tol1e-4_chol', q=q, opt=dict(init='field', stepping='exact_direct', tolerance=1e-4, cholesky=True)))]
            (HERE / f'speed_{name}_K{k}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    # ADDENDUM 2 (speed loop, validation only; DESIGN.md addendum 2): variants of the addendum-1 selection R256/K16, q in {0, 160}.
    V2 = {'cn_tol1e-4_chol': dict(), 'cn_tol1e-4_chol_dt0.05': dict(dt=0.05), 'cn_tol1e-4_chol_s1': dict(starts=1),
          'cn_tol1e-4_chol_dt0.05_s1': dict(dt=0.05, starts=1), 'cn_tol1e-4_chol_mom_dt0.05_s1': dict(init='moments', dt=0.05, starts=1),
          'direct_tol1e-4_chol': dict(stepping='exact_direct'), 'direct_tol1e-4_chol_s1': dict(stepping='exact_direct', starts=1),
          'direct_tol1e-4_chol_mom': dict(stepping='exact_direct', init='moments'),
          'direct_tol1e-4_chol_mom_s1': dict(stepping='exact_direct', init='moments', starts=1)}
    cfg = panel('vp_R256', 16, [], [], [('validation', 921777, 256)], [64, 128], 256, 16, 32, 32)
    cfg['save_selection_arms'] = 'nmrom_'
    cfg['rom_arms'] = [dict(name=f'nmrom_q{q}_field_{v}' if '_mom' not in v else f'nmrom_q{q}_{v}', q=q,
                            opt=dict(dict(init='field', tolerance=1e-4, cholesky=True), **o)) for q in (0, 160) for v, o in V2.items()]
    (HERE / 'speed2_vp_R256_K16.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('written')
