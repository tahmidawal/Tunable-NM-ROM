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
if __name__ == '__main__':
    for r in (256, 320):
        name = f'vp_R{r}'
        (HERE / f'train_{name}.json').write_text(json.dumps(dict(training=training(r)), indent=1) + '\n')
        for k in (16, 32):
            cfg = panel(name, k, LADDER[r]['ladder'], LADDER[r]['extra'], [('validation', 921777, 256)], [64, 128], 256, 8, 32, 16)
            (HERE / f'val_{name}_K{k}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('written')
