"""Write configs/panel_<problem><n>.json for one cell.

    make_panel_config.py <problem> <n> <cell-dir-relative-to-l3d, e.g. runs/tr_p32p64/output/poisson32> [--smoke]

The Table-1 settings are copied from the source lanes' configs (poisson-bank-knob-3d config-cube-*.json,
heat-bank-knob configs/h3d.json / h3d256.json) — no choice is made here. The operator list is every arm in
<cell-dir>/<arm>/result.json (the pulled training outputs of that cell) (one size per family, fixed before training; nothing is selected
from panel data). An arm without a complete result is listed with its failure.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    problem, n = sys.argv[1], int(sys.argv[2])
    run = Path(sys.argv[3])
    smoke = '--smoke' in sys.argv
    ops = []
    for arm_dir in sorted(p for p in (HERE / run).iterdir() if p.is_dir()):
        res_p = arm_dir / 'result.json'
        rec = dict(arm=arm_dir.name)
        if not res_p.exists():
            rec.update(family=arm_dir.name.split('-')[0], failure='no result.json (training did not finish)')
        else:
            res = json.loads(res_p.read_text())
            rec.update(family=res['family'])
            if not res.get('complete'):
                rec['failure'] = res.get('failure', 'incomplete training (signal or checkpoint not reproduced)')
            else:
                ck = arm_dir / 'best.pt'
                rec.update(checkpoint=(str(run / arm_dir.name / 'best.pt') if smoke else f'checkpoints/{arm_dir.name}/best.pt'), sha256=res['best_checkpoint_sha256'],
                           training=dict(epochs=res['epochs_completed'], steps=res['optimisation_steps'],
                                         best_epoch=res['best_epoch'], stop_reason=res['stop_reason'],
                                         micro_batch=res['micro_batch'], parameters=res['real_parameter_count'],
                                         validation_mean_case=res['validation_best_checkpoint']['mean_case'],
                                         validation_worst_case=res['validation_best_checkpoint']['worst_case'],
                                         job_id=res['job_id'], memory_fraction=res['memory_fraction'],
                                         training_seconds=res['training_seconds']))
                if ck.exists():
                    assert sha(ck) == res['best_checkpoint_sha256'], arm_dir
        ops.append(rec)
    common = dict(problem=problem, n=n, expected=f'expected/{problem}_{n}.json', order_seed=2026092400 + n,
                  timing_rounds=1 if smoke else 3, burn_seconds=0.05, reproduction_tolerance=1e-6, reproduction_tolerance_fom=1e-3, operators=ops,
                  smoke=dict())
    if problem == 'poisson':
        src = json.loads((HERE / 'expected' / f'poisson_{n}.json').read_text())
        cfg = dict(common, cohort=('final' if n in (32, 64) else 'development'), final_count=64,
                   final_parameters_sha256='27ec2cf52d2eb0ad84d6b3e83c46f504a67c0dd56a7a1f968263d3179430be60',
                   checkpoint_sha256=dict(bank='6c2298776709973b614b0ab3eed9fcd68d9d39e8d9639d40f1a8da5d0b973c63',
                                          head='258d28457800b3cceb7eab2e1f1075166f0287a246713034bb0bffb024409e01'),
                   field_chunk=8192, weak_tests=512, R_ladder=[16, 32, 48, 64, 96, 128], accurate_Rp=128, fast_Rp=64,
                   cg_iterations_per_interval=40, cg_tolerances=[0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
                   source_table1_job=src['job_id'], table1_fom='fom_cg_rtol0.01')
        assert cfg['cohort'] == src['cohort']['role']
    else:
        src_cfg = json.loads((HERE / 'deps' / 'heat_configs' / ('h3d256.json' if n == 256 else 'h3d.json')).read_text())
        cg = {k: v for k, v in src_cfg['cg_arms'].items()}
        cfg = dict(common, model=src_cfg['model'],
                   model_sha256={'bank': sha(HERE / 'deps/heat/inputs/vp_R320/bank.pkl'),
                                 'head': sha(HERE / 'deps/heat/inputs/vp_R320/head_K32.pkl')},
                   times=src_cfg['times'], diffusivity=src_cfg['diffusivity'], heldout_seed=921099, heldout_count=64,
                   edges=src_cfg['edges'], tests=src_cfg['tests'], accurate_Rp=320, fast_Rp=128,
                   rom_dt=src_cfg['rom_defaults']['dt'], cg_arms=cg, table1_fom='fom_cncg_dt0.025_rtol1e-4')
    out = HERE / 'configs' / (f'panel_{problem}{n}' + ('_smoke' if smoke else '') + '.json')
    out.write_text(json.dumps(cfg, indent=1) + '\n')
    print(out, [(o['arm'], o.get('failure', 'ok')) for o in ops])


if __name__ == '__main__':
    main()
