"""spectral-fom, Heat 2D/3D: modal (DST) Crank-Nicolson and exact propagation vs the heat-bank-knob settings.

ROM arms: the heat lane's selected arms, built exactly as `experiments/heat-bank-knob/hbk_run.py` builds them (nested
rotated bank K.build_bank without the parent, K.nested_tsqr_r, K.nested_weak_matrix, K.truncated_model,
K.make_stages / K.make_linear with the lane's family options), staged from the lane's pinned commit.
Query scope = the lane's: supplied interior initial field on the GPU -> all six output fields (here the harness also
does the synchronised host->device copy first, outside the timed region).
Truth / error (the lane's): exact modal propagation core.make_propagate with the discrete eigenvalues; error per
output time = relative L2 (core.rel_errors); an arm's error = max over cases x the six times.
Spectral subjects: modal_exp_{fft,mm}, modal_cn_{fft,mm} (dt = the paper's CN dt).
Validation (untimed): modal_exp vs the lane's truth on every timed case (<= 1e-10); modal_cn vs the paper's CN-CG at
rtol 1e-12 on case 0 (<= 1e-9); CONTROL modal_exp with continuum eigenvalues must exceed 1e-10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import hbk_core as K
import spec_core as SC
import spec_timing as ST


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    lane = json.loads(Path(cfg['lane_config']).read_text())
    lane_root = Path(cfg['lane_root'])
    if a.smoke:
        cfg.update(repetitions=2, burn_seconds=0.001, cooldown_seconds=0.1, phase_dummy_seconds=0.1, case_count=2)
    n = int(cfg['intervals'])
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    uuid0 = ST.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory
    except FileNotFoundError:
        inventory = None
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)

    model = C.load_model(lane['model'], lane_root / 'inputs')
    d = model['d']
    times, nu = lane['times'], lane['diffusivity']
    prep = np.load(lane_root / lane['prep'])
    T, L = prep['T'], prep['L']
    rot_info = json.loads(str(prep['rotation_info']))
    assert K.sha_array(T) == rot_info['T_sha256'] and K.sha_array(L) == rot_info['L_sha256']
    cname, seed, count = next(c for c in lane['cohorts'] if c[0] == cfg['cohort'])
    draws = C.family(model['family'], seed, count)[:cfg.get('case_count', count)]
    tj = json.loads((lane_root / 'inputs' / Path(lane['model']['bank']).parent / 'training.json').read_text())
    train = C.family(model['family'], tj['config']['train_seed'], tj['config']['train_count'])
    assert not any(np.array_equal(x_, y_) for x_ in train for y_ in draws)
    R_ = dict(problem=f'heat{d}d', config=cfg, lane_config=lane, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
              gpu_uuid=uuid0, nvidia_smi=inventory, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), intervals=n, dim=d, model_sha256=model['sha256'],
              cohort=dict(name=cname, seed=seed, count=int(count), used=int(len(draws)), sha256=K.sha_array(draws), draws=np.asarray(draws).tolist()),
              error_convention='relative L2 per output time vs exact modal propagation (discrete eigenvalues); max over cases x the six times',
              complete=False)
    save = lambda: C.dump(out / 'result.json', R_)

    # ---- bank + operators (hbk_run.py, parent-free)
    edges = list(lane['edges'])
    nested, _ = K.build_bank(model, n, T, edges, False)
    modes = C.mode_list(lane['tests'], n, d)
    lam_m = C.mode_eigs(n, modes)
    rtri = K.nested_tsqr_r(nested)
    a_rot = K.nested_weak_matrix(nested, modes, n, d)
    fam_opts = {f: {**lane['rom_defaults'], **o} for f, o in lane['families'].items()}
    rom = []
    for role, name in cfg['rom_arms'].items():
        p_ = name.split('_')
        if name.startswith('nmrom_'):
            Rp, q, fam = int(p_[1][1:]), int(p_[2][1:]), p_[3]
        else:
            assert name.startswith('lin_')
            Rp, q, fam = int(p_[1][1:]), None, p_[2]
        nb = edges.index(Rp)
        proj = (lambda b, v, nb=nb: K.nproject(b, v, nb))
        expd = (lambda c, b, nb=nb: K.nexpand(c, b, nb))
        setup = dict(n=n, d=d, times=times, nu=nu, modes=modes, a=a_rot[:, :Rp], rtri=rtri[:Rp, :Rp], mode_lam=lam_m,
                     directions=L[:Rp] @ np.asarray(model['directions']))
        opt = fam_opts[fam]
        if q is not None:
            tm = K.truncated_model(model, L, Rp)
            st = K.make_stages(tm, setup, q, opt, proj, expd)
        else:
            st = K.make_linear(setup, opt['init'], opt['stepping'], opt.get('dt'), proj, expd)
        fn = (lambda u, st=st: st['query'](u, nested))
        rom.append(dict(name=name, role=role, fn=fn))
    lam_d = C.eig_grid(n, d)
    prop = C.make_propagate(d)
    tjx = jnp.asarray(times)
    u0s = [np.asarray(C.initial_grid(n, d, dr)) for dr in draws]
    truth = [np.asarray(prop(jnp.asarray(u), lam_d, tjx, nu)) for u in u0s]
    for r_ in rom:
        r_['call'] = (lambda c, fn=r_['fn']: ST.query(fn, u0s[c]))
    spec = []
    for variant in cfg['spectral_variants']:
        for scheme in ('exp', 'cn'):
            fn = SC.make_heat(n, d, nu, times, variant, scheme, dt=cfg['cn_dt'])
            spec.append(dict(name=f'modal_{scheme}_{variant}', role='spectral', scheme=scheme,
                             call=(lambda c, fn=fn: ST.query(fn, u0s[c]))))
    print('SETUP', el(), flush=True)
    save()

    def errs(f, c):
        return np.asarray(C.rel_errors(jnp.asarray(f), jnp.asarray(truth[c]))).tolist()

    # ---- V: validation
    V = dict(exp_vs_truth=[], cn_vs_cncg=None)
    for s_ in spec:
        if s_['scheme'] == 'exp':
            for c in range(len(draws)):
                V['exp_vs_truth'].append(dict(name=s_['name'], case=c, worst=max(errs(s_['call'](c)[0], c))))
    cg = C.make_cg(n, times, dict(dt=cfg['cn_dt'], tolerance=cfg['cncg_validation_rtol'], max_iterations=20000), nu)
    fcg, info = jax.device_get(cg(jnp.asarray(u0s[0])))
    fcn = next(s_ for s_ in spec if s_['scheme'] == 'cn')['call'](0)[0]
    info = np.asarray(info).reshape(-1, 3)
    V['cn_vs_cncg'] = dict(rtol=cfg['cncg_validation_rtol'], cg_failures=int((info[:, 2] != 1).sum()),
                           worst=float(max(np.asarray(C.rel_errors(jnp.asarray(fcn), jnp.asarray(fcg))))))
    lam_c = C.eig_grid(n, d, True)
    fc = np.asarray(prop(jnp.asarray(u0s[0]), lam_c, tjx, nu))
    V['control'] = dict(name='modal_exp_continuum_eigenvalues_CONTROL', worst=max(errs(fc, 0)), must_exceed=cfg['spectral_limit'])
    V['control']['failed_as_required'] = bool(V['control']['worst'] > cfg['spectral_limit'])
    V['gates'] = dict(exp_exact=bool(max(r['worst'] for r in V['exp_vs_truth']) <= cfg['spectral_limit']),
                      cn_matches_cncg=bool(V['cn_vs_cncg']['cg_failures'] == 0 and V['cn_vs_cncg']['worst'] <= cfg['cn_agreement_limit']),
                      control_failed=V['control']['failed_as_required'])
    R_['validation'] = V
    print('VALIDATION', V['gates'], V['cn_vs_cncg'], V['control']['worst'], el(), flush=True)
    save()

    # ---- T: A-B-A timing
    saved = {}
    stride = max(1, n // 256)
    subsl = (slice(None),) + (slice(stride - 1, None, stride),) * d

    def record(sub_, c, rep_, phase, field, row, raw):
        assert np.isfinite(field).all(), sub_['name']
        h = K.sha_array(field)
        key = (sub_['name'], c)
        if key not in saved:
            np.save(out / 'fields' / f"{sub_['name']}_sub_case{c}.npy", field[subsl])
            if c < 2:
                np.save(out / 'fields' / f"{sub_['name']}_case{c}.npy", field)
            saved[key] = h
        e_ = errs(field, c)
        return dict(field_sha256=h, matches_saved_field=bool(h == saved[key]), same_grid_per_time=e_,
                    same_grid_worst=float(max(e_)), sub_errors=np.asarray(C.rel_errors(jnp.asarray(field[subsl]), jnp.asarray(truth[c][subsl]))).tolist())
    for c in range(len(draws)):
        np.save(out / 'fields' / f'truth_sub_case{c}.npy', truth[c][subsl])
        if c < 2:
            np.save(out / 'fields' / f'truth_case{c}.npy', truth[c])
    for s_ in rom + spec:
        for _ in range(2):
            s_['call'](0)
    harness = ST.ABA(cfg, uuid0, record, log=lambda s_: print(s_, el(), flush=True), save=save)
    R_['invocations'] = harness.invocations
    gates = harness.run(rom, spec, list(range(len(draws))), cfg['repetitions'], cfg['order_seed'])
    R_['phase_breaks'] = harness.breaks
    R_['timing_gates'] = gates
    R_['timings'] = harness.timings(rom, spec)
    R_['errors'] = {}
    for s_ in rom + spec:
        ev = [x_['same_grid_worst'] for x_ in harness.invocations if x_['name'] == s_['name'] and x_['rep'] == 0
              and x_['phase'] in ('romA1', 'spec')]
        R_['errors'][s_['name']] = dict(worst=float(max(ev)), median=float(np.median(ev)), per_case=ev)
    R_['gates'] = dict(**V['gates'], deterministic=all(x_['matches_saved_field'] for x_ in harness.invocations),
                       drift=gates['drift']['passed'], neighbour=gates['neighbour']['passed'],
                       device_guard=ST.gpu_uuid() == uuid0)
    print('GATES', R_['gates'], flush=True)
    print('TIMINGS', {k: round(v['median_ms'], 3) for k, v in R_['timings'].items()}, flush=True)
    print('ERRORS%', {k: round(100 * v['worst'], 5) for k, v in R_['errors'].items()}, flush=True)
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(json.dumps(R_['gates']) + '\n')


if __name__ == '__main__':
    main()
