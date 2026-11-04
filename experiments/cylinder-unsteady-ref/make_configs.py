"""Generate every G5 reference job configuration from one place (DESIGN-G5.md §4).

Usage: make_configs.py smoke            -> configs/smoke/*.json
       make_configs.py <stage> ...      (later stages are added by dated amendments)
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MESH_SHA = '528ffff38bb0f4d6d542f1abd4cba7c9ca2ba0427871caca806527c63ae6b092'

BASE = {
    'H': 0.41, 'xmin': 0.0, 'xmax': 2.2, 'center': [0.2, 0.2], 'radius': 0.05,
    'U_mean': 1.0, 'U_mean0': 0.2, 'ramp_time': 1.0, 'nu': 0.001,
    'velocity_order': 3, 'pressure_order': 2, 'geometry_order': 4,
    'mesh_path': 'inputs/admitted_mesh.vol.bin', 'mesh_sha256': MESH_SHA,
    'assembly_bonus_order': 6, 'diagnostic_order': 12, 'threads': 16,
    'inverse_backend': 'umfpack', 'max_dofs': 500000,
    'newton_max': 30, 'nonlinear_relative_tolerance': 1e-10,
    'damping_sequence': [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
    'linear_relative_tolerance': 1e-10, 'gmres_maxsteps': 60, 'refactor_every': 10,
    'scheme': 'sbdf2', 'startup_substeps': 4, 'snapshot_chunk': 50, 'energy_blowup': 1e3,
    'periodicity': {'n': 3, 'cl_tol': 1e-4, 'cd_rel_tol': 1e-4, 'period_rel_tol': 1e-4},
    'kick': {'amplitude': 1.0, 't_start': 1.0, 'duration': 0.5, 'center': [0.3, 0.2], 'sigma': 0.02},
    'initial': {'kind': 'steady_newton'},
    'wall_limit_seconds': 3600,
    'precision': 'float64',
    # FD1-admitted Re 20 P4/P3 reference (obstacle-2d artifacts/g2-population-re20-02/retained/
    # p4_g4_h3_baseline/result.json, config sha 2fbe6cd9...; reaction-force drag not stored there,
    # gradient-stress order-16 surface coefficient used): R1a gate, 2e-3 relative.
    'steady_reference': {'CD': 5.579548183731313, 'CL': 0.010615714195317974,
                         'source': 'FD1 Re20 P4/P3 gradient_order16 coefficient'},
}


def cfg(**kw):
    c = json.loads(json.dumps(BASE))
    for k, v in kw.items():
        c[k] = v
    return c


def smoke():
    # Smoke only: compressed ramp/kick so that full inflow speed is reached within the
    # 0.2 s windows of a and b (SBDF2 stability at production speed); NOT a production
    # setting. c (Oseen) and d (P4/P3) stop at t=0.025 (U_mean=0.6, kick off): they are
    # import/assembly/factorization and per-step cost checks only, not stability results.
    fast = {'ramp_time': 0.05, 'kick': dict(BASE['kick'], t_start=0.06, duration=0.04)}
    st = lambda dt, dur: [{'name': 'smoke', 'dt': dt, 'duration': dur, 'save_every_steps': int(round(1 / 400 / dt)),
                           'diagnostics_every_saves': 10}]
    return {
        'a_sbdf2_p3_dt1600': cfg(**fast, stages=st(1 / 1600, 0.2), wall_limit_seconds=360),
        'b_sbdf2_p3_dt800': cfg(**fast, stages=st(1 / 800, 0.2), wall_limit_seconds=240),
        'c_oseen_p3_dt1600': cfg(**fast, scheme='oseen_bdf2', stages=st(1 / 1600, 0.025), wall_limit_seconds=240),
        'd_sbdf2_p4_dt1600': cfg(**fast, velocity_order=4, pressure_order=3, stages=st(1 / 1600, 0.025),
                                 wall_limit_seconds=240),
    }


def smoke02():
    # Amendment G5-A1 smoke: frozen-Oseen stability at full speed (e, f), fixed Oseen cost (g),
    # assembly bonus order 2 cost and Re-20 drag effect (h). Same compressed smoke ramp.
    fast = {'ramp_time': 0.05, 'kick': dict(BASE['kick'], t_start=0.06, duration=0.04)}
    st = lambda dt, dur: [{'name': 'smoke', 'dt': dt, 'duration': dur, 'save_every_steps': int(round(1 / 400 / dt)),
                           'diagnostics_every_saves': 10}]
    fo = {'scheme': 'frozen_oseen', 'refactor_every': 10}
    return {
        'e_frozen_p3_dt1600': cfg(**fast, **fo, stages=st(1 / 1600, 0.2), wall_limit_seconds=420),
        'f_frozen_p3_dt800': cfg(**fast, **fo, stages=st(1 / 800, 0.2), wall_limit_seconds=300),
        'g_oseen_p3_dt1600': cfg(**fast, scheme='oseen_bdf2', stages=st(1 / 1600, 0.03), wall_limit_seconds=300),
        'h_frozen_p3_dt1600_bonus2': cfg(**fast, scheme='frozen_oseen', refactor_every=25, assembly_bonus_order=2,
                                         stages=st(1 / 1600, 0.05), wall_limit_seconds=240),
    }


def spinup():
    # DESIGN-G5 4.2 + G5-A2: Re 100 from the Re 20 Newton state, sin^2 ramp t_r=1, kick on
    # [1,1.5], S3 at dt_spin=1/800 (K=10) until the developed-flow rule, cap t=15. End state = S*.
    return {'spinup_re100': cfg(scheme='frozen_oseen', refactor_every=10, wall_limit_seconds=16900,
                                stages=[{'name': 'spinup', 'dt': 1 / 800, 'duration': 15.0, 'until_periodic': True,
                                         'require_periodic': True, 'save_every_steps': 80,
                                         'diagnostics_every_saves': 1, 'save_state': True}])}


K_FOR_DT = {800: 10, 1600: 20, 3200: 40, 6400: 80}   # G5-A2: frozen lag K*dt = 1/80 s


def study(state, state_sha, dts=(800, 1600, 3200, 6400), startup16=(1600, 3200), p4=()):
    # DESIGN-G5 4.4 R1b/R1c: from S*, window W = 0.35 s, saves every 1/400 s.
    init = {'kind': 'state', 'path': state, 'sha256': state_sha}
    def one(n, **kw):
        return cfg(scheme='frozen_oseen', refactor_every=K_FOR_DT[n], kick=None, initial=init, wall_limit_seconds=5400,
                   stages=[{'name': 'study', 'dt': 1 / n, 'duration': 0.35, 'save_every_steps': int(round(n / 400)),
                            'diagnostics_every_saves': 4}], **kw)
    out = {f'time_dt{n}': one(n) for n in dts}
    out.update({f'startup16_dt{n}': one(n, startup_substeps=16) for n in startup16})
    out.update({f'space_p4_dt{n}': one(n, velocity_order=4, pressure_order=3) for n in p4})
    return out


def settle(Re, state, state_sha, k=4):
    # G5-A3: discarded settle at dt 1/800 (K=10) until the developed-flow rule, cap 4 s.
    return {f'settle_re{Re:g}_p{k}': cfg(nu=0.1 / Re, velocity_order=k, pressure_order=k - 1, scheme='frozen_oseen',
                                       refactor_every=10, kick=None, wall_limit_seconds=10200,
                                       initial={'kind': 'state', 'path': state, 'sha256': state_sha},
                                       stages=[{'name': 'settle', 'dt': 1 / 800, 'duration': 4.0, 'until_periodic': True,
                                                'require_periodic': True, 'save_every_steps': 80,
                                                'diagnostics_every_saves': 1, 'save_state': True}])}


def study_p4(state, state_sha, dts=(800, 1600, 3200, 6400), startup16=(1600, 3200), p3=()):
    # G5-A3: R1b on the production P4/P3 from S* (embedded); R1c P3/P2 runs at dt0 later.
    init = {'kind': 'state', 'path': state, 'sha256': state_sha}
    def one(n, k=4, **kw):
        return cfg(scheme='frozen_oseen', refactor_every=K_FOR_DT[n], kick=None, initial=init, wall_limit_seconds=14000,
                   velocity_order=k, pressure_order=k - 1,
                   stages=[{'name': 'study', 'dt': 1 / n, 'duration': 0.35, 'save_every_steps': int(round(n / 400)),
                            'diagnostics_every_saves': 4}], **kw)
    out = {f'time_p4_dt{n}': one(n) for n in dts}
    out.update({f'startup16_p4_dt{n}': one(n, startup_substeps=16) for n in startup16})
    out.update({f'space_p3_dt{n}': one(n, k=3) for n in p3})
    return out


def record(Re, state, state_sha, dt0, k=4):
    # DESIGN-G5 4.2: 0.7 s settle at dt0, then the 1.7 s recorded window saved every 1/400 s.
    n = dt0
    return {f'record_re{Re:g}_p{k}': cfg(nu=0.1 / Re, velocity_order=k, pressure_order=k - 1, scheme='frozen_oseen',
                                       refactor_every=K_FOR_DT[n], kick=None, wall_limit_seconds=40000,
                                       initial={'kind': 'state', 'path': state, 'sha256': state_sha},
                                       stages=[{'name': 'settle_dt0', 'dt': 1 / n, 'duration': 0.7},
                                               {'name': 'record', 'dt': 1 / n, 'duration': 1.7,
                                                'save_every_steps': int(round(n / 400)), 'diagnostics_every_saves': 4,
                                                'save_state': True}])}


def main():
    stage = sys.argv[1]
    if stage == 'study_p4':
        sets = study_p4(sys.argv[2], sys.argv[3])
    elif stage == 'space_p3':
        sets = study_p4(sys.argv[2], sys.argv[3], dts=(), startup16=(), p3=(int(sys.argv[4]),))
    elif stage == 'record':
        sets = record(float(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif stage == 'settle':
        sets = settle(float(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif stage == 'study':
        sets = study(sys.argv[2], sys.argv[3])
    else:
        sets = {'smoke': smoke, 'smoke02': smoke02, 'spinup': spinup}[stage]()
    d = HERE / 'configs' / stage
    d.mkdir(parents=True, exist_ok=True)
    for name, c in sets.items():
        (d / f'{name}.json').write_text(json.dumps(c, indent=2) + '\n')
        print(d / f'{name}.json')


if __name__ == '__main__':
    main()
