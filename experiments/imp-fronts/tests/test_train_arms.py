"""imp-fronts Phase B retrain pipeline: CPU checks of train2d/ and its two arms (DESIGN B1, B2, B9; Amendments 1-4).

    JAX_PLATFORMS=cpu bash .../campaign/jaxslot.sh 3600 /home/tahmid/Dev/.venv/bin/python \
        experiments/imp-fronts/tests/test_train_arms.py [--dry] [--json <path>] [--keep]

Every JAX part runs in its own subprocess (the arms' generators share the module name `burgers2d_film`).
  [S1] train2d/ is byte-identical to jcp-robust train2d at b94591b81 (git blobs) and passes its own PROVENANCE.
  [S2] stage_arm.sh base -> tree byte-identical to jcp-robust's FILES.txt entries (same file set).
  [S3] stage_arm.sh fid / cov -> differ from base exactly in the ARM.json-declared paths (+ PROVENANCE.json); the
       staged PROVENANCE passes run_seed.sh's guard (run_seed.sh itself, TRAIN2D_DRYRUN=1, STAGES=none).
  [S4] C-B1 analogue: a deliberately mismatched sha256 in PROVENANCE.json, and an edited generator byte with an
       unchanged PROVENANCE.json, are both rejected by run_seed.sh's guard (exit 1, "MISMATCH").
  [C-B4] cov: complete parameter tables (seed 0 x 576, seed 1000 x 4032): max s >= 900, all s <= 1000,
       min nu >= 0.0016, cx/cy/w/a bitwise = incumbent; incumbent tables FAIL max s >= 900 (evaluator run on the real
       incumbent sample_params); the gate's incumbent replica equals the incumbent module; U mapping identity.
  [cov-newton] cov generator at 256 nodes on the 2 most extreme real trajectories (max s; min nu): finite, Newton
       relative residual <= 1e-8 (rollout's own and the independent residual check of Stage A/B).
  [fid-contract] fid generator: rollout contract (shapes, state 0 = blob_ic bitwise, zero walls, rel_res = 0,
       finiteness sentinel fires on NaN, unregistered IC rejected), capture-every-5 = lattice path, and identity
       with jcp-fom2d psjax2d.make_ps_query (N=128, dt=0.001) <= 1e-12.
  [C-B3] fid gate on all dev6 cases: worst <= 1e-4 vs E0 on the 257^2 lattice; PS self-convergence N128/dt1e-3 vs
       N192/dt5e-4 <= 1e-4; incumbent upwind generator (runs/a1 upw_dev6_*_n257.npz, generator sha256 verified) FAILS
       the same bar; the gate fails closed on a non-finite state.
  [dry] (--dry) TRAIN2D_DRYRUN=1 chaining run (N=64) of the staged fid and cov trees: Stage A -> B -> rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]                     # experiments/imp-fronts
T2 = LANE / 'train2d'
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
ROBUST = ROOT / 'worktrees/2026-10-19-jcp-robust'
ROBUST_COMMIT = 'b94591b8121e52afe2acbb5b4f7f3c560963ede4'
FOM2D = ROOT / 'worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d'
PY = '/home/tahmid/Dev/.venv/bin/python'
SCRATCH = Path('/tmp/claude-1002/imp-fronts-train-arms-scratch')
INC_GEN = 'c521b36a9bae5471f0f3c1644d03451109c84688cf3808f9c5ea6ffea18493e4'
GB = 'stageB/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom'
RESULTS = {}
fails = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        fails.append(msg)
    return cond


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(d):
    d = Path(d)
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def stage(arm):
    job = SCRATCH / f'stage_{arm}'
    shutil.rmtree(job, ignore_errors=True)
    r = subprocess.run(['bash', str(T2 / 'stage_arm.sh'), arm, str(job)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return job, job / 'experiments/imp-fronts/train2d'


def guard(t, out, extra_env=None):
    env = dict(os.environ, PY=PY, TRAIN2D_DRYRUN='1', STAGES='none', PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu')
    env.update(extra_env or {})
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run(['bash', str(t / 'run_seed.sh'), '1', str(out)], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------------------------------------------- static parts
def static():
    print('[S1] train2d copy = jcp-robust train2d @ b94591b81 (git blobs)')
    names = subprocess.check_output(['git', '-C', str(ROBUST), 'ls-tree', '-r', '--name-only', ROBUST_COMMIT,
                                     'experiments/jcp-robust/train2d/'], text=True).split()
    pre = 'experiments/jcp-robust/train2d/'
    same = True
    for n in names:
        blob = subprocess.check_output(['git', '-C', str(ROBUST), 'show', f'{ROBUST_COMMIT}:{n}'])
        same &= (T2 / n[len(pre):]).read_bytes() == blob
    lane = {p for p in tree(T2) if not p.startswith('arms/') and p != 'stage_arm.sh'}
    check(same and lane == {n[len(pre):] for n in names},
          f'{len(names)} files byte-identical to the committed jcp-robust blobs, no other files outside arms/')
    prov = json.loads((T2 / 'PROVENANCE.json').read_text())
    check(all(sha(T2 / f['path']) == f['sha256'] for f in prov['files']), 'lane copy passes its PROVENANCE.json')

    print('[S2] staged base tree = jcp-robust FILES.txt entries')
    jb, tb = stage('base')
    fl = [l.strip() for l in (ROBUST / 'experiments/jcp-robust/train2d/FILES.txt').read_text().splitlines()
          if l.strip() and not l.startswith('#')]
    want = {e[len(pre):]: sha(ROBUST / e) for e in fl}
    got = tree(tb)
    check(got == want, f'base: {len(got)} staged files, set and bytes equal to jcp-robust ({len(want)} entries)')
    RESULTS['base_tree'] = dict(n=len(got), identical=got == want)

    print('[S3] staged arm trees differ only in their declared files; staged PROVENANCE passes run_seed.sh guard')
    for arm in ('fid', 'cov', 'bnd'):
        ja, ta = stage(arm)
        ga = tree(ta)
        spec = json.loads((T2 / 'arms' / arm / 'ARM.json').read_text())
        declared = {f['dst'] for f in spec['files']} | {'PROVENANCE.json'}
        diff = {p for p in set(ga) | set(want) if ga.get(p) != want.get(p)}
        check(diff == declared, f'{arm}: changed paths = declared ({len(declared)}): {sorted(diff)}')
        if arm == 'bnd':
            check(ga['stageA/deps/burgers2d_film.py'] == ga[f'{GB}/burgers2d_film.py'] == INC_GEN
                  and ga['stageB/code/sep_common.py'] == ga['rotation/sep_common.py'] != want['stageB/code/sep_common.py']
                  and ga['stageA/deps/sep_common.py'] != want['stageA/deps/sep_common.py'],
                  'bnd: generator = incumbent in both places; sep_common replaced in Stage A, Stage B and rotation')
        else:
            check(ga['stageA/deps/burgers2d_film.py'] == ga[f'{GB}/burgers2d_film.py'] != INC_GEN,
                  f'{arm}: generator replaced in BOTH places (Stage A deps, Stage B burgers2d-coord-rom), identical copies')
        p = json.loads((ta / 'PROVENANCE.json').read_text())
        ed = sorted(f['path'] for f in p['files'] if f.get('edited'))
        check(all(sha(ta / f['path']) == f['sha256'] for f in p['files']) and set(f['path'] for f in p['files'])
              >= declared - {'PROVENANCE.json'}, f'{arm}: PROVENANCE covers every declared path; edited=true: {ed}')
        rc, log = guard(ta, SCRATCH / f'guard_{arm}')
        check(rc == 0 and 'PROVENANCE sha256 OK' in log, f'{arm}: run_seed.sh guard accepts the staged tree (rc={rc})')
        RESULTS[f'{arm}_tree'] = dict(changed=sorted(diff), edited_true=ed, guard_rc=rc)
    rc, log = guard(tb, SCRATCH / 'guard_base')
    check(rc == 0 and 'PROVENANCE sha256 OK' in log, f'base: run_seed.sh guard accepts the staged tree (rc={rc})')

    print('[S4] C-B1 analogue: run_seed.sh guard rejects mismatches')
    for arm, how in (('fid', 'provenance'), ('fid', 'file'), ('cov', 'provenance'), ('base', 'provenance')):
        job, t = stage(arm)
        if how == 'provenance':
            p = json.loads((t / 'PROVENANCE.json').read_text())
            e = next(f for f in p['files'] if f['path'] == 'stageA/deps/burgers2d_film.py')
            e['sha256'] = e['sha256'][:-1] + ('0' if e['sha256'][-1] != '0' else '1')
            (t / 'PROVENANCE.json').write_text(json.dumps(p, indent=1) + '\n')
        else:
            g = t / GB / 'burgers2d_film.py'
            g.write_bytes(g.read_bytes() + b'\n')
        rc, log = guard(t, SCRATCH / f'guardbad_{arm}_{how}')
        check(rc == 1 and 'MISMATCH' in log, f'{arm}/{how}: rejected (rc={rc}) '
              + (log.strip().splitlines()[-1] if log.strip() else ''))
        RESULTS[f'C-B1_{arm}_{how}'] = dict(rc=rc, rejected=rc == 1 and 'MISMATCH' in log)


# ------------------------------------------------------------------------------------------------- JAX parts
def part(name, deps, timeout=3000):
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1', JAX_DEFAULT_MATMUL_PRECISION='highest')
    for v in ('N', 'SEED', 'STEPS', 'N_TRAIN', 'N_VAL', 'NEWTON_ITERS', 'LIN_TOL', 'LIN_MAXITER'):
        env.pop(v, None)
    t = time.time()
    r = subprocess.run([PY, '-u', __file__, '--part', name, '--deps', str(deps)], capture_output=True, text=True,
                       env=env, timeout=timeout)
    out = [l for l in r.stdout.splitlines() if not l.startswith('RESULT ')]
    print('\n'.join(out), flush=True)
    res = [json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith('RESULT ')]
    if r.returncode != 0 or not res:
        print(r.stderr[-3000:])
        check(False, f'part {name} crashed (rc={r.returncode})')
        return None
    for c, msg in res[0]['checks']:
        check(c, msg)
    RESULTS[name] = dict(res[0]['data'], seconds=time.time() - t)
    return res[0]


def run_part(name, deps):
    sys.path.insert(0, deps)
    import numpy as np
    checks, data = [], {}
    ck = lambda c, m: checks.append((bool(c), m))
    if name == 'cb4':
        import importlib.util
        import burgers2d_film as bf
        import arm_gate as G
        spec = importlib.util.spec_from_file_location('bf_inc', T2 / 'stageA/deps/burgers2d_film.py')
        inc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inc)
        r = G.c_b4(bf.sample_params)
        for k, v in r['populations'].items():
            print(f"  {k}: arm max s {v['arm']['max_s']:.2f} min s {v['arm']['min_s']:.2f} min nu {v['arm']['min_nu']:.6f}"
                  f" max nu {v['arm']['max_nu']:.4f} n(s>=900) {v['arm']['n_s_ge_900']} | incumbent max s "
                  f"{v['incumbent']['max_s']:.2f} min nu {v['incumbent']['min_nu']:.5f}  checks {v['checks']}")
            for c, ok in v['checks'].items():
                ck(ok, f'C-B4 {k} ({v["m"]}): {c}')
        ri = G.c_b4(inc.sample_params)
        ck(ri['passed'] is False and all(not v['checks']['max_s_ge_900'] for v in ri['populations'].values()),
           'C-B4 must-fail: the evaluator FAILS the real incumbent tables (max s >= 900 false for both populations)')
        for nm, seed, m in G.POPULATIONS:
            a = inc.sample_params(seed=seed, m=m)[:5]
            b = G.incumbent_table(seed, m)
            ck(all(np.array_equal(p, q) for p, q in zip(a, b)), f'gate incumbent replica = incumbent module ({nm})')
            rng = np.random.default_rng(seed)
            for _ in range(4):
                rng.uniform(0, 1, m)
            U = rng.uniform(0.0, 1.0, m)
            ck(np.array_equal(np.exp(np.log(0.01) + (np.log(0.1) - np.log(0.01)) * U), a[4]),
               f'U identity ({nm}): incumbent nu = exp(log .01 + (log .1 - log .01) U) bitwise, U = the arm\'s draw')
            arm = bf.sample_params(seed=seed, m=m)
            lo = np.maximum(0.0016, arm[3] / 1000)
            ck(np.allclose(np.log(arm[4] / lo) / np.log(0.1 / lo), U, rtol=0, atol=1e-12),
               f'arm nu uses the same U ({nm})')
            zi, za = inc.sample_params(seed=seed, m=m)[5], arm[5]
            ck(np.array_equal(zi[:, :4], za[:, :4]) and np.allclose(
                za[:, 4], ((np.log(arm[4]) - np.log(np.sqrt(0.001))) / (0.5 * np.log(10.0))).astype(np.float32)),
               f'z: first four columns identical, log-nu column by the unchanged formula ({nm})')
        data = dict(cov=r['populations'], incumbent_eval_passed=ri['passed'])
    elif name == 'covnewton':
        import jax.numpy as jnp
        import jax
        import burgers2d_film as bf
        cands = []
        for seed, m in ((0, 576), (1000, 4032)):
            cx, cy, w, a, nu, _ = bf.sample_params(seed=seed, m=m)
            s = a / nu
            cands += [(float(s[i]), float(nu[i]), seed, int(i), cx[i], cy[i], w[i], a[i]) for i in range(m)]
        pick = [max(cands, key=lambda c: c[0]), min(cands, key=lambda c: c[1])]
        ro, res_fn = bf.make_rollout(256)
        chk = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res_fn(u1, u0, nu_)) / jnp.linalg.norm(u0)))
        U0 = np.stack([bf.blob_ic(256, c[4], c[5], c[6], c[7]) for c in pick])
        nuj = jnp.asarray([c[1] for c in pick])
        t = time.time()
        snaps, res = ro(jnp.asarray(U0), nuj)
        sec = time.time() - t
        worst = max(float(jnp.max(chk(snaps[k + 1], snaps[k], nuj))) for k in range(bf.NUM_STEPS))
        fin = bool(jnp.all(jnp.isfinite(snaps)))
        print(f'  trajectories (s, nu, seed, index): {[c[:4] for c in pick]}; rollout {sec:.0f}s CPU; '
              f'max rollout rel_res {float(jnp.max(res)):.3e}; independent residual {worst:.3e}; finite {fin}')
        ck(fin and float(jnp.max(res)) <= 1e-8 and worst <= 1e-8,
           'cov: extreme real trajectories finite, Newton rel residual <= 1e-8 (rollout and independent check)')
        data = dict(trajectories=[c[:4] for c in pick], rollout_seconds_cpu=sec, max_rel_res=float(jnp.max(res)),
                    independent_residual=worst, finite=fin)
    elif name == 'fidcontract':
        import jax.numpy as jnp
        sys.path.insert(1, str(LANE))
        import burgers2d_film as bf
        import fcommon as F
        sys.path.append(str(FOM2D))
        import psjax2d
        descs = [F.load_ref('dev6', c)[1]['desc'] for c in (0, 1)]
        ro, res_fn = bf.make_rollout(256)
        U0 = np.stack([bf.blob_ic(256, d['cx'], d['cy'], d['w'], d['a']) for d in descs])
        t = time.time()
        snaps, res = ro(jnp.asarray(U0), jnp.asarray([d['nu'] for d in descs]))
        sec = time.time() - t
        S = np.asarray(snaps)
        walls = S[1:].reshape(51 - 1, 2, 256, 256)
        wall_max = max(np.abs(walls[..., 0, :]).max(), np.abs(walls[..., -1, :]).max(), np.abs(walls[..., :, 0]).max(),
                       np.abs(walls[..., :, -1]).max())
        ck(S.shape == (51, 2, 65536) and np.asarray(res).shape == (50, 2), f'shapes snaps {S.shape} rel_res {np.asarray(res).shape}')
        ck(np.array_equal(S[0], U0), 'state 0 = incumbent nodal blob_ic(256), bitwise')
        ck(wall_max == 0.0 and np.isfinite(S).all() and float(np.max(np.abs(np.asarray(res)))) == 0.0,
           f'walls exactly 0 ({wall_max}), all finite, rel_res = 0')
        bad = np.array(S[5, 0]); bad[1000] = np.nan
        z_ok = float(jnp.linalg.norm(res_fn(jnp.asarray(S[6, 0]), jnp.asarray(S[5, 0]), 0.03)))
        z_bad = float(jnp.linalg.norm(res_fn(jnp.asarray(S[6, 0]), jnp.asarray(bad), 0.03)))
        ck(z_ok == 0.0 and not np.isfinite(z_bad), f'residual sentinel: finite -> {z_ok}, NaN state -> {z_bad}')
        try:
            ro(jnp.asarray(U0 * (1 + 1e-15)), jnp.asarray([0.03, 0.03]))
            ck(False, 'unregistered IC rejected')
        except ValueError:
            ck(True, 'unregistered IC (not from blob_ic) rejected with ValueError')
        d = descs[0]
        desc = (d['cx'], d['cy'], d['w'], d['a'], d['nu'])
        ro257, _ = bf.make_rollout(257)
        s257 = np.asarray(ro257(jnp.asarray(bf.blob_ic(257, *desc[:4])[None]), jnp.asarray([desc[4]]))[0])[:, 0]
        lat = bf.ps_fields([desc], np.arange(257) / 256., every=50, n_out=5)[0]
        d_cap = float(np.abs(s257[10::10].reshape(5, 257, 257) - lat[1:]).max() / np.abs(lat[1:]).max())
        q = psjax2d.make_ps_query(128, 0.001)
        ref, _ = q(psjax2d.ic(desc, 128), desc[4])
        d_ps = float(np.abs(np.asarray(ref) - lat).max() / np.abs(np.asarray(ref)).max())
        print(f'  2-trajectory rollout at 256 nodes {sec:.1f}s CPU; capture-every-5 vs lattice path {d_cap:.2e}; '
              f'vs psjax2d.make_ps_query(128, 0.001) {d_ps:.2e}')
        ck(d_cap <= 1e-12, f'rollout(257) states at t=0.05k = lattice path (rel max {d_cap:.2e} <= 1e-12)')
        ck(d_ps <= 1e-12, f'identity with jcp-fom2d psjax2d N=128 dt=0.001 incl. t0 (rel max {d_ps:.2e} <= 1e-12)')
        data = dict(rollout_seconds_cpu_2traj=sec, capture_vs_lattice=d_cap, vs_psjax2d=d_ps, wall_max=float(wall_max))
    elif name == 'cb3':
        sys.path.insert(1, str(LANE))
        import arm_gate as G
        import fcommon as F
        r = G.check_dev6()
        u128 = r.pop('u128')
        for row in r['rows']:
            print(f"  dev6|{row['case']} eligible={row['eligible']} eps_ref={row['eps_ref']:.2e}  PS vs E0 "
                  f"{row['err_vs_E0']:.3e}  self-conv {row['self_conv']:.3e}")
        print(f"  worst vs E0 {r['worst_err_vs_E0']:.3e} (bar 1e-4); worst self-conv {r['worst_self_conv']:.3e} (bar 1e-4);"
              f" seconds {r['seconds']}")
        ck(r['C_B3_pass'], f"C-B3: fid PS worst {r['worst_err_vs_E0']:.3e} <= 1e-4 vs E0 on all dev6")
        ck(r['self_conv_pass'], f"PS self-convergence worst {r['worst_self_conv']:.3e} <= 1e-4 on all dev6")
        z = np.load(G.INPUT)
        up = []
        for j, c in enumerate(z['case']):
            p = LANE / f'runs/a1/upw_dev6_{int(c):03d}_n257.npz'
            u = np.load(p)
            assert str(u['gen_sha256']) == INC_GEN and int(u['n']) == 257
            f, m = F.load_ref('dev6', int(c))
            ck(np.array_equal(f[1:], z['ref'][j]) and m['n0'] == z['n0'][j], f'gate input dev6|{c} = E0 store')
            up.append(G.metric(u['u'][1:], z['ref'][j], float(z['n0'][j])))
        print('  incumbent upwind (n=257 lattice nodes) vs E0: ' + ' '.join(f'{e:.3e}' for e in up))
        ck(max(up) > G.BAR_E0, f'C-B3 must-fail: incumbent upwind worst {max(up):.3e} > 1e-4 (fails the bar)')
        ck(all(e > G.BAR_E0 for e in up), f'... and fails it on every dev6 case (min {min(up):.3e})')
        g = G.r0_replacement(S_tr=np.zeros((4, 4)), worst=0.0, fingerprint={})
        gb = G.r0_replacement(S_tr=np.array([[0.0, np.nan]]), worst=0.0, fingerprint={})
        gw = G.r0_replacement(S_tr=np.zeros((4, 4)), worst=float('nan'), fingerprint={})
        ck(g['passed'] is True and gb['passed'] is False and gw['passed'] is False,
           'R0 replacement: passes on finite data, fails closed on a NaN state and on a NaN residual sentinel')
        data = dict(r, upwind_vs_E0=up, gate_pass=g['passed'], gate_nan_state=gb['passed'], gate_nan_sentinel=gw['passed'])
    print('RESULT ' + json.dumps(dict(checks=checks, data=data), default=float))


def dry(arm):
    job, t = stage(arm)
    out = job / 'out'
    env = dict(os.environ, PY=PY, TRAIN2D_DRYRUN='1', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1')
    tt = time.time()
    r = subprocess.run(['bash', 'experiments/imp-fronts/train2d/run_seed.sh', '1', str(out)], cwd=job,
                       capture_output=True, text=True, env=env, timeout=3600)
    (SCRATCH / f'dry_{arm}.log').write_text(r.stdout + r.stderr)
    ok = r.returncode == 0 and (out / 'SEED-COMPLETE').exists()
    rec = json.loads((out / 'run_record.json').read_text()) if ok else {}
    a = json.loads((out / 'stageA/train.json').read_text()) if (out / 'stageA/train.json').exists() else {}
    check(ok and a.get('generator_sha256') == sha(t / 'stageA/deps/burgers2d_film.py') != INC_GEN,
          f'dry {arm}: Stage A -> B -> rotation complete with the arm generator (rc={r.returncode}, '
          f'{time.time() - tt:.0f}s, log {SCRATCH / f"dry_{arm}.log"})')
    RESULTS[f'dry_{arm}'] = dict(rc=r.returncode, seconds=time.time() - tt, stage_seconds=rec.get('stage_seconds'),
                                 data_max_fom_rel_residual=a.get('data', {}).get('max_fom_rel_residual'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part')
    ap.add_argument('--deps')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--static-only', action='store_true', help='[S1]-[S4] only (no JAX; no slot needed)')
    a = ap.parse_args()
    if a.part:
        return run_part(a.part, a.deps)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    static()
    if a.static_only:
        print(f'\n{len(fails)} failure(s)')
        sys.exit(1 if fails else 0)
    print('[C-B4] cov parameter tables')
    _, tc = stage('cov')
    part('cb4', tc / 'stageA/deps')
    print('[cov-newton] cov generator, extreme real trajectories, 256 nodes (CPU)')
    part('covnewton', tc / 'stageA/deps')
    _, tf = stage('fid')
    print('[fid-contract] fid generator contract')
    part('fidcontract', tf / 'stageA/deps')
    print('[C-B3] fid gate on dev6 (staged tree, staged inputs)')
    part('cb3', tf / 'stageA/deps')
    if a.dry:
        print('[dry] chaining runs (TRAIN2D_DRYRUN=1)')
        for arm in ('fid', 'cov'):
            dry(arm)
    print(f'\n{len(fails)} failure(s)')
    for f in fails:
        print('  FAIL', f)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(fails=fails, results=RESULTS), indent=1, default=float) + '\n')
    if not a.keep:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
