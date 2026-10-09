"""jcp-wide-bank J2 / J4: 3D test-count trim (1d) and the 3D scaling law (1c) with off-mesh rules. DESIGN.md (+A0-A4)
is the specification.

Model side is the vendored, byte-identical code of the 2026-10-01 3D quadrature lane, imported unchanged:
vendor/quad3d/offmesh.py (point blocks, test block, `make_fsc_rule` = the vendor LM solver with the advection rule
injected, continuum targets) and its vendor/ modules (burgers3d-span/common.py, burgers3d-retry/tables.py). This driver
changes WHICH banks, (R', M) settings and rules run, and what is measured. The tensor rule runs only where the bank
config asks for it (J2: old bank). Per mesh n:

  per bank: lean tables (ordered bank rows on the mesh, nested Gram Cholesky L, A = Phi^T G_hat at M_max, lambda);
            gates: table Gram condition <= 1e8; G1 derivative vs FD; G2 mesh-node off-mesh assembly vs tensor and G4
            solver vs the vendor solver, both on a 32-column / 128-test tensor built for the gate only (A2-13, A3-13);
            G3 Jacobian vs jacfwd at the widest setting.
    per setting (R', M = complete_M(n, kappa R')), largest first; off-mesh blocks built at (R', M) per setting (A3-11):
      1  rollouts on the validation cohort and the certification draws: converged rule first (its certification
         reached states are the rho population), then the check rule, ladder, controls, tensor; refined errors vs the
         513-node reference on the 63^3 lattice (coefficients x bank at the lattice), field-metric distance from the
         converged rollout ||L^T (c - c_conv)|| / ||u0||, LM iterations / exits, eligibility (A0-3).
      2  rho of every rule on the population vs the continuum target (Gauss 80^3), target check (Gauss 64^3).
      3  gates (K-conv on validation + certification, K-target), controls per criterion, m* per family at both tau,
         named diagnostics m_d / m_rho; reached-state Jacobian singular values (A2-10).
      4  A-B-A timing (A0-8): A1 the setting's non-control arms, B one fixed FOM setting, A2; Jacobian microbenchmark.
         Deployed family fixed from this panel (A2-8).
    projection floor of the first R' columns vs the refined reference on the lattice (thin SVD, A2-6).
  final: cross-setting A-B-A panel of every setting's deployed arm (per mesh).
After all meshes: cross-mesh lattice distances of the same arm (A2-13). Outputs result.json + npz side files.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402   vendored, unchanged
from offmesh import C, TB  # noqa: E402

block = jax.block_until_ready


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def lattice65_coords():
    ax = np.arange(1, 64) / 64.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def lattice65_index(n):
    s = (n - 1) // 64
    assert s * 64 == n - 1, n
    k = np.arange(1, 64) * s
    I, J, K = np.meshgrid(k, k, k, indexing='ij')
    ni = n - 2
    return (((I - 1) * ni + (J - 1)) * ni + (K - 1)).ravel()


class MemSampler:
    """DESIGN A4-11: sampling thread (device bytes_in_use, host VmRSS), interval maxima."""

    def __init__(self, period=0.2):
        self.period, self.dev_max, self.host_max = period, 0, 0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def now():
        try:
            ms = jax.devices()[0].memory_stats()
            return dict(bytes_in_use=int(ms.get('bytes_in_use', -1)), lifetime_peak=int(ms.get('peak_bytes_in_use', -1)))
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def host_rss():
        try:
            for line in open('/proc/self/status'):
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
        except Exception:  # noqa: BLE001
            pass
        return -1

    def _run(self):
        while True:
            with self._lock:
                d = self.now().get('bytes_in_use', -1)
                self.dev_max, self.host_max = max(self.dev_max, d), max(self.host_max, self.host_rss())
            time.sleep(self.period)

    def interval(self):
        """Sampled maxima since the previous call (spikes shorter than the period can be missed), plus the
        allocator's current and lifetime-peak values at the boundary; resets the sampled maxima."""
        with self._lock:
            out = dict(device_bytes_in_use_max=self.dev_max, host_rss_max=self.host_max, sample_period_s=self.period,
                       boundary=self.now())
            self.dev_max, self.host_max = 0, 0
        return out


def eligible(rec, nonstat_frac):
    """A0-3 (3D): finite, zero reason-3 exits, at most nonstat_frac of the steps non-stationary (reason 0)."""
    rs = np.asarray(rec['reasons'])
    return bool(rec['finite'] and int((rs == 3).sum()) == 0 and int((rs == 0).sum()) <= nonstat_frac * len(rs))


def arm_eligible(rs, nonstat_frac):
    """DESIGN A7: arm-level eligibility pooled over a cohort (the cited 3D-lane contract): every rollout finite, zero
    reason-3 exits, non-stationary steps (reasons 0 and 2) <= nonstat_frac of all pooled steps."""
    if not rs:
        return False
    reasons = np.concatenate([np.asarray(r['reasons']) for r in rs])
    nonstat = int(((reasons == 0) | (reasons == 2)).sum())        # vendor/quad3d DESIGN: 0 and 2 are non-stationary
    return bool(all(r['finite'] for r in rs) and int((reasons == 3).sum()) == 0 and nonstat <= nonstat_frac * len(reasons))


def select_mstar(recs, rho_max, arms, family, tau, rho_bar, use_d=True, use_rho=True, nonstat_frac=0.01):
    """Smallest-m non-control arm of `family`: all cases eligible, (use_d) worst distance <= tau, (use_rho) worst
    continuum rho <= rho_bar. Returns (name, m) or (None, None)."""
    cands = sorted([a for a in arms if a['family'] == family and not a['control']], key=lambda a: a['m'])
    for a in cands:
        rs = recs.get(a['name'], [])
        if not arm_eligible(rs, nonstat_frac):
            continue
        if use_d:
            d = [r.get('dist_conv') for r in rs]
            if not all(v is not None and np.isfinite(v) for v in d) or max(d) > tau:
                continue
        if use_rho:
            rh = rho_max.get(a['name'])
            if rh is None or not np.isfinite(rh) or rh > rho_bar:
                continue
        return a['name'], a['m']
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda s_: print(f'[{el()}s] {s_}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    taus = dict(primary=float(cfg['tau']), secondary=float(cfg['tau_secondary']))
    rho_bar, nsf, dt = float(cfg['rho_bar']), float(cfg['nonstat_fraction']), float(cfg['dt'])
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi, x64=True,
               precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               offmesh_sha256=sha_file(Path(OM.__file__)), meshes={}, complete=False)
    assert cfg.get('expected_offmesh_sha256') in (None, rep['offmesh_sha256']), rep['offmesh_sha256']
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    mem = MemSampler()

    # ------------------------------------------------------------------ rules (vendored + lane extra), pinned
    rules = {}
    for path, want in cfg['rules_files']:
        pth = ROOT / path
        h = sha_file(pth)
        assert h == want, (path, h)
        z = np.load(pth)
        for k in z.files:
            if k.endswith('_X'):
                nm = k[:-2]
                assert nm not in rules or (np.array_equal(rules[nm][0], z[k]) and np.array_equal(rules[nm][1], z[nm + '_w'])), nm
                rules[nm] = (np.asarray(z[k]), np.asarray(z[nm + '_w']))
    ladder = cfg['ladder']                     # [{name, family}], controls carry control=True
    need = [r['name'] for r in ladder] + [cfg['converged'], cfg['check'], cfg['target'], cfg['target_check']]
    assert all(nm in rules for nm in need), [nm for nm in need if nm not in rules]

    # ------------------------------------------------------------------ cohort, certification, refined reference
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    assert cfg['cohort_seed'] != 923901, 'held-out cohort is not used in this lane'
    assert cfg.get('expected_cohort_sha256') in (None, tab['sha256']), tab['sha256']
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    cert = [(sd, j) for sd, cnt in cfg['cert_draws'] for j in range(cnt)]
    ctabs = {sd: C.table(sd, cnt) for sd, cnt in cfg['cert_draws']}
    rr = Path(cfg['refined_ref'])
    done = json.loads(rr.with_suffix('.done').read_text())
    assert done['sha256'] == sha_file(rr) and done['accepted'], done
    for k_, v_ in cfg['expected_ref'].items():
        assert done[k_] == v_, (k_, done[k_], v_)
    zr = np.load(rr)
    assert int(zr['seed']) == cfg['cohort_seed'] and int(zr['count']) >= cfg['cohort_count']
    RR = {j: np.asarray(zr[f'c{j}']) for j in cases}
    rep['refined'] = dict(path=str(rr), sha256=done['sha256'], done=done)
    X65 = lattice65_coords()

    coefs = {}                                   # (mesh, bank, setting, arm, case) -> (6, R') output coefficients
    for n in cfg['meshes']:
        M_rep = rep['meshes'][str(n)] = dict(banks={}, final_timing=None)
        i65 = lattice65_index(n)
        U0 = {j: block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases}
        NU = {j: float(tab['nu'][j]) for j in cases}
        n0 = {j: float(jnp.linalg.norm(U0[j])) for j in cases}
        UC = {(sd, j): block(jnp.asarray(C.initial_interior(n, ctabs[sd], j))) for sd, j in cert}
        NUC = {(sd, j): float(ctabs[sd]['nu'][j]) for sd, j in cert}
        n0c = {k: float(jnp.linalg.norm(v)) for k, v in UC.items()}
        # initial match of the refined reference on the lattice (A0, K-ref)
        M_rep['refined_initial_match'] = max(rel_max(np.asarray(U0[j])[i65], RR[j][0]) for j in cases)
        assert M_rep['refined_initial_match'] <= 1e-12, M_rep['refined_initial_match']
        n0r = {j: float(np.linalg.norm(RR[j][0])) for j in cases}
        fom = C.make_fom(n, dt, *cfg['fom_timing'])        # B phase of the A-B-A panels
        ridx = jnp.asarray(C.restrict_index(n, 16))
        tcases = cases[:cfg['timing_cases']]
        final = []
        fom_chk = {}
        for bk in cfg['banks']:
            bdir = ROOT / bk['model']
            b = pickle.loads((bdir / 'bank.pkl').read_bytes())
            bsha = sha_file(bdir / 'bank.pkl')
            assert bk.get('expected_sha256') in (None, bsha), (bk['name'], bsha)
            bank = jax.tree_util.tree_map(jnp.asarray, b['params'])
            T = np.asarray(b['rotation'])
            R = T.shape[1]
            spread = b['coefficient_rms_spread']
            assert all(str(r) in spread for r in bk['Rps']) and '32' in spread, ('spread keys', bk['name'], list(spread))
            order, lam_all = C.mode_order(n)
            Mof = {(Rp, k): int(C.complete_M(n, int(k * Rp), order, lam_all)) for Rp in bk['Rps'] for k in bk['kappas']}
            Mmax = max(Mof.values())
            B_rep = M_rep['banks'][bk['name']] = dict(sha256=bsha, R=R, M_of={f'{r}|{k}': v for (r, k), v in Mof.items()},
                                                     settings={}, gates={}, floors={})
            t0 = time.perf_counter()
            use_tensor = bool(bk.get('tensor'))
            tb = TB.build_tables(n, bank, T, sorted(set(bk['Rps'])), lambda r: Mmax, log=log, tensor=False)
            tbt = None
            trs = [r for r in bk['Rps'] if r <= 512]
            if use_tensor and trs:                  # tensor capped at 512 columns (audit w3d-1 item 4; A1/A2-13)
                Rt = max(trs)
                Mt = max(Mof[(r, k)] for r in trs for k in bk['kappas'])
                tbt = TB.build_tables(n, bank, T[:, :Rt], sorted(set(trs)), lambda r: Mt, log=log, tensor=True)
                B_rep['tensor_table'] = dict(R=Rt, M=Mt, bytes=int(tbt['Tsym'].nbytes))
            B_rep['table_seconds'] = time.perf_counter() - t0
            B_rep['gates']['gram_condition'] = tb['gram_cond'] ** 2
            assert B_rep['gates']['gram_condition'] <= 1e8, B_rep['gates']
            kx = tb['kxyz']
            # ------------------------------------------------ gates (G1-G4)
            rng = np.random.default_rng(20261008)
            # G1 (DESIGN A6): the vendored point set (lat4096[:32], as qpanel) with the vendored 2nd-order FD, AND a 4th-order
            # FD on the converged rule's first 32 points (near a corner, where the 2nd-order FD truncation reaches ~2e-6)
            Rw = max(bk['Rps'])
            h = 1e-5

            def fd_err(Xg, order):
                _, Dg = OM.point_blocks(bank, T[:, :Rw], Xg)
                f = lambda k: OM.point_blocks(bank, T[:, :Rw], Xg + k * h)[0]
                fd = ((f(1) - f(-1)) / (2 * h) if order == 2 else
                      (-f(2) + 8 * f(1) - 8 * f(-1) + f(-2)) / (12 * h))
                return rel_max(fd, Dg)
            B_rep['gates']['G1_derivative_vs_fd'] = fd_err(rules[cfg.get('g1_rule', 'lat4096')][0][:32], 2)
            B_rep['gates']['G1_converged_points_fd4'] = fd_err(rules[cfg['converged']][0][:32], 4)
            B_rep['gates']['G1_converged_points_fd2_diag'] = fd_err(rules[cfg['converged']][0][:32], 2)
            Ms = int(C.complete_M(n, 128, order, lam_all))
            tbs = TB.build_tables(n, bank, T[:, :32], [32], lambda r: Ms, log=log, tensor=True)
            rows = jnp.array(tbs['GTb'][0][:32])
            Drows = C.dminus(rows, n)
            cs = jnp.asarray(rng.normal(size=32) / np.sqrt(32))
            Xn = C.interior_coords(n)
            Ju_mesh = 0.
            for s in range(0, Xn.shape[0], 1 << 20):
                Pq = OM.test_block(n, Xn[s:s + (1 << 20)], np.full(min(1 << 20, Xn.shape[0] - s), float(n - 1) ** -3),
                                   tbs['kxyz'][:Ms])
                Ju_mesh = Ju_mesh + OM.contract_offmesh(dict(B=rows[:, s:s + (1 << 20)].T, D=Drows[:, s:s + (1 << 20)].T,
                                                             P=Pq), cs)
            Ju_t = OM.contract_tensor(dict(Ts=tbs['Tsym'][:Ms, :32, :32]), cs)
            B_rep['gates']['G2_meshnodes_offmesh_vs_tensor_32col'] = rel_max(Ju_mesh, Ju_t)
            del rows, Drows, Xn, Pq, Ju_mesh
            d32 = TB.arm_data(tbs, 32, Ms)
            tab_g = C.table(cert[0][0], 1)
            u0g = jnp.asarray(C.initial_interior(n, tab_g, 0))
            trust32 = float(cfg['trust_fraction'] * spread.get('32', spread[min(spread, key=lambda k: int(k))]))
            qv, _ = TB.make_fsc(n, 32, Ms, dt=dt, gtol=cfg['gtol'], trust=trust32)
            qr = OM.make_fsc_rule(n, 32, Ms, OM.contract_tensor, dt=dt, gtol=cfg['gtol'], trust=trust32)
            o_v, o_r = qv(u0g, float(tab_g['nu'][0]), d32, {}), qr(u0g, float(tab_g['nu'][0]), d32, {})
            B_rep['gates']['G4_solver_vs_vendor_fields_32col'] = rel_max(o_r[0], o_v[0])
            B_rep['gates']['G4_solver_vs_vendor_coefs_32col'] = rel_max(o_r[1], o_v[1])
            B_rep['gates']['G4_iterations_reasons_equal'] = all(bool(jnp.all(o_r[i] == o_v[i])) for i in (2, 3, 6))
            del tbs, d32, qv, qr, o_v, o_r
            Mw = Mof[(Rw, max(bk['kappas']))]
            Xr, wr = rules[cfg['converged']]
            d3 = OM.offmesh_tables(n, bank, T[:, :Rw], Xr[:4096], wr[:4096], kx[:Mw])
            c3 = jnp.asarray(rng.normal(size=Rw) / np.sqrt(Rw))
            J_ad = jax.jacfwd(lambda cc: OM.adv_offmesh_batch(d3, cc[None])[0])(c3)
            J_f = OM.contract_offmesh(d3, c3)
            B_rep['gates']['G3_jacobian_vs_jacfwd'] = rel_max(J_f, J_ad)
            B_rep['gates']['G3_adv_half_Jc'] = rel_max(0.5 * J_f @ c3, OM.adv_offmesh_batch(d3, c3[None])[0])
            del d3, J_ad, J_f
            g = B_rep['gates']
            log(f"[{n}|{bk['name']}] gates {g}")
            assert g['G1_derivative_vs_fd'] < 1e-6 and g['G1_converged_points_fd4'] < 1e-6, g
            assert g['G2_meshnodes_offmesh_vs_tensor_32col'] < 1e-12, g
            assert g['G3_jacobian_vs_jacfwd'] < 1e-12 and g['G3_adv_half_Jc'] < 1e-12, g
            assert g['G4_solver_vs_vendor_fields_32col'] <= 1e-12 and g['G4_solver_vs_vendor_coefs_32col'] <= 1e-12, g
            assert g['G4_iterations_reasons_equal'], g
            G65 = TB.feature_rows(bank, T, X65)                     # (R, 63^3) ordered bank on the lattice
            save()

            settings = sorted([(Rp, k) for Rp in bk['Rps'] for k in bk['kappas']], key=lambda t: (-t[0], -t[1]))
            for Rp, kap in settings:
                M = Mof[(Rp, kap)]
                key = f'R{Rp}_k{kap}'           # mesh-independent label (M is shell-completed per mesh)
                ts = time.perf_counter()
                trust = float(cfg['trust_fraction'] * spread[str(Rp)])
                lean = dict(GTb=TB.rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp],
                            lam=tb['lam'][:M])
                lean = jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), lean)
                sv = np.linalg.svd(np.asarray(lean['A']), compute_uv=False)
                S_rep_start_mem = mem.interval()['boundary']
                S_rep = B_rep['settings'][key] = dict(start_memory=S_rep_start_mem, 
                    Rp=Rp, M=M, kappa_nominal=kap, kappa=M / Rp, trust=trust, tensor_bytes=8 * M * Rp * Rp,
                    A_condition=float(sv[0] / sv[-1]), A_rank=int(np.sum(sv > 1e-12 * sv[0])), A_singular_values=sv.tolist(),
                    arms={})
                # the check rule is also a ladder candidate when it appears in the ladder: it runs ONCE (A2-12b) and is
                # aliased under its ladder name after phase 1
                alias = next((r for r in ladder if r['name'] == cfg['check']), None)
                specs = ([dict(name='conv', rule=cfg['converged'], family='ref', control=False),
                          dict(name='check', rule=cfg['check'], family='ref', control=False)] +
                         [dict(name=r['name'], rule=r['name'], family=r['family'], control=bool(r.get('control')))
                          for r in ladder if r['name'] != cfg['check']])
                q_off = OM.make_fsc_rule(n, Rp, M, OM.contract_offmesh, dt=dt, gtol=cfg['gtol'], trust=trust)
                arms = {}
                for sp in specs:
                    Xr, wr = rules[sp['rule']]
                    d = dict(lean, **OM.offmesh_tables(n, bank, T[:, :Rp], Xr, wr, kx[:M]))
                    arms[sp['name']] = dict(sp, m=len(wr), data=d, q=q_off, kind='offmesh')
                if tbt is not None and Rp <= tbt['Tsym'].shape[1] and M <= tbt['Tsym'].shape[0]:
                    d = dict(lean, Ts=block(jnp.asarray(tbt['Tsym'][:M, :Rp, :Rp])))
                    arms['tensor'] = dict(name='tensor', rule='tensor', family='tensor', control=False, m=None, data=d,
                                          kind='tensor', q=OM.make_fsc_rule(n, Rp, M, OM.contract_tensor, dt=dt,
                                                                           gtol=cfg['gtol'], trust=trust))
                for nm, s_ in arms.items():
                    S_rep['arms'][nm] = dict(rule=s_['rule'], family=s_['family'], control=s_['control'], m=s_['m'],
                                             bytes=(8 * s_['m'] * (2 * Rp + M)) if s_['kind'] == 'offmesh'
                                             else 8 * M * Rp * Rp)
                run = lambda nm, u0, nu: arms[nm]['q'](u0, nu, arms[nm]['data'], {})
                Lt = np.asarray(lean['L']).T
                # -------------------------------------------- phase 1: rollouts (validation + certification)
                recs, crecs, pop, kpop = {}, {}, [], []
                for nm in ['conv'] + [x for x in arms if x != 'conv']:
                    t_c = time.perf_counter()
                    rl, cl = [], []
                    for j in cases:
                        o = run(nm, U0[j], NU[j])
                        block(o)
                        Cout = np.asarray(o[1])[::int(round(0.05 / dt))]
                        coefs[(n, bk['name'], key, nm, j)] = Cout
                        F = Cout @ np.asarray(G65[:Rp])
                        e = np.linalg.norm(F - RR[j], axis=1) / n0r[j]
                        rec = dict(case=j, err_refined=e.tolist(), worst_refined=float(e[1:].max()),
                                   reasons=np.asarray(o[3]).tolist(), iterations=np.asarray(o[2]).tolist(),
                                   rejected=np.asarray(o[6]).tolist(),
                                   finite=bool(np.isfinite(np.asarray(o[0])).all() and np.isfinite(Cout).all()))
                        rec['eligible'] = eligible(rec, nsf)
                        if nm != 'conv':
                            dd = np.linalg.norm((Cout - coefs[(n, bk['name'], key, 'conv', j)]) @ Lt.T, axis=1) / n0[j]
                            rec['dist_conv'] = float(dd[1:].max())
                        if j in cfg.get('audit_cases', []):
                            np.savez(out / 'fields' / f'{n}_{bk["name"]}_{key}_{nm}_c{j}.npz', internal=np.asarray(o[1]),
                                     f65=F)
                        rl.append(rec)
                        del o
                    if nm in ('conv', 'check'):
                        for kc in cert:
                            o = run(nm, UC[kc], NUC[kc])
                            block(o)
                            W = np.asarray(o[1])
                            rec = dict(draw=list(kc), reasons=np.asarray(o[3]).tolist(),
                                       finite=bool(np.isfinite(W).all()))
                            rec['eligible'] = eligible(rec, nsf)
                            if nm == 'conv':
                                pop.append(W[1:])
                                kpop.append(np.arange(1, W.shape[0]))
                                coefs[(n, bk['name'], key, 'cert_conv', kc)] = W
                            else:
                                coefs[(n, bk['name'], key, 'cert_check', kc)] = W
                                dd = np.linalg.norm((W - coefs[(n, bk['name'], key, 'cert_conv', kc)]) @ Lt.T, axis=1) / n0c[kc]
                                rec['dist_conv'] = float(dd.max())
                            cl.append(rec)
                            del o
                        crecs[nm] = cl
                    recs[nm] = rl
                    S_rep['arms'][nm].update(
                        compile_and_run_s=time.perf_counter() - t_c,
                        worst_refined=max(r['worst_refined'] for r in rl),
                        median_refined=float(np.median([r['worst_refined'] for r in rl])),
                        worst_dist_conv=(max(r['dist_conv'] for r in rl) if nm != 'conv' else 0.),
                        all_eligible_per_rollout_legacy=all(r['eligible'] for r in rl), arm_eligible_pooled=arm_eligible(rl, nsf),
                        lm_iterations_per_query_median=float(np.median([sum(r['iterations']) for r in rl])),
                        reason_counts={str(k): int(sum(np.sum(np.asarray(r['reasons']) == k) for r in rl)) for k in range(5)},
                        cases=rl)
                    log(f"[{n}|{bk['name']}|{key}] {nm}: worst refined {S_rep['arms'][nm]['worst_refined']:.4%} "
                        f"dist {S_rep['arms'][nm]['worst_dist_conv']:.2e} eligible(pooled, A7) {S_rep['arms'][nm]['arm_eligible_pooled']}")
                if alias is not None:
                    an = alias['name']
                    arms[an] = dict(arms['check'], name=an, family=alias['family'], control=bool(alias.get('control')))
                    recs[an] = recs['check']
                    for j in cases:
                        coefs[(n, bk['name'], key, an, j)] = coefs[(n, bk['name'], key, 'check', j)]
                    S_rep['arms'][an] = dict(S_rep['arms']['check'], family=alias['family'], alias_of='check')
                save()
                # -------------------------------------------- phase 2: rho on the converged rule's reached states
                Cs = np.concatenate(pop)
                tgt = np.concatenate([np.asarray(OM.continuum_adv(n, bank, T, *rules[cfg['target']], kx[:M], Cs[i:i + 64]))
                                      for i in range(0, len(Cs), 64)])          # 64-state batches (A3-11)
                chk = np.concatenate([np.asarray(OM.continuum_adv(n, bank, T, *rules[cfg['target_check']], kx[:M],
                                                                  Cs[i:i + 64])) for i in range(0, len(Cs), 64)])
                tn = np.linalg.norm(tgt, axis=1)
                assert tn.min() > 0, 'zero continuum target'
                rho = lambda v: np.linalg.norm(v - tgt, axis=1) / tn
                S_rep['rho'] = dict(states=len(Cs), target_norm_min=float(tn.min()), check_worst=float(rho(chk).max()),
                                    rules={})
                rho_max = {}
                for nm, s_ in arms.items():
                    if s_['kind'] == 'tensor':
                        v = np.asarray(jax.jit(lambda d, X: jax.lax.map(lambda cc: 0.5 * OM.contract_tensor(d, cc) @ cc, X))(
                            s_['data'], jnp.asarray(Cs)))
                    else:
                        v = np.concatenate([np.asarray(OM.adv_offmesh_batch(s_['data'], jnp.asarray(Cs[i:i + 256])))
                                            for i in range(0, len(Cs), 256)])
                    r_ = rho(v)
                    S_rep['rho']['rules'][nm] = dict(worst=float(r_.max()), median=float(np.median(r_)),
                                                     p95=float(np.quantile(r_, .95)), p90=float(np.quantile(r_, .9)))
                    rho_max[nm] = float(r_.max())
                    if nm == cfg.get('audit_rho_arm'):
                        sub = slice(None, None, max(1, len(Cs) // 64))
                        np.savez(out / 'fields' / f'rho_{n}_{bk["name"]}_{key}_{nm}.npz', Cs=Cs[sub], v=v[sub], tgt=tgt[sub])
                np.savez(out / 'fields' / f'population_{n}_{bk["name"]}_{key}.npz', Cs=Cs, k=np.concatenate(kpop))
                del tgt, chk
                # -------------------------------------------- phase 3: gates, controls, selection
                conv_ok = arm_eligible(recs['conv'], nsf) and arm_eligible(crecs['conv'], nsf)          # A7: pooled
                chk_d = max([r['dist_conv'] for r in recs['check']] + [r['dist_conv'] for r in crecs['check']])
                chk_ok = arm_eligible(recs['check'], nsf) and arm_eligible(crecs['check'], nsf)
                S_rep['gates'] = dict(
                    converged=dict(conv_all_eligible=conv_ok, check_all_eligible=chk_ok, check_worst_distance=chk_d,
                                   bar=cfg['conv_bar'], passed=bool(conv_ok and chk_ok and chk_d <= cfg['conv_bar'])),
                    target=dict(check_worst=S_rep['rho']['check_worst'], bar=cfg['target_bar'],
                                passed=bool(S_rep['rho']['check_worst'] <= cfg['target_bar'])))
                arm_list = [dict(name=nm, family=s_['family'], m=s_['m'], control=s_['control']) for nm, s_ in arms.items()
                            if s_['kind'] == 'offmesh']
                ctrl = {}
                for nm, s_ in arms.items():
                    if s_['control']:
                        dm = max(r['dist_conv'] for r in recs[nm])
                        ae = arm_eligible(recs[nm], nsf)
                        fd = {t_: bool(not (np.isfinite(dm) and dm <= tv)) for t_, tv in taus.items()}
                        fr = bool(not (np.isfinite(rho_max[nm]) and rho_max[nm] <= rho_bar))
                        ctrl[nm] = dict(worst_distance=dm, rho_max=rho_max[nm], all_eligible=ae, fails_distance=fd,
                                        fails_rho=fr, would_be_selected={t_: bool(not fd[t_] and not fr) for t_ in taus})  # A0-2
                disc = {t_: bool(ctrl and not any(v['would_be_selected'][t_] for v in ctrl.values())) for t_ in taus}
                valid = bool(S_rep['gates']['converged']['passed'] and S_rep['gates']['target']['passed'])
                sel = {}
                for t_, tv in taus.items():
                    for fam in cfg['families']:
                        nm_, m_ = select_mstar(recs, rho_max, arm_list, fam, tv, rho_bar, nonstat_frac=nsf)
                        nd, md = select_mstar(recs, rho_max, arm_list, fam, tv, rho_bar, use_rho=False, nonstat_frac=nsf)
                        ok = valid and disc[t_]
                        sel[f'{t_}|{fam}'] = dict(tau=tv, family=fam, gates_passed=ok, available=bool(ok and nm_ is not None),
                                                  reason=(None if ok and nm_ is not None else
                                                          ('gates' if not valid else 'controls' if not disc[t_]
                                                           else 'no ladder member qualifies')),
                                                  arm=nm_ if ok else None,
                                                  m=m_ if ok else None, arm_raw=nm_, m_raw=m_, arm_d=nd, m_d=md)
                for fam in cfg['families']:
                    nr, mr = select_mstar(recs, rho_max, arm_list, fam, 0., rho_bar, use_d=False, nonstat_frac=nsf)
                    sel[f'rho_only|{fam}'] = dict(arm_rho=nr, m_rho=mr)
                S_rep['controls'] = ctrl
                S_rep['certification'] = crecs
                S_rep['selection'] = dict(valid=valid, discriminating=disc, entries=sel)
                # reached-state Jacobian (A2-10), first validation case, converged rule
                Wj = np.asarray(arms['conv']['q'](U0[cases[0]], NU[cases[0]], arms['conv']['data'], {})[1])
                Sd = 1. / (1. + dt * NU[cases[0]] * np.asarray(lean['lam']))
                jc = {}
                for k_ in cfg.get('jacobian_states', [1, 12, 25]):
                    Ju = np.asarray(OM.contract_offmesh(arms['conv']['data'], jnp.asarray(Wj[k_])))
                    A_ = np.asarray(lean['A'])
                    Jm = Sd[:, None] * (A_ + dt * (Ju + NU[cases[0]] * np.asarray(lean['lam'])[:, None] * A_))
                    if not np.isfinite(Jm).all():
                        jc[str(k_)] = dict(finite=False)
                        continue
                    s2 = np.linalg.svd(Jm, compute_uv=False)
                    jc[str(k_)] = dict(singular_values=s2.tolist(), condition=float(s2[0] / s2[-1]), rank=int(np.sum(s2 > 1e-12 * s2[0])),
                                       sigma_min=float(s2[-1]))
                S_rep['jacobian_reached'] = jc
                save()
                # -------------------------------------------- phase 4: A-B-A timing + microbenchmark
                timed = [nm for nm, s_ in arms.items() if not s_['control'] and nm != 'check']   # alias timed by name
                prng = np.random.default_rng(cfg.get('timing_seed', 20261008))
                xb = jnp.ones((2048, 2048))

                def burn(sec):
                    t_ = time.perf_counter()
                    while time.perf_counter() - t_ < sec:
                        block(xb @ xb)
                ref16 = {(nm, j): coefs[(n, bk['name'], key, nm, j)] for nm in timed for j in tcases}
                fchk = {}                           # restricted decoded fields (every 16th node) + full-field sums
                for nm in timed:
                    for j in tcases:
                        o = run(nm, U0[j], NU[j])
                        fchk[(nm, j)] = (np.asarray(o[0][:, ridx]), float(jnp.sum(o[0])), float(jnp.sum(o[0] * o[0])))
                        del o
                for j in tcases:
                    o = fom(U0[j], NU[j])
                    fchk[('fom', j)] = (np.asarray(o[0][:, ridx]), float(jnp.sum(o[0])), float(jnp.sum(o[0] * o[0])))
                    del o

                def chkd(key_, o):
                    r16, s1, s2 = fchk[key_]
                    return max(rel_max(np.asarray(o[0][:, ridx]), r16), abs(float(jnp.sum(o[0])) - s1) / max(abs(s1), 1e-300),
                               abs(float(jnp.sum(o[0] * o[0])) - s2) / s2)
                for nm in timed:                       # warm every signature
                    for j in tcases:
                        block(run(nm, U0[j], NU[j]))
                for j in tcases:
                    block(fom(U0[j], NU[j]))
                inv = []
                for ph, kind, names in (('A1', 'rom', timed), ('B', 'fom', ['fom']), ('A2', 'rom', timed)):
                    order_ = [(nm, j, r) for nm in names for j in tcases for r in range(cfg['reps'])]
                    prng.shuffle(order_)
                    burn(2.0)
                    for nm, j, r in order_:
                        t_ = time.perf_counter()
                        o = run(nm, U0[j], NU[j]) if kind == 'rom' else fom(U0[j], NU[j])
                        block(o)
                        sec = time.perf_counter() - t_
                        ent = dict(phase=ph, name=nm, case=j, rep=r, seconds=sec)
                        if kind == 'rom':
                            ent['max_diff'] = max(rel_max(np.asarray(o[1])[::int(round(0.05 / dt))], ref16[(nm, j)]),
                                                  chkd((nm, j), o))
                        else:
                            ent['max_diff'] = chkd(('fom', j), o)
                        inv.append(ent)
                        del o
                        if sec >= 0.5:
                            time.sleep(min(1.0, sec))
                            burn(0.2)
                tim = {}
                for nm in timed + ['fom']:
                    xs = [r for r in inv if r['name'] == nm]
                    a1 = [r['seconds'] for r in xs if r['phase'] == 'A1']
                    a2 = [r['seconds'] for r in xs if r['phase'] == 'A2']
                    tim[nm] = dict(median_ms=1e3 * float(np.median([r['seconds'] for r in xs])),
                                   drift=(float(np.median(a2) / np.median(a1)) if a1 and a2 else None))
                dr = [v['drift'] for v in tim.values() if v['drift'] is not None]
                det = max(r.get('max_diff', 0.) for r in inv)
                S_rep['timing'] = dict(cases=tcases, reps=cfg['reps'], invocations=inv, subjects=tim,
                                       gates=dict(drift_worst=float(max(max(dr), 1 / min(dr))),
                                                  drift_pass=bool(max(max(dr), 1 / min(dr)) <= 1.10),
                                                  deterministic_max_diff=det, deterministic=bool(det <= 1e-12)))
                mb = {}
                burn(2.0)
                for nm in timed:
                    fn = jax.jit(OM.contract_tensor if arms[nm]['kind'] == 'tensor' else OM.contract_offmesh)
                    cc = jnp.asarray(coefs[(n, bk['name'], key, nm, cases[0])][3])
                    for _ in range(5):
                        block(fn(arms[nm]['data'], cc))
                    tt = []
                    for _ in range(50):
                        t_ = time.perf_counter()
                        block(fn(arms[nm]['data'], cc))
                        tt.append(time.perf_counter() - t_)
                    mb[nm] = dict(jacobian_ms_median=1e3 * float(np.median(tt)), jacobian_ms_min=1e3 * float(np.min(tt)),
                                  seconds=tt)
                S_rep['microbench'] = mb
                tvalid = bool(S_rep['timing']['gates']['drift_pass'] and S_rep['timing']['gates']['deterministic'])
                S_rep['timing']['valid'] = tvalid
                dep = [((tim[e_['arm']]['median_ms'] if tvalid else e_['m']), e_['family'], e_['arm']) for k_, e_ in sel.items()
                       if k_.startswith('primary|') and e_.get('arm') is not None]
                if dep:
                    _, fam_, arm_ = min(dep)
                    S_rep['deployed'] = dict(family=fam_, arm=arm_, m=arms[arm_]['m'], median_ms=tim[arm_]['median_ms'],
                                             timing_valid=tvalid, chosen_by='timed_median' if tvalid else
                                             'smaller_m (K-time failed; diagnostic only, DESIGN A5)')
                    final.append(dict(key=f"{bk['name']}|{key}|{arm_}", q=arms[arm_]['q'], data=arms[arm_]['data'],
                                      ref={j: coefs[(n, bk['name'], key, arm_, j)] for j in tcases},
                                      fchk={j: fchk[(arm_, j)] for j in tcases}))
                    if not fom_chk:
                        fom_chk.update({j: fchk[('fom', j)] for j in tcases})
                else:
                    S_rep['deployed'] = None
                S_rep['seconds'] = time.perf_counter() - ts
                S_rep['memory'] = mem.interval()
                log(f"[{n}|{bk['name']}|{key}] done in {S_rep['seconds']:.0f}s deployed {S_rep['deployed']} "
                    f"timing gates {S_rep['timing']['gates']}")
                save()
                keep = {id(f_['data']) for f_ in final}
                for nm in list(arms):
                    if id(arms[nm]['data']) not in keep:
                        del arms[nm]
                d = s_ = o = None                   # drop loop references to rule blocks (audit w3d-1 item 12)
                del arms, lean, pop, Cs
                jax.clear_caches()
                gc.collect()
            # ---------------------------------------------------- projection floor per R' (thin SVD, A2-6)
            for Rp in sorted(set(bk['Rps'])):
                Gm = np.asarray(G65[:Rp]).T                                   # (63^3, R')
                U_, s_, _ = np.linalg.svd(Gm, full_matrices=False)
                rk = int(np.sum(s_ > 1e-12 * s_[0]))
                U_ = U_[:, :rk]
                per = []
                for j in cases:
                    res_ = RR[j] - (RR[j] @ U_) @ U_.T
                    e = np.linalg.norm(res_, axis=1) / n0r[j]
                    per.append(float(e[1:].max()))
                sol = np.linalg.lstsq(Gm, RR[cases[0]][-1], rcond=1e-12)[0]
                lres = float(np.linalg.norm(Gm @ sol - RR[cases[0]][-1])) / n0r[cases[0]]
                fl_last = float(np.linalg.norm(RR[cases[0]][-1] - (RR[cases[0]][-1] @ U_) @ U_.T)) / n0r[cases[0]]
                assert rk > 0
                B_rep['floors'][str(Rp)] = dict(rank=rk, condition=float(s_[0] / s_[rk - 1]), worst=max(per),
                                                median=float(np.median(per)), per_case=per,
                                                lstsq_vs_svd=(abs(lres - fl_last) / fl_last if fl_last > 1e-12
                                                              else abs(lres - fl_last)))
                B_rep['floors'][str(Rp)]['check_passed'] = bool(np.isfinite(B_rep['floors'][str(Rp)]['lstsq_vs_svd']) and
                                                                B_rep['floors'][str(Rp)]['lstsq_vs_svd'] <= 1e-10)
                del Gm, U_
            del tb, tbt, G65
            gc.collect()
            save()
        # -------------------------------------------------------- final cross-setting panel (this mesh)
        if final and cfg.get('final_panel', True):
            prng = np.random.default_rng(cfg.get('timing_seed', 20261008) + n)
            for f_ in final:
                for j in tcases:
                    block(f_['q'](U0[j], NU[j], f_['data'], {}))
            for j in tcases:
                block(fom(U0[j], NU[j]))
            xb = jnp.ones((2048, 2048))
            inv = []
            for ph, names in (('A1', [f_['key'] for f_ in final]), ('B', ['fom']), ('A2', [f_['key'] for f_ in final])):
                order_ = [(nm, j, r) for nm in names for j in tcases for r in range(cfg['reps'])]
                prng.shuffle(order_)
                t_ = time.perf_counter()
                while time.perf_counter() - t_ < 2.0:
                    block(xb @ xb)
                for nm, j, r in order_:
                    f_ = next((x for x in final if x['key'] == nm), None)
                    t_ = time.perf_counter()
                    o = f_['q'](U0[j], NU[j], f_['data'], {}) if f_ else fom(U0[j], NU[j])
                    block(o)
                    sec = time.perf_counter() - t_
                    ent = dict(phase=ph, name=nm, case=j, rep=r, seconds=sec)
                    r16, s1, s2 = f_['fchk'][j] if f_ else fom_chk[j]
                    ent['max_diff'] = max(rel_max(np.asarray(o[0][:, ridx]), r16),
                                          abs(float(jnp.sum(o[0])) - s1) / max(abs(s1), 1e-300),
                                          abs(float(jnp.sum(o[0] * o[0])) - s2) / s2)
                    if f_:
                        ent['max_diff'] = max(ent['max_diff'], rel_max(np.asarray(o[1])[::int(round(0.05 / dt))], f_['ref'][j]))
                    inv.append(ent)
                    del o
                    if sec >= 0.5:
                        time.sleep(min(1.0, sec))
                        t2 = time.perf_counter()
                        while time.perf_counter() - t2 < 0.2:
                            block(xb @ xb)
            tim = {}
            for nm in [f_['key'] for f_ in final] + ['fom']:
                xs = [r for r in inv if r['name'] == nm]
                a1 = [r['seconds'] for r in xs if r['phase'] == 'A1']
                a2 = [r['seconds'] for r in xs if r['phase'] == 'A2']
                tim[nm] = dict(median_ms=1e3 * float(np.median([r['seconds'] for r in xs])),
                               drift=(float(np.median(a2) / np.median(a1)) if a1 and a2 else None))
            dr = [v['drift'] for v in tim.values() if v['drift'] is not None]
            det = max(r.get('max_diff', 0.) for r in inv)
            M_rep['final_timing'] = dict(retained_bytes=int(sum(sum(x.nbytes for x in jax.tree_util.tree_leaves(f_['data']))
                                                                for f_ in final)), invocations=inv, subjects=tim, gates=dict(
                drift_worst=float(max(max(dr), 1 / min(dr))), drift_pass=bool(max(max(dr), 1 / min(dr)) <= 1.10),
                deterministic_max_diff=det, deterministic=bool(det <= 1e-12)))
        final.clear()
        jax.clear_caches()
        gc.collect()
        save()

    # ------------------------------------------------------------ cross-mesh distances on the lattice (A2-13)
    ms = cfg['meshes']
    if len(ms) > 1:
        xm = {}
        for bk in cfg['banks']:
            b = pickle.loads((ROOT / bk['model'] / 'bank.pkl').read_bytes())
            G65 = np.asarray(TB.feature_rows(jax.tree_util.tree_map(jnp.asarray, b['params']), np.asarray(b['rotation']), X65))
            for (n_, bn, key, nm, j), Cv in coefs.items():
                if n_ != ms[0] or bn != bk['name'] or nm in ('cert_conv', 'cert_check') or not isinstance(j, int):
                    continue
                other = coefs.get((ms[1], bn, key, nm, j))
                if other is None or other.shape != Cv.shape:
                    continue
                Rp = Cv.shape[1]
                d = np.linalg.norm((Cv - other) @ G65[:Rp], axis=1) / np.linalg.norm(RR[j][0])
                xm.setdefault(f'{bn}|{key}|{nm}', []).append(float(d[1:].max()))
            del G65
        rep['cross_mesh'] = {k: dict(worst=max(v), median=float(np.median(v)), cases=len(v)) for k, v in xm.items()}
    np.savez(out / 'fields' / 'coefficients.npz',
             **{f'{n_}|{bn}|{key}|{nm}|{j if isinstance(j, int) else "_".join(map(str, j))}': v
                for (n_, bn, key, nm, j), v in coefs.items()})
    pan = [S_['timing'].get('valid') for m_ in rep['meshes'].values() for b_ in m_['banks'].values()
           for S_ in b_['settings'].values()]
    fin = [m_['final_timing']['gates'] for m_ in rep['meshes'].values() if m_.get('final_timing')]
    rep['timing_valid_jobwide'] = bool(pan and all(pan) and all(g_['drift_pass'] and g_['deterministic'] for g_ in fin))
    rep['seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('W3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
