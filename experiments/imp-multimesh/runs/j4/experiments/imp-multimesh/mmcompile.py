"""Fresh-process compile measurement (DESIGN A5.1). One deployed query per process; persistent cache disabled.

    python mmcompile.py ours <L> <Rp> <rule> <scheme> <out.json>
    python mmcompile.py comp <L> <arm_name> <source> <source_dir> <build_dir> <r> <k> <out.json>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

t_start = time.perf_counter()


def main():
    assert os.environ.get('JAX_ENABLE_COMPILATION_CACHE', '').lower() == 'false'
    import mmcore as M
    import mmbuild as B
    side, L = sys.argv[1], int(sys.argv[2])
    out = {}
    if side == 'ours':
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'jcp-p1-scaling'))
        import p1common as P
        Rp, rule, sc, outp = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
        bank = P.Bank(L, [0, 128, 256, 384, 512] if Rp > 128 else [0, 128])
        bank.ops('frozen')
        st = P.Setting(bank, 'c', Rp, rule, 'frozen', roles={}, nrec=51)
        ph = P.case_list({})[0][2]
        u0 = jnp.asarray(P.Q.e.initial(L, ph))
        info = M.aot_compile(st.q['LSPG']['fused'], u0, jnp.float64(ph[4]), st.data[('main', 'LSPG')], st.cold,
                             st.sched('LSPG', sc, P.DT0, 'prod'))
        out = dict(side='ours', L=L, Rp=Rp, rule=rule, scheme=sc, **info)
    else:
        name, s, sdir, bdir, r, k, outp = sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7]), \
            int(sys.argv[8]), sys.argv[9]
        Phi_s, C_s, rules_s, trust_s = B.load_source(sdir)
        if s == L:
            Phi, R = jnp.asarray(np.asarray(Phi_s[:, :r])), None
            tr = trust_s[r]
        else:
            Phi, R, _ = M.resample_basis(np.asarray(Phi_s[:, :r]), s, L)
            tr = M.BD.trust_radii(M.transform_codes(C_s, s, L, r, R), [r], 2)[r]
        z = np.load(Path(bdir) / f'W_{name}_r{r}_k{k}.npz')
        cell = M.Cell(L, Phi, {r: tr})
        data, _ = cell.ecsw_data(r, z['nodes'], z['weights'])
        fam = M.H.Family(2, 'dev6')
        q = M.RM.make_query('ecsw', 2, L, 'common', trust=tr)
        info = M.aot_compile(q, jnp.asarray(fam.ic(L, 0)), jnp.float64(fam.nu(0)), data,
                             M.RM.schedule(2, 'CNR', 0.005, 'prod'))
        out = dict(side='comp', L=L, arm=name, r=r, k=k, **info)
    out['process_seconds'] = time.perf_counter() - t_start
    Path(outp).write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
