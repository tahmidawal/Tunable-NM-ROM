"""CPU-only fit worker: `varpro.bounded_nnls`, verbatim, in a process that imports no JAX.

    python fitworker.py <design.npy> <b.npy> <m> <seconds> <block> <out.npz> [<sidecar.json>]

The driver builds the design on the GPU, writes it to disk, and launches one of these per
(rung, arm, m) so several rules fit concurrently on the allocation's CPU cores while the GPU
certifies each one as it lands. The fitter is copied here rather than imported because
`varpro` imports JAX and the worker must never touch the device; `smoke_eqtop.py` gates
this copy against `varpro.bounded_nnls` bitwise on the same design.

The sidecar carries `unreachable_residual` and `b_norm` for a QR-compressed design, so the
relative fit reported is the fit on the ORIGINAL rows: ||D_S w - b||^2 = ||R_S w - c||^2 + r^2.
"""
import json
import sys
import time

import numpy as np
import scipy.optimize


def bounded_nnls(G, b, target, seconds, block=None):
    """`varpro.bounded_nnls`, verbatim."""
    t0 = time.perf_counter()
    block = int(block or max(1, target // 64))
    support = np.zeros(0, dtype=int)
    w = np.zeros(0)
    residual = b.copy()
    refits, truncated, reason = 0, False, 'target'
    while len(support) < target:
        if time.perf_counter() - t0 > seconds:
            truncated, reason = True, 'walltime'
            break
        grad = G.T @ residual
        if support.size:
            grad[support] = -np.inf
        take = min(block, target - len(support))
        order = np.argsort(-grad)[:take]
        order = order[grad[order] > 1e-10 * max(np.linalg.norm(b), 1e-300)]
        if order.size == 0:
            reason = 'gradient'
            break
        support = np.concatenate((support, order))
        w, _ = scipy.optimize.nnls(G[:, support], b, maxiter=20 * len(support))
        refits += 1
        active = w > 1e-14
        support, w = support[active], w[active]
        residual = b - G[:, support] @ w
    return np.asarray(support, dtype=int), w, dict(
        fitter='bounded_block_greedy', block=block, target_support=int(target),
        support=int(len(support)), refits=int(refits), truncated=bool(truncated),
        stop_reason=reason, seconds=time.perf_counter() - t0)


def main():
    design, bfile, m, seconds, block, out = sys.argv[1:7]
    side = json.loads(open(sys.argv[7]).read()) if len(sys.argv) > 7 else {}
    t0 = time.perf_counter()
    G = np.array(np.load(design, mmap_mode='r'))
    b = np.load(bfile)
    load = time.perf_counter() - t0
    supp, w, info = bounded_nnls(G, b, int(m), float(seconds), block=int(block))
    r2 = float(np.linalg.norm(G[:, supp] @ w - b)) ** 2 + float(side.get('unreachable_residual', 0.)) ** 2
    bn = float(side.get('b_norm', np.linalg.norm(b)))
    info.update(relative_fit=float(np.sqrt(r2) / max(bn, 1e-300)), load_seconds=load,
                rows=int(G.shape[0]), candidates=int(G.shape[1]), m=int(len(supp)),
                m_target=int(m), truncated=bool(info['truncated']))
    np.savez(out, support=supp, weights=w, info=json.dumps(info))
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
