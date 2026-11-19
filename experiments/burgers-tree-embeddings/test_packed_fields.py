"""Small independent compact/packed equality and geometry-reuse fixtures."""
from common import *
import discrete_weak as W, weak_quadrature as QW, packed_fields as P
import geometry, tensor_metric as TM
from build_local_cache import expand_matrix


def main():
    L, root, rank, M = 32, 4, 3, 80
    basis = jnp.zeros((root, 12, 12)).at[jnp.arange(root), jnp.array([0, 1, 0, 1]), jnp.array([0, 0, 1, 1])].set(2.)
    rotation = jnp.eye(36, rank)
    geometries = [geometry.uniform(1), [[1, 0, 0], [1, 1, 1], [2, 0, 0], [2, 3, 3]]]
    traces = []
    def traced(c, cache):
        traces.append(1)
        return P.fields(c, cache)
    compiled = jax.jit(traced)
    worst_step_change = 0.
    for Q in (16, 32):
        for cells in geometries:
            G, transforms, _, _ = TM.make_geometry(6)(basis, jnp.asarray(cells), *TM.rule(N=32))
            expansion = expand_matrix(rotation, len(cells), root)
            G = expansion.T @ G @ expansion
            original, slices = W.prepare(basis, cells, transforms, rotation, L, M, G)
            q, qslices = QW.prepare(basis, cells, transforms, rotation, L, Q)
            original = dict(original, quadrature=q)
            packed = {key:original[key] for key in ['A', 'indices', 'lam', 'gram']}
            packed['packed'] = P.prepare_stencil(basis, cells, transforms, rotation, L, Q)
            compact = QW.make_fields(qslices, rank)
            c = jnp.zeros(root + len(cells)*rank).at[0].set(.2).at[root:].set(.001)
            expected = compact(c, original)
            actual = compiled(c, packed)
            assert float(jnp.max(jnp.abs(actual-expected))) < 1e-12
            direction = jnp.cos(jnp.arange(len(c)))
            jp = jax.jvp(lambda z:P.fields(z, packed), (c,), (direction,))[1]
            jc = jax.jvp(lambda z:compact(z, original), (c,), (direction,))[1]
            assert float(jnp.max(jnp.abs(jp-jc))) < 1e-11
            test = jnp.sin(jnp.arange(expected.size)).reshape(expected.shape)
            _, vp = jax.vjp(lambda z:P.fields(z, packed), c)
            _, vc = jax.vjp(lambda z:compact(z, original), c)
            assert float(jnp.max(jnp.abs(vp(test)[0]-vc(test)[0]))) < 1e-10
            adv = QW.make_tested(compact, L, Q)
            padv = P.tested_advection(L, Q)
            assert float(jnp.max(jnp.abs(adv(c, original)-padv(c, packed)))) < 1e-12
            compact_center = lambda z, cache:compact(z, cache)[0]
            cp, ep, _ = W.make_initial_fit(P.center)(expected[0], packed)
            cc, ec, _ = W.make_initial_fit(compact_center)(expected[0], original)
            assert max(float(ep), float(ec)) < 1e-7
            assert float(jnp.linalg.norm(P.center(cp, packed)-compact_center(cc, original))/jnp.linalg.norm(expected[0])) < 1e-9
            sp, _ = W.make_solver(P.center, L, theta=.5, advection_tests=padv)
            sc, _ = W.make_solver(compact_center, L, theta=.5, advection_tests=adv)
            pc, ps = sp(c, packed, .0025, .005, 1e-8, 1e-10)
            oc, os = sc(c, original, .0025, .005, 1e-8, 1e-10)
            assert max(float(ps[1]), float(os[1])) < 1e-8
            difference = float(jnp.linalg.norm(P.center(pc, packed)-compact_center(oc, original))/jnp.linalg.norm(expected[0]))
            worst_step_change = max(worst_step_change, difference)
            assert difference < 1e-9
        assert len(traces) == (1 if Q == 16 else 2), 'Geometry values must not create a new compiled field program'
    # Include the root-only case and the exact requested-output boundary convention.
    axis = np.linspace(0, 1, 33)
    empty = P.prepare(basis, [], [], np.empty((36, 0)), axis[None], axis[None])
    root_c = jnp.array([.2, -.01, .02, .005])
    field = P.center(root_c, {'packed':empty})
    assert float(jnp.max(jnp.abs(field[[0, -1], :]))) == 0.
    assert float(jnp.max(jnp.abs(field[:, [0, -1]]))) == 0.
    print('PASS packed fields, JVP/VJP, fine advection, initial fits, CN states; geometry traces', len(traces),
          'worst returned relative change', worst_step_change, flush=True)


if __name__ == '__main__':
    main()
