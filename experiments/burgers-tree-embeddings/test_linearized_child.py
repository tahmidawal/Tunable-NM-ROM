"""Independent weighted field least-squares and frozen-head derivative fixtures."""
from common import *
import decoder, shared_child_head as H, linearized_child as T


def main():
    root, channels, latent = 2, 4, 2
    cells = [[1, 0, 0], [1, 1, 0], [2, 0, 0]]
    rng = np.random.default_rng(7)
    params = dict(linear=jnp.asarray(rng.normal(size=(channels, latent))),
                  net=decoder.init_mlp(jax.random.PRNGKey(9), [latent+3, 12, channels], .1))
    scale = jnp.array([.3, .6, 1.1, .7])
    J = T.zero_map(params, scale, cells)
    g, h = H.geometry(cells)
    direction = jnp.asarray(rng.normal(size=(len(cells), latent)))
    epsilon = 1e-5
    difference = (H.head(params, epsilon*direction, g, h, scale)-H.head(params, -epsilon*direction, g, h, scale))/(2*epsilon)
    assert float(jnp.max(jnp.abs(difference-jnp.einsum('prk,pk->pr', J, direction)))) < 1e-9
    assert float(jnp.max(jnp.abs(H.head(params, jnp.zeros_like(direction), g, h, scale)))) == 0.
    # A nonorthogonal sampled field and nonuniform positive physical weights
    # independently test root refitting and every overlap cross term.
    features = rng.normal(size=(80, root+len(cells)*channels))
    weights = rng.uniform(.1, 1.9, size=80)
    fields = rng.normal(size=(5, 80))
    weighted = features*np.sqrt(weights[:, None])
    truth = fields*np.sqrt(weights[None, :])
    gram = weighted.T@weighted
    rhs = truth@weighted
    norm2 = np.sum(truth**2, axis=1)
    A = np.zeros((features.shape[1], root+len(cells)*latent))
    A[:root, :root] = np.eye(root)
    for p in range(len(cells)):
        A[root+p*channels:root+(p+1)*channels, root+p*latent:root+(p+1)*latent] = np.asarray(J[p])
    independent = np.linalg.lstsq(weighted@A, truth.T, rcond=None)[0].T@A.T
    result = T.fit(jnp.asarray(gram), jnp.asarray(rhs), jnp.asarray(norm2), J, root=root)
    actual = np.asarray(result['coefficients'])
    assert bool(result['finite']) and float(result['normalized_gram_eigenvalues'][0]) > 0.
    assert np.max(np.abs(actual-independent)) < 1e-12
    physical = np.linalg.norm(actual@weighted.T-truth, axis=1)/np.sqrt(norm2)
    assert np.max(np.abs(physical-np.asarray(result['error']))) < 1e-12
    assert float(jnp.max(result['stationarity'])) < 1e-11
    print('PASS frozen-head tangent derivative and joint weighted field projection; max stationarity',
          float(jnp.max(result['stationarity'])), flush=True)


if __name__ == '__main__':
    main()
