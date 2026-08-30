import numpy as np


def ops(n):
    """Standard square MAC grid with n cells per direction.

    Unknowns are u(i,j), i=1..n-1, j=0..n-1 and
    v(i,j), i=0..n-1, j=1..n-1. Boundary-normal velocities are
    homogeneous and eliminated. Pressure is at all n*n cell centers.
    """
    h = 1.0 / n
    u_ids = {(i, j): q for q, (i, j) in enumerate(
        ( (i, j) for i in range(1, n) for j in range(n) ))}
    off = len(u_ids)
    v_ids = {(i, j): off + q for q, (i, j) in enumerate(
        ( (i, j) for i in range(n) for j in range(1, n) ))}
    p_ids = {(i, j): i * n + j for i in range(n) for j in range(n)}
    nv = len(u_ids) + len(v_ids)

    D = np.zeros((n*n, nv))
    for i in range(n):
        for j in range(n):
            q = p_ids[i, j]
            if i + 1 < n:
                D[q, u_ids[i+1, j]] += 1/h
            if i > 0:
                D[q, u_ids[i, j]] -= 1/h
            if j + 1 < n:
                D[q, v_ids[i, j+1]] += 1/h
            if j > 0:
                D[q, v_ids[i, j]] -= 1/h

    Grad = np.zeros((nv, n*n))
    for (i, j), q in u_ids.items():
        Grad[q, p_ids[i, j]] += 1/h
        Grad[q, p_ids[i-1, j]] -= 1/h
    for (i, j), q in v_ids.items():
        Grad[q, p_ids[i, j]] += 1/h
        Grad[q, p_ids[i, j-1]] -= 1/h

    # Vertex streamfunction, zero on boundary; columns are interior vertices.
    psi_ids = {(i, j): q for q, (i, j) in enumerate(
        ( (i, j) for i in range(1, n) for j in range(1, n) ))}
    C = np.zeros((nv, len(psi_ids)))
    for (i, j), q in u_ids.items():
        if (i, j+1) in psi_ids:
            C[q, psi_ids[i, j+1]] += 1/h
        if (i, j) in psi_ids:
            C[q, psi_ids[i, j]] -= 1/h
    for (i, j), q in v_ids.items():
        if (i+1, j) in psi_ids:
            C[q, psi_ids[i+1, j]] -= 1/h
        if (i, j) in psi_ids:
            C[q, psi_ids[i, j]] += 1/h

    # Componentwise vector Laplacian. Normal-direction boundary values are
    # prescribed zero; tangential directions use odd ghosts to impose the
    # wall value zero at the half-grid boundary.
    L = np.zeros((nv, nv))
    for (i, j), q in u_ids.items():
        # x: u lies on x-grid lines, with u=0 at i=0,n.
        L[q, q] -= 2/h**2
        if i - 1 >= 1:
            L[q, u_ids[i-1, j]] += 1/h**2
        if i + 1 <= n - 1:
            L[q, u_ids[i+1, j]] += 1/h**2
        # y: cell-centered tangent component, odd wall ghosts.
        if j == 0:
            L[q, q] -= 3/h**2
            L[q, u_ids[i, j+1]] += 1/h**2
        elif j == n-1:
            L[q, q] -= 3/h**2
            L[q, u_ids[i, j-1]] += 1/h**2
        else:
            L[q, q] -= 2/h**2
            L[q, u_ids[i, j-1]] += 1/h**2
            L[q, u_ids[i, j+1]] += 1/h**2
    for (i, j), q in v_ids.items():
        # x: cell-centered tangent component, odd wall ghosts.
        if i == 0:
            L[q, q] -= 3/h**2
            L[q, v_ids[i+1, j]] += 1/h**2
        elif i == n-1:
            L[q, q] -= 3/h**2
            L[q, v_ids[i-1, j]] += 1/h**2
        else:
            L[q, q] -= 2/h**2
            L[q, v_ids[i-1, j]] += 1/h**2
            L[q, v_ids[i+1, j]] += 1/h**2
        # y: v lies on y-grid lines, with v=0 at j=0,n.
        L[q, q] -= 2/h**2
        if j - 1 >= 1:
            L[q, v_ids[i, j-1]] += 1/h**2
        if j + 1 <= n - 1:
            L[q, v_ids[i, j+1]] += 1/h**2
    return h, u_ids, v_ids, p_ids, psi_ids, D, Grad, C, L


def sine_mode(n, psi_ids, k, ell):
    a = np.zeros(len(psi_ids))
    for (i, j), q in psi_ids.items():
        a[q] = np.sin(k*np.pi*i/n) * np.sin(ell*np.pi*j/n)
    return a


def report(n):
    h, uids, vids, pids, psids, D, Grad, C, L = ops(n)
    rng = np.random.default_rng(20260830 + n)
    adj = np.linalg.norm(D + Grad.T, ord=np.inf)
    dc = np.linalg.norm(D @ C, ord=np.inf)
    rank_d = np.linalg.matrix_rank(D)
    rank_c = np.linalg.matrix_rank(C)

    # Low square set of curl-sine modes.
    qmax = min(4, n-1)
    modes, lambdas = [], []
    for k in range(1, qmax+1):
        for ell in range(1, qmax+1):
            modes.append(C @ sine_mode(n, psids, k, ell))
            lambdas.append(4/h**2 * (np.sin(k*np.pi/(2*n))**2
                                     + np.sin(ell*np.pi/(2*n))**2))
    Phi = np.column_stack(modes)
    lambdas = np.asarray(lambdas)

    p = rng.standard_normal(n*n)
    pressure_abs = np.linalg.norm(Phi.T @ (Grad @ p))
    pressure_scale = np.linalg.norm(Phi) * np.linalg.norm(Grad @ p)
    div_scale = np.linalg.norm(D) * np.linalg.norm(Phi)

    eigen_rels = []
    for col, lam in zip(Phi.T, lambdas):
        eigen_rels.append(np.linalg.norm(L @ col + lam*col)
                          / np.linalg.norm(L @ col))

    # A smooth no-slip, divergence-free synthetic bank generated from clamped
    # streamfunctions sin^2(a*pi*x) sin^2(b*pi*y).
    bank = []
    for a, b in [(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)]:
        coeff = np.zeros(len(psids))
        for (i,j), z in psids.items():
            coeff[z] = np.sin(a*np.pi*i/n)**2 * np.sin(b*np.pi*j/n)**2
        bank.append(C @ coeff)
    G = np.column_stack(bank)
    G, _ = np.linalg.qr(G)
    B = Phi.T @ G
    A = Phi.T @ L @ G
    diag_defect = np.linalg.norm(A + lambdas[:,None]*B) / np.linalg.norm(A)

    # Boundary-adjacent tangential rows and arbitrary row visibility.
    bnd_rows = [q for (i,j),q in uids.items() if j in (0,n-1)]
    bnd_rows += [q for (i,j),q in vids.items() if i in (0,n-1)]
    lev = np.linalg.norm(Phi, axis=1)
    bnd_lev = lev[bnd_rows]
    # Unit row injections, normalized projection by ||Phi||_2.
    spectral = np.linalg.norm(Phi, 2)
    row_detect = bnd_lev / spectral

    print(f"n={n} cells, nv={D.shape[1]}, np={D.shape[0]}, npsi={C.shape[1]}")
    print(f"  ||D+Grad^T||_inf={adj:.3e}; ||D C||_inf={dc:.3e}")
    print(f"  rank(D)={rank_d} (expected {n*n-1}); dim ker(D)={D.shape[1]-rank_d}; rank(C)={rank_c}")
    print(f"  pressure projection abs={pressure_abs:.3e}; normalized={pressure_abs/pressure_scale:.3e}")
    print(f"  ||D Phi||/(||D|| ||Phi||)={np.linalg.norm(D@Phi)/div_scale:.3e}")
    print(f"  curl-sine eigen residual min/median/max={np.min(eigen_rels):.3e}/{np.median(eigen_rels):.3e}/{np.max(eigen_rels):.3e}")
    print(f"  bank-dependent ||A+Lambda B||/||A|| (smooth clamped bank)={diag_defect:.3e}")
    print(f"  boundary row leverage min/median/max={bnd_lev.min():.3e}/{np.median(bnd_lev):.3e}/{bnd_lev.max():.3e}")
    print(f"  normalized unit-boundary-row detect min/median/max={row_detect.min():.3e}/{np.median(row_detect):.3e}/{row_detect.max():.3e}")
    print(f"  all-row zero count={np.sum(lev == 0)}; boundary-row zero count={np.sum(bnd_lev == 0)}")


for n in (4, 8, 16, 32):
    report(n)
