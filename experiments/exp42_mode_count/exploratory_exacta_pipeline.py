"""Corrected exacta->rank pipeline: lam-bar = p-hat - delta with the exact
first-order relation delta = -D_full K substituted, so the winner directions
re-enter the design instead of being consumed by the lam-bar estimate."""
import numpy as np

def design_rows(lb, N, A):
    Lb = lb[A].sum(); c = lb[A]/Lb
    rows = []
    for pos in range(len(A)):
        g = np.zeros((N, N))
        for j in A: g[j, A[pos]] += 1.0
        for j in A:
            for k in A: g[j, k] -= c[pos]
        rows.append(-(1.0/Lb)*g.ravel())
    return np.array(rows), c

def pipeline_sv(P_hat, N):
    p = P_hat.sum(1); M = P_hat/np.maximum(p[:, None], 1e-300)
    lb = p/p.sum()
    D_full, _ = design_rows(lb, N, list(range(N)))          # delta = -D_full K
    rows_d, rhs = [], []
    for i in range(N):
        rest = [a for a in range(N) if a != i]
        q = p + p[i]*M[i]
        D_rest, c = design_rows(lb, N, rest)
        # linearize c^(-i)(p - delta) in delta, delta = -D_full K:
        # dc_j/ddelta_l = -1_{l=j}/(1-p_i) + p_j 1_{l=i}/(1-p_i)^2
        Jc = np.zeros((len(rest), N))
        for pos, j in enumerate(rest):
            Jc[pos, j] = -1.0/(1-p[i])
            Jc[pos, i] = p[j]/(1-p[i])**2
        D_eff = D_rest - Jc @ D_full
        rows_d.append(D_eff); rhs.append(q[rest] - c)
    Dm = np.vstack(rows_d); y = np.concatenate(rhs)
    Bf = [np.outer(np.eye(N)[a], lb).ravel() for a in range(N)] + [np.diag(lb).ravel()]
    Q, _ = np.linalg.qr(np.array(Bf).T)
    P_id = np.eye(N*N) - Q @ Q.T
    Did = Dm @ P_id
    sol, *_ = np.linalg.lstsq(Did, y, rcond=None)
    K_id = (P_id @ sol).reshape(N, N)
    return K_id, lb, np.linalg.matrix_rank(Did, tol=1e-9)

def read_sv(K_id, lb, N, rho):
    A_fam = np.column_stack([np.kron(np.eye(N)[a], lb) for a in range(N)]
                            + [np.diag(lb).ravel()])
    x = np.zeros(N+1)
    for _ in range(400):
        Mm = K_id + (A_fam @ x).reshape(N, N)
        U, s, Vt = np.linalg.svd(Mm)
        X = (U[:, :rho]*s[:rho]) @ Vt[:rho]
        x, *_ = np.linalg.lstsq(A_fam, (X - K_id).ravel(), rcond=None)
    Mhat = K_id + (A_fam @ x).reshape(N, N)
    cols = [lb[k]*Mhat[:, j] - lb[j]*Mhat[:, k]
            for j in range(N) for k in range(j+1, N)]
    return np.linalg.svd(np.column_stack(cols), compute_uv=False)
