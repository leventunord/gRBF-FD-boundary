from itertools import combinations_with_replacement
import numpy as np
from cvxopt import matrix, solvers
import cvxopt

solvers.options['show_progress'] = False

#-- Helper Functions -- #

def get_polynomial_basis(degree, dim):
    """
    Generates multi-indices for a complete polynomial basis in 'dim' dimensions.
    Example: 
        - degree=2, dim=2 -> [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
        - degree=2, dim=3 -> [[0,0,0], [1,0,0], ..., [0,1,1], ..., [0,0,2]]
    """
    indices = []
    for d in range(degree + 1):
        for p in combinations_with_replacement(range(dim), d):
            index = np.zeros(dim, dtype=int)
            for i in p:
                index[i] += 1
            indices.append(index)
    return indices

def get_operator_weights(stencil, tangent_basis, kappa=3, l=4, delta=1e-8, qp=False, operator='lap', weight_matrix='1/K'):
    """
    Args:
        stencil (ndarray): shape (K, n)
        tangent_basis (ndarray): shape (d, n)
    """
    K, n = stencil.shape
    d = tangent_basis.shape[0]


    stencil_center = stencil[0]
    local_coords = (stencil - stencil_center) @ tangent_basis.T # shape: (K, d)

    # normalization
    diameter = np.sqrt(np.max(np.sum((local_coords[:, None, :] - local_coords[None, :, :]) ** 2, axis=-1)))

    # TODO: diameter might be very small
    norm_coords = local_coords / diameter

    poly_basis = get_polynomial_basis(l, d)
    m = len(poly_basis)

    P = np.zeros((K, m))
    for j, alpha in enumerate(poly_basis):
        P[:, j] = np.prod(norm_coords**alpha, axis=1)

    if qp:
        # quadratic programming
        b_eq = np.zeros(m)
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                b_eq[j] = 2.0

        H_diag = np.ones(K + 1)
        H_diag[0] = 1.0 / (K**2) # center point weight
        H_diag[-1] = (100 * K)**2 # slack variable 

        # QP matrices
        H = matrix(np.diag(H_diag))
        q = matrix(np.zeros(K + 1))

        # constrains
        G_np = np.zeros((K + 2, K + 1))
        h_np = np.zeros(K + 2)
        
        row_idx = 0

        # Center sign: w_0 <= 0
        G_np[row_idx, 0] = 1.0 
        row_idx += 1
        
        # Neighbor sign: -w_j - C <= 0 (for j=1..K-1)
        for j in range(1, K):
            G_np[row_idx, j] = -1.0
            G_np[row_idx, -1] = -1.0
            row_idx += 1
            
        # Slack variable sign: -C <= 0
        G_np[row_idx, -1] = -1.0
        row_idx += 1

        # Slack variable upper bound
        G_np[row_idx, -1] = 1.0
        h_np[row_idx] = 10**5
        row_idx += 1
        
        G = matrix(G_np)
        h = matrix(h_np)

        A_np = np.zeros((m, K + 1))
        A_np[:, :K] = P.T # 多项式一致性仅约束权重 w
        
        A = matrix(A_np)
        b = matrix(b_eq)

        # try:
        sol = solvers.qp(H, q, G, h, A, b)
        x_sol = np.array(sol['x']) # shape: (K + 1, 1)
        
        weights = x_sol[:K, :].T # shape: (1, K)
        
        weights = weights / (diameter**2)
        
        return weights
            
        # except ValueError:
        #     # 如果 QP 无解（通常是因为 K 太小，几何太差），返回 None
        #     # 外部循环应该捕获这个 None 并增加 K 重试
        #     return None

    dist = np.sqrt(np.sum((norm_coords[:, None, :] - norm_coords[None, :, :]) ** 2, axis=-1))
    Phi = dist ** (2 * kappa + 1)

    if weight_matrix == '1/K':
        W = np.diag([1.0] + [(1.0 / K)] * (K - 1))
    elif weight_matrix == 'uniform':
        W = np.eye(K)

    r = np.linalg.norm(norm_coords, axis=1) # shape: (K,)

    if operator == 'lap':
        dP = np.zeros((1, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                dP[0, j] = 2.0
        
        dPhi = (4 * kappa**2 + 2 * d * kappa + d - 1) * r**(2 * kappa - 1)
        dPhi = dPhi.reshape(1, -1) # shape: (1, K)
    elif operator == 'grad':
        dP = np.zeros((d, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 1:
                dim_idx = np.where(np.array(alpha) == 1)[0][0]
                dP[dim_idx, j] = 1.0

        coeffs = -(2 * kappa + 1) * r**(2 * kappa - 1)
        dPhi = (coeffs.reshape(-1, 1) * norm_coords).T # shape: (d, K)

    if delta is None:
        Phi_inv = np.linalg.inv(Phi)
    else:
        reg_term = (delta**2) * np.eye(K)
        inv_term = np.linalg.inv(Phi.T @ W @ Phi + reg_term)
        Phi_inv = inv_term @ Phi.T @ W

    P_t_W_P = P.T @ W @ P
    P_t_W_P_inv = np.linalg.inv(P_t_W_P)
    P_inv_block = P_t_W_P_inv @ P.T @ W

    w_poly = dP @ P_inv_block
    
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    if operator == 'lap':
        weights = (w_rbf + w_poly) / (diameter**2) # shape: (1, K)
    elif operator == 'grad':
        weights_local = (w_rbf + w_poly) / diameter # shape: (d, K)
        weights = tangent_basis.T @ weights_local # shape: (n, K)

    return weights
