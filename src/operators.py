from itertools import combinations_with_replacement
import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import linprog

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
        H_diag = np.ones(K + 1)
        H_diag[0] = 1.0 / (K**2) # center point weight
        H_diag[-1] = K**2 # slack variable 

        ### objective
        H = np.diag(H_diag)
        f = np.zeros(K + 1)

        ### inequality contraints
        A = np.zeros((K + 2, K + 1))
        b = np.zeros(K + 2)

        row_id = 0

        # center: # w_0 + C <= 0
        A[row_id, 0] = 1.0
        A[row_id, -1] = 1.0 

        row_id += 1
        
        # neighbors: -w_i - C <= 0
        for i in range(1, K):
            A[row_id, i] = -1.0
            A[row_id, -1] = -1.0
            row_id += 1

        # slack: -C <= 0
        A[row_id, -1] = -1.0

        row_id += 1

        # slack upper bound: C <= 10^5
        A[row_id, -1] = 1.0
        b[row_id] = 10**5

        ### equality contraints
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T

        b_eq = np.zeros(m)
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                b_eq[j] = 2.0 / (diameter**2) # normalized here

        # try:
        sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        if sol['status'] != 'optimal':
            return None

        x_sol = np.array(sol['x']) # shape: (K + 1, 1)
        weights = x_sol[:K, :].T # shape: (1, K)
        
        return weights
            
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
        Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P

    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    w_poly = dP @ P_inv_block
    
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    if operator == 'lap':
        weights = (w_rbf + w_poly) / (diameter**2) # shape: (1, K)
    elif operator == 'grad':
        weights_local = (w_rbf + w_poly) / diameter # shape: (d, K)
        weights = tangent_basis.T @ weights_local # shape: (n, K)

    return weights

def get_operator_weights_v4(stencil, tangent_basis, kappa=3, l=4, delta=1e-8, opt=None, operator='lap', weight_matrix='1/K'):
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

    if opt == 'qp':
        H_diag = np.ones(K + 1)
        H_diag[0] = 1.0 / (K**2) # center point weight
        H_diag[-1] = K**2 # slack variable 

        ### objective
        H = np.diag(H_diag)
        f = np.zeros(K + 1)

        ### inequality constraints
        num_ineq = 1 + 2 * (K - 1) + 2
        A = np.zeros((num_ineq, K + 1))
        b = np.zeros(num_ineq)

        gamma_target = 3.0 
        row_id = 0

        # 1. center: w_0 + gamma_target * C <= 0
        A[row_id, 0] = 1.0
        A[row_id, -1] = gamma_target 
        row_id += 1
        
        # 2. neighbors: |w_i| <= C 
        for i in range(1, K):
            # w_i - C <= 0
            A[row_id, i] = 1.0
            A[row_id, -1] = -1.0
            row_id += 1
            
            # -w_i - C <= 0
            A[row_id, i] = -1.0
            A[row_id, -1] = -1.0
            row_id += 1

        # 3. slack lower bound: -C <= 0
        A[row_id, -1] = -1.0
        row_id += 1

        # 4. slack upper bound: C <= 10^5
        A[row_id, -1] = 1.0
        b[row_id] = 10**5

        ### equality constraints
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T

        b_eq = np.zeros(m)
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                b_eq[j] = 2.0

        # try:
        sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        if sol['status'] != 'optimal':
            return None

        x_sol = np.array(sol['x']) # shape: (K + 1, 1)
        weights = x_sol[:K, :].T # shape: (1, K)
        
        return weights / (diameter**2)
    elif opt == 'lp':
        c_lp = np.zeros(K + 1)
        c_lp[-1] = 1.0

        ### inequality constraints
        num_ineq = 1 + 2 * (K - 1) + 2
        A = np.zeros((num_ineq, K + 1))
        b = np.zeros(num_ineq)

        gamma_target = 3.0 
        row_id = 0

        # 1. center: w_0 + gamma_target * C <= 0
        A[row_id, 0] = 1.0
        A[row_id, -1] = gamma_target 
        row_id += 1
        
        # 2. neighbors: |w_i| <= C 
        for i in range(1, K):
            # w_i - C <= 0
            A[row_id, i] = 1.0
            A[row_id, -1] = -1.0
            row_id += 1
            
            # -w_i - C <= 0
            A[row_id, i] = -1.0
            A[row_id, -1] = -1.0
            row_id += 1

        # 3. slack lower bound: -C <= 0
        A[row_id, -1] = -1.0
        row_id += 1

        # 4. slack upper bound: C <= 10^5
        A[row_id, -1] = 1.0
        b[row_id] = 10**5

        ### equality constraints
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T

        b_eq = np.zeros(m)
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                b_eq[j] = 2.0

        res_lp = linprog(c_lp, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')

        if not res_lp.success:
            return None
        
        weights = res_lp.x[:-1].reshape(1, -1)
        return weights / (diameter**2)
            
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
        Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P

    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    w_poly = dP @ P_inv_block
    
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    if operator == 'lap':
        weights = (w_rbf + w_poly) / (diameter**2) # shape: (1, K)
    elif operator == 'grad':
        weights_local = (w_rbf + w_poly) / diameter # shape: (d, K)
        weights = tangent_basis.T @ weights_local # shape: (n, K)

    return weights

def get_operator_weights_v2(stencil, tangent_basis, kappa=3, l=4, delta=1e-8, qp=False, operator='lap', weight_matrix='1/K'):
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
        H_diag = np.ones(K + 1)
        H_diag[0] = 1.0 / (K**2) # center point weight
        H_diag[-1] = K**2 # slack variable 

        ### objective
        H = np.diag(H_diag)
        f = np.zeros(K + 1)

        ### inequality contraints
        A = np.zeros((K + 2, K + 1))
        b = np.zeros(K + 2)

        row_id = 0

        # center: # w_0 + C <= 0
        A[row_id, 0] = 1.0
        A[row_id, -1] = 1.0 

        row_id += 1
        
        # neighbors: -w_i - C <= 0
        for i in range(1, K):
            A[row_id, i] = -1.0
            A[row_id, -1] = -1.0
            row_id += 1

        # slack: -C <= 0
        A[row_id, -1] = -1.0

        row_id += 1

        # slack upper bound: C <= 10^5
        A[row_id, -1] = 1.0
        b[row_id] = 10**5

        ### equality contraints
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T

        b_eq = np.zeros(m)
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                b_eq[j] = 2.0

        # try:
        sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        if sol['status'] != 'optimal':
            return None

        x_sol = np.array(sol['x']) # shape: (K + 1, 1)
        weights = x_sol[:K, :].T # shape: (1, K)
        
        w_poly = weights
            
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
        Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P

    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    if not qp:
        w_poly = dP @ P_inv_block
    
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    if operator == 'lap':
        weights = (w_rbf + w_poly) / (diameter**2) # shape: (1, K)
    elif operator == 'grad':
        weights_local = (w_rbf + w_poly) / diameter # shape: (d, K)
        weights = tangent_basis.T @ weights_local # shape: (n, K)

    return weights

def get_operator_weights_v3(stencil, tangent_basis, kappa=3, l=4, delta=1e-8, qp=False, operator='lap', weight_matrix='1/K'):
    K, n = stencil.shape
    d = tangent_basis.shape[0]

    stencil_center = stencil[0]
    local_coords = (stencil - stencil_center) @ tangent_basis.T # shape: (K, d)

    # normalization
    diameter = np.sqrt(np.max(np.sum((local_coords[:, None, :] - local_coords[None, :, :]) ** 2, axis=-1)))
    norm_coords = local_coords / diameter

    poly_basis = get_polynomial_basis(l, d)
    m = len(poly_basis)

    P = np.zeros((K, m))
    for j, alpha in enumerate(poly_basis):
        P[:, j] = np.prod(norm_coords**alpha, axis=1)

    # ==========================================
    # 1. 优先计算无约束的理想 RBF+多项式 权重 (归一化空间)
    # ==========================================
    dist = np.sqrt(np.sum((norm_coords[:, None, :] - norm_coords[None, :, :]) ** 2, axis=-1))
    Phi = dist ** (2 * kappa + 1)

    if weight_matrix == '1/K':
        W = np.diag([1.0] + [(1.0 / K)] * (K - 1))
    elif weight_matrix == 'uniform':
        W = np.eye(K)

    r = np.linalg.norm(norm_coords, axis=1)

    if operator == 'lap':
        dP = np.zeros((1, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                dP[0, j] = 2.0
        
        dPhi = (4 * kappa**2 + 2 * d * kappa + d - 1) * r**(2 * kappa - 1)
        dPhi = dPhi.reshape(1, -1)
    elif operator == 'grad':
        dP = np.zeros((d, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 1:
                dim_idx = np.where(np.array(alpha) == 1)[0][0]
                dP[dim_idx, j] = 1.0
        coeffs = -(2 * kappa + 1) * r**(2 * kappa - 1)
        dPhi = (coeffs.reshape(-1, 1) * norm_coords).T

    if delta is None:
        Phi_inv = np.linalg.inv(Phi)
    else:
        reg_term = (delta**2) * np.eye(K)
        Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P
    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    w_poly = dP @ P_inv_block
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    # 包含多项式和 RBF 残差的归一化理想权重
    w_ideal_norm = (w_rbf + w_poly).flatten() 

    # ==========================================
    # 2. 将理想权重混入 QP 优化条件中
    # ==========================================
    if qp:
        # 修改 H: 权重部分的惩罚为 1.0 (对应 ||w - w_ideal||^2)
        H_diag = np.ones(K + 1)
        H_diag[-1] = K**2 # slack variable 惩罚保持不变

        H = np.diag(H_diag)
        
        # 修改 f: 引入 -w_ideal 使得二次型中心偏移到 w_ideal
        f = np.zeros(K + 1)
        f[:K] = -w_ideal_norm 

        # 你的不等式约束 A, b 保持完全不变
        A = np.zeros((K + 2, K + 1))
        b = np.zeros(K + 2)
        row_id = 0
        A[row_id, 0] = 1.0; A[row_id, -1] = 1.0; row_id += 1
        for i in range(1, K):
            A[row_id, i] = -1.0; A[row_id, -1] = -1.0; row_id += 1
        A[row_id, -1] = -1.0; row_id += 1
        A[row_id, -1] = 1.0; b[row_id] = 10**5

        # 等式约束：直接使用归一化的 dP
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T
        b_eq = dP.flatten() # 这里不再除以 diameter**2

        sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        if sol['status'] != 'optimal':
            return None

        x_sol = np.array(sol['x']) 
        weights_norm = x_sol[:K, :].T # shape: (1, K)
    else:
        weights_norm = w_ideal_norm.reshape(1, K)

    # ==========================================
    # 3. 统一解除归一化，恢复物理尺度
    # ==========================================
    if operator == 'lap':
        weights = weights_norm / (diameter**2)
    elif operator == 'grad':
        weights_local = weights_norm / diameter
        weights = tangent_basis.T @ weights_local 

    return weights

def get_operator_weights_test(stencil, tangent_basis, kappa=3, l=4, delta=1e-8, qp=False, operator='lap', weight_matrix='1/K'):
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

    if qp:
        # 1. 设置目标函数：最小化 RBF 逼近误差
        # H = Phi^T @ W @ Phi + 正则化项
        lambda_reg = 0

        H_mat = Phi.T @ W @ Phi + (delta**2 + lambda_reg) * np.eye(K)
        # f = - (dPhi @ W @ Phi)^T
        f_vec = -(dPhi @ W @ Phi).flatten()

        # 2. 不等式约束: 严格保证 w_0 < 0 且 |w_i| <= |w_0| / gamma
        gamma = 1.5  # 设定的比率阈值
        num_ineq = 1 + 2 * (K - 1)
        A = np.zeros((num_ineq, K))
        b = np.zeros(num_ineq)

        row_id = 0
        # 确保中心点必须是负的 (Laplacian 半负定性质)
        A[row_id, 0] = 1.0
        # b[row_id] = -1e-4 
        b[row_id] = 0.0
        row_id += 1

        for i in range(1, K):
            # 约束1: w_0 / gamma + w_i <= 0  (即 w_i <= -w_0 / gamma)
            A[row_id, 0] = 1.0 / gamma
            A[row_id, i] = 1.0
            row_id += 1
            
            # 约束2: w_0 / gamma - w_i <= 0  (即 -w_i <= -w_0 / gamma)
            A[row_id, 0] = 1.0 / gamma
            A[row_id, i] = -1.0
            row_id += 1

        # 3. 等式约束: 多项式必须精确拟合
        A_eq = np.zeros((m, K))
        A_eq[:, :K] = P.T

        b_eq = np.zeros(m)
        if operator == 'lap':
            for j, alpha in enumerate(poly_basis):
                if sum(alpha) == 2 and max(alpha) == 2:
                    b_eq[j] = 2.0 / (diameter**2) # 注意这里进行归一化缩放

        sol = solvers.qp(matrix(H_mat), matrix(f_vec), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        
        if sol['status'] != 'optimal':
            return None

        x_sol = np.array(sol['x']) 
        weights = x_sol.T # shape: (1, K)
        
        return weights

    if delta is None:
        Phi_inv = np.linalg.inv(Phi)
    else:
        reg_term = (delta**2) * np.eye(K)
        Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P

    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    w_poly = dP @ P_inv_block
    
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector

    if operator == 'lap':
        weights = (w_rbf + w_poly) / (diameter**2) # shape: (1, K)
    elif operator == 'grad':
        weights_local = (w_rbf + w_poly) / diameter # shape: (d, K)
        weights = tangent_basis.T @ weights_local # shape: (n, K)

    return weights
