from src import *
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigs
from scipy import sparse


def eig_semi_torus(N=6400, l=4, K=25, l_grad=3, K_grad=20, seed=None, num_eigs=30):
    #-- PARAMETERS --#

    kappa = 3
    delta = 1e-5

    if seed is not None:
        np.random.seed(seed)

    #-- GEOMETRY --#

    theta, phi = sp.symbols('theta phi', real=True)
    R = 2.0
    r = 1.0

    x_sym = (R + r * sp.cos(theta)) * sp.cos(phi)
    y_sym = (R + r * sp.cos(theta)) * sp.sin(phi)
    z_sym = r * sp.sin(theta)

    manifold = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold.compute()

    theta_range = (0, 2*np.pi)
    phi_range = (0, np.pi)

    phi_min = phi_range[0]
    phi_max = phi_range[1]

    num_boundary = 2 * int(np.round(np.sqrt(2*r/R)*np.sqrt(N)))
    num_interior = N - num_boundary

    manifold.sample([theta_range, phi_range], num_interior)

    # sample boundary

    x_sym_left = (R + r * sp.cos(theta)) * sp.cos(phi_min)
    y_sym_left = (R + r * sp.cos(theta)) * sp.sin(phi_min)

    boundary_left = Manifold([theta], [x_sym_left, y_sym_left, z_sym])
    boundary_left.sample([theta_range], num_boundary // 2)

    x_sym_right = (R + r * sp.cos(theta)) * sp.cos(phi_max)
    y_sym_right = (R + r * sp.cos(theta)) * sp.sin(phi_max)

    boundary_right = Manifold([theta], [x_sym_right, y_sym_right, z_sym])
    boundary_right.sample([theta_range], num_boundary // 2)

    manifold.params = np.vstack([
        manifold.params,
        np.insert(boundary_left.params, 1, values=phi_min, axis=1),
        np.insert(boundary_right.params, 1, values=phi_max, axis=1)
    ])

    manifold.points = np.vstack([
        manifold.points,
        boundary_left.points,
        boundary_right.points
    ])

    id_interior = np.arange(num_interior)
    id_boundary = np.arange(num_interior, N)

    # outward normals
    n_vecs = np.zeros((num_boundary, manifold.n)) # shape: (num_boundary, n)

    for i in range(num_boundary):
        n_vecs[i, :] = [0.0, -1.0, 0.0]

    #-- OPERATORS --#

    L = np.zeros((num_interior, N))
    tree_full = cKDTree(manifold.points)

    bad_count = 0
    for i, i_id in enumerate(id_interior):
        K_current = K
        max_K_retries = 15

        K_retries = 0

        best_ratio = -1.0
        best_weights = None
        best_stencil_ids = None

        while True:
            _, stencil_ids = tree_full.query(manifold.points[i_id], K_current)

            weights_lap = get_operator_weights(
                stencil=manifold.points[stencil_ids],
                tangent_basis=manifold.get_local_basis(manifold.params[i])[0],
                operator='lap',
                kappa=kappa,
                l=l,
                delta=delta,
            ) # shape: (1, K)

            if K_current == K:
                initial_weights = weights_lap.copy()
                initial_stencil_ids = stencil_ids.copy()

            w_center = weights_lap[0, 0]
            w_neighbors = weights_lap[0, 1:]
            
            is_positive = w_center > 0.0
            
            # ratio = |w_center| / max(|w_neighbors|)
            ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
            
            is_unstable = ratio < 3.0

            # record best
            if not is_positive:
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_weights = weights_lap.copy()
                    best_stencil_ids = stencil_ids.copy()

            if not is_positive and not is_unstable:
                break

            if K_retries < max_K_retries:
                K_current += 2
                K_retries += 1
            else:
                break

        if best_weights is None:
            weights_lap = initial_weights
            stencil_ids = initial_stencil_ids
        else:
            weights_lap = best_weights
            stencil_ids = best_stencil_ids

        # detecting bad weights
        w_center = weights_lap[0, 0]
        w_neighbors = weights_lap[0, 1:]
        
        is_positive = w_center > 0.0
        
        # ratio = |w_center| / max(|w_neighbors|)
        ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
        
        is_unstable = ratio < 1.0

        if is_positive or is_unstable:
            # qp processing
            K_current = K # fixed to initial K
            max_K_retries = 15

            K_retries = 0

            best_ratio = -1.0
            best_weights = None
            best_stencil_ids = None

            while True:
                _, stencil_ids = tree_full.query(manifold.points[i_id], K_current)

                weights_lap = get_operator_weights(
                    stencil=manifold.points[stencil_ids],
                    tangent_basis=manifold.get_local_basis(manifold.params[i])[0],
                    operator='lap',
                    kappa=kappa,
                    l=l,
                    delta=delta,
                    qp=True
                )

                if weights_lap is not None:
                    # detecting bad weights
                    w_center = weights_lap[0, 0]
                    w_neighbors = weights_lap[0, 1:]
                    
                    is_positive = w_center > 0.0
                    
                    # ratio = |w_center| / max(|w_neighbors|)
                    ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
                    
                    is_unstable = ratio < 1.0

                    if not is_positive:
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_weights = weights_lap.copy()
                            best_stencil_ids = stencil_ids.copy()

                    if not is_positive and not is_unstable:
                        break

                if K_retries < max_K_retries:
                    K_current += 2
                    K_retries += 1
                else:
                    break

            if best_weights is None:
                raise RuntimeError("lb operator sign error")
            else:
                weights_lap = best_weights
                stencil_ids = best_stencil_ids

        L[i, stencil_ids] = weights_lap[0, :]

    D_n = np.zeros((num_boundary, N))
    tree_interior = cKDTree(manifold.points[id_interior])

    for i, b_id in enumerate(id_boundary):
        K_current = K_grad
        max_K_retries = 15

        K_retries = 0

        best_ratio = -1.0
        best_weights = None
        best_stencil_ids = None
        while True:
            _, stencil_ids = tree_interior.query(manifold.points[b_id], K_current-1)

            # append boundary point
            stencil_points = np.vstack((manifold.points[b_id], manifold.points[stencil_ids]))
            stencil_ids = np.append(b_id, stencil_ids)

            weights_grad = get_operator_weights(
                stencil=stencil_points,
                tangent_basis=manifold.get_local_basis(manifold.params[b_id])[0],
                operator='grad',
                kappa=kappa,
                l=l_grad,
                delta=delta,
            ) # shape: (n, K)

            n_vec = n_vecs[i]
            weights_grad_n = n_vec @ weights_grad # shape: (K,)

            if K_current == K_grad:
                initial_weights = weights_grad_n.copy()
                initial_stencil_ids = stencil_ids.copy()

            w_center = weights_grad_n[0]
            w_neighbors = weights_grad_n[1:]
            
            is_positive = w_center > 0.0
            
            # ratio = |w_center| / max(|w_neighbors|)
            ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
            
            is_unstable = ratio < 3.0

            if is_positive:
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_weights = weights_grad_n.copy()
                    best_stencil_ids = stencil_ids.copy()

            if is_positive and not is_unstable: # for gradient, we need positive
                break

            if K_retries < max_K_retries:
                K_current += 2
                K_retries += 1
            else:
                break
        
        if best_weights is None:
            raise RuntimeError("gradient operator sign error")
        else:
            weights_grad_n = best_weights
            stencil_ids = best_stencil_ids

        D_n[i, stencil_ids] = weights_grad_n

    #-- SYSTEM PARTITION --#
    A = -L
    A_II = A[:, id_interior]
    A_IB = A[:, id_boundary]

    B_BI = D_n[:, id_interior] # interior points do not affect the Dirichlet part
    B_BB = D_n[:, id_boundary] + np.eye(num_boundary) 

    # Schur complement
    B_BB_inv = np.linalg.inv(B_BB)

    A_prime = A_II - A_IB @ B_BB_inv @ B_BI

    #-- EIGENVALUES --#
    evals, evecs = eigs(sparse.csr_matrix(A_prime), k=num_eigs, which='SM')
    return evals, evecs