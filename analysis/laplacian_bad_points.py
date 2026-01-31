from src import *
from utils import *
from scipy.spatial import cKDTree
import argparse

def main(args):
    #-- PARAMETERS --#

    N = args.N
    K = args.K
    l = args.l
    kappa = args.kappa
    delta = args.delta
    W = args.W
    W_grad = args.W_grad
    l_grad = args.l_grad

    if args.seed is not None:
        np.random.seed(args.seed)

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
    phi_range = (np.pi / 9,  8 * np.pi / 9)
    # phi_range = (0, np.pi)

    phi_min = phi_range[0]
    phi_max = phi_range[1]

    num_boundary = 2 * int(np.round(np.sqrt(2*r*9/(R*7))*np.sqrt(N))) # TODO: adaptive code
    # num_boundary = 2 * int(np.round(np.sqrt(2*r/R)*np.sqrt(N)))
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

    #-- MANUFACTURED SOLUTION --#

    u_sym = sp.sin(theta) * sp.cos(phi + sp.pi/4)

    u_lap_sym = manifold.get_laplacian(u_sym)
    u_grad_sym = manifold.get_gradient(u_sym)
    if args.screened:
        f_sym = u_sym - u_lap_sym
    else:
        f_sym = -u_lap_sym

    tt = manifold.params[:, 0]
    pp = manifold.params[:, 1]

    f_func = sp.lambdify((theta, phi), f_sym, 'numpy')
    f_vals = f_func(tt, pp)

    u_func = sp.lambdify((theta, phi), u_sym, 'numpy')
    u_vals = u_func(tt, pp)

    u_lap_func = sp.lambdify((theta, phi), u_lap_sym, 'numpy')
    u_lap_vals = u_lap_func(tt, pp)

    u_grad_func = sp.lambdify((theta, phi), u_grad_sym, 'numpy')
    u_grad_vals = u_grad_func(tt, pp) # shape: (n, 1, N)

    u_grad_vals_boundary = u_grad_vals.squeeze()[:, id_boundary].T # shape: (num_boundary, n)

    # outward normal at each boundary point
    n_vecs = np.zeros((num_boundary, manifold.n)) # shape: (num_boundary, n)

    n_vec_left =  - manifold.get_local_basis([0, np.pi/9])[0][1]
    n_vec_right = manifold.get_local_basis([0, 8 * np.pi/9])[0][1]

    for i in range(num_boundary):
        if i < num_boundary // 2:
            n_vecs[i, :] = n_vec_left
        else:
            n_vecs[i, :] = n_vec_right

    g_vals = u_vals[id_boundary] + np.sum(n_vecs * u_grad_vals_boundary, axis=1) # shape: (num_boundary)

    #-- OPERATORS --#

    L = np.zeros((num_interior, N))
    tree_full = cKDTree(manifold.points)

    positive_idx = []
    unstable_idx = []

    for i, i_id in enumerate(id_interior):
        _, stencil_ids = tree_full.query(manifold.points[i_id], K)

        weights_lap = get_operator_weights(
            stencil=manifold.points[stencil_ids],
            tangent_basis=manifold.get_local_basis(manifold.params[i])[0],
            operator='lap',
            kappa=kappa,
            l=l,
            delta=delta,
            weight_matrix=W
        ) # shape: (1, K)

        # detecting bad weights
        w_center = weights_lap[0, 0]
        w_neighbors = weights_lap[0, 1:]
        
        is_positive = w_center > 0.0
        
        # ratio = |w_center| / max(|w_neighbors|)
        ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
        
        is_unstable = ratio < 1.0

        if not args.qp:
            if is_positive:
                positive_idx.append(i)
            elif is_unstable:
                unstable_idx.append(i)

        if is_positive or is_unstable:
            # TODO: QP fix
            if args.qp:
                weights_lap = get_operator_weights(
                    stencil=manifold.points[stencil_ids],
                    tangent_basis=manifold.get_local_basis(manifold.params[i])[0],
                    operator='lap',
                    kappa=kappa,
                    l=l,
                    delta=delta,
                    weight_matrix=W,
                    qp=True
                )

                # detecting bad weights
                w_center = weights_lap[0, 0]
                w_neighbors = weights_lap[0, 1:]
                
                is_positive = w_center > 0.0
                
                # ratio = |w_center| / max(|w_neighbors|)
                ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
                
                is_unstable = ratio < 1.0

                if is_positive:
                    positive_idx.append(i)
                elif is_unstable:
                    unstable_idx.append(i)

        L[i, stencil_ids] = weights_lap[0, :]


    fe_pointwise = np.abs(L @ u_vals - u_lap_vals[id_interior]) # shape: (num_interior,)

    if args.l2:
        fe = np.sqrt(np.sum(fe_pointwise ** 2) / num_interior)
    else:
        fe = np.max(fe_pointwise)
    # st = np.linalg.norm(np.linalg.inv(A_prime), ord=np.inf)

    plot_error_distribution(manifold.params[id_interior], fe_pointwise, pos_ids=positive_idx, unstable_ids=unstable_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-N', type=int, default=6400
    )
    parser.add_argument(
        '-K', type=int, default=50
    )
    parser.add_argument(
        '-l', type=int, default=4
    )
    parser.add_argument(
        '--kappa', type=int, default=3
    )
    parser.add_argument(
        '--delta', type=float, default=1e-5
    )
    parser.add_argument(
        '-W', type=str, default='1/K'
    )
    parser.add_argument(
        '--l_grad', type=int, default=3
    )
    parser.add_argument(
        '--W_grad', type=str, default='1/K'
    )
    parser.add_argument(
        '--save', 
        action='store_true'
    )
    parser.add_argument(
        '--qp', 
        action='store_true'
    )
    parser.add_argument(
        '--l2', 
        action='store_true'
    )
    parser.add_argument(
        '--screened', 
        action='store_true'
    )
    parser.add_argument(
        '--seed', type=int, default=None
    )

    args = parser.parse_args()
    main(args)