from src import *
from scipy.spatial import cKDTree
import argparse
import pickle
import datetime

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

    # np.random.seed(0)

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

    num_boundary = 2 * int(np.round(np.sqrt(2*r/R)*np.sqrt(N)))
    num_interior = N - num_boundary

    manifold.sample([theta_range, phi_range], num_interior)

    # sample boundary

    x_sym_left = (R + r * sp.cos(theta)) * sp.cos(0)
    y_sym_left = (R + r * sp.cos(theta)) * sp.sin(0)

    boundary_left = Manifold([theta], [x_sym_left, y_sym_left, z_sym])
    boundary_left.sample([theta_range], num_boundary // 2)

    x_sym_right = (R + r * sp.cos(theta)) * sp.cos(sp.pi)
    y_sym_right = (R + r * sp.cos(theta)) * sp.sin(sp.pi)

    boundary_right = Manifold([theta], [x_sym_right, y_sym_right, z_sym])
    boundary_right.sample([theta_range], num_boundary // 2)

    manifold.params = np.vstack([
        manifold.params,
        np.insert(boundary_left.params, 1, values=0.0, axis=1),
        np.insert(boundary_right.params, 1, values=np.pi, axis=1)
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

    for i in range(num_boundary):
        n_vecs[i, :] = [0.0, -1.0, 0.0]

    g_vals = u_vals[id_boundary] + np.sum(n_vecs * u_grad_vals_boundary, axis=1) # shape: (num_boundary)

    #-- OPERATORS --#

    L = np.zeros((num_interior, N))
    tree_full = cKDTree(manifold.points)

    for i, i_id in enumerate(id_interior):
        _, stencil_ids = tree_full.query(manifold.points[i_id], K)

        weights_lap = get_operator_weights(
            stencil=manifold.points[stencil_ids],
            tangent_basis=manifold.get_local_basis(manifold.params[i])[0],
            operator='lap',
            kappa=kappa,
            l=l,
            weight_matrix=W
        ) # shape: (1, K)

        L[i, stencil_ids] = weights_lap[0, :]

    D_n = np.zeros((num_boundary, N))
    tree_interior = cKDTree(manifold.points[id_interior])

    for i, b_id in enumerate(id_boundary):
        _, stencil_ids = tree_interior.query(manifold.points[b_id], K-1)

        # append boundary point
        stencil_points = np.vstack((manifold.points[b_id], manifold.points[stencil_ids]))
        stencil_ids = np.append(b_id, stencil_ids)

        weights_grad = get_operator_weights(
            stencil=stencil_points,
            tangent_basis=manifold.get_local_basis(manifold.params[b_id])[0],
            operator='grad',
            kappa=kappa,
            l=l_grad,
            weight_matrix=W
        ) # shape: (n, K)

        n_vec = n_vecs[i]
        weights_grad_n = n_vec @ weights_grad

        D_n[i, stencil_ids] = weights_grad_n

    #-- SYSTEM PARTITION --#
    A = -L
    A_II = A[:, id_interior]
    A_IB = A[:, id_boundary]

    B_BI = D_n[:, id_interior] # interior points do not affect the Dirichlet part
    B_BB = D_n[:, id_boundary] + np.eye(num_boundary) 

    f_I = f_vals[id_interior]
    g_B = g_vals

    # Schur complement
    B_BB_inv = np.linalg.inv(B_BB)

    A_prime = A_II - A_IB @ B_BB_inv @ B_BI
    b_prime = f_I - A_IB @ (B_BB_inv @ g_B)

    u_num_interior = np.linalg.solve(A_prime, b_prime)
    u_num_boundary = B_BB_inv @ (g_B - B_BI @ u_num_interior)

    u_num = np.zeros(N)
    u_num[id_interior] = u_num_interior
    u_num[id_boundary] = u_num_boundary

    #-- VALIDATION --#

    fe_pointwise = np.abs(L @ u_vals - u_lap_vals[id_interior]) # shape: (num_interior,)
    ie_pointwise = np.abs(u_num - u_vals) # shape: (N,)

    fe = np.max(fe_pointwise)
    ie = np.max(ie_pointwise)

    print(f'FE: {fe:.3e} IE: {ie:.3e}')

    if args.save:
        data = {
            'params': manifold.params,
            'points': manifold.points,
            'u_num': u_num,
            'fe': fe_pointwise,
            'ie': ie_pointwise
        }

        now = datetime.datetime.now()
        formatted_time = now.strftime('%m%d_%H%M%S')

        filename = f"N{N}_l{l}_l_grad{l_grad}_{formatted_time}"
        with open(f'./results/poisson_robin_semi_torus/{filename}.pkl', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-N', type=int, default=1600
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
        '--delta', type=float, default=1e-8
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
        '--save', type=bool, default=False
    )

    args = parser.parse_args()
    main(args)