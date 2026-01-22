from src import *
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
    
    np.random.seed(0)

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
    phi_range = (0, 2*np.pi)

    manifold.sample([theta_range, phi_range], N)

    #-- MANUFACTURED SOLUTION --#

    u_sym = sp.sin(theta) * sp.cos(phi + sp.pi/4)

    u_lap_sym = manifold.get_laplacian(u_sym)
    f_sym = u_sym - u_lap_sym

    tt = manifold.xi_vals[:, 0]
    pp = manifold.xi_vals[:, 1]

    f_func = sp.lambdify((theta, phi), f_sym, 'numpy')
    f_vals = f_func(tt, pp)

    u_func = sp.lambdify((theta, phi), u_sym, 'numpy')
    u_vals = u_func(tt, pp)

    u_lap_func = sp.lambdify((theta, phi), u_lap_sym, 'numpy')
    u_lap_vals = u_lap_func(tt, pp)

    #-- SOLVE PROBLEM --#

    L = np.zeros((N, N))
    tree = cKDTree(manifold.points)

    for i in range(N):
        _, stencil_ids = tree.query(manifold.points[i], K)

        weights = get_operator_weights(
            stencil=manifold.points[stencil_ids],
            tangent_basis=manifold.get_local_basis(manifold.xi_vals[i])[0],
            kappa=kappa,
            l=l,
            weight_matrix=W
        ) # shape: (1, K)

        L[i, stencil_ids] = weights[0, :]

    lhs = np.eye(N) - L
    rhs = f_vals

    u_num = np.linalg.solve(lhs, rhs)

    #-- VALIDATION --#
    fe_pointwise = np.abs(L @ u_vals - u_lap_vals)
    ie_pointwise = np.abs(u_num - u_vals)

    fe = np.max(fe_pointwise)
    ie = np.max(ie_pointwise)

    print(f'FE: {fe:.3e} IE: {ie:.3e}')

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
        '-kappa', type=int, default=3
    )
    parser.add_argument(
        '-delta', type=float, default=1e-8
    )
    parser.add_argument(
        '-W', type=str, default='1/K'
    )

    args = parser.parse_args()
    main(args)