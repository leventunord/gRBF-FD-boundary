from src import *
from scipy.spatial import cKDTree

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

N = 6400
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
K = 50

for i in range(N):
    _, stencil_ids = tree.query(manifold.points[i], K)

    weights = get_operator_weights(
        stencil=manifold.points[stencil_ids],
        tangent_basis=manifold.get_local_basis(manifold.xi_vals[i])[0]
    ) # shape: (1, K)

    L[i, stencil_ids] = weights[0, :]

lhs = np.eye(N) - L
rhs = f_vals

u_num = np.linalg.solve(lhs, rhs)

#-- VALIDATION --#

forward_error = np.max(np.abs(L @ u_vals - u_lap_vals))
inverse_error = np.max(np.abs(u_num - u_vals))

print(f'FE: {forward_error:.3e} IE: {inverse_error:.3e}')