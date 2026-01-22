import sympy as sp
import numpy as np

class Manifold:
    def __init__(self, xi_syms, x_syms):
        """
        Args:
            xi (list): list of parameters
            x (list)
        """
        self.d = len(xi_syms) # manifold dimension
        self.n = len(x_syms) # ambient dimension

        self.xi = sp.Matrix(xi_syms) # shape: (d,)
        self.x = sp.Matrix(x_syms) # shape: (n, )

    def compute(self):
        self.J = self.x.jacobian(self.xi)
        self.G = sp.simplify(self.J.T @ self.J)

        self.G_inv = sp.simplify(self.G.inv())
        self.g = sp.simplify(self.G.det())

    def get_local_basis(self, xi_val):
        if not hasattr(self, 'J_func'):
            self.J_func = sp.lambdify(self.xi, self.J, 'numpy')

        J_val = self.J_func(*xi_val) # shape: (n, d)

        Q, _ = np.linalg.qr(J_val, mode='complete')

        tangent_basis = Q[:, :self.d].T # Shape (d, n)
        normal_basis = Q[:, self.d:].T  # Shape (n-d, n)

        return tangent_basis, normal_basis

    def get_gradient(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        return sp.simplify(self.J @ self.G_inv @ du_dxi)

    def get_laplacian(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        V_contra = self.G_inv @ du_dxi 

        scaled_V = sp.sqrt(self.g) * V_contra
        
        divergence_sum = 0
        for i in range(self.d):
            divergence_sum += sp.diff(scaled_V[i], self.xi[i])
            
        laplacian = (1 / sp.sqrt(self.g)) * divergence_sum
        
        return sp.simplify(laplacian)

    def sample(self, xi_ranges, num_points):
        x_syms = [self.x[i] for i in range(self.n)]
        self.x_func = sp.lambdify(self.xi, x_syms, 'numpy')

        xi_vals = np.zeros((num_points, self.d))

        for i, (t_min, t_max) in enumerate(xi_ranges):
            xi_vals[:, i] = np.random.uniform(t_min, t_max, size=num_points)

        args = [xi_vals[:, i] for i in range(self.d)]
        raw_coords = self.x_func(*args)
        coords_broadcasted = [np.broadcast_to(c, (num_points,)) for c in raw_coords]
        
        self.xi_vals = xi_vals # shape: (num_points, d)
        self.points = np.column_stack(coords_broadcasted) # shape: (num_points, n)