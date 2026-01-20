import sympy as sp
import numpy as np

class Manifold:
    def __init__(self, xi_syms, x_syms):
        """
        Args:
            xi (list): list of parameters
            x (list)
        """
        self.m = len(xi_syms) # manifold dimension
        self.n = len(x_syms) # ambient dimension

        self.xi = sp.Matrix(xi_syms) # m-dim vector
        self.x = sp.Matrix(x_syms) # n-dim vector

    def compute(self):
        self.J = self.x.jacobian(self.xi)
        self.G = sp.simplify(self.J.T @ self.J)

        self.G_inv = sp.simplify(self.G.inv())
        self.g = sp.simplify(self.G.det())

    def get_tangent_basis(self, xi_val):
        if not hasattr(self, 'J_func'):
            self.J_func = sp.lambdify(self.xi, self.J, 'numpy')

        J_val = self.J_func(*xi_val) # shape: (n, m)

        Q, _ = np.linalg.qr(J_val, mode='complete')

        tangent_basis = Q[:, :self.m].T # Shape (m, n)
        normal_basis = Q[:, self.m:].T  # Shape (n-m, n)

        return tangent_basis, normal_basis

    def get_gradient(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        return sp.simplify(self.J @ self.G_inv @ du_dxi)

    def get_laplacian(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        V_contra = self.G_inv @ du_dxi 
        
        scaled_V = sp.sqrt(self.g) * V_contra
        
        divergence_sum = 0
        for i in range(self.m):
            divergence_sum += sp.diff(scaled_V[i], self.xi[i])
            
        laplacian = (1 / sp.sqrt(self.g)) * divergence_sum
        
        return sp.simplify(laplacian)