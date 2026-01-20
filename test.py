from src import *

theta, phi = sp.symbols('theta phi', real=True)
R = 2.0
r = 1.0

x_sym = (R + r * sp.cos(theta)) * sp.cos(phi)
y_sym = (R + r * sp.cos(theta)) * sp.sin(phi)
z_sym = r * sp.sin(theta)

torus = Manifold([theta, phi], [x_sym, y_sym, z_sym])
torus.compute()
tangent_basis, _ = torus.get_tangent_basis([0.0, 0.0])
print(tangent_basis)