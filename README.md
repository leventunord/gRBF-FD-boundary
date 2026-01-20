# gRBF-FD-boundary

## Manifold Geometry

### Parameterization

An $m$-dimensional parameterized manifold $\mathcal{M}$ in $\mathbb{R}^n$ is the image of a smooth map:

$$\mathbf{x}: \Omega \in \mathbb{R}^m \to \mathbb{R}^n$$

where $\Omega$ is an open set.

Let $(\xi^1, \cdots, \xi^m)$ be intrinsic parameters, then we have
$$
\mathbf{x}(\mathbf{t}) = \begin{bmatrix} x_1(\xi^1, \cdots, \xi^m) \\ \vdots \\ x_n(\xi^1, \cdots, \xi^m) \end{bmatrix}
$$
and its Jacobian:
$$
J = \left[\frac{\partial \mathbf{x}}{\partial\xi^1}, \cdots, \frac{\partial \mathbf{x}}{\partial\xi^m}\right]
$$
where each column is the tangent basis $\mathbf{t}_i = \partial\mathbf{x}/\partial\xi^i \in \mathbb{R}^n$​.

The geometric tensor $g_{ij}$ is defined by the inner product of two tangent basis vectors $g_{ij} = \mathbf{t}_i \cdot \mathbf{t}_j$ and can be written in matrix form: $G = J^\top J$.

### Local Basis

Perform a full QR decomposition $J = QR$ (where $Q$ is a $n \times n$ matrix) will give the tangent basis and the normal basis at one certain point:

- The first $m$ columns of $Q$ form an orthonormal basis for the tangent space. (Equivalent to Gram-Schmidt Orthogonolization)
- The first $n - m$ columns form an orthonormal basis for the normal space.

### Differential Operators

Let $u: \mathcal{M}\to \mathbb{R}$ be a scalar field. The gradient $\nabla_\mathcal{M}u$ represents the direction and magnitude of the steepest ascent of $u$ along the surface. Let $\nabla_{\xi} u = [\frac{\partial u}{\partial \xi^1}, \cdots, \frac{\partial u}{\partial \xi^m}]^\top$ be the gradient in the parameter space, then:
$$
\nabla_\mathcal{M}u = JG^{-1}\nabla_{\xi} u
$$
and:
$$
\Delta_\mathcal{M} u = \frac{1}{\sqrt{g}} \sum_{i,j=1}^m \frac{\partial}{\partial \xi^i} \left( \sqrt{g} g^{ij} \frac{\partial u}{\partial \xi^j} \right)
$$
For any velocity vector $\mathbf{v} \in T_p\mathcal{M}$, the directional derivative of $u$ is $\partial_{\mathbf{v}} u = \mathbf{v} \cdot \nabla_\mathcal{M} u$.