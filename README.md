# gRBF-FD-boundary

## Manifold Geometry

### Parameterization

An $d$-dimensional parameterized manifold $\mathcal{M}$ in $\mathbb{R}^n$ is the image of a smooth map:

$$\mathbf{x}: \Omega \in \mathbb{R}^d \to \mathbb{R}^n$$

where $\Omega$ is an open set.

Let $(\xi^1, \cdots, \xi^d)$ be intrinsic parameters, then we have
$$
\mathbf{x}(\mathbf{t}) = \begin{bmatrix} x_1(\xi^1, \cdots, \xi^d) \\ \vdots \\ x_n(\xi^1, \cdots, \xi^d) \end{bmatrix}
$$
and its Jacobian:
$$
J = \left[\frac{\partial \mathbf{x}}{\partial\xi^1}, \cdots, \frac{\partial \mathbf{x}}{\partial\xi^d}\right]
$$
where each column is the tangent basis $\mathbf{t}_i = \partial\mathbf{x}/\partial\xi^i \in \mathbb{R}^n$​.

The geometric tensor $g_{ij}$ is defined by the inner product of two tangent basis vectors $g_{ij} = \mathbf{t}_i \cdot \mathbf{t}_j$ and can be written in matrix form: $G = J^\top J$.

### Local Basis

Perform a full QR decomposition $J = QR$ (where $Q$ is a $n \times n$ matrix) will give the tangent basis and the normal basis at one certain point:

- The first $d$ columns of $Q$ form an orthonormal basis for the tangent space. (Equivalent to Gram-Schmidt Orthogonolization)
- The first $n - d$ columns form an orthonormal basis for the normal space.

### Differential Operators

Let $u: \mathcal{M}\to \mathbb{R}$ be a scalar field. The gradient $\nabla_\mathcal{M}u$ represents the direction and magnitude of the steepest ascent of $u$ along the surface. Let $\nabla_{\xi} u = [\frac{\partial u}{\partial \xi^1}, \cdots, \frac{\partial u}{\partial \xi^d}]^\top$ be the gradient in the parameter space, then:
$$
\nabla_\mathcal{M}u = JG^{-1}\nabla_{\xi} u
$$
and:
$$
\Delta_\mathcal{M} u = \frac{1}{\sqrt{g}} \sum_{i,j=1}^d \frac{\partial}{\partial \xi^i} \left( \sqrt{g} g^{ij} \frac{\partial u}{\partial \xi^j} \right)
$$
For any velocity vector $\mathbf{v} \in T_p\mathcal{M}$, the directional derivative of $u$ is $\partial_{\mathbf{v}} u = \mathbf{v} \cdot \nabla_\mathcal{M} u$.

## Sampling

### Random Uniform Sampling in Parameter Space

## Operators

### 1. Geometric Setup: Monge Parametrization

The method operates locally on the tangent space. For a base point $x_0$ on the manifold $\mathcal{M} \subset \mathbb{R}^n$and its $K$-nearest neighbors $S_{x_0} = \{x_{0,1}, \dots, x_{0,K}\}$:

- Tangent Projection: A local coordinate system $\theta(x) = (\theta_1, \dots, \theta_d)$ is defined by projecting the relative vectors onto the orthonormal tangent basis $\{t_1, \dots, t_d\}$ at $x_0$:

  $$\theta_i(x) = t_i(x_0) \cdot (x - x_0)$$

  This forms a **Monge patch**, where the manifold is locally a graph over the tangent plane.

- Simplification of Differential Operators: A crucial property of this coordinate system is that at the base point $x_0$ (where $\theta = 0$), the metric tensor is Euclidean ($g_{ij} = \delta_{ij}$) and the Christoffel symbols vanish ($\Gamma_{ij}^k = 0$).

  Consequently, the Laplace-Beltrami operator $\Delta_\mathcal{M}$ simplifies to the standard Laplacian in the local coordinates $\theta$:
  $$
  \Delta_M f(x_0) = \sum_{i=1}^d \frac{\partial^2 f}{\partial \theta_i^2} (x_0) := \Delta_\theta f(x_0)
  $$

### 2. The Interpolant Construction

The gRBF-FD method approximates the function $f$ using a hybrid ansatz of Polyharmonic Splines (PHS) and Multivariate Polynomials (Poly), both defined over the projected coordinates $\theta$:

$$(\mathcal{G}_{\phi p} f_{x_0})(x) = \sum_{k=1}^K a_k \phi(\|\theta(x) - \theta(x_{0,k})\|) + \sum_{j=1}^m b_j p_{\alpha(j)}(\theta(x))$$

The innovation lies in how coefficients $a$ and $b$ are computed via a **two-step process**:

#### Step 1: Generalized Moving Least-Squares (GMLS) Regression

First, the polynomial coefficients $b$ are computed to capture the smooth trend of $f$. Unlike standard RBF-FD (which enforces exact interpolation), gRBF-FD minimizes a weighted least-squares error:

$$\min_{b} (f_{x_0} - Pb)^T \Lambda (f_{x_0} - Pb)$$

where $P$ is the $K \times m$ Vandermonde-type matrix with components $P_{kj} = p_{\alpha(j)}(\theta(x_{0, k}))$, and $\Lambda$ is a diagonal weight matrix.

The solution is:

$$b = (P^T \Lambda P)^{-1} P^T \Lambda f_{x_0}$$

#### Step 2: PHS Interpolation of the Residual

The residual from the GMLS step is $s_{x_0} = f_{x_0} - Pb$. The PHS coefficients $a$ are computed to interpolate this residual exactly (or via ridge regression):

$$\Phi a = s_{x_0} \implies a = \Phi^{-1} (f_{x_0} - Pb)$$

Where $\Phi$ is a $K\times K$ matrix with $\Phi_{ij} = \phi(\norm{\theta (x_{0, i}) - \theta (x_{0, j}))})$

Substituting $b$, we get the explicit form for $a$:

$$a = \Phi^{-1} [I - P(P^T \Lambda P)^{-1} P^T \Lambda] f_{x_0}$$

### 3. Derivation of the Laplacian Weights

The goal is to find weights $w_k$ such that $\Delta_M f(x_0) \approx \sum_{k=1}^K w_k f(x_{0,k})$.

We apply the local Laplacian $\Delta_\theta$ to the interpolant at $x_0$ (where $\theta(x_0)=0$):

$$\Delta_M f(x_0) \approx (\Delta_\theta \phi_{x_0}) a + (\Delta_\theta p_{x_0}) b$$

Here, the row vectors $\Delta_\theta \phi_{x_0}$ and $\Delta_\theta p_{x_0}$ are the analytic Laplacians of the basis functions evaluated at the origin:

1. PHS Laplacian ($\phi(r) = r^{2\kappa+1}$):

   $$[\Delta_\theta \phi_{x_0}]_k = (4\kappa^2 + 2d\kappa + d - 1) \|\theta(x_0) - \theta(x_{0,k})\|^{2\kappa-1}$$

2. Polynomial Laplacian:

   $$[\Delta_\theta p_{x_0}]_j = \begin{cases} 2 & \text{if } p_{\alpha(j)} \text{ is quadratic (e.g., } \theta_i^2 \text{)} \\ 0 & \text{otherwise} \end{cases}$$

### 4. Final Algebraic System

Substituting the expressions for $a$ and $b$ into the operator equation, the vector of weights $w = (w_1, \dots, w_K)$is given by:

$$w = (\Delta_\theta \phi_{x_0}) \Phi^{-1} [I - P(P^T \Lambda P)^{-1} P^T \Lambda] + (\Delta_\theta p_{x_0}) (P^T \Lambda P)^{-1} P^T \Lambda$$

**Implementation Nuances:**

- **Normalization:** Coordinates $\theta$ are scaled by the stencil diameter $D_{K,max}$. This introduces a scaling factor $1/D_{K,max}^2$ to the final weights.

- **Stability:** If $\Phi$ is singular, $\Phi^{-1}$ is replaced by a regularized pseudo-inverse $(\Phi^T \Lambda_K \Phi + \delta^2 I)^{-1} \Phi^T \Lambda_K$.

- **Weighting:** The paper recommends a specific "central spike" weight $\Lambda_{K}$ where $\lambda_{11}=1$and $\lambda_{kk}=1/K$ for neighbors, which stabilizes the Laplacian matrix.

  