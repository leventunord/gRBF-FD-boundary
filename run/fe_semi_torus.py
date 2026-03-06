from problems import *

N_vals = [1600, 3200, 6400, 12800, 25600]
l_grad_vals = [2, 3, 4, 5]
K_grad_vals = [15, 20, 25, 30]
seeds = np.arange(4)

results = np.zeros((len(N_vals), len(seeds), len(l_grad_vals), 2))

for i, N in enumerate(N_vals):
    for j, seed in enumerate(seeds):
        for l, l_grad in enumerate(l_grad_vals):
            K_grad = K_grad_vals[l]
            fe_i, fe_b, _ = poisson_robin_semi_torus(N=N, l=l_grad, K=K_grad, l_grad=l_grad, K_grad=K_grad, seed=seed)
            fe_i_l2 = np.sqrt(np.sum(fe_i ** 2) / len(fe_i))
            fe_b_l2 = np.sqrt(np.sum(fe_b ** 2) / len(fe_b))

            results[i, j, l, 0] = fe_i_l2
            results[i, j, l, 1] = fe_b_l2

            print(f'N {N} seed {seed} l {l_grad} FE_I {fe_i_l2:.3e} FE_B {fe_b_l2:.3e}')

np.save('./data/fe_semi_torus.npy', results)