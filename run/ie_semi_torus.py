from problems import *

N_vals = [51200]
l_grad_vals = [2, 3, 4, 5]
K_grad_vals = [15, 20, 25, 30]
seeds = np.arange(5)

results = np.zeros((len(N_vals), len(seeds), len(l_grad_vals)))

for i, N in enumerate(N_vals):
    for j, seed in enumerate(seeds):
        for l, l_grad in enumerate(l_grad_vals):
            K_grad = K_grad_vals[l]
            _, _, ie = poisson_robin_semi_torus_mod(N=N, l_grad=l_grad, K_grad=K_grad, seed=seed)
            ie_l2 = np.sqrt(np.sum(ie ** 2) / N)
            results[i, j, l] = ie_l2
            print(f'N {N} seed {seed} l {l_grad} IE {ie_l2:.3e}')

np.save('./data/ie_semi_torus_pre.npy', results)
