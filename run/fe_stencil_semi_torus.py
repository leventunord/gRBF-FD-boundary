from problems import *
import numpy as np

N_vals = [1600, 3200, 6400, 12800, 25600]
seeds = np.arange(4)

results = np.zeros((len(N_vals), len(seeds), 2))

for i, N in enumerate(N_vals):
    for j, seed in enumerate(seeds):
        _, fe_bd_better, _ = poisson_robin_semi_torus(N=N, seed=seed, better_stencil=True)
        _, fe_bd, _ = poisson_robin_semi_torus(N=N, seed=seed, better_stencil=False)
        
        results[i, j, 0] = np.sqrt(np.sum(fe_bd_better ** 2) / len(fe_bd_better))
        results[i, j, 1] = np.sqrt(np.sum(fe_bd ** 2) / len(fe_bd))

np.save('./data/fe_stencil_semi_torus.npy', results)