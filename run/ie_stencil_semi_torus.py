from problems import *
import numpy as np

N_vals = [1600, 3200, 6400, 12800, 25600]
seeds = np.arange(8)

results = np.zeros((len(N_vals), len(seeds), 2))

for i, N in enumerate(N_vals):
    for j, seed in enumerate(seeds):
        _, _, ie_better = poisson_robin_semi_torus(N=N, seed=seed, better_stencil=True)
        _, _, ie = poisson_robin_semi_torus(N=N, seed=seed, better_stencil=False)
        
        results[i, j, 0] = np.sqrt(np.sum(ie_better ** 2) / N)
        results[i, j, 1] = np.sqrt(np.sum(ie ** 2) / N)

np.save('./data/ie_stencil_semi_torus.npy', results)