from problems import *
import h5py

N_vals = [1600, 3200, 6400, 12800, 25600]
l_grad_vals = [2, 3, 4, 5]
K_grad_vals = [15, 20, 25, 30]
seeds = np.arange(4)

with h5py.File('./data/ie_semi_torus.h5', 'w') as f:
    for N in N_vals:
        grp_N = f.create_group(f"N_{N}")
        for i, l_grad in enumerate(l_grad_vals):
            K_grad = K_grad_vals[i]
            
            results = np.zeros((len(seeds), 2)) 
            
            for idx, seed in enumerate(seeds):
                _, _, ie = poisson_robin_semi_torus(N=N, l_grad=l_grad, K_grad=K_grad, seed=seed)
                results[idx, 0] = np.sqrt(np.sum(ie ** 2) / N) # l2
                results[idx, 1] = np.max(ie)                   # max
            
            dset = grp_N.create_dataset(f'l_{l_grad}', data=results)
            dset.attrs['description'] = "[col0: l2, col1: max]" 