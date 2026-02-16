from problems import *
import h5py

num_eigs = 20

N_vals = [1600, 3200, 6400, 12800, 25600]
seeds = np.arange(4)

with h5py.File('./data/eig_semi_sphere.h5', 'w') as f:
    for N in N_vals:
        
        grp_N = f.create_group(f"N_{N}")
        
        for seed in seeds:
            evals, evecs = eig_semi_sphere(N=N, seed=seed, num_eigs=num_eigs)
            
            grp_seed = grp_N.create_group(f"seed_{seed}")
            
            grp_seed.create_dataset("evals", data=evals)
            grp_seed.create_dataset("evecs", data=evecs)