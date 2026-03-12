import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
import numpy as np
from problems.interface_sphere import interface_sphere

def run_experiment(args):
    i, N, j, seed, l_index, l_grad, K_grad = args
    try:
        ie = interface_sphere(N=N, l=4, K=30, l_grad=l_grad, K_grad=K_grad, seed=seed)
        print(f'[Done] N {N:5d} | seed {seed} | l {l_grad} | IE {ie:.3e}', flush=True)
        return (i, j, l_index, ie)
    except Exception as e:
        print(f"[Error] N={N}, seed={seed}, l_grad={l_grad} failed: {e}", flush=True)
        return (i, j, l_index, np.nan)

if __name__ == '__main__':
    N_vals = [1600, 3200, 6400, 12800, 25600, 51200]
    l_grad_vals = [4]
    K_grad_vals = [35]
    seeds = np.arange(12)

    results = np.zeros((len(N_vals), len(seeds), len(l_grad_vals), 1))

    tasks = []
    for i, N in enumerate(N_vals):
        for j, seed in enumerate(seeds):
            for l_index, l_grad in enumerate(l_grad_vals):
                K_grad = K_grad_vals[l_index]
                tasks.append((i, N, j, seed, l_index, l_grad, K_grad))

    tasks.sort(key=lambda x: x[1], reverse=True)

    num_cores = min(mp.cpu_count(), 100) 
    
    start_time = time.time()

    with mp.Pool(processes=num_cores) as pool:
        for res in pool.imap_unordered(run_experiment, tasks):
            i, j, l_index, ie = res
            results[i, j, l_index, 0] = ie

    end_time = time.time()
    print(f"All computations finished in {(end_time - start_time)/60:.2f} minutes.")

    np.save('./data/interface_sphere_d4.npy', results)