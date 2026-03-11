import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
from problems import * 

def run_experiment(args):
    i, N, j, seed, l_index, l_grad, K_grad = args
    try:
        fe_in, fe_bd, ie = robin_semi_torus(N=N, l=5, K=30, l_grad=l_grad, K_grad=K_grad, seed=seed, lap_opt='qp', grad_opt='qp')
        # print(f'[Done] N {N:5d} | seed {seed} | l {l_grad} | IE {ie:.3e}', flush=True)
        return (i, j, l_index, fe_in, fe_bd, ie)
    except Exception as e:
        print(f"[Error] N={N}, seed={seed}, l_grad={l_grad} failed: {e}", flush=True)
        return (i, j, l_index, np.nan, np.nan, np.nan)

if __name__ == '__main__':
    N_vals = [1600, 3200, 6400, 12800, 25600, 51200]
    l_grad_vals = [3, 4, 5, 6]
    K_grad_vals = [25, 30, 35, 40]
    seeds = np.arange(12)

    results = np.zeros((len(N_vals), len(seeds), len(l_grad_vals), 3))

    tasks = []
    for i, N in enumerate(N_vals):
        for j, seed in enumerate(seeds):
            for l_index, l_grad in enumerate(l_grad_vals):
                K_grad = K_grad_vals[l_index]
                tasks.append((i, N, j, seed, l_index, l_grad, K_grad))

    tasks.sort(key=lambda x: x[1], reverse=True)

    num_cores = min(mp.cpu_count(), 50) 
    
    start_time = time.time()

    with mp.Pool(processes=num_cores) as pool:
        for res in pool.imap_unordered(run_experiment, tasks):
            i, j, l_index, fe_in, fe_bd, ie = res
            results[i, j, l_index, 0] = fe_in
            results[i, j, l_index, 1] = fe_bd
            results[i, j, l_index, 2] = ie

    end_time = time.time()
    # print(f"All computations finished in {(end_time - start_time)/60:.2f} minutes.")

    np.save('./data/semi_torus_d5.npy', results)