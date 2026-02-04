import subprocess
import argparse
import random
import re
import datetime
from pathlib import Path
import numpy as np
from utils import *

def main(args):
    if args.csv is None:
        N_vals = [1600, 3200, 6400, 12800, 25600]
        l_grad_vals = [2, 3, 4, 5]
        K_grad_vals = [15, 20, 25, 30]

        trials = 4

        seeds = np.arange(trials)

        now = datetime.datetime.now()
        formatted_time = now.strftime('%m%d_%H%M%S')
        csv_file = f"./results/convergence/{formatted_time}.csv"

        with open(csv_file, "w") as f:
            f.write("N,l_grad,trial,seed,FE,IE\n")
            
            for i, N in enumerate(N_vals):
                for t in range(trials):
                    current_seed = seeds[t]
                    
                    for i in range(len(l_grad_vals)):
                        l_grad = l_grad_vals[i]
                        K_grad = K_grad_vals[i]
                        cmd = f"python poisson_robin_semi_torus.py -N {N} --l_grad {l_grad} --K_grad {K_grad} --seed {current_seed}"
                        output = subprocess.getoutput(cmd)
                        
                        fe = re.search(r"FE:\s+([0-9\.eE\-\+]+)", output).group(1)
                        ie = re.search(r"IE:\s+([0-9\.eE\-\+]+)", output).group(1)
                
                        print(f"[N={N}, l_grad={l_grad}, T={t+1}] FE: {fe} | IE: {ie}")
                        
                        f.write(f"{N},{l_grad},{t+1},{current_seed},{fe},{ie}\n")
                        f.flush()

    else:
        csv_file = args.csv

    data = np.genfromtxt(csv_file, delimiter=',', names=True)
    
    # csv processing
    unique_Ns = np.unique(data['N']).astype(int)
    unique_Ns.sort()
    
    unique_ls = np.unique(data['l_grad']).astype(int)
    unique_ls.sort()

    err_stat_list = []

    for l in unique_ls:
        mean_list = []
        std_list = []
        
        for n in unique_Ns:
            mask = (data['N'] == n) & (data['l_grad'] == l)
            err_vals = data['IE'][mask]
            
            if err_vals.size > 0:
                mean_list.append(np.mean(err_vals))
                std_list.append(np.std(err_vals, ddof=1) if err_vals.size > 1 else 0.0)
            else:
                mean_list.append(np.nan)
                std_list.append(np.nan)

        stat = {
            'mean': np.array(mean_list),
            'std': np.array(std_list),
            'plot_kwargs': {'label': f'deg = {l}'}
        }
        err_stat_list.append(stat)

    plot_convergence(unique_Ns, err_stat_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv', type=Path
    )

    args = parser.parse_args()
    main(args)