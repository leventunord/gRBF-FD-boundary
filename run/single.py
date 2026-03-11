import argparse
from problems import *

def main(args):
    fe_in, fe_bd, ie = robin_semi_torus(
        N=args.N, K=args.K, l=args.l, K_grad=args.K_grad, l_grad=args.l_grad,
        seed=args.seed,
    )
    print(f'FE_IN: {fe_in:.3e} FE_BD: {fe_bd:.3e} IE: {ie:.3e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', type=int, default=6400)
    parser.add_argument('-l', type=int, default=4)
    parser.add_argument('-K', type=int, default=25)
    parser.add_argument('-l_grad', type=int, default=3)
    parser.add_argument('-K_grad', type=int, default=25)

    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    main(args)