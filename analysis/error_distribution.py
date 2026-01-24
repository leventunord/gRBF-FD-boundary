from utils import *
import pickle
import argparse
from pathlib import Path

def main(args):
    file_path = args.input
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    params = data['params']
    error_vals = data['ie']

    plot_error_distribution(params, error_vals)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input', type=Path, required=True
    )

    args = parser.parse_args()
    main(args)