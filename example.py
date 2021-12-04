import numpy as np
import spladtool_forward as st
import spladtool_forward.functional as F
from examples.newton import newton

f = lambda x: x - F.exp(-2.0 * F.sin(4.0 * x) * F.sin(4.0 * x))

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Newton-Raphson Method")
        parser.add_argument('-g', '--initial_guess', type=float, help="Initial guess", required=True)
        parser.add_argument('-t', '--tolerance', type=float, default=1.0e-8, help="Convergence tolerance")
        parser.add_argument('-i', '--maximum_iterations', type=int, default=100, help="Maximum iterations")
        return parser.parse_args()

    args = parse_args()
    newton(f, args.initial_guess, args.tolerance, args.maximum_iterations)