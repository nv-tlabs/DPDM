import numpy as np
import argparse
from utils.util import calculate_frechet_distance


def main(args):
    stats1 = np.load(args.path1)
    stats1_mu = stats1['mu']
    stats1_sigma = stats1['sigma']
    stats2 = np.load(args.path2)
    stats2_mu = stats2['mu']
    stats2_sigma = stats2['sigma']

    fid = calculate_frechet_distance(stats1_mu, stats1_sigma, stats2_mu, stats2_sigma)
    print('FID: %.4f' % fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, required=True)
    parser.add_argument('--path2', type=str, required=True)
    args = parser.parse_args()

    main(args)

