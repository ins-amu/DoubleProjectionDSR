
import argparse
import sys
sys.path.append(".")



from dsrn import utils, plotting


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('dataset')
    parser.add_argument('datafile')
    parser.add_argument('simfile')
    parser.add_argument('outfile')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed')
    args = parser.parse_args()

    utils.seed_all(args.seed)
    plotting.plot_details_old(args.datafile, args.simfile, args.outfile, args.dataset)