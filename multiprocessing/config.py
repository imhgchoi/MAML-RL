import argparse
import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser(description='MAML RL')

    parser.add_argument('--config', type=str, default=None, help='path to the configuration file.')


    # Environment
    env = parser.add_argument_group('Environment')
    env.add_argument('--env-name', type=str, default='StockMarket-v0', 
        choices=['Sparse2DNavigation-v0','Noisy2DNavigation-v0','StockMarket-v0'])

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, default='default', 
                       help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None, help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count()-1,
                       help='number of workers for trajectory sampling (default: {0})'.format(mp.cpu_count()-1))
    misc.add_argument('--use-cuda', action='store_true',
                       help='use cuda (default: false, use cpu). WARNING: Full support for cuda '
                            'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--warm-start', action='store_true', default=False)

    return parser.parse_args()
