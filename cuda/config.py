import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MAML RL')

    parser.add_argument('--config', type=str, default=None, help='path to the configuration file.')


    # Environment
    env = parser.add_argument_group('Environment')
    env.add_argument('--env-name', type=str, default='2DNavigation-v0', 
                     choices=['2DNavigation-v0','Sparse2DNavigation-v0','Noisy2DNavigation-v0'])
    env.add_argument('--env-kwargs', type=dict, default={'low': -0.5, 'high': 0.5})

    # Agent
    agent = parser.add_argument_group('Agent')
    agent.add_argument('--gamma', type=float, default=0.99)
    agent.add_argument('--gae-lambda', type=float, default=0.99)
    agent.add_argument('--first-order', action='store_true')
    agent.add_argument('--hidden-sizes', type=list, default=[64,64])
    agent.add_argument('--nonlinearity', type=str, default='tanh')

    # Optimization
    opt = parser.add_argument_group('Optimization')
    opt.add_argument('--fast-batch-size', type=int, default=20)
    opt.add_argument('--num-steps', type=int, default=1)
    opt.add_argument('--adapt-lr', type=float, default=0.1)
    opt.add_argument('--max-iterations', type=int, default=500)
    opt.add_argument('--meta-batch-size', type=int, default=20)
    opt.add_argument('--max-kl', type=float, default=0.01)
    opt.add_argument('--cg-iters', type=int, default=20)
    opt.add_argument('--cg-damping', type=float, default=1e-05)
    opt.add_argument('--ls-max-steps', type=int, default=15)
    opt.add_argument('--ls-backtrack-ratio', type=float, default=0.8)

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, default='default', help='name of the output folder')
    misc.add_argument('--policy_filedir', type=str, default='out/default/policy.th', help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None, help='random seed')
    misc.add_argument('--use-cpu', action='store_true')
    misc.add_argument('--warm-start', action='store_true', default=False)

    return parser.parse_args()