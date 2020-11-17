import gym
import torch
import json
import os
import yaml
import pdb
from functools import reduce
from operator import mul

from config import get_args
import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.solver.solver import Solver

def resolve_settings(args):
    # set device
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')

    # import yaml config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # directories
    args.output_folder = os.path.join('out', args.output_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)        
    args.policy_filedir = os.path.join(args.output_folder, 'policy.th')
    args.config_filedir = os.path.join(args.output_folder, 'config.json')

    # save config
    with open(args.config_filedir, 'w') as f:
        config.update(vars(args))
        json.dump(config, f, indent=2)

    return args, config

def set_random_seed(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def get_policy_for_env(args, env, hidden_sizes=(100, 100), nonlinearity='relu'):
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = reduce(mul, env.observation_space.shape, 1)
    nonlinearity = getattr(torch, nonlinearity)

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size, output_size,
                                 hidden_sizes=tuple(hidden_sizes),
                                 nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size, output_size,
                                      hidden_sizes=tuple(hidden_sizes),
                                      nonlinearity=nonlinearity)
    if args.warm_start :
        with open(args.policy_filedir, 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device(args.device))
            policy.load_state_dict(state_dict)

    return policy

def main(args, config):

    set_random_seed(args)

    # Environment
    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy & Baseline
    policy = get_policy_for_env(args, env, hidden_sizes=config['hidden-sizes'], 
                                           nonlinearity=config['nonlinearity'])
    policy.share_memory()  # this is done to share memory across processes for multiprocessing
    
    baseline = LinearFeatureBaseline(reduce(mul, env.observation_space.shape, 1))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    # Meta Model
    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    # Solver 
    solver = Solver(args, config, policy, sampler, metalearner)
    solver.train(args, config)


if __name__ == '__main__':

    args = get_args()
    args, config = resolve_settings(args)
    print(args,'\n',config,'\n')
    
    main(args, config)
