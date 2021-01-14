import gym
import torch
import json
import os, sys
import yaml
import pdb
from functools import reduce
from operator import mul

from config import get_args
import env
from policy import CategoricalMLPPolicy, NormalMLPPolicy
from policy.baseline import LinearFeatureBaseline
from metalearner import MAMLTRPO
from sampler import MultiTaskSampler
from solver.solver import Solver

def resolve_settings(config):
    # set device
    config.device = ('cuda' if (torch.cuda.is_available() and not config.use_cpu) else 'cpu')

    # directories
    config.output_folder = os.path.join('out', config.output_folder)
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)        
    config.policy_filedir = os.path.join(config.output_folder, 'policy.th')

    # # save config
    # with open(config.config_filedir, 'w') as f:
    #     json.dump(config, f, indent=2)
    return config

def set_random_seed(config):
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

def get_environment(config) :
    env = gym.make(config.env_name, **config.env_kwargs)
    env.close()
    return env

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

def main(config):

    set_random_seed(config)

    # Environment
    env = get_environment(config)

    # Policy & Baseline
    policy = get_policy_for_env(config, env, hidden_sizes=config.hidden_sizes, 
                                nonlinearity=config.nonlinearity).to(config.device)
    # policy.share_memory()
    baseline = LinearFeatureBaseline(reduce(mul, env.observation_space.shape, 1)).to(config.device)

    # Meta Learner
    metalearner = MAMLTRPO(config, policy, adapt_lr=config.adapt_lr, first_order=config.first_order,
                           device=config.device)

    # Sampler
    sampler = MultiTaskSampler(config, 
                               config.env_name,
                               env_kwargs=config.env_kwargs,
                               batch_size=config.fast_batch_size,
                               adapt_lr=config.adapt_lr,
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=config.seed)

    # Solver 
    solver = Solver(config, policy, sampler, metalearner)
    solver.train(config)


if __name__ == '__main__':

    config = get_args()
    config = resolve_settings(config)
    print(config)
    
    main(config)
