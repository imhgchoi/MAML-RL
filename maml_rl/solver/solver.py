
from tqdm import trange
from maml_rl.utils.reinforcement_learning import get_returns, plot_trajectories

import torch, pdb

class Solver :
	def __init__(self, args, config, policy, sampler, metalearner) :
		self.args = args
		self.config = config
		self.policy = policy
		self.sampler = sampler
		self.metalearner = metalearner

	def train(self, args, config):
	    num_iterations = 0
	    for batch in trange(config['num-batches']):
	        tasks = self.sampler.sample_tasks(num_tasks=config['meta-batch-size'])
	        futures = self.sampler.sample_async(tasks,
	                                       		num_steps=config['num-steps'],
	                                       		fast_lr=config['fast-lr'],
	                                       		gamma=config['gamma'],
	                                      		gae_lambda=config['gae-lambda'],
	                                      		device=args.device)
	        logs = self.metalearner.step(*futures,
	                                	 max_kl=config['max-kl'],
	                                	 cg_iters=config['cg-iters'],
	                                	 cg_damping=config['cg-damping'],
	                                	 ls_max_steps=config['ls-max-steps'],
	                                	 ls_backtrack_ratio=config['ls-backtrack-ratio'])

	        train_episodes, valid_episodes = self.sampler.sample_wait(futures)
	        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
	        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
	        logs.update(tasks=tasks,
	                    num_iterations=num_iterations,
	                    train_returns=get_returns(train_episodes[0]),
	                    valid_returns=get_returns(valid_episodes))

	        # Save policy
	        if args.output_folder is not None:
	            with open(args.policy_filedir, 'wb') as f:
	                torch.save(self.policy.state_dict(), f)
	        

	        # plot_trajectories(tasks, train_episodes[0], valid_episodes)
	        # Evaluate
	        if batch % 10 == 0 :
	        	self.evaluate(args, config)

	def evaluate(self, args, config):
	    logs = {'tasks': []}
	    train_returns, valid_returns = [], []

	    tasks = self.sampler.sample_tasks(num_tasks=config['meta-batch-size'])	
	    train_episodes, valid_episodes = self.sampler.sample(tasks,
	                                                         num_steps=config['num-steps'],
	                                                         fast_lr=config['fast-lr'],
	                                                         gamma=config['gamma'],
	                                                         gae_lambda=config['gae-lambda'],
	                                                         device=args.device)
	    logs['tasks'].extend(tasks)
	    train_return = get_returns(train_episodes[0])
	    valid_return = get_returns(valid_episodes)

	    plot_trajectories(tasks, train_episodes[0], valid_episodes)


	    # for batch in trange(args.num_batches):
	    #     tasks = self.sampler.sample_tasks(num_tasks=args.meta_batch_size)
	    #     train_episodes, valid_episodes = self.sampler.sample(tasks,
	    #                                                     	 num_steps=config['num-steps'],
	    #                                                     	 fast_lr=config['fast-lr'],
	    #                                                     	 gamma=config['gamma'],
	    #                                                     	 gae_lambda=config['gae-lambda'],
	    #                                                     	 device=args.device)

	    #     logs['tasks'].extend(tasks)
	    #     train_returns.append(get_returns(train_episodes[0]))
	    #     valid_returns.append(get_returns(valid_episodes))

	    # logs['train_returns'] = np.concatenate(train_returns, axis=0)
	    # logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

	    # with open(args.output, 'wb') as f:
	    #     np.savez(f, **logs)
