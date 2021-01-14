import numpy as np
import torch, pdb
import pickle
import matplotlib.pyplot as plt

from tqdm import trange
from maml_rl.utils.reinforcement_learning import get_returns, plot_trajectories, plot_rewards

class Solver :
	def __init__(self, args, config, policy, sampler, metalearner) :
		self.args = args
		self.config = config
		self.policy = policy
		self.sampler = sampler
		self.metalearner = metalearner

		if args.warm_start :
			with open('./out/default/trRList.pickle', 'rb') as f :
				self.trRList, self.trRstdList = pickle.load(f)
			f.close()
			with open('./out/default/vlRList.pickle', 'rb') as f :
				self.vlRList, self.vlRstdList = pickle.load(f)
			f.close()
		else :
			self.trRList, self.trRstdList = [], []
			self.vlRList, self.vlRstdList = [], []

	def train(self, args, config):
	    num_iterations = 0
	    for batch in range(config['num-batches']):
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
	                    train_returns=get_returns(tasks, train_episodes[0]),
	                    valid_returns=get_returns(tasks, valid_episodes))

	        # Save policy
	        if args.output_folder is not None:
	            with open(args.policy_filedir, 'wb') as f:
	                torch.save(self.policy.state_dict(), f)
	        

	        # plot_trajectories(tasks, train_episodes[0], valid_episodes)
	        # Evaluate
	        if batch % 10 == 0 :
	        	self.evaluate(args, config, batch)

	def evaluate(self, args, config, iter=np.nan):
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

	    plot_trajectories(tasks, train_episodes[0], valid_episodes)
	    plot_rewards(tasks, train_episodes[0], valid_episodes)

	    train_return = get_returns(tasks, train_episodes[0])
	    valid_return = get_returns(tasks, valid_episodes)

	    self.trRList.append(train_return.mean())
	    self.vlRList.append(valid_return.mean())
	    self.trRstdList.append(train_return.mean(1).std())
	    self.vlRstdList.append(train_return.mean(1).std())

	    trUpper, trLower, vlUpper, vlLower = [], [], [], []
	    for m, s in zip(self.trRList, self.trRstdList) :
	    	trUpper.append(m+s)
	    	trLower.append(m-s)
	    for m, s in zip(self.vlRList, self.vlRstdList) :
	    	vlUpper.append(m+s)
	    	vlLower.append(m-s)

	    X = np.linspace(0, len(self.trRList), len(self.trRList))
	    plt.plot(X, self.trRList)
	    plt.plot(X, self.vlRList)
	    plt.fill_between(X, trUpper, trLower, alpha=0.1)
	    plt.fill_between(X, vlUpper, vlLower, alpha=0.1)
	    plt.legend(['train','validation'])
	    plt.xlabel('Eval No.')
	    plt.ylabel('average return')
	    plt.savefig('./out/eval/vltr_return.png')
	    plt.close()

	    with open('./out/default/trRList.pickle', 'wb') as f:
		    pickle.dump((self.trRList, self.trRstdList), f, pickle.HIGHEST_PROTOCOL)
	    f.close()
	    with open('./out/default/vlRList.pickle', 'wb') as f:
		    pickle.dump((self.vlRList, self.vlRstdList), f, pickle.HIGHEST_PROTOCOL)
	    f.close()

	    msg = "\n::: EVAL [iter {}] :::".format(iter+1)
	    msg += "\nTrain E(E(G)) {0:.2f} | V(E(G)) {1:.2f}".format(train_return.mean(), train_return.mean(1).std()**2)
	    msg += " | E(V(G)) {0:.2f}".format((train_return.std(1)**2).mean())
	    msg += "\nValid E(E(G)) {0:.2f} | V(E(G)) {1:.2f}".format(valid_return.mean(), valid_return.mean(1).std()**2)
	    msg += " | E(V(G)) {0:.2f}".format((valid_return.std(1)**2).mean())

	    print(msg)

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
