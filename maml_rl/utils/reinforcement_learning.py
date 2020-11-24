import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import torch

from maml_rl.utils.torch_utils import weighted_mean, to_numpy

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(tasks, episodes):
    ret = [-np.linalg.norm(np.array(episode.observations)-np.expand_dims(tasks[taskIdx]['goal'], 0),
        axis=2).sum(0) for taskIdx, episode in enumerate(episodes)]
    return to_numpy(ret)
    # return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def reinforce_loss(policy, episodes, params=None):
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)

    return losses.mean()

def plot_trajectories(tasks, train_episodes, valid_episodes):
    for taskIdx, trEpi in enumerate(train_episodes) :
        goal = tasks[taskIdx]['goal']
        observations = trEpi.observations[:,taskIdx,:]
        for i in range(observations.shape[0]-1, 0, -1) :
            if observations[-1,0]==0 and observations[-1,1]==0 :
                observations = observations[:-1,:]
            else :
                break
        
        Xco = observations[:,0].numpy()
        Yco = observations[:,1].numpy()

        if taskIdx % 4 == 0:
            fig, axs = plt.subplots(2, 2, figsize=(16,16))
            axs[0,0].plot(Xco, Yco, linewidth=0.8, color='lightblue')
            axs[0,0].plot(goal[0],goal[1], 'ro')
            axs[0,0].legend(['trajectory','goal'])
            axs[0,0].set_xlabel('x coordinate')
            axs[0,0].set_ylabel('y coordinate')
        elif taskIdx % 4 == 1:
            axs[0,1].plot(Xco, Yco, linewidth=0.8, color='lightblue')
            axs[0,1].plot(goal[0],goal[1],'ro')
            axs[0,1].legend(['trajectory','goal'])
            axs[0,1].set_xlabel('x coordinate')
            axs[0,1].set_ylabel('y coordinate')
        elif taskIdx % 4 == 2:
            axs[1,0].plot(Xco, Yco, linewidth=0.8, color='lightblue')
            axs[1,0].plot(goal[0],goal[1],'ro')
            axs[1,0].legend(['trajectory','goal'])
            axs[1,0].set_xlabel('x coordinate')
            axs[1,0].set_ylabel('y coordinate')
        elif taskIdx % 4 == 3:
            # plt.subplot(4,2,2)
            axs[1,1].plot(Xco, Yco, linewidth=0.8, color='lightblue')
            axs[1,1].plot(goal[0],goal[1],'ro')
            axs[1,1].legend(['trajectory','goal'])
            axs[1,1].set_xlabel('x coordinate')
            axs[1,1].set_ylabel('y coordinate')
            plt.savefig('out/eval/tr_{}.png'.format(str(taskIdx//4)))
            plt.close()
    plt.close()

    for taskIdx, vlEpi in enumerate(valid_episodes) :
        goal = tasks[taskIdx]['goal']
        observations = vlEpi.observations[:,taskIdx,:]
        for i in range(observations.shape[0]-1, 0, -1) :
            if observations[-1,0]==0 and observations[-1,1]==0 :
                observations = observations[:-1,:]
            else :
                break

        Xco = observations[:,0].numpy()
        Yco = observations[:,1].numpy()

        if taskIdx % 4 == 0:
            # plt.figure(figsize=(16.0, 16.0))
            fig, axs = plt.subplots(2, 2, figsize=(16,16))
            # plt.subplot(4,1,1)
            axs[0,0].plot(Xco, Yco, color='b', linewidth=0.8)
            axs[0,0].plot(goal[0],goal[1],'ro')
            axs[0,0].legend(['trajectory','goal'])
            axs[0,0].set_xlabel('x coordinate')
            axs[0,0].set_ylabel('y coordinate')
        elif taskIdx % 4 == 1:
            # plt.subplot(4,2,1)
            axs[0,1].plot(Xco, Yco, color='b', linewidth=0.8)
            axs[0,1].plot(goal[0],goal[1],'ro')
            axs[0,1].legend(['trajectory','goal'])
            axs[0,1].set_xlabel('x coordinate')
            axs[0,1].set_ylabel('y coordinate')
        elif taskIdx % 4 == 2:
            # plt.subplot(4,1,2)
            axs[1,0].plot(Xco, Yco, color='b', linewidth=0.8)
            axs[1,0].plot(goal[0],goal[1],'ro')
            axs[1,0].legend(['trajectory','goal'])
            axs[1,0].set_xlabel('x coordinate')
            axs[1,0].set_ylabel('y coordinate')
        elif taskIdx % 4 == 3:
            # plt.subplot(4,2,2)
            axs[1,1].plot(Xco, Yco, color='b', linewidth=0.8)
            axs[1,1].plot(goal[0],goal[1],'ro')
            axs[1,1].legend(['trajectory','goal'])
            axs[1,1].set_xlabel('x coordinate')
            axs[1,1].set_ylabel('y coordinate')
            plt.savefig('out/eval/vl_{}.png'.format(str(taskIdx//4)))
            plt.close()
    plt.close()

def plot_rewards(tasks, train_episodes, valid_episodes):

    for taskIdx, (trEpi, vlEpi) in enumerate(zip(train_episodes, valid_episodes)) :
        goal = tasks[taskIdx]['goal']
        trS = trEpi.observations[:,taskIdx,:]
        vlS = vlEpi.observations[:,taskIdx,:]
        for i in range(trS.shape[0]-1, 0, -1) :
            if trS[i,0]==0 and trS[i,1]==0 :
                trS[i,:] = torch.Tensor(goal)
            else :
                trS
                break
        for i in range(vlS.shape[0]-1, 0, -1) :
            if vlS[i,0]==0 and vlS[i,1]==0 :
                vlS[i,:] = torch.Tensor(goal)
            else :
                break

        trR = -np.linalg.norm(trS - goal, axis=1)
        vlR = -np.linalg.norm(vlS - goal, axis=1)

        if taskIdx % 4 == 0:
            fig, axs = plt.subplots(2, 2, figsize=(16,16))
            axs[0,0].plot(trR)
            axs[0,0].plot(vlR)
            axs[0,0].legend(['train', 'validation'])
            axs[0,0].set_xlabel('Steps')
            axs[0,0].set_ylabel('Rewards')
        elif taskIdx % 4 == 1:
            axs[0,1].plot(trR)
            axs[0,1].plot(vlR)
            axs[0,1].legend(['train', 'validation'])
            axs[0,1].set_xlabel('Steps')
            axs[0,1].set_ylabel('Rewards')
        elif taskIdx % 4 == 2:
            axs[1,0].plot(trR)
            axs[1,0].plot(vlR)
            axs[1,0].legend(['train', 'validation'])
            axs[1,0].set_xlabel('Steps')
            axs[1,0].set_ylabel('Rewards')
        elif taskIdx % 4 == 3:
            axs[1,1].plot(trR)
            axs[1,1].plot(vlR)
            axs[1,1].legend(['train', 'validation'])
            axs[1,1].set_xlabel('Steps')
            axs[1,1].set_ylabel('Rewards')
            plt.savefig('out/eval/rewards_{}.png'.format(str(taskIdx//4)))
            plt.close()
    plt.close()
