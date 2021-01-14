import torch, pdb

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from sampler import MultiTaskSampler
from metalearner.base import GradientBasedMetaLearner
from utils.torch_utils import (weighted_mean, detach_distribution, to_numpy, vector_to_parameters)
from utils.optimization import conjugate_gradient
from utils.reinforcement_learning import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, config, policy, adapt_lr=0.5, first_order=False, device='cuda'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.config = config
        self.adapt_lr = adapt_lr
        self.first_order = first_order
        # self.device = device

    # def adapt(self, train, first_order=None):
    #     if first_order is None:
    #         first_order = self.first_order
    #     # Loop over the number of steps of adaptation
    #     params = None
    #     inner_loss = reinforce_loss(self.device, self.policy, train, params=params)
    #     params = self.policy.update_params(inner_loss,
    #                                        params=params,
    #                                        step_size=self.fast_lr,
    #                                        first_order=first_order)
    #     # for futures in train_futures:
    #     #     inner_loss = reinforce_loss(self.policy, futures, params=params)
    #     #     params = self.policy.update_params(inner_loss,
    #     #                                        params=params,
    #     #                                        step_size=self.fast_lr,
    #     #                                        first_order=first_order)
    #     return params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters(), retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, train_episodes, valid_episodes, params, old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        # params = self.adapt(train, first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = pi.log_prob(valid_episodes.actions)-old_pi.log_prob(valid_episodes.actions)
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(self.config.device, ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            kls = weighted_mean(self.config.device, kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi

    def step(self, train_episodes, valid_episodes, params, max_kl=1e-3, cg_iters=10,
             cg_damping=1e-2, ls_max_steps=10, ls_backtrack_ratio=0.5):
        num_tasks = len(train_episodes)
        logs = {}

        # Compute the surrogate loss
        old_losses, old_kls, old_pis = [], [], []
        for (param, train, valid) in zip(params, train_episodes, valid_episodes) :
            ls, kl, pi = self.surrogate_loss(train, valid, param, old_pi=None) 
            old_losses.append(ls)
            old_kls.append(kl)
            old_pis.append(pi)

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            new_params = vector_to_parameters(old_params-step_size*step, self.policy.parameters())
            for par, newPar in zip(self.policy.parameters(), new_params) :
                par.data.copy_(newPar)

            losses, kls = [], []
            for param, train, valid, old_pi in zip(params, train_episodes, valid_episodes, old_pis) :
                ls, kl, _ = self.surrogate_loss(train, valid, param, old_pi=old_pi) 
                losses.append(ls)
                kls.append(kl)

            # losses, kls, _ = self._async_gather([
            #     self.surrogate_loss(train, valid, old_pi=old_pi)
            #     for (train, valid, old_pi)
            #     in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            new_params = vector_to_parameters(old_params-step_size*step, self.policy.parameters())
            for par, newPar in zip(self.policy.parameters(), new_params) :
                par.data.copy_(newPar)

        return logs
