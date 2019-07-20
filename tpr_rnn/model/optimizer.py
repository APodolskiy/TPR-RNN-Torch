import torch
from torch.optim.optimizer import Optimizer


class NAdam(Optimizer):
    """
    NAdam optimizer implementation based on the Keras code.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, momentum_decay=4e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter with index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter with index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay parameter: {weight_decay}")
        if not 0.0 < momentum_decay < 1.0:
            raise ValueError(f"Invalid momentum decay parameter: {momentum_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, momentum_decay=momentum_decay)
        super(NAdam, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("NAdam doesn't support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    # initialize state
                    state['step'] = 0
                    state['exp_grad'] = torch.zeros_like(p)
                    state['exp_grad_sq'] = torch.zeros_like(p)
                    state['momentum_decay'] = 1.

                state['step'] += 1

                exp_grad = state['exp_grad']
                exp_grad_sq = state['exp_grad_sq']
                momentum_decay = state['momentum_decay']
                beta1, beta2 = group['betas']

                if group["weight_decay"] != 0:
                    grad = grad.add_(group['weight_decay'], p.data)

                momentum_corr = beta1 * (1. - 0.5*(0.96**state['step'] * momentum_decay))
                momentum_decay_current = momentum_corr * momentum_decay

                momentum_corr_next = beta1 * (1. - 0.5*(0.96**(state['step'] + 1) * momentum_decay))
                momentum_decay_next = momentum_corr_next * momentum_corr * momentum_decay
                state['momentum_decay'] = momentum_decay_next

                exp_grad.mul_(beta1).add_(1 - beta1, grad)
                exp_grad_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_grad_prime = exp_grad.div(1. - momentum_decay_current)
                exp_grad_sq_prime = exp_grad_sq.div(1. - beta2 ** state['step'])

                denom = exp_grad_sq_prime.sqrt_().add_(group['eps'])

                step_size_1 = group['lr']*(1. - momentum_corr) / (1. - momentum_decay_current)
                step_size_2 = group['lr']*momentum_corr_next

                p.data.addcdiv_(-step_size_1, grad, denom)
                p.data.addcdiv_(-step_size_2, exp_grad_prime, denom)

        return loss
