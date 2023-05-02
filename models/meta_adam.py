import math
import torch
from torch.optim.optimizer import Optimizer

class MetaAdam(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MetaAdam, self).__init__(model.params(), defaults)
        self.model = model
        # self.lr = lr
        # self.grad = grad

    def __setstate__(self, state):
        super(MetaAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, grad, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for (name, p), g in zip(self.model.named_params(self.model), grad):
                p_grad = g
                if p_grad is None:
                    continue
                # print('*' * 20, name)
                if p_grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    p_grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, p_grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, p_grad, p_grad)
                exp_avg = exp_avg * beta1 + (1 - beta1) * p_grad
                exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * p_grad * p_grad
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    # denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    denom = torch.sqrt(max_exp_avg_sq) + group['eps']
                else:
                    # denom = exp_avg_sq.sqrt().add_(group['eps'])
                    denom = torch.sqrt(exp_avg_sq) + group['eps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # bias_correction1 = to_var(torch.Tensor([bias_correction1]))
                # bias_correction2 = to_var(torch.Tensor([bias_correction2]))
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                # print(step_size)

                # exp_avg = to_var(exp_avg)
                # denom = to_var(denom)
                tmp = p - step_size * exp_avg / denom
                # print(tmp)
                # tmp.addcdiv_(-step_size, exp_avg, denom)
                self.model.set_param(self.model, name, tmp)


        return loss