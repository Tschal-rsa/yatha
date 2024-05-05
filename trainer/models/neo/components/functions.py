from typing import Callable

import torch
from torch import Tensor


class FunctionInterface(torch.autograd.Function):
    apply: Callable[..., Tensor]


class Binarize(FunctionInterface):
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X >= 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeNFRL(FunctionInterface):
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X >= 0, torch.ones_like(X), -torch.ones_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class GradGraft(FunctionInterface):
    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class RRL_G:
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha, self.beta = alpha, beta

    def __call__(self, x: Tensor) -> Tensor:
        return 1.0 - 1.0 / (1.0 - (x * self.alpha) ** self.beta)


class RRL_P:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(self, x: Tensor) -> Tensor:
        return 1.0 / (1.0 - x) ** self.gamma


class AIWA:
    def __init__(self, bound: float, eps: float) -> None:
        self.bound, self.eps = bound, eps

    def __call__(self, x: Tensor, W: Tensor, andness: Tensor) -> Tensor:
        r = 1.0 / andness - 1.0
        W_power = W**r
        # eps_x: x = x.clamp(self.eps, 1.0)
        x = x.clamp(self.eps ** (1.0 / self.bound - 1.0), 1.0)
        # eps_matmul: (x @ W_power + self.eps)
        # eps_tail: (... + self.eps) ** (1 / r)
        return (x @ W_power / W_power.sum(dim=0)) ** (1 / r)


class AWA:
    def __init__(self, bound: float, eps: float) -> None:
        self.eps = eps ** (1.0 / bound - 1.0)

    def __call__(self, x: Tensor, W: Tensor, andness: Tensor) -> Tensor:
        r = 1.0 / andness - 1.0
        W_power = W**r
        # eps_x: x = x.clamp(self.eps, 1.0)
        x = x.clamp(self.eps, 1.0)
        # eps_matmul: (x @ W_power + self.eps)
        # eps_tail: (... + self.eps) ** (1 / r)
        return (x @ W_power / W.sum(dim=0)) ** (1.0 / r)


class AWE:
    def __init__(self, bound: float, eps: float) -> None:
        self.eps = eps ** (1.0 / bound - 1.0)

    def __call__(self, x: Tensor, W: Tensor, andness: Tensor) -> Tensor:
        r = 1.0 / andness - 1.0
        exp_W = W.exp()
        exp_Wr = exp_W**r
        x = x.clamp(self.eps, 1.0 - self.eps)
        x_exp_W = x @ exp_W
        nx_exp_W = exp_W.sum(0) - x_exp_W
        numer = x @ exp_Wr
        denom = (x_exp_W * nx_exp_W).sqrt()
        return ((numer / denom).log() + 0.5) * (1.0 / r)


def conjunction(x: Tensor, Wb: Tensor) -> Tensor:
    res = (1 - x) @ Wb
    return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))


def disjunction(x: Tensor, Wb: Tensor) -> Tensor:
    res = x @ Wb
    return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))


def mask_conjunction(x: Tensor, Wb: Tensor) -> Tensor:
    res = x.unsqueeze(-1) * Wb
    res = torch.where(res == 0, res + 2, res)
    return res.amin(1)


def mask_disjunction(x: Tensor, Wb: Tensor) -> Tensor:
    res = x.unsqueeze(-1) * Wb
    res = torch.where(res == 0, res - 2, res)
    return res.amax(1)
