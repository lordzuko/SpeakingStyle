import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)


class RevGrad(torch.autograd.Function):
    """
    A gradient reversal layer.
    This layer has no parameters, and simply reverses the gradient in the backward pass.
    See https://www.codetd.com/en/article/11984164, https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    , extended to 'TADAM: Task dependent adaptive metric for improved few-shot learning'
    """
    def __init__(self):
        super(FiLM, self).__init__()
        self.s_gamma = nn.Parameter(torch.ones(1,), requires_grad=True)
        self.s_beta = nn.Parameter(torch.ones(1,), requires_grad=True)

    def forward(self, x, gammas, betas):
        """
        x -- [B, T, H]
        gammas -- [B, 1, H]
        betas -- [B, 1, H]
        """
        gammas = self.s_gamma * gammas.expand_as(x)
        betas = self.s_beta * betas.expand_as(x)
        return (gammas + 1.0) * x + betas


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x
