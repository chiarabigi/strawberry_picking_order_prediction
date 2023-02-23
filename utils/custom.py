import torch
from torch import Tensor


class CustomLeakyReLU(torch.nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \min(0, x) + \text{negative\_slope} * \max(0, x) + intercept

    The domain is [0, 1]
    """
    __constants__ = ['positive_slope', 'negative_slope', 'intercept']
    minx: float
    maxx: float
    negative_slope: float
    positive_slope: float

    def __init__(self, positive_slope: float = 0.1, negative_slope: float = 0.0, minx: float = -0.01, maxx: float = 0.4) -> None:
        super(CustomLeakyReLU, self).__init__()
        self.positive_slope = positive_slope
        self.negative_slope = negative_slope
        self.minx = minx
        self.maxx = maxx

    def forward(self, input: Tensor) -> Tensor:
        return leaky_relu(input, self.positive_slope, self.negative_slope, self.minx, self.maxx)


def leaky_relu(input: Tensor, positive_slope: float = 0.1, negative_slope: float = 0.0, minx: float = -0.01, maxx: float = 0.4) -> Tensor:

    result = negative_slope * torch.minimum(torch.zeros_like(input), input) + positive_slope * torch.maximum(torch.zeros_like(input), input)  # + intercept * torch.ones_like(input)

    return result


class CustomMSE(torch.nn.Module):

    def __init__(self) -> None:
        super(CustomMSE, self).__init__()

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:

        result = 1 / pred.size(dim=0) * torch.square(torch.matmul(true.t(), pred.__pow__(-1)) - torch.ones(pred.size(dim=0)))

        return result


class mySigmoid(torch.nn.Module):
    def __init__(self, beta):
        super(mySigmoid, self).__init__()
        self.beta = beta
    def forward(self, data):
        x = data - self.beta
        output = 1 / (1 + torch.exp(-x))
        return output