import torch
from torch import Tensor


class CustomLeakyReLU(torch.nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \min(0, x) + \text{negative\_slope} * \max(0, x) + intercept

    The domain is [0, 1]
    """
    __constants__ = ['negative_slope', 'intercept']
    intercept: float
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, intercept: float = 0.1) -> None:
        super(CustomLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.intercept = intercept

    def forward(self, input: Tensor) -> Tensor:
        return leaky_relu(input, self.negative_slope, self.intercept)


def leaky_relu(input: Tensor, negative_slope: float = 0.01, intercept: float = 0.1) -> Tensor:

    result = min(0, input) + negative_slope * max(0, input) + intercept
    return result
