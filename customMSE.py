import torch
from torch import Tensor


class CustomMSE(torch.nn.Module):

    def __init__(self) -> None:
        super(CustomMSE, self).__init__()

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:

        result = 1 / pred.size(dim=0) * torch.square(torch.matmul(true.t(), pred.__pow__(-1)) - torch.ones(pred.size(dim=0)))

        return result
