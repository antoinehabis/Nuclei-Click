import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch import nn
import torch


def loss_function(output_e, outputs, imgs, lamda=1e-5, device=torch.device("cuda")):
    criterion = nn.MSELoss()
    assert (
        outputs.shape == imgs.shape
    ), f"outputs.shape : {outputs.shape} != imgs.shape : {imgs.shape}"
    loss1 = criterion(outputs, imgs)
    output_e.backward(torch.ones(output_e.size()).to(device), retain_graph=True)
    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.sqrt(torch.sum(torch.pow(imgs.grad, 2)))
    imgs.grad.data.zero_()
    loss = loss1 + lamda * loss2
    return loss
