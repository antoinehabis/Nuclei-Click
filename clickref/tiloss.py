import numpy as np
import torch
from torchgeometry.losses.dice import DiceLoss
from torch.nn import NLLLoss
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = input

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)



######################
# functional interface
######################

    def dice_loss(
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Sørensen-Dice Coefficient loss.

        See :class:`~torchgeometry.losses.DiceLoss` for details.
        """
        return DiceLoss()(input, target)
class TI_Loss(torch.nn.Module):
    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, -1)
        self.ce_loss_func = torch.nn.NLLLoss(reduction='none')

        if self.dim == 2 : 
            self.sum_dim_list = [1,2,3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3 :
            self.sum_dim_list = [1,2,3,4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True) # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False) # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)


    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                                        [[0,0,0],[0,1,0],[0,0,0]],
                                        [[0,1,0],[1,1,1],[0,1,0]],
                                        [[0,0,0],[0,1,0],[0,0,0]]
                                    ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))
        
        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel,axis=0), axis=0))


    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
            
            # Get Neighbourhood Information
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding='same')
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding='same')
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map


    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, c, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = torch.exp(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(),dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(),y[:,0].long()),dim=1)
        ce_tensor[:,0] = ce_tensor[:,0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()
        return ce_loss_value


if __name__ == "__main__":
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    Set the dim (2 for 2D; 3 for 3D) correctly to run relevant code.
    The samples provided enforce the following interactions:
        Enforce class 1 to be completely surrounded by class 2
        Enforce class 2 to be excluded from class 3
        Enforce class 3 to be excluded from class 4
    """

    # Parameters for creating random input
    num_classes = height = width = depth = 5

    dim = 2

    if dim == 2:
        x = torch.rand(1,num_classes,height,width)
        y = torch.randint(0, num_classes, (1,1,height,width))

        ti_loss_weight = 1e-4
        ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1,2]], exclusion=[[2,3],[3,4]])
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)


    elif dim == 3:
        x = torch.rand(1,num_classes,depth,height,width)
        y = torch.randint(0, num_classes, (1,1,depth,height,width))

        ti_loss_weight = 1e-6
        ti_loss_func = TI_Loss(dim=3, connectivity=26, inclusion=[[1,2]], exclusion=[[2,3],[3,4]], min_thick=1)
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)
        

class Finalloss(torch.nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, dice_weight = 1., ti_weight = 1e-4) -> None:
        
        super(Finalloss, self).__init__()
        
        self.ti_weight = 1e-4
        self.ti_loss_func = TI_Loss(dim=2, connectivity=8, inclusion=[[2,1]],exclusion=[])
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self.ce = NLLLoss()
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        
        return (
            self.ti_weight 
            * self.ti_loss_func(input, target) 
            + self.dice_weight 
            * self.dice_loss(torch.exp(input),torch.squeeze(target).to(torch.int64))
            + self.ce(input, torch.squeeze(target).to(torch.int64))
        )


